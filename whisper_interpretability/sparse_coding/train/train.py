from functools import partial
from math import inf
from time import perf_counter

import numpy as np
import torch
from absl import app, flags, logging
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import RAdam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data.distributed import DistributedSampler

from base_train import train_init
from global_utils import (
    Metadata,
    device,
    dist_logging,
    dump_checkpoint_on_kill,
    load_checkpoint,
    prepare_tb_logging,
    save_checkpoint,
    set_seeds,
    snapshot_memory_usage,
)
from sparse_coding.train.autoencoder import AutoEncoder
from sparse_coding.train.dataset import ActivationDataset, collate_fn, TokenEmbeddingDataset

torch.backends.cudnn.benchmark = True
logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS  # See base_train.py for others

# Optimization
flags.DEFINE_boolean(
    "lr_schedule",
    True,
    "state if an learning rate scheduler is required during training",
)
flags.DEFINE_float("scheduler_decay", 1.0 / 3, "proportion of training steps over which lr decays")
flags.DEFINE_float("clip_thresh", 1.0, "value to clip gradients to")

# Regularization
flags.DEFINE_float("weight_decay", 0.0, "weight decay to use for training")

# Sparse Coding hyperparams
flags.DEFINE_integer("n_dict_components", None, "number of components in Sparse Dictionary")
flags.DEFINE_float("recon_alpha", 1e-5, "multiplier for the l1 'sparsity' component of the loss")
flags.mark_flag_as_required("n_dict_components")


def validate(
    FLAGS,
    model,
    recon_loss_fn,
    dataloader_args,
):
    model.eval()
    losses_recon = []
    losses_l1 = []

    val_dataset = ActivationDataset(FLAGS.val_data)
    val_loader = iter(torch.utils.data.DataLoader(val_dataset, shuffle=True, **dataloader_args))
    for activations in val_loader:
        with torch.no_grad() and autocast():
            activations = activations.to(device)
            pred, c = model(activations)
            losses_recon.append(recon_loss_fn(pred, activations).item())
            losses_l1.append(torch.norm(c, 1, dim=2).mean().item())

    model.train()
    return np.array(losses_recon).mean(), np.array(losses_l1).mean()


def mse_loss(input, target, ignored_index, reduction):
    # mse_loss with ignored_index
    mask = target == ignored_index
    out = (input[~mask] - target[~mask]) ** 2
    if reduction == "mean":
        return out.mean()
    elif reduction == "None":
        return out


def train(FLAGS, global_rank=0):
    torch.set_num_threads(1)
    set_seeds(FLAGS.seed)

    # train_dataset = ActivationDataset(dbl_path=FLAGS.train_data)
    train_dataset = TokenEmbeddingDataset()
    feat_dim = next(iter(train_dataset)).shape[-1]
    model = AutoEncoder(feat_dim, FLAGS.n_dict_components).to(device)
    if FLAGS.n_devices > 1:
        dist_model = DDP(model, device_ids=[global_rank])
        model = dist_model.module
    else:
        dist_model = model

    # setup logging
    meta = Metadata(history_size=FLAGS.log_tb_every)
    memory_usage = snapshot_memory_usage()
    tb_logger = prepare_tb_logging(FLAGS.expdir)
    if memory_usage is not None:
        dist_logging(f"pretrain_mem {memory_usage['allocated']:.5f}GB", rank=global_rank)

    if FLAGS.model_out is None:
        FLAGS.model_out = FLAGS.expdir + "/model"

    dist_logging(
        "Model: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1.0e6),
        rank=global_rank,
    )

    optimizer = RAdam(
        dist_model.parameters(), eps=1e-5, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=FLAGS.steps, eta_min=0)

    dataloader_kwargs = {
        "batch_size": FLAGS.batch_size,
        "pin_memory": False,
        "drop_last": True,
        "num_workers": FLAGS.dl_max_workers,
        "collate_fn": collate_fn,
    }
    if FLAGS.n_devices > 1:
        # Must shuffle to get even split of different languages
        train_sampler = DistributedSampler(
            train_dataset, rank=global_rank, drop_last=True, shuffle=True
        )
        train_loader = iter(
            torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **dataloader_kwargs)
        )
    else:
        train_loader = iter(
            torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
        )

    # Object that contains the main state of the train loop
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "step": 0,
        "best_val_loss": inf,
        "total_speech_seconds_seen": 0,
        "total_non_speech_seconds_seen": 0,
        "total_time_ms": 0,
    }

    meta["effective_batch_size"] = FLAGS.batch_size
    meta["model_params"] = sum(x.numel() for x in dist_model.parameters())

    if FLAGS.checkpoint:
        # loading state_dicts in-place
        print(f"Checkpoint: {FLAGS.checkpoint}")
        load_checkpoint(state, FLAGS.checkpoint, device=device)

    if global_rank == 0:
        dump_checkpoint_on_kill(global_rank=global_rank)

    # Set seeds so each rank gets different data
    set_seeds(FLAGS.seed + state["step"] + global_rank)

    recon_loss_fn = partial(mse_loss, ignored_index=-1, reduction="mean")
    while True:
        forward_time = 0
        backward_time = 0
        losses_recon = []
        losses_l1 = []
        for _ in range(FLAGS.grad_acc_steps):
            try:
                activations = next(train_loader).to(device)
            except StopIteration:
                train_loader = iter(
                    torch.utils.data.DataLoader(train_dataset, shuffle=True, **dataloader_kwargs)
                )
                activations = next(train_loader).to(device)
            # Forward pass
            with autocast():
                start_time = perf_counter()
                pred, c = dist_model(activations)  # bsz, seq_len, n_classes
                forward_time += perf_counter() - start_time
                loss_recon = FLAGS.recon_alpha * recon_loss_fn(pred, activations)
                loss_l1 = torch.norm(c, 1, dim=2).mean()
                loss = loss_recon + loss_l1
                losses_recon.append(loss_recon.item())
                losses_l1.append(loss_l1.item())
                meta.snapshot_memory_usage("mem_fwd")

                # Backward pass
                start_time = perf_counter()
                loss.backward()
                backward_time += perf_counter() - start_time

        torch.nn.utils.clip_grad_norm_(dist_model.parameters(), FLAGS.clip_thresh)
        optimizer.step()
        scheduler.step()
        dist_model.zero_grad()
        state["step"] += 1
        meta["loss_recon"] = sum(losses_recon) / FLAGS.grad_acc_steps
        meta["loss_l1"] = sum(losses_l1) / FLAGS.grad_acc_steps
        meta["time_backward"] = backward_time

        meta.snapshot_memory_usage("mem_bwd")

        if state["step"] % FLAGS.log_every == 0 and global_rank == 0:
            dist_logging(f"step {state['step']}, loss {loss.item():.3f}", rank=global_rank)

            # log training losses
            if state["step"] % FLAGS.log_tb_every == 0 and global_rank == 0:
                tb_logger.add_scalar("train/loss", loss, state["step"])
                tb_logger.add_scalar("train/loss_recon", meta["loss_recon"], state["step"])
                tb_logger.add_scalar("train/loss_l1", meta["loss_l1"], state["step"])
                tb_logger.add_scalar("train/lr", scheduler.get_last_lr()[0], state["step"])
                # log timings but ignoring first step
                if state["step"] > 1:
                    meta.log_tb_timings(tb_logger, state["step"])

        # save out model periodically
        if state["step"] % FLAGS.save_every == 0:
            save_checkpoint(state, FLAGS.checkpoint_out + ".step" + str(state["step"]))

        # validate periodically
        if state["step"] % FLAGS.val_every == 0 and global_rank == 0:
            dist_logging("Starting to validate", rank=global_rank)
            val_loss_recon, val_loss_l1 = validate(
                FLAGS,
                model,
                recon_loss_fn,
                dataloader_kwargs,
            )
            dist_logging(
                f"{state['step']} validation, loss_recon={val_loss_recon:.3f}",
                rank=global_rank,
            )
            # log validation losses
            tb_logger.add_scalar("val/loss_recon", val_loss_recon, state["step"])
            tb_logger.add_scalar("val/loss_l1", val_loss_l1, state["step"])
            if val_loss_recon.item() < state["best_val_loss"]:
                dist_logging("Saving new best validation", rank=global_rank)
                state["best_val_loss"] = val_loss_recon.item()
                save_checkpoint(state, FLAGS.checkpoint_out + ".bestval")

                # Save PyTorch model for PR area calculation
                pytorch_model_path = FLAGS.model_out[:-3] + ".bestval"
                torch.save(model, pytorch_model_path)

        if FLAGS.steps != -1 and state["step"] >= FLAGS.steps:
            break

    save_checkpoint(state, FLAGS.checkpoint_out + ".step" + str(state["step"]))


if __name__ == "__main__":
    app.run(partial(train_init, train))
