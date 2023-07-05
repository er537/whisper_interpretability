from functools import partial
from math import inf
from time import perf_counter

import numpy as np
import torch
from absl import app, flags, logging
from torch.cuda.amp import autocast
from torch.optim import RAdam
from torch.nn.functional import interpolate
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from probes.train.dataset import MultiClassDataset, collate_fn
from probes.utils.activation_caches import (
    WhisperActivationCache,
)
from probes.train.probe_model import Probe
from probes.train.base_train import train_init
from utils import (
    load_checkpoint,
    save_checkpoint,
    save_model,
    Metadata,
    device,
    dump_checkpoint_on_kill,
    prepare_tb_logging,
    set_seeds,
    snapshot_memory_usage,
    dist_logging,
)

torch.backends.cudnn.benchmark = True
logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS  # See base_train.py for others

# Optimization
flags.DEFINE_boolean(
    "lr_schedule",
    True,
    "state if an learning rate scheduler is required during training",
)
flags.DEFINE_float(
    "scheduler_decay", 1.0 / 3, "proportion of training steps over which lr decays"
)
flags.DEFINE_float("clip_thresh", 1.0, "value to clip gradients to")

# Regularization
flags.DEFINE_float("weight_decay", 0.0, "weight decay to use for training")

flags.DEFINE_integer(
    "num_audio_clips",
    100,
    "used to determine the number of training steps if not specified",
)
flags.DEFINE_string("probe_layer", None, "layer of base model to train probe on")
flags.DEFINE_string("whisper_model", "tiny", "which whisper model to use")
flags.DEFINE_integer("val_samples", 100, "Number of samples to validate on")
flags.DEFINE_integer("num_train_samples", 2297662, "Number of samples in train dataset")

flags.mark_flag_as_required("probe_layer")


def get_class_freq(labels):
    """Count number of non-speech and speech labels in a tensor"""
    return (labels.detach().cpu() == 0).sum(), (labels.detach().cpu() == 1).sum()


def resample_labels(labels, pred):
    """
    mfccs are downsampled inside the model so we need to downsample the labels by the same fraction
    """
    return interpolate(labels.unsqueeze(0).float(), size=pred.shape[1]).squeeze(0)


def validate(
    val_data, val_samples, probe_layer, model, whisper_model, loss_fn, dataloader_args
):
    model.eval()
    losses = []
    frames_seen = 0
    accs = []

    val_dataset = MultiClassDataset(sql_path=val_data, num_entries=val_samples)
    val_loader = iter(
        torch.utils.data.DataLoader(val_dataset, shuffle=True, **dataloader_args)
    )
    # reset random seed for deterministic validation
    for data, labels in val_loader:
        data, labels = data.to(device), labels.to(device)

        with torch.no_grad() and autocast():
            whisper_model.forward(data)
            activations = whisper_model.activations[f"{probe_layer}.output"].to(device)
            whisper_model.reset_state()
            activations = activations.mean(dim=1)
            pred = model(activations)
            # labels = resample_labels(labels, pred)
            # pred = torch.permute(pred, (0, 2, 1))  # bsz, n_classes, seq_len
            labels = labels[:, 0].long()
            losses.append(loss_fn(pred, labels).item())
            num_speech, num_non_speech = get_class_freq(labels)
            frames_seen += num_non_speech + num_speech

            # Calculate accuracy as well
            acc = (torch.argmax(pred, dim=1) == labels).sum() / len(labels)
            accs.append(acc.item())

    model.train()
    return np.array(losses).mean(), np.array(accs).mean(), frames_seen


def get_probe_feat_dim(probe_layer, model_name):
    dataset = MultiClassDataset(num_entries=1)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    mels, *_ = next(iter(dataloader))
    whisper_model = WhisperActivationCache(
        activations_to_cache=probe_layer, model_name=model_name
    )
    whisper_model.forward(mels.to(device))
    activations_shape = whisper_model.activations[
        f"{probe_layer}.output"
    ].shape  # (bsz, seq_len, d_model)
    return activations_shape[-1]


def train(FLAGS, global_rank=0):
    torch.set_num_threads(1)
    set_seeds(FLAGS.seed)

    fd = get_probe_feat_dim(FLAGS.probe_layer, FLAGS.whisper_model)
    model = Probe(feat_dim=fd).to(device)
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
        dist_logging(
            f"pretrain_mem {memory_usage['allocated']:.5f}GB", rank=global_rank
        )

    if FLAGS.model_out is None:
        FLAGS.model_out = FLAGS.expdir + "/model"

    dist_logging(
        "Model: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1.0e6),
        rank=global_rank,
    )

    optimizer = RAdam(
        dist_model.parameters(), eps=1e-5, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
    )
    if FLAGS.steps == -1:
        FLAGS.steps = FLAGS.num_audio_clips // FLAGS.batch_size
    scheduler = CosineAnnealingLR(optimizer, T_max=FLAGS.steps, eta_min=0)

    train_dataset = MultiClassDataset(
        sql_path=FLAGS.train_data, num_entries=FLAGS.num_train_samples
    )
    dataloader_kwargs = {
        "batch_size": FLAGS.batch_size,
        "pin_memory": False,
        "drop_last": True,
        "num_workers": FLAGS.dl_max_workers,
        "collate_fn": collate_fn,
    }
    if FLAGS.n_devices > 1:
        train_sampler = DistributedSampler(
            train_dataset, rank=global_rank, drop_last=True, shuffle=True
        )
        train_loader = iter(
            torch.utils.data.DataLoader(
                train_dataset, sampler=train_sampler, **dataloader_kwargs
            )
        )
    else:
        train_loader = iter(
            torch.utils.data.DataLoader(
                train_dataset, shuffle=True, **dataloader_kwargs
            )
        )

    whisper_model = WhisperActivationCache(
        activations_to_cache=FLAGS.probe_layer, model_name=FLAGS.whisper_model
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

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=-1)
    while True:
        forward_time = 0
        backward_time = 0
        losses = []
        for _ in range(FLAGS.grad_acc_steps):
            data, labels = next(train_loader)
            data, labels = data.to(device), labels.to(device)
            num_speech, num_non_speech = get_class_freq(labels)
            state["total_speech_seconds_seen"] += num_speech / 100
            state["total_non_speech_seconds_seen"] += num_non_speech / 100

            # Forward pass
            with autocast():
                start_time = perf_counter()
                whisper_model.forward(data)
                activations = whisper_model.activations[
                    f"{FLAGS.probe_layer}.output"
                ].to(device)
                activations = activations.mean(dim=1)
                whisper_model.reset_state()
                pred = dist_model(activations)
                # labels = resample_labels(labels, pred)
                forward_time += perf_counter() - start_time
                # pred = torch.permute(pred, (0, 2, 1))  # bsz, n_classes, seq_len
                loss = loss_fn(pred, labels[:, 0].long())
                losses.append(loss.item())
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
        meta["loss"] = sum(losses) / FLAGS.grad_acc_steps
        meta["time_backward"] = backward_time

        meta.snapshot_memory_usage("mem_bwd")

        if state["step"] % FLAGS.log_every == 0 and global_rank == 0:
            dist_logging(
                f"step {state['step']}, loss {loss.item():.3f}", rank=global_rank
            )

            # log training losses
            if state["step"] % FLAGS.log_tb_every == 0 and global_rank == 0:
                tb_logger.add_scalar("train/loss", loss, state["step"])
                tb_logger.add_scalar(
                    "train/lr", scheduler.get_last_lr()[0], state["step"]
                )
                data_seen_hours = (
                    (
                        state["total_speech_seconds_seen"]
                        + state["total_non_speech_seconds_seen"]
                    )
                    / 60.0
                    / 60.0
                )
                speech_frac = (
                    state["total_speech_seconds_seen"]
                    / (
                        state["total_non_speech_seconds_seen"]
                        + state["total_speech_seconds_seen"]
                    )
                    * 100
                )
                tb_logger.add_scalar(
                    "train/data_seen_(hrs)", data_seen_hours, state["step"]
                )
                tb_logger.add_scalar(
                    "train/speech_percentage", speech_frac, state["step"]
                )
                # log timings but ignoring first step
                if state["step"] > 1:
                    meta.log_tb_timings(tb_logger, state["step"])

        # save out model periodically
        if state["step"] % FLAGS.save_every == 0:
            save_model(model, FLAGS.model_out + ".step" + str(state["step"]))
            save_checkpoint(state, FLAGS.checkpoint_out + ".step" + str(state["step"]))

        # validate periodically
        if state["step"] % FLAGS.val_every == 0 and global_rank == 0:
            dist_logging("Starting to validate", rank=global_rank)
            val_loss, val_acc, val_frames_seen = validate(
                FLAGS.val_data,
                FLAGS.val_samples,
                FLAGS.probe_layer,
                model,
                whisper_model,
                loss_fn,
                dataloader_kwargs,
            )
            dist_logging(
                f"{state['step']} validation, loss={val_loss:.3f}, "
                f"acc={val_acc:.3f}, "
                f"{val_frames_seen:,} frames validated",
                rank=global_rank,
            )
            # log validation losses
            tb_logger.add_scalar("val/loss", val_loss, state["step"])
            tb_logger.add_scalar("val/acc", val_acc, state["step"])
            if val_loss.item() < state["best_val_loss"]:
                dist_logging("Saving new best validation", rank=global_rank)
                save_model(model, FLAGS.model_out + ".bestval")
                state["best_val_loss"] = val_loss.item()
                save_checkpoint(state, FLAGS.checkpoint_out + ".bestval")

                # Save PyTorch model for PR area calculation
                pytorch_model_path = FLAGS.model_out[:-3] + ".bestval"
                torch.save(model, pytorch_model_path)

        if FLAGS.steps != -1 and state["step"] >= FLAGS.steps:
            break

    save_model(model, FLAGS.model_out)


if __name__ == "__main__":
    app.run(partial(train_init, train))
