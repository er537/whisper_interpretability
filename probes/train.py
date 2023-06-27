from functools import partial
from math import inf
from time import perf_counter

import numpy as np
import torch
from absl import app, flags, logging
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from probes.dataset import VADDataset
from probes.whisper_model import WhipserActivationCache
from probes.probe import Probe
from aladdin import util
from aladdin.am.train.augmentation import AugmentationFunction
from aladdin.base_train import train_init
from aladdin.dataloader.streaming import BatchTimer
from aladdin.util import (  # maybe_log_grad_scaler,
    EmptyScheduler,
    FixedRandomState,
    FlatCA,
    GradScaler,
    Metadata,
    RAdam,
    device,
    dist_logging,
    dump_checkpoint_on_kill,
    dump_memory_allocator_trace,
    initialize_torch_settings,
    load_checkpoint,
    load_model,
    prepare_standard_logging,
    prepare_tb_logging,
    save_checkpoint,
    set_seeds,
    snapshot_memory_usage,
)
from aladdin.vad.train.dataset import VadDataLoader, VadDataset
from aladdin.vad.train.model.mini_model import (
    Model,
)  # TODO: allow option to select either model.py or mini_model.py
from aladdin.vad.utils import save_model

torch.backends.cudnn.benchmark = True

FLAGS = flags.FLAGS  # See base_train.py for others
# Paths
flags.DEFINE_string(
    "RIRs_dir", None, "optional path to the RIRs dir for reverb augmentation"
)
flags.DEFINE_string(
    "bootstrap_model_path",
    None,
    "path to Silero model directory",
)

# Optimization
flags.DEFINE_boolean(
    "lr_schedule",
    True,
    "state if an learning rate scheduler is required during training",
)
flags.DEFINE_integer("val_batch_size", 16, "batch size used during validation")
flags.DEFINE_integer(
    "train_window_size",
    4_000,
    "num audio samples (raw audio) to push into model at once",
)
flags.DEFINE_integer("val_window_size", 4_000, "window size for validation")
flags.DEFINE_integer("buffer_len", 10_000, "size of shuffle buffer used by VadDataset")
flags.DEFINE_integer(
    "test_pr_every", 10000, "evaluate area under PR curve every n steps"
)
flags.DEFINE_float(
    "scheduler_decay", 1.0 / 3, "proportion of training steps over which lr decays"
)
flags.DEFINE_float("clip_thresh", 1.0, "value to clip gradients to")

# Filtering non-speech samples
flags.DEFINE_float(
    "bootstrap_threshold",
    0.1,
    "segments of non-speech with probability of speech greater than this value will be discarded if bootstrapping",
)
flags.DEFINE_boolean(
    "train_filter", True, "filter negatives samples in training dataloader"
)
flags.DEFINE_boolean("train_balance", True, "class equality in training dataloader")
flags.DEFINE_boolean(
    "val_filter", True, "filter negatives samples in validation dataloader"
)
flags.DEFINE_boolean("val_balance", True, "class equality in validation dataloader")

# Regularization
flags.DEFINE_float("weight_decay", 0.0, "weight decay to use for training")
flags.DEFINE_float("wav_augment_prob", 0.0, "prob of applying pitch in time domain")
flags.DEFINE_float(
    "telephony_8khz_prob",
    0.0,
    "prob of downsampling, applying phone codec and upsampling",
)
flags.DEFINE_float("RIRs_prob", 0.0, "prob of applying RIRs to scp")
flags.DEFINE_float("vol_perturbation_prob", 0.0, "prob of augmenting volume")


def get_class_freq(labels):
    """Count number of non-speech and speech labels in a tensor"""
    return (labels.detach().cpu() == 0).sum(), (labels.detach().cpu() == 1).sum()


def validate(val_datastream, model, whisper_model, loss_fn):
    model.eval()
    losses = []
    frames_seen = 0
    accs = []

    # reset random seed for determisitic validation
    with FixedRandomState(0):
        for data, labels in val_datastream:
            data, labels = data.to(device), labels.to(device)

            with torch.no_grad():
                whisper_model.forward()
                pred = model(whisper_model.activations[f"{FLAGS.probe_layer}.output"])
                pred = model(data)
                losses.append(loss_fn(pred, labels).item())
                frames_seen += labels.numel()

                # Calculate accuracy as well
                acc = (torch.argmax(pred, dim=1) == labels).sum() / len(labels)
                accs.append(acc.item())

    model.train()
    return np.array(losses).mean(), np.array(accs).mean(), frames_seen


def get_probe_feat_dim(datapath):
    dataset = VADDataset(dblx_path=datapath)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    whisper_model = WhipserActivationCache(
        dataloader=dataloader, activations_to_cache=[FLAGS.probe_layer]
    )
    whisper_model.forward()
    activations_shape = whisper_model.activations[
        "encoder.blocks.0.output"
    ].shape  # (bsz, seq_len, d_model)
    return activations_shape[-1]


def train(FLAGS, global_rank=0):
    torch.set_num_threads(1)
    set_seeds(FLAGS.seed)

    fd = get_probe_feat_dim(FLAGS.train_dblx)
    ## TODO: get shape of required layer and init probe
    model = Probe(feat_dim=fd).to(device)

    # setup logging
    if global_rank == 0:
        tb_logger = prepare_tb_logging(FLAGS.expdir)
        prepare_standard_logging("training", FLAGS.debug)
    meta = Metadata(history_size=FLAGS.log_tb_every)
    memory_usage = snapshot_memory_usage()
    if memory_usage is not None:
        dist_logging(f"pretrain_mem {memory_usage['allocated']:.5f}GB", global_rank)

    if FLAGS.model_out is None:
        FLAGS.model_out = FLAGS.expdir + "/model"

    dist_logging(
        "Model: %.2fM" % (sum(p.numel() for p in model.parameters()) / 1.0e6),
        global_rank,
    )

    optimizer = RAdam(
        model.parameters(), eps=1e-5, lr=FLAGS.lr, weight_decay=FLAGS.weight_decay
    )
    scaler = GradScaler()
    if FLAGS.lr_schedule:
        scheduler = FlatCA(
            optimizer,
            steps=FLAGS.steps,
            eta_min=0,
            decay_proportion=FLAGS.scheduler_decay,
        )
    else:
        scheduler = EmptyScheduler(optimizer)

    train_dataset = VADDataset(dblx_path=FLAGS.train_dblx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=FLAGS.batch_size
    )
    train_whisper_model = WhipserActivationCache(
        dataloader=train_loader, activations_to_cache=[FLAGS.probe_layer]
    )
    val_dataset = VADDataset(dblx_path=FLAGS.val_dblx)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=10)
    val_whisper_model = WhipserActivationCache(
        dataloader=val_loader, activations_to_cache=[FLAGS.probe_layer]
    )
    # Object that contains the main state of the train loop
    state = {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "scaler": scaler,
        "step": 0,
        "best_val_loss": inf,
        "total_speech_seconds_seen": 0,
        "total_non_speech_seconds_seen": 0,
        "total_time_ms": 0,
    }

    meta["effective_batch_size"] = FLAGS.batch_size
    meta["model_params"] = sum(x.numel() for x in model.parameters())

    if FLAGS.checkpoint:
        # loading state_dicts in-place
        print(f"Checkpoint: {FLAGS.checkpoint}")
        load_checkpoint(state, FLAGS.checkpoint, device=device)

    if global_rank == 0:
        dump_checkpoint_on_kill(global_rank=global_rank)

    # Set seeds so each rank gets different data
    set_seeds(FLAGS.seed + state["step"] + global_rank)

    loss_fn = torch.nn.CrossEntropyLoss(reduction="mean")
    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        num_non_speech, num_speech = get_class_freq(labels)
        state["total_speech_seconds_seen"] += (
            num_speech * FLAGS.train_window_size / 16000
        )
        state["total_non_speech_seconds_seen"] += (
            num_non_speech * FLAGS.train_window_size / 16000
        )

        # Forward pass
        forward_time = 0
        with autocast():
            start_time = perf_counter()
            train_whisper_model.forward()
            pred = model(train_whisper_model.activations[f"{FLAGS.probe_layer}.output"])
            forward_time += perf_counter() - start_time
            loss = loss_fn(pred, labels)
            meta.snapshot_memory_usage("mem_fwd")

        # Backward pass
        backward_time = 0
        start_time = perf_counter()
        scaler.scale(loss).backward()
        backward_time += perf_counter() - start_time

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), FLAGS.clip_thresh)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        model.zero_grad()
        state["step"] += 1
        meta["loss"] = loss.item()
        meta["time_forward"] = forward_time
        meta["time_backward"] = backward_time

        meta.snapshot_memory_usage("mem_bwd")

        if state["step"] % FLAGS.log_every == 0 and global_rank == 0:
            dist_logging(f" step {state['step']}, loss {loss.item():.3f}", global_rank)

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

        # validate periodically
        if state["step"] % FLAGS.val_every == 0 and global_rank == 0:
            dist_logging("Starting to validate", global_rank)
            val_loss, val_acc, val_frames_seen = validate(val_loader, model, loss_fn)
            logging.info(
                f"{state['step']} validation, loss={val_loss:.3f}, "
                f"acc={val_acc:.3f}, "
                f"{val_frames_seen:,} frames validated"
            )
            # log validation losses
            tb_logger.add_scalar("val/loss", val_loss, state["step"])
            tb_logger.add_scalar("val/acc", val_acc, state["step"])
            if val_loss.item() < state["best_val_loss"]:
                logging.info("Saving new best validation")
                save_model(model, FLAGS.model_out + ".bestval")
                state["best_val_loss"] = val_loss.item()
                save_checkpoint(state, FLAGS.checkpoint_out + ".bestval")

                # Save PyTorch model for PR area calculation
                pytorch_model_path = FLAGS.model_out[:-3] + ".bestval"
                torch.save(model, pytorch_model_path)

        # save out model periodically
        if state["step"] % FLAGS.save_every == 0:
            save_model(model, FLAGS.model_out + ".step" + str(state["step"]))
            save_checkpoint(state, FLAGS.checkpoint_out + ".step" + str(state["step"]))

        if state["step"] >= FLAGS.steps:
            break

        if util.preemption_fn is not None:
            util.preemption_fn(state, FLAGS.expdir, FLAGS.checkpoint_out)
            util.preemption_fn = None

    save_model(model, FLAGS.model_out)


if __name__ == "__main__":
    app.run(partial(train_init, train))
