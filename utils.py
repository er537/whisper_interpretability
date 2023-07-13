import torch
import whisper
from abc import ABC, abstractmethod
from subprocess import CalledProcessError, run
import numpy as np
import glob
import gc
import random
from pathlib import Path
from collections import defaultdict, deque
from functools import partial
import signal
import os
import threading
from absl import logging
import time
import sys
from typing import Callable


device = "cuda" if torch.cuda.is_available() else "cpu"


def dist_logging(message, rank=0):
    if rank == 0:
        logging.info(message)


def get_mels_from_dblx(dblx_path, num_samples):
    batch_mels = []
    with open(dblx_path, "r") as f:
        for _ in range(num_samples):
            line = f.readline().split(" ")
            audio_path = line[1]
            start_time = float(line[2])
            end_time = float(line[3])
            audio = load_audio(audio_path)
            audio = trim_audio(audio, start_time=start_time, end_time=end_time)
            audio = whisper.pad_or_trim(audio.flatten())
            mels = torch.tensor(whisper.log_mel_spectrogram(audio)).to(device)
            batch_mels.append(mels)
    return torch.stack(batch_mels, dim=0)


def trim_audio(
    array: np.array,
    start_time: float,
    end_time: float,
    sample_rate: int = 16_000,
):
    """
    Trim the audio file base array to n_samples, as expected by the encoder.
    """
    start_frame = int(sample_rate * start_time)
    end_frame = int(sample_rate * end_time)

    return array[start_frame:end_frame]


def load_audio(file: str, sample_rate_hz: int = 16_000):
    """
    Taken from Whisper repo: https://github.com/openai/whisper/blob/main/whisper/audio.py

    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sample_rate_hz: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate_hz),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


class BaseActivationModule(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        activations_to_cache: list = ["all"],
        hook_fn: Callable = None,
    ):
        assert model is not None, "no model found"
        self.model = model
        self.step = 0
        self.activations = {}
        self.hooks = []
        self.activations_to_cache = activations_to_cache
        self.hook_fn = hook_fn

    def forward(self, x: torch.tensor):
        self.model.zero_grad()
        self.step += 1
        self.register_hooks()
        model_out = self.custom_forward(self.model, x)
        self.remove_hooks()
        return model_out

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.activations_to_cache or self.activations_to_cache == "all":
                hook_fn = (
                    self.hook_fn
                    if self.hook_fn is not None
                    else self._get_caching_hook(name)
                )
                forward_hook = module.register_forward_hook(hook_fn)
                self.hooks.append(forward_hook)

    def _get_caching_hook(self, name):
        def hook(module, input, output):
            if len(output) > 1:
                output_ = output[0].detach().cpu()
            else:
                output_ = output.detach().cpu()
            self.activations[f"{name}"] = output_

        return hook

    def cluser_activations(self, name):
        raise NotImplementedError

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    @abstractmethod
    def custom_forward(self, dataloader, model: torch.nn.Module) -> dict:
        """
        Should be overidden inside child class to match specific model.
        """
        raise NotImplementedError

    def reset_state(self):
        self.activations = {}


def get_checkpoint_to_start_from(checkpoint_path):
    all_checkpoints = glob.glob(f"{checkpoint_path}*")
    candidate_checkpoints = [
        x
        for x in all_checkpoints
        if ".dist." not in x and ".nan." not in x and ".jump." not in x
    ]

    if candidate_checkpoints:
        candidate_checkpoints.sort(key=os.path.getctime)
        return candidate_checkpoints[-1]
    else:
        return None


def save_model(
    model,
    save_path,
):
    """Saves VAD model as torchscript model"""
    scripted_model = torch.jit.script(model)
    torch.jit.save(scripted_model, save_path)


def save_checkpoint(state, save_path):
    """
    Consumes a generic state dictionary. Unpacks state_dict
    for each element of state if required.
    """

    if "model" in state:
        # we need to call state_dict() on all ranks in case it is calling all_gather
        model = state["model"]

    checkpoint = {}
    for k, v in state.items():
        if hasattr(v, "state_dict"):
            checkpoint[k] = v.state_dict()
        else:
            checkpoint[k] = v
    torch.save(checkpoint, save_path)

    if "model" in state:
        state["model"] = model


def load_checkpoint(
    state,
    load_path,
    device="cpu",
):
    """
    Updates a generic state dictionary. Takes the items in 'checkpoint', and pushes them
    into the preloaded state values
    """
    checkpoint = torch.load(load_path, map_location=device)
    for k, v in state.items():
        if hasattr(v, "load_state_dict"):
            v.load_state_dict(checkpoint[k])
        else:
            state[k] = checkpoint[k]
    del checkpoint
    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    if "torch_rng_state" in state:
        torch.set_rng_state(state["torch_rng_state"])
    if "random_rng_state" in state:
        random.setstate(state["random_rng_state"])
    if "cuda_rng_state" in state:
        torch.cuda.set_rng_state(state["cuda_rng_state"])
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.is_autocast_enabled():
        torch.clear_autocast_cache()

    gc.collect()


class BaseMetadata(dict):

    """Simple Dict for storing metrics/stats. Also maintains a history, useful for aggregation"""

    def __init__(self, history_size=1):
        self.history = defaultdict(partial(deque, maxlen=history_size))

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        # check required for deserialization
        if hasattr(self, "history"):
            self.history[key].append(value)


class Metadata(BaseMetadata):

    """Generic implementation of BaseMetadata storage object, can be subclassed or replaced"""

    def __init__(self, history_size=1):
        super().__init__(history_size)
        self["epoch"] = 0

    def get_standard_metrics(self, rank, step, get_timings=True, get_memory=True):
        items = [
            f"rank {rank}, epoch {self['epoch']}, step {step}, loss {self['loss']:.3f}"
        ]
        if get_timings:
            for metric in self.history:
                if metric.startswith("time_"):
                    items.append(f"{metric} {self[metric]:.3f}s")
        if get_memory:
            if "mem_fwd_peak_allocated" in self:
                items.append(f"f_peak_mem {self['mem_fwd_peak_allocated']:.2f}GB")
            if "mem_fwd_allocated" in self:
                items.append(f"f_mem {self['mem_fwd_allocated']:.2f}GB")
            if "mem_bwd_peak_allocated" in self:
                items.append(f"b_peak_mem {self['mem_bwd_peak_allocated']:.2f}GB")
            if "mem_bwd_allocated" in self:
                items.append(f"b_mem {self['mem_bwd_allocated']:.2f}GB")
            if "mem_fwd_peak_reserved" in self:
                items.append(f"f_rsvd_mem {self['mem_fwd_peak_reserved']:.2f}GB")
            if "mem_fwd_free" in self:
                items.append(f"f_free_mem {self['mem_fwd_free']:.2f}GB")
        return ", ".join(items)

    def log_tb_timings(self, tb_logger, step: int):
        for metric in self:
            if metric.startswith("time_"):
                metric_data = np.array(self.history[metric])
                tb_logger.add_scalar(f"z_time/mean_{metric}", metric_data.mean(), step)
                tb_logger.add_scalar(f"z_time/max_{metric}", metric_data.max(), step)

    def log_tb_memory(self, tb_logger, step: int):
        for prefix in ["mem_fwd", "mem_bwd"]:
            for metric in ["peak_allocated", "allocated"]:
                metric_data = np.array(self.history[f"{prefix}_{metric}"])
                tb_logger.add_scalar(
                    f"z_{prefix}/max_{metric}", metric_data.max(), step
                )

        peak_reserved = np.array(self.history["mem_fwd_peak_reserved"])
        tb_logger.add_scalar("z_mem_fwd/mean_peak_reserved", peak_reserved.mean(), step)
        tb_logger.add_scalar("z_mem_fwd/max_peak_reserved", peak_reserved.max(), step)
        free = np.array(self.history["mem_fwd_free"])
        tb_logger.add_scalar("z_mem_fwd/mean_free", free.mean(), step)

    def snapshot_memory_usage(self, prefix):
        for key, val in snapshot_memory_usage().items():
            self[f"{prefix}_{key}"] = val


def set_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def snapshot_memory_usage():
    """
    peak_allocated: max memory allocated by PyTorch since last snapshot
    allocated: current memory allocated by PyTorch at time of snapshot
    peak_reseved: max memory reserved by PyTorch since last snapshot
    reserved: current memory reserved by PyTorch at time of snapshot
    free: available gpu memory (gpu total - reserved - other) where other is cuda init / kaldi etc
    """
    if not torch.cuda.is_available():
        return {
            "peak_allocated": 0,
            "allocated": 0,
            "peak_reserved": 0,
            "reserved": 0,
            "free": 0,
        }

    memory_usage = {
        "peak_allocated": torch.cuda.max_memory_allocated() / (1024**3),
        "allocated": torch.cuda.memory_allocated() / (1024**3),
        "peak_reserved": torch.cuda.max_memory_reserved() / (1024**3),
        "reserved": torch.cuda.memory_reserved() / (1024**3),
        "free": torch.cuda.mem_get_info()[0] / (1024**3),
    }

    torch.cuda.reset_peak_memory_stats()
    return memory_usage


def dump(state, expdir, checkpoint_out, global_rank=0, exit_out=True):
    with threading.Lock():
        current_step = state["step"]
        out_path = checkpoint_out + ".step" + str(current_step) + ".preempted"
        previous_preemption_checkpoints = glob.glob(f"{expdir}/models/*.preempted*")
        previous_preemption_checkpoints = [
            i for i in previous_preemption_checkpoints if str(current_step) not in i
        ]
        save_checkpoint(state, out_path)

        # Kill the process on global_rank==0.
        # This will kill all the processes on all the nodes in the distributed setting.
        if global_rank > 0:
            return
        Path(f"{expdir}/done_preemption").touch(exist_ok=True)
        print(f"Saving done - preemption handled successfully - {out_path}", flush=True)

        for f in previous_preemption_checkpoints:
            target = Path(f).resolve(strict=True)
            target.unlink(missing_ok=True)

        if exit_out is True:
            print(
                "Killing all processes after 10 seconds - they will correctly exit with status 1"
            )
            time.sleep(10)  # Give other processes more time to save out.
            sys.exit(1)


def dump_checkpoint_on_kill(global_rank=0):
    """
    Register handler on current process and all subprocesses which checkpoints the training state.

    Handle kill signals SIGUSR2 (dump then exit) and SIGUSR1 (dump and continue) by
    setting preemption_fn, which must then be called at the end of the training step, then reset
    to None.
    """

    main_pid = os.getpid()

    def signal_handler(signal_num, frame):
        if os.getpid() == main_pid:
            exit_out = signal_num == signal.SIGUSR2
            if global_rank == 0:
                print(f"Received kill signal {signal_num}", flush=True)
                print(
                    f"Setting preemption_fn with exit={exit_out} - completing final train step",
                    flush=True,
                )
            global preemption_fn
            preemption_fn = partial(dump, global_rank=global_rank, exit_out=exit_out)

    for s in [signal.SIGUSR2, signal.SIGUSR1]:
        signal.signal(s, signal_handler)


def prepare_tb_logging(path=None):
    """
    Ensures that the dir for logging exists and returns a tensorboard logger.
    """
    from torch.utils.tensorboard import SummaryWriter  # dot

    logdir_path = Path(path)
    logdir_path.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(logdir_path, flush_secs=10)


class GradScaler(torch.cuda.amp.GradScaler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._prev_scale = None
        self._skip_count = 0

    def update(self, *args, **kwargs):
        # additionally track changes in scale
        self._prev_scale = self._scale.clone()
        super().update(*args, **kwargs)
        if self._scale < self._prev_scale:
            self._skip_count += 1

    def get_skip_count(self):
        return self._skip_count

    def get_prev_scale(self):
        return self._prev_scale.item()

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["prev_scale"] = self._prev_scale
        state_dict["skip_count"] = self._skip_count
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "prev_scale" in state_dict:
            self._prev_scale = state_dict["prev_scale"]
        if "skip_count" in state_dict:
            self._skip_count = state_dict["skip_count"]
