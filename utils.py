import torch
from abc import ABC, abstractmethod
from subprocess import CalledProcessError, run
import numpy as np
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


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
    assert len(array) > end_frame - start_frame

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
    def __init__(self, model: torch.nn.Module, activations_to_cache: str = "all"):
        assert model is not None, "no model found"
        self.model = model
        self.step = 0
        self.activations = {}
        self.hooks = []
        self.activations_to_cache = activations_to_cache

    def forward(self):
        self.model.zero_grad()
        self.step += 1
        self.register_hooks()
        model_out = self.custom_forward(self.model)
        self.remove_hooks()
        return model_out

    def register_hooks(self):
        for name, module in self.model.named_modules():
            if name in self.activations_to_cache or self.activations_to_cache == "all":
                forward_hook = module.register_forward_hook(self._get_hook(name))
                self.hooks.append(forward_hook)

    def _get_hook(self, name):
        def hook(module, input, output):
            output_ = output.detach().cpu()
            input_ = input[0].detach().cpu()
            with torch.no_grad():
                if f"{name}.input" in self.activations:
                    self.activations[f"{name}.input"] = torch.cat(
                        (self.activations[f"{name}.input"], input_), dim=0
                    )
                    self.activations[f"{name}.output"] = torch.cat(
                        (self.activations[f"{name}.output"], output_), dim=0
                    )
                else:
                    self.activations[f"{name}.input"] = input_
                    self.activations[f"{name}.output"] = output_

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
