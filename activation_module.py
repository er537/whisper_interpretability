from abc import ABC, abstractmethod
import torch
import whisper

from dataset import LibriSpeech
from utils import device


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
            output_ = output[0].detach().cpu()
            input_ = input[0].detach().cpu()
            with torch.no_grad():
                self.activations[f"{name}.input"] = input_
                self.activations[f"{name}.output"] = output_

        return hook

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


class WhipserActivationModule(BaseActivationModule):
    def __init__(
        self,
        model_name: str = "tiny.en",
        language: str = "en",
        activations_to_cache: list = [],
    ):
        self.model = whisper.load_model(model_name)
        self.loader = self._get_dataloader()
        self.language = language
        self.activations_to_cache = (
            activations_to_cache if len(activations_to_cache) > 0 else "all"
        )
        self.named_modules = {name: mod for name, mod in self.model.named_modules()}
        super().__init__(self.model, self.activations_to_cache)

    def _get_dataloader(self):
        dataset = LibriSpeech("test-clean")
        return iter(torch.utils.data.DataLoader(dataset, batch_size=1))

    def custom_forward(self, model: torch.nn.Module) -> dict:
        options = whisper.DecodingOptions(
            language=self.language, without_timestamps=False, fp16=False
        )
        audio, sample_rate, texts, num_frames = next(self.loader)
        assert sample_rate == 16000
        audio = whisper.pad_or_trim(audio.flatten()).to(device)
        mels = whisper.log_mel_spectrogram(audio)
        results = model.decode(mels, options)
        return results, texts
