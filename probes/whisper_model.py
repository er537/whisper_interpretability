import torch
import whisper

from utils import device, BaseActivationModule


class WhipserActivationCache(BaseActivationModule):
    def __init__(
        self,
        dataloader: torch.utils.data.dataloader,
        model_name: str = "base",
        activations_to_cache: list = ["encoder.blocks.0"],
    ):
        self.model = whisper.load_model(model_name)
        self.dataloader = iter(dataloader)
        self.activations_to_cache = (
            activations_to_cache if len(activations_to_cache) > 0 else "all"
        )
        self.named_modules = list(
            {name: mod for name, mod in self.model.named_modules()}
        )
        super().__init__(self.model, self.activations_to_cache)

    def custom_forward(self, model: torch.nn.Module) -> dict:
        options = whisper.DecodingOptions(
            without_timestamps=False, fp16=(device == "cuda")
        )
        model.to(device)
        mels, labels = next(self.dataloader)
        print(f"Running Whisper on mels:{mels.shape}")
        mels = mels.to(device)
        output = model.decode(mels, options)
        return output, labels
