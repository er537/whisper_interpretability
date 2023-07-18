import torch
import whisper_repo

from activation_similarities.dataset import ClasswiseDataset
from util import device, BaseActivationModule


class WhipserActivationModule(BaseActivationModule):
    def __init__(
        self,
        model_name: str = "tiny.en",
        language: str = "en",
        data_class: str = "NON_SPEECH",
        activations_to_cache: list = [],
        samples_per_class: int = 480_000,
        batch_size: int = 1,
    ):
        self.model = whisper_repo.load_model(model_name)
        self.loader = self._get_dataloader(data_class, samples_per_class, batch_size)
        self.language = language
        self.activations_to_cache = (
            activations_to_cache if len(activations_to_cache) > 0 else "all"
        )
        self.named_modules = list(
            {name: mod for name, mod in self.model.named_modules()}
        )
        super().__init__(self.model, self.activations_to_cache)

    def _get_dataloader(self, data_class, samples_per_class, batch_size):
        dataset = ClasswiseDataset(
            class_labels=[data_class],
            pad=True,
            audio_samples_per_class=samples_per_class,
        )
        return iter(torch.utils.data.DataLoader(dataset, batch_size))

    def custom_forward(self, model: torch.nn.Module) -> dict:
        options = whisper_repo.DecodingOptions(
            language=self.language, without_timestamps=False, fp16=False
        )
        model.to(device)
        for mels, labels in self.loader:
            print(f"Decoding:{mels.shape}")
            mels = mels.to(device)
            model.decode(mels, options)
