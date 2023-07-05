import torch
import whisper
import torchaudio

from utils import device, BaseActivationModule


class WhisperActivationCache(BaseActivationModule):
    """
    Use hooks in BaseActivationModule to cache intermediate activations while running forward pass
    """

    def __init__(
        self,
        model_name: str = "tiny",
        activations_to_cache: list = ["encoder.blocks.0"], # pass "all" to cache all activations
    ):
        self.model = whisper.load_model(model_name).to(device)
        self.activations_to_cache = activations_to_cache
        self.named_modules = list(
            {name: mod for name, mod in self.model.named_modules()}
        )
        super().__init__(self.model, self.activations_to_cache)

    def custom_forward(self, model: torch.nn.Module, mels) -> dict:
        options = whisper.DecodingOptions(
            without_timestamps=False, fp16=(device == "cuda")
        )
        output = model.decode(mels, options)
        return output


class Wav2VecActivationCache:
    """
    Wav2Vec returns the output from every transformer block as a list of features,
    so we don't need to use hooks
    """

    def __init__(
        self,
        layer_idx_to_cache: int = 0,  # idx in range {0, 10}
    ):
        self.model = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_model().to(device)
        self.layer_idx_to_cache = int(layer_idx_to_cache)
        self.activations = {}

    def forward(self, waveforms) -> dict:
        with torch.inference_mode():
            features, _ = self.model.extract_features(waveforms.squeeze(1))
        self.activations[f"{self.layer_idx_to_cache}.output"] = features[
            self.layer_idx_to_cache
        ]

    @staticmethod
    def reset_state():
        return
