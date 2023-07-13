import torch
import whisper

from whisper_repo.model import Whisper
from utils import device, get_mels_from_dblx
from probes.utils.activation_caches import WhisperActivationCache


DBLX_DICT = {
    "/data/artefacts/am/de/v2023.02_q1rebuild/data_train/train.dblx": "de",
    "/data/artefacts/am/fr/new_normaliser/data_train/train.dblx": "fr",
    "/data/artefacts/am/es/v2023.03_q1rebuild/data_train/train.dblx": "es",
    "/data/artefacts/am/ru/v2023.02_q1rebuild/data_train/train.dblx": "ru",
    "/data/artefacts/am/en/v2023.03_full_reseg/data_train/train.dblx": "en",
}  # file_path, lang_code
NUM_SAMPLES = 10
NUM_LAYERS = 4


def hook_fn(module, input, output):
    return torch.zeros_like(output[0])


def get_layerwise_scores():
    whisper_model = whisper.load_model("tiny")
    ModelDims = whisper_model.dims
    hacked_model = Whisper(ModelDims)
    hacked_model.load_state_dict(whisper_model.state_dict())
    score_dict = {}  # layer_name, score
    for layer in range(NUM_LAYERS):
        layer_scores = []
        actv_mod = WhisperActivationCache(
            model=hacked_model,
            activations_to_cache=[f"decoder.blocks.{layer}.cross_attn"],
            hook_fn=hook_fn,
        )
        for dblx_path, lang_code in DBLX_DICT.items():
            mels = get_mels_from_dblx(dblx_path, NUM_SAMPLES)
            output = actv_mod.forward(mels.to(device))
            actv_mod.reset_state()
            lang_probs = [
                output[i].language_probs[lang_code] for i in range(len(output))
            ]
            layer_scores.append(sum(lang_probs) / len(lang_probs))
        score_dict[layer] = sum(layer_scores) / len(layer_scores)
    return score_dict


if __name__ == "__main__":
    score_dict = get_layerwise_scores()
    print(score_dict)
