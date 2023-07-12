import torch
import whisper

from whisper_repo.model import Whisper
from utils import load_audio, trim_audio, device
from probes.utils.activation_caches import WhisperActivationCache


DBLX_DICT = {
    "/data/artefacts/am/de/v2023.02_q1rebuild/data_train/train.dblx": "de",
    "/data/artefacts/am/fr/new_normaliser/data_train/train.dblx": "fr",
    "/data/artefacts/am/es/v2023.03_q1rebuild/data_train/train.dblx": "es",
    "/data/artefacts/am/ru/v2023.02_q1rebuild/data_train/train.dblx": "ru",
    "/data/artefacts/am/ru/v2023.02_q1rebuild/data_train/train.dblx": "en",
}  # file_path, lang_code
NUM_SAMPLES = 10
NUM_LAYERS = 4
NUM_HEADS = 6


def get_mels(dblx_path):
    batch_mels = []
    with open(dblx_path, "r") as f:
        for _ in range(NUM_SAMPLES):
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
            activations_to_cache=[f"decoder.blocks.1.cross_attn"],
            hook_fn=hook_fn(),
        )
        for dblx_path, lang_code in DBLX_DICT.items():
            mels = get_mels(dblx_path)
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
