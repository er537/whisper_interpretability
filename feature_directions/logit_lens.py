import torch
import whisper
import numpy as np
from collections import defaultdict

from whisper_repo import Whisper
from probes.utils.activation_caches import WhisperActivationCache
from utils import get_mels_from_dblx, device
from whisper_repo.tokenizer import get_tokenizer

tokenizer = get_tokenizer(multilingual=True)

DBLX_DICT = {
    "/data/artefacts/am/de/v2023.02_q1rebuild/data_train/train.dblx": "de",
    "/data/artefacts/am/fr/new_normaliser/data_train/train.dblx": "fr",
    "/data/artefacts/am/es/v2023.03_q1rebuild/data_train/train.dblx": "es",
    "/data/artefacts/am/ru/v2023.02_q1rebuild/data_train/train.dblx": "ru",
    "/data/artefacts/am/en/v2023.03_full_reseg/data_train/train.dblx": "en",
}  # file_path, lang_code
NUM_SAMPLES = 10
NUM_LAYERS = 4

def get_lang_probability(activations, lang_code: str, model: torch.nn.Module):
    for name, param in model.named_parameters():
        if name == "decoder.token_embedding.weight":
            token_unembed = torch.transpose(param, 0, 1)
    for name, mod in model.named_modules():
        if name == "decoder.ln":
            ln = mod
    lang_token = tokenizer.encode(
        f"<|{lang_code}|>", allowed_special={f"<|{lang_code}|>"}
    )
    activations = ln(activations)
    logits = (activations @ token_unembed.half())[:, 0]
    mask = torch.ones(logits.shape[-1], dtype=torch.bool)
    mask[list(tokenizer.all_language_tokens)] = False
    logits[:, mask] = -np.inf
    return torch.nn.functional.softmax(logits, dim=-1)[:, lang_token]


def run_logit_lens():
    whisper_model = whisper.load_model("tiny")
    ModelDims = whisper_model.dims
    hacked_model = Whisper(ModelDims)
    hacked_model.load_state_dict(whisper_model.state_dict())
    score_dict = defaultdict(list)  # layer_name, score
    actv_mod = WhisperActivationCache(
        model=hacked_model,
        activations_to_cache=[f"decoder.blocks.{layer}" for layer in range(NUM_LAYERS)],
    )
    for dblx_path, lang_code in DBLX_DICT.items():
        mels = get_mels_from_dblx(dblx_path, NUM_SAMPLES)
        actv_mod.forward(mels)
        for layer in range(NUM_LAYERS):
            activations = actv_mod.activations[f"decoder.blocks.{layer}"].to(device)
            lang_probs = get_lang_probability(activations, lang_code, hacked_model)
            score_dict[layer].append(torch.mean(lang_probs).item())
        actv_mod.reset_state()
    return score_dict


if __name__ == "__main__":
    score_dict = run_logit_lens()
    layerwise_dict = {}
    for layer, scores in score_dict.items():
        layerwise_dict[layer] = sum(scores) / len(scores)
    print(layerwise_dict)
