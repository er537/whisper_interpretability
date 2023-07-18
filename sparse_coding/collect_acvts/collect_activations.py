import torch
from collections import defaultdict
import warnings
import fire
import pathlib
import os

from util import device
from utils.activation_caches import WhisperActivationCache
from sparse_coding.collect_acvts.dataset import WhisperMelsDataset

MODEL_NAME = "tiny"
OUT_DIR = f"/exp/ellenar/sparse_coding/whisper_activations_{MODEL_NAME}"


def get_activations(
    activations_to_cache: list = [
        "encoder.blocks.3",
        "decoder.blocks.3",
        "decoder.blocks.3.cross_attn.out",
    ],
):
    if not device == "cuda":
        warnings.warn("This is much faster if you run it on a GPU")
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    actv_cache = WhisperActivationCache(
        model_name=MODEL_NAME, activations_to_cache=activations_to_cache
    )
    dataset = WhisperMelsDataset()
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=10))
    for data, lang_codes, audio_paths in dataloader:
        actv_cache.reset_state()
        actv_cache.forward(data.to(device))
        for layer in activations_to_cache:
            layer_actvs = actv_cache.activations[f"{layer}"].to("cpu")
            for i, (lang_code, audio_path) in enumerate(zip(lang_codes, audio_paths)):
                out_path_ext = get_out_path_ext(
                    audio_path=audio_path, lang_code=lang_code
                )
                torch.save(layer_actvs[i], f"{OUT_DIR}/{out_path_ext}_{layer}")


def get_out_path_ext(audio_path, lang_code):
    """
    Use the dirname, basename and lang_code to form a unique file path extension
    """
    dirname = os.path.dirname(audio_path).split("/")[-1]
    basename = os.path.basename(audio_path)
    return f"{dirname}_{basename}_{lang_code}"


if __name__ == "__main__":
    fire.Fire(get_activations)
