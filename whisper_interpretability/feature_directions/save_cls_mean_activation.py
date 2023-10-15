import pathlib
import warnings
from collections import defaultdict

import fire
import torch
from global_utils import device
from global_whisper_utils import LibriSpeechDataset, WhisperActivationCache
from probes.train.dataset import MultiClassDataset

OUT_DIR = "/exp/ellenar/whisper_activations"
LANG_CODE = "fr"

"""
Save the mean activation for all layers in [activations_to_cache] to disk
Used for activation steering eg save the mean french activation and use it to 'steer' german text
"""


def get_activations(
    activations_to_cache: list = [
        "encoder.blocks.0",
        "encoder.blocks.1",
        "encoder.blocks.2",
        "encoder.blocks.3",
    ],
    num_samples: int = 100,
    class_label: str = LANG_CODE,
    model_name: str = "tiny",
    sql_path: str = f"/home/ellenar/probes/just_{LANG_CODE}_val.sql",
    batch_size: int = 50,
):
    if not device == "cuda":
        warnings.warn("This is much faster if you run it on a GPU")
    actv_cache = WhisperActivationCache(
        model_name=model_name, activations_to_cache=activations_to_cache
    )
    dataset = MultiClassDataset(
        num_entries=num_samples, class_labels=[class_label], sql_path=sql_path
    )
    # dataset = LibriSpeechDataset(url="dev-clean", return_mels=True)
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size))
    all_actvs = defaultdict(list)  # layer_name: activations
    for i, (data, *labels) in enumerate(dataloader):
        actv_cache.reset_state()
        actv_cache.forward(data.to(device))
        for layer in activations_to_cache:
            actvs = actv_cache.activations[f"{layer}"].to(dtype=torch.float32).to("cpu")
            all_actvs[layer].append(actvs)
        if i >= num_samples / batch_size:
            break
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    for layer, actvs in all_actvs.items():
        actvs = torch.cat(actvs, dim=0).mean(
            dim=0
        )  # mean over all batches but retain sequence length
        torch.save(actvs, f"{OUT_DIR}/{model_name}_{layer}_{class_label}")
        print(f"Saved {actvs.shape} activations for layer {layer} to disk")


if __name__ == "__main__":
    fire.Fire(get_activations)
