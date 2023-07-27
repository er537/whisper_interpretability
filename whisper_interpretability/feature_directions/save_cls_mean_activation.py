import torch
from collections import defaultdict
import warnings
import fire
import pathlib

from global_utils import device
from global_whisper_utils import WhisperActivationCache
from probes.train.dataset import MultiClassDataset

OUT_DIR = "/exp/ellenar/whisper_activations"
LANG_CODE = "de"

"""
Save the mean activation for all layers in [activations_to_cache] to disk
Used for activation steering eg save the mean french activation and use it to 'steer' german text
"""


def get_activations(
    activations_to_cache: list = [
        "decoder.blocks.0.cross_attn.out",
        "decoder.blocks.1.cross_attn.out",
        "decoder.blocks.2.cross_attn.out",
        "decoder.blocks.3.cross_attn.out",
    ],
    num_samples: int = 100,
    class_label: str = LANG_CODE,
    model_name: str = "tiny",
    sql_path: str = f"/home/ellenar/probes/just_{LANG_CODE}_val.sql",
):
    if not device == "cuda":
        warnings.warn("This is much faster if you run it on a GPU")
    actv_cache = WhisperActivationCache(
        model_name=model_name, activations_to_cache=activations_to_cache
    )
    dataset = MultiClassDataset(
        num_entries=num_samples, class_labels=[class_label], sql_path=sql_path
    )
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=50))
    all_actvs = defaultdict(list)  # layer_name: activations
    for data, labels in dataloader:
        actv_cache.reset_state()
        actv_cache.forward(data.to(device))
        for layer in activations_to_cache:
            mean_actvs = (
                actv_cache.activations[f"{layer}.output"]
                .to(dtype=torch.float32)
                .mean(dim=1)
                .to("cpu")
            )
            all_actvs[layer].append(mean_actvs)
    pathlib.Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    for layer, actvs in all_actvs.items():
        actvs = torch.cat(actvs, dim=0)
        torch.save(actvs, f"{OUT_DIR}/{model_name}_{layer}_{class_label}")
        print(f"Saved {actvs.shape} activations for layer {layer} to disk")


if __name__ == "__main__":
    fire.Fire(get_activations)
