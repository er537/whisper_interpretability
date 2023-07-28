import json

import fire
import numpy as np
import torch

from activation_similarities.activation_module import WhipserActivationModule


def cache_activations(activations_to_cache, dataclass):
    module = WhipserActivationModule(
        activations_to_cache=[activations_to_cache],
        data_class=dataclass,
        samples_per_class=30,
    )
    module.forward()
    return module


def get_layerwise_sim_scores(a, b):
    intra_a = torch.mean(torch.mm(a, a.transpose(0, 1)))
    intra_b = torch.mean(torch.mm(b, b.transpose(0, 1)))
    inter = torch.mean(torch.mm(a, b.transpose(0, 1)))
    return (intra_a, intra_b, inter)


def get_activations(
    layer: str = "encoder.blocks.0",
    dataclasses: list = ["SPEECH", "NON_SPEECH"],
):
    activations = {dataclass: torch.empty(0) for dataclass in dataclasses}
    for dataclass in dataclasses:
        actv_module = cache_activations(layer, dataclass)
        layer_activations = actv_module.activations[f"{layer}.output"]
        # concat all batches together
        activations[dataclass] = torch.reshape(layer_activations, (-1, 384))

    all_activations = [(activations[dataclass]) for dataclass in dataclasses]
    all_activations = torch.cat(all_activations, dim=0)
    mean_activation = torch.mean(all_activations, dim=0)

    normed_activations = {}
    for dataclass in dataclasses:
        normed_activations = activations[dataclass] - mean_activation
        normed_activations = normed_activations / normed_activations.norm(dim=1)[:, None]

    return activations


def get_sim_scores(
    layers: list = ["encoder.blocks.3"],
    dataclasses: list = ["SPEECH", "NON_SPEECH"],
    out_path: str = None,
):
    sim_scores = {}
    for layer in layers:
        activations = get_activations(layer, dataclasses)
        sim_scores[layer] = get_layerwise_sim_scores(
            activations[dataclasses[0]], activations[dataclasses[1]]
        )
    if out_path:
        json_ = json.dumps(sim_scores, indent=4)
        with open(out_path, "w") as f:
            f.write(json_)
    print(sim_scores)
    return sim_scores


def get_svd(x):
    U, S, Vh = np.linalg.svd(x, full_matrices=False, compute_uv=True, hermitian=False)
    return U, S, Vh


if __name__ == "__main__":
    fire.Fire(get_sim_scores)
