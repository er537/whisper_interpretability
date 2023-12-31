from typing import Optional

import torch
from global_utils import device


def get_features(
    feature_type: str = "learnt", chk_path: Optional[str] = None, feature_idx: int = None, d_model: int = 1536
):
    """
    Inputs:
    feature_type: "learnt", "neuron_basis" or "rand_orth"
    chk_path: path to autoencoder checkpoint for learnt features
    d_model:

    Returns: 'features' learnt by autoencoder, neuron basis or a random orthogonal basis
    """
    chk = torch.load(chk_path)
    encoder_weight = chk["model"]["decoder.weight"]
    if feature_type == "learnt":
        encoder_bias = chk["model"]["encoder_bias"]
    elif feature_type == "neuron_basis":
        encoder_weight = torch.eye(d_model, d_model)
        encoder_bias = torch.zeros(d_model)
    elif feature_type == "rand_orth":
        encoder_weight = torch.nn.init.orthogonal_(
            torch.empty(d_model, d_model)
        )
        encoder_bias = torch.zeros(d_model)
    else:
        raise Exception("type must be 'learnt', 'neuron_basis' or 'rand_orth'")
    if feature_idx is not None:
        feature = encoder_weight[:, feature_idx]  # [d_model]
        return feature.to(device), encoder_bias[feature_idx].to(device)
    return encoder_weight.to(device), encoder_bias.to(device)
