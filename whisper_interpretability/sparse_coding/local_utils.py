from typing import Optional

import torch

from global_utils import device


def get_features(feature_type: str = "learnt", chk_path: Optional[str] = None):
    """
    Inputs:
    feature_type: "learnt", "neuron_basis" or "rand_orth"
    chk_path: path to autoencoder checkpoint for learnt features

    Returns: 'features' learnt by autoencoder, neuron basis or a random orthogonal basis
    """
    chk = torch.load(chk_path)
    encoder_weight = chk["model"]["decoder.weight"]
    if feature_type == "learnt":
        encoder_bias = chk["model"]["encoder_bias"]
    elif feature_type == "neuron_basis":
        encoder_weight = torch.eye(encoder_weight.shape[0], encoder_weight.shape[0])
        encoder_bias = torch.zeros(encoder_weight.shape[0])
    elif feature_type == "rand_orth":
        encoder_weight = torch.nn.init.orthogonal_(
            torch.empty(encoder_weight.shape[0], encoder_weight.shape[0])
        )
        encoder_bias = torch.zeros(encoder_weight.shape[0])
    else:
        raise Exception("type must be 'learnt', 'neuron_basis' or 'rand_orth'")
    return encoder_weight.to(device), encoder_bias.to(device)
