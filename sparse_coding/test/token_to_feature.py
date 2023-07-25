import torch
import warnings
import fire
from torch import nn
from collections import defaultdict
import pickle
import os
from typing import List

from util import device
from utils.activation_caches import WhisperActivationCache
from sparse_coding.collect_acvts.dataset import WhisperMelsDataset
from whisper_repo.tokenizer import get_tokenizer

tokenizer = get_tokenizer(multilingual=True)

MODEL_NAME = "tiny"
OUT_DIR = f"/exp/ellenar/sparse_coding/whisper_activations_{MODEL_NAME}"


def get_features(feature_type: str = "learnt", chk_path: str = None):
    """
    type: "learnt", "neuron_basis" or "rand_orth"
    return either features learnt by autoencoder, neuron basis or a random orthogonal basis
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


def get_activations(
    num_entries: int,  # num of samples in dataset
    split: str,  # train or val
    sql_path: str,
    out_path: str,
    activations_to_cache: str = "decoder.blocks.2.mlp.0",
    selected_tokens: List[str] = [
        " small",
        " tiny",
        " mini",
        " beautiful",
        " stunning",
        " pretty",
    ],  # tokens of interest
    batch_size=50,
    chk_path: str = "/exp/ellenar/sparse_coding/train/20230722_whisper_tiny_decoder.blocks.2.mlp.0_n_dict_components_2000_l1_alpha_5e-4/models/checkpoint.pt.step800",
    feature_type: str = "learnt",
):
    if not device == "cuda":
        warnings.warn("This is much faster if you run it on a GPU")

    dataset = WhisperMelsDataset(
        max_num_entries=num_entries, split=split, sql_path=sql_path
    )
    actv_cache = WhisperActivationCache(
        model_name=MODEL_NAME, activations_to_cache=[activations_to_cache]
    )
    encoder_weight, encoder_bias = get_features(
        feature_type=feature_type, chk_path=chk_path
    )
    token_activation_dict = defaultdict(
        lambda: [0, torch.zeros(encoder_weight.shape[-1])]
    )  # {token: [num_points: int, mean_activations: Tensor]}
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size))
    selected_token_idxs = convert_token_strs_to_idxs(selected_tokens)
    feature_actvs_running_mean = torch.zeros(encoder_weight.shape[-1])
    running_mean_count = 0
    for batch_idx, (data, *_) in enumerate(dataloader):
        actv_cache.reset_state()
        output = actv_cache.forward(data.to(device))
        actvs = actv_cache.activations[activations_to_cache]
        batch_feature_activations = nn.ReLU()(
            (actvs.to(device).float() @ encoder_weight) + encoder_bias
        ).cpu()  # shape = (bsz, seq_len, dict_size)
        feature_actvs_running_mean += torch.mean(
            torch.mean(batch_feature_activations, dim=1), dim=0
        )
        running_mean_count += 1
        for i in range(len(output)):
            tokens = output[i].tokens
            feature_activations = batch_feature_activations[i, :, :]
            for token_idx in selected_token_idxs:
                if token_idx in tokens:
                    token_activations = feature_activations[tokens.index(token_idx), :]
                    num_points, token_actv_means = token_activation_dict[
                        tokenizer.decode([token_idx])
                    ]
                    token_activation_dict[tokenizer.decode([token_idx])] = [
                        num_points + 1,
                        ((num_points * token_actv_means) + token_activations)
                        / (num_points + 1),
                    ]
        print(f"Processed batch {batch_idx} of {num_entries//batch_size}")

    print(f"Saving")
    dirname = os.path.dirname(out_path)
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    torch.save(
        feature_actvs_running_mean / running_mean_count,
        f"{out_path}_{activations_to_cache}_feature_actv_means.pt",
    )
    with open(
        f"{out_path}_{activations_to_cache}_per_token_activations.pkl", "wb"
    ) as outfile:
        pickle.dump(dict(token_activation_dict), outfile)


def convert_token_strs_to_idxs(tokens_strs):
    token_idxs = []
    for token_str in tokens_strs:
        token_idxs += tokenizer.encode(token_str)
    return token_idxs


if __name__ == "__main__":
    fire.Fire(get_activations)
