import torch
import warnings
import fire
import heapq
from functools import partial
from torch import nn
from collections import defaultdict
import pickle
import os

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
    max_num_entries: int,
    split: str,  # train or val
    sql_path: str,
    out_path: str,
    activations_to_cache: list = [
        "decoder.blocks.2.mlp.0",
    ],
    batch_size=100,
    chk_path: str = "/exp/ellenar/sparse_coding/train/20230722_whisper_tiny_decoder.blocks.2.mlp.0_n_dict_components_2000_l1_alpha_5e-4/models/checkpoint.pt.step800",
    feature_type: str = "learnt",
):
    if not device == "cuda":
        warnings.warn("This is much faster if you run it on a GPU")

    featurewise_max_activating_fragments = defaultdict(
        partial(MaxActivatingFragments, 10)
    )  # key=feature_idx, value=max_activating_fragments
    dataset = WhisperMelsDataset(
        max_num_entries=max_num_entries, split=split, sql_path=sql_path
    )
    actv_cache = WhisperActivationCache(
        model_name=MODEL_NAME, activations_to_cache=activations_to_cache
    )
    encoder_weight, encoder_bias = get_features(
        feature_type=feature_type, chk_path=chk_path
    )
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size))
    for batch_idx, (data, lang_codes, audio_paths) in enumerate(dataloader):
        actv_cache.reset_state()
        output = actv_cache.forward(data.to(device))
        for layer in activations_to_cache:
            layer_actvs = actv_cache.activations[f"{layer}"]
            batch_feature_activations = nn.ReLU()(
                (layer_actvs.to(device).float() @ encoder_weight) + encoder_bias
            ).cpu()
            for i, audio_path in enumerate(audio_paths):
                tokens = output[i].tokens
                for feature_idx in range(batch_feature_activations.shape[-1]):
                    # for every feature_activation, maybe add it to the list of max activating fragments
                    feature_activation_scores = batch_feature_activations[
                        i, :, feature_idx
                    ]
                    featurewise_max_activating_fragments[feature_idx].heappushpop(
                        feature_activation_scores, audio_path, tokens
                    )
        print(f"Processed batch {batch_idx} of {max_num_entries//batch_size}")
    save_max_activating_fragments(featurewise_max_activating_fragments, out_path)


def save_max_activating_fragments(
    featurewise_max_activating_fragments: dict, out_path: str
):
    for (
        feature_idx,
        max_activating_fragments,
    ) in featurewise_max_activating_fragments.items():
        print(f"Saving {feature_idx} of {len(featurewise_max_activating_fragments)}")
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        with open(f"{out_path}/{feature_idx}.pkl", "wb") as outfile:
            pickle.dump(max_activating_fragments, outfile)


class MaxActivatingFragments(list):
    def __init__(self, max_len):
        super().__init__()
        self.metadata = (
            {}
        )  # keys are activation scores, values are audio paths and transcripts
        self.max_len = max_len
        self.mean_activation = 0
        self.num_activations = 0  # used to calculate running mean

    def heappushpop(self, activations, audio_path, tokens):
        max_activation = torch.max(activations).item()
        self.mean_activation = (
            (self.mean_activation * self.num_activations)
            + torch.mean(activations).item()
        ) / (self.num_activations + 1)
        self.num_activations += 1
        transcript = tokenizer.decode(tokens)
        self.metadata[max_activation] = [audio_path, transcript, tokens, activations]
        if len(self) == self.max_len:
            min_activation = heapq.heappushpop(self, max_activation)
            try:
                del self.metadata[min_activation]
            except:
                KeyError
        else:
            heapq.heappush(self, max_activation)


if __name__ == "__main__":
    # example usage:
    # python3 -m sparse_coding.collect_acvts.collect_activations --max_num_entries 100 --split val --sql_path {outpath}/val_dbl.sql
    fire.Fire(get_activations)
