import torch
import warnings
import fire
import heapq
from functools import partial
from torch import nn
from collections import defaultdict
import pickle
import os

from global_utils import device
from sparse_coding.local_utils import get_features
from global_whisper_utils import (
    WhisperMelsDataset,
    LibriSpeechDataset,
    WhisperActivationCache,
)
from whisper_repo.tokenizer import get_tokenizer

tokenizer = get_tokenizer(multilingual=True)

MODEL_NAME = "tiny"
OUT_DIR = f"/exp/ellenar/sparse_coding/whisper_activations_{MODEL_NAME}_LibriSpeech"

"""
Find the max activating dataset examples for every feature in dictionary
Features are either:
- 'learnt' by sparse coding, 
- the neuron basis 
- a random orthogonal basis
"""


def get_feature_activations(
    out_path: str,
    split: str,  # train or val
    max_num_entries: int = 0,
    sql_path: str = None,
    activations_to_cache: list = [
        "decoder.token_embedding",
    ],
    batch_size=100,
    chk_path: str = "/exp/ellenar/sparse_coding/train/20230726_whisper_tiny_decoder.token_embedding_n_dict_components_2000_l1_alpha_5e-5_LibriSpeech/models/checkpoint.pt.step5000",
    feature_type: str = "learnt",
    dataset_name: str = "am",
    k: int = 10,
):
    """
    Compute and save the top k activating fragments for each feature
    """
    if not device == "cuda":
        warnings.warn("This is much faster if you run it on a GPU")

    featurewise_max_activating_fragments = defaultdict(
        partial(MaxActivatingFragments, k=k)
    )  # key=feature_idx, value=max_activating_fragments
    if dataset_name == "am":
        dataset = WhisperMelsDataset(
            max_num_entries=max_num_entries, split=split, sql_path=sql_path
        )
    else:
        assert dataset_name == "LibriSpeech", "dataset should be am or LibriSpeech"
        if split == "train":
            dataset = LibriSpeechDataset()
        elif split == "val":
            dataset = LibriSpeechDataset(url="dev-clean")
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
        print(f"Processed batch {batch_idx} of size {batch_size}")
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
    """
    Class to store k max activating fragements for each feature.
    Uses heapq to push new activations onto the stack and pop the smallest off the top
    """

    def __init__(self, k):
        super().__init__()
        self.metadata = (
            {}
        )  # keys are activation scores, values are audio paths and transcripts
        self.k = k
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
        if len(self) == self.k:
            min_activation = heapq.heappushpop(self, max_activation)
            try:
                del self.metadata[min_activation]
            except:
                KeyError
        else:
            heapq.heappush(self, max_activation)


if __name__ == "__main__":
    fire.Fire(get_feature_activations)
