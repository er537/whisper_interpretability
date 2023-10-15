import heapq
import os
import warnings
from collections import defaultdict
from functools import partial
import pandas as pd

import fire
import torch
from global_utils import device
from global_whisper_utils import (
    LibriSpeechDataset,
    WhisperActivationCache,
)
from sparse_coding.local_utils import get_features
from torch import nn
from whisper_repo.tokenizer import get_tokenizer

tokenizer = get_tokenizer(multilingual=True)

MODEL_NAME = "tiny"

"""
Find the max activating dataset examples for every feature in dictionary
Features are either:
- "learnt" by sparse coding,
- "neuron_basis"
- "rand_orth"
"""


def get_feature_activations(
    out_dir: str,
    layer_name: str = "encoder.blocks.3.mlp.0",
    num_samples_to_process: int = 10_000,
    num_samples_to_collect: int = 10,
    batch_size=50,
    chk_path: str = "/exp/ellenar/sparse_coding/train/20230801_whisper_tiny_decoder.blocks.2.mlp.0_n_dict_components_400_recon_alpha_1e5_LibriSpeech/models/checkpoint.pt.step10900",  # only required for learnt dictionary features
    d_model: int = 1536, # only required for neuron_basis and rand_orth, 1536 for mlp, 384 for res stream
    feature_type: str = "neuron_basis",
):
    """
    Compute and save the top `num_samples_to_collect` activating fragments for each feature
    """
    if not device == "cuda":
        warnings.warn("This is much faster if you run it on a GPU")
    if feature_type == "learnt":
        assert chk_path is not None, "Must provide a checkpoint path for learnt features"
    else:
        assert d_model is not None, "Must provide a d_model for neuron_basis or rand_orth features"

    featurewise_max_activating_fragments = defaultdict(
        partial(MaxActivatingFragments, num_samples_to_collect=num_samples_to_collect)
    )  # key=feature_idx, value=max_activating_fragments
    dataset = LibriSpeechDataset(return_mels=True, url="train-other-500")
    actv_cache = WhisperActivationCache(
        model_name=MODEL_NAME, activations_to_cache=[layer_name]
    )
    encoder_weight, encoder_bias = get_features(feature_type=feature_type, chk_path=chk_path, d_model=d_model)
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size))

    i = 0
    for batch_idx, (data, lang_codes, audio_paths) in enumerate(dataloader):
        if i > num_samples_to_process:
            break
        actv_cache.reset_state()
        output = actv_cache.forward(data.to(device))
        layer_actvs = actv_cache.activations[f"{layer_name}"]
        batch_feature_activations = nn.ReLU()(
            (layer_actvs.to(device).float() @ encoder_weight) + encoder_bias
        ).cpu()
        for j, audio_path in enumerate(audio_paths):
            tokens = output[j].tokens
            for feature_idx in range(batch_feature_activations.shape[-1]):
                # for every feature_activation,
                # maybe add it to the list of max activating fragments
                feature_activation_scores = batch_feature_activations[j, :, feature_idx]
                featurewise_max_activating_fragments[feature_idx].heappushpop(
                    feature_activation_scores, audio_path, tokens
                )
        print(f"Processed batch {batch_idx} of {num_samples_to_process // batch_size}")
        i += batch_size

    df = dict_to_dataframe(featurewise_max_activating_fragments)
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)
    df.to_csv(f"{out_dir}/{layer_name}_{feature_type}.csv", index=False)

def dict_to_dataframe(featurewise_max_activating_fragments: dict):
    """
    Convert the dictionary of MaxActivatingFragments to a dataframe using feature_idx as the index
    """
    df = pd.DataFrame()
    for feature_idx, max_activating_fragments in featurewise_max_activating_fragments.items():
        all_activation_scores = []
        all_audio_paths = []
        all_transcripts = []
        for activation_score, metadata in max_activating_fragments.metadata.items():
            audio_path, transcript = metadata
            all_activation_scores.append(activation_score)
            all_audio_paths.append(audio_path)
            all_transcripts.append(transcript)

        temp_df = pd.DataFrame(
            {
                "activation_scores": [all_activation_scores],
                "audio_paths": [all_audio_paths],
                "transcripts": [all_transcripts],
                "mean_activation": max_activating_fragments.mean_activation,
            },
            index=[feature_idx]
        )
        df = pd.concat([df, temp_df])

    if len(df) == 0:
        print(featurewise_max_activating_fragments)
        raise ValueError("No max activating fragments found")
    return df

class MaxActivatingFragments(list):
    """
    Class to store `num_samples_to_collect` max activating fragments for a given feature
    Uses heapq to push new activations onto the stack and pop the smallest off the top
    """

    def __init__(self, num_samples_to_collect):
        super().__init__()
        self.metadata = {}  # keys are activation scores, values are audio paths and transcripts
        self.num_samples_to_collect = num_samples_to_collect
        self.mean_activation = 0
        self.num_activations = 0  # used to calculate running mean

    def heappushpop(self, activations, audio_path, tokens):
        max_activation = torch.max(activations).item()
        self.mean_activation = (
            (self.mean_activation * self.num_activations) + torch.mean(activations).item()
        ) / (self.num_activations + 1)
        self.num_activations += 1
        transcript = tokenizer.decode(tokens)
        self.metadata[max_activation] = [audio_path, transcript]
        if len(self) == self.num_samples_to_collect:
            min_activation = heapq.heappushpop(self, max_activation)
            try:
                del self.metadata[min_activation]
            except KeyError:
                pass
        else:
            heapq.heappush(self, max_activation)


if __name__ == "__main__":
    fire.Fire(get_feature_activations)
