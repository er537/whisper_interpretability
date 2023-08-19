import heapq
import pathlib
import pickle
from collections import defaultdict
from functools import partial
from typing import Optional

import fire
import torch
import torch.nn as nn
import whisper
from audiolize.dataprep.segmented_dataset import SegmentedLibriSpeechDataset
from global_utils import device
from sparse_coding.local_utils import get_features
from whisper_repo.tokenizer import get_tokenizer

tokenizer = get_tokenizer(multilingual=True)

MODEL_NAME = "tiny"

"""
Find the max activating dataset examples from a dataset of words for a single logit
"""

WORD_LIST = [" the", " I", " in", " to", " is", " that", " when", " on", " at"]


def get_max_activating_fragments(
    out_path: str,
    word_list: Optional[list] = WORD_LIST,
    max_num_samples: int = 100,
    batch_size: int = 100,
    k: int = 10,
    autoencoder_path: Optional[str] = None,
    layer_name: str = "decoder.ln",
    feature_type: str = "learnt",  # "learnt", "neuron_basis" or "rand_orth"
    audio_chunk_size: int = 8000,  # num audio samples per fragment
):
    dataset = SegmentedLibriSpeechDataset(chunk_size=audio_chunk_size)
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size))
    options = whisper.DecodingOptions(
        without_timestamps=False, fp16=(device == "cuda"), language="en"
    )
    model_size = "tiny"
    model = whisper.load_model(model_size)
    max_activating_fragments = defaultdict(partial(MaxActivatingFragments, k))
    global actvs  # store the activations from the final layer norm
    actvs = {}

    hook = register_hooks(model, layer_name=layer_name)
    if layer_name == "decoder.ln":
        for name, param in model.named_parameters():
            if name == "decoder.token_embedding.weight":
                unembed = param.permute(1, 0)
    else:
        assert (
            autoencoder_path is not None
        ), "Must provide autoencoder path to work out the shape of the features"
        encoder_weight, encoder_bias = get_features(
            feature_type=feature_type, chk_path=autoencoder_path
        )

    n_samples = 0
    while n_samples < max_num_samples:
        mels, audio_paths, start_idxs = next(dataloader)
        mels = mels.to(device)
        outputs = model.decode(mels, options=options)
        actvs = actvs[layer_name]
        if layer_name == "decoder.ln":
            max_activating_fragments = maybe_add_logit_to_max_activating_fragments(
                actvs,
                outputs,
                audio_paths,
                start_idxs,
                word_list,
                max_activating_fragments,
                unembed,
            )
        else:
            max_activating_fragments = maybe_add_feature_to_max_activating_fragments(
                actvs,
                outputs,
                audio_paths,
                start_idxs,
                max_activating_fragments,
                encoder_weight,
                encoder_bias,
                audio_chunk_size,
            )
        actvs = {}  # reset activations
        n_samples += batch_size

    hook.remove()
    save_max_activating_fragments(
        max_activating_fragments, f"{out_path}/{layer_name}_{max_num_samples}/{feature_type}"
    )


def maybe_add_feature_to_max_activating_fragments(
    actvs,
    output,
    audio_paths,
    start_idxs,
    max_activating_fragments,
    encoder_weight,
    encoder_bias,
    audio_chunk_size,
):
    batch_feature_activations = nn.ReLU()(
        (actvs.to(device).float() @ encoder_weight) + encoder_bias
    ).cpu()
    num_embeds = int(
        audio_chunk_size / (160 * 2)
    )  # mels are 160 audio samples long, embeds are downsampled by 2
    batch_feature_activations = batch_feature_activations[:, :num_embeds, :]
    for i, audio_path in enumerate(audio_paths):
        tokens = output[i].tokens[
            :5
        ]  # only consider the first 5 tokens to remove effect of hallucinations
        text = tokenizer.decode(tokens)
        for feature_idx in range(batch_feature_activations.shape[-1]):
            # for every feature_activation,
            # maybe add it to the list of max activating fragments
            feature_activation_scores = batch_feature_activations[i, :, feature_idx]
            max_activation, argmax_idx = torch.max(feature_activation_scores, dim=0)
            max_activating_fragments[feature_idx].heappushpop(
                max_activation.item(), audio_path, start_idxs[i], text, argmax_idx.item()
            )
    return max_activating_fragments


def maybe_add_logit_to_max_activating_fragments(
    actvs, output, audio_paths, start_idxs, word_list, max_activating_fragments, unembed
):
    logits = torch.einsum("bld,dv->blv", actvs, unembed)[
        :, :5, :
    ]  # only consider the first 5 logits to remove effect of hallucinations
    actvs = {}
    for i, audio_path in enumerate(audio_paths):
        tokens = output[i].tokens[
            :5
        ]  # only consider the first 5 tokens to remove effect of hallucinations
        text = tokenizer.decode(tokens)
        # go through logits, map to token and check if in tokens_list
        for j in range(logits[i].shape[0]):
            token = torch.argmax(logits[i][j])
            word = tokenizer.decode([token])
            if word in word_list:
                print(f"Word '{word}' found")
                max_activating_fragments[word].heappushpop(
                    torch.max(logits[i][j]).item(), audio_path, start_idxs[i], text, j
                )
    return max_activating_fragments


def save_max_activating_fragments(featurewise_max_activating_fragments: dict, out_path: str):
    for feature_id, metadata in featurewise_max_activating_fragments.items():
        print(f"Saving {feature_id} fragments")
        pathlib.Path.mkdir(pathlib.Path(out_path), exist_ok=True, parents=True)
        with open(f"{out_path}/{str(feature_id).strip(' ')}.pkl", "wb") as outfile:
            pickle.dump(metadata, outfile)


class MaxActivatingFragments(list):
    """
    Class to store k fragments with the max logit/feature activation for each token/feature.
    Use heapq to push new logit onto the stack and pop the smallest off the top
    """

    def __init__(self, k):
        super().__init__()
        self.metadata = {}  # keys are logits, values are audio paths and start_idxs
        self.k = k

    def heappushpop(self, max_activation, audio_path, start_idx, tokens, argmax_idx):
        self.metadata[max_activation] = [audio_path, start_idx, tokens, argmax_idx]
        if len(self) == self.k:
            min_logit = heapq.heappushpop(self, max_activation)
            try:
                # incase of repeated logit values having previously been deleted from the dict
                del self.metadata[min_logit]
            except KeyError:
                pass
        else:
            heapq.heappush(self, max_activation)


def get_activations_hook(name):
    def hook_fn(mod, input_, output_):
        activations = input_[0].detach().float()
        if "decoder" in name:
            # hack to remove sos tokens in the decoder
            if activations.shape[1] > 1:
                return
        if name in actvs:
            actvs[name] = torch.cat((actvs[name], activations), dim=1)
        else:
            actvs[name] = activations

    return hook_fn


def register_hooks(model, layer_name="decoder.ln"):
    for name, module in model.named_modules():
        if name == layer_name:
            hook = module.register_forward_hook(get_activations_hook(name))
    return hook


if __name__ == "__main__":
    fire.Fire(get_max_activating_fragments)
