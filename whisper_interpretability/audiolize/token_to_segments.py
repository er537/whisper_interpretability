import heapq
import os
import pickle
from collections import defaultdict
from functools import partial

import fire
import torch
import whisper
from dataprep.segmented_dataset import SegmentedLibriSpeechDataset
from global_utils import device
from whisper_repo.tokenizer import get_tokenizer

tokenizer = get_tokenizer(multilingual=True)

MODEL_NAME = "tiny"

"""
Find the max activating dataset examples from a dataset of words for a single logit
"""


def get_token_activations(
    out_path: str,
    tokens_list: list,
    max_num_samples: int = 0,
    batch_size=100,
    k: int = 10,
):
    dataset = SegmentedLibriSpeechDataset()
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size))
    options = whisper.DecodingOptions(
        without_timestamps=False, fp16=(device == "cuda"), language="en"
    )
    model_size = "tiny"
    model = whisper.load_model(model_size)
    n_samples = 0
    max_activating_fragments = defaultdict(partial(MaxActivatingFragments, k))
    global actvs  # store the activations from the final layer norm
    actvs = {}
    for name, mod in model.named_modules():
        if name == "decoder.ln":
            mod.register_forward_hook(get_logit_hook(name))

    for name, param in model.named_parameters():
        if name == "decoder.token_embedding.weight":
            unembed = param.permute(1, 0)

    while n_samples < max_num_samples:
        mels, audio_paths, start_idxs = next(dataloader)
        mels = mels.to(device)
        outputs = model.decode(mels, options=options)
        for output in outputs:
            logits = torch.einsum("bld,dv->blv", actvs["decoder.ln"], unembed)
            actvs = {}
            for i in len(output):
                audio_path = audio_paths[i]
                start_idx = start_idxs[i]
                # go through logits, map to token and check if in tokens_list
                for j in range(logits[i].shape[0]):
                    token = torch.argmax(logits[i][j])
                    if token in tokens_list:
                        max_activating_fragments[token].heappushpop(
                            torch.max(logits[i][j]).item(), audio_path, start_idx
                        )
        n_samples += batch_size

    save_max_activating_fragments(max_activating_fragments, out_path)


def save_max_activating_fragments(featurewise_max_activating_fragments: dict, out_path: str):
    for token, max_activating_fragments in featurewise_max_activating_fragments.items():
        print(f"Saving {token} fragments")
        if not os.path.exists(out_path):
            os.mkdir(out_path)
        with open(f"{out_path}/{token}.pkl", "wb") as outfile:
            pickle.dump(max_activating_fragments, outfile)


class MaxActivatingFragments(list):
    """
    Class to store k max activating fragments per logit.
    Use heapq to push new logit onto the stack and pop the smallest off the top
    """

    def __init__(self, k):
        super().__init__()
        self.metadata = {}  # keys are logits, values are audio paths and start_idxs
        self.k = k

    def heappushpop(self, logit, audio_path, start_idx):
        self.metadata[logit] = [audio_path, start_idx]
        if len(self) == self.k:
            min_logit = heapq.heappushpop(self, logit)
            try:
                # incase of repeated logit values having previously been deleted from the dict
                del self.metadata[min_logit]
            except KeyError:
                pass
        else:
            heapq.heappush(self, logit)


def get_logit_hook(name):
    def hook_fn(mod, input_, output_):
        logits = input_[0].detach().float()
        if logits.shape[1] > 1:
            return
        if name in actvs:
            actvs[name] = torch.cat((actvs[name], logits), dim=1)
        else:
            actvs[name] = logits

    return hook_fn


if __name__ == "__main__":
    fire.Fire(get_token_activations)
