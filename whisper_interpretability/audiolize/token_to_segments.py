import heapq
import os
import pickle
import warnings
from collections import defaultdict
from functools import partial

import fire
import torch
from dataprep.segmented_dataset import SegmentedDataset
from global_utils import device
from torch import nn
from whisper_repo.tokenizer import get_tokenizer

tokenizer = get_tokenizer(multilingual=True)

MODEL_NAME = "tiny"

"""
Find the max activating dataset examples from a dataset of words for a single logit
"""


def get_logit_activations(
    out_path: str, split: str, num_samples: int = 0, batch_size=100, k: int = 10  # train or val
):
    dataset = SegmentedDataset()
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=batch_size))
    for batch in dataloader:
        pass


if __name__ == "__main__":
    fire.Fire(get_logit_activations)
