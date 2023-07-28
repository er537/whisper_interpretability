import os
import pathlib
import warnings
from typing import Optional

import fire
import torch

from global_utils import device, todays_date
from global_whisper_utils import LibriSpeechDataset, WhisperActivationCache, WhisperMelsDataset

MODEL_NAME = "tiny"
OUT_DIR = f"/exp/ellenar/sparse_coding/whisper_activations_{MODEL_NAME}"

"""
Script to collect activations from an intermediate layer of Whisper
max_num_entries: Number of audio files (sammples) to use
sql_path: path to store sql database used by dataloader
layer_to_cache: named module from whisper.named_modules() to cache activations from
"""


def get_activations(
    max_num_entries: Optional[int],  # only required for am dataset
    sql_path: Optional[str],  # only required for am dataset
    split: str = "train",  # train or val
    layer_to_cache: str = "decoder.token_embedding",
    dataset_name: str = "am",
):
    if not device == "cuda":
        warnings.warn("This is much faster if you run it on a GPU")
    actv_cache = WhisperActivationCache(
        model_name=MODEL_NAME, activations_to_cache=[layer_to_cache]
    )
    if dataset_name == "am":
        assert sql_path is not None, "Please provide an SQL path"
        assert max_num_entries is not None, "Please provide the max_num_entries"
        dataset = WhisperMelsDataset(
            max_num_entries=max_num_entries, split=split, sql_path=sql_path
        )
    else:
        assert dataset_name == "LibriSpeech", "dataset should be either am or LibriSpeech"
        if split == "train":
            dataset = LibriSpeechDataset(url="train-other-500")
        elif split == "val":
            dataset = LibriSpeechDataset(url="dev-clean")
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=100))
    for data, lang_codes, audio_paths in dataloader:
        actv_cache.reset_state()
        actv_cache.forward(data.to(device))
        for layer in layer_to_cache:
            layer_actvs = actv_cache.activations[f"{layer}"].to("cpu")
            for i, (lang_code, audio_path) in enumerate(zip(lang_codes, audio_paths)):
                out_path_ext = get_out_path_ext(audio_path=audio_path, lang_code=lang_code)
                pathlib.Path(f"{OUT_DIR}_{dataset_name}/{split}/{layer}").mkdir(
                    parents=True, exist_ok=True
                )
                torch.save(
                    layer_actvs[i],
                    f"{OUT_DIR}_{dataset_name}/{split}/{layer}/{todays_date}_{out_path_ext}.pt",
                )


def get_out_path_ext(audio_path, lang_code):
    """
    Use the dirname, basename and lang_code to form a unique file path extension
    """
    dirname = os.path.dirname(audio_path).split("/")[-1]
    basename = os.path.basename(audio_path)
    return f"{dirname}_{basename}_{lang_code}"


if __name__ == "__main__":
    fire.Fire(get_activations)
