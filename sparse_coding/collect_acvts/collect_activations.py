import torch
import warnings
import fire
import pathlib
import os

from util import device
from utils.activation_caches import WhisperActivationCache
from sparse_coding.collect_acvts.dataset import WhisperMelsDataset, LibriSpeechDataset

MODEL_NAME = "tiny"
OUT_DIR = f"/exp/ellenar/sparse_coding/whisper_activations_{MODEL_NAME}"


def get_activations(
    max_num_entries: int = None,
    split: str = "train",  # train or val
    sql_path: str = None,
    activations_to_cache: list = ["decoder.token_embedding"],
    dataset_name: str = "am",
):
    if not device == "cuda":
        warnings.warn("This is much faster if you run it on a GPU")
    actv_cache = WhisperActivationCache(
        model_name=MODEL_NAME, activations_to_cache=activations_to_cache
    )
    if dataset_name == "am":
        dataset = WhisperMelsDataset(
            max_num_entries=max_num_entries, split=split, sql_path=sql_path
        )
    else:
        assert dataset_name == "LibriSpeech", "dataset should be am or LibriSpeech"
        dataset = LibriSpeechDataset()
    dataloader = iter(torch.utils.data.DataLoader(dataset, batch_size=100))
    for data, lang_codes, audio_paths in dataloader:
        actv_cache.reset_state()
        actv_cache.forward(data.to(device))
        for layer in activations_to_cache:
            layer_actvs = actv_cache.activations[f"{layer}"].to("cpu")
            for i, (lang_code, audio_path) in enumerate(zip(lang_codes, audio_paths)):
                out_path_ext = get_out_path_ext(
                    audio_path=audio_path, lang_code=lang_code
                )
                pathlib.Path(f"{OUT_DIR}_{dataset_name}/{split}/{layer}").mkdir(
                    parents=True, exist_ok=True
                )
                torch.save(
                    layer_actvs[i],
                    f"{OUT_DIR}_{dataset_name}/{split}/{layer}/{out_path_ext}.pt",
                )


def get_out_path_ext(audio_path, lang_code):
    """
    Use the dirname, basename and lang_code to form a unique file path extension
    """
    dirname = os.path.dirname(audio_path).split("/")[-1]
    basename = os.path.basename(audio_path)
    return f"{dirname}_{basename}_{lang_code}"


if __name__ == "__main__":
    # example usage:
    # python3 -m sparse_coding.collect_acvts.collect_activations --max_num_entries 100 --split val --sql_path {outpath}/val_dbl.sql
    fire.Fire(get_activations)
