import logging
import os
import random
import sqlite3 as sqlite
import warnings
from subprocess import CalledProcessError, run
from typing import Callable, Optional

import numpy as np
import torch
import torchaudio
import whisper
import whisper_repo
from global_utils import BaseActivationModule, device
from jaxtyping import Float
from torch import Tensor

warnings.filterwarnings(
    action="ignore", category=UserWarning
)  # whisper.log_mel_spectrogram generates a verbose warning

"""
A collection of functions specifically related to whisper commonly used throughout the repo
"""
EN_TRAIN_DBLX_DICT = {
    "/data/artefacts/am/en/v2023.03_full_reseg/data_train/train.dblx": "en",
}
EN_VAL_DBLX_DICT = {
    "/data/artefacts/am/en/v2023.03_full_reseg/data_train/valid.dblx": "en",
}

MULTILANG_DBLX_DICT = {
    "/data/artefacts/am/de/v2023.02_q1rebuild/data_train/train.dblx": "de",
    "/data/artefacts/am/fr/new_normaliser/data_train/train.dblx": "fr",
    "/data/artefacts/am/es/v2023.03_q1rebuild/data_train/train.dblx": "es",
    "/data/artefacts/am/ru/v2023.02_q1rebuild/data_train/train.dblx": "ru",
    "/data/artefacts/am/en/v2023.03_full_reseg/data_train/train.dblx": "en",
}  # file_path, lang_code


def collate_fn(batch):
    data = torch.stack([x[0].permute(1, 0) for x in batch], dim=0)
    lang_codes = [x[1] for x in batch]
    audio_paths = [x[2] for x in batch]
    return data, lang_codes, audio_paths


def trim_audio(
    array: np.array,
    start_time: float,
    end_time: float,
    sample_rate: int = 16_000,
):
    """
    Trim the audio file base array to n_samples, as expected by the encoder.
    """
    start_frame = int(sample_rate * start_time)
    end_frame = int(sample_rate * end_time)

    return array[start_frame:end_frame]


def load_audio(file: str, sample_rate_hz: int = 16_000):
    """
    Taken from Whisper repo: https://github.com/openai/whisper/blob/main/whisper/audio.py

    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sample_rate_hz: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """

    # This launches a subprocess to decode audio while down-mixing
    # and resampling as necessary.  Requires the ffmpeg CLI in PATH.
    # fmt: off
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-threads", "0",
        "-i", file,
        "-f", "s16le",
        "-ac", "1",
        "-acodec", "pcm_s16le",
        "-ar", str(sample_rate_hz),
        "-"
    ]
    # fmt: on
    try:
        out = run(cmd, capture_output=True, check=True).stdout
    except CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def get_mels_from_audio_path(
    audio_path: str, start_time_s: Optional[float] = None, end_time_s: Optional[float] = None
):
    audio = whisper.load_audio(audio_path)
    if start_time_s is not None and end_time_s is not None:
        audio = trim_audio(audio, start_time_s, end_time_s)
    audio = whisper_repo.pad_or_trim(audio.flatten())
    mels = torch.tensor(whisper_repo.log_mel_spectrogram(audio)).to(device)
    return mels


def get_mels_from_dblx(dblx_path, num_samples):
    batch_mels = []
    with open(dblx_path, "r") as f:
        for _ in range(num_samples):
            line = f.readline().split(" ")
            audio_path = line[1]
            start_time = float(line[2])
            end_time = float(line[3])
            mels = get_mels_from_audio_path(audio_path, start_time, end_time)
            batch_mels.append(mels)
    return torch.stack(batch_mels, dim=0)


class WhisperMelsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        max_num_entries: int = 100,  # maximum number of audio samples in sql
        sql_path: str = "/exp/ellenar/sparse_coding/data/val_dbl.sql",
        split: str = "train",  # train or val
    ):
        """
        Generate mels from (potentially padded) 30s chunks of audio
        in the format expected as input by whisper
        We store the dataset in a saved sql for fast querying -
        Note - if the sql_path already exists we DO NOT rebuild it
        """
        super().__init__()
        self.max_num_entries = max_num_entries
        self.sql_path = sql_path
        self.dblx_dict = EN_TRAIN_DBLX_DICT if split == "train" else EN_VAL_DBLX_DICT
        if not os.path.exists(sql_path):
            print("Building sql")
            self._build_sql()
        else:
            self.conn = sqlite.connect(sql_path)
            db_size = self.get_size_of_db()
            print(f"Total entries in sql={db_size}")

    def _build_sql(self):
        """
        Builds an sql from a text file containing the paths to audio files.
        The sql will contain an equal split from each class (max_entries//n_classes)
        """
        self._init_sql()
        total_entries = 0
        entries_per_class = self.max_num_entries // len(self.dblx_dict)
        with sqlite.connect(f"{self.sql_path}") as conn:
            cur = conn.cursor()
            for dblx_path, lang in self.dblx_dict.items():
                with open(dblx_path, "r") as f:
                    entries_in_class = 0
                    while True:
                        line = f.readline()
                        if not line:
                            raise Exception(
                                f"Please provide dblx's with at least {entries_per_class} entries"
                            )
                        id, audio_path, start_time, end_time, *_ = line.split(" ")
                        cur.execute(
                            "INSERT INTO data VALUES(?,?,?,?,?);",
                            (
                                f"{total_entries}",
                                audio_path,
                                lang,
                                float(start_time),
                                float(end_time),
                            ),
                        )
                        total_entries += 1
                        entries_in_class += 1
                        if entries_in_class == entries_per_class:
                            break
        assert total_entries <= self.max_num_entries
        self.conn = sqlite.connect(self.sql_path)
        print(f"Total entries in sql={total_entries}")

    def _init_sql(self):
        dir_path = os.path.dirname(self.sql_path)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        with sqlite.connect(f"{self.sql_path}") as conn:
            create_str = (
                "CREATE TABLE IF NOT EXISTS "
                "data(key TEXT PRIMARY KEY, audio_path TEXT, label TEXT, start_time FLOAT, end_time FLOAT)"  # noqa E501
            )
            logging.info("Generating SQLITE db from utterances")
            cur = conn.cursor()
            cur.execute(create_str)

    def get_size_of_db(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM data")
        return len(cur.fetchall())

    def __getitem__(self, idx):
        audio_path, lang_code, start_time, end_time = self.conn.execute(
            "SELECT audio_path, label, start_time, end_time FROM data WHERE key = ? LIMIT 1",
            (idx,),
        ).fetchone()
        mels = get_mels_from_audio_path(audio_path, float(start_time), float(end_time))
        return mels, lang_code, audio_path

    def __len__(self):
        results = self.conn.execute("SELECT * from data").fetchall()
        return len(results)


class LibriSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, root="/exp/ellenar/LibriSpeech", url="dev-clean", return_mels=False):
        super().__init__()
        if not os.path.exists(f"{root}/LibriSpeech/{url}"):
            download = True
        else:
            download = False
        try:
            self.dataset = torchaudio.datasets.LIBRISPEECH(download=download, url=url, root=root)
        except RuntimeError:
            print("Downloading dataset")
            self.dataset = torchaudio.datasets.LIBRISPEECH(download=download, url=url, root=root)
        self.root = root
        self.return_mels = return_mels

    def __getitem__(self, idx):
        idx = int(random.random() * len(self.dataset))  # randomly sample
        audio_path, sample_rate, transcript, *_ = self.dataset.get_metadata(idx)
        if self.return_mels:
            mels = get_mels_from_audio_path(audio_path=f"{self.root}/LibriSpeech/{audio_path}")
            return mels, "en", f"{self.root}/LibriSpeech/{audio_path}"
        else:
            return f"{self.root}/LibriSpeech/{audio_path}"

    def __len__(self):
        return len(self.dataset)


class WhisperActivationCache(BaseActivationModule):
    """
    Use hooks in BaseActivationModule to cache intermediate activations while running forward pass
    """

    def __init__(
        self,
        hook_fn: Optional[Callable] = None,
        model: Optional[torch.nn.Module] = None,  # if model not provided load whisper model by name
        model_name: str = "tiny",
        activations_to_cache: list = ["encoder.blocks.0"],  # pass "all" to cache all activations
    ):
        self.model = (
            model.to(device) if model is not None else whisper.load_model(model_name).to(device)
        )
        self.activations_to_cache = activations_to_cache
        self.named_modules = list({name: mod for name, mod in self.model.named_modules()})
        super().__init__(
            model=self.model,
            activations_to_cache=self.activations_to_cache,
            hook_fn=hook_fn,
        )

    def custom_forward(
        self, model: torch.nn.Module, mels: Float[Tensor, "bsz seq_len n_mels"]
    ):  # noqa: F821
        options = whisper.DecodingOptions(without_timestamps=False, fp16=(device == "cuda"))
        output = model.decode(mels, options)
        return output

    def _get_caching_hook(self, name):
        # custom caching function for whisper
        def hook(module, input, output):
            if "decoder" in name:
                # we don't cache the first activations that correspond to the sos/lang tokens
                if output.shape[1] > 1:
                    del self.activations[f"{name}"]
                    return
            output_ = output.detach().cpu()
            if name in self.activations:
                self.activations[f"{name}"] = torch.cat(
                    (self.activations[f"{name}"], output_), dim=1
                )
            else:
                self.activations[f"{name}"] = output_

        return hook
