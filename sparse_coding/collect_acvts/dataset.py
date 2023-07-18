import torch
import whisper_repo
import sqlite3 as sqlite
import warnings
import logging
import os
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings(
    action="ignore", category=UserWarning
)  # whisper.log_mel_spectrogram generates a verbose warning


from util import load_audio, trim_audio, dict_hash

DBLX_DICT = {
    "/data/artefacts/am/de/v2023.02_q1rebuild/data_train/train.dblx": "de",
    "/data/artefacts/am/fr/new_normaliser/data_train/train.dblx": "fr",
    "/data/artefacts/am/es/v2023.03_q1rebuild/data_train/train.dblx": "es",
    "/data/artefacts/am/ru/v2023.02_q1rebuild/data_train/train.dblx": "ru",
    "/data/artefacts/am/en/v2023.03_full_reseg/data_train/train.dblx": "en",
}  # file_path, lang_code
dblx_hash = dict_hash(DBLX_DICT)


def init_sql(sql_path):
    dir_path = os.path.dirname(sql_path)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    with sqlite.connect(f"{sql_path}") as conn:
        create_str = (
            "CREATE TABLE IF NOT EXISTS "
            "data(key TEXT PRIMARY KEY, audio_path TEXT, label TEXT, start_time FLOAT, end_time FLOAT)"
        )
        logging.info("Generating SQLITE db from utterances")
        cur = conn.cursor()
        cur.execute(create_str)


def collate_fn(batch):
    data = torch.stack([x[0].permute(1, 0) for x in batch], dim=0)
    lang_codes = [x[1] for x in batch]
    audio_paths = [x[2] for x in batch]
    return data, lang_codes, audio_paths


class WhisperMelsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        max_num_entries: int = 10_000,
        sql_path: str = f"/exp/ellenar/sparse_coding/data/train_{dict_hash}.sql",
    ):
        super().__init__()
        self.max_num_entries = max_num_entries
        self.sql_path = sql_path
        if not os.path.exists(sql_path):
            self._build_sql()
        self.conn = sqlite.connect(sql_path)

    def _build_sql(self):
        init_sql(self.sql_path)
        total_entries = 0
        entries_per_class = self.max_num_entries // len(DBLX_DICT)
        with sqlite.connect(f"{self.sql_path}") as conn:
            cur = conn.cursor()
            for dblx_path, lang in DBLX_DICT.items():
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

    def _get_mels(self, raw_audio):
        padded_audio = whisper_repo.pad_or_trim(raw_audio.flatten())
        non_padded_frac = min(1.0, raw_audio.shape[0] / padded_audio.shape[0])
        mels = torch.tensor(whisper_repo.log_mel_spectrogram(padded_audio))
        return mels, non_padded_frac

    def get_size_of_db(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM data")
        return len(cur.fetchall())

    def __getitem__(self, idx):
        audio_path, lang_code, start_time, end_time = self.conn.execute(
            "SELECT audio_path, label, start_time, end_time FROM data WHERE key = ? LIMIT 1",
            (idx,),
        ).fetchone()
        audio = load_audio(audio_path)
        trimmed_audio = trim_audio(audio, float(start_time), float(end_time))
        x, non_padded_frac = self._get_mels(torch.tensor(trimmed_audio))
        return x, lang_code, audio_path

    def __len__(self):
        results = self.conn.execute("SELECT * from data").fetchall()
        return len(results)


def get_out_path(audio_path, lang_code):
    """
    Use the dirname, basename and lang_code to form a unique file path extension
    """
