import logging
import os
import sqlite3 as sqlite
from pathlib import Path

import torch
import whisper
from global_utils import device
from torch.nn.utils.rnn import pad_sequence
from whisper_repo.tokenizer import get_tokenizer

tokenizer = get_tokenizer(multilingual=False)

WORDS = [
    " high",
    " low",
    " hello",
    " hi",
    " computer",
    "Computer",
    " scream",
    " dream",
    " seat",
    " feet",
    " walk",
    " talk",
    " on",
    " in",
    " when",
    " where",
    " big",
    " small",
    " day",
    " night",
    " happy",
    " good",
    " up",
    " down",
    " book",
    " group",
    " day",
    " way",
    " light",
    " white",
    " fast",
    " first",
    " angry",
    " hungry",
    " old",
    " new",
]


def collate_fn(batch):
    data = pad_sequence(batch, batch_first=True, padding_value=-1)
    return data


class ActivationDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dbl_path: str = "/exp/ellenar/sparse_coding/whisper_activations_tiny/val/decoder.blocks.3_actvs/val.dbl",
        rebuild_sql: bool = False,
        rank: int = 0,  # only build sql on rank 0
    ):
        """
        Takes a file (dbl) containing a list of file paths to saved out activations,
        builds an sql and randomly samples them
        """
        super().__init__()
        self.sql_path = f"{dbl_path.strip('.dbl')}.sql"
        if not os.path.exists(self.sql_path) or rebuild_sql:
            if rank == 0:
                self._build_sql(dbl_path)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        self.conn = sqlite.connect(self.sql_path)

    def _build_sql(self, dbl_path):
        Path(self.sql_path).unlink(missing_ok=True)
        self._init_sql()
        total_entries = 0
        with sqlite.connect(f"{self.sql_path}") as conn:
            cur = conn.cursor()
            with open(dbl_path, "r") as f:
                while True:
                    activations_path = f.readline().strip("\n")
                    if not activations_path:
                        break
                    cur.execute(
                        "INSERT INTO data VALUES(?,?);",
                        (f"{total_entries}", activations_path),
                    )
                    total_entries += 1
        print(f"Total entries in sql={total_entries}")

    def _init_sql(self):
        dir_path = os.path.dirname(self.sql_path)
        if not os.path.isdir(dir_path):
            os.mkdir(dir_path)
        with sqlite.connect(f"{self.sql_path}") as conn:
            create_str = (
                "CREATE TABLE IF NOT EXISTS " "data(key TEXT PRIMARY KEY, activations_path TEXT)"
            )
            logging.info("Generating SQLITE db from utterances")
            cur = conn.cursor()
            cur.execute(create_str)

    def __getitem__(self, idx):
        activations_path = self.conn.execute(
            "SELECT activations_path FROM data WHERE key = ? LIMIT 1",
            (idx,),
        ).fetchone()
        activations = torch.load(activations_path[0])
        return activations

    def __len__(self):
        results = self.conn.execute("SELECT * from data").fetchall()
        return len(results)


class TokenEmbeddingDataset(torch.utils.data.Dataset):
    def __init__(self, words: list = WORDS, model_name: str = "tiny"):
        super().__init__()
        model = whisper.load_model(model_name)
        for name, param in model.named_parameters():
            if name == "decoder.token_embedding.weight":
                self.embed_layer = param.to(device)
        self.words = words
        self.token_embeddings = []
        for word in words:
            token_idx = tokenizer.encode(word)
            self.token_embeddings.append(self.embed_layer[token_idx, :])

    def __getitem__(self, idx):
        embed = self.token_embeddings[idx].detach()
        return torch.nn.functional.normalize(embed, dim=1)

    def __len__(self):
        return len(self.token_embeddings)
