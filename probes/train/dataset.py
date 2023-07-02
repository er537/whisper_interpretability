import torch
import whisper_repo
import sqlite3 as sqlite
import warnings
from torch.nn.utils.rnn import pad_sequence

warnings.filterwarnings(
    action="ignore", category=UserWarning
)  # whisper.log_mel_spectrogram generates a verbose warning


from utils import load_audio, trim_audio


def collate_fn(batch):
    data = [x[0].permute(1, 0) for x in batch]
    labels = [x[1] for x in batch]
    data = pad_sequence(data, batch_first=True, padding_value=-1)
    labels = pad_sequence(labels, batch_first=True, padding_value=-1)
    return data.permute(0, 2, 1), labels


class MultiClassDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        num_entries,
        sql_path="/home/ellenar/probes/fr_de_train.sql",
        class_labels=["fr", "de"],
        pad=True,  ## pad/trim for Whisper
        audio_ret_type="mfcc",  # mfcc or waveform
    ):
        super().__init__()
        self.conn = sqlite.connect(sql_path)
        self.length = num_entries
        self.pad = pad
        self.class_labels = class_labels
        self.audio_ret_type = audio_ret_type

    def _get_mels(self, raw_audio):
        if self.pad:
            padded_audio = whisper_repo.pad_or_trim(raw_audio.flatten())
        else:
            padded_audio = raw_audio
        non_padded_frac = min(1.0, raw_audio.shape[0] / padded_audio.shape[0])
        mels = torch.tensor(whisper_repo.log_mel_spectrogram(padded_audio))
        return mels, non_padded_frac

    def get_size_of_db(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM data")
        return len(cur.fetchall())

    def __getitem__(self, idx):
        audio_path, label, start_time, end_time = self.conn.execute(
            "SELECT audio_path, label, start_time, end_time FROM data WHERE key = ? LIMIT 1",
            (idx,),
        ).fetchone()
        assert label in self.class_labels, f"Unknown label {label}"
        label = torch.tensor(self.class_labels.index(label))  # label->idx
        audio = load_audio(audio_path)
        trimmed_audio = trim_audio(audio, float(start_time), float(end_time))
        if self.audio_ret_type == "mfcc":
            x, non_padded_frac = self._get_mels(torch.tensor(trimmed_audio))
            labels = label.repeat(
                int(non_padded_frac * x.shape[1])
            )  # upsample to give the same no. of labels non padded mfccs
            labels = torch.cat(
                (labels, -1 * torch.ones(x.shape[1] - labels.shape[0])), dim=0
            )  # use -1 to denote padding frames
        elif self.audio_ret_type == "waveform":
            x = torch.tensor(trimmed_audio).unsqueeze(0)
            labels = label.repeat(x.shape[1])
        else:
            raise Exception(f"audio return type {self.audio_ret_type} unknown")
        return x, labels

    def __len__(self):
        return self.length
