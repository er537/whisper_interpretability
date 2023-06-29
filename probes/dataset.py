import torch
import whisper
import sqlite3 as sqlite
import warnings

warnings.filterwarnings(
    action="ignore", category=UserWarning
)  # whisper.log_mel_spectrogram generates a verbose warning


from utils import device, load_audio, trim_audio


class VADDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        sql_path="/home/ellenar/testing.sql",
        class_labels=["NON_SPEECH", "SPEECH"],
        pad=True,  ## pad/trim for Whisper
        num_entries=14037397,
    ):
        super().__init__()
        self.conn = sqlite.connect(sql_path)
        self.length = num_entries
        self.pad = pad
        self.class_labels = class_labels

    def _get_mels(self, raw_audio):
        if self.pad:
            raw_audio = whisper.pad_or_trim(raw_audio.flatten())
        mels = torch.tensor(whisper.log_mel_spectrogram(raw_audio)).to(device)
        return mels

    def get_size_of_db(self):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM data")
        return len(cur.fetchall())

    def __getitem__(self, idx):
        audio_path, label, start_time, end_time = self.conn.execute(
            "SELECT audio_path, label, start_time, end_time FROM data WHERE key = ? LIMIT 1",
            (idx,),
        ).fetchone()
        audio = load_audio(audio_path)
        trimmed_audio = torch.tensor(
            trim_audio(audio, float(start_time), float(end_time))
        )
        mels = self._get_mels(trimmed_audio)
        return mels, label

    def __len__(self):
        return self.length
