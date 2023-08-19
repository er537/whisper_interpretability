# %%
import random

import torch
import whisper
from global_whisper_utils import LibriSpeechDataset, device, load_audio


class SegmentedLibriSpeechDataset(torch.utils.data.IterableDataset):
    def __init__(self, chunk_size=8000):
        super().__init__()
        self.LibriSpeech = iter(LibriSpeechDataset(return_mels=False))
        self.chunk_size = chunk_size  # 8000 audio samples -> 0.5s

    def __iter__(self):
        for audio_path in self.LibriSpeech:
            audio = load_audio(audio_path)
            start_idx = random.randint(0, len(audio) - self.chunk_size)
            audio_seg = audio[start_idx : start_idx + self.chunk_size]
            audio_seg = whisper.pad_or_trim(audio_seg.flatten())
            mels = torch.tensor(whisper.log_mel_spectrogram(audio_seg)).to(device)
            yield mels, audio_path, start_idx

    def __len__(self):
        return len(self.LibriSpeech)
