# %%
import json
import os
from collections import defaultdict

import torch
import whisper
import whisper_repo
from global_whisper_utils import LibriSpeechDataset, device, load_audio

splits_dir = "/data/artefacts/am_test/en/en-NSC/8356d08aa1761193ef0fe78462552d56/splits"
json_dir = (
    "/exp/ellenar/lang_wers/en/20230815/standard/wer/en-NSC/engine_output/decode_rescore/json"
)


def collate_fn(batch):
    mels, texts, wav_paths, start_times, end_times = zip(*batch)
    return mels, texts, wav_paths, start_times, end_times


class SegmentedLibriSpeechDataset(torch.utils.data.IterableDataset):
    def __init__(self, splits_dir: str = splits_dir, json_dir: str = json_dir):
        super().__init__()
        self.LibriSpeech = iter(LibriSpeechDataset(return_mels=False))

    def __iter__(self):
        for audio_path in self.LibriSpeech:
            audio = load_audio(audio_path)
            # 8000 frame chunks -> 0.5s
            chunk_size = 8000
            for i in range(0, len(audio) - chunk_size, chunk_size):
                audio_seg = audio[i : i + chunk_size]
                audio_seg = whisper.pad_or_trim(audio_seg.flatten())
                mels = torch.tensor(whisper.log_mel_spectrogram(audio_seg)).to(device)
                yield mels, audio_path, i

    def __len__(self):
        return len(self.LibriSpeech)
