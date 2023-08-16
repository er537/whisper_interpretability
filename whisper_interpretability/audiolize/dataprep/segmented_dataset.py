# %%
import json
import os

import torch
import whisper
import whisper_repo
from global_whisper_utils import device, trim_audio

splits_dir = "/data/artefacts/am_test/en/en-NSC/8356d08aa1761193ef0fe78462552d56/splits"
json_dir = (
    "/exp/ellenar/lang_wers/en/20230815/standard/wer/en-NSC/engine_output/decode_rescore/json"
)


def collate_fn(batch):
    mels, texts, wav_paths, start_times, end_times = zip(*batch)
    return mels, texts, wav_paths, start_times, end_times


class SegmentedDataset(torch.utils.data.IterableDataset):
    def __init__(self, splits_dir: str = splits_dir, json_dir: str = json_dir):
        super().__init__()
        self.splits_dir = splits_dir
        self.json_dir = json_dir

    def __iter__(self):
        for file in os.listdir(splits_dir):
            split = file.split(".")[0]
            with open(os.path.join(splits_dir, file), "r") as f:
                wav_path = f.readline().split(" ")[1]
                audio = whisper.load_audio(wav_path)
            with open(os.path.join(json_dir, f"decode_rescore.{split}.json"), "r") as f:
                json_data = json.load(f)
                results = json_data["results"]
                for i in range(len(results)):
                    type_, content, start_time, end_time = (
                        results[i]["type"],
                        results[i]["alternatives"][0]["content"],
                        results[i]["start_time"],
                        results[i]["end_time"],
                    )
                    if type_ == "word":
                        audio = trim_audio(audio, start_time, end_time)
                        audio_seg = whisper.pad_or_trim(audio)
                        mels = torch.tensor(whisper_repo.log_mel_spectrogram(audio_seg)).to(device)
                        yield mels, content, wav_path, start_time, end_time

    def __len__(self):
        return len(os.listdir(splits_dir))
