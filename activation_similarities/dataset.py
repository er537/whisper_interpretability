import torchaudio
import torch
import os
import numpy as np
import whisper_repo

from global_utils import device, load_audio, trim_audio


class LibriSpeech(torch.utils.data.Dataset):
    """
    A simple class to wrap LibriSpeech and trim/pad the audio to 30 seconds.
    It will drop the last few seconds of a very small portion of the utterances.
    """

    def __init__(self, split="test-clean", device=device):
        self.dataset = torchaudio.datasets.LIBRISPEECH(
            root=os.path.expanduser("~/.cache"),
            url=split,
            download=True,
        )
        self.device = device

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        audio, sample_rate, text, _, _, _ = self.dataset[item]
        return audio, sample_rate, text, audio.shape[1]


class ClasswiseDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        audio_samples_per_class=48_0000,
        dblx_path="/data/artefacts/vad/top_4_no_header/filtered/filtered_train.dblx",
        class_labels=["SPEECH"],
        pad=True,  ## pad/trim for Whisper
    ):
        super().__init__()
        self.data = {class_label: torch.empty(0) for class_label in class_labels}
        self.audio_samples_per_class = audio_samples_per_class
        self.samples_per_batch = 48_0000  # default window size used by Whisper
        self.pad = pad
        self.build_dataset(dblx_path, audio_samples_per_class, class_labels)

    def build_dataset(self, dblx_path, audio_samples_per_class, class_labels):
        """
        Fill self.samples with audio_samples_per_class
        """
        # TODO: this is kinda inefficient as we parse every audio file even if that class if already 'full'
        with open(dblx_path, "r") as f:
            completed = False
            while not completed:
                line = f.readline()
                _, audio_path, start_time, end_time, label = line.split(" ")
                label = label.strip("\n")

                if label not in class_labels:
                    continue

                if self.data[label].shape[0] > audio_samples_per_class:
                    # this class already full, continue until all other classes are full
                    continue

                audio = load_audio(audio_path)
                trimmed_audio = trim_audio(audio, float(start_time), float(end_time))

                if self.data[label].shape[0] == 0:
                    self.data[label] = trimmed_audio
                else:
                    self.data[label] = np.concatenate(
                        (self.data[label], trimmed_audio), axis=0
                    )

                completed = all(
                    frames.shape[0] > audio_samples_per_class
                    for frames in self.data.values()
                )
        for label in self.data.keys():
            self.data[label] = self.data[label][: self.audio_samples_per_class]
            print(f"Class label:{label}, Number of Samples:{self.data[label].shape[0]}")

    def __iter__(self):
        for label, samples in self.data.items():
            samples_yielded = 0
            while samples_yielded < self.audio_samples_per_class:
                audio = samples[
                    samples_yielded : samples_yielded + self.samples_per_batch
                ]
                if self.pad:
                    audio = whisper_repo.pad_or_trim(audio.flatten())
                mels = torch.tensor(whisper_repo.log_mel_spectrogram(audio)).to(device)

                yield mels, label
                samples_yielded += self.samples_per_batch
