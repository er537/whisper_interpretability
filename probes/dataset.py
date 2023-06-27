import torch
import whisper
import math
import random

from utils import device, load_audio, trim_audio


class VADDataset(torch.utils.data.IterableDataset):
    def __init__(
        self,
        dblx_path="/home/ellenar/testing_vad.dblx",
        class_labels=["SPEECH", "NON_SPEECH"],
        pad=True,  ## pad/trim for Whisper
        max_audio_samples=math.inf,
        samples_per_batch=480_000,
        batch_size=10,
    ):
        super().__init__()
        self.dblx = open(dblx_path, "r")
        self.datasets = {class_label: [] for class_label in class_labels}
        self.samples_per_batch = 48_0000  # default window size used by Whisper
        self.pad = pad
        self.max_audio_samples = max_audio_samples
        self.batch_size = batch_size
        self.samples_per_batch = samples_per_batch
        self.class_labels = class_labels

    def _get_mels(self, raw_audio):
        if self.pad:
            raw_audio = whisper.pad_or_trim(raw_audio.flatten())
        mels = torch.tensor(whisper.log_mel_spectrogram(raw_audio)).to(device)
        return mels

    def __iter__(self):
        """
        Keep a rolling buffer (self.datasets) of audio samples for each class, yield once we have a sample of the required class
        """
        chosen_dataclass = random.sample(self.class_labels, 1)[0]
        while True:
            line = self.dblx.readline()
            if not line:
                break

            _, audio_path, start_time, end_time, label = line.split(" ")
            label = label.strip("\n")
            if label not in self.class_labels:
                continue

            audio = load_audio(audio_path)
            trimmed_audio = torch.tensor(
                trim_audio(audio, float(start_time), float(end_time))
            )
            self.datasets[label].append(trimmed_audio)

            if label == chosen_dataclass:
                audio = self.datasets[label].pop(0)
                mels = self._get_mels(audio)
                labels = torch.tensor(self.class_labels.index(label)).repeat(
                    mels.shape[1]
                )
                yield mels, labels
                chosen_dataclass = random.sample(self.class_labels, 1)[0]  # new class

        # exhaust remaining buffers
        incomplete = {label: True for label in self.class_labels}
        while any(incomplete.values()):
            for label in self.class_labels:
                if len(self.datasets[label]) > 0:
                    audio = self.datasets[label].pop(0)
                    num_non_padded_frames = len(audio) // 160
                    mels = self._get_mels(audio)
                    labels = torch.tensor(self.class_labels.index(label)).repeat(
                        num_non_padded_frames
                    )
                    pad_frames = mels.shape[1] - num_non_padded_frames
                    labels = torch.nn.functional.pad(
                        labels, (0, pad_frames), "constant", -1
                    )
                    yield mels, labels
                else:
                    incomplete[label] = False
