import torchaudio
import torch
import os
import numpy as np
from collections import defaultdict

from utils import device, load_audio, trim_audio

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
    def __init__(self, audio_samples_per_class=1000, dblx_path='/data/artefacts/vad/top_4_no_header/filtered/filtered_train.dblx', batch_size=10):
        super().__init__()
        self.build_dataset(dblx_path, audio_samples_per_class)
        self.data = defaultdict(list)
        self.build_dataset()
        self.max_frames_per_input = 48_0000
        self.audio_samples_per_class = audio_samples_per_class
        self.batch_size = batch_size
        
    def build_dataset(self, dblx_path, audio_samples_per_class):
        """
        Fill self.samples with audio_samples_per_class
        """
        # TODO: this is kinda inefficient as we parse every audio file even if that class if already 'full'
        with open(dblx_path, 'r') as f:
            while not completed:
                line = f.readline()
                _, audio_path, start_time, end_time, label = line.split(' ')
                if self.data[label].shape[0] > audio_samples_per_class:
                    continue
                audio = load_audio(audio_path)
                trimmed_audio = trim_audio(audio, float(start_time), float(end_time))
                self.data[label] = np.concatenate((self.data[label], trimmed_audio), axis=0)
                completed = all(frames.shape[0] > audio_samples_per_class for frames in self.data.keys())
    
    def __iter__(self):
        # TODO: drop last?
        for label, samples in self.data.keys():
            samples_yielded = 0
            while samples_yielded < self.audio_samples_per_class:
                audio_batch = []
                for _ in range(self.batch_size):
                    audio_batch.append(samples[samples_yielded:samples_yielded+self.audio_samples_per_class])
                    samples_yielded += self.audio_samples_per_class
                
                yield np.stack(audio_batch, dim=0), label.repeat(self.batch_size)
        