"""
Unsupervised dataset for contrastive learning.
Each call returns two independently-augmented views of the same audio.
"""
import os
import random
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import Dataset


class ContrastiveAudioDataset(Dataset):
    """
    Loads audio clips, applies two random augmentations to create a pair of views.

    Since train.csv has no start/end columns (NaN), uses random offset sampling
    for each audio file.
    """

    def __init__(self, csv_path, audio_dir, sr=32000, duration=5,
                 n_mels=128, fmin=20, fmax=16000,
                 augmentation=None, max_offset=None):
        """
        Args:
            csv_path: path to train.csv
            audio_dir: path to train_audio directory
            sr: sample rate
            duration: clip duration in seconds
            n_mels, fmin, fmax: mel spectrogram params
            augmentation: callable that takes audio np.ndarray [target_length] and returns augmented audio
            max_offset: max random offset to try (for long files, limit to save time)
        """
        self.df = pd.read_csv(csv_path)
        self.audio_dir = audio_dir
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.target_length = sr * duration
        self.augmentation = augmentation
        self.max_offset = max_offset

        # Build filepath
        self.df['filepath'] = self.df['filename'].apply(
            lambda x: os.path.join(audio_dir, x)
        )

        # Cache file durations to avoid repeated librosa.get_duration calls
        self._duration_cache = {}

    def _get_duration(self, path):
        if path not in self._duration_cache:
            try:
                self._duration_cache[path] = librosa.get_duration(path=path)
            except Exception:
                self._duration_cache[path] = None
        return self._duration_cache[path]

    def _load_audio(self, filepath):
        """Load audio with random offset, pad/truncate to fixed length."""
        total_dur = self._get_duration(filepath)

        if total_dur and total_dur > self.duration:
            max_offset_val = total_dur - self.duration
            if self.max_offset is not None:
                max_offset_val = min(max_offset_val, self.max_offset)
            offset = random.uniform(0, max_offset_val)
        else:
            offset = 0

        try:
            audio, _ = librosa.load(
                filepath, sr=self.sr, offset=offset,
                duration=self.duration, mono=True
            )
        except Exception:
            audio = np.zeros(self.target_length, dtype=np.float32)

        # Pad/truncate
        if len(audio) < self.target_length:
            diff = self.target_length - len(audio)
            pad_before = random.randint(0, diff)
            audio = np.pad(audio, (pad_before, diff - pad_before), mode='constant')
        else:
            audio = audio[:self.target_length]

        return audio

    def _audio_to_spec(self, audio):
        """Convert audio waveform to mel spectrogram tensor [1, n_mels, T]."""
        spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax
        )
        spec = librosa.power_to_db(spec, ref=np.max)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        return torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Returns two augmented views of the same audio.
        Returns: (view1, view2), each [1, n_mels, T]
        """
        filepath = self.df.iloc[idx]['filepath']

        # Load audio once
        audio = self._load_audio(filepath)

        # Apply two independent audio-level augmentations, then convert to spec
        view1_audio = self.augmentation(audio.copy())
        view2_audio = self.augmentation(audio.copy())
        view1 = self._audio_to_spec(view1_audio)
        view2 = self._audio_to_spec(view2_audio)

        return view1, view2
