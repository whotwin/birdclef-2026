"""
Audio-level augmentations for contrastive learning.
All augmentations operate on raw audio waveforms [target_length].
"""
import random
import numpy as np
import librosa


class MultiNoise:
    """Add white, pink, or brown noise (randomly chosen per call)."""

    def __init__(self, noise_std=0.005):
        self.noise_std = noise_std

    def __call__(self, audio):
        """
        Args:
            audio: np.ndarray [target_length], float32
        Returns:
            np.ndarray [target_length], float32
        """
        noise_type = random.choice(['white', 'pink', 'brown'])
        n = len(audio)

        if noise_type == 'white':
            noise = np.random.randn(n).astype(np.float32) * self.noise_std
        elif noise_type == 'pink':
            # Pink noise via successive integration of white noise
            white = np.random.randn(n).astype(np.float32)
            noise = np.cumsum(white).astype(np.float32)
            noise = noise - noise[0]  # zero-mean
            std = np.std(noise)
            if std > 1e-6:
                noise = noise / std * self.noise_std
            else:
                noise = np.zeros(n, dtype=np.float32)
        else:  # brown
            white = np.random.randn(n).astype(np.float32)
            noise = np.cumsum(white).astype(np.float32)
            # Apply leaky integration to keep bounded
            noise = noise - 0.998 * np.roll(noise, 1)
            noise[0] = 0
            std = np.std(noise)
            if std > 1e-6:
                noise = noise / std * self.noise_std
            else:
                noise = np.zeros(n, dtype=np.float32)

        return (audio + noise).astype(np.float32)


class VolumeJitter:
    """Random volume scaling on audio waveform."""

    def __init__(self, jitter_range=0.2):
        self.jitter_range = jitter_range

    def __call__(self, audio):
        """
        Args:
            audio: np.ndarray [target_length], float32
        Returns:
            np.ndarray [target_length], float32
        """
        scale = 1.0 + random.uniform(-self.jitter_range, self.jitter_range)
        return (audio * scale).astype(np.float32)


class SpeedChange:
    """Change audio speed via time-stretch (resampling)."""

    def __init__(self, speed_range=(0.8, 1.2)):
        self.speed_range = speed_range

    def __call__(self, audio):
        """
        Args:
            audio: np.ndarray [target_length], float32
        Returns:
            np.ndarray [target_length], float32 (resampled back to original length)
        """
        speed = random.uniform(*self.speed_range)
        # time_stretch returns audio at different length
        stretched = librosa.effects.time_stretch(audio, rate=speed)

        # Resample back to original target length via linear interpolation
        target_len = len(audio)
        if len(stretched) == target_len:
            return stretched.astype(np.float32)
        indices = np.linspace(0, len(stretched) - 1, target_len)
        resampled = np.interp(indices, np.arange(len(stretched)), stretched).astype(np.float32)
        return resampled


class RandomSilentCut:
    """Randomly cut out silent segments in the audio."""

    def __init__(self, n_cuts=2, max_cut_ratio=0.05):
        self.n_cuts = n_cuts
        self.max_cut_ratio = max_cut_ratio

    def __call__(self, audio):
        """
        Args:
            audio: np.ndarray [target_length], float32
        Returns:
            np.ndarray [target_length], float32
        """
        n = len(audio)
        result = audio.copy()

        for _ in range(self.n_cuts):
            max_cut_len = int(n * self.max_cut_ratio)
            if max_cut_len < 1:
                continue
            cut_len = random.randint(1, max_cut_len)
            start = random.randint(0, n - cut_len)
            result[start:start + cut_len] = 0.0

        return result.astype(np.float32)


class Compose:
    """Compose multiple audio augmentations."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)
        return audio


def get_contrastive_augmentation(
    noise_std=0.005,
    volume_jitter=0.2,
    speed_range=(0.8, 1.2),
    n_silent_cuts=2,
    max_cut_ratio=0.05,
):
    """Build the audio-level contrastive augmentation pipeline."""
    return Compose([
        MultiNoise(noise_std=noise_std),
        VolumeJitter(jitter_range=volume_jitter),
        SpeedChange(speed_range=speed_range),
        RandomSilentCut(n_cuts=n_silent_cuts, max_cut_ratio=max_cut_ratio),
    ])
