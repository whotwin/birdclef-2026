"""
Spectrogram augmentations for contrastive learning.
All augmentations operate on mel spectrograms [1, n_mels, T].
"""
import random
import numpy as np
import torch


class SpecAugment:
    """SpecAugment: randomly mask frequency and time bands."""

    def __init__(self, freq_mask_param=20, time_mask_param=40,
                 n_freq_mask=2, n_time_mask=2, p=0.5):
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.n_freq_mask = n_freq_mask
        self.n_time_mask = n_time_mask
        self.p = p

    def __call__(self, spec):
        """
        Args:
            spec: Tensor [1, n_mels, T]
        Returns:
            Augmented tensor [1, n_mels, T]
        """
        if random.random() > self.p:
            return spec

        n_mels, T = spec.shape[1], spec.shape[2]
        spec = spec.clone()

        # Frequency masking
        for _ in range(self.n_freq_mask):
            f = random.randint(0, min(self.freq_mask_param, n_mels - 1))
            f0 = random.randint(0, n_mels - f)
            spec[:, f0:f0 + f, :] = 0

        # Time masking
        for _ in range(self.n_time_mask):
            t = random.randint(0, min(self.time_mask_param, T - 1))
            t0 = random.randint(0, T - t)
            spec[:, :, t0:t0 + t] = 0

        return spec


class TimeCrop:
    """Random time-domain crop with padding."""

    def __init__(self, crop_ratio=0.1, pad_value=0.0):
        self.crop_ratio = crop_ratio
        self.pad_value = pad_value

    def __call__(self, spec):
        """
        Args:
            spec: Tensor [1, n_mels, T]
        Returns:
            Cropped and padded tensor [1, n_mels, T]
        """
        _, n_mels, T = spec.shape
        crop_size = int(T * (1.0 - self.crop_ratio))
        if crop_size >= T:
            return spec

        start = random.randint(0, T - crop_size)
        cropped = spec[:, :, start:start + crop_size]

        # Pad back to original length
        pad_left = start
        pad_right = T - (start + crop_size)
        return torch.nn.functional.pad(cropped, (pad_left, pad_right), value=self.pad_value)


class VolumeJitter:
    """Random volume scaling."""

    def __init__(self, jitter_range=0.2):
        self.jitter_range = jitter_range

    def __call__(self, spec):
        scale = 1.0 + random.uniform(-self.jitter_range, self.jitter_range)
        return spec * scale


class GaussianNoise:
    """Add Gaussian noise."""

    def __init__(self, std=0.005):
        self.std = std

    def __call__(self, spec):
        noise = torch.randn_like(spec) * self.std
        return spec + noise


class Compose:
    """Compose multiple augmentations."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, spec):
        for t in self.transforms:
            spec = t(spec)
        return spec


def get_contrastive_augmentation(
    freq_mask_param=20,
    time_mask_param=40,
    n_freq_mask=2,
    n_time_mask=2,
    volume_jitter=0.2,
    noise_std=0.005,
    time_crop_ratio=0.1,
):
    """Build the standard contrastive augmentation pipeline."""
    return Compose([
        TimeCrop(crop_ratio=time_crop_ratio),
        SpecAugment(
            freq_mask_param=freq_mask_param,
            time_mask_param=time_mask_param,
            n_freq_mask=n_freq_mask,
            n_time_mask=n_time_mask,
            p=0.5,
        ),
        VolumeJitter(jitter_range=volume_jitter),
        GaussianNoise(std=noise_std),
    ])
