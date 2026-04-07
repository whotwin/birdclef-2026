"""Audio to spectrogram conversion for BirdCLEF 2026."""
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Optional
import torch


def load_audio(
    path: Path | str,
    sr: int = 32000,
    offset: float = 0.0,
    duration: Optional[float] = None,
) -> np.ndarray:
    """Load audio file.

    Args:
        path: Path to audio file (.ogg or other formats)
        sr: Target sample rate
        offset: Start reading at this time (seconds)
        duration: Read this many seconds. If None, read entire file.

    Returns:
        Audio array (samples,)
    """
    audio, orig_sr = librosa.load(path, sr=sr, offset=offset, duration=duration)
    return audio


def audio_to_melspec(
    audio: np.ndarray,
    sr: int = 32000,
    n_mels: int = 128,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Convert audio to mel spectrogram.

    Args:
        audio: Audio array (samples,)
        sr: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length

    Returns:
        Mel spectrogram (n_mels, time)
    """
    spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmin=0,
        fmax=16000,
        power=2.0,
    )
    # Convert to dB scale
    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db


def audio_to_mfcc(
    audio: np.ndarray,
    sr: int = 32000,
    n_mfcc: int = 40,
    n_fft: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Convert audio to MFCC features.

    Returns:
        MFCC (n_mfcc, time)
    """
    mfcc = librosa.feature.mfcc(
        y=audio, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length
    )
    return mfcc


def pad_or_truncate(audio: np.ndarray, target_samples: int) -> np.ndarray:
    """Pad or truncate audio to target length.

    Args:
        audio: Audio array (samples,)
        target_samples: Target number of samples

    Returns:
        Audio array of exactly target_samples length
    """
    if len(audio) < target_samples:
        pad_len = target_samples - len(audio)
        audio = np.pad(audio, (0, pad_len), mode="constant")
    elif len(audio) > target_samples:
        audio = audio[:target_samples]
    return audio


def prepare_window(
    audio: np.ndarray,
    sr: int = 32000,
    window_duration: float = 5.0,
    target_samples: Optional[int] = None,
) -> np.ndarray:
    """Ensure audio window is exactly the right length.

    Args:
        audio: Audio array
        sr: Sample rate
        window_duration: Expected duration in seconds
        target_samples: Target samples (if None, computed from sr * duration)

    Returns:
        Audio of exact length
    """
    if target_samples is None:
        target_samples = int(sr * window_duration)
    return pad_or_truncate(audio, target_samples)


class AudioProcessor:
    """Processes audio files to spectrograms."""

    def __init__(
        self,
        sr: int = 32000,
        n_mels: int = 128,
        n_fft: int = 2048,
        hop_length: int = 512,
        window_duration: float = 5.0,
        cache_dir: Optional[Path] = None,
    ):
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window_duration = window_duration
        self.cache_dir = cache_dir
        self.target_samples = int(sr * window_duration)

    def process_file(
        self,
        path: Path | str,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> np.ndarray:
        """Load audio and convert to mel spectrogram.

        Args:
            path: Audio file path
            offset: Start offset in seconds
            duration: Duration in seconds (if None, reads rest of file)

        Returns:
            Mel spectrogram (n_mels, time)
        """
        audio = load_audio(path, sr=self.sr, offset=offset, duration=duration)
        spec = audio_to_melspec(
            audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return spec

    def process_fixed_window(
        self,
        path: Path | str,
        offset: float = 0.0,
        duration: float = 5.0,
    ) -> np.ndarray:
        """Load audio, pad/truncate to fixed window, then convert to mel spectrogram.

        Args:
            path: Audio file path
            offset: Start offset in seconds
            duration: Expected window duration (used for padding)

        Returns:
            Mel spectrogram (n_mels, time)
        """
        audio = load_audio(path, sr=self.sr, offset=offset, duration=duration)
        audio = pad_or_truncate(audio, self.target_samples)
        spec = audio_to_melspec(
            audio,
            sr=self.sr,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )
        return spec
