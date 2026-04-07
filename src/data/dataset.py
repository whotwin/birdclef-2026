"""Dataset classes for BirdCLEF 2026."""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List
import torch
from torch.utils.data import Dataset

from src.data.spectrogram import AudioProcessor, pad_or_truncate
from src.utils import (
    TRAIN_AUDIO_DIR,
    TRAIN_SOUNDSCAPES_DIR,
    TRAIN_LABELS_CSV,
    SAMPLE_SUBMISSION_CSV,
    SAMPLE_RATE,
    WINDOW_DURATION,
    N_MELS,
)


def load_species_columns() -> List[str]:
    """Get species list from sample_submission.csv columns."""
    df = pd.read_csv(SAMPLE_SUBMISSION_CSV, nrows=1, encoding="utf-8")
    return df.columns[1:].tolist()


def parse_time_to_seconds(time_str: str) -> float:
    """Parse HH:MM:SS string to seconds."""
    parts = time_str.split(":")
    h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    return float(h * 3600 + m * 60 + s)


def filename_to_row_id(filename: str, offset_seconds: float) -> str:
    """Convert soundscape filename + offset to row_id format."""
    base = Path(filename).stem  # removes .ogg
    return f"{base}_{int(offset_seconds)}"


class SoundscapeDataset(Dataset):
    """Dataset for labeled soundscape windows.

    Each item is a 5-second window from a labeled soundscape.
    Labels are semicolon-separated species IDs in 'primary_label' column.
    """

    def __init__(
        self,
        labels_csv: Path = TRAIN_LABELS_CSV,
        soundscapes_dir: Path = TRAIN_SOUNDSCAPES_DIR,
        transform=None,
    ):
        self.soundscapes_dir = soundscapes_dir
        self.transform = transform

        # Load labels
        self.labels_df = pd.read_csv(labels_csv, encoding="utf-8")

        # Load species columns from submission format
        self.species_list = load_species_columns()
        self.species_to_idx = {s: i for i, s in enumerate(self.species_list)}
        self.n_classes = len(self.species_list)

        self.processor = AudioProcessor(
            sr=SAMPLE_RATE, n_mels=N_MELS, window_duration=WINDOW_DURATION
        )

        # Pre-compute row_ids for each window
        self._build_row_ids()

    def _build_row_ids(self):
        """Pre-compute row_id for each label row."""
        self.row_ids = []
        for _, row in self.labels_df.iterrows():
            offset = parse_time_to_seconds(row["start"])
            row_id = filename_to_row_id(row["filename"], offset)
            self.row_ids.append(row_id)

    def __len__(self) -> int:
        return len(self.labels_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        row = self.labels_df.iloc[idx]
        row_id = self.row_ids[idx]
        soundscape_path = self.soundscapes_dir / row["filename"]

        # Load and process audio
        offset_sec = parse_time_to_seconds(row["start"])
        if soundscape_path.exists():
            audio = self.processor.load_audio(
                soundscape_path, offset=offset_sec, duration=WINDOW_DURATION
            )
            audio = pad_or_truncate(audio, self.processor.target_samples)
            spec = self.processor.audio_to_melspec(audio)
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        else:
            # Zeros if file missing
            n_time = int(SAMPLE_RATE * WINDOW_DURATION / 512) + 1
            spec = np.zeros((N_MELS, n_time))

        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)

        # Convert semicolon-separated labels to multi-hot
        labels = np.zeros(self.n_classes, dtype=np.float32)
        species_ids = row["primary_label"].split(";")
        for sp in species_ids:
            sp = sp.strip()
            if sp in self.species_to_idx:
                labels[self.species_to_idx[sp]] = 1.0

        labels_tensor = torch.from_numpy(labels).float()
        return spec_tensor, labels_tensor, row_id


class InferenceDataset(Dataset):
    """Dataset for generating predictions on test soundscapes.

    Each item is a 5-second window, and we generate the spectrogram on-the-fly.
    """

    def __init__(
        self,
        soundscapes_dir: Path = TRAIN_SOUNDSCAPES_DIR,
        row_ids: Optional[List[str]] = None,
    ):
        self.soundscapes_dir = soundscapes_dir
        self.species_list = load_species_columns()
        self.n_classes = len(self.species_list)

        self.processor = AudioProcessor(
            sr=SAMPLE_RATE, n_mels=N_MELS, window_duration=WINDOW_DURATION
        )
        self.row_ids = row_ids or []

    @staticmethod
    def parse_row_id(row_id: str) -> Tuple[str, float]:
        """Parse row_id to get soundscape filename and offset."""
        parts = row_id.rsplit("_", 1)
        offset = float(parts[1])
        base = parts[0]
        filename = base + ".ogg"
        return filename, offset

    def __len__(self) -> int:
        return len(self.row_ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        row_id = self.row_ids[idx]
        filename, offset = self.parse_row_id(row_id)
        path = self.soundscapes_dir / filename

        if path.exists():
            audio = self.processor.load_audio(path, offset=offset, duration=WINDOW_DURATION)
            audio = pad_or_truncate(audio, self.processor.target_samples)
            spec = self.processor.audio_to_melspec(audio)
            spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        else:
            n_time = int(SAMPLE_RATE * WINDOW_DURATION / 512) + 1
            spec = np.zeros((N_MELS, n_time))

        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        return spec_tensor, row_id


class TrainAudioDataset(Dataset):
    """Dataset for short train_audio clips.

    Each audio file belongs to one species label.
    Used for pre-training species classification.
    """

    def __init__(
        self,
        train_audio_dir: Path = TRAIN_AUDIO_DIR,
        transform=None,
    ):
        self.train_audio_dir = train_audio_dir
        self.transform = transform

        # Build file list: (filepath, species_label)
        self.samples = []
        species_dirs = sorted(train_audio_dir.glob("*"))
        self.species_list = [d.name for d in species_dirs if d.is_dir()]
        self.species_to_idx = {s: i for i, s in enumerate(self.species_list)}
        self.n_classes = len(self.species_list)

        for species_dir in species_dirs:
            if not species_dir.is_dir():
                continue
            species = species_dir.name
            for audio_file in species_dir.glob("*.ogg"):
                self.samples.append((audio_file, species))

        self.processor = AudioProcessor(
            sr=SAMPLE_RATE, n_mels=N_MELS, window_duration=WINDOW_DURATION
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        filepath, species = self.samples[idx]
        audio = self.processor.load_audio(filepath)
        audio = pad_or_truncate(audio, self.processor.target_samples)
        spec = self.processor.audio_to_melspec(audio)
        spec = (spec - spec.mean()) / (spec.std() + 1e-6)
        spec_tensor = torch.from_numpy(spec).float().unsqueeze(0)
        label = self.species_to_idx[species]
        return spec_tensor, label
