"""Constants and paths for BirdCLEF 2026."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT
TRAIN_AUDIO_DIR = DATA_DIR / "train_audio"
TRAIN_SOUNDSCAPES_DIR = DATA_DIR / "train_soundscapes"
TEST_SOUNDSCAPES_DIR = DATA_DIR / "test_soundscapes"
TRAIN_CSV = DATA_DIR / "train.csv"
TRAIN_LABELS_CSV = DATA_DIR / "train_soundscapes_labels.csv"
TAXONOMY_CSV = DATA_DIR / "taxonomy.csv"
SAMPLE_SUBMISSION_CSV = DATA_DIR / "sample_submission.csv"

# Audio settings
SAMPLE_RATE = 32000
WINDOW_DURATION = 5.0  # seconds
WINDOW_SIZE = int(SAMPLE_RATE * WINDOW_DURATION)  # 160000 samples

# Spectrogram settings
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 0
FMAX = 16000
