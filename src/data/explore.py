"""Data exploration for BirdCLEF 2026."""
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from src.utils import (
    TRAIN_AUDIO_DIR,
    TRAIN_SOUNDSCAPES_DIR,
    TRAIN_CSV,
    TRAIN_LABELS_CSV,
    TAXONOMY_CSV,
    SAMPLE_SUBMISSION_CSV,
)


def explore_train_csv():
    """Explore train.csv - short clip metadata."""
    df = pd.read_csv(TRAIN_CSV, encoding="utf-8", nrows=5)
    print(f"\n=== train.csv ===")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head(3).to_string())
    total = sum(1 for _ in open(TRAIN_CSV, encoding="utf-8")) - 1
    print(f"Total rows: {total:,}")


def explore_train_audio():
    """Explore train_audio directory structure."""
    species_dirs = list(TRAIN_AUDIO_DIR.glob("*"))
    species_counts = {}
    for sd in species_dirs:
        if sd.is_dir():
            files = list(sd.glob("*.ogg"))
            species_counts[sd.name] = len(files)

    print(f"\n=== train_audio/ ===")
    print(f"Species directories: {len(species_counts)}")
    counts = list(species_counts.values())
    print(f"  Total files: {sum(counts)}")
    print(f"  Min/Max/Mean per species: {min(counts)}/{max(counts)}/{np.mean(counts):.1f}")
    few = {k: v for k, v in species_counts.items() if v <= 2}
    print(f"  Species with <=2 samples: {len(few)}")


def explore_soundscapes():
    """Explore soundscape files."""
    files = list(TRAIN_SOUNDSCAPES_DIR.glob("*.ogg"))
    print(f"\n=== train_soundscapes/ ===")
    print(f"Total files: {len(files)}")

    train_files = [f for f in files if "_Train_" in f.name]
    print(f"  Train soundscapes: {len(train_files)}")

    # Check durations
    try:
        import soundfile as sf
        durations = []
        for f in train_files[:5]:
            info = sf.info(f)
            durations.append(info.duration)
        print(f"  Sample durations: {durations}s")
    except ImportError:
        print(f"  soundfile not installed, skipping duration check")
    except Exception as e:
        print(f"  Could not read durations: {e}")


def explore_labels():
    """Explore train_soundscapes_labels.csv."""
    df = pd.read_csv(TRAIN_LABELS_CSV, encoding="utf-8")
    print(f"\n=== train_soundscapes_labels.csv ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head(5).to_string())

    # Parse semicolon-separated species
    def parse_labels(s):
        return s.split(";")

    all_species = set()
    for s in df["primary_label"]:
        all_species.update(parse_labels(s))

    print(f"\nUnique species in labels: {len(all_species)}")
    print(f"Sample species: {list(all_species)[:10]}")

    # Labels per row
    labels_per_row = df["primary_label"].apply(lambda s: len(parse_labels(s)))
    print(f"\nSpecies per window: mean={labels_per_row.mean():.1f}, min={labels_per_row.min()}, max={labels_per_row.max()}")

    # Unique files labeled
    print(f"\nUnique soundscape files labeled: {df['filename'].nunique()}")

    # Coverage: labeled windows vs total possible from train soundscapes
    return df


def explore_taxonomy():
    """Explore taxonomy.csv."""
    df = pd.read_csv(TAXONOMY_CSV)
    print(f"\n=== taxonomy.csv ===")
    print(f"Shape: {df.shape}")
    print(df.head(5).to_string())

    class_counts = df["class_name"].value_counts()
    print(f"\nClass distribution:")
    for cls, cnt in class_counts.items():
        print(f"  {cls}: {cnt}")


def explore_sample_submission():
    """Explore sample_submission.csv format."""
    df = pd.read_csv(SAMPLE_SUBMISSION_CSV, encoding="utf-8")
    print(f"\n=== sample_submission.csv ===")
    print(f"Shape: {df.shape}")
    id_col = df.columns[0]
    print(f"ID column: '{id_col}'")
    print(f"Sample row_id: {df.iloc[0][id_col]}")

    # Parse row_id format
    sample = df.iloc[0][id_col]
    parts = sample.rsplit("_", 1)
    offset = float(parts[1])
    base = parts[0]
    print(f"  Parsed: base='{base}', offset={offset}s")

    # Count windows per soundscape
    df["base"] = df[id_col].apply(lambda x: x.rsplit("_", 1)[0])
    windows_per_file = df.groupby("base").size()
    print(f"  Windows per soundscape: mean={windows_per_file.mean():.1f}, min={windows_per_file.min()}, max={windows_per_file.max()}")

    # Check value range
    numeric_df = df.iloc[:, 1:].apply(pd.to_numeric, errors="coerce")
    vals = numeric_df.values.ravel()
    vals = vals[~np.isnan(vals)]
    print(f"  Value range: [{vals.min():.6f}, {vals.max():.6f}]")


def main():
    print("=" * 60)
    print("BirdCLEF 2026 - Data Exploration")
    print("=" * 60)

    explore_train_csv()
    explore_train_audio()
    explore_soundscapes()
    explore_labels()
    explore_taxonomy()
    explore_sample_submission()


if __name__ == "__main__":
    main()
