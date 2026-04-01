"""Inference script for BirdCLEF 2026."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.dataset import InferenceDataset, load_species_columns
from src.models.baseline import get_model
from src.utils import TRAIN_SOUNDSCAPES_DIR, SAMPLE_SUBMISSION_CSV


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
):
    """Generate predictions for all windows."""
    model.eval()
    all_row_ids = []
    all_probs = []

    for batch_idx, (specs, row_ids) in enumerate(loader):
        specs = specs.to(device)
        logits = model(specs)
        probs = torch.sigmoid(logits).cpu().numpy()
        all_row_ids.extend(row_ids)
        all_probs.append(probs)

        if batch_idx % 50 == 0:
            print(f"  Batch {batch_idx}/{len(loader)}")

    return all_row_ids, np.vstack(all_probs)


def main():
    parser = argparse.ArgumentParser(description="Generate BirdCLEF 2026 predictions")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument(
        "--test_dir",
        type=str,
        default=str(TRAIN_SOUNDSCAPES_DIR),
        help="Directory containing test soundscapes",
    )
    parser.add_argument(
        "--use_train_labels",
        action="store_true",
        help="Validate on train_soundscapes instead of actual test",
    )
    parser.add_argument(
        "--submission_csv",
        type=str,
        default=str(SAMPLE_SUBMISSION_CSV),
        help="Path to sample_submission.csv",
    )
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    species_list = load_species_columns()
    n_classes = len(species_list)
    print(f"Number of classes: {n_classes}")

    # Load model
    model = get_model(
        n_classes=n_classes,
        backbone=args.backbone,
        pretrained=False,
    ).to(device)

    if args.checkpoint:
        state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint: {args.checkpoint}")
    else:
        print("WARNING: No checkpoint provided, using random weights!")

    # Determine row_ids
    test_dir = Path(args.test_dir)
    if args.use_train_labels:
        from src.utils import TRAIN_LABELS_CSV
        labels_df = pd.read_csv(TRAIN_LABELS_CSV, encoding="utf-8")

        def filename_to_row_id(filename, start):
            parts = start.split(":")
            offset = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            base = Path(filename).stem
            return f"{base}_{offset}"

        row_ids = [
            filename_to_row_id(row["filename"], row["start"])
            for _, row in labels_df.iterrows()
        ]
        print(f"Validating on {len(row_ids)} train windows")
    else:
        submission_df = pd.read_csv(args.submission_csv, encoding="utf-8")
        row_ids = submission_df["row_id"].tolist()
        print(f"Predicting {len(row_ids)} test windows")

    # Create dataset and loader
    inference_dataset = InferenceDataset(
        soundscapes_dir=test_dir,
        row_ids=row_ids,
    )
    loader = DataLoader(
        inference_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Predict
    print("Running inference...")
    row_ids_out, probs = predict(model, loader, device)

    # Build submission
    submission = pd.DataFrame(probs, columns=species_list)
    submission.insert(0, "row_id", row_ids_out)

    output_path = Path(args.output)
    submission.to_csv(output_path, index=False)
    print(f"\nSaved to: {output_path}")
    print(f"Shape: {submission.shape}")
    print(submission.head(3))


if __name__ == "__main__":
    main()
