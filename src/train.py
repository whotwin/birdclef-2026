"""Training script for BirdCLEF 2026."""
import argparse
import json
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import train_test_split

from src.data.dataset import SoundscapeDataset, TrainAudioDataset, load_species_columns
from src.models.baseline import get_model
from src.utils import PROJECT_ROOT, N_MELS, WINDOW_DURATION


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0
    n_batches = 0

    for batch_idx, (specs, labels, _) in enumerate(loader):
        specs = specs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    return total_loss / n_batches


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    n_batches = 0

    for specs, labels, _ in loader:
        specs = specs.to(device)
        labels = labels.to(device)
        logits = model(specs)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / n_batches


def main():
    parser = argparse.ArgumentParser(description="Train BirdCLEF 2026 model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone", type=str, default="resnet34")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--max_windows", type=int, default=None,
                        help="Limit number of training windows for fast testing")
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device}")

    # Load species list
    species_list = load_species_columns()
    n_classes = len(species_list)
    print(f"Number of classes: {n_classes}")

    # Create full dataset
    print("Loading soundscape dataset...")
    full_dataset = SoundscapeDataset()
    print(f"Total windows: {len(full_dataset)}")

    # Split into train/val
    n_total = len(full_dataset)
    indices = np.arange(n_total)
    train_idx, val_idx = train_test_split(
        indices, test_size=0.1, random_state=42
    )

    if args.max_windows:
        train_idx = train_idx[:args.max_windows]
        val_idx = val_idx[:min(args.max_windows // 10, len(val_idx))]

    train_dataset = Subset(full_dataset, train_idx)
    val_dataset = Subset(full_dataset, val_idx)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Create model
    model = get_model(
        n_classes=n_classes,
        backbone=args.backbone,
        pretrained=True,
        dropout=args.dropout,
    ).to(device)

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # Loss, optimizer, scheduler
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Training loop
    best_val_loss = float("inf")
    checkpoint_dir = PROJECT_ROOT / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(1, args.epochs + 1):
        print(f"\n=== Epoch {epoch}/{args.epochs} ===")
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        elapsed = time.time() - t0
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {elapsed:.0f}s")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        # Save checkpoint
        ckpt_path = checkpoint_dir / f"checkpoint_epoch{epoch}.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "args": vars(args),
            },
            ckpt_path,
        )
        print(f"  Saved: {ckpt_path}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = checkpoint_dir / "best.pt"
            torch.save(model.state_dict(), best_path)
            print(f"  New best! Saved to {best_path}")

    # Save training history
    with open(checkpoint_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved in: {checkpoint_dir}")


if __name__ == "__main__":
    main()
