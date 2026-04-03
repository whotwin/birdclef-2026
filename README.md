# BirdCLEF 2026

Multi-label bird (and other animal) audio classification from soundscape recordings in the Pantanal wetland, Brazil.

## Competition Overview

**Task**: Given 5-second audio windows from soundscape recordings, predict the probability of each species' presence among 234 classes (birds, amphibians, insects, mammals).

**Evaluation**: Multi-label classification (CoLER or mAP - to confirm)

## Project Structure

```
src/
  data/
    explore.py      # Dataset exploration
    dataset.py      # Dataset classes
    spectrogram.py  # Audio -> Mel spectrogram
  models/
    baseline.py     # Pre-trained ResNet/EfficientNet
  train.py          # Training script
  inference.py      # Inference script
  utils.py          # Config & paths
main.py             # CLI entry point
```

## Quick Start

```bash
# Install dependencies
pip install -e .

# Explore data
python main.py explore

# Train
python main.py train --epochs 10 --batch_size 32 --max_windows 5000

# Validate
python main.py validate --checkpoint checkpoints/best.pt

# Generate submission
python main.py predict --checkpoint checkpoints/best.pt --output submission.csv
```

## Model

- **Backbone**: Pre-trained ResNet34 (ImageNet), treating mel spectrograms as images
- **Head**: Dropout → FC(512) → ReLU → Dropout → FC(234)
- **Loss**: BCEWithLogitsLoss (multi-label)
- **Optimizer**: AdamW + CosineAnnealingLR
