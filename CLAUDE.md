# BirdCLEF 2026 - Project Context

## Competition
Kaggle BirdCLEF 2026 - Multi-label audio classification of bird (and other animal) species from soundscape recordings in the Pantanal wetland, Brazil.

## Task
Given 5-second audio windows from soundscape recordings, predict the probability of each species' presence among 234 classes.

## Data
- `train_audio/`: Short clips organized by species_id (~1-20 clips per species)
- `train_soundscapes/`: Long continuous recordings (.ogg)
- `train_soundscapes_labels.csv`: 5-second window labels (row_id + 234 species columns)
- `sample_submission.csv`: Submission format (row_id + 234 probability columns)
- `taxonomy.csv`: Species taxonomy info

## Project Structure
```
src/
  data/
    explore.py      # Data exploration script
    dataset.py      # Dataset classes (TrainAudioDataset, SoundscapeDataset, InferenceDataset)
    spectrogram.py  # Audio -> Mel spectrogram conversion
  models/
    baseline.py     # Pre-trained ResNet/EfficientNet model
  train.py          # Training script
  inference.py      # Inference/prediction script
  utils.py          # Constants and paths
main.py             # CLI entry point
```

## Commands
```bash
python main.py explore                          # Explore dataset
python main.py train --epochs 10 --batch_size 32  # Train model
python main.py predict --checkpoint checkpoints/best.pt  # Predict
python main.py validate --checkpoint checkpoints/best.pt  # Validate on train set
```

## Model
- Pre-trained ResNet34 (ImageNet) as backbone
- Mel spectrograms treated as single-channel images
- Multi-label BCEWithLogitsLoss
- AdamW optimizer + CosineAnnealingLR

## Dependencies
librosa, torch, torchaudio, pandas, numpy, scipy, scikit-learn, soundfile, tqdm
