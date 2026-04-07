"""
Linear evaluation: freeze backbone, train only a linear head on labeled data.
Reports Macro ROC-AUC to measure quality of learned representations.
"""
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import librosa
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from encoder import SimCLREncoder, LinearEvalHead


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LabeledAudioDataset(Dataset):
    """Dataset for labeled audio files (train_audio + train.csv)."""

    def __init__(self, df, submission_df, sr=32000, duration=5,
                 n_mels=128, fmin=20, fmax=16000):
        self.df = df.reset_index(drop=True)
        self.sr = sr
        self.duration = duration
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.target_length = sr * duration

        self.all_species = sorted(submission_df.iloc[:, 1:].columns.tolist())
        self.species_to_idx = {s: i for i, s in enumerate(self.all_species)}

    def __len__(self):
        return len(self.df)

    def _load_audio(self, filepath, start=None):
        if start is not None:
            offset = start
        else:
            offset = random.uniform(0, max(0, self._get_dur(filepath) - self.duration))

        try:
            audio, _ = librosa.load(filepath, sr=self.sr, offset=offset, duration=self.duration, mono=True)
        except Exception:
            audio = np.zeros(self.target_length, dtype=np.float32)

        if len(audio) < self.target_length:
            diff = self.target_length - len(audio)
            pad_before = random.randint(0, diff)
            audio = np.pad(audio, (pad_before, diff - pad_before), mode='constant')
        else:
            audio = audio[:self.target_length]
        return audio

    def _get_dur(self, path):
        try:
            return librosa.get_duration(path=path)
        except Exception:
            return 0

    def _audio_to_spec(self, audio):
        spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax
        )
        spec = librosa.power_to_db(spec, ref=np.max)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        return torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filepath = row['filepath']
        audio = self._load_audio(filepath)
        spec = self._audio_to_spec(audio)

        label = torch.zeros(len(self.all_species))
        sp = str(row['primary_label']).strip()
        if sp in self.species_to_idx:
            label[self.species_to_idx[sp]] = 1.0

        return spec, label


def calculate_metrics(y_true, y_pred):
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                aucs.append(auc)
            except ValueError:
                pass
    return np.mean(aucs) if aucs else 0.0


def train_linear_eval(pretrained_ckpt, cfg, fold=0):
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['training']['seed'])

    # ── Load data ──────────────────────────────────────────────────────────
    train_df = pd.read_csv(cfg['data']['csv_path'])
    submission_df = pd.read_csv(cfg['data']['submission_csv'])
    train_df['filepath'] = train_df['filename'].apply(
        lambda x: os.path.join(cfg['data']['audio_dir'], x)
    )

    # Stratified split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(skf.split(train_df, train_df['primary_label'])):
        if i == fold:
            break
    train_data = train_df.iloc[train_idx]
    val_data = train_df.iloc[val_idx]

    train_ds = LabeledAudioDataset(train_data, submission_df,
                                   sr=cfg['data']['sr'], duration=cfg['data']['duration'],
                                   n_mels=cfg['data']['n_mels'], fmin=cfg['data']['fmin'], fmax=cfg['data']['fmax'])
    val_ds = LabeledAudioDataset(val_data, submission_df,
                                 sr=cfg['data']['sr'], duration=cfg['data']['duration'],
                                 n_mels=cfg['data']['n_mels'], fmin=cfg['data']['fmin'], fmax=cfg['data']['fmax'])

    train_loader = DataLoader(train_ds, batch_size=cfg['linear_eval']['batch_size'],
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg['linear_eval']['batch_size'],
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"Fold {fold} | Train: {len(train_ds)}, Val: {len(val_ds)}")

    # ── Load pretrained encoder ──────────────────────────────────────────────
    encoder = SimCLREncoder(
        backbone_name=cfg['model']['backbone'],
        projection_dim=cfg['model']['projection_dim'],
        pretrained=False,
    )
    ckpt = torch.load(pretrained_ckpt, map_location='cpu', weights_only=True)
    encoder.load_state_dict(ckpt['model_state_dict'], strict=False)

    # Rebuild backbone keys (ignore projection head keys)
    backbone_state = {k: v for k, v in ckpt['model_state_dict'].items() if 'projection' not in k}
    encoder.backbone.load_state_dict({k: v for k, v in backbone_state.items() if k in encoder.backbone.state_dict()}, strict=False)
    encoder.global_pool.load_state_dict({k: v for k, v in backbone_state.items() if k in encoder.global_pool.state_dict()}, strict=False)

    model = LinearEvalHead(encoder, num_classes=234, freeze_backbone=True).to(device)
    print(f"Linear eval model loaded from {pretrained_ckpt}")

    # ── Train ───────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.linear_head.parameters(),
        lr=cfg['linear_eval']['lr'],
        weight_decay=cfg['linear_eval']['weight_decay'],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg['linear_eval']['epochs'])
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()

    best_auc = 0.0
    for epoch in range(cfg['linear_eval']['epochs']):
        model.train()
        for specs, labels in tqdm(train_loader, desc=f"Linear Epoch {epoch+1}", leave=False):
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast(device_type=device.type):
                out = model(specs)
                loss = criterion(out, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        scheduler.step()

        # Validate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for specs, labels in val_loader:
                specs = specs.to(device)
                logits = model(specs)
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_labels.append(labels.numpy())
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        auc = calculate_metrics(all_labels, all_preds)
        print(f"  Epoch {epoch+1} | Val AUC: {auc:.4f} | Loss: {loss.item():.4f}")

        if auc > best_auc:
            best_auc = auc

    print(f"\nFold {fold} Linear Eval Best AUC: {best_auc:.4f}")
    return best_auc
