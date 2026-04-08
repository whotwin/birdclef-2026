"""
SimCLR contrastive pretraining loop.
"""
import os
import time
import json
import math
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from encoder import SimCLREncoder
from loss import NTXentLoss
from augmentation import get_contrastive_augmentation
from dataset import ContrastiveAudioDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Logger:
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.log_file = run_dir / "train.log"
        os.makedirs(run_dir, exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
        self.metrics_file = run_dir / "metrics.txt"

    def info(self, msg):
        with open(self.log_file, "a") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        print(msg)

    def save_metric(self, key, value):
        with open(self.metrics_file, "a") as f:
            f.write(f"{key}: {value}\n")


def setup_run(run_dir_base, tag):
    ts = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(run_dir_base, f"{ts}_{tag}")
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    return run_dir


def get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    """Linear warmup then cosine decay."""
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def contrastive_collate_fn(batch):
    """Custom collate: batch = [(view1, view2), ...]"""
    view1_list = []
    view2_list = []
    for v1, v2 in batch:
        view1_list.append(v1)
        view2_list.append(v2)
    return torch.stack(view1_list, dim=0), torch.stack(view2_list, dim=0)


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device, scaler, epoch, log):
    model.train()
    total_loss = 0.0
    pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

    for view1, view2 in pbar:
        view1 = view1.to(device)
        view2 = view2.to(device)

        optimizer.zero_grad()
        with autocast(device_type=device.type):
            z1 = model(view1)
            z2 = model(view2)
            loss = criterion(z1, z2)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.6f}")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate contrastive loss on a subset."""
    model.eval()
    total_loss = 0.0
    count = 0
    for view1, view2 in loader:
        view1 = view1.to(device)
        view2 = view2.to(device)
        z1 = model(view1)
        z2 = model(view2)
        loss = criterion(z1, z2)
        total_loss += loss.item()
        count += 1
        if count >= 100:  # quick eval on subset
            break
    return total_loss / count


def train(cfg):
    device = torch.device(cfg['training']['device'] if torch.cuda.is_available() else 'cpu')
    set_seed(cfg['training']['seed'])

    run_dir = setup_run(cfg['output']['run_dir'], f"simclr_{cfg['model']['backbone']}")
    log = Logger(Path(run_dir))
    log.info(f"Run dir: {run_dir}")
    log.info(f"Config: {json.dumps(cfg, indent=2, default=str)}")

    # ── Augmentation ──────────────────────────────────────────────────────────
    aug = get_contrastive_augmentation(
        noise_std=cfg['augmentation']['noise_std'],
        volume_jitter=cfg['augmentation']['volume_jitter'],
        speed_range=cfg['augmentation'].get('speed_range', [0.8, 1.2]),
        n_silent_cuts=cfg['augmentation'].get('n_silent_cuts', 2),
        max_cut_ratio=cfg['augmentation'].get('max_cut_ratio', 0.05),
    )

    # ── Dataset ─────────────────────────────────────────────────────────────
    dataset = ContrastiveAudioDataset(
        csv_path=cfg['data']['csv_path'],
        audio_dir=cfg['data']['audio_dir'],
        sr=cfg['data']['sr'],
        duration=cfg['data']['duration'],
        n_mels=cfg['data']['n_mels'],
        fmin=cfg['data']['fmin'],
        fmax=cfg['data']['fmax'],
        augmentation=aug,
        soundscapes_dir=cfg['data'].get('soundscapes_dir'),
        n_samples_per_soundscape=cfg['data'].get('n_samples_per_soundscape', 2),
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg['training']['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        collate_fn=contrastive_collate_fn,
        drop_last=True,
    )
    log.info(f"Dataset size: {len(dataset)}, Batches: {len(loader)}")

    # ── Model ───────────────────────────────────────────────────────────────
    model = SimCLREncoder(
        backbone_name=cfg['model']['backbone'],
        projection_dim=cfg['model']['projection_dim'],
        pretrained=cfg['model']['pretrained'],
    ).to(device)
    log.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # ── Loss, Optimizer, Scheduler ──────────────────────────────────────────
    criterion = NTXentLoss(temperature=cfg['training']['temperature'], device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg['training']['lr'],
        weight_decay=float(cfg['training']['weight_decay']),
    )

    total_steps = len(loader) * cfg['training']['epochs']
    warmup_steps = len(loader) * cfg['training']['warmup_epochs']
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler()

    # ── Training Loop ────────────────────────────────────────────────────────
    best_loss = float('inf')
    for epoch in range(cfg['training']['epochs']):
        train_loss = train_one_epoch(
            model, loader, optimizer, scheduler, criterion, device, scaler, epoch, log
        )
        log.info(f"Epoch {epoch+1}/{cfg['training']['epochs']} | Loss: {train_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Save checkpoint
        if (epoch + 1) % cfg['output']['save_every'] == 0 or epoch == cfg['training']['epochs'] - 1:
            ckpt_path = os.path.join(run_dir, "checkpoints", f"epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'cfg': cfg,
            }, ckpt_path)
            log.info(f"  Saved: {ckpt_path}")

        # Save best
        if train_loss < best_loss:
            best_loss = train_loss
            ckpt_path = os.path.join(run_dir, "checkpoints", "best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_loss': train_loss,
                'cfg': cfg,
            }, ckpt_path)
            log.info(f"  New best loss: {best_loss:.4f} -> saved best.pt")

    log.info(f"Training complete. Best loss: {best_loss:.4f}")
    log.info(f"Run dir: {run_dir}")
    return run_dir
