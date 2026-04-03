import torch
import logging
import json
from datetime import datetime
from pathlib import Path
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
from model import BirdClassifier
from datasets import BirdDataset, load_bird_data, hms_to_seconds
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import WeightedRandomSampler


class Logger:
    """同时输出到 stdout 和文件的日志记录器"""
    def __init__(self, run_dir):
        self.run_dir = run_dir
        self.log_file = run_dir / "train.log"
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            datefmt="%H:%M:%S",
            handlers=[
                logging.FileHandler(self.log_file),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

    def info(self, msg):
        self.logger.info(msg)

    def save_result(self, key, value):
        """将结果写入 metrics.txt"""
        with open(self.run_dir / "metrics.txt", "a") as f:
            f.write(f"{key}: {value}\n")


def setup_run(CFG):
    """自动创建本次运行的输出目录，返回 run_dir 路径"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"lr{CFG['lr']}_bs{CFG['batch_size']}_ep{CFG['epochs']}"
    run_dir = Path(f"runs/{ts}_{tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    with open(run_dir / "config.json", "w") as f:
        json.dump(CFG, f, indent=2, default=str)

    return run_dir


# 在 validate_one_epoch 结束后调用
def calculate_metrics(y_true, y_pred_logits):
    # 预处理：确保 y_true 是 numpy 数组
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()

    # 将 logits 转为概率
    y_pred_prob = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()

    # 计算 Macro ROC-AUC（BirdCLEF 官方 metric）
    # 逐类别计算 AUC，只保留正负样本都存在的类
    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            try:
                auc = roc_auc_score(y_true[:, i], y_pred_prob[:, i])
                aucs.append(auc)
            except ValueError:
                pass
    macro_auc = np.mean(aucs) if aucs else 0.0

    return macro_auc


def compute_pos_weight(df, submission_df):
    """计算各类别的正样本权重，用于 BCEWithLogitsLoss"""
    all_species = sorted(submission_df.iloc[:, 1:].columns.tolist())
    species_to_idx = {s: i for i, s in enumerate(all_species)}
    n_samples = len(df)
    pos_counts = np.zeros(len(species_to_idx))

    for _, row in df.iterrows():
        birds = row.get('label_list', [row['primary_label']])
        for bird in birds:
            if bird in species_to_idx:
                pos_counts[species_to_idx[bird]] += 1

    # 权重 = 负样本数 / 正样本数，防止除零
    pos_weight = (n_samples - pos_counts) / (pos_counts + 1e-6)
    # 归一化，使平均权重为 1
    pos_weight = pos_weight / pos_weight.mean()
    return torch.tensor(pos_weight, dtype=torch.float32)

def prepare_cv_folds(df, n_splits=5):
    # 创建分层采样器
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # 初始化 fold 列
    df['fold'] = -1
    
    # 根据 primary_label 进行分层划分
    # 注意：如果某个物种只有 1 个样本，skf 会报错或警告，建议提前过滤或合并极少数样本
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        df.loc[val_idx, 'fold'] = fold
        
    return df

def train_one_epoch(model, loader, optimizer, criterion, device, scaler):
    model.train()
    running_loss = 0.0
    
    # 进度条
    pbar = tqdm(loader, total=len(loader), desc="Training")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 1. 开启混合精度计算 (节省显存并提速)
        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # 2. 反向传播与梯度缩放
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return running_loss / len(loader)

@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(loader, total=len(loader), desc="Validating")
    
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        running_loss += loss.item()
        
        # 记录预测结果用于计算 Metric (如 F1-score)
        all_preds.append(torch.sigmoid(outputs).cpu().numpy())
        all_labels.append(labels.cpu().numpy())
        
    return running_loss / len(loader), np.vstack(all_preds), np.vstack(all_labels)

def get_loaders_for_fold(df, fold, submission_df, batch_size=4):
    """
    根据 fold 编号获取对应的训练和验证 DataLoader
    """
    # 选出当前 fold 作为验证集，其余作为训练集
    train_data = df[df['fold'] != fold].copy()
    valid_data = df[df['fold'] == fold].copy()

    # 实例化 Dataset (基于你之前的定义)
    train_ds = BirdDataset(train_data[['filepath', 'primary_label']], submission_df)
    valid_ds = BirdDataset(valid_data[['filepath', 'primary_label']], submission_df)

    # --- 少数类过采样：基于 primary_label 频率加权抽样 ---
    class_counts = train_data['primary_label'].value_counts()
    max_count = class_counts.max()
    weights = train_data['primary_label'].map(
        lambda x: max_count / class_counts.get(x, 1)
    ).values
    weights = weights / weights.sum() * len(weights)  # 归一化

    oversample_sampler = WeightedRandomSampler(
        weights=weights, num_samples=len(weights), replacement=True
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              sampler=oversample_sampler, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size,
                              shuffle=False, num_workers=2)

    return train_loader, valid_loader

if __name__ == "__main__":
    # --- 1. 配置参数（只需改这里，路径自动生成） ---
    CFG = {
        'lr': 1e-3,
        'batch_size': 256,  # 如果显存报错，调小到 16 或 8
        'epochs': 20,
        'n_folds': 3,
        'pos_weight': True,
        'backbone':'efficientnet_b0',
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    }

    # --- 自动创建输出目录 ---
    run_dir = setup_run(CFG)
    log = Logger(run_dir)
    log.info(f"Run dir: {run_dir}")
    log.info(f"Config: {json.dumps(CFG, default=str)}")

    # --- 2. 准备数据 ---
    DATA_DIR = "./"
    train_df = load_bird_data(DATA_DIR, 'train_audio', 'train.csv')
    soundscapes_df = load_bird_data(DATA_DIR, 'train_soundscapes', 'train_soundscapes_labels.csv')
    submission_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    selected_df = train_df[['filepath', 'start', 'end', 'primary_label']]
    soundscapes_df = soundscapes_df[['filepath', 'start', 'end', 'primary_label']]
    selected_df['start'] = selected_df['start'].apply(hms_to_seconds)
    selected_df['end'] = selected_df['end'].apply(hms_to_seconds)

    soundscapes_df['start'] = soundscapes_df['start'].apply(hms_to_seconds)
    soundscapes_df['end'] = soundscapes_df['end'].apply(hms_to_seconds)
    train_df = pd.concat([selected_df, soundscapes_df], axis=0, ignore_index=True)
    train_df = prepare_cv_folds(train_df, n_splits=CFG['n_folds'])

    # --- 3. 五折训练 ---
    all_oof_preds = []
    all_oof_targets = []

    for fold in range(CFG['n_folds']):
        best_val_auc = 0.0
        train_loader, val_loader = get_loaders_for_fold(train_df, fold, submission_df, batch_size=CFG['batch_size'])
        fold_ckpt_dir = run_dir / "checkpoints" / f"fold{fold}"
        fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

        train_data = train_df[train_df['fold'] != fold].copy()
        if CFG['pos_weight']:
            pos_weight = compute_pos_weight(train_data, submission_df).to(CFG['device'])
            pos_weight = torch.clamp(pos_weight, max=20)
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            criterion = nn.BCEWithLogitsLoss()
        model = BirdClassifier(model_name=CFG['backbone'], num_classes=234).to(CFG['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
        scaler = GradScaler()
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])

        log.info(f"Fold {fold} | Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")

        for epoch in range(CFG['epochs']):
            log.info(f"Epoch {epoch+1}/{CFG['epochs']}")

            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG['device'], scaler)
            val_loss, preds, targets = validate_one_epoch(model, val_loader, criterion, CFG['device'])
            val_auc = calculate_metrics(targets, preds)
            scheduler.step()

            log.info(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                ckpt_path = fold_ckpt_dir / "best.pt"
                torch.save(model.state_dict(), ckpt_path)
                log.info(f"  Best AUC updated: {best_val_auc:.4f} -> saved to {ckpt_path}")

        # 保存 fold OOF 结果
        np.savez_compressed(run_dir / f"oof_fold{fold}.npz", preds=preds, targets=targets)
        log.info(f"Fold {fold} done | Best AUC: {best_val_auc:.4f}")
        log.save_result(f"fold{fold}_best_auc", f"{best_val_auc:.4f}")

        all_oof_preds.append(preds)
        all_oof_targets.append(targets)

    # --- 4. 最终综合评估 ---
    final_preds = np.vstack(all_oof_preds)
    final_targets = np.vstack(all_oof_targets)
    final_auc = calculate_metrics(final_targets, final_preds)
    log.info(f"\n五折交叉验证完成！全量 OOF Macro AUC: {final_auc:.4f}")
    log.save_result("oof_macro_auc", f"{final_auc:.4f}")
    log.info(f"Run dir: {run_dir}")