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
import yaml
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from model import BirdClassifier
from datasets import BirdDataset, load_bird_data, hms_to_seconds
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from torch.utils.data import WeightedRandomSampler


def load_config(config_path):
    """从 YAML 文件加载训练配置"""
    with open(config_path, 'r') as f:
        cfg = yaml.safe_load(f)
    return cfg


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


def setup_run(CFG, config_src=None):
    """自动创建本次运行的输出目录，返回 run_dir 路径"""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"lr{CFG['lr']}_bs{CFG['batch_size']}_ep{CFG['epochs']}"
    run_dir = Path(f"runs/{ts}_{tag}")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    # 保存当前使用的配置
    with open(run_dir / "config.json", "w") as f:
        json.dump(CFG, f, indent=2, default=str)
    # 同时保存原始 yaml 文件
    if config_src and Path(config_src).exists():
        import shutil
        shutil.copy(config_src, run_dir / "config.yaml")

    return run_dir


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
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['primary_label'])):
        df.loc[val_idx, 'fold'] = fold

    return df


# ---- 多卡训练辅助函数 ----

def setup_distributed():
    """初始化 DDP 环境"""
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank


def cleanup_distributed():
    """清理 DDP 环境"""
    if dist.is_initialized():
        dist.destroy_process_group()


def get_device(local_rank, multi_gpu, device_ids):
    """返回正确的设备"""
    if multi_gpu:
        return torch.device(f"cuda:{local_rank}")
    else:
        return torch.device(f"cuda:{device_ids[0]}")


# ---- 数据加载器 ----

def get_loaders_for_fold(df, fold, submission_df, batch_size, multi_gpu=False, world_size=1, rank=0,
                         mix_K=1, mix_scale=0.0, noise_std=0.0):
    """
    根据 fold 编号获取对应的训练和验证 DataLoader
    multi_gpu=True 时使用 DistributedSampler 替代 WeightedRandomSampler
    """
    # 选出当前 fold 作为验证集，其余作为训练集
    train_data = df[df['fold'] != fold].copy()
    valid_data = df[df['fold'] == fold].copy()

    # 实例化 Dataset，传入课程学习参数
    train_ds = BirdDataset(train_data[['filepath', 'primary_label']], submission_df,
                           mix_K=mix_K, mix_scale=mix_scale, noise_std=noise_std)
    valid_ds = BirdDataset(valid_data[['filepath', 'primary_label']], submission_df)

    if multi_gpu:
        # 多卡模式：使用 DistributedSampler
        train_sampler = DistributedSampler(
            train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
        )
        valid_sampler = DistributedSampler(
            valid_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=train_sampler, num_workers=2, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size,
                                  sampler=valid_sampler, num_workers=2, pin_memory=True)
    else:
        # 单卡模式：使用 WeightedRandomSampler 过采样
        class_counts = train_data['primary_label'].value_counts()
        max_count = class_counts.max()
        weights = train_data['primary_label'].map(
            lambda x: max_count / class_counts.get(x, 1)
        ).values
        weights = weights / weights.sum() * len(weights)

        oversample_sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size,
                                  sampler=oversample_sampler, num_workers=2, pin_memory=True)
        valid_loader = DataLoader(valid_ds, batch_size=batch_size,
                                  shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, valid_loader


# ---- 训练 / 验证 ----

def train_one_epoch(model, loader, optimizer, criterion, device, scaler, multi_gpu=False):
    model.train()

    pbar = tqdm(loader, total=len(loader), desc="Training")

    running_loss = 0.0
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with autocast(device_type=device.type):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        pbar.set_postfix(loss=loss.item())

    running_loss = running_loss / len(loader)

    # 多卡模式：跨卡同步 loss 均值
    if multi_gpu:
        tensor_loss = torch.tensor(running_loss, device=device)
        dist.all_reduce(tensor_loss, op=dist.ReduceOp.SUM)
        running_loss = tensor_loss.item() / dist.get_world_size()

    return running_loss


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, multi_gpu=False):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    pbar = tqdm(loader, total=len(loader), desc="Validating")

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item()
        all_preds.append(torch.sigmoid(outputs).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    n_batches = len(loader)
    val_loss = running_loss / n_batches

    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)

    # 多卡模式：汇总所有卡的 loss、预测和标签到 rank 0
    if multi_gpu:
        tensor_loss = torch.tensor(val_loss, device=device)
        preds_tensor = torch.tensor(all_preds, dtype=torch.float32, device=device)
        labels_tensor = torch.tensor(all_labels, dtype=torch.float32, device=device)

        dist.all_reduce(tensor_loss, op=dist.ReduceOp.SUM)
        tensor_loss = tensor_loss / dist.get_world_size()

        gathered_preds = [torch.zeros_like(preds_tensor) for _ in range(dist.get_world_size())]
        gathered_labels = [torch.zeros_like(labels_tensor) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_preds, preds_tensor)
        dist.all_gather(gathered_labels, labels_tensor)

        if dist.get_rank() == 0:
            val_loss = tensor_loss.item()
            all_preds = np.vstack([p.cpu().numpy() for p in gathered_preds])
            all_labels = np.vstack([l.cpu().numpy() for l in gathered_labels])
        else:
            val_loss = tensor_loss.item()
            all_preds = None
            all_labels = None

    return val_loss, all_preds, all_labels


# ---- 课程学习参数调度 ----

def get_curriculum_params(epoch, CFG):
    """根据当前 epoch 返回 BirdDataset 的课程学习参数"""
    warmup = CFG['curriculum_warmup_epochs']
    total = CFG['epochs']
    max_K = CFG['curriculum_K_end']
    max_scale = CFG['curriculum_scale_end']
    max_noise = CFG['curriculum_noise_end']
    segments = CFG.get('curriculum_segments', max_K - 1)  # 阶段数，默认 max_K-1

    if epoch < warmup:
        return {'mix_K': 1, 'mix_scale': 0.0, 'noise_std': 0.0}

    # 将剩余 epoch 均分为 segments 段，每段一种 K 值
    remaining = total - warmup
    step = remaining / segments
    segment = min(int((epoch - warmup) / step), segments - 1)
    K = segment + 2   # K = 2 ~ (segments + 1)，不超过 max_K

    # scale 和 noise 与 K 线性相关（K=2 时最小，K=max_K 时最大）
    ratio = (K - 1) / (max_K - 1)
    scale = max_scale * ratio
    noise = max_noise * ratio

    return {'mix_K': K, 'mix_scale': scale, 'noise_std': noise}


# ---- 单 Fold 训练（供 DDP spawn 调用） ----

def train_fold(fold, train_df, submission_df, CFG, run_dir, rank=0, world_size=1):
    """训练单个 fold，支持单卡和多卡"""
    multi_gpu = CFG['multi_gpu']
    device_ids = CFG.get('device_ids', [0])
    device = get_device(rank, multi_gpu, device_ids)

    log = Logger(run_dir)
    log.info(f"[Rank {rank}] Fold {fold} starting on device {device}")

    train_data = train_df[train_df['fold'] != fold].copy()
    if CFG['pos_weight']:
        pos_weight = compute_pos_weight(train_data, submission_df).to(device)
        pos_weight = torch.clamp(pos_weight, max=5)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    model = BirdClassifier(model_name=CFG['backbone'], num_classes=234).to(device)
    print(f"[Rank {rank}] Model loaded, device={next(model.parameters()).device}", flush=True)

    if multi_gpu:
        model = DDP(model, device_ids=[rank])
        print(f"[Rank {rank}] DDP wrapped", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])
    print(f"[Rank {rank}] Optimizer & scheduler ready", flush=True)

    fold_ckpt_dir = run_dir / "checkpoints" / f"fold{fold}"
    fold_ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_auc = 0.0
    best_preds = None
    best_targets = None

    print(f"[Rank {rank}] Starting epoch loop...", flush=True)

    for epoch in range(CFG['epochs']):
        # 获取当前 epoch 的课程学习参数并重建 train DataLoader
        params = get_curriculum_params(epoch, CFG)
        train_loader, val_loader = get_loaders_for_fold(
            train_df, fold, submission_df,
            batch_size=CFG['batch_size'],
            multi_gpu=multi_gpu, world_size=world_size, rank=rank,
            mix_K=params['mix_K'],
            mix_scale=params['mix_scale'],
            noise_std=params['noise_std'],
        )
        if multi_gpu:
            train_loader.sampler.set_epoch(epoch)

        if (not multi_gpu) or (rank == 0):
            log.info(f"[Rank {rank}] Fold {fold} | Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)} | Curriculum: K={params['mix_K']}, scale={params['mix_scale']:.3f}, noise={params['noise_std']:.5f}")

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler, multi_gpu=multi_gpu)
        val_loss, preds, targets = validate_one_epoch(model, val_loader, criterion, device, multi_gpu=multi_gpu)

        # 只在 rank 0 计算并打印 metrics（validate_one_epoch 已在 rank 0 聚合数据）
        if (not multi_gpu) or (rank == 0):
            val_auc = calculate_metrics(targets, preds)
            scheduler.step()
            log.info(f"  Epoch {epoch+1}/{CFG['epochs']} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val AUC: {val_auc:.4f} | Curriculum: K={params['mix_K']}, scale={params['mix_scale']:.3f}, noise={params['noise_std']:.5f}")

            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_preds = preds
                best_targets = targets
                ckpt_path = fold_ckpt_dir / "best.pt"
                # rank 0 保存模型
                if not multi_gpu:
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    torch.save(model.module.state_dict(), ckpt_path)
                log.info(f"  Best AUC updated: {best_val_auc:.4f} -> saved to {ckpt_path}")
        else:
            # 非 rank 0 也要 step scheduler 保持同步
            scheduler.step()

    # 保存 fold OOF 结果（仅 rank 0）
    if rank == 0:
        np.savez_compressed(run_dir / f"oof_fold{fold}.npz", preds=best_preds, targets=best_targets)
        log.info(f"Fold {fold} done | Best AUC: {best_val_auc:.4f}")
        log.save_result(f"fold{fold}_best_auc", f"{best_val_auc:.4f}")

    return best_val_auc, best_preds, best_targets


# ---- 主入口 ----

def main_worker(rank, world_size, train_df, submission_df, CFG, run_dir):
    """DDP worker 入口"""
    multi_gpu = CFG['multi_gpu']
    device_ids = CFG.get('device_ids', [0])

    if multi_gpu:
        local_rank = rank
        setup_distributed()
        device = get_device(local_rank, multi_gpu, device_ids)
    else:
        device = get_device(rank, multi_gpu, device_ids)

    # 只在 rank 0 设置 Logger 写文件（避免多进程冲突）
    if rank == 0:
        log = Logger(run_dir)
        log.info(f"Using device: {device}")
        log.info(f"Config: {json.dumps({k: v for k, v in CFG.items() if k != 'device_ids'}, default=str)}")
    else:
        log = None

    # 单卡模式直接训练所有 fold；多卡模式每个 worker 训练自己的 fold
    if multi_gpu:
        # 多卡：每个 rank 只训练分配给它的 fold
        fold = rank % CFG['n_folds']
        best_val_auc, preds, targets = train_fold(
            fold, train_df, submission_df, CFG, run_dir, rank=rank, world_size=world_size
        )
        # 同步最终 AUC
        tensor_auc = torch.tensor(best_val_auc, device=device)
        dist.all_reduce(tensor_auc, op=dist.ReduceOp.MIN)
        if rank == 0:
            final_auc = tensor_auc.item()
            log = Logger(run_dir)
            log.info(f"\nMulti-GPU training complete! Final OOF AUC (approximate from rank 0): {final_auc:.4f}")
            cleanup_distributed()
    else:
        # 单卡：顺序训练所有 fold
        all_oof_preds = []
        all_oof_targets = []
        for fold in range(CFG['n_folds']):
            best_val_auc, preds, targets = train_fold(
                fold, train_df, submission_df, CFG, run_dir, rank=rank, world_size=world_size
            )
            all_oof_preds.append(preds)
            all_oof_targets.append(targets)

        # 最终综合评估
        final_preds = np.vstack(all_oof_preds)
        final_targets = np.vstack(all_oof_targets)
        final_auc = calculate_metrics(final_targets, final_preds)
        if log:
            log.info(f"\n五折交叉验证完成！全量 OOF Macro AUC: {final_auc:.4f}")
            log.save_result("oof_macro_auc", f"{final_auc:.4f}")
            log.info(f"Run dir: {run_dir}")


if __name__ == "__main__":
    import sys
    sys.stderr = sys.stdout  # 确保错误输出不缓冲
    # --- 加载配置 ---
    config_path = Path(__file__).parent / "config.yaml"
    CFG = load_config(config_path)
    CFG['lr'] = float(CFG['lr'])

    # --- 自动创建输出目录 ---
    run_dir = setup_run(CFG, config_src=config_path)

    # --- 准备数据 ---
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

    # --- 判断启动方式 ---
    # torchrun 设置 RANK 环境变量，自动管理分布式进程
    is_torchrun = os.environ.get("RANK") is not None
    multi_gpu = CFG['multi_gpu'] and is_torchrun
    rank = int(os.environ.get("RANK", 0)) if is_torchrun else 0
    world_size = int(os.environ.get("WORLD_SIZE", 1)) if is_torchrun else 1
    CFG['multi_gpu'] = multi_gpu

    log = Logger(run_dir) if rank == 0 else None
    if rank == 0:
        log.info(f"Run dir: {run_dir}")
        log.info(f"Config: multi_gpu={multi_gpu}, is_torchrun={is_torchrun}")

    main_worker(rank=rank, world_size=world_size, train_df=train_df,
                submission_df=submission_df, CFG=CFG, run_dir=run_dir)
