import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import warnings
from model import BirdClassifier
from datasets import BirdDataset, load_bird_data, hms_to_seconds
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, average_precision_score

# 在 validate_one_epoch 结束后调用
def calculate_metrics(y_true, y_pred_logits, threshold=0.5):
    # 1. 预处理数据
    # 确保 y_true 是 numpy 数组
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
        
    # 将 logits 转为概率
    y_pred_prob = torch.sigmoid(torch.tensor(y_pred_logits)).numpy()
    # 转为二值标签
    y_pred_label = (y_pred_prob > threshold).astype(int)
    
    # 2. 计算 Macro F1
    # zero_division=0 已经处理了 F1 的警告问题
    f1 = f1_score(y_true, y_pred_label, average='macro', zero_division=0)
    
    # 3. 计算 mAP (修复 UserWarning)
    # 核心逻辑：只计算那些在 y_true 中至少出现过一次的类别
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning) # 双重保险：强制忽略该函数内的 UserWarning
        
        # 找出 y_true 中有正样本的列 (类)
        # np.any(..., axis=0) 会返回一个布尔向量，代表哪些列有 1
        classes_with_positives = np.any(y_true > 0, axis=0)
        
        if np.sum(classes_with_positives) > 0:
            # 只对有正样本的列进行计算
            map_score = average_precision_score(
                y_true[:, classes_with_positives], 
                y_pred_prob[:, classes_with_positives], 
                average='macro'
            )
        else:
            # 如果极端情况下整个验证集一个正样本都没有
            map_score = 0.0
            
    return f1, map_score

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
    
    # 实例化 DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    valid_loader = DataLoader(valid_ds, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, valid_loader

if __name__ == "__main__":
    all_oof_preds = []
    all_oof_targets = []
    # --- 1. 配置参数 ---
    CFG = {
        'lr': 1e-3,
        'batch_size': 128, # 如果显存报错，调小到 16 或 8
        'epochs': 25,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'model_path': 'models/best_model.pth'
    }

    # --- 2. 准备数据 (假设你已定义好 BirdDataset) ---
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
    # 生成 5 折索引
    train_df = prepare_cv_folds(train_df, n_splits=5)
    for fold in range(5):
        best_val_f1 = 0.0
        train_loader, val_loader = get_loaders_for_fold(train_df, fold, submission_df, batch_size=CFG['batch_size'])
        fold_model_path = f"models/resnet34_fold{fold}_best.pth"
        # --- 3. 实例化模型、优化器和损失函数 ---
        model = BirdClassifier(num_classes=234).to(CFG['device'])
        optimizer = torch.optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss() # 适配多标签
        scaler = GradScaler() # 配合混合精度
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG['epochs'])

        # --- 4. 训练主循环 ---
        best_val_loss = float('inf')

        for epoch in range(CFG['epochs']):
            print(f"\nEpoch {epoch+1}/{CFG['epochs']}")
            
            # 训练
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG['device'], scaler)
            
            # 验证
            val_loss, preds, targets = validate_one_epoch(model, val_loader, criterion, CFG['device'])

            val_f1, val_map = calculate_metrics(targets, preds)
            
            # 学习率调整
            scheduler.step()

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val mAP: {val_map:.4f}")
            
            # 保存最优模型
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save(model.state_dict(), f"models/best_f1_fold{fold}.pth")
                print(f"🏆 发现更高 F1，模型已更新")
        all_oof_preds.append(preds)
        all_oof_targets.append(targets)

    # 最终综合评估
    final_preds = np.vstack(all_oof_preds)
    final_targets = np.vstack(all_oof_targets)
    final_f1, _ = calculate_metrics(final_targets, final_preds)
    print(f"\n✅ 五折交叉验证完成！全量 OOF Macro F1: {final_f1:.4f}")