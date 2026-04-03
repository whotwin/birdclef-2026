import pandas as pd
import os
import torch.nn as nn
import librosa
import torch
import random
import numpy as np
from torch.utils.data import Dataset, DataLoader

def hms_to_seconds(hms_str):
    """将 '00:00:20' 转换为 20.0"""
    if pd.isna(hms_str) or str(hms_str).strip() == "":
        return np.nan  # 保持为空，触发 Dataset 的随机截取逻辑
    
    try:
        # 处理可能的格式：'20' 或 '00:20' 或 '00:00:20'
        parts = str(hms_str).split(':')
        if len(parts) == 1:
            return float(parts[0])
        elif len(parts) == 2:
            return float(parts[0]) * 60 + float(parts[1])
        elif len(parts) == 3:
            return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    except Exception:
        return np.nan

def load_bird_data(base_path, audio_dir="train_audio", csv_name="train_metadata.csv"):
    """
    针对 BirdCLEF 2026 的数据读取脚本
    :param base_path: 数据集根目录
    :param audio_dir: 音频文件目录名
    :param csv_name: 要读取的 CSV 文件名
    """
    csv_path = os.path.join(base_path, csv_name)
    
    # 1. 读取 CSV
    df = pd.read_csv(csv_path)
    
    # 2. 检查并添加 start 和 end 字段
    # 如果没有这两个列，则初始化为 NaN（空白填充）
    if 'start' not in df.columns:
        df['start'] = np.nan
    if 'end' not in df.columns:
        df['end'] = np.nan
    
    # 3. 针对 Soundscape 数据的去重处理
    # 注意：对于全量 NaN 的数据，drop_duplicates 会保留一行
    initial_count = len(df)
    df = df.drop_duplicates(subset=["filename", "start"], keep="first")
    if len(df) < initial_count:
        print(f"清理重复窗口: {initial_count} -> {len(df)}")
    
    # 4. 路径补全
    if 'filename' in df.columns:
        # 你可以根据实际存放位置修改这里的子目录名称
        audio_dir = os.path.join(base_path, audio_dir)
        df['filepath'] = df['filename'].apply(lambda x: os.path.join(audio_dir, x))
    
    # 5. 标签解析：将字符串转为 List，方便 Dataset 循环处理
    if 'primary_label' in df.columns:
        # 将 "bird1;bird2" 转为 ["bird1", "bird2"]
        # 如果是单标签 "bird1"，则转为 ["bird1"]
        df['label_list'] = df['primary_label'].astype(str).str.split(';')
        
    return df

class BirdDataset(Dataset):
    def __init__(self, df, submission_df, sr=32000, duration=5, is_train=True):
        """
        :param df: 包含 'filepath' 和 'primary_label' 列的 DataFrame
        :param submission_df: 包含提交格式的 DataFrame
        :param sr: 采样率 (BirdCLEF 常标配 32000)
        :param duration: 训练切片长度 (秒)
        """
        self.df = df.reset_index(drop=True)
        self.sr = sr
        self.duration = duration
        self.is_train = is_train
        
        # 预先建立标签映射 (234 类)
        self.all_species = sorted(submission_df.iloc[:, 1:].columns.tolist())  # 假设第一列是 ID 列
        self.species_to_idx = {s: i for i, s in enumerate(self.all_species)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        path = row['filepath']
        target_length = self.sr * self.duration # 5秒对应的采样点数 (如 160000)

        # --- 1. 确定读取范围 (Offset & Duration) ---
        # 检查是否有 start 和 end 字段
        has_valid_time = 'start' in row and not pd.isna(row['start']) and str(row['start']).strip() != ""
        if has_valid_time:
            # 精准读取模式
            start_sec = float(row['start'])
            read_duration = self.duration # 固定读 5 秒
            offset = start_sec
        else:
            # 随机读取模式
            try:
                # 获取总时长 (librosa.get_duration 不会加载整个文件，速度快)
                total_duration = librosa.get_duration(path=path)
                
                if total_duration > self.duration:
                    # 随机选择一个起始点，确保不超出范围
                    offset = random.uniform(0, total_duration - self.duration)
                    read_duration = self.duration
                else:
                    # 音频本身不足 5 秒
                    offset = 0
                    read_duration = total_duration
            except Exception:
                offset = 0
                read_duration = self.duration

        # --- 2. 加载音频 ---
        try:
            audio, _ = librosa.load(path, sr=self.sr, offset=offset, duration=read_duration, mono=True)
        except Exception as e:
            print(f"读取失败 {path}: {e}")
            audio = np.zeros(target_length)

        # --- 3. 长度补齐 (随机前后填充) ---
        current_length = len(audio)
        if current_length < target_length:
            diff = target_length - current_length
            # 随机决定前面补多少，后面补多少
            pad_before = random.randint(0, diff)
            pad_after = diff - pad_before
            audio = np.pad(audio, (pad_before, pad_after), mode='constant')
        elif current_length > target_length:
            # 预防万一读取超长
            audio = audio[:target_length]

        # --- 4. 提取 Mel 频谱图 ---
        spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=128, fmin=20, fmax=16000
        )
        spec = librosa.power_to_db(spec, ref=np.max)
        
        # 标准化
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        
        # --- 5. 转换为 Tensor ---
        spec_t = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        # --- 6. 处理多标签 (适应分号分隔的 birds) ---
        label = torch.zeros(len(self.all_species))
        # 优先读 label_list (之前 load_bird_data 处理好的)，如果没有则读 primary_label
        birds = row.get('label_list', [row['primary_label']])
        
        for bird in birds:
            if bird in self.species_to_idx:
                label[self.species_to_idx[bird]] = 1.0
        
        return spec_t, label

if __name__ == "__main__":
    # 测试函数
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
    combined_df = pd.concat([selected_df, soundscapes_df], axis=0, ignore_index=True)
    bird_dataset = BirdDataset(combined_df, submission_df)
    dataloader = DataLoader(bird_dataset, batch_size=4, shuffle=True)
    for i, (specs, labels) in enumerate(dataloader):
        print(f"Batch {i}: specs shape {specs.shape}, labels shape {labels.shape}")
        if i >= 20:  # 只看前 20 个 batch
            break
    print(f"数据集大小: {len(bird_dataset)}")
    print(soundscapes_df.head())
# --- 使用示例 ---
# DATA_DIR = "/path/to/birdclef-2026"
# train_df = load_bird_data(DATA_DIR, "train_metadata.csv")

# 查看前 5 行
# print(train_df.head())

# 快速统计：看每个物种有多少个样本
# print(train_df['primary_label'].value_counts())