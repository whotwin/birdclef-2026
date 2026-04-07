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
    def __init__(self, df, submission_df, sr=32000, duration=5, is_train=True,
                 mix_K=1, mix_scale=0.0, noise_std=0.0):
        """
        :param df: 包含 'filepath' 和 'primary_label' 列的 DataFrame
        :param submission_df: 包含提交格式的 DataFrame
        :param sr: 采样率 (BirdCLEF 常标配 32000)
        :param duration: 训练切片长度 (秒)
        :param mix_K: 混叠音频数量（课程学习参数）
        :param mix_scale: 混叠音量的最大 scale（其他音频 scale ∈ [0, mix_scale]，课程学习参数）
        :param noise_std: 白噪声标准差（课程学习参数）
        """
        self.df = df.reset_index(drop=True)
        self.sr = sr
        self.duration = duration
        self.is_train = is_train
        self.mix_K = mix_K
        self.mix_scale = mix_scale
        self.noise_std = noise_std
        self.target_length = sr * duration

        # 预先建立标签映射 (234 类)
        self.all_species = sorted(submission_df.iloc[:, 1:].columns.tolist())
        self.species_to_idx = {s: i for i, s in enumerate(self.all_species)}

    def __len__(self):
        return len(self.df)

    def _load_audio(self, row):
        """加载单条音频（pad/truncate 到固定长度），返回波形。"""
        path = row['filepath']
        has_valid_time = 'start' in row and not pd.isna(row['start']) and str(row['start']).strip() != ""
        if has_valid_time:
            offset = float(row['start'])
            read_duration = self.duration
        else:
            try:
                total_duration = librosa.get_duration(path=path)
                if total_duration > self.duration:
                    offset = random.uniform(0, total_duration - self.duration)
                    read_duration = self.duration
                else:
                    offset = 0
                    read_duration = total_duration
            except Exception:
                offset = 0
                read_duration = self.duration

        try:
            audio, _ = librosa.load(path, sr=self.sr, offset=offset, duration=read_duration, mono=True)
        except Exception as e:
            print(f"读取失败 {path}: {e}")
            audio = np.zeros(self.target_length)

        # Pad/truncate
        if len(audio) < self.target_length:
            diff = self.target_length - len(audio)
            pad_before = random.randint(0, diff)
            audio = np.pad(audio, (pad_before, diff - pad_before), mode='constant')
        else:
            audio = audio[:self.target_length]
        return audio

    def _get_label(self, row):
        """返回单条标签向量 [num_classes]。"""
        label = np.zeros(len(self.all_species), dtype=np.float32)
        birds = row.get('label_list', [row['primary_label']])
        for bird in birds:
            if bird in self.species_to_idx:
                label[self.species_to_idx[bird]] = 1.0
        return label

    def __getitem__(self, idx):
        """
        随机采样 K 条音频，混合叠加为一条，标签取 OR。
        返回: (spec [1, 128, T], label [num_classes])
        """
        K = self.mix_K

        # 采样 K 个索引（包含当前 idx）
        indices = [idx] + [random.randint(0, len(self.df) - 1) for _ in range(K - 1)]
        rows = [self.df.iloc[i] for i in indices]

        # 加载并混合波形：主音频(idx) + 其他音频按 [0, mix_scale] 缩放后叠加
        main_audio = self._load_audio(rows[0])
        mixed = main_audio.copy()
        for r in rows[1:]:
            scale = random.uniform(0, self.mix_scale)
            other_audio = self._load_audio(r)
            mixed = mixed + other_audio * scale

        # 叠加后归一化，防止 clipping
        mix_weight = 1.0 + self.mix_scale * (K - 1)
        mixed = mixed / max(mix_weight, 1e-6)

        # 添加白噪声
        if self.noise_std > 0:
            mixed = mixed + np.random.randn(*mixed.shape).astype(np.float32) * self.noise_std

        # 混合标签 = OR
        labels = np.stack([self._get_label(r) for r in rows])
        combined_label = np.any(labels > 0, axis=0).astype(np.float32)

        # 提取 Mel 频谱图
        spec = librosa.feature.melspectrogram(
            y=mixed, sr=self.sr, n_mels=128, fmin=20, fmax=16000
        )
        spec = librosa.power_to_db(spec, ref=np.max)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        spec_t = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)   # [1, 128, T]
        label_t = torch.tensor(combined_label, dtype=torch.float32)     # [num_classes]

        return spec_t, label_t

if __name__ == "__main__":
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

    # 测试普通模式（mix_range=1，不混合）
    ds1 = BirdDataset(combined_df, submission_df, mix_range=(1, 1))
    loader1 = DataLoader(ds1, batch_size=4, shuffle=True)
    specs, labels = next(iter(loader1))
    print(f"[mix_range=(1,1)] specs: {specs.shape}, labels: {labels.shape}")

    # 测试混合模式（mix_range=(1,4)，每次随机混合1~4条）
    ds2 = BirdDataset(combined_df, submission_df, mix_range=(1, 4))
    loader2 = DataLoader(ds2, batch_size=4, shuffle=True)
    specs2, labels2 = next(iter(loader2))
    print(f"[mix_range=(1,4)] specs: {specs2.shape}, labels: {labels2.shape}")
    print(f"数据集大小: {len(ds2)}")
# --- 使用示例 ---
# DATA_DIR = "/path/to/birdclef-2026"
# train_df = load_bird_data(DATA_DIR, "train_metadata.csv")

# 查看前 5 行
# print(train_df.head())

# 快速统计：看每个物种有多少个样本
# print(train_df['primary_label'].value_counts())