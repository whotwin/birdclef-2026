import pandas as pd
import os
import torch.nn as nn
import librosa
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def load_bird_data(base_path, csv_name="train_metadata.csv"):
    """
    针对 BirdCLEF 2026 的数据读取脚本
    :param base_path: 数据集根目录 (包含 train_audio, train_metadata.csv 等)
    :param csv_name: 要读取的 CSV 文件名
    """
    csv_path = os.path.join(base_path, csv_name)
    
    # 1. 读取 CSV
    df = pd.read_csv(csv_path)
    
    # 2. 针对 Soundscape 数据的去重处理 (参考你文档 1.3 节)
    if 'start' in df.columns:
        initial_count = len(df)
        df = df.drop_duplicates(subset=["filename", "start"], keep="first")
        print(f"清理重复窗口: {initial_count} -> {len(df)}")
    
    # 3. 路径补全：将文件名转换为绝对路径，方便 librosa 加载
    # 假设音频在 train_audio 文件夹下
    if 'filename' in df.columns:
        audio_dir = os.path.join(base_path, "train_audio")
        df['filepath'] = df['filename'].apply(lambda x: os.path.join(audio_dir, x))

    # 4. 标签解析：将 "22961;23158" 这种字符串转为 List
    if 'primary_label' in df.columns:
        df['label_list'] = df['primary_label'].str.split(';')

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
        
        # 1. 读取音频 (只读前 5 秒或随机偏移)
        try:
            # 训练时可以随机 offset 以增强数据，推理时固定从 0 开始
            audio, _ = librosa.load(path, sr=self.sr, duration=self.duration, mono=True)
        except Exception as e:
            print(f"读取失败 {path}: {e}")
            audio = np.zeros(self.sr * self.duration)

        # 2. 长度补齐 (Padding)
        target_length = self.sr * self.duration
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))

        # 3. 提取 Mel 频谱图 (核心算法步骤)
        spec = librosa.feature.melspectrogram(
            y=audio, sr=self.sr, n_mels=128, fmin=20, fmax=16000
        )
        spec = librosa.power_to_db(spec, ref=np.max)
        
        # 4. 标准化 (Min-Max 或 Mean-Std)
        spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
        
        # 5. 转换为 Tensor (Channel, Freq, Time)
        spec_t = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)

        # 6. 处理多标签 (Multi-hot Encoding)
        # 即使只有 primary_label，也建议转为 234 维向量以适配 BCE 损失
        label = torch.zeros(len(self.all_species))
        label[self.species_to_idx[row['primary_label']]] = 1.0
        
        return spec_t, label

if __name__ == "__main__":
    # 测试函数
    DATA_DIR = "./"
    train_df = load_bird_data(DATA_DIR, 'train.csv')
    submission_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
    selected_df = train_df[['filepath', 'primary_label']]
    bird_dataset = BirdDataset(selected_df, submission_df)
    dataloader = DataLoader(bird_dataset, batch_size=4, shuffle=True)
    for i, (specs, labels) in enumerate(dataloader):
        print(f"Batch {i}: specs shape {specs.shape}, labels shape {labels.shape}")
        if i >= 1:  # 只看前 2 个 batch
            break
    print(f"数据集大小: {len(bird_dataset)}")
    print(selected_df.head())
# --- 使用示例 ---
# DATA_DIR = "/path/to/birdclef-2026"
# train_df = load_bird_data(DATA_DIR, "train_metadata.csv")

# 查看前 5 行
# print(train_df.head())

# 快速统计：看每个物种有多少个样本
# print(train_df['primary_label'].value_counts())