"""
BirdCLEF 2026 — Inference script for test soundscape predictions.
生成 submission.csv，参考 sample_submission.csv 的格式。

用法：
    python submission.py --checkpoint runs/.../best.pt
    python submission.py --checkpoint runs/.../best.pt --batch_size 64 --device cuda
"""
import argparse
import math
import os
import glob

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
import librosa
from tqdm import tqdm

# ─── 常量 ───────────────────────────────────────────────────────────────────
SAMPLE_RATE = 32000
DURATION = 5.0           # seconds
TARGET_LENGTH = SAMPLE_RATE * DURATION  # 160000 samples
N_MELS = 128
FMAX = 16000
FMIN = 20
TEST_AUDIO_DIR = "./test_soundscapes"
SUBMISSION_TEMPLATE = "./sample_submission.csv"


# ─── 模型定义（内嵌，不依赖 model.py） ─────────────────────────────────────────
class BirdClassifier(nn.Module):
    def __init__(self, model_name='efficientnet_b0', num_classes=234, pretrained=False):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=1,
            num_classes=0,
            global_pool='',
        )
        num_features = self.backbone.num_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)
        return logits


# ─── 音频处理 ────────────────────────────────────────────────────────────────
def load_audio_window(path, offset, duration, sr=SAMPLE_RATE):
    """读取指定 offset 的 audio 片段，不足 5s 则 zero pad"""
    try:
        audio, _ = librosa.load(path, sr=sr, offset=offset, duration=duration, mono=True)
    except Exception:
        return np.zeros(TARGET_LENGTH, dtype=np.float32)

    if len(audio) < TARGET_LENGTH:
        audio = np.pad(audio, (0, TARGET_LENGTH - len(audio)), mode='constant')
    else:
        audio = audio[:TARGET_LENGTH]
    return audio


def audio_to_spec(audio, sr=SAMPLE_RATE):
    """将一维音频转为 Mel spectrogram tensor"""
    spec = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    spec = librosa.power_to_db(spec, ref=np.max)
    spec = (spec - spec.min()) / (spec.max() - spec.min() + 1e-6)
    spec_t = torch.tensor(spec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # [1,1,128,T]
    return spec_t


# ─── 滑动窗口推理 ────────────────────────────────────────────────────────────
@torch.no_grad()
def predict_audio(model, path, device):
    """
    对一个音频文件推理，返回所有窗口聚合后的概率向量 [num_classes]
    - 音频 > 5s：滑动窗口，逐窗口推理，取 max 聚合
    - 音频 <= 5s：单窗口推理
    - 读取失败：返回 None（由调用方填 baseline）
    """
    try:
        total_dur = librosa.get_duration(path=path)
    except Exception:
        return None

    if total_dur <= DURATION:
        # 单窗口
        audio = load_audio_window(path, offset=0.0, duration=DURATION)
        spec = audio_to_spec(audio).to(device)
        logits = model(spec)
        probs = torch.sigmoid(logits).cpu().numpy().flatten()
    else:
        # 滑动窗口，取 max
        n_windows = math.ceil(total_dur / DURATION)
        all_probs = []
        for i in range(n_windows):
            offset = i * DURATION
            audio = load_audio_window(path, offset=offset, duration=DURATION)
            spec = audio_to_spec(audio).to(device)
            logits = model(spec)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.append(probs)
        probs = np.maximum.reduce(all_probs)

    return probs


# ─── row_id 解析 ────────────────────────────────────────────────────────────
def parse_row_id(row_id):
    """
    从 row_id 解析出音频文件路径和起始秒数。
    row_id 格式：BC2026_Test_0001_S05_20250227_010002_5
    拆解：{prefix}_{test_id}_{site}_{date}_{time}_{start_sec}
    """
    # 最后一段是起始秒数
    parts = row_id.rsplit('_', 1)
    if len(parts) != 2:
        return None, None
    prefix = parts[0]
    start_sec = float(parts[1])

    # 去掉末尾的起始秒数得到音频文件名部分
    # BC2026_Test_0001_S05_20250227_010002
    audio_basename = prefix + '.ogg'
    audio_path = os.path.join(TEST_AUDIO_DIR, audio_basename)
    return audio_path, start_sec


# ─── 主推理入口 ──────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 inference")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="模型权重路径（.pt）")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output", type=str, default="submission.csv")
    parser.add_argument("--backbone", type=str, default="efficientnet_b0")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # 加载模型
    print(f"Loading model from {args.checkpoint} ...")
    submission_template = pd.read_csv(SUBMISSION_TEMPLATE)
    species_cols = submission_template.columns[1:].tolist()
    num_classes = len(species_cols)

    model = BirdClassifier(model_name=args.backbone, num_classes=num_classes,
                           pretrained=False).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded. Device: {device}, num_classes: {num_classes}")

    # 遍历 test_soundscapes 中的所有 .ogg 文件
    audio_files = sorted(glob.glob(os.path.join(TEST_AUDIO_DIR, "*.ogg")))
    print(f"Found {len(audio_files)} audio files in {TEST_AUDIO_DIR}")

    # 逐文件推理并聚合结果
    # 格式：{audio_basename: {species: prob, ...}}
    file_predictions = {}  # basename -> np.array (num_classes,)

    for audio_path in tqdm(audio_files, desc="Predicting files"):
        basename = os.path.basename(audio_path)
        probs = predict_audio(model, audio_path, device)
        if probs is None:
            probs = np.full(num_classes, 1.0 / num_classes)
        file_predictions[basename] = probs

    # 生成 submission
    print("Building submission...")
    baseline_prob = 1.0 / num_classes
    results = []

    for _, row in tqdm(submission_template.iterrows(), total=len(submission_template),
                       desc="Generating rows"):
        row_id = row['row_id']
        audio_path, start_sec = parse_row_id(row_id)

        if audio_path and os.path.exists(audio_path):
            # 该文件已被推理
            audio_basename = os.path.basename(audio_path)
            probs = file_predictions.get(audio_basename)
            if probs is not None:
                row_values = probs
            else:
                row_values = np.full(num_classes, baseline_prob)
        else:
            row_values = np.full(num_classes, baseline_prob)

        results.append({**{'row_id': row_id}, **{sp: float(row_values[i])
                                                  for i, sp in enumerate(species_cols)}})

    submission = pd.DataFrame(results)
    submission = submission[['row_id'] + species_cols]
    submission.to_csv(args.output, index=False)
    print(f"\nSubmission saved to {args.output}")
    print(f"Shape: {submission.shape}")
    print(submission.head())


if __name__ == "__main__":
    main()
