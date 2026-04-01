# BirdCLEF 2026 - 训练方案说明

> 本文档记录当前训练流程的设计思路、数据构成原理及已知问题。

---

## 1. 数据现状总览

### 1.1 三类数据源

本项目有三个潜在数据源，它们的规模和用途差异巨大：

| 数据源 | 文件数 | 标注情况 | 用途 |
|---|---|---|---|
| `train_audio/` | 35,549 个 .ogg | 每个文件有单一物种标签 | 短音频片段，物种纯净 |
| `train_soundscapes/` | 10,658 个 .ogg | 仅 66 个有标注（739 窗口） | 长录音，可切分为 5 秒窗口 |
| `test_soundscapes/` | 待 rerun 更新 | 无标签 | 最终推理目标 |

### 1.2 当前只用了一小部分数据

```
已标注的 5 秒窗口: 739 个（去重后）← 实际参与训练
train_audio 片段:  35,549 个 ← 已有 Dataset 类但未接入训练流程
```

**问题**：训练集极小（739 个窗口，665 训练 / 74 验证），对 234 分类任务严重不足。

### 1.3 标签重复问题（已修复）

`train_soundscapes_labels.csv` 中每个 `(filename, start)` 对出现了两次，导致原始有 1478 行，实际只有 **739 个独立窗口**。已在 `SoundscapeDataset.__init__` 中通过 `drop_duplicates(subset=["filename", "start"], keep="first")` 去重。

---

## 2. 当前训练流程

### 2.1 数据处理 pipeline

```
长录音 .ogg (60秒)
    │
    ▼
librosa.load(offset, duration=5.0)   ← 从长录音中切出 5 秒窗口
    │
    ▼
pad_or_truncate(160,000 采样点)       ← 不足 5 秒填充零
    │
    ▼
librosa.feature.melspectrogram(...)   ← Mel 频谱图 (128, time)
    │
    ▼
librosa.power_to_db(spec)             ← 转 dB 尺度
    │
    ▼
(spec - mean) / (std + 1e-6)         ← 标准化
    │
    ▼
torch.tensor → (1, 128, time)         ← 模型输入
```

### 2.2 标签处理

每个 5 秒窗口的 `primary_label` 列是分号分隔的物种 ID 字符串（如 `22961;23158;24321`），通过 `species_to_idx` 映射转为 **多热向量（multi-hot vector）**，维度为 234。

使用 `BCEWithLogitsLoss`（多标签二分类交叉熵）训练，每个物种独立做二分类。

### 2.3 模型架构

```
输入: (batch, 1, 128, time)  ← Mel 频谱图（单通道）
    │
    ▼
修改 conv1: (1, 64, 7×7)    ← ResNet 原为 (3, 64, 7×7)，改为单通道
    │                         ← 权重用 RGB 均值初始化（迁移学习）
    ▼
ResNet34 骨干 (ImageNet 预训练)
    │
    ▼
移除 FC 层
    │
    ▼
Dropout(0.3) → Linear(512, 512) → ReLU → Dropout(0.3) → Linear(512, 234)
    │
    ▼
输出: (batch, 234) logits
```

总参数量：约 21.7M

### 2.4 训练配置

| 参数 | 值 |
|---|---|
| 损失函数 | `BCEWithLogitsLoss` |
| 优化器 | `AdamW` (lr=1e-3, weight_decay=1e-4) |
| 学习率调度 | `CosineAnnealingLR` |
| 批大小 | 32 |
| 训练轮数 | 10（默认） |
| 验证集划分 | 10%（random_state=42） |
| 检查点 | 每轮保存 + 验证损失最优保存 `best.pt` |

---

## 3. 关键设计决策及原因

### 3.1 为什么用 Mel 频谱图而不是原始波形？

鸟类叫声的关键信息集中在特定频率范围内，Mel 频谱图将线性频率映射到 Mel 频率尺度（更符合人耳感知），能更好地表征鸟类声音的特征。ResNet 本身是图像分类模型，将频谱图视为"图像"输入是最自然的迁移方式。

### 3.2 为什么只用了 SoundscapeDataset 而没用 TrainAudioDataset？

`TrainAudioDataset` 已实现但训练流程未接入，原因：
- 短音频片段每个只有单一物种标签，是"干净"的分类数据
- 但短音频和长录音的声学环境差异很大（背景噪声、录音质量、时间尺度）
- 混合训练可能引入分布不匹配问题

**建议后续**：先用 `TrainAudioDataset` 做物种分类预训练，再用 `SoundscapeDataset` 微调（两阶段训练）。

### 3.3 为什么有 28 个物种在 train_audio 里找不到？

`train_audio/` 只覆盖了 206 个物种，另有 28 个物种只有 `train_soundscapes/` 中的标注窗口数据（且只有部分有标签）。这 28 个物种只能从 739 个有标签的窗口中间接学习，样本极其稀少。

---

## 4. 当前流程的不足与改进方向

| 问题 | 当前状态 | 改进方向 |
|---|---|---|
| 训练集极小（739 窗口） | 仅用有标注的 66 个录音 | 利用 35,549 个短音频片段做预训练 |
| 99.4% 录音无标注 | 直接丢弃 | 尝试无监督/自监督学习（如 BYOL、SimCLR） |
| 28 个物种样本极少 | 无法有效学习 | 数据增强、过采样、迁移学习 |
| 每窗口多物种标签 | 全部视为正样本 | 可尝试标签平滑、类别权重调整 |
| 未使用 MFCC 等其他特征 | 仅 Mel 频谱图 | 特征拼接或多输入模型 |

---

## 5. 文件对应关系

| 文件 | 职责 |
|---|---|
| `src/data/dataset.py` | `SoundscapeDataset`（训练）、`InferenceDataset`（推理）、`TrainAudioDataset`（预训练备选） |
| `src/data/spectrogram.py` | `AudioProcessor`：音频加载、Mel 频谱图、MFCC |
| `src/models/baseline.py` | `SpectrogramClassifier`：ResNet/EfficientNet 骨干 + 分类头 |
| `src/train.py` | 训练循环、验证、保存检查点 |
| `src/inference.py` | 推理、生成 submission.csv |
| `src/utils.py` | 所有路径和超参数常量 |
| `inference.ipynb` | 独立推理 Notebook（不依赖项目内部模块） |
