# BirdCLEF 2026 - 项目说明

> 多标签鸟类（及其他动物）声音分类，来自巴西潘塔纳尔湿地（Pantanal wetland）的声景录音。

## 1. 比赛背景

**任务**：给定声景录音中的 5 秒音频窗口，预测 234 个物种中每个物种的存在概率。

**物种分类**（共 234 类）：
- 鸟类（Aves）：162 种
- 两栖动物（Amphibia）：35 种
- 昆虫（Insecta）：28 种
- 哺乳动物（Mammalia）：8 种
- 爬行动物（Reptilia）：1 种

**评估指标**：多标签分类（CoLER 或 mAP，具体待确认）

**地理背景**：巴西潘塔纳尔湿地（世界最大热带湿地）

---

## 2. 数据说明

### 数据文件

| 文件 | 说明 |
|---|---|
| `train_audio/` | 35,549 个短音频片段（.ogg），按物种 ID 子目录组织 |
| `train_soundscapes/` | 10,658 个长连续录音（.ogg） |
| `train_soundscapes_labels.csv` | 标注数据：66 个录音文件的 1,478 个 5 秒窗口标签（物种以分号分隔） |
| `train.csv` | 短音频元数据：物种 ID、经纬度、学名、俗名、来源（iNaturalist / Xeno-canto） |
| `taxonomy.csv` | 234 个物种的分类信息 |
| `sample_submission.csv` | 提交格式示例：row_id + 234 个物种概率列 |

### 标注格式

`train_soundscapes_labels.csv` 中 `primary_label` 列为分号分隔的物种 ID 列表，例如：

```
filename,start,end,primary_label
BC2026_Train_0001,.ogg,00:00:00,00:00:05,blufoot1
BC2026_Train_0001,.ogg,00:00:05,00:00:10,blufoot1;sunbit1
```

### row_id 格式

```
{BC2026_Train/Test_XXXX}_{偏移秒数}
```
例如：`BC2026_Test_0001_S05_20250227_010002_5` → 测试集第 0001 号录音、偏移量 5 秒处。

---

## 3. 项目结构

```
d:/Competition/BirdCLEF/birdclef-2026/
├── main.py                    # CLI 入口（explore/train/predict/validate）
├── src/
│   ├── utils.py              # 路径常量、音频/频谱参数
│   ├── train.py              # 训练脚本
│   ├── inference.py          # 推理脚本
│   ├── data/
│   │   ├── dataset.py        # SoundscapeDataset、InferenceDataset、TrainAudioDataset
│   │   ├── spectrogram.py    # 音频加载、Mel 频谱图转换、MFCC
│   │   └── explore.py        # 数据探索脚本
│   └── models/
│       └── baseline.py       # 预训练模型定义（ResNet / EfficientNet）
├── train.csv
├── taxonomy.csv
├── train_soundscapes_labels.csv
├── sample_submission.csv
├── train_audio/              # 35,549 个 .ogg 短音频
├── train_soundscapes/        # 10,658 个 .ogg 长录音
└── test_soundscapes/         # 测试集（隐藏目录，赛中更新）
```

---

## 4. 模型架构

**核心思路**：将 Mel 频谱图视为单通道图像，用预训练图像分类模型进行特征提取。

### 骨干网络（可切换）

| 骨干网络 | 输入处理 |
|---|---|
| ResNet18 / ResNet34 / ResNet50 | 将第一层卷积改为单通道，输入 (1, H, W) 的频谱图 |
| EfficientNet-B0 | 将单通道复制为 3 通道，适配 ImageNet 预训练权重 |

### 分类头

```
Dropout(0.3) → Linear(in_features, 512) → ReLU → Dropout(0.3) → Linear(512, 234)
```

输出原始 logits（训练时配合 `BCEWithLogitsLoss`）。

---

## 5. 音频与频谱参数

定义于 `src/utils.py`：

| 参数 | 值 | 说明 |
|---|---|---|
| `SAMPLE_RATE` | 32,000 Hz | 采样率 |
| `WINDOW_DURATION` | 5.0 秒 | 窗口长度 |
| `WINDOW_SIZE` | 160,000 采样点 | 5 秒 × 32000 Hz |
| `N_MELS` | 128 | Mel 滤波器组数量 |
| `N_FFT` | 2048 | FFT 窗口大小 |
| `HOP_LENGTH` | 512 | 帧移 |
| `FMIN` | 0 Hz | 最低频率 |
| `FMAX` | 16,000 Hz | 最高频率 |

Mel 频谱图转换为 dB 尺度后输入模型。

---

## 6. 训练流程

- **数据集**：仅使用 `SoundscapeDataset`（1,478 个标注窗口），10% 验证集划分
- **损失函数**：`BCEWithLogitsLoss`（多标签二分类交叉熵）
- **优化器**：`AdamW`（默认学习率 1e-3，weight_decay 1e-4）
- **学习率调度**：`CosineAnnealingLR`
- **检查点保存**：
  - 每轮保存 `checkpoints/checkpoint_epoch{N}.pt`
  - 验证损失最优时保存 `checkpoints/best.pt`
  - 训练历史保存至 `checkpoints/history.json`

> **注意**：`TrainAudioDataset` 类已实现但训练脚本未使用，短音频片段暂未参与训练。

---

## 7. 使用方法

### 依赖安装

```bash
pip install librosa torch torchaudio pandas numpy scipy scikit-learn soundfile tqdm
# 或
pip install -e .
```

### CLI 命令

```bash
# 数据探索
python main.py explore

# 训练
python main.py train --epochs 10 --batch_size 32 --backbone resnet34 --dropout 0.3 --lr 1e-3

# 验证（在训练集声景上评估）
python main.py validate --checkpoint checkpoints/best.pt --backbone resnet34

# 生成提交文件
python main.py predict --checkpoint checkpoints/best.pt --backbone resnet34 --output submission.csv
```

### train 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--epochs` | 10 | 训练轮数 |
| `--batch_size` | 32 | 批大小 |
| `--lr` | 1e-3 | 学习率 |
| `--backbone` | resnet34 | 骨干网络（resnet18/34/50, efficientnet_b0） |
| `--dropout` | 0.3 | Dropout 比率 |
| `--max_windows` | 无 | 限制窗口数量，用于快速测试 |
| `--device` | 自动检测 | 设备（cuda/cpu） |

### predict 参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--checkpoint` | 必填 | 模型检查点路径 |
| `--backbone` | resnet34 | 骨干网络（需与训练时一致） |
| `--batch_size` | 64 | 批大小 |
| `--output` | submission.csv | 输出文件路径 |
| `--test_dir` | train_soundscapes/ | 测试音频目录 |
| `--use_train_labels` | False | 启用后对训练集声景进行验证 |
| `--submission_csv` | sample_submission.csv | 提交格式参考 CSV |

---

## 8. 已知限制

1. **训练集规模小**：仅 1,478 个标注窗口用于训练，10% 划分后约 1,330 个训练样本，对 234 分类任务而言非常有限。
2. **pyproject.toml 依赖为空**：`pyproject.toml` 中 `dependencies = []`，需手动安装依赖包。
3. **测试集目录为空**：`test_soundscapes/` 目前为空，测试音频可能在隐藏目录中，赛程中更新。
4. **长录音与标注不匹配**：`train_soundscapes/` 有 10,658 个录音文件，但只有 66 个有标注。`SoundscapeDataset` 对缺失文件返回零频谱图。
5. **短音频片段未参与训练**：`TrainAudioDataset` 类存在但未被训练流程使用。
