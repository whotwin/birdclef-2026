"""频谱分析脚本 - 分析训练集声景窗口的频谱特征与标签关系

用法:
    # 分析一个标注的声景窗口（从 train_soundscapes_labels.csv）
    python src/analysis/spectrum_analysis.py single --idx 0

    # 对比多个窗口的频谱特征（按标签分组）
    python src/analysis/spectrum_analysis.py compare --n 20

    # 分析每个物种的典型频率分布（从 train_audio 短音频）
    python src/analysis/spectrum_analysis.py species_profiles --species houspa --max_clips 5

    # 统计不同物种在哪些频率段能量最强
    python src/analysis/spectrum_analysis.py freq_distribution --n_windows 50
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # 无 GUI 环境保存图片
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import librosa
import librosa.display

# 配置中文字体（Windows 系统）
CHINESE_FONTS = ["SimHei", "Microsoft YaHei", "STSong", "STKaiti", "KaiTi", "FangSong"]
available_font = next(
    (f for f in CHINESE_FONTS if f in [ff.name for ff in fm.fontManager.ttflist]),
    None
)
if available_font:
    plt.rcParams["font.sans-serif"] = [available_font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题
else:
    # 无中文字体时只用英文
    plt.rcParams["axes.unicode_minus"] = False

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import (
    TRAIN_SOUNDSCAPES_DIR,
    TRAIN_AUDIO_DIR,
    TRAIN_LABELS_CSV,
    TAXONOMY_CSV,
    SAMPLE_SUBMISSION_CSV,
    SAMPLE_RATE,
    WINDOW_DURATION,
    N_MELS,
    N_FFT,
    HOP_LENGTH,
    FMAX,
)


# ─────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────

def load_window_audio(ogg_path: Path, start_sec: float, duration: float = 5.0) -> np.ndarray:
    """从长录音中截取指定时间窗口的音频。"""
    audio, _ = librosa.load(ogg_path, sr=SAMPLE_RATE, offset=start_sec, duration=duration)
    # 不足 5 秒的填充零
    target = int(SAMPLE_RATE * duration)
    if len(audio) < target:
        audio = np.pad(audio, (0, target - len(audio)))
    return audio


def compute_stft(audio: np.ndarray) -> np.ndarray:
    """短时傅里叶变换，返回 magnitude spectrogram。"""
    D = librosa.stft(audio, n_fft=N_FFT, hop_length=HOP_LENGTH)
    return np.abs(D)


def compute_melspec(audio: np.ndarray) -> np.ndarray:
    """Mel 频谱图（dB 尺度）。"""
    spec = librosa.feature.melspectrogram(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=0, fmax=FMAX, power=2.0
    )
    return librosa.power_to_db(spec, ref=np.max)


def freq_band_energy(spec: np.ndarray, freqs: np.ndarray, n_bands: int = 8) -> np.ndarray:
    """将频谱划分为 n_bands 个频率段，返回每段的平均能量。"""
    min_f, max_f = freqs[0], freqs[-1]
    edges = np.linspace(min_f, max_f, n_bands + 1)
    energies = []
    for i in range(n_bands):
        low, high = edges[i], edges[i + 1]
        mask = (freqs >= low) & (freqs < high)
        energies.append(np.mean(spec[mask]))
    return np.array(energies)


def get_taxonomy_name(primary_label: str, taxonomy_df: pd.DataFrame) -> str:
    """根据 primary_label 查学名和俗名。"""
    row = taxonomy_df[taxonomy_df["primary_label"] == primary_label]
    if len(row) == 0:
        return primary_label
    row = row.iloc[0]
    return f"{row.get('common_name', primary_label)} ({row.get('scientific_name', primary_label)})"


# ─────────────────────────────────────────────
# 模式 1: 分析单个标注窗口
# ─────────────────────────────────────────────

def analyze_single(idx: int, output_dir: Path):
    """加载第 idx 个标注窗口，展示多维度频谱分析并保存图片。"""
    labels_df = pd.read_csv(TRAIN_LABELS_CSV)
    taxonomy_df = pd.read_csv(TAXONOMY_CSV)

    row = labels_df.iloc[idx]
    filename = row["filename"]
    start = row["start"]
    species_list = str(row["primary_label"]).split(";") if pd.notna(row["primary_label"]) else []

    # 解析开始时间（秒）
    parts = start.split(":")
    start_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])

    ogg_path = TRAIN_SOUNDSCAPES_DIR / filename
    if not ogg_path.exists():
        print(f"文件不存在: {ogg_path}")
        return

    audio = load_window_audio(ogg_path, start_sec)
    stft = compute_stft(audio)
    mel = compute_melspec(audio)
    mfcc = librosa.feature.mfcc(y=audio, sr=SAMPLE_RATE, n_mfcc=20, n_fft=N_FFT, hop_length=HOP_LENGTH)
    delta_mfcc = librosa.feature.delta(mfcc)

    # STFT 频率轴
    freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
    mel_freqs = librosa.mel_frequencies(n_mels=N_MELS, fmin=0, fmax=FMAX)

    # 每段频率能量
    band_energies = freq_band_energy(stft.mean(axis=1), freqs, n_bands=8)

    # 创建图：5 行 3 列
    fig, axes = plt.subplots(5, 3, figsize=(18, 20))
    fig.suptitle(
        f"声景窗口分析\n文件: {filename}  |  开始: {start}  |  标签: {', '.join(species_list)}",
        fontsize=14, y=1.01
    )

    # 1. 波形图
    t = np.arange(len(audio)) / SAMPLE_RATE
    axes[0, 0].plot(t, audio, linewidth=0.3, color="steelblue")
    axes[0, 0].set_title("波形 (Waveform)")
    axes[0, 0].set_xlabel("时间 (s)")
    axes[0, 0].set_ylabel("振幅")

    # 2. STFT 幅度谱
    librosa.display.specshow(
        librosa.amplitude_to_db(stft, ref=np.max),
        sr=SAMPLE_RATE, hop_length=HOP_LENGTH, x_coords=None,
        x_axis="time", y_axis="hz", ax=axes[0, 1], cmap="viridis"
    )
    axes[0, 1].set_title("STFT 幅度谱 (dB)")
    axes[0, 1].set_ylim(0, 8000)

    # 3. 频率段能量柱状图
    band_labels = [f"{int(i*FMAX/8/1000)}k" for i in range(1, 9)]
    bars = axes[0, 2].bar(band_labels, band_energies, color="steelblue", edgecolor="navy")
    axes[0, 2].set_title("各频段平均能量 (STFT)")
    axes[0, 2].set_xlabel("频率段")
    axes[0, 2].set_ylabel("平均幅度")
    for bar, sp in zip(bars, species_list):
        bar.set_color("coral")
        bar.set_edgecolor("darkred")

    # 4. Mel 频谱图（模型输入）
    img = librosa.display.specshow(
        mel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
        x_axis="time", y_axis="mel", ax=axes[1, 0], cmap="magma"
    )
    plt.colorbar(img, ax=axes[1, 0], format="%+2.0f dB")
    axes[1, 0].set_title("Mel 频谱图 (模型输入, dB)")

    # 5. Mel 频谱按时间平均（能量分布）
    mel_mean = mel.mean(axis=1)
    axes[1, 1].plot(mel_mean, np.arange(N_MELS)[::-1], color="darkorange")
    axes[1, 1].fill_betweenx(np.arange(N_MELS)[::-1], mel_mean, alpha=0.3, color="orange")
    axes[1, 1].set_title("Mel 频谱 - 时间平均能量")
    axes[1, 1].set_xlabel("能量 (dB)")
    axes[1, 1].set_ylabel("Mel 频段")

    # 6. Mel 频谱按频率平均（时间包络）
    axes[1, 2].plot(mel.mean(axis=0), color="purple")
    axes[1, 2].set_title("Mel 频谱 - 频率平均（时间包络）")
    axes[1, 2].set_xlabel("时间帧")
    axes[1, 2].set_ylabel("平均能量 (dB)")

    # 7. MFCC
    librosa.display.specshow(mfcc, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                             x_axis="time", y_axis="hz", ax=axes[2, 0], cmap="coolwarm")
    axes[2, 0].set_title("MFCC (20 系数)")

    # 8. MFCC 时间平均
    for i in range(min(10, mfcc.shape[0])):
        axes[2, 1].plot(mfcc[i] - i * 3, label=f"MFCC {i+1}")
    axes[2, 1].set_title("MFCC 时间平均（偏移显示）")
    axes[2, 1].set_xlabel("时间帧")
    axes[2, 1].legend(fontsize=6, loc="upper right")

    # 9. MFCC delta
    librosa.display.specshow(delta_mfcc, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                             x_axis="time", y_axis="hz", ax=axes[2, 2], cmap="RdBu")
    axes[2, 2].set_title("MFCC Delta（动态特征）")

    # 10-12. 标注物种详情
    species_taxonomy = []
    for sp in species_list[:6]:
        name = get_taxonomy_name(sp, taxonomy_df)
        species_taxonomy.append(f"{sp} → {name}")

    info_text = "\n".join(species_taxonomy) if species_taxonomy else "无标注"
    axes[3, 0].text(0.1, 0.5, f"标注物种 ({len(species_list)} 种):\n\n{info_text}",
                    fontsize=11, verticalalignment="center", family="monospace")
    axes[3, 0].axis("off")
    axes[3, 0].set_title("窗口标签信息")

    # 每帧能量随时间变化
    frame_rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)[0]
    axes[3, 1].plot(frame_rms, color="green")
    axes[3, 1].set_title("每帧 RMS 能量")
    axes[3, 1].set_xlabel("时间帧")
    axes[3, 1].set_ylabel("RMS")

    # 频谱质心（声音亮度）
    spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    t_frames = np.arange(len(spec_centroid)) * HOP_LENGTH / SAMPLE_RATE
    axes[3, 2].plot(t_frames, spec_centroid, color="crimson", linewidth=1.5)
    axes[3, 2].set_title("频谱质心（声音亮度/Hz）")
    axes[3, 2].set_xlabel("时间 (s)")
    axes[3, 2].set_ylabel("Hz")

    # 带宽 & 滚降点
    spec_bw = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]
    spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)[0]

    axes[4, 0].plot(t_frames, spec_bw, label="带宽 (Hz)", color="teal")
    axes[4, 0].plot(t_frames, spec_rolloff, label="滚降点 (Hz)", color="orange", alpha=0.7)
    axes[4, 0].set_title("频谱带宽 & 滚降点")
    axes[4, 0].set_xlabel("时间 (s)")
    axes[4, 0].set_ylabel("Hz")
    axes[4, 0].legend()

    # 频谱对比度（谐波 vs 非谐波）
    spec_contrast = librosa.feature.spectral_contrast(
        y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_bands=6
    )
    librosa.display.specshow(spec_contrast, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                              x_axis="time", ax=axes[4, 1], cmap="plasma")
    axes[4, 1].set_title("频谱对比度 (6 频段)")

    # 过零率
    zcr = librosa.feature.zero_crossing_rate(audio, hop_length=HOP_LENGTH)[0]
    axes[4, 2].plot(zcr, color="navy", linewidth=0.8)
    axes[4, 2].set_title("过零率 (Zero Crossing Rate)")
    axes[4, 2].set_xlabel("时间帧")
    axes[4, 2].set_ylabel("ZCR")

    plt.tight_layout()
    out_path = output_dir / f"single_window_{idx}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_path}")


# ─────────────────────────────────────────────
# 模式 2: 对比多个窗口的频谱特征
# ─────────────────────────────────────────────

def compare_windows(n: int, output_dir: Path):
    """随机抽取 n 个标注窗口，对比频谱特征分布。"""
    labels_df = pd.read_csv(TRAIN_LABELS_CSV)
    taxonomy_df = pd.read_csv(TAXONOMY_CSV)

    sample = labels_df.sample(min(n, len(labels_df)), random_state=42)

    all_centroids = []
    all_bandwidths = []
    all_rolloffs = []
    all_rms = []
    species_count = {}

    for _, row in sample.iterrows():
        filename = row["filename"]
        start = row["start"]
        species_str = str(row["primary_label"]) if pd.notna(row["primary_label"]) else ""

        parts = start.split(":")
        start_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        ogg_path = TRAIN_SOUNDSCAPES_DIR / filename

        if not ogg_path.exists():
            continue

        try:
            audio = load_window_audio(ogg_path, start_sec)
        except Exception:
            continue

        centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
        bw = librosa.feature.spectral_bandwidth(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
        rms = librosa.feature.rms(y=audio, hop_length=HOP_LENGTH)

        all_centroids.append(centroid.mean())
        all_bandwidths.append(bw.mean())
        all_rolloffs.append(rolloff.mean())
        all_rms.append(rms.mean())

        for sp in species_str.split(";"):
            sp = sp.strip()
            if sp:
                species_count[sp] = species_count.get(sp, 0) + 1

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle(f"多窗口频谱统计 (n={len(all_centroids)})", fontsize=14)

    axes[0, 0].hist(all_centroids, bins=30, color="crimson", edgecolor="darkred", alpha=0.7)
    axes[0, 0].set_title("频谱质心分布")
    axes[0, 0].set_xlabel("Hz")
    axes[0, 0].set_ylabel("窗口数量")

    axes[0, 1].hist(all_bandwidths, bins=30, color="teal", edgecolor="darkcyan", alpha=0.7)
    axes[0, 1].set_title("频谱带宽分布")
    axes[0, 1].set_xlabel("Hz")

    axes[0, 2].hist(all_rolloffs, bins=30, color="orange", edgecolor="darkorange", alpha=0.7)
    axes[0, 2].set_title("频谱滚降点分布")
    axes[0, 2].set_xlabel("Hz")

    axes[1, 0].hist(all_rms, bins=30, color="green", edgecolor="darkgreen", alpha=0.7)
    axes[1, 0].set_title("RMS 能量分布")
    axes[1, 0].set_xlabel("RMS 值")

    # 散点：质心 vs 带宽
    axes[1, 1].scatter(all_centroids, all_bandwidths, alpha=0.5, c="purple")
    axes[1, 1].set_title("质心 vs 带宽")
    axes[1, 1].set_xlabel("质心 (Hz)")
    axes[1, 1].set_ylabel("带宽 (Hz)")

    # 物种出现频率
    if species_count:
        top_species = dict(sorted(species_count.items(), key=lambda x: x[1], reverse=True)[:15])
        names = [get_taxonomy_name(sp, taxonomy_df).split("(")[0].strip()[:15] for sp in top_species]
        axes[1, 2].barh(names, list(top_species.values()), color="steelblue", edgecolor="navy")
        axes[1, 2].set_title("最常见标注物种 (Top 15)")
        axes[1, 2].set_xlabel("出现次数")

    plt.tight_layout()
    out_path = output_dir / "compare_windows.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_path}")
    print(f"\n各指标统计:")
    print(f"  质心:    mean={np.mean(all_centroids):.0f}Hz, std={np.std(all_centroids):.0f}Hz")
    print(f"  带宽:    mean={np.mean(all_bandwidths):.0f}Hz, std={np.std(all_bandwidths):.0f}Hz")
    print(f"  滚降点:  mean={np.mean(all_rolloffs):.0f}Hz, std={np.std(all_rolloffs):.0f}Hz")
    print(f"  RMS:     mean={np.mean(all_rms):.4f}, std={np.std(all_rms):.4f}")


# ─────────────────────────────────────────────
# 模式 3: 分析特定物种在短音频中的频率特征
# ─────────────────────────────────────────────

def species_profiles(species: str, max_clips: int, output_dir: Path):
    """加载某物种在 train_audio 中的短音频片段，分析其典型频率特征。"""
    taxonomy_df = pd.read_csv(TAXONOMY_CSV)
    name = get_taxonomy_name(species, taxonomy_df)

    # 找该物种的音频目录
    species_dir = TRAIN_AUDIO_DIR / species
    if not species_dir.exists():
        print(f"物种目录不存在: {species_dir}")
        # 尝试在 taxonomy 中查找
        all_species = [d.name for d in TRAIN_AUDIO_DIR.iterdir() if d.is_dir()]
        # 精确匹配
        matches = [s for s in all_species if s == species]
        if not matches:
            print(f"可用物种目录示例: {all_species[:5]}")
            return
        species_dir = TRAIN_AUDIO_DIR / matches[0]

    clips = list(species_dir.glob("*.ogg"))[:max_clips]
    if not clips:
        print(f"未找到 {species} 的音频文件")
        return

    fig, axes = plt.subplots(len(clips) + 1, 3, figsize=(18, 3 * (len(clips) + 1)))
    fig.suptitle(f"物种频率特征分析: {name} ({species}), {len(clips)} 个片段", fontsize=14)

    all_mel_specs = []
    all_centroids = []
    all_rolloffs = []

    for i, clip_path in enumerate(clips):
        try:
            audio, _ = librosa.load(clip_path, sr=SAMPLE_RATE)
        except Exception as e:
            print(f"  跳过 {clip_path}: {e}")
            continue

        # 固定到 5 秒
        audio = audio[: int(SAMPLE_RATE * 5)]
        if len(audio) < int(SAMPLE_RATE * 5):
            audio = np.pad(audio, (0, int(SAMPLE_RATE * 5) - len(audio)))

        mel = compute_melspec(audio)
        all_mel_specs.append(mel)

        centroid = librosa.feature.spectral_centroid(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
        rolloff = librosa.feature.spectral_rolloff(y=audio, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH)
        all_centroids.append(centroid.mean())
        all_rolloffs.append(rolloff.mean())

        librosa.display.specshow(mel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                 x_axis="time", y_axis="mel", ax=axes[i, 0], cmap="magma")
        axes[i, 0].set_title(f"{clip_path.name}")

        # 时间平均能量
        mel_mean = mel.mean(axis=1)
        axes[i, 1].plot(mel_mean, np.arange(N_MELS)[::-1], color="darkorange")
        axes[i, 1].fill_betweenx(np.arange(N_MELS)[::-1], mel_mean, alpha=0.3)
        axes[i, 1].set_xlim(mel_mean.min() - 2, mel_mean.max() + 2)

        # 质心/滚降
        t_frames = np.arange(len(centroid[0])) * HOP_LENGTH / SAMPLE_RATE
        axes[i, 2].plot(t_frames, centroid[0], label="质心", color="crimson")
        axes[i, 2].plot(t_frames, rolloff[0], label="滚降", color="orange", alpha=0.7)
        axes[i, 2].legend(fontsize=8)
        axes[i, 2].set_ylim(0, 16000)

    # 底部：汇总统计
    if all_mel_specs:
        avg_mel = np.mean(all_mel_specs, axis=0)
        librosa.display.specshow(avg_mel, sr=SAMPLE_RATE, hop_length=HOP_LENGTH,
                                 x_axis="time", y_axis="mel", ax=axes[-1, 0], cmap="magma")
        axes[-1, 0].set_title(f"平均 Mel 频谱 (n={len(all_mel_specs)})")

        avg_mel_mean = avg_mel.mean(axis=1)
        axes[-1, 1].plot(avg_mel_mean, np.arange(N_MELS)[::-1], color="darkred", linewidth=2)
        axes[-1, 1].fill_betweenx(np.arange(N_MELS)[::-1], avg_mel_mean, alpha=0.3, color="red")
        axes[-1, 1].set_title("平均频率能量分布")
        axes[-1, 1].set_xlabel("能量 (dB)")

        axes[-1, 2].bar(["质心 (Hz)", "滚降点 (Hz)"],
                        [np.mean(all_centroids), np.mean(all_rolloffs)],
                        color=["crimson", "orange"], edgecolor="darkred")
        axes[-1, 2].set_title(f"频率统计 (mean)\n质心={np.mean(all_centroids):.0f}Hz  滚降={np.mean(all_rolloffs):.0f}Hz")
        axes[-1, 2].set_ylabel("Hz")

    plt.tight_layout()
    out_path = output_dir / f"species_profile_{species}.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_path}")


# ─────────────────────────────────────────────
# 模式 4: 频段能量与物种标签的统计分析
# ─────────────────────────────────────────────

def freq_label_correlation(n_windows: int, output_dir: Path):
    """对 n_windows 个标注窗口，计算每帧频段能量与物种标签之间的相关性。

    分析思路：
    1. 对每个窗口提取 8 个频段能量
    2. 统计每个物种出现时各频段的平均能量
    3. 找出哪些频段与哪些物种强相关
    """
    labels_df = pd.read_csv(TRAIN_LABELS_CSV)
    taxonomy_df = pd.read_csv(TAXONOMY_CSV)
    species_cols = pd.read_csv(SAMPLE_SUBMISSION_CSV, nrows=0).columns[1:].tolist()  # 234 个物种

    sample = labels_df.sample(min(n_windows, len(labels_df)), random_state=42)

    # 统计: species -> list of band energies
    species_band_energies: dict = {sp: [] for sp in species_cols}

    for _, row in sample.iterrows():
        filename = row["filename"]
        start = row["start"]
        species_str = str(row["primary_label"]) if pd.notna(row["primary_label"]) else ""

        parts = start.split(":")
        start_sec = int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
        ogg_path = TRAIN_SOUNDSCAPES_DIR / filename

        if not ogg_path.exists():
            continue

        try:
            audio = load_window_audio(ogg_path, start_sec)
        except Exception:
            continue

        stft = compute_stft(audio)
        freqs = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT)
        bands = freq_band_energy(stft.mean(axis=1), freqs, n_bands=8)

        present_species = [sp.strip() for sp in species_str.split(";") if sp.strip()]
        for sp in present_species:
            if sp in species_band_energies:
                species_band_energies[sp].append(bands)

    # 计算每个物种在各频段的平均能量
    band_labels = [f"{i*2000}-{(i+1)*2000}Hz" for i in range(8)]
    species_stats = {}
    for sp, energies in species_band_energies.items():
        if len(energies) >= 3:  # 至少 3 个样本
            species_stats[sp] = np.mean(energies, axis=0)

    if not species_stats:
        print("没有足够的样本进行统计分析")
        return

    # 找出最具代表性的物种（频段能量差异最大的）
    def band_variance(band_energies):
        if len(band_energies) < 2:
            return 0
        return np.std(band_energies, axis=0).sum()

    top_species = sorted(species_stats.items(), key=lambda x: band_variance(x[1]), reverse=True)[:20]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"频段能量与物种标签相关性分析 (n={len(sample)} 窗口)", fontsize=14)

    # 图1: Top 物种的频段能量热力图
    top_arr = np.array([v for _, v in top_species])
    top_names = [f"{sp}\n({get_taxonomy_name(sp, taxonomy_df).split('(')[0].strip()[:10]})"
                 for sp, _ in top_species]
    im = axes[0, 0].imshow(top_arr, aspect="auto", cmap="YlOrRd")
    axes[0, 0].set_xticks(range(8))
    axes[0, 0].set_xticklabels(band_labels, rotation=30, ha="right", fontsize=8)
    axes[0, 0].set_yticks(range(len(top_names)))
    axes[0, 0].set_yticklabels(top_names, fontsize=7)
    axes[0, 0].set_title("Top 20 物种 × 频段能量热力图")
    plt.colorbar(im, ax=axes[0, 0], label="平均能量")
    axes[0, 0].set_xlabel("频率段")
    axes[0, 0].set_ylabel("物种")

    # 图2: 物种总数分布（哪些物种最常出现）
    counts = {sp: len(v) for sp, v in species_band_energies.items() if len(v) > 0}
    top_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:15]
    names_c = [f"{get_taxonomy_name(sp, taxonomy_df).split('(')[0].strip()[:12]}" for sp, _ in top_counts]
    axes[0, 1].barh(names_c, [c for _, c in top_counts], color="steelblue", edgecolor="navy")
    axes[0, 1].set_title("最常出现的物种 (Top 15)")
    axes[0, 1].set_xlabel("出现次数")

    # 图3: 每个频段被哪些物种激活最多
    band_species_count = [[] for _ in range(8)]
    for sp, energies in species_band_energies.items():
        if len(energies) >= 1:
            mean_energy = np.mean(energies, axis=0)
            for b in range(8):
                band_species_count[b].append((sp, mean_energy[b]))

    top_per_band = []
    for b in range(8):
        sorted_sp = sorted(band_species_count[b], key=lambda x: x[1], reverse=True)[:3]
        top_per_band.append([f"{sp}" for sp, _ in sorted_sp])

    band_str = [f"Band {i}\n" + "\n".join(top_per_band[i][:2]) for i in range(8)]
    # 显示：每个频段能量最高的物种
    top_sp_per_band = []
    for b in range(8):
        sorted_sp = sorted(band_species_count[b], key=lambda x: x[1], reverse=True)[:3]
        top_sp_per_band.append("\n".join([
            f"{sp}({get_taxonomy_name(sp, taxonomy_df).split('(')[0].strip()[:8]})"
            for sp, _ in sorted_sp
        ]))

    table_data = [[f"Band {i}\n{2000*i}-{2000*(i+1)}Hz", top_sp_per_band[i]] for i in range(8)]
    axes[1, 0].axis("off")
    table = axes[1, 0].table(
        cellText=table_data,
        colLabels=["频率段", "能量最高的物种"],
        cellLoc="left", loc="center"
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    axes[1, 0].set_title("各频段最具代表性物种")

    # 图4: 高频 vs 低频能量分布
    low_freq_energy = [np.mean(v[:, 0]) + np.mean(v[:, 1]) for v in species_stats.values() if len(v) > 1]
    high_freq_energy = [np.mean(v[:, -1]) + np.mean(v[:, -2]) for v in species_stats.values() if len(v) > 1]

    low = np.array([np.mean(v[:2]) for v in species_stats.values() if len(v) > 1])
    high = np.array([np.mean(v[-2:]) for v in species_stats.values() if len(v) > 1])

    axes[1, 1].scatter(low, high, alpha=0.6, c="teal")
    max_val = max(low.max(), high.max()) * 1.1
    axes[1, 1].plot([0, max_val], [0, max_val], "r--", alpha=0.5, label="低频=高频")
    axes[1, 1].set_xlabel("低频能量 (0-4kHz)")
    axes[1, 1].set_ylabel("高频能量 (12-16kHz)")
    axes[1, 1].set_title("物种: 低频 vs 高频能量分布")
    axes[1, 1].legend()

    plt.tight_layout()
    out_path = output_dir / "freq_label_correlation.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"已保存: {out_path}")

    # 打印文字报告
    print(f"\n=== 文字报告 ===")
    print(f"分析窗口数: {len(sample)}")
    print(f"涉及物种数: {len([v for v in species_band_energies.values() if len(v) > 0])}")
    print("\n各频段最具代表性的物种:")
    for b in range(8):
        sorted_sp = sorted(band_species_count[b], key=lambda x: x[1], reverse=True)[:3]
        names_str = ", ".join([f"{sp}({get_taxonomy_name(sp, taxonomy_df).split('(')[0].strip()})" for sp, _ in sorted_sp])
        print(f"  {band_labels[b]}: {names_str}")


# ─────────────────────────────────────────────
# 主入口
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BirdCLEF 2026 频谱分析")
    subparsers = parser.add_subparsers(dest="command", help="分析模式")

    p_single = subparsers.add_parser("single", help="分析单个标注窗口")
    p_single.add_argument("--idx", type=int, default=0, help="窗口索引（从 train_soundscapes_labels.csv）")
    p_single.add_argument("--output", type=str, default="outputs", help="输出目录")

    p_compare = subparsers.add_parser("compare", help="对比多个窗口频谱统计")
    p_compare.add_argument("--n", type=int, default=50, help="采样窗口数量")
    p_compare.add_argument("--output", type=str, default="outputs", help="输出目录")

    p_profile = subparsers.add_parser("species_profiles", help="分析特定物种在短音频中的频率特征")
    p_profile.add_argument("--species", type=str, required=True, help="物种 ID (primary_label)")
    p_profile.add_argument("--max_clips", type=int, default=5, help="最多分析几个片段")
    p_profile.add_argument("--output", type=str, default="outputs", help="输出目录")

    p_corr = subparsers.add_parser("freq_distribution", help="频段能量与物种标签统计分析")
    p_corr.add_argument("--n_windows", type=int, default=100, help="采样窗口数量")
    p_corr.add_argument("--output", type=str, default="outputs", help="输出目录")

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True, parents=True)

    if args.command == "single":
        analyze_single(args.idx, output_dir)
    elif args.command == "compare":
        compare_windows(args.n, output_dir)
    elif args.command == "species_profiles":
        species_profiles(args.species, args.max_clips, output_dir)
    elif args.command == "freq_distribution":
        freq_label_correlation(args.n_windows, output_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
