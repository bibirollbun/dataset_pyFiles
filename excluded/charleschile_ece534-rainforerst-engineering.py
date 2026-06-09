# ============================================================
# Cell 1: Install & Imports & Config
# ============================================================
!pip install timm noisereduce --quiet

import os
import math
import random
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import librosa
import librosa.display

from scipy.signal import butter, filtfilt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from tqdm import tqdm

try:
    import noisereduce as nr
    HAS_NR = True
except ImportError:
    HAS_NR = False
    print("noisereduce not installed, will skip noise reduction.")

class CFG:
    seed = 42
    sample_rate = 32000
    n_mels = 256
    fmin = 80
    fmax = 15000
    duration = 10.0        # seconds
    train_batch_size = 24
    valid_batch_size = 24
    # traning ephoch
    epochs = 15            # 为了在一天内跑完，短一点即可
    lr = 3e-4
    model_name = "tf_efficientnet_b2"
    num_workers = 4
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_subset = True     # 只用子集，适合课程项目
    subset_n = 16941       # 使用多少条样本（可以调）
    train_metadata_path = "/kaggle/input/birdclef-2023/train_metadata.csv"
    audio_dir = "/kaggle/input/birdclef-2023/train_audio"
    model_save_path = "/kaggle/working/tf_efficientnet_b2.pth"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(CFG.seed)
print("Using device:", CFG.device)


# ============================================================
# Cell 2: Preprocessing (bandpass + noise reduction + mel)
# 相当于 preprocessing.py
# ============================================================

def bandpass_filter(data, sr, low=1000, high=12000, order=4):
    """简单带通滤波，突出鸟叫频段"""
    nyq = 0.5 * sr
    low_norm = low / nyq
    high_norm = high / nyq
    b, a = butter(order, [low_norm, high_norm], btype='band')
    filtered = filtfilt(b, a, data)
    return filtered


def apply_noise_reduction(audio, sr):
    """如果有 noisereduce，则做降噪，否则原样返回"""
    if HAS_NR:
        return nr.reduce_noise(y=audio, sr=sr)
    else:
        return audio


def load_audio(path, sr):
    audio, _ = librosa.load(path, sr=sr, mono=True)
    return audio


def crop_or_pad(audio, sr, duration, random_start=True):
    """裁剪或填充到固定时长 duration 秒"""
    target_len = int(sr * duration)
    if len(audio) < target_len:
        pad_len = target_len - len(audio)
        audio = np.concatenate([audio, np.zeros(pad_len, dtype=audio.dtype)])
    elif len(audio) > target_len:
        if random_start:
            start = np.random.randint(0, len(audio) - target_len + 1)
        else:
            start = 0
        audio = audio[start:start + target_len]
    return audio


def audio_to_mel(audio, sr, n_mels, fmin, fmax):
    """audio -> log-mel spectrogram"""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        fmin=fmin,
        fmax=fmax
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # 标准化有助于训练
    mean = mel_db.mean()
    std = mel_db.std() + 1e-6
    mel_db = (mel_db - mean) / std
    return mel_db.astype(np.float32)


def preprocess_audio_segment(audio, sr):
    """完整预处理：带通滤波 + 降噪 + mel"""
    audio = bandpass_filter(audio, sr)
    audio = apply_noise_reduction(audio, sr)
    mel = audio_to_mel(audio, sr, CFG.n_mels, CFG.fmin, CFG.fmax)
    return mel


# ============================================================
# Cell 3: Dataset & Label Encoding
# 相当于 dataset.py
# ============================================================

metadata = pd.read_csv(CFG.train_metadata_path)
print("Total rows in metadata:", len(metadata))

# 为了课程项目，一般只用部分数据（比如 subset_n 条）
if CFG.use_subset:
    # 只保留 rating >= 3 的较高质量样本再采样
    filtered = metadata[metadata['rating'] >= 3.0]
    if len(filtered) >= CFG.subset_n:
        metadata = filtered.sample(CFG.subset_n, random_state=CFG.seed)
    else:
        metadata = filtered
    metadata = metadata.reset_index(drop=True)
    print("Using subset size:", len(metadata))

# label 编码
primary_labels = metadata['primary_label'].unique()
primary_labels = np.sort(primary_labels)
label2id = {label: i for i, label in enumerate(primary_labels)}
id2label = {i: label for label, i in label2id.items()}
num_classes = len(primary_labels)
print("Num classes:", num_classes)


class BirdClefDataset(Dataset):
    def __init__(self, df, audio_dir, sr, duration, is_train=True):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.sr = sr
        self.duration = duration
        self.is_train = is_train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['primary_label']
        label_id = label2id[label]

        # audio path：train_audio/primary_label/filename
        filename = row['filename']
        # species_dir = os.path.join(self.audio_dir, label)
        # file_path = os.path.join(species_dir, filename)
        # 如果 filename 已经自带子目录，不要重复加入 primary_label
        if "/" in filename:
            file_path = os.path.join(self.audio_dir, filename)
        else:
            file_path = os.path.join(self.audio_dir, label, filename)

        # 读取音频
        audio = load_audio(file_path, self.sr)
        # 裁剪/填充到固定时长
        audio = crop_or_pad(audio, self.sr, self.duration, random_start=self.is_train)
        # 预处理 -> mel
        mel = preprocess_audio_segment(audio, self.sr)  # (n_mels, time)

        # 转成 tensor，增加 channel 维度
        mel_tensor = torch.tensor(mel).unsqueeze(0)  # (1, n_mels, time)

        # one-hot 多标签向量（这里只用 primary_label）
        target = np.zeros(num_classes, dtype=np.float32)
        target[label_id] = 1.0
        target_tensor = torch.tensor(target)

        return mel_tensor, target_tensor


# train/valid 划分
from sklearn.model_selection import train_test_split

train_df, valid_df = train_test_split(
    metadata,
    test_size=0.2,
    random_state=CFG.seed
)

train_ds = BirdClefDataset(train_df, CFG.audio_dir, CFG.sample_rate, CFG.duration, is_train=True)
valid_ds = BirdClefDataset(valid_df, CFG.audio_dir, CFG.sample_rate, CFG.duration, is_train=False)

train_loader = DataLoader(
    train_ds,
    batch_size=CFG.train_batch_size,
    shuffle=True,
    num_workers=CFG.num_workers,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_ds,
    batch_size=CFG.valid_batch_size,
    shuffle=False,
    num_workers=CFG.num_workers,
    pin_memory=True
)

print("Train batches:", len(train_loader), "Valid batches:", len(valid_loader))


# ============================================================
# Cell 4: EfficientNet Model
# 相当于 model.py
# ============================================================

class BirdCLEFModel(nn.Module):
    def __init__(self, model_name, num_classes, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=3   # 我们会把单通道 mel repeat 成 3 通道
        )
        if hasattr(self.backbone, "get_classifier"):
            in_features = self.backbone.get_classifier().in_features
            self.backbone.reset_classifier(0)
        else:
            # fallback
            in_features = self.backbone.num_features
            self.backbone.classifier = nn.Identity()
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: (B, 1, H, W) -> (B, 3, H, W)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits


model = BirdCLEFModel(CFG.model_name, num_classes)
model.to(CFG.device)
print("Model created.")


# ============================================================
# Cell 5: Training Loop
# 相当于 train.py
# ============================================================

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=CFG.epochs
)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for mel, target in tqdm(loader, desc="Train", leave=False):
        mel = mel.to(device)       # (B, 1, n_mels, time)
        target = target.to(device) # (B, num_classes)

        optimizer.zero_grad()
        logits = model(mel)
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * mel.size(0)
    epoch_loss = running_loss / len(loader.dataset)
    return epoch_loss


def validate_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds_all = []
    targets_all = []

    with torch.no_grad():
        for mel, target in tqdm(loader, desc="Valid", leave=False):
            mel = mel.to(device)
            target = target.to(device)

            logits = model(mel)
            loss = criterion(logits, target)

            running_loss += loss.item() * mel.size(0)
            preds_all.append(torch.sigmoid(logits).cpu().numpy())
            targets_all.append(target.cpu().numpy())

    epoch_loss = running_loss / len(loader.dataset)
    preds_all = np.concatenate(preds_all, axis=0)
    targets_all = np.concatenate(targets_all, axis=0)

    # 简单计算一个平均 accuracy-like 指标：取 argmax
    pred_labels = preds_all.argmax(axis=1)
    true_labels = targets_all.argmax(axis=1)
    acc = (pred_labels == true_labels).mean()

    return epoch_loss, acc


best_val_loss = np.inf
history = {
    "train_loss": [],
    "valid_loss": [],
    "valid_acc": []
}

for epoch in range(1, CFG.epochs + 1):
    print(f"Epoch {epoch}/{CFG.epochs}")
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, CFG.device)
    valid_loss, valid_acc = validate_one_epoch(model, valid_loader, criterion, CFG.device)
    scheduler.step()

    history["train_loss"].append(train_loss)
    history["valid_loss"].append(valid_loss)
    history["valid_acc"].append(valid_acc)

    print(f"  train_loss: {train_loss:.4f}  valid_loss: {valid_loss:.4f}  valid_acc: {valid_acc:.4f}")

    if valid_loss < best_val_loss:
        best_val_loss = valid_loss
        torch.save(model.state_dict(), CFG.model_save_path)
        print(f"  Saved best model to {CFG.model_save_path}")

print("Training done. Best val loss:", best_val_loss)


# ============================================================
# Cell 6: Simple Visualization (for your report)
# ============================================================

plt.figure(figsize=(8,4))
plt.plot(history["train_loss"], label="train_loss")
plt.plot(history["valid_loss"], label="valid_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.title("Training & Validation Loss")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(history["valid_acc"], marker="o")
plt.xlabel("Epoch")
plt.ylabel("Valid Acc (argmax)")
plt.title("Validation Accuracy-like metric")
plt.show()


# ============================================================
# Cell 7: Inference Helper
# 相当于 inference.py
# ============================================================

def load_trained_model(model_path):
    model = BirdCLEFModel(CFG.model_name, num_classes)
    sd = torch.load(model_path, map_location=CFG.device)
    model.load_state_dict(sd)
    model.to(CFG.device)
    model.eval()
    return model


def predict_on_file(model, file_path, window_sec=5.0, threshold=0.5):
    """
    对一条长音频做滑窗预测，返回每个窗口的 top class 及概率
    """
    sr = CFG.sample_rate
    audio = load_audio(file_path, sr)
    results = []

    win_len = int(window_sec * sr)
    if len(audio) < win_len:
        audio = np.concatenate([audio, np.zeros(win_len - len(audio))])

    # 每个窗口移动 window_sec 秒（可以改成有重叠）
    for start in range(0, len(audio) - win_len + 1, win_len):
        seg = audio[start:start + win_len]
        mel = preprocess_audio_segment(seg, sr)  # (n_mels, time)
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        mel_tensor = mel_tensor.to(CFG.device)

        with torch.no_grad():
            logits = model(mel_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        top_idx = probs.argmax()
        top_prob = probs[top_idx]
        label_name = id2label[top_idx]

        if top_prob >= threshold:
            results.append({
                "start_sec": start / sr,
                "end_sec": (start + win_len) / sr,
                "label": label_name,
                "prob": float(top_prob)
            })

    return results


# ============================================================
# Cell 8: Example inference on one training file
# ============================================================

# 加载最佳模型
best_model = load_trained_model(CFG.model_save_path)

# 从 valid_df 中随便拿一条样本来测试
sample_row = valid_df.iloc[0]
sample_label = sample_row["primary_label"]
sample_filename = sample_row["filename"]
if "/" in sample_filename:
    # filename 带子目录，例如 "gyhspa1/XC610092.ogg"
    sample_path = os.path.join(CFG.audio_dir, sample_filename)
else:
    sample_path = os.path.join(CFG.audio_dir, sample_label, sample_filename)
print("Sample file:", sample_path)

pred_results = predict_on_file(best_model, sample_path, window_sec=CFG.duration, threshold=0.3)
print("Prediction windows:")
for r in pred_results:
    print(r)

with torch.no_grad():
    mel = preprocess_audio_segment(load_audio(sample_path, CFG.sample_rate), CFG.sample_rate)
    mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(CFG.device)
    logits = best_model(mel_tensor)
    probs = torch.sigmoid(logits).cpu().numpy()[0]

top_idx = probs.argmax()
print("Top class:", id2label[top_idx])
print("Top prob:", probs[top_idx])


def predict_on_file(model, file_path, window_sec=5.0, threshold=0.1):
    sr = CFG.sample_rate
    audio = load_audio(file_path, sr)
    results = []

    win_len = int(window_sec * sr)
    if len(audio) < win_len:
        audio = np.concatenate([audio, np.zeros(win_len - len(audio))])

    for start in range(0, len(audio) - win_len + 1, win_len):
        seg = audio[start:start + win_len]

        mel = preprocess_audio_segment(seg, sr)
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(CFG.device)

        with torch.no_grad():
            logits = model(mel_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        top_idx = probs.argmax()
        top_prob = float(probs[top_idx])
        label_name = id2label[top_idx]

        results.append({
            "start_sec": start / sr,
            "end_sec": (start + win_len) / sr,
            "label": label_name,
            "prob": top_prob,
            "is_bird": top_prob >= threshold
        })

    return results

pred_results = predict_on_file(best_model, sample_path, threshold=0.05)
for r in pred_results:
    print(r)


# ============================================================
# Cell 9: Multi-window prediction + visualization
# ============================================================

import matplotlib.pyplot as plt
import numpy as np

def visualize_prediction_windows(model, file_path, window_sec=5.0):
    sr = CFG.sample_rate
    audio = load_audio(file_path, sr)

    win_len = int(window_sec * sr)
    results = []
    probs_all = []
    labels_all = []

    # 分窗预测
    for start in range(0, len(audio) - win_len + 1, win_len):
        seg = audio[start:start + win_len]
        mel = preprocess_audio_segment(seg, sr)
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(CFG.device)

        with torch.no_grad():
            logits = model(mel_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        top_idx = probs.argmax()
        top_label = id2label[top_idx]
        top_prob = float(probs[top_idx])

        results.append((start/sr, (start+win_len)/sr, top_label, top_prob))
        probs_all.append(top_prob)
        labels_all.append(top_label)

    # 打印表格
    print("=== Window Predictions Table ===")
    for r in results:
        print(f"{r[0]:5.1f}s - {r[1]:5.1f}s | {r[2]:10s} | prob={r[3]:.3f}")

    # ---- 图 1：概率折线图 ----
    plt.figure(figsize=(14,5))
    plt.plot(probs_all, marker='o')
    plt.title("Window-level Top1 Probability over Time", fontsize=15)
    plt.xlabel("Window Index")
    plt.ylabel("Probability")
    plt.grid(True)
    plt.show()

    # ---- 图 2：Label 时间线图 ----
    plt.figure(figsize=(14,3))
    plt.plot(labels_all, marker='o')
    plt.title("Window-level Predicted Labels", fontsize=15)
    plt.xlabel("Window Index")
    plt.ylabel("Label")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.show()

    return results


# ============================================================
# Cell 10: Run multi-window visualization on sample audio
# ============================================================

# 随机从验证集取一个例子
sample_row = valid_df.sample(1).iloc[0]
sample_label = sample_row["primary_label"]
sample_filename = sample_row["filename"]

if "/" in sample_filename:
    sample_path = os.path.join(CFG.audio_dir, sample_filename)
else:
    sample_path = os.path.join(CFG.audio_dir, sample_label, sample_filename)

print("Testing file:", sample_path)

# 运行可视化
viz_results = visualize_prediction_windows(best_model, sample_path, window_sec=CFG.duration)



# ============================================================
# Cell 11: Predict multiple audio files and produce a summary table
# ============================================================

def predict_multiple_samples(model, df, num_samples=5):
    rows = df.sample(num_samples)
    table = []

    for _, row in rows.iterrows():
        label = row["primary_label"]
        filename = row["filename"]

        if "/" in filename:
            path = os.path.join(CFG.audio_dir, filename)
        else:
            path = os.path.join(CFG.audio_dir, label, filename)

        # 单次推理
        audio = load_audio(path, CFG.sample_rate)
        mel = preprocess_audio_segment(audio, CFG.sample_rate)
        mel_tensor = torch.tensor(mel).unsqueeze(0).unsqueeze(0).to(CFG.device)

        with torch.no_grad():
            logits = model(mel_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

        top_idx = probs.argmax()
        pred_label = id2label[top_idx]
        pred_prob = float(probs[top_idx])

        table.append([filename, label, pred_label, pred_prob])

    df_table = pd.DataFrame(table,
                            columns=["filename", "true_label", "pred_label", "prob"])
    return df_table

# 运行它
summary_table = predict_multiple_samples(best_model, valid_df, num_samples=10)
summary_table



# ============================================================
# Cell 12: Plot Mel Spectrogram for sample file
# ============================================================

audio = load_audio(sample_path, CFG.sample_rate)
mel = preprocess_audio_segment(audio, CFG.sample_rate)

plt.figure(figsize=(12,4))
librosa.display.specshow(mel, sr=CFG.sample_rate, hop_length=512, x_axis='time', y_axis='mel')
plt.colorbar(label='dB')
plt.title("Mel Spectrogram of Sample Audio")
plt.show()


