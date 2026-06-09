import os
import sys
import glob
import math
import random
from time import time
from pathlib import Path

import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler


ROOT_INPUT = "/kaggle/input"
SR = 32000
DURATION = 4.0
SAMPLES = int(SR * DURATION)
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
FMIN = 20
FMAX = SR // 2
POWER = 2.0

BATCH_SIZE = 32
EPOCHS = 8
LR = 1e-3
NUM_WORKERS = 2
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRINT_EVERY = 20


random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)

print("Device:", DEVICE)
print("Searching under", ROOT_INPUT)


train_csv_path = "/kaggle/input/freesound-audio-tagging/train.csv"
test_csv_path  = "/kaggle/input/freesound-audio-tagging/test_post_competition.csv"
sample_sub_path = "/kaggle/input/freesound-audio-tagging/sample_submission.csv"


train_df = pd.read_csv(train_csv_path)
test_df  = pd.read_csv(test_csv_path)


unique_labels = sorted(train_df['label'].unique().tolist())
LABELS = unique_labels
NUM_CLASSES = len(LABELS)
label_to_idx = {l: i for i, l in enumerate(LABELS)}
print("Extracted labels ({}):".format(NUM_CLASSES), LABELS)


train_audio_dir = "/kaggle/input/freesound-audio-tagging/audio_train"
train_count = 9473
test_audio_dir = "/kaggle/input/freesound-audio-tagging/audio_test"
test_count = 9400


def load_audio(path, sr=SR, samples=SAMPLES):
    try:
        data, orig_sr = sf.read(path, dtype='float32')
    except Exception:
        data, orig_sr = librosa.load(path, sr=None)
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if orig_sr != sr:
        data = librosa.resample(y=data, orig_sr=orig_sr, target_sr=sr)
    if len(data) < samples:
        pad = samples - len(data)
        data = np.pad(data, (0, pad), mode='constant')
    else:
        data = data[:samples]
    return data


def wav_to_log_mel(wav, sr=SR, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=FMIN, fmax=FMAX, power=POWER):
    mel = librosa.feature.melspectrogram(
        y=wav, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax, power=power
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)
    return log_mel.astype(np.float32)


def label_to_multihot_str(label_str):
    arr = np.zeros(NUM_CLASSES, dtype=np.float32)
    if isinstance(label_str, str) and label_str.strip() != '':
        parts = label_str.strip().split()
        for p in parts:
            if p in label_to_idx:
                arr[label_to_idx[p]] = 1.0
            else:
                if p.lower() in {k.lower(): v for k, v in label_to_idx.items()}:
                    # map lower-case back
                    for k, v in label_to_idx.items():
                        if k.lower() == p.lower():
                            arr[v] = 1.0
                            break
                else:
                    # ignore unknown
                    pass
    return arr


class FreesoundDataset(Dataset):
    def __init__(self, df, audio_dir, is_test=False):
        self.df = df.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        fname = row['fname']
        path = os.path.join(self.audio_dir, fname)
        if not os.path.exists(path):
            found = list(Path(self.audio_dir).rglob(fname))
            if found:
                path = str(found[0])
            else:
                raise FileNotFoundError(f"{path} not found")
        wav = load_audio(path)
        feat = wav_to_log_mel(wav)  # shape (n_mels, time)
        feat = (feat - feat.mean()) / (feat.std() + 1e-9)
        x = torch.from_numpy(feat).unsqueeze(0)  # (1, n_mels, time)
        if self.is_test:
            return x.float(), fname
        else:
            lab = row['label']
            y = label_to_multihot_str(lab)
            return x.float(), torch.from_numpy(y)


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        return x


class AudioCNN(nn.Module):
    def __init__(self, n_classes=NUM_CLASSES, in_ch=1):
        super().__init__()
        self.enc = nn.Sequential(
            ConvBlock(in_ch, 16),
            ConvBlock(16, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256, pool=False),
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.enc(x)
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def apk(actual, predicted, k=3):
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            hits += 1.0
            score += hits / (i + 1.0)
    # normalize by min(len(actual), k)
    denom = min(len(actual), k)
    return score / denom if denom > 0 else 0.0


def mapk(actuals, predicteds, k=3):
    return np.mean([apk(a, p, k) for a, p in zip(actuals, predicteds)])


def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    n = 0
    t0 = time()
    for i, (x, y) in enumerate(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        with autocast():
            logits = model(x)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * x.size(0)
        n += x.size(0)
        if i % PRINT_EVERY == 0:
            print(f"Epoch {epoch} iter {i}/{len(loader)} loss {loss.item():.4f}")
    avg = running_loss / n
    print(f"Epoch {epoch} finished in {time()-t0:.0f}s avg_loss={avg:.4f}")
    return avg


def validate_and_map(model, loader):
    model.eval()
    criterion = nn.BCEWithLogitsLoss()
    running_loss = 0.0
    n = 0
    actuals = []
    predicteds = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            loss = criterion(logits, y)
            running_loss += loss.item() * x.size(0)
            n += x.size(0)
            probs = torch.sigmoid(logits).cpu().numpy()
            for row_idx in range(probs.shape[0]):
                p = probs[row_idx]
                topk = np.argsort(p)[-3:][::-1]
                predicted_labels = [LABELS[i] for i in topk]
                predicteds.append(predicted_labels)
            y_np = y.cpu().numpy()
            for row_idx in range(y_np.shape[0]):
                gt_idx = np.where(y_np[row_idx] > 0.5)[0].tolist()
                if len(gt_idx) == 0:
                    actuals.append([])
                else:
                    actuals.append([LABELS[i] for i in gt_idx])
    avg_loss = running_loss / n
    map3 = mapk(actuals, predicteds, k=3)
    print(f"Validation loss {avg_loss:.4f} MAP@3 {map3:.4f}")
    return avg_loss, map3



def predict_test_and_write(model, loader, out_path="submission.csv"):
    model.eval()
    rows = []
    with torch.no_grad():
        for x, fnames in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            for i in range(probs.shape[0]):
                p = probs[i]
                topk = np.argsort(p)[-3:][::-1]
                labels_pred = [LABELS[idx] for idx in topk]
                rows.append((fnames[i], " ".join(labels_pred)))
    sub = pd.DataFrame(rows, columns=["fname", "label"])
    sub.to_csv(out_path, index=False)
    print("Wrote", out_path, "rows:", len(sub))



def main():
    tr_df, val_df = train_test_split(train_df, test_size=0.1, random_state=SEED, shuffle=True, stratify=None)
    tr_df = tr_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    print("Train samples:", len(tr_df), "Val samples:", len(val_df), "Test samples:", len(test_df))

    train_ds = FreesoundDataset(tr_df, train_audio_dir, is_test=False)
    val_ds   = FreesoundDataset(val_df, train_audio_dir, is_test=False)
    test_ds  = FreesoundDataset(test_df, test_audio_dir, is_test=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = AudioCNN(n_classes=NUM_CLASSES, in_ch=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    scaler = GradScaler()

    best_map3 = -1.0
    best_epoch = -1
    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
        val_loss, val_map3 = validate_and_map(model, val_loader)
        scheduler.step(val_loss)
        # save best by map3
        if val_map3 > best_map3:
            best_map3 = val_map3
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model.pth")
            print(f"Saved best_model.pth (epoch {epoch}) MAP@3={val_map3:.4f}")
    print("Training finished. Best epoch:", best_epoch, "best MAP@3:", best_map3)

    # load best and predict test
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
    predict_test_and_write(model, test_loader, out_path="submission.csv")



if __name__ == "__main__":
    main()

