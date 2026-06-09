import os
from pathlib import Path
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import math
import json

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import torchaudio
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights



DATA_ROOT = Path("/kaggle/input/birdclef-2021")
OUT_DIR = Path("./output")
CACHE_DIR = OUT_DIR / "mel_cache"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
OUT_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
CHECKPOINT_DIR.mkdir(exist_ok=True)
WEIGHTS_PATH = (
    "/kaggle/input/efficientnet-b0-imagenet-weights/"
    "efficientnet_b0_imagenet.pth"
)


SAMPLE_RATE = 32000
N_MELS = 128
CLIP_DURATION = 10
CLIP_SAMPLES = SAMPLE_RATE * CLIP_DURATION
HOP_LENGTH = 512
N_FFT = 2048

BATCH_SIZE = 32
EPOCHS = 8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42


train_meta = pd.read_csv(DATA_ROOT / "train_metadata.csv")
print("Train metadata:", train_meta.shape)
species = sorted(train_meta["primary_label"].unique())
label2idx = {lbl: i for i, lbl in enumerate(species)}
idx2label = {i: lbl for lbl, i in label2idx.items()}
NUM_CLASSES = len(species)
print("Num species:", NUM_CLASSES)


def wav_to_log_mel(wave: np.ndarray, sr=SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH):
    mel = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_db = (mel_db + 80.0) / 80.0
    return mel_db.astype(np.float32)


def cache_mel_for_file(filepath: Path, out_path: Path):
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    total_samples = len(y)
    n_segments = max(1, total_samples // CLIP_SAMPLES)
    saved = []
    for i in range(n_segments):
        start = i * CLIP_SAMPLES
        clip = y[start:start + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES:
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        mel = wav_to_log_mel(clip)
        seg_path = out_path / f"{filepath.stem}_seg{i}.npy"
        if not seg_path.exists():
            np.save(seg_path, mel)
        saved.append(seg_path)
    return saved



class ShortAudioDataset(Dataset):
    def __init__(self, df_meta, data_root, label2idx,
                 cache_dir=CACHE_DIR, training=True):
        self.df = df_meta.reset_index(drop=True)
        self.data_root = Path(data_root) / "train_short_audio"
        self.label2idx = label2idx
        self.training = training
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=16)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=24)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        species = row["primary_label"]
        filename = row["filename"]

        audio_path = self.data_root / species / filename
        if not audio_path.exists():
            found = list(self.data_root.rglob(filename))
            if found:
                audio_path = found[0]
            else:
                raise FileNotFoundError(audio_path)

        mel_cache = self.cache_dir / f"{audio_path.stem}.npy"

        if not mel_cache.exists():
            y, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

            if len(y) < CLIP_SAMPLES:
                y = np.pad(y, (0, CLIP_SAMPLES - len(y)))
            else:
                if self.training:
                    start = np.random.randint(0, len(y) - CLIP_SAMPLES)
                    y = y[start:start + CLIP_SAMPLES]
                else:
                    y = y[:CLIP_SAMPLES]

            mel = wav_to_log_mel(y)
            np.save(mel_cache, mel)
        else:
            mel = np.load(mel_cache)

        mel = torch.from_numpy(mel).unsqueeze(0)  # (1, n_mels, T)

        if self.training:
            if torch.rand(1).item() < 0.5:
                mel = self.freq_mask(mel)
            if torch.rand(1).item() < 0.5:
                mel = self.time_mask(mel)

        label = self.label2idx[row["primary_label"]]
        return mel, label



import timm
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0

WEIGHTS_PATH = (
    "/kaggle/input/efficientnet-b0-imagenet-weights/efficientnet_b0_imagenet.pth"
)

class EfficientNetWrapper(nn.Module):
    def __init__(self, out_dim=512):
        super().__init__()

        self.net = efficientnet_b0(weights=None)

        state_dict = torch.load(WEIGHTS_PATH, map_location="cpu")
        self.net.load_state_dict(state_dict, strict=False)

        old_conv = self.net.features[0][0]
        self.net.features[0][0] = nn.Conv2d(
            in_channels=1,
            out_channels=old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )

        self.net.classifier = nn.Identity()

        self.head = nn.Sequential(
            nn.Linear(1280, out_dim),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x



USE_TIMM = True


class BirdClassifier(nn.Module):
    def __init__(self, backbone, out_dim=512, n_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = backbone
        self.proj = nn.Linear(out_dim, n_classes)

    def forward(self, x):
        feats = self.backbone(x)  # (B, out_dim)
        logits = self.proj(feats)
        return logits


train_df, valid_df = train_test_split(train_meta, test_size=0.2, random_state=SEED, stratify=train_meta["primary_label"])

train_ds = ShortAudioDataset(train_df, DATA_ROOT, label2idx, training=True)
valid_ds = ShortAudioDataset(valid_df, DATA_ROOT, label2idx, training=True)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)



backbone = EfficientNetWrapper(out_dim=512)
model = BirdClassifier(
    backbone,
    out_dim=512,
    n_classes=NUM_CLASSES
).to(DEVICE)

for p in model.backbone.parameters():
    p.requires_grad = False

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)


best_f1 = 0.0
for epoch in range(1, EPOCHS + 1):
    if (epoch == 3):
        for p in model.backbone.parameters():
            p.requires_grad = True

    model.train()
    losses = []
    for x, y in tqdm(train_loader, desc=f"Train {epoch}"):
        x = x.to(DEVICE)
        y = y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    avg_loss = np.mean(losses)

    # валидация
    model.eval()
    preds = []
    targs = []
    with torch.no_grad():
        for x, y in tqdm(valid_loader, desc=f"Val {epoch}"):
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            logits = model(x)
            pred = logits.argmax(dim=1)
            preds.extend(pred.cpu().numpy())
            targs.extend(y.cpu().numpy())
    val_f1 = f1_score(targs, preds, average='macro')
    print(f"Epoch {epoch}: train_loss={avg_loss:.4f} val_f1={val_f1:.4f}")
    scheduler.step(val_f1)
    if val_f1 > best_f1:
        best_f1 = val_f1
        torch.save(model.state_dict(), CHECKPOINT_DIR / f"best_epoch{epoch}_f1{val_f1:.4f}.pt")
print("Best val f1:", best_f1)

TEST_SSG_DIR = DATA_ROOT / "test_soundscapes"

TOP_K = 3



def infer_soundscape_file(model, filepath: Path, top_k=TOP_K):
    # возвращает список (end_time_sec, [label_codes...])
    y, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    total_seconds = math.ceil(len(y) / SAMPLE_RATE)
    results = []
    n_segments = int(len(y) // CLIP_SAMPLES)
    for i in range(n_segments):
        start = i * CLIP_SAMPLES
        clip = y[start:start + CLIP_SAMPLES]
        if len(clip) < CLIP_SAMPLES:
            clip = np.pad(clip, (0, CLIP_SAMPLES - len(clip)))
        mel = wav_to_log_mel(clip)          # (n_mels, T)
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,n_mels,T)
        with torch.no_grad():
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        top_idx = probs.argsort()[-top_k:][::-1]
        top_labels = [idx2label[i] for i in top_idx]
        end_time = (i+1) * CLIP_DURATION   # конец окна в секундах
        results.append((end_time, top_labels))
    return results



submission_rows = []
for filepath in tqdm(sorted(TEST_SSG_DIR.glob("*.ogg")), desc="Soundscapes"):
    soundscape_id = filepath.stem  # обычно формат like "12345_SSW_20170429" или id — для row_id обычно нужен только часть перед расширением.
    preds = infer_soundscape_file(model, filepath, top_k=TOP_K)
    for end_time, labels in preds:
        # row_id по формату соревнования: soundscape_[soundscape_id]_[end_time]
        row_id = f"soundscape_{soundscape_id}_{end_time}"
        # birds — пробел разделённые коды (или 'nocall' если пусто)
        birds = " ".join(labels) if labels else "nocall"
        submission_rows.append({"row_id": row_id, "birds": birds})

submission_df = pd.DataFrame(submission_rows)
submission_df.to_csv("submission.csv", index=False)
print("Saved submission:", "submission.csv")

