import os, random, math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# аудио
import soundfile as sf
import librosa

# модель (spectrogram-as-image)
try:
    import timm
except Exception:
    timm = None

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder



@dataclass
class CFG:
    seed: int = 42

    # audio
    sr: int = 32000               # ресемплим с 44.1kHz → 32kHz для скорости
    clip_seconds: float = 5.0     # фиксированная длина окна (train + TTA)
    n_mels: int = 128
    fmin: int = 20
    fmax: int = 16000

    # spectrogram
    n_fft: int = 1024
    hop_length: int = 320         # 10ms @32kHz
    win_length: int = 1024

    # train
    batch_size: int = 32
    num_workers: int = 2
    epochs: int = 8
    lr: float = 3e-4
    weight_decay: float = 1e-2
    label_smoothing_noisy: float = 0.10  # для noisy (manually_verified=0)
    noisy_weight: float = 0.5           # уменьшаем вклад noisy в loss

    # model
    backbone: str = "efficientnet_b0"    # timm
    pretrained: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(CFG.seed)

# Kaggle: /kaggle/input/freesound-audio-tagging/
ROOT = Path("/kaggle/input/freesound-audio-tagging")
if not ROOT.exists():
    ROOT = Path("./freesound-audio-tagging")  # локально

TRAIN_AUDIO = ROOT / "audio_train"
TEST_AUDIO  = ROOT / "audio_test"
TRAIN_CSV   = ROOT / "train.csv"
SAMPLE_SUB  = ROOT / "sample_submission.csv"

print("ROOT:", ROOT)
print("train.csv exists:", TRAIN_CSV.exists())
print("audio_train exists:", TRAIN_AUDIO.exists())
print("audio_test exists:", TEST_AUDIO.exists())



df = pd.read_csv(TRAIN_CSV)
display(df.head())
print(df.columns.tolist())
print("rows:", len(df))
print(df['manually_verified'].value_counts(dropna=False))

label_counts = df['label'].value_counts()
display(label_counts.head(10))
print("num_classes:", df['label'].nunique())



def audio_info(path: Path):
    y, sr = sf.read(path)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return len(y)/sr, sr

sample_files = df.sample(5, random_state=CFG.seed)['fname'].tolist()
for fn in sample_files:
    dur, sr0 = audio_info(TRAIN_AUDIO / fn)
    print(fn, "dur_s=", round(dur, 3), "sr=", sr0)



def load_audio_mono(path: Path, target_sr: int) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y

def crop_or_pad(y: np.ndarray, target_len: int, train: bool) -> np.ndarray:
    if len(y) >= target_len:
        if train:
            start = np.random.randint(0, len(y) - target_len + 1)
        else:
            start = (len(y) - target_len) // 2
        y = y[start:start+target_len]
    else:
        pad = target_len - len(y)
        y = np.pad(y, (0, pad), mode="constant")
    return y

def log_mel(y: np.ndarray) -> np.ndarray:
    S = librosa.feature.melspectrogram(
        y=y,
        sr=CFG.sr,
        n_fft=CFG.n_fft,
        hop_length=CFG.hop_length,
        win_length=CFG.win_length,
        n_mels=CFG.n_mels,
        fmin=CFG.fmin,
        fmax=CFG.fmax,
        power=2.0
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    S_db = (S_db - S_db.min()) / (S_db.max() - S_db.min() + 1e-6)  # [0,1]
    return S_db.astype(np.float32)



le = LabelEncoder()
df['target'] = le.fit_transform(df['label'])
num_classes = len(le.classes_)
print("classes:", num_classes)

verified = df[df['manually_verified'] == 1].copy()
train_v_idx, val_idx = train_test_split(
    verified.index, test_size=0.20, random_state=CFG.seed, stratify=verified['target']
)

train_df = pd.concat([df.loc[train_v_idx], df[df['manually_verified'] == 0]], ignore_index=True)
val_df   = df.loc[val_idx].reset_index(drop=True)

print("train:", len(train_df), "val:", len(val_df))
print("val manually_verified unique:", val_df['manually_verified'].unique())

class FreesoundDataset(Dataset):
    def __init__(self, meta: pd.DataFrame, audio_dir: Path, train: bool):
        self.meta = meta.reset_index(drop=True)
        self.audio_dir = audio_dir
        self.train = train
        self.target_len = int(CFG.sr * CFG.clip_seconds)

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, i):
        row = self.meta.iloc[i]
        y = load_audio_mono(self.audio_dir / row.fname, CFG.sr)
        y = crop_or_pad(y, self.target_len, train=self.train)

        # простые waveform-ауги (можно расширять)
        if self.train:
            if np.random.rand() < 0.5:
                y = y * (0.7 + 0.6*np.random.rand())  # random gain
            if np.random.rand() < 0.3:
                y = y + 0.005*np.random.randn(len(y)).astype(np.float32)  # noise

        x = log_mel(y)                         # [mels, time]
        x = torch.from_numpy(x).unsqueeze(0)   # [1, mels, time]

        target = int(row.target)
        is_noisy = (int(row.manually_verified) == 0)
        sample_weight = CFG.noisy_weight if is_noisy else 1.0

        return x, target, sample_weight, is_noisy

train_ds = FreesoundDataset(train_df, TRAIN_AUDIO, train=True)
val_ds   = FreesoundDataset(val_df, TRAIN_AUDIO, train=False)

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_ds, batch_size=CFG.batch_size*2, shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True)



import torch
import timm

WEIGHTS = "/kaggle/input/efficientnet-b0/pytorch/default/1/efficientnet_b0_ra_in1k.pth"

# 1) создаём модель БЕЗ checkpoint_path (иначе strict загрузка и падение)
model = timm.create_model(
    "efficientnet_b0.ra_in1k",
    pretrained=False,
    in_chans=1,
    num_classes=num_classes,   # 41
).to(CFG.device)

# 2) читаем веса
ckpt = torch.load(WEIGHTS, map_location="cpu")
state = ckpt.get("state_dict", ckpt)

# 3) на всякий: убираем префиксы module./model.
new_state = {}
for k, v in state.items():
    if k.startswith("module."):
        k = k[len("module."):]
    if k.startswith("model."):
        k = k[len("model."):]
    new_state[k] = v
state = new_state

# 4) адаптируем первый conv: RGB -> 1 канал (среднее по каналам)
if "conv_stem.weight" in state:
    w = state["conv_stem.weight"]  # [32,3,3,3]
    if w.ndim == 4 and w.shape[1] == 3:
        state["conv_stem.weight"] = w.mean(dim=1, keepdim=True)  # -> [32,1,3,3]

# 5) выкидываем голову ImageNet (1000 классов)
state.pop("classifier.weight", None)
state.pop("classifier.bias", None)

# 6) грузим нестрого
missing, unexpected = model.load_state_dict(state, strict=False)
print("missing:", len(missing))
print("unexpected:", len(unexpected))
print("пример missing:", missing[:5])



@torch.no_grad()
def map_at_3(probs: torch.Tensor, targets: torch.Tensor) -> float:
    top3 = probs.topk(3, dim=1).indices  # [N,3]
    score = 0.0
    for i in range(len(targets)):
        t = targets[i].item()
        hits = (top3[i] == t).nonzero(as_tuple=False)
        if len(hits) > 0:
            rank = int(hits[0].item())  # 0/1/2
            score += 1.0 / (rank + 1)
    return score / len(targets)



def ce_with_optional_smoothing(logits, target, smoothing: float):
    if smoothing <= 0:
        return F.cross_entropy(logits, target, reduction="none")
    n_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        true_dist = torch.zeros_like(log_probs)
        true_dist.fill_(smoothing / (n_classes - 1))
        true_dist.scatter_(1, target.unsqueeze(1), 1.0 - smoothing)
    return -(true_dist * log_probs).sum(dim=1)

def train_one_epoch(model, loader, optimizer, scaler):
    model.train()
    losses = []
    for x, y, w, is_noisy in loader:
        x = x.to(CFG.device, non_blocking=True)
        y = y.to(CFG.device, non_blocking=True)
        w = w.to(CFG.device, non_blocking=True).float()
        is_noisy = is_noisy.to(CFG.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(CFG.device=="cuda")):
            logits = model(x)

            loss_vec = torch.zeros_like(w)
            mask_noisy = is_noisy.bool()
            mask_clean = ~mask_noisy

            if mask_clean.any():
                loss_vec[mask_clean] = ce_with_optional_smoothing(logits[mask_clean], y[mask_clean], smoothing=0.0)
            if mask_noisy.any():
                loss_vec[mask_noisy] = ce_with_optional_smoothing(
                    logits[mask_noisy], y[mask_noisy], smoothing=CFG.label_smoothing_noisy
                )

            loss = (loss_vec * w).mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())

    return float(np.mean(losses))

@torch.no_grad()
def validate(model, loader):
    model.eval()
    all_probs, all_t = [], []
    losses = []
    for x, y, w, is_noisy in loader:
        x = x.to(CFG.device, non_blocking=True)
        y = y.to(CFG.device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=(CFG.device=="cuda")):
            logits = model(x)
            loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits.float(), dim=1).cpu()
        all_probs.append(probs)
        all_t.append(y.cpu())
        losses.append(loss.item())

    probs = torch.cat(all_probs, dim=0)
    t = torch.cat(all_t, dim=0)
    score = map_at_3(probs, t)
    return float(np.mean(losses)), float(score)

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)
scaler = torch.cuda.amp.GradScaler(enabled=(CFG.device=="cuda"))

best_score = -1
best_path = "best_model.pt"

for epoch in range(1, CFG.epochs+1):
    tr_loss = train_one_epoch(model, train_loader, optimizer, scaler)
    va_loss, va_map3 = validate(model, val_loader)
    scheduler.step()

    if va_map3 > best_score:
        best_score = va_map3
        torch.save({"model": model.state_dict(), "classes": le.classes_}, best_path)

    print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} | val_loss {va_loss:.4f} | val_mAP@3 {va_map3:.4f} | best {best_score:.4f}")

print("Best saved to:", best_path)



@torch.no_grad()
def predict_file(path: Path, model: nn.Module) -> np.ndarray:
    model.eval()
    y = load_audio_mono(path, CFG.sr)
    target_len = int(CFG.sr * CFG.clip_seconds)

    crops = []
    if len(y) <= target_len:
        crops = [crop_or_pad(y, target_len, train=False)]
    else:
        starts = [0, (len(y)-target_len)//2, len(y)-target_len]
        for st in starts:
            crops.append(y[st:st+target_len])

    probs = []
    for c in crops:
        x = torch.from_numpy(log_mel(c)).unsqueeze(0).unsqueeze(0).to(CFG.device)  # [1,1,mels,time]
        logits = model(x)
        p = F.softmax(logits.float(), dim=1).cpu().numpy()[0]
        probs.append(p)

    return np.mean(probs, axis=0)

ckpt = torch.load(best_path, map_location=CFG.device, weights_only=False)
model.load_state_dict(ckpt["model"])
classes = ckpt.get("classes", None)




sub = pd.read_csv(SAMPLE_SUB)
test_files = sub['fname'].tolist()

pred_labels = []
for fn in test_files:
    p = predict_file(TEST_AUDIO / fn, model)
    top3 = p.argsort()[::-1][:3]
    pred_labels.append(" ".join(le.inverse_transform(top3)))

sub['label'] = pred_labels
sub.to_csv("submission.csv", index=False)
sub.head()


