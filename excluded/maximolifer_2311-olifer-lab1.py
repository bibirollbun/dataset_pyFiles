from pathlib import Path
DATA = Path("/kaggle/input/freesound-audio-tagging")

TRAIN_DIR = DATA / "audio_train"
TEST_DIR  = DATA / "audio_test"
SAMPLE_SUB = DATA / "sample_submission.csv"



from dataclasses import dataclass
import random
import numpy as np
import torch

@dataclass
class CFG:
    seed: int = 42

    # audio
    sr: int = 32000
    duration: float = 4.0
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 320
    fmin: int = 20
    fmax: int = 16000

    # train
    train_bs: int = 32
    valid_bs: int = 64
    epochs: int = 10
    lr: float = 3e-4
    wd: float = 1e-2
    num_workers: int = 2

    # aug
    specaug_p: float = 0.4

    # noisy handling
    noisy_weight: float = 0.5
    noisy_label_smooth: float = 0.10

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)
print("torch:", torch.__version__, "| device:", CFG.device)



import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def read_meta(DATA: Path):
    curated = DATA / "train_curated.csv"
    noisy   = DATA / "train_noisy.csv"
    train   = DATA / "train.csv"

    if curated.exists() and noisy.exists():
        df_c = pd.read_csv(curated); df_c["is_noisy"] = 0
        df_n = pd.read_csv(noisy);   df_n["is_noisy"] = 1
        df = pd.concat([df_c, df_n], ignore_index=True)
        mode = "curated_noisy"
    else:
        df = pd.read_csv(train)
        if "manually_verified" in df.columns:
            df["is_noisy"] = (df["manually_verified"] == 0).astype(int)
        else:
            df["is_noisy"] = 0
        mode = "train_csv"
    return df, mode

df, split_mode = read_meta(DATA)

df["filepath"] = df["fname"].apply(lambda x: str(TRAIN_DIR / x))
df = df[df["filepath"].map(os.path.exists)].reset_index(drop=True)

le = LabelEncoder()
df["label_idx"] = le.fit_transform(df["label"])
CLASSES = le.classes_.tolist()
num_classes = len(CLASSES)

# val — только clean
clean_df = df[df["is_noisy"] == 0].copy()
train_clean, valid_df = train_test_split(
    clean_df, test_size=0.1, random_state=CFG.seed, stratify=clean_df["label_idx"]
)

# train — clean_train + noisy
train_df = pd.concat([train_clean, df[df["is_noisy"] == 1]], ignore_index=True)

print("mode:", split_mode)
print("classes:", num_classes)
print("train:", len(train_df), "| val:", len(valid_df), "| noisy_share_train:", train_df["is_noisy"].mean())
print("noisy_share_val:", valid_df["is_noisy"].mean())



import soundfile as sf
import librosa

TARGET_SAMPLES = int(CFG.sr * CFG.duration)

def load_audio_mono(path: str, target_sr=CFG.sr) -> np.ndarray:
    wav, sr = sf.read(path, always_2d=False)
    wav = wav.astype(np.float32)
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav

def fix_length(wav: np.ndarray, target=TARGET_SAMPLES, train=True) -> np.ndarray:
    if len(wav) < target:
        wav = np.pad(wav, (0, target - len(wav)), mode="constant")
    elif len(wav) > target:
        start = np.random.randint(0, len(wav) - target + 1) if train else (len(wav) - target) // 2
        wav = wav[start:start + target]
    return wav

def wav_to_logmel(wav: np.ndarray) -> np.ndarray:
    m = librosa.feature.melspectrogram(
        y=wav, sr=CFG.sr,
        n_fft=CFG.n_fft, hop_length=CFG.hop_length,
        n_mels=CFG.n_mels, fmin=CFG.fmin, fmax=CFG.fmax, power=2.0
    )
    logm = librosa.power_to_db(m, ref=np.max)
    logm = (logm - logm.mean()) / (logm.std() + 1e-6)  # per-sample z-score
    return logm.astype(np.float32)

def spec_augment(spec: np.ndarray, max_mask_pct=0.1, num_masks=2) -> np.ndarray:
    spec = spec.copy()
    n_mels, n_steps = spec.shape
    for _ in range(num_masks):
        if np.random.rand() < 0.5:
            f = max(1, int(max_mask_pct * n_mels))
            f0 = np.random.randint(0, max(1, n_mels - f))
            spec[f0:f0+f, :] = 0
        else:
            t = max(1, int(max_mask_pct * n_steps))
            t0 = np.random.randint(0, max(1, n_steps - t))
            spec[:, t0:t0+t] = 0
    return spec



from torch.utils.data import Dataset, DataLoader

class FSDKDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train=True):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        wav = load_audio_mono(row.filepath)
        wav = fix_length(wav, TARGET_SAMPLES, train=self.train)
        spec = wav_to_logmel(wav)

        if self.train and np.random.rand() < CFG.specaug_p:
            spec = spec_augment(spec)

        x = torch.from_numpy(spec).unsqueeze(0)  # [1, n_mels, T]
        y = torch.tensor(row.label_idx).long()

        is_noisy = int(row.is_noisy)
        w = CFG.noisy_weight if is_noisy else 1.0

        return x, y, torch.tensor(w).float(), torch.tensor(is_noisy).long()

train_loader = DataLoader(FSDKDataset(train_df, True), batch_size=CFG.train_bs, shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
valid_loader = DataLoader(FSDKDataset(valid_df, False), batch_size=CFG.valid_bs, shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True)

print("batches:", len(train_loader), len(valid_loader))



import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )
    def forward(self, x): 
        return self.block(x)

class CnnSpec(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 32), ConvBlock(32, 64),
            ConvBlock(64, 128), ConvBlock(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)

model = CnnSpec(num_classes).to(CFG.device)
print("model:", model.__class__.__name__)



import torch.nn.functional as F
import torch

def ce_smooth(logits, y, smoothing=0.0):
    if smoothing <= 0:
        return F.cross_entropy(logits, y, reduction="none")
    n = logits.size(1)
    logp = F.log_softmax(logits, dim=1)
    with torch.no_grad():
        true = torch.zeros_like(logp).fill_(smoothing / (n - 1))
        true.scatter_(1, y.unsqueeze(1), 1 - smoothing)
    return -(true * logp).sum(dim=1)

def batch_loss(logits, y, w, is_noisy):
    is_noisy = is_noisy.bool()
    loss = torch.zeros_like(w)

    if (~is_noisy).any():
        loss[~is_noisy] = ce_smooth(logits[~is_noisy], y[~is_noisy], smoothing=0.0)
    if is_noisy.any():
        loss[is_noisy] = ce_smooth(logits[is_noisy], y[is_noisy], smoothing=CFG.noisy_label_smooth)

    return (loss * w).mean()



import numpy as np

def mapk3_score(y_true, y_pred_logits, k=3):
    preds = np.argsort(-y_pred_logits, axis=1)[:, :k]
    gains = []
    for t, p in zip(y_true, preds):
        if t in p:
            rank = np.where(p == t)[0][0] + 1
            gains.append(1.0 / rank)
        else:
            gains.append(0.0)
    return float(np.mean(gains))

# torch 2.6+ AMP
scaler = torch.amp.GradScaler('cuda', enabled=(CFG.device=="cuda"))

optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)

def run_epoch(loader, train=True):
    model.train(train)
    total_loss = 0.0
    y_true, y_pred = [], []

    for x, y, w, is_noisy in loader:
        x = x.to(CFG.device, non_blocking=True)
        y = y.to(CFG.device, non_blocking=True)
        w = w.to(CFG.device, non_blocking=True)
        is_noisy = is_noisy.to(CFG.device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=(CFG.device=="cuda")):
            logits = model(x)
            loss = batch_loss(logits, y, w, is_noisy)

        if train:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item() * x.size(0)
        y_true.append(y.detach().cpu().numpy())
        y_pred.append(logits.detach().cpu().numpy())

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    return total_loss / len(loader.dataset), mapk3_score(y_true, y_pred, 3)

best = -1.0
for epoch in range(1, CFG.epochs + 1):
    tr_loss, tr_map3 = run_epoch(train_loader, True)
    va_loss, va_map3 = run_epoch(valid_loader, False)
    scheduler.step()

    print(f"Epoch {epoch:02d} | tr_loss {tr_loss:.4f} map3 {tr_map3:.4f} | va_loss {va_loss:.4f} map3 {va_map3:.4f}")

    if va_map3 > best:
        best = va_map3
        torch.save({"model": model.state_dict(), "classes": CLASSES}, "best_model.pt")

print("Best val mAP@3:", best)



def make_crops(wav: np.ndarray, target: int):
    if len(wav) <= target:
        return [fix_length(wav, target, train=False)]
    starts = [0, (len(wav) - target)//2, len(wav) - target]
    return [wav[s:s+target] for s in starts]

@torch.no_grad()
def predict_file_tta(path: str) -> str:
    wav = load_audio_mono(path)
    crops = make_crops(wav, TARGET_SAMPLES)

    logits_all = []
    for c in crops:
        spec = wav_to_logmel(c)
        x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(CFG.device)  # [1,1,M,T]
        logits = model(x).float().cpu().numpy()[0]
        logits_all.append(logits)

    logits_mean = np.mean(logits_all, axis=0)
    top3 = np.argsort(-logits_mean)[:3]
    return " ".join([CLASSES[i] for i in top3])



state = torch.load("best_model.pt", map_location=CFG.device)
model.load_state_dict(state["model"])
CLASSES = state["classes"]
model.eval()

sub = pd.read_csv(SAMPLE_SUB)

sub["label"] = [predict_file_tta(str(TEST_DIR / f)) for f in sub["fname"]]
sub.to_csv("submission.csv", index=False)

print("saved:", "submission.csv")
sub.head()


