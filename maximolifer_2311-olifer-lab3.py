# === (опционально) timm ===
try:
    import timm
except ImportError:
    !pip -q install timm
    import timm

import os, random, math, warnings
from dataclasses import dataclass
from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.functional as AF

warnings.filterwarnings("ignore")

print("torch:", torch.__version__, "| cuda:", torch.cuda.is_available())
print("torchaudio:", torchaudio.__version__)

@dataclass
class CFG:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # audio
    sr: int = 32000
    win_sec: float = 8.0          # окно обучения (сек)
    hop_sec_test: float = 2.0     # шаг на тесте (сек) - меньше => лучше, но медленнее

    # mel
    n_mels: int = 128
    n_fft: int = 1024
    hop_length: int = 320
    fmin: int = 20
    fmax: int = 16000

    # sampling
    fp_ratio: float = 2.0         # сколько FP берем относительно TP
    bg_ratio: float = 1.0         # сколько background окон относительно TP

    # training
    train_bs: int = 32
    valid_bs: int = 64
    epochs: int = 12
    lr: float = 3e-4
    wd: float = 1e-2
    num_workers: int = 2

    # regularization
    specaug_p: float = 0.5
    fp_focus: float = 3.0         # усиление лосса по FP-классу

    # model
    backbone: str = "tf_efficientnet_b0.ns_jft_in1k"  # сильный b0, можно b1/b2
    pretrained: bool = True

def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

set_seed(CFG.seed)
print("device:", CFG.device)

WIN_SAMPLES = int(CFG.sr * CFG.win_sec)
HOP_SAMPLES_TEST = int(CFG.sr * CFG.hop_sec_test)



DATA = Path("/kaggle/input/rfcx-species-audio-detection")
TRAIN_DIR = DATA / "train"
TEST_DIR  = DATA / "test"
TP_CSV = DATA / "train_tp.csv"
FP_CSV = DATA / "train_fp.csv"
SUB_CSV = DATA / "sample_submission.csv"

for p in [DATA, TRAIN_DIR, TEST_DIR, TP_CSV, FP_CSV, SUB_CSV]:
    assert p.exists(), f"Не найдено: {p}"

tp = pd.read_csv(TP_CSV)
fp = pd.read_csv(FP_CSV)
sub = pd.read_csv(SUB_CSV)

species_cols = [c for c in sub.columns if c != "recording_id"]
num_classes = len(species_cols)

print("train_tp:", tp.shape, "| train_fp:", fp.shape)
print("sample_submission:", sub.shape, "| num_classes:", num_classes)
print("species cols:", species_cols[:5], "...")



def audio_path(root: Path, recording_id: str) -> Path:
    for ext in (".flac", ".wav", ".ogg", ".mp3"):
        p = root / f"{recording_id}{ext}"
        if p.exists():
            return p
    p = root / recording_id
    if p.exists():
        return p
    raise FileNotFoundError(f"Не нашёл аудио для {recording_id} в {root}")

rid0 = tp["recording_id"].iloc[0]
p0 = audio_path(TRAIN_DIR, rid0)
print("example:", rid0, "->", p0.name)



def audio_path(root: Path, recording_id: str) -> Path:
    # В RFCx обычно .flac
    for ext in (".flac", ".wav", ".ogg", ".mp3"):
        p = root / f"{recording_id}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Не нашёл аудио для {recording_id} в {root}")

@lru_cache(maxsize=512)
def _load_audio_cached(path_str: str):
    wav, sr = torchaudio.load(path_str)  # [ch, T]
    wav = wav.mean(dim=0)                # mono [T]
    return wav, sr
    
def load_segment(path: Path, start_s: float, win_sec: float, target_sr=CFG.sr) -> torch.Tensor:
    info = torchaudio.info(str(path))
    sr0 = info.sample_rate
    frame_offset = max(0, int(start_s * sr0))
    num_frames = int(win_sec * sr0)

    wav, sr = torchaudio.load(str(path), frame_offset=frame_offset, num_frames=num_frames)
    wav = wav.mean(0)
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)

    # доводим до ровно WIN_SAMPLES
    if wav.numel() < WIN_SAMPLES:
        wav = F.pad(wav, (0, WIN_SAMPLES - wav.numel()))
    else:
        wav = wav[:WIN_SAMPLES]
    return wav

def load_audio_mono_nocache(path: Path, target_sr=CFG.sr) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.mean(dim=0)
    if sr != target_sr:
        wav = AF.resample(wav, sr, target_sr)
    return wav


def crop_or_pad(wav: torch.Tensor, start_s: float, win_samples: int) -> torch.Tensor:
    start = int(round(start_s * CFG.sr))
    end = start + win_samples
    if start < 0:
        start = 0
        end = win_samples

    if end <= wav.numel():
        return wav[start:end]

    out = torch.zeros(win_samples, dtype=wav.dtype)
    if start < wav.numel():
        seg = wav[start:]
        out[:seg.numel()] = seg
    return out



# события TP по записи
tp_by_rec = {}
for r in tp.itertuples(index=False):
    tp_by_rec.setdefault(r.recording_id, []).append((float(r.t_min), float(r.t_max), int(r.species_id)))

def labels_for_window(events, t0: float, t1: float, num_classes: int) -> np.ndarray:
    y = np.zeros(num_classes, dtype=np.float32)
    for a, b, sid in events:
        if a < t1 and b > t0:  # overlap
            if 0 <= sid < num_classes:
                y[sid] = 1.0
    return y



def build_samples(tp: pd.DataFrame, fp: pd.DataFrame,
                  fp_ratio=CFG.fp_ratio, bg_ratio=CFG.bg_ratio) -> pd.DataFrame:
    rows = []

    # --- TP окна: вокруг центра события
    for r in tp.itertuples(index=False):
        center = 0.5*(float(r.t_min) + float(r.t_max))
        t0 = center - CFG.win_sec/2
        t1 = t0 + CFG.win_sec
        rows.append((r.recording_id, t0, t1, "tp", -1))

    # --- FP hard negatives: берём подвыборку
    n_fp = int(len(tp) * fp_ratio)
    fp_s = fp.sample(n=min(n_fp, len(fp)), random_state=CFG.seed)
    for r in fp_s.itertuples(index=False):
        center = 0.5*(float(r.t_min) + float(r.t_max))
        t0 = center - CFG.win_sec/2
        t1 = t0 + CFG.win_sec
        rows.append((r.recording_id, t0, t1, "fp", int(r.species_id)))

    # --- Background: случайные окна из тех же записей (учим “шум/насекомых”)
    # RFCx записи обычно ~60с, поэтому для простоты выбираем t0 в [0, 60-win]
    n_bg = int(len(tp) * bg_ratio)
    recs = tp["recording_id"].unique()
    for _ in range(n_bg):
        rid = recs[np.random.randint(0, len(recs))]
        max_t0 = max(0.0, 60.0 - CFG.win_sec)
        t0 = float(np.random.uniform(0.0, max_t0))
        t1 = t0 + CFG.win_sec
        rows.append((rid, t0, t1, "bg", -1))

    df = pd.DataFrame(rows, columns=["recording_id","t0","t1","kind","fp_species_id"])
    return df.sample(frac=1.0, random_state=CFG.seed).reset_index(drop=True)

samples = build_samples(tp, fp)
print("samples:", samples.shape)
print(samples["kind"].value_counts())
samples.head()



from sklearn.model_selection import GroupShuffleSplit

gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=CFG.seed)
tr_idx, va_idx = next(gss.split(samples, groups=samples["recording_id"]))

train_s = samples.iloc[tr_idx].reset_index(drop=True)
valid_s = samples.iloc[va_idx].reset_index(drop=True)

print("train:", len(train_s), "| valid:", len(valid_s))
print("unique rec train/valid:", train_s.recording_id.nunique(), "/", valid_s.recording_id.nunique())



import torchaudio
import torch

# Создай один раз (вне цикла), чтобы не пересоздавать каждый батч
_mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=CFG.sr,
    n_fft=CFG.n_fft,
    hop_length=CFG.hop_length,
    n_mels=CFG.n_mels,
    f_min=CFG.fmin,
    f_max=CFG.fmax,
    power=2.0,
).to(CFG.device)

_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0).to(CFG.device)

def wav_to_logmel_torch(wav: torch.Tensor) -> torch.Tensor:
    """
    wav: [B, WIN] float32 на GPU
    return: [B, 1, M, T] float32 на GPU
    """
    # MelSpectrogram ожидает [B, T]
    S = _mel(wav)                 # [B, M, T] power
    S = _db(S)                    # [B, M, T] dB, без -inf

    # z-norm per-sample (очень важно eps)
    mean = S.mean(dim=(1,2), keepdim=True)
    std  = S.std(dim=(1,2), keepdim=True).clamp_min(1e-4)
    S = (S - mean) / std

    # финальная защита
    S = torch.nan_to_num(S, nan=0.0, posinf=0.0, neginf=0.0)

    return S.unsqueeze(1)         # [B,1,M,T]


def spec_augment_torch(x: torch.Tensor, p=0.5, max_mask_pct=0.10, num_masks=2) -> torch.Tensor:
    """
    x: [B,1,M,T] на GPU
    """
    if p <= 0 or (torch.rand(1).item() > p):
        return x
    B, C, M, T = x.shape
    x = x.clone()
    for _ in range(num_masks):
        if torch.rand(1).item() < 0.5:
            # freq mask
            f = max(1, int(max_mask_pct * M))
            f0 = torch.randint(0, max(1, M - f + 1), (1,)).item()
            x[:, :, f0:f0+f, :] = 0
        else:
            # time mask
            t = max(1, int(max_mask_pct * T))
            t0 = torch.randint(0, max(1, T - t + 1), (1,)).item()
            x[:, :, :, t0:t0+t] = 0
    return x



class RFCXWindowDataset(Dataset):
    def __init__(self, df: pd.DataFrame, train: bool):
        self.df = df.reset_index(drop=True)
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        rid = row.recording_id
        kind = row.kind

        wav_full = load_audio_mono_nocache(audio_path(TRAIN_DIR, rid))       # torch [T] CPU
        wav = crop_or_pad(wav_full, float(row.t0), WIN_SAMPLES)      # torch [WIN] CPU

        # label multi-hot по пересечению с TP
        events = tp_by_rec.get(rid, [])
        y = labels_for_window(events, float(row.t0), float(row.t1), num_classes)  # np [C]

        # weights: по умолчанию 1
        w = np.ones(num_classes, dtype=np.float32)

        if kind == "fp":
            # hard negative: усиливаем штраф только по fp_species_id
            sid = int(row.fp_species_id)
            y[sid] = 0.0
            w[:] = 0.0
            w[sid] = CFG.fp_focus

        # возвращаем waveform (CPU), y/w (CPU) -> на GPU перенесем батчем
        return wav.numpy().astype(np.float32), y, w
        
CFG.num_workers = 0

train_loader = DataLoader(RFCXWindowDataset(train_s, True),
                          batch_size=CFG.train_bs, shuffle=True,
                          num_workers=CFG.num_workers, pin_memory=True,
                          drop_last=True, persistent_workers=False)

valid_loader = DataLoader(RFCXWindowDataset(valid_s, False),
                          batch_size=CFG.valid_bs, shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True,
                          persistent_workers=False)

xb0, yb0, wb0 = next(iter(train_loader))
print("wave batch:", xb0.shape, "| y:", yb0.shape, "| w:", wb0.shape)



class SpecNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(
            CFG.backbone,
            pretrained=CFG.pretrained,  # интернет нужен только здесь
            in_chans=1,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.backbone(x)

model = SpecNet(num_classes).to(CFG.device)
print("params:", sum(p.numel() for p in model.parameters())/1e6, "M")



def _one_sample_lwlrap(truth, scores):
    pos = np.where(truth > 0)[0]
    if len(pos) == 0:
        return (None, None)
    rank = scores.argsort()[::-1]
    prec = []
    hit = 0
    for i, k in enumerate(rank, start=1):
        if truth[k] > 0:
            hit += 1
            prec.append(hit / i)
    return (pos, np.array(prec, dtype=np.float32))

def lwlrap(truth, scores):
    C = truth.shape[1]
    per_class_prec = [[] for _ in range(C)]
    for t, s in zip(truth, scores):
        res = _one_sample_lwlrap(t, s)
        if res[0] is None:
            continue
        pos, prec = res
        for cls, p in zip(pos, prec):
            per_class_prec[cls].append(p)

    per_class_lwlrap = np.array([np.mean(v) if len(v) else 0.0 for v in per_class_prec], dtype=np.float32)
    weights = truth.sum(axis=0)
    weights = weights / (weights.sum() + 1e-12)
    return float((per_class_lwlrap * weights).sum())



# pos_weight по train_s (примерно)
def estimate_pos_weight(df: pd.DataFrame) -> torch.Tensor:
    Ys = []
    for r in df.sample(n=min(1200, len(df)), random_state=CFG.seed).itertuples(index=False):
        events = tp_by_rec.get(r.recording_id, [])
        y = labels_for_window(events, float(r.t0), float(r.t1), num_classes)
        Ys.append(y)
    Y = np.stack(Ys)
    pos = Y.sum(axis=0)
    neg = len(Y) - pos
    pw = (neg + 1e-6) / (pos + 1e-6)
    return torch.tensor(pw, device=CFG.device).float()

pos_weight = estimate_pos_weight(train_s)
bce = nn.BCEWithLogitsLoss(reduction="none", pos_weight=pos_weight)

scaler = torch.amp.GradScaler('cuda', enabled=(CFG.device=="cuda"))
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)



def to_tensor(x, device):
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    return torch.from_numpy(x).to(device, non_blocking=True)

import numpy as np
import torch
import torch.nn.functional as F

bce = torch.nn.BCEWithLogitsLoss(reduction="none")

def run_epoch(loader, train: bool):
    model.train(train)
    losses = []
    all_y, all_p = [], []

    for wav, y, w in loader:
        # wav: [B, WIN]  y: [B,C]  w: [B,C]
        wav = wav.to(CFG.device, non_blocking=True).float()
        y   = y.to(CFG.device, non_blocking=True).float()
        w   = w.to(CFG.device, non_blocking=True).float()

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(CFG.device == "cuda")):
            x = wav_to_logmel_torch(wav)          # [B,1,M,T] GPU
            if train:
                x = spec_augment_torch(x, p=CFG.specaug_p)

            logits = model(x)                     # [B,C]

            # БАЗОВЫЙ СТАБИЛЬНЫЙ ЛОСС (без бутстрэпа) — сначала доведи до finite!
            loss_mat = bce(logits, y)             # [B,C]
            denom = w.sum().clamp_min(1.0)        # защита от 0
            loss = (loss_mat * w).sum() / denom

        # защита от NaN/Inf
        if not torch.isfinite(loss):
            print("⚠️ non-finite loss, skip batch")
            continue

        if train:
            scaler.scale(loss).backward()
            # клиппинг до step (важно)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            scaler.step(optimizer)
            scaler.update()

        losses.append(loss.item())
        all_y.append(y.detach().cpu().numpy())
        all_p.append(logits.detach().cpu().numpy())

    all_y = np.concatenate(all_y) if len(all_y) else np.zeros((0, num_classes), np.float32)
    all_p = np.concatenate(all_p) if len(all_p) else np.zeros((0, num_classes), np.float32)

    return float(np.mean(losses)) if losses else float("nan"), lwlrap(all_y, all_p)



best = -1.0
for e in range(1, CFG.epochs + 1):
    tr_loss, tr_lwl = run_epoch(train_loader, True)
    va_loss, va_lwl = run_epoch(valid_loader, False)
    scheduler.step()

    print(f"Epoch {e:02d} | train loss {tr_loss:.4f} lwlrap {tr_lwl:.4f} | valid loss {va_loss:.4f} lwlrap {va_lwl:.4f}")

    if va_lwl > best:
        best = va_lwl
        torch.save({"model": model.state_dict(), "species_cols": species_cols}, "best_model.pt")

print("Best valid lwlrap:", best)



import gc, torch

_load_audio_cached.cache_clear()
del train_loader, valid_loader
gc.collect()
torch.cuda.empty_cache()



@torch.inference_mode()
def predict_recording(path: Path) -> np.ndarray:
    wav = load_audio_mono_nocache(path)
    T = wav.numel()

    # нарезаем окна с hop_sec_test
    starts = list(range(0, max(1, T - WIN_SAMPLES + 1), HOP_SAMPLES_TEST))
    if len(starts) == 0:
        starts = [0]

    probs_max = torch.zeros(num_classes, device=CFG.device)

    bs = 32
    for i in range(0, len(starts), bs):
        batch_starts = starts[i:i+bs]
        batch = []
        for s in batch_starts:
            seg = wav[s:s+WIN_SAMPLES]
            if seg.numel() < WIN_SAMPLES:
                pad = torch.zeros(WIN_SAMPLES - seg.numel())
                seg = torch.cat([seg, pad], dim=0)
            batch.append(seg)

        batch = torch.stack(batch, dim=0).to(CFG.device)  # [B, WIN]
        x = wav_to_logmel_torch(batch)                    # [B,1,M,T]
        logits = model(x)
        probs = torch.sigmoid(logits).max(dim=0).values   # max-pool по окнам
        probs_max = torch.maximum(probs_max, probs)

    return probs_max.detach().cpu().numpy()

# загружаем лучший чекпойнт
ckpt = torch.load("best_model.pt", map_location="cpu")
model.load_state_dict(ckpt["model"], strict=True)
model.eval()

rows = []
for rid in sub["recording_id"].tolist():
    p = audio_path(TEST_DIR, rid)
    probs = predict_recording(p)
    rows.append([rid] + probs.tolist())

out = pd.DataFrame(rows, columns=["recording_id"] + species_cols)
out.to_csv("submission.csv", index=False)

print("Saved submission.csv:", out.shape)
out.head()


