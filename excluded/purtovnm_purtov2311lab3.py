try:
    import timm
except ImportError:
    !pip -q install timm
    import timm

import os, random, copy, warnings
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchaudio
import torchaudio.functional as AF

from sklearn.model_selection import KFold
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class CFG:
    seed: int = 563
    sr: int = 48_000
    win_sec: int = 10
    win_len: int = sr * win_sec

    img_h: int = 224
    img_w: int = 400

    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    top_db: float = 80.0

    epochs: int = 20
    batch_size: int = 8
    lr: float = 2e-4
    wd: float = 1e-2
    n_folds: int = 5
    num_workers: int = 2

    cache_images: bool = True
    backbone: str = "resnet50"
    pretrained: bool = True

    hop_test_sec: int = 10  # ÑˆĞ°Ğ³ Ñ�ĞµĞ³Ğ¼ĞµĞ½Ñ‚Ğ° Ğ½Ğ° Ñ‚ĞµÑ�Ñ‚Ğµ (10 = Ğ±ĞµĞ· Ğ¿ĞµÑ€ĞµĞºÑ€Ñ‹Ñ‚Ğ¸Ñ�)

cfg = CFG()

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(cfg.seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("âœ… device:", device)
print("torch:", torch.__version__, "| torchaudio:", torchaudio.__version__)

# ----------------------------
# 2) Data
# ----------------------------
DATA_DIR = "/kaggle/input/rfcx-species-audio-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train_tp.csv")
TRAIN_AUDIO_DIR = os.path.join(DATA_DIR, "train")
TEST_AUDIO_DIR = os.path.join(DATA_DIR, "test")
SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

train_tp = pd.read_csv(TRAIN_CSV)
sub = pd.read_csv(SUB_CSV)

labels = 24
species_cols = [c for c in sub.columns if c != "recording_id"]

# ĞºĞ°Ğº Ğ² Ñ‚Ğ²Ğ¾Ñ‘Ğ¼ Ñ…Ğ¾Ñ€Ğ¾ÑˆĞµĞ¼ Ğ½Ğ¾ÑƒÑ‚Ğµ: fmin/fmax Ğ¸Ğ· Ñ€Ğ°Ğ·Ğ¼ĞµÑ‚ĞºĞ¸ + pad
fmin_hz = int(train_tp["f_min"].min() * 0.90)
fmax_hz = int(train_tp["f_max"].max() * 1.10)

print("âœ… train_tp:", train_tp.shape, "| fmin/fmax:", fmin_hz, fmax_hz)


def load_audio(recording_id: str, audio_dir: str) -> torch.Tensor:
    path = os.path.join(audio_dir, f"{recording_id}.flac")
    wav, sr0 = torchaudio.load(path)   # [ch, T]
    wav = wav.mean(dim=0)              # mono [T]
    if sr0 != cfg.sr:
        wav = AF.resample(wav, sr0, cfg.sr)
    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    return wav

def slice_centered(wav: torch.Tensor, t_min_s: float, t_max_s: float) -> torch.Tensor:
    center = 0.5 * (float(t_min_s) + float(t_max_s))
    center_i = int(round(center * cfg.sr))
    start = max(center_i - cfg.win_len // 2, 0)
    end = min(start + cfg.win_len, wav.numel())
    start = max(end - cfg.win_len, 0)

    clip = wav[start:end]
    if clip.numel() < cfg.win_len:
        clip = F.pad(clip, (0, cfg.win_len - clip.numel()))
    return clip



mel = torchaudio.transforms.MelSpectrogram(
    sample_rate=cfg.sr,
    n_fft=cfg.n_fft,
    hop_length=cfg.hop_length,
    n_mels=cfg.n_mels,
    f_min=fmin_hz,
    f_max=fmax_hz,
    power=2.0,
)

def power_to_db_like_librosa(S: torch.Tensor, top_db: float = 80.0) -> torch.Tensor:
    # S: [M, T] power
    S = torch.clamp(S, min=1e-10)
    db = 10.0 * torch.log10(S)
    db = db - db.max()             # Ğ¼Ğ°ĞºÑ�Ğ¸Ğ¼ÑƒĞ¼ = 0
    db = torch.clamp(db, min=-top_db, max=0.0)
    return db

@torch.no_grad()
def clip_to_image_uint8(wav_clip: torch.Tensor) -> np.ndarray:
    # wav_clip: [T] on CPU
    S = mel(wav_clip.unsqueeze(0)).squeeze(0)     # [M, T]
    db = power_to_db_like_librosa(S, cfg.top_db)  # [-top_db..0]

    # resize Ñ‡ĞµÑ€ĞµĞ· torch (Ğ±ĞµĞ· skimage)
    x = db.unsqueeze(0).unsqueeze(0)  # [1,1,M,T]
    x = F.interpolate(x, size=(cfg.img_h, cfg.img_w), mode="bilinear", align_corners=False)
    x = x.squeeze(0).squeeze(0)       # [H,W]

    # mean/std then minmax -> uint8 (ĞºĞ°Ğº Ñƒ Ñ‚ĞµĞ±Ñ�)
    eps = 1e-6
    x = (x - x.mean()) / (x.std() + eps)
    x_min = x.min()
    x_max = x.max()
    x = (x - x_min) / (x_max - x_min + eps)       # [0..1]
    img = (x * 255.0).clamp(0,255).byte().cpu().numpy()
    return img



class ImgAug:
    def __init__(self, p=0.7):
        self.p = p

    def __call__(self, img_u8: np.ndarray) -> np.ndarray:
        if random.random() > self.p:
            return img_u8

        x = img_u8.astype(np.float32) / 255.0

        # random gamma
        if random.random() < 0.5:
            g = np.random.uniform(0.7, 1.4)
            x = np.clip(x, 0, 1) ** g

        # random noise
        if random.random() < 0.5:
            n = np.random.normal(0, 0.02, size=x.shape).astype(np.float32)
            x = np.clip(x + n, 0, 1)

        # time flip (Ğ³Ğ¾Ñ€Ğ¸Ğ·Ğ¾Ğ½Ñ‚Ğ°Ğ»ÑŒĞ½Ñ‹Ğ¹)
        if random.random() < 0.3:
            x = x[:, ::-1]

        return (x * 255.0).astype(np.uint8)

augment = ImgAug(p=0.7)



# ----------------------------
spec_cache = {}

def make_one(idx: int):
    row = train_tp.iloc[idx]
    wav = load_audio(row["recording_id"], TRAIN_AUDIO_DIR)
    clip = slice_centered(wav, row["t_min"], row["t_max"])
    img = clip_to_image_uint8(clip)
    return idx, img

if cfg.cache_images:
    print("âš™ï¸� Caching mel images...")
    with ThreadPoolExecutor() as ex:
        res = list(tqdm(ex.map(make_one, range(len(train_tp))), total=len(train_tp)))
    spec_cache = {i: img for i, img in res}
    print("âœ… cached:", len(spec_cache))


class RFCXEventDataset(Dataset):
    def __init__(self, idxs, train: bool):
        self.idxs = np.asarray(idxs)
        self.train = train

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, k):
        i = int(self.idxs[k])
        row = train_tp.iloc[i]
        y = int(row["species_id"])

        if cfg.cache_images:
            img = spec_cache[i]
        else:
            wav = load_audio(row["recording_id"], TRAIN_AUDIO_DIR)
            clip = slice_centered(wav, row["t_min"], row["t_max"])
            img = clip_to_image_uint8(clip)

        if self.train:
            img = augment(img)

        # 3 ĞºĞ°Ğ½Ğ°Ğ»Ğ° ĞºĞ°Ğº Ğ² Ñ‚Ğ²Ğ¾Ñ‘Ğ¼ Ñ€Ğ°Ğ±Ğ¾Ñ‡ĞµĞ¼ Ğ½Ğ¾ÑƒÑ‚Ğµ
        x = np.stack([img, img, img], axis=0).astype(np.float32) / 255.0
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long) 
def get_model():
    try:
        m = timm.create_model(cfg.backbone, pretrained=cfg.pretrained, num_classes=labels, in_chans=3)
    except Exception:
        print("âš ï¸� pretrained failed -> pretrained=False")
        m = timm.create_model(cfg.backbone, pretrained=False, num_classes=labels, in_chans=3)
    return m.to(device)

loss_fn = nn.CrossEntropyLoss()

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    for x, y in loader:
        x = x.to(device, dtype=torch.float32, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = loss_fn(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)

    return total_loss / max(total, 1), total_correct / max(total, 1)

def train_fold(fold, tr_idx, va_idx, out_dir="models_timm_clean"):
    os.makedirs(out_dir, exist_ok=True)

    ds_tr = RFCXEventDataset(tr_idx, train=True)
    ds_va = RFCXEventDataset(va_idx, train=False)

    dl_tr = DataLoader(ds_tr, batch_size=cfg.batch_size, shuffle=True,
                       num_workers=cfg.num_workers, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=64, shuffle=False,
                       num_workers=cfg.num_workers, pin_memory=True, drop_last=False)

    model = get_model()
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", patience=3)

    best_acc = -1.0
    best_wts = copy.deepcopy(model.state_dict())
    save_path = os.path.join(out_dir, f"fold{fold}.pt")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        run_loss, seen = 0.0, 0

        for x, y in dl_tr:
            x = x.to(device, dtype=torch.float32, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            run_loss += loss.item() * x.size(0)
            seen += x.size(0)

        tr_loss = run_loss / max(seen, 1)
        va_loss, va_acc = evaluate(model, dl_va)
        sch.step(va_loss)

        print(f"[Fold {fold}] Epoch {epoch:02d}/{cfg.epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | val_acc={va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            best_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_wts)
    torch.save(model.state_dict(), save_path)
    print(f"âœ… Fold {fold} best val_acc={best_acc:.4f} | saved: {save_path}")
    return save_path


kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
X = np.arange(len(train_tp))
y = train_tp["species_id"].values

fold_models = []
for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y)):
    print("=" * 80)
    print(f"ğŸš€ Start Fold {fold}/{cfg.n_folds-1}")
    fold_models.append(train_fold(fold, tr_idx, va_idx))

print("âœ… trained folds:", fold_models)


@torch.no_grad()
def load_test_wav(path: str) -> torch.Tensor:
    wav, sr0 = torchaudio.load(path)
    wav = wav.mean(dim=0)
    if sr0 != cfg.sr:
        wav = AF.resample(wav, sr0, cfg.sr)
    wav = torch.nan_to_num(wav, nan=0.0, posinf=0.0, neginf=0.0)
    return wav

@torch.no_grad()
def wav_to_segments(wav: torch.Tensor) -> list[torch.Tensor]:
    hop = cfg.hop_test_sec * cfg.sr
    segs = []
    for start in range(0, max(1, wav.numel()), hop):
        clip = wav[start:start + cfg.win_len]
        if clip.numel() < cfg.win_len:
            clip = F.pad(clip, (0, cfg.win_len - clip.numel()))
        segs.append(clip)
        if start + cfg.win_len >= wav.numel():
            break
    return segs

@torch.no_grad()
def predict_one_file(models, wav: torch.Tensor) -> np.ndarray:
    segs = wav_to_segments(wav)

    # Ğ³Ğ¾Ñ‚Ğ¾Ğ²Ğ¸Ğ¼ Ğ±Ğ°Ñ‚Ñ‡Ğ°Ğ¼Ğ¸, Ñ‡Ñ‚Ğ¾Ğ±Ñ‹ Ğ½Ğµ Ğ´ĞµÑ€Ğ¶Ğ°Ñ‚ÑŒ Ğ²Ñ�Ñ‘ Ñ�Ñ€Ğ°Ğ·Ñƒ
    all_model_preds = []

    for model in models:
        model.eval()
        best = torch.zeros(labels, device=device)

        bs = 32
        for i in range(0, len(segs), bs):
            batch = segs[i:i+bs]
            imgs = []
            for clip in batch:
                img = clip_to_image_uint8(clip)  # uint8 HxW
                x = np.stack([img, img, img], axis=0).astype(np.float32) / 255.0
                imgs.append(x)
            xb = torch.from_numpy(np.stack(imgs, axis=0)).to(device)

            logits = model(xb)
            probs = torch.softmax(logits, dim=1)           # [B,24]
            best = torch.maximum(best, probs.max(dim=0).values)

        all_model_preds.append(best)

    avg = torch.stack(all_model_preds, dim=0).mean(dim=0)
    return avg.detach().cpu().numpy()

def load_members(paths):
    ms = []
    for p in paths:
        m = get_model()
        m.load_state_dict(torch.load(p, map_location=device))
        m.eval()
        ms.append(m)
    return ms

members = load_members(fold_models)
print("âœ… Loaded members:", len(members))

rows = []
test_files = sorted(os.listdir(TEST_AUDIO_DIR))
for tf in tqdm(test_files, desc="Predicting test"):
    rid = os.path.splitext(tf)[0]
    wav = load_test_wav(os.path.join(TEST_AUDIO_DIR, tf))
    probs = predict_one_file(members, wav)
    rows.append([rid] + probs.tolist())

out = pd.DataFrame(rows, columns=["recording_id"] + species_cols)
out.to_csv("submission.csv", index=False)
print("âœ… Saved submission.csv:", out.shape)
out.head()

