!pip -q install -U pydicom pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg



import os, gc, math, time, random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

import cv2
import pydicom

from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(42)


@dataclass
class CFG:
    seed: int = 42

    # speed + stability for RSNA DICOM
    img_size: int = 320        # 512 is slower; 384 is usually best tradeoff on Kaggle
    batch_size: int = 8          # if OOM -> 4
    grad_accum: int = 2          # if batch=4 -> 4
    num_workers: int = 8         # if dataloader crashes -> 2, if stable -> try 8

    # training
    epochs: int = 6             
    lr: float = 2e-4
    wd: float = 1e-4

    # CNN backbone
    model_name: str = "tf_efficientnet_b0.ns_jft_in1k"

    # folds
    n_folds: int = 5
    fold: int = 0

    # mixed precision
    use_amp: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

CFG = CFG()

print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("device:", CFG.device)



INPUT = Path("/kaggle/input/rsna-breast-cancer-detection")
train_images_dir = INPUT / "train_images"

train_df = pd.read_csv(INPUT / "train.csv")
test_df  = pd.read_csv(INPUT / "test.csv")

print("train_df:", train_df.shape, "test_df:", test_df.shape)
print("train columns:", train_df.columns.tolist())

# density task
df = train_df.copy()
df = df[df["density"].notna()].reset_index(drop=True)
df["density"] = df["density"].astype(str)

dens2id = {"A":0, "B":1, "C":2, "D":3}
id2dens = {v:k for k,v in dens2id.items()}

df = df[df["density"].isin(dens2id.keys())].reset_index(drop=True)
df["target"] = df["density"].map(dens2id).astype(int)

print("density counts:\n", df["density"].value_counts())
print("rows used:", len(df))



df["fold"] = -1
gkf = GroupKFold(n_splits=CFG.n_folds)

for fold, (tr_idx, va_idx) in enumerate(gkf.split(df, df["target"], groups=df["patient_id"])):
    df.loc[va_idx, "fold"] = fold

print(df["fold"].value_counts())



def read_dicom_as_uint8(path: Path, img_size: int) -> np.ndarray:
    dcm = pydicom.dcmread(str(path))
    img = dcm.pixel_array.astype(np.float32)

    # handle MONOCHROME1 inversion
    if getattr(dcm, "PhotometricInterpretation", "") == "MONOCHROME1":
        img = img.max() - img

    # percentile clipping to reduce extreme values
    lo, hi = np.percentile(img, (0.5, 99.5))
    img = np.clip(img, lo, hi)

    # normalize to 0..255 uint8
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)
    img = (img * 255.0).astype(np.uint8)

    # resize
    img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return img


def get_transforms(train: bool):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(scale=(0.97, 1.03), translate_percent=(0.0, 0.01), rotate=(-5, 5), p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.12, contrast_limit=0.12, p=0.4),
            # intentionally no GaussNoise to avoid parameter version mismatch
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.25,0.25,0.25)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=(0.5,0.5,0.5), std=(0.25,0.25,0.25)),
            ToTensorV2(),
        ])



class RSNADensityDataset(Dataset):
    def __init__(self, df, images_dir: Path, img_size: int, train: bool):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.img_size = img_size
        self.train = train
        self.tfms = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        dcm_path = self.images_dir / str(r.patient_id) / f"{r.image_id}.dcm"

        img = read_dicom_as_uint8(dcm_path, self.img_size)   # (H,W) uint8
        img3 = np.stack([img, img, img], axis=-1)            # (H,W,3)

        x = self.tfms(image=img3)["image"]                  # tensor (3,H,W)
        y = torch.tensor(int(r.target), dtype=torch.long)   # 0..3
        return x, y


def make_loaders(tr_df, va_df):
    train_ds = RSNADensityDataset(tr_df, train_images_dir, CFG.img_size, train=True)
    valid_ds = RSNADensityDataset(va_df, train_images_dir, CFG.img_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(CFG.num_workers > 0),
        prefetch_factor=4 if CFG.num_workers > 0 else None,
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(CFG.num_workers > 0),
        prefetch_factor=4 if CFG.num_workers > 0 else None,
    )
    return train_loader, valid_loader



NUM_CLASSES = 4

class Model(nn.Module):
    def __init__(self, model_name: str, num_classes: int = 4):
        super().__init__()
        self.net = timm.create_model(model_name, pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.net(x)


def class_weights_from_targets(targets, num_classes=4):
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    w = counts.sum() / (counts + 1e-6)
    w = w / w.mean()
    return torch.tensor(w, dtype=torch.float32)


def build_components(tr_df):
    model = Model(CFG.model_name, NUM_CLASSES).to(CFG.device)

    w = class_weights_from_targets(tr_df["target"].values, NUM_CLASSES).to(CFG.device)
    criterion = nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=(CFG.use_amp and CFG.device == "cuda"))
    return model, criterion, optimizer, scheduler, scaler



def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0

    for step, (x, y) in enumerate(loader):
        x = x.to(CFG.device, non_blocking=True)
        y = y.to(CFG.device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(CFG.use_amp and CFG.device == "cuda")):
            logits = model(x)
            loss = criterion(logits, y) / CFG.grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % CFG.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        running_loss += loss.item() * CFG.grad_accum

    return running_loss / max(len(loader), 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []

    for x, y in loader:
        x = x.to(CFG.device, non_blocking=True)
        logits = model(x)
        pred = torch.argmax(logits, dim=1).detach().cpu().numpy()

        ys.append(y.numpy())
        ps.append(pred)

    ys = np.concatenate(ys)
    ps = np.concatenate(ps)

    acc = accuracy_score(ys, ps)
    f1 = f1_score(ys, ps, average="macro")
    cm = confusion_matrix(ys, ps)
    return acc, f1, cm



fold = CFG.fold

tr_small = df[df.fold != fold].sample(200, random_state=CFG.seed).reset_index(drop=True)
va_small = df[df.fold == fold].sample(200, random_state=CFG.seed).reset_index(drop=True)

train_loader, valid_loader = make_loaders(tr_small, va_small)
model, criterion, optimizer, scheduler, scaler = build_components(tr_small)

t0 = time.time()
loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
acc, f1, cm = evaluate(model, valid_loader)
print("smoke loss:", loss)
print("smoke acc:", acc, "smoke macro-f1:", f1)
print("smoke confusion matrix:\n", cm)
print("time (s):", time.time() - t0)

del model
gc.collect()
torch.cuda.empty_cache()



fold = CFG.fold

tr_df = df[df.fold != fold].reset_index(drop=True)
va_df = df[df.fold == fold].reset_index(drop=True)

train_loader, valid_loader = make_loaders(tr_df, va_df)

print("train rows:", len(tr_df), "val rows:", len(va_df))
print("batches per epoch:", len(train_loader))




import time

def estimate_epoch_time_from_loader(loader, n_batches=100):
    n_batches = min(n_batches, len(loader))
    it = iter(loader)

    t0 = time.time()
    for _ in range(n_batches):
        x, y = next(it)
    dt = time.time() - t0

    sec_per_batch = dt / max(n_batches, 1)
    total_batches = len(loader)
    est_epoch_sec = sec_per_batch * total_batches

    print("measured batches:", n_batches)
    print("sec per batch (data pipeline):", round(sec_per_batch, 3))
    print("batches per epoch:", total_batches)
    print("estimated epoch time (minutes):", round(est_epoch_sec / 60, 1))
    return sec_per_batch, est_epoch_sec

sec_per_batch, est_epoch_sec = estimate_epoch_time_from_loader(train_loader, n_batches=100)

# Rough training time estimates for different fold counts (ignores validation cost, so add ~10-25%)
for folds in [1, 3, 5]:
    total_hours = (est_epoch_sec * CFG.epochs * folds) / 3600
    print("approx training hours for", folds, "fold(s) at", CFG.epochs, "epochs:", round(total_hours, 2))
print("note: add ~10-25% extra for validation and overhead")



from tqdm import tqdm

CACHE = Path("/kaggle/working/cache_png")
CACHE.mkdir(parents=True, exist_ok=True)

def cache_png_for_df(df_part: pd.DataFrame, cache_dir: Path, img_size: int):
    missing = 0
    for r in tqdm(df_part.itertuples(index=False), total=len(df_part)):
        out_dir = cache_dir / str(r.patient_id)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{r.image_id}.png"
        if out_path.exists():
            continue

        dcm_path = train_images_dir / str(r.patient_id) / f"{r.image_id}.dcm"
        if not dcm_path.exists():
            missing += 1
            continue

        img = read_dicom_as_uint8(dcm_path, img_size)
        cv2.imwrite(str(out_path), img)

    print("missing dicoms:", missing)

fold = CFG.fold
tr_df = df[df.fold != fold].reset_index(drop=True)
va_df = df[df.fold == fold].reset_index(drop=True)

cache_png_for_df(tr_df, CACHE, CFG.img_size)
cache_png_for_df(va_df, CACHE, CFG.img_size)

print("cache path:", CACHE)



from pathlib import Path
CACHE = Path("/kaggle/input/png-dicom-projet-ai/cache_png")
print("cache exists:", CACHE.exists())
!find /kaggle/input/png-dicom-projet-ai/cache_png -name "*.png" | wc -l
!du -sh /kaggle/input/png-dicom-projet-ai/cache_png



import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class RSNADensityPNGDataset(Dataset):
    def __init__(self, df, cache_dir: Path, img_size: int, train: bool):
        self.df = df.reset_index(drop=True)
        self.cache_dir = cache_dir
        self.img_size = img_size
        self.train = train
        self.tfms = get_transforms(train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        png_path = self.cache_dir / str(r.patient_id) / f"{r.image_id}.png"

        img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            # should not happen if cache is complete
            raise FileNotFoundError(str(png_path))

        if img.shape[0] != self.img_size or img.shape[1] != self.img_size:
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)

        img3 = np.stack([img, img, img], axis=-1)
        x = self.tfms(image=img3)["image"]
        y = torch.tensor(int(r.target), dtype=torch.long)
        return x, y

def make_png_loaders(tr_df, va_df):
    train_ds = RSNADensityPNGDataset(tr_df, CACHE, CFG.img_size, train=True)
    valid_ds = RSNADensityPNGDataset(va_df, CACHE, CFG.img_size, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=CFG.batch_size,
        shuffle=True,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(CFG.num_workers > 0),
        prefetch_factor=4 if CFG.num_workers > 0 else None,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=(CFG.num_workers > 0),
        prefetch_factor=4 if CFG.num_workers > 0 else None,
    )
    return train_loader, valid_loader



CFG.img_size = 384        # if time tight -> 320
CFG.batch_size = 16       # if OOM -> 8 (and increase grad_accum)
CFG.grad_accum = 1        # if batch_size=8 -> set 2
CFG.num_workers = 4       # 4 is usually stable; try 8 if you want
CFG.epochs = 6            # you can raise to 8 if it’s fast
CFG.lr = 2e-4
CFG.model_name = "tf_efficientnet_b0.ns_jft_in1k"
CFG.use_amp = True
CFG.device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", CFG.device)



import time
import torch

def estimate_epoch_time_from_loader(loader, n_batches=200, device=None):
    if device is None:
        device = CFG.device if "CFG" in globals() else ("cuda" if torch.cuda.is_available() else "cpu")

    it = iter(loader)

    # warmup (a few batches)
    warm = min(10, n_batches)
    for _ in range(warm):
        x, y = next(it)
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

    # measure
    t0 = time.time()
    n = 0
    for _ in range(n_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        n += 1

    dt = time.time() - t0
    sec_per_batch = dt / max(1, n)

    batches_per_epoch = len(loader)
    est_epoch_sec = sec_per_batch * batches_per_epoch

    print("measured batches:", n)
    print("sec per batch (data pipeline):", round(sec_per_batch, 3))
    print("batches per epoch:", batches_per_epoch)
    print("estimated epoch time (minutes):", round(est_epoch_sec / 60, 1))

    return sec_per_batch, est_epoch_sec



fold = CFG.fold
tr_df = df[df.fold != fold].reset_index(drop=True)
va_df = df[df.fold == fold].reset_index(drop=True)

train_loader, valid_loader = make_png_loaders(tr_df, va_df)

sec_per_batch, est_epoch_sec = estimate_epoch_time_from_loader(train_loader, n_batches=200)
print("sec/batch:", sec_per_batch, "epoch minutes:", est_epoch_sec/60)



import time
import torch
import numpy as np

def benchmark_train_steps(model, loader, criterion, optimizer, scaler, n_steps=200):
    model.train()
    t0 = time.time()
    it = iter(loader)
    steps = 0
    losses = []

    for _ in range(n_steps):
        try:
            x, y = next(it)
        except StopIteration:
            break

        x = x.to(CFG.device, non_blocking=True)
        y = y.to(CFG.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type="cuda", enabled=(CFG.use_amp and CFG.device=="cuda")):
            logits = model(x)
            loss = criterion(logits, y)

        if CFG.use_amp and CFG.device == "cuda":
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        steps += 1

    dt = time.time() - t0
    sec_per_step = dt / max(1, steps)
    est_epoch_min = (sec_per_step * len(loader)) / 60

    print("steps:", steps)
    print("sec/step (full train compute):", round(sec_per_step, 3))
    print("estimated epoch time (minutes):", round(est_epoch_min, 1))
    print("mean loss:", round(float(np.mean(losses)), 4))
    return sec_per_step, est_epoch_min



fold = CFG.fold
tr_df = df[df.fold != fold].reset_index(drop=True)
va_df = df[df.fold == fold].reset_index(drop=True)
train_loader, valid_loader = make_png_loaders(tr_df, va_df)

model, criterion, optimizer, scheduler, scaler = build_components(tr_df)

_ = benchmark_train_steps(model, train_loader, criterion, optimizer, scaler, n_steps=200)



CFG.epochs = 6
CFG.num_workers = 4   # stable; 8 is ok too
CFG.use_amp = True
CFG.device = "cuda" if torch.cuda.is_available() else "cpu"
FOLDS_TO_TRAIN = [0, 1, 2]
print("device:", CFG.device, "folds:", FOLDS_TO_TRAIN, "epochs:", CFG.epochs)



import time
from pathlib import Path
import numpy as np

def train_one_fold(fold):
    tr_df = df[df.fold != fold].reset_index(drop=True)
    va_df = df[df.fold == fold].reset_index(drop=True)

    train_loader, valid_loader = make_png_loaders(tr_df, va_df)
    model, criterion, optimizer, scheduler, scaler = build_components(tr_df)

    best_f1 = -1.0
    best_path = Path(f"/kaggle/working/best_density_fold{fold}.pth")
    last_path = Path(f"/kaggle/working/last_density_fold{fold}.pth")

    for epoch in range(CFG.epochs):
        t0 = time.time()

        tr_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler)
        val_acc, val_f1, cm = evaluate(model, valid_loader)
        scheduler.step()

        dt = time.time() - t0
        print(f"[fold {fold}] epoch {epoch+1}/{CFG.epochs} loss={tr_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} time={dt/60:.1f} min")
        print("cm:\n", cm)

        # always save last (crash-safe)
        torch.save(
            {"model": model.state_dict(), "fold": fold, "epoch": epoch+1, "cfg": CFG.__dict__, "dens2id": dens2id},
            last_path
        )

        # save best
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {"model": model.state_dict(), "fold": fold, "epoch": epoch+1, "cfg": CFG.__dict__, "dens2id": dens2id},
                best_path
            )

    return best_f1, str(best_path), str(last_path)



from pathlib import Path
p = Path("/kaggle/working/WRITE_TEST.txt")
p.write_text("write test ok\n")
print("wrote:", p, "exists:", p.exists(), "size:", p.stat().st_size)
!ls -lah /kaggle/working | head -50



results = []
for fold in FOLDS_TO_TRAIN:
    best_f1, best_path, last_path = train_one_fold(fold)
    results.append((fold, best_f1, best_path, last_path))

print("\n=== CV RESULTS ===")
for fold, f1, best_path, last_path in results:
    print(f"fold {fold}: best_f1={f1:.4f} best={best_path}")

mean_f1 = float(np.mean([r[1] for r in results]))
std_f1  = float(np.std([r[1] for r in results]))
print(f"mean best_f1 over folds: {mean_f1:.4f} ± {std_f1:.4f}")



print(f"fold {fold} using model {CFG.model_name}")



from pathlib import Path
p = Path("/kaggle/working/WRITE_TEST.txt")
p.write_text("write test ok\n")
print("wrote:", p, "exists:", p.exists(), "size:", p.stat().st_size)
!ls -lah /kaggle/working | head -50



import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# ---------- output dir ----------
EVAL_DIR = Path("/kaggle/working/eval_outputs")
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# ---------- which folds exist ----------
fold_ckpts = {}
for f in [0,1,2,3,4]:
    p = Path(f"/kaggle/working/best_density_fold{f}.pth")
    if p.exists():
        fold_ckpts[f] = str(p)

print("Found checkpoints:", fold_ckpts)
assert len(fold_ckpts) > 0, "No best_density_fold*.pth found in /kaggle/working"

# ---------- model build ----------
def build_eval_model():
    return Model(CFG.model_name, NUM_CLASSES)

@torch.no_grad()
def predict_on_loader(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(CFG.device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(dim=1).detach().cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    return np.concatenate(ys), np.concatenate(ps)

lines = []
def log(s):
    print(s)
    lines.append(s)

rows = []

for fold, ckpt_path in fold_ckpts.items():
    tr_df = df[df.fold != fold].reset_index(drop=True)
    va_df = df[df.fold == fold].reset_index(drop=True)
    _, valid_loader = make_png_loaders(tr_df, va_df)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = build_eval_model()
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(CFG.device)

    y_true, y_pred = predict_on_loader(model, valid_loader)

    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    cm  = confusion_matrix(y_true, y_pred)

    cm_path = EVAL_DIR / f"cm_fold{fold}.csv"
    pd.DataFrame(
        cm,
        index=[f"true_{id2dens[i]}" for i in range(NUM_CLASSES)],
        columns=[f"pred_{id2dens[i]}" for i in range(NUM_CLASSES)],
    ).to_csv(cm_path, index=True)

    rep = classification_report(
        y_true, y_pred,
        target_names=[id2dens[i] for i in range(NUM_CLASSES)],
        digits=4
    )

    log("\n" + "="*90)
    log(f"FOLD {fold} | model={CFG.model_name} | img={CFG.img_size} | bs={CFG.batch_size}")
    log(f"accuracy={acc:.4f}  macroF1={f1m:.4f}")
    log("confusion matrix (rows=true, cols=pred):")
    log(str(cm))
    log("per-class report:")
    log(rep)
    log(f"saved confusion matrix: {cm_path}")

    rows.append({"fold": fold, "accuracy": acc, "macroF1": f1m})

metrics_df = pd.DataFrame(rows).sort_values("fold")
acc_mean, acc_std = metrics_df["accuracy"].mean(), metrics_df["accuracy"].std()
f1_mean,  f1_std  = metrics_df["macroF1"].mean(),  metrics_df["macroF1"].std()

log("\n" + "="*90)
log("SUMMARY (mean ± std over folds)")
log(f"accuracy: {acc_mean:.4f} ± {acc_std:.4f}")
log(f"macroF1 : {f1_mean:.4f} ± {f1_std:.4f}")
log("="*90)

report_path = EVAL_DIR / "evaluation_report.txt"
report_path.write_text("\n".join(lines))

metrics_path = EVAL_DIR / "fold_metrics.csv"
metrics_df.to_csv(metrics_path, index=False)

print("\nFILES NOW ON DISK:")
!ls -lah /kaggle/working/eval_outputs | head -200
print("\nFirst lines of report:")
!head -40 /kaggle/working/eval_outputs/evaluation_report.txt



import random
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from pathlib import Path

XAI_DIR = Path("/kaggle/working/gradcam_outputs")
XAI_DIR.mkdir(parents=True, exist_ok=True)

fold_for_xai = sorted(list(fold_ckpts.keys()))[0]
ckpt_path = fold_ckpts[fold_for_xai]
print("Using fold for XAI:", fold_for_xai, ckpt_path)

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.h1 = target_layer.register_forward_hook(self._forward_hook)
        self.h2 = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inp, out):
        self.activations = out

    def _backward_hook(self, module, grad_in, grad_out):
        self.gradients = grad_out[0]

    def close(self):
        self.h1.remove()
        self.h2.remove()

    def __call__(self, x, class_idx=None):
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[:, class_idx].sum()
        score.backward()

        grads = self.gradients
        acts  = self.activations
        weights = grads.mean(dim=(2,3), keepdim=True)
        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-6)
        return cam.squeeze().detach().cpu().numpy(), logits.detach().cpu()

def find_last_conv_layer(model):
    for name, m in reversed(list(model.named_modules())):
        if isinstance(m, torch.nn.Conv2d):
            return name, m
    raise RuntimeError("No Conv2d layer found")

def overlay_cam(gray_img, cam, out_size):
    gray = cv2.resize(gray_img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.resize(heat, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    overlay = cv2.addWeighted(base, 0.55, heat, 0.45, 0)
    return overlay

# load model
ckpt = torch.load(ckpt_path, map_location="cpu")
model = Model(CFG.model_name, NUM_CLASSES)
model.load_state_dict(ckpt["model"], strict=True)
model.to(CFG.device)
model.eval()

layer_name, target_layer = find_last_conv_layer(model)
print("Grad-CAM layer:", layer_name)

va_df = df[df.fold == fold_for_xai].reset_index(drop=True)

# pick 12 examples ~3 per class
picked = []
for cls in [0,1,2,3]:
    idxs = va_df.index[va_df["target"] == cls].tolist()
    random.shuffle(idxs)
    picked.extend(idxs[:3])

if len(picked) < 12:
    rem = [i for i in range(len(va_df)) if i not in picked]
    random.shuffle(rem)
    picked.extend(rem[:(12-len(picked))])
picked = picked[:12]

cammer = GradCAM(model, target_layer)

saved = []
for k, idx in enumerate(picked, start=1):
    r = va_df.iloc[idx]
    png_path = CACHE / str(r.patient_id) / f"{r.image_id}.png"
    img = cv2.imread(str(png_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        continue

    img = cv2.resize(img, (CFG.img_size, CFG.img_size), interpolation=cv2.INTER_AREA)
    img3 = np.stack([img, img, img], axis=-1)
    x = get_transforms(False)(image=img3)["image"].unsqueeze(0).to(CFG.device)

    cam, logits = cammer(x)
    pred = int(logits.argmax(dim=1).item())
    true = int(r.target)

    overlay = overlay_cam(img, cam, CFG.img_size)

    out_name = f"fold{fold_for_xai}_k{k:02d}_true{id2dens[true]}_pred{id2dens[pred]}_pid{r.patient_id}_img{r.image_id}.png"
    out_path = XAI_DIR / out_name
    cv2.imwrite(str(out_path), overlay)
    saved.append(out_path)

cammer.close()

print("Saved", len(saved), "Grad-CAM overlays to:", XAI_DIR)
!ls -lah /kaggle/working/gradcam_outputs | head -200



import json
import torch
import pandas as pd
from pathlib import Path

MODEL_DIR = Path("/kaggle/working/model_exports")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# choose your best fold (based on your results fold 0 was best)
FINAL_FOLD = 0
src_ckpt = Path(f"/kaggle/working/best_density_fold{FINAL_FOLD}.pth")
assert src_ckpt.exists(), f"missing checkpoint: {src_ckpt}"

ckpt = torch.load(str(src_ckpt), map_location="cpu")

# save weights-only (best for portability)
weights_path = MODEL_DIR / "final_model_state_dict.pth"
torch.save(ckpt["model"], str(weights_path))

meta = {
    "task": "ABCD_density_classification",
    "num_classes": NUM_CLASSES,
    "model_name": CFG.model_name,
    "img_size": CFG.img_size,
    "normalization_mean": [0.5,0.5,0.5],
    "normalization_std": [0.25,0.25,0.25],
    "id2dens": id2dens,
    "dens2id": dens2id,
}
meta_path = MODEL_DIR / "model_meta.json"
meta_path.write_text(json.dumps(meta, indent=2))

map_df = pd.DataFrame({"class_id": list(id2dens.keys()), "label": list(id2dens.values())})
map_path = MODEL_DIR / "label_mapping.csv"
map_df.to_csv(map_path, index=False)

print("Exported model package:")
print(" -", weights_path)
print(" -", meta_path)
print(" -", map_path)
!ls -lah /kaggle/working/model_exports | head -200


