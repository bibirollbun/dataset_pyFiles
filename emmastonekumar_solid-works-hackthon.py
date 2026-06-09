# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os, glob, random
from pathlib import Path
import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as T
import torchvision.models as tvm

# --------------------
# Config
# --------------------
SEED = 42
IMG_SIZE = 384          # if OOM, reduce to 320 or 256
BATCH_SIZE = 16
EPOCHS = 12
LR = 2e-4

PARTS = ["bolt", "locatingpin", "nut", "washer"]
REQUIRED = {"image_name", *PARTS}

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# --------------------
# Auto-find competition directory + files
# --------------------
ROOT = Path("/kaggle/input")

# find a directory that contains images + csv
comp_dir = None
best_score = -1
for d in ROOT.iterdir():
    if not d.is_dir():
        continue
    imgs = list(d.rglob("*.png")) + list(d.rglob("*.jpg")) + list(d.rglob("*.jpeg"))
    csvs = list(d.rglob("*.csv"))
    if len(imgs) == 0 or len(csvs) == 0:
        continue
    name = d.name.lower()
    score = (5 if "solidworks" in name else 0) + (5 if "hackathon" in name else 0) + min(10, len(imgs)//50) + min(10, len(csvs))
    if score > best_score:
        best_score = score
        comp_dir = d

print("COMP_DIR:", comp_dir)

all_csv = sorted(comp_dir.rglob("*.csv"))
print("CSVs found:", [p.name for p in all_csv])

label_csv = None
sample_sub_csv = None
for p in all_csv:
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    cols = set(df.columns)
    if REQUIRED.issubset(cols):
        label_csv = p
    if "sample" in p.name.lower() and REQUIRED.issubset(cols):
        sample_sub_csv = p

print("label_csv:", label_csv)
print("sample_sub_csv:", sample_sub_csv)

train_df = pd.read_csv(label_csv)
train_df.head()


# --------------------
# Map image_name -> file path
# --------------------
img_files = []
for ext in ["*.png","*.jpg","*.jpeg"]:
    img_files += list(comp_dir.rglob(ext))

name2path = {p.name: p for p in img_files}

train_names = set(train_df["image_name"].astype(str).tolist())
train_paths = {n: name2path[n] for n in train_names if n in name2path}
print("Train images found:", len(train_paths), "/", len(train_names))

# Get test names: prefer sample_submission if present, else "all images not in train"
if sample_sub_csv is not None:
    sample_df = pd.read_csv(sample_sub_csv)
    test_names = sample_df["image_name"].astype(str).tolist()
else:
    test_names = sorted([n for n in name2path.keys() if n not in train_names])

print("Test images:", len(test_names))
print("Example test names:", test_names[:5])


# --------------------
# Dataset + transforms
# --------------------
max_counts = {c: int(train_df[c].max()) for c in PARTS}
print("Max counts:", max_counts)

train_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomApply([T.RandomRotation(180)], p=0.7),
    T.RandomHorizontalFlip(0.5),
    T.RandomVerticalFlip(0.5),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

valid_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

class PartsDataset(Dataset):
    def __init__(self, df, name2path, tfms, is_test=False):
        self.df = df.reset_index(drop=True)
        self.name2path = name2path
        self.tfms = tfms
        self.is_test = is_test

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = str(row["image_name"])
        img = Image.open(self.name2path[name]).convert("RGB")
        x = self.tfms(img)

        if self.is_test:
            return name, x

        y = torch.tensor([int(row[c]) for c in PARTS], dtype=torch.long)
        return name, x, y


# --------------------
# Stratified split by (bolt,pin,nut,washer) tuple
# --------------------
from collections import defaultdict

def stratified_split(df, valid_frac=0.15, seed=42):
    rng = random.Random(seed)
    groups = defaultdict(list)
    for i, r in df.reset_index(drop=True).iterrows():
        key = tuple(int(r[c]) for c in PARTS)
        groups[key].append(i)

    tr_idx, va_idx = [], []
    for k, idxs in groups.items():
        rng.shuffle(idxs)
        n = len(idxs)
        n_valid = 1 if n <= 2 else max(1, int(n * valid_frac))
        va_idx.extend(idxs[:n_valid])
        tr_idx.extend(idxs[n_valid:])
    return np.array(tr_idx), np.array(va_idx)

tr_idx, va_idx = stratified_split(train_df, valid_frac=0.15, seed=SEED)
print(len(tr_idx), len(va_idx))

tr_df = train_df.iloc[tr_idx].copy()
va_df = train_df.iloc[va_idx].copy()

tr_ds = PartsDataset(tr_df, train_paths, train_tfms, is_test=False)
va_ds = PartsDataset(va_df, train_paths, valid_tfms, is_test=False)

tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)


# --------------------
# Model: ResNet34 backbone + 4 heads (count classification)
# --------------------
class MultiHeadResNet(nn.Module):
    def __init__(self, max_counts):
        super().__init__()
        try:
            base = tvm.resnet34(weights=tvm.ResNet34_Weights.IMAGENET1K_V1)
            print("Loaded ResNet34 pretrained.")
        except Exception as e:
            print("No pretrained weights, random init:", repr(e))
            base = tvm.resnet34(weights=None)

        feat_dim = base.fc.in_features
        base.fc = nn.Identity()
        self.backbone = base
        self.dropout = nn.Dropout(0.2)

        self.heads = nn.ModuleDict({
            p: nn.Linear(feat_dim, max_counts[p] + 1) for p in PARTS
        })

    def forward(self, x):
        f = self.backbone(x)
        f = self.dropout(f)
        return {p: self.heads[p](f) for p in PARTS}

model = MultiHeadResNet(max_counts).to(device)

criterions = {p: nn.CrossEntropyLoss() for p in PARTS}
opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))


# --------------------
# Train + validate (exact-match)
# --------------------
def exact_match(preds, y):
    # preds, y: [B,4]
    return (preds == y).all(dim=1).float().mean().item()

best_em = -1.0
best_state = None

for epoch in range(1, EPOCHS+1):
    model.train()
    tr_loss = 0.0
    for _, x, y in tr_loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = 0.0
            for j, p in enumerate(PARTS):
                loss = loss + criterions[p](logits[p], y[:, j])

        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        tr_loss += loss.item() * x.size(0)

    tr_loss /= len(tr_ds)

    model.eval()
    va_loss = 0.0
    em_list = []
    with torch.no_grad():
        for _, x, y in va_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            loss = 0.0
            pred_cols = []
            for j, p in enumerate(PARTS):
                loss = loss + criterions[p](logits[p], y[:, j])
                pred_cols.append(torch.argmax(logits[p], dim=1))
            preds = torch.stack(pred_cols, dim=1)

            va_loss += loss.item() * x.size(0)
            em_list.append((preds == y).all(dim=1).float().cpu())

    va_loss /= len(va_ds)
    va_em = torch.cat(em_list).mean().item()

    print(f"epoch {epoch:02d} | train_loss {tr_loss:.4f} | valid_loss {va_loss:.4f} | exact_match {va_em:.4f}")

    if va_em > best_em:
        best_em = va_em
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

print("Best valid exact-match:", best_em)

model.load_state_dict(best_state)
model.to(device)


# --------------------
# Predict test with simple rotation TTA + make submission.csv
# --------------------
test_df = pd.DataFrame({"image_name": test_names})
test_ds = PartsDataset(test_df, name2path, valid_tfms, is_test=True)
test_loader = DataLoader(test_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=True)

def rot90_batch(x, k):
    return torch.rot90(x, k=k, dims=(2,3))

@torch.no_grad()
def predict_tta(model, loader, rots=(0,1,2,3)):
    model.eval()
    rows = []
    for names, x in loader:
        x = x.to(device, non_blocking=True)

        probs_sum = {p: None for p in PARTS}
        for k in rots:
            xr = rot90_batch(x, k)
            logits = model(xr)
            for p in PARTS:
                pr = torch.softmax(logits[p], dim=1)
                probs_sum[p] = pr if probs_sum[p] is None else (probs_sum[p] + pr)

        pred_cols = []
        for p in PARTS:
            avgp = probs_sum[p] / len(rots)
            pred_cols.append(torch.argmax(avgp, dim=1))
        preds = torch.stack(pred_cols, dim=1).cpu().numpy()

        for i, n in enumerate(names):
            row = {"image_name": n}
            for j, p in enumerate(PARTS):
                row[p] = int(preds[i, j])
            rows.append(row)

    return pd.DataFrame(rows, columns=["image_name", *PARTS])

sub = predict_tta(model, test_loader, rots=(0,1,2,3))
sub.head()


sub.to_csv("submission.csv", index=False)
print("Saved submission.csv with shape:", sub.shape)
print(sub.columns.tolist())

