
import os
import math
import random
import time
import gc
from pathlib import Path
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import models, transforms

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

SEED = 42
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)




# Try to autodetect Kaggle input path. Fallback to a relative path.
DEFAULT_KAGGLE_DIR = Path("/kaggle/input/histopathologic-cancer-detection")
LOCAL_DIR = Path("../input/histopathologic-cancer-detection")

if DEFAULT_KAGGLE_DIR.exists():
    DATA_DIR = DEFAULT_KAGGLE_DIR
elif LOCAL_DIR.exists():
    DATA_DIR = LOCAL_DIR
else:
    # Edit this path if running elsewhere
    DATA_DIR = Path("./histopathologic-cancer-detection")

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR  = DATA_DIR / "test"
LABELS_CSV = DATA_DIR / "train_labels.csv"

print("DATA_DIR:", DATA_DIR)
print("TRAIN_DIR:", TRAIN_DIR.exists(), "TEST_DIR:", TEST_DIR.exists(), "LABELS_CSV:", LABELS_CSV.exists())




@dataclass
class Config:
    image_size: int = 128              # ResNet can take 128x128
    batch_size: int = 256
    epochs: int = 2
    lr: float = 3e-4
    weight_decay: float = 1e-4
    num_workers: int = 2
    folds: int = 5
    train_fold: int = 0                # which fold to train in this run
    patience: int = 2                  # early stopping patience
    t_max: int = 6                     # CosineAnnealingLR T_max
    model_name: str = "resnet18"
    save_dir: str = "./checkpoints"
    mixed_precision: bool = True

CFG = Config()
os.makedirs(CFG.save_dir, exist_ok=True)
print(asdict(CFG))




class HistoDataset(Dataset):
    def __init__(self, df, img_dir, image_size=128, is_train=True):
        self.df = df.reset_index(drop=True).copy()
        self.img_dir = Path(img_dir)
        self.is_train = is_train
        size = image_size

        self.train_tfms = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25]),
        ])
        self.valid_tfms = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.25,0.25,0.25]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_id = row["id"]
        path = self.img_dir / f"{img_id}.tif"
        with Image.open(path) as im:
            im = im.convert("RGB")
            if self.is_train:
                im = self.train_tfms(im)
            else:
                im = self.valid_tfms(im)

        if "label" in row:
            y = torch.tensor(row["label"], dtype=torch.float32)
            return im, y
        else:
            return im, img_id




def build_model(model_name="resnet18"):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Replace the final layer for binary classification
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, 1)
    else:
        raise ValueError("Unsupported model_name")
    return model

def get_class_weights(labels):
    # Compute positive and negative weights for BCEWithLogitsLoss pos_weight
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0:
        return torch.tensor([1.0])
    pos_weight = torch.tensor([neg / max(pos, 1.0)])
    return pos_weight

def train_one_epoch(model, loader, criterion, optimizer, scaler=None):
    model.train()
    running_loss = 0.0
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE)
        targets = targets.view(-1, 1).to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type=DEVICE if DEVICE != "cpu" else "cpu", dtype=torch.float16, enabled=True):
                logits = model(imgs)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    probs_list = []
    targets_list = []
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE)
        targets = targets.view(-1, 1).to(DEVICE)
        logits = model(imgs)
        loss = criterion(logits, targets)
        running_loss += loss.item() * imgs.size(0)
        probs = torch.sigmoid(logits).detach().cpu().numpy().ravel().tolist()
        probs_list.extend(probs)
        targets_list.extend(targets.detach().cpu().numpy().ravel().tolist())

    avg_loss = running_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(targets_list, probs_list)
    except Exception:
        auc = float("nan")
    return avg_loss, auc, np.array(probs_list), np.array(targets_list)




assert LABELS_CSV.exists(), "train_labels.csv not found. Please set DATA_DIR correctly."
labels_df = pd.read_csv(LABELS_CSV)
print(labels_df.head(), labels_df.shape, labels_df.label.mean())
skf = StratifiedKFold(n_splits=CFG.folds, shuffle=True, random_state=SEED)

folds = []
for f, (_, val_idx) in enumerate(skf.split(labels_df["id"], labels_df["label"])):
    fold = labels_df.copy()
    fold["fold"] = -1
    fold.loc[val_idx, "fold"] = f
    folds.append(fold)

# Use one fold for a quick baseline
df = folds[CFG.train_fold]
train_df = df[df.fold != CFG.train_fold].drop(columns=["fold"])
valid_df = df[df.fold == CFG.train_fold].drop(columns=["fold"])

print("Train size:", len(train_df), "Valid size:", len(valid_df))




train_ds = HistoDataset(train_df, TRAIN_DIR, image_size=CFG.image_size, is_train=True)
valid_ds = HistoDataset(valid_df, TRAIN_DIR, image_size=CFG.image_size, is_train=False)

# Weighted sampling to reduce class imbalance
labels_np = train_df["label"].values
class_sample_count = np.array([len(np.where(labels_np == t)[0]) for t in [0,1]])
weights = 1. / class_sample_count
samples_weight = np.array([weights[t] for t in labels_np])
samples_weight = torch.from_numpy(samples_weight).float()
sampler = WeightedRandomSampler(samples_weight, num_samples=len(samples_weight), replacement=True)

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, sampler=sampler,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
valid_loader = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False,
                          num_workers=CFG.num_workers, pin_memory=True, drop_last=False)




model = build_model(CFG.model_name).to(DEVICE)

# Loss
pos_weight = get_class_weights(train_df["label"].values).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=CFG.t_max)

scaler = torch.cuda.amp.GradScaler(enabled=CFG.mixed_precision and DEVICE == "cuda")

best_auc = -1.0
best_path = os.path.join(CFG.save_dir, f"{CFG.model_name}_fold{CFG.train_fold}.pt")
no_improve = 0

for epoch in range(1, CFG.epochs + 1):
    t0 = time.time()
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, scaler=scaler)
    val_loss, val_auc, _, _ = validate(model, valid_loader, criterion)
    scheduler.step()
    dt = time.time() - t0
    print(f"Epoch {epoch:02d} | {dt:.1f}s | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | val_auc {val_auc:.4f}")

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save({"model": model.state_dict(), "cfg": asdict(CFG)}, best_path)
        no_improve = 0
        print("Saved new best model to", best_path)
    else:
        no_improve += 1
        if no_improve >= CFG.patience:
            print("Early stopping triggered")
            break

print("Best AUC:", best_auc)




# Build test loader
assert TEST_DIR.exists(), "test directory not found"
test_ids = [p.stem for p in sorted(TEST_DIR.glob("*.tif"))]
test_df = pd.DataFrame({"id": test_ids})

test_ds = HistoDataset(test_df, TEST_DIR, image_size=CFG.image_size, is_train=False)
test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False,
                         num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

# Load best checkpoint
ckpt = torch.load(best_path, map_location=DEVICE)
model = build_model(CFG.model_name).to(DEVICE)
model.load_state_dict(ckpt["model"])
model.eval()

probs = []
with torch.no_grad():
    for imgs, ids in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        p = torch.sigmoid(logits).detach().cpu().numpy().ravel()
        probs.extend(p.tolist())

submission = pd.DataFrame({"id": test_df["id"], "label": np.array(probs)})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote", submission_path, "with", len(submission), "rows")




# Sanity check on validation predictions and optimal threshold
val_loss, val_auc, val_probs, val_targets = validate(model, valid_loader, criterion)
print("Validation AUC with reloaded model:", val_auc)

# Find a threshold that maximizes F1 on the validation set
best_thr, best_f1 = 0.5, -1
for thr in np.linspace(0.1, 0.9, 17):
    preds = (val_probs >= thr).astype(int)
    tp = ((preds == 1) & (val_targets == 1)).sum()
    fp = ((preds == 1) & (val_targets == 0)).sum()
    fn = ((preds == 0) & (val_targets == 1)).sum()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    if f1 > best_f1:
        best_f1, best_thr = f1, thr
print(f"Best F1 {best_f1:.4f} at thr {best_thr:.2f}")


