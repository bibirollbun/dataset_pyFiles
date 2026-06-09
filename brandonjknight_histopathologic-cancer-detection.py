# Step 1. Setup
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from PIL import Image

# Torch / DL
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# Metrics
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Check environment
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())



# Define paths (Kaggle auto-mounts competition data under /kaggle/input)
DATA_DIR = "/kaggle/input/histopathologic-cancer-detection"

TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS = os.path.join(DATA_DIR, "train_labels.csv")

# Load labels
df = pd.read_csv(LABELS)
print(df.head())
print("Train images:", len(os.listdir(TRAIN_DIR)))
print("Test images:", len(os.listdir(TEST_DIR)))


# Randomly select and display one image with its label to verify dataset integrity

sample_id = df.sample(1).iloc[0]["id"]
label = df.sample(1).iloc[0]["label"]

img_path = os.path.join(TRAIN_DIR, f"{sample_id}.tif")
img = Image.open(img_path)
plt.imshow(img)
plt.title(f"Label: {label}")
plt.axis("off")
plt.show()


# Plot class balance
ax = df['label'].value_counts().sort_index().plot(
    kind='bar',
    color=['steelblue', 'indianred']
)
ax.set_xticklabels(['Non-cancer (0)', 'Cancer (1)'], rotation=0)
ax.set_title("Class Balance in Training Data")
ax.set_ylabel("Count")
plt.show()

print("Class distribution:")
print(df['label'].value_counts(normalize=True))



# Show random samples from each class
def show_samples(label, n=6):
    sample_df = df[df['label'] == label].sample(n)
    plt.figure(figsize=(12, 6))
    for i, img_id in enumerate(sample_df['id']):
        img_path = os.path.join(TRAIN_DIR, f"{img_id}.tif")
        img = Image.open(img_path)
        plt.subplot(2, 3, i+1)
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"Label: {label}")
    plt.suptitle(f"Random Samples (label={label})")
    plt.show()

show_samples(0, n=6)  # non-cancer
show_samples(1, n=6)  # cancer



# Verify that every ID in the labels CSV has a corresponding image file
def image_exists(img_id):
    path = os.path.join(TRAIN_DIR, f"{img_id}.tif")
    return os.path.exists(path)

df['exists'] = df['id'].apply(image_exists)

missing = (~df['exists']).sum()
print(f"Missing images: {missing}")

# Drop any rows without a corresponding image
df = df[df['exists']].drop(columns=['exists']).reset_index(drop=True)
print("Cleaned dataset size:", len(df))



# Check unique label values
print("Unique labels:", df['label'].unique())

# Quick counts
print(df['label'].value_counts())



# Try opening a small sample of images to confirm no corruption
from tqdm.notebook import tqdm

corrupt_count = 0
for img_id in tqdm(df['id'].sample(500)):  # check random 500 images
    path = os.path.join(TRAIN_DIR, f"{img_id}.tif")
    try:
        _ = Image.open(path)
    except:
        corrupt_count += 1

print("Corrupt images found:", corrupt_count)


from sklearn.model_selection import train_test_split
from torchvision import transforms

# 80/20 stratified split
train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"]
)
print("Train size:", len(train_df), " | Val size:", len(val_df))
print("Train class counts:\n", train_df["label"].value_counts())
print("Val class counts:\n",   val_df["label"].value_counts())

# Image size (competition tiles are 96x96)
IMG_SIZE = 96

# ImageNet normalization (for pretrained CNNs)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

# Data augmentations for train; only normalization for val
train_tfms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

val_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
import torch

class HistoDataset(Dataset):
    def __init__(self, df, img_dir, tfms=None, return_ids=False):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.tfms = tfms
        self.return_ids = return_ids

    def __len__(self):
        return len(self.df)

    def _img_path(self, _id):
        # Competition files are .tif; if you ever switch sources, you can add fallbacks here
        return self.img_dir / f"{_id}.tif"

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._img_path(row["id"])
        img = Image.open(img_path).convert("RGB")
        if self.tfms:
            img = self.tfms(img)
        label = torch.tensor(row["label"], dtype=torch.long)
        if self.return_ids:
            return img, label, row["id"]
        return img, label

BATCH_SIZE = 128
NUM_WORKERS = 2  # Kaggle usually does fine with 2–4

train_ds = HistoDataset(train_df, TRAIN_DIR, tfms=train_tfms)
val_ds   = HistoDataset(val_df,   TRAIN_DIR, tfms=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)

len(train_ds), len(val_ds)



# WeightedRandomSampler so each batch is more balanced
from torch.utils.data import WeightedRandomSampler
import numpy as np

use_sampler = False  # set to True to enable

if use_sampler:
    class_counts = train_df["label"].value_counts().to_dict()
    # weight for each class = 1 / freq
    weights = train_df["label"].map(lambda y: 1.0 / class_counts[y]).values
    sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=NUM_WORKERS, pin_memory=True)
    print("Using WeightedRandomSampler")
else:
    print("Using random shuffle (no sampler)")


# Device setup
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)


# Device setup
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Model setup
import torch.nn as nn
from torchvision import models

def build_model():
    # No internet in Kaggle => don't request pretrained weights
    model = models.resnet18(weights=None)  # randomly initialized
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)   # binary logit
    return model

model = build_model().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


from pathlib import Path
import torch, json, pandas as pd

# Where to save during this run
CKPT_DIR = Path("/kaggle/working/checkpoints")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
LAST_CKPT = CKPT_DIR / "last.pt"
BEST_CKPT = CKPT_DIR / "best.pt"
HIST_CSV  = CKPT_DIR / "history.csv"

def save_checkpoint(epoch, model, optimizer, scheduler, best_auc, history_df):
    torch.save({
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "best_auc": best_auc,
    }, LAST_CKPT)
    # persist history table too
    history_df.to_csv(HIST_CSV, index=False)

def save_best(model, best_auc):
    torch.save({"model_state": model.state_dict(), "best_auc": best_auc}, BEST_CKPT)

def try_resume(model, optimizer=None, scheduler=None):
    # 1) Try local /kaggle/working checkpoint
    src = None
    if LAST_CKPT.exists():
        src = LAST_CKPT
    else:
        # 2) Or try prior notebook output you attached via "Add Data"
        #    (edit this path to match the mounted output dataset name)
        prev = Path("/kaggle/input/your-previous-notebook-output/checkpoints/last.pt")
        if prev.exists():
            src = prev

    start_epoch, best_auc = 1, -float("inf")
    if src is not None:
        state = torch.load(src, map_location="cpu")
        model.load_state_dict(state["model_state"])
        if optimizer is not None and "optimizer_state" in state and state["optimizer_state"] is not None:
            optimizer.load_state_dict(state["optimizer_state"])
        if scheduler is not None and "scheduler_state" in state and state["scheduler_state"] is not None:
            scheduler.load_state_dict(state["scheduler_state"])
        best_auc = float(state.get("best_auc", best_auc))
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"Resumed from {src} at epoch {start_epoch}, best_auc={best_auc:.4f}")
    else:
        print("No checkpoint to resume from; starting fresh.")
    # load prior history if present
    hist_path = HIST_CSV if HIST_CSV.exists() else Path(str(CKPT_DIR).replace("/working/","/input/your-previous-notebook-output/"))/"checkpoints/history.csv"
    if hist_path.exists():
        hist_df = pd.read_csv(hist_path)
    else:
        hist_df = pd.DataFrame(columns=["epoch","train_loss","val_roc_auc","val_acc","val_f1","val_pr_auc"])
    return start_epoch, best_auc, hist_df



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score, confusion_matrix,
    precision_recall_curve, roc_curve, auc
)
import torch

@torch.no_grad()
def evaluate(model, loader, device=DEVICE):
    model.eval()
    all_probs, all_labels = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x).squeeze(1)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        all_probs.append(probs)
        all_labels.append(y.detach().cpu().numpy())
    probs  = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds  = (probs >= 0.5).astype(int)

    roc  = roc_auc_score(labels, probs) if len(np.unique(labels))>1 else np.nan
    acc  = accuracy_score(labels, preds)
    f1   = f1_score(labels, preds, zero_division=0)
    prec, rec, _ = precision_recall_curve(labels, probs)
    pr_auc = auc(rec, prec)

    return {
        "roc_auc": roc, "acc": acc, "f1": f1, "pr_auc": pr_auc,
        "probs": probs, "labels": labels, "preds": preds,
        "prec": prec, "rec": rec
    }

def plot_confusion(cm, labels=("Non-cancer","Cancer"), title="Confusion Matrix"):
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(title)
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, str(z), ha='center', va='center')
    plt.colorbar(im); plt.show()



from tqdm.auto import tqdm

scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, verbose=True
)

def train_one_epoch(model, loader, optimizer, device=DEVICE):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            logits = model(x).squeeze(1)
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)

EPOCHS = 12
patience = 3
best_auc = -np.inf
wait = 0
history = []

for epoch in range(1, EPOCHS+1):
    train_loss = train_one_epoch(model, train_loader, optimizer)
    val = evaluate(model, val_loader)

    scheduler.step(val["roc_auc"])  # reduce LR on plateau

    history.append({"epoch": epoch, "train_loss": train_loss,
                    "val_roc_auc": val["roc_auc"], "val_acc": val["acc"],
                    "val_f1": val["f1"], "val_pr_auc": val["pr_auc"]})
    print(f"Epoch {epoch:02d} | loss {train_loss:.4f} | "
          f"AUC {val['roc_auc']:.4f} | Acc {val['acc']:.4f} | "
          f"F1 {val['f1']:.4f} | PR-AUC {val['pr_auc']:.4f}")

    # early stopping on ROC-AUC
    if val["roc_auc"] > best_auc:
        best_auc = val["roc_auc"]
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            print("Early stopping triggered.")
            break

# restore best weights
model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
val_best = evaluate(model, val_loader)



# History table
hist_df = pd.DataFrame(history)
display(hist_df)

# Training loss
plt.figure()
plt.plot(hist_df["epoch"], hist_df["train_loss"])
plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title("Training Loss"); plt.show()

# Validation metrics
plt.figure()
plt.plot(hist_df["epoch"], hist_df["val_roc_auc"], label="ROC-AUC")
plt.plot(hist_df["epoch"], hist_df["val_acc"], label="Accuracy")
plt.plot(hist_df["epoch"], hist_df["val_f1"], label="F1")
plt.plot(hist_df["epoch"], hist_df["val_pr_auc"], label="PR-AUC")
plt.xlabel("Epoch"); plt.title("Validation Metrics"); plt.legend(); plt.show()

# ROC curve
fpr, tpr, _ = roc_curve(val_best["labels"], val_best["probs"])
rocA = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={rocA:.3f}")
plt.plot([0,1],[0,1],'--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("ROC Curve"); plt.legend(); plt.show()

# PR curve
plt.figure()
plt.plot(val_best["rec"], val_best["prec"], label=f"PR-AUC={val_best['pr_auc']:.3f}")
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend(); plt.show()

# Confusion matrix at 0.5 threshold
cm = confusion_matrix(val_best["labels"], (val_best["probs"]>=0.5).astype(int))
plot_confusion(cm)
print({
    "Val ROC-AUC": float(val_best["roc_auc"]),
    "Val PR-AUC": float(val_best["pr_auc"]),
    "Val Accuracy": float(val_best["acc"]),
    "Val F1": float(val_best["f1"])
})





