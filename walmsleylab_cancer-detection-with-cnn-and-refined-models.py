# Histopathologic Cancer Detection

import os, random, gc
from pathlib import Path
from contextlib import nullcontext
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import (
    roc_auc_score, accuracy_score, roc_curve, confusion_matrix,
    precision_recall_curve, average_precision_score
)
from tqdm.auto import tqdm
import shutil

# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

DATA_ROOT  = Path("/kaggle/input/histopathologic-cancer-detection")
TRAIN_DIR  = DATA_ROOT / "train"
LABELS_CSV = DATA_ROOT / "train_labels.csv"

# Model selection
MODEL_MODE = "pretrained"
RUN_ALL_MODELS = True
MODEL_GRID = ["simple", "improved", "multiscale", "pretrained"]
SHOW_PLOTS=True

FAST_DEMO  = True

IMG_SIZE    = 96
BATCH_SIZE  = 128
LR          = 0.01
MOMENTUM    = 0.9
WEIGHT_DECAY= 1e-4
EPOCHS_FULL = 9
DEMO_EPOCHS = 3
PATIENCE    = 3
SEED        = 42

DEMO_TRAIN = 8000
DEMO_VAL   = 2000

NUM_WORKERS = min(4, os.cpu_count() or 2)

FIG_SINGLE = (6, 4)
FIG_DOUBLE = (10, 4)

def fig_grid(rows, cols=3):
    return (4 * cols, 3 * rows)

plt.rcParams.update({
    "figure.dpi": 110,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "axes.labelsize": 10
})

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
gc.disable()

if DEVICE.type == "cuda":
    try:
        autocast = torch.amp.autocast
        scaler = torch.amp.GradScaler()
    except Exception:
        autocast = torch.cuda.amp.autocast
        scaler = torch.cuda.amp.GradScaler()
else:
    autocast = lambda *args, **kwargs: nullcontext()
    scaler = None

# Where to save artifacts
OUT_ROOT = Path("/kaggle/working")
RESULTS_ROOT = OUT_ROOT / "results"
MODELS_ROOT  = OUT_ROOT / "models"
PLOTS_ROOT   = RESULTS_ROOT / "plots"
RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
MODELS_ROOT.mkdir(parents=True, exist_ok=True)
PLOTS_ROOT.mkdir(parents=True, exist_ok=True)

def save_fig(name, dpi=120):
    path = PLOTS_ROOT / f"{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

# ---------------------------------------------------------------------
# Load labels
# ---------------------------------------------------------------------

labels = pd.read_csv(LABELS_CSV)

# ---------------------------------------------------------------------
# EDA (unchanged)
# ---------------------------------------------------------------------

def plot_class_samples(df, root, label, title, n=9, rows=3):
    assert n % rows == 0, "n must be divisible by rows"
    cols = n // rows

    subset = df[df["label"] == label].sample(n, random_state=SEED)

    fig, axes = plt.subplots(rows, cols, figsize=(2*cols, 2*rows))
    axes = axes.flatten()

    for ax, (_, r) in zip(axes, subset.iterrows()):
        img = Image.open(root / f"{r.id}.tif")
        ax.imshow(img)
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1)
    plt.tight_layout()
    save_fig(f"samples_label_{label}.png")

plot_class_samples(
    labels,
    TRAIN_DIR,
    label=0,
    title="Healthy Tissue Patches (Binary Class = 0)"
)

plot_class_samples(
    labels,
    TRAIN_DIR,
    label=1,
    title="Tumor Tissue Patches (Binary Class = 1)"
)

def plot_edge_samples(df, root, n=9, rows=3, min_std=10):
    """
    Displays samples as a patch-style grid:
    Original | Grayscale | Edges
    """
    assert n % rows == 0, "n must be divisible by rows"
    cols = 3  # Original | Grayscale | Edges

    samples = []
    while len(samples) < n:
        r = df.sample(1).iloc[0]
        img = cv2.imread(str(root / f"{r.id}.tif"))
        if img is None:
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if gray.std() < min_std:
            continue

        edges = cv2.Canny(gray, 50, 150)
        samples.append((img, gray, edges))

    fig, axes = plt.subplots(
        rows, cols,
        figsize=(2 * cols, 2 * rows),
        sharex=True,
        sharey=True
    )
    
    fig.suptitle(
        "Edge Structures in Patches",
        fontsize=14,
        fontweight="bold",
        y=1
    )
    
    for r in range(rows):
        img, gray, edges = samples[r]
    
        axes[r, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[r, 1].imshow(gray, cmap="gray")
        axes[r, 2].imshow(edges, cmap="gray")
    
        if r == 0:
            axes[r, 0].set_title("Original", fontweight="normal")
            axes[r, 1].set_title("Grayscale", fontweight="normal")
            axes[r, 2].set_title("Edges", fontweight="normal")
    
        for c in range(3):
            axes[r, c].axis("off")
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    save_fig("edge_structures.png")

plot_edge_samples(labels, TRAIN_DIR, n=9, rows=3)

# Class balance
class_counts_full = labels["label"].value_counts().sort_index()
colors = ["#f4c430", "#c0392b"]
plt.figure(figsize=FIG_SINGLE)
bars = plt.bar(
    ["0 (Healthy)", "1 (Tumor)"],
    class_counts_full.values,
    color=colors
)
plt.xlabel("Binary Class")
plt.ylabel("Number of Samples")
plt.suptitle(
    "Initial Binary Classification Distribution",
    fontsize=14,
    fontweight="bold",
    y=1
)
plt.margins(y=0.2)
for bar in bars:
    height = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        height,
        f"{int(height)}",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold"
    )
plt.tight_layout()
save_fig("class_balance.png")

# Per-image mean intensity
sample_means = []
for _, row in labels.sample(2000, random_state=SEED).iterrows():
    img = np.asarray(Image.open(TRAIN_DIR / f"{row.id}.tif")).astype(np.float32)
    sample_means.append(img.mean() / 255.0)
plt.figure(figsize=FIG_SINGLE)
plt.hist(sample_means, bins=40, color="tab:blue", edgecolor="black", alpha=0.85)
plt.xlabel("Mean Pixel Intensity")
plt.ylabel("Number of Samples")
plt.suptitle(
    "Per-Image Mean Pixel Intensity",
    fontsize=14,
    fontweight="bold",
    y=1
)
plt.tight_layout()
save_fig("mean_intensity.png")

def plot_color_space(df, root, n=2000):
    sample = df.sample(n, random_state=SEED)
    hues, sats = [], []
    for _, r in sample.iterrows():
        img = cv2.imread(str(root / f"{r.id}.tif"))
        if img is None:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, _ = cv2.split(hsv)
        hues.append(h.mean())
        sats.append(s.mean())

    plt.figure(figsize=FIG_SINGLE)
    plt.scatter(hues, sats, s=6, alpha=0.4)
    plt.xlabel("Mean Hue")
    plt.ylabel("Mean Saturation")
    plt.suptitle(
        "Per Patch Color Distribution (HSV Space)",
        fontsize=14,
        fontweight="bold",
        y=1
    )
    plt.grid(alpha=0.3)
    save_fig("color_space_hs.png")

plot_color_space(labels, TRAIN_DIR)

def plot_blob_sizes(df, root, n=300, min_area=30):
    sizes = []
    for _, r in df.sample(n, random_state=SEED).iterrows():
        img = cv2.imread(str(root / f"{r.id}.tif"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        edges = cv2.Canny(img, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > min_area:
                sizes.append(area)
    plt.figure(figsize=FIG_SINGLE)
    plt.hist(sizes, bins=50, edgecolor="black", alpha=0.8)
    plt.xscale("log")
    plt.xlabel("Blob Area (Pixels)")
    plt.ylabel("Number of Samples")
    plt.suptitle(
        "Structure Size as Identified by Edge Detection",
        fontsize=14,
        fontweight="bold",
        y=1
    )
    save_fig("blob_sizes.png")

plot_blob_sizes(labels, TRAIN_DIR)

def plot_rgb_means(df, root, n=1500):
    sample = df.sample(n, random_state=SEED)
    rgb_means = []
    for _, r in sample.iterrows():
        img = np.asarray(Image.open(root / f"{r.id}.tif")) / 255.0
        rgb_means.append(img.mean(axis=(0,1)))
    rgb_means = np.array(rgb_means)
    plt.figure(figsize=FIG_SINGLE)
    plt.hist(rgb_means[:,0], bins=40, alpha=0.6, label="Red")
    plt.hist(rgb_means[:,1], bins=40, alpha=0.6, label="Green")
    plt.hist(rgb_means[:,2], bins=40, alpha=0.6, label="Blue")
    plt.xlabel("Mean Channel Intensity")
    plt.ylabel("Number of Samples")
    plt.suptitle(
        "Per-Image Mean RGB Channel Intensities",
        fontsize=14,
        fontweight="bold",
        y=1
    )
    plt.legend()
    save_fig("rgb_means.png")

plot_rgb_means(labels, TRAIN_DIR)

# ---------------------------------------------------------------------
# Split data
# ---------------------------------------------------------------------

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
train_idx, val_idx = next(sss.split(labels[["id"]], labels["label"]))
train_df = labels.iloc[train_idx].reset_index(drop=True)
val_df   = labels.iloc[val_idx].reset_index(drop=True)
if FAST_DEMO:
    train_df = train_df.sample(DEMO_TRAIN, random_state=SEED)
    val_df   = val_df.sample(DEMO_VAL, random_state=SEED)

epochs = DEMO_EPOCHS if FAST_DEMO else EPOCHS_FULL

print(
    f"Training Mode: {'Quick' if FAST_DEMO else 'Thorough'}, "
    f"Device: {DEVICE}, "
    f"Training Samples: {len(train_df)}, "
    f"Validation Samples: {len(val_df)}"
)

# ---------------------------------------------------------------------
# Dataset + transforms
# ---------------------------------------------------------------------

class PCamDataset(Dataset):
    def __init__(self, df, root, transform):
        self.paths = [Path(root) / f"{i}.tif" for i in df.id.values]
        self.labels = torch.tensor(df.label.values, dtype=torch.float32)
        self.transform = transform

    def __len__(self): 
        return len(self.labels)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.transform(img), self.labels[i]

if FAST_DEMO:
    train_tfms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
else:
    train_tfms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.1, contrast=0.1),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])

val_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
])

# Balanced sampler + pos_weight (unchanged semantics)
labels_np = train_df["label"].values
class_counts = np.bincount(labels_np)
class_weights = 1.0 / class_counts
sample_weights = class_weights[labels_np]

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

neg, pos = class_counts
criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([neg / pos], device=DEVICE)
)

train_loader = DataLoader(
    PCamDataset(train_df, TRAIN_DIR, train_tfms),
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=NUM_WORKERS,
    pin_memory=(DEVICE.type == "cuda"),
    persistent_workers=NUM_WORKERS > 0
)
val_loader = DataLoader(
    PCamDataset(val_df, TRAIN_DIR, val_tfms),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True
)

# ---------------------------------------------------------------------
# Models (Simple + Improved + MultiScale + ResNet18)
# ---------------------------------------------------------------------

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,32,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64,128,3,padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(128,1)

    def forward(self,x):
        x = self.features(x)
        return self.fc(x.flatten(1))

WEIGHTS_ROOT = Path("/kaggle/input/resnet-18/pytorch/default")

# Find the version directory (e.g., "1", "2", etc.)
version_dirs = [p for p in WEIGHTS_ROOT.iterdir() if p.is_dir()]

assert len(version_dirs) > 0, "No ResNet version directories found"

version_dir = sorted(version_dirs)[-1]  # take latest
RESNET18_WEIGHTS = version_dir / "ResNet-18.pth"

print("Using ResNet weights at:", RESNET18_WEIGHTS)
print("Exists:", RESNET18_WEIGHTS.exists())
HAS_PRETRAINED = RESNET18_WEIGHTS.is_file()
print("HAS_PRETRAINED", HAS_PRETRAINED)

class ResNet18Binary(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        from torchvision.models import resnet18

        m = resnet18(weights=None)

        if weights_path is not None:
            print(f"Loading ResNet18 backbone from {weights_path}")
            state = torch.load(weights_path, map_location="cpu")
            state = {
                k: v for k, v in state.items()
                if not k.startswith("fc.")
            }
            missing, unexpected = m.load_state_dict(state, strict=False)
            print("Loaded backbone.")
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)

        self.features = nn.Sequential(*list(m.children())[:-1])
        self.fc = nn.Linear(512, 1)

    def forward(self, x):
        return self.fc(self.features(x).flatten(1))

class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch, stride=1):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
            )

        self.features = nn.Sequential(
            block(3, 32),
            block(32, 32, stride=2),
            block(32, 64),
            block(64, 64, stride=2),
            block(64, 128),
            block(128, 128, stride=2),
            block(128, 256),
            nn.AdaptiveAvgPool2d(1)
        )

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)

class MultiScaleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def branch(in_ch, out_ch, k):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=k//2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.SiLU(),
            )

        self.stage1 = nn.ModuleDict({
            "1x1": branch(3, 16, 1),
            "3x3": branch(3, 32, 3),
            "5x5": branch(3, 16, 5),
        })

        self.stage2 = nn.ModuleDict({
            "1x1": branch(64, 32, 1),
            "3x3": branch(64, 64, 3),
            "5x5": branch(64, 32, 5),
        })

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128, 1)

    def forward(self, x, return_activations=False):
        activations = {}

        for i, stage in enumerate([self.stage1, self.stage2], start=1):
            outs = {}
            for k, layer in stage.items():
                o = layer(x)
                outs[k] = o
                if return_activations:
                    activations[f"s{i}_{k}"] = o

            x = torch.cat(list(outs.values()), dim=1)
            x = torch.nn.functional.avg_pool2d(x, 2)

        x = self.pool(x).flatten(1)
        y = self.head(x)

        if return_activations:
            return y, activations
        return y

def build_model(mode):
    if mode == "simple":
        return SimpleCNN()

    elif mode == "improved":
        return ImprovedCNN()

    elif mode == "pretrained":
        weights = RESNET18_WEIGHTS if HAS_PRETRAINED else None
        model = ResNet18Binary(weights_path=weights)

        # Investigate the merit of freezing weights with low epoch count in the future.
        # if FAST_DEMO and weights is not None:
        #     for p in model.features.parameters():
        #         p.requires_grad = False

        return model

    elif mode == "multiscale":
        return MultiScaleCNN()

    else:
        raise ValueError(f"Unknown MODEL_MODE: {mode}")

# ---------------------------------------------------------------------
# Training helpers (very close to original behavior)
# ---------------------------------------------------------------------

def run_epoch(model, optimizer, loader, train=True):
    model.train(train)
    total, preds, trues = 0.0, [], []

    for x,y in tqdm(loader, leave=False):
        x = x.to(DEVICE, memory_format=torch.channels_last, non_blocking=True)
        y = y.to(DEVICE).unsqueeze(1)

        with autocast("cuda"):
            out = model(x)
            loss = criterion(out, y)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward(); optimizer.step()

        total += loss.item() * x.size(0)
        preds.append(torch.sigmoid(out).detach().cpu())
        trues.append(y.detach().cpu())

    preds = torch.cat(preds).numpy()
    trues = torch.cat(trues).numpy()
    auc = roc_auc_score(trues, preds)
    acc = accuracy_score(trues>0.5, preds>0.5)
    return total/len(loader.dataset), auc, acc, preds, trues

def collect_scale_stats(model, loader, max_batches=20):
    from collections import defaultdict
    model.eval()
    stats = defaultdict(list)
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            if i >= max_batches:
                break
            x = x.to(DEVICE)
            _, acts = model(x, return_activations=True)
            for k, v in acts.items():
                stats[k].append(v.abs().mean().item())
    return {k: np.mean(v) for k, v in stats.items()}

# ---------------------------------------------------------------------
# One full experiment (train + plots + saving)
# ---------------------------------------------------------------------

def run_experiment(mode: str):
    print("\n" + "="*70)
    print(f"Running model: {mode}")
    print("="*70)

    model = build_model(mode).to(DEVICE, memory_format=torch.channels_last)

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
    )
    if mode == "pretrained":
        for g in optimizer.param_groups:
            g["lr"] = LR * 0.1

    history = {"train_loss":[], "train_auc":[], "val_loss":[], "val_auc":[], "val_acc":[]}
    best_auc = 0.0
    best_state = None
    patience = 0

    for e in range(epochs):
        print(f"\nEpoch {e+1}/{epochs}")
        tl, ta, _, _, _ = run_epoch(model, optimizer, train_loader, True)
        vl, va, vac, vp, vt = run_epoch(model, optimizer, val_loader, False)

        history["train_loss"].append(tl)
        history["train_auc"].append(ta)
        history["val_loss"].append(vl)
        history["val_auc"].append(va)
        history["val_acc"].append(vac)

        print(f"train_loss={tl:.4f} train_auc={ta:.4f}")
        print(f"val_loss={vl:.4f} val_auc={va:.4f} val_acc={vac:.4f}")

        if va > best_auc:
            best_auc = va
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("Early stopping")
                break

    print("Best Validation AUC:", best_auc)

    # Save model
    model_path = MODELS_ROOT / f"pcam_{mode}_best.pth"
    torch.save(best_state, model_path)
    print(f"Saved model to {model_path}")

    # Visualizations (as before)
    epochs_range = range(1, len(history["train_loss"]) + 1)

    # Loss
    plt.figure(figsize=FIG_SINGLE)
    plt.plot(epochs_range, history["train_loss"], label="Training")
    plt.plot(epochs_range, history["val_loss"], label="Validation")
    plt.suptitle("Optimization Performance", fontsize=14, fontweight="bold", y=1)
    plt.xlabel("Epoch")
    plt.ylabel("Binary Cross-Entropy Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.figtext(
        0.5, -0.15,
        "Lower loss indicates improved optimization of model parameters.",
        ha="center", fontsize=9
    )
    save_fig(f"{mode}_loss_curve.png")

    # AUC curve
    plt.figure(figsize=FIG_SINGLE)
    plt.plot(epochs_range, history["train_auc"], label="Training AUC")
    plt.plot(epochs_range, history["val_auc"], label="Validation AUC")
    plt.suptitle("Discrimination Performance", fontsize=14, fontweight="bold", y=1)
    plt.xlabel("Epoch")
    plt.ylabel("Area Under ROC Curve (AUC)")
    plt.ylim(0.0, 1.0)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.figtext(
        0.5, -0.15,
        "AUC measures the ability to rank tumor patches above healthy patches across all decision thresholds.",
        ha="center", fontsize=9
    )
    save_fig(f"{mode}_auc_curve.png")

    # ROC & PR use last epoch's vt/vp (as before)
    y_true = vt.ravel()
    y_prob = vp.ravel()

    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {best_auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random Classifier")
    plt.suptitle("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold", y=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.figtext(
        0.5, -0.18,
        "ROC curves evaluate classifier performance independent of a fixed decision threshold.",
        ha="center", fontsize=9
    )
    save_fig(f"{mode}_roc_curve.png")

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    plt.figure(figsize=(5, 5))
    plt.plot(recall, precision, label=f"AP = {avg_precision:.4f}")
    plt.suptitle("Precision–Recall Curve", fontsize=14, fontweight="bold", y=1)
    plt.xlabel("Recall (Sensitivity)")
    plt.ylabel("Precision (Positive Predictive Value)")
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.05)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.figtext(
        0.5, -0.18,
        "Precision–Recall curves are especially informative for imbalanced medical datasets.",
        ha="center", fontsize=9
    )
    save_fig(f"{mode}_pr_curve.png")

    # Confusion matrices
    threshold_fixed = 0.5
    y_pred_fixed = (y_prob >= threshold_fixed).astype(int)
    cm_fixed = confusion_matrix(y_true, y_pred_fixed)

    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]

    y_pred_opt = (y_prob >= best_threshold).astype(int)
    cm_opt = confusion_matrix(y_true, y_pred_opt)

    print(f"Confusion Matrix (Fixed Threshold = {threshold_fixed}):\n{cm_fixed}")
    print(f"\nOptimal Threshold (Youden J): {best_threshold:.3f}")
    print("Confusion Matrix (Optimal Threshold):\n", cm_opt)

    # Multi-scale scale-usage plot (only for multiscale)
    if mode == "multiscale":
        scale_stats = collect_scale_stats(model, val_loader)
        labels_s = list(scale_stats.keys())
        values_s = list(scale_stats.values())
        plt.figure(figsize=(8, 4))
        plt.bar(labels_s, values_s)
        plt.ylabel("Mean Activation Magnitude")
        plt.title("Relative Utilization of Spatial Scales in Multi-Scale CNN")
        plt.xticks(rotation=45)
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        save_fig(f"{mode}_scale_usage.png")

    # Summary dict for comparison
    return {
        "model": mode,
        "best_val_auc": best_auc,
        "final_val_acc": history["val_acc"][-1],
        "avg_precision": avg_precision,
        "epochs_run": len(history["train_loss"])
    }

# ---------------------------------------------------------------------
# Run 1 or many models and compare
# ---------------------------------------------------------------------

results = []
modes_to_run = MODEL_GRID if RUN_ALL_MODELS else [MODEL_MODE]

for m in modes_to_run:
    res = run_experiment(m)
    results.append(res)

results_df = pd.DataFrame(results).set_index("model").sort_values("best_val_auc", ascending=False)
results_path = RESULTS_ROOT / "model_comparison.csv"
results_df.to_csv(results_path)
print("\nModel comparison:")
print(results_df)
print(f"\nSaved comparison table to: {results_path}")




DATA_ROOT  = Path("/kaggle/input/histopathologic-cancer-detection")
TEST_DIR = DATA_ROOT / "test"
# Define model loader
class PCamTestDataset(Dataset):
    def __init__(self, root, transform):
        self.paths = sorted(root.glob("*.tif"))
        self.ids = [p.stem for p in self.paths]
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        return self.ids[i], self.transform(img)

# Load best model
model = build_model(MODEL_MODE).to(DEVICE)
model.load_state_dict(
    torch.load(MODELS_ROOT / f"pcam_{MODEL_MODE}_best.pth", map_location=DEVICE)
)
model.eval()

# Inference
test_ds = PCamTestDataset(TEST_DIR, val_tfms)
test_loader = DataLoader(
    test_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS
)

pred_ids = []
pred_probs = []

with torch.no_grad():
    for ids, x in tqdm(test_loader):
        x = x.to(DEVICE)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().ravel()

        pred_ids.extend(ids)
        pred_probs.extend(probs)

submission = pd.DataFrame({
    "id": pred_ids,
    "label": pred_probs
})

submission_path = Path("/kaggle/working/submission.csv")
submission.to_csv(submission_path, index=False)

print("Saved submission to:", submission_path)



from pathlib import Path
import shutil

export_dir = Path("/kaggle/working/export")
export_dir.mkdir(exist_ok=True)

shutil.copytree(PLOTS_ROOT, export_dir / "plots")
shutil.copytree(MODELS_ROOT, export_dir / "models")
shutil.copy(RESULTS_ROOT / "model_comparison.csv", export_dir)

shutil.make_archive(
    "/kaggle/working/results_export",
    "zip",
    export_dir
)

print("Saved results_export.zip")

