'''
!pip install torch torchvision timm albumentations==1.2.1 opencv-python scikit-learn pandas
!pip install grad-cam
'''


#!/usr/bin/env python3
"""
Train script for APTOS 2019 using ConvNeXt V2 & Swin V2 backbones,
GeM pooling, SmoothL1 regression loss, warmup+cosine LR schedule, and QWK-based model checkpointing.

Usage:
    python train_aptos_convnext_swin.py --data_csv train.csv --img_dir train_images/
"""

import os
import math
import random
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler

import timm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import cohen_kappa_score
import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------------------------------
# Config / hyperparameters
# -------------------------------
DATA_CSV = "/kaggle/input/aptos2019-blindness-detection/train.csv"  # CSV with ['id_code', 'diagnosis']
IMG_DIR = "/kaggle/input/aptos2019-blindness-detection/train_images"
MODEL_NAME = "convnextv2_base.fcmae"
ALT_MODEL = "swinv2_base_window12to24_192to384"  # Optional ensemble model
INPUT_SIZE = 512
BATCH_SIZE = 8
EPOCHS = 20
FOLD = 0
N_SPLITS = 5
SEED = 42
OUTDIR = Path("outputs")
USE_AMP = True  # Use mixed precision
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-2
ACCUM_STEPS = 1  # Gradient accumulation

# === ENVIRONMENT SETUP ===
OUTDIR.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# Reproducibility
# -------------------------------
def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_everything(SEED)

# -------------------------------
# Dataset
# -------------------------------
class AptosDataset(Dataset):
    def __init__(self, df, img_dir, transforms=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transforms = transforms

    def __len__(self):
        return len(self.df)

    def load_image(self, fname):
        p = self.img_dir / fname
        img = Image.open(p).convert("RGB")
        return np.array(img)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.load_image(f"{row['id_code']}.png")
        if self.transforms:
            data = self.transforms(image=img)
            img = data["image"]
        target = torch.tensor(row["diagnosis"], dtype=torch.float32)  # regression target (0..4)
        return img, target


def get_transforms(input_size=INPUT_SIZE):
    train_transforms = A.Compose([
        A.RandomResizedCrop(size=(input_size, input_size), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(),
        A.VerticalFlip(p=0.1),
        A.Affine(
            scale=(0.8, 1.2),
            translate_percent=(0.1, 0.1),
            rotate=(-180, 180),
            shear=(-10, 10),
            p=0.5
        ),
        A.RandomBrightnessContrast(0.2, 0.2),
        A.HueSaturationValue(10, 20, 20),
        A.GaussNoise(gauss_noise_var_limit=(10.0, 50.0), mean=0, p=0.2),
        A.MotionBlur(p=0.2),
        A.CoarseDropout(
            max_holes=8,
            max_height=int(input_size * 0.08),
            max_width=int(input_size * 0.08),
            min_holes=1,
            mask_fill_value=0,
            p=0.3
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    valid_transforms = A.Compose([
        A.Resize(height=input_size, width=input_size),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transforms, valid_transforms

# -------------------------------
# GeM Pooling
# -------------------------------
class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, C, H, W)
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), (x.size(-2), x.size(-1))).pow(1.0 / self.p)

    def __repr__(self):
        return f"GeM(p={self.p.data.tolist()[0]:.4f}, eps={self.eps})"

# -------------------------------
# Model wrappers
# -------------------------------
class RegressionModel(nn.Module):
    def __init__(self, backbone_name, pretrained=True, out_features=1):
        super().__init__()
        # create backbone with num_classes=0 to get features
        self.backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0, global_pool="")
        feature_dim = self.backbone.num_features
        self.pool = GeM()
        self.fc = nn.Linear(feature_dim, out_features)

    def forward(self, x):
        # timm backbone expects NCHW; returns features (B, C, H, W)
        feat = self.backbone.forward_features(x)  # some timm models have forward_features
        pooled = self.pool(feat).flatten(1)
        out = self.fc(pooled).squeeze(1)
        return out

# -------------------------------
# Loss: Smooth L1 (Huber-like)
# -------------------------------
def get_loss():
    # We keep Smooth L1 (L1 with beta). Could combine with other losses as needed.
    return nn.SmoothL1Loss()

# -------------------------------
# LR Scheduler: Warmup + CosineAnnealing
# -------------------------------
def get_scheduler(optimizer, num_warmup_epochs, max_epochs, last_epoch=-1):
    # We'll implement linear warmup via LambdaLR followed by CosineAnnealingLR
    def lr_lambda(current_epoch):
        if current_epoch < num_warmup_epochs:
            return float(current_epoch) / float(max(1, num_warmup_epochs))
        else:
            # cosine decay for remaining epochs
            progress = float(current_epoch - num_warmup_epochs) / float(max(1, max_epochs - num_warmup_epochs))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

# -------------------------------
# Metrics
# -------------------------------
def qwk(true, pred):
    # Convert predictions to nearest class via thresholds (default)
    # Here we expect true and pred to be 1D numpy arrays
    # Default thresholds (can be optimized later)
    thr = [0.5, 1.5, 2.5, 3.5]
    pred_bin = np.digitize(pred, thr)
    return cohen_kappa_score(true, pred_bin, weights='quadratic')

def optimize_thresholds(y_true, y_pred, search_space=None):
    # simple grid search on first threshold (like write-up)
    if search_space is None:
        search_space = np.linspace(0.3, 0.9, 13)
    best_thr = [0.5, 1.5, 2.5, 3.5]
    best_k = -1
    for t0 in search_space:
        thr = [t0, 1.5, 2.5, 3.5]
        pred_bin = np.digitize(y_pred, thr)
        k = cohen_kappa_score(y_true, pred_bin, weights='quadratic')
        if k > best_k:
            best_k = k
            best_thr = thr
    return best_thr, best_k

# -------------------------------
# Training & validation loops
# -------------------------------
def train_one_epoch(model, optimizer, loader, device, scaler, loss_fn, epoch, accum_steps=1):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    for step, (imgs, targets) in enumerate(loader):
        imgs = imgs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with autocast('cuda', enabled=USE_AMP):
            outputs = model(imgs)
            loss = loss_fn(outputs, targets) / accum_steps

        if USE_AMP:
            scaler.scale(loss).backward()
            if (step + 1) % accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            loss.backward()
            if (step + 1) % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        running_loss += loss.item() * accum_steps

    return running_loss / len(loader)

@torch.no_grad()
def validate_one_epoch(model, loader, device):
    model.eval()
    preds = []
    gts = []
    for imgs, targets in loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        preds.append(outputs.detach().cpu().numpy())
        gts.append(targets.numpy())
    preds = np.concatenate(preds)
    gts = np.concatenate(gts)
    # compute QWK using default thresholds
    kappa = qwk(gts, preds)
    return kappa, gts, preds


def run_training(df, img_dir, fold=FOLD, n_splits=N_SPLITS, seed=SEED):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = list(skf.split(df, df["diagnosis"]))
    train_idx, valid_idx = splits[fold]
    train_df = df.iloc[train_idx].reset_index(drop=True)
    valid_df = df.iloc[valid_idx].reset_index(drop=True)

    train_tf, valid_tf = get_transforms(INPUT_SIZE)
    train_ds = AptosDataset(train_df, img_dir, transforms=train_tf)
    valid_ds = AptosDataset(valid_df, img_dir, transforms=valid_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE*2, shuffle=False, num_workers=4, pin_memory=True)

    # two models for ensemble option (train sequentially)
    backbones = [MODEL_NAME, ALT_MODEL]
    best_info = {}

    for backbone_name in backbones:
        model = RegressionModel(backbone_name, pretrained=True, out_features=1)
        model = model.to(DEVICE)

        # small trick: freeze backbone for first epoch if desired (optional)
        # for name, p in model.backbone.named_parameters():
        #     p.requires_grad = False

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        scheduler = get_scheduler(optimizer, num_warmup_epochs=2, max_epochs=EPOCHS)
        loss_fn = get_loss()
        scaler_local = GradScaler('cuda') if USE_AMP else None

        best_kappa = -999
        best_epoch = -1
        best_model_path = OUTDIR / f"best_{Path(backbone_name).name}_fold{fold}.pt"
        
        print(f"\n=== Training backbone: {backbone_name} ===")
        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, optimizer, train_loader, DEVICE, scaler_local, loss_fn, epoch, accum_steps=ACCUM_STEPS)
            scheduler.step(epoch)  # calls the lambda
            val_kappa, y_val, y_pred = validate_one_epoch(model, valid_loader, DEVICE)
            print(f"[{backbone_name}] Epoch {epoch+1}/{EPOCHS} | TrainLoss {train_loss:.4f} | ValQWK {val_kappa:.5f} | LR {optimizer.param_groups[0]['lr']:.2e}")

            # save best by QWK
            if val_kappa > best_kappa:
                best_kappa = val_kappa
                best_epoch = epoch
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scaler_state": scaler_local.state_dict() if scaler_local else None,
                    "val_kappa": val_kappa,
                    "epoch": epoch,
                    "backbone": backbone_name
                }, best_model_path)
                # also save the raw predictions for threshold tuning
                np.save(OUTDIR / f"val_pred_{Path(backbone_name).name}_fold{fold}.npy", y_pred)
                np.save(OUTDIR / f"val_true_{Path(backbone_name).name}_fold{fold}.npy", y_val)

        print(f"Best val QWK for {backbone_name}: {best_kappa:.5f} at epoch {best_epoch}")
        best_info[backbone_name] = {"best_kappa": best_kappa, "best_epoch": best_epoch, "model_path": str(best_model_path)}

    # After training both backbones, we can ensemble their saved predictions on the validation set
    # and find optimized thresholds.
    preds_all = []
    truths = None
    for backbone_name in backbones:
        pred_path = OUTDIR / f"val_pred_{Path(backbone_name).name}_fold{fold}.npy"
        true_path = OUTDIR / f"val_true_{Path(backbone_name).name}_fold{fold}.npy"
        if pred_path.exists():
            p = np.load(pred_path)
            preds_all.append(p)
            if truths is None:
                truths = np.load(true_path)
    if len(preds_all) > 0:
        ensemble_pred = np.mean(np.vstack(preds_all), axis=0)
        best_thr, best_k = optimize_thresholds(truths, ensemble_pred)
        print(f"Ensemble validation QWK after threshold tuning: {best_k:.5f} with thresholds {best_thr}")

    return best_info


print("CUDA available:", torch.cuda.is_available(), " Device:", DEVICE)
df = pd.read_csv(DATA_CSV)
# Ensure file ids are strings (APTOS filenames)
df['id_code'] = df['id_code'].astype(str)
# If diagnosis not numeric or has stray values, coerce
df['diagnosis'] = pd.to_numeric(df['diagnosis'], errors='coerce').fillna(0).astype(int)
print("Loaded CSV:", df.shape)

# Run training for the requested fold
info = run_training(df, IMG_DIR, fold=FOLD, n_splits=N_SPLITS, seed=SEED)
print("Training finished. Models info:", info)


from sklearn.metrics import cohen_kappa_score, f1_score, roc_auc_score, confusion_matrix, classification_report

val_qwk = cohen_kappa_score(y_true, y_pred, weights='quadratic')
f1 = f1_score(y_true, y_pred, average='macro')
print(confusion_matrix(y_true, y_pred))
print(classification_report(y_true, y_pred))

