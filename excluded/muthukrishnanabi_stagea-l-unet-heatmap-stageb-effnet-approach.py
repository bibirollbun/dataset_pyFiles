# import pydicom
# import json

# d = pydicom.dcmread("/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647/1.2.826.0.1.3680043.8.498.10124807242473374136099471315028464450.dcm")
# print(d)



# import pandas as pd

# # Replace 'your_file.csv' with the actual path to your CSV file
# file_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv' 
# df = pd.read_csv(file_path)

# # Display the first few rows of the DataFrame
# print("train_localizers.CSV Data (first 5 rows):")
# print(df.head())
# print("train_localizers.CSV Data describe()")
# print(df.describe())
# print("train_localizers.CSV Data Info()")
# print(df.info())


# Cell 0: Install dependencies (safe versions). Kaggle already has many packages;
# we force-install monai and ensure protobuf and tqdm are present.
# This is idempotent and quiet to avoid noisy logs.

import sys
import os

# Only run pip installs if packages missing to avoid slow installs on Kaggle
def pip_install_if_missing(pkg, import_name=None, version=None):
    try:
        __import__(import_name or pkg)
    except Exception:
        pkg_str = pkg + (("==" + version) if version else "")
        print(f"Installing {pkg_str} ...")
        os.system(f"{sys.executable} -m pip install --no-warn-script-location --quiet {pkg_str}")

# Required libs
pip_install_if_missing("tqdm", "tqdm")
pip_install_if_missing("monai", "monai")   # adjust version if needed
pip_install_if_missing("timm", "timm")
pip_install_if_missing("nibabel", "nibabel")
pip_install_if_missing("pydicom", "pydicom")
pip_install_if_missing("scikit-learn", "sklearn")
pip_install_if_missing("opencv-python-headless", "cv2")
pip_install_if_missing("protobuf", "google.protobuf")

!pip install -q "protobuf"
print("All required packages should be present.")


# ============================================================
# Cell 1 — Imports + AMP + Global Seed
# ============================================================

import os, sys, math, time, gc, random, json
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib
import pydicom
import cv2

from scipy.ndimage import zoom as ndi_zoom
from scipy.ndimage import label as cc_label

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# New recommended AMP API
from torch import amp

import timm
from tqdm.auto import tqdm
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

# reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

print("PyTorch Version:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())



# ============================================================
# Cell 2 — Global Config (Matches Your Hardware Precisely)
# ============================================================

class CFG:
    # paths
    DATA_ROOT = "/kaggle/input/rsna-intracranial-aneurysm-detection"
    FILTERED_ROOT = "/kaggle/input/rsna-filtered-set"
    SERIES_DIR = "series"
    SEG_DIR = "segmentations"

    OUT_DIR = "/kaggle/working/RSNA_FULL_PIPELINE"
    os.makedirs(OUT_DIR, exist_ok=True)

    # hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers = 0               # Kaggle safe
    mixed_precision = True        # use AMP
    batch_size = 1                # MUST for 3D
    grad_accum = 8
    max_epochs_a = 8             # Segmentation stage
    max_epochs_b = 5             # Classifier stage
    save_every = 3

    # preprocessing (Stage A)
    target_spacing = 0.8
    a_patch_shape = (80, 96, 96)
    a_clip = (-300, 1000)
    base_ch = 24
    lr_a = 2e-4

    # candidate extraction
    cand_prob_thr = 0.20
    cand_min_dist_mm = 6.0
    max_cands_per_series = 150

    # Stage B (2.5D classifier)
    b_num_slices = 9
    b_patch_size = 160
    b_lr = 1e-4
    b_emb_dim = 512

cfg = CFG()

print(cfg.__dict__)



# ============================================================
# Cell 3 — Load CSV Metadata
# ============================================================

train_csv = os.path.join(cfg.DATA_ROOT, "train.csv")
train_masked_csv = os.path.join(cfg.FILTERED_ROOT, "train_masked.csv")
train_localizers_csv = os.path.join(cfg.DATA_ROOT, "train_localizers.csv")

train_df = pd.read_csv(train_csv)
train_masked_df = pd.read_csv(train_masked_csv)
localizers_df = pd.read_csv(train_localizers_csv)

print("train_df:", train_df.shape)
print("train_masked_df:", train_masked_df.shape)
print("localizers_df:", localizers_df.shape)

# Only keep UIDs that have cowseg available
def cowseg_exists(uid):
    p = Path(cfg.DATA_ROOT) / cfg.SEG_DIR / f"{uid}_cowseg.nii"
    p2 = Path(cfg.FILTERED_ROOT) / "segmentations" / f"{uid}_cowseg.nii"
    return p.exists() or p2.exists()

train_masked_df["has_cowseg"] = train_masked_df["SeriesInstanceUID"].apply(cowseg_exists)
df_a = train_masked_df[train_masked_df["has_cowseg"]].reset_index(drop=True)
df_a = df_a[0:60] #comment this for full dataset
print("Stage A usable:", df_a.shape)



# ============================================================
# Cell 4 — DICOM + NIfTI Loader (No warnings, stable)
# ============================================================

def load_dicom_series(series_uid):
    """Load DICOM series → (volume[z,y,x], spacing[z,y,x])."""
    series_path = Path(cfg.DATA_ROOT) / cfg.SERIES_DIR / series_uid
    files = sorted(list(series_path.glob("*.dcm")))
    if len(files) == 0:
        raise FileNotFoundError(f"No DICOM found for {series_uid}")

    # sort by InstanceNumber robustly
    inst_list = []
    for f in files:
        try:
            d = pydicom.dcmread(str(f), stop_before_pixels=True)
            inst = int(getattr(d, "InstanceNumber", 0))
        except Exception:
            inst = 0
        inst_list.append((inst, f))

    inst_list = sorted(inst_list, key=lambda x: x[0])
    slices = [pydicom.dcmread(str(f)) for _, f in inst_list]

    vol = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)

    intercept = float(getattr(slices[0], "RescaleIntercept", 0))
    slope = float(getattr(slices[0], "RescaleSlope", 1))
    vol = vol * slope + intercept

    px = getattr(slices[0], "PixelSpacing", [1.0, 1.0])
    px = np.array(px, dtype=np.float32)

    # spacing in z
    try:
        z0 = slices[0].ImagePositionPatient[2]
        z1 = slices[1].ImagePositionPatient[2]
        sz = abs(z1 - z0)
    except Exception:
        sz = float(getattr(slices[0], "SliceThickness", 1.0))

    spacing = np.array([sz, px[0], px[1]], dtype=np.float32)

    # (z,y,x), spacing[z,y,x]
    return vol, spacing


def load_nifti(path):
    """Load NIfTI → array(z,y,x), spacing[z,y,x]."""
    nii = nib.load(path)
    arr = nii.get_fdata().astype(np.float32)
    arr = np.transpose(arr, (2,1,0))  # to (z,y,x)
    pixdim = nii.header.get_zooms()
    spacing = np.array([pixdim[2], pixdim[1], pixdim[0]], dtype=np.float32)
    return arr, spacing



# ============================================================
# Cell 5 — Preprocessing (Window, Resample, Crop/PAD)
# ============================================================

def hu_window(vol, min_hu, max_hu):
    vol = np.clip(vol, min_hu, max_hu)
    return ((vol - min_hu) / (max_hu - min_hu)).astype(np.float32)


def resample_to_spacing(vol, spacing, target_spacing, order=1):
    factors = spacing / target_spacing
    factors = np.clip(factors, 0.1, 10.0)
    res = ndi_zoom(vol, zoom=factors, order=order, mode="nearest")
    return res, np.array([target_spacing]*3, dtype=np.float32)


def center_crop_or_pad(vol, shape):
    """Produce exact shape (Z,Y,X)."""
    Z, Y, X = shape
    out = np.zeros(shape, dtype=vol.dtype)

    z0 = max((Z - vol.shape[0])//2, 0)
    y0 = max((Y - vol.shape[1])//2, 0)
    x0 = max((X - vol.shape[2])//2, 0)

    z1 = min(z0 + vol.shape[0], Z)
    y1 = min(y0 + vol.shape[1], Y)
    x1 = min(x0 + vol.shape[2], X)

    vz0 = max(0, -(Z - vol.shape[0])//2)
    vy0 = max(0, -(Y - vol.shape[1])//2)
    vx0 = max(0, -(X - vol.shape[2])//2)

    out[z0:z1, y0:y1, x0:x1] = vol[
        vz0:(vz0 + (z1-z0)),
        vy0:(vy0 + (y1-y0)),
        vx0:(vx0 + (x1-x0))
    ]

    return out


def preprocess_volume_for_stage_a(vol, spacing):
    vol = hu_window(vol, cfg.a_clip[0], cfg.a_clip[1])
    res, _ = resample_to_spacing(vol, spacing, cfg.target_spacing, order=1)
    res = center_crop_or_pad(res, cfg.a_patch_shape)
    return res, cfg.target_spacing



# ============================================================
# Cell 6 — cowseg Mask Loader + Multi-channel GT creator
# ============================================================

def find_cowseg(uid):
    p1 = Path(cfg.DATA_ROOT) / cfg.SEG_DIR / f"{uid}_cowseg.nii"
    p2 = Path(cfg.FILTERED_ROOT) / "segmentations" / f"{uid}_cowseg.nii"
    if p1.exists(): return str(p1)
    if p2.exists(): return str(p2)
    return None


def load_clean_mask(uid):
    """Load _cowseg.nii and sanitize to 0..13."""
    path = find_cowseg(uid)
    if path is None:
        return None, None
    arr, sp = load_nifti(path)
    arr = np.nan_to_num(arr).astype(np.int32)
    arr[arr < 0] = 0
    arr[arr > 13] = 0
    return arr, sp


def build_multichannel_mask(mask, spacing):
    """Convert mask → (14,Z,Y,X) one-hot."""
    if mask is None:
        return np.zeros((14,) + cfg.a_patch_shape, dtype=np.uint8)

    res, _ = resample_to_spacing(mask.astype(np.float32),
                                 spacing,
                                 cfg.target_spacing,
                                 order=0)
    res = center_crop_or_pad(res.astype(np.int32), cfg.a_patch_shape)

    out = np.zeros((14,) + cfg.a_patch_shape, dtype=np.uint8)
    for lbl in range(1,14):
        out[lbl-1] = (res == lbl).astype(np.uint8)
    return out



# ============================================================
# Cell 7 — Stage A Dataset + Loader
# ============================================================

class StageADataset(Dataset):
    """Loads DICOM + cowseg mask → normalized 3D tensor + 14-channel GT."""
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["SeriesInstanceUID"]

        try:
            vol, sp = load_dicom_series(uid)
        except Exception:
            vol = np.zeros(cfg.a_patch_shape, dtype=np.float32)
            sp = np.array([cfg.target_spacing]*3)

        vol_prep, _ = preprocess_volume_for_stage_a(vol, sp)

        mask, msp = load_clean_mask(uid)
        gt = build_multichannel_mask(mask, msp if msp is not None else sp)

        x = torch.from_numpy(vol_prep[None]).float()     # (1,Z,Y,X)
        y = torch.from_numpy(gt).float()                 # (14,Z,Y,X)

        return x, y, uid


# 5-fold split (like earlier)
gkf = GroupKFold(n_splits=5)
fold = 0
indices = list(gkf.split(df_a, groups=df_a["SeriesInstanceUID"]))
train_idx, val_idx = indices[fold]

train_df_a = df_a.iloc[train_idx].reset_index(drop=True)
val_df_a   = df_a.iloc[val_idx].reset_index(drop=True)

train_ds_a = StageADataset(train_df_a)
val_ds_a   = StageADataset(val_df_a)

train_loader_a = DataLoader(
    train_ds_a,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

val_loader_a = DataLoader(
    val_ds_a,
    batch_size=1,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
)

print("Stage A train:", len(train_ds_a), "val:", len(val_ds_a))



# # ============================================================
# # Cell 8 — LightUNet3D (VRAM-safe, stable for T4/P100)
# # ============================================================

# class DoubleConv3D(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super().__init__()
#         self.block = nn.Sequential(
#             nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm3d(out_ch),
#             nn.LeakyReLU(0.1, inplace=True),

#             nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
#             nn.InstanceNorm3d(out_ch),
#             nn.LeakyReLU(0.1, inplace=True),
#         )

#     def forward(self, x):
#         return self.block(x)


# class LightUNet3D(nn.Module):
#     """Lightweight 3D UNet suitable for 13GB VRAM."""
#     def __init__(self, in_ch=1, out_ch=14, base_ch=16):
#         super().__init__()
#         b = base_ch

#         # Encoder
#         self.enc1 = DoubleConv3D(in_ch, b)
#         self.pool1 = nn.MaxPool3d(2)

#         self.enc2 = DoubleConv3D(b, b*2)
#         self.pool2 = nn.MaxPool3d(2)

#         self.enc3 = DoubleConv3D(b*2, b*4)
#         self.pool3 = nn.MaxPool3d(2)

#         self.enc4 = DoubleConv3D(b*4, b*8)
#         self.pool4 = nn.MaxPool3d(2)

#         # Bottleneck
#         self.bottleneck = DoubleConv3D(b*8, b*16)

#         # Decoder
#         self.up4 = nn.ConvTranspose3d(b*16, b*8, kernel_size=2, stride=2)
#         self.dec4 = DoubleConv3D(b*16, b*8)

#         self.up3 = nn.ConvTranspose3d(b*8, b*4, kernel_size=2, stride=2)
#         self.dec3 = DoubleConv3D(b*8, b*4)

#         self.up2 = nn.ConvTranspose3d(b*4, b*2, kernel_size=2, stride=2)
#         self.dec2 = DoubleConv3D(b*4, b*2)

#         self.up1 = nn.ConvTranspose3d(b*2, b, kernel_size=2, stride=2)
#         self.dec1 = DoubleConv3D(b*2, b)

#         self.out_conv = nn.Conv3d(b, out_ch, kernel_size=1)

#     def forward(self, x):
#         c1 = self.enc1(x)
#         p1 = self.pool1(c1)

#         c2 = self.enc2(p1)
#         p2 = self.pool2(c2)

#         c3 = self.enc3(p2)
#         p3 = self.pool3(c3)

#         c4 = self.enc4(p3)
#         p4 = self.pool4(c4)

#         bn = self.bottleneck(p4)

#         u4 = self.up4(bn)
#         d4 = self.dec4(torch.cat([u4, c4], dim=1))

#         u3 = self.up3(d4)
#         d3 = self.dec3(torch.cat([u3, c3], dim=1))

#         u2 = self.up2(d3)
#         d2 = self.dec2(torch.cat([u2, c2], dim=1))

#         u1 = self.up1(d2)
#         d1 = self.dec1(torch.cat([u1, c1], dim=1))

#         return self.out_conv(d1)



import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# DoubleConv3D (keeps same style)
# ---------------------------
class DoubleConv3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),

            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_ch),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, x):
        return self.block(x)

# ---------------------------
# Attention Gate (3D)
# ---------------------------
class AttentionGate3D(nn.Module):
    """
    Attention gate that takes decoder gating signal g and encoder features x,
    produces attention coefficients to scale x.
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        # 1x1 conv to reduce channels
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.InstanceNorm3d(1),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, g, x):
        """
        g: gating signal (from decoder), shape [B, F_g, ...]
        x: encoder feature map to be modulated, shape [B, F_l, ...]
        returns: x * attention_map (same shape as x)
        """
        # Reduce and add
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # [B,1,D,H,W] with values in (0,1)
        return x * psi  # broadcast multiply

# ---------------------------
# LightUNet3D (Attention U-Net implementation, same API)
# ---------------------------
class LightUNet3D(nn.Module):
    """Attention U-Net in the same footprint as previous LightUNet3D.
       Signature kept identical: LightUNet3D(in_ch=1, out_ch=14, base_ch=16)
    """
    def __init__(self, in_ch=1, out_ch=14, base_ch=16):
        super().__init__()
        b = base_ch

        # Encoder (same layout as your previous LightUNet3D)
        self.enc1 = DoubleConv3D(in_ch, b)
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = DoubleConv3D(b, b*2)
        self.pool2 = nn.MaxPool3d(2)

        self.enc3 = DoubleConv3D(b*2, b*4)
        self.pool3 = nn.MaxPool3d(2)

        self.enc4 = DoubleConv3D(b*4, b*8)
        self.pool4 = nn.MaxPool3d(2)

        # Bottleneck
        self.bottleneck = DoubleConv3D(b*8, b*16)

        # Attention gates (match channels of encoder + decoder)
        # F_g = gating channels (decoder), F_l = encoder feature channels
        self.att4 = AttentionGate3D(F_g=b*8, F_l=b*8, F_int=b*4)
        self.att3 = AttentionGate3D(F_g=b*4, F_l=b*4, F_int=b*2)
        self.att2 = AttentionGate3D(F_g=b*2, F_l=b*2, F_int=b)
        self.att1 = AttentionGate3D(F_g=b,   F_l=b,   F_int=max(b//2, 1))

        # Decoder (ConvTranspose3d + DoubleConv)
        self.up4 = nn.ConvTranspose3d(b*16, b*8, kernel_size=2, stride=2)
        self.dec4 = DoubleConv3D(b*16, b*8)

        self.up3 = nn.ConvTranspose3d(b*8, b*4, kernel_size=2, stride=2)
        self.dec3 = DoubleConv3D(b*8, b*4)

        self.up2 = nn.ConvTranspose3d(b*4, b*2, kernel_size=2, stride=2)
        self.dec2 = DoubleConv3D(b*4, b*2)

        self.up1 = nn.ConvTranspose3d(b*2, b, kernel_size=2, stride=2)
        self.dec1 = DoubleConv3D(b*2, b)

        # Output conv (same out_ch as before)
        self.out_conv = nn.Conv3d(b, out_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        c1 = self.enc1(x)     # [B, b, ...]
        p1 = self.pool1(c1)

        c2 = self.enc2(p1)    # [B, 2b, ...]
        p2 = self.pool2(c2)

        c3 = self.enc3(p2)    # [B, 4b, ...]
        p3 = self.pool3(c3)

        c4 = self.enc4(p3)    # [B, 8b, ...]
        p4 = self.pool4(c4)

        bn = self.bottleneck(p4)  # [B, 16b, ...]

        # Decoder + attention-modulated skips
        u4 = self.up4(bn)             # -> [B, 8b, ...]
        a4 = self.att4(g=u4, x=c4)    # apply attention on encoder features
        d4 = self.dec4(torch.cat([u4, a4], dim=1))

        u3 = self.up3(d4)
        a3 = self.att3(g=u3, x=c3)
        d3 = self.dec3(torch.cat([u3, a3], dim=1))

        u2 = self.up2(d3)
        a2 = self.att2(g=u2, x=c2)
        d2 = self.dec2(torch.cat([u2, a2], dim=1))

        u1 = self.up1(d2)
        a1 = self.att1(g=u1, x=c1)
        d1 = self.dec1(torch.cat([u1, a1], dim=1))

        return self.out_conv(d1)



# ============================================================
# Cell 9 — DiceLoss + FocalBCE + CombinedLoss
# ============================================================

# class DiceLoss(nn.Module):
#     def __init__(self, smooth=1e-6):
#         super().__init__()
#         self.smooth = smooth

#     def forward(self, logits, targets):
#         probs = torch.sigmoid(logits)
#         probs = probs.reshape(probs.shape[0], probs.shape[1], -1)
#         targets = targets.reshape(targets.shape[0], targets.shape[1], -1)

#         intersection = (probs * targets).sum(-1)
#         denom = probs.sum(-1) + targets.sum(-1)

#         dice = (2*intersection + self.smooth) / (denom + self.smooth)
#         return 1 - dice.mean()


# class FocalBCE(nn.Module):
#     """Good for heavy label imbalance."""
#     def __init__(self, gamma=2.0, alpha=0.25):
#         super().__init__()
#         self.gamma = gamma
#         self.alpha = alpha

#     def forward(self, logits, targets):
#         bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
#         pt = torch.exp(-bce)
#         focal = self.alpha * ((1-pt)**self.gamma) * bce
#         return focal.mean()


# class HeatmapLoss(nn.Module):
#     """Dice + Focal = stable training for micro-aneurysm-like lesions."""
#     def __init__(self):
#         super().__init__()
#         self.dice = DiceLoss()
#         self.focal = FocalBCE()

#     def forward(self, logits, targets):
#         return 0.7 * self.dice(logits, targets) + 0.3 * self.focal(logits, targets)



# ============================================================
# Cell 9 — DiceLoss + FocalBCE + HeatmapLoss (Focal-Tversky + FocalBCE)
# ============================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """Classic soft-Dice (kept for compatibility)."""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: tensor [B, C, ...] (raw logits)
        targets: tensor [B, C, ...] (binary or one-hot for each channel)
        """
        probs = torch.sigmoid(logits)
        # flatten per sample per channel
        probs = probs.reshape(probs.shape[0], probs.shape[1], -1)
        targets = targets.reshape(targets.shape[0], targets.shape[1], -1)

        intersection = (probs * targets).sum(-1)
        denom = probs.sum(-1) + targets.sum(-1)

        dice = (2 * intersection + self.smooth) / (denom + self.smooth)
        # dice shape: [B, C] -> mean over channels then over batch
        return 1.0 - dice.mean()


class FocalBCE(nn.Module):
    """Focal variant of BCE; same API as before."""
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        """
        logits: [B, C, ...]
        targets: [B, C, ...]
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')  # same shape as logits
        pt = torch.exp(-bce)  # pt = sigmoid(logit) when target=1, (1-sigmoid) when target=0 analog
        focal = self.alpha * ((1 - pt) ** self.gamma) * bce
        return focal.mean()


# --- Utility: Tversky / Focal-Tversky (multi-channel aware) ---
def tversky_index_from_logits(logits, targets, alpha=0.7, beta=0.3, eps=1e-6):
    """
    Compute Tversky index per-sample per-channel from logits.
    logits: [B, C, ...]
    targets: [B, C, ...] (binary)
    returns: tensor [B, C] of Tversky indices
    """
    prob = torch.sigmoid(logits)
    # flatten spatial dims
    prob = prob.reshape(prob.shape[0], prob.shape[1], -1)
    targ = targets.reshape(targets.shape[0], targets.shape[1], -1)

    TP = (prob * targ).sum(-1)
    FP = (prob * (1 - targ)).sum(-1)
    FN = ((1 - prob) * targ).sum(-1)

    tversky = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    return tversky  # [B, C]


def focal_tversky_loss(logits, targets, alpha=0.7, beta=0.3, gamma=0.75):
    """
    Focal-Tversky loss averaged over channels and batch.
    """
    tversky = tversky_index_from_logits(logits, targets, alpha=alpha, beta=beta)
    # tversky in [0,1]; loss = (1 - T)^gamma
    loss = (1 - tversky) ** gamma
    return loss.mean()


class HeatmapLoss(nn.Module):
    """
    Combined loss for heatmap training.
    Uses Focal-Tversky (for tiny lesion sensitivity) + Focal BCE (for stability).
    We keep the same outward behavior as previous HeatmapLoss (single forward(logits, targets)).
    """
    def __init__(self, wt_tversky=0.7, wt_focalbce=0.3, 
                 tversky_alpha=0.7, tversky_beta=0.3, tversky_gamma=0.75,
                 focal_gamma=2.0, focal_alpha=0.25):
        super().__init__()
        self.wt_tversky = wt_tversky
        self.wt_focalbce = wt_focalbce
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.tversky_gamma = tversky_gamma
        self.focal = FocalBCE(gamma=focal_gamma, alpha=focal_alpha)

    def forward(self, logits, targets):
        """
        logits: [B, C, ...], targets: [B, C, ...] (binary)
        returns: scalar loss
        """
        # ensure float tensors
        logits = logits.float()
        targets = targets.float()

        ft = focal_tversky_loss(logits, targets, alpha=self.tversky_alpha, beta=self.tversky_beta, gamma=self.tversky_gamma)
        fb = self.focal(logits, targets)
        loss = self.wt_tversky * ft + self.wt_focalbce * fb
        return loss



# ============================================================
# Cell 10 — Stage A Training Loop (AMP-safe, tqdm, stable)
# ============================================================

def train_stage_a():
    device = cfg.device
    device_type = cfg.device_type

    model = LightUNet3D(
        in_ch=1,
        out_ch=14,
        base_ch=cfg.base_ch
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_a)
    criterion = HeatmapLoss()

    scaler = amp.GradScaler(enabled=cfg.mixed_precision)
    best_val = 1e9

    history = {"train": [], "val": []}

    for epoch in range(cfg.max_epochs_a):
        # ---- Training ----
        model.train()
        train_loss = 0.0

        pbar = tqdm(train_loader_a, desc=f"Epoch {epoch+1}/{cfg.max_epochs_a} [Train]", leave=False)

        optimizer.zero_grad()

        for step, (x, y, uid) in enumerate(pbar):
            x = x.to(device)
            y = y.to(device)

            with amp.autocast(device_type=device_type, enabled=cfg.mixed_precision):
                logits = model(x)
                loss = criterion(logits, y) / cfg.grad_accum

            scaler.scale(loss).backward()

            if (step+1) % cfg.grad_accum == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * cfg.grad_accum
            pbar.set_postfix(loss=train_loss / (step+1))

        train_loss /= len(train_loader_a)
        history["train"].append(train_loss)

        # ---- Validation ----
        model.eval()
        val_loss = 0.0

        pbar = tqdm(val_loader_a, desc=f"Epoch {epoch+1}/{cfg.max_epochs_a} [Val]", leave=False)

        with torch.no_grad():
            for x, y, uid in pbar:
                x = x.to(device)
                y = y.to(device)

                with amp.autocast(device_type=device_type, enabled=cfg.mixed_precision):
                    logits = model(x)
                    loss = criterion(logits, y)

                val_loss += loss.item()
                pbar.set_postfix(loss=val_loss / (len(val_loader_a)))

        val_loss /= len(val_loader_a)
        history["val"].append(val_loss)

        print(f"[Epoch {epoch+1}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Save best checkpoint
        best_path = os.path.join(cfg.OUT_DIR, "stageA_best.pth")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), best_path)
            print("   → Saved BEST model")

        # Periodic checkpoint
        if (epoch+1) % cfg.save_every == 0:
            ckpt_path = os.path.join(cfg.OUT_DIR, f"stageA_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), ckpt_path)
            print("   → Saved checkpoint", ckpt_path)

    return model, history



# ============================================================
# Cell 11 — Stage A Inference + Candidate Extraction
# ============================================================

def extract_candidates_from_heatmap(hm, spacing, prob_thr=cfg.cand_prob_thr,
                                    min_dist_mm=cfg.cand_min_dist_mm, max_cands=cfg.max_cands_per_series):
    """
    hm: 3D numpy array (Z,Y,X) with values 0..1
    spacing: scalar or (z,y,x) mm spacing (here we use cfg.target_spacing)
    """
    mask = hm > prob_thr
    if mask.sum() == 0:
        return []
    labeled, num = cc_label(mask)
    candidates = []
    for lab in range(1, num+1):
        comp = (labeled == lab)
        if comp.sum() == 0:
            continue
        # centroid by max probability
        idx = np.unravel_index(np.argmax(hm * comp), hm.shape)
        prob = float(hm[idx])
        candidates.append((int(idx[0]), int(idx[1]), int(idx[2]), prob))
    # sort by prob desc
    candidates = sorted(candidates, key=lambda x: x[3], reverse=True)
    # NMS by physical distance
    selected = []
    for cz, cy, cx, p in candidates:
        keep = True
        for s in selected:
            dz = (cz - s[0]) * cfg.target_spacing
            dy = (cy - s[1]) * cfg.target_spacing
            dx = (cx - s[2]) * cfg.target_spacing
            if math.sqrt(dx*dx + dy*dy + dz*dz) < min_dist_mm:
                keep = False; break
        if keep:
            selected.append((cz, cy, cx, p))
        if len(selected) >= max_cands:
            break
    return selected


def stage_a_inference_and_save(model_path, meta_df=df_a, out_candidates=os.path.join(cfg.OUT_DIR, "stageA_candidates.csv")):
    device = cfg.device
    device_type = cfg.device_type
    model = LightUNet3D(in_ch=1, out_ch=14, base_ch=cfg.base_ch).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    cand_rows = []
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Stage A inference"):
        uid = row["SeriesInstanceUID"]
        try:
            vol, spacing = load_dicom_series(uid)
        except Exception:
            continue
        vol_prep, _ = preprocess_volume_for_stage_a(vol, spacing)
        img = torch.from_numpy(vol_prep[None,None].astype(np.float32)).to(device)
        with torch.no_grad(), amp.autocast(device_type=device_type, enabled=cfg.mixed_precision):
            logits = model(img)  # (1,14,Z,Y,X)
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # (14,Z,Y,X)
            heatmap = probs.max(0)  # (Z,Y,X)
            # normalize heatmap for safety
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
            else:
                heatmap = heatmap
            cands = extract_candidates_from_heatmap(heatmap, cfg.target_spacing)
            for cz, cy, cx, p in cands:
                cand_rows.append({"SeriesInstanceUID": uid, "z": int(cz), "y": int(cy), "x": int(cx), "prob_a": float(p)})
    cand_df = pd.DataFrame(cand_rows)
    cand_df.to_csv(out_candidates, index=False)
    print("Saved candidates:", out_candidates, "count:", len(cand_df))
    return cand_df



# ============================================================
# Cell 12 — Build Stage B Labels (GT overlap)
# ============================================================

def build_stageb_labels(cand_df, pos_radius_mm=4.0):
    """
    For each candidate (in Stage A preprocessed coordinates), label positive if within pos_radius_mm
    of any non-zero voxel in the cowseg mask (after same preprocess/resample/crop).
    """
    labeled = []
    for idx, row in tqdm(cand_df.iterrows(), total=len(cand_df), desc="Labeling candidates"):
        uid = row["SeriesInstanceUID"]
        cz, cy, cx = int(row["z"]), int(row["y"]), int(row["x"])
        mask, msp = load_clean_mask(uid)
        if mask is None:
            row['label'] = 0
            labeled.append(row)
            continue
        # resample mask and crop to stage A shape
        mask_res, _ = resample_to_spacing(mask.astype(np.float32), msp, cfg.target_spacing, order=0)
        mask_res = center_crop_or_pad(mask_res.astype(np.int32), cfg.a_patch_shape)
        # get GT coords
        coords = np.argwhere(mask_res > 0)
        if coords.shape[0] == 0:
            row['label'] = 0
            labeled.append(row)
            continue
        dz = (coords[:,0] - cz) * cfg.target_spacing
        dy = (coords[:,1] - cy) * cfg.target_spacing
        dx = (coords[:,2] - cx) * cfg.target_spacing
        dist = np.sqrt(dx*dx + dy*dy + dz*dz)
        row['label'] = int((dist <= pos_radius_mm).any())
        labeled.append(row)
    labeled_df = pd.DataFrame(labeled)
    pos = labeled_df['label'].sum()
    print("Labeled candidates:", len(labeled_df), "positives:", int(pos))
    return labeled_df



# ============================================================
# Cell 13 — Stage B Dataset (2.5D stacks) + DataLoader
# ============================================================

class StageBDataset(Dataset):
    def __init__(self, cand_df):
        self.df = cand_df.reset_index(drop=True)

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row["SeriesInstanceUID"]
        cz, cy, cx = int(row["z"]), int(row["y"]), int(row["x"])
        label = float(row.get("label", 0.0))

        # load preprocessed volume (Stage A preprocessing)
        try:
            vol, sp = load_dicom_series(uid)
            vol_prep, _ = preprocess_volume_for_stage_a(vol, sp)
        except Exception:
            vol_prep = np.zeros(cfg.a_patch_shape, dtype=np.float32)

        K = cfg.b_num_slices
        half = K // 2
        Z,Y,X = vol_prep.shape

        # axial slices
        z_idxs = np.clip(np.arange(cz-half, cz+half+1), 0, Z-1)
        axial = vol_prep[z_idxs]  # (K,Y,X)

        # coronal: fix y axis
        y_idxs = np.clip(np.arange(cy-half, cy+half+1), 0, Y-1)
        coronal = vol_prep[:, y_idxs, :]  # (Z,K,X) -> transpose to (K,Z,X)
        coronal = np.transpose(coronal, (1,0,2))

        # sagittal: fix x axis
        x_idxs = np.clip(np.arange(cx-half, cx+half+1), 0, X-1)
        sagittal = vol_prep[:, :, x_idxs]  # (Z,Y,K) -> transpose to (K,Z,Y)
        sagittal = np.transpose(sagittal, (2,0,1))

        # helper crop+resize function
        def crop_resize(stack):
            K, H, W = stack.shape
            min_hw = min(H, W)
            y0 = (H - min_hw)//2
            x0 = (W - min_hw)//2
            crop = stack[:, y0:y0+min_hw, x0:x0+min_hw]
            out = np.zeros((K, cfg.b_patch_size, cfg.b_patch_size), dtype=np.float32)
            for i in range(K):
                out[i] = cv2.resize(crop[i], (cfg.b_patch_size, cfg.b_patch_size), interpolation=cv2.INTER_LINEAR)
            return out

        axial = crop_resize(axial)
        coronal = crop_resize(coronal)
        sagittal = crop_resize(sagittal)

        inp = np.concatenate([axial, coronal, sagittal], axis=0)  # (3K, H, W)
        inp = torch.from_numpy(inp).float()

        return inp, torch.tensor(label).float(), uid

def get_stageb_loaders(labeled_df, batch_size=8):
    # group split by series uid to avoid leakage
    gkf = GroupKFold(n_splits=5)
    fold = 0
    tr_idx, va_idx = list(gkf.split(labeled_df, groups=labeled_df["SeriesInstanceUID"]))[fold]
    tr_df = labeled_df.iloc[tr_idx].reset_index(drop=True)
    va_df = labeled_df.iloc[va_idx].reset_index(drop=True)

    tr_ds = StageBDataset(tr_df)
    va_ds = StageBDataset(va_df)

    tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    va_loader = DataLoader(va_ds, batch_size=batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return tr_loader, va_loader



# ============================================================
# Cell 14 — Stage B Model (EfficientNet-B0) and helper
# ============================================================

class StageBModel(nn.Module):
    def __init__(self, in_ch, backbone_name="tf_efficientnet_b0", emb_dim=cfg.b_emb_dim):
        super().__init__()
        self.preconv = nn.Conv2d(in_ch, 3, kernel_size=1)  # map to 3 channels
        self.backbone = timm.create_model(backbone_name, pretrained=True, features_only=True)
        feats = self.backbone.feature_info.channels()[-1]
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(feats, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(emb_dim, 1)
        )

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.preconv(x)
        feats = self.backbone(x)[-1]
        pooled = self.pool(feats).flatten(1)
        logit = self.fc(pooled).squeeze(1)
        return logit



# ============================================================
# Cell 15 — Train Stage B (AMP-safe, tqdm, AUC)
# ============================================================

def train_stage_b(tr_loader, va_loader, save_path=os.path.join(cfg.OUT_DIR, "stageB_best.pth")):
    device = cfg.device
    device_type = cfg.device_type
    in_ch = 3 * cfg.b_num_slices

    model = StageBModel(in_ch=in_ch).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.b_lr)
    criterion = nn.BCEWithLogitsLoss()
    scaler = amp.GradScaler(enabled=cfg.mixed_precision)

    best_auc = 0.0
    history = {"train_loss": [], "val_loss": [], "val_auc": []}

    for epoch in range(cfg.max_epochs_b):
        model.train()
        tr_loss = 0.0
        pbar = tqdm(tr_loader, desc=f"Stage B Train Epoch {epoch+1}/{cfg.max_epochs_b}", leave=False)
        for imgs, labels, uids in pbar:
            imgs = imgs.to(device)
            labels = labels.to(device)

            with amp.autocast(device_type=device_type, enabled=cfg.mixed_precision):
                logits = model(imgs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            tr_loss += loss.item() * imgs.size(0)
            pbar.set_postfix(train_loss=tr_loss / ((pbar.n + 1) * (imgs.size(0))))

        tr_loss /= len(tr_loader.dataset)
        history["train_loss"].append(tr_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        all_preds, all_gts = [], []
        with torch.no_grad():
            pbar = tqdm(va_loader, desc=f"Stage B Val Epoch {epoch+1}/{cfg.max_epochs_b}", leave=False)
            for imgs, labels, uids in pbar:
                imgs = imgs.to(device)
                labels = labels.to(device)
                with amp.autocast(device_type=device_type, enabled=cfg.mixed_precision):
                    logits = model(imgs)
                    loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                preds = torch.sigmoid(logits).detach().cpu().numpy()
                all_preds.append(preds)
                all_gts.append(labels.detach().cpu().numpy())

        val_loss /= len(va_loader.dataset)
        all_preds = np.concatenate(all_preds)
        all_gts = np.concatenate(all_gts)
        try:
            auc = roc_auc_score(all_gts, all_preds)
        except:
            auc = 0.0

        history["val_loss"].append(val_loss)
        history["val_auc"].append(auc)

        print(f"[Stage B] Epoch {epoch+1} | train_loss={tr_loss:.4f} | val_loss={val_loss:.4f} | val_auc={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), save_path)
            print("   → Saved best Stage B model:", save_path)

    return model, history



# ============================================================
# Cell 16 — Build Final Series-level CSV (Stage A -> per-artery)
# ============================================================

def build_final_csv_from_stage_a(model_a_path, meta_df=df_a, out_csv=os.path.join(cfg.OUT_DIR, "final_output_stageA.csv"), artery_thr=0.25):
    device = cfg.device
    device_type = cfg.device_type
    model = LightUNet3D(in_ch=1, out_ch=14, base_ch=cfg.base_ch).to(device)
    model.load_state_dict(torch.load(model_a_path, map_location=device))
    model.eval()

    rows = []
    for idx, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc="Stage A series -> final CSV"):
        uid = row["SeriesInstanceUID"]
        try:
            vol, sp = load_dicom_series(uid)
        except Exception:
            continue
        vol_prep, _ = preprocess_volume_for_stage_a(vol, sp)
        img = torch.from_numpy(vol_prep[None,None].astype(np.float32)).to(device)
        with torch.no_grad(), amp.autocast(device_type=device_type, enabled=cfg.mixed_precision):
            logits = model(img)
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # (14,Z,Y,X)
            # per-artery max in channels 0..12 (13 arteries)
            per_artery_max = probs[:13].reshape(13,-1).max(axis=1)
            # flat = probs[:13].reshape(13, -1)
            # per_artery_max = np.percentile(flat, 99, axis=1) #chat says this two lines start from flat= is for some changes
            bin_preds = per_artery_max
            # bin_preds = (per_artery_max > artery_thr).astype(int) #this gives round off value so we comment it
            aneurysm_present = int(bin_preds.sum() > 0)
            # map to required column order (from your doc)
            col_order = [
                "Left Infraclinoid ICA","Right Infraclinoid ICA",
                "Left Supraclinoid ICA","Right Supraclinoid ICA",
                "Left MCA","Right MCA","AComm",
                "Left ACA","Right ACA",
                "Left PComm","Right PComm",
                "Basilar Tip","Other Posterior Circulation"
            ]
            # mapping assumption: our channel index 0..12 corresponds to doc order? If not, remap accordingly.
            # Here we assign channels 0..12 directly to columns in that order.
            rec = {"SeriesInstanceUID": uid}
            for i, col in enumerate(col_order):
                rec[col] = float(bin_preds[i])
            rec["Aneurysm Present"] = int(aneurysm_present)
            rows.append(rec)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_csv, index=False)
    print("Saved final CSV:", out_csv, "rows:", len(out_df))
    return out_df



# ============================================================
# Cell 17 — Plot and Save Training Histories
# ============================================================

def plot_history_stagea(history_a, save_path=os.path.join(cfg.OUT_DIR, "stageA_loss.png")):
    plt.figure(figsize=(6,4))
    plt.plot(history_a["train"], label="train_loss")
    plt.plot(history_a["val"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Stage A: Train/Val Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print("Saved Stage A loss plot:", save_path)
    plt.show()

def plot_history_stageb(history_b, save_path=os.path.join(cfg.OUT_DIR, "stageB_metrics.png")):
    plt.figure(figsize=(6,4))
    plt.plot(history_b["train_loss"], label="train_loss")
    plt.plot(history_b["val_loss"], label="val_loss")
    plt.plot(history_b["val_auc"], label="val_auc")
    plt.xlabel("Epoch")
    plt.title("Stage B: Loss & AUC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    print("Saved Stage B metrics plot:", save_path)
    plt.show()



# ============================================================
# Cell 18 — Sample segmentation visualizer (saves images)
# ============================================================

def save_sample_segmentations(model_path, meta_df=df_a.sample(min(6, len(df_a))), out_dir=os.path.join(cfg.OUT_DIR, "samples")):
    os.makedirs(out_dir, exist_ok=True)
    device = cfg.device
    device_type = cfg.device_type
    model = LightUNet3D(in_ch=1, out_ch=14, base_ch=cfg.base_ch).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    for idx, row in meta_df.iterrows():
        uid = row["SeriesInstanceUID"]
        try:
            vol, sp = load_dicom_series(uid)
        except Exception:
            continue
        vol_prep, _ = preprocess_volume_for_stage_a(vol, sp)
        img_t = torch.from_numpy(vol_prep[None,None].astype(np.float32)).to(device)
        with torch.no_grad(), amp.autocast(device_type=device_type, enabled=cfg.mixed_precision):
            logits = model(img_t)
            probs = torch.sigmoid(logits)[0].cpu().numpy()  # (14,Z,Y,X)
            heatmap = probs.max(0)
        # pick central axial slice
        zc = heatmap.shape[0] // 2
        axial_slice = vol_prep[zc]
        hm_slice = heatmap[zc]
        # normalize images
        axial_img = (axial_slice - axial_slice.min()) / (axial_slice.max() - axial_slice.min() + 1e-8)
        hm_img = (hm_slice - hm_slice.min()) / (hm_slice.max() - hm_slice.min() + 1e-8)
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        ax[0].imshow(axial_img, cmap='gray')
        ax[0].set_title(f"{uid} axial (z={zc})")
        ax[1].imshow(axial_img, cmap='gray')
        ax[1].imshow(hm_img, cmap='jet', alpha=0.5)
        ax[1].set_title("Heatmap overlay")
        plt.tight_layout()
        save_path = os.path.join(out_dir, f"{uid}_sample_z{zc}.png")
        plt.savefig(save_path, dpi=200)
        plt.close(fig)
        print("Saved sample:", save_path)



# DRIVER (run as a single cell after Cells 1..18)

# 1) Train Stage A
model_a, history_a = train_stage_a()

# 2) Save Stage A plots
plot_history_stagea(history_a)

# 3) Stage A inference -> candidates
cand_df = stage_a_inference_and_save(os.path.join(cfg.OUT_DIR, "stageA_best.pth"))



# 4) Label candidates (Stage B labels)
labeled_df = build_stageb_labels(cand_df)
labeled_df.to_csv(os.path.join(cfg.OUT_DIR, "stageB_labeled_candidates.csv"), index=False)

# 5) Build Stage B loaders
tr_loader_b, va_loader_b = get_stageb_loaders(labeled_df, batch_size=8)

# 6) Train Stage B
model_b, history_b = train_stage_b(tr_loader_b, va_loader_b)
plot_history_stageb(history_b)


# 7) Build final per-series CSV via Stage A
final_df = build_final_csv_from_stage_a(os.path.join(cfg.OUT_DIR, "stageA_best.pth"))
final_df.head()

# 8) Save sample segmentations (optional)
save_sample_segmentations(os.path.join(cfg.OUT_DIR, "stageA_best.pth"))



# final_df = build_final_csv_from_stage_a(os.path.join(cfg.OUT_DIR, "stageA_best.pth"))
# final_df.head()


# uid = train_df_a.iloc[0]["SeriesInstanceUID"]
# mask, msp = load_clean_mask(uid)
# print("Original mask unique:", np.unique(mask)[:20])

# mask_res, _ = resample_to_spacing(mask.astype(np.float32), msp, cfg.target_spacing, order=0)
# mask_res = center_crop_or_pad(mask_res.astype(np.int32), cfg.a_patch_shape)
# print("Resampled mask unique:", np.unique(mask_res)[:20])

# print("Foreground voxels:", (mask_res > 0).sum())



# uid = df_a.iloc[0]["SeriesInstanceUID"]

# vol, sp = load_dicom_series(uid)
# vol_prep, _ = preprocess_volume_for_stage_a(vol, sp)
# img = torch.from_numpy(vol_prep[None,None].astype(np.float32)).to(cfg.device)

# model = LightUNet3D(in_ch=1, out_ch=14, base_ch=cfg.base_ch).to(cfg.device)
# model.load_state_dict(torch.load(os.path.join(cfg.OUT_DIR, "stageA_best.pth"), map_location=cfg.device))
# model.eval()

# with torch.no_grad(), amp.autocast(device_type=cfg.device_type, enabled=cfg.mixed_precision):
#     logits = model(img)
#     probs = torch.sigmoid(logits)[0].cpu().numpy()

# print("Global min prob:", probs.min())
# print("Global max prob:", probs.max())
# print("Mean prob:", probs.mean())
# print("Median prob:", np.median(probs))


