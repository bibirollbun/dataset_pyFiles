# ====================================================
# RSNA INTRACRANIAL ANEURYSM - STAGE 1 TRAINING (v2)
# Uses Stage-0 prebuilt v2 cache (volumes, masks, pseudo_masks, brainmasks, manifest)
# Two-phase training with per-sample segmentation weights and rich progress logs:
#   Phase 1: real masks weighted (real_seg_weight), synthetic seg weight = 0.0
#   Phase 2: real same, synthetic seg weight = small (default 0.075)
# Saves: stage1_phase1_best.pth, stage1_phase2_best.pth, stage1_segmentation_best.pth
# ====================================================

import os
import math
import time
import random
import numpy as np
import pandas as pd
import cv2
from scipy import ndimage
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from tqdm import tqdm


# ====================================================
# Config
# ====================================================
class Config:
    # --- Paths ---
    PREBUILT_ROOT = "/kaggle/input/rsna2025-v2-intracranial-aneurysm-detection-nb153/stage1_AneurysmNet_prebuilt_v2"
    MANIFEST_PATH = os.path.join(PREBUILT_ROOT, "meta/manifest.csv")
    VOLUMES_DIR   = os.path.join(PREBUILT_ROOT, "volumes")
    MASKS_DIR     = os.path.join(PREBUILT_ROOT, "masks")          # real
    PSEUDO_DIR    = os.path.join(PREBUILT_ROOT, "pseudo_masks")   # synthetic
    BRAINMASKS_DIR= os.path.join(PREBUILT_ROOT, "brainmasks")
    MANIFEST_EXTRA_FIELDS = True  # stage-0 adds brainmask_relpath, brain_voxel_fraction

    # --- Data ---
    TARGET_SIZE = (48, 112, 112)  # (D,H,W)
    USE_BRAINMASKS = True
    BRAINMASK_KEY = 'm'
    BRAINMASK_MIN_FRAC = 0.02  # if below, skip masking

    # --- Training ---
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MIXED_PRECISION = True
    STAGE1_BATCH_SIZE = 12  # bump to 12/16 if GPU mem allows
    NUM_WORKERS = 4        # bump to 2 original setting /4/8 unsupported on kaggle to reduce CPU bottlenecks
    PREFETCH_FACTOR = 4
    PERSISTENT_WORKERS = True
    STAGE1_LR = 2e-4
    WEIGHT_DECAY = 1e-4
    EPOCHS_PHASE1 = 15
    EPOCHS_PHASE2 = 10
    EARLY_STOP_PATIENCE = 5
    GRAD_ACCUM_STEPS = 8
    # Validation throughput
    VAL_BATCH_MULT = 1  # keep validation batch moderate to avoid I/O stalls
    VAL_NUM_WORKERS = 4  # allow more workers for validation to feed GPUs

    # --- Segmentation weights ---
    REAL_SEG_DEFAULT_W = 0.9      # strengthen real supervision by default
    PHASE1_SYNTH_SEG_W = 0.0
    PHASE2_SYNTH_SEG_W = 0.05     # slightly lower synthetic weight in fine-tune
    FOCAL_LOSS_WEIGHT = 0.2

    # --- Loss variants ---
    USE_TVERSKY = True
    TV_ALPHA = 0.3
    TV_BETA  = 0.7

    # --- Augmentation ---
    ZOOM_AUG_ENABLED = True        # enable positive-centric zoom aug
    ZOOM_AUG_POS_FRAC = 0.65       # probability to apply on positives with mask
    ZOOM_JITTER_VOX = 3            # jitter center by Â±voxels


    # --- Splits ---
    FOLDS = 1   # set >1 later if you want CV here
    SEED = 42

# ====================================================
# Utils
# ====================================================

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # speeds up fixed-size convs


def load_manifest_df() -> pd.DataFrame:
    df = pd.read_csv(Config.MANIFEST_PATH)
    # Required columns: series_id, label, vol_relpath
    for col in ["series_id", "label", "vol_relpath"]:
        if col not in df.columns:
            raise RuntimeError(f"Manifest missing required column: {col}")
    return df


def gpu_mem_str():
    if not torch.cuda.is_available():
        return "cpu"
    try:
        a = torch.cuda.memory_allocated() / (1024**3)
        r = torch.cuda.memory_reserved() / (1024**3)
        return f"{a:.2f}G/{r:.2f}G"
    except Exception:
        return "gpu"


# ====================================================
# Dataset
# ====================================================
class PrebuiltDataset(Dataset):
    def __init__(self, df: pd.DataFrame, phase_synth_w: float):
        self.df = df.reset_index(drop=True)
        self.phase_synth_w = float(phase_synth_w)
        # Oversample real-mask rows 3x to increase real supervision frequency
        try:
            has_real = self.df['mask_relpath'].fillna('').str.startswith('masks/')
            real_df = self.df[has_real]
            if len(real_df) > 0:
                self.df = pd.concat([self.df, real_df, real_df, real_df], ignore_index=True)
        except Exception:
            pass

    def __len__(self):
        return len(self.df)

    def _load_volume(self, sid: str) -> np.ndarray:
        path = os.path.join(Config.VOLUMES_DIR, f"{sid}.npy")
        v = np.load(path).astype(np.float32)  # (D,H,W), 0..1
        v = np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0)
        return v

    def _load_brainmask(self, sid: str, frac: Optional[float], shp: Tuple[int,int,int]):
        if not Config.USE_BRAINMASKS:
            return None
        if frac is not None and float(frac) < Config.BRAINMASK_MIN_FRAC:
            return None
        p = os.path.join(Config.BRAINMASKS_DIR, f"{sid}_brainmask.npz")
        if not os.path.exists(p):
            return None
        try:
            bm = np.load(p)[Config.BRAINMASK_KEY].astype(np.float32)
            bm = np.nan_to_num(bm, nan=0.0, posinf=1.0, neginf=0.0)
            if bm.shape != shp or bm.sum() <= 0:
                return None
            # guarantee a small safety margin by eroding high-frequency holes
            try:
                ker = np.ones((3,3), np.uint8)
                for z in range(bm.shape[0]):
                    bm[z] = cv2.morphologyEx(bm[z].astype(np.uint8), cv2.MORPH_CLOSE, ker)
            except Exception:
                pass
            return bm
        except Exception:
            return None

    def _load_mask(self, sid: str, mask_rel: str, is_synth: int, label: int) -> Tuple[np.ndarray, bool]:
        # Returns (mask[D,H,W] float32 in {0,1}, is_synthetic: bool)
        if isinstance(mask_rel, str) and len(mask_rel) > 0:
            if mask_rel.startswith('masks/'):
                p = os.path.join(Config.PREBUILT_ROOT, mask_rel)
                if os.path.exists(p):
                    m = np.load(p).astype(np.float32)
                    m = np.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0)
                    return (m > 0).astype(np.float32), False
            elif mask_rel.startswith('pseudo_masks/'):
                p = os.path.join(Config.PREBUILT_ROOT, mask_rel)
                if os.path.exists(p):
                    m = np.load(p).astype(np.float32)
                    m = np.nan_to_num(m, nan=0.0, posinf=1.0, neginf=0.0)
                    return (m > 0).astype(np.float32), True
        # Fallbacks
        D,H,W = Config.TARGET_SIZE
        if int(label) == 1:
            return np.zeros((D,H,W), dtype=np.float32), True
        else:
            return np.zeros((D,H,W), dtype=np.float32), False

    def _largest_component_bbox(self, mask: np.ndarray):
        if mask is None or mask.max() <= 0:
            return None
        labeled, num = ndimage.label(mask > 0)
        if num == 0:
            return None
        best_ct = 0
        best_cid = 0
        for cid in range(1, num+1):
            ct = int((labeled == cid).sum())
            if ct > best_ct:
                best_ct = ct
                best_cid = cid
        comp = (labeled == best_cid)
        idx = np.argwhere(comp)
        if idx.size == 0:
            return None
        z0,y0,x0 = idx.min(axis=0)
        z1,y1,x1 = idx.max(axis=0)
        return int(z0), int(z1), int(y0), int(y1), int(x0), int(x1)

    def _resize_volume_mask(self, vol: np.ndarray, msk: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        td, th, tw = Config.TARGET_SIZE
        D,H,W = vol.shape
        # depth index selection
        if D != td:
            idx = np.linspace(0, max(D-1,0), num=td).astype(int) if D>0 else np.zeros(td, dtype=int)
            vol = vol[idx]
            msk = msk[idx]
        # per-slice resize
        if (H, W) != (th, tw):
            outv = np.empty((td, th, tw), dtype=np.float32)
            outm = np.empty((td, th, tw), dtype=np.float32)
            for i in range(td):
                outv[i] = cv2.resize(vol[i].astype(np.float32), (tw, th))
                outm[i] = cv2.resize(msk[i].astype(np.float32), (tw, th), interpolation=cv2.INTER_NEAREST)
            vol, msk = outv, (outm > 0.5).astype(np.float32)
        return vol.astype(np.float32), msk.astype(np.float32)

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        sid = str(r['series_id'])
        label = int(r['label'])
        mask_rel = r.get('mask_relpath', '') if isinstance(r.get('mask_relpath', ''), str) else ''
        is_synth_col = int(r.get('is_synthetic', 0))
        real_seg_weight = r.get('real_seg_weight', np.nan)
        brain_frac = r.get('brain_voxel_fraction', np.nan)

        vol = self._load_volume(sid)  # (D,H,W)
        bm = self._load_brainmask(sid, brain_frac if pd.notna(brain_frac) else None, vol.shape)
        if bm is not None:
            vol = vol * bm  # gate

        mask, is_synth = self._load_mask(sid, mask_rel, is_synth_col, label)

        # Positive-centric zoom augmentation
        if Config.ZOOM_AUG_ENABLED and (mask.max() > 0) and (random.random() < Config.ZOOM_AUG_POS_FRAC):
            bbox = self._largest_component_bbox(mask)
            if bbox is not None:
                z0,z1,y0,y1,x0,x1 = bbox
                # expand bbox with jitter
                j = int(Config.ZOOM_JITTER_VOX)
                zc = max(0, min(vol.shape[0]-1, (z0+z1)//2 + random.randint(-j, j)))
                yc = max(0, min(vol.shape[1]-1, (y0+y1)//2 + random.randint(-j*2, j*2)))
                xc = max(0, min(vol.shape[2]-1, (x0+x1)//2 + random.randint(-j*2, j*2)))
                # choose cube edge roughly covering bbox
                dz = max(4, z1 - z0 + 6)
                dy = max(16, y1 - y0 + 24)
                dx = max(16, x1 - x0 + 24)
                edge_z = min(vol.shape[0], dz)
                edge_y = min(vol.shape[1], dy)
                edge_x = min(vol.shape[2], dx)
                z1a = max(0, zc - edge_z//2); z2a = min(vol.shape[0], z1a + edge_z)
                y1a = max(0, yc - edge_y//2); y2a = min(vol.shape[1], y1a + edge_y)
                x1a = max(0, xc - edge_x//2); x2a = min(vol.shape[2], x1a + edge_x)
                v_crop = vol[z1a:z2a, y1a:y2a, x1a:x2a]
                m_crop = mask[z1a:z2a, y1a:y2a, x1a:x2a]
                # resize back to target size
                vol, mask = self._resize_volume_mask(v_crop, m_crop)

        # per-sample seg weight
        if is_synth:
            seg_w = self.phase_synth_w
        else:
            if pd.notna(real_seg_weight):
                try:
                    rsw = float(real_seg_weight)
                except Exception:
                    rsw = Config.REAL_SEG_DEFAULT_W
            else:
                rsw = Config.REAL_SEG_DEFAULT_W
            seg_w = float(np.clip(rsw, 0.2, 1.0))

        # to tensors
        vol_t = torch.from_numpy(vol).unsqueeze(0)         # [1,D,H,W]
        mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        label_t = torch.tensor([float(label)], dtype=torch.float32)
        segw_t  = torch.tensor([float(seg_w)], dtype=torch.float32)

        return {
            'series_id': sid,
            'volume': vol_t,
            'mask': mask_t,
            'label': label_t,
            'seg_weight': segw_t,
            'is_synthetic_mask': torch.tensor([1.0 if is_synth else 0.0], dtype=torch.float32),
        }

# ====================================================
# Simple 3D UNet + classifier head
# ====================================================
class ConvBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1), nn.GroupNorm(num_groups=8, num_channels=out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1), nn.GroupNorm(num_groups=8, num_channels=out_ch), nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.net(x)

class UNet3D(nn.Module):
    def __init__(self, in_ch=1, base=24):
        super().__init__()
        b = base
        self.enc1 = ConvBlock3D(in_ch, b)
        self.pool1 = nn.MaxPool3d(2)
        self.enc2 = ConvBlock3D(b, b*2)
        self.pool2 = nn.MaxPool3d(2)
        self.enc3 = ConvBlock3D(b*2, b*4)
        self.pool3 = nn.MaxPool3d((2,2,2))
        self.bott = ConvBlock3D(b*4, b*8)
        self.up3 = nn.ConvTranspose3d(b*8, b*4, 2, stride=2)
        self.dec3 = ConvBlock3D(b*8, b*4)
        self.up2 = nn.ConvTranspose3d(b*4, b*2, 2, stride=2)
        self.dec2 = ConvBlock3D(b*4, b*2)
        self.up1 = nn.ConvTranspose3d(b*2, b, 2, stride=2)
        self.dec1 = ConvBlock3D(b*2, b)
        self.seg_head = nn.Conv3d(b, 1, 1)
        # classification head from bottleneck features
        self.cls_pool = nn.AdaptiveAvgPool3d(1)
        self.cls_head = nn.Linear(b*8, 1)

    def forward(self, x):  # x: [B,1,D,H,W]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        b  = self.bott(self.pool3(e3))
        # decoder
        d3 = self.up3(b)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)
        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)
        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)
        seg = self.seg_head(d1)  # [B,1,D,H,W]
        # classifier from bottleneck
        cls = self.cls_head(self.cls_pool(b).flatten(1))  # [B,1]
        return seg, cls

# ====================================================
# Losses
# ====================================================
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps
    def forward(self, logits, targets, reduction='mean'):
        probs = torch.sigmoid(logits)
        num = 2 * (probs * targets).sum(dim=(2,3,4)) + self.eps
        den = (probs.pow(2) + targets.pow(2)).sum(dim=(2,3,4)) + self.eps
        dice = 1 - (num / den)  # per-sample
        if reduction == 'none':
            return dice
        return dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, eps=1e-6):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
    def forward(self, logits, targets, reduction='mean'):
        probs = torch.sigmoid(logits).clamp(self.eps, 1-self.eps)
        ce = -(targets*torch.log(probs) + (1-targets)*torch.log(1-probs))
        pt = torch.where(targets==1, probs, 1-probs)
        loss = self.alpha * (1-pt).pow(self.gamma) * ce
        loss = loss.mean(dim=(2,3,4))  # per-sample
        if reduction == 'none':
            return loss
        return loss.mean()

class EnhancedCombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice_loss = DiceLoss()
        self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        # foreground-weighted BCE: compute pos_weight dynamically per-batch
        self.bce_vox = nn.BCEWithLogitsLoss(reduction='none')
        self.bce_cls = nn.BCEWithLogitsLoss()
    def forward(self, seg_logits, cls_logits, seg_targets, cls_targets, seg_weights: torch.Tensor):
        # clamp seg logits to avoid AMP overflow
        seg_logits = torch.nan_to_num(seg_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        B = seg_logits.shape[0]
        dice_ps = self.dice_loss(seg_logits, seg_targets, reduction='none')
        focal_ps= self.focal_loss(seg_logits, seg_targets, reduction='none')
        # compute foreground weighting
        with torch.no_grad():
            fg = seg_targets.sum(dim=(2,3,4)).clamp(min=1.0)
            tot = torch.tensor(seg_targets[0,0].numel(), device=seg_targets.device, dtype=seg_targets.dtype)
            bg = (tot - fg).clamp(min=1.0)
            pos_w = (bg / fg).view(-1, 1, 1, 1, 1)
        bce_elem = F.binary_cross_entropy_with_logits(seg_logits, seg_targets, weight=pos_w.expand_as(seg_targets), reduction='none')
        bce_ps  = bce_elem.view(B, -1).mean(dim=1)
        dice_ps = torch.nan_to_num(dice_ps, nan=0.0)
        focal_ps= torch.nan_to_num(focal_ps, nan=0.0)
        bce_ps  = torch.nan_to_num(bce_ps,  nan=0.0)
        seg_ps  = 0.5*dice_ps + 0.3*bce_ps + Config.FOCAL_LOSS_WEIGHT*focal_ps
        seg_ps  = torch.nan_to_num(seg_ps, nan=0.0)
        seg_w   = seg_weights.view(-1)
        if (seg_w == 0).all():
            seg_loss = seg_ps.new_tensor(0.0)
        else:
            seg_loss = (seg_ps * seg_w).mean()
        # classification loss
        cls_logits = torch.nan_to_num(cls_logits, nan=0.0, posinf=20.0, neginf=-20.0)
        cls_loss= self.bce_cls(cls_logits.view(-1), cls_targets.view(-1))
        total   = seg_loss + cls_loss
        return total, seg_loss.detach(), cls_loss.detach()

# ====================================================
# Train / Validate with progress bars
# ====================================================

def train_epoch(model, loader, optimizer, criterion, scaler, epoch=None, phase_name="P1"):
    model.train()
    t_loss = t_seg = t_cls = 0.0
    n = 0
    pbar = tqdm(loader, desc=f"Train {phase_name}{'' if epoch is None else f' [ep {epoch}]'}", leave=False, mininterval=0.1)
    optimizer.zero_grad(set_to_none=True)
    for iter_idx, batch in enumerate(pbar):
        vol   = batch['volume'].to(Config.DEVICE, non_blocking=True)
        try:
            vol = vol.to(memory_format=torch.channels_last_3d)
        except Exception:
            pass
        mask  = batch['mask'].to(Config.DEVICE, non_blocking=True)
        label = batch['label'].to(Config.DEVICE, non_blocking=True)
        segw  = batch['seg_weight'].to(Config.DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=Config.MIXED_PRECISION):
            seg_logits, cls_logits = model(vol)
            loss, seg_loss, cls_loss = criterion(seg_logits, cls_logits, mask, label, segw)
        try:
            scaler.scale(loss / Config.GRAD_ACCUM_STEPS).backward()
            if ((iter_idx + 1) % Config.GRAD_ACCUM_STEPS) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
        except Exception:
            optimizer.zero_grad(set_to_none=True)
            continue
        bs = vol.size(0)
        t_loss += loss.item()*bs; t_seg += seg_loss.item()*bs; t_cls += cls_loss.item()*bs; n += bs
        pbar.set_postfix(loss=f"{t_loss/max(n,1):.4f}", seg=f"{t_seg/max(n,1):.4f}", cls=f"{t_cls/max(n,1):.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")
    # finalize leftover accumulation
    if (len(loader) % Config.GRAD_ACCUM_STEPS) != 0:
        try:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        except Exception:
            pass
        finally:
            optimizer.zero_grad(set_to_none=True)
    return t_loss/n, t_seg/n, t_cls/n

@torch.no_grad()
def validate_epoch(model, loader, criterion, epoch=None, phase_name="P1"):
    model.eval()
    t_loss = t_seg = t_cls = 0.0
    n = 0
    all_probs = []
    all_labels= []
    pbar = tqdm(loader, desc=f"Valid {phase_name}{'' if epoch is None else f' [ep {epoch}]'}", leave=False, mininterval=0.1)
    for batch in pbar:
        vol   = batch['volume'].to(Config.DEVICE, non_blocking=True)
        try:
            vol = vol.to(memory_format=torch.channels_last_3d)
        except Exception:
            pass
        mask  = batch['mask'].to(Config.DEVICE, non_blocking=True)
        label = batch['label'].to(Config.DEVICE, non_blocking=True)
        segw  = batch['seg_weight'].to(Config.DEVICE, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=Config.MIXED_PRECISION):
            seg_logits, cls_logits = model(vol)
            loss, seg_loss, cls_loss = criterion(seg_logits, cls_logits, mask, label, segw)
        bs = vol.size(0)
        t_loss += loss.item()*bs; t_seg += seg_loss.item()*bs; t_cls += cls_loss.item()*bs; n += bs
        pbar.set_postfix(loss=f"{t_loss/max(n,1):.4f}", seg=f"{t_seg/max(n,1):.4f}", cls=f"{t_cls/max(n,1):.4f}")
        all_probs.append(torch.sigmoid(cls_logits).detach().cpu().view(-1).numpy())
        all_labels.append(label.detach().cpu().view(-1).numpy())
    all_probs = np.concatenate(all_probs) if len(all_probs)>0 else np.array([])
    all_labels = np.concatenate(all_labels) if len(all_labels)>0 else np.array([])
    auc = np.nan
    try:
        if len(all_probs)>0 and len(np.unique(all_labels)) > 1:
            auc = float(roc_auc_score(all_labels, all_probs))
    except Exception:
        pass
    return t_loss/n, t_seg/n, t_cls/n, auc

# ====================================================
# Main
# ====================================================

def run_training():
    set_seed(Config.SEED)
    # TF32 for better throughput
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
    df = load_manifest_df()

    # Build a single stratified split (can expand to CV later)
    y = df['label'].astype(int).values
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=Config.SEED)
    train_idx, val_idx = next(skf.split(np.zeros_like(y), y))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)

    # Phase 1 datasets/loads
    ds_train_p1 = PrebuiltDataset(train_df, phase_synth_w=Config.PHASE1_SYNTH_SEG_W)
    ds_val_p1   = PrebuiltDataset(val_df,   phase_synth_w=Config.PHASE1_SYNTH_SEG_W)
    dl_train = DataLoader(ds_train_p1, batch_size=Config.STAGE1_BATCH_SIZE, shuffle=True,
                          num_workers=Config.NUM_WORKERS, pin_memory=True,
                          prefetch_factor=Config.PREFETCH_FACTOR,
                          persistent_workers=Config.PERSISTENT_WORKERS)
    dl_val   = DataLoader(ds_val_p1,   batch_size=Config.STAGE1_BATCH_SIZE * Config.VAL_BATCH_MULT, shuffle=False,
                          num_workers=Config.VAL_NUM_WORKERS, pin_memory=True,
                          prefetch_factor=Config.PREFETCH_FACTOR,
                          persistent_workers=Config.PERSISTENT_WORKERS)

    model = UNet3D(in_ch=1, base=24).to(Config.DEVICE)
    try:
        model = model.to(memory_format=torch.channels_last_3d)
    except Exception:
        pass
    # Multi-GPU (if available): enable DP for train and val
    dp_enabled = False
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        dp_enabled = True
        print(f"Using {torch.cuda.device_count()} GPUs via DataParallel")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.STAGE1_LR, weight_decay=Config.WEIGHT_DECAY)
    criterion = EnhancedCombinedLoss().to(Config.DEVICE)
    scaler = torch.amp.GradScaler('cuda', enabled=Config.MIXED_PRECISION)

    best_loss = float('inf'); best_state = None; patience = 0
    for epoch in range(1, Config.EPOCHS_PHASE1+1):
        print(f"\n[Phase 1] Epoch {epoch}/{Config.EPOCHS_PHASE1}")
        tr_loss, tr_seg, tr_cls = train_epoch(model, dl_train, optimizer, criterion, scaler, epoch=epoch, phase_name='P1')
        # Always validate every epoch
        va_loss, va_seg, va_cls, va_auc = validate_epoch(model, dl_val, criterion, epoch=epoch, phase_name='P1')
        print(f"Train Loss: {tr_loss:.4f} | Seg: {tr_seg:.4f} | Cls: {tr_cls:.4f} | GPU {gpu_mem_str()}")
        print(f" Val  Loss: {va_loss:.4f} | Seg: {va_seg:.4f} | Cls: {va_cls:.4f} | AUC: {va_auc if not np.isnan(va_auc) else 'NA'} | GPU {gpu_mem_str()}")
        if va_loss < best_loss - 1e-5:
            best_loss = va_loss; best_state = {k:v.detach().cpu() for k,v in (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()).items()}; patience = 0
            torch.save(best_state, 'stage1_phase1_best.pth')
            print("ðŸ’¾ Saved Phase 1 best checkpoint")
        else:
            patience += 1
            if patience >= Config.EARLY_STOP_PATIENCE:
                print("Early stopping Phase 1")
                break

    if best_state is not None:
        if isinstance(model, nn.DataParallel):
            model.module.load_state_dict(best_state)
        else:
            model.load_state_dict(best_state)

    # Phase 2: small synthetic weight
    print("\n====== PHASE 2: enabling small synthetic seg supervision ======")
    ds_train_p2 = PrebuiltDataset(train_df, phase_synth_w=Config.PHASE2_SYNTH_SEG_W)
    ds_val_p2   = PrebuiltDataset(val_df,   phase_synth_w=Config.PHASE2_SYNTH_SEG_W)
    dl_train2 = DataLoader(ds_train_p2, batch_size=Config.STAGE1_BATCH_SIZE, shuffle=True,
                           num_workers=Config.NUM_WORKERS, pin_memory=True,
                           prefetch_factor=Config.PREFETCH_FACTOR,
                           persistent_workers=Config.PERSISTENT_WORKERS)
    dl_val2   = DataLoader(ds_val_p2,   batch_size=Config.STAGE1_BATCH_SIZE * Config.VAL_BATCH_MULT, shuffle=False,
                           num_workers=Config.VAL_NUM_WORKERS, pin_memory=True,
                           prefetch_factor=Config.PREFETCH_FACTOR,
                           persistent_workers=Config.PERSISTENT_WORKERS)

    # optional: lower LR a bit for fine-tune
    for g in optimizer.param_groups:
        g['lr'] = Config.STAGE1_LR * 0.5

    best2 = float('inf'); best2_state = None; patience = 0
    for epoch in range(1, Config.EPOCHS_PHASE2+1):
        print(f"\n[Phase 2] Epoch {epoch}/{Config.EPOCHS_PHASE2}")
        tr_loss, tr_seg, tr_cls = train_epoch(model, dl_train2, optimizer, criterion, scaler, epoch=epoch, phase_name='P2')
        va_loss, va_seg, va_cls, va_auc = validate_epoch(model, dl_val2, criterion, epoch=epoch, phase_name='P2')
        print(f"Train Loss: {tr_loss:.4f} | Seg: {tr_seg:.4f} | Cls: {tr_cls:.4f} | GPU {gpu_mem_str()}")
        print(f" Val  Loss: {va_loss:.4f} | Seg: {va_seg:.4f} | Cls: {va_cls:.4f} | AUC: {va_auc if not np.isnan(va_auc) else 'NA'} | GPU {gpu_mem_str()}")
        if va_loss < best2 - 1e-5:
            best2 = va_loss; best2_state = {k:v.detach().cpu() for k,v in (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()).items()}; patience = 0
            torch.save(best2_state, 'stage1_phase2_best.pth')
            print("ðŸ’¾ Saved Phase 2 best checkpoint")
        else:
            patience += 1
            if patience >= Config.EARLY_STOP_PATIENCE:
                print("Early stopping Phase 2")
                break

    final_state = best2_state or best_state or (model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict())
    torch.save(final_state, 'stage1_segmentation_best.pth')
    print("\nâœ… Stage 1 complete. Saved: stage1_segmentation_best.pth")


if __name__ == '__main__':
    run_training()


