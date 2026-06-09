# =========================
# Imports & Global Settings
# =========================

import os
import gc
import shutil
from collections import OrderedDict
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import polars as pl

import pydicom
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch import amp

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit

from tqdm.auto import tqdm

# Reproducibility-ish (kept light for speed)
def seed_everything(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(42)


# =====================
# Competition Constants
# =====================
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

# Paths
TRAIN_CSV_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
SERIES_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"

# Processing / Model config
TARGET_SIZE = (64, 64, 64)      # final (D,H,W)
TARGET_SPACING_MM = 1.0         # isotropic resample
CTA_WINDOW = (300.0, 700.0)     # (center, width) for CT (CTA)
MRI_Z_CLIP = 3.0                # clip z-score to ±3σ
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()  # enable mixed precision on GPU

# Training knobs
DO_TRAIN = True                 # This notebook is for training
EPOCHS = 50
BATCH_SIZE = 4
LR = 1e-3
WEIGHT_DECAY = 1e-5
ANEURYSM_PRESENT_BOOST = 1.0
PATIENCE = 8  # early stopping

# Runtime practicality
TRAIN_MAX_SERIES = 512
VAL_MAX_SERIES = 128

# Avoid CUDA in DataLoader workers (to prevent fork-CUDA issue)
NUM_WORKERS_TRAIN = 2 if torch.cuda.is_available() else 0
NUM_WORKERS_VAL = 0
PIN_MEMORY = False
PERSISTENT_WORKERS = True if NUM_WORKERS_TRAIN > 0 else False


# ==========================
# DICOM Processing Utilities
# ==========================
def _safe_zoom(volume: np.ndarray, zoom_factors: Tuple[float, ...], order: int = 1) -> np.ndarray:
    """Robust wrapper around ndimage.zoom to avoid rank mismatch and invalid factors."""
    volume = np.nan_to_num(volume, copy=False)
    zf = tuple(float(max(1e-6, f)) for f in zoom_factors)  # avoid zeros/negatives
    if len(zf) != volume.ndim:
        if len(zf) > volume.ndim:
            zf = zf[:volume.ndim]
        else:
            zf = (1.0,) * (volume.ndim - len(zf)) + zf
    return ndimage.zoom(volume, zf, order=order)

def _resize_slice(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2D slice to (out_h, out_w) using safe zoom."""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float32, copy=False)
    zy = out_h / max(h, 1)
    zx = out_w / max(w, 1)
    return _safe_zoom(arr, (zy, zx), order=1).astype(np.float32, copy=False)

# ==========================
# DICOM Series Processor
# ==========================
class DICOMProcessor:
    """Process DICOM series into normalized 3D volumes."""

    def __init__(
        self,
        target_size: Tuple[int, int, int] = TARGET_SIZE,
        target_spacing_mm: float = TARGET_SPACING_MM,
        cta_window: Tuple[float, float] = CTA_WINDOW,
        mri_z_clip: float = MRI_Z_CLIP,
    ):
        self.target_size = target_size
        self.target_spacing_mm = target_spacing_mm
        self.cta_window = cta_window
        self.mri_z_clip = mri_z_clip
        
        # Adjustment counters
        self.slope_adjustments = 0
        self.intercept_adjustments = 0
        self.adaptive_windowing_count = 0

    def _validate_and_apply_rescale(self, sl: np.ndarray, ds) -> np.ndarray:
        """Validate slope/intercept values and apply rescaling."""
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        
        # Validate slope
        if slope <= 0 or not np.isfinite(slope) or abs(slope) > 1000:
            slope = 1.0
            self.slope_adjustments += 1
        
        # Validate intercept with range clamping for extreme values
        if not np.isfinite(intercept):
            intercept = 0.0
            self.intercept_adjustments += 1
        elif abs(intercept) > 10000:
            intercept = np.clip(intercept, -2000, 0)
            self.intercept_adjustments += 1
        
        # Apply rescaling
        rescaled = sl * slope + intercept
        
        # Validate result
        if np.any(~np.isfinite(rescaled)):
            rescaled = np.nan_to_num(rescaled, copy=False)
        
        # Post-rescale range check
        min_val, max_val = rescaled.min(), rescaled.max()
        if min_val < -5000 or max_val > 10000:
            rescaled = np.clip(rescaled, -3000, 5000)
        
        return rescaled

    def _log_adjustment_summary(self):
        """Log summary of adjustments made during processing."""
        print(f"Processing adjustments - Slope: {self.slope_adjustments}, Intercept: {self.intercept_adjustments}, Adaptive windowing: {self.adaptive_windowing_count}")

    def load_dicom_series(self, series_path: str) -> np.ndarray:
        """Return (D,H,W) float32 volume in [0,1]."""
        try:
            # Collect DICOM datasets
            dicoms = []
            for root, _, files in os.walk(series_path):
                for f in files:
                    if f.endswith(".dcm"):
                        try:
                            ds = pydicom.dcmread(os.path.join(root, f), force=True)
                            if hasattr(ds, "PixelData"):
                                dicoms.append(ds)
                        except Exception:
                            continue
            if not dicoms:
                raise ValueError(f"No valid DICOM files with pixel data in {series_path}")

            dicoms = self._sort_slices(dicoms)
            has_multiframe = any(getattr(ds, "NumberOfFrames", 1) > 1 for ds in dicoms)
            spacing = self._get_spacing(dicoms, has_multiframe=has_multiframe)

            # Choose base HxW
            base_h, base_w = self._choose_base_shape(dicoms)

            modality_tag = (getattr(dicoms[0], "Modality", "") or "").upper()
            vol_slices = []

            for ds in dicoms:
                arr = ds.pixel_array
                # standardize to (N,H,W) where N=number of frames (1 if 2D)
                if arr.ndim >= 3:
                    h, w = arr.shape[-2], arr.shape[-1]
                    n = int(np.prod(arr.shape[:-2]))
                    arr = arr.reshape(n, h, w)
                    frames = arr
                else:
                    frames = arr[np.newaxis, ...]  # shape (1,H,W)

                for sl in frames:
                    sl = sl.astype(np.float32)

                    # Handle MONOCHROME1 inversion
                    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                        sl = sl.max() - sl

                    # Apply validated rescaling
                    sl = self._validate_and_apply_rescale(sl, ds)

                    sl = _resize_slice(sl, base_h, base_w)
                    vol_slices.append(sl)

            if len(vol_slices) == 0:
                raise ValueError("No valid slices extracted.")

            volume = np.stack(vol_slices, axis=0).astype(np.float32)  # (D,H,W)

            # Normalize by modality -> [0,1]
            volume = self._normalize_by_modality(volume, modality_tag)

            # Isotropic resample (mm-based)
            if self.target_spacing_mm is not None:
                dz, dy, dx = spacing
                z, y, x = volume.shape
                newD = max(1, int(round(z * dz / self.target_spacing_mm)))
                newH = max(1, int(round(y * dy / self.target_spacing_mm)))
                newW = max(1, int(round(x * dx / self.target_spacing_mm)))
                volume = _safe_zoom(volume, (newD / z, newH / y, newW / x), order=1)

            # Resize to target grid
            tz, ty, tx = self.target_size
            z, y, x = volume.shape
            volume = _safe_zoom(volume, (tz / z, ty / y, tx / x), order=1).astype(np.float32)

            return volume

        except Exception:
            return np.zeros(self.target_size, dtype=np.float32)

    def _sort_slices(self, ds_list: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
        try:
            orient = np.array(ds_list[0].ImageOrientationPatient, dtype=np.float32)
            row = orient[:3]; col = orient[3:]
            normal = np.cross(row, col)
            def sort_key(ds):
                ipp = np.array(getattr(ds, "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
                return float(np.dot(ipp, normal))
            return sorted(ds_list, key=sort_key)
        except Exception:
            return sorted(ds_list, key=lambda ds: getattr(ds, "InstanceNumber", 0))

    def _get_spacing(self, ds_sorted: List[pydicom.dataset.FileDataset], has_multiframe: bool = False) -> Tuple[float, float, float]:
        try:
            dy, dx = map(float, ds_sorted[0].PixelSpacing)
        except Exception:
            ps = getattr(ds_sorted[0], "PixelSpacing", [1.0, 1.0])
            dy, dx = float(ps[0]), float(ps[1])

        if has_multiframe:
            dz = float(getattr(ds_sorted[0], "SpacingBetweenSlices", getattr(ds_sorted[0], "SliceThickness", 1.0)))
        else:
            zs = []
            for i in range(1, len(ds_sorted)):
                p0 = np.array(getattr(ds_sorted[i-1], "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
                p1 = np.array(getattr(ds_sorted[i], "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
                d = np.linalg.norm(p1 - p0)
                if d > 0:
                    zs.append(d)
            if zs:
                dz = float(np.median(zs))
            else:
                dz = float(getattr(ds_sorted[0], "SliceThickness", 1.0))

        dz = dz if (dz > 0 and np.isfinite(dz)) else 1.0
        dy = dy if (dy > 0 and np.isfinite(dy)) else 1.0
        dx = dx if (dx > 0 and np.isfinite(dx)) else 1.0
        return (dz, dy, dx)

    def _choose_base_shape(self, ds_list: List[pydicom.dataset.FileDataset]) -> Tuple[int, int]:
        shapes = []
        for ds in ds_list:
            try:
                h, w = int(ds.Rows), int(ds.Columns)
            except Exception:
                arr = ds.pixel_array
                h, w = arr.shape[-2], arr.shape[-1]
            shapes.append((h, w))
        vals, counts = np.unique(shapes, return_counts=True, axis=0)
        base = tuple(vals[counts.argmax()])
        return int(base[0]), int(base[1])

    def _normalize_by_modality(self, volume: np.ndarray, modality_tag: str) -> np.ndarray:
        """CT: adaptive windowing for extreme ranges; MR: z-score -> clip -> [0,1]."""
        volume = np.nan_to_num(volume, copy=False)
        
        if modality_tag == "CT":
            min_val, max_val = volume.min(), volume.max()
            
            # Check if values are in normal CT range
            if min_val >= -2000 and max_val <= 4000:
                # Normal range: use standard windowing
                c, w = self.cta_window
                lo, hi = c - w / 2.0, c + w / 2.0
            else:
                # Extreme range: use adaptive windowing
                self.adaptive_windowing_count += 1
                
                # Percentile-based adaptive window
                p1, p99 = np.percentile(volume, [1, 99])
                margin = (p99 - p1) * 0.1
                lo = p1 - margin
                hi = p99 + margin
                
                # Ensure minimum window width
                if hi - lo < 100:
                    center = (hi + lo) / 2
                    lo = center - 50
                    hi = center + 50
            
            v = np.clip(volume, lo, hi)
            v = (v - lo) / (hi - lo + 1e-6)
            return v.astype(np.float32, copy=False)
        else:
            # MRI processing
            mean = float(volume.mean())
            std = float(volume.std() + 1e-6)
            
            # Validate statistics
            if std < 1e-6 or not np.isfinite(mean) or not np.isfinite(std):
                return np.full_like(volume, 0.5, dtype=np.float32)
            
            # Check dynamic range
            min_val, max_val = volume.min(), volume.max()
            if max_val - min_val < 1e-6:
                return np.full_like(volume, 0.5, dtype=np.float32)
            
            v = (volume - mean) / std
            zc = float(self.mri_z_clip)
            v = np.clip(v, -zc, zc)
            v = (v + zc) / (2.0 * zc)
            return v.astype(np.float32, copy=False)


# ==============================
# Preprocessed Dataset Container
# ==============================
class PreprocessedDataset(Dataset):
    """Dataset for preprocessed volumes stored in memory."""

    def __init__(self, volumes: Dict[str, np.ndarray], data_df: pd.DataFrame):
        self.volumes = volumes
        self.data_df = data_df.reset_index(drop=True)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        series_id = row[ID_COL]
        
        volume = self.volumes[series_id]
        if not volume.flags.writeable:
            volume = volume.copy()

        labels = row[LABEL_COLS].values.astype(np.float32)
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # (1,D,H,W)
        labels_tensor = torch.from_numpy(labels)
        return volume_tensor, labels_tensor

# =======================
# Model Class
# =======================
class Simple3DCNN(nn.Module):
    """Lightweight 3D CNN for multi-label classification (returns logits)."""

    def __init__(self, num_classes: int = len(LABEL_COLS)):
        super(Simple3DCNN, self).__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(16)
        self.pool1 = nn.MaxPool3d(2)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(32)
        self.pool2 = nn.MaxPool3d(2)

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm3d(64)
        self.pool3 = nn.MaxPool3d(2)

        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm3d(128)
        self.pool4 = nn.MaxPool3d(2)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.fc1 = nn.Linear(128 * 2 * 2 * 2, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (B,1,D,H,W)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x)); x = self.dropout1(x)
        x = F.relu(self.fc2(x)); x = self.dropout2(x)
        x = self.fc3(x)  # logits
        return x

# ============================
# Metric: Mean Weighted ColAUC
# ============================
AP_COL = 'Aneurysm Present'
LOC_COLS = LABEL_COLS[:-1]

def mean_weighted_colwise_auc(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame):
    """Implements the competition metric."""
    aucs = {}
    for c in LABEL_COLS:
        y_t = y_true_df[c].values
        y_p = y_pred_df[c].values
        if len(np.unique(y_t)) < 2:
            auc = 0.5
        else:
            auc = roc_auc_score(y_t, y_p)
        aucs[c] = auc
    ap = aucs[AP_COL]
    others = float(np.mean([aucs[c] for c in LOC_COLS]))
    final = 0.5 * (ap + others)
    return final, aucs

# =====================
# Training & Evaluation
# =====================
def compute_pos_weight(train_df: pd.DataFrame, label_cols: list, eps: float = 1.0) -> torch.Tensor:
    """pos_weight = (neg + eps) / (pos + eps) per column to counter class imbalance."""
    total = float(len(train_df))
    pos = train_df[label_cols].sum(axis=0).astype(float)
    neg = total - pos
    w = (neg + eps) / (pos + eps)
    return torch.tensor(w.values, dtype=torch.float32, device=DEVICE)

@torch.no_grad()
def evaluate_model(model: nn.Module, val_dataset: PreprocessedDataset, batch_size: int = 1):
    """Validation using preprocessed data."""
    model.eval()
    dl = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=0,
        pin_memory=False, persistent_workers=False
    )

    preds, trues = [], []
    val_loss = 0.0
    # Use pos_weight for consistency with training
    pw = compute_pos_weight(val_dataset.data_df, LABEL_COLS, eps=1.0).clone()
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    for vols, labels in tqdm(dl, desc="Val", leave=False):
        vols = vols.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with amp.autocast(device_type='cuda', enabled=USE_AMP):
            logits = model(vols)
            loss = criterion(logits, labels)
            probs = torch.sigmoid(logits)
        val_loss += float(loss.item()) * vols.size(0)
        preds.append(probs.cpu().numpy())
        trues.append(labels.cpu().numpy())

    y_pred = np.vstack(preds)
    y_true = np.vstack(trues)
    y_pred_df = pd.DataFrame(y_pred, columns=LABEL_COLS)
    y_true_df = pd.DataFrame(y_true, columns=LABEL_COLS)

    final, aucs = mean_weighted_colwise_auc(y_true_df, y_pred_df)
    val_loss = val_loss / max(len(val_dataset), 1)
    return final, aucs, val_loss

def preprocess_all_data(data_df: pd.DataFrame, series_dir: str, processor: DICOMProcessor) -> Dict[str, np.ndarray]:
    """Preprocess all DICOM series once at the beginning."""
    print("Preprocessing all DICOM series...")
    volumes = {}
    
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc="Processing"):
        series_id = row[ID_COL]
        series_path = os.path.join(series_dir, series_id)
        volume = processor.load_dicom_series(series_path)
        volumes[series_id] = volume
    
    processor._log_adjustment_summary()
    return volumes

def train_model(
    train_df: pd.DataFrame,
    series_dir: str,
    processor: DICOMProcessor,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    aneurysm_present_boost: float = ANEURYSM_PRESENT_BOOST,
    patience: int = PATIENCE,
    save_path: str = "/kaggle/working/model_weights.pth",
    warm_start_path: str = None,
    monitor: str = "auc",  # "auc" (maximize MW-ColAUC) or "loss" (minimize val_loss)
) -> nn.Module:
    """Train 3D CNN with preprocessed data."""
    assert monitor in {"auc", "loss"}, "monitor must be 'auc' or 'loss'"

    # Stratified split by AP (simple hold-out)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    ap = train_df[AP_COL].values
    train_idx, val_idx = next(sss.split(train_df, ap))

    tr_df = train_df.iloc[train_idx].copy()
    va_df = train_df.iloc[val_idx].copy()

    # Optional subsampling for runtime practicality
    if TRAIN_MAX_SERIES is not None and len(tr_df) > TRAIN_MAX_SERIES:
        tr_df = tr_df.sample(TRAIN_MAX_SERIES, random_state=42)
    if VAL_MAX_SERIES is not None and len(va_df) > VAL_MAX_SERIES:
        va_df = va_df.sample(VAL_MAX_SERIES, random_state=42)

    # Combine for preprocessing
    combined_df = pd.concat([tr_df, va_df], ignore_index=True)
    
    # Preprocess all data once
    all_volumes = preprocess_all_data(combined_df, series_dir, processor)
    
    # Create datasets
    train_volumes = {sid: all_volumes[sid] for sid in tr_df[ID_COL]}
    val_volumes = {sid: all_volumes[sid] for sid in va_df[ID_COL]}
    
    ds_tr = PreprocessedDataset(train_volumes, tr_df)
    ds_va = PreprocessedDataset(val_volumes, va_df)
    
    dl_tr = DataLoader(
        ds_tr, batch_size=batch_size, shuffle=True,
        num_workers=NUM_WORKERS_TRAIN, pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=2 if NUM_WORKERS_TRAIN > 0 else None
    )

    # Create model
    model = Simple3DCNN(num_classes=len(LABEL_COLS)).to(DEVICE)

    # Warm start if provided
    if warm_start_path and os.path.exists(warm_start_path):
        try:
            state = torch.load(warm_start_path, map_location='cpu')
            model.load_state_dict(state, strict=False)
            print(f"Warm-started from {warm_start_path}")
        except Exception as e:
            print(f"[Warm start warn] {e}")

    # Loss with pos_weight
    pos_weight = compute_pos_weight(tr_df, LABEL_COLS, eps=1.0).clone()
    if aneurysm_present_boost != 1.0:
        pos_weight[-1] = pos_weight[-1] * float(aneurysm_present_boost)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=("max" if monitor == "auc" else "min"),
        patience=1
    )
    scaler = amp.GradScaler(enabled=USE_AMP)

    # Initialize best score
    best_score = -float('inf') if monitor == "auc" else float('inf')
    best_state = None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        n_samples = 0

        pbar = tqdm(dl_tr, desc=f"Train {epoch}/{epochs}", leave=False)
        for vols, labels in pbar:
            vols = vols.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with amp.autocast(device_type='cuda', enabled=USE_AMP):
                logits = model(vols)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.item()) * vols.size(0)
            n_samples += vols.size(0)
            pbar.set_postfix(loss=f"{running/max(n_samples,1):.4f}")

        train_loss = running / max(n_samples, 1)

        # Validation
        try:
            final_auc, per_col, val_loss = evaluate_model(model, ds_va, batch_size=1)
            print(f"[Epoch {epoch}/{epochs}] train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | MW-ColAUC={final_auc:.4f}")
        except Exception as e:
            print(f"[Eval warning] {e}")
            final_auc, val_loss = -1.0, train_loss + 1.0

        # Choose monitored score
        score = final_auc if monitor == "auc" else val_loss

        # Step scheduler with monitored score
        scheduler.step(score)

        # Early stopping with monitored score
        is_better = (score > best_score) if monitor == "auc" else (score < best_score)
        if is_better:
            best_score = score
            best_state = model.state_dict()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {patience} epochs).")
                break

    # Load best and save
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path} (selected by monitor='{monitor}', best_score={best_score:.6f})")

    # Final validation summary
    try:
        final_auc, per_col, val_loss = evaluate_model(model, ds_va, batch_size=1)
        print(f"[Final Val] val_loss={val_loss:.4f} | MW-ColAUC={final_auc:.4f}")
    except Exception as e:
        print(f"[Eval warning] {e}")

    model.eval()
    return model


# =======================================
# Global Initialization, Training & Save
# =======================================
print("Initializing processor...")
processor = DICOMProcessor(
    target_size=TARGET_SIZE,
    target_spacing_mm=TARGET_SPACING_MM,
    cta_window=CTA_WINDOW,
    mri_z_clip=MRI_Z_CLIP,
)

model = None

if DO_TRAIN:
    try:
        full_df = pd.read_csv(TRAIN_CSV_PATH)
        print(f"Training on up to {TRAIN_MAX_SERIES} train series, batch={BATCH_SIZE}, epochs={EPOCHS}, patience={PATIENCE} ...")
        model = train_model(
            train_df=full_df,
            series_dir=SERIES_DIR,
            processor=processor,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            aneurysm_present_boost=ANEURYSM_PRESENT_BOOST,
            patience=PATIENCE,
            save_path="/kaggle/working/model_weights.pth",
            warm_start_path="/kaggle/input/model_weights.pth" if os.path.exists("/kaggle/input/model_weights.pth") else None,
            monitor="auc",
        )
        
    except Exception as e:
        print(f"[Train warning] {e}")
        model = Simple3DCNN(num_classes=len(LABEL_COLS)).to(DEVICE)
        torch.save(model.state_dict(), "/kaggle/working/model_weights.pth")
        print("Saved a randomly initialized model to /kaggle/working/model_weights.pth")
else:
    model = Simple3DCNN(num_classes=len(LABEL_COLS)).to(DEVICE)
    torch.save(model.state_dict(), "/kaggle/working/model_weights.pth")
    print("Training disabled. Saved untrained model to /kaggle/working/model_weights.pth")

print("Training notebook completed. Best epoch weights are saved at /kaggle/working/model_weights.pth")

