# =========================
# Imports & Device Settings
# =========================
import os
import gc
import shutil
from collections import OrderedDict
from typing import Tuple, List

import numpy as np
import polars as pl
import pydicom
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp

import kaggle_evaluation.rsna_inference_server

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()


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

# =========
# File paths
# =========
MODEL_WEIGHTS_PATH = "/kaggle/input/voxel-by-voxel-3d-cnn-train/model_weights.pth"

# ==========================
# Processing configurations
# ==========================
TARGET_SIZE = (64, 64, 64)      # final volume dimensions (D,H,W)
TARGET_SPACING_MM = 1.0         # isotropic spacing in millimeters
CTA_WINDOW = (300.0, 700.0)     # CT windowing (center, width)
MRI_Z_CLIP = 3.0                # z-score clipping for MRI
LRU_CAPACITY = 8                # memory cache capacity


# ==========================
# Volume resizing utilities
# ==========================
def _safe_zoom(volume: np.ndarray, zoom_factors: Tuple[float, ...], order: int = 1) -> np.ndarray:
    """Apply ndimage zoom with protection against invalid factors and shapes."""
    volume = np.nan_to_num(volume, copy=False)
    zf = tuple(float(max(1e-6, f)) for f in zoom_factors)
    if len(zf) != volume.ndim:
        if len(zf) > volume.ndim:
            zf = zf[:volume.ndim]
        else:
            zf = (1.0,) * (volume.ndim - len(zf)) + zf
    return ndimage.zoom(volume, zf, order=order)

def _resize_slice(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2D slice to specified dimensions using bilinear interpolation."""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float32, copy=False)
    zy = out_h / max(h, 1)
    zx = out_w / max(w, 1)
    return _safe_zoom(arr, (zy, zx), order=1).astype(np.float32, copy=False)

# ==========================
# DICOM series processor
# ==========================
class DICOMProcessor:
    """Process DICOM series into normalized 3D volumes with LRU caching."""

    def __init__(
        self,
        target_size: Tuple[int, int, int] = TARGET_SIZE,
        target_spacing_mm: float = TARGET_SPACING_MM,
        cta_window: Tuple[float, float] = CTA_WINDOW,
        mri_z_clip: float = MRI_Z_CLIP,
        lru_capacity: int = LRU_CAPACITY,
    ):
        self.target_size = target_size
        self.target_spacing_mm = target_spacing_mm
        self.cta_window = cta_window
        self.mri_z_clip = mri_z_clip
        self.memory_cache = OrderedDict()
        self.lru_capacity = lru_capacity
        
        # Processing adjustment counters
        self.slope_adjustments = 0
        self.intercept_adjustments = 0
        self.adaptive_windowing_count = 0

    def _cache_put(self, key: str, vol: np.ndarray):
        """Store volume in LRU cache."""
        self.memory_cache[key] = vol
        self.memory_cache.move_to_end(key)
        if len(self.memory_cache) > self.lru_capacity:
            self.memory_cache.popitem(last=False)

    def _cache_get(self, key: str):
        """Retrieve volume from LRU cache."""
        if key in self.memory_cache:
            vol = self.memory_cache[key]
            self.memory_cache.move_to_end(key)
            return vol
        return None

    def _validate_and_apply_rescale(self, sl: np.ndarray, ds) -> np.ndarray:
        """Validate slope/intercept values and apply rescaling with robust error handling."""
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

    def load_dicom_series(self, series_path: str) -> np.ndarray:
        """Load and process DICOM series into normalized 3D volume."""
        series_id = os.path.basename(series_path)

        # Check memory cache first
        cached_volume = self._cache_get(series_id)
        if cached_volume is not None and isinstance(cached_volume, np.ndarray) and cached_volume.shape == self.target_size:
            return cached_volume

        try:
            # Collect DICOM datasets with pixel data
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

            # Sort slices spatially
            dicoms = self._sort_slices(dicoms)

            # Determine if multiframe series
            has_multiframe = any(getattr(ds, "NumberOfFrames", 1) > 1 for ds in dicoms)

            # Calculate spatial spacing
            spacing = self._get_spacing(dicoms, has_multiframe=has_multiframe)

            # Determine consistent base shape
            base_h, base_w = self._choose_base_shape(dicoms)

            # Extract modality
            modality_tag = (getattr(dicoms[0], "Modality", "") or "").upper()

            # Process pixel data from all datasets
            vol_slices = []
            for ds in dicoms:
                arr = ds.pixel_array
                # Standardize to (N,H,W) where N=number of frames (1 if 2D)
                if arr.ndim >= 3:
                    h, w = arr.shape[-2], arr.shape[-1]
                    n = int(np.prod(arr.shape[:-2]))
                    arr = arr.reshape(n, h, w)
                    frames = arr
                else:
                    frames = arr[np.newaxis, ...]  # shape (1,H,W)

                for sl in frames:
                    sl = sl.astype(np.float32)

                    # Handle inverted grayscale
                    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                        sl = sl.max() - sl

                    # Apply validated rescaling
                    sl = self._validate_and_apply_rescale(sl, ds)

                    # Standardize slice dimensions
                    sl = _resize_slice(sl, base_h, base_w)
                    vol_slices.append(sl)

            if len(vol_slices) == 0:
                raise ValueError("No valid slices extracted.")

            # Create 3D volume
            volume = np.stack(vol_slices, axis=0).astype(np.float32)  # (D,H,W)

            # Apply modality-specific normalization
            volume = self._normalize_by_modality(volume, modality_tag)

            # Resample to isotropic spacing
            if self.target_spacing_mm is not None:
                dz, dy, dx = spacing
                z, y, x = volume.shape
                newD = max(1, int(round(z * dz / self.target_spacing_mm)))
                newH = max(1, int(round(y * dy / self.target_spacing_mm)))
                newW = max(1, int(round(x * dx / self.target_spacing_mm)))
                volume = _safe_zoom(volume, (newD / z, newH / y, newW / x), order=1)

            # Resize to target dimensions
            tz, ty, tx = self.target_size
            z, y, x = volume.shape
            volume = _safe_zoom(volume, (tz / z, ty / y, tx / x), order=1).astype(np.float32)

            # Cache and return
            self._cache_put(series_id, volume)
            return volume

        except Exception:
            vol = np.zeros(self.target_size, dtype=np.float32)
            self._cache_put(series_id, vol)
            return vol

    def _sort_slices(self, ds_list: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
        """Sort DICOM slices by spatial position along normal vector."""
        try:
            orient = np.array(ds_list[0].ImageOrientationPatient, dtype=np.float32)
            row = orient[:3]
            col = orient[3:]
            normal = np.cross(row, col)
            
            def sort_key(ds):
                ipp = np.array(getattr(ds, "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
                return float(np.dot(ipp, normal))
            
            return sorted(ds_list, key=sort_key)
        except Exception:
            return sorted(ds_list, key=lambda ds: getattr(ds, "InstanceNumber", 0))

    def _get_spacing(self, ds_sorted: List[pydicom.dataset.FileDataset], has_multiframe: bool = False) -> Tuple[float, float, float]:
        """Calculate spatial spacing in millimeters (dz, dy, dx)."""
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

        # Validate spacing values
        dz = dz if (dz > 0 and np.isfinite(dz)) else 1.0
        dy = dy if (dy > 0 and np.isfinite(dy)) else 1.0
        dx = dx if (dx > 0 and np.isfinite(dx)) else 1.0
        return (dz, dy, dx)

    def _choose_base_shape(self, ds_list: List[pydicom.dataset.FileDataset]) -> Tuple[int, int]:
        """Select most common image dimensions as base shape."""
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
        """Apply modality-specific normalization with adaptive windowing for CT."""
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
            # MRI processing with robust statistics validation
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

# =======================
# 3D CNN model
# =======================
class Simple3DCNN(nn.Module):
    """Lightweight 3D CNN for multi-label aneurysm classification."""
    
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
        """Forward pass through 3D CNN."""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)  # return logits
        return x


# =====================================
# Initialize processor and load model
# =====================================
# Create DICOM processor with LRU caching
processor = DICOMProcessor(
    target_size=TARGET_SIZE,
    target_spacing_mm=TARGET_SPACING_MM,
    cta_window=CTA_WINDOW,
    mri_z_clip=MRI_Z_CLIP,
    lru_capacity=LRU_CAPACITY,
)

# Initialize model and load trained weights
model = Simple3DCNN(num_classes=len(LABEL_COLS)).to(DEVICE)
try:
    state = torch.load(MODEL_WEIGHTS_PATH, map_location='cpu')
    model.load_state_dict(state, strict=True)
    print(f"Loaded weights from {MODEL_WEIGHTS_PATH}")
except Exception as e:
    print(f"[Weight loading warning] {e}\nProceeding with randomly initialized weights.")
model.eval()

# ==============
# Inference API
# ==============
@torch.no_grad()
def predict(series_path: str) -> pl.DataFrame:
    """Main prediction function called by evaluation server."""
    try:
        # Process DICOM series to normalized volume
        volume = processor.load_dicom_series(series_path)  # (D,H,W) in [0,1]
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,D,H,W)

        # Run inference with mixed precision
        with amp.autocast(device_type='cuda', enabled=USE_AMP):
            logits = model(volume_tensor)               # (1,14)
            probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()

        # Format results as DataFrame
        result_df = pl.DataFrame(data=[probs], schema=LABEL_COLS, orient='row')

        # Clean up GPU memory
        del volume_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"[Prediction error] {e}")
        # Return neutral probabilities on failure
        result_df = pl.DataFrame(data=[[0.5] * len(LABEL_COLS)], schema=LABEL_COLS, orient='row')

    # Clean up temporary files
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    return result_df


# ==========================
# Run evaluation server
# ==========================
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    try:
        sub = pl.read_parquet('/kaggle/working/submission.parquet')
        print(sub.head())
    except Exception as e:
        print(f"Submission parquet not found: {e}")

