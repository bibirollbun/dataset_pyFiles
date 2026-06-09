%%time
import os
import gc
import shutil
from collections import OrderedDict
from typing import Tuple, List
import seaborn as sns
import pandas as pd
import cv2
from glob import glob
import numpy as np
import polars as pl
import pydicom
from scipy import ndimage
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import amp
import warnings
warnings.filterwarnings('ignore')

import kaggle_evaluation.rsna_inference_server

# Device / AMP
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
USE_AMP = torch.cuda.is_available()


%%time
train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
localizer_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")

print("Training samples:", len(train_df))
print("Aneurysm prevalence: {:.2f}%".format(train_df["Aneurysm Present"].mean() * 100))   


%%time
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Age distribution
sns.histplot(train_df, x="PatientAge", hue="Aneurysm Present", ax=axes[0], bins=30)
axes[0].set_title("Age Distribution")

# Sex distribution
sns.countplot(data=train_df, x="PatientSex", hue="Aneurysm Present", ax=axes[1])
axes[1].set_title("Sex vs Aneurysm")

# Modality distribution
sns.countplot(data=train_df, x="Modality", ax=axes[2])
axes[2].set_title("Imaging Modality")

plt.tight_layout()
plt.show()


%%time
def load_dicom_slice(dicom_path):
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array
    img = (img - img.min()) / (img.max() - img.min() + 1e-6)  # Normalize
    return (img * 255).astype(np.uint8)

def get_first_aneurysm_series():
    return train_df[train_df["Aneurysm Present"] == 1].iloc[0]["SeriesInstanceUID"]

def get_first_normal_series():
    return train_df[train_df["Aneurysm Present"] == 0].iloc[0]["SeriesInstanceUID"]   


%%time
def plot_aneurysm_with_localizer(series_uid):
    # Get all DICOMs in series
    series_path = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_uid}"
    dicom_files = sorted(glob(os.path.join(series_path, "*.dcm")))
    
    if len(dicom_files) == 0:
        print("No DICOMs found.")
        return

    # Find matching localizer
    loc_row = localizer_df[localizer_df["SeriesInstanceUID"] == series_uid]
    if len(loc_row) == 0:
        print("No localizer for this series.")
        return

    sop_uid = loc_row.iloc[0]["SOPInstanceUID"]
    x, y = eval(loc_row.iloc[0]["coordinates"])  # e.g., "[120, 150]"
    location = loc_row.iloc[0]["location"]

    # Find and show that slice
    target_file = None
    for f in dicom_files:
        if sop_uid in f:
            target_file = f
            break

    if not target_file:
        print("Localizer SOP not found in series.")
        return

    img = load_dicom_slice(target_file)

    plt.figure(figsize=(8, 8))
    plt.imshow(img, cmap="gray")
    plt.scatter(x, y, color="red", s=200, marker='x')
    plt.title(f"Aneurysm: {location} (x={x}, y={y})\nSeries: {series_uid}")
    plt.axis("off")
    plt.show()

# Show one aneurysm case
aneurysm_series = get_first_aneurysm_series()
#print(aneurysm_series)
plot_aneurysm_with_localizer(aneurysm_series) 


%%time
def load_dicom_volume(series_path):
    dcm_paths = sorted(glob(os.path.join(series_path, "*.dcm")))
    datasets = [pydicom.dcmread(p) for p in dcm_paths]
    
    # Sort by Z position
    z_pos = [float(d.ImagePositionPatient[2]) for d in datasets]
    sorted_dcms = [d for _, d in sorted(zip(z_pos, datasets))]
    
    volume = np.stack([d.pixel_array for d in sorted_dcms])
    return volume

def create_mip(volume, axis=0):
    """Create MIP along given axis"""
    return np.max(volume, axis=axis)

def plot_mip_example(series_uid):
    series_path = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_uid}"
    volume = load_dicom_volume(series_path)
    
    # Create MIPs
    mip_axial = create_mip(volume, axis=0)  # (H, W)
    mip_coronal = create_mip(volume, axis=1)  # (D, W)
    mip_sagittal = create_mip(volume, axis=2)  # (D, H)

    # Normalize
    def norm(img):
        return (img - img.min()) / (img.max() - img.min() + 1e-6)

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(norm(mip_axial), cmap="gray")
    axes[0].set_title("Axial MIP")
    axes[0].axis("off")

    axes[1].imshow(norm(mip_coronal), cmap="gray")
    axes[1].set_title("Coronal MIP")
    axes[1].axis("off")

    axes[2].imshow(norm(mip_sagittal), cmap="gray")
    axes[2].set_title("Sagittal MIP")
    axes[2].axis("off")

    plt.suptitle(f"MIPs - Series: {series_uid}")
    plt.tight_layout()
    plt.show()

# Show MIPs for aneurysm case
plot_mip_example(aneurysm_series)   


%%time
normal_series = get_first_normal_series()

print("✅ Aneurysm Case")
plot_mip_example(aneurysm_series)

print("✅ Normal Case")
plot_mip_example(normal_series)   


%%time
# Aneurysm presence by modality
plt.figure(figsize=(8, 5))
sns.countplot(data=train_df, x='Modality', hue='Aneurysm Present')
plt.title('Aneurysm Distribution by Imaging Modality')
plt.xlabel('Modality')
plt.ylabel('Count')
plt.legend(title='Aneurysm Present', labels=['No', 'Yes'])
plt.show()

# Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(data=train_df, x='PatientAge', hue='Aneurysm Present', bins=30, kde=True)
plt.title('Patient Age Distribution by Aneurysm Status')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


%%time
from typing import Tuple, List

# =====================
# Competition Constants
# =====================
ID_COL: str = 'SeriesInstanceUID'

LABEL_COLS: List[str] = [
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
# File Paths
# =========
MODEL_WEIGHTS_PATH: str = "/kaggle/input/voxel-by-voxel-3d-cnn-train/model_weights.pth"

# ==========================
# Processing & Model Configs
# ==========================
TARGET_SIZE: Tuple[int, int, int] = (64, 64, 64)      # (Depth, Height, Width)
TARGET_SPACING_MM: float = 1.0                        # Isotropic voxel spacing (mm)
CTA_WINDOW: Tuple[float, float] = (300.0, 700.0)      # (center, width) for CT windowing
MRI_Z_CLIP: float = 3.0                               # Z-score clipping threshold for MRI
LRU_CAPACITY: int = 8                                 # Cache size for LRU memory   


%%time
# ==========================
# Utility Resizing Functions
# ==========================
from typing import Tuple, Sequence
import numpy as np
from scipy import ndimage

def _safe_zoom(
    volume: np.ndarray,
    zoom_factors: Sequence[float],
    order: int = 1
) -> np.ndarray:
    """
    Safely resample a volume using zoom factors.
    
    Ensures zoom factors are valid and compatible with volume dimensions.
    Uses `ndimage.zoom` with bounds protection.
    
    Args:
        volume: Input array of any dimension.
        zoom_factors: Scaling factors per dimension (applied in order).
        order: Interpolation order (0=nearest, 1=linear, 3=cubic).
    
    Returns:
        Resampled array with shape determined by zoom_factors.
    """
    # Ensure finite values
    volume = np.nan_to_num(volume, copy=False)

    # Normalize zoom_factors to match volume dimensions
    ndim = volume.ndim
    if len(zoom_factors) != ndim:
        # Extend or truncate zoom_factors to match ndim
        if len(zoom_factors) > ndim:
            zoom_factors = zoom_factors[:ndim]
        else:
            # Prepend 1.0s for leading dimensions (e.g., time, batch)
            zoom_factors = (1.0,) * (ndim - len(zoom_factors)) + tuple(zoom_factors)
    
    # Clamp zoom factors to avoid numerical issues
    zoom_factors = tuple(max(1e-6, float(f)) for f in zoom_factors)
    
    return ndimage.zoom(volume, zoom_factors, order=order, mode='nearest')


def _resize_slice(
    arr: np.ndarray,
    out_h: int,
    out_w: int
) -> np.ndarray:
    """
    Resize a 2D array to target height and width using linear interpolation.
    
    Args:
        arr: 2D input array (H, W).
        out_h: Target height.
        out_w: Target width.
    
    Returns:
        Resized 2D array of shape (out_h, out_w), float32 dtype.
    """
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float32, copy=False)
    
    # Avoid division by zero
    scale_h = out_h / h
    scale_w = out_w / w
    
    return _safe_zoom(arr, (scale_h, scale_w), order=1).astype(np.float32, copy=False)   


%%time
# === IMPORTS ===
from typing import List, Tuple, Optional
import os
import numpy as np
from collections import OrderedDict
import pydicom
from scipy import ndimage

# === CONSTANTS (Ensure these are defined earlier in the notebook) ===
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

TARGET_SIZE = (64, 64, 64)
TARGET_SPACING_MM = 1.0
CTA_WINDOW = (300.0, 700.0)
MRI_Z_CLIP = 3.0
LRU_CAPACITY = 8   


%%time
# ==========================
# Utility Resizing Functions
# ==========================
def _safe_zoom(
    volume: np.ndarray,
    zoom_factors: Tuple[float, ...],
    order: int = 1
) -> np.ndarray:
    """Safely resample volume using zoom factors with dimension alignment and bounds protection."""
    volume = np.nan_to_num(volume, nan=0.0, posinf=None, neginf=None)
    
    ndim = volume.ndim
    if len(zoom_factors) != ndim:
        if len(zoom_factors) > ndim:
            zoom_factors = zoom_factors[:ndim]
        else:
            zoom_factors = (1.0,) * (ndim - len(zoom_factors)) + tuple(zoom_factors)
    
    zoom_factors = tuple(max(1e-6, float(f)) for f in zoom_factors)
    return ndimage.zoom(volume, zoom_factors, order=order, mode='nearest')


def _resize_slice(
    arr: np.ndarray,
    out_h: int,
    out_w: int
) -> np.ndarray:
    """Resize 2D array to (out_h, out_w) using linear interpolation."""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float32, copy=False)
    
    scale_h, scale_w = out_h / h, out_w / w
    return _safe_zoom(arr, (scale_h, scale_w), order=1).astype(np.float32, copy=False)   


%%time
# ==========================
# DICOM Series Processor
# ==========================
from collections import OrderedDict
from typing import Tuple, Optional
import numpy as np

class DICOMProcessor:
    """
    Convert a DICOM series folder into a normalized 3D volume (D, H, W) in [0, 1].
    Applies spacing correction, resizing, and modality-specific intensity normalization.
    """
    
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
        self.lru_capacity = lru_capacity
        self.memory_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    # -------------------
    # LRU Cache Management
    # -------------------
    def _cache_get(self, key: str) -> Optional[np.ndarray]:
        """Retrieve volume from cache and update access order."""
        if key in self.memory_cache:
            # Move to end to mark as recently used
            self.memory_cache.move_to_end(key)
            return self.memory_cache[key]
        return None

    def _cache_put(self, key: str, vol: np.ndarray) -> None:
        """Insert volume into cache, enforcing LRU eviction policy."""
        self.memory_cache[key] = vol
        self.memory_cache.move_to_end(key)
        if len(self.memory_cache) > self.lru_capacity:
            self.memory_cache.popitem(last=False)  # Remove oldest   


%%time
# ==========================
# Utility Resizing Functions
# ==========================
def _safe_zoom(
    volume: np.ndarray,
    zoom_factors: Tuple[float, ...],
    order: int = 1
) -> np.ndarray:
    """Safely resample volume using zoom factors with dimension alignment and bounds protection."""
    volume = np.nan_to_num(volume, nan=0.0, posinf=None, neginf=None)
    
    ndim = volume.ndim
    if len(zoom_factors) != ndim:
        if len(zoom_factors) > ndim:
            zoom_factors = zoom_factors[:ndim]
        else:
            zoom_factors = (1.0,) * (ndim - len(zoom_factors)) + tuple(zoom_factors)
    
    zoom_factors = tuple(max(1e-6, float(f)) for f in zoom_factors)
    return ndimage.zoom(volume, zoom_factors, order=order, mode='nearest')


def _resize_slice(
    arr: np.ndarray,
    out_h: int,
    out_w: int
) -> np.ndarray:
    """Resize 2D array to (out_h, out_w) using linear interpolation."""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float32, copy=False)
    
    scale_h, scale_w = out_h / h, out_w / w
    return _safe_zoom(arr, (scale_h, scale_w), order=1).astype(np.float32, copy=False)   


    # -------------------
    # LRU Cache Management
    # -------------------
    def _cache_put(self, key: str, vol: np.ndarray) -> None:
        """Insert volume into cache, enforcing LRU eviction policy."""
        self.memory_cache[key] = vol
        self.memory_cache.move_to_end(key)
        if len(self.memory_cache) > self.lru_capacity:
            self.memory_cache.popitem(last=False)  # Remove least recently used   


 def _sort_slices(self, ds_list: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
        """
        Sort slices by spatial position along the slice normal vector.
        Falls back to InstanceNumber if geometric sorting fails.
        """
        try:
            orient = np.array(ds_list[0].ImageOrientationPatient, dtype=np.float32)
            row = orient[:3]
            col = orient[3:]
            normal = np.cross(row, col)
            ipp_list = []
            for ds in ds_list:
                ipp = np.array(getattr(ds, "ImagePositionPatient", [0.0, 0.0, 0.0]), dtype=np.float32)
                ipp_list.append(np.dot(ipp, normal))
            sorted_idx = np.argsort(ipp_list)
            return [ds_list[i] for i in sorted_idx]
        except Exception:
            return sorted(ds_list, key=lambda ds: getattr(ds, "InstanceNumber", 0))   


 def _choose_base_shape(self, ds_list: List[pydicom.dataset.FileDataset]) -> Tuple[int, int]:
        """
        Choose the most frequent (Rows, Columns) as base 2D shape.
        Falls back to pixel_array shape if metadata missing.
        """
        shapes = []
        for ds in ds_list:
            try:
                h, w = int(ds.Rows), int(ds.Columns)
                if h > 0 and w > 0:
                    shapes.append((h, w))
                    continue
            except Exception:
                pass

            # Fallback: use pixel_array
            arr = ds.pixel_array
            if arr.ndim >= 2:
                h, w = arr.shape[-2], arr.shape[-1]
                if h > 0 and w > 0:
                    shapes.append((h, w))

        if not shapes:
            return self.target_size[1], self.target_size[2]  # Fallback to target (H, W)

        # Find most frequent shape
        shapes_arr = np.array(shapes)
        unique_shapes, counts = np.unique(shapes_arr, axis=0, return_counts=True)
        best_shape = unique_shapes[np.argmax(counts)]
        return int(best_shape[0]), int(best_shape[1])   


 def _normalize_by_modality(self, volume: np.ndarray, modality_tag: str) -> np.ndarray:
        """
        Normalize intensity based on modality:
          - CT: Apply CTA window (center, width) and scale to [0,1].
          - MR (or other): Z-score, clip outliers, then scale to [0,1].
        
        Returns:
            Normalized volume with values in [0,1], same shape and dtype float32.
        """
        # Ensure no NaNs or infinities
        volume = np.nan_to_num(volume, nan=0.0, posinf=None, neginf=None)

        if modality_tag == "CT":
            # Apply CT windowing: (volume → [0,1])
            center, width = self.cta_window
            lower = center - width / 2.0
            upper = center + width / 2.0
            volume = np.clip(volume, lower, upper)
            volume = (volume - lower) / (upper - lower + 1e-6)  # Avoid division by zero
        else:
            # MRI or unknown: z-score + clip + scale
            mean = float(np.mean(volume))
            std = float(np.std(volume))
            if std < 1e-6:
                std = 1e-6  # Prevent division by zero
            volume = (volume - mean) / std
            # Clip outliers
            z_clip = self.mri_z_clip
            volume = np.clip(volume, -z_clip, z_clip)
            # Scale to [0,1]
            volume = (volume + z_clip) / (2.0 * z_clip)

        return volume.astype(np.float32, copy=False)   


    def load_dicom_series(self, series_path: str) -> np.ndarray:
        """
        Load and preprocess a DICOM series into a normalized 3D volume (D, H, W) in [0,1].
        
        Returns:
            np.ndarray: Shape self.target_size, dtype float32, values in [0,1].
        """
        series_id = os.path.basename(series_path.strip("/"))

        # Check memory cache first
        cached = self._cache_get(series_id)
        if cached is not None and cached.shape == self.target_size:
            return cached

        try:
            # Step 1: Load valid DICOMs with PixelData
            dicoms = []
            for root, _, files in os.walk(series_path):
                for f in files:
                    if f.lower().endswith(".dcm"):
                        dcm_path = os.path.join(root, f)
                        try:
                            ds = pydicom.dcmread(dcm_path, force=True, stop_before_pixels=False)
                            if hasattr(ds, "PixelData"):
                                dicoms.append(ds)
                        except Exception as e:
                            print(f"[DICOM Read Error] {series_id}: {e}")
            if not dicoms:
                raise ValueError("No valid DICOM files found")

            # Step 2: Sort slices spatially
            dicoms = self._sort_slices(dicoms)

            # Step 3: Extract metadata
            has_multiframe = any(getattr(ds, "NumberOfFrames", 1) > 1 for ds in dicoms)
            spacing = self._get_spacing(dicoms, has_multiframe=has_multiframe)
            base_h, base_w = self._choose_base_shape(dicoms)
            modality_tag = getattr(dicoms[0], "Modality", "").upper()

            # Step 4: Decode and preprocess slices
            vol_slices = []
            for ds in dicoms:
                arr = ds.pixel_array.astype(np.float32)

                # Handle multi-frame (e.g., 3D volumes in single file)
                                # Handle multi-frame (e.g., 3D volumes in single file)
                if arr.ndim >= 3:
                    n_frames = int(np.prod(arr.shape[:-2]))
                    h, w = arr.shape[-2:]
                    frames = arr.reshape(n_frames, h, w)
                else:
                    frames = arr[np.newaxis, ...]

                for frame in frames:
                    # Handle photometric interpretation
                    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
                        frame = frame.max() - frame

                    # Apply rescale (Hounsfield units for CT)
                    slope = float(getattr(ds, "RescaleSlope", 1.0))
                    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                    frame = frame * slope + intercept

                    # Resize to common base resolution
                    frame = _resize_slice(frame, base_h, base_w)
                    vol_slices.append(frame)

            if not vol_slices:
                raise ValueError("No frames extracted")

            # Stack into (D, H, W)
            volume = np.stack(vol_slices, axis=0).astype(np.float32)

            # Step 5: Modality-specific normalization
            volume = self._normalize_by_modality(volume, modality_tag)

            # Step 6: Isotropic resampling to target spacing
            if self.target_spacing_mm is not None:
                dz, dy, dx = spacing
                z, y, x = volume.shape
                new_shape = (
                    max(1, int(round(z * dz / self.target_spacing_mm))),
                    max(1, int(round(y * dy / self.target_spacing_mm))),
                    max(1, int(round(x * dx / self.target_spacing_mm))),
                )
                scale_factors = (new_shape[0] / z, new_shape[1] / y, new_shape[2] / x)
                volume = _safe_zoom(volume, scale_factors, order=1)

            # Step 7: Final resize to target grid size
            tz, ty, tx = self.target_size
            z, y, x = volume.shape
            if (z, y, x) != (tz, ty, tx):
                scale_factors = (tz / z, ty / y, tx / x)
                volume = _safe_zoom(volume, scale_factors, order=1)

            volume = volume.astype(np.float32, copy=False)

            # Cache and return
            self._cache_put(series_id, volume)
            return volume

        except Exception as e:
            print(f"[Processor] Error processing {series_id}: {e}")
            # Return zero-filled volume as fallback
            fallback = np.zeros(self.target_size, dtype=np.float32)
            self._cache_put(series_id, fallback)
            return fallback   


%%time
# =======================
# 3D CNN (Logits Output)
# =======================
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Simple3DCNN(nn.Module):
    """
    A compact 3D Convolutional Neural Network for medical volume classification.
    
    Architecture:
        - 4x (Conv3D → BatchNorm3D → ReLU → MaxPool3D)
        - Adaptive average pooling to (2,2,2)
        - 3-layer MLP with dropout
    Output: raw logits (no sigmoid/softmax applied)
    
    Input:  (B, 1, D, H, W)  - single-channel 3D volumes
    Output: (B, num_classes)  - logits
    """
    
    def __init__(self, num_classes: int = len(LABEL_COLS)):
        super(Simple3DCNN, self).__init__()
        self.num_classes = num_classes

        # Encoder: 4 downsampling blocks
        self.features = nn.Sequential(
            # Block 1
            nn.Conv3d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Block 2
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Block 3
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            
            # Block 4
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
        )

        # Global pooling + classifier
        self.global_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 2 * 2 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(128, num_classes)  # Logits (no activation)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (B, 1, D, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        # Ensure input is float32
        if x.dtype != torch.float32:
            x = x.float()

        # Feature extraction
        x = self.features(x)  # (B, 128, d, h, w)

        # Global pooling
        x = self.global_pool(x)  # (B, 128, 2, 2, 2)

        # Flatten
        x = torch.flatten(x, 1)  # (B, 128*8)

        # Classification
        x = self.classifier(x)  # (B, num_classes)

        return x   


%%time
# ==============
# Inference API
# ==============
@torch.no_grad()
def predict(series_path: str) -> pl.DataFrame:
    """Server calls this function. Assumes global `model` and `processor` are ready."""
    try:
        # CPU preprocessing
        volume = processor.load_dicom_series(series_path)  # (D,H,W) in [0,1]
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(DEVICE)  # (1,1,D,H,W)

        # Forward pass
        with amp.autocast(device_type='cuda', enabled=USE_AMP):
            logits = model(volume_tensor)               # (1,14)
            probs = torch.sigmoid(logits).cpu().numpy().flatten().tolist()

        result_df = pl.DataFrame(data=[probs], schema=LABEL_COLS, orient='row')

        # Cleanup
        del volume_tensor
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except Exception as e:
        print(f"[Predict] Error: {e}")
        result_df = pl.DataFrame(data=[[0.5] * len(LABEL_COLS)], schema=LABEL_COLS, orient='row')

    # Remove shared temp if present
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    return result_df


%%time
# ==========================
# Run the Evaluation Server
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
        print(f"Submission parquet not found yet: {e}")

