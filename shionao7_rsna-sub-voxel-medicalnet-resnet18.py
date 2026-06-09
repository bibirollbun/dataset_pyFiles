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

# Device / AMP
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
# Filepaths
# =========
MODEL_WEIGHTS_PATH = "/kaggle/input/rsna-train-voxel-medicalnet-resnet18-v1/model_weights.pth"

# ==========================
# Processing / Model Configs
# ==========================
TARGET_SIZE = (64, 64, 64)      # (D,H,W) after final resize
TARGET_SPACING_MM = 1.0         # isotropic spacing in mm
CTA_WINDOW = (300.0, 700.0)     # (center, width) for CT
MRI_Z_CLIP = 3.0                # z-score clip for MR
LRU_CAPACITY = 8                # in-memory LRU capacity

# =====================
# MedicalNet モジュール読み込み
# =====================
import sys
sys.path.append("/kaggle/input/rsna2025-medicalnet-model/MedicalNet")
from models.resnet import resnet10, resnet18, resnet34, resnet50 

# =====================
# 使用するバックボーンの指定
# =====================
BACKBONE = 'resnet18'  # resnet10 / resnet18 / resnet34 / resnet50 から選択


# ==========================
# Utility Resizing Functions
# ==========================
def _safe_zoom(volume: np.ndarray, zoom_factors: Tuple[float, ...], order: int = 1) -> np.ndarray:
    """Wrapper around ndimage.zoom with guards for invalid factors and shapes."""
    volume = np.nan_to_num(volume, copy=False)
    zf = tuple(float(max(1e-6, f)) for f in zoom_factors)
    if len(zf) != volume.ndim:
        if len(zf) > volume.ndim:
            zf = zf[:volume.ndim]
        else:
            zf = (1.0,) * (volume.ndim - len(zf)) + zf
    return ndimage.zoom(volume, zf, order=order)

def _resize_slice(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2D slice to (out_h, out_w) with bilinear order=1."""
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
    """Turn a DICOM series folder into a normalized 3D volume (D,H,W) in [0,1]."""

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

    # ---- LRU (memory only) ----
    def _cache_put(self, key: str, vol: np.ndarray):
        self.memory_cache[key] = vol
        self.memory_cache.move_to_end(key)
        if len(self.memory_cache) > self.lru_capacity:
            self.memory_cache.popitem(last=False)

    def _cache_get(self, key: str):
        if key in self.memory_cache:
            vol = self.memory_cache[key]
            self.memory_cache.move_to_end(key)
            return vol
        return None

    # ---- Public API ----
    def load_dicom_series(self, series_path: str) -> np.ndarray:
        """Return (D,H,W) float32 in [0,1]."""
        series_id = os.path.basename(series_path)

        # Memory cache
        m = self._cache_get(series_id)
        if m is not None and isinstance(m, np.ndarray) and m.shape == self.target_size:
            return m

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
                        except Exception as e:
                            print(f"[DICOM read] {e}")
                            continue
            if not dicoms:
                raise ValueError(f"No valid DICOM files in {series_path}")

            # Sort slices by plane position along normal vector
            dicoms = self._sort_slices(dicoms)

            # Detect multiframe for spacing logic
            has_multiframe = any(getattr(ds, "NumberOfFrames", 1) > 1 for ds in dicoms)

            # Pixel spacing (dy,dx) and slice interval (dz)
            spacing = self._get_spacing(dicoms, has_multiframe=has_multiframe)

            # Choose base HxW without decoding entire stack twice
            base_h, base_w = self._choose_base_shape(dicoms)

            # Get modality tag
            modality_tag = (getattr(dicoms[0], "Modality", "") or "").upper()

            # Decode to slices (note: no per-frame reordering beyond dataset sorting)
            vol_slices = []
            for ds in dicoms:
                arr = ds.pixel_array
                if arr.ndim >= 3:
                    h, w = arr.shape[-2], arr.shape[-1]
                    n = int(np.prod(arr.shape[:-2]))
                    frames = arr.reshape(n, h, w)
                else:
                    frames = arr[np.newaxis, ...]

                for sl in frames:
                    sl = sl.astype(np.float32)

                    # Handle MONOCHROME1
                    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                        sl = sl.max() - sl

                    # Rescale
                    slope = float(getattr(ds, "RescaleSlope", 1.0))
                    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                    sl = sl * slope + intercept

                    # Resize 2D slice to base HxW
                    sl = _resize_slice(sl, base_h, base_w)
                    vol_slices.append(sl)

            if len(vol_slices) == 0:
                raise ValueError("No valid slices extracted")

            # Stack to (D,H,W)
            volume = np.stack(vol_slices, axis=0).astype(np.float32)

            # Modality-wise normalization to [0,1]
            volume = self._normalize_by_modality(volume, modality_tag)

            # Isotropic resample in mm
            if self.target_spacing_mm is not None:
                dz, dy, dx = spacing
                z, y, x = volume.shape
                newD = max(1, int(round(z * dz / self.target_spacing_mm)))
                newH = max(1, int(round(y * dy / self.target_spacing_mm)))
                newW = max(1, int(round(x * dx / self.target_spacing_mm)))
                volume = _safe_zoom(volume, (newD / z, newH / y, newW / x), order=1)

            # Final resize to target grid
            tz, ty, tx = self.target_size
            z, y, x = volume.shape
            volume = _safe_zoom(volume, (tz / z, ty / y, tx / x), order=1).astype(np.float32)

            # Put into cache and return
            self._cache_put(series_id, volume)
            return volume

        except Exception as e:
            print(f"[Processor] Error: {e}")
            vol = np.zeros(self.target_size, dtype=np.float32)
            self._cache_put(series_id, vol)
            return vol

    # ---- Helpers ----
    def _sort_slices(self, ds_list: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
        """Sort by dot(ImagePositionPatient, normal) or InstanceNumber."""
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
        """Return (dz, dy, dx) in mm with robust fallbacks."""
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
            dz = float(np.median(zs)) if zs else float(getattr(ds_sorted[0], "SliceThickness", 1.0))

        dz = dz if (dz > 0 and np.isfinite(dz)) else 1.0
        dy = dy if (dy > 0 and np.isfinite(dy)) else 1.0
        dx = dx if (dx > 0 and np.isfinite(dx)) else 1.0
        return (dz, dy, dx)

    def _choose_base_shape(self, ds_list: List[pydicom.dataset.FileDataset]) -> Tuple[int, int]:
        """Pick the most frequent (Rows, Columns) as base 2D shape."""
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
        """CT: window to [0,1]; MR: z-score -> clip -> [0,1]."""
        volume = np.nan_to_num(volume, copy=False)
        if modality_tag == "CT":
            c, w = self.cta_window
            lo, hi = c - w / 2.0, c + w / 2.0
            v = np.clip(volume, lo, hi)
            v = (v - lo) / (hi - lo + 1e-6)
            return v.astype(np.float32, copy=False)
        else:
            mean = float(volume.mean())
            std = float(volume.std() + 1e-6)
            v = (volume - mean) / std
            zc = float(self.mri_z_clip)
            v = np.clip(v, -zc, zc)
            v = (v + zc) / (2.0 * zc)
            return v.astype(np.float32, copy=False)


# =====================
# MedicalNet ResNet 推論用分類モデル
# =====================
class MedicalNet3DClassifier(nn.Module):
    def __init__(self, backbone_type='resnet18', num_classes=len(LABEL_COLS),
                 weight_path=None, device='cuda'):
        super().__init__()

        # ResNet の種類選択
        backbone_dict = {
            'resnet10': resnet10,
            'resnet18': resnet18,
            'resnet34': resnet34,
            'resnet50': resnet50
        }
        if backbone_type not in backbone_dict:
            raise ValueError(f"Invalid backbone_type: {backbone_type}")
        
        backbone_fn = backbone_dict[backbone_type]

        # backbone の生成（セグヘッド無効）
        self.backbone = backbone_fn(
            sample_input_W=TARGET_SIZE[1],
            sample_input_H=TARGET_SIZE[0],
            sample_input_D=MRI_Z_CLIP,
            shortcut_type='B',
            no_cuda=False,
            num_seg_classes=0
        )

        # 最終チャネル数（ResNet の構造に依存）
        channel_dict = {'resnet10': 512, 'resnet18': 512, 'resnet34': 512, 'resnet50': 2048}
        in_features = channel_dict[backbone_type]

        # 分類ヘッド
        self.classifier = nn.Linear(in_features, num_classes)
        state_dict = torch.load(weight_path, map_location=device)
        self.load_state_dict(state_dict)
        self.device = device
        self.to(device)
        self.eval()  # 推論用に eval モード

    def forward(self, x):
        # backbone の forward を使い conv_seg は無視
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)

        # Global Average Pooling
        x = x.mean(dim=[2, 3, 4])
        x = self.classifier(x)
        return x


# Initialize processor (in-memory LRU only)
processor = DICOMProcessor(
    target_size=TARGET_SIZE,
    target_spacing_mm=TARGET_SPACING_MM,
    cta_window=CTA_WINDOW,
    mri_z_clip=MRI_Z_CLIP,
    lru_capacity=LRU_CAPACITY,
)

# =====================
# Create MedicalNet3DClassifier
# =====================
model = MedicalNet3DClassifier(
    backbone_type=BACKBONE,
    num_classes=len(LABEL_COLS),
    weight_path=MODEL_WEIGHTS_PATH,
    device=DEVICE
)
model.eval()

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




