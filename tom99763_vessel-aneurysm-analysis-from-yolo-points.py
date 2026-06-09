# Standard library
import gc
import json
import os
import queue
import shutil
import sys
import threading
import time
import warnings
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import ast

# Third-party: general
import cv2
import numpy as np
import pandas as pd
import polars as pl
import pydicom
from scipy import ndimage
from skimage.filters import frangi
import matplotlib.pyplot as plt

# Third-party: ML/DL
import cupy as cp
from cupyx.scipy.ndimage import zoom
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
sys.path.insert(0, "/kaggle/input/ultralytcs-timm-rsna/ultralytics-timm")
# YOLO
from ultralytics import YOLO

# Transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Competition API
import kaggle_evaluation.rsna_inference_server

# Warnings config
warnings.filterwarnings("ignore")

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Optimization settings
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ====================================================
# Competition constants
# ====================================================
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

# YOLO label mappings
YOLO_LABELS_TO_IDX = {
    'Anterior Communicating Artery': 0,
    'Basilar Tip': 1,
    'Left Anterior Cerebral Artery': 2,
    'Left Infraclinoid Internal Carotid Artery': 3,
    'Left Middle Cerebral Artery': 4,
    'Left Posterior Communicating Artery': 5,
    'Left Supraclinoid Internal Carotid Artery': 6,
    'Other Posterior Circulation': 7,
    'Right Anterior Cerebral Artery': 8,
    'Right Infraclinoid Internal Carotid Artery': 9,
    'Right Middle Cerebral Artery': 10,
    'Right Posterior Communicating Artery': 11,
    'Right Supraclinoid Internal Carotid Artery': 12
}

YOLO_LABELS = sorted(list(YOLO_LABELS_TO_IDX.keys()))


EFF_LABELS_TO_IDX = {
    'Aneurysm Present': 0,
    'Anterior Communicating Artery': 1,
    'Basilar Tip': 2,
    'Left Anterior Cerebral Artery': 3,
    'Left Infraclinoid Internal Carotid Artery': 4,
    'Left Middle Cerebral Artery': 5,
    'Left Posterior Communicating Artery': 6,
    'Left Supraclinoid Internal Carotid Artery': 7,
    'Other Posterior Circulation': 8,
    'Right Anterior Cerebral Artery': 9,
    'Right Infraclinoid Internal Carotid Artery': 10,
    'Right Middle Cerebral Artery': 11,
    'Right Posterior Communicating Artery': 12,
    'Right Supraclinoid Internal Carotid Artery': 13
}

EFF_LABELS = sorted(list(EFF_LABELS_TO_IDX.keys()))


# ====================================================
# YOLO Configuration
# ====================================================
IMG_SIZE = 512
BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "32"))
MAX_WORKERS = 4

YOLO_MODEL_CONFIGS = [
    {
        "path": "/kaggle/input/rsna-yolo-models/cv_y11m_more_negatives_fold02/weights/best.pt",
        "fold": "0",
        "weight": 1.0,
        "name": "YOLOv11m_fold0"
    },
    {
        "path": "/kaggle/input/rsna-yolo-models/cv_y11m_more_negatives_fold3/weights/best.pt",
        "fold": "3",
        "weight": 1.0,
        "name": "YOLOv11m_fold1"
    }
]


def load_yolo_models():
    """Load all YOLO models"""
    models = []
    for config in YOLO_MODEL_CONFIGS:
        model = YOLO(config["path"])
        model.to(device)
        
        model_dict = {
            "model": model,
            "weight": config["weight"],
            "name": config["name"],
            "fold": config["fold"]
        }
        models.append(model_dict)
    return models


YOLO_MODELS = load_yolo_models()


def read_dicom_frames_hu(path: Path) -> List[Tuple[float, np.ndarray]]:
    """Read DICOM file and return list of (slice_position, HU frame)"""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))

    # Compute slice location using orientation + position
    try:
        orientation = np.array(ds.ImageOrientationPatient).reshape(2, 3)
        row_cos, col_cos = orientation
        normal = np.cross(row_cos, col_cos)  # slice normal vector
        position = np.array(ds.ImagePositionPatient)
        slice_loc = float(np.dot(position, normal))  # projection along normal
    except Exception:
        # Fallback: SliceLocation / InstanceNumber
        slice_loc = float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", 0.0)))

    frames: List[Tuple[float, np.ndarray]] = []

    if pix.ndim == 2:
        img = pix.astype(np.float32)
        frames.append((slice_loc, img * slope + intercept))
    elif pix.ndim == 3:
        # RGB or multi-frame
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            try:
                gray = cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            except Exception:
                gray = pix[..., 0].astype(np.float32)
            frames.append((slice_loc, gray * slope + intercept))
        else:
            for i in range(pix.shape[0]):
                frm = pix[i].astype(np.float32)
                # tiny offset ensures consistent ordering for multi-frame
                frames.append((slice_loc + i * 1e-3, frm * slope + intercept))
    return frames


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """Min-max normalization to 0-255 with optional flipping"""
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        norm = np.zeros_like(img, dtype=np.uint8)
    else:
        norm = (img - mn) / (mx - mn)
        norm = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return norm


def process_dicom_file(dcm_path: Path) -> List[Tuple[float, np.ndarray]]:
    """Process single DICOM file -> list of (slice_loc, image) tuples"""
    try:
        frames = read_dicom_frames_hu(dcm_path)
        processed_slices = []
        for loc, f in frames:
            img_u8 = min_max_normalize(f)
            if img_u8.ndim == 2:
                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            processed_slices.append((loc, img_u8))
        return processed_slices
    except Exception as e:
        print(f"Failed processing {dcm_path.name}: {e}")
        return []


def collect_series_slices(series_dir: Path) -> List[Path]:
    """Collect all DICOM files in a series directory (recursively)."""
    dcm_paths: List[Path] = []
    try:
        for root, _, files in os.walk(series_dir):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dcm_paths.append(Path(root) / f)
    except Exception as e:
        print(f"Failed to walk series dir {series_dir}: {e}")
    return dcm_paths


def slice_sort_key(path: Path) -> float:
    """Compute a robust slice sort key (orientation + position) for a single DICOM file"""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        orientation = np.array(ds.ImageOrientationPatient).reshape(2, 3)
        row_cos, col_cos = orientation
        normal = np.cross(row_cos, col_cos)
        position = np.array(ds.ImagePositionPatient)
        return float(np.dot(position, normal))
    except Exception:
        # fallback
        try:
            return float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", 0.0)))
        except:
            return 0.0

def process_dicom_for_yolo(series_path):
    series_path = Path(series_path)
    dicom_files = collect_series_slices(series_path)
    
    # Sort DICOM files by orientation+position before processing
    dicom_files.sort(key=slice_sort_key)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_dicom_file, dicom_files))
    
    # Flatten into (loc, img)
    all_slices_with_loc = [item for sublist in results for item in sublist]
    
    # Already sorted by dicom_files order, but double-check (safe)
    all_slices_with_loc.sort(key=lambda x: x[0])
    
    # Extract just the images
    all_slices = [img for _, img in all_slices_with_loc]
    
    # Now dicom_files matches the sorted slices
    dcm_list = [f.stem for f in dicom_files]
    return all_slices


def nms_3d_points(points, iou_thresh=2.0):
    """
    Simple 3D NMS for point detections.
    points: np.array of shape [N, 5] or [N,6], columns=[z,y,x,prob,class,...]
    iou_thresh: minimum distance (in voxels) to suppress a point
    Returns: indices of points to keep
    """
    if len(points) == 0:
        return []

    keep = []
    pts = points[:, :3]  # z,y,x
    scores = points[:, 3]
    order = scores.argsort()[::-1]

    suppressed = np.zeros(len(points), dtype=bool)

    for idx in order:
        if suppressed[idx]:
            continue
        keep.append(idx)
        dists = np.linalg.norm(pts - pts[idx], axis=1)
        suppressed[dists < iou_thresh] = True
        suppressed[idx] = False  # keep current
    return keep


@torch.no_grad()
def predict_yolo_ensemble(slices, conf_yolo, iou_thresh=2.0, k = 3):
    if not slices:
        return 0.1, np.ones(len(YOLO_LABELS)) * 0.1

    location_preds = {f'MODEL{i}': [] for i in range(len(YOLO_MODELS))}

    for model_idx, model_dict in enumerate(YOLO_MODELS):
        model = model_dict["model"]
        weight = model_dict["weight"]

        for i in range(0, len(slices), BATCH_SIZE):
            batch_slices = slices[i:i+BATCH_SIZE]
            z_idxes = [i + batch_idx for batch_idx in range(len(batch_slices))]
            results = model.predict(
                batch_slices,
                verbose=False,
                batch=len(batch_slices),
                device="cuda:0",
                conf=conf_yolo
            )

            for z_idx, r in enumerate(results):
                if r is None or r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
                    continue
                confs = r.boxes.conf.cpu().numpy()
                clses = r.boxes.cls.cpu().numpy()
                xyxy = r.boxes.xyxy.cpu().numpy()

                for j in range(len(confs)):
                    x1, y1, x2, y2 = xyxy[j]
                    x_center = (x1 + x2)/2
                    y_center = (y1 + y2)/2
                    point = np.array([z_idxes[z_idx], y_center, x_center, confs[j], clses[j], model_idx])
                    location_preds[f'MODEL{model_idx}'].append(point)

    # Apply NMS per model and per class
    final_preds = {f'MODEL{i}': [] for i in range(len(YOLO_MODELS))}
    for model_key, points in location_preds.items():
        points = np.array(points)
        if points.shape[0] == 0:
            continue
        for cls in np.unique(points[:,4]):
            cls_points = points[points[:,4]==cls]
            keep_idx = nms_3d_points(cls_points, iou_thresh=iou_thresh)
            final_preds[model_key].extend(cls_points[keep_idx])
        model_preds = torch.tensor(final_preds[model_key])
        values, indices = model_preds[:, -3].topk(k)   # get top-k values & indices
        final_preds[model_key] = model_preds[indices].numpy()
    return final_preds


def load_loc_labels(root: Path) -> pd.DataFrame:
    label_df = pd.read_csv(root / "train_localizers.csv")
    if "x" not in label_df.columns or "y" not in label_df.columns:
        label_df["x"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["x"])  # type: ignore[arg-type]
        label_df["y"] = label_df["coordinates"].map(lambda s: ast.literal_eval(s)["y"])  # type: ignore[arg-type]
    # Standardize dtypes
    label_df["SeriesInstanceUID"] = label_df["SeriesInstanceUID"].astype(str)
    label_df["SOPInstanceUID"] = label_df["SOPInstanceUID"].astype(str)
    return label_df


data_path = Path('/kaggle/input/rsna-intracranial-aneurysm-detection')
df = pd.read_csv(data_path/'train.csv')
df_loc = load_loc_labels(data_path)


gt_loc = df_loc[['x', 'y']]


uid = df_loc.iloc[32].SeriesInstanceUID	
row  = df[df.SeriesInstanceUID==uid]
row_loc = df_loc[df_loc.SeriesInstanceUID==uid]
gt_loc = row_loc[['x', 'y']].values.astype('int32')
series_path = data_path/f'series/{uid}'
row


all_slices = process_dicom_for_yolo(series_path)


location_preds = predict_yolo_ensemble(all_slices, conf_yolo = 0.01, iou_thresh=5.0, k = 3)


location_preds['MODEL0'].shape, location_preds['MODEL1'].shape


def load_dicom_series(series_dir: Path):
    """
    Load a DICOM series into a 3D HU volume with affine matrix.

    Args:
        series_dir (Path): directory containing DICOM files

    Returns:
        volume (np.ndarray): 3D array (Z, Y, X) in HU
        affine (np.ndarray): 4x4 voxel-to-world affine
    """
    # Collect DICOMs
    dcm_paths = [Path(series_dir) / f for f in os.listdir(series_dir) if f.lower().endswith(".dcm")]
    if not dcm_paths:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")

    slices = [pydicom.dcmread(str(p), force=True) for p in dcm_paths]

    # --- Orientation ---
    orientation = slices[0].get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
    orientation = np.array(orientation, dtype=np.float32).reshape(2, 3)
    row_cos, col_cos = orientation
    normal = np.cross(row_cos, col_cos)

    # --- Sorting ---
    if hasattr(slices[0], "ImagePositionPatient"):
        slices.sort(key=lambda ds: np.dot(ds.get("ImagePositionPatient", [0, 0, 0]), normal))
    else:
        # fallback: sort by InstanceNumber if available
        slices.sort(key=lambda ds: getattr(ds, "InstanceNumber", 0))

    # --- HU scaling ---
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    volume = np.stack([ds.pixel_array for ds in slices]).astype(np.float32)
    volume = volume * slope + intercept

    if volume.ndim == 4 and volume.shape[0] == 1:
        volume = volume[0]  # remove extra batch dimension

    slice_names = [Path(ds.filename).name for ds in slices]
    return volume, slice_names


def normalize_vol(vol):
    p2, p98 = np.percentile(vol, (2, 98))
    mask = (vol >= p2) & (vol <= p98)
    mean = np.mean(vol[mask])
    std = np.std(vol[mask]) + 1e-6

    vol = (vol - mean) / std
    vol = np.clip((vol - vol.min()) / (vol.max() - vol.min() + 1e-6), 0, 1)
    return vol


class AneurysmVolumeProcessor3Planes:
    def __init__(self, N=96, K_axial=5, K_sagittal=15, K_coronal=15,
                 Nr=64, Ntheta=128, augment=False, device='cpu', n_workers=4):
        self.N = N
        self.K_axial = K_axial
        self.K_sagittal = K_sagittal
        self.K_coronal = K_coronal
        self.Nr = Nr
        self.Ntheta = Ntheta
        self.augment = augment
        self.device = device
        self.n_workers = n_workers

    def __call__(self, volume, yolo_points):
        def process_point(point):
            x, y, z = map(int, map(round, point))
            planes = {}

            # --- Axial ---
            K = self.K_axial
            z_min, z_max = max(0, z-K//2), min(volume.shape[0], z+K//2+1)
            y_min, y_max = max(0, y-self.N//2), min(volume.shape[1], y+self.N//2)
            x_min, x_max = max(0, x-self.N//2), min(volume.shape[2], x+self.N//2)
            axial_patch = volume[z_min:z_max, y_min:y_max, x_min:x_max].copy()
            axial_patch = self._pad_patch(axial_patch, (K, self.N, self.N))
            planes['axial'] = axial_patch

            # --- Sagittal ---
            K = self.K_sagittal
            x_min, x_max = max(0, x-K//2), min(volume.shape[2], x+K//2+1)
            z_min, z_max = max(0, z-self.N//2), min(volume.shape[0], z+self.N//2)
            y_min, y_max = max(0, y-self.N//2), min(volume.shape[1], y+self.N//2)
            sag_patch = volume[z_min:z_max, y_min:y_max, x_min:x_max].copy()
            sag_patch = np.transpose(sag_patch, (2, 0, 1))  # (x, z, y)
            sag_patch = self._pad_patch(sag_patch, (K, self.N, self.N))
            planes['sagittal'] = sag_patch

            # --- Coronal ---
            K = self.K_coronal
            y_min, y_max = max(0, y-K//2), min(volume.shape[1], y+K//2+1)
            z_min, z_max = max(0, z-self.N//2), min(volume.shape[0], z+self.N//2)
            x_min, x_max = max(0, x-self.N//2), min(volume.shape[2], x+self.N//2)
            cor_patch = volume[z_min:z_max, y_min:y_max, x_min:x_max].copy()
            cor_patch = np.transpose(cor_patch, (1, 0, 2))  # (y, z, x)
            cor_patch = self._pad_patch(cor_patch, (K, self.N, self.N))
            planes['coronal'] = cor_patch

            # --- Cartesian & Log-Polar features ---
            cartesian_channels, logpolar_channels = [], []
            for plane_name, patch in planes.items():
                center_slice = patch[patch.shape[0] // 2]
                mip = np.max(patch, axis=0)

                # stack [center_slice, mip] only
                cartesian_channels.append(np.stack([center_slice, mip], axis=0))

                cx, cy = self.N / 2, self.N / 2
                logpolar_channels.append(np.stack([
                    self._logpolar(center_slice, cx, cy),
                    self._logpolar(mip, cx, cy)
                ], axis=0))

            return {
                'cartesian': torch.from_numpy(np.stack(cartesian_channels, axis=0)).float(),
                'logpolar': torch.from_numpy(np.stack(logpolar_channels, axis=0)).float(),
                'axial': torch.from_numpy(planes['axial']).float(),
                'sagittal': torch.from_numpy(planes['sagittal']).float(),
                'coronal': torch.from_numpy(planes['coronal']).float()
            }

        # ✅ order preserved
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            outputs = list(executor.map(process_point, yolo_points))

        return outputs
        
    def _pad_patch(self, patch, shape):
        K, N, _ = shape
        padded = np.zeros(shape, dtype=patch.dtype)
        dz, dy, dx = patch.shape
        padded[:dz, :dy, :dx] = patch
        return padded

    def _logpolar(self, img, cx, cy):
        img = img.astype(np.float32)
        max_radius = np.sqrt(
            max(cx, img.shape[1]-cx)**2 + 
            max(cy, img.shape[0]-cy)**2
        )
        logpolar_img = cv2.logPolar(
            img, center=(cx, cy),
            M=self.Nr / np.log(max_radius + 1e-6),
            flags=cv2.INTER_LINEAR + cv2.WARP_FILL_OUTLIERS
        )
        return cv2.resize(logpolar_img, (self.Ntheta, self.Nr))


vol, slice_names = load_dicom_series(series_path)


gt_loc_z = np.array([[slice_names.index(f'{row_loc.SOPInstanceUID.values[i]}.dcm')] for i in range(row_loc.SOPInstanceUID.values.shape[0])])
gt_loc_xy = row_loc[['x', 'y']].values.astype('int32')
gt_loc_3d = np.concatenate([gt_loc_xy, gt_loc_z], axis=-1)


gt_loc_3d


vol_norm = normalize_vol(vol)


vol_norm.shape


N=64


proc = AneurysmVolumeProcessor3Planes(N=N, K_axial=15, K_sagittal=15, K_coronal=15,
                 Nr=N, Ntheta=N, augment=False, device='cpu')


yolo_points_0 = location_preds['MODEL0'][:, [2, 1, 0]].astype('int32') #x, y, z
yolo_points_1 = location_preds['MODEL1'][:, [2, 1, 0]].astype('int32')


output_0 = proc(vol_norm, yolo_points_0)
output_1 = proc(vol_norm, yolo_points_1)


def show_features_with_center(outputs, yolo_points, gt_points=None, N=96):
    """
    Visualize Cartesian and Log-Polar channels with YOLO and GT markers.
    Handles cropped patches (local coordinate transform).
    
    outputs: list of dicts with keys:
        'cartesian' [3 planes, 3 channels, N, N], 
        'logpolar' [3 planes, 3 channels, Nr, Ntheta]
    yolo_points: list of YOLO (x, y, z) in full volume coordinates
    gt_points: optional list of GT (x, y, z) in full volume coordinates
    N: patch size
    """
    plane_titles = ['Axial', 'Sagittal', 'Coronal']
    channel_titles = ['Center Slice', 'MIP', 'Vesselness MIP']

    for idx, patch in enumerate(outputs):
        x, y, z = map(int, map(round, yolo_points[idx]))
        print(f'Patch {idx} at YOLO point {yolo_points[idx]}')

        cartesian = patch['cartesian'].numpy()   # [3, 3, N, N]
        logpolar = patch['logpolar'].numpy()     # [3, 3, Nr, Ntheta]

        N = cartesian.shape[-1]
        Nr, Ntheta = logpolar.shape[-2:]

        cx_cart, cy_cart = N//2, N//2
        cx_log, cy_log = Ntheta//2, Nr//2

        # --- compute GT local coords (Axial only for now) ---
        gx_local, gy_local = None, None
        if gt_points is not None:
            gx, gy = gt_points[:, 0], gt_points[:, 1]
            gx_local = gx - (x - N//2)
            gy_local = gy - (y - N//2)

        for p_idx, plane in enumerate(plane_titles):
            plt.figure(figsize=(12, 6))
            plt.suptitle(f'{plane} Plane - Patch {idx}', fontsize=16)

            # Cartesian channels
            for c_idx, ch in enumerate(channel_titles):
                plt.subplot(2, 3, c_idx+1)
                plt.imshow(cartesian[p_idx, c_idx], cmap='gray')

                # YOLO predicted center
                plt.scatter(cx_cart, cy_cart, s=40, c='blue', marker='x', label='YOLO center')

                # Ground truth center (only for Axial plane, if inside crop)
                if plane == 'Axial' and gx_local is not None:
                    if 0 <= gx_local < N and 0 <= gy_local < N:
                        plt.scatter(gx_local, gy_local, s=40, c='red', marker='o', label='GT center')

                plt.title(f'{ch} - Cartesian')
                plt.axis('off')

                if c_idx == 0 and plane == 'Axial' and gx_local is not None:
                    plt.legend()

            # Log-polar channels
            for c_idx, ch in enumerate(channel_titles):
                plt.subplot(2, 3, c_idx+4)
                plt.imshow(logpolar[p_idx, c_idx], cmap='gray', origin='lower')
                plt.scatter(cx_log, cy_log, s=30, c='blue', marker='x')
                plt.title(f'{ch} - Log-Polar')
                plt.xlabel('Theta / Azimuth')
                plt.ylabel('Radius (log)')
                plt.axis('on')

            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()


output_0[0]['cartesian'].shape


for name in ['axial', 'sagittal', 'coronal']:
    print(name)
    print(output_0[0][name].shape)
    print(output_1[0][name].shape)


# show_features_with_center(output_0, yolo_points_0, gt_loc, N)


# show_features_with_center(output_1, yolo_points_1, gt_loc, N)


import pywt
import plotly.graph_objects as go


def get_wavelet_and_gt_loc(points, output, idx, view):
    gt_patch = np.array([
        gt_loc_3d[0, 2] - points[idx][2],  # depth (z)
        gt_loc_3d[0, 1] - points[idx][1],  # row   (y)
        gt_loc_3d[0, 0] - points[idx][0]   # col   (x)
    ])/2
    
    x =  output[0][view]
    x_np = x.numpy()
    
    # 3D DWT
    coeffs = pywt.dwtn(x_np, wavelet='bior3.5', axes=(0,1,2))  # 3D Haar
    normalized_coeffs = {}
    for band, band_data in coeffs.items():
        normalized_coeffs[band] = (band_data - np.mean(band_data)) / (np.std(band_data) + 1e-10)
    return normalized_coeffs, gt_patch


coeffs_ax, gt_patch_ax = get_wavelet_and_gt_loc(yolo_points_0, output_0, 0, 'axial')

coeffs_sag, gt_patch_sag = get_wavelet_and_gt_loc(yolo_points_0, output_0, 0, 'sagittal')

coeffs_cor, gt_patch_cor = get_wavelet_and_gt_loc(yolo_points_0, output_0, 0, 'coronal')


aneurysm_map = np.sqrt(
    sum(np.square(coeffs_ax[band]) 
        for band in coeffs_ax if band != 'aaa')  # 'aaa' = low-freq band
)


fig = go.Figure(data=go.Volume(
    x=np.arange(aneurysm_map.shape[0]).repeat(aneurysm_map.shape[1]*aneurysm_map.shape[2]),
    y=np.tile(np.arange(aneurysm_map.shape[1]).repeat(aneurysm_map.shape[2]), aneurysm_map.shape[0]),
    z=np.tile(np.arange(aneurysm_map.shape[2]), aneurysm_map.shape[0]*aneurysm_map.shape[1]),
    value=aneurysm_map.flatten(),
    isomin=np.percentile(aneurysm_map, 95),  # show top 5% anomalies
    isomax=aneurysm_map.max(),
    opacity=0.1,  # lower for more transparency
    surface_count=20,  # number of layers
    colorscale="Hot"
))

fig.update_layout(scene=dict(
    xaxis=dict(visible=False),
    yaxis=dict(visible=False),
    zaxis=dict(visible=False)
))

fig.show()


def visualize_wavelet3d_all(coeffs, gt_point=None, mode='slices'):
    bands = list(coeffs.keys())

    if mode == 'slices':
        n_bands = len(bands)
        fig, axes = plt.subplots(n_bands, 3, figsize=(10, 3*n_bands))

        if n_bands == 1:
            axes = np.expand_dims(axes, 0)

        for i, band in enumerate(bands):
            vol = np.array(coeffs[band])
            d, h, w = vol.shape
            dz, dy, dx = d//2, h//2, w//2

            # Axial (Z fixed)
            axes[i, 0].imshow(vol[dz, :, :], cmap='gray')
            axes[i, 0].set_title(f"{band} - Axial")
            if gt_point is not None:
                axes[i, 0].scatter(gt_point[2] + dx, gt_point[1] + dy, c='r', s=40, marker='x')

            # Coronal (Y fixed)
            axes[i, 1].imshow(vol[:, dy, :], cmap='gray')
            axes[i, 1].set_title(f"{band} - Coronal")
            if gt_point is not None:
                axes[i, 1].scatter(gt_point[2] + dx, gt_point[0] + dz, c='r', s=40, marker='x')

            # Sagittal (X fixed)
            axes[i, 2].imshow(vol[:, :, dx], cmap='gray')
            axes[i, 2].set_title(f"{band} - Sagittal")
            if gt_point is not None:
                axes[i, 2].scatter(gt_point[1] + dy, gt_point[0] + dz, c='r', s=40, marker='x')

        plt.tight_layout()
        plt.show()

    elif mode == 'volume':
        for i, band in enumerate(bands):
            vol = np.array(coeffs[band])
            d, h, w = vol.shape

            fig = go.Figure()

            # Volume rendering
            fig.add_trace(go.Volume(
                x=np.arange(d).repeat(h*w),
                y=np.tile(np.arange(h).repeat(w), d),
                z=np.tile(np.arange(w), d*h),
                value=vol.flatten(),
                opacity=0.1,
                surface_count=15,
                colorscale='jet',
            ))

            # Add GT point as red sphere with offsets
            if gt_point is not None:
                dz, dy, dx = d//2, h//2, w//2
                fig.add_trace(go.Scatter3d(
                    x=[gt_point[0] + dz],
                    y=[gt_point[1] + dy],
                    z=[gt_point[2] + dx],
                    mode='markers',
                    marker=dict(size=10, color='red'),
                    name='GT Point'
                ))

            fig.update_layout(
                title=f"3D Wavelet Volume - {band}",
                scene=dict(
                    xaxis_title="Depth",
                    yaxis_title="Height",
                    zaxis_title="Width"
                )
            )
            fig.show()
    else:
        raise ValueError("mode must be 'slices' or 'volume'")



visualize_wavelet3d_all(coeffs_ax, gt_patch_ax, mode='volume')


coeffs_ax['aaa'].shape


visualize_wavelet3d_all(coeffs_ax, gt_patch_ax, mode='slices')


visualize_wavelet3d_all(coeffs_ax, gt_patch_ax, mode='volume')


visualize_wavelet3d_all(coeffs_sag, gt_patch_sag, mode='slices')


visualize_wavelet3d_all(coeffs_sag, gt_patch_sag, mode='volume')


visualize_wavelet3d_all(coeffs_cor, gt_patch_sag, mode='slices')


visualize_wavelet3d_all(coeffs_cor, gt_patch_sag, mode='volume')




