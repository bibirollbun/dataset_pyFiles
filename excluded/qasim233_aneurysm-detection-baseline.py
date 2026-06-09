# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd

df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
df.head()


df.Modality.value_counts()


df["Aneurysm Present"].value_counts()


df2 = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")
df2.head()


df.groupby('Modality')['Aneurysm Present'].value_counts().unstack(fill_value=0)


df['SeriesInstanceUID'].nunique()


import os
import random
import json
import ast
import copy
import time
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

# --------------------------
# CONFIG - edit paths here
# --------------------------
CONFIG = {
    "data_root": "/kaggle/input/rsna-intracranial-aneurysm-detection",  # root folder containing 'series', 'train.csv', 'train_localizers.csv', 'segmentations' if present
    "series_folder": "series",
    "train_csv": "train.csv",
    "localizers_csv": "train_localizers.csv",
    "cache_dir": "./cache_images",
    "manifest_path": "./selected_series_manifest.csv",
    "modalities_to_process": None,  # None => process all modalities found; or e.g. ['MRA','CTA']
    "per_modality_total": 300,
    "per_modality_pos": 150,
    "per_modality_neg": 150,
    "random_seed": 42,
    "target_image_size": 224,
    "batch_size": 16,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "model_name": "resnet50",  # options: resnet18, resnet34, resnet50
    "head_lr": 1e-4,
    "ft_lr": 1e-6,
    "weight_decay": 1e-4,
    "epochs_head": 5,
    "epochs_finetune": 15,
    "patience": 5,  # early stopping patience
    "save_dir": "./checkpoints",
    "augment_fill_shortfall": True,  # automatically plan augmentation to reach required counts
    "use_mip_for_modality": ["MRA", "CTA"],  # default use MIP for angiographic modalities
    "windowing": {  # default windowing strategy: use DICOM WindowCenter/Width when available; otherwise percentile
        "use_dicom_window": True,
        "percentile_clip": (1, 99)
    },
    "cache_npy": True,  # store processed 2D images as .npy in cache_dir
    "verbose": True
}

os.makedirs(CONFIG["cache_dir"], exist_ok=True)
os.makedirs(CONFIG["save_dir"], exist_ok=True)

# --------------------------
# Utilities & reproducibility
# --------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True  # deterministic, but can slow things
    # torch.backends.cudnn.benchmark = False

set_seed(CONFIG["random_seed"])

def log(msg: str):
    if CONFIG["verbose"]:
        print(msg)

# --------------------------
# 1) Balanced selection manifest
# --------------------------
def create_balanced_manifest(train_csv_path: str,
                             manifest_path: str,
                             modalities_to_process: Optional[List[str]] = None,
                             pos_target: int = 150,
                             neg_target: int = 150,
                             augment_fill_shortfall: bool = True) -> pd.DataFrame:
    """
    Create a CSV manifest with balanced series per modality.

    Output columns:
      SeriesInstanceUID, PatientID (if present), Modality, Aneurysm Present (0/1),
      augment_copies (int): how many augmented copies to generate (for training only)
    """
    df = pd.read_csv(train_csv_path)
    # Normalize modality strings
    df['Modality'] = df['Modality'].astype(str).str.upper().str.strip()
    if modalities_to_process:
        modalities = modalities_to_process
    else:
        modalities = sorted(df['Modality'].unique().tolist())
    manifest_rows = []
    for modality in modalities:
        group = df[df['Modality'] == modality]
        pos = group[group['Aneurysm Present'] == 1].copy()
        neg = group[group['Aneurysm Present'] == 0].copy()
        # sample without replacement
        chosen_pos = pos.sample(min(len(pos), pos_target), random_state=CONFIG["random_seed"])
        chosen_neg = neg.sample(min(len(neg), neg_target), random_state=CONFIG["random_seed"])
        pos_aug_needed = pos_target - len(chosen_pos)
        neg_aug_needed = neg_target - len(chosen_neg)
        # For augmentation: select seed rows to augment (repeat with augmentation)
        def plan_aug(seed_df, aug_needed):
            plan = {}
            if aug_needed <= 0:
                return plan
            if len(seed_df) == 0:
                return plan
            # repeat random choices from seed_df
            seed_ids = seed_df['SeriesInstanceUID'].tolist()
            # choose with replacement
            chosen = random.choices(seed_ids, k=aug_needed)
            for sid in chosen:
                plan[sid] = plan.get(sid, 0) + 1
            return plan
        pos_plan = plan_aug(chosen_pos if len(chosen_pos)>0 else pos, pos_aug_needed) if augment_fill_shortfall else {}
        neg_plan = plan_aug(chosen_neg if len(chosen_neg)>0 else neg, neg_aug_needed) if augment_fill_shortfall else {}
        # build manifest rows
        for _, r in chosen_pos.iterrows():
            augment_copies = pos_plan.get(r['SeriesInstanceUID'], 0)
            manifest_rows.append({
                "SeriesInstanceUID": r['SeriesInstanceUID'],
                "PatientID": r.get('PatientID', None),
                "Modality": modality,
                "Aneurysm Present": int(r['Aneurysm Present']),
                "augment_copies": augment_copies
            })
        for sid, cnt in pos_plan.items():
            # duplicates from selected seeds: include rows for augmented copies (they'll be flagged by augment_index)
            manifests_seed = df[df['SeriesInstanceUID'] == sid].iloc[0]
            manifest_rows.append({
                "SeriesInstanceUID": manifests_seed['SeriesInstanceUID'],
                "PatientID": manifests_seed.get('PatientID', None),
                "Modality": modality,
                "Aneurysm Present": int(manifests_seed['Aneurysm Present']),
                "augment_copies": 1  # each entry will represent one augmented image
            })
        for _, r in chosen_neg.iterrows():
            augment_copies = neg_plan.get(r['SeriesInstanceUID'], 0)
            manifest_rows.append({
                "SeriesInstanceUID": r['SeriesInstanceUID'],
                "PatientID": r.get('PatientID', None),
                "Modality": modality,
                "Aneurysm Present": int(r['Aneurysm Present']),
                "augment_copies": augment_copies
            })
        for sid, cnt in neg_plan.items():
            manifests_seed = df[df['SeriesInstanceUID'] == sid].iloc[0]
            manifest_rows.append({
                "SeriesInstanceUID": manifests_seed['SeriesInstanceUID'],
                "PatientID": manifests_seed.get('PatientID', None),
                "Modality": modality,
                "Aneurysm Present": int(manifests_seed['Aneurysm Present']),
                "augment_copies": 1
            })
        log(f"Modality {modality}: chosen_pos={len(chosen_pos)} chosen_neg={len(chosen_neg)} pos_aug_needed={pos_aug_needed} neg_aug_needed={neg_aug_needed}")
    manifest_df = pd.DataFrame(manifest_rows).reset_index(drop=True)
    manifest_df.to_csv(manifest_path, index=False)
    log(f"Saved manifest to {manifest_path}; total rows = {len(manifest_df)}")
    return manifest_df

# --------------------------
# 2) DICOM loading and volume reconstruction
# --------------------------
def read_dicom_headers_quick(paths: List[str]) -> List[pydicom.dataset.FileDataset]:
    out = []
    for p in paths:
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            ds.__filepath = p
            out.append(ds)
        except Exception as e:
            log(f"Warning: failed reading header {p}: {e}")
    return out

def sort_dicom_slices(headers: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
    # prefer ImagePositionPatient z coordinate
    def z_coord(hdr):
        ipp = getattr(hdr, "ImagePositionPatient", None)
        if ipp is not None and len(ipp) >= 3:
            return float(ipp[2])
        inst = getattr(hdr, "InstanceNumber", None)
        if inst is not None:
            return float(inst)
        # fallback to file name sort
        return float(0)
    headers_sorted = sorted(headers, key=z_coord)
    return headers_sorted

def load_series_volume(series_path: str) -> Tuple[np.ndarray, List[pydicom.dataset.FileDataset]]:
    """
    Load all .dcm files in series_path and return a 3D numpy array (Z, H, W) and sorted headers.
    Applies Rescale Slope/Intercept to pixel_array if present.
    """
    dicom_files = [os.path.join(series_path, f) for f in os.listdir(series_path) if f.lower().endswith('.dcm')]
    if len(dicom_files) == 0:
        raise FileNotFoundError(f"No DICOM files found in {series_path}")
    headers = read_dicom_headers_quick(dicom_files)
    headers_sorted = sort_dicom_slices(headers)
    slices = []
    for hdr in headers_sorted:
        p = getattr(hdr, "__filepath", None)
        try:
            ds_full = pydicom.dcmread(p, force=True)
            arr = ds_full.pixel_array.astype(np.float32)
            slope = float(getattr(ds_full, "RescaleSlope", 1.0))
            intercept = float(getattr(ds_full, "RescaleIntercept", 0.0))
            arr = arr * slope + intercept
            slices.append(arr)
        except Exception as e:
            log(f"Warning: failed reading pixels from {p}: {e}")
    if len(slices) == 0:
        raise RuntimeError(f"No pixel data extracted for series {series_path}")
    volume = np.stack(slices, axis=0)  # shape (Z,H,W)
    return volume, headers_sorted

# --------------------------
# 3) Localizer handling
# --------------------------
def load_localizers(localizers_csv_path: str) -> pd.DataFrame:
    if not os.path.exists(localizers_csv_path):
        log("No localizers CSV found; continuing without localizers.")
        return pd.DataFrame()
    locs = pd.read_csv(localizers_csv_path)
    # Ensure coordinates parsed
    if 'coordinates' in locs.columns:
        def parse_coords(c):
            if pd.isna(c):
                return {}
            try:
                if isinstance(c, str):
                    return ast.literal_eval(c) if (c.strip().startswith("{") or c.strip().startswith("[")) else json.loads(c)
                elif isinstance(c, dict):
                    return c
                else:
                    return {}
            except Exception:
                try:
                    return json.loads(c)
                except Exception:
                    return {}
        locs['coords_parsed'] = locs['coordinates'].apply(parse_coords) if 'coordinates' in locs.columns else [{}]*len(locs)
    else:
        locs['coords_parsed'] = [{}]*len(locs)
    return locs

def find_localizer_for_series(locs_df: pd.DataFrame, series_uid: str) -> Optional[Dict[str,Any]]:
    if locs_df is None or locs_df.empty:
        return None
    df = locs_df[locs_df['SeriesInstanceUID'] == series_uid]
    if df.shape[0] == 0:
        return None
    # return first match as dict (could be multiple)
    row = df.iloc[0]
    return {
        "SOPInstanceUID": row.get('SOPInstanceUID', None),
        "coords": row.get('coords_parsed', {})
    }

# --------------------------
# 4) 3D -> 2D conversion (MIP, slice, cropping)
# --------------------------
def volume_to_2d_image(volume: np.ndarray,
                       headers_sorted: List[pydicom.dataset.FileDataset],
                       method: str = "mip",
                       localizer_sop: Optional[str] = None,
                       local_coords: Optional[Dict[str, float]] = None,
                       crop_size: int = 128) -> Tuple[np.ndarray, Dict[str,Any]]:
    """
    Converts a 3D Z,H,W volume to a single 2D image.
    method: 'mip' or 'slice'
    If method == 'slice' and localizer_sop is provided, selects that slice.
    Returns (image_2d, info)
    """
    info = {"method": method, "used_slice_index": None, "used_crop": None}
    if method == "mip":
        img = np.max(volume, axis=0)
        info["used_slice_index"] = "mip"
    elif method == "slice":
        # Attempt to find index by SOPInstanceUID
        idx = None
        if localizer_sop:
            for i, hdr in enumerate(headers_sorted):
                if getattr(hdr, "SOPInstanceUID", None) == localizer_sop:
                    idx = i
                    break
        if idx is None:
            # fallback to middle slice
            idx = volume.shape[0] // 2
        img = volume[idx]
        info["used_slice_index"] = int(idx)
    else:
        # default to mip
        img = np.max(volume, axis=0)
        info["used_slice_index"] = "mip_default"
    # crop around local_coords if given
    H, W = img.shape
    if local_coords and "x" in local_coords and "y" in local_coords:
        x = int(round(local_coords["x"]))
        y = int(round(local_coords["y"]))
        half = crop_size // 2
        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(W, x + half)
        y2 = min(H, y + half)
        crop = img[y1:y2, x1:x2]
        if crop.size > 0:
            info["used_crop"] = (x1, y1, x2, y2)
            # if crop not full size, pad
            pad_h = crop_size - crop.shape[0]
            pad_w = crop_size - crop.shape[1]
            if pad_h > 0 or pad_w > 0:
                crop = np.pad(crop, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=np.min(img))
            img = crop
    return img.astype(np.float32), info

# --------------------------
# 5) Windowing / normalization / resize
# --------------------------
def apply_windowing(img: np.ndarray,
                    headers_sorted: Optional[List[pydicom.dataset.FileDataset]] = None,
                    use_dicom_window: bool = True,
                    percentile_clip: Tuple[int,int] = (1,99)) -> np.ndarray:
    """
    img: 2D float32 numpy array
    If DICOM tags available (WindowCenter/Width) in headers_sorted[0], apply that.
    Otherwise use percentile clipping.
    Returns image in range [0,1].
    """
    if use_dicom_window and headers_sorted:
        # try to find window center / width from any header that has them
        center, width = None, None
        for hdr in headers_sorted:
            wc = getattr(hdr, "WindowCenter", None)
            ww = getattr(hdr, "WindowWidth", None)
            if wc is not None and ww is not None:
                try:
                    center = float(wc[0] if isinstance(wc, (list, tuple)) else wc)
                    width = float(ww[0] if isinstance(ww, (list, tuple)) else ww)
                    break
                except Exception:
                    continue
        if center is not None and width is not None and width > 0:
            low = center - width / 2.0
            high = center + width / 2.0
            imgw = np.clip(img, low, high)
            imgw = (imgw - low) / (high - low)
            imgw = np.nan_to_num(imgw, nan=0.0, posinf=1.0, neginf=0.0)
            return imgw.astype(np.float32)
    # fallback percentile clipping
    lowp, highp = np.percentile(img, percentile_clip[0]), np.percentile(img, percentile_clip[1])
    imgc = np.clip(img, lowp, highp)
    if highp - lowp > 0:
        imgc = (imgc - lowp) / (highp - lowp)
    else:
        imgc = imgc - lowp
    imgc = np.nan_to_num(imgc, nan=0.0, posinf=1.0, neginf=0.0)
    return imgc.astype(np.float32)

def resize_and_to_uint8(img: np.ndarray, target_size: int = 224) -> np.ndarray:
    # input img in [0,1], float32
    h, w = img.shape
    if h == target_size and w == target_size:
        out = (img * 255.0).astype(np.uint8)
        return out
    out = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    out = np.clip(out * 255.0, 0, 255).astype(np.uint8)
    return out

# --------------------------
# 6) Dataset & transforms
# --------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_train_transforms(target_size: int):
    return A.Compose([
        A.RandomRotate90(p=0.2),  # 0/90/180/270 random sometimes - helpful
        A.Rotate(limit=10, p=0.5),
        A.Flip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=0, p=0.2),
        A.Resize(target_size, target_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

def get_val_transforms(target_size: int):
    return A.Compose([
        A.Resize(target_size, target_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2()
    ])

class AneurysmSeriesDataset(Dataset):
    def __init__(self, manifest_df: pd.DataFrame, data_root: str, series_folder: str,
                 localizers_df: pd.DataFrame = None, transforms=None, cache_dir: str = None,
                 use_mip_for_modality: List[str] = None, target_size: int = 224):
        """
        manifest_df: DataFrame where each row is a series to include. augment_copies in manifest represent duplication for underrepresented classes.
        NOTE: If a SeriesInstanceUID appears multiple times in manifest (due to augmentation plan), we treat each row as a separate sample.
        """
        self.manifest = manifest_df.reset_index(drop=True).copy()
        self.data_root = data_root
        self.series_folder = series_folder
        self.localizers_df = localizers_df
        self.transforms = transforms
        self.cache_dir = cache_dir
        self.use_mip_for_modality = set(use_mip_for_modality or [])
        self.target_size = target_size
        # expand manifest rows where augment_copies > 0 into repeated rows flagged as augmented
        expanded_rows = []
        for _, row in self.manifest.iterrows():
            base = row.to_dict()
            aug = int(base.get('augment_copies', 0))
            # first add original
            expanded_rows.append({**base, "is_augmented": False})
            # additional augmented entries
            for i in range(aug):
                expanded_rows.append({**base, "is_augmented": True, "augment_index": i})
        self.entries = pd.DataFrame(expanded_rows).reset_index(drop=True)
        log(f"Dataset entries: {len(self.entries)} (including augmented copies).")
    def __len__(self):
        return len(self.entries)
    def __getitem__(self, idx):
        row = self.entries.iloc[idx]
        series_uid = row['SeriesInstanceUID']
        modality = str(row['Modality']).upper()
        label = int(row['Aneurysm Present'])
        is_augmented = bool(row.get('is_augmented', False))
        # cache filename key
        cache_name = f"{series_uid}"
        if is_augmented:
            cache_name = f"{series_uid}_aug{row.get('augment_index',0)}"
        cache_path = os.path.join(self.cache_dir, cache_name + ".npy") if self.cache_dir else None
        if cache_path and os.path.exists(cache_path):
            # load cached processed image (uint8 with shape H,W or H,W,3)
            img_uint8 = np.load(cache_path)
        else:
            # load series, convert to 2D, window, resize -> uint8
            series_path = os.path.join(self.data_root, self.series_folder, series_uid)
            try:
                volume, headers_sorted = load_series_volume(series_path)
            except Exception as e:
                raise RuntimeError(f"Failed to load series {series_uid} at {series_path}: {e}")
            # localizer
            localizer = find_localizer_for_series(self.localizers_df, series_uid) if (self.localizers_df is not None and not self.localizers_df.empty) else None
            local_coords = localizer.get('coords') if localizer else {}
            local_sop = localizer.get('SOPInstanceUID') if localizer else None
            # choose method
            method = "mip" if modality in self.use_mip_for_modality else "slice"
            img2d, info = volume_to_2d_image(volume, headers_sorted, method=method, localizer_sop=local_sop, local_coords=local_coords, crop_size=min(128, self.target_size))
            # window & normalize
            imgw = apply_windowing(img2d, headers_sorted=headers_sorted if CONFIG["windowing"]["use_dicom_window"] else None,
                                   use_dicom_window=CONFIG["windowing"]["use_dicom_window"],
                                   percentile_clip=CONFIG["windowing"]["percentile_clip"])
            img_uint8 = resize_and_to_uint8(imgw, target_size=self.target_size)
            # save cache
            if cache_path:
                try:
                    np.save(cache_path, img_uint8)
                except Exception as e:
                    log(f"Warning: failed to save cache {cache_path}: {e}")
        # At this point img_uint8 is H,W uint8. Convert to 3-channel
        if img_uint8.ndim == 2:
            img_rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=2)
        elif img_uint8.ndim == 3 and img_uint8.shape[2] == 3:
            img_rgb = img_uint8
        else:
            # handle odd channel shapes
            img_rgb = np.stack([img_uint8[...,0], img_uint8[...,0], img_uint8[...,0]], axis=2)
        # If row flagged as augmented: we will apply augmentations (transforms passed in) which include randomness
        if self.transforms:
            # Albumentations expects image in HWC format uint8
            augmented = self.transforms(image=img_rgb)
            img_tensor = augmented['image']  # this is a torch.Tensor [C,H,W], normalized
        else:
            # fallback transform: convert to tensor and normalize
            t = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
            ])
            img_tensor = t(img_rgb)
        return img_tensor, label, series_uid

# --------------------------
# 7) Model building & training helpers
# --------------------------
def build_model(model_name: str = "resnet50", pretrained: bool = True, num_classes: int = 2, in_channels: int = 3):
    if model_name.lower() == "resnet50":
        model = models.resnet50(pretrained=pretrained)
    elif model_name.lower() == "resnet34":
        model = models.resnet34(pretrained=pretrained)
    elif model_name.lower() == "resnet18":
        model = models.resnet18(pretrained=pretrained)
    else:
        raise ValueError("Unsupported model_name")
    # adapt first conv if in_channels != 3
    if in_channels != 3:
        old_conv = model.conv1
        new_conv = nn.Conv2d(in_channels, old_conv.out_channels, kernel_size=old_conv.kernel_size,
                             stride=old_conv.stride, padding=old_conv.padding, bias=old_conv.bias is not None)
        # initialize new_conv weights by averaging existing channels
        with torch.no_grad():
            if in_channels == 1:
                new_conv.weight[:] = old_conv.weight.mean(dim=1, keepdim=True)
        model.conv1 = new_conv
    # replace fc
    in_feats = model.fc.in_features
    model.fc = nn.Linear(in_feats, num_classes)
    return model

def compute_metrics(y_true: List[int], y_pred_probs: List[float], threshold: float = 0.5) -> Dict[str, float]:
    probs = np.array(y_pred_probs)
    y_true = np.array(y_true)
    y_pred = (probs >= threshold).astype(int)
    metrics = {}
    try:
        metrics['auc'] = float(roc_auc_score(y_true, probs))
    except Exception:
        metrics['auc'] = float('nan')
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['precision'] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics['recall'] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics['f1'] = float(f1_score(y_true, y_pred, zero_division=0))
    return metrics

# --------------------------
# 8) Training loop (two-phase)
# --------------------------
def train_model(manifest_df: pd.DataFrame, localizers_df: pd.DataFrame, data_root: str, series_folder: str):
    # Prepare train/val split (stratified by modality & label)
    # We will create splits within each modality to ensure 80/20 stratified
    train_idx = []
    val_idx = []
    for modality, group in manifest_df.groupby('Modality'):
        grp = group.reset_index(drop=True)
        # stratify by 'Aneurysm Present'
        if len(grp) < 2:
            # small groups: put all to train
            train_idx += grp.index.tolist()
            continue
        # use train_test_split but keep indices consistent with manifest_df index — we'll use local group indexing and map back
        gtrain, gval = train_test_split(grp, test_size=0.2, stratify=grp['Aneurysm Present'], random_state=CONFIG["random_seed"])
        # get original indices
        train_idx += gtrain.index.tolist()
        val_idx += gval.index.tolist()
    train_df = manifest_df.iloc[train_idx].reset_index(drop=True)
    val_df = manifest_df.iloc[val_idx].reset_index(drop=True)
    log(f"Train size: {len(train_df)}  Val size: {len(val_df)}")
    # Datasets
    train_ds = AneurysmSeriesDataset(train_df, data_root=data_root, series_folder=series_folder,
                                     localizers_df=localizers_df, transforms=get_train_transforms(CONFIG["target_image_size"]),
                                     cache_dir=CONFIG["cache_dir"], use_mip_for_modality=CONFIG["use_mip_for_modality"],
                                     target_size=CONFIG["target_image_size"])
    val_ds = AneurysmSeriesDataset(val_df, data_root=data_root, series_folder=series_folder,
                                     localizers_df=localizers_df, transforms=get_val_transforms(CONFIG["target_image_size"]),
                                     cache_dir=CONFIG["cache_dir"], use_mip_for_modality=CONFIG["use_mip_for_modality"],
                                     target_size=CONFIG["target_image_size"])
    # DataLoaders
    train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=CONFIG["num_workers"], pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=CONFIG["num_workers"], pin_memory=True)
    # Model
    model = build_model(CONFIG["model_name"], pretrained=True, num_classes=2, in_channels=3)
    model = model.to(CONFIG["device"])
    # Loss
    criterion = nn.CrossEntropyLoss()
    # Phase 1: train head only
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG["head_lr"], weight_decay=CONFIG["weight_decay"])
    scaler = torch.cuda.amp.GradScaler(enabled=(CONFIG["device"].startswith("cuda")))
    best_val_auc = -1.0
    best_epoch = -1
    history_rows = []
    # helper functions
    def run_one_epoch(epoch_idx, loader, training=True):
        if training:
            model.train()
        else:
            model.eval()
        running_loss = 0.0
        y_true_all = []
        y_prob_all = []
        pbar = tqdm(loader, desc=("Train" if training else "Val") + f" Epoch {epoch_idx}", leave=False)
        for batch in pbar:
            imgs, labels, sids = batch
            imgs = imgs.to(CONFIG["device"], non_blocking=True)
            labels = labels.to(CONFIG["device"], non_blocking=True)
            with torch.set_grad_enabled(training):
                with torch.cuda.amp.autocast(enabled=(CONFIG["device"].startswith("cuda"))):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                if training:
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
            running_loss += float(loss.item()) * imgs.shape[0]
            probs = F.softmax(outputs.detach(), dim=1)[:,1].cpu().numpy()
            y_prob_all.extend(probs.tolist())
            y_true_all.extend(labels.cpu().numpy().tolist())
            pbar.set_postfix(loss=running_loss / ((pbar.n + 1) * loader.batch_size))
        avg_loss = running_loss / len(loader.dataset)
        metrics = compute_metrics(y_true_all, y_prob_all)
        metrics['loss'] = avg_loss
        return metrics
    # training head
    log("Starting Phase 1: training classifier head only.")
    patience_cnt = 0
    for epoch in range(CONFIG["epochs_head"]):
        t0 = time.time()
        train_metrics = run_one_epoch(epoch, train_loader, training=True)
        val_metrics = run_one_epoch(epoch, val_loader, training=False)
        elapsed = time.time() - t0
        log(f"[Head] Epoch {epoch+1}/{CONFIG['epochs_head']}  train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} time={elapsed:.1f}s")
        history_rows.append({"phase":"head","epoch":epoch+1,"train_loss":train_metrics['loss'],"val_loss":val_metrics['loss'],
                             "val_auc":val_metrics['auc'],"val_f1":val_metrics['f1']})
        # checkpoint best
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_head.pth"))
            log(f"Saved best_head.pth with val_auc={best_val_auc:.4f}")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["patience"]:
                log("Early stopping head training due to no improvement.")
                break
    # Phase 2: fine-tune entire model
    log("Starting Phase 2: fine-tuning entire model.")
    for p in model.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["ft_lr"], weight_decay=CONFIG["weight_decay"])
    patience_cnt = 0
    for epoch in range(CONFIG["epochs_finetune"]):
        t0 = time.time()
        train_metrics = run_one_epoch(epoch, train_loader, training=True)
        val_metrics = run_one_epoch(epoch, val_loader, training=False)
        elapsed = time.time() - t0
        log(f"[FT ] Epoch {epoch+1}/{CONFIG['epochs_finetune']}  train_loss={train_metrics['loss']:.4f} val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} time={elapsed:.1f}s")
        history_rows.append({"phase":"finetune","epoch":epoch+1,"train_loss":train_metrics['loss'],"val_loss":val_metrics['loss'],
                             "val_auc":val_metrics['auc'],"val_f1":val_metrics['f1']})
        # checkpoint best
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_finetune.pth"))
            log(f"Saved best_finetune.pth with val_auc={best_val_auc:.4f}")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= CONFIG["patience"]:
                log("Early stopping finetune due to no improvement.")
                break
    # save final model
    torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "final_model.pth"))
    # save history
    hist_df = pd.DataFrame(history_rows)
    hist_df.to_csv(os.path.join(CONFIG["save_dir"], "training_history.csv"), index=False)
    log(f"Training complete. Best val_auc={best_val_auc:.4f} at epoch {best_epoch}. Artifacts in {CONFIG['save_dir']}")

# --------------------------
# 9) Main execution
# --------------------------
def main():
    data_root = CONFIG["data_root"]
    train_csv_path = os.path.join(data_root, CONFIG["train_csv"])
    localizers_csv_path = os.path.join(data_root, CONFIG["localizers_csv"])
    manifest_path = CONFIG["manifest_path"]
    # 1) Create balanced manifest if not exists
    if not os.path.exists(manifest_path):
        manifest_df = create_balanced_manifest(train_csv_path, manifest_path,
                                               modalities_to_process=CONFIG["modalities_to_process"],
                                               pos_target=CONFIG["per_modality_pos"],
                                               neg_target=CONFIG["per_modality_neg"],
                                               augment_fill_shortfall=CONFIG["augment_fill_shortfall"])
    else:
        manifest_df = pd.read_csv(manifest_path)
        log(f"Loaded existing manifest from {manifest_path}")
    # 2) Load localizers (if available)
    localizers_df = load_localizers(localizers_csv_path) if os.path.exists(localizers_csv_path) else pd.DataFrame()
    # 3) Train model
    train_model(manifest_df, localizers_df, data_root, CONFIG["series_folder"])

if __name__ == "__main__":
    main()



import os
import pandas as pd
import numpy as np
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import json
from pathlib import Path
import random
from collections import defaultdict

warnings.filterwarnings('ignore')

class Config:
    """Configuration class for the medical imaging pipeline"""
    
    # Data paths
    DATA_ROOT = r"/kaggle/input/rsna-intracranial-aneurysm-detection"  # Update this path
    SERIES_PATH = os.path.join(DATA_ROOT, "series")
    TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
    TRAIN_LOCALIZERS_CSV = os.path.join(DATA_ROOT, "train_localizers.csv")
    
    # Sampling parameters
    SAMPLES_PER_CLASS = 150
    TOTAL_SAMPLES_PER_MODALITY = 300
    
    # Image parameters
    IMAGE_SIZE = 224
    WINDOWING_CENTER = 40  # Brain window center
    WINDOWING_WIDTH = 80   # Brain window width
    
    # Training parameters
    BATCH_SIZE = 32
    LEARNING_RATE_PHASE1 = 1e-4  # For classifier head only
    LEARNING_RATE_PHASE2 = 1e-6  # For full fine-tuning
    EPOCHS_PHASE1 = 5
    EPOCHS_PHASE2 = 15
    VALIDATION_SPLIT = 0.2
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model parameters
    MODEL_NAME = 'resnet50'
    NUM_CLASSES = 2

class DICOMLoader:
    """Handles DICOM file loading and 3D to 2D conversion"""
    
    @staticmethod
    def load_dicom_series(series_path):
        """Load all DICOM files from a series directory and return as 3D volume"""
        dcm_files = []
        dicom_dir = Path(series_path)
        
        if not dicom_dir.exists():
            print(f"Warning: Directory does not exist: {series_path}")
            return None
            
        for dcm_file in dicom_dir.glob("*.dcm"):
            try:
                ds = pydicom.dcmread(str(dcm_file), force=True)
                if hasattr(ds, 'pixel_array') and ds.pixel_array is not None:
                    # Handle different pixel data types
                    pixel_array = ds.pixel_array.astype(np.float32)
                    if pixel_array.size > 0:  # Check if array is not empty
                        dcm_files.append((ds, pixel_array))
            except Exception as e:
                print(f"Warning: Error loading {dcm_file}: {e}")
                continue
        
        if not dcm_files:
            print(f"Warning: No valid DICOM files found in {series_path}")
            return None
        
        # Sort by instance number if available
        try:
            dcm_files.sort(key=lambda x: int(x[0].InstanceNumber) if hasattr(x[0], 'InstanceNumber') else 0)
        except:
            pass
        
        # Stack into 3D volume
        try:
            pixel_arrays = [dcm_data[1] for dcm_data in dcm_files]
            
            # Ensure all arrays have the same shape
            shapes = [arr.shape for arr in pixel_arrays]
            if len(set(shapes)) > 1:
                # Resize to the most common shape
                from collections import Counter
                most_common_shape = Counter(shapes).most_common(1)[0][0]
                pixel_arrays = [cv2.resize(arr, (most_common_shape[1], most_common_shape[0])) 
                              if arr.shape != most_common_shape else arr 
                              for arr in pixel_arrays]
            
            volume = np.stack(pixel_arrays, axis=0)
            return volume, dcm_files[0][0]  # Return volume and first DICOM for metadata
        except Exception as e:
            print(f"Warning: Error stacking DICOM files from {series_path}: {e}")
            # Return the first valid image as a single slice
            if dcm_files:
                return dcm_files[0][1][np.newaxis, ...], dcm_files[0][0]
            return None
    
    @staticmethod
    def convert_3d_to_2d(volume, method='middle_slice'):
        """Convert 3D volume to 2D image using specified method"""
        if volume is None or volume.size == 0:
            return None
        
        # Ensure volume is at least 2D
        if len(volume.shape) == 1:
            return None
        elif len(volume.shape) == 2:
            return volume
        elif len(volume.shape) >= 3:
            if method == 'middle_slice':
                return volume[volume.shape[0] // 2]
            elif method == 'max_intensity_projection':
                return np.max(volume, axis=0)
            elif method == 'mean_projection':
                return np.mean(volume, axis=0)
            else:
                return volume[volume.shape[0] // 2]
        else:
            return None

class ImagePreprocessor:
    """Handles image preprocessing including windowing, normalization, and resizing"""
    
    @staticmethod
    def apply_windowing(image, center=40, width=80):
        """Apply windowing to enhance contrast"""
        if image is None:
            return None
        
        # Ensure image is a numpy array
        if not isinstance(image, np.ndarray):
            return None
        
        # Handle empty or invalid arrays
        if image.size == 0 or not np.isfinite(image).any():
            return None
        
        min_val = center - width // 2
        max_val = center + width // 2
        
        try:
            windowed = np.clip(image, min_val, max_val)
            windowed = (windowed - min_val) / (max_val - min_val)
            return windowed
        except Exception as e:
            print(f"Warning: Windowing failed: {e}")
            return None
    
    @staticmethod
    def normalize_image(image):
        """Normalize image to [0, 1] range"""
        if image is None:
            return None
        
        try:
            # Handle constant images
            if image.max() == image.min():
                return np.zeros_like(image)
            
            normalized = (image - image.min()) / (image.max() - image.min())
            return normalized
        except Exception as e:
            print(f"Warning: Normalization failed: {e}")
            return None
    
    @staticmethod
    def resize_image(image, target_size=224):
        """Resize image to target size"""
        if image is None:
            return None
        
        try:
            # Ensure image is 2D
            if len(image.shape) != 2:
                return None
            
            resized = cv2.resize(image, (target_size, target_size))
            return resized
        except Exception as e:
            print(f"Warning: Resizing failed: {e}")
            return None
    
    @staticmethod
    def preprocess_dicom_image(image, config):
        """Complete preprocessing pipeline for DICOM image"""
        if image is None:
            # Return a blank image if input is None
            return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
        
        # Apply windowing
        windowed = ImagePreprocessor.apply_windowing(
            image, config.WINDOWING_CENTER, config.WINDOWING_WIDTH
        )
        
        if windowed is None:
            # Fallback: try simple normalization
            windowed = ImagePreprocessor.normalize_image(image)
        
        if windowed is None:
            # Final fallback: return blank image
            return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
        
        # Normalize
        normalized = ImagePreprocessor.normalize_image(windowed)
        if normalized is None:
            return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
        
        # Convert to uint8 for OpenCV operations
        try:
            normalized_uint8 = (normalized * 255).astype(np.uint8)
        except:
            return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
        
        # Resize
        resized = ImagePreprocessor.resize_image(normalized_uint8, config.IMAGE_SIZE)
        if resized is None:
            return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.float32)
        
        # Convert back to float and normalize to [0, 1]
        final_image = resized.astype(np.float32) / 255.0
        
        return final_image

class DataSampler:
    """Handles balanced data sampling and augmentation"""
    
    def __init__(self, config):
        self.config = config
    
    def load_metadata(self):
        """Load train.csv and train_localizers.csv"""
        train_df = pd.read_csv(self.config.TRAIN_CSV)
        try:
            localizers_df = pd.read_csv(self.config.TRAIN_LOCALIZERS_CSV)
        except:
            localizers_df = None
            print("Warning: Could not load train_localizers.csv")
        return train_df, localizers_df
    
    def sample_balanced_data(self, train_df):
        """Create balanced dataset for each modality"""
        balanced_data = {}
        
        modalities = train_df['Modality'].unique()
        print(f"Found modalities: {modalities}")
        
        for modality in modalities:
            modality_data = train_df[train_df['Modality'] == modality]
            
            positive_samples = modality_data[modality_data['Aneurysm Present'] == 1]
            negative_samples = modality_data[modality_data['Aneurysm Present'] == 0]
            
            print(f"\n{modality} - Positive: {len(positive_samples)}, Negative: {len(negative_samples)}")
            
            # Sample or augment to reach target count
            selected_positive = self._sample_or_augment(positive_samples, self.config.SAMPLES_PER_CLASS)
            selected_negative = self._sample_or_augment(negative_samples, self.config.SAMPLES_PER_CLASS)
            
            balanced_data[modality] = {
                'positive': selected_positive,
                'negative': selected_negative
            }
            
            print(f"Selected - Positive: {len(selected_positive)}, Negative: {len(selected_negative)}")
        
        return balanced_data
    
    def _sample_or_augment(self, samples, target_count):
        """Sample or augment data to reach target count"""
        if len(samples) >= target_count:
            return samples.sample(n=target_count, random_state=42)
        else:
            # Need augmentation
            augmentation_needed = target_count - len(samples)
            print(f"Augmentation needed: {augmentation_needed} samples")
            
            # Simple augmentation by repeating samples with slight modifications
            augmented_samples = []
            original_samples = samples.copy()
            
            for i in range(augmentation_needed):
                sample_to_duplicate = original_samples.iloc[i % len(original_samples)].copy()
                # Mark as augmented for tracking
                sample_to_duplicate['is_augmented'] = True
                augmented_samples.append(sample_to_duplicate)
            
            # Combine original and augmented
            all_samples = pd.concat([original_samples, pd.DataFrame(augmented_samples)], ignore_index=True)
            return all_samples

class MedicalImageDataset(Dataset):
    """PyTorch dataset for medical images"""
    
    def __init__(self, series_data, config, transform=None, is_training=True):
        self.series_data = series_data
        self.config = config
        self.transform = transform
        self.is_training = is_training
        
        # Prepare data list
        self.data_list = []
        for _, row in series_data.iterrows():
            self.data_list.append({
                'series_uid': row['SeriesInstanceUID'],
                'label': row['Aneurysm Present'],
                'modality': row['Modality'],
                'is_augmented': row.get('is_augmented', False)
            })
        
        print(f"Dataset created with {len(self.data_list)} samples")
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        item = self.data_list[idx]
        series_uid = item['series_uid']
        label = item['label']
        is_augmented = item['is_augmented']
        
        # Load DICOM series
        series_path = os.path.join(self.config.SERIES_PATH, series_uid)
        volume_data = DICOMLoader.load_dicom_series(series_path)
        
        if volume_data is None:
            # Return a blank image if loading fails
            print(f"Warning: Failed to load series {series_uid}, using blank image")
            image = np.zeros((self.config.IMAGE_SIZE, self.config.IMAGE_SIZE), dtype=np.float32)
        else:
            volume, dicom_metadata = volume_data
            # Convert 3D to 2D
            image_2d = DICOMLoader.convert_3d_to_2d(volume, method='middle_slice')
            # Preprocess
            image = ImagePreprocessor.preprocess_dicom_image(image_2d, self.config)
        
        # Apply additional augmentation for augmented samples
        if is_augmented and self.is_training:
            image = self._apply_augmentation(image)
        
        # Convert to 3-channel for ResNet
        image = np.stack([image, image, image], axis=0)  # Shape: (3, H, W)
        
        # Apply transforms
        if self.transform:
            image = torch.from_numpy(image).float()
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).float()
        
        label = torch.tensor(label, dtype=torch.long)
        
        return image, label
    
    def _apply_augmentation(self, image):
        """Apply simple augmentation for training"""
        if image is None or image.size == 0:
            return image
        
        try:
            # Random rotation
            if random.random() > 0.5:
                angle = random.uniform(-10, 10)
                h, w = image.shape
                M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1)
                image = cv2.warpAffine(image, M, (w, h))
            
            # Random flip
            if random.random() > 0.5:
                image = cv2.flip(image, 1)  # Horizontal flip
            
            # Brightness adjustment
            if random.random() > 0.5:
                brightness_factor = random.uniform(0.8, 1.2)
                image = np.clip(image * brightness_factor, 0, 1)
        except Exception as e:
            print(f"Warning: Augmentation failed: {e}")
        
        return image

class ResNetModel(nn.Module):
    """ResNet model for binary classification"""
    
    def __init__(self, model_name='resnet50', num_classes=2, pretrained=True):
        super(ResNetModel, self).__init__()
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Replace final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        return self.backbone(x)
    
    def freeze_backbone(self):
        """Freeze all layers except the final classifier"""
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the final layer
        for param in self.backbone.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning"""
        for param in self.backbone.parameters():
            param.requires_grad = True

class Trainer:
    """Training manager for the ResNet model"""
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        self.device = config.DEVICE
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
    
    def train_phase1(self, train_loader, val_loader):
        """Phase 1: Train only the classifier head"""
        print("\n=== Phase 1: Training classifier head ===")
        
        # Freeze backbone
        self.model.freeze_backbone()
        
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.LEARNING_RATE_PHASE1
        )
        
        self._train_epochs(train_loader, val_loader, optimizer, self.config.EPOCHS_PHASE1, "Phase1")
    
    def train_phase2(self, train_loader, val_loader):
        """Phase 2: Fine-tune entire model"""
        print("\n=== Phase 2: Fine-tuning entire model ===")
        
        # Unfreeze all layers
        self.model.unfreeze_all()
        
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE_PHASE2,
            weight_decay=1e-4
        )
        
        self._train_epochs(train_loader, val_loader, optimizer, self.config.EPOCHS_PHASE2, "Phase2")
    
    def _train_epochs(self, train_loader, val_loader, optimizer, epochs, phase_name):
        """Train for specified number of epochs"""
        best_val_acc = 0.0
        
        for epoch in range(epochs):
            # Training
            train_loss, train_acc = self._train_epoch(train_loader, optimizer)
            
            # Validation
            val_loss, val_acc = self._validate_epoch(val_loader)
            
            # Store metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            print(f"{phase_name} Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print("-" * 50)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), f'best_model_{phase_name.lower()}.pth')
    
    def _train_epoch(self, train_loader, optimizer):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(tqdm(train_loader, desc="Training")):
            try:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
            except Exception as e:
                print(f"Warning: Error in training batch {batch_idx}: {e}")
                continue
        
        avg_loss = total_loss / len(train_loader) if len(train_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in tqdm(val_loader, desc="Validation"):
                try:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    
                    total_loss += loss.item()
                    pred = output.argmax(dim=1)
                    correct += pred.eq(target).sum().item()
                    total += target.size(0)
                except Exception as e:
                    print(f"Warning: Error in validation batch: {e}")
                    continue
        
        avg_loss = total_loss / len(val_loader) if len(val_loader) > 0 else 0
        accuracy = correct / total if total > 0 else 0
        
        return avg_loss, accuracy
    
    def evaluate(self, test_loader):
        """Evaluate model on test set"""
        self.model.eval()
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc="Evaluating"):
                try:
                    data, target = data.to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1)
                    
                    all_preds.extend(pred.cpu().numpy())
                    all_targets.extend(target.cpu().numpy())
                except Exception as e:
                    print(f"Warning: Error in evaluation batch: {e}")
                    continue
        
        if len(all_targets) == 0:
            return 0.0, "No valid predictions", np.array([[0, 0], [0, 0]])
        
        accuracy = accuracy_score(all_targets, all_preds)
        report = classification_report(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)
        
        return accuracy, report, cm
    
    def plot_training_history(self):
        """Plot training history"""
        if not self.train_losses:
            print("No training history to plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

class MedicalImagingPipeline:
    """Main pipeline class that orchestrates the entire process"""
    
    def __init__(self, config):
        self.config = config
        
    def run_complete_pipeline(self):
        """Run the complete pipeline from data loading to model evaluation"""
        print("Starting Medical Imaging Pipeline for Aneurysm Detection")
        print(f"Device: {self.config.DEVICE}")
        print("=" * 60)
        
        # Step 1: Load metadata and sample balanced data
        print("\n1. Loading metadata and creating balanced dataset...")
        sampler = DataSampler(self.config)
        train_df, localizers_df = sampler.load_metadata()
        balanced_data = sampler.sample_balanced_data(train_df)
        
        # Step 2: Prepare datasets for each modality
        print("\n2. Preparing datasets...")
        all_datasets = {}
        
        for modality, data in balanced_data.items():
            print(f"\nProcessing {modality}...")
            
            # Combine positive and negative samples
            combined_data = pd.concat([data['positive'], data['negative']], ignore_index=True)
            
            # Split into train and validation
            train_data, val_data = train_test_split(
                combined_data, 
                test_size=self.config.VALIDATION_SPLIT, 
                stratify=combined_data['Aneurysm Present'],
                random_state=42
            )
            
            print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")
            
            # Define transforms
            train_transform = transforms.Compose([
                transforms.RandomRotation(10),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            val_transform = transforms.Compose([
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            # Create datasets
            train_dataset = MedicalImageDataset(train_data, self.config, train_transform, is_training=True)
            val_dataset = MedicalImageDataset(val_data, self.config, val_transform, is_training=False)
            
            # Create data loaders with error handling
            train_loader = DataLoader(
                train_dataset, 
                batch_size=self.config.BATCH_SIZE, 
                shuffle=True, 
                num_workers=0,  # Set to 0 to avoid multiprocessing issues
                pin_memory=False
            )
            val_loader = DataLoader(
                val_dataset, 
                batch_size=self.config.BATCH_SIZE, 
                shuffle=False, 
                num_workers=0,
                pin_memory=False
            )
            
            all_datasets[modality] = {
                'train_loader': train_loader,
                'val_loader': val_loader,
                'train_size': len(train_dataset),
                'val_size': len(val_dataset)
            }
        
        # Step 3: Train models for each modality
        print("\n3. Training models...")
        trained_models = {}
        
        for modality, datasets in all_datasets.items():
            print(f"\n{'='*20} Training {modality} Model {'='*20}")
            
            # Initialize model
            model = ResNetModel(
                model_name=self.config.MODEL_NAME,
                num_classes=self.config.NUM_CLASSES,
                pretrained=True
            )
            
            # Initialize trainer
            trainer = Trainer(model, self.config)
            
            # Two-phase training
            trainer.train_phase1(datasets['train_loader'], datasets['val_loader'])
            trainer.train_phase2(datasets['train_loader'], datasets['val_loader'])
            
            # Evaluate
            print(f"\nEvaluating {modality} model...")
            accuracy, report, cm = trainer.evaluate(datasets['val_loader'])
            
            print(f"Final Validation Accuracy: {accuracy:.4f}")
            print("\nClassification Report:")
            print(report)
            
            # Plot confusion matrix
            try:
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                plt.title(f'Confusion Matrix - {modality}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.savefig(f'confusion_matrix_{modality}.png', dpi=300, bbox_inches='tight')
                plt.show()
            except Exception as e:
                print(f"Warning: Could not plot confusion matrix: {e}")
            
            # Plot training history
            trainer.plot_training_history()
            
            trained_models[modality] = {
                'model': model,
                'trainer': trainer,
                'accuracy': accuracy,
                'report': report,
                'confusion_matrix': cm
            }
        
        # Step 4: Summary
        print("\n" + "="*60)
        print("TRAINING SUMMARY")
        print("="*60)
        
        for modality, results in trained_models.items():
            print(f"{modality}: Validation Accuracy = {results['accuracy']:.4f}")
        
        return trained_models

def main():
    """Main execution function"""
    # Initialize configuration
    config = Config()
    
    # Verify data paths exist
    if not os.path.exists(config.DATA_ROOT):
        print(f"Error: Data root path does not exist: {config.DATA_ROOT}")
        print("Please update the DATA_ROOT path in the Config class")
        return
    
    if not os.path.exists(config.TRAIN_CSV):
        print(f"Error: train.csv not found at: {config.TRAIN_CSV}")
        return
    
    # Create and run pipeline
    pipeline = MedicalImagingPipeline(config)
    trained_models = pipeline.run_complete_pipeline()
    
    print("\nPipeline completed successfully!")
    print(f"Trained models available for {len(trained_models)} modalities")

if __name__ == "__main__":
    main()

