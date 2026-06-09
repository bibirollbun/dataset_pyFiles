
import os
import gc
import re
import cv2
import math
import numpy as np
import pandas as pd
import polars as pl
import pydicom
import torch
import torch.nn as nn
import timm
from collections import defaultdict
from typing import List, Tuple
import shutil
from sklearn.metrics import roc_auc_score

# Kaggle server
import kaggle_evaluation.rsna_inference_server

# ========= Competition schema =========
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

# Optional allowlist (not used in modeling; provided for compliance/reference)
DICOM_TAG_ALLOWLIST = [
    'BitsAllocated','BitsStored','Columns','FrameOfReferenceUID','HighBit',
    'ImageOrientationPatient','ImagePositionPatient','InstanceNumber','Modality',
    'PatientID','PhotometricInterpretation','PixelRepresentation','PixelSpacing',
    'PlanarConfiguration','RescaleIntercept','RescaleSlope','RescaleType','Rows',
    'SOPClassUID','SOPInstanceUID','SamplesPerPixel','SliceThickness',
    'SpacingBetweenSlices','StudyInstanceUID','TransferSyntaxUID',
]

# ========= Inference config =========
IMG_SIZE = 224
OFFSETS = (-2, -1, 0, 1, 2)   # window length 5
IN_CHANS = len(OFFSETS)
BATCH_SIZE = 16
AGGREGATE = "max"  # max/mean/topk_mean
USE_ROI = False     # coords not available on test â†’ use same stream for full+roi
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Where model weights are stored. Attach a Dataset with your .pth files if needed.
CANDIDATE_MODEL_DIRS = [
    "/kaggle/input/rsna-2025-ia-ct-224-efficientnet/pytorch/default/1",        # attached datasets
    "/kaggle/working",      # runtime dir
    ".",                    # current dir
]

# ========= Model definition (Hybrid full + ROI + coords) =========
class HybridAneurysmModel(nn.Module):
    def __init__(self, base_model_name: str, num_classes: int):
        super().__init__()
        self.backbone = timm.create_model(base_model_name, in_chans=IN_CHANS, num_classes=0, pretrained=False)
        self.feature_dim = self.backbone.num_features
        self.coord_fc = nn.Sequential(nn.Linear(2, 32), nn.ReLU(), nn.Linear(32, 64))
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(self.feature_dim * 2 + 64, num_classes))

    def forward(self, x_full: torch.Tensor, x_roi: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        f_full = self.backbone(x_full)
        f_roi  = self.backbone(x_roi)
        f_coord = self.coord_fc(coords.float())
        return self.fc(torch.cat([f_full, f_roi, f_coord], dim=1))

# ========= Helpers =========
def sort_dicom_slices(filepaths: List[str]):
    dicoms = [pydicom.dcmread(fp, force=True) for fp in filepaths]
    try:
        dicoms.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    except Exception:
        dicoms.sort(key=lambda d: int(getattr(d, 'InstanceNumber', 0)))
    return dicoms

def series_to_tensor_chw(dicoms) -> np.ndarray:
    # Resize all to IMG_SIZE and per-series z-score normalization
    resized = []
    for d in dicoms:
        arr = d.pixel_array
        if arr is None or arr.size == 0:
            continue
        arr = arr.astype(np.float32)
        arr = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        resized.append(arr)
    if len(resized) == 0:
        # fallback to zeros to avoid crashes (rare)
        vol = np.zeros((1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    else:
        vol = np.stack(resized, axis=0)  # [N,H,W]
    mean = float(vol.mean())
    std = float(vol.std()) + 1e-6
    vol = (vol - mean) / std
    # return as CHW for convenience when building windows
    return np.transpose(vol, (0, 1, 2))  # still [N,H,W]; windows will transpose to CHW later

def take_window_from_volume(vol_nhw: np.ndarray, center_idx: int, offsets=OFFSETS) -> np.ndarray:
    # vol_nhw: [N,H,W] float32
    N = vol_nhw.shape[0]
    idxs = [min(max(0, center_idx + o), N - 1) for o in offsets]
    win = vol_nhw[idxs, :, :]              # [len(offsets),H,W]
    return win.astype(np.float32, copy=False)

def coords_to_px(coords: np.ndarray, img_size: int) -> Tuple[int, int]:
    # coords are zeros on test; keep util for API compatibility
    x, y = float(coords[0]), float(coords[1])
    if 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
        x *= img_size; y *= img_size
    return int(round(x)), int(round(y))

def crop_and_resize_chw(img_chw: np.ndarray, x1: int, y1: int, x2: int, y2: int, out_size: int) -> np.ndarray:
    img_hwc = np.transpose(np.asarray(img_chw), (1, 2, 0))
    crop = img_hwc[y1:y2, x1:x2]
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        crop = img_hwc
    crop = crop.astype(np.float32, copy=False)
    crop = np.ascontiguousarray(crop)
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return np.transpose(crop, (2, 0, 1))

def window_to_full_and_roi(win_chw: np.ndarray, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if USE_ROI and np.any(coords != 0):
        cx, cy = coords_to_px(coords, IMG_SIZE)
        r = max(12, int(0.15 * IMG_SIZE))
        x1 = max(0, cx - r); y1 = max(0, cy - r)
        x2 = min(IMG_SIZE - 1, cx + r); y2 = min(IMG_SIZE - 1, cy + r)
        roi = crop_and_resize_chw(win_chw, x1, y1, x2, y2, IMG_SIZE)
        return win_chw, roi
    # No coords on test â†’ identical streams
    return win_chw, win_chw

# ========= Checkpoint discovery/loading =========
_ckpt_cache = None  # type: ignore
_models = None      # type: ignore

_ckpt_regex = re.compile(r"([^/\\]+)_hybrid_fold(\d+)\.pth$")

def discover_checkpoints() -> List[Tuple[str, str]]:
    # Returns list of (arch_name, path)
    found: List[Tuple[str, str]] = []
    for base in CANDIDATE_MODEL_DIRS:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            for f in files:
                if f.endswith('.pth') and ('_hybrid_fold' in f or '_best_wAUC' in f):
                    m = _ckpt_regex.search(f)
                    if m:
                        arch = m.group(1)
                    else:
                        # heuristic: arch is everything before first _fold or _hybrid
                        arch = f.split('_hybrid_fold')[0].split('_fold')[0]
                    found.append((arch, os.path.join(root, f)))
    # stable ordering
    found.sort(key=lambda x: x[1])
    return found

def load_hybrid_model(arch_name: str, weight_path: str) -> nn.Module:
    model = HybridAneurysmModel(base_model_name=arch_name, num_classes=len(LABEL_COLS))
    state = torch.load(weight_path, map_location=DEVICE)
    if isinstance(state, dict) and any(k.startswith('module.') for k in state.keys()):
        state = {k.replace('module.', '', 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval().to(DEVICE)
    return model

def get_models() -> List[Tuple[str, nn.Module]]:
    global _ckpt_cache, _models
    if _models is not None:
        return _models
    _ckpt_cache = discover_checkpoints()
    if not _ckpt_cache:
        raise FileNotFoundError('No model checkpoints found. Attach a dataset with *_hybrid_fold*.pth files.')
    mods: List[Tuple[str, nn.Module]] = []
    for arch, path in _ckpt_cache:
        try:
            m = load_hybrid_model(arch, path)
            mods.append((arch, m))
        except Exception:
            # skip incompatible files
            continue
    if not mods:
        raise RuntimeError('Failed to load any checkpoints from discovered files.')
    _models = mods
    print(f"Loaded {len(_models)} models")
    return _models

# ========= Per-series sliding-window inference =========
@torch.no_grad()
def predict_series_probs(dicoms) -> np.ndarray:
    models = get_models()
    # Build normalized volume [N,H,W]
    vol = series_to_tensor_chw(dicoms)
    N = vol.shape[0]
    # Prepare coords zeros on test
    coords = np.zeros((N, 2), dtype=np.float32)

    all_model_probs = []
    for _, model in models:
        batch_full, batch_roi, batch_coords = [], [], []
        probs_accum = []
        for c in range(N):
            win = take_window_from_volume(vol, c, OFFSETS)   # [C,H,W]
            win_chw = np.transpose(win, (0, 1, 2))           # still [C,H,W]
            full_chw, roi_chw = window_to_full_and_roi(win_chw, coords[c])
            batch_full.append(full_chw)
            batch_roi.append(roi_chw)
            batch_coords.append(coords[c])
            # flush by batch
            if len(batch_full) == BATCH_SIZE or c == N - 1:
                xb_full = torch.from_numpy(np.stack(batch_full).astype(np.float32)).to(DEVICE)
                xb_roi  = torch.from_numpy(np.stack(batch_roi).astype(np.float32)).to(DEVICE)
                cb      = torch.from_numpy(np.stack(batch_coords).astype(np.float32)).to(DEVICE)
                logits = model(xb_full, xb_roi, cb)
                probs = torch.sigmoid(logits).cpu().numpy()
                probs_accum.append(probs)
                batch_full.clear(); batch_roi.clear(); batch_coords.clear()
        probs_all = np.concatenate(probs_accum, axis=0) if probs_accum else np.zeros((1, len(LABEL_COLS)), dtype=np.float32)
        if AGGREGATE == 'max':
            series_prob = probs_all.max(axis=0)
        elif AGGREGATE == 'mean':
            series_prob = probs_all.mean(axis=0)
        else:  # topk_mean
            k = max(1, N // 5)
            series_prob = np.sort(probs_all, axis=0)[-k:].mean(axis=0)
        all_model_probs.append(series_prob)
        # free memory between models
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    # ensemble (probability average)
    return np.mean(np.stack(all_model_probs, axis=0), axis=0)

# ========= Kaggle-required predict(series_path) =========
def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
    series_id = os.path.basename(series_path)

    # Try reading just one DICOM to check the Modality
    first_dcm = None
    for root, _, files in os.walk(series_path):
        for f in files:
            if f.endswith('.dcm'):
                try:
                    first_dcm = pydicom.dcmread(os.path.join(root, f), stop_before_pixels=True)
                    break
                except Exception:
                    continue
        if first_dcm:
            break

    # Check modality
    modality = getattr(first_dcm, 'Modality', '').upper() if first_dcm else ''
    if modality != 'CT':
        zeros = [[series_id] + [0.0] * len(LABEL_COLS)]
        predictions = pl.DataFrame(data=zeros, schema=[ID_COL, *LABEL_COLS], orient='row')
        return predictions.drop(ID_COL)

    # Proceed with full DICOM loading and processing
    filepaths = []
    for root, _, files in os.walk(series_path):
        for f in files:
            if f.endswith('.dcm'):
                filepaths.append(os.path.join(root, f))
    dicoms = sort_dicom_slices(filepaths)

    # Inference
    probs = predict_series_probs(dicoms)

    # Build output (one row)
    data = [[series_id] + probs.tolist()]
    predictions = pl.DataFrame(data=data, schema=[ID_COL, *LABEL_COLS], orient='row')

    # Required cleanup to avoid disk pressure
    shutil.rmtree('/kaggle/shared', ignore_errors=True)

    # Server expects features only (without ID_COL)
    return predictions.drop(ID_COL)




SERIES_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/"
DESIRED_SERIES = "1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647"
#
import pandas as pd
import random
from IPython.display import display

# Load the data
df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")

# Filter only CTA modality
cta_df = df[df['Modality'] == 'CTA']

# Pick a random SeriesInstanceUID from CTA cases
DESIRED_SERIES = random.choice(cta_df['SeriesInstanceUID'].unique())

# Filter the CTA DataFrame for the selected SeriesInstanceUID
filtered_df = cta_df[cta_df['SeriesInstanceUID'] == DESIRED_SERIES].reset_index(drop=True)

# Display
print("Randomly selected SeriesInstanceUID:", DESIRED_SERIES)
display(filtered_df)




if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    p = predict(SERIES_PATH+DESIRED_SERIES)
    display(p)


def compute_weighted_auc(y_true, y_prob, class_names):
    """Robust wAUC: handles single-column vs multi-class shape mismatches.
    - If one of y_true/y_prob has 1 column and class_names has 14, align to AP only.
    - Otherwise, truncate to the minimum common number of classes.
    - Skips classes with single-label targets or constant predictions.
    Returns (weighted_auc, ap_auc, others_mean, per_class_auc_dict, skipped_names)
    """
    y_true = np.atleast_2d(np.asarray(y_true))
    y_prob = np.atleast_2d(np.asarray(y_prob))

    # Ensure 2D (N, C)
    if y_true.ndim != 2:
        y_true = y_true.reshape(y_true.shape[0], -1)
    if y_prob.ndim != 2:
        y_prob = y_prob.reshape(y_prob.shape[0], -1)

    C_true = y_true.shape[1]
    C_prob = y_prob.shape[1]
    C_names = len(class_names)

    ap_name = "Aneurysm Present"
    # Case: one side is binary (C==1) and the other has many classes â†’ evaluate AP only
    if (C_true == 1 and C_prob > 1) or (C_prob == 1 and C_true > 1) or (C_true == 1 and C_prob == 1 and C_names != 1):
        if ap_name in class_names and max(C_true, C_prob) >= 1:
            ap_idx = class_names.index(ap_name) if C_names > 1 else 0
            if C_true > 1:
                y_true = y_true[:, [ap_idx]] if ap_idx < C_true else y_true[:, [C_true - 1]]
            if C_prob > 1:
                y_prob = y_prob[:, [ap_idx]] if ap_idx < C_prob else y_prob[:, [C_prob - 1]]
            class_names = [ap_name]
        else:
            # Fallback: just keep the first column from both
            y_true = y_true[:, [0]]
            y_prob = y_prob[:, [0]]
            class_names = [ap_name] if C_names >= 1 else ["class0"]
    else:
        # General case: align by truncating to minimum number of classes
        K = min(C_true, C_prob, C_names)
        if K <= 0:
            raise ValueError(f"No common classes to evaluate: y_true {y_true.shape}, y_prob {y_prob.shape}, names {C_names}")
        if (C_true != K) or (C_prob != K) or (C_names != K):
            y_true = y_true[:, :K]
            y_prob = y_prob[:, :K]
            class_names = list(class_names)[:K]

    # Now shapes must match
    assert y_true.shape == y_prob.shape, f"shape mismatch after alignment: {y_true.shape} vs {y_prob.shape}"

    aucs, skipped = {}, []
    for i, name in enumerate(class_names):
        yi = y_true[:, i]
        pi = y_prob[:, i]
        # skip if only one class present, or predictions are constant
        if np.unique(yi).size < 2 or np.allclose(pi, pi[0]):
            skipped.append(name)
            continue
        try:
            aucs[name] = roc_auc_score(yi, pi)
        except ValueError:
            skipped.append(name)

    ap_auc = aucs.get(ap_name, np.nan)
    others = [v for k, v in aucs.items() if k != ap_name]
    others_mean = np.mean(others) if len(others) else np.nan

    # if either term is missing, use the one available
    if np.isnan(ap_auc) and np.isnan(others_mean):
        weighted_auc = np.nan
    elif np.isnan(ap_auc):
        weighted_auc = others_mean
    elif np.isnan(others_mean):
        weighted_auc = ap_auc
    else:
        weighted_auc = 0.5 * (ap_auc + others_mean)

    return weighted_auc, ap_auc, others_mean, aucs, skipped



"""
all_probs, all_targets = [], []
#y   = pd.to_numeric(filtered_df[LABEL_COLS], errors="coerce").fillna(0.0).values.astype(np.float32)
y = filtered_df[LABEL_COLS].apply(pd.to_numeric, errors="coerce").fillna(0.0).values.astype(np.float32)

all_probs.append(p)
all_targets.append(y)

print("y shape:", y.shape)
print("p shape:", p.shape)
print("Labels:", LABEL_COLS)
print("Unique values per class:")
for i, name in enumerate(LABEL_COLS):
    print(f"{name}: {np.unique(y[:, i])}")


wAUC, ap_auc, others_mean, per_class_auc, skipped = compute_weighted_auc(all_targets, all_probs, LABEL_COLS)
print(f"wAUC {wAUC:.4f} | AP {ap_auc:.4f} | others {others_mean:.4f}")
"""




"""
# ==== Multi-series evaluation (balanced sample) to avoid NaN AUCs ====
import os, random, numpy as np, pandas as pd

def df_to_numpy(df):
    try:
        import polars as pl
        if isinstance(df, pl.DataFrame):
            return df.to_numpy()
    except Exception:
        pass
    if isinstance(df, pd.DataFrame):
        return df.values
    return np.asarray(df)

# Load train metadata and filter CTA
train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
cta_df = train_df[train_df['Modality'] == 'CTA'].copy()

# Balanced sample of positives/negatives for "Aneurysm Present"
pos_ids = cta_df[cta_df['Aneurysm Present'] == 1][ID_COL].astype(str).tolist()
neg_ids = cta_df[cta_df['Aneurysm Present'] == 0][ID_COL].astype(str).tolist()

k_per_class = 50  # adjust if you want more/less
sample_ids = random.sample(pos_ids, min(k_per_class, len(pos_ids))) + \
             random.sample(neg_ids, min(k_per_class, len(neg_ids)))
random.shuffle(sample_ids)

all_probs, all_targets = [], []
failed = []
for sid in sample_ids:
    series_path = os.path.join(SERIES_PATH, sid)
    try:
        p_df = predict(series_path)  # returns 1x14 DataFrame without ID
        p = df_to_numpy(p_df).astype(np.float32)
        y = (cta_df.loc[cta_df[ID_COL].astype(str) == sid, LABEL_COLS]
                    .apply(pd.to_numeric, errors='coerce')
                    .fillna(0.0)
                    .values
                    .astype(np.float32))
        # flatten (1,14) -> (14,)
        if p.ndim == 2 and p.shape[0] == 1:
            p = p[0]
        if y.ndim == 2 and y.shape[0] == 1:
            y = y[0]
        all_probs.append(p)
        all_targets.append(y)
    except Exception as e:
        failed.append((sid, str(e)))

all_probs = np.asarray(all_probs, dtype=np.float32)
all_targets = np.asarray(all_targets, dtype=np.float32)

print("Eval N:", all_targets.shape, "Failed:", len(failed))
wAUC, ap_auc, others_mean, per_class_auc, skipped = compute_weighted_auc(all_targets, all_probs, LABEL_COLS)
print(f"wAUC {wAUC:.4f} | AP {ap_auc:.4f} | others {others_mean:.4f} | skipped: {skipped[:5]}{'...' if len(skipped)>5 else ''}")
"""


# ========= Start RSNA server =========
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))




