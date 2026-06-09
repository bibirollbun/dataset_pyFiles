# ============================================================
# STEP 1: RSNA IA — Balanced CTA+MRA Subset Builder
# Outputs: balanced_cta_mra_subset_50.csv
# ============================================================
import os
from pathlib import Path
import math
import json
import numpy as np
import pandas as pd

# ---------------------- CONFIG ----------------------
CONFIG = {
    "train_csv": "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv",
    "localizers_csv": "/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv",
    "segmentations_dir": "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations",
    "modalities_to_keep": ["CTA", "MRA"],
    "fractions": [0.20, 0.80],
    "seed": 18,
    "multiframe_uids_csv": None,
    "min_per_site_20": 3,
    "min_per_site_50": 6,
}

# Robust pathing
def _safe_path(p):
    p = Path(p)
    return p if p.exists() else Path("/kaggle/working") / p.name

train_csv = _safe_path(CONFIG["train_csv"])
localizers_csv = _safe_path(CONFIG["localizers_csv"])
segs_dir = Path(CONFIG["segmentations_dir"])

# Load data
train = pd.read_csv(train_csv)
loc = pd.read_csv(localizers_csv)

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
    'Anterior Communicating Artery', 'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery', 'Basilar Tip', 'Other Posterior Circulation',
]
GLOBAL_COL = 'Aneurysm Present'

# Keep only CTA/MRA
train = train[train['Modality'].isin(CONFIG["modalities_to_keep"])].copy().reset_index(drop=True)

# Flags
train['has_localizer'] = train['SeriesInstanceUID'].isin(loc['SeriesInstanceUID'])
if segs_dir.exists():
    seg_uids = set(p.stem for p in segs_dir.glob("*.nii*"))
else:
    seg_uids = set()
train['has_segmentation'] = train['SeriesInstanceUID'].isin(seg_uids)

# Helper: sample with priority
def _sample_class_with_priority(df_class: pd.DataFrame, n_target: int, seed: int):
    if n_target <= 0:
        return df_class.iloc[0:0].copy()

    counts_by_mod = df_class['Modality'].value_counts()
    total = counts_by_mod.sum()
    desired = {m: int(round(n_target * (counts_by_mod[m] / total))) for m in counts_by_mod.index}

    # Fix rounding drift
    drift = n_target - sum(desired.values())
    if drift != 0:
        order = counts_by_mod.sort_values(ascending=False).index.tolist()
        i = 0
        while drift != 0 and i < len(order):
            m = order[i]
            desired[m] += 1 if drift > 0 else -1
            drift += -1 if drift > 0 else 1
            i = (i + 1) % len(order)

    df_class = df_class.sort_values(['has_localizer', 'has_segmentation'], ascending=False)
    out_parts = []
    for m, want in desired.items():
        block = df_class[df_class['Modality'] == m]
        take = min(want, len(block))
        out_parts.append(block.head(take))

    sampled = pd.concat(out_parts).drop_duplicates(subset=['SeriesInstanceUID'])
    if len(sampled) < n_target:
        remain = df_class[~df_class['SeriesInstanceUID'].isin(sampled['SeriesInstanceUID'])]
        topup = remain.head(n_target - len(sampled))
        sampled = pd.concat([sampled, topup])

    return sampled.sample(frac=1.0, random_state=seed).reset_index(drop=True)

def _enforce_min_site_coverage(df_pos: pd.DataFrame, min_per_site: int, seed: int):
    if min_per_site <= 0 or df_pos.empty:
        return df_pos

    cov = {site: int(df_pos[site].sum()) for site in LABEL_COLS}
    pool = df_pos.sort_values(['has_localizer', 'has_segmentation'], ascending=False).copy()
    extra = []
    seen = set(df_pos['SeriesInstanceUID'])

    for site in LABEL_COLS:
        need = max(0, min_per_site - cov[site])
        if need == 0:
            continue
        cand = pool[(pool[site] == 1) & (~pool['SeriesInstanceUID'].isin(seen))]
        if not cand.empty:
            take = cand.head(need)
            extra.append(take)
            seen.update(take['SeriesInstanceUID'].tolist())
            cov[site] += len(take)

    if extra:
        boosted = pd.concat([df_pos] + extra, ignore_index=True)
        boosted = boosted.sort_values(['has_localizer', 'has_segmentation'], ascending=False)
        boosted = boosted.head(len(df_pos)).copy()
        return boosted.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return df_pos

def build_balanced_subset(df_all: pd.DataFrame, frac: float, seed: int, min_per_site: int = 0):
    df_all = df_all.copy()
    pos = df_all[df_all[GLOBAL_COL] == 1].copy()
    neg = df_all[df_all[GLOBAL_COL] == 0].copy()
    n_total = int(math.floor(len(df_all) * frac))
    n_per_class = min(len(pos), len(neg), n_total // 2)

    pos_s = _sample_class_with_priority(pos, n_per_class, seed=seed)
    neg_s = _sample_class_with_priority(neg, n_per_class, seed=seed)

    if min_per_site > 0:
        pos_s = _enforce_min_site_coverage(pos_s, min_per_site=min_per_site, seed=seed)

    subset = pd.concat([pos_s, neg_s], ignore_index=True)
    return subset.sample(frac=1.0, random_state=seed).reset_index(drop=True)

# Build 50% subset
out50 = build_balanced_subset(
    train,
    frac=CONFIG["fractions"][1],
    seed=CONFIG["seed"],
    min_per_site=CONFIG["min_per_site_50"]
)
out50.to_csv("balanced_cta_mra_subset_50.csv", index=False)

print("Subset created: balanced_cta_mra_subset_50.csv")
print(f"Total rows: {len(out50)}")


# ============================================================
# STEP 2: RSNA IA — 2.5D Preprocessing for "Teacher" Training
# (Updated: central Z-slab + thinning + light in-plane auto-crop)
# - Handles corrupt/empty DICOMs safely
# - Handles single-frame AND multi-frame series
# - Moderate Z-cropping (keeps central head/neck band)
# - CTA: 3 HU windows as RGB (multi-window)
# - MRA: z-score → clip → min-max, replicated to 3 channels
# - Saves per-slice SOPInstanceUIDs for localizer-guided training
# Produces:
#   /kaggle/working/preproc_teacher/{SeriesUID}.npz
#   /kaggle/working/preproc_teacher_index.csv
# ============================================================

# 1) Ensure DICOM + JPEG2000 decoding works on Kaggle
!pip install "numpy<2.0" "pandas<2.2.0" "scipy<1.13.0" \
             "pydicom" "pylibjpeg-libjpeg" "pylibjpeg-openjpeg" \
             -U --force-reinstall --no-deps -q

import os, sys, json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import cv2
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# Try importing decoding libs (safe no-op if missing)
try:
    import pylibjpeg  # noqa: F401
    import pylibjpeg_openjpeg  # noqa: F401
except Exception:
    pass

# -------------------- CONFIG --------------------
SERIES_ROOT = Path("/kaggle/input/rsna-intracranial-aneurysm-detection/series")

# Point this to your subset file (e.g. your balanced 50% subset)
SUBSET_CSV = Path("balanced_cta_mra_subset_50.csv")  # change to _20 if needed

OUT_DIR = Path("/kaggle/working/preproc_teacher")
OUT_DIR.mkdir(parents=True, exist_ok=True)

INDEX_CSV_OUT = Path("/kaggle/working/preproc_teacher/preproc_teacher_index.csv")

# Geometry / size
TARGET_HW = (224, 224)  # (H, W)

# Z-cropping / thinning
KEEP_Z_FRAC = 0.7          # if None, use adaptive fraction
ADAPTIVE_KEEP_Z = False
MIN_KEEP_FRAC = 0.50        # never keep less than 50% of slices
MAX_KEEP_FRAC = 0.80        # never keep more than 80% of slices
THINNING_STRIDE = 1         # keep every 2nd slice after mid-crop
MAX_SLICES = 160            # cap total slices per series; set None to disable

# In-plane auto-crop (light)
AUTO_CROP = True
AUTO_CROP_PAD = 8           # padding (pixels) around brain/body mask

# CTA HU windows (3-channel RGB)
CTA_WINDOWS_RGB = [
    (40.0, 400.0),   # soft / angiographic-ish
    (100.0, 700.0),  # vessel / contrast
    (600.0, 2800.0)  # bone
]

# MRA normalization
MRA_CLIP_STD = 5.0

# -------------------- Helpers --------------------

def _safe_pixel_spacing(ds):
    """Return (dy, dx) spacing with fallbacks."""
    ps = getattr(ds, "PixelSpacing", None)
    if ps is None:
        return 1.0, 1.0
    if isinstance(ps, (str, bytes)):
        vals = [float(x) for x in str(ps).replace("\\", " ").split()]
    else:
        vals = [float(x) for x in ps]
    if len(vals) == 1:
        vals = [vals[0], vals[0]]
    return float(vals[0]), float(vals[1])

def _safe_z_from_ds(ds):
    """
    Try to extract a sortable z-position:
    - Prefer ImagePositionPatient[2]
    - Fallback to InstanceNumber
    - As last resort, 0.0
    """
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None:
        try:
            if isinstance(ipp, (str, bytes)):
                vals = [float(x) for x in str(ipp).replace("\\", " ").split()]
            else:
                vals = [float(v) for v in ipp]
            if len(vals) >= 3:
                return float(vals[2])
        except Exception:
            pass
    # Fallback: InstanceNumber
    inst = getattr(ds, "InstanceNumber", None)
    try:
        return float(inst)
    except Exception:
        return 0.0

def _load_singleframe_series(files):
    """
    Load a single-frame DICOM series into a list of slices & SOP IDs.
    Returns:
        slices: list of np.ndarray [H,W] float32 (in HU or MR units)
        sops:   list of str SOPInstanceUID
        dz, dy, dx: spacing (approximate)
    """
    slices = []
    sops = []
    zs = []
    dy = dx = dz = 1.0

    for f in files:
        try:
            ds = pydicom.dcmread(str(f))
        except Exception:
            continue

        # Skip if no pixels
        if not hasattr(ds, "pixel_array"):
            continue

        try:
            px = ds.pixel_array
        except Exception:
            continue

        if px is None or px.size == 0:
            continue
        # handle RGB or other weird dims by squeezing
        if px.ndim > 2:
            # If [H,W,3], convert to grayscale; if [1,H,W] etc, squeeze
            if px.ndim == 3 and px.shape[-1] == 3:
                r, g, b = px[..., 0], px[..., 1], px[..., 2]
                px = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
            else:
                px = np.squeeze(px)
        if px.ndim != 2 or px.shape[0] == 0 or px.shape[1] == 0:
            continue

        try:
            slope = float(getattr(ds, "RescaleSlope", 1.0))
            inter = float(getattr(ds, "RescaleIntercept", 0.0))
        except Exception:
            slope, inter = 1.0, 0.0

        img = px.astype(np.float32) * slope + inter
        z = _safe_z_from_ds(ds)
        sop = str(getattr(ds, "SOPInstanceUID", f.name))

        slices.append(img)
        sops.append(sop)
        zs.append(z)

        # Spacings (rough)
        dy, dx = _safe_pixel_spacing(ds)
        try:
            dz = float(getattr(ds, "SliceThickness", 1.0))
        except Exception:
            dz = 1.0

    if len(slices) == 0:
        raise ValueError("No valid single-frame slices loaded")

    zs = np.array(zs, dtype=float)
    order = np.argsort(zs)
    slices_sorted = [slices[i] for i in order]
    sops_sorted = [sops[i] for i in order]
    return slices_sorted, sops_sorted, (dz, dy, dx)

def _load_multiframe_series(file_path):
    """
    Load a multi-frame DICOM file.
    Returns:
        slices: list of [H,W] float32 images (one per frame)
        sops:   list of SOPInstanceUID (repeated per frame)
        dz, dy, dx: spacing (approximate)
    """
    ds = pydicom.dcmread(str(file_path))

    if not hasattr(ds, "pixel_array"):
        raise ValueError("Multi-frame: no pixel_array")

    try:
        arr = ds.pixel_array
    except Exception as e:
        raise ValueError(f"Multi-frame pixel_array error: {e}")

    if arr is None or arr.size == 0:
        raise ValueError("Multi-frame: empty pixel_array")

    # shapes: [F,H,W] or [F,1,H,W] or [F,H,W,3]
    if arr.ndim == 4:
        # if [F,1,H,W] -> squeeze channel
        if arr.shape[1] in (1,):
            arr = arr[:, 0]
        elif arr.shape[-1] == 3:
            # convert to grayscale per frame
            arr_list = []
            for f in range(arr.shape[0]):
                frame = arr[f]
                if frame.ndim == 3 and frame.shape[-1] == 3:
                    r, g, b = frame[..., 0], frame[..., 1], frame[..., 2]
                    arr_list.append((0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32))
                else:
                    arr_list.append(np.squeeze(frame).astype(np.float32))
            arr = np.stack(arr_list, axis=0)
        else:
            arr = np.squeeze(arr)
    elif arr.ndim == 3:
        # [F,H,W] is fine
        pass
    else:
        # Unexpected
        arr = np.squeeze(arr)
        if arr.ndim != 3:
            raise ValueError(f"Unexpected multi-frame shape: {arr.shape}")

    F, H, W = arr.shape

    try:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        inter = float(getattr(ds, "RescaleIntercept", 0.0))
    except Exception:
        slope, inter = 1.0, 0.0

    vol = arr.astype(np.float32) * slope + inter

    # For per-frame z ordering, try PerFrameFunctionalGroupsSequence
    zs = None
    pffg = getattr(ds, "PerFrameFunctionalGroupsSequence", None)
    if pffg:
        zvals = []
        for f in pffg:
            pos_seq = f.get("PlanePositionSequence", None) or f.get("PlanePositionPatientSequence", None)
            if pos_seq:
                ipp = pos_seq[0].get("ImagePositionPatient", None)
                if ipp is not None:
                    try:
                        zvals.append(float(ipp[2]))
                    except Exception:
                        zvals.append(0.0)
                else:
                    zvals.append(0.0)
            else:
                zvals.append(0.0)
        if len(zvals) == F:
            zs = np.array(zvals, dtype=float)

    if zs is None:
        # fallback: order as-is
        zs = np.arange(F, dtype=float)

    order = np.argsort(zs)
    vol = vol[order]

    try:
        dy, dx = _safe_pixel_spacing(ds)
    except Exception:
        dy, dx = 1.0, 1.0
    try:
        dz = float(getattr(ds, "SliceThickness", 1.0))
    except Exception:
        dz = 1.0

    sop = str(getattr(ds, "SOPInstanceUID", file_path.name))
    sops = [sop for _ in range(vol.shape[0])]

    slices = [vol[i] for i in range(vol.shape[0])]
    return slices, sops, (dz, dy, dx)

def load_dicom_series(series_uid):
    """
    Load a DICOM series (single- or multi-frame) into:
        vol_hu: [Z,H,W] float32 (HU for CT, raw for MR)
        sops:   [Z] list of SOPInstanceUID strings
        spacing: (dz,dy,dx)
    """
    sdir = SERIES_ROOT / series_uid
    if not sdir.exists():
        raise FileNotFoundError(f"Series directory not found: {sdir}")

    files = sorted(list(sdir.glob("**/*.dcm")))
    if not files:
        raise FileNotFoundError(f"No DICOM files in {sdir}")

    # Peek first file header
    ds0 = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
    is_multi = hasattr(ds0, "NumberOfFrames") or hasattr(ds0, "PerFrameFunctionalGroupsSequence")

    if is_multi:
        slices, sops, spacing = _load_multiframe_series(files[0])
    else:
        slices, sops, spacing = _load_singleframe_series(files)

    if len(slices) == 0:
        raise ValueError(f"No valid slices loaded for series {series_uid}")

    vol = np.stack(slices, axis=0).astype(np.float32)  # [Z,H,W]
    return vol, sops, spacing

# --------- Normalization & geometry helpers ---------

def _hu_to01(x, lo, hi):
    x = np.clip(x, lo, hi)
    return (x - lo) / (hi - lo + 1e-6)

def normalize_cta_to_rgb(vol_hu):
    """
    CTA: apply 3 HU windows → stack as RGB channels.
    Input:
        vol_hu: [Z,H,W] float32 (HU)
    Output:
        vol_rgb: [Z,3,H,W] float32 in [0,1]
    """
    Z, H, W = vol_hu.shape
    out = np.empty((Z, 3, H, W), dtype=np.float32)
    for z in range(Z):
        slice_hu = vol_hu[z]
        chans = []
        for (lo, hi) in CTA_WINDOWS_RGB:
            x = np.clip(slice_hu, lo, hi)
            x = (x - lo) / (hi - lo + 1e-6)
            chans.append(x.astype(np.float32))
        out[z] = np.stack(chans, axis=0)
    return out

def normalize_mra_to_rgb(vol_raw):
    """
    MRA: z-score → clip → min-max; replicate to 3 channels.
    Input:
        vol_raw: [Z,H,W] float32
    Output:
        vol_rgb: [Z,3,H,W] float32 in [0,1]
    """
    v = vol_raw.astype(np.float32)
    m = float(v.mean())
    s = float(v.std() + 1e-6)
    v = (v - m) / s
    v = np.clip(v, -MRA_CLIP_STD, MRA_CLIP_STD)
    v = (v - v.min()) / (v.max() - v.min() + 1e-6)
    Z, H, W = v.shape
    vol_rgb = np.repeat(v[:, None, :, :], 3, axis=1)  # [Z,3,H,W]
    return vol_rgb

def compute_bbox_from_vol01(vol01, pad=AUTO_CROP_PAD):
    """
    Compute a 2D bbox (y0,y1,x0,x1) from a normalized [Z,H,W] volume.
    We OR across Z, then find non-zero area in XY.
    """
    Z, H, W = vol01.shape
    # Binary mask of "tissue" vs background
    mask = (vol01 > 0.05).astype(np.uint8)  # simple, robust threshold
    M = mask.any(axis=0).astype(np.uint8)   # [H,W]
    ys, xs = np.where(M > 0)
    if ys.size == 0:
        return None
    y0 = max(0, ys.min() - pad)
    y1 = min(H, ys.max() + pad + 1)
    x0 = max(0, xs.min() - pad)
    x1 = min(W, xs.max() + pad + 1)
    return (y0, y1, x0, x1)

def crop_with_bbox(vol, bbox):
    if bbox is None:
        return vol
    y0, y1, x0, x1 = bbox
    return vol[:, y0:y1, x0:x1]

def adaptive_keep_frac(Z):
    """
    Adaptive fraction of slices to keep based on depth,
    clamped into [MIN_KEEP_FRAC, MAX_KEEP_FRAC].
    """
    if Z <= 0:
        return 1.0
    frac = 200.0 / float(max(Z, 1))  # more aggressive for thick volumes
    frac = max(MIN_KEEP_FRAC, min(MAX_KEEP_FRAC, frac))
    return float(frac)

def keep_middle_z(vol, sops, frac=None):
    """
    Keep a central Z slab of the volume (and matching sops).
    Returns vol_mid, (z_start,z_end), sops_mid
    """
    Z = vol.shape[0]
    if Z == 0:
        raise ValueError("Empty volume in keep_middle_z")

    if frac is None:
        frac = adaptive_keep_frac(Z) if ADAPTIVE_KEEP_Z else MAX_KEEP_FRAC
    k = max(1, int(round(Z * frac)))
    k = min(Z, k)
    start = (Z - k) // 2
    end = start + k
    vol_mid = vol[start:end]
    sops_mid = sops[start:end]
    return vol_mid, (start, end), sops_mid

def thin_and_cap(vol, sops):
    """
    Apply thinning stride and cap the number of slices; keep sops aligned.
    """
    Z = vol.shape[0]
    idx = np.arange(Z, dtype=int)

    if THINNING_STRIDE and THINNING_STRIDE > 1:
        idx = idx[::THINNING_STRIDE]

    if MAX_SLICES is not None and len(idx) > MAX_SLICES:
        idx = np.linspace(0, len(idx) - 1, MAX_SLICES).round().astype(int)

    vol_out = vol[idx]
    sops_out = [sops[i] for i in idx]
    return vol_out, sops_out

def resize_volume(vol_3ch, target_hw):
    """
    Resize [Z,3,H,W] to [Z,3,target_H,target_W] using cv2.INTER_AREA.
    """
    Z, C, H, W = vol_3ch.shape
    th, tw = target_hw
    out = np.empty((Z, C, th, tw), dtype=np.float32)
    for z in range(Z):
        for c in range(C):
            out[z, c] = cv2.resize(vol_3ch[z, c], (tw, th), interpolation=cv2.INTER_AREA)
    return out

# -------------------- Main Preprocessing --------------------

if not SUBSET_CSV.exists():
    raise FileNotFoundError(
        f"Subset CSV not found at {SUBSET_CSV}. "
        "Make sure you have generated balanced_cta_mra_subset_XX.csv first."
    )

subset = pd.read_csv(SUBSET_CSV)
subset["Modality"] = subset["Modality"].astype(str).str.upper().str.strip()
subset = subset[subset["Modality"].isin(["CTA", "MRA"])].reset_index(drop=True)

print(f"Preprocessing {len(subset)} CTA/MRA series from {SUBSET_CSV.name} ...")

index_rows = []

for i, row in tqdm(subset.iterrows(), total=len(subset)):
    uid = str(row["SeriesInstanceUID"])
    mod = row["Modality"]

    try:
        # Load raw volume and SOPs
        vol_raw, sops, spacing = load_dicom_series(uid)   # [Z,H,W], list[str], (dz,dy,dx)
        Z0, H0, W0 = vol_raw.shape

        # ---- In-plane normalization for bbox (not final input) ----
        if mod == "CTA":
            # use the softest CTA window for a robust tissue mask
            lo_soft, hi_soft = CTA_WINDOWS_RGB[0]
            base01 = _hu_to01(vol_raw, lo_soft, hi_soft)
        else:
            # quick MRA norm: z-score + clip + min-max
            v = vol_raw.astype(np.float32)
            m = float(v.mean())
            s = float(v.std() + 1e-6)
            v = (v - m) / s
            v = np.clip(v, -MRA_CLIP_STD, MRA_CLIP_STD)
            base01 = (v - v.min()) / (v.max() - v.min() + 1e-6)

        # ---- In-plane auto-crop (XY) ----
        if AUTO_CROP:
            bbox = compute_bbox_from_vol01(base01, pad=AUTO_CROP_PAD)
            if bbox is not None:
                vol_raw = crop_with_bbox(vol_raw, bbox)
            else:
                bbox = (-1, -1, -1, -1)
        else:
            bbox = (-1, -1, -1, -1)

        # ---- Z cropping: keep central band ----
        vol_mid, z_bounds, sops_mid = keep_middle_z(vol_raw, sops, frac=KEEP_Z_FRAC)

        # ---- Thinning + cap ----
        vol_mid, sops_mid = thin_and_cap(vol_mid, sops_mid)

        # Final Z/H/W after cropping + thinning
        Z, H, W = vol_mid.shape

        if Z == 0 or H == 0 or W == 0:
            raise ValueError("Volume became empty after cropping / thinning")

        # ---- Final normalization to 3-channel [0,1] ----
        if mod == "CTA":
            vol_3ch = normalize_cta_to_rgb(vol_mid)       # [Z,3,H,W] in [0,1]
        else:
            vol_3ch = normalize_mra_to_rgb(vol_mid)       # [Z,3,H,W] in [0,1]

        # ---- Resize ----
        vol_resized = resize_volume(vol_3ch, TARGET_HW)   # [Z,3,Ht,Wt]

        # Store as uint8 to save space
        vol_uint8 = (np.clip(vol_resized, 0.0, 1.0) * 255.0).astype(np.uint8)

        # Safety: ensure sops length = Z; if mismatch, fix by clipping or repeating
        if len(sops_mid) != vol_uint8.shape[0]:
            if len(sops_mid) == 1:
                sops_mid = [sops_mid[0] for _ in range(vol_uint8.shape[0])]
            else:
                tmp = list(sops_mid)[:vol_uint8.shape[0]]
                if len(tmp) < vol_uint8.shape[0]:
                    while len(tmp) < vol_uint8.shape[0]:
                        tmp.append(tmp[-1])
                sops_mid = tmp

        npz_path = OUT_DIR / f"{uid}.npz"
        np.savez_compressed(
            npz_path,
            volume=vol_uint8,                          # [Z,3,H,W] uint8
            sops=np.array(sops_mid, dtype=object),     # [Z] object/str
            spacing=np.array(spacing, dtype=np.float32),  # (dz,dy,dx)
            modality=np.array(mod),
            series_uid=np.array(uid),
            bbox=np.array(bbox, dtype=np.int32),       # (y0,y1,x0,x1) or (-1,...)
            z_bounds=np.array(z_bounds, dtype=np.int32)  # (z_start,z_end) before thinning
        )

        index_rows.append({
            "SeriesInstanceUID": uid,
            "Modality": mod,
            "npz_path": str(npz_path),
            "z_depth": int(vol_uint8.shape[0]),
            "h": int(vol_uint8.shape[2]),
            "w": int(vol_uint8.shape[3]),
            "spacing_dz": float(spacing[0]),
            "spacing_dy": float(spacing[1]),
            "spacing_dx": float(spacing[2]),
            "bbox_y0": int(bbox[0]),
            "bbox_y1": int(bbox[1]),
            "bbox_x0": int(bbox[2]),
            "bbox_x1": int(bbox[3]),
            "z_start": int(z_bounds[0]),
            "z_end": int(z_bounds[1]),
            "z_orig": int(Z0),
        })

    except Exception as e:
        print(f"[WARN] {uid}: {type(e).__name__}: {e}", file=sys.stderr)
        continue

index_df = pd.DataFrame(index_rows)
index_df.to_csv(INDEX_CSV_OUT, index=False)

print("\n=== DONE ===")
print(f"Series processed: {len(index_df)} / {len(subset)}")
print(f"NPZ dir: {OUT_DIR}")
print(f"Index CSV: {INDEX_CSV_OUT}")




