# Parallel precache RSNA series to .npy (image CHW float16 + coords float32)
# - Train coords from train_localizers.csv (mid-slice SOPInstanceUID). Missing â†’ zeros
# - Test coords zeros
# - Image: 2.5D (5 slices), per-series normalized, resized, CHW float16
# - Uses all CPUs via ProcessPoolExecutor (caps per-worker threads)
# - Cache dir: /kaggle/working/cache

import os
import ast
import cv2
import numpy as np
import pandas as pd
import pydicom
from tqdm.auto import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Sequence

# ========= Config =========
IMG_SIZE = 224 #456 for EfficientNetB5 , 224 for EfficientNetB0
SERIES_ROOT_TRAIN = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"
SERIES_ROOT_TEST  = "/kaggle/input/rsna-intracranial-aneurysm-detection/test/series"
TRAIN_CSV         = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
LOCALIZER_CSV     = "/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv"
CACHE_DIR_5c      = "/kaggle/working/cache"
os.makedirs(CACHE_DIR_5c, exist_ok=True)

ENABLE_ZIP_IMG = True
ALL_SLICES = True #True for all slices in a series, otherwise just 5 slices

DEBUG = True #True for precaching 5 series




# Load the CSV file
train_df = pd.read_csv(TRAIN_CSV)

# Check the value counts for the 'Modality' column
train_df['Modality'].value_counts()



# Cap intra-op threads inside workers to avoid oversubscription
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
try:
    cv2.setNumThreads(0)
except Exception:
    pass

NUM_CPUS = os.cpu_count() or 2
MAX_WORKERS = max(1, NUM_CPUS )  # NUM_CPUS - 1 to keep 1 core free



# ========= Helpers =========
def cache_paths(sid: str):
    img_ext = "npz" if ENABLE_ZIP_IMG else "npy"
    return (
        os.path.join(CACHE_DIR_5c, f"{sid}_img.{img_ext}"),
        os.path.join(CACHE_DIR_5c, f"{sid}_coords.npy"),
    )

def cache_paths_5c(sid: str):
    img_ext = "npz" if ENABLE_ZIP_IMG else "npy"
    return (
        os.path.join(CACHE_DIR_5c, f"{sid}_img.{img_ext}"),
        os.path.join(CACHE_DIR_5c, f"{sid}_coords.npy"),
    )

def sort_dicom_slices(filepaths):
    dicoms = [pydicom.dcmread(fp, force=True) for fp in filepaths]
    try:
        dicoms.sort(key=lambda d: float(d.ImagePositionPatient[2]))
    except Exception:
        dicoms.sort(key=lambda d: int(getattr(d, "InstanceNumber", 0)))
    return dicoms

def parse_coordinates(coord_str: str):
    d = ast.literal_eval(coord_str)
    return float(d["x"]), float(d["y"])

def normalize_and_save_coords(dicoms, loc_row, coord_path):
    import numpy as np
    if len(loc_row) > 0:
        coords = loc_row['coordinates'].values[0].astype(np.float32)
    else:
        coords = np.array([0.0, 0.0], dtype=np.float32)

    # Use middle slice geometry
    H0, W0 = dicoms[len(dicoms)//2].pixel_array.shape
    # If coords look like pixels (>1), convert to normalized [0,1]
    if np.max(coords) > 1.0:
        coords = np.array([coords[0] / W0, coords[1] / H0], dtype=np.float32)

    # Clamp to [0,1]
    coords = np.clip(coords, 0.0, 1.0)
    np.save(coord_path, coords)




# ========= Metadata =========
#train_df = pd.read_csv(TRAIN_CSV)
train_df = train_df[train_df["Modality"] == "CTA"].reset_index(drop=True)
total_count = len(train_df["SeriesInstanceUID"].astype(str).unique().tolist())
print(total_count)

train_sids = sorted(train_df["SeriesInstanceUID"].astype(str).unique().tolist())
if DEBUG:
    train_sids = train_sids[:10]


test_sids = []
if os.path.isdir(SERIES_ROOT_TEST):
    test_sids = sorted([d for d in os.listdir(SERIES_ROOT_TEST) if os.path.isdir(os.path.join(SERIES_ROOT_TEST, d))])

# SOPInstanceUID -> (x, y) mapping for fast lookup in workers
def to_xy_array(val):
    # handle NaN
    if pd.isna(val):
        return np.array([0.0, 0.0], dtype=np.float32)

    obj = val
    if isinstance(val, str):
        s = val.strip()
        if s == "":
            return np.array([0.0, 0.0], dtype=np.float32)
        try:
            obj = ast.literal_eval(s)
        except Exception:
            return np.array([0.0, 0.0], dtype=np.float32)

    if isinstance(obj, dict):
        x = obj.get("x") if "x" in obj else obj.get("X")
        y = obj.get("y") if "y" in obj else obj.get("Y")
        if x is None or y is None:
            return np.array([0.0, 0.0], dtype=np.float32)
        return np.array([float(x), float(y)], dtype=np.float32)

    if isinstance(obj, (list, tuple, np.ndarray)) and len(obj) >= 2:
        return np.array([float(obj[0]), float(obj[1])], dtype=np.float32)

    return np.array([0.0, 0.0], dtype=np.float32)

localizer_df = pd.read_csv(LOCALIZER_CSV)
localizer_df["coordinates"] = localizer_df["coordinates"].map(to_xy_array)




# ========= Worker =========
def precache_one(sid: str, is_train: bool, offsets: Sequence[int]) -> tuple[str, bool, str | None]:
    try:
        # choose cache path fn by number of slices requested
        use_5c = len(offsets) == 5
        cache_paths_fn = cache_paths_5c if use_5c else cache_paths

        img_path, coord_path = cache_paths_fn(sid)
        
        if os.path.exists(img_path) and os.path.exists(coord_path):
            return sid, True, None

        series_root = SERIES_ROOT_TRAIN if is_train else SERIES_ROOT_TEST
        series_path = os.path.join(series_root, sid)
        if not os.path.isdir(series_path):
            return sid, False, f"missing series folder"

        dcm_files = [os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith(".dcm")]
        if not dcm_files:
            return sid, False, "no dicoms"

        dicoms = sort_dicom_slices(dcm_files)
        n = len(dicoms)
        mid = n // 2

        # Map SOP -> slice index
        sop_to_idx = {str(getattr(d, "SOPInstanceUID", "")): i for i, d in enumerate(dicoms)}

        # Default center: series mid
        center_idx = mid
        picked_coords = None

        if is_train and os.path.exists(LOCALIZER_CSV):
            # localizers for this series and present in this series' SOPs
            cands = localizer_df[
                (localizer_df["SeriesInstanceUID"].astype(str) == sid) &
                (localizer_df["SOPInstanceUID"].astype(str).isin(sop_to_idx.keys()))
            ].copy()

            if len(cands) > 0:
                # If multiple, pick the one nearest series mid
                cands["idx"] = cands["SOPInstanceUID"].astype(str).map(sop_to_idx)
                pick = cands.iloc[(cands["idx"] - mid).abs().argmin()]
                center_idx = int(pick["idx"])

                coords = pick["coordinates"].astype(np.float32)  # already parsed to [x,y]
                H0, W0 = dicoms[center_idx].pixel_array.shape
                if np.max(coords) > 1.0:  # pixel -> normalized
                    coords = np.array([coords[0] / W0, coords[1] / H0], dtype=np.float32)
                picked_coords = np.clip(coords, 0.0, 1.0)

        # Build 5-slice indices around the chosen center
        if offsets is None:
            offsets = (-2, -1, 0, 1, 2)
        idxs = [min(max(0, center_idx + o), n - 1) for o in offsets]
            
        #idxs = [max(0, mid - 1), mid, min(len(dicoms) - 1, mid + 1)]
        
        # Build resized slice list to ensure consistent HxW before stacking
        slices_resized = []
        indices = range(len(dicoms)) if ALL_SLICES else idxs
        for i in indices:
            arr = dicoms[i].pixel_array
            if arr is None or arr.size == 0:
                return sid, False, "empty pixel_array"
            arr = arr.astype(np.float32)
            arr_resized = cv2.resize(arr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            slices_resized.append(arr_resized)

        if len(slices_resized) == 0:
            return sid, False, "no slices"

        img = np.stack(slices_resized, axis=-1)  # H,W,C (C = num slices used)
        img = (img - img.mean()) / (img.std() + 1e-6)

        img_chw = np.transpose(img, (2, 0, 1)).astype(np.float16)  # CHW float16
        if ENABLE_ZIP_IMG:
            np.savez_compressed(img_path, img_chw)
        else:
            np.save(img_path, img_chw)


        # Save coords: prefer picked_coords (centered on localizer slice); else fallback to mid-slice localizer; else zeros
        if picked_coords is not None:
            np.save(coord_path, picked_coords.astype(np.float32))
        else:
            mid_sop = str(getattr(dicoms[mid], "SOPInstanceUID", ""))
            if is_train and os.path.exists(LOCALIZER_CSV):
                loc_row = localizer_df[localizer_df["SOPInstanceUID"].astype(str) == mid_sop]
                normalize_and_save_coords(dicoms, loc_row, coord_path)
            else:
                np.save(coord_path, np.array([0.0, 0.0], dtype=np.float32))

        """
        if is_train and sop_to_coords:
            sop = getattr(dicoms[mid], "SOPInstanceUID", None)
            if sop is not None and str(sop) in sop_to_coords:
                cx, cy = sop_to_coords[str(sop)]
                coords = np.array([cx, cy], dtype=np.float32)
            else:
                coords = np.array([0.0, 0.0], dtype=np.float32)
        else:
            coords = np.array([0.0, 0.0], dtype=np.float32)
        """

        
        #np.save(coord_path, coords)
        return sid, True, None
    except Exception as e:
        return sid, False, str(e)

def parallel_precache(sids, is_train: bool, offsets, desc: str):
    failures = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(precache_one, sid, is_train, offsets): sid for sid in sids}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            sid, ok, err = fut.result()
            if not ok:
                failures.append((sid, err))
    return failures




# ========= Run =========
print(f"CPUs: {NUM_CPUS}, workers: {MAX_WORKERS}")
print(f"Caching to: {CACHE_DIR_5c}")

train_fail = parallel_precache(train_sids, is_train=True, offsets = (-2, -1, 0, 1, 2), desc="train cache")
if test_sids:
    test_fail  = parallel_precache(test_sids,  is_train=False, offsets = (-2, -1, 0, 1, 2), desc="test cache")
else:
    test_fail = []

print(f"Done. Train failures: {len(train_fail)} | Test failures: {len(test_fail)}")
if train_fail[:5]:
    print("Sample train failures:", train_fail[:5])
if test_fail[:5]:
    print("Sample test failures:", test_fail[:5])





if train_sids:
    s = train_sids[0]
    ip, cp = cache_paths(s)
    if os.path.exists(ip) and os.path.exists(cp):
        if ENABLE_ZIP_IMG and ip.endswith('.npz'):
            z = np.load(ip)
            a = z[list(z.files)[0]]  # stored array in .npz
        else:
            a = np.load(ip, mmap_mode="r")
        c = np.load(cp)
        print(f"Sample {s}: image {a.shape} {a.dtype}, coords {c.shape} {c.dtype}")

