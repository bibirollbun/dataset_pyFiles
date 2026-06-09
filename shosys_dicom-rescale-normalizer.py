# Notebook cell: run-this-as-is in Kaggle

from __future__ import annotations
import os, csv, shutil, logging
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pydicom
from pydicom.tag import Tag
from pydicom.uid import UID
from pydicom.filewriter import dcmwrite
from pydicom.valuerep import DSfloat
from pydicom.pixel_data_handlers.util import apply_modality_lut

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **kwargs): return x  # fallback

# -------------------- CONFIG (edit here) --------------------
MODE = "audit"                 # "audit" | "retag" | "bake"
INPUT_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"      # leave "" to auto-detect
OUTPUT_DIR = "/kaggle/working/dcm_rescale_out"
REPORT_CSV = "/kaggle/working/dcm_rescale_report.csv"

DESIRED_INTERCEPT = 0.0        # for retag/bake
DESIRED_SLOPE = 1.0            # for retag/bake
MODALITY_FILTER = "ANY"         # "CT" or "ANY"
COPY_UNCHANGED = True          # mirror unchanged files in retag/bake
CONDITIONAL_BAKE = True        # bake only when not already baked
DECOMPRESS = False             # bake compressed TS (needs gdcm/pylibjpeg)
VERBOSE = True                 # more logs
# ------------------------------------------------------------

# ---- logging (why: reproducible, quiet by default) ----
def setup_logger(verbose: bool) -> logging.Logger:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level,
                        format='[%(asctime)s] %(levelname)s - %(message)s',
                        datefmt='%H:%M:%S',
                        force=True)
    return logging.getLogger("dcm_rescaler")

logger = setup_logger(VERBOSE)

# ---- tags ----
MODALITY              = Tag(0x0008, 0x0060)
RESCALE_INTERCEPT     = Tag(0x0028, 0x1052)
RESCALE_SLOPE         = Tag(0x0028, 0x1053)
PIXEL_DATA            = Tag(0x7FE0, 0x0010)
BITS_ALLOCATED        = Tag(0x0028, 0x0100)
BITS_STORED           = Tag(0x0028, 0x0101)
HIGH_BIT              = Tag(0x0028, 0x0102)
PIXEL_REPRESENTATION  = Tag(0x0028, 0x0103)
SMALLEST_PIXEL_VALUE  = Tag(0x0028, 0x0106)
LARGEST_PIXEL_VALUE   = Tag(0x0028, 0x0107)

# ---- helpers ----
def auto_detect_input_dir(root="/kaggle/input") -> str:
    """Why: zero-config on Kaggle; picks the first folder containing .dcm."""
    for dirpath, _, files in os.walk(root):
        if any(f.lower().endswith(".dcm") for f in files):
            return dirpath
    raise FileNotFoundError(f"No DICOM files found under {root} (add a dataset as Input).")

def gather_dicoms(root: str) -> List[str]:
    out: List[str] = []
    for d, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".dcm"):
                out.append(os.path.join(d, f))
    return out

def modality_ok(ds: pydicom.Dataset, wanted: str) -> bool:
    if wanted.upper() == "ANY": return True
    return str(ds.get(MODALITY, "")).upper() == wanted.upper()

def is_compressed(ds: pydicom.Dataset) -> bool:
    ts: UID = ds.file_meta.TransferSyntaxUID
    return ts.is_compressed

def ensure_parent(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)

def rel_dst(src_root: str, dst_root: str, path: str) -> str:
    return os.path.join(dst_root, os.path.relpath(path, src_root))

def set_rescale_tags(ds: pydicom.Dataset, intercept: float, slope: float) -> bool:
    """Why: DSfloat preserves VR precision."""
    changed = False
    cur_i = float(ds.get(RESCALE_INTERCEPT, 0.0))
    cur_s = float(ds.get(RESCALE_SLOPE, 1.0))
    if cur_i != intercept:
        ds[RESCALE_INTERCEPT] = pydicom.DataElement(RESCALE_INTERCEPT, "DS", DSfloat(intercept))
        changed = True
    if cur_s != slope:
        ds[RESCALE_SLOPE] = pydicom.DataElement(RESCALE_SLOPE, "DS", DSfloat(slope))
        changed = True
    return changed

def is_already_baked(ds: pydicom.Dataset, mean_tol: float = 1e-5, max_tol: float = 1.0) -> bool:
    """Why: avoid rebaking HU; tolerate tiny rounding."""
    sv = ds.pixel_array
    hu = apply_modality_lut(sv, ds)
    if sv.shape != hu.shape: return False
    diff = np.asarray(hu, np.float32) - np.asarray(sv, np.float32)
    return (float(np.nanmean(np.abs(diff))) <= mean_tol) and (float(np.nanmax(np.abs(diff))) <= max_tol)

def to_int16_signed(arr: np.ndarray) -> Tuple[np.ndarray, bool]:
    a = arr.astype(np.float32)
    neg = bool(np.nanmin(a) < 0)
    a = np.nan_to_num(a, nan=0.0)
    a = np.clip(a, np.iinfo(np.int16).min, np.iinfo(np.int16).max)
    return a.astype(np.int16), neg

# ---- modes ----
def run_audit(input_dir: str, report_csv: str) -> None:
    files = gather_dicoms(input_dir)
    rows: List[Dict[str, Any]] = []
    counts: Dict[Tuple[str, float, float], int] = {}
    for p in tqdm(files, desc="Audit"):
        try:
            ds = pydicom.dcmread(p, stop_before_pixels=True, force=True)
            if not modality_ok(ds, MODALITY_FILTER): continue
            mod = str(ds.get(MODALITY, ""))
            slope = float(ds.get(RESCALE_SLOPE, 1.0))
            inter = float(ds.get(RESCALE_INTERCEPT, 0.0))
            rows.append({"path": p, "modality": mod, "slope": slope, "intercept": inter})
            counts[(mod, slope, inter)] = counts.get((mod, slope, inter), 0) + 1
        except Exception as e:
            rows.append({"path": p, "modality": "", "slope": "", "intercept": "", "error": str(e)})

    ensure_parent(report_csv)
    keys = sorted({k for r in rows for k in r.keys()})
    with open(report_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader(); w.writerows(rows)
    logger.info("Audit written: %s (rows=%d)", report_csv, len(rows))
    for (mod, s, i), n in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"{mod:>3} slope={s:<10} intercept={i:<10} count={n}")

def run_retag(input_dir: str, output_dir: str, report_csv: str) -> None:
    files = gather_dicoms(input_dir)
    rows: List[Dict[str, Any]] = []
    changed = unchanged = skipped = failed = 0

    for src in tqdm(files, desc="Retag"):
        dst = rel_dst(input_dir, output_dir, src)
        try:
            ds = pydicom.dcmread(src, force=True)  # keep PixelData intact; avoid pixel_array
            if not modality_ok(ds, MODALITY_FILTER):
                rows.append({"src": src, "dst": dst, "status":"skipped", "reason":"modality", "note":str(ds.get(MODALITY,""))})
                skipped += 1; continue

            did = set_rescale_tags(ds, DESIRED_INTERCEPT, DESIRED_SLOPE)
            ensure_parent(dst)
            dcmwrite(dst, ds, write_like_original=True)  # preserves raw encoding
            rows.append({"src": src, "dst": dst, "status":"changed" if did else "unchanged", "reason":"", "note":""})
            changed += int(did); unchanged += int(not did)

            if (not did) and COPY_UNCHANGED:
                try: shutil.copy2(src, dst)
                except Exception: pass
        except Exception as e:
            rows.append({"src": src, "dst": dst, "status":"failed", "reason":"exception", "note":str(e)})
            failed += 1

    ensure_parent(report_csv)
    with open(report_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["src","dst","status","reason","note"])
        w.writeheader(); w.writerows(rows)
    logger.info("Retag done. changed=%d, unchanged=%d, skipped=%d, failed=%d", changed, unchanged, skipped, failed)
    logger.info("Report: %s (rows=%d)", report_csv, len(rows))

def run_bake(input_dir: str, output_dir: str, report_csv: str) -> None:
    files = gather_dicoms(input_dir)
    rows: List[Dict[str, Any]] = []
    changed = unchanged = skipped = failed = 0

    for src in tqdm(files, desc="Bake"):
        dst = rel_dst(input_dir, output_dir, src)
        try:
            ds = pydicom.dcmread(src, force=True)
            if not modality_ok(ds, MODALITY_FILTER):
                rows.append({"src": src, "dst": dst, "status":"skipped", "reason":"modality", "note":str(ds.get(MODALITY,""))})
                skipped += 1; continue

            if CONDITIONAL_BAKE:
                try:
                    if is_already_baked(ds) and float(ds.get(RESCALE_SLOPE,1.0))==1.0 and float(ds.get(RESCALE_INTERCEPT,0.0))==0.0:
                        ensure_parent(dst)
                        if COPY_UNCHANGED: shutil.copy2(src, dst)
                        rows.append({"src": src, "dst": dst, "status":"unchanged", "reason":"conditional", "note":"already_baked"})
                        unchanged += 1; continue
                except Exception:
                    pass

            if is_compressed(ds):
                if not DECOMPRESS:
                    rows.append({"src": src, "dst": dst, "status":"skipped", "reason":"compressed", "note":str(ds.file_meta.TransferSyntaxUID)})
                    skipped += 1; continue
                try:
                    ds.decompress()
                except Exception as de:
                    rows.append({"src": src, "dst": dst, "status":"failed", "reason":"decompress_error", "note":str(de)})
                    failed += 1; continue

            sv = ds.pixel_array
            hu = apply_modality_lut(sv, ds).astype(np.float32)
            hu_i16, has_neg = to_int16_signed(hu)

            ds[PIXEL_DATA].value = hu_i16.tobytes()
            ds.Rows, ds.Columns = int(hu_i16.shape[-2]), int(hu_i16.shape[-1])
            ds[BITS_ALLOCATED] = pydicom.DataElement(BITS_ALLOCATED, "US", 16)
            ds[BITS_STORED]    = pydicom.DataElement(BITS_STORED,    "US", 16)
            ds[HIGH_BIT]       = pydicom.DataElement(HIGH_BIT,       "US", 15)
            ds[PIXEL_REPRESENTATION] = pydicom.DataElement(PIXEL_REPRESENTATION, "US", 1 if has_neg else 0)
            ds[SMALLEST_PIXEL_VALUE] = pydicom.DataElement(SMALLEST_PIXEL_VALUE, "SS" if has_neg else "US", int(np.min(hu_i16)))
            ds[LARGEST_PIXEL_VALUE]  = pydicom.DataElement(LARGEST_PIXEL_VALUE,  "SS" if has_neg else "US", int(np.max(hu_i16)))

            set_rescale_tags(ds, DESIRED_INTERCEPT, DESIRED_SLOPE)

            ensure_parent(dst)
            dcmwrite(dst, ds, write_like_original=False)
            rows.append({"src": src, "dst": dst, "status":"changed", "reason":"", "note":""})
            changed += 1
        except Exception as e:
            rows.append({"src": src, "dst": dst, "status":"failed", "reason":"exception", "note":str(e)})
            failed += 1

    ensure_parent(report_csv)
    with open(report_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["src","dst","status","reason","note"])
        w.writeheader(); w.writerows(rows)
    logger.info("Bake done. changed=%d, unchanged=%d, skipped=%d, failed=%d", changed, unchanged, skipped, failed)
    logger.info("Report: %s (rows=%d)", report_csv, len(rows))

# ---- runner ----
if not INPUT_DIR:
    INPUT_DIR = auto_detect_input_dir("/kaggle/input")
logger.info(f"Mode={MODE} | Input={INPUT_DIR} | Output={OUTPUT_DIR}")
if MODE == "audit":
    run_audit(INPUT_DIR, REPORT_CSV)
elif MODE == "retag":
    run_retag(INPUT_DIR, OUTPUT_DIR, REPORT_CSV)
elif MODE == "bake":
    run_bake(INPUT_DIR, OUTPUT_DIR, REPORT_CSV)
else:
    raise ValueError("MODE must be one of: 'audit', 'retag', 'bake'")



# File: update_dicom_rescale.py

import os
import pydicom

RESCALE_INTERCEPT_TAG = (0x0028, 0x1052)  # RescaleIntercept
RESCALE_SLOPE_TAG = (0x0028, 0x1053)      # RescaleSlope

def update_rescale_tags(filepath):
    try:
        ds = pydicom.dcmread(filepath)
        changed = False

        # Set or correct RescaleIntercept
        if RESCALE_INTERCEPT_TAG in ds:
            if float(ds[RESCALE_INTERCEPT_TAG].value) != 0.0:
                ds[RESCALE_INTERCEPT_TAG].value = "0"
                changed = True
        else:
            ds.add_new(RESCALE_INTERCEPT_TAG, "DS", "0")
            changed = True

        # Set or correct RescaleSlope
        if RESCALE_SLOPE_TAG in ds:
            if float(ds[RESCALE_SLOPE_TAG].value) != 1.0:
                ds[RESCALE_SLOPE_TAG].value = "1"
                changed = True
        else:
            ds.add_new(RESCALE_SLOPE_TAG, "DS", "1")
            changed = True

        if changed:
            ds.save_as(filepath)
            print(f"Updated: {filepath}")
        else:
            print(f"No change: {filepath}")

    except Exception as e:
        print(f"Failed to process {filepath}: {e}")

def process_dicom_directory(directory):
    for root, _, files in os.walk(directory):
        for fname in files:
            if fname.lower().endswith(".dcm"):
                fullpath = os.path.join(root, fname)
                update_rescale_tags(fullpath)

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python update_dicom_rescale.py <dicom_directory>")
    else:
        process_dicom_directory(sys.argv[1])

