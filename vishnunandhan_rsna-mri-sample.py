# === RSNA 2024: ONE SAMPLE PER CONDITION (from provided CSVs + competition images) ===
# Inputs:
#   - /kaggle/input/csv-files/*.csv (one CSV per condition, with study_id, series_id, instance_number, x, y, score/label)
#   - /kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/<study>/<series>/<instance>.dcm
#
# Outputs (per condition):
#   - /kaggle/working/condition_samples/<condition>_full.png  (full slice + red marker)
#   - /kaggle/working/condition_samples/<condition>_crop_224.png (224×224 crop around (x,y))
#   - /kaggle/working/condition_samples/condition_samples_manifest.csv

import os, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

CSV_DIR      = Path("/kaggle/input/csv-files")
IMG_ROOT     = Path("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images")
OUT_DIR      = Path("/kaggle/working/condition_samples")
CROP_SIZE    = 224  # output crop size
MARKER_PX    = 6
OUT_DIR.mkdir(parents=True, exist_ok=True)

def read_dicom_rgb(dcm_path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(dcm_path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:  # multiframe → take mid-frame
        arr = arr[arr.shape[0]//2]
    vmin, vmax = np.percentile(arr, [1, 99])
    arr = np.clip((arr - vmin) / max(vmax - vmin, 1e-6), 0, 1) * 255.0
    arr = arr.astype("uint8")
    return np.stack([arr]*3, axis=-1)  # RGB

def safe_crop_box(w, h, cx, cy, box):
    half = box // 2
    left   = int(max(0, min(cx - half, w - box)))
    top    = int(max(0, min(cy - half, h - box)))
    right  = left + box
    bottom = top  + box
    return left, top, right, bottom

def save_full_with_marker(img_rgb: np.ndarray, x: float, y: float, out_path: Path, marker=MARKER_PX):
    im = Image.fromarray(img_rgb)
    d  = ImageDraw.Draw(im)
    d.line((x-marker, y, x+marker, y), fill=(255,0,0), width=2)
    d.line((x, y-marker, x, y+marker), fill=(255,0,0), width=2)
    im.save(out_path, format="PNG", optimize=True)

def save_crop(img_rgb: np.ndarray, x: float, y: float, out_path: Path, crop_size=CROP_SIZE):
    H, W = img_rgb.shape[:2]
    left, top, right, bottom = safe_crop_box(W, H, x, y, min(crop_size, min(W, H)))
    patch = Image.fromarray(img_rgb[top:bottom, left:right])
    if patch.size != (crop_size, crop_size):
        patch = patch.resize((crop_size, crop_size), Image.BICUBIC)
    patch.save(out_path, format="PNG", optimize=True)

def normalize_colnames(df: pd.DataFrame) -> pd.DataFrame:
    # Make common cols accessible regardless of exact casing
    cmap = {c.lower(): c for c in df.columns}
    # Required columns (with fallback aliases)
    need = {
        "study_id": ["study_id","study","StudyInstanceUID"],
        "series_id": ["series_id","series","SeriesInstanceUID"],
        "instance_number": ["instance_number","image_id","sop_instance_uid","instance"],
        "x": ["x","X","coord_x","cx"],
        "y": ["y","Y","coord_y","cy"],
    }
    rename = {}
    for std, alts in need.items():
        for a in alts:
            if a.lower() in cmap:
                rename[cmap[a.lower()]] = std
                break
    df = df.rename(columns=rename)
    missing = [k for k in ["study_id","series_id","instance_number","x","y"] if k not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df

def resolve_instance_path(study_id: str, series_id: str, instance_id: str) -> Path:
    # Most RSNA slices are "<instance_number>.dcm". Some are SOPInstanceUID.dcm or without extension.
    candidates = [
        IMG_ROOT / study_id / series_id / f"{instance_id}.dcm",
        IMG_ROOT / study_id / series_id / f"{int(float(instance_id))}.dcm" if instance_id.replace('.','',1).isdigit() else None,
        IMG_ROOT / study_id / series_id / f"{instance_id}",
    ]
    for c in candidates:
        if c and c.exists() and c.is_file():
            return c
    # Fallback: first file that contains instance_id in name, else any .dcm
    series_dir = IMG_ROOT / study_id / series_id
    if series_dir.exists():
        hits = sorted([p for p in series_dir.glob("*") if instance_id in p.name])
        if hits:
            return hits[0]
        dcm_any = sorted(series_dir.glob("*.dcm"))
        if dcm_any:
            return dcm_any[len(dcm_any)//2]  # mid-slice fallback
    raise FileNotFoundError(f"Could not locate DICOM for {study_id}/{series_id}/{instance_id}")

# Gather CSVs (each assumed to correspond to a condition)
csv_files = sorted(glob.glob(str(CSV_DIR / "*.csv")))
if not csv_files:
    raise FileNotFoundError("No CSVs found in /kaggle/input/csv-files. Please add the dataset ‘reeyav/csv-files’ as an input.")

manifest = []
for csv_path in csv_files:
    cond_raw = Path(csv_path).stem  # use filename as condition name
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[c for c in df.columns if c.lower() in {"study_id","study","series_id","series","instance_number","image_id","x","y"}])
    df = normalize_colnames(df)

    if df.empty:
        print(f"[WARN] {cond_raw}: CSV has no usable rows.")
        continue

    # Choose one representative row (first). If you prefer, filter by score/label here.
    row = df.iloc[0]
    sid = str(row["study_id"])
    seid = str(row["series_id"])
    inst = str(row["instance_number"])
    x = float(row["x"]); y = float(row["y"])

    # Resolve and load image
    try:
        dcm_path = resolve_instance_path(sid, seid, inst)
    except Exception as e:
        print(f"[WARN] {cond_raw}: {e}")
        continue

    try:
        img = read_dicom_rgb(dcm_path)
    except Exception as e:
        print(f"[WARN] {cond_raw}: failed reading {dcm_path} — {e}")
        continue

    # Save images
    safe_cond = re.sub(r"[^a-zA-Z0-9_]+","_", cond_raw)[:60] or "condition"
    full_png = OUT_DIR / f"{safe_cond}_full.png"
    crop_png = OUT_DIR / f"{safe_cond}_crop_{CROP_SIZE}.png"
    save_full_with_marker(img, x, y, full_png)
    save_crop(img, x, y, crop_png, CROP_SIZE)

    print(f"Saved {cond_raw}:")
    print(f"  Full: {full_png}")
    print(f"  Crop: {crop_png}")

    manifest.append({
        "condition": cond_raw,
        "study_id": sid,
        "series_id": seid,
        "instance_number": inst,
        "x": x, "y": y,
        "dicom_path": str(dcm_path),
        "full_png": str(full_png),
        "crop_png": str(crop_png),
    })

# Write manifest and show a quick gallery of full-slices
man_csv = OUT_DIR / "condition_samples_manifest.csv"
pd.DataFrame(manifest).to_csv(man_csv, index=False)
print("\nManifest:", man_csv)

if manifest:
    cols = 3
    rows = int(np.ceil(len(manifest)/cols))
    plt.figure(figsize=(cols*4, rows*4))
    for i, rec in enumerate(manifest, 1):
        img = plt.imread(rec["full_png"])
        plt.subplot(rows, cols, i)
        plt.imshow(img)
        plt.title(rec["condition"], fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("No samples were saved — check CSV contents/column names.")



# === One cropped ROI per condition (VGG16-style) ============================
# Inputs:
#   - /kaggle/input/csv-files/*.csv (one CSV per condition; must include study_id, series_id, instance_number, x, y)
#   - /kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/<study>/<series>/<instance>.dcm
#
# Output (per condition):
#   /kaggle/working/vgg16_style_crops/<condition>_crop_224.png
#   plus a small gallery

import os, re, glob
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from PIL import Image

CSV_DIR   = Path("/kaggle/input/csv-files")
IMG_ROOT  = Path("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images")
OUT_DIR   = Path("/kaggle/working/vgg16_style_crops")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Helpers ---------------------------------------------------------------

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Map common aliases -> required cols."""
    cmap = {c.lower(): c for c in df.columns}
    need = {
        "study_id": ["study_id","study","StudyInstanceUID"],
        "series_id": ["series_id","series","SeriesInstanceUID"],
        "instance_number": ["instance_number","image_id","sop_instance_uid","instance"],
        "x": ["x","cx","coord_x"],
        "y": ["y","cy","coord_y"],
    }
    rename = {}
    for std, alts in need.items():
        for a in alts:
            if a.lower() in cmap:
                rename[cmap[a.lower()]] = std
                break
    df = df.rename(columns=rename)
    missing = [k for k in ["study_id","series_id","instance_number","x","y"] if k not in df.columns]
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    return df

def resolve_dicom(study_id: str, series_id: str, inst_id: str) -> Path:
    """Find the DICOM file for (study, series, instance)."""
    cands = [
        IMG_ROOT / study_id / series_id / f"{inst_id}.dcm",
        IMG_ROOT / study_id / series_id / f"{inst_id}",
    ]
    if inst_id.replace('.','',1).isdigit():
        cands.insert(1, IMG_ROOT / study_id / series_id / f"{int(float(inst_id))}.dcm")
    for c in cands:
        if c.exists():
            return c
    serdir = IMG_ROOT / study_id / series_id
    if serdir.exists():
        hits = sorted([p for p in serdir.glob("*") if inst_id in p.name])
        if hits: return hits[0]
        dcms = sorted(serdir.glob("*.dcm"))
        if dcms: return dcms[len(dcms)//2]
    raise FileNotFoundError(f"No DICOM for {study_id}/{series_id}/{inst_id}")

def read_dicom_uint8_rgb(path: Path) -> np.ndarray:
    """Load DICOM → robust 1–99% normalization → uint8 RGB (H,W,3)."""
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:  # multi-frame
        arr = arr[arr.shape[0]//2]
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
    img = arr.astype("uint8")
    return np.stack([img]*3, axis=-1)

def clamp_crop_box(w, h, cx, cy, box):
    half = box // 2
    left   = int(max(0, min(cx - half, w - box)))
    top    = int(max(0, min(cy - half, h - box)))
    right  = left + box
    bottom = top  + box
    return left, top, right, bottom

def crop_around_xy_to_224(img_rgb: np.ndarray, x: float, y: float,
                          crop_box: int = 128, out_size=(224,224)) -> Image.Image:
    """Center crop a square around (x,y) with clamping, then resize to 224×224 RGB."""
    H, W = img_rgb.shape[:2]
    box = min(crop_box, min(W, H))
    left, top, right, bottom = clamp_crop_box(W, H, x, y, box)
    patch = Image.fromarray(img_rgb[top:bottom, left:right])
    if patch.size != out_size:
        patch = patch.resize(out_size, Image.BICUBIC)
    if patch.mode != "RGB":
        patch = patch.convert("RGB")
    return patch

# --- Main: one cropped sample per condition --------------------------------

csv_files = sorted(glob.glob(str(CSV_DIR / "*.csv")))
if not csv_files:
    raise FileNotFoundError("No CSVs found in /kaggle/input/csv-files")

saved = []
for cpath in csv_files:
    cond_name = Path(cpath).stem
    try:
        df = pd.read_csv(cpath)
        df = df.dropna(how="any")  # ensure required fields present
        df = normalize_cols(df)
        if df.empty:
            print(f"[WARN] {cond_name}: CSV empty after cleaning")
            continue

        # pick ONE example (first row)
        row = df.iloc[0]
        sid, seid, inst = str(row["study_id"]), str(row["series_id"]), str(row["instance_number"])
        x, y = float(row["x"]), float(row["y"])

        dcm = resolve_dicom(sid, seid, inst)
        img = read_dicom_uint8_rgb(dcm)
        crop = crop_around_xy_to_224(img, x, y, crop_box=128, out_size=(224,224))

        safe = re.sub(r"[^a-zA-Z0-9_]+","_", cond_name)[:60] or "condition"
        out_png = OUT_DIR / f"{safe}_crop_224.png"
        crop.save(out_png, format="PNG", optimize=True)
        print(f"Saved: {cond_name} -> {out_png}")
        saved.append((cond_name, str(out_png)))
    except Exception as e:
        print(f"[WARN] Skipping {cond_name}: {e}")

# --- Tiny gallery -----------------------------------------------------------

if saved:
    cols = 3
    rows = int(np.ceil(len(saved)/cols))
    plt.figure(figsize=(cols*4, rows*4))
    for i, (name, p) in enumerate(saved, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(plt.imread(p))
        plt.title(name, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("No crops saved. Check CSV columns and dataset paths.")



# === RSNA 2024: One cropped ROI per condition =====================
# Conditions: Spinal Canal Stenosis, Left/Right Neural Foraminal Narrowing,
#             Left/Right Subarticular Stenosis
#
# Input:
#   - /kaggle/input/csv-files/*.csv  (one per condition; must include study_id, series_id, instance_number, x, y)
#   - /kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images/<study>/<series>/<instance>.dcm
#
# Output:
#   - /kaggle/working/rsna_condition_crops/<condition>_crop_224.png
#   - plus a gallery plot
# ==================================================================

import os, glob, re
from pathlib import Path
import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from PIL import Image

CSV_DIR   = Path("/kaggle/input/csv-files")
IMG_ROOT  = Path("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train_images")
OUT_DIR   = Path("/kaggle/working/rsna_condition_crops")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ---- Helpers ------------------------------------------------------

def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    cmap = {c.lower(): c for c in df.columns}
    need = {
        "study_id": ["study_id","study","StudyInstanceUID"],
        "series_id": ["series_id","series","SeriesInstanceUID"],
        "instance_number": ["instance_number","image_id","sop_instance_uid","instance"],
        "x": ["x","coord_x","cx"],
        "y": ["y","coord_y","cy"],
    }
    rename = {}
    for std, alts in need.items():
        for a in alts:
            if a.lower() in cmap:
                rename[cmap[a.lower()]] = std
                break
    df = df.rename(columns=rename)
    return df

def resolve_dicom(study_id: str, series_id: str, inst_id: str) -> Path:
    candidates = [
        IMG_ROOT / study_id / series_id / f"{inst_id}.dcm",
        IMG_ROOT / study_id / series_id / f"{inst_id}",
    ]
    if inst_id.replace('.','',1).isdigit():
        candidates.insert(1, IMG_ROOT / study_id / series_id / f"{int(float(inst_id))}.dcm")
    for c in candidates:
        if c.exists():
            return c
    serdir = IMG_ROOT / study_id / series_id
    if serdir.exists():
        hits = [p for p in serdir.glob("*") if inst_id in p.name]
        if hits: return hits[0]
        dcms = sorted(serdir.glob("*.dcm"))
        if dcms: return dcms[len(dcms)//2]
    raise FileNotFoundError(f"No DICOM for {study_id}/{series_id}/{inst_id}")

def read_dicom_uint8_rgb(path: Path) -> np.ndarray:
    ds = pydicom.dcmread(str(path))
    arr = ds.pixel_array.astype(np.float32)
    if arr.ndim == 3:
        arr = arr[arr.shape[0]//2]
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
    img = arr.astype("uint8")
    return np.stack([img]*3, axis=-1)

def clamp_crop_box(w, h, cx, cy, box):
    half = box // 2
    left   = int(max(0, min(cx - half, w - box)))
    top    = int(max(0, min(cy - half, h - box)))
    right  = left + box
    bottom = top  + box
    return left, top, right, bottom

def crop_around_xy(img_rgb: np.ndarray, x: float, y: float,
                   crop_box: int = 128, out_size=(224,224)) -> Image.Image:
    H, W = img_rgb.shape[:2]
    box = min(crop_box, min(W, H))
    left, top, right, bottom = clamp_crop_box(W, H, x, y, box)
    patch = Image.fromarray(img_rgb[top:bottom, left:right])
    if patch.size != out_size:
        patch = patch.resize(out_size, Image.BICUBIC)
    return patch.convert("RGB")

# ---- Main ---------------------------------------------------------

target_conditions = {
    "Spinal_Canal_Stenosis": "Spinal Canal Stenosis",
    "Left_Neural_Foraminal_Narrowing": "Left Neural Foraminal Narrowing",
    "Right_Neural_Foraminal_Narrowing": "Right Neural Foraminal Narrowing",
    "Left_Subarticular_Stenosis": "Left Subarticular Stenosis",
    "Right_Subarticular_Stenosis": "Right Subarticular Stenosis",
}

saved = []
for csv_path in glob.glob(str(CSV_DIR / "*.csv")):
    cond_file = Path(csv_path).stem
    if cond_file not in target_conditions:
        continue
    df = pd.read_csv(csv_path)
    df = df.dropna(how="any")
    df = normalize_cols(df)
    if df.empty: 
        continue
    row = df.iloc[0]
    sid, seid, inst = str(row["study_id"]), str(row["series_id"]), str(row["instance_number"])
    x, y = float(row["x"]), float(row["y"])

    dcm = resolve_dicom(sid, seid, inst)
    img = read_dicom_uint8_rgb(dcm)
    crop = crop_around_xy(img, x, y, crop_box=128, out_size=(224,224))

    out_name = cond_file + "_crop_224.png"
    out_path = OUT_DIR / out_name
    crop.save(out_path, format="PNG", optimize=True)

    print(f"Saved {cond_file} -> {out_path}")
    saved.append((target_conditions[cond_file], str(out_path)))

# ---- Gallery ------------------------------------------------------

if saved:
    cols = 3
    rows = int(np.ceil(len(saved)/cols))
    plt.figure(figsize=(cols*4, rows*4))
    for i, (name, p) in enumerate(saved, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(plt.imread(p))
        plt.title(name, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()



# ---- Main (fuzzy filename match; pick ONE sample per base condition) ----
import re, glob
from pathlib import Path

patterns = {
    r"\bspinal[_ ]?canal[_ ]?stenosis\b": "Spinal Canal Stenosis",
    r"\bneural[_ ]?foram(inal)?[_ ]?narrowing\b": "Neural Foraminal Narrowing",
    r"\bsubarticular[_ ]?stenosis\b": "Subarticular Stenosis",
}

# track if we've already saved a sample for a base condition
got = {pretty: False for pretty in patterns.values()}
saved = []

csv_paths = sorted(glob.glob(str(CSV_DIR / "*.csv")))
for csv_path in csv_paths:
    stem = Path(csv_path).stem.lower()

    # find which base condition this file belongs to (if any)
    pretty = None
    for pat, label in patterns.items():
        if re.search(pat, stem, flags=re.IGNORECASE):
            pretty = label
            break
    if pretty is None or got[pretty]:
        continue  # skip non-condition CSVs or already satisfied

    # load, normalize, choose one row, save crop
    df = pd.read_csv(csv_path)
    df = df.dropna(how="any")
    df = normalize_cols(df)
    if df.empty:
        continue

    row = df.iloc[0]
    sid, seid, inst = str(row["study_id"]), str(row["series_id"]), str(row["instance_number"])
    x, y = float(row["x"]), float(row["y"])

    dcm = resolve_dicom(sid, seid, inst)
    img = read_dicom_uint8_rgb(dcm)
    crop = crop_around_xy(img, x, y, crop_box=128, out_size=(224,224))

    safe = re.sub(r"[^a-zA-Z0-9_]+","_", pretty)[:60]
    out_path = OUT_DIR / f"{safe}_crop_224.png"
    crop.save(out_path, format="PNG", optimize=True)

    print(f"Saved {pretty} -> {out_path}  (from {Path(csv_path).name})")
    saved.append((pretty, str(out_path)))
    got[pretty] = True

# gallery (unchanged)



# ---- Five-condition saver: left/right aware --------------------------------
import re, glob
from pathlib import Path

# base patterns -> canonical base name
base_patterns = {
    r"\bspinal[_ ]?canal[_ ]?stenosis\b": "Spinal Canal Stenosis",
    r"\bneural[_ ]?foram(inal)?[_ ]?narrowing\b": "Neural Foraminal Narrowing",
    r"\bsubarticular[_ ]?stenosis\b": "Subarticular Stenosis",
}
sides = ["left","right"]

# Track what we've already saved (avoid duplicates)
done = set()
saved = []

for csv_path in sorted(glob.glob(str(CSV_DIR / "*.csv"))):
    stem = Path(csv_path).stem.lower()

    # Which base condition?
    base = None
    for pat, label in base_patterns.items():
        if re.search(pat, stem, flags=re.I):
            base = label
            break
    if base is None:
        continue

    # Side from filename if present
    side = None
    for s in sides:
        if re.search(rf"\b{s}\b", stem, flags=re.I):
            side = s.capitalize()
            break

    # Load/normalize
    df = pd.read_csv(csv_path)
    df = df.dropna(how="any")
    df = normalize_cols(df)

    # Try to read side from a column if not in filename
    if side is None:
        for col in df.columns:
            if col.lower() in {"side","laterality"}:
                val = str(df.iloc[0][col]).strip().lower()
                if val in {"left","right"}:
                    side = val.capitalize()
                break

    # Pick one row
    if df.empty:
        continue
    row = df.iloc[0]
    sid, seid, inst = str(row["study_id"]), str(row["series_id"]), str(row["instance_number"])
    x, y = float(row["x"]), float(row["y"])

    # Build label: include side when applicable
    label = f"{side} {base}" if (side and base != "Spinal Canal Stenosis") else base

    # Skip if we already saved this label
    if label in done:
        continue

    # Save crop
    dcm = resolve_dicom(sid, seid, inst)
    img = read_dicom_uint8_rgb(dcm)
    crop = crop_around_xy(img, x, y, crop_box=128, out_size=(224,224))

    safe = re.sub(r"[^a-zA-Z0-9_]+","_", label)[:60]
    out_path = OUT_DIR / f"{safe}_crop_224.png"
    crop.save(out_path, format="PNG", optimize=True)

    print(f"Saved {label} -> {out_path}  (from {Path(csv_path).name})")
    saved.append((label, str(out_path)))
    done.add(label)

# quick gallery
if saved:
    cols = 3
    rows = int(np.ceil(len(saved)/cols))
    plt.figure(figsize=(cols*4, rows*4))
    for i, (name, p) in enumerate(saved, 1):
        plt.subplot(rows, cols, i)
        plt.imshow(plt.imread(p))
        plt.title(name, fontsize=10)
        plt.axis('off')
    plt.tight_layout()
    plt.show()





