import os, sys, gc, math, random, json, time, warnings
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Detect Kaggle
ON_KAGGLE = Path("/kaggle/input").exists()
print("Running on Kaggle:", ON_KAGGLE)

# Paths
COMPETITION_SLUG = "byu-locating-bacterial-flagellar-motors-2025"
BASE_DIR = Path(f"/kaggle/input/{COMPETITION_SLUG}") if ON_KAGGLE else Path(f"./data/{COMPETITION_SLUG}")
OUTPUT_DIR = Path("/kaggle/working") if ON_KAGGLE else Path("./outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("BASE_DIR:", BASE_DIR.resolve())
print("OUTPUT_DIR:", OUTPUT_DIR.resolve())

# Filenames (will auto-detect, but you can override)
TRAIN_LABELS = BASE_DIR / "train_labels.csv"
SAMPLE_SUB = BASE_DIR / "sample_submission.csv"
TRAIN_DIR = BASE_DIR / "train"
TEST_DIR  = BASE_DIR / "test"

def log(msg: str):
    print(time.strftime("[%Y-%m-%d %H:%M:%S]"), msg)


# Check that folders/files exist
for p in [TRAIN_LABELS, SAMPLE_SUB, TRAIN_DIR, TEST_DIR]:
    log(f"{p} exists? {p.exists()}")

# Load sample submission if present
sample_sub = None
if SAMPLE_SUB.exists():
    sample_sub = pd.read_csv(SAMPLE_SUB)
    log(f"sample_submission shape: {sample_sub.shape}")
    display(sample_sub.head())
else:
    log("No sample_submission.csv found. We'll build IDs by scanning TEST_DIR.")

# Helper: get list of test tomo_ids
def discover_test_ids() -> List[str]:
    ids = []
    if sample_sub is not None and 'tomo_id' in sample_sub.columns:
        return sample_sub['tomo_id'].astype(str).tolist()
    # else discover by scanning test/
    if TEST_DIR.exists():
        # Case A: .npy volumes directly under test/
        for f in sorted(TEST_DIR.glob("*.npy")):
            ids.append(f.stem)
        # Case B: per-tomo subdirectories (with jpg slices)
        for sub in sorted(TEST_DIR.iterdir()):
            if sub.is_dir():
                jpgs = list(sub.glob("*.jpg")) + list(sub.glob("*.jpeg")) + list(sub.glob("*.png"))
                if jpgs:
                    ids.append(sub.name)
    return sorted(list(dict.fromkeys(ids)))

test_ids = discover_test_ids()
log(f"Discovered {len(test_ids)} test tomo_ids (first 5): {test_ids[:5]}")
assert len(test_ids) > 0, "No test tomograms found. Place files under BASE_DIR/test/ or provide sample_submission.csv."


from PIL import Image

def load_volume(tomo_id: str) -> np.ndarray:
    # Try to load a 3D volume for given tomo_id.
    # Priority: test/{tomo_id}.npy -> test/{tomo_id}/*.jpg as z-stack.
    # Returns float32 array (Z,Y,X) normalized to [0,1].
    npy_path = TEST_DIR / f"{tomo_id}.npy"
    if npy_path.exists():
        vol = np.load(npy_path)
        vol = np.asarray(vol, dtype=np.float32)
    else:
        slice_dir = TEST_DIR / tomo_id
        assert slice_dir.exists(), f"Cannot find npy or slice dir for {tomo_id}"
        slices = sorted(list(slice_dir.glob("*.jpg")) + list(slice_dir.glob("*.jpeg")) + list(slice_dir.glob("*.png")))
        assert len(slices) > 0, f"No slices found for {tomo_id}"
        imgs = []
        for p in slices:
            img = Image.open(p).convert("F")  # 32-bit float grayscale
            imgs.append(np.asarray(img, dtype=np.float32))
        vol = np.stack(imgs, axis=0)  # (Z,Y,X)

    # Normalize per volume robustly
    vmin, vmax = np.percentile(vol, [1, 99])
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(vol)) if np.isfinite(np.nanmin(vol)) else 0.0
        vmax = float(np.nanmax(vol)) if np.isfinite(np.nanmax(vol)) else 1.0
        if vmax <= vmin:
            vmax = vmin + 1.0
    vol = np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32)
    return vol


def predict_brightest_voxel(vol: np.ndarray, none_threshold: Optional[float]=None) -> Tuple[float,float,float]:
    # Returns (z, y, x) of brightest voxel. If max < threshold -> (-1,-1,-1).
    m = float(vol.max())
    if none_threshold is not None and m < none_threshold:
        return (-1.0, -1.0, -1.0)
    idx = int(np.argmax(vol))  # flattened index
    z, y, x = np.unravel_index(idx, vol.shape)
    return (float(z), float(y), float(x))


PRED_NONE_THRESHOLD = None  # Example: set to 0.02 to abstain on very dark volumes

rows = []
for i, tid in enumerate(test_ids):
    if (i % 25) == 0:
        log(f"Inferencing {i}/{len(test_ids)}...")
    vol = load_volume(tid)
    z, y, x = predict_brightest_voxel(vol, none_threshold=PRED_NONE_THRESHOLD)
    rows.append({"tomo_id": tid, "Motor axis 0": z, "Motor axis 1": y, "Motor axis 2": x})

submission = pd.DataFrame(rows)

# Align to sample submission column order if available
if 'sample_sub' in globals() and sample_sub is not None:
    expected_cols = list(sample_sub.columns)
    for c in ["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"]:
        assert c in submission.columns, f"Missing column {c}"
    submission = submission[expected_cols]

sub_path = (Path('/kaggle/working') if ON_KAGGLE else Path('./outputs')) / "submission.csv"
submission.to_csv(sub_path, index=False)
log(f"Saved submission to: {sub_path}")
display(submission.head())


N_SHOW = min(3, len(test_ids))

for tid in test_ids[:N_SHOW]:
    vol = load_volume(tid)
    z, y, x = predict_brightest_voxel(vol, none_threshold=PRED_NONE_THRESHOLD)
    z_int = int(z) if z >= 0 else vol.shape[0] // 2
    slice_img = vol[z_int]
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(slice_img, cmap="gray")
    if z >= 0:
        plt.scatter([x], [y], s=60, marker="x")
        plt.title(f"{tid}  z={z:.0f}, y={y:.0f}, x={x:.0f}")
    else:
        plt.title(f"{tid}  (no motor predicted)")
    plt.axis("off")
    plt.show()

