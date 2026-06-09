# =======================
# 0. Setup & Extraction
# =======================

!pip install py7zr -q

import py7zr
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display

plt.style.use("ggplot")

DATA_DIR = Path("/kaggle/input/statoil-iceberg-classifier-challenge")
WORK_DIR = Path("/kaggle/working")

print("Files in DATA_DIR:")
for p in DATA_DIR.iterdir():
    print("  ", p)

train_7z = DATA_DIR / "train.json.7z"
test_7z  = DATA_DIR / "test.json.7z"

# Show archive contents (useful for debugging)
with py7zr.SevenZipFile(train_7z, mode="r") as z:
    print("\ntrain.json.7z members:", z.getnames())
with py7zr.SevenZipFile(test_7z, mode="r") as z:
    print("test.json.7z members:", z.getnames())

# Extract both archives to WORK_DIR
with py7zr.SevenZipFile(train_7z, mode="r") as z:
    z.extractall(path=WORK_DIR)
with py7zr.SevenZipFile(test_7z, mode="r") as z:
    z.extractall(path=WORK_DIR)

print("\nFiles in WORK_DIR after extraction:")
for p in WORK_DIR.iterdir():
    print("  ", p)

# Find train/test JSONs in /kaggle/working
json_files = list(WORK_DIR.rglob("*.json"))
if not json_files:
    raise FileNotFoundError("No .json files found in /kaggle/working after extraction.")

train_candidates = [p for p in json_files if "train" in p.stem.lower()]
test_candidates  = [p for p in json_files if "test"  in p.stem.lower()]

if not train_candidates:
    raise FileNotFoundError("Could not find a train JSON file (no name containing 'train').")
if not test_candidates:
    raise FileNotFoundError("Could not find a test JSON file (no name containing 'test').")

TRAIN_JSON = train_candidates[0]
TEST_JSON  = test_candidates[0]

print(f"\nUsing train JSON: {TRAIN_JSON}")
print(f"Using test JSON:  {TEST_JSON}")

train = pd.read_json(TRAIN_JSON)
test  = pd.read_json(TEST_JSON)

print("\nTrain shape:", train.shape)
print("Test shape:", test.shape)
display(train.head())

# =======================
# 1. Helper Functions
# =======================

IMG_SIZE = 75

def band_to_array(band_list):
    """Convert flattened list (length 75*75) into 2D array."""
    arr = np.array(band_list, dtype=np.float32).reshape(IMG_SIZE, IMG_SIZE)
    return arr

def to_float_angle(x):
    """Convert inc_angle to float, with 'na' -> np.nan."""
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan

def compute_band_features(df: pd.DataFrame) -> pd.DataFrame:
    """Global per-image statistics for band_1 & band_2."""
    stats = {
        "b1_mean": [],
        "b1_std": [],
        "b1_min": [],
        "b1_max": [],
        "b2_mean": [],
        "b2_std": [],
        "b2_min": [],
        "b2_max": [],
        "b12_mean": [],      # mean of (b1 + b2)/2
        "b12_std": [],
        "hv_ratio_mean": [], # mean of HV / (HH + HV + eps)
    }

    for _, row in df.iterrows():
        b1 = band_to_array(row["band_1"])
        b2 = band_to_array(row["band_2"])

        b12 = (b1 + b2) / 2.0
        H = b1 + b2
        eps = 1e-6
        hv_ratio = b2 / (H + eps)

        stats["b1_mean"].append(float(b1.mean()))
        stats["b1_std"].append(float(b1.std()))
        stats["b1_min"].append(float(b1.min()))
        stats["b1_max"].append(float(b1.max()))

        stats["b2_mean"].append(float(b2.mean()))
        stats["b2_std"].append(float(b2.std()))
        stats["b2_min"].append(float(b2.min()))
        stats["b2_max"].append(float(b2.max()))

        stats["b12_mean"].append(float(b12.mean()))
        stats["b12_std"].append(float(b12.std()))
        stats["hv_ratio_mean"].append(float(hv_ratio.mean()))

    stats_df = pd.DataFrame(stats, index=df.index)
    return stats_df

def compute_shape_and_object_features(df: pd.DataFrame, k: float = 0.5) -> pd.DataFrame:
    """
    Compute:
      - Simple blob shape features from band_1
      - Object vs background brightness and contrast in both bands
      - Object vs background HV/(HH+HV) difference
      - Object centroid distance from center

    Segmentation:
      mask = band_1 > (mean + k * std)  (bright blob)
    """
    total_pixels = IMG_SIZE * IMG_SIZE
    eps = 1e-6

    features = {
        "blob_area": [],
        "blob_aspect_ratio": [],
        "obj_frac": [],
        "obj_b1_mean": [],
        "bg_b1_mean": [],
        "obj_b2_mean": [],
        "bg_b2_mean": [],
        "obj_b1_std": [],
        "bg_b1_std": [],
        "obj_b2_std": [],
        "bg_b2_std": [],
        "obj_bg_b1_contrast": [],
        "obj_bg_b2_contrast": [],
        "obj_bg_b1_ratio": [],
        "obj_bg_b2_ratio": [],
        "obj_hv_ratio_mean": [],
        "bg_hv_ratio_mean": [],
        "obj_bg_hv_ratio_diff": [],
        "obj_centroid_dist": [],
    }

    center_y = (IMG_SIZE - 1) / 2.0
    center_x = (IMG_SIZE - 1) / 2.0

    for _, row in df.iterrows():
        b1 = band_to_array(row["band_1"])
        b2 = band_to_array(row["band_2"])

        mean = float(b1.mean())
        std  = float(b1.std())
        if std < 1e-6:
            th = mean
        else:
            th = mean + k * std

        mask = b1 > th
        area = int(mask.sum())
        obj_frac = area / total_pixels if total_pixels > 0 else 0.0

        if area == 0 or area == total_pixels:
            # Degenerate segmentation: no clear object vs background separation
            features["blob_area"].append(area)
            features["blob_aspect_ratio"].append(0.0)
            features["obj_frac"].append(obj_frac)
            for key in [
                "obj_b1_mean", "bg_b1_mean", "obj_b2_mean", "bg_b2_mean",
                "obj_b1_std",  "bg_b1_std",  "obj_b2_std",  "bg_b2_std",
                "obj_bg_b1_contrast", "obj_bg_b2_contrast",
                "obj_bg_b1_ratio",    "obj_bg_b2_ratio",
                "obj_hv_ratio_mean",  "bg_hv_ratio_mean",
                "obj_bg_hv_ratio_diff", "obj_centroid_dist",
            ]:
                features[key].append(np.nan)
            continue

        # Shape: aspect ratio
        ys, xs = np.where(mask)
        h = int(ys.max() - ys.min() + 1)
        w = int(xs.max() - xs.min() + 1)
        aspect = float(w / h) if h > 0 else 0.0

        # Object vs background indices
        obj_idx = mask
        bg_idx  = ~mask

        # Brightness statistics
        obj_b1 = b1[obj_idx]
        bg_b1  = b1[bg_idx]
        obj_b2 = b2[obj_idx]
        bg_b2  = b2[bg_idx]

        obj_b1_mean = float(obj_b1.mean())
        bg_b1_mean  = float(bg_b1.mean())
        obj_b2_mean = float(obj_b2.mean())
        bg_b2_mean  = float(bg_b2.mean())

        obj_b1_std = float(obj_b1.std())
        bg_b1_std  = float(bg_b1.std())
        obj_b2_std = float(obj_b2.std())
        bg_b2_std  = float(bg_b2.std())

        obj_bg_b1_contrast = obj_b1_mean - bg_b1_mean
        obj_bg_b2_contrast = obj_b2_mean - bg_b2_mean
        obj_bg_b1_ratio    = obj_b1_mean / (bg_b1_mean + eps)
        obj_bg_b2_ratio    = obj_b2_mean / (bg_b2_mean + eps)

        # HV ratio inside object vs background
        H  = b1 + b2
        hv_ratio = b2 / (H + eps)
        obj_hv_ratio_mean = float(hv_ratio[obj_idx].mean())
        bg_hv_ratio_mean  = float(hv_ratio[bg_idx].mean())
        obj_bg_hv_ratio_diff = obj_hv_ratio_mean - bg_hv_ratio_mean

        # Object centroid distance from center
        cy = float(ys.mean())
        cx = float(xs.mean())
        obj_centroid_dist = float(np.sqrt((cy - center_y) ** 2 + (cx - center_x) ** 2))

        # Store
        features["blob_area"].append(area)
        features["blob_aspect_ratio"].append(aspect)
        features["obj_frac"].append(obj_frac)

        features["obj_b1_mean"].append(obj_b1_mean)
        features["bg_b1_mean"].append(bg_b1_mean)
        features["obj_b2_mean"].append(obj_b2_mean)
        features["bg_b2_mean"].append(bg_b2_mean)

        features["obj_b1_std"].append(obj_b1_std)
        features["bg_b1_std"].append(bg_b1_std)
        features["obj_b2_std"].append(obj_b2_std)
        features["bg_b2_std"].append(bg_b2_std)

        features["obj_bg_b1_contrast"].append(obj_bg_b1_contrast)
        features["obj_bg_b2_contrast"].append(obj_bg_b2_contrast)
        features["obj_bg_b1_ratio"].append(obj_bg_b1_ratio)
        features["obj_bg_b2_ratio"].append(obj_bg_b2_ratio)

        features["obj_hv_ratio_mean"].append(obj_hv_ratio_mean)
        features["bg_hv_ratio_mean"].append(bg_hv_ratio_mean)
        features["obj_bg_hv_ratio_diff"].append(obj_bg_hv_ratio_diff)

        features["obj_centroid_dist"].append(obj_centroid_dist)

    return pd.DataFrame(features, index=df.index)

def show_examples(df: pd.DataFrame, n_icebergs: int = 3, n_ships: int = 3):
    """Show a few example images (HH, HV, and pseudo-RGB)."""
    fig, axes = plt.subplots(
        n_icebergs + n_ships, 3, figsize=(9, 3 * (n_icebergs + n_ships))
    )

    icebergs = df[df["is_iceberg"] == 1].sample(n=n_icebergs, random_state=42)
    ships    = df[df["is_iceberg"] == 0].sample(n=n_ships,    random_state=42)
    rows = pd.concat([icebergs, ships])

    for i, (_, row) in enumerate(rows.iterrows()):
        b1 = band_to_array(row["band_1"])
        b2 = band_to_array(row["band_2"])
        rgb = np.dstack([
            (b1 - b1.min()) / (b1.max() - b1.min() + 1e-6),
            (b2 - b2.min()) / (b2.max() - b2.min() + 1e-6),
            (b1 + b2 - (b1 + b2).min())
            / ((b1 + b2).max() - (b1 + b2).min() + 1e-6),
        ])

        ax1, ax2, ax3 = axes[i]

        ax1.imshow(b1, cmap="gray")
        ax1.set_title(f"Band 1 (HH)\nlabel={row['is_iceberg']}")
        ax1.axis("off")

        ax2.imshow(b2, cmap="gray")
        ax2.set_title("Band 2 (HV)")
        ax2.axis("off")

        ax3.imshow(rgb)
        ax3.set_title("Pseudo-RGB (HH, HV, mean)")
        ax3.axis("off")

    plt.tight_layout()
    plt.show()


# ============ 1.1 TABLE KEYS (HUMAN-FRIENDLY LEGENDS) ============

def print_angle_table_key():
    print("\nKey for 'Incidence angle stats by label' table:")
    print("  - Each row index is `is_iceberg` (0 = ship, 1 = iceberg).")
    print("  - Each cell can be read as: [is_iceberg, statistic on incidence angle].")
    print("  - Columns:")
    print("      count  : number of samples in this class with a valid incidence angle.")
    print("      mean   : [is_iceberg, average incidence angle in degrees].")
    print("      median : [is_iceberg, median incidence angle in degrees].")
    print("      std    : [is_iceberg, standard deviation of incidence angle].")
    print("      min    : [is_iceberg, minimum incidence angle].")
    print("      max    : [is_iceberg, maximum incidence angle].")

def print_band_table_key():
    print("\nKey for 'Global band statistics by label' table:")
    print("  - Each row index is `is_iceberg` (0 = ship, 1 = iceberg).")
    print("  - Each cell can be read as: [is_iceberg, statistic of a global image feature].")
    print("  - The columns are a two-level index: (feature_name, statistic).")
    print("    Where statistic is one of: mean, std.")
    print("  - feature_name meanings:")
    print("      b1_mean       : per-image mean brightness of band_1 (HH polarization), averaged over all pixels.")
    print("      b1_std        : per-image std of brightness in band_1 (HH).")
    print("      b2_mean       : per-image mean brightness of band_2 (HV polarization).")
    print("      b2_std        : per-image std of brightness in band_2 (HV).")
    print("      b12_mean      : per-image mean of (band_1 + band_2)/2, i.e. combined brightness.")
    print("      b12_std       : per-image std of (band_1 + band_2)/2.")
    print("      hv_ratio_mean : per-image mean of HV / (HH + HV), i.e. average polarization ratio.")
    print("  - Example cell: (row=1, col=('b1_mean','mean'))")
    print("      → [is_iceberg=1, average over all iceberg images of their HH mean brightness].")

def print_obj_table_key():
    print("\nKey for 'Object/background feature statistics by label' table:")
    print("  - Each row index is `is_iceberg` (0 = ship, 1 = iceberg).")
    print("  - Each cell can be read as: [is_iceberg, statistic of an object/background feature].")
    print("  - The columns are a two-level index: (feature_name, statistic).")
    print("    Where statistic is usually: mean, std.")
    print("  - feature_name meanings (object is the bright blob in band_1; background is the rest):")
    print("      blob_area              : number of pixels in the bright object (approximate target size).")
    print("      blob_aspect_ratio      : width / height of the object's bounding box (elongation, >1 = more stretched horizontally).")
    print("      obj_frac               : object area / total image area (fraction of image occupied by the object).")
    print("      obj_bg_b1_contrast     : obj_mean(HH) - bg_mean(HH); how much brighter the object is than ocean in HH.")
    print("      obj_bg_b2_contrast     : obj_mean(HV) - bg_mean(HV); how much brighter the object is than ocean in HV.")
    print("      obj_bg_b1_ratio        : obj_mean(HH) / bg_mean(HH); brightness ratio (HH) object vs background.")
    print("      obj_bg_b2_ratio        : obj_mean(HV) / bg_mean(HV); brightness ratio (HV) object vs background.")
    print("      obj_hv_ratio_mean      : mean HV/(HH+HV) inside the object (polarization ratio of object).")
    print("      bg_hv_ratio_mean       : mean HV/(HH+HV) in background (polarization ratio of ocean).")
    print("      obj_bg_hv_ratio_diff   : obj_hv_ratio_mean - bg_hv_ratio_mean; how different object's polarization is vs ocean.")
    print("      obj_centroid_dist      : distance from object centroid to image center (in pixels).")
    print("  - Example cell: (row=0, col=('obj_bg_b1_contrast','mean'))")
    print("      → [is_iceberg=0, average HH brightness contrast between ship and surrounding ocean].")

# =======================
# 2. EDA Flow
# =======================

# --- Basic label & incidence angle analysis ---
label_counts = train["is_iceberg"].value_counts().sort_index()
label_ratio = label_counts / label_counts.sum()

print("\nLabel counts:")
print(label_counts)
print("\nLabel ratios:")
print(label_ratio)

train["inc_angle_float"] = train["inc_angle"].map(to_float_angle)
test["inc_angle_float"]  = test["inc_angle"].map(to_float_angle)

n_missing_angle_train = train["inc_angle_float"].isna().sum()
n_missing_angle_test  = test["inc_angle_float"].isna().sum()

print("\nMissing inc_angle in train:", n_missing_angle_train)
print("Missing inc_angle in test:", n_missing_angle_test)

angle_stats_by_label = (
    train.groupby("is_iceberg")["inc_angle_float"]
    .agg(["count", "mean", "median", "std", "min", "max"])
)
print("\nIncidence angle stats by label:")
display(angle_stats_by_label)
print_angle_table_key()

# Plots: class counts & angle KDE
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].bar(["Ship (0)", "Iceberg (1)"], label_counts.values)
axes[0].set_title("Class Counts")
axes[0].set_ylabel("Count")

for label, grp in train.groupby("is_iceberg"):
    grp["inc_angle_float"].dropna().plot(
        kind="kde",
        ax=axes[1],
        label=f"is_iceberg={label}",
    )

axes[1].set_title("Incidence Angle Distribution (Train)")
axes[1].set_xlabel("Incidence angle (degrees)")
axes[1].legend()

plt.tight_layout()
plt.show()

# --- Global band stats ---
print("\nComputing per-image band statistics for train...")
train_band_stats = compute_band_features(train)
train = pd.concat([train, train_band_stats], axis=1)

print("Computing per-image band statistics for test...")
test_band_stats = compute_band_features(test)
test = pd.concat([test, test_band_stats], axis=1)

band_summary_by_label = (
    train.groupby("is_iceberg")[
        [
            "b1_mean", "b1_std", "b2_mean", "b2_std",
            "b12_mean", "b12_std", "hv_ratio_mean"
        ]
    ]
    .agg(["mean", "std"])
)
print("\nGlobal band statistics by label:")
display(band_summary_by_label)
print_band_table_key()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for label, grp in train.groupby("is_iceberg"):
    grp["b1_mean"].plot(kind="kde", ax=axes[0], label=f"is_iceberg={label}")
axes[0].set_title("Distribution of Band 1 Mean (HH)")
axes[0].set_xlabel("Band 1 mean (dB)")
axes[0].legend()

for label, grp in train.groupby("is_iceberg"):
    grp["b2_mean"].plot(kind="kde", ax=axes[1], label=f"is_iceberg={label}")
axes[1].set_title("Distribution of Band 2 Mean (HV)")
axes[1].set_xlabel("Band 2 mean (dB)")
axes[1].legend()

for label, grp in train.groupby("is_iceberg"):
    grp["hv_ratio_mean"].plot(kind="kde", ax=axes[2], label=f"is_iceberg={label}")
axes[2].set_title("Distribution of HV/(HH+HV) Mean")
axes[2].set_xlabel("Mean hv_ratio")
axes[2].legend()

plt.tight_layout()
plt.show()

# --- Shape + object/background features ---
print("\nComputing shape + object/background features for train...")
train_obj = compute_shape_and_object_features(train)
train = pd.concat([train, train_obj], axis=1)

print("Computing shape + object/background features for test...")
test_obj = compute_shape_and_object_features(test)
test = pd.concat([test, test_obj], axis=1)

obj_summary_by_label = (
    train.groupby("is_iceberg")[
        [
            "blob_area", "blob_aspect_ratio", "obj_frac",
            "obj_bg_b1_contrast", "obj_bg_b2_contrast",
            "obj_bg_b1_ratio", "obj_bg_b2_ratio",
            "obj_hv_ratio_mean", "bg_hv_ratio_mean",
            "obj_bg_hv_ratio_diff", "obj_centroid_dist",
        ]
    ]
    .agg(["mean", "std"])
)
print("\nObject/background feature statistics by label:")
display(obj_summary_by_label)
print_obj_table_key()

# Plots for object-background brightness features
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

for label, grp in train.groupby("is_iceberg"):
    grp["obj_bg_b1_contrast"].dropna().plot(
        kind="kde", ax=axes[0], label=f"is_iceberg={label}"
    )
axes[0].set_title("Obj-Bg Contrast in Band 1 (HH)")
axes[0].set_xlabel("obj_mean(HH) - bg_mean(HH)")
axes[0].legend()

for label, grp in train.groupby("is_iceberg"):
    grp["obj_bg_b2_contrast"].dropna().plot(
        kind="kde", ax=axes[1], label=f"is_iceberg={label}"
    )
axes[1].set_title("Obj-Bg Contrast in Band 2 (HV)")
axes[1].set_xlabel("obj_mean(HV) - bg_mean(HV)")
axes[1].legend()

for label, grp in train.groupby("is_iceberg"):
    grp["obj_bg_hv_ratio_diff"].dropna().plot(
        kind="kde", ax=axes[2], label=f"is_iceberg={label}"
    )
axes[2].set_title("Obj-Bg HV Ratio Difference")
axes[2].set_xlabel("obj_mean(HV/(HH+HV)) - bg_mean(HV/(HH+HV))")
axes[2].legend()

plt.tight_layout()
plt.show()

# Plots: shape distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 4))

for label, grp in train.groupby("is_iceberg"):
    grp["blob_area"].plot(kind="kde", ax=axes[0], label=f"is_iceberg={label}")
axes[0].set_title("Distribution of Blob Area (pixels)")
axes[0].set_xlabel("Blob area")
axes[0].legend()

for label, grp in train.groupby("is_iceberg"):
    grp["blob_aspect_ratio"].plot(kind="kde", ax=axes[1], label=f"is_iceberg={label}")
axes[1].set_title("Distribution of Blob Aspect Ratio (width/height)")
axes[1].set_xlabel("Aspect ratio")
axes[1].legend()

for label, grp in train.groupby("is_iceberg"):
    grp["obj_frac"].plot(kind="kde", ax=axes[2], label=f"is_iceberg={label}")
axes[2].set_title("Object Area Fraction")
axes[2].set_xlabel("Object pixels / total pixels")
axes[2].legend()

plt.tight_layout()
plt.show()

# Example images
show_examples(train)

# Correlations (include new features)
corr_cols = [
    "is_iceberg",
    "inc_angle_float",
    "b1_mean", "b1_std",
    "b2_mean", "b2_std",
    "b12_mean", "b12_std",
    "hv_ratio_mean",
    "blob_area", "blob_aspect_ratio", "obj_frac",
    "obj_bg_b1_contrast", "obj_bg_b2_contrast",
    "obj_bg_b1_ratio", "obj_bg_b2_ratio",
    "obj_hv_ratio_mean", "bg_hv_ratio_mean",
    "obj_bg_hv_ratio_diff",
    "obj_centroid_dist",
]

corr_df = train[corr_cols].corr()
corr_with_target = corr_df["is_iceberg"].sort_values(ascending=False)
print("\nCorrelation with is_iceberg (including new features):")
print(corr_with_target)

plt.figure(figsize=(8, 10))
corr_with_target.drop("is_iceberg").plot(kind="barh")
plt.title("Correlation of Features with is_iceberg")
plt.xlabel("Pearson correlation")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()



import os, json, random, math, subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, roc_curve, auc, confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

import matplotlib.pyplot as plt

# -------------------------
# 0. Helper: seeding
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

BASE_SEED = 42
set_seed(BASE_SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------
# 1. Extract JSONs from 7z (into /data/processed)
# -------------------------
TRAIN_JSON = "/data/processed/train.json"
TEST_JSON  = "/data/processed/test.json"

if not os.path.exists(TRAIN_JSON) or not os.path.exists(TEST_JSON):
    subprocess.run([
        "7z", "x",
        "/kaggle/input/statoil-iceberg-classifier-challenge/train.json.7z",
        "-o/",
        "-y"
    ], check=True)
    subprocess.run([
        "7z", "x",
        "/kaggle/input/statoil-iceberg-classifier-challenge/test.json.7z",
        "-o/",
        "-y"
    ], check=True)

assert os.path.exists(TRAIN_JSON), f"Train json not found at {TRAIN_JSON}"
assert os.path.exists(TEST_JSON),  f"Test json not found at {TEST_JSON}"

# -------------------------
# 2. Load and preprocess JSON data
# -------------------------
train_df = pd.read_json(TRAIN_JSON)
test_df  = pd.read_json(TEST_JSON)

IMG_SIZE = 75

def extract_bands(df):
    band1 = np.stack(df["band_1"].apply(lambda x: np.array(x).reshape(IMG_SIZE, IMG_SIZE)).values)
    band2 = np.stack(df["band_2"].apply(lambda x: np.array(x).reshape(IMG_SIZE, IMG_SIZE)).values)
    inc   = pd.to_numeric(df["inc_angle"], errors="coerce")
    inc   = inc.fillna(inc.mean()).values.astype(np.float32)
    return band1.astype(np.float32), band2.astype(np.float32), inc

train_b1, train_b2, train_inc = extract_bands(train_df)
test_b1,  test_b2,  test_inc  = extract_bands(test_df)
labels = train_df["is_iceberg"].values.astype(np.float32)

print("Train shape:", train_b1.shape, "Test shape:", test_b1.shape)

# Global normalization stats
all_pixels = np.concatenate([train_b1.reshape(-1), train_b2.reshape(-1)])
PIX_MEAN = all_pixels.mean().astype(np.float32)
PIX_STD  = all_pixels.std().astype(np.float32)
print("Pixel mean/std:", PIX_MEAN, PIX_STD)

# Composite magnitude
train_comp = np.sqrt(train_b1**2 + train_b2**2).astype(np.float32)
test_comp  = np.sqrt(test_b1**2 + test_b2**2).astype(np.float32)

# ===================================================
# 3. Bootstrap masks & train MaskNet (once, shared across seeds)
# ===================================================
def bootstrap_masks(comp):
    n = comp.shape[0]
    masks = np.zeros_like(comp, dtype=np.float32)
    for i in range(n):
        img = comp[i]
        thresh = np.percentile(img, 99.0)
        m = img >= thresh
        if m.sum() < 5:
            thresh = np.percentile(img, 98.0)
            m = img >= thresh
        if m.sum() == 0:
            max_idx = np.unravel_index(np.argmax(img), img.shape)
            m[max_idx] = True
        masks[i] = m.astype(np.float32)
    return masks

bootstrap_train_masks = bootstrap_masks(train_comp)

class MaskDataset(Dataset):
    def __init__(self, comp_imgs, masks, augment=True):
        self.comp = comp_imgs
        self.masks = masks
        self.augment = augment

    def __len__(self):
        return self.comp.shape[0]

    def __getitem__(self, idx):
        img = self.comp[idx]
        mask = self.masks[idx]
        img = (img - PIX_MEAN) / (PIX_STD + 1e-6)
        img_t = torch.from_numpy(img[None, ...].astype(np.float32))
        mask_t = torch.from_numpy(mask[None, ...].astype(np.float32))

        if self.augment:
            if random.random() < 0.5:
                img_t = torch.flip(img_t, dims=[2])
                mask_t = torch.flip(mask_t, dims=[2])
            if random.random() < 0.5:
                img_t = torch.flip(img_t, dims=[1])
                mask_t = torch.flip(mask_t, dims=[1])

        return img_t, mask_t

class MaskNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2)  # 75 -> 37
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2)  # 37 -> 18

        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, output_padding=1)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64+64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2, output_padding=1)
        self.dec2 = nn.Sequential(
            nn.Conv2d(32+32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.out_conv = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        b = self.bottleneck(p2)

        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))
        u2 = self.up2(d1)
        d2 = self.dec2(torch.cat([u2, e1], dim=1))
        logits = self.out_conv(d2)
        return logits

mask_dataset = MaskDataset(train_comp, bootstrap_train_masks, augment=True)
mask_loader = DataLoader(mask_dataset, batch_size=64, shuffle=True)
mask_net = MaskNet().to(device)
mask_opt = torch.optim.Adam(mask_net.parameters(), lr=1e-3, weight_decay=1e-4)
mask_criterion = nn.BCEWithLogitsLoss()

print("Training MaskNet...")
for epoch in range(5):
    mask_net.train()
    total_loss = 0.0
    n = 0
    for imgs, masks in mask_loader:
        imgs = imgs.to(device)
        masks = masks.to(device)
        mask_opt.zero_grad()
        logits = mask_net(imgs)
        loss = mask_criterion(logits, masks)
        loss.backward()
        mask_opt.step()
        total_loss += loss.item() * imgs.size(0)
        n += imgs.size(0)
    print(f"MaskNet Epoch {epoch+1} loss={total_loss/max(n,1):.4f}")

def infer_masks(mask_net, comp_imgs, batch_size=64):
    ds = MaskDataset(comp_imgs, np.zeros_like(comp_imgs), augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    mask_net.eval()
    all_masks = []
    with torch.no_grad():
        for imgs, _ in loader:
            imgs = imgs.to(device)
            logits = mask_net(imgs)
            probs = torch.sigmoid(logits).cpu().numpy()  # (B,1,H,W)
            all_masks.append(probs[:, 0])
    return np.concatenate(all_masks, axis=0)

print("Inferring learned masks...")
learned_train_masks = infer_masks(mask_net, train_comp)
learned_test_masks  = infer_masks(mask_net, test_comp)

# ===================================================
# 4. Handcrafted features from learned masks
# ===================================================
def features_from_masks(comp, masks, thresh=0.5):
    """
    Simple brightness/size features used before.
    """
    n = comp.shape[0]
    brightness = np.zeros((n, 2), dtype=np.float32)
    size = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        img = comp[i]
        m = masks[i] >= thresh
        if m.sum() < 5:
            t2 = np.percentile(img, 98.0)
            m = img >= t2
        if m.sum() == 0:
            max_idx = np.unravel_index(np.argmax(img), img.shape)
            m[max_idx] = True
        obj_vals = img[m]
        bg_vals  = img[~m]
        if bg_vals.size == 0:
            bg_vals = img.reshape(-1)
        obj_mean = float(obj_vals.mean())
        bg_mean  = float(bg_vals.mean())
        ratio = obj_mean / (bg_mean + 1e-6)
        diff  = obj_mean - bg_mean
        area_fraction = m.mean()
        brightness[i] = [ratio, diff]
        size[i] = [area_fraction]
    return brightness, size

train_brightness, train_size = features_from_masks(train_comp, learned_train_masks)
test_brightness,  test_size  = features_from_masks(test_comp,  learned_test_masks)

feat_all_train = np.concatenate([train_brightness, train_size], axis=1)
feat_mean = feat_all_train.mean(axis=0)
feat_std  = feat_all_train.std(axis=0) + 1e-6

def standardize_feats(feats, idx):
    return (feats - feat_mean[idx]) / feat_std[idx]

train_brightness_std = standardize_feats(train_brightness, slice(0,2))
test_brightness_std  = standardize_feats(test_brightness,  slice(0,2))
# size not used as its own classifier

# ---- New richer tabular features expert ----
def compute_handcrafted_features(b1, b2, masks, thresh=0.5):
    """
    Compute a set of scalar features per image.

    Features (D=11):
      0: obj_bg_b2_ratio
      1: obj_bg_b1_ratio
      2: b2_std
      3: b1_std
      4: b12_std
      5: blob_area
      6: obj_frac
      7: bg_hv_ratio_mean
      8: b1_mean
      9: b12_mean
      10: obj_bg_b2_contrast
    """
    n, H, W = b1.shape
    feats = np.zeros((n, 11), dtype=np.float32)
    eps = 1e-6

    for i in range(n):
        im1 = b1[i]
        im2 = b2[i]
        m_soft = masks[i]
        m = m_soft >= thresh

        if m.sum() < 5:
            # fallback: top 2% of composite magnitude
            comp_img = np.sqrt(im1**2 + im2**2)
            t2 = np.percentile(comp_img, 98.0)
            m = comp_img >= t2
        if m.sum() == 0:
            comp_img = np.sqrt(im1**2 + im2**2)
            max_idx = np.unravel_index(np.argmax(comp_img), comp_img.shape)
            m[max_idx] = True

        bg = ~m

        # object/background means in HV and HH
        obj_b2 = im2[m]
        obj_b1 = im1[m]
        bg_b2  = im2[bg] if bg.sum() > 0 else im2.reshape(-1)
        bg_b1  = im1[bg] if bg.sum() > 0 else im1.reshape(-1)

        obj_b2_mean = float(obj_b2.mean())
        bg_b2_mean  = float(bg_b2.mean())
        obj_b1_mean = float(obj_b1.mean())
        bg_b1_mean  = float(bg_b1.mean())

        obj_bg_b2_ratio = obj_b2_mean / (bg_b2_mean + eps)
        obj_bg_b1_ratio = obj_b1_mean / (bg_b1_mean + eps)

        # global stats
        b2_std  = float(im2.std())
        b1_std  = float(im1.std())
        b12     = 0.5 * (im1 + im2)
        b12_std = float(b12.std())

        b1_mean  = float(im1.mean())
        b12_mean = float(b12.mean())

        # area features
        blob_area = float(m.sum())
        obj_frac  = blob_area / float(H * W)

        # background hv / (hh + hv)
        denom = im1 + im2
        hv_ratio = im2 / (denom + eps)
        if bg.sum() > 0:
            bg_hv_ratio_mean = float(hv_ratio[bg].mean())
        else:
            bg_hv_ratio_mean = float(hv_ratio.mean())

        # contrast in HV
        obj_bg_b2_contrast = obj_b2_mean - bg_b2_mean

        feats[i] = [
            obj_bg_b2_ratio,
            obj_bg_b1_ratio,
            b2_std,
            b1_std,
            b12_std,
            blob_area,
            obj_frac,
            bg_hv_ratio_mean,
            b1_mean,
            b12_mean,
            obj_bg_b2_contrast,
        ]

    return feats

train_full_feats = compute_handcrafted_features(train_b1, train_b2, learned_train_masks)
test_full_feats  = compute_handcrafted_features(test_b1,  test_b2,  learned_test_masks)

full_feat_mean = train_full_feats.mean(axis=0)
full_feat_std  = train_full_feats.std(axis=0) + 1e-6

train_full_feats_std = (train_full_feats - full_feat_mean) / full_feat_std
test_full_feats_std  = (test_full_feats  - full_feat_mean) / full_feat_std

# ===================================================
# 5. Dataset with selective augmentations
# ===================================================
class IcebergDataset(Dataset):
    """
    Modes:
        - 'hv_cnn'          : HV band (band_2) only  [gentle aug]
        - 'hh_cnn'          : HH band (band_1) only  [gentle aug]
        - 'composite_cnn'   : 3-channel composite    [strong aug]
        - 'surround_cnn'    : composite surroundings [strong aug]
        - 'brightness_feat' : brightness features (2D)
        - 'feature_feat'    : full handcrafted tabular features (11D)
        - 'all_models'      : dict for ensemble
    """
    def __init__(self, data_dict, indices, mode="hv_cnn", augment=False, aug_kind="gentle"):
        self.data = data_dict
        self.indices = np.array(indices)
        self.mode = mode
        self.augment = augment
        self.aug_kind = aug_kind

    def __len__(self):
        return len(self.indices)

    def _normalize_img(self, img):
        return (img - PIX_MEAN) / (PIX_STD + 1e-6)

    def _torch_from_np(self, arr):
        return torch.from_numpy(arr.astype(np.float32))

    def _augment_image(self, img):
        C, H, W = img.shape

        # flips
        if random.random() < 0.5:
            img = torch.flip(img, dims=[2])
        if random.random() < 0.5:
            img = torch.flip(img, dims=[1])

        # 0/90/180/270 rotation
        k = random.randint(0, 3)
        img = torch.rot90(img, k, dims=[1, 2])

        if self.aug_kind == "gentle":
            # small brightness jitter
            if random.random() < 0.5:
                factor = random.uniform(0.9, 1.1)
                img = img * factor

            # small Gaussian noise
            if random.random() < 0.3:
                noise_std = 0.02
                img = img + torch.randn_like(img) * noise_std

            # small cutout
            if random.random() < 0.3:
                cutout_frac = 0.15
                cutout_size = int(IMG_SIZE * cutout_frac)
                cx = random.randint(0, IMG_SIZE - 1)
                cy = random.randint(0, IMG_SIZE - 1)
                x1 = max(cx - cutout_size // 2, 0)
                y1 = max(cy - cutout_size // 2, 0)
                x2 = min(cx + cutout_size // 2, IMG_SIZE)
                y2 = min(cy + cutout_size // 2, IMG_SIZE)
                img[:, y1:y2, x1:x2] = 0.0

        else:  # strong
            # small affine
            if random.random() < 0.7:
                angle = random.uniform(-10, 10)
                t_frac = 0.05
                max_dx = W * t_frac
                max_dy = H * t_frac
                translations = (
                    random.uniform(-max_dx, max_dx),
                    random.uniform(-max_dy, max_dy),
                )
                scale = random.uniform(0.9, 1.1)
                shear = 0.0
                img = TF.affine(
                    img, angle=angle,
                    translate=translations,
                    scale=scale,
                    shear=[shear, shear]
                )

            # random zoom
            if random.random() < 0.5:
                zoom = random.uniform(0.85, 1.0)
                new_size = int(IMG_SIZE * zoom)
                if new_size < IMG_SIZE:
                    top = random.randint(0, IMG_SIZE - new_size)
                    left = random.randint(0, IMG_SIZE - new_size)
                    cropped = img[:, top:top+new_size, left:left+new_size].unsqueeze(0)
                    img = F.interpolate(
                        cropped, size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False
                    )[0]

            # brightness jitter
            if random.random() < 0.5:
                factor = random.uniform(0.8, 1.2)
                img = img * factor

            # contrast jitter
            if random.random() < 0.5:
                contrast = random.uniform(0.8, 1.2)
                mean = img.mean(dim=(1,2), keepdim=True)
                img = (img - mean) * contrast + mean

            # speckle-like noise
            if random.random() < 0.5:
                speckle_std = 0.08
                noise = 1.0 + speckle_std * torch.randn_like(img)
                img = img * noise

            # small Gaussian noise
            if random.random() < 0.3:
                noise_std = 0.02
                img = img + torch.randn_like(img) * noise_std

            # stronger cutout
            if random.random() < 0.6:
                cutout_frac = 0.25
                cutout_size = int(IMG_SIZE * cutout_frac)
                cx = random.randint(0, IMG_SIZE - 1)
                cy = random.randint(0, IMG_SIZE - 1)
                x1 = max(cx - cutout_size // 2, 0)
                y1 = max(cy - cutout_size // 2, 0)
                x2 = min(cx + cutout_size // 2, IMG_SIZE)
                y2 = min(cy + cutout_size // 2, IMG_SIZE)
                img[:, y1:y2, x1:x2] = 0.0

            # channel dropout for multi-channel
            if C > 1 and random.random() < 0.3:
                ch = random.randrange(C)
                img[ch, :, :] = 0.0

        return img

    def __getitem__(self, idx):
        i = int(self.indices[idx])
        b1 = self.data["band1"][i]
        b2 = self.data["band2"][i]
        mask = self.data["masks"][i]
        brightness = self.data["brightness"][i]
        full_feat  = self.data["full_feats"][i]
        inc = self.data["inc"][i]
        label_arr = self.data.get("labels", None)
        y = float(label_arr[i]) if label_arr is not None else 0.0

        hh = self._normalize_img(b1)[None, ...]
        hv = self._normalize_img(b2)[None, ...]
        comp3 = np.stack([b1, b2, (b1 + b2) / 2.0], axis=0)
        comp3 = self._normalize_img(comp3)

        mask_img = (mask >= 0.5).astype(np.float32)[None, ...]  # (1,H,W)
        surround = comp3 * (1.0 - mask_img)

        hh_t       = self._torch_from_np(hh)
        hv_t       = self._torch_from_np(hv)
        comp_t     = self._torch_from_np(comp3)
        surround_t = self._torch_from_np(surround)

        if self.augment and self.mode in ["hv_cnn", "hh_cnn", "composite_cnn", "surround_cnn"]:
            if self.mode == "hv_cnn":
                hv_t = self._augment_image(hv_t)
            elif self.mode == "hh_cnn":
                hh_t = self._augment_image(hh_t)
            elif self.mode == "composite_cnn":
                comp_t = self._augment_image(comp_t)
            elif self.mode == "surround_cnn":
                surround_t = self._augment_image(surround_t)

        brightness_t = self._torch_from_np(brightness)
        full_feat_t  = self._torch_from_np(full_feat)
        y_t = torch.tensor(y, dtype=torch.float32)

        if self.mode == "hv_cnn":
            return hv_t, y_t
        elif self.mode == "hh_cnn":
            return hh_t, y_t
        elif self.mode == "composite_cnn":
            return comp_t, y_t
        elif self.mode == "surround_cnn":
            return surround_t, y_t
        elif self.mode == "brightness_feat":
            return brightness_t, y_t
        elif self.mode == "feature_feat":
            return full_feat_t, y_t
        elif self.mode == "all_models":
            return {
                "hv": hv_t,
                "hh": hh_t,
                "composite": comp_t,
                "surround": surround_t,
                "brightness": brightness_t,
                "features": full_feat_t,
                "label": y_t,
            }
        else:
            raise ValueError("Unknown mode " + self.mode)

# ===================================================
# 6. Models
# ===================================================
class DeepConvNet(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out.squeeze(1)

class AltConvNet(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        act = nn.SiLU()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 48, kernel_size=5, padding=2),
            nn.BatchNorm2d(48),
            act,

            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            act,
            nn.MaxPool2d(2),          # 75 -> 37
            nn.Dropout2d(0.1),

            nn.Conv2d(48, 96, kernel_size=5, padding=2),
            nn.BatchNorm2d(96),
            act,

            nn.Conv2d(96, 96, kernel_size=3, padding=1),
            nn.BatchNorm2d(96),
            act,
            nn.MaxPool2d(2),          # 37 -> 18
            nn.Dropout2d(0.15),

            nn.Conv2d(96, 192, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(192),
            act,

            nn.Conv2d(192, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            act,
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 128),
            act,
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        feat = self.features(x)
        out = self.classifier(feat)
        return out.squeeze(1)

class FeatureMLP(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
        )
    def forward(self, x):
        return self.net(x).squeeze(1)

class CombinedEnsemble(nn.Module):
    def __init__(self,
                 model_hv, model_hh, model_comp, model_surround,
                 model_brightness,
                 model_hv_alt, model_hh_alt, model_comp_alt, model_surround_alt,
                 model_fullfeat):
        super().__init__()
        self.model_hv = model_hv
        self.model_hh = model_hh
        self.model_comp = model_comp
        self.model_surround = model_surround
        self.model_brightness = model_brightness

        self.model_hv_alt = model_hv_alt
        self.model_hh_alt = model_hh_alt
        self.model_comp_alt = model_comp_alt
        self.model_surround_alt = model_surround_alt

        self.model_fullfeat = model_fullfeat

        # 10 base logits:
        # hv, hh, comp, sur, bright,
        # hv_alt, hh_alt, comp_alt, sur_alt, fullfeat
        self.head = nn.Sequential(
            nn.Linear(10, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
        )

    def forward(self, batch):
        hv = batch["hv"]
        hh = batch["hh"]
        comp = batch["composite"]
        sur = batch["surround"]
        bright = batch["brightness"]
        feats = batch["features"]

        logit_hv = self.model_hv(hv)
        logit_hh = self.model_hh(hh)
        logit_comp = self.model_comp(comp)
        logit_sur = self.model_surround(sur)
        logit_bright = self.model_brightness(bright)

        logit_hv_alt = self.model_hv_alt(hv)
        logit_hh_alt = self.model_hh_alt(hh)
        logit_comp_alt = self.model_comp_alt(comp)
        logit_sur_alt = self.model_surround_alt(sur)

        logit_fullfeat = self.model_fullfeat(feats)

        stacked = torch.stack(
            [logit_hv, logit_hh, logit_comp, logit_sur,
             logit_bright,
             logit_hv_alt, logit_hh_alt, logit_comp_alt, logit_sur_alt,
             logit_fullfeat],
            dim=1
        )
        out = self.head(stacked)
        return out.squeeze(1)

# ===================================================
# 7. Training utilities
# ===================================================
def apply_mixup(x, y, alpha=0.0):
    if alpha <= 0.0:
        return x, y
    lam = np.random.beta(alpha, alpha)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1 - lam) * x[perm]
    y_mix = lam * y + (1 - lam) * y[perm]
    return x_mix, y_mix

def smooth_labels(y, smoothing=0.0):
    if smoothing <= 0.0:
        return y
    return y * (1 - smoothing) + 0.5 * smoothing

def train_simple_model(model, train_loader, val_loader,
                       max_epochs=30, lr=1e-3, weight_decay=1e-4,
                       label_smoothing=0.1, mixup_alpha=0.2,
                       early_stop_patience=5, name="model"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_logloss = float("inf")
    patience = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        n_train = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            x_mix, y_mix = apply_mixup(x, y, alpha=mixup_alpha)
            y_smooth = smooth_labels(y_mix, smoothing=label_smoothing)

            optimizer.zero_grad()
            logits = model(x_mix)
            loss = F.binary_cross_entropy_with_logits(logits, y_smooth)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            n_train += y.size(0)
        train_loss /= max(n_train, 1)

        model.eval()
        val_loss = 0.0
        n_val = 0
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                loss = F.binary_cross_entropy_with_logits(logits, y)
                val_loss += loss.item() * y.size(0)
                n_val += y.size(0)
                probs = torch.sigmoid(logits)
                val_probs.append(probs.cpu().numpy())
                val_targets.append(y.cpu().numpy())
        val_loss /= max(n_val, 1)
        val_probs = np.concatenate(val_probs)
        val_targets = np.concatenate(val_targets)
        val_logloss = log_loss(val_targets, val_probs)

        improved = val_logloss < best_val_logloss
        if improved:
            best_val_logloss = val_logloss
            best_state = model.state_dict()
            patience = 0
        else:
            patience += 1

        print(f"[{name}] Epoch {epoch:02d} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_logloss={val_logloss:.4f} "
              f"{'(*)' if improved else ''}")
        if patience >= early_stop_patience:
            print(f"[{name}] Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def train_ensemble_head(ensemble, train_loader, val_loader,
                        max_epochs=30, lr=1e-3, weight_decay=1e-4,
                        label_smoothing=0.1, mixup_alpha=0.2,
                        early_stop_patience=5):
    # freeze base models
    for m in [
        ensemble.model_hv, ensemble.model_hh, ensemble.model_comp,
        ensemble.model_surround, ensemble.model_brightness,
        ensemble.model_hv_alt, ensemble.model_hh_alt,
        ensemble.model_comp_alt, ensemble.model_surround_alt,
        ensemble.model_fullfeat
    ]:
        for p in m.parameters():
            p.requires_grad = False
        m.eval()

    ensemble = ensemble.to(device)
    optimizer = torch.optim.Adam(ensemble.head.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = None
    best_val_logloss = float("inf")
    patience = 0

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_logloss": [],
    }

    for epoch in range(1, max_epochs + 1):
        ensemble.train()
        train_loss = 0.0
        n_train = 0
        for batch in train_loader:
            y = batch["label"].to(device)
            hv = batch["hv"].to(device)
            hh = batch["hh"].to(device)
            comp = batch["composite"].to(device)
            sur = batch["surround"].to(device)
            bright = batch["brightness"].to(device)
            feats = batch["features"].to(device)

            with torch.no_grad():
                logit_hv = ensemble.model_hv(hv)
                logit_hh = ensemble.model_hh(hh)
                logit_comp = ensemble.model_comp(comp)
                logit_sur  = ensemble.model_surround(sur)
                logit_br   = ensemble.model_brightness(bright)

                logit_hv_alt = ensemble.model_hv_alt(hv)
                logit_hh_alt = ensemble.model_hh_alt(hh)
                logit_comp_alt = ensemble.model_comp_alt(comp)
                logit_sur_alt = ensemble.model_surround_alt(sur)

                logit_fullfeat = ensemble.model_fullfeat(feats)

            stacked = torch.stack(
                [logit_hv, logit_hh, logit_comp, logit_sur,
                 logit_br,
                 logit_hv_alt, logit_hh_alt, logit_comp_alt, logit_sur_alt,
                 logit_fullfeat],
                dim=1
            )
            stacked, y_mix = apply_mixup(stacked, y, alpha=mixup_alpha)
            y_smooth = smooth_labels(y_mix, smoothing=label_smoothing)

            optimizer.zero_grad()
            logits = ensemble.head(stacked).squeeze(1)
            loss = F.binary_cross_entropy_with_logits(logits, y_smooth)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * y.size(0)
            n_train += y.size(0)
        train_loss /= max(n_train, 1)

        ensemble.eval()
        val_loss = 0.0
        n_val = 0
        val_probs = []
        val_targets = []
        with torch.no_grad():
            for batch in val_loader:
                y = batch["label"].to(device)
                hv = batch["hv"].to(device)
                hh = batch["hh"].to(device)
                comp = batch["composite"].to(device)
                sur = batch["surround"].to(device)
                bright = batch["brightness"].to(device)
                feats = batch["features"].to(device)

                logit_hv = ensemble.model_hv(hv)
                logit_hh = ensemble.model_hh(hh)
                logit_comp = ensemble.model_comp(comp)
                logit_sur  = ensemble.model_surround(sur)
                logit_br   = ensemble.model_brightness(bright)

                logit_hv_alt = ensemble.model_hv_alt(hv)
                logit_hh_alt = ensemble.model_hh_alt(hh)
                logit_comp_alt = ensemble.model_comp_alt(comp)
                logit_sur_alt = ensemble.model_surround_alt(sur)

                logit_fullfeat = ensemble.model_fullfeat(feats)

                stacked = torch.stack(
                    [logit_hv, logit_hh, logit_comp, logit_sur,
                     logit_br,
                     logit_hv_alt, logit_hh_alt, logit_comp_alt, logit_sur_alt,
                     logit_fullfeat],
                    dim=1
                )
                logits = ensemble.head(stacked).squeeze(1)
                loss = F.binary_cross_entropy_with_logits(logits, y)

                val_loss += loss.item() * y.size(0)
                n_val += y.size(0)
                probs = torch.sigmoid(logits)
                val_probs.append(probs.cpu().numpy())
                val_targets.append(y.cpu().numpy())
        val_loss /= max(n_val, 1)
        val_probs = np.concatenate(val_probs)
        val_targets = np.concatenate(val_targets)
        val_logloss = log_loss(val_targets, val_probs)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_logloss"].append(val_logloss)

        improved = val_logloss < best_val_logloss
        if improved:
            best_val_logloss = val_logloss
            best_state = ensemble.state_dict()
            patience = 0
        else:
            patience += 1

        print(f"[Ensemble] Epoch {epoch:02d} | train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_logloss={val_logloss:.4f} "
              f"{'(*)' if improved else ''}")
        if patience >= early_stop_patience:
            print("[Ensemble] Early stopping")
            break

    if best_state is not None:
        ensemble.load_state_dict(best_state)
    return ensemble, history

# ===================================================
# 8. Helper: ensemble forward for a batch
# ===================================================
def ensemble_forward_batch(ensemble_model, batch):
    hv = batch["hv"].to(device)
    hh = batch["hh"].to(device)
    comp = batch["composite"].to(device)
    sur = batch["surround"].to(device)
    bright = batch["brightness"].to(device)
    feats = batch["features"].to(device)

    logit_hv = ensemble_model.model_hv(hv)
    logit_hh = ensemble_model.model_hh(hh)
    logit_comp = ensemble_model.model_comp(comp)
    logit_sur  = ensemble_model.model_surround(sur)
    logit_br   = ensemble_model.model_brightness(bright)

    logit_hv_alt = ensemble_model.model_hv_alt(hv)
    logit_hh_alt = ensemble_model.model_hh_alt(hh)
    logit_comp_alt = ensemble_model.model_comp_alt(comp)
    logit_sur_alt = ensemble_model.model_surround_alt(sur)

    logit_fullfeat = ensemble_model.model_fullfeat(feats)

    stacked = torch.stack(
        [logit_hv, logit_hh, logit_comp, logit_sur,
         logit_br,
         logit_hv_alt, logit_hh_alt, logit_comp_alt, logit_sur_alt,
         logit_fullfeat],
        dim=1
    )
    logits = ensemble_model.head(stacked).squeeze(1)
    return logits

# ===================================================
# 9. Run pipeline for one seed (with plots)
# ===================================================
def run_one_seed(seed: int):
    print("\n" + "#"*40)
    print(f"### Running seed = {seed}")
    print("#"*40)
    set_seed(seed)

    train_data = {
        "band1": train_b1,
        "band2": train_b2,
        "inc": train_inc,
        "masks": (learned_train_masks >= 0.5).astype(np.float32),
        "brightness": train_brightness_std,
        "full_feats": train_full_feats_std,
        "labels": labels,
    }
    test_data = {
        "band1": test_b1,
        "band2": test_b2,
        "inc": test_inc,
        "masks": (learned_test_masks >= 0.5).astype(np.float32),
        "brightness": test_brightness_std,
        "full_feats": test_full_feats_std,
        "labels": None,
    }

    idx_all = np.arange(len(labels))
    train_idx, val_idx = train_test_split(
        idx_all, test_size=0.2, random_state=seed, stratify=labels
    )

    BATCH_CNN = 64
    BATCH_FEAT = 64

    # HV: gentle aug
    train_ds_hv = IcebergDataset(train_data, train_idx, mode="hv_cnn", augment=True,  aug_kind="gentle")
    val_ds_hv   = IcebergDataset(train_data, val_idx,   mode="hv_cnn", augment=False, aug_kind="gentle")
    train_loader_hv = DataLoader(train_ds_hv, batch_size=BATCH_CNN, shuffle=True)
    val_loader_hv   = DataLoader(val_ds_hv,   batch_size=BATCH_CNN, shuffle=False)

    # HH: gentle aug
    train_ds_hh = IcebergDataset(train_data, train_idx, mode="hh_cnn", augment=True,  aug_kind="gentle")
    val_ds_hh   = IcebergDataset(train_data, val_idx,   mode="hh_cnn", augment=False, aug_kind="gentle")
    train_loader_hh = DataLoader(train_ds_hh, batch_size=BATCH_CNN, shuffle=True)
    val_loader_hh   = DataLoader(val_ds_hh,   batch_size=BATCH_CNN, shuffle=False)

    # Composite: strong aug
    train_ds_comp = IcebergDataset(train_data, train_idx, mode="composite_cnn", augment=True,  aug_kind="strong")
    val_ds_comp   = IcebergDataset(train_data, val_idx,   mode="composite_cnn", augment=False, aug_kind="strong")
    train_loader_comp = DataLoader(train_ds_comp, batch_size=BATCH_CNN, shuffle=True)
    val_loader_comp   = DataLoader(val_ds_comp,   batch_size=BATCH_CNN, shuffle=False)

    # Surround: strong aug
    train_ds_sur = IcebergDataset(train_data, train_idx, mode="surround_cnn", augment=True,  aug_kind="strong")
    val_ds_sur   = IcebergDataset(train_data, val_idx,   mode="surround_cnn", augment=False, aug_kind="strong")
    train_loader_sur = DataLoader(train_ds_sur, batch_size=BATCH_CNN, shuffle=True)
    val_loader_sur   = DataLoader(val_ds_sur,   batch_size=BATCH_CNN, shuffle=False)

    # Brightness (2D)
    train_ds_bright = IcebergDataset(train_data, train_idx, mode="brightness_feat", augment=False)
    val_ds_bright   = IcebergDataset(train_data, val_idx,   mode="brightness_feat", augment=False)
    train_loader_bright = DataLoader(train_ds_bright, batch_size=BATCH_FEAT, shuffle=True)
    val_loader_bright   = DataLoader(val_ds_bright,   batch_size=BATCH_FEAT, shuffle=False)

    # Full handcrafted feature expert (11D)
    train_ds_fullfeat = IcebergDataset(train_data, train_idx, mode="feature_feat", augment=False)
    val_ds_fullfeat   = IcebergDataset(train_data, val_idx,   mode="feature_feat", augment=False)
    train_loader_fullfeat = DataLoader(train_ds_fullfeat, batch_size=BATCH_FEAT, shuffle=True)
    val_loader_fullfeat   = DataLoader(val_ds_fullfeat,   batch_size=BATCH_FEAT, shuffle=False)

    # Train individual models
    EPOCHS_CNN_MAX = 30
    PATIENCE_CNN = 6

    # Original CNNs
    model_hv = train_simple_model(
        DeepConvNet(in_channels=1),
        train_loader_hv, val_loader_hv,
        max_epochs=EPOCHS_CNN_MAX, lr=1e-3, weight_decay=1e-4,
        label_smoothing=0.1, mixup_alpha=0.2,
        early_stop_patience=PATIENCE_CNN,
        name="HV CNN"
    )

    model_hh = train_simple_model(
        DeepConvNet(in_channels=1),
        train_loader_hh, val_loader_hh,
        max_epochs=EPOCHS_CNN_MAX, lr=1e-3, weight_decay=1e-4,
        label_smoothing=0.1, mixup_alpha=0.2,
        early_stop_patience=PATIENCE_CNN,
        name="HH CNN"
    )

    model_comp = train_simple_model(
        DeepConvNet(in_channels=3),
        train_loader_comp, val_loader_comp,
        max_epochs=EPOCHS_CNN_MAX, lr=1e-3, weight_decay=1e-4,
        label_smoothing=0.1, mixup_alpha=0.2,
        early_stop_patience=PATIENCE_CNN,
        name="Composite CNN"
    )

    # Surround: no mixup
    model_sur = train_simple_model(
        DeepConvNet(in_channels=3),
        train_loader_sur, val_loader_sur,
        max_epochs=EPOCHS_CNN_MAX, lr=1e-3, weight_decay=1e-4,
        label_smoothing=0.1, mixup_alpha=0.0,
        early_stop_patience=PATIENCE_CNN,
        name="Surround CNN"
    )

    # Alt CNNs (second expert) on all four views
    model_hv_alt = train_simple_model(
        AltConvNet(in_channels=1),
        train_loader_hv, val_loader_hv,
        max_epochs=EPOCHS_CNN_MAX, lr=1e-3, weight_decay=1e-4,
        label_smoothing=0.1, mixup_alpha=0.2,
        early_stop_patience=PATIENCE_CNN,
        name="HV Alt CNN"
    )

    model_hh_alt = train_simple_model(
        AltConvNet(in_channels=1),
        train_loader_hh, val_loader_hh,
        max_epochs=EPOCHS_CNN_MAX, lr=1e-3, weight_decay=1e-4,
        label_smoothing=0.1, mixup_alpha=0.2,
        early_stop_patience=PATIENCE_CNN,
        name="HH Alt CNN"
    )

    model_comp_alt = train_simple_model(
        AltConvNet(in_channels=3),
        train_loader_comp, val_loader_comp,
        max_epochs=EPOCHS_CNN_MAX, lr=1e-3, weight_decay=1e-4,
        label_smoothing=0.1, mixup_alpha=0.2,
        early_stop_patience=PATIENCE_CNN,
        name="Composite Alt CNN"
    )

    # Surround alt: also no mixup
    model_sur_alt = train_simple_model(
        AltConvNet(in_channels=3),
        train_loader_sur, val_loader_sur,
        max_epochs=EPOCHS_CNN_MAX, lr=1e-3, weight_decay=1e-4,
        label_smoothing=0.1, mixup_alpha=0.0,
        early_stop_patience=PATIENCE_CNN,
        name="Surround Alt CNN"
    )

    # Brightness MLP: 2D features
    model_brightness = train_simple_model(
        FeatureMLP(in_dim=2),
        train_loader_bright, val_loader_bright,
        max_epochs=60, lr=3e-4, weight_decay=5e-5,
        label_smoothing=0.0, mixup_alpha=0.0,
        early_stop_patience=12,
        name="Brightness MLP"
    )

    # Full tabular features MLP: 11D
    model_fullfeat = train_simple_model(
        FeatureMLP(in_dim=train_full_feats_std.shape[1]),
        train_loader_fullfeat, val_loader_fullfeat,
        max_epochs=60, lr=3e-4, weight_decay=5e-5,
        label_smoothing=0.0, mixup_alpha=0.0,
        early_stop_patience=12,
        name="FullFeature MLP"
    )

    # Ensemble
    train_ds_all = IcebergDataset(train_data, train_idx, mode="all_models", augment=False)
    val_ds_all   = IcebergDataset(train_data, val_idx,   mode="all_models", augment=False)
    train_loader_all = DataLoader(train_ds_all, batch_size=BATCH_CNN, shuffle=True)
    val_loader_all   = DataLoader(val_ds_all,   batch_size=BATCH_CNN, shuffle=False)

    ensemble_model = CombinedEnsemble(
        model_hv, model_hh, model_comp, model_sur,
        model_brightness,
        model_hv_alt, model_hh_alt, model_comp_alt, model_sur_alt,
        model_fullfeat
    )

    ensemble_model, ensemble_history = train_ensemble_head(
        ensemble_model, train_loader_all, val_loader_all,
        max_epochs=30, lr=1e-3, weight_decay=1e-4,
        label_smoothing=0.1, mixup_alpha=0.2,
        early_stop_patience=6
    )

    # ----------------------------------
    # Plots: training/validation curves
    # ----------------------------------
    epochs = np.arange(1, len(ensemble_history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, ensemble_history["train_loss"], label="Train loss (BCE)")
    plt.plot(epochs, ensemble_history["val_loss"], label="Val loss (BCE)")
    plt.plot(epochs, ensemble_history["val_logloss"], label="Val logloss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"Ensemble training curves (seed {seed})")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ----------------------------------
    # Evaluate ensemble on validation set
    # ----------------------------------
    ensemble_model.eval()
    val_probs_list = []
    val_targets_list = []

    with torch.no_grad():
        for batch in val_loader_all:
            logits = ensemble_forward_batch(ensemble_model, batch)
            probs = torch.sigmoid(logits)

            val_probs_list.append(probs.cpu().numpy())
            val_targets_list.append(batch["label"].cpu().numpy())

    val_probs = np.concatenate(val_probs_list)
    val_targets = np.concatenate(val_targets_list)

    # ----------------------------------
    # Plot ROC curve
    # ----------------------------------
    fpr, tpr, _ = roc_curve(val_targets, val_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Ensemble ROC on validation (seed {seed})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

    # ----------------------------------
    # Confusion matrix at 0.5 threshold
    # ----------------------------------
    pred_labels = (val_probs >= 0.5).astype(int)
    cm = confusion_matrix(val_targets, pred_labels)

    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    fig.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Ship (0)", "Iceberg (1)"])
    ax.set_yticklabels(["Ship (0)", "Iceberg (1)"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion matrix (val, thr=0.5)")

    # annotate cells
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, cm[i, j],
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black"
            )
    plt.tight_layout()
    plt.show()

    # ----------------------------------
    # Histogram of predicted probabilities by class
    # ----------------------------------
    plt.figure(figsize=(8, 5))
    plt.hist(val_probs[val_targets == 0], bins=20, alpha=0.6, label="Ship (0)")
    plt.hist(val_probs[val_targets == 1], bins=20, alpha=0.6, label="Iceberg (1)")
    plt.xlabel("Predicted probability of iceberg (class 1)")
    plt.ylabel("Count")
    plt.title("Validation probability distributions by class")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ----------------------------------
    # Visualize some validation samples with predictions
    # ----------------------------------
    num_vis = min(16, len(val_ds_all))
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()

    for j in range(num_vis):
        sample = val_ds_all[j]  # mode="all_models": dict with 'composite' and 'label'
        img_t = sample["composite"]  # (3, H, W), normalized
        true_label = sample["label"].item()

        # Build a single-sample batch to feed to ensemble
        batch_single = {
            "hv": sample["hv"].unsqueeze(0),
            "hh": sample["hh"].unsqueeze(0),
            "composite": sample["composite"].unsqueeze(0),
            "surround": sample["surround"].unsqueeze(0),
            "brightness": sample["brightness"].unsqueeze(0),
            "features": sample["features"].unsqueeze(0),
        }

        with torch.no_grad():
            logits_single = ensemble_forward_batch(ensemble_model, batch_single)
            prob_single = torch.sigmoid(logits_single).item()
        pred_label = 1 if prob_single >= 0.5 else 0

        # prepare image for plotting
        img_np = img_t.numpy().transpose(1, 2, 0)  # (H, W, 3)
        # normalize to [0,1] for display
        img_min, img_max = img_np.min(), img_np.max()
        if img_max > img_min:
            img_disp = (img_np - img_min) / (img_max - img_min)
        else:
            img_disp = np.zeros_like(img_np)

        ax = axes[j]
        ax.imshow(img_disp)
        ax.axis("off")
        ax.set_title(
            f"T:{int(true_label)}  P:{int(pred_label)}\nprob={prob_single:.2f}",
            fontsize=9
        )

    # hide any unused axes
    for k in range(num_vis, len(axes)):
        axes[k].axis("off")

    plt.suptitle(f"Validation samples and ensemble predictions (seed {seed})", fontsize=14)
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Inference on test
    # -------------------------
    test_indices = np.arange(len(test_b1))
    test_data_for_ds = test_data.copy()
    test_data_for_ds["labels"] = np.zeros(len(test_b1), dtype=np.float32)
    test_ds_all = IcebergDataset(test_data_for_ds, test_indices, mode="all_models", augment=False)
    test_loader_all = DataLoader(test_ds_all, batch_size=BATCH_CNN, shuffle=False)

    ensemble_model.eval()
    test_probs = []
    with torch.no_grad():
        for batch in test_loader_all:
            logits = ensemble_forward_batch(ensemble_model, batch)
            probs = torch.sigmoid(logits)
            test_probs.append(probs.cpu().numpy())
    test_probs = np.concatenate(test_probs)
    return test_probs

# ===================================================
# 10. Multi-seed runs & averaged submission
# ===================================================
seeds = [42, 1337, 2025]
all_test_preds = []

for s in seeds:
    preds = run_one_seed(s)
    all_test_preds.append(preds)

all_test_preds = np.stack(all_test_preds, axis=0)  # (num_seeds, N_test)
avg_test_probs = all_test_preds.mean(axis=0)

submission = pd.DataFrame({
    "id": test_df["id"],
    "is_iceberg": avg_test_probs
})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Saved submission to:", submission_path)


