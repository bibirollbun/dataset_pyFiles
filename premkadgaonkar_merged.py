import pandas as pd
import os

# Path where Kaggle mounts the competition dataset
data_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

# List whatâ€™s inside
print(os.listdir(data_path))

# Load train.csv
train_df = pd.read_csv(os.path.join(data_path, "train.csv"))

train_df.shape
train_df.head()


train_label_coordinates = pd.read_csv(os.path.join(data_path, "train_label_coordinates.csv"))


train_label_coordinates


import os

data_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

# list a few study folders
print(os.listdir(os.path.join(data_path, "train_images"))[:5])



study_id = "4003253"
study_path = os.path.join(data_path, "train_images", study_id)

print("Series inside study:", os.listdir(study_path)[:5])



series_id = os.listdir(study_path)[0]  # take first series
series_path = os.path.join(study_path, series_id)

print("DICOM slices inside series:", os.listdir(series_path)[:5])



import pydicom
import matplotlib.pyplot as plt

dcm_file = os.path.join(series_path, os.listdir(series_path)[0])  # first slice
dcm = pydicom.dcmread(dcm_file)

print(dcm)  # metadata
plt.imshow(dcm.pixel_array, cmap='gray')
plt.axis('off')
plt.show()



import pydicom
import matplotlib.pyplot as plt
import os

# Pick one study and series (from your earlier code)
study_id = "4003253"
series_id = os.listdir(os.path.join(data_path, "train_images", study_id))[0]
series_path = os.path.join(data_path, "train_images", study_id, series_id)

# Get all slices in this series and sort by filename (important!)
dcm_files = sorted(os.listdir(series_path))

# Pick 5 slices spread across the series
sample_slices = [dcm_files[i] for i in [0, len(dcm_files)//4, len(dcm_files)//2, 3*len(dcm_files)//4, -1]]

# Plot them
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for ax, fname in zip(axes, sample_slices):
    dcm_path = os.path.join(series_path, fname)
    dcm = pydicom.dcmread(dcm_path)
    ax.imshow(dcm.pixel_array, cmap='gray')
    ax.set_title(fname)
    ax.axis('off')

plt.show()



import pandas as pd
import os

coords_path = os.path.join(data_path, "train_label_coordinates.csv")
coords_df = pd.read_csv(coords_path)

print(coords_df.shape)
print(coords_df.head())



import pydicom
import matplotlib.pyplot as plt
import os

# Pick one row from coords_df
row = coords_df.iloc[0]

study_id = str(row.study_id)
series_id = str(row.series_id)
instance_number = int(row.instance_number)

# Build path to the series folder
series_path = os.path.join(data_path, "train_images", study_id, series_id)

# Get all slices (files) in this series
dcm_files = sorted(os.listdir(series_path))

# instance_number in metadata starts from 1, so adjust index
dcm_file = os.path.join(series_path, dcm_files[instance_number - 1])

# Load the DICOM slice
dcm = pydicom.dcmread(dcm_file)
img = dcm.pixel_array

# Plot image with coordinate marked
plt.figure(figsize=(6,6))
plt.imshow(img, cmap='gray')
plt.scatter(row.x, row.y, c='red', s=40, label=f"{row.condition} {row.level}")
plt.legend()
plt.axis('off')
plt.show()



import pydicom
import matplotlib.pyplot as plt

# Pick a study to visualize
study_id = "4003253"

# Filter coords for this study
study_coords = coords_df[coords_df.study_id == int(study_id)]

# Pick one series from this study
series_id = str(study_coords.series_id.iloc[0])
series_path = os.path.join(data_path, "train_images", study_id, series_id)

# Get all slices in this series
dcm_files = sorted(os.listdir(series_path))

# Choose a slice number that has multiple labels (for demonstration)
slice_num = study_coords.instance_number.iloc[0]
dcm_file = os.path.join(series_path, dcm_files[slice_num - 1])

# Load the slice
dcm = pydicom.dcmread(dcm_file)
img = dcm.pixel_array

# Plot slice with all coordinates from this slice
plt.figure(figsize=(7,7))
plt.imshow(img, cmap='gray')

for _, row in study_coords[study_coords.instance_number == slice_num].iterrows():
    plt.scatter(row.x, row.y, c='red', s=40)
    plt.text(row.x+5, row.y+5, f"{row.level}", color='yellow', fontsize=9)

plt.title(f"Study {study_id} - Slice {slice_num}")
plt.axis('off')
plt.show()



train_study_ids = set(train_df.study_id.astype(str))
image_study_ids = set(os.listdir(os.path.join(data_path, "train_images")))

missing_studies = train_study_ids - image_study_ids
extra_studies = image_study_ids - train_study_ids

print("Missing studies in images:", len(missing_studies))
print("Extra studies not in train.csv:", len(extra_studies))



import os
import pydicom
import numpy as np
import cv2
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt

# Base path (same as before)
data_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

# Parameters
PATCH_SIZE = 128
OUTPUT_DIR = "/kaggle/working/patches_windowed"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load label coordinates
coords_df = pd.read_csv(os.path.join(data_path, "train_label_coordinates.csv"))

# ============== Helper Functions ==============

def window_image(img, center, width):
    """Apply DICOM windowing to convert 16-bit -> 8-bit visible range."""
    img_min = center - width / 2
    img_max = center + width / 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min)
    img = (img * 255).astype(np.uint8)
    return img

def safe_crop(img, center_x, center_y, size=PATCH_SIZE):
    """Crop square patch centered at (x,y) safely even near borders."""
    h, w = img.shape
    half = size // 2
    x1, x2 = center_x - half, center_x + half
    y1, y2 = center_y - half, center_y + half
    patch = np.zeros((size, size), dtype=img.dtype)
    x1_img, x2_img = max(0, x1), min(w, x2)
    y1_img, y2_img = max(0, y1), min(h, y2)
    x1_patch, y1_patch = x1_img - x1, y1_img - y1
    x2_patch, y2_patch = x1_patch + (x2_img - x1_img), y1_patch + (y2_img - y1_img)
    patch[y1_patch:y2_patch, x1_patch:x2_patch] = img[y1_img:y2_img, x1_img:x2_img]
    return patch

# ============== Cropping Loop ==============

saved, skipped = 0, 0
sample_patches = []  # to visualize

for idx, row in tqdm(coords_df.iterrows(), total=len(coords_df)):
    study_id = str(row["study_id"])
    series_id = str(row["series_id"])
    inst_no = int(row["instance_number"])
    x, y = int(row["x"]), int(row["y"])

    condition = str(row['condition']).replace(" ", "_")
    level = str(row['level']).replace("/", "-")

    series_path = os.path.join(data_path, "train_images", study_id, series_id)

    try:
        dcm_file = os.path.join(series_path, f"{inst_no}.dcm")
        dcm = pydicom.dcmread(dcm_file)
        img = dcm.pixel_array.astype(np.float32)

        # Apply DICOM windowing
        wc = float(getattr(dcm, "WindowCenter", 300))
        ww = float(getattr(dcm, "WindowWidth", 600))
        img = window_image(img, wc, ww)

        patch = safe_crop(img, x, y, PATCH_SIZE)

        # Skip if patch is nearly blank (to avoid pure black)
        if np.mean(patch) < 5:
            skipped += 1
            continue

        out_path = os.path.join(OUTPUT_DIR, f"{study_id}_{series_id}_{inst_no}_{condition}_{level}.png")
        ok = cv2.imwrite(out_path, patch)

        if ok:
            saved += 1
            # collect a few random samples for preview
            if len(sample_patches) < 5 and np.random.rand() < 0.001:
                sample_patches.append((patch, f"{condition}_{level}"))
        else:
            skipped += 1
    except Exception as e:
        skipped += 1
        continue

print(f"âœ… Done. Saved: {saved} patches | Skipped: {skipped}")
print(f"âœ… Output dir: {OUTPUT_DIR}")

# ============== Quick Visualization ==============

if sample_patches:
    plt.figure(figsize=(15,3))
    for i, (patch, title) in enumerate(sample_patches):
        plt.subplot(1, len(sample_patches), i+1)
        plt.imshow(patch, cmap='gray')
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
else:
    print("âš ï¸� No sample patches collected for preview. (Try increasing probability)")



import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATCH_DIR = "/kaggle/working/patches_windowed"

# List all patches
all_patches = os.listdir(PATCH_DIR)
print(f"Total patches found: {len(all_patches)}")

# Randomly sample up to 12 patches for visualization
sample_files = random.sample(all_patches, min(12, len(all_patches)))

plt.figure(figsize=(12, 8))
for i, fname in enumerate(sample_files):
    img_path = os.path.join(PATCH_DIR, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.subplot(3, 4, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(fname.split("_")[3])  # condition
    plt.axis('off')
plt.tight_layout()
plt.show()

# ---------- Black Pixel Check ----------
def is_black_patch(img, threshold=10, black_ratio=0.9):
    """
    Returns True if more than 90% of pixels are below intensity 10.
    """
    return np.mean(img < threshold) > black_ratio

black_count = 0
check_limit = min(1000, len(all_patches))  # check up to 1000 randomly

for fname in random.sample(all_patches, check_limit):
    img_path = os.path.join(PATCH_DIR, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None and is_black_patch(img):
        black_count += 1

black_ratio = black_count / check_limit * 100
print(f"\nğŸ§© Quality check summary:")
print(f"Checked {check_limit} random patches")
print(f"Black/empty patches: {black_count} ({black_ratio:.2f}%)")

if black_ratio > 20:
    print("âš ï¸� Warning: Too many black patches! You may need to adjust crop or window settings.")
else:
    print("âœ… Looks good! Most patches contain visible anatomical content.")
import os
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt

PATCH_DIR = "/kaggle/working/patches_windowed"

# List all patches
all_patches = os.listdir(PATCH_DIR)
print(f"Total patches found: {len(all_patches)}")

# Randomly sample up to 12 patches for visualization
sample_files = random.sample(all_patches, min(12, len(all_patches)))

plt.figure(figsize=(12, 8))
for i, fname in enumerate(sample_files):
    img_path = os.path.join(PATCH_DIR, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    plt.subplot(3, 4, i + 1)
    plt.imshow(img, cmap='gray')
    plt.title(fname.split("_")[3])  # condition
    plt.axis('off')
plt.tight_layout()
plt.show()

# ---------- Black Pixel Check ----------
def is_black_patch(img, threshold=10, black_ratio=0.9):
    """
    Returns True if more than 90% of pixels are below intensity 10.
    """
    return np.mean(img < threshold) > black_ratio

black_count = 0
check_limit = min(1000, len(all_patches))  # check up to 1000 randomly

for fname in random.sample(all_patches, check_limit):
    img_path = os.path.join(PATCH_DIR, fname)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is not None and is_black_patch(img):
        black_count += 1

black_ratio = black_count / check_limit * 100
print(f"\nğŸ§© Quality check summary:")
print(f"Checked {check_limit} random patches")
print(f"Black/empty patches: {black_count} ({black_ratio:.2f}%)")

if black_ratio > 20:
    print("âš ï¸� Warning: Too many black patches! You may need to adjust crop or window settings.")
else:
    print("âœ… Looks good! Most patches contain visible anatomical content.")



train_csv = pd.read_csv("/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification/train.csv")
print(train_csv.head())
print(train_csv.columns.tolist())



import pandas as pd
import os

# Paths
data_path = "/kaggle/input/rsna-2024-lumbar-spine-degenerative-classification"

# Load CSVs
coords_df = pd.read_csv(os.path.join(data_path, "train_label_coordinates.csv"))
train_df = pd.read_csv(os.path.join(data_path, "train.csv"))

# Define mapping from "condition" + "level" â†’ column name in train.csv
def get_column_name(row):
    cond = row["condition"]
    lvl = row["level"].replace("/", "_")
    
    if "Spinal Canal" in cond:
        return f"spinal_canal_stenosis_{lvl.lower()}"
    elif "Left Neural" in cond:
        return f"left_neural_foraminal_narrowing_{lvl.lower()}"
    elif "Right Neural" in cond:
        return f"right_neural_foraminal_narrowing_{lvl.lower()}"
    elif "Left Subarticular" in cond:
        return f"left_subarticular_stenosis_{lvl.lower()}"
    elif "Right Subarticular" in cond:
        return f"right_subarticular_stenosis_{lvl.lower()}"
    else:
        return None

coords_df["target_col"] = coords_df.apply(get_column_name, axis=1)

# Merge severity from train.csv
merged_df = coords_df.merge(
    train_df,
    on="study_id",
    how="left"
)

# Add actual severity labels from the right column
merged_df["severity"] = merged_df.apply(lambda r: r[r["target_col"]] if pd.notnull(r["target_col"]) else None, axis=1)

merged_df = merged_df[["study_id", "series_id", "instance_number", "condition", "level", "x", "y", "severity"]]
merged_df.head()


import os
import os
import pandas as pd

PATCH_DIR = "/kaggle/working/patches_windowed"  # âœ… use your real patch folder

# 1. Get filenames
patch_files = os.listdir(PATCH_DIR)
patch_df = pd.DataFrame({"filename": patch_files})

# 2. Parse study_id, series_id, instance_number, condition, level
def parse_filename(fname):
    parts = fname.replace(".png", "").split("_")
    study_id, series_id, instance_number = parts[:3]
    condition = "_".join(parts[3:-1])  # includes 'Stenosis' or 'Narrowing'
    level = parts[-1]                  # e.g., 'L4-L5'
    return pd.Series([study_id, series_id, int(instance_number), condition, level])

patch_df[["study_id", "series_id", "instance_number", "condition", "level"]] = \
    patch_df["filename"].apply(parse_filename)


def normalize_text(s):
    return (
        s.strip()
         .replace(" ", "_")
         .replace("/", "_")
         .replace("-", "_")
         .replace("__", "_")
         .lower()
    )

# Normalize text fields
merged_df["condition_norm"] = merged_df["condition"].apply(normalize_text)
merged_df["level_norm"] = merged_df["level"].apply(normalize_text)
patch_df["condition_norm"] = patch_df["condition"].apply(normalize_text)
patch_df["level_norm"] = patch_df["level"].apply(normalize_text)

# Ensure matching datatypes
for col in ["study_id", "series_id", "instance_number"]:
    merged_df[col] = merged_df[col].astype(str)
    patch_df[col] = patch_df[col].astype(str)

# Merge labels
train_ready = patch_df.merge(
    merged_df[
        ["study_id", "series_id", "instance_number", "condition_norm", "level_norm", "severity"]
    ],
    on=["study_id", "series_id", "instance_number", "condition_norm", "level_norm"],
    how="left"
)

# Drop unmatched
train_ready = train_ready.dropna(subset=["severity"]).reset_index(drop=True)

print("âœ… Patches matched correctly:", len(train_ready))
print("ğŸ”¸ Unique severity labels:", train_ready["severity"].unique())
train_ready.head()


# Encode string labels â†’ numeric
severity_map = {
    "Normal/Mild": 0,
    "Moderate": 1,
    "Severe": 2
}
train_ready["severity_encoded"] = train_ready["severity"].map(severity_map)

print(train_ready["severity_encoded"].value_counts())



# ---------- Fixed script (minimal changes) ----------
import os, random, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
from PIL import Image
import timm

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_curve, auc
)

# ----------------------------
# CONFIG
# ----------------------------
PATCH_DIR = "/kaggle/working/patches_windowed"
BATCH_SIZE = 32
NUM_EPOCHS = 15
LR = 1e-4
PATIENCE = 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
NUM_WORKERS = 4

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE.type == "cuda":
    torch.cuda.manual_seed_all(SEED)

# ----------------------------
# 1. LOAD METADATA (train_ready)
# ----------------------------
df = train_ready.copy()   # <-- your variable from earlier cell

# make sure we have filename
if "filename" not in df.columns:
    if "image_path" in df.columns:
        df["filename"] = df["image_path"].apply(lambda p: os.path.basename(p))
    else:
        raise RuntimeError("train_ready must contain 'filename' or 'image_path'.")

# severity encoding
if "severity" in df.columns:
    le = LabelEncoder()
    df["severity_str"] = df["severity"].astype(str)
    df["label"] = le.fit_transform(df["severity_str"])
    class_names = list(le.classes_)
elif "severity_encoded" in df.columns:
    df["label"] = df["severity_encoded"].astype(int)
    class_names = ["Normal/Mild", "Moderate", "Severe"]
else:
    raise RuntimeError("train_ready must contain 'severity' or 'severity_encoded'.")

num_classes = len(np.unique(df["label"]))
print("Classes:", class_names, "num_classes:", num_classes)

# ---- FIXED CLASS WEIGHTS 1, 2, 4 ----
if len(class_names) != 3:
    raise RuntimeError("Expected exactly 3 severity classes for weights [1,2,4].")

severity_weight_map = {
    "Normal/Mild": 1.0,
    "Moderate": 2.0,
    "Severe": 4.0,
}
class_weight_values = np.array(
    [severity_weight_map.get(c, 1.0) for c in class_names],
    dtype=np.float32
)
print("Using class weights:", dict(zip(class_names, class_weight_values)))


# ----------------------------
# 2. TRAIN / VAL / TEST SPLIT (by study_id if exists)
# ----------------------------
if "study_id" in df.columns:
    studies = df["study_id"].unique()
    np.random.shuffle(studies)
    n = len(studies)

    n_test = max(1, int(0.05 * n))  # 5% test
    test_studies = set(studies[:n_test])

    remaining = studies[n_test:]
    n_rem = len(remaining)
    n_val = int(0.2 * n_rem)        # 20% of remaining as val

    val_studies = set(remaining[:n_val])
    train_studies = set(remaining[n_val:])

    train_df = df[df["study_id"].isin(train_studies)].reset_index(drop=True)
    val_df   = df[df["study_id"].isin(val_studies)].reset_index(drop=True)
    test_df  = df[df["study_id"].isin(test_studies)].reset_index(drop=True)

else:
    # fallback if no study_id
    m = len(df)
    idx = np.arange(m)
    np.random.shuffle(idx)

    n_test = max(1, int(0.05 * m))
    test_idx = idx[:n_test]
    remaining_idx = idx[n_test:]

    n_val = int(0.2 * len(remaining_idx))
    val_idx = remaining_idx[:n_val]
    train_idx = remaining_idx[n_val:]

    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df   = df.iloc[val_idx].reset_index(drop=True)
    test_df  = df.iloc[test_idx].reset_index(drop=True)

print("Train rows:", len(train_df), "Val rows:", len(val_df), "Test rows:", len(test_df))
print("Train class counts:\n", train_df["label"].value_counts().sort_index())
print("Test class counts:\n",  test_df["label"].value_counts().sort_index())


# ----------------------------
# 3. DATASET + DATALOADERS  (FIXED: create train_loader, val_loader, test_loader)
# ----------------------------
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(8),
    transforms.ColorJitter(brightness=0.12, contrast=0.12),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

class SpineDataset(Dataset):
    def __init__(self, df, patch_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.patch_dir = patch_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.patch_dir, row["filename"])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = int(row["label"])
        return img, label

train_ds = SpineDataset(train_df, PATCH_DIR, transform=train_transform)
val_ds   = SpineDataset(val_df,   PATCH_DIR, transform=val_transform)
test_ds  = SpineDataset(test_df,  PATCH_DIR, transform=val_transform)

# Weighted sampler for train (you earlier built sample_weights; use it here)
class_counts = train_df["label"].value_counts().sort_index().values
class_weights_inv = 1.0 / (class_counts + 1e-8)
sample_weights = np.array([class_weights_inv[l] for l in train_df["label"]], dtype=np.float32)
sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

# Use sampler for train_loader (do NOT set shuffle=True when using sampler)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                          num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)


# ----------------------------
# 4. MODEL BUILDERS
# ----------------------------
def build_resnet50(num_classes):
    m = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

def build_mobilenetv2(num_classes):
    m = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    num_feats = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_feats, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return m

def build_densenet169(num_classes):
    m = models.densenet169(weights=models.DenseNet169_Weights.IMAGENET1K_V1)
    num_feats = m.classifier.in_features
    m.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(num_feats, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return m

def build_effnet_b0(num_classes):
    m = timm.create_model("efficientnet_b0", pretrained=True, num_classes=num_classes)
    return m

class ResNetEffB0Ensemble(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.resnet = build_resnet50(num_classes)
        self.effb0  = build_effnet_b0(num_classes)

    def forward(self, x):
        logits1 = self.resnet(x)
        logits2 = self.effb0(x)
        return (logits1 + logits2) / 2.0


# ----------------------------
# 5. TRAIN + EVAL FUNCTION
# ----------------------------
def train_and_evaluate(model, name, num_epochs=NUM_EPOCHS, lr=LR, patience=PATIENCE):
    model = model.to(DEVICE)

    # ---- FIXED CLASS WEIGHTS 1,2,4 ----
    cw_tensor = torch.tensor(class_weight_values, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=2, factor=0.5, verbose=False
    )

    best_val_loss = float("inf")
    patience_counter = 0

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_y_true = best_y_pred = best_y_prob = None

    for epoch in range(num_epochs):
        # ---- TRAIN ----
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)
            preds = outputs.argmax(1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total
        train_acc  = running_correct / running_total

        # ---- VALIDATION ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds, all_probs, all_labels = [], [], []

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                loss = criterion(outputs, labels)

                probs = torch.softmax(outputs, dim=1)
                preds = probs.argmax(1)

                val_loss += loss.item() * imgs.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        val_loss_epoch = val_loss / val_total
        val_acc_epoch  = val_correct / val_total

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss_epoch)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc_epoch)

        print(f"[{name}] Epoch {epoch+1}/{num_epochs} "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_loss_epoch:.4f} val_acc={val_acc_epoch:.4f}")

        scheduler.step(val_loss_epoch)

        if val_loss_epoch < best_val_loss - 1e-6:
            best_val_loss = val_loss_epoch
            patience_counter = 0
            best_y_true = np.array(all_labels)
            best_y_pred = np.array(all_preds)
            best_y_prob = np.array(all_probs)
            torch.save(model.state_dict(), f"best_{name}_val.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"[{name}] Early stopping at epoch {epoch+1}")
                break

    # reload best validation weights before returning
    model.load_state_dict(torch.load(f"best_{name}_val.pth", map_location=DEVICE))

    return history, best_y_true, best_y_pred, best_y_prob


def evaluate_on_test(model, name):
    model = model.to(DEVICE)
    model.eval()

    cw_tensor = torch.tensor(class_weight_values, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=cw_tensor)

    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_preds, all_probs, all_labels = [], [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = probs.argmax(1)

            test_loss += loss.item() * imgs.size(0)
            test_correct += (preds == labels).sum().item()
            test_total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    test_loss = test_loss / test_total if test_total else float('nan')
    test_acc  = test_correct / test_total if test_total else float('nan')
    print(f"[{name}] TEST: loss={test_loss:.4f} acc={test_acc:.4f}")

    return test_loss, test_acc, np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ----------------------------
# 6. RUN MODELS
# ----------------------------
models_dict = {
    "resnet50":            build_resnet50(num_classes),
    "mobilenet_v2":        build_mobilenetv2(num_classes),
    "densenet169":         build_densenet169(num_classes),
    "efficientnet_b0":     build_effnet_b0(num_classes),
}

results = {}

for name, model in models_dict.items():
    print("\n" + "="*60)
    print("Training (train/val):", name)
    start = time.time()

    history, y_true_val, y_pred_val, y_prob_val = train_and_evaluate(model, name)
    elapsed = (time.time() - start) / 60
    print(f"Done {name} train/val in {elapsed:.2f} min")

    # VAL REPORT
    print(f"\n{name} â€” Validation classification report:")
    print(classification_report(y_true_val, y_pred_val, target_names=class_names, digits=4))

    # Evaluate (no retrain) on test using the best validation weights already loaded
    test_loss, test_acc, y_true_test, y_pred_test, y_prob_test = evaluate_on_test(model, name)

    # store
    results[name] = {
        "history": history,
        "val_y_true": y_true_val,
        "val_y_pred": y_pred_val,
        "val_y_prob": y_prob_val,
        "test_y_true": y_true_test,
        "test_y_pred": y_pred_test,
        "test_y_prob": y_prob_test,
        "test_acc": test_acc,
        "test_loss": test_loss,
    }


    cm_val = confusion_matrix(y_true_val, y_pred_val)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_val, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Validation Confusion Matrix â€” {name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # VAL ROC (per model)
    y_true_val_bin = label_binarize(y_true_val, classes=list(range(num_classes)))
    plt.figure(figsize=(6,5))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_val_bin[:, i], y_prob_val[:, i])
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={model_auc:.2f})")
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.title(f"Validation ROC â€” {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ---- DIRECT TEST EVAL (NO RETRAIN) ----
    print(f"\nEvaluating on TEST set (no retraining) for {name}...")
    test_loss, test_acc, y_true_test, y_pred_test, y_prob_test = \
        evaluate_on_test(model, name)


    print(f"\n{name} â€” TEST classification report:")
    print(classification_report(y_true_test, y_pred_test,
                                target_names=class_names, digits=4))

    cm_test = confusion_matrix(y_true_test, y_pred_test)
    plt.figure(figsize=(5,4))
    sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Test Confusion Matrix â€” {name}")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.show()

    # TEST ROC (per model)
    y_true_test_bin = label_binarize(y_true_test, classes=list(range(num_classes)))
    plt.figure(figsize=(6,5))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_test_bin[:, i], y_prob_test[:, i])
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={model_auc:.2f})")
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.title(f"Test ROC â€” {name}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # store everything
    results[name] = {
        "history": history,
        "val_y_true": y_true_val,
        "val_y_pred": y_pred_val,
        "val_y_prob": y_prob_val,
        "test_y_true": y_true_test,
        "test_y_pred": y_pred_test,
        "test_y_prob": y_prob_test,
        "test_acc": test_acc,
        "test_loss": test_loss,
    }

# Validation loss comparison
plt.figure(figsize=(12,5))
max_epochs = max(len(r["history"]["val_loss"]) for r in results.values())

def pad(arr, n):
    return arr + [np.nan]*(n - len(arr))

for name, res in results.items():
    epochs = np.arange(1, max_epochs+1)
    val_loss = pad(res["history"]["val_loss"], max_epochs)
    plt.plot(epochs, val_loss, marker='o', label=name)
plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Validation Loss â€” All Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Validation accuracy comparison
plt.figure(figsize=(12,5))
for name, res in results.items():
    epochs = np.arange(1, max_epochs+1)
    val_acc = pad(res["history"]["val_acc"], max_epochs)
    plt.plot(epochs, val_acc, marker='o', label=name)
plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Validation Accuracy â€” All Models")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Validation ROC comparison per class
for class_idx, cname in enumerate(class_names):
    plt.figure(figsize=(7,6))
    for name, res in results.items():
        y_true = res["val_y_true"]
        y_prob = res["val_y_prob"]
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_prob[:, class_idx])
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={model_auc:.2f})")
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.title(f"Validation ROC Comparison â€” {cname}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# ----------------------------
# TEST COMPARISON PLOTS
# ----------------------------

# Test accuracy comparison (bar plot)
plt.figure(figsize=(8,5))
model_names = []
test_accs = []
for name, res in results.items():
    model_names.append(name)
    test_accs.append(res["test_acc"])

plt.bar(model_names, test_accs)
plt.ylabel("Test Accuracy")
plt.title("Test Accuracy â€” All Models")
plt.xticks(rotation=45, ha="right")
plt.grid(axis='y')
plt.tight_layout()
plt.show()

# Test ROC comparison per class (all models)
for class_idx, cname in enumerate(class_names):
    plt.figure(figsize=(7,6))
    for name, res in results.items():
        y_true = res["test_y_true"]
        y_prob = res["test_y_prob"]
        y_true_bin = label_binarize(y_true, classes=list(range(num_classes)))
        fpr, tpr, _ = roc_curve(y_true_bin[:, class_idx], y_prob[:, class_idx])
        model_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={model_auc:.2f})")
    plt.plot([0,1],[0,1],'k--',lw=1)
    plt.title(f"Test ROC Comparison â€” {cname}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


