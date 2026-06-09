!pip install -U ultralytics


# Run in a single notebook cell
!pip install -q ultralytics==8.0.114  # YOLOv8 (pick a working version)
!pip install -q grad-cam pydicom



!pip install torch==2.5.1 torchvision==0.20.1



pip install -U ultralytics


import os, sys, math, time, random, shutil, json
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF

import pydicom

# Paths - Kaggle competition dataset
INPUT_DIR = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection")
TRAIN_DICOM_DIR = INPUT_DIR / "train"
TEST_DICOM_DIR  = INPUT_DIR / "test"
TRAIN_CSV = INPUT_DIR / "train.csv"
SAMPLE_SUB = INPUT_DIR / "sample_submission.csv"

# Working directories (where we'll write converted images and labels, and save models)
WORK_DIR = Path("/kaggle/input/chest-xray/kaggle/working/vindr")
IMG_DIR = WORK_DIR / "images"
LAB_DIR = WORK_DIR / "labels"
MODEL_DIR = Path("/kaggle/working/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


import os, random
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import pydicom

# =====================
# CONFIG
# =====================
OUT_IMG_SIZE = (512, 512)   # None -> keep original resolution
MAX_IMAGES_PER_SPLIT = None  # set to an int for debugging
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Paths (update these according to Kaggle dataset mount)
TRAIN_CSV = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv")
SAMPLE_SUB = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv")
TRAIN_DICOM_DIR = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train")
TEST_DICOM_DIR = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/test")


# Create folders
for split in ["train", "val", "test"]:
    (IMG_DIR / split).mkdir(parents=True, exist_ok=True)
    (LAB_DIR / split).mkdir(parents=True, exist_ok=True)

# =====================
# Load train.csv
# =====================
df = pd.read_csv(TRAIN_CSV)
print("Total annotation rows:", len(df))
unique_train_image_ids = df['image_id'].unique().tolist()
print("Unique images with annotations:", len(unique_train_image_ids))

# Collect all dicom files in train dir
all_train_files = [f.stem for f in TRAIN_DICOM_DIR.iterdir() if f.suffix.lower() == '.dicom']
image_ids = sorted(list(set(all_train_files)))
print("Available dicom images in train folder:", len(image_ids))

# Train/val split (90/10)
random.shuffle(image_ids)
val_frac = 0.1
n_val = int(len(image_ids) * val_frac)
val_ids = image_ids[:n_val]
train_ids = image_ids[n_val:]
print("Train ids:", len(train_ids), "Val ids:", len(val_ids))

# =====================
# Conversion helpers
# =====================
def dicom_to_pil(dicom_path):
    ds = pydicom.dcmread(str(dicom_path))
    img = ds.pixel_array.astype(np.float32)
    lo, hi = np.percentile(img, (0.5, 99.5))
    img = np.clip(img, lo, hi)
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    img = (img * 255).astype(np.uint8)
    pil = Image.fromarray(img).convert("RGB")
    return pil

def write_yolo_label(image_id, boxes_for_image, out_label_path, img_w, img_h):
    lines = []
    for box in boxes_for_image:
        x_min, y_min, x_max, y_max = box['x_min'], box['y_min'], box['x_max'], box['y_max']
        class_id = int(box['class_id'])
        x_min = max(0, x_min); y_min = max(0, y_min)
        x_max = min(img_w-1, x_max); y_max = min(img_h-1, y_max)
        width = x_max - x_min
        height = y_max - y_min
        if width <= 0 or height <= 0: 
            continue
        cx = x_min + width/2.0
        cy = y_min + height/2.0
        cx_norm = cx / img_w
        cy_norm = cy / img_h
        w_norm = width / img_w
        h_norm = height / img_h
        lines.append(f"{class_id} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f}")
    if len(lines) == 0:
        open(out_label_path, 'w').close()
        return
    with open(out_label_path, 'w') as f:
        f.write("\n".join(lines))

# =====================
# Converters with SKIP logic
# =====================
def convert_split(ids_list, split_name, max_images=None):
    pbar = ids_list if max_images is None else ids_list[:max_images]
    count, skipped = 0, 0
    for img_id in pbar:
        out_img_path = IMG_DIR / split_name / f"{img_id}.jpg"
        out_lbl_path = LAB_DIR / split_name / f"{img_id}.txt"

        # âœ… Skip if already exists
        if out_img_path.exists() and out_lbl_path.exists():
            skipped += 1
            continue

        dicom_path = TRAIN_DICOM_DIR / f"{img_id}.dicom"
        if not dicom_path.exists():
            continue
        try:
            pil = dicom_to_pil(dicom_path)
        except Exception as e:
            print("Failed to read", dicom_path, e)
            continue

        if OUT_IMG_SIZE is not None:
            pil = pil.resize(OUT_IMG_SIZE)

        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_img_path, quality=95)

        boxes = df[df['image_id'] == img_id]
        write_yolo_label(img_id, boxes.to_dict('records'), out_lbl_path, pil.width, pil.height)

        count += 1
        if count % 500 == 0:
            print(f"{split_name}: converted {count} new images (skipped {skipped})...")
    print(f"Finished {split_name} -> converted {count}, skipped {skipped}.")

def convert_test(ids_list, max_images=None):
    pbar = ids_list if max_images is None else ids_list[:max_images]
    count, skipped = 0, 0
    for img_id in pbar:
        out_img_path = IMG_DIR / "test" / f"{img_id}.jpg"
        out_lbl_path = LAB_DIR / "test" / f"{img_id}.txt"

        # âœ… Skip if already exists
        if out_img_path.exists() and out_lbl_path.exists():
            skipped += 1
            continue

        dicom_path = TEST_DICOM_DIR / f"{img_id}.dicom"
        if not dicom_path.exists():
            continue
        try:
            pil = dicom_to_pil(dicom_path)
        except Exception as e:
            print("Failed to read", dicom_path, e)
            continue

        if OUT_IMG_SIZE is not None:
            pil = pil.resize(OUT_IMG_SIZE)

        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        pil.save(out_img_path, quality=95)

        # empty label
        open(out_lbl_path, 'w').close()

        count += 1
        if count % 500 == 0:
            print(f"test: converted {count} new images (skipped {skipped})...")
    print(f"Finished test -> converted {count}, skipped {skipped}.")

# =====================
# Run conversions
# =====================
print("Converting TRAIN split ...")
convert_split(train_ids, "train", max_images=MAX_IMAGES_PER_SPLIT)

print("Converting VAL split ...")
convert_split(val_ids, "val", max_images=MAX_IMAGES_PER_SPLIT)

print("Converting TEST split ...")
sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub['image_id'].tolist()
convert_test(test_ids, max_images=MAX_IMAGES_PER_SPLIT)



import os

def print_folder_structure(root_dir, indent=""):
    """
    Prints only the folder structure (ignores files).
    """
    items = sorted([item for item in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, item))])
    for i, item in enumerate(items):
        path = os.path.join(root_dir, item)
        connector = "â””â”€â”€ " if i == len(items) - 1 else "â”œâ”€â”€ "
        print(indent + connector + item)
        print_folder_structure(path, indent + ("    " if i == len(items) - 1 else "â”‚   "))


print_folder_structure("/kaggle/input/")



# Run this single shell cell
!pip install -q ultralytics==8.0.114 grad-cam pydicom timm



# Python cell
from pathlib import Path
import os, random, time, json, math
import numpy as np, pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torchvision.transforms.functional as TF

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report, confusion_matrix

# ultralytics
from ultralytics import YOLO

# global paths (update if needed)
WORK_DIR = Path("/kaggle/working/vindr")
IMG_DIR = WORK_DIR / "images"
LAB_DIR = WORK_DIR / "labels"
MODEL_DIR = Path("/kaggle/working/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_CSV = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/train.csv")
SAMPLE_SUB = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/sample_submission.csv")
TEST_DICOM_DIR = Path("/kaggle/input/vinbigdata-chest-xray-abnormalities-detection/test")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



# Build class list (match earlier)
class_names = [
    "Aortic_enlargement","Atelectasis","Calcification","Cardiomegaly",
    "Consolidation","ILD","Infiltration","Lung_Opacity","Nodule_Mass",
    "Other_lesion","Pleural_effusion","Pleural_thickening",
    "Pneumothorax","Pulmonary_fibrosis","No_finding"
]
NUM_CLASSES = len(class_names)

# Read train.csv and build multi-hot label dict
df = pd.read_csv(TRAIN_CSV)
# group by image_id
targets = {}
for img_id, g in df.groupby("image_id"):
    vec = np.zeros(NUM_CLASSES, dtype=np.float32)
    for cid in g['class_id'].values:
        vec[int(cid)] = 1.0
    targets[img_id] = vec

# Some images may be missing in df (no annotation) -> treat as No_finding
# Ensure that for converted images, we have label vectors
train_img_dir = IMG_DIR / "train"
val_img_dir   = IMG_DIR / "val"

train_ids = [p.stem for p in train_img_dir.glob("*.jpg")]
val_ids   = [p.stem for p in val_img_dir.glob("*.jpg")]

# if an image not in targets -> treat as no finding (class 14 = No_finding)
for img in train_ids + val_ids:
    if img not in targets:
        vec = np.zeros(NUM_CLASSES, dtype=np.float32)
        vec[14] = 1.0
        targets[img] = vec

# Dataset class
class MultiLabelCXRDataset(Dataset):
    def __init__(self, image_dir, img_ids, targets_dict, transform=None):
        self.image_dir = Path(image_dir)
        self.img_ids = img_ids
        self.targets = targets_dict
        self.transform = transform

    def __len__(self): return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_path = self.image_dir / f"{img_id}.jpg"
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.targets[img_id], dtype=torch.float32)
        return img, label, img_id



BATCH = 32

train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

train_ds = MultiLabelCXRDataset(train_img_dir, train_ids, targets, transform=train_tfms)
val_ds   = MultiLabelCXRDataset(val_img_dir, val_ids, targets, transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
val_loader   = DataLoader(val_ds, batch_size=BATCH, shuffle=False, num_workers=4, pin_memory=True)
print("Train / Val sizes:", len(train_ds), len(val_ds))



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.amp import autocast, GradScaler

# Model
NUM_CLASSES = 15  # change as needed
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_cls = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model_cls.fc = nn.Linear(model_cls.fc.in_features, NUM_CLASSES)
model_cls = model_cls.to(device)

# Loss, optimizer, scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model_cls.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# AMP
scaler = GradScaler("cuda")



import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.models import resnet50, ResNet50_Weights
from torch.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# ğŸ”¹ Config
NUM_CLASSES = 15   # VinBig competition has 14 findings
EPOCHS = 20
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ğŸ”¹ Model
model_cls = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model_cls.fc = nn.Linear(model_cls.fc.in_features, NUM_CLASSES)  # multilabel
model_cls = model_cls.to(device)

# ğŸ”¹ Loss, optimizer, scheduler
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model_cls.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# ğŸ”¹ AMP scaler
scaler = GradScaler("cuda")

best_val_loss = float("inf")
best_path = MODEL_DIR / "resnet50_multilabel_best.pth"

for epoch in range(EPOCHS):
    print(f"\n===== Epoch {epoch+1}/{EPOCHS} =====")
    
    # ================= TRAIN =================
    model_cls.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    for imgs, labels, _ in train_loader:  # assuming your dataset returns (img, label, id)
        imgs, labels = imgs.to(device), labels.to(device).float()
        optimizer.zero_grad()

        with autocast("cuda"):
            outputs = model_cls(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)

        # collect preds for metrics
        preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
        all_preds.append(preds)
        all_labels.append(labels.int().cpu())

    # concat predictions
    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()

    train_loss = running_loss / len(train_loader.dataset)
    train_acc = accuracy_score(all_labels, all_preds)
    train_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    train_prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    train_rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    # ================= VALIDATION =================
    model_cls.eval()
    val_loss = 0.0
    val_preds, val_labels = [], []

    with torch.no_grad():
        for imgs, labels, _ in val_loader:
            imgs, labels = imgs.to(device), labels.to(device).float()
            with autocast("cuda"):
                outputs = model_cls(imgs)
                loss = criterion(outputs, labels)
            val_loss += loss.item() * imgs.size(0)

            preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
            val_preds.append(preds)
            val_labels.append(labels.int().cpu())

    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = accuracy_score(val_labels, val_preds)
    val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
    val_prec = precision_score(val_labels, val_preds, average="macro", zero_division=0)
    val_rec = recall_score(val_labels, val_preds, average="macro", zero_division=0)

    # ğŸ”¹ Scheduler step
    scheduler.step(val_loss)

    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | F1: {train_f1:.4f} | "
          f"Prec: {train_prec:.4f} | Rec: {train_rec:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | "
          f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")

    # save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model_cls.state_dict(), best_path)
        print("âœ… Saved best model at", best_path)

# Save final model
torch.save(model_cls.state_dict(), MODEL_DIR / "resnet50_multilabel_final.pth")
print("ğŸ�� Training finished. Best val loss:", best_val_loss)



# load best
model_cls.load_state_dict(torch.load(MODEL_DIR/"resnet50_multilabel_best.pth", map_location=device))
model_cls.eval()

y_true = []
y_prob = []

with torch.no_grad():
    for imgs, labels, _ in val_loader:
        imgs = imgs.to(device)
        out = model_cls(imgs)
        probs = torch.sigmoid(out).cpu().numpy()
        y_prob.append(probs)
        y_true.append(labels.numpy())

y_prob = np.vstack(y_prob)
y_true = np.vstack(y_true)

# threshold per class 0.5
y_pred = (y_prob >= 0.5).astype(int)

# metrics: per-class F1, precision, recall and macro/micro
per_class_f1 = []
per_class_prec = []
per_class_rec = []
for i in range(NUM_CLASSES):
    p = precision_score(y_true[:,i], y_pred[:,i], zero_division=0)
    r = recall_score(y_true[:,i], y_pred[:,i], zero_division=0)
    f = f1_score(y_true[:,i], y_pred[:,i], zero_division=0)
    per_class_f1.append(f); per_class_prec.append(p); per_class_rec.append(r)
    print(f"{class_names[i]:20s}  Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}")

macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
print("Macro F1:", macro_f1, "Micro F1:", micro_f1)

# show confusion-like stats for "No_finding" vs any finding (binary)
true_any = (y_true[:, :14].sum(axis=1) > 0).astype(int)  # any of first 14 classes
pred_any = (y_pred[:, :14].sum(axis=1) > 0).astype(int)
print("Overall detection of any abnormality - Acc/F1/Rec:", accuracy_score(true_any, pred_any),
      f1_score(true_any, pred_any), recall_score(true_any, pred_any))



# =========================
# Test / Confusion / Grad-CAM
# =========================
# Run this in the same environment where model_cls, val_loader, class_names, MODEL_DIR, device exist.

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import os
from pathlib import Path

# For Grad-CAM
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# Output folder for visuals
OUT_VIS = Path("/kaggle/working/vis")
OUT_VIS.mkdir(parents=True, exist_ok=True)

# ------------- 1) Load the best model weights ----------------
best_path = MODEL_DIR / "resnet50_multilabel_best.pth"
print("Loading model from:", best_path)
state = torch.load(best_path, map_location=device)
# state is a state_dict (you saved model.state_dict())
try:
    model_cls.load_state_dict(state)
except RuntimeError:
    # maybe saved as dict with 'model_state_dict'
    if isinstance(state, dict) and 'model_state_dict' in state:
        model_cls.load_state_dict(state['model_state_dict'])
    else:
        raise
model_cls.to(device)
model_cls.eval()
print("Model loaded.")

# ------------- 2) Run inference on validation set --------------
y_true_list = []
y_prob_list = []
img_ids_list = []

with torch.no_grad():
    for imgs, labels, img_ids in val_loader:
        imgs = imgs.to(device)
        logits = model_cls(imgs)                     # shape (B, C)
        probs = torch.sigmoid(logits).cpu().numpy() # multilabel probs
        y_prob_list.append(probs)
        y_true_list.append(labels.numpy())
        img_ids_list.extend(img_ids)

y_prob = np.vstack(y_prob_list)   # (N, C)
y_true = np.vstack(y_true_list).astype(int)  # (N, C)

# Choose threshold (0.5 default)
TH = 0.5
y_pred = (y_prob >= TH).astype(int)

print("Shapes: y_true", y_true.shape, "y_prob", y_prob.shape, "y_pred", y_pred.shape)

# ------------- 3) Per-class metrics ----------------
per_class_prec = []
per_class_rec = []
per_class_f1 = []
per_class_support = y_true.sum(axis=0).astype(int)

for i, cname in enumerate(class_names):
    p = precision_score(y_true[:, i], y_pred[:, i], zero_division=0)
    r = recall_score(y_true[:, i], y_pred[:, i], zero_division=0)
    f = f1_score(y_true[:, i], y_pred[:, i], zero_division=0)
    per_class_prec.append(p)
    per_class_rec.append(r)
    per_class_f1.append(f)
    print(f"{i:02d} {cname:20s} | Precision: {p:.3f}  Recall: {r:.3f}  F1: {f:.3f}  Support: {per_class_support[i]}")

macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
micro_f1 = f1_score(y_true, y_pred, average='micro', zero_division=0)
print("\nMacro F1:", macro_f1, "Micro F1:", micro_f1)

# ------------- 4) Plot per-class F1 bar chart ---------------
plt.figure(figsize=(12,4))
plt.bar(range(len(class_names)), per_class_f1, tick_label=class_names)
plt.xticks(rotation=45, ha='right')
plt.ylabel("F1 score")
plt.title("Per-class F1 on Validation")
plt.tight_layout()
plt.savefig(OUT_VIS / "per_class_f1.png", dpi=150)
plt.show()

# ------------- 5) Binary confusion matrix: No_finding vs Any finding -------------
# In your class list "No_finding" is last (index 14). Adjust if different.
NO_FIND_IDX = class_names.index("No_finding")
true_any = (y_true[:, :NO_FIND_IDX].sum(axis=1) > 0).astype(int)  # any abnormality in first 14 classes
pred_any = (y_pred[:, :NO_FIND_IDX].sum(axis=1) > 0).astype(int)

cm = confusion_matrix(true_any, pred_any)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No finding (true/neg)", "Any finding (true/pos)"])
fig, ax = plt.subplots(figsize=(5,4))
disp.plot(ax=ax)
plt.title("Binary confusion matrix: any abnormality vs no_finding")
plt.savefig(OUT_VIS / "binary_confusion_any_vs_none.png", dpi=150)
plt.show()

# ------------- 6) Per-class confusion-like numbers (TP/FP/FN/TN) -------------
tp = np.logical_and(y_true == 1, y_pred == 1).sum(axis=0)
fp = np.logical_and(y_true == 0, y_pred == 1).sum(axis=0)
fn = np.logical_and(y_true == 1, y_pred == 0).sum(axis=0)
tn = np.logical_and(y_true == 0, y_pred == 0).sum(axis=0)

# Save a CSV of per-class stats
import pandas as pd
per_class_df = pd.DataFrame({
    "class": class_names,
    "support": per_class_support,
    "tp": tp,
    "fp": fp,
    "fn": fn,
    "tn": tn,
    "precision": per_class_prec,
    "recall": per_class_rec,
    "f1": per_class_f1
})
per_class_df.to_csv(OUT_VIS / "per_class_stats.csv", index=False)
print("Saved per-class stats to", OUT_VIS / "per_class_stats.csv")
per_class_df.head(15)

# ------------- 7) (Optional) AUC per class -------------
# Only valid if ground-truth has both positive and negative examples for the class
from sklearn.metrics import roc_auc_score
auc_per_class = []
for i in range(len(class_names)):
    try:
        auc = roc_auc_score(y_true[:, i], y_prob[:, i])
    except Exception:
        auc = np.nan
    auc_per_class.append(auc)
print("\nPer-class AUCs (NaN means not computable):")
for i, a in enumerate(auc_per_class):
    print(f"{i:02d} {class_names[i]:20s} AUC: {a}")

# Save AUCs
pd.DataFrame({"class": class_names, "auc": auc_per_class}).to_csv(OUT_VIS / "per_class_auc.csv", index=False)



# ------------------ Corrected Grad-CAM block ------------------
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# make sure model is on the desired device (you already did this earlier)
# model_cls.to(device)
model_cls.eval()

# choose target layers for ResNet50: layer4 is the last conv block
target_layers = [model_cls.layer4]

# helper to convert PIL to float RGB array [0,1]
def pil_to_float_rgb(img_pil, size=(224,224)):
    img = img_pil.resize(size)
    arr = np.array(img).astype(np.float32) / 255.0
    if arr.ndim == 2:
        arr = np.stack([arr]*3, axis=-1)
    return arr

# Build list of samples to visualize (re-using y_true, img_ids_list from earlier inference)
samples_to_visualize = []
for cls_idx in range(len(class_names)):
    inds = np.where(y_true[:, cls_idx] == 1)[0]
    if len(inds) == 0:
        continue
    idx = inds[0]
    samples_to_visualize.append((idx, cls_idx))
    if len(samples_to_visualize) >= 40:
        break

print("Will generate Grad-CAM for", len(samples_to_visualize), "samples.")

# Create GradCAM object WITHOUT use_cuda argument (works with current version)
cam = GradCAM(model=model_cls, target_layers=target_layers)  # no use_cuda

# Run Grad-CAM for each sample
from PIL import Image as PILImage
for sample_idx, target_cls in samples_to_visualize:
    img_id = img_ids_list[sample_idx]
    img_path = val_img_dir / f"{img_id}.jpg"
    pil = PILImage.open(img_path).convert("RGB")

    # preprocess exactly as for validation (resize + normalize)
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    input_tensor = preprocess(pil).unsqueeze(0).to(device)

    # target is the class index we want visualization for
    targets = [ClassifierOutputTarget(target_cls)]
    # compute cam (returns HxW numpy)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    # overlay on the original (float RGB 0..1)
    rgb_float = pil_to_float_rgb(pil, size=(grayscale_cam.shape[1], grayscale_cam.shape[0]))
    cam_image = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    out_file = OUT_VIS / f"gradcam_img_{img_id}_class_{class_names[target_cls]}.png"
    PILImage.fromarray(cam_image).save(out_file)
    print("Saved grad-cam to", out_file)

# cleanup
del cam
print("Grad-CAM generation done. Visuals are in:", OUT_VIS)



# Grad-CAM visualization cell (adapted for your code)
# Assumes: model_cls (or model), device, val_loader, val_img_dir, class_names exist.

!pip install -q grad-cam

from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, AblationCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torchvision.transforms as T
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image

OUT_VIS = Path("/kaggle/working/vis")
OUT_VIS.mkdir(parents=True, exist_ok=True)

# ---------- Config ----------
# Choose sample index: set SAMPLE_INDEX to an int (0..N-1) to visualize that sample from val_loader
# or set SAMPLE_INDEX = None to pick a random sample.
SAMPLE_INDEX = None   # e.g. 0 or None for random
# Choose target class index to visualize. If None -> uses model's top predicted class for that image.
TARGET_CLASS = None   # e.g. 0..14 or None
# Whether to use GradCAMPlusPlus (sometimes sharper) fallback to GradCAM if not available
USE_CAM_METHOD = "gradcam"  # options: "gradcam", "gradcampp"
# ------------------------------------------------

# helper: inverse normalization (to get image for plotting)
mean = np.array([0.485, 0.456, 0.406])
std  = np.array([0.229, 0.224, 0.225])
def denormalize_tensor(tensor):   # tensor: C,H,W in torch
    arr = tensor.cpu().numpy()
    arr = np.transpose(arr, (1,2,0))  # HWC
    arr = (arr * std[None,None,:]) + mean[None,None,:]
    arr = np.clip(arr, 0.0, 1.0)
    return arr

# pick one sample from validation set
all_imgs = []
all_labels = []
all_ids = []
for imgs, labels, ids in val_loader:
    # store a batch at a time (not memory heavy for single sample selection)
    all_imgs.append(imgs)       # tensor B,C,H,W normalized
    all_labels.append(labels)   # tensor B,C
    all_ids.extend(ids)
# flatten batches to list
imgs_tensor = torch.cat(all_imgs, dim=0)        # N,C,H,W
labels_tensor = torch.cat(all_labels, dim=0)    # N,C
N = imgs_tensor.shape[0]

if N == 0:
    raise RuntimeError("Validation loader is empty or not accessible.")

if SAMPLE_INDEX is None:
    idx = random.randrange(N)
else:
    idx = int(SAMPLE_INDEX) % N

input_tensor = imgs_tensor[idx:idx+1].to(device)   # 1,C,H,W
gt_vector = labels_tensor[idx].cpu().numpy()
img_id = all_ids[idx]

# if TARGET_CLASS None -> choose top predicted class index for this sample
with torch.no_grad():
    logits = model_cls(input_tensor.to(device))             # shape 1,C
    probs = torch.sigmoid(logits).cpu().numpy()[0]         # multilabel probabilities

if TARGET_CLASS is None:
    # pick class with highest probability (argmax)
    target_cls = int(np.argmax(probs))
else:
    target_cls = int(TARGET_CLASS)

print(f"Visualizing sample idx={idx} img_id={img_id} target_class={target_cls} ({class_names[target_cls]})")
print("Top predicted probs (first 6):", probs[:6])

# prepare RGB image (float 0..1) to overlay CAM onto.
# We prefer to load original JPG for better quality (val_img_dir exists) if available.
img_path = Path(val_img_dir) / f"{img_id}.jpg"
if img_path.exists():
    pil = Image.open(img_path).convert("RGB")
    rgb_for_overlay = np.array(pil.resize((input_tensor.shape[3], input_tensor.shape[2]))).astype(np.float32) / 255.0
else:
    # fallback: use the denormalized tensor from loader
    rgb_for_overlay = denormalize_tensor(input_tensor[0]).astype(np.float32)

# choose target layers for ResNet50: layer4 (last conv block)
# For ResNet50 the last conv block is model_cls.layer4; for ResNet18 it's similar.
try:
    target_layers = [model_cls.layer4]
except Exception:
    # fallback: if your model variable is named differently
    try:
        target_layers = [model.layer4]
    except Exception:
        raise RuntimeError("Could not find layer4 on the model. Inspect your model to select appropriate target layer.")

# Instantiate CAM object robustly across grad-cam versions
cam = None
if USE_CAM_METHOD.lower() == "gradcampp":
    try:
        cam = GradCAMPlusPlus(model=model_cls, target_layers=target_layers)
    except Exception:
        cam = GradCAM(model=model_cls, target_layers=target_layers)
else:
    try:
        cam = GradCAM(model=model_cls, target_layers=target_layers)
    except TypeError:
        # some versions accept device arg, try passing device string
        try:
            cam = GradCAM(model=model_cls, target_layers=target_layers, device=str(device))
        except Exception:
            cam = GradCAMPlusPlus(model=model_cls, target_layers=target_layers)

# Build target (ClassifierOutputTarget expects the class index for classification)
targets = [ClassifierOutputTarget(target_cls)]

# compute cam (returns numpy HxW for each image)
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]  # HxW

# overlay
cam_image = show_cam_on_image(rgb_for_overlay, grayscale_cam, use_rgb=True)

# Plot side-by-side original and cam overlay
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(12,6))
ax1.imshow(rgb_for_overlay)
ax1.set_title(f"Original - id:{img_id}\nGT positive classes: {[class_names[i] for i,v in enumerate(gt_vector) if v==1]}")
ax1.axis('off')

ax2.imshow(cam_image)
ax2.set_title(f"Grad-CAM -> {class_names[target_cls]} (pred prob {probs[target_cls]:.3f})")
ax2.axis('off')

plt.tight_layout()
out_file = OUT_VIS / f"gradcam_sample_{img_id}_class_{class_names[target_cls].replace(' ','_')}.png"
plt.savefig(out_file, dpi=150)
plt.show()

print("Saved visualization to:", out_file)



# LAB_DIR structure: labels/{train,val,test}/{image_id}.txt contains lines "class_id cx cy w h"
# We will create a single-class labels folder for YOLO training: labels_yolo_single/{train,val,test}
LABEL_SINGLE_DIR = WORK_DIR / "labels_single"
LABEL_SINGLE_DIR.mkdir(parents=True, exist_ok=True)

for split in ["train","val","test"]:
    (LABEL_SINGLE_DIR / split).mkdir(parents=True, exist_ok=True)
    src_dir = LAB_DIR / split
    dst_dir = LABEL_SINGLE_DIR / split
    for txt in src_dir.glob("*.txt"):
        dst_txt = dst_dir / txt.name
        with open(txt, 'r') as f:
            lines = [l.strip() for l in f if l.strip()]
        out_lines = []
        for l in lines:
            # l = "class_id cx cy w h" (class_id from 0..14)
            parts = l.split()
            cls = int(parts[0])
            if cls == 14:
                # No finding -> skip (no boxes)
                continue
            # else map any class -> 0 (abnormal)
            cx, cy, w, h = parts[1], parts[2], parts[3], parts[4]
            out_lines.append(f"0 {cx} {cy} {w} {h}")
        # write out (possibly empty)
        with open(dst_txt, 'w') as fo:
            fo.write("\n".join(out_lines))
print("Converted labels to single-class in", LABEL_SINGLE_DIR)



import yaml
yolo_yaml = {
    "path": str(WORK_DIR),   # base path
    "train": "images/train",
    "val":   "images/val",
    "test":  "images/test",
    "nc": 1,
    "names": ["abnormal"]
}
YAML_PATH = WORK_DIR / "yolov8_abnormal.yaml"
with open(YAML_PATH, 'w') as f:
    yaml.dump(yolo_yaml, f)
print("Saved YAML:", YAML_PATH)



!pip install -U ultralytics



# choose a YOLO model - yolov8n (nano) for quick test, switch to yolov8m or yolov8l for better accuracy
yolo_model = YOLO("yolov8n.pt")  # or path to a trained weights
# Train: set epochs small to test, increase for final runs
results = yolo_model.train(data=str(YAML_PATH), epochs=20, imgsz=640, batch=8, project=str(WORK_DIR/"yolov8_runs"), name="abnormal_singleclass")
print("YOLO training finished. Check outputs in", WORK_DIR/"yolov8_runs")





