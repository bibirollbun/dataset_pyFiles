# imports & global config

import os
import json
import time
from math import inf

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix,precision_recall_fscore_support,roc_auc_score,average_precision_score)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models

import torch.cuda.amp as amp
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Kaggle herbarium data root
DATA_DIR = "/kaggle/input/herbarium-2022-fgvc9"

train_meta_path = os.path.join(DATA_DIR, "train_metadata.json")
test_meta_path  = os.path.join(DATA_DIR, "test_metadata.json")

print("DATA_DIR:", DATA_DIR)



with open(train_meta_path, "r") as f:
    train_meta = json.load(f)

ann_df = pd.DataFrame(train_meta["annotations"])
img_df = pd.DataFrame(train_meta["images"])
cat_df = pd.DataFrame(train_meta["categories"])

merged = ann_df.merge(img_df, on="image_id", how="left")
merged = merged.merge(cat_df, on="category_id", how="left")

merged["image_path"] = merged["file_name"].apply(
    lambda fn: os.path.join(DATA_DIR, "train_images", fn)
)

core_cols = [
    "image_id",
    "image_path",
    "category_id",
    "genus_id",
    "family",
    "genus",
    "species",
    "scientificName",
]

core_df = merged[core_cols].copy()
print("core_df shape:", core_df.shape)
core_df.head()



toxic_genus_list = [
    "Toxicodendron", 
    "Euphorbia",     
    "Urtica",         
    "Cicuta",         
    "Conium",        
    "Heracleum",     
]

core_df["do_not_touch"] = core_df["genus"].isin(toxic_genus_list).astype(int)

print("Toxic genera used:", toxic_genus_list)
print("\ndo_not_touch counts:")
print(core_df["do_not_touch"].value_counts())
core_df.head()



TEST_FRACTION = 0.05

strat_col = core_df["do_not_touch"]
rest_df, test_df = train_test_split(
    core_df,
    test_size=TEST_FRACTION,
    random_state=42,
    stratify=strat_col,
)

print("Realistic test size:", len(test_df))
print(test_df["do_not_touch"].value_counts())

# Training pool = rest_df
toxic_df = rest_df[rest_df["do_not_touch"] == 1].reset_index(drop=True)
safe_df  = rest_df[rest_df["do_not_touch"] == 0].reset_index(drop=True)

print("\nToxic (train pool) count:", len(toxic_df))
print("Safe  (train pool) count:", len(safe_df))


image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

strong_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.75, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
        saturation=0.25,
        hue=0.03,
    ),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

class PlantDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row["image_path"]

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            img = self.transform(img)

        toxic_label = int(row["do_not_touch"])
        return img, torch.tensor(toxic_label, dtype=torch.long)


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# base efficienet with imagenet weights
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(in_features, 1)  
)
model = model.to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
PATIENCE = 4

# final checkpoint from the half decent notebook i have decided to place my faith in
PRETRAINED_CKPT_PATH = "/kaggle/input/runv1-rotation-checkpoints/final_rich_checkpoint.pth" 

if os.path.exists(PRETRAINED_CKPT_PATH):
    print(f"Loading pre-trained checkpoint from: {PRETRAINED_CKPT_PATH}")
    ckpt = torch.load(PRETRAINED_CKPT_PATH, map_location=device)

    # Try to guess whether it's a pure state_dict or full checkpoint
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
        print("loaded model state dict from checkpoint")
    else:
        model.load_state_dict(ckpt)
        print("raw state from checkpoint")
else:
    print("No good, time to starting over ")



BATCH_SIZE   = 64
NUM_WORKERS  = 4
NUM_ROTATIONS = 40
SAFE_PER_ROTATION = 20000

safe_sample_initial = safe_df.sample(n=len(toxic_df), random_state=42)
balanced_df = (
    pd.concat([toxic_df, safe_sample_initial])
    .sample(frac=1.0, random_state=42)
    .reset_index(drop=True)
)

train_df, val_df = train_test_split(
    balanced_df,
    test_size=0.2,
    random_state=42,
    stratify=balanced_df["do_not_touch"]
)

train_dataset = PlantDataset(train_df, transform=image_transforms)
val_dataset   = PlantDataset(val_df,   transform=image_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

test_dataset = PlantDataset(test_df, transform=image_transforms)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

print("Train batches:", len(train_loader))
print("Val batches:  ", len(val_loader))
print("Realistic test samples:", len(test_df))
print("Realistic test batches:", len(test_loader))


def compute_metrics(y_true, y_probs, threshold=0.5):
    y_pred = (y_probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred)

    metrics = {
        "precision": float(precision),
        "recall":    float(recall),
        "f1":        float(f1),
        "confusion_matrix": cm.tolist(),
    }

    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_probs))
    except Exception:
        metrics["roc_auc"] = None

    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_probs))
    except Exception:
        metrics["pr_auc"] = None

    return metrics

def evaluate_loader_timed(model, loader, device, use_amp=True, print_progress_every=500):
    model.eval()
    all_probs = []
    all_labels = []
    batch_times = []
    start = time.time()

    with torch.no_grad():
        for i, (images, labels) in enumerate(loader, start=1):
            t0 = time.time()
            images = images.to(device)
            labels = labels.to(device)

            if use_amp and device.startswith("cuda"):
                with torch.amp.autocast('cuda'):
                    outputs = model(images)
            else:
                outputs = model(images)

            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

            batch_time = time.time() - t0
            batch_times.append(batch_time)

            if i % print_progress_every == 0:
                elapsed = time.time() - start
                avg_b   = sum(batch_times) / len(batch_times)
                processed = i * loader.batch_size
                rate = processed / elapsed if elapsed > 0 else 0.0
                print(f"[eval] batches={i} avg_batch_s={avg_b:.4f} processed={processed} rate_samples/s={rate:.2f}")

    if not all_probs:
        return np.array([]), np.array([]), 0.0

    all_probs  = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    avg_batch_time = sum(batch_times) / len(batch_times) if batch_times else 0.0

    return all_labels, all_probs, avg_batch_time



safe_shuffled = safe_df.sample(frac=1.0, random_state=123).reset_index(drop=True)
rot_model_path = "balanced_do_not_touch_model_rotating_safe_20rotations_v2.pth"
best_val_f1 = 0.0   # track best rotating-validation score

print("\nstarting rotating training \n")

for rot in range(NUM_ROTATIONS):
    print("=" * 60)
    print(f"=== ROTATION {rot+1}/{NUM_ROTATIONS} ===")

    start = (rot * SAFE_PER_ROTATION) % len(safe_shuffled)
    end   = start + SAFE_PER_ROTATION

    if end <= len(safe_shuffled):
        safe_chunk = safe_shuffled.iloc[start:end]
    else:
        safe_chunk = pd.concat([
            safe_shuffled.iloc[start:], 
            safe_shuffled.iloc[:end - len(safe_shuffled)]
        ]).reset_index(drop=True)

    train_chunk_df = pd.concat([toxic_df, safe_chunk]).sample(
        frac=1.0, random_state=42 + rot
    ).reset_index(drop=True)

    train_dataset_rot = PlantDataset(train_chunk_df, transform=strong_transforms)
    train_loader_rot = DataLoader(
        train_dataset_rot, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    # one rotation
    model.train()
    train_loss = 0.0
    train_total = 0

    for images, labels in train_loader_rot:
        images = images.to(device)
        labels = labels.float().unsqueeze(1).to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * images.size(0)
        train_total += images.size(0)

    avg_train_loss = train_loss / train_total

    # validate
    val_labels, val_probs, _ = evaluate_loader_timed(
        model, val_loader, device, use_amp=True, print_progress_every=200
    )
    val_metrics = compute_metrics(val_labels, val_probs, threshold=0.5)

    print(f"Rotation {rot+1} Train Loss: {avg_train_loss:.4f}")
    print(f"Val Metrics: {val_metrics}")

    # save checpoints
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "rotation": rot,
        "val_metrics": val_metrics,
    }
    torch.save(ckpt, f"rotation_{rot+1}_checkpoint_v2.pth")

    # best by f1 score
    current_f1 = val_metrics["f1"]
    if current_f1 > best_val_f1:
        best_val_f1 = current_f1
        torch.save(ckpt, rot_model_path)
        print(f"  New BEST rotating model saved → {rot_model_path} (F1={current_f1:.4f})")

print("\finished rotating training.")
print(f"Best rotating-validation F1: {best_val_f1:.4f}")



print("\nevaluating final model on realistic test set and hope for an imrpovemnt")

test_labels, test_probs, avg_batch_s_test = evaluate_loader_timed(
    model, test_loader, device, use_amp=True, print_progress_every=500
)
test_metrics = compute_metrics(test_labels, test_probs, threshold=0.5)

estimated_full_time_h = (len(test_loader) * avg_batch_s_test) / 3600.0 if avg_batch_s_test > 0 else None

print(f"Avg batch seconds during full eval approx: {avg_batch_s_test:.4f}")
print(f"Estimated full-test hours: {estimated_full_time_h:.2f}")
print("Realistic test metrics:", test_metrics)

final_ckpt = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "final_test_metrics": test_metrics,
    "note": "Version 2 —continued from V1, trained with rotating safe strategy; tune threshold for different risk profiles.",
}
torch.save(final_ckpt, "final_v2_rich_checkpoint.pth")
print("saved final rich checkpoint: final_v2_rich_checkpoint.pth")


