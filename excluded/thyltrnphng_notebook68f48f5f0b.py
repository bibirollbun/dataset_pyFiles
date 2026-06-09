# =========================================================
# CELL 1 - IMPORTS & SEED
# =========================================================
import os, gc, random
import numpy as np
import pandas as pd

import h5py
from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import timm
from tqdm import tqdm

from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import (
    roc_auc_score, roc_curve, auc,
    confusion_matrix, classification_report
)

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



# =========================================================
# CELL 2 - PATHS (KAGGLE ISIC 2024)
# =========================================================
BASE = "/kaggle/input/isic-2024-challenge"

TRAIN_META = f"{BASE}/train-metadata.csv"
TEST_META  = f"{BASE}/test-metadata.csv"
TRAIN_H5   = f"{BASE}/train-image.hdf5"
TEST_H5    = f"{BASE}/test-image.hdf5"

print("BASE files:", os.listdir(BASE))
print("TRAIN_META exists:", os.path.exists(TRAIN_META))
print("TRAIN_H5 exists:", os.path.exists(TRAIN_H5))



# =========================================================
# CELL 3 - LOAD METADATA
# =========================================================
df = pd.read_csv(TRAIN_META)

# Các cột quan trọng tối thiểu cho pipeline baseline:
need_cols = ["isic_id", "target", "patient_id"]
for c in need_cols:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

print(df.shape)
print(df["target"].value_counts())
df.head()



# =========================================================
# CELL 4 - (OPTION) SUBSET SAMPLE (GIỐNG Ý BẠN)

# =========================================================
USE_SUBSET = True

NEG_N = 19650
POS_N = 393

if USE_SUBSET:
    df_pos = df[df["target"] == 1].sample(n=min(POS_N, (df["target"]==1).sum()), random_state=42)
    df_neg = df[df["target"] == 0].sample(n=min(NEG_N, (df["target"]==0).sum()), random_state=42)
    df = pd.concat([df_pos, df_neg], axis=0).sample(frac=1, random_state=42).reset_index(drop=True)

print("After subset:", df["target"].value_counts())
print("Total:", len(df))



# =========================================================
# CELL 5 - SPLIT TRAIN / VAL / TEST (PATIENT-LEVEL)
# =========================================================
# 80% train, 10% val, 10% test (theo patient)
gss1 = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, temp_idx = next(gss1.split(df, df["target"], groups=df["patient_id"]))
df_train = df.iloc[train_idx].reset_index(drop=True)
df_temp  = df.iloc[temp_idx].reset_index(drop=True)

gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
val_idx, test_idx = next(gss2.split(df_temp, df_temp["target"], groups=df_temp["patient_id"]))
df_val  = df_temp.iloc[val_idx].reset_index(drop=True)
df_test = df_temp.iloc[test_idx].reset_index(drop=True)

print("Train:", len(df_train), df_train["target"].mean())
print("Val:  ", len(df_val),   df_val["target"].mean())
print("Test: ", len(df_test),  df_test["target"].mean())

print("Patient overlap check:",
      len(set(df_train.patient_id) & set(df_val.patient_id)),
      len(set(df_train.patient_id) & set(df_test.patient_id)),
      len(set(df_val.patient_id) & set(df_test.patient_id)))



# =========================================================
# CELL 6 - TRANSFORMS (TorchVision style - đơn giản, ổn định)
# =========================================================
import torchvision.transforms as T

IMG_SIZE = 224

train_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    # augment nhẹ (baseline):
    T.RandomHorizontalFlip(p=0.5),
    T.RandomVerticalFlip(p=0.2),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])

val_tfms = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
])



# =========================================================
# CELL 7 - DATASET: READ IMAGE FROM HDF5 (BYTES -> PIL)
# =========================================================
class ISICH5Dataset(Dataset):
    def __init__(self, df, h5_path, transform=None):
        self.df = df.reset_index(drop=True)
        self.h5_path = h5_path
        self.transform = transform
        self.h5 = None

    def _init_h5(self):
        if self.h5 is None:
            self.h5 = h5py.File(self.h5_path, "r")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        self._init_h5()
        row = self.df.iloc[idx]
        key = row["isic_id"]

        # HDF5 value là bytes (jpg/png)
        data = self.h5[key][()]
        img = Image.open(BytesIO(data)).convert("RGB")

        y = torch.tensor(row["target"], dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, y



# =========================================================
# CELL 8 - DATALOADERS
# =========================================================
BATCH_SIZE = 32
NUM_WORKERS = 2

train_ds = ISICH5Dataset(df_train, TRAIN_H5, transform=train_tfms)
val_ds   = ISICH5Dataset(df_val,   TRAIN_H5, transform=val_tfms)
test_ds  = ISICH5Dataset(df_test,  TRAIN_H5, transform=val_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

print("Loaders ready:", len(train_ds), len(val_ds), len(test_ds))



# =========================================================
# CELL 9 - QUICK VISUAL CHECK (1-3 ảnh) để chắc chắn đọc HDF5 OK
# =========================================================
x, y = next(iter(train_loader))
print("Batch x:", x.shape, "Batch y:", y[:10].tolist())

# show 3 images (unnormalize)
def show_img(t):
    t = t.clone()
    mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
    t = t * std + mean
    t = t.permute(1,2,0).numpy()
    t = np.clip(t, 0, 1)
    return t

plt.figure(figsize=(9,3))
for i in range(3):
    plt.subplot(1,3,i+1)
    plt.imshow(show_img(x[i].cpu()))
    plt.title(f"y={int(y[i].item())}")
    plt.axis("off")
plt.suptitle("Sample images loaded from HDF5")
plt.show()



# =========================================================
# CELL 10 - MODEL + LOSS (IMBALANCE) + OPTIMIZER + AMP
# =========================================================
MODEL_NAME = "tf_efficientnet_b0"
model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=1).to(device)

neg = int((df_train["target"] == 0).sum())
pos = int((df_train["target"] == 1).sum())
pos_weight = torch.tensor([neg / max(pos, 1)], device=device)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)

# AMP API mới: torch.amp
use_amp = (device.type == "cuda")
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

print("Train neg/pos:", neg, pos, "| pos_weight:", float(pos_weight.item()))



# =========================================================
# CELL 11 - TRAIN / PREDICT / EVALUATE
# =========================================================
EPOCHS = 30  # tăng lên 10-20 nếu bạn muốn tốt hơn

def train_one_epoch():
    model.train()
    total_loss = 0.0

    for x, y in tqdm(train_loader, desc="Train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)  # [B,1]

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * x.size(0)

    return total_loss / len(train_loader.dataset)

@torch.no_grad()
def predict_probs(loader):
    model.eval()
    probs_list, y_list = [], []
    for x, y in tqdm(loader, desc="Predict", leave=False):
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        probs_list.append(probs)
        y_list.append(y.numpy().reshape(-1))
    return np.concatenate(probs_list), np.concatenate(y_list)

def evaluate(loader, name="VAL", threshold=0.5):
    probs, ys = predict_probs(loader)

    auc_score = roc_auc_score(ys, probs) if len(np.unique(ys)) == 2 else np.nan
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(ys, pred, labels=[0,1]).ravel()

    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)

    print(f"\n=== {name} EVAL ===")
    print(f"AUC: {auc_score:.4f}")
    print(f"Precision@{threshold}: {precision:.4f}")
    print(f"Recall@{threshold}: {recall:.4f}")
    print("Confusion (tn, fp, fn, tp):", (tn, fp, fn, tp))
    print(classification_report(ys, pred, digits=4))

    return auc_score, recall, precision, (tn, fp, fn, tp), probs, ys



# =========================================================
# CELL 12 - RUN TRAINING (SAVE BEST BY VAL AUC)
# =========================================================
best_val_auc = -1
best_path = "/kaggle/working/best_effnet_b0.pth"
train_losses = []

for epoch in range(1, EPOCHS+1):
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    loss = train_one_epoch()
    train_losses.append(loss)
    print(f"\nEpoch {epoch}/{EPOCHS} | Train Loss: {loss:.4f}")

    val_auc, val_recall, val_prec, _, _, _ = evaluate(val_loader, name="VAL", threshold=0.5)

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        torch.save(model.state_dict(), best_path)
        print(f"✅ Saved best model: {best_path}")

print("\nBest VAL AUC:", best_val_auc)



# CELL 13 - PLOT TRAIN LOSS (như loss curve)
# =========================================================
plt.figure(figsize=(6,4))
plt.plot(range(1, len(train_losses)+1), train_losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Train Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.show()



# =========================================================
# CELL 14 - LOAD BEST & BASELINE TEST @0.5
# =========================================================
model.load_state_dict(torch.load(best_path, map_location=device))

test_auc05, test_recall05, test_prec05, test_cm05, test_probs, test_y = evaluate(
    test_loader, name="TEST", threshold=0.5
)



# =========================================================
# CELL 15 - THRESHOLD TUNING (TĂNG RECALL) + PLOTS
# =========================================================
# lấy prob + y_true từ VAL
val_auc05, val_recall05, val_prec05, val_cm05, val_probs, val_y = evaluate(
    val_loader, name="VAL (for threshold tuning)", threshold=0.5
)

def metrics_at_threshold(y_true, probs, thr):
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
    recall = tp / (tp + fn + 1e-9)
    precision = tp / (tp + fp + 1e-9)
    beta = 2
    f2 = (1+beta**2) * precision * recall / (beta**2 * precision + recall + 1e-9)
    return {"thr":thr,"precision":precision,"recall":recall,"f2":f2,"tn":tn,"fp":fp,"fn":fn,"tp":tp}

thresholds = np.linspace(0.01, 0.99, 99)
rows = [metrics_at_threshold(val_y, val_probs, thr) for thr in thresholds]

# chọn theo F2 (ưu tiên recall)
best_by_f2 = max(rows, key=lambda r: r["f2"])
best_thr = best_by_f2["thr"]
print("Best threshold by F2 on VAL:", best_by_f2)

# evaluate test @ best_thr
test_metrics = metrics_at_threshold(test_y, test_probs, best_thr)
print("\n=== TEST @ best_thr ===")
print("threshold:", best_thr)
print("precision:", round(test_metrics["precision"],4))
print("recall:", round(test_metrics["recall"],4))
print("f2:", round(test_metrics["f2"],4))
print("Confusion (tn, fp, fn, tp):", (test_metrics["tn"], test_metrics["fp"], test_metrics["fn"], test_metrics["tp"]))

# plot threshold curves
prec_list = [r["precision"] for r in rows]
rec_list  = [r["recall"] for r in rows]
f2_list   = [r["f2"] for r in rows]

plt.figure(figsize=(8,5))
plt.plot(thresholds, rec_list, label="Recall (VAL)")
plt.plot(thresholds, prec_list, label="Precision (VAL)")
plt.plot(thresholds, f2_list, label="F2 (VAL)")
plt.axvline(best_thr, linestyle="--", label=f"Best thr = {best_thr:.2f}")
plt.xlabel("Threshold")
plt.ylabel("Score")
plt.title("Threshold tuning on Validation set")
plt.legend()
plt.grid(True)
plt.show()

# PR trade-off curve (by threshold)
plt.figure(figsize=(6,6))
plt.plot(rec_list, prec_list)
plt.xlabel("Recall (VAL)")
plt.ylabel("Precision (VAL)")
plt.title("Precision–Recall trade-off (by threshold)")
plt.grid(True)
plt.show()

# confusion matrix plot helper
def plot_cm(cm, title):
    plt.figure(figsize=(4,4))
    plt.imshow(cm)
    plt.title(title)
    plt.xticks([0,1], ["Pred 0","Pred 1"])
    plt.yticks([0,1], ["True 0","True 1"])
    for (i,j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

cm05 = confusion_matrix(test_y, (test_probs >= 0.5).astype(int), labels=[0,1])
plot_cm(cm05, "TEST Confusion Matrix @ thr=0.50")

cm_opt = confusion_matrix(test_y, (test_probs >= best_thr).astype(int), labels=[0,1])
plot_cm(cm_opt, f"TEST Confusion Matrix @ thr={best_thr:.2f}")

tn05, fp05, fn05, tp05 = cm05.ravel()
tn, fp, fn, tp = cm_opt.ravel()
print("FN giảm:", fn05, "→", fn, "(giảm", fn05 - fn, ")")
print("FP tăng:", fp05, "→", fp, "(tăng", fp - fp05, ")")




# =========================================================
# CELL 16 - ROC CURVE (VAL & TEST)
# =========================================================
from sklearn.metrics import roc_curve, auc

# ROC VAL
fpr_v, tpr_v, thr_v = roc_curve(val_y, val_probs)
auc_v = auc(fpr_v, tpr_v)

# ROC TEST
fpr_t, tpr_t, thr_t = roc_curve(test_y, test_probs)
auc_t = auc(fpr_t, tpr_t)

plt.figure(figsize=(6,6))
plt.plot(fpr_v, tpr_v, label=f"VAL ROC (AUC = {auc_v:.3f})")
plt.plot(fpr_t, tpr_t, label=f"TEST ROC (AUC = {auc_t:.3f})")
plt.plot([0,1], [0,1], linestyle="--", label="Random guess")
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR / Recall)")
plt.title("ROC Curve – EfficientNet-B0")
plt.legend()
plt.grid(True)
plt.show()

print(f"VAL AUC:  {auc_v:.4f}")
print(f"TEST AUC: {auc_t:.4f}")


