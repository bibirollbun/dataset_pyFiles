# baseline_3d_cnn_rsna.py
# Full script: train simple 3D CNN on DICOM series and provide predict() for Kaggle RSNA gateway.
# Copy -> paste vào notebook Kaggle, tắt Internet, Save & Run All.

import os
import glob
import math
import random
from tqdm import tqdm
import numpy as np
import pandas as pd
import pydicom

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# -----------------------
# Config
# -----------------------
ROOT = "/kaggle/input/rsna-intracranial-aneurysm-detection"  # competition input
TRAIN_CSV = os.path.join(ROOT, "train.csv")
TRAIN_SERIES_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/train"  # if exists (adjust)
WORK_DIR = "/kaggle/working"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 8
EPOCHS = 8
LEARNING_RATE = 1e-4
NUM_SLICES = 32        # depth for volume (adjustable)
TARGET_HW = 128        # height/width to resize/crop to
SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# -----------------------
# Labels (as used before)
# -----------------------
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present'
]

# -----------------------
# Utilities: load dicom series -> 3D numpy volume
# -----------------------
def load_series_to_volume(series_dir):
    paths = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
    if len(paths) == 0:
        # sometimes files may have other extension; return zeros
        return None
    slices = []
    for p in paths:
        try:
            d = pydicom.dcmread(p)
            arr = d.pixel_array.astype(np.float32)
            # convert via rescale if present
            if hasattr(d, "RescaleIntercept") and hasattr(d, "RescaleSlope"):
                arr = arr * float(d.RescaleSlope) + float(d.RescaleIntercept)
            slices.append((int(getattr(d, "InstanceNumber", 0)), arr))
        except Exception:
            continue
    if len(slices) == 0:
        return None
    slices = sorted(slices, key=lambda x: x[0])
    volume = np.stack([s[1] for s in slices], axis=0)  # (D, H, W)
    return volume

def center_crop_or_pad(img, target_h, target_w):
    if img.ndim == 3:  # có channel
        h, w = img.shape[-2:]
    else:  # grayscale
        h, w = img.shape
    # pad if smaller
    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)
    if pad_h > 0 or pad_w > 0:
        top = pad_h//2
        bottom = pad_h - top
        left = pad_w//2
        right = pad_w - left
        img = np.pad(img, ((top,bottom),(left,right)), mode='constant', constant_values=0)
        h, w = img.shape
    # crop center if bigger
    start_h = max(0, (h - target_h)//2)
    start_w = max(0, (w - target_w)//2)
    return img[start_h:start_h+target_h, start_w:start_w+target_w]

def resample_slices(volume, target_num):
    D, H, W = volume.shape
    if D == target_num:
        return volume
    # simple linear interpolation along depth
    zs = np.linspace(0, D-1, target_num).astype(np.float32)
    out = []
    for z in zs:
        z0 = int(np.floor(z))
        z1 = min(D-1, z0+1)
        if z0 == z1:
            out.append(volume[z0])
        else:
            w1 = z - z0
            w0 = 1 - w1
            out.append(volume[z0]*w0 + volume[z1]*w1)
    return np.stack(out, axis=0)

def normalize_volume(vol):
    # clip extreme values then z-score per volume
    v = vol.astype(np.float32)
    v = np.clip(v, np.percentile(v,1), np.percentile(v,99))
    mean = v.mean()
    std = v.std() if v.std()>0 else 1.0
    v = (v - mean) / std
    return v

# -----------------------
# Dataset
# -----------------------
class RSNADataset(Dataset):
    def __init__(self, df, series_root, num_slices=NUM_SLICES, target_hw=TARGET_HW, mode='train'):
        self.df = df.reset_index(drop=True)
        self.series_root = series_root
        self.num_slices = num_slices
        self.target_hw = target_hw
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        uid = self.df.loc[idx, "SeriesInstanceUID"]
        series_dir = os.path.join(self.series_root, str(uid))
        vol = load_series_to_volume(series_dir)
        if vol is None:
            # fallback zeros
            vol = np.zeros((self.num_slices, self.target_hw, self.target_hw), dtype=np.float32)
        else:
            # resize each slice to target_hw x target_hw via center crop/pad
            slices = []
            for s in vol:
                s2 = center_crop_or_pad(s, self.target_hw, self.target_hw)
                slices.append(s2)
            vol = np.stack(slices, axis=0)  # (D,H,W)
            vol = resample_slices(vol, self.num_slices)
        vol = normalize_volume(vol)
        # channel-first for 3D conv: (C=1, D, H, W)
        x = torch.from_numpy(vol).unsqueeze(0).float()
        if self.mode == 'train' and "Aneurysm Present" in self.df.columns:
            labels = self.df.loc[idx, LABEL_COLS].values.astype(np.float32)
            y = torch.from_numpy(labels)
            return x, y
        else:
            return x, str(uid)

# -----------------------
# Simple 3D CNN model
# -----------------------
class Simple3DNet(nn.Module):
    def __init__(self, in_ch=1, num_outputs=14):
        super().__init__()
        def conv_block(in_ch, out_ch, ks=3, stride=1, pad=1):
            return nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=ks, stride=stride, padding=pad),
                nn.BatchNorm3d(out_ch),
                nn.ReLU(inplace=True),
                nn.MaxPool3d((1,2,2))  # pool spatial dims only
            )
        self.enc1 = conv_block(in_ch, 16)
        self.enc2 = conv_block(16, 32)
        self.enc3 = conv_block(32, 64)
        self.enc4 = conv_block(64, 128)
        # global pool
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(128, num_outputs)

    def forward(self, x):
        # x: (B,1,D,H,W)
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.global_pool(x)  # (B, C,1,1,1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # logits for BCEWithLogitsLoss

# -----------------------
# Prepare train/val
# -----------------------
train_df = pd.read_csv(TRAIN_CSV)
# if "Aneurysm Present" not present but 'any' exist, adapt
if "Aneurysm Present" not in train_df.columns and "any" in train_df.columns:
    train_df["Aneurysm Present"] = train_df["any"]
# Ensure label cols exist in correct order; if not present create zeros
for c in LABEL_COLS:
    if c not in train_df.columns:
        train_df[c] = 0

# split by SeriesInstanceUID
train_uids, val_uids = train_test_split(train_df["SeriesInstanceUID"].unique(), test_size=0.15, random_state=SEED)
train_df_sub = train_df[train_df["SeriesInstanceUID"].isin(train_uids)].reset_index(drop=True)
val_df_sub = train_df[train_df["SeriesInstanceUID"].isin(val_uids)].reset_index(drop=True)

# Determine series root: Kaggle dataset may place DICOM series under a folder named 'train' inside input
# Try to find folder with UID subfolders
possible_roots = [
    "/kaggle/input/rsna-intracranial-aneurysm-detection/train",
    "/kaggle/input/rsna-intracranial-aneurysm-detection/series", 
    "/kaggle/working/train"
]
SERIES_ROOT = None
for r in possible_roots:
    if os.path.exists(r):
        # check first few uids
        sample_uid = str(train_df_sub.loc[0, "SeriesInstanceUID"])
        if os.path.exists(os.path.join(r, sample_uid)):
            SERIES_ROOT = r
            break
# fallback to first possible existing root
if SERIES_ROOT is None:
    for r in possible_roots:
        if os.path.exists(r):
            SERIES_ROOT = r
            break
if SERIES_ROOT is None:
    SERIES_ROOT = "/kaggle/working"  # best-effort

print("Using SERIES_ROOT =", SERIES_ROOT)

train_dataset = RSNADataset(train_df_sub, SERIES_ROOT, mode='train')
val_dataset = RSNADataset(val_df_sub, SERIES_ROOT, mode='train')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# -----------------------
# Train
# -----------------------
model = Simple3DNet(in_ch=1, num_outputs=len(LABEL_COLS)).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5, verbose=True)

best_auc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    total_loss = 0.0
    for xb, yb in tqdm(train_loader, desc=f"Train E{epoch}"):
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        logits = model(xb)
        loss = criterion(logits, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    avg_loss = total_loss / len(train_loader.dataset)
    # validation
    model.eval()
    all_targets = []
    all_preds = []
    with torch.no_grad():
        for xb, yb in tqdm(val_loader, desc=f"Val E{epoch}"):
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = model(xb)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_preds.append(probs)
            all_targets.append(yb.cpu().numpy())
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    # compute AUC for "Aneurysm Present" (last column) and mean AUC across labels (if possible)
    aucs = []
    for i in range(all_targets.shape[1]):
        try:
            a = roc_auc_score(all_targets[:, i], all_preds[:, i])
        except Exception:
            a = float('nan')
        aucs.append(a)
    mean_auc = np.nanmean([a for a in aucs if not math.isnan(a)])
    aneurysm_auc = aucs[-1]
    print(f"Epoch {epoch} loss={avg_loss:.4f} mean_auc={mean_auc:.4f} aneurysm_auc={aneurysm_auc:.4f}")
    scheduler.step(mean_auc if not math.isnan(mean_auc) else avg_loss)
    # save best by aneurysm_auc
    if not math.isnan(aneurysm_auc) and aneurysm_auc > best_auc:
        best_auc = aneurysm_auc
        torch.save(model.state_dict(), os.path.join(WORK_DIR, "best_3dnet.pth"))
        print("Saved best model", best_auc)

# if no save happened, save last
if not os.path.exists(os.path.join(WORK_DIR, "best_3dnet.pth")):
    torch.save(model.state_dict(), os.path.join(WORK_DIR, "last_3dnet.pth"))

# -----------------------
# Inference predict() for Kaggle gateway
# The RSNA gateway will place a folder /kaggle/shared/<SeriesInstanceUID> with .dcm files.
# We'll read from that path, preprocess identical to dataset, run model, return DataFrame with LABEL_COLS (same order)
# -----------------------
model_file = os.path.join(WORK_DIR, "best_3dnet.pth")
if os.path.exists(model_file):
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
model.eval()

def predict(series_instance_uids):
    if isinstance(series_instance_uids, str):
        uids = [series_instance_uids]
    else:
        uids = list(series_instance_uids)
    results = []
    for uid in uids:
        series_dir = os.path.join("/kaggle/shared", str(uid))
        vol = load_series_to_volume(series_dir)
        if vol is None:
            vol = np.zeros((NUM_SLICES, TARGET_HW, TARGET_HW), dtype=np.float32)
        else:
            slices = [center_crop_or_pad(s, TARGET_HW, TARGET_HW) for s in vol]
            vol = np.stack(slices, axis=0)
            vol = resample_slices(vol, NUM_SLICES)
        vol = normalize_volume(vol)
        x = torch.from_numpy(vol).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # (1,1,D,H,W)
        with torch.no_grad():
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
        probs = np.clip(probs, 0.001, 0.999)
        row = {"SeriesInstanceUID": str(uid)}
        for i, col in enumerate(LABEL_COLS):
            row[col] = float(probs[i])
        results.append(row)
    return pd.DataFrame(results)

# -----------------------
# Hook into Kaggle inference server (same as your previous)
# -----------------------
try:
    import kaggle_evaluation
    inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway()
        try:
            display(pd.read_parquet('/kaggle/working/submission.parquet'))
        except:
            print("No submission file found")
except Exception as e:
    print("No kaggle_evaluation available or running locally. You can still call predict(uids). Error:", e)


