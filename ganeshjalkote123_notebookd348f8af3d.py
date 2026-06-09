# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
from tqdm import tqdm
import numpy as np
import pandas as pd

KAGGLE_DATA_PATH = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025"
TRAIN_CSV = os.path.join(KAGGLE_DATA_PATH, "train.csv")
MAX_SAMPLES = 1000

def load_from_csv_lookup(kaggle_root=KAGGLE_DATA_PATH, train_csv=TRAIN_CSV, max_samples=MAX_SAMPLES):
    print(f"\nğŸ“Š Building file index under: {kaggle_root}")
    # 1) Build dict: basename -> fullpath (if duplicates, keep first; report duplicates)
    basename_to_path = {}
    dup_count = 0
    for root, _, files in os.walk(kaggle_root):
        for f in files:
            if f.startswith('.'):
                continue
            if f.lower().endswith('.npy') or f.lower().endswith('.mat'):
                b = f.strip()
                full = os.path.join(root, f)
                if b in basename_to_path:
                    dup_count += 1
                else:
                    basename_to_path[b] = full
    print(f"  Indexed {len(basename_to_path)} hyperspectral files (duplicates={dup_count})")

    # 2) Read train.csv
    if not os.path.exists(train_csv):
        raise FileNotFoundError(f"train.csv not found at {train_csv}")
    df = pd.read_csv(train_csv)
    if 'id' not in df.columns and 'filename' in df.columns:
        df.rename(columns={'filename': 'id'}, inplace=True)
    if 'id' not in df.columns:
        raise ValueError("train.csv does not contain an 'id' column")

    # 3) Normalize ids: ensure they have .npy extension if needed
    ids = df['id'].astype(str).str.strip().tolist()
    # if IDs lack extension, try to append .npy
    normalized = []
    for sid in ids:
        if sid in basename_to_path:
            normalized.append(sid)
        elif sid + '.npy' in basename_to_path:
            normalized.append(sid + '.npy')
        elif sid.lower() in basename_to_path:
            normalized.append(sid.lower())
        elif sid.lower() + '.npy' in basename_to_path:
            normalized.append(sid.lower() + '.npy')
        else:
            normalized.append(None)  # mark missing

    # 4) Load files that exist (respect MAX_SAMPLES)
    data = []
    labels = []
    missing = []
    for orig_id, mapped in zip(ids, normalized):
        if mapped is None:
            missing.append(orig_id)
            continue
        path = basename_to_path.get(mapped)
        if path is None:
            missing.append(orig_id)
            continue
        try:
            arr = np.load(path)
            # sanity check: must be 3D
            if arr is None or getattr(arr, 'ndim', None) != 3:
                print(f"  âš ï¸� skipping {path} â€” not 3D (shape={getattr(arr, 'shape', None)})")
                continue
            data.append(arr.astype(np.float32))
            # if train.csv has label column use it, else set dummy
            if 'label' in df.columns:
                # find label for this row
                row_label = df.loc[df['id'] == orig_id, 'label'].values
                if len(row_label) > 0:
                    labels.append(row_label[0])
                else:
                    # fallback: check same id with extension
                    lbl = df.loc[df['id'] == orig_id + '.npy', 'label'].values
                    labels.append(lbl[0] if len(lbl) else -1)
            else:
                labels.append(-1)
        except Exception as e:
            print(f"  âš ï¸� Error loading {path}: {e}")
        if len(data) >= max_samples:
            break

    print(f"\nâœ“ Successfully loaded {len(data)} samples (requested max {max_samples})")
    if len(missing) > 0:
        print(f"â�— {len(missing)} ids from CSV were not found in file index (showing up to 10):")
        print(missing[:10])
    return data, labels, basename_to_path

# Example usage
data, labels, index = load_from_csv_lookup()



# ---------- PCA REDUCTION CELL (must run before training cell) ----------
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

TARGET_H, TARGET_W = 32, 32
PCA_BANDS = 30

def resize_image(img, th, tw):
    h, w, b = img.shape
    ys = np.linspace(0, h-1, th).astype(int)
    xs = np.linspace(0, w-1, tw).astype(int)
    return img[np.ix_(ys, xs)]

processed = []
processed_labels = []

for img, lab in zip(data, labels):
    if img is None or img.ndim != 3:
        continue
    small = resize_image(img, TARGET_H, TARGET_W).astype(np.float32)
    processed.append(small)
    processed_labels.append(lab)

arr = np.stack(processed)  # (N,H,W,B)
N, H, W, B = arr.shape

flat = arr.reshape(-1, B)     # (N*H*W, bands)
scaler = StandardScaler()
flat_scaled = scaler.fit_transform(flat)

pca = PCA(n_components=PCA_BANDS)
flat_pca = pca.fit_transform(flat_scaled)

data_pca_list = flat_pca.reshape(N, H, W, PCA_BANDS).tolist()

print("PCA done. Reduced image shape:", np.array(data_pca_list[0]).shape)





import os
import numpy as np
import matplotlib.pyplot as plt

# Paths (adjust if needed)
DATA_ROOT = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025"
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

# 1) Load train.csv and build index of .npy files
df = pd.read_csv(TRAIN_CSV_PATH)
ids = df['id'].astype(str).str.strip().tolist()
labels = df['label'].tolist()

# Build file index (same idea as your script)
file_index = {}
for r, _, files in os.walk(DATA_ROOT):
    for f in files:
        if f.lower().endswith(".npy"):
            file_index[f] = os.path.join(r, f)

print("Found", len(file_index), "npy files")

# 2) Pick first 1â€“2 valid images
loaded_imgs = []
loaded_labels = []
for fid, lab in zip(ids, labels):
    key = fid if fid.endswith(".npy") else fid + ".npy"
    if key in file_index:
        arr = np.load(file_index[key])
        if arr is not None and arr.ndim == 3:
            loaded_imgs.append(arr)
            loaded_labels.append(lab)
            if len(loaded_imgs) >= 6:   # change to 1 if you want only one image
                break

print(f"Loaded {len(loaded_imgs)} hyperspectral images")

# 3) Helper to convert hyperspectral cube to RGB-like image
def hs_to_rgb(img, bands=(10, 30, 50)):
    # img: (H, W, B)
    r = img[:, :, bands[0]]
    g = img[:, :, bands[1]]
    b = img[:, :, bands[2]]

    rgb = np.stack([r, g, b], axis=-1).astype(np.float32)

    # robust normalization using percentiles
    p1 = np.percentile(rgb, 1, axis=(0,1), keepdims=True)
    p99 = np.percentile(rgb, 99, axis=(0,1), keepdims=True)
    rgb = np.clip((rgb - p1) / (p99 - p1 + 1e-8), 0, 1)

    return rgb
    
# 4) Show the images
plt.figure(figsize=(10, 5))
for i, (img, lab) in enumerate(zip(loaded_imgs, loaded_labels)):
    rgb = hs_to_rgb(img, bands=(5, 20, 40))  # you can try other band combos too
    plt.subplot(1, len(loaded_imgs), i+1)
    plt.imshow(rgb)
    plt.axis('off')
    plt.title(f"Label: {lab}")
plt.tight_layout()
plt.show()



# ---------------- FULL END-TO-END TRAINING SCRIPT (single cell) ----------------
# Copy-paste this entire cell into your Kaggle notebook and run.
import os, random, time
from collections import Counter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# -------------------- CONFIG --------------------
KAGGLE_DATA_PATH = "/kaggle/input/beyond-visible-spectrum-ai-for-agriculture-2025"
TRAIN_CSV_PATH = os.path.join(KAGGLE_DATA_PATH, "train.csv")

SUBSET = 750                # 500-700 recommended for memory
TARGET_H, TARGET_W = 32, 32  # small spatial dims -> reduces RAM
PCA_BANDS = 40               # spectral dims after PCA
PATCH_SIZE = 24
PATCHES_PER_IMAGE = 16
BATCH_SIZE = 64
EPOCHS = 60
LR = 1e-3
WEIGHT_DECAY = 1e-5
PATIENCE = 8                 # early stopping patience
OUTPUT_PATH = "/kaggle/working"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()
NUM_WORKERS = 0              # keep 0 on Kaggle to reduce memory overhead

print("Device:", DEVICE, "AMP:", USE_AMP)
print(f"SUBSET={SUBSET}, TARGET={TARGET_H}x{TARGET_W}, PCA_BANDS={PCA_BANDS}, PATCH_SIZE={PATCH_SIZE}, PATCHES_PER_IMAGE={PATCHES_PER_IMAGE}")

# -------------------- Optional: if you already loaded images set this True and provide existing_data/existing_labels --------------------
LOAD_FROM_EXISTING = False
existing_data = None   # set to list of numpy arrays if LOAD_FROM_EXISTING=True
existing_labels = None

# -------------------- UTIL: index files from dataset --------------------
def build_index(root):
    idx = {}
    for r, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".npy"):
                idx[f] = os.path.join(r, f)
    return idx

# -------------------- UTIL: simple resize (nearest sampling) --------------------
def resize_image(img, th, tw):
    h, w, b = img.shape
    ys = np.linspace(0, h-1, th).astype(int)
    xs = np.linspace(0, w-1, tw).astype(int)
    return img[np.ix_(ys, xs)]

# -------------------- 1) LOAD SUBSET via train.csv --------------------
if LOAD_FROM_EXISTING and existing_data is not None and existing_labels is not None:
    print("Loading from existing in-memory data.")
    data = existing_data[:SUBSET]
    labels = existing_labels[:SUBSET]
else:
    print("Building file index under:", KAGGLE_DATA_PATH)
    index = build_index(KAGGLE_DATA_PATH)
    print("Indexed", len(index), ".npy files")

    if not os.path.exists(TRAIN_CSV_PATH):
        raise FileNotFoundError("train.csv not found at: " + TRAIN_CSV_PATH)
    df = pd.read_csv(TRAIN_CSV_PATH)
    if 'id' not in df.columns:
        raise ValueError("train.csv must contain an 'id' column")

    ids = df['id'].astype(str).str.strip().tolist()
    labels_csv = df['label'].tolist() if 'label' in df.columns else [None]*len(ids)

    # sample subset indices randomly
    all_indices = list(range(len(ids)))
    random.seed(42)
    random.shuffle(all_indices)
    chosen = []
    data = []
    labels = []
    for i in all_indices:
        fid = ids[i]
        key = fid if fid.endswith(".npy") else fid + ".npy"
        if key in index:
            try:
                arr = np.load(index[key])
                if arr is None or arr.ndim != 3:
                    continue
                data.append(arr.astype(np.float32))
                labels.append(labels_csv[i])
                chosen.append(key)
            except Exception as e:
                # skip bad files
                print("Skipping", key, "error:", e)
        if len(data) >= SUBSET:
            break
    print(f"Loaded {len(data)} samples from CSV (requested {SUBSET}). Missing or corrupted files may be skipped.")

if len(data) == 0:
    raise RuntimeError("No data loaded. Check dataset path and train.csv mapping.")

# -------------------- 2) Resize images to target small dims --------------------
processed = []
processed_labels = []
for img, lab in zip(data, labels):
    try:
        small = resize_image(img, TARGET_H, TARGET_W).astype(np.float32)
        processed.append(small)
        processed_labels.append(lab)
    except Exception as e:
        print("Skipping a sample during resize:", e)
print("Resized images count:", len(processed), "shape example:", processed[0].shape)

# -------------------- 3) PCA (flatten across pixels) --------------------
print("Running StandardScaler + PCA on pixels (this may take a moment)...")
arr = np.stack(processed, axis=0)   # (N,H,W,B)
N, H, W, B = arr.shape
flat = arr.reshape(-1, B)

scaler = StandardScaler()
flat_scaled = scaler.fit_transform(flat)

pca = PCA(n_components=PCA_BANDS, svd_solver='randomized', random_state=42)
flat_reduced = pca.fit_transform(flat_scaled)
data_pca = flat_reduced.reshape(N, H, W, PCA_BANDS).astype(np.float32)
print("PCA done. Reduced shape:", data_pca.shape)
# convert to list for compatibility with downstream code
data_pca_list = [data_pca[i] for i in range(data_pca.shape[0])]
labels_list = processed_labels

# -------------------- 4) Sample random patches per image --------------------
def sample_random_patches(image, n_patches, patch_size):
    H, W, C = image.shape
    patches = []
    if H < patch_size or W < patch_size:
        return patches
    for _ in range(n_patches):
        top = random.randint(0, H - patch_size)
        left = random.randint(0, W - patch_size)
        p = image[top:top+patch_size, left:left+patch_size, :].copy()
        patches.append(p)
    return patches

print("Sampling patches - this multiplies dataset size by PATCHES_PER_IMAGE...")
random.seed(42)
all_patches = []
all_patch_labels = []
for img, lab in zip(data_pca_list, labels_list):
    ps = sample_random_patches(img, PATCHES_PER_IMAGE, PATCH_SIZE)
    for p in ps:
        all_patches.append(np.transpose(p, (2,0,1)).astype(np.float32))  # (C,H,W)
        all_patch_labels.append(lab)

all_patches = np.array(all_patches)
print("Total patches shape:", all_patches.shape, "Total labels:", len(all_patch_labels))
if len(all_patch_labels) == 0:
    raise RuntimeError("No patches sampled - adjust PATCH_SIZE or check image sizes.")

# -------------------- 5) Label encode & split --------------------
le = LabelEncoder()
y_encoded = le.fit_transform(all_patch_labels)
num_classes = len(le.classes_)
print("Num classes:", num_classes, "class counts (sampled patches):", Counter(y_encoded))

can_stratify = (min(Counter(y_encoded).values()) >= 2) and (num_classes > 1)
X_train, X_val, y_train, y_val = train_test_split(
    all_patches, y_encoded, test_size=0.15, random_state=42,
    stratify=y_encoded if can_stratify else None
)
print("Train/Val sizes (patches):", len(X_train), len(X_val))

# -------------------- 6) Dataset, augmentations, sampler --------------------
AUG_PROB = 0.6
def augment_patch_np(x):
    # x shape: (C,H,W)
    # horizontal flip
    if random.random() < 0.5:
        x = x[:,:, ::-1].copy()
    # vertical flip
    if random.random() < 0.5:
        x = x[:, ::-1, :].copy()
    # band-wise gaussian noise
    x = x + np.random.normal(0, 0.01, size=x.shape).astype(np.float32)
    x = np.nan_to_num(x).astype(np.float32)
    return x

class AugPatchDataset(Dataset):
    def __init__(self, X, y, augment=False):
        self.X = X
        self.y = y
        self.augment = augment
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        x = self.X[idx].copy()
        label = int(self.y[idx])
        if self.augment and random.random() < AUG_PROB:
            x = augment_patch_np(x)
        return torch.from_numpy(x).float(), torch.tensor(label).long()

train_counts = Counter(y_train)
class_weights = {cls: 1.0 / count for cls, count in train_counts.items()}
sample_weights = np.array([class_weights[y] for y in y_train], dtype=np.float32)
sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

train_ds = AugPatchDataset(X_train, y_train, augment=True)
val_ds = AugPatchDataset(X_val, y_val, augment=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# -------------------- 7) Model --------------------
class BetterCNN(nn.Module):
    def __init__(self, in_ch, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_classes)
        )
    def forward(self, x):
        x = self.net(x)
        return self.classifier(x)

in_ch = all_patches.shape[1]
model = BetterCNN(in_ch=in_ch, n_classes=num_classes).to(DEVICE)
print("Model params:", sum(p.numel() for p in model.parameters()))

# -------------------- 8) Loss, optimizer, scheduler --------------------
weights_for_loss = np.array([1.0 / (train_counts[i] if i in train_counts else 1.0) for i in range(num_classes)], dtype=np.float32)
weights_for_loss = torch.tensor(weights_for_loss).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights_for_loss)
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.cuda.amp.GradScaler() if USE_AMP else None

# -------------------- 9) Training loop with early stopping --------------------
best_val = -1.0
best_epoch = -1
no_improve = 0
history = {'train_loss':[], 'train_acc':[], 'val_loss':[], 'val_acc':[]}
start_time = time.time()

for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0; running_correct = 0; running_total = 0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        optimizer.zero_grad()
        if USE_AMP:
            with torch.cuda.amp.autocast():
                out = model(xb); loss = criterion(out, yb)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
        else:
            out = model(xb); loss = criterion(out, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        running_loss += loss.item() * xb.size(0)
        preds = out.argmax(dim=1)
        running_correct += (preds == yb).sum().item()
        running_total += xb.size(0)
    scheduler.step()
    train_loss = running_loss / max(1, running_total)
    train_acc = 100.0 * running_correct / max(1, running_total)

    # validation
    model.eval()
    v_loss = 0.0; v_correct = 0; v_total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            if USE_AMP:
                with torch.cuda.amp.autocast():
                    out = model(xb); loss = criterion(out, yb)
            else:
                out = model(xb); loss = criterion(out, yb)
            v_loss += loss.item() * xb.size(0)
            preds = out.argmax(dim=1)
            v_correct += (preds == yb).sum().item()
            v_total += xb.size(0)
    val_loss = v_loss / max(1, v_total)
    val_acc = 100.0 * v_correct / max(1, v_total)

    history['train_loss'].append(train_loss); history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss); history['val_acc'].append(val_acc)

    print(f"[{epoch}/{EPOCHS}] Train loss {train_loss:.4f} acc {train_acc:.2f}% | Val loss {val_loss:.4f} acc {val_acc:.2f}%")

    # early stopping & save best
    if val_acc > best_val + 1e-6:
        best_val = val_acc
        best_epoch = epoch
        no_improve = 0
        torch.save(model.state_dict(), os.path.join(OUTPUT_PATH, "best_model.pth"))
        print("  âœ“ New best model saved:", best_val)
    else:
        no_improve += 1
    if no_improve >= PATIENCE:
        print(f"No improvement for {PATIENCE} epochs â€” early stopping.")
        break

time_elapsed = time.time() - start_time
print("Training finished in {:.1f}s â€” best val acc {:.2f}% at epoch {}".format(time_elapsed, best_val, best_epoch))

# -------------------- 10) Save final pipeline --------------------
best_model_path = os.path.join(OUTPUT_PATH, "best_model.pth")
if os.path.exists(best_model_path):
    model.load_state_dict(torch.load(best_model_path, map_location=DEVICE))

pca_bands_val = None
try:
    if isinstance(data_pca_list, list) and len(data_pca_list) > 0:
        pca_bands_val = data_pca_list[0].shape[2]
except Exception:
    pca_bands_val = PCA_BANDS

torch.save({
    "model_state_dict": model.state_dict(),
    "label_encoder": le,
    "config": {
        "patch_size": PATCH_SIZE,
        "patches_per_image": PATCHES_PER_IMAGE,
        "pca_bands": pca_bands_val
    },
    "performance": {"best_val_acc": best_val, "history": history}
}, os.path.join(OUTPUT_PATH, "complete_model_final.pth"))
print("Saved complete_model_final.pth and best_model.pth to", OUTPUT_PATH)

# -------------------- 11) Final evaluation & plots --------------------
eval_loader = val_loader
all_preds, all_trues = [], []
model.eval()
with torch.no_grad():
    for xb, yb in eval_loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        out = model(xb)
        preds = out.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_trues.extend(yb.cpu().numpy())

all_preds = np.array(all_preds); all_trues = np.array(all_trues)
overall_acc = 100.0 * (all_preds == all_trues).mean()
print(f"\nFinal validation accuracy: {overall_acc:.2f}% (best during training: {best_val:.2f}%)")

cm = confusion_matrix(all_trues, all_preds, labels=range(num_classes))
per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-12)
print("\nWorst 10 classes by val acc:")
worst = np.argsort(per_class_acc)[:10]
for idx in worst:
    print(f"  idx {idx} label {le.classes_[idx]} acc {100*per_class_acc[idx]:.2f}% samples {int(cm.sum(axis=1)[idx])}")

plt.figure(figsize=(10,4))
plt.subplot(1,2,1); plt.plot(history['train_loss'], label='train'); plt.plot(history['val_loss'], label='val'); plt.title('Loss'); plt.legend()
plt.subplot(1,2,2); plt.plot(history['train_acc'], label='train'); plt.plot(history['val_acc'], label='val'); plt.title('Accuracy'); plt.legend()
plt.tight_layout(); plt.show()

# visualize a few val patches with predictions
plt.figure(figsize=(12,3))
sample_k = min(8, len(X_val))
with torch.no_grad():
    sample_batch = torch.from_numpy(X_val[:sample_k]).float().to(DEVICE)
    outs = model(sample_batch)
    preds = outs.argmax(1).cpu().numpy()
for i in range(sample_k):
    patch = X_val[i]
    rgb = np.transpose(patch[:3], (1,2,0))
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)
    plt.subplot(1, sample_k, i+1); plt.imshow(rgb); plt.axis('off'); plt.title(f"P{preds[i]}")
plt.show()

print("Done. Best val acc: {:.2f}%".format(best_val))
# -----------------------------------------------------------------------------------------



from sklearn.metrics import precision_score, recall_score, f1_score

# Calculate metrics
precision = precision_score(all_trues, all_preds, average='weighted')
recall = recall_score(all_trues, all_preds, average='weighted')
f1 = f1_score(all_trues, all_preds, average='weighted')

print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Accuracy: {overall_acc:.4f}")


