# Sir's method: extract /kaggle/input/cifar-10/test.7z using libarchive
!apt -y install -qq libarchive-dev
!pip -q install libarchive

import os, glob, time
import libarchive.public

os.chdir("/kaggle/working")  # extract into /kaggle/working
src = "/kaggle/input/cifar-10/test.7z"

t0 = time.time()
cnt = 0
for _ in libarchive.public.file_pour(src):  # creates ./test with PNGs
    cnt += 1
    if cnt % 5000 == 0:
        print(f"[extract] entries processed: {cnt} | elapsed: {int(time.time()-t0)}s")

print("[done] entries processed:", cnt)
print("Top-level PNGs in ./test:", len(glob.glob('test/*.png')))
print("Peek:", sorted([p.split('/')[-1] for p in glob.glob('test/*.png')])[:3],
             "...",
             sorted([p.split('/')[-1] for p in glob.glob('test/*.png')])[-3:])



# Sir's method for train.7z
import os, glob, time
import libarchive.public

os.chdir("/kaggle/working")
src = "/kaggle/input/cifar-10/train.7z"

t0 = time.time()
cnt = 0
for _ in libarchive.public.file_pour(src):  # creates ./train with PNGs
    cnt += 1
    if cnt % 2000 == 0:
        print(f"[extract] entries processed: {cnt} | elapsed: {int(time.time()-t0)}s")

print("[done] entries processed:", cnt)
print("Top-level PNGs in ./train:", len(glob.glob('train/*.png')))
print("Peek:", sorted([p.split('/')[-1] for p in glob.glob('train/*.png')])[:3],
             "...",
             sorted([p.split('/')[-1] for p in glob.glob('train/*.png')])[-3:])



# ===== Step: build train/val loaders from extracted PNGs =====
from pathlib import Path
from PIL import Image
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_DIR = Path("/kaggle/working/train")
CSV_PATH  = Path("/kaggle/input/cifar-10/trainLabels.csv")

# 1) id -> file path (top-level PNGs)
id2path = {int(p.stem): p for p in TRAIN_DIR.glob("*.png")}
print("train png ids found:", len(id2path))

# 2) labels dataframe
df = pd.read_csv(CSV_PATH)  # columns: id,label
classes = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
lab2idx  = {c:i for i,c in enumerate(classes)}

# 3) stratified split 90/10
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
tr_idx, va_idx = next(sss.split(df['id'], df['label']))
df_tr, df_va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
print("train size:", len(df_tr), "| val size:", len(df_va))

# 4) transforms 
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2470, 0.2435, 0.2616)
tf_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
tf_val = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# 5) dataset wrapping the id->path map
class CIFARTrainPNG(Dataset):
    def __init__(self, frame, id2path, tf):
        self.f, self.id2path, self.tf = frame, id2path, tf
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        r = self.f.iloc[i]
        img = Image.open(self.id2path[int(r['id'])]).convert("RGB")
        return self.tf(img), lab2idx[r['label']]

BATCH_SIZE = 128
train_loader = DataLoader(CIFARTrainPNG(df_tr, id2path, tf_train),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(CIFARTrainPNG(df_va, id2path, tf_val),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

# quick sanity: fetch one batch
xb, yb = next(iter(train_loader))
print("Batch:", xb.shape, yb.shape, "| device:", DEVICE)
print("Label sample (first 10):", [classes[i] for i in yb[:10].tolist()])



# ==========================================================
# CIFAR-10 (Kaggle) — Train CNN with CutMix + SWA
# ==========================================================

import os, math, time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedShuffleSplit
import torchvision.transforms as T
from torch.optim.lr_scheduler import LambdaLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torch.amp import autocast, GradScaler

# -------------------- Setup --------------------
SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)

# ---------- Paths ----------
TRAIN_DIR = Path("/kaggle/working/train")
CSV_PATH  = Path("/kaggle/input/cifar-10/trainLabels.csv")

# ---------- Classes / normalization ----------
CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
lab2idx = {c:i for i,c in enumerate(CLASSES)}
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

# ---------- Build id -> path map ----------
id2path = {int(p.stem): p for p in TRAIN_DIR.glob("*.png")}
print("train png ids found:", len(id2path))
df = pd.read_csv(CSV_PATH)
assert set(df['id']).issubset(set(id2path.keys()))

# ---------- Split ----------
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
tr_idx, va_idx = next(sss.split(df['id'], df['label']))
df_tr, df_va = df.iloc[tr_idx].reset_index(drop=True), df.iloc[va_idx].reset_index(drop=True)
print("train size:", len(df_tr), "| val size:", len(df_va))

# ---------- Transforms ----------
tf_train = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])
tf_val = T.Compose([
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

# ---------- Dataset ----------
class CIFARTrainPNG(Dataset):
    def __init__(self, frame, id2path, tf):
        self.f, self.id2path, self.tf = frame, id2path, tf
    def __len__(self): return len(self.f)
    def __getitem__(self, i):
        r = self.f.iloc[i]
        img = Image.open(self.id2path[int(r['id'])]).convert("RGB")
        return self.tf(img), lab2idx[r['label']]

# ---------- Loaders ----------
BATCH = 128
train_loader = DataLoader(CIFARTrainPNG(df_tr, id2path, tf_train),
                          batch_size=BATCH, shuffle=True,  num_workers=2, pin_memory=True)
val_loader   = DataLoader(CIFARTrainPNG(df_va, id2path, tf_val),
                          batch_size=BATCH, shuffle=False, num_workers=2, pin_memory=True)

# ---------- Model ----------
class CustomCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1, bias=False), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1, bias=False), nn.BatchNorm2d(512), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.head(x)

model = CustomCNN().to(DEVICE)

# ---------- Optimizer, scheduler, loss ----------
base_lr = 1e-3
optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
steps_per_epoch = len(train_loader)
EPOCHS = 200
warmup_steps = 500
total_steps = EPOCHS * steps_per_epoch

def lr_lambda(step):
    if step < warmup_steps:
        return step / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * t))

scheduler = LambdaLR(optimizer, lr_lambda)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scaler = GradScaler(device="cuda" if DEVICE=="cuda" else "cpu")

# ---------- CutMix ----------
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = math.sqrt(1. - lam)
    cut_w, cut_h = int(W*cut_rat), int(H*cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = max(cx-cut_w//2,0), max(cy-cut_h//2,0)
    x2, y2 = min(cx+cut_w//2,W), min(cy+cut_h//2,H)
    return x1,y1,x2,y2

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0), device=x.device)
    ya, yb = y, y[idx]
    x1,y1,x2,y2 = rand_bbox(x.size(), lam)
    xm = x.clone()
    xm[:,:,x1:x2, y1:y2] = x[idx,:,x1:x2, y1:y2]
    lam = 1 - ((x2-x1)*(y2-y1) / (x.size(-1)*x.size(-2)))
    return xm, ya, yb, lam

# ---------- SWA ----------
swa_start = 180
swa_model = AveragedModel(model)

# ---------- Training ----------
best_acc = 0.0
for epoch in range(1, EPOCHS+1):
    model.train()
    t0 = time.time()
    run_loss, correct, total = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
            if np.random.rand() < 0.3:
                xm, ya, yb2, lam = cutmix_data(xb, yb, alpha=1.0)
                logits = model(xm)
                loss = lam * criterion(logits, ya) + (1 - lam) * criterion(logits, yb2)
            else:
                logits = model(xb)
                loss = criterion(logits, yb)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        run_loss += loss.item()

    model.eval()
    v_correct = v_total = v_loss_sum = 0.0
    with torch.no_grad(), autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
        for vx, vy in val_loader:
            vx, vy = vx.to(DEVICE), vy.to(DEVICE)
            out = model(vx)
            v_loss = criterion(out, vy)
            v_loss_sum += v_loss.item() * vy.size(0)
            v_correct += out.argmax(1).eq(vy).sum().item()
            v_total += vy.size(0)

    val_acc = v_correct / max(1, v_total)
    val_loss = v_loss_sum / max(1, v_total)
    if epoch >= swa_start: swa_model.update_parameters(model)
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch {epoch:03d}/{EPOCHS} | val loss {val_loss:.4f} acc {val_acc:.4f} | best {best_acc:.4f} | time {time.time()-t0:.1f}s")

if EPOCHS >= swa_start:
    update_bn(train_loader, swa_model, device=DEVICE)
    torch.save(swa_model.state_dict(), "best_swa.pth")

print("✅ Training complete. Model files saved.")



# ==========================================================
# Predict + Generate submission.csv
# ==========================================================

# ==========================================================
# Predict + Generate submission.csv
# ==========================================================

import os
import torch
import pandas as pd
from pathlib import Path
from collections import OrderedDict
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TEST_DIR  = Path("/kaggle/working/test")

CLASSES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
idx2lab = CLASSES
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

tf_test = T.Compose([
    T.ToTensor(),
    T.Normalize(MEAN, STD),
])

class TestDataset(Dataset):
    def __init__(self, files, tf):
        self.files, self.tf = files, tf
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        p = self.files[i]
        img = Image.open(p).convert("RGB")
        return self.tf(img), int(Path(p).stem)

class CustomCNN(torch.nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, padding=1, bias=False), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, padding=1, bias=False), torch.nn.BatchNorm2d(64), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(64, 128, 3, padding=1, bias=False), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, 3, padding=1, bias=False), torch.nn.BatchNorm2d(128), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(128, 256, 3, padding=1, bias=False), torch.nn.BatchNorm2d(256), torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, 3, padding=1, bias=False), torch.nn.BatchNorm2d(256), torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Conv2d(256, 512, 3, padding=1, bias=False), torch.nn.BatchNorm2d(512), torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, 3, padding=1, bias=False), torch.nn.BatchNorm2d(512), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(1),
        )
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(512, num_classes))
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.head(x)

ckpt_path = "best_swa.pth" if os.path.exists("best_swa.pth") else "best_model.pth"
print(f"Loading checkpoint: {ckpt_path}")

state = torch.load(ckpt_path, map_location=DEVICE)
model = CustomCNN().to(DEVICE)

if any(k.startswith("module.") for k in state.keys()):
    new_state = OrderedDict()
    for k, v in state.items():
        if k.startswith("module."):
            new_state[k[len("module."):]] = v
        elif k != "n_averaged":
            new_state[k] = v
    state = new_state

model.load_state_dict(state, strict=True)
model.eval()

test_files = sorted(list(TEST_DIR.glob("*.png")), key=lambda x: int(x.stem))
test_loader = DataLoader(TestDataset(test_files, tf_test),
                         batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

preds = []
with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=(DEVICE=="cuda")):
    for xb, ids in tqdm(test_loader, desc="Predicting"):
        xb = xb.to(DEVICE)
        out = model(xb)
        preds.extend(zip(ids.numpy(), out.argmax(1).cpu().numpy()))

sub = pd.DataFrame(preds, columns=["id", "label_idx"])
sub["label"] = sub["label_idx"].apply(lambda x: idx2lab[x])
sub = sub[["id", "label"]]
sub.to_csv("submission.csv", index=False)
print("✅ submission.csv saved successfully. Rows:", len(sub))
sub.head()


