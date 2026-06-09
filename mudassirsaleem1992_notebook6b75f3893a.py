import math, random, os, time
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# CIFAR-10 normalization
C10_MEAN = (0.4914, 0.4822, 0.4465)
C10_STD  = (0.2470, 0.2435, 0.2616)

# Train transforms (augmentation)
train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(C10_MEAN, C10_STD),
])

# Validation/Test transforms (no augmentation)
eval_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(C10_MEAN, C10_STD),
])

root = "./data"
full_train = datasets.CIFAR10(root=root, train=True, download=True, transform=train_tf)

VAL_SIZE = 5000
TRAIN_SIZE = len(full_train) - VAL_SIZE
train_set, val_set = random_split(
    full_train, [TRAIN_SIZE, VAL_SIZE], generator=torch.Generator().manual_seed(SEED)
)
# IMPORTANT: use clean eval transforms for validation
val_set.dataset.transform = eval_tf

BATCH_SIZE = 128
NUM_WORKERS = 2

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
)
val_loader   = DataLoader(
    val_set,   batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True, drop_last=False
)

idx_to_class = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
print("Train/Val sizes:", len(train_set), len(val_set))


def apply_cutout(x, size=8):
    # x: [B,3,32,32]
    if size <= 0: return x
    B, C, H, W = x.size()
    y = x.clone()
    cx = torch.randint(W, (B,), device=x.device)
    cy = torch.randint(H, (B,), device=x.device)
    x1 = (cx - size//2).clamp(0, W)
    x2 = (cx + size//2).clamp(0, W)
    y1 = (cy - size//2).clamp(0, H)
    y2 = (cy + size//2).clamp(0, H)
    for i in range(B):
        y[i, :, y1[i]:y2[i], x1[i]:x2[i]] = 0.
    return y

def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def mixup_data(x, y, alpha=0.2):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0), device=x.device)
    y_a, y_b = y, y[index]
    B, C, H, W = x.size()
    x1, y1, x2, y2 = rand_bbox(W, H, lam)
    new_x = x.clone()
    new_x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam_adj = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return new_x, y_a, y_b, lam_adj


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)

class WRNBasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, drop_rate):
        super().__init__()
        self.equal = (in_planes == out_planes)
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(in_planes, out_planes, stride)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes, 1)
        self.dropout = nn.Dropout(p=drop_rate) if drop_rate > 0 else nn.Identity()
        self.shortcut = nn.Identity() if self.equal else nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.relu1(self.bn1(x))
        res = x if self.equal else self.shortcut(out)
        out = self.conv1(out)
        out = self.relu2(self.bn2(out))
        out = self.dropout(out)
        out = self.conv2(out)
        return out + res

def make_group(block, in_ch, out_ch, n, stride, drop_rate):
    layers = [block(in_ch, out_ch, stride, drop_rate)]
    for _ in range(1, n):
        layers.append(block(out_ch, out_ch, 1, drop_rate))
    return nn.Sequential(*layers)

class WideResNet(nn.Module):
    def __init__(self, depth=28, widen=10, drop_rate=0.3, num_classes=10):
        super().__init__()
        assert (depth - 4) % 6 == 0, "depth should be 6n+4"
        n = (depth - 4) // 6
        k = widen
        self.stem   = conv3x3(3, 16, 1)
        self.group1 = make_group(WRNBasicBlock, 16,    16*k, n, 1, drop_rate)
        self.group2 = make_group(WRNBasicBlock, 16*k,  32*k, n, 2, drop_rate)
        self.group3 = make_group(WRNBasicBlock, 32*k,  64*k, n, 2, drop_rate)
        self.bn   = nn.BatchNorm2d(64*k)
        self.relu = nn.ReLU(inplace=True)
        self.fc   = nn.Linear(64*k, num_classes)

        # Kaiming init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.stem(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = self.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)

model = WideResNet(depth=28, widen=10, drop_rate=0.3, num_classes=10).to(device)
n_params = sum(p.numel() for p in model.parameters())
print(f"Model params: {n_params/1e6:.2f}M")


EPOCHS = 75
BASE_LR = 3e-4
WEIGHT_DECAY = 5e-2
WARMUP_EPOCHS = 10

optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

def lr_schedule(t):
    # Cosine after warmup
    if t < WARMUP_EPOCHS:
        return (t+1) / WARMUP_EPOCHS
    T = EPOCHS - WARMUP_EPOCHS
    tt = (t - WARMUP_EPOCHS + 1) / T
    return 0.5 * (1 + math.cos(math.pi * tt))

def set_lr(optimizer, base, scale):
    for g in optimizer.param_groups:
        g['lr'] = base * scale

# EMA for stability & a small boost
class EMA:
    def __init__(self, m, decay=0.999):
        self.m = m
        self.shadow = {k: v.detach().clone() for k, v in m.state_dict().items()}
        self.decay = decay
    @torch.no_grad()
    def update(self):
        for k, v in self.m.state_dict().items():
            self.shadow[k] = self.decay * self.shadow[k] + (1 - self.decay) * v.detach()
    def state_dict(self):
        return self.shadow

ema = EMA(model, decay=0.999)

LABEL_SMOOTH = 0.1
ce = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
USE_CUTOUT = True
CUTOUT_SIZE = 8

best_acc = 0.0

def accuracy(loader, net):
    net.eval()
    tot = cor = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = net(x)
            pred = logits.argmax(1)
            cor += (pred == y).sum().item()
            tot += y.size(0)
    return cor / tot


# Training loop 
for epoch in range(EPOCHS):
    model.train()
    set_lr(optimizer, BASE_LR, lr_schedule(epoch))

    t0 = time.time()
    tr_cor = tr_tot = 0
    tr_loss_sum = 0.

    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        if USE_CUTOUT: x = apply_cutout(x, CUTOUT_SIZE)

        # Randomly choose MixUp or CutMix (50/50)
        if np.random.rand() < 0.5:
            x_mix, y_a, y_b, lam = mixup_data(x, y, alpha=MIXUP_ALPHA)
        else:
            x_mix, y_a, y_b, lam = cutmix_data(x, y, alpha=CUTMIX_ALPHA)

        logits = model(x_mix)
        loss = lam * ce(logits, y_a) + (1-lam) * ce(logits, y_b)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        ema.update()

        tr_loss_sum += loss.item() * x.size(0)
        # quick progress metric (not exact for mixed labels)
        preds = logits.argmax(1)
        tr_cor += (preds == y).sum().item()
        tr_tot += y.size(0)

    # Evaluate with EMA weights
    with torch.no_grad():
        saved = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema.state_dict(), strict=True)
        val_acc = accuracy(val_loader, model)
        model.load_state_dict(saved, strict=True)

    tr_loss = tr_loss_sum / tr_tot
    tr_acc = tr_cor / tr_tot
    best_acc = max(best_acc, val_acc)
    print(f"Epoch {epoch+1:03d}/{EPOCHS} | lr {optimizer.param_groups[0]['lr']:.5f} | "
          f"train {tr_acc:.4f}/{tr_loss:.4f} | val {val_acc:.4f} | best {best_acc:.4f} | "
          f"{time.time()-t0:.1f}s")

print("Best val acc:", best_acc)


import torch, time

def _unwrap_state_dict(m):
    return m.state_dict()

best_acc = float(globals().get("best_acc", 0.0))

# Try to save EMA weights (if this cell is run right after training)
try:
    if 'ema' in globals() and ema is not None:
        ema_ckpt = {
            "ema": True,
            "model": {k: v.cpu() for k, v in ema.state_dict().items()},
            "best_acc": best_acc,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        torch.save(ema_ckpt, "best_from_session_ema.pth")
        print("✓ Saved EMA -> best_from_session_ema.pth")
    else:
        print("EMA not present; skipping EMA save.")
except Exception as e:
    print("EMA save skipped:", e)

# Always save plain model
plain_ckpt = {
    "ema": False,
    "model": {k: v.cpu() for k, v in _unwrap_state_dict(model).items()},
    "best_acc": best_acc,
    "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
}
torch.save(plain_ckpt, "best_from_session_model.pth")
print("✓ Saved plain model -> best_from_session_model.pth")


# Offline accuracy on CIFAR-10 test set
from torchvision import datasets, transforms

# Rebuild model exactly like training
net = WideResNet(depth=28, widen=10, drop_rate=0.3, num_classes=10).to(device)

# Prefer EMA checkpoint if available
use_ema = False
if os.path.exists("best_from_session_ema.pth"):
    ckpt = torch.load("best_from_session_ema.pth", map_location=device)
    state = ckpt.get("model", ckpt)
    net.load_state_dict(state, strict=True)
    use_ema = True
else:
    ckpt = torch.load("best_from_session_model.pth", map_location=device)
    state = ckpt.get("model", ckpt)
    net.load_state_dict(state, strict=True)

net.eval()

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(C10_MEAN, C10_STD),
])
test_set = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_tf)
test_loader = DataLoader(
    test_set, batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

correct = total = 0
with torch.no_grad():
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = net(x).argmax(1)
        correct += (pred == y).sum().item()
        total   += y.size(0)

offline_acc = correct / total
print(f"Official CIFAR-10 test accuracy (offline estimate) [{ 'EMA' if use_ema else 'Plain' }]: {offline_acc:.4f}")


# Robust extract + verify + flatten, then recount
!pip -q install py7zr

import os, glob, shutil, py7zr
from pathlib import Path

# 1) Locate the archive
cands = glob.glob("/kaggle/input/**/test.7z", recursive=True)
print("Candidates:", cands)
assert cands, "Couldn't find test.7z. Make sure the CIFAR-10 competition is attached as an input."
test7z = cands[0]
print("Using:", test7z, "| size (bytes):", os.path.getsize(test7z))

# 2) Clean output dir
OUTDIR = Path("test")
if OUTDIR.exists():
    shutil.rmtree(OUTDIR)
OUTDIR.mkdir(parents=True, exist_ok=True)

# 3) Peek inside the archive, then extract
with py7zr.SevenZipFile(test7z, mode="r") as z:
    names = z.getnames()
    print("First 10 archive entries:", names[:10])
    z.extractall(path=str(OUTDIR))

# 4) Count recursively (handles nested folders inside the archive)
pngs_recursive = glob.glob(str(OUTDIR / "**" / "*.png"), recursive=True)
print("PNG files found (recursive):", len(pngs_recursive))

# 5) If files are nested (e.g., test/test/1.png), flatten into ./test
moved = 0
for p in pngs_recursive:
    p = Path(p)
    dest = OUTDIR / p.name
    if p.resolve() != dest.resolve():
        if not dest.exists():
            shutil.move(str(p), str(dest))
            moved += 1
print("Moved from nested dirs:", moved)

# 6) Remove empty subfolders under ./test (optional)
for root, dirs, files in os.walk(OUTDIR, topdown=False):
    if Path(root) == OUTDIR:
        continue
    if not os.listdir(root):
        os.rmdir(root)

# 7) Final count should be ~300000
final_count = len(list(OUTDIR.glob("*.png")))
print("Final PNG count in ./test:", final_count)
assert final_count > 0, "No PNGs found after extraction/flatten."


import csv, glob
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# --- 1) Load EMA (if present) or plain checkpoint ---
# Try an EMA checkpoint first (common names), else fall back to plain
ckpt_path_ema_candidates = [
    "best_from_session_ema.pth",
    "best_wrn2810.pth",              # if your EMA was saved under this name previously
]
ckpt_path_plain_candidates = [
    "best_from_session_model.pth",
    "best_wrn2810.pth",
]

state_loaded = False
if 'ema' in globals() and isinstance(ema, dict):
    try:
        model.load_state_dict(ema, strict=False)
        state_loaded = True
        print("Loaded model weights from in-memory EMA dict.")
    except Exception as e:
        print("In-memory EMA load failed:", e)

if not state_loaded:
    # Try file EMA
    for p in ckpt_path_ema_candidates:
        if Path(p).exists():
            print("Loading EMA checkpoint:", p)
            ckpt = torch.load(p, map_location=device)
            # allow either {'model': ...} or full EMA dict
            state = ckpt.get('model', ckpt)
            model.load_state_dict(state, strict=False)
            state_loaded = True
            break

if not state_loaded:
    # Fallback: plain checkpoint
    for p in ckpt_path_plain_candidates:
        if Path(p).exists():
            print("Loading plain checkpoint:", p)
            ckpt = torch.load(p, map_location=device)
            state = ckpt.get('model', ckpt)
            model.load_state_dict(state, strict=True)
            state_loaded = True
            break

assert state_loaded, "No model weights found. Place an EMA or plain checkpoint in the working dir."

model.eval()

# --- 2) Dataset & loader ---
test_tf_final = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(C10_MEAN, C10_STD),
])

# Sorted by numeric filename so ids align (1.png, 2.png, ...)
test_files = sorted(glob.glob('test/*.png'), key=lambda p: int(Path(p).stem))

class TestFolder(Dataset):
    def __init__(self, files, tfm):
        self.files = files; self.tfm = tfm
    def __len__(self): 
        return len(self.files)
    def __getitem__(self, i):
        fp = self.files[i]
        img = Image.open(fp).convert('RGB')
        return self.tfm(img), int(Path(fp).stem)

# Use the same constants you used during training; define them if needed
# BATCH_SIZE = 256
# NUM_WORKERS = 2
test_loader_big = DataLoader(
    TestFolder(test_files, test_tf_final),
    batch_size=BATCH_SIZE, shuffle=False,
    num_workers=NUM_WORKERS, pin_memory=True
)

IDX_TO_CLASS = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

# --- 3) Batched inference and write CSV ---
rows = []
with torch.no_grad():
    for x, ids in test_loader_big:
        x = x.to(device, non_blocking=True)
        pred = model(x).argmax(1).cpu().tolist()
        for i, p in zip(ids.tolist(), pred):
            rows.append((i, IDX_TO_CLASS[p]))

rows.sort(key=lambda t: t[0])

with open('submission.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id','label'])
    w.writerows(rows)

print("Wrote submission.csv with", len(rows), "rows.")


import pandas as pd

df = pd.read_csv("submission.csv")
print(df.head())
print("Rows:", len(df))
print("Columns:", list(df.columns))

# Must be exactly 300000 rows, two columns id,label
assert list(df.columns) == ["id", "label"]
assert len(df) == 300000

# ids should be 1..300000 with no gaps
assert df["id"].is_monotonic_increasing
assert df["id"].iloc[0] == 1 and df["id"].iloc[-1] == 300000
assert df["id"].nunique() == 300000

# labels must be in the allowed set
allowed = {'airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck'}
assert set(df["label"]).issubset(allowed)

print("✅ submission.csv looks good.")

