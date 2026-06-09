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


!apt install libarchive-dev
!pip install libarchive


import os, glob
from libarchive.public import file_reader

SRC = "/kaggle/input/cifar-10"
WORK = "/kaggle/working"
TRAIN_OUT = f"{WORK}/train"
TEST_OUT  = f"{WORK}/test"
os.makedirs(TRAIN_OUT, exist_ok=True)
os.makedirs(TEST_OUT, exist_ok=True)

def extract_7z(archive_path: str, out_dir: str, skip_if_present: bool = True):
    """Extracts a .7z archive using libarchive into out_dir, with progress every 1000 files."""
    # Skip if already extracted (common in Kaggle to avoid rework on reruns)
    if skip_if_present and any(p.name.endswith(".png") for p in os.scandir(out_dir)):
        print(f"✔ {out_dir} already contains files. Skipping: {os.path.basename(archive_path)}")
        return

    cnt = 0
    print(f"Extracting {archive_path} -> {out_dir}")
    with file_reader(archive_path) as entries:
        for entry in entries:
            # Some entries can be directories; handle both cases
            rel = entry.pathname
            # In these archives, files are flat (e.g., '1.png'), but we normalize anyway
            out_path = os.path.join(out_dir, os.path.basename(rel.rstrip("/")))
            if rel.endswith("/"):
                os.makedirs(out_path, exist_ok=True)
                continue
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            with open(out_path, "wb") as f:
                for block in entry.get_blocks():
                    f.write(block)
            cnt += 1
            if cnt % 1000 == 0:
                print(f"{cnt} files...")

    print(f"✔ Done: extracted {cnt} files to {out_dir}")

# Run extraction for both archives based on your listing
extract_7z(f"{SRC}/train.7z", TRAIN_OUT)
extract_7z(f"{SRC}/test.7z",  TEST_OUT)

# Quick sanity counts
train_count = len(glob.glob(f"{TRAIN_OUT}/*.png"))
test_count  = len(glob.glob(f"{TEST_OUT}/*.png"))
print(f"Counts -> train: {train_count} | test: {test_count}")


# ==== setup ====
import os, math, random, glob, time, gc, numpy as np, pandas as pd
from PIL import Image

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)

SRC = "/kaggle/input/cifar-10"
WORK = "/kaggle/working"
TRAIN_DIR = f"{WORK}/train"
TEST_DIR  = f"{WORK}/test"
assert os.path.exists(TRAIN_DIR) and os.path.exists(TEST_DIR), "Extract train/test to /kaggle/working/* first."

# Canonical CIFAR-10 stats
MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)



# ==== labels/split ====
from sklearn.model_selection import StratifiedShuffleSplit

train_df = pd.read_csv(f"{SRC}/trainLabels.csv")  # columns: id,label
train_df["file"] = train_df["id"].astype(str) + ".png"

CLASSES = sorted(train_df["label"].unique().tolist())
cls2idx = {c:i for i,c in enumerate(CLASSES)}
idx2cls = {i:c for c,i in cls2idx.items()}
num_classes = len(CLASSES)
print("Classes:", CLASSES)

sss = StratifiedShuffleSplit(n_splits=1, test_size=5000, random_state=SEED)
tr_idx, va_idx = next(sss.split(train_df["file"], train_df["label"]))
tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[va_idx].reset_index(drop=True)
len(tr_df), len(va_df)



# ==== dataset & loaders ====
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(),
    transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),  # strong & reliable
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.2), value=0)
])
valid_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

class CIFAR10Files(Dataset):
    def __init__(self, root_dir, df, transform):
        self.root = root_dir
        self.df = df.reset_index(drop=True)
        self.t = transform
        self.has_labels = "label" in df.columns
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(f"{self.root}/{r['file']}").convert("RGB")
        img = self.t(img) if self.t else img
        if self.has_labels: return img, cls2idx[r["label"]]
        return img, r["file"]

BATCH_SIZE = 256   # drop to 128 if you OOM
NUM_WORKERS = 2    # Kaggle default-safe

train_loader = DataLoader(CIFAR10Files(TRAIN_DIR, tr_df, train_tfms),
                          batch_size=BATCH_SIZE, shuffle=True, drop_last=True,
                          num_workers=NUM_WORKERS, pin_memory=True)
valid_loader = DataLoader(CIFAR10Files(TRAIN_DIR, va_df, valid_tfms),
                          batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
len(train_loader), len(valid_loader)



# ==== model ====
from torchvision.models import resnet34

def make_cifar_resnet34(num_classes=10):
    m = resnet34(weights=None)                  # <-- NO pretrained weights (assignment compliant)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 3x3, stride 1
    m.maxpool = nn.Identity()                   # no 7x7/stride-2/maxpool for 32x32
    m.fc = nn.Linear(512, num_classes)
    return m

model = make_cifar_resnet34(num_classes).to(DEVICE)
print("Params (M):", round(sum(p.numel() for p in model.parameters())/1e6, 2))



# ==== training setup ====
from contextlib import nullcontext

# MixUp/CutMix helpers
def rand_bbox(size, lam):
    W, H = size[2], size[3]
    cut_rat = math.sqrt(1. - lam)
    cut_w, cut_h = int(W*cut_rat), int(H*cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, y1 = np.clip(cx - cut_w//2, 0, W), np.clip(cy - cut_h//2, 0, H)
    x2, y2 = np.clip(cx + cut_w//2, 0, W), np.clip(cy + cut_h//2, 0, H)
    return x1, y1, x2, y2

def mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam*criterion(pred, y_a) + (1-lam)*criterion(pred, y_b)

# Float-only EMA (avoids dtype crashes)
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items()}
        self.backup = None
    def update(self, model):
        with torch.no_grad():
            for k,v in model.state_dict().items():
                if v.is_floating_point():
                    self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0-self.decay)
                else:
                    self.shadow[k] = v.detach().clone()
    def apply_to(self, model):
        self.backup = {k: v.detach().clone() for k,v in model.state_dict().items()}
        model.load_state_dict(self.shadow, strict=True)
    def restore(self, model):
        model.load_state_dict(self.backup, strict=True); self.backup=None

def accuracy(logits, y): return (logits.argmax(1) == y).float().mean().item()

# Training knobs
EPOCHS = 80
BASE_LR = 0.2
WEIGHT_DECAY = 5e-4
LABEL_SMOOTH = 0.1
MIXUP_ALPHA = 0.2
CUTMIX_ALPHA = 1.0
MIX_PROB = 0.7

optimizer = torch.optim.SGD(model.parameters(), lr=BASE_LR, momentum=0.9,
                            nesterov=True, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

# New AMP API (no deprecation warnings)
if DEVICE.type == "cuda":
    scaler = torch.amp.GradScaler("cuda")
    def autocast_ctx(): return torch.amp.autocast(device_type="cuda")
else:
    scaler = torch.amp.GradScaler(enabled=False)
    def autocast_ctx(): return nullcontext()

ema = EMA(model, decay=0.999)
best_acc = 0.0
BEST_PATH = f"{WORK}/best_resnet34_cifar.pth"
CKPT_DIR = f"{WORK}/ckpts"; os.makedirs(CKPT_DIR, exist_ok=True)



# ==== train ====
for epoch in range(1, EPOCHS+1):
    model.train()
    tr_loss, tr_correct, seen = 0.0, 0, 0

    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE, non_blocking=True), yb.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        use_mix = (np.random.rand() < MIX_PROB)
        if use_mix:
            r = np.random.rand()
            if r < 0.5 and MIXUP_ALPHA > 0:
                lam = np.random.beta(MIXUP_ALPHA, MIXUP_ALPHA)
                idx = torch.randperm(xb.size(0), device=xb.device)
                xb2 = lam*xb + (1-lam)*xb[idx]
                y_a, y_b = yb, yb[idx]
                with autocast_ctx():
                    logits = model(xb2); loss = mix_criterion(criterion, logits, y_a, y_b, lam)
            else:
                lam = np.random.beta(CUTMIX_ALPHA, CUTMIX_ALPHA)
                idx = torch.randperm(xb.size(0), device=xb.device)
                y_a, y_b = yb, yb[idx]
                x1,y1,x2,y2 = rand_bbox(xb.size(), lam)
                xbm = xb.clone(); xbm[:,:,y1:y2,x1:x2] = xb[idx,:,y1:y2,x1:x2]
                lam = 1 - ((x2-x1)*(y2-y1)/(xb.size(-1)*xb.size(-2)))
                with autocast_ctx():
                    logits = model(xbm); loss = mix_criterion(criterion, logits, y_a, y_b, lam)
        else:
            with autocast_ctx():
                logits = model(xb); loss = criterion(logits, yb)

        scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
        ema.update(model)

        tr_loss += loss.item()*xb.size(0)
        seen += xb.size(0)
        if not use_mix:
            tr_correct += (logits.argmax(1) == yb).float().sum().item()

    scheduler.step()
    tr_loss /= seen; tr_acc = tr_correct / max(seen,1)

    # --- validation (EMA weights) ---
    model.eval(); ema.apply_to(model)
    va_loss, va_correct, va_seen = 0.0, 0, 0
    with torch.no_grad():
        for xb, yb in valid_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            with autocast_ctx():
                logits = model(xb); loss = criterion(logits, yb)
            va_loss += loss.item()*xb.size(0)
            va_correct += (logits.argmax(1)==yb).float().sum().item()
            va_seen += xb.size(0)
    ema.restore(model)

    va_loss /= va_seen; va_acc = va_correct / va_seen

    # Save best + rolling ckpts to avoid losing progress
    if va_acc > best_acc:
        best_acc = va_acc
        torch.save({"model": model.state_dict(), "ema": ema.shadow, "acc": best_acc, "epoch": epoch}, BEST_PATH)
    if epoch % 5 == 0:
        torch.save({"model": model.state_dict(), "ema": ema.shadow, "acc": va_acc, "epoch": epoch},
                   f"{CKPT_DIR}/ep{epoch:03d}.pth")

    if epoch % 5 == 0 or epoch == 1:
        print(f"Epoch {epoch:02d}/{EPOCHS} | train {tr_loss:.4f} | valid {va_loss:.4f} acc {va_acc:.4f} | best {best_acc:.4f}")

print("Best val acc:", best_acc)



# ==== inference & submission ====
import os, pandas as pd, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet34
from torchvision import transforms
from PIL import Image

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WORK = "/kaggle/working"
SRC  = "/kaggle/input/cifar-10"
TEST_DIR = f"{WORK}/test"
BEST_PATH = f"{WORK}/best_resnet34_cifar.pth"

# Rebuild label map (needed for submission)
train_df = pd.read_csv(f"{SRC}/trainLabels.csv")
CLASSES  = sorted(train_df["label"].unique().tolist())
idx2cls  = {i:c for i,c in enumerate(CLASSES)}

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)
valid_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
hflip = transforms.RandomHorizontalFlip(p=1.0)

def make_cifar_resnet34(num_classes=10):
    m = resnet34(weights=None)  # no pretrained (assignment-compliant)
    m.conv1 = nn.Conv2d(3,64,3,1,1,bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, num_classes)
    return m

def _strip_module(sd):
    return { (k.replace("module.","",1) if k.startswith("module.") else k): v for k,v in sd.items() }

class CIFAR10Test(Dataset):
    def __init__(self, root):
        self.root = root
        self.files = sorted([p.name for p in os.scandir(root) if p.name.endswith(".png")],
                            key=lambda s: int(s.split(".")[0]))
    def __len__(self): return len(self.files)
    def __getitem__(self, i):
        f = self.files[i]
        img = Image.open(f"{self.root}/{f}").convert("RGB")
        return img, f  # return PIL; we'll collate manually

# Build model and load best (EMA applied for stable preds)
m = make_cifar_resnet34(len(CLASSES)).to(DEVICE)
ckpt = torch.load(BEST_PATH, map_location="cpu")
m.load_state_dict(_strip_module(ckpt["model"]), strict=True)
ema_shadow = _strip_module(ckpt["ema"])
backup = {k: v.detach().clone() for k,v in m.state_dict().items()}
m.load_state_dict(ema_shadow, strict=True)

# Important: custom collate_fn so PIL images aren't auto-collated
test_loader = DataLoader(
    CIFAR10Test(TEST_DIR),
    batch_size=512, shuffle=False, num_workers=2, pin_memory=True,
    collate_fn=lambda batch: batch  # <-- returns list of (PIL, name)
)

ids, preds = [], []
m.eval()
with torch.no_grad():
    for batch in test_loader:
        imgs, names = zip(*batch)  # tuples of PIL images and filenames
        xb  = torch.stack([valid_tfms(im) for im in imgs]).to(DEVICE, non_blocking=True)
        xbF = torch.stack([valid_tfms(hflip(im)) for im in imgs]).to(DEVICE, non_blocking=True)
        logits = m(xb) + m(xbF)
        p = logits.argmax(1).tolist()
        ids.extend([n.split(".")[0] for n in names])
        preds.extend([idx2cls[i] for i in p])

# restore original weights (optional)
m.load_state_dict(backup, strict=True)

sub = pd.DataFrame({"id": ids, "label": preds})
sub.to_csv(f"{WORK}/submission.csv", index=False)
print("✔ submission.csv", sub.shape)
print(sub.head())





