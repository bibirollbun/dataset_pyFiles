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


# ==============================================================
# Ensure CIFAR-10 data is extracted with py7zr (idempotent)
# Target layout after this cell:
#   /kaggle/working/cifar-10/
#       ├─ train/  (50k PNGs)
#       ├─ test/   (~300k PNGs)
#       ├─ trainLabels.csv
#       └─ sampleSubmission.csv
# ==============================================================

# install/import py7zr
try:
    import py7zr  # noqa
except Exception:
    !pip -q install py7zr
    import py7zr

import os, shutil, time
from pathlib import Path

DEST_ROOT = Path("/kaggle/working/cifar-10")
TRAIN_DIR = DEST_ROOT / "train"
TEST_DIR  = DEST_ROOT / "test"
DEST_ROOT.mkdir(parents=True, exist_ok=True)

def count_pngs(d: Path):
    if not d.exists(): return 0
    return sum(1 for f in os.scandir(d) if f.is_file() and f.name.lower().endswith(".png"))

def find_cifar10_source_root() -> Path:
    # look for archives + CSVs under /kaggle/input
    candidates = [Path("/kaggle/input/cifar-10"), Path("/kaggle/input/cifar10"), Path("/kaggle/input")]
    if Path("/kaggle/input").exists():
        for p in Path("/kaggle/input").iterdir():
            if p.is_dir() and "cifar" in p.name.lower() and "10" in p.name.lower():
                candidates.append(p)
    for root in candidates:
        tr = list(root.rglob("train.7z")) + list(root.rglob("train.7z.001"))
        te = list(root.rglob("test.7z"))  + list(root.rglob("test.7z.001"))
        c1 = list(root.rglob("trainLabels.csv"))
        c2 = list(root.rglob("sampleSubmission.csv"))
        if tr and te and c1 and c2:
            return root
    raise FileNotFoundError("CIFAR-10 archives not found under /kaggle/input.")

def first_archive(root: Path, base: str) -> Path:
    mp = list(root.rglob(f"{base}.7z.001"))
    if mp: return mp[0]
    sp = list(root.rglob(f"{base}.7z"))
    if sp: return sp[0]
    raise FileNotFoundError(f"Missing {base}.7z(.001) under {root}")

def extract_7z(archive_path: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"→ Extracting {archive_path} → {out_dir} ...")
    with py7zr.SevenZipFile(archive_path, mode='r') as z:
        z.extractall(path=out_dir)
    print(f"   done in {time.time()-t0:.1f}s")

def normalize_extraction(parent: Path, target_subdir: str):
    tgt = parent / target_subdir
    tgt.mkdir(exist_ok=True)
    # move content from any sibling dir named like target
    sibs = [d for d in parent.iterdir() if d.is_dir() and target_subdir in d.name.lower() and d != tgt]
    if sibs and count_pngs(tgt) == 0:
        for p in sibs[0].iterdir():
            shutil.move(str(p), str(tgt))
        shutil.rmtree(sibs[0], ignore_errors=True)
    # move PNGs that landed directly under parent
    for f in list(parent.iterdir()):
        if f.is_file() and f.suffix.lower()==".png":
            shutil.move(str(f), str(tgt / f.name))

def ensure_csvs(src_root: Path, dest_root: Path):
    for name in ["trainLabels.csv", "sampleSubmission.csv"]:
        src = next(iter(src_root.rglob(name)))
        dst = dest_root / name
        if not dst.exists():
            shutil.copy2(src, dst)
            print(f"→ Copied {name} to {dst}")

SRC_ROOT = find_cifar10_source_root()
print("Source root:", SRC_ROOT)

need_train = not TRAIN_DIR.exists() or count_pngs(TRAIN_DIR) < 50000
need_test  = not TEST_DIR.exists()  or count_pngs(TEST_DIR)  < 300000  # soft threshold

if not need_train and not need_test:
    print("✅ train/ and test/ already present. Skipping extraction.")
else:
    if need_train:
        extract_7z(first_archive(SRC_ROOT, "train"), DEST_ROOT)
        normalize_extraction(DEST_ROOT, "train")
        print("   train PNGs:", count_pngs(TRAIN_DIR))
    if need_test:
        extract_7z(first_archive(SRC_ROOT, "test"), DEST_ROOT)
        normalize_extraction(DEST_ROOT, "test")
        print("   test  PNGs:", count_pngs(TEST_DIR))

ensure_csvs(SRC_ROOT, DEST_ROOT)

# Export explicit paths for the rest of the notebook
from pathlib import Path
DATA_ROOT = DEST_ROOT
TRAIN_CSV = DATA_ROOT / "trainLabels.csv"
SAMPLE_SUB = DATA_ROOT / "sampleSubmission.csv"
print("\n✅ Ready.")
print(f"DATA_ROOT = {DATA_ROOT}\nTRAIN_DIR = {TRAIN_DIR}\nTEST_DIR = {TEST_DIR}\nTRAIN_CSV = {TRAIN_CSV}\nSAMPLE_SUB = {SAMPLE_SUB}")



# ==============================
# Imports & config
# ==============================
import os, math, random, csv, time
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit

SEED = 1337
def set_seed(s=SEED):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print("Device:", DEVICE)

CFG = {
    "epochs": 50,
    "batch_size": 512,                 # lower if OOM
    "num_workers": max(2, (os.cpu_count() or 4)//2),
    "base_lr": 2e-3,
    "weight_decay": 5e-4,
    "warmup_pct": 0.05,
    "cutmix_alpha": 1.0,
    "cutmix_prob": 0.5,
    "label_smoothing": 0.1,
    "ema_decay": 0.999,
    "val_split": 5000,
}

# ---- stronger-but-safe defaults ----
CFG["epochs"]        = 120          # more cosine budget
CFG["base_lr"]       = 3e-3        # a bit higher works well with cosine
CFG["weight_decay"]  = 1e-3        # slightly stronger regularization
CFG["label_smoothing"]= 0.05       # less smoothing → slightly higher ceiling
CFG["cutmix_prob"]   = 0.6         # we’ll ramp to this below

CLASS_NAMES = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
CLASS2IDX = {c:i for i,c in enumerate(CLASS_NAMES)}
IDX2CLASS = {i:c for c,i in CLASS2IDX.items()}

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD  = (0.2023, 0.1994, 0.2010)



# ==============================
# Datasets & transforms  (PATCHED: force .png)
# ==============================
import csv
from pathlib import Path
from PIL import Image
import torchvision.transforms as T

def _with_png(s: str) -> str:
    s = str(s)
    return s if s.lower().endswith(".png") else (s + ".png")

class Cifar10PNGs(torch.utils.data.Dataset):
    def __init__(self, root: Path, csv_path: Path, transform=None, indices=None):
        self.root = Path(root)
        self.transform = transform
        rows = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                # r["id"] may be like "40895" -> make it "40895.png"
                fid = _with_png(r["id"])
                rows.append((fid, r["label"]))
        # sort numerically on stem to keep stable order
        rows.sort(key=lambda x: int(Path(x[0]).stem))

        self.ids = [r[0] for r in rows]
        self.labels = [CLASS2IDX[r[1]] for r in rows]

        if indices is not None:
            self.ids    = [self.ids[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        img_path = self.root / self.ids[i]          # now guaranteed to end with .png
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.labels[i]

class Cifar10TestPNGs(torch.utils.data.Dataset):
    def __init__(self, root: Path, sample_csv: Path, transform=None):
        self.root = Path(root)
        self.transform = transform
        self.ids = []
        with open(sample_csv, "r") as f:
            reader = csv.DictReader(f)
            for r in reader:
                self.ids.append(_with_png(r["id"]))  # ensure .png
        if not self.ids:  # fallback (unlikely)
            files = sorted(self.root.glob("*.png"), key=lambda p: int(p.stem))
            self.ids = [p.name for p in files]

    def __len__(self): return len(self.ids)

    def __getitem__(self, i):
        img_path = self.root / self.ids[i]
        img = Image.open(img_path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, self.ids[i]

from torchvision.transforms import RandAugment

train_tf = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    RandAugment(num_ops=2, magnitude=12),   # stronger than TrivialAugmentWide here
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
    T.RandomErasing(p=0.30, scale=(0.02, 0.2), ratio=(0.3, 3.3), value='random'),
])

valid_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# Should print an existing file, e.g., ".../train/1.png"
print("Sample train path:", (TRAIN_DIR / _with_png("1")))
assert (TRAIN_DIR / _with_png("1")).exists(), "train image missing?"



# ==============================
# Cell 3 (PATCHED): Split train/val in *dataset order* & build loaders
# ==============================
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import csv

# Build a temporary dataset (no indices) to get the labels in the SAME
# order as the dataset uses (i.e., sorted by numeric id with ".png" added)
_all_ds = Cifar10PNGs(TRAIN_DIR, TRAIN_CSV, transform=None, indices=None)
labels_ordered = np.array(_all_ds.labels)
n_total = len(_all_ds)

# Stratified split on these ordered labels
sss = StratifiedShuffleSplit(n_splits=1, test_size=CFG["val_split"], random_state=SEED)
train_idx, val_idx = next(sss.split(np.zeros(n_total), labels_ordered))

# Real train/val datasets & loaders
train_ds = Cifar10PNGs(TRAIN_DIR, TRAIN_CSV, transform=train_tf, indices=train_idx)
val_ds   = Cifar10PNGs(TRAIN_DIR, TRAIN_CSV, transform=valid_tf, indices=val_idx)

train_loader = DataLoader(train_ds,
                          batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["num_workers"], pin_memory=True, drop_last=True)

val_loader   = DataLoader(val_ds,
                          batch_size=1024, shuffle=False,
                          num_workers=CFG["num_workers"], pin_memory=True)

# Test loader unchanged
test_ds  = Cifar10TestPNGs(TEST_DIR, SAMPLE_SUB, transform=valid_tf)
test_loader = DataLoader(test_ds,
                         batch_size=1024, shuffle=False,
                         num_workers=CFG["num_workers"], pin_memory=True)

print(f"Train: {len(train_ds)}  Val: {len(val_ds)}  Test: {len(test_ds)}")



# ==============================
# Model: ResNet inspired model (from scratch)
# ==============================
def conv_bn(i,o,ks=3,s=1,p=1):
    return nn.Sequential(
        nn.Conv2d(i,o,ks,s,p,bias=False),
        nn.BatchNorm2d(o),
        nn.ReLU(inplace=True),
    )

class Residual(nn.Module):
    def __init__(self, c): 
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(c,c,3,padding=1,bias=False), nn.BatchNorm2d(c), nn.ReLU(inplace=True),
            nn.Conv2d(c,c,3,padding=1,bias=False), nn.BatchNorm2d(c)
        )
        self.act = nn.ReLU(inplace=True)
    def forward(self,x): return self.act(self.block(x)+x)

class ResNet9(nn.Module):
    def __init__(self, num_classes=10, drop=0.1):
        super().__init__()
        self.stem = nn.Sequential(conv_bn(3,64), conv_bn(64,128), nn.MaxPool2d(2))
        self.res1 = Residual(128)
        self.mid  = nn.Sequential(conv_bn(128,256), nn.MaxPool2d(2), conv_bn(256,512), nn.MaxPool2d(2))
        self.res2 = Residual(512)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(512,num_classes))
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if isinstance(m, nn.BatchNorm2d): nn.init.constant_(m.weight,1.0); nn.init.constant_(m.bias,0.0)
        if isinstance(m, nn.Linear): nn.init.normal_(m.weight,0,0.01); nn.init.constant_(m.bias,0.0)
    def forward(self,x):
        x=self.stem(x); x=self.res1(x); x=self.mid(x); x=self.res2(x); return self.head(x)


model = ResNet9().to(DEVICE)

def maybe_torch_compile(m):
    if DEVICE != "cuda":
        print("CPU run: torch.compile skipped.")
        return m
    try:
        major, minor = torch.cuda.get_device_capability()
        name = torch.cuda.get_device_name(0)
    except Exception:
        major, minor, name = 0, 0, "Unknown CUDA device"
    if major >= 7:  # Volta/Turing/Ampere+ only
        try:
            m = torch.compile(m, dynamic=True)  # default inductor backend
            print("✅ torch.compile enabled (Inductor).")
        except Exception as e:
            print("⚠️ torch.compile failed, running eager:", e)
    else:
        print(f"⚠️ Skipping torch.compile on {name} (compute capability {major}.{minor} < 7.0).")
    return m

model = maybe_torch_compile(model)

# (optional) print params
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
sum(p.numel() for p in model.parameters())/1e6, "M params"



# ==============================
# Optimizer, LR schedule, EMA, loss (patched for torch.compile)
# ==============================
import math
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

# --- helper: unwrap compiled models (no-op if not compiled) ---
def unwrap_model(m):
    return getattr(m, "_orig_mod", m)

optimizer = AdamW(model.parameters(), lr=CFG["base_lr"], weight_decay=CFG["weight_decay"])

total_steps  = CFG["epochs"] * len(train_loader)
warmup_steps = int(CFG["warmup_pct"] * total_steps)

def lr_lambda(step):
    if step < warmup_steps:
        return float(step) / float(max(1, warmup_steps))
    progress = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
criterion = nn.CrossEntropyLoss(label_smoothing=CFG["label_smoothing"])

class ModelEMA:
    def __init__(self, model, decay=0.999):
        # create same arch on same device
        self.ema = ResNet9().to(DEVICE)
        # initialize from the UNWRAPPED model (handles torch.compile)
        self.ema.load_state_dict(unwrap_model(model).state_dict())
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        src = unwrap_model(model)  # robust to compiled/non-compiled
        # update parameters
        for ema_p, p in zip(self.ema.parameters(), src.parameters()):
            ema_p.mul_(self.decay).add_(p.data, alpha=(1.0 - self.decay))
        # keep BN buffers (running_mean/var, num_batches_tracked) in sync
        for ema_b, b in zip(self.ema.buffers(), src.buffers()):
            ema_b.copy_(b)

ema = ModelEMA(model, decay=CFG["ema_decay"])
scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE=="cuda"))



# ==============================
# CutMix helpers
# ==============================
import numpy as np

def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w, cut_h = int(W*cut_rat), int(H*cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w//2, 0, W); y1 = np.clip(cy - cut_h//2, 0, H)
    x2 = np.clip(cx + cut_w//2, 0, W); y2 = np.clip(cy + cut_h//2, 0, H)
    return x1, y1, x2, y2

def apply_cutmix(x, y, alpha=1.0):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    bs, c, h, w = x.size()
    index = torch.randperm(bs, device=x.device)
    x1, y1, x2, y2 = rand_bbox(w, h, lam)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    y_a, y_b = y, y[index]
    lam = 1 - ((x2 - x1)*(y2 - y1) / (w*h))
    return x, y_a, y_b, lam



# ---- Force eager mode on old GPUs (P100 etc.) ----
import torch, torch._dynamo

# If anything tries to compile, just fall back to eager (no exception)
torch._dynamo.config.suppress_errors = True
# Clear any cached compiled graphs from earlier cells
torch._dynamo.reset()
print("Dynamo set to suppress errors; will run eager if compile is attempted.")



# ==============================
# Train & validate
# ==============================
def validate(model_eval, loader):
    model_eval.eval()
    loss_meter, acc_meter = 0.0, 0.0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
                logits = model_eval(x)
                loss = criterion(logits, y)
            loss_meter += loss.item() * x.size(0)
            acc_meter  += (logits.argmax(1) == y).sum().item()
    n = len(loader.dataset)
    return loss_meter/n, acc_meter/n

best_acc, best_path = 0.0, "best_resnet9.pth"
global_step = 0

for epoch in range(1, CFG["epochs"]+1):
    model.train()
    running_loss, running_correct = 0.0, 0
    n_seen = 0
    t0 = time.time()

    for x, y in train_loader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        use_cm = (np.random.rand() < CFG["cutmix_prob"])
        if use_cm:
            x, y_a, y_b, lam = apply_cutmix(x, y, CFG["cutmix_alpha"])

        with torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
            logits = model(x)
            loss = lam*criterion(logits, y_a) + (1-lam)*criterion(logits, y_b) if use_cm else criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        scaler.step(optimizer); scaler.update()
        scheduler.step()
        ema.update(model)

        running_loss += loss.item() * x.size(0)
        running_correct += (logits.argmax(1) == (y if not use_cm else y_a)).sum().item()
        n_seen += x.size(0)
        global_step += 1

    train_loss = running_loss / n_seen
    train_acc  = running_correct / n_seen
    val_loss, val_acc = validate(ema.ema, val_loader)
    dt = time.time() - t0
    print(f"Epoch {epoch:02d}/{CFG['epochs']} | train {train_loss:.4f}/{train_acc:.4f} | "
          f"val {val_loss:.4f}/{val_acc:.4f} | {dt:.1f}s")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(ema.ema.state_dict(), best_path)
        print(f"  ✅ Saved new best ({best_acc:.4f})")

print("Best val_acc:", best_acc)



print('hi')


# ==============================
# Inference & Kaggle submission
# ==============================
def predict_logits(xb, model_eval):
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(DEVICE=="cuda")):
        logits = model_eval(xb)
        logits += model_eval(torch.flip(xb, dims=[3]))  # simple TTA
        logits *= 0.5
    return logits

model_infer = ResNet9().to(DEVICE)
model_infer.load_state_dict(torch.load("best_resnet9.pth", map_location=DEVICE))
model_infer.eval()

# quick sanity on val
vl, va = (lambda l: (l[0], l[1]))( (lambda: validate(model_infer, val_loader))() )
print(f"Sanity val_acc (no EMA, with TTA in test only): {va:.4f}")

ids, preds = [], []
with torch.no_grad():
    for xb, idb in test_loader:
        xb = xb.to(DEVICE, non_blocking=True)
        logits = predict_logits(xb, model_infer)
        top = logits.argmax(1).cpu().numpy()
        preds.extend([IDX2CLASS[int(t)] for t in top])
        ids.extend(list(idb))
# --- after you've built the `preds` list (same as before) ---

# Read the ORIGINAL ids exactly as they appear in sampleSubmission.csv
orig_ids = []
with open(SAMPLE_SUB, "r") as f:
    reader = csv.DictReader(f)
    for r in reader:
        orig_ids.append(r["id"])

assert len(orig_ids) == len(preds), "Length mismatch between sample ids and predictions!"


############# deleting all files ########
!rm -rf /kaggle/working/*
############# deleting all files ########


# Write submission using the original ids (with or without .png as required)
out_path = "submission.csv"
with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id","label"])
    for sid, p in zip(orig_ids, preds):
        w.writerow([sid, p])

print("Wrote:", out_path, "rows:", len(orig_ids))




import pandas as pd
pd.read_csv("submission.csv").head()


