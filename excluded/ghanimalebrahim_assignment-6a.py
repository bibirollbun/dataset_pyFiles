# ===========================
# Cell 1: Imports • Config • Transforms • Extract (.7z) • Datasets • Loaders
# ===========================
import os, random, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10

# ----- Repro & Device -----
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# ----- Kaggle roots -----
KAGGLE_ROOT = Path("/kaggle/input/cifar-10")
WORK_ROOT   = Path("/kaggle/working/cifar10")  # where we extract
WORK_TRAIN  = WORK_ROOT / "train"
WORK_TEST   = WORK_ROOT / "test"
USE_KAGGLE  = KAGGLE_ROOT.exists()

# ----- Labels -----
LABELS = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
label2id = {c:i for i,c in enumerate(LABELS)}
id2label = {i:c for c,i in label2id.items()}

# ----- Transforms -----
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)),
    transforms.RandomErasing(p=0.5, scale=(0.02,0.15), ratio=(0.3,3.3), value="random"),
])
test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914,0.4822,0.4465), std=(0.2023,0.1994,0.2010)),
])

VAL_RATIO   = 0.1
NUM_WORKERS = 4

def make_loader(ds, bs, shuffle):
    return DataLoader(
        ds, batch_size=bs, shuffle=shuffle,
        num_workers=NUM_WORKERS, pin_memory=True,
        persistent_workers=(NUM_WORKERS > 0), drop_last=shuffle
    )

# --------------------------
# Extract helper for .7z archives
# --------------------------
def ensure_extracted():
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    train_pngs = list(WORK_TRAIN.glob("*.png"))
    test_pngs  = list(WORK_TEST.glob("*.png"))

    need_train = len(train_pngs) == 0
    need_test  = len(test_pngs)  == 0

    if not (need_train or need_test):
        print(f"[extract] Already extracted: train={len(train_pngs)} test={len(test_pngs)}")
        return

    try:
        import py7zr  # noqa
    except Exception:
        import sys, subprocess
        print("[extract] Installing py7zr …")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "py7zr"])
        import py7zr  # noqa

    if need_train:
        src = KAGGLE_ROOT / "train.7z"
        assert src.exists(), f"Missing {src}"
        WORK_TRAIN.mkdir(parents=True, exist_ok=True)
        print("[extract] Extracting train.7z →", WORK_TRAIN)
        with py7zr.SevenZipFile(src, mode='r') as z:
            z.extractall(path=WORK_TRAIN)

    if need_test:
        src = KAGGLE_ROOT / "test.7z"
        assert src.exists(), f"Missing {src}"
        WORK_TEST.mkdir(parents=True, exist_ok=True)
        print("[extract] Extracting test.7z →", WORK_TEST)
        with py7zr.SevenZipFile(src, mode='r') as z:
            z.extractall(path=WORK_TEST)

    # flatten nested folders if any
    for split_dir in [WORK_TRAIN, WORK_TEST]:
        subdirs = [p for p in split_dir.iterdir() if p.is_dir()]
        if subdirs:
            for sd in subdirs:
                for p in sd.rglob("*.png"):
                    target = split_dir / p.name
                    if not target.exists():
                        p.replace(target)
            for sd in subdirs:
                try: sd.rmdir()
                except Exception: pass

if USE_KAGGLE:
    ensure_extracted()

# --------------------------
# Datasets (competition-style)
# --------------------------
if USE_KAGGLE:
    csv_path = KAGGLE_ROOT / "trainLabels.csv"
    assert csv_path.exists(), f"Missing labels CSV at {csv_path}"

    df_all = pd.read_csv(csv_path)
    id2path = {p.stem: p for p in WORK_TRAIN.glob("*.png")}
    before = len(df_all)
    df_all = df_all[df_all["id"].astype(str).isin(id2path.keys())].reset_index(drop=True)
    missing = before - len(df_all)
    print(f"[cifar-10] kept {len(df_all)}/{before} | missing files after extract: {missing}")
    assert len(df_all) > 0, "No train images found after extraction."

    idx = np.arange(len(df_all)); np.random.shuffle(idx)
    v = int(len(idx) * VAL_RATIO)
    val_idx, tr_idx = idx[:v], idx[v:]

    class CSVCIFARTrain(Dataset):
        def __init__(self, df, id2path, transform):
            self.df, self.id2path, self.transform = df, id2path, transform
        def __len__(self): return len(self.df)
        def __getitem__(self, i):
            row = self.df.iloc[i]
            pid = str(row["id"])
            img = Image.open(self.id2path[pid]).convert("RGB")
            return self.transform(img), label2id[row["label"]]

    train_ds = CSVCIFARTrain(df_all.iloc[tr_idx].reset_index(drop=True), id2path, train_tfms)
    val_ds   = CSVCIFARTrain(df_all.iloc[val_idx].reset_index(drop=True), id2path, test_tfms)

    test_ids = [p.stem for p in WORK_TEST.glob("*.png")]
    class FolderTest(Dataset):
        def __init__(self, ids, base, transform):
            self.ids, self.base, self.transform = ids, base, transform
        def __len__(self): return len(self.ids)
        def __getitem__(self, i):
            pid = self.ids[i]
            img = Image.open(self.base / f"{pid}.png").convert("RGB")
            return self.transform(img), pid
    test_ds = FolderTest(test_ids, WORK_TEST, test_tfms)

else:
    train_all = CIFAR10(root="./", train=True,  download=True, transform=train_tfms)
    val_all   = CIFAR10(root="./", train=True,  download=True, transform=test_tfms)
    n = len(train_all)
    idx = np.arange(n); np.random.shuffle(idx)
    v = int(n * VAL_RATIO)
    val_idx, tr_idx = idx[:v], idx[v:]
    train_ds, val_ds = Subset(train_all, tr_idx), Subset(val_all, val_idx)
    test_ds = CIFAR10(root="./", train=False, download=True, transform=test_tfms)

# ----- Build loaders -----
BATCH = 384  # P100 16GB; fallback if needed
try:
    train_dl = make_loader(train_ds, BATCH, True)
    val_dl   = make_loader(val_ds,   BATCH, False)
    test_dl  = make_loader(test_ds,  BATCH, False)
except RuntimeError:
    BATCH = 256
    train_dl = make_loader(train_ds, BATCH, True)
    val_dl   = make_loader(val_ds,   BATCH, False)
    test_dl  = make_loader(test_ds,  BATCH, False)

print("Device:", device, "| Batch:", BATCH, "| Using Kaggle:", USE_KAGGLE)



# ===========================
# Cell 2: Custom residual net (PreAct + SE + DropPath)
# ===========================
import torch, torch.nn as nn

def conv3x3(c_in, c_out, stride=1, groups=1):
    return nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1,
                     groups=groups, bias=False)

def conv1x1(c_in, c_out, stride=1):
    return nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, bias=False)

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0): super().__init__(); self.p=float(p)
    def forward(self, x):
        if (not self.training) or self.p==0.0: return x
        keep = 1 - self.p
        shape = (x.size(0),) + (1,)*(x.ndim-1)
        return x / keep * x.new_empty(shape).bernoulli_(keep)

class SE(nn.Module):
    def __init__(self, ch, r=16):
        super().__init__()
        m = max(8, ch//r)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(ch, m, 1, bias=True), nn.ReLU(True),
                                nn.Conv2d(m, ch, 1, bias=True), nn.Sigmoid())
    def forward(self, x): return x * self.fc(self.pool(x))

class PreActBlock(nn.Module):
    def __init__(self, c_in, c_out, stride=1, se=False, drop_path=0.0):
        super().__init__()
        self.bn1  = nn.BatchNorm2d(c_in);  self.act1 = nn.ReLU(True)
        self.conv1= conv3x3(c_in, c_out, stride)
        self.bn2  = nn.BatchNorm2d(c_out); self.act2 = nn.ReLU(True)
        self.conv2= conv3x3(c_out, c_out, 1)
        self.se   = SE(c_out) if se else nn.Identity()
        self.down = conv1x1(c_in, c_out, stride) if (stride!=1 or c_in!=c_out) else None
        self.drop = DropPath(drop_path)
    def forward(self, x):
        idt = x
        out = self.act1(self.bn1(x))
        if self.down is not None: idt = self.down(out)
        out = self.conv1(out)
        out = self.conv2(self.act2(self.bn2(out)))
        out = self.se(out)
        out = self.drop(out)
        return out + idt

class GhanimNetV2(nn.Module):
    def __init__(self, num_classes=10, se_every=2, drop_path_max=0.15, widths=(96,192,384,512), depths=(3,4,6,9)):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, widths[0], 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(widths[0]), nn.ReLU(True))
        total_blocks = sum(depths)
        def make_stage(c_in, c_out, n, stride, start_idx):
            blocks=[]
            for i in range(n):
                dp = drop_path_max * (start_idx+i) / max(1,total_blocks-1)
                blocks.append(PreActBlock(c_in if i==0 else c_out, c_out,
                                          stride if i==0 else 1,
                                          se=((i%se_every)==(se_every-1)), drop_path=dp))
            return nn.Sequential(*blocks)
        idx=0
        self.layer1 = make_stage(widths[0], widths[0], depths[0], 1, idx); idx+=depths[0]
        self.layer2 = make_stage(widths[0], widths[1], depths[1], 2, idx); idx+=depths[1]
        self.layer3 = make_stage(widths[1], widths[2], depths[2], 2, idx); idx+=depths[2]
        self.layer4 = make_stage(widths[2], widths[3], depths[3], 2, idx); idx+=depths[3]
        self.bn_head= nn.BatchNorm2d(widths[3]); self.act_head=nn.ReLU(True)
        self.pool = nn.AdaptiveAvgPool2d(1); self.drop=nn.Dropout(0.2)
        self.fc = nn.Linear(widths[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d): nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear): nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)
    def forward(self, x):
        x=self.stem(x)
        x=self.layer1(x); x=self.layer2(x); x=self.layer3(x); x=self.layer4(x)
        x=self.act_head(self.bn_head(x))
        x=self.pool(x).flatten(1)
        x=self.drop(x)
        return self.fc(x)

model = GhanimNetV2(num_classes=10).to(device).to(memory_format=torch.channels_last)
print("Params (M):", sum(p.numel() for p in model.parameters())/1e6)



# ===========================
# Cell 3: Aug utils • Loss • AdamW • Warmup+Cosine • Train/Val
# ===========================
import math
import numpy as np
import torch
import torch.nn as nn

# ----- CutMix -----
def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_cutmix(x, y, alpha=1.0):
    if alpha <= 0: return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    bs, c, h, w = x.size()
    index = torch.randperm(bs, device=x.device)
    x1,y1_,x2,y2_ = rand_bbox(w, h, lam)
    x[:, :, y1_:y2_, x1:x2] = x[index, :, y1_:y2_, x1:x2]
    lam = 1 - ((x2-x1)*(y2_-y1_) / (w*h))
    return x, y, y[index], lam

# ----- Hyperparams / Optim / Scheduler / Loss -----
EPOCHS        = 200
base_lr       = 1.2e-3 * (BATCH / 256)
weight_decay  = 6e-4
cutmix_p      = 0.7
cutmix_alpha  = 1.0
label_smooth  = 0.10
grad_clip     = 1.0  # None to disable

opt   = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
crit  = nn.CrossEntropyLoss(label_smoothing=label_smooth)

# Accumulation-aware schedule (per step)
ACCUM = 1 if BATCH >= 256 else 2
steps_per_epoch = math.ceil(len(train_dl) / ACCUM)
total_steps  = max(1, EPOCHS * steps_per_epoch)
warmup_steps = int(0.10 * total_steps)

def lr_lambda(step):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * t))

sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)

# AMP scaler (new API)
scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

# ----- Train / Val epoch functions (EMA supported) -----
def run_epoch(dl, train=True, ema=None, cutmix_prob=cutmix_p, cutmix_alpha=cutmix_alpha):
    model.train(train)
    total, correct, total_loss = 0, 0, 0.0
    step_in_accum = 0

    for x, y in dl:
        x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)

        if train:
            y1, y2, lam = y, y, 1.0
            use_cm = (np.random.rand() < cutmix_prob)
            if use_cm:
                with torch.no_grad():
                    x, y1, y2, lam = apply_cutmix(x.clone(), y, alpha=cutmix_alpha)

            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(x)
                loss = lam * crit(logits, y1) + (1 - lam) * crit(logits, y2) if use_cm else crit(logits, y)

            scaler.scale(loss / ACCUM).backward()
            step_in_accum += 1

            if step_in_accum % ACCUM == 0:
                if grad_clip is not None:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                if ema is not None:
                    ema.update(model)  # per-step EMA
        else:
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                logits = model(x)
                loss = crit(logits, y)

        total_loss += loss.item() * x.size(0)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += x.size(0)

    return total_loss / total, correct / total



# ===========================
# Cell 4: Training loop • checkpoint save/resume • device-aware EMA
# ===========================
from pathlib import Path
import torch

CKPT_PATH = Path("checkpoint.pth")
BEST_PATH = Path("best.pth")         # EMA best
BEST_RAW_PATH = Path("best_raw.pth") # raw best
RESUME    = True

# ----- Device-aware EMA -----
class EMA:
    def __init__(self, model, decay=0.999, device=None):
        self.decay = float(decay)
        self.device = device if device is not None else next(model.parameters()).device
        self.shadow = {}
        self.register(model)
    @torch.no_grad()
    def register(self, model):
        self.shadow = {}
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k] = v.detach().to(self.device, dtype=torch.float32).clone()
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                v32 = v.detach().to(self.device, dtype=torch.float32)
                self.shadow[k].mul_(self.decay).add_(v32, alpha=1.0 - self.decay)
    @torch.no_grad()
    def store(self, model):
        self.backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    @torch.no_grad()
    def copy_to(self, model):
        sd = model.state_dict()
        for k, v in sd.items():
            if k in self.shadow and v.dtype.is_floating_point:
                sd[k].copy_(self.shadow[k].to(v.dtype))
        model.load_state_dict(sd)
    @torch.no_grad()
    def restore(self, model):
        if hasattr(self, "backup"):
            model.load_state_dict(self.backup); del self.backup
    def state_dict(self):  # save on CPU
        return {k: t.cpu() for k, t in self.shadow.items()}
    def load_state_dict(self, state, device):
        self.shadow = {k: t.to(device, dtype=torch.float32) for k, t in state.items()}
        self.device = device

# Move model to device and init EMA
model = model.to(device).to(memory_format=torch.channels_last)
ema = EMA(model, decay=0.999, device=device)

start_epoch = 1
best_acc = 0.0      # best EMA val acc
best_raw = 0.0      # best raw val acc

# ----- Resume (optional) -----
if RESUME and CKPT_PATH.exists():
    print(f"Resuming from {CKPT_PATH} …")
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(ckpt["model"], strict=True)
    model = model.to(device).to(memory_format=torch.channels_last)
    opt.load_state_dict(ckpt["opt"])
    sched.load_state_dict(ckpt["sched"])
    if "scaler" in ckpt:
        try: scaler.load_state_dict(ckpt["scaler"])
        except Exception: pass
    if "ema" in ckpt and isinstance(ckpt["ema"], dict) and len(ckpt["ema"]) > 0:
        ema.load_state_dict(ckpt["ema"], device=device)
    else:
        ema.register(model)
    best_acc    = ckpt.get("best_acc", 0.0)
    best_raw    = ckpt.get("best_raw", 0.0)
    start_epoch = ckpt.get("epoch", 0) + 1
    print(f"Start at epoch {start_epoch}, best val acc (EMA) {best_acc:.4f}, raw {best_raw:.4f}")

# ----- Train loop -----
for epoch in range(start_epoch, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_dl, train=True, ema=ema)
    # eval raw
    va_loss_raw, va_acc_raw = run_epoch(val_dl, train=False)
    # eval EMA
    ema.store(model); ema.copy_to(model)
    va_loss_ema, va_acc_ema = run_epoch(val_dl, train=False)
    ema.restore(model)

    # save latest
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "opt":   opt.state_dict(),
        "sched": sched.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.state_dict(),
        "best_acc": best_acc,
        "best_raw": best_raw,
        "batch": BATCH,
    }, CKPT_PATH)

    # best raw
    if va_acc_raw > best_raw:
        best_raw = va_acc_raw
        torch.save({"model": model.state_dict(), "acc": best_raw, "epoch": epoch}, BEST_RAW_PATH)

    # best EMA
    if va_acc_ema > best_acc:
        best_acc = va_acc_ema
        torch.save({"model": ema.state_dict(), "acc": best_acc, "epoch": epoch}, BEST_PATH)

    print(f"Epoch {epoch:03d}/{EPOCHS} | "
          f"train {tr_loss:.4f}/{tr_acc:.4f} | "
          f"val_raw {va_loss_raw:.4f}/{va_acc_raw:.4f} | "
          f"val_ema {va_loss_ema:.4f}/{va_acc_ema:.4f} | "
          f"best_raw {best_raw:.4f} | best_ema {best_acc:.4f}")



# ===========================
# Cell 5: Inference & Submission (force RAW best to avoid bad EMA)
# ===========================
import torch, pandas as pd
from pathlib import Path

RAW = Path("best_raw.pth")     # strong weights from raw model
EMA = Path("best.pth")         # will be good only after re-training with per-step EMA
assert RAW.exists() or EMA.exists(), "No checkpoint found. Train first."

# --- Load RAW by default (robust) ---
ckpt = torch.load(RAW if RAW.exists() else EMA, map_location="cpu")
state = ckpt["model"] if "model" in ckpt else ckpt
model.load_state_dict(state, strict=True)
model = model.to(device).to(memory_format=torch.channels_last)
model.eval()

@torch.no_grad()
def predict_tta(x):  # identity + hflip
    with torch.amp.autocast(device_type='cuda', enabled=(device.type=='cuda')):
        p1 = model(x).softmax(1)
        p2 = model(torch.flip(x, dims=[3])).softmax(1)
        return (p1 + p2) * 0.5

if USE_KAGGLE:
    ids, preds = [], []
    with torch.no_grad():
        for x, img_ids in test_dl:
            x = x.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            pred = predict_tta(x).argmax(1).tolist()
            preds.extend([id2label[i] for i in pred]); ids.extend(img_ids)
    sub = pd.DataFrame({"id": ids, "label": preds}).sort_values("id").reset_index(drop=True)
    sub.to_csv("submission.csv", index=False)
    print("Saved submission.csv (raw-best)")
else:
    # local sanity (torchvision test)
    tot=cor=0
    with torch.no_grad():
        for x,y in test_dl:
            x=x.to(device).to(memory_format=torch.channels_last); y=y.to(device)
            cor += (predict_tta(x).argmax(1)==y).sum().item(); tot += y.numel()
    print("Torchvision CIFAR-10 test accuracy:", cor/tot)





