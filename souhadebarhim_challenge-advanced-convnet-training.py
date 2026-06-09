# ============================================
# CIFAR-10: Single-Cell Train → Infer → Submit
# - Custom ResNet-like + SE (no off-the-shelf)
# - AdamW + OneCycleLR (non-basic)
# - RandAugment + MixUp/CutMix + RandomErasing (non-basic)
# - AMP + channels_last
# - Extract 300k test.7z → ./test, infer, write submission.csv
# ============================================
import os, sys, time, math, random, warnings, glob, shutil, subprocess, tempfile
warnings.filterwarnings("ignore")

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler

# -----------------------
# Repro + device
# -----------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Device:", device)

# -----------------------
# Data & augmentation
# -----------------------
CIFAR_MEAN=(0.4914,0.4822,0.4465)
CIFAR_STD =(0.2470,0.2435,0.2616)

# Try RandAugment; fall back to AutoAugment(CIFAR10)
try:
    from torchvision.transforms import RandAugment
    extra_aug = RandAugment(num_ops=2, magnitude=9)
except Exception:
    from torchvision.transforms import AutoAugment, AutoAugmentPolicy
    extra_aug = AutoAugment(AutoAugmentPolicy.CIFAR10)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    extra_aug,
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    transforms.RandomErasing(p=0.25, scale=(0.02,0.2), ratio=(0.3,3.3))
])
test_tfms  = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

DATA_ROOT="/kaggle/working"
_base = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=None)
idx = np.arange(len(_base)); np.random.default_rng(SEED).shuffle(idx)
VAL_SZ=5000
train_idx, val_idx = idx[VAL_SZ:], idx[:VAL_SZ]

train_set = Subset(datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=train_tfms), train_idx.tolist())
val_set   = Subset(datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=test_tfms),  val_idx.tolist())

BATCH_TRAIN, BATCH_VAL = 256, 512
train_loader = DataLoader(train_set, batch_size=BATCH_TRAIN, shuffle=True,  num_workers=2, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_VAL,   shuffle=False, num_workers=2, pin_memory=True)

print(f"Train rows: {len(train_set)} | Val rows: {len(val_set)}")

# -----------------------
# Custom model (ResBlock + SE)
# -----------------------
class SE(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.fc1 = nn.Conv2d(c, c//r, 1)
        self.fc2 = nn.Conv2d(c//r, c, 1)
    def forward(self, x):
        s = x.mean(dim=(2,3), keepdim=True)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_c)
        self.se    = SE(out_c, r=16)
        self.skip  = (nn.Identity() if (in_c==out_c and stride==1)
                      else nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False),
                                         nn.BatchNorm2d(out_c)))
    def forward(self, x):
        s = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.bn2(self.conv2(x))
        x = self.se(x)
        x = F.relu(x + s, inplace=True)
        return x

class CifarSEnet(nn.Module):
    def __init__(self, num_classes=10, width=64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, width, 3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True)
        )
        self.layer1 = nn.Sequential(ResBlock(width, width, 1), ResBlock(width, width, 1))
        self.layer2 = nn.Sequential(ResBlock(width, width*2, 2), ResBlock(width*2, width*2, 1))
        self.layer3 = nn.Sequential(ResBlock(width*2, width*4, 2), ResBlock(width*4, width*4, 1))
        self.head   = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(width*4, num_classes))
        self.apply(self._init)
    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01); nn.init.zeros_(m.bias)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = self.head(x)
        return x

model = CifarSEnet(num_classes=10, width=64).to(device).to(memory_format=torch.channels_last)

# -----------------------
# MixUp / CutMix utilities
# -----------------------
def rand_bbox(W, H, lam):
    cut_rat = math.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

def apply_mixup_cutmix(x, y, p_mixup=0.5, p_cutmix=0.5, alpha_mix=0.4, alpha_cut=1.0):
    do_mixup  = np.random.rand() < p_mixup
    do_cutmix = (not do_mixup) and (np.random.rand() < p_cutmix)
    if do_mixup:
        lam = np.random.beta(alpha_mix, alpha_mix)
        idx = torch.randperm(x.size(0), device=x.device)
        x = lam * x + (1 - lam) * x[idx]
        y_a, y_b = y, y[idx]
        return x, y_a, y_b, lam, "mixup"
    elif do_cutmix:
        lam = np.random.beta(alpha_cut, alpha_cut)
        idx = torch.randperm(x.size(0), device=x.device)
        x1,y1,x2,y2 = rand_bbox(x.size(3), x.size(2), lam)
        x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
        lam = 1 - ((x2-x1)*(y2-y1) / (x.size(-1)*x.size(-2)))
        y_a, y_b = y, y[idx]
        return x, y_a, y_b, lam, "cutmix"
    else:
        return x, y, y, 1.0, "none"

# -----------------------
# Optimizer / LR / EMA / Loss / AMP
# -----------------------
EPOCHS = 60
BASE_LR = 5e-3
WEIGHT_DECAY = 5e-4

opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)
steps_per_epoch = len(train_loader)
sched = torch.optim.lr_scheduler.OneCycleLR(
    opt, max_lr=BASE_LR, epochs=EPOCHS, steps_per_epoch=steps_per_epoch,
    pct_start=0.2, div_factor=10.0, final_div_factor=1e3, anneal_strategy='cos'
)
crit = nn.CrossEntropyLoss(label_smoothing=0.05)
scaler = GradScaler()

# Simple EMA of weights (optional but helpful)
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k,v in model.state_dict().items() if v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            if k in self.shadow and v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1-self.decay)
    def load_shadow(self, model):
        state = model.state_dict()
        for k, v in self.shadow.items():
            state[k].copy_(v)

ema = EMA(model, decay=0.999)

# -----------------------
# Train / Eval loops
# -----------------------
def run_epoch(dl, train=True):
    model.train(train)
    total = correct = loss_sum = 0.0
    for (xb, yb) in dl:
        xb = xb.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        yb = yb.to(device, non_blocking=True)

        if train:
            xb_aug, ya, yb2, lam, mode = apply_mixup_cutmix(xb, yb)  # mix/cut/none
            opt.zero_grad(set_to_none=True)
            with autocast():
                logits = model(xb_aug)
                if mode == "none":
                    loss = crit(logits, ya)
                    pred  = logits.argmax(1)
                    correct += (pred == ya).sum().item()
                else:
                    loss = lam*crit(logits, ya) + (1-lam)*crit(logits, yb2)
                    # for accuracy, use primary labels as proxy
                    pred = logits.argmax(1)
                    correct += (pred == ya).sum().item()*lam + (pred == yb2).sum().item()*(1-lam)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            ema.update(model)
            loss_sum += loss.item() * xb.size(0)
            total    += xb.size(0)
        else:
            with torch.no_grad(), autocast():
                logits = model(xb)
                loss = crit(logits, yb)
                pred = logits.argmax(1)
            loss_sum += loss.item() * xb.size(0)
            correct  += (pred == yb).sum().item()
            total    += xb.size(0)

    return loss_sum/total, correct/total

best_acc = 0.0
best_state = None

for e in range(1, EPOCHS+1):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    # evaluate with EMA weights for stability
    saved = {k:v.detach().clone() for k,v in model.state_dict().items()}
    ema.load_shadow(model)
    va_loss, va_acc = run_epoch(val_loader, False)
    model.load_state_dict(saved)  # restore train weights

    if va_acc > best_acc:
        best_acc = va_acc
        best_state = {k:v.detach().cpu() for k,v in model.state_dict().items()}
    print(f"Ep{e:02d}/{EPOCHS}  lr={sched.get_last_lr()[0]:.5f}  train_acc={tr_acc:.3f}  val_acc(EMA)={va_acc:.3f}")

torch.save(best_state, "best_custom.pt")
print(f"✅ Saved best_custom.pt (best val_acc={best_acc:.4f})")

# -----------------------
# Extract Kaggle test set (300k) to ./test once
# -----------------------
COMP_DIR = "/kaggle/input/cifar-10"
TEST_7Z  = os.path.join(COMP_DIR, "test.7z")
TEST_DIR = "./test"
TOTAL    = 300_000
assert os.path.exists(TEST_7Z), "Missing /kaggle/input/cifar-10/test.7z"

def ensure_7z():
    if shutil.which("7z"): return "7z"
    subprocess.run(["apt-get","update","-y"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["apt-get","install","-y","p7zip-full"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    assert shutil.which("7z"), "7z not found after install"
    return "7z"

os.makedirs(TEST_DIR, exist_ok=True)
have = len(glob.glob(os.path.join(TEST_DIR, "*.png")))
if have >= TOTAL:
    print(f"✅ Already extracted: {have}/{TOTAL}")
else:
    sevenz = ensure_7z()
    print(f"Extracting test set to {os.path.abspath(TEST_DIR)} ...")
    proc = subprocess.Popen([sevenz, "x", TEST_7Z, "-y", "-o."], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    last = -1
    while proc.poll() is None:
        done = len(glob.glob(os.path.join(TEST_DIR, "*.png")))
        if done != last:
            pct = 100.0 * min(done, TOTAL) / TOTAL
            print(f"\rprogress: {done}/{TOTAL} ({pct:5.1f}%)", end="")
            last = done
        time.sleep(1)
    done = len(glob.glob(os.path.join(TEST_DIR, "*.png")))
    print(f"\rprogress: {done}/{TOTAL} (100.0%)")
    print(f"✅ Extracted: {done}/{TOTAL}")

# -----------------------
# Inference on test → submission.csv
# -----------------------
import pandas as pd
from PIL import Image

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
SAMPLE_SUB  = os.path.join(COMP_DIR, "sampleSubmission.csv")
assert os.path.exists(SAMPLE_SUB), "Missing sampleSubmission.csv"
sample = pd.read_csv(SAMPLE_SUB)
ids = sample["id"].astype(int).tolist()
N = len(ids)
print("IDs to predict:", N)

# Rebuild model and load best weights (EMA-evaluated checkpoint)
model_inf = CifarSEnet(num_classes=10, width=64).to(device).to(memory_format=torch.channels_last)
state = torch.load("best_custom.pt", map_location="cpu")
model_inf.load_state_dict(state)
model_inf.eval()

eval_tfm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(CIFAR_MEAN, CIFAR_STD)])

BATCH = 1024
pred_idx = np.empty(N, dtype=np.int64)

def batched(iterable, n):
    for i in range(0, len(iterable), n):
        yield i, iterable[i:i+n]

with torch.no_grad():
    done=0
    for start, id_batch in batched(ids, BATCH):
        imgs=[]
        for i in id_batch:
            p = os.path.join(TEST_DIR, f"{i}.png")
            img = Image.open(p).convert("RGB")
            imgs.append(eval_tfm(img))
        xb = torch.stack(imgs).to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        with autocast():
            logits = model_inf(xb)
        preds = logits.argmax(1).cpu().numpy()
        pred_idx[start:start+len(id_batch)] = preds
        done += len(id_batch)
        if done % 10000 == 0:
            print(f"progress: {done}/{N}")

# Build submission
labels = [CLASS_NAMES[i] for i in pred_idx.tolist()]
sub = pd.DataFrame({"id": np.array(ids, dtype=np.int32), "label": labels})
sub.to_csv("submission.csv", index=False)
print("✅ Wrote submission.csv")
print(sub.head())
print("shape:", sub.shape, "| ids unique:", sub['id'].is_unique)



# ================================
# Extra Cell: Export standalone submission(s)
# - Ensures a clean submission.csv
# - Also writes a timestamped copy for easy download
# ================================
import os, time, numpy as np, pandas as pd

CLASS_NAMES = ['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']

def build_from_npy(ids_path="ids.npy", pred_path="pred_idx.npy"):
    ids = np.load(ids_path)
    pred_idx = np.load(pred_path)
    labels = [CLASS_NAMES[i] for i in pred_idx.tolist()]
    return pd.DataFrame({"id": ids.astype(int), "label": labels})

# Prefer rebuilding from npy artifacts if they exist; else reuse existing CSV
if os.path.exists("ids.npy") and os.path.exists("pred_idx.npy"):
    sub = build_from_npy()
else:
    sub = pd.read_csv("submission.csv")  # fallback if already created

# Write main file
sub.to_csv("submission.csv", index=False)

# Also write a timestamped standalone copy (shows up separately in Kaggle Output)
stamp = time.strftime("%Y%m%d-%H%M%S")
standalone_name = f"submission_standalone_{stamp}.csv"
sub.to_csv(standalone_name, index=False)

print("✅ Wrote:", "submission.csv", "and", standalone_name)
print("Tip: Both files appear in the Kaggle 'Output' tab; you can download either one directly.")


