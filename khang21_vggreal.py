import os

ROOT = "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC"
print("ROOT exists:", os.path.exists(ROOT))
print("train exists:", os.path.isdir(os.path.join(ROOT, "train")))
print("val   exists:", os.path.isdir(os.path.join(ROOT, "val")))

if os.path.isdir(os.path.join(ROOT, "train")):
    n_classes_train = len([d for d in os.listdir(os.path.join(ROOT, "train")) 
                           if os.path.isdir(os.path.join(ROOT, "train", d))])
    n_classes_val   = len([d for d in os.listdir(os.path.join(ROOT, "val")) 
                           if os.path.isdir(os.path.join(ROOT, "val", d))])
    print("train classes:", n_classes_train, "| val classes:", n_classes_val)  # mong đợi ≈ 1000

    # liệt kê thử vài lớp đầu
    print("sample train classes:", sorted(os.listdir(os.path.join(ROOT, "train")))[:5])



# =========================================================
# VGG16-BN on ImageNet-2012 — FAST/LARGE SUBSET (Kaggle-ready)
# - Subset cân bằng: 200 lớp × 400 ảnh/lớp từ train/ (không cần val.txt)
# - Tách 10% subset làm validation (đúng transform)
# - Optim: AdamW + Warmup(5 epoch) → Cosine, GradClip, AMP (torch.amp)
# - Regularization: MixUp (phase-out cuối kỳ), Label Smoothing, ColorJitter nhẹ
# - Metrics: Top-1 / Top-5 / mIoU + 2 ảnh mẫu (đúng/sai) ở best epoch
# - Checkpoint load an toàn với PyTorch >= 2.6
# =========================================================

import os, random, time, math
from collections import defaultdict
import numpy as np
import torch, torch.nn as nn, torch.optim as optim
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile
from glob import glob
import matplotlib.pyplot as plt

# ------------ Robust PIL ------------
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -------------------- Config --------------------
SEED = 1337
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
try: torch.set_float32_matmul_precision("high")
except: pass
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True
print("Device:", DEVICE)

CFG = {
    "DATA_ROOT": "/kaggle/input/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC",

    # SUBSET lớn: có thể hạ xuống nếu IO/GPU yếu (vd: 150 × 200)
    "SUBSET_CLASSES": 200,
    "IMGS_PER_CLASS": 400,
    "VAL_RATIO": 0.10,

    "MODEL": "vgg16_bn",        # 'vgg16' | 'vgg16_bn'
    "PRETRAINED": True,         # fine-tune từ weights chuẩn
    "OPTIM": "auto",            # 'auto' | 'adamw' | 'sgd' (auto: AdamW khi PRETRAINED)

    "BATCH_SIZE": 64,           # nếu OOM → 48 hoặc 32
    "EPOCHS": 40,
    "LR": 1e-4,                 # fine-tune AdamW
    "WEIGHT_DECAY": 1e-4,
    "MOMENTUM": 0.9,            # cho SGD nếu dùng

    "IMG_SIZE": 224,
    "NUM_WORKERS": 4,           # IO cao → 4 là hợp lý trên Kaggle T4
    "PREFETCH_FACTOR": 2,
    "LOG_EVERY": 200,           # in ETA mỗi 200 batch

    "LABEL_SMOOTH": 0.02,
    "MIXUP_ALPHA": 0.3,         # mạnh, sẽ phase-out về 0 ở 6 epoch cuối
    "MIXUP_PHASEOUT_EPOCHS": 6, # giảm dần alpha -> 0 ở 6 epoch cuối
    "CLIP_GRAD_NORM": 1.0,

    "WARMUP_EPOCHS": 5,         # warmup LR 5 epoch rồi mới cosine
}

TRAIN_DIR = os.path.join(CFG["DATA_ROOT"], "train")
assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"

# -------------------- Transforms --------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
train_tfms = transforms.Compose([
    transforms.RandomResizedCrop(CFG["IMG_SIZE"], scale=(0.08, 1.0), ratio=(3/4, 4/3)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.1, 0.1, 0.1, 0.05),  # nhẹ vì MixUp cao
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])
val_tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(CFG["IMG_SIZE"]),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# -------------------- Build balanced subset --------------------
def list_class_dirs(train_dir):
    return sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])

ALL_CLASSES = list_class_dirs(TRAIN_DIR)
rng = np.random.default_rng(SEED)
if CFG["SUBSET_CLASSES"] is not None and CFG["SUBSET_CLASSES"] < len(ALL_CLASSES):
    chosen = rng.choice(ALL_CLASSES, size=CFG["SUBSET_CLASSES"], replace=False)
    CLASSES_WNID = sorted(list(chosen))
else:
    CLASSES_WNID = ALL_CLASSES

class_to_idx = {wnid: i for i, wnid in enumerate(CLASSES_WNID)}
idx_to_wnid = {i: wnid for wnid, i in class_to_idx.items()}
NUM_CLASSES = len(CLASSES_WNID)
print(f"[SUBSET] Using: {NUM_CLASSES} classes × {CFG['IMGS_PER_CLASS']} imgs/class")

def collect_images_for_class(wnid, limit=None):
    folder = os.path.join(TRAIN_DIR, wnid)
    files = []
    for ext in ("*.JPEG","*.jpeg","*.jpg","*.png","*.bmp"):
        files += glob(os.path.join(folder, ext))
    rng.shuffle(files)
    return files[:limit] if (limit is not None) else files

train_items, val_items = [], []
for wnid in CLASSES_WNID:
    paths = collect_images_for_class(wnid, CFG["IMGS_PER_CLASS"])
    if len(paths) == 0:
        continue
    n_val = max(1, int(len(paths) * CFG["VAL_RATIO"]))
    val_p = paths[:n_val]; train_p = paths[n_val:]
    y = class_to_idx[wnid]
    train_items += [(p, y) for p in train_p]
    val_items   += [(p, y) for p in val_p]

print(f"[DATA] train images: {len(train_items)} | val images: {len(val_items)}")

# -------------------- Simple dataset --------------------
class SimpleImageDataset(Dataset):
    def __init__(self, items, transform):
        self.items = items
        self.transform = transform
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        path, y = self.items[i]
        img = Image.open(path).convert("RGB")
        if self.transform: img = self.transform(img)
        return img, y

train_ds = SimpleImageDataset(train_items, train_tfms)
val_ds   = SimpleImageDataset(val_items,   val_tfms)

# -------------------- DataLoaders (safe prefetch) --------------------
def make_loader(ds, train=True):
    kwargs = dict(
        batch_size=CFG["BATCH_SIZE"],
        shuffle=train,
        num_workers=CFG["NUM_WORKERS"],
        pin_memory=True,
        persistent_workers=(CFG["NUM_WORKERS"] > 0),
    )
    if CFG["NUM_WORKERS"] > 0 and "prefetch_factor" in DataLoader.__init__.__code__.co_varnames:
        kwargs["prefetch_factor"] = CFG["PREFETCH_FACTOR"]
    return DataLoader(ds, **kwargs)

train_loader = make_loader(train_ds, train=True)
val_loader   = make_loader(val_ds,   train=False)
print(f"[DATA] steps/epoch: train={len(train_loader)} | val={len(val_loader)}")

# -------------------- Model --------------------
from torchvision.models import vgg16, vgg16_bn
# chọn weights enum; fallback cho torchvision cũ
try:
    from torchvision.models import VGG16_Weights, VGG16_BN_Weights
    if CFG["MODEL"] == "vgg16_bn":
        weights = VGG16_BN_Weights.IMAGENET1K_V1 if CFG["PRETRAINED"] else None
        model = vgg16_bn(weights=weights)
    else:
        weights = VGG16_Weights.IMAGENET1K_V1 if CFG["PRETRAINED"] else None
        model = vgg16(weights=weights)
except Exception:
    model = vgg16_bn(pretrained=bool(CFG["PRETRAINED"])) if CFG["MODEL"]=="vgg16_bn" \
            else vgg16(pretrained=bool(CFG["PRETRAINED"]))

# thay head theo NUM_CLASSES
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

# -------------------- Loss / Optim / Scheduler / AMP --------------------
criterion = nn.CrossEntropyLoss(label_smoothing=CFG["LABEL_SMOOTH"])

def pick_optimizer():
    if CFG["OPTIM"].lower() == "auto":
        return "adamw" if CFG["PRETRAINED"] else "sgd"
    return CFG["OPTIM"].lower()

OPT = pick_optimizer()
if OPT == "adamw":
    optimizer = optim.AdamW(model.parameters(), lr=CFG["LR"], weight_decay=CFG["WEIGHT_DECAY"])
    print("Using AdamW")
elif OPT == "sgd":
    optimizer = optim.SGD(model.parameters(), lr=CFG["LR"], momentum=CFG["MOMENTUM"],
                          weight_decay=CFG["WEIGHT_DECAY"], nesterov=True)
    print("Using SGD+Momentum")
else:
    raise ValueError("OPTIM must be 'auto' | 'adamw' | 'sgd'")

# Warmup (5e) → Cosine
WARMUP_EPOCHS = CFG["WARMUP_EPOCHS"]
cosine_epochs = max(1, CFG["EPOCHS"] - WARMUP_EPOCHS)
warmup = LinearLR(optimizer, start_factor=1e-2, end_factor=1.0, total_iters=WARMUP_EPOCHS)
cosine = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[WARMUP_EPOCHS])

# AMP (ưu tiên torch.amp; fallback cuda.amp nếu quá cũ)
try:
    autocast = torch.amp.autocast
    GradScaler = torch.amp.GradScaler
    scaler = GradScaler('cuda', enabled=(DEVICE=="cuda"))
except AttributeError:
    autocast = torch.cuda.amp.autocast
    GradScaler = torch.cuda.amp.GradScaler
    scaler = GradScaler(enabled=(DEVICE=="cuda"))

# -------------------- MixUp --------------------
def maybe_mixup(inputs, targets, alpha):
    if alpha is None or alpha <= 0:
        return inputs, targets, None, False
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(inputs.size(0), device=inputs.device)
    mixed = lam * inputs + (1 - lam) * inputs[idx]
    return mixed, (targets, targets[idx], lam), lam, True

def apply_mix_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def current_mixup_alpha(epoch, total_epochs, base_alpha, phaseout_last):
    # epoch: 1..EPOCHS
    if base_alpha <= 0 or phaseout_last <= 0: return base_alpha
    remain = total_epochs - epoch + 1
    if remain > phaseout_last: return base_alpha
    factor = max(0.0, (remain - 1) / max(1, phaseout_last - 1))  # tuyến tính về 0
    return base_alpha * factor

# -------------------- Metrics --------------------
def accuracy_from_logits(logits, targets):
    preds = logits.argmax(1)
    return (preds == targets).float().mean().item()

def topk_accuracy_from_logits(logits, targets, k=5):
    topk = logits.topk(k, dim=1).indices
    return topk.eq(targets.view(-1,1)).any(dim=1).float().mean().item()

def iou_from_confmat(C):
    K = C.shape[0]; ious = []
    for c in range(K):
        TP = C[c, c]; FP = C[:, c].sum() - TP; FN = C[c, :].sum() - TP
        den = TP + FP + FN
        ious.append(float("nan") if den == 0 else TP / den)
    return np.array(ious)

# -------------------- Train / Eval --------------------
def train_one_epoch(model, loader, epoch, total_epochs):
    model.train()
    total_loss = total_acc = n = 0
    t0 = time.time()

    alpha_now = current_mixup_alpha(epoch, total_epochs, CFG["MIXUP_ALPHA"], CFG["MIXUP_PHASEOUT_EPOCHS"])
    for step, (images, labels) in enumerate(loader, 1):
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        inputs, tgt_mix, lam, mixed = maybe_mixup(images, labels, alpha_now)

        with autocast('cuda', enabled=(DEVICE=="cuda")):
            outputs = model(inputs)
            if mixed:
                y_a, y_b, lam_ = tgt_mix
                loss = apply_mix_criterion(criterion, outputs, y_a, y_b, lam_)
            else:
                loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        if CFG["CLIP_GRAD_NORM"] and CFG["CLIP_GRAD_NORM"] > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), CFG["CLIP_GRAD_NORM"])
        scaler.step(optimizer); scaler.update()

        bs = labels.size(0)
        total_loss += loss.item() * bs
        total_acc  += accuracy_from_logits(outputs.detach(), labels) * bs
        n += bs

        if step % CFG["LOG_EVERY"] == 0 or step == len(loader):
            elapsed = time.time() - t0
            eta = elapsed / step * (len(loader) - step)
            print(f"  step {step}/{len(loader)} | loss {loss.item():.4f} | "
                  f"elapsed {elapsed/60:.1f}m | ETA {eta/60:.1f}m | mixup α={alpha_now:.3f}")

    return total_loss / n, total_acc / n

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    totL = totA1 = totA5 = n = 0
    C = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast('cuda', enabled=(DEVICE=="cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)
        bs = labels.size(0)
        totL += loss.item() * bs
        totA1 += accuracy_from_logits(outputs, labels) * bs
        totA5 += topk_accuracy_from_logits(outputs, labels, k=5) * bs
        n += bs
        preds = outputs.argmax(1).cpu().numpy()
        for p, t in zip(preds, labels.cpu().numpy()):
            C[t, p] += 1
    L = totL / n; A1 = totA1 / n; A5 = totA5 / n
    ious = iou_from_confmat(C); miou = np.nanmean(ious); maxiou = np.nanmax(ious)
    return L, A1, A5, ious, miou, maxiou, C

def show_two_samples(loader, model, title="Samples"):
    model.eval()
    images, labels = next(iter(loader))
    images = images.to(DEVICE)
    with torch.no_grad():
        with autocast('cuda', enabled=(DEVICE=="cuda")):
            outputs = model(images)
    preds = outputs.argmax(1).cpu().numpy()
    labels_np = labels.numpy()
    images_cpu = images.cpu()
    mean = torch.tensor(IMAGENET_MEAN)[:,None,None]
    std  = torch.tensor(IMAGENET_STD)[:,None,None]
    idxc = idxw = None
    for i in range(len(labels_np)):
        if preds[i]==labels_np[i] and idxc is None: idxc=i
        if preds[i]!=labels_np[i] and idxw is None: idxw=i
        if idxc is not None and idxw is not None: break
    idxs = [i for i in (idxc, idxw) if i is not None]
    if not idxs:
        print("No suitable samples in this batch."); return
    for i in idxs:
        img = torch.clamp(images_cpu[i]*std+mean, 0, 1)
        plt.figure(); plt.imshow(np.transpose(img.numpy(), (1,2,0)))
        plt.title(f"{title} | pred={preds[i]} (wnid={idx_to_wnid.get(preds[i],'?')}) "
                  f"vs true={labels_np[i]} (wnid={idx_to_wnid.get(labels_np[i],'?')})")
        plt.axis("off"); plt.show()

# -------------------- Train Loop --------------------
history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[], "val_top5":[], "val_miou":[], "val_max_iou":[]}
best_epoch, best_miou = 0, -1.0

print(f"\n[RUN] epochs={CFG['EPOCHS']}, LR={CFG['LR']}, WD={CFG['WEIGHT_DECAY']}, "
      f"MixUp={CFG['MIXUP_ALPHA']} (phase-out {CFG['MIXUP_PHASEOUT_EPOCHS']}e), "
      f"LS={CFG['LABEL_SMOOTH']}")

for epoch in range(1, CFG["EPOCHS"]+1):
    print(f"\n===== Epoch {epoch}/{CFG['EPOCHS']} =====")
    t0 = time.time()
    trL, trA = train_one_epoch(model, train_loader, epoch, CFG["EPOCHS"])
    vaL, vaA1, vaA5, vaIOUs, vaMIoU, vaMaxIoU, vaC = evaluate(model, val_loader)
    scheduler.step()

    history["train_loss"].append(trL); history["train_acc"].append(trA)
    history["val_loss"].append(vaL);   history["val_acc"].append(vaA1)
    history["val_top5"].append(vaA5);  history["val_miou"].append(vaMIoU); history["val_max_iou"].append(vaMaxIoU)

    dt = time.time() - t0
    print(f"Epoch {epoch} | {dt/60:.1f}m | "
          f"Train L/A={trL:.4f}/{trA:.4f} | Val L/A1/A5={vaL:.4f}/{vaA1:.4f}/{vaA5:.4f} "
          f"| mIoU {vaMIoU:.4f} maxIoU {vaMaxIoU:.4f}")

    if vaMIoU > best_miou:
        best_miou, best_epoch = vaMIoU, epoch
        torch.save({"epoch":epoch, "model":model.state_dict(), "history":history},
                   "best_vgg_fast_imagenet.pth")
        # thêm weights-only cho lần load sau thật sạch
        torch.save(model.state_dict(), "best_vgg_weights_only.pth")
        print("  → Saved new best (by val mIoU)")

print("Best epoch:", best_epoch, "| best mIoU:", best_miou)

# -------------------- Load best & visualize --------------------
# Cách 1: load full (PyTorch >= 2.6 cần weights_only=False)
ckpt = torch.load("best_vgg_fast_imagenet.pth", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"])
history = ckpt.get("history", history)
be = ckpt["epoch"]
print("Best epoch checkpoint:", be)

# (Tuỳ chọn) Cách 2: load weights-only (nếu muốn)
# state = torch.load("best_vgg_weights_only.pth", map_location=DEVICE, weights_only=True)
# model.load_state_dict(state)

val_loss, val_acc1, val_acc5, val_ious, val_miou, val_max_iou, _ = evaluate(model, val_loader)
print(f"Validation | loss {val_loss:.4f} | Top1 {val_acc1:.4f} | Top5 {val_acc5:.4f} "
      f"| mIoU {val_miou:.4f} | maxIoU {val_max_iou:.4f}")
show_two_samples(val_loader, model, title=f"Best Epoch {be} Samples")

# -------------------- Plots --------------------
epochs = range(1, len(history["train_loss"])+1)
plt.figure(); plt.plot(epochs, history["train_loss"], label="Train Loss"); plt.plot(epochs, history["val_loss"], label="Val Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Learning Curve - Loss"); plt.legend(); plt.show()

plt.figure(); plt.plot(epochs, history["train_acc"], label="Train Top-1"); plt.plot(epochs, history["val_acc"], label="Val Top-1"); plt.plot(epochs, history["val_top5"], label="Val Top-5")
plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Learning Curve - Accuracy"); plt.legend(); plt.show()

plt.figure(); plt.plot(epochs, history["val_miou"], label="Val mIoU")
plt.xlabel("Epoch"); plt.ylabel("mIoU"); plt.title("Validation mIoU per Epoch"); plt.legend(); plt.show()


