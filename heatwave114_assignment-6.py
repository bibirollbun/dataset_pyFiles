!pip -q install timm==1.0.3 --no-warn-conflicts
!apt install libarchive-dev
!pip install libarchive


import os, sys, math, time, random, glob, csv
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt
import libarchive.public  # for Kaggle test.7z

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 1337
DATA_DIR = "./data"

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True

seed_everything()
print("Device:", DEVICE)


class CutOut(object):
    def __init__(self, n_holes=1, length=8):
        self.n_holes, self.length = n_holes, length
    def __call__(self, img):  # img: Tensor CxHxW
        h, w = img.shape[1], img.shape[2]
        mask = torch.ones((h, w), dtype=img.dtype, device=img.device)
        for _ in range(self.n_holes):
            y = np.random.randint(h); x = np.random.randint(w)
            y1 = np.clip(y - self.length//2, 0, h); y2 = np.clip(y + self.length//2, 0, h)
            x1 = np.clip(x - self.length//2, 0, w); x2 = np.clip(x + self.length//2, 0, w)
            mask[int(y1):int(y2), int(x1):int(x2)] = 0.
        return img * mask.expand_as(img)

MEAN = (0.4914, 0.4822, 0.4465)
STD  = (0.2023, 0.1994, 0.2010)

train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    CutOut(n_holes=1, length=8),
])

val_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    bs, _, H, W = x.size()
    index = torch.randperm(bs)
    y_a, y_b = y, y[index]
    x1, y1, x2, y2 = rand_bbox(W, H, lam)  # you already have rand_bbox(W,H,lam)
    x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (W * H))  # area-corrected
    return x, y_a, y_b, lam

def collate_cutmix_mixup(batch,
                         p_cutmix=0.5,       # turn this up/down
                         p_mixup=0.0,        # set >0 to enable mixup too
                         alpha_cutmix=1.0,
                         alpha_mixup=0.2):
    imgs = torch.stack([b[0] for b in batch], 0)
    targets = torch.tensor([b[1] for b in batch], dtype=torch.long)

    u = np.random.rand()
    if u < p_cutmix:
        imgs, y1, y2, lam = cutmix_data(imgs, targets, alpha_cutmix)
        return imgs, (y1, y2, lam)
    elif u < p_cutmix + p_mixup:
        imgs, y1, y2, lam = mixup_data(imgs, targets, alpha_mixup)
        return imgs, (y1, y2, lam)
    else:
        # no mixing this batch
        return imgs, (targets, None, None)


def rand_bbox(W, H, lam):
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w//2, 0, W); y1 = np.clip(cy - cut_h//2, 0, H)
    x2 = np.clip(cx + cut_w//2, 0, W); y2 = np.clip(cy + cut_h//2, 0, H)
    return x1, y1, x2, y2



train_set = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tfms)
val_set   = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=val_tfms)

CLASSES = train_set.classes
print("Classes:", CLASSES)

BATCH_SIZE = 256
NUM_WORKERS = 2
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
train_set = datasets.CIFAR10(DATA_DIR, train=True,  download=True, transform=train_tfms)
val_set   = datasets.CIFAR10(DATA_DIR, train=False, download=True, transform=val_tfms)

train_loader = DataLoader(
    train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS,
    pin_memory=True, drop_last=True,
    collate_fn=lambda b: collate_cutmix_mixup(b, p_cutmix=0.5, p_mixup=0.0)  # CutMix on
)
val_loader = DataLoader(
    val_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
)



class DropPath(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0.0:
            return x
        keep = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        rand = torch.rand(shape, dtype=x.dtype, device=x.device)
        return x * (rand < keep) / keep

class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(c, c//r, 1), nn.GELU(),
            nn.Conv2d(c//r, c, 1), nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.fc(self.pool(x))

class ConvNeXtBlock(nn.Module):
    def __init__(self, c, exp=4, drop_path=0.0):
        super().__init__()
        self.dw = nn.Conv2d(c, c, kernel_size=7, padding=3, groups=c)
        self.gn = nn.GroupNorm(1, c)   # channel LN
        hidden = int(exp * c)
        self.pw1 = nn.Conv2d(c, hidden, 1)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(hidden, c, 1)
        self.se  = SE(c, r=8)
        self.drop = DropPath(drop_path)

    def forward(self, x):
        residual = x
        x = self.dw(x)
        x = self.gn(x)
        x = self.pw1(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.se(x)
        x = residual + self.drop(x)
        return x

class DownsampleResidual(nn.Module):
    def __init__(self, in_ch, out_ch, k=2, norm='gn'):
        super().__init__()
        self.main = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=k, padding=0, bias=False)
        self.skip = nn.Sequential(
            nn.AvgPool2d(kernel_size=k, stride=k),
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        )
        if norm == 'gn':
            self.norm = nn.GroupNorm(1, out_ch)     # matches your ConvNeXt-style normalization
        else:
            self.norm = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return self.norm(self.main(x) + self.skip(x))


class CifarNeXt(nn.Module):
    def __init__(self, dims=(64, 128, 256, 384), depths=(2,3,4,2),
                 num_classes=10, drop_path_rate=0.1):
        super().__init__()
        # Stem (32 -> 16)
        self.stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(dims[0], dims[0], kernel_size=3, stride=1, padding=1)
        )

        stages = []
        total_blocks = sum(depths)
        dp_rates = torch.linspace(0, drop_path_rate, total_blocks).tolist()
        i = 0
        in_c = dims[0]

        for s, (c, d) in enumerate(zip(dims, depths)):
            if s > 0:
                # swap in the residual downsample
                stages.append(DownsampleResidual(in_c, c, k=2, norm='gn'))
                in_c = c
            blocks = []
            for _ in range(d):
                blocks.append(ConvNeXtBlock(c, exp=4, drop_path=dp_rates[i]))
                i += 1
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)

        # --- NEW: global long skip from stem -> final channels/spatial (16x16 -> 2x2)
        # 16 -> 2 is a factor of 8, so AvgPool2d(8,8) does the spatial match.
        self.stem2last = nn.Sequential(
            nn.AvgPool2d(kernel_size=8, stride=8),        # 16x16 -> 2x2
            nn.Conv2d(dims[0], dims[-1], kernel_size=1, bias=False),
            nn.GroupNorm(1, dims[-1])
        )

        self.head = nn.Sequential(
            nn.GroupNorm(1, dims[-1]),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(dims[-1], num_classes)
        )

        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        s = self.stem(x)        # 16x16, dims[0]
        y = self.stages(s)      # ends at 2x2, dims[-1]
        y = y + self.stem2last(s)   # <-- global long skip
        y = self.head(y)
        return y



def cifarnext_tiny(num_classes=10):
    # tuned for CIFAR-10 speed/quality
    return CifarNeXt(dims=(64,128,256,384), depths=(2,3,4,2), num_classes=num_classes, drop_path_rate=0.15)

model = cifarnext_tiny(num_classes=len(CLASSES)).to(DEVICE)
print("Params (M):", round(sum(p.numel() for p in model.parameters())/1e6, 3))



# after creating model:
ema_decay = 0.999
ema = {n: p.detach().clone() for n,p in model.named_parameters() if p.requires_grad}

def ema_update(model, ema, decay=ema_decay):
    with torch.no_grad():
        for (n, p) in model.named_parameters():
            if p.requires_grad:
                ema[n].mul_(decay).add_(p.detach(), alpha=1.0 - decay)

# in training loop, right after optimizer.step():
ema_update(model, ema)

@torch.no_grad()
def load_ema(model, ema):
    for (n, p) in model.named_parameters():
        if p.requires_grad:
            p.copy_(ema[n])


EPOCHS = 1000          
BASE_LR = 3e-3
WARMUP_EPOCHS = 3
WEIGHT_DECAY = 6e-4
LABEL_SMOOTH = 0.05


class LabelSmoothingLoss(nn.Module):
    def __init__(self, n_classes, smoothing=0.0):
        super().__init__()
        self.conf = 1.0 - smoothing
        self.smoothing = smoothing
        self.n = n_classes
    def forward(self, logits, target):
        logp = F.log_softmax(logits, dim=1)
        with torch.no_grad():
            true = torch.full_like(logp, self.smoothing/(self.n-1))
            true.scatter_(1, target.unsqueeze(1), self.conf)
        return torch.mean(torch.sum(-true * logp, dim=1))

criterion = LabelSmoothingLoss(len(CLASSES), LABEL_SMOOTH)
optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=WEIGHT_DECAY)

def cosine_warmup_factor(epoch):
    if epoch < WARMUP_EPOCHS:
        return (epoch + 1) / max(1, WARMUP_EPOCHS)
    progress = (epoch - WARMUP_EPOCHS) / max(1, EPOCHS - WARMUP_EPOCHS)
    return 0.5 * (1 + math.cos(math.pi * progress))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=cosine_warmup_factor)



best_acc = 0.0
best_path = "/kaggle/working/best_model.pth"
best_path_final = "/kaggle/working/best_model_final.pth"

def train_one_epoch(epoch):
    model.train()
    total, correct, loss_sum = 0, 0, 0.0

    for x, ytuple in train_loader:
        # ytuple = (y_main, y_aux_or_None, lam_or_None)
        y, y2, lam = ytuple
        x = x.to(DEVICE, non_blocking=True)
        y = y.to(DEVICE, non_blocking=True)
        y2 = y2.to(DEVICE, non_blocking=True) if y2 is not None else None

        logits = model(x)
        if y2 is not None:  # mixed batch (CutMix or MixUp)
            loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y2)
        else:
            loss = criterion(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        preds = logits.argmax(1)
        # accuracy judged against primary y; this is standard practice
        correct += (preds == y).sum().item()
        total += x.size(0)

    scheduler.step()
    return loss_sum/total, correct/total


@torch.no_grad()
def validate():
    model.eval()
    total, correct, loss_sum = 0, 0, 0.0
    for x, y in val_loader:
        x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        total += x.size(0)
    return loss_sum/total, correct/total

for epoch in range(1, EPOCHS+1):
    tr_loss, tr_acc = train_one_epoch(epoch)
    va_loss, va_acc = validate()

    # EXACT print you requested
    print(f"Epoch {epoch:03d}/{EPOCHS} | val loss {va_loss:.4f} acc {va_acc:.4f}")

    # EXACT checkpoint behavior
    if va_acc > best_acc:
        best_acc = va_acc
        torch.save({"model": model.state_dict(), "classes": CLASSES}, best_path)
    torch.save({"model": model.state_dict(), "classes": CLASSES}, best_path_final)

print("Best val acc:", best_acc)


# model = torch.load("/kaggle/input/resnet18-ass4/pytorch/default/1/best_model.pth", map_location=DEVICE)
checkpoint = torch.load("/kaggle/working/best_model.pth", map_location=DEVICE)

# Rebuild model
def resnet18_cifar10(*args, **kwargs):  # stub kept only to match your snippet name if reused elsewhere
    raise NotImplementedError("We use CifarNeXt instead of ResNet.")

model = cifarnext_tiny(num_classes=len(checkpoint["classes"]))
model.load_state_dict(checkpoint["model"])
model.to(DEVICE)
model.eval()


all_preds, all_targets = [], []

with torch.no_grad():
    for x, y in val_loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(1).cpu().numpy()
        all_preds.append(preds)
        all_targets.append(y.numpy())

y_true = np.concatenate(all_targets)
y_pred = np.concatenate(all_preds)

acc = accuracy_score(y_true, y_pred)
prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average="macro", zero_division=0
)
prec_weighted, rec_weighted, f1_weighted, _ = precision_recall_fscore_support(
    y_true, y_pred, average="weighted", zero_division=0
)

print(f"Accuracy: {acc:.4f}")
print(f"Macro  - Precision: {prec_macro:.4f}  Recall: {rec_macro:.4f}  F1: {f1_macro:.4f}")
print(f"Weighted - Precision: {prec_weighted:.4f}  Recall: {rec_weighted:.4f}  F1: {f1_weighted:.4f}")

print("\nPer-class report:")
print(classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0))

cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASSES)))
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
disp.plot(ax=ax[0], xticks_rotation=45, colorbar=False)
ax[0].set_title("Confusion Matrix (Counts)")

cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True)
disp_norm = ConfusionMatrixDisplay(confusion_matrix=cm_norm, display_labels=CLASSES)
disp_norm.plot(ax=ax[1], xticks_rotation=45, colorbar=False, values_format=".2f")
ax[1].set_title("Confusion Matrix (Row-Normalized)")

plt.tight_layout(); plt.show()


cnt = 0

for entry in libarchive.public.file_pour('/kaggle/input/cifar-10/test.7z'):
    cnt += 1
    if cnt % 75000 == 0: print(cnt)


test_tfms = val_tfms

def load_image(path):
    return test_tfms(Image.open(path).convert("RGB"))

# IMPORTANT: iterate ids in numeric order
test_dir = "test"
test_ids = sorted([int(os.path.splitext(os.path.basename(p))[0]) for p in glob.glob(os.path.join(test_dir, "*.png"))])
print("Test images:", len(test_ids))

BATCH = 512
pred_labels = []

with torch.no_grad():
    batch = []
    current_ids = []
    for i in test_ids:
        x = load_image(os.path.join(test_dir, f"{i}.png"))
        batch.append(x)
        current_ids.append(i)
        if len(batch) == BATCH:
            xb = torch.stack(batch).to(DEVICE)
            probs = model(xb).softmax(1)
            preds = probs.argmax(1).cpu().numpy()
            pred_labels += [(j, CLASSES[p]) for j,p in zip(current_ids, preds)]
            batch, current_ids = [], []
    if batch:
        xb = torch.stack(batch).to(DEVICE)
        probs = model(xb).softmax(1)
        preds = probs.argmax(1).cpu().numpy()
        pred_labels += [(j, CLASSES[p]) for j,p in zip(current_ids, preds)]


with open("submission.csv","w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["id","label"])
    for i,lab in sorted(pred_labels, key=lambda t:t[0]):
        w.writerow([i, lab])
print("Wrote submission.csv")

