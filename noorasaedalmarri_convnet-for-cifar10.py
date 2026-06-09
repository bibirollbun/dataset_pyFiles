# =====================
# Cell 0: Environment
# =====================

import os, math, random, time, glob, json
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision import datasets, transforms, models

from PIL import Image

SEED = 123
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device:', DEVICE)


# =====================
# Cell 1: Config
# =====================
class CFG:
    # Data & training
    img_size = 32
    epochs = 120            # Increase if you have more time (e.g., 200)
    batch_size = 128
    num_workers = 2

    # Optimizer / LR
    lr = 0.1
    weight_decay = 5e-4
    momentum = 0.9
    warmup_epochs = 5

    # Regularization
    label_smoothing = 0.1
    mixup_alpha = 0.2
    cutmix_alpha = 1.0      # set to 0 to disable CutMix
    mixup_prob = 0.7        # probability to apply Mixup/CutMix per batch

    # EMA
    use_ema = True
    ema_decay = 0.999
       
    # Paths
    work_dir = '.'
    submission_path = os.path.join(work_dir, 'submission.csv')

    # Classes for CIFAR-10 (match Kaggle label strings)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

cfg = CFG()


# =====================
# Cell 2: Transforms
# =====================

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2023, 0.1994, 0.2010)

train_transform = transforms.Compose([
    transforms.RandomCrop(cfg.img_size, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(num_ops=2, magnitude=9),  # randaugment is available in torchvision >= 0.13
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])


# =====================
# Cell 3: Data
# =====================

data_root = './data'
train_set = datasets.CIFAR10(root=data_root, train=True,  download=True, transform=train_transform)
valid_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=valid_transform)

train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True,
                          num_workers=cfg.num_workers, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=cfg.batch_size, shuffle=False,
                          num_workers=cfg.num_workers, pin_memory=True)


# =====================
# Cell 4: Model (ResNet-18) 
# =====================

class ResNet18Scratch(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        m = models.resnet18(weights=None)  # ensures random init
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        self.model = m
        self._init_weights()

    def _init_weights(self):
        # He (Kaiming) init for convs, zero bias; normal for linear
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.ones_(m.weight)
                    nn.init.zeros_(m.bias)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.model(x)

model = ResNet18Scratch(num_classes=10).to(DEVICE)


# =====================
# Cell 5: Losses (Label smoothing), MixUp/CutMix, EMA
# =====================
class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, pred, target):
        # pred: (B, C), logits; target: (B,) int
        log_probs = F.log_softmax(pred, dim=-1)
        n_classes = pred.size(-1)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=-1))

criterion = LabelSmoothingCE(cfg.label_smoothing)

# MixUp & CutMix utilities

def rand_bbox(W, H, lam):
    # for CutMix; returns bbox coords
    cut_rat = math.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    return x1, y1, x2, y2

@torch.no_grad()
def update_ema(ema_model, model, decay):
    msd = model.state_dict()
    for k, v in ema_model.state_dict().items():
        if k in msd:
            ema_model.state_dict()[k].copy_(decay * v + (1 - decay) * msd[k])

ema_model = None
if cfg.use_ema:
    ema_model = ResNet18Scratch(num_classes=10).to(DEVICE)
    ema_model.load_state_dict(model.state_dict())
    for p in ema_model.parameters():
        p.requires_grad_(False)


# =====================
# Cell 6: Optimizer & Scheduler with Warmup
# =====================
optimizer = SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
scheduler = CosineAnnealingLR(optimizer, T_max=cfg.epochs - cfg.warmup_epochs)

scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))

# Warmup helper
class WarmupScheduler:
    def __init__(self, optimizer, warmup_epochs, base_lr):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.base_lr = base_lr
        self.current_epoch = 0
    def step(self):
        self.current_epoch += 1
    def set_epoch_lr(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
            for pg in self.optimizer.param_groups:
                pg['lr'] = lr

warmup = WarmupScheduler(optimizer, cfg.warmup_epochs, cfg.lr)


# =====================
# Cell 7: Train & Validate loops
# =====================

def train_one_epoch(epoch):
    model.train()
    running_loss, running_correct, total = 0.0, 0, 0

    for images, targets in train_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)

        # Decide on MixUp/CutMix
        use_mix = random.random() < cfg.mixup_prob
        if use_mix:
            r = random.random()
            if r < 0.5 and cfg.mixup_alpha > 0:  # MixUp
                lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
                idx = torch.randperm(images.size(0)).to(DEVICE)
                mixed = lam * images + (1 - lam) * images[idx]
                targets_a, targets_b = targets, targets[idx]
                with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
                    logits = model(mixed)
                    loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
            elif cfg.cutmix_alpha > 0:            # CutMix
                lam = np.random.beta(cfg.cutmix_alpha, cfg.cutmix_alpha)
                idx = torch.randperm(images.size(0)).to(DEVICE)
                x1, y1, x2, y2 = rand_bbox(images.size(3), images.size(2), lam)
                images2 = images[idx].clone()
                images[:, :, y1:y2, x1:x2] = images2[:, :, y1:y2, x1:x2]
                lam = 1 - ((x2 - x1) * (y2 - y1) / (images.size(-1) * images.size(-2)))
                targets_a, targets_b = targets, targets[idx]
                with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
                    logits = model(images)
                    loss = lam * criterion(logits, targets_a) + (1 - lam) * criterion(logits, targets_b)
            else:
                with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
                    logits = model(images)
                    loss = criterion(logits, targets)
        else:
            with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
                logits = model(images)
                loss = criterion(logits, targets)

        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if cfg.use_ema:
            update_ema(ema_model, model, cfg.ema_decay)

        # Metrics
        with torch.no_grad():
            preds = logits.argmax(1)
            running_correct += (preds == targets).sum().item()
            total += targets.size(0)
            running_loss += loss.item() * targets.size(0)

    # Scheduler step
    if epoch < cfg.warmup_epochs:
        warmup.set_epoch_lr(epoch)
    else:
        scheduler.step()

    train_acc = running_correct / total
    train_loss = running_loss / total
    return train_loss, train_acc


@torch.no_grad()
def evaluate(use_ema=True):
    net = ema_model if (cfg.use_ema and use_ema) else model
    net.eval()
    total, correct, running_loss = 0, 0, 0.0

    for images, targets in valid_loader:
        images, targets = images.to(DEVICE), targets.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
            logits = net(images)
            loss = criterion(logits, targets)
        preds = logits.argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        running_loss += loss.item() * targets.size(0)

    return running_loss/total, correct/total


# =====================
# Cell 8: Training loop
# =====================
best_acc = 0.0
best_path = os.path.join(cfg.work_dir, 'best_model.pth')

for epoch in range(cfg.epochs):
    t0 = time.time()
    tr_loss, tr_acc = train_one_epoch(epoch)
    val_loss, val_acc = evaluate(use_ema=True)
    dt = time.time() - t0

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save((ema_model if cfg.use_ema else model).state_dict(), best_path)

    # Print progress
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1:03d}/{cfg.epochs} | lr {current_lr:.5f} | "
          f"train_loss {tr_loss:.4f} acc {tr_acc:.4f} | val_loss {val_loss:.4f} acc {val_acc:.4f} | {dt:.1f}s")

print('Best Val Acc:', best_acc)


# =====================
# Cell 9: Load best model and evaluate (final estimate of Kaggle score)
# =====================
if os.path.exists(best_path):
    if cfg.use_ema:
        ema_model.load_state_dict(torch.load(best_path, map_location='cpu'))
    else:
        model.load_state_dict(torch.load(best_path, map_location='cpu'))

final_loss, final_acc = evaluate(use_ema=True)
print(f"Final (official CIFAR-10 test) — Loss: {final_loss:.4f}, Acc: {final_acc:.4f}")


# =====================
# Cell 10: Prepare Kaggle test set (from test.7z) — Decompression (Kaggle Only)
# =====================

!apt -y install libarchive-dev
!pip install -q libarchive
import libarchive.public
cnt = 0
for entry in libarchive.public.file_pour('/kaggle/input/cifar-10/test.7z'):
     cnt += 1
     if cnt % 1000 == 0:
         print(cnt)
import glob
print('Num files in test/:', len(glob.glob('test/*')))


# =====================
# Cell 11: Kaggle test dataset & submission writer
# =====================
class KaggleCIFAR10Test(torch.utils.data.Dataset):
    def __init__(self, root='test', transform=None):
        self.root = Path(root)
        self.files = sorted(self.root.glob('*.png'), key=lambda p: int(p.stem))
        self.transform = transform if transform is not None else valid_transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        p = self.files[idx]
        img = Image.open(p).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, int(p.stem)

@torch.no_grad()
def predict_test_and_write_submission(model_to_use=None, out_csv='submission.csv'):
    net = model_to_use if model_to_use is not None else (ema_model if cfg.use_ema else model)
    net.eval()

    test_ds = KaggleCIFAR10Test(root='test', transform=valid_transform)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False,
                             num_workers=cfg.num_workers, pin_memory=True)

    ids = []
    labels = []
    for images, idnums in test_loader:
        images = images.to(DEVICE)
        with torch.cuda.amp.autocast(enabled=(DEVICE.type=='cuda')):
            logits = net(images)
            preds = logits.argmax(1).detach().cpu().numpy()
        ids.extend(idnums.numpy().tolist())
        labels.extend([cfg.classes[p] for p in preds])

    # Write submission
    import pandas as pd
    df = pd.DataFrame({'id': ids, 'label': labels})
    df = df.sort_values('id')
    df.to_csv(out_csv, index=False)
    print('Wrote:', out_csv, 'with shape', df.shape)




predict_test_and_write_submission(out_csv=cfg.submission_path)


!ls -lh submission.csv
!head submission.csv

