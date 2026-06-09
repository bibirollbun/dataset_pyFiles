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


# Cell 1: Train a pretrained-architecture model (random init) on CIFAR-10
# - Model: ResNet18 (torchvision), no pretrained weights
# - Transforms: RandomCrop + HorizontalFlip + RandomErasing
# - Saves best weights to: best_resnet18.pt
# ======================================================
import os, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torch.cuda.amp import autocast, GradScaler

# Repro + device
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print("Device:", device)

# CIFAR-10 stats
CIFAR_MEAN = (0.4914,0.4822,0.4465)
CIFAR_STD  = (0.2470,0.2435,0.2616)

# Transforms
train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    transforms.RandomErasing(p=0.25),
])
test_tfms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
])

# Load CIFAR-10
DATA_ROOT = "/kaggle/working"
_base = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=None)
idx = np.arange(len(_base))
np.random.default_rng(SEED).shuffle(idx)
VAL_SZ = 5000
train_idx, val_idx = idx[VAL_SZ:], idx[:VAL_SZ]

train_set = Subset(datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=train_tfms), train_idx.tolist())
val_set   = Subset(datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=test_tfms), val_idx.tolist())

BATCH_TRAIN, BATCH_VAL = 256, 512
train_loader = DataLoader(train_set, batch_size=BATCH_TRAIN, shuffle=True, num_workers=2, pin_memory=True)
val_loader   = DataLoader(val_set, batch_size=BATCH_VAL, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train rows: {len(train_set)} | Val rows: {len(val_set)}")

# Model: ResNet18, no pretrained weights
model = models.resnet18(weights=None)
model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # adapt for CIFAR-10
model.maxpool = nn.Identity()  # remove initial downsample
model.fc = nn.Linear(model.fc.in_features, 10)  # CIFAR-10 classes
model = model.to(device).to(memory_format=torch.channels_last)

# Optimizer, scheduler, loss, AMP
EPOCHS = 60
opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4, nesterov=True)
sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
crit = nn.CrossEntropyLoss(label_smoothing=0.05)
scaler = GradScaler()

# Training function
def run_epoch(dl, train=True):
    model.train(train)
    tot = correct = loss_sum = 0.0
    for x, y in dl:
        x = x.to(device, non_blocking=True).contiguous(memory_format=torch.channels_last)
        y = y.to(device, non_blocking=True)
        if train: opt.zero_grad(set_to_none=True)
        with autocast():
            logits = model(x)
            loss = crit(logits, y)
        if train:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        loss_sum += loss.item() * x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
        tot += x.size(0)
    return loss_sum/tot, correct/tot

# Training loop
best = (0.0, None)
for e in range(1, EPOCHS+1):
    tr_loss, tr_acc = run_epoch(train_loader, True)
    va_loss, va_acc = run_epoch(val_loader, False)
    sch.step()
    if va_acc > best[0]:
        best = (va_acc, {k: v.detach().cpu() for k, v in model.state_dict().items()})
    print(f"Ep{e:02d}/{EPOCHS} train_acc={tr_acc:.3f} val_acc={va_acc:.3f}")

# Save best weights
torch.save(best[1], "best_resnet18.pt")
print("✅ Saved best_resnet18.pt")





