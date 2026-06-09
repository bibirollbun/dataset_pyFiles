# Cell 1 — Installs (run first)
!apt-get update -qq
# Graphviz used if you want architecture rendering later (optional)
!apt-get install -y graphviz -qq
!pip install -q timm


# Cell 2 — Imports & config
import os
import math
import random
from pathlib import Path
from datetime import datetime
import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from torchvision import transforms, datasets

# mixed precision - FIXED IMPORT
from torch.cuda.amp import autocast, GradScaler  # This is the correct import

# reproducibility (rest of cell remains the same)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# Cell 3 — Hyperparameters & helpers
DATA_ROOT = "/kaggle/working/data_cifar"
os.makedirs(DATA_ROOT, exist_ok=True)

# Training settings
BATCH_SIZE = 128
EPOCHS = 200
LR = 1e-3
WEIGHT_DECAY = 1e-4
WARMUP_EPOCHS = 10
NUM_WORKERS = 0  # ⚠️ CHANGE THIS TO 0 to fix the multiprocessing issue ⚠️

# CutMix params
CUTMIX_PROB = 0.6
CUTMIX_ALPHA = 1.0

# MixUp params
MIXUP_ALPHA = 0.2
MIXUP_PROB = 0.3

# Label map
LABEL_MAP = {
    0: "airplane", 1: "automobile", 2: "bird", 3: "cat", 4: "deer",
    5: "dog", 6: "frog", 7: "horse", 8: "ship", 9: "truck"
}

def accuracy_from_logits(logits, labels):
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# Cell 4 — Enhanced Transforms & dataset loaders
try:
    from torchvision.transforms import RandAugment
    rand_augment = RandAugment(num_ops=3, magnitude=9)  # Increased from 2 to 3
    print("Using RandAugment")
except Exception:
    rand_augment = None
    print("RandAugment not available; using strong Compose")

mean = (0.4914, 0.4822, 0.4465)
std  = (0.2470, 0.2435, 0.2616)

train_transform = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(15),
]

if rand_augment is not None:
    train_transform.append(rand_augment)
else:
    # Enhanced fallback
    train_transform += [
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.1),
    ]

train_transform += [
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
]

train_transform = transforms.Compose(train_transform)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean, std),
])


# Cell 5 — Load datasets
train_full = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=train_transform)
test_official = datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=test_transform)

VAL_SIZE = 5000
TRAIN_SIZE = len(train_full) - VAL_SIZE
train_set, val_set = random_split(train_full, [TRAIN_SIZE, VAL_SIZE], generator=torch.Generator().manual_seed(SEED))

# Set num_workers=0 for all DataLoaders
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_official, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print("Train set:", len(train_set), "Val set:", len(val_set), "Official test:", len(test_official))


# Cell 6 — Enhanced augmentation utilities (CutMix + MixUp)
def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def apply_cutmix(images, targets, alpha):
    if alpha <= 0:
        return images, targets, None, None
    lam = np.random.beta(alpha, alpha)
    B = images.size(0)
    index = torch.randperm(B).to(images.device)
    bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
    images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size(-1) * images.size(-2)))
    targets2 = targets[index]
    return images, targets, targets2, lam

def apply_mixup(images, targets, alpha):
    if alpha <= 0:
        return images, targets, None, None
    lam = np.random.beta(alpha, alpha)
    B = images.size(0)
    index = torch.randperm(B).to(images.device)
    mixed_images = lam * images + (1 - lam) * images[index]
    targets_a, targets_b = targets, targets[index]
    return mixed_images, targets_a, targets_b, lam


# Cell 6.5 — MixUp utility
def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# Cell 7 — Enhanced Custom ResNet-style model
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, dropout_rate=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.dropout = nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class CustomResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, base_width=64, dropout_rate=0.0):
        super().__init__()
        self.in_planes = base_width
        self.conv1 = nn.Conv2d(3, base_width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(base_width)
        self.relu  = nn.ReLU(inplace=True)
        # residual layers
        self.layer1 = self._make_layer(block, base_width, num_blocks[0], stride=1, dropout_rate=dropout_rate)
        self.layer2 = self._make_layer(block, base_width*2, num_blocks[1], stride=2, dropout_rate=dropout_rate)
        self.layer3 = self._make_layer(block, base_width*4, num_blocks[2], stride=2, dropout_rate=dropout_rate)
        self.layer4 = self._make_layer(block, base_width*8, num_blocks[3], stride=2, dropout_rate=dropout_rate)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc  = nn.Linear(base_width*8*block.expansion, num_classes)
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_rate=0.0):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s, dropout_rate))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# Enhanced model - deeper and wider
MODEL_WIDTH = 64
model = CustomResNet(BasicBlock, [3,4,6,3], num_classes=10, base_width=MODEL_WIDTH, dropout_rate=0.1).to(device)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")


# Cell 8 — Optimizer (AdamW) and LR scheduler with warmup+cosine
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# total steps per epoch
steps_per_epoch = len(train_loader)
total_steps = EPOCHS * steps_per_epoch
warmup_steps = WARMUP_EPOCHS * steps_per_epoch

# lambda function for LR: linear warmup then cosine decay
def lr_lambda(current_step):
    if current_step < warmup_steps:
        return float(current_step) / float(max(1, warmup_steps))
    # cosine decay
    progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))

scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

scaler = GradScaler()


# Cell 9 — Enhanced train & validate loops (with CutMix, MixUp, AMP & label smoothing)
best_val_acc = 0.0
best_path = "/kaggle/working/best_model.pth"
history = {"train_loss":[], "train_acc":[], "val_loss":[], "val_acc":[]}

# Cell 9 — Enhanced Train & validate loops
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    cnt = 0
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}", leave=False)
    
    for batch in pbar:
        imgs, labels = batch
        imgs = imgs.to(device)
        labels = labels.to(device).squeeze()

        # Data augmentation selection
        aug_type = random.random()
        
        with torch.amp.autocast("cuda"):
            if aug_type < CUTMIX_PROB:
                # CutMix
                imgs, labels_a, labels_b, lam = apply_cutmix(imgs.clone(), labels, CUTMIX_ALPHA)
                outputs = model(imgs)
                loss = lam * F.cross_entropy(outputs, labels_a) + (1 - lam) * F.cross_entropy(outputs, labels_b)
                acc = (lam * (outputs.argmax(dim=1) == labels_a).float().mean().item() +
                       (1-lam) * (outputs.argmax(dim=1) == labels_b).float().mean().item())
            
            elif aug_type < CUTMIX_PROB + MIXUP_PROB:
                # MixUp
                imgs, labels_a, labels_b, lam = mixup_data(imgs, labels, MIXUP_ALPHA)
                outputs = model(imgs)
                loss = mixup_criterion(F.cross_entropy, outputs, labels_a, labels_b, lam)
                acc = (lam * (outputs.argmax(dim=1) == labels_a).float().mean().item() +
                       (1-lam) * (outputs.argmax(dim=1) == labels_b).float().mean().item())
            
            else:
                # Standard
                outputs = model(imgs)
                loss = F.cross_entropy(outputs, labels)
                acc = (outputs.argmax(dim=1) == labels).float().mean().item()

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        running_loss += loss.item() * imgs.size(0)
        running_acc += acc * imgs.size(0)
        cnt += imgs.size(0)
        pbar.set_postfix(loss=running_loss/cnt, acc=running_acc/cnt, lr=optimizer.param_groups[0]['lr'])
    train_loss = running_loss / cnt
    train_acc  = running_acc / cnt

    # Validation
    model.eval()
    val_loss = 0.0
    val_acc = 0.0
    vcnt = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(device)
            labels = labels.to(device).squeeze()
            outputs = model(imgs)
            loss = F.cross_entropy(outputs, labels)
            val_loss += loss.item() * imgs.size(0)
            val_acc  += (outputs.argmax(dim=1) == labels).float().sum().item()
            vcnt += imgs.size(0)
    val_loss /= vcnt
    val_acc  = val_acc / vcnt

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), best_path)
        print("Saved best model:", best_path)

print("Training finished. Best val acc:", best_val_acc)


# Proper TTA implementation
def evaluate_with_tta_fixed(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader):
            images, labels = images.to(device), labels.to(device)
            
            # Original
            outputs1 = model(images)
            
            # Horizontal flip
            outputs2 = model(torch.flip(images, [3]))
            
            # Average predictions
            avg_outputs = (outputs1 + outputs2) / 2
            preds = avg_outputs.argmax(dim=1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    return correct / total

# Test the fixed TTA
fixed_tta_acc = evaluate_with_tta_fixed(model, test_loader)
print(f"Fixed TTA accuracy: {fixed_tta_acc:.4f}")


# Cell 11 — Prepare Kaggle test dataset (images are png)
from PIL import Image
import glob
import pandas as pd

# Adjust this path to where your uploaded extracted test images are mounted
# Example: "/kaggle/input/convnettest/test/test" or competition input path "/kaggle/input/cifar-10/test"
KAGGLE_EXTRACTED_FOLDER = "/kaggle/input/convnettest/kaggle/working/test/test"  # <- change if needed

# quick check
files = sorted([str(p) for p in Path(KAGGLE_EXTRACTED_FOLDER).rglob("*") if p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
print("Found test images:", len(files))
assert len(files) >= 300000 or len(files) > 0, "No test images found - please set KAGGLE_EXTRACTED_FOLDER correctly"

class KaggleTestDataset(Dataset):
    def __init__(self, files, transform=None):
        self.files = files
        self.transform = transform
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, Path(path).stem  # returns id as string


kaggle_test_ds = KaggleTestDataset(files, transform=test_transform)
#kaggle_test_loader = DataLoader(kaggle_test_ds, batch_size=256, shuffle=False, num_workers=NUM_WORKERS)
kaggle_test_loader = DataLoader(kaggle_test_ds, batch_size=256, shuffle=False, num_workers=0)  # num_workers=0


# Cell 12 — Predict and save submission.csv (Kaggle format requires names)
model.load_state_dict(torch.load("/kaggle/working/best_model.pth"))
model.eval()

ids = []
preds = []
with torch.no_grad():
    for imgs, names in tqdm(kaggle_test_loader):
        imgs = imgs.to(device)
        outputs = model(imgs)
        labels_pred = outputs.argmax(dim=1).cpu().numpy().tolist()
        preds.extend(labels_pred)
        ids.extend([int(n) for n in names])

# Map numeric preds to class names
preds_names = [LABEL_MAP[int(p)] for p in preds]

# Ensure sorted by id
df = pd.DataFrame({"id": ids, "label": preds_names})
df = df.sort_values("id")
df.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission saved: /kaggle/working/submission.csv Rows:", len(df))


# Cell 13 — Quick checks
df_check = pd.read_csv("/kaggle/working/submission.csv")
print("Rows:", len(df_check))
print(df_check.head(10))
print("ID min/max:", df_check['id'].min(), df_check['id'].max())

