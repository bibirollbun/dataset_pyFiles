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


import pandas as pd
import os

data_path = "/kaggle/input/itj-labs-fruit-classification-challenge"

for root, dirs, files in os.walk(data_path):
    print(root, " → ", len(files), "files")



from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dir = f"/kaggle/input/itj-labs-fruit-classification-challenge/fruit_dataset_10_classes/train"

train_dataset = ImageFolder(train_dir, transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print("Classes:", train_dataset.classes)



import torch
import torch.nn as nn
import timm

device = torch.device("cpu")

num_classes = len(train_dataset.classes)

model = timm.create_model("efficientnet_b0", pretrained=True)
model.classifier = nn.Linear(model.classifier.in_features, num_classes)

model = model.to(device)



# CPU-friendly training template (transfer learning + grad accumulation + TTA)
!pip install -q timm albumentations==1.2.1

import os, random, math, time
from glob import glob
from PIL import Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# DEVICE
device = torch.device("cpu")
torch.set_num_threads(4)   # tune this (1-8)

# ---------- Dataset wrapper (Image paths + labels) ----------
class ImageDataset(Dataset):
    def __init__(self, image_paths, labels=None, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        p = self.image_paths[idx]
        img = np.array(Image.open(p).convert("RGB"))
        if self.transform:
            img = self.transform(image=img)['image']
        label = -1 if self.labels is None else int(self.labels[idx])
        return img, label, os.path.basename(p)

# ---------- Augmentations (train/valid/test) ----------
IMG_SIZE = 224
train_aug = A.Compose([
    A.RandomResizedCrop(IMG_SIZE, IMG_SIZE, scale=(0.6, 1.0)),
    A.HorizontalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.15, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.CoarseDropout(max_holes=1, max_height=32, max_width=32, p=0.3),
    A.Normalize(),
    ToTensorV2(),
])

valid_aug = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2(),
])

tta_aug = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.HorizontalFlip(p=0.5),
    A.Normalize(),
    ToTensorV2(),
])

# ---------- Model builder (MobileNetV2 head) ----------
def build_model(num_classes, backbone='mobilenetv2_100'):
    model = timm.create_model(backbone, pretrained=True, num_classes=0, global_pool='avg')
    in_ch = model.num_features
    head = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_ch, 512),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(512),
        nn.Dropout(0.2),
        nn.Linear(512, num_classes)
    )
    model.reset_classifier(0)  # keep features
    model = nn.Sequential(model, head)
    return model

# ---------- Example: prepare file lists ----------
DATA_DIR = "/kaggle/input/itj-labs-fruit-classification-challenge"
# adjust paths below to dataset layout; example assumes train/{class}/*.jpg and test/*.jpg
train_root = "/kaggle/input/itj-labs-fruit-classification-challenge/fruit_dataset_10_classes/train"
test_root  = "/kaggle/input/itj-labs-fruit-classification-challenge/fruit_dataset_10_classes/test"
# build lists
classes = sorted([d for d in os.listdir(train_root) if os.path.isdir(os.path.join(train_root,d))])
train_paths, train_labels = [], []
for idx, cl in enumerate(classes):
    files = glob(os.path.join(train_root, cl, "*"))
    train_paths += files
    train_labels += [idx]*len(files)

print("Found classes:", classes)
num_classes = len(classes)

# ---------- Training helpers ----------
def train_one_epoch(model, loader, optimizer, criterion, device, accum_steps=1):
    model.train()
    running_loss = 0.0
    preds, targets = [], []
    optimizer.zero_grad()
    for i, (images, labels, _) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels) / accum_steps
        loss.backward()
        if (i+1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
        running_loss += loss.item() * images.size(0) * accum_steps
        preds += outputs.argmax(1).cpu().numpy().tolist()
        targets += labels.cpu().numpy().tolist()
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(targets, preds)
    return avg_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    preds, targets = [], []
    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds += outputs.argmax(1).cpu().numpy().tolist()
            targets += labels.cpu().numpy().tolist()
    avg_loss = running_loss / len(loader.dataset)
    acc = accuracy_score(targets, preds)
    return avg_loss, acc

# ---------- Cross-validation training (single fold example) ----------
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
fold = 0
for tr_idx, val_idx in skf.split(train_paths, train_labels):
    fold += 1
    print(f"\n=== Fold {fold} ===")
    tr_paths = [train_paths[i] for i in tr_idx]
    tr_labels = [train_labels[i] for i in tr_idx]
    val_paths = [train_paths[i] for i in val_idx]
    val_labels = [train_labels[i] for i in val_idx]

    train_ds = ImageDataset(tr_paths, tr_labels, transform=train_aug)
    val_ds   = ImageDataset(val_paths, val_labels, transform=valid_aug)

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=False)

    model = build_model(num_classes, backbone='mobilenetv2_100')
    model = model.to(device)

    # Freeze backbone initially
    for param in model[0].parameters(): param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)

    best_val_acc = 0.0
    EPOCHS = 8
    accum_steps = 2  # gradient accumulation to emulate larger batch

    # Train head first (few epochs)
    for epoch in range(2):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device, accum_steps=accum_steps)
        vloss, vacc = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Head Epoch {epoch+1} train_loss={loss:.4f} train_acc={acc:.4f} val_acc={vacc:.4f}")

    # Unfreeze backbone and fine-tune
    for param in model[0].parameters(): param.requires_grad = True
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    for epoch in range(EPOCHS):
        loss, acc = train_one_epoch(model, train_loader, optimizer, criterion, device, accum_steps=accum_steps)
        vloss, vacc = validate(model, val_loader, criterion, device)
        scheduler.step()
        print(f"FT Epoch {epoch+1} train_loss={loss:.4f} train_acc={acc:.4f} val_acc={vacc:.4f}")
        if vacc > best_val_acc:
            best_val_acc = vacc
            torch.save(model.state_dict(), f"best_model_fold{fold}.pth")
            print("Saved best model.")

    # For quick runs, break after one fold; remove break to run all folds
    break

# ---------- TTA inference (simple horizontal flip) ----------
def tta_predict(model, image_paths, tta_transforms=[valid_aug, tta_aug]):
    model.eval()
    preds_sum = np.zeros((len(image_paths), num_classes))
    ds_list = []
    for t in tta_transforms:
        ds = ImageDataset(image_paths, labels=None, transform=t)
        loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=2, pin_memory=False)
        with torch.no_grad():
            all_preds = []
            for images, _, names in loader:
                images = images.to(device)
                out = model(images)
                all_preds.append(torch.softmax(out, dim=1).cpu().numpy())
            all_preds = np.vstack(all_preds)
            preds_sum += all_preds
    preds_avg = preds_sum / len(tta_transforms)
    return preds_avg.argmax(1)

# Example inference on test set:
# test_images = sorted(glob(os.path.join(test_root, "*.jpg")))
# model = build_model(num_classes); model.load_state_dict(torch.load("best_model_fold1.pth")); model.to(device)
# preds = tta_predict(model, test_images)
# submission = pd.DataFrame({"id":[os.path.basename(x) for x in test_images], "label": preds})
# submission.to_csv("submission.csv", index=False)



import os

test_root = "/kaggle/input/itj-labs-fruit-classification-challenge/fruit_dataset_10_classes/test"
print(os.listdir(test_root)[:20])



# ====== SUBMISSION CELL: recursive test folders (handles class subfolders) ======

import os
from glob import glob
import numpy as np
import pandas as pd
import torch

# exact test folder you showed
test_root = "/kaggle/input/itj-labs-fruit-classification-challenge/fruit_dataset_10_classes/test"
model_path = "best_model_fold1.pth"   # change if you saved with a different name

# 1) find images recursively
exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"]
test_images = []
for e in exts:
    test_images += glob(os.path.join(test_root, "**", e), recursive=True)
test_images = sorted(test_images)
print("Detected test images:", len(test_images))
if len(test_images) == 0:
    raise ValueError(f"No images found under {test_root} (recursive).")

# 2) check model exists
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}. Train + save the model first.")

# 3) (re)build and load model
model = build_model(num_classes, backbone='mobilenetv2_100')
model.load_state_dict(torch.load(model_path, map_location=device))
model = model.to(device)
model.eval()

# 4) safer TTA predict that also returns confidence
def tta_predict_with_conf(model, image_paths, tta_transforms=[valid_aug, tta_aug], batch_size=32):
    model.eval()
    prob_sum = np.zeros((len(image_paths), num_classes), dtype=np.float32)
    from torch.utils.data import DataLoader
    for t in tta_transforms:
        ds = ImageDataset(image_paths, labels=None, transform=t)
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=False)
        idx = 0
        with torch.no_grad():
            for images, _, names in loader:
                images = images.to(device)
                out = model(images)                 # logits
                p = torch.softmax(out, dim=1).cpu().numpy()
                batch_n = p.shape[0]
                prob_sum[idx: idx+batch_n, :] += p
                idx += batch_n
    prob_avg = prob_sum / max(1, len(tta_transforms))
    top_idx = np.argmax(prob_avg, axis=1)
    top_conf = prob_avg[np.arange(len(prob_avg)), top_idx]
    return top_idx, top_conf

# 5) run inference
labels_pred, confs = tta_predict_with_conf(model, test_images, tta_transforms=[valid_aug, tta_aug], batch_size=32)

# 6) build submission DataFrame
# use file basenames without extension as id (common format). Change to os.path.basename(...) if you want extension included.
ids = [os.path.splitext(os.path.basename(p))[0] for p in test_images]
submission = pd.DataFrame({
    "id": ids,
    "label": labels_pred,
    "confidence": confs
})

# Optional: map numeric labels to class names (uncomment if you want names instead)
# submission['label_name'] = submission['label'].apply(lambda x: train_dataset.classes[x])

submission.to_csv("submission.csv", index=False)
print("Saved submission.csv with", len(submission), "rows.")
print(submission.head(10))





