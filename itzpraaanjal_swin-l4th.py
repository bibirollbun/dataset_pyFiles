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


import os
import cv2
import timm
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

import torch
import torch.nn as nn
from torch.cuda import amp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ======================
# CONFIG
# ======================
class CFG:
    seed = 42
    model_name = "swin_large_patch4_window12_384"
    img_size = 384
    batch_size = 4             # works on T4 (16GB)
    grad_accum = 2             # effective batch 8
    epochs = 8
    lr = 2e-5
    num_classes = 5
    num_workers = 2
    train_csv = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
    images = "/kaggle/input/cassava-leaf-disease-classification/train_images/"


# ======================
# SEED
# ======================
def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything()


# ======================
# DATASET
# ======================
class CassavaDS(Dataset):
    def __init__(self, df, tfm=None):
        self.df = df
        self.tfm = tfm
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(CFG.images + row.image_id)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.tfm:
            img = self.tfm(img)
        
        return img, row.label


# ======================
# AUGMENTATIONS
# ======================
train_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.RandAugment(2, 8),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])

valid_tfms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((CFG.img_size, CFG.img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),
])


# ======================
# MODEL
# ======================
def build_model():
    model = timm.create_model(
        CFG.model_name,
        pretrained=True,
        num_classes=CFG.num_classes
    )
    return model


# ======================
# TRAIN LOOP
# ======================
device = "cuda"


def train_one_epoch(model, loader, optimizer, criterion, scaler):
    model.train()
    correct, total = 0, 0
    loss_sum = 0
    
    optimizer.zero_grad()
    
    for step, (x, y) in enumerate(tqdm(loader)):
        x, y = x.to(device), y.to(device)

        with amp.autocast():
            out = model(x)
            loss = criterion(out, y) / CFG.grad_accum

        scaler.scale(loss).backward()

        if (step + 1) % CFG.grad_accum == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        loss_sum += loss.item() * CFG.grad_accum * x.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += x.size(0)

    return loss_sum / total, correct / total


def valid_one_epoch(model, loader, criterion):
    model.eval()
    correct, total = 0, 0
    loss_sum = 0
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)

            loss_sum += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            total += x.size(0)

    return loss_sum / total, correct / total


# ======================
# TRAINING SETUP
# ======================
df = pd.read_csv(CFG.train_csv)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(skf.split(df, df.label))

train_ds = CassavaDS(df.iloc[train_idx], train_tfms)
valid_ds = CassavaDS(df.iloc[val_idx], valid_tfms)

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers)
valid_loader = DataLoader(valid_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers)

model = build_model().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
scaler = amp.GradScaler()

best_acc = 0


# ======================
# TRAINING LOOP
# ======================
for epoch in range(1, CFG.epochs + 1):
    tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
    val_loss, val_acc = valid_one_epoch(model, valid_loader, criterion)
    
    print(f"\nEPOCH {epoch}")
    print(f"Train     Loss: {tr_loss:.4f} | Acc: {tr_acc*100:.2f}%")
    print(f"Valid     Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")
    
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_swin_large.pth")
        print("✔ Saved New Best Model (Swin-L)")



import matplotlib.pyplot as plt

train_loss = [0.4238, 0.3109, 0.2575, 0.2022, 0.1534, 0.1191, 0.0930, 0.0741]
valid_loss = [0.3343, 0.3363, 0.3333, 0.3647, 0.4191, 0.4333, 0.4907, 0.5985]

train_acc = [85.52, 89.39, 91.00, 92.93, 94.81, 96.02, 96.80, 97.42]
valid_acc = [88.90, 88.86, 89.04, 88.01, 88.29, 88.60, 88.79, 87.76]

epochs = list(range(1, 9))

plt.figure(figsize=(10,5))
plt.plot(epochs, train_loss, label="Train Loss")
plt.plot(epochs, valid_loss, label="Valid Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,5))
plt.plot(epochs, train_acc, label="Train Accuracy")
plt.plot(epochs, valid_acc, label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy (%)")
plt.title("Training vs Validation Accuracy")
plt.legend()
plt.grid()
plt.show()





