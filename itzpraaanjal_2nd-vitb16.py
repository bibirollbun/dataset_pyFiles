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
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import pandas as pd
from PIL import Image
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler



TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images"
CSV_PATH  = "/kaggle/input/cassava-leaf-disease-classification/train.csv"

BATCH_SIZE = 32
EPOCHS = 15
LR = 3e-5
IMG_SIZE = 224
NUM_CLASSES = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"



train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(0.2,0.2,0.2,0.1),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

valid_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])



class CassavaDataset(Dataset):
    def __init__(self, df, img_root, transform=None):
        self.df = df
        self.img_root = img_root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_root, row.image_id)
        img = Image.open(img_path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = row.label
        return img, label



df = pd.read_csv(CSV_PATH)

# (Optional) You can split into train/valid. Kaggle dataset has no valid set.
from sklearn.model_selection import train_test_split
train_df, valid_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df.label)

train_ds = CassavaDataset(train_df, TRAIN_DIR, train_transform)
valid_ds = CassavaDataset(valid_df, TRAIN_DIR, valid_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)



model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=True,
    num_classes=NUM_CLASSES
)

model.to(DEVICE)



criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

scaler = GradScaler()



def train_one_epoch(epoch):
    model.train()
    running_loss = 0
    total = 0
    correct = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast():
            outputs = model(imgs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += preds.eq(labels).sum().item()

    print(f"Epoch {epoch} Train Loss: {running_loss/len(train_loader):.4f}  Acc: {100*correct/total:.2f}%")



def validate(epoch):
    model.eval()
    running_loss = 0
    total = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in valid_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

    acc = 100 * correct / total
    print(f"Epoch {epoch} Valid Loss: {running_loss/len(valid_loader):.4f}  Acc: {acc:.2f}%")
    return acc



best_acc = 0

for epoch in range(1, EPOCHS+1):
    train_one_epoch(epoch)
    val_acc = validate(epoch)
    scheduler.step()

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_vit_b16.pth")
        print("✔ Saved New Best Model")


