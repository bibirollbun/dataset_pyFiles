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


import zipfile
# ===== 1. zipãƒ•ã‚¡ã‚¤ãƒ«ã�®è§£å‡� =====
input_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
work_dir = "/kaggle/working/"

with zipfile.ZipFile(input_dir + "train.zip", "r") as zip_ref:
    zip_ref.extractall(work_dir + "train/")

with zipfile.ZipFile(input_dir + "test.zip", "r") as zip_ref:
    zip_ref.extractall(work_dir + "test/")


# å¿…è¦�ã�ªãƒ©ã‚¤ãƒ–ãƒ©ãƒªã�®ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆ
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.model_selection import train_test_split



# ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ã�®è¨­å®š
BATCH_SIZE = 64
EPOCHS = 3
IMG_SIZE = 128
LR = 0.001

# GPUã�Œä½¿ã�ˆã‚‹ã�‹ç¢ºèª�
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("âœ… ä½¿ç”¨ãƒ‡ãƒ�ã‚¤ã‚¹:", device)



# ã‚«ã‚¹ã‚¿ãƒ  Dataset ã‚¯ãƒ©ã‚¹
class CatsDogsDataset(Dataset):
    def __init__(self, filepaths, labels=None, transform=None):
        self.filepaths = filepaths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        image = Image.open(self.filepaths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        if self.labels is not None:
            label = self.labels[idx]
            return image, label
        else:
            return image



# Kaggle ã�®å…¥åŠ›ãƒ‘ã‚¹ã�«å¿œã�˜ã�¦å¤‰æ›´ã�—ã�¦ã��ã� ã�•ã�„
# ä¾‹: /kaggle/input/dogs-vs-cats/train/
train_dir = "/kaggle/working/train/train"

all_imgs = [os.path.join(train_dir, fname) for fname in os.listdir(train_dir) if fname.endswith('.jpg')]
labels = [1 if 'dog' in fname else 0 for fname in os.listdir(train_dir) if fname.endswith('.jpg')]

train_paths, val_paths, train_labels, val_labels = train_test_split(
    all_imgs, labels, test_size=0.2, stratify=labels, random_state=42
)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_dataset = CatsDogsDataset(train_paths, train_labels, transform)
val_dataset = CatsDogsDataset(val_paths, val_labels, transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)



# EfficientNet (è»¢ç§»å­¦ç¿’) ãƒ¢ãƒ‡ãƒ«ã�®æ§‹ç¯‰
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)



# æ��å¤±é–¢æ•°ã�¨æœ€é�©åŒ–æ‰‹æ³•ã�®å®šç¾©
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)



# å­¦ç¿’ãƒ«ãƒ¼ãƒ—
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

    train_acc = correct / total
    print(f"ğŸ“˜ Epoch {epoch+1}: Loss={total_loss:.4f}, Accuracy={train_acc:.4f}")



# ãƒ¢ãƒ‡ãƒ«ã�®è©•ä¾¡
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == targets).sum().item()
        total += targets.size(0)

val_acc = correct / total
print(f"âœ… æ¤œè¨¼ç²¾åº¦ (Validation Accuracy): {val_acc:.4f}")


