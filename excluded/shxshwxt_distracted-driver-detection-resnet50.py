# ===============================
# Imports
# ===============================
import os
import random
from glob import glob
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score



# ===============================
# Config
# ===============================
DATA_DIR = "/kaggle/input/state-farm-distracted-driver-detection"
TRAIN_DIR = os.path.join(DATA_DIR, "imgs/train")
TEST_DIR  = os.path.join(DATA_DIR, "imgs/test")

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 10
LR = 2e-4
NUM_CLASSES = 10
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)

# ===============================
# Reproducibility
# ===============================
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if DEVICE == "cuda":
    torch.cuda.manual_seed_all(SEED)



# List class folders
classes = sorted(os.listdir(TRAIN_DIR))
print("Classes:", classes)

# Count images per class
for cls in classes:
    print(cls, ":", len(os.listdir(os.path.join(TRAIN_DIR, cls))))



filepaths = []
labels = []

for idx, cls in enumerate(classes):
    imgs = glob(os.path.join(TRAIN_DIR, cls, "*.jpg"))
    filepaths.extend(imgs)
    labels.extend([idx] * len(imgs))

df = pd.DataFrame({
    "filepath": filepaths,
    "label": labels
})

print("Total images:", len(df))
df.head()



train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["label"],
    random_state=SEED
)

print("Train:", len(train_df), "Validation:", len(val_df))



class DriverDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.loc[idx, "filepath"]).convert("RGB")
        label = self.df.loc[idx, "label"]

        if self.transform:
            img = self.transform(img)

        return img, label



train_transform = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

train_ds = DriverDataset(train_df, train_transform)
val_ds   = DriverDataset(val_df, val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=2)



model = models.resnet50(pretrained=True)

# Replace final layer
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)



def train_epoch(model, loader):
    model.train()
    losses, preds, targets = [], [], []

    for imgs, labels in tqdm(loader):
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        preds.extend(outputs.argmax(1).cpu().numpy())
        targets.extend(labels.cpu().numpy())

    return np.mean(losses), accuracy_score(targets, preds)


def val_epoch(model, loader):
    model.eval()
    losses, preds, targets = [], [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader):
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            loss = criterion(outputs, labels)

            losses.append(loss.item())
            preds.extend(outputs.argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())

    return np.mean(losses), accuracy_score(targets, preds)



best_acc = 0

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss, train_acc = train_epoch(model, train_loader)
    val_loss, val_acc = val_epoch(model, val_loader)

    scheduler.step()

    print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
    print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Best model saved")



import torch.nn.functional as F

model.load_state_dict(torch.load("best_model.pth"))
model.eval()

test_images = sorted(glob(os.path.join(TEST_DIR, "*.jpg")))

rows = []

with torch.no_grad():
    for img_path in tqdm(test_images):
        img = Image.open(img_path).convert("RGB")
        img = val_transform(img).unsqueeze(0).to(DEVICE)

        outputs = model(img)
        probs = F.softmax(outputs, dim=1).cpu().numpy()[0]

        row = {
            "img": os.path.basename(img_path)
        }

        for i in range(10):
            row[f"c{i}"] = probs[i]

        rows.append(row)


# Create submission dataframe
submission = pd.DataFrame(rows)

# Ensure correct column order
submission = submission[
    ["img", "c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9"]
]

submission.to_csv("submission.csv", index=False)
submission.head()

