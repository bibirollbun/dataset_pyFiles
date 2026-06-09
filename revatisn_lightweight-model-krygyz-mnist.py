import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torchvision.transforms as T

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42
EPOCHS = 50        
BATCH_SIZE = 256
LR = 1e-3
IMG_SIZE = 50
NUM_CLASSES = 36

def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything()

train_df = pd.read_csv("/kaggle/input/kyrgyz-language-hand-written-letter-kyrgyz-mnist/train.csv")
test_df  = pd.read_csv("/kaggle/input/kyrgyz-language-hand-written-letter-kyrgyz-mnist/test.csv")

FEATURE_COLS = [c for c in train_df.columns if c not in ["id", "label"]]

X = train_df[FEATURE_COLS].values.astype(np.float32)
y = train_df["label"].values.astype(np.int64) - 1      # 1..36 -> 0..35
X_test = test_df[FEATURE_COLS].values.astype(np.float32)

X /= 255.0
X_test /= 255.0

X = X.reshape(-1, 1, IMG_SIZE, IMG_SIZE)
X_test = X_test.reshape(-1, 1, IMG_SIZE, IMG_SIZE)

print("Train:", X.shape, "Test:", X_test.shape, "Labels:", y.min(), y.max())

# simple 80/20 split
X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

class KyrgyzDataset(Dataset):
    def __init__(self, images, labels=None, is_train=True):
        self.images = images
        self.labels = labels
        self.is_train = is_train

        self.train_transform = T.Compose([
            T.ToTensor(),
            T.RandomRotation(8),
        ])
        self.valid_transform = T.Compose([
            T.ToTensor(),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (1,50,50)
        img = np.transpose(img, (1, 2, 0))  # (50,50,1)

        if self.is_train:
            img = self.train_transform(img)
        else:
            img = self.valid_transform(img)

        if self.labels is not None:
            label = self.labels[idx]
            return img, label
        else:
            return img

train_ds = KyrgyzDataset(X_tr, y_tr, is_train=True)
val_ds   = KyrgyzDataset(X_val, y_val, is_train=False)
test_ds  = KyrgyzDataset(X_test, labels=None, is_train=False)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

class TinyCNN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 25x25

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 12x12

            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 6x6
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # 128x1x1
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

model = TinyCNN(NUM_CLASSES).to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(1, EPOCHS + 1):
    # train
    model.train()
    tr_loss = 0.0
    tr_preds, tr_targets = [], []
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        tr_loss += loss.item() * imgs.size(0)
        tr_preds.extend(logits.argmax(1).cpu().numpy())
        tr_targets.extend(labels.cpu().numpy())
    tr_loss /= len(train_ds)
    tr_acc = accuracy_score(tr_targets, tr_preds)

    # val
    model.eval()
    val_loss = 0.0
    val_preds, val_targets = [], []
    with torch.inference_mode():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)
            logits = model(imgs)
            loss = criterion(logits, labels)
            val_loss += loss.item() * imgs.size(0)
            val_preds.extend(logits.argmax(1).cpu().numpy())
            val_targets.extend(labels.cpu().numpy())
    val_loss /= len(val_ds)
    val_acc = accuracy_score(val_targets, val_preds)

    print(f"Epoch {epoch:02d} | "
          f"train_loss={tr_loss:.4f} acc={tr_acc:.4f} | "
          f"val_loss={val_loss:.4f} acc={val_acc:.4f}")

model.eval()
all_probs = []
with torch.inference_mode():
    for imgs in test_loader:
        imgs = imgs.to(DEVICE)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

all_probs = np.concatenate(all_probs, axis=0)
test_pred_labels = all_probs.argmax(axis=1) + 1   # back to 1..36

sub = pd.DataFrame({
    "id": test_df["id"],
    "label": test_pred_labels
})
sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Saved submission.csv")





