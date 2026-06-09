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
import zipfile
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from tqdm import tqdm



# ===== 解凍処理 =====
data_dir = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"
out_dir = "/kaggle/working/"

for archive in ["train.zip", "test.zip"]:
    with zipfile.ZipFile(os.path.join(data_dir, archive), "r") as zip_ref:
        zip_ref.extractall(os.path.join(out_dir, archive.split('.')[0]))

train_root = os.path.join(out_dir, "train")
test_root = os.path.join(out_dir, "test")



# ===== パラメータ =====
BATCH_SIZE = 64
EPOCHS = 3
IMG_SIZE = 128
LR = 0.001

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(" Device:", device)


# ===== Datasetクラス =====
class CatsDogsDataset(Dataset):
    def __init__(self, paths, labels=None, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img


# ===== データ準備 =====
train_dir = os.path.join(train_root, "train")
test_dir = os.path.join(test_root, "test")

file_list = os.listdir(train_dir)
paths = [os.path.join(train_dir, fname) for fname in file_list]
labels = [1 if "dog" in fname else 0 for fname in file_list]

X_train, X_val, y_train, y_val = train_test_split(paths, labels, test_size=0.1, random_state=42)

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
])

train_loader = DataLoader(CatsDogsDataset(X_train, y_train, transform), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(CatsDogsDataset(X_val, y_val, transform), batch_size=BATCH_SIZE)


# ===== モデルの構築（MobileNetV2） =====
model = models.mobilenet_v2(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# ===== 学習 =====
for ep in range(EPOCHS):
    model.train()
    total_loss = 0.0
    loop = tqdm(train_loader, desc=f"[Epoch {ep+1}/{EPOCHS}]", leave=False)
    for imgs, lbls in loop:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        loop.set_postfix(loss=total_loss / (loop.n + 1e-9))
    print(f"[Epoch {ep+1}] Training Loss: {total_loss/len(train_loader):.4f}")
print(" Training complete!")


# ===== 推論 =====
test_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".jpg")], key=lambda x: int(x.split('.')[0]))
test_paths = [os.path.join(test_dir, fname) for fname in test_files]

test_loader = DataLoader(CatsDogsDataset(test_paths, transform=transform), batch_size=BATCH_SIZE)

model.eval()
all_preds = []
with torch.no_grad():
    loop = tqdm(test_loader, desc=" Predicting", leave=False)
    for batch in loop:
        batch = batch.to(device)
        probs = torch.softmax(model(batch), dim=1)[:, 1]
        all_preds.extend(probs.cpu().numpy())


# ===== 提出用CSV作成 =====
submission = pd.DataFrame({
    "id": [int(fname.split(".")[0]) for fname in test_files],
    "label": all_preds
})
submission.to_csv("submission.csv", index=False)
print(" submission.csv saved!")


# ===== 評価 =====
from sklearn.metrics import accuracy_score, f1_score

true_vals, pred_vals = [], []
with torch.no_grad():
    for imgs, lbls in val_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        preds = torch.argmax(model(imgs), dim=1)
        true_vals.extend(lbls.cpu().numpy())
        pred_vals.extend(preds.cpu().numpy())

acc = accuracy_score(true_vals, pred_vals)
f1 = f1_score(true_vals, pred_vals)
print(f" Validation Accuracy: {acc:.4f}, F1 Score: {f1:.4f}")


# ===== モデル保存とファイル確認 =====
torch.save(model.state_dict(), "model.pth")
print(" model.pth saved")

print(" Working Directory Files:")
for f in os.listdir(out_dir):
    print("-", f)

