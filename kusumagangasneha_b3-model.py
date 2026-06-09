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
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import timm

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)


DATASET_DIR = "/kaggle/input/aptos2019-blindness-detection"

CSV_PATH = DATASET_DIR + "/train.csv"
IMG_DIR  = DATASET_DIR + "/train_images"

df = pd.read_csv(CSV_PATH)
df["id_code"] = df["id_code"].astype(str)

# full path for each image
df["filepath"] = df["id_code"].apply(lambda x: f"{IMG_DIR}/{x}.png")

print("Total training images:", len(df))
df.head()


def cvd_class(x):
    x = int(x)
    if x <= 1:  return 0  # low
    elif x == 2: return 1  # medium
    else: return 2  # high (3,4)

df["cvd_risk"] = df["diagnosis"].apply(cvd_class)
df["cvd_risk"].value_counts()


train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["cvd_risk"],
    random_state=42
)

print("Train size:", len(train_df))
print("Val size:", len(val_df))


class APTOSDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["filepath"]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = torch.tensor(int(row["cvd_risk"]))
        return img, label


img_size = 380   # EfficientNet-B4/B5 recommended (better accuracy)

train_transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

val_transform = T.Compose([
    T.Resize((img_size, img_size)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])


train_ds = APTOSDataset(train_df, transform=train_transform)
val_ds   = APTOSDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)


model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=3)
model.to(device)


from sklearn.utils.class_weight import compute_class_weight

weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0,1,2]),
    y=df["cvd_risk"]
)

weights = torch.tensor(weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


epochs = 5

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = criterion(preds, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")


torch.save(model.state_dict(), "/kaggle/working/efficientnet_b3_aptos_cvd.pth")
print("MODEL SAVED!")


model.eval()
all_preds, all_labels = [], []

with torch.no_grad():
    for imgs, labels in val_loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        preds = model(imgs)
        preds = torch.argmax(preds, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, average="weighted")
rec  = recall_score(all_labels, all_preds, average="weighted")
f1   = f1_score(all_labels, all_preds, average="weighted")
cm   = confusion_matrix(all_labels, all_preds)

print("Accuracy :", acc)
print("Precision:", prec)
print("Recall   :", rec)
print("F1 Score :", f1)
print("\nConfusion Matrix:\n", cm)

