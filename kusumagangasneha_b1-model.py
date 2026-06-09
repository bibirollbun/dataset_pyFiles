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
import torchvision.transforms as transforms
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import timm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


DATASET_DIR = "/kaggle/input/aptos2019-blindness-detection"
CSV_PATH = os.path.join(DATASET_DIR, "train.csv")
IMG_DIR  = os.path.join(DATASET_DIR, "train_images")

df = pd.read_csv(CSV_PATH)
df["id_code"] = df["id_code"].astype(str)
df["filepath"] = df["id_code"].apply(lambda x: f"{IMG_DIR}/{x}.png")

print("Total images:", len(df))


def cvd_class(x):
    if x <= 1:
        return 0   # Low
    elif x == 2:
        return 1   # Medium
    else:
        return 2   # High

df["cvd_risk"] = df["diagnosis"].apply(cvd_class)


train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    stratify=df["cvd_risk"],
    random_state=42
)


class APTOSDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img = Image.open(self.df.loc[idx, "filepath"]).convert("RGB")
        label = int(self.df.loc[idx, "cvd_risk"])

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)


img_size = 240   # EfficientNet-B1

train_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.3, 0.3, 0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((img_size, img_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])


train_dataset = APTOSDataset(train_df, train_transform)
val_dataset   = APTOSDataset(val_df, val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=2)


model = timm.create_model(
    "efficientnet_b1",
    pretrained=True,
    num_classes=3
)
model.to(device)


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0,1,2]),
    y=df["cvd_risk"]
)

class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


epochs = 5

for epoch in range(epochs):
    model.train()
    running_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f}")


torch.save(model.state_dict(), "/kaggle/working/efficientnet_b1_cvd.pth")
print(" Model saved!")


model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in val_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

print("Accuracy :", accuracy_score(y_true, y_pred))
print("Precision:", precision_score(y_true, y_pred, average="weighted"))
print("Recall   :", recall_score(y_true, y_pred, average="weighted"))
print("F1 Score :", f1_score(y_true, y_pred, average="weighted"))
print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))

