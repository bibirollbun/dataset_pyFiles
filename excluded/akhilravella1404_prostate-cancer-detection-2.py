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


!pip install timm --quiet  # Hugely helpful for loading DeiT/Vision Transformers

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import timm  # for transformer models like DeiT



LABEL_CSV = "/kaggle/input/prostate-cancer-grade-assessment/train.csv"
RESIZED_IMG_DIR = "/kaggle/input/panda-resized-train-data-512x512/train_images/train_images"

df = pd.read_csv(LABEL_CSV)[["image_id", "isup_grade"]]
df["image_path"] = df["image_id"].apply(lambda x: os.path.join(RESIZED_IMG_DIR, f"{x}.png"))
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["isup_grade"], random_state=42)
train_df.to_csv("/kaggle/working/train.csv", index=False)
val_df.to_csv("/kaggle/working/val.csv", index=False)


class ProstateDataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.data = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["isup_grade"])
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = ProstateDataset("/kaggle/working/train.csv", transform)
val_dataset = ProstateDataset("/kaggle/working/val.csv", transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)


class ChatCADTransformer(nn.Module):
    def __init__(self, num_classes=6):
        super(ChatCADTransformer, self).__init__()
        self.backbone = timm.create_model("deit_small_patch16_224", pretrained=True)
        self.backbone.head = nn.Identity()  # remove original head
        self.fc = nn.Sequential(
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        return self.fc(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ChatCADTransformer().to(device)


criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=2e-4)

def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss = criterion(out, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Training Loss: {total_loss:.2f}")

def evaluate(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            prob = torch.softmax(out, dim=1)
            pred = torch.argmax(prob, dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            y_prob.extend(prob.cpu().numpy())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    cm = confusion_matrix(y_true, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Kappa: {kappa:.4f}")
    print(f"ROC AUC: {roc:.4f}")
    ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(6))).plot(cmap='Blues')
    plt.title("Validation Confusion Matrix")
    plt.show()


for epoch in range(5):
    print(f"\nEpoch {epoch+1}")
    train_one_epoch(model, train_loader)
    evaluate(model, val_loader)

