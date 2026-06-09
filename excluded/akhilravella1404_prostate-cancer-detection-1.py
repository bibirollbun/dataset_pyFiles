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
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class MILDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        label = int(row["isup_grade"])

        if self.transform:
            image = self.transform(image)

        return image.unsqueeze(0), label


LABEL_CSV = "/kaggle/input/prostate-cancer-grade-assessment/train.csv"
RESIZED_IMG_DIR = "/kaggle/input/panda-resized-train-data-512x512/train_images/train_images"

df = pd.read_csv(LABEL_CSV)[["image_id", "isup_grade"]]
df["image_path"] = df["image_id"].apply(lambda x: os.path.join(RESIZED_IMG_DIR, f"{x}.png"))
df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

train_df, val_df = train_test_split(df, test_size=0.2, stratify=df["isup_grade"], random_state=42)
train_df.to_csv("/kaggle/working/train.csv", index=False)
val_df.to_csv("/kaggle/working/val.csv", index=False)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

train_dataset = MILDataset("/kaggle/working/train.csv", transform=transform)
val_dataset = MILDataset("/kaggle/working/val.csv", transform=transform)

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)


from torchvision.models import resnet18, ResNet18_Weights

class SimpleMILClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(SimpleMILClassifier, self).__init__()
        self.feature_extractor = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.feature_extractor.fc = nn.Identity()

        self.attention = nn.Sequential(
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = x.squeeze(1)
        feats = self.feature_extractor(x)
        attn_weights = torch.softmax(self.attention(feats), dim=0)
        bag_rep = torch.sum(attn_weights * feats, dim=0, keepdim=True)
        out = self.classifier(bag_rep)
        return out



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleMILClassifier().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=3e-4)  # Slightly higher LR for faster convergence


def train_one_epoch(model, loader):
    model.train()
    total_loss = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Training Loss: {total_loss:.4f}")

def evaluate_full(model, loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            prob = torch.softmax(out, dim=1)
            pred = torch.argmax(prob, dim=1)
            y_true.append(y.item())
            y_pred.append(pred.item())
            y_prob.append(prob.cpu().numpy()[0])

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    roc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    cm = confusion_matrix(y_true, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Kappa Score: {kappa:.4f}")
    print(f"ROC AUC: {roc:.4f}")

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=list(range(6)))
    disp.plot(cmap='Blues')
    plt.title("Validation Confusion Matrix")
    plt.show()



for epoch in range(7):  
    print(f"Epoch {epoch+1}")
    train_one_epoch(model, train_loader)
    evaluate_full(model, val_loader)


