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


# Install dependencies if on Colab:
# !pip install timm pandas scikit-learn matplotlib tqdm

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---- Paths ----
# Set your Kaggle dataset extraction directory here
IMG_DIR = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train"
CSV_PATH = "/kaggle/input/siim-isic-melanoma-classification/train.csv"

# ---- Data Preparation ----
class ISICDatasetSSL(Dataset):
    def __init__(self, img_dir, csv_path=None, transform=None, labeled=False):
        self.img_dir = img_dir
        self.transform = transform
        if labeled:
            df = pd.read_csv(csv_path)
            self.images = df['image_name'].tolist()
            self.labels = df['target'].tolist()
        else:
            self.images = [img_name[:-4] for img_name in os.listdir(self.img_dir) if img_name.endswith('.jpg')]
            self.labels = None
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx] + ".jpg")
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.labels is not None:
            return img, self.labels[idx]
        return img

# Transforms
ssl_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])
eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Datasets and DataLoaders
ssl_dataset = ISICDatasetSSL(IMG_DIR, transform=ssl_transform, labeled=False)
N = 2000
ssl_dataset.images = ssl_dataset.images[:N]
# ssl_loader = DataLoader(ssl_dataset, batch_size=64, shuffle=True, num_workers=2)
ssl_loader = DataLoader(ssl_dataset, batch_size=16, shuffle=True, num_workers=0)

# For downstream: train/test split
df = pd.read_csv(CSV_PATH)
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['target'], random_state=42)
train_dataset = ISICDatasetSSL(IMG_DIR, csv_path=CSV_PATH, transform=eval_transform, labeled=True)
val_dataset   = ISICDatasetSSL(IMG_DIR, csv_path=CSV_PATH, transform=eval_transform, labeled=True)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=32)

# ---- SSL Pretrain with DINO-like loss (Simplified Student-Teacher) ----
device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = timm.create_model('efficientnet_b2', pretrained=False, num_classes=0).to(device)
head = nn.Linear(1408, 256).to(device)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(head.parameters()), lr=1e-4)

def dino_loss(student_out):
    # Simple version: encourage outputs to be uniform, as in entropy maximization
    p = torch.nn.functional.softmax(student_out, dim=-1)
    return -(p * torch.log(p + 1e-7)).sum(dim=1).mean()

ssl_train_losses = []      # stores avg SSL loss per epoch
downstream_aucs = []       # stores AUC after every epoch

# print("Starting SSL pretraining...")
# for epoch in range(10):
#     encoder.train()
#     head.train()
#     running_loss = 0.0
#     for imgs in tqdm(ssl_loader):
#         imgs = imgs.to(device)
#         # Student output
#         feats = encoder(imgs)
#         student_out = head(feats)
#         loss = dino_loss(student_out)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         running_loss += loss.item()
#     print(f"Epoch {epoch+1} SSL Loss: {running_loss/len(ssl_loader):.4f}")

# # ---- Feature extraction ----
# def extract_features(dataloader, encoder, device='cuda'):
#     encoder.eval()
#     all_feats = []
#     all_labels = []
#     with torch.no_grad():
#         for items in tqdm(dataloader):
#             imgs, labels = items
#             imgs = imgs.to(device)
#             feats = encoder(imgs)
#             all_feats.append(feats.cpu())
#             all_labels.extend(labels)
#     return torch.cat(all_feats), np.array(all_labels)

# train_feats, train_labels = extract_features(train_loader, encoder, device)
# val_feats, val_labels     = extract_features(val_loader, encoder, device)

# # ---- Downstream classifier (Logistic Regression) ----
# from sklearn.linear_model import LogisticRegression
# clf = LogisticRegression(max_iter=1000)
# clf.fit(train_feats, train_labels)
# val_preds_proba = clf.predict_proba(val_feats)[:,1]
# val_preds = clf.predict(val_feats)

# # ---- Evaluation Metrics and Plots ----
# auc_score = roc_auc_score(val_labels, val_preds_proba)
# fpr, tpr, _ = roc_curve(val_labels, val_preds_proba)
# cm = confusion_matrix(val_labels, val_preds, normalize='true')
# accuracy = accuracy_score(val_labels, val_preds)
# sensitivity = cm[1,1]  # Recall/True positive rate for malignant
# specificity = cm[0,0]  # True negative rate for benign

# print(f"AUC: {auc_score:.4f} | Accuracy: {accuracy:.4f}")
# print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")

# # Plots: Losses, AUC, ROC, Sensitivity vs Specificity, Confusion Matrix
# epochs = np.arange(1, 11)
# dummy_train_losses = np.linspace(1.3, 0.95, 10)    # Replace with real values
# dummy_val_losses = np.linspace(1.3, 1.05, 10)      # Replace with real values
# dummy_aucs = np.linspace(0.5, auc_score, 10)       # Replace with real AUC per epoch

# plt.figure(figsize=(10,4))
# plt.subplot(1,2,1)
# plt.plot(epochs, dummy_train_losses, label='Train')
# plt.plot(epochs, dummy_val_losses, label='Val')
# plt.title("Losses")
# plt.legend()
# plt.subplot(1,2,2)
# plt.plot(epochs, dummy_aucs, label='AUC')
# plt.title(f"AUC (Best: {auc_score:.4f})")
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(fpr, tpr, label=f'AUC = {auc_score:.4f}')
# plt.plot([0, 1], [0, 1], 'k--')
# plt.title('ROC Curve')
# plt.legend()
# plt.show()

# plt.figure()
# plt.plot(epochs, np.clip(np.linspace(1.0, sensitivity, 10), 0, 1), label="Sensitivity")
# plt.plot(epochs, np.clip(np.linspace(0.0, specificity, 10), 0, 1), label="Specificity")
# plt.title("Sensitivity vs Specificity")
# plt.legend()
# plt.show()

# plt.figure()
# plt.imshow(cm, cmap='Blues')
# plt.title('Confusion Matrix (Normalized)')
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         plt.text(j, i, f"{cm[i, j]*100:.2f}%", ha="center", va="center",
#                  color="white" if cm[i, j]>0.5 else "black")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# plt.show()



# ---- Feature extraction ----
def extract_features(dataloader, encoder, device='cuda'):
    encoder.eval()
    all_feats = []
    all_labels = []
    with torch.no_grad():
        for items in tqdm(dataloader):
            imgs, labels = items
            imgs = imgs.to(device)
            feats = encoder(imgs)
            all_feats.append(feats.cpu())
            all_labels.extend(labels)
    return torch.cat(all_feats), np.array(all_labels)
    
print("Starting SSL pretraining...")
for epoch in range(10):
    encoder.train()
    head.train()
    running_loss = 0.0

    for imgs in tqdm(ssl_loader):
        imgs = imgs.to(device)
        feats = encoder(imgs)
        student_out = head(feats)
        loss = dino_loss(student_out)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(ssl_loader)
    ssl_train_losses.append(avg_loss)

    print(f"Epoch {epoch+1} SSL Loss: {avg_loss:.4f}")
    
    # ---- Evaluate downstream performance after this SSL epoch ----
    train_feats, train_labels = extract_features(train_loader, encoder, device)
    val_feats, val_labels     = extract_features(val_loader, encoder, device)
    
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_feats, train_labels)
    
    val_preds_proba = clf.predict_proba(val_feats)[:,1]
    epoch_auc = roc_auc_score(val_labels, val_preds_proba)
    
    downstream_aucs.append(epoch_auc)
    print(f"Epoch {epoch+1} AUC: {epoch_auc:.4f}")


# ---- Evaluation Metrics and Plots ----
auc_score = roc_auc_score(val_labels, val_preds_proba)
fpr, tpr, _ = roc_curve(val_labels, val_preds_proba)
cm = confusion_matrix(val_labels, val_preds, normalize='true')
accuracy = accuracy_score(val_labels, val_preds)
sensitivity = cm[1,1]  # Recall/True positive rate for malignant
specificity = cm[0,0]  # True negative rate for benign

print(f"AUC: {auc_score:.4f} | Accuracy: {accuracy:.4f}")
print(f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")


epochs = np.arange(1, len(ssl_train_losses) + 1)

plt.figure(figsize=(12,5))
plt.plot(epochs, ssl_train_losses, marker='o', label='SSL Training Loss')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("SSL Pretraining Loss")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12,5))
plt.plot(epochs, downstream_aucs, marker='o', label='Validation AUC', color='green')
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.title("Downstream AUC After Each SSL Epoch")
plt.legend()
plt.grid(True)
plt.show()

