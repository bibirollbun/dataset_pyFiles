!pip install --upgrade pillow
!pip install timm==0.4.12
!pip install facenet-pytorch==2.5.2
!pip install torch==1.9.1+cpu torchvision==0.10.1+cpu torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html




import os, zipfile, glob
import pandas as pd
import torch

# ------------------------------
# 0) Unzip the entire dataset
# ------------------------------
zip_path = "/kaggle/input/ffc40-degraded/kaggle"
extract_dir = "/kaggle/working/ffc40-degraded"

if not os.path.exists(extract_dir):
    os.makedirs(extract_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(extract_dir)
    print("âœ… Extracted zip")
else:
    print("ðŸ“‚ Already extracted")

# ------------------------------
# 1) Check faces_cache path
# ------------------------------
faces_cache_dir = "/kaggle/input/ffc40-degraded/kaggle/working/ff-c40/faces_cache"
print("Looking for .pt files in:", faces_cache_dir)

# ------------------------------
# 2) Verify that files exist
# ------------------------------
pt_files = glob.glob(os.path.join(faces_cache_dir, "**/*.pt"), recursive=True)
print("Number of .pt files found:", len(pt_files))

# If 0, the path is wrong â†’ stop here
if len(pt_files) == 0:
    raise ValueError("No .pt files found! Check your path inside the zip.")

# ------------------------------
# 3) Build DataFrame
# ------------------------------
df = pd.DataFrame({"face_path": pt_files})
print(df.head())

# ------------------------------
# 4) Quick check: inspect one .pt
# ------------------------------
# Pick one .pt file
sample_path = pt_files[0]
data = torch.load(sample_path)

# Print full content safely
print("Type:", type(data))
if isinstance(data, dict):
    print("Keys in dict:", data.keys())
    for k,v in data.items():
        print(f"Key: {k}, Type: {type(v)}, ", end="")
        if isinstance(v, torch.Tensor):
            print(f"Shape: {v.shape}")
        else:
            print(f"Value preview: {v}")
else:
    # maybe it's a tensor or something else
    print(data)



import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import glob, os, pandas as pd

# -------------------------
# Dataset
# -------------------------
class CachedFaceDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = torch.load(self.df.iloc[idx]["face_path"])
        img, label = data["face"].float(), data["label"]
        if img.ndim == 3 and img.shape[0] != 3:
            img = img.permute(2,0,1)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label).long()

# -------------------------
# Build DataFrame & Split
# -------------------------
faces_cache_dir = "/kaggle/input/ffc40-degraded/kaggle/working/ff-c40/faces_cache"
files = glob.glob(os.path.join(faces_cache_dir, "*.pt"))
df = pd.DataFrame({"face_path": files})
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

num_total = len(df)
num_train = int(0.8*num_total)
num_val   = int(0.1*num_total)
num_test  = num_total - num_train - num_val

train_df = df.iloc[:num_train]
val_df   = df.iloc[num_train:num_train+num_val]
test_df  = df.iloc[num_train+num_val:]

print(f"Total: {num_total}, Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# -------------------------
# Dataloaders
# -------------------------
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tf = transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
train_loader = DataLoader(CachedFaceDataset(train_df, transform=tf),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(CachedFaceDataset(val_df, transform=tf),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(CachedFaceDataset(test_df, transform=tf),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

import torch.nn as nn
import torch.optim as optim
import timm

# -------------------------
# Model
# -------------------------
model = timm.create_model('xception', pretrained=True, num_classes=2).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

from tqdm import tqdm
import matplotlib.pyplot as plt

NUM_EPOCHS = 10

train_losses, val_losses = [], []
train_accs, val_accs = [], []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss, total, correct = 0.0, 0, 0
    loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{NUM_EPOCHS}]", leave=False)
    for imgs, labels in loop:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * imgs.size(0)
        preds = out.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += imgs.size(0)
        
        loop.set_postfix(loss=loss.item(), acc=correct/total)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    # -------------------------
    # Validation
    # -------------------------
    model.eval()
    val_loss, val_total, val_correct = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            out = model(imgs)
            loss = criterion(out, labels)
            val_loss += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            val_correct += (preds == labels).sum().item()
            val_total += imgs.size(0)
    
    val_loss_epoch = val_loss / val_total
    val_acc_epoch  = val_correct / val_total
    val_losses.append(val_loss_epoch)
    val_accs.append(val_acc_epoch)
    
    print(f"Epoch {epoch+1}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.4f} | Val Loss={val_loss_epoch:.4f}, Val Acc={val_acc_epoch:.4f}")

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_accs, label="Train Acc")
plt.plot(val_accs, label="Val Acc")
plt.title("Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np

model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        out = model(imgs)
        preds = out.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

acc = np.mean(np.array(all_preds) == np.array(all_labels))
cm = confusion_matrix(all_labels, all_preds)

print(f"Test Accuracy: {acc:.4f}")

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["REAL","FAKE"], yticklabels=["REAL","FAKE"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Test Confusion Matrix")
plt.show()





