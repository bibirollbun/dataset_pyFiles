import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import pydicom

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, roc_curve

# ========== CONFIGURATION ==========
DATA_DIR = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 3
BATCH_SIZE = 8

# ========== LOAD LABELS ==========
labels_df = pd.read_csv(LABELS_CSV)
labels_df["BraTS21ID"] = labels_df["BraTS21ID"].astype(str).str.zfill(5)

# ========== DATASET ==========
class BrainMRIDataset(Dataset):
    def __init__(self, patient_ids, labels, root_dir, transform=None):
        self.patient_ids = patient_ids
        self.labels = labels
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels[idx]

        flair_dir = os.path.join(self.root_dir, patient_id, "FLAIR")
        flair_files = sorted([f for f in os.listdir(flair_dir) if f.endswith(".dcm")])
        mid_file = flair_files[len(flair_files) // 2]
        dcm_path = os.path.join(flair_dir, mid_file)

        dcm = pydicom.dcmread(dcm_path)
        img = dcm.pixel_array
        img = Image.fromarray(img).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)

# ========== TRANSFORMS ==========
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# ========== DATA SPLIT ==========
subset_df = labels_df.sample(n=300, random_state=42).reset_index(drop=True)
train_df = subset_df.sample(frac=0.7, random_state=42)
temp_df = subset_df.drop(train_df.index)
val_df = temp_df.sample(frac=0.5, random_state=42)
test_df = temp_df.drop(val_df.index)

train_ds = BrainMRIDataset(train_df["BraTS21ID"].tolist(), train_df["MGMT_value"].tolist(), TRAIN_DIR, transform)
val_ds = BrainMRIDataset(val_df["BraTS21ID"].tolist(), val_df["MGMT_value"].tolist(), TRAIN_DIR, transform)
test_ds = BrainMRIDataset(test_df["BraTS21ID"].tolist(), test_df["MGMT_value"].tolist(), TRAIN_DIR, transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

# ========== MODEL ==========
model = models.efficientnet_b0(pretrained=True)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
model = model.to(DEVICE)

# ========== TRAIN ==========
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.unsqueeze(1).to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{EPOCHS}] - Loss: {avg_loss:.4f}")

# ========== EVALUATION ==========
def evaluate(loader):
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int).flatten()
            y_true.extend(labels.numpy())
            y_pred.extend(preds)
            y_prob.extend(probs.flatten())
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    return acc, f1, kappa, auc, y_true, y_prob

val_results = evaluate(val_loader)
test_results = evaluate(test_loader)

print(f"\nValidation - Accuracy: {val_results[0]:.4f}, F1: {val_results[1]:.4f}, Kappa: {val_results[2]:.4f}, AUC: {val_results[3]:.4f}")
print(f"Test      - Accuracy: {test_results[0]:.4f}, F1: {test_results[1]:.4f}, Kappa: {test_results[2]:.4f}, AUC: {test_results[3]:.4f}")

# ========== ROC CURVE ==========
fpr, tpr, _ = roc_curve(test_results[4], test_results[5])
plt.figure()
plt.plot(fpr, tpr, label='ROC Curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Test ROC Curve')
plt.legend()
plt.grid()
plt.show()


