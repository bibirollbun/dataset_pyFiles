# =============================
# 1. Imports
# =============================
import os
import numpy as np
import pandas as pd
import pydicom
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
import cv2

# =============================
# 2. Configurations
# =============================
class CFG:
    path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/"
    num_classes = 1
    batch_size = 8
    num_epochs = 5
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (224, 224)

# =============================
# 3. Dataset
# =============================
class BrainTumorDataset(Dataset):
    def __init__(self, csv_file, folder):
        self.labels = pd.read_csv(csv_file)
        self.folder = folder

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patient_id = str(self.labels.iloc[idx, 0]).zfill(5)
        label = self.labels.iloc[idx, 1]
        img = self.load_mri_image(patient_id)
        return img, torch.tensor(label, dtype=torch.float32)

    def load_mri_image(self, folder, scan_type="FLAIR"):
        path_file = os.path.join(CFG.path, 'train', folder, scan_type)
        slices = [s for s in os.listdir(path_file) if s.endswith('.dcm')]
        slices.sort()
        mid = len(slices) // 2
        selected = [slices[mid-1], slices[mid], slices[mid+1]]
        imgs = []
        for s in selected:
            dcm = pydicom.dcmread(os.path.join(path_file, s))
            img = dcm.pixel_array
            img = cv2.resize(img, CFG.img_size)
            imgs.append(img)
        imgs = np.stack(imgs, axis=0)
        imgs = imgs / np.max(imgs)
        return torch.tensor(imgs, dtype=torch.float32)

# =============================
# 4. Model
# =============================
class ImageClf(nn.Module):
    def __init__(self):
        super(ImageClf, self).__init__()
        self.enc = models.resnet152(pretrained=True)
        self.enc.fc = nn.Identity()
        self.pooler = nn.AdaptiveAvgPool2d((1,1))
        self.clf = nn.Linear(2048, 1)

    def forward(self, x):
        x = self.enc.conv1(x)
        x = self.enc.bn1(x)
        x = self.enc.relu(x)
        x = self.enc.maxpool(x)
        x = self.enc.layer1(x)
        x = self.enc.layer2(x)
        x = self.enc.layer3(x)
        x = self.enc.layer4(x)
        x = self.pooler(x).squeeze(-1).squeeze(-1)
        x = self.clf(x)
        return x.squeeze(-1)

# =============================
# 5. Training and Evaluation
# =============================

def train_fn(model, loader, optimizer, criterion):
    model.train()
    all_preds = []
    all_labels = []
    for imgs, labels in loader:
        imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = torch.sigmoid(outputs) > 0.5
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    return acc


def eval_fn(model, loader):
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(CFG.device), labels.to(CFG.device)

            outputs = model(imgs)
            probs = torch.sigmoid(outputs)

            # Handle NaNs early
            probs = torch.nan_to_num(probs, nan=0.0)

            preds = probs > 0.5

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)

    try:
        roc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        roc = float('nan')  # If only one class present in y_true, AUC can't be computed

    return acc, f1, kappa, roc


# =============================
# 6. Run everything
# =============================
full_dataset = BrainTumorDataset(CFG.path + 'train_labels.csv', CFG.path)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])
train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False)

model = ImageClf().to(CFG.device)
optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)
criterion = nn.BCEWithLogitsLoss()

for epoch in range(CFG.num_epochs):
    train_acc = train_fn(model, train_loader, optimizer, criterion)
    val_acc, val_f1, val_kappa, val_roc = eval_fn(model, val_loader)
    print(f"Epoch {epoch+1}/{CFG.num_epochs} => Train Acc: {train_acc:.4f} | Val Acc: {val_acc:.4f} | F1: {val_f1:.4f} | Kappa: {val_kappa:.4f} | ROC-AUC: {val_roc:.4f}")

# Final test evaluation
test_acc, test_f1, test_kappa, test_roc = eval_fn(model, test_loader)
print("\n=== FINAL TEST RESULTS ===")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Test F1 Score: {test_f1:.4f}")
print(f"Test Cohen's Kappa: {test_kappa:.4f}")
print(f"Test ROC AUC: {test_roc:.4f}")


