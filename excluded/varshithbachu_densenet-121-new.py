import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score, roc_curve, confusion_matrix
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
import pydicom
from PIL import Image

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# CSV and image root
data_csv = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv'
data_root = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train'

df = pd.read_csv(data_csv)
df = df.dropna()  # drop rows with missing labels

# Train/val/test split
train_ids, test_ids = train_test_split(df, test_size=0.2, random_state=42, stratify=df['MGMT_value'])
train_ids, val_ids = train_test_split(train_ids, test_size=0.1, random_state=42, stratify=train_ids['MGMT_value'])

# Image transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

class BrainDataset(Dataset):
    def __init__(self, df, root_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_id = str(self.df.loc[idx, 'BraTS21ID']).zfill(5)
        label = int(self.df.loc[idx, 'MGMT_value'])

        flair_path = os.path.join(self.root_dir, patient_id, 'FLAIR')
        slices = sorted(os.listdir(flair_path))
        mid_slice = slices[len(slices) // 2]

        dicom = pydicom.dcmread(os.path.join(flair_path, mid_slice))
        img = dicom.pixel_array.astype(np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
        img = Image.fromarray((img * 255).astype(np.uint8)).convert('L')

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label).long()

# Loaders
train_ds = BrainDataset(train_ids, data_root, transform)
val_ds = BrainDataset(val_ids, data_root, transform)
test_ds = BrainDataset(test_ids, data_root, transform)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=2)

# Model
model = models.densenet121(pretrained=True)
model.features.conv0 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # for 1-channel
model.classifier = nn.Linear(model.classifier.in_features, 2)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Training loop
train_losses, val_losses, train_accs, val_accs = [], [], [], []
for epoch in range(10):
    model.train()
    correct, total, loss_sum = 0, 0, 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss = criterion(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * x.size(0)
        _, pred = torch.max(out, 1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    train_losses.append(loss_sum / total)
    train_accs.append(correct / total)

    # Validation
    model.eval()
    correct, total, val_loss = 0, 0, 0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            val_loss += loss.item() * x.size(0)
            _, pred = torch.max(out, 1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    val_losses.append(val_loss / total)
    val_accs.append(correct / total)

    print(f"Epoch {epoch+1}/10 - Train Acc: {train_accs[-1]:.4f}, Val Acc: {val_accs[-1]:.4f}")

# Test evaluation
model.eval()
y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        out = model(x)
        prob = torch.softmax(out, 1)[:, 1].cpu().numpy()
        pred = torch.argmax(out, 1).cpu().numpy()
        y_true.extend(y.numpy())
        y_pred.extend(pred)
        y_prob.extend(prob)

# Metrics
test_acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_prob)
print(f"\nTest Accuracy: {test_acc:.4f}, F1: {f1:.4f}, Kappa: {kappa:.4f}, AUC: {auc:.4f}")

# Graphs
plt.figure()
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Val Acc')
plt.legend()
plt.title('Accuracy over Epochs')
plt.show()

plt.figure()
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.legend()
plt.title('Loss over Epochs')
plt.show()

fpr, tpr, _ = roc_curve(y_true, y_prob)
plt.figure()
plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.title('ROC Curve')
plt.legend()
plt.show()

cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


