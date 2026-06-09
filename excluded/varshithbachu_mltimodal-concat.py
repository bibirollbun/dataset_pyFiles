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
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, roc_auc_score
from skimage.transform import resize

# =============================
# 2. Configurations
# =============================
class CFG:
    train_folder = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train"
    csv_file = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
    num_classes = 2
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = (224, 224)

# =============================
# 3. Dataset
# =============================
class BrainTumorDataset(Dataset):
    def __init__(self, csv_file, folder, transform=None):
        self.labels = pd.read_csv(csv_file)
        self.folder = folder
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patient_id = str(self.labels.iloc[idx, 0]).zfill(5)
        label = self.labels.iloc[idx, 1]
        flair_folder = os.path.join(self.folder, patient_id, "T1wCE")
        slices = []
        if os.path.exists(flair_folder):
            for fname in sorted(os.listdir(flair_folder)):
                if fname.endswith(".dcm"):
                    dcm = pydicom.dcmread(os.path.join(flair_folder, fname))
                    slices.append(dcm.pixel_array)
        if len(slices) == 0:
            img = np.zeros(CFG.img_size)
        else:
            mid_slice = slices[len(slices) // 2]
            img = resize(mid_slice, CFG.img_size)

        img = np.stack([img]*3, axis=0)  # Convert to 3 channels
        img = torch.tensor(img, dtype=torch.float32)

        if self.transform:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.long)

# =============================
# 4. Model
# =============================
class ResNetClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = models.resnet152(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):
        return self.backbone(x)

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

        preds = outputs.argmax(dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

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
            preds = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    kappa = cohen_kappa_score(all_labels, all_preds)
    roc = roc_auc_score(all_labels, all_probs)
    return acc, f1, kappa, roc

# =============================
# 6. Run everything
# =============================
# Prepare dataset
full_dataset = BrainTumorDataset(CFG.csv_file, CFG.train_folder)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size

train_ds, val_ds, test_ds = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_ds, batch_size=CFG.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=CFG.batch_size, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=CFG.batch_size, shuffle=False)

# Model, optimizer, loss
model = ResNetClassifier().to(CFG.device)
optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate)
criterion = nn.CrossEntropyLoss()

# Training loop
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



import matplotlib.pyplot as plt

# Parsed results from your training
train_acc_history = [0.5037, 0.4939, 0.5183, 0.5037, 0.5501, 0.5795, 0.6161, 0.5892, 0.5990, 0.5648]
val_acc_history = [0.5402, 0.5517, 0.4713, 0.5977, 0.5287, 0.5402, 0.5402, 0.5517, 0.5057, 0.4828]
val_f1_history = [0.6774, 0.6667, 0.2581, 0.6847, 0.5941, 0.6296, 0.6364, 0.6061, 0.6055, 0.1818]
val_kappa_history = [0.0365, 0.0696, -0.0204, 0.1719, 0.0429, 0.0574, 0.0549, 0.0921, -0.0146, 0.0106]
val_roc_history = [0.5337, 0.4966, 0.5048, 0.6164, 0.5469, 0.5175, 0.4889, 0.5422, 0.5345, 0.5151]

# Epochs
epochs = range(1, len(train_acc_history) + 1)

# Plotting
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
plt.plot(epochs, train_acc_history, label='Train Accuracy')
plt.plot(epochs, val_acc_history, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(epochs, val_f1_history, label='Validation F1 Score', color='orange')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Validation F1 Score over Epochs')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(epochs, val_kappa_history, label='Validation Kappa', color='green')
plt.xlabel('Epoch')
plt.ylabel('Kappa')
plt.title('Validation Kappa over Epochs')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(epochs, val_roc_history, label='Validation ROC AUC', color='red')
plt.xlabel('Epoch')
plt.ylabel('ROC AUC')
plt.title('Validation ROC AUC over Epochs')
plt.legend()

plt.tight_layout()
plt.show()


