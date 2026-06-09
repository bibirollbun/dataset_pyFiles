import warnings 
warnings.filterwarnings('ignore')


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch 
from  torch.utils.data import Dataset, DataLoader 
from torchvision.transforms import Resize
import pydicom
import glob
import os
import cv2
from sklearn.metrics import accuracy_score, roc_auc_score
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms as T


BASE_PATH = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification'
train_folder = os.path.join(BASE_PATH,'train')
test_folder = os.path.join(BASE_PATH,'test')
train_labels = pd.read_csv(os.path.join(BASE_PATH,'train_labels.csv'))


required_slices=50


train_labels['BraTS21ID'] = train_labels['BraTS21ID'].apply(lambda x: str(x).zfill(5))
train_dict = train_labels.set_index('BraTS21ID')['MGMT_value'].to_dict()
train_labels.head()


patient_ids = [ids for ids in list(train_dict.keys()) if ids not in ['00109','00123','00709'] ]
len(patient_ids)


class BrDataset(Dataset):
    def __init__(self, base_path, patient_ids, label_dict, resize_size=(224, 224), required_slices=32, is_train=True):
        self.base_path = base_path
        self.patient_ids = patient_ids
        self.label_dict = label_dict
        self.modalities = ['FLAIR', 'T1w','T1wCE','T2w']
        self.required_slices = required_slices
        self.is_train = is_train

        # Albumentations transform
        self.transform = A.Compose([
            A.Resize(height=resize_size[0], width=resize_size[1]),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.1),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.5),
            # A.Normalize(mean=0.5, std=0.5),
        ])

    def load_slices(self, case_id, modality):
        paths = sorted(
            glob.glob(os.path.join(self.base_path, 'train', case_id, modality, '*.dcm')),
            key=lambda x: int(pydicom.dcmread(x).InstanceNumber)
        )

        slices = []
        for path in paths:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype('float32')
            if np.sum(img) > 0:
                slices.append(img)

        num_slices = len(slices)
        if num_slices == 0:
            return torch.zeros((self.required_slices, 224, 224), dtype=torch.float32)

        # Uniform + Center Sampling
        if num_slices >= self.required_slices:
            center = num_slices // 2
            start = max(0, center - self.required_slices // 2)
            end = start + self.required_slices
            if end > num_slices:
                end = num_slices
                start = end - self.required_slices
            indices = np.linspace(start, end - 1, self.required_slices).astype(int)
            slices = [slices[i] for i in indices]
        else:
            pad = self.required_slices - num_slices
            pad_start = pad // 2
            pad_end = pad - pad_start
            h, w = slices[0].shape
            zero = np.zeros((h, w), dtype=np.float32)
            slices = [zero] * pad_start + slices + [zero] * pad_end

        volume = []
        for s in slices:
            aug = self.transform(image=s)
            img_tensor = torch.tensor(aug["image"])
            volume.append(img_tensor)

        volume = torch.stack(volume)
        return volume

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        case_id = str(self.patient_ids[idx]).zfill(5)
        vol = []
        for modality in self.modalities:
            volume = self.load_slices(case_id, modality)
            vol.append(volume)
        vol = torch.stack(vol)  # shape: [2, required_slices, H, W]

        label = torch.tensor(self.label_dict[case_id], dtype=torch.long)
        return vol, label



dataset = BrDataset(BASE_PATH, patient_ids, train_dict)
vol,label = dataset[0]
print("vol shape:", vol.shape)  # Expect: torch.Size([2, 32, 224, 224])



import matplotlib.pyplot as plt
import math

dataset = BrDataset(BASE_PATH, patient_ids, train_dict)
vol, label = dataset[10]             # vol shape: [2, N, H, W]
flair_volume = vol[2]                # shape: [N, H, W]
num_slices = flair_volume.shape[0]

cols = 6
rows = math.ceil(num_slices / cols)

# Plot the slices
fig, axes = plt.subplots(rows, cols, figsize=(16, rows * 2))
fig.suptitle(f"FLAIR modality - {num_slices} slices", fontsize=16)

for i in range(rows * cols):
    r, c = divmod(i, cols)
    ax = axes[r, c] if rows > 1 else axes[c]

    if i < num_slices:
        slice_2d = flair_volume[i].numpy()
        ax.imshow(slice_2d, cmap='gray')
        ax.set_title(f"Slice {i}")
    ax.axis("off")

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.show()



import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionModule(nn.Module):
    def __init__(self, input_dim, attention_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, attention_dim),
            nn.Tanh(),
            nn.Linear(attention_dim, 1)
        )

    def forward(self, features):  # [B, T, F]
        scores = self.attention(features)         # [B, T, 1]
        weights = torch.softmax(scores, dim=1)    # attention weights
        context = torch.sum(weights * features, dim=1)  # [B, F]
        return context

class TumorClassifier3DConvLSTM(nn.Module):
    def __init__(self, hidden_dim=128, attention_dim=64):
        super().__init__()

        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels=4, out_channels=16, kernel_size=3, padding=1),  # 4 modalities
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),  # → [B, 16, T/2, H/2, W/2]
            nn.Dropout3d(0.1),

            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),  # → [B, 32, T/4, H/4, W/4]
            nn.Dropout3d(0.1),
        )

        # LSTM will be initialized dynamically
        self.lstm = nn.LSTM(
            input_size=1,  # placeholder
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.attention = AttentionModule(hidden_dim * 2, attention_dim)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        # x: [B, 4, T, H, W]
        x = self.conv3d(x)  # → [B, 32, T/4, H/4, W/4]
        b, c, t, h, w = x.shape

        x = x.permute(0, 2, 1, 3, 4)       # → [B, T, C, H, W]
        x = x.contiguous().view(b, t, -1)  # → [B, T, C*H*W]

        if self.lstm.input_size != x.shape[-1]:
            self.lstm = nn.LSTM(
                input_size=x.shape[-1],
                hidden_size=self.lstm.hidden_size,
                num_layers=self.lstm.num_layers,
                batch_first=True,
                bidirectional=True
            ).to(x.device)

        lstm_out, _ = self.lstm(x)           # [B, T, 2*hidden_dim]
        context = self.attention(lstm_out)   # [B, 2*hidden_dim]
        out = self.classifier(context)       # [B, 2]
        return out



from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for fold, (train_idx, val_idx) in enumerate(skf.split(train_labels['BraTS21ID'], train_labels['MGMT_value'])):
    train_ids = train_labels.iloc[train_idx]['BraTS21ID'].astype(str).tolist()
    val_ids = train_labels.iloc[val_idx]['BraTS21ID'].astype(str).tolist()
    break  # Only using first fold for now



len(train_ids),len(val_ids)


import os
from torch.utils.data import DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import torch

# =================== SETTINGS ===================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 2
EPOCHS = 10
LR = 1e-4
PATIENCE = 3
BEST_MODEL_PATH = "/kaggle/working/best_model.pt"

# =================== DATALOADERS ===================
train_dataset = BrDataset(BASE_PATH, train_ids, train_dict, resize_size=(224, 224), required_slices=32, is_train=True)
val_dataset = BrDataset(BASE_PATH, val_ids, train_dict, resize_size=(224, 224), required_slices=32, is_train=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# =================== MODEL + LOSS + OPTIM ===================
model = TumorClassifier3DConvLSTM().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# =================== TRAINING LOOP WITH EARLY STOPPING ===================
def train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epochs=10):
    best_auc = 0
    patience_counter = 0

    for epoch in range(epochs):
        print(f"\n--- Epoch {epoch+1}/{epochs} ---")

        # -------- TRAIN --------
        model.train()
        total_loss = 0

        for x, y in tqdm(train_loader):
            x, y = x.to(device), y.to(device).long()

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Train Loss: {avg_loss:.4f}")

        # -------- VALIDATION --------
        model.eval()
        val_preds, val_targets = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device).long()
                out = model(x)
                prob = F.softmax(out, dim=1)[:, 1]

                val_preds.extend(prob.cpu().numpy())
                val_targets.extend(y.cpu().numpy())

        val_auc = roc_auc_score(val_targets, val_preds)
        val_preds_cls = (torch.tensor(val_preds) > 0.5).int()
        val_acc = accuracy_score(val_targets, val_preds_cls)

        print(f"Val AUC: {val_auc:.4f} | Val Accuracy: {val_acc:.4f}")

        # -------- EARLY STOPPING & CHECKPOINT --------
        if val_auc > best_auc:
            print(f"AUC improved from {best_auc:.4f} → {val_auc:.4f}. Saving model.")
            best_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print("Early stopping triggered.")
            break

# =================== RUN ===================
train_and_validate(model, train_loader, val_loader, criterion, optimizer, device, epochs=EPOCHS)





