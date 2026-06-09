!pip install pydicom 


# Basic imports
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
import cv2

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torchvision.transforms import Resize
from sklearn.model_selection import StratifiedKFold 

# Fix random seed
torch.manual_seed(42)
np.random.seed(42)

# Path to your dataset
BASE_PATH = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train'
LABEL_CSV = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv'

test_path = '/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test'



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# import torch
# import torch_xla
# import torch_xla.core.xla_model as xm

# # Instead of 'cuda' use:
# device = xm.xla_device()


print(device)


# Load CSV with patient IDs and labels
label_df = pd.read_csv(LABEL_CSV)
print(label_df.head())


label_dict = {
    f"{int(row.BraTS21ID  ):05d}":int(row.MGMT_value) for _, row in label_df.iterrows()
    if row.BraTS21ID   not in [109, 123, 709]

}
len(label_dict.keys())


class BrDataset(Dataset):
    def __init__(self,base_path,patient_ids,label_dict,resize_size = (128,128)):
        self.base_path = base_path 
        self.patient_ids = patient_ids 
        self.label_dict = label_dict 
        self.modalities = ['FLAIR','T1w']
        self.resize = Resize(resize_size)
    def load_slices(self,case_id,modality):
        paths = sorted(
            glob.glob(os.path.join(self.base_path,case_id,modality,"*.dcm")),
            key = lambda x:int(pydicom.dcmread(x).InstanceNumber)
        )

        slices = []
        for path in paths:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)

            # Normalize if not blank 
            if np.max(img)>0:
                img = (img-np.min(img))/(np.max(img)-np.min(img)+1e5)
                slices.append(img)
        if len(slices)==0:
            return None

        num_slices = len(slices)
        if num_slices<18:
            pad_needed = 18 - num_slices
            pad_start = pad_needed//2 
            pad_end = num_slices - pad_start 
            h,w = slices[0].shape 
            zero_slice = np.zeros((h,w),dtype = np.float32)
            slices = [zero_slice] * pad_start + slices + [zero_slice] * (18 - len(slices) - pad_start)

        else:
            mid = num_slices//2 
            start = max(0, mid - 9)
            slices = slices[start:start+18]
        resized = [self.resize(torch.tensor(s)[None,...]) for s in slices]
        volume = torch.stack(resized).squeeze(1)
        return volume 
    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self,idx):
        case_id = str(self.patient_ids[idx]).zfill(5)
        vols = []
        for modality in self.modalities:
            v = self.load_slices(case_id, modality)
            if v is None:
                # Try next sample if something failed
                return self.__getitem__((idx + 1) % len(self))
            vols.append(v)
        x = torch.stack(vols)
        y = torch.tensor(self.label_dict[case_id], dtype=torch.long)
        return x,y
            


# dataset = BrDataset(BASE_PATH, patient_ids, label_dict)
# loader = DataLoader(dataset, batch_size=4, shuffle=True)

# tens,y = next(iter(loader))
# tens[0].shape
# y


class BrTestDataset(Dataset):
    def __init__(self, base_path, patient_ids, resize_size=(128, 128)):
        self.base_path = base_path
        self.patient_ids = patient_ids
        self.modalities = ["FLAIR", "T1w"]
        self.resize = Resize(resize_size)

    def load_slices(self, case_id, modality):
        paths = sorted(
            glob.glob(os.path.join(self.base_path, case_id, modality, "*.dcm")),
            key=lambda x: int(pydicom.dcmread(x).InstanceNumber)
        )

        slices = []
        for path in paths:
            dcm = pydicom.dcmread(path)
            img = dcm.pixel_array.astype(np.float32)
            if np.max(img) > 0:
                img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-5)
                slices.append(img)

        if len(slices) == 0:
            return None
        num_slices = len(slices)
        # Pad or take center 18
        if len(slices) < 18:
            pad_needed = 18 - len(slices)
            pad_start = pad_needed // 2
            pad_end = pad_needed - pad_start
            h, w = slices[0].shape
            zero_slice = np.zeros((h, w), dtype=np.float32)
            slices = [zero_slice] * pad_start + slices + [zero_slice] * pad_end
        else:
            mid = num_slices // 2
            start = mid - 9
            slices = slices[start:start + 18]

        resized = [self.resize(torch.tensor(s)[None, ...]) for s in slices]
        volume = torch.stack(resized).squeeze(1)  # [18, H, W]
        return volume

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        case_id = str(self.patient_ids[idx]).zfill(5)
        vols = []

        for modality in self.modalities:
            v = self.load_slices(case_id, modality)
            if v is None:
                return self.__getitem__((idx + 1) % len(self))
            vols.append(v)

        x = torch.stack(vols)  # [2, 18, 128, 128]
        return x, case_id



class TumorClassifier3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels = 2,out_channels = 16,kernel_size=3,padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size = 2)

        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)                            # → [B, 32, 9, 64, 64]
        self.pool2 = nn.MaxPool3d(2)                                                        # → [B, 32, 4, 32, 32]

        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)                            # → [B, 64, 4, 32, 32]
        self.pool3 = nn.AdaptiveAvgPool3d(1)                                                # → [B, 64, 1, 1, 1]

        self.fc = nn.Linear(64, 2)  # 2 output classes: 0 or 1

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))  # → [B, 32, 9, 64, 64]
        x = self.pool2(x)          # → [B, 32, 4, 32, 32]

        x = F.relu(self.conv3(x))  # → [B, 64, 4, 32, 32]
        x = self.pool3(x)   
        x = x.view(x.size(0), -1)  # → [B, 64]
        x = self.fc(x)  
        return x 



from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm
import torch.nn.functional as F

def train_one_fold(model, train_loader, val_loader, optimizer, criterion, device, num_epochs=2):
    best_auc = 0

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")

        # ---------------- Train ----------------
        model.train()
        train_loss = 0

        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)         # [B, 2, 18, 128, 128]
            labels = labels.to(device).long()  # [B]

            optimizer.zero_grad()
            outputs = model(inputs)            # [B, 2]
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Train Loss: {train_loss / len(train_loader):.4f}")

        # ---------------- Validation ----------------
        model.eval()
        val_preds = []
        val_labels = []

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device).long()

                outputs = model(inputs)
                probs = F.softmax(outputs, dim=1)[:, 1]  # Prob of class 1 (MGMT=1)

                val_preds.extend(probs.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        val_auc = roc_auc_score(val_labels, val_preds)
        val_preds_cls = (np.array(val_preds) > 0.5).astype(int)
        val_acc = accuracy_score(val_labels, val_preds_cls)

        print(f"Val Accuracy: {val_acc:.4f}  |  Val AUC: {val_auc:.4f}")

        if val_auc > best_auc:
            best_auc = val_auc
            # Optional: save best model
            # torch.save(model.state_dict(), f"best_model_fold{fold}.pt")

    return best_auc



skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
test_ids = sorted(os.listdir(test_path))
test_dataset = BrTestDataset(base_path=test_path, patient_ids=test_ids)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

all_fold_preds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(label_df['BraTS21ID'], label_df['MGMT_value'])):
    print(f"\n========== Fold {fold + 1} ==========")

    train_ids = label_df.iloc[train_idx]['BraTS21ID'].values
    val_ids = label_df.iloc[val_idx]['BraTS21ID'].values

    train_dataset = BrDataset(BASE_PATH, train_ids, label_dict)
    val_dataset = BrDataset(BASE_PATH, val_ids, label_dict)

    train_loader = DataLoader(train_dataset, batch_size=6, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=6, shuffle=False, num_workers=2)

    model = TumorClassifier3D().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    _ = train_one_fold(model, train_loader, val_loader, optimizer, criterion, device)

    # Inference on test set for this fold
    model.eval()
    fold_preds = []

    with torch.no_grad():
        for inputs, case_id in tqdm(test_loader):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1]  # class 1
            fold_preds.extend(probs.cpu().numpy())

    all_fold_preds.append(fold_preds)



# Average predictions from 5 folds
avg_preds = np.mean(all_fold_preds, axis=0)

# Save submission
submission = pd.DataFrame({
    "BraTS21ID": [int(pid) for pid in test_ids],
    "MGMT_value": avg_preds
})

submission.to_csv("submission.csv", index=False)
print("Saved final submission.")

