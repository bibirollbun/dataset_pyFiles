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
import cv2
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

!pip install -q pydicom
import pydicom
import glob



import os
import pandas as pd

# Read the labels
df = pd.read_csv('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv')
df['BraTS21ID'] = df['BraTS21ID'].astype(str).str.zfill(5)
df.head()



def load_dicom_volume(patient_id, modality):
    # Folder path: e.g., train/00753/T1w/*
    base = f'/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/{patient_id}/{modality}/'
    dicom_files = sorted(glob.glob(base + "*.dcm"), key=lambda x: int(x.split('-')[-1].split('.')[0]))

    slices = []
    for file in dicom_files:
        ds = pydicom.dcmread(file)
        img = ds.pixel_array.astype(np.float32)
        slices.append(img)
    
    volume = np.stack(slices, axis=-1)  # shape: (H, W, D)
    return volume


def load_multichannel_slice(patient_id, target_shape=(224, 224)):
    modalities = ['T1w', 'T1wCE', 'T2w', 'FLAIR']
    slices = []
    for mod in modalities:
        vol = load_dicom_volume(patient_id, mod)  # shape: (H, W, D)
        vol = np.nan_to_num(vol)
        vol = (vol - vol.mean()) / (vol.std() + 1e-5)
        mid = vol.shape[2] // 2
        slice_img = vol[:, :, mid]
        # Resize to target_shape
        slice_img = cv2.resize(slice_img, target_shape, interpolation=cv2.INTER_LINEAR)
        slices.append(slice_img)
    stacked = np.stack(slices, axis=0)  # (4, 224, 224)
    return stacked.astype(np.float32)



test_id = '00753'
sample = load_multichannel_slice(test_id)
print("Shape:", sample.shape)  # should be (4, 224, 224)

# Visualize each modality
import matplotlib.pyplot as plt

for i, mod in enumerate(['T1w', 'T1wCE', 'T2w', 'FLAIR']):
    plt.subplot(1, 4, i+1)
    plt.imshow(sample[i], cmap='gray')
    plt.title(mod)
    plt.axis('off')
plt.show()


class MultiModalityDataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        pid = self.df.iloc[idx]['BraTS21ID']
        label = self.df.iloc[idx]['MGMT_value']
        img = load_multichannel_slice(pid)
        return torch.tensor(img), torch.tensor(label, dtype=torch.float32)


class FourModalityCNN(nn.Module):
    def __init__(self):
        super(FourModalityCNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))  # new
        self.fc1 = nn.Linear(32, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (B, 16, 112, 112)
        x = self.pool(F.relu(self.conv2(x)))  # (B, 32, 56, 56)
        x = self.gap(x)                       # (B, 32, 1, 1)
        x = x.view(x.size(0), -1)             # (B, 32)
        x = F.relu(self.fc1(x))               # (B, 128)
        return torch.sigmoid(self.fc2(x)).squeeze()



train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

train_ds = MultiModalityDataset(train_df)
val_ds = MultiModalityDataset(val_df)

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=16)


sample = next(iter(train_loader))
print("Sample loaded:", sample[0].shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FourModalityCNN().to(device)

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(5):
    model.train()
    train_loss = 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(device), y.to(device)
            preds = model(X) > 0.5
            correct += (preds == y.bool()).sum().item()
            total += y.size(0)

    acc = correct / total
    print(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val Acc: {acc:.4f}")




