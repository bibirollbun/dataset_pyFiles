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
        os.path.join(dirname, filename)

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


pip install monai nibabel torch torchvision



# utils/dataset.py
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch

class TomoDataset(Dataset):
    def __init__(self, data_root, csv_path, transform=None):
        self.data_root = data_root
        self.labels_df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        tomo_id = row["tomo_id"]
        tomo_folder = os.path.join(self.data_root, tomo_id)
        slices = sorted(os.listdir(tomo_folder))
        volume = [np.array(Image.open(os.path.join(tomo_folder, s))) for s in slices]
        volume = np.stack(volume, axis=0).astype(np.float32)  # [D, H, W]

        if self.transform:
            volume = self.transform(volume)

        label = np.array([row["Motor axis 0"], row["Motor axis 1"], row["Motor axis 2"]], dtype=np.float32)
        has_motor = not np.isnan(label).any()
        label = label if has_motor else np.array([-1, -1, -1], dtype=np.float32)

        return torch.from_numpy(volume).unsqueeze(0), torch.tensor(label), torch.tensor(has_motor, dtype=torch.float32)



from monai.transforms import Compose, ScaleIntensity, Resize, ToTensor

transform = Compose([
    ScaleIntensity(),
    Resize((64, 128, 128)),  # D, H, W
])



# utils/dataset.py
import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import torch

class TomoDataset(Dataset):
    def __init__(self, data_root, csv_path, transform=None):
        self.data_root = data_root
        self.labels_df = pd.read_csv(csv_path)
        self.transform = transform

    def __len__(self):
        return len(self.labels_df)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        tomo_id = row["tomo_id"]
        tomo_folder = os.path.join(self.data_root, tomo_id)
        slices = sorted(os.listdir(tomo_folder))
        volume = [np.array(Image.open(os.path.join(tomo_folder, s))) for s in slices]
        volume = np.stack(volume, axis=0).astype(np.float32)  # [D, H, W]

        if self.transform:
            volume = self.transform(volume)

        label = np.array([row["Motor axis 0"], row["Motor axis 1"], row["Motor axis 2"]], dtype=np.float32)
        has_motor = not np.isnan(label).any()
        label = label if has_motor else np.array([-1, -1, -1], dtype=np.float32)

        return torch.from_numpy(volume).unsqueeze(0), torch.tensor(label), torch.tensor(has_motor, dtype=torch.float32)



import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class TomoDataset(Dataset):
    def __init__(self, root_dir, label_csv, target_shape=(64, 128, 128)):
        self.root_dir = root_dir
        self.labels = pd.read_csv(label_csv)

        # Only include rows with motor annotations and folders that exist
        self.labels = self.labels.dropna(subset=['Motor axis 0', 'Motor axis 1', 'Motor axis 2'])
        self.labels = self.labels[self.labels['tomo_id'].apply(
            lambda tid: os.path.isdir(os.path.join(root_dir, tid))
        )].reset_index(drop=True)

        self.target_depth, self.target_height, self.target_width = target_shape

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        row = self.labels.iloc[idx]
        tomo_id = row['tomo_id']
        tomo_path = os.path.join(self.root_dir, tomo_id)

        # Load and resize all slices
        slice_files = sorted([
            f for f in os.listdir(tomo_path)
            if f.lower().endswith(('.jpg', '.png'))
        ])

        resized_slices = []
        for fname in slice_files:
            img_path = os.path.join(tomo_path, fname)
            img = Image.open(img_path).convert('L')
            img = img.resize((self.target_width, self.target_height))
            img_np = np.array(img, dtype=np.float32) / 255.0
            resized_slices.append(img_np)

        # Stack to [D, H, W]
        volume = np.stack(resized_slices, axis=0)

        # === Handle depth: pad or crop to self.target_depth ===
        depth = volume.shape[0]
        if depth < self.target_depth:
            pad_before = (self.target_depth - depth) // 2
            pad_after = self.target_depth - depth - pad_before
            volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant', constant_values=0)
        elif depth > self.target_depth:
            start = (depth - self.target_depth) // 2
            volume = volume[start:start + self.target_depth]

        # Final shape [1, D, H, W]
        volume = np.expand_dims(volume, axis=0)
        volume_tensor = torch.tensor(volume, dtype=torch.float32)

        # Label: 3D motor position
        target = row[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values.astype(np.float32)
        target_tensor = torch.tensor(target)

        return volume_tensor, target_tensor



from monai.networks.nets import resnet

def get_model():
    model = resnet.resnet18(spatial_dims=3, n_input_channels=1, num_classes=3)  # For x, y, z regression
    return model



from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim

# Dataset & loader
train_dataset = TomoDataset(
    root_dir="/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train",
    label_csv="/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv",
    target_shape=(64, 64, 64)
)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)

# Model, loss, optimizer
model = get_model().cuda()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
epochs = 5
model.train()
for epoch in range(epochs):
    epoch_loss = 0
    for i, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(train_loader):.4f}")
torch.save(model.state_dict(), "best_model.pth")    



class TestTomoDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, target_shape=(64, 128, 128)):
        self.root_dir = root_dir
        self.tomo_ids = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.tomo_ids.sort()  # Ensure consistent ordering
        self.target_depth, self.target_height, self.target_width = target_shape

    def __len__(self):
        return len(self.tomo_ids)

    def __getitem__(self, idx):
        tomo_id = self.tomo_ids[idx]
        tomo_path = os.path.join(self.root_dir, tomo_id)

        # Load and resize all slices
        slice_files = sorted([
            f for f in os.listdir(tomo_path)
            if f.lower().endswith(('.jpg', '.png'))
        ])

        resized_slices = []
        for fname in slice_files:
            img_path = os.path.join(tomo_path, fname)
            img = Image.open(img_path).convert('L')
            img = img.resize((self.target_width, self.target_height))
            img_np = np.array(img, dtype=np.float32) / 255.0
            resized_slices.append(img_np)

        volume = np.stack(resized_slices, axis=0)

        # Pad/crop to target depth
        depth = volume.shape[0]
        if depth < self.target_depth:
            pad_before = (self.target_depth - depth) // 2
            pad_after = self.target_depth - depth - pad_before
            volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
        elif depth > self.target_depth:
            start = (depth - self.target_depth) // 2
            volume = volume[start:start + self.target_depth]

        volume = np.expand_dims(volume, axis=0)  # [1, D, H, W]
        volume_tensor = torch.tensor(volume, dtype=torch.float32)

        return volume_tensor, tomo_id



test_dataset=TestTomoDataset(
    root_dir='/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test',
    target_shape=(64,64,64)
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
model = get_model().cuda()
model.load_state_dict(torch.load("best_model.pth"))  # load your trained weights
model.eval()

# Predict
results = []
with torch.no_grad():
    for volume, tomo_id in test_loader:
        volume = volume.cuda()
        pred = model(volume)  # [1, 3]
        pred = pred.squeeze().cpu().numpy()  # [3]
        results.append({
            "tomo_id": tomo_id[0],
            "Motor axis 0": pred[0],
            "Motor axis 1": pred[1],
            "Motor axis 2": pred[2]
        })



submission_df = pd.DataFrame(results)
submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())

