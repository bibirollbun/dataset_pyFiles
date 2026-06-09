import h5py
import numpy as np
import pandas as pd
import cv2
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as T



f = h5py.File(r'/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5')

train_images = f['images/Train'] # 6 images with id's like S_1,S_2 to S_6
test_image = f['images/Test'] # 1 image with id S_7

train_spots = f['spots/Train'] # this contains spot data for each 6 ids first 2 columns are x and y and column 3 to 37 are cell abundance
test_spot = f['spots/Test']


def extract_patches(h5_file, patch_size=128, val_ratio=0.1):
    images = h5_file['images/Train']
    spots = h5_file['spots/Train']

    train_patches = []
    train_targets = []
    val_patches = []
    val_targets = []

    half = patch_size // 2

    for slide_id in images.keys():
        img = np.array(images[slide_id])  # shape: (H, W, 3)

        # Fix: handle structured array
        spot_data = pd.DataFrame(spots[slide_id][()])

        coords = spot_data.iloc[:, :2].astype(int).values  # x, y
        targets = spot_data.iloc[:, 2:].values  # shape: (num_spots, 35)

        num_spots = coords.shape[0]
        indices = np.arange(num_spots)
        np.random.shuffle(indices)

        split_idx = int((1 - val_ratio) * num_spots)
        train_idx, val_idx = indices[:split_idx], indices[split_idx:]

        for idx in train_idx:
            x, y = coords[idx]
            if x - half < 0 or y - half < 0 or x + half > img.shape[1] or y + half > img.shape[0]:
                continue
            patch = img[y - half:y + half, x - half:x + half]
            if patch.shape[:2] == (patch_size, patch_size):
                train_patches.append(patch)
                train_targets.append(targets[idx])

        for idx in val_idx:
            x, y = coords[idx]
            if x - half < 0 or y - half < 0 or x + half > img.shape[1] or y + half > img.shape[0]:
                continue
            patch = img[y - half:y + half, x - half:x + half]
            if patch.shape[:2] == (patch_size, patch_size):
                val_patches.append(patch)
                val_targets.append(targets[idx])

    return (
        np.array(train_patches), np.array(train_targets),
        np.array(val_patches), np.array(val_targets)
    )



train_patches, train_targets, val_patches, val_targets = extract_patches(f)

print("Train patches:", train_patches.shape)
print("Train targets:", train_targets.shape)
print("Val patches:", val_patches.shape)
print("Val targets:", val_targets.shape)



import torch
from torch.utils.data import Dataset
from torchvision import transforms

class SpotPatchDataset(Dataset):
    def __init__(self, patches, targets):
        self.patches = patches
        self.targets = targets

        self.transform = transforms.Compose([
            transforms.ToTensor(),  # Convert to [0,1] and shape [C,H,W]
            
        ])

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        image = self.patches[idx]
        target = self.targets[idx]
        image = self.transform(image)
        target = torch.tensor(target, dtype=torch.float32)
        return image, target



from torch.utils.data import DataLoader

train_dataset = SpotPatchDataset(train_patches, train_targets)
val_dataset = SpotPatchDataset(val_patches, val_targets)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)



import torch.nn as nn
from torchvision import models

class ResNetRegressor(nn.Module):
    def __init__(self, output_dim=35):
        super(ResNetRegressor, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        
        # Replace final FC layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, output_dim)

    def forward(self, x):
        return self.backbone(x)



model = ResNetRegressor(output_dim=35)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Loss: MSE for regression
criterion = nn.MSELoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)



def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)

def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

    return running_loss / len(dataloader.dataset)



num_epochs = 5

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss = validate(model, val_loader, criterion, device)

    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")



def extract_test_patches(h5_file, patch_size=128):
    test_img = np.array(h5_file['images/Test']['S_7'])
    test_spot_data = pd.DataFrame(np.array(h5_file['spots/Test']['S_7']))  # shape: (num_spots, 37)
    
    coords = test_spot_data.iloc[:, :2].astype(int).values  # x, y
    half = patch_size // 2

    patches = []
    valid_coords = []

    for coord in coords:
        x, y = coord
        if x - half < 0 or y - half < 0 or x + half > test_img.shape[1] or y + half > test_img.shape[0]:
            continue
        patch = test_img[y - half:y + half, x - half:x + half]
        if patch.shape[:2] == (patch_size, patch_size):
            patches.append(patch)
            valid_coords.append(coord)

    return np.array(patches), np.array(valid_coords)



test_patches, test_coords = extract_test_patches(f)
test_patches_tensor = torch.tensor(test_patches).permute(0, 3, 1, 2).float()   # (N, 3, 224, 224)




model.eval()
predictions = []

with torch.no_grad():
    for i in range(0, len(test_patches_tensor), 32):
        batch = test_patches_tensor[i:i+32].to(device)
        outputs = model(batch).cpu().numpy()
        predictions.append(outputs)

predictions = np.vstack(predictions)  # shape: (num_valid_spots, 35)



df_preds = pd.DataFrame(predictions, columns=[f'cell_type_{i+1}' for i in range(35)])
df_preds.insert(0, 'ID', range(len(df_preds)))



df_preds


df_preds.to_csv('submission.csv', index=False)

