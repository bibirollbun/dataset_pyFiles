# Import necessary libraries
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision.transforms import Compose, RandomRotation, RandomHorizontalFlip, RandomVerticalFlip
from PIL import Image
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
TARGET_VOXEL_SPACING = 10  # Angstroms per voxel
SIGMA_FACTOR = 3  # 3σ covers the threshold distance
CONFIDENCE_THRESHOLD = 0.5  # Tune based on validation
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Load training labels
train_labels = pd.read_csv("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv")


# Advanced Data Augmentation
def get_augmentations():
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.GaussianBlur(p=0.2),
        A.RandomBrightnessContrast(p=0.2),
        ToTensorV2()
    ])



# Verify column names
print("Columns in train_labels.csv:", train_labels.columns)

# If shape columns are missing, calculate them from the tomogram data
if 'Array shape axis 0' not in train_labels.columns:
    print("Shape columns missing. Calculating shape from tomogram data...")
    train_labels['Array shape axis 0'] = 0  # Placeholder, will be updated
    train_labels['Array shape axis 1'] = 0  # Placeholder, will be updated
    train_labels['Array shape axis 2'] = 0  # Placeholder, will be updated




# Remove 2D augmentations and replace with 3D-compatible ones
def get_augmentations():
    return A.Compose([
        # Add 3D-compatible augmentations here
        # Example: Random rotation around the Z-axis
        ToTensorV2()
    ])
# Configuration
TARGET_SHAPE = (128, 128, 128)  # Fixed shape for all tomograms

# Dataset class
class TomogramDataset(Dataset):
    def __init__(self, root_dir, labels_df, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.tomo_ids = labels_df['tomo_id'].unique()
        self.labels_df = labels_df.set_index('tomo_id')
        self.avg_voxel_spacing = labels_df['Voxel spacing'].mean()

    def __len__(self):
        return len(self.tomo_ids)

    def __getitem__(self, idx):
        tomo_id = self.tomo_ids[idx]
        tomo_dir = os.path.join(self.root_dir, tomo_id)
        
        # Load and stack slices
        slices = []
        slice_files = sorted(os.listdir(tomo_dir), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for slice_file in slice_files:
            img = Image.open(os.path.join(tomo_dir, slice_file))
            slices.append(np.array(img))
        volume = np.stack(slices, axis=0)  # Shape (Z, Y, X)
        volume = volume.astype(np.float32)
        
        # Normalize
        volume = (volume - volume.mean()) / volume.std()
        
        # Get labels and metadata
        motor_coords = []
        if tomo_id in self.labels_df.index:
            records = self.labels_df.loc[[tomo_id]]
            motor_coords = records[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values
            voxel_spacing = records['Voxel spacing'].iloc[0]
            
            # Calculate shape if missing
            if 'Array shape axis 0' not in records.columns:
                original_shape = volume.shape
            else:
                original_shape = (records['Array shape axis 0'].iloc[0],
                                 records['Array shape axis 1'].iloc[0],
                                 records['Array shape axis 2'].iloc[0])
        else:
            voxel_spacing = self.avg_voxel_spacing
            original_shape = volume.shape
        
        # Ensure voxel_spacing is valid
        if voxel_spacing <= 0:
            print(f"Warning: Invalid voxel_spacing ({voxel_spacing}) for tomogram {tomo_id}. Using default value.")
            voxel_spacing = TARGET_VOXEL_SPACING
        
        # Calculate scale_factor
        scale_factor = voxel_spacing / TARGET_VOXEL_SPACING
        if scale_factor <= 0:
            print(f"Warning: Invalid scale_factor ({scale_factor}) for tomogram {tomo_id}. Using default value.")
            scale_factor = 1.0
        
        # Calculate new_shape
        new_shape = tuple(int(d * scale_factor) for d in original_shape)
        if any(d <= 0 for d in new_shape):
            print(f"Warning: Invalid new_shape {new_shape} for tomogram {tomo_id}. Using original shape.")
            new_shape = original_shape
        
        # Debugging: Print metadata
        print(f"Tomogram {tomo_id}: voxel_spacing={voxel_spacing}, original_shape={original_shape}, scale_factor={scale_factor}, new_shape={new_shape}")
        
        # Resample volume to target voxel spacing
        try:
            resampled_volume = resize(volume, new_shape, order=1, preserve_range=True)
        except Exception as e:
            print(f"Error resizing tomogram {tomo_id}: {e}. Using original volume.")
            resampled_volume = volume
        
        # Resize to fixed shape
        resampled_volume = resize(resampled_volume, TARGET_SHAPE, order=1, preserve_range=True)
        
        # Adjust coordinates for resampled volume
        resampled_coords = []
        for coord in motor_coords:
            resampled_coord = [c * scale_factor for c in coord]
            resampled_coords.append(resampled_coord)
        
        # Generate target heatmap
        heatmap = np.zeros_like(resampled_volume)
        sigma = (1000 / TARGET_VOXEL_SPACING) / SIGMA_FACTOR
        for coord in resampled_coords:
            z, y, x = coord
            z = int(round(z))
            y = int(round(y))
            x = int(round(x))
            if 0 <= z < resampled_volume.shape[0] and \
               0 <= y < resampled_volume.shape[1] and \
               0 <= x < resampled_volume.shape[2]:
                heatmap += self.create_gaussian(heatmap.shape, (z, y, x), sigma)
        
        # Apply transformations (if any)
        if self.transform:
            augmented = self.transform(image=resampled_volume, mask=heatmap)
            resampled_volume, heatmap = augmented['image'], augmented['mask']
        
        return (torch.tensor(resampled_volume, dtype=torch.float32).unsqueeze(0),
                torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0))

    def create_gaussian(self, shape, center, sigma):
        zz, yy, xx = np.indices(shape)
        d_sq = (zz - center[0])**2 + (yy - center[1])**2 + (xx - center[2])**2
        gaussian = np.exp(-d_sq / (2 * sigma**2))
        return gaussian

# Custom collate function to handle variable-sized tensors
def custom_collate(batch):
    volumes = [item[0] for item in batch]
    heatmaps = [item[1] for item in batch]
    
    # Stack volumes and heatmaps
    volumes = torch.stack(volumes, dim=0)
    heatmaps = torch.stack(heatmaps, dim=0)
    
    return volumes, heatmaps

# Create dataset and dataloader
dataset = TomogramDataset(root_dir="/kaggle/input/train", labels_df=train_labels, transform=None)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate)



# Advanced 3D U-Net with residual connections
class ResidualUNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(ResidualUNet3D, self).__init__()
        features = init_features
        # Encoder
        self.encoder1 = self._block(in_channels, features, name="enc1")
        self.pool1 = nn.MaxPool3d(2, 2)
        self.encoder2 = self._block(features, features*2, name="enc2")
        self.pool2 = nn.MaxPool3d(2, 2)
        self.encoder3 = self._block(features*2, features*4, name="enc3")
        self.pool3 = nn.MaxPool3d(2, 2)
        self.encoder4 = self._block(features*4, features*8, name="enc4")
        self.pool4 = nn.MaxPool3d(2, 2)
        
        # Bottleneck
        self.bottleneck = self._block(features*8, features*16, name="bottleneck")
        
        # Decoder
        self.upconv4 = nn.ConvTranspose3d(features*16, features*8, 2, 2)
        self.decoder4 = self._block(features*16, features*8)
        self.upconv3 = nn.ConvTranspose3d(features*8, features*4, 2, 2)
        self.decoder3 = self._block(features*8, features*4)
        self.upconv2 = nn.ConvTranspose3d(features*4, features*2, 2, 2)
        self.decoder2 = self._block(features*4, features*2)
        self.upconv1 = nn.ConvTranspose3d(features*2, features, 2, 2)
        self.decoder1 = self._block(features*2, features)
        
        self.conv = nn.Conv3d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        enc5 = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(enc5)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return torch.sigmoid(self.conv(dec1))
    
    @staticmethod
    def _block(in_channels, features, name=None):
        return nn.Sequential(
            nn.Conv3d(in_channels, features, kernel_size=3, padding=1),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True),
            nn.Conv3d(features, features, kernel_size=3, padding=1),
            nn.BatchNorm3d(features),
            nn.ReLU(inplace=True)
        )



# Focal Loss for class imbalance
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


# Training loop
def train_model(model, dataloader, criterion, optimizer, scheduler, device):
    model.train()
    running_loss = 0.0
    for inputs, targets in tqdm(dataloader, desc="Training"):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    scheduler.step(running_loss / len(dataloader))
    return running_loss / len(dataloader)



# Main execution
if __name__ == "__main__":
    # Load dataset
    dataset = TomogramDataset(root_dir="/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train", labels_df=train_labels, transform=get_augmentations())
    train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize model, optimizer, and scheduler
    model = ResidualUNet3D().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    criterion = FocalLoss()
    
    # Train model
    for epoch in range(EPOCHS):
        print(f"Epoch {epoch+1}/{EPOCHS}")
        train_loss = train_model(model, train_loader, criterion, optimizer, scheduler, DEVICE)
        print(f"Train Loss: {train_loss}")
    
    # Save model
    torch.save(model.state_dict(), "residual_unet3d.pth")

    # Inference and submission
    test_dir = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test"
    test_tomo_ids = os.listdir(test_dir)
    predictions = []

    for tomo_id in test_tomo_ids:
        tomo_path = os.path.join(test_dir, tomo_id)
        slices = []
        slice_files = sorted(os.listdir(tomo_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))
        for slice_file in slice_files:
            img = Image.open(os.path.join(tomo_path, slice_file))
            slices.append(np.array(img))
        volume = np.stack(slices, axis=0).astype(np.float32)
        volume = (volume - volume.mean()) / volume.std()
        
        # Predict
        with torch.no_grad():
            input_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
            output = model(input_tensor).cpu().numpy().squeeze()
        
        # Find peak
        max_val = np.max(output)
        if max_val < CONFIDENCE_THRESHOLD:
            predictions.append({'tomo_id': tomo_id, 'coords': (-1, -1, -1)})
            continue
        
        # Get peak coordinates
        z, y, x = np.unravel_index(np.argmax(output), output.shape)
        predictions.append({'tomo_id': tomo_id, 'coords': (z, y, x)})
    



    # Create submission
    submission = pd.DataFrame([{'tomo_id': p['tomo_id'], 
                                'Motor axis 0': p['coords'][0], 
                                'Motor axis 1': p['coords'][1], 
                                'Motor axis 2': p['coords'][2]} for p in predictions])
    submission.to_csv("submission.csv", index=False)


