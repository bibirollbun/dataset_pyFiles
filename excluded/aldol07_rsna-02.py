from glob import glob
import pydicom as dicom #for dicom files
import nibabel as nib #for nii files

import os
import shutil
import gc
from collections import defaultdict
from typing import Tuple, List

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import polars as pl
import pydicom
from scipy import ndimage
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import kaggle_evaluation.rsna_inference_server

import warnings
warnings.filterwarnings('ignore')


train_images = glob("/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647/*")


path = '/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381.nii'


train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
label_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")


plt.style.use('default')
fig, axes = plt.subplots(4,4, figsize=(12,12))
train_images
for i, ax in enumerate(axes.reshape(-1)):
    img_path = train_images[i]
    img = dicom.dcmread(img_path)  
    ax.imshow(img.pixel_array)
plt.show()


img = nib.load(path).get_fdata()
img.shape


plt.style.use('default')
fig, axes = plt.subplots(4,4, figsize=(12,12))
for i, ax in enumerate(axes.reshape(-1)):
    ax.imshow(img[:,:,1 + i])
plt.show()


# Check class imbalance
print("Aneurysm Present: 1 =", train_df['Aneurysm Present'].mean()*100, "%")
# Check modality distribution
print(train_df['Modality'].value_counts())
# Check location-wise prevalence (critical for multi-label)
locations = [col for col in train_df.columns if 'Artery' in col or 'Communicating' in col]
print(train_df[locations].sum() / len(train_df))


train_df.head()


train_df['PatientAge'].describe()


train_df['PatientAge'] = train_df['PatientAge'].astype(int)
plt.figure(figsize=(10,6))
sns.histplot(train_df['PatientAge'], bins=20, kde=False, color=sns.color_palette("rocket")[4])  
plt.xlabel('Patient Age')
plt.ylabel('Count')
plt.title('Distribution of Patient Age')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

# Create cross-tabulation with proportions or counts
ctab = pd.crosstab(train_df['PatientSex'], train_df['Aneurysm Present'])

# Plot grouped bar chart
ctab.plot(kind='bar', 
          color=sns.color_palette("pastel"), 
          figsize=(8, 6), 
          width=0.8)

# Labels and title
plt.xlabel('Patient Sex')
plt.ylabel('Count')
plt.title('Aneurysm Presence by Patient Sex')
plt.legend(title='Aneurysm Present', labels=['No', 'Yes'])
plt.xticks(rotation=0)

# Add value labels on bars (optional, improves readability)
for container in plt.gca().containers:
    plt.bar_label(container, fmt='%d', padding=3)

plt.tight_layout()
plt.show()


# Plot pie chart with counts shown on each slice
train_df['Modality'].value_counts().plot(kind='pie', autopct='%d')

# Optional: Improve layout and title
plt.title('Distribution of Modality')
plt.ylabel('')  # Hide the y-label (default is 'Modality' from pandas)
plt.show()


ID_COL = 'SeriesInstanceUID'

LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

DICOM_TAG_ALLOWLIST = [
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit',
    'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber', 'Modality',
    'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
    'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows',
    'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness',
    'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID',
]


# Model configuration
TARGET_SIZE = (64, 64, 64)  # Reduced size for memory efficiency
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DICOMProcessor:
    """Process DICOM series into normalized 3D volumes"""
    
    def __init__(self, target_size: Tuple[int, int, int] = TARGET_SIZE):
        self.target_size = target_size
        self.scaler = StandardScaler()
    
    def load_dicom_series(self, series_path: str) -> np.ndarray:
        """Load and process a DICOM series into a 3D volume"""
        try:
            # Get all DICOM files
            dicom_files = []
            for root, _, files in os.walk(series_path):
                for file in files:
                    if file.endswith('.dcm'):
                        dicom_files.append(os.path.join(root, file))
            
            if not dicom_files:
                raise ValueError(f"No DICOM files found in {series_path}")
            
            # Load DICOMs 
            dicoms = []
            # Initialize dicoms_with_position here (critical fix)
            dicoms_with_position = []
            for filepath in dicom_files:
                try:
                    ds = pydicom.dcmread(filepath, force=True)
                    if hasattr(ds, 'PixelData'):
                        # Extract Z-position (ImagePositionPatient[2]) with robust error handling
                        z_pos = None
                        if hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) >= 3:
                            try:
                                z_pos = float(ds.ImagePositionPatient[2])
                            except (ValueError, TypeError):
                                # Handle case where ImagePositionPatient has non-numeric values
                                pass
                        dicoms_with_position.append((ds, filepath, z_pos))
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue
            
            if not dicoms:
                raise ValueError(f"No valid DICOM files with pixel data in {series_path}")
            
            # Sort by instance number
            # dicoms.sort(key=lambda x: getattr(x[0], 'InstanceNumber', 0))
            # CRITICAL FIX: Sort by physical Z-position, not InstanceNumber
            # First, filter out any slices without valid Z-position
            valid_slices = [(ds, filepath, z_pos) for ds, filepath, z_pos in dicoms_with_position if z_pos is not None]
            
            if not valid_slices:
                # If no slices have valid Z-position, fall back to InstanceNumber as last resort
                print(f"Warning: No valid ImagePositionPatient found in {series_path}, using InstanceNumber")
                dicoms_with_position.sort(key=lambda x: getattr(x[0], 'InstanceNumber', 0))
                dicoms = [(ds, filepath) for ds, filepath, _ in dicoms_with_position]
            else:
                # Primary method: Sort by physical Z-position
                valid_slices.sort(key=lambda x: x[2])  # Sort by z_pos
                dicoms = [(ds, filepath) for ds, filepath, _ in valid_slices]
            
            # Extract volume
            volume_slices = []
            for ds, _ in dicoms:
                try:
                    # Get pixel array
                    pixel_array = ds.pixel_array.astype(np.float32)
                    
                    # Apply rescale if available
                    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                        slope = float(ds.RescaleSlope)
                        intercept = float(ds.RescaleIntercept)
                        pixel_array = pixel_array * slope + intercept
                    
                    volume_slices.append(pixel_array)
                except Exception as e:
                    print(f"Error processing slice: {e}")
                    continue
            
            if not volume_slices:
                raise ValueError("No valid slices extracted")
            
            # Stack into 3D volume
            volume = np.stack(volume_slices, axis=0)  # Shape: (depth, height, width)
            
            # Normalize and resize
            volume = self.preprocess_volume(volume)
            
            return volume
            
        except Exception as e:
            print(f"Error processing series {series_path}: {e}")
            # Return zeros if processing fails
            return np.zeros(self.target_size, dtype=np.float32)
    
    def preprocess_volume(self, volume: np.ndarray) -> np.ndarray:
        """Preprocess 3D volume: normalize, clip, resize"""
        # Handle potential issues
        if volume.size == 0:
            return np.zeros(self.target_size, dtype=np.float32)
        
        # Clip extreme values (robust to outliers)
        p1, p99 = np.percentile(volume, [1, 99])
        volume = np.clip(volume, p1, p99)
        
        # Normalize to [0, 1]
        volume_min, volume_max = volume.min(), volume.max()
        if volume_max > volume_min:
            volume = (volume - volume_min) / (volume_max - volume_min)
        
        # Resize to target size
        if volume.shape != self.target_size:
            zoom_factors = [
                self.target_size[i] / volume.shape[i] for i in range(3)
            ]
            volume = ndimage.zoom(volume, zoom_factors, order=1)
        
        return volume.astype(np.float32)


class Simple3DCNN(nn.Module):
    """Lightweight 3D CNN for aneurysm detection"""
    
    def __init__(self, num_classes: int = len(LABEL_COLS)):
        super(Simple3DCNN, self).__init__()
        
        # 3D Convolutional layers
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)
        self.conv4 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool3d(2)
        
        # Adaptive pooling to handle variable sizes
        self.adaptive_pool = nn.AdaptiveAvgPool3d((2, 2, 2))
        
        # Fully connected layers
        self.fc1 = nn.Linear(128 * 2 * 2 * 2, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(128, num_classes)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm3d(16)
        self.bn2 = nn.BatchNorm3d(32)
        self.bn3 = nn.BatchNorm3d(64)
        self.bn4 = nn.BatchNorm3d(128)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, depth, height, width)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        # Adaptive pooling
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return torch.sigmoid(x)


class AneurysmDataset(Dataset):
    """Dataset for loading training data"""
    
    def __init__(self, data_df: pd.DataFrame, series_dir: str, processor: DICOMProcessor):
        self.data_df = data_df
        self.series_dir = series_dir
        self.processor = processor
        
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        series_id = row[ID_COL]
        
        # Load volume
        series_path = os.path.join(self.series_dir, series_id)
        volume = self.processor.load_dicom_series(series_path)
        
        # Get labels
        labels = row[LABEL_COLS].values.astype(np.float32)
        
        # Convert to tensor and add channel dimension
        volume_tensor = torch.from_numpy(volume).unsqueeze(0)  # Add channel dim
        labels_tensor = torch.from_numpy(labels)
        
        return volume_tensor, labels_tensor


# Global model and processor
model = None
processor = None

def initialize_model():
    """Initialize model and processor (called once)"""
    global model, processor
    
    if model is not None:
        return
    
    print("Initializing model...")
    processor = DICOMProcessor(TARGET_SIZE)
    model = Simple3DCNN(num_classes=len(LABEL_COLS))
    
    # Load pre-trained weights if available
    try:
        if os.path.exists('/kaggle/input/model_weights.pth'):
            model.load_state_dict(torch.load('/kaggle/input/model_weights.pth', map_location='cpu'))
            print("Loaded pre-trained weights")
        else:
            print("No pre-trained weights found, using random initialization")
    except Exception as e:
        print(f"Error loading weights: {e}")
    
    model.to(DEVICE)
    model.eval()
    print(f"Model initialized on {DEVICE}")


def train_model(series_dir="/kaggle/input/rsna-intracranial-aneurysm-detection/series",
                num_epochs=2, batch_size=4, checkpoint_path="checkpoint.pth"):
    """
    Quick training function for Kaggle GPU (P100).
    Trains on a small subset so you can see output quickly.
    """

    # Load training CSV
    train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")

    # Create processor + model
    processor = DICOMProcessor(TARGET_SIZE)
    model = Simple3DCNN(num_classes=len(LABEL_COLS))
    model.to(DEVICE)

    # Resume training if checkpoint exists
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    # Use only 50 random samples for fast training
    dataset = AneurysmDataset(train_df.sample(50, random_state=42), series_dir, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_idx, (volumes, labels) in enumerate(dataloader):
            volumes, labels = volumes.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✅ Checkpoint saved to {checkpoint_path}")



train_model(num_epochs=2, batch_size=4)



def predict(series_path, model=None):
    """
    Loads a DICOM series and outputs model predictions.
    Returns a numpy array of probabilities.
    """
    global processor
    if model is None:
        model = Simple3DCNN(num_classes=len(LABEL_COLS))
        model.load_state_dict(torch.load("checkpoint.pth", map_location=DEVICE))
        model.to(DEVICE)
        model.eval()

    if processor is None:
        processor = DICOMProcessor(TARGET_SIZE)

    # Process DICOM into volume
    try:
        volume = processor.load_dicom_series(series_path)
    except Exception as e:
        print(f"Failed to process {series_path}: {e}")
        volume = np.zeros(TARGET_SIZE, dtype=np.float32)

    # Convert to tensor and add batch dimension
    volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        preds = model(volume_tensor).cpu().numpy().flatten()

    return preds



# -----------------------------
# Quick submission to appear on leaderboard
# -----------------------------
import os
import torch
import numpy as np
import pandas as pd

# Use your existing LABEL_COLS and TARGET_SIZE
# DEVICE already defined
# Simple3DCNN and DICOMProcessor already defined

# Load trained model
model = Simple3DCNN(num_classes=len(LABEL_COLS))
model.load_state_dict(torch.load("checkpoint.pth", map_location=DEVICE))
model.to(DEVICE)
model.eval()

# Initialize processor
processor = DICOMProcessor(TARGET_SIZE)

# Function to predict one series
def predict_fast(series_path):
    try:
        volume = processor.load_dicom_series(series_path)
    except:
        volume = np.zeros(TARGET_SIZE, dtype=np.float32)
    vol_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        preds = model(vol_tensor).cpu().numpy().flatten()
    return preds

# Directory with series
series_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"
series_ids = os.listdir(series_dir)

# Limit for quick leaderboard submission
LIMIT = 200  # Adjust to speed up; you can increase later if needed
series_ids = series_ids[:LIMIT]

submission = []
for sid in series_ids:
    spath = os.path.join(series_dir, sid)
    preds = predict_fast(spath)
    submission.append([sid] + preds.tolist())

# Create submission DataFrame
cols = ["ID"] + LABEL_COLS
submission_df = pd.DataFrame(submission, columns=cols)

# Save in Kaggle-required format
submission_df.to_parquet("submission.parquet", index=False)
print(f"✅ Quick submission file saved: submission.parquet")


