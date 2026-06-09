# Install ultralytics (YOLOv8), pydicom, and OpenCV
!pip install --upgrade pip
!pip install ultralytics pydicom opencv-python-headless


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

label_df.head()

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

def train_model(series_dir: str = "/kaggle/input/rsna-intracranial-aneurysm-detection/series", num_epochs: int = 100, batch_size: int = 32):
    """Training function (for reference - would be run separately)"""
    
    # Load training data
    train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
    
    # Initialize components
    processor = DICOMProcessor(TARGET_SIZE)
    model = Simple3DCNN(num_classes=len(LABEL_COLS))
    model.to(DEVICE)
    
    # Create dataset and dataloader
    dataset = AneurysmDataset(train_df, series_dir, processor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2)
    
    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (volumes, labels) in enumerate(dataloader):
            volumes, labels = volumes.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(volumes)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        scheduler.step(avg_loss)
        print(f'Epoch {epoch+1}/{num_epochs} completed, Average Loss: {avg_loss:.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'model_weights.pth')
    print("Model saved to model_weights.pth")

# train_model()

def predict(series_path: str) -> pl.DataFrame:
    """Make prediction for a single series"""
    
    # Initialize model on first call
    initialize_model()
    
    series_id = os.path.basename(series_path)
    
    try:
        # Process the DICOM series
        volume = processor.load_dicom_series(series_path)
        
        # Convert to tensor and add batch dimension
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        volume_tensor = volume_tensor.to(DEVICE)
        
        # Make prediction
        with torch.no_grad():
            predictions = model(volume_tensor)
            predictions = predictions.cpu().numpy().flatten()
        
        # Create result DataFrame
        result_data = [[series_id] + predictions.tolist()]
        result_df = pl.DataFrame(
            data=result_data,
            schema=[ID_COL] + LABEL_COLS,
            orient='row'
        )
        
        # Clean up memory
        del volume_tensor
        torch.cuda.empty_cache()
        gc.collect()
        
    except Exception as e:
        print(f"Error predicting for series {series_id}: {e}")
        # Return baseline predictions (0.5 for all classes)
        result_data = [[series_id] + [0.5] * len(LABEL_COLS)]
        result_df = pl.DataFrame(
            data=result_data,
            schema=[ID_COL] + LABEL_COLS,
            orient='row'
        )
    
    # Mandatory cleanup
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    return result_df.drop(ID_COL)


inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))










