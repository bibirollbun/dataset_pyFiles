from IPython.display import Image, display
display(Image('/kaggle/input/photo-1/1.png'))


from IPython.display import Image, display
display(Image('/kaggle/input/photo-2/2.png'))


import os
import pydicom
import matplotlib.pyplot as plt
import pandas as pd
import nibabel as nib
import numpy as np
import seaborn as sns

sns.set(style="whitegrid")



### Segmentation Mask
seg_path = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/1.2.826.0.1.3680043.8.498.10759842474698331813589731619457567641_cowseg.nii"
seg = nib.load(seg_path)
seg_data = seg.get_fdata()

plt.imshow(seg_data[:, :, seg_data.shape[2] // 2], cmap='gray')
plt.title("Middle slice of aneurysm mask (.nii)")
plt.axis('off')
plt.show()



study_uid = "1.2.826.0.1.3680043.8.498.10009383108068795488741533244914370182"
folder = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{study_uid}"

slices = [pydicom.dcmread(os.path.join(folder, f)) for f in os.listdir(folder)]
slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
volume = np.stack([s.pixel_array for s in slices])

plt.imshow(volume[len(volume)//2], cmap='gray')
plt.title("Middle slice of DICOM series")
plt.axis('off')
plt.show()



localizers_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")
series_id = "1.2.826.0.1.3680043.8.498.10491885999343016971277789732392506995"
localizers_series = localizers_df[localizers_df['SeriesInstanceUID'] == series_id]

if not localizers_series.empty:
    first = localizers_series.iloc[0]
    sop_uid = first['SOPInstanceUID']
    coord = eval(first['coordinates'])

    slices = [pydicom.dcmread(os.path.join(f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_id}", f)) 
              for f in os.listdir(f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{series_id}")]
    target_slice = next((s for s in slices if s.SOPInstanceUID == sop_uid), None)

    if target_slice:
        img = target_slice.pixel_array
        plt.imshow(img, cmap='gray')
        plt.annotate('Aneurysm', xy=(coord['x'], coord['y']),
                     xytext=(coord['x']+10, coord['y']-10),
                     arrowprops=dict(color='red', lw=1))
        plt.title("Annotated aneurysm location")
        plt.axis('off')
        plt.show()



train_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
localizer_df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv")

print("Train CSV Shape:", train_df.shape)
print("Localizer CSV Shape:", localizer_df.shape)

# Preview training labels
train_df.head()



num_positive = train_df['Aneurysm Present'].sum()
print(f"âœ… Number of aneurysm-positive cases: {num_positive}")


from IPython.display import Image, display
display(Image('/kaggle/input/photo-3/3.png'))


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='Aneurysm Present', palette='Reds')
plt.title("Distribution of Aneurysm Presence")
plt.xlabel("Aneurysm Present (0 = No, 1 = Yes)")
plt.ylabel("Count")
plt.show()

# Calculate percentage
aneurysm_rate = train_df['Aneurysm Present'].mean() * 100
print(f"ğŸ’‰ Aneurysm present in {aneurysm_rate:.2f}% of cases.")



anatomy_cols = [col for col in train_df.columns if col not in 
                ['SeriesInstanceUID','Modality','PatientAge','PatientSex','Aneurysm Present']]

vessel_counts = train_df[anatomy_cols].sum().sort_values(ascending=False)

# Plot vessel frequency
plt.figure(figsize=(10,6))
sns.barplot(x=vessel_counts.values, y=vessel_counts.index, palette="mako")
plt.title("Frequency of Aneurysms by Vessel Location")
plt.xlabel("Number of Cases")
plt.ylabel("Vessel Location")
plt.show()

pd.DataFrame({"Vessel Location": vessel_counts.index, "Count": vessel_counts.values})



# Age distribution
plt.figure(figsize=(8,4))
sns.histplot(train_df['PatientAge'], bins=30, kde=True, color="teal")
plt.title("Patient Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()

# Sex distribution
plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='PatientSex', palette="pastel")
plt.title("Patient Sex Distribution")
plt.xlabel("Sex (M/F)")
plt.ylabel("Count")
plt.show()

# Age vs Aneurysm Presence
plt.figure(figsize=(8,4))
sns.boxplot(data=train_df, x='Aneurysm Present', y='PatientAge', palette='coolwarm')
plt.title("Age vs Aneurysm Presence")
plt.xlabel("Aneurysm Present (0 = No, 1 = Yes)")
plt.ylabel("Age")
plt.show()



plt.figure(figsize=(6,4))
sns.countplot(data=train_df, x='Modality', palette='Blues')
plt.title("Distribution of Imaging Modalities")
plt.xlabel("Imaging Modality")
plt.ylabel("Count")
plt.show()

modality_dist = train_df['Modality'].value_counts(normalize=True) * 100
print("Modality distribution (%):")
display(modality_dist)



import pydicom
import os
import random

aneurysm_scans = train_df[train_df['Aneurysm Present'] == 1]['SeriesInstanceUID'].tolist()
example_uid = random.choice(aneurysm_scans)
example_path = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{example_uid}"

# List files and load a mid-slice
dcm_files = sorted(os.listdir(example_path))
dcm = pydicom.dcmread(f"{example_path}/{dcm_files[len(dcm_files)//2]}")
img = dcm.pixel_array

plt.figure(figsize=(6,6))
plt.imshow(img, cmap='gray')
plt.title(f"Sample DICOM Scan\nSeries: {example_uid} | Modality: {dcm.Modality}")
plt.axis("off")
plt.show()



coords = localizer_df[localizer_df['SeriesInstanceUID'] == example_uid]

print(f"ğŸ”´ Found {len(coords)} aneurysm localization points for this scan.")

# Overlay aneurysm points on the DICOM slice
plt.figure(figsize=(6,6))
plt.imshow(img, cmap='gray')
for _, row in coords.iterrows():
    x, y = eval(row['coordinates']) 
    plt.scatter(x, y, color='red', s=40, label='Aneurysm')
plt.title("Aneurysm Localization Overlay")
plt.axis("off")
plt.show()



import os
import pandas as pd

series_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'
series_folders = [f for f in os.listdir(series_dir) if os.path.isdir(os.path.join(series_dir, f))]

series_counts = []
for folder in series_folders:
    dcm_files = os.listdir(os.path.join(series_dir, folder))
    series_counts.append({'SeriesInstanceUID': folder, 'NumSlices': len(dcm_files)})

df_series = pd.DataFrame(series_counts)
df_series.describe()


import pydicom

sample_path = os.path.join(series_dir, series_folders[0])
sample_file = os.listdir(sample_path)[0]
dcm = pydicom.dcmread(os.path.join(sample_path, sample_file))

print(f"Orientation: {dcm.ImageOrientationPatient}")
print(f"Voxel spacing: {dcm.PixelSpacing}")
print(f"Slice Thickness: {dcm.SliceThickness}")


import matplotlib.pyplot as plt
import matplotlib.widgets as widgets
import numpy as np

def load_series(series_path):
    files = sorted(os.listdir(series_path), key=lambda x: pydicom.dcmread(os.path.join(series_path, x)).InstanceNumber)
    images = [pydicom.dcmread(os.path.join(series_path, f)).pixel_array for f in files]
    return np.stack(images)

volume = load_series(os.path.join(series_dir, series_folders[0]))

# Scrollable plot
from ipywidgets import interact
@interact(slice=(0, volume.shape[0]-1))
def show_slice(slice=0):
    plt.imshow(volume[slice], cmap='gray')
    plt.axis('off')
    plt.show()


import os
import pydicom

# Count slices and shape per series
dicom_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'
series_stats = []

for series_id in os.listdir(dicom_dir)[:50]:  # limit for speed
    series_path = os.path.join(dicom_dir, series_id)
    files = os.listdir(series_path)
    num_slices = len(files)
    sample_dcm = pydicom.dcmread(os.path.join(series_path, files[0]))
    shape = (sample_dcm.Rows, sample_dcm.Columns)
    series_stats.append({"SeriesInstanceUID": series_id, "Slices": num_slices, "Shape": shape})

pd.DataFrame(series_stats).value_counts("Shape").plot(kind="barh", title="Common Image Resolutions")


voxel_data = []

for series_id in os.listdir(dicom_dir)[:50]:
    slices = []
    for f in sorted(os.listdir(os.path.join(dicom_dir, series_id))):
        path = os.path.join(dicom_dir, series_id, f)
        dcm = pydicom.dcmread(path)
        slices.append(dcm)

    try:
        spacing = slices[0].PixelSpacing
        thickness = float(slices[0].SliceThickness)
        voxel_data.append({
            "SeriesInstanceUID": series_id,
            "PixelSpacingX": spacing[0],
            "PixelSpacingY": spacing[1],
            "SliceThickness": thickness,
            "NumSlices": len(slices)
        })
    except:
        continue

voxel_df = pd.DataFrame(voxel_data)
sns.histplot(voxel_df["SliceThickness"], bins=20)
plt.title("Slice Thickness Distribution")


seg_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations"
seg_series = [f.replace(".nii.gz", "") for f in os.listdir(seg_dir)]
print("Segmented Series:", len(seg_series))

# % of series with segmentation
seg_percent = len(seg_series) / len(os.listdir(dicom_dir)) * 100
print(f"{seg_percent:.2f}% of series have vessel segmentation.")


import os
import shutil
import gc
from collections import defaultdict
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
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

# Competition constants
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
            
            # Load DICOMs and sort by instance number
            dicoms = []
            for filepath in dicom_files:
                try:
                    ds = pydicom.dcmread(filepath, force=True)
                    if hasattr(ds, 'PixelData'):
                        dicoms.append((ds, filepath))
                except Exception as e:
                    print(f"Error reading {filepath}: {e}")
                    continue
            
            if not dicoms:
                raise ValueError(f"No valid DICOM files with pixel data in {series_path}")
            
            # Sort by instance number
            dicoms.sort(key=lambda x: getattr(x[0], 'InstanceNumber', 0))
            
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

def train_model(train_df_path: str, series_dir: "/kaggle/input/rsna-intracranial-aneurysm-detection/series", num_epochs: int = 50, batch_size: int = 32):
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

# Competition server setup
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))

