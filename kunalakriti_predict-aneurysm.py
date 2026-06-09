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


# Basic imports
import os
import numpy as np
import pydicom
from scipy import ndimage
import torch
import torch.nn as nn
from torch import sigmoid
import polars as pl

# Competition-specific imports
import kaggle_evaluation.rsna_inference_server

# Our model architecture (must match training)
class Simple3DCNN(nn.Module):
    def __init__(self, num_labels=14):
        super(Simple3DCNN, self).__init__()
        
        # Same architecture as in training
        self.conv1 = nn.Conv3d(1, 8, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool3d(2)
        
        self.conv2 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool3d(2)
        
        self.conv3 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool3d(2)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32 * 8 * 8 * 8, 128)
        self.fc2 = nn.Linear(128, num_labels)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = self.pool3(torch.relu(self.conv3(x)))
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Constants (must match preprocessing)
TARGET_SIZE = (64, 64, 64)
CTA_WINDOW = (300.0, 700.0)
MRI_Z_CLIP = 3.0
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

# Check if we're running in competition environment
IS_COMPETITION = os.getenv('KAGGLE_IS_COMPETITION_RERUN') is not None
print(f"Running in {'competition' if IS_COMPETITION else 'local test'} mode")


# Simplified DICOM processor for inference
def _safe_zoom(volume: np.ndarray, zoom_factors: tuple, order: int = 1) -> np.ndarray:
    """Safely resize a volume."""
    volume = np.nan_to_num(volume, copy=False)
    zf = tuple(max(1e-6, f) for f in zoom_factors)
    if len(zf) != volume.ndim:
        if len(zf) > volume.ndim:
            zf = zf[:volume.ndim]
        else:
            zf = (1.0,) * (volume.ndim - len(zf)) + zf
    return ndimage.zoom(volume, zf, order=1)

def _resize_slice(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2D slice."""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float32, copy=False)
    zy = out_h / max(h, 1)
    zx = out_w / max(w, 1)
    return _safe_zoom(arr, (zy, zx), order=1).astype(np.float32, copy=False)

def process_dicom_series(series_path: str) -> np.ndarray:
    """Process a single DICOM series into a normalized 64x64x64 volume."""
    try:
        # Collect DICOM files
        dicoms = []
        for root, _, files in os.walk(series_path):
            for f in files:
                if f.endswith(".dcm"):
                    try:
                        ds = pydicom.dcmread(os.path.join(root, f), force=True)
                        if hasattr(ds, "PixelData"):
                            dicoms.append(ds)
                    except:
                        continue
        
        if not dicoms:
            raise ValueError("No valid DICOM files found")
        
        # Sort slices
        try:
            orient = np.array(dicoms[0].ImageOrientationPatient, dtype=np.float32)
            row = orient[:3]
            col = orient[3:]
            normal = np.cross(row, col)
            def sort_key(ds):
                ipp = np.array(getattr(ds, "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
                return float(np.dot(ipp, normal))
            dicoms = sorted(dicoms, key=sort_key)
        except:
            dicoms = sorted(dicoms, key=lambda ds: getattr(ds, "InstanceNumber", 0))
        
        # Get spacing
        try:
            dy, dx = map(float, dicoms[0].PixelSpacing)
        except:
            dy, dx = 1.0, 1.0
            
        zs = []
        for i in range(1, len(dicoms)):
            p0 = np.array(getattr(dicoms[i-1], "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
            p1 = np.array(getattr(dicoms[i], "ImagePositionPatient", [0, 0, 0]), dtype=np.float32)
            d = np.linalg.norm(p1 - p0)
            if d > 0:
                zs.append(d)
        dz = float(np.median(zs)) if zs else float(getattr(dicoms[0], "SliceThickness", 1.0))
        
        # Choose base shape
        shapes = []
        for ds in dicoms:
            try:
                h, w = int(ds.Rows), int(ds.Columns)
            except:
                arr = ds.pixel_array
                h, w = arr.shape[-2], arr.shape[-1]
            shapes.append((h, w))
        base_h, base_w = max(shapes, key=shapes.count)
        
        # Process slices
        vol_slices = []
        modality_tag = (getattr(dicoms[0], "Modality", "") or "").upper()
        
        for ds in dicoms:
            arr = ds.pixel_array
            if arr.ndim >= 3:
                h, w = arr.shape[-2], arr.shape[-1]
                n = int(np.prod(arr.shape[:-2]))
                arr = arr.reshape(n, h, w)
                frames = arr
            else:
                frames = arr[np.newaxis, ...]
            
            for sl in frames:
                sl = sl.astype(np.float32)
                
                # Handle MONOCHROME1
                if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                    sl = sl.max() - sl
                
                # Apply rescaling
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                sl = sl * slope + intercept
                sl = np.nan_to_num(sl, copy=False)
                
                # Resize
                sl = _resize_slice(sl, base_h, base_w)
                vol_slices.append(sl)
        
        if not vol_slices:
            raise ValueError("No valid slices extracted")
        
        volume = np.stack(vol_slices, axis=0).astype(np.float32)
        
        # Normalize by modality
        if modality_tag == "CT":
            c, w = CTA_WINDOW
            lo, hi = c - w / 2.0, c + w / 2.0
            volume = np.clip(volume, lo, hi)
            volume = (volume - lo) / (hi - lo + 1e-6)
        else:
            mean = float(volume.mean())
            std = float(volume.std() + 1e-6)
            volume = (volume - mean) / std
            volume = np.clip(volume, -MRI_Z_CLIP, MRI_Z_CLIP)
            volume = (volume + MRI_Z_CLIP) / (2.0 * MRI_Z_CLIP)
        
        # Resample to target size
        tz, ty, tx = TARGET_SIZE
        z, y, x = volume.shape
        volume = _safe_zoom(volume, (tz/z, ty/y, tx/x), order=1).astype(np.float32)
        
        return volume
    
    except Exception as e:
        # Return a zero volume if processing fails
        print(f"Error processing {series_path}: {str(e)}")
        return np.zeros(TARGET_SIZE, dtype=np.float32)


# Path to your trained model
MODEL_PATH = "/kaggle/input/train-model-for-aneurysm-3/best_model.pth"  # Update this path

# Create and load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Simple3DCNN(num_labels=len(LABEL_COLS)).to(device)

try:
    # Try to load the model
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()  # Set to evaluation mode
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model: {str(e)}")
    print("Creating a random model for demonstration (will not score well)")
    model = Simple3DCNN(num_labels=len(LABEL_COLS)).to(device)
    model.eval()


# PREDICT FUNCTION DEFINITION
def predict(series_path: str) -> pl.DataFrame:
    """
    Process a DICOM series and return predictions.
    
    Args:
        series_path: Path to the DICOM series directory
        
    Returns:
        Polars DataFrame with predictions for all 14 labels
    """
    # Extract SeriesInstanceUID from the path
    series_id = os.path.basename(series_path)
    
    # Process the DICOM series
    volume = process_dicom_series(series_path)
    
    # Prepare for model input
    volume_tensor = torch.FloatTensor(volume).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,D,H,W)
    
    # Get predictions
    with torch.no_grad():
        outputs = model(volume_tensor)
        probs = sigmoid(outputs).cpu().numpy()[0]  # Convert to probabilities
    
    # Create DataFrame with predictions
    # Must have exactly the 14 label columns
    predictions = {
        'SeriesInstanceUID': [series_id],
    }
    
    # Add all label predictions
    for i, col in enumerate(LABEL_COLS):
        predictions[col] = [float(probs[i])]
    
    return pl.DataFrame(predictions)


# INFERENCE SERVER
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # This runs when submitting to the competition
    print("Running in competition mode - serving predictions")
    inference_server.serve()
else:
    # This runs when testing locally
    print("Running in local test mode")
    inference_server.run_local_gateway()
    
    # Display the submission file for verification
    submission_path = '/kaggle/working/submission.parquet'
    if os.path.exists(submission_path):
        print("\nLocal test submission results:")
        display(pl.read_parquet(submission_path))
        print("\nSubmission file saved to:", submission_path)
    else:
        print("Warning: Submission file not created. Check your predict function.")

