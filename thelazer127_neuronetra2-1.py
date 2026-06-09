# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import numpy as np
import pydicom
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
import warnings
import gc

# ----------------------------------------------------------------------
# Suppress warnings
# ----------------------------------------------------------------------
warnings.filterwarnings('ignore')

# ----------------------------------------------------------------------
# DICOM PROCESSING CLASS
# ----------------------------------------------------------------------

class DICOMPreprocessorKaggle:
    """
    DICOM preprocessing system for Kaggle Code Competition
    Converts original DICOMPreprocessor logic to single series processing
    """
    
    def __init__(self, target_shape: Tuple[int, int, int] = (32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape
        
    def load_dicom_series(self, series_path: str) -> Tuple[List[pydicom.Dataset], str]:
        """Load DICOM series"""
        series_path = Path(series_path)
        series_name = series_path.name
        
        # Search for DICOM files
        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_path}")
        
        # Load DICOM datasets
        datasets = []
        for filepath in dicom_files:
            try:
                ds = pydicom.dcmread(filepath, force=True)
                datasets.append(ds)
            except Exception:
                # Silently skip bad DICOM files (Robustness)
                continue
        
        if not datasets:
            raise ValueError(f"No valid DICOM files in {series_path}")
            
        return datasets, series_name
    
    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
        """Extract position information for each slice"""
        slice_info = []
        
        for i, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': i,
                'instance_number': getattr(ds, 'InstanceNumber', i),
            }
            
            # Get z-coordinate from ImagePositionPatient and ImageOrientationPatient 
            try:
                ipp = np.array(getattr(ds, 'ImagePositionPatient', None))
                iop = np.array(getattr(ds, 'ImageOrientationPatient', None))
                n_vec = np.cross(iop[:3],iop[3:])
                info['z_position'] = float((ipp*n_vec).sum())
            except Exception:
                info['z_position'] = float(i)
                
            slice_info.append(info)
        
        return slice_info
    
    def sort_slices_by_position(self, slice_info: List[Dict]) -> List[Dict]:
        """Sort slices by z-coordinate"""
        return sorted(slice_info, key=lambda x: x['z_position'])
    
    def get_windowing_params(self, ds: pydicom.Dataset, img: np.ndarray = None) -> Tuple[Optional[float], Optional[float]]:
        """Get windowing parameters based on modality"""
        modality = getattr(ds, 'Modality', 'CT')
        
        if modality == 'CT':
            # Fixed window for CTA [0, 500] HU range for normalization
            return 50.0, 350.0 # Returning nominal center/width for logic path
        elif modality == 'MR':
            return None, None # Use statistical normalization
        else:
            return 50.0, 350.0 # Default to CTA windowing
    
    def apply_windowing_or_normalize(self, img: np.ndarray, center: Optional[float], width: Optional[float]) -> np.ndarray:
        """Apply CTA fixed-window or statistical normalization"""
        
        if center is not None and width is not None:
            # Fixed CTA windowing for CT/CTA
            p1, p99 = 0, 500 # Fixed window [0, 500] HU for robust CTA preprocessing
            
            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)
                return result
            else:
                return np.zeros_like(img, dtype=np.uint8)
        
        else:
            # Statistical normalization (for MR or fallback)
            p1, p99 = np.percentile(img, [1, 99])
            
            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)
                return result
            else:
                # If image has no variation (safety)
                return np.zeros_like(img, dtype=np.uint8)
    
    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        """Extract 2D pixel array and apply DICOM rescaling"""
        
        # Get pixel data
        img = ds.pixel_array.astype(np.float32)
        
        # For 3D volume case (multiple frames) - select middle frame
        if img.ndim == 3:
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]
            
        # Convert color image to grayscale
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            
        # Apply RescaleSlope and RescaleIntercept (Keep this for correctness)
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        
        # NOTE: Your original code hardcoded slope/intercept to 1/0. I'm restoring the DICOM standard
        # behavior for better robustness, as the previous failed submission might have been due
        # to not applying these tags on certain test files.
        if slope != 1 or intercept != 0:
            img = img * float(slope) + float(intercept)
            
        return img
    
    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """Resize 3D volume to target size"""
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        
        if current_shape == target_shape:
            return volume
        
        # 3D resizing using scipy.ndimage
        zoom_factors = [
            target_shape[i] / current_shape[i] for i in range(3)
        ]
        
        # Resize with linear interpolation
        resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        
        # Padding if necessary
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]
        
        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')
            
        return resized_volume[:self.target_depth, :self.target_height, :self.target_width].astype(np.uint8)
    
    def process_series(self, series_path: str) -> np.ndarray:
        """Process DICOM series and return as NumPy array (for Kaggle: no file saving)"""
        try:
            # 1. Load DICOM files
            datasets, series_name = self.load_dicom_series(series_path)
            
            # Check first DICOM to determine 3D/2D
            first_ds = datasets[0]
            first_img = first_ds.pixel_array
            
            if len(datasets) == 1 and first_img.ndim == 3:
                # Case 1: Single 3D DICOM file (e.g., from an MR scan)
                return self._process_single_3d_dicom(first_ds, series_name)
            else:
                # Case 2: Multiple 2D DICOM files (standard CT/CTA series)
                return self._process_multiple_2d_dicoms(datasets, series_name)
            
        except Exception as e:
            # CRITICAL ROBUSTNESS FIX: Return a zero volume on any failure
            print(f"!! CRITICAL DICOM ERROR processing series {Path(series_path).name}: {e}. Returning blank volume.")
            # Ensure the returned shape and dtype match the expected output
            return np.zeros(
                (self.target_depth, self.target_height, self.target_width), 
                dtype=np.uint8
            )
    
    def _process_single_3d_dicom(self, ds: pydicom.Dataset, series_name: str) -> np.ndarray:
        """Process single 3D DICOM file"""
        volume = self.extract_pixel_array(ds) # extract_pixel_array handles slope/intercept
        volume = volume.astype(np.float32)
        
        window_center, window_width = self.get_windowing_params(ds)
        
        # Apply windowing/normalization to each slice
        processed_slices = []
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            processed_img = self.apply_windowing_or_normalize(slice_img, window_center, window_width)
            processed_slices.append(processed_img)
        
        volume = np.stack(processed_slices, axis=0)
        
        # 3D resize
        final_volume = self.resize_volume_3d(volume)
        return final_volume
    
    def _process_multiple_2d_dicoms(self, datasets: List[pydicom.Dataset], series_name: str) -> np.ndarray:
        """Process multiple 2D DICOM files"""
        
        slice_info = self.extract_slice_info(datasets)
        sorted_slices = self.sort_slices_by_position(slice_info)
        
        # Use parameters from the first slice for consistency
        first_ds = sorted_slices[0]['dataset']
        window_center, window_width = self.get_windowing_params(first_ds)
        
        processed_slices = []
        
        for slice_data in sorted_slices:
            ds = slice_data['dataset']
            img = self.extract_pixel_array(ds) # img has slope/intercept applied
            processed_img = self.apply_windowing_or_normalize(img, window_center, window_width)
            
            # 2D resize before stacking
            resized_img = cv2.resize(processed_img, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR)
            
            processed_slices.append(resized_img)

        volume = np.stack(processed_slices, axis=0)
        
        # 3D resize (for depth normalization)
        final_volume = self.resize_volume_3d(volume)
        return final_volume

# ----------------------------------------------------------------------
# WRAPPER FUNCTIONS (API-compatible)
# ----------------------------------------------------------------------

def process_dicom_series_kaggle(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)) -> np.ndarray:
    """DICOM processing function for Kaggle inference (single series)"""
    preprocessor = DICOMPreprocessorKaggle(target_shape=target_shape)
    return preprocessor.process_series(series_path)

def process_dicom_series_safe(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)) -> np.ndarray:
    """Safe DICOM processing with memory cleanup and error handling."""
    # The DICOMPreprocessorKaggle.process_series now handles exceptions internally
    try:
        volume = process_dicom_series_kaggle(series_path, target_shape)
        return volume
    finally:
        gc.collect()

# ----------------------------------------------------------------------
# TEST FUNCTION (Unchanged)
# ----------------------------------------------------------------------

def test_single_series(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)):
    """Test processing for single series"""
    try:
        # Execute processing
        volume = process_dicom_series_safe(series_path, target_shape)
        return volume
        
    except Exception:
        # This block is for non-DICOM-related errors (e.g., file system failure)
        return None


import sys
import gc
import json
import shutil
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Data handling
import numpy as np
import polars as pl
import pandas as pd

# Medical imaging
import pydicom
import cv2

# ML/DL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm

# Transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Competition API
import kaggle_evaluation.rsna_inference_server

# DICOM preprocessor (DICOMPreprocessorKaggle class defined in previous cell)
# In actual use, define in the same file or import appropriately

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#print(f"Using device: {device}")

# ====================================================
# Competition constants
# ====================================================
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
print(np.array(LABEL_COLS)[[1,0,3,2,5,4,6,8,7,10,9,11,12,13]])

# ====================================================
# Configuration
# ====================================================
class InferenceConfig:
    # Model settings
    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    size = 384
    target_cols = LABEL_COLS
    num_classes = len(target_cols)
    in_chans = 32
    
    # Preprocessing settings
    target_shape = (32, 384, 384)  # (depth, height, width)
    
    # Inference settings
    batch_size = 1
    use_amp = True
    use_tta = False  # TTA is prohibited due to left/right positional information
    tta_transforms = 0
    
    # Model paths
    model_dir = '/kaggle/input/rsna2025-effnetv2-32ch'
    n_fold = 5
    trn_fold = [0, 1, 2, 3, 4]
    
    # Ensemble weights (equal weight for all folds)
    ensemble_weights = None  # None means equal weights

CFG = InferenceConfig()

# ====================================================
# Transforms
# ====================================================
def get_inference_transform():
    """Get inference transformation"""
    return A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(),
        ToTensorV2(),
    ])

# TTA is not used due to left/right positional information
# def get_tta_transforms():
#     """TTA is prohibited for brain aneurysms due to left/right positioning"""
#     pass

# ====================================================
# Model Loading Functions
# ====================================================
# Global variables
MODELS = {}
TRANSFORM = None
TTA_TRANSFORMS = None

def load_model_fold(fold: int) -> nn.Module:
    """Load a single fold model"""
    model_path = Path(CFG.model_dir) / f'{CFG.model_name}_fold{fold}_best.pth'
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    #print(f"Loading fold {fold} model from {model_path}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Initialize model with same architecture as training
    model = timm.create_model(
        CFG.model_name, 
        num_classes=CFG.num_classes, 
        pretrained=False,  # Don't load pretrained weights
        in_chans=CFG.in_chans
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    model.eval()
    
    #print(f"Successfully loaded fold {fold} model")
    return model

def load_models():
    """Load all fold models"""
    global MODELS, TRANSFORM, TTA_TRANSFORMS
    
    #print("Loading all fold models...")
    
    for fold in CFG.trn_fold:
        try:
            MODELS[fold] = load_model_fold(fold)
        except Exception as e:
            print(f"Warning: Could not load fold {fold}: {e}")
    
    if not MODELS:
        raise ValueError("No models were loaded successfully")
    
    # Initialize transforms
    TRANSFORM = get_inference_transform()
    # TTA is not used due to left/right positioning
    TTA_TRANSFORMS = None
    
    #print(f"Loaded {len(MODELS)} models: folds {list(MODELS.keys())}")
    
    # Warm up models
    #print("Warming up models...")
    dummy_image = torch.randn(1, CFG.in_chans, CFG.size, CFG.size).to(device)
    
    with torch.no_grad():
        for fold, model in MODELS.items():
            _ = model(dummy_image)
    
    #print("Models ready for inference!")

# ====================================================
# Prediction Functions
# ====================================================
def predict_single_model(model: nn.Module, image: np.ndarray) -> np.ndarray:
    """Make prediction with a single model (NO TTA due to left/right anatomy)"""
    
    # Same processing as training code
    # image shape: (D, H, W) = (32, 384, 384)
    image = image.transpose(1, 2, 0)  # (D,H,W) -> (H,W,D) = (384, 384, 32)
    
    # Apply same transform as training
    transformed = TRANSFORM(image=image)
    image_tensor = transformed['image']  # Shape: (32, 384, 384)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 32, 384, 384)
    
    with torch.no_grad():
        with autocast(enabled=CFG.use_amp):
            output = model(image_tensor)
            return torch.sigmoid(output).cpu().numpy().squeeze()

def predict_ensemble(image: np.ndarray) -> np.ndarray:
    """Make ensemble prediction across all folds"""
    all_predictions = []
    weights = []
    
    for fold, model in MODELS.items():
        pred = predict_single_model(model, image)
        all_predictions.append(pred)
        
        # Use equal weights if not specified
        if CFG.ensemble_weights is not None:
            weights.append(CFG.ensemble_weights.get(fold, 1.0))
        else:
            weights.append(1.0)
    
    # Weighted average
    weights = np.array(weights) / np.sum(weights)
    predictions = np.array(all_predictions)
    
    return np.average(predictions, weights=weights, axis=0)

def _predict_inner(series_path: str) -> pl.DataFrame:
    """Main prediction logic (internal)."""
    global MODELS
    
    # Load models if not already loaded
    if not MODELS:
        load_models()
    
    # Extract series ID
    series_id = os.path.basename(series_path)
    
    try:
        # Process DICOM series using our preprocessor
        volume = process_dicom_series_safe(series_path, CFG.target_shape)
        
        # Make ensemble prediction
        final_pred = (predict_ensemble(volume) + predict_ensemble(np.flip(volume,-1))[[1,0,3,2,5,4,6,8,7,10,9,11,12,13]])/2
        
        # Create output dataframe
        predictions_df = pl.DataFrame(
            data=[[series_id] + final_pred.tolist()],
            schema=[ID_COL] + LABEL_COLS,
            orient='row'
        )
        
        # Return without ID column, as required by the API
        return predictions_df.drop(ID_COL)
        
    except Exception as e:
        #print(f"Error processing {series_id}: {e}")
        # Return conservative predictions
        conservative_preds = [0.1] * len(LABEL_COLS)
        predictions_df = pl.DataFrame(
            data=[conservative_preds],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions_df

# ====================================================
# DICOM Processing (using DICOMPreprocessorKaggle defined in previous cell)
# ====================================================
def process_dicom_series_safe(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)) -> np.ndarray:
    """
    Safe DICOM processing with memory cleanup
    Uses DICOMPreprocessorKaggle defined in previous cell
    
    Args:
        series_path: Path to DICOM series
        target_shape: Target volume size (depth, height, width)
    
    Returns:
        np.ndarray: Processed volume
    """
    try:
        preprocessor = DICOMPreprocessorKaggle(target_shape=target_shape)
        volume = preprocessor.process_series(series_path)
        return volume
    finally:
        # Memory cleanup
        gc.collect()

def predict_fallback(series_path: str) -> pl.DataFrame:
    """Fallback prediction function"""
    #print(f"Using fallback predictions for {os.path.basename(series_path)}")
    
    # Return conservative predictions
    conservative_preds = [0.1] * len(LABEL_COLS)
    predictions_df = pl.DataFrame(
        data=[conservative_preds],
        schema=LABEL_COLS,
        orient='row'
    )
    
    # Clean up
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    return predictions_df

def predict(series_path: str) -> pl.DataFrame:
    """
    Top-level prediction function passed to the server.
    It calls the core logic and guarantees cleanup in a `finally` block.
    """
    try:
        # Call the internal prediction logic
        return _predict_inner(series_path)
    except Exception as e:
        #print(f"Error during prediction for {os.path.basename(series_path)}: {e}")
        #print("Using fallback predictions.")
        # Return a fallback dataframe with the correct schema
        conservative_preds = [0.1] * len(LABEL_COLS)
        predictions = pl.DataFrame(
            data=[conservative_preds],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions
    finally:
        # This code is required to prevent "out of disk space" and "directory not empty" errors.
        # It deletes the shared folder and then immediately recreates it, ensuring it's
        # empty and ready for the next prediction.
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        # Also perform memory cleanup here
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ====================================================
# Main Execution
# ====================================================

# Load models at startup
load_models()

# Initialize the inference server with our main `predict` function.
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

# Check if the notebook is running in the competition environment or a local session.
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    
    submission_df = pl.read_parquet('/kaggle/working/submission.parquet')
    display(submission_df)

