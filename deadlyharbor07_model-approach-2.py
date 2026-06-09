input_monaipath = "/kaggle/input/monai-v060-deep-learning-in-healthcare-imaging/MONAI-1.0.0"
import sys
sys.path.append(input_monaipath)



import monai


# efficient net combined for 12 labels, 0 and 12 by radimagent and gru
"""
RSNA 2025 Brain Aneurysm Detection - Robust Inference Pipeline
Maintains exact preprocessing from training while adding robustness and optimizations
"""

import os
import sys
import gc
import json
import shutil
import warnings
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
import polars as pl
import pandas as pd

import pydicom
import cv2
from scipy import ndimage

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import kaggle_evaluation.rsna_inference_server

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ====================================================
# Constants
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

# Anatomical groupings for post-processing
LEFT_VESSEL_INDICES = [0, 2, 4, 7, 9]
RIGHT_VESSEL_INDICES = [1, 3, 5, 8, 10]
MIDLINE_VESSEL_INDICES = [6, 11, 12]  # Anterior communicating, Basilar, Other posterior

# ====================================================
# Configuration
# ====================================================
@dataclass
class InferenceConfig:
    """Configuration for robust inference"""
    # Model settings
    model_name: str = "tf_efficientnetv2_s.in21k_ft_in1k"
    size: int = 384
    num_classes: int = 14
    in_chans: int = 32
    
    # CRITICAL: Keep exact preprocessing from training
    target_shape: Tuple[int, int, int] = (32, 384, 384)
    force_rescale_identity: bool = True  # Keep slope=1, intercept=0 as in training
    fixed_normalization_range: Tuple[int, int] = (0, 500)  # Keep hardcoded values
    
    # Inference settings
    batch_size: int = 1
    use_amp: bool = True
    use_tta: bool = False  # No horizontal flips due to L/R anatomy
    use_brightness_tta: bool = True  # Safe TTA that doesn't affect L/R
    
    # Model paths
    model_dir: str = '/kaggle/input/rsna2025-effnetv2-32ch'
    n_fold: int = 5
    trn_fold: List[int] = None
    
    # Ensemble settings
    ensemble_method: str = 'geometric_mean'  # Better for probabilities
    temperature_scaling: float = 1.0
    
    # Robustness settings
    fallback_predictions: float = 0.1  # Conservative but not too low
    max_retries: int = 3
    cleanup_after_each: bool = True
    
    def __post_init__(self):
        if self.trn_fold is None:
            self.trn_fold = list(range(self.n_fold))

CFG = InferenceConfig()

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {device}")

# ====================================================
# DICOM Preprocessing (Maintain exact training logic)
# ====================================================
class DICOMPreprocessorRobust:
    """
    Robust DICOM preprocessing that maintains exact training preprocessing
    """
    
    def __init__(self, target_shape: Tuple[int, int, int] = (32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape
        self.stats = {}  # Collect statistics for debugging
        
    def load_dicom_series(self, series_path: str) -> Tuple[List[pydicom.Dataset], str]:
        """Load DICOM series with robust error handling"""
        series_path = Path(series_path)
        series_name = series_path.name
        
        # Find all DICOM files
        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_path}")
        
        logger.debug(f"Found {len(dicom_files)} DICOM files in series {series_name}")
        
        # Load DICOM datasets with error handling
        datasets = []
        load_errors = 0
        
        for filepath in dicom_files:
            try:
                ds = pydicom.dcmread(filepath, force=True)
                # Verify pixel data exists
                if hasattr(ds, 'pixel_array'):
                    datasets.append(ds)
                else:
                    load_errors += 1
            except Exception as e:
                logger.debug(f"Failed to load {filepath}: {e}")
                load_errors += 1
        
        if not datasets:
            raise ValueError(f"No valid DICOM files in {series_path} (failed: {load_errors}/{len(dicom_files)})")
        
        if load_errors > 0:
            logger.warning(f"Loaded {len(datasets)}/{len(dicom_files)} DICOM files (errors: {load_errors})")
        
        return datasets, series_name
    
    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
        """Extract slice position information"""
        slice_info = []
        
        for i, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': i,
                'instance_number': getattr(ds, 'InstanceNumber', i),
            }
            
            # Get z-coordinate with multiple fallbacks
            try:
                position = getattr(ds, 'ImagePositionPatient', None)
                if position is not None and len(position) >= 3:
                    info['z_position'] = float(position[2])
                elif hasattr(ds, 'SliceLocation'):
                    info['z_position'] = float(ds.SliceLocation)
                else:
                    info['z_position'] = float(info['instance_number'])
            except Exception:
                info['z_position'] = float(i)
            
            slice_info.append(info)
        
        return slice_info
    
    def sort_slices_by_position(self, slice_info: List[Dict]) -> List[Dict]:
        """Sort slices by z-coordinate and remove duplicates"""
        # Remove duplicates based on z-position
        seen_positions = {}
        unique_slices = []
        
        for info in slice_info:
            z_pos = round(info['z_position'], 3)
            if z_pos not in seen_positions:
                seen_positions[z_pos] = info
                unique_slices.append(info)
        
        sorted_slices = sorted(unique_slices, key=lambda x: x['z_position'])
        
        if len(sorted_slices) != len(slice_info):
            logger.debug(f"Removed {len(slice_info) - len(sorted_slices)} duplicate slices")
        
        return sorted_slices
    
    def get_windowing_params(self, ds: pydicom.Dataset) -> Tuple[Optional[str], Optional[str]]:
        """
        CRITICAL: Return exact values used in training
        The original code returns "CT", "CT" for CT modality
        """
        modality = getattr(ds, 'Modality', 'CT')
        
        if modality == 'CT':
            # Return exactly what training code expects
            return "CT", "CT"
        elif modality == 'MR':
            return None, None
        else:
            return None, None
    
    def apply_windowing_or_normalize(self, img: np.ndarray, center: Optional[str], 
                                    width: Optional[str]) -> np.ndarray:
        """
        CRITICAL: Apply exact normalization from training
        Must maintain the hardcoded p1=0, p99=500 for CT
        """
        if center is not None and width is not None:
            # For CT: use hardcoded normalization (this is what the model expects)
            p1, p99 = 0, 500  # CRITICAL: These exact values were used in training
            
            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)
                return result
            else:
                # Fallback
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    return result
                else:
                    return np.zeros_like(img, dtype=np.uint8)
        else:
            # For MR: statistical normalization
            p1, p99 = np.percentile(img, [1, 99])
            
            if p99 > p1:
                normalized = np.clip(img, p1, p99)
                normalized = (normalized - p1) / (p99 - p1)
                result = (normalized * 255).astype(np.uint8)
                return result
            else:
                # Fallback
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    normalized = (img - img_min) / (img_max - img_min)
                    result = (normalized * 255).astype(np.uint8)
                    return result
                else:
                    return np.zeros_like(img, dtype=np.uint8)
    
    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        """
        Extract 2D pixel array
        CRITICAL: Force slope=1, intercept=0 as in original training code
        """
        img = ds.pixel_array.astype(np.float32)
        
        # Handle 3D volumes (select middle frame)
        if img.ndim == 3:
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]
        
        # Convert color to grayscale if needed
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        # CRITICAL: Force identity transform (this is what training used)
        if CFG.force_rescale_identity:
            slope, intercept = 1, 0
        else:
            slope = getattr(ds, 'RescaleSlope', 1)
            intercept = getattr(ds, 'RescaleIntercept', 0)
        
        if slope != 1 or intercept != 0:
            img = img * float(slope) + float(intercept)
        
        return img
    
    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """Resize 3D volume to target size"""
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        
        if current_shape == target_shape:
            return volume
        
        # Calculate zoom factors
        zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]
        
        # Resize with linear interpolation
        resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        
        # Ensure exact size
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]
        
        # Pad if necessary
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]
        
        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')
        
        return resized_volume.astype(np.uint8)
    
    def process_series(self, series_path: str) -> np.ndarray:
        """Process DICOM series"""
        datasets, series_name = self.load_dicom_series(series_path)
        
        first_ds = datasets[0]
        first_img = first_ds.pixel_array
        
        if len(datasets) == 1 and first_img.ndim == 3:
            # Single 3D DICOM
            return self._process_single_3d_dicom(first_ds, series_name)
        else:
            # Multiple 2D DICOMs
            return self._process_multiple_2d_dicoms(datasets, series_name)
    
    def _process_single_3d_dicom(self, ds: pydicom.Dataset, series_name: str) -> np.ndarray:
        """Process single 3D DICOM file"""
        volume = ds.pixel_array.astype(np.float32)
        
        # CRITICAL: Force identity transform
        if CFG.force_rescale_identity:
            slope, intercept = 1, 0
        else:
            slope = getattr(ds, 'RescaleSlope', 1)
            intercept = getattr(ds, 'RescaleIntercept', 0)
        
        if slope != 1 or intercept != 0:
            volume = volume * float(slope) + float(intercept)
        
        window_center, window_width = self.get_windowing_params(ds)
        
        # Process each slice
        processed_slices = []
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            processed_img = self.apply_windowing_or_normalize(slice_img, window_center, window_width)
            processed_slices.append(processed_img)
        
        volume = np.stack(processed_slices, axis=0)
        return self.resize_volume_3d(volume)
    
    def _process_multiple_2d_dicoms(self, datasets: List[pydicom.Dataset], series_name: str) -> np.ndarray:
        """Process multiple 2D DICOM files"""
        slice_info = self.extract_slice_info(datasets)
        sorted_slices = self.sort_slices_by_position(slice_info)
        
        first_ds = sorted_slices[0]['dataset']
        window_center, window_width = self.get_windowing_params(first_ds)
        
        processed_slices = []
        for slice_data in sorted_slices:
            ds = slice_data['dataset']
            img = self.extract_pixel_array(ds)
            processed_img = self.apply_windowing_or_normalize(img, window_center, window_width)
            resized_img = cv2.resize(processed_img, (self.target_width, self.target_height))
            processed_slices.append(resized_img)
        
        volume = np.stack(processed_slices, axis=0)
        return self.resize_volume_3d(volume)

# ====================================================
# Model Loading and Ensemble
# ====================================================
class ModelEnsemble:
    """Robust model ensemble with multiple prediction strategies"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.models = {}
        self.transform = None
        self.device = device
        
    def get_inference_transform(self):
        """Get inference transformation"""
        return A.Compose([
            A.Resize(self.config.size, self.config.size),
            A.Normalize(),
            ToTensorV2(),
        ])
    
    def load_models(self):
        """Load all fold models with robust error handling"""
        logger.info("Loading ensemble models...")
        
        successful_loads = 0
        failed_loads = []
        
        for fold in self.config.trn_fold:
            try:
                model = self._load_single_model(fold)
                self.models[fold] = model
                successful_loads += 1
                logger.info(f"Successfully loaded fold {fold}")
            except Exception as e:
                logger.warning(f"Failed to load fold {fold}: {e}")
                failed_loads.append(fold)
        
        if successful_loads == 0:
            raise RuntimeError(f"Failed to load any models. Attempted folds: {self.config.trn_fold}")
        
        if failed_loads:
            logger.warning(f"Loaded {successful_loads}/{len(self.config.trn_fold)} models. Failed: {failed_loads}")
        
        # Initialize transform
        self.transform = self.get_inference_transform()
        
        # Warm up models
        self._warm_up_models()
        
        logger.info(f"Model ensemble ready with {len(self.models)} models")
    
    def _load_single_model(self, fold: int) -> nn.Module:
        """Load a single fold model with multiple path attempts"""
        # Try different naming conventions
        possible_paths = [
            Path(self.config.model_dir) / f'{self.config.model_name}_fold{fold}_best.pth',
            Path(self.config.model_dir) / f'fold{fold}_best.pth',
            Path(self.config.model_dir) / f'model_fold{fold}.pth',
        ]
        
        model_path = None
        for path in possible_paths:
            if path.exists():
                model_path = path
                break
        
        if model_path is None:
            raise FileNotFoundError(f"No model file found for fold {fold}. Tried: {possible_paths}")
        
        logger.debug(f"Loading model from {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Initialize model
        model = timm.create_model(
            self.config.model_name,
            num_classes=self.config.num_classes,
            pretrained=False,
            in_chans=self.config.in_chans
        )
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                # Assume the checkpoint is the state dict itself
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
        
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def _warm_up_models(self):
        """Warm up models for consistent inference speed"""
        logger.debug("Warming up models...")
        dummy_input = torch.randn(1, self.config.in_chans, self.config.size, self.config.size).to(self.device)
        
        with torch.no_grad():
            for fold, model in self.models.items():
                _ = model(dummy_input)
    
    def predict_single_model(self, model: nn.Module, image: np.ndarray, 
                           apply_tta: bool = False) -> np.ndarray:
        """Predict with single model, optionally with TTA"""
        predictions = []
        
        # Original prediction
        image_transposed = image.transpose(1, 2, 0)  # (D,H,W) -> (H,W,D)
        transformed = self.transform(image=image_transposed)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            with autocast(enabled=self.config.use_amp):
                output = model(image_tensor)
                if self.config.temperature_scaling != 1.0:
                    output = output / self.config.temperature_scaling
                pred = torch.sigmoid(output).cpu().numpy().squeeze()
                predictions.append(pred)
        
        # Brightness TTA (safe for L/R anatomy)
        if apply_tta and self.config.use_brightness_tta:
            # Slightly brighter
            image_bright = np.clip(image * 1.05, 0, 255).astype(np.uint8)
            image_bright_t = image_bright.transpose(1, 2, 0)
            transformed = self.transform(image=image_bright_t)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                with autocast(enabled=self.config.use_amp):
                    output = model(image_tensor)
                    if self.config.temperature_scaling != 1.0:
                        output = output / self.config.temperature_scaling
                    pred = torch.sigmoid(output).cpu().numpy().squeeze()
                    predictions.append(pred)
            
            # Slightly darker
            image_dark = np.clip(image * 0.95, 0, 255).astype(np.uint8)
            image_dark_t = image_dark.transpose(1, 2, 0)
            transformed = self.transform(image=image_dark_t)
            image_tensor = transformed['image'].unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                with autocast(enabled=self.config.use_amp):
                    output = model(image_tensor)
                    if self.config.temperature_scaling != 1.0:
                        output = output / self.config.temperature_scaling
                    pred = torch.sigmoid(output).cpu().numpy().squeeze()
                    predictions.append(pred)
        
        # Average TTA predictions
        return np.mean(predictions, axis=0)
    
    def predict_ensemble(self, image: np.ndarray) -> np.ndarray:
        """Make ensemble prediction across all loaded models"""
        if not self.models:
            raise RuntimeError("No models loaded for ensemble prediction")
        
        all_predictions = []
        
        for fold, model in self.models.items():
            try:
                pred = self.predict_single_model(model, image, apply_tta=True)
                all_predictions.append(pred)
            except Exception as e:
                logger.warning(f"Prediction failed for fold {fold}: {e}")
                continue
        
        if not all_predictions:
            raise RuntimeError("All model predictions failed")
        
        predictions = np.array(all_predictions)
        
        # Apply ensemble method
        if self.config.ensemble_method == 'average':
            result = np.mean(predictions, axis=0)
        elif self.config.ensemble_method == 'geometric_mean':
            # Geometric mean (better for probabilities)
            # Add small epsilon to avoid log(0)
            result = np.exp(np.mean(np.log(predictions + 1e-10), axis=0))
        elif self.config.ensemble_method == 'median':
            result = np.median(predictions, axis=0)
        else:
            result = np.mean(predictions, axis=0)
        
        return result

# ====================================================
# Post-processing
# ====================================================
def apply_anatomical_constraints(predictions: np.ndarray) -> np.ndarray:
    """Apply medical knowledge-based post-processing"""
    predictions = predictions.copy()
    
    # 1. Ensure consistency with "Aneurysm Present" flag
    aneurysm_present_idx = -1
    location_max = np.max(predictions[:-1])
    
    if location_max > 0.5:
        # If any location has high probability, ensure overall flag is high
        predictions[aneurysm_present_idx] = max(
            predictions[aneurysm_present_idx], 
            location_max * 0.95
        )
    elif predictions[aneurysm_present_idx] < 0.15:
        # If overall probability is very low, reduce all locations
        predictions[:-1] = predictions[:-1] * 0.7
    
    # 2. Apply mild bilateral symmetry boost (aneurysms can be bilateral)
    for left_idx, right_idx in zip(LEFT_VESSEL_INDICES, RIGHT_VESSEL_INDICES):
        left_prob = predictions[left_idx]
        right_prob = predictions[right_idx]
        
        # If one side has very high probability, slightly boost the other
        if left_prob > 0.7:
            predictions[right_idx] = max(right_prob, left_prob * 0.2)
        if right_prob > 0.7:
            predictions[left_idx] = max(left_prob, right_prob * 0.2)
    
    # 3. Ensure minimum probability for safety (avoid exact zeros)
    min_prob = 0.001
    predictions = np.maximum(predictions, min_prob)
    
    # 4. Ensure maximum probability isn't too confident
    max_prob = 0.999
    predictions = np.minimum(predictions, max_prob)
    
    return predictions

# ====================================================
# Main Inference Pipeline
# ====================================================
# Global variables
MODEL_ENSEMBLE = None
PREPROCESSOR = None

def initialize_pipeline():
    """Initialize the complete inference pipeline"""
    global MODEL_ENSEMBLE, PREPROCESSOR
    
    logger.info("Initializing inference pipeline...")
    
    # Initialize preprocessor
    PREPROCESSOR = DICOMPreprocessorRobust(CFG.target_shape)
    
    # Initialize and load models
    MODEL_ENSEMBLE = ModelEnsemble(CFG)
    MODEL_ENSEMBLE.load_models()
    
    logger.info("Pipeline initialization complete")

def process_dicom_series_safe(series_path: str) -> np.ndarray:
    """Process DICOM series with error handling and retries"""
    global PREPROCESSOR
    
    if PREPROCESSOR is None:
        PREPROCESSOR = DICOMPreprocessorRobust(CFG.target_shape)
    
    last_error = None
    for attempt in range(CFG.max_retries):
        try:
            volume = PREPROCESSOR.process_series(series_path)
            return volume
        except Exception as e:
            last_error = e
            logger.warning(f"Attempt {attempt + 1} failed: {e}")
            if attempt < CFG.max_retries - 1:
                # Clean up before retry
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    
    raise RuntimeError(f"Failed to process DICOM after {CFG.max_retries} attempts: {last_error}")




import numpy as np

def add_gaussian_noise(img, prob=0.2, mean=0.0, std=0.01):
    if np.random.rand() < prob:
        noise = np.random.normal(mean, std, img.shape).astype(np.float32)
        return np.clip(img + noise, 0.0, 1.0)
    return img

def scale_intensity(img, prob=0.3, factors=0.1):
    if np.random.rand() < prob:
        scale = 1.0 + np.random.uniform(-factors, factors)
        return np.clip(img * scale, 0.0, 1.0)
    return img

def shift_intensity(img, prob=0.3, offsets=0.1):
    if np.random.rand() < prob:
        shift = np.random.uniform(-offsets, offsets)
        return np.clip(img + shift, 0.0, 1.0)
    return img

def augment(img):
    """ img: np.array, already normalized [0,1] """
    img = add_gaussian_noise(img)
    img = scale_intensity(img)
    img = shift_intensity(img)
    return img



#  Approach - z index sorting->radimagenet embeddings->gru
# GLOBALS
ALLOWED_TAGS=[
    "SOPClassUID",
    "SOPInstanceUID",
    "Modality",
    "PatientID",
    "SliceThickness",
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "InstanceNumber",
    "ImagePositionPatient",
    "ImageOrientationPatient",
    "FrameOfReferenceUID",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "Rows",
    "Columns",
    "PixelSpacing",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "PixelData"
]
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



# Preprocessing code for one series

# Imports for z index sorting 
import os
import pydicom as dcm
import numpy as np
import cv2
import json

# Radimage net embeddings import
import torch
import torch.nn as nn
import monai
# from monai.transforms import (
#     Compose,RandGaussianNoise, RandScaleIntensity, RandShiftIntensity
# )

# Resize embeddings for lstm 
from torch.utils.data import Dataset

# 1.Load a single scan and return np array for resnet
def load_series(series_path):
    """Load a scan and does z index sorting and with returns npy format"""
    # Load all dicom files in series
    dcm_files = [os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith(".dcm")]
    datasets = []
    for f in dcm_files:
        try:
            datasets.append(dcm.dcmread(f))
        except:
            continue
            
    # Check metadata availability
    first_ds = datasets[0]
    series_id = getattr(first_ds, "SeriesInstanceUID", "UNKNOWN")

    # Sort  with z index
    try:
        datasets.sort(key=lambda ds: float(ds.ImagePositionPatient[2]))
    except Exception:
        datasets.sort(key=lambda ds: int(ds.InstanceNumber))

    # Extract pixel data
    slices = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        arr = cv2.resize(arr, (224, 224))  # resize for ResNet
        arr = (arr - np.min(arr)) / (np.max(arr) - np.min(arr) + 1e-5)  # normalize
        slices.append(arr)

    volume = np.stack(slices, axis=0)  # shape = (num_slices, 224, 224)

    # Save processed volume
    return volume,series_id

# 2.Rad imagenet embeddings

radimagenet_path = "/kaggle/input/radimagenet_50/pytorch/default/1/ResNet50.pt"
"""
Embedding generator with augmentation using RadImageNet ResNet50
"""
import torch
import torch.nn as nn
from torchvision.models import resnet50


def get_feature_extractor(radimagenet_path: str):
    resnet = monai.networks.nets.resnet50(spatial_dims=2, n_input_channels=3)

    state_dict = torch.load(radimagenet_path, map_location="cpu")
    # checkpoint wrapped like {"state_dict": ...}
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # remove "backbone." prefix
    new_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace("backbone.", "")  # strip prefix
        new_state_dict[new_k] = v

    # load into resnet
    resnet.load_state_dict(new_state_dict, strict=False)

    feature_extractor = nn.Sequential(*list(resnet.children())[:-1])  # remove fc
    feature_extractor.eval()
    return feature_extractor



# Init embedder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Embedder =get_feature_extractor("/kaggle/input/radimagenet_50/pytorch/default/1/ResNet50.pt").to(device)
# Define augmentation pipeline
# augment = Compose([
#     RandGaussianNoise(prob=0.2, mean=0.0, std=0.01),               # scanner noise
#     RandScaleIntensity(factors=0.1, prob=0.3),                     # intensity scaling
#     RandShiftIntensity(offsets=0.1, prob=0.3),                     # intensity shifting
# ])


# 3. Make suitable for lsmtgru model

SEQ_LEN = 800
EMB_DIM = 2048

def embedding_resizer(raw_embedding: np.ndarray, seq_len: int = SEQ_LEN):
    """
    Resize embeddings to fixed length (seq_len, EMB_DIM).
    Also returns an attention mask (1 for real slices, 0 for padding).
    """
    n_slices = raw_embedding.shape[0]

    # Case 1: Too short → pad
    if n_slices < seq_len:
        pad_len = seq_len - n_slices
        pad = np.zeros((pad_len, raw_embedding.shape[1]), dtype=raw_embedding.dtype)
        resized = np.concatenate([raw_embedding, pad], axis=0)
        mask = np.concatenate([np.ones(n_slices), np.zeros(pad_len)])

    # Case 2: Too long → sample evenly
    elif n_slices > seq_len:
        indices = np.linspace(0, n_slices - 1, seq_len).astype(int)
        resized = raw_embedding[indices]
        mask = np.ones(seq_len)

    # Case 3: Exact length
    else:
        resized = raw_embedding
        mask = np.ones(seq_len)

    return resized, mask





"""WARM UP"""

# # Warm up
# # example data series
# series_path='/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.99887675554378211308175946117895608384'

# # loader is fine
# volume,series_id=load_series(series_path)
# # print(series_id)
# # print(volume.shape)
# # print(np.unique(volume))

# # now embeddings
# slices=volume
# embeddings = []
# with torch.no_grad():
#         for s in slices:
#             # augment slice (still single-channel)
#             s_aug = augment(s[np.newaxis, :, :])  # (1, H, W)

#             # expand to 3 channels (grayscale → RGB-like)
#             img = np.repeat(s_aug, 3, axis=0)  # (3, 224, 224)
#             img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3, 224, 224)

#             # forward pass
#             feat = Embedder(img)  # (1, 2048, 1, 1)
#             feat = torch.flatten(feat, 1)  # (1, 2048)

#             embeddings.append(feat.cpu().numpy())

# # stack → (num_slices, 2048)
# embeddings = np.vstack(embeddings)
# # print(f"✅ Generated embeddings for series {series_id} → {embeddings.shape}")

#  # resize
# embedding, mask = embedding_resizer(embeddings, SEQ_LEN)
# embedding = torch.tensor(embedding, dtype=torch.float32)   # (SEQ_LEN, EMB_DIM)
# mask = torch.tensor(mask, dtype=torch.float32)             # (SEQ_LEN,)



# print(embedding.shape)
# print(mask.shape)



# # Make predictions
# embedding, mask = embedding.to(device), mask.to(device)
# logits = model(embedding.unsqueeze(0), mask.unsqueeze(0))
# probs = torch.sigmoid(logits).detach().cpu()


# probs.shape


# 4. Create and load model

import torch
import torch.nn as nn

class AneurysmGRU(nn.Module):
    def __init__(self,
                 input_dim=2048,
                 hidden_dim=512,
                 num_layers=2,
                 num_classes=14,
                 bidirectional=True,
                 dropout=0.3):
        super(AneurysmGRU, self).__init__()

        self.gru = nn.GRU(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )

        self.bidirectional = bidirectional
        self.hidden_dim = hidden_dim

        # Linear head
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Sequential(
            nn.Linear(out_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, input_dim)
        mask: (batch, seq_len)   [1 = real, 0 = pad]
        """
        # GRU forward
        out, _ = self.gru(x)   # (batch, seq_len, hidden_dim*2)

        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            out = out * mask           # zero out padded timesteps

        # Global average pooling (mask-aware)
        if mask is not None:
            summed = torch.sum(out, dim=1)             # (batch, hidden_dim*2)
            counts = torch.sum(mask, dim=1) + 1e-6     # (batch, 1)
            pooled = summed / counts                   # mean pooling
        else:
            pooled = out.mean(dim=1)

        # Classifier
        logits = self.fc(pooled)  # (batch, num_classes)

        return logits
model = AneurysmGRU(
    input_dim=2048,
    hidden_dim=800,
    num_layers=4,
    num_classes=14,
    bidirectional=True,
    dropout=0.4
)
last_model_path='/kaggle/input/gru_rnn/pytorch/default/1/model_approach_2_fold0_best.pth'
checkpoint = torch.load(last_model_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model = model.to(device)
model.eval()


import polars as pl
import kaggle_evaluation.rsna_inference_server

def _predict_inner(series_path):
    volume,series_id=load_series(series_path)
    # shape = (num_slices, 224, 224)

    slices=volume
    embeddings = []
    with torch.no_grad():
        for s in slices:
            # augment slice (still single-channel)
            s_aug = augment(s)  # (1, H, W)

            # expand to 3 channels (grayscale → RGB-like)
            img = np.repeat(s_aug[np.newaxis, :, :], 3, axis=0)  # (3, 224, 224)
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 3, 224, 224)

            # forward pass
            feat = Embedder(img)  # (1, 2048, 1, 1)
            feat = torch.flatten(feat, 1)  # (1, 2048)

            embeddings.append(feat.cpu().numpy())

    # stack → (num_slices, 2048)
    embeddings = np.vstack(embeddings)

    # resize
    embedding, mask = embedding_resizer(embeddings, SEQ_LEN)
    embedding = torch.tensor(embedding, dtype=torch.float32)   # (SEQ_LEN, EMB_DIM)
    mask = torch.tensor(mask, dtype=torch.float32)             # (SEQ_LEN,)

    # Make predictions
    embedding, mask = embedding.to(device), mask.to(device)
    logits = model(embedding.unsqueeze(0), mask.unsqueeze(0))
    probs = torch.sigmoid(logits).detach().cpu()

    # Now polars dataframe
    predictions_df = pl.DataFrame(
            data=[[series_id] + probs[0].tolist()],
            schema=[ID_COL] + LABEL_COLS,
            orient='row'
        )
    # Return without ID column, as required by the API
    return predictions_df.drop(ID_COL)


def predict1(series_path: str) -> pl.DataFrame:
    """Main prediction function for the inference server"""
    global MODEL_ENSEMBLE
    
    # Initialize if needed
    if MODEL_ENSEMBLE is None:
        initialize_pipeline()
    
    series_id = os.path.basename(series_path)
    
    try:
        # Process DICOM
        volume = process_dicom_series_safe(series_path)
        logger.debug(f"Processed volume shape: {volume.shape}, dtype: {volume.dtype}")
        
        # Make ensemble prediction
        raw_predictions = MODEL_ENSEMBLE.predict_ensemble(volume)
        
        # Apply post-processing
        final_predictions = apply_anatomical_constraints(raw_predictions)
        
        # Create output dataframe
        predictions_df = pl.DataFrame(
            data=[final_predictions.tolist()],
            schema=LABEL_COLS,
            orient='row'
        )
        
        logger.info(f"Successfully predicted for {series_id}")
        return predictions_df
        
    except Exception as e:
        logger.error(f"Prediction failed for {series_id}: {e}", exc_info=True)
        
        # Return conservative fallback predictions
        conservative_preds = [CFG.fallback_predictions] * len(LABEL_COLS)
        predictions_df = pl.DataFrame(
            data=[conservative_preds],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions_df
        
    finally:
        # Clean up after each prediction
        if CFG.cleanup_after_each:
            shared_dir = '/kaggle/shared'
            if os.path.exists(shared_dir):
                shutil.rmtree(shared_dir, ignore_errors=True)
                os.makedirs(shared_dir, exist_ok=True)
            
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()




import sys
import shutil
import warnings
import gc
warnings.filterwarnings('ignore')
def predict(series_path: str) -> pl.DataFrame:
    """
    Top-level prediction function that blends EfficientNet + GRU outputs.
    Uses EfficientNet as the main model, but overrides columns 0 and 12 with GRU's predictions.
    """
    try:
        predictions_en = predict1(series_path)   # EfficientNet output
        predictions_gru = _predict_inner(series_path)  # GRU output
        
        # Try with label 2
        label_0_name = LABEL_COLS[10]
        # label_12_name = LABEL_COLS[12]
        print(predictions_en)
        print(predictions_gru)
        predictions_en = predictions_en.with_columns([
            predictions_gru[label_0_name].alias(label_0_name),
            # predictions_gru[label_12_name].alias(label_12_name),
        ])
        
        return predictions_en
    
    except Exception as e:
        # Conservative fallback
        conservative_preds = [0.1] * len(LABEL_COLS)
        predictions = pl.DataFrame(
            data=[conservative_preds],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions
    
    finally:
        # Cleanup
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()



# sample_ans


sample_ans=predict('/kaggle/input/rsna-intracranial-aneurysm-detection/series/1.2.826.0.1.3680043.8.498.99887675554378211308175946117895608384')


sample_ans


# Main execution

# Initialize the inference server with our main `predict` function.
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

# Check if the notebook is running in the competition environment or a local session.
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # make sure to give it an empty directory
    tmp_dir = "/kaggle/working/tmp_gateway"
    os.makedirs(tmp_dir, exist_ok=True)
    inference_server.run_local_gateway(file_share_dir=tmp_dir)
    
    submission_df = pl.read_parquet('/kaggle/working/submission.parquet')
    display(submission_df)





# ====================================================
# Main Execution
# ====================================================
if __name__ == "__main__":
    try:
        # Initialize pipeline at startup
        initialize_pipeline()
        
        # Create inference server
        inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
        
        # Check environment and run appropriate mode
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            logger.info("Running in competition environment")
            inference_server.serve()
        else:
            logger.info("Running in local test mode")
            inference_server.run_local_gateway()
            
            # Display submission for verification
            submission_path = '/kaggle/working/submission.parquet'
            if os.path.exists(submission_path):
                submission_df = pl.read_parquet(submission_path)
                print("\nSubmission Preview:")
                print(submission_df)
            else:
                logger.warning("No submission file found")
                
    except Exception as e:
        logger.error(f"Critical error in main execution: {e}", exc_info=True)
        sys.exit(1)




