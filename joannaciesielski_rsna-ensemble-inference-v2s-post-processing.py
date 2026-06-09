
# ============================================================================
# CELL 1: IMPORTS
# ============================================================================

import os
import shutil
from collections import defaultdict
import numpy as np
import pandas as pd
import polars as pl
import pydicom
from pathlib import Path
from scipy.ndimage import zoom, rotate as scipy_rotate
import cv2
from typing import List, Tuple, Dict
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

import kaggle_evaluation.rsna_inference_server

print("âœ… All imports successful")

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"âœ… Using device: {device}")
print(f"   Available GPUs: {torch.cuda.device_count()}")


# ============================================================================
# CELL 2: CONFIGURATION + POST-PROCESSING RULES
# ============================================================================

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

# Model configuration
class Config:
    MODEL_NAME = 'tf_efficientnetv2_s.in21k_ft_in1k'
    INPUT_SIZE = (32, 384, 384)
    IN_CHANNELS = 32
    NUM_CLASSES = 14
    SIZE = 384
    
    # Ensemble settings
    MODEL_DIR = '/kaggle/input/rsna-aneurysm-efficientnet-ensemble-5fold/ensemble_models'
    N_FOLDS = 5
    FOLDS = [0, 1, 2, 3, 4]
    
    # Post-processing settings
    USE_POST_PROCESSING = True
    
    # Ensemble weights (equal or CV-based)
    ENSEMBLE_WEIGHTS = None  # Equal weights

config = Config()

# ============================================================================
# POST-PROCESSING FUNCTIONS
# ============================================================================

def apply_consistency_rules(predictions: np.ndarray) -> np.ndarray:
    """
    Apply logical consistency rules to predictions.
    
    Args:
        predictions: (14,) array [13 location probs + 1 aneurysm prob]
    
    Returns:
        processed predictions: (14,) array
    """
    predictions = predictions.copy()
    
    location_probs = predictions[:-1]
    aneurysm_prob = predictions[-1]
    
    # Rule 1: Aneurysm probability should be at least as high as max location
    # Logic: If any location has high probability, aneurysm should too
    max_location_prob = location_probs.max()
    if aneurysm_prob < max_location_prob:
        predictions[-1] = max(aneurysm_prob, max_location_prob * 0.95)
    
    # Rule 2: If all locations are very low, moderate aneurysm probability
    # Logic: Can't have aneurysm if no location shows evidence
    if max_location_prob < 0.15:
        predictions[-1] = min(predictions[-1], 0.6)
    
    # Rule 3: If any location is very confident, boost aneurysm
    # Logic: High location confidence implies aneurysm present
    if max_location_prob > 0.75:
        predictions[-1] = max(predictions[-1], 0.75)
    
    # Rule 4: Smooth extreme predictions slightly
    # Logic: Avoid overconfident predictions
    predictions = np.clip(predictions, 0.001, 0.999)
    
    # Rule 5: If multiple locations are moderately high, boost aneurysm
    # Logic: Multiple affected arteries suggest aneurysm
    high_locations = (location_probs > 0.4).sum()
    if high_locations >= 3:
        predictions[-1] = max(predictions[-1], 0.65)
    
    return predictions

def temperature_scaling(predictions: np.ndarray, temperature: float = 1.2) -> np.ndarray:
    """
    Apply temperature scaling for calibration.
    
    Args:
        predictions: probability predictions
        temperature: scaling factor (>1 = less confident, <1 = more confident)
    
    Returns:
        calibrated predictions
    """
    epsilon = 1e-7
    predictions = np.clip(predictions, epsilon, 1 - epsilon)
    
    # Convert to logits
    logits = np.log(predictions / (1 - predictions))
    
    # Scale
    scaled_logits = logits / temperature
    
    # Convert back to probabilities
    scaled_probs = 1 / (1 + np.exp(-scaled_logits))
    
    return scaled_probs

def apply_post_processing(predictions: np.ndarray) -> np.ndarray:
    """
    Apply all post-processing steps.
    
    Args:
        predictions: (14,) array of raw predictions
    
    Returns:
        processed predictions: (14,) array
    """
    if not config.USE_POST_PROCESSING:
        return predictions
    
    # Step 1: Consistency rules
    predictions = apply_consistency_rules(predictions)
    
    # Step 2: Temperature scaling (slightly less confident)
    predictions = temperature_scaling(predictions, temperature=1.15)
    
    # Step 3: Final clipping
    predictions = np.clip(predictions, 0.001, 0.999)
    
    return predictions

print("=" * 80)
print("âœ… POST-PROCESSING CONFIGURATION")
print("=" * 80)
print(f"Post-processing enabled: {config.USE_POST_PROCESSING}")
print(f"Rules:")
print(f"  1. Aneurysm prob â‰¥ max location prob (consistency)")
print(f"  2. Low locations â†’ moderate aneurysm (logic)")
print(f"  3. High location â†’ boost aneurysm (confidence)")
print(f"  4. Clip extremes (calibration)")
print(f"  5. Multiple locations â†’ boost aneurysm (evidence)")
print(f"  6. Temperature scaling: 1.15 (slight smoothing)")


# ============================================================================
# CELL 3: DICOM PREPROCESSOR
# ============================================================================

class DICOMPreprocessor:
    """DICOM preprocessing for inference"""
    
    def __init__(self, target_shape: Tuple[int, int, int] = (32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape
    
    def load_dicom_series(self, series_path: str) -> List:
        """Load DICOM files from series path"""
        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_path}")
        
        datasets = []
        for filepath in dicom_files:
            try:
                ds = pydicom.dcmread(filepath, force=True)
                datasets.append(ds)
            except:
                continue
        
        if not datasets:
            raise ValueError(f"No valid DICOM files")
        
        return datasets
    
    def extract_slice_info(self, datasets: List) -> List[Dict]:
        """Extract slice position information"""
        slice_info = []
        for i, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': i,
                'instance_number': getattr(ds, 'InstanceNumber', i),
            }
            try:
                ipp = np.array(getattr(ds, 'ImagePositionPatient', None))
                iop = np.array(getattr(ds, 'ImageOrientationPatient', None))
                n_vec = np.cross(iop[:3], iop[3:])
                info['z_position'] = -float((ipp * n_vec).sum())
            except:
                info['z_position'] = float(i)
            slice_info.append(info)
        return slice_info
    
    def sort_slices(self, slice_info: List[Dict]) -> List[Dict]:
        """Sort slices by z-position"""
        return sorted(slice_info, key=lambda x: x['z_position'])
    
    def apply_normalization(self, img: np.ndarray, modality: str) -> np.ndarray:
        """Apply normalization"""
        p1, p99 = np.percentile(img, [1, 99])
        
        if modality == 'CT' or modality == 'CTA':
            p1, p99 = 0, 500
        
        if p99 > p1:
            normalized = np.clip(img, p1, p99)
            normalized = (normalized - p1) / (p99 - p1)
            return (normalized * 255).astype(np.uint8)
        else:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                normalized = (img - img_min) / (img_max - img_min)
                return (normalized * 255).astype(np.uint8)
            else:
                return np.zeros_like(img, dtype=np.uint8)
    
    def extract_pixel_array(self, ds) -> np.ndarray:
        """Extract pixel array from DICOM"""
        img = ds.pixel_array.astype(np.float32)
        if img.ndim == 3:
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        return img
    
    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """Resize 3D volume to target shape"""
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        
        if current_shape == target_shape:
            return volume
        
        zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]
        resized = zoom(volume, zoom_factors, order=1, mode='nearest')
        resized = resized[:self.target_depth, :self.target_height, :self.target_width]
        
        pad_width = [
            (0, max(0, self.target_depth - resized.shape[0])),
            (0, max(0, self.target_height - resized.shape[1])),
            (0, max(0, self.target_width - resized.shape[2]))
        ]
        
        if any(pw[1] > 0 for pw in pad_width):
            resized = np.pad(resized, pad_width, mode='edge')
        
        return resized.astype(np.uint8)
    
    def process_series(self, series_path: str) -> np.ndarray:
        """Main processing pipeline"""
        datasets = self.load_dicom_series(series_path)
        
        first_ds = datasets[0]
        first_img = first_ds.pixel_array
        
        if len(datasets) == 1 and first_img.ndim == 3:
            return self._process_3d_dicom(first_ds)
        else:
            return self._process_2d_dicoms(datasets)
    
    def _process_3d_dicom(self, ds) -> np.ndarray:
        """Process single 3D DICOM"""
        volume = ds.pixel_array.astype(np.float32)
        modality = getattr(ds, 'Modality', 'CT')
        
        processed_slices = []
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            processed = self.apply_normalization(slice_img, modality)
            processed_slices.append(processed)
        
        volume = np.stack(processed_slices, axis=0)
        return self.resize_volume_3d(volume)
    
    def _process_2d_dicoms(self, datasets: List) -> np.ndarray:
        """Process multiple 2D DICOMs"""
        slice_info = self.extract_slice_info(datasets)
        sorted_slices = self.sort_slices(slice_info)
        
        first_ds = sorted_slices[0]['dataset']
        modality = getattr(first_ds, 'Modality', 'CT')
        
        processed_slices = []
        for slice_data in sorted_slices:
            ds = slice_data['dataset']
            img = self.extract_pixel_array(ds)
            processed = self.apply_normalization(img, modality)
            resized = cv2.resize(processed, (self.target_width, self.target_height))
            processed_slices.append(resized)
        
        volume = np.stack(processed_slices, axis=0)
        return self.resize_volume_3d(volume)

preprocessor = DICOMPreprocessor(target_shape=config.INPUT_SIZE)
print("âœ… DICOM Preprocessor ready")


# ============================================================================
# CELL 4: LOAD MODELS
# ============================================================================

print("Loading ensemble models...")

# Storage for models
MODELS = {}

# Transforms (base)
transform = A.Compose([
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        max_pixel_value=255.0
    ),
    ToTensorV2(),
])

# Load all fold models
for fold in config.FOLDS:
    model_path = os.path.join(config.MODEL_DIR, f'fold{fold}_best_model.pth')
    
    print(f"Loading fold {fold}...")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    # Create model
    model = timm.create_model(
        config.MODEL_NAME,
        pretrained=False,
        num_classes=config.NUM_CLASSES,
        in_chans=config.IN_CHANNELS
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    MODELS[fold] = model
    print(f"  âœ… Fold {fold} loaded (AUC: {checkpoint['metric']:.4f})")

print(f"\nâœ… All {len(MODELS)} models loaded!")


# ============================================================================
# CELL 5: PREDICT FUNCTION WITH POST-PROCESSING
# ============================================================================

def predict_single_model(model: nn.Module, image: np.ndarray) -> np.ndarray:
    """Make prediction with a single model"""
    # Transpose for transforms: (D,H,W) -> (H,W,D)
    image = image.transpose(1, 2, 0)
    
    # Apply transforms
    transformed = transform(image=image)
    image_tensor = transformed['image']  # (32, 384, 384)
    image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 32, 384, 384)
    
    with torch.no_grad():
        output = model(image_tensor)
        return torch.sigmoid(output).cpu().numpy().squeeze()

def predict_ensemble(image: np.ndarray) -> np.ndarray:
    """
    Make ensemble prediction.
    
    Args:
        image: (D, H, W) numpy array (preprocessed volume)
    
    Returns:
        final_prediction: (14,) numpy array
    """
    all_fold_predictions = []
    
    # Get predictions from all folds
    for fold, model in MODELS.items():
        pred = predict_single_model(model, image)
        all_fold_predictions.append(pred)
    
    # Equal-weight average across folds
    predictions = np.array(all_fold_predictions)
    ensemble_pred = np.mean(predictions, axis=0)
    
    # Apply post-processing
    final_pred = apply_post_processing(ensemble_pred)
    
    return final_pred

def predict(series_path: str) -> pl.DataFrame:
    """
    Main prediction function for API.
    """
    series_id = os.path.basename(series_path)
    
    try:
        # Process DICOM series
        volume = preprocessor.process_series(series_path)
        
        # Make ensemble prediction with post-processing
        final_pred = predict_ensemble(volume)
        
        # Create output dataframe
        result = pl.DataFrame(
            data=[[series_id] + final_pred.tolist()],
            schema=[ID_COL] + LABEL_COLS,
            orient='row',
        )
        
        return result.drop(ID_COL)
        
    except Exception as e:
        # Conservative fallback
        conservative_preds = [0.1] * len(LABEL_COLS)
        result = pl.DataFrame(
            data=[conservative_preds],
            schema=LABEL_COLS,
            orient='row',
        )
        return result
    finally:
        # Cleanup
        shutil.rmtree('/kaggle/shared', ignore_errors=True)
        os.makedirs('/kaggle/shared', exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

print("âœ… Predict function defined (WITH POST-PROCESSING)")


# ============================================================================
# CELL 6: START INFERENCE SERVER
# ============================================================================

# Create inference server
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

# Run server
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("ğŸ�† COMPETITION MODE: Starting inference server...")
    inference_server.serve()
else:
    print("ğŸ§ª LOCAL TEST MODE: Running on sample data...")
    inference_server.run_local_gateway()
    
    print("\nğŸ“Š Sample predictions:")
    display(pl.read_parquet('/kaggle/working/submission.parquet'))
    print("\nâœ… Local test complete! (WITH POST-PROCESSING)")

