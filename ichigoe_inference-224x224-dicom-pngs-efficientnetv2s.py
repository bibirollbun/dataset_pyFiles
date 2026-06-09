# Setup and imports for RSNA Intracranial Aneurysm Detection inference
import os
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
from pydicom.pixel_data_handlers.util import convert_color_space

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

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")


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

class InferenceConfig:
    # Model settings - aligned with training
    NUM_FRAMES = 8  # Updated to match training
    IMAGE_SIZE = 224
    NUM_CLASSES = 14
    
    # Model configuration - matching training exactly
    MODEL_NAME_BACKBONE = "tf_efficientnetv2_s.in1k"
    USE_METADATA = True
    USE_WINDOWING = True
    USE_3CHANNEL_INPUT = True
    USE_CLAHE = True
    
    # Model path - update according to your trained model
    model_path = "/kaggle/input/train-224x224-dicom-pngs-efficientnetb0/eightframe_efficientnetv2s_best.pth"
    
    # Inference settings
    batch_size = 1
    use_amp = True
    
    # Processing settings
    debug_mode = False

CFG = InferenceConfig()

# Add alias to match training code
Config = InferenceConfig

print(f"Configuration loaded:")
print(f"- Frames: {CFG.NUM_FRAMES}")
print(f"- Backbone: {CFG.MODEL_NAME_BACKBONE}")
print(f"- Image size: {CFG.IMAGE_SIZE}")
print(f"- Model path: {CFG.model_path}")
print(f"- Use windowing: {CFG.USE_WINDOWING}")
print(f"- Use CLAHE: {CFG.USE_CLAHE}")
print(f"- Use metadata: {CFG.USE_METADATA}")
print(f"- Config alias created for compatibility")


# 8-Frame EfficientNetV2-S model - matching training exactly
class ImprovedMultiFrameModel(nn.Module):
    """Model with EfficientNetV2-S and metadata integration for 8-frame processing"""
    def __init__(self, num_frames=8, num_classes=14, pretrained=True):
        super(ImprovedMultiFrameModel, self).__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.use_3channel = CFG.USE_3CHANNEL_INPUT
        self.use_metadata = CFG.USE_METADATA
        
        # Backbone: EfficientNetV2-S
        print(f"Loading backbone: {CFG.MODEL_NAME_BACKBONE}")
        self.backbone = timm.create_model(
            CFG.MODEL_NAME_BACKBONE,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        
        self.feature_dim = self.backbone.num_features
        print(f"Backbone {CFG.MODEL_NAME_BACKBONE}: {self.feature_dim} features")
        
        # Metadata processing
        if self.use_metadata:
            self.meta_fc = nn.Sequential(
                nn.Linear(2, 16),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(16, 32),
                nn.ReLU()
            )
            classifier_input_dim = self.feature_dim + 32
        else:
            classifier_input_dim = self.feature_dim
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x, meta=None):
        # 3-channel input processing for 8-frame data
        features = self.backbone(x)  # (batch_size, feature_dim)
        
        # Metadata integration
        if self.use_metadata and meta is not None:
            meta_features = self.meta_fc(meta)
            features = torch.cat([features, meta_features], dim=1)
        
        # Classification
        output = self.classifier(features)
        return output

print("Model architecture defined (matching training)")


# DICOM processing utilities with CLAHE - matching training exactly
def get_windowing_params(modality: str) -> Tuple[float, float]:
    """Get optimal windowing parameters for different modalities (match training)"""
    windows = {
        'CT': (40, 80),
        'CTA': (50, 350), 
        'MRA': (600, 1200),
        'MRI': (40, 80),
        'MR': (40, 80),
        'MRI T2': (40, 80),
        'MRI T1post': (40, 80),
    }
    return windows.get(modality, (40, 80))

def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Apply DICOM windowing to normalize image intensities (match training)"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)

def apply_clahe_normalization(img: np.ndarray, modality: str) -> np.ndarray:
    """Apply CLAHE with modality-specific optimization (match training)"""
    if not CFG.USE_CLAHE:
        return img
        
    # Create CLAHE object with modality-specific parameters
    if modality in ['CTA', 'MRA']:
        # Vascular imaging: stronger contrast improvement
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
        # Additional contrast boost for vascular structures
        img_clahe = cv2.convertScaleAbs(img_clahe, alpha=1.1, beta=5)
    elif modality in ['MRI', 'MR', 'MRI T2', 'MRI T1post']:
        # MRI: gentler improvement with gamma correction
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
        # Apply gamma correction for better tissue contrast
        img_clahe = np.power(img_clahe / 255.0, 0.9) * 255
        img_clahe = img_clahe.astype(np.uint8)
    else:
        # CT: standard CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img.astype(np.uint8))
    
    return img_clahe

def robust_normalization(volume: np.ndarray) -> np.ndarray:
    """Apply robust normalization using percentiles (match training)"""
    # Use percentile-based normalization instead of min-max for robustness
    p1, p99 = np.percentile(volume.flatten(), [1, 99])
    volume_norm = np.clip(volume, p1, p99)
    
    if p99 > p1:
        volume_norm = (volume_norm - p1) / (p99 - p1 + 1e-7)
    else:
        volume_norm = np.zeros_like(volume_norm)
        
    return (volume_norm * 255).astype(np.uint8)

print("DICOM processing functions ready (with CLAHE and 8-frame processing)")


def create_3channel_input_8frame(volume: np.ndarray) -> np.ndarray:
    """Create 3-channel input from 8-frame volume optimized for aneurysm detection"""
    if len(volume) == 0:
        return np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 3), dtype=np.uint8)
    
    # Middle slice (most important for anatomical reference)
    middle_slice = volume[len(volume) // 2]
    
    # Maximum Intensity Projection (MIP) - optimized for vascular structures
    mip = np.max(volume, axis=0)
    
    # Standard deviation projection for texture analysis
    std_proj = np.std(volume, axis=0).astype(np.float32)
    
    # Normalize standard deviation projection with robust method
    if std_proj.max() > std_proj.min():
        p1, p99 = np.percentile(std_proj, [5, 95])
        std_proj = np.clip(std_proj, p1, p99)
        std_proj = ((std_proj - p1) / (p99 - p1 + 1e-7) * 255).astype(np.uint8)
    else:
        std_proj = np.zeros_like(std_proj, dtype=np.uint8)
    
    return np.stack([middle_slice, mip, std_proj], axis=-1)

def smart_8_frame_sampling(volume_paths: List[str], series_uid: str = None) -> List[str]:
    """Intelligent 8-frame sampling strategy using every other frame"""
    n = len(volume_paths)
    
    if n <= 8:
        # If we have 8 or fewer frames, use all available
        result = volume_paths[:]
        # Pad with repetitions if needed
        while len(result) < 8:
            result.extend(volume_paths[:8-len(result)])
        return result[:8]
    
    # Skip every other frame starting from a strategic position
    # Start from 10% into the volume to avoid empty slices at the beginning
    start_idx = max(0, int(n * 0.1))
    
    # Calculate step size to get 8 frames with good coverage
    available_frames = n - start_idx
    step = max(1, available_frames // 8)
    
    indices = []
    current_idx = start_idx
    while len(indices) < 8 and current_idx < n:
        indices.append(current_idx)
        current_idx += step
    
    # If we need more frames, fill from the remaining
    while len(indices) < 8:
        remaining = [i for i in range(n) if i not in indices]
        if remaining:
            indices.append(remaining[len(indices) % len(remaining)])
        else:
            indices.append(indices[-1])  # Duplicate last frame
    
    return [volume_paths[i] for i in indices[:8]]

print("8-frame processing functions ready (matching training)")


def extract_sort_key(path: str) -> Tuple[float, float, str]:
    """Extract sorting key from DICOM file for proper ordering (match training)"""
    try:
        ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        instance_number = getattr(ds, 'InstanceNumber', None)
        position = getattr(ds, 'ImagePositionPatient', [None, None, None])
        z = position[2] if position and len(position) == 3 else None
        if instance_number is not None:
            return (int(instance_number), 0, path)
        elif z is not None:
            return (float('inf'), float(z), path)
        else:
            return (float('inf'), float('inf'), path)
    except:
        return (float('inf'), float('inf'), path)

def sort_dicom_paths(dcm_paths: List[str]) -> List[str]:
    """Sort DICOM paths by medical metadata (match training)"""
    if not dcm_paths:
        return []
    
    sort_info = []
    for path in dcm_paths:
        sort_info.append(extract_sort_key(path))
    
    sort_info.sort()
    return [x[2] for x in sort_info]

def extract_metadata_from_dicom(dicom_path: str) -> Tuple[float, float]:
    """Extract age and sex metadata from DICOM (match training)"""
    try:
        ds = pydicom.dcmread(dicom_path, stop_before_pixels=True, force=True)
        
        # Age processing (match training exactly)
        age = getattr(ds, 'PatientAge', '50')
        if pd.isna(age) or age is None:
            age = 50
        elif isinstance(age, str):
            age = int(''.join(filter(str.isdigit, age[:3])) or '50')
        age = min(float(age), 100.0) / 100.0
        
        # Sex processing (match training exactly)
        sex = getattr(ds, 'PatientSex', 'M')
        sex = 1.0 if sex == 'M' else 0.0
        
        return age, sex
        
    except Exception as e:
        if CFG.debug_mode:
            print(f"Error extracting metadata from {dicom_path}: {e}")
        return 0.5, 1.0  # Default values

print("DICOM sorting and metadata extraction ready")


def process_single_dicom(dicom_path: str, modality: str = 'CTA') -> Optional[np.ndarray]:
    """Process a single DICOM file with CLAHE (match training processing)"""
    try:
        # Read DICOM with force=True for better compatibility
        dicom = pydicom.dcmread(dicom_path, force=True)
        
        # Check for pixel data
        if 'PixelData' not in dicom:
            if CFG.debug_mode:
                print(f"Warning: No pixel data in {dicom_path}")
            return None
            
        # Extract pixel array
        img = dicom.pixel_array
        
        # Check if image is valid
        if img is None or img.size == 0:
            if CFG.debug_mode:
                print(f"Warning: Empty pixel array in {dicom_path}")
            return None
            
        # Handle photometric interpretation
        interp = getattr(dicom, 'PhotometricInterpretation', 'MONOCHROME2')
        
        # Handle YBR color space conversion
        if interp == "YBR_FULL":
            try:
                img = convert_color_space(img, 'YBR_FULL', 'RGB')
            except:
                pass
        
        # Convert to grayscale if multi-channel
        if img.ndim == 3:
            if interp in ["RGB", "YBR_FULL"]:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            elif img.shape[2] == 1:
                img = img[:, :, 0]
            else:
                img = img[:, :, 0]  # Take first channel
        
        # Ensure 2D image
        if img.ndim != 2:
            return None
            
        # Apply rescale if available (match training)
        if hasattr(dicom, 'RescaleSlope') and hasattr(dicom, 'RescaleIntercept'):
            img = img.astype(np.float32)
            img = img * dicom.RescaleSlope + dicom.RescaleIntercept
            
        # Apply windowing if requested (match training)
        if CFG.USE_WINDOWING:
            window_center, window_width = get_windowing_params(modality)
            img = apply_dicom_windowing(img, window_center, window_width)
        else:
            # Normalize without windowing
            img = img.astype(np.float32)
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        
        # Handle MONOCHROME1 (inverted grayscale)
        if interp == "MONOCHROME1":
            img = 255 - img
            
        # Apply CLAHE improvement (match training)
        img = apply_clahe_normalization(img, modality)
            
        # Validate before resize
        if img.shape[0] == 0 or img.shape[1] == 0:
            return None
            
        # High quality resize (match training)
        img = cv2.resize(img, (CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        
        return img
        
    except Exception as e:
        if CFG.debug_mode:
            print(f"Error processing {dicom_path}: {e}")
        return None

print("Single DICOM processing ready (with CLAHE)")


def process_dicom_series(series_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Process DICOM series and return 8-frame 3-channel tensor + metadata (match training)"""
    series_path = Path(series_path)
    
    # Find all DICOM files
    dicom_files = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                dicom_files.append(os.path.join(root, file))
    
    if not dicom_files:
        if CFG.debug_mode:
            print(f"Warning: No DICOM files found in {series_path}")
        return create_dummy_tensor_and_metadata()
    
    # Sort files by medical metadata (match training)
    sorted_files = sort_dicom_paths(dicom_files)
    
    # Apply smart 8-frame sampling (match training)
    sampled_files = smart_8_frame_sampling(sorted_files)
    
    # Get modality and metadata from first file
    try:
        first_dicom = pydicom.dcmread(sampled_files[0], stop_before_pixels=True)
        modality = getattr(first_dicom, 'Modality', 'CTA')
        age, sex = extract_metadata_from_dicom(sampled_files[0])
    except:
        modality = 'CTA'
        age, sex = 0.5, 1.0
    
    # Process each DICOM file to create volume
    processed_images = []
    for dicom_path in sampled_files:
        img = process_single_dicom(dicom_path, modality)
        if img is not None:
            processed_images.append(img)
        else:
            # Add zero image if processing fails
            processed_images.append(np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), dtype=np.uint8))
    
    # Ensure we have exactly 8 frames
    while len(processed_images) < 8:
        if processed_images:
            processed_images.append(processed_images[-1])  # Duplicate last frame
        else:
            processed_images.append(np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE), dtype=np.uint8))
    
    # Convert to numpy volume
    volume = np.array(processed_images[:8])  # Take exactly 8 frames
    
    # Apply robust normalization (match training)
    volume = robust_normalization(volume)
    
    # Create 3-channel input (match training exactly)
    image_3channel = create_3channel_input_8frame(volume)
    
    # Apply normalization (match training)
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    try:
        transformed = transform(image=image_3channel)
        image_tensor = transformed['image']  # (C, H, W)
    except:
        # Create dummy tensor on transform failure
        dummy_img = np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 3), dtype=np.uint8)
        transformed = transform(image=dummy_img)
        image_tensor = transformed['image']
    
    # Create metadata tensor (match training)
    metadata_tensor = torch.tensor([age, sex], dtype=torch.float32)
    
    return image_tensor, metadata_tensor

def create_dummy_tensor_and_metadata() -> Tuple[torch.Tensor, torch.Tensor]:
    """Create dummy tensor and metadata when processing fails"""
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    dummy_img = np.zeros((CFG.IMAGE_SIZE, CFG.IMAGE_SIZE, 3), dtype=np.uint8)
    transformed = transform(image=dummy_img)
    image_tensor = transformed['image']
    
    # Default metadata (age=50%, sex=male)
    metadata_tensor = torch.tensor([0.5, 1.0], dtype=torch.float32)
    
    return image_tensor, metadata_tensor

print("8-frame 3-channel series processing ready (matching training)")


# Global variables
MODEL = None

def load_model() -> nn.Module:
    """Load trained 8-frame model (match training architecture)"""
    print(f"Loading 8-frame model from: {CFG.model_path}")
    
    if not os.path.exists(CFG.model_path):
        raise FileNotFoundError(f"Model file not found: {CFG.model_path}")
    
    # Initialize model (match training exactly)
    model = ImprovedMultiFrameModel(
        num_frames=CFG.NUM_FRAMES,
        num_classes=CFG.NUM_CLASSES,
        pretrained=False  # Loading trained weights
    )
    
    try:
        # Try loading with weights_only=True first (safest)
        checkpoint = torch.load(CFG.model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint)
        print("Loaded model weights successfully (weights_only=True)")
        
    except Exception as e1:
        print(f"Failed with weights_only=True: {e1}")
        try:
            # Fallback: load full checkpoint
            checkpoint = torch.load(CFG.model_path, map_location='cpu', weights_only=False)
            
            # Load weights (match training checkpoint structure)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'best_score' in checkpoint:
                    print(f"Loaded model with best score: {checkpoint['best_score']:.6f}")
                if 'epoch' in checkpoint:
                    print(f"Best epoch: {checkpoint['epoch']}")
                if 'aneurysm_auc' in checkpoint:
                    print(f"Aneurysm AUC: {checkpoint['aneurysm_auc']:.6f}")
                if 'avg_individual_auc' in checkpoint:
                    print(f"Avg Individual AUC: {checkpoint['avg_individual_auc']:.6f}")
            else:
                model.load_state_dict(checkpoint)
            print("Loaded model with full checkpoint")
            
        except Exception as e2:
            print(f"Failed with full checkpoint: {e2}")
            # Last resort: try to extract only state_dict
            try:
                checkpoint = torch.load(CFG.model_path, map_location='cpu', weights_only=False)
                # Extract only the model weights
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                else:
                    state_dict = checkpoint
                
                model.load_state_dict(state_dict)
                print("Loaded model with extracted state_dict")
                
            except Exception as e3:
                raise RuntimeError(f"All loading methods failed: {e1}, {e2}, {e3}")
    
    # Move to device and set eval mode
    model = model.to(device)
    model.eval()
    
    return model

def initialize_model():
    """Initialize 8-frame model and warm up"""
    global MODEL
    
    if MODEL is None:
        MODEL = load_model()
        
        # Warm up model with 3-channel input + metadata
        print("Warming up 8-frame model...")
        dummy_input = torch.randn(1, 3, CFG.IMAGE_SIZE, CFG.IMAGE_SIZE).to(device)  # 3-channel
        dummy_metadata = torch.randn(1, 2).to(device)  # age + sex
        
        with torch.no_grad():
            with autocast(enabled=CFG.use_amp):
                _ = MODEL(dummy_input, dummy_metadata)
        
        print("8-frame model ready for inference!")

print("Model loading functions ready (8-frame + metadata)")


def predict_series(series_path: str) -> np.ndarray:
    """Make prediction for a single series using 8-frame model"""
    global MODEL
    
    # Initialize model if needed
    if MODEL is None:
        initialize_model()
    
    try:
        # Process DICOM series (8-frame 3-channel + metadata)
        image_tensor, metadata_tensor = process_dicom_series(series_path)
        
        # Add batch dimension and move to device
        image_tensor = image_tensor.unsqueeze(0).to(device)  # (1, C, H, W)
        metadata_tensor = metadata_tensor.unsqueeze(0).to(device)  # (1, 2)
        
        # Make prediction with metadata
        with torch.no_grad():
            with autocast(enabled=CFG.use_amp):
                logits = MODEL(image_tensor, metadata_tensor)
                probabilities = torch.sigmoid(logits)
        
        # Convert to numpy
        predictions = probabilities.cpu().numpy()[0]
        
        # Validate predictions
        predictions = np.clip(predictions, 0.0, 1.0)
        predictions = np.nan_to_num(predictions, nan=0.1)
        
        return predictions
        
    except Exception as e:
        if CFG.debug_mode:
            print(f"Error in prediction: {e}")
        return create_fallback_predictions()

def create_fallback_predictions() -> np.ndarray:
    """Create conservative fallback predictions (match training distribution)"""
    # Conservative predictions based on training data distribution
    fallback_values = np.array([
        0.05, 0.05, 0.08, 0.08,  # Carotid arteries
        0.12, 0.12,              # Middle cerebral arteries  
        0.15,                    # Anterior communicating
        0.06, 0.06,              # Anterior cerebral arteries
        0.07, 0.07,              # Posterior communicating
        0.09,                    # Basilar tip
        0.11,                    # Other posterior circulation
        0.43                     # Aneurysm present (training distribution)
    ])
    return fallback_values

def predict_inner(series_path: str) -> pl.DataFrame:
    """Internal prediction logic with 8-frame processing"""
    # Extract series ID for logging
    series_id = os.path.basename(series_path)
    
    if CFG.debug_mode:
        print(f"Processing series: {series_id} (8-frame)")
    
    # Make prediction using 8-frame model
    predictions = predict_series(series_path)
    
    # Create output dataframe (API requires no SeriesInstanceUID column)
    predictions_df = pl.DataFrame(
        data=[predictions.tolist()],
        schema=LABEL_COLS,
        orient='row'
    )
    
    if CFG.debug_mode:
        print(f"Prediction range: {predictions.min():.6f} - {predictions.max():.6f}")
        print(f"Aneurysm Present: {predictions[-1]:.6f}")
    
    return predictions_df

print("Prediction functions ready (8-frame + metadata)")


def predict(series_path: str) -> pl.DataFrame:
    """
    Main prediction function for Kaggle API (8-frame processing).
    This function is called by the inference server for each test series.
    """
    try:
        # Call internal prediction logic (8-frame)
        return predict_inner(series_path)
        
    except Exception as e:
        print(f"Error during 8-frame prediction for {os.path.basename(series_path)}: {e}")
        print("Using fallback predictions.")
        
        # Return fallback predictions
        fallback_preds = create_fallback_predictions()
        predictions_df = pl.DataFrame(
            data=[fallback_preds.tolist()],
            schema=LABEL_COLS,
            orient='row'
        )
        
        return predictions_df
        
    finally:
        # Required cleanup to prevent disk space issues
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

print("Main API function ready (8-frame)")


# Main execution
def main():
    """Main execution function for 8-frame inference"""
    print("="*70)
    print("RSNA INTRACRANIAL ANEURYSM DETECTION - 8-FRAME INFERENCE")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: 8-Frame EfficientNetV2-S with Metadata")
    print(f"Frames: {CFG.NUM_FRAMES}")
    print(f"Backbone: {CFG.MODEL_NAME_BACKBONE}")
    print(f"Image size: {CFG.IMAGE_SIZE}")
    print(f"Use windowing: {CFG.USE_WINDOWING}")
    print(f"Use CLAHE: {CFG.USE_CLAHE}")
    print(f"Use metadata: {CFG.USE_METADATA}")
    print(f"3-Channel input: {CFG.USE_3CHANNEL_INPUT}")
    print("-" * 70)
    
    try:
        # Pre-load 8-frame model
        initialize_model()
        
        # Initialize inference server
        print("Initializing inference server...")
        inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)
        
        # Run inference
        if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
            print("Running in competition mode...")
            inference_server.serve()
        else:
            print("Running in local gateway mode...")
            inference_server.run_local_gateway()
            
            # Display results if available
            submission_path = '/kaggle/working/submission.parquet'
            if os.path.exists(submission_path):
                try:
                    submission_df = pl.read_parquet(submission_path)
                    print(f"\nSubmission preview:")
                    print(f"Shape: {submission_df.shape}")
                    print(submission_df.head())
                    
                    # Show aneurysm present statistics
                    if 'Aneurysm Present' in submission_df.columns:
                        aneurysm_stats = submission_df['Aneurysm Present'].describe()
                        print(f"\nAneurysm Present statistics:")
                        print(aneurysm_stats)
                        
                except Exception as e:
                    print(f"Could not read submission file: {e}")
        
        print("\n" + "="*70)
        print("8-FRAME INFERENCE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"Critical error: {e}")
        print("This may indicate model loading or API configuration issues.")
        raise e

# Run main execution
if __name__ == "__main__":
    main()

