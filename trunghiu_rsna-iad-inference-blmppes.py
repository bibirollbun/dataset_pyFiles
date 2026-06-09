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
    # Model settings (must match training)
    num_frames = 16
    image_size = 224
    num_classes = 14
    
    # Model path
    model_path = "/kaggle/input/rsna-saved-models/best_png_baseline.pth"
    
    # Inference settings
    batch_size = 1
    use_amp = True
    use_windowing = True
    
    # Processing settings
    debug_mode = False

CFG = InferenceConfig()

# Add alias to match training code (important for pickle loading)
Config = InferenceConfig

print(f"Configuration loaded:")
print(f"- Frames: {CFG.num_frames}")
print(f"- Image size: {CFG.image_size}")
print(f"- Model path: {CFG.model_path}")
print(f"- Use windowing: {CFG.use_windowing}")
print(f"- Config alias created for compatibility")


# Multi-Frame EfficientNet model (identical to training)
class MultiFrameEfficientNet(nn.Module):
    def __init__(self, num_frames=16, num_classes=14, pretrained=True):
        super(MultiFrameEfficientNet, self).__init__()
        self.num_frames = num_frames
        self.num_classes = num_classes
        
        # Load EfficientNetB0 as backbone
        self.backbone = timm.create_model(
            'efficientnet_b0', 
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool='avg'
        )
        
        # Get feature dimension from backbone
        self.feature_dim = self.backbone.num_features  # 1280 for EfficientNetB0
        
        # Temporal aggregation layer
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, num_classes)
        )
        
    def forward(self, x):
        # x shape: (batch_size, num_frames, channels, height, width)
        batch_size, num_frames, channels, height, width = x.shape
        
        # Reshape to process all frames at once
        x = x.view(batch_size * num_frames, channels, height, width)
        
        # Extract features using backbone
        features = self.backbone(x)  # (batch_size * num_frames, feature_dim)
        
        # Reshape back to separate frames
        features = features.view(batch_size, num_frames, self.feature_dim)
        
        # Temporal aggregation: average pooling across frames
        # Transpose for AdaptiveAvgPool1d: (batch_size, feature_dim, num_frames)
        features = features.transpose(1, 2)
        pooled_features = self.temporal_pool(features)  # (batch_size, feature_dim, 1)
        pooled_features = pooled_features.squeeze(-1)  # (batch_size, feature_dim)
        
        # Classification (output logits, not probabilities)
        output = self.classifier(pooled_features)
        
        return output

print("Model architecture defined")


# DICOM processing utilities
def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Apply DICOM windowing to enhance image contrast"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)

def get_windowing_params(modality: str) -> Tuple[float, float]:
    """Get appropriate windowing for different modalities"""
    windows = {
        'CT': (40, 80),
        'CTA': (50, 350),
        'MRA': (600, 1200),
        'MRI': (40, 80),
        'MRI T2': (40, 80),
        'MRI T1post': (40, 80),
    }
    return windows.get(modality, (50, 350))  # Default to CTA

def extract_sort_key(path: str) -> Tuple[float, float, str]:
    """Extract sorting key from DICOM file for proper ordering"""
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
    """Sort DICOM paths by medical metadata for proper slice ordering"""
    if not dcm_paths:
        return []
    
    sort_info = []
    for path in dcm_paths:
        sort_info.append(extract_sort_key(path))
    
    sort_info.sort()
    return [x[2] for x in sort_info]

print("DICOM processing functions ready")


def process_single_dicom(dicom_path: str, modality: str = 'CTA') -> Optional[np.ndarray]:
    """Process a single DICOM file and return processed image"""
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
            
        # Apply windowing if requested
        if CFG.use_windowing:
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
            
        # Validate before resize
        if img.shape[0] == 0 or img.shape[1] == 0:
            return None
            
        # Resize to target size
        img = cv2.resize(img, (CFG.image_size, CFG.image_size))
        
        # Convert to RGB (3 channels)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        return img
        
    except Exception as e:
        if CFG.debug_mode:
            print(f"Error processing {dicom_path}: {e}")
        return None

def process_dicom_series(series_path: str) -> np.ndarray:
    """Process DICOM series and return multi-frame tensor"""
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
        return create_dummy_tensor()
    
    # Sort files by medical metadata
    sorted_files = sort_dicom_paths(dicom_files)
    
    # Get modality from first file
    try:
        first_dicom = pydicom.dcmread(sorted_files[0], stop_before_pixels=True)
        modality = getattr(first_dicom, 'Modality', 'CTA')
    except:
        modality = 'CTA'
    
    # Process each DICOM file
    processed_images = []
    for dicom_path in sorted_files:
        img = process_single_dicom(dicom_path, modality)
        if img is not None:
            processed_images.append(img)
    
    if not processed_images:
        if CFG.debug_mode:
            print(f"Warning: No images processed successfully for {series_path}")
        return create_dummy_tensor()
    
    # Sample frames to match target number
    sampled_images = sample_frames(processed_images, CFG.num_frames)
    
    # Apply normalization (match training)
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    frame_tensors = []
    for img in sampled_images:
        try:
            transformed = transform(image=img)
            frame_tensors.append(transformed['image'])
        except:
            # Create dummy tensor on transform failure
            dummy_img = np.zeros((CFG.image_size, CFG.image_size, 3), dtype=np.uint8)
            transformed = transform(image=dummy_img)
            frame_tensors.append(transformed['image'])
    
    # Stack frames
    multi_frame_tensor = torch.stack(frame_tensors)  # (num_frames, C, H, W)
    
    return multi_frame_tensor

def sample_frames(images: List[np.ndarray], target_frames: int) -> List[np.ndarray]:
    """Sample frames to match target number (same logic as training)"""
    total_frames = len(images)
    
    if total_frames >= target_frames:
        # Uniform subsampling
        indices = np.linspace(0, total_frames-1, target_frames, dtype=int)
    else:
        # Repeat frames to reach target number
        repeat_factor = target_frames // total_frames
        remainder = target_frames % total_frames
        
        indices = list(range(total_frames)) * repeat_factor
        if remainder > 0:
            indices.extend(np.linspace(0, total_frames-1, remainder, dtype=int))
    
    return [images[i] for i in indices[:target_frames]]

def create_dummy_tensor() -> torch.Tensor:
    """Create dummy tensor when processing fails"""
    transform = A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])
    
    dummy_images = []
    for _ in range(CFG.num_frames):
        dummy_img = np.zeros((CFG.image_size, CFG.image_size, 3), dtype=np.uint8)
        transformed = transform(image=dummy_img)
        dummy_images.append(transformed['image'])
    
    return torch.stack(dummy_images)

print("DICOM series processing ready")


# Global variables
MODEL = None

# Add this alias to match training code's Config class
Config = InferenceConfig

def load_model() -> nn.Module:
    """Load trained model with Config compatibility fix"""
    print(f"Loading model from: {CFG.model_path}")
    
    if not os.path.exists(CFG.model_path):
        raise FileNotFoundError(f"Model file not found: {CFG.model_path}")
    
    # Initialize model first
    model = MultiFrameEfficientNet(
        num_frames=CFG.num_frames,
        num_classes=CFG.num_classes,
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
            
            # Load weights
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'best_score' in checkpoint:
                    print(f"Loaded model with best score: {checkpoint['best_score']:.6f}")
                if 'epoch' in checkpoint:
                    print(f"Best epoch: {checkpoint['epoch']}")
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
    """Initialize model and warm up"""
    global MODEL
    
    if MODEL is None:
        MODEL = load_model()
        
        # Warm up model
        print("Warming up model...")
        dummy_input = torch.randn(1, CFG.num_frames, 3, CFG.image_size, CFG.image_size).to(device)
        
        with torch.no_grad():
            with autocast(enabled=CFG.use_amp):
                _ = MODEL(dummy_input)
        
        print("Model ready for inference!")

print("Model loading functions ready (with Config fix)")


def predict_series(series_path: str) -> np.ndarray:
    """Make prediction for a single series"""
    global MODEL
    
    # Initialize model if needed
    if MODEL is None:
        initialize_model()
    
    try:
        # Process DICOM series
        series_tensor = process_dicom_series(series_path)
        
        # Add batch dimension and move to device
        series_tensor = series_tensor.unsqueeze(0).to(device)  # (1, num_frames, C, H, W)
        
        # Make prediction
        with torch.no_grad():
            with autocast(enabled=CFG.use_amp):
                logits = MODEL(series_tensor)
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
    """Create conservative fallback predictions"""
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

def _predict_inner(series_path: str) -> pl.DataFrame:
    """Internal prediction logic"""
    # Extract series ID for logging
    series_id = os.path.basename(series_path)
    
    if CFG.debug_mode:
        print(f"Processing series: {series_id}")
    
    # Make prediction
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

print("Prediction functions ready")


def predict(series_path: str) -> pl.DataFrame:
    """
    Main prediction function for Kaggle API.
    This function is called by the inference server for each test series.
    """
    try:
        # Call internal prediction logic
        return _predict_inner(series_path)
        
    except Exception as e:
        print(f"Error during prediction for {os.path.basename(series_path)}: {e}")
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

print("Main API function ready")


# Main execution
def main():
    """Main execution function"""
    print("="*70)
    print("RSNA INTRACRANIAL ANEURYSM DETECTION - INFERENCE")
    print("="*70)
    print(f"Device: {device}")
    print(f"Model: Multi-Frame EfficientNetB0")
    print(f"Frames: {CFG.num_frames}")
    print(f"Image size: {CFG.image_size}")
    print(f"Use windowing: {CFG.use_windowing}")
    print("-" * 70)
    
    try:
        # Pre-load model
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
                except Exception as e:
                    print(f"Could not read submission file: {e}")
        
        print("\n" + "="*70)
        print("INFERENCE COMPLETED SUCCESSFULLY!")
        print("="*70)
        
    except Exception as e:
        print(f"Critical error: {e}")
        print("This may indicate model loading or API configuration issues.")
        raise e

# Run main execution
if __name__ == "__main__":
    main()

