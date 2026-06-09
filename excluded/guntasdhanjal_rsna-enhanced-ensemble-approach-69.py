import os
import sys
import gc
import json
import shutil
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple
from IPython.display import display

# Data handling
import numpy as np
import polars as pl
import pandas as pd

# Medical imaging
import pydicom
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import kaggle_evaluation.rsna_inference_server

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# Competition constants - these are the 14 target labels
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

# Enhanced Model Configuration
SELECTED_MODEL = 'ensemble'  # Options: 'tf_efficientnetv2_s', 'convnext_small', 'swin_small_patch4_window7_224', 'ensemble'

MODEL_PATHS = {
    'tf_efficientnetv2_s': '/kaggle/input/rsna-iad-trained-models/models/tf_efficientnetv2_s_fold0_best.pth',
    'convnext_small': '/kaggle/input/rsna-iad-trained-models/models/convnext_small_fold0_best.pth',
    'swin_small_patch4_window7_224': '/kaggle/input/rsna-iad-trained-models/models/swin_small_patch4_window7_224_fold0_best.pth'
}

class InferenceConfig:
    # Model selection
    model_selection = SELECTED_MODEL
    use_ensemble = (SELECTED_MODEL == 'ensemble')
    
    # Enhanced image processing
    image_size = 512  # Increased from typical 224 for better detail
    num_slices = 32   # Balanced between context and efficiency
    use_windowing = True
    
    # Improved inference settings
    batch_size = 1
    use_amp = True
    use_tta = True
    tta_transforms = 8  # Increased TTA for better robustness
    
    # Optimized ensemble weights based on validation performance
    ensemble_weights = {
        'tf_efficientnetv2_s': 0.4,    # Strong performer on this dataset
        'convnext_small': 0.3,         # Good balance of speed/accuracy
        'swin_small_patch4_window7_224': 0.3  # Excellent for spatial relationships
    }

CFG = InferenceConfig()

# Global variables for model management
MODELS = {}
TRANSFORM = None
TTA_TRANSFORMS = None


class MultiBackboneModel(nn.Module):
    """
    Flexible model that can use different backbones with metadata integration.
    
    Key features:
    - Supports CNN and Transformer architectures
    - Integrates patient metadata (age, sex)
    - Robust feature extraction with proper pooling
    """
    
    def __init__(self, model_name, num_classes=14, pretrained=True, 
                 drop_rate=0.3, drop_path_rate=0.2):
        super().__init__()
        
        self.model_name = model_name
        
        # Initialize backbone based on architecture type
        if 'swin' in model_name:
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained,
                in_chans=3,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                img_size=CFG.image_size,
                num_classes=0,
                global_pool=''
            )
        else:
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained,
                in_chans=3,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                num_classes=0,
                global_pool=''
            )
        
        # Auto-detect feature dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, CFG.image_size, CFG.image_size)
            features = self.backbone(dummy_input)
            
            if len(features.shape) == 4:
                num_features = features.shape[1]
                self.needs_pool = True
            elif len(features.shape) == 3:
                num_features = features.shape[-1]
                self.needs_pool = False
                self.needs_seq_pool = True
            else:
                num_features = features.shape[1]
                self.needs_pool = False
                self.needs_seq_pool = False
        
        print(f"Model {model_name}: detected {num_features} features, output shape: {features.shape}")
        
        if self.needs_pool:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Enhanced metadata processing
        self.meta_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # Robust classifier with batch normalization
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, image, meta):
        # Extract and pool image features appropriately
        img_features = self.backbone(image)
        
        if hasattr(self, 'needs_pool') and self.needs_pool:
            img_features = self.global_pool(img_features)
            img_features = img_features.flatten(1)
        elif hasattr(self, 'needs_seq_pool') and self.needs_seq_pool:
            img_features = img_features.mean(dim=1)
        elif len(img_features.shape) == 4:
            img_features = F.adaptive_avg_pool2d(img_features, 1).flatten(1)
        elif len(img_features.shape) == 3:
            img_features = img_features.mean(dim=1)
        
        # Process metadata
        meta_features = self.meta_fc(meta)
        
        # Combine and classify
        combined = torch.cat([img_features, meta_features], dim=1)
        output = self.classifier(combined)
        
        return output


def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """
    Apply DICOM windowing to enhance contrast for specific tissue types.
    This is crucial for medical imaging as different modalities require different contrast.
    """
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)

def get_windowing_params(modality: str) -> Tuple[float, float]:
    """
    Get optimized windowing parameters for different imaging modalities.
    These values are based on medical imaging best practices.
    """
    windows = {
        'CT': (40, 80),      # Brain window
        'CTA': (50, 350),    # Angiography window
        'MRA': (600, 1200),  # MR angiography
        'MRI': (40, 80),     # Standard brain
    }
    return windows.get(modality, (40, 80))

def extract_enhanced_metadata(ds) -> Dict:
    """Extract and validate metadata from DICOM headers"""
    metadata = {}
    
    # Modality extraction
    metadata['modality'] = getattr(ds, 'Modality', 'CT')
    
    # Enhanced age processing
    try:
        age_str = getattr(ds, 'PatientAge', '050Y')
        age = int(''.join(filter(str.isdigit, age_str[:3])) or '50')
        metadata['age'] = min(max(age, 0), 100)  # Clamp between 0-100
    except:
        metadata['age'] = 50
    
    # Sex processing
    try:
        sex = getattr(ds, 'PatientSex', 'M')
        metadata['sex'] = 1 if sex.upper() == 'M' else 0
    except:
        metadata['sex'] = 0
    
    # Additional metadata that could be useful
    metadata['slice_thickness'] = getattr(ds, 'SliceThickness', 1.0)
    metadata['pixel_spacing'] = getattr(ds, 'PixelSpacing', [1.0, 1.0])
    
    return metadata

def extract_enhanced_metadata(ds) -> Dict:
    """Extract and validate metadata from DICOM headers"""
    metadata = {}
    
    # Modality extraction
    metadata['modality'] = getattr(ds, 'Modality', 'CT')
    
    # Enhanced age processing
    try:
        age_str = getattr(ds, 'PatientAge', '050Y')
        age = int(''.join(filter(str.isdigit, age_str[:3])) or '50')
        metadata['age'] = min(max(age, 0), 100)  # Clamp between 0-100
    except:
        metadata['age'] = 50
    
    # Sex processing
    try:
        sex = getattr(ds, 'PatientSex', 'M')
        metadata['sex'] = 1 if sex.upper() == 'M' else 0
    except:
        metadata['sex'] = 0
    
    return metadata

def create_enhanced_multichannel_input(volume: np.ndarray) -> np.ndarray:
    """Create sophisticated multi-channel input"""
    # Channel 1: Middle slice (anatomical detail)
    middle_slice = volume[CFG.num_slices // 2]
    
    # Channel 2: Maximum Intensity Projection (highlights vessels)
    mip = np.max(volume, axis=0)
    
    # Channel 3: Standard deviation projection (highlights variability)
    std_proj = np.std(volume, axis=0).astype(np.float32)
    
    # Normalize std projection
    if std_proj.max() > std_proj.min():
        std_proj = ((std_proj - std_proj.min()) / (std_proj.max() - std_proj.min()) * 255).astype(np.uint8)
    else:
        std_proj = np.zeros_like(std_proj, dtype=np.uint8)
    
    # Stack channels
    image = np.stack([middle_slice, mip, std_proj], axis=-1)
    return image


def process_dicom_series(series_path: str) -> Tuple[np.ndarray, Dict]:
    """
    Enhanced DICOM series processing with robust error handling.
    
    Returns:
        - volume: 3D numpy array of processed slices
        - metadata: Dictionary containing patient and imaging parameters
    """
    series_path = Path(series_path)
    
    # Find all DICOM files
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()
    
    if len(all_filepaths) == 0:
        print(f"Warning: No DICOM files found in {series_path}")
        volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
        metadata = {'age': 50, 'sex': 0, 'modality': 'CT'}
        return volume, metadata
    
    slices = []
    metadata = {}
    
    for i, filepath in enumerate(all_filepaths):
        try:
            ds = pydicom.dcmread(filepath, force=True)
            img = ds.pixel_array.astype(np.float32)
            
            # Handle different image formats
            if img.ndim == 3:
                if img.shape[-1] == 3:  # RGB
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:  # Multi-frame
                    img = img[0] if img.shape[0] < img.shape[-1] else img[:, :, 0]
            
            # Extract metadata from first file
            if i == 0:
                metadata = extract_enhanced_metadata(ds)
            
            # Apply rescaling if available
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
            
            # Apply modality-specific windowing
            if CFG.use_windowing:
                window_center, window_width = get_windowing_params(metadata['modality'])
                img = apply_dicom_windowing(img, window_center, window_width)
            else:
                # Normalize to 0-255
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            
            # Resize to target size
            img = cv2.resize(img, (CFG.image_size, CFG.image_size))
            slices.append(img)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    # Smart slice sampling
    if len(slices) == 0:
        volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
    else:
        volume = np.array(slices)
        if len(slices) > CFG.num_slices:
            # Use linear sampling to maintain spatial relationships
            indices = np.linspace(0, len(slices) - 1, CFG.num_slices).astype(int)
            volume = volume[indices]
        elif len(slices) < CFG.num_slices:
            # Pad with edge slices rather than zeros
            pad_size = CFG.num_slices - len(slices)
            volume = np.pad(volume, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
    
    return volume, metadata


def get_inference_transform():
    """Standard inference normalization"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_enhanced_tta_transforms():
    """
    Enhanced test-time augmentation with medical imaging considerations.
    
    I've carefully selected augmentations that preserve anatomical relationships
    while providing meaningful variations for ensemble predictions.
    """
    transforms = [
        # Original
        A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Horizontal flip (preserves medical anatomy)
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Slight rotation (medical scans can have positioning variations)
        A.Compose([
            A.Rotate(limit=5, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Minor scaling (simulates different patient sizes)
        A.Compose([
            A.RandomScale(scale_limit=0.05, p=1.0),
            A.PadIfNeeded(min_height=CFG.image_size, min_width=CFG.image_size, border_mode=0),
            A.CenterCrop(CFG.image_size, CFG.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Contrast adjustment (simulates different scan settings)
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Combination transforms
        A.Compose([
            A.HorizontalFlip(p=1.0),
            A.Rotate(limit=3, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.05, contrast_limit=0.05, p=1.0),
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        
        # Elastic deformation (very subtle, simulates slight positioning differences)
        A.Compose([
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=0, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
    ]
    
    return transforms


def load_single_model(model_name: str, model_path: str) -> nn.Module:
    """Load and initialize a single model with proper error handling"""
    print(f"Loading {model_name} from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract configurations
    model_config = checkpoint.get('model_config', {})
    training_config = checkpoint.get('training_config', {})
    
    # Update global config if needed
    if 'image_size' in training_config:
        CFG.image_size = training_config['image_size']
    
    # Initialize model
    model = MultiBackboneModel(
        model_name=model_name,
        num_classes=training_config.get('num_classes', 14),
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.0
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    best_score = checkpoint.get('best_score', 'N/A')
    print(f"âœ“ Loaded {model_name} with validation score: {best_score}")
    
    return model

def load_models():
    """Load all required models based on configuration"""
    global MODELS, TRANSFORM, TTA_TRANSFORMS
    
    print(" Loading models...")
    
    if CFG.use_ensemble:
        print("Using ensemble approach with multiple models:")
        for model_name, model_path in MODEL_PATHS.items():
            try:
                MODELS[model_name] = load_single_model(model_name, model_path)
                print(f"    {model_name} loaded successfully")
            except Exception as e:
                print(f"    Could not load {model_name}: {e}")
    else:
        print(f"Using single model: {CFG.model_selection}")
        if CFG.model_selection in MODEL_PATHS:
            model_path = MODEL_PATHS[CFG.model_selection]
            MODELS[CFG.model_selection] = load_single_model(CFG.model_selection, model_path)
        else:
            raise ValueError(f"Unknown model: {CFG.model_selection}")
    
    # Initialize transforms
    TRANSFORM = get_inference_transform()
    if CFG.use_tta:
        TTA_TRANSFORMS = get_enhanced_tta_transforms()
        print(f"âœ“ TTA enabled with {len(TTA_TRANSFORMS)} transforms")
    
    print(f"âœ“ Models ready: {list(MODELS.keys())}")
    
    # Model warm-up
    print(" Warming up models...")
    dummy_image = torch.randn(1, 3, CFG.image_size, CFG.image_size).to(device)
    dummy_meta = torch.randn(1, 2).to(device)
    
    with torch.no_grad():
        for name, model in MODELS.items():
            _ = model(dummy_image, dummy_meta)
            print(f"   âœ“ {name} warmed up")
    
    print("ğŸš€ Ready for inference!")


def create_enhanced_multichannel_input(volume: np.ndarray) -> np.ndarray:
    """
    Create sophisticated multi-channel input that captures different aspects of the data.
    
    This is one of my key innovations - instead of just using raw slices,
    I create different projections that highlight different anatomical features.
    """
    
    # Channel 1: Middle slice (anatomical detail)
    middle_slice = volume[CFG.num_slices // 2]
    
    # Channel 2: Maximum Intensity Projection (highlights vessels)
    mip = np.max(volume, axis=0)
    
    # Channel 3: Standard deviation projection (highlights variability/movement)
    std_proj = np.std(volume, axis=0).astype(np.float32)
    
    # Normalize std projection
    if std_proj.max() > std_proj.min():
        std_proj = ((std_proj - std_proj.min()) / (std_proj.max() - std_proj.min()) * 255).astype(np.uint8)
    else:
        std_proj = np.zeros_like(std_proj, dtype=np.uint8)
    
    # Stack channels
    image = np.stack([middle_slice, mip, std_proj], axis=-1)
    
    return image

def predict_single_model(model: nn.Module, image: np.ndarray, meta_tensor: torch.Tensor) -> np.ndarray:
    """Make robust predictions with a single model using TTA"""
    predictions = []
    
    if CFG.use_tta and TTA_TRANSFORMS:
        # Test-time augmentation for better robustness
        for i, transform in enumerate(TTA_TRANSFORMS[:CFG.tta_transforms]):
            try:
                aug_image = transform(image=image)['image']
                aug_image = aug_image.unsqueeze(0).to(device)
                
                with torch.no_grad():
                    with autocast(enabled=CFG.use_amp):
                        output = model(aug_image, meta_tensor)
                        pred = torch.sigmoid(output)
                        predictions.append(pred.cpu().numpy())
            except Exception as e:
                print(f"Warning: TTA transform {i} failed: {e}")
                continue
        
        if predictions:
            # Average TTA predictions
            return np.mean(predictions, axis=0).squeeze()
        else:
            # Fallback to single prediction if all TTA failed
            print("Warning: All TTA transforms failed, using single prediction")
    
    # Single prediction (fallback or no TTA)
    image_tensor = TRANSFORM(image=image)['image']
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    with torch.no_grad():
        with autocast(enabled=CFG.use_amp):
            output = model(image_tensor, meta_tensor)
            return torch.sigmoid(output).cpu().numpy().squeeze()

def predict_ensemble(image: np.ndarray, meta_tensor: torch.Tensor) -> np.ndarray:
    """
    Make ensemble predictions with sophisticated weighting.
    
    I use weighted averaging based on each model's validation performance.
    """
    all_predictions = []
    weights = []
    
    for model_name, model in MODELS.items():
        try:
            pred = predict_single_model(model, image, meta_tensor)
            all_predictions.append(pred)
            weights.append(CFG.ensemble_weights.get(model_name, 1.0))
            print(f"âœ“ {model_name} prediction completed")
        except Exception as e:
            print(f"â�Œ {model_name} prediction failed: {e}")
            continue
    
    if not all_predictions:
        print("â�Œ All model predictions failed!")
        return np.full(14, 0.1)  # Conservative fallback
    
    # Weighted ensemble
    weights = np.array(weights) / np.sum(weights)
    predictions = np.array(all_predictions)
    
    final_pred = np.average(predictions, weights=weights, axis=0)
    
    print(f"âœ“ Ensemble completed with {len(all_predictions)} models")
    return final_pred

def _predict_inner(series_path: str) -> pl.DataFrame:
    """
    Main prediction logic with comprehensive error handling.
    
    This is where all the magic happens - from DICOM processing
    to final predictions.
    """
    global MODELS
    
    # Ensure models are loaded
    if not MODELS:
        load_models()
    
    # Extract series identifier
    series_id = os.path.basename(series_path)
    print(f"ğŸ”� Processing series: {series_id}")
    
    try:
        # Process DICOM series
        volume, metadata = process_dicom_series(series_path)
        print(f"âœ“ Processed {volume.shape[0]} slices, modality: {metadata['modality']}")
        
        # Create enhanced multi-channel input
        image = create_enhanced_multichannel_input(volume)
        
        # Prepare metadata tensor
        age_normalized = metadata['age'] / 100.0  # Normalize age
        sex = metadata['sex']
        meta_tensor = torch.tensor([[age_normalized, sex]], dtype=torch.float32).to(device)
        
        print(f"âœ“ Patient info - Age: {metadata['age']}, Sex: {'M' if sex else 'F'}")
        
        # Make predictions
        if CFG.use_ensemble:
            final_pred = predict_ensemble(image, meta_tensor)
        else:
            model = MODELS[CFG.model_selection]
            final_pred = predict_single_model(model, image, meta_tensor)
        
        # Create output DataFrame
        predictions_df = pl.DataFrame(
            data=[[series_id] + final_pred.tolist()],
            schema=[ID_COL] + LABEL_COLS,
            orient='row'
        )
        
        # Log prediction summary
        aneurysm_prob = final_pred[-1]  # Last column is "Aneurysm Present"
        print(f"âœ“ Prediction completed - Aneurysm probability: {aneurysm_prob:.4f}")
        
        # Return without ID column as required by API
        return predictions_df.drop(ID_COL)
        
    except Exception as e:
        print(f" Error processing {series_id}: {e}")
        # Return conservative predictions
        return pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
            schema=LABEL_COLS,
            orient='row'
        )


def predict_fallback(series_path: str) -> pl.DataFrame:
    """
    Fallback prediction function for when everything goes wrong.
    
    In medical AI, it's better to give conservative predictions
    than to crash the system.
    """
    series_id = os.path.basename(series_path)
    print(f"âš ï¸�  Using fallback predictions for {series_id}")
    
    # Conservative predictions (low probability for all locations)
    predictions = pl.DataFrame(
        data=[[0.1] * len(LABEL_COLS)],
        schema=LABEL_COLS,
        orient='row'
    )
    
    # Clean up any leftover files
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    return predictions

def predict(series_path: str) -> pl.DataFrame:
    """
    Top-level prediction function with comprehensive error handling.
    
    This function is called by the Kaggle inference server for each series.
    It guarantees cleanup and never crashes, which is crucial for competition.
    """
    try:
        return _predict_inner(series_path)
    
    except torch.cuda.OutOfMemoryError:
        print(" CUDA out of memory! Cleaning up and retrying...")
        torch.cuda.empty_cache()
        gc.collect()
        
        try:
            return _predict_inner(series_path)
        except:
            print(" Retry failed, using fallback")
            return predict_fallback(series_path)
    
    except Exception as e:
        print(f" Unexpected error: {e}")
        print("Using conservative fallback predictions")
        
        # Return safe predictions
        predictions = pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions
    
    finally:
        # Critical cleanup to prevent "out of disk space" errors
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# Load models at startup
print("*Starting RSNA Intracranial Aneurysm Detection Inference")
print("=" * 60)

load_models()

print("\n" + "=" * 60)
print("*Configuration Summary:")
print(f"   â€¢ Strategy: {CFG.model_selection}")
print(f"   â€¢ Image Size: {CFG.image_size}x{CFG.image_size}")
print(f"   â€¢ TTA Transforms: {CFG.tta_transforms}")
print(f"   â€¢ Models: {list(MODELS.keys())}")
print("=" * 60)

# Initialize the inference server
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

# Run inference
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("*Running in competition mode...")
    inference_server.serve()
else:
    print("ğŸ§ª Running in local test mode...")
    inference_server.run_local_gateway()
    
    # Display results
    try:
        submission_df = pl.read_parquet('/kaggle/working/submission.parquet')
        print("\n Submission Preview:")
        display(submission_df.head())
        
        # Quick statistics
        aneurysm_present_mean = submission_df['Aneurysm Present'].mean()
        print(f"\n Average aneurysm probability: {aneurysm_present_mean:.4f}")
        
    except Exception as e:
        print(f"Could not load submission file: {e}")

print("\n Inference completed successfully!")







