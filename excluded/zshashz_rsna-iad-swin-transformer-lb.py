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

# Model selection - Change this to select which model to use for inference
# Options: 'tf_efficientnetv2_s', 'convnext_small', 'swin_small_patch4_window7_224', 'ensemble'
SELECTED_MODEL = 'swin_small_patch4_window7_224' 


# Model paths configuration
MODEL_PATHS = {
    'tf_efficientnetv2_s': '/kaggle/input/rsna-iad-trained-models/models/tf_efficientnetv2_s_fold0_best.pth',
    'convnext_small': '/kaggle/input/rsna-iad-trained-models/models/convnext_small_fold0_best.pth',
    'swin_small_patch4_window7_224': '/kaggle/input/rsna-iad-trained-models/models/swin_small_patch4_window7_224_fold0_best.pth'
}

class InferenceConfig:
    # Model selection
    model_selection = SELECTED_MODEL
    use_ensemble = (SELECTED_MODEL == 'ensemble')
    
    # Default model settings (will be overridden by checkpoint)
    image_size = 512
    num_slices = 32
    use_windowing = True
    
    # Inference settings
    batch_size = 1
    use_amp = True
    use_tta = True
    tta_transforms = 4
    
    # Ensemble weights (if using ensemble)
    ensemble_weights = {
        'tf_efficientnetv2_s': 0.4,
        'convnext_small': 0.3,
        'swin_small_patch4_window7_224': 0.3
    }

CFG = InferenceConfig()



class MultiBackboneModel(nn.Module):
    """Flexible model that can use different backbones"""
    def __init__(self, model_name, num_classes=14, pretrained=True, 
                 drop_rate=0.3, drop_path_rate=0.2):
        super().__init__()
        
        self.model_name = model_name
        
        if 'swin' in model_name:
            # Swin transformer requires 224x224 by default
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained,
                in_chans=3,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                img_size=CFG.image_size,  # Override default size
                num_classes=0,  # Remove classifier head
                global_pool=''  # Remove global pooling
            )
        else:
            self.backbone = timm.create_model(
                model_name, 
                pretrained=pretrained,
                in_chans=3,
                drop_rate=drop_rate,
                drop_path_rate=drop_path_rate,
                num_classes=0,  # Remove classifier head
                global_pool=''  # Remove global pooling
            )
        
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, CFG.image_size, CFG.image_size)
            features = self.backbone(dummy_input)
            
            if len(features.shape) == 4:
                # Conv features (batch, channels, height, width)
                num_features = features.shape[1]
                self.needs_pool = True
            elif len(features.shape) == 3:
                # Transformer features (batch, sequence, features)
                num_features = features.shape[-1]
                self.needs_pool = False
                self.needs_seq_pool = True
            else:
                # Already flat features (batch, features)
                num_features = features.shape[1]
                self.needs_pool = False
                self.needs_seq_pool = False
        
        print(f"Model {model_name}: detected {num_features} features, output shape: {features.shape}")
        
        # Add global pooling for models that output spatial features
        if self.needs_pool:
            self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Metadata processing
        self.meta_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # Combined classifier with batch norm for stability
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
        # Extract image features
        img_features = self.backbone(image)
        
        # Apply appropriate pooling based on model type
        if hasattr(self, 'needs_pool') and self.needs_pool:
            # Conv features - apply global pooling
            img_features = self.global_pool(img_features)
            img_features = img_features.flatten(1)
        elif hasattr(self, 'needs_seq_pool') and self.needs_seq_pool:
            # Transformer features - average across sequence dimension
            img_features = img_features.mean(dim=1)
        elif len(img_features.shape) == 4:
            # Fallback for any 4D output
            img_features = F.adaptive_avg_pool2d(img_features, 1).flatten(1)
        elif len(img_features.shape) == 3:
            # Fallback for any 3D output
            img_features = img_features.mean(dim=1)
        
        # Process metadata
        meta_features = self.meta_fc(meta)
        
        # Combine features
        combined = torch.cat([img_features, meta_features], dim=1)
        
        # Classification
        output = self.classifier(combined)
        
        return output


def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    """Apply DICOM windowing"""
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
    }
    return windows.get(modality, (40, 80))

def process_dicom_series(series_path: str) -> Tuple[np.ndarray, Dict]:
    """Process a DICOM series and extract metadata"""
    series_path = Path(series_path)
    
    # Find all DICOM files
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()
    
    if len(all_filepaths) == 0:
        # Return default values
        volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
        metadata = {'age': 50, 'sex': 0, 'modality': 'CT'}
        return volume, metadata
    
    # Process DICOM files
    slices = []
    metadata = {}
    
    for i, filepath in enumerate(all_filepaths):
        try:
            ds = pydicom.dcmread(filepath, force=True)
            img = ds.pixel_array.astype(np.float32)
            
            # Handle multi-frame or color images
            if img.ndim == 3:
                if img.shape[-1] == 3:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    img = img[:, :, 0]
            
            # Extract metadata from first file
            if i == 0:
                metadata['modality'] = getattr(ds, 'Modality', 'CT')
                
                try:
                    age_str = getattr(ds, 'PatientAge', '050Y')
                    age = int(''.join(filter(str.isdigit, age_str[:3])) or '50')
                    metadata['age'] = min(age, 100)
                except:
                    metadata['age'] = 50
                
                try:
                    sex = getattr(ds, 'PatientSex', 'M')
                    metadata['sex'] = 1 if sex == 'M' else 0
                except:
                    metadata['sex'] = 0
            
            # Apply rescale if available
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept
            
            # Apply windowing
            if CFG.use_windowing:
                window_center, window_width = get_windowing_params(metadata['modality'])
                img = apply_dicom_windowing(img, window_center, window_width)
            else:
                img_min, img_max = img.min(), img.max()
                if img_max > img_min:
                    img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                else:
                    img = np.zeros_like(img, dtype=np.uint8)
            
            # Resize
            img = cv2.resize(img, (CFG.image_size, CFG.image_size))
            slices.append(img)
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    # Handle slice sampling
    if len(slices) == 0:
        volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
    else:
        volume = np.array(slices)
        if len(slices) > CFG.num_slices:
            indices = np.linspace(0, len(slices) - 1, CFG.num_slices).astype(int)
            volume = volume[indices]
        elif len(slices) < CFG.num_slices:
            pad_size = CFG.num_slices - len(slices)
            volume = np.pad(volume, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
    
    return volume, metadata



def get_inference_transform():
    """Get inference transformation"""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

def get_tta_transforms():
    """Get test time augmentation transforms"""
    transforms = [
        A.Compose([  # Original
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([  # Horizontal flip
            A.HorizontalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([  # Vertical flip
            A.VerticalFlip(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ]),
        A.Compose([  # 90 degree rotation
            A.RandomRotate90(p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2()
        ])
    ]
    return transforms




# Global variables
MODELS = {}
TRANSFORM = None
TTA_TRANSFORMS = None

def load_single_model(model_name: str, model_path: str) -> nn.Module:
    """Load a single model"""
    print(f"Loading {model_name} from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint with weights_only=False to handle numpy scalars
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract config
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
    
    print(f"Loaded {model_name} with best score: {checkpoint.get('best_score', 'N/A'):.4f}")
    
    return model

def load_models():
    """Load models based on configuration"""
    global MODELS, TRANSFORM, TTA_TRANSFORMS
    
    print("Loading models...")
    
    if CFG.use_ensemble:
        # Load all models for ensemble
        for model_name, model_path in MODEL_PATHS.items():
            try:
                MODELS[model_name] = load_single_model(model_name, model_path)
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {e}")
    else:
        # Load single selected model
        if CFG.model_selection in MODEL_PATHS:
            model_path = MODEL_PATHS[CFG.model_selection]
            MODELS[CFG.model_selection] = load_single_model(CFG.model_selection, model_path)
        else:
            raise ValueError(f"Unknown model: {CFG.model_selection}")
    
    # Initialize transforms
    TRANSFORM = get_inference_transform()
    if CFG.use_tta:
        TTA_TRANSFORMS = get_tta_transforms()
    
    print(f"Models loaded: {list(MODELS.keys())}")
    
    # Warm up models
    print("Warming up models...")
    dummy_image = torch.randn(1, 3, CFG.image_size, CFG.image_size).to(device)
    dummy_meta = torch.randn(1, 2).to(device)
    
    with torch.no_grad():
        for model in MODELS.values():
            _ = model(dummy_image, dummy_meta)
    
    print("Ready for inference!")





def predict_single_model(model: nn.Module, image: np.ndarray, meta_tensor: torch.Tensor) -> np.ndarray:
    """Make prediction with a single model"""
    predictions = []
    
    if CFG.use_tta and TTA_TRANSFORMS:
        # Test time augmentation
        for transform in TTA_TRANSFORMS[:CFG.tta_transforms]:
            aug_image = transform(image=image)['image']
            aug_image = aug_image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                with autocast(enabled=CFG.use_amp):
                    output = model(aug_image, meta_tensor)
                    pred = torch.sigmoid(output)
                    predictions.append(pred.cpu().numpy())
        
        # Average TTA predictions
        return np.mean(predictions, axis=0).squeeze()
    else:
        # Single prediction
        image_tensor = TRANSFORM(image=image)['image']
        image_tensor = image_tensor.unsqueeze(0).to(device)
        
        with torch.no_grad():
            with autocast(enabled=CFG.use_amp):
                output = model(image_tensor, meta_tensor)
                return torch.sigmoid(output).cpu().numpy().squeeze()

def predict_ensemble(image: np.ndarray, meta_tensor: torch.Tensor) -> np.ndarray:
    """Make ensemble prediction"""
    all_predictions = []
    weights = []
    
    for model_name, model in MODELS.items():
        pred = predict_single_model(model, image, meta_tensor)
        all_predictions.append(pred)
        weights.append(CFG.ensemble_weights.get(model_name, 1.0))
    
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
    
    # Process DICOM series
    volume, metadata = process_dicom_series(series_path)
    
    # Create multi-channel input
    middle_slice = volume[CFG.num_slices // 2]
    mip = np.max(volume, axis=0)
    std_proj = np.std(volume, axis=0).astype(np.float32)
    
    # Normalize std projection
    if std_proj.max() > std_proj.min():
        std_proj = ((std_proj - std_proj.min()) / (std_proj.max() - std_proj.min()) * 255).astype(np.uint8)
    else:
        std_proj = np.zeros_like(std_proj, dtype=np.uint8)
    
    image = np.stack([middle_slice, mip, std_proj], axis=-1)
    
    # Prepare metadata
    age_normalized = metadata['age'] / 100.0
    sex = metadata['sex']
    meta_tensor = torch.tensor([[age_normalized, sex]], dtype=torch.float32).to(device)
    
    # Make predictions
    if CFG.use_ensemble:
        final_pred = predict_ensemble(image, meta_tensor)
    else:
        # Use single selected model
        model = MODELS[CFG.model_selection]
        final_pred = predict_single_model(model, image, meta_tensor)
    
    # Create output dataframe
    predictions_df = pl.DataFrame(
        data=[[series_id] + final_pred.tolist()],
        schema=[ID_COL] + LABEL_COLS,
        orient='row'
    )

    
    # Return without ID column, as required by the API
    return predictions_df.drop(ID_COL)


def predict_fallback(series_path: str) -> pl.DataFrame:
    """Fallback prediction function"""
    series_id = os.path.basename(series_path)
    
    # Return conservative predictions
    predictions = pl.DataFrame(
        data=[[series_id] + [0.1] * len(LABEL_COLS)],
        schema=[ID_COL] + LABEL_COLS,
        orient='row'
    )
    
    # Clean up
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    return predictions.drop(ID_COL)

def predict(series_path: str) -> pl.DataFrame:
    """
    Top-level prediction function passed to the server.
    It calls the core logic and guarantees cleanup in a `finally` block.
    """
    try:
        # Call the internal prediction logic
        return _predict_inner(series_path)
    except Exception as e:
        print(f"Error during prediction for {os.path.basename(series_path)}: {e}")
        print("Using fallback predictions.")
        # Return a fallback dataframe with the correct schema
        predictions = pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
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



## 9. Main Execution

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

