from IPython.display import Image, display

img_path = "/kaggle/input/rsna-logo/header.png"

display(Image(filename=img_path))


from IPython.display import Image, display

img_path = "/kaggle/input/rsna-logo/intracranial aneursyms.png"

display(Image(filename=img_path))


from IPython.display import Image, display

img_path = "/kaggle/input/rsna-logo/imaging modalities.png"

display(Image(filename=img_path))


from IPython.display import Image, display

img_path = "/kaggle/input/rsna-logo/multitask model architecture.png"

display(Image(filename=img_path))


# ============================================================================
# COMPATIBLE ENHANCED RSNA 2025 INTRACRANIAL ANEURYSM DETECTION PIPELINE
# Maintains compatibility with existing trained models while adding improvements
# ============================================================================

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

# ============= Data handling =========================
import numpy as np
import polars as pl
import pandas as pd

# =============== Medical imaging ======================
import pydicom
import cv2
from scipy import ndimage
from sklearn.preprocessing import RobustScaler

# ================ ML/DL ======================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm

# ================== Transformations ===============
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ============== Competition API =======================
import kaggle_evaluation.rsna_inference_server

# ================= Set device =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ======================== Competition constants =================================
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

# =================  Model selection - Change this to select which model to use for inference ===================
SELECTED_MODEL = 'ensemble' 

# =====================  Model paths configuration  ===============================
MODEL_PATHS = {
    'tf_efficientnetv2_s': '/kaggle/input/rsna-iad-trained-models/models/tf_efficientnetv2_s_fold0_best.pth',
    'convnext_small': '/kaggle/input/rsna-iad-trained-models/models/convnext_small_fold0_best.pth',
    'swin_small_patch4_window7_224': '/kaggle/input/rsna-iad-trained-models/models/swin_small_patch4_window7_224_fold0_best.pth'
}

class InferenceConfig:
    # =========== Model selection ============
    model_selection = SELECTED_MODEL
    use_ensemble = (SELECTED_MODEL == 'ensemble')
    
    # ================= Default model settings (will be overridden by checkpoint) ======================
    image_size = 512
    num_slices = 32
    use_windowing = True
    
    # ==================== Inference settings
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

# ============================================================================
# ENHANCED DICOM PROCESSING FUNCTIONS (keeping original metadata format)
# ============================================================================

def adaptive_windowing(img: np.ndarray, modality: str, percentile_range: tuple = (5, 95)) -> np.ndarray:
    """Apply adaptive windowing based on actual image statistics"""
    
    # Get image statistics
    img_flat = img.flatten()
    img_flat = img_flat[img_flat > 0]  # Remove zero padding
    
    if len(img_flat) == 0:
        return np.zeros_like(img, dtype=np.uint8)
    
    # Calculate adaptive window based on percentiles
    low_val = np.percentile(img_flat, percentile_range[0])
    high_val = np.percentile(img_flat, percentile_range[1])
    
    # Modality-specific adjustments
    if modality in ['CTA']:
        # Expand range for better vessel contrast
        window_width = (high_val - low_val) * 1.5
        window_center = (high_val + low_val) / 2
    elif modality in ['MRA']:
        # Tighter window for MR angiography
        window_width = (high_val - low_val) * 1.2
        window_center = high_val * 0.7  # Bias toward high intensities
    else:
        # Standard windowing
        window_width = high_val - low_val
        window_center = (high_val + low_val) / 2
    
    # Apply windowing
    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    img_windowed = np.clip(img, img_min, img_max)
    
    # Normalize to 0-255
    if img_max > img_min:
        img_normalized = ((img_windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img_normalized = np.zeros_like(img, dtype=np.uint8)
    
    return img_normalized

def create_multi_projection_image(volume: np.ndarray) -> np.ndarray:
    """Create rich multi-channel representation from 3D volume"""
    
    # Get volume dimensions
    depth, height, width = volume.shape
    
    # ========== Channel 1: Adaptive MIP (Maximum Intensity Projection) ==========
    # Use only middle 70% of slices to avoid edge artifacts
    start_idx = int(depth * 0.15)
    end_idx = int(depth * 0.85)
    core_volume = volume[start_idx:end_idx]
    mip = np.max(core_volume, axis=0)
    
    # ========== Channel 2: Weighted average of high-intensity slices ==========
    # Focus on slices with high contrast (likely vessels)
    slice_means = np.mean(volume, axis=(1,2))
    top_percentile = np.percentile(slice_means, 75)
    high_intensity_mask = slice_means >= top_percentile
    
    if np.any(high_intensity_mask):
        weighted_avg = np.mean(volume[high_intensity_mask], axis=0)
    else:
        weighted_avg = np.mean(volume, axis=0)
    
    # ========== Channel 3: Standard deviation projection (texture information) ==========
    # Use sliding window to capture local variations
    std_proj = np.zeros_like(volume[0])
    window_size = min(5, depth // 4)
    
    for i in range(depth - window_size + 1):
        window_std = np.std(volume[i:i+window_size], axis=0)
        std_proj = np.maximum(std_proj, window_std)
    
    # Normalize all channels to 0-255
    channels = []
    for channel in [mip, weighted_avg, std_proj]:
        if channel.max() > channel.min():
            channel_norm = ((channel - channel.min()) / 
                          (channel.max() - channel.min()) * 255).astype(np.uint8)
        else:
            channel_norm = np.zeros_like(channel, dtype=np.uint8)
        channels.append(channel_norm)
    
    return np.stack(channels, axis=-1)

def process_dicom_series_enhanced(series_path: str) -> Tuple[np.ndarray, Dict]:
    """Enhanced DICOM processing with better preprocessing"""
    series_path = Path(series_path)
    
    # Find all DICOM files
    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    
    if len(all_filepaths) == 0:
        volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
        metadata = {'age': 50, 'sex': 0, 'modality': 'CT'}
        return volume, metadata
    
    # Sort files by instance number for proper ordering
    dicom_data = []
    metadata = {}
    
    for i, filepath in enumerate(all_filepaths):
        try:
            ds = pydicom.dcmread(filepath, force=True)
            img = ds.pixel_array.astype(np.float32)
            
            # Handle multi-dimensional images
            if img.ndim == 3:
                if img.shape[-1] == 3:
                    img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
                else:
                    img = img[:, :, 0]
            
            # Get instance number for proper sorting
            instance_num = getattr(ds, 'InstanceNumber', i)
            
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
            
            # Apply rescaling
            if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                img = img * ds.RescaleSlope + ds.RescaleIntercept
            
            dicom_data.append((instance_num, img))
            
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
            continue
    
    if len(dicom_data) == 0:
        volume = np.zeros((CFG.num_slices, CFG.image_size, CFG.image_size), dtype=np.uint8)
        return volume, metadata
    
    # Sort by instance number
    dicom_data.sort(key=lambda x: x[0])
    raw_slices = [data[1] for data in dicom_data]
    
    # Apply adaptive windowing to entire volume first
    volume_3d = np.stack(raw_slices, axis=0)
    volume_windowed = adaptive_windowing(volume_3d, metadata['modality'])
    
    # Resize slices
    processed_slices = []
    for slice_img in volume_windowed:
        resized = cv2.resize(slice_img, (CFG.image_size, CFG.image_size))
        processed_slices.append(resized)
    
    # Handle slice count standardization
    volume = np.array(processed_slices)
    if len(processed_slices) > CFG.num_slices:
        indices = np.linspace(0, len(processed_slices) - 1, CFG.num_slices).astype(int)
        volume = volume[indices]
    elif len(processed_slices) < CFG.num_slices:
        pad_size = CFG.num_slices - len(processed_slices)
        volume = np.pad(volume, ((0, pad_size), (0, 0), (0, 0)), mode='edge')
    
    return volume, metadata

# ============================================================================
# COMPATIBLE MODEL ARCHITECTURE - keeps original metadata network
# ============================================================================

class MultiBackboneModel(nn.Module):
    """Flexible model that can use different backbones - COMPATIBLE VERSION"""
    def __init__(self, model_name, num_classes=14, pretrained=True, 
                 drop_rate=0.3, drop_path_rate=0.2):
        super().__init__()
        
        self.model_name = model_name
        
        # Handle Swin Transformers separately
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
        
        # Dynamic feature dimension detection
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
        
        # KEEP ORIGINAL metadata network - 2 features (age, sex) to maintain compatibility
        self.meta_fc = nn.Sequential(
            nn.Linear(2, 16),  # ORIGINAL: age_norm, sex only
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU()
        )
        
        # Final classification head
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
        # Extract features from backbone
        img_features = self.backbone(image)
        
        # Apply pooling
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
        
        # Concatenate features
        combined = torch.cat([img_features, meta_features], dim=1)
        
        # Final classification
        output = self.classifier(combined)
        
        return output

# ============================================================================
# TRANSFORMS
# ============================================================================

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

# ============================================================================
# MODEL LOADING
# ============================================================================

MODELS = {}
TRANSFORM = None
TTA_TRANSFORMS = None

def load_single_model(model_name: str, model_path: str) -> nn.Module:
    """Load a single model"""
    print(f"Loading {model_name} from {model_path}...")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_config = checkpoint.get('model_config', {})
    training_config = checkpoint.get('training_config', {})
    
    if 'image_size' in training_config:
        CFG.image_size = training_config['image_size']
    
    model = MultiBackboneModel(
        model_name=model_name,
        num_classes=training_config.get('num_classes', 14),
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=0.0
    )
    
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
        for model_name, model_path in MODEL_PATHS.items():
            try:
                MODELS[model_name] = load_single_model(model_name, model_path)
            except Exception as e:
                print(f"Warning: Could not load {model_name}: {e}")
    else:
        if CFG.model_selection in MODEL_PATHS:
            model_path = MODEL_PATHS[CFG.model_selection]
            MODELS[CFG.model_selection] = load_single_model(CFG.model_selection, model_path)
        else:
            raise ValueError(f"Unknown model: {CFG.model_selection}")
    
    TRANSFORM = get_inference_transform()
    if CFG.use_tta:
        TTA_TRANSFORMS = get_tta_transforms()
    
    print(f"Models loaded: {list(MODELS.keys())}")
    
    # Warm up models
    print("Warming up models...")
    dummy_image = torch.randn(1, 3, CFG.image_size, CFG.image_size).to(device)
    dummy_meta = torch.randn(1, 2).to(device)  # BACK TO 2 features for compatibility
    
    with torch.no_grad():
        for model in MODELS.values():
            _ = model(dummy_image, dummy_meta)
    
    print("Ready for inference!")

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_single_model(model: nn.Module, image: np.ndarray, meta_tensor: torch.Tensor) -> np.ndarray:
    """Make prediction with a single model"""
    predictions = []
    
    if CFG.use_tta and TTA_TRANSFORMS:
        for transform in TTA_TRANSFORMS[:CFG.tta_transforms]:
            aug_image = transform(image=image)['image']
            aug_image = aug_image.unsqueeze(0).to(device)
            
            with torch.no_grad():
                with autocast(enabled=CFG.use_amp):
                    output = model(aug_image, meta_tensor)
                    pred = torch.sigmoid(output)
                    predictions.append(pred.cpu().numpy())
        
        return np.mean(predictions, axis=0).squeeze()
    else:
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
    
    weights = np.array(weights) / np.sum(weights)
    predictions = np.array(all_predictions)
    
    return np.average(predictions, weights=weights, axis=0)

def predict_inner_enhanced(series_path: str) -> pl.DataFrame:
    """Enhanced prediction with improved preprocessing and post-processing"""
    global MODELS
    
    if not MODELS:
        load_models()
    
    series_id = os.path.basename(series_path)
    
    # Use enhanced processing
    volume, metadata = process_dicom_series_enhanced(series_path)
    
    # Create enhanced multi-channel image
    image = create_multi_projection_image(volume)
    
    # KEEP ORIGINAL metadata format - only age and sex (2 features for compatibility)
    age_normalized = metadata['age'] / 100.0
    sex = metadata['sex']
    meta_tensor = torch.tensor([[age_normalized, sex]], dtype=torch.float32).to(device)
    
    # Generate predictions
    if CFG.use_ensemble:
        final_pred = predict_ensemble(image, meta_tensor)
    else:
        model = MODELS[CFG.model_selection]
        final_pred = predict_single_model(model, image, meta_tensor)
    
    # Apply post-processing calibration
    location_preds = final_pred[:-1]  # All except "Aneurysm Present"
    aneurysm_present_raw = final_pred[-1]
    
    # If any location has high confidence, boost overall prediction
    max_location_pred = np.max(location_preds)
    if max_location_pred > 0.7:
        aneurysm_present_calibrated = min(0.95, aneurysm_present_raw + 0.1)
    elif max_location_pred > 0.5:
        aneurysm_present_calibrated = min(0.90, aneurysm_present_raw + 0.05)
    else:
        aneurysm_present_calibrated = aneurysm_present_raw
    
    # Combine calibrated predictions
    final_pred = np.concatenate([location_preds, [aneurysm_present_calibrated]])
    
    predictions_df = pl.DataFrame(
        data=[[series_id] + final_pred.tolist()],
        schema=[ID_COL] + LABEL_COLS,
        orient='row'
    )
    
    return predictions_df.drop(ID_COL)

def predict_fallback(series_path: str) -> pl.DataFrame:
    """Fallback prediction function"""
    series_id = os.path.basename(series_path)
    
    predictions = pl.DataFrame(
        data=[[series_id] + [0.1] * len(LABEL_COLS)],
        schema=[ID_COL] + LABEL_COLS,
        orient='row'
    )
    
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    return predictions.drop(ID_COL)

def predict(series_path: str) -> pl.DataFrame:
    """
    Top-level prediction function passed to the server.
    """
    try:
        return predict_inner_enhanced(series_path)
    except Exception as e:
        print(f"Error during prediction for {os.path.basename(series_path)}: {e}")
        print("Using fallback predictions.")
        predictions = pl.DataFrame(
            data=[[0.1] * len(LABEL_COLS)],
            schema=LABEL_COLS,
            orient='row'
        )
        return predictions
    finally:
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# ============================================================================
# MAIN EXECUTION
# ============================================================================

# Pre-load all models and transforms
load_models()

# Initialize the Kaggle evaluation server
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

# Environment-aware execution
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    
    # Load and display submission
    submission_df = pl.read_parquet('/kaggle/working/submission.parquet')
    display(submission_df)


# ========== Display RSNA logo at completion ==========
from IPython.display import Image, display

try:
    display(Image('/kaggle/input/rsna-logo/ending.png'))
except FileNotFoundError:
    print("RSNA logo not found at /kaggle/input/rsna-logo/ending.png")
except Exception as e:
    print(f"Error displaying RSNA logo: {e}")

