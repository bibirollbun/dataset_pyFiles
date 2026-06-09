import os
import sys
import gc
import json
import shutil
import warnings
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np
import polars as pl
import pandas as pd
import pydicom
import cv2
from scipy import ndimage

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import kaggle_evaluation.rsna_inference_server

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CFG:
    # Competition constants
    ID_COL = 'SeriesInstanceUID'
    LABEL_COLS = [
        'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery',
        'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
        'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
        'Anterior Communicating Artery', 'Left Anterior Cerebral Artery', 'Right Anterior Cerebral Artery',
        'Left Posterior Communicating Artery', 'Right Posterior Communicating Artery',
        'Basilar Tip', 'Other Posterior Circulation', 'Aneurysm Present',
    ]

    # --- Pipeline 1: 3D Voxel Model Config ---
    P1_MODEL_DIR = '/kaggle/input/rsna2025-effnetv2-32ch/'
    P1_MODEL_NAME = 'tf_efficientnetv2_s.in21k_ft_in1k'
    P1_MODEL_FOLDS = [0, 1, 2, 3, 4]
    P1_INPUT_SHAPE = (32, 256, 256)  # D, H, W
    P1_IN_CHANS = 32

    # --- Pipeline 2: 2.5D Projection Model Config ---
    P2_MODEL_DIR = '/kaggle/input/rsna-iad-trained-models/models/'
    P2_INPUT_SIZE = 512
    P2_MODELS = {
        'tf_efficientnetv2_s': 'tf_efficientnetv2_s_fold0_best.pth',
        'convnext_small': 'convnext_small_fold0_best.pth',
        'swin_small_patch4_window7_224': 'swin_small_patch4_window7_224_fold0_best.pth',
    }
    P2_ENSEMBLE_WEIGHTS = {
        'tf_efficientnetv2_s': 0.4,
        'convnext_small': 0.3,
        'swin_small_patch4_window7_224': 0.3,
    }

    # --- Final Ensemble Weights ---
    # Weight for combining the two pipeline predictions
    PIPELINE_3D_WEIGHT = 0.5
    PIPELINE_2D_WEIGHT = 0.5

    # --- Inference Config ---
    USE_TTA = True


def load_and_sort_dicom_series(series_path: str) -> List[pydicom.Dataset]:
    """Loads and sorts DICOM files by slice position."""
    series_path = Path(series_path)
    dicom_files = [os.path.join(r, f) for r, _, fs in os.walk(series_path) for f in fs if f.endswith('.dcm')]
    
    if not dicom_files:
        raise ValueError(f"No DICOM files found in {series_path}")

    datasets = []
    for f in dicom_files:
        try:
            ds = pydicom.dcmread(f, force=True)
            if hasattr(ds, 'pixel_array'):
                datasets.append(ds)
        except Exception:
            continue
    
    if not datasets:
        raise ValueError(f"No valid DICOM files could be read from {series_path}")

    # Sort by slice location or instance number as a fallback
    datasets.sort(key=lambda ds: float(getattr(ds, 'ImagePositionPatient', [0, 0, 0])[2] if hasattr(ds, 'ImagePositionPatient') else getattr(ds, 'InstanceNumber', 0)))
    return datasets

def process_to_volume(datasets: List[pydicom.Dataset]) -> Tuple[np.ndarray, Dict]:
    """Processes sorted DICOMs into a 3D volume and extracts metadata."""
    slices = []
    metadata = {}
    
    for i, ds in enumerate(datasets):
        img = ds.pixel_array.astype(np.float32)
        
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        img = img * slope + intercept

        # Use CTA windowing as a robust default
        center, width = 50, 350
        img_min = center - width / 2
        img_max = center + width / 2
        img = np.clip(img, img_min, img_max)
        
        slices.append(img)

        if i == 0: # Extract metadata from the first slice
            try:
                age_str = getattr(ds, 'PatientAge', '050Y')
                metadata['age'] = int(''.join(filter(str.isdigit, age_str[:3])) or '50')
            except:
                metadata['age'] = 50
            metadata['sex'] = 1 if getattr(ds, 'PatientSex', 'M') == 'M' else 0

    if not slices:
        raise ValueError("Could not extract any pixel arrays from the DICOM series.")
        
    return np.stack(slices, axis=0), metadata

def unified_preprocessor(series_path: str) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Main preprocessing function.
    Returns:
        - volume_3d (np.ndarray): (D, H, W) for the 3D pipeline.
        - proj_image_2d (np.ndarray): (H, W, 3) for the 2.5D pipeline.
        - metadata (Dict): Patient age and sex.
    """
    datasets = load_and_sort_dicom_series(series_path)
    full_volume, metadata = process_to_volume(datasets)
    
    # --- 1. Create 3D Voxel Volume ---
    target_d, target_h, target_w = CFG.P1_INPUT_SHAPE
    zoom_factors = [
        target_d / full_volume.shape[0],
        target_h / full_volume.shape[1],
        target_w / full_volume.shape[2]
    ]
    volume_3d = ndimage.zoom(full_volume, zoom_factors, order=1, mode='nearest')
    
    # Normalize to [0, 255] uint8
    vol_min, vol_max = volume_3d.min(), volume_3d.max()
    if vol_max > vol_min:
        volume_3d = ((volume_3d - vol_min) / (vol_max - vol_min) * 255).astype(np.uint8)
    else:
        volume_3d = np.zeros_like(volume_3d, dtype=np.uint8)
    
    # --- 2. Create 2.5D Projection Image ---
    size = CFG.P2_INPUT_SIZE
    
    # Projections are created from the original full_volume for max quality
    middle_slice = cv2.resize(full_volume[full_volume.shape[0] // 2], (size, size))
    mip = cv2.resize(np.max(full_volume, axis=0), (size, size))
    std_proj = cv2.resize(np.std(full_volume, axis=0), (size, size))

    # Normalize each channel to [0, 255] uint8
    def normalize_channel(ch):
        ch_min, ch_max = ch.min(), ch.max()
        if ch_max > ch_min:
            return ((ch - ch_min) / (ch_max - ch_min) * 255).astype(np.uint8)
        return np.zeros_like(ch, dtype=np.uint8)

    proj_image_2d = np.stack([
        normalize_channel(middle_slice),
        normalize_channel(mip),
        normalize_channel(std_proj)
    ], axis=-1)

    return volume_3d, proj_image_2d, metadata


# --- Pipeline 1: 3D Model ---
class Timm3DModel(nn.Module):
    def __init__(self, model_name, pretrained=False):
        super().__init__()
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            in_chans=CFG.P1_IN_CHANS,
            num_classes=len(CFG.LABEL_COLS)
        )
    def forward(self, x):
        return self.model(x)

# --- Pipeline 2: 2.5D Model ---
class MultiBackboneModel(nn.Module):
    def __init__(self, model_name, num_classes=len(CFG.LABEL_COLS), pretrained=False):
        super().__init__()
        
        # Conditionally add img_size only for Swin models
        create_kwargs = {
            'pretrained': pretrained,
            'num_classes': 0,
            'in_chans': 3,
            'global_pool': '',
        }
        if 'swin' in model_name:
            create_kwargs['img_size'] = CFG.P2_INPUT_SIZE
            
        self.backbone = timm.create_model(model_name, **create_kwargs)
        
        with torch.no_grad():
            dummy_features = self.backbone(torch.randn(1, 3, CFG.P2_INPUT_SIZE, CFG.P2_INPUT_SIZE))
            if len(dummy_features.shape) == 4: # Conv features
                num_features = dummy_features.shape[1]
                self.pool = nn.AdaptiveAvgPool2d(1)
            else: # Transformer features
                num_features = dummy_features.shape[-1]
                self.pool = lambda x: x.mean(dim=1)

        # This meta_fc matches the saved model, including a Dropout layer
        self.meta_fc = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 32),
            nn.ReLU()
        )

        # This complex classifier matches the saved model
        self.classifier = nn.Sequential(
            nn.Linear(num_features + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, image, meta):
        img_features = self.backbone(image)
        img_features = self.pool(img_features).flatten(1)
        meta_features = self.meta_fc(meta)
        combined = torch.cat([img_features, meta_features], dim=1)
        return self.classifier(combined)

# --- Transforms ---
def get_tta_transforms():
    # Define safe augmentations. Avoid HorizontalFlip due to anatomical left/right.
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

TRANSFORM_3D = A.Compose([A.Normalize(mean=0.5, std=0.5), ToTensorV2()])
TRANSFORM_2D_INFERENCE = A.Compose([A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()])
TRANSFORM_2D_TTA = get_tta_transforms()


# Global variables to hold the loaded models
MODELS_3D = []
MODELS_2D = {}

def load_all_models():
    """Loads all models for both pipelines into the global variables."""
    # Pipeline 1: 3D Models
    for fold in CFG.P1_MODEL_FOLDS:
        model_path = os.path.join(CFG.P1_MODEL_DIR, f"{CFG.P1_MODEL_NAME}_fold{fold}_best.pth")
        model = Timm3DModel(CFG.P1_MODEL_NAME)
        # Load the state_dict from the checkpoint file
        sd = torch.load(model_path, map_location=device, weights_only=False)['model']
        # Load the state_dict into the correct submodule (model.model)
        model.model.load_state_dict(sd)
        model.to(device).eval()
        MODELS_3D.append(model)
    print(f"Loaded {len(MODELS_3D)} 3D models.")

    # Pipeline 2: 2.5D Models
    for name, path in CFG.P2_MODELS.items():
        model_path = os.path.join(CFG.P2_MODEL_DIR, path)
        model = MultiBackboneModel(name)
        sd = torch.load(model_path, map_location=device, weights_only=False)['model_state_dict']
        model.load_state_dict(sd)
        model.to(device).eval()
        MODELS_2D[name] = model
    print(f"Loaded {len(MODELS_2D)} 2.5D models.")

def predict_3d_pipeline(volume_3d: np.ndarray) -> np.ndarray:
    """Runs the 3D pipeline and returns averaged predictions."""
    with torch.no_grad(), autocast():
        # Transpose (D, H, W) -> (H, W, D) for Albumentations
        image_tensor = TRANSFORM_3D(image=volume_3d.transpose(1, 2, 0))['image']
        image_tensor = image_tensor.unsqueeze(0).to(device) # (1, D, H, W)
        
        all_preds = []
        for model in MODELS_3D:
            output = model(image_tensor)
            all_preds.append(torch.sigmoid(output).cpu().numpy())
            
    return np.mean(all_preds, axis=0).squeeze()

def predict_2d_pipeline(proj_image_2d: np.ndarray, metadata: Dict) -> np.ndarray:
    """Runs the 2.5D pipeline and returns weighted-average predictions."""
    with torch.no_grad(), autocast():
        meta_tensor = torch.tensor([[metadata['age'] / 100.0, metadata['sex']]], dtype=torch.float32).to(device)
        
        all_model_preds = []
        for name, model in MODELS_2D.items():
            
            # TTA for this model
            tta_preds = []
            if CFG.USE_TTA:
                for _ in range(2): # TTA iterations
                    image_tensor = TRANSFORM_2D_TTA(image=proj_image_2d)['image'].unsqueeze(0).to(device)
                    output = model(image_tensor, meta_tensor)
                    tta_preds.append(torch.sigmoid(output).cpu().numpy())
            else:
                 image_tensor = TRANSFORM_2D_INFERENCE(image=proj_image_2d)['image'].unsqueeze(0).to(device)
                 output = model(image_tensor, meta_tensor)
                 tta_preds.append(torch.sigmoid(output).cpu().numpy())
            
            # Average TTA predictions
            model_pred = np.mean(tta_preds, axis=0)
            all_model_preds.append(model_pred)

    # Weighted average of model predictions
    weights = np.array([CFG.P2_ENSEMBLE_WEIGHTS[name] for name in MODELS_2D.keys()])
    predictions = np.array(all_model_preds).squeeze(axis=1)
    
    return np.average(predictions, weights=weights, axis=0)


def predict(series_path: str) -> pl.DataFrame:
    """Top-level prediction function for the Kaggle server."""
    try:
        # 1. Unified preprocessing
        volume_3d, proj_image_2d, metadata = unified_preprocessor(series_path)

        # 2. Run Pipeline 1 (3D)
        preds_3d = predict_3d_pipeline(volume_3d)

        # 3. Run Pipeline 2 (2.5D)
        preds_2d = predict_2d_pipeline(proj_image_2d, metadata)

        # 4. Final weighted ensemble of both pipelines
        final_preds = (CFG.PIPELINE_3D_WEIGHT * preds_3d) + (CFG.PIPELINE_2D_WEIGHT * preds_2d)

        # 5. Post-processing: ensure 'Aneurysm Present' is at least the max of others
        max_location_prob = np.max(final_preds[:-1])
        final_preds[-1] = np.max([final_preds[-1], max_location_prob])
        
        # Create output dataframe in the required format
        return pl.DataFrame([final_preds.tolist()], schema=CFG.LABEL_COLS)

    except Exception as e:
        print(f"Error processing {os.path.basename(series_path)}: {e}. Returning fallback.")
        return pl.DataFrame([[0.1] * len(CFG.LABEL_COLS)], schema=CFG.LABEL_COLS)
    finally:
        # Crucial memory and disk space cleanup for Kaggle environment
        shared_dir = '/kaggle/shared'
        shutil.rmtree(shared_dir, ignore_errors=True)
        os.makedirs(shared_dir, exist_ok=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

# Load all models at startup before the server begins processing
load_all_models()

# Initialize the inference server with our main `predict` function
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

# This part automatically handles whether to run for a local test or the final submission
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    # Display submission dataframe for local testing and verification
    submission_df = pl.read_parquet('/kaggle/working/submission.parquet')
    display(submission_df)




