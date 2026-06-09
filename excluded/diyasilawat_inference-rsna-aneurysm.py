import os
import shutil
from collections import defaultdict
from pathlib import Path
import gc

import numpy as np
import pandas as pd
import polars as pl
import pydicom
import cv2
from scipy import ndimage
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import kaggle_evaluation.rsna_inference_server

import warnings
warnings.filterwarnings('ignore')


# ====================================================
# CONFIGURATION
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

class Config:
    model_dir = '/kaggle/input/rsna-aneurysm-5fold-models/models'
    cache_dir = '/kaggle/input/rsna-aneurysm-cache/cache'
    
    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    size = 384
    in_chans = 32
    num_classes = 14
    target_shape = (32, 384, 384)
    folds = [0, 1, 2, 3, 4]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_cache = True

CFG = Config()


# ====================================================
# DICOM PREPROCESSING
# ====================================================
class DICOMPreprocessorKaggle:
    def __init__(self, target_shape: Tuple[int, int, int] = (32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape
        
    def load_dicom_series(self, series_path: str) -> List[pydicom.Dataset]:
        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found")
        
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
    
    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
        slice_info = []
        
        for i, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': i,
                'instance_number': getattr(ds, 'InstanceNumber', i),
            }
            
            try:
                position = getattr(ds, 'ImagePositionPatient', None)
                if position is not None and len(position) >= 3:
                    info['z_position'] = float(position[2])
                elif hasattr(ds, "SliceLocation"):
                    info['z_position'] = float(getattr(ds, "SliceLocation"))
                else:
                    info['z_position'] = float(info['instance_number'])
            except:
                info['z_position'] = float(i)
            
            slice_info.append(info)
        
        return slice_info
    
    def sort_slices_by_position(self, slice_info: List[Dict]) -> List[Dict]:
        return sorted(slice_info, key=lambda x: x['z_position'])
    
    def apply_windowing_or_normalize(self, img: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(img, [1, 99])
        
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
    
    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        img = ds.pixel_array.astype(np.float32)
        
        if img.ndim == 3:
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]
        
        if img.ndim == 3 and img.shape[-1] == 3:
            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
        
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        img = img * slope + intercept
        
        return img
    
    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        
        if current_shape == target_shape:
            return volume
        
        zoom_factors = [target_shape[i] / current_shape[i] for i in range(3)]
        resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]
        
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]
        
        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')
        
        return resized_volume.astype(np.uint8)
    
    def process_series(self, series_path: str) -> np.ndarray:
        datasets = self.load_dicom_series(series_path)
        first_ds = datasets[0]
        first_img = first_ds.pixel_array
        
        if len(datasets) == 1 and first_img.ndim == 3:
            return self._process_single_3d_dicom(first_ds)
        else:
            return self._process_multiple_2d_dicoms(datasets)
    
    def _process_single_3d_dicom(self, ds: pydicom.Dataset) -> np.ndarray:
        volume = ds.pixel_array.astype(np.float32)
        
        slope = float(getattr(ds, 'RescaleSlope', 1))
        intercept = float(getattr(ds, 'RescaleIntercept', 0))
        volume = volume * slope + intercept
        
        processed_slices = []
        for i in range(volume.shape[0]):
            processed_img = self.apply_windowing_or_normalize(volume[i])
            processed_slices.append(processed_img)
        
        volume = np.stack(processed_slices, axis=0)
        return self.resize_volume_3d(volume)
    
    def _process_multiple_2d_dicoms(self, datasets: List[pydicom.Dataset]) -> np.ndarray:
        slice_info = self.extract_slice_info(datasets)
        sorted_slices = self.sort_slices_by_position(slice_info)
        
        processed_slices = []
        for slice_data in sorted_slices:
            ds = slice_data['dataset']
            img = self.extract_pixel_array(ds)
            processed_img = self.apply_windowing_or_normalize(img)
            resized_img = cv2.resize(processed_img, (self.target_width, self.target_height))
            processed_slices.append(resized_img)

        volume = np.stack(processed_slices, axis=0)
        return self.resize_volume_3d(volume)

def process_dicom_series_safe(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)) -> np.ndarray:
    try:
        preprocessor = DICOMPreprocessorKaggle(target_shape=target_shape)
        return preprocessor.process_series(series_path)
    finally:
        gc.collect()


# ====================================================
# LOAD MODELS 
# ====================================================
def load_model(fold):
    model = timm.create_model(
        CFG.model_name,
        pretrained=False,
        num_classes=CFG.num_classes,
        in_chans=CFG.in_chans
    )
    
    model_path = f'{CFG.model_dir}/{CFG.model_name}_fold{fold}_best.pth'
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)  # ← FIXED
    model.load_state_dict(checkpoint['model'])
    model.to(CFG.device)
    model.eval()
    
    return model

print('Loading models...')
models = [load_model(fold) for fold in CFG.folds]
print(f'Loaded {len(models)} models')

# Transform
transform = A.Compose([
    A.Resize(CFG.size, CFG.size),
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2(),
])

# ====================================================
# PREDICT FUNCTION (CALLED PER SERIES)
# ====================================================
def predict(series_path: str) -> pl.DataFrame | pd.DataFrame:
    """Make a prediction for one series."""
    
    series_id = os.path.basename(series_path)
    
    try:
        # Check cache first
        cache_path = Path(CFG.cache_dir) / f"{series_id}.npy"
        
        if CFG.use_cache and cache_path.exists():
            volume = np.load(cache_path)
        else:
            volume = process_dicom_series_safe(series_path, CFG.target_shape)
        
        # Preprocess
        volume = volume.transpose(1, 2, 0)
        image = transform(image=volume)['image']
        image = image.unsqueeze(0).to(CFG.device)
        
        # Ensemble prediction
        with torch.no_grad(), autocast():
            all_preds = []
            for model in models:
                output = model(image)
                pred = torch.sigmoid(output).cpu().numpy()[0]
                all_preds.append(pred)
        
        final_pred = np.mean(all_preds, axis=0)
        
        # Create result dataframe
        predictions = pl.DataFrame(
            data=[[series_id] + final_pred.tolist()],
            schema=[ID_COL, *LABEL_COLS],
            orient='row',
        )
        
    except Exception as e:
        print(f'Error processing {series_id}: {e}')
        # Return default predictions
        predictions = pl.DataFrame(
            data=[[series_id] + [0.5] * len(LABEL_COLS)],
            schema=[ID_COL, *LABEL_COLS],
            orient='row',
        )
    
    finally:
        torch.cuda.empty_cache()
        gc.collect()
    
    # Verify format
    if isinstance(predictions, pl.DataFrame):
        assert predictions.columns == [ID_COL, *LABEL_COLS]
    elif isinstance(predictions, pd.DataFrame):
        assert (predictions.columns == [ID_COL, *LABEL_COLS]).all()
    else:
        raise TypeError('The predict function must return a DataFrame')

    # IMPORTANT: Prevent disk space errors
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    
    # Return WITHOUT the ID column (as required)
    return predictions.drop(ID_COL)

# ====================================================
# START INFERENCE SERVER
# ====================================================
inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))

