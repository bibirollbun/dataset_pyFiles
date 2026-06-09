


# ==========================================
# PHASE 1: ENSEMBLE 3-FOLD, 4-EPOCH TRAINING SCRIPT
# ==========================================
# This is the full, non-debug script.
# It trains 3 models across 3 folds (9 models total)
# with all bug fixes (PatientID, normalization, etc.)
# ==========================================

# -------------------------
# 1. GLOBAL IMPORTS
# -------------------------
import os
import glob
import random
import warnings
import numpy as np
import pandas as pd
import cv2
import functools
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict
import gc
import shutil

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
import pydicom

warnings.filterwarnings('ignore')
print(f"Imports successful. Numpy version: {np.__version__}")

# -------------------------
# 2. GLOBAL CONFIGURATION
# -------------------------
class Config:
    # --- Set DEBUG = False for the full run ---
    DEBUG = False
    
    # --- Path to your PNG dataset ---
    DATA_DIR = "/kaggle/input/another1" 
    CVT_PNG_DIR = os.path.join(DATA_DIR, "cvt_png")
    SERIES_MAPPING_PATH = os.path.join(DATA_DIR, "series_index_mapping.csv")
    LOCALIZERS_PATH = os.path.join(DATA_DIR, "train_localizers_with_relative.csv")
    
    # --- Path to original competition data ---
    ORIGINAL_DATA_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection"
    TRAIN_CSV_PATH = os.path.join(ORIGINAL_DATA_DIR, "train.csv")
    ORIGINAL_SERIES_DIR = os.path.join(ORIGINAL_DATA_DIR, "series")
    
    # --- Model Hyperparameters ---
    NUM_FRAMES = 8
    IMAGE_SIZE = 224
    NUM_CLASSES = 14
    BATCH_SIZE = 6 # Use 6 for full run
    NUM_EPOCHS = 4 # <-- 4 EPOCHS for ~10-11 hour run
    LEARNING_RATE = 5e-5
    
    # --- Train all 3 models ---
    MODELS_TO_TRAIN = [
        ("effnetv2s", "tf_efficientnetv2_s.in1k"),
        ("convnext_tiny", "convnext_tiny.fb_in1k"),
        ("maxvit_tiny", "maxvit_tiny_tf_224.in1k"),
    ]
    
    # --- Feature Flags ---
    USE_METADATA = True
    USE_WINDOWING = True
    USE_3CHANNEL_INPUT = True
    USE_IMPROVED_LOSS = True
    USE_CLAHE = True
    USE_STRONG_AUGMENTATION = True
    
    # --- Dataloader & System ---
    NUM_WORKERS = 2
    PIN_MEMORY = True
    PREFETCH_FACTOR = 2
    PERSISTENT_WORKERS = True
    
    # --- CV & Training Loop ---
    NUM_FOLDS = 3 # <-- 3 FOLDS for balance
    ACCUMULATION_STEPS = 5
    EARLY_STOPPING_PATIENCE = 5 # Patience of 5 on a 4-epoch run means it will likely run all 4
    USE_GROUP_CV = True
    CACHE_SIZE = 100 # Full cache size
    OUTPUT_DIR = "/kaggle/working"

config = Config()

# -------------------------
# 3. GLOBAL SEED & DEVICE
# -------------------------
def set_seed(seed: int = 42, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
set_seed(42, deterministic=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if not torch.cuda.is_available():
    print("WARNING: CUDA not available. Training will be VERY slow.")

# -------------------------
# 4. GLOBAL TARGETS
# -------------------------
TARGET_COLS = [
    'Left Infraclinoid Internal Carotid Artery', 'Right Infraclinoid Internal Carotid Artery', 
    'Left Supraclinoid Internal Carotid Artery', 'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery', 'Right Middle Cerebral Artery',
    'Anterior Communicating Artery', 'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery', 'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery', 'Basilar Tip',
    'Other Posterior Circulation', 'Aneurysm Present'
]

# -------------------------
# 5. GLOBAL HELPER FUNCTIONS
# -------------------------
def get_windowing_params(modality: str) -> Tuple[float, float]:
    windows = {'CT': (40, 80), 'CTA': (50, 350), 'MRA': (600, 1200), 'MRI': (40, 80), 'MR': (40, 80)}
    return windows.get(modality, (40, 80))

def apply_dicom_windowing(img: np.ndarray, window_center: float, window_width: float) -> np.ndarray:
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min + 1e-7)
    return (img * 255).astype(np.uint8)

def apply_clahe_normalization(img: np.ndarray, modality: str) -> np.ndarray:
    if not config.USE_CLAHE: return img.astype(np.uint8)
    img = img.astype(np.uint8)
    if modality in ['CTA', 'MRA']:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
        img_clahe = cv2.convertScaleAbs(img_clahe, alpha=1.1, beta=5)
    elif modality in ['MRI', 'MR']:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
        img_clahe = np.power(img_clahe / 255.0, 0.9) * 255
        img_clahe = img_clahe.astype(np.uint8)
    else:
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        img_clahe = clahe.apply(img)
    return img_clahe

def robust_normalization(volume: np.ndarray) -> np.ndarray:
    if volume.size == 0:
        return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
    p1, p99 = np.percentile(volume.flatten(), [1, 99])
    volume_norm = np.clip(volume, p1, p99)
    if p99 > p1:
        volume_norm = (volume_norm - p1) / (p99 - p1 + 1e-7)
    else:
        volume_norm = np.zeros_like(volume_norm)
    return (volume_norm * 255).astype(np.uint8)

def create_3channel_input_8frame(volume: np.ndarray) -> np.ndarray:
    if len(volume) == 0:
        return np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
    middle_slice = volume[len(volume) // 2]
    mip = np.max(volume, axis=0)
    std_proj = np.std(volume, axis=0).astype(np.float32)
    if std_proj.max() > std_proj.min():
        p1, p99 = np.percentile(std_proj, [5, 95])
        std_proj = np.clip(std_proj, p1, p99)
        std_proj = ((std_proj - p1) / (p99 - p1 + 1e-7) * 255).astype(np.uint8)
    else:
        std_proj = np.zeros_like(std_proj, dtype=np.uint8)
    return np.stack([middle_slice, mip, std_proj], axis=-1)

def smart_8_frame_sampling(volume_paths: List[str]) -> List[str]:
    n = len(volume_paths)
    if n == 0: return []
    if n <= 8:
        result = volume_paths[:]
        while len(result) < 8:
            result.extend(volume_paths[:8-len(result)])
        return result[:8]
    start_idx = max(0, int(n * 0.1))
    available_frames = n - start_idx
    step = max(1, available_frames // 8)
    indices = [start_idx + i * step for i in range(8)]
    indices = [min(i, n - 1) for i in indices]
    if len(set(indices)) < 8:
        indices = np.linspace(start_idx, n-1, 8).astype(int).tolist()
    return [volume_paths[i] for i in indices]

def resolve_dicom_path(dicom_entry, series_uid):
    if dicom_entry and os.path.exists(dicom_entry):
        return dicom_entry
    series_dir = os.path.join(config.ORIGINAL_SERIES_DIR, series_uid)
    if dicom_entry:
        possible = os.path.join(series_dir, os.path.basename(dicom_entry))
        if os.path.exists(possible):
            return possible
    if os.path.exists(series_dir):
        candidates = sorted(glob.glob(os.path.join(series_dir, "*.dcm")))
        if candidates:
            if dicom_entry:
                sop_uid = os.path.splitext(os.path.basename(dicom_entry))[0]
                for cand_path in candidates:
                    if sop_uid in cand_path:
                        return cand_path
            return candidates[0] 
    raise FileNotFoundError(f"No DICOM file for series {series_uid}. Entry='{dicom_entry}'")

# -------------------------
# 6. LOAD DATA & EDA
# -------------------------
print("Loading main CSVs...")
try:
    if not os.path.exists(config.TRAIN_CSV_PATH):
        raise FileNotFoundError(f"Original train.csv not found at {config.TRAIN_CSV_PATH}")
    if not os.path.exists(config.SERIES_MAPPING_PATH):
        raise FileNotFoundError(f"series_index_mapping.csv not found at {config.SERIES_MAPPING_PATH}")
            
    train_df = pd.read_csv(config.TRAIN_CSV_PATH)
    series_mapping_df = pd.read_csv(config.SERIES_MAPPING_PATH)
    
    print(f"train.csv shape: {train_df.shape}")
    print(f"series_index_mapping.csv shape: {series_mapping_df.shape}")

    # --- MERGE PatientID INTO train_df ---
    if 'PatientID' in series_mapping_df.columns:
        patient_id_map = series_mapping_df[['SeriesInstanceUID', 'PatientID']].drop_duplicates()
        train_df = train_df.merge(patient_id_map, on='SeriesInstanceUID', how='left')
        print(f"train_df shape after merging PatientID: {train_df.shape}")
        
        if train_df['PatientID'].isnull().any():
            print("Warning: Some series in train.csv were not found in series_mapping_df.")
            train_df['PatientID'] = train_df['PatientID'].fillna(train_df['SeriesInstanceUID'])
    else:
        print("ERROR: 'PatientID' column not found in series_index_mapping.csv!")
    
    # --- DEBUG SUBSETTING ---
    if config.DEBUG:
        print(f"--- DEBUG MODE: Subsetting data to 100 samples ---")
        train_df = train_df.sample(100, random_state=42).reset_index(drop=True)
        print(f"New train_df shape: {train_df.shape}")
    
    # --- Statistics ---
    print("\n--- Label Distribution ---")
    label_counts = train_df[TARGET_COLS].sum().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=label_counts.values, y=label_counts.index, palette="viridis")
    plt.title("Distribution of Aneurysm Locations (Positive Cases)")
    plt.xlabel("Number of Positive Cases")
    plt.ylabel("Aneurysm Location")
    plt.show()

    # --- Image Visualization ---
    print("\n--- Visualizing Sample 2.5D Model Inputs (from DICOM fallback) ---")
    
    def load_volume_for_viz(series_uid: str, row) -> np.ndarray:
        series_data = series_mapping_df[
            series_mapping_df['SeriesInstanceUID'] == series_uid
        ].sort_values('relative_index')
        
        if series_data.empty: 
            return np.zeros((8, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
            
        modality = row.get('Modality', 'CT')
        all_indices = list(range(len(series_data)))
        sampled_indices_str = smart_8_frame_sampling([str(i) for i in all_indices])
        sampled_indices = [int(i) for i in sampled_indices_str]
        sampled_data = series_data.iloc[sampled_indices]
            
        volume = []
        for _, dicom_row in sampled_data.iterrows():
            dicom_entry = dicom_row.get('dicom_filename', None)
            try:
                dicom_path = resolve_dicom_path(dicom_entry, series_uid)
                ds = pydicom.dcmread(dicom_path, force=True)
                img = ds.pixel_array.astype(np.float32)
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                window_center, window_width = get_windowing_params(modality)
                img = apply_dicom_windowing(img, window_center, window_width)
                img = apply_clahe_normalization(img, modality)
                img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                volume.append(img)
            except Exception as e:
                pass 
        
        if not volume:
            return np.zeros((8, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
            
        while len(volume) < 8:
            volume.append(volume[-1]) 
        return np.array(volume)

    positive_cases = train_df[train_df['Aneurysm Present'] == 1].sample(min(2, len(train_df[train_df['Aneurysm Present'] == 1])), random_state=42)
    negative_cases = train_df[train_df['Aneurysm Present'] == 0].sample(min(2, len(train_df[train_df['Aneurysm Present'] == 0])), random_state=42)
    sample_df = pd.concat([positive_cases, negative_cases])

    if not sample_df.empty:
        fig, axes = plt.subplots(len(sample_df), 3, figsize=(12, len(sample_df) * 4))
        if len(sample_df) == 1: 
            axes = np.array([axes])
        fig.suptitle("Model Input (Middle Slice, Max-IP, Std-IP)", fontsize=16, y=1.02)

        for i, (idx, row) in enumerate(sample_df.iterrows()):
            volume = load_volume_for_viz(row['SeriesInstanceUID'], row)
            volume_norm = robust_normalization(volume)
            img_3channel = create_3channel_input_8frame(volume_norm)
            
            status = "POSITIVE" if row['Aneurysm Present'] == 1 else "NEGATIVE"
            
            axes[i, 0].imshow(img_3channel[:, :, 0], cmap='bone')
            axes[i, 0].set_title(f"Series (Aneurysm {status})\nChannel 0: Middle Slice")
            axes[i, 0].axis('off')
            
            axes[i, 1].imshow(img_3channel[:, :, 1], cmap='bone')
            axes[i, 1].set_title(f"Series (Aneurysm {status})\nChannel 1: Max-IP")
            axes[i, 1].axis('off')
            
            axes[i, 2].imshow(img_3channel[:, :, 2], cmap='bone')
            axes[i, 2].set_title(f"Series (Aneurysm {status})\nChannel 2: Std-IP")
            axes[i, 2].axis('off')
        
        plt.tight_layout()
        plt.show()
    else:
        print("No samples to display in EDA (DEBUG subset might be too small).")
    
except FileNotFoundError as e:
    print(f"ERROR: {e}")
    print("Please make sure your PNG dataset ('another1') and the original")
    print("competition data ('rsna-intracranial-aneurysm-detection') are both added.")
    train_df = None

# -------------------------
# 7. MODEL & LOSS DEFINITION
# -------------------------
class ImprovedMultiFrameModel(nn.Module):
    def __init__(self, model_name_backbone: str, num_frames=8, num_classes=14, pretrained=True):
        super(ImprovedMultiFrameModel, self).__init__()
        self.model_name_backbone = model_name_backbone
        self.num_frames = num_frames
        self.num_classes = num_classes
        self.use_metadata = config.USE_METADATA
        print(f"Loading backbone: {self.model_name_backbone}")
        
        self.backbone = timm.create_model(
            self.model_name_backbone,
            pretrained=pretrained,
            num_classes=0,
            global_pool='avg'
        )
        self.feature_dim = self.backbone.num_features
        print(f"Backbone {self.model_name_backbone}: {self.feature_dim} features")
        
        if self.use_metadata:
            self.meta_fc = nn.Sequential(
                nn.Linear(2, 16), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(16, 32), nn.ReLU()
            )
            classifier_input_dim = self.feature_dim + 32
        else:
            classifier_input_dim = self.feature_dim
            
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 512),
            nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, meta=None):
        features = self.backbone(x)
        if self.use_metadata and meta is not None:
            meta_features = self.meta_fc(meta)
            features = torch.cat([features, meta_features], dim=1)
        output = self.classifier(features)
        return output

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()

class ImprovedLoss(nn.Module):
    def __init__(self, aneurysm_weight=3.0, focal_weight=0.3):
        super(ImprovedLoss, self).__init__()
        weights = torch.ones(config.NUM_CLASSES)
        weights[-1] = aneurysm_weight
        self.register_buffer('weights', weights)
        self.focal_loss = FocalLoss(alpha=1, gamma=2)
        self.focal_weight = focal_weight
        
    def forward(self, outputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(outputs, targets, reduction='none')
        weighted_bce = (bce_loss * self.weights.to(outputs.device)).mean()
        focal_loss_val = self.focal_loss(outputs, targets)
        return (1 - self.focal_weight) * weighted_bce + self.focal_weight * focal_loss_val

# -------------------------
# 8. TRAINING: DATASET CLASS
# -------------------------
def create_frame_paths_8frame_structured(train_df, series_mapping_df):
    frame_paths = {}
    print("Creating 8-frame optimized structured paths from PNG dataset...")
    
    all_png_files = {}
    print(f"Scanning PNG directory: {config.CVT_PNG_DIR}")
    if not os.path.exists(config.CVT_PNG_DIR):
        print(f"WARNING: PNG Directory not found at {config.CVT_PNG_DIR}")
        print("The dataset will fall back to reading DICOMs, which will be much slower.")
    
    for root, _, files in os.walk(config.CVT_PNG_DIR):
        for file in files:
            if file.endswith(".png"):
                try:
                    series_uid = os.path.basename(os.path.dirname(root))
                    if series_uid not in all_png_files:
                        all_png_files[series_uid] = []
                    all_png_files[series_uid].append(os.path.join(root, file))
                except:
                    pass
    print(f"Found PNGs for {len(all_png_files)} series.")

    for series_uid in tqdm(train_df['SeriesInstanceUID'].unique(), desc="Processing series"):
        series_data = series_mapping_df[series_mapping_df['SeriesInstanceUID'] == series_uid]
        if series_data.empty:
            frame_paths[series_uid] = {'paths': [], 'is_dummy': True}
            continue

        found_paths = []
        if series_uid in all_png_files:
            png_files = sorted(list(set(all_png_files[series_uid])), key=lambda x: os.path.basename(x))
            if png_files:
                found_paths = png_files
        
        if not found_paths:
            dicom_dir = os.path.join(config.ORIGINAL_SERIES_DIR, series_uid)
            if os.path.exists(dicom_dir):
                num_frames = len(series_data)
                found_paths = [f"dummy_dicom_path_{i:04d}.dcm" for i in range(num_frames)]

        if found_paths:
            sampled = smart_8_frame_sampling(found_paths)
            is_dummy = any(p.startswith('dummy_dicom_path') for p in sampled)
            frame_paths[series_uid] = {'paths': sampled, 'is_dummy': is_dummy}
        else:
            frame_paths[series_uid] = {'paths': [], 'is_dummy': True}
            
    return frame_paths

class EightFrameDataset(Dataset):
    def __init__(self, df, frame_paths_dict, series_mapping_df, num_frames=8, transform=None, is_training=True):
        self.df = df.reset_index(drop=True)
        self.frame_paths_dict = frame_paths_dict
        self.series_mapping_df = series_mapping_df
        self.num_frames = num_frames
        self.transform = transform
        self.is_training = is_training
        self._cache = {}
        self._cache_keys = []
        self._max_cache_size = config.CACHE_SIZE

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        
        row = self.df.iloc[idx]
        series_uid = row['SeriesInstanceUID']
        labels = torch.tensor(row[TARGET_COLS].values.astype(np.float32))
        metadata = self._extract_metadata(row)
        
        try:
            image = self._load_8frame_3channel_image(series_uid, row)
        except Exception as e:
            dummy_image = np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE, 3), dtype=np.uint8)
            if self.transform:
                image = self.transform(image=dummy_image)['image']
            else:
                image = torch.from_numpy(dummy_image).permute(2,0,1).float()

        result = (image, labels, metadata)
        self._update_cache(idx, result)
        return result

    def _update_cache(self, idx, data):
        if len(self._cache) >= self._max_cache_size:
            oldest_idx = self._cache_keys.pop(0)
            if oldest_idx in self._cache:
                del self._cache[oldest_idx]
        self._cache[idx] = data
        self._cache_keys.append(idx)

    def _extract_metadata(self, row) -> torch.Tensor:
        if not config.USE_METADATA:
            return torch.tensor([0.0, 0.0], dtype=torch.float32)
        
        age = row.get('PatientAge', 50)
        if pd.isna(age):
            age = 50
        elif isinstance(age, str):
            age = int(''.join(filter(str.isdigit, age[:3])) or '50')
        age = min(float(age), 100.0) / 100.0
        
        sex = row.get('PatientSex', 'M')
        sex = 1.0 if sex == 'M' else 0.0
        return torch.tensor([age, sex], dtype=torch.float32)

    def _load_8frame_3channel_image(self, series_uid: str, row) -> torch.Tensor:
        entry = self.frame_paths_dict.get(series_uid, {'paths': [], 'is_dummy': True})
        paths = entry['paths']
        is_dummy = entry.get('is_dummy', True)
        
        if len(paths) == 0:
            raise FileNotFoundError(f"No paths found for series {series_uid}")
            
        if is_dummy:
            volume = self._load_volume_from_dicom_8frame(series_uid, row)
        else:
            volume = self._load_volume_from_png_8frame(paths)
            
        volume_norm = robust_normalization(volume)
        image = create_3channel_input_8frame(volume_norm)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        return image

    def _load_volume_from_png_8frame(self, paths: List[str]) -> np.ndarray:
        volume = []
        for path in paths:
            try:
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                    volume.append(img)
                else:
                    volume.append(np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8))
            except Exception:
                volume.append(np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8))
        
        if not volume:
            return np.zeros((8, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
        return np.array(volume)

    def _load_volume_from_dicom_8frame(self, series_uid: str, row) -> np.ndarray:
        series_data = self.series_mapping_df[self.series_mapping_df['SeriesInstanceUID'] == series_uid].sort_values('relative_index')
        if series_data.empty:
            return np.zeros((8, config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8)
            
        modality = row.get('Modality', 'CT')
        
        if len(series_data) <= 8:
            sampled_data = series_data
        else:
            all_indices = list(range(len(series_data)))
            sampled_indices_str = smart_8_frame_sampling([str(i) for i in all_indices])
            sampled_indices = [int(i) for i in sampled_indices_str]
            sampled_data = series_data.iloc[sampled_indices]
            
        volume = []
        for _, dicom_row in sampled_data.iterrows():
            dicom_entry = dicom_row.get('dicom_filename', None)
            try:
                dicom_path = resolve_dicom_path(dicom_entry, series_uid)
                ds = pydicom.dcmread(dicom_path, force=True)
                img = ds.pixel_array.astype(np.float32)
                
                if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
                    img = img * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
                        
                if config.USE_WINDOWING:
                    window_center, window_width = get_windowing_params(modality)
                    img = apply_dicom_windowing(img, window_center, window_width)
                else:
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
                    else:
                        img = np.zeros_like(img, dtype=np.uint8)
                        
                img = apply_clahe_normalization(img, modality)
                img = cv2.resize(img, (config.IMAGE_SIZE, config.IMAGE_SIZE), interpolation=cv2.INTER_AREA)
                volume.append(img)
            except Exception as e:
                volume.append(np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8))
        
        while len(volume) < 8:
            if volume:
                volume.append(volume[-1])
            else:
                volume.append(np.zeros((config.IMAGE_SIZE, config.IMAGE_SIZE), dtype=np.uint8))
                
        return np.array(volume[:8])

# -------------------------
# 9. TRAINING: TRANSFORMS
# -------------------------
def get_transforms(model_backbone: str):
    """
    Returns the correct train and validation transforms for a given model.
    """
    if 'tf_' in model_backbone:
        print(f"Using TF normalization (0.5, 0.5) for: {model_backbone}")
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        print(f"Using PyTorch normalization (0.485...) for: {model_backbone}")
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if config.USE_STRONG_AUGMENTATION:
        train_transform = A.Compose([
            A.Rotate(limit=15, p=0.7),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=10, p=0.6),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.6),
            A.RandomGamma(gamma_limit=(80, 120), p=0.4),
            A.GaussNoise(var_limit=(10, 80), p=0.4),
            A.Blur(blur_limit=3, p=0.2),
            A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        train_transform = A.Compose([
            A.Rotate(limit=10, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
            A.GaussNoise(var_limit=(10, 50), p=0.2),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])

    val_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return train_transform, val_transform

# -------------------------
# 10. TRAINING: LOSS & METRIC
# -------------------------
def get_loss_function():
    if config.USE_IMPROVED_LOSS:
        return ImprovedLoss(aneurysm_weight=3.0, focal_weight=0.3).to(device)
    else:
        weights = torch.ones(config.NUM_CLASSES, device=device)
        weights[-1] = 3.0
        return nn.BCEWithLogitsLoss(pos_weight=weights)

def calculate_competition_metric(y_true, y_pred):
    individual_aucs = []
    for i in range(13):
        try:
            if len(np.unique(y_true[:, i])) > 1:
                auc = roc_auc_score(y_true[:, i], y_pred[:, i])
            else:
                auc = 0.5
            individual_aucs.append(auc)
        except:
            individual_aucs.append(0.5)
            
    try:
        if len(np.unique(y_true[:, 13])) > 1:
            aneurysm_present_auc = roc_auc_score(y_true[:, 13], y_pred[:, 13])
        else:
            aneurysm_present_auc = 0.5
    except:
        aneurysm_present_auc = 0.5
        
    avg_individual = np.mean(individual_aucs)
    final_score = (aneurysm_present_auc + avg_individual) / 2
    return final_score, aneurysm_present_auc, avg_individual

# -------------------------
# 11. TRAINING: HELPERS
# -------------------------
def save_checkpoint(model, model_name_backbone, optimizer, scheduler, epoch, best_score, val_loss, out_dir, model_name_prefix, fold):
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(
        out_dir,
        f"{model_name_prefix}_fold{fold}_epoch{epoch}_score{best_score:.6f}.pth"
    )
    torch.save({
        'epoch': epoch,
        'model_name_backbone': model_name_backbone,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_score': best_score,
        'val_loss': val_loss,
    }, model_path)
    return model_path

# --- FAST CV SPLIT FUNCTION ---
def create_robust_cv_split(train_df_local, n_splits=5):
    print("Creating patient-separated cross-validation split (from DataFrame)...")
    
    if 'PatientID' not in train_df_local.columns:
        print("ERROR: 'PatientID' column not in train_df! Check Section 6 merge.")
        print("Falling back to StratifiedKFold (NOT patient-aware).")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(skf.split(train_df_local, train_df_local['Aneurysm Present']))
        
    groups = train_df_local['PatientID']
    n_groups = train_df_local['PatientID'].nunique()
    print(f"Total unique patients found: {n_groups}")
    
    if n_groups < n_splits:
        print(f"Warning: Only {n_groups} patients. Falling back to StratifiedKFold.")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        return list(skf.split(train_df_local, train_df_local['Aneurysm Present']))
        
    gkf = GroupKFold(n_splits=n_splits)
    splits = list(gkf.split(train_df_local, groups=groups))
    return splits

# -------------------------
# 12. TRAINING: TRAIN/VAL LOOPS
# -------------------------
def train_epoch_optimized(model, train_loader, criterion, optimizer, scaler, device, accumulation_steps):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()
    
    for batch_idx, (images, targets, metadata) in enumerate(tqdm(train_loader, desc="Training")):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        metadata = metadata.to(device, non_blocking=True)
        
        with torch.cuda.amp.autocast():
            outputs = model(images, metadata)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps
            
        scaler.scale(loss).backward()
        
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        running_loss += loss.item() * accumulation_steps
        
    return running_loss / len(train_loader)

def validate_epoch_optimized(model, val_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        for images, targets, metadata in tqdm(val_loader, desc="Validating"):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            metadata = metadata.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast():
                logits = model(images, metadata)
                loss = criterion(logits, targets)
                
            outputs = torch.sigmoid(logits)
            running_loss += loss.item()
            all_outputs.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
            
    all_outputs = np.concatenate(all_outputs)
    all_targets = np.concatenate(all_targets)
    
    final_score, aneurysm_auc, avg_individual = calculate_competition_metric(all_targets, all_outputs)
    return running_loss / len(val_loader), final_score, aneurysm_auc, avg_individual

# -------------------------
# 13. TRAINING: MAIN EXECUTION
# -------------------------
if train_df is not None:
    try:
        # Build frame paths dict ONCE
        frame_paths_dict = create_frame_paths_8frame_structured(train_df, series_mapping_df)
        
        # Filter train_df to valid series ONCE
        valid_series = [uid for uid, entry in frame_paths_dict.items() if len(entry['paths']) > 0]
        train_df_filtered = train_df[train_df['SeriesInstanceUID'].isin(valid_series)].copy()
        print(f"Filtered train data shape (series with paths): {train_df_filtered.shape}")
        
        # Create CV splits ONCE
        cv_splits = create_robust_cv_split(train_df_filtered, config.NUM_FOLDS)
        
        # --- Outer loop for Models ---
        for model_name_prefix, model_backbone in config.MODELS_TO_TRAIN:
            print(f"\n\n{'='*50}")
            print(f"STARTING {config.NUM_FOLDS}-FOLD TRAINING FOR MODEL: {model_name_prefix} (Backbone: {model_backbone})")
            print(f"{'='*50}")
            
            # --- GET MODEL-SPECIFIC TRANSFORMS ---
            train_transform, val_transform = get_transforms(model_backbone)
            
            all_fold_scores = []
            
            # --- Inner loop for Folds ---
            for fold in range(config.NUM_FOLDS):
                print(f"\n--- Fold {fold}/{config.NUM_FOLDS - 1} ---")
                
                # --- 1. Get Fold Data ---
                train_indices, val_indices = cv_splits[fold]
                train_fold_df = train_df_filtered.iloc[train_indices]
                val_fold_df = train_df_filtered.iloc[val_indices]
                print(f"Train size: {len(train_fold_df)}, Val size: {len(val_fold_df)}")

                # --- 2. Create Datasets & Loaders (with correct transforms) ---
                train_dataset = EightFrameDataset(train_fold_df, frame_paths_dict, series_mapping_df, 
                                                  num_frames=config.NUM_FRAMES, transform=train_transform, is_training=True)
                val_dataset = EightFrameDataset(val_fold_df, frame_paths_dict, series_mapping_df, 
                                                num_frames=config.NUM_FRAMES, transform=val_transform, is_training=False)
                
                train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, 
                                          num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, 
                                          drop_last=True, prefetch_factor=config.PREFETCH_FACTOR, 
                                          persistent_workers=config.PERSISTENT_WORKERS)
                val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, 
                                        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY, 
                                        prefetch_factor=config.PREFETCH_FACTOR, 
                                        persistent_workers=config.PERSISTENT_WORKERS)
                print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

                # --- 3. Initialize Model, Optimizer, etc. (Fresh for each fold) ---
                model = ImprovedMultiFrameModel(model_name_backbone=model_backbone, 
                                                num_frames=config.NUM_FRAMES, 
                                                num_classes=config.NUM_CLASSES, 
                                                pretrained=True)
                model = model.to(device)
                
                criterion = get_loss_function()
                optimizer = AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)
                scheduler = CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS, eta_min=1e-6)
                scaler = torch.cuda.amp.GradScaler()

                # --- 4. Run Training Loop for this fold ---
                best_score = 0.0
                best_epoch = 0
                patience_counter = 0
                
                for epoch in range(config.NUM_EPOCHS):
                    print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS}")
                    
                    train_loss = train_epoch_optimized(model, train_loader, criterion, optimizer, scaler, device, config.ACCUMULATION_STEPS)
                    val_loss, val_score, aneurysm_auc, avg_individual = validate_epoch_optimized(model, val_loader, criterion, device)
                    scheduler.step()
                    
                    print(f"Model {model_name_prefix} Fold {fold} Epoch {epoch+1} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}, Val Score: {val_score:.6f}")
                    print(f"Aneurysm AUC: {aneurysm_auc:.4f}, Avg Location AUC: {avg_individual:.4f}")
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_epoch = epoch + 1
                        patience_counter = 0
                        ckpt_path = save_checkpoint(model, model_backbone, optimizer, scheduler, epoch+1, best_score, val_loss, 
                                                    config.OUTPUT_DIR, model_name_prefix, fold)
                        print(f"Saved checkpoint (Best Score): {os.path.basename(ckpt_path)}")
                    else:
                        patience_counter += 1
                        print(f"No improvement. Patience: {patience_counter}/{config.EARLY_STOPPING_PATIENCE}")
                        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
                            print(f"Early stopping at epoch {epoch+1}")
                            break
                    
                    torch.cuda.empty_cache()
                    gc.collect()

                print(f"\nModel {model_name_prefix} Fold {fold} finished. Best Score: {best_score:.6f} at epoch {best_epoch}")
                all_fold_scores.append(best_score)
                
                # Clean up memory before next fold
                del model, train_dataset, val_dataset, train_loader, val_loader, optimizer, scheduler, scaler
                gc.collect()
                torch.cuda.empty_cache()
                
            print(f"\n--- 3-Fold Training Finished for {model_name_prefix} ---")
            print(f"Fold Scores: {all_fold_scores}")
            print(f"Average CV Score: {np.mean(all_fold_scores):.6f}")

        print("\n\n===== ALL 9 MODELS (3 Models x 3 Folds) TRAINING COMPLETE =====")
        print("Your models are saved in the '/kaggle/working' directory.")
        print("Please download them, upload as a new dataset, and use the inference notebook.")

    except Exception as e:
        print(f"An unexpected error occurred during the main loop: {e}")
        import traceback
        traceback.print_exc()

else:
    print("ERROR: 'train_df' was not loaded in Section 6. Halting script.")

print("\nScript finished.")













