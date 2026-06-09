import os
import numpy as np
import pandas as pd
import pydicom
import cv2
from pathlib import Path
from scipy import ndimage
import warnings
import gc
from typing import List, Dict, Tuple, Optional
from tqdm.auto import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')

# ====================================================
# DICOM PREPROCESSING
# ====================================================
class DICOMPreprocessorKaggle:
    def __init__(self, target_shape: Tuple[int, int, int] = (32, 384, 384)):
        self.target_depth, self.target_height, self.target_width = target_shape
        
    def load_dicom_series(self, series_path: str) -> Tuple[List[pydicom.Dataset], str]:
        series_path = Path(series_path)
        series_name = series_path.name
        
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
            raise ValueError(f"No valid DICOM files in {series_path}")
        
        return datasets, series_name
    
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
    
    def get_windowing_params(self, ds: pydicom.Dataset) -> Tuple[Optional[float], Optional[float]]:
        modality = getattr(ds, 'Modality', 'CT')
        if modality == 'CT':
            return 50, 350
        return None, None
    
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
        datasets, series_name = self.load_dicom_series(series_path)
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
# CONFIGURATION - FOLDS 2 & 3
# ====================================================
class Config:
    data_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection'
    series_dir = f'{data_dir}/series'
    train_csv = f'{data_dir}/train.csv'
    output_dir = './outputs'
    
    # CHANGE THIS to your saved dataset path
    cache_dir = '/kaggle/input/rsna-aneurysm-cache/cache'
    
    model_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    size = 384
    in_chans = 32
    
    target_cols = [
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
    num_classes = len(target_cols)
    target_shape = (32, 384, 384)
    
    n_fold = 5
    trn_fold = [2, 3]  # TRAINING FOLDS 2 & 3
    epochs = 15
    batch_size = 10
    num_workers = 4
    
    lr = 3e-4
    weight_decay = 1e-5
    max_grad_norm = 1000
    use_amp = True
    
    early_stopping_patience = 15
    use_cache = True
    
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

CFG = Config()

os.makedirs(CFG.output_dir, exist_ok=True)

# ====================================================
# SEED
# ====================================================
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

set_seed(CFG.seed)

# ====================================================
# TRANSFORMS
# ====================================================
def get_train_transform():
    return A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, p=0.3),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])

def get_valid_transform():
    return A.Compose([
        A.Resize(CFG.size, CFG.size),
        A.Normalize(),
        ToTensorV2(),
    ])

# ====================================================
# DATASET
# ====================================================
class AneurysmDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_id = row['SeriesInstanceUID']
        
        try:
            cache_path = Path(CFG.cache_dir) / f"{series_id}.npy"
            
            if CFG.use_cache and cache_path.exists():
                volume = np.load(cache_path)
            else:
                series_path = Path(CFG.series_dir) / series_id
                volume = process_dicom_series_safe(str(series_path), CFG.target_shape)
                if CFG.use_cache:
                    np.save(cache_path, volume)
            
            volume = volume.transpose(1, 2, 0)
            
            if self.transform:
                volume = self.transform(image=volume)['image']
            
            labels = row[CFG.target_cols].values.astype(np.float32)
            
            return {
                'image': volume,
                'labels': torch.tensor(labels, dtype=torch.float32),
            }
            
        except Exception as e:
            return {
                'image': torch.zeros(CFG.in_chans, CFG.size, CFG.size),
                'labels': torch.zeros(CFG.num_classes, dtype=torch.float32),
            }

# ====================================================
# MODEL
# ====================================================
def build_model():
    model = timm.create_model(
        CFG.model_name,
        pretrained=True,
        num_classes=CFG.num_classes,
        in_chans=CFG.in_chans
    )
    return model

# ====================================================
# METRICS
# ====================================================
def calculate_auc(preds, labels):
    try:
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
        
        aucs = []
        for i in range(CFG.num_classes):
            if len(np.unique(labels[:, i])) > 1:
                aucs.append(roc_auc_score(labels[:, i], preds[:, i]))
        
        return np.mean(aucs) if aucs else 0.0
    except:
        return 0.0

# ====================================================
# TRAINING
# ====================================================
def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    losses = []
    all_preds = []
    all_labels = []
    
    for batch in tqdm(loader, desc='Train'):
        images = batch['image'].to(CFG.device)
        labels = batch['labels'].to(CFG.device)
        
        with autocast(enabled=CFG.use_amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        losses.append(loss.item())
        all_preds.append(torch.sigmoid(outputs).detach())
        all_labels.append(labels.detach())
    
    auc = calculate_auc(torch.cat(all_preds), torch.cat(all_labels))
    return np.mean(losses), auc

def valid_epoch(model, loader, criterion):
    model.eval()
    losses = []
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Valid'):
            images = batch['image'].to(CFG.device)
            labels = batch['labels'].to(CFG.device)
            
            with autocast(enabled=CFG.use_amp):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            losses.append(loss.item())
            all_preds.append(torch.sigmoid(outputs))
            all_labels.append(labels)
    
    auc = calculate_auc(torch.cat(all_preds), torch.cat(all_labels))
    return np.mean(losses), auc

# ====================================================
# FOLD TRAINING
# ====================================================
def train_fold(fold, train_df, valid_df):
    print(f'\nFold {fold}: Train={len(train_df)}, Valid={len(valid_df)}')
    
    train_ds = AneurysmDataset(train_df, transform=get_train_transform())
    valid_ds = AneurysmDataset(valid_df, transform=get_valid_transform())
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=CFG.batch_size, 
        shuffle=True,
        num_workers=CFG.num_workers, 
        pin_memory=True, 
        drop_last=True
    )
    valid_loader = DataLoader(
        valid_ds, 
        batch_size=CFG.batch_size, 
        shuffle=False,
        num_workers=CFG.num_workers, 
        pin_memory=True
    )
    
    model = build_model().to(CFG.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs, eta_min=1e-6)
    scaler = GradScaler(enabled=CFG.use_amp)
    
    best_auc = 0.0
    patience = 0
    
    for epoch in range(1, CFG.epochs + 1):
        print(f'\nEpoch {epoch}/{CFG.epochs}')
        
        train_loss, train_auc = train_epoch(model, train_loader, criterion, optimizer, scaler)
        valid_loss, valid_auc = valid_epoch(model, valid_loader, criterion)
        scheduler.step()
        
        print(f'Train: Loss={train_loss:.4f}, AUC={train_auc:.4f}')
        print(f'Valid: Loss={valid_loss:.4f}, AUC={valid_auc:.4f}')
        
        if valid_auc > best_auc:
            best_auc = valid_auc
            patience = 0
            
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'auc': best_auc
            }, f'{CFG.output_dir}/{CFG.model_name}_fold{fold}_best.pth')
            
            print(f'Saved: AUC={best_auc:.4f}')
        else:
            patience += 1
        
        if patience >= CFG.early_stopping_patience:
            print('Early stopping')
            break
        
        torch.cuda.empty_cache()
        gc.collect()
    
    print(f'Fold {fold} Best: {best_auc:.4f}')
    
    del model, optimizer, scheduler
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_auc

# ====================================================
# MAIN
# ====================================================
def main():
    df = pd.read_csv(CFG.train_csv)
    print(f'Dataset: {len(df)} samples')
    
    skf = StratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    df['fold'] = -1
    
    for fold, (_, val_idx) in enumerate(skf.split(df, df['Aneurysm Present'])):
        df.loc[val_idx, 'fold'] = fold
    
    scores = []
    for fold in CFG.trn_fold:
        train_df = df[df['fold'] != fold]
        valid_df = df[df['fold'] == fold]
        
        score = train_fold(fold, train_df, valid_df)
        scores.append(score)
    
    print(f'\nResults:')
    for i, fold in enumerate(CFG.trn_fold):
        print(f'Fold {fold}: {scores[i]:.4f}')
    print(f'Mean: {np.mean(scores):.4f}')

if __name__ == '__main__':
    main()

