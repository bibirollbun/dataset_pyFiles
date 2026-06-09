import sys
import os
import multiprocessing
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import shutil
import gc
import inspect
# Data
import json
import polars as pl
import pandas as pd

# Maths
import numpy as np
# Image
import pydicom
import cv2
from scipy import ndimage
from scipy.ndimage import zoom
# ML/DL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
# Train
from torch.utils.data import DataLoader
import torch.optim as optim
import timm
from tqdm import tqdm


# Transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings('ignore')


# Seeding for reproducibility
def set_global_seed(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    
# Set device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("DEVICE:", DEVICE)

# ====================================================
# Competition constants
# ====================================================
SERIES_ROOT_TRAIN = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"
TRAIN_CSV         = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
LOCALIZER_CSV     = "/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv"

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
TARGET_COL = 'Aneurysm Present'

DEBUG = False
TRAIN = False

TARGET_SIZE = (32, 384, 384) 

MODEL_DIR = '/kaggle/input/efficientnetv2-s'#''/kaggle/input/rsna2025-effnetv2-32ch'
MODEL_NAME = "tf_efficientnetv2_s.in21k_ft_in1k"

set_global_seed(42)



class DICOMPreprocessorKaggle:
    """
    DICOM preprocessing system for Kaggle Code Competition
    Converts original DICOMPreprocessor logic to single series processing
    """
    
    def __init__(self, target_shape: Tuple[int, int, int] = TARGET_SIZE):
        self.target_depth, self.target_height, self.target_width = target_shape

    def _process_slice(self, img): #TODO
        img_target_shape = (self.target_height, self.target_width)
        if img_target_shape[0] > np.shape(img)[0] or img_target_shape[1] > np.shape(img)[1]:
            interpolation = cv2.INTER_CUBIC  # better for upscaling
        else:
            interpolation = cv2.INTER_AREA   # better for downscaling
        image = cv2.resize(img, img_target_shape, interpolation=interpolation)
        return np.array(image).astype(np.float32)
    
    def extract_slice_info(self, datasets: List[pydicom.Dataset]) -> List[Dict]:
        """
        Extract position information for each slice
        """
        slice_info = []
        
        for i, ds in enumerate(datasets):
            info = {
                'dataset': ds,
                'index': i,
                'instance_number': getattr(ds, 'InstanceNumber', i),
            }
            
            # Get z-coordinate from ImagePositionPatient
            try:
                position = getattr(ds, 'ImagePositionPatient', None)
                if position is not None and len(position) >= 3:
                    info['z_position'] = float(position[2])
                else:
                    # Fallback: use InstanceNumber
                    info['z_position'] = float(info['instance_number'])
            except Exception as e:
                info['z_position'] = float(i)
            
            slice_info.append(info)
        
        return slice_info
    
    def sort_slices_by_position(self, slice_info: List[Dict]) -> List[Dict]:
        """
        Sort slices by z-coordinate
        """
        # Sort by z-coordinate
        sorted_slices = sorted(slice_info, key=lambda x: x['z_position'])
        
        if DEBUG:
            print(f"Sorted {len(sorted_slices)} slices by z-position")
            print(f"Z-range: {sorted_slices[0]['z_position']:.2f} to {sorted_slices[-1]['z_position']:.2f}")
        
        return sorted_slices

    
    def process_series(self, series_path: str) -> np.ndarray:
        """
        Process DICOM series and return as NumPy array (for Kaggle: no file saving)
        """
        volume_target_shape = (self.target_depth, self.target_height, self.target_width)
        dicom_files =  sorted(os.listdir(series_path))
        
        if len(dicom_files) == 1:
            dcm = pydicom.dcmread(os.path.join(series_path, dicom_files[0]))
            volume = dcm.pixel_array  # (D, H, W)
            processed_slices = [self._process_slice(slice_) for slice_ in volume]
        else:
            slices = [pydicom.dcmread(os.path.join(series_path, f)).pixel_array
                    for f in dicom_files]
            slices_info = self.extract_slice_info(slices)
            sorted_slices = self.sort_slices_by_position(slices_info)
            processed_slices = [self._process_slice(slice_['dataset']) for slice_ in sorted_slices]
        volume = np.array(processed_slices, dtype=np.float32)
        volume = volume.astype(np.float32)
        volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-8)
        factors = [t / s for s, t in zip(volume.shape, volume_target_shape)]
        if DEBUG:
            print(series_path,len(dicom_files), volume.shape, flush = True)
        resized_vol = zoom(volume, zoom=factors, order=1)
        
        return resized_vol
    

def process_dicom_series_kaggle(series_path: str, target_shape: Tuple[int, int, int] = TARGET_SIZE) -> np.ndarray:
    """
    DICOM processing function for Kaggle inference (single series)
    
    Args:
        series_path: Path to DICOM series
        target_shape: Target volume size (depth, height, width)
    
    Returns:
        np.ndarray: Processed volume
    """
    preprocessor = DICOMPreprocessorKaggle(target_shape=target_shape)
    return preprocessor.process_series(series_path)

# Safe processing function with memory cleanup
def process_dicom_series_safe(series_path: str, target_shape: Tuple[int, int, int] = TARGET_SIZE) -> np.ndarray:
    """
    Safe DICOM processing with memory cleanup
    
    Args:
        series_path: Path to DICOM series
        target_shape: Target volume size (depth, height, width)
    
    Returns:
        np.ndarray: Processed volume
    """
    try:
        volume = process_dicom_series_kaggle(series_path, target_shape)
        return volume
    finally:
        # Memory cleanup
        gc.collect()

# Test function
def test_single_series(series_path: str, target_shape: Tuple[int, int, int] = TARGET_SIZE):
    """
    Test processing for single series
    """
    if DEBUG:
        print(f"Testing single series: {series_path}")
    
    # Execute processing
    volume = process_dicom_series_safe(series_path, target_shape)
    
    # Display results
    if DEBUG:
        print(f"  Successfully processed series")
        print(f"  Volume shape: {volume.shape}")
        print(f"  Volume dtype: {volume.dtype}")
        print(f"  Volume range: [{volume.min()}, {volume.max()}]")
    
    return volume


import torch
from torch.utils.data import Dataset
import numpy as np

TRANSFORM = A.Compose([
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(p=0.2),
    A.GaussianBlur(p=0.1),
    A.Normalize(mean=0.0, std=1.0),
    ToTensorV2()
])


class MyDataset(Dataset):
    def __init__(self, df, label_cols, target_shape, transform=TRANSFORM, series_path = SERIES_ROOT_TRAIN):
        """
        df : DataFrame avec au moins:
            - 'series_path' : chemin dossier DICOM
            - colonnes de labels (multi + binaire)
        label_cols : liste dans l'ordre des colonnes de labels (multi + dernier=binaire)
        target_shape : (depth, height, width)
        transform : Ã©ventuelle transformation Torch/Albumentations
        """
        self.df = df.reset_index(drop=True).replace(['', None, 'nan', 'NaN'], 0)
        self.label_cols = label_cols
        self.target_shape = target_shape
        self.transform = transform
        self.preprocessor = DICOMPreprocessorKaggle(target_shape=target_shape)
        self.series_path = series_path
        self.extract_labels_and_paths()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # --- PrÃ©traitement du volume DICOM ---
        vol = self.preprocessor.process_series(row['series_path'])  # NumPy [D,H,W]
        # Conversion en tenseur float32
        vol_tensor = torch.from_numpy(vol).float()  # [D,H,W] 
        if self.transform:
            # Si ton transform attend HWC -> permuter
            vol_tensor = vol_tensor.permute(1, 2, 0)  # [H,W,C]
            vol_tensor = self.transform(image=vol_tensor.numpy())["image"]
        
        # --- Extraction des labels ---
        vals = row[self.label_cols].to_numpy(dtype=np.float32, na_value=0.0)
        labels_all = torch.tensor(vals, dtype=torch.float32)
        label_binary = labels_all[-1]   # dernier = anÃ©vrisme 0/1
        label_multi = torch.argmax(labels_all[:-1]).long()  # index classe fine

        return vol_tensor, label_multi, label_binary
        
    def extract_labels_and_paths(self):
        """
        Extrait labels_class, labels_binary et mes_paths
        Ã  partir d'un DataFrame fusionnÃ©.
    
        Args:
            dicom_dir (str): Chemin racine contenant les fichiers DICOM.
    
        Returns:
            tuple: (mes_paths, labels_class, labels_binary)
        """
        # --- 1) labels_class ---
        labels_class = []
        for _, row in self.df.iterrows():
            sub_labels = row[self.label_cols[:-1]]
            if sub_labels.max() == 1:
                idx = sub_labels[sub_labels == 1].index[0]
                class_index = self.label_cols[:-1].index(idx)
            else:
                class_index = -1  # ou autre valeur sentinelle
            labels_class.append(class_index)
    
        # --- 2) labels_binary ---
        labels_binary = self.df['Aneurysm Present'].astype(int).tolist()
        
        # --- 3) mes_paths ---
        mes_paths = [
            os.path.join(self.series_path, str(uid))
            for uid in self.df["SeriesInstanceUID"]
        ]
        self.df["series_path"] = mes_paths
        return mes_paths, labels_class, labels_binary



# Competition API
import kaggle_evaluation.rsna_inference_server

class CFG:
    seed = 42
    debug = False
    train = False

    # DICOM
    target_shape = TARGET_SIZE
    threshold = 0.2

    # ModÃ¨le
    backbone_name = "tf_efficientnetv2_s.in21k_ft_in1k"
    in_chans = 32
    num_classes = 13

    model_params = {
        "backbone_name": backbone_name,
        "in_chans": in_chans,
        "num_classes": num_classes,
        "pretrained": False
    }

    backbone_params = {
        "drop_rate": 0.2,
        "drop_path_rate": 0.2,
        "global_pool": "avg",
        "act_layer": "silu",
        "output_stride": 32
    }

    # EntraÃ®nement
    batch_size = 15
    num_workers = multiprocessing.cpu_count()
    lr = 1e-4
    weight_decay = 1e-5
    alpha = 1.0
    epochs = 3
    n_folds = 1
    ensemble_weights = {0: 1.0}
    use_amp = True

    # Chemins
    model_dir = MODEL_DIR
    data_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection"
    train_csv = f"{data_dir}/train.csv"
    series_root = f"{data_dir}/series"

    # Labels
    label_cols = LABEL_COLS


# ====================================================
# ModÃ¨le hiÃ©rarchique
# ====================================================
class EfficientNetV2Hierarchical(nn.Module):
    def __init__(self, backbone_name, in_chans, num_classes, pretrained=False, **backbone_kwargs):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,
            in_chans=in_chans,
            **backbone_kwargs
        )
        in_features = self.backbone.num_features
        self.head_binary = nn.Linear(in_features, 1)
        self.head_multiclass = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        out_bin = self.head_binary(feats)
        out_multi = torch.softmax(self.head_multiclass(feats), dim=1)
        return out_bin, out_multi

# ====================================================
# Chargement modÃ¨le
# ====================================================
def load_hierarchical_model(model_path, backbone_name, in_chans, num_classes, device=DEVICE):
    model = EfficientNetV2Hierarchical(
        backbone_name=backbone_name,
        in_chans=in_chans,
        num_classes=num_classes,
        pretrained=False
    )
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    return model

# ====================================================
# InfÃ©rence hiÃ©rarchique
# ====================================================
def hierarchical_inference(model, images, threshold=0.5):
    """
    images: tensor [B, C, H, W]
    Retourne: score binaire, prediction multiclasses ou None
    """
    with torch.no_grad():
        out_bin, out_multi = model(images)
        prob_bin = torch.sigmoid(out_bin)                # probas en [0, 1]
        has_aneurysm = (prob_bin >= threshold).squeeze(1)
        
        preds_multi = []
        for i in range(images.size(0)):
            if has_aneurysm[i]:
                preds_multi.append(out_multi[i])
            else:
                preds_multi.append(None)
                
    return out_bin.cpu(), preds_multi


def train_model(model, dataloader, optimizer, device, epochs=10, alpha=1.0):
    scaler = torch.cuda.amp.GradScaler()  # prÃ©cision mixte
    bce_loss = nn.BCELoss()
    ce_loss = nn.CrossEntropyLoss()

    model.to(device)
    model.train()
    
    for epoch in range(epochs):
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)
        
        for images, y_multi, y_bin in loop:
            images = images.to(device, non_blocking=True)
            y_multi = y_multi.to(device, non_blocking=True)
            y_bin = y_bin.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            
            with torch.cuda.amp.autocast():  # prÃ©cision mixte
                out_bin, out_multi = model(images)
                loss_bin = bce_loss(out_bin.squeeze(1), y_bin)
                loss_multi = ce_loss(out_multi, y_multi)
                loss = loss_bin + alpha * loss_multi

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1} â€” Loss: {epoch_loss:.4f}")
        
        # LibÃ©ration mÃ©moire intermÃ©diaire
        torch.cuda.empty_cache()

def train_model_hierarchical(model, dataloader, optimizer, device, epochs=10, alpha=1.0):
    scaler = torch.cuda.amp.GradScaler()
    bce_loss = nn.BCEWithLogitsLoss()
    ce_loss = nn.CrossEntropyLoss()

    model.to(device)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        loop = tqdm(dataloader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=True)

        for images, y_multi, y_bin in loop:
            images = images.to(device, non_blocking=True)
            y_multi = y_multi.to(device, non_blocking=True)
            y_bin = y_bin.float().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast():
                out_bin, out_multi = model(images)
                loss_bin = bce_loss(out_bin.squeeze(1), y_bin)

                # --- SÃ©lectionner uniquement les cas positifs pour la perte multiâ€‘classe ---
                pos_mask = (y_bin == 1)
                if pos_mask.any():
                    loss_multi = ce_loss(out_multi[pos_mask], y_multi[pos_mask])
                else:
                    loss_multi = torch.tensor(0.0, device=device)

                loss = loss_bin + alpha * loss_multi

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(dataloader.dataset)
        print(f"Epoch {epoch+1} â€” Loss: {epoch_loss:.4f}")
        if False:
            # Sauvegarde intermÃ©diaire
            cfg_dict = {k: getattr(CFG, k) for k in dir(CFG) if not k.startswith("__") and not callable(getattr(CFG, k))}
            torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'cfg': cfg_dict
            }, f"checkpoint_epoch_{epoch}.pth")
            torch.cuda.empty_cache()



if TRAIN:
    # HypothÃ¨se : tes donnÃ©es prÃ©â€‘chargÃ©es
    # images: Tensor [N, 32, 384, 384]
    # labels: Tensor oneâ€‘hot [N, 14] (13 classes fines + 1 binaire)
    # Ici juste du fake pour lâ€™exemple
    N = 64
    
    train_df = pd.read_csv(TRAIN_CSV)
    # cta_df = train_df[train_df['Modality'] == "CTA"].sample(n=10, random_state=42)
    dataset = MyDataset(train_df, label_cols=LABEL_COLS, target_shape=TARGET_SIZE, series_path = SERIES_ROOT_TRAIN)
    dataloader = DataLoader(
        dataset,
        batch_size=15,          # petit batch pour la mÃ©moire
        shuffle=True,
        num_workers=os.cpu_count(),         # ajuster selon le CPU
        pin_memory=True        # accÃ©lÃ¨re le transfert vers GPU
    )
    
    model = EfficientNetV2Hierarchical(
        backbone_name="tf_efficientnetv2_s.in21k_ft_in1k",
        in_chans=32,
        num_classes=13,
        pretrained=False
    )
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    
    train_model_hierarchical(
        model,
        dataloader,
        optimizer,
        device=DEVICE,
        epochs=3,
        alpha=1.0    # pondÃ©ration de la perte multiâ€‘classe
    )
    
    # Sauvegarde
    torch.save({
    'model': model.state_dict(),
    'cfg': CFG.model_params,
    'epoch': 3
    }, "hierarchical_model.pth")
    print("ModÃ¨le entraÃ®nÃ© et sauvegardÃ© avec succÃ¨s.")


def load_models():
    global MODELS
    MODELS = {}

    valid_params = inspect.signature(EfficientNetV2Hierarchical.__init__).parameters
    allowed_keys = set(valid_params.keys()) - {'self'}
    filtered_params = {k: v for k, v in CFG.model_params.items() if k in allowed_keys}

    for fold in range(CFG.n_folds):
        model_path = f"{CFG.model_dir}/hierarchical_model.pth"
        model = EfficientNetV2Hierarchical(**filtered_params, **CFG.backbone_params).to(DEVICE)
        state_dict = torch.load(model_path, map_location=DEVICE)
        model.load_state_dict(state_dict['model'], strict=False)
        model.eval()
        MODELS[fold] = model

    print("ModÃ¨le chargÃ© avec succÃ¨s.")




# ====================================================
# DICOM Processing
# ====================================================
def process_dicom_series_safe(series_path: str, target_shape: Tuple[int, int, int] = CFG.target_shape) -> np.ndarray:
    try:
        preprocessor = DICOMPreprocessorKaggle(target_shape=target_shape)
        volume = preprocessor.process_series(series_path)
        return volume
    finally:
        gc.collect()

# ====================================================
# Prediction Functions
# ====================================================
def predict_single_model(model: nn.Module, volume: np.ndarray) -> Dict:
    assert isinstance(volume, np.ndarray), f"Expected np.ndarray, got {type(volume)}"

    image = volume.transpose(1, 2, 0)  # (D,H,W) â†’ (H,W,D)
    transformed = TRANSFORM(image=image)
    image_tensor = transformed['image'].unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        with autocast(enabled=CFG.use_amp):
            out_bin, out_multi = model(image_tensor)
            prob_bin = torch.sigmoid(out_bin).item()
            prob_multi = torch.softmax(out_multi, dim=1).squeeze(0).tolist()

            if prob_bin >= CFG.threshold:
                return {
                    "aneurysm_score": prob_bin,
                    "class_prediction": prob_multi
                }
            else:
                return {
                    "aneurysm_score": prob_bin,
                    "class_prediction": [0.1] * CFG.num_classes
                }

def predict_ensemble(volume: np.ndarray) -> Dict:
    all_preds = []
    weights = []

    for fold, model in MODELS.items():
        pred = predict_single_model(model, volume)
        all_preds.append(pred)
        weights.append(CFG.ensemble_weights.get(fold, 1.0))

    weights = np.array(weights) / np.sum(weights)
    aneurysm_scores = np.array([p["aneurysm_score"] for p in all_preds])
    class_preds = np.array([p["class_prediction"] for p in all_preds])

    final_score = float(np.average(aneurysm_scores, weights=weights))
    final_class = list(np.average(class_preds, weights=weights, axis=0))

    return {
        "aneurysm_score": final_score,
        "class_prediction": final_class
    }

def predict_fallback(series_path: str) -> pl.DataFrame:
    conservative_preds = [0.0] * len(CFG.label_cols)
    predictions_df = pl.DataFrame(
        data=[conservative_preds],
        schema=CFG.label_cols,
        orient='row'
    )
    shutil.rmtree('/kaggle/shared', ignore_errors=True)
    return predictions_df

def _predict_inner(series_path: str) -> pl.DataFrame:
    if not MODELS:
        load_models()

    series_id = os.path.basename(series_path)
    volume = process_dicom_series_safe(series_path, CFG.target_shape)
    prediction = predict_ensemble(volume)

    row = [series_id] + prediction["class_prediction"] + [prediction["aneurysm_score"]]
    schema = [ID_COL] + CFG.label_cols
    predictions_df = pl.DataFrame(data=[row], schema=schema, orient='row')

    return predictions_df.drop(ID_COL)

def predict(series_path: str) -> pl.DataFrame:
    try:
        predictions = _predict_inner(series_path)
    except Exception as e:
        print(f"âš ï¸� Erreur sur {series_path}: {e}")
        predictions = predict_fallback(series_path)

    shared_dir = '/kaggle/shared'
    shutil.rmtree(shared_dir, ignore_errors=True)
    os.makedirs(shared_dir, exist_ok=True)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return predictions

def predict_batch(images: torch.Tensor, model: nn.Module) -> List[Dict]:
    images = images.to(DEVICE)
    with torch.no_grad():
        out_bin, out_multi = model(images)
        prob_bin = torch.sigmoid(out_bin).squeeze(1)
        has_aneurysm = (prob_bin >= CFG.threshold)

        results = []
        for i in range(images.size(0)):
            score = prob_bin[i].item()
            if has_aneurysm[i]:
                class_scores = out_multi[i].softmax(dim=0).tolist()
            else:
                class_scores = [0.1] * CFG.num_classes

            results.append({
                "aneurysm_score": score,
                "class_prediction": class_scores
            })

    return results


if DEBUG:
    load_models()
    serie_name = "1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647"
    test_path = os.path.join(CFG.series_root, serie_name)
    volume = process_dicom_series_safe(test_path)
    print("Shape:", volume.shape)
    print("Min/Max:", volume.min(), volume.max())

    test_path = os.path.join(CFG.series_root, serie_name)
    pred_df = predict(test_path)
    print(pred_df)


if DEBUG:
    train_df = pd.read_csv(TRAIN_CSV)
    sample_df = train_df.sample(n=5, random_state=42)

    dataset = MyDataset(
        sample_df,
        label_cols=LABEL_COLS,
        target_shape=CFG.target_shape,
        series_path=SERIES_ROOT_TRAIN
    )

    dataloader = DataLoader(
        dataset,
        batch_size=5,
        shuffle=True,
        num_workers=0
    )

    model = EfficientNetV2Hierarchical(**CFG.model_params, **CFG.backbone_params).to(DEVICE)
    model.load_state_dict(torch.load(f"{CFG.model_dir}/hierarchical_model.pth", map_location=DEVICE)['model'])
    model.eval()

    for images, _, _ in dataloader:
        result = predict_batch(images, model)
        print("ğŸ”� RÃ©sultat de prÃ©diction sur batch d'entraÃ®nement :")
        print(result)
        break


if DEBUG:
    shared_dir = '/kaggle/shared'
    shutil.rmtree(shared_dir, ignore_errors=True)
    os.makedirs(shared_dir, exist_ok=True)


# ====================================================
# Main Execution
# ====================================================

# Load models at startup
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

