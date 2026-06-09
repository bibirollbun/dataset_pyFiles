!pip install -q monai timm pydicom nibabel torchio


import os
import multiprocessing as mp
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import monai
from monai.data import CacheDataset, DataLoader as MonaiDataLoader
from monai.transforms import (
    EnsureChannelFirstd, Resized, NormalizeIntensityd, RandFlipd,
    RandRotated, Compose, ToTensord
)
import pydicom
from pathlib import Path
from tqdm import tqdm
from functools import partial
from sklearn.metrics import roc_auc_score
import SimpleITK as sitk
import h5py
import torchio as tio
from collections import OrderedDict
from typing import Tuple, List
from scipy import ndimage
import ast
from sklearn.model_selection import train_test_split
import nibabel as nb
import shutil


# Config
DATA_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/"
TRAIN_CSV = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
LOCALIZERS_CSV = "/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv"
SEGMENTATIONS_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/"
OUTPUT_DIR = "/kaggle/working/preprocessed/"
BATCH_SIZE = 8
NUM_EPOCHS = 50
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_SIZE = (128, 128, 64)  # Downsample resolution
TARGET_SPACING_MM = 1.0  # Isotropic spacing
CTA_WINDOW = (-1000, 1000)  # HU window for CTA
MRI_Z_CLIP = 3.0  # Z-score clip for MRI
LRU_CAPACITY = 10  # Memory cache capacity
ID_COL = "SeriesInstanceUID"
LABEL_COLS = [
    "Left Infraclinoid Internal Carotid Artery",
    "Right Infraclinoid Internal Carotid Artery",
    "Left Supraclinoid Internal Carotid Artery",
    "Right Supraclinoid Internal Carotid Artery",
    "Left Middle Cerebral Artery",
    "Right Middle Cerebral Artery",
    "Anterior Communicating Artery",
    "Left Anterior Cerebral Artery",
    "Right Anterior Cerebral Artery",
    "Left Posterior Communicating Artery",
    "Right Posterior Communicating Artery",
    "Basilar Tip",
    "Other Posterior Circulation",
    "Aneurysm Present"
]

# Utility Functions
def _safe_zoom(volume: np.ndarray, zoom_factors: Tuple[float, ...], order: int = 1) -> np.ndarray:
    """Robust wrapper around ndimage.zoom to avoid rank mismatch and invalid factors."""
    volume = np.nan_to_num(volume, copy=False)
    zf = tuple(float(max(1e-6, f)) for f in zoom_factors)  # avoid zeros/negatives
    if len(zf) != volume.ndim:
        if len(zf) > volume.ndim:
            zf = zf[:volume.ndim]
        else:
            zf = (1.0,) * (volume.ndim - len(zf)) + zf
    return ndimage.zoom(volume, zf, order=order)

def _resize_slice(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Resize a 2D slice to (out_h, out_w) using safe zoom."""
    h, w = arr.shape
    if h == out_h and w == out_w:
        return arr.astype(np.float32, copy=False)
    zy = out_h / max(h, 1)
    zx = out_w / max(w, 1)
    return _safe_zoom(arr, (zy, zx), order=1).astype(np.float32, copy=False)

# DICOM Processor Class
class DICOMProcessor:
    def __init__(
        self,
        target_size: Tuple[int, int, int] = TARGET_SIZE,
        target_spacing_mm: float = TARGET_SPACING_MM,
        cta_window: Tuple[float, float] = CTA_WINDOW,
        mri_z_clip: float = MRI_Z_CLIP,
        lru_capacity: int = LRU_CAPACITY,
    ):
        self.target_size = target_size
        self.target_spacing_mm = target_spacing_mm
        self.cta_window = cta_window
        self.mri_z_clip = mri_z_clip
        self.memory_cache = OrderedDict()
        self.lru_capacity = lru_capacity

    def _cache_put(self, key: str, vol: np.ndarray):
        self.memory_cache[key] = vol
        self.memory_cache.move_to_end(key)
        if len(self.memory_cache) > self.lru_capacity:
            self.memory_cache.popitem(last=False)

    def _cache_get(self, key: str):
        if key in self.memory_cache:
            vol = self.memory_cache[key]
            self.memory_cache.move_to_end(key)
            return vol
        return None

    def _sort_slices(self, ds_list: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
        try:
            orient = np.array(ds_list[0].ImageOrientationPatient, dtype=np.float32)
            row, col = orient[:3], orient[3:]
            normal = np.cross(row, col)
            return sorted(ds_list, key=lambda ds: float(np.dot(np.array(ds.ImagePositionPatient, dtype=np.float32), normal)))
        except Exception:
            return sorted(ds_list, key=lambda ds: getattr(ds, "InstanceNumber", 0))

    def _get_spacing(self, ds_sorted: List[pydicom.dataset.FileDataset], has_multiframe: bool = False) -> Tuple[float, float, float]:
        try:
            dy, dx = map(float, ds_sorted[0].PixelSpacing)
        except Exception:
            dy, dx = 1.0, 1.0
        if has_multiframe:
            dz = float(getattr(ds_sorted[0], "SpacingBetweenSlices", getattr(ds_sorted[0], "SliceThickness", 1.0)))
        else:
            zs = [np.linalg.norm(np.array(ds_sorted[i].ImagePositionPatient, dtype=np.float32) - np.array(ds_sorted[i-1].ImagePositionPatient, dtype=np.float32)) for i in range(1, len(ds_sorted))]
            dz = np.median([z for z in zs if z > 0]) if zs else float(getattr(ds_sorted[0], "SliceThickness", 1.0))
        return (dz if dz > 0 else 1.0, dy if dy > 0 else 1.0, dx if dx > 0 else 1.0)

    def _choose_base_shape(self, ds_list: List[pydicom.dataset.FileDataset]) -> Tuple[int, int]:
        shapes = [(int(ds.Rows), int(ds.Columns)) for ds in ds_list if hasattr(ds, "Rows") and hasattr(ds, "Columns")]
        if not shapes:
            shapes = [(ds.pixel_array.shape[-2], ds.pixel_array.shape[-1]) for ds in ds_list if hasattr(ds, "pixel_array")]
        vals, counts = np.unique(shapes, return_counts=True, axis=0)
        return tuple(int(x) for x in vals[counts.argmax()]) if vals.size else (512, 512)

    def _normalize_by_modality(self, volume: np.ndarray, modality_tag: str) -> np.ndarray:
        volume = np.nan_to_num(volume, copy=False)
        if modality_tag == "CT":
            c, w = self.cta_window
            lo, hi = c - w / 2.0, c + w / 2.0
            return np.clip((volume - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
        else:
            mean = float(volume.mean())
            std = float(volume.std() + 1e-6)
            v = np.clip((volume - mean) / std, -self.mri_z_clip, self.mri_z_clip)
            return ((v + self.mri_z_clip) / (2.0 * self.mri_z_clip)).astype(np.float32)

    def load_dicom_series(self, series_path: str) -> np.ndarray:
        series_id = os.path.basename(series_path)
        m = self._cache_get(series_id)
        if m is not None and m.shape == self.target_size:
            return m

        try:
            dicoms = []
            for root, _, files in os.walk(series_path):
                for f in files:
                    if f.endswith(".dcm"):
                        try:
                            ds = pydicom.dcmread(os.path.join(root, f), force=True)
                            if hasattr(ds, "PixelData"):
                                dicoms.append(ds)
                        except Exception as e:
                            print(f"[DICOM read] {series_id}: {e}")
                            continue
            if not dicoms:
                raise ValueError(f"No valid DICOM files in {series_path}")

            dicoms = self._sort_slices(dicoms)
            has_multiframe = any(getattr(ds, "NumberOfFrames", 1) > 1 for ds in dicoms)
            spacing = self._get_spacing(dicoms, has_multiframe)
            base_h, base_w = self._choose_base_shape(dicoms)
            modality_tag = getattr(dicoms[0], "Modality", "").upper()

            vol_slices = []
            for ds in dicoms:
                arr = ds.pixel_array
                if arr.ndim >= 3:
                    h, w = arr.shape[-2], arr.shape[-1]
                    n = int(np.prod(arr.shape[:-2]))
                    arr = arr.reshape(n, h, w)
                else:
                    arr = arr[np.newaxis, ...]
                for sl in arr:
                    sl = sl.astype(np.float32)
                    if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
                        sl = sl.max() - sl
                    slope = float(getattr(ds, "RescaleSlope", 1.0))
                    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                    sl = sl * slope + intercept
                    sl = _resize_slice(sl, base_h, base_w)
                    vol_slices.append(sl)

            if not vol_slices:
                raise ValueError(f"No valid slices in {series_id}")
            volume = np.stack(vol_slices, axis=0)

            # Resample to target spacing
            dz, dy, dx = spacing
            z, y, x = volume.shape
            newD = max(1, int(round(z * dz / self.target_spacing_mm)))
            newH = max(1, int(round(y * dy / self.target_spacing_mm)))
            newW = max(1, int(round(x * dx / self.target_spacing_mm)))
            volume = _safe_zoom(volume, (newD / z, newH / y, newW / x), order=1)

            # Resize to target grid
            tz, ty, tx = self.target_size
            z, y, x = volume.shape
            volume = _safe_zoom(volume, (tz / z, ty / y, tx / x), order=1).astype(np.float32)

            volume = self._normalize_by_modality(volume, modality_tag)
            self._cache_put(series_id, volume)
            return volume

        except Exception as e:
            print(f"[Processor] Error processing {series_id}: {e}")
            vol = np.zeros(self.target_size, dtype=np.float32)
            self._cache_put(series_id, vol)
            return vol
            
def process_row(args):
    row_dict, data_dir, label_cols, processor = args
    series_id = row_dict["SeriesInstanceUID"]
    series_path = os.path.join(data_dir, series_id)
    labels = [row_dict.get(col, 0) for col in label_cols]
    volume = processor.load_dicom_series(series_path)
    return series_path, volume, np.array(labels, dtype=np.float32)

def preprocess_dataset():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    train_df = pd.read_csv(TRAIN_CSV)
    # train_df = train_df.sample(n=50, random_state=42)  
    print(f"Preprocessing {len(train_df)} series (with both positive & negative cases)")

    localizers_df = pd.read_csv(LOCALIZERS_CSV)
    def parse_coordinates(coord_str):
        try:
            d = ast.literal_eval(coord_str)
            return pd.Series({"center_x": d.get("x", np.nan), "center_y": d.get("y", np.nan), "center_z": d.get("z", np.nan) if "z" in d else np.nan})
        except:
            return pd.Series({"center_x": np.nan, "center_y": np.nan, "center_z": np.nan})
    coord_df = localizers_df["coordinates"].apply(parse_coordinates)
    localizers_df = pd.concat([localizers_df.drop(columns=["coordinates"]), coord_df], axis=1)
    localizers_df = localizers_df.groupby("SeriesInstanceUID")[["center_x", "center_y", "center_z"]].mean().reset_index()

    train_df = train_df.merge(localizers_df, on="SeriesInstanceUID", how="left")
    for col in ["center_x", "center_y", "center_z"]:
        train_df[col] = train_df[col].fillna(0)

    print("Label distribution (Aneurysm Present):")
    print(train_df["Aneurysm Present"].value_counts())

    processor = DICOMProcessor(target_size=TARGET_SIZE, target_spacing_mm=TARGET_SPACING_MM, cta_window=CTA_WINDOW, mri_z_clip=MRI_Z_CLIP)
    args = [(row.to_dict(), DATA_DIR, LABEL_COLS, processor) for _, row in train_df.iterrows()]

    series_data = []
    results = []
    for arg in tqdm(args, desc="Preprocessing"):
        results.append(process_row(arg))

    for series_path, volume, labels in results:
        if volume is not None and not np.all(volume == 0):
            series_id = os.path.basename(series_path)
            np.save(os.path.join(OUTPUT_DIR, f"{series_id}.npy"), volume)
            # Removed segmentation saving
            series_data.append({
                "image": os.path.join(OUTPUT_DIR, f"{series_id}.npy"),
                "labels": labels.tolist()
            })

    pd.DataFrame(series_data).to_csv("series_metadata.csv", index=False)
    print("Preprocessing done!")

class AneurysmDataset(Dataset):
    def __init__(self, data: pd.DataFrame, transform=None):
        self.data = data
        self.transform = transform
        self.segmentations_dir = SEGMENTATIONS_DIR

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = np.load(row["image"]).astype(np.float32)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)
        
        # Load and preprocess segmentation mask on-the-fly
        series_id = os.path.basename(row["image"]).replace(".npy", "")
        seg_path = os.path.join(self.segmentations_dir, f"{series_id}.nii.gz")
        if os.path.exists(seg_path):
            try:
                seg_img = nb.load(seg_path)
                seg_data = seg_img.get_fdata()
                tz, ty, tx = TARGET_SIZE
                z, y, x = seg_data.shape
                seg_data = _safe_zoom(seg_data, (tz / z, ty / y, tx / x), order=0).astype(np.float32)
                seg_data = (seg_data > 0).astype(np.float32)
            except Exception as e:
                print(f"Error loading segmentation for {series_id}: {e}")
                seg_data = np.zeros(TARGET_SIZE, dtype=np.float32)
        else:
            seg_data = np.zeros(TARGET_SIZE, dtype=np.float32)
        
        if seg_data.ndim == 3:
            seg_data = np.expand_dims(seg_data, axis=0)
        
        labels = np.array(ast.literal_eval(row["labels"]), dtype=np.float32) if isinstance(row["labels"], str) else np.array(row["labels"], dtype=np.float32)
        
        if self.transform:
            subject = tio.Subject(image=tio.ScalarImage(tensor=image), segmentation=tio.LabelMap(tensor=seg_data))
            transformed = self.transform(subject)
            image = transformed["image"].numpy()
            seg_data = transformed["segmentation"].numpy()

        image_tensor = torch.from_numpy(image).float()
        segmentation_tensor = torch.from_numpy(seg_data).float()
        labels_tensor = torch.from_numpy(labels)
        return image_tensor, segmentation_tensor, labels_tensor


train_transforms = tio.Compose([tio.RandomFlip(axes=(0,), p=0.5), tio.RandomAffine(degrees=10, p=0.5), tio.ToCanonical()])
val_transforms = tio.Compose([tio.ToCanonical()])

# # Import additional MONAI modules for U-Net and DiceLoss
from monai.networks.nets import UNet
from monai.losses import DiceLoss

class MultiTaskUNet(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.unet = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,  # For binary segmentation
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm="batch",
            dropout=0.1
        )
        # Classification head: Adjust input size to match seg_logits (1 channel after pooling)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(1, num_classes)  # Changed from 256 to 1 to match seg_logits channels

    def forward(self, x):
        # U-Net forward pass
        seg_logits = self.unet(x)  # Shape: (B, 1, D, H, W)
        
        # Classification head: Pool segmentation output
        features = self.pool(seg_logits)  # Shape: (B, 1, 1, 1, 1)
        features = features.view(features.size(0), -1)  # Shape: (B, 1)
        class_logits = self.fc(features)  # Shape: (B, num_classes)
        
        return class_logits, seg_logits


# Training Loop
def train_one_epoch(model, loader, optimizer, scaler, criterion_class, criterion_seg):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Training"):
        images, seg_masks, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            class_outputs, seg_outputs = model(images)
            loss_class = criterion_class(class_outputs, labels)
            loss_seg = criterion_seg(seg_outputs, seg_masks)
            loss = loss_class + loss_seg  # Combined loss (equal weight; adjust if needed)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return np.mean(losses)

def validate(model, loader, criterion_class, criterion_seg):
    model.eval()
    preds, truths = [], []
    losses_class, losses_seg = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images, seg_masks, labels = batch[0].to(DEVICE), batch[1].to(DEVICE), batch[2].to(DEVICE)
            class_outputs, seg_outputs = model(images)
            loss_class = criterion_class(class_outputs, labels)
            loss_seg = criterion_seg(seg_outputs, seg_masks)
            losses_class.append(loss_class.item())
            losses_seg.append(loss_seg.item())
            preds.append(class_outputs.cpu().numpy())
            truths.append(labels.cpu().numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    
    aucs = []
    for i in range(len(LABEL_COLS)):
        y_true = truths[:, i]
        y_pred = preds[:, i]
        if len(np.unique(y_true)) == 1:
            print(f"Warning: Only one class present in column {LABEL_COLS[i]}. AUC set to 0.5.")
            aucs.append(0.5)
        else:
            aucs.append(roc_auc_score(y_true, y_pred))
    
    final_score = (aucs[-1] + np.mean(aucs[:-1])) / 2
    return np.mean(losses_class), np.mean(losses_seg), final_score, aucs

def main():
    # Preprocess
    preprocess_dataset()

    # Data
    metadata_file = "/kaggle/working/series_metadata.csv"
    metadata = pd.read_csv(metadata_file)
    metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)
    train_idx = int(0.8 * len(metadata))
    train_data = metadata[:train_idx]
    val_data = metadata[train_idx:]

    train_dataset = AneurysmDataset(train_data, train_transforms)
    val_dataset = AneurysmDataset(val_data, val_transforms)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Model
    model = MultiTaskUNet(num_classes=len(LABEL_COLS)).to(DEVICE)
    criterion_class = nn.BCEWithLogitsLoss()
    criterion_seg = DiceLoss(sigmoid=True)  # DiceLoss for segmentation
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    # Early stopping parameters
    patience = 5
    counter = 0
    best_score = 0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion_class, criterion_seg)
        val_loss_class, val_loss_seg, val_score, aucs = validate(model, val_loader, criterion_class, criterion_seg)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss Class: {val_loss_class:.4f}, "
              f"Val Loss Seg: {val_loss_seg:.4f}, "
              f"Val AUC: {val_score:.4f}")

        # Nếu tốt hơn -> lưu lại
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            torch.save(model.state_dict(), "best_model.pth")
            counter = 0  # reset bộ đếm
        else:
            counter += 1
            print(f"No improvement. EarlyStopping counter: {counter}/{patience}")

        # Nếu chờ đủ patience mà không cải thiện -> dừng
        if counter >= patience:
            print(f"Early stopping at epoch {epoch+1}. Best epoch was {best_epoch+1} with score {best_score:.4f}")
            break


if __name__ == "__main__":
    main()


# metadata_file = "/kaggle/working/series_metadata.csv"
# metadata = pd.read_csv(metadata_file)
# metadata.head(5)


# print(metadata['labels'].iloc[0])


# print(metadata['image'].iloc[0])


# class MultiTaskUNet(nn.Module):
#     def __init__(self, num_classes=14):
#         super().__init__()
#         self.unet = UNet(
#             spatial_dims=3,
#             in_channels=1,
#             out_channels=1,
#             channels=(16, 32, 64, 128, 256),
#             strides=(2, 2, 2, 2),
#             num_res_units=2,
#             norm="batch",
#             dropout=0.1
#         )
#         # Separate feature extractor for classification
#         self.feature_extractor = nn.Sequential(
#             nn.Conv3d(1, 16, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(2),
#             nn.Conv3d(16, 32, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(2),
#             nn.Conv3d(32, 64, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(2),
#             nn.Conv3d(64, 128, kernel_size=3, padding=1),
#             nn.ReLU(),
#             nn.MaxPool3d(2),
#             nn.Conv3d(128, 256, kernel_size=3, padding=1),
#             nn.ReLU()
#         )
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Linear(256, num_classes)

#     def forward(self, x):
#         # Segmentation path
#         seg_logits = self.unet(x)
        
#         # Classification path
#         features = self.feature_extractor(x)  # Shape: (B, 256, D', H', W')
#         features = self.pool(features)  # Shape: (B, 256, 1, 1, 1)
#         features = features.view(features.size(0), -1)  # Shape: (B, 256)
#         class_logits = self.fc(features)  # Shape: (B, num_classes)
        
#         return class_logits, seg_logits


# import os
# import shutil
# from collections import defaultdict
# import numpy as np
# import torch
# import torch.nn as nn
# import pandas as pd
# import polars as pl
# import pydicom
# from torch.cuda.amp import autocast
# from tqdm import tqdm

# import monai
# import kaggle_evaluation.rsna_inference_server

# # Configuration from your previous code
# ID_COL = 'SeriesInstanceUID'
# LABEL_COLS = [
#     'Left Infraclinoid Internal Carotid Artery',
#     'Right Infraclinoid Internal Carotid Artery',
#     'Left Supraclinoid Internal Carotid Artery',
#     'Right Supraclinoid Internal Carotid Artery',
#     'Left Middle Cerebral Artery',
#     'Right Middle Cerebral Artery',
#     'Anterior Communicating Artery',
#     'Left Anterior Cerebral Artery',
#     'Right Anterior Cerebral Artery',
#     'Left Posterior Communicating Artery',
#     'Right Posterior Communicating Artery',
#     'Basilar Tip',
#     'Other Posterior Circulation',
#     'Aneurysm Present',
# ]
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TARGET_SIZE = (128, 128, 64)
# CTA_WINDOW = (-1000, 1000)

# # DICOM Tag Allowlist
# DICOM_TAG_ALLOWLIST = [
#     'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID', 'HighBit',
#     'ImageOrientationPatient', 'ImagePositionPatient', 'InstanceNumber', 'Modality',
#     'PatientID', 'PhotometricInterpretation', 'PixelRepresentation', 'PixelSpacing',
#     'PlanarConfiguration', 'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows',
#     'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel', 'SliceThickness',
#     'SpacingBetweenSlices', 'StudyInstanceUID', 'TransferSyntaxUID',
# ]

# # Utility Functions (simplified from your code)
# def _safe_zoom(volume: np.ndarray, zoom_factors: tuple, order: int = 1) -> np.ndarray:
#     """Robust wrapper around ndimage.zoom."""
#     volume = np.nan_to_num(volume, copy=False)
#     zf = tuple(float(max(1e-6, f)) for f in zoom_factors)
#     if len(zf) != volume.ndim:
#         if len(zf) > volume.ndim:
#             zf = zf[:volume.ndim]
#         else:
#             zf = (1.0,) * (volume.ndim - len(zf)) + zf
#     return ndimage.zoom(volume, zf, order=order)

# def _resize_slice(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
#     """Resize a 2D slice to (out_h, out_w)."""
#     h, w = arr.shape
#     if h == out_h and w == out_w:
#         return arr.astype(np.float32, copy=False)
#     zy = out_h / max(h, 1)
#     zx = out_w / max(w, 1)
#     return _safe_zoom(arr, (zy, zx), order=1).astype(np.float32, copy=False)

# # Model Definition
# class EfficientNet3D(nn.Module):
#     def __init__(self, num_classes=14):
#         super().__init__()
#         self.backbone = monai.networks.nets.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=1, pretrained=False)
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Linear(self.backbone._fc.in_features, num_classes)
#         self.backbone._fc = nn.Identity()

#     def forward(self, x):
#         if x.dim() != 5:
#             raise ValueError(f"Expected 5D input tensor, got {x.shape}")
#         x = self.backbone(x)
#         if x.dim() < 3:
#             x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

# # Prediction Function
# def predict(series_path: str) -> pl.DataFrame:
#     """Make a prediction for the given series."""
#     series_id = os.path.basename(series_path)
    
#     # Load all DICOM files
#     all_filepaths = []
#     for root, _, files in os.walk(series_path):
#         for file in files:
#             if file.endswith('.dcm'):
#                 all_filepaths.append(os.path.join(root, file))
#     all_filepaths.sort()

#     # Process DICOM series into a volume
#     vol_slices = []
#     for filepath in tqdm(all_filepaths, desc=f"Processing {series_id}"):
#         try:
#             ds = pydicom.dcmread(filepath, force=True)
#             arr = ds.pixel_array
#             if arr.ndim >= 3:
#                 h, w = arr.shape[-2], arr.shape[-1]
#                 n = int(np.prod(arr.shape[:-2]))
#                 arr = arr.reshape(n, h, w)
#             else:
#                 arr = arr[np.newaxis, ...]
#             for sl in arr:
#                 sl = sl.astype(np.float32)
#                 if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
#                     sl = sl.max() - sl
#                 slope = float(getattr(ds, "RescaleSlope", 1.0))
#                 intercept = float(getattr(ds, "RescaleIntercept", 0.0))
#                 sl = sl * slope + intercept
#                 sl = _resize_slice(sl, TARGET_SIZE[1], TARGET_SIZE[2])  # Resize 2D slices
#                 vol_slices.append(sl)
#         except Exception as e:
#             print(f"[DICOM read] {series_id}: {e}")
#             continue

#     if not vol_slices:
#         print(f"No valid slices in {series_id}")
#         volume = np.zeros(TARGET_SIZE, dtype=np.float32)
#     else:
#         volume = np.stack(vol_slices, axis=0)
#         # Resample and resize to target size
#         z, y, x = volume.shape
#         volume = _safe_zoom(volume, (TARGET_SIZE[0] / z, TARGET_SIZE[1] / y, TARGET_SIZE[2] / x), order=1).astype(np.float32)

#     # Normalize based on modality (assuming CTA for test set)
#     c, w = CTA_WINDOW
#     lo, hi = c - w / 2.0, c + w / 2.0
#     volume = np.clip((volume - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)

#     # Prepare input tensor
#     volume = np.expand_dims(volume, axis=0)  # Add channel dimension: (1, D, H, W)
#     volume = torch.from_numpy(volume).float().unsqueeze(0).to(DEVICE)  # Add batch dimension: (1, 1, D, H, W)

#     # Load trained model
#     model = EfficientNet3D(num_classes=len(LABEL_COLS)).to(DEVICE)
#     model.load_state_dict(torch.load("best_model.pth"))  # Load your trained model
#     model.eval()

#     # Inference
#     with torch.no_grad(), autocast():
#         outputs = model(volume)
#         predictions = torch.sigmoid(outputs).cpu().numpy()[0]  # Convert logits to probabilities

#     # Create prediction DataFrame
#     predictions = pl.DataFrame(
#         data=[[series_id] + predictions.tolist()],
#         schema=[ID_COL, *LABEL_COLS],
#         orient='row',
#     )

#     # Clean up temporary files
#     shutil.rmtree('/kaggle/shared', ignore_errors=True)

#     return predictions.drop(ID_COL)

# # Inference Server Setup
# inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

# if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
#     inference_server.serve()
# else:
#     inference_server.run_local_gateway()
#     display(pl.read_parquet('/kaggle/working/submission.parquet'))






# # Preprocessing Function
# def preprocess_dataset():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     train_df = pd.read_csv(TRAIN_CSV)

#     # Keep only aneurysm cases (for now, can expand to include negatives later)
#     train_df = train_df[train_df["Aneurysm Present"] == 1].copy()
#     print(f"Preprocessing {len(train_df)} series with aneurysm")

#     # --- Load and parse localizers ---
#     localizers_df = pd.read_csv(LOCALIZERS_CSV)

#     def parse_coordinates(coord_str):
#         try:
#             d = ast.literal_eval(coord_str)
#             return pd.Series({
#                 "center_x": d.get("x", np.nan),
#                 "center_y": d.get("y", np.nan),
#                 "center_z": d.get("z", np.nan) if "z" in d else np.nan
#             })
#         except Exception:
#             return pd.Series({"center_x": np.nan, "center_y": np.nan, "center_z": np.nan})

#     coord_df = localizers_df["coordinates"].apply(parse_coordinates)
#     localizers_df = pd.concat([localizers_df.drop(columns=["coordinates"]), coord_df], axis=1)

#     # Average coordinates per series
#     localizers_df = localizers_df.groupby("SeriesInstanceUID")[["center_x", "center_y", "center_z"]].mean().reset_index()

#     # Merge with training data
#     train_df = train_df.merge(localizers_df, on="SeriesInstanceUID", how="left")
#     train_df = train_df[:100]

#     # Fill NaNs with global mean
#     for col in ["center_x", "center_y", "center_z"]:
#         train_df[col] = train_df[col].fillna(train_df[col].mean())

#     # --- Prepare multiprocessing args ---
#     processor = DICOMProcessor(
#         target_size=TARGET_SIZE,
#         target_spacing_mm=TARGET_SPACING_MM,
#         cta_window=CTA_WINDOW,
#         mri_z_clip=MRI_Z_CLIP
#     )

#     args = [(row._asdict(), DATA_DIR, LABEL_COLS, processor) 
#             for row in train_df.itertuples(index=False)]

#     series_data = []
#     with mp.Pool(processes=min(mp.cpu_count(), 4)) as pool:
#         results = list(tqdm(pool.imap(process_row, args),
#                             total=len(train_df), desc="Preprocessing"))

#     for series_path, volume, labels in results:
#         if volume is not None and not np.all(volume == 0):
#             series_id = os.path.basename(series_path)
#             np.save(os.path.join(OUTPUT_DIR, f"{series_id}.npy"), volume)
            
#             # Load and process segmentation mask
#             seg_path = os.path.join(SEGMENTATIONS_DIR, f"{series_id}.nii.gz")  # Adjust extension if needed
#             try:
#                 seg_mask = sitk.ReadImage(seg_path)
#                 seg_array = sitk.GetArrayFromImage(seg_mask)
#                 # Resize segmentation mask to match volume shape
#                 seg_array = _safe_zoom(seg_array, (TARGET_SIZE[0] / seg_array.shape[0],
#                                                  TARGET_SIZE[1] / seg_array.shape[1],
#                                                  TARGET_SIZE[2] / seg_array.shape[2]), order=0)
#                 seg_array = (seg_array > 0).astype(np.float32)  # Binary mask
#             except Exception as e:
#                 print(f"[Segmentation] Error loading {series_id}: {e}")
#                 seg_array = np.zeros(TARGET_SIZE, dtype=np.float32)

#             # Crop volume using segmentation mask (simple bounding box approach)
#             if np.any(seg_array):
#                 coords = np.where(seg_array)
#                 z_min, z_max = max(0, coords[0].min() - 10), min(TARGET_SIZE[0], coords[0].max() + 10)
#                 y_min, y_max = max(0, coords[1].min() - 10), min(TARGET_SIZE[1], coords[1].max() + 10)
#                 x_min, x_max = max(0, coords[2].min() - 10), min(TARGET_SIZE[2], coords[2].max() + 10)
#                 volume_cropped = volume[z_min:z_max, y_min:y_max, x_min:x_max]
#                 # Resize back to TARGET_SIZE if cropped region is smaller
#                 if volume_cropped.shape != TARGET_SIZE:
#                     volume_cropped = _safe_zoom(volume_cropped, (TARGET_SIZE[0] / volume_cropped.shape[0],
#                                                                TARGET_SIZE[1] / volume_cropped.shape[1],
#                                                                TARGET_SIZE[2] / volume_cropped.shape[2]))
#             else:
#                 volume_cropped = volume

#             np.save(os.path.join(OUTPUT_DIR, f"{series_id}_cropped.npy"), volume_cropped)
#             series_data.append({
#                 "image": os.path.join(OUTPUT_DIR, f"{series_id}_cropped.npy"),
#                 "labels": labels.tolist()
#             })

#     pd.DataFrame(series_data).to_csv(os.path.join(OUTPUT_DIR, "series_metadata.csv"), index=False)
#     print("Preprocessing done!")

# # Dataset
# class AneurysmDataset(Dataset):
#     def __init__(self, data_file: str, transform=None):
#         self.data = pd.read_csv(data_file)
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
#         image = np.load(row["image"]).astype(np.float32)
#         # Ensure image is 3D (D, H, W) and add channel dimension
#         if image.ndim == 3:
#             image = np.expand_dims(image, axis=0)  # Shape: (1, D, H, W)
#         elif image.ndim == 4 and image.shape[0] == 1:
#             image = image.squeeze(0)  # Ensure no extra batch dimension from transform
        
#         # Parse labels string using ast.literal_eval
#         labels = np.array(ast.literal_eval(row["labels"]), dtype=np.float32) if isinstance(row["labels"], str) else np.array(row["labels"], dtype=np.float32)
        
#         # Light augmentation with torchio
#         if self.transform:
#             subject = tio.Subject(image=tio.ScalarImage(tensor=image))
#             transformed = self.transform(subject)
#             image = transformed["image"].numpy()

#         # Ensure 5D tensor for model (batch is added by DataLoader)
#         image_tensor = torch.from_numpy(image).float()  # Shape: (1, D, H, W)
#         labels_tensor = torch.from_numpy(labels)
#         return image_tensor, labels_tensor

