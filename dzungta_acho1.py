!pip install -q monai timm pydicom nibabel torchio


# import os
# import multiprocessing as mp
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# import monai
# from monai.data import CacheDataset, DataLoader as MonaiDataLoader
# from monai.transforms import (
#     EnsureChannelFirstd, Resized, NormalizeIntensityd, RandFlipd,
#     RandRotated, Compose, ToTensord
# )
# import pydicom
# from pathlib import Path
# from tqdm import tqdm
# from functools import partial
# from sklearn.metrics import roc_auc_score
# import SimpleITK as sitk
# import h5py
# import torchio as tio
# from collections import OrderedDict
# from typing import Tuple, List
# from scipy import ndimage
# import ast


# # Config
# DATA_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/"
# TRAIN_CSV = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
# LOCALIZERS_CSV = "/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv"
# SEGMENTATIONS_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/"
# OUTPUT_DIR = "/kaggle/working/preprocessed/"
# BATCH_SIZE = 8
# NUM_EPOCHS = 5
# LR = 1e-4
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# TARGET_SIZE = (128, 128, 64)  # Downsample resolution
# TARGET_SPACING_MM = 1.0  # Isotropic spacing
# CTA_WINDOW = (-1000, 1000)  # HU window for CTA
# MRI_Z_CLIP = 3.0  # Z-score clip for MRI
# LRU_CAPACITY = 10  # Memory cache capacity
# ID_COL = "SeriesInstanceUID"
# LABEL_COLS = [
#     "Left Infraclinoid Internal Carotid Artery",
#     "Right Infraclinoid Internal Carotid Artery",
#     "Left Supraclinoid Internal Carotid Artery",
#     "Right Supraclinoid Internal Carotid Artery",
#     "Left Middle Cerebral Artery",
#     "Right Middle Cerebral Artery",
#     "Anterior Communicating Artery",
#     "Left Anterior Cerebral Artery",
#     "Right Anterior Cerebral Artery",
#     "Left Posterior Communicating Artery",
#     "Right Posterior Communicating Artery",
#     "Basilar Tip",
#     "Other Posterior Circulation",
#     "Aneurysm Present"
# ]

# # Utility Functions
# def _safe_zoom(volume: np.ndarray, zoom_factors: Tuple[float, ...], order: int = 1) -> np.ndarray:
#     """Robust wrapper around ndimage.zoom to avoid rank mismatch and invalid factors."""
#     volume = np.nan_to_num(volume, copy=False)
#     zf = tuple(float(max(1e-6, f)) for f in zoom_factors)  # avoid zeros/negatives
#     if len(zf) != volume.ndim:
#         if len(zf) > volume.ndim:
#             zf = zf[:volume.ndim]
#         else:
#             zf = (1.0,) * (volume.ndim - len(zf)) + zf
#     return ndimage.zoom(volume, zf, order=order)

# def _resize_slice(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
#     """Resize a 2D slice to (out_h, out_w) using safe zoom."""
#     h, w = arr.shape
#     if h == out_h and w == out_w:
#         return arr.astype(np.float32, copy=False)
#     zy = out_h / max(h, 1)
#     zx = out_w / max(w, 1)
#     return _safe_zoom(arr, (zy, zx), order=1).astype(np.float32, copy=False)

# # DICOM Processor Class
# class DICOMProcessor:
#     def __init__(
#         self,
#         target_size: Tuple[int, int, int] = TARGET_SIZE,
#         target_spacing_mm: float = TARGET_SPACING_MM,
#         cta_window: Tuple[float, float] = CTA_WINDOW,
#         mri_z_clip: float = MRI_Z_CLIP,
#         lru_capacity: int = LRU_CAPACITY,
#     ):
#         self.target_size = target_size
#         self.target_spacing_mm = target_spacing_mm
#         self.cta_window = cta_window
#         self.mri_z_clip = mri_z_clip
#         self.memory_cache = OrderedDict()
#         self.lru_capacity = lru_capacity

#     def _cache_put(self, key: str, vol: np.ndarray):
#         self.memory_cache[key] = vol
#         self.memory_cache.move_to_end(key)
#         if len(self.memory_cache) > self.lru_capacity:
#             self.memory_cache.popitem(last=False)

#     def _cache_get(self, key: str):
#         if key in self.memory_cache:
#             vol = self.memory_cache[key]
#             self.memory_cache.move_to_end(key)
#             return vol
#         return None

#     def _sort_slices(self, ds_list: List[pydicom.dataset.FileDataset]) -> List[pydicom.dataset.FileDataset]:
#         try:
#             orient = np.array(ds_list[0].ImageOrientationPatient, dtype=np.float32)
#             row, col = orient[:3], orient[3:]
#             normal = np.cross(row, col)
#             return sorted(ds_list, key=lambda ds: float(np.dot(np.array(ds.ImagePositionPatient, dtype=np.float32), normal)))
#         except Exception:
#             return sorted(ds_list, key=lambda ds: getattr(ds, "InstanceNumber", 0))

#     def _get_spacing(self, ds_sorted: List[pydicom.dataset.FileDataset], has_multiframe: bool = False) -> Tuple[float, float, float]:
#         try:
#             dy, dx = map(float, ds_sorted[0].PixelSpacing)
#         except Exception:
#             dy, dx = 1.0, 1.0
#         if has_multiframe:
#             dz = float(getattr(ds_sorted[0], "SpacingBetweenSlices", getattr(ds_sorted[0], "SliceThickness", 1.0)))
#         else:
#             zs = [np.linalg.norm(np.array(ds_sorted[i].ImagePositionPatient, dtype=np.float32) - np.array(ds_sorted[i-1].ImagePositionPatient, dtype=np.float32)) for i in range(1, len(ds_sorted))]
#             dz = np.median([z for z in zs if z > 0]) if zs else float(getattr(ds_sorted[0], "SliceThickness", 1.0))
#         return (dz if dz > 0 else 1.0, dy if dy > 0 else 1.0, dx if dx > 0 else 1.0)

#     def _choose_base_shape(self, ds_list: List[pydicom.dataset.FileDataset]) -> Tuple[int, int]:
#         shapes = [(int(ds.Rows), int(ds.Columns)) for ds in ds_list if hasattr(ds, "Rows") and hasattr(ds, "Columns")]
#         if not shapes:
#             shapes = [(ds.pixel_array.shape[-2], ds.pixel_array.shape[-1]) for ds in ds_list if hasattr(ds, "pixel_array")]
#         vals, counts = np.unique(shapes, return_counts=True, axis=0)
#         return tuple(int(x) for x in vals[counts.argmax()]) if vals.size else (512, 512)

#     def _normalize_by_modality(self, volume: np.ndarray, modality_tag: str) -> np.ndarray:
#         volume = np.nan_to_num(volume, copy=False)
#         if modality_tag == "CT":
#             c, w = self.cta_window
#             lo, hi = c - w / 2.0, c + w / 2.0
#             return np.clip((volume - lo) / (hi - lo + 1e-6), 0, 1).astype(np.float32)
#         else:
#             mean = float(volume.mean())
#             std = float(volume.std() + 1e-6)
#             v = np.clip((volume - mean) / std, -self.mri_z_clip, self.mri_z_clip)
#             return ((v + self.mri_z_clip) / (2.0 * self.mri_z_clip)).astype(np.float32)

#     def load_dicom_series(self, series_path: str) -> np.ndarray:
#         series_id = os.path.basename(series_path)
#         m = self._cache_get(series_id)
#         if m is not None and m.shape == self.target_size:
#             return m

#         try:
#             dicoms = []
#             for root, _, files in os.walk(series_path):
#                 for f in files:
#                     if f.endswith(".dcm"):
#                         try:
#                             ds = pydicom.dcmread(os.path.join(root, f), force=True)
#                             if hasattr(ds, "PixelData"):
#                                 dicoms.append(ds)
#                         except Exception as e:
#                             print(f"[DICOM read] {series_id}: {e}")
#                             continue
#             if not dicoms:
#                 raise ValueError(f"No valid DICOM files in {series_path}")

#             dicoms = self._sort_slices(dicoms)
#             has_multiframe = any(getattr(ds, "NumberOfFrames", 1) > 1 for ds in dicoms)
#             spacing = self._get_spacing(dicoms, has_multiframe)
#             base_h, base_w = self._choose_base_shape(dicoms)
#             modality_tag = getattr(dicoms[0], "Modality", "").upper()

#             vol_slices = []
#             for ds in dicoms:
#                 arr = ds.pixel_array
#                 if arr.ndim >= 3:
#                     h, w = arr.shape[-2], arr.shape[-1]
#                     n = int(np.prod(arr.shape[:-2]))
#                     arr = arr.reshape(n, h, w)
#                 else:
#                     arr = arr[np.newaxis, ...]
#                 for sl in arr:
#                     sl = sl.astype(np.float32)
#                     if getattr(ds, "PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
#                         sl = sl.max() - sl
#                     slope = float(getattr(ds, "RescaleSlope", 1.0))
#                     intercept = float(getattr(ds, "RescaleIntercept", 0.0))
#                     sl = sl * slope + intercept
#                     sl = _resize_slice(sl, base_h, base_w)
#                     vol_slices.append(sl)

#             if not vol_slices:
#                 raise ValueError(f"No valid slices in {series_id}")
#             volume = np.stack(vol_slices, axis=0)

#             # Resample to target spacing
#             dz, dy, dx = spacing
#             z, y, x = volume.shape
#             newD = max(1, int(round(z * dz / self.target_spacing_mm)))
#             newH = max(1, int(round(y * dy / self.target_spacing_mm)))
#             newW = max(1, int(round(x * dx / self.target_spacing_mm)))
#             volume = _safe_zoom(volume, (newD / z, newH / y, newW / x), order=1)

#             # Resize to target grid
#             tz, ty, tx = self.target_size
#             z, y, x = volume.shape
#             volume = _safe_zoom(volume, (tz / z, ty / y, tx / x), order=1).astype(np.float32)

#             volume = self._normalize_by_modality(volume, modality_tag)
#             self._cache_put(series_id, volume)
#             return volume

#         except Exception as e:
#             print(f"[Processor] Error processing {series_id}: {e}")
#             vol = np.zeros(self.target_size, dtype=np.float32)
#             self._cache_put(series_id, vol)
#             return vol


# def process_row(args):
#     row_dict, data_dir, label_cols, processor = args
#     series_id = row_dict["SeriesInstanceUID"]
#     series_path = os.path.join(data_dir, series_id)

#     # Labels
#     labels = [row_dict.get(col, 0) for col in label_cols]

#     # Load DICOM
#     volume = processor.load_dicom_series(series_path)
#     return series_path, volume, np.array(labels, dtype=np.float32)

# # def preprocess_dataset():
# #     os.makedirs(OUTPUT_DIR, exist_ok=True)
# #     train_df = pd.read_csv(TRAIN_CSV)

# #     # Keep only aneurysm cases
# #     train_df = train_df[train_df["Aneurysm Present"] == 1].copy()
# #     print(f"Preprocessing {len(train_df)} series with aneurysm")

# #     # --- Load and parse localizers ---
# #     localizers_df = pd.read_csv(LOCALIZERS_CSV)

# #     def parse_coordinates(coord_str):
# #         try:
# #             d = ast.literal_eval(coord_str)
# #             return pd.Series({
# #                 "center_x": d.get("x", np.nan),
# #                 "center_y": d.get("y", np.nan),
# #                 "center_z": d.get("z", np.nan) if "z" in d else np.nan
# #             })
# #         except Exception:
# #             return pd.Series({"center_x": np.nan, "center_y": np.nan, "center_z": np.nan})

# #     coord_df = localizers_df["coordinates"].apply(parse_coordinates)
# #     localizers_df = pd.concat([localizers_df.drop(columns=["coordinates"]), coord_df], axis=1)

# #     # Average coordinates per series
# #     localizers_df = localizers_df.groupby("SeriesInstanceUID")[["center_x", "center_y", "center_z"]].mean().reset_index()

# #     # Merge with training data
# #     train_df = train_df.merge(localizers_df, on="SeriesInstanceUID", how="left")
# #     train_df = train_df[:100]

# #     # Fill NaNs with global mean
# #     for col in ["center_x", "center_y", "center_z"]:
# #         train_df[col] = train_df[col].fillna(train_df[col].mean())

# #     # --- Prepare multiprocessing args ---
# #     processor = DICOMProcessor(
# #         target_size=TARGET_SIZE,
# #         target_spacing_mm=TARGET_SPACING_MM,
# #         cta_window=CTA_WINDOW,
# #         mri_z_clip=MRI_Z_CLIP
# #     )

# #     args = [(row._asdict(), DATA_DIR, LABEL_COLS, processor) 
# #             for row in train_df.itertuples(index=False)]

# #     series_data = []
# #     with mp.Pool(processes=min(mp.cpu_count(), 4)) as pool:
# #         results = list(tqdm(pool.imap(process_row, args),
# #                             total=len(train_df), desc="Preprocessing"))

# #     for series_path, volume, labels in results:
# #         if volume is not None and not np.all(volume == 0):
# #             series_id = os.path.basename(series_path)
# #             np.save(os.path.join(OUTPUT_DIR, f"{series_id}.npy"), volume)
# #             series_data.append({
# #                 "image": os.path.join(OUTPUT_DIR, f"{series_id}.npy"),
# #                 "labels": labels
# #             })

# #     pd.DataFrame(series_data).to_csv(os.path.join(OUTPUT_DIR, "series_metadata.csv"), index=False)
# #     print("Preprocessing done!")

# # # Dataset
# # class AneurysmDataset(Dataset):
# #     def __init__(self, data_file: str, transform=None):
# #         self.data = pd.read_csv(data_file)
# #         self.transform = transform

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, idx):
# #         row = self.data.iloc[idx]
# #         image = np.load(row["image"]).astype(np.float32)
# #         labels = np.array(eval(row["labels"]) if isinstance(row["labels"], str) else row["labels"], dtype=np.float32)
        
# #         # Light augmentation with torchio
# #         if self.transform:
# #             subject = tio.Subject(image=tio.ScalarImage(tensor=image[np.newaxis, ...]))
# #             transformed = self.transform(subject)
# #             image = transformed["image"].numpy().squeeze(0)

# #         image_tensor = torch.from_numpy(image).unsqueeze(0) if image.ndim == 3 else torch.from_numpy(image)  # Ensure (1, D, H, W)
# #         labels_tensor = torch.from_numpy(labels)
# #         return image_tensor, labels_tensor

# # train_transforms = tio.Compose([
# #     tio.RandomFlip(axes=(0,), p=0.5),
# #     tio.RandomAffine(degrees=10, p=0.5),
# #     tio.ToCanonical()  # Ensure canonical orientation, implicit tensor conversion
# # ])
# # val_transforms = tio.Compose([
# #     tio.ToCanonical()  # Ensure canonical orientation
# # ])


# # Preprocessing Function
# def preprocess_dataset():
#     os.makedirs(OUTPUT_DIR, exist_ok=True)
#     train_df = pd.read_csv(TRAIN_CSV)

#     # Keep only aneurysm cases
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
#     train_df = train_df[:50]

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
#             # Save labels as a JSON-compatible string
#             series_data.append({
#                 "image": os.path.join(OUTPUT_DIR, f"{series_id}.npy"),
#                 "labels": labels.tolist()  # Convert to list for JSON compatibility
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
#         # Parse labels string using ast.literal_eval
#         labels = np.array(ast.literal_eval(row["labels"]), dtype=np.float32) if isinstance(row["labels"], str) else np.array(row["labels"], dtype=np.float32)
        
#         # Light augmentation with torchio
#         if self.transform:
#             subject = tio.Subject(image=tio.ScalarImage(tensor=image[np.newaxis, ...]))
#             transformed = self.transform(subject)
#             image = transformed["image"].numpy().squeeze(0)

#         image_tensor = torch.from_numpy(image).unsqueeze(0) if image.ndim == 3 else torch.from_numpy(image)  # Ensure (1, D, H, W)
#         labels_tensor = torch.from_numpy(labels)
#         return image_tensor, labels_tensor
# # Model
# class EfficientNet3D(nn.Module):
#     def __init__(self, num_classes=14):
#         super().__init__()
#         self.backbone = monai.networks.nets.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=1)
#         self.pool = nn.AdaptiveAvgPool3d(1)
#         self.fc = nn.Linear(self.backbone._fc.in_features, num_classes)
#         self.backbone._fc = nn.Identity()

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.pool(x).view(x.size(0), -1)
#         x = self.fc(x)
#         return torch.sigmoid(x)

        
# # Training Loop
# def train_one_epoch(model, loader, optimizer, scaler, criterion):
#     model.train()
#     losses = []
#     for batch in tqdm(loader, desc="Training"):
#         images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
#         optimizer.zero_grad()
#         with torch.cuda.amp.autocast():
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#         scaler.scale(loss).backward()
#         scaler.step(optimizer)
#         scaler.update()
#         losses.append(loss.item())
#     return np.mean(losses)

# def validate(model, loader, criterion):
#     model.eval()
#     preds, truths = [], []
#     losses = []
#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Validating"):
#             images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             losses.append(loss.item())
#             preds.append(outputs.cpu().numpy())
#             truths.append(labels.cpu().numpy())
#     preds = np.concatenate(preds)
#     truths = np.concatenate(truths)
#     aucs = [roc_auc_score(truths[:, i], preds[:, i]) for i in range(len(LABEL_COLS))]
#     final_score = (aucs[-1] + np.mean(aucs[:-1])) / 2
#     return np.mean(losses), final_score, aucs

# # Main
# def main():
#     # Preprocess
#     preprocess_dataset()

#     # Data
#     metadata_file = os.path.join(OUTPUT_DIR, "series_metadata.csv")
#     train_idx = int(0.8 * len(pd.read_csv(metadata_file)))
#     train_data = pd.read_csv(metadata_file)[:train_idx]
#     val_data = pd.read_csv(metadata_file)[train_idx:]
#     train_dataset = CacheDataset(AneurysmDataset(metadata_file, train_transforms), cache_rate=0.1, num_workers=4)
#     val_dataset = CacheDataset(AneurysmDataset(metadata_file, val_transforms), cache_rate=0.1, num_workers=4)
#     train_loader = MonaiDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, prefetch_factor=2)
#     val_loader = MonaiDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, prefetch_factor=2)

#     # Model
#     model = EfficientNet3D(num_classes=len(LABEL_COLS)).to(DEVICE)
#     criterion = nn.BCELoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
#     scaler = torch.cuda.amp.GradScaler()

#     # Train
#     best_score = 0
#     for epoch in range(NUM_EPOCHS):
#         train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
#         val_loss, val_score, aucs = validate(model, val_loader, criterion)
#         print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_score:.4f}")
#         if val_score > best_score:
#             best_score = val_score
#             torch.save(model.state_dict(), "best_model.pth")



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

# Config
DATA_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/series/"
TRAIN_CSV = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"
LOCALIZERS_CSV = "/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv"
SEGMENTATIONS_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/"
OUTPUT_DIR = "/kaggle/working/preprocessed/"
BATCH_SIZE = 8
NUM_EPOCHS = 30
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

    # Labels
    labels = [row_dict.get(col, 0) for col in label_cols]

    # Load DICOM
    volume = processor.load_dicom_series(series_path)
    return series_path, volume, np.array(labels, dtype=np.float32)

def preprocess_dataset():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df = pd.read_csv(TRAIN_CSV)
    # train_df = train_df.sample(n=100, random_state=42)

    print(f"Preprocessing {len(train_df)} series (with both positive & negative cases)")

    # --- Load and parse localizers ---
    localizers_df = pd.read_csv(LOCALIZERS_CSV)

    def parse_coordinates(coord_str):
        try:
            d = ast.literal_eval(coord_str)
            return pd.Series({
                "center_x": d.get("x", np.nan),
                "center_y": d.get("y", np.nan),
                "center_z": d.get("z", np.nan) if "z" in d else np.nan
            })
        except Exception:
            return pd.Series({"center_x": np.nan, "center_y": np.nan, "center_z": np.nan})

    coord_df = localizers_df["coordinates"].apply(parse_coordinates)
    localizers_df = pd.concat([localizers_df.drop(columns=["coordinates"]), coord_df], axis=1)

    # Average coordinates per series
    localizers_df = localizers_df.groupby("SeriesInstanceUID")[["center_x", "center_y", "center_z"]].mean().reset_index()

    # Merge with training data (keep all rows)
    train_df = train_df.merge(localizers_df, on="SeriesInstanceUID", how="left")

    # Fill missing coordinates with global mean
    for col in ["center_x", "center_y", "center_z"]:
        train_df[col] = train_df[col].fillna(0)

    print("Label distribution (Aneurysm Present):")
    print(train_df["Aneurysm Present"].value_counts())

    # --- Prepare multiprocessing args ---
    processor = DICOMProcessor(
        target_size=TARGET_SIZE,
        target_spacing_mm=TARGET_SPACING_MM,
        cta_window=CTA_WINDOW,
        mri_z_clip=MRI_Z_CLIP
    )

    args = [(row.to_dict(), DATA_DIR, LABEL_COLS, processor) 
        for _, row in train_df.iterrows()]


    series_data = []
    with mp.Pool(processes=min(mp.cpu_count(), 4)) as pool:
        results = list(tqdm(pool.imap(process_row, args),
                            total=len(train_df), desc="Preprocessing"))

    for series_path, volume, labels in results:
        if volume is not None and not np.all(volume == 0):
            series_id = os.path.basename(series_path)
            np.save(os.path.join(OUTPUT_DIR, f"{series_id}.npy"), volume)
            series_data.append({
                "image": os.path.join(OUTPUT_DIR, f"{series_id}.npy"),
                "labels": labels.tolist()
            })

    pd.DataFrame(series_data).to_csv(os.path.join(OUTPUT_DIR, "series_metadata.csv"), index=False)
    print("Preprocessing done!")

# Dataset
class AneurysmDataset(Dataset):
    def __init__(self, data_file: str, transform=None):
        self.data = pd.read_csv(data_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = np.load(row["image"]).astype(np.float32)
        # Ensure image is 3D (D, H, W) and add channel dimension
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)  # Shape: (1, D, H, W)
        elif image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(0)  # Ensure no extra batch dimension from transform
        # Parse labels string using ast.literal_eval
        labels = np.array(ast.literal_eval(row["labels"]), dtype=np.float32) if isinstance(row["labels"], str) else np.array(row["labels"], dtype=np.float32)
        
        # Light augmentation with torchio
        if self.transform:
            subject = tio.Subject(image=tio.ScalarImage(tensor=image))
            transformed = self.transform(subject)
            image = transformed["image"].numpy()

        # Ensure 5D tensor for model (batch is added by DataLoader)
        image_tensor = torch.from_numpy(image).float()  # Shape: (1, D, H, W)
        labels_tensor = torch.from_numpy(labels)
        return image_tensor, labels_tensor

train_transforms = tio.Compose([
    tio.RandomFlip(axes=(0,), p=0.5),
    tio.RandomAffine(degrees=10, p=0.5),
    tio.ToCanonical()  # Ensure canonical orientation
])
val_transforms = tio.Compose([
    tio.ToCanonical()  # Ensure canonical orientation
])

# Model
class EfficientNet3D(nn.Module):
    def __init__(self, num_classes=14):
        super().__init__()
        self.backbone = monai.networks.nets.EfficientNetBN("efficientnet-b0", spatial_dims=3, in_channels=1, pretrained=False)
        self.pool = nn.AdaptiveAvgPool3d(1)  # Output: (batch_size, channels, 1, 1, 1)
        self.fc = nn.Linear(self.backbone._fc.in_features, num_classes)
        self.backbone._fc = nn.Identity()

    def forward(self, x):
        # Ensure input is 5D (batch_size, channels, depth, height, width)
        if x.dim() != 5:
            raise ValueError(f"Expected 5D input tensor (batch_size, channels, depth, height, width), got {x.shape}")
        
        x = self.backbone(x)
        # Check and adjust backbone output dimensions
        if x.dim() < 3:  # If backbone output is 2D, reshape to add dummy spatial dimensions
            x = x.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # Add 3 dummy dimensions
        x = self.pool(x)  # Ensure pooling works on at least 5D tensor
        x = x.view(x.size(0), -1)  # Flatten to (batch_size, features)
        x = self.fc(x)  # Return raw logits
        return x  # Removed torch.sigmoid()

# Training Loop
def train_one_epoch(model, loader, optimizer, scaler, criterion):
    model.train()
    losses = []
    for batch in tqdm(loader, desc="Training"):
        images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            # Debug: Print input shape to verify
            # print(f"Input shape to model: {images.shape}")
            outputs = model(images)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        losses.append(loss.item())
    return np.mean(losses)

def validate(model, loader, criterion):
    model.eval()
    preds, truths = [], []
    losses = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            images, labels = batch[0].to(DEVICE), batch[1].to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            losses.append(loss.item())
            preds.append(outputs.cpu().numpy())
            truths.append(labels.cpu().numpy())
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    
    # Calculate AUCs, handling single-class cases
    aucs = []
    for i in range(len(LABEL_COLS)):
        y_true = truths[:, i]
        y_pred = preds[:, i]
        if len(np.unique(y_true)) == 1:  # Check if only one class is present
            print(f"Warning: Only one class present in column {LABEL_COLS[i]}. AUC set to 0.5.")
            aucs.append(0.5)  # Default to 0.5 (no discrimination)
        else:
            aucs.append(roc_auc_score(y_true, y_pred))
    
    final_score = (aucs[-1] + np.mean(aucs[:-1])) / 2
    return np.mean(losses), final_score, aucs

# Main
def main():
    # Preprocess
    preprocess_dataset()

    # Data
    metadata_file = os.path.join(OUTPUT_DIR, "series_metadata.csv")
    metadata = pd.read_csv(metadata_file)
    metadata = metadata.sample(frac=1, random_state=42).reset_index(drop=True)
    train_idx = int(0.8 * len(metadata))
    train_data = metadata[:train_idx]
    val_data = metadata[train_idx:]


    train_dataset = CacheDataset(AneurysmDataset(metadata_file, train_transforms), cache_rate=0.1, num_workers=4)
    val_dataset = CacheDataset(AneurysmDataset(metadata_file, val_transforms), cache_rate=0.1, num_workers=4)
    train_loader = MonaiDataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, prefetch_factor=2)
    val_loader = MonaiDataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, prefetch_factor=2)

    # Model
    model = EfficientNet3D(num_classes=len(LABEL_COLS)).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()  # Changed to BCEWithLogitsLoss
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.cuda.amp.GradScaler()

    # Early stopping parameters
    patience = 5
    counter = 0
    best_score = 0
    best_epoch = 0

    for epoch in range(NUM_EPOCHS):
        train_loss = train_one_epoch(model, train_loader, optimizer, scaler, criterion)
        val_loss, val_score, aucs = validate(model, val_loader, criterion)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
              f"Train Loss: {train_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
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


# sid = "1.2.826.0.1.3680043.8.498.11208788596258922886794998326857227331"
# orig = train_df.loc[train_df["SeriesInstanceUID"] == sid, LABEL_COLS].values
# new = df.loc[df["image"].str.contains(sid), "labels"].values
# print("Original:", orig)
# print("Preprocessed:", new)



if __name__ == "__main__":
    main()


metadata_file = os.path.join(OUTPUT_DIR, "series_metadata.csv")
metadata = pd.read_csv(metadata_file)
metadata.head(5)


print(metadata['labels'].iloc[0])


print(metadata['image'].iloc[0])






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

