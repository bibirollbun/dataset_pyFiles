import gc
import torch
import kaggle_evaluation.rsna_inference_server

gc.collect()
torch.cuda.empty_cache()


#Imports
import pydicom
import os
from collections import defaultdict
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torch.nn.functional as F
import torch.optim as optim
from skimage.transform import resize
!pip install tqdm
from tqdm import tqdm
import random
from glob import glob


#Generate masks
import pandas as pd
import pydicom
from tqdm import tqdm
import cv2


os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


!pip install nibabel

import os
import nibabel as nib
import numpy as np

# Config
seg_root = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations"  # path to your .nii.gz segmentation folder
out_root = "/kaggle/working/masks_npy"
os.makedirs(out_root, exist_ok=True)

# Define label map for the 13 locations (use same index order as your train.csv)
LABEL_MAP = {
    "Left Infraclinoid Internal Carotid Artery": 0,
    "Right Infraclinoid Internal Carotid Artery": 1,
    "Left Supraclinoid Internal Carotid Artery": 2,
    "Right Supraclinoid Internal Carotid Artery": 3,
    "Left Middle Cerebral Artery": 4,
    "Right Middle Cerebral Artery": 5,
    "Anterior Communicating Artery": 6,
    "Left Anterior Cerebral Artery": 7,
    "Right Anterior Cerebral Artery": 8,
    "Left Posterior Communicating Artery": 9,
    "Right Posterior Communicating Artery": 10,
    "Basilar Tip": 11,
    "Other Posterior Circulation": 12,
}

for fname in os.listdir(seg_root):
    if not fname.endswith(".nii.gz"):
        continue

    series_uid = fname.replace(".nii.gz", "")
    path = os.path.join(seg_root, fname)

    # Load segmentation volume
    seg_nii = nib.load(path)
    seg_data = seg_nii.get_fdata()  # shape: [H, W, D]

    # Transpose to [D, H, W] (if needed)
    seg_data = np.transpose(seg_data, (2, 0, 1))

    # One-hot encode to [13, D, H, W]
    mask = np.zeros((13, *seg_data.shape), dtype=np.uint8)
    for label_name, class_idx in LABEL_MAP.items():
        mask[class_idx] = (seg_data == (class_idx + 1))  # Labels often start from 1

    np.save(os.path.join(out_root, f"{series_uid}.npy"), mask)



import os
import torch
import numpy as np
import pydicom
import torch.nn.functional as F
from tqdm import tqdm

def preprocess_and_save_all(raw_dicom_root, output_dir, scale_factor=0.25):
    os.makedirs(output_dir, exist_ok=True)
    uids = os.listdir(raw_dicom_root)

    for uid in tqdm(uids, desc="Preprocessing DICOM volumes"):
        try:
            series_path = os.path.join(raw_dicom_root, uid)
            dicom_files = sorted([os.path.join(series_path, f) for f in os.listdir(series_path)])

            # Read and stack slices
            slices = np.stack([pydicom.dcmread(p).pixel_array for p in dicom_files])
            volume = (slices.astype(np.float32) - np.mean(slices)) / np.std(slices)  # Normalize

            volume_tensor = torch.tensor(volume).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
            volume_downsampled = F.interpolate(volume_tensor, scale_factor=scale_factor, mode='trilinear', align_corners=False)
            volume_downsampled = volume_downsampled.squeeze(0).squeeze(0)  # [D, H, W]

            # Save
            torch.save(volume_downsampled, os.path.join(output_dir, f'{uid}.pt'))

        except Exception as e:
            print(f"Skipping {uid}: {e}")



class PatchedDicomDataset(Dataset):
    def __init__(self, df_meta, dicom_root, preprocessed_root, transforms=None, crop_size=None):
        self.df = df_meta
        self.root = dicom_root
        self.preprocessed_root = preprocessed_root
        self.transforms = transforms
        self.crop_size = crop_size  # (depth, height, width), e.g. (32, 64, 64)

    def __len__(self):
        return len(self.df)

    def random_crop_3d(self, volume, crop_size):
        C, D, H, W = volume.shape
        cd, ch, cw = crop_size
        if D < cd or H < ch or W < cw:
            # Instead of error, return original volume or center crop smaller crop
            print(f"Warning: crop size {crop_size} bigger than volume {volume.shape}, skipping crop.")
            return volume
        d1 = random.randint(0, D - cd)
        h1 = random.randint(0, H - ch)
        w1 = random.randint(0, W - cw)
    
        return volume[:, d1:d1+cd, h1:h1+ch, w1:w1+cw]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        uid = row['SeriesInstanceUID']
    
        volume_path = os.path.join(self.preprocessed_root, f"{uid}.pt")
        if not os.path.exists(volume_path):
            raise FileNotFoundError(f"Preprocessed volume not found: {volume_path}")
    
        volume_tensor = torch.load(volume_path).float()  # e.g. [1, 1, 26, 560, 560]
        
        # Remove extra dims if any (expect [C, D, H, W])
        while volume_tensor.dim() > 4:
            volume_tensor = volume_tensor.squeeze(0)
        if volume_tensor.dim() != 4:
            raise ValueError(f"Volume tensor shape invalid after squeezing: {volume_tensor.shape}")
    
        if self.crop_size is not None:
            volume_tensor = self.random_crop_3d(volume_tensor, self.crop_size)
    
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
            
            label = torch.tensor(row[LABEL_COLS].astype(float).values, dtype=torch.float32)

    
        mask_downsampled = None
        if 'mask_path' in row and isinstance(row['mask_path'], str) and os.path.exists(row['mask_path']):
            mask = torch.tensor(np.load(row['mask_path'])).unsqueeze(0).long()
            if self.crop_size is not None:
                mask_downsampled = self.random_crop_3d(mask, self.crop_size)
            else:
                mask_downsampled = mask
    
        return volume_tensor, label, mask_downsampled


class FilteredDataset(Dataset):
    def __init__(self, base_dataset, min_depth=2):
        self.valid_indices = []
        print("Filtering dataset for minimum depth:", min_depth)
        for idx in range(len(base_dataset)):
            try:
                volume, _, _ = base_dataset[idx]
                depth = volume.shape[1]  # shape is [1, D, H, W]
                if depth >= min_depth:
                    self.valid_indices.append(idx)
            except (IndexError, ValueError, AssertionError) as e:
                print(f"Skipping index {idx} due to data error: {e}")
        self.base = base_dataset

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        return self.base[self.valid_indices[idx]]


import pydicom
from tqdm import tqdm
import pickle

CACHE_PATH = "/kaggle/working/filtered_series_uids.pkl"
DICOM_ROOT = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"
MIN_DEPTH = 2

if os.path.exists(CACHE_PATH):
    print("ğŸ”„ Loading cached filtered SeriesInstanceUIDs...")
    with open(CACHE_PATH, "rb") as f:
        filtered_uids = pickle.load(f)
else:
    print("ğŸ”� Fast filtering using DICOM metadata only...")

    filtered_uids = []
    for uid in tqdm(os.listdir(DICOM_ROOT)):
        series_dir = os.path.join(DICOM_ROOT, uid)
        if not os.path.isdir(series_dir):
            continue
        files = sorted(os.listdir(series_dir))
        if len(files) < MIN_DEPTH:
            continue
        first_file = os.path.join(series_dir, files[0])
        try:
            dcm = pydicom.dcmread(first_file, stop_before_pixels=True)
            if hasattr(dcm, 'NumberOfFrames') and dcm.NumberOfFrames < MIN_DEPTH:
                continue
            filtered_uids.append(uid)
        except Exception as e:
            continue

    print(f"âœ… Filtered {len(filtered_uids)} valid SeriesInstanceUIDs.")
    with open(CACHE_PATH, "wb") as f:
        pickle.dump(filtered_uids, f)


pd.Series(filtered_uids, name="SeriesInstanceUID").to_csv("/kaggle/working/filtered_uids.csv", index=False)


df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
filtered_uids = pd.read_csv("/kaggle/working/filtered_uids.csv")["SeriesInstanceUID"].tolist()

df_filtered = df[df["SeriesInstanceUID"].isin(filtered_uids)].reset_index(drop=True)


subset_df = df.sample(n=10, random_state=42)
dicom_root = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"


preprocessed_root = "/kaggle/working/preprocessed"
os.makedirs(preprocessed_root, exist_ok=True)

for idx, row in tqdm(subset_df.iterrows(), total=len(subset_df)):
    uid = row['SeriesInstanceUID']
    paths = sorted([os.path.join(dicom_root, uid, f) for f in os.listdir(os.path.join(dicom_root, uid))])
    
    slices = np.stack([pydicom.dcmread(p).pixel_array for p in paths], axis=0)
    volume = (slices.astype(np.float32) - np.mean(slices)) / np.std(slices)
    
    volume_tensor = torch.tensor(volume).float().unsqueeze(0)  # shape [1, D, H, W]
    
    # Optional: downsample volume here if you want to save memory later
    
    save_path = os.path.join(preprocessed_root, f"{uid}.pt")
    torch.save(volume_tensor, save_path)


class ClassifierNet(nn.Module):
    def __init__(self, input_shape=(1, 32, 64, 64), num_classes=14):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool3d(2)
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(16, 14)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x  # [B, 14]

# Example usage:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clf = ClassifierNet(input_shape=(1, 32, 64, 64)).to(device)

# Test with dummy input
dummy_input = torch.randn(1, 1, 32, 64, 64).to(device)
output = clf(dummy_input)
print(output.shape)  # Should print: torch.Size([1, 1])


class Simple3DSegmentationNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(2)

        self.enc2 = nn.Sequential(
            nn.Conv3d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.up2 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        self.up1 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Conv3d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        
        self.out_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)          # [B,16,D,H,W]
        p1 = self.pool1(enc1)        # [B,16,D/2,H/2,W/2]
    
        enc2 = self.enc2(p1)         # [B,32,D/2,H/2,W/2]
        p2 = self.pool2(enc2)        # [B,32,D/4,H/4,W/4]
    
        bottleneck = self.bottleneck(p2)  # [B,64,D/4,H/4,W/4]
    
        up2 = self.up2(bottleneck)          # [B,32,D/2,H/2,W/2]
    
        # Resize enc2 to match up2 spatial dims before concatenation
        if up2.shape[2:] != enc2.shape[2:]:
            enc2 = F.interpolate(enc2, size=up2.shape[2:], mode='trilinear', align_corners=False)
    
        cat2 = torch.cat([up2, enc2], dim=1)  # skip connection
    
        dec2 = self.dec2(cat2)               # [B,32,D/2,H/2,W/2]
    
        up1 = self.up1(dec2)                 # [B,16,D,H,W]
    
        # Resize enc1 if needed before concat
        if up1.shape[2:] != enc1.shape[2:]:
            enc1 = F.interpolate(enc1, size=up1.shape[2:], mode='trilinear', align_corners=False)
    
        cat1 = torch.cat([up1, enc1], dim=1)
    
        dec1 = self.dec1(cat1)               # [B,16,D,H,W]
    
        out = self.out_conv(dec1)            # [B,out_channels,D,H,W]
    
        return out


from torch.utils.data import DataLoader
import torch.optim as optim
from torch.cuda.amp import GradScaler

dicom_root = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"
preprocessed_root = '/kaggle/working/preprocessed'

# Load metadata CSV
df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")

# Filter dataframe to keep only UIDs that have a .pt preprocessed file
preprocessed_files = set(f[:-3] for f in os.listdir(preprocessed_root) if f.endswith('.pt'))
df_filtered = df[df['SeriesInstanceUID'].isin(preprocessed_files)].reset_index(drop=True)
print(f"Filtered dataframe size: {len(df_filtered)}")

# Initialize dataset with filtered dataframe
dataset = PatchedDicomDataset(df_meta=df_filtered, dicom_root=dicom_root, crop_size=(32, 64, 64), preprocessed_root=preprocessed_root)

# DataLoader
def custom_collate(batch):
    # Filter None samples (whole sample)
    batch = [b for b in batch if b is not None]
    vols = torch.utils.data.dataloader.default_collate([b[0] for b in batch])
    labels = torch.utils.data.dataloader.default_collate([b[1] for b in batch])
    masks = []
    for b in batch:
        if b[2] is None:
            masks.append(torch.zeros_like(b[0], dtype=torch.long))  # or whatever shape you want
        else:
            masks.append(b[2])
    masks = torch.utils.data.dataloader.default_collate(masks)
    return vols, labels, masks

loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2, collate_fn=custom_collate)

clf = ClassifierNet(input_shape=(1, 32, 64, 64)).to(device)
seg = Simple3DSegmentationNet(in_channels=1, out_channels=2).to(device)

# Losses
classification_loss = nn.BCEWithLogitsLoss()
segmentation_loss = nn.CrossEntropyLoss()

# Models, loss, optimizer, scaler (your existing definitions)
optimizer = optim.Adam(list(clf.parameters()) + list(seg.parameters()), lr=1e-4)
scaler = GradScaler()

# Training loop
n_epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for epoch in range(n_epochs):
    running_loss = 0.0
    for vol, label, mask in tqdm(loader, desc=f"Training Epoch {epoch+1}", leave=True):
        vol = vol.to(device).float()
        label = label.to(device).float()
        if mask is not None:
            mask = mask.to(device).long()

        optimizer.zero_grad()

        with torch.amp.autocast(device_type='cuda'):
            pred = clf(vol)
            loss_c = classification_loss(pred, label)


            seg_pred = seg(vol)
            if mask.dim() == 4:
                mask = mask.unsqueeze(1)  # [B, 1, D, H, W]
            
            # Resize and compute loss
            mask_resized = F.interpolate(mask.float(), size=seg_pred.shape[2:], mode='trilinear', align_corners=False)
            loss_s = segmentation_loss(seg_pred, mask_resized.squeeze(1).long())

            loss = loss_c + loss_s

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.empty_cache()
        gc.collect()

        running_loss += loss.item()

    print(f"Epoch {epoch+1} Loss: {running_loss:.4f}")




df = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv")
df_test = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/kaggle_evaluation/test.csv")
print("Total SeriesInstanceUIDs:", df["SeriesInstanceUID"].nunique())


#save weights

torch.save(clf.state_dict(), 'classification_model.pth')
torch.save(seg.state_dict(), 'segmentation_model.pth')




#load weights
clf.load_state_dict(torch.load('classification_model.pth'))
seg.load_state_dict(torch.load('segmentation_model.pth'))
#clf.eval()
#seg.eval()



def preprocess_and_save_test_series(uid, dicom_root, save_root):

    
    series_path = os.path.join(dicom_root, uid)
    dicom_files = sorted(
        [os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith('.dcm')],
        key=lambda x: int(pydicom.dcmread(x).InstanceNumber)
    )
    slices = [pydicom.dcmread(f).pixel_array for f in dicom_files]
    if len(slices) == 0:
        print(f"No dicom slices found for {uid}")
        return False
    
    volume = np.stack(slices).astype(np.float32)
    volume = (volume - volume.min()) / (volume.max() - volume.min() + 1e-5)
    volume_tensor = torch.tensor(volume).unsqueeze(0)  # Shape: [1, D, H, W]
    
    # Save tensor
    os.makedirs(save_root, exist_ok=True)
    torch.save(volume_tensor, os.path.join(save_root, f"{uid}.pt"))
    print(f"Saved preprocessed tensor for {uid}")
    return True


# Preprocess all test UIDs
for uid in df_test["SeriesInstanceUID"]:
    preprocess_and_save_test_series(uid, dicom_root="/kaggle/input/rsna-intracranial-aneurysm-detection/kaggle_evaluation/series", save_root=preprocessed_root)


#def run_model_on_uid_debug(uid, model, device, preprocessed_root, crop_size=(32, 64, 64)):
#    import torch.nn.functional as F
#    import os
#
 #   pt_path = os.path.join(preprocessed_root, f"{uid}.pt")
#
 #   if not os.path.isfile(pt_path):
  #      print(f"Warning: Preprocessed file not found for UID {uid}, returning 0.5")
   #     return 0.5
#
#    volume = torch.load(pt_path)

#    if len(volume.shape) == 4:
 #       volume = volume.unsqueeze(0)
#    elif len(volume.shape) == 3:
#        volume = volume.unsqueeze(0).unsqueeze(0)
#    else:
      #  raise ValueError(f"Unexpected volume shape: {volume.shape}")

#    _, C, D, H, W = volume.shape
#    td, th, tw = crop_size

 #   pad_d = max(td - D, 0)
#    pad_h = max(th - H, 0)
#    pad_w = max(tw - W, 0)
#    volume = F.pad(volume, 
#                   [pad_w // 2, pad_w - pad_w // 2,
#                    pad_h // 2, pad_h - pad_h // 2,
#                    pad_d // 2, pad_d - pad_d // 2])

#    _, _, D, H, W = volume.shape
#    start_d = (D - td) // 2
#    start_h = (H - th) // 2
#    start_w = (W - tw) // 2
#    volume = volume[:, :, start_d:start_d+td, start_h:start_h+th, start_w:start_w+tw]

#    volume = volume.to(device).float()

#    model.eval()
#    with torch.no_grad():
 #       logits = model(volume)
#        prob = torch.sigmoid(logits).item()

#    print(f"UID: {uid}, prediction: {prob:.4f}")

#    return prob

#df_test["label"] = df_test["SeriesInstanceUID"].apply(
#    lambda uid: run_model_on_uid_debug(uid, clf, device, preprocessed_root)
#)


#def run_model_on_uid(uid, model, device, preprocessed_root, crop_size=(32, 64, 64)):
#    pt_path = os.path.join(preprocessed_root, f"{uid}.pt")
#    if not os.path.isfile(pt_path):
#        print(f"Warning: Preprocessed file not found for UID {uid}, returning 0.5")
#        return 0.5

#    volume = torch.load(pt_path)

#    if len(volume.shape) == 4:
#        volume = volume.unsqueeze(0)
#    elif len(volume.shape) == 3:
 #       volume = volume.unsqueeze(0).unsqueeze(0)
#    else:
 #       raise ValueError(f"Unexpected volume shape: {volume.shape}")

#    _, C, D, H, W = volume.shape
#    td, th, tw = crop_size

#    pad_d = max(td - D, 0)
#    pad_h = max(th - H, 0)
#    pad_w = max(tw - W, 0)
#    volume = F.pad(volume,
#                   [pad_w // 2, pad_w - pad_w // 2,
#                    pad_h // 2, pad_h - pad_h // 2,
#                    pad_d // 2, pad_d - pad_d // 2])

#    _, _, D, H, W = volume.shape
#    start_d = (D - td) // 2
#    start_h = (H - th) // 2
#    start_w = (W - tw) // 2
#    volume = volume[:, :, start_d:start_d + td, start_h:start_h + th, start_w:start_w + tw]

#    volume = volume.to(device).float()

#    model.eval()
#    with torch.no_grad():
#        logits = model(volume)
#        prob = torch.sigmoid(logits).item()

#    print(f"UID: {uid}, prediction: {prob:.4f}")
 #   return prob


# Load test CSV with SeriesInstanceUID
#df_test = pd.read_csv("/kaggle/input/rsna-intracranial-aneurysm-detection/kaggle_evaluation/test.csv")

# Generate predictions for all UIDs in test set
#df_test["label"] = df_test["SeriesInstanceUID"].apply(
#    lambda uid: run_model_on_uid(uid, clf, device, preprocessed_root)
#)

# Save submission file as submission.parquet (competition required format)
#df_test[["SeriesInstanceUID", "label"]].to_parquet("submission.parquet", index=False)


import os
import torch
import polars as pl
import numpy as np
import pydicom
import shutil
from torchvision import transforms

# Your label columns (exactly as required)
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

ID_COL = "SeriesInstanceUID"

# Load model weights if saved
# clf.load_state_dict(torch.load("path/to/classifier_weights.pth"))  # <-- OPTIONAL

def preprocess_volume(series_path):
    """Loads DICOMs and returns a preprocessed volume tensor."""
    dcm_files = sorted([os.path.join(series_path, f) for f in os.listdir(series_path) if f.endswith(".dcm")])
    slices = [pydicom.dcmread(f).pixel_array.astype(np.float32) for f in dcm_files]
    volume = np.stack(slices, axis=0)  # Shape: [D, H, W]

    # Normalize
    volume = (volume - np.mean(volume)) / (np.std(volume) + 1e-5)

    # Resize or pad to expected shape (32, 64, 64)
    volume = resize_or_pad(volume, (32, 64, 64))

    # Add channel dimension
    tensor = torch.tensor(volume).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    return tensor

def resize_or_pad(vol, target_shape):
    """Pads or center-crops to match target_shape=(D,H,W)."""
    d, h, w = vol.shape
    td, th, tw = target_shape

    # Pad or crop depth
    if d < td:
        pad_d = (td - d) // 2
        vol = np.pad(vol, ((pad_d, td - d - pad_d), (0,0), (0,0)), mode='constant')
    elif d > td:
        start_d = (d - td) // 2
        vol = vol[start_d:start_d+td]

    # Pad or crop height
    if h < th:
        pad_h = (th - h) // 2
        vol = np.pad(vol, ((0,0), (pad_h, th - h - pad_h), (0,0)), mode='constant')
    elif h > th:
        start_h = (h - th) // 2
        vol = vol[:, start_h:start_h+th]

    # Pad or crop width
    if w < tw:
        pad_w = (tw - w) // 2
        vol = np.pad(vol, ((0,0), (0,0), (pad_w, tw - w - pad_w)), mode='constant')
    elif w > tw:
        start_w = (w - tw) // 2
        vol = vol[:, :, start_w:start_w+tw]

    return vol

# Your trained classifier
clf.eval()

@torch.no_grad()
def predict(series_path: str) -> pl.DataFrame:
    try:
        series_id = os.path.basename(series_path)
        volume_tensor = preprocess_volume(series_path).to(device)
    
        output = clf(volume_tensor)
        probs = torch.sigmoid(output).cpu().numpy().flatten().tolist()
    
        if probs is None or len(probs) != 14 or any(np.isnan(probs)):
            print(f"Invalid prediction for {series_id}")
            probs = [0.5] * 14
    
        # Here we output the same value for all 14 columns (for demo purposes)
        # You can extend your model to predict all LABEL_COLS if you train for them
        pred_row = [series_id] + probs
        result = pl.DataFrame([pred_row], schema=[ID_COL] + LABEL_COLS)
    
        #shutil.rmtree('/kaggle/shared', ignore_errors=True)
        return result
    except Exception as e:
        print(f"Error processing {series_path}: {e}")
        raise e    




for fname in os.listdir("/kaggle/working"):
    if fname != "submission.parquet":
        path = os.path.join("/kaggle/working", fname)
        if os.path.isfile(path):
            os.remove(path)
        elif os.path.isdir(path):
            shutil.rmtree(path)

shutil.rmtree('/kaggle/shared', ignore_errors=True)



inference_server = kaggle_evaluation.rsna_inference_server.RSNAInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway()
    display(pl.read_parquet('/kaggle/working/submission.parquet'))

