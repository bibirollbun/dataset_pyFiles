!pip install -q nibabel dicom2nifti pydicom


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score

import nibabel as nib
import nibabel.orientations as nio
import pydicom
from scipy.ndimage import zoom, binary_dilation, rotate
from scipy.ndimage.filters import gaussian_filter
import time


class CFG:
    """Global configuration"""
    # Paths
    data_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection'
    train_csv = f'{data_dir}/train.csv'
    localizers_csv = f'{data_dir}/train_localizers.csv'
    series_dir = f'{data_dir}/series'
    segmentations_dir = f'{data_dir}/segmentations'
    
    work_dir = '/kaggle/working'
    nifti_cache_dir = f'{work_dir}/nifti_cache'
    roi_crops_dir = f'{work_dir}/roi_crops'
    
    # Subset for testing
    use_subset = True
    n_subset_samples = 1000
    
    # Model
    img_size = (128, 128, 128)
    base_channels = 16
    num_vessel_classes = 14  # 0=bg + 13 vessels
    num_aneurysm_classes = 2
    num_location_classes = 13
    num_modality_classes = 4
    
    # Training
    batch_size = 1
    num_epochs = 100
    learning_rate = 1e-3
    weight_decay = 3e-5
    train_split = 0.7
    val_split = 0.15
    
    # Loss weights
    vessel_seg_weight = 1.0
    aneurysm_seg_weight = 2.0
    binary_cls_weight = 3.0
    location_cls_weight = 2.5
    coord_reg_weight = 1.5
    modality_cls_weight = 0.2
    
    # Augmentation
    use_flip_augmentation = True
    flip_probability = 0.5
    use_rotation = True
    use_scaling = True
    use_noise = True
    
    # TTA
    use_tta = True
    tta_n_flips = 8
    
    # Optimization
    mixed_precision = True
    
    # Other
    seed = 42
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


np.random.seed(CFG.seed)
torch.manual_seed(CFG.seed)


LOCATION_NAMES = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral lo Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation'
]

LEFT_RIGHT_VESSEL_PAIRS = {
    3: 4, 4: 3,    # PComm
    5: 6, 6: 5,    # Infraclinoid ICA
    7: 8, 8: 7,    # Supraclinoid ICA
    9: 10, 10: 9,  # MCA
    11: 12, 12: 11 # ACA
}

LEFT_RIGHT_LOCATION_PAIRS = {
    0: 1, 1: 0,    # Infraclinoid ICA
    2: 3, 3: 2,    # Supraclinoid ICA
    4: 5, 5: 4,    # MCA
    7: 8, 8: 7,    # ACA
    9: 10, 10: 9   # PComm
}

MODALITY_MAP = {'CTA': 0, 'MRA': 1, 'MRI_T1': 2, 'MRI_T2': 3}


os.makedirs(CFG.nifti_cache_dir, exist_ok=True)
os.makedirs(CFG.roi_crops_dir, exist_ok=True)

train_df = pd.read_csv(CFG.train_csv)
localizers_df = pd.read_csv(CFG.localizers_csv)


def reorient_to_lps(nifti_img):
    """Reorient NIfTI image to LPS orientation"""
    orig_ornt = nio.io_orientation(nifti_img.affine)
    lps_ornt = nio.axcodes2ornt(('L', 'P', 'S'))
    transform = nio.ornt_transform(orig_ornt, lps_ornt)
    data_reoriented = nio.apply_orientation(nifti_img.get_fdata(), transform)
    affine_reoriented = nifti_img.affine @ nio.inv_ornt_aff(transform, nifti_img.shape)
    return nib.Nifti1Image(data_reoriented, affine_reoriented)


def load_dicom_to_nifti(series_uid, cache_dir):
    """Load DICOM series and convert to NIfTI with caching"""
    cache_path = Path(cache_dir) / f"{series_uid}.nii.gz"
    
    if cache_path.exists():
        return nib.load(str(cache_path))
    
    series_path = Path(CFG.series_dir) / series_uid
    if not series_path.exists():
        return None
    
    dicom_files = sorted(list(series_path.glob('*.dcm')))
    if len(dicom_files) == 0:
        return None
    
    slices = []
    for dcm_file in dicom_files:
        try:
            dcm = pydicom.dcmread(str(dcm_file))
            _ = dcm.pixel_array
            slices.append(dcm)
        except:
            continue
    
    if len(slices) == 0:
        return None
    
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        try:
            slices.sort(key=lambda x: int(x.InstanceNumber))
        except:
            pass
    
    first_array = slices[0].pixel_array
    if len(first_array.shape) == 3:
        first_array = first_array[0]
    
    img_shape = first_array.shape
    volume = np.zeros((img_shape[0], img_shape[1], len(slices)), dtype=np.float32)
    
    for i, dcm_slice in enumerate(slices):
        try:
            arr = dcm_slice.pixel_array.astype(np.float32)
            if len(arr.shape) == 3:
                arr = arr[0]
            if arr.shape != img_shape:
                arr = zoom(arr, (img_shape[0]/arr.shape[0], img_shape[1]/arr.shape[1]), order=1)
            volume[:, :, i] = arr
        except:
            volume[:, :, i] = np.zeros(img_shape, dtype=np.float32)
    
    affine = np.eye(4)
    nifti_img = nib.Nifti1Image(volume, affine)
    nifti_img = reorient_to_lps(nifti_img)
    
    nib.save(nifti_img, str(cache_path))
    return nifti_img


def load_vessel_segmentation(series_uid, segmentations_dir):
    """Load vessel segmentation from segmentations folder"""
    seg_path = Path(segmentations_dir) / f"{series_uid}_cowseg.nii"
    
    if seg_path.exists():
        seg_nii = nib.load(str(seg_path))
        seg_nii = reorient_to_lps(seg_nii)
        return seg_nii.get_fdata().astype(np.int64)
    
    return None


def create_bbox_from_mask(mask, padding_ratio=0.15):
    """Create 3D bounding box from binary/multi-class mask"""
    coords = np.where(mask > 0)
    
    if len(coords[0]) == 0:
        return None
    
    x_min, x_max = coords[0].min(), coords[0].max()
    y_min, y_max = coords[1].min(), coords[1].max()
    z_min, z_max = coords[2].min(), coords[2].max()
    
    x_range = max(x_max - x_min, 1)  # Mínimo 1
    y_range = max(y_max - y_min, 1)
    z_range = max(z_max - z_min, 1)
    
    x_pad = int(x_range * padding_ratio)
    y_pad = int(y_range * padding_ratio)
    z_pad = int(z_range * padding_ratio)
    
    x_min = max(0, x_min - x_pad)
    x_max = min(mask.shape[0], x_max + x_pad + 1)
    y_min = max(0, y_min - y_pad)
    y_max = min(mask.shape[1], y_max + y_pad + 1)
    z_min = max(0, z_min - z_pad)
    z_max = min(mask.shape[2], z_max + z_pad + 1)
    
    # Validar bbox
    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
        return None
    
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def create_bbox_from_localizers(localizers_df_subset, volume_shape):
    """Create bbox from localizer coordinates when no segmentation available"""
    if len(localizers_df_subset) == 0:
        return None
    
    all_coords = []
    for _, row in localizers_df_subset.iterrows():
        try:
            coords = eval(row['coordinates'])
            if isinstance(coords, dict):
                x = coords.get('x', volume_shape[0] / 2)
                y = coords.get('y', volume_shape[1] / 2)
                z = coords.get('z', volume_shape[2] / 2)
                all_coords.append([x, y, z])
            else:
                if len(coords) >= 2:
                    x = coords[0]
                    y = coords[1]
                    z = coords[2] if len(coords) > 2 else volume_shape[2] / 2
                    all_coords.append([x, y, z])
        except:
            continue
    
    if len(all_coords) == 0:
        return None
    
    all_coords = np.array(all_coords)
    
    x_center = all_coords[:, 0].mean()
    y_center = all_coords[:, 1].mean()
    z_center = all_coords[:, 2].mean()
    
    box_size = 180
    half_size = box_size // 2
    
    x_min = max(0, int(x_center - half_size))
    x_max = min(volume_shape[0], int(x_center + half_size))
    y_min = max(0, int(y_center - half_size))
    y_max = min(volume_shape[1], int(y_center + half_size))
    z_min = max(0, int(z_center - half_size))
    z_max = min(volume_shape[2], int(z_center + half_size))
    
    # Validar bbox
    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
        return None
    
    return (x_min, x_max, y_min, y_max, z_min, z_max)


def extract_roi(volume, bbox):
    """Extract ROI crop from volume using bbox"""
    if bbox is None:
        return volume
    
    x_min, x_max, y_min, y_max, z_min, z_max = bbox
    
    # Validar bbox
    if x_min >= x_max or y_min >= y_max or z_min >= z_max:
        return volume
    
    roi = volume[x_min:x_max, y_min:y_max, z_min:z_max]
    
    # Verificar que ROI no esté vacío
    if roi.size == 0 or 0 in roi.shape:
        return volume
    
    return roi


def resize_volume(volume, target_size):
    """Resize volume to target size"""
    # Verificar volumen válido
    if volume.size == 0 or 0 in volume.shape:
        return np.zeros(target_size, dtype=volume.dtype)
    
    zoom_factors = [target_size[i] / volume.shape[i] for i in range(3)]
    resized = zoom(volume, zoom_factors, order=1)
    
    resized = resized[:target_size[0], :target_size[1], :target_size[2]]
    
    if resized.shape != target_size:
        padded = np.zeros(target_size, dtype=resized.dtype)
        padded[:resized.shape[0], :resized.shape[1], :resized.shape[2]] = resized
        resized = padded
    
    return resized


def process_stage1_roi_extraction(series_uid, localizers_df, target_size):
    """Complete Stage 1 pipeline: load, extract ROI, resize"""
    nifti_img = load_dicom_to_nifti(series_uid, CFG.nifti_cache_dir)
    if nifti_img is None:
        return None, None, None
    
    volume = nifti_img.get_fdata().astype(np.float32)
    
    # Verificar volumen válido
    if volume.size == 0 or 0 in volume.shape:
        return None, None, None
    
    vessel_seg = load_vessel_segmentation(series_uid, CFG.segmentations_dir)
    
    if vessel_seg is not None and vessel_seg.size > 0:
        bbox = create_bbox_from_mask(vessel_seg)
        if bbox is not None:
            vessel_seg_roi = extract_roi(vessel_seg, bbox)
            vessel_seg_resized = resize_volume(vessel_seg_roi, target_size).astype(np.int64)
        else:
            vessel_seg_resized = None
    else:
        loc_subset = localizers_df[localizers_df['SeriesInstanceUID'] == series_uid]
        bbox = create_bbox_from_localizers(loc_subset, volume.shape)
        vessel_seg_resized = None
    
    volume_roi = extract_roi(volume, bbox)
    
    # Verificar que ROI sea válido
    if volume_roi.size == 0 or 0 in volume_roi.shape:
        volume_resized = resize_volume(volume, target_size)
    else:
        volume_resized = resize_volume(volume_roi, target_size)
    
    return volume_resized, vessel_seg_resized, bbox


def visualize_stage1_pipeline(series_uid, localizers_df):
    """Visualize Stage 1 ROI extraction pipeline"""
    volume, vessel_seg, bbox = process_stage1_roi_extraction(
        series_uid, localizers_df, CFG.img_size
    )
    
    if volume is None:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    slices_idx = [volume.shape[2]//4, volume.shape[2]//2, 3*volume.shape[2]//4]
    
    for i, slice_idx in enumerate(slices_idx):
        axes[0, i].imshow(volume[:, :, slice_idx], cmap='gray')
        axes[0, i].set_title(f'Volume Slice {slice_idx}')
        axes[0, i].axis('off')
        
        if vessel_seg is not None:
            axes[1, i].imshow(volume[:, :, slice_idx], cmap='gray', alpha=0.7)
            axes[1, i].imshow(vessel_seg[:, :, slice_idx], cmap='jet', alpha=0.3, vmin=0, vmax=13)
            axes[1, i].set_title(f'Vessel Seg Overlay {slice_idx}')
        else:
            axes[1, i].imshow(volume[:, :, slice_idx], cmap='gray')
            axes[1, i].set_title(f'No Segmentation {slice_idx}')
        axes[1, i].axis('off')
    
    plt.suptitle(f'Stage 1 ROI: {series_uid[:30]}...', fontsize=14)
    plt.tight_layout()
    plt.show()


def generate_pseudo_vessel_mask(volume, target_size):
    """Generate pseudo vessel segmentation when real segmentation not available"""
    threshold = np.percentile(volume, 85)
    binary_mask = (volume > threshold).astype(np.int64)
    
    struct = np.ones((3, 3, 3))
    binary_mask = binary_dilation(binary_mask, structure=struct, iterations=2).astype(np.int64)
    
    vessel_mask = np.zeros_like(binary_mask, dtype=np.int64)
    
    vessel_coords = np.where(binary_mask > 0)
    if len(vessel_coords[0]) > 0:
        weights = np.array([0.05, 0.08, 0.08, 0.08, 0.08, 0.12, 0.12, 0.12, 0.12, 0.05, 0.05, 0.03, 0.02])
        labels = np.random.choice(13, size=len(vessel_coords[0]), p=weights) + 1
        vessel_mask[vessel_coords] = labels
    
    return vessel_mask


def create_aneurysm_mask(series_uid, localizers_df, target_size, radius=12):
    """Create aneurysm segmentation mask from localizer coordinates"""
    mask = np.zeros(target_size, dtype=np.int64)
    
    loc_subset = localizers_df[localizers_df['SeriesInstanceUID'] == series_uid]
    
    for _, row in loc_subset.iterrows():
        try:
            coords = eval(row['coordinates'])
            if len(coords) >= 3:
                x = int(coords[0] * target_size[0] / 512)
                y = int(coords[1] * target_size[1] / 512)
                z = int(coords[2] * target_size[2] / 512)
                
                x = np.clip(x, 0, target_size[0]-1)
                y = np.clip(y, 0, target_size[1]-1)
                z = np.clip(z, 0, target_size[2]-1)
                
                for i in range(max(0, x-radius), min(target_size[0], x+radius)):
                    for j in range(max(0, y-radius), min(target_size[1], y+radius)):
                        for k in range(max(0, z-radius), min(target_size[2], z+radius)):
                            dist = np.sqrt((i-x)**2 + (j-y)**2 + (k-z)**2)
                            if dist <= radius:
                                mask[i, j, k] = 1
        except:
            continue
    
    return mask


def create_aneurysm_heatmap(series_uid, localizers_df, target_size, sigma=15):
    """Create gaussian heatmap centered on aneurysm for loss weighting"""
    heatmap = np.zeros(target_size, dtype=np.float32)
    
    loc_subset = localizers_df[localizers_df['SeriesInstanceUID'] == series_uid]
    
    for _, row in loc_subset.iterrows():
        try:
            coords = eval(row['coordinates'])
            if len(coords) >= 3:
                x = int(coords[0] * target_size[0] / 512)
                y = int(coords[1] * target_size[1] / 512)
                z = int(coords[2] * target_size[2] / 512)
                
                x = np.clip(x, 0, target_size[0]-1)
                y = np.clip(y, 0, target_size[1]-1)
                z = np.clip(z, 0, target_size[2]-1)
                
                point_heatmap = np.zeros(target_size, dtype=np.float32)
                point_heatmap[x, y, z] = 1.0
                point_heatmap = gaussian_filter(point_heatmap, sigma=sigma)
                heatmap += point_heatmap
        except:
            continue
    
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()
    
    heatmap = heatmap * 5.0 + 1.0
    
    return heatmap


def extract_location_labels(row):
    """Extract 13-location binary labels from dataframe row"""
    labels = np.zeros(13, dtype=np.float32)
    for i, loc_name in enumerate(LOCATION_NAMES):
        labels[i] = float(row.get(loc_name, 0))
    return labels


def swap_vessel_labels(mask):
    """Swap left-right vessel labels after horizontal flip"""
    swapped = mask.copy()
    for left_label, right_label in LEFT_RIGHT_VESSEL_PAIRS.items():
        swapped[mask == left_label] = right_label
    return swapped


def swap_location_labels(labels):
    """Swap left-right location labels after horizontal flip"""
    swapped = labels.copy()
    for left_idx, right_idx in LEFT_RIGHT_LOCATION_PAIRS.items():
        swapped[left_idx] = labels[right_idx]
        swapped[right_idx] = labels[left_idx]
    return swapped


class AneurysmDataset(Dataset):
    """Multi-task dataset for aneurysm detection"""
    
    def __init__(self, df, localizers_df, target_size, augment):
        self.df = df.reset_index(drop=True)
        self.localizers_df = localizers_df
        self.target_size = target_size
        self.augment = augment
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        series_uid = row['SeriesInstanceUID']
        
        volume, vessel_seg, _ = process_stage1_roi_extraction(
            series_uid, self.localizers_df, self.target_size
        )
        
        if volume is None:
            volume = np.random.randn(*self.target_size).astype(np.float32)
        
        p1, p99 = np.percentile(volume, 1), np.percentile(volume, 99)
        volume = np.clip(volume, p1, p99)
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            volume = (volume - v_min) / (v_max - v_min)
        
        if vessel_seg is None:
            vessel_seg = generate_pseudo_vessel_mask(volume, self.target_size)
        
        aneurysm_mask = create_aneurysm_mask(series_uid, self.localizers_df, self.target_size)
        aneurysm_heatmap = create_aneurysm_heatmap(series_uid, self.localizers_df, self.target_size)
        
        location_labels = extract_location_labels(row)
        binary_label = float(row['Aneurysm Present'])
        
        modality = row.get('Modality', 'CTA')
        if 'T1' in modality:
            modality_label = MODALITY_MAP['MRI_T1']
        elif 'T2' in modality:
            modality_label = MODALITY_MAP['MRI_T2']
        else:
            modality_label = MODALITY_MAP.get(modality, 0)
        
        if self.augment:
            if CFG.use_flip_augmentation and np.random.random() > 0.5:
                volume = np.flip(volume, axis=0).copy()
                vessel_seg = swap_vessel_labels(np.flip(vessel_seg, axis=0).copy())
                aneurysm_mask = np.flip(aneurysm_mask, axis=0).copy()
                aneurysm_heatmap = np.flip(aneurysm_heatmap, axis=0).copy()
                location_labels = swap_location_labels(location_labels)
            
            if CFG.use_rotation and np.random.random() > 0.7:
                angle = np.random.uniform(-15, 15)
                axes = np.random.choice([0, 1, 2], size=2, replace=False)
                volume = rotate(volume, angle, axes=axes, reshape=False, order=1)
                vessel_seg = rotate(vessel_seg, angle, axes=axes, reshape=False, order=0)
                aneurysm_mask = rotate(aneurysm_mask, angle, axes=axes, reshape=False, order=0)
            
            if CFG.use_noise and np.random.random() > 0.7:
                volume = volume + np.random.randn(*volume.shape) * 0.05
                volume = np.clip(volume, 0, 1)
        
        volume = torch.from_numpy(volume).unsqueeze(0).float()
        vessel_seg = torch.from_numpy(vessel_seg).long()
        aneurysm_mask = torch.from_numpy(aneurysm_mask).long()
        aneurysm_heatmap = torch.from_numpy(aneurysm_heatmap).float()
        location_labels = torch.from_numpy(location_labels).float()
        binary_label = torch.tensor([binary_label], dtype=torch.float32)
        modality_label = torch.tensor(modality_label, dtype=torch.long)
        
        return {
            'image': volume,
            'vessel_mask': vessel_seg,
            'aneurysm_mask': aneurysm_mask,
            'aneurysm_heatmap': aneurysm_heatmap,
            'location_labels': location_labels,
            'binary_label': binary_label,
            'modality_label': modality_label,
            'series_uid': series_uid
        }


class CrossAttentionPooling(nn.Module):
    """Cross-attention pooling for global feature aggregation"""
    
    def __init__(self, dim, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.query = nn.Parameter(torch.randn(1, 1, dim))
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.proj = nn.Linear(dim, dim)
    
    def forward(self, x):
        B, C, D, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        
        q = self.query.expand(B, -1, -1)
        kv = self.kv(x_flat).reshape(B, -1, 2, self.num_heads, C // self.num_heads)
        k, v = kv.unbind(2)
        
        q = q.reshape(B, 1, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, 1, C)
        out = self.proj(out)
        
        return out.squeeze(1)


class ResidualBlock3D(nn.Module):
    """3D Residual block"""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class DecoderBlock3D(nn.Module):
    """3D Decoder block with skip connections"""
    
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv = nn.ConvTranspose3d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels // 2 + skip_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class nnXNetLite(nn.Module):
    """Lightweight nnXNet-inspired multi-task architecture"""
    
    def __init__(self, in_channels, num_vessel_classes, num_aneurysm_classes,
                 num_location_classes, num_modality_classes, base_channels):
        super().__init__()
        
        self.enc1 = ResidualBlock3D(in_channels, base_channels)
        self.enc2 = ResidualBlock3D(base_channels, base_channels * 2, stride=2)
        self.enc3 = ResidualBlock3D(base_channels * 2, base_channels * 4, stride=2)
        self.enc4 = ResidualBlock3D(base_channels * 4, base_channels * 8, stride=2)
        
        self.bottleneck = ResidualBlock3D(base_channels * 8, base_channels * 16, stride=2)
        
        self.cross_attn = CrossAttentionPooling(base_channels * 16)
        
        self.dec4 = DecoderBlock3D(base_channels * 16, base_channels * 8, base_channels * 8)
        self.dec3 = DecoderBlock3D(base_channels * 8, base_channels * 4, base_channels * 4)
        self.dec2 = DecoderBlock3D(base_channels * 4, base_channels * 2, base_channels * 2)
        self.dec1 = DecoderBlock3D(base_channels * 2, base_channels, base_channels)
        
        self.vessel_seg_head = nn.Conv3d(base_channels, num_vessel_classes, 1)
        self.aneurysm_seg_head = nn.Conv3d(base_channels, num_aneurysm_classes, 1)
        
        self.binary_cls_head = nn.Sequential(
            nn.Linear(base_channels * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
        
        self.location_cls_head = nn.Sequential(
            nn.Linear(base_channels * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_location_classes)
        )
        
        self.coord_reg_head = nn.Sequential(
            nn.Linear(base_channels * 16, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )
        
        self.modality_cls_head = nn.Sequential(
            nn.Linear(base_channels * 16, 64),
            nn.ReLU(),
            nn.Linear(64, num_modality_classes)
        )
    
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        
        bottleneck = self.bottleneck(e4)
        
        global_feat = self.cross_attn(bottleneck)
        
        d4 = self.dec4(bottleneck, e4)
        d3 = self.dec3(d4, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        
        vessel_seg = self.vessel_seg_head(d1)
        aneurysm_seg = self.aneurysm_seg_head(d1)
        
        binary_cls = self.binary_cls_head(global_feat)
        location_cls = self.location_cls_head(global_feat)
        coord_reg = self.coord_reg_head(global_feat)
        modality_cls = self.modality_cls_head(global_feat)
        
        return {
            'vessel_seg': vessel_seg,
            'aneurysm_seg': aneurysm_seg,
            'binary_cls': binary_cls,
            'location_cls': location_cls,
            'coord_reg': coord_reg,
            'modality_cls': modality_cls
        }


class DiceLoss(nn.Module):
    """Dice loss for segmentation"""
    
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.shape[1])
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3).float()
        
        intersection = (pred * target_one_hot).sum(dim=(2, 3, 4))
        union = pred.sum(dim=(2, 3, 4)) + target_one_hot.sum(dim=(2, 3, 4))
        
        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class HeatmapWeightedCE(nn.Module):
    """Cross-entropy with spatial heatmap weighting"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, heatmap):
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        weighted_loss = ce_loss * heatmap
        return weighted_loss.mean()


class MultiTaskLoss(nn.Module):
    def __init__(self, vessel_seg_weight, aneurysm_seg_weight, binary_cls_weight,
                 location_cls_weight, coord_reg_weight, modality_cls_weight):
        super().__init__()
        self.vessel_seg_weight = vessel_seg_weight
        self.aneurysm_seg_weight = aneurysm_seg_weight
        self.binary_cls_weight = binary_cls_weight
        self.location_cls_weight = location_cls_weight
        self.coord_reg_weight = coord_reg_weight
        self.modality_cls_weight = modality_cls_weight
        
        self.dice_loss = DiceLoss()
        self.heatmap_ce = HeatmapWeightedCE()
    
    def forward(self, outputs, targets):
        losses = {}
        
        vessel_ce = F.cross_entropy(outputs['vessel_seg'], targets['vessel_mask'])
        vessel_dice = self.dice_loss(outputs['vessel_seg'], targets['vessel_mask'])
        losses['vessel_seg'] = (vessel_ce + vessel_dice) * self.vessel_seg_weight
        
        aneurysm_ce = self.heatmap_ce(
            outputs['aneurysm_seg'],
            targets['aneurysm_mask'],
            targets['aneurysm_heatmap']
        )
        aneurysm_dice = self.dice_loss(outputs['aneurysm_seg'], targets['aneurysm_mask'])
        losses['aneurysm_seg'] = (aneurysm_ce + aneurysm_dice) * self.aneurysm_seg_weight
        
        losses['binary_cls'] = F.binary_cross_entropy_with_logits(
            outputs['binary_cls'],
            targets['binary_label']
        ) * self.binary_cls_weight
        
        losses['location_cls'] = F.binary_cross_entropy_with_logits(
            outputs['location_cls'],
            targets['location_labels']
        ) * self.location_cls_weight
        
        # FIX: Solo calcular coord loss si hay aneurismas reales
        has_aneurysm = targets['binary_label'] > 0.5
        if has_aneurysm.any():
            aneurysm_coords = []
            for i in range(len(targets['aneurysm_mask'])):
                mask = targets['aneurysm_mask'][i].cpu().numpy()
                coords_idx = np.where(mask > 0)
                
                # FIX: Verificar si hay píxeles antes de calcular mean
                if len(coords_idx[0]) > 0:
                    coords = np.array([coords_idx[0].mean(), coords_idx[1].mean(), coords_idx[2].mean()])
                else:
                    # Default al centro si máscara vacía
                    coords = np.array([mask.shape[0]//2, mask.shape[1]//2, mask.shape[2]//2], dtype=np.float32)
                
                aneurysm_coords.append(coords)
            
            aneurysm_coords = torch.tensor(aneurysm_coords, device=outputs['coord_reg'].device).float()
            
            losses['coord_reg'] = F.mse_loss(
                outputs['coord_reg'][has_aneurysm.squeeze()],
                aneurysm_coords[has_aneurysm.squeeze()]
            ) * self.coord_reg_weight
        else:
            losses['coord_reg'] = torch.tensor(0.0, device=outputs['coord_reg'].device)
        
        losses['modality_cls'] = F.cross_entropy(
            outputs['modality_cls'],
            targets['modality_label']
        ) * self.modality_cls_weight
        
        losses['total'] = sum(losses.values())
        
        return losses


def train_epoch(model, loader, criterion, optimizer, scaler, device):
    """Train one epoch"""
    model.train()
    total_losses = {}
    
    for batch in loader:
        image = batch['image'].to(device)
        targets = {
            'vessel_mask': batch['vessel_mask'].to(device),
            'aneurysm_mask': batch['aneurysm_mask'].to(device),
            'aneurysm_heatmap': batch['aneurysm_heatmap'].to(device),
            'location_labels': batch['location_labels'].to(device),
            'binary_label': batch['binary_label'].to(device),
            'modality_label': batch['modality_label'].to(device)
        }
        
        optimizer.zero_grad()
        
        with autocast(enabled=CFG.mixed_precision):
            outputs = model(image)
            losses = criterion(outputs, targets)
        
        scaler.scale(losses['total']).backward()
        scaler.step(optimizer)
        scaler.update()
        
        for key, val in losses.items():
            if key not in total_losses:
                total_losses[key] = 0
            total_losses[key] += val.item()
    
    for key in total_losses:
        total_losses[key] /= len(loader)
    
    return total_losses


def validate(model, loader, criterion, device):
    """Validate model"""
    model.eval()
    total_losses = {}
    all_binary_preds = []
    all_binary_labels = []
    
    with torch.no_grad():
        for batch in loader:
            image = batch['image'].to(device)
            targets = {
                'vessel_mask': batch['vessel_mask'].to(device),
                'aneurysm_mask': batch['aneurysm_mask'].to(device),
                'aneurysm_heatmap': batch['aneurysm_heatmap'].to(device),
                'location_labels': batch['location_labels'].to(device),
                'binary_label': batch['binary_label'].to(device),
                'modality_label': batch['modality_label'].to(device)
            }
            
            outputs = model(image)
            losses = criterion(outputs, targets)
            
            for key, val in losses.items():
                if key not in total_losses:
                    total_losses[key] = 0
                total_losses[key] += val.item()
            
            binary_pred = torch.sigmoid(outputs['binary_cls']).cpu().numpy()
            binary_label = targets['binary_label'].cpu().numpy()
            
            all_binary_preds.extend(binary_pred)
            all_binary_labels.extend(binary_label)
    
    for key in total_losses:
        total_losses[key] /= len(loader)
    
    all_binary_preds = np.array(all_binary_preds)
    all_binary_labels = np.array(all_binary_labels)
    
    if len(np.unique(all_binary_labels)) > 1:
        total_losses['auc'] = roc_auc_score(all_binary_labels, all_binary_preds)
    else:
        total_losses['auc'] = 0.0
    
    total_losses['acc'] = accuracy_score(all_binary_labels, all_binary_preds > 0.5)
    
    return total_losses


def predict_with_tta(model, volume, device, n_flips=8):
    """Test-time augmentation with 8 flips and label swapping"""
    model.eval()
    
    transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [2]),  # X flip
        lambda x: torch.flip(x, [3]),  # Y flip
        lambda x: torch.flip(x, [4]),  # Z flip
        lambda x: torch.flip(x, [2, 3]),  # XY flip
        lambda x: torch.flip(x, [2, 4]),  # XZ flip
        lambda x: torch.flip(x, [3, 4]),  # YZ flip
        lambda x: torch.flip(x, [2, 3, 4])  # XYZ flip
    ]
    
    x_flip_indices = [1, 4, 5, 7]
    
    all_predictions = []
    
    with torch.no_grad():
        for i, transform in enumerate(transforms[:n_flips]):
            vol_transformed = transform(volume)
            outputs = model(vol_transformed.to(device))
            
            if i in x_flip_indices:
                location_probs = torch.sigmoid(outputs['location_cls'])
                location_probs_np = location_probs.cpu().numpy()[0]
                location_probs_swapped = swap_location_labels(location_probs_np)
                outputs['location_cls'] = torch.from_numpy(location_probs_swapped).unsqueeze(0).to(device)
            
            all_predictions.append({
                'binary_cls': torch.sigmoid(outputs['binary_cls']).cpu().numpy(),
                'location_cls': torch.sigmoid(outputs['location_cls']).cpu().numpy(),
                'aneurysm_seg': torch.softmax(outputs['aneurysm_seg'], dim=1).cpu().numpy()
            })
    
    final_pred = {
        'binary_cls': np.mean([p['binary_cls'] for p in all_predictions], axis=0),
        'location_cls': np.mean([p['location_cls'] for p in all_predictions], axis=0),
        'aneurysm_seg': np.mean([p['aneurysm_seg'] for p in all_predictions], axis=0)
    }
    
    return final_pred


def extract_aneurysm_info(aneurysm_seg, location_probs, img_size):
    """Extract aneurysm location info from predictions"""
    binary_mask = (aneurysm_seg[0, 1] > 0.5).astype(np.uint8)
    
    if binary_mask.sum() > 0:
        coords = np.array(np.where(binary_mask > 0)).mean(axis=1)
        location_idx = location_probs[0].argmax()
        location_name = LOCATION_NAMES[location_idx]
        confidence = location_probs[0].max()
    else:
        coords = np.array([img_size[0]//2, img_size[1]//2, img_size[2]//2])
        location_idx = location_probs[0].argmax()
        location_name = LOCATION_NAMES[location_idx]
        confidence = location_probs[0].max()
    
    return {
        'coordinates': coords,
        'location': location_name,
        'confidence': confidence,
        'all_location_probs': {LOCATION_NAMES[i]: location_probs[0, i] for i in range(13)}
    }


def create_balanced_subset(train_df, localizers_df, segmentations_dir, n_samples):
    """Create balanced subset with preference for cases with segmentation"""
    available_segs = [f.replace('_cowseg.nii', '') for f in os.listdir(segmentations_dir) if f.endswith('.nii')]
    
    train_with_seg = train_df[train_df['SeriesInstanceUID'].isin(available_segs)]
    
    pos = train_with_seg[train_with_seg['Aneurysm Present'] == 1]
    neg = train_with_seg[train_with_seg['Aneurysm Present'] == 0]
    
    n_pos = min(n_samples // 2, len(pos))
    n_neg = min(n_samples - n_pos, len(neg))
    
    if n_pos > 0:
        pos_sample = pos.sample(n_pos, random_state=CFG.seed)
    else:
        pos_sample = pd.DataFrame()
    
    if n_neg > 0:
        neg_sample = neg.sample(n_neg, random_state=CFG.seed)
    else:
        neg_sample = pd.DataFrame()
    
    subset = pd.concat([pos_sample, neg_sample]).reset_index(drop=True)
    
    return subset


if CFG.use_subset:
    train_subset = create_balanced_subset(
        train_df, localizers_df, CFG.segmentations_dir, CFG.n_subset_samples
    )
else:
    train_subset = train_df

train_split_df, temp_df = train_test_split(
    train_subset, test_size=(1 - CFG.train_split), random_state=CFG.seed, stratify=train_subset['Aneurysm Present']
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, random_state=CFG.seed, stratify=temp_df['Aneurysm Present']
)


for i in range(min(2, len(train_subset))):
    series_uid = train_subset.iloc[i]['SeriesInstanceUID']
    visualize_stage1_pipeline(series_uid, localizers_df)


train_dataset = AneurysmDataset(train_split_df, localizers_df, CFG.img_size, augment=True)
val_dataset = AneurysmDataset(val_df, localizers_df, CFG.img_size, augment=False)
test_dataset = AneurysmDataset(test_df, localizers_df, CFG.img_size, augment=False)

train_loader = DataLoader(train_dataset, batch_size=CFG.batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=CFG.batch_size, shuffle=False, num_workers=2)


model = nnXNetLite(
    in_channels=1,
    num_vessel_classes=CFG.num_vessel_classes,
    num_aneurysm_classes=CFG.num_aneurysm_classes,
    num_location_classes=CFG.num_location_classes,
    num_modality_classes=CFG.num_modality_classes,
    base_channels=CFG.base_channels
).to(CFG.device)


model


criterion = MultiTaskLoss(
    vessel_seg_weight=CFG.vessel_seg_weight,
    aneurysm_seg_weight=CFG.aneurysm_seg_weight,
    binary_cls_weight=CFG.binary_cls_weight,
    location_cls_weight=CFG.location_cls_weight,
    coord_reg_weight=CFG.coord_reg_weight,
    modality_cls_weight=CFG.modality_cls_weight
)


optimizer = optim.Adam(model.parameters(), lr=CFG.learning_rate, weight_decay=CFG.weight_decay)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.num_epochs)
scaler = GradScaler(enabled=CFG.mixed_precision)


history = {'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_acc': []}
best_val_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(CFG.num_epochs):
    train_losses = train_epoch(model, train_loader, criterion, optimizer, scaler, CFG.device)
    val_losses = validate(model, val_loader, criterion, CFG.device)
    scheduler.step()
    
    history['train_loss'].append(train_losses['total'])
    history['val_loss'].append(val_losses['total'])
    history['val_auc'].append(val_losses['auc'])
    history['val_acc'].append(val_losses['acc'])
    
    print(f"Epoch {epoch+1}/{CFG.num_epochs}")
    print(f"  Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f}")
    print(f"  Val AUC: {val_losses['auc']:.4f} | Val Acc: {val_losses['acc']:.4f}")
    
    if val_losses['total'] < best_val_loss:
        best_val_loss = val_losses['total']
        torch.save(model.state_dict(), f'{CFG.work_dir}/best_model.pth')
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break


fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history['train_loss'], label='Train')
axes[0].plot(history['val_loss'], label='Val')
axes[0].set_title('Loss')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(history['val_auc'], label='AUC', color='green')
axes[1].set_title('Validation AUC')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(history['val_acc'], label='Accuracy', color='orange')
axes[2].set_title('Validation Accuracy')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.show()


model.load_state_dict(torch.load(f'{CFG.work_dir}/best_model.pth'))
test_losses = validate(model, test_loader, criterion, CFG.device)


print("Test Results:")
print(f"  Total Loss: {test_losses['total']:.4f}")
print(f"  Binary Loss: {test_losses['binary_cls']:.4f}")
print(f"  AUC: {test_losses['auc']:.4f}")
print(f"  Accuracy: {test_losses['acc']:.4f}")


if CFG.use_tta:
    sample_idx = 0
    sample_batch = test_dataset[sample_idx]
    volume = sample_batch['image'].unsqueeze(0)
    
    tta_pred = predict_with_tta(model, volume, CFG.device, n_flips=CFG.tta_n_flips)
    
    aneurysm_info = extract_aneurysm_info(
        tta_pred['aneurysm_seg'],
        tta_pred['location_cls'],
        CFG.img_size
    )
    
    print("\nTTA Prediction Results:")
    print(f"  Aneurysm Present: {tta_pred['binary_cls'][0][0]:.4f}")
    print(f"  Location: {aneurysm_info['location']}")
    print(f"  Confidence: {aneurysm_info['confidence']:.4f}")
    print(f"  Coordinates: {aneurysm_info['coordinates']}")
    print(f"\n  All Location Probabilities:")
    for loc, prob in aneurysm_info['all_location_probs'].items():
        print(f"    {loc}: {prob:.4f}")


def visualize_with_ground_truth(model, dataset, idx, device, localizers_df):
    """Visualize with explicit ground truth location"""
    model.eval()
    batch = dataset[idx]
    series_uid = batch['series_uid']
    
    loc_gt = localizers_df[localizers_df['SeriesInstanceUID'] == series_uid]
    
    print(f"\n=== Ground Truth Info ===")
    print(f"Series: {series_uid}")
    print(f"Binary label (has aneurysm): {batch['binary_label'].item()}")
    print(f"GT mask sum: {batch['aneurysm_mask'].sum().item()}")
    print(f"Localizers entries: {len(loc_gt)}")
    
    if len(loc_gt) > 0:
        print("\nLocalizer data:")
        for _, row in loc_gt.iterrows():
            print(f"  Location: {row['location']}")
            print(f"  Coordinates: {row['coordinates']}")
    else:
        print("  No localizer data (negative case)")
    
    image = batch['image'].unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(image)
    
    volume = batch['image'].squeeze().cpu().numpy()
    aneurysm_pred = torch.softmax(outputs['aneurysm_seg'], dim=1)[0, 1].cpu().numpy()
    aneurysm_gt = batch['aneurysm_mask'].cpu().numpy()
    
    binary_pred = torch.sigmoid(outputs['binary_cls']).item()
    location_probs = torch.sigmoid(outputs['location_cls']).cpu().numpy()[0]
    location_pred = LOCATION_NAMES[location_probs.argmax()]
    coord_pred = outputs['coord_reg'].cpu().numpy()[0]
    
    # Parse GT centroid
    if aneurysm_gt.sum() > 0:
        gt_coords = np.where(aneurysm_gt > 0)
        gt_centroid = [gt_coords[0].mean(), gt_coords[1].mean(), gt_coords[2].mean()]
        use_centroid = gt_centroid
        has_gt = True
    elif len(loc_gt) > 0:
        try:
            coords = eval(loc_gt.iloc[0]['coordinates'])
            # Handle dict format {'x': ..., 'y': ..., 'z': ...} or list/tuple
            if isinstance(coords, dict):
                x = coords.get('x', coords.get('y', volume.shape[0] / 2))
                y = coords.get('y', coords.get('x', volume.shape[1] / 2))
                z = coords.get('z', volume.shape[2] / 2)  # Default to middle if no z
            else:
                x = coords[0] if len(coords) > 0 else volume.shape[0] / 2
                y = coords[1] if len(coords) > 1 else volume.shape[1] / 2
                z = coords[2] if len(coords) > 2 else volume.shape[2] / 2
            
            # Scale to current volume
            gt_centroid = [
                x * volume.shape[0] / 512,
                y * volume.shape[1] / 512,
                z * volume.shape[2] / 512
            ]
            use_centroid = gt_centroid
            has_gt = True
        except:
            use_centroid = coord_pred
            has_gt = False
    else:
        use_centroid = coord_pred
        has_gt = False
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    slices = [
        max(0, int(use_centroid[2]) - 16),
        int(use_centroid[2]),
        min(volume.shape[2] - 1, int(use_centroid[2]) + 16)
    ]
    
    for i, s in enumerate(slices):
        axes[0, i].imshow(volume[:, :, s], cmap='gray')
        axes[0, i].set_title(f'Volume {s}')
        axes[0, i].axis('off')
        
        axes[1, i].imshow(volume[:, :, s], cmap='gray', alpha=0.7)
        
        pred_slice = aneurysm_pred[:, :, s]
        if pred_slice.max() > 0.01:
            axes[1, i].imshow(pred_slice, cmap='Reds', alpha=0.4, vmin=0, vmax=0.5)
        
        if aneurysm_gt[:, :, s].sum() > 0:
            axes[1, i].contour(aneurysm_gt[:, :, s], colors='lime', linewidths=3, levels=[0.5])
        
        if s == int(use_centroid[2]):
            if has_gt:
                axes[1, i].plot(use_centroid[1], use_centroid[0], 'g*', markersize=25, 
                               markeredgewidth=2, markeredgecolor='white', label='GT')
            axes[1, i].plot(coord_pred[1], coord_pred[0], 'r*', markersize=20, 
                           markeredgewidth=2, markeredgecolor='yellow', label='Pred')
            axes[1, i].legend(loc='upper right')
        
        axes[1, i].set_title(f'Pred {s} (max={pred_slice.max():.3f})')
        axes[1, i].axis('off')
    
    title = f'GT: {batch["binary_label"].item():.0f} | Pred: {binary_pred:.3f} | {location_pred[:30]}'
    plt.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.show()
    
    print(f"\n=== Prediction ===")
    print(f"Binary: {binary_pred:.4f}")
    print(f"Location: {location_pred} (conf: {location_probs.max():.4f})")
    print(f"Pred coords: {coord_pred}")
    if has_gt:
        print(f"GT coords: {use_centroid}")


for i in range(len(test_dataset)):
    visualize_with_ground_truth(model, test_dataset, i, CFG.device, localizers_df)

