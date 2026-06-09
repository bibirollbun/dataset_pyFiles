from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2

import sys
import matplotlib.pyplot as plt 
sys.path.append('/kaggle/input/rsna-iad-vesselfm-codebase')
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from monai.inferers import SlidingWindowInfererAdapt
from skimage.morphology import remove_small_objects
from skimage.exposure import equalize_hist
from utils.data import generate_transforms
from utils.io import determine_reader_writer
import os
from monai.transforms import LoadImaged, Spacingd, LoadImage
from monai.networks.nets import DynUNet
import SimpleITK as sitk
import yaml
import torch.nn as nn
from scipy.ndimage import label
from tqdm import  tqdm
import pydicom
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import pandas as pd
from matplotlib.widgets import Slider
import ipywidgets as widgets
from IPython.display import HTML
from matplotlib.animation import FuncAnimation
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    Resized,
    RandCropByLabelClassesd,
    SpatialPadd,
    ConcatItemsd,
    ToTensord,
    Lambdad,
)

from monai.transforms import MapTransform
from skimage import morphology, filters


yaml_path = '/kaggle/input/rsna-iad-vesselfm-codebase/configs/inference.yaml'
MAX_WORKERS = 4
with open(yaml_path, 'r') as f:
    config = yaml.safe_load(f)


class CFG:
    ckpt_path = '/kaggle/input/rsna-iad-vesselfm-13-classes/pytorch/default/2/vesselfm_13_classes_dynunet-val_dice0.5047.ckpt'
    model_structure = {
        'in_channels': 1,
        'out_channels': 14,
        'spatial_dims': 3,
        'strides': [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]] , # 5 levels
        'kernel_size': [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        'upsample_kernel_size': [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        'filters': [32, 64, 128, 256, 320],
        'res_block': True}
    device = 'cuda:0'
    thrd = 0.1

    #sliding window
    batch_size= 1
    patch_size= [128, 128, 128]
    overlap= 0.5
    mode= "constant"
    sigma_scale= 0.125
    padding_mode= "constant"

    #volume transform
    transforms_config = config['transforms_config']
    transforms_config.insert(1, {'Resize': {
        'spatial_size': (128, 384, 384),
        'mode': "trilinear"
    }},)
    
    tta = config['tta']
    post = config['post']
    merging = config['merging']


def load_dicom_series(series_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a DICOM series into a 3D HU volume with affine matrix.

    Args:
        series_dir (Path): directory containing DICOM files

    Returns:
        volume (np.ndarray): 3D array (Z, Y, X) in HU
        affine (np.ndarray): 4x4 voxel-to-world affine
    """
    # Collect DICOMs
    dcm_paths = [Path(series_dir) / f for f in os.listdir(series_dir) if f.lower().endswith(".dcm")]
    if not dcm_paths:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")
    
    slices = [pydicom.dcmread(str(p), force=True) for p in dcm_paths]

    # Orientation & sorting
    orientation = np.array(slices[0].ImageOrientationPatient).reshape(2, 3)
    row_cos, col_cos = orientation
    normal = np.cross(row_cos, col_cos)
    slices.sort(key=lambda ds: np.dot(ds.ImagePositionPatient, normal))

    # HU scaling
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    volume = np.stack([ds.pixel_array for ds in slices]).astype(np.float32)
    volume = volume * slope + intercept

    # Spacing
    Δx, Δy = [float(x) for x in slices[0].PixelSpacing]
    positions = [np.array(ds.ImagePositionPatient) for ds in slices]
    slice_positions = [np.dot(p, normal) for p in positions]
    Δz = float(np.mean(np.diff(slice_positions)))

    # Affine
    slice_cos = normal
    origin = np.array(slices[0].ImagePositionPatient)
    affine = np.eye(4)
    affine[0:3, 0] = row_cos * Δx
    affine[0:3, 1] = col_cos * Δy
    affine[0:3, 2] = slice_cos * Δz
    affine[0:3, 3] = origin

    return volume


class ModalityIntensityScalingd(MapTransform):
    """
    Modality-agnostic normalization:
      - Robust z-score within 2–98 percentiles
      - Then rescale to [0, 1]
    """
    def __init__(self, keys=("Image",)):
        super().__init__(keys)

    def __call__(self, data):
        d = dict(data)
        img = d[self.keys[0]]

        # robust stats
        p2, p98 = np.percentile(img, (2, 98))
        mask = (img >= p2) & (img <= p98)
        mean = np.mean(img[mask])
        std = np.std(img[mask]) + 1e-6

        img = (img - mean) / std
        img = np.clip((img - img.min()) / (img.max() - img.min() + 1e-6), 0, 1)

        d[self.keys[0]] = img.astype(np.float32)
        return d

def _generate_transforms(vol_size, input_size, mode):
    if mode == "train":
        return Compose([
            EnsureChannelFirstd(keys=["Image", "Mask"], channel_dim="no_channel"),
            EnsureTyped(keys=["Image", "Mask"]),
            Resized(keys=["Image", "Mask"], spatial_size=vol_size, mode=["trilinear", "nearest"]),
            RandCropByLabelClassesd(
                keys=["Image", "Mask"],
                label_key="Mask",
                spatial_size=input_size,
                num_classes=13,
                ratios=[1] * 13,
                num_samples=4,
                image_key="Image",
                allow_smaller=True,
            ),
            ModalityIntensityScalingd(keys=["Image"]),
            SpatialPadd(keys=["Image", "Mask"], spatial_size=input_size, mode="constant", method="symmetric"),
            ConcatItemsd(keys=["Image", "Mask"], name=["Image", "Mask"], dim=0),
            ToTensord(keys=["Image", "Mask"]),
        ])
    elif mode == 'val':
        return Compose([
            EnsureChannelFirstd(keys=["Image", "Mask"], channel_dim="no_channel"),
            EnsureTyped(keys=["Image", "Mask"]),
            Resized(keys=["Image", "Mask"], spatial_size=vol_size, mode=["trilinear", "nearest"]),
            ModalityIntensityScalingd(keys=["Image"]),
            ToTensord(keys=["Image", "Mask"]),
        ])
    else:
        return Compose([
            EnsureChannelFirstd(keys=["Image"], channel_dim="no_channel"),
            EnsureTyped(keys=["Image"]),
            Resized(keys=["Image"], spatial_size=vol_size, mode=["trilinear"]),
            ModalityIntensityScalingd(keys=["Image"]),
            ToTensord(keys=["Image"]),
        ])



def cca_remove(mask_preds, min_size = 1000, sigma = 0.5):
    '''
    mask_preds: (D, H, W)
    '''
    #to 2d mip
    mip_2d = mask_preds.max(axis=0)
    
    # Remove small bright spots (choose an area threshold)
    smoothed_mip = filters.gaussian(mip_2d, sigma=sigma)
    cleaned_mip = morphology.remove_small_objects(smoothed_mip > 0, min_size=min_size)  
    
    # Morphological opening for noise removal
    selem = morphology.disk(2)  
    cleaned_mip = morphology.opening(cleaned_mip, selem)
    
    # Convert boolean mask back to vessel intensity (optional thresholding)
    cleaned_mip = cleaned_mip.astype(np.uint8) * mip_2d.max()
    
    # Median filter smoothing
    cleaned_mip_mask = filters.median(cleaned_mip, morphology.disk(1))

    masked_vol = (cleaned_mip_mask[None]!=0) * mask_preds

    new_mip_mask = masked_vol.max(axis=0)
    
    return masked_vol, new_mip_mask


def load_model(cfg):
    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device, weights_only=False)['state_dict']
    ckpt = {k.replace("model.", ""): v for k, v in ckpt.items()}
    model = DynUNet(**CFG.model_structure)
    model.load_state_dict(ckpt)
    model.eval()
    return model.to(cfg.device)


sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=CFG.patch_size, sw_batch_size=1, overlap=0.5,
        )


transforms = generate_transforms(CFG.transforms_config)


model = load_model(CFG)


df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
df_loc = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')
seg_path = Path('/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations')
series_path = Path('/kaggle/input/rsna-intracranial-aneurysm-detection/series')
seg_uids = [name.split('.nii')[0] for name in os.listdir(seg_path) if 'cowseg' not in name]


#uid = '1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647'
#uid = '1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647'
uid = '1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317'
print(uid in seg_uids)
path = series_path/uid
mask_path = seg_path/f'{uid}_cowseg.nii'
vol_path = seg_path/f'{uid}.nii'
df[df.SeriesInstanceUID	== uid]


vol_ = load_dicom_series(path)
vol = np.flip(vol_, axis=1).copy()
vol = np.flip(vol, axis=2).copy()

transform_test = _generate_transforms((128, 384, 384), (128, 128, 128), 'test')
_input = {'Image': vol}

test_data = transform_test(_input)
test_vol = test_data['Image'].to(CFG.device)


with torch.no_grad():
    pred_logits = sliding_window_inferer(test_vol[None], model)


test_mip, _ = test_vol[0].max(dim=0)
test_mip = test_mip.cpu()

pred_probs = pred_logits.softmax(dim=1)
pred_mask = pred_logits.argmax(dim=1)

mask_mip, _ = pred_mask[0].max(dim=0)
mask_mip = mask_mip.cpu()


new_mask, new_mask_mip = cca_remove(pred_mask[0].cpu().numpy())


fig, ax = plt.subplots(ncols=3, figsize = (16, 16))
ax[0].imshow(test_mip)
ax[1].imshow(mask_mip)
ax[1].set_title('before process')
ax[2].imshow(new_mask_mip)
ax[2].set_title('after process')


!tar xfvz /kaggle/input/ultralytics-offlineinstall-yolo12-weights/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages


# ML/DL
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
import timm

# Transformations
import albumentations as A
from albumentations.pytorch import ToTensorV2

# YOLO
from ultralytics import YOLO
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.cm as cm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Optimization settings
torch.set_float32_matmul_precision('medium')
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_yolo_models():
    """Load all YOLO models"""
    models = []
    for config in YOLO_MODEL_CONFIGS:
        model = YOLO(config["path"])
        model.to(device)
        
        model_dict = {
            "model": model,
            "weight": config["weight"],
            "name": config["name"],
            "fold": config["fold"]
        }
        models.append(model_dict)
    return models


# YOLO label mappings
YOLO_LABELS_TO_IDX = {
    'Anterior Communicating Artery': 0,
    'Basilar Tip': 1,
    'Left Anterior Cerebral Artery': 2,
    'Left Infraclinoid Internal Carotid Artery': 3,
    'Left Middle Cerebral Artery': 4,
    'Left Posterior Communicating Artery': 5,
    'Left Supraclinoid Internal Carotid Artery': 6,
    'Other Posterior Circulation': 7,
    'Right Anterior Cerebral Artery': 8,
    'Right Infraclinoid Internal Carotid Artery': 9,
    'Right Middle Cerebral Artery': 10,
    'Right Posterior Communicating Artery': 11,
    'Right Supraclinoid Internal Carotid Artery': 12
}
YOLO_LABELS = sorted(list(YOLO_LABELS_TO_IDX.keys()))


# ====================================================
IMG_SIZE = 512
BATCH_SIZE = int(os.getenv("YOLO_BATCH_SIZE", "32"))
MAX_WORKERS = 4

YOLO_MODEL_CONFIGS = [
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_y11m_with_mix_up_mosaic_fold0/weights/best.pt",
        "fold": "0",
        "weight": 1.0,
        "name": "YOLOv11n_fold0"
    },
    {
        "path": "/kaggle/input/rsna-sergio-models/cv_y11m_with_mix_up_mosaic_fold1/weights/best.pt",
        "fold": "1",
        "weight": 1.0,
        "name": "YOLOv11n_fold1"
    },  
    #{
    #    "path": "/kaggle/input/rsna-sergio-models/cv_y11m_with_mix_up_mosaic_fold2/weights/best.pt",
    #    "fold": "2",
    #    "weight": 1.0,
    #    "name": "YOLOv11n_fold2"
    #}
]

YOLO_MODELS = load_yolo_models()


def read_dicom_frames_hu(path: Path) -> List[Tuple[float, np.ndarray]]:
    """Read DICOM file and return list of (slice_position, HU frame)"""
    ds = pydicom.dcmread(str(path), force=True)
    pix = ds.pixel_array
    slope = float(getattr(ds, 'RescaleSlope', 1.0))
    intercept = float(getattr(ds, 'RescaleIntercept', 0.0))

    # Compute slice location using orientation + position
    try:
        orientation = np.array(ds.ImageOrientationPatient).reshape(2, 3)
        row_cos, col_cos = orientation
        normal = np.cross(row_cos, col_cos)  # slice normal vector
        position = np.array(ds.ImagePositionPatient)
        slice_loc = float(np.dot(position, normal))  # projection along normal
    except Exception:
        # Fallback: SliceLocation / InstanceNumber
        slice_loc = float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", 0.0)))

    frames: List[Tuple[float, np.ndarray]] = []

    if pix.ndim == 2:
        img = pix.astype(np.float32)
        frames.append((slice_loc, img * slope + intercept))
    elif pix.ndim == 3:
        # RGB or multi-frame
        if pix.shape[-1] == 3 and pix.shape[0] != 3:
            try:
                gray = cv2.cvtColor(pix.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            except Exception:
                gray = pix[..., 0].astype(np.float32)
            frames.append((slice_loc, gray * slope + intercept))
        else:
            for i in range(pix.shape[0]):
                frm = pix[i].astype(np.float32)
                # tiny offset ensures consistent ordering for multi-frame
                frames.append((slice_loc + i * 1e-3, frm * slope + intercept))
    return frames


def min_max_normalize(img: np.ndarray) -> np.ndarray:
    """Min-max normalization to 0-255 with optional flipping"""
    mn, mx = float(img.min()), float(img.max())
    if mx - mn < 1e-6:
        norm = np.zeros_like(img, dtype=np.uint8)
    else:
        norm = (img - mn) / (mx - mn)
        norm = (norm * 255.0).clip(0, 255).astype(np.uint8)
    return norm


def process_dicom_file(dcm_path: Path) -> List[Tuple[float, np.ndarray]]:
    """Process single DICOM file -> list of (slice_loc, image) tuples"""
    try:
        frames = read_dicom_frames_hu(dcm_path)
        processed_slices = []
        for loc, f in frames:
            img_u8 = min_max_normalize(f)
            if img_u8.ndim == 2:
                img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
            processed_slices.append((loc, img_u8))
        return processed_slices
    except Exception as e:
        print(f"Failed processing {dcm_path.name}: {e}")
        return []


def collect_series_slices(series_dir: Path) -> List[Path]:
    """Collect all DICOM files in a series directory (recursively)."""
    dcm_paths: List[Path] = []
    try:
        for root, _, files in os.walk(series_dir):
            for f in files:
                if f.lower().endswith('.dcm'):
                    dcm_paths.append(Path(root) / f)
    except Exception as e:
        print(f"Failed to walk series dir {series_dir}: {e}")
    return dcm_paths


def slice_sort_key(path: Path) -> float:
    """Compute a robust slice sort key (orientation + position) for a single DICOM file"""
    try:
        ds = pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        orientation = np.array(ds.ImageOrientationPatient).reshape(2, 3)
        row_cos, col_cos = orientation
        normal = np.cross(row_cos, col_cos)
        position = np.array(ds.ImagePositionPatient)
        return float(np.dot(position, normal))
    except Exception:
        # fallback
        try:
            return float(getattr(ds, "SliceLocation", getattr(ds, "InstanceNumber", 0.0)))
        except:
            return 0.0

def process_dicom_for_yolo(series_path):
    series_path = Path(series_path)
    dicom_files = collect_series_slices(series_path)
    
    # Sort DICOM files by orientation+position before processing
    dicom_files.sort(key=slice_sort_key)
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_dicom_file, dicom_files))
    
    # Flatten into (loc, img)
    all_slices_with_loc = [item for sublist in results for item in sublist]
    
    # Already sorted by dicom_files order, but double-check (safe)
    all_slices_with_loc.sort(key=lambda x: x[0])
    
    # Extract just the images
    all_slices = [img for _, img in all_slices_with_loc]
    
    # Now dicom_files matches the sorted slices
    dcm_list = [f.stem for f in dicom_files]
    return all_slices


@torch.no_grad()
def predict_yolo_ensemble(slices):
    if not slices:
        return 0.1, np.ones(len(YOLO_LABELS)) * 0.1
    ensemble_cls_preds = []
    ensemble_loc_preds = []
    total_weight = 0.0
    vol_size = (len(slices), slices[0].shape[0], slices[0].shape[1])
    location_preds = {f'MODEL{i}': [] for i in range(len(YOLO_MODELS))}
    
    for model_idx, model_dict in enumerate(YOLO_MODELS):
        model = model_dict["model"]
        weight = model_dict["weight"]
        all_detections = []
        try:
            max_conf_all = 0.0
            per_class_max = np.zeros(len(YOLO_LABELS), dtype=np.float32)
            
            # Process in batches
            for i in range(0, len(slices), BATCH_SIZE):
                batch_slices = slices[i:i+BATCH_SIZE]
                z_idxes = [i + batch_idx  for batch_idx in range(len(batch_slices))]
                results = model.predict(
                    batch_slices, 
                    verbose=False, 
                    batch=len(batch_slices), 
                    device="cuda:0", 
                    conf=0.01
                )
                
                for z_idx, r in enumerate(results):
                    if r is None or r.boxes is None or r.boxes.conf is None or len(r.boxes) == 0:
                        continue
                    try:
                        confs = r.boxes.conf
                        clses = r.boxes.cls
                        for j in range(len(confs)):
                            c = float(confs[j].item())
                            k = int(clses[j].item())
                            if c > max_conf_all:
                                max_conf_all = c
                            if 0 <= k < len(YOLO_LABELS) and c > per_class_max[k]:
                                per_class_max[k] = c
                            x1, y1, x2, y2 = r.boxes.xyxy[j].cpu().numpy()
                            x_center = (x1 + x2) / 2
                            y_center = (y1 + y2) / 2
                            point = np.array([round(z_idxes[z_idx]), round(y_center), round(x_center)])
                            location_preds[f'MODEL{model_idx}'].append([*point, float(c), k, model_idx]) #z, y, x, prob, class, model index
                            
                    except Exception as e:
                        print(e)
                        try:
                            batch_max = float(r.boxes.conf.max().item())
                            if batch_max > max_conf_all:
                                max_conf_all = batch_max
                        except Exception:
                            pass
            ensemble_cls_preds.append(max_conf_all * weight)
            ensemble_loc_preds.append(per_class_max * weight)
            total_weight += weight
            
        except Exception as e:
            print(f"Error in model {model_dict['name']}: {e}")
            ensemble_cls_preds.append(0.1 * weight)
            ensemble_loc_preds.append(np.ones(len(YOLO_LABELS)) * 0.1 * weight)
            total_weight += weight

    location_preds = {k: np.array(v) for k, v in location_preds.items()}
    
    if total_weight > 0:
        final_cls_pred = sum(ensemble_cls_preds) / len(ensemble_cls_preds)
        final_loc_preds = sum(ensemble_loc_preds) / total_weight
    else:
        final_cls_pred = 0.1
        final_loc_preds = np.ones(len(YOLO_LABELS)) * 0.1
    return final_cls_pred, final_loc_preds, location_preds


all_slices = process_dicom_for_yolo(path)


final_cls_pred, final_loc_preds, location_preds = predict_yolo_ensemble(all_slices)


SEG_LABELS_TO_IDX = {
    "BG": 0,
    "Other Posterior Circulation": 1,
    "Basilar Tip": 2,
    "Right Posterior Communicating Artery": 3,
    "Left Posterior Communicating Artery": 4,
    "Right Infraclinoid Internal Carotid Artery": 5,
    "Left Infraclinoid Internal Carotid Artery": 6,
    "Right Supraclinoid Internal Carotid Artery": 7,
    "Left Supraclinoid Internal Carotid Artery": 8,
    "Right Middle Cerebral Artery": 9,
    "Left Middle Cerebral Artery": 10,
    "Right Anterior Cerebral Artery": 11,
    "Left Anterior Cerebral Artery": 12,
    "Anterior Communicating Artery": 13,
}

# Invert SEG map
SEG_IDX_TO_LABELS = {v: k for k, v in SEG_LABELS_TO_IDX.items()}

# Build YOLO→SEG mapping
YOLO_TO_SEG = {
    yolo_idx: SEG_LABELS_TO_IDX[name]
    for name, yolo_idx in YOLO_LABELS_TO_IDX.items()
    if name in SEG_LABELS_TO_IDX
}

max_yolo_idx = max(YOLO_TO_SEG.keys())
yolo2seg_lookup = np.zeros(max_yolo_idx + 1, dtype=int)
for yolo_idx, seg_idx in YOLO_TO_SEG.items():
    yolo2seg_lookup[yolo_idx] = seg_idx


def resize_mask_3d(volume, target_size):
    """
    Resize a 3D volume [D, H, W] to [d, h, w] using nearest neighbor interpolation (SimpleITK).
    """
    d, h, w = target_size
    image = sitk.GetImageFromArray(volume)
    old_size = image.GetSize()
    old_spacing = image.GetSpacing()
    
    new_size = [w, h, d]  # SITK uses (x, y, z)
    new_spacing = [
        old_spacing[i] * (old_size[i] / new_size[i]) for i in range(3)
    ]
    
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(new_size)
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    
    resized_img = resampler.Execute(image)
    reszied_mask = sitk.GetArrayFromImage(resized_img)
    return np.flip(np.flip(reszied_mask, axis=2), axis=1)


all_slices_vol = np.array(all_slices)[..., 0]


resized_mask = resize_mask_3d(new_mask, all_slices_vol.shape)


loc_preds_2d_1 = location_preds['MODEL0'][:, [1, 2]] #y, x
loc_preds_2d_2 = location_preds['MODEL1'][:, [1, 2]] #y, x
conf1 = location_preds['MODEL0'][:, 3]
conf2 = location_preds['MODEL1'][:, 3]


fig, ax = plt.subplots(figsize=(12, 12))


#label map
cmap_base = plt.get_cmap("jet", len(SEG_LABELS_TO_IDX))
colors = [cmap_base(i) for i in range(len(SEG_LABELS_TO_IDX))]
colors[0] = (0.25, 0.25, 0.25, 1.0)
cmap_custom = mcolors.ListedColormap(colors)

# Image in grayscale
ax.imshow(vol_.max(0), cmap="gray", alpha=0.5)

# Mask with modified cmap
ax.imshow(resized_mask.max(0), alpha=0.25, cmap=cmap_custom)


# Normalize confidence [0,1] for colormap
norm = plt.Normalize(0, 1)
cmap_conf = cm.get_cmap("plasma")  # high contrast confidence heatmap

# Overlay detections
sc1 = ax.scatter(
    loc_preds_2d_1[:, 1], loc_preds_2d_1[:, 0],
    c=conf1, cmap=cmap_conf, norm=norm,
    s=conf1 * 400, alpha=0.9, marker="o",
    edgecolors="k", linewidths=0.3,
    label="Fold0 predicted points"
)

sc2 = ax.scatter(
    loc_preds_2d_2[:, 1], loc_preds_2d_2[:, 0],
    c=conf2, cmap=cmap_conf, norm=norm,
    s=conf2 * 400, alpha=0.9, marker="^",
    linewidths=0.8,
    label="Fold1 predicted points"
)

# Colorbar axis (to the right of the plot)
cbar = fig.colorbar(sc1, ax=ax, orientation="vertical", fraction=0.046, pad=0.04)
cbar.set_label("Confidence (0 = low, 1 = high)", fontsize=12)

# Legends
unique_classes = np.unique(resized_mask)
handles = [
    mpatches.Patch(color=cmap_custom(idx), label=label)
    for label, idx in SEG_LABELS_TO_IDX.items()
    if idx in unique_classes
]

# Predicted points legend
scatter_legend = ax.legend(
    markerscale=2, fontsize=12, handlelength=2, handletextpad=0.8,
    loc="lower right", title="Predicted Points"
)
ax.add_artist(scatter_legend)

# Segmentation legend
ax.legend(
    handles=handles,
    loc='upper center', bbox_to_anchor=(0.5, 1.05),
    ncol=3, fancybox=True, shadow=True
)

ax.axis("off")
ax.set_title("MIP view on segmentation prediction and predicted points", fontsize=20, pad=40)

plt.tight_layout(rect=[0, 0, 0.95, 1])  # leave space for colorbar on right
plt.show()


# for i in range(0, 100, 5):
#     depth_1 = location_preds['MODEL0'][:, 0]
#     depth_2 = location_preds['MODEL1'][:, 0]
#     loc_slice_preds_1 = location_preds['MODEL0'][depth_1==i]
#     loc_slice_preds_2 = location_preds['MODEL1'][depth_2==i]
#     fig, ax = plt.subplots(ncols=3, figsize = (16, 16))
#     ax[0].imshow(resized_mask[i])
#     ax[0].set_title('predicted mask')
#     ax[1].imshow(vol_[i])
#     ax[1].scatter(loc_slice_preds_1[:, 2], loc_slice_preds_1[:, 1], color = 'red', label ='yolo-fold0 preds')
#     ax[1].scatter(loc_slice_preds_2[:, 2], loc_slice_preds_2[:, 1], color = 'white', label ='yolo-fold1 preds')
#     ax[1].legend()
#     ax[1].set_title('vol from nii preprocess')
#     ax[2].imshow(all_slices_vol[i])
#     ax[2].set_title('vol from yolo preprocess')
#     plt.show()


def check_valid(yolo_loc_cls_preds, seg_loc_cls_preds):
    return yolo2seg_lookup[yolo_loc_cls_preds]==seg_loc_cls_preds


loc_cls_preds_1 = location_preds['MODEL0'][:, -2].astype('int32')
loc_cls_preds_2 = location_preds['MODEL1'][:, -2].astype('int32')

loc_preds_1  = location_preds['MODEL0'][:, [0, 1, 2]].astype('int32')
loc_preds_2  = location_preds['MODEL1'][:, [0, 1, 2]].astype('int32')


seg_cls_preds_1 = resized_mask[loc_preds_1[:, 0], loc_preds_1[:, 1], loc_preds_1[:, 2]]
seg_cls_preds_2 = resized_mask[loc_preds_2[:, 0], loc_preds_2[:, 1], loc_preds_2[:, 2]]


valid_mask_1 = check_valid(loc_cls_preds_1, seg_cls_preds_1)
valid_mask_2 = check_valid(loc_cls_preds_2, seg_cls_preds_2)


print('yolo-fold0 invalid point percentage:', round(100 * np.sum(np.logical_not(valid_mask_1))/len(valid_mask_1), 2), '%')
print('yolo-fold1 invalid point percentage:', round(100 * np.sum(np.logical_not(valid_mask_2))/len(valid_mask_2), 2), '%')

