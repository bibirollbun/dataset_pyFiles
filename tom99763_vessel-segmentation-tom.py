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


def load_model(cfg):
    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device, weights_only=False)['state_dict']
    ckpt = {k.replace("model.", ""): v for k, v in ckpt.items()}
    model = DynUNet(**CFG.model_structure)
    model.load_state_dict(ckpt)
    model.eval()
    return model.to(cfg.device)


df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')


model = load_model(CFG)


reader = determine_reader_writer('nii')()
transforms = generate_transforms(CFG.transforms_config)


uid = '1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381'
data_path = '/kaggle/input/rsna-intracranial-aneurysm-detection'
vol_path = f'{data_path}/segmentations/{uid}.nii'
mask_path = f'{data_path}/segmentations/{uid}_cowseg.nii'
vol = reader.read_images(vol_path)[0].astype(np.float32)
mask = reader.read_images(mask_path)[0]


sample = df[df.SeriesInstanceUID==uid]
modality = sample.Modality.values[0]
sample


modality


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


sliding_window_inferer = SlidingWindowInfererAdapt(
            roi_size=CFG.patch_size, sw_batch_size=1, overlap=0.5,
        )


T = _generate_transforms((128, 384, 384), (128, 128, 128), 'val')
_input = {'Image': vol, 'Mask': mask, 'modality': modality}


o = T(_input)
_vol, _mask = o['Image'].to(CFG.device), o['Mask']


mip, _ = _vol[0].max(dim=0)
mip = mip.cpu()


with torch.no_grad():
    pred_logits = sliding_window_inferer(_vol[None], model)


pred_probs = pred_logits.softmax(dim=1)
pred_mask = pred_logits.argmax(dim=1)
mask_mip, _ = pred_mask[0].max(dim=0)
mask_mip = mask_mip.cpu()


np.unique(df.Modality.values)


plt.imshow(_vol[0].max(dim=0)[0].cpu())


_mask_mip = _mask[0].max(dim=0)[0]


pred_mask.shape


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


_, new_mask_mip = cca_remove(pred_mask[0].cpu().numpy())


fig, ax = plt.subplots(ncols=4, figsize = (16, 16))
ax[0].imshow(vol.max(axis=0))
ax[1].imshow(mask_mip)
ax[1].set_title('before process')
ax[2].imshow(new_mask_mip)
ax[2].set_title('after process')
ax[3].imshow(_mask_mip)
ax[3].set_title('ground truth')


mask_mip.unique()


np.unique(new_mask_mip)


n_classes = 13
cols = 3
rows = int(np.ceil(n_classes / cols))

fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(cols*4, rows*4))

# Flatten the axes array for easy indexing
ax = ax.flatten()

for i in range(n_classes):
    ax[i].imshow(new_mask_mip == i+1, cmap='gray')
    ax[i].imshow(mip, alpha=0.5)
    ax[i].set_title(f"Class {i+1}")
    ax[i].axis('off')

# Hide unused subplots (if n_classes is not a multiple of cols)
for j in range(n_classes, len(ax)):
    ax[j].axis('off')

plt.tight_layout()
plt.show()


from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2


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


MAX_WORKERS = 4


df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
df_loc = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')
seg_path = Path('/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations')
series_path = Path('/kaggle/input/rsna-intracranial-aneurysm-detection/series')
seg_uids = [name.split('.nii')[0] for name in os.listdir(seg_path) if 'cowseg' not in name]


uid = '1.2.826.0.1.3680043.8.498.10092666779602341135460882241562348436'
path = series_path/uid
mask_path = seg_path/f'{uid}_cowseg.nii'
vol_path = seg_path/f'{uid}.nii'
df[df.SeriesInstanceUID	== uid]


transforms = generate_transforms(CFG.transforms_config)


vol = load_dicom_series(path)
vol = np.flip(vol, axis=1).copy()
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


_, new_mask_mip = cca_remove(pred_mask[0].cpu().numpy())


fig, ax = plt.subplots(ncols=3, figsize = (16, 16))
ax[0].imshow(test_mip)
ax[1].imshow(mask_mip)
ax[1].set_title('before process')
ax[2].imshow(new_mask_mip)
ax[2].set_title('after process')




