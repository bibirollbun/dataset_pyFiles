from skimage.filters import sobel, frangi
from skimage.morphology import white_tophat, disk
from scipy.ndimage import gaussian_gradient_magnitude
from __future__ import annotations
import os
import ast
import math
import multiprocessing as mp
from pathlib import Path
import sys
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
import pydicom
import cv2
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt

try:
    from scipy.ndimage import zoom as nd_zoom
except ImportError:
    raise SystemExit("scipy is required. Install via: pip install scipy")
    
#current_dir = Path(__file__).parent
#parent_dir = current_dir.parent
#sys.path.insert(0, str(parent_dir))

TARGET_DEPTH = 32
TARGET_SIZE = 384
HU_MIN = -1200.0
HU_MAX = 4000.0
STORE_NORMALIZED = False  # Set True to revert to [0,1] scaling

# Globals for worker processes

data_path = '/kaggle/input/rsna-intracranial-aneurysm-detection'


import SimpleITK as sitk
from skimage.filters import frangi
from skimage.morphology import skeletonize, remove_small_objects
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter


data_path = '/kaggle/input/rsna-intracranial-aneurysm-detection'
windows = {
    'CT': (40, 80),
    'CTA': (50, 350),
    'MRA': (600, 1200),
    'MR': (600, 1200),
    'MRI': (40, 80),
}

LABELS_TO_IDX = {
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

IMG_SIZE = 512
FACTOR = 1
SEED = 42
N_FOLDS = 5
CORES = 4


def _load_series_dicom_paths(series_uid: str, root: Path) -> List[Path]:
    series_dir = root / 'series' / series_uid
    paths = []
    for r, _, files in os.walk(series_dir):
        for f in files:
            if f.endswith('.dcm'):
                paths.append(Path(r) / f)
    return paths


def _read_dicom(path: Path):
    ds = pydicom.dcmread(str(path), force=True)
    arr = ds.pixel_array.astype(np.float32)
    if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
        arr = arr * float(ds.RescaleSlope) + float(ds.RescaleIntercept)
    return ds, arr


def _extract_slice_position(ds) -> float:
    # Prefer ImagePositionPatient z, fallback to InstanceNumber
    if hasattr(ds, 'ImagePositionPatient') and len(ds.ImagePositionPatient) == 3:
        try:
            return float(ds.ImagePositionPatient[2])
        except Exception:
            pass
    if hasattr(ds, 'InstanceNumber'):
        try:
            return float(ds.InstanceNumber)
        except Exception:
            pass
    return 0.0


def _resample_depth(volume: np.ndarray, target_depth: int) -> np.ndarray:
    if volume.shape[0] == target_depth:
        return volume
    depth_zoom = target_depth / volume.shape[0]
    # zoom along depth only, order=1 linear
    return nd_zoom(volume, (depth_zoom, 1.0, 1.0), order=1)


def _resize_inplane(volume: np.ndarray, target_hw: int) -> np.ndarray:
    d, h, w = volume.shape
    if h == target_hw and w == target_hw:
        return volume
    resized = np.empty((d, target_hw, target_hw), dtype=volume.dtype)
    for i in range(d):
        resized[i] = cv2.resize(volume[i], (target_hw, target_hw), interpolation=cv2.INTER_LINEAR)
    return resized


def _clip_or_normalize(volume: np.ndarray) -> np.ndarray:
    """Either clip-only (raw HU retained in range) or clip+normalize to [0,1]."""
    vol = np.clip(volume, HU_MIN, HU_MAX).astype(np.float32)
    if STORE_NORMALIZED:
        vol = (vol - HU_MIN) / (HU_MAX - HU_MIN)
    return vol


def extract_details(vol):
    mip = vol.max(axis=0, keepdims=True)
    std_proj = vol.std(axis=0, keepdims=True)
    edges = np.stack([sobel(slice_) for slice_ in vol], axis=0)
    edge_proj = edges.max(axis=0, keepdims=True)
    vesselness = np.stack([frangi(slice_) for slice_ in vol], axis=0)
    vessel_proj = vesselness.max(axis=0, keepdims=True)
    gradmag = np.stack([gaussian_gradient_magnitude(slice_, sigma=1) for slice_ in vol], axis=0)
    grad_proj = gradmag.max(axis=0, keepdims=True)
    extracted_vol = np.concatenate([vol, mip, std_proj, edge_proj, vessel_proj, grad_proj], axis=0) #(32 + 5, 384, 384)
    return extracted_vol


def _process_single_series(uid: str, root: Path) -> Dict[str, Any]:
    try:
        dcm_paths = _load_series_dicom_paths(uid, root)
        if not dcm_paths:
            return {"series_uid": uid, "volume_filename": None, "num_slices_raw": 0}
        slices: List[Tuple[float, np.ndarray]] = []
        for p in dcm_paths:
            try:
                ds, arr = _read_dicom(p)
                # If multi-frame (arr.ndim==3) stack frames individually
                if arr.ndim == 3 and arr.shape[-1] != 3:
                    for fi in range(arr.shape[0]):
                        slices.append((_extract_slice_position(ds) + fi * 0.001, arr[fi].astype(np.float32)))
                else:
                    if arr.ndim == 3 and arr.shape[-1] == 3:
                        # Convert RGB to grayscale
                        arr = cv2.cvtColor(arr.astype(np.float32), cv2.COLOR_BGR2GRAY)
                    slices.append((_extract_slice_position(ds), arr.astype(np.float32)))
            except Exception:
                continue
        if not slices:
            return {"series_uid": uid, "volume_filename": None, "num_slices_raw": 0}
        # Sort by z
        slices.sort(key=lambda x: x[0])
        vol = np.stack([s[1] for s in slices], axis=0)  # (D, H, W)
        num_raw = vol.shape[0]
        # Clip HU range (optionally normalize based on STORE_NORMALIZED)
        vol = _clip_or_normalize(vol)
        # Depth resample
        vol = _resample_depth(vol, TARGET_DEPTH)
        # In-plane resize
        vol = _resize_inplane(vol, TARGET_SIZE)
        # Save
        vol_filename = f"{uid}_d{TARGET_DEPTH}_sz{TARGET_SIZE}.npz"
        # Save meta: [HU_MIN, HU_MAX, normalized_flag]
        meta = np.array([HU_MIN, HU_MAX, 1.0 if STORE_NORMALIZED else 0.0], dtype=np.float32)
        return {"series_uid": uid, "volume_filename": vol_filename, "num_slices_raw": num_raw, 'volume': vol, 'meta': meta}
    except Exception as e:
        return {"series_uid": uid, "volume_filename": None, "error": str(e), "num_slices_raw": 0}



root = Path(data_path)
processed = root / 'processed'
vol_dir = processed / 'volumes_3d'
train_df = pd.read_csv(root / 'train.csv')
label_df = pd.read_csv(root / 'train_localizers.csv')
mf_dicom_uids = pd.read_csv(root / 'multiframe_dicoms.csv') if (root / 'multiframe_dicoms.csv').exists() else pd.DataFrame(columns=['SeriesInstanceUID'])

ignore_uids = set([
    '1.2.826.0.1.3680043.8.498.11145695452143851764832708867797988068',
    '1.2.826.0.1.3680043.8.498.35204126697881966597435252550544407444',
    '1.2.826.0.1.3680043.8.498.87480891990277582946346790136781912242',
]) | set(mf_dicom_uids['SeriesInstanceUID'].tolist())

train_df = train_df[~train_df['SeriesInstanceUID'].isin(ignore_uids)].reset_index(drop=True)
train_df['fold_id'] = 0

skf = StratifiedKFold(n_splits=N_FOLDS, random_state=SEED, shuffle=True)
for fold, (_, val_idx) in enumerate(skf.split(train_df['SeriesInstanceUID'], train_df['Aneurysm Present'])):
    train_df.loc[val_idx, 'fold_id'] = fold

uids = train_df['SeriesInstanceUID'].unique().tolist()
print(f"Preparing 3D volumes for {len(uids)} series -> target shape ({TARGET_DEPTH}, {TARGET_SIZE}, {TARGET_SIZE})")


uid = uids[-1]
series_dict  = _process_single_series(uid, root)
vol = series_dict['volume']


import SimpleITK as sitk


def vessel_enhancement(vol, sigmas=(1, 3)):
    """
    Apply 3D Frangi filter to enhance vessel-like structures.
    vol: (D, H, W) numpy array
    """
    # skimage.frangi supports 3D directly if input is 3D
    ves = frangi(vol, sigmas=sigmas, black_ridges=False)
    ves = (ves - ves.min()) / (ves.max() - ves.min() + 1e-8)  # normalize [0,1]
    return ves

# ---------------------------
# 2. Vessel skeletonization
# ---------------------------
def vessel_skeleton(ves, thresh=0.2):
    """
    Threshold vesselness map and apply 3D skeletonization.
    """
    mask = ves > thresh
    skel = skeletonize(mask.astype(np.uint8))
    return mask, skel

# ---------------------------
# 3. Registration to atlas
# ---------------------------
def register_to_atlas(atlas_np, moving_np, atlas_spacing=(1.0,1.0,1.0), moving_spacing=(1.0,1.0,1.0), do_bspline=True):
    """
    Register moving volume to atlas using Rigid -> Affine -> (optional B-spline).
    Returns: final_transform, fixed_atlas_img
    """

    # Convert numpy → SimpleITK images
    atlas_img = sitk.GetImageFromArray(atlas_np.astype(np.float32))
    moving_img = sitk.GetImageFromArray(moving_np.astype(np.float32))
    atlas_img.SetSpacing(atlas_spacing)
    moving_img.SetSpacing(moving_spacing)

    # --- Rigid registration ---
    rigid_reg = sitk.ImageRegistrationMethod()
    rigid_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    rigid_reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200)
    rigid_reg.SetInterpolator(sitk.sitkLinear)
    rigid_tx = sitk.VersorRigid3DTransform()
    rigid_reg.SetInitialTransform(rigid_tx)
    rigid_tx = rigid_reg.Execute(atlas_img, moving_img)

    # --- Affine registration (fixed) ---
    affine_reg = sitk.ImageRegistrationMethod()
    affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
    affine_reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=200)
    affine_reg.SetInterpolator(sitk.sitkLinear)

    # Initialize affine from rigid
    affine_tx = sitk.AffineTransform(3)
    affine_tx.SetMatrix(rigid_tx.GetMatrix())
    affine_tx.SetTranslation(rigid_tx.GetTranslation())

    affine_reg.SetInitialTransform(affine_tx, inPlace=False)
    affine_tx = affine_reg.Execute(atlas_img, moving_img)

    final_tx = affine_tx

    # --- Optional BSpline refinement ---
    if do_bspline:
        # Setup BSpline registration
        grid_physical_spacing = [50.0, 50.0, 50.0]  # adjust to your volume size
        image_spacing = atlas_img.GetSpacing()
        image_size = atlas_img.GetSize()
        mesh_size = [int(sz*spc/gsp + 0.5)
                     for sz, spc, gsp in zip(image_size, image_spacing, grid_physical_spacing)]
        bspline_tx = sitk.BSplineTransformInitializer(atlas_img, mesh_size)
    
        bspline_reg = sitk.ImageRegistrationMethod()
        bspline_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=32)
        bspline_reg.SetOptimizerAsGradientDescent(learningRate=0.1,
                                                  numberOfIterations=200,
                                                  convergenceMinimumValue=1e-6,
                                                  convergenceWindowSize=10)
        bspline_reg.SetInterpolator(sitk.sitkLinear)
        bspline_reg.SetInitialTransform(bspline_tx, inPlace=False)
    
        bspline_tx = bspline_reg.Execute(atlas_img, moving_img)
    
        # ✅ Use CompositeTransform to combine affine + bspline
        final_tx = sitk.CompositeTransform(3)
        final_tx.AddTransform(affine_tx)
        final_tx.AddTransform(bspline_tx)
    else:
        final_tx = affine_tx

    return final_tx, atlas_img


# Step 1: Vessel enhancement
ves = vessel_enhancement(vol, sigmas=(1,2,3))

# Step 2: Skeleton
mask, skel = vessel_skeleton(ves, thresh=0.2)

# Step 3: Registration (atlas must be provided)
atlas_vol = np.random.rand(32,384,384).astype(np.float32)
final_tx, atlas_img = register_to_atlas(atlas_vol, vol)

# Visualization
proj_ves = ves.max(axis=0)
proj_skel = skel.max(axis=0)

plt.figure()
plt.subplot(1,2,1); plt.imshow(proj_ves, cmap="gray"); plt.title("Vesselness Projection")
plt.subplot(1,2,2); plt.imshow(proj_skel, cmap="gray"); plt.title("Skeleton Projection")
plt.show()




