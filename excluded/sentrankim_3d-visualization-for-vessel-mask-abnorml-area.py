!pip install pyvista


import os
import glob
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.ndimage import binary_dilation

import pydicom
import nibabel as nib
import SimpleITK as sitk
import pyvista as pv



class config:
    root_dicom_folder ='/kaggle/input/rsna-intracranial-aneurysm-detection/series/'
    root_mask = '/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/'
    


import pandas as pd
train_localizer = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv')
train_localizer['path_folder_dicom'] = train_localizer['SeriesInstanceUID'].apply(lambda uid: config.root_dicom_folder + uid )
train_localizer['path_slice'] = train_localizer.apply(lambda row: config.root_dicom_folder + row.SeriesInstanceUID + '/' + row.SOPInstanceUID + '.dcm',axis=1)
train_localizer['path_seg_mask'] = train_localizer['SeriesInstanceUID'].apply(lambda name: config.root_mask + name + '/' +  name + '_cowseg.nii')

## only keep vessel mask

list_uid = os.listdir('/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations')
train_localizer = train_localizer[train_localizer['SeriesInstanceUID'].isin(list_uid)]



import SimpleITK as sitk
import numpy as np
from pathlib import Path

def load_dicom_volume_sitk(series_dir):
    """
    Load a DICOM series into a NumPy volume using SimpleITK.
    
    Args:
        series_dir (str or Path): Path to folder containing a DICOM series.
    
    Returns:
        volume (np.ndarray): 3D array (z, y, x) in float32.
        spacing (tuple): Physical spacing between voxels (z, y, x) in mm.
        origin (tuple): Physical origin (x, y, z) in mm.
        direction (tuple): Direction cosines.
    """
    series_dir = Path(series_dir)
    reader = sitk.ImageSeriesReader()
    
    dicom_names = reader.GetGDCMSeriesFileNames(str(series_dir))
    if not dicom_names:
        raise FileNotFoundError(f"No DICOM files found in {series_dir}")
    
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    # Convert to NumPy array (z, y, x)
    # volume = sitk.GetArrayFromImage(image).astype(np.float32)
    
    return image


from pathlib import Path
import SimpleITK as sitk
import torch
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import generate_binary_structure

import numpy as np
import pydicom
from pathlib import Path
from scipy.ndimage import binary_dilation

def spherical_struct(radius):
    """
    Create a spherical structuring element for 3D dilation.

    Args:
        radius (int): Radius in voxels.

    Returns:
        np.ndarray: 3D spherical structuring element (bool array).
    """
    L = np.arange(-radius, radius + 1)
    X, Y, Z = np.meshgrid(L, L, L, indexing="ij")
    return (X**2 + Y**2 + Z**2) <= radius**2

def build_mask_for_volume(path_dcm, path_slices, coords, dilation_radius=10):
    """
    Build a 3D binary mask for a DICOM volume given labeled slice coordinates.
    
    Parameters:
        path_dcm (str): Path to DICOM folder for the volume.
        path_slices (list[str]): Paths to the DICOM slice files containing labels.
        coords (list[tuple]): (x, y) coordinates for each labeled slice.
        dilation_radius (int): Dilation radius in voxels.
    
    Returns:
        np.ndarray: 3D mask (z, y, x)
    """
    path_dcm = Path(path_dcm)
    dicom_files = sorted(path_dcm.glob("*.dcm"))
    if not dicom_files:
        raise FileNotFoundError(f"No DICOM files found in {path_dcm}")

    # Read metadata for size
    ds0 = pydicom.dcmread(str(dicom_files[0]), stop_before_pixels=True)
    rows, cols = ds0.Rows, ds0.Columns
    num_slices = len(dicom_files)

    # Map SOPInstanceUID to slice index
    sop_uid_to_index = {}
    for idx, f in enumerate(dicom_files):
        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        sop_uid_to_index[ds.SOPInstanceUID] = idx

    # Initialize empty mask
    mask = np.zeros((num_slices, rows, cols), dtype=np.uint8)

    # Assign points to mask
    for path_slice, point in zip(path_slices, coords):
        ds = pydicom.dcmread(str(path_slice), stop_before_pixels=True)
        # print(ds.SOPInstanceUID,sop_uid_to_index)
        sop_uid = ds.SOPInstanceUID
        if sop_uid not in sop_uid_to_index:
            continue
        z_idx = sop_uid_to_index[sop_uid]

        # Ensure point is tuple
        if isinstance(point, str):
            point = eval(point)
        x, y = int(point[0]), int(point[1])
        if 0 <= y < rows and 0 <= x < cols:
            mask[z_idx, y, x] = 1
        # mask[z_idx, ...] = dilate_mask_2d(mask[z_idx, ...],radius=dilation_radius)

    # Dilate mask in 3D
    if dilation_radius > 0:
        from scipy.ndimage import generate_binary_structure
        # struct = generate_binary_structure(3, 1)
        struct = spherical_struct(dilation_radius)
        mask = binary_dilation(mask, structure=struct, iterations=dilation_radius) #.astype(np.uint8)

    return mask

import numpy as np
import plotly.graph_objects as go

def vis_3d_gpu(vol_3d: np.ndarray, mask_3d: np.ndarray, sample_rate: float = 1.0):
    """
    GPU-accelerated 3D visualization using Plotly WebGL backend.
    Shows volume and mask using voxel coordinates.
    
    Parameters:
        vol_3d (np.ndarray): Binary vessel volume.
        mask_3d (np.ndarray): Binary abnormal region mask.
        sample_rate (float): Fraction of voxels to visualize for performance (0.1–0.5 is typical).
    """
    assert vol_3d.shape == mask_3d.shape, "Volume and mask must have the same shape"

    # Get voxel positions
    vol_coords = np.argwhere(vol_3d > 0)
    mask_coords = np.argwhere(mask_3d > 0)

    # Optionally subsample for speed
    if sample_rate < 1.0:
        vol_coords = vol_coords[np.random.choice(len(vol_coords), int(len(vol_coords) * sample_rate), replace=False)]
        mask_coords = mask_coords[np.random.choice(len(mask_coords), int(len(mask_coords) * sample_rate), replace=False)]

    fig = go.Figure()

    # Vessel volume points
    if len(vol_coords) > 0:
        fig.add_trace(go.Scatter3d(
            x=vol_coords[:, 2], y=vol_coords[:, 1], z=vol_coords[:, 0],
            mode='markers',
            marker=dict(size=2, color='lightgray', opacity=0.2),
            name='Vessel Volume'
        ))

    # Abnormal mask points
    if len(mask_coords) > 0:
        fig.add_trace(go.Scatter3d(
            x=mask_coords[:, 2], y=mask_coords[:, 1], z=mask_coords[:, 0],
            mode='markers',
            marker=dict(size=2, color='red', opacity=0.7),
            name='Abnormal Mask'
        ))

    fig.update_layout(
        scene=dict(aspectmode='data'),
        margin=dict(l=0, r=0, b=0, t=30),
        title="GPU-accelerated 3D visualization (Plotly WebGL)",
        showlegend=True
    )

    fig.show()



# seg_mask = nii.get_fdata()


# seg_mask.shape


mask_3d = build_mask_for_volume(path_dcm, path_slices, coords,dilation_radius=2)
print(mask_3d.shape)


mask_3d = mask_3d[::-1, :, :]


mask_3d.shape


 seg_mask = np.transpose(seg_mask, (2,0,1))



seg_mask.shape


 vis_3d_gpu(seg_mask, mask_3d)


def numpy_to_sitk_mask(mask_np, reference_image):
    sitk_mask = sitk.GetImageFromArray(mask_np.astype(np.uint8))  # (z, y, x)
    sitk_mask.CopyInformation(reference_image)
    return sitk_mask


from tqdm import tqdm

for uid, sub in tqdm(train_localizer.groupby('path_folder_dicom')):
    
    coords = sub['coordinates'].tolist()
    coords = [(int(eval(cood)['x']),int(eval(cood)['y'])) for  cood in coords]
    
    path_dcm = sub['path_folder_dicom'].iloc[0]
    path_slices = sub['path_slice'].tolist()

    # numpy 3d mask
    mask_3d = build_mask_for_volume(path_dcm, path_slices, coords,dilation_radius=2)


    volume_3d = load_dicom_volume_sitk(path_dcm)
    
    
    # nii = nib.load(sub['path_seg_mask'].values[0])

    seg_sitk = sitk.ReadImage(sub['path_seg_mask'].values[0])

    sitk_mask_3d = numpy_to_sitk_mask(mask_3d, seg_sitk)
    resampled_mask = sitk.Resample(sitk_mask_3d, seg_sitk, sitk.Transform(), sitk.sitkNearestNeighbor, 0)

    vol_3d = sitk.Resample(volume_3d, seg_sitk, sitk.Transform(), sitk.sitkNearestNeighbor, 0)
    volume_3d = sitk.GetArrayFromImage(volume_3d).astype(np.float32)
    
    # Convert back to numpy for visualization
    abnormal_mask = sitk.GetArrayFromImage(resampled_mask)
    
    seg_mask = sitk.GetArrayFromImage(seg_sitk)
    
    # seg_mask = nii.get_fdata()
    # seg_mask = np.transpose(seg_mask, (2,1,0))
    seg_mask = seg_mask[:, ::-1, :]


    # vis_3d_gpu(seg_mask.astype(np.uint8), volume_3d.astype(np.uint8))

    vis_3d_gpu(seg_mask, abnormal_mask)
    
    print('done')
    break


volume_3d = (volume_3d-volume_3d.min())/(volume_3d.max()-volume_3d.min())*255


vis_3d_gpu(seg_mask, volume_3d)




