!pip install monai


from monai.inferers import sliding_window_inference
import numpy as np
import os
from torch.utils.data import Dataset
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, EnsureTyped,ConcatItemsd,ToTensord,SpatialPadd,Lambda,
    Spacingd, Resized, RandFlipd, RandRotate90d, RandScaleIntensityd ,RandCropByLabelClassesd, MapTransform
)
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np
from monai.networks.nets import DynUNet
import sys
from pathlib import Path
from typing import Tuple
import pandas as pd 
import SimpleITK as sitk
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import nibabel as nib



from collections import OrderedDict
from typing import Tuple, List


import pydicom
import cv2
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from scipy import ndimage
import warnings
import gc
import pandas as pd
warnings.filterwarnings('ignore')

class DICOMPreprocessorKaggle:
    """
    DICOM preprocessing system for Kaggle Code Competition
    Converts original DICOMPreprocessor logic to single series processing
    """
    
    def __init__(self, target_shape: Tuple[int, int, int] = (128, 128, 128)):
        self.target_depth, self.target_height, self.target_width = target_shape
        
    def load_dicom_series(self, series_path: str) -> Tuple[List[pydicom.Dataset], str]:
        """
        Load DICOM series
        """
        series_path = Path(series_path)
        series_name = series_path.name
        
        # Search for DICOM files
        dicom_files = []
        for root, _, files in os.walk(series_path):
            for file in files:
                if file.endswith('.dcm'):
                    dicom_files.append(os.path.join(root, file))
        
        if not dicom_files:
            raise ValueError(f"No DICOM files found in {series_path}")
        
        #print(f"Found {len(dicom_files)} DICOM files in series {series_name}")
        
        # Load DICOM datasets
        datasets = []
        for filepath in dicom_files:
            try:
                ds = pydicom.dcmread(filepath, force=True)
                datasets.append(ds)
            except Exception as e:
                #print(f"Failed to load {filepath}: {e}")
                continue
        
        if not datasets:
            raise ValueError(f"No valid DICOM files in {series_path}")
        
        return datasets, series_name
    
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
                    #print("ImagePositionPatient not found, using InstanceNumber")
            except Exception as e:
                info['z_position'] = float(i)
                #print(f"Failed to extract position info: {e}")
            
            slice_info.append(info)
        
        return slice_info
    
    def sort_slices_by_position(self, slice_info: List[Dict]) -> List[Dict]:
        """
        Sort slices by z-coordinate
        """
        # Sort by z-coordinate
        sorted_slices = sorted(slice_info, key=lambda x: x['z_position'])
        
        #print(f"Sorted {len(sorted_slices)} slices by z-position")
        #print(f"Z-range: {sorted_slices[0]['z_position']:.2f} to {sorted_slices[-1]['z_position']:.2f}")
        
        return sorted_slices
    
    def get_windowing_params(self, ds: pydicom.Dataset, img: np.ndarray = None) -> Tuple[Optional[float], Optional[float]]:
        """
        Get windowing parameters based on modality
        """
        modality = getattr(ds, 'Modality', 'CT')
        
        if modality == 'CT':
            # For CT, apply CTA (angiography) settings
            center, width = (50, 350)
            #print(f"Using CTA windowing for CT: Center={center}, Width={width}")
            # return center, width
            return None, None
            
        elif modality == 'MR':
            # For MR, skip windowing (statistical normalization only)
            #print("MR modality detected: skipping windowing, using statistical normalization")
            return None, None
            
        else:
            # Unexpected modality (safety measure)
            #print(f"Unexpected modality '{modality}', using CTA windowing")
            #return (50, 350)
            return None, None
    
    def apply_windowing_or_normalize(self, img: np.ndarray, center: Optional[float], width: Optional[float]) -> np.ndarray:
        """
        Apply windowing or statistical normalization
        """
        p1, p99 = np.percentile(img, [1, 99])
        volume = np.clip(img, p1, p99)
        volume = (volume - p1) / (p99 - p1 + 1e-7)
        volume = (volume * 255).astype(np.uint8)
        return volume
        # if center is not None and width is not None:
        #     # # Windowing processing (for CT/CTA)
        #     # img_min = center - width / 2
        #     # img_max = center + width / 2
            
        #     # windowed = np.clip(img, img_min, img_max)
        #     # windowed = (windowed - img_min) / (img_max - img_min + 1e-7)
        #     # result = (windowed * 255).astype(np.uint8)
            
        #     # #print(f"Applied windowing: [{img_min:.1f}, {img_max:.1f}] → [0, 255]")
        #     # return result
            
        #     # Statistical normalization (for CT as well)
        #     # Normalize using 1-99 percentiles
        #     p1, p99 = np.percentile(img, [1, 99])
        #     # p1, p99 = 0, 500
            
        #     if p99 > p1:
        #         normalized = np.clip(img, p1, p99)
        #         normalized = (normalized - p1) / (p99 - p1)
        #         result = (normalized * 255).astype(np.uint8)
                
        #         #print(f"Applied statistical normalization: [{p1:.1f}, {p99:.1f}] → [0, 255]")
        #         return result
        #     else:
        #         # Fallback: min-max normalization
        #         img_min, img_max = img.min(), img.max()
        #         if img_max > img_min:
        #             normalized = (img - img_min) / (img_max - img_min)
        #             result = (normalized * 255).astype(np.uint8)
        #             #print(f"Applied min-max normalization: [{img_min:.1f}, {img_max:.1f}] → [0, 255]")
        #             return result
        #         else:
        #             # If image has no variation
        #             #print("Image has no variation, returning zeros")
        #             return np.zeros_like(img, dtype=np.uint8)
        
        # else:
        #     # Statistical normalization (for MR)
        #     # Normalize using 1-99 percentiles
        #     p1, p99 = np.percentile(img, [1, 99])
            
        #     if p99 > p1:
        #         normalized = np.clip(img, p1, p99)
        #         normalized = (normalized - p1) / (p99 - p1)
        #         result = (normalized * 255).astype(np.uint8)
                
        #         #print(f"Applied statistical normalization: [{p1:.1f}, {p99:.1f}] → [0, 255]")
        #         return result
        #     else:
        #         # Fallback: min-max normalization
        #         img_min, img_max = img.min(), img.max()
        #         if img_max > img_min:
        #             normalized = (img - img_min) / (img_max - img_min)
        #             result = (normalized * 255).astype(np.uint8)
        #             #print(f"Applied min-max normalization: [{img_min:.1f}, {img_max:.1f}] → [0, 255]")
        #             return result
        #         else:
        #             # If image has no variation
        #             #print("Image has no variation, returning zeros")
        #             return np.zeros_like(img, dtype=np.uint8)
    
    def extract_pixel_array(self, ds: pydicom.Dataset) -> np.ndarray:
        """
        Extract 2D pixel array from DICOM and apply preprocessing (for 2D DICOM series)
        """
        # Get pixel data
        img = ds.pixel_array.astype(np.float32)
        
        # For 3D volume case (multiple frames) - select middle frame
        if img.ndim == 3:
            #print(f"3D DICOM in 2D processing - using middle frame from shape: {img.shape}")
            frame_idx = img.shape[0] // 2
            img = img[frame_idx]
            #print(f"Selected frame {frame_idx} from 3D DICOM")
        
        # Convert color image to grayscale
        # if img.ndim == 3 and img.shape[-1] == 3:
            # img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
            #print("Converted color image to grayscale")
        
        # Apply RescaleSlope and RescaleIntercept
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        # slope, intercept = 1, 0
        if slope != 1 or intercept != 0:
            img = img * float(slope) + float(intercept)
            #print(f"Applied rescaling: slope={slope}, intercept={intercept}")
        
        return img
    
    def resize_volume_3d(self, volume: np.ndarray) -> np.ndarray:
        """
        Resize 3D volume to target size
        """
        current_shape = volume.shape
        target_shape = (self.target_depth, self.target_height, self.target_width)
        
        if current_shape == target_shape:
            return volume
        
        #print(f"Resizing volume from {current_shape} to {target_shape}")
        
        # 3D resizing using scipy.ndimage
        zoom_factors = [
            target_shape[i] / current_shape[i] for i in range(3)
        ]
        
        # Resize with linear interpolation
        resized_volume = ndimage.zoom(volume, zoom_factors, order=1, mode='nearest')
        
        # Clip to exact size just in case
        resized_volume = resized_volume[:self.target_depth, :self.target_height, :self.target_width]
        
        # Padding if necessary
        pad_width = [
            (0, max(0, self.target_depth - resized_volume.shape[0])),
            (0, max(0, self.target_height - resized_volume.shape[1])),
            (0, max(0, self.target_width - resized_volume.shape[2]))
        ]
        
        if any(pw[1] > 0 for pw in pad_width):
            resized_volume = np.pad(resized_volume, pad_width, mode='edge')
        
        #print(f"Final volume shape: {resized_volume.shape}")
        return resized_volume.astype(np.uint8)
    
    def process_series(self, series_path: str) -> np.ndarray:
        """
        Process DICOM series and return as NumPy array (for Kaggle: no file saving)
        """
        try:
            # 1. Load DICOM files
            datasets, series_name = self.load_dicom_series(series_path)
            
            # Check first DICOM to determine 3D/2D
            first_ds = datasets[0]
            first_img = first_ds.pixel_array
            
            if len(datasets) == 1 and first_img.ndim == 3:
                # Case 1: Single 3D DICOM file
                #print(f"Processing single 3D DICOM with shape: {first_img.shape}")
                return self._process_single_3d_dicom(first_ds, series_name)
            else:
                # Case 2: Multiple 2D DICOM files
                #print(f"Processing {len(datasets)} 2D DICOM files")
                return self._process_multiple_2d_dicoms(datasets, series_name)
            
        except Exception as e:
            #print(f"Failed to process series {series_path}: {e}")
            raise
    
    def _process_single_3d_dicom(self, ds: pydicom.Dataset, series_name: str) -> np.ndarray:
        """
        Process single 3D DICOM file (for Kaggle: no file saving)
        """
        # Get pixel array
        volume = ds.pixel_array.astype(np.float32)
        
        # Apply RescaleSlope and RescaleIntercept
        slope = getattr(ds, 'RescaleSlope', 1)
        intercept = getattr(ds, 'RescaleIntercept', 0)
        slope, intercept = 1, 0
        if slope != 1 or intercept != 0:
            volume = volume * float(slope) + float(intercept)
            # #print(f"Applied rescaling: slope={slope}, intercept={intercept}")
        
        # Get windowing settings
        window_center, window_width = self.get_windowing_params(ds)
        
        # Apply windowing to each slice
        processed_slices = []
        for i in range(volume.shape[0]):
            slice_img = volume[i]
            processed_img = self.apply_windowing_or_normalize(slice_img, window_center, window_width)
            processed_slices.append(processed_img)
        
        volume = np.stack(processed_slices, axis=0)
        ##print(f"3D volume shape after windowing: {volume.shape}")
        
        # 3D resize
        final_volume = self.resize_volume_3d(volume)
        
        ##print(f"Successfully processed 3D DICOM series {series_name}")
        return final_volume
    
    def _process_multiple_2d_dicoms(self, datasets: List[pydicom.Dataset], series_name: str) -> np.ndarray:
        """
        Process multiple 2D DICOM files (for Kaggle: no file saving)
        """
        slice_info = self.extract_slice_info(datasets)
        sorted_slices = self.sort_slices_by_position(slice_info)
        first_img = self.extract_pixel_array(sorted_slices[0]['dataset'])
        window_center, window_width = self.get_windowing_params(sorted_slices[0]['dataset'], first_img)
        processed_slices = []
        
        for slice_data in sorted_slices:
            ds = slice_data['dataset']
            img = self.extract_pixel_array(ds)
            processed_img = self.apply_windowing_or_normalize(img, window_center, window_width)
            resized_img = cv2.resize(processed_img, (self.target_width, self.target_height))
            
            processed_slices.append(resized_img)

        volume = np.stack(processed_slices, axis=0)
        ##print(f"2D slices stacked to volume shape: {volume.shape}")
        final_volume = self.resize_volume_3d(volume)
        
        ##print(f"Successfully processed 2D DICOM series {series_name}")
        return final_volume

def process_dicom_series_kaggle(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)) -> np.ndarray:
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
def process_dicom_series_safe(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)) -> np.ndarray:
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
def test_single_series(series_path: str, target_shape: Tuple[int, int, int] = (32, 384, 384)):
    """
    Test processing for single series
    """
    try:
        #print(f"Testing single series: {series_path}")
        
        # Execute processing
        volume = process_dicom_series_safe(series_path, target_shape)
        
        # Display results
        #print(f"✓ Successfully processed series")
        #print(f"  Volume shape: {volume.shape}")
        #print(f"  Volume dtype: {volume.dtype}")
        #print(f"  Volume range: [{volume.min()}, {volume.max()}]")
        
        return volume
        
    except Exception as e:
        #print(f"✗ Failed to process series: {e}")
        return None


from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import os
import pydicom

def load_dicom_series(series_dir: Path) -> Optional[np.ndarray]:
    """
    Load a DICOM series as a 3D volume (D,H,W) in HU.
    Skips series that do not have required metadata.

    Args:
        series_dir (Path): path to folder containing DICOM files

    Returns:
        volume (np.ndarray) or None: 3D volume, or None if metadata missing
    """
    dcm_paths = [Path(series_dir) / f for f in os.listdir(series_dir) if f.lower().endswith(".dcm")]
    if not dcm_paths:
        return None

    slices = [pydicom.dcmread(str(p), force=True) for p in dcm_paths]

    # Skip if metadata missing
    if not (hasattr(slices[0], "ImagePositionPatient") and hasattr(slices[0], "ImageOrientationPatient")):
        return None

    try:
        orientation = np.array(slices[0].ImageOrientationPatient).reshape(2, 3)
        row_cos, col_cos = orientation
        normal = np.cross(row_cos, col_cos)
        slices.sort(key=lambda ds: np.dot(np.array(ds.ImagePositionPatient), normal))
    except Exception:
        return None

    # Apply HU scaling
    slope = float(getattr(slices[0], "RescaleSlope", 1.0))
    intercept = float(getattr(slices[0], "RescaleIntercept", 0.0))
    slice_arrays = [ds.pixel_array.astype(np.float32) * slope + intercept for ds in slices]

    # Ensure all slices have the same shape
    shapes = [s.shape for s in slice_arrays]
    if len(set(shapes)) > 1:
        max_h = max(s[0] for s in shapes)
        max_w = max(s[1] for s in shapes)
        slice_arrays = [np.pad(s, ((0,max_h-s.shape[0]), (0,max_w-s.shape[1])), mode='constant') 
                        if s.shape != (max_h,max_w) else s 
                        for s in slice_arrays]

    volume = np.stack(slice_arrays, axis=0)  # (D,H,W)
    volume=np.transpose(volume,(2,1,0))
    return volume


import torch.nn as nn


class DicomSeriesDataset(Dataset):
    def __init__(self, series_dir, load_dicom_series, transform=None,modalities=['CTA']):
        self.series_dir = series_dir
        self.load_dicom_series = load_dicom_series
        self.transform = transform

        self.df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
        self.series_uids = []
        for modality in modalities:
            self.series_uids.extend(self.df[self.df['Modality'] == modality]['SeriesInstanceUID'].to_list())
        

    def __len__(self):
        return len(self.series_uids)

    def __getitem__(self, idx):
        uid = self.series_uids[idx]
        series_path = os.path.join(self.series_dir, uid)
        volume = self.load_dicom_series(series_path)
        modality= self.df[self.df['SeriesInstanceUID'] == uid]['Modality'].iloc[0]
        # volume = apply_dicom_windowing(volume,modality)
        data_dict = {"Image": volume}
        data_dict = self.transform(data_dict)

        return data_dict["Image"], uid

        


class NpzDataset(Dataset):
    def __init__(self, series_dir, transform=None):
        self.series_dir = series_dir
        self.load_dicom_series = load_dicom_series
        self.transform = transform

        self.df = pd.read_csv('/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv')
        self.series_uids = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(os.path.join(series_dir,'*.npz'))]

        

    def __len__(self):
        return len(self.series_uids)

    def __getitem__(self, idx):
        uid = self.series_uids[idx]
        series_path = os.path.join(self.series_dir, uid +'.npz')
        data = np.load(series_path)
        volume = data['volume']
        modality= self.df[self.df['SeriesInstanceUID'] == uid]['Modality'].iloc[0]
        # volume = apply_dicom_windowing(volume,modality)
        data_dict = {"Image": volume}
        data_dict = self.transform(data_dict)

        return data_dict["Image"], uid



import torch
import matplotlib.pyplot as plt

def show_mip(image, mask=None, uid=None):
    # image, mask: torch.Tensor or numpy.ndarray [C, D, H, W] or [D, H, W]
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    if mask is not None and isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    if image.ndim == 4:  # [C,D,H,W]
        image = image[0]  # take first channel

    mips = [
        image.max(axis=0),  # axial (D projection)
        image.max(axis=1),  # coronal (H projection)
        image.max(axis=2),  # sagittal (W projection)
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    for i, mip in enumerate(mips):
        axes[i].imshow(mip, cmap="gray")
        axes[i].set_title(["Axial MIP", "Coronal MIP", "Sagittal MIP"][i])
        axes[i].axis("off")

    if mask is not None:
        mask_mips = [
            mask.max(axis=0),
            mask.max(axis=1),
            mask.max(axis=2),
        ]
        for i, m in enumerate(mask_mips):
            axes[i].imshow(m, cmap='jet', alpha=0.4)

    if uid is not None:
        plt.suptitle(f"Series UID: {uid}")
    plt.show()


num_classes=14
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = DynUNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=num_classes,  # same as before
            strides= [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        kernel_size= [[3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        upsample_kernel_size= [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
        filters= [32, 64, 128, 256, 320, 320],
    
    
    res_block = True
)
chkpt = torch.load('/kaggle/input/rsna-segm-1kepochs-128/best_model.pth',map_location=device)
state_dict = chkpt

# Remove 'module.' prefix if it exists
new_state_dict = {}
for k, v in state_dict.items():
    if k.startswith("module."):
        new_state_dict[k.replace("module.", "")] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=False)


import glob


series_dir = "/kaggle/input/save-3d-volumes-as-npz-all-data-128/processed_train/"
tfms =Compose([
            EnsureChannelFirstd(keys=["Image"], channel_dim="no_channel"),
            EnsureTyped(keys=["Image"],dtype=[torch.float32]),
            # Resized(keys=["Image"], spatial_size=(384,384,128), mode=["trilinear"]),
            
            ToTensord(keys=["Image"]),
        ])
mo = ['CTA' , 'MRA' , 'MRI T1post', 'MRI T2']
processor = DICOMPreprocessorKaggle((128,128,128))
dataset = NpzDataset(series_dir, tfms)

# dataset=
dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


from tqdm import tqdm
import pydicom
from scipy import ndimage


model = model.to(device)
model.eval()
with torch.no_grad():
    for vol, uid in tqdm(dataloader):
        vol = vol.to(device)  # [B,1,D,H,W]
        # pred = sliding_window_inference(
        #     inputs=vol,
        #     roi_size=(64, 64, 64),  # patch size
        #     sw_batch_size=8,
        #     predictor=model,
        #     overlap=0.5)
        pred= model(vol)
        # pred_labels = torch.sigmoid(pred)
        # pred_labels = pred_labels > 0.5
        pred = torch.softmax(pred, dim=1)  # convert logits → probabilities
        # pred_labels = torch.argmax(pred, dim=1, keepdim=True) 
        pred_labels = torch.argmax(pred, dim=1)
        print(pred.shape)
        show_mip(vol.cpu().numpy().squeeze(),pred_labels.cpu().numpy().squeeze())
        np.savez(f"{uid}_pred.npz",volume=vol.cpu().numpy().squeeze(),mask=pred_labels.cpu().numpy().squeeze())
        # print('After postproc-')
        # show_mip(vol.cpu().numpy().squeeze(),postprocess_vessels(pred_labels.cpu().numpy().squeeze()))

        print(f'UID : {uid}')


# path = "/kaggle/working/('1.2.826.0.1.3680043.8.498.10004044428023505108375152878107656647',)_pred.npz"
# mask = np.load(path)['mask']



import gc

gc.collect()
torch.cuda.empty_cache()

