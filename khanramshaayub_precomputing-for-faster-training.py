##---------------------- IMPORTING ALL LIBRARIES -------------------------
import pandas as pd
import pydicom #for dicom files
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from pathlib import Path
import os
import cv2
import matplotlib.pyplot as plt
import timm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchmetrics.classification import AUROC
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.cuda.amp import autocast, GradScaler
from torch.nn import BCEWithLogitsLoss
import polars as pl
from torch.cuda.amp import autocast
import gc
import shutil
from IPython.display import display


from dataclasses import dataclass

@dataclass
class Config:
    image_size = 512
    num_slices = 32
    
cfg = Config()


#---------------------------------- IMAGE PROCESSING ---------------------------------

def adaptive_windowing(image, modality):
    percentile_range = (5, 95)

    img_flat = image.flatten()
    img_flat = img_flat[img_flat > 0]
    if len(img_flat) == 0:
        return np.zeros_like(image, dtype=np.uint8)
    low_val = np.percentile(img_flat, percentile_range[0])
    
    high_val = np.percentile(img_flat, percentile_range[1])

    if modality in ['CTA']:
        window_width = (high_val - low_val) * 1.5
        window_center = (high_val + low_val) / 2
    elif modality in ['MRA']:
        window_width = (high_val - low_val) * 1.2
        window_center = high_val * 0.7
    else:
        window_width = high_val - low_val
        window_center = (high_val + low_val) / 2    

    img_min = window_center - window_width / 2
    img_max = window_center + window_width / 2
    img_windowed = np.clip(image, img_min, img_max)

## normalize 0-255
    if img_max > img_min:
        img_normalized = ((img_windowed - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else: 
        img_normalized = np.zeros_like(image, dtype = np.uint8)
    return img_normalized


def process_dicom_series(series_path):

    all_filepaths = []
    for root, _, files in os.walk(series_path):
        for file in files:
            if file.endswith('.dcm'):
                all_filepaths.append(os.path.join(root, file))
    all_filepaths.sort()
    
    if len(all_filepaths) == 0: 
        #no files extracted
        print("Ops! NO files extracted")
        volume = np.zeros((cfg.num_slices, cfg.image_size, cfg.image_size), dtype = np.uint8)
        metadeta = {'age': 40, 'sex': 0, 'modality': 'CT'}
        return volume, metadata

    metadata = {}
    dicom_data = []
    for i, filepath in enumerate(all_filepaths):
        ds = pydicom.dcmread(filepath, force = True)
        img = ds.pixel_array 
        if img.ndim == 3:
            if img.shape[-1] == 3:
                img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
            else:
                img = img[:, :, 0]
        instance_num = getattr(ds, 'InstanceNumber', i) # get instance number for proper slice sorting
        if i == 0: # extract the metadata
            metadata["modality"] = getattr(ds, 'Modality', 'CT')
            try: 
                age_str = getattr(ds, 'PatientAge', '50Y')
                age = int(''.join(filter(str.isdigit, age_str[:3])) or '50')
                metadata['age'] = min(age, 100)
            except:
                metadata['age'] = 50
            try:
                sex = getattr(ds, 'PatientSex', 'M')
                metadata['sex'] = 1 if sex == 'M' else 0
            except:
                metadata['sex'] = 0
        
        #### ------ APPLY RESCALING------------
        if hasattr(ds, 'RescaleSlope') and hasattr(ds, 'RescaleIntercept'):
            img = img * ds.RescaleSlope + ds.RescaleIntercept
            
        #input arrays shape not matching so resizing image should be done before stacking
        img_resized = cv2.resize(img, (cfg.image_size, cfg.image_size))
        dicom_data.append((instance_num, img_resized))

    if len(dicom_data) == 0:
        volume = np.zeros((cfg.num_slices, cfg.image_size, cfg.image_size), dtype = np.uint8)
        return volume, metadata
            
    dicom_data.sort(key = lambda x: x[0])
    raw_slices = [d[1] for d in dicom_data]
    volume_3d = np.stack(raw_slices, axis = 0) #stacking on top of each other (row-wise)

    ## apply modality specific intensity windowing
    volume_windowed = adaptive_windowing(volume_3d, metadata['modality'])

    ## Resize slices 
    processed_slices = []
    for img in volume_windowed:
        resized = cv2.resize(img, (cfg.image_size, cfg.image_size))
        processed_slices.append(resized)
    volume = np.array(processed_slices)

    if len(processed_slices) > cfg.num_slices:
        indices = np.linspace(0, len(processed_slices) - 1, cfg.num_slices).astype(int)
        volume = volume[indices]
    elif len(processed_slices) < cfg.num_slices:
        pad_size = cfg.num_slices - len(processed_slices)
        volume = np.pad(volume, ((0,pad_size), (0,0) , (0,0)), mode = 'edge')
    return volume, metadata
    
#####----------------- CREATE RICH MULTI-CHANNEL REPRESENTATON OF 3D IMAGE------------------
"""
Since "volumes" obtained are 3D, in order to pass it in 2D network we need a smart way fro converting 3D to 2D without losing finer details 


"""
def create_multichannel_img(volume):
    depth, height, width = volume.shape
    #--------Channel 1: Adaptive maximum intensity projection---------
    start = int(depth * 0.15)
    end = int(depth * 0.85)
    core_vol = volume[start: end]
    mip = np.max(core_vol, axis = 0)

    #------ Channel 2: weighted avg of high intensity slices------
    #------------ Focusing on brightness-----------
    slices_means = np.mean(volume, axis = (1,2))
    top_percentile = np.percentile(slices_means, 75)
    high_intensity = slices_means >= top_percentile
    if np.any(high_intensity):
        weighted_avg = np.mean(volume[high_intensity], axis = 0)
    else:
        weighted_avg = np.mean(volume, axis = 0)

    ##-------- Channel 3: Std projection ----------
    std_proj = np.zeros_like(volume[0])
    window_size = min(5, depth // 4)
    for i in range(depth - window_size + 1):
        window_std = np.std(volume[i : i + window_size], axis = 0)
        std_proj = np.maximum(std_proj, window_std)
    #------- normalize all channels 0-255-----------
    channels = []
    for channel in [mip, weighted_avg, std_proj]:
        if channel.max() > channel.min():
            channel_norm = ((channel - channel.min()) / (channel.max() - channel.min()) * 255).astype(np.uint8)
        else:  
            channel_norm = np.zeros_like(channel, dtype = np.uint8)
        channels.append(channel_norm)
    return np.stack(channels, axis = -1)



import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm
images_path = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"
preprocessed_path = "/kaggle/working/preprocessed_npy"
os.makedirs(preprocessed_path, exist_ok=True)

def preprocess_single_npy(row):
    """
    Save as float32 numpy array (preserves full precision)
    """
    instance_id = row['SeriesInstanceUID']
    series_dir = os.path.join(images_path, instance_id)
    
    if not os.path.exists(series_dir):
        return None
    
    volume, metadata = process_dicom_series(series_dir)
    img = create_multichannel_img(volume)  # (H, W, 3) uint8
    
    # Convert back to float32 for better precision
    img_float = img.astype(np.float32) / 255.0  # Normalize to [0, 1]
    
    # Save as numpy array (much better for medical data!)
    output_path = os.path.join(preprocessed_path, f"{instance_id}.npy")
    np.save(output_path, img_float)
    
    return instance_id

# Precompute
subset_df = pd.read_csv("/kaggle/input/subset3000/train_subset_3000.csv")

with Pool(processes=os.cpu_count()) as pool:
    results = list(tqdm(
        pool.imap(preprocess_single_npy, [row for _, row in subset_df.iterrows()]),
        total=len(subset_df),
        desc="Precomputing .npy images"
    ))

print(f"Done! {len([r for r in results if r])} images saved")


!pip install kaggle --quiet
!mkdir -p ~/.kaggle
!echo '{"username":"putusername","key":"putyourkey"}' > ~/.kaggle/kaggle.json
!chmod 600 ~/.kaggle/kaggle.json


!kaggle datasets init -p /kaggle/working/preprocessed_npy


!kaggle datasets create -p /kaggle/working/preprocessed_npy --dir-mode zip




metadata['title'] = "RSNA Aneurysm Preprocessed NPY 3000"
metadata['id'] = "khanramshaayub/rsna-aneurysm-preprocessed-3000"  
metadata['licenses'] = [{"name": "CC0-1.0"}]

metadata['subtitle'] = "Preprocessed multichannel images for RSNA Intracranial Aneurysm Detection"
metadata['description'] = "Preprocessed .npy files (512x512x3, float32) from DICOM series. Ready for fast training."

with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print("\n✓ Updated metadata:")
print(json.dumps(metadata, indent=2))


!kaggle datasets create -p /kaggle/working/preprocessed_npy --dir-mode zip


import shutil

shutil.make_archive("/kaggle/working/rsna_preprocessed_npy", 'zip', "/kaggle/working/preprocessed_npy")


import numpy as np
import matplotlib.pyplot as plt
import os

preprocessed_path = "/kaggle/input/rsna-preprocessed-images"

def load_precomputed_npy(instance_id):
    npy_path = os.path.join(preprocessed_path, f"{instance_id}.npy")
    if os.path.exists(npy_path):
        img = np.load(npy_path)  # shape (H, W, C), dtype float32, normalized [0,1]
        return img
    else:
        return None
img = load_precomputed_npy("1.2.826.0.1.3680043.8.498.10005158603912009425635473100344077317")


img.shape


img_uint8 = (img * 255).astype(np.uint8)

plt.figure(figsize=(15, 4))
plt.subplot(141)
plt.imshow(img_uint8[:,:,0], cmap='gray')
plt.title('Channel 0: MIP')
plt.axis('off')

plt.subplot(142)
plt.imshow(img_uint8[:,:,1], cmap='gray')
plt.title('Channel 1: Weighted Avg')
plt.axis('off')

plt.subplot(143)
plt.imshow(img_uint8[:,:,2], cmap='gray')
plt.title('Channel 2: Std Projection')
plt.axis('off')

plt.subplot(144)
plt.imshow(img_uint8)
plt.title('Combined RGB')
plt.axis('off')

plt.tight_layout()
plt.show()

