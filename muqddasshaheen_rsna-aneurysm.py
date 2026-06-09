# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import shutil
import random

# Input dataset folder (RSNA series folders)
src_folder = "/kaggle/input/rsna-intracranial-aneurysm-detection/series"

# Output folder for 10GB subset
dst_folder = "/kaggle/working/train_subset"
os.makedirs(dst_folder, exist_ok=True)

# List all series folders (patients)
series_folders = [os.path.join(src_folder, d) for d in os.listdir(src_folder) 
                  if os.path.isdir(os.path.join(src_folder, d))]

print("Total series/patient folders:", len(series_folders))

# Shuffle folders for randomness
random.shuffle(series_folders)

# Copy DICOMs folder-wise until 10GB
max_size = 10 * (1024**3)  # 10GB
copied_size = 0
copied_series = 0

for folder in series_folders:
    # Get all dicom files in this series
    dicoms = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".dcm")]

    # Check if adding this series exceeds 10GB
    series_size = sum(os.path.getsize(f) for f in dicoms)
    if copied_size + series_size > max_size:
        # If adding full series exceeds, copy files one by one until limit
        for f in dicoms:
            fsize = os.path.getsize(f)
            if copied_size + fsize > max_size:
                break
            shutil.copy(f, dst_folder)
            copied_size += fsize
        break

    # Copy entire series folder
    for f in dicoms:
        shutil.copy(f, dst_folder)
    copied_size += series_size
    copied_series += 1

print(f"✅ Copied {copied_series} series/patients, total size: {copied_size/(1024**3):.2f} GB")
print(f"Subset saved in: {dst_folder}")



import os
import pydicom
import cv2
import numpy as np
from tqdm import tqdm

# Input folder: multiple patients DICOM subset
src_folder = "/kaggle/working/train_subset"

# Output folder: preprocessed images
dst_folder = "/kaggle/working/preprocessed_images"
os.makedirs(dst_folder, exist_ok=True)

def window_image(img, window_center, window_width):
    """Apply DICOM windowing"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min) * 255.0
    return img.astype(np.uint8)

def preprocess_dicom_safe(dicom_path, output_path, size=(256, 256)):
    try:
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array.astype(np.float32)

        # Skip empty / zero-size images
        if img is None or img.size == 0:
            print(f"⚠ Skipping empty DICOM: {dicom_path}")
            return

        # Apply windowing if available
        try:
            wc = dcm.WindowCenter
            ww = dcm.WindowWidth
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = wc[0]
            if isinstance(ww, pydicom.multival.MultiValue):
                ww = ww[0]
            img = window_image(img, wc, ww)
        except:
            # fallback normalization
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
            img = img.astype(np.uint8)

        # Resize
        if img.size == 0:
            print(f"⚠ Skipping DICOM with empty image after processing: {dicom_path}")
            return

        img = cv2.resize(img, size)

        # Ensure 2D for cv2.imwrite
        if img.ndim == 2:
            cv2.imwrite(output_path, img)
        elif img.ndim == 3:
            # take first channel if multi-channel
            cv2.imwrite(output_path, img[:, :, 0])
        else:
            print(f"⚠ Skipping invalid shape: {dicom_path}, shape={img.shape}")

    except Exception as e:
        print(f"❌ Error in {dicom_path}: {e}")

# Process all DICOM files in subset
for f in tqdm(os.listdir(src_folder)):
    if f.endswith(".dcm"):
        dicom_path = os.path.join(src_folder, f)
        output_path = os.path.join(dst_folder, f.replace(".dcm", ".png"))
        preprocess_dicom_safe(dicom_path, output_path)

print("✅ Preprocessing complete! Images saved in:", dst_folder)



import os
import pydicom
import cv2
import numpy as np
from tqdm import tqdm

# Input folder: multiple patients DICOM subset
src_folder = "/kaggle/working/train_subset"

# Output folder: preprocessed images
dst_folder = "/kaggle/working/preprocessed_images"
os.makedirs(dst_folder, exist_ok=True)

def window_image(img, window_center, window_width):
    """Apply DICOM windowing"""
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    img = (img - img_min) / (img_max - img_min) * 255.0
    return img.astype(np.uint8)

def preprocess_dicom_safe(dicom_path, output_path, size=(256, 256)):
    try:
        dcm = pydicom.dcmread(dicom_path)
        img = dcm.pixel_array.astype(np.float32)

        # Skip empty / zero-size images
        if img is None or img.size == 0:
            print(f"⚠ Skipping empty DICOM: {dicom_path}")
            return

        # Apply windowing if available
        try:
            wc = dcm.WindowCenter
            ww = dcm.WindowWidth
            if isinstance(wc, pydicom.multival.MultiValue):
                wc = wc[0]
            if isinstance(ww, pydicom.multival.MultiValue):
                ww = ww[0]
            img = window_image(img, wc, ww)
        except:
            # fallback normalization
            img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255.0
            img = img.astype(np.uint8)

        # Resize
        if img.size == 0:
            print(f"⚠ Skipping DICOM with empty image after processing: {dicom_path}")
            return

        img = cv2.resize(img, size)

        # Ensure 2D for cv2.imwrite
        if img.ndim == 2:
            cv2.imwrite(output_path, img)
        elif img.ndim == 3:
            # take first channel if multi-channel
            cv2.imwrite(output_path, img[:, :, 0])
        else:
            print(f"⚠ Skipping invalid shape: {dicom_path}, shape={img.shape}")

    except Exception as e:
        print(f"❌ Error in {dicom_path}: {e}")

# Process all DICOM files in subset
for f in tqdm(os.listdir(src_folder)):
    if f.endswith(".dcm"):
        dicom_path = os.path.join(src_folder, f)
        output_path = os.path.join(dst_folder, f.replace(".dcm", ".png"))
        preprocess_dicom_safe(dicom_path, output_path)

print("✅ Preprocessing complete! Images saved in:", dst_folder)


