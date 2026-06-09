!pip install nibabel pydicom opencv-python matplotlib pandas seaborn scikit-image


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nibabel as nib
import pydicom
import cv2
from skimage import exposure


# path 
data_root = "/kaggle/input/rsna-intracranial-aneurysm-detection/"

train_df = pd.read_csv(os.path.join(data_root, "train.csv"))
train_localizers_df = pd.read_csv(os.path.join(data_root, "train_localizers.csv"))

print(train_df.head())
print(train_localizers_df.head())


import os
import nibabel as nib
import matplotlib.pyplot as plt

# Paths
SERIES_DIR = os.path.join(data_root, "series")
SEG_DIR = os.path.join(data_root, "segmentations")

# List all .nii or .nii.gz files
nii_files = [f for f in os.listdir(SEG_DIR) if f.endswith((".nii", ".nii.gz"))]

print("Found", len(nii_files), "NIfTI files")

# Loop through the first few files and visualize
for i, nii_file in enumerate(nii_files[:3]):  # change 3 → number of patients you want to preview
    nii_path = os.path.join(SEG_DIR, nii_file)
    img = nib.load(nii_path)
    img_data = img.get_fdata()

    print(f"{i+1}. File: {nii_file} | Shape: {img_data.shape}")

    # Show middle slice
    slice_idx = img_data.shape[2] // 2
    plt.imshow(img_data[:, :, slice_idx], cmap="gray")
    plt.title(f"{nii_file} - Middle Slice")
    plt.axis("off")
    plt.show()


def load_dicom_series(series_folder):
    files = [pydicom.dcmread(os.path.join(series_folder, f)) 
             for f in os.listdir(series_folder) if f.endswith(".dcm")]
    files.sort(key=lambda x: int(x.InstanceNumber))  # sort slices
    
    volume = np.stack([f.pixel_array for f in files], axis=0)
    return volume

# Example: pick first SeriesInstanceUID
series_id = train_df.iloc[0]["SeriesInstanceUID"]
series_path = os.path.join(SERIES_DIR, series_id)

volume = load_dicom_series(series_path)
print("Volume shape:", volume.shape)

# Show middle slice
plt.imshow(volume[volume.shape[0]//2], cmap="gray")
plt.title(f"Series {series_id}")
plt.axis("off")
plt.show()


import glob

# Find all DICOM files recursively
dcm_files = glob.glob(os.path.join(SERIES_DIR, "**", "*.dcm"), recursive=True)

print("Found", len(dcm_files), "DICOM files")
print("Example:", dcm_files[:5])


import pydicom
import numpy as np
import matplotlib.pyplot as plt

# Example: pick first 1 series (all slices inside same folder)
one_series = sorted(dcm_files[:50])  # change number depending on your dataset
volume = []

for dcm_path in one_series:
    dcm = pydicom.dcmread(dcm_path)
    volume.append(dcm.pixel_array)

volume = np.stack(volume, axis=-1)  # shape: (H, W, num_slices)
print("3D Volume shape:", volume.shape)

# Show middle slice
slice_idx = volume.shape[2] // 2
plt.imshow(volume[:, :, slice_idx], cmap="gray")
plt.title("Middle Slice of 3D CTA Volume")
plt.axis("off")
plt.show()


from skimage import exposure
import numpy as np
import matplotlib.pyplot as plt

img_data = dcm.pixel_array

def preprocess_image(img):
    """Normalize (0–1) and enhance contrast of a slice or volume."""
    img = img.astype(np.float32)

    # Avoid divide by zero
    if np.max(img) > np.min(img):
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    else:
        img = np.zeros_like(img, dtype=np.float32)

    # Contrast Limited Adaptive Histogram Equalization (CLAHE)
    img = exposure.equalize_adapthist(img, clip_limit=0.03)

    return img

# Example: pick a middle slice from 3D volume
if img_data.ndim == 3:   # for NIfTI or stacked DICOMs
    slice_idx = img_data.shape[2] // 2
    slice_img = img_data[:, :, slice_idx]
else:  # single 2D DICOM
    slice_img = img_data

# Apply preprocessing
slice_img_prep = preprocess_image(slice_img)

# Show result
plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(slice_img, cmap="gray")
plt.title("Original Slice")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(slice_img_prep, cmap="gray")
plt.title("Preprocessed Slice")
plt.axis("off")

plt.show()


import glob, os

series_folder = os.path.join(data_root, "series")
seg_folder = os.path.join(data_root, "segmentations")

series_files = sorted(glob.glob(os.path.join(series_folder, "**", "*.dcm"), recursive=True))
seg_files = sorted(glob.glob(os.path.join(seg_folder, "**", "*.dcm"), recursive=True))

print("Series files found:", len(series_files))
print("Segmentation files found:", len(seg_files))

# Print a few examples
print("Example series:", series_files[:3])
print("Example segmentation:", seg_files[:3])


series_files = sorted(glob.glob(os.path.join(series_folder, "**", "*"), recursive=True))
series_files = [f for f in series_files if os.path.isfile(f)]

seg_files = sorted(glob.glob(os.path.join(seg_folder, "**", "*"), recursive=True))
seg_files = [f for f in seg_files if os.path.isfile(f)]

print("Series files found:", len(series_files))
print("Segmentation files found:", len(seg_files))


import pydicom

dcm = pydicom.dcmread(series_files[0])
print(dcm)
plt.imshow(dcm.pixel_array, cmap="gray")
plt.show()










