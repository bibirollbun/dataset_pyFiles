# Importing libraries
import os
import pandas as pd
import matplotlib.pyplot as plt
import pydicom # for working with DICOM file (Digital Imaging and Communications in Medicine)
import nibabel as nib #for reading and writing neuroimaging formats like .nii and .nii.gz
import cv2 # image processing(resizing, filtering, transformed)
from glob import glob # file paths matching


data_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection'



train_df = pd.read_csv(os.path.join(data_dir, 'train.csv'))
print('Shape:', train_df.shape)





train_df.head()



print(train_df.columns.tolist())


train_df.info()


label_cols = [c for c in train_df.columns if c not in ['SeriesInstanceUID', 'Modality', 'PatientAge', 'PatientSex']]
train_df[label_cols].sum().sort_values(ascending=False).plot(kind='bar', figsize=(10,4), title="Label Counts")
plt.show()


# trying to see the middle slice of 1st series

# Pick first series from train.csv
series_id = train_df.iloc[0]['SeriesInstanceUID']
series_path = os.path.join(data_dir, "series", series_id)

# List DICOM files in that series
dcm_files = sorted(os.listdir(series_path))
print(f"Series {series_id} has {len(dcm_files)} slices")

# Load and show the middle slice
mid_slice_path = os.path.join(series_path, dcm_files[len(dcm_files)//2])
dcm = pydicom.dcmread(mid_slice_path)

plt.imshow(dcm.pixel_array, cmap='gray')
plt.title(f"Series {series_id} - Slice {len(dcm_files)//2}")
plt.axis('off')
plt.show()



# Load the segmentation and corresponding DICOM
series_id = "1.2.826.0.1.3680043.8.498.10035643165968342618460849823699311381"

# Paths
series_path = os.path.join(data_dir, "series", series_id)
seg_file = os.path.join(data_dir, "segmentations", f"{series_id}.nii")

# Load segmentation
seg_img = nib.load(seg_file)
seg_data = seg_img.get_fdata()

# Load middle DICOM slice
dcm_files = sorted(os.listdir(series_path))
mid_slice_path = os.path.join(series_path, dcm_files[len(dcm_files)//2])
dcm = pydicom.dcmread(mid_slice_path)
dcm_data = dcm.pixel_array



# Overlay segmentation on top of the DICOM slice
import numpy as np

# Take same middle slice from segmentation
seg_slice = seg_data[:, :, seg_data.shape[2]//2]

plt.figure(figsize=(8,8))

# Show base DICOM in grayscale
plt.imshow(dcm_data, cmap="gray")

# Overlay segmentation (make sure it's binary/mask-like)
plt.imshow(seg_slice, cmap="jet", alpha=0.4)  # alpha controls transparency

plt.title(f"Overlay for Series {series_id}")
plt.axis("off")
plt.show()



# resampling (segmentation mask and DICOM scan are in the same space before overlaying)
import nibabel as nib
import numpy as np
import SimpleITK as sitk

# Load segmentation with SimpleITK
seg_sitk = sitk.ReadImage(seg_file)

# Load DICOM series with SimpleITK
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(series_path)
reader.SetFileNames(dicom_names)
dicom_sitk = reader.Execute()

# Resample segmentation to DICOM space
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(dicom_sitk)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
seg_resampled = resampler.Execute(seg_sitk)

# Convert to numpy for visualization
dicom_np = sitk.GetArrayFromImage(dicom_sitk)[dicom_sitk.GetSize()[2]//2]
seg_np = sitk.GetArrayFromImage(seg_resampled)[dicom_sitk.GetSize()[2]//2]

plt.imshow(dicom_np, cmap="gray")
plt.imshow(seg_np, cmap="jet", alpha=0.4)
plt.axis("off")
plt.show()



import os
import pydicom
import numpy as np
import matplotlib.pyplot as plt

# Pick one series
series_id = train_df.iloc[0]['SeriesInstanceUID']
series_path = os.path.join(data_dir, "series", series_id)

# Read all DICOMs (only once)
dcm_list = []
for f in os.listdir(series_path):
    dcm = pydicom.dcmread(os.path.join(series_path, f))
    dcm_list.append((dcm.InstanceNumber, dcm.pixel_array))

# Sort by InstanceNumber
dcm_list.sort(key=lambda x: x[0])

# Build volume
volume = np.stack([arr for _, arr in dcm_list], axis=-1)
print("Volume shape:", volume.shape)  # (H, W, NumSlices)

# Normalize for display
volume_norm = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))

# Show single middle slice
mid_slice_idx = volume.shape[2] // 2
plt.figure(figsize=(5, 5))
plt.imshow(volume_norm[:, :, mid_slice_idx], cmap='gray')
plt.title("Single Middle Slice (2D)")
plt.axis('off')
plt.show()

# Show several slices to get 3D feel
fig, axes = plt.subplots(1, 5, figsize=(15, 5))
slice_indices = np.linspace(0, volume.shape[2]-1, 5, dtype=int)
for ax, idx in zip(axes, slice_indices):
    ax.imshow(volume_norm[:, :, idx], cmap='gray')
    ax.set_title(f"Slice {idx}")
    ax.axis('off')
plt.suptitle("Different Slices from the 3D Volume")
plt.show()



def load_dicom_series(series_id, data_dir):
    series_path = os.path.join(data_dir, "series", series_id)
    dcm_files = glob(os.path.join(series_path, "*.dcm"))

    # Read all dicoms once
    dicoms = [pydicom.dcmread(f) for f in dcm_files]

    # Sort them by InstanceNumber
    dicoms.sort(key=lambda x: int(x.InstanceNumber))

    # Convert to 3D numpy array
    slices = [d.pixel_array.astype(np.float32) for d in dicoms]
    volume = np.stack(slices, axis=-1)  # shape: (H, W, depth)
    print(slices)
    print("----------------------------------------------------")
    print(volume)

    return volume



def normalize(volume):
    min_val = np.min(volume)
    max_val = np.max(volume)
    return (volume - min_val) / (max_val - min_val + 1e-8)



# -----------Medical Imaging-----------------------------
# # if we have just CT scan we can go with this 
# def normalize(volume, min_bound=-1000, max_bound=400):
#     volume = np.clip(volume, min_bound, max_bound)
#     return (volume - min_bound) / (max_bound - min_bound)



# def resize_volume(volume, target_shape=(128, 128, 64)):
#     # Resize each slice in-plane
#     resized_slices = [cv2.resize(volume[:, :, i], target_shape[:2]) for i in range(volume.shape[2])]
#     volume_resized = np.stack(resized_slices, axis=-1)

#     # Resize depth
#     if volume_resized.shape[2] != target_shape[2]:
#         depth_indices = np.linspace(0, volume_resized.shape[2] - 1, target_shape[2]).astype(int)
#         volume_resized = volume_resized[:, :, depth_indices]

#     return volume_resized

# ------------------------Medical Imaging--------------------------------------------------
# we will be using this because in medical imaging, small details (tiny vessels, lesions, aneurysms) matter a lot, and crude slice resampling might lose them.
import scipy.ndimage

def resize_volume(volume, target_shape=(128, 128, 64)):
    factors = (
        target_shape[0] / volume.shape[0],
        target_shape[1] / volume.shape[1],
        target_shape[2] / volume.shape[2],
    )
    return scipy.ndimage.zoom(volume, factors, order=1)  # order=1 = linear



def preprocess_series(series_id, data_dir, target_shape=(128, 128, 64)):
    volume = load_dicom_series(series_id, data_dir)
    volume = normalize(volume)
    print(volume)
    volume = resize_volume(volume, target_shape)
    print(volume)
    volume = np.expand_dims(volume, axis=-1)  # add channel dimension (H, W, D, C) tensorflow
    print(volume)
    return volume.astype(np.float32)


series_id = train_df.iloc[0]['SeriesInstanceUID']
processed_vol = preprocess_series(series_id, data_dir)
print("Processed volume shape:", processed_vol.shape)  # (1, 128, 128, 64)




