# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
import glob
import random


input_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection"
image_dir = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/"


image_files =os.listdir(image_dir)
print(len(image_files))


files = os.listdir(input_dir)

for file in files:
    print(file)


train_df = pd.read_csv(input_dir + "/train.csv")
print(train_df.shape)
train_df.head()


train_df.info()


print(train_df.describe())


train_loc_df = pd.read_csv(input_dir + "/train_localizers.csv")
print(train_loc_df.shape)
train_loc_df.head()



train_loc_df.info()


print(train_loc_df.describe())


# Select the last 14 columns (assuming they are the label/object columns)
label_columns = train_df.columns[-14:]

# Count how many times each label is present (i.e., sum the 1s per column)
label_counts = train_df[label_columns].sum().sort_values(ascending=False)

# Print the counts
print("Object presence counts:")
print(label_counts)

# Plot a bar chart
plt.figure(figsize=(12, 6))
label_counts.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title("Frequency of Each Object Label in Training Data")
plt.xlabel("Object Label")
plt.ylabel("Count (Presence = 1)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


object_counts = train_loc_df['location'].value_counts()

# Print the frequency counts
print("Frequency of each unique object in 'location':")
print(object_counts)

# Plot a bar chart of the frequency counts
plt.figure(figsize=(12, 6))
object_counts.plot(kind='bar', color='cornflowerblue', edgecolor='black')
plt.title("Frequency of Unique Objects in location")
plt.xlabel("location")
plt.ylabel("Frequency")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()  # Adjust layout to ensure everything fits without overlap
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


def load_dicom_series(series_path):
    dicoms = []
    for fname in sorted(os.listdir(series_path)):
        if fname.endswith('.dcm'):
            dcm = pydicom.dcmread(os.path.join(series_path, fname))
            dicoms.append(dcm)
    dicoms.sort(key=lambda x: float(x.ImagePositionPatient[2]))  # Sort by slice location
    images = np.stack([d.pixel_array for d in dicoms])
    sop_uids = [d.SOPInstanceUID for d in dicoms]
    return images, sop_uids, dicoms

def load_nii_seg(seg_path):
    nii = nib.load(seg_path)
    data = nii.get_fdata()
    return data.astype(np.uint8)


def show_slice(dcm_slice, seg_slice, overlay=False):
    plt.figure(figsize=(12, 4))
    
    # DICOM
    plt.subplot(1, 3, 1)
    plt.imshow(dcm_slice, cmap='gray')
    plt.title("DICOM Slice")
    
    # Segmentation
    if seg_slice is not None:
        plt.subplot(1, 3, 2)
        plt.imshow(seg_slice, cmap='gray')
        plt.title("Segmentation Slice")
        
        # Overlay
        if overlay:
            plt.subplot(1, 3, 3)
            plt.imshow(dcm_slice, cmap='gray')
            plt.imshow(seg_slice, cmap='Reds', alpha=0.4)
            plt.title("Overlay")
    
    plt.tight_layout()
    plt.show()


# Example paths
series_id = train_loc_df.SeriesInstanceUID[0]	
sop_uid_target = train_loc_df.SOPInstanceUID[0]
print(series_id, sop_uid_target)



series_path = f'{input_dir}/series/{series_id}'
seg_path = f'{input_dir}/segmentations/{sop_uid_target}/{sop_uid_target}.nii'  


# Load data
dcm_images, sop_uids, dicoms = load_dicom_series(series_path)
seg_volume = load_nii_seg(seg_nii_list[0])


# Find the matching slice index
slice_idx = sop_uids.index(sop_uid_target)
print(f"Matching slice index: {slice_idx}")


# Get slices
dcm_slice = dcm_images[slice_idx]
seg_slice = seg_volume[slice_idx]  # Assuming same depth alignment
print(dcm_slice.shape, seg_slice.shape)

# Plot
show_slice(dcm_slice, seg_slice, overlay=True)


seg_nii_list = glob.glob(f'{input_dir}/segmentations/*/*.nii')
print(len(seg_nii_list))


nii_seg = load_nii_seg(seg_nii_list[0])
print(nii_seg.shape)


def plot_random_slices(nii_path, num_slices=25):
    # Load the NIfTI image
    nii_img = nib.load(nii_path)
    data = nii_img.get_fdata()
    
    # Ensure it's a 3D volume
    if data.ndim != 3:
        raise ValueError("Input NIfTI image is not 3D.")

    # Get the number of slices along the z-axis
    total_slices = data.shape[2]

    # Choose random slice indices
    random_indices = random.sample(range(total_slices), min(num_slices, total_slices))
    random_indices.sort()  # Optional: sort to display slices in order

    # Plot the slices
    fig, axes = plt.subplots(1, len(random_indices), figsize=(20, 5))
    for i, idx in enumerate(random_indices):
        axes[i].imshow(data[:, :, idx], cmap='gray')
        axes[i].set_title(f"Slice {idx}")
        axes[i].axis('off')

    plt.tight_layout()
    plt.show()


plot_random_slices(seg_nii_list[1])




