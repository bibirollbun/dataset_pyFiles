# General utilities
import os
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# DICOM handling
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# Warnings
import warnings
warnings.filterwarnings("ignore")



# 3D visualization (optional for later steps)
import nibabel as nib  # for working with NifTI if needed
import cv2  # for interpolation / resizing in preprocessing
from IPython.display import display, HTML  # for cleaner output



# Step 2: Load Metadata
BASE_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection"

train_df = pd.read_csv(f"{BASE_DIR}/train.csv")
localizer_df = pd.read_csv(f"{BASE_DIR}/train_localizers.csv")

print("train.csv shape:", train_df.shape)
print("train_localizers.csv shape:", localizer_df.shape)

# Aneurysm presence distribution
print(train_df["Aneurysm Present"].value_counts())

# Show some rows
train_df.head()



# Step 3: Visualize a Series
def load_dicom_volume(series_path):
    files = [pydicom.dcmread(os.path.join(series_path, f)) 
             for f in os.listdir(series_path) if f.endswith(".dcm")]
    files.sort(key=lambda x: int(x.InstanceNumber))
    
    # Convert to 3D volume
    volume = np.stack([f.pixel_array for f in files])
    
    return volume, files


def show_dicom_slices_grid(volume, num_rows=4, num_cols=6):
    total_slices = num_rows * num_cols
    interval = len(volume) // total_slices
    selected_slices = [volume[i * interval] for i in range(total_slices)]

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))
    for i, ax in enumerate(axes.flat):
        ax.imshow(selected_slices[i], cmap='gray')
        ax.axis('off')
    plt.suptitle("Sample DICOM Slices", fontsize=16)
    plt.tight_layout()
    plt.show()



series_dir = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series"
sample_uids = train_df["SeriesInstanceUID"].unique()[:3]  # visualize first 3 patients

for uid in sample_uids:
    print(f"Series UID: {uid}")
    path = os.path.join(series_dir, uid)
    
    try:
        volume, files = load_dicom_volume(path)
        print(f"Volume shape: {volume.shape}")
        show_dicom_slices_grid(volume)
    except Exception as e:
        print(f"Error in series {uid}: {e}")



def scroll_dicom_series(volume):
    from IPython.display import display, clear_output
    import time
    
    for i in range(volume.shape[0]):
        plt.imshow(volume[i], cmap='gray')
        plt.title(f"Slice {i + 1}/{volume.shape[0]}")
        plt.axis('off')
        display(plt.gcf())
        clear_output(wait=True)
        time.sleep(0.1)  # Adjust speed

# Example for one series:
volume, _ = load_dicom_volume(os.path.join(series_dir, sample_uids[0]))
scroll_dicom_series(volume)



from skimage.transform import resize

def preprocess_dicom_series(series_path, target_shape=(64, 128, 128), normalize="z-score"):
    """
    Preprocess a DICOM series: loads, sorts, normalizes, and resizes to target shape.
    
    Args:
        series_path (str): Path to DICOM series folder.
        target_shape (tuple): Output shape (Depth, Height, Width)
        normalize (str): 'z-score' or 'min-max'
    
    Returns:
        np.ndarray: Preprocessed 3D volume of shape target_shape
    """
    # Step 1: Load and sort slices
    slices = []
    for fname in os.listdir(series_path):
        if fname.endswith(".dcm"):
            dcm = pydicom.dcmread(os.path.join(series_path, fname))
            slices.append(dcm)
    
    # Sort by position in Z or instance number
    try:
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    except:
        slices.sort(key=lambda x: int(x.InstanceNumber))
    
    # Stack to form a volume
    volume = np.stack([s.pixel_array for s in slices])
    volume = volume.astype(np.float32)

    # Step 2: Intensity normalization
    if normalize == "min-max":
        volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume) + 1e-5)
    elif normalize == "z-score":
        mean = np.mean(volume)
        std = np.std(volume)
        volume = (volume - mean) / (std + 1e-5)
    
    # Step 3: Resize to target shape
    volume_resized = resize(volume, output_shape=target_shape, preserve_range=True, mode='constant')
    
    return volume_resized



sample_uid = train_df.iloc[0]["SeriesInstanceUID"]
sample_path = os.path.join(series_dir, sample_uid)

volume_cleaned = preprocess_dicom_series(sample_path, target_shape=(64, 128, 128))
print("Processed volume shape:", volume_cleaned.shape)

# Visualize a few slices
plt.figure(figsize=(10, 6))
for i in range(6):
    plt.subplot(2, 3, i+1)
    plt.imshow(volume_cleaned[i * 10], cmap="gray")
    plt.axis("off")
    plt.title(f"Slice {i * 10}")
plt.tight_layout()
plt.show()





