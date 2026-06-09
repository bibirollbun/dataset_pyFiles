# --- Core Libraries ---
import os
import glob
import numpy as np
import pandas as pd

# --- DICOM Handling ---
import pydicom

# --- Visualization ---
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm

# --- Settings for better display ---
pd.set_option('display.max_columns', 50)
sns.set_style('whitegrid')


# --- Define the base path for the competition data ---
BASE_PATH = '/kaggle/input/rsna-intracranial-aneurysm-detection/'

# --- Define paths to specific files and directories ---
TRAIN_CSV_PATH = os.path.join(BASE_PATH, 'train.csv')
LOCALIZERS_CSV_PATH = os.path.join(BASE_PATH, 'train_localizers.csv')
SERIES_DIR = os.path.join(BASE_PATH, 'series/')


# --- Load the training labels and localizer data ---
df_train = pd.read_csv(TRAIN_CSV_PATH)
df_localizers = pd.read_csv(LOCALIZERS_CSV_PATH)

# --- Display the first few rows of the training dataframe ---
print("Training Data Head:")
display(df_train.head())

# --- Display basic info about the training dataframe ---
print("\nTraining Data Info:")
df_train.info()

# --- Display the first few rows of the localizers dataframe ---
print("\nLocalizers Data Head:")
display(df_localizers.head())

print("\nLocalizers Data Info:")
df_localizers.info()


def load_dicom_series(series_uid, base_path):
    """
    Loads a DICOM series and returns it as a 3D NumPy array.
    
    Args:
        series_uid (str): The SeriesInstanceUID of the scan to load.
        base_path (str): The base directory for the series data.
        
    Returns:
        np.ndarray: A 3D array representing the medical scan.
    """
    # Find all DICOM files for the given series UID
    series_path = os.path.join(base_path, series_uid, '*.dcm')
    dicom_files = glob.glob(series_path)
    
    # Read all DICOM files and store them with their instance number
    slices = []
    for file_path in dicom_files:
        dicom = pydicom.dcmread(file_path)
        # The 'InstanceNumber' tag is crucial for ordering the slices correctly
        slices.append((int(dicom.InstanceNumber), dicom))
    
    # Sort the slices based on their instance number
    slices.sort(key=lambda x: x[0])
    
    # Stack the pixel data of the sorted slices to form a 3D volume
    volume_3d = np.stack([s[1].pixel_array for s in slices])
    
    return volume_3d


# Select the 13 location columns
location_columns = df_train.columns[4:17]

# For each row, check if the sum of location labels is greater than 0
# This creates a boolean Series (True if aneurysm exists, False otherwise)
derived_aneurysm_present = (df_train[location_columns].sum(axis=1) > 0).astype(int)

# Compare our derived result with the actual 'Aneurysm Present' column
# The .all() method will return True only if every single row matches.
is_consistent = (derived_aneurysm_present == df_train['Aneurysm Present']).all()

if is_consistent:
    print("✅ Sanity Check Passed: 'Aneurysm Present' is consistent with the 13 location labels.")
else:
    print("❌ Sanity Check Failed: There is a discrepancy between the main target and location labels.")


plt.figure(figsize=(8, 6))
sns.countplot(x='Aneurysm Present', data=df_train, palette='viridis')
plt.title('Overall Aneurysm Presence Distribution', fontsize=16)
plt.xlabel('Aneurysm Present (0 = No, 1 = Yes)', fontsize=12)
plt.ylabel('Number of Scans', fontsize=12)

# Add annotations
ax = plt.gca()
for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points', fontsize=12)
plt.show()


# Select the 13 location columns
location_columns = df_train.columns[4:17]

# Calculate the sum of positive cases for each location
location_counts = df_train[location_columns].sum().sort_values(ascending=False)

plt.figure(figsize=(12, 8))
sns.barplot(x=location_counts.values, y=location_counts.index, orient='h', palette='crest')
plt.title('Distribution of Aneurysms by Location', fontsize=16)
plt.xlabel('Number of Positive Cases', fontsize=12)
plt.ylabel('Aneurysm Location', fontsize=12)
plt.show()


# Select a sample series ID that has a known aneurysm
sample_uid = df_localizers['SeriesInstanceUID'].iloc[0]

print(f"Loading and visualizing scan for Series UID: {sample_uid}")

# Load the 3D volume
brain_scan_3d = load_dicom_series(sample_uid, SERIES_DIR)

print(f"Scan loaded. Shape: {brain_scan_3d.shape}")

# --- Visualize three slices from the volume (e.g., start, middle, end) ---
num_slices = brain_scan_3d.shape[0]
slice_indices = [0, num_slices // 2, num_slices - 1]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, slice_idx in enumerate(slice_indices):
    ax = axes[i]
    ax.imshow(brain_scan_3d[slice_idx], cmap='bone')
    ax.set_title(f'Slice {slice_idx + 1}/{num_slices}')
    ax.axis('off')

plt.suptitle(f'Sample Slices for Series: {sample_uid}', fontsize=16)
plt.show()


# We'll use the 'brain_scan_3d' volume we loaded earlier for the sample_uid
# Its shape corresponds to (z, y, x)

print(f"Original 3D volume shape: {brain_scan_3d.shape} -> (Z, Y, X)")

# --- Calculate the three MIPs ---

# Axial view (Top-down): Project along the Z-axis (axis 0)
mip_axial = np.max(brain_scan_3d, axis=0)

# Coronal view (Front-back): Project along the Y-axis (axis 1)
mip_coronal = np.max(brain_scan_3d, axis=1)

# Sagittal view (Side): Project along the X-axis (axis 2)
mip_sagittal = np.max(brain_scan_3d, axis=2)


# --- Visualize the three projections ---
fig, axes = plt.subplots(1, 3, figsize=(18, 8))

# Axial Plot
axes[0].imshow(mip_axial, cmap='bone')
axes[0].set_title('Axial View (Top-down)', fontsize=14)
axes[0].axis('off')

# Coronal Plot
axes[1].imshow(mip_coronal, cmap='bone')
axes[1].set_title('Coronal View (Front-back)', fontsize=14)
axes[1].axis('off')

# Sagittal Plot
axes[2].imshow(mip_sagittal, cmap='bone')
axes[2].set_title('Sagittal View (Side)', fontsize=14)
axes[2].axis('off')

plt.suptitle(f'Maximum Intensity Projections for Scan: {sample_uid}', fontsize=18)
plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to make room for suptitle
plt.show()


# --- Get the specific localization info for our sample scan ---
sample_uid = '1.2.826.0.1.3680043.8.498.10076056930521523789588901704956188485'
aneurysm_details = df_localizers[df_localizers['SeriesInstanceUID'] == sample_uid].iloc[0]

# --- Get the specific slice identifier and the coordinates string ---
slice_sop_uid = aneurysm_details['SOPInstanceUID']
coords_str = aneurysm_details['coordinates']

# --- Parse the coordinate string ---
import ast
parsed_coords = ast.literal_eval(coords_str)

# --- Robustly handle the coordinate format ---
coords_dict = None
if isinstance(parsed_coords, list) and parsed_coords:
    # If it's a list of dictionaries, take the first one
    coords_dict = parsed_coords[0]
elif isinstance(parsed_coords, dict):
    # If it's already a dictionary, use it directly
    coords_dict = parsed_coords

# --- Proceed if we successfully extracted a coordinate dictionary ---
if coords_dict and 'x' in coords_dict and 'y' in coords_dict:
    x_coord = coords_dict['x']
    y_coord = coords_dict['y']

    # --- Construct the full path to the specific DICOM slice file ---
    slice_path = os.path.join(SERIES_DIR, sample_uid, f'{slice_sop_uid}.dcm')

    # --- Read the single DICOM file for that slice ---
    dicom_slice = pydicom.dcmread(slice_path)
    slice_pixels = dicom_slice.pixel_array

    # --- Visualize the slice and highlight the aneurysm ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(slice_pixels, cmap='bone')
    
    # Draw a red circle to mark the aneurysm's location
    highlight_circle = plt.Circle((x_coord, y_coord), radius=20, color='red', fill=False, linewidth=2)
    ax.add_patch(highlight_circle)

    ax.set_title(f'Aneurysm Location on Slice: {slice_sop_uid}')
    ax.axis('off')
    plt.show()

else:
    print(f"Could not parse valid x, y coordinates from: {coords_str}")


import SimpleITK as sitk
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import ast

# --- Define Paths (using the same UID from the example) ---
uid = '1.2.826.0.1.3680043.8.498.10076056930521523789588901704956188485'
dicom_series_path = f"/kaggle/input/rsna-intracranial-aneurysm-detection/series/{uid}"
nii_seg_path = f"/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/{uid}_cowseg.nii"

# --- Step 1: Load Both Volumes with Full Spatial Metadata ---
reader = sitk.ImageSeriesReader()
dicom_names = reader.GetGDCMSeriesFileNames(dicom_series_path)
reader.SetFileNames(dicom_names)
dicom_image_sitk = reader.Execute()
seg_image_sitk = sitk.ReadImage(nii_seg_path)

# --- Step 2: Resample the Segmentation to Match the DICOM Space ---
resampler = sitk.ResampleImageFilter()
resampler.SetReferenceImage(dicom_image_sitk)
resampler.SetInterpolator(sitk.sitkNearestNeighbor)
resampled_seg_sitk = resampler.Execute(seg_image_sitk)

# --- Step 3: Convert Aligned Volumes to NumPy Arrays ---
dicom_vol = sitk.GetArrayFromImage(dicom_image_sitk)
resampled_seg_vol = sitk.GetArrayFromImage(resampled_seg_sitk)

# --- Step 4: Create the MIP and Visualize the Overlay ---
mip = np.max(dicom_vol, axis=0)
y_coords, x_coords = np.where(resampled_seg_vol.sum(axis=0) > 0)

plt.figure(figsize=(10, 10))
plt.imshow(mip, cmap="gray")
plt.scatter(x_coords, y_coords, s=1, c='cyan', alpha=0.2, label="Aligned Segmentation")

# --- THIS IS THE CORRECTED COORDINATE PARSING LOGIC ---
coords_str = df_localizers[df_localizers["SeriesInstanceUID"] == uid].coordinates.iloc[0]
parsed_obj = ast.literal_eval(coords_str)

coords_dict = None
if isinstance(parsed_obj, list) and parsed_obj:
    coords_dict = parsed_obj[0]  # Handles format: [{'x': 1, 'y': 2}]
elif isinstance(parsed_obj, dict):
    coords_dict = parsed_obj      # Handles format: {'x': 1, 'y': 2}

if coords_dict:
    lx, ly = int(coords_dict['x']), int(coords_dict['y'])
    plt.scatter([lx], [ly], c="red", s=40, label="Ground Truth Aneurysm", marker="X")
# ---------------------------------------------------------

plt.title("Correctly Aligned Segmentation on DICOM MIP", fontsize=16)
plt.axis("off")
plt.legend()
plt.show()

