import os
import json
import glob
import warnings
import pydicom
import seaborn as sns
import numpy as np
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from random import sample
from scipy.ndimage import zoom

warnings.filterwarnings("ignore")


# Define paths
train_csv_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv'     
localizers_csv_path = '/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv'
series_dir = '/kaggle/input/rsna-intracranial-aneurysm-detection/series'                

# Load train.csv as df
df = pd.read_csv(train_csv_path)
# Load train_localizers.csv as loc_df
loc_df = pd.read_csv(localizers_csv_path)


import cv2
from scipy.ndimage import rotate

CMAP = 'gray'

def rotate_image_and_coordinates(img, x, y, angle_degrees):
    """
    Rotate image and adjust coordinates accordingly.
    
    Args:
        img: 2D numpy array (image)
        x, y: original coordinates
        angle_degrees: rotation angle in degrees (positive = counterclockwise)
    
    Returns:
        rotated_img: rotated image
        new_x, new_y: adjusted coordinates
    """
    height, width = img.shape
    
    # Find the background value (minimum value in the image, or corners)
    # Using corners to determine background value
    corner_values = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
    background_value = min(corner_values)
    # Alternative: use minimum value in entire image
    # background_value = img.min()
    
    # Rotate image using scipy.ndimage.rotate with proper background value
    rotated_img = rotate(img, angle_degrees, reshape=False, mode='constant', cval=background_value)
    
    # For coordinate transformation, we need to use the same rotation as scipy
    # scipy.ndimage.rotate rotates counterclockwise, so we use the same angle
    angle_rad = np.radians(-angle_degrees)  # Note: negative because of coordinate system
    cos_angle = np.cos(angle_rad)
    sin_angle = np.sin(angle_rad)
    
    # Center of rotation (image center)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    
    # Translate coordinates to origin (center of image)
    x_centered = x - center_x
    y_centered = y - center_y
    
    # Apply rotation transformation
    new_x_centered = x_centered * cos_angle - y_centered * sin_angle
    new_y_centered = x_centered * sin_angle + y_centered * cos_angle
    
    # Translate back to image coordinates
    new_x = new_x_centered + center_x
    new_y = new_y_centered + center_y
    
    # Clamp coordinates to image bounds
    new_x = np.clip(new_x, 0, width - 1)
    new_y = np.clip(new_y, 0, height - 1)
    
    return rotated_img, new_x, new_y

def plot_labeled_slice(row_number, localizers_df, train_df, ax_image, ax_label, ax_overlay, rotation_angle=0):
    row = localizers_df.iloc[row_number]
    series_uid = row['SeriesInstanceUID']
    sop_uid = row['SOPInstanceUID']
    coordinates = row['coordinates']
    location = row['location']
    
    # Get Modality from train_df
    modality = train_df[train_df['SeriesInstanceUID'] == series_uid]['Modality'].iloc[0] if not train_df[train_df['SeriesInstanceUID'] == series_uid].empty else 'Unknown'
    
    # Find the DICOM file for the SOPInstanceUID
    folders = glob.glob("../input/rsna-intracranial-aneurysm-detection/series/*")
    series_path = None
    for folder in folders:
        if series_uid in folder:
            series_path = folder
            break
    
    if not series_path:
        print(f"No folder found for SeriesInstanceUID: {series_uid}")
        return
    
    # Get all DICOM files in the series
    files = sorted(glob.glob(os.path.join(series_path, "*.dcm")))
    dicom_file = None
    for file in files:
        if sop_uid in file:
            dicom_file = file
            break
    
    if not dicom_file:
        print(f"No DICOM file found for SOPInstanceUID: {sop_uid}")
        return
    
    # Load DICOM image
    ex = pydicom.dcmread(dicom_file)
    if len(ex.pixel_array.shape) > 2:
        print(f"Skipping {sop_uid}: Image has more than 2 dimensions")
        return
    
    img = ex.pixel_array
    img = np.flipud(img)  # Flip vertically to match DICOM top-left origin
    height, width = img.shape
    
    # Parse coordinates
    coordinates = json.loads(coordinates.replace("'", '"'))
    x = coordinates["x"]
    y = height - 1 - coordinates["y"]  # Adjust for flipud
    
    # Apply rotation if specified
    if rotation_angle != 0:
        img, x, y = rotate_image_and_coordinates(img, x, y, rotation_angle)
    
    # Plot Image
    ax_image.imshow(img, cmap=CMAP, origin='upper', aspect=1)
    title = f'Image (Rotated {rotation_angle}째)\nSeries: {series_uid[:8]}...' if rotation_angle != 0 else f'Image\nSeries: {series_uid[:8]}...'
    ax_image.set_title(title)
    ax_image.axis('off')
    
    # Plot Label (black background with red rectangle and text)
    ax_label.imshow(np.zeros_like(img), cmap=CMAP, origin='upper', aspect=1)
    rect = patches.Rectangle((x-10, y-10), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
    ax_label.add_patch(rect)
    ax_label.text(x, y+35, location, color='yellow', fontsize=8, ha='center', va='bottom')
    ax_label.text(x, y+55, f'Modality: {modality}', color='red', fontsize=8, ha='center', va='bottom')
    ax_label.set_title('Label')
    ax_label.axis('off')
    
    # Plot Overlay
    ax_overlay.imshow(img, cmap=CMAP, origin='upper', aspect=1)
    rect = patches.Rectangle((x-10, y-10), 20, 20, linewidth=1, edgecolor='r', facecolor='none')
    ax_overlay.add_patch(rect)
    ax_overlay.text(x, y+35, location, color='yellow', fontsize=8, ha='center', va='bottom')
    ax_overlay.text(x, y+55, f'Modality: {modality}', color='red', fontsize=8, ha='center', va='bottom')
    ax_overlay.set_title('Overlay')
    ax_overlay.axis('off')

# Function to plot for a given modality with rotation augmentation
def plot_modality_slices(modality, label=None, row_indices=None, rotation_angles=None):
    """
    Plot modality slices with optional rotation augmentation.
    
    Args:
        modality: Image modality (e.g., 'CTA', 'MRA')
        label: Optional location filter
        row_indices: Optional specific row indices to use
        rotation_angles: List of rotation angles to apply (one per row). If None, no rotation.
    """
    # Filter for series with Aneurysm Present and localization data
    df_aneurysm = df[df['Aneurysm Present'] == 1]
    loc_series = loc_df['SeriesInstanceUID'].unique()
    df_aneurysm_loc = df_aneurysm[df_aneurysm['SeriesInstanceUID'].isin(loc_series)]
    df_modality = df_aneurysm_loc[df_aneurysm_loc['Modality'] == modality]
    
    # Get sorted unique SeriesInstanceUID for the modality
    modality_series_sorted = sorted(df_modality['SeriesInstanceUID'].unique())
    
    # Select rows from loc_df for the modality
    modality_loc_rows = loc_df[loc_df['SeriesInstanceUID'].isin(modality_series_sorted)]
    
    # Optionally filter by label (aneurysm location)
    if label is not None:
        modality_loc_rows = modality_loc_rows[modality_loc_rows['location'] == label]
    
    # If specific row indices are provided, try to use them
    selected_rows = []
    if row_indices is not None:
        for idx in row_indices:
            if idx < len(modality_loc_rows) and modality_loc_rows.iloc[idx]['SeriesInstanceUID'] in modality_series_sorted:
                selected_rows.append(modality_loc_rows.index[idx])
    
    # If fewer than 5 rows selected (or none), fall back to first 5 sorted series
    if len(selected_rows) < 5:
        if row_indices is not None:
            print(f"Requested indices {row_indices} invalid or insufficient. Selecting first {5 - len(selected_rows)} sorted series.")
        remaining_series = [s for s in modality_series_sorted if s not in [loc_df.iloc[idx]['SeriesInstanceUID'] for idx in selected_rows]]
        for series in remaining_series[:5 - len(selected_rows)]:
            series_rows = modality_loc_rows[modality_loc_rows['SeriesInstanceUID'] == series]
            if not series_rows.empty:
                selected_rows.append(series_rows.index[0])
    
    # If fewer than 5 series found, warn
    if len(selected_rows) < 5:
        print(f"Only {len(selected_rows)} {modality} series with aneurysms and localization data found (label: {label}).")
    
    # Set default rotation angles if not provided
    if rotation_angles is None:
        rotation_angles = [0] * len(selected_rows)
    elif len(rotation_angles) < len(selected_rows):
        # Extend rotation_angles to match selected_rows length
        rotation_angles.extend([0] * (len(selected_rows) - len(rotation_angles)))
    
    # Create subplot grid
    fig, axs = plt.subplots(len(selected_rows), 3, figsize=(15, 5 * len(selected_rows)))
    
    # Handle case of single row
    if len(selected_rows) == 1:
        axs = [axs]
    
    # Plot for each selected row with rotation
    for i, row_number in enumerate(selected_rows):
        rotation_angle = rotation_angles[i]
        print(f"\nDisplaying labeled slice for series {loc_df.iloc[row_number]['SeriesInstanceUID'][:8]}... (row {row_number}) with rotation: {rotation_angle}째")
        plot_labeled_slice(row_number, loc_df, df, axs[i][0], axs[i][1], axs[i][2], rotation_angle)

    plt.tight_layout()
    plt.show()


# Plot the same image with different rotations
plot_modality_slices(modality='CTA', label='Left Middle Cerebral Artery', row_indices=[0, 0, 0, 0, 0], rotation_angles=[0, 15, 30, 45, -15])




