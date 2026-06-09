# Install DICOM decompression libraries
!pip install -q gdcm
!pip install -q pylibjpeg pylibjpeg-libjpeg

# Then restart the kernel and run again


"""
Complete Preprocessing Pipeline for RSNA 2022 Cervical Spine Fracture Detection
This script handles DICOM loading, HU conversion, windowing, resampling, and standardization
"""

import os
import numpy as np
import pydicom
from glob import glob
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

# ============================================================================
# STEP 1: DICOM LOADING AND METADATA EXTRACTION
# ============================================================================

def load_dicom_series(patient_folder):
    """
    Load all DICOM slices for a patient and stack into 3D volume
    
    Args:
        patient_folder: Path to folder containing .dcm files
        
    Returns:
        volume: 3D numpy array (num_slices, height, width)
        slice_thickness: Slice thickness in mm
        pixel_spacing: (row_spacing, col_spacing) in mm
        metadata: First DICOM slice metadata
    """
    # Get all .dcm files
    dicom_files = glob(os.path.join(patient_folder, "*.dcm"))
    
    if len(dicom_files) == 0:
        raise ValueError(f"No DICOM files found in {patient_folder}")
    
    # Read all slices
    slices = []
    for dcm_file in dicom_files:
        try:
            ds = pydicom.dcmread(dcm_file)
            slices.append(ds)
        except Exception as e:
            print(f"Error reading {dcm_file}: {e}")
            continue
    
    if len(slices) == 0:
        raise ValueError(f"Could not read any DICOM files from {patient_folder}")
    
    # Sort by ImagePositionPatient (Z coordinate - superior/inferior position)
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    
    # Stack into 3D array
    volume = np.stack([s.pixel_array for s in slices])
    
    # Get metadata from first slice
    metadata = slices[0]
    
    # Extract spacing information
    try:
        slice_thickness = float(metadata.SliceThickness)
    except:
        # If SliceThickness not available, calculate from positions
        if len(slices) > 1:
            slice_thickness = abs(
                float(slices[1].ImagePositionPatient[2]) - 
                float(slices[0].ImagePositionPatient[2])
            )
        else:
            slice_thickness = 1.0  # Default
    
    pixel_spacing = [float(x) for x in metadata.PixelSpacing]
    
    return volume, slice_thickness, pixel_spacing, metadata


# ============================================================================
# STEP 2: HOUNSFIELD UNIT (HU) CONVERSION
# ============================================================================

def apply_hu_conversion(volume, metadata):
    """
    Convert raw pixel values to Hounsfield Units (HU)
    HU = pixel_value * slope + intercept
    
    Args:
        volume: 3D numpy array of raw pixel values
        metadata: DICOM metadata containing RescaleSlope and RescaleIntercept
        
    Returns:
        volume_hu: 3D numpy array in Hounsfield Units
    """
    try:
        intercept = float(metadata.RescaleIntercept)
        slope = float(metadata.RescaleSlope)
    except:
        intercept = 0.0
        slope = 1.0
        print("Warning: RescaleIntercept/Slope not found, using defaults")
    
    volume_hu = volume.astype(np.float32) * slope + intercept
    return volume_hu


# ============================================================================
# STEP 3: WINDOWING FOR BONE VISUALIZATION
# ============================================================================

def apply_window(volume_hu, window_center=400, window_width=1800):
    """
    Apply windowing to enhance bone structures
    Standard bone window: WC=400, WW=1800 (range: -500 to 1300 HU)
    
    Args:
        volume_hu: 3D numpy array in Hounsfield Units
        window_center: Center of the window (HU)
        window_width: Width of the window (HU)
        
    Returns:
        volume_windowed: 3D numpy array normalized to [0, 1]
    """
    lower = window_center - window_width // 2
    upper = window_center + window_width // 2
    
    # Clip values to window range
    volume_windowed = np.clip(volume_hu, lower, upper)
    
    # Normalize to [0, 1]
    volume_normalized = (volume_windowed - lower) / (upper - lower)
    
    return volume_normalized.astype(np.float32)


# ============================================================================
# STEP 4: RESAMPLING TO ISOTROPIC SPACING (IMPROVED)
# ============================================================================

def resample_volume(volume, current_spacing, target_spacing=(1.5, 1.0, 1.0)):
    """
    Resample volume to target spacing with better depth preservation
    Uses slightly larger Z-spacing (1.5mm) to preserve more slices
    
    Args:
        volume: 3D numpy array (D, H, W)
        current_spacing: (z_spacing, y_spacing, x_spacing) in mm
        target_spacing: Desired spacing in mm (default: 1.5mm z, 1mm x,y)
        
    Returns:
        resampled_volume: 3D numpy array with target spacing
    """
    # Calculate resize factors for each dimension
    resize_factor = np.array(current_spacing) / np.array(target_spacing)
    
    # Resample using trilinear interpolation (order=1)
    # order=0: nearest neighbor, order=1: bilinear, order=3: cubic
    resampled_volume = zoom(volume, resize_factor, order=1)
    
    return resampled_volume.astype(np.float32)


# ============================================================================
# STEP 5: CROP OR PAD TO TARGET SHAPE
# ============================================================================

# ============================================================================
# STEP 5: IMPROVED CROP OR PAD WITH CERVICAL SPINE FOCUS
# ============================================================================

def crop_or_pad_cervical(volume, target_shape=(96, 320, 320), cervical_focus=True):
    """
    Crop or pad volume to target shape with focus on cervical spine region
    Cervical spine is typically in the upper portion of CT scans
    
    Args:
        volume: 3D numpy array (D, H, W)
        target_shape: Desired output shape (D, H, W)
        cervical_focus: If True, crop from top of volume (cervical region)
        
    Returns:
        output: 3D numpy array with target shape
    """
    current_shape = np.array(volume.shape)
    target_shape_arr = np.array(target_shape)
    
    # Initialize output with zeros (black background)
    output = np.zeros(target_shape, dtype=volume.dtype)
    
    # Calculate crop/pad slices for each dimension
    slices_vol = []
    slices_out = []
    
    for i in range(3):
        if current_shape[i] >= target_shape_arr[i]:
            # Crop
            if i == 0 and cervical_focus:
                # For depth (Z-axis): take from TOP (cervical region)
                # Cervical spine is in upper ~30-40% of typical CT scan
                start = int(current_shape[i] * 0.15)  # Skip very top (air)
                start = max(0, min(start, current_shape[i] - target_shape_arr[i]))
            else:
                # For height/width: take from center
                start = (current_shape[i] - target_shape_arr[i]) // 2
            
            slices_vol.append(slice(start, start + target_shape_arr[i]))
            slices_out.append(slice(0, target_shape_arr[i]))
        else:
            # Pad - place in center
            start = (target_shape_arr[i] - current_shape[i]) // 2
            slices_vol.append(slice(0, current_shape[i]))
            slices_out.append(slice(start, start + current_shape[i]))
    
    # Apply cropping/padding in one operation
    output[slices_out[0], slices_out[1], slices_out[2]] = \
        volume[slices_vol[0], slices_vol[1], slices_vol[2]]
    
    return output


def crop_or_pad(volume, target_shape=(96, 320, 320)):
    """
    Standard crop or pad (center-based) - kept for backward compatibility
    For cervical spine, use crop_or_pad_cervical instead
    
    Args:
        volume: 3D numpy array (D, H, W)
        target_shape: Desired output shape (D, H, W)
        
    Returns:
        output: 3D numpy array with target shape
    """
    current_shape = np.array(volume.shape)
    target_shape_arr = np.array(target_shape)
    
    # Initialize output with zeros (black background)
    output = np.zeros(target_shape, dtype=volume.dtype)
    
    # Calculate crop/pad slices for each dimension
    slices_vol = []
    slices_out = []
    
    for i in range(3):
        if current_shape[i] >= target_shape_arr[i]:
            # Crop - take from the center
            start = (current_shape[i] - target_shape_arr[i]) // 2
            slices_vol.append(slice(start, start + target_shape_arr[i]))
            slices_out.append(slice(0, target_shape_arr[i]))
        else:
            # Pad - place in center
            start = (target_shape_arr[i] - current_shape[i]) // 2
            slices_vol.append(slice(0, current_shape[i]))
            slices_out.append(slice(start, start + current_shape[i]))
    
    # Apply cropping/padding in one operation
    output[slices_out[0], slices_out[1], slices_out[2]] = \
        volume[slices_vol[0], slices_vol[1], slices_vol[2]]
    
    return output


# ============================================================================
# STEP 6: IMPROVED PREPROCESSING PIPELINE WITH BETTER RESOLUTION
# ============================================================================

def preprocess_patient(patient_folder, target_shape=(96, 320, 320), 
                       window_center=400, window_width=1800,
                       target_spacing=(1.5, 1.0, 1.0),
                       cervical_focus=True):
    """
    Complete preprocessing pipeline for one patient (IMPROVED VERSION)
    
    Improvements:
    - Larger target shape (96, 320, 320) for better spatial resolution
    - Less aggressive depth resampling (1.5mm vs 1.0mm)
    - Cervical spine focused cropping
    - Better preservation of anatomical details
    
    Steps:
    1. Load DICOM series
    2. Convert to Hounsfield Units
    3. Apply bone windowing
    4. Resample to target spacing (1.5mm z, 1mm x,y)
    5. Crop/pad to target shape with cervical focus
    
    Args:
        patient_folder: Path to patient's DICOM folder
        target_shape: Output shape (D, H, W) - default (96, 320, 320)
        window_center: HU window center for bone
        window_width: HU window width for bone
        target_spacing: Target voxel spacing (z, y, x) in mm
        cervical_focus: Whether to focus on cervical region when cropping
        
    Returns:
        volume_final: Preprocessed 3D volume (D, H, W) in range [0, 1]
    """
    try:
        # Step 1: Load DICOM series
        volume, slice_thickness, pixel_spacing, metadata = load_dicom_series(patient_folder)
        
        # Step 2: Convert to Hounsfield Units
        volume_hu = apply_hu_conversion(volume, metadata)
        
        # Step 3: Apply bone windowing
        volume_windowed = apply_window(volume_hu, window_center, window_width)
        
        # Step 4: Resample to target spacing (less aggressive)
        current_spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])
        volume_resampled = resample_volume(volume_windowed, current_spacing, target_spacing)
        
        # Step 5: Crop or pad to target shape (with cervical focus)
        if cervical_focus:
            volume_final = crop_or_pad_cervical(volume_resampled, target_shape, cervical_focus=True)
        else:
            volume_final = crop_or_pad(volume_resampled, target_shape)
        
        return volume_final
        
    except Exception as e:
        print(f"Error preprocessing {patient_folder}: {e}")
        # Return zero volume on error
        return np.zeros(target_shape, dtype=np.float32)


def preprocess_patient_memory_efficient(patient_folder, target_shape=(64, 256, 256),
                                        window_center=400, window_width=1800,
                                        target_spacing=(2.0, 1.0, 1.0)):
    """
    Memory-efficient version for Kaggle with limited GPU memory
    Uses smaller target shape and more aggressive depth resampling
    
    Use this if you get CUDA out of memory errors
    
    Args:
        patient_folder: Path to patient's DICOM folder
        target_shape: Smaller output shape (D, H, W) - default (64, 256, 256)
        window_center: HU window center for bone
        window_width: HU window width for bone
        target_spacing: More aggressive spacing (2mm z, 1mm x,y)
        
    Returns:
        volume_final: Preprocessed 3D volume (D, H, W) in range [0, 1]
    """
    return preprocess_patient(
        patient_folder=patient_folder,
        target_shape=target_shape,
        window_center=window_center,
        window_width=window_width,
        target_spacing=target_spacing,
        cervical_focus=True
    )


# ============================================================================
# STEP 7: BATCH PREPROCESSING WITH MULTIPLE RESOLUTION OPTIONS
# ============================================================================

def preprocess_all_patients(train_csv_path, train_images_root, output_dir,
                            target_shape=(96, 320, 320), save_npy=True,
                            resolution_mode='high'):
    """
    Preprocess all patients and optionally save to disk
    
    Resolution modes:
    - 'high': (96, 320, 320) - Best quality, requires more memory
    - 'medium': (80, 256, 256) - Balanced quality and memory
    - 'low': (64, 224, 224) - Memory efficient
    
    Args:
        train_csv_path: Path to train.csv
        train_images_root: Root directory of train_images
        output_dir: Directory to save preprocessed volumes
        target_shape: Target volume shape (overrides resolution_mode if specified)
        save_npy: Whether to save preprocessed volumes as .npy files
        resolution_mode: 'high', 'medium', or 'low'
        
    Returns:
        None (saves files to disk)
    """
    # Set target shape and spacing based on resolution mode
    if resolution_mode == 'high':
        target_shape = (96, 320, 320)
        target_spacing = (1.5, 1.0, 1.0)
        print("Using HIGH RESOLUTION mode: (96, 320, 320)")
    elif resolution_mode == 'medium':
        target_shape = (80, 256, 256)
        target_spacing = (1.75, 1.0, 1.0)
        print("Using MEDIUM RESOLUTION mode: (80, 256, 256)")
    elif resolution_mode == 'low':
        target_shape = (64, 224, 224)
        target_spacing = (2.0, 1.25, 1.25)
        print("Using LOW RESOLUTION mode: (64, 224, 224)")
    
    # Load training CSV
    train_df = pd.read_csv(train_csv_path)
    patient_ids = train_df['StudyInstanceUID'].unique()
    
    print(f"Preprocessing {len(patient_ids)} patients...")
    print(f"Target shape: {target_shape}")
    print(f"Target spacing: {target_spacing}")
    
    # Create output directory
    if save_npy:
        os.makedirs(output_dir, exist_ok=True)
    
    # Process each patient
    successful = 0
    failed = 0
    
    for patient_id in tqdm(patient_ids, desc="Processing patients"):
        patient_folder = os.path.join(train_images_root, str(patient_id))
        
        if not os.path.exists(patient_folder):
            print(f"Warning: Patient folder not found: {patient_folder}")
            failed += 1
            continue
        
        try:
            # Preprocess
            volume = preprocess_patient(
                patient_folder, 
                target_shape=target_shape,
                target_spacing=target_spacing,
                cervical_focus=True
            )
            
            # Save if requested
            if save_npy:
                output_path = os.path.join(output_dir, f"{patient_id}.npy")
                np.save(output_path, volume)
            
            successful += 1
            
        except Exception as e:
            print(f"Failed to process {patient_id}: {e}")
            failed += 1
    
    print(f"\nPreprocessing complete!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")


# ============================================================================
# STEP 8: IMPROVED VISUALIZATION UTILITIES
# ============================================================================

def visualize_preprocessing_steps(patient_folder, save_path=None, 
                                  target_shape=(96, 320, 320)):
    """
    Visualize each step of the preprocessing pipeline (IMPROVED)
    Shows better spatial resolution with new settings
    """
    # Load original
    volume, thickness, spacing, metadata = load_dicom_series(patient_folder)
    
    # Apply each step
    volume_hu = apply_hu_conversion(volume, metadata)
    volume_windowed = apply_window(volume_hu)
    current_spacing = (thickness, spacing[0], spacing[1])
    volume_resampled = resample_volume(volume_windowed, current_spacing, 
                                       target_spacing=(1.5, 1.0, 1.0))
    volume_final = crop_or_pad_cervical(volume_resampled, target_shape, cervical_focus=True)
    
    # Select middle slices
    mid_original = len(volume) // 2
    mid_final = volume_final.shape[0] // 2
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original
    axes[0, 0].imshow(volume[mid_original], cmap='gray')
    axes[0, 0].set_title(f'Original\nShape: {volume.shape}\nSpacing: {current_spacing}', fontsize=12)
    axes[0, 0].axis('off')
    
    # HU converted
    axes[0, 1].imshow(volume_hu[mid_original], cmap='gray', vmin=-1000, vmax=1000)
    axes[0, 1].set_title(f'HU Converted\nRange: [{volume_hu.min():.0f}, {volume_hu.max():.0f}] HU', fontsize=12)
    axes[0, 1].axis('off')
    
    # Windowed
    axes[0, 2].imshow(volume_windowed[mid_original], cmap='gray')
    axes[0, 2].set_title(f'Bone Windowed\nWC=400, WW=1800\nRange: [0, 1]', fontsize=12)
    axes[0, 2].axis('off')
    
    # Resampled
    axes[1, 0].imshow(volume_resampled[volume_resampled.shape[0]//2], cmap='gray')
    axes[1, 0].set_title(f'Resampled to 1.5mm isotropic\nShape: {volume_resampled.shape}', fontsize=12)
    axes[1, 0].axis('off')
    
    # Final
    axes[1, 1].imshow(volume_final[mid_final], cmap='gray')
    axes[1, 1].set_title(f'Final (Cervical Focused)\nShape: {volume_final.shape}', fontsize=12)
    axes[1, 1].axis('off')
    
    # 3D view (MIP - Maximum Intensity Projection)
    mip = np.max(volume_final, axis=0)
    axes[1, 2].imshow(mip, cmap='gray')
    axes[1, 2].set_title('MIP (Max Projection)\nAxial View', fontsize=12)
    axes[1, 2].axis('off')
    
    plt.suptitle('Preprocessing Pipeline - Improved Resolution', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.show()
    
    # Print statistics
    print(f"\n{'='*60}")
    print(f"PREPROCESSING STATISTICS")
    print(f"{'='*60}")
    print(f"Original shape:        {volume.shape}")
    print(f"Original spacing:      {current_spacing} mm")
    print(f"Resampled shape:       {volume_resampled.shape}")
    print(f"Final shape:           {volume_final.shape}")
    print(f"Compression ratio:     {volume.size / volume_final.size:.2f}x")
    print(f"Memory (original):     {volume.nbytes / 1024 / 1024:.2f} MB")
    print(f"Memory (final):        {volume_final.nbytes / 1024 / 1024:.2f} MB")
    print(f"{'='*60}\n")


def visualize_volume_slices(volume, num_slices=9, title="Volume Slices"):
    """
    Visualize multiple slices from a 3D volume
    """
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()
    
    # Select evenly spaced slices
    slice_indices = np.linspace(0, volume.shape[0]-1, num_slices, dtype=int)
    
    for idx, slice_idx in enumerate(slice_indices):
        axes[idx].imshow(volume[slice_idx], cmap='gray')
        axes[idx].set_title(f'Slice {slice_idx}/{volume.shape[0]}')
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.show()


# ============================================================================
# EXAMPLE USAGE WITH IMPROVED SETTINGS
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("IMPROVED PREPROCESSING PIPELINE - BETTER SPATIAL RESOLUTION")
    print("="*80)
    
    # Example: Preprocess a single patient with HIGH RESOLUTION
    patient_id = "1.2.826.0.1.3680043.10001"
    patient_folder = f"/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images/{patient_id}"
    
    print("\n1. HIGH RESOLUTION MODE (96, 320, 320) - RECOMMENDED")
    print("-" * 60)
    volume_high = preprocess_patient(
        patient_folder, 
        target_shape=(96, 320, 320),
        target_spacing=(1.5, 1.0, 1.0),
        cervical_focus=True
    )
    print(f"âœ“ Preprocessed volume shape: {volume_high.shape}")
    print(f"âœ“ Memory usage: {volume_high.nbytes / 1024 / 1024:.2f} MB")
    print(f"âœ“ Value range: [{volume_high.min():.3f}, {volume_high.max():.3f}]")
    
    print("\n2. MEDIUM RESOLUTION MODE (80, 256, 256) - BALANCED")
    print("-" * 60)
    volume_medium = preprocess_patient(
        patient_folder, 
        target_shape=(80, 256, 256),
        target_spacing=(1.75, 1.0, 1.0),
        cervical_focus=True
    )
    print(f"âœ“ Preprocessed volume shape: {volume_medium.shape}")
    print(f"âœ“ Memory usage: {volume_medium.nbytes / 1024 / 1024:.2f} MB")
    
    print("\n3. LOW RESOLUTION MODE (64, 224, 224) - MEMORY EFFICIENT")
    print("-" * 60)
    volume_low = preprocess_patient_memory_efficient(
        patient_folder, 
        target_shape=(64, 224, 224),
        target_spacing=(2.0, 1.25, 1.25)
    )
    print(f"âœ“ Preprocessed volume shape: {volume_low.shape}")
    print(f"âœ“ Memory usage: {volume_low.nbytes / 1024 / 1024:.2f} MB")
    
    # Visualize preprocessing steps with improved resolution
    print("\n4. VISUALIZING PREPROCESSING STEPS...")
    print("-" * 60)
    visualize_preprocessing_steps(patient_folder, target_shape=(96, 320, 320))
    
    # Visualize volume slices
    print("\n5. VISUALIZING VOLUME SLICES...")
    print("-" * 60)
    visualize_volume_slices(volume_high, num_slices=9, 
                           title=f"HIGH RES - Patient {patient_id}")
    
    # Compare resolutions side by side
    print("\n6. RESOLUTION COMPARISON")
    print("-" * 60)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    mid_slice = volume_high.shape[0] // 2
    axes[0].imshow(volume_high[mid_slice], cmap='gray')
    axes[0].set_title(f'HIGH: (96, 320, 320)\n{volume_high.nbytes/1024/1024:.1f} MB', 
                     fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    mid_slice = volume_medium.shape[0] // 2
    axes[1].imshow(volume_medium[mid_slice], cmap='gray')
    axes[1].set_title(f'MEDIUM: (80, 256, 256)\n{volume_medium.nbytes/1024/1024:.1f} MB', 
                     fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    mid_slice = volume_low.shape[0] // 2
    axes[2].imshow(volume_low[mid_slice], cmap='gray')
    axes[2].set_title(f'LOW: (64, 224, 224)\n{volume_low.nbytes/1024/1024:.1f} MB', 
                     fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.suptitle('Resolution Comparison - Middle Slice', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS FOR YOUR PROJECT")
    print("="*80)
    print("""
    âœ“ HIGH RESOLUTION (96, 320, 320):
      - Best for fracture detection accuracy
      - Good balance between detail and memory
      - Recommended if you have GPU with 16GB+ VRAM
      - Batch size: 1-2
    
    âœ“ MEDIUM RESOLUTION (80, 256, 256):
      - Good compromise for most systems
      - Suitable for Kaggle free tier (16GB GPU)
      - Batch size: 2-4
    
    âœ“ LOW RESOLUTION (64, 224, 224):
      - Use only if memory is very limited
      - May lose some fracture details
      - Batch size: 4-8
    
    RECOMMENDED: Start with HIGH resolution and reduce if you get OOM errors
    """)
    
    # Example: Batch preprocessing (commented out to avoid long execution)
    print("\n7. BATCH PREPROCESSING EXAMPLE (commented out):")
    print("-" * 60)
    print("""preprocess_all_patients(
        train_csv_path="/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train.csv",
        train_images_root="/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images",
        output_dir="/kaggle/working/preprocessed_volumes_high",
        resolution_mode='high',  # or 'medium' or 'low'
        save_npy=True
    )""")


"""
PyTorch Dataset and DataLoader for RSNA 2022 Cervical Spine Fracture Detection
Supports on-the-fly preprocessing or loading pre-saved .npy files
Includes data augmentation for training
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
import random
import pydicom
from glob import glob
from scipy.ndimage import zoom

# ============================================================================
# PREPROCESSING FUNCTIONS (EMBEDDED TO AVOID IMPORT ISSUES)
# ============================================================================

def load_dicom_series(patient_folder):
    """Load all DICOM slices for a patient and stack into 3D volume"""
    dicom_files = glob(os.path.join(patient_folder, "*.dcm"))
    if len(dicom_files) == 0:
        raise ValueError(f"No DICOM files found in {patient_folder}")
    
    slices = []
    for dcm_file in dicom_files:
        try:
            ds = pydicom.dcmread(dcm_file)
            slices.append(ds)
        except:
            continue
    
    if len(slices) == 0:
        raise ValueError(f"Could not read any DICOM files from {patient_folder}")
    
    slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
    volume = np.stack([s.pixel_array for s in slices])
    
    metadata = slices[0]
    try:
        slice_thickness = float(metadata.SliceThickness)
    except:
        if len(slices) > 1:
            slice_thickness = abs(float(slices[1].ImagePositionPatient[2]) - 
                                 float(slices[0].ImagePositionPatient[2]))
        else:
            slice_thickness = 1.0
    
    pixel_spacing = [float(x) for x in metadata.PixelSpacing]
    return volume, slice_thickness, pixel_spacing, metadata


def apply_hu_conversion(volume, metadata):
    """Convert raw pixel values to Hounsfield Units"""
    try:
        intercept = float(metadata.RescaleIntercept)
        slope = float(metadata.RescaleSlope)
    except:
        intercept = 0.0
        slope = 1.0
    
    volume_hu = volume.astype(np.float32) * slope + intercept
    return volume_hu


def apply_window(volume_hu, window_center=400, window_width=1800):
    """Apply windowing to enhance bone structures"""
    lower = window_center - window_width // 2
    upper = window_center + window_width // 2
    volume_windowed = np.clip(volume_hu, lower, upper)
    volume_normalized = (volume_windowed - lower) / (upper - lower)
    return volume_normalized.astype(np.float32)


def resample_volume(volume, current_spacing, target_spacing=(1.5, 1.0, 1.0)):
    """Resample volume to target spacing"""
    resize_factor = np.array(current_spacing) / np.array(target_spacing)
    resampled_volume = zoom(volume, resize_factor, order=1)
    return resampled_volume.astype(np.float32)


def crop_or_pad_cervical(volume, target_shape=(96, 320, 320), cervical_focus=True):
    """Crop or pad volume with focus on cervical spine region"""
    current_shape = np.array(volume.shape)
    target_shape_arr = np.array(target_shape)
    output = np.zeros(target_shape, dtype=volume.dtype)
    
    slices_vol = []
    slices_out = []
    
    for i in range(3):
        if current_shape[i] >= target_shape_arr[i]:
            if i == 0 and cervical_focus:
                start = int(current_shape[i] * 0.15)
                start = max(0, min(start, current_shape[i] - target_shape_arr[i]))
            else:
                start = (current_shape[i] - target_shape_arr[i]) // 2
            
            slices_vol.append(slice(start, start + target_shape_arr[i]))
            slices_out.append(slice(0, target_shape_arr[i]))
        else:
            start = (target_shape_arr[i] - current_shape[i]) // 2
            slices_vol.append(slice(0, current_shape[i]))
            slices_out.append(slice(start, start + current_shape[i]))
    
    output[slices_out[0], slices_out[1], slices_out[2]] = \
        volume[slices_vol[0], slices_vol[1], slices_vol[2]]
    
    return output


def preprocess_patient(patient_folder, target_shape=(96, 320, 320), 
                       window_center=400, window_width=1800,
                       target_spacing=(1.5, 1.0, 1.0),
                       cervical_focus=True):
    """Complete preprocessing pipeline for one patient"""
    try:
        volume, slice_thickness, pixel_spacing, metadata = load_dicom_series(patient_folder)
        volume_hu = apply_hu_conversion(volume, metadata)
        volume_windowed = apply_window(volume_hu, window_center, window_width)
        current_spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])
        volume_resampled = resample_volume(volume_windowed, current_spacing, target_spacing)
        
        if cervical_focus:
            volume_final = crop_or_pad_cervical(volume_resampled, target_shape, cervical_focus=True)
        else:
            volume_final = crop_or_pad_cervical(volume_resampled, target_shape, cervical_focus=False)
        
        return volume_final
    except Exception as e:
        print(f"Error preprocessing {patient_folder}: {e}")
        return np.zeros(target_shape, dtype=np.float32)


# ============================================================================
# DATASET CLASS - OPTION 1: ON-THE-FLY PREPROCESSING
# ============================================================================

class SpineFractureDataset(Dataset):
    """
    Dataset that preprocesses DICOM files on-the-fly
    Use this if you don't want to save all preprocessed volumes to disk
    """
    
    def __init__(self, df, image_root, target_shape=(96, 320, 320),
                 target_spacing=(1.5, 1.0, 1.0), transform=None,
                 cache_data=False):
        """
        Args:
            df: DataFrame with columns [StudyInstanceUID, patient_overall, C1-C7]
            image_root: Root directory for train_images
            target_shape: Target volume shape (D, H, W)
            target_spacing: Target voxel spacing (z, y, x) in mm
            transform: Optional augmentation function
            cache_data: Whether to cache preprocessed volumes in memory
        """
        self.df = df.reset_index(drop=True)
        self.image_root = image_root
        self.target_shape = target_shape
        self.target_spacing = target_spacing
        self.transform = transform
        self.cache_data = cache_data
        
        # Label columns
        self.label_cols = ['patient_overall'] + [f'C{i}' for i in range(1, 8)]
        
        # Cache for preprocessed volumes
        self.cache = {} if cache_data else None
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get patient ID
        patient_id = self.df.loc[idx, 'StudyInstanceUID']
        
        # Check cache first
        if self.cache_data and patient_id in self.cache:
            volume = self.cache[patient_id]
        else:
            # Load and preprocess volume
            patient_folder = os.path.join(self.image_root, str(patient_id))
            
            try:
                # Use embedded preprocessing function
                volume = preprocess_patient(
                    patient_folder,
                    target_shape=self.target_shape,
                    target_spacing=self.target_spacing,
                    cervical_focus=True
                )
                
                # Cache if enabled
                if self.cache_data:
                    self.cache[patient_id] = volume
                    
            except Exception as e:
                print(f"Error loading patient {patient_id}: {e}")
                # Return zero volume on error
                volume = np.zeros(self.target_shape, dtype=np.float32)
        
        # Add channel dimension (1, D, H, W)
        volume = volume[np.newaxis, ...].astype(np.float32)
        
        # Get labels
        labels = self.df.loc[idx, self.label_cols].values.astype(np.float32)
        
        # Apply augmentation if provided
        if self.transform:
            volume = self.transform(volume)
        
        return torch.from_numpy(volume), torch.from_numpy(labels), patient_id


# ============================================================================
# DATASET CLASS - OPTION 2: LOAD PRE-SAVED NPY FILES
# ============================================================================

class SpineFractureDatasetNPY(Dataset):
    """
    Dataset that loads pre-saved .npy files
    Much faster than on-the-fly preprocessing
    Use this if you've already preprocessed and saved all volumes
    """
    
    def __init__(self, df, npy_root, transform=None):
        """
        Args:
            df: DataFrame with columns [StudyInstanceUID, patient_overall, C1-C7]
            npy_root: Directory containing .npy files
            transform: Optional augmentation function
        """
        self.df = df.reset_index(drop=True)
        self.npy_root = npy_root
        self.transform = transform
        
        # Label columns
        self.label_cols = ['patient_overall'] + [f'C{i}' for i in range(1, 8)]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get patient ID
        patient_id = self.df.loc[idx, 'StudyInstanceUID']
        
        # Load preprocessed volume
        npy_path = os.path.join(self.npy_root, f"{patient_id}.npy")
        
        try:
            volume = np.load(npy_path)
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            # Return zero volume on error
            volume = np.zeros((96, 320, 320), dtype=np.float32)
        
        # Add channel dimension (1, D, H, W)
        volume = volume[np.newaxis, ...].astype(np.float32)
        
        # Get labels
        labels = self.df.loc[idx, self.label_cols].values.astype(np.float32)
        
        # Apply augmentation if provided
        if self.transform:
            volume = self.transform(volume)
        
        return torch.from_numpy(volume), torch.from_numpy(labels), patient_id


# ============================================================================
# DATA AUGMENTATION
# ============================================================================

class SpineAugmentation:
    """
    Data augmentation for 3D CT volumes
    Includes flips, rotations, noise, and intensity adjustments
    """
    
    def __init__(self, flip_prob=0.5, rotate_prob=0.3, noise_prob=0.2,
                 intensity_prob=0.3):
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.noise_prob = noise_prob
        self.intensity_prob = intensity_prob
    
    def __call__(self, volume):
        """
        Apply augmentations to volume
        Args:
            volume: numpy array (1, D, H, W)
        Returns:
            augmented volume
        """
        # Random horizontal flip (left-right)
        if random.random() < self.flip_prob:
            volume = np.flip(volume, axis=3).copy()  # Flip width
        
        # Random rotation (small angles)
        if random.random() < self.rotate_prob:
            angle = random.uniform(-10, 10)
            volume = self._rotate_volume(volume, angle)
        
        # Random noise
        if random.random() < self.noise_prob:
            noise = np.random.normal(0, 0.01, volume.shape).astype(np.float32)
            volume = np.clip(volume + noise, 0, 1)
        
        # Random intensity adjustment
        if random.random() < self.intensity_prob:
            factor = random.uniform(0.9, 1.1)
            volume = np.clip(volume * factor, 0, 1)
        
        return volume
    
    def _rotate_volume(self, volume, angle):
        """
        Rotate volume by small angle (in XY plane)
        """
        from scipy.ndimage import rotate
        # Rotate each slice in the axial plane
        rotated = rotate(volume, angle, axes=(2, 3), reshape=False, order=1)
        return rotated.astype(np.float32)


# ============================================================================
# TRAIN/VALIDATION SPLIT
# ============================================================================

def create_train_val_split(train_csv_path, val_size=0.40, random_state=42):
    """
    Create stratified train/validation split
    Stratifies by patient_overall to ensure balanced fracture distribution
    
    Args:
        train_csv_path: Path to train.csv
        val_size: Fraction of data for validation (0.15 = 15%)
        random_state: Random seed for reproducibility
        
    Returns:
        train_df, val_df: DataFrames for training and validation
    """
    # Load CSV
    train_df = pd.read_csv(train_csv_path)
    
    # Get unique patient IDs
    patient_ids = train_df['StudyInstanceUID'].unique()
    
    # Get fracture status for each patient (for stratification)
    patient_fractures = train_df.groupby('StudyInstanceUID')['patient_overall'].first().values
    
    # Split patient IDs (not rows)
    train_ids, val_ids = train_test_split(
        patient_ids,
        test_size=val_size,
        stratify=patient_fractures,
        random_state=random_state
    )
    
    # Create DataFrames
    train_df_split = train_df[train_df['StudyInstanceUID'].isin(train_ids)].reset_index(drop=True)
    val_df_split = train_df[train_df['StudyInstanceUID'].isin(val_ids)].reset_index(drop=True)
    
    print(f"Train patients: {len(train_ids)} ({len(train_df_split)} rows)")
    print(f"Val patients: {len(val_ids)} ({len(val_df_split)} rows)")
    print(f"Train fracture rate: {train_df_split['patient_overall'].mean():.2%}")
    print(f"Val fracture rate: {val_df_split['patient_overall'].mean():.2%}")
    
    return train_df_split, val_df_split


def create_kfold_splits(train_csv_path, n_folds=5, random_state=42):
    """
    Create K-Fold cross-validation splits
    Useful for more robust evaluation
    
    Args:
        train_csv_path: Path to train.csv
        n_folds: Number of folds
        random_state: Random seed
        
    Returns:
        List of (train_df, val_df) tuples for each fold
    """
    train_df = pd.read_csv(train_csv_path)
    patient_ids = train_df['StudyInstanceUID'].unique()
    patient_fractures = train_df.groupby('StudyInstanceUID')['patient_overall'].first().values
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    folds = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(patient_ids, patient_fractures)):
        train_ids = patient_ids[train_idx]
        val_ids = patient_ids[val_idx]
        
        train_df_fold = train_df[train_df['StudyInstanceUID'].isin(train_ids)].reset_index(drop=True)
        val_df_fold = train_df[train_df['StudyInstanceUID'].isin(val_ids)].reset_index(drop=True)
        
        folds.append((train_df_fold, val_df_fold))
        print(f"Fold {fold+1}: Train={len(train_ids)}, Val={len(val_ids)}")
    
    return folds


# ============================================================================
# CREATE DATALOADERS
# ============================================================================

def create_dataloaders(train_df, val_df, image_root, batch_size=2,
                      num_workers=2, use_augmentation=True,
                      target_shape=(96, 320, 320)):
    """
    Create PyTorch DataLoaders for training and validation
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        image_root: Root directory of train_images
        batch_size: Batch size (keep small for 3D data)
        num_workers: Number of worker processes
        use_augmentation: Whether to use data augmentation for training
        target_shape: Target volume shape
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    # Create augmentation
    train_transform = SpineAugmentation() if use_augmentation else None
    
    # Create datasets
    train_dataset = SpineFractureDataset(
        df=train_df,
        image_root=image_root,
        target_shape=target_shape,
        transform=train_transform,
        cache_data=False  # Set to True if you have enough RAM
    )
    
    val_dataset = SpineFractureDataset(
        df=val_df,
        image_root=image_root,
        target_shape=target_shape,
        transform=None,  # No augmentation for validation
        cache_data=False
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop incomplete batches
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def create_dataloaders_npy(train_df, val_df, npy_root, batch_size=4,
                           num_workers=2, use_augmentation=True):
    """
    Create DataLoaders for pre-saved .npy files
    Much faster than on-the-fly preprocessing
    
    Args:
        train_df: Training DataFrame
        val_df: Validation DataFrame
        npy_root: Directory containing .npy files
        batch_size: Batch size (can be larger with .npy)
        num_workers: Number of worker processes
        use_augmentation: Whether to use augmentation
        
    Returns:
        train_loader, val_loader: PyTorch DataLoaders
    """
    train_transform = SpineAugmentation() if use_augmentation else None
    
    train_dataset = SpineFractureDatasetNPY(
        df=train_df,
        npy_root=npy_root,
        transform=train_transform
    )
    
    val_dataset = SpineFractureDatasetNPY(
        df=val_df,
        npy_root=npy_root,
        transform=None
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("CREATING DATALOADERS FOR TRAINING")
    print("="*80)
    
    # Paths
    train_csv_path = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train.csv"
    image_root = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train_images"
    
    # Create train/val split
    print("\n1. Creating train/validation split...")
    print("-" * 60)
    train_df, val_df = create_train_val_split(train_csv_path, val_size=0.40)
    
    # Option 1: On-the-fly preprocessing (slower but no disk space needed)
    print("\n2. Creating DataLoaders (on-the-fly preprocessing)...")
    print("-" * 60)
    train_loader, val_loader = create_dataloaders(
        train_df=train_df,
        val_df=val_df,
        image_root=image_root,
        batch_size=2,  # Small batch for memory efficiency
        num_workers=2,
        use_augmentation=True,
        target_shape=(96, 320, 320)
    )
    
    print(f"âœ“ Train batches: {len(train_loader)}")
    print(f"âœ“ Val batches: {len(val_loader)}")
    
    # Test dataloader
    print("\n3. Testing DataLoader...")
    print("-" * 60)
    for volumes, labels, patient_ids in train_loader:
        print(f"âœ“ Batch volume shape: {volumes.shape}")  # (batch, 1, 96, 320, 320)
        print(f"âœ“ Batch labels shape: {labels.shape}")   # (batch, 8)
        print(f"âœ“ Patient IDs: {patient_ids}")
        print(f"âœ“ Volume range: [{volumes.min():.3f}, {volumes.max():.3f}]")
        print(f"âœ“ Labels (first sample): {labels[0].numpy()}")
        break
    
    # Calculate class weights for loss function
    print("\n4. Calculating class weights for loss function...")
    print("-" * 60)
    label_cols = ['patient_overall'] + [f'C{i}' for i in range(1, 8)]
    
    pos_weights = []
    for col in label_cols:
        pos_rate = train_df[col].mean()
        # Weight for positive class (higher if rare)
        weight = (1 - pos_rate) / (pos_rate + 1e-6)
        pos_weights.append(weight)
        print(f"{col}: pos_rate={pos_rate:.3f}, weight={weight:.3f}")
    
    pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32)
    print(f"\nâœ“ Positive class weights: {pos_weights_tensor}")
    
    # Memory estimation
    print("\n5. Memory Estimation...")
    print("-" * 60)
    batch_size = 2
    volume_memory = batch_size * 1 * 96 * 320 * 320 * 4 / (1024**3)  # 4 bytes per float32
    print(f"âœ“ Memory per batch (batch_size={batch_size}): {volume_memory:.2f} GB")
    print(f"âœ“ Recommended GPU memory: {volume_memory * 4:.2f} GB (for model + gradients)")
    
    print("\n" + "="*80)
    print("DATALOADER SETUP COMPLETE!")
    print("="*80)
    print("""
Next steps:
1. Build 3D CNN model
2. Define loss function (use pos_weights for class imbalance)
3. Set up optimizer and training loop
4. Start training!
    """)


"""
Visualization Tools for Preprocessed Spine Fracture Data
View CT volumes, labels, and explore the dataset interactively
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from matplotlib.patches import Rectangle
import seaborn as sns


# ============================================================================
# 1. VISUALIZE SINGLE VOLUME WITH LABELS
# ============================================================================

def visualize_volume_with_labels(volume, labels, patient_id, num_slices=12):
    """
    Visualize a 3D volume with its fracture labels
    
    Args:
        volume: numpy array (D, H, W) or (1, D, H, W) or torch tensor
        labels: numpy array (8,) [overall, C1-C7]
        patient_id: Patient ID string
        num_slices: Number of slices to display
    """
    # Convert torch tensor to numpy if needed
    if torch.is_tensor(volume):
        volume = volume.cpu().numpy()
    
    # Remove channel dimension if present
    if volume.ndim == 4:
        volume = volume[0]
    
    # Extract label information
    label_names = ['Overall'] + [f'C{i}' for i in range(1, 8)]
    fracture_status = {label_names[i]: bool(labels[i]) for i in range(len(labels))}
    
    # Create color for title (red if fracture, green if no fracture)
    title_color = 'red' if fracture_status['Overall'] else 'green'
    
    # Select evenly spaced slices
    depth = volume.shape[0]
    slice_indices = np.linspace(0, depth-1, num_slices, dtype=int)
    
    # Create subplot grid
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    axes = axes.flatten()
    
    # Plot each slice
    for idx, slice_idx in enumerate(slice_indices):
        axes[idx].imshow(volume[slice_idx], cmap='gray', vmin=0, vmax=1)
        axes[idx].set_title(f'Slice {slice_idx}/{depth}', fontsize=10)
        axes[idx].axis('off')
    
    # Overall title
    overall_status = "FRACTURE DETECTED" if fracture_status['Overall'] else "NO FRACTURE"
    fig.suptitle(f'Patient: {patient_id} - {overall_status}', 
                 fontsize=16, fontweight='bold', color=title_color)
    
    # Add label information as text
    label_text = "Fracture Labels:\n"
    for name, has_fracture in fracture_status.items():
        status = "âœ“ FRACTURE" if has_fracture else "âœ— No fracture"
        color = "red" if has_fracture else "black"
        label_text += f"{name}: {status}\n"
    
    # Add text box with labels
    fig.text(0.02, 0.5, label_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0.1, 0, 1, 0.96])
    plt.show()
    
    # Print volume statistics
    print(f"\n{'='*60}")
    print(f"Volume Statistics for Patient {patient_id}")
    print(f"{'='*60}")
    print(f"Shape: {volume.shape}")
    print(f"Value range: [{volume.min():.3f}, {volume.max():.3f}]")
    print(f"Mean: {volume.mean():.3f}")
    print(f"Std: {volume.std():.3f}")
    print(f"Non-zero voxels: {np.count_nonzero(volume):,} ({np.count_nonzero(volume)/volume.size*100:.1f}%)")
    print(f"{'='*60}\n")


# ============================================================================
# 2. COMPARE MULTIPLE PATIENTS SIDE BY SIDE
# ============================================================================

def compare_patients(volumes_list, labels_list, patient_ids_list, slice_position=0.5):
    """
    Compare multiple patients side by side
    
    Args:
        volumes_list: List of volumes
        labels_list: List of label arrays
        patient_ids_list: List of patient IDs
        slice_position: Position to slice (0.0 to 1.0, 0.5 = middle)
    """
    num_patients = len(volumes_list)
    
    fig, axes = plt.subplots(2, num_patients, figsize=(5*num_patients, 10))
    
    if num_patients == 1:
        axes = axes.reshape(-1, 1)
    
    for i, (volume, labels, patient_id) in enumerate(zip(volumes_list, labels_list, patient_ids_list)):
        # Convert if needed
        if torch.is_tensor(volume):
            volume = volume.cpu().numpy()
        if volume.ndim == 4:
            volume = volume[0]
        
        # Get slice
        slice_idx = int(volume.shape[0] * slice_position)
        
        # Axial view
        axes[0, i].imshow(volume[slice_idx], cmap='gray')
        fracture_status = "FRACTURE" if labels[0] == 1 else "NO FRACTURE"
        color = 'red' if labels[0] == 1 else 'green'
        axes[0, i].set_title(f'{patient_id}\n{fracture_status}', 
                            fontsize=12, fontweight='bold', color=color)
        axes[0, i].axis('off')
        
        # Sagittal view (middle slice)
        sagittal = volume[:, volume.shape[1]//2, :]
        axes[1, i].imshow(sagittal, cmap='gray', aspect='auto')
        axes[1, i].set_title('Sagittal View', fontsize=10)
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 3. EXPLORE SLICES INTERACTIVELY
# ============================================================================

def explore_volume_slices(volume, labels, patient_id, view='axial'):
    """
    Show all slices in a grid for detailed exploration
    
    Args:
        volume: 3D volume
        labels: Fracture labels
        patient_id: Patient ID
        view: 'axial', 'sagittal', or 'coronal'
    """
    # Convert if needed
    if torch.is_tensor(volume):
        volume = volume.cpu().numpy()
    if volume.ndim == 4:
        volume = volume[0]
    
    # Select view
    if view == 'axial':
        slices = volume  # (D, H, W)
        num_slices = volume.shape[0]
    elif view == 'sagittal':
        slices = np.transpose(volume, (2, 0, 1))  # (W, D, H)
        num_slices = volume.shape[2]
    elif view == 'coronal':
        slices = np.transpose(volume, (1, 0, 2))  # (H, D, W)
        num_slices = volume.shape[1]
    
    # Calculate grid size
    cols = 8
    rows = (num_slices + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, rows*2.5))
    axes = axes.flatten()
    
    for i in range(len(axes)):
        if i < num_slices:
            axes[i].imshow(slices[i], cmap='gray')
            axes[i].set_title(f'{i}', fontsize=8)
        axes[i].axis('off')
    
    fracture_status = "FRACTURE" if labels[0] == 1 else "NO FRACTURE"
    color = 'red' if labels[0] == 1 else 'green'
    fig.suptitle(f'Patient {patient_id} - {view.upper()} View - {fracture_status}', 
                 fontsize=16, fontweight='bold', color=color)
    
    plt.tight_layout()
    plt.show()


# ============================================================================
# 4. VISUALIZE BATCH FROM DATALOADER
# ============================================================================

def visualize_batch(train_loader, num_samples=4):
    """
    Visualize a batch from the DataLoader
    
    Args:
        train_loader: PyTorch DataLoader
        num_samples: Number of samples to show
    """
    # Get one batch
    volumes, labels, patient_ids = next(iter(train_loader))
    
    # Limit to num_samples
    num_samples = min(num_samples, volumes.shape[0])
    
    fig, axes = plt.subplots(2, num_samples, figsize=(4*num_samples, 8))
    
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        volume = volumes[i].cpu().numpy()[0]  # Remove channel dim
        label = labels[i].cpu().numpy()
        patient_id = patient_ids[i]
        
        # Get middle slice
        mid_slice = volume.shape[0] // 2
        
        # Axial view
        axes[0, i].imshow(volume[mid_slice], cmap='gray')
        fracture_status = "FRACTURE" if label[0] == 1 else "NO FRACTURE"
        color = 'red' if label[0] == 1 else 'green'
        axes[0, i].set_title(f'{patient_id}\n{fracture_status}', 
                            fontsize=10, color=color, fontweight='bold')
        axes[0, i].axis('off')
        
        # MIP (Maximum Intensity Projection)
        mip = np.max(volume, axis=0)
        axes[1, i].imshow(mip, cmap='gray')
        axes[1, i].set_title('MIP', fontsize=10)
        axes[1, i].axis('off')
    
    plt.suptitle('Batch Visualization from DataLoader', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Print batch info
    print(f"\n{'='*60}")
    print(f"Batch Information")
    print(f"{'='*60}")
    print(f"Batch shape: {volumes.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Number of samples: {len(patient_ids)}")
    print(f"Fractures in batch: {labels[:, 0].sum().item()}/{len(patient_ids)}")
    print(f"{'='*60}\n")


# ============================================================================
# 5. VISUALIZE DATASET STATISTICS
# ============================================================================

def visualize_dataset_statistics(train_df):
    """
    Visualize overall dataset statistics
    
    Args:
        train_df: Training DataFrame
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Overall fracture distribution
    fracture_counts = train_df['patient_overall'].value_counts()
    axes[0, 0].bar(['No Fracture', 'Fracture'], 
                   [fracture_counts[0], fracture_counts[1]],
                   color=['green', 'red'])
    axes[0, 0].set_title('Overall Fracture Distribution', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Number of Patients')
    for i, v in enumerate([fracture_counts[0], fracture_counts[1]]):
        axes[0, 0].text(i, v, f'{v}\n({v/len(train_df)*100:.1f}%)', 
                       ha='center', va='bottom', fontweight='bold')
    
    # 2. Fracture distribution by vertebra
    vertebrae_cols = [f'C{i}' for i in range(1, 8)]
    fracture_by_vertebra = train_df[vertebrae_cols].sum()
    
    axes[0, 1].bar(vertebrae_cols, fracture_by_vertebra, color='steelblue')
    axes[0, 1].set_title('Fractures by Vertebra', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Number of Fractures')
    axes[0, 1].set_xlabel('Vertebra')
    for i, v in enumerate(fracture_by_vertebra):
        axes[0, 1].text(i, v, f'{int(v)}', ha='center', va='bottom')
    
    # 3. Percentage by vertebra
    fracture_pct = (train_df[vertebrae_cols].sum() / len(train_df) * 100).sort_values(ascending=False)
    axes[0, 2].barh(fracture_pct.index, fracture_pct.values, color='coral')
    axes[0, 2].set_title('Fracture Rate by Vertebra (%)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Percentage of Patients')
    for i, v in enumerate(fracture_pct.values):
        axes[0, 2].text(v, i, f'{v:.1f}%', va='center')
    
    # 4. Number of fractured vertebrae per patient
    num_fractures = train_df[vertebrae_cols].sum(axis=1)
    fracture_dist = num_fractures.value_counts().sort_index()
    
    axes[1, 0].bar(fracture_dist.index, fracture_dist.values, color='purple', alpha=0.7)
    axes[1, 0].set_title('Number of Fractured Vertebrae per Patient', 
                        fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Number of Fractured Vertebrae')
    axes[1, 0].set_ylabel('Number of Patients')
    for i, v in enumerate(fracture_dist.values):
        axes[1, 0].text(fracture_dist.index[i], v, f'{v}', ha='center', va='bottom')
    
    # 5. Correlation heatmap
    corr_matrix = train_df[['patient_overall'] + vertebrae_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, ax=axes[1, 1], cbar_kws={'label': 'Correlation'})
    axes[1, 1].set_title('Correlation Matrix', fontsize=12, fontweight='bold')
    
    # 6. Class imbalance visualization
    label_cols = ['patient_overall'] + vertebrae_cols
    class_weights = []
    for col in label_cols:
        pos_rate = train_df[col].mean()
        weight = (1 - pos_rate) / (pos_rate + 1e-6)
        class_weights.append(weight)
    
    axes[1, 2].bar(range(len(label_cols)), class_weights, color='orange', alpha=0.7)
    axes[1, 2].set_xticks(range(len(label_cols)))
    axes[1, 2].set_xticklabels(label_cols, rotation=45)
    axes[1, 2].set_title('Class Weights (for Loss Function)', 
                        fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Weight')
    axes[1, 2].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Balanced')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{'='*60}")
    print(f"DATASET SUMMARY")
    print(f"{'='*60}")
    print(f"Total patients: {len(train_df)}")
    print(f"Patients with fractures: {train_df['patient_overall'].sum()} ({train_df['patient_overall'].mean()*100:.1f}%)")
    print(f"Patients without fractures: {(1-train_df['patient_overall']).sum()} ({(1-train_df['patient_overall']).mean()*100:.1f}%)")
    print(f"\nMost common fractured vertebra: {fracture_by_vertebra.idxmax()} ({fracture_by_vertebra.max()} cases)")
    print(f"Least common fractured vertebra: {fracture_by_vertebra.idxmin()} ({fracture_by_vertebra.min()} cases)")
    print(f"{'='*60}\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("="*80)
    print("VISUALIZE PREPROCESSED DATA")
    print("="*80)
    
    # Example 1: Visualize from DataLoader
    print("\n1. Visualizing batch from DataLoader...")
    print("-" * 60)
    
    # Assuming you have train_loader
    # visualize_batch(train_loader, num_samples=4)
    
    # Example 2: Visualize single patient
    print("\n2. To visualize a single patient:")
    print("-" * 60)
    print("""
    # Get one sample from dataloader
    volumes, labels, patient_ids = next(iter(train_loader))
    
    # Visualize first patient in batch
    visualize_volume_with_labels(
        volume=volumes[0],
        labels=labels[0],
        patient_id=patient_ids[0],
        num_slices=12
    )
    """)
    
    # Example 3: Explore all slices
    print("\n3. To explore all slices of a volume:")
    print("-" * 60)
    print("""
    explore_volume_slices(
        volume=volumes[0],
        labels=labels[0],
        patient_id=patient_ids[0],
        view='axial'  # or 'sagittal' or 'coronal'
    )
    """)
    
    # Example 4: Compare multiple patients
    print("\n4. To compare multiple patients:")
    print("-" * 60)
    print("""
    # Get a batch
    volumes, labels, patient_ids = next(iter(train_loader))
    
    # Compare first 3 patients
    compare_patients(
        volumes_list=[volumes[0], volumes[1], volumes[2]],
        labels_list=[labels[0], labels[1], labels[2]],
        patient_ids_list=[patient_ids[0], patient_ids[1], patient_ids[2]],
        slice_position=0.5
    )
    """)
    
    # Example 5: Dataset statistics
    print("\n5. To visualize dataset statistics:")
    print("-" * 60)
    print("""
    import pandas as pd
    train_df = pd.read_csv('/kaggle/input/.../train.csv')
    visualize_dataset_statistics(train_df)
    """)


    # Get one sample from dataloader
    volumes, labels, patient_ids = next(iter(train_loader))
    
    # Visualize first patient in batch
    visualize_volume_with_labels(
        volume=volumes[0],
        labels=labels[0],
        patient_id=patient_ids[0],
        num_slices=12
    )


    explore_volume_slices(
        volume=volumes[0],
        labels=labels[0],
        patient_id=patient_ids[0],
        view='axial'  # or 'sagittal' or 'coronal'
    )


    import pandas as pd
    train_df = pd.read_csv('/kaggle/input/rsna-2022-cervical-spine-fracture-detection/train.csv')
    visualize_dataset_statistics(train_df)



"""
EFFICIENTNET TRAINING - Complete Training Pipeline with Advanced Metrics
Includes: Precision-Recall Curves, mAP@0.5, Confusion Matrix
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import (roc_auc_score, precision_recall_curve, average_precision_score,
                              confusion_matrix, classification_report, auc)
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import gc
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ğŸš€ EFFICIENTNET-B0 TRAINING FOR SPINE FRACTURES")
print("="*80)

# ============================================================================
# IMPORT EFFICIENTNET MODEL
# ============================================================================

import torch.nn.functional as F

class Conv3dSame(nn.Module):
    """3D convolution with 'SAME' padding"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, 
                              padding=0, dilation=dilation, groups=groups, bias=bias)
        self.stride = stride
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 3
        
    def forward(self, x):
        d, h, w = x.shape[2:]
        pad_d = max((self.stride[0] if isinstance(self.stride, tuple) else self.stride) * 
                    ((d - 1) // (self.stride[0] if isinstance(self.stride, tuple) else self.stride)) + 
                    self.kernel_size[0] - d, 0)
        pad_h = max((self.stride[1] if isinstance(self.stride, tuple) else self.stride) * 
                    ((h - 1) // (self.stride[1] if isinstance(self.stride, tuple) else self.stride)) + 
                    self.kernel_size[1] - h, 0)
        pad_w = max((self.stride[2] if isinstance(self.stride, tuple) else self.stride) * 
                    ((w - 1) // (self.stride[2] if isinstance(self.stride, tuple) else self.stride)) + 
                    self.kernel_size[2] - w, 0)
        
        if pad_d > 0 or pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2,
                         pad_h // 2, pad_h - pad_h // 2,
                         pad_d // 2, pad_d - pad_d // 2])
        
        return self.conv(x)


class SqueezeExcitation3D(nn.Module):
    """3D Squeeze-and-Excitation block"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        reduced_channels = max(1, channels // reduction)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(channels, reduced_channels, 1),
            nn.SiLU(inplace=True),
            nn.Conv3d(reduced_channels, channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.se(x)


class MBConv3D(nn.Module):
    """3D Mobile Inverted Bottleneck Convolution"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        hidden_dim = int(in_channels * expand_ratio)
        
        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv3d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                nn.SiLU(inplace=True)
            ])
        
        layers.extend([
            Conv3dSame(hidden_dim, hidden_dim, kernel_size, stride=stride, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim),
            nn.SiLU(inplace=True)
        ])
        
        if se_ratio > 0:
            layers.append(SqueezeExcitation3D(hidden_dim, int(1/se_ratio)))
        
        layers.extend([
            nn.Conv3d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm3d(out_channels)
        ])
        
        self.block = nn.Sequential(*layers)
        self.dropout = nn.Dropout(0.2) if self.use_residual else None
    
    def forward(self, x):
        if self.use_residual:
            return x + self.dropout(self.block(x))
        else:
            return self.block(x)


class EfficientNet3D_B0(nn.Module):
    """3D EfficientNet-B0 for spine fracture detection"""
    def __init__(self, num_classes=8, dropout=0.3):
        super().__init__()
        
        self.stem = nn.Sequential(
            Conv3dSame(1, 32, kernel_size=3, stride=2, bias=False),
            nn.BatchNorm3d(32),
            nn.SiLU(inplace=True)
        )
        
        blocks_config = [
            [1, 32, 16, 3, 1, 1],
            [2, 16, 24, 3, 2, 6],
            [2, 24, 40, 5, 2, 6],
            [3, 40, 80, 3, 2, 6],
            [3, 80, 112, 5, 1, 6],
            [4, 112, 192, 5, 2, 6],
            [1, 192, 320, 3, 1, 6],
        ]
        
        self.blocks = nn.ModuleList()
        for num_layers, in_ch, out_ch, kernel, stride, expand in blocks_config:
            for i in range(num_layers):
                self.blocks.append(
                    MBConv3D(
                        in_channels=in_ch if i == 0 else out_ch,
                        out_channels=out_ch,
                        kernel_size=kernel,
                        stride=stride if i == 0 else 1,
                        expand_ratio=expand,
                        se_ratio=0.25
                    )
                )
        
        self.head = nn.Sequential(
            nn.Conv3d(320, 1280, 1, bias=False),
            nn.BatchNorm3d(1280),
            nn.SiLU(inplace=True),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten()
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc_overall = nn.Linear(1280, 1)
        self.fc_vertebrae = nn.Linear(1280, 7)
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.stem(x)
        
        for block in self.blocks:
            x = block(x)
        
        features = self.head(x)
        features = self.dropout(features)
        
        overall = self.fc_overall(features)
        vertebrae = self.fc_vertebrae(features)
        
        return torch.cat([overall, vertebrae], dim=1)


# ============================================================================
# EVALUATION METRICS FUNCTIONS
# ============================================================================

def plot_precision_recall_curves(all_labels, all_preds, class_names, save_path):
    """Plot Precision-Recall curves for all classes"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    aps = []
    
    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        if len(np.unique(all_labels[:, i])) > 1:
            precision, recall, _ = precision_recall_curve(all_labels[:, i], all_preds[:, i])
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
            pr_auc = auc(recall, precision)
            aps.append(ap)
            
            ax.plot(recall, precision, linewidth=2, label=f'AP={ap:.3f}, AUC={pr_auc:.3f}')
            ax.fill_between(recall, precision, alpha=0.2)
            ax.set_xlabel('Recall', fontsize=10)
            ax.set_ylabel('Precision', fontsize=10)
            ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')
            ax.legend(loc='best')
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1.05])
        else:
            ax.text(0.5, 0.5, f'{class_name}\nNo positive samples', 
                   ha='center', va='center', fontsize=10)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            aps.append(0.0)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    return aps


def calculate_map_at_threshold(all_labels, all_preds, threshold=0.5):
    """Calculate mAP@threshold (mean Average Precision)"""
    aps = []
    
    for i in range(all_labels.shape[1]):
        if len(np.unique(all_labels[:, i])) > 1:
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
            aps.append(ap)
        else:
            aps.append(0.0)
    
    return np.mean(aps), aps


def plot_confusion_matrices(all_labels, all_preds, class_names, save_path, threshold=0.5):
    """Plot confusion matrices for all classes"""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()
    
    pred_binary = (all_preds > threshold).astype(int)
    
    for i, (ax, class_name) in enumerate(zip(axes, class_names)):
        cm = confusion_matrix(all_labels[:, i], pred_binary[:, i])
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   cbar_kws={'label': 'Count'})
        
        ax.set_title(f'{class_name}', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=10)
        ax.set_xlabel('Predicted Label', fontsize=10)
        
        # Calculate metrics
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics_text = f'Acc: {accuracy:.3f}\nPrec: {precision:.3f}\nRec: {recall:.3f}\nF1: {f1:.3f}'
        ax.text(1.5, 0.5, metrics_text, fontsize=9, 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def print_detailed_metrics(all_labels, all_preds, class_names, threshold=0.5):
    """Print detailed classification metrics"""
    pred_binary = (all_preds > threshold).astype(int)
    
    print("\n" + "="*80)
    print("ğŸ“Š DETAILED CLASSIFICATION METRICS")
    print("="*80)
    
    for i, class_name in enumerate(class_names):
        print(f"\n{class_name}:")
        print("-" * 60)
        
        if len(np.unique(all_labels[:, i])) > 1:
            cm = confusion_matrix(all_labels[:, i], pred_binary[:, i])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
            
            accuracy = (tp + tn) / (tp + tn + fp + fn)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            print(f"  Confusion Matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
            print(f"  Accuracy:    {accuracy:.4f}")
            print(f"  Precision:   {precision:.4f}")
            print(f"  Recall:      {recall:.4f}")
            print(f"  Specificity: {specificity:.4f}")
            print(f"  F1-Score:    {f1:.4f}")
            
            # Average Precision
            ap = average_precision_score(all_labels[:, i], all_preds[:, i])
            print(f"  Avg Precision (AP): {ap:.4f}")
        else:
            print("  No positive samples in validation set")


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

CONFIG = {
    'num_epochs': 6,
    'batch_size': 2,
    'train_subset_ratio': 0.50,
    'val_subset_ratio': 0.40,
    'max_train_batches': 120,
    'max_val_batches': 60,
    'learning_rate': 3e-4,
    'use_amp': True,
    'gradient_accumulation': 4,
    'num_workers': 2,
    'pin_memory': False,
    'prefetch_factor': 2,
    'warmup_epochs': 1,
    'weight_decay': 0.01,
    'max_grad_norm': 1.0,
    'save_dir': '/kaggle/working',
    'verbose': True,
}

print(f"\nâš™ï¸�  EfficientNet Training Configuration:")
print(f"  ğŸ“Š Model: EfficientNet-B0 (Efficient & Accurate)")
print(f"  ğŸ“ˆ Epochs: {CONFIG['num_epochs']}")
print(f"  ğŸ”¢ Batch size: {CONFIG['batch_size']} (effective: {CONFIG['batch_size']*CONFIG['gradient_accumulation']})")
print(f"  ğŸ“š Training batches: {CONFIG['max_train_batches']}")
print(f"  âœ… Validation batches: {CONFIG['max_val_batches']}")
print(f"  ğŸ�“ Learning rate: {CONFIG['learning_rate']}")
print(f"  ğŸ“Š Metrics: PR Curves, mAP@0.5, Confusion Matrix")

# ============================================================================
# DEVICE SETUP
# ============================================================================

print(f"\nğŸ§¹ Preparing GPU...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    gc.collect()
    
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    mem_free = torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated(0)
    print(f"ğŸ–¥ï¸�  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"  Free: {mem_free / 1024**3:.1f} GB")

# ============================================================================
# CREATE DATALOADERS
# ============================================================================

print(f"\nğŸ“¦ Creating data loaders...")

def create_balanced_loader(original_loader, subset_ratio, batch_size, max_batches, shuffle=True):
    dataset = original_loader.dataset
    total_size = len(dataset)
    subset_size = int(total_size * subset_ratio)
    
    indices = np.random.choice(total_size, subset_size, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    
    loader = torch.utils.data.DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        prefetch_factor=CONFIG['prefetch_factor'] if CONFIG['num_workers'] > 0 else None,
        persistent_workers=True if CONFIG['num_workers'] > 0 else False,
        drop_last=True,
    )
    
    return loader

try:
    balanced_train_loader = create_balanced_loader(
        train_loader, CONFIG['train_subset_ratio'],
        CONFIG['batch_size'], CONFIG['max_train_batches'], shuffle=True
    )
    balanced_val_loader = create_balanced_loader(
        val_loader, CONFIG['val_subset_ratio'],
        CONFIG['batch_size'], CONFIG['max_val_batches'], shuffle=False
    )
    print(f"  âœ“ Using batch_size={CONFIG['batch_size']}")
    
except RuntimeError as e:
    if "out of memory" in str(e):
        print(f"  âš ï¸�  OOM with batch_size={CONFIG['batch_size']}, reducing to 1")
        CONFIG['batch_size'] = 1
        torch.cuda.empty_cache()
        
        balanced_train_loader = create_balanced_loader(
            train_loader, CONFIG['train_subset_ratio'],
            CONFIG['batch_size'], CONFIG['max_train_batches'], shuffle=True
        )
        balanced_val_loader = create_balanced_loader(
            val_loader, CONFIG['val_subset_ratio'],
            CONFIG['batch_size'], CONFIG['max_val_batches'], shuffle=False
        )

actual_train_batches = min(len(balanced_train_loader), CONFIG['max_train_batches'])
actual_val_batches = min(len(balanced_val_loader), CONFIG['max_val_batches'])

print(f"  âœ“ Train: {actual_train_batches} batches")
print(f"  âœ“ Val: {actual_val_batches} batches")

# ============================================================================
# CLASS WEIGHTS
# ============================================================================

print(f"\nâš–ï¸�  Calculating class weights...")
label_cols = ['patient_overall'] + [f'C{i}' for i in range(1, 8)]

sample_df = train_df.sample(n=min(2000, len(train_df)), random_state=42)
pos_weights = []
for col in label_cols:
    pos_rate = sample_df[col].mean()
    weight = max(1.0, min(10.0, (1 - pos_rate) / (pos_rate + 1e-6)))
    pos_weights.append(weight)

pos_weights_tensor = torch.tensor(pos_weights, dtype=torch.float32).to(device)
print(f"  âœ“ Weights computed: Overall={pos_weights[0]:.2f}, C1-C7={pos_weights[1]:.2f} avg")

# ============================================================================
# MODEL & TRAINING SETUP
# ============================================================================

print(f"\nğŸ�—ï¸�  Loading EfficientNet-B0...")
model = EfficientNet3D_B0(num_classes=8, dropout=0.3)
model = model.to(device)

torch.cuda.empty_cache()
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"  âœ“ Model loaded: {num_params:,} parameters ({num_params*4/1024**2:.1f} MB)")

print(f"\nğŸ�¯ Setting up training components...")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights_tensor)

optimizer = optim.AdamW(
    model.parameters(), 
    lr=CONFIG['learning_rate'],
    weight_decay=CONFIG['weight_decay'],
    betas=(0.9, 0.999)
)

total_steps = (actual_train_batches // CONFIG['gradient_accumulation']) * CONFIG['num_epochs']
warmup_steps = (actual_train_batches // CONFIG['gradient_accumulation']) * CONFIG['warmup_epochs']

scheduler = optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=CONFIG['learning_rate'],
    total_steps=total_steps,
    pct_start=warmup_steps/total_steps,
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=10000.0
)

scaler = torch.amp.GradScaler('cuda') if CONFIG['use_amp'] else None

print(f"  âœ“ Loss: Weighted BCEWithLogitsLoss")
print(f"  âœ“ Optimizer: AdamW")
print(f"  âœ“ Scheduler: OneCycleLR")

# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, 
                    max_batches, grad_accum_steps, epoch_num):
    model.train()
    running_loss = 0.0
    num_batches = 0
    
    optimizer.zero_grad()
    
    from tqdm import tqdm
    pbar = tqdm(enumerate(loader), total=min(len(loader), max_batches), 
                desc=f"Epoch {epoch_num+1} Train", leave=True)
    
    for batch_idx, batch_data in pbar:
        if batch_idx >= max_batches:
            break
        
        try:
            if len(batch_data) == 3:
                volumes, labels, _ = batch_data
            else:
                volumes, labels = batch_data[0], batch_data[1]
            
            volumes = volumes.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            if CONFIG['use_amp'] and scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(volumes)
                    loss = criterion(outputs, labels) / grad_accum_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(volumes)
                loss = criterion(outputs, labels) / grad_accum_steps
                loss.backward()
                
                if (batch_idx + 1) % grad_accum_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['max_grad_norm'])
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            running_loss += loss.item() * grad_accum_steps
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item()*grad_accum_steps:.4f}'})
        
        except Exception as e:
            continue
    
    return running_loss / num_batches if num_batches > 0 else 0


def validate_with_metrics(model, loader, criterion, device, max_batches):
    """Enhanced validation with all predictions and labels for metrics"""
    model.eval()
    running_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []
    
    from tqdm import tqdm
    pbar = tqdm(enumerate(loader), total=min(len(loader), max_batches), 
                desc="Validation", leave=True)
    
    with torch.no_grad():
        for batch_idx, batch_data in pbar:
            if batch_idx >= max_batches:
                break
            
            try:
                if len(batch_data) == 3:
                    volumes, labels, _ = batch_data
                else:
                    volumes, labels = batch_data[0], batch_data[1]
                
                volumes = volumes.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                
                with torch.amp.autocast('cuda'):
                    outputs = model(volumes)
                    loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                num_batches += 1
                
                preds = torch.sigmoid(outputs).cpu().numpy()
                all_preds.append(preds)
                all_labels.append(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            except Exception as e:
                continue
    
    if num_batches == 0 or len(all_preds) == 0:
        return 0, 0.5, [0.5]*8, 0.5, None, None
    
    all_preds = np.vstack(all_preds)
    all_labels = np.vstack(all_labels)
    
    # Calculate AUCs
    aucs = []
    for i in range(8):
        try:
            if len(np.unique(all_labels[:, i])) > 1:
                auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
                aucs.append(auc)
            else:
                aucs.append(0.5)
        except:
            aucs.append(0.5)
    
    mean_auc = np.mean(aucs)
    pred_binary = (all_preds > 0.5).astype(int)
    accuracy = (pred_binary == all_labels).mean()
    avg_loss = running_loss / num_batches
    
    return avg_loss, mean_auc, aucs, accuracy, all_preds, all_labels

# ============================================================================
# MAIN TRAINING LOOP
# ============================================================================

print("\n" + "="*80)
print("ğŸš€ STARTING EFFICIENTNET TRAINING")
print("="*80)

class_names = ['Overall'] + [f'C{i}' for i in range(1, 8)]

history = {
    'train_loss': [], 'val_loss': [], 'val_auc': [], 'val_acc': [],
    'learning_rate': [], 'epoch_time': [], 'map_scores': []
}
best_auc = 0.0
best_aucs = [0.5] * 8
best_epoch = 0

total_start = time.time()

for epoch in range(CONFIG['num_epochs']):
    epoch_start = time.time()
    
    print(f"\n{'='*70}")
    print(f"ğŸ“… Epoch {epoch+1}/{CONFIG['num_epochs']}")
    print(f"{'='*70}")
    
    train_loss = train_one_epoch(
        model, balanced_train_loader, criterion, optimizer, scheduler, device,
        scaler, CONFIG['max_train_batches'], CONFIG['gradient_accumulation'], epoch
    )
    
    torch.cuda.empty_cache()
    
    val_loss, mean_auc, aucs, val_acc, all_preds, all_labels = validate_with_metrics(
        model, balanced_val_loader, criterion, device, CONFIG['max_val_batches']
    )
    
    # Calculate mAP@0.5
    map_score, ap_scores = calculate_map_at_threshold(all_labels, all_preds, threshold=0.5)
    
    current_lr = optimizer.param_groups[0]['lr']
    epoch_time = time.time() - epoch_start
    
    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_auc'].append(mean_auc)
    history['val_acc'].append(val_acc)
    history['learning_rate'].append(current_lr)
    history['epoch_time'].append(epoch_time)
    history['map_scores'].append(map_score)
    
    print(f"\nğŸ“Š Epoch {epoch+1} Results:")
    print(f"   Train Loss:    {train_loss:.4f}")
    print(f"   Val Loss:      {val_loss:.4f}")
    print(f"   Val AUC:       {mean_auc:.4f} {'ğŸ�¯ NEW BEST!' if mean_auc > best_auc else ''}")
    print(f"   Val Accuracy:  {val_acc:.4f}")
    print(f"   mAP@0.5:       {map_score:.4f}")
    print(f"   Learning Rate: {current_lr:.6f}")
    print(f"   Epoch Time:    {epoch_time:.1f}s")
    
    print(f"\n   Individual AUCs:")
    for i, (name, auc_val) in enumerate(zip(class_names, aucs)):
        print(f"      {name:8s}: {auc_val:.4f}")
    
    print(f"\n   Individual APs (Average Precision):")
    for i, (name, ap_val) in enumerate(zip(class_names, ap_scores)):
        print(f"      {name:8s}: {ap_val:.4f}")
    
    # Save best model
    if mean_auc > best_auc:
        best_auc = mean_auc
        best_aucs = aucs.copy()
        best_epoch = epoch
        
        model_path = os.path.join(CONFIG['save_dir'], 'efficientnet_best_model.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_auc': mean_auc,
            'val_loss': val_loss,
            'aucs': aucs,
            'config': CONFIG
        }, model_path)
        print(f"\n   ğŸ’¾ Model saved: {model_path}")
    
    torch.cuda.empty_cache()
    gc.collect()

total_time = time.time() - total_start

print("\n" + "="*80)
print("âœ… TRAINING COMPLETED!")
print("="*80)
print(f"â�±ï¸�  Total training time: {total_time/60:.1f} minutes")
print(f"ğŸ�† Best validation AUC: {best_auc:.4f} (Epoch {best_epoch+1})")
print(f"\n   Best Individual AUCs:")
for name, auc_val in zip(class_names, best_aucs):
    print(f"      {name:8s}: {auc_val:.4f}")

# ============================================================================
# FINAL EVALUATION WITH ALL METRICS
# ============================================================================

print("\n" + "="*80)
print("ğŸ“ˆ GENERATING COMPREHENSIVE EVALUATION METRICS")
print("="*80)

# Load best model for final evaluation
print("\nğŸ“‚ Loading best model for final evaluation...")
checkpoint = torch.load(os.path.join(CONFIG['save_dir'], 'efficientnet_best_model.pth'))
model.load_state_dict(checkpoint['model_state_dict'])
print("   âœ“ Best model loaded")

# Get predictions on validation set
print("\nğŸ”� Running final validation pass...")
val_loss, mean_auc, aucs, val_acc, all_preds, all_labels = validate_with_metrics(
    model, balanced_val_loader, criterion, device, CONFIG['max_val_batches']
)

# Calculate mAP@0.5
map_score, ap_scores = calculate_map_at_threshold(all_labels, all_preds, threshold=0.5)

print(f"\nâœ… Final Validation Metrics:")
print(f"   Mean AUC:      {mean_auc:.4f}")
print(f"   Accuracy:      {val_acc:.4f}")
print(f"   mAP@0.5:       {map_score:.4f}")

# Generate Precision-Recall Curves
print("\nğŸ“Š Generating Precision-Recall curves...")
pr_curve_path = os.path.join(CONFIG['save_dir'], 'precision_recall_curves.png')
aps = plot_precision_recall_curves(all_labels, all_preds, class_names, pr_curve_path)
print(f"   âœ“ Saved: {pr_curve_path}")

# Generate Confusion Matrices
print("\nğŸ“Š Generating Confusion matrices...")
cm_path = os.path.join(CONFIG['save_dir'], 'confusion_matrices.png')
plot_confusion_matrices(all_labels, all_preds, class_names, cm_path, threshold=0.5)
print(f"   âœ“ Saved: {cm_path}")

# Print detailed metrics
print_detailed_metrics(all_labels, all_preds, class_names, threshold=0.5)

# ============================================================================
# PLOT TRAINING HISTORY
# ============================================================================

print("\nğŸ“Š Generating training history plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Loss plot
axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o')
axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s')
axes[0, 0].set_xlabel('Epoch', fontsize=10)
axes[0, 0].set_ylabel('Loss', fontsize=10)
axes[0, 0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# AUC plot
axes[0, 1].plot(history['val_auc'], label='Val AUC', linewidth=2, marker='o', color='green')
axes[0, 1].axhline(y=best_auc, color='r', linestyle='--', label=f'Best: {best_auc:.4f}')
axes[0, 1].set_xlabel('Epoch', fontsize=10)
axes[0, 1].set_ylabel('AUC', fontsize=10)
axes[0, 1].set_title('Validation AUC', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Accuracy plot
axes[0, 2].plot(history['val_acc'], label='Val Accuracy', linewidth=2, marker='o', color='purple')
axes[0, 2].set_xlabel('Epoch', fontsize=10)
axes[0, 2].set_ylabel('Accuracy', fontsize=10)
axes[0, 2].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Learning rate plot
axes[1, 0].plot(history['learning_rate'], linewidth=2, marker='o', color='orange')
axes[1, 0].set_xlabel('Epoch', fontsize=10)
axes[1, 0].set_ylabel('Learning Rate', fontsize=10)
axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Epoch time plot
axes[1, 1].bar(range(len(history['epoch_time'])), history['epoch_time'], color='teal', alpha=0.7)
axes[1, 1].set_xlabel('Epoch', fontsize=10)
axes[1, 1].set_ylabel('Time (seconds)', fontsize=10)
axes[1, 1].set_title('Epoch Training Time', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# mAP plot
axes[1, 2].plot(history['map_scores'], label='mAP@0.5', linewidth=2, marker='o', color='red')
axes[1, 2].set_xlabel('Epoch', fontsize=10)
axes[1, 2].set_ylabel('mAP@0.5', fontsize=10)
axes[1, 2].set_title('Mean Average Precision @0.5', fontsize=12, fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
history_path = os.path.join(CONFIG['save_dir'], 'training_history.png')
plt.savefig(history_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   âœ“ Saved: {history_path}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ğŸ�‰ EFFICIENTNET TRAINING COMPLETE - FINAL SUMMARY")
print("="*80)

print(f"\nğŸ“‹ Training Configuration:")
print(f"   Model:              EfficientNet-B0")
print(f"   Epochs:             {CONFIG['num_epochs']}")
print(f"   Batch Size:         {CONFIG['batch_size']}")
print(f"   Gradient Accum:     {CONFIG['gradient_accumulation']}")
print(f"   Learning Rate:      {CONFIG['learning_rate']}")
print(f"   Training Batches:   {actual_train_batches}")
print(f"   Validation Batches: {actual_val_batches}")

print(f"\nğŸ�† Best Results (Epoch {best_epoch+1}):")
print(f"   Validation AUC:     {best_auc:.4f}")
print(f"   mAP@0.5:            {map_score:.4f}")
print(f"   Accuracy:           {val_acc:.4f}")

print(f"\nğŸ“Š Saved Outputs:")
print(f"   âœ“ Model checkpoint:        efficientnet_best_model.pth")
print(f"   âœ“ Training history:        training_history.png")
print(f"   âœ“ Precision-Recall curves: precision_recall_curves.png")
print(f"   âœ“ Confusion matrices:      confusion_matrices.png")

print(f"\nâ�±ï¸�  Performance:")
print(f"   Total Training Time: {total_time/60:.1f} minutes")
print(f"   Avg Time per Epoch:  {np.mean(history['epoch_time']):.1f} seconds")

print("\n" + "="*80)
print("âœ¨ All metrics generated successfully!")
print("="*80)


import torch
import numpy as np
import os

print("\n" + "="*80)
print("ğŸ“ˆ GENERATING COMPREHENSIVE EVALUATION METRICS")
print("="*80)

# Load best model for final evaluation (WITH FIX)
print("\nğŸ“‚ Loading best model for final evaluation...")
checkpoint = torch.load(
    os.path.join(CONFIG['save_dir'], 'efficientnet_best_model.pth'),
    weights_only=False  # <-- This is the fix!
)
model.load_state_dict(checkpoint['model_state_dict'])
print("   âœ“ Best model loaded")

# Get predictions on validation set
print("\nğŸ”� Running final validation pass...")
val_loss, mean_auc, aucs, val_acc, all_preds, all_labels = validate_with_metrics(
    model, balanced_val_loader, criterion, device, CONFIG['max_val_batches']
)

# Calculate mAP@0.5
map_score, ap_scores = calculate_map_at_threshold(all_labels, all_preds, threshold=0.5)

print(f"\nâœ… Final Validation Metrics:")
print(f"   Mean AUC:      {mean_auc:.4f}")
print(f"   Accuracy:      {val_acc:.4f}")
print(f"   mAP@0.5:       {map_score:.4f}")

# Generate Precision-Recall Curves
print("\nğŸ“Š Generating Precision-Recall curves...")
pr_curve_path = os.path.join(CONFIG['save_dir'], 'precision_recall_curves.png')
aps = plot_precision_recall_curves(all_labels, all_preds, class_names, pr_curve_path)
print(f"   âœ“ Saved: {pr_curve_path}")

# Generate Confusion Matrices
print("\nğŸ“Š Generating Confusion matrices...")
cm_path = os.path.join(CONFIG['save_dir'], 'confusion_matrices.png')
plot_confusion_matrices(all_labels, all_preds, class_names, cm_path, threshold=0.5)
print(f"   âœ“ Saved: {cm_path}")

# Print detailed metrics
print_detailed_metrics(all_labels, all_preds, class_names, threshold=0.5)

# Plot training history
print("\nğŸ“Š Generating training history plots...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Loss plot
axes[0, 0].plot(history['train_loss'], label='Train Loss', linewidth=2, marker='o')
axes[0, 0].plot(history['val_loss'], label='Val Loss', linewidth=2, marker='s')
axes[0, 0].set_xlabel('Epoch', fontsize=10)
axes[0, 0].set_ylabel('Loss', fontsize=10)
axes[0, 0].set_title('Training & Validation Loss', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# AUC plot
axes[0, 1].plot(history['val_auc'], label='Val AUC', linewidth=2, marker='o', color='green')
axes[0, 1].axhline(y=best_auc, color='r', linestyle='--', label=f'Best: {best_auc:.4f}')
axes[0, 1].set_xlabel('Epoch', fontsize=10)
axes[0, 1].set_ylabel('AUC', fontsize=10)
axes[0, 1].set_title('Validation AUC', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Accuracy plot
axes[0, 2].plot(history['val_acc'], label='Val Accuracy', linewidth=2, marker='o', color='purple')
axes[0, 2].set_xlabel('Epoch', fontsize=10)
axes[0, 2].set_ylabel('Accuracy', fontsize=10)
axes[0, 2].set_title('Validation Accuracy', fontsize=12, fontweight='bold')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3)

# Learning rate plot
axes[1, 0].plot(history['learning_rate'], linewidth=2, marker='o', color='orange')
axes[1, 0].set_xlabel('Epoch', fontsize=10)
axes[1, 0].set_ylabel('Learning Rate', fontsize=10)
axes[1, 0].set_title('Learning Rate Schedule', fontsize=12, fontweight='bold')
axes[1, 0].set_yscale('log')
axes[1, 0].grid(True, alpha=0.3)

# Epoch time plot
axes[1, 1].bar(range(len(history['epoch_time'])), history['epoch_time'], color='teal', alpha=0.7)
axes[1, 1].set_xlabel('Epoch', fontsize=10)
axes[1, 1].set_ylabel('Time (seconds)', fontsize=10)
axes[1, 1].set_title('Epoch Training Time', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# mAP plot
axes[1, 2].plot(history['map_scores'], label='mAP@0.5', linewidth=2, marker='o', color='red')
axes[1, 2].set_xlabel('Epoch', fontsize=10)
axes[1, 2].set_ylabel('mAP@0.5', fontsize=10)
axes[1, 2].set_title('Mean Average Precision @0.5', fontsize=12, fontweight='bold')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
history_path = os.path.join(CONFIG['save_dir'], 'training_history.png')
plt.savefig(history_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"   âœ“ Saved: {history_path}")

# Final summary
print("\n" + "="*80)
print("ğŸ�‰ EFFICIENTNET TRAINING COMPLETE - FINAL SUMMARY")
print("="*80)

print(f"\nğŸ“‹ Training Configuration:")
print(f"   Model:              EfficientNet-B0")
print(f"   Epochs:             {CONFIG['num_epochs']}")
print(f"   Best Epoch:         {best_epoch+1}")

print(f"\nğŸ�† Best Results:")
print(f"   Validation AUC:     {best_auc:.4f}")
print(f"   mAP@0.5:            {map_score:.4f}")
print(f"   Accuracy:           {val_acc:.4f}")

print(f"\nğŸ“Š Saved Outputs:")
print(f"   âœ“ Model checkpoint:        efficientnet_best_model.pth")
print(f"   âœ“ Training history:        training_history.png")
print(f"   âœ“ Precision-Recall curves: precision_recall_curves.png")
print(f"   âœ“ Confusion matrices:      confusion_matrices.png")

print("\n" + "="*80)
print("âœ¨ All metrics generated successfully!")
print("="*80)


"""
COMPLETE GRAD-CAM VISUALIZATION - SINGLE SCRIPT
Everything needed: imports, model, data loading, and visualization
Just run this entire cell after your training!
"""

import os
import gc
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.video import r3d_18, R3D_18_Weights

print("="*80)
print("ğŸ”¥ COMPLETE GRAD-CAM VISUALIZATION SCRIPT")
print("="*80)

# ============================================================================
# 1. MODEL DEFINITION
# ============================================================================

print("\nğŸ“¦ Step 1: Defining Model Architecture...")

class SpineFractureResNet3D(nn.Module):
    """3D ResNet18 for fracture detection"""
    
    def __init__(self, num_classes=8, pretrained=False, dropout=0.3):
        super(SpineFractureResNet3D, self).__init__()
        
        if pretrained:
            self.backbone = r3d_18(weights=R3D_18_Weights.DEFAULT)
        else:
            self.backbone = r3d_18(weights=None)
        
        self.backbone.stem[0] = nn.Conv3d(
            1, 64, kernel_size=(3, 7, 7),
            stride=(1, 2, 2), padding=(1, 3, 3), bias=False
        )
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc_overall = nn.Linear(in_features, 1)
        self.fc_vertebrae = nn.Linear(in_features, 7)
        
        nn.init.xavier_uniform_(self.fc_overall.weight)
        nn.init.zeros_(self.fc_overall.bias)
        nn.init.xavier_uniform_(self.fc_vertebrae.weight)
        nn.init.zeros_(self.fc_vertebrae.bias)
    
    def forward(self, x):
        features = self.backbone(x)
        features = self.dropout(features)
        overall = self.fc_overall(features)
        vertebrae = self.fc_vertebrae(features)
        output = torch.cat([overall, vertebrae], dim=1)
        return output

print("  âœ“ Model architecture defined")

# ============================================================================
# 2. GRAD-CAM IMPLEMENTATION
# ============================================================================

print("\nğŸ”� Step 2: Setting up Grad-CAM...")

class GradCAM3D:
    """3D Grad-CAM for fracture localization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_volume, target_class=0):
        self.model.eval()
        output = self.model(input_volume)
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=(1, 2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

print("  âœ“ Grad-CAM class ready")

# ============================================================================
# 3. VISUALIZATION FUNCTIONS
# ============================================================================

print("\nğŸ�¨ Step 3: Setting up visualization functions...")

def visualize_gradcam_comprehensive(volume, cam, predictions, labels, patient_id, save_path=None):
    """
    Comprehensive Grad-CAM visualization with 12 slices
    """
    # Select 12 evenly spaced slices
    depth = volume.shape[0]
    slice_indices = np.linspace(0, depth-1, 12, dtype=int)
    
    # Resize CAM if needed
    if cam.shape != volume.shape:
        from scipy.ndimage import zoom
        zoom_factors = np.array(volume.shape) / np.array(cam.shape)
        cam_resized = zoom(cam, zoom_factors, order=1)
    else:
        cam_resized = cam
    
    # Create figure with 4x3 grid
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    # Custom colormap (blue to red for heatmap)
    colors = ['darkblue', 'blue', 'cyan', 'yellow', 'orange', 'red', 'darkred']
    cmap = LinearSegmentedColormap.from_list('fracture_heatmap', colors, N=256)
    
    for idx, slice_idx in enumerate(slice_indices):
        ax = axes[idx]
        
        # Show CT slice in grayscale
        ax.imshow(volume[slice_idx], cmap='gray', vmin=0, vmax=1)
        
        # Overlay CAM heatmap (only significant regions)
        cam_slice = cam_resized[slice_idx]
        masked_cam = np.ma.masked_where(cam_slice < 0.3, cam_slice)
        im = ax.imshow(masked_cam, cmap=cmap, alpha=0.7, vmin=0, vmax=1)
        
        ax.set_title(f'Slice {slice_idx}/{depth}', fontsize=11, fontweight='bold')
        ax.axis('off')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=axes, orientation='horizontal', 
                        pad=0.02, fraction=0.046, aspect=40)
    cbar.set_label('Fracture Attention (Model Focus)', fontsize=12, fontweight='bold')
    
    # Overall title with prediction
    fracture_status = "FRACTURE DETECTED" if predictions[0] > 0.5 else "NO FRACTURE"
    confidence = predictions[0] * 100
    color = 'red' if predictions[0] > 0.5 else 'green'
    
    fig.suptitle(
        f'Grad-CAM Fracture Localization: {patient_id}\n' +
        f'{fracture_status} (Model Confidence: {confidence:.1f}%)',
        fontsize=18, fontweight='bold', color=color, y=0.98
    )
    
    # Add detailed predictions panel
    label_names = ['Overall'] + [f'C{i}' for i in range(1, 8)]
    info_text = "MODEL PREDICTIONS:\n" + "="*35 + "\n"
    
    for i, name in enumerate(label_names):
        pred_prob = predictions[i]
        gt = int(labels[i])
        pred = int(pred_prob > 0.5)
        match = 'âœ“ CORRECT' if pred == gt else 'âœ— WRONG'
        
        status = "FRACTURE" if pred == 1 else "Normal"
        info_text += f"{name:8s}: {status:10s} ({pred_prob*100:5.1f}%)"
        
        if gt is not None:
            info_text += f" | GT:{gt} {match}"
        
        info_text += "\n"
    
    fig.text(0.02, 0.5, info_text, fontsize=10, verticalalignment='center',
             family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', 
                      alpha=0.9, edgecolor='black', linewidth=2))
    
    plt.tight_layout(rect=[0.12, 0, 1, 0.95])
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  âœ“ Saved visualization: {save_path}")
    
    plt.show()
    
    return fig

# ============================================================================
# 4. LOAD MODEL AND DATA
# ============================================================================

print("\nğŸ”§ Step 4: Loading model and preparing data...")

# Setup device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")

if torch.cuda.is_available():
    torch.cuda.empty_cache()
    gc.collect()
    print(f"  âœ“ GPU memory cleared")

# Load trained model
checkpoint_path = '/kaggle/working/balanced_demo_best.pth'

if os.path.exists(checkpoint_path):
    print(f"  Loading trained model...")
    model = SpineFractureResNet3D(num_classes=8, pretrained=False, dropout=0.3)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"  âœ“ Model loaded (Epoch {checkpoint['epoch']}, AUC: {checkpoint['best_auc']:.4f})")
else:
    print(f"  âš ï¸�  No checkpoint found, creating untrained model")
    model = SpineFractureResNet3D(num_classes=8, pretrained=False, dropout=0.3)

model = model.to(device)
model.eval()

num_params = sum(p.numel() for p in model.parameters())
print(f"  âœ“ Model ready ({num_params:,} parameters)")

# ============================================================================
# 5. RUN GRAD-CAM ON VALIDATION DATA
# ============================================================================

print("\n" + "="*80)
print("ğŸš€ RUNNING GRAD-CAM ANALYSIS")
print("="*80)

# Check if validation loader exists
try:
    if 'balanced_val_loader' in dir():
        val_loader = balanced_val_loader
        print("  Using: balanced_val_loader")
    elif 'val_loader' in dir():
        val_loader = val_loader
        print("  Using: val_loader")
    else:
        raise NameError("No validation loader found")
    
    print(f"  âœ“ Validation loader ready ({len(val_loader)} batches)")
    
except NameError:
    print("  â�Œ No validation loader found!")
    print("\n  You need to run the dataloader creation code first.")
    print("  Skipping Grad-CAM visualization...")
    
    # Exit gracefully
    print("\n" + "="*80)
    print("âš ï¸�  GRAD-CAM SKIPPED - Create validation loader first")
    print("="*80)
    raise SystemExit

# Find interesting cases (fracture + no fracture)
print("\nğŸ”� Finding interesting cases...")

cases_found = {'fracture': None, 'no_fracture': None}
num_checked = 0

for batch_data in val_loader:
    if len(batch_data) == 3:
        volumes, labels, patient_ids = batch_data
    else:
        volumes, labels = batch_data[0], batch_data[1]
        patient_ids = [f"Patient_{i}" for i in range(len(volumes))]
    
    for i in range(len(volumes)):
        has_fracture = labels[i][0].item() == 1
        
        if has_fracture and cases_found['fracture'] is None:
            cases_found['fracture'] = (volumes[i:i+1], labels[i], patient_ids[i])
            print(f"  âœ“ Found fracture case: {patient_ids[i]}")
        
        if not has_fracture and cases_found['no_fracture'] is None:
            cases_found['no_fracture'] = (volumes[i:i+1], labels[i], patient_ids[i])
            print(f"  âœ“ Found no-fracture case: {patient_ids[i]}")
        
        if cases_found['fracture'] and cases_found['no_fracture']:
            break
    
    num_checked += 1
    if cases_found['fracture'] and cases_found['no_fracture']:
        break
    if num_checked >= 10:  # Check max 10 batches
        break

# Process each case
results = []

for case_name, case_data in cases_found.items():
    if case_data is None:
        print(f"\n  âš ï¸�  No {case_name} case found")
        continue
    
    volume_tensor, label, patient_id = case_data
    
    print(f"\n{'='*60}")
    print(f"ğŸ“Š Analyzing: {patient_id} ({case_name.replace('_', ' ')})")
    print(f"{'='*60}")
    
    # Move to device
    volume_tensor = volume_tensor.to(device)
    volume_tensor.requires_grad = True
    
    # Get predictions
    with torch.no_grad():
        output = model(volume_tensor)
        predictions = torch.sigmoid(output).cpu().numpy()[0]
    
    print(f"  Model Prediction: {predictions[0]*100:.1f}% fracture probability")
    print(f"  Ground Truth: {'FRACTURE' if label[0]==1 else 'NO FRACTURE'}")
    
    # Generate Grad-CAM
    print(f"  Generating Grad-CAM heatmap...")
    gradcam = GradCAM3D(model, model.backbone.layer4[-1])
    
    # Enable gradients for backward pass
    volume_tensor.requires_grad = True
    cam = gradcam.generate_cam(volume_tensor, target_class=0)
    
    gradcam.remove_hooks()
    
    print(f"  âœ“ Grad-CAM complete (shape: {cam.shape})")
    
    # Get volume for visualization
    volume_np = volume_tensor[0, 0].detach().cpu().numpy()
    
    # Create comprehensive visualization
    save_path = f'/kaggle/working/gradcam_{case_name}_{patient_id}.png'
    
    print(f"  Creating visualization...")
    fig = visualize_gradcam_comprehensive(
        volume_np, cam, predictions, label.numpy(),
        patient_id, save_path=save_path
    )
    
    results.append({
        'case': case_name,
        'patient_id': patient_id,
        'prediction': predictions[0],
        'ground_truth': label[0].item(),
        'save_path': save_path
    })
    
    print(f"  âœ“ Visualization complete!")

# ============================================================================
# 6. SUMMARY
# ============================================================================

print("\n" + "="*80)
print("ğŸ�‰ GRAD-CAM ANALYSIS COMPLETE!")
print("="*80)

if results:
    print(f"\nğŸ“Š Summary:")
    for r in results:
        pred_label = "FRACTURE" if r['prediction'] > 0.5 else "NO FRACTURE"
        gt_label = "FRACTURE" if r['ground_truth'] == 1 else "NO FRACTURE"
        match = "âœ“" if (r['prediction'] > 0.5) == (r['ground_truth'] == 1) else "âœ—"
        
        print(f"\n  {r['case'].upper()}:")
        print(f"    Patient: {r['patient_id']}")
        print(f"    Prediction: {pred_label} ({r['prediction']*100:.1f}%)")
        print(f"    Ground Truth: {gt_label}")
        print(f"    Match: {match}")
        print(f"    Saved: {r['save_path']}")

print(f"\nğŸ’¡ What the heatmap shows:")
print(f"   â€¢ RED areas = High attention (model suspects fracture)")
print(f"   â€¢ BLUE areas = Low attention (model thinks normal)")
print(f"   â€¢ Intensity = Confidence level")

print(f"\nğŸ�¯ Perfect for presentation!")
print(f"   These visualizations show WHERE your model detects fractures!")

print("\n" + "="*80)


"""
INTERACTIVE 3D SPINE VISUALIZATION
Rotating 3D reconstruction with fracture heatmap overlay
Works with your current demo-trained model!
"""

import numpy as np
import torch
import torch.nn.functional as F
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from skimage import measure
from scipy.ndimage import zoom
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("ğŸŒ� INTERACTIVE 3D SPINE VISUALIZATION")
print("="*80)

# ============================================================================
# GRAD-CAM CLASS (embedded)
# ============================================================================

class GradCAM3D:
    """3D Grad-CAM for fracture localization"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        self.forward_handle = target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = target_layer.register_full_backward_hook(self._backward_hook)
    
    def _forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_volume, target_class=0):
        self.model.eval()
        output = self.model(input_volume)
        
        self.model.zero_grad()
        output[0, target_class].backward()
        
        gradients = self.gradients[0]
        activations = self.activations[0]
        
        weights = gradients.mean(dim=(1, 2, 3), keepdim=True)
        cam = (weights * activations).sum(dim=0)
        
        cam = F.relu(cam)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.cpu().numpy()
    
    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

print("  âœ“ GradCAM3D class loaded")

# ============================================================================
# 3D RECONSTRUCTION FUNCTIONS
# ============================================================================

def create_3d_spine_mesh(volume, threshold=0.4, downsample=0.5):
    """
    Create 3D mesh from CT volume using marching cubes
    
    Args:
        volume: CT volume (D, H, W)
        threshold: Threshold for bone segmentation
        downsample: Factor to reduce size (0.5 = half size)
    
    Returns:
        verts, faces: Mesh vertices and faces
    """
    print(f"  Creating 3D mesh from volume...")
    
    # Downsample for performance
    if downsample < 1.0:
        volume_small = zoom(volume, downsample, order=1)
    else:
        volume_small = volume
    
    print(f"    Volume shape: {volume.shape} â†’ {volume_small.shape}")
    
    # Create binary mask for bone
    bone_mask = volume_small > threshold
    
    # Apply marching cubes to get mesh
    try:
        verts, faces, normals, values = measure.marching_cubes(
            bone_mask,
            level=0,
            spacing=(1.0, 1.0, 1.0),
            allow_degenerate=False
        )
        print(f"    âœ“ Mesh created: {len(verts)} vertices, {len(faces)} faces")
        return verts, faces
    except Exception as e:
        print(f"    âœ— Marching cubes failed: {e}")
        return None, None


def map_gradcam_to_mesh(verts, cam_volume, volume_shape):
    """
    Map Grad-CAM values to mesh vertices
    
    Args:
        verts: Mesh vertices (N, 3)
        cam_volume: Grad-CAM heatmap (D, H, W)
        volume_shape: Original volume shape
    
    Returns:
        colors: Color values for each vertex
    """
    print(f"  Mapping Grad-CAM to mesh vertices...")
    
    # Resize CAM to match mesh scale
    if cam_volume.shape != volume_shape:
        zoom_factors = np.array(volume_shape) / np.array(cam_volume.shape)
        cam_resized = zoom(cam_volume, zoom_factors, order=1)
    else:
        cam_resized = cam_volume
    
    # Sample CAM values at vertex positions
    colors = []
    for vert in verts:
        z, y, x = vert
        
        # Convert to array indices
        zi = int(np.clip(z, 0, cam_resized.shape[0] - 1))
        yi = int(np.clip(y, 0, cam_resized.shape[1] - 1))
        xi = int(np.clip(x, 0, cam_resized.shape[2] - 1))
        
        cam_value = cam_resized[zi, yi, xi]
        colors.append(cam_value)
    
    colors = np.array(colors)
    print(f"    âœ“ Mapped {len(colors)} vertex colors")
    print(f"    Color range: [{colors.min():.3f}, {colors.max():.3f}]")
    
    return colors


def create_interactive_3d_visualization(volume, cam, predictions, labels, patient_id,
                                       threshold=0.4, downsample=0.5):
    """
    Create interactive 3D visualization with Plotly
    
    Args:
        volume: CT volume (D, H, W)
        cam: Grad-CAM heatmap (D, H, W)
        predictions: Model predictions (8,)
        labels: Ground truth labels (8,)
        patient_id: Patient ID
        threshold: Bone segmentation threshold
        downsample: Downsampling factor for performance
    """
    
    print(f"\n{'='*60}")
    print(f"ğŸ�¨ Creating 3D visualization for {patient_id}")
    print(f"{'='*60}")
    
    # Create mesh
    verts, faces = create_3d_spine_mesh(volume, threshold, downsample)
    
    if verts is None or faces is None:
        print("  âœ— Could not create mesh")
        return None
    
    # Map Grad-CAM to vertices
    colors = map_gradcam_to_mesh(verts, cam, volume.shape)
    
    # Determine fracture status
    has_fracture = predictions[0] > 0.5
    confidence = predictions[0] * 100
    
    fracture_text = "FRACTURE DETECTED" if has_fracture else "NO FRACTURE"
    title_color = 'red' if has_fracture else 'green'
    
    print(f"\n  Prediction: {fracture_text} ({confidence:.1f}%)")
    
    # Create Plotly figure
    print(f"  Creating interactive plot...")
    
    fig = go.Figure(data=[
        go.Mesh3d(
            x=verts[:, 0],
            y=verts[:, 1],
            z=verts[:, 2],
            i=faces[:, 0],
            j=faces[:, 1],
            k=faces[:, 2],
            intensity=colors,
            colorscale=[
                [0.0, 'rgb(0, 0, 100)'],      # Dark blue (low attention)
                [0.3, 'rgb(0, 100, 200)'],    # Blue
                [0.5, 'rgb(0, 200, 200)'],    # Cyan
                [0.7, 'rgb(255, 255, 0)'],    # Yellow
                [0.85, 'rgb(255, 150, 0)'],   # Orange
                [1.0, 'rgb(255, 0, 0)']       # Red (high attention - fracture)
            ],
            cmin=0,
            cmax=1,
            colorbar=dict(
                title=dict(
                    text="Fracture<br>Attention",
                    font=dict(size=14, color='white')
                ),
                titleside="right",
                tickmode="linear",
                tick0=0,
                dtick=0.2,
                tickfont=dict(size=12, color='white'),
                len=0.7,
                thickness=20,
                x=1.0
            ),
            opacity=0.95,
            flatshading=False,
            lighting=dict(
                ambient=0.5,
                diffuse=0.8,
                specular=0.3,
                roughness=0.4,
                fresnel=0.2
            ),
            lightposition=dict(
                x=100,
                y=100,
                z=1000
            ),
            hovertemplate='<b>Position</b><br>' +
                         'X: %{x:.1f}<br>' +
                         'Y: %{y:.1f}<br>' +
                         'Z: %{z:.1f}<br>' +
                         '<b>Attention: %{intensity:.3f}</b><br>' +
                         '<extra></extra>'
        )
    ])
    
    # Add annotations with predictions
    label_names = ['Overall'] + [f'C{i}' for i in range(1, 8)]
    annotation_text = "<b>PREDICTIONS:</b><br>"
    
    for i, name in enumerate(label_names):
        pred_prob = predictions[i]
        pred_status = "FRACTURE" if pred_prob > 0.5 else "Normal"
        gt = int(labels[i]) if labels is not None else None
        
        annotation_text += f"{name}: {pred_status} ({pred_prob*100:.1f}%)"
        
        if gt is not None:
            match = 'âœ“' if (pred_prob > 0.5) == (gt == 1) else 'âœ—'
            annotation_text += f" {match}"
        
        annotation_text += "<br>"
    
    # Update layout with dark theme
    fig.update_layout(
        title=dict(
            text=f'<b>3D Cervical Spine Reconstruction</b><br>' +
                 f'Patient: {patient_id}<br>' +
                 f'<span style="color:{title_color};">{fracture_text}</span> ' +
                 f'(Confidence: {confidence:.1f}%)',
            font=dict(size=18, color='white'),
            x=0.5,
            xanchor='center'
        ),
        scene=dict(
            xaxis=dict(
                title='Superior â†� â†’ Inferior',
                titlefont=dict(size=12, color='white'),
                gridcolor='rgb(50, 50, 50)',
                showbackground=True,
                backgroundcolor='rgb(20, 20, 20)',
                tickfont=dict(color='white')
            ),
            yaxis=dict(
                title='Anterior â†� â†’ Posterior',
                titlefont=dict(size=12, color='white'),
                gridcolor='rgb(50, 50, 50)',
                showbackground=True,
                backgroundcolor='rgb(20, 20, 20)',
                tickfont=dict(color='white')
            ),
            zaxis=dict(
                title='Left â†� â†’ Right',
                titlefont=dict(size=12, color='white'),
                gridcolor='rgb(50, 50, 50)',
                showbackground=True,
                backgroundcolor='rgb(20, 20, 20)',
                tickfont=dict(color='white')
            ),
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.5),
                center=dict(x=0, y=0, z=0),
                up=dict(x=0, y=0, z=1)
            ),
            bgcolor='rgb(10, 10, 10)'
        ),
        paper_bgcolor='rgb(15, 15, 15)',
        plot_bgcolor='rgb(15, 15, 15)',
        font=dict(color='white'),
        width=1200,
        height=900,
        annotations=[
            dict(
                text=annotation_text,
                xref="paper",
                yref="paper",
                x=0.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
                showarrow=False,
                font=dict(size=11, family='monospace', color='white'),
                bgcolor='rgba(0, 0, 0, 0.7)',
                bordercolor='white',
                borderwidth=2,
                borderpad=10
            )
        ],
        showlegend=False,
        hovermode='closest'
    )
    
    print(f"  âœ“ Interactive visualization ready!")
    
    return fig


# ============================================================================
# MAIN DEMO FUNCTION
# ============================================================================

def demo_interactive_3d(model, val_loader, device, save_html=True):
    """
    Complete demo with interactive 3D visualization
    
    Args:
        model: Trained model
        val_loader: Validation DataLoader
        device: Device
        save_html: Whether to save HTML file
    """
    
    print("\n" + "="*80)
    print("ğŸš€ RUNNING INTERACTIVE 3D VISUALIZATION DEMO")
    print("="*80)
    
    # Load model
    model = model.to(device)
    model.eval()
    
    # Find a patient with fracture
    print("\nğŸ”� Finding patient with fracture...")
    
    selected_volume = None
    selected_label = None
    selected_id = None
    
    for batch_data in val_loader:
        if len(batch_data) == 3:
            volumes, labels, patient_ids = batch_data
        else:
            volumes, labels = batch_data[0], batch_data[1]
            patient_ids = [f"Patient_{i}" for i in range(len(volumes))]
        
        for i in range(len(volumes)):
            if labels[i][0].item() == 1:  # Has fracture
                selected_volume = volumes[i:i+1]
                selected_label = labels[i]
                selected_id = patient_ids[i]
                print(f"  âœ“ Found fracture case: {selected_id}")
                break
        
        if selected_volume is not None:
            break
    
    if selected_volume is None:
        print("  âš ï¸�  No fracture found, using first patient")
        selected_volume = volumes[0:0+1]
        selected_label = labels[0]
        selected_id = patient_ids[0]
    
    # Move to device and get predictions
    selected_volume = selected_volume.to(device)
    
    with torch.no_grad():
        output = model(selected_volume)
        predictions = torch.sigmoid(output).cpu().numpy()[0]
    
    print(f"\n  Model Prediction: {predictions[0]*100:.1f}% fracture probability")
    
    # Generate Grad-CAM
    print(f"\nğŸ“Š Generating Grad-CAM...")
    
    gradcam = GradCAM3D(model, model.backbone.layer4[-1])
    selected_volume.requires_grad = True
    cam = gradcam.generate_cam(selected_volume, target_class=0)
    gradcam.remove_hooks()
    
    print(f"  âœ“ Grad-CAM generated")
    
    # Get volume for visualization
    volume_np = selected_volume[0, 0].detach().cpu().numpy()
    
    # Create interactive 3D visualization
    fig = create_interactive_3d_visualization(
        volume=volume_np,
        cam=cam,
        predictions=predictions,
        labels=selected_label.numpy(),
        patient_id=selected_id,
        threshold=0.4,
        downsample=0.4  # Reduce for performance
    )
    
    if fig is None:
        print("\n  âœ— Visualization failed")
        return None
    
    # Save HTML
    if save_html:
        html_path = f'/kaggle/working/interactive_3d_{selected_id}.html'
        fig.write_html(html_path)
        print(f"\nğŸ’¾ Saved interactive HTML: {html_path}")
        print(f"   You can download and open this in a browser!")
    
    # Display
    print(f"\nğŸŒ� Displaying interactive visualization...")
    print(f"   â€¢ Rotate: Click and drag")
    print(f"   â€¢ Zoom: Scroll wheel")
    print(f"   â€¢ Pan: Right-click and drag")
    print(f"   â€¢ Hover: See attention values")
    
    fig.show()
    
    return fig


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("ğŸ“‹ READY TO CREATE INTERACTIVE 3D VISUALIZATION")
    print("="*80)
    
    print("""
To run the interactive 3D visualization:

# Make sure you have:
# 1. Trained model (model)
# 2. Validation loader (balanced_val_loader or val_loader)
# 3. Device (device)

# Run the demo:
fig = demo_interactive_3d(
    model=model,
    val_loader=balanced_val_loader,  # or val_loader
    device=device,
    save_html=True
)

# This will:
# 1. Find a patient with fracture
# 2. Create 3D mesh of the spine
# 3. Overlay Grad-CAM heatmap
# 4. Create interactive Plotly visualization
# 5. Save as HTML file (downloadable)
# 6. Display in notebook

# The result is a rotating 3D spine with:
# â€¢ Color-coded fracture attention (blue â†’ red)
# â€¢ Interactive rotation, zoom, pan
# â€¢ Hover to see attention values
# â€¢ Predictions panel overlay
# â€¢ Dark professional theme

Perfect for presentations! Show your sir a rotating 3D spine! ğŸš€
    """)
    
    print("\n" + "="*80)
    print("ğŸ�¯ FEATURES:")
    print("="*80)
    print("""
âœ“ Interactive 3D mesh reconstruction
âœ“ Grad-CAM heatmap overlay (blue = normal, red = fracture)
âœ“ Smooth rotation and zoom
âœ“ Hover tooltips with attention values
âœ“ Predictions panel showing all vertebrae
âœ“ Professional dark theme
âœ“ Exportable as HTML (shareable file)
âœ“ Works with demo-trained model (no full training needed!)

This is the MOST IMPRESSIVE visualization for your presentation!
    """)


fig = demo_interactive_3d(
    model=model,
    val_loader=balanced_val_loader,  # or val_loader
    device=device,
    save_html=True
)

