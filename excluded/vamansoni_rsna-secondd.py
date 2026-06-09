!pip install -q pydicom nibabel scikit-image monai tqdm scikit-learn

import pandas as pd
import numpy as np
import nibabel as nib
import os
import ast
import pydicom
from scipy.ndimage import zoom
from skimage.exposure import equalize_adapthist
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from monai.data import DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.transforms import Compose, RandCropByPosNegLabeld, SpatialPadd
from tqdm import tqdm
import matplotlib.pyplot as plt

print("Setup complete.")


# --- PATHS ---
BASE_INPUT_DIR = '/kaggle/input/rsna-intracranial-aneurysm-detection/'
ORIGINAL_DICOM_DIR = os.path.join(BASE_INPUT_DIR, 'series')
LOCALIZER_CSV_PATH = os.path.join(BASE_INPUT_DIR, 'train_localizers.csv')
VESSEL_SEG_DIR = os.path.join(BASE_INPUT_DIR, 'segmentations') # <-- NEW PATH ADDED

PROCESSED_IMAGES_DIR = '/kaggle/working/processed_images/'
MASKS_DIR = '/kaggle/working/processed_masks/'

# --- DATA & PREPROCESSING PARAMETERS ---
TARGET_SPACING = (1.0, 1.0, 1.0)
SUBSET_SIZE = 200

# --- TRAINING HYPERPARAMETERS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-5
BATCH_SIZE = 4
NUM_EPOCHS = 25
VAL_SPLIT = 0.2
PATCH_SIZE = (96, 96, 96)

# --- MAPPING ---
LOCATION_TO_CHANNEL = { 'Other Posterior Circulation': 0, 'Basilar Tip': 1, 'Right Posterior Communicating Artery': 2, 'Left Posterior Communicating Artery': 3, 'Right Infraclinoid Internal Carotid Artery': 4, 'Left Infraclinoid Internal Carotid Artery': 5, 'Right Supraclinoid Internal Carotid Artery': 6, 'Left Supraclinoid Internal Carotid Artery': 7, 'Right Middle Cerebral Artery': 8, 'Left Middle Cerebral Artery': 9, 'Right Anterior Cerebral Artery': 10, 'Left Anterior Cerebral Artery': 11, 'Anterior Communicating Artery': 12 }

print(f"Configuration complete. Using device: {DEVICE}")


def resample_volume(volume, o_spacing, t_spacing):
    zoom_factor = [o / t for o, t in zip(o_spacing, t_spacing)]
    return zoom(volume, zoom_factor, order=1)

def normalize_intensity(volume, clip_percentiles=(0.5, 99.5)):
    if np.all(volume == 0): return volume
    lower, upper = np.percentile(volume, clip_percentiles)
    clipped = np.clip(volume, lower, upper)
    min_val, max_val = np.min(clipped), np.max(clipped)
    if max_val - min_val == 0: return np.zeros_like(clipped)
    return (clipped - min_val) / (max_val - min_val)

def enhance_contrast_clahe(volume):
    enhanced_volume = np.zeros_like(volume)
    for i in range(volume.shape[2]):
        enhanced_volume[:, :, i] = equalize_adapthist(volume[:, :, i], clip_limit=0.01)
    return enhanced_volume

def reconstruct_from_dicom(series_path):
    if not os.path.isdir(series_path): return None, None
    dicom_files = [pydicom.dcmread(os.path.join(series_path, f)) for f in os.listdir(series_path) if f.endswith('.dcm')]
    if not dicom_files: return None, None
    def get_slice_pos(dcm):
        if 'ImagePositionPatient' in dcm: return float(dcm.ImagePositionPatient[2])
        elif 'SliceLocation' in dcm: return float(dcm.SliceLocation)
        else: return float(dcm.get('InstanceNumber', 0))
    dicom_files.sort(key=get_slice_pos)
    p_space = dicom_files[0].get('PixelSpacing', [1.0, 1.0])
    s_thick = abs(get_slice_pos(dicom_files[1]) - get_slice_pos(dicom_files[0])) if len(dicom_files) > 1 else 1.0
    if s_thick < 1e-3: return None, None
    o_space = [float(p_space[0]), float(p_space[1]), float(s_thick)]
    volume = np.stack([s.pixel_array for s in dicom_files], axis=-1)
    return volume, o_space

def parse_coords(s):
    try: return ast.literal_eval(s)
    except: return [np.nan, np.nan]

def get_original_dicom_info(series_uid):
    s_path = os.path.join(ORIGINAL_DICOM_DIR, series_uid)
    if not os.path.isdir(s_path): return None, None
    dcms = [pydicom.dcmread(os.path.join(s_path, f), stop_before_pixels=True) for f in os.listdir(s_path) if f.endswith('.dcm')]
    if not dcms: return None, None
    def get_slice_pos(dcm):
        if 'ImagePositionPatient' in dcm: return float(dcm.ImagePositionPatient[2])
        elif 'SliceLocation' in dcm: return float(dcm.SliceLocation)
        else: return float(dcm.get('InstanceNumber', 0))
    dcms.sort(key=get_slice_pos)
    sops = [d.SOPInstanceUID for d in dcms]
    return sops

def draw_sphere(mask, center, radius):
    center = [int(round(c)) for c in center]
    x_c, y_c, z_c = center
    for i in range(max(0, x_c - radius), min(mask.shape[0], x_c + radius + 1)):
        for j in range(max(0, y_c - radius), min(mask.shape[1], y_c + radius + 1)):
            for k in range(max(0, z_c - radius), min(mask.shape[2], z_c + radius + 1)):
                if (i - x_c)**2 + (j - y_c)**2 + (k - z_c)**2 <= radius**2:
                    mask[i, j, k] = 1
    return mask

print("All helper functions defined.")


# --- FINAL, CORRECTED Cell for Data Preparation ---

def process_dicom_series(series_uid):
    """
    This single function handles the full pipeline for one series:
    1. Reconstructs from DICOM.
    2. Preprocesses the image (resample, normalize, CLAHE).
    3. Creates the corresponding multi-channel mask.
    Returns the processed image and mask, or None if an error occurs.
    """
    series_path = os.path.join(ORIGINAL_DICOM_DIR, series_uid)
    if not os.path.isdir(series_path): return None, None

    # --- Reconstruction and Metadata Gathering (robust version) ---
    dicom_files = [pydicom.dcmread(os.path.join(series_path, f)) for f in os.listdir(series_path) if f.endswith('.dcm')]
    if not dicom_files: return None, None
    
    def get_slice_pos(dcm):
        if 'ImagePositionPatient' in dcm: return float(dcm.ImagePositionPatient[2])
        elif 'SliceLocation' in dcm: return float(dcm.SliceLocation)
        else: return float(dcm.get('InstanceNumber', 0))
    dicom_files.sort(key=get_slice_pos)

    p_space = dicom_files[0].get('PixelSpacing', [1.0, 1.0])
    s_thick = abs(get_slice_pos(dicom_files[1]) - get_slice_pos(dicom_files[0])) if len(dicom_files) > 1 else 1.0
    if s_thick < 1e-3: return None, None
    
    original_spacing = [float(p_space[0]), float(p_space[1]), float(s_thick)]
    volume = np.stack([s.pixel_array for s in dicom_files], axis=-1)
    if volume.ndim > 3: volume = np.squeeze(volume)
    if volume.ndim != 3: return None, None
    
    # --- Image Preprocessing ---
    resampled = resample_volume(volume, original_spacing, TARGET_SPACING)
    normalized = normalize_intensity(resampled)
    enhanced_image = enhance_contrast_clahe(normalized)

    # --- Mask Generation ---
    sops = [d.SOPInstanceUID for d in dicom_files]
    zoom_factor = [o / t for o, t in zip(original_spacing, TARGET_SPACING)]
    
    mask = np.zeros((13,) + enhanced_image.shape, dtype=np.uint8)
    aneurysms_in_series = localizer_df[localizer_df['SeriesInstanceUID'] == series_uid]

    for _, row in aneurysms_in_series.iterrows():
        try:
            z = sops.index(row['SOPInstanceUID'])
            center = [row['x'] * zoom_factor[0], row['y'] * zoom_factor[1], z * zoom_factor[2]]
            ch_idx = LOCATION_TO_CHANNEL.get(row['location'])
            if ch_idx is not None:
                mask[ch_idx] = draw_sphere(mask[ch_idx], center, radius=3)
        except (ValueError, AttributeError):
            pass

    if np.sum(mask) == 0:
        return None, None # Skip series if no aneurysm was successfully drawn
        
    final_mask = np.transpose(mask, (0, 3, 1, 2))
    
    return enhanced_image, final_mask

# --- MAIN EXECUTION ---
!rm -rf /kaggle/working/*
os.makedirs(PROCESSED_IMAGES_DIR, exist_ok=True)
os.makedirs(MASKS_DIR, exist_ok=True)

localizer_df = pd.read_csv(LOCALIZER_CSV_PATH)
localizer_df['coordinates'] = localizer_df['coordinates'].apply(parse_coords)
coords_df = pd.DataFrame(localizer_df['coordinates'].tolist(), columns=['x', 'y'], index=localizer_df.index)
localizer_df = pd.concat([localizer_df, coords_df], axis=1)
localizer_df.dropna(subset=['x', 'y'], inplace=True)
localizer_df['x'] = localizer_df['x'].astype(int); localizer_df['y'] = localizer_df['y'].astype(int)

positive_series_uids = localizer_df['SeriesInstanceUID'].unique().tolist()
series_to_process = positive_series_uids[:SUBSET_SIZE]

print(f"--- Starting Data Preparation for {len(series_to_process)} series ---")
for series_uid in tqdm(series_to_process, desc="Preparing Data"):
    image, mask = process_dicom_series(series_uid)
    
    if image is not None and mask is not None:
        affine = np.eye(4)
        nib.save(nib.Nifti1Image(image.astype(np.float32), affine), os.path.join(PROCESSED_IMAGES_DIR, f"{series_uid}.nii.gz"))
        nib.save(nib.Nifti1Image(mask.astype(np.uint8), affine), os.path.join(MASKS_DIR, f"{series_uid}.nii.gz"))

print("\n--- Data Preparation Complete! ---")


## ğŸ‘�ï¸� VISUALIZATION: Before vs. After (Same Aneurysm Slice)

print("--- Finding and displaying the same aneurysm slice before and after processing ---")

# --- 1. Find the location of an aneurysm in the raw data ---
localizer_df = pd.read_csv(LOCALIZER_CSV_PATH)
if not localizer_df.empty:
    # Pick the first aneurysm from the list
    first_aneurysm_info = localizer_df.iloc[0]
    sample_series_uid = first_aneurysm_info['SeriesInstanceUID']
    aneurysm_sop_uid = first_aneurysm_info['SOPInstanceUID']

    # Get original DICOM info to find the raw slice index
    # We need to call the full reconstruct function to get original_spacing
    raw_volume_for_spacing, original_spacing = reconstruct_from_dicom(os.path.join(ORIGINAL_DICOM_DIR, sample_series_uid))
    sops = get_original_dicom_info(sample_series_uid)
    
    aneurysm_slice_idx_raw = -1
    if sops:
        try:
            aneurysm_slice_idx_raw = sops.index(aneurysm_sop_uid)
        except ValueError:
            print(f"Warning: Aneurysm slice {aneurysm_sop_uid} not found in series {sample_series_uid}.")

    if aneurysm_slice_idx_raw != -1 and raw_volume_for_spacing is not None:
        raw_volume = raw_volume_for_spacing
        if raw_volume.ndim > 3: raw_volume = np.squeeze(raw_volume)
        
        # --- 2. Display the RAW DICOM slice ---
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(raw_volume[:, :, aneurysm_slice_idx_raw], cmap='gray')
        plt.title(f"Raw DICOM\nSlice index: {aneurysm_slice_idx_raw}")
        plt.axis('off')

        # --- 3. Display the PROCESSED NIfTI slice ---
        processed_path = os.path.join(PROCESSED_IMAGES_DIR, f"{sample_series_uid}.nii.gz")
        if os.path.exists(processed_path):
            # Calculate the corresponding slice index in the resampled volume
            zoom_factor_z = original_spacing[2] / TARGET_SPACING[2]
            aneurysm_slice_idx_processed = int(round(aneurysm_slice_idx_raw * zoom_factor_z))

            processed_data = nib.load(processed_path).get_fdata()
            
            plt.subplot(1, 2, 2)
            # Ensure the processed slice index is within bounds
            aneurysm_slice_idx_processed = min(aneurysm_slice_idx_processed, processed_data.shape[2] - 1)
            plt.imshow(processed_data[:, :, aneurysm_slice_idx_processed], cmap='gray')
            plt.title(f"Processed Scan\nSlice index: {aneurysm_slice_idx_processed}")
            plt.axis('off')
            plt.show()
        else:
            print("Processed file not found. Please run the Data Preparation cell first.")
else:
    print("Localizer CSV is empty.")


import matplotlib.pyplot as plt
import nibabel as nib
import os
import numpy as np

print("--- Verifying a sample image and mask ---")

# 1. Get a list of the masks you just created
mask_files = [f for f in os.listdir(MASKS_DIR) if f.endswith('.nii.gz')]

if not mask_files:
    print("â�Œ Verification failed: No mask files were found in the output directory.")
else:
    # 2. Pick one sample to check
    sample_filename = mask_files[0]
    print(f"Checking sample: {sample_filename}")

    image_path = os.path.join(PROCESSED_IMAGES_DIR, sample_filename)
    mask_path = os.path.join(MASKS_DIR, sample_filename)

    # 3. Load the image and the 13-channel mask
    image_nii = nib.load(image_path)
    mask_nii = nib.load(mask_path)
    image_data = image_nii.get_fdata() # Expected shape: (H, W, D)
    mask_data = mask_nii.get_fdata()   # Expected shape: (C, D, H, W)

    print(f"Loaded image shape: {image_data.shape}")
    print(f"Loaded mask shape: {mask_data.shape}")

    # 4. Find a slice that contains the aneurysm
    # To do this, we "flatten" the 13 channels by taking the max value at each voxel
    combined_mask_3d = np.max(mask_data, axis=0) # Shape: (D, H, W)
    
    # Find the 3D coordinates of any voxel that is not zero
    aneurysm_coords = np.argwhere(combined_mask_3d > 0)
    
    if len(aneurysm_coords) > 0:
        # Get the Z-slice index (depth) from the first found coordinate
        slice_idx_d = aneurysm_coords[0][0]
        print(f"âœ… Aneurysm found on slice (depth index): {slice_idx_d}")

        # 5. Plot the corresponding slices side-by-side
        plt.figure(figsize=(12, 6))

        # Plot the image slice. Image is (H, W, D), so we get the slice with [:, :, slice_idx]
        plt.subplot(1, 2, 1)
        plt.imshow(image_data[:, :, slice_idx_d], cmap='gray', origin='lower')
        plt.title(f'Processed Image (Slice {slice_idx_d})')
        plt.axis('off')

        # Plot the mask slice. Combined mask is (D, H, W), so we get the slice with [slice_idx, :, :]
        plt.subplot(1, 2, 2)
        plt.imshow(combined_mask_3d[slice_idx_d, :, :], cmap='gray', origin='lower')
        plt.title(f'Generated Mask (Slice {slice_idx_d})')
        plt.axis('off')

        plt.show()
    else:
        # This shouldn't happen with the new data prep script, but it's a good safety check
        print("âš ï¸� Verification warning: A mask file was found, but it appears to be all black.")


from sklearn.model_selection import train_test_split
from monai.transforms import Compose, SpatialPadd, RandCropByPosNegLabeld
from torch.utils.data import Dataset
from monai.data import DataLoader

# --- Transform Pipelines (No changes needed) ---
train_transforms = Compose([
    SpatialPadd(keys=['image', 'mask'], spatial_size=PATCH_SIZE, method='end'),
    RandCropByPosNegLabeld(keys=['image', 'mask'], label_key='mask', spatial_size=PATCH_SIZE, pos=1, neg=1, num_samples=2, image_key='image', image_threshold=0)
])
val_transforms = Compose([
    SpatialPadd(keys=['image', 'mask'], spatial_size=PATCH_SIZE, method='end'),
    RandCropByPosNegLabeld(keys=['image', 'mask'], label_key='mask', spatial_size=PATCH_SIZE, pos=1, neg=0, num_samples=1)
])

# --- UPGRADED Dataset Class for 2-Channel Input ---
class AneurysmDataset(Dataset):
    def __init__(self, image_dir, mask_dir, vessel_dir, filenames, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.vessel_dir = vessel_dir # <-- Store path to vessel masks
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        try:
            # 1. Load preprocessed image and aneurysm mask
            img_path = os.path.join(self.image_dir, fname)
            mask_path = os.path.join(self.mask_dir, fname)
            img = nib.load(img_path).get_fdata().astype(np.float32)
            aneurysm_mask = nib.load(mask_path).get_fdata().astype(np.float32)

            # 2. Load corresponding vessel segmentation mask
            vessel_path = os.path.join(self.vessel_dir, fname)
            if os.path.exists(vessel_path):
                vessel_mask = nib.load(vessel_path).get_fdata().astype(np.float32)
                # Resample vessel mask to match the processed image's shape
                zoom_factor = np.array(img.shape) / np.array(vessel_mask.shape)
                vessel_mask = zoom(vessel_mask, zoom_factor, order=0)
            else:
                # If no vessel mask exists, create a blank one
                vessel_mask = np.zeros_like(img)

            # 3. Stack image and vessel mask into a 2-channel input
            # The final shape will be (2, H, W, D)
            img = np.stack([img, vessel_mask], axis=0)
            
            # Transpose to PyTorch's expected format: (C, D, H, W)
            img = np.transpose(img, (0, 3, 1, 2))
            
            sample = {'image': img, 'mask': aneurysm_mask}
            
            if self.transform:
                sample = self.transform(sample)
                
            return sample
        except Exception as e:
            print(f"\n  - Error loading file {fname}: {e}. Skipping.")
            return self.__getitem__((idx + 1) % len(self))

# --- Create Datasets and DataLoaders ---
labeled_files = sorted([f for f in os.listdir(MASKS_DIR) if f.endswith('.nii.gz')])
train_files, val_files = train_test_split(labeled_files, test_size=VAL_SPLIT, random_state=42)

# Pass the VESSEL_SEG_DIR path when creating the datasets
train_dataset = AneurysmDataset(PROCESSED_IMAGES_DIR, MASKS_DIR, VESSEL_SEG_DIR, train_files, transform=train_transforms)
val_dataset = AneurysmDataset(PROCESSED_IMAGES_DIR, MASKS_DIR, VESSEL_SEG_DIR, val_files, transform=val_transforms)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"âœ… DataLoaders are ready with 2-channel input.")


import matplotlib.pyplot as plt

print("--- Verifying a batch from the DataLoader ---")

try:
    # Get one batch of data from the training loader
    batch = next(iter(train_loader))
    img_tensor = batch['image'][0]  # Get the first image in the batch
    mask_tensor = batch['mask'][0] # Get the corresponding mask

    print(f"Image tensor shape: {img_tensor.shape}")
    print(f"Mask tensor shape: {mask_tensor.shape}")

    # Visualize a slice from the middle of the 3D patch
    slice_idx = img_tensor.shape[1] // 2 # Middle slice (dim 1 is Depth)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_tensor[0, slice_idx, :, :].cpu(), cmap='gray')
    plt.title(f'Image Patch (Slice {slice_idx})')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    # Take the max across the 13 channels to see the combined mask
    plt.imshow(torch.max(mask_tensor, dim=0)[0][slice_idx, :, :].cpu(), cmap='gray')
    plt.title(f'Mask Patch (Slice {slice_idx})')
    plt.axis('off')
    plt.show()

except Exception as e:
    print(f"An error occurred while fetching a batch: {e}")


model = UNet(
    spatial_dims=3,
    in_channels=2, # <-- EDIT: Changed from 1 to 2 to accept the new input
    out_channels=13,
    channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2), num_res_units=2
).to(DEVICE)

loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True)
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

print(f"Model upgraded for 2-channel input.")


# --- FINAL CORRECTED Training & Validation Loop ---

best_val_loss = float('inf')
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    print(f"\n--- Epoch {epoch + 1}/{NUM_EPOCHS} ---")
    
    # --- Training Phase ---
    model.train()
    epoch_loss = 0
    for batch_data in tqdm(train_loader, desc="Training"):
        # --- FIX IS HERE ---
        # The DataLoader has already prepared the batch correctly.
        # Access the tensors directly from the dictionary.
        inputs = batch_data['image'].to(DEVICE)
        labels = batch_data['mask'].to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    print(f"Epoch {epoch + 1} Average Training Loss: {avg_train_loss:.4f}")

    # --- Validation Phase ---
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Validating"):
            # --- APPLY THE SAME FIX HERE ---
            inputs = batch_data['image'].to(DEVICE)
            labels = batch_data['mask'].to(DEVICE)
            
            outputs = model(inputs)
            val_loss += loss_function(outputs, labels).item()
    
    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)
    print(f"Epoch {epoch + 1} Average Validation Loss: {avg_val_loss:.4f}")

    # --- Save the Best Model ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), '/kaggle/working/best_model.pth')
        print(f"ğŸ�‰ New best model saved with validation loss: {best_val_loss:.4f}")

print("\n--- Training Complete! ---")


plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Training & Validation Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

