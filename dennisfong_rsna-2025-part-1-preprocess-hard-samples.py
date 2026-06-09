import os
import json
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
from tqdm import tqdm

# --- Config ---
SEGMENTATION_DIR = "/kaggle/input/rsna-intracranial-aneurysm-detection/segmentations/"
OUTPUT_PATCHES_DIR = "./patches_from_segmentations/"
OUTPUT_MASKS_DIR = "./aligned_masks/"

TARGET_SPACING = np.array([1.0, 1.0, 1.0])  # z, y, x in mm
STACK_SIZE = 5
PATCH_SIZE = (224, 224)
NUM_SAMPLES_PER_LABEL = 16

DEBUG = True

# Vessel label mapping
VESSEL_LABELS = {
    1: "Other Posterior Circulation",
    2: "Basilar Tip",
    3: "Right Posterior Communicating Artery",
    4: "Left Posterior Communicating Artery",
    5: "Right Infraclinoid Internal Carotid Artery",
    6: "Left Infraclinoid Internal Carotid Artery",
    7: "Right Supraclinoid Internal Carotid Artery",
    8: "Left Supraclinoid Internal Carotid Artery",
    9: "Right Middle Cerebral Artery",
    10: "Left Middle Cerebral Artery",
    11: "Right Anterior Cerebral Artery",
    12: "Left Anterior Cerebral Artery",
    13: "Anterior Communicating Artery"
}

os.makedirs(OUTPUT_PATCHES_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASKS_DIR, exist_ok=True)



# --- Helper Functions ---

def load_image_and_mask(series_uid):
    image_path = os.path.join(SEGMENTATION_DIR, f"{series_uid}.nii")
    mask_path = os.path.join(SEGMENTATION_DIR, f"{series_uid}_cowseg.nii")

    if not os.path.exists(image_path) or not os.path.exists(mask_path):
        raise FileNotFoundError(f"Missing image or mask for {series_uid}")

    image_sitk = sitk.ReadImage(image_path)
    mask_sitk = sitk.ReadImage(mask_path)

    image_np = sitk.GetArrayFromImage(image_sitk)
    mask_np = sitk.GetArrayFromImage(mask_sitk)

    spacing = list(image_sitk.GetSpacing())[::-1]  # Convert to z, y, x
    origin = image_sitk.GetOrigin()
    direction = image_sitk.GetDirection()

    return image_np, mask_np, spacing, origin, direction


def resample_volume(volume, original_spacing, new_spacing, order=3):
    resize_factor = original_spacing / new_spacing
    return zoom(volume, resize_factor, order=order, mode='nearest')


def normalize_intensity(image, modality):
    if modality == 'CT':
        WL, WW = 300, 600
        min_val = WL - WW // 2
        max_val = WL + WW // 2
        image = np.clip(image, min_val, max_val)
        image = (image - min_val) / (max_val - min_val)
    else:
        mean, std = np.mean(image), np.std(image)
        if std > 0:
            image = (image - mean) / std
    return image


def save_mask_as_nii(mask_np, save_path, spacing, origin=None, direction=None):
    mask_sitk = sitk.GetImageFromArray(mask_np.astype(np.uint8))
    mask_sitk.SetSpacing(tuple(spacing[::-1]))  # Convert to x, y, z
    if origin is not None:
        mask_sitk.SetOrigin(origin)
    if direction is not None:
        mask_sitk.SetDirection(direction)
    sitk.WriteImage(mask_sitk, save_path)


def hard_sample_around_label(resampled_mask, label, num_samples=NUM_SAMPLES_PER_LABEL, radius=10):
    vessel_coords = np.argwhere(resampled_mask == label)
    if len(vessel_coords) == 0:
        return np.empty((0, 3), dtype=int)

    neg_coords = []
    np.random.shuffle(vessel_coords)

    for coord in vessel_coords:
        if len(neg_coords) >= num_samples:
            break
        z, y, x = coord
        z_min, z_max = max(0, z - radius), min(resampled_mask.shape[0], z + radius + 1)
        y_min, y_max = max(0, y - radius), min(resampled_mask.shape[1], y + radius + 1)
        x_min, x_max = max(0, x - radius), min(resampled_mask.shape[2], x + radius + 1)

        neighborhood = resampled_mask[z_min:z_max, y_min:y_max, x_min:x_max]
        bg_voxels = np.argwhere(neighborhood == 0)
        if len(bg_voxels) == 0:
            continue

        chosen_bg = bg_voxels[np.random.choice(len(bg_voxels))]
        global_coord = chosen_bg + np.array([z_min, y_min, x_min])
        neg_coords.append(global_coord)

    if len(neg_coords) > num_samples:
        indices = np.random.choice(len(neg_coords), num_samples, replace=False)
        neg_coords = np.array(neg_coords)[indices]
    else:
        neg_coords = np.array(neg_coords)

    return neg_coords


def create_2d5_stack(volume, center_z, center_y, center_x, patch_size, stack_size):
    z_dim, y_dim, x_dim = volume.shape
    half_stack = stack_size // 2
    half_patch_y, half_patch_x = patch_size[0] // 2, patch_size[1] // 2

    stack = []
    for i in range(center_z - half_stack, center_z + half_stack + 1):
        if 0 <= i < z_dim:
            slice_data = volume[i]
            crop_y_min = max(0, center_y - half_patch_y)
            crop_y_max = min(y_dim, center_y + half_patch_y)
            crop_x_min = max(0, center_x - half_patch_x)
            crop_x_max = min(x_dim, center_x + half_patch_x)

            cropped_slice = slice_data[crop_y_min:crop_y_max, crop_x_min:crop_x_max]

            pad_y = patch_size[0] - cropped_slice.shape[0]
            pad_x = patch_size[1] - cropped_slice.shape[1]

            padded_slice = np.pad(cropped_slice,
                                  ((pad_y // 2, pad_y - pad_y // 2),
                                   (pad_x // 2, pad_x - pad_x // 2)),
                                  mode='constant')
            stack.append(padded_slice)
        else:
            stack.append(np.zeros(patch_size))
    return np.stack(stack)


# Updated main pipeline to store coordinates
def generate_for_all_segmentations():
    """
    Main function to generate and save hard samples along with their metadata.
    For each series:
    1. Load and preprocess image/mask
    2. Generate hard samples
    3. Save patches and coordinates
    """
    nii_files = sorted([
        f for f in os.listdir(SEGMENTATION_DIR)
        if f.endswith(".nii") and "_cowseg" in f
    ])

    if DEBUG:
        nii_files = nii_files[:10]

    for fname in tqdm(nii_files, desc="Processing segmentation masks"):
        series_uid = fname.replace("_cowseg.nii", "")

        try:
            print(f"\n--- Processing {series_uid} ---")
            image_volume, mask_np, original_spacing, origin, direction = load_image_and_mask(series_uid)

            resampled_image = resample_volume(image_volume, np.array(original_spacing), TARGET_SPACING, order=3)
            resampled_mask = resample_volume(mask_np, np.array(original_spacing), TARGET_SPACING, order=0).astype(np.uint8)

            normalized_image = normalize_intensity(resampled_image, modality='CT')

            # Save aligned mask
            mask_out_dir = os.path.join(OUTPUT_MASKS_DIR, series_uid)
            os.makedirs(mask_out_dir, exist_ok=True)
            mask_path = os.path.join(mask_out_dir, "aligned_mask.nii.gz")
            save_mask_as_nii(resampled_mask, mask_path, spacing=TARGET_SPACING, origin=origin, direction=direction)

            # Output patches and metadata
            patch_out_dir = os.path.join(OUTPUT_PATCHES_DIR, series_uid)
            os.makedirs(patch_out_dir, exist_ok=True)

            # Create metadata dictionary for this series
            series_metadata = {
                'series_uid': series_uid,
                'samples': []
            }

            for label in range(1, 14):  # Vessel labels 1â€“13
                coords = hard_sample_around_label(resampled_mask, label)
                for i, (z, y, x) in enumerate(coords):
                    # Generate and save patch
                    patch = create_2d5_stack(normalized_image, z, y, x, PATCH_SIZE, STACK_SIZE)
                    patch_fname = f"hard_label{label}_{i}.npy"
                    patch_path = os.path.join(patch_out_dir, patch_fname)
                    np.save(patch_path, patch)

                    # Store metadata
                    sample_info = {
                        'patch_file': patch_fname,
                        'label': int(label),
                        'coordinates': {
                            'z': int(z),
                            'y': int(y),
                            'x': int(x)
                        }
                    }
                    series_metadata['samples'].append(sample_info)

            # Save metadata
            metadata_path = os.path.join(patch_out_dir, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(series_metadata, f, indent=2)

            print(f"âœ“ Finished {series_uid}")

        except Exception as e:
            print(f"â�Œ Error in {series_uid}: {e}")# Cell removed to avoid duplication


# --- Run ---
if __name__ == "__main__":
    generate_for_all_segmentations()


# --- Visualization Functions ---

def visualize_hard_sample_patch(patch_path, title=None):
    """
    Visualizes a 2.5D patch (stack of slices) with proper windowing.
    
    Args:
        patch_path: Path to the .npy file containing the patch
        title: Optional title for the plot
    """
    # Load the patch
    patch = np.load(patch_path)
    
    # Create a figure with subplots for each slice
    fig, axes = plt.subplots(1, STACK_SIZE, figsize=(15, 3))
    if title:
        fig.suptitle(title, fontsize=12)
    else:
        fig.suptitle('2.5D Patch Visualization (5 consecutive slices)', fontsize=12)
    
    for i, ax in enumerate(axes):
        ax.imshow(patch[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Slice {i+1}')
    
    plt.tight_layout()
    plt.show()

def visualize_hard_sample_location(series_uid, coords, label=None, radius=20):
    """
    Visualizes the location of a hard sample in the context of the original volume.
    Shows the central slice with the patch location marked.
    
    Args:
        series_uid: ID of the CT series
        coords: Dictionary containing z, y, x coordinates
        label: The vessel label number (1-13)
        radius: Size of the context region to show around the patch
    """
    # Get vessel name if label is provided
    vessel_name = VESSEL_LABELS.get(label, "Unknown") if label is not None else "Unknown"
    # Load the original image and mask
    image_volume, mask_np, original_spacing, _, _ = load_image_and_mask(series_uid)
    
    # Get original dimensions
    orig_depth, orig_height, orig_width = image_volume.shape
    
    # Calculate the inverse transformation (from resampled to original space)
    spacing_scale = np.array(original_spacing) / TARGET_SPACING
    
    # Transform coordinates from resampled space (1mm) back to original space
    # Use floor division to ensure we get valid indices
    z = min(orig_depth - 1, max(0, int(coords['z'] / spacing_scale[0])))
    y = min(orig_height - 1, max(0, int(coords['y'] / spacing_scale[1])))
    x = min(orig_width - 1, max(0, int(coords['x'] / spacing_scale[2])))
    
    # Calculate patch size in original space
    # We need to scale the patch size by the ratio of spacings
    # If original spacing is larger than 1mm, the patch should be smaller in voxel units
    patch_size_y = int(PATCH_SIZE[0] * (TARGET_SPACING[1] / original_spacing[1]))
    patch_size_x = int(PATCH_SIZE[1] * (TARGET_SPACING[2] / original_spacing[2]))
    
    # Ensure minimum patch size
    patch_size_y = max(patch_size_y, 32)
    patch_size_x = max(patch_size_x, 32)
    
    # Scale radius according to spacing
    scaled_radius = int(radius * (TARGET_SPACING[1] / max(original_spacing[1:])))  # Use max of x,y spacing
    
    # Extract regions for visualization with bounds checking
    z_min = max(0, z - scaled_radius)
    z_max = min(orig_depth, z + scaled_radius)
    y_min = max(0, y - scaled_radius)
    y_max = min(orig_height, y + scaled_radius)
    x_min = max(0, x - scaled_radius)
    x_max = min(orig_width, x + scaled_radius)
    
    # Get the slices with bounds checking
    image_slice = image_volume[z]
    mask_slice = mask_np[z]
    
    # Create figure
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Hard Sample Location (Series: {series_uid})\nVessel: {vessel_name}', fontsize=12)
    
    # Plot full slice with zoomed area marked
    ax1.imshow(image_slice, cmap='gray')
    
    # Add vessel overlay for the full slice
    if label is not None:
        # Create binary mask for the specific vessel in full slice
        full_vessel_mask = (mask_slice == label)
        # Use a high-contrast colormap
        vessel_cmap = plt.cm.RdYlBu  # Red-Yellow-Blue colormap for better contrast
        vessel_cmap.set_bad(alpha=0)
        full_mask_overlay = np.ma.masked_where(~full_vessel_mask, np.ones_like(mask_slice))
        ax1.imshow(full_mask_overlay, cmap=vessel_cmap, alpha=0.6)  # Increased opacity
    
    # Draw the bounding box for the zoomed area
    zoom_width = x_max - x_min
    zoom_height = y_max - y_min
    rect = plt.Rectangle((x_min, y_min), 
                        zoom_width, zoom_height,
                        fill=False, color='red', linewidth=2)
    ax1.add_patch(rect)
    
    # Add center point
    ax1.plot(x, y, 'r+', markersize=10)
    
    # Add size annotation
    ax1.text(x_min, y_min-5, 
             f'Zoom area: {zoom_width}x{zoom_height}px\nSpacing: {original_spacing[1]:.2f}x{original_spacing[2]:.2f}mm', 
             color='red', fontsize=8)
    
    ax1.set_title(f'Full Slice with Zoom Location\n({VESSEL_LABELS[label] if label else "Unknown Vessel"})')
    ax1.axis('off')
    
    # Plot zoomed context region
    context_img = image_slice[y_min:y_max, x_min:x_max]
    context_mask = mask_slice[y_min:y_max, x_min:x_max]
    
    # Draw patch size in zoomed view
    patch_rect = plt.Rectangle((x-x_min-patch_size_x//2, y-y_min-patch_size_y//2),
                              patch_size_x, patch_size_y,
                              fill=False, color='yellow', linewidth=1, linestyle='--')
    
    ax2.imshow(context_img, cmap='gray')
    ax2.add_patch(patch_rect)  # Add patch size indicator to zoomed view
    ax2.set_title('Zoomed Context (Image)')
    ax2.axis('off')
    
    # Plot target vessel overlay in context region
    ax2.imshow(context_img, cmap='gray')
    
    if label is not None:
        # Create binary mask for the specific vessel
        target_vessel_mask = (context_mask == label)
        # Use high-contrast colormap for target vessel
        vessel_cmap = plt.cm.RdYlBu
        vessel_cmap.set_bad(alpha=0)  # Make non-vessel areas transparent
        
        # Create masked array for overlay
        mask_overlay = np.ma.masked_where(~target_vessel_mask, np.ones_like(context_mask))
        ax2.imshow(mask_overlay, cmap=vessel_cmap, alpha=0.7)  # Increased opacity
        ax2.set_title(f'Target Vessel Only\n({VESSEL_LABELS[label]})')
    else:
        ax2.set_title('Target Vessel\n(No specific vessel)')
    ax2.axis('off')

    # Plot all vessels overlay in context region
    ax3.imshow(context_img, cmap='gray')
    # Create mask for all vessels
    all_vessels_mask = (context_mask > 0)
    # Use distinct colors for different vessels
    all_vessels_cmap = plt.cm.nipy_spectral  # More distinct color separation
    all_vessels_cmap.set_bad(alpha=0)
    
    # Create masked array for all vessels overlay
    all_mask_overlay = np.ma.masked_where(~all_vessels_mask, context_mask)
    ax3.imshow(all_mask_overlay, cmap=all_vessels_cmap, alpha=0.6)  # Increased opacity
    ax3.set_title('All Vessels Overlay')
    ax3.axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_random_hard_samples(n_samples=5):
    """
    Visualizes random hard samples from the dataset.
    
    Args:
        n_samples: Number of random samples to visualize
    """
    # Get list of all patch directories
    series_dirs = [d for d in os.listdir(OUTPUT_PATCHES_DIR) 
                  if os.path.isdir(os.path.join(OUTPUT_PATCHES_DIR, d))]
    
    for _ in range(n_samples):
        # Randomly select a series
        series_uid = np.random.choice(series_dirs)
        patch_dir = os.path.join(OUTPUT_PATCHES_DIR, series_uid)
        
        # Load metadata
        metadata_path = os.path.join(patch_dir, 'metadata.json')
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except FileNotFoundError:
            print(f"Metadata not found for series {series_uid}, skipping...")
            continue
        
        # Randomly select a sample from metadata
        if not metadata['samples']:
            continue
            
        sample = np.random.choice(metadata['samples'])
        patch_path = os.path.join(patch_dir, sample['patch_file'])
        
        vessel_label = sample['label']
        print(f"\nVisualizing hard sample:")
        print(f"Series: {series_uid}")
        print(f"Vessel: {VESSEL_LABELS[vessel_label]} (Label {vessel_label})")
        print(f"Coordinates: z={sample['coordinates']['z']}, "
              f"y={sample['coordinates']['y']}, x={sample['coordinates']['x']}")
        
        # Visualize both patch and its location
        title = f"Hard Sample - {VESSEL_LABELS[vessel_label]}"
        visualize_hard_sample_patch(patch_path, title)
        visualize_hard_sample_location(series_uid, sample['coordinates'], label=vessel_label)
        print("-" * 80)



# Example: Visualize 5 random hard samples with error handling
print("Visualizing 5 random hard samples from the dataset...")
try:
    visualize_random_hard_samples(n_samples=5)
except Exception as e:
    print(f"Error during visualization: {str(e)}")
    print("Original coordinates:", sample['coordinates'])
    print("Transformed coordinates:", {'z': z, 'y': y, 'x': x})
    print("Original spacing:", original_spacing)
    print("Image volume shape:", image_volume.shape)



# Example usage: Visualize 5 random hard samples
print("Visualizing 5 random hard samples from the dataset...")
visualize_random_hard_samples(n_samples=5)


