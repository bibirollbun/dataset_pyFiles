import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# Define paths and load data
folder_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_00e047"
output_path = "/kaggle/working/" 

slice_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.jpg')])
print(f"Found {len(slice_files)} slice files")

temp_dir = os.path.join(output_path, "temp_frames")
os.makedirs(temp_dir, exist_ok=True)

def create_multiview_animation():
        
    sample_indices = list(range(0, len(slice_files), 4))  # Take every 4th slice
    
    print(f"Loading {len(sample_indices)} slices...")
    slices = [imageio.imread(os.path.join(folder_path, slice_files[i])) for i in sample_indices]
    volume = np.stack(slices, axis=0)
    
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    
    print(f"Volume shape: {volume.shape}")
    
    frame_files = []
    num_frames = min(30, volume.shape[0])  # Limit number of frames
    
    for i in range(num_frames):
        # Calculate positions for slices (move through the volume)
        z_pos = int((i / num_frames) * volume.shape[0])
        y_pos = int(volume.shape[1] / 2)
        x_pos = int(volume.shape[2] / 2)
        
        # Create a figure with 3 subplots for different views
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY plane (top-down view)
        axes[0].imshow(volume[z_pos], cmap='gray')
        axes[0].set_title(f"XY Plane (z={z_pos})")
        axes[0].axhline(y=y_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[0].axvline(x=x_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # XZ plane (side view)
        axes[1].imshow(volume[:, y_pos, :], cmap='gray')
        axes[1].set_title(f"XZ Plane (y={y_pos})")
        axes[1].axhline(y=z_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[1].axvline(x=x_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # YZ plane (front view)
        axes[2].imshow(volume[:, :, x_pos], cmap='gray')
        axes[2].set_title(f"YZ Plane (x={x_pos})")
        axes[2].axhline(y=z_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[2].axvline(x=y_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Remove ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save the frame
        frame_file = os.path.join(temp_dir, f"multiview_{i:03d}.png")
        plt.savefig(frame_file, dpi=100)
        plt.close(fig)
        
        frame_files.append(frame_file)
    
    # Create a GIF from the frames
    print("Creating GIF...")
    frames = [imageio.imread(f) for f in frame_files]
    gif_path = os.path.join(output_path, "multiview_raw.gif")
    imageio.mimsave(gif_path, frames, duration=0.2)
 
    return gif_path

multiview_path = create_multiview_animation()

# Clean up temp files
print("Cleaning up temporary files...")
for file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, file))
os.rmdir(temp_dir)

print("Process complete!")


import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy import signal  # Added import for signal.convolve2d

# Define paths and load data
folder_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_00e047"
output_path = "/kaggle/working/" 
slice_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.jpg')])
print(f"Found {len(slice_files)} slice files")
temp_dir = os.path.join(output_path, "temp_frames")
os.makedirs(temp_dir, exist_ok=True)

def wiener_filter(img, kernel_size=11, noise_power=0.02):
    """
    Apply Wiener filter for denoising a grayscale image.
    
    Parameters:
        img: Input grayscale image
        kernel_size: Size of the kernel (odd number)
        noise_power: Estimated noise power
        
    Returns:
        Denoised image
    """
    img_norm = img.astype(np.float32) / 255.0
    
    # Create a local mean filter
    kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
    
    # Compute local mean
    img_mean = signal.convolve2d(img_norm, kernel, mode='same')
    
    # Compute local variance
    img_sqr_mean = signal.convolve2d(img_norm**2, kernel, mode='same')
    img_var = img_sqr_mean - img_mean**2
    
    # Ensure variance is positive
    img_var = np.maximum(img_var, 0)
    
    # Apply Wiener filter formula
    denoised = img_mean + ((img_var - noise_power) / np.maximum(img_var, noise_power)) * (img_norm - img_mean)
    
    denoised = np.clip(denoised, 0, 1)
    
    return denoised  # Return as float in [0,1] range to match volume normalization

def create_multiview_animation():
        
    sample_indices = list(range(0, len(slice_files), 4))  # Take every 4th slice
    
    print(f"Loading {len(sample_indices)} slices...")
    
    # Load slices and apply Wiener filter
    slices = []
    for i in sample_indices:
        img = imageio.imread(os.path.join(folder_path, slice_files[i]))
        # Apply Wiener filter with specified parameters
        filtered_img = wiener_filter(img, kernel_size=11, noise_power=0.02)
        slices.append(filtered_img)
    
    volume = np.stack(slices, axis=0)
    
    # No need to normalize again since wiener_filter already returns values in [0,1]
    # Just make sure all values are within [0,1] range
    volume = np.clip(volume, 0, 1)
    
    print(f"Volume shape: {volume.shape}")
    
    frame_files = []
    num_frames = min(30, volume.shape[0])  # Limit number of frames
    
    for i in range(num_frames):
        # Calculate positions for slices (move through the volume)
        z_pos = int((i / num_frames) * volume.shape[0])
        y_pos = int(volume.shape[1] / 2)
        x_pos = int(volume.shape[2] / 2)
        
        # Create a figure with 3 subplots for different views
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY plane (top-down view)
        axes[0].imshow(volume[z_pos], cmap='gray')
        axes[0].set_title(f"XY Plane (z={z_pos})")
        axes[0].axhline(y=y_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[0].axvline(x=x_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # XZ plane (side view)
        axes[1].imshow(volume[:, y_pos, :], cmap='gray')
        axes[1].set_title(f"XZ Plane (y={y_pos})")
        axes[1].axhline(y=z_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[1].axvline(x=x_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # YZ plane (front view)
        axes[2].imshow(volume[:, :, x_pos], cmap='gray')
        axes[2].set_title(f"YZ Plane (x={x_pos})")
        axes[2].axhline(y=z_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[2].axvline(x=y_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Remove ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save the frame
        frame_file = os.path.join(temp_dir, f"multiview_{i:03d}.png")
        plt.savefig(frame_file, dpi=100)
        plt.close(fig)
        
        frame_files.append(frame_file)
    
    # Create a GIF from the frames
    print("Creating GIF...")
    frames = [imageio.imread(f) for f in frame_files]
    gif_path = os.path.join(output_path, "multiview_wiener_.gif")
    imageio.mimsave(gif_path, frames, duration=0.2)
 
    return gif_path

multiview_path = create_multiview_animation()

# Clean up temp files
print("Cleaning up temporary files...")
for file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, file))
os.rmdir(temp_dir)
print("Process complete!")


!pip install bm3d

import os
import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import bm3d  # Import the BM3D library

# Define paths and load data
folder_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_00e047"
output_path = "/kaggle/working/" 
slice_files = sorted([file for file in os.listdir(folder_path) if file.endswith('.jpg')])
print(f"Found {len(slice_files)} slice files")

temp_dir = os.path.join(output_path, "temp_frames")
os.makedirs(temp_dir, exist_ok=True)

def create_multiview_animation():
        
    sample_indices = list(range(0, len(slice_files), 4))  # Take every 4th slice
    
    print(f"Loading {len(sample_indices)} slices...")
    # Load original slices
    slices = [imageio.imread(os.path.join(folder_path, slice_files[i])) for i in sample_indices]
    
    # Convert images to float and normalize to [0, 1] for BM3D processing
    normalized_slices = []
    for img in slices:
        img_float = img.astype(np.float32) / 255.0
        normalized_slices.append(img_float)
    
    # Apply BM3D denoising to each slice
    print("Applying BM3D denoising with sigma_psd=0.15...")
    denoised_slices = []
    for i, img in enumerate(normalized_slices):
        print(f"Denoising slice {i+1}/{len(normalized_slices)}...")
        denoised_img = bm3d.bm3d(img, sigma_psd=0.15)
        denoised_slices.append(denoised_img)
    
    # Stack slices to create volume
    volume = np.stack(denoised_slices, axis=0)
    
    # Normalize the denoised volume for visualization
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    
    print(f"Volume shape: {volume.shape}")
    
    frame_files = []
    num_frames = min(30, volume.shape[0])  # Limit number of frames
    
    for i in range(num_frames):
        # Calculate positions for slices (move through the volume)
        z_pos = int((i / num_frames) * volume.shape[0])
        y_pos = int(volume.shape[1] / 2)
        x_pos = int(volume.shape[2] / 2)
        
        # Create a figure with 3 subplots for different views
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # XY plane (top-down view)
        axes[0].imshow(volume[z_pos], cmap='gray')
        axes[0].set_title(f"XY Plane (z={z_pos})")
        axes[0].axhline(y=y_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[0].axvline(x=x_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # XZ plane (side view)
        axes[1].imshow(volume[:, y_pos, :], cmap='gray')
        axes[1].set_title(f"XZ Plane (y={y_pos})")
        axes[1].axhline(y=z_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[1].axvline(x=x_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # YZ plane (front view)
        axes[2].imshow(volume[:, :, x_pos], cmap='gray')
        axes[2].set_title(f"YZ Plane (x={x_pos})")
        axes[2].axhline(y=z_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        axes[2].axvline(x=y_pos, color='r', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # Remove ticks
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        
        # Save the frame
        frame_file = os.path.join(temp_dir, f"multiview_{i:03d}.png")
        plt.savefig(frame_file, dpi=100)
        plt.close(fig)
        
        frame_files.append(frame_file)
    
    # Create a GIF from the frames
    print("Creating GIF...")
    frames = [imageio.imread(f) for f in frame_files]
    gif_path = os.path.join(output_path, "multiview_denoised.gif")
    imageio.mimsave(gif_path, frames, duration=0.2)
 
    return gif_path

# Make sure BM3D is installed
try:
    import bm3d
except ImportError:
    print("Installing BM3D...")
    !pip install bm3d

# Create the animation
multiview_path = create_multiview_animation()

# Clean up temp files
print("Cleaning up temporary files...")
for file in os.listdir(temp_dir):
    os.remove(os.path.join(temp_dir, file))
os.rmdir(temp_dir)

print(f"Process complete! Denoised GIF saved at: {multiview_path}")

