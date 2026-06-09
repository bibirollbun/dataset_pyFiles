# Install required packages
from IPython.display import clear_output
!pip install -q bm3d n2v tensorflow==2.13.0 csbdeep pillow tqdm
clear_output()


import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import glob
import time
import cv2
from skimage import img_as_float
import bm3d
from scipy import signal
from n2v.models import N2V
import warnings
from tqdm.notebook import tqdm

# Suppress warnings and tqdm output
warnings.filterwarnings('ignore')
tqdm.__init__ = lambda *args, **kwargs: None
tqdm.update = lambda *args, **kwargs: None
tqdm.close = lambda *args, **kwargs: None
tqdm.__iter__ = lambda self: iter([])

# Define paths
data_path = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_00e047'
MODEL_PATH = '/kaggle/input/byu-denoising-cryo-et-with-noise2void/models/'

def load_volume(folder_path, max_slices=30, start_slice=0):
    """Load a subset of slices from the volume"""
    all_files = sorted(glob.glob(os.path.join(folder_path, '*.jpg')))
    selected_files = all_files[start_slice:start_slice+max_slices]
    
    volume = []
    for file in selected_files:
        img = np.array(Image.open(file).convert('L'))  # Convert to grayscale
        volume.append(img)
    
    return np.array(volume)

def process_cv2_denoise(volume, h=10, template_window_size=7, search_window_size=21):
    """Process each slice with OpenCV's fastNlMeansDenoising"""
    output = np.zeros_like(volume)
    start_time = time.time()
    
    for i, slice_img in enumerate(volume):
        output[i] = cv2.fastNlMeansDenoising(
            slice_img, 
            None, 
            h=h,
            templateWindowSize=template_window_size,
            searchWindowSize=search_window_size
        )
    
    total_time = time.time() - start_time
    return output, total_time

def process_gaussian(volume, kernel_size=5):
    """Process each slice with Gaussian blur"""
    output = np.zeros_like(volume)
    start_time = time.time()
    
    for i, slice_img in enumerate(volume):
        output[i] = cv2.GaussianBlur(slice_img, (kernel_size, kernel_size), 0)
    
    total_time = time.time() - start_time
    return output, total_time

def process_bm3d(volume, sigma_psd=0.1):
    """Process volume with BM3D denoising"""
    output = np.zeros_like(volume, dtype=np.float32)
    start_time = time.time()
    
    for i, slice_img in enumerate(volume):
        # Convert to float and normalize
        slice_float = img_as_float(slice_img)
        
        # Apply BM3D denoising
        denoised_slice = bm3d.bm3d(slice_float, sigma_psd=sigma_psd)
        
        # Convert back to uint8
        output[i] = (denoised_slice * 255).astype(np.uint8)
    
    total_time = time.time() - start_time
    return output, total_time

def process_wiener(volume, kernel_size=5, noise_power=0.01):
    """Process volume with Wiener filter"""
    output = np.zeros_like(volume, dtype=np.float32)
    start_time = time.time()
    
    for i, slice_img in enumerate(volume):
        # Normalize to [0, 1] range
        slice_norm = slice_img.astype(np.float32) / 255.0
        
        # Create a local mean filter
        kernel = np.ones((kernel_size, kernel_size)) / (kernel_size**2)
        
        # Compute local mean
        img_mean = signal.convolve2d(slice_norm, kernel, mode='same')
        
        # Compute local variance
        img_sqr_mean = signal.convolve2d(slice_norm**2, kernel, mode='same')
        img_var = img_sqr_mean - img_mean**2
        
        # Ensure variance is positive
        img_var = np.maximum(img_var, 0)
        
        # Apply Wiener filter formula
        denoised = img_mean + ((img_var - noise_power) / np.maximum(img_var, noise_power)) * (slice_norm - img_mean)
        
        # Clip values to valid range and convert back to 8-bit
        denoised = np.clip(denoised, 0, 1)
        output[i] = (denoised * 255).astype(np.uint8)
    
    total_time = time.time() - start_time
    return output, total_time

def process_crude_3d(volume, kernel_size=3):
    """Process with crude 3D denoising by averaging neighboring slices"""
    output = np.zeros_like(volume)
    start_time = time.time()
    
    for i in range(volume.shape[0]):
        # Get neighboring slices
        slice_min = max(0, i - kernel_size//2)
        slice_max = min(volume.shape[0], i + kernel_size//2 + 1)
        # Average the neighboring slices
        neighbors = volume[slice_min:slice_max]
        # Apply 2D denoising to each slice and then average
        denoised_neighbors = []
        for neighbor in neighbors:
            denoised = cv2.fastNlMeansDenoising(neighbor, None, h=10)
            denoised_neighbors.append(denoised)
        output[i] = np.mean(denoised_neighbors, axis=0).astype(np.uint8)
    
    total_time = time.time() - start_time
    return output, total_time

def process_noise2void(volume):
    """Process volume with pre-trained Noise2Void model"""
    output = np.zeros_like(volume, dtype=np.uint8)
    start_time = time.time()
    
    # Redirect standard output to suppress N2V loading/processing messages
    import os, sys
    original_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    
    try:
        # Load the pre-trained Noise2Void model
        model = N2V(config=None, name='n2v_cryoET_8slices', basedir=MODEL_PATH)
        
        for i, slice_img in enumerate(volume):
            # Add dimensions for N2V (SYXC format where S is sample/batch dimension)
            img_for_pred = slice_img[np.newaxis, ..., np.newaxis]
            
            # Apply Noise2Void denoising
            denoised = model.predict(img_for_pred, axes='SYXC')
            
            # Remove the batch and channel dimensions
            denoised_img = denoised[0, ..., 0]
            
            # Convert to uint8 for display
            if denoised_img.dtype != np.uint8:
                denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)
                
            output[i] = denoised_img
    finally:
        # Restore standard output
        sys.stdout.close()
        sys.stdout = original_stdout
    
    total_time = time.time() - start_time
    return output, total_time

def display_results(original, results_dict, slice_idx=None):
    """Display comparison of original and processed results"""
    if slice_idx is None:
        slice_idx = original.shape[0] // 2  # Middle slice
    
    num_results = len(results_dict) + 1  # +1 for original
    fig_width = 6 * num_results
    
    plt.figure(figsize=(fig_width, 6))
    
    # Display original
    plt.subplot(1, num_results, 1)
    plt.imshow(original[slice_idx], cmap='gray')
    plt.title('Original')
    plt.axis('off')
    
    # Display results
    for i, (name, result) in enumerate(results_dict.items(), 2):
        plt.subplot(1, num_results, i)
        plt.imshow(result[slice_idx], cmap='gray')
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('denoising_comparison.png')
    plt.show()

def run_denoising_comparison(data_path, max_slices=30, start_slice=120):
    """Run full denoising comparison with minimal output"""
    # Load the volume data quietly
    volume = load_volume(data_path, max_slices=max_slices, start_slice=start_slice)
    
    # Store results and times for comparison
    results = {}
    times = {}
    
    # Suppress output during processing
    import sys
    original_stdout = sys.stdout
    null_output = open(os.devnull, 'w')
    sys.stdout = null_output
    
    try:
        # Run the different denoising methods
        results['OpenCV NLMeans'], times['OpenCV NLMeans'] = process_cv2_denoise(volume)
        results['Gaussian'], times['Gaussian'] = process_gaussian(volume)
        results['BM3D'], times['BM3D'] = process_bm3d(volume)
        results['Wiener'], times['Wiener'] = process_wiener(volume)
        results['Crude 3D'], times['Crude 3D'] = process_crude_3d(volume)
        results['Noise2Void'], times['Noise2Void'] = process_noise2void(volume)
    finally:
        # Restore stdout
        sys.stdout = original_stdout
        null_output.close()
    
    # Display the results
    display_results(volume, results)
    
    # Calculate noise reduction statistics
    slice_idx = volume.shape[0] // 2  # Middle slice
    original_noise = np.std(volume[slice_idx])
    noise_reduction = {}
    
    for name, result in results.items():
        noise_after = np.std(result[slice_idx])
        reduction_percent = (1 - noise_after/original_noise) * 100
        noise_reduction[name] = reduction_percent
    
    # Print just the key information
    print(f"Volume shape: {volume.shape}")
    print("\nProcessing Time Comparison:")
    for name, time_value in times.items():
        print(f"{name}: {time_value:.2f} seconds")
    
    print("\nNoise Reduction Statistics:")
    for name, reduction in noise_reduction.items():
        print(f"{name}: {reduction:.1f}% noise reduction")
    
    # Create a zoomed crop view for clearer comparison
    center_slice = volume.shape[0] // 2
    h, w = volume[center_slice].shape
    center_y, center_x = h//2, w//2
    crop_size = 200
    
    # Create a new figure for the cropped view
    plt.figure(figsize=(20, 10))
    
    # Original cropped
    plt.subplot(1, len(results) + 1, 1)
    crop_original = volume[center_slice][center_y-crop_size//2:center_y+crop_size//2, 
                         center_x-crop_size//2:center_x+crop_size//2]
    plt.imshow(crop_original, cmap='gray')
    plt.title('Original (Center Crop)')
    plt.axis('off')
    
    # Processed cropped
    for i, (name, result) in enumerate(results.items(), 2):
        crop_result = result[center_slice][center_y-crop_size//2:center_y+crop_size//2, 
                            center_x-crop_size//2:center_x+crop_size//2]
        plt.subplot(1, len(results) + 1, i)
        plt.imshow(crop_result, cmap='gray')
        plt.title(f'{name} (Center Crop)')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('denoising_comparison_cropped.png')
    plt.show()
    
    return volume, results, times, noise_reduction


volume, results, times, noise_reduction = run_denoising_comparison(data_path)


def wiener_filter(img, kernel_size=5, noise_power=0.012):
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
    
    denoised_img = (denoised * 255).astype(np.uint8)
    
    return denoised_img

to_show={'00e047':169,
        '00e463':222,
        '1da097':34}

for tomo_id, Motor_axis_0 in to_show.items():
    input_image = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_' + tomo_id + '/slice_' + str(Motor_axis_0).zfill(4) + '.jpg'
    img = cv2.imread(input_image, cv2.IMREAD_GRAYSCALE)
    
    denoised_img = wiener_filter(img, kernel_size=11, noise_power=0.02)
    
    # Display full images results
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Noisy Image (tomo_{tomo_id}, slice_{Motor_axis_0})')
    axes[0].axis('off')
    
    axes[1].imshow(denoised_img, cmap='gray')
    axes[1].set_title('Denoised Image (Wiener Filter)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Create and display center crops for detailed comparison
    h, w = img.shape
    center_y, center_x = h//2, w//2
    crop_size = 200
    
    # Extract center crops
    crop_original = img[center_y-crop_size//2:center_y+crop_size//2, 
                        center_x-crop_size//2:center_x+crop_size//2]
    crop_denoised = denoised_img[center_y-crop_size//2:center_y+crop_size//2, 
                               center_x-crop_size//2:center_x+crop_size//2]
    
    # Display crops
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(crop_original, cmap='gray')
    axes[0].set_title('Original (Center Crop)')
    axes[0].axis('off')
    
    axes[1].imshow(crop_denoised, cmap='gray')
    axes[1].set_title('Denoised (Center Crop)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


from skimage import io, img_as_float, img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import bm3d

to_show={'00e047':169,
        '00e463':222,
        '1da097':34}

for tomo_id,Motor_axis_0 in to_show.items():
    input_image = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_' + tomo_id + '/slice_' + str(Motor_axis_0).zfill(4) + '.jpg'
    
    noisy_image = img_as_float(io.imread(input_image, as_gray=True))
    denoised_image = bm3d.bm3d(noisy_image, sigma_psd=0.15)
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(noisy_image, cmap='gray')
    axes[0].set_title('Noisy Image')
    axes[0].axis('off')
    
    axes[1].imshow(denoised_image, cmap='gray')
    axes[1].set_title('Denoised Image (BM3D)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Show a zoomed-in crop for better comparison
    h, w = noisy_image.shape
    center_y, center_x = h//2, w//2
    crop_size = 200
    
    # Extract center crops
    crop_original = noisy_image[center_y-crop_size//2:center_y+crop_size//2, 
                              center_x-crop_size//2:center_x+crop_size//2]
    crop_denoised = denoised_image[center_y-crop_size//2:center_y+crop_size//2, 
                                 center_x-crop_size//2:center_x+crop_size//2]
    
    # Display crops
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(crop_original, cmap='gray')
    axes[0].set_title('Original (Center Crop)')
    axes[0].axis('off')
    
    axes[1].imshow(crop_denoised, cmap='gray')
    axes[1].set_title('Denoised (Center Crop)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()


!pip install -q n2v tensorflow==2.13.0 csbdeep pillow tqdm
# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os
from n2v.models import N2V
from skimage import io

# Define the paths to pre-trained model weights and config
MODEL_PATH = '/kaggle/input/byu-denoising-cryo-et-with-noise2void/models/'

# Load the pre-trained Noise2Void model
model = N2V(config=None, name='n2v_cryoET_8slices', basedir=MODEL_PATH)

# List of tomogram IDs and specific slices to process
to_show = {
    '00e047': 169,
    '00e463': 222,
    '1da097': 34
}

# Process and display each example
for tomo_id, motor_axis_0 in to_show.items():
    # Construct input image path
    input_image = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_' + tomo_id + '/slice_' + str(motor_axis_0).zfill(4) + '.jpg'
    
    # Load image
    img = np.array(Image.open(input_image).convert('L'))
    
    # Add dimensions for N2V (SYXC format where S is sample/batch dimension)
    img_for_pred = img[np.newaxis, ..., np.newaxis]
    
    # Apply Noise2Void denoising
    denoised = model.predict(img_for_pred, axes='SYXC')
    
    # Remove the batch and channel dimensions
    denoised_img = denoised[0, ..., 0]
    
    # Convert to uint8 for display
    if denoised_img.dtype != np.uint8:
        denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)
    
    # Calculate noise reduction statistics
    noise_before = np.std(img)
    noise_after = np.std(denoised_img)
    reduction_percent = (1 - noise_after/noise_before) * 100
    
    # Display results
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))
    axes[0].imshow(img, cmap='gray')
    axes[0].set_title(f'Noisy Image (tomo_{tomo_id}, slice_{motor_axis_0})')
    axes[0].axis('off')
    
    axes[1].imshow(denoised_img, cmap='gray')
    axes[1].set_title(f'Denoised Image (Noise2Void) - {reduction_percent:.1f}% noise reduction')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Optional: Show a zoomed-in region for better comparison
    # Select center region (200x200 pixels)
    h, w = img.shape
    center_y, center_x = h//2, w//2
    crop_size = 200
    
    crop_original = img[center_y-crop_size//2:center_y+crop_size//2, 
                        center_x-crop_size//2:center_x+crop_size//2]
    crop_denoised = denoised_img[center_y-crop_size//2:center_y+crop_size//2, 
                               center_x-crop_size//2:center_x+crop_size//2]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(crop_original, cmap='gray')
    axes[0].set_title('Original (Center Crop)')
    axes[0].axis('off')
    
    axes[1].imshow(crop_denoised, cmap='gray')
    axes[1].set_title('Denoised (Center Crop)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

# Batch processing function for denoising multiple slices
def batch_process_n2v(input_dir, output_dir, slice_range=None):
    """
    Process a batch of slices with the pre-trained Noise2Void model
    
    Parameters:
        input_dir: Directory containing input JPEG slices
        output_dir: Directory to save denoised slices
        slice_range: Optional tuple (start, end) to process only a subset of slices
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all JPEG files in the input directory
    all_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.jpg')])
    
    # Apply slice range if specified
    if slice_range is not None:
        start, end = slice_range
        all_files = all_files[start:end]
    
    print(f"Processing {len(all_files)} slices...")
    
    # Process each slice
    for filename in all_files:
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, f"denoised_{filename}")
        
        # Load image
        img = np.array(Image.open(input_path).convert('L'))
        
        # Add dimensions for N2V (SYXC format)
        img_for_pred = img[np.newaxis, ..., np.newaxis]
        
        # Apply Noise2Void denoising
        denoised = model.predict(img_for_pred, axes='SYXC')
        
        # Remove batch and channel dimensions
        denoised_img = denoised[0, ..., 0]
        
        # Convert to uint8 for saving
        if denoised_img.dtype != np.uint8:
            denoised_img = np.clip(denoised_img, 0, 255).astype(np.uint8)
        
        # Save denoised image
        Image.fromarray(denoised_img).save(output_path)
    
    print(f"Batch processing complete. Denoised images saved to {output_dir}")

# Example usage of batch processing (commented out)
# tomo_id = '00e047'
# input_dir = f'/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_{tomo_id}'
# output_dir = f'/kaggle/working/denoised_tomo_{tomo_id}'
# batch_process_n2v(input_dir, output_dir, slice_range=(100, 200))

