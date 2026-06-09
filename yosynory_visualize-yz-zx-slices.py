import os
import numpy as np
import pandas as pd
from PIL import Image
import shutil
import time
import yaml
from pathlib import Path
from tqdm.notebook import tqdm  # Use tqdm.notebook for Jupyter/Kaggle environments

import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)

# Define Kaggle paths
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
train_dir = os.path.join(data_path, "train")

# # Define YOLO dataset structure
# yolo_dataset_dir = "/kaggle/working/yolo_dataset"
# yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
# yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
# yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
# yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

# # Create directories
# for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
#     os.makedirs(dir_path, exist_ok=True)

# # Define constants
# TRUST = 4  # Number of slices above and below center slice (total 2*TRUST + 1 slices)
# BOX_SIZE = 24  # Bounding box size for annotations (in pixels)
# TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation


# Image processing functions
def normalize_slice(slice_data):
    """
    Normalize slice data using 2nd and 98th percentiles
    """
    # Calculate percentiles
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    
    # Clip the data to the percentile range
    clipped_data = np.clip(slice_data, p2, p98)
    
    # Normalize to [0, 255] range
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    
    return np.uint8(normalized)


# Load the label CSV
labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))

# Extract tomograms that have motors
tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
unique_tomos = tomo_df['tomo_id'].unique()
num_unique_tomos = len(unique_tomos)

# Loop through and visualize slices for each tomogram (from 0 to 361)
for num in range(num_unique_tomos):
    src_path = os.path.join(train_dir, unique_tomos[num])
    file_list = [f for f in os.listdir(src_path) if os.path.isfile(os.path.join(src_path, f))]
    file_list = sorted(file_list, key=lambda x: int(x.split('_')[1].split('.')[0]))

    # Get the motor labels for this tomogram
    tomo_motors = labels_df[labels_df['tomo_id'] == unique_tomos[num]]
    motor_counts = []
    for _, motor in tomo_motors.iterrows():
        if pd.isna(motor['Motor axis 0']):
            continue
        motor_counts.append(
            (unique_tomos[num], 
             int(motor['Motor axis 0']), 
             int(motor['Motor axis 1']), 
             int(motor['Motor axis 2']),
             int(motor['Array shape (axis 0)']),
             int(motor['Voxel spacing']))
        )

    z_dim = len(file_list)  # Z-axis = number of image slices

    # Get the size from the first image
    sample_img = Image.open(os.path.join(src_path, file_list[0]))
    y_dim, x_dim = np.array(sample_img).shape

    # Initialize 3D array
    stacked_images = np.zeros((z_dim, y_dim, x_dim), dtype=np.float32)

    # Load and stack images one by one (no normalization)
    for z, file in enumerate(file_list):
        file_path = os.path.join(src_path, file)
        img = Image.open(file_path)
        img_array = np.array(img, dtype=np.float32)
        stacked_images[z] = img_array

    # ---- Convert slice image to RGB and draw red lines ----
    def add_red_lines(image, horizontal=None, vertical=None):
        """ Convert grayscale image to RGB and add red lines at specified positions """
        rgb_img = np.stack([image] * 3, axis=-1)  # Convert to RGB
        rgb_img = (rgb_img / rgb_img.max() * 255).astype(np.uint8)  # Scale to 0–255

        if horizontal is not None:
            rgb_img[horizontal, :, :] = [255, 0, 0]  # Horizontal red line (Y-axis)
        if vertical is not None:
            rgb_img[:, vertical, :] = [255, 0, 0]  # Vertical red line (X-axis)

        return rgb_img

    # Plot slices based on each motor location
    plt.figure(figsize=(15, len(motor_counts) * 5))

    for idx, motor in enumerate(motor_counts):
        middle_x = motor[3]
        middle_y = motor[2]
        middle_z = motor[1]

        # Extract XY, YZ, ZX slices
        xy_slice = normalize_slice(stacked_images[middle_z, :, :])  # XY slice (Z-axis)
        yz_slice = normalize_slice(stacked_images[:, :, middle_x])  # YZ slice (X-axis)
        zx_slice = normalize_slice(stacked_images[:, middle_y, :])  # ZX slice (Y-axis)

        # Draw slice positions on each plane
        xy_rgb = add_red_lines(xy_slice, horizontal=middle_y, vertical=middle_x)
        yz_rgb = add_red_lines(yz_slice, horizontal=middle_z, vertical=middle_y)
        zx_rgb = add_red_lines(zx_slice, horizontal=middle_z, vertical=middle_x)

        # ---- Display slice images for each motor ----
        plt.subplot(len(motor_counts), 3, idx * 3 + 1)
        plt.title(f"No.{num}, File {unique_tomos[num]}, (Voxel {motor[5]}), Motor {idx} - XY Slice")
        plt.imshow(xy_rgb)
        plt.axis('off')

        plt.subplot(len(motor_counts), 3, idx * 3 + 2)
        plt.title(f"YZ Slice")
        plt.imshow(yz_rgb)
        plt.axis('off')

        plt.subplot(len(motor_counts), 3, idx * 3 + 3)
        plt.title(f"ZX Slice")
        plt.imshow(zx_rgb)
        plt.axis('off')

    plt.tight_layout()
    plt.show()

