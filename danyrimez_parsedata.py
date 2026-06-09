import os
import numpy as np
import pandas as pd
from PIL import Image
import shutil
import time
import yaml
from pathlib import Path
from tqdm.notebook import tqdm  # Use tqdm.notebook for Jupyter/Kaggle environments

# Set random seed for reproducibility
np.random.seed(42)

# Define Kaggle paths
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
train_dir = os.path.join(data_path, "train")

# Define YOLO dataset structure
yolo_dataset_dir = "/kaggle/working/yolo_dataset"
yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

# Create directories
for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
    os.makedirs(dir_path, exist_ok=True)

# Define constants
TRUST = 24  # Number of slices above and below center slice (total 2*TRUST + 1 slices)
BOX_SIZE = 24  # Bounding box size for annotations (in pixels)
TRAIN_SPLIT = 0.8  # 80% for training, 20% for validation

# Image processing functions
def normalize_slice(slice_data):
    """
    Normalize slice data using 2nd and 98th percentiles
    """
    """
    # Calculate percentiles
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    
    # Clip the data to the percentile range
    clipped_data = np.clip(slice_data, p2, p98)
    
    # Normalize to [0, 255] range
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    """
    return np.uint8(255*(slice_data-0.5)/0.5)

def prepare_yolo_dataset(trust=TRUST, train_split=TRAIN_SPLIT):
    """
    Extract slices containing motors from tomograms and save to YOLO structure with annotations
    """
    # Load the labels CSV
    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    
    # Count total number of motors
    total_motors = labels_df['Number of motors'].sum()
    print(f"Total number of motors in the dataset: {total_motors}")
    
    # Get unique tomograms that have motors
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = tomo_df['tomo_id'].unique()
    
    print(f"Found {len(unique_tomos)} unique tomograms with motors")
    
    # Perform the train-val split at the tomogram level (not motor level)
    # This ensures all slices from a single tomogram go to either train or val
    np.random.shuffle(unique_tomos)  # Shuffle the tomograms
    split_idx = int(len(unique_tomos) * train_split)
    train_tomos = unique_tomos[:split_idx]
    val_tomos = unique_tomos[split_idx:]
    
    print(f"Split: {len(train_tomos)} tomograms for training, {len(val_tomos)} tomograms for validation")
    
    # Function to process a set of tomograms
    def process_tomogram_set(tomogram_ids, images_dir, labels_dir, set_name):
        motor_counts = []
        for tomo_id in tomogram_ids:
            # Get all motors for this tomogram
            tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
            for _, motor in tomo_motors.iterrows():
                if pd.isna(motor['Motor axis 0']):
                    continue
                motor_counts.append(
                    (tomo_id, 
                     int(motor['Motor axis 0']), 
                     int(motor['Motor axis 1']), 
                     int(motor['Motor axis 2']),
                     int(motor['Array shape (axis 0)']),
                     int(motor['Voxel spacing']))
                )
        
        # print(f"Will process approximately {len(motor_counts) * (2 * trust + 1)} slices for {set_name}")
        
        # Process each motor
        processed_slices = 0
        
        for tomo_id, z_center, y_center, x_center, z_max, voxelsz in tqdm(motor_counts, desc=f"Processing {set_name} motors"):
            
            BOX_SIZE = 400/voxelsz
            # trust = int((BOX_SIZE +1)//2) # BOX_SIZE +16 slices along z (rounded up)
            
            # Calculate range of slices to include => always 20
            z_center = np.round(z_center)
            z_min = max(0, z_center - 5)
            Z = min(z_max - 1, z_center + 5)
            if Z - z_min <10:
                if z_min==0:
                    Z = 10 
                elif Z==z_max-1:
                    z_min = Z - 10
            z_max = Z
            if z_max - z_min != 10:
                print("Weird : ", z_min, z_max)
            trust = int(0.5*BOX_SIZE)  #
            
            # Process each slice in the range
            for z in range(z_min, z_max + 1):
                # Create slice filename
                slice_filename = f"slice_{z:04d}.jpg"
                
                # Source path for the slice
                src_path = os.path.join(train_dir, tomo_id, slice_filename)
                
                if not os.path.exists(src_path):
                    print(f"Warning: {src_path} does not exist, skipping.")
                    continue
                
                # Load and normalize the slice
                img = Image.open(src_path)
                img_array = np.array(img)
                
                # Normalize the image
                normalized_img = img_array # normalize_slice(img_array)
                shape = 640 ; shape2 = shape//2
                x1, x2 = max(0, x_center - shape2), min(normalized_img.shape[1], x_center +shape2)
                y1, y2 = max(0, y_center - shape2), min(normalized_img.shape[0], y_center +shape2)
                # cropped_image = 0.5*np.ones_like(normalized_img[:shape,:shape]) 
                # cropped_image[:x2-x1,:y2-y1] = normalized_img[x1:x2, y1:y2] 
                normalized_img = normalized_img[y1:y2, x1:x2]
                
                # Create destination filename (with unique identifier)
                dest_filename = f"{tomo_id}_zc{z_center:04d}_yc{y_center:04d}_xc{x_center:04d}_{z}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)
                
                # Save the normalized image
                Image.fromarray(normalized_img).save(dest_path)

                if z<z_center + trust and z>z_center - trust:
                    # Get image dimensions
                    img_width, img_height = x2-x1, y2-y1
                    
                    # Create YOLO format label
                    # YOLO format: <class> <x_center> <y_center> <width> <height>
                    # Values are normalized to [0, 1]
                    x_center_norm = (x_center - x1) / img_width
                    y_center_norm = (y_center - y1) / img_height
                    box_width_norm = BOX_SIZE / img_width
                    box_height_norm = BOX_SIZE / img_height 
                    # Write label file
                    label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                    with open(label_path, 'w') as f:
                        # dist = abs(z - z_center)
                        f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                else:
                    # Write label file
                    label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                    with open(label_path, 'w') as f:
                        f.write(f"\n")
                
                processed_slices += 1
                
        return processed_slices, len(motor_counts)
    
    # Process training tomograms
    train_slices, train_motors = process_tomogram_set(train_tomos, yolo_images_train, yolo_labels_train, "training")
    
    # Process validation tomograms
    val_slices, val_motors = process_tomogram_set(val_tomos, yolo_images_val, yolo_labels_val, "validation")
    
    # Create YAML configuration file for YOLO
    yaml_content = {
        'path': yolo_dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'motor'} # {0: 'near', 1: 'far'}
    }
    
    with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\nProcessing Summary:")
    print(f"- Train set: {len(train_tomos)} tomograms, {train_motors} motors, {train_slices} slices")
    print(f"- Validation set: {len(val_tomos)} tomograms, {val_motors} motors, {val_slices} slices")
    print(f"- Total: {len(train_tomos) + len(val_tomos)} tomograms, {train_motors + val_motors} motors, {train_slices + val_slices} slices")
    
    # Return summary info
    return {
        "dataset_dir": yolo_dataset_dir,
        "yaml_path": os.path.join(yolo_dataset_dir, 'dataset.yaml'),
        "train_tomograms": len(train_tomos),
        "val_tomograms": len(val_tomos),
        "train_motors": train_motors,
        "val_motors": val_motors,
        "train_slices": train_slices,
        "val_slices": val_slices
    }

# Run the preprocessing
summary = prepare_yolo_dataset(TRUST)
print(f"\nPreprocessing Complete:")
print(f"- Training data: {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_slices']} slices")
print(f"- Validation data: {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_slices']} slices")
print(f"- Dataset directory: {summary['dataset_dir']}")
print(f"- YAML configuration: {summary['yaml_path']}")
print(f"\nReady for YOLO training!")


import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import os
import numpy as np
import glob

# Define base_dir - this was missing in the original code
# In Kaggle, we can use the working directory as base or remove it completely
# since we're using absolute paths
base_dir = "/kaggle/working"  # or simply use "" if using absolute paths

# Updated paths without concatenating with base_dir since they're already absolute
images_train_dir = yolo_dataset_dir + "/images/train/"
labels_train_dir = yolo_dataset_dir + "/labels/train/"

# Box size for highlighting the motor
BOX_SIZE = 24

def visualize_random_training_samples(num_samples=4):
    """
    Visualize random training samples with YOLO annotations
    
    Args:
        num_samples (int): Number of random images to display
    """
    # Get all image files from the train directory
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(images_train_dir, "**", ext), recursive=True))
    
    # Make sure we have enough images
    if len(image_files) == 0:
        print("No image files found in the train directory!")
        return
        
    num_samples = min(num_samples, len(image_files))
    
    # Select random images
    random_images = random.sample(image_files, num_samples)
    
    # Create a figure with subplots
    rows = int(np.ceil(num_samples / 5))
    cols = 5
    fig, axes = plt.subplots(rows, cols, figsize=(14, 3 * rows))
    
    # Handle the case of a single subplot
    if num_samples == 1:
        axes = np.array([axes])
    
    # Flatten axes array for easy indexing
    axes = axes.flatten()
    
    # Process each selected image
    for i, img_path in enumerate(random_images):
        try:
            # Get corresponding label file
            # YOLO labels have same name but .txt extension instead of image extension
            relative_path = os.path.relpath(img_path, images_train_dir)
            label_path = os.path.join(labels_train_dir, os.path.splitext(relative_path)[0] + '.txt')
            
            # Load the image
            img = Image.open(img_path)
            img_width, img_height = img.size
            
            # Normalize image using percentiles for better visualization
            img_array = np.array(img)
            p2 = np.percentile(img_array, 2)
            p98 = np.percentile(img_array, 98)
            normalized = np.clip(img_array, p2, p98)
            normalized = 255 * (normalized - p2) / (p98 - p2)
            img_normalized = Image.fromarray(np.uint8(normalized))
            
            # Convert image to RGB for colored box
            img_rgb = img_normalized.convert('RGB')
            
            # Create a transparent overlay
            overlay = Image.new('RGBA', img_rgb.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Load YOLO format annotations if they exist
            annotations = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        # YOLO format: class x_center y_center width height
                        # All values are normalized from 0 to 1
                        values = line.strip().split()
                        class_id = int(values[0])
                        x_center = float(values[1]) * img_width
                        y_center = float(values[2]) * img_height
                        width = float(values[3]) * img_width
                        height = float(values[4]) * img_height
                        
                        annotations.append({
                            'class_id': class_id,
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
            
            # Draw all annotations
            for ann in annotations:
                x_center = ann['x_center']
                y_center = ann['y_center']
                width = ann['width']
                height = ann['height']
                
                # Calculate bounding box coordinates
                x1 = max(0, int(x_center - width/2))
                y1 = max(0, int(y_center - height/2))
                x2 = min(img_width, int(x_center + width/2))
                y2 = min(img_height, int(y_center + height/2))
                
                # Draw semi-transparent red rectangle
                draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 64), outline=(255, 0, 0, 200))
                
                # Draw label
                label_text = f"Class {ann['class_id']}"
                draw.text((x1, y1-10), label_text, fill=(255, 0, 0, 255))
            
            # If no annotations found, indicate this
            if not annotations:
                draw.text((10, 10), "No annotations found", fill=(255, 0, 0, 255))
            
            # Composite the overlay onto the original image
            img_rgb = Image.alpha_composite(img_rgb.convert('RGBA'), overlay).convert('RGB')
            
            # Display the image with annotations
            axes[i].imshow(np.array(img_rgb))
            img_name = os.path.basename(img_path)
            axes[i].set_title(f"Image: {img_name}\nAnnotations: {len(annotations)}")
            axes[i].axis('on')
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading image: {os.path.basename(img_path)}", 
                       horizontalalignment='center', verticalalignment='center')
            axes[i].axis('off')
    
    # Handle extra subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"Displayed {num_samples} random images with YOLO annotations")

# Run the visualization
visualize_random_training_samples(15)

