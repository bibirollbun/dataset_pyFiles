# # This Python 3 environment comes with many helpful analytics libraries installed
# # It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# # For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# # Input data files are available in the read-only "../input/" directory
# # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install ultralytics


import pandas as pd
import numpy as np
import cv2 as cv
import os
import shutil
import time
import yaml
from pathlib import Path
from tqdm.notebook import tqdm  # Use tqdm.notebook for Jupyter/Kaggle environments
import random
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import glob

import torch
from matplotlib.patches import Rectangle
from ultralytics import YOLO
import yaml
import json

import warnings


EXTRA_DATA = '/kaggle/input/byu-2025-cryoet-dataset-part-2/dataset'
TRAIN = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train'
TEST = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test'

train_labels = pd.read_csv('/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv')
# extra_data_labels = pd.read_csv('/kaggle/input/byu-2025-cryoet-dataset-part-2/labels.csv')


# missing_samples = [sample for sample in extra_data_labels if sample not in os.listdir(EXTRA_DATA)]
# print(f"No of missing Samples in Cryoet dataset part-2: {len(missing_samples)}")


# Set random seed for reproducibility
np.random.seed(42)

# Define YOLO dataset structure
yolo_dataset_dir = "/kaggle/working/yolo_dataset"
yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
yolo_images_test = os.path.join(yolo_dataset_dir,"images","test")
yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
yolo_labels_test = os.path.join(yolo_dataset_dir,"labels","test")
yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

# Create directories
for dir_path in [yolo_images_train, yolo_images_val, yolo_images_test,
                 yolo_labels_train, yolo_labels_val, yolo_labels_test]:
    os.makedirs(dir_path, exist_ok=True)



# Define constants
TRUST = 3  # Number of slices above and below center slice (total 2*TRUST + 1 slices)
BOX_SIZE = 80  # Bounding box size for annotations (in pixels)
TRAIN_SPLIT = 0.7  # 80% for training, 20% for validation
TEST_SPLIT = 0.20

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



def prepare_yolo_dataset(labels_df,train_dir,trust=TRUST, train_split=TRAIN_SPLIT , test_split=TEST_SPLIT):
    """
    Extract slices containing motors from tomograms and save to YOLO structure with annotations
    """
    # Load the labels CSV
    # labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    
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
    test_split_idx = split_idx + int(len(unique_tomos) * test_split)
    train_tomos = unique_tomos[:split_idx]
    test_tomos = unique_tomos[split_idx:test_split_idx]
    val_tomos = unique_tomos[test_split_idx:]
    
    print(f"Split: {len(train_tomos)} tomograms for training, {len(val_tomos)} tomograms for validation, {len(test_tomos)}")
    
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
                     int(motor['Array shape (axis 0)']))
                )
        
        print(f"Will process approximately {len(motor_counts) * (2 * trust + 1)} slices for {set_name}")
        
        # Process each motor
        processed_slices = 0
        
        for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"Processing {set_name} motors"):
            # Calculate range of slices to include
            z_min = max(0, z_center - trust)
            z_max = min(z_max - 1, z_center + trust)
            
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
                normalized_img = normalize_slice(img_array)
                
                # Create destination filename (with unique identifier)
                dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)
                
                # Save the normalized image
                Image.fromarray(normalized_img).save(dest_path)
                
                # Get image dimensions
                img_width, img_height = img.size
                
                # Create YOLO format label
                # YOLO format: <class> <x_center> <y_center> <width> <height>
                # Values are normalized to [0, 1]
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                box_width_norm = BOX_SIZE / img_width
                box_height_norm = BOX_SIZE / img_height
                
                # Write label file
                label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                
                processed_slices += 1
        
        return processed_slices, len(motor_counts)
    
    # Process training tomograms
    train_slices, train_motors = process_tomogram_set(train_tomos, yolo_images_train, yolo_labels_train, "training")

    # Process test tomograms
    test_slices, test_motors = process_tomogram_set(test_tomos, yolo_images_test, yolo_labels_test, "test")
    # Process validation tomograms
    val_slices, val_motors = process_tomogram_set(val_tomos, yolo_images_val, yolo_labels_val, "validation")
    
    # Create YAML configuration file for YOLO
    yaml_content = {
        'path': yolo_dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'names': {0: 'motor'}
    }
    
    with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\nProcessing Summary:")
    print(f"- Train set: {len(train_tomos)} tomograms, {train_motors} motors, {train_slices} slices")
    print(f"- Test set: {len(test_tomos)} tomograms, {test_motors} motors, {test_slices} slices")
    print(f"- Validation set: {len(val_tomos)} tomograms, {val_motors} motors, {val_slices} slices")
    print(f"- Total: {len(train_tomos) + len(val_tomos) + len(test_tomos)} tomograms, {train_motors + val_motors + test_motors} motors, {train_slices + val_slices + test_slices} slices")
    
    # Return summary info
    return {
        "dataset_dir": yolo_dataset_dir,
        "yaml_path": os.path.join(yolo_dataset_dir, 'dataset.yaml'),
        "train_tomograms": len(train_tomos),
        "test_tomograms": len(test_tomos),
        "val_tomograms": len(val_tomos),
        "train_motors": train_motors,
        "test_motors": test_motors,
        "val_motors": val_motors,
        "train_slices": train_slices,
        "test_slices": test_slices,
        "val_slices": val_slices
    }


# Run the preprocessing
summary = prepare_yolo_dataset(labels_df=train_labels,train_dir=TRAIN,trust=TRUST)
print(f"\nPreprocessing Complete:")
print(f"- Training data: {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_slices']} slices")
print(f"- Testing data: {summary['test_tomograms']} tomograms, {summary['test_motors']} motors, {summary['test_slices']} slices")
print(f"- Validation data: {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_slices']} slices")
print(f"- Dataset directory: {summary['dataset_dir']}")
print(f"- YAML configuration: {summary['yaml_path']}")
print(f"\nReady for YOLO training!")


print(f"No of train samples: {len(os.listdir('/kaggle/working/yolo_dataset/images/train'))}")
print(f"No of test samples: {len(os.listdir('/kaggle/working/yolo_dataset/images/test'))}")
print(f"No of valid samples: {len(os.listdir('/kaggle/working/yolo_dataset/images/val'))}")


# print(extra_data_labels.columns)


# extra_data_tomo_id = os.listdir(os.path.join(EXTRA_DATA))
# print(len([_id for _id in extra_data_labels['tomo_id'] if _id not in extra_data_tomo_id]))
    


# extra_data_tomo_id = os.listdir(os.path.join(EXTRA_DATA))
# print(len([_id for _id in extra_data_labels['tomo_id'] if _id not in extra_data_tomo_id]))
    


# # Run the preprocessing
# summary = prepare_yolo_dataset(labels_df=extra_data_labels,train_dir=EXTRA_DATA,trust=TRUST)


# print(f"\nPreprocessing Complete:")
# print(f"- Training data: {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_slices']} slices")
# print(f"- Testing data: {summary['test_tomograms']} tomograms, {summary['test_motors']} motors, {summary['test_slices']} slices")
# print(f"- Validation data: {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_slices']} slices")
# print(f"- Dataset directory: {summary['dataset_dir']}")
# print(f"- YAML configuration: {summary['yaml_path']}")
# print(f"\nReady for YOLO training!")


# print(f"No of train samples: {len(os.listdir('/kaggle/working/yolo_dataset/images/train'))}")
# print(f"No of test samples: {len(os.listdir('/kaggle/working/yolo_dataset/images/test'))}")
# print(f"No of valid samples: {len(os.listdir('/kaggle/working/yolo_dataset/images/val'))}")


# Define base_dir - this was missing in the original code
# In Kaggle, we can use the working directory as base or remove it completely
# since we're using absolute paths
base_dir = "/kaggle/working"  # or simply use "" if using absolute paths

# Updated paths without concatenating with base_dir since they're already absolute
images_train_dir = "/kaggle/working/yolo_dataset/images/train/"
labels_train_dir = "/kaggle/working/yolo_dataset/labels/train"

# Box size for highlighting the motor
BOX_SIZE = 80

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
    rows = int(np.ceil(num_samples / 2))
    cols = min(num_samples, 2)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    
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



visualize_random_training_samples(4)


# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Define paths for Kaggle environment
yolo_dataset_dir = "/kaggle/working/yolo_dataset"
yolo_weights_dir = "/kaggle/working/yolo_weights"
# yolo_pretrained_weights = "yolov8n.pt"  # Path to pre-downloaded weights
# yolo_pretrained_weights = "/kaggle/input/yolo11/pytorch/default/1/yolo11n.pt"
# yolo_pretrained_weights = 'yolo11x.yaml'
yolo_pretrained_weights = 'yolo11n.yaml'
# Create weights directory if it doesn't exist
os.makedirs(yolo_weights_dir, exist_ok=True)

def fix_yaml_paths(yaml_path):
    """
    Fix the paths in the YAML file to match the actual Kaggle directories
    
    Args:
        yaml_path (str): Path to the original dataset YAML file
        
    Returns:
        str: Path to the fixed YAML file
    """
    print(f"Fixing YAML paths in {yaml_path}")
    
    # Read the original YAML
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    # Update paths to use actual dataset location
    if 'path' in yaml_data:
        yaml_data['path'] = yolo_dataset_dir
    
    # Create a new fixed YAML in the working directory
    fixed_yaml_path = "/kaggle/working/fixed_dataset.yaml"
    with open(fixed_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
    
    print(f"Created fixed YAML at {fixed_yaml_path} with path: {yaml_data.get('path')}")
    return fixed_yaml_path

def plot_dfl_loss_curve(run_dir):
    """
    Plot the DFL loss curves for train and validation, marking the best model
    
    Args:
        run_dir (str): Directory where the training results are stored
    """
    # Path to the results CSV file
    results_csv = os.path.join(run_dir, 'results.csv')
    
    if not os.path.exists(results_csv):
        print(f"Results file not found at {results_csv}")
        return
    
    # Read results CSV
    results_df = pd.read_csv(results_csv)
    
    # Check if DFL loss columns exist
    train_dfl_col = [col for col in results_df.columns if 'train/dfl_loss' in col]
    val_dfl_col = [col for col in results_df.columns if 'val/dfl_loss' in col]
    
    if not train_dfl_col or not val_dfl_col:
        print("DFL loss columns not found in results CSV")
        print(f"Available columns: {results_df.columns.tolist()}")
        return
    
    train_dfl_col = train_dfl_col[0]
    val_dfl_col = val_dfl_col[0]
    
    # Find the epoch with the best validation loss
    best_epoch = results_df[val_dfl_col].idxmin()
    best_val_loss = results_df.loc[best_epoch, val_dfl_col]
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot training and validation losses
    plt.plot(results_df['epoch'], results_df[train_dfl_col], label='Train DFL Loss')
    plt.plot(results_df['epoch'], results_df[val_dfl_col], label='Validation DFL Loss')
    
    # Mark the best model with a vertical line
    plt.axvline(x=results_df.loc[best_epoch, 'epoch'], color='r', linestyle='--', 
                label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])}, Val Loss: {best_val_loss:.4f})')
    
    # Add labels and legend
    plt.xlabel('Epoch')
    plt.ylabel('DFL Loss')
    plt.title('Training and Validation DFL Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Save the plot in the same directory as weights
    plot_path = os.path.join(run_dir, 'dfl_loss_curve.png')
    plt.savefig(plot_path)
    
    # Also save it to the working directory for easier access
    plt.savefig(os.path.join('/kaggle/working', 'dfl_loss_curve.png'))
    
    print(f"Loss curve saved to {plot_path}")
    plt.close()
    
    # Return the best epoch info
    return best_epoch, best_val_loss

# def train_yolo_model(yaml_path, pretrained_weights_path, epochs=30, batch_size=16, img_size=640,model_file = '/kaggle/input/custom-yolo/pytorch/custom_yolo11_6/1/custome_YOLOv11.yaml'):
#     """
#     Train a YOLO model on the prepared dataset
    
#     Args:
#         yaml_path (str): Path to the dataset YAML file
#         pretrained_weights_path (str): Path to pre-downloaded weights file
#         epochs (int): Number of training epochs
#         batch_size (int): Batch size for training
#         img_size (int): Image size for training
#     """
#     print(f"Loading pre-trained weights from: {pretrained_weights_path}")
    
#     # Load a pre-trained YOLOv8 model
#     # model = YOLO(pretrained_weights_path)
#     # model = YOLO(pretrained_weights_path) # Training Model from scratch
#     model = YOLO(model_file)
#     model.load(pretrained_weights_path,strict=False)
#     # Train the model with early stopping
#     results = model.train(
#         data=yaml_path,
#         epochs=epochs,
#         batch=batch_size,
#         imgsz=img_size,
#         project=yolo_weights_dir,
#         name='motor_detector',
#         exist_ok=True,
#         patience=5,              # Early stopping if no improvement for 5 epochs
#         save_period=5,           # Save checkpoints every 5 epochs
#         val=True,                # Ensure validation is performed
#         verbose=True             # Show detailed output during training
#     )
    
#     # Get the path to the run directory
#     run_dir = os.path.join(yolo_weights_dir, 'motor_detector')
    
#     # Plot and save the loss curve
#     best_epoch_info = plot_dfl_loss_curve(run_dir)
    
#     if best_epoch_info:
#         best_epoch, best_val_loss = best_epoch_info
#         print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")
    
#     return model, results
def train_yolo_model(yaml_path, pretrained_weights_path, epochs=30, batch_size=16, img_size=640, model_file='/kaggle/input/custom-yolo/pytorch/custom_yolo11_19/1/coustome_yolov11-2.yaml'):
    """
    Train a YOLO model on the prepared dataset
    
    Args:
        yaml_path (str): Path to the dataset YAML file
        pretrained_weights_path (str): Path to pre-downloaded weights file
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Image size for training
        model_file (str): Path to custom YOLO architecture YAML
    """
    print(f"Loading custom architecture from: {model_file}")
    print(f"Loading pre-trained weights from: {pretrained_weights_path}")
    
    # Create model from custom architecture
    model = YOLO(model_file)
    # model = YOLO("yolo11.pt")
    # Load pretrained weights if it's a .pt file
    # if pretrained_weights_path.endswith('.pt'):
    #     try:
    #         # Load weights with strict=False to allow partial loading
    #         import torch
    #         checkpoint = torch.load(pretrained_weights_path, map_location='cpu')
            
    #         # Try to load the state dict with strict=False
    #         if 'model' in checkpoint:
    #             model.model.load_state_dict(checkpoint['model'].float().state_dict(), strict=False)
    #         else:
    #             model.model.load_state_dict(checkpoint, strict=False)
            
    #         print("Successfully loaded pretrained weights (partial loading)")
    #     except Exception as e:
    #         print(f"Warning: Could not load pretrained weights: {e}")
    #         print("Training from scratch with custom architecture")
    # else:
    #     print(f"Pretrained weights path is a YAML file, training from scratch")
    
    # Train the model with early stopping
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=yolo_weights_dir,
        name='motor_detector',
        exist_ok=True,
        patience=10,              # Early stopping if no improvement for 5 epochs
        save_period=10,           # Save checkpoints every 5 epochs
        val=True,                # Ensure validation is performed
        verbose=True             # Show detailed output during training
    )
    
    # Get the path to the run directory
    run_dir = os.path.join(yolo_weights_dir, 'motor_detector')
    
    # Plot and save the loss curve
    best_epoch_info = plot_dfl_loss_curve(run_dir)
    
    if best_epoch_info:
        best_epoch, best_val_loss = best_epoch_info
        print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")
    
    return model, results
def predict_on_samples(model, num_samples=4):
    """
    Run predictions on random validation samples and display results
    
    Args:
        model: Trained YOLO model
        num_samples (int): Number of random samples to test
    """
    # Get validation images
    val_dir = os.path.join(yolo_dataset_dir, 'images', 'val')
    if not os.path.exists(val_dir):
        print(f"Validation directory not found at {val_dir}")
        # Try train directory instead if val doesn't exist
        val_dir = os.path.join(yolo_dataset_dir, 'images', 'train')
        print(f"Using train directory for predictions instead: {val_dir}")
        
    if not os.path.exists(val_dir):
        print("No images directory found for predictions")
        return
    
    val_images = os.listdir(val_dir)
    
    if len(val_images) == 0:
        print("No images found for prediction")
        return
    
    # Select random samples
    num_samples = min(num_samples, len(val_images))
    samples = random.sample(val_images, num_samples)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, img_file in enumerate(samples):
        if i >= len(axes):
            break
            
        img_path = os.path.join(val_dir, img_file)
        
        # Run prediction
        results = model.predict(img_path, conf=0.25)[0]
        
        # Load and display the image
        img = Image.open(img_path)
        axes[i].imshow(np.array(img), cmap='gray')
        
        # Draw ground truth box if available (from filename)
        try:
            # This assumes your filenames contain coordinates in a specific format
            parts = img_file.split('_')
            y_part = [p for p in parts if p.startswith('y')]
            x_part = [p for p in parts if p.startswith('x')]
            
            if y_part and x_part:
                y_gt = int(y_part[0][1:])
                x_gt = int(x_part[0][1:].split('.')[0])
                
                box_size = 24
                rect_gt = Rectangle((x_gt - box_size//2, y_gt - box_size//2), 
                              box_size, box_size, 
                              linewidth=1, edgecolor='g', facecolor='none')
                axes[i].add_patch(rect_gt)
        except:
            pass  # Skip ground truth if parsing fails
        
        # Draw predicted boxes (red)
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                rect_pred = Rectangle((x1, y1), x2-x1, y2-y1, 
                                     linewidth=1, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect_pred)
                axes[i].text(x1, y1-5, f'{conf:.2f}', color='red')
        
        axes[i].set_title(f"Image: {img_file}\nGround Truth (green) vs Prediction (red)")
    
    plt.tight_layout()
    
    # Save the predictions plot
    plt.savefig(os.path.join('/kaggle/working', 'predictions.png'))
    plt.show()

# Check and create a dataset YAML if needed
def prepare_dataset():
    """
    Check if dataset exists and create a proper YAML if needed
    
    Returns:
        str: Path to the YAML file to use for training
    """
    # Check if images exist
    train_images_dir = os.path.join(yolo_dataset_dir, 'images', 'train')
    val_images_dir = os.path.join(yolo_dataset_dir, 'images', 'val')
    train_labels_dir = os.path.join(yolo_dataset_dir, 'labels', 'train')
    val_labels_dir = os.path.join(yolo_dataset_dir, 'labels', 'val')
    
    # Print directory existence status
    print(f"Directory status:")
    print(f"- Train images dir exists: {os.path.exists(train_images_dir)}")
    print(f"- Val images dir exists: {os.path.exists(val_images_dir)}")
    print(f"- Train labels dir exists: {os.path.exists(train_labels_dir)}")
    print(f"- Val labels dir exists: {os.path.exists(val_labels_dir)}")
    
    # Check for original YAML file
    original_yaml_path = os.path.join(yolo_dataset_dir, 'dataset.yaml')
    
    if os.path.exists(original_yaml_path):
        print(f"Found original dataset.yaml at {original_yaml_path}")
        # Fix the paths in the YAML
        return fix_yaml_paths(original_yaml_path)
    else:
        print(f"Original dataset.yaml not found, creating a new one")
        
        # Create a new YAML file
        yaml_data = {
            'path': yolo_dataset_dir,
            'train': 'images/train',
            'val': 'images/train' if not os.path.exists(val_images_dir) else 'images/val',
            'names': {0: 'motor'}
        }
        
        new_yaml_path = "/kaggle/working/dataset.yaml"
        with open(new_yaml_path, 'w') as f:
            yaml.dump(yaml_data, f)
            
        print(f"Created new YAML at {new_yaml_path}")
        return new_yaml_path



print("Starting YOLO training process...")

# Prepare dataset and get YAML path
yaml_path = prepare_dataset()
print(f"Using YAML file: {yaml_path}")

# Print YAML file contents
with open(yaml_path, 'r') as f:
    yaml_content = f.read()
print(f"YAML file contents:\n{yaml_content}")

# Train model
print("\nStarting YOLO training...")
model, results = train_yolo_model(
    yaml_path,
    pretrained_weights_path=yolo_pretrained_weights,
    epochs=80  # Using 30 epochs instead of 100 for faster training
)

print("\nTraining complete!")

# Run predictions
print("\nRunning predictions on sample images...")
predict_on_samples(model, num_samples=4)



metrics = model.val(data='/kaggle/working/yolo_dataset/dataset.yaml', split='test')


# Print metrics
print(f"Precision: {metrics.box.p[0]:.2f}")
print(f"Recall: {metrics.box.r[0]:.2f}")
print(f"mAP@0.5: {metrics.box.map50:.3f}")
print(f"mAP@0.5:0.95: {metrics.box.map:.3f}")
print(f"F_1 Score: {2*(metrics.box.p[0]*metrics.box.r[0])/(metrics.box.p[0]+ metrics.box.r[0]):.2f}")


from IPython.display import Image
Image(filename='/kaggle/working/yolo_weights/motor_detector/results.png')




