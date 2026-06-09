!tar xfvz /kaggle/input/k/mathieuduverne/ultralytics-for-offline-install/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages


import plotly.express as px
from PIL import Image, ImageDraw
import random
import seaborn as sns
from matplotlib.patches import Rectangle
from ultralytics import YOLO
import yaml
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import cv2
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
import math


# Define global constants for dataset directories
DATA_DIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
TRAIN_CSV = os.path.join(DATA_DIR, 'train_labels.csv')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = './'
MODEL_DIR = './models'

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Set device: Use GPU if available; otherwise, fall back to CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Set random seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True



# Define YOLO dataset structure and parameters
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
train_dir = os.path.join(data_path, "train")

# Output directories for YOLO dataset (adjust as needed)
yolo_dataset_dir = "/kaggle/working/yolo_dataset"
yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

# Create necessary directories
for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
    os.makedirs(dir_path, exist_ok=True)

# Define constants for processing
TRUST = 4       # Number of slices above and below center slice (total slices = 2*TRUST + 1)
BOX_SIZE = 24   # Bounding box size (in pixels)
TRAIN_SPLIT = 0.8  # 80% training, 20% validation

# Define a helper function for image normalization using percentile-based contrast enhancement.
def normalize_slice(slice_data):
    """
    Normalize slice data using the 2nd and 98th percentiles.
    
    Args:
        slice_data (numpy.array): Input image slice.
    
    Returns:
        np.uint8: Normalized image in the range [0, 255].
    """
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)

# Define the preprocessing function to extract slices, normalize, and generate YOLO annotations.
def prepare_yolo_dataset(trust=TRUST, train_split=TRAIN_SPLIT):
    """
    Extract slices containing motors and save images with corresponding YOLO annotations.
    
    Steps:
    - Load the motor labels.
    - Perform a train/validation split by tomogram.
    - For each motor, extract slices in a range (± trust parameter).
    - Normalize each slice and save it.
    - Generate YOLO format bounding box annotations with a fixed box size.
    - Create a YAML configuration file for YOLO training.
    
    Returns:
        dict: A summary containing dataset statistics and file paths.
    """
    # Load the labels CSV
    labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    
    total_motors = labels_df['Number of motors'].sum()
    print(f"Total number of motors in the dataset: {total_motors}")
    
    # Consider only tomograms with at least one motor
    tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
    unique_tomos = tomo_df['tomo_id'].unique()
    print(f"Found {len(unique_tomos)} unique tomograms with motors")
    
    # Shuffle and split tomograms into train and validation sets
    np.random.shuffle(unique_tomos)
    split_idx = int(len(unique_tomos) * train_split)
    train_tomos = unique_tomos[:split_idx]
    val_tomos = unique_tomos[split_idx:]
    print(f"Split: {len(train_tomos)} tomograms for training, {len(val_tomos)} tomograms for validation")
    
    # Helper function to process a list of tomograms
    def process_tomogram_set(tomogram_ids, images_dir, labels_dir, set_name):
        motor_counts = []
        for tomo_id in tomogram_ids:
            # Get motor annotations for the current tomogram
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
        processed_slices = 0
        
        # Loop over each motor annotation
        for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"Processing {set_name} motors"):
            z_min = max(0, z_center - trust)
            z_max_bound = min(z_max - 1, z_center + trust)
            for z in range(z_min, z_max_bound + 1):
                # Create the slice filename and source path
                slice_filename = f"slice_{z:04d}.jpg"
                src_path = os.path.join(train_dir, tomo_id, slice_filename)
                if not os.path.exists(src_path):
                    print(f"Warning: {src_path} does not exist, skipping.")
                    continue
                
                # Load, normalize, and save the image slice
                img = Image.open(src_path)
                img_array = np.array(img)
                normalized_img = normalize_slice(img_array)
                dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)
                Image.fromarray(normalized_img).save(dest_path)
                
                # Prepare YOLO bounding box annotation (normalized values)
                img_width, img_height = img.size
                x_center_norm = x_center / img_width
                y_center_norm = y_center / img_height
                box_width_norm = BOX_SIZE / img_width
                box_height_norm = BOX_SIZE / img_height
                label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
                with open(label_path, 'w') as f:
                    f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                
                processed_slices += 1
        
        return processed_slices, len(motor_counts)
    
    # Process training tomograms
    train_slices, train_motors = process_tomogram_set(train_tomos, yolo_images_train, yolo_labels_train, "training")
    # Process validation tomograms
    val_slices, val_motors = process_tomogram_set(val_tomos, yolo_images_val, yolo_labels_val, "validation")
    
    # Generate YAML configuration for YOLO training
    yaml_content = {
        'path': yolo_dataset_dir,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'motor'}
    }
    with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"\nProcessing Summary:")
    print(f"- Train set: {len(train_tomos)} tomograms, {train_motors} motors, {train_slices} slices")
    print(f"- Validation set: {len(val_tomos)} tomograms, {val_motors} motors, {val_slices} slices")
    print(f"- Total: {len(train_tomos) + len(val_tomos)} tomograms, {train_motors + val_motors} motors, {train_slices + val_slices} slices")
    
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
print("\nReady for YOLO training!")


# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Define paths for the Kaggle environment
yolo_dataset_dir = "/kaggle/working/yolo_dataset"
yolo_weights_dir = "/kaggle/working/yolo_weights"
yolo_pretrained_weights = "/kaggle/input/k/mathieuduverne/ultralytics-for-offline-install/yolov8n.pt"  # Pre-downloaded weights

# Create the weights directory if it does not exist
os.makedirs(yolo_weights_dir, exist_ok=True)


def fix_yaml_paths(yaml_path):
    """
    Fix the paths in the YAML file to match the actual Kaggle directories.
    
    Args:
        yaml_path (str): Path to the original dataset YAML file.
        
    Returns:
        str: Path to the fixed YAML file.
    """
    print(f"Fixing YAML paths in {yaml_path}")
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.safe_load(f)
    
    if 'path' in yaml_data:
        yaml_data['path'] = yolo_dataset_dir
    
    fixed_yaml_path = "/kaggle/working/fixed_dataset.yaml"
    with open(fixed_yaml_path, 'w') as f:
        yaml.dump(yaml_data, f)
    
    print(f"Created fixed YAML at {fixed_yaml_path} with path: {yaml_data.get('path')}")
    return fixed_yaml_path


def plot_dfl_loss_curve(run_dir):
    """
    Plot the DFL loss curves for training and validation, marking the best model.
    
    Args:
        run_dir (str): Directory where the training results are stored.
    """
    results_csv = os.path.join(run_dir, 'results.csv')
    if not os.path.exists(results_csv):
        print(f"Results file not found at {results_csv}")
        return
    
    results_df = pd.read_csv(results_csv)
    train_dfl_col = [col for col in results_df.columns if 'train/dfl_loss' in col]
    val_dfl_col = [col for col in results_df.columns if 'val/dfl_loss' in col]
    
    if not train_dfl_col or not val_dfl_col:
        print("DFL loss columns not found in results CSV")
        print(f"Available columns: {results_df.columns.tolist()}")
        return
    
    train_dfl_col = train_dfl_col[0]
    val_dfl_col = val_dfl_col[0]
    
    best_epoch = results_df[val_dfl_col].idxmin()
    best_val_loss = results_df.loc[best_epoch, val_dfl_col]
    
    plt.figure(figsize=(10, 6))
    plt.plot(results_df['epoch'], results_df[train_dfl_col], label='Train DFL Loss')
    plt.plot(results_df['epoch'], results_df[val_dfl_col], label='Validation DFL Loss')
    plt.axvline(x=results_df.loc[best_epoch, 'epoch'], color='r', linestyle='--', 
                label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])}, Val Loss: {best_val_loss:.4f})')
    plt.xlabel('Epoch')
    plt.ylabel('DFL Loss')
    plt.title('Training and Validation DFL Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plot_path = os.path.join(run_dir, 'dfl_loss_curve.png')
    plt.savefig(plot_path)
    plt.savefig(os.path.join('/kaggle/working', 'dfl_loss_curve.png'))
    
    print(f"Loss curve saved to {plot_path}")
    plt.close()
    
    return best_epoch, best_val_loss


def train_yolo_model(yaml_path, pretrained_weights_path, epochs=50, img_size=640):
    """
    Train a YOLO model with stabilized hyperparameters to reduce loss fluctuations.

    Args:
        yaml_path (str): Path to the dataset YAML file.
        pretrained_weights_path (str): Path to pre-downloaded weights file.
        epochs (int): Number of training epochs.
        img_size (int): Image size for training.

    Returns:
        model (YOLO): Trained YOLO model.
        results: Training results.
    """
    print(f"Loading pre-trained weights from: {pretrained_weights_path}")
    model = YOLO(pretrained_weights_path)

    results = model.train(
        data=yaml_path,
        epochs=epochs,
        imgsz=img_size,
        project=yolo_weights_dir,
        name='motor_detector_stable',
        exist_ok=True,
        
        # Batch and hardware settings
        batch=16,                # Keep original batch size or increase if memory allows
        workers=4,               # Data loading workers
        device=0,                # Use GPU 0
        amp=True,                # Mixed precision for efficiency
        
        # Learning rate and optimizer settings
        optimizer="AdamW",       # Using AdamW optimizer
        lr0=0.0005,              # Lower initial learning rate for stability
        lrf=0.05,                # Final learning rate factor (higher than before)
        momentum=0.9,            # Slightly lower momentum
        weight_decay=0.001,      # Increased weight decay for regularization
        warmup_epochs=3,         # Add warmup period
        warmup_momentum=0.8,     # Warmup momentum
        warmup_bias_lr=0.1,      # Warmup bias learning rate
        
        # Loss function weights
        box=7.5,                 # Higher box loss weight
        cls=0.5,                 # Lower classification loss weight 
        dfl=1.5,                 # Adjusted DFL loss weight
        
        # Augmentation parameters  
        mosaic=0.7,              # Reduce mosaic probability
        mixup=0.15,              # Slightly reduced mixup
        copy_paste=0.1,          # Add copy-paste augmentation
        degrees=0.0,             # No rotation (specific to this task)
        translate=0.1,           # Add translation augmentation
        scale=0.5,               # Scale augmentation
        shear=0.0,               # No shear (could destabilize training)
        fliplr=0.5,              # Horizontal flip probability
        flipud=0.0,              # No vertical flip (task specific)
        hsv_h=0.015,             # Hue augmentation
        hsv_s=0.5,               # Saturation augmentation
        hsv_v=0.3,               # Value/brightness augmentation
        perspective=0.0,         # No perspective shift for this task
        
        # Stability and convergence parameters
        patience=15,             # Increase patience from 10 to 15
        save_period=5,           # Save model every 5 epochs
        close_mosaic=15,         # Disable mosaic augmentation later in training
        overlap_mask=True,       # Overlap mask for better boundary detection
        
        # Model specific settings
        dropout=0.1,             # Add dropout for regularization
        val=True,                # Keep validation on
        rect=False,              # Don't use rectangular training
        multi_scale=True,        # Add multi-scale training for robustness
    )

    run_dir = os.path.join(yolo_weights_dir, 'motor_detector_stable')
    
    # Plot loss curves
    if 'plot_dfl_loss_curve' in globals():
        best_epoch_info = plot_dfl_loss_curve(run_dir)
        if best_epoch_info:
            best_epoch, best_val_loss = best_epoch_info
            print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")

    return model, results


def plot_learning_rate_schedule(epochs=50, lr0=0.0005, lrf=0.05, warmup_epochs=3):
    """
    Visualize the learning rate schedule with warmup and cosine decay.
    
    Args:
        epochs (int): Total number of epochs
        lr0 (float): Initial learning rate
        lrf (float): Final learning rate factor
        warmup_epochs (int): Number of warmup epochs
    """
    epochs_array = np.arange(epochs)
    lr_array = np.zeros(epochs)
    
    # Warmup phase
    for i in range(warmup_epochs):
        lr_array[i] = lr0 * ((i + 1) / warmup_epochs)
    
    # Cosine decay phase
    for i in range(warmup_epochs, epochs):
        lr_array[i] = lr0 * (((1 - math.cos(math.pi * (i - warmup_epochs) / (epochs - warmup_epochs))) / 2) * (lrf - 1) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_array, lr_array)
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule with Warmup and Cosine Decay')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(os.path.join('/kaggle/working', 'lr_schedule.png'))
    plt.show()
    
    return lr_array


def analyze_training_stability(results_csv):
    """
    Analyze and visualize training stability from results CSV.
    
    Args:
        results_csv (str): Path to the results CSV file.
    """
    results_df = pd.read_csv(results_csv)
    
    # Set up figure
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: All losses
    axs[0, 0].plot(results_df['epoch'], results_df['train/box_loss'], label='Train Box Loss')
    axs[0, 0].plot(results_df['epoch'], results_df['val/box_loss'], label='Val Box Loss')
    axs[0, 0].plot(results_df['epoch'], results_df['train/cls_loss'], label='Train Cls Loss')
    axs[0, 0].plot(results_df['epoch'], results_df['val/cls_loss'], label='Val Cls Loss')
    axs[0, 0].plot(results_df['epoch'], results_df['train/dfl_loss'], label='Train DFL Loss')
    axs[0, 0].plot(results_df['epoch'], results_df['val/dfl_loss'], label='Val DFL Loss')
    axs[0, 0].set_title('All Training and Validation Losses')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 2: Performance metrics
    axs[0, 1].plot(results_df['epoch'], results_df['metrics/precision(B)'], label='Precision')
    axs[0, 1].plot(results_df['epoch'], results_df['metrics/recall(B)'], label='Recall')
    axs[0, 1].plot(results_df['epoch'], results_df['metrics/mAP50(B)'], label='mAP50')
    axs[0, 1].plot(results_df['epoch'], results_df['metrics/mAP50-95(B)'], label='mAP50-95')
    axs[0, 1].set_title('Model Performance Metrics')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Metric Value')
    axs[0, 1].legend()
    axs[0, 1].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 3: Box and Classification Loss Stability
    # Calculate rolling mean and standard deviation to assess stability
    window = 3  # 3-epoch window for smoothing
    axs[1, 0].plot(results_df['epoch'], results_df['train/box_loss'].rolling(window=window, min_periods=1).mean(), 
                  label='Train Box Loss (MA)')
    axs[1, 0].fill_between(results_df['epoch'], 
                         results_df['train/box_loss'].rolling(window=window, min_periods=1).mean() - 
                         results_df['train/box_loss'].rolling(window=window, min_periods=1).std(),
                         results_df['train/box_loss'].rolling(window=window, min_periods=1).mean() + 
                         results_df['train/box_loss'].rolling(window=window, min_periods=1).std(),
                         alpha=0.3)
    axs[1, 0].plot(results_df['epoch'], results_df['val/box_loss'].rolling(window=window, min_periods=1).mean(), 
                  label='Val Box Loss (MA)')
    axs[1, 0].fill_between(results_df['epoch'], 
                         results_df['val/box_loss'].rolling(window=window, min_periods=1).mean() - 
                         results_df['val/box_loss'].rolling(window=window, min_periods=1).std(),
                         results_df['val/box_loss'].rolling(window=window, min_periods=1).mean() + 
                         results_df['val/box_loss'].rolling(window=window, min_periods=1).std(),
                         alpha=0.3)
    axs[1, 0].set_title('Box Loss Stability (Moving Average ± StdDev)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Loss')
    axs[1, 0].legend()
    axs[1, 0].grid(True, linestyle='--', alpha=0.7)
    
    # Plot 4: Train vs Val Loss Ratio (stability indicator)
    train_val_ratio_box = results_df['train/box_loss'] / results_df['val/box_loss']
    train_val_ratio_cls = results_df['train/cls_loss'] / results_df['val/cls_loss']
    train_val_ratio_dfl = results_df['train/dfl_loss'] / results_df['val/dfl_loss']
    
    axs[1, 1].plot(results_df['epoch'], train_val_ratio_box, label='Box Loss Ratio')
    axs[1, 1].plot(results_df['epoch'], train_val_ratio_cls, label='Cls Loss Ratio')
    axs[1, 1].plot(results_df['epoch'], train_val_ratio_dfl, label='DFL Loss Ratio')
    axs[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Ideal Ratio = 1.0')
    axs[1, 1].set_title('Train/Val Loss Ratio (Stability Indicator)')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Train/Val Ratio')
    axs[1, 1].legend()
    axs[1, 1].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join('/kaggle/working', 'training_stability_analysis.png'))
    plt.show()
    
    # Calculate and print stability metrics
    print("Training Stability Analysis:")
    print(f"Box Loss Stability (StdDev): Train = {results_df['train/box_loss'].std():.4f}, Val = {results_df['val/box_loss'].std():.4f}")
    print(f"Cls Loss Stability (StdDev): Train = {results_df['train/cls_loss'].std():.4f}, Val = {results_df['val/cls_loss'].std():.4f}")
    print(f"DFL Loss Stability (StdDev): Train = {results_df['train/dfl_loss'].std():.4f}, Val = {results_df['val/dfl_loss'].std():.4f}")
    
    # Calculate final vs. initial loss ratio (lower is better)
    loss_reduction_train = results_df['train/box_loss'].iloc[-1] / results_df['train/box_loss'].iloc[0]
    loss_reduction_val = results_df['val/box_loss'].iloc[-1] / results_df['val/box_loss'].iloc[0]
    print(f"Loss Reduction (Final/Initial): Train = {loss_reduction_train:.4f}, Val = {loss_reduction_val:.4f}")
    
    return results_df



def predict_on_samples(model, num_samples=4):
    """
    Run predictions on random validation samples and display results.
    
    Args:
        model: Trained YOLO model.
        num_samples (int): Number of random samples to test.
    """
    val_dir = os.path.join(yolo_dataset_dir, 'images', 'val')
    if not os.path.exists(val_dir):
        print(f"Validation directory not found at {val_dir}")
        val_dir = os.path.join(yolo_dataset_dir, 'images', 'train')
        print(f"Using train directory for predictions instead: {val_dir}")
        
    if not os.path.exists(val_dir):
        print("No images directory found for predictions")
        return
    
    val_images = os.listdir(val_dir)
    if len(val_images) == 0:
        print("No images found for prediction")
        return
    
    num_samples = min(num_samples, len(val_images))
    samples = random.sample(val_images, num_samples)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes = axes.flatten()
    
    for i, img_file in enumerate(samples):
        if i >= len(axes):
            break
            
        img_path = os.path.join(val_dir, img_file)
        results = model.predict(img_path, conf=0.25)[0]
        img = Image.open(img_path)
        axes[i].imshow(np.array(img), cmap='gray')
        
        # Draw ground truth box if available (extracted from filename)
        try:
            parts = img_file.split('_')
            y_part = [p for p in parts if p.startswith('y')]
            x_part = [p for p in parts if p.startswith('x')]
            if y_part and x_part:
                y_gt = int(y_part[0][1:])
                x_gt = int(x_part[0][1:].split('.')[0])
                box_size = 24
                rect_gt = Rectangle((x_gt - box_size//2, y_gt - box_size//2), box_size, box_size,
                                      linewidth=1, edgecolor='g', facecolor='none')
                axes[i].add_patch(rect_gt)
        except:
            pass
        
        if len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = box
                rect_pred = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
                axes[i].add_patch(rect_pred)
                axes[i].text(x1, y1-5, f'{conf:.2f}', color='red')
        
        axes[i].set_title(f"Image: {img_file}\nGT (green) vs Pred (red)")
    
    plt.tight_layout()
    plt.savefig(os.path.join('/kaggle/working', 'predictions.png'))
    plt.show()


def prepare_dataset():
    """
    Check if the dataset exists and create/fix a proper YAML file for training.
    
    Returns:
        str: Path to the YAML file to use for training.
    """
    train_images_dir = os.path.join(yolo_dataset_dir, 'images', 'train')
    val_images_dir = os.path.join(yolo_dataset_dir, 'images', 'val')
    train_labels_dir = os.path.join(yolo_dataset_dir, 'labels', 'train')
    val_labels_dir = os.path.join(yolo_dataset_dir, 'labels', 'val')
    
    print(f"Directory status:")
    print(f"- Train images exists: {os.path.exists(train_images_dir)}")
    print(f"- Val images exists: {os.path.exists(val_images_dir)}")
    print(f"- Train labels exists: {os.path.exists(train_labels_dir)}")
    print(f"- Val labels exists: {os.path.exists(val_labels_dir)}")
    
    original_yaml_path = os.path.join(yolo_dataset_dir, 'dataset.yaml')
    if os.path.exists(original_yaml_path):
        print(f"Found original dataset.yaml at {original_yaml_path}")
        return fix_yaml_paths(original_yaml_path)
    else:
        print("Original dataset.yaml not found, creating a new one")
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

def main():
    print("Starting improved YOLO training process...")
    yaml_path = prepare_dataset()
    print(f"Using YAML file: {yaml_path}")
    
    # Visualize learning rate schedule before training
    print("\nVisualizing learning rate schedule...")
    lr_schedule = plot_learning_rate_schedule(epochs=50, lr0=0.0005, lrf=0.05, warmup_epochs=3)
    
    print("\nStarting YOLO training with stabilized hyperparameters...")
    model, results = train_yolo_model(
        yaml_path,
        pretrained_weights_path=yolo_pretrained_weights,
        epochs=50  # Increase epochs for better convergence
    )
    
    print("\nTraining complete!")
    
    # Analyze training stability
    results_csv = os.path.join(yolo_weights_dir, 'motor_detector_stable', 'results.csv')
    if os.path.exists(results_csv):
        print("\nAnalyzing training stability...")
        stability_analysis = analyze_training_stability(results_csv)
    
    print("\nRunning predictions on sample images...")
    predict_on_samples(model, num_samples=4)
    
    print("\nTraining process complete. Check the output directory for results and visualizations.")

if __name__ == "__main__":
    main()

