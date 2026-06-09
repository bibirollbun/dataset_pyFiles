!tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
!pip install --no-index --find-links=./packages ultralytics
!rm -rf ./packages


!pip install plotly


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


# Load the training labels CSV into a pandas DataFrame
train_labels = pd.read_csv(TRAIN_CSV)

# Display basic dataset information
print("Training dataset shape:", train_labels.shape)
print("\nColumns in the dataset:")
print(train_labels.columns.tolist())

# Display basic statistics for numerical columns
print("\nBasic statistics:")
display(train_labels.describe())


# Count unique tomograms in the dataset
unique_tomo_count = train_labels['tomo_id'].nunique()
print(f"\nNumber of unique tomograms: {unique_tomo_count}")

# Compute distribution of motors per tomogram
motors_per_tomo = train_labels.groupby('tomo_id')['Number of motors'].first().value_counts().sort_index()
print("\nDistribution of motors per tomogram:")
print(motors_per_tomo)

# Visualize the distribution with a bar plot
plt.figure(figsize=(8, 5))
motors_per_tomo.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Distribution of Motors per Tomogram')
plt.xlabel('Number of Motors')
plt.ylabel('Frequency')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# Display a few sample rows from the training labels
print("\nSample rows from training labels:")
display(train_labels.head())

# Check for missing values in each column
print("\nMissing values per column:")
display(train_labels.isnull().sum())

# Explore the range of tomogram sizes along each axis
print("\nTomogram size ranges:")
print("Z-axis (slices):", train_labels['Array shape (axis 0)'].min(), "to", train_labels['Array shape (axis 0)'].max())
print("X-axis (width):", train_labels['Array shape (axis 1)'].min(), "to", train_labels['Array shape (axis 1)'].max())
print("Y-axis (height):", train_labels['Array shape (axis 2)'].min(), "to", train_labels['Array shape (axis 2)'].max())

# Display voxel spacing distribution
print("\nVoxel spacing distribution:")
voxel_spacing_counts = train_labels['Voxel spacing'].value_counts().sort_index()
display(voxel_spacing_counts)


fig_motor = px.scatter_3d(
    train_labels, 
    x='Motor axis 0', 
    y='Motor axis 1', 
    z='Motor axis 2',
    color='Number of motors', 
    color_continuous_scale="viridis",  # Using a vibrant color scheme
    size_max=8, 
    width=900, 
    height=600, 
    opacity=0.85, 
    template="plotly_white",  # Lighter theme for better contrast
    title="ðŸš€ 3D Scatter Plot: Motor Axes"
)

fig_motor.update_layout(
    font_size=10,
    legend_font_size=14,
    margin=dict(l=10, r=10, b=10, t=40)
)

fig_motor.show()


fig_shape = px.scatter_3d(
    train_labels, 
    x='Array shape (axis 0)', 
    y='Array shape (axis 1)', 
    z='Array shape (axis 2)',
    color='Number of motors', 
    color_continuous_scale="magma",  # More contrast for clarity
    size_max=8, 
    width=900, 
    height=600, 
    opacity=0.85, 
    template="seaborn",  # New theme for a scientific feel
    title="ðŸ§¬ 3D Scatter Plot: Tomogram Shapes"
)

fig_shape.update_layout(
    font_size=10,
    legend_font_size=14,
    margin=dict(l=10, r=10, b=10, t=40)
)

fig_shape.show()


# Show descriptive statistics
display(train_labels.describe().loc[['mean', 'min', 'max']].T)

# Improved histogram design
train_labels.hist(
    bins=30, 
    figsize=(14, 8), 
    layout=(3, 4), 
    edgecolor="black", 
    color="#4CAF50"  # Greenish color theme
)
plt.suptitle("Feature Distributions", fontsize=16, fontweight='bold', color="darkblue")
plt.tight_layout()


plt.figure(figsize=(9, 5), facecolor="white")
sns.heatmap(
    data=train_labels.corr(numeric_only=True),
    cmap="coolwarm",  # Strong contrast for positive/negative correlations
    vmin=-1, vmax=1,
    linecolor="white", linewidth=0.6,
    annot=True,
    fmt=".2f"
)
plt.title('Correlation Heatmap', fontsize=14, fontweight='bold', color="black")
plt.show()


def plotImages(title, directory, n_images=16, img_size=(128, 128)):
    """
    Display a grid of images from the specified directory.
    
    Args:
        title (str): Title to print before displaying images.
        directory (str): Glob pattern for image files.
        n_images (int): Number of images to display.
        img_size (tuple): Size to resize images for display.
    """
    print(f"ðŸ–¼ {title}")
    image_files = glob.glob(directory)
    
    if not image_files:
        print("No images found.")
        return
    
    plt.figure(figsize=(12, 12))
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    for i, file_path in enumerate(image_files[:n_images]):
        img = cv2.imread(file_path)
        if img is None:
            continue
        img = cv2.resize(img, img_size)
        plt.subplot(4, 4, i+1)
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
    
    plt.suptitle(title, fontsize=14, fontweight='bold', color="darkred")
    plt.show()

plotImages("Bacterial Flagellar Motors - Train Images", "../input/byu-locating-bacterial-flagellar-motors-2025/train/***/**")


def visualize_images(path, n_images=12, is_random=True, figsize=(14, 14)):
      
    plt.figure(figsize=figsize)
    
    image_names = os.listdir(path)
    if is_random:
        image_names = random.sample(image_names, min(len(image_names), n_images))
    else:
        image_names = image_names[:n_images]
    
    w = int(math.sqrt(n_images))
    h = math.ceil(n_images / w)
    
    for ind, image_name in enumerate(image_names):
        img_path = os.path.join(path, image_name)
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.subplot(h, w, ind + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    
    plt.suptitle("Sample Tomogram Images", fontsize=14, fontweight='bold', color="darkblue")
    plt.show()

visualize_images("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/tomo_098751", n_images=9)


# Select a sample tomogram ID to visualize
sample_tomo_id = train_labels['tomo_id'].iloc[0]
print(f"\nVisualizing sample tomogram: {sample_tomo_id}")

# Construct the folder path for the selected tomogram
sample_folder = os.path.join(TRAIN_DIR, sample_tomo_id)

if os.path.exists(sample_folder):
    # Get all JPEG slice files from the tomogram folder
    slice_files = sorted(glob.glob(os.path.join(sample_folder, '*.jpg')))
    print(f"Number of slice files in tomogram '{sample_tomo_id}': {len(slice_files)}")
    
    if slice_files:
        # Load the first slice to check its dimensions
        sample_slice = Image.open(slice_files[0])
        print(f"Dimensions of a sample slice: {sample_slice.size}")
        
        # Plot slices from the beginning, middle, and end of the tomogram
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        slice_indices = [0, len(slice_files)//2, len(slice_files)-1]
        for i, idx in enumerate(slice_indices):
            img = Image.open(slice_files[idx])
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Slice {idx}")
            axes[i].axis('off')
        plt.tight_layout()
        plt.show()
    else:
        print("No slice files found in the folder.")
else:
    print(f"Folder '{sample_folder}' does not exist. Please check the dataset directory.")


!pip install ipywidgets
!pip install progress
from tqdm import tqdm


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
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)

# Define the preprocessing function to extract slices, normalize, and generate YOLO annotations.
def prepare_yolo_dataset(trust=TRUST, train_split=TRAIN_SPLIT):

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


# Set paths for the preprocessed YOLO training images and labels.
images_train_dir = os.path.join(yolo_dataset_dir, "images", "train")
labels_train_dir = os.path.join(yolo_dataset_dir, "labels", "train")

# %%
def visualize_random_training_samples(num_samples=4):
    # Get all image files from the train directory (support multiple image extensions)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(images_train_dir, "**", ext), recursive=True))
    
    if len(image_files) == 0:
        print("No image files found in the train directory!")
        return
        
    num_samples = min(num_samples, len(image_files))
    random_images = random.sample(image_files, num_samples)
    
    # Create subplots for visualization
    rows = int(np.ceil(num_samples / 2))
    cols = min(num_samples, 2)
    fig, axes = plt.subplots(rows, cols, figsize=(14, 5 * rows))
    
    if num_samples == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, img_path in enumerate(random_images):
        try:
            # Determine corresponding label file (YOLO format)
            relative_path = os.path.relpath(img_path, images_train_dir)
            label_path = os.path.join(labels_train_dir, os.path.splitext(relative_path)[0] + '.txt')
            
            # Load and normalize image for display
            img = Image.open(img_path)
            img_width, img_height = img.size
            img_array = np.array(img)
            p2 = np.percentile(img_array, 2)
            p98 = np.percentile(img_array, 98)
            normalized = np.clip(img_array, p2, p98)
            normalized = 255 * (normalized - p2) / (p98 - p2)
            img_normalized = Image.fromarray(np.uint8(normalized))
            
            # Convert to RGB for annotation drawing
            img_rgb = img_normalized.convert('RGB')
            overlay = Image.new('RGBA', img_rgb.size, (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)
            
            # Load YOLO annotations if available
            annotations = []
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        # YOLO format: class x_center y_center width height (normalized values)
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
            
            # Draw annotations on the overlay
            for ann in annotations:
                x_center = ann['x_center']
                y_center = ann['y_center']
                width = ann['width']
                height = ann['height']
                x1 = max(0, int(x_center - width/2))
                y1 = max(0, int(y_center - height/2))
                x2 = min(img_width, int(x_center + width/2))
                y2 = min(img_height, int(y_center + height/2))
                draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 64), outline=(255, 0, 0, 200))
                draw.text((x1, y1-10), f"Class {ann['class_id']}", fill=(255, 0, 0, 255))
            
            # Indicate if no annotations were found
            if not annotations:
                draw.text((10, 10), "No annotations found", fill=(255, 0, 0, 255))
            
            # Composite overlay and display image
            img_rgb = Image.alpha_composite(img_rgb.convert('RGBA'), overlay).convert('RGB')
            axes[i].imshow(np.array(img_rgb))
            img_name = os.path.basename(img_path)
            axes[i].set_title(f"Image: {img_name}\nAnnotations: {len(annotations)}")
            axes[i].axis('on')
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            axes[i].text(0.5, 0.5, f"Error loading image: {os.path.basename(img_path)}",
                         horizontalalignment='center', verticalalignment='center')
            axes[i].axis('off')
    
    # Turn off any extra subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    plt.tight_layout()
    plt.show()
    print(f"Displayed {num_samples} random images with YOLO annotations")

# Run visualization of random training samples
visualize_random_training_samples(4)



# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
torch.manual_seed(42)

# Define paths for the Kaggle environment
yolo_dataset_dir = "/kaggle/working/yolo_dataset"
yolo_weights_dir = "/kaggle/working/yolo_weights"
yolo_pretrained_weights = "/kaggle/input/ultralytics-for-offline-install/yolov8n.pt"  # Pre-downloaded weights

# Create the weights directory if it does not exist
os.makedirs(yolo_weights_dir, exist_ok=True)


def fix_yaml_paths(yaml_path):
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


def train_yolo_model(yaml_path, pretrained_weights_path, epochs=30, batch_size=16, img_size=640):
    print(f"Loading pre-trained weights from: {pretrained_weights_path}")
    model = YOLO(pretrained_weights_path)
    
    results = model.train(
        data=yaml_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=img_size,
        project=yolo_weights_dir,
        name='motor_detector',
        exist_ok=True,
        patience=5,
        save_period=5,
        val=True,
        verbose=True
    )
    
    run_dir = os.path.join(yolo_weights_dir, 'motor_detector')
    best_epoch_info = plot_dfl_loss_curve(run_dir)
    if best_epoch_info:
        best_epoch, best_val_loss = best_epoch_info
        print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")
    
    return model, results


def predict_on_samples(model, num_samples=4):
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


import os


def prepare_dataset():
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
    print("Starting YOLO training process...")
    yaml_path = prepare_dataset()
    print(f"Using YAML file: {yaml_path}")
    with open(yaml_path, 'r') as f:
        print(f"YAML contents:\n{f.read()}")
    
    print("\nStarting YOLO training...")
    model, results = train_yolo_model(
        yaml_path,
        pretrained_weights_path=yolo_pretrained_weights,
        epochs=30  # For demonstration, using 30 epochs
    )
    
    print("\nTraining complete!")
    print("\nRunning predictions on sample images...")
    predict_on_samples(model, num_samples=4)

if __name__ == "__main__":
    main()


# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths for the test data and submission
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/"
test_dir = os.path.join(data_path, "test")
submission_path = "/kaggle/working/submission.csv"

# Path to the best trained model (adjust if necessary)
model_path = "/kaggle/working/yolo_weights/motor_detector/weights/best.pt"

# Define detection and processing parameters
CONFIDENCE_THRESHOLD = 0.45
MAX_DETECTIONS_PER_TOMO = 3
NMS_IOU_THRESHOLD = 0.2
CONCENTRATION = 1  # Process a fraction of slices for fast submission

# GPU profiling context manager for timing
class GPUProfiler:
    def __init__(self, name):
        self.name = name
        self.start_time = None
        
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.time() - self.start_time
        print(f"[PROFILE] {self.name}: {elapsed:.3f}s")

# Set device and dynamic batch size
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8
if device.startswith('cuda'):
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Using GPU: {gpu_name} with {gpu_mem:.2f} GB memory")
    free_mem = gpu_mem - torch.cuda.memory_allocated(0) / 1e9
    BATCH_SIZE = max(8, min(32, int(free_mem * 4)))
    print(f"Dynamic batch size set to {BATCH_SIZE} based on {free_mem:.2f}GB free memory")
else:
    print("GPU not available, using CPU")
    BATCH_SIZE = 4


def normalize_slice(slice_data):

    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)

def preload_image_batch(file_paths):
    images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is None:
            img = np.array(Image.open(path))
        images.append(img)
    return images

def perform_3d_nms(detections, iou_threshold):
    if not detections:
        return []
    
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    final_detections = []
    def distance_3d(d1, d2):
        return np.sqrt((d1['z'] - d2['z'])**2 + (d1['y'] - d2['y'])**2 + (d1['x'] - d2['x'])**2)
    
    box_size = 24
    distance_threshold = box_size * iou_threshold
    
    while detections:
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        detections = [d for d in detections if distance_3d(d, best_detection) > distance_threshold]
    
    return final_detections

def process_tomogram(tomo_id, model, index=0, total=1):
    print(f"Processing tomogram {tomo_id} ({index}/{total})")
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    
    selected_indices = np.linspace(0, len(slice_files)-1, int(len(slice_files) * CONCENTRATION))
    selected_indices = np.round(selected_indices).astype(int)
    slice_files = [slice_files[i] for i in selected_indices]
    
    print(f"Processing {len(slice_files)} out of {len(os.listdir(tomo_dir))} slices (CONCENTRATION={CONCENTRATION})")
    all_detections = []
    
    if device.startswith('cuda'):
        streams = [torch.cuda.Stream() for _ in range(min(4, BATCH_SIZE))]
    else:
        streams = [None]
    
    next_batch_thread = None
    next_batch_images = None
    
    for batch_start in range(0, len(slice_files), BATCH_SIZE):
        if next_batch_thread is not None:
            next_batch_thread.join()
            next_batch_images = None
            
        batch_end = min(batch_start + BATCH_SIZE, len(slice_files))
        batch_files = slice_files[batch_start:batch_end]
        
        next_batch_start = batch_end
        next_batch_end = min(next_batch_start + BATCH_SIZE, len(slice_files))
        next_batch_files = slice_files[next_batch_start:next_batch_end] if next_batch_start < len(slice_files) else []
        if next_batch_files:
            next_batch_paths = [os.path.join(tomo_dir, f) for f in next_batch_files]
            next_batch_thread = threading.Thread(target=preload_image_batch, args=(next_batch_paths,))
            next_batch_thread.start()
        else:
            next_batch_thread = None
        
        sub_batches = np.array_split(batch_files, len(streams))
        for i, sub_batch in enumerate(sub_batches):
            if len(sub_batch) == 0:
                continue
            stream = streams[i % len(streams)]
            with torch.cuda.stream(stream) if stream and device.startswith('cuda') else nullcontext():
                sub_batch_paths = [os.path.join(tomo_dir, slice_file) for slice_file in sub_batch]
                sub_batch_slice_nums = [int(slice_file.split('_')[1].split('.')[0]) for slice_file in sub_batch]
                with GPUProfiler(f"Inference batch {i+1}/{len(sub_batches)}"):
                    sub_results = model(sub_batch_paths, verbose=False)
                for j, result in enumerate(sub_results):
                    if len(result.boxes) > 0:
                        for box_idx, confidence in enumerate(result.boxes.conf):
                            if confidence >= CONFIDENCE_THRESHOLD:
                                x1, y1, x2, y2 = result.boxes.xyxy[box_idx].cpu().numpy()
                                x_center = (x1 + x2) / 2
                                y_center = (y1 + y2) / 2
                                all_detections.append({
                                    'z': round(sub_batch_slice_nums[j]),
                                    'y': round(y_center),
                                    'x': round(x_center),
                                    'confidence': float(confidence)
                                })
        if device.startswith('cuda'):
            torch.cuda.synchronize()
    
    if next_batch_thread is not None:
        next_batch_thread.join()
    
    final_detections = perform_3d_nms(all_detections, NMS_IOU_THRESHOLD)
    final_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    if not final_detections:
        return {'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1}
    
    best_detection = final_detections[0]
    return {
        'tomo_id': tomo_id,
        'Motor axis 0': round(best_detection['z']),
        'Motor axis 1': round(best_detection['y']),
        'Motor axis 2': round(best_detection['x'])
    }

def debug_image_loading(tomo_id):
    tomo_dir = os.path.join(test_dir, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    if not slice_files:
        print(f"No image files found in {tomo_dir}")
        return
        
    print(f"Found {len(slice_files)} image files in {tomo_dir}")
    sample_file = slice_files[len(slice_files)//2]
    img_path = os.path.join(tomo_dir, sample_file)
    
    try:
        img_pil = Image.open(img_path)
        print(f"PIL Image shape: {np.array(img_pil).shape}, dtype: {np.array(img_pil).dtype}")
        img_cv2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        print(f"OpenCV Image shape: {img_cv2.shape}, dtype: {img_cv2.dtype}")
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        print(f"OpenCV RGB Image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
        print("Image loading successful!")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        
    try:
        test_model = YOLO(model_path)
        test_results = test_model([img_path], verbose=False)
        print("YOLO model successfully processed the test image")
    except Exception as e:
        print(f"Error with YOLO processing: {e}")


def generate_submission():
    test_tomos = sorted([d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))])
    total_tomos = len(test_tomos)
    print(f"Found {total_tomos} tomograms in test directory")
    
    if test_tomos:
        debug_image_loading(test_tomos[0])
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    print(f"Loading YOLO model from {model_path}")
    model = YOLO(model_path)
    model.to(device)
    if device.startswith('cuda'):
        model.fuse()
        if torch.cuda.get_device_capability(0)[0] >= 7:
            model.model.half()
            print("Using half precision (FP16) for inference")
    
    results = []
    motors_found = 0
    
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_tomo = {}
        for i, tomo_id in enumerate(test_tomos, 1):
            future = executor.submit(process_tomogram, tomo_id, model, i, total_tomos)
            future_to_tomo[future] = tomo_id
        
        for future in future_to_tomo:
            tomo_id = future_to_tomo[future]
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                result = future.result()
                results.append(result)
                has_motor = not pd.isna(result['Motor axis 0'])
                if has_motor:
                    motors_found += 1
                    print(f"Motor found in {tomo_id} at position: z={result['Motor axis 0']}, y={result['Motor axis 1']}, x={result['Motor axis 2']}")
                else:
                    print(f"No motor detected in {tomo_id}")
                print(f"Current detection rate: {motors_found}/{len(results)} ({motors_found/len(results)*100:.1f}%)")
            except Exception as e:
                print(f"Error processing {tomo_id}: {e}")
                results.append({'tomo_id': tomo_id, 'Motor axis 0': -1, 'Motor axis 1': -1, 'Motor axis 2': -1})
    
    submission_df = pd.DataFrame(results)
    submission_df = submission_df[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\nSubmission complete!")
    print(f"Motors detected: {motors_found}/{total_tomos} ({motors_found/total_tomos*100:.1f}%)")
    print(f"Submission saved to: {submission_path}")
    print("\nSubmission preview:")
    print(submission_df.head())
    return submission_df


if __name__ == "__main__":
    start_time = time.time()
    submission = generate_submission()
    elapsed = time.time() - start_time
    print(f"\nTotal execution time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")




