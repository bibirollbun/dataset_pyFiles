# !tar xfvz /kaggle/input/ultralytics-for-offline-install/archive.tar.gz
# !pip install --no-index --find-links=./packages ultralytics
!pip install plotly scikit-learn
!rm -rf ./packages

!pip install /kaggle/input/ultralytics-timm/ultralytics-8.3.133-py3-none-any.whl --no-deps


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

































# # æ­¤æ ¼ä¸�parse_data.pyå�Œ
# # Define YOLO dataset structure and parameters
# data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025"
# train_dir = os.path.join(data_path, "train")

# # Output directories for YOLO dataset (adjust as needed)
# yolo_dataset_dir = "yolo_dataset/"
# yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
# yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
# yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
# yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

# # Create necessary directories
# for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
#     os.makedirs(dir_path, exist_ok=True)

# # Define constants for processing
# TRUST = 2       # Number of slices above and below center slice (total slices = 2*TRUST + 1)
# BOX_SIZE = 24   # Bounding box size (in pixels)
# TRAIN_SPLIT = 0.8  # 80% training, 20% validation

# # Define a helper function for image normalization using percentile-based contrast enhancement.
# def normalize_slice(slice_data):
#     """
#     Normalize slice data using the 2nd and 98th percentiles.
    
#     Args:
#         slice_data (numpy.array): Input image slice.
    
#     Returns:
#         np.uint8: Normalized image in the range [0, 255].
#     """
#     p2 = np.percentile(slice_data, 2)
#     p98 = np.percentile(slice_data, 98)
#     clipped_data = np.clip(slice_data, p2, p98)
#     normalized = 255 * (clipped_data - p2) / (p98 - p2)
#     return np.uint8(normalized)

# # Define the preprocessing function to extract slices, normalize, and generate YOLO annotations.
# def prepare_yolo_dataset(trust=TRUST, train_split=TRAIN_SPLIT):
#     """
#     Extract slices containing motors and save images with corresponding YOLO annotations.
    
#     Steps:
#     - Load the motor labels.
#     - Perform a train/validation split by tomogram.
#     - For each motor, extract slices in a range (Â± trust parameter).
#     - Normalize each slice and save it.
#     - Generate YOLO format bounding box annotations with a fixed box size.
#     - Create a YAML configuration file for YOLO training.
    
#     Returns:
#         dict: A summary containing dataset statistics and file paths.
#     """
#     # Load the labels CSV
#     labels_df = pd.read_csv(os.path.join(data_path, "train_labels.csv"))
    
#     total_motors = labels_df['Number of motors'].sum()
#     print(f"Total number of motors in the dataset: {total_motors}")
    
#     # Consider only tomograms with at least one motor
#     tomo_df = labels_df[labels_df['Number of motors'] > 0].copy()
#     unique_tomos = tomo_df['tomo_id'].unique()
#     print(f"Found {len(unique_tomos)} unique tomograms with motors")
    
#     # Shuffle and split tomograms into train and validation sets
#     np.random.shuffle(unique_tomos)
#     split_idx = int(len(unique_tomos) * train_split)
#     train_tomos = unique_tomos[:split_idx]
#     val_tomos = unique_tomos[split_idx:]
#     print(f"Split: {len(train_tomos)} tomograms for training, {len(val_tomos)} tomograms for validation")
    
#     # Helper function to process a list of tomograms
#     def process_tomogram_set(tomogram_ids, images_dir, labels_dir, set_name):
#         motor_counts = []
#         for tomo_id in tomogram_ids:
#             # Get motor annotations for the current tomogram
#             tomo_motors = labels_df[labels_df['tomo_id'] == tomo_id]
#             for _, motor in tomo_motors.iterrows():
#                 if pd.isna(motor['Motor axis 0']):
#                     continue
#                 motor_counts.append(
#                     (tomo_id, 
#                      int(motor['Motor axis 0']), 
#                      int(motor['Motor axis 1']), 
#                      int(motor['Motor axis 2']),
#                      int(motor['Array shape (axis 0)']))
#                 )
        
#         print(f"Will process approximately {len(motor_counts) * (2 * trust + 1)} slices for {set_name}")
#         processed_slices = 0
        
#         # Loop over each motor annotation
#         for tomo_id, z_center, y_center, x_center, z_max in tqdm(motor_counts, desc=f"Processing {set_name} motors"):
#             z_min = max(0, z_center - trust)
#             z_max_bound = min(z_max - 1, z_center + trust)
#             for z in range(z_min, z_max_bound + 1):
#                 # Create the slice filename and source path
#                 slice_filename = f"slice_{z:04d}.jpg"
#                 src_path = os.path.join(train_dir, tomo_id, slice_filename)
#                 if not os.path.exists(src_path):
#                     print(f"Warning: {src_path} does not exist, skipping.")
#                     continue
                
#                 # Load, normalize, and save the image slice
#                 img = Image.open(src_path)
#                 img_array = np.array(img)
#                 normalized_img = normalize_slice(img_array)
#                 dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
#                 dest_path = os.path.join(images_dir, dest_filename)
#                 Image.fromarray(normalized_img).save(dest_path)
                
#                 # Prepare YOLO bounding box annotation (normalized values)
#                 img_width, img_height = img.size
#                 x_center_norm = x_center / img_width
#                 y_center_norm = y_center / img_height
#                 box_width_norm = BOX_SIZE / img_width
#                 box_height_norm = BOX_SIZE / img_height
#                 label_path = os.path.join(labels_dir, dest_filename.replace('.jpg', '.txt'))
#                 with open(label_path, 'w') as f:
#                     f.write(f"0 {x_center_norm} {y_center_norm} {box_width_norm} {box_height_norm}\n")
                
#                 processed_slices += 1
        
#         return processed_slices, len(motor_counts)
    
#     # Process training tomograms
#     train_slices, train_motors = process_tomogram_set(train_tomos, yolo_images_train, yolo_labels_train, "training")
#     # Process validation tomograms
#     val_slices, val_motors = process_tomogram_set(val_tomos, yolo_images_val, yolo_labels_val, "validation")
    
#     # Generate YAML configuration for YOLO training
#     yaml_content = {
#         'path': yolo_dataset_dir,
#         'train': 'images/train',
#         'val': 'images/val',
#         'names': {0: 'motor'}
#     }
#     with open(os.path.join(yolo_dataset_dir, 'dataset.yaml'), 'w') as f:
#         yaml.dump(yaml_content, f, default_flow_style=False)
    
#     print(f"\nProcessing Summary:")
#     print(f"- Train set: {len(train_tomos)} tomograms, {train_motors} motors, {train_slices} slices")
#     print(f"- Validation set: {len(val_tomos)} tomograms, {val_motors} motors, {val_slices} slices")
#     print(f"- Total: {len(train_tomos) + len(val_tomos)} tomograms, {train_motors + val_motors} motors, {train_slices + val_slices} slices")
    
#     return {
#         "dataset_dir": yolo_dataset_dir,
#         "yaml_path": os.path.join(yolo_dataset_dir, 'dataset.yaml'),
#         "train_tomograms": len(train_tomos),
#         "val_tomograms": len(val_tomos),
#         "train_motors": train_motors,
#         "val_motors": val_motors,
#         "train_slices": train_slices,
#         "val_slices": val_slices
#     }

# # Run the preprocessing
# summary = prepare_yolo_dataset(TRUST)
# print(f"\nPreprocessing Complete:")
# print(f"- Training data: {summary['train_tomograms']} tomograms, {summary['train_motors']} motors, {summary['train_slices']} slices")
# print(f"- Validation data: {summary['val_tomograms']} tomograms, {summary['val_motors']} motors, {summary['val_slices']} slices")
# print(f"- Dataset directory: {summary['dataset_dir']}")
# print(f"- YAML configuration: {summary['yaml_path']}")
# print("\nReady for YOLO training!")





# # Set random seeds for reproducibility
# np.random.seed(42)
# random.seed(42)
# torch.manual_seed(42)

# # Define paths for the Kaggle environment
# yolo_dataset_dir = "yolo_dataset"
# yolo_weights_dir = "yolo_weights"
# yolo_pretrained_weights = "/kaggle/input/yolo/pytorch/default/2/yolov10m.pt"  # Pre-downloaded weights ###v10

# # Create the weights directory if it does not exist
# os.makedirs(yolo_weights_dir, exist_ok=True)


# def fix_yaml_paths(yaml_path):
#     """
#     Fix the paths in the YAML file to match the actual Kaggle directories.
    
#     Args:
#         yaml_path (str): Path to the original dataset YAML file.
        
#     Returns:
#         str: Path to the fixed YAML file.
#     """
#     print(f"Fixing YAML paths in {yaml_path}")
#     with open(yaml_path, 'r') as f:
#         yaml_data = yaml.safe_load(f)
    
#     if 'path' in yaml_data:
#         yaml_data['path'] = yolo_dataset_dir
    
#     fixed_yaml_path = "/kaggle/working/fixed_dataset.yaml" ###
#     with open(fixed_yaml_path, 'w') as f:
#         yaml.dump(yaml_data, f)
    
#     print(f"Created fixed YAML at {fixed_yaml_path} with path: {yaml_data.get('path')}")
#     return fixed_yaml_path


# def plot_dfl_loss_curve(run_dir):
#     """
#     Plot the DFL loss curves for training and validation, marking the best model.
    
#     Args:
#         run_dir (str): Directory where the training results are stored.
#     """
#     results_csv = os.path.join(run_dir, 'results.csv')
#     if not os.path.exists(results_csv):
#         print(f"Results file not found at {results_csv}")
#         return
    
#     results_df = pd.read_csv(results_csv)
#     train_dfl_col = [col for col in results_df.columns if 'train/dfl_loss' in col]
#     val_dfl_col = [col for col in results_df.columns if 'val/dfl_loss' in col]
    
#     if not train_dfl_col or not val_dfl_col:
#         print("DFL loss columns not found in results CSV")
#         print(f"Available columns: {results_df.columns.tolist()}")
#         return
    
#     train_dfl_col = train_dfl_col[0]
#     val_dfl_col = val_dfl_col[0]
    
#     best_epoch = results_df[val_dfl_col].idxmin()
#     best_val_loss = results_df.loc[best_epoch, val_dfl_col]
    
#     plt.figure(figsize=(10, 6))
#     plt.plot(results_df['epoch'], results_df[train_dfl_col], label='Train DFL Loss')
#     plt.plot(results_df['epoch'], results_df[val_dfl_col], label='Validation DFL Loss')
#     plt.axvline(x=results_df.loc[best_epoch, 'epoch'], color='r', linestyle='--', 
#                 label=f'Best Model (Epoch {int(results_df.loc[best_epoch, "epoch"])}, Val Loss: {best_val_loss:.4f})')
#     plt.xlabel('Epoch')
#     plt.ylabel('DFL Loss')
#     plt.title('Training and Validation DFL Loss')
#     plt.legend()
#     plt.grid(True, linestyle='--', alpha=0.7)
    
#     plot_path = os.path.join(run_dir, 'dfl_loss_curve.png')
#     plt.savefig(plot_path)
#     plt.savefig(os.path.join('/kaggle/working', 'dfl_loss_curve.png')) ###
    
#     print(f"Loss curve saved to {plot_path}")
#     plt.close()
    
#     return best_epoch, best_val_loss


# def train_yolo_model(yaml_path, pretrained_weights_path, epochs=30, batch_size=16, img_size=640):
#     """
#     Train a YOLO model on the prepared dataset.
    
#     Args:
#         yaml_path (str): Path to the dataset YAML file.
#         pretrained_weights_path (str): Path to pre-downloaded weights file.
#         epochs (int): Number of training epochs.
#         batch_size (int): Batch size for training.
#         img_size (int): Image size for training.
#     """
#     print(f"Loading pre-trained weights from: {pretrained_weights_path}")
#     model = YOLO(pretrained_weights_path)
    
#     results = model.train(
#         data=yaml_path,
#         epochs=epochs,
#         batch=batch_size,
#         imgsz=img_size, ###
#         project=yolo_weights_dir,
#         name='motor_detector',
#         exist_ok=True,
#         patience=10, ##
#         save_period=5,
#         val=True,
#         verbose=True
#     )
    
#     run_dir = os.path.join(yolo_weights_dir, 'motor_detector')
#     best_epoch_info = plot_dfl_loss_curve(run_dir)
#     if best_epoch_info:
#         best_epoch, best_val_loss = best_epoch_info
#         print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")
    
#     return model, results



# def predict_on_samples(model, num_samples=4):
#     """
#     Run predictions on random validation samples and display results.
    
#     Args:
#         model: Trained YOLO model.
#         num_samples (int): Number of random samples to test.
#     """
#     val_dir = os.path.join(yolo_dataset_dir, 'images', 'val')
#     if not os.path.exists(val_dir):
#         print(f"Validation directory not found at {val_dir}")
#         val_dir = os.path.join(yolo_dataset_dir, 'images', 'train')
#         print(f"Using train directory for predictions instead: {val_dir}")
        
#     if not os.path.exists(val_dir):
#         print("No images directory found for predictions")
#         return
    
#     val_images = os.listdir(val_dir)
#     if len(val_images) == 0:
#         print("No images found for prediction")
#         return
    
#     num_samples = min(num_samples, len(val_images))
#     samples = random.sample(val_images, num_samples)
    
#     fig, axes = plt.subplots(2, 2, figsize=(12, 12))
#     axes = axes.flatten()
    
#     for i, img_file in enumerate(samples):
#         if i >= len(axes):
#             break
            
#         img_path = os.path.join(val_dir, img_file)
#         results = model.predict(img_path, conf=0.25)[0]
#         img = Image.open(img_path)
#         axes[i].imshow(np.array(img), cmap='gray')
        
#         # Draw ground truth box if available (extracted from filename)
#         try:
#             parts = img_file.split('_')
#             y_part = [p for p in parts if p.startswith('y')]
#             x_part = [p for p in parts if p.startswith('x')]
#             if y_part and x_part:
#                 y_gt = int(y_part[0][1:])
#                 x_gt = int(x_part[0][1:].split('.')[0])
#                 box_size = 24
#                 rect_gt = Rectangle((x_gt - box_size//2, y_gt - box_size//2), box_size, box_size,
#                                       linewidth=1, edgecolor='g', facecolor='none')
#                 axes[i].add_patch(rect_gt)
#         except:
#             pass
        
#         if len(results.boxes) > 0:
#             boxes = results.boxes.xyxy.cpu().numpy()
#             confs = results.boxes.conf.cpu().numpy()
#             for box, conf in zip(boxes, confs):
#                 x1, y1, x2, y2 = box
#                 rect_pred = Rectangle((x1, y1), x2-x1, y2-y1, linewidth=1, edgecolor='r', facecolor='none')
#                 axes[i].add_patch(rect_pred)
#                 axes[i].text(x1, y1-5, f'{conf:.2f}', color='red')
        
#         axes[i].set_title(f"Image: {img_file}\nGround Truth (green) vs Prediction (red)")
    
#     plt.tight_layout()
#     plt.savefig(os.path.join('/kaggle/working', 'predictions.png')) ###
#     plt.show()


# def prepare_dataset():
#     """
#     Check if the dataset exists and create/fix a proper YAML file for training.
    
#     Returns:
#         str: Path to the YAML file to use for training.
#     """
#     train_images_dir = os.path.join(yolo_dataset_dir, 'images', 'train')
#     val_images_dir = os.path.join(yolo_dataset_dir, 'images', 'val')
#     train_labels_dir = os.path.join(yolo_dataset_dir, 'labels', 'train')
#     val_labels_dir = os.path.join(yolo_dataset_dir, 'labels', 'val')
    
#     print(f"Directory status:")
#     print(f"- Train images exists: {os.path.exists(train_images_dir)}")
#     print(f"- Val images exists: {os.path.exists(val_images_dir)}")
#     print(f"- Train labels exists: {os.path.exists(train_labels_dir)}")
#     print(f"- Val labels exists: {os.path.exists(val_labels_dir)}")
    
#     original_yaml_path = os.path.join(yolo_dataset_dir, 'dataset.yaml')
#     if os.path.exists(original_yaml_path):
#         print(f"Found original dataset.yaml at {original_yaml_path}")
#         return fix_yaml_paths(original_yaml_path)
#     else:
#         print("Original dataset.yaml not found, creating a new one")
#         yaml_data = {
#             'path': yolo_dataset_dir,
#             'train': 'images/train',
#             'val': 'images/train' if not os.path.exists(val_images_dir) else 'images/val',
#             'names': {0: 'motor'}
#         }
#         new_yaml_path = "/kaggle/working/dataset.yaml" ###
#         with open(new_yaml_path, 'w') as f:
#             yaml.dump(yaml_data, f)
#         print(f"Created new YAML at {new_yaml_path}")
#         return new_yaml_path

# def main():
#     print("Starting YOLO training process...")
#     yaml_path = prepare_dataset()
#     print(f"Using YAML file: {yaml_path}")
#     with open(yaml_path, 'r') as f:
#         print(f"YAML contents:\n{f.read()}")
    
#     print("\nStarting YOLO training...")
#     model, results = train_yolo_model(
#         yaml_path,
#         pretrained_weights_path=yolo_pretrained_weights,
#         epochs=100  # For demonstration, using 30 epochs ###
#     )
    
#     print("\nTraining complete!")
#     print("\nRunning predictions on sample images...")
#     predict_on_samples(model, num_samples=4)

# if __name__ == "__main__":
#     main()



# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define paths for the test data and submission
data_path = "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/" ###
test_dir = os.path.join(data_path, "test")
submission_path = "/kaggle/working/submission.csv" ###

# Path to the best trained model (adjust if necessary)
model_path = "/kaggle/input/yolov10b_trust2/pytorch/default/3/best_10m_add_new_dataset.pt" ###

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
    """
    Normalize slice data using the 2nd and 98th percentiles.
    """
    p2 = np.percentile(slice_data, 2)
    p98 = np.percentile(slice_data, 98)
    clipped_data = np.clip(slice_data, p2, p98)
    normalized = 255 * (clipped_data - p2) / (p98 - p2)
    return np.uint8(normalized)

def preload_image_batch(file_paths):
    """Preload a batch of images to CPU memory."""
    images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is None:
            img = np.array(Image.open(path))
        images.append(img)
    return images

def perform_3d_nms(detections, iou_threshold):
    """
    Perform 3D Non-Maximum Suppression on detections to merge nearby motors.
    """
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
    """
    Process a single tomogram and return the most confident motor detection.
    """
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
    """
    Debug function to test image loading methods.
    """
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
    """
    Main function to generate the submission file.
    """
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




