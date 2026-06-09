# !pip download -d ./packages ultralytics
# !tar cfvz archive.tar.gz ./packages
import kagglehub, os
kagglehub.dataset_download('rachiteagles/yolo-pkg')
kagglehub.dataset_download('rachiteagles/yolo-model')


os.environ['WANDB_MODE'] = 'offline'


# !tar xfvz archive.tar.gz
# !pip install --no-index --no-deps /kaggle/input/yolo-pkg/yolo/ultralytics-8.3.112-py3-none-any.whl


!cp -r /kaggle/input/mhaf_yolo/pytorch/default/2/MHAF-YOLO-main/ultralytics /kaggle/working/


class TrainingConfig:
    def __init__(self):
        self.batch_size = 8  # Reduced from original 16 if memory issues
        self.imgsz = 640  # Image size
        self.epochs = 100
        self.patience = 20  # Early stopping patience
        self.device = [0,1] if torch.cuda.device_count() > 1 else 0  # Use all available GPUs
        self.optimizer = 'AdamW'
        self.lr0 = 1e-4  # Initial learning rate
        self.lrf = 0.1  # Final learning rate (lr0 * lrf)
        self.momentum = 0.9
        self.weight_decay = 0.0005
        self.warmup_epochs = 3.0
        self.warmup_momentum = 0.8
        self.warmup_bias_lr = 0.1
        self.box = 7.5  # box loss gain
        self.cls = 0.5  # cls loss gain
        self.dfl = 1.5  # dfl loss gain
        self.hsv_h = 0.015  # image HSV-Hue augmentation (fraction)
        self.hsv_s = 0.7  # image HSV-Saturation augmentation (fraction)
        self.hsv_v = 0.4  # image HSV-Value augmentation (fraction)
        self.degrees = 45.0  # image rotation (+/- deg)
        self.translate = 0.1  # image translation (+/- fraction)
        self.scale = 0.5  # image scale (+/- gain)
        self.shear = 0.0  # image shear (+/- deg)
        self.perspective = 0.0001  # image perspective (+/- fraction), range 0-0.001
        self.flipud = 0.5  # image flip up-down (probability)
        self.fliplr = 0.5  # image flip left-right (probability)
        self.mosaic = 1.0  # image mosaic (probability)
        self.mixup = 0.2  # image mixup (probability)
        self.copy_paste = 0.2  # segment copy-paste (probability)
        self.dropout = 0.1  # use dropout regularization


import os, cv2, torch, yaml, threading, time
import pandas as pd
from tqdm import tqdm
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.transforms import Compose, Resize, ToTensor
from torchvision.utils import make_grid
# from kaggle.input.mhafyolo.pytorch.default.1.MHAF-YOLO-main.ultralytics.models import YOLOv10
from ultralytics.models import YOLOv10,YOLO
from concurrent.futures import ThreadPoolExecutor

np.random.seed(42)
torch.manual_seed(42)


# Define YOLO dataset structure
yolo_dataset_dir = "/kaggle/working/yolo_dataset"
yolo_images_train = os.path.join(yolo_dataset_dir, "images", "train")
yolo_images_val = os.path.join(yolo_dataset_dir, "images", "val")
yolo_labels_train = os.path.join(yolo_dataset_dir, "labels", "train")
yolo_labels_val = os.path.join(yolo_dataset_dir, "labels", "val")

# Create directories
for dir_path in [yolo_images_train, yolo_images_val, yolo_labels_train, yolo_labels_val]:
    os.makedirs(dir_path, exist_ok=True)


# Define global constants for dataset directories
DATA_DIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
TRAIN_CSV = os.path.join(DATA_DIR, 'train_labels.csv')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
MODEL_PATH = "/kaggle/input/yolo-model/best.pt"
# MODEL_PATH = "/kaggle/working/yolo_weights/motor_detector/weights/best.pt"
OUTPUT_DIR = './'
MODEL_DIR = './models'
TRUST = 4
TRAIN_SPLIT = 0.8
BOX_SIZE = 24
DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8 
SUBMISSION_PATH = "/kaggle/working/submission.csv"

# Detection parameters
CONFIDENCE_THRESHOLD = 0.45  # Lower threshold to catch more potential motors
MAX_DETECTIONS_PER_TOMO = 1  # Keep track of top N detections per tomogram
NMS_IOU_THRESHOLD = 0.2  # Non-maximum suppression threshold for 3D clustering
CONCENTRATION = 1 # ONLY PROCESS 1/20 slices for fast submission


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



def prepare_yolo_dataset(trust=TRUST, train_split=TRAIN_SPLIT):
    """
    Extract slices containing motors from tomograms and save to YOLO structure with annotations
    """
    # Load the labels CSV
    labels_df = pd.read_csv(TRAIN_CSV)
    
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
                src_path = os.path.join(TRAIN_DIR, tomo_id, slice_filename)
                
                if not os.path.exists(src_path):
                    print(f"Warning: {src_path} does not exist, skipping.")
                    continue
                
                # Load and normalize the slice
                img = cv2.imread(src_path)

                # Get image dimensions
                img_height,img_width,_ = img.shape

                # for angle in [0, 90, 180, 270]:
                # rotated_img, new_x, new_y = rotate_image_and_coords(
                #         img, angle, x_center, y_center, img_width, img_height
                #     )
                img_array = np.array(img)
                
                # Normalize the image
                normalized_img = normalize_slice(img_array)
                
                # Create destination filename (with unique identifier)
                dest_filename = f"{tomo_id}_z{z:04d}_y{y_center:04d}_x{x_center:04d}.jpg"
                dest_path = os.path.join(images_dir, dest_filename)
                
                # Save the normalized image
                Image.fromarray(normalized_img).save(dest_path)
                
                
                
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
    
    # Process validation tomograms
    val_slices, val_motors = process_tomogram_set(val_tomos, yolo_images_val, yolo_labels_val, "validation")
    
    # Create YAML configuration file for YOLO
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
        


# summary = prepare_yolo_dataset(TRUST)


# summary


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


def train_yolo_model(yaml_path, pretrained_weights_path, epochs=30, batch_size=16, img_size=640):
    """
    Train a YOLO model on the prepared dataset
    
    Args:
        yaml_path (str): Path to the dataset YAML file
        pretrained_weights_path (str): Path to pre-downloaded weights file
        epochs (int): Number of training epochs
        batch_size (int): Batch size for training
        img_size (int): Image size for training
    """
    print(f"Loading pre-trained weights from: {pretrained_weights_path}")
    
    # Load a pre-trained YOLOv8 model
    model = YOLOv10(pretrained_weights_path)
    
    # Train the model with early stopping
    results = model.train(
        data=yaml_path,
        epochs=cfg.epochs,
        batch=cfg.batch_size,
        imgsz=cfg.imgsz,
        optimizer=cfg.optimizer,
        lr0=cfg.lr0,
        lrf=cfg.lrf,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        warmup_epochs=cfg.warmup_epochs,
        warmup_momentum=cfg.warmup_momentum,
        warmup_bias_lr=cfg.warmup_bias_lr,
        box=cfg.box,
        cls=cfg.cls,
        dfl=cfg.dfl,
        hsv_h=cfg.hsv_h,
        hsv_s=cfg.hsv_s,
        hsv_v=cfg.hsv_v,
        degrees=cfg.degrees,
        translate=cfg.translate,
        scale=cfg.scale,
        shear=cfg.shear,
        perspective=cfg.perspective,
        flipud=cfg.flipud,
        fliplr=cfg.fliplr,
        mosaic=cfg.mosaic,
        mixup=cfg.mixup,
        copy_paste=cfg.copy_paste,
        dropout=cfg.dropout,
        device=cfg.device,
        cos_lr= True,
        multi_scale= True,
        amp =True,
        project=yolo_weights_dir,
        save_dir='/kaggle/working',
        name='motor_detector',
        # exist_ok=True,
        patience=cfg.patience,
        save_period=5,  # Save checkpoints every 5 epochs
        val=True,  # Ensure validation is performed
        verbose=True,
        
    )
    
    # Get the path to the run directory
    run_dir = os.path.join(yolo_weights_dir, 'motor_detector')
    
    # Plot and save the loss curve
    best_epoch_info = plot_dfl_loss_curve(run_dir)
    
    if best_epoch_info:
        best_epoch, best_val_loss = best_epoch_info
        print(f"\nBest model found at epoch {best_epoch} with validation DFL loss: {best_val_loss:.4f}")
    
    return model, results


yaml_path = '/kaggle/working/yolo_dataset/dataset.yaml'
yolo_pretrained_weights = '/kaggle/input/mhaf_yolon/pytorch/default/1/MAF-YOLOv2-N.pt'
yolo_weights_dir = "yolo_weights"


cfg = TrainingConfig()
# model, results = train_yolo_model(
#         yaml_path,
#         pretrained_weights_path=yolo_pretrained_weights,
#         epochs=40  # Using 30 epochs instead of 100 for faster training
#     )


def debug_image_loading(tomo_id):
    """
    Debug function to check image loading
    """
    tomo_dir = os.path.join(TEST_DIR, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    
    if not slice_files:
        print(f"No image files found in {tomo_dir}")
        return
        
    print(f"Found {len(slice_files)} image files in {tomo_dir}")
    sample_file = slice_files[len(slice_files)//2]  # Middle slice
    img_path = os.path.join(tomo_dir, sample_file)
    
    # Try different loading methods
    try:
        # Method 1: PIL
        img_pil = Image.open(img_path)
        img_array_pil = np.array(img_pil)
        print(f"PIL Image shape: {img_array_pil.shape}, dtype: {img_array_pil.dtype}")
        
        # Method 2: OpenCV
        img_cv2 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        print(f"OpenCV Image shape: {img_cv2.shape}, dtype: {img_cv2.dtype}")
        
        # Method 3: Convert to RGB
        img_rgb = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        print(f"OpenCV RGB Image shape: {img_rgb.shape}, dtype: {img_rgb.dtype}")
        
        print("Image loading successful!")
    except Exception as e:
        print(f"Error loading image {img_path}: {e}")
        
    # Also test with YOLO's built-in loader
    try:
        test_model = YOLO(MODEL_PATH)
        test_results = test_model([img_path], verbose=False)
        print("YOLO model successfully processed the test image")
    except Exception as e:
        print(f"Error with YOLO processing: {e}")


def preload_image_batch(file_paths):
    """Preload a batch of images to CPU memory"""
    images = []
    for path in file_paths:
        img = cv2.imread(path)
        if img is None:
            # Try with PIL as fallback
            img = np.array(Image.open(path))
        images.append(img)
    return images


# GPU profiling context manager
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


def perform_3d_nms(detections, iou_threshold):
    """
    Perform 3D Non-Maximum Suppression on detections to merge nearby motors
    """
    if not detections:
        return []
    
    # Sort by confidence (highest first)
    detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)
    
    # List to store final detections after NMS
    final_detections = []
    
    # Define 3D distance function
    def distance_3d(d1, d2):
        return np.sqrt((d1['z'] - d2['z'])**2 + 
                       (d1['y'] - d2['y'])**2 + 
                       (d1['x'] - d2['x'])**2)
    
    # Maximum distance threshold (based on box size and slice gap)
    box_size = 24  # Same as annotation box size
    distance_threshold = box_size * iou_threshold
    
    # Process each detection
    while detections:
        # Take the detection with highest confidence
        best_detection = detections.pop(0)
        final_detections.append(best_detection)
        
        # Filter out detections that are too close to the best detection
        detections = [d for d in detections if distance_3d(d, best_detection) > distance_threshold]
    
    return final_detections


BATCH_SIZE





def process_tomogram(tomo_id, model, index=0, total=1):
    """
    Process a single tomogram and return the most confident motor detection
    """
    print(f"Processing tomogram {tomo_id} ({index}/{total})")
    
    # Get all slice files for this tomogram
    tomo_dir = os.path.join(TEST_DIR, tomo_id)
    slice_files = sorted([f for f in os.listdir(tomo_dir) if f.endswith('.jpg')])
    
    # Apply CONCENTRATION to reduce the number of slices processed
    # This will process approximately CONCENTRATION fraction of all slices
    selected_indices = np.linspace(0, len(slice_files)-1, int(len(slice_files) * CONCENTRATION))
    selected_indices = np.round(selected_indices).astype(int)
    slice_files = [slice_files[i] for i in selected_indices]
    
    print(f"Processing {len(slice_files)} out of {len(os.listdir(tomo_dir))} slices based on CONCENTRATION={CONCENTRATION}")
    
    # Create a list to store all detections
    all_detections = []
    
    # Create CUDA streams for parallel processing if using GPU
    if DEVICE.startswith('cuda'):
        streams = [torch.cuda.Stream() for _ in range(min(4, BATCH_SIZE))]
    else:
        streams = [None]
    
    # Variables for preloading
    next_batch_thread = None
    next_batch_images = None
    
    # Process slices in batches
    for batch_start in range(0, len(slice_files), BATCH_SIZE):
        # Wait for previous preload thread if it exists
        if next_batch_thread is not None:
            next_batch_thread.join()
            next_batch_images = None
            
        batch_end = min(batch_start + BATCH_SIZE, len(slice_files))
        batch_files = slice_files[batch_start:batch_end]
        
        # Start preloading next batch
        next_batch_start = batch_end
        next_batch_end = min(next_batch_start + BATCH_SIZE, len(slice_files))
        next_batch_files = slice_files[next_batch_start:next_batch_end] if next_batch_start < len(slice_files) else []
        
        if next_batch_files:
            next_batch_paths = [os.path.join(tomo_dir, f) for f in next_batch_files]
            next_batch_thread = threading.Thread(target=preload_image_batch, args=(next_batch_paths,))
            next_batch_thread.start()
        else:
            next_batch_thread = None
        
        # Split batch across streams for parallel processing
        sub_batches = np.array_split(batch_files, len(streams))
        sub_batch_results = []
        
        for i, sub_batch in enumerate(sub_batches):
            if len(sub_batch) == 0:
                continue
                
            stream = streams[i % len(streams)]
            with torch.cuda.stream(stream) if stream and DEVICE.startswith('cuda') else nullcontext():
                # Process sub-batch
                sub_batch_paths = [os.path.join(tomo_dir, slice_file) for slice_file in sub_batch]
                sub_batch_slice_nums = [int(slice_file.split('_')[1].split('.')[0]) for slice_file in sub_batch]
                
                # Run inference with profiling
                with GPUProfiler(f"Inference batch {i+1}/{len(sub_batches)}"):
                    sub_results = model(sub_batch_paths, verbose=False)
                
                # Process each result in this sub-batch
                for j, result in enumerate(sub_results):
                    if len(result.boxes) > 0:
                        boxes = result.boxes
                        for box_idx, confidence in enumerate(boxes.conf):
                            if confidence >= CONFIDENCE_THRESHOLD:
                                # Get bounding box coordinates
                                x1, y1, x2, y2 = boxes.xyxy[box_idx].cpu().numpy()
                                
                                # Calculate center coordinates
                                x_center = (x1 + x2) / 2
                                y_center = (y1 + y2) / 2
                                
                                # Store detection with 3D coordinates
                                all_detections.append({
                                    'z': round(sub_batch_slice_nums[j]),
                                    'y': round(y_center),
                                    'x': round(x_center),
                                    'confidence': float(confidence)
                                })
        
        # Synchronize streams
        if DEVICE.startswith('cuda'):
            torch.cuda.synchronize()
    
    # Clean up thread if still running
    if next_batch_thread is not None:
        next_batch_thread.join()
    
    # 3D Non-Maximum Suppression to merge nearby detections across slices
    final_detections = perform_3d_nms(all_detections, NMS_IOU_THRESHOLD)
    
    # Sort detections by confidence (highest first)
    final_detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # If there are no detections, return NA values
    if not final_detections:
        return {
            'tomo_id': tomo_id,
            'Motor axis 0': -1,
            'Motor axis 1': -1,
            'Motor axis 2': -1
        }
    
    # Take the detection with highest confidence
    best_detection = final_detections[0]
    
    # Return result with integer coordinates
    return {
        'tomo_id': tomo_id,
        'Motor axis 0': round(best_detection['z']),
        'Motor axis 1': round(best_detection['y']),
        'Motor axis 2': round(best_detection['x'])
    }



def generate_submission():
    """
    Main function to generate the submission file
    """
    # Get list of test tomograms
    test_tomos = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
    total_tomos = len(test_tomos)
    
    print(f"Found {total_tomos} tomograms in test directory")
    
    # Debug image loading for the first tomogram
    if test_tomos:
        debug_image_loading(test_tomos[0])
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Initialize model once outside the processing loop
    print(f"Loading YOLO model from {MODEL_PATH}")
    model = YOLO(MODEL_PATH)
    model.to(DEVICE)
    
    # Additional optimizations for inference
    if DEVICE.startswith('cuda'):
        # Fuse conv and bn layers for faster inference
        model.fuse()
        
        # Enable model half precision (FP16) if on compatible GPU
        if torch.cuda.get_device_capability(0)[0] >= 7:  # Volta or newer
            model.model.half()
            print("Using half precision (FP16) for inference")
    
    # Process tomograms with parallelization
    results = []
    motors_found = 0
    
    # Using ThreadPoolExecutor with max_workers=1 since each worker uses the GPU already
    # and we're parallelizing within each tomogram processing
    with ThreadPoolExecutor(max_workers=1) as executor:
        future_to_tomo = {}
        
        # Submit all tomograms for processing
        for i, tomo_id in enumerate(test_tomos, 1):
            future = executor.submit(process_tomogram, tomo_id, model, i, total_tomos)
            future_to_tomo[future] = tomo_id
        
        # Process completed futures as they complete
        for future in future_to_tomo:
            tomo_id = future_to_tomo[future]
            try:
                # Clear CUDA cache between tomograms
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
                result = future.result()
                results.append(result)
                
                # Update motors found count
                has_motor = not pd.isna(result['Motor axis 0'])
                if has_motor:
                    motors_found += 1
                    print(f"Motor found in {tomo_id} at position: "
                          f"z={result['Motor axis 0']}, y={result['Motor axis 1']}, x={result['Motor axis 2']}")
                else:
                    print(f"No motor detected in {tomo_id}")
                    
                print(f"Current detection rate: {motors_found}/{len(results)} ({motors_found/len(results)*100:.1f}%)")
            
            except Exception as e:
                print(f"Error processing {tomo_id}: {e}")
                # Create a default entry for failed tomograms
                results.append({
                    'tomo_id': tomo_id,
                    'Motor axis 0': -1,
                    'Motor axis 1': -1,
                    'Motor axis 2': -1
                })
    
    # Create submission dataframe
    submission_df = pd.DataFrame(results)
    
    # Ensure proper column order
    submission_df = submission_df[['tomo_id', 'Motor axis 0', 'Motor axis 1', 'Motor axis 2']]
    
    # Save the submission file
    submission_df.to_csv(SUBMISSION_PATH, index=False)
    
    print(f"\nSubmission complete!")
    print(f"Motors detected: {motors_found}/{total_tomos} ({motors_found/total_tomos*100:.1f}%)")
    print(f"Submission saved to: {SUBMISSION_PATH}")
    
    # Display first few rows of submission
    print("\nSubmission preview:")
    print(submission_df.head())
    
    return submission_df


generate_submission()




























