# Install required packages
!pip install retrying monai scipy scikit-image wandb imageio gcsfs

# Import libraries
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from monai.networks.nets import UNet
from monai.losses import DiceLoss
import torch.optim as optim
from scipy.ndimage import gaussian_filter, center_of_mass
from scipy.signal import find_peaks
import sklearn.metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob
from IPython.display import Video, display
import wandb
import time
import shutil
import gcsfs  # Dataset is too big for kaggle - After multiple attempts of uploading, I failed miserably. 
import gc
import threading
import logging
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
from retrying import retry
from concurrent.futures import ThreadPoolExecutor, as_completed
import torch.nn.functional as F  # Add this import
# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)


# Ensure the output directory exists
os.makedirs('/kaggle/working', exist_ok=True)

# Configure root logger
logger = logging.getLogger('')
logger.setLevel(logging.INFO)  # Capture INFO and above

# Clear any existing handlers to avoid conflicts
logger.handlers = []

# File handler for detailed logs (INFO, WARNING, ERROR)
file_handler = logging.FileHandler("/kaggle/working/training.log", mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Console handler for errors only
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(logging.Formatter('%(levelname)s - %(message)s'))
logger.addHandler(console_handler)


# Initialize wandb
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
wb_token = user_secrets.get_secret("WANDB")
wandb.login(key=wb_token)
wandb.init(
    project="byu-bacterial-flagellar-motors",
    config={
        "learning_rate": 1e-3,
        "epochs": 50,
        "batch_size": 4,
        "patch_size": (128, 128, 128),
        "gaussian_sigma": 5,
        "architecture": "3D U-Net",
        "optimizer": "Adam",
        "loss_function": "DiceLoss",
        "beta": 2
    }
)


fs = gcsfs.GCSFileSystem(token="anon") # Initialize GCS filesystem
logger.info("Initialized GCS filesystem with anonymous access")

# Define GCS path and local directory
gcs_precomputed_path = "gs://nb153/precomputedmasks"
gcs_preprocessed_path = "gs://nb153/preprocessed"
local_dir = "/kaggle/working/data"
os.makedirs(local_dir, exist_ok=True)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 1. Analyze dataset to identify tomograms with motors
def identify_motor_tomograms(labels_df):
    """Identify tomograms with valid motor annotations."""
    motor_tomograms = []
    for tomo_id in labels_df["tomo_id"].unique():
        tomo_labels = labels_df[labels_df["tomo_id"] == tomo_id].iloc[0]
        if tomo_labels["Number of motors"] > 0 and tomo_labels["Motor axis 0"] != -1:
            motor_tomograms.append(tomo_id)
    logger.info(f"Found {len(motor_tomograms)} tomograms with motors")
    return motor_tomograms

# Load labels and identify motor tomograms
labels_df = pd.read_csv("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv")
tomo_ids = sorted(labels_df["tomo_id"].unique())
motor_tomo_ids = identify_motor_tomograms(labels_df)
logger.info(f"Total tomograms: {len(tomo_ids)}, Motor tomograms: {len(motor_tomo_ids)}")

# Split into train/val/test (80/10/10) using only motor tomograms for training
train_val_ids, _ = train_test_split(tomo_ids, test_size=0.1, random_state=42)
train_ids, val_ids = train_test_split(train_val_ids, test_size=0.1111, random_state=42)
train_ids = [tid for tid in train_ids if tid in motor_tomo_ids]
test_ids = ["tomo_00e047", "tomo_01a877"]  # tomo_003acc Restored all test tomograms
logger.info(f"Train IDs: {len(train_ids)}, Val IDs: {len(val_ids)}, Test IDs: {len(test_ids)}")
logger.info(f"Test IDs: {test_ids}")

# Analyze tomograms
def analyze_tomograms(labels_df):
    total_tomograms = len(labels_df["tomo_id"].unique())
    motor_tomograms = len(labels_df[labels_df["Number of motors"] > 0])
    non_motor_tomograms = total_tomograms - motor_tomograms
    logger.info(f"Total tomograms: {total_tomograms}")
    logger.info(f"Tomograms with motors: {motor_tomograms}")
    logger.info(f"Tomograms without motors: {non_motor_tomograms}")
    return total_tomograms, motor_tomograms, non_motor_tomograms

# Plot tomogram distribution
# def plot_tomogram_distribution(labels_df):
#     total_tomograms = len(labels_df["tomo_id"].unique())
#     motor_tomograms = len(labels_df[labels_df["Number of motors"] > 0])
#     non_motor_tomograms = total_tomograms - motor_tomograms
#     plt.figure(figsize=(8, 5))
#     plt.bar(["Total", "With Motors", "Without Motors"], [total_tomograms, motor_tomograms, non_motor_tomograms], color=["blue", "green", "red"])
#     plt.title("Tomogram Distribution")
#     plt.ylabel("Count")
#     for i, count in enumerate([total_tomograms, motor_tomograms, non_motor_tomograms]):
#         plt.text(i, count + 0.5, str(count), ha="center")
#     plt.show()
#     plt.close()

# Download functions with simultaneous downloading

# Optimized download function with robust error handling
@retry(stop_max_attempt_number=3, wait_fixed=2000)
def download_file(gcs_path, local_path):
    if not os.path.exists(local_path):
        try:
            fs.get(gcs_path, local_path)
            logger.info(f"Downloaded {gcs_path} to {local_path}")
        except Exception as e:
            if "404" in str(e):
                logger.warning(f"GCS file not found: {gcs_path}")
                raise FileNotFoundError(f"GCS file not found: {gcs_path}")
            else:
                raise
    return local_path


def download_npy_and_mask(tomo_id, gcs_preprocessed_path, gcs_precomputed_path, split, local_dir):
    """Download tomogram and mask in parallel with retry."""
    tomo_path = os.path.join(local_dir, f"{tomo_id}.npy")
    mask_path = os.path.join(local_dir, f"{tomo_id}_mask.npy")
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            npy_future = executor.submit(
                download_file,
                f"{gcs_preprocessed_path}/{split}/{tomo_id}/{tomo_id}.npy",
                tomo_path
            )
            mask_future = executor.submit(
                download_file,
                f"{gcs_precomputed_path}/{split}/{tomo_id}_mask.npy",
                mask_path
            )
            tomo_path = npy_future.result()
            mask_path = mask_future.result()
        logger.info(f"Completed parallel download for tomogram {tomo_id}")
        return tomo_path, mask_path
    except FileNotFoundError as e:
        logger.error(f"Skipping tomogram {tomo_id}: {str(e)}")
        return None, None
    except Exception as e:
        logger.error(f"Error downloading tomogram {tomo_id}: {str(e)}")
        raise


# Combined download and preprocess function
def download_and_preprocess(tomo_id, gcs_preprocessed_path, gcs_precomputed_path, split, local_dir, labels_df, patches_per_volume):
    try:
        tomo_path, mask_path = download_npy_and_mask(tomo_id, gcs_preprocessed_path, gcs_precomputed_path, split, local_dir)
        volume = np.load(tomo_path)
        mask = np.load(mask_path)
        tomo_labels = labels_df[labels_df["tomo_id"] == tomo_id].iloc[0]
        num_motors = tomo_labels["Number of motors"]
        logger.info(f"Tomogram {tomo_id} has {num_motors} motors")
        patches, mask_patches = sample_patches(tomo_id, volume, mask, labels_df, patches_per_volume=patches_per_volume)
        del volume, mask
        logger.info(f"Prepared patches for tomogram {tomo_id}")
        return tomo_id, patches, mask_patches
    except Exception as e:
        logger.error(f"Error processing tomogram {tomo_id}: {str(e)}")
        raise

# Sample patches
def sample_patches(tomo_id, volume, mask, labels_df, patch_size=(128, 128, 128), patches_per_volume=128):  # Increased patches
    shape = volume.shape
    patches = []
    mask_patches = []
    tomo_labels = labels_df[labels_df["tomo_id"] == tomo_id]
    motor_coords = []
    for _, row in tomo_labels.iterrows():
        if row["Number of motors"] > 0 and row["Motor axis 0"] != -1:
            z, y, x = int(row["Motor axis 0"]), int(row["Motor axis 1"]), int(row["Motor axis 2"])
            if 0 <= z < shape[0] and 0 <= y < shape[1] and 0 <= x < shape[2]:
                motor_coords.append((z, y, x))
    
    for _ in range(patches_per_volume // 2):
        if motor_coords:
            zc, yc, xc = motor_coords[np.random.randint(len(motor_coords))]
            z = np.clip(zc - patch_size[0]//2 + np.random.randint(-32, 32), 0, shape[0] - patch_size[0])
            y = np.clip(yc - patch_size[1]//2 + np.random.randint(-32, 32), 0, shape[1] - patch_size[1])
            x = np.clip(xc - patch_size[2]//2 + np.random.randint(-32, 32), 0, shape[2] - patch_size[2])
        else:
            z = np.random.randint(0, max(1, shape[0] - patch_size[0]))
            y = np.random.randint(0, max(1, shape[1] - patch_size[1]))
            x = np.random.randint(0, max(1, shape[2] - patch_size[2]))
        patch = volume[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]][np.newaxis, ...]
        mask_patch = mask[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]][np.newaxis, ...]
        patches.append(patch)
        mask_patches.append(mask_patch)
    
    for _ in range(patches_per_volume // 2):
        z = np.random.randint(0, max(1, shape[0] - patch_size[0]))
        y = np.random.randint(0, max(1, shape[1] - patch_size[1]))
        x = np.random.randint(0, max(1, shape[2] - patch_size[2]))
        patch = volume[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]][np.newaxis, ...]
        mask_patch = mask[z:z+patch_size[0], y:y+patch_size[1], x:x+patch_size[2]][np.newaxis, ...]
        patches.append(patch)
        mask_patches.append(mask_patch)
    
    logger.info(f"Sampled {len(patches)} patches for tomogram {tomo_id}")
    return np.array(patches), np.array(mask_patches)

# Patch dataset
class PatchDataset(Dataset):
    def __init__(self, patches, mask_patches):
        self.patches = patches
        self.mask_patches = mask_patches
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        patch = self.patches[idx]
        mask_patch = self.mask_patches[idx]
        return torch.tensor(patch, dtype=torch.float32), torch.tensor(mask_patch, dtype=torch.float32)

# Tomogram dataset
class TomogramDataset(Dataset):
    def __init__(self, tomo_id, gcs_preprocessed_path, local_dir, mode="test"):
        self.tomo_id = tomo_id
        self.gcs_preprocessed_path = gcs_preprocessed_path
        self.gcs_precomputed_path = gcs_preprocessed_path.replace("preprocessed", "precomputedmasks")
        self.local_dir = local_dir
        self.mode = mode
        self.volume = None
    
    def load(self):
        tomo_path, mask_path = download_npy_and_mask(
            self.tomo_id, 
            self.gcs_preprocessed_path, 
            self.gcs_precomputed_path, 
            "train" if self.mode == "val" else self.mode, 
            self.local_dir
        )
        self.volume = np.load(tomo_path)
        logger.info(f"Loaded tomogram {self.tomo_id} (shape: {self.volume.shape})")
        if os.path.exists(mask_path):
            self.mask = np.load(mask_path)
            logger.info(f"Loaded mask {self.tomo_id} (min/max: {self.mask.min()}/{self.mask.max()})")
        else: 
            self.mask = np.zeros_like(self.volume)
            logger.info(f"No mask found for {self.tomo_id}, using zeros (shape: {self.mask.shape})")
    
    def clear(self):
        tomo_path = os.path.join(self.local_dir, f"{self.tomo_id}.npy")
        mask_path = os.path.join(self.local_dir, f"{self.tomo_id}_mask.npy")
        for path in [tomo_path, mask_path]:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Deleted {path}")
        if self.volume is not None:
            del self.volume
            self.volume = None
        if self.mask is not None:
            del self.mask
            self.mask = None
        gc.collect()
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        if self.volume is None:
            self.load()
        volume = torch.from_numpy(self.volume).float()
        if len(volume.shape) == 3:  # (z, y, x) -> (1, 1, z, y, x)
            volume = volume.unsqueeze(0).unsqueeze(0)
        elif len(volume.shape) != 5:
            raise ValueError(f"Unexpected volume shape: {volume.shape}")
        mask = torch.from_numpy(self.mask).float() if self.mask is not None else torch.zeros_like(volume)
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(0).unsqueeze(0)
        elif len(mask.shape) != 5 and self.mask is not None:
            raise ValueError(f"Unexpected mask shape: {mask.shape}")
        return volume, mask


# Initialize model
model = UNet(
    spatial_dims=3,
    in_channels=1,
    out_channels=1,
    channels=(16, 32, 64, 128, 256),
    strides=(2, 2, 2, 2),
    num_res_units=2,
)
if torch.cuda.device_count() > 1:
    logger.info(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
    model = torch.nn.DataParallel(model)
model = model.to(device)
logger.info(f"Model device: {next(model.parameters()).device}")
logger.info(f"GPU count: {torch.cuda.device_count()}, Device: {device}")

# Loss and optimizer
criterion = DiceLoss(sigmoid=True)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.1, patience=5)

# Load checkpoint if available
start_epoch = 0
best_val_loss = float("inf")
train_losses = []
val_losses = []
if os.path.exists("checkpoint.pth"):
    checkpoint = torch.load("checkpoint.pth", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    best_val_loss = checkpoint['best_val_loss']
    start_epoch = checkpoint['epoch'] + 1
    logger.info(f"Resumed from epoch {start_epoch}, best validation loss: {best_val_loss}")
else:
    logger.info("No checkpoint found, starting from scratch.")


# Clean memory after processing tomograms
def clean_memory(tomo_ids, local_dir):
    """Delete tomogram and mask files in parallel and clear GPU memory."""
    def delete_file(path):
        """Helper function to delete a single file."""
        try:
            if os.path.exists(path):
                os.remove(path)
                logger.info(f"Deleted {path}")
        except Exception as e:
            logger.error(f"Error deleting {path}: {str(e)}")

    file_paths = []
    for tomo_id in tomo_ids:
        tomo_path = os.path.join(local_dir, f"{tomo_id}.npy")
        mask_path = os.path.join(local_dir, f"{tomo_id}_mask.npy")
        file_paths.extend([tomo_path, mask_path])

    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(delete_file, file_paths)

    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Cleared memory and GPU cache")

# Training function
def train_epoch(model, loader, criterion, optimizer, epoch, start_epoch, tomo_id):
    model.train()
    epoch_loss = 0.0
    for i, (inputs, targets) in enumerate(tqdm(loader, desc=f"Training tomo {tomo_id}", file=sys.stdout, disable=True)):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    avg_loss = epoch_loss / len(loader)
    logger.info(f"Epoch {epoch+1}, Tomo {tomo_id} completed, Average Loss: {avg_loss:.4f}")
    return avg_loss

# Cache validation tomograms
def cache_validation_tomograms(val_ids, gcs_preprocessed_path, gcs_precomputed_path, local_dir):
    for tomo_id in val_ids:  # Use all val_ids
        tomo_path, mask_path = download_npy_and_mask(tomo_id, gcs_preprocessed_path, gcs_precomputed_path, "train", local_dir)
        logger.info(f"Cached {tomo_id} at {tomo_path}, {mask_path}")
    return val_ids

# Parallel data loading for validation
def load_tomogram_for_validation(tomo_id, gcs_preprocessed_path, gcs_precomputed_path, local_dir, labels_df, patches_per_volume):
    try:
        tomo_path = os.path.join(local_dir, f"{tomo_id}.npy")
        mask_path = os.path.join(local_dir, f"{tomo_id}_mask.npy")
        if not (os.path.exists(tomo_path) and os.path.exists(mask_path)):
            tomo_path, mask_path = download_npy_and_mask(tomo_id, gcs_preprocessed_path, gcs_precomputed_path, "train", local_dir)
        volume = np.load(tomo_path)
        mask = np.load(mask_path)
        patches, mask_patches = sample_patches(tomo_id, volume, mask, labels_df, patches_per_volume=patches_per_volume)
        logger.info(f"Loaded and sampled patches for validation tomogram {tomo_id}")
        return tomo_id, patches, mask_patches, [tomo_path, mask_path]
    except Exception as e:
        logger.error(f"Error loading tomogram {tomo_id}: {str(e)}")
        raise

# Optimized validate function
def validate(model, val_ids, gcs_preprocessed_path, gcs_precomputed_path, local_dir, labels_df, criterion, patch_size):
    model.eval()
    epoch_loss = 0.0
    patches_per_volume = 128  # Increased to match sample_patches
    selected_val_ids = cache_validation_tomograms(val_ids, gcs_preprocessed_path, gcs_precomputed_path, local_dir)
    
    max_workers = min(os.cpu_count(), 8)
    logger.info(f"Using {max_workers} workers for validation data loading")
    
    # Preload all tomograms in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_tomo = {
            executor.submit(
                load_tomogram_for_validation,
                tomo_id,
                gcs_preprocessed_path,
                gcs_precomputed_path,
                local_dir,
                labels_df,
                patches_per_volume
            ): tomo_id for tomo_id in selected_val_ids
        }
        
        results = []
        for future in tqdm(as_completed(future_to_tomo), total=len(selected_val_ids), desc="Loading validation tomograms"):
            tomo_id = future_to_tomo[future]
            try:
                results.append(future.result())
            except Exception as e:
                logger.error(f"Error loading tomogram {tomo_id}: {str(e)}")
                raise

    # Process loaded tomograms
    tomo_ids_processed = []
    for tomo_id, patches, mask_patches, paths in tqdm(results, total=len(selected_val_ids), desc="Validating tomograms"):
        logger.info(f"Validating tomogram {tomo_id}")
        try:
            tomo_labels = labels_df[labels_df["tomo_id"] == tomo_id].iloc[0]
            num_motors = tomo_labels["Number of motors"]
            logger.info(f"Tomogram {tomo_id} has {num_motors} motors")
            
            dataset = PatchDataset(patches, mask_patches)
            loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=max_workers, pin_memory=True)
            
            with torch.no_grad():
                for i, (inputs, targets) in enumerate(loader):
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    epoch_loss += loss.item()
                    logger.info(f"Validation tomo {tomo_id}, Batch {i+1}, Loss: {loss.item():.4f}")
            
            del dataset, loader, patches, mask_patches
            tomo_ids_processed.append(tomo_id)
            clean_memory([tomo_id], local_dir)
        except Exception as e:
            logger.error(f"Error validating tomogram {tomo_id}: {str(e)}")
            raise
    
    # # Clean up all tomograms at once
    # if tomo_ids_processed:
    #     logger.info(f"Cleaning memory for tomograms: {tomo_ids_processed}")
    #     clean_memory(tomo_ids_processed, local_dir)
    
    return epoch_loss / (len(selected_val_ids) * patches_per_volume)


# Prefetch thread for producer-consumer queue
data_queue = queue.Queue(maxsize=2)
def prefetch_batches(train_ids, batch_size, gcs_preprocessed_path, gcs_precomputed_path, local_dir, labels_df, patches_per_volume):
    for batch_start in range(0, len(train_ids), batch_size):
        batch_tomo_ids = train_ids[batch_start:batch_start + batch_size]
        logger.info(f"Prefetching batch of tomograms: {batch_tomo_ids}")
        batch_data = []
        download_start_time = time.time()
        try:
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                future_to_tomo = {
                    executor.submit(
                        download_and_preprocess,
                        tomo_id,
                        gcs_preprocessed_path,
                        gcs_precomputed_path,
                        "train",
                        local_dir,
                        labels_df,
                        patches_per_volume
                    ): tomo_id for tomo_id in batch_tomo_ids
                }
                for future in as_completed(future_to_tomo):
                    batch_data.append(future.result())
            logger.info(f"Batch prefetch time for {batch_tomo_ids}: {time.time() - download_start_time:.2f} seconds")
            data_queue.put((batch_tomo_ids, batch_data))
        except Exception as e:
            logger.error(f"Error prefetching batch {batch_tomo_ids}: {str(e)}")
            raise


# Training loop with batch processing
num_epochs = 10  # Increased for convergence
patience = 10
trigger_times = 0
patches_per_volume_train = 128  # Increased to match sample_patches
batch_size = 3  # Kept as is
metrics_log = []
best_val_loss = float("inf")
train_losses = []
val_losses = []

# Analyze tomograms (no plotting)
total_tomograms, motor_tomograms, non_motor_tomograms = analyze_tomograms(labels_df)

for epoch in range(start_epoch, num_epochs):
    logger.info(f"Starting training - Epoch {epoch+1}/{num_epochs}")
    epoch_train_loss = 0.0
    processed_tomograms = 0
    total_train_tomograms = len(train_ids)  # Use all train_ids

    # Start prefetch thread
    prefetch_thread = threading.Thread(
        target=prefetch_batches,
        args=(train_ids, batch_size, gcs_preprocessed_path, gcs_precomputed_path, local_dir, labels_df, patches_per_volume_train)
    )
    prefetch_thread.start()

    # Process tomograms in batches
    for batch_start in range(0, len(train_ids), batch_size):
        batch_tomo_ids, batch_data = data_queue.get()
        logger.info(f"Processing batch of tomograms: {batch_tomo_ids}")

        # Train on batch
        for tomo_id, patches, mask_patches in batch_data:
            try:
                dataset = PatchDataset(patches, mask_patches)
                loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
                logger.info(f"Training on tomogram {tomo_id}")
                batch_loss = train_epoch(model, loader, criterion, optimizer, epoch, start_epoch, tomo_id)
                epoch_train_loss += batch_loss
                processed_tomograms += 1
                logger.info(f"Completed training on tomogram {tomo_id}, Loss: {batch_loss:.4f}")
                logger.info(f"Processed {processed_tomograms}/{total_train_tomograms} tomograms")
                del dataset, loader, patches, mask_patches
                clean_memory([tomo_id], local_dir)  # Clean after each tomogram
            except Exception as e:
                logger.error(f"Error training tomogram {tomo_id}: {str(e)}")
                raise

    prefetch_thread.join()

    train_loss = epoch_train_loss / max(1, processed_tomograms)
    train_losses.append(train_loss)

    # Validation
    logger.info("Starting validation")
    patch_size = (128, 128, 128)
    val_loss = validate(model, val_ids, gcs_preprocessed_path, gcs_precomputed_path, local_dir, labels_df, criterion, patch_size)
    val_losses.append(val_loss)
    
    # Log metrics
    logger.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    metrics_log.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "val_loss": val_loss
    })
    
    # Save metrics to file
    pd.DataFrame(metrics_log).to_csv("/kaggle/working/training_metrics.csv", index=False)
    logger.info("Saved training metrics to training_metrics.csv")

    scheduler.step(val_loss)

    # Checkpointing and Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }, "checkpoint.pth")
        torch.save(model.state_dict(), "/kaggle/working/best_model.pth")
        logger.info("Saved checkpoint and best model")
    else:
        trigger_times += 1
        if trigger_times >= patience:
            logger.info("Early stopping triggered!")
            break


# 1 Hyperparameter Tuning (Peak Detection Threshold)
# 2 Predict on the validation set,
# 3 Extract motor locations using peak detection
# 4 Tune the threshold to maximize the Fβ-score (β=2).
# 5 Using competition's metric

# Metric implementation
def distance_metric(solution, submission, thresh_ratio, min_radius):
    coordinate_cols = ['Motor axis 0', 'Motor axis 1', 'Motor axis 2']
    label_tensor = solution[coordinate_cols].values.reshape(len(solution), -1, len(coordinate_cols))
    predicted_tensor = submission[coordinate_cols].values.reshape(len(submission), -1, len(coordinate_cols))
    solution['distance'] = np.linalg.norm(label_tensor - predicted_tensor, axis=2).min(axis=1)
    solution['thresholds'] = solution['Voxel spacing'].apply(lambda x: (min_radius * thresh_ratio) / x)
    solution['predictions'] = submission['Has motor'].values
    solution.loc[(solution['distance'] > solution['thresholds']) & (solution['Has motor'] == 1) & (submission['Has motor'] == 1), 'predictions'] = 0
    return solution['predictions'].values

def score(solution, submission, min_radius, beta):
    solution = solution.sort_values('tomo_id').reset_index(drop=True)
    submission = submission.sort_values('tomo_id').reset_index(drop=True)
    if not solution['tomo_id'].eq(submission['tomo_id']).all():
        raise ValueError('Submitted tomo_id values do not match')
    submission['Has motor'] = 1
    select = (submission[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] == -1).any(axis='columns')
    submission.loc[select, 'Has motor'] = 0
    predictions = distance_metric(solution, submission, thresh_ratio=1.0, min_radius=min_radius)
    return sklearn.metrics.fbeta_score(solution['Has motor'].values, predictions, beta=beta)

# Predict full volume
def predict_full_volume(model, volume, patch_size=(128, 128, 128), stride=32, batch_size=8):  # Reduced stride
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # Convert volume to tensor if needed and keep on CPU
    if isinstance(volume, np.ndarray):
        volume = torch.from_numpy(volume).float()
    if len(volume.shape) == 3:  # (z, y, x) -> (1, 1, z, y, x)
        volume = volume.unsqueeze(0).unsqueeze(0)
    elif len(volume.shape) == 4:  # (channels, z, y, x) -> (1, channels, z, y, x)
        volume = volume.unsqueeze(0)
    elif len(volume.shape) != 5:
        raise ValueError(f"Expected volume with 3-5 dimensions, got shape {volume.shape}")
    
    batch, channels, z_size, y_size, x_size = volume.shape
    pz, py, px = patch_size
    output = torch.zeros_like(volume, device='cpu')
    counts = torch.zeros_like(volume, device='cpu')
    
    # Collect patch coordinates
    patches = []
    coords = []
    for z in range(0, z_size, stride):
        for y in range(0, y_size, stride):
            for x in range(0, x_size, stride):
                z_end, y_end, x_end = z + pz, y + py, x + px
                coords.append((z, y, x, min(z_end, z_size), min(y_end, y_size), min(x_end, x_size)))
                patch = volume[:, :, z:min(z_end, z_size), y:min(y_end, y_size), x:min(x_end, x_size)]
                pad_z, pad_y, pad_x = max(0, z_end - z_size), max(0, y_end - y_size), max(0, x_end - x_size)
                if pad_z > 0 or pad_y > 0 or pad_x > 0:
                    patch = F.pad(patch, (0, pad_x, 0, pad_y, 0, pad_z))
                patches.append(patch)
    
    logger.info(f"Processing {len(patches)} patches for volume {volume.shape}")
    
    # Process patches in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(patches), batch_size), desc="Processing patches"):
            batch_patches = torch.cat(patches[i:i+batch_size], dim=0).to(device, non_blocking=True)
            batch_out = torch.sigmoid(model(batch_patches))
            batch_out = batch_out.cpu()
            for j, (z, y, x, z_end, y_end, x_end) in enumerate(coords[i:i+batch_size]):
                output[:, :, z:z_end, y:y_end, x:x_end] += batch_out[j, :, :z_end-z, :y_end-y, :x_end-x]
                counts[:, :, z:z_end, y:y_end, x:x_end] += 1
            del batch_patches, batch_out
            torch.cuda.empty_cache()
    
    output = output / (counts + 1e-8)
    logger.info(f"Predicted volume shape: {output.shape}, patches processed: {len(patches)}")
    return output.numpy()


# Extract motor location
def extract_motor_location(mask, threshold):
    mask = mask.squeeze()
    max_val = mask.max()
    logger.info(f"Extracting motor location, mask max: {max_val:.4f}, threshold: {threshold:.2f}")
    if max_val < threshold:
        return -1, -1, -1, 0
    z, y, x = np.unravel_index(np.argmax(mask), mask.shape)
    region = mask[max(0, z-5):z+6, max(0, y-5):y+6, max(0, x-5):x+6]
    if region.size == 0:
        logger.info(f"Motor at ({z}, {y}, {x}), region empty")
        return z, y, x, 1
    z_offset, y_offset, x_offset = center_of_mass(region)
    z, y, x = z + z_offset - 5, y + y_offset - 5, x + x_offset - 5
    logger.info(f"Motor at ({z:.2f}, {y:.2f}, {x:.2f})")
    return z, y, x, 1


# Tune peak detection threshold
def tune_threshold(model, val_ids, gcs_preprocessed_path, gcs_precomputed_path, local_dir, labels_df, thresholds=np.linspace(0.3, 0.7, 3)):  # Increased thresholds
    best_model_path = "/kaggle/working/best_model.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, weights_only=True))
        logger.info("Loaded best_model.pth")
    else:
        logger.warning("best_model.pth not found, using current model state")
    model.to(device).eval()
    best_threshold = 0.5
    best_fbeta = 0.0
    thresholds_list = []
    fbeta_scores = [] 

    # Check tomo_ab804d motor count
    tomo_labels = labels_df[labels_df["tomo_id"] == "tomo_ab804d"]
    if not tomo_labels.empty:
        num_motors = tomo_labels.iloc[0]["Number of motors"]
        logger.info(f"tomo_ab804d has {num_motors} motors")
    
    for threshold in thresholds:
        predictions = []
        for tomo_id in tqdm(val_ids, desc=f"Tuning threshold {threshold:.2f}"):  # Use all val_ids
            dataset = TomogramDataset(tomo_id, gcs_preprocessed_path, local_dir, mode="val")
            try:
                dataset.load()
                volume, _ = dataset[0]
                volume = volume.to(device, non_blocking=True)
                logger.info(f"Volume shape for {tomo_id}: {volume.shape}")
                pred_mask = predict_full_volume(model, volume, patch_size=(128, 128, 128), stride=32, batch_size=8)
                z, y, x, has_motor = extract_motor_location(pred_mask, threshold)
                predictions.append({"tomo_id": tomo_id, "Motor axis 0": z, "Motor axis 1": y, "Motor axis 2": x, "Has motor": has_motor})
            except Exception as e:
                logger.error(f"Error processing tomogram {tomo_id}: {str(e)}")
                predictions.append({"tomo_id": tomo_id, "Motor axis 0": -1, "Motor axis 1": -1, "Motor axis 2": -1, "Has motor": 0})
            finally:
                dataset.clear()
                clean_memory([tomo_id], local_dir)
                torch.cuda.empty_cache()
                logger.info(f"Cleared GPU memory after {tomo_id}, usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        
        # Create submission and solution DataFrames
        submission_df = pd.DataFrame(predictions)
        solution_data = []
        for tomo_id in val_ids:
            tomo_labels = labels_df[labels_df["tomo_id"] == tomo_id].iloc[0]
            solution_data.append({
                "tomo_id": tomo_id,
                "Motor axis 0": tomo_labels["Motor axis 0"],
                "Motor axis 1": tomo_labels["Motor axis 1"],
                "Motor axis 2": tomo_labels["Motor axis 2"],
                "Voxel spacing": tomo_labels["Voxel spacing"],
                "Has motor": 1 if tomo_labels["Number of motors"] > 0 else 0
            })
        solution_df = pd.DataFrame(solution_data)
        fbeta = score(solution_df, submission_df, min_radius=1000, beta=2)
        logger.info(f"Threshold {threshold:.2f}, Fβ-score: {fbeta:.4f}, TP: {((solution_df['Has motor'] == 1) & (submission_df['Has motor'] == 1)).sum()}, "
                    f"FP: {((solution_df['Has motor'] == 0) & (submission_df['Has motor'] == 1)).sum()}, "
                    f"FN: {((solution_df['Has motor'] == 1) & (submission_df['Has motor'] == 0)).sum()}")
        thresholds_list.append(threshold)
        fbeta_scores.append(fbeta)
        
        if fbeta > best_fbeta:
            best_fbeta = fbeta
            best_threshold = threshold
    
    logger.info(f"Best threshold: {best_threshold:.2f}, Best Fβ-score: {best_fbeta:.4f}")
    return best_threshold
    

# Predict test
def predict_test(model, test_ids, gcs_preprocessed_path, local_dir, threshold):
    if os.path.exists("best_model.pth"):
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        logger.info("Loaded best_model.pth")
    else:
        logger.warning("best_model.pth not found, using current model state")
    model.eval()
    predictions = []
    
    for tomo_id in tqdm(test_ids, desc="Predicting test set"):
        dataset = TomogramDataset(tomo_id, gcs_preprocessed_path, local_dir, mode="test")
        try:
            dataset.load()
            volume, _ = dataset[0]
            pred_mask = predict_full_volume(model, volume, patch_size=(128, 128, 128), stride=32, batch_size=8)
            z, y, x, has_motor = extract_motor_location(pred_mask, threshold)
            predictions.append({"tomo_id": tomo_id, "Motor axis 0": z, "Motor axis 1": y, "Motor axis 2": x})
        except Exception as e:
            logger.error(f"Error predicting tomogram {tomo_id}: {str(e)}")
            predictions.append({"tomo_id": tomo_id, "Motor axis 0": -1, "Motor axis 1": -1, "Motor axis 2": -1})
        finally:
            dataset.clear()
            clean_memory([tomo_id], local_dir)
            torch.cuda.empty_cache()
            logger.info(f"Cleared GPU memory after {tomo_id}, usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    
    return pd.DataFrame(predictions)


# Run pipeline
logger.info("Starting hyperparameter tuning...")
best_threshold = tune_threshold(model, val_ids, gcs_preprocessed_path, gcs_precomputed_path, local_dir, labels_df)  # Use all val_ids
logger.info("Generating test predictions...")
submission_df = predict_test(model, test_ids, gcs_preprocessed_path, local_dir, best_threshold)
submission_df.to_csv("submission.csv", index=False)
logger.info("Submission file created: submission.csv")
# Finish the wandb run
#wandb.finish()




