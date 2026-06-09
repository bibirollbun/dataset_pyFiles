import h5py
import numpy as np
import matplotlib.pyplot as plt


import h5py
import numpy as np
import matplotlib.pyplot as plt
import math

def visualize_slides_with_spots(h5_path, image_group, spot_group, max_slides_to_show=5, cols=1, plot_size_per_image=(10, 8), dot_color='red', dot_size=15, dot_alpha=0.6):
    """
    Visualizes histology slides with spatial transcriptomics spot overlays.

    Args:
        h5_path (str): Path to the HDF5 file.
        image_group (str): The HDF5 group for images (e.g., 'images/Train').
        spot_group (str): The HDF5 group for spot data (e.g., 'spots/Train').
        max_slides_to_show (int): The maximum number of slides to display.
        cols (int): The number of columns in the subplot grid.
        plot_size_per_image (tuple): The (width, height) in inches for each subplot.
        dot_color (str): The color of the spot markers.
        dot_size (int): The size of the spot markers.
        dot_alpha (float): The transparency of the spot markers.
    """
    h5file = None
    try:
        h5file = h5py.File(h5_path, "r")
        images_data = h5file[image_group]
        spots_data = h5file[spot_group]

        slide_names = list(images_data.keys())
        slides_to_show = slide_names[:min(len(slide_names), max_slides_to_show)]
        num_slides = len(slides_to_show)

        if num_slides == 0:
            print(f"No slides found in group '{image_group}'.")
            return

        # Calculate rows and figure dimensions based on desired plot size
        rows = math.ceil(num_slides / cols)
        fig_width = plot_size_per_image[0] * cols
        fig_height = plot_size_per_image[1] * rows
        
        fig, axes = plt.subplots(rows, cols, figsize=(fig_width, fig_height), squeeze=False)
        axes = axes.flatten()

        for i, slide_name in enumerate(slides_to_show):
            image = np.array(images_data[slide_name])
            spots = np.array(spots_data[slide_name])
            
            axes[i].imshow(image, aspect="equal") # 'equal' aspect ratio is often better for images
            axes[i].scatter(spots["x"], spots["y"], color=dot_color, s=dot_size, alpha=dot_alpha)
            axes[i].set_title(slide_name, fontsize=12, pad=10)
            axes[i].axis('off')

        # Hide unused subplots
        for j in range(num_slides, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        plt.show()

    except KeyError as e:
        print(f"Error: Could not find HDF5 group {e}. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    finally:
        if h5file:
            h5file.close()
            print("HDF5 file closed successfully.")


# --- How to use the function to get the desired 5x1 layout ---
H5_FILE_PATH = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"

# Visualize 5 training slides in 1 column and 5 rows with a large view
visualize_slides_with_spots(
    h5_path=H5_FILE_PATH,
    image_group="images/Train",
    spot_group="spots/Train",
    max_slides_to_show=5,
    cols=1,  # <-- Set to 1 column
    plot_size_per_image=(10, 8),  # <-- Set a large size for each plot (width, height)
    dot_size=15,  # <-- Increased dot size for visibility
    dot_alpha=0.6  # <-- Increased opacity for more prominent spots
)


# Visualize Test slide ('S_7')
import h5py
import numpy as np
import matplotlib.pyplot as plt

def visualize_single_slide(h5_path, slide_id, image_group, spot_group, figsize=(10, 8), dot_color='red', dot_size=15, dot_alpha=0.6):
    """
    Visualizes a single histology slide and its corresponding spot overlay.

    Args:
        h5_path (str): Path to the HDF5 file.
        slide_id (str): The name/ID of the specific slide to visualize (e.g., 'S_7').
        image_group (str): The HDF5 group for images (e.g., 'images/Test').
        spot_group (str): The HDF5 group for spot data (e.g., 'spots/Test').
        figsize (tuple): The (width, height) in inches for the figure.
        dot_color (str): The color of the spot markers.
        dot_size (int): The size of the spot markers.
        dot_alpha (float): The transparency of the spot markers.
    """
    try:
        with h5py.File(h5_path, "r") as h5file:
            images_data = h5file[image_group]
            spots_data = h5file[spot_group]

            # --- Robustness Check: Ensure the slide exists before trying to access it ---
            if slide_id not in images_data:
                print(f"Error: Slide ID '{slide_id}' not found in image group '{image_group}'.")
                print(f"Available slides are: {list(images_data.keys())}")
                return

            # Retrieve data for the specific slide
            image = np.array(images_data[slide_id])
            spots = np.array(spots_data[slide_id])
            
            # Plotting
            plt.figure(figsize=figsize)
            plt.imshow(image, aspect='equal') # 'equal' is often better for preserving tissue shape
            plt.scatter(spots["x"], spots["y"], color=dot_color, s=dot_size, alpha=dot_alpha)
            plt.title(f"Slide: {slide_id}", fontsize=14)
            plt.axis('off')
            plt.show()

    except KeyError as e:
        print(f"Error: Could not find HDF5 group {e}. Please check the path.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# --- How to use the new, flexible function ---
H5_FILE_PATH = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"

# Example 1: Replicate the original visualization for 'S_7' but with more prominent spots
print("Visualizing Test slide 'S_7' with prominent spots...")
visualize_single_slide(
    h5_path=H5_FILE_PATH,
    slide_id='S_7',
    image_group='images/Test',
    spot_group='spots/Test',
    figsize=(12, 10), # Slightly larger figure
    dot_size=20       # Larger dots
)


import pandas as pd

# Load and display (x,y) spot locations and cell type annotation table for Train slides
with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as f:
    train_spots = f["spots/Train"]
    
    # Dictionary to store DataFrames for each slide
    train_spot_tables = {}
    
    for slide_name in train_spots.keys():
        # Load dataset as NumPy structured array
        spot_array = np.array(train_spots[slide_name])
        
        # Convert to DataFrame
        df = pd.DataFrame(spot_array)
        
        # Store in dictionary
        train_spot_tables[slide_name] = df

# Example: Display the spots table for slide 'S_1'
train_spot_tables['S_1']


# Display spot table for Test slide (only the spot coordinates on 2D array)
with h5py.File("/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5", "r") as f:
    test_spots = f["spots/Test"]
    spot_array = np.array(test_spots['S_7'])
    test_spot_table = pd.DataFrame(spot_array)
    
# Show the test spots coordinates for slide 'S_7'
test_spot_table


# Create a random submission
# (predictions of cell type abundances for 35 classes across the Test slide spots;
# spot order should be same as in the 'Test' spots table)

# Use the cell type columns from the train spots table; assuming first two columns are (x, y)
cell_type_columns = train_spot_tables['S_1'].columns[2:].values  # Expecting 35 cell types here
indices = test_spot_table.index.values  # All spots on the Test slide

# Create a 2D array of random floats between 0 and 2 for each spot and cell type
prediction_matrix = 2 * np.random.rand(len(indices), len(cell_type_columns))
predicted_labels = pd.DataFrame(prediction_matrix, columns=cell_type_columns, index=indices)

predicted_labels.head()


# Prepare submission DataFrame: spot_id column and then predictions for each cell type
submission_df = predicted_labels.copy()
submission_df.insert(0, 'ID', submission_df.index)

# Save the submission file as submission.csv
submission_df.to_csv("./submission.csv", index=False)
print("Submission file 'submission.csv' created!")


!pip install -U albumentations


import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
import pandas as pd
import logging
from collections import OrderedDict
from typing import Dict, List, Tuple, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import GroupKFold


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def extract_patch(image_array, center_x, center_y, patch_size):
    """
    Extracts a square patch from a large image array centered at (center_x, center_y).
    Handles boundary conditions by padding with zeros if the patch extends beyond the image.

    Args:
        image_array (np.ndarray): The large image array (H, W, C).
        center_x (float or int): The x-coordinate (column) of the patch center.
        center_y (float or int): The y-coordinate (row) of the patch center.
        patch_size (int): The side length of the square patch (e.g., 256).

    Returns:
        np.ndarray: The extracted patch (patch_size, patch_size, C)
                    or None if dimensions are invalid.
    """
    try:
        img_h, img_w, img_c = image_array.shape

        # Ensure patch_size is positive
        if patch_size <= 0:
            logger.error("Error: patch_size must be positive.")
            return None

        half_patch = patch_size // 2

        # Calculate integer center coordinates
        center_y_int = int(round(center_y))
        center_x_int = int(round(center_x))

        # Calculate boundaries of the patch in the large image coordinate system
        y_start_ideal = center_y_int - half_patch
        y_end_ideal = y_start_ideal + patch_size  # End is exclusive in slicing
        x_start_ideal = center_x_int - half_patch
        x_end_ideal = x_start_ideal + patch_size  # End is exclusive in slicing

        # Create the output patch, initialized with zeros
        patch = np.zeros((patch_size, patch_size, img_c), dtype=image_array.dtype)

        # Determine the intersection of the ideal patch and the actual image boundaries
        y_start_img = max(0, y_start_ideal)
        y_end_img = min(img_h, y_end_ideal)
        x_start_img = max(0, x_start_ideal)
        x_end_img = min(img_w, x_end_ideal)

        # Calculate where to place this intersection within the output patch
        y_offset_patch = max(0, -y_start_ideal)
        x_offset_patch = max(0, -x_start_ideal)

        # Calculate the height and width of the region to copy
        copy_height = y_end_img - y_start_img
        copy_width = x_end_img - x_start_img

        # Check if there's any valid region to copy
        if copy_height > 0 and copy_width > 0:
            # Copy the valid region from the image array to the correct position in the patch
            patch[
                y_offset_patch : y_offset_patch + copy_height,
                x_offset_patch : x_offset_patch + copy_width,
                :
            ] = image_array[
                y_start_img : y_end_img,
                x_start_img : x_end_img,
                :
            ]

        return patch
    except Exception as e:
        logger.error(f"Error in extract_patch: {e}")
        return None


class LRUCache(OrderedDict):
    """
    Limit-size cache that discards the least recently used items.
    """
    def __init__(self, maxsize=128):
        self.maxsize = maxsize
        super().__init__()

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


# --- MODULARITY IMPROVEMENT 1: DATA PREPARATION FACTORY ---
# This class handles all the one-time data loading and preparation.
class DataFactory:
    """A builder class to prepare and split the dataset metadata efficiently."""
    def __init__(self, h5_path: str):
        self.h5_path = h5_path
        logger.info(f"DataFactory initialized with H5 file: {h5_path}")
        self.full_spots_df = self._load_all_spots()
        self.target_cols = [col for col in self.full_spots_df.columns if col.startswith('C')]
        if self.target_cols:
            logger.info(f"Identified {len(self.target_cols)} target columns: {self.target_cols}")
            # Ensure correct dtype at the source
            self.full_spots_df[self.target_cols] = self.full_spots_df[self.target_cols].astype(np.float32)
            logger.info("Converted target columns to float32 dtype.")
        else:
            logger.error("CRITICAL ERROR: No target columns (starting with 'c') found! Check data loading.")


    def _load_all_spots(self) -> pd.DataFrame:
        """
        Loads all spot data from all training slides into a single, efficient DataFrame.
        """
        all_spots_list = []
        with h5py.File(self.h5_path, "r") as f:
            train_spots_group = f['spots/Train']
            slide_ids = list(train_spots_group.keys())
            logger.info("Loading all training spots into a unified DataFrame...")
            for slide_id in slide_ids:
                # Pass the structured H5 dataset directly to pd.DataFrame to preserve column names.
                # The [()] ensures the data is fully read into memory.
                spots_df = pd.DataFrame(train_spots_group[slide_id][()])
                
                spots_df['slide_id'] = slide_id
                all_spots_list.append(spots_df)
        
        full_df = pd.concat(all_spots_list, ignore_index=True)
        logger.info(f"Total spots loaded from all slides: {len(full_df)}")
        # For debugging, let's see the columns
        logger.info(f"DataFrame columns loaded: {full_df.columns.tolist()}")
        return full_df
    
    # ... (the rest of your DataFactory class is fine) ...
    def get_split_data(self, train_ids: List[str], val_ids: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_df = self.full_spots_df[self.full_spots_df['slide_id'].isin(train_ids)].reset_index(drop=True)
        val_df = self.full_spots_df[self.full_spots_df['slide_id'].isin(val_ids)].reset_index(drop=True)
        logger.info(f"Train set created with {len(train_df)} spots from {len(train_ids)} slides.")
        logger.info(f"Validation set created with {len(val_df)} spots from {len(val_ids)} slides.")
        return train_df, val_df
        
    def get_test_data(self, test_slide_ids: List[str]) -> pd.DataFrame:
        test_spots_list = []
        with h5py.File(self.h5_path, "r") as f:
            test_spots_group = f['spots/Test']
            for slide_id in test_slide_ids:
                if slide_id in test_spots_group:
                    # Apply the same fix for test data
                    spots_df = pd.DataFrame(test_spots_group[slide_id][()])
                    spots_df['slide_id'] = slide_id
                    test_spots_list.append(spots_df)
        
        test_df = pd.concat(test_spots_list, ignore_index=True)
        logger.info(f"Test set created with {len(test_df)} spots.")
        return test_df





# --- MODULARITY IMPROVEMENT 2: A LEAN AND EFFICIENT DATASET CLASS ---
# This class's only job is to serve data. It's fast and clean.
class HESpotDataset(Dataset):
    def __init__(self, h5_path: str, spots_df: pd.DataFrame, patch_size: int, 
                 target_cols: List[str], transforms: A.Compose, mode: str = 'train'):
        self.h5_path = h5_path
        self.spots_df = spots_df
        self.patch_size = patch_size
        self.target_cols = target_cols
        self.transforms = transforms
        self.mode = mode
        
        # We will use a simple cache for the full images
        self.image_cache = LRUCache(maxsize=8) # LRUCache class should be defined in a prior cell

    def __len__(self) -> int:
        return len(self.spots_df)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        spot_info = self.spots_df.iloc[idx]
        slide_id = spot_info['slide_id']
        
        # Load full image using the LRU cache
        try:
            image_array = self.image_cache[slide_id]
        except KeyError:
            # If not in cache, open the file, read the image, and close.
            # This is simpler and more robust for notebook environments.
            with h5py.File(self.h5_path, 'r') as h5file:
                group = 'Train' if self.mode in ['train', 'val'] else 'Test'
                image_array = np.array(h5file[f'images/{group}'][slide_id])
                self.image_cache[slide_id] = image_array
        
        # Extract patch
        patch = extract_patch(image_array, spot_info['x'], spot_info['y'], self.patch_size)
        
        # Apply transforms using Albumentations
        if self.transforms:
            transformed = self.transforms(image=patch)
            patch_tensor = transformed['image']
        else: # If no transforms, just convert to tensor
            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1)

        # Prepare target
        if self.mode in ['train', 'val']:
            target_np = spot_info[self.target_cols].values.astype(np.float32)
            target = torch.tensor(target_np, dtype=torch.float32)
        else:
            target = torch.tensor([spot_info['x'], spot_info['y']], dtype=torch.float32)

        return patch_tensor, target
        
def get_transforms(mode: str, patch_size: int) -> A.Compose:
    """
    Final, correct transforms. The data is already float32 in [0, 1] range.
    We only need geometric augmentations and conversion to tensor. NO NORMALIZATION.
    """
    if mode == 'train':
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.Resize(patch_size, patch_size),
            # NO A.Normalize() NEEDED!
            ToTensorV2(), # Converts numpy (H,W,C) [0,1] to tensor (C,H,W) [0,1]
        ])
    else: # For validation and test
        return A.Compose([
            A.Resize(patch_size, patch_size),
            # NO A.Normalize() NEEDED!
            ToTensorV2(),
        ])



# --- Main Execution Block ---
if __name__ == '__main__':
    # --- Configuration ---
    H5_FILE_PATH = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
    PATCH_SIZE = 256
    BATCH_SIZE = 32
    NUM_WORKERS = 2 

    # 1. Create the DataFactory (it now discovers slide names and column names automatically)
    data_factory = DataFactory(H5_FILE_PATH)
    
    # 2. Define slide splits dynamically based on discovered slides
    all_train_slides = data_factory.full_spots_df['slide_id'].unique().tolist()
    
    # Use a robust split (e.g., last slide for validation)
    if len(all_train_slides) > 1:
        train_ids = all_train_slides[:-1]
        val_ids = [all_train_slides[-1]]
    else: # Handle case with only one slide
        train_ids = all_train_slides
        val_ids = all_train_slides
        
    # Assume 'S_7' is the correct test slide name
    test_ids = ['S_7'] 

    # 3. Get the pre-processed DataFrames
    train_df, val_df = data_factory.get_split_data(train_ids, val_ids)
    test_df = data_factory.get_test_data(test_ids)

    # 4. Create the lean Dataset objects
    # (HESpotDataset definition should be in a cell before this)
    train_dataset = HESpotDataset(
        h5_path=H5_FILE_PATH,
        spots_df=train_df,
        patch_size=PATCH_SIZE,
        target_cols=data_factory.target_cols,
        transforms=get_transforms('train', PATCH_SIZE), # get_transforms should also be defined
        mode='train'
    )
    val_dataset = HESpotDataset(
        h5_path=H5_FILE_PATH,
        spots_df=val_df,
        patch_size=PATCH_SIZE,
        target_cols=data_factory.target_cols,
        transforms=get_transforms('val', PATCH_SIZE),
        mode='val'
    )

    # 5. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS)

    logger.info(f"Setup complete. Training on {len(train_dataset)} spots, validating on {len(val_dataset)} spots.")


# 5. Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS)

logger.info(f"Setup complete. Training on {len(train_dataset)} spots, validating on {len(val_dataset)} spots.")


# --- Final Visualization Cell ---

logger.info("Visualizing one batch from the training loader...")
patches, targets = next(iter(train_loader))

print(f"Patches batch shape: {patches.shape}")
print(f"Targets batch shape: {targets.shape}")

fig, axes = plt.subplots(4, 4, figsize=(10, 10))
for i, ax in enumerate(axes.flat):
    if i >= len(patches): break
    
    # The data is already in the [0, 1] range. No un-normalization needed.
    patch = patches[i].permute(1, 2, 0).numpy()
    patch = np.clip(patch, 0, 1) # Clipping is a good safety measure
    
    ax.imshow(patch)
    ax.set_title(f"Target C1: {targets[i][0]:.2f}", fontsize=8)
    ax.axis('off')
    
plt.tight_layout()
plt.show()


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm # Use tqdm for non-notebook envs
import os
import time
from scipy.stats import spearmanr # For evaluation metric

# --- Configuration & Device ---
PATCH_SIZE = 256
BATCH_SIZE = 32 # Adjust based on GPU memory
NUM_WORKERS = 2
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 15 # Adjust as needed, use early stopping
PATIENCE = 5 # For early stopping
MODEL_SAVE_PATH = "best_model.pth"
N_CELL_TYPES = 35 # Number of output values

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define Train/Val/Test Slide IDs
all_train_slides = [f'S_{i}' for i in range(1, 7)] # S_1 to S_6
# Simple split for demonstration: S_1-S_5 train, S_6 val
train_slide_ids = all_train_slides[:-1]
val_slide_ids = [all_train_slides[-1]]
test_slide_ids = ['S_7']


# --- Model Definition ---
def get_model(num_classes=N_CELL_TYPES, pretrained=True, dropout_p=0.4): # Added dropout_p argument
    """Loads a pretrained ResNet, adds dropout, and modifies the final layer for regression."""
    # Use weights argument for newer torchvision versions
    if pretrained:
        weights = models.ResNet18_Weights.IMAGENET1K_V1
        model = models.resnet18(weights=weights)
    else:
        model = models.resnet18(weights=None)

    # Get the number of input features for the original fc layer
    num_ftrs = model.fc.in_features

    # Replace the original fc layer with a new sequence including Dropout
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout_p),      # Dropout layer
        nn.Linear(num_ftrs, num_classes) # Final Linear layer for regression
    )

    return model

# --- Instantiate the model with dropout ---
DROPOUT_RATE = 0.5 # Define your desired dropout rate
model = get_model(dropout_p=DROPOUT_RATE).to(device) # Pass the dropout rate
print("\n--- Model Loaded (with Dropout) ---")
# print(model) # Optional: print model structure to see the new Dropout layer

# --- Loss Function, Optimizer, Scheduler ---
criterion = nn.MSELoss() # Mean Squared Error is common for regression tasks like this
optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# Scheduler reduces learning rate when validation metric stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=PATIENCE // 3, verbose=True) # Reduce on plateau of avg spearman

# --- Evaluation Metric Function ---
def calculate_average_spearman(y_preds_all, y_targets_all):
    """Calculates the competition metric: average per-spot Spearman correlation."""
    if not isinstance(y_preds_all, np.ndarray):
        y_preds_all = np.array(y_preds_all)
    if not isinstance(y_targets_all, np.ndarray):
        y_targets_all = np.array(y_targets_all)

    if y_preds_all.shape != y_targets_all.shape:
        print(f"Shape mismatch: preds {y_preds_all.shape}, targets {y_targets_all.shape}")
        return 0.0 # Or raise error

    num_spots = y_preds_all.shape[0]
    if num_spots == 0:
        return 0.0

    spot_spearman_scores = []
    for i in range(num_spots):
        pred_spot = y_preds_all[i, :]
        target_spot = y_targets_all[i, :]

        # Check for constant arrays - spearmanr returns nan
        if np.all(pred_spot == pred_spot[0]) or np.all(target_spot == target_spot[0]):
             # If either prediction or target is constant for a spot, correlation is undefined.
             # Assign 0 or handle as per competition rules (often 0 is reasonable).
             # Or check if prediction matches target constant value -> 1.0? Let's use 0 for undefined.
             score = 0.0
        else:
            correlation, p_value = spearmanr(pred_spot, target_spot)
            if np.isnan(correlation):
                # Handle potential NaNs from spearmanr if variance is zero (should be caught above mostly)
                score = 0.0
            else:
                score = correlation
        spot_spearman_scores.append(score)

    return np.mean(spot_spearman_scores)




# A Trainer Class ---
class Trainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, device, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        
        self.history = {'train_loss': [], 'val_loss': [], 'val_spearman': []}
        self.best_spearman = -1.0
        self.epochs_no_improve = 0
        
    def _train_one_epoch(self):
        """Runs a single training epoch."""
        self.model.train()
        total_loss = 0.0
        
        for patches, targets in tqdm(self.train_loader, desc="Training"):
            patches = patches.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(patches)
            loss = self.criterion(outputs, targets)
            
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            
            # --- IMPROVEMENT: GRADIENT CLIPPING ---
            # Prevents exploding gradients for more stable training
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(self.train_loader)

    def _validate_one_epoch(self):
        """Runs a single validation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for patches, targets in tqdm(self.val_loader, desc="Validating"):
                patches = patches.to(self.device)
                targets = targets.to(self.device)
                
                outputs = self.model(patches)
                loss = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
                
        val_loss = total_loss / len(self.val_loader)
        val_spearman = calculate_average_spearman(all_preds, all_targets)
        return val_loss, val_spearman
        
    def fit(self):
        """The main training loop with early stopping."""
        start_time = time.time()
        
        for epoch in range(self.config['EPOCHS']):
            print(f"\n--- Epoch {epoch + 1}/{self.config['EPOCHS']} ---")
            
            train_loss = self._train_one_epoch()
            val_loss, val_spearman = self._validate_one_epoch()
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_spearman'].append(val_spearman)
            
            print(f"Epoch Summary: Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Spearman: {val_spearman:.4f}")
            
            # Learning rate scheduler step
            self.scheduler.step(val_spearman)
            
            # --- IMPROVEMENT: EARLY STOPPING ---
            if val_spearman > self.best_spearman:
                print(f"Validation Spearman improved from {self.best_spearman:.4f} to {val_spearman:.4f}. Saving model...")
                self.best_spearman = val_spearman
                torch.save(self.model.state_dict(), self.config['MODEL_SAVE_PATH'])
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                print(f"No improvement in validation Spearman for {self.epochs_no_improve} epoch(s).")
                
            if self.epochs_no_improve >= self.config['PATIENCE']:
                print(f"\nEarly stopping triggered after {self.config['PATIENCE']} epochs without improvement.")
                break
                
        end_time = time.time()
        print(f"\nTraining finished in {(end_time - start_time) / 60:.2f} minutes.")
        print(f"Best validation Spearman score: {self.best_spearman:.4f}")
        
        return self.history

# --- Main Execution Block ---
if __name__ == '__main__':
    # (Assuming all data loading cells have been run and train_loader/val_loader exist)
    
    # 1. Define Model, Loss, Optimizer, etc.
    model = get_model(num_classes=N_CELL_TYPES, dropout_p=0.5).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.2, patience=2, verbose=True)
    
    # 2. Bundle config into a dictionary
    config = {
        "EPOCHS": EPOCHS,
        "PATIENCE": PATIENCE,
        "MODEL_SAVE_PATH": MODEL_SAVE_PATH
    }

    # 3. Instantiate and run the trainer
    trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, scheduler, device, config)
    history = trainer.fit()

    # 4. Plot training history
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss History')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(history['val_spearman'], label='Validation Spearman', color='orange')
    plt.title('Validation Spearman Correlation History')
    plt.legend()
    plt.show()


import torch
import pandas as pd
from tqdm.notebook import tqdm
import numpy as np

# --- Configuration ---
# Ensure these variables are consistent with your training setup
MODEL_SAVE_PATH = "best_model.pth"
SUBMISSION_PATH = "submission.csv"
H5_FILE_PATH = "/kaggle/input/el-hackathon-2025/elucidata_ai_challenge_data.h5"
PATCH_SIZE = 256
BATCH_SIZE = 64 # You can often use a larger batch size for inference
NUM_WORKERS = 2
N_CELL_TYPES = 35 # Must match the model's output
test_ids = ['S_7'] # The slide IDs for the test set

# Use the same device as training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device} for inference.")


# --- 1. Load the Trained Model ---
print("\n--- Loading best model weights ---")
# First, instantiate the model architecture (must be identical to the one you trained)
model = get_model(num_classes=N_CELL_TYPES, pretrained=False) # pretrained=False is fine, we are loading our own weights

# Load the saved state dictionary
model.load_state_dict(torch.load(MODEL_SAVE_PATH))

# Move the model to the appropriate device and set to evaluation mode
model.to(device)
model.eval()
print("Model loaded successfully.")


# --- 2. Create the Test DataLoader ---
# Use the same DataFactory and Dataset classes as before
# The DataFactory should already be instantiated as 'data_factory' from training cells
print("\n--- Preparing test data ---")
test_df = data_factory.get_test_data(test_ids)

# We need the 'spot_id' for the submission file. The test DataFrame from the HDF5 file
# might have different column names. Let's assume the spot identifier column is the first one
# if it's not 'x' or 'y'. A more robust way is to check the dtype names.
with h5py.File(H5_FILE_PATH, "r") as f:
    # Let's get the exact names from the test set file
    test_col_names = [name.decode('utf-8') if isinstance(name, bytes) else name for name in f[f'spots/Test/{test_ids[0]}'].dtype.names]

# Let's assume the submission ID is the 'Test set' column, or another unique identifier.
# For this competition, let's create a unique ID from slide and spot index.
if 'spot_id' not in test_df.columns:
    test_df['spot_id'] = test_df.index.astype(str) # Create a default spot ID if not present

test_dataset = HESpotDataset(
    h5_path=H5_FILE_PATH,
    spots_df=test_df,
    patch_size=PATCH_SIZE,
    target_cols=[], # No target columns for test set
    transforms=get_transforms('test', PATCH_SIZE), # Use 'val' or 'test' mode transforms
    mode='test'
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False, # Never shuffle the test set!
    num_workers=NUM_WORKERS
)
print(f"Test data loader created with {len(test_dataset)} spots.")

# --- 3. Prediction Loop ---
print("\n--- Generating predictions on the test set ---")
all_preds = []

with torch.no_grad(): # Disable gradient calculations for speed and memory efficiency
    for patches, _ in tqdm(test_loader, desc="Predicting"):
        patches = patches.to(device)
        
        # Get model outputs
        outputs = model(patches)
        
        # Move predictions to CPU and append to our list
        all_preds.extend(outputs.cpu().numpy())

# Convert list of predictions to a NumPy array
predictions_np = np.array(all_preds)
print(f"Predictions generated. Shape: {predictions_np.shape}")


import pandas as pd
import numpy as np

# --- Configuration ---
SUBMISSION_PATH = "submission.csv"
SAMPLE_SUBMISSION_PATH = "/kaggle/working/submission.csv" 
N_CELL_TYPES = 35

# --- Assume these variables exist from your prediction loop ---
# predictions_np: A NumPy array of shape (num_test_spots, 35) containing your model's predictions.
# test_loader: The DataLoader used for the test set, with shuffle=False.

# Let's create dummy variables for demonstration if they don't exist yet
if 'predictions_np' not in locals():
    print("Creating dummy `predictions_np` for demonstration.")
    # This assumes a test_df DataFrame is available to get the number of spots
    if 'test_df' in locals():
        num_test_spots = len(test_df)
    else: # Fallback if test_df doesn't exist
        print("Warning: `test_df` not found, creating predictions for 1000 spots as an example.")
        num_test_spots = 1000
    predictions_np = np.random.rand(num_test_spots, N_CELL_TYPES)

# --- 4. Create the Submission File (The Final, Correct Way) ---
print("\n--- Creating submission.csv file using sample_submission.csv as a template ---")

try:
    # Step 4.1: Load the sample submission file
    sample_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
    print("Sample submission file loaded successfully.")
    print("Columns found in sample submission:", sample_df.columns.tolist())
    
    # Step 4.2: Verify that the number of predictions matches the number of rows
    if len(sample_df) != len(predictions_np):
        raise ValueError(f"CRITICAL ERROR: Mismatch in prediction rows. "
                         f"Model predicted for {len(predictions_np)} spots, but sample submission has {len(sample_df)} rows.")

    # Step 4.3: Get the names of the target columns from the sample file
    # --- THIS IS THE CORRECTED LINE ---
    # We now look for the column named 'ID' (all caps) to exclude it.
    target_cols = [col for col in sample_df.columns if col != 'ID']
    
    if len(target_cols) == 0:
        raise KeyError("Could not find any target columns in the sample submission file. "
                       "Check that the ID column is actually named 'ID'.")
        
    if len(target_cols) != N_CELL_TYPES:
         raise ValueError(f"CRITICAL ERROR: Mismatch in number of target columns. "
                         f"Model outputs {N_CELL_TYPES}, but sample file expects {len(target_cols)}.")

    print(f"Found ID column 'ID' and {len(target_cols)} target columns.")

    # Step 4.4: Assign your predictions to the target columns
    # The order is preserved because shuffle=False was used in the test DataLoader.
    sample_df[target_cols] = predictions_np
    
    # The 'ID' column from the sample_df is preserved perfectly.
    submission_df = sample_df

    # Step 4.5: Save the final DataFrame to a .csv file
    submission_df.to_csv(SUBMISSION_PATH, index=False)

    print(f"\nSubmission file created successfully at: {SUBMISSION_PATH}")
    print("Final submission file head:")
    print(submission_df.head())

except FileNotFoundError:
    print(f"ERROR: The sample submission file was not found at '{SAMPLE_SUBMISSION_PATH}'. "
          "Please check the file path.")
except KeyError:
    print("ERROR: The column 'ID' was not found in the sample submission file. "
          "Please double-check the exact column name for the spot identifiers in sample_submission.csv.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")


!pip freeze > requitements.txt




