import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import random
import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_recall_curve, average_precision_score
from PIL import Image


# FIX 1: Added proper error handling throughout the code

# Setup device detection with better error handling
def setup_device():
    """
    Set up the best available device: TPU > GPU > CPU
    Returns device type and appropriate PyTorch device
    """
    # Check for TPU
    try:
        import torch_xla.core.xla_model as xm
        print("TPU available, using TPU acceleration")
        device = xm.xla_device()
        return "tpu", device
    except (ImportError, NameError) as e:
        print(f"TPU not available: {e}")

    # Check for GPU
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"Found {device_count} GPU device(s)")
        for i in range(device_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"GPU {i}: {gpu_name}")

        # FIX 2: Added memory check to prevent OOM errors
        total_memory = torch.cuda.get_device_properties(0).total_memory
        print(f"GPU memory: {total_memory / 1e9:.2f} GB")

        device = torch.device("cuda:0")
        # Set performance optimizations
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        return "gpu", device

    # If no accelerator available, use CPU
    print("No GPU or TPU found, using CPU")
    return "cpu", torch.device("cpu")


# Set seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed()

# FIX 3: Added try-except for path definition to handle Kaggle path issues
try:
    # Define data paths
    BASE_PATH = Path("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025")
    if not BASE_PATH.exists():
        raise FileNotFoundError(f"Base path {BASE_PATH} not found")

    TRAIN_PATH = BASE_PATH / "train"
    TEST_PATH = BASE_PATH / "test"
    SAMPLE_SUBMISSION = BASE_PATH / "sample_submission.csv"
    TRAIN_LABELS = BASE_PATH / "train_labels.csv"

    # Verify paths exist
    for path in [TRAIN_PATH, TEST_PATH, SAMPLE_SUBMISSION, TRAIN_LABELS]:
        if not path.exists():
            print(f"Warning: Path {path} does not exist")
            if path == SAMPLE_SUBMISSION:
                print("Warning: Sample submission file might be needed later.")
            if path == TRAIN_LABELS:
                print("Warning: Train labels file might be needed later.")

except Exception as e:
    print(f"Error setting up paths: {e}")
    # FIX: Provide fallback paths
    BASE_PATH = Path(".")
    TRAIN_PATH = BASE_PATH / "train"
    TEST_PATH = BASE_PATH / "test"
    SAMPLE_SUBMISSION = BASE_PATH / "sample_submission.csv"
    TRAIN_LABELS = BASE_PATH / "train_labels.csv"
    print("Using fallback paths in current directory.")

# Set up device
accelerator_type, device = setup_device()
print(f"Using accelerator type: {accelerator_type}")
print(f"Using device: {device}")

# Read training labels with better error handling
train_labels_df = None
try:
    if TRAIN_LABELS.exists():
        train_labels_df = pd.read_csv(TRAIN_LABELS)
        print(f"Labels data shape: {train_labels_df.shape}")
        print("Label file columns:")
        print(train_labels_df.columns.tolist())
        print("Label data first 5 rows:")
        print(train_labels_df.head())
    else:
        print(f"Training labels file not found at: {TRAIN_LABELS}")
except Exception as e:
    print(f"Error reading label file {TRAIN_LABELS}: {e}")

# Read sample submission file
sample_submission_df = None
try:
    if SAMPLE_SUBMISSION.exists():
        sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION)
        print(f"Sample submission template shape: {sample_submission_df.shape}")
        print("Sample submission file columns:")
        print(sample_submission_df.columns.tolist())
        print("Sample submission first 5 rows:")
        print(sample_submission_df.head())
    else:
        print(f"Sample submission file not found at: {SAMPLE_SUBMISSION}")
except Exception as e:
    print(f"Error reading sample submission file {SAMPLE_SUBMISSION}: {e}")


# Get all tomogram folders and slices
def get_data_paths():
    train_tomogram_folders = []
    test_tomogram_folders = []
    train_slices = []
    test_slices = []

    try:
        # Get training data tomogram folders
        if TRAIN_PATH.exists():
            train_tomogram_folders = [f for f in TRAIN_PATH.iterdir() if f.is_dir()]
            train_tomogram_folders.sort()  # Ensure consistent order

            # Get training slices
            for folder in train_tomogram_folders:
                slices = list(folder.glob("*.jpg"))
                slices.sort()  # Ensure slices are ordered
                train_slices.extend(slices)
            print(f"Found {len(train_tomogram_folders)} training tomogram folders.")
            print(f"Found {len(train_slices)} total training slices.")
        else:
            print(f"Warning: Training path {TRAIN_PATH} does not exist")

        # Get test data tomogram folders
        if TEST_PATH.exists():
            test_tomogram_folders = [f for f in TEST_PATH.iterdir() if f.is_dir()]
            test_tomogram_folders.sort()

            # Get test slices
            for folder in test_tomogram_folders:
                slices = list(folder.glob("*.jpg"))
                slices.sort()
                test_slices.extend(slices)
            print(f"Found {len(test_tomogram_folders)} test tomogram folders.")
            print(f"Found {len(test_slices)} total test slices.")
        else:
            print(f"Warning: Test path {TEST_PATH} does not exist")

    except Exception as e:
        print(f"Error getting data paths: {e}")

    return train_tomogram_folders, test_tomogram_folders, train_slices, test_slices


train_tomogram_folders, test_tomogram_folders, train_slices, test_slices = get_data_paths()
# print(f"Number of training tomograms: {len(train_tomogram_folders)}") # Redundant with prints inside function
# print(f"Number of test tomograms: {len(test_tomogram_folders)}")
# print(f"Total number of training slices: {len(train_slices)}")
# print(f"Total number of test slices: {len(test_slices)}")

# Example view of a training folder name
if train_tomogram_folders:
    print(f"Example training tomogram folder name: {train_tomogram_folders[0].name}")
    # View slice filenames in this folder
    slices_example = list(train_tomogram_folders[0].glob("*.jpg"))
    if slices_example:
        print(f"Example slice filename: {slices_example[0].name}")


# Extract tomogram ID and slice index from file path
def extract_tomo_slice_info(file_path):
    # Ensure file_path is a Path object
    file_path = Path(file_path)
    # Extract tomogram ID from path
    tomo_id = file_path.parent.name
    # Extract slice index from filename
    try:
        # FIX 4: More robust extraction of slice index
        filename = file_path.stem
        if '_' in filename:
            slice_idx_str = filename.split('_')[-1]  # Take the part after the last underscore
            slice_idx = int(slice_idx_str)
        else:
            # Fallback if filename format is different (e.g., only digits)
            slice_idx_str = ''.join(filter(str.isdigit, filename))  # Removed the backslash here
            if slice_idx_str:
                slice_idx = int(slice_idx_str)
            else:
                raise ValueError("Could not extract slice index from filename")
    except (ValueError, IndexError, TypeError) as e:
        print(f"Error extracting slice index from {file_path}: {e}. Using default 0.")
        slice_idx = 0  # Default value

    return tomo_id, slice_idx


# FIX 5: More robust label preprocessing
def preprocess_labels(labels_df):
    """
    Process labels dataframe into a dictionary for quick lookup
    Handle different column naming conventions
    """
    if labels_df is None:
        print("Warning: No labels data provided for preprocessing.")
        return {}

    # Check columns and adapt accordingly
    columns = labels_df.columns.tolist()

    # Try to find the right column names based on common patterns
    tomo_col = next((col for col in columns if 'tomo' in col.lower()), None)
    slice_col = next((col for col in columns if 'row' in col.lower() or 'slice' in col.lower()), None)
    x_col = next((col for col in columns if 'axis 0' in col.lower() or col.lower() == 'x'), None)
    y_col = next((col for col in columns if 'axis 1' in col.lower() or col.lower() == 'y'), None)

    # Use default column names if not found and issue warnings
    if tomo_col is None:
        tomo_col = 'tomo_id'
        print(f"Warning: Tomogram ID column not found, defaulting to '{tomo_col}'")
    if slice_col is None:
        slice_col = 'row_id'
        print(f"Warning: Slice index column not found, defaulting to '{slice_col}'")
    if x_col is None:
        x_col = 'Motor axis 0'
        print(f"Warning: X coordinate column not found, defaulting to '{x_col}'")
    if y_col is None:
        y_col = 'Motor axis 1'
        print(f"Warning: Y coordinate column not found, defaulting to '{y_col}'")

    # Verify that the chosen columns exist in the DataFrame
    required_cols = [tomo_col, slice_col, x_col, y_col]
    if not all(col in labels_df.columns for col in required_cols):
        print(
            f"Error: One or more required columns {required_cols} not found in labels DataFrame. Columns available: {labels_df.columns}")
        return {}

    print(f"Using label columns: Tomo='{tomo_col}', Slice='{slice_col}', X='{x_col}', Y='{y_col}'")

    # Create dictionary to store labels
    labels_dict = {}

    # Iterate through label data
    for _, row in labels_df.iterrows():
        try:
            # Convert to proper types with error handling
            tomo_id = str(row[tomo_col])

            # Handle different slice index formats
            slice_val = row[slice_col]
            if isinstance(slice_val, (int, float)) and not np.isnan(slice_val):
                slice_idx = int(slice_val)
            elif isinstance(slice_val, str):
                # Try extracting numeric part if not a clean integer string
                slice_idx_str = ''.join(filter(str.isdigit, slice_val))
                if slice_idx_str:
                    slice_idx = int(slice_idx_str)
                else:
                    print(
                        f"Warning: Could not parse slice index from value '{slice_val}' for tomo '{tomo_id}'. Skipping row.")
                    continue
            else:
                print(
                    f"Warning: Unexpected type for slice index '{slice_val}' (type: {type(slice_val)}) for tomo '{tomo_id}'. Skipping row.")
                continue

            # Extract coordinates
            x_val = row[x_col]
            y_val = row[y_col]
            if pd.isna(x_val) or pd.isna(y_val):
                print(f"Warning: NaN coordinate found for tomo '{tomo_id}', slice {slice_idx}. Skipping point.")
                continue
            x = float(x_val)
            y = float(y_val)

            key = (tomo_id, slice_idx)
            if key not in labels_dict:
                labels_dict[key] = []

            # Add (x, y) coordinates to corresponding (tomo_id, slice_idx) key
            labels_dict[key].append((x, y))

        except (ValueError, TypeError) as e:
            print(f"Error processing row: {row.to_dict()} -> {e}. Skipping row.")
        except Exception as e:
            print(f"Unexpected error processing row: {row.to_dict()} -> {e}. Skipping row.")

    return labels_dict


# FIX 6: Check data before processing
if train_labels_df is not None:
    labels_dict = preprocess_labels(train_labels_df)
    print(f"Number of unique slices with labels processed: {len(labels_dict)}")
    # Show examples of first few labels
    if labels_dict:
        count = 0
        total_points = 0
        for key, points in labels_dict.items():
            if count < 3:
                print(f"Example Label - Slice {key}: {len(points)} marked points: {points[:3]}...")
            count += 1
            total_points += len(points)
        print(f"Total labeled points across all slices: {total_points}")
else:
    print("Warning: No label data (train_labels_df) loaded. labels_dict will be empty.")
    labels_dict = {}


# FIX 7: Improved heatmap creation function
def create_heatmap(img_shape, points, sigma=10):
    """
    Create Gaussian heatmap for given points

    Parameters:
    - img_shape: Target heatmap shape (height, width)
    - points: List of coordinates [(x1, y1), (x2, y2), ...] in original image space
    - sigma: Standard deviation of Gaussian kernel in heatmap space

    Returns:
    - Heatmap numpy array (float32)
    """
    height, width = img_shape  # Expecting (height, width)
    heatmap = np.zeros((height, width), dtype=np.float32)

    # If no points, return zero heatmap
    if not points or len(points) == 0:
        return heatmap

    # Create meshgrid for the heatmap
    y_grid, x_grid = np.mgrid[0:height, 0:width]

    for x, y in points:
        # Coordinates (x,y) need to be mapped to the heatmap space if different from original
        # Assuming for now points are already scaled/relative to heatmap size if needed
        # Ensure coordinates are valid for the heatmap grid
        # Note: OpenCV/numpy uses (row, col) which is (y, x)
        int_x, int_y = int(round(x)), int(round(y))

        # Check bounds carefully (use heatmap shape)
        if 0 <= int_x < width and 0 <= int_y < height:
            # Compute Gaussian values centered at the point (use float coords for center)
            gaussian = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2)) \
 \
                # Update heatmap, taking maximum value to avoid overlap issues
            heatmap = np.maximum(heatmap, gaussian)

    return heatmap


# FIX 8: Improved dataset class with better error handling
class BacterialMotorDataset(Dataset):
    def __init__(self, image_paths, labels_dict=None, transform=None, is_test=False, target_size=(256, 256)):
        self.image_paths = image_paths
        self.labels_dict = labels_dict if labels_dict is not None else {}
        self.transform = transform
        self.is_test = is_test
        self.target_size = target_size  # Store target size (height, width)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            # Read image using PIL (handles paths better, integrates with transforms)
            img_pil = Image.open(img_path).convert('L')
            original_size = img_pil.size  # (width, height)

            # Extract tomogram ID and slice index
            tomo_id, slice_idx = extract_tomo_slice_info(img_path)

            # Apply transforms (input for model)
            if self.transform:
                img_transformed = self.transform(img_pil)
            else:
                # Default transform if none provided
                img_transformed = transforms.ToTensor()(img_pil)  # Might need resize here too if transform is None

            # For test set, return minimal information
            if self.is_test:
                return {
                    'image': img_transformed,
                    'tomo_id': tomo_id,
                    'slice_idx': slice_idx,
                    'image_path': str(img_path),
                    'original_size': original_size[::-1]  # Return as (height, width)
                }

            # --- Training/Validation specific part ---
            key = (tomo_id, slice_idx)
            points_original = self.labels_dict.get(key, [])  # Points in original image coords

            # Scale points to the target heatmap size (which matches the transformed image size)
            # Original image size: original_size (width, height)
            # Target heatmap size: self.target_size (height, width)
            points_scaled = []
            if points_original:
                orig_w, orig_h = original_size
                target_h, target_w = self.target_size
                if orig_w > 0 and orig_h > 0:  # Avoid division by zero
                    scale_x = target_w / orig_w
                    scale_y = target_h / orig_h
                    points_scaled = [(p[0] * scale_x, p[1] * scale_y) for p in points_original]
                else:
                    print(f"Warning: Original image size is zero for {img_path}. Cannot scale points.")

            # Create heatmap label using scaled points and target size
            # Pass target size (height, width) to create_heatmap
            heatmap = create_heatmap(self.target_size, points_scaled, sigma=5)  # Adjust sigma as needed for target size

            # Convert heatmap to tensor
            heatmap_tensor = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0)

            return {
                'image': img_transformed,
                'heatmap': heatmap_tensor,
                'points_original': points_original,  # Keep original points if needed for eval
                'tomo_id': tomo_id,
                'slice_idx': slice_idx,
                'image_path': str(img_path),
                'original_size': original_size[::-1]  # Return as (height, width)
            }

        except FileNotFoundError:
            print(f"Error: Image file not found: {img_path}")
            # Return None or handle appropriately based on collate_fn
            return None  # Collate fn needs to handle None
        except Exception as e:
            print(f"Error processing item {idx}, path {img_path}: {e}")
            # Return None or a default item to avoid breaking the DataLoader
            # Collate fn needs to handle None
            return None


# FIX 9: Simplified transforms with fixed size
target_height, target_width = 256, 256
train_transform = transforms.Compose([
    transforms.Resize((target_height, target_width)),
    # Add Augmentations if desired (e.g., RandomHorizontalFlip)
    # transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # Add Normalization if desired (calculate mean/std from dataset or use imagenet defaults)
    # transforms.Normalize(mean=[0.5], std=[0.5]) # Example for grayscale
])

val_transform = transforms.Compose([
    transforms.Resize((target_height, target_width)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5], std=[0.5]) # Use same normalization as training
])


# FIX 10: Simplified and more robust collate function (Revised)
def custom_collate_fn(batch):
    """
    Collate function that handles potential None items in the batch resulted from dataset errors,
    and correctly collates varying length lists/tuples.
    """
    # Filter out None items
    batch = [item for item in batch if item is not None]

    if not batch:
        # Return an empty dictionary with expected keys if batch is empty after filtering
        print("Warning: Empty batch encountered in collate_fn")
        # IMPORTANT: Make sure to return all possible keys a batch might have
        return {
            'image': torch.Tensor(0),
            'heatmap': torch.Tensor(0),  # Include heatmap, even if empty
            'points_original': [],
            'tomo_id': [],
            'slice_idx': [],
            'image_path': [],
            'original_size': []
        }

    # Manually collate items
    collated_batch = {}

    # Stack image and heatmap tensors
    collated_batch['image'] = torch.stack([item['image'] for item in batch])

    # Conditionally add heatmap and points_original if present (for training/validation)
    if 'heatmap' in batch[0]:
        collated_batch['heatmap'] = torch.stack([item['heatmap'] for item in batch])
    else:
        # If heatmap is not present, it's likely a test batch, so don't add it to the collated batch
        # Or, if you need this key always, you could return an empty tensor of appropriate shape.
        # For prediction, you typically don't need the ground truth heatmap.
        pass  # No need to add this key if it's not meant to be there for test

    if 'points_original' in batch[0]:  # ADD THIS CONDITIONAL CHECK
        collated_batch['points_original'] = [item['points_original'] for item in batch]  # This will be a list of lists
    else:
        # For test batches, points_original is not needed.\
        # You can still add an empty list for consistency if downstream code expects the key.
        collated_batch['points_original'] = []

    # Collect other items into lists (they don't need stacking into tensors)
    collated_batch['tomo_id'] = [item['tomo_id'] for item in batch]  # List of strings
    collated_batch['slice_idx'] = [item['slice_idx'] for item in batch]  # List of integers/tensors
    collated_batch['image_path'] = [item['image_path'] for item in batch]  # List of strings
    collated_batch['original_size'] = [item['original_size'] for item in batch]  # List of tuples

    return collated_batch


# FIX 11: Improved UNet architecture with better initialization
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # Bias=False with BN
            nn.BatchNorm2d(mid_channels), \
            nn.ReLU(inplace=True), \
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),  # Bias=False with BN
            nn.BatchNorm2d(out_channels), \
            nn.ReLU(inplace=True)
        )
        # Initialize weights (optional but often good practice)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175176fdd0f70
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # Initialize weights (optional)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, bilinear=False, init_features=32):  # Added init_features
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        # Use init_features to control the channel sizes
        f = init_features
        self.inc = DoubleConv(n_channels, f)
        self.down1 = Down(f, f * 2)
        self.down2 = Down(f * 2, f * 4)
        self.down3 = Down(f * 4, f * 8)
        self.down4 = Down(f * 8, f * 16 // factor)  # Adjusted for bilinear option
        self.up1 = Up(f * 16, f * 8 // factor, bilinear)  # Adjusted for bilinear option
        self.up2 = Up(f * 8, f * 4 // factor, bilinear)  # Adjusted for bilinear option
        self.up3 = Up(f * 4, f * 2 // factor, bilinear)  # Adjusted for bilinear option
        self.up4 = Up(f * 2, f, bilinear)
        self.outc = OutConv(f, n_classes)

    def forward(self, x):
        # FIX 12: Add input shape check
        if x.dim() != 4 or x.shape[1] != self.n_channels:
            raise ValueError(f"Expected 4D tensor N-{self.n_channels}-H-W, got {x.shape}")

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)  # Output logits directly

        # REMOVED torch.sigmoid - Use BCEWithLogitsLoss instead
        # output = torch.sigmoid(logits)

        return logits  # Return logits


# FIX 14: Training function with proper error handling and GPU memory optimization
# UPDATED val_loader_to_use logic
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, accelerator_type='cpu',
                device=torch.device('cpu')):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    # For TPU acceleration (Keep structure but might need specific imports if actually running on TPU)
    is_tpu = accelerator_type == 'tpu'
    if is_tpu:
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
            print("TPU modules imported successfully for training.")
        except ImportError as e:
            print(f"Warning: Error importing TPU modules, training might proceed on CPU/GPU if available: {e}")
            is_tpu = False  # Fallback if imports fail
            # Update accelerator_type based on fallback? Or assume setup_device handled it?
            # Re-check device if needed, though setup_device should be source of truth

    # For mixed precision training on GPU
    use_amp = False
    scaler = None
    if accelerator_type == 'gpu' and not is_tpu:  # Only use CUDA AMP if GPU is selected and not TPU
        try:
            from torch.cuda.amp import GradScaler, autocast
            use_amp = True
            scaler = GradScaler()
            print("Mixed precision training (AMP) enabled for GPU.")
        except ImportError:
            print("CUDA AMP (GradScaler, autocast) not available, using full precision.")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        # Training phase
        model.train()
        train_loss = 0.0

        # Choose appropriate dataloader based on accelerator
        if is_tpu:
            try:
                train_device_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
                loader_to_use = train_device_loader
                print("Using TPU ParallelLoader for training.")
            except Exception as e:
                print(f"Error setting up TPU dataloader for training: {e}. Using standard loader.")
                loader_to_use = train_loader
        else:
            loader_to_use = train_loader
            print("Using standard DataLoader for training.")

        # Training loop with error handling
        pbar_train = tqdm(loader_to_use, desc=f'Epoch {epoch + 1}/{num_epochs} [Train]', leave=False)
        # batch_count = 0 # 这一行不再需要，因为我们将使用 enumerate 来获取 batch_idx
        processed_items = 0
        # --- FIX: 关键修改在这里 ---
        for batch_idx, batch in enumerate(pbar_train):  # <--- 添加了 enumerate(pbar_train) 来获取 batch_idx
            # --- 结束关键修改 ---
            # Check if collate_fn returned None (due to dataset errors)
            if batch is None:
                print(
                    f"Warning: Skipping None batch in training loop (batch index approx {batch_idx}).")  # 使用 batch_idx
                continue

            try:
                images = batch['image'].to(device)
                heatmaps = batch['heatmap'].to(device)

                # Check for empty batch after potential filtering in collate_fn
                if images.size(0) == 0:
                    print(f"Warning: Empty batch encountered at index {batch_idx}, skipping.")  # 使用 batch_idx
                    continue

                current_batch_size = images.size(0)

                # Forward pass with appropriate precision
                if use_amp:
                    with autocast():
                        outputs = model(images)  # Expecting logits
                        loss = criterion(outputs, heatmaps)  # criterion is BCEWithLogitsLoss
                else:
                    outputs = model(images)  # Expecting logits
                    loss = criterion(outputs, heatmaps)  # criterion is BCEWithLogitsLoss

                # Check for NaN loss
                if torch.isnan(loss):
                    print(f"Warning: NaN loss encountered at batch {batch_idx}. Skipping batch.")  # 使用 batch_idx
                    # Optionally zero gradients here if optimizer state might be affected
                    optimizer.zero_grad()
                    continue

                # Backward pass
                optimizer.zero_grad()

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                elif is_tpu:
                    loss.backward()
                    xm.optimizer_step(optimizer, barrier=True)  # Barrier=True ensures step completion
                else:  # Standard CPU/GPU without AMP
                    loss.backward()
                    optimizer.step()

                # Update statistics
                train_loss += loss.item() * current_batch_size  # Use actual batch size
                processed_items += current_batch_size

                # Update progress bar
                pbar_train.set_postfix(loss=f"{loss.item():.4f}")

                # FIX 15: Periodic GPU memory cleanup (only if using GPU)
                if accelerator_type == 'gpu' and batch_idx % 50 == 0:  # 这里的 batch_idx 现在被正确定义了
                    torch.cuda.empty_cache()

            except Exception as e:
                # Log specific error for the batch and continue
                print(f"\nError during training batch {batch_idx}: {e}")  # 使用 batch_idx
                # Consider adding traceback print here for debugging:
                # import traceback
                # traceback.print_exc()
                continue  # Continue to the next batch

        # Calculate average loss for the epoch
        if processed_items > 0:
            avg_train_loss = train_loss / processed_items
        else:
            avg_train_loss = 0.0
            print("Warning: No items processed in training epoch.")

        # Validation phase
        model.eval()
        val_loss = 0.0
        val_processed_items = 0

        # FIX: Choose appropriate validation dataloader (Corrected Logic)
        if is_tpu:
            try:
                # Make sure pl is available from the TPU check earlier
                val_device_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
                val_loader_to_use = val_device_loader
                print("Using TPU ParallelLoader for validation.")
            except Exception as e:
                print(f"Error setting up TPU validation dataloader: {e}. Using standard loader.")
                val_loader_to_use = val_loader  # Fallback for TPU error
        else:
            val_loader_to_use = val_loader  # Defined for CPU/GPU
            print("Using standard DataLoader for validation.")

        pbar_val = tqdm(val_loader_to_use, desc=f'Epoch {epoch + 1}/{num_epochs} [Val]', leave=False)
        val_batch_count = 0
        with torch.no_grad():  # Ensure no gradients are computed
            for batch in pbar_val:  # 验证循环中没有使用 batch_idx，所以无需修改
                # Check if collate_fn returned None
                if batch is None:
                    print(f"Warning: Skipping None batch in validation loop (batch index approx {val_batch_count}).")
                    val_batch_count += 1
                    continue

                try:
                    images = batch['image'].to(device)
                    heatmaps = batch['heatmap'].to(device)

                    # Check for empty batch
                    if images.size(0) == 0:
                        print(f"Warning: Empty validation batch encountered at index {val_batch_count}, skipping.")
                        val_batch_count += 1
                        continue

                    current_batch_size = images.size(0)

                    # Forward pass (no autocast needed for validation with no_grad usually, but doesn't hurt)
                    if use_amp:
                        with autocast():
                            outputs = model(images)  # Expecting logits
                            loss = criterion(outputs, heatmaps)  # criterion is BCEWithLogitsLoss
                    else:
                        outputs = model(images)  # Expecting logits
                        loss = criterion(outputs, heatmaps)  # criterion is BCEWithLogitsLoss

                    # Check for NaN loss
                    if torch.isnan(loss):
                        print(
                            f"Warning: NaN loss encountered during validation batch {val_batch_count}. Skipping loss accumulation for this batch.")
                        val_batch_count += 1
                        continue

                    # Update statistics
                    val_loss += loss.item() * current_batch_size
                    val_processed_items += current_batch_size

                    # Update progress bar
                    pbar_val.set_postfix(loss=f"{loss.item():.4f}")

                    val_batch_count += 1

                except Exception as e:
                    print(f"\nError during validation batch {val_batch_count}: {e}")
                    # import traceback
                    # traceback.print_exc()
                    val_batch_count += 1
                    continue  # Continue to the next batch

        # Calculate average validation loss
        if val_processed_items > 0:
            avg_val_loss = val_loss / val_processed_items
        else:
            avg_val_loss = float('inf')  # Or handle as error / zero?
            print("Warning: No items processed in validation epoch.")

        # Print epoch results
        print(f'Epoch {epoch + 1}/{num_epochs}: \n' \
              f'\tTrain Loss: {avg_train_loss:.4f}\n' \
              f'\tVal Loss:   {avg_val_loss:.4f}')

        # Update learning rate scheduler based on validation loss
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        new_lr = optimizer.param_groups[0]['lr']
        if new_lr != current_lr:
            print(f"\tLearning rate reduced to {new_lr:.6f}")

        # Save model if validation loss improved
        if avg_val_loss < best_val_loss:
            print(f"\tValidation loss improved from {best_val_loss:.4f} to {avg_val_loss:.4f}. Saving model...")
            best_val_loss = avg_val_loss
            save_path = Path("./best_model.pth")  # Save in working directory
            try:
                model_save = model  # Default save
                # Handle TPU model saving if applicable (might need state dict conversion)
                if is_tpu:
                    # TPU-specific saving often involves getting state_dict
                    # xm.save(model.state_dict(), str(save_path)) might be needed
                    # Or save the CPU version if model was wrapped
                    torch.save(model.state_dict(), save_path)  # Saving state_dict is generally safer
                    print(f"\tTPU Model state_dict saved to {save_path}")
                else:
                    torch.save(model.state_dict(), save_path)  # Save state_dict for flexibility
                    print(f"\tModel state_dict saved to {save_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            print(f"\tValidation loss did not improve from {best_val_loss:.4f}.")

        # Update history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # Plot training progress periodically
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            try:
                plt.figure(figsize=(10, 5))
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training and Validation Loss')
                plt.legend()
                plt.grid(True)
                plot_save_path = Path(f"./training_progress_epoch_{epoch + 1}.png")
                plt.savefig(plot_save_path)
                print(f"\tTraining progress plot saved to {plot_save_path}")
                plt.close()  # Close plot to free memory
            except Exception as e:
                print(f"Error plotting training progress: {e}")

    print("\nTraining finished.")
    return model, history


# FIX 16: Enhanced post-processing function to detect peaks in the heatmap
def detect_points_from_heatmap(heatmap, threshold=0.5, min_distance=10, original_size=None, current_size=None):
    """
    Detect points from heatmap by finding local maxima above a threshold.

    Parameters:
    - heatmap: Predicted heatmap tensor (C, H, W) or numpy array (H, W)
    - threshold: Minimum value to consider as a potential motor peak
    - min_distance: Minimum distance between detected peaks in heatmap pixel coordinates
    - original_size: Tuple (height, width) of the original image for scaling back
    - current_size: Tuple (height, width) of the heatmap if scaling needed

    Returns:
    - List of (x, y) coordinates of detected motors in ORIGINAL image space
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

    # Convert tensor to numpy if needed, ensure it's on CPU
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.squeeze().cpu().numpy()

    # Ensure heatmap is 2D
    if heatmap.ndim != 2:
        print(f"Warning: Expected 2D heatmap, got shape {heatmap.shape}. Attempting to squeeze.")
        heatmap = np.squeeze(heatmap)
        if heatmap.ndim != 2:
            print("Error: Could not convert heatmap to 2D.")
            return []

    # Check if current_size is provided, otherwise use heatmap's shape
    if current_size is None:
        current_size = heatmap.shape  # (height, width)
    current_h, current_w = current_size

    # Apply Gaussian filter to smooth the heatmap (helps find stable peaks)
    # Sigma=1 is often reasonable for peak detection
    heatmap_smoothed = gaussian_filter(heatmap, sigma=1)

    # Find local maxima using maximum_filter
    # Create a footprint for the neighborhood (e.g., 3x3)
    neighborhood = generate_binary_structure(2, 2)  # 8-connectivity
    local_max = maximum_filter(heatmap_smoothed, footprint=neighborhood) == heatmap_smoothed

    # Apply threshold: Only consider maxima above the threshold
    detected_peaks = (heatmap_smoothed > threshold) & local_max

    # Extract coordinates of peaks
    y_indices, x_indices = np.where(detected_peaks)

    # Get heatmap values (intensities) at peaks
    intensities = heatmap_smoothed[y_indices, x_indices]

    # Combine coordinates and intensities
    points_with_intensities = list(zip(x_indices, y_indices, intensities))

    # Sort points by intensity (highest to lowest) - helps in non-maximum suppression step
    points_with_intensities.sort(key=lambda p: p[2], reverse=True)

    # Non-maximum suppression based on min_distance
    # Keep track of points to include in the final list
    final_points_heatmap = []
    suppressed = np.zeros(len(points_with_intensities), dtype=bool)

    for i in range(len(points_with_intensities)):
        if suppressed[i]:
            continue  # Skip if already suppressed

        # Add this point (it's the strongest in its neighborhood so far)
        xi, yi, _ = points_with_intensities[i]
        final_points_heatmap.append((xi, yi))

        # Suppress other points within min_distance
        for j in range(i + 1, len(points_with_intensities)):
            if suppressed[j]:
                continue
            xj, yj, _ = points_with_intensities[j]
            dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist < min_distance:
                suppressed[j] = True

    # --- Scaling back to original image size ---
    final_points_original = []
    if original_size is not None:
        orig_h, orig_w = original_size  # (height, width)
        if current_w > 0 and current_h > 0:  # Check for division by zero
            scale_x = orig_w / current_w
            scale_y = orig_h / current_h
            # Scale and round to nearest integer coordinates
            final_points_original = [(int(round(x * scale_x)), int(round(y * scale_y))) for x, y in
                                     final_points_heatmap]
        else:
            print("Warning: Current heatmap size is zero. Cannot scale points back.")
            final_points_original = final_points_heatmap  # Return heatmap coords if scaling fails
    else:
        # If original_size not provided, return points in heatmap coordinates
        final_points_original = final_points_heatmap

    return final_points_original


# FIX 17: Add evaluation metrics
def calculate_metrics(true_points, pred_points, distance_threshold=20):
    """
    Calculate precision, recall, and F1 score based on distance matching.

    Parameters:
    - true_points: List of ground truth points [(x1, y1), (x2, y2), ...] in original coordinates
    - pred_points: List of predicted points [(x1, y1), (x1, y1), ...] in original coordinates
    - distance_threshold: Maximum distance (in pixels) to consider a prediction a true positive match

    Returns:
    - Dictionary with precision, recall, F1, TP, FP, FN counts
    """
    # Handle edge cases with empty lists
    if not true_points and not pred_points:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0, 'tp': 0, 'fp': 0, 'fn': 0}
    elif not true_points:  # Only predictions (all False Positives)
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': len(pred_points), 'fn': 0}
    elif not pred_points:  # Only ground truth (all False Negatives)
        return {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'tp': 0, 'fp': 0, 'fn': len(true_points)}

    # Convert to numpy arrays for efficient distance calculation
    true_points_arr = np.array(true_points)
    pred_points_arr = np.array(pred_points)

    num_true = len(true_points_arr)
    num_pred = len(pred_points_arr)

    # Calculate pairwise distances
    # Using scipy's cdist is efficient for this
    from scipy.spatial.distance import cdist
    distances = cdist(pred_points_arr, true_points_arr)  # Shape: (num_pred, num_true)

    # Matching using Hungarian algorithm (or greedy matching)
    # Greedy matching: Iterate through predictions, find closest valid true point, mark both as matched.

    true_matched = np.zeros(num_true, dtype=bool)
    pred_matched_indices = -np.ones(num_pred,
                                    dtype=int)  # Store index of matched true point for each pred, -1 if no match
    true_positives = 0

    # Iterate through predictions (can sort by confidence if available, but not here)
    for i in range(num_pred):
        pred_point = pred_points_arr[i]
        min_dist = float('inf')
        best_true_idx = -1

        # Find the closest *unmatched* true point within the threshold
        for j in range(num_true):
            if not true_matched[j]:  # Only consider unmatched true points
                dist = distances[i, j]
                if dist < min_dist and dist <= distance_threshold:
                    min_dist = dist
                    best_true_idx = j

        # If a valid match is found
        if best_true_idx != -1:
            true_positives += 1
            true_matched[best_true_idx] = True  # Mark true point as matched
            pred_matched_indices[i] = best_true_idx  # Mark prediction as matched (implicitly via TP count)

    # Calculate metrics
    false_positives = num_pred - true_positives
    false_negatives = num_true - true_positives

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0
    recall = true_positives / (true_positives + false_negatives) if (
                                                                                true_positives + false_negatives) > 0 else 0.0  # num_true is denominator here
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'tp': int(true_positives),
        'fp': int(false_positives),
        'fn': int(false_negatives)
    }


# FIX 18: Improved evaluation function (renamed predict_and_evaluate)
def predict_and_evaluate(model, data_loader, device, detection_threshold=0.5, distance_threshold=20, is_test_set=False):
    """
    Run model predictions on data_loader and optionally evaluate if labels are available.

    Parameters:
    - model: Trained UNet model
    - data_loader: DataLoader for test or validation data
    - device: Device to run evaluation on
    - detection_threshold: Threshold for peak detection from heatmap
    - distance_threshold: Distance threshold for matching predictions to ground truth
    - is_test_set: Boolean indicating if this is the test set (no ground truth)

    Returns:
    - all_predictions: List of dictionaries containing prediction info for each image.
    - avg_metrics: Dictionary with average evaluation metrics (if not is_test_set).\
    """
    model.eval()
    all_predictions_output = []  # Store predictions in submission format structure
    all_metrics_list = []

    with torch.no_grad():
        pbar = tqdm(data_loader, desc="Predicting/Evaluating", leave=False)
        batch_count = 0
        for batch in pbar:
            # Handle potential None batches from dataset/collate errors
            if batch is None:
                print(f"Warning: Skipping None batch in predict_and_evaluate (batch index approx {batch_count}).")
                batch_count += 1
                continue

            try:
                images = batch['image'].to(device)
                tomo_ids = batch['tomo_id']
                slice_idxs = batch['slice_idx']
                original_sizes = batch['original_size']  # Expecting N x (H, W)

                # Skip empty batches
                if images.size(0) == 0:
                    print(f"Warning: Empty batch encountered at index {batch_count} in predict_and_evaluate, skipping.")
                    batch_count += 1
                    continue

                # Forward pass - get logits
                logits = model(images)
                # Apply sigmoid to get probabilities [0, 1] for heatmap
                heatmaps = torch.sigmoid(logits)

                # Process each image in the batch
                for i in range(images.size(0)):
                    single_heatmap = heatmaps[i]  # Shape (1, H, W) or (H, W)
                    tomo_id = tomo_ids[i]
                    # --- FIX: 关键修改在这里 ---
                    slice_idx = slice_idxs[i]  # <--- 移除了 .item()
                    # --- 结束关键修改 ---
                    orig_size_hw = original_sizes[i] # original_sizes[i] 已经是一个形如 (H, W) 的元组
                    current_heatmap_size = tuple(single_heatmap.shape[-2:])  # (H, W) of heatmap

                    # Detect points from heatmap, scale back to original image size
                    pred_points_orig = detect_points_from_heatmap(
                        heatmap=single_heatmap,
                        threshold=detection_threshold,
                        min_distance=10,  # Min distance in heatmap pixel space
                        original_size=orig_size_hw,  # Target original image size (H, W)
                        current_size=current_heatmap_size  # Current heatmap size (H, W)
                    )

                    # Store predictions
                    # Format ready for prepare_submission
                    for x, y in pred_points_orig:
                        all_predictions_output.append({
                            'tomo_id': tomo_id,
                            'row_id': slice_idx,  # Assuming slice_idx corresponds to row_id
                            'Motor axis 0': x,
                            'Motor axis 1': y,
                            # Optional: add confidence score if needed
                        })

                    # --- Evaluation part (if not test set) ---
                    if not is_test_set:
                        if 'points_original' in batch:
                            # Ensure points_original is handled correctly (might be list of tensors/lists)
                            # This expects a list of (x, y) tuples for the i-th item
                            true_points_orig = batch['points_original'][i]
                            # If points_original is stored differently (e.g., padded tensor), adjust access

                            metrics = calculate_metrics(
                                true_points=true_points_orig,
                                pred_points=pred_points_orig,
                                distance_threshold=distance_threshold
                            )
                            all_metrics_list.append(metrics)
                        else:
                            print(f"Warning: 'points_original' key not found in batch for evaluation.")

                batch_count += 1

            except Exception as e:
                print(f"\nError processing batch {batch_count} in predict_and_evaluate: {e}")
                # import traceback
                # traceback.print_exc()
                batch_count += 1
                continue  # Continue to next batch

    # Calculate average metrics if evaluation was performed
    avg_metrics = {}
    if not is_test_set and all_metrics_list:
        # Sum up TP, FP, FN across all images for micro-average, or average per-image scores for macro-average
        # Example: Macro-average
        avg_precision = np.mean([m['precision'] for m in all_metrics_list])
        avg_recall = np.mean([m['recall'] for m in all_metrics_list])
        avg_f1 = np.mean([m['f1'] for m in all_metrics_list])

        avg_metrics = {
            'precision': avg_precision,
            'recall': avg_recall,
            'f1': avg_f1
        }
        print(f"\nAverage Validation Metrics (Macro) - Precision: {avg_metrics['precision']:.4f}, " \
              f"Recall: {avg_metrics['recall']:.4f}, F1: {avg_metrics['f1']:.4f}")
    elif not is_test_set:
        print("\nNo metrics calculated (all_metrics_list is empty).")

    return all_predictions_output, avg_metrics


# FIX 19: Function to prepare submission file (Revised - simpler, uses direct output from predict_and_evaluate)
def prepare_submission(predictions_list):
    """
    Create a submission DataFrame from predictions.

    Parameters:
    - predictions_list: List of dictionaries, where each dict represents one detected point
                      and has keys 'tomo_id', 'row_id', 'Motor axis 0', 'Motor axis 1'.
                      This list is generated directly by predict_and_evaluate.

    Returns:
    - DataFrame in submission format. Returns an empty DataFrame with correct columns if predictions_list is empty.
    """
    submission_columns = ['tomo_id', 'row_id', 'Motor axis 0', 'Motor axis 1']

    if not predictions_list:
        print("Warning: No predictions were generated. Creating empty submission file.")
        return pd.DataFrame(columns=submission_columns)

    try:
        submission_df = pd.DataFrame(predictions_list)

        # Ensure correct columns and order
        submission_df = submission_df[submission_columns]

        # Optional: Convert coordinate columns to integers if required by competition
        submission_df['Motor axis 0'] = submission_df['Motor axis 0'].round().astype(int)
        submission_df['Motor axis 1'] = submission_df['Motor axis 1'].round().astype(int)
        # Ensure row_id is integer
        submission_df['row_id'] = submission_df['row_id'].astype(int)

        return submission_df

    except KeyError as e:
        print(f"Error preparing submission: Missing expected key {e} in predictions list.")
        print("Returning empty submission DataFrame.")
        return pd.DataFrame(columns=submission_columns)
    except Exception as e:
        print(f"Unexpected error preparing submission: {e}")
        print("Returning empty submission DataFrame.")
        return pd.DataFrame(columns=submission_columns)


# FIX 20: Visualization functions
def visualize_predictions(image_path, true_points=None, pred_points=None, save_path=None):
    """
    Visualize image with ground truth and predicted points (coordinates in original image space).

    Parameters:
    - image_path: Path to image file
    - true_points: List of ground truth points [(x1, y1), (x2, y2), ...]\
    - pred_points: List of predicted points [(x1, y1), (x2, y1), ...]\
    - save_path: Path object or string to save visualization. If None, display only.
    """
    try:
        # Read image using PIL
        img = Image.open(image_path).convert('RGB')  # Convert to RGB for color circles
        img_draw = np.array(img)  # Convert to numpy array for drawing

        # Draw ground truth points (green circles)
        if true_points:
            for x, y in true_points:
                # Ensure coordinates are within bounds
                x, y = int(round(x)), int(round(y))
                if 0 <= x < img_draw.shape[1] and 0 <= y < img_draw.shape[0]:
                    cv2.circle(img_draw, (x, y), radius=7, color=(0, 255, 0), thickness=2)  # Green outline
                    # cv2.circle(img_draw, (x, y), radius=1, color=(0, 255, 0), thickness=-1) # Small center dot

        # Draw predicted points (red crosses or circles)
        if pred_points:
            for x, y in pred_points:
                # Ensure coordinates are within bounds
                x, y = int(round(x)), int(round(y))
                if 0 <= x < img_draw.shape[1] and 0 <= y < img_draw.shape[0]:
                    # Draw a cross
                    cv2.line(img_draw, (x - 5, y), (x + 5, y), color=(255, 0, 0), thickness=2)  # Red horizontal
                    cv2.line(img_draw, (x, y - 5), (x, y + 5), color=(255, 0, 0), thickness=2)  # Red vertical
                    # Alternatively, draw circles
                    # cv2.circle(img_draw, (x, y), radius=7, color=(255, 0, 0), thickness=2) # Red outline

        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(img_draw)
        plt.title(f"Image: {Path(image_path).name}")
        plt.axis('off')  # Hide axes

        # Add legend (simple version using plot handles)
        legend_elements = []
        if true_points:
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w', label='Ground Truth', markerfacecolor='g',
                           markeredgecolor='g', markersize=10))
        if pred_points:
            legend_elements.append(
                plt.Line2D([0], [0], marker='+', color='r', label='Prediction', linestyle='None', markersize=10,
                           markeredgewidth=2))

        if legend_elements:
            plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        # Save or display
        if save_path:
            plt.savefig(str(save_path), dpi=150, bbox_inches='tight')  # Lower DPI for speed if needed
            print(f"Visualization saved to {save_path}")
            plt.close()  # Close the figure to free memory
        else:
            plt.show()

    except FileNotFoundError:
        print(f"Error in visualization: Image not found at {image_path}")
    except Exception as e:
        print(f"Error during visualization for {image_path}: {e}")
        # import traceback
        # traceback.print_exc()
        if plt.gcf().get_axes():  # Close plot if it was created but failed
            plt.close()


# FIX 21: Main execution function (run_pipeline)
# UPDATED loss function and prepare_submission call
def run_pipeline(train_ratio=0.8, batch_size=16, num_epochs=30, learning_rate=1e-4):
    """
    Run the complete training, evaluation, and submission generation pipeline.

    Parameters:
    - train_ratio: Ratio of data to use for training vs validation.
    - batch_size: Batch size for DataLoaders.
    - num_epochs: Number of training epochs.
    - learning_rate: Initial learning rate for the optimizer.
    """
    global labels_dict  # Allow modification if needed, though preprocess happens earlier
    global device, accelerator_type  # Use globally defined device/type
    global train_slices, test_slices, SAMPLE_SUBMISSION  # Use global paths

    try:
        print("\n--- Starting Pipeline ---")
        print("Setting up data loaders...")

        # Split training data into train/val sets
        if not train_slices:
            print("Error: No training slices found. Cannot proceed.")
            return None, None, None

        train_paths, val_paths = train_test_split(
            train_slices,
            test_size=1 - train_ratio,
            random_state=42
        )
        print(f"Training images: {len(train_paths)}, Validation images: {len(val_paths)}")

        # --- Create Datasets ---
        train_dataset = BacterialMotorDataset(
            image_paths=train_paths,
            labels_dict=labels_dict,
            transform=train_transform,
            target_size=(target_height, target_width)  # Pass target size
        )

        val_dataset = BacterialMotorDataset(
            image_paths=val_paths,
            labels_dict=labels_dict,
            transform=val_transform,
            target_size=(target_height, target_width)  # Pass target size
        )

        # If test slices exist, create test dataset
        test_dataset = None
        if test_slices:
            test_dataset = BacterialMotorDataset(
                image_paths=test_slices,
                labels_dict=None,  # No labels for test set
                transform=val_transform,
                is_test=True,
                target_size=(target_height, target_width)  # Pass target size
            )
            print(f"Test images: {len(test_slices)}")
        else:
            print(
                "Warning: No test slices found. Submission file will be based on predictions for validation set if needed, or empty.")

        # --- Create DataLoaders ---
        num_workers = os.cpu_count() // 2 if os.cpu_count() else 2  # Use reasonable number of workers
        print(f"Using {num_workers} workers for DataLoaders.")

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=(accelerator_type == 'gpu'),  # Pin memory only if using GPU
            drop_last=True  # Drop last incomplete batch during training
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Use larger batch size for validation if memory allows
            shuffle=False,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=(accelerator_type == 'gpu')
        )

        # If test slices exist, create test dataloader
        test_loader = None
        if test_dataset is not None:
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size * 2,  # Larger batch size for inference
                shuffle=False,
                num_workers=num_workers,
                collate_fn=custom_collate_fn,
                pin_memory=(accelerator_type == 'gpu')
            )

        # --- Setup Model, Loss, Optimizer, Scheduler ---
        print("Setting up model, loss, optimizer, and scheduler...")

        # Model initialization (assuming UNet is defined above)
        # n_channels=1 for grayscale images, n_classes=1 for heatmap output
        model = UNet(n_channels=1, n_classes=1, bilinear=True, init_features=32).to(device)

        # Loss function: BCEWithLogitsLoss is suitable for heatmap regression
        # It combines sigmoid and BCELoss, which is numerically more stable
        criterion = nn.BCEWithLogitsLoss()

        # Optimizer
        optimizer = Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        # Reduces learning rate when validation loss stops improving
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

        # --- Train Model ---
        print("Starting training...")
        final_model, training_history = train_model(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            scheduler,
            num_epochs,
            accelerator_type,
            device
        )

        # --- Evaluate Model on Validation Set (if training was successful) ---
        validation_metrics = {}
        if final_model is not None and val_loader is not None:
            print("\n--- Evaluating on Validation Set ---")
            _, validation_metrics = predict_and_evaluate(
                final_model,
                val_loader,
                device,
                detection_threshold=0.5,  # Adjust as needed
                distance_threshold=20,  # Adjust as needed
                is_test_set=False
            )
        else:
            print("\nSkipping validation evaluation as model/loader is not available.")

        # --- Generate Submission File (if a test_loader exists) ---
        predictions_for_submission = []
        if test_loader is not None and final_model is not None:
            print("\n--- Generating Predictions for Submission ---")
            predictions_for_submission, _ = predict_and_evaluate(
                final_model,
                test_loader,
                device,
                detection_threshold=0.5,  # Use same threshold as validation or tune separately
                distance_threshold=20,
                # Distance threshold is not used for submission, but required by function signature
                is_test_set=True
            )

            submission_df = prepare_submission(predictions_for_submission)

            # Save submission file
            output_submission_path = Path("./submission.csv")
            submission_df.to_csv(output_submission_path, index=False)
            print(f"\nSubmission file saved to {output_submission_path} with {len(submission_df)} predictions.")
            print("Submission file head:")
            print(submission_df.head())
        else:
            print("\nSkipping submission generation as test data or trained model is not available.")
            # Create an empty submission.csv if no predictions were generated or test data was missing
            empty_submission_df = prepare_submission([])  # Pass empty list
            output_submission_path = Path("./submission.csv")
            empty_submission_df.to_csv(output_submission_path, index=False)
            print(f"\nCreated an empty submission file at {output_submission_path}.")


    except Exception as e:
        print(f"\n--- Pipeline encountered a critical error: {e} ---")
        # import traceback
        # traceback.print_exc() # For full stack trace in output

    return final_model, training_history, validation_metrics


# Run the pipeline
if __name__ == '__main__':
    # Adjust parameters as needed
    HYPERPARAMS = {
        'train_ratio': 0.9,  # Use more data for training if dataset is large
        'batch_size': 32,  # Adjust based on GPU memory (16 or 32 often reasonable)
        'num_epochs': 15,   # Adjust based on convergence observed in plots (start lower)
        # 'num_epochs': 1,  # Keep small for testing
        'learning_rate': 3e-4  # Common starting point for Adam
    }

    # Print system info
    print("--- System Information ---")
    print(f"PyTorch version: {torch.__version__}")
    # setup_device() call already prints GPU info if available
    print(f"Accelerator: {accelerator_type}, Device: {device}")
    print(f"Hyperparameters: {HYPERPARAMS}")

    # Run pipeline
    final_model, training_history, validation_metrics = run_pipeline(
        train_ratio=HYPERPARAMS['train_ratio'],
        batch_size=HYPERPARAMS['batch_size'],
        num_epochs=HYPERPARAMS['num_epochs'],
        learning_rate=HYPERPARAMS['learning_rate']
    )

    if final_model:
        print("\nPipeline completed successfully.")
        if validation_metrics:
            print(f"Final Validation Metrics: {validation_metrics}")
    else:
        print("\nPipeline execution failed.")

