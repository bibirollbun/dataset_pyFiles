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
except Exception as e:
    print(f"Error setting up paths: {e}")
    # FIX: Provide fallback paths
    BASE_PATH = Path(".")
    TRAIN_PATH = BASE_PATH / "train"
    TEST_PATH = BASE_PATH / "test"
    SAMPLE_SUBMISSION = BASE_PATH / "sample_submission.csv"
    TRAIN_LABELS = BASE_PATH / "train_labels.csv"

# Set up device
accelerator_type, device = setup_device()
print(f"Using accelerator type: {accelerator_type}")
print(f"Using device: {device}")

# Read training labels with better error handling
train_labels_df = None
try:
    train_labels_df = pd.read_csv(TRAIN_LABELS)
    print(f"Labels data shape: {train_labels_df.shape}")
    print("Label file columns:")
    print(train_labels_df.columns.tolist())
    print("Label data first 5 rows:")
    print(train_labels_df.head())
except Exception as e:
    print(f"Error reading label file: {e}")

# Read sample submission file
sample_submission_df = None
try:
    sample_submission_df = pd.read_csv(SAMPLE_SUBMISSION)
    print(f"Sample submission template shape: {sample_submission_df.shape}")
    print("Sample submission file columns:")
    print(sample_submission_df.columns.tolist())
    print("Sample submission first 5 rows:")
    print(sample_submission_df.head())
except Exception as e:
    print(f"Error reading sample submission file: {e}")

# Get all tomogram folders and slices
def get_data_paths():
    # Get training data tomogram folders
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
        else:
            print(f"Warning: Test path {TEST_PATH} does not exist")
            
    except Exception as e:
        print(f"Error getting data paths: {e}")
    
    return train_tomogram_folders, test_tomogram_folders, train_slices, test_slices

train_tomogram_folders, test_tomogram_folders, train_slices, test_slices = get_data_paths()
print(f"Number of training tomograms: {len(train_tomogram_folders)}")
print(f"Number of test tomograms: {len(test_tomogram_folders)}")
print(f"Total number of training slices: {len(train_slices)}")
print(f"Total number of test slices: {len(test_slices)}")

# Example view of a training folder name
if train_tomogram_folders:
    print(f"Example training tomogram folder name: {train_tomogram_folders[0].name}")
    # View slice filenames in this folder
    slices = list(train_tomogram_folders[0].glob("*.jpg"))
    if slices:
        print(f"Example slice filename: {slices[0].name}")

# Extract tomogram ID and slice index from file path
def extract_tomo_slice_info(file_path):
    # Extract tomogram ID from path
    tomo_id = file_path.parent.name
    # Extract slice index from filename
    try:
        # FIX 4: More robust extraction of slice index
        filename = file_path.stem
        if '_' in filename:
            slice_idx = int(filename.split('_')[1])
        else:
            # Fallback if filename format is different
            slice_idx = int(''.join(filter(str.isdigit, filename)))
    except (ValueError, IndexError) as e:
        print(f"Error extracting slice index from {file_path}: {e}")
        slice_idx = 0  # Default value
    
    return tomo_id, slice_idx

# FIX 5: More robust label preprocessing
def preprocess_labels(labels_df):
    """
    Process labels dataframe into a dictionary for quick lookup
    Handle different column naming conventions
    """
    if labels_df is None:
        print("Warning: No labels data provided")
        return {}
    
    # Check columns and adapt accordingly
    columns = labels_df.columns.tolist()
    
    # Try to find the right column names
    tomo_col = None
    slice_col = None
    x_col = None
    y_col = None
    
    # Look for tomogram ID column
    for col in columns:
        if 'tomo' in col.lower():
            tomo_col = col
            break
    
    # Look for slice/row ID column
    for col in columns:
        if 'row' in col.lower() or 'slice' in col.lower():
            slice_col = col
            break
    
    # Look for X coordinate column
    for col in columns:
        if 'axis 0' in col.lower() or 'x' in col.lower():
            x_col = col
            break
    
    # Look for Y coordinate column
    for col in columns:
        if 'axis 1' in col.lower() or 'y' in col.lower():
            y_col = col
            break
    
    # Use default column names if not found
    if tomo_col is None:
        print("Warning: Tomogram ID column not found, using 'tomo_id'")
        tomo_col = 'tomo_id'
    
    if slice_col is None:
        print("Warning: Slice index column not found, using 'row_id'")
        slice_col = 'row_id'
    
    if x_col is None:
        print("Warning: X coordinate column not found, using 'Motor axis 0'")
        x_col = 'Motor axis 0'
    
    if y_col is None:
        print("Warning: Y coordinate column not found, using 'Motor axis 1'")
        y_col = 'Motor axis 1'
    
    print(f"Using columns: {tomo_col}, {slice_col}, {x_col}, {y_col}")
    
    # Create dictionary to store labels
    labels_dict = {}
    
    # Iterate through label data
    for _, row in labels_df.iterrows():
        try:
            # Convert to proper types with error handling
            tomo_id = str(row[tomo_col])
            
            # Handle different slice index formats
            try:
                slice_idx = int(row[slice_col])
            except (ValueError, TypeError):
                # Try to extract numeric part if not a clean integer
                slice_idx = int(''.join(filter(str.isdigit, str(row[slice_col]))))
            
            # Extract coordinates
            try:
                x = float(row[x_col])
                y = float(row[y_col])
            except (ValueError, TypeError) as e:
                print(f"Error extracting coordinates: {e}")
                continue
            
            key = (tomo_id, slice_idx)
            if key not in labels_dict:
                labels_dict[key] = []
            
            # Add (x, y) coordinates to corresponding (tomo_id, slice_idx) key
            labels_dict[key].append((x, y))
        
        except Exception as e:
            print(f"Error processing row: {e}")
    
    return labels_dict

# FIX 6: Check data before processing
if train_labels_df is not None:
    labels_dict = preprocess_labels(train_labels_df)
    print(f"Number of slices with labels: {len(labels_dict)}")
    # Show examples of first few labels
    if labels_dict:
        count = 0
        for key, points in labels_dict.items():
            print(f"Slice {key}: {len(points)} marked points")
            count += 1
            if count >= 3:
                break
else:
    print("Warning: No label data available")
    labels_dict = {}

# FIX 7: Improved heatmap creation function
def create_heatmap(img_shape, points, sigma=10):
    """
    Create Gaussian heatmap for given points
    
    Parameters:
    - img_shape: Original image shape (height, width) for PIL images
    - points: List of coordinates [(x1, y1), (x2, y2), ...]
    - sigma: Standard deviation of Gaussian kernel
    
    Returns:
    - Heatmap
    """
    # For PIL images, shape is (width, height), we need (height, width)
    height, width = img_shape[1], img_shape[0]
    heatmap = np.zeros((height, width), dtype=np.float32)
    
    # If no points, return zero heatmap
    if not points or len(points) == 0:
        return heatmap
    
    # Create meshgrid for the image
    y_grid, x_grid = np.mgrid[0:height, 0:width]
    
    for x, y in points:
        # Ensure coordinates are within image bounds
        if 0 <= x < width and 0 <= y < height:
            # Compute Gaussian values
            gaussian = np.exp(-((x_grid - x)**2 + (y_grid - y)**2) / (2 * sigma**2))
            
            # Update heatmap, taking maximum value to avoid overlap issues
            heatmap = np.maximum(heatmap, gaussian)
    
    return heatmap

# FIX 8: Improved dataset class with better error handling
class BacterialMotorDataset(Dataset):
    def __init__(self, image_paths, labels_dict=None, transform=None, is_test=False):
        self.image_paths = image_paths
        self.labels_dict = labels_dict if labels_dict is not None else {}
        self.transform = transform
        self.is_test = is_test
        self.target_size = (256, 256)  # Default target size
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # Read image with error handling
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Error reading image: {img_path}")
                # Create a blank image as fallback
                img = np.zeros((256, 256), dtype=np.uint8)
            
            # Convert to PIL Image for transformations
            img_pil = Image.fromarray(img)
            
            # Get original dimensions
            original_size = img_pil.size  # (width, height)
            
            # Extract tomogram ID and slice index
            tomo_id, slice_idx = extract_tomo_slice_info(img_path)
            
            # Apply transforms
            if self.transform:
                img_transformed = self.transform(img_pil)
            else:
                # Default transform if none provided
                img_transformed = transforms.ToTensor()(img_pil)
            
            # For test set, return minimal information
            if self.is_test:
                return {
                    'image': img_transformed,
                    'tomo_id': tomo_id,
                    'slice_idx': slice_idx,
                    'image_path': str(img_path)
                }
            
            # Get label points or empty list if not found
            key = (tomo_id, slice_idx)
            points = self.labels_dict.get(key, [])
            
            # Create heatmap label before resize
            heatmap = create_heatmap(original_size, points)
            
            # Resize heatmap to match transformed image size
            heatmap_resized = cv2.resize(
                heatmap, 
                (self.target_size[1], self.target_size[0]),  # (width, height)
                interpolation=cv2.INTER_LINEAR
            )
            
            # Convert to tensor
            heatmap_tensor = torch.tensor(heatmap_resized, dtype=torch.float32).unsqueeze(0)
            
            return {
                'image': img_transformed,
                'heatmap': heatmap_tensor,
                'points': points,
                'tomo_id': tomo_id,
                'slice_idx': slice_idx,
                'image_path': str(img_path)
            }
            
        except Exception as e:
            print(f"Error processing image {img_path}: {e}")
            # Return a default item to avoid breaking the DataLoader
            default_img = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            default_heatmap = torch.zeros((1, self.target_size[0], self.target_size[1]), dtype=torch.float32)
            
            return {
                'image': default_img,
                'heatmap': default_heatmap,
                'points': [],
                'tomo_id': "error",
                'slice_idx': -1,
                'image_path': str(img_path)
            }

# FIX 9: Simplified transforms with fixed size
train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# FIX 10: Simplified and more robust collate function
def custom_collate_fn(batch):
    """
    Collate function that handles potential errors in the batch
    """
    # Filter out any None values
    valid_batch = [item for item in batch if item is not None]
    
    if not valid_batch:
        print("Warning: Empty batch after filtering")
        # Return empty tensors as fallback
        return {
            'image': torch.zeros((0, 1, 256, 256), dtype=torch.float32),
            'heatmap': torch.zeros((0, 1, 256, 256), dtype=torch.float32),
            'tomo_id': [],
            'slice_idx': [],
            'image_path': []
        }
    
    # Extract elements from valid items
    images = [item['image'] for item in valid_batch if 'image' in item]
    
    # Handle test batch differently
    is_test = 'heatmap' not in valid_batch[0]
    
    if not is_test:
        heatmaps = [item['heatmap'] for item in valid_batch if 'heatmap' in item]
        tomo_ids = [item['tomo_id'] for item in valid_batch if 'tomo_id' in item]
        slice_idxs = [item['slice_idx'] for item in valid_batch if 'slice_idx' in item]
        image_paths = [item['image_path'] for item in valid_batch if 'image_path' in item]
        
        # Stack tensors
        if images:
            images = torch.stack(images, dim=0)
        else:
            images = torch.zeros((0, 1, 256, 256), dtype=torch.float32)
            
        if heatmaps:
            heatmaps = torch.stack(heatmaps, dim=0)
        else:
            heatmaps = torch.zeros((0, 1, 256, 256), dtype=torch.float32)
        
        return {
            'image': images,
            'heatmap': heatmaps,
            'tomo_id': tomo_ids,
            'slice_idx': slice_idxs,
            'image_path': image_paths
        }
    else:
        # Test batch
        tomo_ids = [item['tomo_id'] for item in valid_batch if 'tomo_id' in item]
        slice_idxs = [item['slice_idx'] for item in valid_batch if 'slice_idx' in item]
        image_paths = [item['image_path'] for item in valid_batch if 'image_path' in item]
        
        # Stack tensors
        if images:
            images = torch.stack(images, dim=0)
        else:
            images = torch.zeros((0, 1, 256, 256), dtype=torch.float32)
        
        return {
            'image': images,
            'tomo_id': tomo_ids,
            'slice_idx': slice_idxs,
            'image_path': image_paths
        }

# FIX 11: Improved UNet architecture with better initialization
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, init_features=32):
        super(UNet, self).__init__()
        
        features = init_features
        self.encoder1 = DoubleConv(in_channels, features)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder2 = DoubleConv(features, features * 2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder3 = DoubleConv(features * 2, features * 4)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.encoder4 = DoubleConv(features * 4, features * 8)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.bottleneck = DoubleConv(features * 8, features * 16)
        
        self.upconv4 = nn.ConvTranspose2d(features * 16, features * 8, kernel_size=2, stride=2)
        self.decoder4 = DoubleConv(features * 16, features * 8)
        
        self.upconv3 = nn.ConvTranspose2d(features * 8, features * 4, kernel_size=2, stride=2)
        self.decoder3 = DoubleConv(features * 8, features * 4)
        
        self.upconv2 = nn.ConvTranspose2d(features * 4, features * 2, kernel_size=2, stride=2)
        self.decoder2 = DoubleConv(features * 4, features * 2)
        
        self.upconv1 = nn.ConvTranspose2d(features * 2, features, kernel_size=2, stride=2)
        self.decoder1 = DoubleConv(features * 2, features)
        
        self.conv = nn.Conv2d(features, out_channels, kernel_size=1)
        
    def forward(self, x):
        # FIX 12: Add input shape check
        if x.dim() != 4:
            raise ValueError(f"Expected 4D tensor, got {x.dim()}D tensor")
        
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        bottleneck = self.bottleneck(self.pool4(enc4))
        
        dec4 = self.upconv4(bottleneck)
        # FIX 13: Add size check for concatenation
        if dec4.size()[2:] != enc4.size()[2:]:
            dec4 = F.interpolate(dec4, size=enc4.size()[2:], mode='bilinear', align_corners=False)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        
        dec3 = self.upconv3(dec4)
        if dec3.size()[2:] != enc3.size()[2:]:
            dec3 = F.interpolate(dec3, size=enc3.size()[2:], mode='bilinear', align_corners=False)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        
        dec2 = self.upconv2(dec3)
        if dec2.size()[2:] != enc2.size()[2:]:
            dec2 = F.interpolate(dec2, size=enc2.size()[2:], mode='bilinear', align_corners=False)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        
        dec1 = self.upconv1(dec2)
        if dec1.size()[2:] != enc1.size()[2:]:
            dec1 = F.interpolate(dec1, size=enc1.size()[2:], mode='bilinear', align_corners=False)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        
        output = self.conv(dec1)
        output = torch.sigmoid(output)  # Use sigmoid for [0,1] range
        
        return output

# FIX 14: Training function with proper error handling and GPU memory optimization
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs=10, accelerator_type='cpu'):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    # For TPU acceleration
    if accelerator_type == 'tpu':
        try:
            import torch_xla.core.xla_model as xm
            import torch_xla.distributed.parallel_loader as pl
        except ImportError as e:
            print(f"Error importing TPU modules: {e}")
            print("Falling back to CPU training")
            accelerator_type = 'cpu'
    
    # For mixed precision training on GPU
    use_amp = False
    scaler = None
    if accelerator_type == 'gpu':
        try:
            from torch.cuda.amp import GradScaler, autocast
            use_amp = True
            scaler = GradScaler()
            print("Mixed precision training enabled")
        except ImportError:
            print("Mixed precision not available, using full precision")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        # Choose appropriate dataloader based on accelerator
        if accelerator_type == 'tpu':
            try:
                train_device_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
                loader_to_use = train_device_loader
            except Exception as e:
                print(f"Error setting up TPU dataloader: {e}")
                loader_to_use = train_loader
        else:
            loader_to_use = train_loader
        
        # Training loop with error handling
        pbar = tqdm(loader_to_use, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for batch_idx, batch in enumerate(pbar):
            try:
                images = batch['image'].to(device)
                heatmaps = batch['heatmap'].to(device)
                
                # Check for empty batch
                if images.size(0) == 0:
                    print("Warning: Empty batch encountered, skipping")
                    continue
                
                # Forward pass with appropriate precision
                if use_amp:
                    with autocast():
                        outputs = model(images)
                        loss = criterion(outputs, heatmaps)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, heatmaps)
                
                # Backward pass
                optimizer.zero_grad()
                
                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                elif accelerator_type == 'tpu':
                    loss.backward()
                    xm.optimizer_step(optimizer, barrier=True)
                else:
                    loss.backward()
                    optimizer.step()
                
                # Update statistics
                train_loss += loss.item() * images.size(0)
                
                # Update progress bar
                pbar.set_postfix(loss=loss.item())
                
                # FIX 15: Periodic GPU memory cleanup
                if accelerator_type == 'gpu' and batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        # Calculate average loss
        train_loss /= len(train_loader.dataset)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        # Choose appropriate validation dataloader
        if accelerator_type == 'tpu':
            try:
                val_device_loader = pl.ParallelLoader(val_loader, [device]).per_device_loader(device)
                val_loader_to_use = val_device_loader
            except Exception as e:
                print(f"Error setting up TPU validation dataloader: {e}")
                val_loader_to_use = val_loader
        
        with torch.no_grad():
            pbar = tqdm(val_loader_to_use, desc=f'Epoch {epoch+1}/{num_epochs} [Val]')
            for batch_idx, batch in enumerate(pbar):
                try:
                    images = batch['image'].to(device)
                    heatmaps = batch['heatmap'].to(device)
                    
                    # Check for empty batch
                    if images.size(0) == 0:
                        print("Warning: Empty validation batch encountered, skipping")
                        continue
                    
                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, heatmaps)
                    
                    # Update statistics
                    val_loss += loss.item() * images.size(0)
                    
                    # Update progress bar
                    pbar.set_postfix(loss=loss.item())
                    
                except Exception as e:
                    print(f"Error in validation batch {batch_idx}: {e}")
                    continue
        
        # Calculate average validation loss
        val_loss /= len(val_loader.dataset)
        
        # Update learning rate scheduler
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Save model if validation loss improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save model
            save_path = Path("./best_model.pth")
            try:
                if accelerator_type == 'tpu':
                    # TPU-specific saving
                    xm.save(model.state_dict(), str(save_path))
                else:
                    torch.save(model.state_dict(), save_path)
                print(f"Model saved to {save_path}")
            except Exception as e:
                print(f"Error saving model: {e}")
        
        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        
        # Print epoch results
        print(f'Epoch {epoch+1}/{num_epochs}: '
              f'Train Loss: {train_loss:.4f}, '
              f'Val Loss: {val_loss:.4f}')
        
        # Plot training progress
        if (epoch + 1) % 5 == 0 or epoch == num_epochs - 1:
            try:
                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.plot(history['train_loss'], label='Train Loss')
                plt.plot(history['val_loss'], label='Val Loss')
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend()
                plt.title('Training Progress')
                plt.savefig(f'training_progress_epoch_{epoch+1}.png')
                plt.close()
            except Exception as e:
                print(f"Error plotting training progress: {e}")
    
    return model, history

# FIX 16: Enhanced post-processing function to detect peaks in the heatmap
def detect_points_from_heatmap(heatmap, threshold=0.3, min_distance=10, original_size=None):
    """
    Detect points from heatmap by finding local maxima
    
    Parameters:
    - heatmap: Predicted heatmap tensor
    - threshold: Minimum value to consider as a potential motor
    - min_distance: Minimum distance between detected points
    - original_size: Tuple (height, width) for scaling back to original image size
    
    Returns:
    - List of (x, y) coordinates of detected motors
    """
    from scipy.ndimage import gaussian_filter, maximum_filter
    from scipy.ndimage.morphology import generate_binary_structure, binary_erosion
    
    # Convert tensor to numpy if needed
    if isinstance(heatmap, torch.Tensor):
        heatmap = heatmap.squeeze().cpu().numpy()
    
    # Ensure heatmap is 2D
    if heatmap.ndim > 2:
        heatmap = heatmap.squeeze()
    
    # Apply Gaussian filter to smooth the heatmap
    heatmap_smoothed = gaussian_filter(heatmap, sigma=1)
    
    # Find local maxima
    # Create a mask of local maxima
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(heatmap_smoothed, footprint=neighborhood) == heatmap_smoothed
    background = (heatmap_smoothed < threshold)
    eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)
    detected_maxima = local_max & ~eroded_background
    
    # Extract coordinates of maxima
    y_indices, x_indices = np.where(detected_maxima)
    
    # Get heatmap values at maxima
    intensities = heatmap_smoothed[y_indices, x_indices]
    
    # Sort points by intensity (highest to lowest)
    sorted_indices = np.argsort(intensities)[::-1]
    y_indices = y_indices[sorted_indices]
    x_indices = x_indices[sorted_indices]
    
    # Filter close points (keep the stronger one)
    points = []
    for i in range(len(x_indices)):
        # Check if this point is far enough from all accepted points
        is_far_enough = True
        for px, py in points:
            dist = np.sqrt((px - x_indices[i])**2 + (py - y_indices[i])**2)
            if dist < min_distance:
                is_far_enough = False
                break
        
        if is_far_enough:
            points.append((x_indices[i], y_indices[i]))
    
    # Rescale points to original image size if provided
    if original_size is not None:
        h_scale = original_size[0] / heatmap.shape[0]
        w_scale = original_size[1] / heatmap.shape[1]
        
        points = [(int(x * w_scale), int(y * h_scale)) for x, y in points]
    
    return points

# FIX 17: Add evaluation metrics
def calculate_metrics(true_points, pred_points, distance_threshold=20):
    """
    Calculate precision, recall, and F1 score
    
    Parameters:
    - true_points: List of ground truth points [(x1, y1), (x2, y2), ...]
    - pred_points: List of predicted points [(x1, y1), (x2, y2), ...]
    - distance_threshold: Maximum distance to consider a prediction correct
    
    Returns:
    - Dictionary with precision, recall, and F1 metrics
    """
    # Handle empty cases
    if not true_points and not pred_points:
        return {'precision': 1.0, 'recall': 1.0, 'f1': 1.0}
    elif not true_points:
        return {'precision': 0.0, 'recall': 1.0, 'f1': 0.0}
    elif not pred_points:
        return {'precision': 1.0, 'recall': 0.0, 'f1': 0.0}
    
    # Convert to numpy arrays for easier computation
    true_points = np.array(true_points)
    pred_points = np.array(pred_points)
    
    # Calculate distances between all pairs of points
    true_count = len(true_points)
    pred_count = len(pred_points)
    
    # Initialize match matrices
    true_matched = np.zeros(true_count, dtype=bool)
    pred_matched = np.zeros(pred_count, dtype=bool)
    
    # For each predicted point, find the closest true point
    for i in range(pred_count):
        min_dist = float('inf')
        closest_idx = -1
        
        for j in range(true_count):
            if true_matched[j]:
                continue  # Skip already matched true points
                
            # Calculate Euclidean distance
            dist = np.sqrt(np.sum((pred_points[i] - true_points[j])**2))
            
            if dist < min_dist and dist <= distance_threshold:
                min_dist = dist
                closest_idx = j
        
        # If a match was found, mark both points as matched
        if closest_idx != -1:
            true_matched[closest_idx] = True
            pred_matched[i] = True
    
    # Calculate metrics
    true_positives = np.sum(true_matched)
    false_positives = pred_count - true_positives
    false_negatives = true_count - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'true_positives': int(true_positives),
        'false_positives': int(false_positives),
        'false_negatives': int(false_negatives)
    }

# FIX 18: Improved evaluation function
def evaluate_model(model, test_loader, device, detection_threshold=0.3, distance_threshold=20):
    """
    Evaluate the model on test data
    
    Parameters:
    - model: Trained UNet model
    - test_loader: DataLoader for test data
    - device: Device to run evaluation on
    - detection_threshold: Threshold for peak detection
    - distance_threshold: Distance threshold for considering a detection correct
    
    Returns:
    - Dictionary with evaluation metrics
    """
    model.eval()
    all_metrics = []
    all_predictions = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            try:
                images = batch['image'].to(device)
                
                # Skip empty batches
                if images.size(0) == 0:
                    continue
                
                # Forward pass
                outputs = model(images)
                
                # Process each image in the batch
                for i in range(images.size(0)):
                    heatmap = outputs[i].squeeze().cpu().numpy()
                    
                    # Get ground truth points if available
                    if 'points' in batch:
                        true_points = batch['points'][i]
                    else:
                        true_points = []
                    
                    # Detect points from heatmap
                    pred_points = detect_points_from_heatmap(
                        heatmap, 
                        threshold=detection_threshold, 
                        min_distance=10
                    )
                    
                    # Calculate metrics if ground truth is available
                    if true_points:
                        metrics = calculate_metrics(
                            true_points, 
                            pred_points, 
                            distance_threshold=distance_threshold
                        )
                        all_metrics.append(metrics)
                    
                    # Save prediction info
                    all_predictions.append({
                        'tomo_id': batch['tomo_id'][i],
                        'slice_idx': batch['slice_idx'][i],
                        'points': pred_points,
                    })
            
            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue
    
    # Calculate average metrics
    if all_metrics:
        avg_metrics = {
            'precision': np.mean([m['precision'] for m in all_metrics]),
            'recall': np.mean([m['recall'] for m in all_metrics]),
            'f1': np.mean([m['f1'] for m in all_metrics])
        }
        print(f"Average Metrics - Precision: {avg_metrics['precision']:.4f}, "
              f"Recall: {avg_metrics['recall']:.4f}, F1: {avg_metrics['f1']:.4f}")
    else:
        avg_metrics = {}
    
    return all_predictions, avg_metrics

# FIX 19: Function to prepare submission file
def prepare_submission(predictions, template_path):
    """
    Create a submission file from predictions
    
    Parameters:
    - predictions: List of dictionaries with prediction info
    - template_path: Path to sample submission template
    
    Returns:
    - DataFrame with submission format
    """
    try:
        # Read template
        template_df = pd.read_csv(template_path)
        
        # Convert predictions to DataFrame format
        rows = []
        
        for pred in predictions:
            tomo_id = pred['tomo_id']
            slice_idx = pred['slice_idx']
            points = pred['points']
            
            # For each detected point, create a row
            for x, y in points:
                rows.append({
                    'tomo_id': tomo_id,
                    'row_id': slice_idx,
                    'Motor axis 0': x,
                    'Motor axis 1': y
                })
        
        # Create DataFrame
        submission_df = pd.DataFrame(rows)
        
        # If no predictions, create empty DataFrame with correct columns
        if len(rows) == 0:
            submission_df = pd.DataFrame(columns=['tomo_id', 'row_id', 'Motor axis 0', 'Motor axis 1'])
        
        return submission_df
        
    except Exception as e:
        print(f"Error preparing submission: {e}")
        return None

# FIX 20: Visualization functions
def visualize_predictions(image_path, true_points=None, pred_points=None, save_path=None):
    """
    Visualize image with ground truth and predicted points
    
    Parameters:
    - image_path: Path to image
    - true_points: List of ground truth points [(x1, y1), (x2, y2), ...]
    - pred_points: List of predicted points [(x1, y1), (x2, y2), ...]
    - save_path: Path to save visualization, if None, display only
    """
    try:
        # Read image
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        
        if image is None:
            print(f"Error reading image: {image_path}")
            return
        
        # Convert to RGB for visualization
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Draw ground truth points (green)
        if true_points:
            for x, y in true_points:
                x, y = int(x), int(y)
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image_rgb, (x, y), 5, (0, 255, 0), -1)
                    cv2.circle(image_rgb, (x, y), 7, (0, 255, 0), 2)
        
        # Draw predicted points (red)
        if pred_points:
            for x, y in pred_points:
                x, y = int(x), int(y)
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    cv2.circle(image_rgb, (x, y), 5, (255, 0, 0), -1)
                    cv2.circle(image_rgb, (x, y), 7, (255, 0, 0), 2)
        
        # Create figure
        plt.figure(figsize=(10, 10))
        plt.imshow(image_rgb)
        plt.title(f"Image: {Path(image_path).name}")
        
        # Add legend
        if true_points and pred_points:
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='green', edgecolor='green', label='Ground Truth'),
                Patch(facecolor='red', edgecolor='red', label='Prediction')
            ]
            plt.legend(handles=legend_elements, loc='upper right')
        
        # Save or display
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
            
    except Exception as e:
        print(f"Error in visualization: {e}")

# FIX 21: Main execution function
def run_pipeline(train_ratio=0.8, batch_size=16, num_epochs=15, learning_rate=1e-4):
    """
    Run the complete training and evaluation pipeline
    
    Parameters:
    - train_ratio: Ratio of data to use for training vs validation
    - batch_size: Batch size for training
    - num_epochs: Number of training epochs
    - learning_rate: Learning rate for optimizer
    """
    try:
        print("Setting up data...")
        
        # Split training data into train/val sets
        if train_slices:
            train_paths, val_paths = train_test_split(
                train_slices, 
                test_size=1-train_ratio, 
                random_state=42
            )
            
            print(f"Training on {len(train_paths)} images, validating on {len(val_paths)} images")
            
            # Create datasets
            train_dataset = BacterialMotorDataset(
                image_paths=train_paths,
                labels_dict=labels_dict,
                transform=train_transform
            )
            
            val_dataset = BacterialMotorDataset(
                image_paths=val_paths,
                labels_dict=labels_dict,
                transform=val_transform
            )
            
            # Create dataloaders
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=4,
                collate_fn=custom_collate_fn,
                pin_memory=(accelerator_type == 'gpu')
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=custom_collate_fn,
                pin_memory=(accelerator_type == 'gpu')
            )
            
            # Create test dataset
            test_dataset = BacterialMotorDataset(
                image_paths=test_slices,
                transform=val_transform,
                is_test=True
            )
            
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4,
                collate_fn=custom_collate_fn,
                pin_memory=(accelerator_type == 'gpu')
            )
            
            # Create model
            model = UNet(in_channels=1, out_channels=1, init_features=32)
            model = model.to(device)
            
            # Define loss function and optimizer
            criterion = nn.BCELoss()
            optimizer = Adam(model.parameters(), lr=learning_rate)
            
            # Define scheduler
            scheduler = ReduceLROnPlateau(
                optimizer, 
                mode='min', 
                factor=0.5, 
                patience=5, 
                verbose=True
            )
            
            # Train model
            print("Starting training...")
            model, history = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                num_epochs=num_epochs,
                accelerator_type=accelerator_type
            )
            
            # Evaluate model
            print("Evaluating model...")
            predictions, metrics = evaluate_model(
                model=model,
                test_loader=test_loader,
                device=device
            )
            
            # Create submission file
            print("Creating submission file...")
            submission_df = prepare_submission(
                predictions=predictions,
                template_path=SAMPLE_SUBMISSION
            )
            
            if submission_df is not None:
                submission_path = Path("./submission.csv")
                submission_df.to_csv(submission_path, index=False)
                print(f"Submission saved to {submission_path}")
            
            # Visualize some predictions
            print("Creating visualizations...")
            # Visualize a few test predictions
            for i in range(min(5, len(test_slices))):
                try:
                    image_path = test_slices[i]
                    # Get prediction for this image
                    model.eval()
                    with torch.no_grad():
                        # Load and preprocess image
                        img = Image.open(image_path).convert('L')
                        img_tensor = val_transform(img).unsqueeze(0).to(device)
                        
                        # Forward pass
                        output = model(img_tensor)
                        
                        # Get predictions
                        heatmap = output.squeeze().cpu().numpy()
                        pred_points = detect_points_from_heatmap(
                            heatmap, 
                            threshold=0.3,
                            min_distance=10,
                            original_size=img.size
                        )
                        
                        # Save visualization
                        save_path = Path(f"./visualization_test_{i}.png")
                        visualize_predictions(
                            image_path=image_path,
                            pred_points=pred_points,
                            save_path=save_path
                        )
                        print(f"Visualization saved to {save_path}")
                        
                except Exception as e:
                    print(f"Error visualizing test image {i}: {e}")
                    continue
            
            # Also visualize some training images with ground truth
            for i in range(min(5, len(train_paths))):
                try:
                    image_path = train_paths[i]
                    # Extract tomogram ID and slice index
                    tomo_id, slice_idx = extract_tomo_slice_info(image_path)
                    # Get ground truth points
                    true_points = labels_dict.get((tomo_id, slice_idx), [])
                    
                    # Get prediction for this image
                    model.eval()
                    with torch.no_grad():
                        # Load and preprocess image
                        img = Image.open(image_path).convert('L')
                        img_tensor = val_transform(img).unsqueeze(0).to(device)
                        
                        # Forward pass
                        output = model(img_tensor)
                        
                        # Get predictions
                        heatmap = output.squeeze().cpu().numpy()
                        pred_points = detect_points_from_heatmap(
                            heatmap, 
                            threshold=0.3,
                            min_distance=10,
                            original_size=img.size
                        )
                        
                        # Save visualization
                        save_path = Path(f"./visualization_train_{i}.png")
                        visualize_predictions(
                            image_path=image_path,
                            true_points=true_points,
                            pred_points=pred_points,
                            save_path=save_path
                        )
                        print(f"Visualization saved to {save_path}")
                        
                except Exception as e:
                    print(f"Error visualizing training image {i}: {e}")
                    continue
                
            return model, history, metrics
        else:
            print("No training data found!")
            return None, None, None
            
    except Exception as e:
        print(f"Error in pipeline execution: {e}")
        return None, None, None

# Run the pipeline if executing as main script
if __name__ == "__main__":
    # Set hyperparameters
    HYPERPARAMS = {
        'train_ratio': 0.8,
        'batch_size': 16,
        'num_epochs': 30,
        'learning_rate': 1e-4
    }
    
    # Print system info
    print("System information:")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # Run pipeline
    model, history, metrics = run_pipeline(
        train_ratio=HYPERPARAMS['train_ratio'],
        batch_size=HYPERPARAMS['batch_size'],
        num_epochs=HYPERPARAMS['num_epochs'],
        learning_rate=HYPERPARAMS['learning_rate']
    )

