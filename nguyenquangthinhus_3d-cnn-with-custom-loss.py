## Imports

import os
import glob
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Define global constants
DATA_DIR = '/kaggle/input/byu-locating-bacterial-flagellar-motors-2025'
TRAIN_CSV = os.path.join(DATA_DIR, 'train_labels.csv')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
TEST_DIR = os.path.join(DATA_DIR, 'test')
OUTPUT_DIR = './'
MODEL_DIR = './models'

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True) 
os.makedirs(MODEL_DIR, exist_ok=True)

# Set device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Set seeds for reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True



# Load the training labels
train_labels = pd.read_csv(TRAIN_CSV)

# Display basic information
print("Training dataset shape:", train_labels.shape)
print("\nColumns in the dataset:")
display(train_labels.columns)

# Basic statistics
print("\nBasic statistics:")
display(train_labels.describe())

# Count unique tomograms
unique_tomo_count = train_labels['tomo_id'].nunique()
print(f"\nNumber of unique tomograms: {unique_tomo_count}")

# Check the distribution of motors per tomogram
motors_per_tomo = train_labels.groupby('tomo_id')['Number of motors'].first().value_counts().sort_index()
print("\nMotors per tomogram distribution:")
print(motors_per_tomo)

# Display a few sample rows
print("\nSample rows:")
display(train_labels.head())

# Check for missing values
print("\nMissing values per column:")
display(train_labels.isnull().sum())

# Check the range of tomogram sizes
print("\nTomogram size ranges:")
print("Z-axis (slices):", train_labels['Array shape (axis 0)'].min(), "to", train_labels['Array shape (axis 0)'].max())
print("X-axis (width):", train_labels['Array shape (axis 1)'].min(), "to", train_labels['Array shape (axis 1)'].max()) 
print("Y-axis (height):", train_labels['Array shape (axis 2)'].min(), "to", train_labels['Array shape (axis 2)'].max())

# Check voxel spacing distribution
print("\nVoxel spacing distribution:")
display(train_labels['Voxel spacing'].value_counts().sort_index())

# Visualize a sample tomogram
# Get one tomogram ID
sample_tomo_id = train_labels['tomo_id'].iloc[0]
print(f"\nVisualizing sample tomogram: {sample_tomo_id}")

# Check the folder structure
sample_folder = os.path.join(TRAIN_DIR, sample_tomo_id)
if os.path.exists(sample_folder):
    # List files in the folder
    files = sorted(glob.glob(os.path.join(sample_folder, '*.jpg')))
    print(f"Number of slice files: {len(files)}")
    
    if files:
        # Load one slice to check dimensions
        sample_slice = Image.open(files[0])
        print(f"Sample slice dimensions: {sample_slice.size}")
        
        # Display a few slices
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Pick slices from beginning, middle, and end
        indices = [0, len(files)//2, len(files)-1]
        for i, idx in enumerate(indices):
            img = Image.open(files[idx])
            axes[i].imshow(img, cmap='gray')
            axes[i].set_title(f"Slice {idx}")
            axes[i].axis('off')
        
        plt.tight_layout()


import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import rotate, zoom
import torch
from torch.utils.data import Dataset

class TomogramDataset(Dataset):
    """
    Dataset for loading 3D tomograms from stacks of 2D JPG slices with augmentation.
    """
    def __init__(self, csv_file, root_dir, train=True, max_slices=64, target_size=(128, 128), 
                 augment_prob=0.5, rotation_range=15, contrast_range=(0.8, 1.2), 
                 brightness_range=(-0.1, 0.1), noise_level=0.02, flip_prob=0.3,
                 zoom_range=(0.9, 1.1)):
        """
        Initialize the dataset.
        
        Args:
            csv_file (str): Path to the CSV file with annotations.
            root_dir (str): Directory with all the tomogram slice directories.
            train (bool): Whether this is for training (enables augmentations).
            max_slices (int): Maximum number of slices to use.
            target_size (tuple): Target size for each 2D slice.
            augment_prob (float): Probability of applying augmentation.
            rotation_range (float): Maximum rotation angle in degrees.
            contrast_range (tuple): Range for contrast adjustment.
            brightness_range (tuple): Range for brightness adjustment.
            noise_level (float): Maximum level of Gaussian noise to add.
            flip_prob (float): Probability of flipping.
            zoom_range (tuple): Range for random zoom.
        """
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.train = train
        self.max_slices = max_slices
        self.target_size = target_size
        
        # Augmentation parameters
        self.augment_prob = augment_prob
        self.rotation_range = rotation_range
        self.contrast_range = contrast_range
        self.brightness_range = brightness_range
        self.noise_level = noise_level
        self.flip_prob = flip_prob
        self.zoom_range = zoom_range
        
        # Process tomogram metadata
        self.process_metadata()
        
        # Cache file paths and shapes
        self.cache_file_paths()
    
    def process_metadata(self):
        # Get unique tomograms
        tomo_ids = self.labels_df['tomo_id'].unique()
        self.tomo_df = pd.DataFrame({'tomo_id': tomo_ids})
        
        # For each tomogram, get its properties
        for tomo_id in tomo_ids:
            tomo_rows = self.labels_df[self.labels_df['tomo_id'] == tomo_id]
            
            # Get motor count
            num_motors = tomo_rows['Number of motors'].iloc[0]
            self.tomo_df.loc[self.tomo_df['tomo_id'] == tomo_id, 'Number of motors'] = num_motors
            
            # Get array shape and voxel spacing
            self.tomo_df.loc[self.tomo_df['tomo_id'] == tomo_id, 'Array shape (axis 0)'] = tomo_rows['Array shape (axis 0)'].iloc[0]
            self.tomo_df.loc[self.tomo_df['tomo_id'] == tomo_id, 'Array shape (axis 1)'] = tomo_rows['Array shape (axis 1)'].iloc[0]
            self.tomo_df.loc[self.tomo_df['tomo_id'] == tomo_id, 'Array shape (axis 2)'] = tomo_rows['Array shape (axis 2)'].iloc[0]
            self.tomo_df.loc[self.tomo_df['tomo_id'] == tomo_id, 'Voxel spacing'] = tomo_rows['Voxel spacing'].iloc[0]
            
            # Get motor axes (use first motor for training)
            if num_motors > 0:
                motor_row = tomo_rows[tomo_rows['Motor axis 0'] != -1].iloc[0]
                self.tomo_df.loc[self.tomo_df['tomo_id'] == tomo_id, 'Motor axis 0'] = motor_row['Motor axis 0']
                self.tomo_df.loc[self.tomo_df['tomo_id'] == tomo_id, 'Motor axis 1'] = motor_row['Motor axis 1']
                self.tomo_df.loc[self.tomo_df['tomo_id'] == tomo_id, 'Motor axis 2'] = motor_row['Motor axis 2']
            else:
                # No motor
                self.tomo_df.loc[self.tomo_df['tomo_id'] == tomo_id, 'Motor axis 0'] = -1
                self.tomo_df.loc[self.tomo_df['tomo_id'] == tomo_id, 'Motor axis 1'] = -1
                self.tomo_df.loc[self.tomo_df['tomo_id'] == tomo_id, 'Motor axis 2'] = -1
    
    def cache_file_paths(self):
        self.slice_files = {}
        
        for idx, row in self.tomo_df.iterrows():
            tomo_id = row['tomo_id']
            tomo_dir = os.path.join(self.root_dir, tomo_id)
            
            # Get all slice files and sort them
            files = sorted(glob.glob(os.path.join(tomo_dir, '*.jpg')))
            self.slice_files[tomo_id] = files
    
    def __len__(self):
        return len(self.tomo_df)
    
    def apply_augmentation(self, volume):
        """Apply various 3D augmentations to the volume."""
        # Only apply augmentation during training with probability
        if not self.train or random.random() > self.augment_prob:
            return volume
            
        # Random rotation (3D)
        if random.random() < 0.7:  # 70% chance of rotation
            angle_x = random.uniform(-self.rotation_range, self.rotation_range)
            angle_y = random.uniform(-self.rotation_range, self.rotation_range)
            angle_z = random.uniform(-self.rotation_range, self.rotation_range)
            
            # Apply rotation around each axis
            if angle_x != 0:
                volume = rotate(volume, angle_x, axes=(1, 2), reshape=False, order=1, mode='constant', cval=0)
            if angle_y != 0:
                volume = rotate(volume, angle_y, axes=(0, 2), reshape=False, order=1, mode='constant', cval=0)
            if angle_z != 0:
                volume = rotate(volume, angle_z, axes=(0, 1), reshape=False, order=1, mode='constant', cval=0)
        
        # Random zoom
        if random.random() < 0.5:  # 50% chance of zoom
            zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
            if zoom_factor != 1:
                # Calculate padding or cropping needed
                orig_shape = volume.shape
                zoomed = zoom(volume, (1, zoom_factor, zoom_factor), order=1, mode='constant', cval=0)
                
                # If zoomed in (zoom_factor > 1), need to crop
                if zoom_factor > 1:
                    z, y, x = zoomed.shape
                    start_y = (y - orig_shape[1]) // 2
                    start_x = (x - orig_shape[2]) // 2
                    volume = zoomed[:, start_y:start_y+orig_shape[1], start_x:start_x+orig_shape[2]]
                # If zoomed out (zoom_factor < 1), need to pad
                else:
                    z, y, x = zoomed.shape
                    pad_y = (orig_shape[1] - y) // 2
                    pad_x = (orig_shape[2] - x) // 2
                    volume = np.pad(zoomed, ((0, 0), (pad_y, orig_shape[1]-y-pad_y), (pad_x, orig_shape[2]-x-pad_x)), 
                                   mode='constant')
        
        # Random flips - using copy to ensure contiguous array
        if random.random() < self.flip_prob:
            volume = np.flip(volume, axis=1).copy()  # Flip horizontally
            
        if random.random() < self.flip_prob:
            volume = np.flip(volume, axis=2).copy()  # Flip vertically
            
        # Random contrast
        if random.random() < 0.6:  # 60% chance of contrast adjustment
            contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
            mean = volume.mean()
            volume = (volume - mean) * contrast_factor + mean
            volume = np.clip(volume, 0, 1)
        
        # Random brightness
        if random.random() < 0.6:  # 60% chance of brightness adjustment
            brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
            volume = volume + brightness_factor
            volume = np.clip(volume, 0, 1)
        
        # Add Gaussian noise
        if random.random() < 0.4:  # 40% chance of adding noise
            noise = np.random.normal(0, self.noise_level, volume.shape)
            volume = volume + noise
            volume = np.clip(volume, 0, 1)
        
        # Random dropout (simulating missing data)
        if random.random() < 0.3:  # 30% chance of dropout
            mask = np.random.rand(*volume.shape) > 0.05  # Dropout 5% of voxels
            volume = volume * mask
            
        # Random intensity shifts for specific regions (simulating artifacts)
        if random.random() < 0.2:  # 20% chance of intensity artifacts
            num_regions = random.randint(1, 3)
            for _ in range(num_regions):
                z_size = random.randint(1, max(2, volume.shape[0] // 10))
                y_size = random.randint(5, max(10, volume.shape[1] // 5))
                x_size = random.randint(5, max(10, volume.shape[2] // 5))
                
                z_start = random.randint(0, volume.shape[0] - z_size)
                y_start = random.randint(0, volume.shape[1] - y_size)
                x_start = random.randint(0, volume.shape[2] - x_size)
                
                intensity_shift = random.uniform(-0.2, 0.2)
                
                region = volume[z_start:z_start+z_size, y_start:y_start+y_size, x_start:x_start+x_size]
                region = region + intensity_shift
                volume[z_start:z_start+z_size, y_start:y_start+y_size, x_start:x_start+x_size] = np.clip(region, 0, 1)
        
        # Ensure the volume is C-contiguous before returning
        return np.ascontiguousarray(volume)
    
    def load_volume(self, tomo_id):
        files = self.slice_files[tomo_id]
        
        # Determine which slices to load
        if self.max_slices is not None and len(files) > self.max_slices:
            # Subsample evenly
            indices = np.linspace(0, len(files)-1, self.max_slices, dtype=int)
            files_to_load = [files[i] for i in indices]
        else:
            files_to_load = files
        
        # Load slices
        slices = []
        for file_path in files_to_load:
            img = Image.open(file_path).convert('L')  # Convert to grayscale
            img = img.resize(self.target_size, Image.BILINEAR)
            slices.append(np.array(img))
        
        # Stack slices to form volume
        volume = np.stack(slices)
        
        # Pad if needed
        if self.max_slices is not None and volume.shape[0] < self.max_slices:
            pad_width = self.max_slices - volume.shape[0]
            pad_before = pad_width // 2
            pad_after = pad_width - pad_before
            volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
        
        # Normalize to [0, 1]
        volume = volume.astype(np.float32) / 255.0
        
        return volume
    
    def __getitem__(self, idx):
        row = self.tomo_df.iloc[idx]
        tomo_id = row['tomo_id']
        
        # Load volume
        volume = self.load_volume(tomo_id)
        
        # Get labels
        motor_axes = row[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']].values.astype(np.float32)
        has_motor = not (motor_axes == -1).all()
        
        # Get original shape for coordinate normalization
        array_shape = np.array([
            row['Array shape (axis 0)'],
            row['Array shape (axis 1)'],
            row['Array shape (axis 2)']
        ], dtype=np.float32)
        
        # Process coordinates
        if not has_motor:
            motor_axes = np.zeros(3, dtype=np.float32)
        else:
            # Apply coordinate jitter in training
            if self.train and random.random() < 0.5:
                # Add small random jitter to coordinates (within 5% of dimension)
                jitter_z = np.random.uniform(-0.05, 0.05) * array_shape[0]
                jitter_x = np.random.uniform(-0.05, 0.05) * array_shape[1]
                jitter_y = np.random.uniform(-0.05, 0.05) * array_shape[2]
                
                motor_axes[0] += jitter_z
                motor_axes[1] += jitter_x
                motor_axes[2] += jitter_y
                
                # Ensure coordinates are still within bounds
                motor_axes[0] = max(0, min(motor_axes[0], array_shape[0] - 1))
                motor_axes[1] = max(0, min(motor_axes[1], array_shape[1] - 1))
                motor_axes[2] = max(0, min(motor_axes[2], array_shape[2] - 1))
            
            # Normalize coordinates to [0, 1]
            motor_axes[0] = motor_axes[0] / array_shape[0]
            motor_axes[1] = motor_axes[1] / array_shape[1]
            motor_axes[2] = motor_axes[2] / array_shape[2]
        
        # Apply augmentation to volume
        volume = self.apply_augmentation(volume)
        
        # Ensure the volume is C-contiguous
        if not volume.flags.c_contiguous:
            volume = np.ascontiguousarray(volume)
        
        # Convert to tensor
        volume = torch.from_numpy(volume).unsqueeze(0).float()  # Add channel dimension
        motor_axes = torch.from_numpy(motor_axes)
        has_motor = torch.tensor([float(has_motor)])
        
        return {
            'tomo_id': tomo_id,
            'volume': volume,
            'has_motor': has_motor,
            'motor_axes': motor_axes,
            'original_shape': torch.tensor([
                row['Array shape (axis 0)'], 
                row['Array shape (axis 1)'], 
                row['Array shape (axis 2)']
            ], dtype=torch.float32),
            'voxel_spacing': torch.tensor([row['Voxel spacing']], dtype=torch.float32)
        }




class TestTomogramDataset(Dataset):
    """Dataset for loading test tomograms with no labels."""
    def __init__(self, root_dir, max_slices=64, target_size=(128, 128)):
        self.root_dir = root_dir
        self.max_slices = max_slices
        self.target_size = target_size
        
        # Get all tomogram directories
        self.tomo_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        
        # Cache file paths
        self.slice_files = {}
        for tomo_id in self.tomo_dirs:
            tomo_dir = os.path.join(self.root_dir, tomo_id)
            files = sorted(glob.glob(os.path.join(tomo_dir, '*.jpg')))
            self.slice_files[tomo_id] = files
    
    def __len__(self):
        return len(self.tomo_dirs)
    
    def load_volume(self, tomo_id):
        files = self.slice_files[tomo_id]
        
        # Get array shape
        z_shape = len(files)
        if z_shape > 0:
            img = Image.open(files[0])
            x_shape, y_shape = img.size
        else:
            raise ValueError(f"No slices found for tomogram {tomo_id}")
        
        # Store original shape
        original_shape = np.array([z_shape, x_shape, y_shape])
        
        # Determine which slices to load
        if self.max_slices is not None and z_shape > self.max_slices:
            # Subsample evenly
            indices = np.linspace(0, z_shape-1, self.max_slices, dtype=int)
            files_to_load = [files[i] for i in indices]
        else:
            files_to_load = files
        
        # Load slices
        slices = []
        for file_path in files_to_load:
            img = Image.open(file_path).convert('L')  # Convert to grayscale
            img = img.resize(self.target_size, Image.BILINEAR)
            slices.append(np.array(img))
        
        # Stack slices to form volume
        volume = np.stack(slices)
        
        # Pad if needed
        if self.max_slices is not None and volume.shape[0] < self.max_slices:
            pad_width = self.max_slices - volume.shape[0]
            pad_before = pad_width // 2
            pad_after = pad_width - pad_before
            volume = np.pad(volume, ((pad_before, pad_after), (0, 0), (0, 0)), mode='constant')
        
        # Normalize to [0, 1]
        volume = volume.astype(np.float32) / 255.0
        
        return volume, original_shape
    
    def __getitem__(self, idx):
        tomo_id = self.tomo_dirs[idx]
        
        # Load volume
        volume, original_shape = self.load_volume(tomo_id)
        
        # Convert to tensor
        volume = torch.from_numpy(volume).unsqueeze(0)  # Add channel dimension
        
        return {
            'tomo_id': tomo_id,
            'volume': volume,
            'original_shape': torch.tensor(original_shape, dtype=torch.float32)
        }



class SEBlock3D(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.spatial_conv = nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        x = x * y
        y = torch.cat([torch.max(x, 1, keepdim=True)[0], torch.mean(x, 1, keepdim=True)], dim=1)
        y = self.spatial_conv(y)
        y = self.spatial_sigmoid(y)
        return x * y

class EnhancedAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(embed_dim)  # Layer norm before attention
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),  # Expansion
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),  # Reduction
            nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(embed_dim)  # Layer norm before FFN
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Input x: (batch, seq_len, embed_dim)
        identity = x
        x = self.ln1(x)
        attn_output, _ = self.mha(x, x, x)
        x = identity + self.dropout(attn_output)  # Residual connection
        identity = x
        x = self.ln2(x)
        ffn_output = self.ffn(x)
        x = identity + self.dropout(ffn_output)  # Residual connection
        return x

class FlagellarMotorNet(nn.Module):
    def __init__(self, input_channels=1, base_filters=16, num_heads=4, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, base_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(base_filters)
        self.se1 = SEBlock3D(base_filters)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        
        self.conv2 = nn.Conv3d(base_filters, base_filters*2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(base_filters*2)
        self.se2 = SEBlock3D(base_filters*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        self.res_conv2 = nn.Conv3d(base_filters, base_filters*2, kernel_size=1, stride=1)
        
        self.conv3 = nn.Conv3d(base_filters*2, base_filters*4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(base_filters*4)
        self.se3 = SEBlock3D(base_filters*4)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        self.res_conv3 = nn.Conv3d(base_filters*2, base_filters*4, kernel_size=1, stride=1)
        
        self.conv4 = nn.Conv3d(base_filters*4, base_filters*8, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(base_filters*8)
        self.se4 = SEBlock3D(base_filters*8)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        self.res_conv4 = nn.Conv3d(base_filters*4, base_filters*8, kernel_size=1, stride=1)
        
        self.attn_channels = base_filters * 8
        self.num_heads = num_heads
        assert self.attn_channels % num_heads == 0, "attn_channels must be divisible by num_heads"
        self.attn_block = EnhancedAttentionBlock(embed_dim=self.attn_channels, num_heads=num_heads, dropout=dropout)
        
        self.spatial_size = 4 * 8 * 8  # Assuming input size reduces to 4x8x8 after pooling
        self.fc_size = self.attn_channels * self.spatial_size
        
        self.fc_presence = nn.Sequential(
            nn.Linear(self.fc_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.fc_location = nn.Sequential(
            nn.Linear(self.fc_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.se1(x)
        x = self.pool1(x)
        
        identity = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se2(x)
        x = self.pool2(x)
        identity = self.pool2(self.res_conv2(identity))
        x = x + identity
        
        identity = x
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.se3(x)
        x = self.pool3(x)
        identity = self.pool3(self.res_conv3(identity))
        x = x + identity
        
        identity = x
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.se4(x)
        x = self.pool4(x)
        identity = self.pool4(self.res_conv4(identity))
        x = x + identity
        
        b, c, d, h, w = x.size()
        x = x.view(b, c, -1).permute(0, 2, 1)  # (batch, seq_len, embed_dim)
        x = self.attn_block(x)
        x = x.permute(0, 2, 1).view(b, c, d, h, w)
        
        x = x.reshape(b, -1)
        presence = self.fc_presence(x)
        location = self.fc_location(x)
        return presence, location
        

class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, bottleneck_factor=4):
        super().__init__()
        bottleneck_channels = out_channels // bottleneck_factor
        self.conv1 = nn.Conv3d(in_channels, bottleneck_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(bottleneck_channels)
        self.conv2 = nn.Conv3d(bottleneck_channels, bottleneck_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(bottleneck_channels)
        self.conv3 = nn.Conv3d(bottleneck_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(out_channels)
        self.ln = nn.LayerNorm(out_channels)  # Normalize across channels
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        shortcut = self.shortcut(identity)
        
        # Reshape for LayerNorm: move channels to the last dimension
        out = out + shortcut  # Shape: [batch, channels, depth, height, width]
        b, c, d, h, w = out.size()
        out = out.permute(0, 2, 3, 4, 1).contiguous()  # Shape: [batch, depth, height, width, channels]
        out = self.ln(out)  # Apply LayerNorm across channels
        out = out.permute(0, 4, 1, 2, 3).contiguous()  # Shape: [batch, channels, depth, height, width]
        
        out = F.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, input_channels=1, base_filters=16):
        super().__init__()
        self.conv1 = nn.Conv3d(input_channels, base_filters, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(base_filters)
        self.pool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(base_filters, base_filters, 2, stride=1)
        self.layer2 = self._make_layer(base_filters, base_filters*2, 2, stride=2)
        self.layer3 = self._make_layer(base_filters*2, base_filters*4, 2, stride=2)
        self.layer4 = self._make_layer(base_filters*4, base_filters*8, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc_presence = nn.Sequential(
            nn.Linear(base_filters*8, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.fc_location = nn.Sequential(
            nn.Linear(base_filters*8, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = [ResBlock3D(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock3D(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        presence = self.fc_presence(x)
        location = self.fc_location(x)
        return presence, location
# EnsembleNet
class EnsembleNet(nn.Module):
    def __init__(self, input_channels=1, base_filters=16, num_heads=4):
        super().__init__()
        self.model1 = FlagellarMotorNet(input_channels=input_channels, base_filters=base_filters, num_heads=num_heads)
        self.model2 = ResNet3D(input_channels=input_channels, base_filters=base_filters)
    
    def forward(self, x):
        p1, l1 = self.model1(x)
        p2, l2 = self.model2(x)
        presence = (p1 + p2) / 2
        location = (l1 + l2) / 2
        return presence, location



# Custom loss function
class FlagellarMotorLoss(nn.Module):
    """
    Custom loss function for flagellar motor detection and localization.
    Combines binary cross-entropy for motor presence with MSE for location.
    """
    def __init__(self, presence_weight=1.0, location_weight=3.0):
        super(FlagellarMotorLoss, self).__init__()
        self.presence_weight = presence_weight
        self.location_weight = location_weight
        self.bce_loss = nn.BCELoss()
        self.mse_loss = nn.MSELoss()
    
    def forward(self, presence_pred, location_pred, presence_true, location_true):
        # Presence loss (binary cross-entropy)
        presence_loss = self.bce_loss(presence_pred, presence_true)
        
        # Location loss (only computed for tomograms with motors)
        if torch.sum(presence_true) > 0:
            # Select only the samples with motors
            has_motor = presence_true.squeeze() > 0.5
            if has_motor.sum() > 0:
                location_pred_with_motor = location_pred[has_motor]
                location_true_with_motor = location_true[has_motor]
                
                location_loss = self.mse_loss(location_pred_with_motor, location_true_with_motor)
                
                # Calculate Euclidean distance (for monitoring)
                euclidean_dist = torch.sqrt(torch.sum((location_pred_with_motor - location_true_with_motor) ** 2, dim=1))
                avg_euclidean_dist = euclidean_dist.mean()
            else:
                location_loss = torch.tensor(0.0, device=presence_loss.device)
                avg_euclidean_dist = torch.tensor(0.0, device=presence_loss.device)
        else:
            location_loss = torch.tensor(0.0, device=presence_loss.device)
            avg_euclidean_dist = torch.tensor(0.0, device=presence_loss.device)
        
        # Total loss
        total_loss = self.presence_weight * presence_loss + self.location_weight * location_loss
        
        return total_loss, presence_loss, location_loss, avg_euclidean_dist





def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    epoch_loss = 0
    epoch_presence_loss = 0
    epoch_location_loss = 0
    epoch_euclidean_dist = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    
    for batch in progress_bar:
        # Move data to device
        volume = batch['volume'].to(device)
        has_motor = batch['has_motor'].to(device)
        motor_axes = batch['motor_axes'].to(device)
        
        # Forward pass
        presence_pred, location_pred = model(volume)
        
        # Calculate loss
        loss, presence_loss, location_loss, euclidean_dist = criterion(
            presence_pred, location_pred, has_motor, motor_axes
        )
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        epoch_loss += loss.item()
        epoch_presence_loss += presence_loss.item()
        epoch_location_loss += location_loss.item()
        epoch_euclidean_dist += euclidean_dist.item()
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': loss.item(),
            'p_loss': presence_loss.item(),
            'l_loss': location_loss.item(),
            'eucl_dist': euclidean_dist.item()
        })
    
    # Calculate average metrics
    num_batches = len(dataloader)
    avg_loss = epoch_loss / num_batches
    avg_presence_loss = epoch_presence_loss / num_batches
    avg_location_loss = epoch_location_loss / num_batches
    avg_euclidean_dist = epoch_euclidean_dist / num_batches
    
    return {
        'loss': avg_loss,
        'presence_loss': avg_presence_loss,
        'location_loss': avg_location_loss,
        'euclidean_dist': avg_euclidean_dist
    }



# Validation Function

def validate(model, dataloader, criterion, device, threshold=0.5):
    """Validate the model"""
    model.eval()
    epoch_loss = 0
    epoch_presence_loss = 0
    epoch_location_loss = 0
    epoch_euclidean_dist = 0
    
    # Track predictions for F-beta score
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    progress_bar = tqdm(dataloader, desc="Validation")
    
    with torch.no_grad():
        for batch in progress_bar:
            # Move data to device
            volume = batch['volume'].to(device)
            has_motor = batch['has_motor'].to(device)
            motor_axes = batch['motor_axes'].to(device)
            original_shape = batch['original_shape'].to(device)
            voxel_spacing = batch['voxel_spacing'].to(device)
            
            # Forward pass
            presence_pred, location_pred = model(volume)
            
            # Calculate loss
            loss, presence_loss, location_loss, euclidean_dist = criterion(
                presence_pred, location_pred, has_motor, motor_axes
            )
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_presence_loss += presence_loss.item()
            epoch_location_loss += location_loss.item()
            epoch_euclidean_dist += euclidean_dist.item()
            
            # Calculate F-beta metrics
            for i in range(len(presence_pred)):
                # Check if model predicts a motor
                pred_has_motor = presence_pred[i].item() > threshold
                true_has_motor = has_motor[i].item() > 0.5
                
                if pred_has_motor and true_has_motor:
                    # Convert normalized coordinates back to original space
                    pred_coords = location_pred[i].cpu().numpy()
                    true_coords = motor_axes[i].cpu().numpy()
                    shape = original_shape[i].cpu().numpy()
                    spacing = voxel_spacing[i].item()
                    
                    # Denormalize coordinates
                    pred_coords_orig = np.array([
                        pred_coords[0] * shape[0],
                        pred_coords[1] * shape[1],
                        pred_coords[2] * shape[2]
                    ])
                    
                    true_coords_orig = np.array([
                        true_coords[0] * shape[0],
                        true_coords[1] * shape[1],
                        true_coords[2] * shape[2]
                    ])
                    
                    # Calculate Euclidean distance in Angstroms
                    dist = np.sqrt(np.sum((pred_coords_orig - true_coords_orig) ** 2)) * spacing
                    
                    # Check if prediction is within threshold (1000 Angstroms)
                    if dist <= 1000:
                        true_positives += 1
                    else:
                        false_positives += 1
                        false_negatives += 1
                elif pred_has_motor and not true_has_motor:
                    false_positives += 1
                elif not pred_has_motor and true_has_motor:
                    false_negatives += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'p_loss': presence_loss.item(),
                'l_loss': location_loss.item(),
                'eucl_dist': euclidean_dist.item()
            })
    
    # Calculate average metrics
    num_batches = len(dataloader)
    avg_loss = epoch_loss / num_batches
    avg_presence_loss = epoch_presence_loss / num_batches
    avg_location_loss = epoch_location_loss / num_batches
    avg_euclidean_dist = epoch_euclidean_dist / num_batches
    
    # Calculate F-beta score (beta=2)
    beta = 2
    if true_positives + false_positives > 0:
        precision = true_positives / (true_positives + false_positives)
    else:
        precision = 0
    
    if true_positives + false_negatives > 0:
        recall = true_positives / (true_positives + false_negatives)
    else:
        recall = 0
    
    if precision + recall > 0:
        f_beta = (1 + beta**2) * precision * recall / ((beta**2 * precision) + recall)
    else:
        f_beta = 0
    
    return {
        'loss': avg_loss,
        'presence_loss': avg_presence_loss,
        'location_loss': avg_location_loss,
        'euclidean_dist': avg_euclidean_dist,
        'f_beta': f_beta,
        'precision': precision,
        'recall': recall,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }




## Prediction function
def predict(model, dataloader, device, threshold=0.5):
    """Generate predictions for test set"""
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            # Move data to device
            volume = batch['volume'].to(device)
            tomo_ids = batch['tomo_id']
            original_shape = batch['original_shape'].to(device)
            
            # Forward pass
            presence_pred, location_pred = model(volume)
            
            # Process predictions
            for i in range(len(presence_pred)):
                tomo_id = tomo_ids[i]
                pred_has_motor = presence_pred[i].item() > threshold
                
                if pred_has_motor:
                    # Convert normalized coordinates back to original space
                    pred_coords = location_pred[i].cpu().numpy()
                    shape = original_shape[i].cpu().numpy()
                    
                    # Denormalize coordinates
                    pred_coords_orig = np.array([
                        pred_coords[0] * shape[0],
                        pred_coords[1] * shape[1],
                        pred_coords[2] * shape[2]
                    ])
                    
                    predictions.append({
                        'tomo_id': tomo_id,
                        'Motor axis 0': pred_coords_orig[0],
                        'Motor axis 1': pred_coords_orig[1],
                        'Motor axis 2': pred_coords_orig[2]
                    })
                else:
                    predictions.append({
                        'tomo_id': tomo_id,
                        'Motor axis 0': -1,
                        'Motor axis 1': -1,
                        'Motor axis 2': -1
                    })
    
    return pd.DataFrame(predictions)



def train_model():
    """Train the model and save checkpoints"""
    # Configuration
    config = {
        'batch_size': 16,
        'num_workers': 2,
        'max_slices': 64,
        'target_size': (128, 128),
        'learning_rate': 0.0005,
        'weight_decay': 0.0001,
        'epochs': 20, 
        'presence_weight': 1.0,
        'location_weight': 3.0,
        'threshold': 0.5,
        'validation_split': 0.2
    }
    
    # Load and preprocess data
    train_df = pd.read_csv(TRAIN_CSV)
    
    # Get unique tomograms
    tomo_ids = train_df['tomo_id'].unique()
    
    # Split tomograms into train and validation sets
    train_tomo_ids, val_tomo_ids = train_test_split(
        tomo_ids, 
        test_size=config['validation_split'], 
        random_state=RANDOM_SEED,
        stratify=train_df.drop_duplicates('tomo_id')['Number of motors'] > 0  # Stratify by motor presence
    )
    
    # Filter train_df to get only the relevant tomograms
    train_set_df = train_df[train_df['tomo_id'].isin(train_tomo_ids)]
    val_set_df = train_df[train_df['tomo_id'].isin(val_tomo_ids)]
    
    # Create temporary CSVs for the datasets
    train_csv = os.path.join(OUTPUT_DIR, 'train_set.csv')
    val_csv = os.path.join(OUTPUT_DIR, 'val_set.csv')
    
    train_set_df.to_csv(train_csv, index=False)
    val_set_df.to_csv(val_csv, index=False)
    
    # Create datasets
    train_dataset = TomogramDataset(
        csv_file=train_csv,
        root_dir=TRAIN_DIR,
        train=True,
        max_slices=config['max_slices'],
        target_size=config['target_size']
    )
    
    val_dataset = TomogramDataset(
        csv_file=val_csv,
        root_dir=TRAIN_DIR,
        train=False,
        max_slices=config['max_slices'],
        target_size=config['target_size']
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    # Print dataset sizes
    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Initialize model
    model = EnsembleNet(
        input_channels=1,
        base_filters=16
    ).to(DEVICE)
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Initialize scheduler
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=5,
        verbose=True
    )
    
    # Initialize loss function
    criterion = FlagellarMotorLoss(
        presence_weight=config['presence_weight'],
        location_weight=config['location_weight']
    )
    
    # Initialize best metrics
    best_val_loss = float('inf')
    best_f_beta = 0
    
    # Training loop
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']}")
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, DEVICE, threshold=config['threshold'])
        
        # Update scheduler
        scheduler.step(val_metrics['loss'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.4f}, Val Loss: {val_metrics['loss']:.4f}")
        print(f"Val F-beta (β=2): {val_metrics['f_beta']:.4f}, Precision: {val_metrics['precision']:.4f}, Recall: {val_metrics['recall']:.4f}")
        print(f"Val Euclidean Dist: {val_metrics['euclidean_dist']:.4f}")
        
        # Save best model (by loss)
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(MODEL_DIR, 'best_model_loss.pth'))
            
            print(f"Saved best model by loss: {best_val_loss:.4f}")
        
        # Save best model (by F-beta)
        if val_metrics['f_beta'] > best_f_beta:
            best_f_beta = val_metrics['f_beta']
            
            # Save model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_metrics': val_metrics,
                'config': config
            }, os.path.join(MODEL_DIR, 'best_model_fbeta.pth'))
            
            print(f"Saved best model by F-beta: {best_f_beta:.4f}")
    
    # Clean up temporary files
    os.remove(train_csv)
    os.remove(val_csv)
    
    print("\nTraining completed!")
    return model



def generate_predictions(model_path=None):
    """Generate predictions for test set"""
    # Configuration
    config = {
        'batch_size': 4,
        'num_workers': 2,
        'max_slices': 64,
        'target_size': (128, 128),
        'threshold': 0.5
    }
    
    # Use specified model path or default
    if model_path is None:
        model_path = os.path.join(MODEL_DIR, 'best_model_fbeta.pth')
    
    # Create test dataset
    test_dataset = TestTomogramDataset(
        root_dir=TEST_DIR,
        max_slices=config['max_slices'],
        target_size=config['target_size']
    )
    
    # Create test dataloader
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Load model or create a new one if not found
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # Initialize model
        model = EnsembleNet(
            input_channels=1,
            base_filters=16
        ).to(DEVICE)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"Loaded model from {model_path}")
        print(f"Model was trained for {checkpoint['epoch']+1} epochs")
        print(f"Validation metrics at checkpoint: F-beta = {checkpoint['val_metrics']['f_beta']:.4f}")
    else:
        print(f"Model not found at {model_path}, creating new model")
        model = EnsembleNet(
            input_channels=1,
            base_filters=16
        ).to(DEVICE)
    
    # Generate predictions
    predictions_df = predict(model, test_loader, DEVICE, threshold=config['threshold'])
    
    # Save predictions
    output_file = os.path.join(OUTPUT_DIR, 'submission.csv')
    predictions_df.to_csv(output_file, index=False)
    
    # Print statistics
    motor_count = (predictions_df[['Motor axis 0', 'Motor axis 1', 'Motor axis 2']] != -1).all(axis=1).sum()
    print(f"Created submission file with {len(predictions_df)} predictions")
    print(f"Number of motors predicted: {motor_count}")
    print(f"Percentage of motors predicted: {motor_count / len(predictions_df) * 100:.2f}%")
    
    return predictions_df



# Train the model
model = train_model()


# Generate predictions 
predictions_df = generate_predictions()


def visualize_predictions(predictions_df, sample_count=3):
    """Visualize a few sample predictions"""
    # Select samples with and without motors
    motors_present = predictions_df[predictions_df['Motor axis 0'] != -1].sample(min(sample_count, len(predictions_df[predictions_df['Motor axis 0'] != -1])))
    motors_absent = predictions_df[predictions_df['Motor axis 0'] == -1].sample(min(sample_count, len(predictions_df[predictions_df['Motor axis 0'] == -1])))
    
    # Combine the samples
    samples = pd.concat([motors_present, motors_absent])
    
    # Display predictions
    print("Sample predictions:")
    for _, row in samples.iterrows():
        tomo_id = row['tomo_id']
        if row['Motor axis 0'] == -1:
            print(f"Tomogram {tomo_id}: No motor detected")
        else:
            coords = (row['Motor axis 0'], row['Motor axis 1'], row['Motor axis 2'])
            print(f"Tomogram {tomo_id}: Motor detected at coordinates {coords}")

    # You could add code here to visualize specific tomogram slices with overlaid predictions
    # This would require loading the tomograms and plotting slices near the predicted motor location











