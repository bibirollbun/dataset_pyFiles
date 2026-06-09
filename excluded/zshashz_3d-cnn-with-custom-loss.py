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



class TomogramDataset(Dataset):
    """
    Dataset for loading 3D tomograms from stacks of 2D JPG slices.
    """
    def __init__(self, csv_file, root_dir, train=True, max_slices=64, target_size=(128, 128)):
        self.labels_df = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.train = train
        self.max_slices = max_slices
        self.target_size = target_size
        
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
        
        # Process coordinates
        if not has_motor:
            motor_axes = np.zeros(3, dtype=np.float32)
        else:
            # Get original shape
            array_shape = np.array([
                row['Array shape (axis 0)'],
                row['Array shape (axis 1)'],
                row['Array shape (axis 2)']
            ], dtype=np.float32)
            
            # Apply data augmentation in training (random jitter to coordinates)
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
        
        # Convert to tensor
        volume = torch.from_numpy(volume).unsqueeze(0)  # Add channel dimension
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





class FlagellarMotorNet(nn.Module):
    """
    3D CNN model for detecting and localizing flagellar motors in tomograms.
    """
    def __init__(self, input_channels=1, base_filters=16):
        super(FlagellarMotorNet, self).__init__()
        
        # Feature extraction layers
        self.conv1 = nn.Conv3d(input_channels, base_filters, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm3d(base_filters)
        self.pool1 = nn.MaxPool3d(kernel_size=2)
        
        self.conv2 = nn.Conv3d(base_filters, base_filters*2, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm3d(base_filters*2)
        self.pool2 = nn.MaxPool3d(kernel_size=2)
        
        self.conv3 = nn.Conv3d(base_filters*2, base_filters*4, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm3d(base_filters*4)
        self.pool3 = nn.MaxPool3d(kernel_size=2)
        
        self.conv4 = nn.Conv3d(base_filters*4, base_filters*8, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm3d(base_filters*8)
        self.pool4 = nn.MaxPool3d(kernel_size=2)
        
        # Calculate the size of the flattened features
        # Assuming input size of [1, 64, 128, 128]
        # After 4 pooling layers (each dividing by 2): [base_filters*8, 4, 8, 8]
        self.fc_size = base_filters * 8 * 4 * 8 * 8
        
        # Classification head (motor presence)
        self.fc_presence = nn.Sequential(
            nn.Linear(self.fc_size, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Regression head (motor location)
        self.fc_location = nn.Sequential(
            nn.Linear(self.fc_size, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 3),
            nn.Sigmoid()  # Normalize coordinates to [0, 1]
        )
    
    def forward(self, x):
        # Feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Motor presence prediction
        presence = self.fc_presence(x)
        
        # Motor location prediction
        location = self.fc_location(x)
        
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
        'batch_size': 32,
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
    model = FlagellarMotorNet(
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
        model = FlagellarMotorNet(
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
        model = FlagellarMotorNet(
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











