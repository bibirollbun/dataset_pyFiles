import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split
import random
# import cv2 # Not used in the provided snippet, can be removed if not needed elsewhere
import csv # Not used in the provided snippet, can be removed if not needed elsewhere
import time
import gc # Garbage Collector for potentially large datasets

# Set random seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False # Set to False for determinism

seed_everything()

# Define data paths (adjust if necessary for your environment)
# Assuming the script runs where '/kaggle/input/...' is accessible
try:
    DATA_PATH = Path('/kaggle/input/waveform-inversion/train_samples')
    if not DATA_PATH.exists():
        raise FileNotFoundError # Fallback if Kaggle path doesn't exist
except FileNotFoundError:
    print("Kaggle path not found, using fallback path './data/train_samples'. Please adjust if needed.")
    DATA_PATH = Path('./data/train_samples') # Example fallback path
    # Create dummy data if path doesn't exist for local testing
    if not DATA_PATH.exists():
        print("Fallback data path not found. Creating dummy data for testing.")
        DATA_PATH.mkdir(parents=True, exist_ok=True)
        # Create a few dummy files
        for i in range(3):
            dummy_seis = np.random.rand(10, 5, 1000, 70).astype(np.float32) # N, C, T, X
            dummy_vel = np.random.rand(10, 1, 70, 70).astype(np.float32) # N, C, Y, X
            np.save(DATA_PATH / f'sample_{i}_seis.npy', dummy_seis)
            np.save(DATA_PATH / f'sample_{i}_vel.npy', dummy_vel)

# Find files to load and create Dataset
all_inputs = sorted([
    f
    for f in DATA_PATH.rglob('*.npy')
    if ('seis' in f.stem) or ('data' in f.stem)
])

if not all_inputs:
    raise FileNotFoundError(f"No input files found in {DATA_PATH}. Check the path and file naming ('seis' or 'data').")

def inputs_files_to_output_files(input_files):
    output_files = []
    for f in input_files:
        # Try replacing 'seis' with 'vel' or 'data' with 'model'
        out_f_vel = Path(str(f).replace('seis', 'vel'))
        out_f_model = Path(str(f).replace('data', 'model'))
        
        # Determine the correct output file based on which input name matched
        if 'seis' in f.stem:
            output_files.append(out_f_vel)
        elif 'data' in f.stem:
             output_files.append(out_f_model)
        else:
            # Fallback or error if naming convention isn't matched
            # For safety, let's assume 'vel' is the target if unsure
             print(f"Warning: Input file {f} doesn't contain 'seis' or 'data'. Assuming corresponding output ends in 'vel'.")
             output_files.append(out_f_vel) # Or handle error as appropriate

    return output_files

all_outputs = inputs_files_to_output_files(all_inputs)

# Verify all output files exist
missing_outputs = [f for f in all_outputs if not f.exists()]
if missing_outputs:
    print(f"Error: Missing output files corresponding to inputs:")
    for f in missing_outputs:
        print(f" - {f}")
    # Attempt to find corresponding input file for better debugging
    # This reverse mapping might be complex depending on exact naming rules
    raise FileNotFoundError("Cannot proceed without all corresponding output files.")

assert len(all_inputs) == len(all_outputs), "Mismatch between number of input and output files found."

# Stratified split (optional, simple split used here)
# If filenames contain patterns indicating different geological settings, stratify=filenames could be useful
train_inputs, valid_inputs = train_test_split(all_inputs, test_size=0.2, random_state=42)
train_outputs = inputs_files_to_output_files(train_inputs)
valid_outputs = inputs_files_to_output_files(valid_inputs)

# Calculate statistics for normalization
def calculate_stats(files, n_files_for_stats=20, n_samples_per_file=100):
    """Calculate mean and std for normalization more robustly"""
    samples = []
    # Use a random subset of files for calculation if there are many files
    files_to_use = random.sample(files, min(len(files), n_files_for_stats)) if len(files) > n_files_for_stats else files

    print(f"Calculating stats from {len(files_to_use)} files...")
    for f in tqdm(files_to_use, desc="Calculating stats"):
        try:
            # Use mmap_mode for memory efficiency
            data = np.load(f, mmap_mode='r')
            # Ensure we don't try to sample more than available
            num_available = data.shape[0]
            indices_to_sample = min(n_samples_per_file, num_available)
            if indices_to_sample > 0:
                indices = np.random.choice(num_available, indices_to_sample, replace=False)
                # Load only the selected samples into memory
                samples.append(data[indices].astype(np.float32)) # Ensure float32
            del data # Release mmap handle
            gc.collect() # Explicitly collect garbage
        except Exception as e:
            print(f"Warning: Could not process file {f} for stats calculation: {e}")
            continue # Skip problematic files

    if not samples:
        print("Warning: No samples collected for stats calculation. Using default values (0, 1).")
        return 0.0, 1.0

    samples = np.concatenate(samples, axis=0) # Use concatenate for list of arrays
    mean = np.mean(samples)
    std = np.std(samples)

    # Avoid division by zero or near-zero std
    if std < 1e-8:
        print(f"Warning: Calculated std is very small ({std}). Setting to 1.0 to avoid issues.")
        std = 1.0

    return mean, std

# Calculate stats using a potentially larger subset for better accuracy
input_mean, input_std = calculate_stats(train_inputs, n_files_for_stats=50, n_samples_per_file=200)
output_mean, output_std = calculate_stats(train_outputs, n_files_for_stats=50, n_samples_per_file=200)

print(f"Input stats: mean={input_mean:.4f}, std={input_std:.4f}")
print(f"Output stats: mean={output_mean:.4f}, std={output_std:.4f}")


# --- Dataset Class ---
class SeismicDataset(Dataset):
    def __init__(self, inputs_files, output_files, n_examples_per_file=500,
                 input_mean=0.0, input_std=1.0, output_mean=0.0, output_std=1.0,
                 augment=False, input_shape=(5, 1000, 70), output_shape=(1, 70, 70)):
        assert len(inputs_files) == len(output_files)
        self.inputs_files = inputs_files
        self.output_files = output_files
        self.n_examples_per_file = n_examples_per_file
        self.input_mean = input_mean
        self.input_std = input_std if input_std > 1e-8 else 1.0 # Safety check
        self.output_mean = output_mean
        self.output_std = output_std if output_std > 1e-8 else 1.0 # Safety check
        self.augment = augment
        self.input_shape = input_shape   # Expected shape C, T, X
        self.output_shape = output_shape # Expected shape C, Y, X

        # Pre-calculate total length and file index mapping
        self.file_lengths = []
        self.file_cumulative_lengths = [0]
        print("Checking dataset file lengths...")
        for f in tqdm(self.inputs_files, desc="Scanning files"):
            try:
                # Fast shape check without loading full data
                with open(f, 'rb') as file_handle:
                    version = np.lib.format.read_magic(file_handle)
                    shape, _, _ = np.lib.format._read_array_header(file_handle, version)
                length = shape[0] # Assuming first dimension is number of samples
                self.file_lengths.append(length)
                self.file_cumulative_lengths.append(self.file_cumulative_lengths[-1] + length)
            except Exception as e:
                print(f"Warning: Could not read header of {f}: {e}. Assuming 0 length.")
                self.file_lengths.append(0)
                # Adjust cumulative sum accordingly - this file contributes 0 length
                self.file_cumulative_lengths.append(self.file_cumulative_lengths[-1])


        self._total_samples = self.file_cumulative_lengths[-1]
        print(f"Total samples found across all files: {self._total_samples}")

        # Determine effective n_examples_per_file if user requested more than available
        # This part is tricky because the original code seemed to define __len__ differently
        # Let's stick to the original logic for now: len is files * n_examples_per_file
        # but ensure __getitem__ handles cases where sample_idx is out of bounds for a file.
        self._len = len(self.inputs_files) * self.n_examples_per_file

    def __len__(self):
        # This length definition means we might try to access indices beyond
        # the actual number of samples in a file if n_examples_per_file is large.
        # __getitem__ needs to handle this gracefully (e.g., by wrapping around or sampling randomly).
        return self._len

    def __getitem__(self, idx):
        # Calculate which file and which sample *within that file's quota*
        file_idx = idx // self.n_examples_per_file
        sample_offset_in_quota = idx % self.n_examples_per_file

        # Get the actual number of samples in the target file
        actual_samples_in_file = self.file_lengths[file_idx]

        if actual_samples_in_file == 0:
             # Handle files that couldn't be read or are empty
             print(f"Warning: File {self.inputs_files[file_idx]} has 0 samples. Returning zeros.")
             X_sample = np.zeros(self.input_shape, dtype=np.float32)
             y_sample = np.zeros(self.output_shape, dtype=np.float32)
             X_orig = X_sample.copy()
             y_orig = y_sample.copy()
             return torch.from_numpy(X_sample), torch.from_numpy(y_sample), torch.from_numpy(X_orig), torch.from_numpy(y_orig)


        # Map the sample_offset_in_quota to an actual index in the file
        # Use modulo to wrap around if n_examples_per_file > actual_samples_in_file
        actual_sample_idx = sample_offset_in_quota % actual_samples_in_file

        X = None
        y = None
        try:
            # Load the specific sample using mmap_mode for efficiency
            X = np.load(self.inputs_files[file_idx], mmap_mode='r')
            y = np.load(self.output_files[file_idx], mmap_mode='r')

            # Ensure indices are within bounds (should be by modulo logic, but double-check)
            if actual_sample_idx >= X.shape[0] or actual_sample_idx >= y.shape[0]:
                 print(f"Warning: Index calculation error. idx={idx}, file_idx={file_idx}, sample_offset={sample_offset_in_quota}, actual_idx={actual_sample_idx}, shapes X:{X.shape}, y:{y.shape}. Using index 0.")
                 actual_sample_idx = 0 # Fallback to index 0

            # Get the specific sample, ensure float32, make copies
            X_sample = X[actual_sample_idx].astype(np.float32).copy()
            y_sample = y[actual_sample_idx].astype(np.float32).copy()

            # Store original data for physics computations if needed later (PINNs might use original scale)
            X_orig = X_sample.copy()
            y_orig = y_sample.copy()

            # --- Data Validation and Reshaping ---
            # Ensure input has expected dimensions (C, T, X)
            if X_sample.shape != self.input_shape:
                 # Attempt common fixes like adding channel dim if missing
                 if len(X_sample.shape) == 2 and self.input_shape[0] == 1: # (T, X) -> (1, T, X)
                     X_sample = np.expand_dims(X_sample, axis=0)
                 elif len(X_sample.shape) == 3 and X_sample.shape[0] == self.input_shape[1] and X_sample.shape[1] == self.input_shape[2] and self.input_shape[0] == 1 : # (T,X,C?) -> (C,T,X) very unlikely
                     print(f"Warning: Unexpected input shape {X_sample.shape} for file {self.inputs_files[file_idx]}, sample {actual_sample_idx}. Attempting transpose. Expected {self.input_shape}")
                     # This case is ambiguous, need more info on data format
                     # X_sample = np.transpose(X_sample, (2, 0, 1)) # Example transpose C last -> C first
                 else:
                     # If shape mismatch persists, raise error or return zeros
                     print(f"Error: Input shape mismatch! Got {X_sample.shape}, expected {self.input_shape} for file {self.inputs_files[file_idx]}, sample {actual_sample_idx}. Returning zeros.")
                     X_sample = np.zeros(self.input_shape, dtype=np.float32)
                     X_orig = X_sample.copy()
                     # Don't necessarily zero y_sample, maybe it's okay
            
            # Ensure output has expected dimensions (C, Y, X)
            if y_sample.shape != self.output_shape:
                if len(y_sample.shape) == 2 and self.output_shape[0] == 1: # (Y, X) -> (1, Y, X)
                    y_sample = np.expand_dims(y_sample, axis=0)
                else:
                    print(f"Error: Output shape mismatch! Got {y_sample.shape}, expected {self.output_shape} for file {self.output_files[file_idx]}, sample {actual_sample_idx}. Returning zeros.")
                    y_sample = np.zeros(self.output_shape, dtype=np.float32)
                    y_orig = y_sample.copy()
                    
            # --- Normalization ---
            X_sample = (X_sample - self.input_mean) / self.input_std
            y_sample = (y_sample - self.output_mean) / self.output_std

            # --- Data Augmentation ---
            if self.augment:
                # Horizontal flip (axis=2 assuming shape C,Y,X or C,T,X)
                if random.random() < 0.5:
                    X_sample = np.flip(X_sample, axis=2).copy() # Use copy() after flip
                    y_sample = np.flip(y_sample, axis=2).copy()

                # Add small random noise
                if random.random() < 0.3:
                    noise_level = random.uniform(0.01, 0.05) * self.input_std # Scale noise relative to std
                    X_sample += np.random.normal(0, noise_level, X_sample.shape).astype(np.float32)

            # Return tensors
            return torch.from_numpy(X_sample), torch.from_numpy(y_sample), torch.from_numpy(X_orig), torch.from_numpy(y_orig)

        except Exception as e:
             print(f"Error loading or processing sample {idx} (file {file_idx}, sample {actual_sample_idx}): {e}")
             # Return zeros or handle error appropriately
             X_sample = np.zeros(self.input_shape, dtype=np.float32)
             y_sample = np.zeros(self.output_shape, dtype=np.float32)
             X_orig = X_sample.copy()
             y_orig = y_sample.copy()
             return torch.from_numpy(X_sample), torch.from_numpy(y_sample), torch.from_numpy(X_orig), torch.from_numpy(y_orig)
        finally:
            # Explicitly delete large objects and collect garbage
            del X, y
            # gc.collect() # Collecting garbage frequently can slow down dataloading


# --- Determine Input/Output Shapes from Data ---
# Load one sample to determine shapes dynamically
print("Determining data shapes...")
try:
    temp_ds = SeismicDataset(train_inputs, train_outputs, n_examples_per_file=1)
    temp_loader = DataLoader(temp_ds, batch_size=1)
    x_sample, y_sample, _, _ = next(iter(temp_loader))
    INPUT_SHAPE = tuple(x_sample.shape[1:]) # Shape after batch dim: (C, T, X)
    OUTPUT_SHAPE = tuple(y_sample.shape[1:]) # Shape after batch dim: (C, Y, X)
    print(f"Detected Input Shape (C, T, X): {INPUT_SHAPE}")
    print(f"Detected Output Shape (C, Y, X): {OUTPUT_SHAPE}")
    del temp_ds, temp_loader, x_sample, y_sample # Clean up
    gc.collect()
except Exception as e:
    print(f"Could not automatically determine shapes: {e}. Using defaults.")
    # Fallback to expected shapes if loading fails
    INPUT_SHAPE = (5, 1000, 70)  # C, T, X
    OUTPUT_SHAPE = (1, 70, 70)   # C, Y, X

# --- Instantiate Datasets and DataLoaders ---
print("Creating datasets...")
# Use detected shapes
dstrain = SeismicDataset(train_inputs, train_outputs, input_mean=input_mean, input_std=input_std,
                         output_mean=output_mean, output_std=output_std, augment=True,
                         input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, n_examples_per_file=500) # Adjust n_examples if needed
dsvalid = SeismicDataset(valid_inputs, valid_outputs, input_mean=input_mean, input_std=input_std,
                         output_mean=output_mean, output_std=output_std, augment=False,
                         input_shape=INPUT_SHAPE, output_shape=OUTPUT_SHAPE, n_examples_per_file=500)

# Adjust batch size based on GPU memory
BATCH_SIZE = 16 # Reduced from 32 as PINNs can be memory intensive
NUM_WORKERS = 2 # Reduced from 4 to avoid potential issues with mmap and multiprocessing

print("Creating dataloaders...")
dltrain = DataLoader(dstrain, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True, num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)
dlvalid = DataLoader(dsvalid, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, drop_last=False, num_workers=NUM_WORKERS, persistent_workers=NUM_WORKERS > 0)


# --- Differential Operators (Potentially for advanced PINN loss) ---
class DifferentialOperators:
    @staticmethod
    def gradient(f, dim, dx=1.0, order=1):
        """Compute gradient of f along a specified dimension using torch.gradient"""
        grads = f
        for _ in range(order):
            grads = torch.gradient(grads, dim=dim, spacing=dx)[0] # torch.gradient returns tuple
        return grads

    @staticmethod
    def laplacian(f, dx=1.0, dy=1.0):
        """Compute Laplacian of f (2D) using torch.gradient"""
        # Assuming f has shape [..., Y, X]
        grad_y = torch.gradient(f, dim=-2, spacing=dy)[0]
        grad_x = torch.gradient(f, dim=-1, spacing=dx)[0]
        
        f_yy = torch.gradient(grad_y, dim=-2, spacing=dy)[0]
        f_xx = torch.gradient(grad_x, dim=-1, spacing=dx)[0]
        
        return f_xx + f_yy

    @staticmethod
    def wave_equation(u, v, dt=0.004, dx=10.0, dy=10.0):
        """
        Wave equation residual: ∂²u/∂t² - v² * (∂²u/∂x² + ∂²u/∂y²) = 0
        Assumes u shape: [batch, time, y, x]
        Assumes v shape: [batch, 1, y, x] or [batch, y, x] (broadcastable)
        """
        # Compute second time derivative (∂²u/∂t²) - using central differences
        # Need padding or slicing to handle boundaries
        u_tt = (u[:, 2:, :, :] - 2 * u[:, 1:-1, :, :] + u[:, :-2, :, :]) / (dt**2) # Shape: [batch, time-2, y, x]

        # Compute Laplacian for each time step in the valid range
        # Need to compute Laplacian on u corresponding to the time steps of u_tt
        u_for_laplacian = u[:, 1:-1, :, :] # Shape: [batch, time-2, y, x]
        laplacian_u = torch.zeros_like(u_for_laplacian)
        
        # Calculate Laplacian spatially for each time step
        # This loop is slow; ideally vectorize if possible or use Conv2d kernel
        # for b in range(u_for_laplacian.shape[0]): # Can likely remove batch loop
        #     for t in range(u_for_laplacian.shape[1]): # Time loop might be unavoidable if memory constrained
        #         laplacian_u[b, t, :, :] = DifferentialOperators.laplacian(u_for_laplacian[b, t, :, :], dx, dy)
        
        # Vectorized Laplacian calculation across time and batch:
        grad_y = torch.gradient(u_for_laplacian, dim=-2, spacing=dy)[0]
        grad_x = torch.gradient(u_for_laplacian, dim=-1, spacing=dx)[0]
        f_yy = torch.gradient(grad_y, dim=-2, spacing=dy)[0]
        f_xx = torch.gradient(grad_x, dim=-1, spacing=dx)[0]
        laplacian_u = f_xx + f_yy # Shape: [batch, time-2, y, x]

        # Ensure v is broadcastable: Add time dimension if needed
        if v.dim() == 3: # [batch, y, x]
            v = v.unsqueeze(1) # -> [batch, 1, y, x]
        if v.dim() == 4 and v.shape[1] == 1: # [batch, 1, y, x]
             v_squared = v**2 # Broadcasts over time dim of laplacian_u implicitly
        else:
             # Handle cases where v might have time dimension? Unlikely for velocity model.
             raise ValueError(f"Unexpected velocity model shape: {v.shape}")


        # Compute residual of wave equation
        # v² * Laplacian might need broadcasting if v is [batch, 1, y, x]
        residual = u_tt - v_squared * laplacian_u # Shape: [batch, time-2, y, x]

        return residual # Residual ideally should be zero

# --- Model Components ---

# Basic ConvBlock with Residual Connection
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False) # Bias false if using BN
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Residual connection
        if in_channels == out_channels:
            self.residual = nn.Identity()
        else:
            # Projection shortcut
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
            
    def forward(self, x):
        identity = self.residual(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += identity # Add residual connection
        out = self.relu(out) # Apply activation after addition
        
        return out

# Simple Attention Block (Channel Attention like SE-Block)
class AttentionBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1) # Global Average Pooling
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c) # Squeeze
        y = self.fc(y).view(b, c, 1, 1) # Excite
        return x * y.expand_as(x) # Scale original feature map

# Physics-based encoder block (could be simpler or more complex)
class PhysicsEncoder(nn.Module):
    def __init__(self, input_channels, hidden_size=32):
        super().__init__()
        # Using ConvBlock for consistency
        self.block1 = ConvBlock(input_channels, hidden_size, kernel_size=5, padding=2)
        self.block2 = ConvBlock(hidden_size, hidden_size, kernel_size=5, padding=2)
        
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

# --- Hybrid UNet PINN Model ---
class HybridUNetPINN(nn.Module):
    def __init__(self, input_channels=INPUT_SHAPE[0], output_channels=OUTPUT_SHAPE[0], 
                 target_size=(OUTPUT_SHAPE[1], OUTPUT_SHAPE[2]), # Y, X
                 physics_weight=0.1):
        super().__init__()
        
        self.target_size = target_size # e.g., (70, 70)
        self.input_channels = input_channels
        self.output_channels = output_channels

        # --- Data-driven encoder path ---
        # Channels: input_channels -> 32 -> 64 -> 128 -> 256
        self.enc1 = ConvBlock(input_channels, 32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Reduces spatial dimensions by 2
        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc3 = ConvBlock(64, 128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.enc4 = ConvBlock(128, 256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # --- Physics encoder path ---
        # Output size should match enc1 output for concatenation later
        self.physics_encoder = PhysicsEncoder(input_channels, hidden_size=32) 
        
        # --- Bottleneck / Middle with attention ---
        # Input to middle is output of enc4 after pooling
        self.middle = ConvBlock(256, 512)
        self.attention = AttentionBlock(512) # Apply attention in the bottleneck
        
        # --- Decoder path with skip connections ---
        # Upsampling can use ConvTranspose2d or Interpolate + Conv
        self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # Input: 512, Output: 256
        self.dec4 = ConvBlock(256 + 256, 256) # Concatenate up4 output and enc4 output
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(128 + 128, 128) # Concatenate up3 output and enc3 output
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(64 + 64, 64)   # Concatenate up2 output and enc2 output
        
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        # Concatenate up1 output, enc1 output, AND physics_features output
        self.dec1 = ConvBlock(32 + 32 + 32, 32) # (up1 + enc1 + physics) -> 32
        
        # --- Final layer to get desired output channels ---
        self.final_conv = nn.Conv2d(32, output_channels, kernel_size=1)
        
        # --- Optional Physics-based refinement layer ---
        # Takes the initial output + physics features to refine the velocity model
        # Input channels: output_channels from final_conv + hidden_size from physics_encoder
        self.physics_refine = nn.Sequential(
            ConvBlock(output_channels + 32, 32), # Process combined features
            nn.Conv2d(32, output_channels, kernel_size=1) # Output refined velocity adjustment
        )
        
        # Physics weight for loss balancing (can be adjusted during training)
        self.physics_weight = physics_weight
        
        # Store normalization parameters for physics loss calculation
        self.output_mean = 0.0
        self.output_std = 1.0
        
    def set_normalization_params(self, output_mean, output_std):
        self.output_mean = output_mean
        self.output_std = output_std if output_std > 1e-8 else 1.0
        
    def forward(self, x, physics_enabled=True):
        # x shape: [B, C_in, T, X_in] -> e.g., [B, 5, 1000, 70]

        # --- Encoder ---
        enc1 = self.enc1(x)     # Output spatial size depends on input T, X_in and conv padding
        enc2 = self.enc2(self.pool1(enc1))
        enc3 = self.enc3(self.pool2(enc2))
        enc4 = self.enc4(self.pool3(enc3))
        
        # --- Physics Branch ---
        # Run in parallel with the encoder
        if physics_enabled:
            # This encoder should output features matching enc1's spatial size
            physics_features = self.physics_encoder(x) 
        else:
            # Create placeholder zeros if physics is disabled
            # Need to know the spatial size of enc1 here. Let's assume it's the same as x for now (if padding='same')
            # If convs change size, need to calculate expected size.
            # Assuming ConvBlock with kernel=3, padding=1 keeps size the same.
            physics_features_shape = (x.size(0), 32, x.size(2), x.size(3))
            physics_features = torch.zeros(physics_features_shape, device=x.device, dtype=x.dtype)
            
        # --- Bottleneck ---
        middle_in = self.pool4(enc4)
        middle_out = self.middle(middle_in)
        middle_out = self.attention(middle_out)
        
        # --- Decoder ---
        # Upsample and concatenate with skip connections
        dec4_in = self.up4(middle_out)
        # Ensure spatial dimensions match enc4 before concatenation
        # Use F.interpolate if needed, but ConvTranspose2d(stride=2) should double the size
        if dec4_in.shape[2:] != enc4.shape[2:]:
            dec4_in = F.interpolate(dec4_in, size=enc4.shape[2:], mode='bilinear', align_corners=False)
        dec4 = self.dec4(torch.cat([dec4_in, enc4], dim=1))
        
        dec3_in = self.up3(dec4)
        if dec3_in.shape[2:] != enc3.shape[2:]:
            dec3_in = F.interpolate(dec3_in, size=enc3.shape[2:], mode='bilinear', align_corners=False)
        dec3 = self.dec3(torch.cat([dec3_in, enc3], dim=1))
        
        dec2_in = self.up2(dec3)
        if dec2_in.shape[2:] != enc2.shape[2:]:
             dec2_in = F.interpolate(dec2_in, size=enc2.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat([dec2_in, enc2], dim=1))
        
        dec1_in = self.up1(dec2)
        if dec1_in.shape[2:] != enc1.shape[2:]:
             dec1_in = F.interpolate(dec1_in, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        
        # Concatenate decoder output, encoder skip connection, and physics features
        # Ensure physics_features spatial dims match enc1
        if physics_features.shape[2:] != enc1.shape[2:]:
            physics_features_resized = F.interpolate(physics_features, size=enc1.shape[2:], mode='bilinear', align_corners=False)
        else:
            physics_features_resized = physics_features
            
        dec1 = self.dec1(torch.cat([dec1_in, enc1, physics_features_resized], dim=1))

        # --- Final Processing & Cropping/Resizing ---
        # The output of dec1 might have spatial dimensions related to the *input* T, X_in
        # We need the output to be the target size (Y, X), e.g., (70, 70)
        
        # Strategy 1: Crop the center (if output is larger than target)
        # Strategy 2: Use a final conv + interpolation (more flexible)
        
        # Let's use Strategy 2 for more generality
        # Apply final conv to get initial prediction
        initial_pred_features = self.final_conv(dec1) # Shape: [B, C_out, H, W]
        
        # Interpolate features to the target size
        initial_pred = F.interpolate(initial_pred_features, size=self.target_size, mode='bilinear', align_corners=False)
        
        # --- Physics-based Refinement (Optional) ---
        if physics_enabled:
            # Also resize physics_features_resized to target size to concatenate
            physics_features_target_size = F.interpolate(physics_features_resized, size=self.target_size, mode='bilinear', align_corners=False)
            
            # Concatenate initial prediction and physics features
            refine_input = torch.cat([initial_pred, physics_features_target_size], dim=1)
            
            # Get the refinement adjustment
            refinement = self.physics_refine(refine_input)
            
            # Add the refinement to the initial prediction (residual style)
            final_pred = initial_pred + refinement
        else:
            final_pred = initial_pred
            
        return final_pred # Shape: [B, C_out, target_Y, target_X]

    def compute_physics_loss(self, seismic_data_norm, velocity_pred_norm, dx=10.0, dy=10.0, dt=0.004):
        """
        Compute physics-informed loss based on proxies or simplified wave equation terms.
        Inputs are normalized, need to denormalize velocity for physical constraints.
        """
        # --- De-normalize Predicted Velocity ---
        # Ensure model has normalization parameters set
        if not hasattr(self, 'output_mean') or not hasattr(self, 'output_std'):
             print("Warning: output_mean/output_std not set on model. Using raw prediction for physics loss.")
             velocity_pred_physical = velocity_pred_norm
        else:
             velocity_pred_physical = velocity_pred_norm * self.output_std + self.output_mean

        # --- Physics Constraint 1: Velocity Smoothness ---
        # Penalize large gradients in the velocity model
        # velocity_pred_physical shape: [B, C_out, Y, X] - assume C_out=1
        if velocity_pred_physical.shape[1] > 1:
            print("Warning: Physics loss assumes single channel velocity output. Using first channel.")
            vel_to_diff = velocity_pred_physical[:, 0:1, :, :]
        else:
            vel_to_diff = velocity_pred_physical

        vel_grad_y = DifferentialOperators.gradient(vel_to_diff, dim=-2, dx=dy)
        vel_grad_x = DifferentialOperators.gradient(vel_to_diff, dim=-1, dx=dx)
        # L2 norm of gradients
        smoothness_loss = torch.mean(vel_grad_x**2 + vel_grad_y**2)

        # --- Physics Constraint 2: Realistic Velocity Bounds ---
        # Penalize velocities outside a plausible physical range (e.g., 1.5 km/s to 6.0 km/s)
        # Use ReLU for one-sided penalty
        lower_bound = 1.5 # km/s
        upper_bound = 6.0 # km/s
        # Ensure output_mean/std are in compatible units (e.g., km/s)
        realistic_bounds_loss = torch.mean(
            F.relu(lower_bound - velocity_pred_physical) + \
            F.relu(velocity_pred_physical - upper_bound)
        )
        
        # --- Physics Constraint 3: Wave Equation Residual (Simplified/Proxy) ---
        # Option A: Use the *actual* wave equation (Computationally expensive, requires input wavefield 'u')
        # This requires the input seismic data to be the actual wavefield 'u'.
        # If seismic_data_norm represents 'u', we need to denormalize it too.
        # wave_eq_loss = torch.mean(DifferentialOperators.wave_equation(u_physical, velocity_pred_physical, dt, dx, dy)**2)

        # Option B: Use a proxy - e.g., penalize mismatch between wavefield gradients and velocity
        # This is highly problem-specific and requires careful formulation.

        # Let's stick to Smoothness and Bounds for this implementation, as they are common and robust.
        # The full wave equation requires careful handling of inputs/outputs.
        
        # --- Combine Physics Losses ---
        # Weights can be tuned
        w_smooth = 1.0
        w_bounds = 0.5 # Can adjust weight relative to smoothness
        
        total_physics_loss = w_smooth * smoothness_loss + w_bounds * realistic_bounds_loss
        
        # Apply the overall physics_weight scaling factor
        return total_physics_loss * self.physics_weight


# --- Combined Loss Function ---
class HybridLoss(nn.Module):
    def __init__(self, alpha=0.8, ssim_weight=0.1):
        super().__init__()
        self.alpha = alpha # Weight for L1 vs MSE/Structural term
        self.ssim_weight = ssim_weight # Weight for the SSIM-like term
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss() # Keep MSE as an option or part of structural term

    def ssim_like_loss(self, pred, target, C1=1e-4, C2=9e-4):
        """ Simplified SSIM-like loss component focusing on structure """
        mu_p = torch.mean(pred, dim=[2, 3], keepdim=True)
        mu_t = torch.mean(target, dim=[2, 3], keepdim=True)
        
        sigma_p_sq = torch.mean((pred - mu_p)**2, dim=[2, 3], keepdim=True)
        sigma_t_sq = torch.mean((target - mu_t)**2, dim=[2, 3], keepdim=True)
        sigma_pt = torch.mean((pred - mu_p) * (target - mu_t), dim=[2, 3], keepdim=True)
        
        # Calculate SSIM components (simplified)
        # Luminance term (using means) - less important if data is normalized
        # Contrast term (using variances)
        # Structure term (using covariance)
        
        # Use the structural term: (2 * sigma_pt + C2) / (sigma_p_sq + sigma_t_sq + C2)
        structural_term = (2.0 * sigma_pt + C2) / (sigma_p_sq + sigma_t_sq + C2)
        
        # Loss is typically 1 - SSIM
        # We average over batch and channels
        return torch.mean(1.0 - structural_term)

    def forward(self, pred, target, physics_loss=None):
        # --- Data-driven component ---
        l1_loss = self.l1(pred, target)
        
        # Optional: Add SSIM-like structural loss
        # structural_loss = self.ssim_like_loss(pred, target)
        
        # Combine L1 and MSE (or structural loss)
        # data_loss = self.alpha * l1_loss + (1 - self.alpha) * structural_loss
        # Simpler: Weighted sum of L1 and MSE
        mse_loss = self.mse(pred, target)
        data_loss = self.alpha * l1_loss + (1 - self.alpha) * mse_loss
        
        # --- Combine with Physics-based loss ---
        if physics_loss is not None:
            total_loss = data_loss + physics_loss # Physics loss is already weighted in compute_physics_loss
        else:
            total_loss = data_loss
            
        return total_loss, data_loss # Return both for logging

# --- Initialization ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = HybridUNetPINN(
    input_channels=INPUT_SHAPE[0], 
    output_channels=OUTPUT_SHAPE[0],
    target_size=(OUTPUT_SHAPE[1], OUTPUT_SHAPE[2]),
    physics_weight=0.1 # Initial physics weight
).to(device)

# Set normalization parameters on the model instance (important for physics loss)
model.set_normalization_params(output_mean, output_std)

# --- Optimizer ---
# Use AdamW with weight decay
# Separate parameter groups for potentially different learning rates (optional but good practice)
physics_param_ids = set(id(p) for p in model.physics_encoder.parameters()) | \
                    set(id(p) for p in model.physics_refine.parameters())

base_params = [p for p in model.parameters() if id(p) not in physics_param_ids]
physics_params = [p for p in model.parameters() if id(p) in physics_param_ids]

optimizer = torch.optim.AdamW([
    {'params': base_params, 'lr': 3e-4, 'weight_decay': 1e-5}, # Slightly higher LR for main path
    {'params': physics_params, 'lr': 1e-4, 'weight_decay': 1e-5} # Lower LR for physics path potentially
], lr=3e-4) # Default LR if not specified in groups

# --- Loss Function ---
criterion = HybridLoss(alpha=0.8) # 80% L1, 20% MSE for data loss

# --- Learning Rate Scheduler ---
# Cosine annealing is a good choice
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# --- Mixed Precision Training Scaler ---
scaler = GradScaler(enabled=torch.cuda.is_available()) # Only enable if using CUDA

# --- Early Stopping ---
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0001, verbose=True): # Increased patience
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False
        self.verbose = verbose
        
    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            if self.verbose:
                print(f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}). Resetting counter.")
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"Validation loss did not improve from {self.best_loss:.6f}. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                if self.verbose:
                    print("Early stopping triggered.")
                self.early_stop = True

early_stopping = EarlyStopping(patience=10) # More patience

# --- Checkpoint Saving ---
def save_checkpoint(model, optimizer, scheduler, epoch, loss, filename="checkpoint.pt"):
    print(f"Saving checkpoint to {filename}...")
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if scheduler is not None:
        checkpoint['scheduler_state_dict'] = scheduler.state_dict()
    torch.save(checkpoint, filename)

# --- Training Function ---
def train_epoch(model, dataloader, optimizer, criterion, device, scaler, current_physics_weight):
    model.train()
    total_loss_accum = 0.0
    data_loss_accum = 0.0
    physics_loss_accum = 0.0
    
    # Update model's internal physics weight
    model.physics_weight = current_physics_weight 
    
    progress_bar = tqdm(dataloader, desc='Train', leave=False)
    for data in progress_bar:
        # Inputs: normalized seismic data [B, C_in, T, X]
        # Targets: normalized velocity model [B, C_out, Y, X]
        # Inputs_orig, Targets_orig: original scale data (not used directly in this training loop version)
        inputs, targets, _, _ = data 
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        optimizer.zero_grad()
        
        # Use autocast for mixed precision
        with autocast(enabled=scaler.is_enabled()):
            # Forward pass
            outputs = model(inputs, physics_enabled=True) # Ensure physics path is active
            
            # Compute physics loss (using normalized inputs/outputs as needed by the function)
            # compute_physics_loss handles denormalization internally if required
            physics_loss = model.compute_physics_loss(inputs, outputs)
            
            # Compute total loss (data loss + physics loss)
            total_loss, data_loss = criterion(outputs, targets, physics_loss)
            
        # Scale loss and backward pass
        scaler.scale(total_loss).backward()
        
        # Gradient clipping (optional but recommended)
        scaler.unscale_(optimizer) # Unscale gradients before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Optimizer step
        scaler.step(optimizer)
        
        # Update scaler for next iteration
        scaler.update()
        
        # Accumulate losses for epoch average
        total_loss_accum += total_loss.item()
        data_loss_accum += data_loss.item() # data_loss from criterion
        physics_loss_accum += physics_loss.item() # physics_loss from model.compute_physics_loss

        # Update progress bar
        progress_bar.set_postfix({
            'loss': total_loss.item(), 
            'data_l': data_loss.item(), 
            'phys_l': physics_loss.item()
        })

    
    n_batches = len(dataloader)
    avg_total_loss = total_loss_accum / n_batches
    avg_data_loss = data_loss_accum / n_batches
    avg_physics_loss = physics_loss_accum / n_batches
    
    return {
        'total_loss': avg_total_loss,
        'data_loss': avg_data_loss,
        'physics_loss': avg_physics_loss
    }

# --- Validation Function ---
def validate(model, dataloader, criterion, device, current_physics_weight):
    model.eval()
    total_loss_accum = 0.0
    data_loss_accum = 0.0
    physics_loss_accum = 0.0
    
    # Ensure model's physics weight is set for validation consistency (though it doesn't affect gradients)
    model.physics_weight = current_physics_weight
    
    with torch.no_grad(): # No gradients needed for validation
        progress_bar = tqdm(dataloader, desc='Valid', leave=False)
        for data in progress_bar:
            inputs, targets, _, _ = data
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass - can disable physics computation here if only evaluating data loss for validation speed
            # outputs = model(inputs, physics_enabled=False) 
            # physics_loss = torch.tensor(0.0) # Set to zero if disabled
            
            # OR evaluate with physics enabled for comparable loss metric
            outputs = model(inputs, physics_enabled=True)
            physics_loss = model.compute_physics_loss(inputs, outputs)

            # Compute loss
            total_loss, data_loss = criterion(outputs, targets, physics_loss)
            
            # Accumulate losses
            total_loss_accum += total_loss.item()
            data_loss_accum += data_loss.item()
            physics_loss_accum += physics_loss.item()

            progress_bar.set_postfix({
                'val_loss': total_loss.item()            
            })

    n_batches = len(dataloader)
    avg_total_loss = total_loss_accum / n_batches
    avg_data_loss = data_loss_accum / n_batches
    avg_physics_loss = physics_loss_accum / n_batches
    
    return {
        'total_loss': avg_total_loss,
        'data_loss': avg_data_loss,
        'physics_loss': avg_physics_loss
    }


# --- Training Loop ---
n_epochs = 50 # Or adjust as needed
history = []
best_valid_loss = float('inf')

# Dynamic weighting schedule for physics loss (optional)
# Starts low, increases, then potentially stabilizes
# Example: Linear increase then constant
warmup_epochs = 10
max_physics_weight = 0.2 # Max weight to reach
physics_weight_schedule = np.concatenate([
    np.linspace(0.01, max_physics_weight, warmup_epochs), # Linear increase
    np.full(n_epochs - warmup_epochs, max_physics_weight) # Stay at max
])
# Ensure schedule length matches n_epochs
if len(physics_weight_schedule) < n_epochs:
     physics_weight_schedule = np.pad(physics_weight_schedule, (0, n_epochs - len(physics_weight_schedule)), mode='edge')
elif len(physics_weight_schedule) > n_epochs:
     physics_weight_schedule = physics_weight_schedule[:n_epochs]


print("Starting training...")
for epoch in range(1, n_epochs + 1):
    start_time = time.time()
    
    # Get current physics weight from schedule
    current_physics_weight = physics_weight_schedule[epoch-1]
    
    # Train
    train_metrics = train_epoch(model, dltrain, optimizer, criterion, device, scaler, current_physics_weight)
    
    # Validate
    valid_metrics = validate(model, dlvalid, criterion, device, current_physics_weight)
    
    # Step the scheduler (after validation)
    if scheduler:
        scheduler.step()
    
    epoch_time = time.time() - start_time
    
    # Log history
    current_lr = optimizer.param_groups[0]['lr'] # Get current LR
    history.append({
        'epoch': epoch,
        'train_loss': train_metrics['total_loss'],
        'valid_loss': valid_metrics['total_loss'],
        'train_data_loss': train_metrics['data_loss'],
        'valid_data_loss': valid_metrics['data_loss'],
        'train_physics_loss': train_metrics['physics_loss'],
        'valid_physics_loss': valid_metrics['physics_loss'],
        'physics_weight': current_physics_weight,
        'lr': current_lr,
        'time': epoch_time
    })
    
    # Print epoch summary
    print(f'Epoch {epoch}/{n_epochs} | T Loss: {train_metrics["total_loss"]:.5f} | V Loss: {valid_metrics["total_loss"]:.5f} | '
          f'T Data L: {train_metrics["data_loss"]:.5f} | V Data L: {valid_metrics["data_loss"]:.5f} | '
          f'T Phys L: {train_metrics["physics_loss"]:.5f} | V Phys L: {valid_metrics["physics_loss"]:.5f} | '
          f'Phys W: {current_physics_weight:.3f} | LR: {current_lr:.6f} | Time: {epoch_time:.2f}s')
    
    # Save best model based on validation loss
    if valid_metrics['total_loss'] < best_valid_loss:
        print(f"Validation loss improved ({best_valid_loss:.6f} --> {valid_metrics['total_loss']:.6f}). Saving best model...")
        best_valid_loss = valid_metrics['total_loss']
        save_checkpoint(model, optimizer, scheduler, epoch, best_valid_loss, 'best_hybrid_pinn_model.pt')
    
    # --- Visualization (every few epochs) ---
    if epoch % 5 == 0 or epoch == 1 or epoch == n_epochs:
        print("Generating visualization...")
        model.eval() # Ensure model is in eval mode for visualization
        with torch.no_grad():
            # Get a batch from validation loader
            try:
                 vis_inputs, vis_targets, _, _ = next(iter(dlvalid))
                 vis_inputs = vis_inputs.to(device)
                 vis_targets = vis_targets.to(device) # Targets remain normalized here
                 
                 # Get predictions with and without physics (for comparison)
                 outputs_with_physics = model(vis_inputs, physics_enabled=True)
                 outputs_without_physics = model(vis_inputs, physics_enabled=False) # Requires model changes or separate pass
                 
                 # Select first item in batch for visualization
                 idx_vis = 0
                 
                 # Denormalize for visualization (using stored stats)
                 target_vis = vis_targets[idx_vis, 0].cpu().numpy() * output_std + output_mean
                 pred_phys_vis = outputs_with_physics[idx_vis, 0].cpu().numpy() * output_std + output_mean
                 pred_no_phys_vis = outputs_without_physics[idx_vis, 0].cpu().numpy() * output_std + output_mean
                 
                 fig, axes = plt.subplots(2, 2, figsize=(12, 10)) # Adjusted size
                 fig.suptitle(f'Epoch {epoch} | Validation Loss: {valid_metrics["total_loss"]:.5f}')
                 
                 # Plot Ground Truth
                 ax = axes[0, 0]
                 im = ax.imshow(target_vis, aspect='auto', cmap='viridis')
                 ax.set_title('Ground Truth Velocity')
                 plt.colorbar(im, ax=ax, label='Velocity (km/s?)') # Add unit guess
                 
                 # Plot Prediction with Physics
                 ax = axes[0, 1]
                 im = ax.imshow(pred_phys_vis, aspect='auto', cmap='viridis')
                 ax.set_title('Prediction (with Physics)')
                 plt.colorbar(im, ax=ax, label='Velocity (km/s?)')
                 
                 # Plot Prediction without Physics
                 ax = axes[1, 0]
                 im = ax.imshow(pred_no_phys_vis, aspect='auto', cmap='viridis')
                 ax.set_title('Prediction (Data-Only)')
                 plt.colorbar(im, ax=ax, label='Velocity (km/s?)')
                 
                 # Plot Difference (Ground Truth - Prediction with Physics)
                 ax = axes[1, 1]
                 diff = target_vis - pred_phys_vis
                 vmax = np.abs(diff).max() # Symmetric color scale around zero
                 im = ax.imshow(diff, aspect='auto', cmap='seismic', vmin=-vmax, vmax=vmax)
                 ax.set_title('Difference (GT - Physics Pred)')
                 plt.colorbar(im, ax=ax, label='Velocity Difference')
                 
                 plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
                 plt.savefig(f'epoch_{epoch}_visualization.png') # Save the figure
                 plt.show()

            except Exception as e:
                 print(f"Error during visualization: {e}")
                 # Continue training even if visualization fails


    # --- Early stopping check ---
    early_stopping(valid_metrics['total_loss'])
    if early_stopping.early_stop:
        break # Exit training loop

print("Training finished.")

# --- Plot Training History ---
print("Plotting training history...")
history_df = pd.DataFrame(history)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True) # Share x-axis (epochs)

# Plot Losses on ax1
ax1.plot(history_df['epoch'], history_df['train_loss'], label='Train Total Loss', color='tab:blue')
ax1.plot(history_df['epoch'], history_df['valid_loss'], label='Valid Total Loss', color='tab:orange')
ax1.plot(history_df['epoch'], history_df['train_data_loss'], label='Train Data Loss', linestyle='--', color='tab:blue', alpha=0.7)
ax1.plot(history_df['epoch'], history_df['valid_data_loss'], label='Valid Data Loss', linestyle='--', color='tab:orange', alpha=0.7)
ax1.plot(history_df['epoch'], history_df['train_physics_loss'], label='Train Physics Loss', linestyle=':', color='tab:green', alpha=0.7)
ax1.plot(history_df['epoch'], history_df['valid_physics_loss'], label='Valid Physics Loss', linestyle=':', color='tab:red', alpha=0.7)
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Losses')
ax1.legend(loc='upper right')
ax1.grid(True, linestyle='--', alpha=0.6)
# Consider log scale if losses vary widely: ax1.set_yscale('log')

# Plot Learning Rate and Physics Weight on ax2
ax2_lr = ax2
ax2_pw = ax2.twinx() # Create secondary y-axis for physics weight

# Plot LR
line_lr, = ax2_lr.plot(history_df['epoch'], history_df['lr'], label='Learning Rate (LR)', color='tab:purple')
ax2_lr.set_xlabel('Epoch')
ax2_lr.set_ylabel('Learning Rate', color=line_lr.get_color())
ax2_lr.tick_params(axis='y', labelcolor=line_lr.get_color())
# Optional: Use log scale for LR if it changes drastically
# ax2_lr.set_yscale('log')

# Plot Physics Weight
line_pw, = ax2_pw.plot(history_df['epoch'], history_df['physics_weight'], label='Physics Weight', color='tab:cyan')
ax2_pw.set_ylabel('Physics Weight', color=line_pw.get_color())
ax2_pw.tick_params(axis='y', labelcolor=line_pw.get_color())
ax2_pw.set_ylim(bottom=0) # Physics weight shouldn't be negative

ax2.set_title('Learning Rate and Physics Weight Schedule')
# Combine legends from both y-axes
lines = [line_lr, line_pw]
labels = [l.get_label() for l in lines]
ax2_lr.legend(lines, labels, loc='center right')
ax2.grid(True, linestyle='--', alpha=0.6)


fig.tight_layout()
plt.savefig('training_history.png') # Save the history plot
plt.show()

print("Script finished.")

