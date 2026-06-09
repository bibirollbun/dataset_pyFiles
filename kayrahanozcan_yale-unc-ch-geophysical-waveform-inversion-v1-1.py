
"""
Yale/UNC-CH - Geophysical Waveform Inversion Competition - Baseline Script

Description:
This script implements a baseline machine learning model for the Kaggle
competition focused on Full Waveform Inversion (FWI). The goal is to estimate
subsurface velocity models from seismic waveform data.

Approach:
- Model: Multi-Layer Perceptron (MLP) applied to pooled input features.
- Input Processing: Max Pooling reduces dimensionality before MLP layers.
- Features: LeakyReLU activations, Dropout, Kaiming Initialization.
- Training: AdamW optimizer, ReduceLROnPlateau scheduler, L1Loss (MAE),
            Mixed Precision (AMP) on GPU, Early Stopping.
- Data Handling: PyTorch Dataset with memory mapping for large files.
- Augmentation: Training data augmentation (noise, horizontal flip).
- Evaluation: Mean Absolute Error (MAE), as per competition rules.
- Inference: Test-Time Augmentation (TTA) using horizontal flips.

Structure:
1. Imports
2. Configuration (CFG Class)
3. Utility Functions (Seeding)
4. Data Loading & Preparation (Datasets, Splitting, Augmentation)
5. Model Definition (FWINet Class)
6. Training & Validation Functions
7. Prediction & Submission Functions
8. Main Execution Block

"""

# ======================================================
# 1. Library Imports
# ======================================================
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
import csv
import time
import gc  # Garbage Collection
import random
import os

print("Libraries imported successfully.")

# Set Matplotlib style for better plots
plt.style.use('seaborn-v0_8-whitegrid')

# ======================================================
# 2. Configuration
# ======================================================
class CFG:
    """Configuration class for hyperparameters and settings."""
    # Paths
    BASE_DIR = '/kaggle/input/waveform-inversion'
    TRAIN_DIR = os.path.join(BASE_DIR, 'train_samples')
    TEST_DIR = os.path.join(BASE_DIR, 'test')
    OUTPUT_DIR = '/kaggle/working/' # Writable directory in Kaggle
    SUBMISSION_FILE = os.path.join(OUTPUT_DIR, 'submission.csv')
    BEST_MODEL_PATH = os.path.join(OUTPUT_DIR, 'best_model_final.pth') # Path to save best model

    # Data Parameters
    N_EXAMPLES_PER_FILE = 500  # Assumed number of samples in each .npy file
    IMG_HEIGHT = 70 # Target velocity model height
    IMG_WIDTH = 70  # Target velocity model width
    INPUT_CHANNELS = 5 # Deduced from typical seismic data shape (Channels, Time, Width)
    INPUT_TIME_STEPS = 1000 # Deduced from typical seismic data shape

    # Model Parameters
    POOL_SIZE = (8, 2) # Kernel size for Max Pooling (Time/Depth, Width)
    HIDDEN_SIZE_FACTOR = 1.0 # Factor to scale hidden layer size relative to output size
    DROPOUT_RATE = 0.3 # Dropout probability for regularization
    OUTPUT_SCALE = 1000.0 # Scaling factor applied to model output
    OUTPUT_OFFSET = 1500.0 # Offset applied to model output (adjusts range)

    # Training Parameters
    SEED = 42 # Random seed for reproducibility
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Auto-detect GPU
    N_EPOCHS = 70 # Maximum number of training epochs (use early stopping)
    BATCH_SIZE = 32 # Samples per batch (adjust based on GPU memory)
    NUM_WORKERS = 2 # Number of parallel workers for DataLoader (adjust based on Kaggle limits)
    LEARNING_RATE = 3e-4 # Initial learning rate for AdamW
    WEIGHT_DECAY = 1e-5 # Weight decay for AdamW (L2 regularization)
    LR_SCHEDULER_PATIENCE = 5 # Epochs to wait for improvement before reducing LR
    LR_SCHEDULER_FACTOR = 0.5 # Factor by which LR is reduced
    EARLY_STOPPING_PATIENCE = 10 # Epochs to wait for improvement before stopping training
    GRADIENT_CLIP_NORM = 1.0 # Maximum norm for gradient clipping
    VALIDATION_SPLIT_RATIO = 0.5 # Use every 1/ratio file for validation (0.5 -> every 2nd file)

    # Augmentation
    AUGMENT_TRAIN = True # Enable/disable training data augmentation
    AUG_NOISE_LEVEL_MIN = 0.001 # Min noise level for augmentation
    AUG_NOISE_LEVEL_MAX = 0.01 # Max noise level for augmentation
    AUG_FLIP_PROB = 0.5 # Probability of applying horizontal flip augmentation

    # TTA Parameters
    USE_TTA = True # Enable/disable Test-Time Augmentation
    TTA_N_AUGMENTATIONS = 1 # Number of *additional* TTA predictions (currently only flip, so 1)

    # Visualization
    VISUALIZATION_INTERVAL = 5 # Visualize validation predictions every N epochs

print(f"Configuration loaded. Device set to: {CFG.DEVICE}")

# ======================================================
# 3. Utility Functions
# ======================================================
def set_seed(seed=CFG.SEED):
    """Sets random seeds for reproducibility across libraries."""
    print(f"Setting random seed to {seed}")
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) # For multi-GPU setups
        # Ensure deterministic behavior for CuDNN operations if reproducibility is critical
        # Note: This can impact performance. Set False if speed is prioritized over exact reproducibility.
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False # Disable benchmarking for determinism
    print("Seed setting complete.")

# Set the seed globally at the start of the script
set_seed(CFG.SEED)

# ======================================================
# 4. Data Loading and Preparation
# ======================================================
def find_data_files(base_dir):
    """Finds all input seismic data files based on naming conventions."""
    print(f"Searching for input data files in: {base_dir}")
    # Sort files for consistent splitting behavior across runs
    files = sorted([
        f for f in Path(base_dir).rglob('*.npy')
        if ('seis' in f.stem) or ('data' in f.stem)
    ])
    print(f"Found {len(files)} potential input files.")
    if not files:
        print("Warning: No input files found in the specified directory.")
    return files

def map_inputs_to_outputs(input_files):
    """Maps input file paths to corresponding output velocity model file paths, checking existence."""
    output_files = []
    missing_outputs = []
    filtered_inputs = []
    print("Mapping input files to output files and verifying existence...")
    for f in tqdm(input_files, desc="Mapping files", leave=False, dynamic_ncols=True):
        # Construct the expected output filename based on convention
        out_f = Path(str(f).replace('seis', 'vel').replace('data', 'model'))
        if out_f.exists():
            output_files.append(out_f)
            filtered_inputs.append(f) # Only keep input if its corresponding output exists
        else:
            print(f"Warning: Output file not found for input {f.name}, skipping this pair.")
            missing_outputs.append(f)

    if missing_outputs:
         print(f"Warning: {len(missing_outputs)} input files were skipped due to missing outputs.")

    assert len(filtered_inputs) == len(output_files), "Input and output file counts mismatch after filtering."
    print(f"Successfully mapped {len(filtered_inputs)} valid input/output file pairs.")
    return filtered_inputs, output_files

def split_train_validation(all_inputs, all_outputs, split_ratio=CFG.VALIDATION_SPLIT_RATIO):
    """
    Splits the data into training and validation sets based on file indices using a stride.
    """
    assert 0 < split_ratio < 1, "Validation split ratio must be between 0 and 1."
    num_files = len(all_inputs)
    if num_files == 0:
        raise ValueError("Cannot split empty file lists.")

    # Calculate stride (e.g., split_ratio=0.5 -> stride=2 -> use every 2nd file for validation)
    stride = max(1, int(round(1.0 / split_ratio))) # Ensure stride is at least 1
    print(f"Splitting data with validation ratio ~{1.0/stride:.2f} (using stride={stride})")

    # Select indices for the validation set using the calculated stride
    valid_indices = list(range(0, num_files, stride))

    # Create lists based on selected indices
    valid_inputs = [all_inputs[i] for i in valid_indices]
    valid_outputs = [all_outputs[i] for i in valid_indices]

    train_inputs = [f for i, f in enumerate(all_inputs) if i not in valid_indices]
    train_outputs = [f for i, f in enumerate(all_outputs) if i not in valid_indices]

    print(f"Total file pairs: {num_files}")
    print(f"Training file pairs: {len(train_inputs)}")
    print(f"Validation file pairs: {len(valid_inputs)}")

    if not train_inputs or not valid_inputs:
         print("Warning: Training or validation set is empty after splitting. Check ratio and file count.")

    return train_inputs, train_outputs, valid_inputs, valid_outputs

class SeismicDataset(Dataset):
    """
    PyTorch Dataset for loading seismic data (input) and velocity models (target).
    Utilizes memory mapping for efficient handling of potentially large files.
    Includes options for data augmentation during training.
    """
    def __init__(self, inputs_files, output_files, n_examples_per_file=CFG.N_EXAMPLES_PER_FILE, augment=False, cfg=CFG):
        """Initializes the dataset."""
        assert len(inputs_files) == len(output_files), "Input and output file counts must match."
        if not inputs_files:
            print("Warning: Initializing SeismicDataset with zero files.")
        self.inputs_files = inputs_files
        self.output_files = output_files
        self.n_examples_per_file = n_examples_per_file
        self.augment = augment
        self.cfg = cfg
        # Basic check for file existence (checks only the first file pair for speed)
        self._check_first_file_exists()

    def _check_first_file_exists(self):
        """Quick check if the first files in the lists exist."""
        if self.inputs_files and not self.inputs_files[0].exists():
             raise FileNotFoundError(f"First input file specified does not exist: {self.inputs_files[0]}")
        if self.output_files and not self.output_files[0].exists():
             raise FileNotFoundError(f"First output file specified does not exist: {self.output_files[0]}")
        print("First input/output file pair checked successfully.")

    def __len__(self):
        """Returns the total number of samples across all files."""
        return len(self.inputs_files) * self.n_examples_per_file

    def _apply_augmentation(self, x, y):
        """Applies configured data augmentation techniques (noise, flip)."""
        # Ensure operating on copies
        x = x.copy()
        y = y.copy()

        # 1. Random Noise Injection
        if np.random.random() < 0.5: # 50% chance to add noise
            noise_level = np.random.uniform(self.cfg.AUG_NOISE_LEVEL_MIN, self.cfg.AUG_NOISE_LEVEL_MAX)
            noise = noise_level * np.random.randn(*x.shape).astype(np.float32)
            x = x + noise

        # 2. Random Horizontal Flip
        if np.random.random() < self.cfg.AUG_FLIP_PROB:
            # Flip input (Channels, Time, Width) along Width axis (axis=2)
            # Flip target (1, Height, Width) along Width axis (axis=2 if dim is 3, axis=1 if dim is 2)
            x = np.flip(x, axis=2).copy() # .copy() ensures positive strides
            # Adjust target flip axis based on actual dimensions before applying this!
            # Assuming y is [1, H, W] -> flip axis 2. If y is [H, W] -> flip axis 1.
            y_flip_axis = 2 if y.ndim == 3 else 1
            y = np.flip(y, axis=y_flip_axis).copy()

        return x, y

    def __getitem__(self, idx):
        """Loads a single sample (input, target) using memory mapping."""
        if len(self.inputs_files) == 0:
            raise IndexError("Dataset is empty, cannot retrieve item.")

        # Determine which file and sample within that file corresponds to the global index 'idx'
        file_idx = idx // self.n_examples_per_file
        sample_idx = idx % self.n_examples_per_file

        if file_idx >= len(self.inputs_files):
             raise IndexError(f"Calculated file index {file_idx} is out of bounds for {len(self.inputs_files)} files.")

        input_file_path = self.inputs_files[file_idx]
        output_file_path = self.output_files[file_idx]

        try:
            # Use memory mapping ('r' mode) for efficient read-only access.
            X_mmap = np.load(input_file_path, mmap_mode='r')
            y_mmap = np.load(output_file_path, mmap_mode='r')

            # Check if sample_idx is valid for the loaded memory maps
            if sample_idx >= X_mmap.shape[0] or sample_idx >= y_mmap.shape[0]:
                raise IndexError(f"Sample index {sample_idx} out of bounds for file {input_file_path.name} (Shapes: X={X_mmap.shape}, y={y_mmap.shape})")

            # Access the specific sample, triggering data read.
            # Convert to float32 and copy data from mmap to ensure it's in memory.
            X_sample = X_mmap[sample_idx].copy().astype(np.float32)
            y_sample = y_mmap[sample_idx].copy().astype(np.float32)

            # Explicitly close mmap objects (optional but good practice)
            del X_mmap, y_mmap

            # Ensure target has a channel dimension: [H, W] -> [1, H, W]
            if y_sample.ndim == 2:
                y_sample = np.expand_dims(y_sample, axis=0)

            # Apply augmentation only if specified for this dataset instance (typically training set)
            if self.augment:
                X_sample, y_sample = self._apply_augmentation(X_sample, y_sample)

            # Convert numpy arrays to PyTorch tensors
            X_tensor = torch.from_numpy(X_sample)
            y_tensor = torch.from_numpy(y_sample)

            return X_tensor, y_tensor

        except FileNotFoundError as e:
            print(f"FATAL ERROR: File not found during loading! {e}")
            raise e
        except IndexError as e:
            print(f"FATAL ERROR: Index out of bounds during loading! {e}")
            raise e
        except Exception as e:
            print(f"FATAL ERROR loading data: File='{input_file_path.name}', Sample Index={sample_idx}. Exception: {e}")
            raise e

print("Dataset class defined.")

# ======================================================
# 5. Model Definition
# ======================================================
class FWINet(nn.Module):
    """
    Feedforward Network (MLP) with initial Pooling layer for Full Waveform Inversion.
    Takes seismic data (Batch, Channels, Time, Width) and predicts velocity maps (Batch, 1, Height, Width).
    """
    def __init__(self, cfg=CFG):
        super().__init__()
        self.cfg = cfg

        # Input dimensions
        input_channels = cfg.INPUT_CHANNELS
        input_time_steps = cfg.INPUT_TIME_STEPS
        input_width = cfg.IMG_WIDTH # Assuming input width relevant for pooling

        # 1. Pooling layer: Reduces Time/Depth and Width dimensions
        self.pool = nn.MaxPool2d(kernel_size=cfg.POOL_SIZE)

        # Calculate the flattened feature size after pooling
        pooled_time = input_time_steps // cfg.POOL_SIZE[0]
        pooled_width = input_width // cfg.POOL_SIZE[1]
        flattened_size = input_channels * pooled_time * pooled_width
        if flattened_size <= 0:
             raise ValueError(f"Flattened size after pooling is non-positive ({flattened_size}). Check input dims ({input_channels}x{input_time_steps}x{input_width}) and POOL_SIZE {cfg.POOL_SIZE}.")
        print(f"Flattened feature size after pooling: {flattened_size}")

        # 2. MLP Layers
        output_size = cfg.IMG_HEIGHT * cfg.IMG_WIDTH
        hidden_size_base = int(output_size * cfg.HIDDEN_SIZE_FACTOR)
        # Define hidden layer sizes, ensuring a minimum size relative to output
        hidden_sizes = [
            max(output_size // 2, hidden_size_base),
            max(output_size // 4, hidden_size_base // 2),
            max(output_size // 8, hidden_size_base // 4)
        ]
        print(f"MLP hidden layer sizes: {hidden_sizes}")

        self.network = nn.Sequential(
            nn.Linear(flattened_size, hidden_sizes[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg.DROPOUT_RATE),

            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg.DROPOUT_RATE),

            nn.Linear(hidden_sizes[1], hidden_sizes[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(cfg.DROPOUT_RATE),

            nn.Linear(hidden_sizes[2], output_size) # Final layer projects to flattened output map size
        )

        # Apply Kaiming initialization, suitable for LeakyReLU
        self._initialize_weights()
        print("Model initialized with Kaiming Normal weights.")

    def _initialize_weights(self):
        """Initializes Linear layer weights using Kaiming Normal."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """Defines the forward pass of the model."""
        batch_size = x.shape[0]

        # Ensure input tensor is float32
        x = x.float()

        # --- 1. Input Feature Scaling (Instance-like normalization per sample) ---
        mean = torch.mean(x, dim=(2, 3), keepdim=True)
        std = torch.std(x, dim=(2, 3), keepdim=True)
        x_norm = (x - mean) / (std + 1e-8) # Epsilon for numerical stability

        # --- 2. Max Pooling ---
        x_pooled = self.pool(x_norm)

        # --- 3. Flatten Features ---
        x_flattened = x_pooled.view(batch_size, -1) # Flatten all dims except batch

        # --- 4. Pass through MLP ---
        output_flat = self.network(x_flattened)

        # --- 5. Reshape to Output Image Dimensions ---
        # Reshape from (batch, height * width) -> (batch, 1, height, width)
        output_image = output_flat.view(batch_size, 1, self.cfg.IMG_HEIGHT, self.cfg.IMG_WIDTH)

        # --- 6. Apply Output Scaling and Offset ---
        # Map network output to the expected physical velocity range
        output_scaled = output_image * self.cfg.OUTPUT_SCALE + self.cfg.OUTPUT_OFFSET

        return output_scaled

print("Model class 'FWINet' defined.")

# ======================================================
# 6. Training and Validation Functions
# ======================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler, cfg):
    """Performs one training epoch."""
    model.train()
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc=f'Training Epoch', leave=False, dynamic_ncols=True)

    for inputs, targets in progress_bar:
        inputs = inputs.to(device, non_blocking=True).float()
        targets = targets.to(device, non_blocking=True).float()

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        if scaler: # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer) # Unscale before clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_NORM)
            scaler.step(optimizer)
            scaler.update()
        else: # Standard precision backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_NORM)
            optimizer.step()

        loss_item = loss.item()
        total_loss += loss_item
        progress_bar.set_postfix(loss=f'{loss_item:.4f}')

    avg_loss = total_loss / len(dataloader)
    # print(f"Epoch Training Avg Loss: {avg_loss:.5f}")
    return avg_loss


def validate_one_epoch(model, dataloader, criterion, device, cfg):
    """Performs one validation epoch."""
    model.eval()
    total_loss = 0.0
    first_batch_outputs = None
    first_batch_targets = None
    progress_bar = tqdm(dataloader, desc=f'Validation Epoch', leave=False, dynamic_ncols=True)

    with torch.inference_mode():
        for i, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device, non_blocking=True).float()
            targets = targets.to(device, non_blocking=True).float()

            with torch.cuda.amp.autocast(enabled=(device == torch.device('cuda'))):
                 outputs = model(inputs)

            loss = criterion(outputs, targets)
            loss_item = loss.item()
            total_loss += loss_item
            progress_bar.set_postfix(loss=f'{loss_item:.4f}')

            # Store the first sample of the first batch for visualization
            if i == 0:
                first_batch_outputs = outputs[0:1].detach().cpu()
                first_batch_targets = targets[0:1].detach().cpu()

    avg_loss = total_loss / len(dataloader)
    # print(f"Epoch Validation Avg Loss: {avg_loss:.5f}")
    return avg_loss, first_batch_outputs, first_batch_targets


def visualize_prediction(target, prediction, epoch, loss, cfg):
    """Visualizes a comparison between ground truth and model prediction."""
    if target is None or prediction is None:
        print(f"Epoch {epoch}: Skipping visualization due to missing sample data.")
        return

    target_np = target.squeeze().numpy()
    prediction_np = prediction.squeeze().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.suptitle(f'Epoch {epoch} | Validation MAE Loss: {loss:.5f}', fontsize=14, y=0.98)

    # Determine shared color range using percentiles for robustness
    vmin = np.percentile(target_np, 1)
    vmax = np.percentile(target_np, 99)

    # Plot Ground Truth
    im1 = axes[0].imshow(target_np, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    axes[0].set_title('Ground Truth Velocity')
    axes[0].set_xlabel('Width Index')
    axes[0].set_ylabel('Depth/Time Index')
    fig.colorbar(im1, ax=axes[0], label='Velocity (units)', fraction=0.046, pad=0.04) # Adjust unit label if known

    # Plot Prediction
    im2 = axes[1].imshow(prediction_np, cmap='viridis', vmin=vmin, vmax=vmax, aspect='auto')
    axes[1].set_title('Predicted Velocity')
    axes[1].set_xlabel('Width Index')
    axes[1].set_ylabel('Depth/Time Index')
    fig.colorbar(im2, ax=axes[1], label='Velocity (units)', fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    fig_path = os.path.join(cfg.OUTPUT_DIR, f'validation_epoch_{epoch}.png')
    try:
        plt.savefig(fig_path, dpi=150)
        # print(f"Saved validation visualization: {fig_path}") # Reduce verbose printing
    except Exception as e:
        print(f"Warning: Error saving visualization - {e}")
    plt.show() # Display plot in notebook context

def plot_history(history, cfg):
    """Plots training/validation loss and learning rate curves."""
    epochs = range(1, len(history['train_loss']) + 1)
    if not epochs:
        print("No history data to plot.")
        return

    fig, ax1 = plt.subplots(figsize=(12, 5))

    # Plot Losses
    color = 'tab:blue'
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('MAE Loss', color=color, fontsize=12)
    ax1.plot(epochs, history['train_loss'], color=color, linestyle='-', marker='o', markersize=4, label='Train Loss')
    ax1.plot(epochs, history['valid_loss'], color='tab:orange', linestyle='--', marker='x', markersize=4, label='Validation Loss')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=10)
    ax1.tick_params(axis='x', labelsize=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot Learning Rate on secondary y-axis
    ax2 = ax1.twinx()
    color = 'tab:green'
    ax2.set_ylabel('Learning Rate', color=color, fontsize=12)
    ax2.plot(epochs, history['lr'], color=color, linestyle=':', marker='s', markersize=4, label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor=color, labelsize=10)
    ax2.ticklabel_format(style='sci', axis='y', scilimits=(-5, 4))
    ax2.legend(loc='upper right', fontsize=10)

    plt.title('Training History: Loss & Learning Rate', fontsize=14)
    fig.tight_layout()

    history_plot_path = os.path.join(cfg.OUTPUT_DIR, 'training_history.png')
    try:
        plt.savefig(history_plot_path, dpi=150)
        print(f"ðŸ“Š Saved training history plot: {history_plot_path}")
    except Exception as e:
        print(f"Warning: Error saving history plot - {e}")
    plt.show()


def run_training(model, train_loader, valid_loader, cfg):
    """Coordinates the model training process."""
    print(f"ðŸš€ Starting training run...")
    model.to(cfg.DEVICE)
    criterion = nn.L1Loss() # MAE Loss
    optimizer = optim.AdamW(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=cfg.LR_SCHEDULER_FACTOR, patience=cfg.LR_SCHEDULER_PATIENCE, verbose=True)
    scaler = torch.cuda.amp.GradScaler() if cfg.DEVICE == torch.device('cuda') else None
    if scaler: print("AMP enabled.")

    history = {'train_loss': [], 'valid_loss': [], 'lr': []}
    best_valid_loss = float('inf')
    best_model_state = None
    epochs_no_improve = 0
    start_time = time.time()

    for epoch in range(1, cfg.N_EPOCHS + 1):
        epoch_start_time = time.time()
        print(f"\n===== Epoch {epoch}/{cfg.N_EPOCHS} =====")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg.DEVICE, scaler, cfg)
        valid_loss, sample_output, sample_target = validate_one_epoch(model, valid_loader, criterion, cfg.DEVICE, cfg)

        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        scheduler.step(valid_loss)

        epoch_duration = time.time() - epoch_start_time
        print(f"Epoch {epoch} Summary | Train Loss: {train_loss:.5f} | Valid Loss: {valid_loss:.5f} (Best: {best_valid_loss:.5f}) | LR: {current_lr:.1e} | Time: {epoch_duration:.2f}s")

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            best_model_state = model.state_dict().copy()
            epochs_no_improve = 0
            print(f"âœ¨ Validation Loss Improved! Saving best model state.")
            # Save immediately in case of interruption
            try:
                 torch.save(best_model_state, cfg.BEST_MODEL_PATH)
                 # print(f"Best model state saved to {cfg.BEST_MODEL_PATH}") # Less verbose
            except Exception as e:
                 print(f"Warning: Error saving best model state - {e}")
        else:
            epochs_no_improve += 1
            print(f"Patience: {epochs_no_improve}/{cfg.EARLY_STOPPING_PATIENCE}")

        if epoch % cfg.VISUALIZATION_INTERVAL == 0 or epoch == 1:
            visualize_prediction(sample_target, sample_output, epoch, valid_loss, cfg)

        if epochs_no_improve >= cfg.EARLY_STOPPING_PATIENCE:
            print(f"\nðŸ›‘ Early stopping triggered after {epoch} epochs.")
            break

        gc.collect()
        if cfg.DEVICE == torch.device('cuda'):
            torch.cuda.empty_cache()

    total_training_time = time.time() - start_time
    print(f"\n===== Training Finished =====")
    print(f"Total Training Time: {total_training_time / 60:.2f} minutes")
    print(f"Best Validation MAE: {best_valid_loss:.5f}")

    if best_model_state:
        print("Loading best model weights achieved during training...")
        model.load_state_dict(best_model_state)
        # Final save already happened when loss improved
    else:
        print("Warning: No improvement observed or training too short. Using final model state.")

    plot_history(history, cfg)
    return history, model # Return history and model with best weights

print("Training and validation helper functions defined.")

# ======================================================
# 7. Prediction and Submission Functions
# ======================================================
class TestDataset(Dataset):
    """Dataset for loading test data files for inference."""
    def __init__(self, test_files):
        self.test_files = sorted(test_files)
        if not self.test_files: print("Warning: TestDataset initialized with zero files.")

    def __len__(self):
        return len(self.test_files)

    def __getitem__(self, i):
        if i >= len(self.test_files): raise IndexError("Test dataset index out of range.")
        test_file_path = self.test_files[i]
        try:
            data = np.load(test_file_path).astype(np.float32)
            oid = test_file_path.stem # Filename without extension is the ID
            return torch.from_numpy(data), oid
        except Exception as e:
            print(f"ERROR loading test file: {test_file_path}. Exception: {e}")
            raise e

def generate_submission(model, test_files, cfg):
    """Generates predictions on test data and creates the submission file."""
    print("\n===== Generating Submission File =====")
    if not test_files:
        print("No test files found. Skipping submission.")
        return

    test_dataset = TestDataset(test_files)
    inference_batch_size = max(1, cfg.BATCH_SIZE // 2) # Use potentially smaller batch for inference
    test_loader = DataLoader(test_dataset, batch_size=inference_batch_size, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    print(f"Created Test DataLoader: Batches={len(test_loader)}, Batch Size={inference_batch_size}")

    # Required columns for submission (oid_ypos + odd x indices)
    x_cols = [f'x_{i}' for i in range(1, cfg.IMG_WIDTH, 2)]
    fieldnames = ['oid_ypos'] + x_cols

    model.eval()
    model.to(cfg.DEVICE)
    results = []
    progress_bar = tqdm(test_loader, desc='Predicting', dynamic_ncols=True)
    inference_start_time = time.time()

    with torch.inference_mode():
        for inputs_batch, oids_batch in progress_bar:
            inputs_batch = inputs_batch.to(cfg.DEVICE, non_blocking=True).float()
            current_batch_size = inputs_batch.shape[0]

            # --- Test-Time Augmentation (TTA) ---
            if cfg.USE_TTA:
                tta_predictions = []
                # Original
                with torch.cuda.amp.autocast(enabled=(cfg.DEVICE == torch.device('cuda'))):
                    outputs_original = model(inputs_batch)
                tta_predictions.append(outputs_original)

                # Flipped
                inputs_flipped = torch.flip(inputs_batch, dims=[3]).clone()
                with torch.cuda.amp.autocast(enabled=(cfg.DEVICE == torch.device('cuda'))):
                    outputs_flipped = model(inputs_flipped)
                outputs_flipped_restored = torch.flip(outputs_flipped, dims=[3]).clone()
                tta_predictions.append(outputs_flipped_restored)

                # Average TTA predictions
                final_outputs = torch.mean(torch.stack(tta_predictions), dim=0)
                del outputs_original, inputs_flipped, outputs_flipped, outputs_flipped_restored, tta_predictions # Cleanup
            else: # No TTA
                 with torch.cuda.amp.autocast(enabled=(cfg.DEVICE == torch.device('cuda'))):
                      final_outputs = model(inputs_batch)

            # --- Process Outputs ---
            y_preds_batch_np = final_outputs.squeeze(1).cpu().numpy() # (Batch, H, W)

            for i in range(current_batch_size):
                oid = oids_batch[i]
                y_pred_single_np = y_preds_batch_np[i] # (H, W)
                for y_pos in range(cfg.IMG_HEIGHT):
                    oid_ypos = f"{oid}_y_{y_pos}"
                    odd_x_values = y_pred_single_np[y_pos, 1::2] # Slice to get odd columns (1, 3, 5...)
                    row_data = {'oid_ypos': oid_ypos}
                    row_data.update(dict(zip(x_cols, odd_x_values)))
                    results.append(row_data)

            del inputs_batch, final_outputs, y_preds_batch_np
            if cfg.DEVICE == torch.device('cuda'): torch.cuda.empty_cache()

    inference_duration = time.time() - inference_start_time
    print(f"Inference finished in {inference_duration:.2f} seconds ({len(results)} rows generated).")

    # --- Write CSV ---
    if not results:
        print("Warning: No results were generated.")
        return
    print(f"Writing {len(results)} rows to submission file: {cfg.SUBMISSION_FILE}")
    submission_df = pd.DataFrame(results)
    submission_df = submission_df[['oid_ypos'] + x_cols] # Ensure correct column order
    try:
        submission_df.to_csv(cfg.SUBMISSION_FILE, index=False, float_format='%.4f') # Format floats
        print(f"âœ… Submission file created successfully!")
    except Exception as e:
        print(f"Error writing submission CSV: {e}")

print("Prediction and submission functions defined.")


# ======================================================
# 8. Main Execution Block
# ======================================================
def main():
    """Main function to orchestrate the entire pipeline."""
    print("\n" + "="*40)
    print("===== FWI MLP Baseline Pipeline Start =====")
    print("="*40 + "\n")

    # --- Print Key Config ---
    print("--- Configuration Summary ---")
    print(f"Device: {CFG.DEVICE}, Seed: {CFG.SEED}")
    print(f"Epochs: {CFG.N_EPOCHS}, Batch Size: {CFG.BATCH_SIZE}, LR: {CFG.LEARNING_RATE}")
    print(f"Output Dir: {CFG.OUTPUT_DIR}")
    print(f"Augmentation: {CFG.AUGMENT_TRAIN}, TTA: {CFG.USE_TTA}\n")

    # Create output directory if it doesn't exist
    Path(CFG.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # --- Data Preparation ---
    print("--- 1. Preparing Data ---")
    try:
        all_input_files = find_data_files(CFG.TRAIN_DIR)
        if not all_input_files: raise FileNotFoundError("No training input files found.")
        all_input_files, all_output_files = map_inputs_to_outputs(all_input_files)
        if not all_input_files: raise ValueError("No valid input/output pairs found.")
        train_inputs, train_outputs, valid_inputs, valid_outputs = split_train_validation(
            all_input_files, all_output_files, CFG.VALIDATION_SPLIT_RATIO
        )
        # Create Datasets
        train_dataset = SeismicDataset(train_inputs, train_outputs, augment=CFG.AUGMENT_TRAIN, cfg=CFG)
        valid_dataset = SeismicDataset(valid_inputs, valid_outputs, augment=False, cfg=CFG)
        # Create DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
            num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True,
            persistent_workers=(CFG.NUM_WORKERS > 0)
        )
        valid_loader = DataLoader(
            valid_dataset, batch_size=max(1, CFG.BATCH_SIZE * 2), shuffle=False, # Often larger valid BS possible
            num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=False,
            persistent_workers=(CFG.NUM_WORKERS > 0)
        )
        print(f"DataLoaders ready: Train batches={len(train_loader)}, Valid batches={len(valid_loader)}")
    except Exception as e:
        print(f"FATAL ERROR during Data Preparation: {e}")
        return # Stop execution if data fails

    # --- Model Initialization ---
    print("\n--- 2. Initializing Model ---")
    try:
        model = FWINet(cfg=CFG)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model '{model.__class__.__name__}' initialized with {num_params:,} trainable parameters.")
    except Exception as e:
        print(f"FATAL ERROR during Model Initialization: {e}")
        return # Stop execution if model fails

    # --- Training ---
    print("\n--- 3. Starting Training ---")
    try:
        history, trained_model = run_training(model, train_loader, valid_loader, cfg=CFG)
        # 'trained_model' holds the model with the best weights loaded
        print("Training complete.")
    except Exception as e:
        print(f"FATAL ERROR during Training: {e}")
        # Optionally try to generate submission with model state before error? For now, stop.
        return

    # --- Prediction & Submission ---
    print("\n--- 4. Generating Submission ---")
    try:
        test_files = list(Path(CFG.TEST_DIR).glob('*.npy'))
        if not test_files:
            print("No test files found. Submission cannot be generated.")
        else:
            print(f"Found {len(test_files)} test files in {CFG.TEST_DIR}")
            # Make sure the best model state is actually in the model instance
            # It should be loaded by run_training if successful.
            generate_submission(trained_model, test_files, cfg=CFG)
    except Exception as e:
        print(f"ERROR during Submission Generation: {e}")
        # Training might have finished, but submission failed.

    print("\n" + "="*40)
    print("===== FWI MLP Baseline Pipeline End =====")
    print("="*40 + "\n")


# --- Entry Point ---
if __name__ == "__main__":
    # Record overall script execution time
    script_start_time = time.time()
    main() # Run the main pipeline
    script_end_time = time.time()
    total_duration = script_end_time - script_start_time
    print(f"\nTotal script execution time: {total_duration / 60:.2f} minutes ({total_duration:.2f} seconds).")

