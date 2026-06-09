# Cell 1: System Resource Check and Configuration Validation
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Check Python version and critical libraries
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

# Check system resources
print("\n=== System Resources ===")

# Check CPU
try:
    import psutil
    cpu_count = psutil.cpu_count(logical=True)
    cpu_physical = psutil.cpu_count(logical=False)
    memory_gb = psutil.virtual_memory().total / (1024**3)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    print(f"CPU: {cpu_physical} physical cores, {cpu_count} logical cores")
    print(f"Memory: {memory_gb:.1f} GB total, {available_memory_gb:.1f} GB available")
    
    # Calculate optimal worker count for DataLoader
    optimal_workers = min(16, max(4, cpu_count - 2))  # Cap at 16, but at least 4
    print(f"Recommended num_workers for DataLoader: {optimal_workers}")
except ImportError:
    cpu_count = os.cpu_count()
    print(f"CPU: {cpu_count} cores detected")
    optimal_workers = min(16, max(4, cpu_count - 2))
    print(f"Recommended num_workers for DataLoader: {optimal_workers}")

# Check GPU
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    print(f"GPU: {device_count} device(s) available")
    for i in range(device_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
        print(f"  - GPU {i}: {gpu_name}, {gpu_memory:.1f} GB memory")
    
    # Test GPU speed (optional)
    print("\nGPU quick benchmark...")
    x = torch.randn(1000, 1000, device="cuda")
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    result = torch.matmul(x, x)
    end.record()
    torch.cuda.synchronize()
    print(f"Matrix multiplication time: {start.elapsed_time(end):.2f} ms")
    
    # Clear memory
    del x, result, start, end
    torch.cuda.empty_cache()
    
    # GPU is available, so provide optimal settings
    print("\n=== Recommended GPU Settings ===")
    print("- mixed_precision: True")
    print("- pin_memory: True")
    print("- persistent_workers: True")
    print(f"- prefetch_factor: {min(4, optimal_workers // 2)}")
else:
    print("GPU: Not available, using CPU only")
    print("\nWARNING: Training will be much slower without GPU acceleration")

# Check where we are
print("\n=== Current Directory ===")
print(f"Current path: {os.getcwd()}")
print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set')}")

print("\n=== Configuration Checklist ===")
checks = [
    "Set num_workers to optimal value",
    "Enable mixed_precision=True for GPU training",
    "Use 200 epochs for full training",
    "Increase early_stopping patience for long runs",
    "Update post_process_binary function for dimension handling", 
    "Enable pin_memory and persistent_workers in DataLoader",
]

for i, check in enumerate(checks, 1):
    print(f"{i}. [ ] {check}")

print("\nReady to begin 200-epoch training. Make sure all checkboxes are addressed.")
print("Estimated training time: ~6-7 hours with GPU acceleration and optimized settings.")


# Cell 2: Import Libraries and Major Configuration Block
import os
import time
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

# Set random seeds for reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==== MASTER CONFIGURATION BLOCK ====
CONFIG = {
    # Runtime configuration
    'approach': 'physics_guided',  # Options: 'thresholding', 'physics_guided', 'feature_detection', 'geo-aware'
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'mixed_precision': True,  # Use mixed precision training for GPU acceleration
    'save_models': True,  # Save models during training
    'compare_approaches': False,  # Run multiple approaches and compare
    'generate_submission': False,  # Toggle submission file generation
    'run_analysis_tools': True,
    
    'in_channels': 5,
    'out_channels': 1,
    'hidden_dim': 64,
   
    # Data configuration 
    'data_dir': Path("/kaggle/input/waveform-inversion/train_samples"),
    'test_dir': Path("/kaggle/input/waveform-inversion/test"),
    'output_dir': Path("./outputs"),
    
    # Dataset parameters
    'val_size': 0.15,  # Validation set size
    'batch_size': 4,  # GPU-friendly batch size
    'num_workers': 4,  # DataLoader workers
    
    # Training parameters
    'num_epochs': 50,
    'learning_rate': 5e-5, # Careful with numeric instability
    'weight_decay': 1e-5,
    'early_stopping': 10,
    'scheduler_patience': 5,
    'prefetch_factor': 2,     # Controls batch prefetching per worker
    
    'grad_clip_value': 1.0, # NaN issues earlier
    'warmup_epochs': 5, # NaN issues earlier

    
    # Model parameters
    'in_channels': 5,
    'out_channels': 1,
    'hidden_dim': 64,  # Base dimensionality for models
    
    # Physics-guided parameters
    'wave_eq_weight': 0.15,
    'slowness_weight': 0.1,
    'layering_weight': 0.05,
    'contrast_weight': 0.2,
    
    # Feature detection parameters; will revist as these may be too small
    'salt_weight': 0.2,
    'fault_weight': 0.1,
    'layer_weight': 0.15,
    'geological_constraint_weight': 0.3,
    
    # Thresholding parameters
    'threshold_method': 'otsu',  # Options: 'otsu', 'mean', 'adaptive'
    'edge_enhancement': 1.5,
    'use_morphology': True,
    
    # Submission parameters
    'ensemble_submission': False,  # Use ensemble of multiple models
    'post_process': True,  # Apply post-processing to predictions
    'submission_path': "submission.csv",
}

# Create output directory
os.makedirs(CONFIG['output_dir'], exist_ok=True)
os.makedirs(CONFIG['output_dir'] / 'models', exist_ok=True)
os.makedirs(CONFIG['output_dir'] / 'visualizations', exist_ok=True)

# Print configuration summary
print(f"Running with approach: {CONFIG['approach']}")
print(f"Device: {CONFIG['device']}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Mixed precision: {CONFIG['mixed_precision']}")

# Initialize experiment name with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"{CONFIG['approach']}_{timestamp}"
print(f"Experiment name: {experiment_name}")

# Save configuration
config_path = CONFIG['output_dir'] / f"config_{experiment_name}.txt"
with open(config_path, 'w') as f:
    for key, value in CONFIG.items():
        f.write(f"{key}: {value}\n")

CONFIG['confirmed'] = True
CONFIG['experiment_name'] = experiment_name




# Cell 3: Data Loading and Dataset Classes
class SeismicDataset(Dataset):
    """Dataset class for seismic data with flexible input/output handling"""
    def __init__(self, input_files, output_files=None, transform=None, 
                 normalize=True, gain=True, augment=False, 
                 binary_threshold=None, feature_mode=False):
        """
        Args:
            input_files: List of seismic data file paths
            output_files: List of velocity map file paths (None for test set)
            transform: Optional transforms to apply
            normalize: Whether to normalize data
            gain: Whether to apply time-dependent gain
            augment: Whether to apply data augmentation
            binary_threshold: If not None, convert velocity maps to binary using this threshold
            feature_mode: If True, extract geological features from velocity maps
        """
        self.input_files = input_files
        self.output_files = output_files
        self.transform = transform
        self.normalize = normalize
        self.gain = gain
        self.augment = augment
        self.binary_threshold = binary_threshold
        self.feature_mode = feature_mode
        
        # Build index map for efficient access
        self.index_map = []
        for i, f in enumerate(self.input_files):
            try:
                data = np.load(f, mmap_mode='r')
                for j in range(data.shape[0]):
                    self.index_map.append((i, j))
            except Exception as e:
                print(f"Error loading {f}: {e}")
        
        # Calculate statistics if normalizing
        if normalize and output_files:
            self.calc_stats()
    
    def calc_stats(self, max_files=10, max_samples=100):
        """Calculate dataset statistics for normalization"""
        print("Calculating dataset statistics...")
        # Input statistics
        in_samples = []
        for i in range(min(max_files, len(self.input_files))):
            try:
                data = np.load(self.input_files[i], mmap_mode='r')
                idx = np.random.choice(data.shape[0], min(max_samples, data.shape[0]), replace=False)
                in_samples.append(data[idx])
            except Exception as e:
                print(f"Error in stats calculation for {self.input_files[i]}: {e}")
        
        if in_samples:
            in_array = np.concatenate(in_samples, axis=0)
            self.in_mean = float(np.mean(in_array))
            self.in_std = float(np.std(in_array))
        else:
            self.in_mean, self.in_std = 0.0, 1.0
            
        # Output statistics (if available)
        if self.output_files:
            out_samples = []
            for i in range(min(max_files, len(self.output_files))):
                try:
                    data = np.load(self.output_files[i], mmap_mode='r')
                    idx = np.random.choice(data.shape[0], min(max_samples, data.shape[0]), replace=False)
                    
                    # Handle different output shapes
                    if len(data.shape) == 4:  # [batch, channel, height, width]
                        out_samples.append(data[idx])
                    elif len(data.shape) == 3:  # [batch, height, width]
                        out_samples.append(data[idx, np.newaxis])
                except Exception as e:
                    print(f"Error in stats calculation for {self.output_files[i]}: {e}")
            
            if out_samples:
                out_array = np.concatenate(out_samples, axis=0)
                self.out_mean = float(np.mean(out_array))
                self.out_std = float(np.std(out_array))
                
                # For binary threshold detection
                if self.binary_threshold is None:
                    try:
                        from skimage.filters import threshold_otsu
                        self.auto_threshold = threshold_otsu(out_array.flatten())
                    except:
                        self.auto_threshold = self.out_mean
                else:
                    self.auto_threshold = self.binary_threshold
            else:
                self.out_mean, self.out_std = 0.0, 1.0
                self.auto_threshold = 0.5
                
        print(f"Input stats: mean={self.in_mean:.4f}, std={self.in_std:.4f}")
        if self.output_files:
            print(f"Output stats: mean={self.out_mean:.4f}, std={self.out_std:.4f}")
            print(f"Auto threshold: {self.auto_threshold:.4f}")
    
    def __len__(self):
        return len(self.index_map)
    
    def apply_gain(self, x):
        """Apply time-dependent gain to seismic data"""
        # Time-dependent gain increases amplitude with time
        time_steps = x.shape[1]
        time = np.linspace(0, 1, time_steps)
        gain = (time ** 2)[:, np.newaxis]  # Square gain with time
        
        # Apply to each channel
        gained_x = x.copy()
        for c in range(x.shape[0]):
            gained_x[c] = x[c] * gain
            
        return gained_x
    
    def extract_geological_features(self, y):
        """Extract geological features from velocity map"""
        # 1. Salt body detection (high velocity regions)
        salt_mask = (y > self.auto_threshold + 0.2 * self.out_std).astype(np.float32)
        
        # 2. Fault detection (using edge detection)
        from scipy import ndimage
        sobel_x = ndimage.sobel(y, axis=1)
        sobel_y = ndimage.sobel(y, axis=0)
        grad_mag = np.sqrt(sobel_x**2 + sobel_y**2)
        grad_mag = grad_mag / (np.max(grad_mag) + 1e-8)
        fault_mask = (grad_mag > 0.3).astype(np.float32)
        
        # 3. Layer detection (horizontal structures)
        # Use horizontal gradient
        layer_edges = np.abs(sobel_y)
        layer_edges = layer_edges / (np.max(layer_edges) + 1e-8)
        layer_mask = (layer_edges > 0.2).astype(np.float32)
        
        # Stack into feature channels [salt, fault, layer]
        features = np.stack([salt_mask, fault_mask, layer_mask], axis=0)
        return features
    
    def apply_augmentation(self, x, y=None):
        """Apply data augmentation"""
        # Flip horizontally with 50% probability
        if np.random.random() > 0.5:
            x = np.flip(x, axis=2).copy()  # Flip along receiver dimension
            if y is not None:
                y = np.flip(y, axis=-1).copy()  # Flip along width
        
        # Add small random noise with 30% probability
        if np.random.random() > 0.7:
            noise_level = np.random.uniform(0, 0.05)
            x = x + np.random.normal(0, noise_level, size=x.shape)
        
        return x, y
        
    def __getitem__(self, idx):
        """Get a single sample from the dataset"""
        file_idx, sample_idx = self.index_map[idx]
        
        # Load input data
        x = np.load(self.input_files[file_idx], mmap_mode='r')[sample_idx].astype(np.float32)
        
        # Apply gain if requested
        if self.gain:
            x = self.apply_gain(x)
            
        # Load output data if available (for training)
        if self.output_files:
            y_data = np.load(self.output_files[file_idx], mmap_mode='r')[sample_idx]
            
            # Handle different output shapes
            if len(y_data.shape) == 2:  # (height, width)
                y = y_data
            elif len(y_data.shape) == 3 and y_data.shape[0] == 1:  # (1, height, width)
                y = y_data[0]
            elif len(y_data.shape) > 2:  # Multi-dimensional
                y = y_data.reshape(-1, y_data.shape[-2], y_data.shape[-1])[0]
            else:
                raise ValueError(f"Unexpected velocity map shape: {y_data.shape}")
                
            y = y.astype(np.float32)
            
            # Convert to binary if threshold is provided
            if self.binary_threshold is not None:
                threshold = self.binary_threshold if self.binary_threshold > 0 else self.auto_threshold
                y = (y > threshold).astype(np.float32)
                
            # Extract geological features if in feature mode
            if self.feature_mode:
                y = self.extract_geological_features(y)
        else:
            # For test set, create a dummy y
            y = np.zeros((1, x.shape[-1], x.shape[-1]), dtype=np.float32)
            
        # Apply data augmentation if enabled
        if self.augment:
            x, y = self.apply_augmentation(x, y)
            
        # Normalize if enabled
        if self.normalize:
            x = (x - self.in_mean) / (self.in_std + 1e-6)
            if self.output_files and not self.feature_mode and self.binary_threshold is None:
                y = (y - self.out_mean) / (self.out_std + 1e-6)
                
        # Convert to PyTorch tensors
        x_tensor = torch.from_numpy(x).float()
        
        # Ensure y has proper dimensions for loss functions
        if not self.feature_mode and len(y.shape) == 2:
            y = y[np.newaxis, ...]  # Add channel dimension if needed
        y_tensor = torch.from_numpy(y).float()
        
        # Apply transforms if provided
        if self.transform:
            x_tensor, y_tensor = self.transform(x_tensor, y_tensor)
            
        return x_tensor, y_tensor


def load_and_prepare_data(config):
    """Prepare datasets and dataloaders based on configuration"""
    data_dir = config['data_dir']
    
    # Find input and output files
    print("Loading competition dataset...")
    input_files = sorted([f for f in data_dir.rglob("*.npy") 
                         if 'seis' in f.name or 'data' in f.name])
    output_files = [Path(str(f).replace("seis", "vel").replace("data", "model")) 
                    for f in input_files]
    
    print(f"Found {len(input_files)} input files and {len(output_files)} output files")
    
    # Verify files exist
    input_files = [f for f in input_files if f.exists()]
    output_files = [f for f in output_files if f.exists()]
    if len(input_files) != len(output_files):
        print(f"Warning: Mismatched file counts - {len(input_files)} inputs, {len(output_files)} outputs")
        # Keep only matching pairs
        input_basenames = [f.stem for f in input_files]
        output_basenames = [f.stem.replace("vel", "seis").replace("model", "data") for f in output_files]
        common_basenames = set(input_basenames) & set(output_basenames)
        input_files = [f for f in input_files if f.stem in common_basenames]
        output_files = [f for f in output_files if f.stem.replace("vel", "seis").replace("model", "data") in common_basenames]
        print(f"Kept {len(input_files)} matching file pairs")
    
    # Split into train and validation sets
    train_in, val_in, train_out, val_out = train_test_split(
        input_files, output_files, test_size=config['val_size'], random_state=SEED
    )
    
    print(f"Training set: {len(train_in)} files")
    print(f"Validation set: {len(val_in)} files")
    
    # Determine binary threshold and feature mode based on approach
    binary_threshold = None
    feature_mode = False
    
    if config['approach'] == 'thresholding':
        binary_threshold = 0  # Use auto-detected threshold
    elif config['approach'] == 'feature_detection':
        feature_mode = True
        
    # Create datasets
    train_ds = SeismicDataset(
        train_in, train_out,
        normalize=True,
        gain=True,
        augment=True,
        binary_threshold=binary_threshold,
        feature_mode=feature_mode
    )
    
    val_ds = SeismicDataset(
        val_in, val_out,
        normalize=True,
        gain=True,
        augment=False,
        binary_threshold=binary_threshold,
        feature_mode=feature_mode
    )
    
# Optimized DataLoader creation
    train_loader = DataLoader(
        train_ds,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config['num_workers'] > 0,
        prefetch_factor=config.get('prefetch_factor', 2)
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config['num_workers'] > 0,
        prefetch_factor=config.get('prefetch_factor', 2)
    )
                
    # Get dataset statistics for normalization and thresholding
    stats = {
        'input_mean': train_ds.in_mean,
        'input_std': train_ds.in_std,
        'output_mean': train_ds.out_mean if hasattr(train_ds, 'out_mean') else 0.0,
        'output_std': train_ds.out_std if hasattr(train_ds, 'out_std') else 1.0,
        'auto_threshold': train_ds.auto_threshold if hasattr(train_ds, 'auto_threshold') else 0.5
    }
    
    return train_loader, val_loader, stats


# Cell 4: Model Architectures
# 1. Physics-Guided Model for Velocity Prediction
class PhysicsGuidedUNet(nn.Module):
    """U-Net with physics-guided components for velocity prediction"""
    def __init__(self, in_channels=5, out_channels=1, hidden_dim=64):
        super().__init__()
        
        # Encoder blocks
        self.enc1 = self._make_encoder_block(in_channels, hidden_dim)
        self.enc2 = self._make_encoder_block(hidden_dim, hidden_dim*2)
        self.enc3 = self._make_encoder_block(hidden_dim*2, hidden_dim*4)
        self.enc4 = self._make_encoder_block(hidden_dim*4, hidden_dim*8)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(hidden_dim*8, hidden_dim*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*16),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dim*16, hidden_dim*16, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*16),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Decoder blocks with skip connections
        self.dec4 = self._make_decoder_block(hidden_dim*16 + hidden_dim*8, hidden_dim*8)
        self.dec3 = self._make_decoder_block(hidden_dim*8 + hidden_dim*4, hidden_dim*4)
        self.dec2 = self._make_decoder_block(hidden_dim*4 + hidden_dim*2, hidden_dim*2)
        self.dec1 = self._make_decoder_block(hidden_dim*2 + hidden_dim, hidden_dim)
        
        # Final output layer
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1),
            nn.Sigmoid()  # Output in [0,1] range
        )
        
        # Pooling layer for downsampling
        self.pool = nn.MaxPool2d(2)
        
        # Physics-guided layers
        # Edge detection for geological boundaries
        self.edge_detector = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.edge_detector.weight.data = torch.tensor([
            [[-1, -1, -1],
             [2, 2, 2],
             [-1, -1, -1]]
        ], dtype=torch.float32).view(1, 1, 3, 3).repeat(out_channels, out_channels, 1, 1)
        self.edge_detector.weight.requires_grad = False
        
    def _make_encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def _make_decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def enhance_contrast(self, x):
        """Enhance contrast in the prediction to match high-contrast ground truth"""
        # Apply tanh with scaling to enhance contrast
        enhanced = torch.tanh(5 * (x - 0.5)) * 0.5 + 0.5
        return enhanced
    
    def enhance_edges(self, x):
        """Enhance geological edges in the prediction"""
        edge_features = self.edge_detector(x)
        enhanced = x + 0.1 * edge_features
        return torch.clamp(enhanced, 0, 1)
    
    def forward(self, x):
        # Input format adjustment - seismic data comes as [B, C, T, R]
        # We'll permute to [B, C, R, T] for 2D convolutions
        x = x.permute(0, 1, 3, 2)
        
        # Encoder path with skip connections
        e1 = self.enc1(x)
        p1 = self.pool(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool(e2)
        
        e3 = self.enc3(p2)
        p3 = self.pool(e3)
        
        e4 = self.enc4(p3)
        p4 = self.pool(e4)
        
        # Bottleneck
        b = self.bottleneck(p4)
        
        # Decoder path with skip connections
        d4 = F.interpolate(b, size=e4.shape[2:], mode='bilinear', align_corners=False)
        d4 = self.dec4(torch.cat([d4, e4], dim=1))
        
        d3 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        # Final output and enhancement
        out = self.final(d1)
        out = self.enhance_contrast(out)
        out = self.enhance_edges(out)
        
        # Ensure output is 70x70 as expected
        out = F.interpolate(out, size=(70, 70), mode='bilinear', align_corners=False)
        
        return out


# 2. Geological Feature Detection Model
class GeologicalFeatureNet(nn.Module):
    """Neural network for detecting geological features from seismic data"""
    def __init__(self, in_channels=5, hidden_dim=64):
        super().__init__()
        
        # Encoder (shared for all features)
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(hidden_dim*4, hidden_dim*8, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*8),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
        # Specialized decoders for different geological features
        
        # 1. Salt body detector - MODIFIED to remove sigmoid for mixed precision compatibility
        self.salt_decoder = nn.Sequential(
            nn.Conv2d(hidden_dim*8, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dim*4, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dim//2, 1, kernel_size=1)
            # Sigmoid removed for BCEWithLogitsLoss compatibility
        )
        
        # 2. Fault detector with edge awareness - MODIFIED to remove sigmoid
        self.fault_decoder = nn.Sequential(
            nn.Conv2d(hidden_dim*8, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dim*4, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dim//2, 1, kernel_size=1)
            # Sigmoid removed for BCEWithLogitsLoss compatibility
        )
        
        # 3. Layer boundary detector with horizontal bias - MODIFIED to remove sigmoid
        self.layer_decoder = nn.Sequential(
            nn.Conv2d(hidden_dim*8, hidden_dim*4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dim*4, hidden_dim*2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dim*2, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dim, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            
            nn.Conv2d(hidden_dim//2, 1, kernel_size=1)
            # Sigmoid removed for BCEWithLogitsLoss compatibility
        )
        
        # Optional velocity reconstruction from geological features - Keep sigmoid here
        self.velocity_reconstruction = nn.Sequential(
            nn.Conv2d(3, hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim//2),
            nn.ReLU(),
            
            nn.Conv2d(hidden_dim//2, hidden_dim//4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_dim//4),
            nn.ReLU(),
            
            nn.Conv2d(hidden_dim//4, 1, kernel_size=1),
            nn.Sigmoid()  # Keep sigmoid here as this is not used with BCE loss
        )
    
    def forward(self, x):
        # Input format adjustment
        x = x.permute(0, 1, 3, 2)  # [B, C, T, R] to [B, C, R, T]
        
        # Encode features
        encoded = self.encoder(x)
        
        # Decode for each geological feature
        salt_mask = self.salt_decoder(encoded)
        fault_lines = self.fault_decoder(encoded)
        layer_boundaries = self.layer_decoder(encoded)
        
        # Ensure all outputs are 70x70
        salt_mask = F.interpolate(salt_mask, size=(70, 70), mode='bilinear', align_corners=False)
        fault_lines = F.interpolate(fault_lines, size=(70, 70), mode='bilinear', align_corners=False)
        layer_boundaries = F.interpolate(layer_boundaries, size=(70, 70), mode='bilinear', align_corners=False)
        
        # Apply sigmoid for the velocity reconstruction input
        salt_prob = torch.sigmoid(salt_mask)
        fault_prob = torch.sigmoid(fault_lines)
        layer_prob = torch.sigmoid(layer_boundaries)
        
        # Combine features for velocity reconstruction 
        features = torch.cat([salt_prob, fault_prob, layer_prob], dim=1)
        velocity = self.velocity_reconstruction(features)
        
        return {
            'salt': salt_mask,         # Now returns logits
            'faults': fault_lines,     # Now returns logits
            'layers': layer_boundaries, # Now returns logits
            'velocity': velocity       # Still returns probabilities
        }


def create_model(config):
    """Create model based on configuration"""
    approach = config['approach']
    device = config['device']

    if approach == 'physics_guided':
        model = PhysicsGuidedUNet(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            hidden_dim=config['hidden_dim']
        )
        print("Created Physics-Guided U-Net model")

    elif approach == 'feature_detection':
        model = GeologicalFeatureNet(
            in_channels=config['in_channels'],
            hidden_dim=config['hidden_dim']
        )
        print("Created Geological Feature Detection model")

    elif approach == 'geo_aware':
        model = GeoAwareConvNet(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            hidden_dim=config['hidden_dim']
        )
        print("Created Geo-Aware ConvNet model")

    else:
        model = PhysicsGuidedUNet(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            hidden_dim=config['hidden_dim']
        )
        print(f"Unknown approach '{approach}', defaulting to Physics-Guided model")

    model = model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} parameters ({trainable_params:,} trainable)")
    
    return model

    
    # Move model to device
    model = model.to(device)
    
    # Print model summary
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params:,} parameters ({trainable_params:,} trainable)")
    
    return model


# Cell 5: GeoAwareConvNet Architecture (2D-Aware)
# {"_collapsed": false}

class GeoAwareConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_standard = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv_dilated = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=(0, 2), dilation=(1, 2))
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv_standard(x)
        x = self.conv_dilated(x)
        x = self.bn(x)
        return self.act(x)


class GeoAwareConvNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            GeoAwareConvBlock(in_channels, hidden_dim),
            GeoAwareConvBlock(hidden_dim, hidden_dim),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        # Resize output to match label dimensions
        x = F.interpolate(x, size=(70, 70), mode='bilinear', align_corners=False)
        # Optionally clamp if you're outputting probabilities
        # x = torch.clamp(x, 0.0, 1.0)
        return x




# Cell 5: Geological Discriminator Network
# {"_collapsed": true}

class GeoDiscriminator(nn.Module):
    """Small CNN to assess geological plausibility of velocity models"""
    def __init__(self, in_channels=1, hidden_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim, hidden_dim*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*2),
            nn.LeakyReLU(0.2),

            nn.Conv2d(hidden_dim*2, hidden_dim*4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(hidden_dim*4),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim*4, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


def train_discriminator(discriminator, dataloader, device, epochs=5):
    disc = discriminator.to(device)
    optimizer = torch.optim.Adam(disc.parameters(), lr=1e-4)
    loss_fn = nn.BCELoss()

    disc.train()
    for epoch in range(epochs):
        total_loss = 0
        for _, targets in dataloader:
            targets = targets.to(device)
            preds = disc(targets)
            labels = torch.ones_like(preds)
            loss = loss_fn(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1} | Discriminator loss: {total_loss/len(dataloader):.4f}")



# Cell 6: Loss Functions (Fixed for all tedious Mixed Precision Issues)
class PhysicsGuidedLoss(nn.Module):
    """Loss function for physics-guided velocity prediction with improved numerical stability"""
    def __init__(self, wave_eq_weight=0.05, slowness_weight=0.1, 
                 layering_weight=0.05, contrast_weight=0.2):
        super().__init__()
        self.wave_eq_weight = wave_eq_weight
        self.slowness_weight = slowness_weight
        self.layering_weight = layering_weight
        self.contrast_weight = contrast_weight
        
        # Edge detection kernels
        self.sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    def _prepare_kernels(self, device):
        """Move kernels to the correct device and data type"""
        dtype = torch.float32  # Always use float32 for kernels
        if self.sobel_x.device != device:
            self.sobel_x = self.sobel_x.to(device=device, dtype=dtype)
            self.sobel_y = self.sobel_y.to(device=device, dtype=dtype)
    
    def data_fidelity_loss(self, pred, target):
        """Standard L1 loss for overall accuracy"""
        return F.l1_loss(pred, target)
    
    def wave_equation_constraint(self, pred):
        """Simplified wave equation physics constraint - FIXED"""
        device = pred.device
        dtype = pred.dtype  # Get the current dtype of pred
        self._prepare_kernels(device)
        
        # Create kernels with matching dtype
        kernel_x = torch.tensor([[[[1.0, -2.0, 1.0]]]], device=device, dtype=dtype)
        kernel_y = torch.tensor([[[[1.0], [-2.0], [1.0]]]], device=device, dtype=dtype)
        
        # Second-order spatial derivatives (Laplacian)
        laplacian_x = F.conv2d(
            F.pad(pred, (1, 1, 0, 0), mode='replicate'),
            kernel_x,
            padding=0
        )
        
        laplacian_y = F.conv2d(
            F.pad(pred, (0, 0, 1, 1), mode='replicate'),
            kernel_y,
            padding=0
        )
        
        laplacian = laplacian_x + laplacian_y
        
        # Clip to prevent extreme values and NaN
        laplacian_abs = torch.clamp(torch.abs(laplacian), 0.0, 100.0)
        
        # Prevent NaN by taking mean with eps
        return torch.mean(laplacian_abs)
    
    def slowness_gradient_constraint(self, pred):
        """Constraint to encourage smoothness in slowness (1/v) - FIXED"""
        device = pred.device
        self._prepare_kernels(device)
        
        # Add small epsilon and clamp to prevent division by zero
        eps = 1e-6
        pred_safe = torch.clamp(pred, min=eps)
        
        # Convert velocity to slowness (1/v) with safe division
        slowness = 1.0 / pred_safe
        
        # Clip slowness to prevent extreme values
        slowness = torch.clamp(slowness, max=100.0)
        
        # Calculate gradients of slowness
        grad_x = F.conv2d(slowness, self.sobel_x, padding=1)
        grad_y = F.conv2d(slowness, self.sobel_y, padding=1)
        
        # Penalize large gradients in slowness (encourage smoothness)
        # Clip gradients to avoid NaN
        grad_x_abs = torch.clamp(torch.abs(grad_x), 0.0, 100.0)
        grad_y_abs = torch.clamp(torch.abs(grad_y), 0.0, 100.0)
        
        return torch.mean(grad_x_abs) + torch.mean(grad_y_abs)
    
    def geological_layering_constraint(self, pred):
        """Encourage horizontal layering typical in geological structures - FIXED"""
        device = pred.device
        self._prepare_kernels(device)
        
        # Calculate horizontal and vertical gradients
        grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        
        # Add epsilon to denominators
        eps = 1e-6
        grad_x_abs = torch.abs(grad_x) + eps
        grad_y_abs = torch.abs(grad_y) + eps
        
        # Ratio of vertical to horizontal gradients (safe division)
        ratio = grad_y_abs / grad_x_abs
        
        # Clip ratio to prevent extreme values
        ratio = torch.clamp(ratio, 0.0, 100.0)
        
        # Use a safer exponential function with clipping
        exp_term = torch.clamp(-ratio, -20.0, 20.0)  # Clamp to avoid overflow
        return torch.mean(torch.exp(exp_term))
    
    def contrast_enhancement_loss(self, pred, target):
        """Loss to encourage high contrast similar to ground truth - FIXED"""
        # Calculate histograms safely by clipping inputs
        pred_safe = torch.clamp(pred, 0.0, 1.0)
        target_safe = torch.clamp(target, 0.0, 1.0)
        
        # Calculate histograms
        try:
            pred_hist = torch.histc(pred_safe, bins=10, min=0, max=1)
            target_hist = torch.histc(target_safe, bins=10, min=0, max=1)
            
            # Normalize histograms (with epsilon for safe division)
            eps = 1e-6
            pred_hist = pred_hist / (pred_hist.sum() + eps)
            target_hist = target_hist / (target_hist.sum() + eps)
            
            # Calculate difference between histograms (EMD approximation)
            hist_diff = F.l1_loss(
                torch.cumsum(pred_hist, dim=0),
                torch.cumsum(target_hist, dim=0)
            )
            
            # Penalize low contrast (encourage bi-modal distribution)
            # Calculate variance safely
            contrast = torch.var(pred_safe)
            contrast_target = torch.var(target_safe)
            
            return hist_diff + torch.abs(contrast - contrast_target)
        except RuntimeError:
            # Fallback if histc fails (should be rare)
            return F.l1_loss(pred, target)
    
    def forward(self, pred, target):
        try:
            # Data fidelity (supervised loss)
            l1_loss = self.data_fidelity_loss(pred, target)
            
            # Physics-guided constraints with safety
            wave_eq_loss = torch.clamp(self.wave_equation_constraint(pred), 0.0, 100.0)
            slowness_loss = torch.clamp(self.slowness_gradient_constraint(pred), 0.0, 100.0)
            layering_loss = torch.clamp(self.geological_layering_constraint(pred), 0.0, 100.0)
            contrast_loss = torch.clamp(self.contrast_enhancement_loss(pred, target), 0.0, 100.0)
            
            # Debug prints to catch any potential issues
            # print(f"L1: {l1_loss.item():.4f}, Wave: {wave_eq_loss.item():.4f}, Slow: {slowness_loss.item():.4f}, Layer: {layering_loss.item():.4f}")
            
            # Combine losses with weights
            total_loss = (
                l1_loss +
                self.wave_eq_weight * wave_eq_loss +
                self.slowness_weight * slowness_loss +
                self.layering_weight * layering_loss +
                self.contrast_weight * contrast_loss
            )
            
            # Final safety check
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print("Warning: NaN or Inf detected in loss calculation. Falling back to L1 loss.")
                return l1_loss
                
            return total_loss
            
        except Exception as e:
            # If any error occurs, fall back to L1 loss
            print(f"Error in loss calculation: {e}. Falling back to L1 loss.")
            return self.data_fidelity_loss(pred, target)


class DiceLoss(nn.Module):
    """Dice loss for better boundary detection - FIXED for tensor compatibility"""
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, pred, target):
        # Ensure tensors have the same shape and use reshape instead of view
        # This fixes the "view size is not compatible" error
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        
        # Calculate Dice coefficient
        intersection = torch.sum(pred_flat * target_flat)
        pred_sum = torch.sum(pred_flat * pred_flat)
        target_sum = torch.sum(target_flat * target_flat)
        
        dice = (2.0 * intersection + self.smooth) / (pred_sum + target_sum + self.smooth)
        
        return 1.0 - dice


class GeologicalFeatureLoss(nn.Module):
    """Loss function for geological feature detection - FIXED for mixed precision"""
    def __init__(self, salt_weight=1.0, fault_weight=1.0, layer_weight=1.0, 
                 constraint_weight=0.3):
        super().__init__()
        self.salt_weight = salt_weight
        self.fault_weight = fault_weight
        self.layer_weight = layer_weight
        self.constraint_weight = constraint_weight
        
        # Base loss functions - FIXED: using BCEWithLogitsLoss for mixed precision compatibility
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
    
    def forward(self, predictions, targets):
        """
        Args:
            predictions: dict with keys 'salt', 'faults', 'layers', 'velocity'
            targets: tensor of shape [B, 3, H, W] with channels [salt, faults, layers]
        """
        # Separate target channels
        salt_target = targets[:, 0:1]
        fault_target = targets[:, 1:2]
        layer_target = targets[:, 2:3]
        
        # Feature detection losses (combine BCE and Dice for better boundary detection)
        # Apply sigmoid for Dice loss since we're now using BCEWithLogitsLoss
        salt_loss = self.bce_loss(predictions['salt'], salt_target) + \
                   self.dice_loss(torch.sigmoid(predictions['salt']), salt_target)
        
        fault_loss = self.bce_loss(predictions['faults'], fault_target) + \
                    self.dice_loss(torch.sigmoid(predictions['faults']), fault_target)
        
        layer_loss = self.bce_loss(predictions['layers'], layer_target) + \
                    self.dice_loss(torch.sigmoid(predictions['layers']), layer_target)
        
        # Apply sigmoid to convert logits to probabilities for constraint functions
        salt_prob = torch.sigmoid(predictions['salt'])
        fault_prob = torch.sigmoid(predictions['faults'])
        layer_prob = torch.sigmoid(predictions['layers'])
        
        # Geological constraints
        # 1. Salt bodies should be continuous (salt body coherence)
        salt_coherence = self.continuity_constraint(salt_prob)
        
        # 2. Faults should be thin, continuous lines
        fault_thin = self.thinness_constraint(fault_prob)
        
        # 3. Layers should be predominantly horizontal
        layer_horizontal = self.horizontal_bias_constraint(layer_prob)
        
        # 4. Geological compatibility between features
        compatibility = self.feature_compatibility_constraint(
            salt_prob, fault_prob, layer_prob
        )
        
        # Combine all losses with weights
        total_loss = (
            self.salt_weight * salt_loss +
            self.fault_weight * fault_loss +
            self.layer_weight * layer_loss +
            self.constraint_weight * (
                salt_coherence + 
                fault_thin + 
                layer_horizontal + 
                compatibility
            )
        )
        
        return total_loss
    
    def continuity_constraint(self, pred):
        """Penalize fragmented structures"""
        # Calculate gradient magnitude
        grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        # Only penalize gradients within the predicted region
        mask = (pred > 0.5).float()
        masked_grad_x = grad_x * mask[:, :, :, :-1]
        masked_grad_y = grad_y * mask[:, :, :-1, :]
        
        return torch.mean(masked_grad_x) + torch.mean(masked_grad_y)
    
    def thinness_constraint(self, pred):
        """Encourage thin fault lines"""
        # Dilated prediction
        dilated = F.max_pool2d(pred, kernel_size=3, stride=1, padding=1)
        
        # Eroded prediction
        kernel = torch.ones(1, 1, 3, 3).to(pred.device)
        eroded = 1.0 - F.max_pool2d(1.0 - pred, kernel_size=3, stride=1, padding=1)
        
        # Difference between dilated and eroded should be large for thin structures
        thinness = torch.mean(dilated - eroded)
        
        return -thinness  # Negative to encourage thinness
    
    def horizontal_bias_constraint(self, pred):
        """Encourage predominantly horizontal layers"""
        # Vertical vs horizontal gradient ratio
        grad_x = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
        grad_y = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
        
        # Mean gradients
        mean_grad_x = torch.mean(grad_x)
        mean_grad_y = torch.mean(grad_y)
        
        # Horizontal layers have larger vertical gradients
        return mean_grad_x - mean_grad_y  # Minimize for horizontal bias
    
    def feature_compatibility_constraint(self, salt, faults, layers):
        """Enforce compatibility between different geological features - FIXED for mixed precision"""
        # 1. Salt bodies should have clear boundaries that align with layer boundaries
        salt_boundary = torch.abs(
            F.avg_pool2d(salt, kernel_size=3, stride=1, padding=1) - salt
        )
        
        # This eliminates the autocast error while maintaining the same functionality
        boundary_alignment = F.mse_loss(salt_boundary, layers)
        
        # Alternative approach using BCE:
        # with torch.cuda.amp.autocast(enabled=False):
        #     boundary_alignment = F.binary_cross_entropy(salt_boundary, layers)
        
        # 2. Faults should often terminate layers
        fault_layer_interaction = torch.mean(faults * layers)
        
        # 3. Faults rarely cut through salt bodies
        fault_salt_exclusion = torch.mean(faults * salt)
        
        return boundary_alignment + fault_layer_interaction - fault_salt_exclusion


def create_loss_function(config):
    """Create loss function based on configuration"""
    approach = config['approach']
    device = config['device']
    
    if approach == 'physics_guided':
        criterion = PhysicsGuidedLoss(
            wave_eq_weight=config['wave_eq_weight'],
            slowness_weight=config['slowness_weight'],
            layering_weight=config['layering_weight'],
            contrast_weight=config['contrast_weight']
        )
        print("Created Physics-Guided Loss")
    elif approach == 'feature_detection':
        criterion = GeologicalFeatureLoss(
            salt_weight=config['salt_weight'],
            fault_weight=config['fault_weight'],
            layer_weight=config['layer_weight'],
            constraint_weight=config['geological_constraint_weight']
        )
        print("Created Geological Feature Loss")
    elif approach == 'geo_aware':
        # You could use a combined loss function for GeoAware approach
        criterion = nn.L1Loss()  # Base loss
        print("Created Geo-Aware Loss (L1)")
    else:
        # Default to L1 loss
        criterion = nn.L1Loss()
        print(f"Unknown approach '{approach}', defaulting to L1 Loss")
    
    return criterion.to(device)


# Cell 7: Training and Evaluation Functions with Enhanced Visualizations

def train_model(model, train_loader, val_loader, criterion, config):
    """Train model with validation and early stopping - FIXED for PyTorch 2.0+ compatibility"""
    device = config['device']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    early_stopping = config['early_stopping']
    output_dir = config['output_dir']
    grad_clip_value = config.get('grad_clip_value', 1.0)  # Default to 1.0 if not specified
    
    # Create experiment subdirectory
    experiment_path = output_dir / 'models' / config['experiment_name']
    os.makedirs(experiment_path, exist_ok=True)
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Create warmup scheduler if requested
    if config.get('warmup_epochs', 0) > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, 
            start_factor=0.1, 
            end_factor=1.0, 
            total_iters=config['warmup_epochs'] * len(train_loader)
        )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=config['scheduler_patience']
    )
    
    # Mixed precision training (updated for PyTorch 2.0+)
    if config['mixed_precision'] and torch.cuda.is_available():
        # Use proper 'cuda' device parameter
        scaler = torch.amp.GradScaler(device_type='cuda')
    else:
        scaler = None
    
    # Training metrics
    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses = []
    val_losses = []
    
    # Start training
    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()
    
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        # Training phase
        model.train()
        epoch_train_loss = 0
        
        # Progress tracking
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]")
        
        for batch_idx, (x, y) in enumerate(progress):
            x, y = x.to(device), y.to(device)
            
            # Mixed precision forward pass
            if scaler:
                with torch.amp.autocast(device_type='cuda'):
                    if config['approach'] == 'feature_detection':
                        pred = model(x)
                        loss = criterion(pred, y)
                    else:
                        pred = model(x)
                        loss = criterion(pred, y)
                
                # Mixed precision backward pass with gradient clipping
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                
                # Unscale gradients for clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
                
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard forward pass
                if config['approach'] == 'feature_detection':
                    pred = model(x)
                    loss = criterion(pred, y)
                else:
                    pred = model(x)
                    loss = criterion(pred, y)
                
                # Standard backward pass with gradient clipping
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_value)
                optimizer.step()
            
            # Apply warmup scheduler if in warmup period
            if config.get('warmup_epochs', 0) > 0 and epoch <= config['warmup_epochs']:
                warmup_scheduler.step()
            
            # Skip NaN losses in the average calculation
            if not torch.isnan(loss) and not torch.isinf(loss):
                epoch_train_loss += loss.item()
            else:
                print(f"Warning: NaN/Inf loss detected at batch {batch_idx}. Skipping in average calculation.")
            
            # Update progress bar with non-NaN loss
            if not torch.isnan(loss) and not torch.isinf(loss):
                progress.set_postfix({'loss': loss.item()})
            else:
                progress.set_postfix({'loss': 'NaN/Inf'})
        
        # Calculate average training loss (safely handling NaN batches)
        if len(train_loader) > 0:
            avg_train_loss = epoch_train_loss / len(train_loader)
        else:
            avg_train_loss = float('nan')
        
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        epoch_val_loss = 0
        
        # Additional metrics for physics-guided model
        l1_error = 0
        ssim_values = 0
        
        progress = tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Valid]")
        
        with torch.no_grad():
            for x, y in progress:
                x, y = x.to(device), y.to(device)
                
                # Forward pass
                if config['approach'] == 'feature_detection':
                    pred = model(x)
                    loss = criterion(pred, y)
                    
                    # Use reconstructed velocity for metrics
                    pred_velocity = pred['velocity']
                else:
                    pred = model(x)
                    loss = criterion(pred, y)
                    pred_velocity = pred
                
                # Update metrics
                epoch_val_loss += loss.item()
                
                # Calculate additional metrics
                l1_error += F.l1_loss(pred_velocity, y[:, 0:1] if y.shape[1] > 1 else y).item()
                
                # Normalized for SSIM calculation
                pred_norm = (pred_velocity - pred_velocity.min()) / (pred_velocity.max() - pred_velocity.min() + 1e-8)
                target_norm = (y[:, 0:1] if y.shape[1] > 1 else y - y.min()) / (y.max() - y.min() + 1e-8)
                
                # Simple SSIM approximation
                c1, c2 = 0.01**2, 0.03**2
                mu_x = F.avg_pool2d(pred_norm, kernel_size=11, stride=1, padding=5)
                mu_y = F.avg_pool2d(target_norm, kernel_size=11, stride=1, padding=5)
                
                sigma_x = F.avg_pool2d(pred_norm**2, kernel_size=11, stride=1, padding=5) - mu_x**2
                sigma_y = F.avg_pool2d(target_norm**2, kernel_size=11, stride=1, padding=5) - mu_y**2
                sigma_xy = F.avg_pool2d(pred_norm * target_norm, kernel_size=11, stride=1, padding=5) - mu_x * mu_y
                
                ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
                       ((mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2))
                       
                ssim_values += torch.mean(ssim).item()
        
        # Calculate average validation metrics
        avg_val_loss = epoch_val_loss / len(val_loader)
        avg_l1_error = l1_error / len(val_loader)
        avg_ssim = ssim_values / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Update learning rate
        scheduler.step(avg_val_loss)
        
        # Print epoch results
        epoch_time = time.time() - epoch_start
        print(f"Epoch {epoch}/{num_epochs} completed in {epoch_time:.2f}s | "
              f"Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | "
              f"L1 Error: {avg_l1_error:.6f} | SSIM: {avg_ssim:.6f}")
        
        # Visualize predictions - Enhanced with display
        visualize_frequency = config.get('visualize_every', 5)
        if epoch % visualize_frequency == 0 or epoch == 1 or epoch == num_epochs:
            visualize_predictions(model, val_loader, device, epoch, config)
        
        # Check for improvement
        if avg_val_loss < best_val_loss:
            improvement = (best_val_loss - avg_val_loss) / best_val_loss * 100
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            
            # Save best model
            if config['save_models']:
                best_model_path = experiment_path / f"best_model_epoch_{epoch}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': best_val_loss,
                    'train_losses': train_losses,
                    'val_losses': val_losses
                }, best_model_path)
                
                # Also save with standard name
                torch.save(model.state_dict(), experiment_path / "best_model.pt")
                
                print(f"✅ New best model saved with {improvement:.2f}% improvement")
        else:
            early_stop_counter += 1
            
        # Early stopping check
        if early_stop_counter >= early_stopping:
            print(f"Early stopping triggered after {epoch} epochs")
            break
            
        # Save checkpoint every 10 epochs
        if epoch % 10 == 0 and config['save_models']:
            checkpoint_path = experiment_path / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch}")
    
    # Save final model
    if config['save_models']:
        final_path = experiment_path / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'loss': avg_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, final_path)
        print(f"✅ Final model saved at: {final_path}")
    
    # Plot loss curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{config["approach"].capitalize()} Model - Loss Curves')
    plt.legend()
    plt.grid(True)
    loss_curve_path = output_dir / 'visualizations' / f"{experiment_name}_loss_curves.png"
    plt.savefig(loss_curve_path)
    
    # Display loss curves
    try:
        from IPython.display import display, Image
        plt.close()
        display(Image(str(loss_curve_path)))
    except:
        plt.show()
        plt.close()
    
    # Report training summary
    total_time = time.time() - start_time
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    return model, train_losses, val_losses

def visualize_predictions(model, loader, device, epoch, config):
    """Generate visualizations based on the selected approach with enhanced display"""
    model.eval()
    
    # Get a batch of data
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    
    # Make predictions
    with torch.no_grad():
        if config['approach'] == 'feature_detection':
            predictions = model(x)
        else:
            predictions = model(x)
    
    # Create visualization based on approach
    if config['approach'] == 'feature_detection':
        viz_path = visualize_geological_features(x, y, predictions, epoch, config)
    else:
        viz_path = visualize_velocity_prediction(x, y, predictions, epoch, config)
    
    # Display the visualization
    try:
        from IPython.display import display, Image
        print(f"Visualizations for epoch {epoch}:")
        display(Image(str(viz_path)))
    except Exception as e:
        print(f"Could not display image directly: {e}")
        print(f"Visualization saved at: {viz_path}")

def visualize_velocity_prediction(x, y, pred, epoch, config):
    """Visualize velocity model predictions with enhanced display"""
    # Convert tensors to numpy
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    pred_np = pred.cpu().numpy()
    
    # Create figure
    n_samples = min(4, x.size(0))
    fig, axs = plt.subplots(n_samples, 4, figsize=(16, 4 * n_samples))
    
    # Handle single sample case
    if n_samples == 1:
        axs = np.array([axs])
    
    for i in range(n_samples):
        # Input seismic data (first channel)
        axs[i, 0].imshow(x_np[i, 0], cmap='seismic', aspect='auto')
        axs[i, 0].set_title(f"Input Seismic #{i+1}")
        
        # Ground truth velocity
        if y_np.shape[1] > 1:
            y_plot = y_np[i, 0]  # For feature mode, use first channel (salt)
        else:
            y_plot = y_np[i, 0]
            
        axs[i, 1].imshow(y_plot, cmap='magma', origin='lower')
        axs[i, 1].set_title(f"Ground Truth #{i+1}")
        
        # Predicted velocity
        axs[i, 2].imshow(pred_np[i, 0], cmap='magma', origin='lower')
        axs[i, 2].set_title(f"Prediction #{i+1}")
        
        # Absolute error
        abs_error = np.abs(pred_np[i, 0] - y_plot)
        axs[i, 3].imshow(abs_error, cmap='inferno', origin='lower')
        axs[i, 3].set_title(f"Abs Error #{i+1}")
        
        # Remove axis ticks
        for j in range(4):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    
    plt.tight_layout()
    plt.suptitle(f"Physics-Guided Velocity Prediction - Epoch {epoch}", y=1.02, fontsize=16)
    
    # Save figure
    viz_path = config['output_dir'] / 'visualizations' / f"velocity_pred_epoch_{epoch}.png"
    plt.savefig(viz_path, bbox_inches='tight')
    plt.close()
    
    return viz_path
    
def visualize_geological_features(x, y, predictions, epoch, config):
    """Visualize geological feature predictions with enhanced display"""
    # Convert tensors to numpy
    x_np = x.cpu().numpy()
    y_np = y.cpu().numpy()
    
    # Apply sigmoid to convert logits to probabilities for visualization
    salt_pred = torch.sigmoid(predictions['salt']).cpu().numpy()
    fault_pred = torch.sigmoid(predictions['faults']).cpu().numpy()
    layer_pred = torch.sigmoid(predictions['layers']).cpu().numpy()
    velocity_pred = predictions['velocity'].cpu().numpy()
    
    # Create figure
    n_samples = min(3, x.size(0))
    fig, axs = plt.subplots(n_samples, 5, figsize=(20, 5 * n_samples))
    
    # Handle single sample case
    if n_samples == 1:
        axs = np.array([axs])
    
    for i in range(n_samples):
        # Input seismic data (first channel)
        axs[i, 0].imshow(x_np[i, 0], cmap='seismic', aspect='auto')
        axs[i, 0].set_title(f"Input Seismic #{i+1}")
        
        # Salt body detection
        salt_true = y_np[i, 0]
        axs[i, 1].imshow(salt_pred[i, 0], cmap='viridis')
        axs[i, 1].contour(salt_true, colors='r', linewidths=0.5, levels=[0.5])
        axs[i, 1].set_title(f"Salt Detection #{i+1}")
        
        # Fault detection
        fault_true = y_np[i, 1]
        axs[i, 2].imshow(fault_pred[i, 0], cmap='viridis')
        axs[i, 2].contour(fault_true, colors='r', linewidths=0.5, levels=[0.5])
        axs[i, 2].set_title(f"Fault Detection #{i+1}")
        
        # Layer detection
        layer_true = y_np[i, 2]
        axs[i, 3].imshow(layer_pred[i, 0], cmap='viridis')
        axs[i, 3].contour(layer_true, colors='r', linewidths=0.5, levels=[0.5])
        axs[i, 3].set_title(f"Layer Detection #{i+1}")
        
        # Reconstructed velocity
        axs[i, 4].imshow(velocity_pred[i, 0], cmap='magma', origin='lower')
        axs[i, 4].set_title(f"Reconstructed Velocity #{i+1}")
        
        # Remove axis ticks
        for j in range(5):
            axs[i, j].set_xticks([])
            axs[i, j].set_yticks([])
    
    plt.tight_layout()
    plt.suptitle(f"Geological Feature Detection - Epoch {epoch}", y=1.02, fontsize=16)
    
    # Save figure
    viz_path = config['output_dir'] / 'visualizations' / f"features_epoch_{epoch}.png" 
    plt.savefig(viz_path, bbox_inches='tight')
    plt.close()
    
    return viz_path


# Cell 8: Discriminator Loss Hook (Modular Helper)
# {"_collapsed": true}



def add_geodisc_loss(config, pred_velocity, discriminator):
    """Apply geological discriminator loss if enabled in config."""
    if config.get('use_geodiscriminator', False):
        with torch.no_grad():
            realism_score = discriminator(pred_velocity).squeeze()
            realism_loss = (1 - realism_score).mean()
        return config['geodisc_weight'] * realism_loss
    return 0.


def train_model(model, train_loader, val_loader, criterion, config, discriminator=None):
    device = config['device']
    num_epochs = config['num_epochs']
    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    early_stopping = config['early_stopping']
    output_dir = config['output_dir']

    experiment_path = output_dir / 'models' / config['experiment_name']
    os.makedirs(experiment_path, exist_ok=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=config['scheduler_patience'])
    scaler = torch.cuda.amp.GradScaler() if config['mixed_precision'] and torch.cuda.is_available() else None

    best_val_loss = float('inf')
    early_stop_counter = 0
    train_losses, val_losses = [], []

    print(f"Starting training for {num_epochs} epochs...")
    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        model.train()
        epoch_train_loss = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} [Train]"):
            x, y = x.to(device), y.to(device)

            if scaler:
                with torch.amp.autocast('cuda'):
                    pred = model(x)
                    loss = criterion(pred, y)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred = model(x)
                loss = criterion(pred, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if isinstance(pred, dict):
                pred_velocity = pred.get('velocity', pred)
            else:
                pred_velocity = pred

            loss += add_geodisc_loss(config, pred_velocity, discriminator)
            epoch_train_loss += loss.item()

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch}/{num_epochs} [Valid]"):
                x, y = x.to(device), y.to(device)
                pred = model(x)
                val_loss = criterion(pred, y)

                if isinstance(pred, dict):
                    pred_velocity = pred.get('velocity', pred)
                else:
                    pred_velocity = pred

                val_loss += add_geodisc_loss(config, pred_velocity, discriminator)
                epoch_val_loss += val_loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f} | Val Loss = {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            early_stop_counter = 0
            if config['save_models']:
                torch.save(model.state_dict(), experiment_path / "best_model.pt")
        else:
            early_stop_counter += 1

        if early_stop_counter >= early_stopping:
            print(f"Early stopping at epoch {epoch}")
            break

    total_time = time.time() - start_time
    print(f"Training completed in {total_time / 60:.2f} minutes")
    return model, train_losses, val_losses


def load_model_for_submission(config):
    from models import GeoAwareConvNet, GeologicalFeatureNet, PhysicsGuidedUNet

    model_name = config.get('approach', 'physics_guided')
    if model_name == 'geo_aware':
        model = GeoAwareConvNet(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            hidden_dim=config['hidden_dim']
        )
    elif model_name == 'feature_detection':
        model = GeologicalFeatureNet(
            in_channels=config['in_channels'],
            hidden_dim=config['hidden_dim']
        )
    else:
        model = PhysicsGuidedUNet(
            in_channels=config['in_channels'],
            out_channels=config['out_channels'],
            hidden_dim=config['hidden_dim']
        )
    return model.to(config['device'])



# Cell 9: Threshold-Based Approach

def apply_thresholding(data, method='otsu', threshold=None, stats=None):
    """Apply thresholding to create binary velocity model"""
    from skimage import filters
    
    # Determine threshold
    if threshold is not None:
        # Use provided threshold
        thresh = threshold
    elif method == 'otsu':
        # Otsu's adaptive thresholding
        try:
            thresh = filters.threshold_otsu(data)
        except:
            # Fallback to mean if Otsu fails
            thresh = np.mean(data)
    elif method == 'mean':
        # Mean-based thresholding
        thresh = np.mean(data)
    elif method == 'adaptive':
        # Local adaptive thresholding
        thresh = filters.threshold_local(data, block_size=15)
    elif stats is not None:
        # Use pre-calculated statistics
        thresh = stats['auto_threshold']
    else:
        # Default to mean
        thresh = np.mean(data)
    
    # Apply threshold
    binary = (data > thresh).astype(np.float32)
    
    return binary


def post_process_binary(binary, config):
    """Apply post-processing to binary velocity model - FIXED for dimension mismatch"""
    from scipy import ndimage
    
    # Copy to avoid modifying original
    result = binary.copy()
    
    # Check input dimensions
    input_dims = len(result.shape)
    
    # Apply morphological operations if enabled
    if config.get('use_morphology', True):
        # Create structure element with matching dimensions
        if input_dims == 2:
            # 2D input
            structure = np.ones((2, 2))
            structure_close = np.ones((3, 3))
        elif input_dims == 3:
            # 3D input (batch, height, width)
            if result.shape[0] == 1:
                # Single channel, treat as 2D after squeezing
                result = result.squeeze(0)
                structure = np.ones((2, 2))
                structure_close = np.ones((3, 3))
                # Flag that we need to unsqueeze later
                needs_unsqueeze = True
            else:
                # Multi-batch, use 3D structure
                structure = np.ones((1, 2, 2))
                structure_close = np.ones((1, 3, 3))
                needs_unsqueeze = False
        elif input_dims == 4:
            # 4D input (batch, channels, height, width)
            structure = np.ones((1, 1, 2, 2))
            structure_close = np.ones((1, 1, 3, 3))
            needs_unsqueeze = False
        else:
            # Unexpected dimension, skip morphology
            print(f"Warning: Skipping morphology for unusual dimensions: {result.shape}")
            structure = None
            
        # Apply morphology if we have a valid structure
        if structure is not None:
            try:
                # Clean up small artifacts
                result = ndimage.binary_opening(result, structure=structure)
                result = ndimage.binary_closing(result, structure=structure_close)
                
                # Restore original dimensions if needed
                if input_dims == 3 and 'needs_unsqueeze' in locals() and needs_unsqueeze:
                    result = result[np.newaxis, ...]
            except Exception as e:
                print(f"Warning: Morphology operation failed: {e}")
                print(f"Input shape: {binary.shape}, Structure shape: {structure.shape}")
    
    # Apply edge enhancement if specified
    if config.get('edge_enhancement', 0) > 0:
        # Calculate edges
        edge_x = ndimage.sobel(binary, axis=-2)
        edge_y = ndimage.sobel(binary, axis=-1)
        edges = np.sqrt(edge_x**2 + edge_y**2)
        
        # Normalize edge strength
        edge_strength = edges / (np.max(edges) + 1e-8)
        
        # Enhance edges
        enhanced = result.copy()
        enhanced[edge_strength > 0.3] = 1.0
        
        # Blend with original based on enhancement strength
        alpha = config['edge_enhancement']
        result = (1 - alpha) * result + alpha * enhanced
    
    return result


def process_with_thresholding(input_data, config, stats=None):
    """Process seismic data using thresholding approach"""
    # Apply gain to seismic data
    time_steps = input_data.shape[1]
    time = np.linspace(0, 1, time_steps)
    gain = (time ** 2)[:, np.newaxis]
    
    gained_data = input_data.copy()
    for c in range(input_data.shape[0]):
        gained_data[c] = input_data[c] * gain
    
    # Take mean across channels and time
    # We need a 2D map that we can threshold
    mean_data = np.mean(gained_data, axis=(0, 1))
    
    # Apply thresholding
    binary = apply_thresholding(
        mean_data, 
        method=config['threshold_method'],
        stats=stats
    )
    
    # Apply post-processing
    processed = post_process_binary(binary, config)
    
    return processed


def generate_threshold_submission(test_dir, output_path, config, stats=None):
    """Generate submission using thresholding approach"""
    # Get test files
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.npy")))
    print(f"Found {len(test_files)} test files")
    
    rows = []
    
    # Process files with progress tracking
    progress = tqdm(test_files, desc="Generating threshold-based submission")
    
    for filepath in progress:
        # Get file ID
        oid = os.path.splitext(os.path.basename(filepath))[0]
        
        # Load data
        data = np.load(filepath)
        
        # Check shape and extract if needed
        if len(data.shape) == 4:  # (batch, channels, time, receivers)
            data = data[0]  # Use first sample if batched
        
        # Process with thresholding
        binary_pred = process_with_thresholding(data, config, stats)
        
        # Ensure correct shape (70x70)
        if binary_pred.shape != (70, 70):
            from skimage.transform import resize
            binary_pred = resize(binary_pred, (70, 70), order=0, preserve_range=True)
        
        # Format for submission (all rows, odd columns)
        for y in range(70):
            row_id = f"{oid}_y_{y}"
            row_data = binary_pred[y, 1:70:2]  # Extract odd-indexed columns
            rows.append([row_id] + row_data.tolist())
    
    # Create submission DataFrame
    columns = ["ID"] + [f"x_{i}" for i in range(1, 70, 2)]
    submission_df = pd.DataFrame(rows, columns=columns)
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Threshold-based submission saved at: {output_path}")
    
    return submission_df


# Cell 10: Implement edge-based velocity modeling as a post-processing step

import numpy as np
from scipy import ndimage
from skimage import feature, segmentation, filters, morphology, measure, color
import matplotlib.pyplot as plt

def extract_edges(feature_map, method='sobel', threshold=0.2):
    """Extract edges from a feature probability map"""
    if method == 'sobel':
        edges_x = ndimage.sobel(feature_map, axis=1)
        edges_y = ndimage.sobel(feature_map, axis=0)
        edges = np.sqrt(edges_x**2 + edges_y**2)
    elif method == 'canny':
        edges = feature.canny(feature_map, sigma=1.0, low_threshold=0.1, high_threshold=0.3)
    else:
        edges = filters.scharr(feature_map)
    
    # Normalize and threshold
    if edges.max() > 0:
        edges = edges / edges.max()
    edges = (edges > threshold).astype(np.float32)
    
    return edges

def multi_scale_edge_detection(seismic_data, scales=[1, 2, 3]):
    """Apply edge detection at multiple scales for robust edge extraction"""
    # If multi-channel, convert to single channel
    if seismic_data.ndim > 2:
        # Use mean across channels or first channel
        if seismic_data.shape[0] > 1:
            data = np.mean(seismic_data, axis=0)
        else:
            data = seismic_data[0]
    else:
        data = seismic_data
    
    # Apply edge detection at multiple scales
    edges_multi_scale = np.zeros_like(data)
    
    for scale in scales:
        # Smooth data at this scale
        smoothed = ndimage.gaussian_filter(data, sigma=scale)
        
        # Detect edges
        edges_x = ndimage.sobel(smoothed, axis=1)
        edges_y = ndimage.sobel(smoothed, axis=0)
        edges = np.sqrt(edges_x**2 + edges_y**2)
        
        # Normalize
        if edges.max() > 0:
            edges = edges / edges.max()
        
        # Accumulate
        edges_multi_scale = np.maximum(edges_multi_scale, edges)
    
    return edges_multi_scale

def combine_edges(edge_map, feature_edges, weights={'main': 0.5, 'salt': 0.3, 'faults': 0.2, 'layers': 0.1}):
    """Combine multiple edge maps with weights"""
    # Start with main edge map
    combined = weights['main'] * edge_map
    
    # Add feature-specific edges
    for feature_name, edge in feature_edges.items():
        if feature_name in weights:
            combined += weights[feature_name] * edge
    
    # Normalize
    if combined.max() > 0:
        combined = combined / combined.max()
    
    return combined

def close_gaps(edge_map, threshold=0.3, min_size=3):
    """Close small gaps in edge map to create closed contours"""
    # Threshold edge map
    binary_edges = (edge_map > threshold).astype(np.uint8)
    
    # Apply morphological closing to bridge small gaps
    closed_edges = morphology.binary_closing(binary_edges, morphology.disk(min_size//2))
    
    # Clean up small isolated edge segments
    cleaned_edges = morphology.remove_small_objects(closed_edges, min_size=min_size)
    
    return cleaned_edges.astype(np.float32)

def segment_image(edge_map, threshold=0.3, compactness=30):
    """Segment image based on edge map using watershed"""
    # Invert edge map to get ridge map (watershed works on valleys)
    ridge_map = 1 - edge_map
    
    # Use watershed algorithm to segment
    # Create markers for watershed (can be improved with feature info)
    binary_edges = (edge_map > threshold)
    distance = ndimage.distance_transform_edt(~binary_edges)
    
    # Create watershed markers
    local_max = feature.peak_local_max(distance, footprint=np.ones((3, 3)), labels=~binary_edges)
    markers = np.zeros(distance.shape, dtype=np.int32)
    markers[tuple(local_max.T)] = np.arange(1, len(local_max) + 1)
    
    # Apply watershed
    segmented = segmentation.watershed(ridge_map, markers, mask=~binary_edges)
    
    # Count regions
    num_regions = len(np.unique(segmented)) - 1  # Exclude 0 (background)
    
    return segmented, num_regions

def classify_region(region_mask, feature_maps, thresholds={'salt': 0.5, 'faults': 0.3, 'layers': 0.4}):
    """Classify region based on feature map probabilities"""
    # For each feature type, calculate average probability within the region
    region_probs = {}
    for feature_name, feature_map in feature_maps.items():
        avg_prob = np.mean(feature_map[region_mask])
        region_probs[feature_name] = avg_prob
    
    # Classify based on highest probability
    if region_probs['salt'] > thresholds['salt']:
        return 'salt'
    elif region_probs['layers'] > thresholds['layers']:
        # Further classify layers by depth
        y_indices = np.where(region_mask)[0]
        avg_depth = np.mean(y_indices) / region_mask.shape[0]
        layer_number = int(avg_depth * 10)  # 0-9 based on depth
        return f'layer_{layer_number}'
    elif region_probs['faults'] > thresholds['faults']:
        return 'fault'
    else:
        # Background - classify by depth for realistic layering
        y_indices = np.where(region_mask)[0]
        avg_depth = np.mean(y_indices) / region_mask.shape[0]
        layer_number = int(avg_depth * 5)  # 0-4 based on depth
        return f'background_{layer_number}'

def get_velocity_for_region_type(region_type):
    """Assign velocity based on region type"""
    # Typical velocity values (m/s) for different geological structures
    if region_type == 'salt':
        # Salt bodies have high velocity
        return 4500.0
    elif region_type.startswith('layer_'):
        # Layers get increasing velocity with depth
        layer_number = int(region_type.split('_')[1])
        return 2000.0 + layer_number * 200.0
    elif region_type == 'fault':
        # Fault zones often have slightly lower velocity
        return 2300.0
    elif region_type.startswith('background_'):
        # Background also increases with depth
        layer_number = int(region_type.split('_')[1])
        return 2200.0 + layer_number * 150.0
    else:
        # Default velocity
        return 2500.0

def apply_geological_constraints(velocity_model, edge_map, edge_weight=0.2):
    """Apply geological constraints to ensure realistic velocity model"""
    # 1. Ensure velocity increases with depth (general trend in geology)
    rows, cols = velocity_model.shape
    depth_trend = np.linspace(0.9, 1.1, rows).reshape(-1, 1)
    velocity_model = velocity_model * depth_trend
    
    # 2. Enhance velocity contrast at edges
    edge_enhanced = velocity_model.copy()
    edge_gradient = ndimage.gaussian_gradient_magnitude(velocity_model, sigma=1.0)
    
    # Only enhance where we have detected edges
    where_edges = (edge_map > 0.2)
    edge_enhanced[where_edges] = velocity_model[where_edges] * (1.0 + edge_weight * edge_gradient[where_edges])
    
    # 3. Smooth very small regions while preserving edges
    # Create edge-preserving smoothing mask
    smooth_mask = 1.0 - (edge_map > 0.2).astype(np.float32)
    
    # Apply selective smoothing
    smoothed = ndimage.gaussian_filter(edge_enhanced, sigma=1.0)
    final_velocity = edge_enhanced * (1.0 - smooth_mask) + smoothed * smooth_mask
    
    return final_velocity

def edge_based_velocity_modeling(feature_maps, seismic_data=None):
    """Generate velocity model using edge detection and region filling"""
    # 1. Get feature maps and convert to numpy if needed
    if isinstance(feature_maps, dict) and 'velocity' in feature_maps:
        # If we're using the feature detection model output, it contains velocity
        if isinstance(feature_maps['velocity'], torch.Tensor):
            initial_velocity = feature_maps['velocity'].cpu().numpy().squeeze()
        else:
            initial_velocity = feature_maps['velocity'].squeeze()
    else:
        # Fallback if no velocity is provided
        initial_velocity = np.ones((70, 70)) * 2500.0
    
    salt_map = feature_maps['salt'] if isinstance(feature_maps, dict) else feature_maps
    if isinstance(salt_map, torch.Tensor):
        salt_map = torch.sigmoid(salt_map).cpu().numpy().squeeze()
    else:
        salt_map = salt_map.squeeze()
    
    # Convert other feature maps if available
    if isinstance(feature_maps, dict):
        processed_maps = {}
        for key, value in feature_maps.items():
            if key in ['salt', 'faults', 'layers']:
                if isinstance(value, torch.Tensor):
                    processed_maps[key] = torch.sigmoid(value).cpu().numpy().squeeze()
                else:
                    processed_maps[key] = value.squeeze()
    else:
        # If not a dictionary, create dummy feature maps
        processed_maps = {
            'salt': salt_map,
            'faults': np.zeros_like(salt_map),
            'layers': np.zeros_like(salt_map)
        }
    
    # 2. Extract edges from each feature map
    feature_edges = {
        'salt': extract_edges(processed_maps['salt']),
        'faults': extract_edges(processed_maps['faults']),
        'layers': extract_edges(processed_maps['layers'])
    }
    
    # 3. Get edge map from seismic data if available
    if seismic_data is not None and not isinstance(seismic_data, type(None)):
        # Direct edge detection from seismic
        seismic_edges = multi_scale_edge_detection(seismic_data)
    else:
        # Fallback to using feature maps for edge detection
        combined_feature = (processed_maps['salt'] + 
                          processed_maps.get('faults', np.zeros_like(salt_map)) + 
                          processed_maps.get('layers', np.zeros_like(salt_map))) / 3.0
        seismic_edges = extract_edges(combined_feature)
    
    # 4. Combine edges
    combined_edges = combine_edges(seismic_edges, feature_edges)
    enhanced_edges = close_gaps(combined_edges)
    
    # 5. Segment image into regions
    segmented, num_regions = segment_image(enhanced_edges)
    
    # 6. Assign velocity values to each region
    velocity_model = np.zeros_like(segmented, dtype=float)
    region_types = {}
    
    for region_id in range(1, num_regions + 1):
        # Create mask for this region
        region_mask = (segmented == region_id)
        
        # Skip very small regions
        if np.sum(region_mask) < 5:
            continue
            
        # Determine region characteristics
        region_type = classify_region(region_mask, processed_maps)
        region_types[region_id] = region_type
        
        # Assign appropriate velocity
        velocity_value = get_velocity_for_region_type(region_type)
        
        # Fill the region
        velocity_model[region_mask] = velocity_value
    
    # Handle any unfilled areas
    if np.any(velocity_model == 0):
        velocity_model[velocity_model == 0] = 2500.0  # Default velocity
    
    # 7. Apply geological constraints
    final_velocity = apply_geological_constraints(velocity_model, enhanced_edges)
    
    # 8. If initial velocity is provided, blend with edge-based model
    alpha = 0.7  # Weight for edge-based model
    final_velocity = alpha * final_velocity + (1 - alpha) * initial_velocity
    
    # 9. Ensure output range matches expected range
    # Assuming output should be normalized to [0,1]
    if np.max(final_velocity) > 0:
        normalized = (final_velocity - np.min(final_velocity)) / (np.max(final_velocity) - np.min(final_velocity))
    else:
        normalized = final_velocity
    
    # For debugging: visualize the edge detection and segmentation
    # visualize_edge_segmentation(processed_maps, enhanced_edges, segmented, normalized)
    
    return normalized

def bimodality_coefficient(data):
    """Calculate the bimodality coefficient of a distribution.
    
    A value greater than 0.555 suggests bimodality.
    Formula: (skewness²+1)/(kurtosis+3*(n-1)²/((n-2)*(n-3)))
    """
    n = len(data)
    if n < 4:
        return 0.0
        
    # Calculate moments
    mean = np.mean(data)
    var = np.var(data)
    std = np.sqrt(var)
    
    # Skewness
    skewness = np.mean(((data - mean) / std) ** 3) if std > 0 else 0
    
    # Kurtosis (using Fisher's definition: normal = 0)
    kurtosis = np.mean(((data - mean) / std) ** 4) - 3 if std > 0 else 0
    
    # Bimodality coefficient
    numerator = skewness**2 + 1
    denominator = kurtosis + (3 * (n-1)**2)/((n-2)*(n-3))
    
    return numerator / denominator

def visualize_edge_segmentation(feature_maps, edges, segmented, velocity):
    """Visualize the edge-based segmentation process"""
    plt.figure(figsize=(16, 12))
    
    plt.subplot(231)
    plt.imshow(feature_maps['salt'], cmap='viridis')
    plt.title('Salt Feature Map')
    plt.colorbar()
    
    plt.subplot(232)
    plt.imshow(edges, cmap='gray')
    plt.title('Enhanced Edges')
    plt.colorbar()
    
    plt.subplot(233)
    plt.imshow(segmented, cmap='nipy_spectral')
    plt.title(f'Segmentation ({len(np.unique(segmented))-1} regions)')
    plt.colorbar()
    
    plt.subplot(234)
    if 'faults' in feature_maps:
        plt.imshow(feature_maps['faults'], cmap='plasma')
        plt.title('Faults Feature Map')
    else:
        plt.imshow(np.zeros_like(feature_maps['salt']), cmap='plasma')
        plt.title('No Fault Features')
    plt.colorbar()
    
    plt.subplot(235)
    if 'layers' in feature_maps:
        plt.imshow(feature_maps['layers'], cmap='cividis')
        plt.title('Layers Feature Map')
    else:
        plt.imshow(np.zeros_like(feature_maps['salt']), cmap='cividis')
        plt.title('No Layer Features')
    plt.colorbar()
    
    plt.subplot(236)
    plt.imshow(velocity, cmap='magma')
    plt.title('Final Velocity Model')
    plt.colorbar()
    
    plt.tight_layout()
    plt.savefig('edge_based_segmentation.png')
    plt.close()

def post_process_prediction_with_edges(pred_np, config, stats=None, seismic_data=None):
    """Apply edge-based post-processing to model predictions"""
    # For feature detection, use the edge-based velocity modeling
    if config['approach'] == 'feature_detection':
        # Apply edge-based velocity modeling
        edge_based_velocity = edge_based_velocity_modeling(pred_np, seismic_data)
        
        if config['post_process']:
            # Apply binary thresholding to get clean velocity model
            binary = (edge_based_velocity > 0.5).astype(np.float32)
            processed = post_process_binary(binary, config)
            return processed
        else:
            # Return edge-based velocity directly
            return edge_based_velocity
    else:
        # For other approaches, use the standard post-processing instead of recursively calling
        # Original implementation (copied here to avoid recursion)
        if config['approach'] == 'thresholding':
            # For thresholding, apply edge enhancement
            if config['post_process']:
                return post_process_binary(pred_np, config)
            else:
                return pred_np
        else:  # physics_guided or geo_aware
            # For physics-guided approach
            if config['post_process']:
                # Enhance contrast for clearer boundaries
                from scipy import ndimage
                
                # Smooth with edge-preserving bilateral filter if available
                try:
                    from skimage.restoration import denoise_bilateral
                    smoothed = denoise_bilateral(
                        pred_np.squeeze(), 
                        sigma_color=0.1, 
                        sigma_spatial=1,
                        multichannel=False
                    )
                except:
                    # Fallback to gaussian filter
                    smoothed = ndimage.gaussian_filter(pred_np.squeeze(), sigma=0.5)
                
                # Apply contrast adjustment
                p2, p98 = np.percentile(smoothed, (2, 98))
                enhanced = np.clip((smoothed - p2) / (p98 - p2), 0, 1)
                
                # Convert to binary if very high contrast
                hist, bins = np.histogram(enhanced, bins=50)
                if np.max(hist) > 0.4 * np.sum(hist):
                    # Likely bimodal distribution, apply thresholding
                    thresh = apply_thresholding(enhanced, method=config['threshold_method'], stats=stats)
                    processed = post_process_binary(thresh, config)
                    return processed
                else:
                    # Not clearly bimodal, return enhanced version
                    return enhanced
            else:
                # Return raw prediction
                return pred_np.squeeze()




# Cell 12: Inference and Submission Functions

def load_best_model(config):
    """Load the best model for inference"""
    # Path to best model
    model_path = config['output_dir'] / 'models' / experiment_name / "best_model.pt"
    standard_path = config['output_dir'] / 'models' / experiment_name / "best_model.pt"
    
    # Create model
    model = create_model(config)
    
    # Try to load model weights
    try:
        # First try the full checkpoint format
        checkpoint = torch.load(model_path, map_location=config['device'])
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded model from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
        else:
            # If not a checkpoint, try direct state dict
            model.load_state_dict(checkpoint)
            print(f"Loaded model state dict from {model_path}")
    except FileNotFoundError:
        try:
            # Try standard path as fallback
            model.load_state_dict(torch.load(standard_path, map_location=config['device']))
            print(f"Loaded model from standard path: {standard_path}")
        except FileNotFoundError:
            print(f"Warning: No model found at {model_path} or {standard_path}")
            print("Using untrained model")
    
    # Set to evaluation mode
    model.eval()
    
    return model


def post_process_prediction(pred_np, config, stats=None):
    """Apply post-processing to model predictions"""
    # Determine approach-specific processing
    if config['approach'] == 'feature_detection':
        # For feature detection, use the reconstructed velocity
        velocity = pred_np['velocity']
        if config['post_process']:
            # Apply binary thresholding to get clean velocity model
            binary = (velocity > 0.5).astype(np.float32)
            processed = post_process_binary(binary, config)
            return processed
        else:
            # Return raw velocity reconstruction
            return velocity    
           
    elif config['approach'] == 'thresholding':
        # For thresholding, apply edge enhancement
        if config['post_process']:
            return post_process_binary(pred_np, config)
        else:
            return pred_np
            
    else:  # physics_guided
        # For physics-guided approach
        if config['post_process']:
            # Enhance contrast for clearer boundaries
            from scipy import ndimage
            
            # Smooth with edge-preserving bilateral filter if available
            try:
                from skimage.restoration import denoise_bilateral
                smoothed = denoise_bilateral(
                    pred_np.squeeze(), 
                    sigma_color=0.1, 
                    sigma_spatial=1,
                    multichannel=False
                )
            except:
                # Fallback to gaussian filter
                smoothed = ndimage.gaussian_filter(pred_np.squeeze(), sigma=0.5)
            
            # Apply contrast adjustment
            p2, p98 = np.percentile(smoothed, (2, 98))
            enhanced = np.clip((smoothed - p2) / (p98 - p2), 0, 1)
            
            # Convert to binary if very high contrast
            hist, bins = np.histogram(enhanced, bins=50)
            if np.max(hist) > 0.4 * np.sum(hist):
                # Likely bimodal distribution, apply thresholding
                thresh = apply_thresholding(enhanced, method=config['threshold_method'], stats=stats)
                processed = post_process_binary(thresh, config)
                return processed
            else:
                # Not clearly bimodal, return enhanced version
                return enhanced
        else:
            # Return raw prediction
            return pred_np.squeeze()


def generate_model_submission(model, test_dir, output_path, config, stats=None):
    """Generate submission using trained model"""
    # Get test files
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.npy")))
    print(f"Found {len(test_files)} test files")
    
    rows = []
    device = config['device']
    
    # Process files with progress tracking
    progress = tqdm(test_files, desc=f"Generating {config['approach']} submission")
    
    with torch.no_grad():
        for filepath in progress:
            # Get file ID
            oid = os.path.splitext(os.path.basename(filepath))[0]
            
            # Load data
            data = np.load(filepath)
            
            # Check shape and extract if needed
            if len(data.shape) == 4:  # (batch, channels, time, receivers)
                data = data[0]  # Use first sample if batched
            
            # Apply gain
            time_steps = data.shape[1]
            time = np.linspace(0, 1, time_steps)
            gain = (time ** 2)[:, np.newaxis]
            
            gained_data = data.copy()
            for c in range(data.shape[0]):
                gained_data[c] = data[c] * gain
                
            # Normalize if statistics are available
            if stats:
                gained_data = (gained_data - stats['input_mean']) / (stats['input_std'] + 1e-6)
            
            # Convert to tensor and add batch dimension
            x = torch.from_numpy(gained_data).float().unsqueeze(0).to(device)
            
            # Predict with model
            if config['approach'] == 'feature_detection':
                pred = model(x)
                # Convert dict of tensors to dict of numpy arrays
                pred_np = {k: v.cpu().numpy() for k, v in pred.items()}
            else:
                pred = model(x)
                pred_np = pred.cpu().numpy()
            
            # Apply post-processing
            processed = post_process_prediction(pred_np, config, stats)
            
            # Ensure correct shape (70x70)
            if processed.shape != (70, 70):
                processed = processed.squeeze()
                if processed.shape != (70, 70):
                    from skimage.transform import resize
                    processed = resize(processed, (70, 70), order=1, preserve_range=True)
            
            # Format for submission (all rows, odd columns)
            for y in range(70):
                row_id = f"{oid}_y_{y}"
                row_data = processed[y, 1:70:2]  # Extract odd-indexed columns
                rows.append([row_id] + row_data.tolist())
    
    # Create submission DataFrame
    columns = ["ID"] + [f"x_{i}" for i in range(1, 70, 2)]
    submission_df = pd.DataFrame(rows, columns=columns)
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Model-based submission saved at: {output_path}")
    
    return submission_df


def ensemble_predictions(models, x, config):
    """Generate ensemble prediction from multiple models"""
    device = config['device']
    ensemble_preds = []
    
    with torch.no_grad():
        # Basic predictions
        for model in models:
            if config['approach'] == 'feature_detection':
                pred = model(x)
                # Use velocity reconstruction
                ensemble_preds.append(pred['velocity'])
            else:
                pred = model(x)
                ensemble_preds.append(pred)
                
            # Add test-time augmentation if enabled
            if config.get('use_tta', False):
                # Horizontal flip
                x_flip = torch.flip(x, dims=[3])
                if config['approach'] == 'feature_detection':
                    pred_flip = model(x_flip)
                    # Flip back the prediction
                    velocity_flip = torch.flip(pred_flip['velocity'], dims=[3])
                    ensemble_preds.append(velocity_flip)
                else:
                    pred_flip = model(x_flip)
                    pred_flip = torch.flip(pred_flip, dims=[3])
                    ensemble_preds.append(pred_flip)
    
    # Average all predictions
    ensemble_pred = torch.mean(torch.stack(ensemble_preds), dim=0)
    
    return ensemble_pred


def generate_ensemble_submission(models, test_dir, output_path, config, stats=None):
    """Generate submission using ensemble of models"""
    # Get test files
    test_files = sorted(glob.glob(os.path.join(test_dir, "*.npy")))
    print(f"Found {len(test_files)} test files")
    
    rows = []
    device = config['device']
    
    # Process files with progress tracking
    progress = tqdm(test_files, desc="Generating ensemble submission")
    
    with torch.no_grad():
        for filepath in progress:
            # Get file ID
            oid = os.path.splitext(os.path.basename(filepath))[0]
            
            # Load data
            data = np.load(filepath)
            
            # Check shape and extract if needed
            if len(data.shape) == 4:  # (batch, channels, time, receivers)
                data = data[0]  # Use first sample if batched
            
            # Apply gain
            time_steps = data.shape[1]
            time = np.linspace(0, 1, time_steps)
            gain = (time ** 2)[:, np.newaxis]
            
            gained_data = data.copy()
            for c in range(data.shape[0]):
                gained_data[c] = data[c] * gain
                
            # Normalize if statistics are available
            if stats:
                gained_data = (gained_data - stats['input_mean']) / (stats['input_std'] + 1e-6)
            
            # Convert to tensor and add batch dimension
            x = torch.from_numpy(gained_data).float().unsqueeze(0).to(device)
            
            # Get ensemble prediction
            ensemble_pred = ensemble_predictions(models, x, config)
            pred_np = ensemble_pred.cpu().numpy()
            
            # Apply post-processing
            processed = post_process_prediction(pred_np, config, stats)
            
            # Ensure correct shape (70x70)
            if processed.shape != (70, 70):
                processed = processed.squeeze()
                if processed.shape != (70, 70):
                    from skimage.transform import resize
                    processed = resize(processed, (70, 70), order=1, preserve_range=True)
            
            # Format for submission (all rows, odd columns)
            for y in range(70):
                row_id = f"{oid}_y_{y}"
                row_data = processed[y, 1:70:2]  # Extract odd-indexed columns
                rows.append([row_id] + row_data.tolist())
    
    # Create submission DataFrame
    columns = ["ID"] + [f"x_{i}" for i in range(1, 70, 2)]
    submission_df = pd.DataFrame(rows, columns=columns)
    
    # Save submission
    submission_df.to_csv(output_path, index=False)
    print(f"✅ Ensemble submission saved at: {output_path}")
    
    return submission_df


def generate_submission(config, stats=None):
    """Generate submission based on selected approach"""
    approach = config['approach']
    output_path = config['submission_path']
    
    if approach == 'thresholding':
        # Use direct thresholding approach (no training)
        return generate_threshold_submission(
            config['test_dir'], 
            output_path, 
            config, 
            stats
        )
    elif config['ensemble_submission']:
        # Load multiple models for ensemble
        model_dir = config['output_dir'] / 'models' / experiment_name
        model_files = list(model_dir.glob("*model_epoch_*.pt"))
        
        if len(model_files) > 1:
            print(f"Found {len(model_files)} models for ensemble")
            models = []
            
            for model_file in model_files:
                model = create_model(config)
                # Load checkpoint
                checkpoint = torch.load(model_file, map_location=config['device'])
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                    
                model.eval()
                models.append(model)
                
            return generate_ensemble_submission(
                models,
                config['test_dir'],
                output_path,
                config,
                stats
            )
        else:
            # Fall back to single model if not enough models found
            print("Not enough models for ensemble, using best model")
            model = load_best_model(config)
            return generate_model_submission(
                model, 
                config['test_dir'], 
                output_path, 
                config, 
                stats
            )
    else:
        # Use single best model
        model = load_best_model(config)
        return generate_model_submission(
            model, 
            config['test_dir'], 
            output_path, 
            config, 
            stats
        )


# Cell 13: Run Main Pipeline

def run_pipeline(config):
    """Run the complete pipeline based on configuration"""
    # 1. Prepare datasets
    train_loader, val_loader, stats = load_and_prepare_data(config)
    
    # 2. Create model and loss function
    if config['approach'] != 'thresholding':
        model = create_model(config)
        criterion = create_loss_function(config)
        
        # 3. Train model
        model, train_losses, val_losses = train_model(
            model, train_loader, val_loader, criterion, config
        )
    else:
        print("Using thresholding approach, skipping model training")
        model = None
    
    # 4. Generate submission
    submission_df = generate_submission(config, stats)
    
    print(f"Pipeline completed for {config['approach']} approach")
    print(f"Submission saved at: {config['submission_path']}")
    
    return submission_df


# Execute the pipeline
if __name__ == "__main__":
    # Run the selected approach
    submission_df = run_pipeline(CONFIG)
    
    # If comparison mode is enabled, run multiple approaches
    if CONFIG['compare_approaches']:
        approaches = ['thresholding', 'physics_guided', 'feature_detection']
        results = {}
        
        for approach in approaches:
            if approach != CONFIG['approach']:  # Skip already run approach
                print(f"\nRunning comparison pipeline for {approach} approach")
                
                # Create copy of config with new approach
                comp_config = CONFIG.copy()
                comp_config['approach'] = approach
                comp_config['submission_path'] = f"submission_{approach}.csv"
                
                # Run pipeline with this approach
                results[approach] = run_pipeline(comp_config)
        
        # Show comparison summary
        print("\nComparison of approaches completed:")
        for approach in approaches:
            path = f"submission_{approach}.csv" if approach != CONFIG['approach'] else CONFIG['submission_path']
            print(f"- {approach.capitalize()}: {path}")
            


# Cell 14: Edge-Based Post-Processing Using Existing Model
# {\"_collapsed\": false}\n

# Create configuration for edge-based processing
CONFIG_EDGE = CONFIG.copy()
CONFIG_EDGE.update({
    'post_process': True,
    'use_edge_based': True,
    'submission_path': "submission_edge_based.csv",
})

# Dynamically resolve saved model path
best_model_path = CONFIG['output_dir'] / 'models' / CONFIG['experiment_name'] / 'best_model.pt'
print(f"Using existing trained model: {best_model_path}")

# Extract model approach from experiment name
if CONFIG['experiment_name'].startswith('geo_aware'):
    original_approach = 'geo_aware'
elif CONFIG['experiment_name'].startswith('physics_guided'):
    original_approach = 'physics_guided'
elif CONFIG['experiment_name'].startswith('feature_detection'):
    original_approach = 'feature_detection'
elif CONFIG['experiment_name'].startswith('thresholding'):
    original_approach = 'thresholding'
else:
    # Try to infer from directory structure
    dir_name = str(best_model_path.parent)
    if 'geo_aware' in dir_name:
        original_approach = 'geo_aware'
    elif 'physics' in dir_name:
        original_approach = 'physics_guided'
    elif 'feature' in dir_name:
        original_approach = 'feature_detection'
    elif 'threshold' in dir_name:
        original_approach = 'thresholding'
    else:
        print("WARNING: Could not determine model approach from experiment name.")
        print("Attempting to load model to determine architecture...")
        # Default to current approach and we'll try to adapt later
        original_approach = CONFIG['approach']

print(f"Detected original model approach: {original_approach}")
CONFIG_EDGE['approach'] = original_approach

# Load the model using the detected approach
model = create_model(CONFIG_EDGE)

if best_model_path.exists():
    try:
        model.load_state_dict(torch.load(best_model_path, map_location=CONFIG_EDGE['device']))
        model.eval()
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"Error loading model with detected approach: {e}")
        print("Attempting to try other model architectures...")
        
        # Try all approaches if first attempt fails
        for approach in ['geo_aware', 'physics_guided', 'feature_detection', 'thresholding']:
            if approach == original_approach:
                continue  # Skip already tried approach
                
            print(f"Trying {approach} architecture...")
            CONFIG_EDGE['approach'] = approach
            model = create_model(CONFIG_EDGE)
            
            try:
                model.load_state_dict(torch.load(best_model_path, map_location=CONFIG_EDGE['device']))
                model.eval()
                print(f"✅ Model loaded successfully using {approach} architecture")
                break
            except Exception:
                print(f"Failed to load using {approach} architecture")
        
        if model is None:
            raise FileNotFoundError(f"❌ Could not load model with any architecture. Please check the model path.")
else:
    raise FileNotFoundError(f"❌ No model found at {best_model_path}. Run training first.")

# Get dataset statistics
_, _, stats = load_and_prepare_data(CONFIG_EDGE)

# Generate submission with edge-based post-processing
def generate_edge_based_submission():
    """Generate submission using edge-based post-processing on existing model"""
    original_post_process = globals()['post_process_prediction']
    globals()['post_process_prediction'] = post_process_prediction_with_edges

    print("Generating submission with edge-based post-processing...")
    submission_df = generate_model_submission(
        model,
        CONFIG_EDGE['test_dir'],
        CONFIG_EDGE['submission_path'],
        CONFIG_EDGE,
        stats
    )

    globals()['post_process_prediction'] = original_post_process
    print(f"✅ Edge-based submission saved to: {CONFIG_EDGE['submission_path']}")
    return submission_df



# Run it
submission_df = generate_edge_based_submission()


# Cell 15: Additional Analysis and Visualization Tools (Optional)
# <codecell>
# {\"_collapsed\": true}\n
if CONFIG.get('run_analysis_tools', True):

    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats
    from scipy import ndimage
    from skimage import filters
    from scipy import fft
    import os

    def analyze_velocity_models(output_files, max_samples=10):
        """Analyze velocity models to understand their characteristics"""
        sample_files = output_files[:min(len(output_files), max_samples)]
        samples = []

        for f in sample_files:
            try:
                data = np.load(f, mmap_mode='r')
                if len(data.shape) == 4:
                    samples.append(data[:5])
                elif len(data.shape) == 3:
                    samples.append(data[:5, np.newaxis])
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if not samples:
            print("No samples loaded for analysis")
            return

        all_samples = np.concatenate(samples, axis=0).squeeze()
        print(f"Analyzing {all_samples.shape[0]} velocity models")

        mean_vel = np.mean(all_samples)
        std_vel = np.std(all_samples)
        min_vel = np.min(all_samples)
        max_vel = np.max(all_samples)

        print(f"Velocity statistics: mean={mean_vel:.2f}, std={std_vel:.2f}, min={min_vel:.2f}, max={max_vel:.2f}")

        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.hist(all_samples.flatten(), bins=50)
        plt.title("Velocity Distribution")
        plt.xlabel("Velocity Value")
        plt.ylabel("Frequency")

        kde = scipy_stats.gaussian_kde(all_samples.flatten())
        x = np.linspace(min_vel, max_vel, 1000)
        plt.plot(x, kde(x) * len(all_samples.flatten()) * (max_vel - min_vel) / 50, 'r-', linewidth=2)

        grad_y, grad_x = np.gradient(all_samples[0])
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)

        plt.subplot(132)
        plt.hist(grad_mag.flatten(), bins=50)
        plt.title("Gradient Magnitude Distribution")
        plt.xlabel("Gradient Magnitude")
        plt.ylabel("Frequency")

        plt.subplot(133)
        layer_profile = np.mean(all_samples, axis=(0, 2))
        plt.plot(layer_profile, np.arange(len(layer_profile)))
        plt.title("Average Horizontal Layer Profile")
        plt.xlabel("Average Velocity")
        plt.ylabel("Depth")
        plt.gca().invert_yaxis()

        os.makedirs(CONFIG['output_dir'] / 'visualizations', exist_ok=True)
        plt.tight_layout()
        plt.savefig(CONFIG['output_dir'] / 'visualizations' / "velocity_analysis.png")
        plt.show()

        plt.figure(figsize=(15, 5 * min(3, len(all_samples))))
        for i in range(min(3, len(all_samples))):
            plt.subplot(min(3, len(all_samples)), 3, i*3 + 1)
            plt.imshow(all_samples[i], cmap='magma', origin='lower')
            plt.title(f"Velocity Model #{i+1}")
            plt.colorbar()

            grad_y, grad_x = np.gradient(all_samples[i])
            edges = np.sqrt(grad_x**2 + grad_y**2)

            plt.subplot(min(3, len(all_samples)), 3, i*3 + 2)
            plt.imshow(edges, cmap='viridis', origin='lower')
            plt.title(f"Edge Detection #{i+1}")
            plt.colorbar()

            try:
                thresh = filters.threshold_otsu(all_samples[i])
            except:
                thresh = np.mean(all_samples[i])
            binary = (all_samples[i] > thresh).astype(float)

            plt.subplot(min(3, len(all_samples)), 3, i*3 + 3)
            plt.imshow(binary, cmap='gray', origin='lower')
            plt.title(f"Binary Threshold (t={thresh:.2f}) #{i+1}")

        plt.tight_layout()
        plt.savefig(CONFIG['output_dir'] / 'visualizations' / "velocity_examples.png")
        plt.show()

        
        
        return {
            'mean': mean_vel,
            'std': std_vel,
            'min': min_vel,
            'max': max_vel,
            'bimodal': bimodality_coefficient(all_samples.flatten()) > 0.555
        }
        
    def visualize_seismic_data(input_files, max_samples=5):
        """Visualize seismic data to understand input characteristics"""
        sample_files = input_files[:min(len(input_files), max_samples)]
        samples = []

        for f in sample_files:
            try:
                data = np.load(f, mmap_mode='r')
                samples.append(data[:1])
            except Exception as e:
                print(f"Error loading {f}: {e}")

        if not samples:
            print("No samples loaded for visualization")
            return

        all_samples = np.concatenate(samples, axis=0)
        print(f"Visualizing {all_samples.shape[0]} seismic data samples")

        plt.figure(figsize=(15, 4 * min(3, len(all_samples))))

        for i in range(min(3, len(all_samples))):
            plt.subplot(min(3, len(all_samples)), 3, i*3 + 1)
            plt.imshow(all_samples[i, 0], cmap='seismic', aspect='auto')
            plt.title(f"Raw Seismic (Ch 1) #{i+1}")
            plt.colorbar()

            time_steps = all_samples.shape[2]
            time = np.linspace(0, 1, time_steps)
            gain = (time ** 2)[:, np.newaxis]
            gained = all_samples[i, 0] * gain

            plt.subplot(min(3, len(all_samples)), 3, i*3 + 2)
            plt.imshow(gained, cmap='seismic', aspect='auto')
            plt.title(f"With Time Gain #{i+1}")
            plt.colorbar()

            freq = np.abs(fft.fft2(all_samples[i, 0]))
            freq = np.fft.fftshift(freq)

            plt.subplot(min(3, len(all_samples)), 3, i*3 + 3)
            plt.imshow(np.log(freq + 1), cmap='viridis', aspect='auto')
            plt.title(f"Frequency Spectrum #{i+1}")
            plt.colorbar()

        os.makedirs(CONFIG['output_dir'] / 'visualizations', exist_ok=True)
        plt.tight_layout()
        plt.savefig(CONFIG['output_dir'] / 'visualizations' / "seismic_visualization.png")
        plt.show()
    
    # Actually call the analysis functions with data from the configuration
    print("Running analysis and visualization tools...")
    
    # Find input and output files using the same pattern as in load_and_prepare_data
    input_files = sorted([f for f in CONFIG['data_dir'].rglob("*.npy") 
                         if 'seis' in f.name or 'data' in f.name])
    output_files = [Path(str(f).replace("seis", "vel").replace("data", "model")) 
                    for f in input_files]
    
    # Verify files exist (same as in load_and_prepare_data function)
    input_files = [f for f in input_files if f.exists()]
    output_files = [f for f in output_files if f.exists()]
    
    if len(input_files) > 0 and len(output_files) > 0:
        print(f"Found {len(input_files)} input files and {len(output_files)} output files for analysis")
        
        # Run the visualization functions
        visualize_seismic_data(input_files)
        velocity_stats = analyze_velocity_models(output_files)
        
        # Display the statistics summary
        if velocity_stats:
            print("\nVelocity Model Statistics Summary:")
            for key, value in velocity_stats.items():
                print(f"- {key}: {value}")
    else:
        print("No files found for analysis. Please check data paths in configuration.")


# Cell: Visualizing Ground Truth vs Predictions for Best Model

def visualize_model_predictions(model, data_loader, config, num_samples=5):
    """
    Generate visualizations comparing ground truth to model predictions
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader containing validation or test data
        config: Configuration dictionary
        num_samples: Number of samples to visualize
    """
    device = config['device']
    model.eval()
    
    # Get samples from the data loader
    samples = []
    targets = []
    
    with torch.no_grad():
        for x, y in data_loader:
            samples.append(x)
            targets.append(y)
            if len(samples) >= num_samples:
                break
    
    # Concatenate samples if we got multiple batches
    if len(samples) > 1:
        x_samples = torch.cat(samples, dim=0)[:num_samples]
        y_samples = torch.cat(targets, dim=0)[:num_samples]
    else:
        x_samples = samples[0][:num_samples]
        y_samples = targets[0][:num_samples]
    
    # Move to device
    x_samples = x_samples.to(device)
    y_samples = y_samples.to(device)
    
    # Generate predictions
    predictions = []
    with torch.no_grad():
        for i in range(num_samples):
            x = x_samples[i:i+1]  # Add batch dimension
            pred = model(x)
            predictions.append(pred.cpu())
    
    # Visualize comparisons
    fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5*num_samples))
    
    # Handle single sample case
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_samples):
        # Extract single sample, ground truth and prediction
        seismic = x_samples[i].cpu().numpy()
        truth = y_samples[i].cpu().numpy()
        pred = predictions[i].numpy()
        
        # Ensure we're working with the right dimensions
        if seismic.ndim > 3:
            seismic = seismic[0]  # Take first channel if multi-channel
        
        if truth.ndim > 2:
            if truth.shape[0] == 1:  # Single channel
                truth = truth[0]
            else:  # Multi-channel (for feature detection)
                truth = truth[0]  # Just take first channel for visualization
        
        if pred.ndim > 2:
            if pred.shape[0] == 1:  # Single channel
                pred = pred[0]
            elif isinstance(pred, dict):  # Feature detection output
                pred = pred['velocity'][0]
            else:
                pred = pred[0]  # Just take first channel
        
        # Calculate error
        error = np.abs(pred - truth)
        
        # Plot seismic data (first channel)
        axes[i, 0].imshow(seismic[0], cmap='seismic', aspect='auto')
        axes[i, 0].set_title(f"Input Seismic #{i+1}")
        axes[i, 0].axis('off')
        
        # Plot ground truth
        axes[i, 1].imshow(truth, cmap='magma', aspect='auto')
        axes[i, 1].set_title(f"Ground Truth #{i+1}")
        axes[i, 1].axis('off')
        
        # Plot prediction
        axes[i, 2].imshow(pred, cmap='magma', aspect='auto')
        axes[i, 2].set_title(f"Prediction #{i+1}")
        axes[i, 2].axis('off')
        
        # Plot error
        im = axes[i, 3].imshow(error, cmap='inferno', aspect='auto')
        axes[i, 3].set_title(f"Absolute Error #{i+1}")
        axes[i, 3].axis('off')
        
        # Add colorbar for error
        plt.colorbar(im, ax=axes[i, 3])
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs(config['output_dir'] / 'visualizations', exist_ok=True)
    fig_path = config['output_dir'] / 'visualizations' / 'ground_truth_vs_predictions.png'
    plt.savefig(fig_path)
    
    print(f"Visualizations saved to {fig_path}")
    
    # Display the figure if in notebook environment
    plt.show()
    
    return fig_path

# Now call the function with your best model and validation loader
print("Generating ground truth vs. prediction visualizations...")

# Get the best model
best_model = load_best_model(CONFIG)

# Get a data loader for visualization
# Re-use validation data loader from training
_, val_loader, _ = load_and_prepare_data(CONFIG)

# Generate visualizations
vis_path = visualize_model_predictions(best_model, val_loader, CONFIG, num_samples=5)

print(f"✅ Visualization completed and saved to: {vis_path}")

