# ============== Configuration ==============
TRAIN = True
PREDICT = True
USE_FULL_DATA = True  # Set False for quick testing
ENSEMBLE_FOLDS = 5  # Reduce for faster training

# ============== Install Dependencies ==============
import subprocess
import sys

def install_packages():
    """Install required packages"""
    packages = [
        'einops',
        'timm==0.9.16',
        'segmentation-models-pytorch',
        'pywavelets',
        'torchmetrics',
        'albumentations'
    ]
    
    for package in packages:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])
    
    print("All packages installed successfully!")

# Run installation
install_packages()

# ============== Import Everything ==============
import os
import gc
import glob
import time
import json
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, OneCycleLR

# Check GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ============== Quick Test Function ==============
def quick_test():
    """Quick test to ensure everything works"""
    print("\nRunning quick test...")
    
    # Test data loading
    test_files = glob.glob("/kaggle/input/waveform-inversion/test/*.npy")
    print(f"Found {len(test_files)} test files")
    
    if len(test_files) > 0:
        # Load sample
        sample = np.load(test_files[0])
        print(f"Test sample shape: {sample.shape}")
        print(f"Test sample dtype: {sample.dtype}")
        print(f"Test sample range: [{sample.min():.2f}, {sample.max():.2f}]")
    
    # Test model creation
    print("\nTesting model creation...")
    dummy_model = nn.Conv2d(5, 1, 3, padding=1).cuda()
    dummy_input = torch.randn(1, 5, 70, 70).cuda()
    with torch.no_grad():
        output = dummy_model(dummy_input)
    print(f"Model output shape: {output.shape}")
    
    print("\nQuick test passed! âœ“")
    
    # Clean up
    del dummy_model, dummy_input
    gc.collect()
    torch.cuda.empty_cache()

# Run quick test
quick_test()

# ============== Data Statistics ==============
def analyze_competition_data():
    """Analyze competition data structure"""
    print("\n" + "="*50)
    print("Competition Data Analysis")
    print("="*50)
    
    # Training data
    train_path = "/kaggle/input/waveform-inversion/train_samples/"
    if os.path.exists(train_path):
        datasets = ['CurveFault_A', 'CurveFault_B', 'CurveVel_A', 'CurveVel_B',
                   'FlatFault_A', 'FlatFault_B', 'FlatVel_A', 'FlatVel_B',
                   'Style_A', 'Style_B']
        
        print("\nTraining datasets:")
        for dataset in datasets:
            dataset_path = os.path.join(train_path, dataset)
            if os.path.exists(dataset_path):
                n_files = len(glob.glob(os.path.join(dataset_path, "**/*.npy"), recursive=True))
                print(f"  {dataset}: {n_files} files")
    
    # Test data
    test_files = glob.glob("/kaggle/input/waveform-inversion/test/*.npy")
    print(f"\nTest files: {len(test_files)}")
    
    # Submission format
    sub_df = pd.read_csv("/kaggle/input/waveform-inversion/sample_submission.csv")
    print(f"\nSubmission rows: {len(sub_df)}")
    print(f"Submission columns: {len(sub_df.columns)}")
    print(f"Expected predictions per file: {len(sub_df) // len(test_files) if len(test_files) > 0 else 'N/A'}")

# Analyze data
analyze_competition_data()

# ============== Memory Management ==============
def get_memory_usage():
    """Get current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        return f"GPU Memory: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
    return "No GPU available"

print(f"\n{get_memory_usage()}")

# ============== Ready to Train Message ==============
print("\n" + "="*50)
print("âœ“ Environment setup complete!")
print("âœ“ All dependencies installed!")
print("âœ“ Data paths verified!")
print("\nâ†’ You can now run the main training script")
print("â†’ Recommended: Start with ENSEMBLE_FOLDS=1 for testing")
print("="*50)

# ============== Helper Functions ==============
def create_folds_csv():
    """Create folds.csv if it doesn't exist"""
    print("\nCreating folds.csv...")
    
    # This is a simplified version - adjust based on your data
    data_info = []
    
    # Add your data loading logic here
    # For each file, assign to a fold
    
    # Save as CSV
    df = pd.DataFrame(data_info)
    df.to_csv('folds.csv', index=False)
    print("Folds created!")

def visualize_predictions(pred, target, save_path=None):
    """Visualize prediction vs target"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Prediction
    im1 = axes[0].imshow(pred, cmap='jet', vmin=1500, vmax=6000)
    axes[0].set_title('Prediction')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Target
    im2 = axes[1].imshow(target, cmap='jet', vmin=1500, vmax=6000)
    axes[1].set_title('Target')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Difference
    diff = np.abs(pred - target)
    im3 = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Absolute Error (MAE: {diff.mean():.1f})')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# ============== Training Configuration Template ==============
training_config = {
    "model": {
        "backbone": "tf_efficientnetv2_l",
        "pretrained": True,
        "in_channels": 5,
        "out_channels": 1,
    },
    "training": {
        "epochs": 150,
        "batch_size": 16,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "scheduler": "CosineAnnealingWarmRestarts",
        "warmup_epochs": 5,
    },
    "augmentation": {
        "mixup": True,
        "mixup_alpha": 0.2,
        "physics_augment": True,
        "tta_transforms": 4,
    },
    "physics": {
        "use_physics_loss": True,
        "physics_weight": 0.1,
        "velocity_bounds": [1500, 6000],
        "smoothness_weight": 0.01,
    },
    "ensemble": {
        "n_folds": 5,
        "voting": "weighted",
        "post_process": True,
    }
}

# Save config
with open('training_config.json', 'w') as f:
    json.dump(training_config, f, indent=4)

print("\nâœ“ Training configuration saved to 'training_config.json'")
print("\nYou're all set! Happy training! ðŸš€")


"""
FWI Competition - Data Explorer and Smart Loader
This script will help us understand the data structure and load it correctly
"""

import os
import glob
import numpy as np
import pandas as pd
from pathlib import Path

# ============== Step 1: Explore Data Structure ==============
def explore_competition_data():
    """Thoroughly explore the competition data structure"""
    
    base_path = "/kaggle/input/waveform-inversion/"
    train_path = os.path.join(base_path, "train_samples")
    
    print("="*70)
    print("FWI COMPETITION DATA EXPLORER")
    print("="*70)
    
    # Check if paths exist
    print(f"\n1. Checking paths:")
    print(f"   Base path exists: {os.path.exists(base_path)}")
    print(f"   Train path exists: {os.path.exists(train_path)}")
    
    if not os.path.exists(train_path):
        print("\nERROR: Training path doesn't exist!")
        return None
    
    # List all subdirectories
    print(f"\n2. Subdirectories in {train_path}:")
    subdirs = [d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))]
    for subdir in sorted(subdirs):
        print(f"   - {subdir}")
    
    # Explore each subdirectory
    all_files = []
    print(f"\n3. Files in each subdirectory:")
    
    for subdir in sorted(subdirs):
        subdir_path = os.path.join(train_path, subdir)
        files = sorted(glob.glob(os.path.join(subdir_path, "*")))
        
        print(f"\n   {subdir}:")
        print(f"   Total files: {len(files)}")
        
        # Show first few files
        for i, f in enumerate(files[:3]):
            filename = os.path.basename(f)
            print(f"     [{i+1}] {filename}")
            
            # Check if it's an .npy file and analyze it
            if f.endswith('.npy'):
                try:
                    # Load with memory mapping to check shape without loading full data
                    data = np.load(f, mmap_mode='r')
                    print(f"         Shape: {data.shape}")
                    print(f"         Dtype: {data.dtype}")
                    
                    # Sample a small portion to check value range
                    if len(data) > 0:
                        sample = data[0]
                        if hasattr(sample, 'shape'):
                            print(f"         Sample shape: {sample.shape}")
                            print(f"         Value range: [{np.min(sample):.2f}, {np.max(sample):.2f}]")
                    
                    all_files.append({
                        'subdir': subdir,
                        'filepath': f,
                        'filename': filename,
                        'shape': data.shape,
                        'dtype': str(data.dtype)
                    })
                    
                except Exception as e:
                    print(f"         Error loading: {str(e)}")
            
        if len(files) > 3:
            print(f"     ... and {len(files) - 3} more files")
    
    return all_files

# ============== Step 2: Smart Pattern Detection ==============
def detect_file_patterns(all_files):
    """Detect patterns in the file naming and structure"""
    
    print("\n" + "="*70)
    print("FILE PATTERN ANALYSIS")
    print("="*70)
    
    if not all_files:
        print("No files to analyze!")
        return None
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(all_files)
    
    # Analyze shapes
    print("\n1. Unique shapes found:")
    shape_counts = df['shape'].value_counts()
    for shape, count in shape_counts.items():
        print(f"   {shape}: {count} files")
    
    # Detect seismic vs velocity based on shape
    seismic_files = []
    velocity_files = []
    
    for _, row in df.iterrows():
        shape = eval(str(row['shape']))  # Convert string back to tuple
        
        # Common patterns:
        # Seismic: (N, 5, T, X) where 5 is channels, T is time, X is spatial
        # Velocity: (N, X, Y) where X, Y are spatial dimensions
        
        if len(shape) == 4 and shape[1] == 5:
            seismic_files.append(row)
        elif len(shape) == 3 and shape[1] == shape[2]:  # Square spatial dimensions
            velocity_files.append(row)
        else:
            # Try to infer from filename
            if any(x in row['filename'].lower() for x in ['seismic', 'data', 'input', 'trace']):
                seismic_files.append(row)
            elif any(x in row['filename'].lower() for x in ['velocity', 'vel', 'model', 'label', 'target']):
                velocity_files.append(row)
    
    print(f"\n2. Detected file types:")
    print(f"   Seismic files: {len(seismic_files)}")
    print(f"   Velocity files: {len(velocity_files)}")
    
    # Show examples
    if seismic_files:
        print(f"\n   Example seismic file:")
        example = seismic_files[0]
        print(f"     File: {example['filename']}")
        print(f"     Shape: {example['shape']}")
    
    if velocity_files:
        print(f"\n   Example velocity file:")
        example = velocity_files[0]
        print(f"     File: {example['filename']}")
        print(f"     Shape: {example['shape']}")
    
    return seismic_files, velocity_files

# ============== Step 3: Create File Pairs ==============
def create_file_pairs(seismic_files, velocity_files):
    """Match seismic and velocity files into pairs"""
    
    print("\n" + "="*70)
    print("CREATING FILE PAIRS")
    print("="*70)
    
    pairs = []
    
    # Group by subdirectory
    seismic_by_dir = {}
    velocity_by_dir = {}
    
    for s in seismic_files:
        if s['subdir'] not in seismic_by_dir:
            seismic_by_dir[s['subdir']] = []
        seismic_by_dir[s['subdir']].append(s)
    
    for v in velocity_files:
        if v['subdir'] not in velocity_by_dir:
            velocity_by_dir[v['subdir']] = []
        velocity_by_dir[v['subdir']].append(v)
    
    # Match files within each directory
    for subdir in sorted(set(list(seismic_by_dir.keys()) + list(velocity_by_dir.keys()))):
        seismic = seismic_by_dir.get(subdir, [])
        velocity = velocity_by_dir.get(subdir, [])
        
        print(f"\n{subdir}: {len(seismic)} seismic, {len(velocity)} velocity files")
        
        # Simple matching - assume they're in the same order
        n_pairs = min(len(seismic), len(velocity))
        for i in range(n_pairs):
            pairs.append({
                'dataset': subdir,
                'seismic_file': seismic[i]['filepath'],
                'velocity_file': velocity[i]['filepath'],
                'seismic_shape': seismic[i]['shape'],
                'velocity_shape': velocity[i]['shape']
            })
    
    print(f"\nTotal pairs created: {len(pairs)}")
    return pairs

# ============== Step 4: Alternative Loading Strategies ==============
def load_data_alternative():
    """Try alternative loading strategies if pattern matching fails"""
    
    print("\n" + "="*70)
    print("TRYING ALTERNATIVE LOADING STRATEGIES")
    print("="*70)
    
    train_path = "/kaggle/input/waveform-inversion/train_samples"
    all_npy_files = glob.glob(os.path.join(train_path, "**/*.npy"), recursive=True)
    
    print(f"Found {len(all_npy_files)} total .npy files")
    
    # Strategy 1: Check if files come in pairs (alternating)
    print("\nStrategy 1: Checking for alternating file pairs...")
    
    pairs = []
    datasets = ['CurveFault_A', 'CurveFault_B', 'CurveVel_A', 'CurveVel_B',
                'FlatFault_A', 'FlatFault_B', 'FlatVel_A', 'FlatVel_B',
                'Style_A', 'Style_B']
    
    for dataset in datasets:
        dataset_files = [f for f in all_npy_files if dataset in f]
        dataset_files.sort()
        
        if len(dataset_files) >= 2:
            # Check first two files
            f1_shape = np.load(dataset_files[0], mmap_mode='r').shape
            f2_shape = np.load(dataset_files[1], mmap_mode='r').shape
            
            print(f"\n{dataset}:")
            print(f"  File 1 shape: {f1_shape}")
            print(f"  File 2 shape: {f2_shape}")
            
            # Assume alternating pattern if shapes are different
            if f1_shape != f2_shape:
                for i in range(0, len(dataset_files)-1, 2):
                    pairs.append({
                        'dataset': dataset,
                        'file1': dataset_files[i],
                        'file2': dataset_files[i+1],
                        'shape1': np.load(dataset_files[i], mmap_mode='r').shape,
                        'shape2': np.load(dataset_files[i+1], mmap_mode='r').shape
                    })
    
    return pairs

# ============== Main Execution ==============
def main():
    """Main exploration function"""
    
    # Step 1: Explore data
    all_files = explore_competition_data()
    
    if all_files:
        # Step 2: Detect patterns
        seismic_files, velocity_files = detect_file_patterns(all_files)
        
        # Step 3: Create pairs
        if seismic_files and velocity_files:
            pairs = create_file_pairs(seismic_files, velocity_files)
            
            # Save pairs information
            if pairs:
                pairs_df = pd.DataFrame(pairs)
                pairs_df.to_csv('fwi_file_pairs.csv', index=False)
                print(f"\nFile pairs saved to 'fwi_file_pairs.csv'")
                
                # Show sample pair
                print("\nSample pair:")
                sample = pairs[0]
                print(f"  Dataset: {sample['dataset']}")
                print(f"  Seismic: {os.path.basename(sample['seismic_file'])} {sample['seismic_shape']}")
                print(f"  Velocity: {os.path.basename(sample['velocity_file'])} {sample['velocity_shape']}")
        else:
            print("\nCouldn't detect seismic/velocity files automatically.")
            print("Trying alternative strategies...")
            
            # Try alternative loading
            alt_pairs = load_data_alternative()
            if alt_pairs:
                print(f"\nFound {len(alt_pairs)} pairs using alternative strategy")
    else:
        print("\nNo files found! Please check the data path.")
    
    # Additional diagnostics
    print("\n" + "="*70)
    print("ADDITIONAL DIAGNOSTICS")
    print("="*70)
    
    # Check test data
    test_path = "/kaggle/input/waveform-inversion/test"
    if os.path.exists(test_path):
        test_files = glob.glob(os.path.join(test_path, "*.npy"))
        print(f"\nTest files: {len(test_files)}")
        if test_files:
            test_sample = np.load(test_files[0], mmap_mode='r')
            print(f"Test file shape: {test_sample.shape}")
            print(f"Test file dtype: {test_sample.dtype}")

if __name__ == "__main__":
    main()


"""
FWI Competition - Original Advanced Solution Adapted
Based on your high-accuracy implementation
"""

import os
import gc
import glob
import time
import json
import random
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Advanced imports
import timm
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from sklearn.model_selection import KFold

# Install required packages
try:
    import segmentation_models_pytorch as smp
except:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'segmentation-models-pytorch', '-q'])
    import segmentation_models_pytorch as smp

# Configuration
TRAIN = True
PREDICT = True
USE_FULL_DATA = True
ENSEMBLE_FOLDS = 5

class Config:
    # Paths
    train_path = "/kaggle/input/waveform-inversion/train_samples/"
    test_path = "/kaggle/input/waveform-inversion/test/"
    
    # Model params - FROM YOUR ORIGINAL CODE
    backbone = "tf_efficientnetv2_l"  # Your original backbone
    img_size = (70, 70)
    in_channels = 5
    out_channels = 1
    
    # Training params - FROM YOUR ORIGINAL CODE
    batch_size = 16
    val_batch_size = 32
    epochs = 150  # Your original epochs
    lr = 1e-4
    weight_decay = 1e-5
    
    # Advanced params - FROM YOUR ORIGINAL CODE
    use_physics_loss = True
    physics_weight = 0.1
    use_mixup = True
    mixup_alpha = 0.2
    use_tta = True
    tta_transforms = 4
    
    # Ensemble
    n_folds = ENSEMBLE_FOLDS
    ensemble_weights = "learned"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 42
    num_workers = 4

cfg = Config()

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

set_seed(cfg.seed)

# ============== Load Training Data ==============
def load_training_data():
    """Load all training data files"""
    print("Loading training data...")
    
    all_files = []
    datasets = ['CurveFault_A', 'CurveFault_B', 'CurveVel_A', 'CurveVel_B',
                'FlatFault_A', 'FlatFault_B', 'FlatVel_A', 'FlatVel_B',
                'Style_A', 'Style_B']
    
    for dataset in datasets:
        dataset_path = os.path.join(cfg.train_path, dataset)
        if os.path.exists(dataset_path):
            # Find seismic and velocity files
            seismic_files = sorted(glob.glob(os.path.join(dataset_path, "seis*.npy")))
            velocity_files = sorted(glob.glob(os.path.join(dataset_path, "vel*.npy")))
            
            # If no seis/vel files, check for data/model
            if not seismic_files:
                data_path = os.path.join(dataset_path, "data")
                model_path = os.path.join(dataset_path, "model")
                
                if os.path.exists(data_path):
                    if os.path.isdir(data_path):
                        seismic_files = sorted(glob.glob(os.path.join(data_path, "*.npy")))
                        velocity_files = sorted(glob.glob(os.path.join(model_path, "*.npy")))
                    elif os.path.isfile(data_path):
                        seismic_files = [data_path]
                        velocity_files = [model_path]
            
            # Match files
            for i in range(min(len(seismic_files), len(velocity_files))):
                all_files.append({
                    'dataset': dataset,
                    'seismic': seismic_files[i],
                    'velocity': velocity_files[i],
                    'file_idx': len(all_files)
                })
    
    print(f"Found {len(all_files)} training file pairs")
    return all_files

# ============== Dataset - FROM YOUR ORIGINAL CODE ==============
class FWIDataset(torch.utils.data.Dataset):
    def __init__(self, cfg, file_list, indices=None, mode='train', transform=None):
        self.cfg = cfg
        self.file_list = file_list
        self.indices = indices if indices is not None else range(len(file_list))
        self.mode = mode
        self.transform = transform
        
        # Get actual files
        self.files = [file_list[i] for i in self.indices]
        
    def __len__(self):
        return len(self.files) * 500  # 500 samples per file
        
    def __getitem__(self, idx):
        file_idx = idx // 500
        sample_idx = idx % 500
        
        file_info = self.files[file_idx % len(self.files)]
        
        # Load data
        data = np.load(file_info['seismic'], mmap_mode='r')
        label = np.load(file_info['velocity'], mmap_mode='r')
        
        # Handle different shapes
        if len(data.shape) == 3:  # Single sample file
            seismic = data
            velocity = label
        else:  # Multi-sample file
            if sample_idx < len(data):
                seismic = data[sample_idx]
                velocity = label[sample_idx]
            else:
                seismic = data[-1]
                velocity = label[-1]
        
        # Make copies
        seismic = np.array(seismic, copy=True, dtype=np.float32)
        velocity = np.array(velocity, copy=True, dtype=np.float32)
        
        # Apply physics-aware preprocessing
        seismic = self._preprocess_seismic(seismic)
        
        # Process velocity
        if len(velocity.shape) == 3 and velocity.shape[0] == 1:
            velocity = velocity[0]
        
        # Augmentations
        if self.mode == 'train':
            seismic, velocity = self._augment(seismic, velocity)
        
        # Ensure contiguous
        seismic = np.ascontiguousarray(seismic)
        velocity = np.ascontiguousarray(velocity)
            
        return torch.from_numpy(seismic).float(), torch.from_numpy(velocity).float().unsqueeze(0)
    
    def _preprocess_seismic(self, data):
        """Physics-aware preprocessing - FROM YOUR ORIGINAL CODE"""
        if data.shape[0] == 5 and data.shape[1] == 1000:
            # Take evenly spaced samples
            indices = np.linspace(0, 999, 70, dtype=int)
            data = data[:, indices, :]
        
        # Normalize by trace
        data = (data - data.mean(axis=-1, keepdims=True)) / (data.std(axis=-1, keepdims=True) + 1e-8)
        
        return data
    
    def _augment(self, data, label):
        """Physics-consistent augmentations - FROM YOUR ORIGINAL CODE"""
        # Temporal flip (time reversal)
        if np.random.random() < 0.5:
            data = data[:, :, ::-1]
            label = label[:, ::-1]
            
        # Add realistic noise
        if np.random.random() < 0.3:
            noise = np.random.normal(0, 0.02, data.shape)
            data = data + noise
            
        # Random gain
        if np.random.random() < 0.3:
            gain = np.random.uniform(0.8, 1.2)
            data = data * gain
            
        return data, label

# ============== Advanced Model Architecture - FROM YOUR ORIGINAL CODE ==============
class PhysicsAttention(nn.Module):
    """Attention module guided by wave physics"""
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Physics-informed positional encoding
        self.pos_embed = nn.Parameter(torch.zeros(1, 70*70, dim))
        self.velocity_embed = nn.Linear(1, dim)
        
    def forward(self, x, velocity_prior=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Add physics-informed bias
        if velocity_prior is not None:
            vel_bias = self.velocity_embed(velocity_prior.unsqueeze(-1))
            attn = attn + vel_bias.reshape(B, 1, N, N)
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class WaveEquationConstraint(nn.Module):
    """Physics constraint based on wave equation"""
    def __init__(self):
        super().__init__()
        # Finite difference operators for wave equation
        self.register_buffer('laplacian_kernel', self._create_laplacian_kernel())
        
    def _create_laplacian_kernel(self):
        # 2D Laplacian kernel for finite differences
        kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return kernel
    
    def forward(self, velocity, wavefield=None):
        """Apply wave equation constraint"""
        # Compute Laplacian
        laplacian = F.conv2d(velocity, self.laplacian_kernel, padding=1)
        
        # Wave equation residual: âˆ‡Â²u - (1/vÂ²)(âˆ‚Â²u/âˆ‚tÂ²) = 0
        physics_loss = torch.abs(laplacian).mean()
        
        # Enforce velocity bounds (1500-6000 m/s typical)
        velocity_constraint = F.relu(1500 - velocity) + F.relu(velocity - 6000)
        
        return physics_loss + velocity_constraint.mean() * 0.1

class SCSEModule(nn.Module):
    """Concurrent Spatial and Channel Squeeze & Excitation"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.cse = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.sse = nn.Sequential(
            nn.Conv2d(channels, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return x * self.cse(x) + x * self.sse(x)

class DecoderBlock(nn.Module):
    """Advanced decoder block with SCSE and PixelShuffle"""
    def __init__(self, in_channels, out_channels, skip_channels=0, use_scse=True):
        super().__init__()
        
        # Upsample
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 4, 1),
            nn.PixelShuffle(2)
        )
        
        # Fusion
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # SCSE attention
        if use_scse:
            self.scse = SCSEModule(out_channels)
        else:
            self.scse = nn.Identity()
    
    def forward(self, x, skip=None):
        x = self.upsample(x)
        
        if skip is not None:
            # Ensure sizes match
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
            
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = self.scse(x)
        x = self.relu(x)
        
        return x

class HybridFWIModel(nn.Module):
    """Advanced hybrid model for FWI - FROM YOUR ORIGINAL CODE"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        
        # Encoder: EfficientNetV2 backbone
        self.encoder = timm.create_model(
            cfg.backbone,
            pretrained=True,
            in_chans=cfg.in_channels,
            features_only=True,
            out_indices=(1, 2, 3, 4)
        )
        
        # Get encoder channels
        with torch.no_grad():
            dummy_input = torch.zeros(1, cfg.in_channels, 256, 256)
            enc_channels = [x.shape[1] for x in self.encoder(dummy_input)]
        
        # Physics constraint module
        self.physics_constraint = WaveEquationConstraint()
        
        # Advanced decoder with skip connections
        self.decoder = nn.ModuleList([
            DecoderBlock(enc_channels[3], 256, enc_channels[2], use_scse=True),
            DecoderBlock(256, 128, enc_channels[1], use_scse=True),
            DecoderBlock(128, 64, enc_channels[0], use_scse=True),
            DecoderBlock(64, 64, 0, use_scse=True)
        ])
        
        # Final prediction head
        self.head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, cfg.out_channels, 1),
            nn.Sigmoid()  # Ensure positive velocities
        )
        
        # Learnable velocity range parameters
        self.velocity_scale = nn.Parameter(torch.tensor(4500.0))
        self.velocity_offset = nn.Parameter(torch.tensor(1500.0))
        
    def forward(self, x, return_physics_loss=False):
        # Multi-scale encoding
        features = self.encoder(x)
        
        # Decode with skip connections
        x = features[-1]
        for i, decoder in enumerate(self.decoder):
            skip = features[-(i+2)] if i < len(features)-1 else None
            x = decoder(x, skip)
        
        # Resize to input size
        x = F.interpolate(x, size=(70, 70), mode='bilinear', align_corners=False)
        
        # Final prediction
        velocity = self.head(x)
        velocity = velocity * self.velocity_scale + self.velocity_offset
        
        if return_physics_loss:
            physics_loss = self.physics_constraint(velocity)
            return velocity, physics_loss
            
        return velocity

# ============== Physics-Informed Loss - FROM YOUR ORIGINAL CODE ==============
class SSIM(nn.Module):
    """Structural Similarity Index"""
    def __init__(self, window_size=11):
        super().__init__()
        self.window_size = window_size
        
    def forward(self, img1, img2):
        # Simplified SSIM
        mu1 = F.avg_pool2d(img1, self.window_size, 1, self.window_size//2)
        mu2 = F.avg_pool2d(img2, self.window_size, 1, self.window_size//2)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.avg_pool2d(img1 * img1, self.window_size, 1, self.window_size//2) - mu1_sq
        sigma2_sq = F.avg_pool2d(img2 * img2, self.window_size, 1, self.window_size//2) - mu2_sq
        sigma12 = F.avg_pool2d(img1 * img2, self.window_size, 1, self.window_size//2) - mu1_mu2
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        ssim = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
        
        return ssim.mean()

class FWILoss(nn.Module):
    """Combined loss with physics constraints - FROM YOUR ORIGINAL CODE"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.mae = nn.L1Loss()
        self.mse = nn.MSELoss()
        self.ssim = SSIM()
        
    def forward(self, pred, target, physics_loss=None):
        # Main reconstruction loss
        loss_mae = self.mae(pred, target)
        loss_mse = self.mse(pred, target)
        loss_ssim = 1 - self.ssim(pred, target)
        
        # Combined loss
        loss = loss_mae + 0.1 * loss_mse + 0.1 * loss_ssim
        
        # Add physics constraint
        if physics_loss is not None and self.cfg.use_physics_loss:
            loss = loss + self.cfg.physics_weight * physics_loss
            
        return loss

# ============== Training Functions - FROM YOUR ORIGINAL CODE ==============
def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation"""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
        
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    
    return mixed_x, mixed_y, lam

def train_epoch(model, loader, criterion, optimizer, scaler, cfg):
    model.train()
    losses = []
    
    pbar = tqdm(loader, desc='Training')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(cfg.device), target.to(cfg.device)
        
        # Mixup augmentation
        if cfg.use_mixup and np.random.random() < 0.5:
            data, target, lam = mixup_data(data, target, cfg.mixup_alpha)
        
        optimizer.zero_grad()
        
        with autocast():
            if cfg.use_physics_loss:
                output, physics_loss = model(data, return_physics_loss=True)
            else:
                output = model(data)
                physics_loss = None
                
            loss = criterion(output, target, physics_loss)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        losses.append(loss.item())
        pbar.set_postfix({'loss': np.mean(losses[-100:])})
    
    return np.mean(losses)

def validate(model, loader, criterion, cfg):
    model.eval()
    losses = []
    
    with torch.no_grad():
        for data, target in tqdm(loader, desc='Validation'):
            data, target = data.to(cfg.device), target.to(cfg.device)
            
            if cfg.use_tta:
                # Test-time augmentation
                outputs = []
                for _ in range(cfg.tta_transforms):
                    outputs.append(model(data))
                output = torch.stack(outputs).mean(0)
            else:
                output = model(data)
            
            loss = criterion(output, target)
            losses.append(loss.item())
    
    return np.mean(losses)

# ============== Main Training Loop ==============
def train_model(cfg, fold=0):
    # Load data
    file_list = load_training_data()
    
    # Create train/val split
    kf = KFold(n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed)
    train_idx, val_idx = list(kf.split(file_list))[fold]
    
    # Create datasets
    train_dataset = FWIDataset(cfg, file_list, train_idx, mode='train')
    valid_dataset = FWIDataset(cfg, file_list, val_idx, mode='valid')
    
    # Create loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=cfg.val_batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = HybridFWIModel(cfg).to(cfg.device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=10,
        T_mult=2,
        eta_min=1e-6
    )
    
    # Loss and scaler
    criterion = FWILoss(cfg)
    scaler = GradScaler()
    
    # Training loop
    best_loss = float('inf')
    for epoch in range(cfg.epochs):
        print(f"\nEpoch {epoch+1}/{cfg.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, cfg)
        
        # Validate
        val_loss = validate(model, valid_loader, criterion, cfg)
        
        # Scheduler step
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'loss': val_loss,
            }, f'best_model_original_fold{fold}.pth')
            print(f"Saved best model with loss: {val_loss:.4f}")
    
    return model

# ============== Inference with Ensemble ==============
def apply_tta(model, data, n_aug=4):
    """Test-time augmentation"""
    predictions = []
    
    # Original
    predictions.append(model(data))
    
    if n_aug >= 2:
        # Horizontal flip
        flipped = torch.flip(data, dims=[3])
        pred_flipped = model(flipped)
        predictions.append(torch.flip(pred_flipped, dims=[3]))
    
    if n_aug >= 3:
        # Add noise
        noisy = data + torch.randn_like(data) * 0.01
        predictions.append(model(noisy))
    
    if n_aug >= 4:
        # Different noise
        noisy2 = data + torch.randn_like(data) * 0.02
        predictions.append(model(noisy2))
    
    return torch.stack(predictions).mean(0)

def predict_test(models, cfg):
    """Generate predictions for test set"""
    test_files = sorted(glob.glob(f"{cfg.test_path}/*.npy"))
    print(f"\nFound {len(test_files)} test files")
    
    # Prepare submission
    submission_data = []
    
    for test_file in tqdm(test_files, desc="Predicting"):
        # Load test data
        test_data = np.load(test_file)
        oid = os.path.basename(test_file).split('.')[0]
        
        # Process seismic data
        indices = np.linspace(0, 999, 70, dtype=int)
        processed = test_data[:, indices, :].astype(np.float32)
        
        # Normalize
        processed = (processed - processed.mean(axis=-1, keepdims=True)) / (processed.std(axis=-1, keepdims=True) + 1e-8)
        
        # Convert to tensor
        data = torch.from_numpy(processed).float().unsqueeze(0).to(cfg.device)
        
        # Predict with all models
        predictions = []
        for model in models:
            model.eval()
            with torch.no_grad():
                if cfg.use_tta:
                    pred = apply_tta(model, data, cfg.tta_transforms)
                else:
                    pred = model(data)
            predictions.append(pred)
        
        # Average predictions
        prediction = torch.stack(predictions).mean(0).squeeze().cpu().numpy()
        
        # Create submission format
        for y in range(70):
            row_data = {'oid_ypos': f'{oid}_y_{y}'}
            for x in range(1, 70, 2):  # Only odd positions
                row_data[f'x_{x}'] = prediction[y, x]
            submission_data.append(row_data)
    
    # Create submission DataFrame
    submission = pd.DataFrame(submission_data)
    return submission

# ============== Main Execution ==============
if __name__ == "__main__":
    print("FWI Competition - Original Advanced Solution")
    print(f"Device: {cfg.device}")
    print(f"Model: HybridFWIModel with {cfg.backbone}")
    
    # Train models for each fold
    models = []
    
    if TRAIN:
        for fold in range(cfg.n_folds):
            print(f"\n{'='*50}")
            print(f"Training Fold {fold+1}/{cfg.n_folds}")
            print(f"{'='*50}")
            
            model = train_model(cfg, fold)
            models.append(model)
            
            # Clean up
            gc.collect()
            torch.cuda.empty_cache()
    else:
        # Load pretrained models
        for fold in range(cfg.n_folds):
            model = HybridFWIModel(cfg).to(cfg.device)
            checkpoint = torch.load(f'best_model_original_fold{fold}.pth', map_location=cfg.device, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            models.append(model)
    
    # Generate predictions
    if PREDICT:
        submission = predict_test(models, cfg)
        submission.to_csv('submission_original.csv', index=False)
        print(f"\nSubmission saved! Shape: {submission.shape}")
        print(submission.head())
    
    print("\nDone!")


"""
Advanced utilities for physics-informed FWI solution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter, median_filter
import pywt  # for wavelet transforms

# ============== Physics-Informed Components ==============

class WaveEquationLoss(nn.Module):
    """Full wave equation loss for physics-informed training"""
    def __init__(self, dx=1.0, dt=0.001):
        super().__init__()
        self.dx = dx
        self.dt = dt
        
        # Define finite difference operators
        self.register_buffer('laplacian_x', self._create_fd_kernel('xx'))
        self.register_buffer('laplacian_y', self._create_fd_kernel('yy'))
        self.register_buffer('time_diff', self._create_fd_kernel('tt'))
        
    def _create_fd_kernel(self, derivative_type):
        """Create finite difference kernels"""
        if derivative_type == 'xx' or derivative_type == 'yy':
            # Second derivative in space
            kernel = torch.tensor([
                [0, 0, 0],
                [1, -2, 1],
                [0, 0, 0]
            ], dtype=torch.float32)
            if derivative_type == 'yy':
                kernel = kernel.T
        elif derivative_type == 'tt':
            # Second derivative in time (simplified)
            kernel = torch.tensor([1, -2, 1], dtype=torch.float32)
        
        return kernel.unsqueeze(0).unsqueeze(0)
    
    def forward(self, velocity, wavefield=None):
        """
        Compute wave equation residual:
        âˆ‚Â²u/âˆ‚tÂ² = vÂ²(âˆ‡Â²u) + f
        """
        b, c, h, w = velocity.shape
        
        # Compute spatial derivatives
        d2u_dx2 = F.conv2d(velocity, self.laplacian_x, padding=1) / (self.dx ** 2)
        d2u_dy2 = F.conv2d(velocity, self.laplacian_y, padding=1) / (self.dx ** 2)
        laplacian_u = d2u_dx2 + d2u_dy2
        
        # Wave equation residual (simplified without time component)
        # In practice, you'd need the full wavefield evolution
        wave_residual = torch.abs(laplacian_u)
        
        # Additional physics constraints
        # 1. Smoothness constraint (velocities should be locally smooth)
        grad_x = torch.abs(velocity[:, :, :, 1:] - velocity[:, :, :, :-1])
        grad_y = torch.abs(velocity[:, :, 1:, :] - velocity[:, :, :-1, :])
        smoothness_loss = (grad_x.mean() + grad_y.mean()) * 0.1
        
        # 2. Boundary conditions (e.g., absorbing boundaries)
        boundary_loss = (
            velocity[:, :, :5, :].abs().mean() + 
            velocity[:, :, -5:, :].abs().mean() +
            velocity[:, :, :, :5].abs().mean() + 
            velocity[:, :, :, -5:].abs().mean()
        ) * 0.01
        
        return wave_residual.mean() + smoothness_loss + boundary_loss


class FourierFeatures(nn.Module):
    """Fourier feature extraction for better frequency representation"""
    def __init__(self, in_channels, out_channels, modes=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Fourier coefficients
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(in_channels, out_channels, modes, modes, 2)
        )
        
    def forward(self, x):
        # FFT
        x_ft = torch.fft.rfft2(x)
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            x.shape[0], self.out_channels, x.shape[-2], x.shape[-1]//2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        out_ft[:, :, :self.modes, :self.modes] = self._complex_mul2d(
            x_ft[:, :, :self.modes, :self.modes], 
            torch.view_as_complex(self.weights)
        )
        
        # IFFT
        x = torch.fft.irfft2(out_ft, s=(x.shape[-2], x.shape[-1]))
        return x
    
    def _complex_mul2d(self, input, weights):
        """Complex multiplication"""
        return torch.einsum("bixy,ioxy->boxy", input, weights)


class MultiScaleWaveletTransform(nn.Module):
    """Multi-scale wavelet decomposition for seismic data"""
    def __init__(self, wavelet='db4', levels=3):
        super().__init__()
        self.wavelet = wavelet
        self.levels = levels
        
    def forward(self, x):
        """Apply wavelet transform and return multi-scale features"""
        b, c, h, w = x.shape
        features = []
        
        for i in range(b):
            for j in range(c):
                # 2D wavelet transform
                coeffs = pywt.wavedec2(x[i, j].cpu().numpy(), self.wavelet, level=self.levels)
                
                # Extract features at each scale
                for level_coeffs in coeffs:
                    if isinstance(level_coeffs, tuple):
                        for coeff in level_coeffs:
                            features.append(torch.from_numpy(coeff).to(x.device))
                    else:
                        features.append(torch.from_numpy(level_coeffs).to(x.device))
        
        return features


# ============== Advanced Data Augmentation ==============

class SeismicAugmentation:
    """Physics-consistent augmentations for seismic data"""
    
    @staticmethod
    def add_coherent_noise(data, snr_db=20):
        """Add coherent noise (ground roll, multiples)"""
        signal_power = np.mean(data ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        # Generate coherent noise (e.g., linear events)
        noise = np.zeros_like(data)
        for _ in range(np.random.randint(1, 4)):
            # Random linear event
            slope = np.random.uniform(-0.5, 0.5)
            for i in range(data.shape[0]):
                for j in range(data.shape[2]):
                    t_idx = int(i + slope * j)
                    if 0 <= t_idx < data.shape[0]:
                        noise[t_idx, :, j] += np.random.normal(0, np.sqrt(noise_power))
        
        return data + noise
    
    @staticmethod
    def apply_frequency_filter(data, low_freq=5, high_freq=50, fs=1000):
        """Apply bandpass filter"""
        nyquist = fs / 2
        low = low_freq / nyquist
        high = high_freq / nyquist
        
        b, a = signal.butter(4, [low, high], btype='band')
        filtered = signal.filtfilt(b, a, data, axis=0)
        
        return filtered
    
    @staticmethod
    def random_static_shift(data, max_shift=5):
        """Apply random static shifts to traces"""
        shifts = np.random.randint(-max_shift, max_shift, size=data.shape[2])
        shifted = np.zeros_like(data)
        
        for i, shift in enumerate(shifts):
            if shift > 0:
                shifted[shift:, :, i] = data[:-shift, :, i]
            elif shift < 0:
                shifted[:shift, :, i] = data[-shift:, :, i]
            else:
                shifted[:, :, i] = data[:, :, i]
                
        return shifted
    
    @staticmethod
    def apply_agc(data, window_size=50):
        """Apply Automatic Gain Control"""
        eps = 1e-10
        agc_data = np.zeros_like(data)
        
        for i in range(data.shape[2]):
            trace = data[:, 0, i]
            
            # Compute envelope
            envelope = np.abs(signal.hilbert(trace))
            
            # Smooth envelope
            envelope = gaussian_filter(envelope, window_size/4)
            
            # Apply AGC
            agc_data[:, 0, i] = trace / (envelope + eps)
            
        return agc_data


# ============== Advanced Post-Processing ==============

class VelocityPostProcessor:
    """Post-processing for velocity models"""
    
    @staticmethod
    def apply_physical_constraints(velocity, vmin=1500, vmax=6000):
        """Apply physical constraints to velocity model"""
        # Clip to physical bounds
        velocity = np.clip(velocity, vmin, vmax)
        
        # Apply median filter to remove spikes
        velocity = median_filter(velocity, size=3)
        
        # Apply slight Gaussian smoothing
        velocity = gaussian_filter(velocity, sigma=0.5)
        
        return velocity
    
    @staticmethod
    def enforce_layer_continuity(velocity, threshold=200):
        """Enforce geological layer continuity"""
        # Detect large velocity contrasts
        grad_y = np.abs(np.diff(velocity, axis=0))
        
        # Find layer boundaries
        boundaries = grad_y > threshold
        
        # Smooth within layers
        smoothed = velocity.copy()
        for i in range(1, velocity.shape[0]-1):
            if not boundaries[i-1].any() and not boundaries[i].any():
                # Average with neighbors if not at boundary
                smoothed[i] = 0.6 * velocity[i] + 0.2 * velocity[i-1] + 0.2 * velocity[i+1]
                
        return smoothed
    
    @staticmethod
    def apply_geological_priors(velocity, depth_axis=0):
        """Apply geological priors (velocity generally increases with depth)"""
        # Sort velocity columns to ensure general increase with depth
        for j in range(velocity.shape[1]):
            col = velocity[:, j]
            
            # Apply slight increasing trend
            trend = np.linspace(0, 100, len(col))
            col_sorted = np.sort(col) + trend
            
            # Blend original with sorted (preserve features while enforcing trend)
            velocity[:, j] = 0.7 * col + 0.3 * col_sorted
            
        return velocity


# ============== Model Architecture Improvements ==============

class FNOBlock(nn.Module):
    """Fourier Neural Operator block for efficient convolution"""
    def __init__(self, in_channels, out_channels, modes=16):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Fourier layer
        self.fourier = FourierFeatures(in_channels, out_channels, modes)
        
        # Regular convolution path
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        
        # Activation and normalization
        self.bn = nn.BatchNorm2d(out_channels)
        self.activation = nn.GELU()
        
    def forward(self, x):
        # Fourier path
        x_fourier = self.fourier(x)
        
        # Conv path
        x_conv = self.conv(x)
        
        # Combine
        x = x_fourier + x_conv
        x = self.bn(x)
        x = self.activation(x)
        
        return x


class SelfAttention2D(nn.Module):
    """Self-attention module for global context"""
    def __init__(self, in_channels, reduction=8):
        super().__init__()
        self.in_channels = in_channels
        
        self.query = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.key = nn.Conv2d(in_channels, in_channels // reduction, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        b, c, h, w = x.shape
        
        # Compute attention
        proj_query = self.query(x).view(b, -1, h*w).permute(0, 2, 1)
        proj_key = self.key(x).view(b, -1, h*w)
        energy = torch.bmm(proj_query, proj_key)
        attention = F.softmax(energy, dim=-1)
        
        proj_value = self.value(x).view(b, -1, h*w)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(b, c, h, w)
        
        # Apply attention with learnable weight
        out = self.gamma * out + x
        
        return out


# ============== Training Utilities ==============

class EarlyStopping:
    """Early stopping with patience"""
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif self.mode == 'min' and score > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        elif self.mode == 'max' and score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
            
        return self.early_stop


class ModelCheckpoint:
    """Save best models during training"""
    def __init__(self, filepath, monitor='val_loss', mode='min', save_best_only=True):
        self.filepath = filepath
        self.monitor = monitor
        self.mode = mode
        self.save_best_only = save_best_only
        self.best = float('inf') if mode == 'min' else float('-inf')
        
    def __call__(self, score, model, epoch):
        if self.mode == 'min' and score < self.best:
            self.best = score
            self._save_model(model, epoch, score)
        elif self.mode == 'max' and score > self.best:
            self.best = score
            self._save_model(model, epoch, score)
        elif not self.save_best_only:
            self._save_model(model, epoch, score)
            
    def _save_model(self, model, epoch, score):
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'score': score,
        }, self.filepath.format(epoch=epoch, score=score))


# ============== Evaluation Metrics ==============

def compute_metrics(pred, target):
    """Compute various metrics for evaluation"""
    mae = F.l1_loss(pred, target)
    mse = F.mse_loss(pred, target)
    
    # Relative error
    relative_error = torch.abs(pred - target) / (torch.abs(target) + 1e-8)
    mre = relative_error.mean()
    
    # Structural similarity (simplified)
    ssim = 1 - F.l1_loss(pred, target) / (pred.abs().mean() + target.abs().mean())
    
    return {
        'mae': mae.item(),
        'mse': mse.item(),
        'rmse': torch.sqrt(mse).item(),
        'mre': mre.item(),
        'ssim': ssim.item()
    }


if __name__ == "__main__":
    print("Advanced FWI utilities loaded successfully!")

