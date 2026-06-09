# Cell 1: Import Libraries
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.parallel import DataParallel
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import warnings
import scipy
from scipy import signal
import math
from typing import Dict, List, Tuple
import time
import glob
import pickle
from PIL import Image
from skimage.transform import resize


# Cell 2: Configuration and Setup
# Set seeds for reproducibility
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  # for reproducibility

seed_everything()
warnings.filterwarnings('ignore')

# Check if CUDA is available and get GPU count
print(f"CUDA available: {torch.cuda.is_available()}")
num_gpus = torch.cuda.device_count()
print(f"Number of GPUs available: {num_gpus}")
for i in range(num_gpus):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Add spectrogram directory to configuration
CONFIG = {
    # Data parameters
    'target_freq': 100,  
    'window_size_seconds': 10,  
    'overlap_ratio': 0.5,
    'channels': ['Fp1', 'Fp2', 'F7', 'F8', 'F3', 'F4', 'T3', 'T4', 'C3', 'C4', 'T5', 'T6', 'P3', 'P4', 'O1', 'O2', 'Fz', 'Cz', 'Pz'],
    
    # Spectrogram parameters
    'use_spectrograms': True,
    'spec_height': 128,
    'spec_width': 256,
    
    # Training parameters
    'batch_size': 32,  # Reduced from 64 to 32 for training batches
    'num_epochs': 10,
    'learning_rate': 3e-4,
    'weight_decay': 1e-5,
    'fold_count': 3,
    'patience': 3,
    
    # Model parameters
    'model_name': 'MultimodalEEGNet',
    'dropout_rate': 0.5,
    'n_classes': 6,
    
    # Multi-GPU settings
    'use_multi_gpu': True,
    'num_workers': 1,  # Reduced to 1 worker
    'pin_memory': True,
    'mixed_precision': True,
    
    # Memory management
    'batch_processing_size': 100,  # Reduced from 500 to 100
    'max_file_time': 15,  # Reduced from 30 to 15 seconds
}

# Set up mixed precision training if available
if CONFIG['mixed_precision'] and torch.cuda.is_available():
    try:
        from torch.cuda.amp import autocast, GradScaler
        scaler = GradScaler()
        print("Mixed precision training enabled")
    except ImportError:
        CONFIG['mixed_precision'] = False
        print("Mixed precision training not available, disabled")

# Update paths to include spectrograms
INPUT_DIR = "/kaggle/input/hms-harmful-brain-activity-classification"
TRAIN_DIR = f"{INPUT_DIR}/train_eegs"
TEST_DIR = f"{INPUT_DIR}/test_eegs"
TRAIN_SPECTROGRAMS_DIR = f"{INPUT_DIR}/train_spectrograms" 
TEST_SPECTROGRAMS_DIR = f"{INPUT_DIR}/test_spectrograms"
OUTPUT_DIR = "/kaggle/working"
os.makedirs(OUTPUT_DIR, exist_ok=True)




# Cell 3: Data Loading and Exploration
# Load metadata
train_metadata = pd.read_csv(f"{INPUT_DIR}/train.csv")
test_metadata = pd.read_csv(f"{INPUT_DIR}/test.csv")
submission_df = pd.read_csv(f"{INPUT_DIR}/sample_submission.csv")

print(f"Train samples: {len(train_metadata)}")
print(f"Test samples: {len(test_metadata)}")

# Examine training data
print("Sample of training metadata:")
print(train_metadata.head())

# Check column names to avoid errors
print(f"\nAvailable columns: {train_metadata.columns.tolist()}")

# Determine the label column
if 'expert_consensus' in train_metadata.columns:
    label_column = 'expert_consensus'
elif 'label' in train_metadata.columns:
    label_column = 'label'
else:
    print("Warning: Could not find expected label column. Available columns:", train_metadata.columns.tolist())
    # Try to identify the most likely label column
    possible_label_columns = [col for col in train_metadata.columns if 'label' in col.lower() or 'class' in col.lower() or 'consensus' in col.lower()]
    if possible_label_columns:
        label_column = possible_label_columns[0]
        print(f"Using '{label_column}' as the label column")
    else:
        raise ValueError("No suitable label column found in the data")

# Create label mapping
labels_map = {label: idx for idx, label in enumerate(sorted(train_metadata[label_column].unique()))}
idx_to_label = {v: k for k, v in labels_map.items()}

print(f"\nClasses: {train_metadata[label_column].unique()}")
print(f"Class distribution: \n{train_metadata[label_column].value_counts()}")
print(f"Label mapping: {labels_map}")

# Update n_classes in CONFIG based on actual number of classes
CONFIG['n_classes'] = len(labels_map)
print(f"Number of classes: {CONFIG['n_classes']}")


# Cell 4: Processing Functions (EEG and Spectrogram)
# EEG data processing functions
def load_eeg_file(file_path):
    """Load EEG data from a file and return as a pandas DataFrame."""
    try:
        eeg_df = pd.read_parquet(file_path)
        return eeg_df
    except Exception as e:
        print(f"Error loading EEG file {file_path}: {e}")
        return None

def preprocess_eeg(eeg_df, target_freq=100):
    """Preprocess EEG data: handle missing values, filter, and downsample."""
    if eeg_df is None:
        return None
        
    # Determine if there's a time column, if not estimate the sampling frequency
    if 'time' in eeg_df.columns:
        time_diff = eeg_df['time'].diff().median()
        orig_freq = round(1 / time_diff)
    else:
        # Assume a standard sampling rate if time column not available
        orig_freq = 200  # Common EEG sampling frequency
        print(f"No time column found. Assuming original sampling frequency of {orig_freq} Hz.")
    
    # Fill missing values (if any)
    eeg_df = eeg_df.fillna(method='ffill').fillna(method='bfill')
    
    # Extract EEG signal columns, excluding non-EEG columns if present
    available_channels = [ch for ch in CONFIG['channels'] if ch in eeg_df.columns]
    if len(available_channels) < len(CONFIG['channels']):
        print(f"Warning: Only {len(available_channels)} out of {len(CONFIG['channels'])} channels found in data.")
    
    eeg_signals = eeg_df[available_channels].values
    
    # Apply bandpass filter (0.5-40 Hz) to remove noise - simplified for speed
    try:
        sos = signal.butter(4, [0.5, 40], 'bandpass', fs=orig_freq, output='sos')
        filtered_signals = np.zeros_like(eeg_signals)
        for i in range(eeg_signals.shape[1]):
            filtered_signals[:, i] = signal.sosfilt(sos, eeg_signals[:, i])
    except Exception as e:
        print(f"Error during filtering: {e}")
        filtered_signals = eeg_signals  # Use original signals if filtering fails
    
    # Downsample if needed
    if target_freq < orig_freq:
        # Calculate downsampling factor
        downsample_factor = orig_freq // target_freq
        filtered_signals = filtered_signals[::downsample_factor, :]
    
    return filtered_signals

def segment_eeg(eeg_data, window_size, overlap_ratio):
    """Segment EEG data into windows with overlap."""
    if eeg_data is None or eeg_data.shape[0] < window_size:
        return np.array([])
        
    step_size = int(window_size * (1 - overlap_ratio))
    segments = []
    
    # Calculate how many complete segments we can extract
    num_segments = (eeg_data.shape[0] - window_size) // step_size + 1
    
    # Limit to a maximum of 5 segments per recording to save memory
    max_segments = 5
    num_segments = min(num_segments, max_segments)
    
    for i in range(num_segments):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        segment = eeg_data[start_idx:end_idx, :]
        segments.append(segment)
    
    return np.array(segments)

# Spectrogram processing functions
def load_spectrogram(file_path):
    """Load spectrogram data."""
    if file_path is None:
        return None
        
    try:
        # Assuming the spectrograms are stored as parquet or image files
        if file_path.endswith('.parquet'):
            spec_df = pd.read_parquet(file_path)
            return spec_df
        elif file_path.endswith(('.png', '.jpg', '.jpeg')):
            # Load as image
            img = Image.open(file_path)
            # Convert to numpy array
            spec_array = np.array(img)
            return spec_array
        else:
            print(f"Unsupported spectrogram format: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading spectrogram {file_path}: {e}")
        return None

def preprocess_spectrogram(spec_data, target_height=128, target_width=256):
    """Preprocess spectrogram data."""
    if spec_data is None:
        # Return empty spectrogram if data is None
        return np.zeros((target_height, target_width))
        
    # Handle different input types
    if isinstance(spec_data, pd.DataFrame):
        # Convert dataframe to 2D array if needed
        # This depends on your spectrogram format
        spec_array = spec_data.values
    else:
        spec_array = spec_data
    
    # Ensure we have a 2D array
    if len(spec_array.shape) > 2:
        # If it's an RGB image, convert to grayscale
        from skimage.color import rgb2gray
        spec_array = rgb2gray(spec_array)
    
    # Resize if needed and if dimensions are valid
    try:
        if spec_array.shape[0] != target_height or spec_array.shape[1] != target_width:
            spec_array = resize(spec_array, (target_height, target_width), 
                              anti_aliasing=True, preserve_range=True)
    except Exception as e:
        print(f"Error resizing spectrogram: {e}")
        return np.zeros((target_height, target_width))
    
    # Normalize to [0, 1]
    try:
        min_val = np.min(spec_array)
        max_val = np.max(spec_array)
        if max_val > min_val:
            spec_array = (spec_array - min_val) / (max_val - min_val)
        else:
            spec_array = np.zeros_like(spec_array)
    except Exception as e:
        print(f"Error normalizing spectrogram: {e}")
        return np.zeros((target_height, target_width))
    
    return spec_array


class MultimodalDataset(Dataset):
    def __init__(self, eeg_files, spectrogram_files, labels=None, is_test=False):
        self.eeg_files = eeg_files
        self.spectrogram_files = spectrogram_files
        self.labels = labels
        self.is_test = is_test
        
        # Calculate window size in samples
        self.window_size = int(CONFIG['window_size_seconds'] * CONFIG['target_freq'])
        
        # Precompute the segments to save memory during training
        self.eeg_segments = []
        self.spectrogram_segments = []
        self.segment_labels = []
        self.segment_to_file_idx = []
        
        for i, (eeg_file, spec_file) in enumerate(tqdm(zip(eeg_files, spectrogram_files), 
                                                       desc="Processing files", 
                                                       total=len(eeg_files))):
            try:
                # Process EEG
                eeg_df = load_eeg_file(eeg_file)
                eeg_processed = preprocess_eeg(eeg_df, CONFIG['target_freq'])
                
                # Segment the EEG
                eeg_segments = segment_eeg(
                    eeg_processed, 
                    self.window_size, 
                    CONFIG['overlap_ratio']
                )
                
                # Process spectrogram if available
                spec_segments = []
                if CONFIG['use_spectrograms'] and spec_file is not None:
                    spec_data = load_spectrogram(spec_file)
                    if spec_data is not None:
                        spec_processed = preprocess_spectrogram(
                            spec_data, 
                            CONFIG['spec_height'], 
                            CONFIG['spec_width']
                        )
                        # Create a "segment" for each EEG segment
                        # In a real implementation, you may want to segment spectrograms too
                        spec_segments = [spec_processed] * len(eeg_segments)
                    else:
                        # Create empty spectrogram placeholders if loading failed
                        empty_spec = np.zeros((CONFIG['spec_height'], CONFIG['spec_width']))
                        spec_segments = [empty_spec] * len(eeg_segments)
                else:
                    # Create empty spectrogram placeholders if not using spectrograms
                    empty_spec = np.zeros((CONFIG['spec_height'], CONFIG['spec_width']))
                    spec_segments = [empty_spec] * len(eeg_segments)
                
                # Store segments and their corresponding labels
                for eeg_segment, spec_segment in zip(eeg_segments, spec_segments):
                    # Normalize each EEG segment
                    eeg_segment = (eeg_segment - np.mean(eeg_segment, axis=0)) / (np.std(eeg_segment, axis=0) + 1e-8)
                    
                    self.eeg_segments.append(eeg_segment)
                    self.spectrogram_segments.append(spec_segment)
                    
                    if not self.is_test:
                        self.segment_labels.append(labels[i])
                    self.segment_to_file_idx.append(i)
            except Exception as e:
                print(f"Error processing files {eeg_file} / {spec_file}: {e}")
        
        print(f"Created dataset with {len(self.eeg_segments)} segments from {len(eeg_files)} files")
    
    def __len__(self):
        return len(self.eeg_segments)
    
    def __getitem__(self, idx):
        eeg_segment = self.eeg_segments[idx]
        spec_segment = self.spectrogram_segments[idx]
        
        # Convert to tensors
        eeg_tensor = torch.tensor(eeg_segment, dtype=torch.float32)
        spec_tensor = torch.tensor(spec_segment, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        
        # Permute EEG for channel-first format
        eeg_tensor = eeg_tensor.permute(1, 0)  # [channels, time]
        if not self.is_test:
                    label = self.segment_labels[idx]
                    return (eeg_tensor, spec_tensor), label, self.segment_to_file_idx[idx]
        else:
            return (eeg_tensor, spec_tensor), self.segment_to_file_idx[idx]


# Cell 5: Memory-Optimized Multimodal Dataset Class
class BatchMultimodalDataset(Dataset):
    def __init__(self, eeg_files, spectrogram_files, labels=None, is_test=False):
        self.eeg_files = eeg_files
        self.spectrogram_files = spectrogram_files
        self.labels = labels
        self.is_test = is_test
        
        # Calculate window size in samples
        self.window_size = int(CONFIG['window_size_seconds'] * CONFIG['target_freq'])
        
        # Precompute the segments to save memory during training
        self.eeg_segments = []
        self.spectrogram_segments = []
        self.segment_labels = []
        self.segment_to_file_idx = []
        
        # Track statistics
        skipped_files = 0
        processed_files = 0
        
        # For batch processing, process files with a timeout
        for i, (eeg_file, spec_file) in enumerate(tqdm(zip(eeg_files, spectrogram_files), 
                                                      desc="Processing files", 
                                                      total=len(eeg_files))):
            start_time = time.time()
            try:
                # Check if we should skip this file due to timeout
                if time.time() - start_time > CONFIG['max_file_time']:
                    print(f"Timeout skipping file {i}")
                    skipped_files += 1
                    continue
                
                # Load and process EEG
                eeg_df = load_eeg_file(eeg_file)
                if eeg_df is None:
                    skipped_files += 1
                    continue
                    
                eeg_processed = preprocess_eeg(eeg_df, CONFIG['target_freq'])
                if eeg_processed is None:
                    skipped_files += 1
                    continue
                
                # Check for timeout again
                if time.time() - start_time > CONFIG['max_file_time']:
                    print(f"Processing taking too long for file {i}")
                    skipped_files += 1
                    continue
                
                # Segment the EEG
                eeg_segments = segment_eeg(eeg_processed, self.window_size, CONFIG['overlap_ratio'])
                if len(eeg_segments) == 0:
                    skipped_files += 1
                    continue
                
                # Process spectrogram if available
                spec_processed = None
                if CONFIG['use_spectrograms'] and spec_file is not None:
                    # Check for timeout
                    if time.time() - start_time > CONFIG['max_file_time'] * 0.7:  # 70% of max time
                        print(f"Not enough time for spectrogram, skipping for file {i}")
                    else:
                        try:
                            spec_data = load_spectrogram(spec_file)
                            if spec_data is not None:
                                spec_processed = preprocess_spectrogram(
                                    spec_data, 
                                    CONFIG['spec_height'], 
                                    CONFIG['spec_width']
                                )
                        except Exception as e:
                            print(f"Error with spectrogram {spec_file}: {e}")
                
                # If spectrogram processing failed, use empty spectrograms
                if spec_processed is None:
                    spec_processed = np.zeros((CONFIG['spec_height'], CONFIG['spec_width']))
                
                # Create segments
                for seg_idx, eeg_segment in enumerate(eeg_segments):
                    # Normalize each EEG segment
                    try:
                        norm_segment = (eeg_segment - np.mean(eeg_segment, axis=0)) / (np.std(eeg_segment, axis=0) + 1e-8)
                        self.eeg_segments.append(norm_segment)
                        self.spectrogram_segments.append(spec_processed)
                        
                        if not self.is_test:
                            self.segment_labels.append(labels[i])
                        self.segment_to_file_idx.append(i)
                    except Exception as e:
                        print(f"Error normalizing segment {seg_idx} from file {i}: {e}")
                        continue
                
                processed_files += 1
                
                # Clean up memory
                del eeg_df, eeg_processed, eeg_segments
                if spec_data is not None:
                    del spec_data
                
                # Perform garbage collection periodically
                if i % 10 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing file {i} ({eeg_file}): {e}")
                skipped_files += 1
                continue
                
            # Stop if we've spent too much time on this batch
            if time.time() - start_time > CONFIG['max_file_time'] * 2:
                print(f"File {i} took too long, forcing garbage collection")
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        print(f"Created dataset with {len(self.eeg_segments)} segments from {processed_files} files")
        print(f"Skipped {skipped_files} problematic files")
    
    def __len__(self):
        return len(self.eeg_segments)
    
    def __getitem__(self, idx):
        eeg_segment = self.eeg_segments[idx]
        spec_segment = self.spectrogram_segments[idx]
        
        # Convert to tensors
        eeg_tensor = torch.tensor(eeg_segment, dtype=torch.float32)
        spec_tensor = torch.tensor(spec_segment, dtype=torch.float32).unsqueeze(0)  # Add channel dim
        
        # Permute EEG for channel-first format
        eeg_tensor = eeg_tensor.permute(1, 0)  # [channels, time]
        
        if not self.is_test:
            label = self.segment_labels[idx]
            return (eeg_tensor, spec_tensor), label, self.segment_to_file_idx[idx]
        else:
            return (eeg_tensor, spec_tensor), self.segment_to_file_idx[idx]



# Cell 6: Multimodal Model Architecture
class MultimodalEEGNet(nn.Module):
    def __init__(self, 
                 num_eeg_channels=len(CONFIG['channels']), 
                 num_classes=CONFIG['n_classes'], 
                 dropout_rate=CONFIG['dropout_rate']):
        super(MultimodalEEGNet, self).__init__()
        
        # EEG branch - Same as previous EEGNet
        self.eeg_conv1 = nn.Conv1d(
            in_channels=num_eeg_channels, 
            out_channels=32,
            kernel_size=64, 
            stride=2, 
            padding=32, 
            bias=False
        )
        self.eeg_batchnorm1 = nn.BatchNorm1d(32)
        
        self.eeg_depthwise_conv = nn.Conv1d(
            in_channels=32, 
            out_channels=64,
            kernel_size=16, 
            groups=32,
            stride=2, 
            padding=8, 
            bias=False
        )
        self.eeg_batchnorm2 = nn.BatchNorm1d(64)
        self.eeg_activation = nn.ELU()
        self.eeg_avgpool1 = nn.AvgPool1d(kernel_size=4, stride=4)
        self.eeg_dropout1 = nn.Dropout(dropout_rate)
        
        self.eeg_seperable_conv = nn.Conv1d(
            in_channels=64, 
            out_channels=64, 
            kernel_size=16, 
            padding=8, 
            bias=False
        )
        self.eeg_batchnorm3 = nn.BatchNorm1d(64)
        self.eeg_avgpool2 = nn.AvgPool1d(kernel_size=8, stride=8)
        self.eeg_dropout2 = nn.Dropout(dropout_rate)
        
        # Spectrogram branch - CNN for spectrogram processing
        self.spec_conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.spec_batchnorm1 = nn.BatchNorm2d(16)
        self.spec_maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.spec_conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.spec_batchnorm2 = nn.BatchNorm2d(32)
        self.spec_maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.spec_conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.spec_batchnorm3 = nn.BatchNorm2d(64)
        self.spec_maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.spec_dropout = nn.Dropout(dropout_rate)
        
        # Calculate feature dimensions after pooling operations
        # EEG branch: 64 channels * 7 time points = 448
        # Spectrogram branch: 64 * (height/8) * (width/8) = 64 * 16 * 32 = 32,768
        eeg_features = 64 * 7
        spec_features = 64 * (CONFIG['spec_height'] // 8) * (CONFIG['spec_width'] // 8)
        
        # Fusion and classification layers
        self.eeg_fc = nn.Linear(eeg_features, 128)
        self.spec_fc = nn.Linear(spec_features, 128)
        
        self.fusion_dropout = nn.Dropout(dropout_rate)
        self.fusion_fc = nn.Linear(256, 128)  # Combined features
        self.classifier = nn.Linear(128, num_classes)
        
        self.activation = nn.ELU()
    
    def forward(self, x):
        try:
            # Split inputs
            eeg_input, spec_input = x
            
            # Process EEG branch
            eeg = self.eeg_conv1(eeg_input)
            eeg = self.eeg_batchnorm1(eeg)
            
            eeg = self.eeg_depthwise_conv(eeg)
            eeg = self.eeg_batchnorm2(eeg)
            eeg = self.eeg_activation(eeg)
            eeg = self.eeg_avgpool1(eeg)
            eeg = self.eeg_dropout1(eeg)
            
            eeg = self.eeg_seperable_conv(eeg)
            eeg = self.eeg_batchnorm3(eeg)
            eeg = self.eeg_activation(eeg)
            eeg = self.eeg_avgpool2(eeg)
            eeg = self.eeg_dropout2(eeg)
            
            eeg = eeg.view(eeg.size(0), -1)  # Flatten
            eeg = self.eeg_fc(eeg)
            eeg = self.activation(eeg)
            
            # Process Spectrogram branch
            spec = self.spec_conv1(spec_input)
            spec = self.spec_batchnorm1(spec)
            spec = self.activation(spec)
            spec = self.spec_maxpool1(spec)
            
            spec = self.spec_conv2(spec)
            spec = self.spec_batchnorm2(spec)
            spec = self.activation(spec)
            spec = self.spec_maxpool2(spec)
            
            spec = self.spec_conv3(spec)
            spec = self.spec_batchnorm3(spec)
            spec = self.activation(spec)
            spec = self.spec_maxpool3(spec)
            spec = self.spec_dropout(spec)
            
            spec = spec.view(spec.size(0), -1)  # Flatten
            spec = self.spec_fc(spec)
            spec = self.activation(spec)
            
            # Combine features
            combined = torch.cat((eeg, spec), dim=1)
            combined = self.fusion_dropout(combined)
            combined = self.fusion_fc(combined)
            combined = self.activation(combined)
            
            # Classification
            output = self.classifier(combined)
            
            return output
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            # Return zeros as a fallback
            return torch.zeros(eeg_input.size(0), self.classifier.out_features, device=eeg_input.device)


# Cell 7: Training and Evaluation Functions
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, fold):
    best_val_loss = float('inf')
    patience_counter = 0
    
    # For storing metrics
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    # Wrap model with DataParallel if using multiple GPUs
    if CONFIG['use_multi_gpu'] and torch.cuda.device_count() > 1:
        model = DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        batch_count = 0
        
        for batch_idx, ((eeg_data, spec_data), labels, _) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training")):
            try:
                eeg_data, spec_data, labels = eeg_data.to(device), spec_data.to(device), labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                if CONFIG['mixed_precision']:
                    # Mixed precision training
                    with autocast():
                        outputs = model((eeg_data, spec_data))
                        loss = criterion(outputs, labels)
                    
                    # Scale the loss and backpropagate
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Regular training
                    outputs = model((eeg_data, spec_data))
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                
                # Print statistics
                if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                    print(f"Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
                
                # Clear memory
                del eeg_data, spec_data, outputs, loss
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error in training batch {batch_idx}: {e}")
                continue
        
        if batch_count > 0:
            train_loss /= batch_count
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []
        val_batch_count = 0
        
        with torch.no_grad():
            for (eeg_data, spec_data), labels, _ in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation"):
                try:
                    eeg_data, spec_data, labels = eeg_data.to(device), spec_data.to(device), labels.to(device)
                    
                    if CONFIG['mixed_precision']:
                        with autocast():
                            outputs = model((eeg_data, spec_data))
                            loss = criterion(outputs, labels)
                    else:
                        outputs = model((eeg_data, spec_data))
                        loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    val_batch_count += 1
                    
                    _, predicted = torch.max(outputs, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())
                    
                    # Clear memory
                    del eeg_data, spec_data, outputs, loss
                    
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        if val_batch_count > 0:
            val_loss /= val_batch_count
        val_losses.append(val_loss)
        
        # Calculate validation accuracy
        if len(val_preds) > 0 and len(val_true) > 0:
            val_accuracy = accuracy_score(val_true, val_preds)
            val_accuracies.append(val_accuracy)
        else:
            val_accuracy = 0.0
            val_accuracies.append(0.0)
            print("Warning: No valid predictions for validation accuracy calculation")
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
        print(f"Epoch time: {epoch_time:.2f} seconds")
        
        # Check if we should save the model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Save the model
            try:
                if isinstance(model, DataParallel):
                    torch.save(model.module.state_dict(), f"{OUTPUT_DIR}/model_fold{fold}_epoch{epoch+1}.pt")
                else:
                    torch.save(model.state_dict(), f"{OUTPUT_DIR}/model_fold{fold}_epoch{epoch+1}.pt")
                print(f"Model saved at epoch {epoch+1}")
            except Exception as e:
                print(f"Error saving model: {e}")
            
            patience_counter = 0
        else:
            patience_counter += 1
            
        # Early stopping
        if patience_counter >= CONFIG['patience']:
            print(f"Early stopping triggered after epoch {epoch+1}")
            break
        
        # Update learning rate
        scheduler.step()
        
        # Cleanup to prevent memory leaks
        torch.cuda.empty_cache()
        gc.collect()
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title(f'Fold {fold} - Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title(f'Fold {fold} - Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/training_curves_fold{fold}.png")
    
    return model

def predict(model, test_loader):
    model.eval()
    all_predictions = []
    file_indices = []
    
    with torch.no_grad():
        for batch_idx, ((eeg_data, spec_data), file_idx) in enumerate(tqdm(test_loader, desc="Generating predictions")):
            try:
                eeg_data, spec_data = eeg_data.to(device), spec_data.to(device)
                
                if CONFIG['mixed_precision']:
                    with autocast():
                        outputs = model((eeg_data, spec_data))
                else:
                    outputs = model((eeg_data, spec_data))
                    
                probabilities = F.softmax(outputs, dim=1)
                all_predictions.append(probabilities.cpu().numpy())
                file_indices.extend(file_idx.cpu().numpy())
                
                # Clean up memory
                del eeg_data, spec_data, outputs, probabilities
                if batch_idx % 10 == 0:
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error during prediction batch {batch_idx}: {e}")
                # Continue with next batch if there's an error
                continue
    
    # Concatenate all predictions
    if len(all_predictions) > 0:
        all_predictions = np.vstack(all_predictions)
    else:
        print("Warning: No predictions were generated")
        all_predictions = np.array([])
    
    return all_predictions, np.array(file_indices)

def aggregate_predictions(predictions, file_indices, num_files, num_classes):
    """Aggregate predictions from multiple segments to file-level predictions."""
    # Initialize file-level predictions
    file_preds = np.zeros((num_files, num_classes))
    counts = np.zeros(num_files)
    
    # Sum up predictions for each file
    if len(predictions) > 0:
        for pred, file_idx in zip(predictions, file_indices):
            file_preds[file_idx] += pred
            counts[file_idx] += 1
    
    # Average predictions
    for i in range(num_files):
        if counts[i] > 0:
            file_preds[i] /= counts[i]
        else:
            # If no predictions for this file, use a uniform distribution
            file_preds[i] = np.ones(num_classes) / num_classes
            print(f"Warning: No predictions for file index {i}")
    
    return file_preds


def main_test():
    """Run a test with a small dataset to verify the pipeline."""
    print("Running test with a smaller dataset...")
    
    # Monitor memory usage
    if torch.cuda.is_available():
        print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Use only a small subset for testing
    subset_size = 100  # Just use 100 files for testing
    
    # Prepare file paths - use only a subset
    train_eeg_files = [os.path.join(TRAIN_DIR, f"{file_id}.parquet") for file_id in train_metadata['eeg_id'][:subset_size]]
    test_eeg_files = [os.path.join(TEST_DIR, f"{file_id}.parquet") for file_id in test_metadata['eeg_id'][:10]]  # Just a few test files
    
    # Prepare spectrogram file paths
    train_spec_files = [None] * len(train_eeg_files)  # Start with EEG only for testing
    test_spec_files = [None] * len(test_eeg_files)
    
    # Temporarily disable spectrogram usage for faster testing
    CONFIG['use_spectrograms'] = False
    
    # Convert labels to numerical values for the subset
    train_labels = np.array([labels_map[label] for label in train_metadata[label_column][:subset_size]])
    
    # Just use 2 folds for testing
    CONFIG['fold_count'] = 2
    
    # Define cross-validation strategy
    kfold = StratifiedKFold(n_splits=CONFIG['fold_count'], shuffle=True, random_state=42)
    
    # Loop through folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_eeg_files, train_labels)):
        print(f"\n{'='*20} Test Fold {fold+1}/{CONFIG['fold_count']} {'='*20}\n")
        
        # Split data for this fold - use fewer files for testing
        fold_train_eeg_files = [train_eeg_files[i] for i in train_idx][:100]  # Just use 100 files
        fold_val_eeg_files = [train_eeg_files[i] for i in val_idx][:20]      # Just use 20 files
        fold_train_spec_files = [train_spec_files[i] for i in train_idx][:100]
        fold_val_spec_files = [train_spec_files[i] for i in val_idx][:20]
        fold_train_labels = train_labels[train_idx][:100]
        fold_val_labels = train_labels[val_idx][:20]
        
        # Create datasets with verbose error reporting
        try:
            print(f"Creating training dataset for test fold {fold+1}...")
            train_dataset = MultimodalDataset(fold_train_eeg_files, fold_train_spec_files, fold_train_labels)
            
            print(f"Successfully created training dataset with {len(train_dataset)} segments")
            print(f"Creating validation dataset for test fold {fold+1}...")
            val_dataset = MultimodalDataset(fold_val_eeg_files, fold_val_spec_files, fold_val_labels)
            print(f"Successfully created validation dataset with {len(val_dataset)} segments")
            
            print("Test successful! The pipeline is working correctly.")
            return
            
        except Exception as e:
            print(f"Error in test fold {fold+1}: {e}")
            import traceback
            traceback.print_exc()
            return


# Cell 8: Main Competition Function - Batched Processing
def main_competition_batched():
    # Use 30% of the total training data
    total_samples = len(train_metadata)
    subset_size = int(total_samples * 0.3)  # ~32,000 samples
    
    print(f"Running competition solution with {subset_size} samples (30% of total data) using batched processing...")
    
    # Monitor memory usage
    if torch.cuda.is_available():
        print(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Prepare file paths
    train_eeg_files = [os.path.join(TRAIN_DIR, f"{file_id}.parquet") for file_id in train_metadata['eeg_id'][:subset_size]]
    test_eeg_files = [os.path.join(TEST_DIR, f"{file_id}.parquet") for file_id in test_metadata['eeg_id']]
    
    # Determine if spectrograms can be used
    has_spectrograms = 'spectrogram_id' in train_metadata.columns
    if has_spectrograms:
        print("Found spectrogram data, using multimodal approach")
        CONFIG['use_spectrograms'] = True
        train_spec_files = []
        for i, spec_id in enumerate(train_metadata['spectrogram_id'][:subset_size]):
            spec_path = os.path.join(TRAIN_SPECTROGRAMS_DIR, f"{spec_id}.parquet")
            if not os.path.exists(spec_path):
                # Try other possible extensions
                for ext in ['.png', '.jpg', '.npy']:
                    alt_path = os.path.join(TRAIN_SPECTROGRAMS_DIR, f"{spec_id}{ext}")
                    if os.path.exists(alt_path):
                        spec_path = alt_path
                        break
            train_spec_files.append(spec_path)
            
        test_spec_files = []
        for i, spec_id in enumerate(test_metadata['spectrogram_id']):
            spec_path = os.path.join(TEST_SPECTROGRAMS_DIR, f"{spec_id}.parquet")
            if not os.path.exists(spec_path):
                # Try other possible extensions
                for ext in ['.png', '.jpg', '.npy']:
                    alt_path = os.path.join(TEST_SPECTROGRAMS_DIR, f"{spec_id}{ext}")
                    if os.path.exists(alt_path):
                        spec_path = alt_path
                        break
            test_spec_files.append(spec_path)
    else:
        print("No spectrogram data found, using EEG only")
        CONFIG['use_spectrograms'] = False
        train_spec_files = [None] * len(train_eeg_files)
        test_spec_files = [None] * len(test_eeg_files)
    
    # Convert labels to numerical values
    train_labels = np.array([labels_map[label] for label in train_metadata[label_column][:subset_size]])
    
    # Run a more practical number of folds
    CONFIG['fold_count'] = 3  # Use fewer folds to save time
    
    # Define cross-validation strategy
    kfold = StratifiedKFold(n_splits=CONFIG['fold_count'], shuffle=True, random_state=42)
    
    # Store fold predictions for ensemble
    fold_test_preds = []
    
    # Loop through folds
    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_eeg_files, train_labels)):
        print(f"\n{'='*20} Fold {fold+1}/{CONFIG['fold_count']} {'='*20}\n")
        
        # Split data for this fold
        fold_train_eeg_files = [train_eeg_files[i] for i in train_idx]
        fold_val_eeg_files = [train_eeg_files[i] for i in val_idx]
        fold_train_spec_files = [train_spec_files[i] for i in train_idx]
        fold_val_spec_files = [train_spec_files[i] for i in val_idx]
        fold_train_labels = train_labels[train_idx]
        fold_val_labels = train_labels[val_idx]
        
        # Process in batches
        batch_size = CONFIG['batch_processing_size']
        
        # Create training datasets in batches
        train_datasets = []
        for batch_start in range(0, len(fold_train_eeg_files), batch_size):
            batch_end = min(batch_start + batch_size, len(fold_train_eeg_files))
            print(f"Processing training batch {batch_start//batch_size + 1}/{(len(fold_train_eeg_files) + batch_size - 1)//batch_size}...")
            
            batch_train_eeg = fold_train_eeg_files[batch_start:batch_end]
            batch_train_spec = fold_train_spec_files[batch_start:batch_end]
            batch_train_labels = fold_train_labels[batch_start:batch_end]
            
            try:
                batch_dataset = BatchMultimodalDataset(batch_train_eeg, batch_train_spec, batch_train_labels)
                if len(batch_dataset) > 0:
                    train_datasets.append(batch_dataset)
                    
                # Force garbage collection after each batch
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing training batch: {e}")
                import traceback
                traceback.print_exc()
        
        # Create validation datasets in batches
        val_datasets = []
        for batch_start in range(0, len(fold_val_eeg_files), batch_size):
            batch_end = min(batch_start + batch_size, len(fold_val_eeg_files))
            print(f"Processing validation batch {batch_start//batch_size + 1}/{(len(fold_val_eeg_files) + batch_size - 1)//batch_size}...")
            
            batch_val_eeg = fold_val_eeg_files[batch_start:batch_end]
            batch_val_spec = fold_val_spec_files[batch_start:batch_end]
            batch_val_labels = fold_val_labels[batch_start:batch_end]
            
            try:
                batch_dataset = BatchMultimodalDataset(batch_val_eeg, batch_val_spec, batch_val_labels)
                if len(batch_dataset) > 0:
                    val_datasets.append(batch_dataset)
                    
                # Force garbage collection after each batch
                gc.collect()
                torch.cuda.empty_cache()
                
            except Exception as e:
                print(f"Error processing validation batch: {e}")
                import traceback
                traceback.print_exc()
        
        # Combine datasets and create data loaders
        if len(train_datasets) > 0 and len(val_datasets) > 0:
            train_dataset = ConcatDataset(train_datasets)
            val_dataset = ConcatDataset(val_datasets)
            
            print(f"Combined training dataset has {len(train_dataset)} segments")
            print(f"Combined validation dataset has {len(val_dataset)} segments")
            
            # Create data loaders
            train_loader = DataLoader(
                train_dataset, 
                batch_size=CONFIG['batch_size'], 
                shuffle=True, 
                num_workers=CONFIG['num_workers'],
                pin_memory=CONFIG['pin_memory']
            )
            
            val_loader = DataLoader(
                val_dataset, 
                batch_size=CONFIG['batch_size'], 
                shuffle=False, 
                num_workers=CONFIG['num_workers'],
                pin_memory=CONFIG['pin_memory']
            )
            
            # Initialize model
            model = MultimodalEEGNet(
                num_eeg_channels=len(CONFIG['channels']),
                num_classes=CONFIG['n_classes'],
                dropout_rate=CONFIG['dropout_rate']
            ).to(device)
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = AdamW(
                model.parameters(), 
                lr=CONFIG['learning_rate'], 
                weight_decay=CONFIG['weight_decay']
            )
            
            # Define learning rate scheduler
            scheduler = CosineAnnealingLR(
                optimizer, 
                T_max=CONFIG['num_epochs']
            )
            
            # Train the model
            try:
                model = train_model(
                    model, 
                    train_loader, 
                    val_loader, 
                    criterion, 
                    optimizer, 
                    scheduler, 
                    CONFIG['num_epochs'],
                    fold
                )
                
                # Find the best model for this fold
                model_files = glob.glob(f"{OUTPUT_DIR}/model_fold{fold}_*.pt")
                if model_files:
                    best_model_path = sorted(model_files)[-1]
                    print(f"Using model: {best_model_path}")
                    
                    # Create a clean model instance for inference
                    inference_model = MultimodalEEGNet(
                        num_eeg_channels=len(CONFIG['channels']),
                        num_classes=CONFIG['n_classes'],
                        dropout_rate=CONFIG['dropout_rate']
                    ).to(device)
                    
                    inference_model.load_state_dict(torch.load(best_model_path))
                    
                    # If using multiple GPUs, wrap the inference model
                    if CONFIG['use_multi_gpu'] and torch.cuda.device_count() > 1:
                        inference_model = DataParallel(inference_model)
                    
                    # Process test data in batches
                    all_test_preds = []
                    
                    # Break test prediction into manageable batches
                    test_batch_size = min(CONFIG['batch_processing_size'], 100)  # Smaller batch for test files
                    
                    for test_batch_start in range(0, len(test_eeg_files), test_batch_size):
                        test_batch_end = min(test_batch_start + test_batch_size, len(test_eeg_files))
                        print(f"Processing test batch {test_batch_start//test_batch_size + 1}/{(len(test_eeg_files) + test_batch_size - 1)//test_batch_size}...")
                        
                        batch_test_eeg = test_eeg_files[test_batch_start:test_batch_end]
                        batch_test_spec = test_spec_files[test_batch_start:test_batch_end]
                        
                        try:
                            # Create test dataset for this batch
                            batch_test_dataset = BatchMultimodalDataset(batch_test_eeg, batch_test_spec, is_test=True)
                            
                            if len(batch_test_dataset) > 0:
                                batch_test_loader = DataLoader(
                                    batch_test_dataset, 
                                    batch_size=CONFIG['batch_size'], 
                                    shuffle=False, 
                                    num_workers=CONFIG['num_workers'],
                                    pin_memory=CONFIG['pin_memory']
                                )
                                
                                # Generate predictions
                                batch_preds, batch_indices = predict(inference_model, batch_test_loader)
                                
                                # Adjust indices to account for batch offset
                                adjusted_indices = [idx + test_batch_start for idx in batch_indices]
                                
                                # Store predictions
                                if len(batch_preds) > 0:
                                    for pred, file_idx in zip(batch_preds, adjusted_indices):
                                        all_test_preds.append((pred, file_idx))
                                
                                # Clean up
                                del batch_test_dataset, batch_test_loader, batch_preds, batch_indices
                                gc.collect()
                                torch.cuda.empty_cache()
                                
                        except Exception as e:
                            print(f"Error processing test batch: {e}")
                            import traceback
                            traceback.print_exc()
                    
                    # Create file-level predictions
                    if all_test_preds:
                        # Extract predictions and indices
                        test_preds = np.array([p[0] for p in all_test_preds])
                        test_indices = np.array([p[1] for p in all_test_preds])
                        
                        # Aggregate to file level
                        file_preds = aggregate_predictions(
                            test_preds, 
                            test_indices, 
                            len(test_eeg_files), 
                            CONFIG['n_classes']
                        )
                        
                        fold_test_preds.append(file_preds)
                    else:
                        print(f"No predictions generated for fold {fold}")
                else:
                    print(f"No model files found for fold {fold}")
            except Exception as e:
                print(f"Error in training/inference for fold {fold}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"Insufficient data for fold {fold}, skipping")
        
        # Free up memory
        if 'model' in locals():
            del model
        if 'inference_model' in locals():
            del inference_model
        if 'train_dataset' in locals():
            del train_dataset
        if 'val_dataset' in locals():
            del val_dataset
        if 'train_loader' in locals():
            del train_loader
        if 'val_loader' in locals():
            del val_loader
        
        # Free up batch datasets
        del train_datasets, val_datasets
        
        # Clean up all other memory
        gc.collect()
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            print(f"GPU memory after fold {fold+1}: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # Ensemble predictions from all folds
    if fold_test_preds:
        ensemble_preds = np.mean(fold_test_preds, axis=0)
        
        # Create submission file
        submission = pd.DataFrame()
        submission['eeg_id'] = test_metadata['eeg_id']
        
        for label, idx in labels_map.items():
            submission[label] = ensemble_preds[:, idx]
        
        # Save submission file
        submission.to_csv(f"{OUTPUT_DIR}/submission.csv", index=False)
        print(f"Submission file saved to {OUTPUT_DIR}/submission.csv")
    else:
        print("No predictions were generated from any fold")




# Cell 10: Run the Competition Pipeline
# First, run the test to make sure everything works
print("Running quick test to verify pipeline...")
main_test()

# If the test is successful, run the full competition solution with batched processing
print("\nRunning full competition solution with 30% of the data using batched processing...")
main_competition_batched()




