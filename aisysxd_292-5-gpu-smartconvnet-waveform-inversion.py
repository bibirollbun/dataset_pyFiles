import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import csv
import os
import logging
from typing import List, Tuple, Dict, Optional, Union
import gc
# Импорты для оптимизации GPU
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import time


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

CONFIG = {
    # Parameters for model
    'input_channels': 5,
    'output_size': 70 * 70,
    'feature_channels': [32, 64, 128, 256, 512],  # Added one more level for deeper network
    'dropout_rate': 0.3,
    
    # Parameters for training
    'batch_size': 64,
    'num_workers': 4,
    'pin_memory': True,
    'learning_rate': 1e-5,  # Further reduced learning rate for better convergence
    'weight_decay': 5e-5,   # Adjusted weight decay for better regularization
    'n_epochs': 100,        # Increased number of epochs
    'early_stopping_patience': 15,  # Increased patience for early stopping
    
    # Paths to data
    'train_data_path': '/kaggle/input/waveform-inversion/train_samples',
    'test_data_path': '/kaggle/input/waveform-inversion/test',
    'output_file': 'submission.csv',
    'log_file': 'training_log.csv',  # File for saving training logs
    
    
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    'save_model_path': 'smartconvnet_model.pth',
    'plot_every_n_epochs': 4,
    
    # Parameters for GPU optimization
    'use_mixed_precision': True,  # Use mixed precision (float16/float32)
    'use_cudnn_benchmark': True,  # CUDNN optimization for repetitive operations
    'prefetch_factor': 2,         # Data prefetch factor
    'use_distributed': False,     # Use distributed training (for multiple GPUs)
    'gpu_memory_fraction': 0.95,  # GPU memory fraction to use (0.0-1.0)
    'gradient_accumulation_steps': 1,  # Number of steps for gradient accumulation
    
    # Parameters for logging
    'log_batch_interval': 10,     # Batch logging interval
    'detailed_logging': True,     # Detailed logging
    
    # Loss function parameters
    'edge_weight': 1.0,           # Weight for edge preservation in loss function
    'use_cyclic_lr': True         # Use cyclic learning rate scheduler
}

# GPU optimization
if torch.cuda.is_available():
    # Enable cuDNN benchmark for repetitive operations
    if CONFIG['use_cudnn_benchmark']:
        torch.backends.cudnn.benchmark = True
        logger.info("cuDNN benchmark enabled")
    
    # Set GPU memory fraction
    if CONFIG['gpu_memory_fraction'] < 1.0:
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory
            allocated_memory = int(total_memory * CONFIG['gpu_memory_fraction'])
            torch.cuda.set_per_process_memory_fraction(CONFIG['gpu_memory_fraction'])
            logger.info(f"GPU memory fraction set to {CONFIG['gpu_memory_fraction']}")
        except Exception as e:
            logger.warning(f"Failed to set GPU memory fraction: {e}")
    
    # Output information about available GPUs
    gpu_count = torch.cuda.device_count()
    logger.info(f"Available GPUs: {gpu_count}")
    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        logger.info(f"GPU {i}: {gpu_name}, Memory: {gpu_memory:.2f} GB")

# Setup paths to data files
def setup_paths() -> Tuple[List[Path], List[Path], List[Path]]:
    """Setup paths to data files"""
    logger.info("Setup paths to data files")
    
    # Check for Kaggle data
    if not os.path.exists(CONFIG['train_data_path']):
        try:
            import kagglehub
            kagglehub.login()
            kagglehub.competition_download('waveform-inversion')
            logger.info('Kaggle data downloaded')
        except Exception as e:
            logger.error(f"Failed to download Kaggle data: {e}")
            raise
    
    # Find input files
    all_inputs = [
        f for f in Path(CONFIG['train_data_path']).rglob('*.npy')
        if ('seis' in f.stem) or ('data' in f.stem)
    ]
    
    # Get corresponding output files
    all_outputs = [
        Path(str(f).replace('seis', 'vel').replace('data', 'model'))
        for f in all_inputs
    ]
    
    # Check for missing output files
    if not all(f.exists() for f in all_outputs):
        missing = [str(f) for f in all_outputs if not f.exists()]
        logger.error(f"Missing output files: {missing[:5]}...")
        raise FileNotFoundError(f"Missing output files: {len(missing)}")
    
    # Split into training and validation sets
    train_inputs = [all_inputs[i] for i in range(0, len(all_inputs), 2)]  # Every second file
    valid_inputs = [f for f in all_inputs if f not in train_inputs]
    
    logger.info(f"Found {len(all_inputs)} input files")
    logger.info(f"Training set: {len(train_inputs)} files")
    logger.info(f"Validation set: {len(valid_inputs)} files")
    
    return all_inputs, train_inputs, valid_inputs


def get_output_files(input_files: List[Path]) -> List[Path]:
    """Get corresponding output files"""
    return [
        Path(str(f).replace('seis', 'vel').replace('data', 'model'))
        for f in input_files
    ]


class SeismicDataset(Dataset):
    """Seismic dataset with GPU optimization"""
    
    def __init__(self, inputs_files: List[Path], output_files: List[Path], n_examples_per_file: int = 500,
                 cache_size: int = 100, preload: bool = False):
        """
        Initialize dataset
        
        Args:
            inputs_files: List of paths to input files
            output_files: List of paths to output files
            n_examples_per_file: Number of examples per file
            cache_size: Size of cache for storing preloaded data
            preload: Preload data into memory
        """
        assert len(inputs_files) == len(output_files), "Number of input and output files must match"
        self.inputs_files = inputs_files
        self.output_files = output_files
        self.n_examples_per_file = n_examples_per_file
        self.cache_size = cache_size
        self.preload = preload
        
        # Cache for storing loaded data
        self.cache = {}
        
        # Preload data
        if self.preload:
            logger.info(f"Preloading data into memory ({len(inputs_files)} files)...")
            for i, (input_file, output_file) in enumerate(zip(inputs_files, output_files)):
                if i >= self.cache_size:
                    break
                    
                try:
                    X = np.load(input_file)
                    y = np.load(output_file)
                    
                    # Convert to tensors and move to CPU memory
                    X_tensor = torch.from_numpy(X).float().cpu()
                    y_tensor = torch.from_numpy(y).float().cpu()
                    
                    self.cache[i] = (X_tensor, y_tensor)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"Preloaded {i + 1}/{min(len(inputs_files), self.cache_size)} files")
                except Exception as e:
                    logger.error(f"Failed to preload file {input_file}: {e}")
            
            logger.info(f"Preloading completed. {len(self.cache)} files preloaded")

    def __len__(self) -> int:
        return len(self.inputs_files) * self.n_examples_per_file

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Calculate file index and sample index within file
        file_idx = idx // self.n_examples_per_file
        sample_idx = idx % self.n_examples_per_file
        
        # Check for data in cache
        if file_idx in self.cache:
            X_tensor, y_tensor = self.cache[file_idx]
            return X_tensor[sample_idx].clone(), y_tensor[sample_idx].clone()
        
        # If data is not in cache, load it
        try:
            # Load data using memory mapping for memory economy
            X = np.load(self.inputs_files[file_idx], mmap_mode='r')
            y = np.load(self.output_files[file_idx], mmap_mode='r')
            
            # Convert to PyTorch tensors
            X_sample = torch.from_numpy(X[sample_idx].copy()).float()
            y_sample = torch.from_numpy(y[sample_idx].copy()).float()
            
            # Add to cache if cache size is not exceeded
            if len(self.cache) < self.cache_size and file_idx not in self.cache:
                # Load entire file into cache
                X_tensor = torch.from_numpy(X.copy()).float().cpu()
                y_tensor = torch.from_numpy(y.copy()).float().cpu()
                self.cache[file_idx] = (X_tensor, y_tensor)
                
                # Release memory if cache is too large
                if len(self.cache) > self.cache_size:
                    # Remove the oldest element
                    oldest_key = next(iter(self.cache))
                    del self.cache[oldest_key]
                    gc.collect()
            
            return X_sample, y_sample
        except Exception as e:
            logger.error(f"Failed to load example {idx} (file {file_idx}, example {sample_idx}): {e}")
            raise
        finally:
            # Explicit memory release
            if 'X' in locals():
                del X
            if 'y' in locals():
                del y
            gc.collect()


class TestDataset(Dataset):
    """Test dataset with GPU optimization"""
    
    def __init__(self, test_files: List[Path], preload: bool = False):
        """
        Initialize test dataset
        
        Args:
            test_files: List of paths to test files
            preload: Preload data into memory
        """
        self.test_files = test_files
        self.preload = preload
        self.cache = {}
        
        logger.info(f"Test dataset: {len(test_files)} files")
        
        # Preload data
        if self.preload and len(test_files) <= 1000:  # Limit on number of files for preloading
            logger.info(f"Preloading test data into memory...")
            for i, test_file in enumerate(test_files):
                try:
                    data = np.load(test_file)
                    tensor_data = torch.from_numpy(data).float().cpu()
                    self.cache[i] = (tensor_data, test_file.stem)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"Preloaded {i + 1}/{len(test_files)} test files")
                except Exception as e:
                    logger.error(f"Failed to preload test file {test_file}: {e}")
            
            logger.info(f"Preloading test data completed")

    def __len__(self) -> int:
        return len(self.test_files)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, str]:
        # Проверка наличия данных в кэше
        if i in self.cache:
            return self.cache[i][0].clone(), self.cache[i][1]
        
        test_file = self.test_files[i]
        try:
            # Загрузка данных и преобразование в тензор PyTorch
            data = np.load(test_file)
            tensor_data = torch.from_numpy(data).float()
            return tensor_data, test_file.stem
        except Exception as e:
            logger.error(f"Ошибка при загрузке тестового файла {test_file}: {e}")
            raise


class ConvBlock(nn.Module):
    """Блок свертки с нормализацией и активацией"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, padding: int = 1, use_batchnorm: bool = True):
        """
        Инициализация блока свертки
        
        Args:
            in_channels: Количество входных каналов
            out_channels: Количество выходных каналов
            kernel_size: Размер ядра свертки
            stride: Шаг свертки
            padding: Отступ для сохранения размерности
            use_batchnorm: Использовать ли пакетную нормализацию
        """
        super().__init__()
        
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, 
                     stride=stride, padding=padding, bias=not use_batchnorm)
        ]
        
        if use_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
            
        layers.append(nn.ReLU(inplace=True))
        
        self.block = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через блок"""
        return self.block(x)


class ResidualBlock(nn.Module):
    """Блок с остаточными связями"""
    
    def __init__(self, channels: int):
        """
        Инициализация блока с остаточными связями
        
        Args:
            channels: Количество каналов
        """
        super().__init__()
        
        self.block = nn.Sequential(
            ConvBlock(channels, channels),
            ConvBlock(channels, channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через блок с остаточной связью"""
        return x + self.block(x)


class DownsampleBlock(nn.Module):
    """Блок понижения размерности"""
    
    def __init__(self, in_channels: int, out_channels: int):
        """
        Инициализация блока понижения размерности
        
        Args:
            in_channels: Количество входных каналов
            out_channels: Количество выходных каналов
        """
        super().__init__()
        
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, stride=2),  # Уменьшаем размер в 2 раза
            ResidualBlock(out_channels)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Прямой проход через блок понижения размерности"""
        return self.block(x)


class EnhancedResidualBlock(nn.Module):
    """Enhanced residual block with more layers and better gradient flow"""
    
    def __init__(self, channels: int, expansion: int = 4):
        """
        Initialize enhanced residual block
        
        Args:
            channels: Number of input/output channels
            expansion: Channel expansion factor for inner layers
        """
        super().__init__()
        
        inner_channels = channels // expansion
        
        self.bottleneck = nn.Sequential(
            # Reduce channels
            nn.Conv2d(channels, inner_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            
            # 3x3 convolution
            nn.Conv2d(inner_channels, inner_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
            
            # Expand channels back
            nn.Conv2d(inner_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels)
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection"""
        identity = x
        out = self.bottleneck(x)
        out += identity  # Skip connection
        return self.relu(out)


class EnhancedSmartConvNet(nn.Module):
    """Enhanced CNN with residual connections and U-Net like architecture"""
    
    def __init__(self, input_channels: int = CONFIG['input_channels'], 
                 output_size: int = CONFIG['output_size'],
                 feature_channels: List[int] = CONFIG['feature_channels'],
                 dropout_rate: float = CONFIG['dropout_rate']):
        """
        Initialize enhanced model
        
        Args:
            input_channels: Number of input channels
            output_size: Output tensor size
            feature_channels: List of channel counts for each level
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Initial block
        self.input_block = nn.Sequential(
            nn.Conv2d(input_channels, feature_channels[0], kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(feature_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Encoder blocks with residual connections
        self.encoder_blocks = nn.ModuleList()
        self.downsample_layers = nn.ModuleList()
        
        for i in range(len(feature_channels)-1):
            # Residual block at current resolution
            self.encoder_blocks.append(
                nn.Sequential(
                    EnhancedResidualBlock(feature_channels[i]),
                    EnhancedResidualBlock(feature_channels[i])
                )
            )
            
            # Downsampling to next resolution
            self.downsample_layers.append(
                nn.Sequential(
                    nn.Conv2d(feature_channels[i], feature_channels[i+1], 
                             kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(feature_channels[i+1]),
                    nn.ReLU(inplace=True)
                )
            )
        
        # Bottom level processing
        self.bottom = nn.Sequential(
            EnhancedResidualBlock(feature_channels[-1]),
            EnhancedResidualBlock(feature_channels[-1])
        )
        
        # Global pooling and fully connected layers
        self.global_pool = nn.AdaptiveAvgPool2d((8, 8))
        
        # Fully connected layers with improved structure
        fc_input_size = feature_channels[-1] * 8 * 8
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(fc_input_size, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            
            nn.Linear(2048, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate/2),
            
            nn.Linear(1024, output_size)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            x: Input tensor [batch_size, channels, height, width]
            
        Returns:
            Output tensor [batch_size, 1, 70, 70]
        """
        batch_size = x.size(0)
        
        # Initial processing
        x = self.input_block(x)
        
        # Encoder path with skip connections
        features = [x]
        for i, (encoder, downsampler) in enumerate(zip(self.encoder_blocks, self.downsample_layers)):
            # Apply residual blocks
            x = encoder(x)
            # Store features for potential skip connections
            features.append(x)
            # Downsample
            x = downsampler(x)
        
        # Bottom processing
        x = self.bottom(x)
        
        # Global pooling
        x = self.global_pool(x)
        
        # Fully connected layers
        x = self.fc(x)
        
        # Reshape and scale to expected output range
        return x.view(batch_size, 1, 70, 70) * 1000 + 1500


def combined_loss(pred: torch.Tensor, target: torch.Tensor, edge_weight: float = CONFIG['edge_weight']) -> torch.Tensor:
    """
    Combined loss function with edge preservation
    
    Args:
        pred: Predicted values
        target: Target values
        edge_weight: Weight for edge loss component
        
    Returns:
        Loss value
    """
    # L1 loss (MAE)
    l1_loss = F.l1_loss(pred, target)
    
    # Gradient loss for preserving sharp edges
    # Create Sobel kernels for edge detection
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32).view(1, 1, 3, 3).to(pred.device)
    
    # Calculate gradients for predictions and targets
    pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
    pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
    target_grad_x = F.conv2d(target, sobel_x, padding=1)
    target_grad_y = F.conv2d(target, sobel_y, padding=1)
    
    # Calculate edge loss
    grad_loss = F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y)
    
    # Add structural similarity component for better boundary preservation
    # Normalize inputs for SSIM calculation
    pred_norm = (pred - pred.mean()) / (pred.std() + 1e-8)
    target_norm = (target - target.mean()) / (target.std() + 1e-8)
    
    # Simple structural similarity term
    ssim_loss = F.mse_loss(pred_norm, target_norm)
    
    # Combined loss with increased edge weight
    return l1_loss + edge_weight * grad_loss + 0.2 * ssim_loss


def create_data_loaders() -> Tuple[DataLoader, DataLoader]:
    """Create data loaders for training and validation with GPU optimization"""
    # Set up data file paths
    _, train_inputs, valid_inputs = setup_paths()
    
    # Get corresponding output files
    train_outputs = get_output_files(train_inputs)
    valid_outputs = get_output_files(valid_inputs)
    
    # Create datasets with preloading for acceleration
    dstrain = SeismicDataset(
        train_inputs, 
        train_outputs, 
        preload=True,
        cache_size=min(50, len(train_inputs))  # Cache up to 50 files
    )
    
    dsvalid = SeismicDataset(
        valid_inputs, 
        valid_outputs,
        preload=True,
        cache_size=min(20, len(valid_inputs))  # Cache up to 20 files
    )
    
    # Create data loaders with optimized parameters
    dltrain = DataLoader(
        dstrain, 
        batch_size=CONFIG['batch_size'], 
        shuffle=True, 
        pin_memory=CONFIG['pin_memory'], 
        drop_last=True, 
        num_workers=CONFIG['num_workers'], 
        persistent_workers=True,
        prefetch_factor=CONFIG['prefetch_factor']
    )
    
    dlvalid = DataLoader(
        dsvalid, 
        batch_size=CONFIG['batch_size'], 
        shuffle=False, 
        pin_memory=CONFIG['pin_memory'], 
        drop_last=False, 
        num_workers=CONFIG['num_workers'], 
        persistent_workers=True,
        prefetch_factor=CONFIG['prefetch_factor']
    )
    
    return dltrain, dlvalid


class EarlyStopping:
    """Early stopping mechanism to prevent overfitting"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'min'):
        """
        Initialize early stopping
        
        Args:
            patience: Number of epochs without improvement before stopping
            min_delta: Minimum change to be considered as improvement
            mode: 'min' for metrics to minimize, 'max' for metrics to maximize
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score: float) -> bool:
        """
        Check if training should be stopped
        
        Args:
            score: Current metric value
            
        Returns:
            True if training should stop, False otherwise
        """
        if self.best_score is None:
            self.best_score = score
            return False
            
        if self.mode == 'min':
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:  # mode == 'max'
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
                
        if self.counter >= self.patience:
            logger.info(f"Early stopping: {self.counter} epochs without improvement")
            return True
            
        return False


def train_model(model: nn.Module, train_loader: DataLoader, valid_loader: DataLoader) -> Dict:
    """
    Train model with GPU optimization
    
    Args:
        model: Model to train
        train_loader: Training data loader
        valid_loader: Validation data loader
        
    Returns:
        Training history
    """
    logger.info("Starting model training")
    
    # Set up optimizer and loss function
    criterion = combined_loss  # Use combined loss function
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=CONFIG['learning_rate'], 
        weight_decay=CONFIG['weight_decay']
    )
    
    # Set up learning rate scheduler
    if CONFIG['use_cyclic_lr']:
        # Cyclic learning rate scheduler for better convergence
        scheduler = torch.optim.lr_scheduler.CyclicLR(
            optimizer,
            base_lr=CONFIG['learning_rate'] / 10,  # Lower bound of learning rate
            max_lr=CONFIG['learning_rate'] * 5,    # Upper bound of learning rate
            step_size_up=len(train_loader) * 2,    # Steps per half cycle (2 epochs)
            mode='triangular2',                    # Triangular mode with amplitude reduction
            cycle_momentum=False                   # Don't cycle momentum
        )
        logger.info("Using Cyclic Learning Rate scheduler")
    else:
        # Traditional ReduceLROnPlateau scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5,     # Reduce learning rate by half
            patience=5,     # After 5 epochs without improvement
            threshold=0.01, # Threshold for measuring improvement
            verbose=True
        )
        logger.info("Using ReduceLROnPlateau scheduler")
    
    # Set up early stopping
    early_stopping = EarlyStopping(patience=CONFIG['early_stopping_patience'])
    
    # Set up mixed precision for faster training on GPU
    scaler = GradScaler(enabled=CONFIG['use_mixed_precision'] and torch.cuda.is_available())
    
    # Training history
    history = []
    best_valid_loss = float('inf')
    
    # Create log file
    if CONFIG['detailed_logging']:
        with open(CONFIG['log_file'], 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'phase', 'batch', 'loss', 'lr'])
    
    # Training loop
    for epoch in range(1, CONFIG['n_epochs'] + 1):
        epoch_start_time = time.time()
        logger.info(f'Epoch [{epoch:02d}/{CONFIG["n_epochs"]}] Starting training')
        
        # Training phase
        model.train()
        train_losses = []
        
        # Clear CUDA cache before training
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Progress bar for training
        pbar = tqdm(train_loader, desc=f'Training (epoch {epoch})', leave=False)
        
        # Gradient accumulation counter
        accumulation_steps = 0
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            # Move data to device
            inputs = inputs.to(CONFIG['device'], non_blocking=True)
            targets = targets.to(CONFIG['device'], non_blocking=True)
            
            # Use mixed precision for faster computation
            with autocast(enabled=CONFIG['use_mixed_precision'] and torch.cuda.is_available()):
                # Forward pass
                outputs = model(inputs)
                
                # Calculate loss
                loss = criterion(outputs, targets)
                
                # Scale loss for gradient accumulation
                loss = loss / CONFIG['gradient_accumulation_steps']
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            if CONFIG['gradient_accumulation_steps'] == 1:  # Only if not using accumulation
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Increment accumulation counter
            accumulation_steps += 1
            
            # Update weights after accumulating gradients
            if accumulation_steps >= CONFIG['gradient_accumulation_steps']:
                # Optimization with scaling
                scaler.step(optimizer)
                scaler.update()
                
                # Update learning rate if using cyclic scheduler
                if CONFIG['use_cyclic_lr']:
                    scheduler.step()
                    
                optimizer.zero_grad(set_to_none=True)  # More efficient gradient zeroing
                accumulation_steps = 0
            
            # Save losses
            batch_loss = loss.item() * CONFIG['gradient_accumulation_steps']
            train_losses.append(batch_loss)
            
            # Update progress bar
            pbar.set_postfix({'loss': np.mean(train_losses[-100:]), 'lr': optimizer.param_groups[0]['lr']})
            
            # Log for each batch
            if CONFIG['detailed_logging'] and (batch_idx % CONFIG['log_batch_interval'] == 0):
                with open(CONFIG['log_file'], 'a', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch, 'train', batch_idx, batch_loss, optimizer.param_groups[0]['lr']])
        
        # If there are remaining accumulated gradients
        if accumulation_steps > 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Calculate average training loss
        avg_train_loss = np.mean(train_losses)
        logger.info(f'Training loss: {avg_train_loss:.5f}')
        
        # Validation phase
        model.eval()
        valid_losses = []
        
        # Clear CUDA cache before validation
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(tqdm(valid_loader, desc=f'Validation (epoch {epoch})', leave=False)):
                # Move data to device
                inputs = inputs.to(CONFIG['device'], non_blocking=True)
                targets = targets.to(CONFIG['device'], non_blocking=True)
                
                # Use mixed precision for faster computation
                with autocast(enabled=CONFIG['use_mixed_precision'] and torch.cuda.is_available()):
                    # Forward pass
                    outputs = model(inputs)
                    
                    # Calculate loss
                    loss = criterion(outputs, targets)
                
                batch_loss = loss.item()
                valid_losses.append(batch_loss)
                
                # Log for each batch
                if CONFIG['detailed_logging'] and (batch_idx % CONFIG['log_batch_interval'] == 0):
                    with open(CONFIG['log_file'], 'a', newline='') as f:
                        writer = csv.writer(f)
                        writer.writerow([epoch, 'valid', batch_idx, batch_loss, optimizer.param_groups[0]['lr']])
        
        # Calculate average validation loss
        avg_valid_loss = np.mean(valid_losses)
        epoch_time = time.time() - epoch_start_time
        
        # Detailed log at end of epoch
        logger.info(f'Epoch {epoch} completed in {epoch_time:.2f} sec.')
        logger.info(f'Training loss: {avg_train_loss:.5f} | Validation loss: {avg_valid_loss:.5f} | LR: {optimizer.param_groups[0]["lr"]:.8f}')
        
        # Update learning rate scheduler if not using cyclic LR
        if not CONFIG['use_cyclic_lr']:
            scheduler.step(avg_valid_loss)
        
        # Save history
        history.append({
            'epoch': epoch,
            'train': avg_train_loss,
            'valid': avg_valid_loss,
            'lr': optimizer.param_groups[0]['lr'],
            'time': epoch_time
        })
        
        # Write epoch summary to log file
        if CONFIG['detailed_logging']:
            with open(CONFIG['log_file'], 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([epoch, 'summary', '', f'{avg_train_loss:.5f}/{avg_valid_loss:.5f}', optimizer.param_groups[0]['lr']])
        
        # Save best model
        if avg_valid_loss < best_valid_loss:
            best_valid_loss = avg_valid_loss
            torch.save(model.state_dict(), CONFIG['save_model_path'])
            logger.info(f'Saved best model with validation loss: {best_valid_loss:.5f}')
        
        # Visualize results
        if epoch % CONFIG['plot_every_n_epochs'] == 0:
            y = targets[0, 0].detach().cpu()
            y_pred = outputs[0, 0].detach().cpu()
            
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            fig.suptitle(f'Epoch {epoch} | Validation: {avg_valid_loss:.5f}')
            ax[0].imshow(y)
            ax[0].set_title('Ground Truth')
            ax[1].imshow(y_pred)
            ax[1].set_title('Prediction')
            plt.show()
            
            # Save image
            plt.savefig(f'epoch_{epoch}_comparison.png')
            plt.close()
        
        # Check for early stopping
        if early_stopping(avg_valid_loss):
            logger.info(f'Early stopping at epoch {epoch}')
            break
        
        # Clear memory at end of epoch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Load best model
    model.load_state_dict(torch.load(CONFIG['save_model_path']))
    logger.info("Training completed")
    
    # Save full training history to CSV
    history_df = pd.DataFrame(history)
    history_df.to_csv('training_history.csv', index=False)
    logger.info("Training history saved to training_history.csv")
    
    # Visualize training history
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Losses
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.plot(history_df['epoch'], history_df['train'], 'b-', label='Training')
    ax1.plot(history_df['epoch'], history_df['valid'], 'r-', label='Validation')
    ax1.tick_params(axis='y')
    ax1.grid(True)
    
    # Learning rate
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate')
    ax2.plot(history_df['epoch'], history_df['lr'], 'g--', label='LR')
    ax2.tick_params(axis='y')
    
    # Legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.title('Training History')
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()
    
    return history


def predict_and_save(model: nn.Module, test_files: List[Path]) -> None:
    """
    Predict and save results with GPU optimization
    
    Args:
        model: Trained model
        test_files: List of paths to test files
    """
    logger.info("Starting prediction")
    
    # Create test dataset and data loader
    ds = TestDataset(
        test_files, 
        preload=len(test_files) < 500  # Preload only if not too many files
    )
    
    # Increase batch size for prediction
    batch_size = CONFIG['batch_size'] * 2  # Can use larger batch size for prediction
    
    dl = DataLoader(
        ds, 
        batch_size=batch_size, 
        num_workers=CONFIG['num_workers'], 
        pin_memory=CONFIG['pin_memory'],
        prefetch_factor=CONFIG['prefetch_factor']
    )
    
    # Prepare column names for CSV file
    x_cols = [f'x_{i}' for i in range(1, 70, 2)]
    fieldnames = ['oid_ypos'] + x_cols
    
    # Save predictions to CSV file
    with open(CONFIG['output_file'], 'wt', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_predictions = 0
        
        # Switch model to evaluation mode
        model.eval()
        
        # Clear CUDA cache before prediction
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Iterate over batches of test data
        for inputs, oids_test in tqdm(dl, desc='Prediction'):
            # Move data to device with non-blocking transfer
            inputs = inputs.to(CONFIG['device'], non_blocking=True)
            
            # Use mixed precision for faster computation
            with torch.inference_mode(), autocast(enabled=CONFIG['use_mixed_precision'] and torch.cuda.is_available()):
                # Forward pass
                outputs = model(inputs)
            
            # Move results to CPU and convert to numpy
            y_preds = outputs[:, 0].cpu().numpy()
            
            # Save predictions
            for y_pred, oid_test in zip(y_preds, oids_test):
                for y_pos in range(70):
                    row = dict(
                        zip(
                            x_cols,
                            [y_pred[y_pos, x_pos] for x_pos in range(1, 70, 2)]
                        )
                    )
                    row['oid_ypos'] = f"{oid_test}_y_{y_pos}"
                    
                    writer.writerow(row)
                    total_predictions += 1
                
                # Periodic memory cleanup for large number of predictions
                if total_predictions % 10000 == 0:
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
        
        logger.info(f"Prediction completed. Saved {total_predictions} rows to {CONFIG['output_file']}")


def main() -> None:
    """Main function"""
    logger.info("Starting")
    
    try:
        # Print system information
        logger.info(f"Device: {CONFIG['device']}")
        if torch.cuda.is_available():
            logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"Available GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
            logger.info(f"Mixed precision: {'enabled' if CONFIG['use_mixed_precision'] else 'disabled'}")
        
        # Create data loaders
        train_loader, valid_loader = create_data_loaders()
        
        # Create model
        model = EnhancedSmartConvNet().to(CONFIG['device'])
        
        # Print model information
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Model created: {model.__class__.__name__}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Train model
        train_model(model, train_loader, valid_loader)
        
        # Get list of test files
        test_files = list(Path(CONFIG['test_data_path']).glob('*.npy'))
        logger.info(f"Found {len(test_files)} test files")
        
        # Predict and save results
        predict_and_save(model, test_files)
        
        logger.info("Completed successfully")
    except Exception as e:
        logger.error(f"Error occurred: {e}")
        raise


if __name__ == "__main__":
    # Set deterministic behavior for reproducibility
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run main function
    main()




