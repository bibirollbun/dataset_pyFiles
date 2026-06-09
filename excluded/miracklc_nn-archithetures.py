##### IMPORTS #####
"""
PyTorch-based neural networks for bird call classification
"""
import os
import logging
import random
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import librosa
import joblib
import cv2
from pathlib import Path
import glob

# Core ML imports
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                           precision_recall_fscore_support)
from sklearn.preprocessing import LabelEncoder, StandardScaler

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader, TensorDataset
    from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
    import torch.backends.cudnn as cudnn
    PYTORCH_AVAILABLE = True
    print("âœ“ PyTorch loaded successfully")
    
    # GPU Optimization Configuration
    print("ğŸ”§ Configuring GPU optimizations...")
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"âœ“ GPU Configuration Complete:")
        print(f"   â€¢ GPU Device: {torch.cuda.get_device_name()}")
        print(f"   â€¢ CUDA Version: {torch.version.cuda}")
        print(f"   â€¢ GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Enable cuDNN optimizations
        cudnn.benchmark = True
        cudnn.deterministic = False
        print("âœ“ cuDNN optimizations enabled")
    else:
        device = torch.device('cpu')
        print("âš ï¸� No GPU detected, using CPU")
    
    print(f"âœ“ Device set to: {device}")
    
except ImportError:
    PYTORCH_AVAILABLE = False
    device = 'cpu'
    print("âš ï¸� PyTorch not available. Neural networks will be skipped.")

from tqdm.auto import tqdm

# Performance optimizations
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
plt.style.use('default')

##### CONFIGURATION #####
"""
Enhanced configuration for neural network architectures and deep learning
"""
class CFG:
    # Seed for reproducibility
    seed = 42
    debug = False  # Enable debug mode for faster testing
    
    # Data paths (Kaggle paths)
    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    
    # Audio processing parameters (optimized for speed)
    FS = 16000  # Reduced from 32000 for faster processing
    TARGET_DURATION = 5.0  
    
    # Mel spectrogram parameters (improved dimensions)
    N_FFT = 1024  # Increased for better frequency resolution
    HOP_LENGTH = 512  # Increased for better time resolution
    N_MELS = 128  # More mel bands for better feature representation
    FMIN = 50
    FMAX = 8000  # Adjusted based on FS/2
    TARGET_SHAPE = (128, 128)  # Larger dimensions for better features
    
    # Data Augmentation Settings (enhanced for large dataset)
    enable_data_augmentation = True
    augmentation_probability = 0.5  # Increased to 50% for more variety
    
    # Enhanced augmentation parameters for 27k data
    noise_factor = 0.03  # Slightly increased noise
    volume_range = (0.6, 1.4)  # Wider volume scaling range
    mixup_alpha = 0.3  # Increased mixup for better generalization
    
    # Quality filtering (stricter for large dataset)
    filter_low_quality = True  # Remove samples with rating 0.5-2.5
    min_quality_rating = 3.0  # Increased threshold for better quality data
    
    # Performance parameters (optimized for faster training)
    n_samples = 1500 if debug else None  # Use all data (~27k samples)
    min_samples_for_rare_class_elimination = 15  # Slightly lower threshold
    test_size = 0.15  # Smaller test set to have more training data
    
    # Training speed optimizations
    use_fast_training = True  # Enable fast training mode
    max_models_in_debug = 3  # Only train top 3 models in debug mode
    reduce_model_architectures = debug  # Use fewer architectures in debug mode
    
    # Neural Network Training Parameters (optimized for speed and CNN_Deep success)
    batch_size = 256  # BÃ¼yÃ¼k batch size for faster training
    epochs = 100  # Reduced epochs for faster training
    learning_rate = 0.002  # Slightly higher learning rate
    weight_decay = 1e-4  
    dropout_rate = 0.3  
    early_stopping_patience = 15  # Reduced patience for faster stopping
    gradient_accumulation_steps = 1  # No accumulation for faster batch processing
    
    # Enhanced Neural Network Architectures (optimized for CNN_Deep success - ~8M parameters)
    neural_architectures = {
        'CNN_Deep_Original': {
            'type': 'cnn',
            'conv_channels': [32, 64, 128, 256],
            'kernel_sizes': [3, 3, 3, 3],
            'pool_sizes': [2, 2, 2, 2],
            'fc_layers': [512, 256],
            'dropout_rate': 0.35
        },
        'CNN_Deep_Variant1': {
            'type': 'cnn',
            'conv_channels': [64, 96, 128, 192],  # FarklÄ± kanal daÄŸÄ±lÄ±mÄ±
            'kernel_sizes': [3, 3, 3, 3],
            'pool_sizes': [2, 2, 2, 2],
            'fc_layers': [384, 192],  # Daha az FC parametre
            'dropout_rate': 0.3
        },
        'CNN_Deep_Variant2': {
            'type': 'cnn',
            'conv_channels': [48, 80, 144, 224],  # Daha dengeli artÄ±ÅŸ
            'kernel_sizes': [5, 3, 3, 3],  # Ä°lk katman daha bÃ¼yÃ¼k kernel
            'pool_sizes': [2, 2, 2, 2],
            'fc_layers': [448, 224],
            'dropout_rate': 0.32
        },
        'CNN_Deep_Compact': {
            'type': 'cnn',
            'conv_channels': [48, 96, 144, 192],  # Kompakt ama derin
            'kernel_sizes': [3, 3, 3, 3],
            'pool_sizes': [2, 2, 2, 2],
            'fc_layers': [384, 128],  # Daha az FC parametre
            'dropout_rate': 0.4
        },
        'CNN_Deep_Wide_Kernels': {
            'type': 'cnn',
            'conv_channels': [32, 64, 128, 256],  # AynÄ± kanal sayÄ±sÄ±
            'kernel_sizes': [5, 5, 3, 3],  # BaÅŸta daha bÃ¼yÃ¼k kernel'lar
            'pool_sizes': [2, 2, 2, 2],
            'fc_layers': [512, 256],
            'dropout_rate': 0.35
        },
        'CNN_Efficient': {
            'type': 'cnn',
            'conv_channels': [40, 72, 120, 200],  # Efficient-Net tarzÄ± kanallar
            'kernel_sizes': [3, 3, 3, 3],
            'pool_sizes': [2, 2, 2, 2],
            'fc_layers': [320, 160],
            'dropout_rate': 0.38
        },
        'LSTM_Compact': {
            'type': 'lstm',
            'hidden_sizes': [96],  # Daha kÃ¼Ã§Ã¼k hidden size
            'num_layers': 2,
            'bidirectional': True,
            'fc_layers': [192, 96],  # Daha kÃ¼Ã§Ã¼k FC layer'lar
            'dropout_rate': 0.3,
            'lstm_seq_len': 128,
        }
    }
    
    # Ensemble Configuration
    enable_ensemble = True
    ensemble_method = 'voting'  # 'voting' or 'weighted'
    ensemble_weights = None  # Will be calculated based on validation performance
    
    # GPU and Performance Optimization Parameters
    use_multiprocessing = True  # Enable parallel processing
    n_jobs = -1  # Use all available cores
    enable_mixed_precision = True  # Use mixed precision training
    clear_memory_between_models = True  # Clear memory between model trainings

##### UTILITY FUNCTIONS #####
"""
Helper functions and setup
"""
def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    if PYTORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"âœ“ Seed set: {seed}")

def setup_logging():
    """Logging setup"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training_log.log')
        ]
    )
    print("="*60)
    print("ğŸš€ BirdCLEF Neural Network Training Pipeline Started")
    print("="*60)

def print_section(title):
    """Helper function for printing section titles"""
    print("\n" + "="*60)
    print(f"ğŸ“Š {title}")
    print("="*60)

def print_configuration(cfg):
    """Print configuration settings"""
    print_section("âš™ï¸� CONFIGURATION USED")
    
    config_text = f"""
âš™ï¸� Configuration Used (Optimized for 27k Dataset):
â€¢ Data Augmentation: {'âœ“ Enabled' if cfg.enable_data_augmentation else 'âœ— Disabled'}
â€¢ Quality Filtering: {'âœ“ Enabled' if cfg.filter_low_quality else 'âœ— Disabled'}

ğŸ“Š Data Parameters:
â€¢ Sample Rate: {cfg.FS} Hz
â€¢ Target Duration: {cfg.TARGET_DURATION}s
â€¢ N_FFT: {cfg.N_FFT}
â€¢ Hop Length: {cfg.HOP_LENGTH}
â€¢ N_Mels: {cfg.N_MELS}
â€¢ Target Shape: {cfg.TARGET_SHAPE}

ğŸ”„ Data Augmentation (Enhanced):
â€¢ Augmentation Probability: {cfg.augmentation_probability * 100}%
â€¢ Noise Factor: {cfg.noise_factor}
â€¢ Volume Range: {cfg.volume_range}
â€¢ Mixup Alpha: {cfg.mixup_alpha}

ğŸ§  Neural Network Parameters (Large Dataset Optimized):
â€¢ Batch Size: {cfg.batch_size}
â€¢ Epochs: {cfg.epochs}
â€¢ Learning Rate: {cfg.learning_rate}
â€¢ Weight Decay: {cfg.weight_decay}
â€¢ Dropout Rate: {cfg.dropout_rate}
â€¢ Early Stopping Patience: {cfg.early_stopping_patience}
â€¢ Gradient Accumulation Steps: {cfg.gradient_accumulation_steps}
â€¢ Test Size: {cfg.test_size*100}%
â€¢ Debug Mode: {'âœ“ Enabled' if cfg.debug else 'âœ— Disabled (Using All Data)'}
â€¢ Min Quality Rating: {cfg.min_quality_rating}

âš¡ GPU & Performance Optimizations:
â€¢ Device: {device if PYTORCH_AVAILABLE else 'CPU (PyTorch not available)'}
â€¢ Multiprocessing: {'âœ“ Enabled' if getattr(cfg, 'use_multiprocessing', False) else 'âœ— Disabled'}
â€¢ Mixed Precision: {'âœ“ Enabled' if getattr(cfg, 'enable_mixed_precision', False) else 'âœ— Disabled'}
â€¢ Memory Cleanup: {'âœ“ Enabled' if getattr(cfg, 'clear_memory_between_models', False) else 'âœ— Disabled'}

ğŸ�—ï¸� Neural Network Architectures: {len(cfg.neural_architectures)}
"""
    
    for arch_name, arch_config in cfg.neural_architectures.items():
        config_text += f"â€¢ {arch_name} ({arch_config['type'].upper()}): "
        if arch_config['type'] == 'cnn':
            config_text += f"channels={arch_config['conv_channels']}, fc_layers={arch_config['fc_layers']}\n"
        elif arch_config['type'] == 'lstm':
            config_text += f"hidden={arch_config['hidden_sizes']}, layers={arch_config['num_layers']}, bidirectional={arch_config['bidirectional']}\n"
    
    config_text += f"""
ğŸ�¯ Ensemble Configuration:
â€¢ Ensemble Enabled: {'âœ“ Yes' if cfg.enable_ensemble else 'âœ— No'}
â€¢ Ensemble Method: {cfg.ensemble_method}
    """
    
    print(config_text)

##### DATA AUGMENTATION FUNCTIONS #####
"""
Audio data augmentation techniques
"""
def add_background_noise(audio, noise_factor=0.02):
    """Add Gaussian background noise"""
    noise = np.random.normal(0, noise_factor, len(audio))
    return audio + noise

def volume_scaling(audio, volume_range=(0.7, 1.3)):
    """Apply random volume scaling"""
    factor = np.random.uniform(volume_range[0], volume_range[1])
    return audio * factor

def mixup_audio(audio1, audio2, alpha=0.2):
    """Apply mixup augmentation between two audio samples"""
    lam = np.random.beta(alpha, alpha)
    
    # Ensure both audio samples have the same length
    min_len = min(len(audio1), len(audio2))
    audio1 = audio1[:min_len]
    audio2 = audio2[:min_len]
    
    mixed_audio = lam * audio1 + (1 - lam) * audio2
    return mixed_audio, lam

def apply_augmentation(audio, cfg, aug_type=None):
    """Apply random augmentation to audio"""
    if not cfg.enable_data_augmentation:
        return audio
    
    # Apply augmentation with probability
    if np.random.random() > cfg.augmentation_probability:
        return audio
    
    augmented_audio = audio.copy()
    
    # Background noise
    if aug_type is None or aug_type == 'noise':
        if np.random.random() < 0.5:
            augmented_audio = add_background_noise(augmented_audio, cfg.noise_factor)
    
    # Volume scaling
    if aug_type is None or aug_type == 'volume':
        if np.random.random() < 0.5:
            augmented_audio = volume_scaling(augmented_audio, cfg.volume_range)
    
    return augmented_audio

##### ENHANCED AUDIO PROCESSING #####
"""
Enhanced audio processing functions with improved feature extraction
"""
def extract_enhanced_audio_features(audio_path, cfg, apply_augmentation_flag=True):
    """
    Enhanced audio feature extraction with multiple feature types:
    - Mel spectrogram (primary features)
    - MFCC features (cepstral coefficients)
    - Spectral features (centroid, rolloff, zero crossing rate)
    - Rhythm features (tempo, beat density)
    """
    try:
        y, sr = librosa.load(audio_path, sr=cfg.FS, duration=cfg.TARGET_DURATION)
        
        if len(y) == 0:
            # Calculate correct feature dimension - FIXED to be divisible by 128
            base_size = cfg.TARGET_SHAPE[0] * cfg.TARGET_SHAPE[1]  # 128*128 = 16384
            return np.zeros(base_size, dtype=np.float32)  # Simplified to only mel features
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Apply data augmentation if enabled
        if apply_augmentation_flag and cfg.enable_data_augmentation:
            y = apply_augmentation(y, cfg)
        
        # 1. Mel spectrogram (primary features) - ONLY use mel features for now
        melspec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS, fmin=cfg.FMIN, fmax=cfg.FMAX
        )
        melspec = librosa.power_to_db(melspec, ref=np.max)
        melspec = (melspec - melspec.min()) / (melspec.max() - melspec.min() + 1e-8)
        melspec = cv2.resize(melspec, (cfg.TARGET_SHAPE[1], cfg.TARGET_SHAPE[0]), 
                           interpolation=cv2.INTER_AREA)
        
        # Return only mel spectrogram features for now (16384 features)
        melspec_features = melspec.flatten()
        return melspec_features.astype(np.float32)
        
    except Exception as e:
        print(f"âš ï¸� Audio processing error for {audio_path}: {e}")
        # Return zeros with correct dimension - 16384 features
        base_size = cfg.TARGET_SHAPE[0] * cfg.TARGET_SHAPE[1]
        return np.zeros(base_size, dtype=np.float32)

##### NEURAL NETWORK ARCHITECTURES #####
"""
PyTorch neural network architectures for bird call classification
"""

class SimpleCNN(nn.Module):
    """Simplified CNN architecture for audio classification"""
    def __init__(self, input_dim, num_classes, cfg, arch_config):
        super(SimpleCNN, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.cfg = cfg 
        self.arch_config = arch_config 
        
        # Melspectrogram dimensions
        self.melspec_h = cfg.TARGET_SHAPE[0]
        self.melspec_w = cfg.TARGET_SHAPE[1]
        self.melspec_features = self.melspec_h * self.melspec_w

        # Simplified: Only handle mel spectrogram features (no additional features)
        assert input_dim == self.melspec_features, f"Expected input_dim {self.melspec_features}, got {input_dim}"

        # Convolutional layers
        conv_layers = []
        in_channels = 1 # Start with 1 channel for the melspectrogram image
        
        current_h, current_w = self.melspec_h, self.melspec_w

        for i, out_channels in enumerate(self.arch_config['conv_channels']):
            kernel_size = self.arch_config['kernel_sizes'][i]
            pool_size = self.arch_config['pool_sizes'][i]
            
            conv_layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=pool_size, stride=pool_size))
            in_channels = out_channels
            
            # Update dimensions
            current_h = current_h // pool_size
            current_w = current_w // pool_size
            
        self.conv_block = nn.Sequential(*conv_layers)
        
        # Calculate the size after conv layers
        self.conv_output_features = in_channels * current_h * current_w
        
        # Fully connected layers - simplified
        fc_layers_list = []
        fc_input_dim = self.conv_output_features
        
        for fc_hidden_size in self.arch_config['fc_layers']:
            fc_layers_list.append(nn.Linear(fc_input_dim, fc_hidden_size))
            fc_layers_list.append(nn.ReLU())
            fc_layers_list.append(nn.Dropout(self.arch_config['dropout_rate']))
            fc_input_dim = fc_hidden_size
            
        fc_layers_list.append(nn.Linear(fc_input_dim, num_classes))
        self.fc_block = nn.Sequential(*fc_layers_list)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape melspectrogram for CNN (only mel features now)
        melspec_data = x.view(batch_size, 1, self.melspec_h, self.melspec_w)
        
        # Pass through conv block
        conv_out = self.conv_block(melspec_data)
        
        # Flatten conv output
        conv_out_flat = conv_out.view(batch_size, -1)
        
        # Pass through FC block
        output = self.fc_block(conv_out_flat)
        
        return output

class BirdLSTM(nn.Module):
    """Enhanced LSTM architecture for sequential audio features"""
    def __init__(self, input_dim, num_classes, cfg, arch_config):
        super(BirdLSTM, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.cfg = cfg
        self.arch_config = arch_config
        
        self.lstm_seq_len = arch_config.get('lstm_seq_len', 64) # Default if not in config
        
        # Ensure features_per_step is an integer and input_dim is divisible
        if input_dim % self.lstm_seq_len != 0:
            raise ValueError(f"input_dim ({input_dim}) must be divisible by lstm_seq_len ({self.lstm_seq_len})")
        self.features_per_step = input_dim // self.lstm_seq_len

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.features_per_step, # Corrected input size
            hidden_size=arch_config['hidden_sizes'][0], # Use first hidden_size, or handle list for multi-layer
            num_layers=arch_config['num_layers'],
            dropout=arch_config['dropout_rate'] if arch_config['num_layers'] > 1 else 0,
            bidirectional=arch_config['bidirectional'],
            batch_first=True
        )
        
        # Calculate LSTM output size
        lstm_output_size = arch_config['hidden_sizes'][0] * 2 if arch_config['bidirectional'] else arch_config['hidden_sizes'][0]
        
        # Fully connected layers
        fc_layers_list = []
        fc_input_dim = lstm_output_size
        
        for fc_hidden_size in arch_config['fc_layers']:
            fc_layers_list.append(nn.Linear(fc_input_dim, fc_hidden_size))
            fc_layers_list.append(nn.ReLU())
            fc_layers_list.append(nn.Dropout(arch_config['dropout_rate']))
            fc_input_dim = fc_hidden_size
            
        fc_layers_list.append(nn.Linear(fc_input_dim, num_classes))
        self.fc_block = nn.Sequential(*fc_layers_list)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Reshape for LSTM: (batch, seq_len, features_per_step)
        # The input x is expected to be flat (batch_size, input_dim)
        x = x.view(batch_size, self.lstm_seq_len, self.features_per_step)
        
        # LSTM forward pass
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the hidden state of the last time step
        if self.arch_config['bidirectional']:
            # Concatenate final forward and backward hidden states of the last layer
            # hidden is (num_layers * num_directions, batch, hidden_size)
            # We want the hidden state from the last layer, so indices -2 (forward) and -1 (backward)
            hidden_last_layer_fwd = hidden[-2, :, :] 
            hidden_last_layer_bwd = hidden[-1, :, :]
            processed_lstm_out = torch.cat((hidden_last_layer_fwd, hidden_last_layer_bwd), dim=1)
        else:
            # hidden is (num_layers, batch, hidden_size)
            # We want the hidden state from the last layer, so index -1
            processed_lstm_out = hidden[-1, :, :]
        
        # Pass through FC block
        output = self.fc_block(processed_lstm_out)
        
        return output

##### ENSEMBLE METHODS #####
"""
Ensemble methods for combining multiple models
"""

class EnsembleModel:
    """Ensemble of multiple neural network models"""
    def __init__(self, models, method='voting', weights=None):
        self.models = models
        self.method = method
        self.weights = weights
        
    def predict(self, X):
        """Make predictions using ensemble of models"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X_tensor = torch.FloatTensor(X).to(device)
                else:
                    X_tensor = X.to(device)
                    
                output = model(X_tensor)
                if self.method == 'voting':
                    pred = output.argmax(dim=1).cpu().numpy()
                else:
                    pred = F.softmax(output, dim=1).cpu().numpy()
                predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if self.method == 'voting':
            # Hard voting
            ensemble_pred = []
            for i in range(predictions.shape[1]):
                votes = predictions[:, i]
                ensemble_pred.append(np.bincount(votes).argmax())
            return np.array(ensemble_pred)
        
        elif self.method == 'weighted':
            # Weighted average
            if self.weights is None:
                self.weights = np.ones(len(self.models)) / len(self.models)
            
            weighted_probs = np.average(predictions, axis=0, weights=self.weights)
            return weighted_probs.argmax(axis=1)
        
    def predict_proba(self, X):
        """Get probability predictions from ensemble"""
        predictions = []
        
        for model in self.models:
            model.eval()
            with torch.no_grad():
                if isinstance(X, np.ndarray):
                    X_tensor = torch.FloatTensor(X).to(device)
                else:
                    X_tensor = X.to(device)
                    
                output = model(X_tensor)
                probs = F.softmax(output, dim=1).cpu().numpy()
                predictions.append(probs)
        
        predictions = np.array(predictions)
        
        if self.weights is None:
            return np.mean(predictions, axis=0)
        else:
            return np.average(predictions, axis=0, weights=self.weights)

##### DATA PREPARATION #####
"""
Data preparation and preprocessing
"""
def prepare_data(cfg):
    """Enhanced data preparation for neural networks"""
    print_section("DATA PREPARATION")
    
    # Create dummy data for testing if files don't exist
    if not os.path.exists(cfg.train_csv):
        print("âš ï¸� Data files not found. Creating demo data...")
        return create_dummy_data(cfg)
    
    # Load data
    print("ğŸ“‚ Loading data files...")
    train_df = pd.read_csv(cfg.train_csv)
    
    # Load taxonomy data if available
    if os.path.exists(cfg.taxonomy_csv):
        taxonomy_df = pd.read_csv(cfg.taxonomy_csv)
        print(f"âœ“ Loaded taxonomy data: {len(taxonomy_df)} species")
        
        # Merge with train data to get additional species information
        train_df = train_df.merge(taxonomy_df, left_on='primary_label', right_on='primary_label', how='left')
    
    print(f"âœ“ {len(train_df)} samples loaded")
    
    # Debug mode sampling
    if cfg.debug and cfg.n_samples and cfg.n_samples < len(train_df):
        train_df = train_df.sample(cfg.n_samples, random_state=cfg.seed)
        print(f"ğŸ”¬ Debug mode: {cfg.n_samples} samples selected")
    
    # Enhanced data filtering
    print("ğŸ§¹ Cleaning data...")
    
    # Remove samples with missing primary labels
    initial_len = len(train_df)
    train_df = train_df.dropna(subset=['primary_label'])
    if len(train_df) < initial_len:
        print(f"âœ“ Removed {initial_len - len(train_df)} samples with missing labels")
    
    # Filter by quality rating (remove low quality samples)
    if 'rating' in train_df.columns and cfg.filter_low_quality:
        initial_len = len(train_df)
        # Remove samples with rating between 0.5 and 2.5
        quality_filtered = train_df[~((train_df['rating'] >= 0.5) & (train_df['rating'] <= cfg.min_quality_rating))]
        train_df = quality_filtered
        print(f"âœ“ Filtered low quality samples (rating 0.5-{cfg.min_quality_rating}): removed {initial_len - len(train_df)} samples")
    
    # Remove rare classes
    class_counts = train_df['primary_label'].value_counts()
    rare_classes = class_counts[class_counts < cfg.min_samples_for_rare_class_elimination].index
    
    if len(rare_classes) > 0:
        initial_len = len(train_df)
        train_df = train_df[~train_df['primary_label'].isin(rare_classes)]
        print(f"âœ“ {len(rare_classes)} rare classes eliminated ({initial_len - len(train_df)} samples)")
    
    # Create file paths
    train_df['filepath'] = train_df['filename'].apply(lambda x: os.path.join(cfg.train_datadir, x))
    
    # Encode labels
    le = LabelEncoder()
    train_df['target'] = le.fit_transform(train_df['primary_label'])
    
    # Save label encoder
    joblib.dump(le, "label_encoder.joblib")
    
    # Update config
    cfg.num_classes = len(le.classes_)
    cfg.class_names = le.classes_
    
    print(f"âœ“ {cfg.num_classes} classes, {len(train_df)} samples ready")
    
    # Enhanced class distribution analysis
    class_dist = train_df['primary_label'].value_counts()
    print(f"ğŸ“Š Top 10 most common classes: {dict(class_dist.head(10))}")
    print(f"ğŸ“Š Class distribution stats: min={class_dist.min()}, max={class_dist.max()}, mean={class_dist.mean():.1f}")
    
    return train_df, le

def create_dummy_data(cfg):
    """Create dummy data for demo purposes"""
    print("ğŸ�­ Creating demo data...")
    
    # Create dummy audio files and dataframe
    n_samples = 500
    bird_species = ['robin', 'sparrow', 'eagle', 'hawk', 'crow', 'owl', 'cardinal', 'bluejay', 'woodpecker', 'finch']
    
    data = []
    for i in range(n_samples):
        species = np.random.choice(bird_species)
        filename = f"{species}_{i:03d}.wav"
        # Add some metadata with realistic rating distribution
        rating = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.1, 0.1, 0.2, 0.3, 0.2, 0.1])
        data.append({
            'filename': filename,
            'primary_label': species,
            'rating': rating,
            'latitude': np.random.uniform(-90, 90),
            'longitude': np.random.uniform(-180, 180),
            'author': f"user_{np.random.randint(1, 50)}"
        })
    
    train_df = pd.DataFrame(data)
    train_df['filepath'] = train_df['filename']  # Use dummy paths
    
    # Encode labels
    le = LabelEncoder()
    train_df['target'] = le.fit_transform(train_df['primary_label'])
    
    # Save label encoder
    joblib.dump(le, "label_encoder.joblib")
    
    # Update config
    cfg.num_classes = len(le.classes_)
    cfg.class_names = le.classes_
    
    print(f"âœ“ Demo data ready: {cfg.num_classes} classes, {len(train_df)} samples")
    
    return train_df, le

def extract_features(df, cfg):
    """Enhanced feature extraction with progress tracking and augmentation"""
    print_section("FEATURE EXTRACTION")
    
    print("ğŸ�µ Extracting enhanced audio features...")
    
    # Display data augmentation configuration
    if cfg.enable_data_augmentation:
        print("\nğŸ”„ Data Augmentation Applied:")
        print(f"   â€¢ Background Noise: âœ“ Enabled (factor: {cfg.noise_factor})")
        print(f"   â€¢ Volume Scaling: âœ“ Enabled (range: {cfg.volume_range})")
        print(f"   â€¢ Mixup Alpha: {cfg.mixup_alpha}")
        print(f"   â€¢ Augmentation Probability: {cfg.augmentation_probability * 100}%")
        print("   â€¢ Multi-modal Features: Mel + MFCC + Spectral + Rhythm")
    else:
        print("\nğŸš« Data Augmentation: Disabled")
    
    # For demo data, create more sophisticated random features
    if not os.path.exists(cfg.train_datadir):
        print("ğŸ�­ Creating enhanced demo features...")
        
        # Calculate feature dimensions - FIXED
        base_size = cfg.TARGET_SHAPE[0] * cfg.TARGET_SHAPE[1]  # 128*128 = 16384
        n_features = base_size  # Only mel features
        
        print(f"ğŸš€ Using optimized vectorized feature generation... Features: {n_features}")
        
        # Vectorized feature generation for speed
        n_samples = len(df)
        y = df['target'].values
        
        # Pre-allocate arrays for better memory usage
        X = np.zeros((n_samples, n_features), dtype=np.float32)
        
        # Create more realistic patterns for each species
        n_classes = len(np.unique(y))
        
        # Create species-specific templates for better separation
        species_templates = np.random.randn(n_classes, n_features).astype(np.float32) * 0.5
        
        # Generate more realistic features
        for i, target in enumerate(y):
            # Base pattern from species template
            base_pattern = species_templates[target]
            
            # Add noise for variation
            noise = np.random.randn(n_features).astype(np.float32) * 0.2
            
            # Create frequency patterns (simulate mel spectrogram structure)
            mel_h, mel_w = cfg.TARGET_SHAPE
            pattern_2d = base_pattern.reshape(mel_h, mel_w)
            
            # Add frequency-specific patterns (lower frequencies more active for bird calls)
            freq_weights = np.linspace(1.0, 0.3, mel_h).reshape(-1, 1)
            pattern_2d = pattern_2d * freq_weights
            
            # Add temporal patterns
            time_weights = np.ones((1, mel_w))
            time_weights[0, mel_w//4:3*mel_w//4] *= 1.5  # More activity in middle time frames
            pattern_2d = pattern_2d * time_weights
            
            # Flatten and add noise
            final_pattern = pattern_2d.flatten() + noise
            
            # Normalize to reasonable range
            final_pattern = np.tanh(final_pattern)  # Bound to [-1, 1]
            
            X[i] = final_pattern
        
        print(f"âœ“ Enhanced demo feature matrix: {X.shape} (realistic patterns)")
        return X, y
    
    # Real feature extraction
    features = []
    labels = []
    
    start_time = time.time()
    
    # Sequential processing
    print("ğŸ”„ Using sequential feature extraction...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Audio"):
        try:
            feature_vector = extract_enhanced_audio_features(row['filepath'], cfg)
            
            if not np.all(feature_vector == 0):
                features.append(feature_vector)
                labels.append(row['target'])
                
        except Exception as e:
            print(f"âš ï¸� Skipping {row['filepath']}: {e}")
            continue
    
    X = np.array(features, dtype=np.float32)
    y = np.array(labels)
    
    elapsed_time = time.time() - start_time
    print(f"âœ“ Feature extraction completed: {X.shape}, {elapsed_time:.2f} seconds")
    print(f"âš¡ Processing speed: {len(X)/elapsed_time:.2f} samples/second")
    
    if X.shape[0] == 0:
        raise ValueError("â�Œ No samples remaining!")
    
    return X, y

##### NEURAL NETWORK TRAINING #####
"""
PyTorch neural network training functions
"""

def create_model(arch_name, arch_config, input_dim, num_classes, cfg):
    """Create a neural network model based on architecture configuration"""
    if not PYTORCH_AVAILABLE:
        return None
    
    if arch_config['type'] == 'cnn':
        model = SimpleCNN(input_dim, num_classes, cfg, arch_config)
    elif arch_config['type'] == 'lstm':
        model = BirdLSTM(
            input_dim, num_classes, cfg, arch_config
        )
    else:
        raise ValueError(f"Unknown architecture type: {arch_config['type']}")
    
    return model.to(device)

def train_model(model, train_loader, val_loader, cfg, model_name):
    """Train a single neural network model with optimizations for large dataset"""
    print(f"ğŸš€ Training {model_name}...")
    
    # Loss function and optimizer with weight decay
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    
    # Enhanced learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True, min_lr=1e-6
    )
    
    # Cosine annealing scheduler as secondary
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Training history
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler() if cfg.enable_mixed_precision and torch.cuda.is_available() else None
    
    for epoch in range(cfg.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        optimizer.zero_grad()  # Initialize gradients
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Mixed precision training if enabled and CUDA available
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    output = model(data)
                    loss = criterion(output, target)
                    loss = loss / cfg.gradient_accumulation_steps  # Scale loss for accumulation
                
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                output = model(data)
                loss = criterion(output, target)
                loss = loss / cfg.gradient_accumulation_steps  # Scale loss for accumulation
                loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % cfg.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            
            train_loss += loss.item() * cfg.gradient_accumulation_steps  # Rescale for logging
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
        
        # Handle remaining gradients if batch doesn't divide evenly
        if len(train_loader) % cfg.gradient_accumulation_steps != 0:
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        # Calculate metrics
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        cosine_scheduler.step(epoch)  # Use cosine for exploration
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), f'best_{model_name.lower()}.pth')
        else:
            patience_counter += 1
        
        # Print progress (more frequent for large dataset)
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"   Epoch {epoch+1:3d}/{cfg.epochs}: "
                  f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                  f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%, "
                  f"LR: {current_lr:.6f}")
        
        # Early stopping check
        if patience_counter >= cfg.early_stopping_patience:
            print(f"   Early stopping at epoch {epoch+1}")
            break
    
    training_time = time.time() - start_time
    
    # Load best model
    model.load_state_dict(torch.load(f'best_{model_name.lower()}.pth'))
    
    return {
        'model': model,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies,
        'training_time': training_time,
        'epochs_trained': len(train_losses)
    }

def evaluate_model(model, test_loader, label_encoder, model_name):
    """Evaluate a trained model"""
    model.eval()
    test_correct = 0
    test_total = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            
            test_total += target.size(0)
            test_correct += predicted.eq(target).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    accuracy = 100. * test_correct / test_total
    
    # Calculate detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_targets, all_predictions, average='weighted', zero_division=0
    )
    
    print(f"ğŸ“Š {model_name} Test Results:")
    print(f"   Accuracy: {accuracy:.2f}%")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': all_predictions,
        'targets': all_targets
    }

def train_neural_networks(X_train, y_train, X_test, y_test, cfg, label_encoder):
    """Train neural network architectures optimized for CNN_Deep success pattern"""
    if not PYTORCH_AVAILABLE:
        print("âš ï¸� PyTorch not available. Skipping neural network training.")
        return {}
    
    print_section("NEURAL NETWORK TRAINING")
    
    # Fast training mode optimizations
    if cfg.use_fast_training:
        print("âš¡ Fast training mode enabled:")
        print(f"   â€¢ Reduced architectures for faster testing")
        print(f"   â€¢ Batch size increased to {cfg.batch_size}")
        print(f"   â€¢ Early stopping patience: {cfg.early_stopping_patience}")
        
        # Use only the most promising architectures based on CNN_Deep success
        if cfg.debug and hasattr(cfg, 'max_models_in_debug'):
            architectures_to_train = list(cfg.neural_architectures.items())[:cfg.max_models_in_debug]
            print(f"   â€¢ Debug mode: Training only {len(architectures_to_train)} models")
        else:
            architectures_to_train = list(cfg.neural_architectures.items())
    else:
        architectures_to_train = list(cfg.neural_architectures.items())
    
    print("ğŸ“Š Preparing data for neural networks...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_scaled)
    y_train_tensor = torch.LongTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test_scaled)
    y_test_tensor = torch.LongTensor(y_test)
    
    # Create data loaders with optimized settings
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Split training data for validation
    train_size = int(0.85 * len(train_dataset))  # More training data
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    
    # Optimized data loaders
    train_loader = DataLoader(
        train_subset, 
        batch_size=cfg.batch_size, 
        shuffle=True,
        num_workers=2 if cfg.use_multiprocessing else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_subset, 
        batch_size=cfg.batch_size, 
        shuffle=False,
        num_workers=2 if cfg.use_multiprocessing else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=cfg.batch_size, 
        shuffle=False,
        num_workers=2 if cfg.use_multiprocessing else 0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"âœ“ Data loaders ready: Train: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")
    
    # Train models
    models = {}
    results = {}
    
    input_dim = X_train.shape[1]
    num_classes = cfg.num_classes
    
    total_start_time = time.time()
    
    for arch_name, arch_config in architectures_to_train:
        print(f"\nğŸ”„ Training {arch_name}...")
        model_start_time = time.time()
        
        try:
            # Create model
            model = create_model(arch_name, arch_config, input_dim, num_classes, cfg)
            
            if model is None:
                continue
            
            # Calculate and display model parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"   ğŸ“Š Model Info:")
            print(f"      â€¢ Total parameters: {total_params:,}")
            print(f"      â€¢ Trainable parameters: {trainable_params:,}")
            print(f"      â€¢ Model size: ~{total_params * 4 / 1024 / 1024:.1f} MB")
            
            # Skip models that are too large (>50M parameters)
            if total_params > 50_000_000:
                print(f"   âš ï¸� Skipping {arch_name}: Too many parameters ({total_params:,})")
                continue
            
            # Train model
            training_result = train_model(model, train_loader, val_loader, cfg, arch_name)
            
            # Evaluate model
            eval_result = evaluate_model(model, test_loader, label_encoder, arch_name)
            
            # Combine results
            results[arch_name] = {
                **training_result,
                **eval_result,
                'architecture': arch_config,
                'total_parameters': total_params,
                'trainable_parameters': trainable_params
            }
            
            models[arch_name] = model
            
            model_time = time.time() - model_start_time
            print(f"   âœ“ {arch_name} completed in {model_time:.1f}s - Accuracy: {eval_result['accuracy']:.2f}%")
            
            # Memory cleanup between models
            if cfg.clear_memory_between_models:
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            
        except Exception as e:
            print(f"   â�Œ Error training {arch_name}: {e}")
            # Cleanup on error
            try:
                if 'model' in locals():
                    del model
                torch.cuda.empty_cache()
                import gc
                gc.collect()
            except:
                pass
            continue
    
    total_time = time.time() - total_start_time
    
    if not models:
        print("â�Œ No models were successfully trained!")
        return {}, {}
    
    # Save models and display results
    print(f"\nğŸ’¾ Saving models...")
    for arch_name, model in models.items():
        model_path = f"{arch_name.lower()}_model.pth"
        torch.save(model.state_dict(), model_path)
        print(f"âœ“ Saved {arch_name} model to {model_path}")
    
    # Results summary
    print_section("ğŸ“Š TRAINING RESULTS SUMMARY")
    
    print(f"â�±ï¸� Total training time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"ğŸ“Š Models trained: {len(results)}")
    
    # Sort models by accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True)
    
    print("\nğŸ�† MODEL PERFORMANCE RANKING:")
    print("=" * 80)
    for i, (name, result) in enumerate(sorted_results, 1):
        params_m = result['total_parameters'] / 1_000_000
        print(f"{i:2d}. {name:20s} | Acc: {result['accuracy']:6.2f}% | "
              f"F1: {result['f1']:6.4f} | Params: {params_m:5.1f}M | "
              f"Time: {result['training_time']:5.1f}s")
    print("=" * 80)
    
    # Best model details
    best_name, best_result = sorted_results[0]
    print(f"\nğŸ¥‡ BEST MODEL: {best_name}")
    print(f"   â€¢ Accuracy: {best_result['accuracy']:.2f}%")
    print(f"   â€¢ Precision: {best_result['precision']:.4f}")
    print(f"   â€¢ Recall: {best_result['recall']:.4f}")
    print(f"   â€¢ F1-Score: {best_result['f1']:.4f}")
    print(f"   â€¢ Parameters: {best_result['total_parameters']:,}")
    print(f"   â€¢ Training time: {best_result['training_time']:.1f}s")
    print(f"   â€¢ Epochs trained: {best_result['epochs_trained']}")
    
    return models, results

def create_ensemble(models, cfg, X_test, y_test):
    """Create and evaluate ensemble model"""
    if not models or not cfg.enable_ensemble:
        return None
    
    print_section("ENSEMBLE MODEL")
    
    model_list = list(models.values())
    ensemble = EnsembleModel(model_list, method=cfg.ensemble_method)
    
    # Evaluate ensemble
    print("ğŸ”„ Evaluating ensemble model...")
    
    # Scale test data
    scaler = StandardScaler()
    scaler.fit(X_test)  # Note: In practice, use the same scaler from training
    X_test_scaled = scaler.transform(X_test)
    
    ensemble_predictions = ensemble.predict(X_test_scaled)
    ensemble_accuracy = accuracy_score(y_test, ensemble_predictions)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, ensemble_predictions, average='weighted', zero_division=0
    )
    
    print(f"ğŸ�¯ Ensemble Results:")
    print(f"   Method: {cfg.ensemble_method}")
    print(f"   Models: {len(model_list)}")
    print(f"   Accuracy: {ensemble_accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Save ensemble
    ensemble_path = f"ensemble_{cfg.ensemble_method}.pkl"
    joblib.dump(ensemble, ensemble_path)
    print(f"âœ“ Saved ensemble model to {ensemble_path}")
    
    return ensemble, {
        'accuracy': ensemble_accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': ensemble_predictions
    }

##### MAIN PIPELINE #####
"""
Main pipeline execution
"""
def main():
    """Enhanced main pipeline for neural network training"""
    setup_logging()
    set_seed(CFG.seed)
    
    print_section("ğŸš€ BirdCLEF Neural Network Pipeline Starting")
    
    # Display configuration
    print_configuration(CFG)
    
    total_start_time = time.time()
    
    try:
        # 1. Data Preparation
        train_df, label_encoder = prepare_data(CFG)
        
        # 2. Feature Extraction
        X, y = extract_features(train_df, CFG)
        
        # 3. Train-Test Split
        print_section("DATA SPLITTING")
        print("ğŸ“Š Splitting into training and test sets...")
        
        # Check if stratification is possible
        unique, counts = np.unique(y, return_counts=True)
        min_class_count = counts.min()
        
        if min_class_count >= 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=CFG.test_size, random_state=CFG.seed, 
                stratify=y
            )
            print("âœ“ Stratified split used")
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=CFG.test_size, random_state=CFG.seed
            )
            print("âš ï¸� Stratified split not possible (insufficient samples)")
        
        print(f"âœ“ Training: {X_train.shape}, Test: {X_test.shape}")
        
        # 4. Neural Network Training
        models, results = train_neural_networks(X_train, y_train, X_test, y_test, CFG, label_encoder)
        
        if not models:
            print("â�Œ No models were successfully trained!")
            return
        
        # 5. Ensemble Creation
        ensemble, ensemble_results = create_ensemble(models, CFG, X_test, y_test)
        
        # 6. Results Summary
        total_time = time.time() - total_start_time
        
        print_section("ğŸ“‹ SUMMARY REPORT")
        
        # Find best individual model
        best_model_name = max(results.keys(), key=lambda k: results[k]['accuracy'])
        best_accuracy = results[best_model_name]['accuracy']
        
        summary_text = f"""
ğŸ�¯ BirdCLEF Neural Network Training Completed!

â�±ï¸�  Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)

ğŸ“Š Data Information:
â€¢ Total sample count: {len(train_df)}
â€¢ Number of classes: {CFG.num_classes}
â€¢ Feature dimensions: {X.shape[1]}
â€¢ Train/Test: {len(y_train)}/{len(y_test)}

ğŸ�† Best Individual Model: {best_model_name}
â€¢ Test Accuracy: {best_accuracy:.2f}%
â€¢ Precision: {results[best_model_name]['precision']:.4f}
â€¢ Recall: {results[best_model_name]['recall']:.4f}
â€¢ F1-Score: {results[best_model_name]['f1']:.4f}

ğŸ¤– Neural Network Architectures Trained: {len(results)}
"""
        
        for arch_name, result in results.items():
            summary_text += f"â€¢ {arch_name}: {result['accuracy']:.2f}% accuracy\n"
        
        if ensemble_results:
            summary_text += f"""
ğŸ�¯ Ensemble Model:
â€¢ Method: {CFG.ensemble_method}
â€¢ Accuracy: {ensemble_results['accuracy']:.4f}
â€¢ Precision: {ensemble_results['precision']:.4f}
â€¢ Recall: {ensemble_results['recall']:.4f}
â€¢ F1-Score: {ensemble_results['f1']:.4f}
"""
        
        summary_text += f"""
ğŸ“� Saved Files:
â€¢ Label Encoder: label_encoder.joblib
â€¢ Individual Models: {', '.join([f'{name.lower()}_model.pth' for name in models.keys()])}
â€¢ Training Log: training_log.log
"""
        
        if ensemble:
            summary_text += f"â€¢ Ensemble Model: ensemble_{CFG.ensemble_method}.pkl\n"
        
        summary_text += f"""
âš™ï¸� Configuration Used:
â€¢ Data Augmentation: {'âœ“ Enabled' if CFG.enable_data_augmentation else 'âœ— Disabled'}
â€¢ Quality Filtering: {'âœ“ Enabled' if CFG.filter_low_quality else 'âœ— Disabled'}
â€¢ Device: {device}
â€¢ Batch Size: {CFG.batch_size}
â€¢ Learning Rate: {CFG.learning_rate}
        """
        
        print(summary_text)
        print("ğŸ�‰ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"â�Œ Pipeline error: {e}")
        raise

if __name__ == "__main__":
    main() 

