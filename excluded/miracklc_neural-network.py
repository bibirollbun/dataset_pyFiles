##### IMPORTS #####
"""
This section imports all required libraries.
CPU-based models are selected for performance optimization.
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

# Scikit-learn imports
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, learning_curve
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, 
                           precision_recall_fscore_support, make_scorer)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif, chi2

# CPU Optimized Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC

# Neural Network imports
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
    print("âœ“ TensorFlow loaded successfully")
    
    # GPU Optimization Configuration
    print("ğŸ”§ Configuring GPU optimizations...")
    
    # Check GPU availability
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to prevent allocation of all GPU memory
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            
            # Set mixed precision for faster training
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            
            print(f"âœ“ GPU Configuration Complete:")
            print(f"   â€¢ Available GPUs: {len(gpus)}")
            print(f"   â€¢ Memory growth enabled")
            print(f"   â€¢ Mixed precision enabled (float16)")
            print(f"   â€¢ GPU devices: {[gpu.name for gpu in gpus]}")
            
        except RuntimeError as e:
            print(f"âš ï¸� GPU configuration error: {e}")
    else:
        print("âš ï¸� No GPU detected, using CPU")
    
    # Enable XLA compilation for faster execution
    tf.config.optimizer.set_jit(True)
    print("âœ“ XLA compilation enabled")
    
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("âš ï¸� TensorFlow not available. Neural networks will be skipped.")

from tqdm.auto import tqdm

# Performance optimizations
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')
plt.style.use('default')  # Using default instead of seaborn for compatibility

##### CONFIGURATION #####
"""
Enhanced configuration for better model performance with all new features
"""
class CFG:
    # Seed for reproducibility
    seed = 42
    debug = False  # Changed from True to False to use all data
    
    # Data paths (Kaggle paths)
    train_datadir = '/kaggle/input/birdclef-2025/train_audio'
    train_csv = '/kaggle/input/birdclef-2025/train.csv'
    taxonomy_csv = '/kaggle/input/birdclef-2025/taxonomy.csv'
    train_soundscapes_dir = '/kaggle/input/birdclef-2025/train_soundscapes'
    test_soundscapes_dir = '/kaggle/input/birdclef-2025/test_soundscapes'
    
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
    
    # Data Augmentation Settings (easy toggle)
    enable_data_augmentation = True
    augmentation_probability = 0.3  # 30% chance to apply each augmentation
    
    # Augmentation parameters
    noise_factor = 0.02  # Background noise intensity
    volume_range = (0.7, 1.3)  # Volume scaling range
    mixup_alpha = 0.2  # Mixup parameter
    
    # Pseudo-labeling settings (easy toggle)
    enable_pseudo_labeling = False  # Change to True to enable
    pseudo_confidence_threshold = 0.8  # Minimum confidence for pseudo-labels
    pseudo_max_samples_per_class = 50  # Limit pseudo-samples per class
    
    # Quality filtering
    filter_low_quality = True  # Remove samples with rating 0.5-2.5
    min_quality_rating = 2.5  # Minimum rating to keep
    
    # Performance parameters
    n_samples = None  # Use all data (was: 1500 if debug else None)
    min_samples_for_rare_class_elimination = 10  # Higher threshold
    test_size = 0.2
    cv_folds = 3  # Keep at 3 for speed
    
    # PCA parameters
    pca_variance_threshold = 0.95
    
    # GPU and Performance Optimization Parameters
    use_multiprocessing = False  # Disabled for stability (parallel feature extraction was causing timeouts)
    n_jobs = 1  # Use single thread for feature extraction
    batch_size_neural_networks = 64  # Increased batch size for GPU
    neural_network_epochs = 50  # Reduced epochs for faster training
    early_stopping_patience = 5  # Reduced patience for faster convergence
    enable_mixed_precision = True  # Use mixed precision training
    enable_xla = True  # Enable XLA compilation
    prefetch_buffer_size = 2  # Will be set to tf.data.AUTOTUNE if TensorFlow is available
    
    # Overfitting detection parameters
    overfitting_threshold = 0.15  # Maximum acceptable gap between train and val loss
    max_overfitting_models = 3  # Maximum number of overfitting models to skip before stopping
    
    # Memory optimization
    clear_memory_between_models = True  # Clear memory between model trainings
    reduce_model_architectures = debug  # Use fewer architectures in debug mode
    
    # Disable multiprocessing in debug mode for stability (already disabled by default for stability)
    if debug:
        use_multiprocessing = False
        n_jobs = 1
        print("ğŸ”¬ Debug mode: Single-threaded processing for stability")
    
    # Enhanced model configuration with additional models and better hyperparameters
    models_to_train = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=2000, solver='lbfgs', random_state=seed, n_jobs=-1),
            'param_grid': {
                'classifier__C': [0.01, 0.1, 1.0, 10.0],
                'classifier__solver': ['lbfgs', 'liblinear']
            }
        },
        'RandomForest': {
            'model': RandomForestClassifier(random_state=seed, n_jobs=-1),
            'param_grid': {
                'classifier__n_estimators': [100, 200, 300],
                'classifier__max_depth': [10, 20, None],
                'classifier__min_samples_split': [2, 5],
                'classifier__min_samples_leaf': [1, 2]
            }
        },
        'DecisionTree': {
            'model': DecisionTreeClassifier(random_state=seed),
            'param_grid': {
                'classifier__max_depth': [10, 20, 30, None],
                'classifier__min_samples_split': [2, 5, 10],
                'classifier__min_samples_leaf': [1, 2, 4],
                'classifier__criterion': ['gini', 'entropy']
            }
        },
        'KNeighbors': {
            'model': KNeighborsClassifier(n_jobs=-1),
            'param_grid': {
                'classifier__n_neighbors': [3, 5, 7, 9],
                'classifier__weights': ['uniform', 'distance'],
                'classifier__metric': ['euclidean', 'manhattan']
            }
        },
        'GaussianNB': {
            'model': GaussianNB(),
            'param_grid': {}
        }
    }

##### UTILITY FUNCTIONS #####
"""
Helper functions and setup
"""
def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
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
    print("ğŸš€ BirdCLEF Model Training Pipeline Started")
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
âš™ï¸� Configuration Used:
â€¢ Data Augmentation: {'âœ“ Enabled' if cfg.enable_data_augmentation else 'âœ— Disabled'}
â€¢ Pseudo-labeling: {'âœ“ Enabled' if cfg.enable_pseudo_labeling else 'âœ— Disabled'}
â€¢ Quality Filtering: {'âœ“ Enabled' if cfg.filter_low_quality else 'âœ— Disabled'}

ğŸ“Š Data Parameters:
â€¢ Sample Rate: {cfg.FS} Hz
â€¢ Target Duration: {cfg.TARGET_DURATION}s
â€¢ N_FFT: {cfg.N_FFT}
â€¢ Hop Length: {cfg.HOP_LENGTH}
â€¢ N_Mels: {cfg.N_MELS}
â€¢ Target Shape: {cfg.TARGET_SHAPE}

ğŸ§  Model Parameters:
â€¢ PCA Variance Threshold: {cfg.pca_variance_threshold*100}%
â€¢ Test Size: {cfg.test_size*100}%
â€¢ CV Folds: {cfg.cv_folds}
â€¢ Debug Mode: {'âœ“ Enabled' if cfg.debug else 'âœ— Disabled'}
â€¢ Sample Limit: {cfg.n_samples if cfg.debug else 'None (All data)'}

âš¡ GPU & Performance Optimizations:
â€¢ Multiprocessing: {'âœ“ Enabled' if getattr(cfg, 'use_multiprocessing', False) else 'âœ— Disabled'}
â€¢ N_Jobs: {getattr(cfg, 'n_jobs', 1)}
â€¢ Neural Network Batch Size: {getattr(cfg, 'batch_size_neural_networks', 32)}
â€¢ Neural Network Epochs: {getattr(cfg, 'neural_network_epochs', 100)}
â€¢ Early Stopping Patience: {getattr(cfg, 'early_stopping_patience', 10)}
â€¢ Mixed Precision: {'âœ“ Enabled' if getattr(cfg, 'enable_mixed_precision', False) else 'âœ— Disabled'}
â€¢ XLA Compilation: {'âœ“ Enabled' if getattr(cfg, 'enable_xla', False) else 'âœ— Disabled'}
â€¢ Memory Cleanup: {'âœ“ Enabled' if getattr(cfg, 'clear_memory_between_models', False) else 'âœ— Disabled'}
â€¢ Reduced Architectures (Debug): {'âœ“ Enabled' if getattr(cfg, 'reduce_model_architectures', False) else 'âœ— Disabled'}

ğŸ¤– Models to Train: {len(cfg.models_to_train)}
â€¢ {', '.join(cfg.models_to_train.keys())}
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
            # Calculate correct feature dimension
            base_size = cfg.TARGET_SHAPE[0] * cfg.TARGET_SHAPE[1]
            additional_features = 13*4 + 6 + 2  # MFCC stats + spectral + rhythm
            return np.zeros(base_size + additional_features, dtype=np.float32)
        
        # Normalize audio
        y = librosa.util.normalize(y)
        
        # Apply data augmentation if enabled
        if apply_augmentation_flag and cfg.enable_data_augmentation:
            y = apply_augmentation(y, cfg)
        
        # 1. Mel spectrogram (primary features)
        melspec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=cfg.N_FFT, hop_length=cfg.HOP_LENGTH,
            n_mels=cfg.N_MELS, fmin=cfg.FMIN, fmax=cfg.FMAX
        )
        melspec = librosa.power_to_db(melspec, ref=np.max)
        melspec = (melspec - melspec.min()) / (melspec.max() - melspec.min() + 1e-8)
        melspec = cv2.resize(melspec, (cfg.TARGET_SHAPE[1], cfg.TARGET_SHAPE[0]), 
                           interpolation=cv2.INTER_AREA)
        
        # 2. MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_stats = np.array([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.max(mfcc, axis=1),
            np.min(mfcc, axis=1)
        ]).flatten()
        
        # 3. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
        
        # Aggregate spectral features
        spectral_features = np.array([
            np.mean(spectral_centroid), np.std(spectral_centroid),
            np.mean(spectral_rolloff), np.std(spectral_rolloff),
            np.mean(zero_crossing_rate), np.std(zero_crossing_rate)
        ])
        
        # 4. Rhythm features
        try:
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            rhythm_features = np.array([tempo, len(beats) / len(y) * sr])  # tempo and beat density
        except:
            rhythm_features = np.array([0.0, 0.0])  # fallback
        
        # Combine all features
        melspec_features = melspec.flatten()
        combined_features = np.concatenate([
            melspec_features,
            mfcc_stats,
            spectral_features,
            rhythm_features
        ])
        
        return combined_features.astype(np.float32)
        
    except Exception as e:
        print(f"âš ï¸� Audio processing error for {audio_path}: {e}")
        # Return zeros with correct dimension
        base_size = cfg.TARGET_SHAPE[0] * cfg.TARGET_SHAPE[1]
        additional_features = 13*4 + 6 + 2  # MFCC stats + spectral + rhythm
        return np.zeros(base_size + additional_features, dtype=np.float32)

def audio_to_melspec(audio_path, cfg):
    """Legacy mel spectrogram extraction for backward compatibility"""
    return extract_enhanced_audio_features(audio_path, cfg)

##### PSEUDO-LABELING FUNCTIONS #####
"""
Pseudo-labeling implementation for soundscapes
"""
def extract_soundscape_files(cfg):
    """Extract .ogg files from train_soundscapes directory"""
    if not os.path.exists(cfg.train_soundscapes_dir):
        print(f"âš ï¸� Soundscapes directory not found: {cfg.train_soundscapes_dir}")
        return []
    
    # Find all .ogg files
    ogg_files = glob.glob(os.path.join(cfg.train_soundscapes_dir, "**/*.ogg"), recursive=True)
    print(f"âœ“ Found {len(ogg_files)} soundscape files")
    return ogg_files

def generate_pseudo_labels(model, soundscape_files, cfg, label_encoder):
    """Generate pseudo-labels for soundscape files"""
    if not cfg.enable_pseudo_labeling or not soundscape_files:
        return [], []
    
    print_section("PSEUDO-LABELING")
    print(f"ğŸ�·ï¸� Generating pseudo-labels for {len(soundscape_files)} files...")
    
    pseudo_features = []
    pseudo_labels = []
    
    for file_path in tqdm(soundscape_files[:200], desc="Processing soundscapes"):  # Limit for speed
        try:
            # Extract features (without augmentation for inference)
            features = extract_enhanced_audio_features(file_path, cfg, apply_augmentation_flag=False)
            
            if np.all(features == 0):
                continue
            
            # Predict with confidence
            features_reshaped = features.reshape(1, -1)
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features_reshaped)[0]
                confidence = np.max(probabilities)
                predicted_class = np.argmax(probabilities)
            else:
                # For models without predict_proba, use decision function if available
                predicted_class = model.predict(features_reshaped)[0]
                confidence = 0.5  # Default moderate confidence
            
            # Only keep high-confidence predictions
            if confidence >= cfg.pseudo_confidence_threshold:
                pseudo_features.append(features)
                pseudo_labels.append(predicted_class)
                
        except Exception as e:
            print(f"âš ï¸� Error processing {file_path}: {e}")
            continue
    
    # Limit samples per class
    if pseudo_features:
        pseudo_features = np.array(pseudo_features)
        pseudo_labels = np.array(pseudo_labels)
        
        # Balance pseudo-samples per class
        balanced_features = []
        balanced_labels = []
        
        for class_id in np.unique(pseudo_labels):
            class_mask = pseudo_labels == class_id
            class_features = pseudo_features[class_mask]
            
            # Limit samples per class
            if len(class_features) > cfg.pseudo_max_samples_per_class:
                indices = np.random.choice(len(class_features), cfg.pseudo_max_samples_per_class, replace=False)
                class_features = class_features[indices]
            
            balanced_features.extend(class_features)
            balanced_labels.extend([class_id] * len(class_features))
        
        pseudo_features = np.array(balanced_features)
        pseudo_labels = np.array(balanced_labels)
        
        print(f"âœ“ Generated {len(pseudo_features)} pseudo-labeled samples")
        
        # Show distribution
        unique, counts = np.unique(pseudo_labels, return_counts=True)
        for class_id, count in zip(unique, counts):
            class_name = label_encoder.classes_[class_id] if class_id < len(label_encoder.classes_) else "Unknown"
            print(f"   {class_name}: {count} samples")
    
    return pseudo_features, pseudo_labels

##### DATA PREPARATION #####
"""
Veri hazÄ±rlama ve Ã¶n iÅŸleme
"""

def process_audio_chunk(chunk_data):
    """Process a chunk of audio files for parallel processing"""
    chunk_df, cfg = chunk_data
    chunk_features = []
    chunk_labels = []
    chunk_aug_count = 0
    
    for _, row in chunk_df.iterrows():
        try:
            feature_vector = extract_enhanced_audio_features(row['filepath'], cfg)
            
            if not np.all(feature_vector == 0):
                chunk_features.append(feature_vector)
                chunk_labels.append(row['target'])
                
                if cfg.enable_data_augmentation and np.random.random() < cfg.augmentation_probability:
                    chunk_aug_count += 1
                    
        except Exception as e:
            print(f"âš ï¸� Skipping {row['filepath']}: {e}")
            continue
    
    return chunk_features, chunk_labels, chunk_aug_count

def prepare_data(cfg):
    """Enhanced data preparation with metadata utilization"""
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
    n_samples = 500  # Increased for better testing
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
            'rating': rating,  # Quality rating 0-5
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
    """Enhanced feature extraction with progress tracking and augmentation - Sequential Processing Only"""
    print_section("FEATURE EXTRACTION")
    
    print("ğŸ�µ Extracting enhanced audio features...")
    
    # Display data augmentation configuration
    if cfg.enable_data_augmentation:
        print("\nğŸ”„ Data Augmentation Optimizations Applied:")
        print(f"   â€¢ Background Noise: âœ“ Enabled (factor: {cfg.noise_factor})")
        print(f"   â€¢ Volume Scaling: âœ“ Enabled (range: {cfg.volume_range})")
        print(f"   â€¢ Mixup Alpha: {cfg.mixup_alpha}")
        print(f"   â€¢ Augmentation Probability: {cfg.augmentation_probability * 100}%")
        print("   â€¢ Multi-modal Features: Mel + MFCC + Spectral + Rhythm")
    else:
        print("\nğŸš« Data Augmentation: Disabled")
    
    # Performance settings
    print(f"âš¡ Processing Method: Sequential (reliable and stable)")
    print(f"   â€¢ Processing mode: Single-threaded for maximum stability")
    print(f"   â€¢ Memory optimization: Enabled")
    
    # Detailed Feature Breakdown
    base_size = cfg.TARGET_SHAPE[0] * cfg.TARGET_SHAPE[1]
    additional_features = 13*4 + 6 + 2  # MFCC stats + spectral + rhythm
    n_features = base_size + additional_features
    
    print(f"\nğŸ�µ Multi-Modal Feature Breakdown:")
    print(f"   ğŸ“Š Mel Spectrogram (Primary Features):")
    print(f"      â†’ {base_size} features ({cfg.TARGET_SHAPE})")
    print(f"      â†’ Frequency content over time (what frequencies are present when)")
    print(f"      â†’ Captures bird call patterns, harmonics, and spectral envelope")
    print(f"   ğŸ�¼ MFCC Features (Cepstral Analysis):")
    print(f"      â†’ {13*4} features (13 coeffs Ã— 4 statistics: mean, std, max, min)")
    print(f"      â†’ Captures vocal tract shape and timbre characteristics")
    print(f"      â†’ Essential for species-specific vocal signature recognition")
    print(f"   ğŸŒŠ Spectral Features (Signal Properties):")
    print(f"      â†’ 6 features (centroid, rolloff, zero-crossing rate)")
    print(f"      â†’ Spectral Centroid: brightness/sharpness of sound")
    print(f"      â†’ Spectral Rolloff: energy distribution across frequencies")
    print(f"      â†’ Zero Crossing Rate: noisiness vs tonality")
    print(f"   ğŸ¥� Rhythm Features (Temporal Patterns):")
    print(f"      â†’ 2 features (tempo, beat density)")
    print(f"      â†’ Tempo: rhythmic speed of bird calls")
    print(f"      â†’ Beat Density: timing patterns and call repetition rate")
    
    print(f"\nğŸ“ˆ Combined Feature Power:")
    print(f"   â€¢ Total Feature Dimension: {n_features}")
    print(f"   â€¢ Mel Spectrogram: {base_size} ({base_size/n_features*100:.1f}%)")
    print(f"   â€¢ MFCC Statistics: {13*4} ({13*4/n_features*100:.1f}%)")
    print(f"   â€¢ Spectral Features: 6 ({6/n_features*100:.1f}%)")
    print(f"   â€¢ Rhythm Features: 2 ({2/n_features*100:.1f}%)")
    print(f"   â†’ Comprehensive audio signature for species identification")
    
    # For demo data, create more sophisticated random features
    if not os.path.exists(cfg.train_datadir):
        print("ğŸ�­ Creating enhanced demo features...")
        
        # Use the already calculated feature dimensions
        print(f"ğŸš€ Using optimized vectorized feature generation...")
        
        # Vectorized feature generation for speed
        n_samples = len(df)
        y = df['target'].values
        
        # Pre-allocate arrays for better memory usage
        X = np.zeros((n_samples, n_features), dtype=np.float32)
        
        # Vectorized generation using broadcasting
        species_ids = y.reshape(-1, 1)
        
        # Base mel spectrogram features
        mel_base = np.random.rand(n_samples, base_size).astype(np.float32) * 0.5
        mel_patterns = species_ids * 0.1  # Species-specific patterns
        X[:, :base_size] = mel_base + mel_patterns
        
        # MFCC features
        mfcc_base = np.random.rand(n_samples, 52).astype(np.float32) * 0.3
        mfcc_patterns = species_ids * 0.05
        X[:, base_size:base_size+52] = mfcc_base + mfcc_patterns
        
        # Spectral and rhythm features
        remaining_features = n_features - base_size - 52
        other_base = np.random.rand(n_samples, remaining_features).astype(np.float32) * 0.2
        other_patterns = species_ids * 0.02
        X[:, base_size+52:] = other_base + other_patterns
        
        print(f"âœ“ Enhanced demo feature matrix: {X.shape} (vectorized generation)")
        return X, y
    
    # Sequential feature extraction - reliable and stable
    features = []
    labels = []
    
    start_time = time.time()
    augmentation_count = 0
    
    print("ğŸ”„ Using sequential feature extraction (stable and reliable)...")
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Audio"):
        try:
            feature_vector = extract_enhanced_audio_features(row['filepath'], cfg)
            
            if not np.all(feature_vector == 0):
                features.append(feature_vector)
                labels.append(row['target'])
                
                # Count augmentations applied (rough estimate)
                if cfg.enable_data_augmentation and np.random.random() < cfg.augmentation_probability:
                    augmentation_count += 1
                    
        except Exception as e:
            print(f"âš ï¸� Skipping {row['filepath']}: {e}")
            continue
    
    X = np.array(features, dtype=np.float32)  # Use float32 for memory efficiency
    y = np.array(labels)
    
    elapsed_time = time.time() - start_time
    print(f"âœ“ Feature extraction completed: {X.shape}, {elapsed_time:.2f} seconds")
    
    # Safe division to avoid division by zero
    if len(X) > 0:
        print(f"âš¡ Processing speed: {len(X)/elapsed_time:.2f} samples/second")
    else:
        print("âš¡ Processing speed: 0.00 samples/second")
    
    if cfg.enable_data_augmentation:
        print(f"ğŸ”„ Augmentation Summary:")
        print(f"   â€¢ ~{augmentation_count} samples received augmentation")
        if len(X) > 0:
            print(f"   â€¢ Augmentation rate: ~{(augmentation_count/len(X)*100):.1f}%")
        else:
            print(f"   â€¢ Augmentation rate: 0.0%")
    
    if X.shape[0] == 0:
        raise ValueError("â�Œ No samples remaining after feature extraction!")
    
    return X, y

def apply_mixup_features(X, y, cfg):
    """Apply mixup augmentation to feature vectors"""
    if not cfg.enable_data_augmentation:
        return X, y
    
    print("ğŸ”„ Applying mixup augmentation...")
    print(f"   â€¢ Original samples: {len(X)}")
    
    mixed_X = []
    mixed_y = []
    
    # Keep original data
    mixed_X.extend(X)
    mixed_y.extend(y)
    
    # Generate mixup samples
    n_mixup = int(len(X) * 0.2)  # 20% additional mixup samples
    print(f"   â€¢ Generating {n_mixup} mixup samples (20% of original)")
    print(f"   â€¢ Mixup alpha parameter: {cfg.mixup_alpha}")
    
    for i in range(n_mixup):
        # Select two random samples
        idx1, idx2 = np.random.choice(len(X), 2, replace=False)
        
        # Mixup features
        lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
        mixed_feature = lam * X[idx1] + (1 - lam) * X[idx2]
        
        # For classification, use the label of the dominant sample
        mixed_label = y[idx1] if lam > 0.5 else y[idx2]
        
        mixed_X.append(mixed_feature)
        mixed_y.append(mixed_label)
    
    X_mixed = np.array(mixed_X)
    y_mixed = np.array(mixed_y)
    
    print(f"âœ“ Mixup applied: {X.shape} â†’ {X_mixed.shape}")
    print(f"   â€¢ Data increase: {((len(X_mixed) - len(X)) / len(X) * 100):.1f}%")
    
    return X_mixed, y_mixed

##### PCA ANALYSIS #####
"""
PCA analizi ve optimum bileÅŸen sayÄ±sÄ± belirleme
"""
def analyze_pca_components(X_train, cfg):
    """Determine optimal number of components through PCA analysis - optimized version"""
    print_section("PCA ANALYSIS")
    
    print("ğŸ”� Analyzing explained variance with PCA...")
    
    # Optimization: Use randomized SVD for faster computation on large datasets
    n_samples, n_features = X_train.shape
    use_randomized = n_features > 1000 or n_samples > 5000
    
    if use_randomized:
        print("âš¡ Using randomized SVD for faster computation on large dataset")
        # Estimate optimal n_components for randomized PCA
        max_components = min(n_samples, n_features, 500)  # Limit for speed
        pca_full = PCA(n_components=max_components, svd_solver='randomized', random_state=cfg.seed)
    else:
        print("ğŸ”„ Using full SVD for complete analysis")
        pca_full = PCA(n_components=None, random_state=cfg.seed)
    
    # Optimized fitting with progress
    print(f"   â€¢ Dataset shape: {X_train.shape}")
    print(f"   â€¢ Analysis method: {'Randomized SVD' if use_randomized else 'Full SVD'}")
    
    start_time = time.time()
    pca_full.fit(X_train)
    pca_time = time.time() - start_time
    
    print(f"âœ“ PCA analysis completed in {pca_time:.2f} seconds")
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Visualization with optimized plotting
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 
             marker='o', linestyle='--', markersize=3)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Cumulative Explained Variance')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=cfg.pca_variance_threshold, color='r', linestyle=':', 
                label=f'{cfg.pca_variance_threshold*100}% Variance Threshold')
    plt.legend()
    
    # Individual variance contribution (limit to first 20 for clarity)
    plt.subplot(2, 2, 2)
    n_components_to_show = min(20, len(pca_full.explained_variance_ratio_))
    plt.bar(range(1, n_components_to_show + 1), 
            pca_full.explained_variance_ratio_[:n_components_to_show])
    plt.xlabel('Component Number')
    plt.ylabel('Explained Variance Ratio')
    plt.title(f'Variance Contribution of First {n_components_to_show} Components')
    plt.grid(True, alpha=0.3)
    
    # Find optimal number of components
    n_components_chosen = np.argmax(cumulative_variance >= cfg.pca_variance_threshold) + 1
    
    # Ensure we don't exceed the number of available components
    n_components_chosen = min(n_components_chosen, len(cumulative_variance))
    
    # Show different thresholds
    plt.subplot(2, 2, 3)
    thresholds = [0.90, 0.95, 0.99]
    threshold_components = []
    for thresh in thresholds:
        # Handle case where threshold might not be reached
        thresh_idx = np.argmax(cumulative_variance >= thresh)
        if cumulative_variance[thresh_idx] >= thresh:
            n_comp = thresh_idx + 1
        else:
            n_comp = len(cumulative_variance)  # Use all components
        threshold_components.append(n_comp)
        plt.axhline(y=thresh, linestyle='--', alpha=0.7, 
                   label=f'{thresh*100}% â†’ {n_comp} components')
    
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-', alpha=0.7)
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Variance')
    plt.title('Different Variance Thresholds')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Summary table
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # Calculate compression ratio
    compression_ratio = (n_features - n_components_chosen) / n_features * 100
    
    summary_text = f"""PCA ANALYSIS RESULTS

Total feature count: {n_features:,}
Samples analyzed: {n_samples:,}
Analysis time: {pca_time:.2f}s

Variance Thresholds:
â€¢ 90% variance: {threshold_components[0]} components
â€¢ 95% variance: {threshold_components[1]} components  
â€¢ 99% variance: {threshold_components[2]} components

Selected: {n_components_chosen} components
({cfg.pca_variance_threshold*100}% variance)

Dimensionality reduction: 
{n_features:,} â†’ {n_components_chosen:,}
({compression_ratio:.1f}% compression)

Speed: {n_samples*n_features/pca_time/1e6:.1f}M ops/sec"""
    
    plt.text(0.1, 0.5, summary_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('pca_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("âœ“ PCA analysis completed")
    print(f"ğŸ“Š {n_components_chosen} components selected for {cfg.pca_variance_threshold*100}% variance")
    print(f"ğŸ“‰ Dimensionality reduction: {n_features:,} â†’ {n_components_chosen:,} ({compression_ratio:.1f}% compression)")
    print(f"âš¡ Processing speed: {n_samples*n_features/pca_time/1e6:.1f}M operations/second")
    
    return n_components_chosen

def apply_pca_transformation(X_train, X_test, n_components, cfg):
    """Apply PCA transformation"""
    print(f"ğŸ”„ Applying PCA transformation ({n_components} components)...")
    
    pca = PCA(n_components=n_components, random_state=cfg.seed)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    
    # Save PCA
    joblib.dump(pca, "pca_transformer.joblib")
    
    print(f"âœ“ PCA applied: {X_train.shape} â†’ {X_train_reduced.shape}")
    
    return X_train_reduced, X_test_reduced, pca

##### MODEL TRAINING & VALIDATION #####
"""
Model eÄŸitimi ve doÄŸrulama
"""
def train_and_evaluate_models(X_train, y_train, X_test, y_test, cfg, label_encoder):
    """Optimized model training with validation and comprehensive analysis for all models"""
    print_section("MODEL TRAINING AND EVALUATION")
    
    results = []
    best_model = None
    best_model_name = ""
    best_accuracy = -1
    all_model_analyses = []
    
    for model_name, model_info in cfg.models_to_train.items():
        print(f"\nğŸ¤– Training model: {model_name}")
        
        # Create pipeline
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', model_info['model'])
        ])
        
        start_time = time.time()
        
        if not model_info['param_grid']:
            # No hyperparameter tuning
            print("âš¡ Direct training (no hyperparameter optimization)")
            pipeline.fit(X_train, y_train)
            best_estimator = pipeline
            
            # Cross-validation score
            from sklearn.model_selection import cross_val_score
            cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cfg.cv_folds, 
                                      scoring='accuracy', n_jobs=-1)
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
            
        else:
            # Grid search with cross-validation
            print("ğŸ”� Starting GridSearchCV...")
            grid_search = GridSearchCV(
                pipeline, model_info['param_grid'],
                cv=cfg.cv_folds, scoring='accuracy',
                n_jobs=-1, verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_
            cv_mean = grid_search.best_score_
            cv_std = grid_search.cv_results_['std_test_score'][grid_search.best_index_]
            
            print(f"âœ“ Best parameters: {grid_search.best_params_}")
        
        # Test performance
        test_accuracy = accuracy_score(y_test, best_estimator.predict(X_test))
        training_time = time.time() - start_time
        
        # Store results
        result = {
            'model_name': model_name,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'test_accuracy': test_accuracy,
            'training_time': training_time,
            'best_estimator': best_estimator
        }
        results.append(result)
        
        print(f"ğŸ“Š {model_name} Results:")
        print(f"   CV Accuracy: {cv_mean:.4f} (Â±{cv_std:.4f})")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Training Time: {training_time:.2f} seconds")
        
        # Perform detailed analysis for each model
        print(f"\nğŸ”� Performing detailed analysis for {model_name}...")
        
        # 1. Detailed Evaluation (Confusion Matrix, Classification Report)
        try:
            accuracy, precision, recall, f1 = detailed_model_evaluation(
                best_estimator, X_test, y_test, label_encoder, model_name
            )
            
            # 2. Overfitting Analysis
            overfitting_results = analyze_overfitting(
                best_estimator, X_train, y_train, X_test, y_test, model_name
            )
            
            # 3. Feature Importance Analysis (create dummy PCA for this function)
            class DummyPCA:
                def __init__(self):
                    self.components_ = None
            
            dummy_pca = DummyPCA()
            analyze_feature_importance(best_estimator, model_name, dummy_pca)
            
            # Store comprehensive analysis results
            model_analysis = {
                'model_name': model_name,
                'model': best_estimator,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'overfitting_analysis': overfitting_results,
                'training_time': training_time
            }
            all_model_analyses.append(model_analysis)
            
        except Exception as e:
            print(f"âš ï¸� Error in detailed analysis for {model_name}: {e}")
        
        # Update best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = best_estimator
            best_model_name = model_name
        
        print(f"âœ“ Analysis completed for {model_name}")
    
    # Create comprehensive comparison visualizations
    create_comprehensive_model_comparison(all_model_analyses, label_encoder)
    
    # Results summary
    print_section("MODEL COMPARISON RESULTS")
    
    results_df = pd.DataFrame([{
        'Model': r['model_name'],
        'CV Accuracy': f"{r['cv_mean']:.4f} Â± {r['cv_std']:.4f}",
        'Test Accuracy': f"{r['test_accuracy']:.4f}",
        'Training Time (s)': f"{r['training_time']:.2f}"
    } for r in results])
    
    print(results_df.to_string(index=False))
    
    print(f"\nğŸ�† BEST MODEL: {best_model_name} (Test Accuracy: {best_accuracy:.4f})")
    
    return best_model, best_model_name, results

##### DETAILED EVALUATION #####
"""
DetaylÄ± model deÄŸerlendirmesi
"""
def detailed_model_evaluation(model, X_test, y_test, label_encoder, model_name):
    """Comprehensive model evaluation"""
    print_section(f"DETAILED EVALUATION - {model_name}")
    
    # Predictions
    y_pred = model.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    print(f"ğŸ“Š {model_name} Test Metrics:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall: {recall:.4f}")
    print(f"   F1-Score: {f1:.4f}")
    
    # Classification Report
    print("\nğŸ“‹ CLASSIFICATION REPORT:")
    print("-" * 60)
    class_report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, 
                                       zero_division=0)
    print(class_report)
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(15, 10))
    
    # Main confusion matrix
    plt.subplot(2, 2, 1)
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Show only top classes for clarity
    max_classes = min(15, len(label_encoder.classes_))
    top_classes_idx = np.argsort(np.bincount(y_test))[-max_classes:]
    
    cm_display = cm_norm[top_classes_idx][:, top_classes_idx]
    class_names_display = label_encoder.classes_[top_classes_idx]
    
    sns.heatmap(cm_display, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names_display, yticklabels=class_names_display)
    plt.title(f'Confusion Matrix - {model_name}\n(Top {max_classes} Classes)')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    
    # Per-class accuracy
    plt.subplot(2, 2, 2)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    class_counts = np.bincount(y_test)
    
    # Sort by accuracy
    sorted_idx = np.argsort(per_class_acc)[-15:]  # Top 15
    
    plt.barh(range(len(sorted_idx)), per_class_acc[sorted_idx])
    plt.yticks(range(len(sorted_idx)), label_encoder.classes_[sorted_idx])
    plt.xlabel('Class Accuracy')
    plt.title('Per-Class Accuracy (Top 15)')
    plt.grid(True, alpha=0.3)
    
    # Class distribution in test set
    plt.subplot(2, 2, 3)
    test_class_counts = pd.Series(y_test).value_counts().head(15)
    test_class_names = [label_encoder.classes_[i] for i in test_class_counts.index]
    
    plt.bar(range(len(test_class_counts)), test_class_counts.values)
    plt.xticks(range(len(test_class_counts)), test_class_names, rotation=45)
    plt.ylabel('Sample Count')
    plt.title('Class Distribution in Test Set (Top 15)')
    plt.grid(True, alpha=0.3)
    
    # Model performance summary
    plt.subplot(2, 2, 4)
    plt.axis('off')
    summary_text = f"""MODEL PERFORMANCE SUMMARY

Model: {model_name}

Overall Metrics:
â€¢ Accuracy: {accuracy:.4f}
â€¢ Precision: {precision:.4f}
â€¢ Recall: {recall:.4f}
â€¢ F1-Score: {f1:.4f}

Test Set:
â€¢ Total samples: {len(y_test)}
â€¢ Number of classes: {len(np.unique(y_test))}
â€¢ Best class accuracy: {per_class_acc.max():.4f}
â€¢ Worst class accuracy: {per_class_acc.min():.4f}"""
    
    plt.text(0.1, 0.5, summary_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='center', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(f'evaluation_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return accuracy, precision, recall, f1

##### FEATURE IMPORTANCE ANALYSIS #####
"""
Ã–zellik Ã¶nem analizi
"""
def analyze_feature_importance(model, model_name, pca, feature_names=None):
    """Feature importance analysis for supported models"""
    print_section(f"FEATURE IMPORTANCE ANALYSIS - {model_name}")
    
    try:
        # Get the actual classifier from pipeline
        if hasattr(model, 'named_steps'):
            classifier = model.named_steps['classifier']
        else:
            classifier = model
        
        importance_scores = None
        importance_type = ""
        
        # Random Forest
        if hasattr(classifier, 'feature_importances_'):
            importance_scores = classifier.feature_importances_
            importance_type = "Gini Importance"
            
        # Logistic Regression
        elif hasattr(classifier, 'coef_'):
            if len(classifier.coef_.shape) > 1:
                # Multi-class: use mean absolute coefficients
                importance_scores = np.mean(np.abs(classifier.coef_), axis=0)
            else:
                importance_scores = np.abs(classifier.coef_[0])
            importance_type = "Coefficient Magnitude"
        
        if importance_scores is not None:
            # PCA component importance
            n_components = len(importance_scores)
            component_names = [f'PC{i+1}' for i in range(n_components)]
            
            # Sort by importance
            sorted_idx = np.argsort(importance_scores)[-20:]  # Top 20
            
            plt.figure(figsize=(12, 8))
            
            plt.subplot(2, 2, 1)
            plt.barh(range(len(sorted_idx)), importance_scores[sorted_idx])
            plt.yticks(range(len(sorted_idx)), [component_names[i] for i in sorted_idx])
            plt.xlabel(f'{importance_type}')
            plt.title(f'Top 20 Most Important PCA Components\n{model_name}')
            plt.grid(True, alpha=0.3)
            
            # Cumulative importance
            plt.subplot(2, 2, 2)
            sorted_importance = np.sort(importance_scores)[::-1]
            cumulative_importance = np.cumsum(sorted_importance) / np.sum(sorted_importance)
            
            plt.plot(range(1, len(cumulative_importance) + 1), cumulative_importance)
            plt.xlabel('Number of Components')
            plt.ylabel('Cumulative Importance')
            plt.title('Cumulative Feature Importance')
            plt.grid(True, alpha=0.3)
            plt.axhline(y=0.8, color='r', linestyle='--', label='80% Threshold')
            plt.axhline(y=0.9, color='orange', linestyle='--', label='90% Threshold')
            plt.legend()
            
            # Top 10 detailed
            plt.subplot(2, 2, 3)
            top_10_idx = sorted_idx[-10:]
            plt.pie(importance_scores[top_10_idx], 
                   labels=[component_names[i] for i in top_10_idx],
                   autopct='%1.1f%%', startangle=90)
            plt.title('Distribution of Top 10 Most Important Components')
            
            # Statistics
            plt.subplot(2, 2, 4)
            plt.axis('off')
            
            # How many components for 80% and 90% importance
            comp_80 = np.argmax(cumulative_importance >= 0.8) + 1
            comp_90 = np.argmax(cumulative_importance >= 0.9) + 1
            
            stats_text = f"""FEATURE IMPORTANCE STATISTICS

Model: {model_name}
Importance Metric: {importance_type}

Component Statistics:
â€¢ Total components: {len(importance_scores)}
â€¢ For 80% importance: {comp_80} components
â€¢ For 90% importance: {comp_90} components

Most important component: {component_names[sorted_idx[-1]]}
Importance: {importance_scores[sorted_idx[-1]]:.4f}

Importance distribution:
â€¢ Maximum: {importance_scores.max():.4f}
â€¢ Average: {importance_scores.mean():.4f}
â€¢ Minimum: {importance_scores.min():.4f}"""
            
            plt.text(0.1, 0.5, stats_text, transform=plt.gca().transAxes,
                     fontsize=10, verticalalignment='center', fontfamily='monospace')
            
            plt.tight_layout()
            plt.savefig(f'feature_importance_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print(f"âœ“ Feature importance analysis completed for {model_name}")
            print(f"   Most important component: {component_names[sorted_idx[-1]]} ({importance_scores[sorted_idx[-1]]:.4f})")
            print(f"   {comp_80} components sufficient for 80% importance")
            
        else:
            print(f"âš ï¸� Feature importance analysis not supported for {model_name}")
            
    except Exception as e:
        print(f"â�Œ Feature importance analysis error: {e}")

##### OVERFITTING ANALYSIS #####
"""
AÅŸÄ±rÄ± Ã¶ÄŸrenme analizi
"""
def analyze_overfitting(model, X_train, y_train, X_test, y_test, model_name):
    """Analyze potential overfitting/underfitting"""
    print_section(f"OVERFITTING ANALYSIS - {model_name}")
    
    try:
        # Learning curves
        train_sizes, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=3,
            train_sizes=np.linspace(0.1, 1.0, 10),
            scoring='accuracy', n_jobs=-1, random_state=CFG.seed
        )
        
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Get final scores
        train_final_accuracy = accuracy_score(y_train, model.predict(X_train))
        test_final_accuracy = accuracy_score(y_test, model.predict(X_test))
        
        # Determine overfitting status
        overfitting_gap = train_final_accuracy - test_final_accuracy
        
        if overfitting_gap > 0.1:
            status = "ğŸ”´ OVERFITTING"
            status_color = 'red'
        elif overfitting_gap > 0.05:
            status = "ğŸŸ¡ MILD OVERFITTING"
            status_color = 'orange'
        elif test_final_accuracy < 0.3:
            status = "ğŸ”µ UNDERFITTING"
            status_color = 'blue'
        else:
            status = "ğŸŸ¢ BALANCED LEARNING"
            status_color = 'green'
        
        plt.figure(figsize=(15, 10))
        
        # Learning curve
        plt.subplot(2, 3, 1)
        plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
        plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1, color='red')
        plt.xlabel('Training Set Size')
        plt.ylabel('Accuracy Score')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Performance gap visualization
        plt.subplot(2, 3, 2)
        gap_values = train_mean - val_mean
        plt.plot(train_sizes, gap_values, 'o-', color='purple')
        plt.axhline(y=0.1, color='red', linestyle='--', label='Overfitting threshold')
        plt.axhline(y=0.05, color='orange', linestyle='--', label='Acceptable threshold')
        plt.xlabel('Training Set Size')
        plt.ylabel('Training - Validation Gap')
        plt.title('Performance Gap')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Final comparison
        plt.subplot(2, 3, 3)
        labels = ['Training', 'Test']
        scores = [train_final_accuracy, test_final_accuracy]
        colors = ['blue', 'red']
        
        bars = plt.bar(labels, scores, color=colors, alpha=0.7)
        plt.ylabel('Accuracy')
        plt.title('Final Performance Comparison')
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, scores):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                     f'{score:.3f}', ha='center', va='bottom')
        
        plt.grid(True, alpha=0.3)
        
        # Validation curve for key hyperparameter (if applicable)
        plt.subplot(2, 3, 4)
        try:
            if hasattr(model.named_steps['classifier'], 'C'):  # Logistic Regression
                param_name = 'classifier__C'
                param_range = [0.01, 0.1, 1, 10, 100]
            elif hasattr(model.named_steps['classifier'], 'n_estimators'):  # Random Forest
                param_name = 'classifier__n_estimators'
                param_range = [10, 50, 100, 200, 500]
            elif hasattr(model.named_steps['classifier'], 'n_neighbors'):  # KNN
                param_name = 'classifier__n_neighbors'
                param_range = [1, 3, 5, 7, 9, 11]
            else:
                param_name = None
                
            if param_name:
                train_scores_val, test_scores_val = validation_curve(
                    model, X_train, y_train, param_name=param_name,
                    param_range=param_range, cv=3, scoring='accuracy', n_jobs=-1
                )
                
                train_mean_val = np.mean(train_scores_val, axis=1)
                test_mean_val = np.mean(test_scores_val, axis=1)
                
                plt.plot(param_range, train_mean_val, 'o-', color='blue', label='Training')
                plt.plot(param_range, test_mean_val, 'o-', color='red', label='Validation')
                plt.xlabel(param_name.split('__')[1])
                plt.ylabel('Accuracy')
                plt.title('Validation Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
                if param_name == 'classifier__C':
                    plt.xscale('log')
            else:
                plt.text(0.5, 0.5, 'Validation curve\nnot available\nfor this model', 
                         ha='center', va='center', transform=plt.gca().transAxes)
                plt.axis('off')
                
        except Exception as e:
            plt.text(0.5, 0.5, f'Validation curve\nerror:\n{str(e)[:50]}...', 
                     ha='center', va='center', transform=plt.gca().transAxes)
            plt.axis('off')
        
        # Recommendations
        plt.subplot(2, 3, 5)
        plt.axis('off')
        
        # Generate recommendations
        recommendations = []
        if overfitting_gap > 0.1:
            recommendations = [
                "â€¢ Collect more training data",
                "â€¢ Increase regularization parameters",
                "â€¢ Reduce model complexity",
                "â€¢ Use dropout or early stopping",
                "â€¢ Tune parameters with cross-validation"
            ]
        elif overfitting_gap > 0.05:
            recommendations = [
                "â€¢ Slightly increase regularization",
                "â€¢ Use more cross-validation",
                "â€¢ Apply feature selection"
            ]
        elif test_final_accuracy < 0.3:
            recommendations = [
                "â€¢ Increase model complexity",
                "â€¢ Add more features",
                "â€¢ Try different model architecture",
                "â€¢ Improve data preprocessing"
            ]
        else:
            recommendations = [
                "â€¢ Model performance is balanced",
                "â€¢ Current configuration is suitable",
                "â€¢ Optional fine-tuning possible"
            ]
        
        rec_text = f"""MODEL STATUS
{status}

Performance Metrics:
â€¢ Training Accuracy: {train_final_accuracy:.4f}
â€¢ Test Accuracy: {test_final_accuracy:.4f}
â€¢ Performance Gap: {overfitting_gap:.4f}

RECOMMENDATIONS:
""" + "\n".join(recommendations)
        
        plt.text(0.05, 0.95, rec_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        # Summary metrics
        plt.subplot(2, 3, 6)
        plt.axis('off')
        
        # Calculate additional metrics
        final_train_val_gap = train_mean[-1] - val_mean[-1]
        learning_efficiency = (val_mean[-1] - val_mean[0]) / (train_sizes[-1] - train_sizes[0])
        
        metrics_text = f"""DETAILED METRICS

Learning Curve:
â€¢ Initial CV score: {val_mean[0]:.4f}
â€¢ Final CV score: {val_mean[-1]:.4f}
â€¢ Learning efficiency: {learning_efficiency:.6f}

Overfitting Signals:
â€¢ Train-Test gap: {overfitting_gap:.4f}
â€¢ Train-CV gap: {final_train_val_gap:.4f}

Model Stability:
â€¢ CV standard deviation: {val_std[-1]:.4f}
â€¢ Training standard deviation: {train_std[-1]:.4f}"""
        
        plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(f'overfitting_analysis_{model_name.lower()}.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Log results
        print(f"ğŸ“Š {model_name} Overfitting Analysis:")
        print(f"   {status}")
        print(f"   Training Accuracy: {train_final_accuracy:.4f}")
        print(f"   Test Accuracy: {test_final_accuracy:.4f}")
        print(f"   Performance Gap: {overfitting_gap:.4f}")
        
        return {
            'status': status,
            'train_accuracy': train_final_accuracy,
            'test_accuracy': test_final_accuracy,
            'overfitting_gap': overfitting_gap,
            'recommendations': recommendations
        }
        
    except Exception as e:
        print(f"â�Œ Overfitting analysis error: {e}")
        return None

##### NEURAL NETWORK TRAINING #####
"""
Neural Network training with different architectures
"""
def create_neural_network(input_dim, num_classes, architecture, dropout_rate=0.2):
    """Create neural network with specified architecture and GPU optimizations"""
    if not TENSORFLOW_AVAILABLE:
        return None
    
    model = keras.Sequential()
    
    # Input layer with optimized initialization
    model.add(layers.Dense(
        architecture[0], 
        activation='relu', 
        input_shape=(input_dim,),
        kernel_initializer='he_normal',  # Better for ReLU
        bias_initializer='zeros'
    ))
    model.add(layers.BatchNormalization())
    
    # Hidden layers with optimizations
    for i in range(1, len(architecture)):
        model.add(layers.Dense(
            architecture[i], 
            activation='relu',
            kernel_initializer='he_normal',
            bias_initializer='zeros'
        ))
        model.add(layers.BatchNormalization())
        if dropout_rate > 0:
            model.add(layers.Dropout(dropout_rate))
    
    # Output layer
    if num_classes > 2:
        model.add(layers.Dense(num_classes, activation='softmax', name='predictions'))
        loss = 'categorical_crossentropy'
    else:
        model.add(layers.Dense(1, activation='sigmoid', name='predictions'))
        loss = 'binary_crossentropy'
    
    # Mixed precision optimization for output layer
    if tf.config.list_physical_devices('GPU') and hasattr(tf.keras.mixed_precision, 'Policy'):
        # Cast to float32 for numerical stability in mixed precision
        model.add(layers.Activation('linear', dtype='float32'))
    
    # Compile model with optimized settings
    optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,  # Better for mixed precision
        amsgrad=False
    )
    
    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy'],
        # Enable XLA compilation if available
        jit_compile=True if tf.config.list_physical_devices('GPU') else False
    )
    
    return model, loss

def train_neural_networks(X_train, y_train, X_test, y_test, num_classes, cfg):
    """Train different neural network architectures with GPU optimizations and overfitting detection"""
    if not TENSORFLOW_AVAILABLE:
        print("âš ï¸� TensorFlow not available. Skipping neural network training.")
        return []
    
    print_section("NEURAL NETWORK TRAINING")
    
    # Define different architectures to test - with optimization for debug mode
    if hasattr(cfg, 'reduce_model_architectures') and cfg.reduce_model_architectures:
        print("ğŸ”¬ Debug mode: Using reduced architecture set for faster testing")
        architectures = [
            [128, 64],
            [256, 128, 64],
            [128, 128, 64]
        ]
    else:
        architectures = [
            [128, 64],
            [256, 128, 64],
            [128, 128, 64],
            [512, 256, 128, 64],
            [64, 32],
            [256, 128],
            [128, 64, 32],
            [512, 256, 128],
            [1024, 512, 256, 128],
            [256, 256, 128, 64]
        ]
    
    # GPU Optimization settings
    batch_size = getattr(cfg, 'batch_size_neural_networks', 64)
    epochs = getattr(cfg, 'neural_network_epochs', 50)
    patience = getattr(cfg, 'early_stopping_patience', 5)
    
    # Overfitting detection settings
    overfitting_threshold = getattr(cfg, 'overfitting_threshold', 0.15)
    max_overfitting_models = getattr(cfg, 'max_overfitting_models', 3)
    
    print(f"âš¡ GPU Optimization Settings:")
    print(f"   â€¢ Batch size: {batch_size} (optimized for GPU)")
    print(f"   â€¢ Max epochs: {epochs}")
    print(f"   â€¢ Early stopping patience: {patience}")
    print(f"   â€¢ Mixed precision: {getattr(cfg, 'enable_mixed_precision', True)}")
    print(f"   â€¢ XLA compilation: {getattr(cfg, 'enable_xla', True)}")
    
    print(f"ğŸ”� Overfitting Detection Settings:")
    print(f"   â€¢ Overfitting threshold: {overfitting_threshold} (max acceptable train-val loss gap)")
    print(f"   â€¢ Max overfitting models to skip: {max_overfitting_models}")
    print(f"   â€¢ Strategy: Skip overfitting models and continue until balanced model found")
    
    # Prepare data for neural networks
    if num_classes > 2:
        y_train_cat = to_categorical(y_train, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)
    else:
        y_train_cat = y_train
        y_test_cat = y_test
    
    # Scale features for neural networks - using float32 for GPU efficiency
    print("ğŸ”„ Scaling features with float32 precision for GPU efficiency...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)
    
    # Create TensorFlow datasets for better GPU utilization
    print("ğŸ“Š Creating optimized TensorFlow datasets...")
    
    def create_tf_dataset(X, y, batch_size, is_training=True):
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
        if is_training:
            dataset = dataset.shuffle(buffer_size=1000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset
    
    train_dataset = create_tf_dataset(X_train_scaled, y_train_cat, batch_size, is_training=True)
    test_dataset = create_tf_dataset(X_test_scaled, y_test_cat, batch_size, is_training=False)
    
    results = []
    overfitting_count = 0
    successful_models = 0
    
    print(f"ğŸ§  Training {len(architectures)} different neural network architectures...")
    print(f"ğŸ“Š Input dimension: {X_train.shape[1]}, Output classes: {num_classes}")
    print("=" * 80)
    
    for i, arch in enumerate(architectures):
        print(f"\nğŸ”„ Training Architecture {i+1}/{len(architectures)}: {'/'.join(map(str, arch))}")
        
        try:
            # Create model with optimizations
            model, loss_type = create_neural_network(X_train.shape[1], num_classes, arch)
            
            if model is None:
                continue
            
            # Enhanced callbacks for GPU optimization
            callbacks = []
            
            # Early stopping
            early_stopping = keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=patience,
                restore_best_weights=True,
                verbose=0
            )
            callbacks.append(early_stopping)
            
            # Learning rate reduction on plateau
            lr_reducer = keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=max(2, patience//2),
                min_lr=1e-7,
                verbose=0
            )
            callbacks.append(lr_reducer)
            
            # Mixed precision loss scaling (if enabled)
            if getattr(cfg, 'enable_mixed_precision', True) and tf.config.list_physical_devices('GPU'):
                # Model already uses mixed precision policy from global setting
                pass
            
            # Train model with optimized settings
            start_time = time.time()
            
            with tf.device('/GPU:0' if tf.config.list_physical_devices('GPU') else '/CPU:0'):
                history = model.fit(
                    train_dataset,
                    validation_data=test_dataset,
                    epochs=epochs,
                    callbacks=callbacks,
                    verbose=0,
                    # Additional GPU optimizations
                    steps_per_epoch=len(X_train) // batch_size,
                    validation_steps=len(X_test) // batch_size
                )
            
            training_time = time.time() - start_time
            
            # Get final training and validation losses for overfitting detection
            final_train_loss = history.history['loss'][-1]
            final_val_loss = history.history['val_loss'][-1]
            loss_gap = final_val_loss - final_train_loss
            
            # Overfitting detection with detailed explanation
            is_overfitting = loss_gap > overfitting_threshold
            
            if is_overfitting:
                overfitting_count += 1
                print(f"   ğŸš¨ OVERFITTING DETECTED!")
                print(f"   ğŸ“Š Train Loss: {final_train_loss:.4f}")
                print(f"   ğŸ“Š Validation Loss: {final_val_loss:.4f}")
                print(f"   ğŸ“Š Loss Gap: {loss_gap:.4f} (threshold: {overfitting_threshold})")
                print(f"   ğŸ’¡ Since there is a massive gap between validation loss ({final_val_loss:.4f}) and train loss ({final_train_loss:.4f}), this indicates that the model is overfitting. Skipping this model.")
                
                # Check if we should stop trying more models
                if overfitting_count >= max_overfitting_models:
                    print(f"   â›” Reached maximum overfitting models limit ({max_overfitting_models}). Continuing to search for balanced model...")
                
                # Clean up and skip this model
                del model
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
                continue
            
            # Model passed overfitting check - evaluate it
            print(f"   âœ… Model passed overfitting check (gap: {loss_gap:.4f} < {overfitting_threshold})")
            
            # Evaluate model
            test_loss, test_accuracy = model.evaluate(test_dataset, verbose=0)
            train_loss, train_accuracy = model.evaluate(train_dataset, verbose=0)
            
            # Get predictions for additional metrics
            print(f"   ğŸ“Š Computing detailed metrics...")
            y_pred_probs = model.predict(test_dataset, verbose=0)
            
            if num_classes > 2:
                y_pred = np.argmax(y_pred_probs, axis=1)
                y_true = np.argmax(y_test_cat, axis=1)
            else:
                y_pred = (y_pred_probs > 0.5).astype(int).flatten()
                y_true = y_test_cat
            
            # Calculate additional metrics
            precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
            
            # Store results
            arch_str = '/'.join(map(str, arch + [num_classes]))
            results.append({
                'Architecture': arch_str,
                'Parameters': model.count_params(),
                'Train_Accuracy': train_accuracy,
                'Test_Accuracy': test_accuracy,
                'Train_Loss': train_loss,
                'Test_Loss': test_loss,
                'Loss_Gap': loss_gap,
                'Precision': precision,
                'Recall': recall,
                'F1_Score': f1,
                'Training_Time': training_time,
                'Epochs_Trained': len(history.history['loss']),
                'Overfitting_Gap': train_accuracy - test_accuracy,
                'Model': model,
                'History': history
            })
            
            successful_models += 1
            print(f"   âœ“ Acc: {test_accuracy:.4f}, Loss: {test_loss:.4f}, Time: {training_time:.1f}s, Epochs: {len(history.history['loss'])}")
            
            # Memory cleanup between models
            if getattr(cfg, 'clear_memory_between_models', True):
                del model
                tf.keras.backend.clear_session()
                # Force garbage collection
                import gc
                gc.collect()
            
        except Exception as e:
            print(f"   â�Œ Error training architecture {arch}: {e}")
            # Cleanup on error
            try:
                tf.keras.backend.clear_session()
                import gc
                gc.collect()
            except:
                pass
            continue
    
    # Summary of training process
    print(f"\nğŸ“Š Training Summary:")
    print(f"   â€¢ Total architectures attempted: {len(architectures)}")
    print(f"   â€¢ Successful models: {successful_models}")
    print(f"   â€¢ Overfitting models skipped: {overfitting_count}")
    print(f"   â€¢ Success rate: {(successful_models/len(architectures)*100):.1f}%")
    
    if not results:
        print("â�Œ No neural networks were successfully trained!")
        if overfitting_count > 0:
            print(f"ğŸ’¡ All {overfitting_count} models showed overfitting. Consider:")
            print("   â€¢ Reducing model complexity")
            print("   â€¢ Adding more regularization")
            print("   â€¢ Collecting more training data")
            print("   â€¢ Increasing dropout rate")
        return []
    
    # Display results table
    print_section("NEURAL NETWORK RESULTS")
    
    # Performance statistics
    total_training_time = sum(r['Training_Time'] for r in results)
    avg_time_per_model = total_training_time / len(results)
    
    print(f"âš¡ Performance Summary:")
    print(f"   â€¢ Total training time: {total_training_time:.1f}s ({total_training_time/60:.1f}m)")
    print(f"   â€¢ Average time per model: {avg_time_per_model:.1f}s")
    print(f"   â€¢ Models trained: {len(results)}")
    
    # Create results DataFrame with loss gap information
    results_df = pd.DataFrame([{
        'Architecture': r['Architecture'],
        'Parameters': f"{r['Parameters']:,}",
        'Test_Accuracy': f"{r['Test_Accuracy']:.4f}",
        'Test_Loss': f"{r['Test_Loss']:.4f}",
        'Loss_Gap': f"{r['Loss_Gap']:.4f}",
        'Precision': f"{r['Precision']:.4f}",
        'F1_Score': f"{r['F1_Score']:.4f}",
        'Overfitting': f"{r['Overfitting_Gap']:.4f}",
        'Time(s)': f"{r['Training_Time']:.1f}",
        'Epochs': r['Epochs_Trained']
    } for r in results])
    
    # Sort by test accuracy
    results_df = results_df.sort_values('Test_Accuracy', ascending=False)
    
    print("ğŸ�† NEURAL NETWORK PERFORMANCE TABLE")
    print("=" * 130)
    print(results_df.to_string(index=False))
    print("=" * 130)
    
    # Find best model
    best_result = max(results, key=lambda x: x['Test_Accuracy'])
    print(f"\nğŸ¥‡ BEST NEURAL NETWORK:")
    print(f"   Architecture: {best_result['Architecture']}")
    print(f"   Test Accuracy: {best_result['Test_Accuracy']:.4f}")
    print(f"   Test Loss: {best_result['Test_Loss']:.4f}")
    print(f"   Loss Gap: {best_result['Loss_Gap']:.4f}")
    print(f"   Parameters: {best_result['Parameters']:,}")
    print(f"   Training Time: {best_result['Training_Time']:.1f} seconds")
    print(f"   Epochs Trained: {best_result['Epochs_Trained']}")
    
    # Visualize results
    visualize_neural_network_results(results)
    
    return results

def visualize_neural_network_results(results):
    """Visualize neural network training results"""
    if not results:
        return
    
    plt.figure(figsize=(20, 15))
    
    # 1. Accuracy comparison
    plt.subplot(3, 4, 1)
    architectures = [r['Architecture'] for r in results]
    test_accuracies = [r['Test_Accuracy'] for r in results]
    train_accuracies = [r['Train_Accuracy'] for r in results]
    
    x = range(len(architectures))
    width = 0.35
    
    plt.bar([i - width/2 for i in x], train_accuracies, width, label='Train', alpha=0.8)
    plt.bar([i + width/2 for i in x], test_accuracies, width, label='Test', alpha=0.8)
    plt.xlabel('Architecture')
    plt.ylabel('Accuracy')
    plt.title('Train vs Test Accuracy')
    plt.xticks(x, [arch.replace('/', '\n') for arch in architectures], rotation=45, fontsize=8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Loss comparison
    plt.subplot(3, 4, 2)
    test_losses = [r['Test_Loss'] for r in results]
    train_losses = [r['Train_Loss'] for r in results]
    
    plt.bar([i - width/2 for i in x], train_losses, width, label='Train', alpha=0.8)
    plt.bar([i + width/2 for i in x], test_losses, width, label='Test', alpha=0.8)
    plt.xlabel('Architecture')
    plt.ylabel('Loss')
    plt.title('Train vs Test Loss')
    plt.xticks(x, [arch.replace('/', '\n') for arch in architectures], rotation=45, fontsize=8)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. Parameter count vs accuracy
    plt.subplot(3, 4, 3)
    param_counts = [r['Parameters'] for r in results]
    plt.scatter(param_counts, test_accuracies, alpha=0.7, s=100)
    plt.xlabel('Number of Parameters')
    plt.ylabel('Test Accuracy')
    plt.title('Parameters vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Add labels for each point
    for i, arch in enumerate(architectures):
        plt.annotate(arch.replace('/', '\n'), (param_counts[i], test_accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=6)
    
    # 4. Training time vs accuracy
    plt.subplot(3, 4, 4)
    training_times = [r['Training_Time'] for r in results]
    plt.scatter(training_times, test_accuracies, alpha=0.7, s=100)
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Test Accuracy')
    plt.title('Training Time vs Accuracy')
    plt.grid(True, alpha=0.3)
    
    # 5. Overfitting analysis
    plt.subplot(3, 4, 5)
    overfitting_gaps = [r['Overfitting_Gap'] for r in results]
    colors = ['red' if gap > 0.1 else 'orange' if gap > 0.05 else 'green' for gap in overfitting_gaps]
    
    plt.bar(x, overfitting_gaps, color=colors, alpha=0.7)
    plt.xlabel('Architecture')
    plt.ylabel('Overfitting Gap')
    plt.title('Overfitting Analysis')
    plt.xticks(x, [arch.replace('/', '\n') for arch in architectures], rotation=45, fontsize=8)
    plt.axhline(y=0.1, color='red', linestyle='--', label='High overfitting')
    plt.axhline(y=0.05, color='orange', linestyle='--', label='Mild overfitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 6. F1 Score comparison
    plt.subplot(3, 4, 6)
    f1_scores = [r['F1_Score'] for r in results]
    plt.bar(x, f1_scores, alpha=0.8, color='purple')
    plt.xlabel('Architecture')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison')
    plt.xticks(x, [arch.replace('/', '\n') for arch in architectures], rotation=45, fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 7. Training curves for best model
    plt.subplot(3, 4, 7)
    best_result = max(results, key=lambda x: x['Test_Accuracy'])
    history = best_result['History']
    
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Best Model Training Curve\n{best_result["Architecture"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 8. Loss curves for best model
    plt.subplot(3, 4, 8)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Best Model Loss Curve\n{best_result["Architecture"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 9. Epochs trained
    plt.subplot(3, 4, 9)
    epochs_trained = [r['Epochs_Trained'] for r in results]
    plt.bar(x, epochs_trained, alpha=0.8, color='teal')
    plt.xlabel('Architecture')
    plt.ylabel('Epochs Trained')
    plt.title('Training Epochs (Early Stopping)')
    plt.xticks(x, [arch.replace('/', '\n') for arch in architectures], rotation=45, fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # 10. Precision vs Recall
    plt.subplot(3, 4, 10)
    precisions = [r['Precision'] for r in results]
    recalls = [r['Recall'] for r in results]
    plt.scatter(recalls, precisions, alpha=0.7, s=100)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision vs Recall')
    plt.grid(True, alpha=0.3)
    
    # Add labels
    for i, arch in enumerate(architectures):
        plt.annotate(arch, (recalls[i], precisions[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=6)
    
    # 11. Performance summary table
    plt.subplot(3, 4, 11)
    plt.axis('off')
    
    # Sort results by accuracy
    sorted_results = sorted(results, key=lambda x: x['Test_Accuracy'], reverse=True)
    
    summary_text = "TOP 5 MODELS:\n\n"
    for i, r in enumerate(sorted_results[:5]):
        summary_text += f"{i+1}. {r['Architecture']}\n"
        summary_text += f"   Acc: {r['Test_Accuracy']:.4f}\n"
        summary_text += f"   Loss: {r['Test_Loss']:.4f}\n"
        summary_text += f"   Params: {r['Parameters']:,}\n\n"
    
    plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    # 12. Performance statistics
    plt.subplot(3, 4, 12)
    plt.axis('off')
    
    stats_text = f"""NEURAL NETWORK STATISTICS

Total models trained: {len(results)}

Accuracy Statistics:
â€¢ Best: {max(test_accuracies):.4f}
â€¢ Worst: {min(test_accuracies):.4f}
â€¢ Average: {np.mean(test_accuracies):.4f}
â€¢ Std: {np.std(test_accuracies):.4f}

Parameter Statistics:
â€¢ Largest: {max(param_counts):,}
â€¢ Smallest: {min(param_counts):,}
â€¢ Average: {int(np.mean(param_counts)):,}

Training Time:
â€¢ Total: {sum(training_times):.1f}s
â€¢ Average: {np.mean(training_times):.1f}s"""
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('neural_network_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("âœ“ Neural network results visualization saved as 'neural_network_results.png'")

##### MAIN PIPELINE #####
"""
Ana Ã§alÄ±ÅŸma pipeline'Ä±
"""
def main():
    """Enhanced main pipeline with all new features"""
    setup_logging()
    set_seed(CFG.seed)
    
    print_section("ğŸš€ BirdCLEF ML Pipeline Starting")
    
    # Display configuration
    print_configuration(CFG)
    
    total_start_time = time.time()
    
    try:
        # 1. Data Preparation
        train_df, label_encoder = prepare_data(CFG)
        
        # 2. Feature Extraction
        X, y = extract_features(train_df, CFG)
        
        # 3. Apply Mixup Augmentation
        if CFG.enable_data_augmentation:
            X, y = apply_mixup_features(X, y, CFG)
        
        # 4. Train-Test Split
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
        
        # 5. PCA Analysis
        optimal_n_components = analyze_pca_components(X_train, CFG)
        X_train_reduced, X_test_reduced, pca = apply_pca_transformation(
            X_train, X_test, optimal_n_components, CFG
        )
        
        # 6. Model Training (now includes comprehensive analysis for all models)
        best_model, best_model_name, all_results = train_and_evaluate_models(
            X_train_reduced, y_train, X_test_reduced, y_test, CFG, label_encoder
        )
        
        if best_model is None:
            print("â�Œ No model was successfully trained!")
            return
        
        # 7. Detailed Evaluation
        accuracy, precision, recall, f1 = detailed_model_evaluation(
            best_model, X_test_reduced, y_test, label_encoder, best_model_name
        )
        
        # 8. Pseudo-labeling (if enabled)
        if CFG.enable_pseudo_labeling:
            soundscape_files = extract_soundscape_files(CFG)
            if soundscape_files:
                pseudo_X, pseudo_y = generate_pseudo_labels(best_model, soundscape_files, CFG, label_encoder)
                
                if len(pseudo_X) > 0:
                    # Apply PCA to pseudo-features
                    pseudo_X_reduced = pca.transform(pseudo_X)
                    
                    # Combine with original training data
                    X_train_enhanced = np.vstack([X_train_reduced, pseudo_X_reduced])
                    y_train_enhanced = np.concatenate([y_train, pseudo_y])
                    
                    print("ğŸ”„ Retraining best model with pseudo-labels...")
                    
                    # Retrain the best model with enhanced data
                    best_model.fit(X_train_enhanced, y_train_enhanced)
                    
                    # Evaluate enhanced model
                    enhanced_accuracy = accuracy_score(y_test, best_model.predict(X_test_reduced))
                    print(f"ğŸ“Š Enhanced model accuracy: {enhanced_accuracy:.4f}")
                    accuracy = enhanced_accuracy  # Update final accuracy
        
        # 9. Feature Importance Analysis
        analyze_feature_importance(best_model, best_model_name, pca)
        
        # 10. Overfitting Analysis
        overfitting_results = analyze_overfitting(
            best_model, X_train_reduced, y_train, X_test_reduced, y_test, best_model_name
        )
        
        # 11. Neural Network Training
        neural_network_results = train_neural_networks(X_train_reduced, y_train, X_test_reduced, y_test, CFG.num_classes, CFG)
        
        # 12. Save Best Model
        print_section("MODEL SAVING")
        
        # Get final accuracy for filename
        final_accuracy = accuracy_score(y_test, best_model.predict(X_test_reduced))
        model_filename = f"best_model_{best_model_name.lower()}_acc_{final_accuracy:.3f}.joblib"
        joblib.dump(best_model, model_filename)
        print(f"âœ“ Best model saved: {model_filename}")
        
        # Save best neural network if available
        if neural_network_results:
            best_nn_result = max(neural_network_results, key=lambda x: x['Test_Accuracy'])
            nn_filename = f"best_neural_network_{best_nn_result['Architecture'].replace('/', '_')}_acc_{best_nn_result['Test_Accuracy']:.3f}.h5"
            best_nn_result['Model'].save(nn_filename)
            print(f"âœ“ Best neural network saved: {nn_filename}")
        
        # 13. Final Summary
        total_time = time.time() - total_start_time
        
        print_section("ğŸ“‹ SUMMARY REPORT")
        
        # Neural network summary
        nn_summary = ""
        if neural_network_results:
            best_nn = max(neural_network_results, key=lambda x: x['Test_Accuracy'])
            nn_summary = f"""
ğŸ§  Neural Network Results:
â€¢ Best Architecture: {best_nn['Architecture']}
â€¢ Best NN Accuracy: {best_nn['Test_Accuracy']:.4f}
â€¢ Best NN Loss: {best_nn['Test_Loss']:.4f}
â€¢ Total NN Models Trained: {len(neural_network_results)}
"""
        else:
            nn_summary = "\nğŸ§  Neural Networks: Not available (TensorFlow not installed)"
        
        summary_text = f"""
ğŸ�¯ BirdCLEF Model Training Completed!

â�±ï¸�  Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)

ğŸ“Š Data Information:
â€¢ Total sample count: {len(train_df)}
â€¢ Number of classes: {CFG.num_classes}
â€¢ Feature dimensions: {X.shape[1]} â†’ {optimal_n_components} (PCA)
â€¢ Train/Test: {len(y_train)}/{len(y_test)}

ğŸ�† Best Traditional Model: {best_model_name}
â€¢ Test Accuracy: {final_accuracy:.4f}
{nn_summary}
ğŸ“� Saved Files:
â€¢ Model: {model_filename}
â€¢ Label Encoder: label_encoder.joblib
â€¢ PCA Transformer: pca_transformer.joblib
â€¢ Training Log: training_log.log

ğŸ“ˆ Visualizations Generated:
â€¢ PCA Analysis: pca_analysis.png
â€¢ Comprehensive Model Comparison: comprehensive_model_comparison.png
â€¢ Individual Model Evaluations: evaluation_[model_name].png (for each model)
â€¢ Individual Feature Importance: feature_importance_[model_name].png (for each model)
â€¢ Individual Overfitting Analysis: overfitting_analysis_[model_name].png (for each model)
â€¢ Neural Network Results: neural_network_results.png

ğŸ”� Comprehensive Analysis Completed:
â€¢ Confusion Matrix: âœ“ Generated for ALL {len(CFG.models_to_train)} models
â€¢ Classification Report: âœ“ Generated for ALL {len(CFG.models_to_train)} models  
â€¢ Overfitting Analysis: âœ“ Generated for ALL {len(CFG.models_to_train)} models
â€¢ Feature Importance: âœ“ Generated for ALL {len(CFG.models_to_train)} models
â€¢ Model Comparison: âœ“ Comprehensive comparison across all models

âš™ï¸� Configuration Used:
â€¢ Data Augmentation: {'âœ“ Enabled' if CFG.enable_data_augmentation else 'âœ— Disabled'}
â€¢ Pseudo-labeling: {'âœ“ Enabled' if CFG.enable_pseudo_labeling else 'âœ— Disabled'}
â€¢ Quality Filtering: {'âœ“ Enabled' if CFG.filter_low_quality else 'âœ— Disabled'}
        """
        
        print(summary_text)
        print("ğŸ�‰ Pipeline completed successfully!")
        
    except Exception as e:
        print(f"â�Œ Pipeline error: {e}")
        raise

##### COMPREHENSIVE MODEL COMPARISON #####
"""
Comprehensive comparison of all models
"""
def create_comprehensive_model_comparison(all_model_analyses, label_encoder):
    """Create comprehensive comparison visualizations for all models"""
    if not all_model_analyses:
        print("âš ï¸� No model analyses available for comparison")
        return
    
    print_section("COMPREHENSIVE MODEL COMPARISON")
    
    # Extract data for comparison
    model_names = [analysis['model_name'] for analysis in all_model_analyses]
    accuracies = [analysis['accuracy'] for analysis in all_model_analyses]
    precisions = [analysis['precision'] for analysis in all_model_analyses]
    recalls = [analysis['recall'] for analysis in all_model_analyses]
    f1_scores = [analysis['f1'] for analysis in all_model_analyses]
    training_times = [analysis['training_time'] for analysis in all_model_analyses]
    
    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Performance Metrics Comparison
    plt.subplot(3, 4, 1)
    x = range(len(model_names))
    width = 0.2
    
    plt.bar([i - 1.5*width for i in x], accuracies, width, label='Accuracy', alpha=0.8)
    plt.bar([i - 0.5*width for i in x], precisions, width, label='Precision', alpha=0.8)
    plt.bar([i + 0.5*width for i in x], recalls, width, label='Recall', alpha=0.8)
    plt.bar([i + 1.5*width for i in x], f1_scores, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 1)
    
    # 2. Training Time Comparison
    plt.subplot(3, 4, 2)
    bars = plt.bar(model_names, training_times, alpha=0.8, color='orange')
    plt.xlabel('Models')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, time_val in zip(bars, training_times):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(training_times)*0.01,
                f'{time_val:.1f}s', ha='center', va='bottom', fontsize=9)
    
    # 3. Accuracy vs Training Time Scatter
    plt.subplot(3, 4, 3)
    scatter = plt.scatter(training_times, accuracies, s=100, alpha=0.7, c=range(len(model_names)), cmap='viridis')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs Training Time')
    plt.grid(True, alpha=0.3)
    
    # Add model name labels
    for i, name in enumerate(model_names):
        plt.annotate(name, (training_times[i], accuracies[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=8)
    
    # 4. Overfitting Analysis Summary
    plt.subplot(3, 4, 4)
    overfitting_gaps = []
    overfitting_statuses = []
    
    for analysis in all_model_analyses:
        if analysis['overfitting_analysis']:
            gap = analysis['overfitting_analysis']['overfitting_gap']
            status = analysis['overfitting_analysis']['status']
            overfitting_gaps.append(gap)
            overfitting_statuses.append(status)
        else:
            overfitting_gaps.append(0)
            overfitting_statuses.append("Unknown")
    
    # Color coding for overfitting
    colors = []
    for gap in overfitting_gaps:
        if gap > 0.1:
            colors.append('red')
        elif gap > 0.05:
            colors.append('orange')
        else:
            colors.append('green')
    
    bars = plt.bar(model_names, overfitting_gaps, color=colors, alpha=0.7)
    plt.xlabel('Models')
    plt.ylabel('Overfitting Gap')
    plt.title('Overfitting Analysis')
    plt.xticks(rotation=45)
    plt.axhline(y=0.1, color='red', linestyle='--', alpha=0.7, label='High overfitting')
    plt.axhline(y=0.05, color='orange', linestyle='--', alpha=0.7, label='Mild overfitting')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 5-8. Individual Confusion Matrices for each model
    for i, analysis in enumerate(all_model_analyses[:4]):  # Show first 4 models
        plt.subplot(3, 4, 5 + i)
        
        model = analysis['model']
        model_name = analysis['model_name']
        
        # Generate predictions for confusion matrix
        from sklearn.datasets import make_classification
        # Since we need test data, we'll use the stored model to predict
        # This is a placeholder - in real implementation, we'd pass X_test, y_test
        
        plt.text(0.5, 0.5, f'{model_name}\nConfusion Matrix\n(Individual analysis\ncompleted above)', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=10)
        plt.axis('off')
    
    # 9. Model Rankings
    plt.subplot(3, 4, 9)
    
    # Sort models by accuracy
    sorted_indices = sorted(range(len(accuracies)), key=lambda i: accuracies[i], reverse=True)
    sorted_names = [model_names[i] for i in sorted_indices]
    sorted_accuracies = [accuracies[i] for i in sorted_indices]
    
    bars = plt.barh(range(len(sorted_names)), sorted_accuracies, alpha=0.8)
    plt.yticks(range(len(sorted_names)), sorted_names)
    plt.xlabel('Accuracy')
    plt.title('Model Rankings by Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Add accuracy values
    for i, (bar, acc) in enumerate(zip(bars, sorted_accuracies)):
        width = bar.get_width()
        plt.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'{acc:.3f}', ha='left', va='center', fontsize=9)
    
    # 10. Performance Distribution
    plt.subplot(3, 4, 10)
    
    metrics_data = [accuracies, precisions, recalls, f1_scores]
    metrics_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    box_plot = plt.boxplot(metrics_data, labels=metrics_labels, patch_artist=True)
    
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('Score')
    plt.title('Performance Distribution Across Models')
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    
    # 11. Summary Statistics Table
    plt.subplot(3, 4, 11)
    plt.axis('off')
    
    best_model_idx = accuracies.index(max(accuracies))
    worst_model_idx = accuracies.index(min(accuracies))
    fastest_model_idx = training_times.index(min(training_times))
    
    summary_text = f"""MODEL COMPARISON SUMMARY

ğŸ�† Best Performing Model:
{model_names[best_model_idx]}
â€¢ Accuracy: {accuracies[best_model_idx]:.4f}
â€¢ F1-Score: {f1_scores[best_model_idx]:.4f}
â€¢ Training Time: {training_times[best_model_idx]:.1f}s

âš¡ Fastest Training Model:
{model_names[fastest_model_idx]}
â€¢ Training Time: {training_times[fastest_model_idx]:.1f}s
â€¢ Accuracy: {accuracies[fastest_model_idx]:.4f}

ğŸ“Š Overall Statistics:
â€¢ Models compared: {len(model_names)}
â€¢ Accuracy range: {min(accuracies):.3f} - {max(accuracies):.3f}
â€¢ Avg accuracy: {np.mean(accuracies):.3f}
â€¢ Total training time: {sum(training_times):.1f}s"""
    
    plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace')
    
    # 12. Efficiency Analysis (Accuracy per second)
    plt.subplot(3, 4, 12)
    
    efficiency_scores = [acc / time if time > 0 else 0 for acc, time in zip(accuracies, training_times)]
    
    bars = plt.bar(model_names, efficiency_scores, alpha=0.8, color='purple')
    plt.xlabel('Models')
    plt.ylabel('Accuracy per Second')
    plt.title('Training Efficiency\n(Accuracy / Training Time)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, eff in zip(bars, efficiency_scores):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + max(efficiency_scores)*0.01,
                f'{eff:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('comprehensive_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print detailed comparison table
    print("\n" + "="*100)
    print("ğŸ“Š DETAILED MODEL COMPARISON TABLE")
    print("="*100)
    
    comparison_df = pd.DataFrame({
        'Model': model_names,
        'Accuracy': [f"{acc:.4f}" for acc in accuracies],
        'Precision': [f"{prec:.4f}" for prec in precisions],
        'Recall': [f"{rec:.4f}" for rec in recalls],
        'F1-Score': [f"{f1:.4f}" for f1 in f1_scores],
        'Training_Time(s)': [f"{time:.2f}" for time in training_times],
        'Efficiency': [f"{eff:.4f}" for eff in efficiency_scores],
        'Overfitting_Gap': [f"{gap:.4f}" for gap in overfitting_gaps]
    })
    
    # Sort by accuracy
    comparison_df = comparison_df.sort_values('Accuracy', ascending=False)
    print(comparison_df.to_string(index=False))
    print("="*100)
    
    print(f"\nâœ… Comprehensive model comparison completed!")
    print(f"ğŸ“� Visualization saved as: comprehensive_model_comparison.png")
    print(f"ğŸ�† Best model: {model_names[best_model_idx]} (Accuracy: {accuracies[best_model_idx]:.4f})")
    print(f"âš¡ Most efficient: {model_names[efficiency_scores.index(max(efficiency_scores))]} (Efficiency: {max(efficiency_scores):.4f})")

if __name__ == "__main__":
    main() 

