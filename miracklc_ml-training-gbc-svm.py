##### IMPORTS #####
"""
This section imports all required libraries.
CPU-based models are selected for performance optimization.

NOTE: Due to the extended training times of SVM and GradientBoosting models, 
this pipeline is configured to train on a limited dataset of 1500 samples 
for demonstration and testing purposes.
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

from tqdm.auto import tqdm

# Performance optimizations
warnings.filterwarnings('ignore')
plt.style.use('default')  # Using default instead of seaborn for compatibility

##### CONFIGURATION #####
"""
Enhanced configuration for better model performance with all new features
"""
class CFG:
    # Seed for reproducibility
    seed = 42
    debug = True
    
    # Data paths (adjust these for your local setup)
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
    enable_pseudo_labeling = False  # Set to True to enable
    pseudo_confidence_threshold = 0.8  # Minimum confidence for pseudo-labels
    pseudo_max_samples_per_class = 50  # Limit pseudo-samples per class
    
    # Quality filtering
    filter_low_quality = True  # Remove samples with rating 0.5-2.5
    min_quality_rating = 2.5  # Minimum rating to keep
    
    # Performance parameters
    n_samples = 1500 if debug else None  # Small sample for quick testing
    min_samples_for_rare_class_elimination = 10  # Higher threshold
    test_size = 0.2
    cv_folds = 3  # Keep at 3 for speed
    
    # PCA parameters
    pca_variance_threshold = 0.95
    
    # Enhanced model configuration with additional models and better hyperparameters
    models_to_train = {
        'SVM': {
            'model': SVC(random_state=seed),
            'param_grid': {
                'classifier__C': [0.1, 1.0, 10.0],  
                'classifier__kernel': ['linear', 'rbf'], 
                'classifier__gamma': ['scale']  
            }
        },
        'GradientBoosting': {
            'model': GradientBoostingClassifier(random_state=seed),
            'param_grid': {
                'classifier__n_estimators': [100, 200],  # Keep 2 values
                'classifier__learning_rate': [0.1, 0.2],  # Reduced from 3 to 2 values
                'classifier__max_depth': [3, 5],  # Reduced from 3 to 2 values
                'classifier__subsample': [0.8]  # Reduced from 2 to 1 value
            }
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
    """Enhanced feature extraction with progress tracking and augmentation"""
    print_section("FEATURE EXTRACTION")
    
    print("ğŸ�µ Extracting enhanced audio features...")
    
    # For demo data, create more sophisticated random features
    if not os.path.exists(cfg.train_datadir):
        print("ğŸ�­ Creating enhanced demo features...")
        
        # Calculate correct feature dimension
        base_size = cfg.TARGET_SHAPE[0] * cfg.TARGET_SHAPE[1]
        additional_features = 13*4 + 6 + 2  # MFCC stats + spectral + rhythm
        n_features = base_size + additional_features
        
        # Create more realistic features based on species
        X = []
        y = df['target'].values
        
        for idx, row in df.iterrows():
            # Create species-specific patterns
            species_id = row['target']
            
            # Base mel spectrogram features with some species-specific patterns
            mel_features = np.random.rand(base_size) * 0.5 + species_id * 0.1
            
            # MFCC features with species variation
            mfcc_features = np.random.rand(52) * 0.3 + species_id * 0.05  # 13*4
            
            # Spectral features
            spectral_features = np.random.rand(6) * 0.2 + species_id * 0.02
            
            # Rhythm features
            rhythm_features = np.random.rand(2) * 0.1 + species_id * 0.01
            
            combined = np.concatenate([mel_features, mfcc_features, spectral_features, rhythm_features])
            X.append(combined)
        
        X = np.array(X).astype(np.float32)
        print(f"âœ“ Enhanced demo feature matrix: {X.shape}")
        return X, y
    
    features = []
    labels = []
    
    start_time = time.time()
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Audio"):
        try:
            feature_vector = extract_enhanced_audio_features(row['filepath'], cfg)
            
            if not np.all(feature_vector == 0):
                features.append(feature_vector)
                labels.append(row['target'])
        except Exception as e:
            print(f"âš ï¸� Skipping {row['filepath']}: {e}")
            continue
    
    X = np.array(features)
    y = np.array(labels)
    
    elapsed_time = time.time() - start_time
    print(f"âœ“ Feature extraction completed: {X.shape}, {elapsed_time:.2f} seconds")
    
    if X.shape[0] == 0:
        raise ValueError("â�Œ No samples remaining!")
    
    return X, y

def apply_mixup_features(X, y, cfg):
    """Apply mixup augmentation to feature vectors"""
    if not cfg.enable_data_augmentation:
        return X, y
    
    print("ğŸ”„ Applying mixup augmentation...")
    
    mixed_X = []
    mixed_y = []
    
    # Keep original data
    mixed_X.extend(X)
    mixed_y.extend(y)
    
    # Generate mixup samples
    n_mixup = int(len(X) * 0.2)  # 20% additional mixup samples
    for _ in range(n_mixup):
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
    return X_mixed, y_mixed

##### PCA ANALYSIS #####
"""
PCA analizi ve optimum bileÅŸen sayÄ±sÄ± belirleme
"""
def analyze_pca_components(X_train, cfg):
    """Determine optimal number of components through PCA analysis"""
    print_section("PCA ANALYSIS")
    
    print("ğŸ”� Analyzing explained variance with PCA...")
    
    # Full PCA analysis
    pca_full = PCA(n_components=None, random_state=cfg.seed)
    pca_full.fit(X_train)
    
    cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
    
    # Find optimal number of components
    n_components_chosen = np.argmax(cumulative_variance >= cfg.pca_variance_threshold) + 1
    
    # Ensure we don't exceed the number of features
    n_components_chosen = min(n_components_chosen, X_train.shape[1])
    
    print("âœ“ PCA analysis completed")
    print(f"ğŸ“Š {n_components_chosen} components selected for {cfg.pca_variance_threshold*100}% variance")
    print(f"ğŸ“‰ Dimensionality reduction: {X_train.shape[1]} â†’ {n_components_chosen} ({((X_train.shape[1] - n_components_chosen) / X_train.shape[1] * 100):.1f}% reduction)")
    
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
    """Optimized model training with validation"""
    print_section("MODEL TRAINING AND EVALUATION")
    
    results = []
    best_model = None
    best_model_name = ""
    best_accuracy = -1
    
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
        
        # Update best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = best_estimator
            best_model_name = model_name
        
        print(f"ğŸ“Š {model_name} Results:")
        print(f"   CV Accuracy: {cv_mean:.4f} (Â±{cv_std:.4f})")
        print(f"   Test Accuracy: {test_accuracy:.4f}")
        print(f"   Training Time: {training_time:.2f} seconds")
    
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
        
        # 6. Model Training
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
        
        # 11. Save Best Model
        print_section("MODEL SAVING")
        model_filename = f"best_model_{best_model_name.lower()}_acc_{accuracy:.3f}.joblib"
        joblib.dump(best_model, model_filename)
        print(f"âœ“ Best model saved: {model_filename}")
        
        # 12. Final Summary
        total_time = time.time() - total_start_time
        
        print_section("ğŸ“‹ SUMMARY REPORT")
        
        summary_text = f"""
ğŸ�¯ BirdCLEF Model Training Completed!

â�±ï¸�  Total Time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)

ğŸ“Š Data Information:
â€¢ Total sample count: {len(train_df)}
â€¢ Number of classes: {CFG.num_classes}
â€¢ Feature dimensions: {X.shape[1]} â†’ {optimal_n_components} (PCA)
â€¢ Train/Test: {len(y_train)}/{len(y_test)}

ğŸ�† Best Model: {best_model_name}
â€¢ Test Accuracy: {accuracy:.4f}
â€¢ Precision: {precision:.4f}
â€¢ Recall: {recall:.4f}
â€¢ F1-Score: {f1:.4f}

ğŸ“� Saved Files:
â€¢ Model: {model_filename}
â€¢ Label Encoder: label_encoder.joblib
â€¢ PCA Transformer: pca_transformer.joblib
â€¢ Training Log: training_log.log

ğŸ“ˆ Visualizations:
â€¢ PCA Analysis: pca_analysis.png
â€¢ Model Evaluation: evaluation_{best_model_name.lower()}.png
â€¢ Feature Importance: feature_importance_{best_model_name.lower()}.png
â€¢ Overfitting Analysis: overfitting_analysis_{best_model_name.lower()}.png

âš™ï¸� Configuration Used:
â€¢ Data Augmentation: {'âœ“ Enabled' if CFG.enable_data_augmentation else 'âœ— Disabled'}
â€¢ Pseudo-labeling: {'âœ“ Enabled' if CFG.enable_pseudo_labeling else 'âœ— Disabled'}
â€¢ Quality Filtering: {'âœ“ Enabled' if CFG.filter_low_quality else 'âœ— Disabled'}
        """
        
        print(summary_text)
        print("ğŸ�‰ Pipeline completed successfully!")
        
        # Performance recommendations
    except Exception as e:
        print(f"â�Œ Pipeline error: {e}")
        raise

if __name__ == "__main__":
    main() 

