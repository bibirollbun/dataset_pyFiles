# ==============================================================================
# CMI Detect Behavior with Sensor Data - Memory-Optimized Implementation
# Enhanced with robust memory management and resource optimization
# ==============================================================================

# 1. Imports and Setup
# ==============================================================================

import os
import gc
import json
import joblib
import numpy as np
import pandas as pd
import polars as pl
import random
import copy
import warnings
import psutil
from pathlib import Path
from typing import Dict, List, Tuple, Any
from contextlib import contextmanager

warnings.filterwarnings("ignore")

# Standard libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.utils import Sequence, to_categorical, pad_sequences
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate, GRU, GaussianNoise
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import backend as K

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Set matplotlib style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

print("All libraries imported successfully!")

# ==============================================================================
# 2. Memory Management System
# ==============================================================================

class MemoryManager:
    """Comprehensive memory management and monitoring system"""
    
    def __init__(self, memory_threshold=0.85, verbose=True):
        """
        Initialize memory manager
        
        Args:
            memory_threshold: Memory usage threshold (0-1) to trigger cleanup
            verbose: Print memory status
        """
        self.memory_threshold = memory_threshold
        self.verbose = verbose
        self.peak_memory = 0
        self.cleanup_count = 0
        
    def get_memory_usage(self):
        """Get current memory usage statistics"""
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        virtual_memory = psutil.virtual_memory()
        
        memory_stats = {
            'process_memory_mb': memory_info.rss / 1024 / 1024,
            'process_memory_percent': process.memory_percent(),
            'system_memory_percent': virtual_memory.percent / 100,
            'available_memory_gb': virtual_memory.available / 1024 / 1024 / 1024
        }
        
        # Update peak memory
        if memory_stats['process_memory_mb'] > self.peak_memory:
            self.peak_memory = memory_stats['process_memory_mb']
        
        return memory_stats
    
    def check_memory_status(self, context=""):
        """Check and print memory status"""
        stats = self.get_memory_usage()
        
        if self.verbose:
            print(f"[Memory {context}] Process: {stats['process_memory_mb']:.1f}MB "
                  f"({stats['process_memory_percent']:.1f}%) | "
                  f"System: {stats['system_memory_percent']:.1f}% | "
                  f"Available: {stats['available_memory_gb']:.1f}GB")
        
        return stats
    
    def force_cleanup(self, aggressive=False):
        """Aggressive memory cleanup"""
        if self.verbose:
            print(f"[Memory Manager] Performing {'aggressive' if aggressive else 'standard'} cleanup...")
        
        # Clear TensorFlow session
        try:
            K.clear_session()
        except:
            pass
        
        # Force garbage collection multiple times
        for _ in range(3 if aggressive else 2):
            gc.collect()
        
        # Clear matplotlib cache
        try:
            plt.close('all')
        except:
            pass
        
        self.cleanup_count += 1
        
        if self.verbose:
            stats = self.get_memory_usage()
            print(f"[Memory Manager] Cleanup #{self.cleanup_count} completed. "
                  f"Current usage: {stats['process_memory_mb']:.1f}MB")
    
    def memory_check_decorator(self, func):
        """Decorator to monitor memory usage around function calls"""
        def wrapper(*args, **kwargs):
            if self.verbose:
                print(f"\n[Memory Check] Starting {func.__name__}")
            
            # Pre-execution memory check
            pre_stats = self.check_memory_status("PRE")
            
            # Check if cleanup needed
            if pre_stats['system_memory_percent'] > self.memory_threshold:
                self.force_cleanup()
            
            try:
                # Execute function
                result = func(*args, **kwargs)
                
                # Post-execution memory check
                post_stats = self.check_memory_status("POST")
                
                return result
                
            except MemoryError as e:
                print(f"[Memory Manager] Memory error in {func.__name__}: {e}")
                self.force_cleanup(aggressive=True)
                raise
            
        return wrapper
    
    @contextmanager
    def memory_context(self, context_name=""):
        """Context manager for memory monitoring"""
        print(f"\n[Memory Context] Entering {context_name}")
        pre_stats = self.check_memory_status("ENTER")
        
        try:
            yield self
        finally:
            post_stats = self.check_memory_status("EXIT")
            memory_diff = post_stats['process_memory_mb'] - pre_stats['process_memory_mb']
            
            if self.verbose:
                print(f"[Memory Context] {context_name} completed. "
                      f"Memory change: {memory_diff:+.1f}MB")
            
            # Cleanup if memory usage increased significantly
            if memory_diff > 500:  # 500MB threshold
                self.force_cleanup()

# Global memory manager instance
memory_manager = MemoryManager(verbose=True)

# ==============================================================================
# 3. Enhanced Configuration with Memory Optimization
# ==============================================================================

def get_config():
    """Configuration dictionary optimized for memory usage"""
    config = {
        'sr': 10,
        'seq_len': 64,  # Reduced from 128 to save memory
        'stride': 2,    # Increased stride to reduce sequences
        'batch_size': 16,  # Reduced batch size
        'epochs': 20,
        'folds': 3,     # Reduced folds to save memory
        'l2': 1e-4,
        'optimizer': 'adam',
        'patience': 7,  # Increased patience for better convergence
        'verbose': 1,
        'learning_rate': 1e-3,
        'memory_threshold': 0.80,
        'checkpoint_frequency': 5,  # Save model every 5 epochs
        'early_stop_on_memory': True,
    }
    return config

# ==============================================================================
# 4. Memory-Optimized Data Loading
# ==============================================================================

def load_csv_data_chunked(file_path: str, chunk_size: int = 10000) -> pd.DataFrame:
    """Load CSV data in chunks to manage memory"""
    try:
        if os.path.exists(file_path):
            print(f"Loading {file_path} in chunks...")
            
            # Read in chunks and concatenate
            chunks = []
            total_rows = 0
            
            for chunk in pd.read_csv(file_path, chunksize=chunk_size):
                chunks.append(chunk)
                total_rows += len(chunk)
                
                # Memory check every 100k rows
                if total_rows % 100000 == 0:
                    memory_manager.check_memory_status(f"Loaded {total_rows} rows")
            
            df = pd.concat(chunks, ignore_index=True)
            del chunks  # Free memory immediately
            gc.collect()
            
            print(f"Loaded {file_path}: {df.shape}")
            return df
        else:
            print(f"File not found: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return None

@memory_manager.memory_check_decorator
def load_all_data():
    """Load all available data files with memory management"""
    data_paths = [
        "/kaggle/input/cmi-detect-behavior-with-sensor-data/",
        "./data/",
        "../input/cmi-detect-behavior-with-sensor-data/"
    ]
    
    data = {}
    files_to_load = {
        'train': 'train.csv',
        'test': 'test.csv', 
        'train_demographics': 'train_demographics.csv',
        'test_demographics': 'test_demographics.csv'
    }
    
    for base_path in data_paths:
        if os.path.exists(base_path):
            print(f"Found data directory: {base_path}")
            for key, filename in files_to_load.items():
                full_path = os.path.join(base_path, filename)
                
                with memory_manager.memory_context(f"Loading {filename}"):
                    df = load_csv_data_chunked(full_path)
                    if df is not None:
                        data[key] = df
            break
    
    return data

def extract_sensor_features_optimized(df: pd.DataFrame) -> np.ndarray:
    """Extract sensor features with memory optimization"""
    print("[Feature Extraction] Starting optimized feature extraction...")
    
    # Define sensor columns in priority order
    priority_sensors = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    thermal_cols = [col for col in df.columns if col.startswith('thm_')][:20]  # Limit thermal sensors
    tof_cols = [col for col in df.columns if col.startswith('tof_')][:30]      # Limit ToF sensors
    
    sensor_cols = priority_sensors + thermal_cols + tof_cols
    available_cols = [col for col in sensor_cols if col in df.columns]
    
    if not available_cols:
        print("Warning: No sensor columns found!")
        return np.array([])
    
    print(f"Using {len(available_cols)} sensor features: {available_cols[:10]}...")
    
    # Process features in chunks to manage memory
    chunk_size = 50000
    feature_chunks = []
    
    for i in range(0, len(df), chunk_size):
        chunk = df.iloc[i:i+chunk_size]
        chunk_features = chunk[available_cols].values
        chunk_features = np.nan_to_num(chunk_features, nan=0.0, posinf=0.0, neginf=0.0)
        feature_chunks.append(chunk_features)
        
        if i % (chunk_size * 5) == 0:  # Memory check every 250k rows
            memory_manager.check_memory_status(f"Processed {min(i+chunk_size, len(df))} rows")
    
    features = np.vstack(feature_chunks)
    del feature_chunks, chunk_features  # Immediate cleanup
    gc.collect()
    
    print(f"[Feature Extraction] Completed. Shape: {features.shape}")
    return features

# ==============================================================================
# 5. Memory-Optimized Model Architecture
# ==============================================================================

def build_memory_efficient_model(config: Dict, input_shape: Tuple[int, int]) -> Model:
    """Build memory-efficient model architecture"""
    
    def residual_block(x, filters, kernel_size=3):
        """Lightweight residual block"""
        res = x
        x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(config['l2']))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(config['l2']))(x)
        x = BatchNormalization()(x)
        
        # Skip connection with dimension matching
        if res.shape[-1] != filters:
            res = Conv1D(filters, 1, padding='same')(res)
        
        x = add([res, x])
        x = Activation('relu')(x)
        return x

    input_layer = Input(shape=input_shape, name='sensor_input')
    
    # Light noise for regularization
    x = GaussianNoise(0.005)(input_layer)
    
    # Initial feature extraction
    x = Conv1D(32, 5, padding='same', kernel_regularizer=l2(config['l2']))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.2)(x)
    
    # Residual blocks with progressive complexity
    x = residual_block(x, 32)
    x = Dropout(0.2)(x)
    x = residual_block(x, 64)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)
    
    # Recurrent processing
    x = Bidirectional(GRU(32, return_sequences=False, kernel_regularizer=l2(config['l2'])))(x)
    x = Dropout(0.4)(x)
    
    # Classification head
    x = Dense(64, activation='relu', kernel_regularizer=l2(config['l2']))(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu', kernel_regularizer=l2(config['l2']))(x)
    x = Dropout(0.3)(x)
    
    # Output layer
    output_layer = Dense(4, activation='softmax', name='behavior_output')(x)  # 4 classes
    
    model = Model(inputs=input_layer, outputs=output_layer)
    
    print(f"[Model Architecture] Built memory-efficient model")
    print(f"Total parameters: {model.count_params():,}")
    
    return model

# ==============================================================================
# 6. Enhanced Model Callbacks with Peak Accuracy Tracking
# ==============================================================================

class PeakAccuracyTracker(tf.keras.callbacks.Callback):
    """Custom callback to track peak accuracy and save best models"""
    
    def __init__(self, save_path='best_model_peak_acc.h5', verbose=1):
        super().__init__()
        self.save_path = save_path
        self.verbose = verbose
        self.peak_val_accuracy = 0.0
        self.peak_train_accuracy = 0.0
        self.peak_epoch = 0
        self.accuracy_history = []
        self.val_accuracy_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        current_val_acc = logs.get('val_accuracy', 0)
        current_train_acc = logs.get('accuracy', 0)
        
        self.accuracy_history.append(current_train_acc)
        self.val_accuracy_history.append(current_val_acc)
        
        # Check for new peak validation accuracy
        if current_val_acc > self.peak_val_accuracy:
            self.peak_val_accuracy = current_val_acc
            self.peak_train_accuracy = current_train_acc
            self.peak_epoch = epoch + 1
            
            # Save the model at peak performance
            self.model.save(self.save_path)
            
            if self.verbose:
                print(f"\n[Peak Accuracy] New peak validation accuracy: {self.peak_val_accuracy:.4f} "
                      f"(train: {self.peak_train_accuracy:.4f}) at epoch {self.peak_epoch}")
                print(f"[Peak Accuracy] Model saved to {self.save_path}")
    
    def plot_accuracy_curves(self):
        """Plot accuracy curves with peak markers"""
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        epochs = range(1, len(self.accuracy_history) + 1)
        plt.plot(epochs, self.accuracy_history, 'b-', label='Training Accuracy', linewidth=2)
        plt.plot(epochs, self.val_accuracy_history, 'r-', label='Validation Accuracy', linewidth=2)
        
        # Mark peak accuracy
        plt.scatter([self.peak_epoch], [self.peak_val_accuracy], 
                   color='red', s=100, zorder=5, label=f'Peak Val Acc: {self.peak_val_accuracy:.4f}')
        plt.scatter([self.peak_epoch], [self.peak_train_accuracy], 
                   color='blue', s=100, zorder=5, label=f'Peak Train Acc: {self.peak_train_accuracy:.4f}')
        
        plt.title('Model Accuracy Over Time')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Zoomed view around peak
        plt.subplot(2, 2, 2)
        zoom_start = max(0, self.peak_epoch - 10)
        zoom_end = min(len(epochs), self.peak_epoch + 10)
        zoom_epochs = epochs[zoom_start:zoom_end]
        
        if zoom_epochs:
            plt.plot(zoom_epochs, self.accuracy_history[zoom_start:zoom_end], 
                    'b-', label='Training Accuracy', linewidth=2)
            plt.plot(zoom_epochs, self.val_accuracy_history[zoom_start:zoom_end], 
                    'r-', label='Validation Accuracy', linewidth=2)
            plt.scatter([self.peak_epoch], [self.peak_val_accuracy], 
                       color='red', s=100, zorder=5)
            plt.title(f'Zoomed View Around Peak (Epoch {self.peak_epoch})')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Accuracy improvement over time
        plt.subplot(2, 2, 3)
        val_acc_diff = np.diff(self.val_accuracy_history)
        plt.plot(range(2, len(epochs) + 1), val_acc_diff, 'g-', label='Val Accuracy Change', linewidth=2)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.title('Validation Accuracy Change per Epoch')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy Change')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Summary statistics
        plt.subplot(2, 2, 4)
        stats_text = f"""Peak Performance Summary:
        
Peak Validation Accuracy: {self.peak_val_accuracy:.4f}
Peak Training Accuracy: {self.peak_train_accuracy:.4f}
Peak Epoch: {self.peak_epoch}

Final Validation Accuracy: {self.val_accuracy_history[-1]:.4f}
Final Training Accuracy: {self.accuracy_history[-1]:.4f}

Accuracy Range (Val): {min(self.val_accuracy_history):.4f} - {max(self.val_accuracy_history):.4f}
Accuracy Range (Train): {min(self.accuracy_history):.4f} - {max(self.accuracy_history):.4f}
        """
        
        plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, 
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()

class MemoryMonitorCallback(tf.keras.callbacks.Callback):
    """Monitor memory usage during training"""
    
    def __init__(self, memory_manager, check_frequency=5):
        super().__init__()
        self.memory_manager = memory_manager
        self.check_frequency = check_frequency
        
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.check_frequency == 0:
            self.memory_manager.check_memory_status(f"Epoch {epoch+1}")
            
            # Force cleanup if memory usage is high
            stats = self.memory_manager.get_memory_usage()
            if stats['system_memory_percent'] > 0.85:
                print(f"[Memory Monitor] High memory usage detected, performing cleanup...")
                self.memory_manager.force_cleanup()

# ==============================================================================
# 7. Memory-Optimized Data Generator
# ==============================================================================

class MemoryEfficientDataGenerator(Sequence):
    """Memory-efficient data generator with cleanup"""
    
    def __init__(self, X, y, config, shuffle=True, memory_manager=None):
        self.X = X
        self.y = y
        self.config = config
        self.shuffle = shuffle
        self.memory_manager = memory_manager
        self.indices = np.arange(len(X))
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        return int(np.ceil(len(self.X) / self.config['batch_size']))
    
    def __getitem__(self, idx):
        start_idx = idx * self.config['batch_size']
        end_idx = min((idx + 1) * self.config['batch_size'], len(self.X))
        batch_indices = self.indices[start_idx:end_idx]
        
        X_batch = self.X[batch_indices].copy()  # Explicit copy to manage memory
        y_batch = self.y[batch_indices].copy()
        
        return X_batch, y_batch
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        # Periodic cleanup
        if hasattr(self, 'memory_manager') and self.memory_manager:
            if np.random.random() < 0.1:  # 10% chance per epoch end
                gc.collect()

# ==============================================================================
# 8. Memory-Optimized Training Pipeline
# ==============================================================================

@memory_manager.memory_check_decorator
def prepare_sequences_optimized(X, y, config):
    """Prepare sequences with memory optimization"""
    print(f"[Sequence Preparation] Creating sequences with length {config['seq_len']}, stride {config['stride']}")
    
    seq_len = config['seq_len']
    stride = config['stride']
    
    # Calculate number of sequences to pre-allocate arrays
    num_sequences = (len(X) - seq_len) // stride + 1
    print(f"[Sequence Preparation] Will create {num_sequences:,} sequences")
    
    # Pre-allocate arrays
    sequences = np.zeros((num_sequences, seq_len, X.shape[1]), dtype=np.float32)
    labels = np.zeros(num_sequences, dtype=np.int32)
    
    # Fill arrays in chunks
    chunk_size = 1000
    for i, start_idx in enumerate(range(0, len(X) - seq_len + 1, stride)):
        sequences[i] = X[start_idx:start_idx + seq_len]
        labels[i] = y[start_idx + seq_len - 1]
        
        # Periodic memory check
        if i % (chunk_size * 10) == 0 and i > 0:
            memory_manager.check_memory_status(f"Sequences: {i}/{num_sequences}")
    
    print(f"[Sequence Preparation] Completed. Shape: {sequences.shape}")
    return sequences, labels

@memory_manager.memory_check_decorator
def train_model_with_validation_optimized(X_train, y_train, X_val, y_val, config):
    """Memory-optimized training with peak accuracy tracking"""
    
    print("\n" + "="*60)
    print("STARTING MEMORY-OPTIMIZED TRAINING")
    print("="*60)
    
    with memory_manager.memory_context("Sequence Preparation"):
        # Prepare sequences
        X_train_seq, y_train_seq = prepare_sequences_optimized(X_train, y_train, config)
        X_val_seq, y_val_seq = prepare_sequences_optimized(X_val, y_val, config)
    
    # Clear original arrays to save memory
    del X_train, y_train, X_val, y_val
    gc.collect()
    
    print(f"Training sequences: {X_train_seq.shape}")
    print(f"Validation sequences: {X_val_seq.shape}")
    
    with memory_manager.memory_context("Model Building"):
        # Build and compile model
        input_shape = (X_train_seq.shape[1], X_train_seq.shape[2])
        model = build_memory_efficient_model(config, input_shape)
        
        # Compile model
        optimizer = Adam(learning_rate=config['learning_rate'])
        model.compile(
            optimizer=optimizer, 
            loss='sparse_categorical_crossentropy',  # Use sparse for memory efficiency
            metrics=['accuracy']
        )
    
    # Print model summary
    model.summary()
    
    # Initialize peak accuracy tracker
    peak_tracker = PeakAccuracyTracker(save_path='best_model_peak_accuracy.h5', verbose=1)
    memory_monitor = MemoryMonitorCallback(memory_manager, check_frequency=3)
    
    # Enhanced callbacks
    callbacks = [
        peak_tracker,
        memory_monitor,
        EarlyStopping(
            monitor='val_accuracy',
            patience=config['patience'], 
            restore_best_weights=True, 
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.5, 
            patience=4, 
            verbose=1,
            mode='max',
            min_lr=1e-6
        ),
        ModelCheckpoint(
            'checkpoint_model.h5', 
            monitor='val_accuracy',
            save_best_only=True, 
            verbose=1,
            mode='max'
        )
    ]
    
    print(f"\n[Training] Starting training for {config['epochs']} epochs...")
    
    try:
        with memory_manager.memory_context("Model Training"):
            # Train model with memory monitoring
            history = model.fit(
                X_train_seq, y_train_seq,
                validation_data=(X_val_seq, y_val_seq),
                epochs=config['epochs'],
                batch_size=config['batch_size'],
                callbacks=callbacks,
                verbose=config['verbose'],
                shuffle=True
            )
        
        print(f"\n[Training] Completed successfully!")
        print(f"Peak validation accuracy: {peak_tracker.peak_val_accuracy:.4f} at epoch {peak_tracker.peak_epoch}")
        
        # Plot training curves with peak accuracy
        peak_tracker.plot_accuracy_curves()
        
        # Final memory status
        memory_manager.check_memory_status("Training Complete")
        
        return model, history, peak_tracker
        
    except Exception as e:
        print(f"[Training] Error occurred: {e}")
        memory_manager.force_cleanup(aggressive=True)
        raise

# ==============================================================================
# 9. Enhanced Main Pipeline with Memory Management
# ==============================================================================

def main_training_pipeline_optimized():
    """Complete memory-optimized training pipeline"""
    
    print("="*80)
    print("CMI BEHAVIOR DETECTION - MEMORY-OPTIMIZED PIPELINE")
    print("="*80)
    
    # Initialize memory tracking
    memory_manager.check_memory_status("Pipeline Start")
    
    # Load configuration
    config = get_config()
    print(f"Configuration: {config}")
    
    # Load data with memory management
    print("\n1. Loading Data...")
    with memory_manager.memory_context("Data Loading"):
        data = load_all_data()
    
    if not data or 'train' not in data or data['train'] is None:
        print("No training data found. Please ensure data files are available.")
        return None
    
    # Prepare training data
    print("\n3. Preparing Training Data...")
    train_df = data['train']
    
    with memory_manager.memory_context("Feature Extraction"):
        # Extract features
        X = extract_sensor_features_optimized(train_df)
        
        if X.size == 0:
            print("No sensor features found!")
            return None
        
        # Prepare labels
        if 'behavior' in train_df.columns:
            le = LabelEncoder()
            y = le.fit_transform(train_df['behavior'])
            class_names = list(le.classes_)
            print(f"Classes: {class_names}")
        else:
            print("No behavior column found for labels!")
            return None
    
    # Clear train_df to save memory
    del train_df
    gc.collect()
    
    with memory_manager.memory_context("Data Normalization"):
        # Normalize features in chunks if dataset is large
        if X.shape[0] > 100000:
            print("[Normalization] Large dataset detected, using chunked normalization...")
            scaler = StandardScaler()
            
            # Fit scaler on subset
            sample_size = min(50000, X.shape[0])
            sample_idx = np.random.choice(X.shape[0], sample_size, replace=False)
            scaler.fit(X[sample_idx])
            
            # Transform in chunks
            chunk_size = 10000
            X_scaled_chunks = []
            
            for i in range(0, X.shape[0], chunk_size):
                chunk = X[i:i+chunk_size]
                chunk_scaled = scaler.transform(chunk)
                X_scaled_chunks.append(chunk_scaled)
                
                if i % (chunk_size * 5) == 0:
                    memory_manager.check_memory_status(f"Normalized {i+chunk_size}/{X.shape[0]}")
            
            X_scaled = np.vstack(X_scaled_chunks)
            del X_scaled_chunks, chunk, chunk_scaled
            gc.collect()
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
    
    # Clear original X to
#*********************************************************************************************
# Clear original X to save memory
    del X
    gc.collect()
    
    print(f"Features prepared: {X_scaled.shape}")
    print(f"Labels prepared: {y.shape}")
    print(f"Memory after preparation:")
    memory_manager.check_memory_status("Data Prepared")
    
    # Train-validation split with stratification
    print("\n4. Splitting Data...")
    with memory_manager.memory_context("Data Splitting"):
        X_train, X_val, y_train, y_val = train_test_split(
            X_scaled, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=y
        )
        
        print(f"Train set: {X_train.shape}, {y_train.shape}")
        print(f"Validation set: {X_val.shape}, {y_val.shape}")
        
        # Clear full dataset from memory
        del X_scaled, y
        gc.collect()
    
    # Model training with memory optimization
    print("\n5. Training Model...")
    try:
        model, history, peak_tracker = train_model_with_validation_optimized(
            X_train, y_train, X_val, y_val, config
        )
        
        print(f"\n6. Training Summary:")
        print(f"Peak validation accuracy: {peak_tracker.peak_val_accuracy:.4f}")
        print(f"Peak training accuracy: {peak_tracker.peak_train_accuracy:.4f}")
        print(f"Peak achieved at epoch: {peak_tracker.peak_epoch}")
        
        # Save final results
        with memory_manager.memory_context("Results Saving"):
            # Save training history
            history_dict = {
                'accuracy': peak_tracker.accuracy_history,
                'val_accuracy': peak_tracker.val_accuracy_history,
                'peak_val_accuracy': peak_tracker.peak_val_accuracy,
                'peak_train_accuracy': peak_tracker.peak_train_accuracy,
                'peak_epoch': peak_tracker.peak_epoch
            }
            
            try:
                import json
                with open('training_history.json', 'w') as f:
                    json.dump(history_dict, f, indent=2)
                print("Training history saved to training_history.json")
            except Exception as e:
                print(f"Could not save training history: {e}")
            
            # Save scaler and label encoder
            try:
                joblib.dump(scaler, 'feature_scaler.pkl')
                joblib.dump(le, 'label_encoder.pkl')
                print("Scaler and label encoder saved")
            except Exception as e:
                print(f"Could not save preprocessing objects: {e}")
        
        # Final memory report
        print(f"\n7. Memory Usage Summary:")
        print(f"Peak memory usage: {memory_manager.peak_memory:.1f}MB")
        print(f"Total cleanups performed: {memory_manager.cleanup_count}")
        memory_manager.check_memory_status("Pipeline Complete")
        
        return {
            'model': model,
            'history': history_dict,
            'peak_tracker': peak_tracker,
            'scaler': scaler,
            'label_encoder': le,
            'config': config,
            'class_names': class_names
        }
        
    except Exception as e:
        print(f"Training failed: {e}")
        memory_manager.force_cleanup(aggressive=True)
        return None

# ==============================================================================
# 10. Evaluation and Prediction Functions
# ==============================================================================

@memory_manager.memory_check_decorator
def evaluate_model_performance(results_dict, X_test=None, y_test=None):
    """Comprehensive model evaluation with memory management"""
    
    if results_dict is None:
        print("No results to evaluate")
        return
    
    model = results_dict['model']
    peak_tracker = results_dict['peak_tracker']
    class_names = results_dict['class_names']
    
    print("\n" + "="*60)
    print("MODEL PERFORMANCE EVALUATION")
    print("="*60)
    
    # Training performance summary
    print(f"\n1. Training Performance Summary:")
    print(f"   Peak Validation Accuracy: {peak_tracker.peak_val_accuracy:.4f}")
    print(f"   Peak Training Accuracy: {peak_tracker.peak_train_accuracy:.4f}")
    print(f"   Peak Epoch: {peak_tracker.peak_epoch}")
    print(f"   Final Validation Accuracy: {peak_tracker.val_accuracy_history[-1]:.4f}")
    
    # Performance analysis
    val_acc_improvement = peak_tracker.peak_val_accuracy - peak_tracker.val_accuracy_history[0]
    print(f"   Validation Accuracy Improvement: {val_acc_improvement:.4f}")
    
    # Overfitting analysis
    accuracy_gap = peak_tracker.peak_train_accuracy - peak_tracker.peak_val_accuracy
    print(f"   Train-Val Accuracy Gap: {accuracy_gap:.4f}")
    
    if accuracy_gap > 0.1:
        print("     Warning: Possible overfitting detected")
    elif accuracy_gap < 0.05:
        print("    Good generalization performance")
    else:
        print("     Moderate generalization performance")
    
    # Test set evaluation if provided
    if X_test is not None and y_test is not None:
        print(f"\n2. Test Set Evaluation:")
        
        with memory_manager.memory_context("Test Evaluation"):
            # Prepare test sequences
            config = results_dict['config']
            X_test_seq, y_test_seq = prepare_sequences_optimized(X_test, y_test, config)
            
            # Predict in batches to manage memory
            batch_size = config['batch_size'] * 2  # Larger batch for inference
            test_predictions = []
            test_true_labels = []
            
            for i in range(0, len(X_test_seq), batch_size):
                batch_X = X_test_seq[i:i+batch_size]
                batch_y = y_test_seq[i:i+batch_size]
                
                batch_pred = model.predict(batch_X, verbose=0)
                batch_pred_classes = np.argmax(batch_pred, axis=1)
                
                test_predictions.extend(batch_pred_classes)
                test_true_labels.extend(batch_y)
                
                if i % (batch_size * 5) == 0:
                    memory_manager.check_memory_status(f"Test prediction batch {i//batch_size + 1}")
            
            # Calculate test accuracy
            test_accuracy = accuracy_score(test_true_labels, test_predictions)
            print(f"   Test Accuracy: {test_accuracy:.4f}")
            
            # Classification report
            print(f"\n   Classification Report:")
            print(classification_report(test_true_labels, test_predictions, 
                                      target_names=class_names))
            
            # Confusion matrix
            cm = confusion_matrix(test_true_labels, test_predictions)
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title('Confusion Matrix - Test Set')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
    
    # Model architecture summary
    print(f"\n3. Model Architecture Summary:")
    print(f"   Total Parameters: {model.count_params():,}")
    print(f"   Model Size: {model.count_params() * 4 / 1024 / 1024:.2f} MB (float32)")
    
    # Memory usage summary
    print(f"\n4. Memory Usage Summary:")
    print(f"   Peak Memory Usage: {memory_manager.peak_memory:.1f} MB")
    print(f"   Total Cleanups: {memory_manager.cleanup_count}")

@memory_manager.memory_check_decorator
def predict_behavior_sequences(model, X_new, scaler, label_encoder, config, sequence_length=None):
    """Make predictions on new sensor data"""
    
    print("\n" + "="*50)
    print("BEHAVIOR PREDICTION")
    print("="*50)
    
    if sequence_length is None:
        sequence_length = config['seq_len']
    
    # Normalize features
    with memory_manager.memory_context("Feature Normalization"):
        X_scaled = scaler.transform(X_new)
    
    # Create sequences
    with memory_manager.memory_context("Sequence Creation"):
        X_sequences, _ = prepare_sequences_optimized(
            X_scaled, 
            np.zeros(len(X_scaled)),  # Dummy labels
            config
        )
    
    print(f"Created {len(X_sequences)} prediction sequences")
    
    # Predict in batches
    predictions = []
    confidence_scores = []
    batch_size = config['batch_size'] * 2
    
    with memory_manager.memory_context("Prediction"):
        for i in range(0, len(X_sequences), batch_size):
            batch_X = X_sequences[i:i+batch_size]
            batch_pred = model.predict(batch_X, verbose=0)
            
            batch_classes = np.argmax(batch_pred, axis=1)
            batch_confidence = np.max(batch_pred, axis=1)
            
            predictions.extend(batch_classes)
            confidence_scores.extend(batch_confidence)
            
            if i % (batch_size * 5) == 0:
                print(f"Processed {i + len(batch_X)}/{len(X_sequences)} sequences")
    
    # Convert predictions to labels
    predicted_behaviors = label_encoder.inverse_transform(predictions)
    
    # Create results summary
    results = {
        'predictions': predicted_behaviors,
        'prediction_classes': predictions,
        'confidence_scores': confidence_scores,
        'sequence_count': len(predictions)
    }
    
    # Print summary
    unique_behaviors, counts = np.unique(predicted_behaviors, return_counts=True)
    print(f"\nPrediction Summary:")
    for behavior, count in zip(unique_behaviors, counts):
        percentage = (count / len(predictions)) * 100
        avg_confidence = np.mean([conf for i, conf in enumerate(confidence_scores) 
                                 if predicted_behaviors[i] == behavior])
        print(f"  {behavior}: {count} sequences ({percentage:.1f}%) - "
              f"Avg confidence: {avg_confidence:.3f}")
    
    return results

# ==============================================================================
# 11. Utility Functions
# ==============================================================================

def save_training_artifacts(results_dict, save_dir='./training_artifacts/'):
    """Save all training artifacts for later use"""
    
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\nSaving training artifacts to {save_dir}...")
    
    try:
        # Save model
        results_dict['model'].save(os.path.join(save_dir, 'final_model.h5'))
        print(" Model saved")
        
        # Save preprocessing objects
        joblib.dump(results_dict['scaler'], os.path.join(save_dir, 'scaler.pkl'))
        joblib.dump(results_dict['label_encoder'], os.path.join(save_dir, 'label_encoder.pkl'))
        print(" Preprocessing objects saved")
        
        # Save configuration
        with open(os.path.join(save_dir, 'config.json'), 'w') as f:
            json.dump(results_dict['config'], f, indent=2)
        print(" Configuration saved")
        
        # Save training history
        history_data = {
            'accuracy': results_dict['history']['accuracy'],
            'val_accuracy': results_dict['history']['val_accuracy'],
            'peak_val_accuracy': results_dict['history']['peak_val_accuracy'],
            'peak_train_accuracy': results_dict['history']['peak_train_accuracy'],
            'peak_epoch': results_dict['history']['peak_epoch'],
            'class_names': results_dict['class_names']
        }
        
        with open(os.path.join(save_dir, 'training_history.json'), 'w') as f:
            json.dump(history_data, f, indent=2)
        print(" Training history saved")
        
        print(f"All artifacts saved successfully to {save_dir}")
        
    except Exception as e:
        print(f" Error saving artifacts: {e}")

def load_training_artifacts(load_dir='./training_artifacts/'):
    """Load previously saved training artifacts"""
    
    print(f"Loading training artifacts from {load_dir}...")
    
    try:
        # Load model
        model = load_model(os.path.join(load_dir, 'final_model.h5'))
        
        # Load preprocessing objects
        scaler = joblib.load(os.path.join(load_dir, 'scaler.pkl'))
        label_encoder = joblib.load(os.path.join(load_dir, 'label_encoder.pkl'))
        
        # Load configuration
        with open(os.path.join(load_dir, 'config.json'), 'r') as f:
            config = json.load(f)
        
        # Load training history
        with open(os.path.join(load_dir, 'training_history.json'), 'r') as f:
            history_data = json.load(f)
        
        results = {
            'model': model,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'config': config,
            'history': history_data,
            'class_names': history_data['class_names']
        }
        
        print(" All artifacts loaded successfully")
        return results
        
    except Exception as e:
        print(f" Error loading artifacts: {e}")
        return None

# ==============================================================================
# 12. Main Execution Block
# ==============================================================================

if __name__ == "__main__":
    print("Starting CMI Behavior Detection Pipeline...")
    
    try:
        # Run the complete training pipeline
        results = main_training_pipeline_optimized()
        
        if results is not None:
            # Evaluate model performance
            evaluate_model_performance(results)
            
            # Save training artifacts
            save_training_artifacts(results)
            
            print("\n" + "="*80)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*80)
            print(f"Peak validation accuracy achieved: {results['peak_tracker'].peak_val_accuracy:.4f}")
            print(f"Model and artifacts saved for future use.")
            
            # Final cleanup
            memory_manager.force_cleanup(aggressive=True)
            
        else:
            print("Pipeline failed to complete.")
            
    except Exception as e:
        print(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()
        memory_manager.force_cleanup(aggressive=True)
    
    finally:
        print(f"\nFinal memory status:")
        memory_manager.check_memory_status("Pipeline End")

