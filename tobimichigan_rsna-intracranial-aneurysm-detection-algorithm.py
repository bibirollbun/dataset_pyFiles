"""
RSNA Intracranial Aneurysm Detection - Full Production Pipeline
Detects presence and location of intracranial aneurysms in multimodal imaging data
"""

import os
import gc
import sys
import json
import shutil
import warnings
import psutil
from pathlib import Path
from collections import defaultdict
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import pydicom
import cv2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve,
    confusion_matrix, classification_report, f1_score
)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks, optimizers
from tensorflow.keras.applications import EfficientNetB0, ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings('ignore')

# Configure TensorFlow for memory efficiency
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
            tf.config.experimental.set_virtual_device_configuration(
                gpu,
                [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)]
            )
    except RuntimeError as e:
        print(f"GPU configuration error: {e}")

# Global Constants
ID_COL = 'SeriesInstanceUID'
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',
]

DICOM_TAG_ALLOWLIST = [
    'BitsAllocated', 'BitsStored', 'Columns', 'FrameOfReferenceUID',
    'HighBit', 'ImageOrientationPatient', 'ImagePositionPatient',
    'InstanceNumber', 'Modality', 'PatientID', 'PhotometricInterpretation',
    'PixelRepresentation', 'PixelSpacing', 'PlanarConfiguration',
    'RescaleIntercept', 'RescaleSlope', 'RescaleType', 'Rows',
    'SOPClassUID', 'SOPInstanceUID', 'SamplesPerPixel',
    'SliceThickness', 'SpacingBetweenSlices', 'StudyInstanceUID',
    'TransferSyntaxUID',
]

BASE_PATH = '/kaggle/input/rsna-intracranial-aneurysm-detection'
WORKING_PATH = '/kaggle/working'
IMG_SIZE = 224
BATCH_SIZE = 8
EPOCHS = 15
MAX_SERIES_SAMPLE = 200
MEMORY_LIMIT_GB = 14


class MemoryMonitor:
    """Real-time memory monitoring and safety checks"""
    
    def __init__(self, limit_gb: float = MEMORY_LIMIT_GB):
        self.limit_bytes = limit_gb * 1024**3
        self.process = psutil.Process()
    
    def get_usage(self) -> Dict[str, float]:
        mem = self.process.memory_info()
        return {
            'used_gb': mem.rss / 1024**3,
            'percent': (mem.rss / self.limit_bytes) * 100
        }
    
    def is_safe(self, threshold: float = 0.85) -> bool:
        usage = self.get_usage()
        return usage['percent'] < (threshold * 100)
    
    def log_usage(self, stage: str = ""):
        usage = self.get_usage()
        print(f"[{stage}] Memory: {usage['used_gb']:.2f}GB ({usage['percent']:.1f}%)")
    
    def force_cleanup(self):
        """Aggressive memory cleanup"""
        gc.collect()
        if gpus:
            try:
                tf.keras.backend.clear_session()
            except:
                pass
        gc.collect()


def weighted_multilabel_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_weights: Optional[List[float]] = None,
) -> float:
    """Compute weighted AUC for multilabel classification"""
    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n_classes = y_true.shape[1]
    
    try:
        individual_aucs = roc_auc_score(y_true, y_scores, average=None)
    except ValueError:
        return 0.0
    
    if class_weights is None:
        weights_array = np.ones(n_classes)
    else:
        weights_array = np.asarray(class_weights)
    
    weights_array = weights_array / np.sum(weights_array)
    return float(np.sum(individual_aucs * weights_array))


def calculate_competition_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate the competition metric with special weighting"""
    weights = [1.0] * 13 + [13.0]  # Last column (Aneurysm Present) has weight 13
    return weighted_multilabel_auc(y_true, y_pred, class_weights=weights)


class DataLoader:
    """Memory-efficient data loading with chunking"""
    
    def __init__(self, base_path: str, monitor: MemoryMonitor):
        self.base_path = Path(base_path)
        self.monitor = monitor
        self.train_df = None
        self.localizers_df = None
        
    def load_metadata(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load training metadata with memory efficiency"""
        print("\n=== Loading Metadata ===")
        self.monitor.log_usage("Before metadata load")
        
        train_path = self.base_path / 'train.csv'
        localizers_path = self.base_path / 'train_localizers.csv'
        
        if train_path.exists():
            self.train_df = pd.read_csv(train_path)
            print(f"Loaded {len(self.train_df)} training records")
        else:
            print("WARNING: train.csv not found, creating dummy data")
            self.train_df = self._create_dummy_data()
        
        if localizers_path.exists():
            self.localizers_df = pd.read_csv(localizers_path)
            print(f"Loaded {len(self.localizers_df)} localizer records")
        else:
            print("WARNING: train_localizers.csv not found")
            self.localizers_df = pd.DataFrame()
        
        self.monitor.log_usage("After metadata load")
        return self.train_df, self.localizers_df
    
    def _create_dummy_data(self) -> pd.DataFrame:
        """Create synthetic data for testing"""
        n_samples = 500
        data = {
            'SeriesInstanceUID': [f'series_{i}' for i in range(n_samples)],
            'Modality': np.random.choice(['CTA', 'MRA', 'MRI'], n_samples),
            'PatientAge': np.random.randint(20, 80, n_samples),
            'PatientSex': np.random.choice(['M', 'F'], n_samples),
        }
        
        for col in LABEL_COLS:
            if col == 'Aneurysm Present':
                data[col] = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
            else:
                data[col] = np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
        
        return pd.DataFrame(data)
    
    def load_dicom_series(self, series_id: str, max_slices: int = 64) -> Optional[np.ndarray]:
        """Load DICOM series with memory constraints"""
        series_path = self.base_path / 'series' / series_id
        
        if not series_path.exists():
            return None
        
        dcm_files = sorted(list(series_path.glob('*.dcm')))[:max_slices]
        
        if not dcm_files:
            return None
        
        images = []
        for dcm_file in dcm_files:
            try:
                ds = pydicom.dcmread(str(dcm_file), force=True)
                img = ds.pixel_array
                
                # Normalize
                img = img.astype(np.float32)
                if img.max() > img.min():
                    img = (img - img.min()) / (img.max() - img.min())
                
                # Resize
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
            except Exception as e:
                continue
        
        if not images:
            return None
        
        # Stack and take middle slices
        volume = np.array(images)
        return volume
    
    def extract_features_from_series(self, series_id: str) -> Dict[str, Any]:
        """Extract features from a DICOM series"""
        features = {
            'series_id': series_id,
            'num_slices': 0,
            'mean_intensity': 0.0,
            'std_intensity': 0.0,
            'slice_thickness': 0.0,
            'pixel_spacing_x': 0.0,
            'pixel_spacing_y': 0.0,
        }
        
        volume = self.load_dicom_series(series_id)
        
        if volume is not None and len(volume) > 0:
            features['num_slices'] = len(volume)
            features['mean_intensity'] = float(np.mean(volume))
            features['std_intensity'] = float(np.std(volume))
        
        # Try to get DICOM metadata
        series_path = self.base_path / 'series' / series_id
        if series_path.exists():
            dcm_files = list(series_path.glob('*.dcm'))
            if dcm_files:
                try:
                    ds = pydicom.dcmread(str(dcm_files[0]), force=True)
                    if hasattr(ds, 'SliceThickness'):
                        features['slice_thickness'] = float(ds.SliceThickness)
                    if hasattr(ds, 'PixelSpacing'):
                        features['pixel_spacing_x'] = float(ds.PixelSpacing[0])
                        features['pixel_spacing_y'] = float(ds.PixelSpacing[1])
                except:
                    pass
        
        return features


class FeatureEngineering:
    """Comprehensive feature engineering pipeline"""
    
    def __init__(self, monitor: MemoryMonitor):
        self.monitor = monitor
        self.scaler = StandardScaler()
        self.le_modality = LabelEncoder()
        self.le_sex = LabelEncoder()
        
    def engineer_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Create advanced features from metadata"""
        print("\n=== Feature Engineering ===")
        self.monitor.log_usage("Before feature engineering")
        
        df = df.copy()
        
        # Handle missing values
        if 'PatientAge' in df.columns:
            df['PatientAge'] = df['PatientAge'].fillna(df['PatientAge'].median())
        
        # Encode categorical variables
        if 'Modality' in df.columns:
            if is_train:
                df['Modality_encoded'] = self.le_modality.fit_transform(df['Modality'].fillna('Unknown'))
            else:
                df['Modality_encoded'] = self.le_modality.transform(df['Modality'].fillna('Unknown'))
            
            # One-hot encode modality
            modality_dummies = pd.get_dummies(df['Modality'], prefix='modality')
            df = pd.concat([df, modality_dummies], axis=1)
        
        if 'PatientSex' in df.columns:
            if is_train:
                df['PatientSex_encoded'] = self.le_sex.fit_transform(df['PatientSex'].fillna('U'))
            else:
                df['PatientSex_encoded'] = self.le_sex.transform(df['PatientSex'].fillna('U'))
        
        # Age-based features
        if 'PatientAge' in df.columns:
            df['age_group'] = pd.cut(df['PatientAge'], bins=[0, 30, 50, 70, 100], labels=[0, 1, 2, 3])
            df['age_group'] = df['age_group'].astype(int)
            df['age_squared'] = df['PatientAge'] ** 2
            df['age_log'] = np.log1p(df['PatientAge'])
        
        # Location-based aggregations
        location_cols = [col for col in LABEL_COLS if col != 'Aneurysm Present']
        if all(col in df.columns for col in location_cols):
            df['total_aneurysm_locations'] = df[location_cols].sum(axis=1)
            df['left_aneurysms'] = df[[col for col in location_cols if 'Left' in col]].sum(axis=1)
            df['right_aneurysms'] = df[[col for col in location_cols if 'Right' in col]].sum(axis=1)
            df['anterior_aneurysms'] = df[[col for col in location_cols if 'Anterior' in col]].sum(axis=1)
            df['posterior_aneurysms'] = df[[col for col in location_cols if 'Posterior' in col]].sum(axis=1)
        
        self.monitor.log_usage("After feature engineering")
        gc.collect()
        
        return df
    
    def scale_features(self, X_train: np.ndarray, X_val: np.ndarray = None, 
                      X_test: np.ndarray = None) -> Tuple:
        """Scale numerical features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        results = [X_train_scaled]
        
        if X_val is not None:
            results.append(self.scaler.transform(X_val))
        
        if X_test is not None:
            results.append(self.scaler.transform(X_test))
        
        return tuple(results) if len(results) > 1 else results[0]


class ImageFeatureExtractor:
    """Extract features from medical images using CNN"""
    
    def __init__(self, monitor: MemoryMonitor):
        self.monitor = monitor
        self.model = None
        
    def build_feature_extractor(self):
        """Build a lightweight CNN for feature extraction"""
        base_model = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            pooling='avg'
        )
        base_model.trainable = False
        
        self.model = base_model
        return self.model
    
    def extract_features_batch(self, images: np.ndarray) -> np.ndarray:
        """Extract features from batch of images"""
        if self.model is None:
            self.build_feature_extractor()
        
        # Convert grayscale to RGB
        if len(images.shape) == 3:
            images = np.stack([images] * 3, axis=-1)
        elif images.shape[-1] == 1:
            images = np.repeat(images, 3, axis=-1)
        
        features = self.model.predict(images, batch_size=BATCH_SIZE, verbose=0)
        return features


class AneurysmDetectionModel:
    """Main model for aneurysm detection with ensemble approach"""
    
    def __init__(self, monitor: MemoryMonitor):
        self.monitor = monitor
        self.cnn_model = None
        self.rf_model = None
        self.gb_model = None
        self.history = None
        self.best_weights = None
        self.best_val_metric = 0.0
        
    def build_cnn_model(self, input_shape: Tuple[int, int, int], num_outputs: int = 14):
        """Build CNN model with attention mechanism"""
        inputs = layers.Input(shape=input_shape)
        
        # Efficient CNN backbone
        x = layers.Conv2D(32, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(64, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D(2)(x)
        
        x = layers.Conv2D(128, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        
        # Attention mechanism
        attention = layers.Conv2D(1, 1, activation='sigmoid')(x)
        x = layers.Multiply()([x, attention])
        
        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        
        outputs = layers.Dense(num_outputs, activation='sigmoid')(x)
        
        model = models.Model(inputs=inputs, outputs=outputs)
        
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        
        self.cnn_model = model
        return model
    
    def train_cnn(self, X_train: np.ndarray, y_train: np.ndarray,
                  X_val: np.ndarray, y_val: np.ndarray,
                  class_weights: Dict = None):
        """Train CNN with early stopping and checkpointing"""
        print("\n=== Training CNN Model ===")
        self.monitor.log_usage("Before CNN training")
        
        if len(X_train.shape) == 3:
            X_train = np.expand_dims(X_train, -1)
            X_val = np.expand_dims(X_val, -1)
        
        if self.cnn_model is None:
            self.build_cnn_model(X_train.shape[1:], y_train.shape[1])
        
        # Callbacks
        checkpoint_path = os.path.join(WORKING_PATH, 'best_cnn_model.h5')
        
        callbacks_list = [
            callbacks.EarlyStopping(
                monitor='val_auc',
                patience=5,
                mode='max',
                restore_best_weights=True,
                verbose=1
            ),
            callbacks.ModelCheckpoint(
                checkpoint_path,
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=1
            ),
            callbacks.ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=3,
                mode='max',
                verbose=1
            )
        ]
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            fill_mode='nearest'
        )
        
        self.history = self.cnn_model.fit(
            datagen.flow(X_train, y_train, batch_size=BATCH_SIZE),
            validation_data=(X_val, y_val),
            epochs=EPOCHS,
            callbacks=callbacks_list,
            class_weight=class_weights,
            verbose=1
        )
        
        self.monitor.log_usage("After CNN training")
        self.monitor.force_cleanup()
        
        return self.history
    
    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray):
        """Train ensemble of classical ML models"""
        print("\n=== Training Ensemble Models ===")
        self.monitor.log_usage("Before ensemble training")
        
        # Random Forest
        print("Training Random Forest...")
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=42
        )
        self.rf_model.fit(X_train, y_train)
        
        rf_val_pred = self.rf_model.predict_proba(X_val)
        rf_val_pred = np.array([pred[:, 1] for pred in rf_val_pred]).T
        rf_score = calculate_competition_metric(y_val, rf_val_pred)
        print(f"Random Forest validation score: {rf_score:.4f}")
        
        gc.collect()
        
        # Gradient Boosting
        print("Training Gradient Boosting...")
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # Train on main target first
        self.gb_model.fit(X_train, y_train[:, -1])
        
        gb_val_pred = self.gb_model.predict_proba(X_val)[:, 1]
        
        self.monitor.log_usage("After ensemble training")
        gc.collect()
        
        return rf_score
    
    def predict_ensemble(self, X: np.ndarray, X_img: np.ndarray = None) -> np.ndarray:
        """Generate ensemble predictions"""
        predictions = []
        weights = []
        
        # CNN predictions
        if self.cnn_model is not None and X_img is not None:
            if len(X_img.shape) == 3:
                X_img = np.expand_dims(X_img, -1)
            cnn_pred = self.cnn_model.predict(X_img, batch_size=BATCH_SIZE, verbose=0)
            predictions.append(cnn_pred)
            weights.append(0.5)
        
        # Random Forest predictions
        if self.rf_model is not None:
            rf_pred = self.rf_model.predict_proba(X)
            rf_pred = np.array([pred[:, 1] for pred in rf_pred]).T
            predictions.append(rf_pred)
            weights.append(0.3)
        
        # Gradient Boosting predictions (for main target)
        if self.gb_model is not None:
            gb_pred = self.gb_model.predict_proba(X)[:, 1]
            gb_pred_full = np.zeros((len(gb_pred), 14))
            gb_pred_full[:, -1] = gb_pred
            predictions.append(gb_pred_full)
            weights.append(0.2)
        
        if not predictions:
            return np.ones((len(X), 14)) * 0.5
        
        # Weighted average
        weights = np.array(weights) / sum(weights)
        ensemble_pred = sum(w * p for w, p in zip(weights, predictions))
        
        return ensemble_pred


class ModelEvaluator:
    """Comprehensive model evaluation and visualization"""
    
    def __init__(self, monitor: MemoryMonitor, save_path: str = WORKING_PATH):
        self.monitor = monitor
        self.save_path = Path(save_path)
        self.save_path.mkdir(exist_ok=True)
        
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray,
                      dataset_name: str = "Validation") -> Dict[str, float]:
        """Comprehensive model evaluation"""
        print(f"\n=== Evaluating on {dataset_name} Set ===")
        
        metrics = {}
        
        # Competition metric
        comp_score = calculate_competition_metric(y_true, y_pred)
        metrics['competition_metric'] = comp_score
        print(f"Competition Metric: {comp_score:.4f}")
        
        # Per-class AUC
        for idx, col in enumerate(LABEL_COLS):
            try:
                auc = roc_auc_score(y_true[:, idx], y_pred[:, idx])
                metrics[f'auc_{col}'] = auc
                print(f"{col}: AUC = {auc:.4f}")
            except:
                metrics[f'auc_{col}'] = 0.0
        
        # Overall metrics
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        try:
            overall_f1 = f1_score(y_true, y_pred_binary, average='weighted')
            metrics['f1_weighted'] = overall_f1
            print(f"Weighted F1 Score: {overall_f1:.4f}")
        except:
            metrics['f1_weighted'] = 0.0
        
        return metrics
    
    def plot_training_history(self, history):
        """Plot training history"""
        if history is None:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        axes[0, 0].plot(history.history['loss'], label='Train Loss')
        axes[0, 0].plot(history.history['val_loss'], label='Val Loss')
        axes[0, 0].set_title('Model Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy
        axes[0, 1].plot(history.history['accuracy'], label='Train Accuracy')
        axes[0, 1].plot(history.history['val_accuracy'], label='Val Accuracy')
        axes[0, 1].set_title('Model Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # AUC
        if 'auc' in history.history:
            axes[1, 0].plot(history.history['auc'], label='Train AUC')
            axes[1, 0].plot(history.history['val_auc'], label='Val AUC')
            axes[1, 0].set_title('Model AUC')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('AUC')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        plt.tight_layout()
        plt.show()
        plt.savefig(self.save_path / 'training_history.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved training history plot to {self.save_path / 'training_history.png'}")
    
    def plot_roc_curves(self, y_true: np.ndarray, y_pred: np.ndarray,
                       dataset_name: str = "Validation"):
        """Plot ROC curves for all classes"""
        n_classes = y_true.shape[1]
        n_cols = 4
        n_rows = (n_classes + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
        axes = axes.flatten()
        
        for idx, col in enumerate(LABEL_COLS):
            try:
                fpr, tpr, _ = roc_curve(y_true[:, idx], y_pred[:, idx])
                auc = roc_auc_score(y_true[:, idx], y_pred[:, idx])
                
                axes[idx].plot(fpr, tpr, label=f'AUC = {auc:.3f}')
                axes[idx].plot([0, 1], [0, 1], 'k--', label='Random')
                axes[idx].set_title(col, fontsize=10)
                axes[idx].set_xlabel('False Positive Rate')
                axes[idx].set_ylabel('True Positive Rate')
                axes[idx].legend()
                axes[idx].grid(True)
            except:
                axes[idx].text(0.5, 0.5, 'No valid data', ha='center', va='center')
                axes[idx].set_title(col, fontsize=10)
        
        # Hide empty subplots
        for idx in range(n_classes, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.show()
        plt.savefig(self.save_path / f'roc_curves_{dataset_name.lower()}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved ROC curves to {self.save_path / f'roc_curves_{dataset_name.lower()}.png'}")
    
    def plot_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray,
                               dataset_name: str = "Validation"):
        """Plot confusion matrices for main target"""
        y_pred_binary = (y_pred[:, -1] > 0.5).astype(int)
#************************************************************************************************

        y_true_binary = y_true[:, -1].astype(int)
        
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Aneurysm', 'Aneurysm'],
                   yticklabels=['No Aneurysm', 'Aneurysm'])
        plt.title(f'Confusion Matrix - {dataset_name} Set\nAneurysm Present')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.show()
        plt.savefig(self.save_path / f'confusion_matrix_{dataset_name.lower()}.png',
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix to {self.save_path / f'confusion_matrix_{dataset_name.lower()}.png'}")
        
        # Print classification report
        print("\nClassification Report (Aneurysm Present):")
        print(classification_report(y_true_binary, y_pred_binary, 
                                   target_names=['No Aneurysm', 'Aneurysm']))


class ProductionPipeline:
    """End-to-end production pipeline"""
    
    def __init__(self, base_path: str = BASE_PATH):
        self.base_path = base_path
        self.monitor = MemoryMonitor()
        self.data_loader = DataLoader(base_path, self.monitor)
        self.feature_eng = FeatureEngineering(self.monitor)
        self.image_extractor = ImageFeatureExtractor(self.monitor)
        self.model = AneurysmDetectionModel(self.monitor)
        self.evaluator = ModelEvaluator(self.monitor)
        
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
    def prepare_data(self, test_size: float = 0.2, val_size: float = 0.1):
        """Prepare training, validation, and test datasets"""
        print("\n" + "="*60)
        print("RSNA INTRACRANIAL ANEURYSM DETECTION PIPELINE")
        print("="*60)
        
        # Load metadata
        train_df, localizers_df = self.data_loader.load_metadata()
        
        # Feature engineering
        train_df = self.feature_eng.engineer_features(train_df, is_train=True)
        
        # Split data
        train_val_df, test_df = train_test_split(
            train_df, 
            test_size=test_size, 
            random_state=42,
            stratify=train_df['Aneurysm Present'] if 'Aneurysm Present' in train_df.columns else None
        )
        
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size / (1 - test_size),
            random_state=42,
            stratify=train_val_df['Aneurysm Present'] if 'Aneurysm Present' in train_val_df.columns else None
        )
        
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        
        print(f"\nDataset splits:")
        print(f"  Training: {len(train_df)} samples")
        print(f"  Validation: {len(val_df)} samples")
        print(f"  Test: {len(test_df)} samples")
        
        self.monitor.log_usage("After data preparation")
        
        return train_df, val_df, test_df
    
    def extract_features(self, df: pd.DataFrame, feature_cols: List[str]) -> np.ndarray:
        """Extract numerical features from dataframe"""
        available_cols = [col for col in feature_cols if col in df.columns]
        
        if not available_cols:
            # Return dummy features if no columns available
            return np.random.randn(len(df), 10)
        
        X = df[available_cols].fillna(0).values
        return X
    
    def prepare_image_data(self, df: pd.DataFrame, max_samples: int = None) -> np.ndarray:
        """Prepare image data for a dataset"""
        if max_samples is not None:
            df = df.head(max_samples)
        
        images = []
        valid_indices = []
        
        print(f"Loading images for {len(df)} samples...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            series_id = row[ID_COL]
            volume = self.data_loader.load_dicom_series(series_id, max_slices=32)
            
            if volume is not None and len(volume) > 0:
                # Take middle slice
                mid_idx = len(volume) // 2
                img = volume[mid_idx]
                images.append(img)
                valid_indices.append(idx)
            
            if not self.monitor.is_safe():
                print("Memory limit approaching, stopping image loading")
                break
        
        if not images:
            # Return dummy images if no real images loaded
            print("WARNING: No images loaded, using synthetic data")
            return np.random.randn(len(df), IMG_SIZE, IMG_SIZE)
        
        images = np.array(images)
        print(f"Loaded {len(images)} valid images")
        
        return images
    
    def run_training(self, use_images: bool = False):
        """Run the complete training pipeline"""
        print("\n=== Starting Training Pipeline ===")
        self.monitor.log_usage("Start of training")
        
        # Prepare data
        train_df, val_df, test_df = self.prepare_data()
        
        # Define feature columns
        feature_cols = []
        if 'PatientAge' in train_df.columns:
            feature_cols.extend(['PatientAge', 'age_squared', 'age_log', 'age_group'])
        if 'PatientSex_encoded' in train_df.columns:
            feature_cols.append('PatientSex_encoded')
        if 'Modality_encoded' in train_df.columns:
            feature_cols.append('Modality_encoded')
        
        # Add engineered features
        if 'total_aneurysm_locations' in train_df.columns:
            feature_cols.extend([
                'total_aneurysm_locations', 'left_aneurysms', 'right_aneurysms',
                'anterior_aneurysms', 'posterior_aneurysms'
            ])
        
        # Extract features
        X_train = self.extract_features(train_df, feature_cols)
        X_val = self.extract_features(val_df, feature_cols)
        X_test = self.extract_features(test_df, feature_cols)
        
        # Extract labels
        y_train = train_df[LABEL_COLS].values
        y_val = val_df[LABEL_COLS].values
        y_test = test_df[LABEL_COLS].values
        
        # Scale features
        X_train, X_val, X_test = self.feature_eng.scale_features(X_train, X_val, X_test)
        
        print(f"\nFeature shapes:")
        print(f"  X_train: {X_train.shape}")
        print(f"  y_train: {y_train.shape}")
        
        # Train ensemble models
        self.model.train_ensemble(X_train, y_train, X_val, y_val)
        
        # Train CNN if using images
        X_train_img, X_val_img, X_test_img = None, None, None
        if use_images:
            try:
                X_train_img = self.prepare_image_data(train_df, max_samples=MAX_SERIES_SAMPLE)
                X_val_img = self.prepare_image_data(val_df, max_samples=MAX_SERIES_SAMPLE // 2)
                
                # Align samples
                min_train = min(len(X_train), len(X_train_img))
                min_val = min(len(X_val), len(X_val_img))
                
                X_train = X_train[:min_train]
                y_train = y_train[:min_train]
                X_train_img = X_train_img[:min_train]
                
                X_val = X_val[:min_val]
                y_val = y_val[:min_val]
                X_val_img = X_val_img[:min_val]
                
                # Train CNN
                self.model.train_cnn(X_train_img, y_train, X_val_img, y_val)
                
                # Evaluate with images
                X_test_img = self.prepare_image_data(test_df, max_samples=MAX_SERIES_SAMPLE // 4)
                min_test = min(len(X_test), len(X_test_img))
                X_test = X_test[:min_test]
                y_test = y_test[:min_test]
                X_test_img = X_test_img[:min_test]
                
            except Exception as e:
                print(f"Error loading images: {e}")
                print("Continuing with tabular features only")
                use_images = False
        
        # Generate predictions
        print("\n=== Generating Predictions ===")
        
        val_pred = self.model.predict_ensemble(X_val, X_val_img)
        test_pred = self.model.predict_ensemble(X_test, X_test_img)
        
        # Evaluate
        val_metrics = self.evaluator.evaluate_model(y_val, val_pred, "Validation")
        test_metrics = self.evaluator.evaluate_model(y_test, test_pred, "Test")
        
        # Visualizations
        if self.model.history:
            self.evaluator.plot_training_history(self.model.history)
        
        self.evaluator.plot_roc_curves(y_val, val_pred, "Validation")
        self.evaluator.plot_roc_curves(y_test, test_pred, "Test")
        self.evaluator.plot_confusion_matrices(y_val, val_pred, "Validation")
        self.evaluator.plot_confusion_matrices(y_test, test_pred, "Test")
        
        # Save results
        self.save_results(val_metrics, test_metrics)
        
        self.monitor.log_usage("End of training")
        
        return val_metrics, test_metrics
    
    def save_results(self, val_metrics: Dict, test_metrics: Dict):
        """Save evaluation results to JSON"""
        results = {
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'model_config': {
                'img_size': IMG_SIZE,
                'batch_size': BATCH_SIZE,
                'epochs': EPOCHS,
                'max_series_sample': MAX_SERIES_SAMPLE
            }
        }
        
        results_path = Path(WORKING_PATH) / 'evaluation_results.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=4)
        
        print(f"\nSaved results to {results_path}")
    
    def create_submission(self, test_df: pd.DataFrame, predictions: np.ndarray, 
                         output_path: str = None):
        """Create submission file"""
        if output_path is None:
            output_path = Path(WORKING_PATH) / 'submission.csv'
        
        submission = pd.DataFrame({
            ID_COL: test_df[ID_COL].values
        })
        
        for idx, col in enumerate(LABEL_COLS):
            submission[col] = predictions[:, idx]
        
        submission.to_csv(output_path, index=False)
        print(f"\nSubmission file saved to {output_path}")
        
        return submission


def main():
    """Main execution function"""
    try:
        # Initialize pipeline
        pipeline = ProductionPipeline(BASE_PATH)
        
        # Run training
        val_metrics, test_metrics = pipeline.run_training(use_images=False)
        
        # Print final results
        print("\n" + "="*60)
        print("FINAL RESULTS")
        print("="*60)
        print(f"\nValidation Competition Metric: {val_metrics['competition_metric']:.4f}")
        print(f"Test Competition Metric: {test_metrics['competition_metric']:.4f}")
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("="*60)
        
    except Exception as e:
        print(f"\nERROR: Pipeline failed with exception:")
        print(f"{type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Final cleanup
        gc.collect()
        if gpus:
            try:
                tf.keras.backend.clear_session()
            except:
                pass


if __name__ == "__main__":
    main()




