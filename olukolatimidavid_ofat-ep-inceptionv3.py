!pip -q install optuna


import os
import gc
import cv2
import json
import pickle
import random
import optuna
import logging
import warnings
import pandas as pd
import numpy as np
import pydicom as dcm
from tqdm import tqdm
from enum import Enum
import seaborn as sns
import multiprocessing as mp
from datetime import datetime
import matplotlib.pyplot as plt
from collections import Counter
warnings.filterwarnings('ignore')
from dataclasses import dataclass, asdict, field
from sklearn.model_selection import train_test_split
from typing import Dict, List, Optional, Callable, Any, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score, confusion_matrix


try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    print("start method has already been set; ignore")
    pass

from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)  # hide backend INFO & WARNING


# Set up logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'        # only ERROR+FATAL from C++
os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'       # turn off cuDNN auto‑tune (optional)
os.environ['TF_DISABLE_XLA_SLOW_OP_ALARM'] = '1'
os.environ['ABSL_CPP_MIN_LOG_LEVEL'] = '2'
logger = logging.getLogger(__name__)


# Set seeds for reproducibility
SEED = 23
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
tf.random.set_seed(SEED)
tf.get_logger().setLevel(logging.ERROR)


# # Configure TensorFlow for memory optimization
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth for GPU
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU memory growth enabled for {len(gpus)} GPU(s)")
    except RuntimeError as e:
        print(e)


from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical


def clear_memory():
    """Comprehensive memory clearing function"""
    # Clear TensorFlow session
    tf.keras.backend.clear_session()
    
    # Force garbage collection
    gc.collect()
    
    # Clear CUDA cache if available
    if tf.config.list_physical_devices('GPU'):
        try:
            tf.keras.utils.clear_session()
        except:
            pass


class RSNADataPreprocessor:
    """
    Data preprocessing for RSNA Pneumonia Detection.
    Handles loading/resizing/normalizing images via a streaming tf.data pipeline.
    """
    def __init__(
        self,
        root_path: str,
        df: pd.DataFrame,
        image_size: Tuple[int, int] = (224, 224),
        target_column: str = 'Target',
        patient_id_column: str = 'patientId'
    ):
        self.root_path = root_path
        self.df = df.copy()
        self.image_size = image_size
        self.target_column = target_column
        self.patient_id_column = patient_id_column

    def _load_raw_image(self, patient_id: bytes) -> np.ndarray:
        """
        Read a single DICOM file (bytes→str) and return a uint8 H×W array.
        Normalizes in float32 then scales back to uint8.
        """
        if hasattr(patient_id, "numpy"):
            patient_id = patient_id.numpy()
        pid = patient_id.decode('utf-8')
        
        dicom_path = os.path.join(
            self.root_path, 'stage_2_train_images', f"{pid}"
        )
        try:
            ds = dcm.dcmread(dicom_path)
            img = ds.pixel_array.astype(np.float32)
            mn, mx = img.min(), img.max()
            img = (img - mn) / max(mx - mn, 1e-6)  # Min Max scaler to 0-1 range
            return img   
        except Exception as e:
            # fallback to black image
            print('Something is seriously wrong with the DICOM file!!!')
            print('\n'*5, e)
            return np.zeros(self.image_size, dtype=np.float32)

    def preprocess_image(self, raw_img: np.ndarray) -> np.ndarray:
        """
        Resize to self.image_size, convert to RGB if necessary, normalize to [0,1].
        """
        resized = cv2.resize(raw_img, self.image_size, interpolation=cv2.INTER_AREA)
        
        # Handle channel conversions
        if resized.ndim == 2:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[-1] == 1:
            resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
        elif resized.shape[-1] == 4:
            resized = cv2.cvtColor(resized, cv2.COLOR_RGBA2RGB)
        
        # Normalize to [0,1]
        return resized

    def _load_and_preprocess(self, pid, label):
        """
        TF wrapper: loads raw + preprocess_image → ([H,W,3], label)
        """
        raw = tf.py_function(
            func=self._load_raw_image, inp=[pid], Tout=tf.float32
        )
        raw.set_shape(self.image_size)
        
        img = tf.py_function(
            func=lambda x: self.preprocess_image(x.numpy()), inp=[raw], Tout=tf.float32
        )
        img.set_shape((*self.image_size, 3))
        return img, tf.cast(label, tf.int32)


    def prepare_dataset(
        self,
        batch_size: int = 32,
        validation_split: float = 0.15,
        test_split: float = 0.15,
        random_state: int = 42
        ) -> Dict[str, tf.data.Dataset]:
        """
        Builds train/val/test tf.data pipelines with stratified splits,
        and prints class distribution per split.
        """
        # 1) Extract IDs and labels
        ids = self.df[self.patient_id_column].astype(str).tolist()
        labels = self.df[self.target_column].tolist()
        total = len(labels)
    
        # 2) Split off (val+test) from train
        train_ids, rest_ids, train_labels, rest_labels = train_test_split(
            ids, labels,
            test_size=(validation_split + test_split),
            stratify=labels,
            random_state=random_state
        )
        # 3) Split rest → val / test
        val_ids, test_ids, val_labels, test_labels = train_test_split(
            rest_ids, rest_labels,
            test_size=(test_split / (validation_split + test_split)),
            stratify=rest_labels,
            random_state=random_state
        )
    
        # 4) Print class distributions
        print("Total samples:", total)
        print("Train split:", len(train_labels), "->", Counter(train_labels))
        print("Validation split:", len(val_labels), "->", Counter(val_labels))
        print("Test split:", len(test_labels), "->", Counter(test_labels))
    
        # 5) Build tf.data.Dataset for each
        def make_ds(id_list, lbl_list, do_shuffle):
            paths = [f"{pid}.dcm".encode() for pid in id_list]
            ds = tf.data.Dataset.from_tensor_slices((paths, lbl_list))
            if do_shuffle:
                ds = ds.shuffle(buffer_size=len(id_list), seed=random_state)
            ds = ds.map(self._load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
            ds = ds.batch(batch_size)
            return ds.prefetch(tf.data.AUTOTUNE)
    
        result = {
            "train_dataset": make_ds(train_ids, train_labels, do_shuffle=True),
            "val_dataset":   make_ds(val_ids,   val_labels,   do_shuffle=False),
            "test_dataset":  make_ds(test_ids,  test_labels,  do_shuffle=False),
            "input_shape":   (*self.image_size, 3),
            "num_classes":   int(len(np.unique(labels))),
        }
    
        # Cleanup
        del ids, labels
        del train_ids, train_labels, rest_ids, rest_labels
        del val_ids, val_labels, test_ids, test_labels
        gc.collect()
    
        return result


PATH = "/kaggle/input/rsna-pneumonia-detection-challenge"


import sys
dataset_path = '/kaggle/input/rsna-pneumonia-dataset'
sys.path.append(dataset_path)


df = pd.read_csv(f'{dataset_path}/train_class_df.csv')
print('Shape of the dataset:', df.shape)


# show sample of the dataset
df.sample()


# view the class distribution of the dataset
df.Target.value_counts()


preprocessor = RSNADataPreprocessor(
    root_path=PATH,
    df=df
)


# Prepare dataset
data_dict = preprocessor.prepare_dataset(
    batch_size=64,
    validation_split=0.15,
    test_split=0.15,
    random_state=SEED
)

train_dataset = data_dict['train_dataset']
val_dataset = data_dict['val_dataset']
test_dataset = data_dict['test_dataset']


# Custom F1 Score metric for Keras
def f1_score_metric(y_true, y_pred):
    """Custom F1 score metric for binary classification in Keras"""
    y_pred = tf.cast(y_pred > 0.5, tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    
    # Calculate precision and recall
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1 - y_pred))
    
    precision = tp / (tp + fp + tf.keras.backend.epsilon())
    recall = tp / (tp + fn + tf.keras.backend.epsilon())
    
    f1 = 2 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
    return f1


class OptimizerType:
    ADAM = "adam"
    SGD = "sgd"

class ArchitectureType:
    VGG16 = "vgg16"
    RESNET50 = "resnet50"
    INCEPTIONV3 = "inceptionv3"
    MOBILENETV2 = "mobilenetv2"

class SamplerType:
    RANDOM = "Random_Search"
    BAYESIAN = "Bayesian_Optimization"
    GRID = "Grid_Search"

class HyperparameterType:
    LEARNING_RATE = "learning_rate"
    BATCH_SIZE = "batch_size"
    DROPOUT_RATE = "dropout_rate"
    EPOCHS = "epochs"
    OPTIMIZER = "optimizer"
    ALL = "all"


@dataclass
class HyperparameterSpace:
    """Configuration class for hyperparameter search spaces"""
    learning_rate: List[float] = field(default_factory=lambda: [1e-5, 1e-4, 1e-3, 1e-2, 1e-1])
    batch_size: List[int] = field(default_factory=lambda: [16, 32, 64, 128, 256])
    dropout_rate: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5])
    epochs: List[int] = field(default_factory=lambda: [10, 20, 30, 40, 50])
    optimizers: List[str] = field(default_factory=lambda: [OptimizerType.ADAM, OptimizerType.SGD])

    
    
    def __post_init__(self):
        if self.optimizers is None:
            self.optimizers = [OptimizerType.ADAM, OptimizerType.SGD]
        if self.batch_size is None:
            self.batch_size = [16, 32, 64, 128, 256]
        if self.epochs is None:
            self.epochs = list(range(10, 51, 10))  

@dataclass
class DefaultHyperparameters:
    """Default hyperparameter values"""
    learning_rate: float = 0.0001
    batch_size: int = 64
    optimizer: str = OptimizerType.ADAM
    epochs: int = 20  
    dropout_rate: float = 0.2

@dataclass
class ExperimentResult:
    """Data class to store experiment results"""
    trial_number: int
    hyperparameters: Dict[str, Any]
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    test_metrics: Dict[str, float]
    history: Dict[str, List[float]]
    min_train_loss: float
    min_val_loss: float
    max_train_f1: float
    max_val_f1: float
    max_train_accuracy: float  
    max_val_accuracy: float   
    training_time: float
    final_epoch: int
    # store test confusion matrix
    test_confusion_matrix: List[List[int]]


def model_builder(
    input_shape: Tuple[int, int, int],
    params: Dict[str, Any],
    architecture: str = 'vgg16'
) -> keras.Model:
    """
    Build a binary-classification model with a pretrained backbone.
    """
    arch = architecture.lower()
    backbones = {
        'vgg16': keras.applications.VGG16,
        'resnet50': keras.applications.ResNet50,
        'inceptionv3': keras.applications.InceptionV3,
        'mobilenetv2': keras.applications.MobileNetV2,
    }
    
    if arch not in backbones:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    BaseModel = backbones[arch]
    
    # Create base model with memory optimization
    base = BaseModel(
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )
    base.trainable = False
    
    inputs = keras.Input(shape=input_shape)
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    
    dr = params.get('dropout_rate', 0.2)
    if not 0 <= dr <= 1: 
        raise ValueError("`dropout_rate` must be between 0 and 1.")
        
    x = layers.Dropout(dr)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    
    model = models.Model(inputs, outputs, name=f"{arch}_binary")
    return model


def create_batched_dataset(dataset, batch_size):
    """Create batched dataset from unbatched dataset with memory optimization"""
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)  # Reduced prefetch buffer or use AUTOTUNE


class CNNHyperparameterOptimizer:
    """Memory-optimized CNN hyperparameter optimizer using Optuna"""
    
    def __init__(self, train_dataset, val_dataset, test_dataset, 
                 hyperparameter_space: HyperparameterSpace,
                 architecture: str = ArchitectureType.VGG16,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 num_classes: int = 2,
                 results_dir: str = "hyperparameter_results",
                 default_hyperparameters: DefaultHyperparameters = None):
        
        self.train_dataset = train_dataset.unbatch()
        self.val_dataset = val_dataset.unbatch()
        self.test_dataset = test_dataset.unbatch()
        self.hyperparameter_space = hyperparameter_space
        self.architecture = architecture
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.results_dir = results_dir
        self.default_hyperparameters = default_hyperparameters or DefaultHyperparameters()
        
        # Create results directory
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Store experiment results count instead of all results to save memory
        self.experiment_count = 0
        
    def create_cnn_model(self, trial, optimize_param: str = HyperparameterType.ALL):
        """Create CNN model with hyperparameters from trial"""
        
        # Start with default values
        hyperparams = {
            'learning_rate': self.default_hyperparameters.learning_rate,
            'batch_size': self.default_hyperparameters.batch_size,
            'dropout_rate': self.default_hyperparameters.dropout_rate,
            'optimizer': self.default_hyperparameters.optimizer,
            'epochs': self.default_hyperparameters.epochs
        }
        
        # Override with trial suggestions based on what we're optimizing
        if optimize_param == HyperparameterType.ALL:
            hyperparams['learning_rate'] = trial.suggest_categorical(
                'learning_rate', 
                self.hyperparameter_space.learning_rate
            )
            hyperparams['batch_size'] = trial.suggest_categorical(
                'batch_size', 
                self.hyperparameter_space.batch_size
            )
            hyperparams['dropout_rate'] = trial.suggest_categorical(
                'dropout_rate',
                self.hyperparameter_space.dropout_rate
            )
            hyperparams['optimizer'] = trial.suggest_categorical(
                'optimizer', 
                self.hyperparameter_space.optimizers
            )
            hyperparams['epochs'] = trial.suggest_categorical(
                'epochs', 
                self.hyperparameter_space.epochs
            )
        elif optimize_param == HyperparameterType.LEARNING_RATE:
            hyperparams['learning_rate'] = trial.suggest_categorical(
                'learning_rate', 
                self.hyperparameter_space.learning_rate
            )
        elif optimize_param == HyperparameterType.BATCH_SIZE:
            hyperparams['batch_size'] = trial.suggest_categorical(
                'batch_size', 
                self.hyperparameter_space.batch_size
            )
        elif optimize_param == HyperparameterType.DROPOUT_RATE:
            hyperparams['dropout_rate'] = trial.suggest_categorical(
                'dropout_rate',
                self.hyperparameter_space.dropout_rate
            )
        elif optimize_param == HyperparameterType.OPTIMIZER:
            hyperparams['optimizer'] = trial.suggest_categorical(
                'optimizer', 
                self.hyperparameter_space.optimizers
            )
        elif optimize_param == HyperparameterType.EPOCHS:
            hyperparams['epochs'] = trial.suggest_categorical(
                'epochs', 
                self.hyperparameter_space.epochs
            )
        
        # Create model parameters
        model_params = {
            'dropout_rate': hyperparams['dropout_rate']
        }
        
        # Create model using model_builder
        model = model_builder(self.input_shape, model_params, self.architecture)
        
        # Configure optimizer
        if hyperparams['optimizer'] == OptimizerType.ADAM:
            optimizer = keras.optimizers.Adam(learning_rate=hyperparams['learning_rate'])
        elif hyperparams['optimizer'] == OptimizerType.SGD:
            optimizer = keras.optimizers.SGD(learning_rate=hyperparams['learning_rate'], momentum=0.9)
        
        # Compile model with F1 score metric
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=[f1_score_metric, 'accuracy']  # Added custom F1 score metric
        )
        
        return model, hyperparams
    
    def evaluate_model_on_dataset(self, model, dataset):
        """Evaluate model on dataset and return true labels and predictions"""
        
        y_true_list = []
        y_pred_list = []
        
        for batch_x, batch_y in dataset:
            # Get predictions
            predictions = model.predict(batch_x, verbose=0)
            
            # Store true labels and predictions
            y_true_list.append(batch_y.numpy())
            y_pred_list.append(predictions)
        
        # Concatenate all batches
        y_true = np.concatenate(y_true_list, axis=0)
        y_pred = np.concatenate(y_pred_list, axis=0)
        
        return y_true, y_pred
    
    # Modified calculate_metrics method to include confusion matrix
    def calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive metrics including AUC and confusion matrix"""
        
        # Convert predictions to proper format
        if len(y_pred.shape) > 1 and y_pred.shape[1] == 1:
            y_pred_classes = (y_pred > 0.5).astype(int).flatten()
            y_pred_probs = y_pred.flatten()
        else:
            y_pred_classes = (y_pred > 0.5).astype(int)
            y_pred_probs = y_pred
        
        if len(y_true.shape) > 1:
            y_true = y_true.flatten()
        
        y_true = y_true.astype(int)
        
        try:
            auc_score = float(roc_auc_score(y_true, y_pred_probs))
        except ValueError:
            auc_score = 0.5
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred_classes)
        
        metrics = {
            'accuracy': float(accuracy_score(y_true, y_pred_classes)),
            'precision': float(precision_score(y_true, y_pred_classes, average='binary', zero_division=0)),
            'recall': float(recall_score(y_true, y_pred_classes, average='binary', zero_division=0)),
            'f1_score': float(f1_score(y_true, y_pred_classes, average='binary', zero_division=0)),
            'auc': auc_score,
            'confusion_matrix': cm.tolist()  # Convert to list for JSON serialization
        }
        
        # Clean up temporary variables
        del y_pred_classes, y_pred_probs
        gc.collect()
        
        return metrics
    
    # Modified objective method to use F1 score for optimization and save test confusion matrix
    def objective(self, trial, optimize_param: str = HyperparameterType.ALL):
        """Memory-optimized objective function for Optuna optimization"""

        start_time = datetime.now()
        try:
            # Create model and get hyperparameters
            model, hyperparams = self.create_cnn_model(trial, optimize_param)
            
            # Create batched datasets with the chosen batch size
            train_batched = create_batched_dataset(self.train_dataset, hyperparams['batch_size'])
            val_batched = create_batched_dataset(self.val_dataset, hyperparams['batch_size'])
            test_batched = create_batched_dataset(self.test_dataset, hyperparams['batch_size'])
            
            # Train model
            history = model.fit(
                train_batched,
                validation_data=val_batched,
                epochs=hyperparams['epochs'],
                verbose=0
            )
            
            # Get training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate metrics on all datasets
            train_y_true, train_y_pred = self.evaluate_model_on_dataset(model, train_batched)
            val_y_true, val_y_pred = self.evaluate_model_on_dataset(model, val_batched)
            test_y_true, test_y_pred = self.evaluate_model_on_dataset(model, test_batched)
            
            train_metrics = self.calculate_metrics(train_y_true, train_y_pred)
            val_metrics = self.calculate_metrics(val_y_true, val_y_pred)
            test_metrics = self.calculate_metrics(test_y_true, test_y_pred)
            
            # Extract history statistics - include both F1 and accuracy
            min_train_loss = float(min(history.history['loss']))
            min_val_loss = float(min(history.history['val_loss']))
            max_train_f1 = float(max(history.history['f1_score_metric']))
            max_val_f1 = float(max(history.history['val_f1_score_metric']))
            max_train_accuracy = float(max(history.history['accuracy']))  # Added max train accuracy
            max_val_accuracy = float(max(history.history['val_accuracy']))  # Added max val accuracy
            final_epoch = len(history.history['loss'])
            
            # Convert history to JSON serializable format
            history_dict = {
                'loss': [float(v) for v in history.history['loss']],
                'val_loss': [float(v) for v in history.history['val_loss']],
                'f1_score_metric': [float(v) for v in history.history['f1_score_metric']],
                'val_f1_score_metric': [float(v) for v in history.history['val_f1_score_metric']],
                'accuracy': [float(v) for v in history.history['accuracy']],
                'val_accuracy': [float(v) for v in history.history['val_accuracy']]
            }
            
            # Store experiment result
            hyperparameters_with_config = hyperparams.copy()
            hyperparameters_with_config['architecture'] = self.architecture
            hyperparameters_with_config['freeze_base'] = True
            hyperparameters_with_config['optimized_parameter'] = optimize_param
            
            experiment_result = ExperimentResult(
                trial_number=trial.number,
                hyperparameters=hyperparameters_with_config,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                history=history_dict,
                min_train_loss=min_train_loss,
                min_val_loss=min_val_loss,
                max_train_f1=max_train_f1,
                max_val_f1=max_val_f1,
                max_train_accuracy=max_train_accuracy,  # Added max train accuracy
                max_val_accuracy=max_val_accuracy,      # Added max val accuracy
                training_time=training_time,
                final_epoch=final_epoch,
                # Only store test confusion matrix
                test_confusion_matrix=test_metrics['confusion_matrix']
            )
            
            # Save confusion matrix plot (only for test set)
            self.save_confusion_matrix_plot(experiment_result)
            
            # Save experiment result immediately
            self.save_experiment_result(experiment_result)
            self.experiment_count += 1
            
            # Get validation f1_score for optimization
            val_score = val_metrics['f1_score']
            
            # Comprehensive cleanup
            del (model, train_batched, val_batched, test_batched, history,
                 train_y_true, train_y_pred, val_y_true, val_y_pred, 
                 test_y_true, test_y_pred, experiment_result, history_dict)
            
            # Clear memory
            clear_memory()
            
            return val_score
            
        except Exception as e:
            print('*'*60, '\n'*5, f"Trial {trial.number} failed with error: {str(e)}", '*'*60, '\n'*5)
            clear_memory()
            return 0.0
    
    # Modified method to save only test confusion matrix plot
    def save_confusion_matrix_plot(self, result: ExperimentResult):
        """Save confusion matrix plot for test set only"""
        
        # Create confusion matrix directory
        cm_dir = os.path.join(self.current_results_dir, self.sampler_type, 'confusion_matrices')
        os.makedirs(cm_dir, exist_ok=True)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(result.test_confusion_matrix, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['No Pneumonia', 'Pneumonia'],
                   yticklabels=['No Pneumonia', 'Pneumonia'])
        plt.title(f'Test Set Confusion Matrix\nTrial {result.trial_number}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Create filename
        filename = f'confusion_matrix_trial_{result.trial_number:03d}_test.png'
        filepath = os.path.join(cm_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()  # Close figure to save memory
    
    # Modified save_experiment_result method to print both F1 and accuracy
    def save_experiment_result(self, result: ExperimentResult):
        """Save individual experiment result with memory optimization"""
        
        # Convert result to dictionary
        result_dict = asdict(result)
        
        # Add minimal experiment setup information
        result_dict['experiment_setup'] = {
            'architecture': self.architecture,
            'freeze_base': True,
            'input_shape': self.input_shape,
            'num_classes': self.num_classes,
            'optimized_parameter': result.hyperparameters.get('optimized_parameter', 'all')
        }
        
        # Create descriptive filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        arch = self.architecture
        opt = result.hyperparameters.get('optimizer', 'unknown')
        batch_size = result.hyperparameters.get('batch_size', 'unknown')
        lr = result.hyperparameters.get('learning_rate', 0)
        dropout = result.hyperparameters.get('dropout_rate', 0)
        epochs = result.hyperparameters.get('epochs', 0)
        optimized_param = result.hyperparameters.get('optimized_parameter', 'all')
        
        lr_str = f"{lr:.0e}" if lr > 0 else "0"
        
        filename = (f"experiment_trial_{result.trial_number:03d}_{arch}_{optimized_param}_{opt}_"
                   f"bs{batch_size}_lr{lr_str}_drop{dropout:.2f}_ep{epochs}_{timestamp}.json")
        
        # Create sampler-specific directory
        sampler_dir = os.path.join(self.current_results_dir, self.sampler_type)
        os.makedirs(sampler_dir, exist_ok=True)
        
        filepath = os.path.join(sampler_dir, filename)
        
        # Save as JSON
        with open(filepath, 'w') as f:
            json.dump(result_dict, f, indent=2, default=str)
        
        # Print concise trial information with both F1 and accuracy
        print(f"Trial {result.trial_number:3d}: "
              f"Val F1={result.val_metrics['f1_score']:.4f}, "
              f"Val Acc={result.val_metrics['accuracy']:.4f}, "
              f"Test F1={result.test_metrics['f1_score']:.4f}, "
              f"Test Acc={result.test_metrics['accuracy']:.4f}, "
              f"Time={result.training_time:.1f}s")

    def optimize(self, n_trials: int = 50, 
                 sampler_type: str = SamplerType.BAYESIAN,
                 optimize_param: str = HyperparameterType.ALL):
        """Run hyperparameter optimization"""

        self.sampler_type = sampler_type
        self.optimize_param = optimize_param
        
        # Create sampler-specific results directory
        param_suffix = f"_{optimize_param}" if optimize_param != HyperparameterType.ALL else ""
        self.current_results_dir = f"{self.results_dir}{param_suffix}"
        os.makedirs(self.current_results_dir, exist_ok=True)
        
        # Choose sampler
        if sampler_type == SamplerType.RANDOM:
            sampler = optuna.samplers.RandomSampler(seed=SEED)
        elif sampler_type == SamplerType.GRID:
            # build the search_space dict for GridSampler
            if optimize_param == HyperparameterType.ALL:
                search_space = {
                    'learning_rate': self.hyperparameter_space.learning_rate,
                    'batch_size':    self.hyperparameter_space.batch_size,
                    'dropout_rate':  self.hyperparameter_space.dropout_rate,
                    'optimizer':     self.hyperparameter_space.optimizers,
                    'epochs':        self.hyperparameter_space.epochs,
                }
            else:
                # only grid over the single parameter
                search_space = {
                    optimize_param: getattr(self.hyperparameter_space, optimize_param)
                }
        
            sampler = optuna.samplers.GridSampler(search_space)

        else:
            # Use TPESampler with MedianPruner for Bayesian optimization
            sampler = optuna.samplers.TPESampler(seed=SEED)
        
        # Create pruner for Bayesian optimization
        pruner = None
        if sampler_type == SamplerType.BAYESIAN:
            pruner = optuna.pruners.MedianPruner(
                n_startup_trials=3, # start pruning after 3 trials
                n_warmup_steps=5,  # allow 5 epochs before pruning
                interval_steps=2   # check pruning every 2 epochs
            )
        
        # Create study
        study = optuna.create_study(
            direction='maximize',
            sampler=sampler,
            pruner=pruner,
            study_name=f"CNN_Hyperparameter_Exploration_{optimize_param}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        
        # Run optimization
        print(f"Starting hyperparameter exploration with {n_trials} trials...")
        print(f"Architecture: {self.architecture}")
        print(f"Sampler: {sampler_type}")
        print(f"Optimizing parameter: {optimize_param}")
        if pruner:
            print(f"Pruner: {pruner.__class__.__name__}")
        print(f"Target metric: Validation Accuracy")
        print(f"Results will be saved to: {self.current_results_dir}")
        
        # Print default values being used
        print("\nDefault hyperparameter values:")
        print(f"  Learning rate: {self.default_hyperparameters.learning_rate}")
        print(f"  Batch size: {self.default_hyperparameters.batch_size}")
        print(f"  Dropout rate: {self.default_hyperparameters.dropout_rate}")
        print(f"  Optimizer: {self.default_hyperparameters.optimizer}")
        print(f"  Epochs: {self.default_hyperparameters.epochs}")
        
        if optimize_param != HyperparameterType.ALL:
            print(f"\nOnly '{optimize_param}' will be optimized, others will use default values.")
        
        print("-" * 100)
        
        # Create objective function with optimize_param
        def objective_with_param(trial):
            return self.objective(trial, optimize_param)
        
        study.optimize(objective_with_param, n_trials=n_trials)
        
        # Save complete results
        self.save_complete_results(study, optimize_param)
        
        return study
    
    # Modified save_complete_results method to show both F1 and accuracy in summary
    def save_complete_results(self, study, optimize_param):
        """Save complete optimization results with memory optimization"""
        
        # Create sampler-specific directory
        sampler_dir = os.path.join(self.current_results_dir, self.sampler_type)
        os.makedirs(sampler_dir, exist_ok=True)
        
        # Save study object
        study_filename = f"optuna_study_{optimize_param}.pkl"
        study_filepath = os.path.join(sampler_dir, study_filename)
        
        with open(study_filepath, 'wb') as f:
            pickle.dump(study, f)
        
        # Create summary with only essential data
        all_trials_data = []
        for trial in study.trials:
            if trial.value is not None:
                trial_data = {
                    'trial_number': trial.number,
                    'value': trial.value,
                    'params': trial.params,
                    'state': trial.state.name
                }
                all_trials_data.append(trial_data)
        
        # Save summary results
        summary_results = {
            'exploration_info': {
                'n_trials': len(study.trials),
                'n_complete_trials': len([t for t in study.trials if t.state.name == 'COMPLETE']),
                'n_pruned_trials': len([t for t in study.trials if t.state.name == 'PRUNED']),
                'optimization_direction': study.direction.name,
                'sampler': study.sampler.__class__.__name__,
                'target_metric': 'validation_f1_score',
                'optimized_parameter': optimize_param,
                'architecture': self.architecture,
                'experiment_count': self.experiment_count
            },
            'best_trial': {
                'number': study.best_trial.number,
                'value': study.best_trial.value,
                'params': study.best_trial.params
            } if study.best_trial else None,
            'all_trials': all_trials_data
        }
        
        summary_filename = f"hyperparameter_exploration_summary_{self.architecture}_{optimize_param}_{self.sampler_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        summary_filepath = os.path.join(sampler_dir, summary_filename)
        
        with open(summary_filepath, 'w') as f:
            json.dump(summary_results, f, indent=2)
        
        print("\n" + "=" * 80)
        print("HYPERPARAMETER EXPLORATION COMPLETED!")
        print("=" * 80)
        print(f"Total trials: {len(study.trials)}")
        print(f"Completed trials: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")
        if study.best_trial:
            print(f"Best trial: {study.best_trial.number}")
            print(f"Best validation F1 score: {study.best_trial.value:.4f}")
            print("Best hyperparameters:")
            for key, value in study.best_trial.params.items():
                print(f"  {key}: {value}")
        print(f"\nResults saved to: {sampler_dir}")
        print(f"Total experiments processed: {self.experiment_count}")
        
        return summary_results


import warnings
warnings.filterwarnings('ignore')


# learning_rate: List[float] =  [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
# batch_size: List[int] =  [16, 32, 64, 128, 256]
# dropout_rate: List[float] =  [0.1, 0.2, 0.3, 0.4, 0.5]
# epochs: List[int] = [10, 20, 30, 40, 50]
# optimizers: List[str] = [OptimizerType.ADAM, OptimizerType.SGD]


# Define hyperparameter space for exploration
hyperparameter_space = HyperparameterSpace(
    learning_rate=[1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
    batch_size=[16, 32, 64, 128, 256],
    dropout_rate=[0.1, 0.2, 0.3, 0.4, 0.5],
    epochs=[10, 20, 30, 40, 50],
    optimizers=[OptimizerType.ADAM, OptimizerType.SGD]
)

# Define custom default hyperparameters 
default_hyperparams = DefaultHyperparameters(
    learning_rate=0.0001,
    batch_size=64,
    optimizer=OptimizerType.ADAM,
    epochs=20,
    dropout_rate=0.2
)


SETTINGS = {
    "architecture": "INCEPTIONV3",
    "sampler_types": ["GRID"],# "BAYESIAN", RANDOM
    "hyperparameter_settings": {
        # "BATCH_SIZE": 10,
        # # "LEARNING_RATE": 30,
        # "DROPOUT_RATE": 5,
        # "OPTIMIZER": 30,
        "EPOCHS": 5,
        # "ALL": 60,
    },
    "input_shape": (224, 224, 3),
    "num_classes": 2,
}
SETTINGS["results_root"] = f"cnn_hyperparameter_exploration_{SETTINGS['architecture']}"


for sampler_name in SETTINGS["sampler_types"]:
    sampler = getattr(SamplerType,sampler_name)
    for hp_name, n_trials in SETTINGS["hyperparameter_settings"].items():
        hp_type = getattr(HyperparameterType,hp_name)
        # construct a descriptive results directory
        results_dir = f"{SETTINGS['results_root']}/{sampler_name.lower()}" # _{hp_name.lower()}
        
        optimizer = CNNHyperparameterOptimizer(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            hyperparameter_space=hyperparameter_space,
            architecture=getattr(ArchitectureType, SETTINGS["architecture"]),
            input_shape=SETTINGS["input_shape"],
            num_classes=SETTINGS["num_classes"],
            results_dir=results_dir,
            default_hyperparameters=default_hyperparams
        )
        
        print(f"Starting optimization: sampler={sampler_name}, optimize={hp_name}, trials={n_trials}")
        study = optimizer.optimize(
            n_trials=n_trials,
            sampler_type=sampler,
            optimize_param=hp_type
        )
        print(f"Finished: results in {results_dir}\n")










