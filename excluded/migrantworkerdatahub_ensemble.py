# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# INITIAL SETUP AND PACKAGE INSTALLATION
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DRW Crypto Market Prediction Pipeline - Setup and Global Configuration
This module provides the foundation for all prediction models with dynamic model discovery
"""

import subprocess
import sys
import os
import gc
import warnings
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime

warnings.filterwarnings('ignore')

# Install only essential base packages
print("Installing base packages...")
base_packages = ["pandas", "numpy", "scipy", "scikit-learn"]
subprocess.check_call([sys.executable, "-m", "pip", "install"] + base_packages + ["--quiet"])

# Import base packages after installation
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.base import clone, BaseEstimator, RegressorMixin
from sklearn.metrics import mean_absolute_error

# Memory management function
def aggressive_memory_cleanup():
    """Aggressively clean up memory to prevent kernel death"""
    # PyTorch cleanup
    if 'torch' in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except:
            pass
    
    # TensorFlow cleanup
    if 'tensorflow' in sys.modules:
        try:
            import tensorflow as tf
            tf.keras.backend.clear_session()
            if hasattr(tf.compat, 'v1'):
                tf.compat.v1.reset_default_graph()
        except:
            pass
    
    # Remove loaded modules to free memory
    modules_to_remove = ['xgboost', 'lightgbm', 'tensorflow', 'torch', 'sklearn', 'gplearn', 'catboost']
    for module in list(sys.modules.keys()):
        if any(module.startswith(mod) for mod in modules_to_remove):
            try:
                del sys.modules[module]
            except:
                pass
    
    # Multiple garbage collection passes
    for _ in range(3):
        gc.collect()

# Configure compute environment
def configure_compute_environment():
    """Configure compute environment for optimal performance"""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Default to CPU
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
    os.environ['OMP_NUM_THREADS'] = '4'
    os.environ['MKL_NUM_THREADS'] = '4'

@dataclass
class ModelOutput:
    """Represents an output file from a model"""
    file_path: str
    file_type: str  # 'submission', 'analysis', 'features', etc.
    created_at: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelRecord:
    """Complete record of a model's execution and outputs"""
    name: str
    directory: str
    status: str = "not_started"
    score: Optional[float] = None
    error_message: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    outputs: Dict[str, ModelOutput] = field(default_factory=dict)  # key: output_name, value: ModelOutput
    config: Dict[str, Any] = field(default_factory=dict)
    
    def add_output(self, output_name: str, file_path: str, file_type: str = "submission", metadata: Optional[Dict] = None):
        """Register an output file from this model"""
        self.outputs[output_name] = ModelOutput(
            file_path=file_path,
            file_type=file_type,
            created_at=datetime.now(),
            metadata=metadata or {}
        )
    
    def get_submission_files(self) -> List[str]:
        """Get all submission files from this model"""
        return [output.file_path for output in self.outputs.values() 
                if output.file_type == "submission" and os.path.exists(output.file_path)]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'name': self.name,
            'directory': self.directory,
            'status': self.status,
            'score': self.score,
            'error_message': self.error_message,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'outputs': {
                name: {
                    'file_path': output.file_path,
                    'file_type': output.file_type,
                    'created_at': output.created_at.isoformat(),
                    'metadata': output.metadata
                }
                for name, output in self.outputs.items()
            },
            'config': self.config
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ModelRecord':
        """Create from dictionary"""
        # Convert time fields
        if data.get('start_time'):
            data['start_time'] = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            data['end_time'] = datetime.fromisoformat(data['end_time'])
        
        # Convert outputs
        outputs = {}
        for name, output_data in data.get('outputs', {}).items():
            outputs[name] = ModelOutput(
                file_path=output_data['file_path'],
                file_type=output_data['file_type'],
                created_at=datetime.fromisoformat(output_data['created_at']),
                metadata=output_data.get('metadata', {})
            )
        data['outputs'] = outputs
        
        return cls(**data)

@dataclass
class GlobalConfig:
    """Central configuration with dynamic model registry"""
    # Base paths
    base_dir: str = "/kaggle/working/sub-models"
    train_path: str = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path: str = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path: str = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Model registry - using ModelRecord for complete tracking
    model_registry: Dict[str, ModelRecord] = field(default_factory=dict)
    
    # Pipeline metadata
    pipeline_start_time: Optional[datetime] = None
    pipeline_end_time: Optional[datetime] = None
    
    def __post_init__(self):
        """Initialize configuration"""
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)
        self.load_state()
    
    def register_model(self, name: str, directory: Optional[str] = None, config: Optional[Dict] = None) -> ModelRecord:
        """Register a new model in the pipeline"""
        if directory is None:
            directory = os.path.join(self.base_dir, name)
        
        # Create directory if it doesn't exist
        Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Create or update model record
        if name in self.model_registry:
            model_record = self.model_registry[name]
            model_record.directory = directory
            if config:
                model_record.config.update(config)
        else:
            model_record = ModelRecord(
                name=name,
                directory=directory,
                config=config or {}
            )
            self.model_registry[name] = model_record
        
        self.save_state()
        return model_record
    
    def register_model_output(self, model_name: str, output_name: str, file_path: str, 
                            file_type: str = "submission", metadata: Optional[Dict] = None):
        """Register an output file from a model"""
        if model_name not in self.model_registry:
            self.register_model(model_name)
        
        model_record = self.model_registry[model_name]
        model_record.add_output(output_name, file_path, file_type, metadata)
        self.save_state()
    
    def update_model_status(self, name: str, status: str, 
                           score: Optional[float] = None,
                           error_message: Optional[str] = None):
        """Update the status of a model"""
        if name not in self.model_registry:
            self.register_model(name)
        
        model = self.model_registry[name]
        model.status = status
        
        if status == "running":
            model.start_time = datetime.now()
        elif status in ["completed", "failed"]:
            model.end_time = datetime.now()
        
        if score is not None:
            model.score = score
        
        if error_message is not None:
            model.error_message = error_message
        
        self.save_state()
    
    def get_model_submissions(self, model_name: str) -> List[str]:
        """Get all submission files for a specific model"""
        if model_name not in self.model_registry:
            return []
        
        return self.model_registry[model_name].get_submission_files()
    
    def get_all_submissions(self) -> Dict[str, List[str]]:
        """Get all available submission files from all models"""
        submissions = {}
        for name, model in self.model_registry.items():
            if model.status == "completed":
                submission_files = model.get_submission_files()
                if submission_files:
                    submissions[name] = submission_files
        return submissions
    
    def get_latest_submission(self, model_name: str) -> Optional[str]:
        """Get the most recent submission file from a model"""
        submissions = self.get_model_submissions(model_name)
        if not submissions:
            return None
        
        # Get the output with the latest creation time
        model = self.model_registry[model_name]
        submission_outputs = [(name, output) for name, output in model.outputs.items() 
                            if output.file_type == "submission" and os.path.exists(output.file_path)]
        
        if not submission_outputs:
            return None
        
        # Sort by creation time and return the latest
        submission_outputs.sort(key=lambda x: x[1].created_at, reverse=True)
        return submission_outputs[0][1].file_path
    
    def get_model_summary(self) -> pd.DataFrame:
        """Get a summary of all models as a DataFrame"""
        data = []
        for name, model in self.model_registry.items():
            runtime = None
            if model.start_time and model.end_time:
                runtime = (model.end_time - model.start_time).total_seconds()
            
            submission_count = len(model.get_submission_files())
            
            data.append({
                'Model': name,
                'Status': model.status,
                'Score': model.score,
                'Runtime (s)': runtime,
                'Directory': model.directory,
                'Submissions': submission_count,
                'Total Outputs': len(model.outputs),
                'Error': model.error_message[:50] if model.error_message else None
            })
        
        return pd.DataFrame(data)
    
    def get_execution_summary(self) -> str:
        """Get a formatted summary of model execution status"""
        summary_df = self.get_model_summary()
        summary_lines = []
        
        if summary_df.empty:
            return "No models registered yet."
        
        # Status counts
        status_counts = summary_df['Status'].value_counts()
        summary_lines.append(f"Total models: {len(summary_df)}")
        
        for status in ['completed', 'failed', 'running', 'not_started']:
            count = status_counts.get(status, 0)
            if count > 0:
                emoji = {'completed': 'âœ…', 'failed': 'â�Œ', 'running': 'ğŸ”„', 'not_started': 'â�¸ï¸�'}[status]
                summary_lines.append(f"{emoji} {status}: {count}")
        
        # Completed models with outputs
        completed = summary_df[summary_df['Status'] == 'completed']
        if not completed.empty:
            summary_lines.append("\nCompleted models:")
            for _, row in completed.iterrows():
                score_str = f"score: {row['Score']:.4f}" if pd.notna(row['Score']) else "score: N/A"
                outputs_str = f"outputs: {row['Total Outputs']}"
                summary_lines.append(f"  âœ… {row['Model']} ({score_str}, {outputs_str})")
        
        # Failed models
        failed = summary_df[summary_df['Status'] == 'failed']
        if not failed.empty:
            summary_lines.append("\nFailed models:")
            for _, row in failed.iterrows():
                error_str = row['Error'] if pd.notna(row['Error']) else "Unknown error"
                summary_lines.append(f"  â�Œ {row['Model']}: {error_str}...")
        
        return "\n".join(summary_lines)
    
    def save_state(self):
        """Save current state to disk"""
        state = {
            'model_registry': {name: model.to_dict() for name, model in self.model_registry.items()},
            'pipeline_start_time': self.pipeline_start_time.isoformat() if self.pipeline_start_time else None,
            'pipeline_end_time': self.pipeline_end_time.isoformat() if self.pipeline_end_time else None
        }
        
        state_path = os.path.join(self.base_dir, 'pipeline_state.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2)
        
        # Save summary as CSV
        summary_path = os.path.join(self.base_dir, 'model_summary.csv')
        self.get_model_summary().to_csv(summary_path, index=False)
    
    def load_state(self):
        """Load state from disk if available"""
        state_path = os.path.join(self.base_dir, 'pipeline_state.json')
        
        if os.path.exists(state_path):
            try:
                with open(state_path, 'r') as f:
                    state = json.load(f)
                
                self.model_registry = {
                    name: ModelRecord.from_dict(data) 
                    for name, data in state.get('model_registry', {}).items()
                }
                
                if state.get('pipeline_start_time'):
                    self.pipeline_start_time = datetime.fromisoformat(state['pipeline_start_time'])
                if state.get('pipeline_end_time'):
                    self.pipeline_end_time = datetime.fromisoformat(state['pipeline_end_time'])
                    
            except Exception as e:
                print(f"Warning: Could not load previous state: {e}")
    
    def reset_pipeline(self):
        """Reset all models to initial state"""
        self.model_registry.clear()
        self.pipeline_start_time = None
        self.pipeline_end_time = None
        self.save_state()

# Data utility functions
def reduce_memory_usage(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """Reduce memory usage by optimizing data types"""
    if verbose:
        start_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory usage before: {start_mem:.2f} MB")
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    if verbose:
        end_mem = df.memory_usage().sum() / 1024**2
        print(f"Memory usage after: {end_mem:.2f} MB ({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)")
    
    return df

# Initialize environment
configure_compute_environment()

# Initialize global configuration
global_config = GlobalConfig()

print("ğŸš€ DRW Crypto Prediction Pipeline - Setup Complete")
print("="*80)
print(f"Base directory: {global_config.base_dir}")
print(f"Training data: {global_config.train_path}")
print(f"Test data: {global_config.test_path}")
print(f"\nModels will be automatically registered when they run.")
print("Each model should register its outputs using global_config.register_model_output()")

# Display execution status if any
if global_config.model_registry:
    print("\nPrevious execution status:")
    print(global_config.get_execution_summary())
else:
    print("\nNo previous execution history found - starting fresh")
print("="*80)


# XGBOOST THREE-MODEL PIPELINE IMPLEMENTATION
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DRW Crypto Market Prediction - XGBoost Three-Model Pipeline
This module implements a time-weighted ensemble of three XGBoost models
trained on different time windows to capture both long-term and recent patterns
"""

import subprocess
import sys
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
import gc
import warnings
import os
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Install required packages for this pipeline
print("Installing packages for XGBoost pipeline...")
packages_to_install = [
    'xgboost==2.0.3',
    'shap==0.44.0'
]

for package in packages_to_install:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# Import XGBoost after installation
import xgboost as xgb

class XGBoostConfiguration:
    """Configuration for XGBoost Three-Model Pipeline"""
    
    def __init__(self):
        # Model registration
        self.model_name = "xgboost"
        self.model_directory = os.path.join(global_config.base_dir, "triple_xgboost")
        
        # Register model with global configuration
        global_config.register_model(self.model_name, self.model_directory)
        
        # Data paths from global configuration
        self.train_path = global_config.train_path
        self.test_path = global_config.test_path
        self.sample_sub_path = global_config.sample_sub_path
        
        # Model-specific feature selection
        self.selected_features = [
            "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
            "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
            "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
        ]
        
        # XGBoost hyperparameters
        self.xgb_params = {
            "tree_method": "hist",  # Changed from gpu_hist for better compatibility
            "device": "cpu",
            "colsample_bylevel": 0.4778015829774066,
            "colsample_bynode": 0.362764358742407,
            "colsample_bytree": 0.7107423488010493,
            "gamma": 1.7094857725240398,
            "learning_rate": 0.02213323588455387,
            "max_depth": 20,
            "max_leaves": 12,
            "min_child_weight": 16,
            "n_estimators": 1667,
            "n_jobs": -1,
            "random_state": 42,
            "reg_alpha": 39.352415706891264,
            "reg_lambda": 75.44843704068275,
            "subsample": 0.06566669853471274,
            "verbosity": 0
        }
        
        # Model configurations for time windows
        self.model_configs = [
            {"name": "model_1_full_data", "percent": 1.00, "description": "Full Data"},
            {"name": "model_2_recent_75", "percent": 0.75, "description": "75% Recent"},
            {"name": "model_3_recent_50", "percent": 0.50, "description": "50% Recent"}
        ]
        
        # Cross-validation parameters
        self.n_folds = 5
        self.random_state = 42
        self.shuffle = True
        self.decay_factor = 0.95
        self.early_stopping_rounds = 25
        self.verbose_eval = 200
        
        # Output paths
        self.intermediate_dir = os.path.join(self.model_directory, "sub_models")
        self.submission_file = os.path.join(self.model_directory, "submission.csv")
        self.results_file = os.path.join(self.model_directory, "ensemble_results.csv")
        self.shap_features_path = os.path.join(self.model_directory, "shap_features.csv")
        
        # Ensure directories exist
        Path(self.model_directory).mkdir(parents=True, exist_ok=True)
        Path(self.intermediate_dir).mkdir(parents=True, exist_ok=True)

class XGBoostDataProcessor:
    """Data processing utilities for XGBoost pipeline"""
    
    @staticmethod
    def create_time_weights(n_samples: int, decay_factor: float = 0.95) -> np.ndarray:
        """Create exponential decay weights for time series data"""
        positions = np.arange(n_samples)
        normalized_positions = positions / (n_samples - 1)
        weights = decay_factor ** (1 - normalized_positions)
        weights = weights * n_samples / weights.sum()
        return weights
    
    @staticmethod
    def load_data(config: XGBoostConfiguration) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Load and prepare data for XGBoost models"""
        print("Loading data...")
        train = pd.read_parquet(config.train_path).reset_index(drop=True)
        test = pd.read_parquet(config.test_path).reset_index(drop=True)
        sample = pd.read_csv(config.sample_sub_path)
        
        # Verify feature availability
        available_features = [f for f in config.selected_features if f in train.columns]
        missing_features = set(config.selected_features) - set(available_features)
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features not found in data")
            print(f"Missing features: {missing_features}")
        
        # Select only available features
        train = train[available_features + ["label"]]
        test = test[available_features]
        
        # Reduce memory usage
        train = reduce_memory_usage(train, verbose=False)
        test = reduce_memory_usage(test, verbose=False)
        
        print(f"Data loaded - Train: {train.shape}, Test: {test.shape}")
        print(f"Using {len(available_features)} features")
        
        return train, test, sample

class XGBoostModelTrainer:
    """Model training utilities for XGBoost pipeline"""
    
    def __init__(self, config: XGBoostConfiguration):
        self.config = config
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series, 
                   X_valid: pd.DataFrame, y_valid: pd.Series, 
                   sample_weights: np.ndarray) -> xgb.XGBRegressor:
        """Train a single XGBoost model with early stopping"""
        model = xgb.XGBRegressor(**self.config.xgb_params)
        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,
            eval_set=[(X_valid, y_valid)],
            early_stopping_rounds=self.config.early_stopping_rounds,
            verbose=self.config.verbose_eval
        )
        return model
    
    def prepare_windowed_data(self, train_df: pd.DataFrame, train_idx: np.ndarray,
                            cutoff: int, features: List[str]) -> Tuple[pd.DataFrame, pd.Series, np.ndarray]:
        """Prepare training data for windowed models"""
        train_idx_recent = train_idx[train_idx >= cutoff]
        train_idx_recent_adjusted = train_idx_recent - cutoff
        
        train_recent = train_df.iloc[cutoff:].reset_index(drop=True)
        
        X_train = train_recent.iloc[train_idx_recent_adjusted][features]
        y_train = train_recent.iloc[train_idx_recent_adjusted]["label"]
        
        sample_weights_recent = XGBoostDataProcessor.create_time_weights(
            len(train_recent), self.config.decay_factor
        )
        train_weights = sample_weights_recent[train_idx_recent_adjusted]
        
        return X_train, y_train, train_weights

class XGBoostEnsembleBuilder:
    """Ensemble building and evaluation utilities"""
    
    def __init__(self, config: XGBoostConfiguration):
        self.config = config
    
    def evaluate_models(self, predictions: Dict[str, np.ndarray], 
                       train_labels: pd.Series) -> pd.DataFrame:
        """Evaluate individual models and ensemble combinations"""
        scores = {}
        
        # Individual model scores
        for model_name, preds in predictions.items():
            scores[model_name] = pearsonr(train_labels, preds)[0]
        
        # Simple ensemble
        simple_ensemble = np.mean(list(predictions.values()), axis=0)
        scores['simple_ensemble'] = pearsonr(train_labels, simple_ensemble)[0]
        
        # Weighted ensemble based on individual scores
        weights = np.array([scores[name] for name in predictions.keys()])
        weights = weights / weights.sum()
        
        weighted_ensemble = np.average(list(predictions.values()), axis=0, weights=weights)
        scores['weighted_ensemble'] = pearsonr(train_labels, weighted_ensemble)[0]
        
        # Create results dataframe
        results_data = []
        for i, (name, score) in enumerate(scores.items()):
            weight = weights[i] if i < len(weights) else np.nan
            results_data.append({
                'model': name,
                'pearson_correlation': score,
                'weight_in_final': weight
            })
        
        return pd.DataFrame(results_data)
    
    def create_final_ensemble(self, test_predictions: Dict[str, np.ndarray],
                            oof_scores: Dict[str, float]) -> np.ndarray:
        """Create final ensemble predictions"""
        # Use weighted ensemble if it performs better
        weights = np.array([oof_scores[name] for name in test_predictions.keys()])
        weights = weights / weights.sum()
        
        final_predictions = np.average(list(test_predictions.values()), axis=0, weights=weights)
        
        return final_predictions

class XGBoostThreeModelPipeline:
    """Main pipeline orchestrating the three-model XGBoost approach"""
    
    def __init__(self):
        self.config = XGBoostConfiguration()
        self.data_processor = XGBoostDataProcessor()
        self.trainer = XGBoostModelTrainer(self.config)
        self.ensemble_builder = XGBoostEnsembleBuilder(self.config)
    
    def run_cross_validation(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
        """Execute cross-validation for all three models"""
        features = [c for c in train_df.columns if c != "label"]
        
        # Calculate cutoffs for time windows
        cutoff_75 = int(len(train_df) * 0.25)
        cutoff_50 = int(len(train_df) * 0.50)
        
        print(f"\nTime window configurations:")
        print(f"  Model 1: Full data ({len(train_df):,} samples)")
        print(f"  Model 2: 75% recent ({len(train_df) - cutoff_75:,} samples)")
        print(f"  Model 3: 50% recent ({len(train_df) - cutoff_50:,} samples)")
        
        # Initialize prediction storage
        oof_predictions = {
            'model_1': np.zeros(len(train_df)),
            'model_2': np.zeros(len(train_df)),
            'model_3': np.zeros(len(train_df))
        }
        
        test_predictions = {
            'model_1': np.zeros(len(test_df)),
            'model_2': np.zeros(len(test_df)),
            'model_3': np.zeros(len(test_df))
        }
        
        # Create sample weights for full dataset
        sample_weights_full = self.data_processor.create_time_weights(
            len(train_df), self.config.decay_factor
        )
        
        # Cross-validation
        kf = KFold(n_splits=self.config.n_folds, shuffle=self.config.shuffle, 
                  random_state=self.config.random_state)
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df)):
            print(f"\nProcessing fold {fold + 1}/{self.config.n_folds}")
            
            X_valid = train_df.iloc[valid_idx][features]
            y_valid = train_df.iloc[valid_idx]["label"]
            X_test = test_df[features]
            
            # Model 1: Full data
            X_train = train_df.iloc[train_idx][features]
            y_train = train_df.iloc[train_idx]["label"]
            train_weights = sample_weights_full[train_idx]
            
            model1 = self.trainer.train_model(X_train, y_train, X_valid, y_valid, train_weights)
            oof_predictions['model_1'][valid_idx] = model1.predict(X_valid)
            test_predictions['model_1'] += model1.predict(X_test) / self.config.n_folds
            
            # Model 2: 75% recent
            X_train_75, y_train_75, weights_75 = self.trainer.prepare_windowed_data(
                train_df, train_idx, cutoff_75, features
            )
            
            if len(X_train_75) > 0:
                model2 = self.trainer.train_model(X_train_75, y_train_75, X_valid, y_valid, weights_75)
                
                # Handle predictions for validation set
                valid_idx_recent = valid_idx[valid_idx >= cutoff_75]
                if len(valid_idx_recent) > 0:
                    X_valid_recent = train_df.iloc[valid_idx_recent][features]
                    oof_predictions['model_2'][valid_idx_recent] = model2.predict(X_valid_recent)
                
                # Use model 1 predictions for samples before cutoff
                valid_idx_old = valid_idx[valid_idx < cutoff_75]
                if len(valid_idx_old) > 0:
                    oof_predictions['model_2'][valid_idx_old] = oof_predictions['model_1'][valid_idx_old]
                
                test_predictions['model_2'] += model2.predict(X_test) / self.config.n_folds
            
            # Model 3: 50% recent
            X_train_50, y_train_50, weights_50 = self.trainer.prepare_windowed_data(
                train_df, train_idx, cutoff_50, features
            )
            
            if len(X_train_50) > 0:
                model3 = self.trainer.train_model(X_train_50, y_train_50, X_valid, y_valid, weights_50)
                
                valid_idx_recent = valid_idx[valid_idx >= cutoff_50]
                if len(valid_idx_recent) > 0:
                    X_valid_recent = train_df.iloc[valid_idx_recent][features]
                    oof_predictions['model_3'][valid_idx_recent] = model3.predict(X_valid_recent)
                
                valid_idx_old = valid_idx[valid_idx < cutoff_50]
                if len(valid_idx_old) > 0:
                    oof_predictions['model_3'][valid_idx_old] = oof_predictions['model_1'][valid_idx_old]
                
                test_predictions['model_3'] += model3.predict(X_test) / self.config.n_folds
        
        return {
            'oof_predictions': oof_predictions,
            'test_predictions': test_predictions,
            'train_labels': train_df["label"]
        }
    
    def save_feature_importance(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        """Generate and save SHAP feature importance analysis"""
        try:
            import shap
            
            features = [c for c in train_df.columns if c != "label"]
            sample_weights = self.data_processor.create_time_weights(
                len(train_df), self.config.decay_factor
            )
            
            print("\nGenerating SHAP feature importance...")
            
            # Train model for SHAP analysis
            model = xgb.XGBRegressor(**self.config.xgb_params)
            model.fit(
                train_df[features], 
                train_df["label"],
                sample_weight=sample_weights,
                verbose=0
            )
            
            # Calculate SHAP values on a sample
            sample_size = min(1000, len(test_df))
            test_sample = test_df[features].iloc[:sample_size]
            
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(test_sample)
            
            # Create feature importance dataframe
            feature_importance = pd.DataFrame({
                'feature': features,
                'importance': np.abs(shap_values).mean(axis=0)
            }).sort_values('importance', ascending=False)
            
            feature_importance.to_csv(self.config.shap_features_path, index=False)
            print(f"Feature importance saved to {self.config.shap_features_path}")
            
            # Register SHAP features output with global configuration
            global_config.register_model_output(
                self.config.model_name,
                'shap_features',
                self.config.shap_features_path,
                'analysis',
                metadata={'top_features': feature_importance.head(10)['feature'].tolist()}
            )
            
            # Save top features
            print("\nTop 10 most important features:")
            for idx, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
            
        except Exception as e:
            print(f"Warning: Could not generate SHAP feature importance: {e}")
    
    def run(self) -> float:
        """Execute the complete XGBoost three-model pipeline"""
        print("\nStarting XGBoost Three-Model Pipeline")
        print("="*80)
        
        # Update model status
        global_config.update_model_status(self.config.model_name, "running")
        
        # Load data
        train_df, test_df, sample_df = self.data_processor.load_data(self.config)
        
        # Run cross-validation
        print("\nRunning cross-validation...")
        cv_results = self.run_cross_validation(train_df, test_df)
        
        # Evaluate models
        print("\nEvaluating model performance...")
        ensemble_results = self.ensemble_builder.evaluate_models(
            cv_results['oof_predictions'],
            cv_results['train_labels']
        )
        
        # Display results
        print("\nModel Performance Summary:")
        for _, row in ensemble_results.iterrows():
            if pd.notna(row['pearson_correlation']):
                print(f"  {row['model']}: {row['pearson_correlation']:.4f}")
        
        # Get best ensemble score
        best_score = ensemble_results['pearson_correlation'].max()
        best_model = ensemble_results.loc[
            ensemble_results['pearson_correlation'].idxmax(), 'model'
        ]
        
        print(f"\nBest performing approach: {best_model} (score: {best_score:.4f})")
        
        # Create final predictions
        oof_scores = {
            name: ensemble_results[ensemble_results['model'] == name]['pearson_correlation'].values[0]
            for name in cv_results['test_predictions'].keys()
        }
        
        final_predictions = self.ensemble_builder.create_final_ensemble(
            cv_results['test_predictions'],
            oof_scores
        )
        
        # Save submission
        sample_df["prediction"] = final_predictions
        sample_df.to_csv(self.config.submission_file, index=False)
        print(f"\nSubmission saved to {self.config.submission_file}")
        
        # Register submission with global configuration
        global_config.register_model_output(
            self.config.model_name,
            'submission',
            self.config.submission_file,
            'submission',
            metadata={
                'best_score': float(best_score),
                'best_model': best_model,
                'n_models': len(self.config.model_configs)
            }
        )
        
        # Save ensemble results
        ensemble_results.to_csv(self.config.results_file, index=False)
        
        # Register ensemble results with global configuration
        global_config.register_model_output(
            self.config.model_name,
            'ensemble_results',
            self.config.results_file,
            'analysis',
            metadata={
                'model_scores': ensemble_results.set_index('model')['pearson_correlation'].to_dict()
            }
        )
        
        # Generate feature importance
        self.save_feature_importance(train_df, test_df)
        
        print("\nXGBoost pipeline completed successfully")
        
        return best_score

# Main execution
if __name__ == "__main__":
    try:
        print("\nğŸ“Š Running XGBoost Three-Model Pipeline")
        print("-"*80)
        
        # Clean memory before starting
        aggressive_memory_cleanup()
        
        # Create and run pipeline
        pipeline = XGBoostThreeModelPipeline()
        final_score = pipeline.run()
        
        # Update global configuration with success
        global_config.update_model_status('xgboost', 'completed', score=final_score)
        print("\nâœ… XGBoost pipeline completed successfully")
        
    except Exception as e:
        # Update global configuration with failure
        error_msg = str(e)
        global_config.update_model_status('xgboost', 'failed', error_message=error_msg)
        print(f"\nâ�Œ XGBoost pipeline failed: {error_msg}")
        raise
        
    finally:
        # Clean up memory
        aggressive_memory_cleanup()
        
        # Display current execution status
        print("\n" + "="*80)
        print("Current Execution Status:")
        print(global_config.get_execution_summary())
        print("="*80)


# AUTOENCODER DEEP MLP PIPELINE IMPLEMENTATION
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DRW Crypto Market Prediction - AutoEncoder Deep MLP Pipeline
This module implements a deep learning approach using autoencoders for feature extraction
combined with a multi-layer perceptron with residual connections for prediction
"""

import subprocess
import sys
import os
import gc
import warnings
import json
import random
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# Install required packages for this pipeline
print("Installing packages for AutoEncoder Deep MLP pipeline...")
packages_to_install = [
    'torch==2.0.1',
    'torchvision==0.15.2',
    'tqdm==4.65.0'
]

# Special handling for PyTorch to ensure CPU-only installation
print("Installing PyTorch (CPU version)...")
subprocess.check_call([
    sys.executable, "-m", "pip", "install", 
    "torch==2.0.1", "torchvision==0.15.2",
    "--index-url", "https://download.pytorch.org/whl/cpu",
    "--quiet"
])

# Install remaining packages
subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm==4.65.0", "--quiet"])

@dataclass
class AutoEncoderConfiguration:
    """Configuration for AutoEncoder Deep MLP Pipeline"""
    
    # Model identification
    model_name: str = "autoencoder"
    model_directory: str = ""
    
    # Model-specific features
    feature_columns: List[str] = field(default_factory=list)
    
    # AutoEncoder architecture parameters
    encoding_size: int = 128
    hidden_size: int = 256
    dropout: float = 0.7
    ae_dropout: float = 0.3
    noise_std: float = 0.05
    num_blocks: int = 8
    
    # Training parameters
    num_epochs: int = 80
    batch_size: int = 4096
    learning_rate: float = 0.0001
    weight_decay: float = 5e-3
    patience: int = 10
    min_lr: float = 1e-6
    
    # Loss weights
    mse_weight: float = 0.25
    corr_weight: float = 0.6
    ae_weight: float = 0.15
    
    # Cross-validation parameters
    n_splits: int = 5
    max_train_size: int = 100_000_000
    gap: int = 1
    
    # Random seed
    seed: int = 42
    
    def __post_init__(self):
        """Initialize configuration and register models"""
        if not self.model_directory:
            self.model_directory = os.path.join(global_config.base_dir, "autoencoder_deepmlp")
        
        # Register both autoencoder variants with global configuration
        global_config.register_model("autoencoder_simple", self.model_directory)
        global_config.register_model("autoencoder_weighted", self.model_directory)
        
        # Set default features if not provided
        if not self.feature_columns:
            self.feature_columns = [
                "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
                "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
                "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume",
                "X188", "X207", "X219", "X233", "X245"
            ]
        
        # Output paths
        self.intermediate_dir = os.path.join(self.model_directory, "fold_models")
        self.simple_submission_path = os.path.join(self.model_directory, "ensemble_simple_submission.csv")
        self.weighted_submission_path = os.path.join(self.model_directory, "ensemble_weighted_submission.csv")
        self.config_path = os.path.join(self.model_directory, "config.json")
        self.performance_metrics_path = os.path.join(self.model_directory, "performance_metrics.json")
        
        # Ensure directories exist
        Path(self.model_directory).mkdir(parents=True, exist_ok=True)
        Path(self.intermediate_dir).mkdir(parents=True, exist_ok=True)
    
    def save(self):
        """Save configuration to JSON file"""
        config_dict = {k: v for k, v in self.__dict__.items() 
                      if not k.startswith('_') and k not in ['model_directory', 'intermediate_dir']}
        with open(self.config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)

class RandomSeedManager:
    """Manages random seeds across all libraries for reproducibility"""
    
    @staticmethod
    def set_all_seeds(seed: int):
        """Set random seeds for all relevant libraries"""
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        if 'torch' in sys.modules:
            import torch
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

class TorchEnvironmentManager:
    """Manages PyTorch environment configuration to avoid conflicts"""
    
    @staticmethod
    def configure_environment():
        """Configure PyTorch environment for CPU execution"""
        # Force CPU usage to avoid GPU/CUDA issues
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        os.environ['PYTORCH_NO_CUDA_MEMORY_CACHING'] = '1'
        
        # Disable Triton to avoid registration conflicts
        os.environ['TRITON_CACHE_DIR'] = f'/tmp/triton_cache_{os.getpid()}'
        os.environ['DISABLE_TRITON'] = '1'
        
        # Clean torch modules if already loaded
        torch_modules = [
            'torch', 'torchvision', 'torchaudio', 'triton',
            'torch.nn', 'torch.optim', 'torch.utils', 'torch.cuda'
        ]
        
        for module in list(sys.modules.keys()):
            if any(module.startswith(torch_mod) for torch_mod in torch_modules):
                try:
                    del sys.modules[module]
                except:
                    pass
        
        gc.collect()

class AutoEncoderDataProcessor:
    """Data processing utilities for AutoEncoder pipeline"""
    
    @staticmethod
    def load_and_prepare_data(config: AutoEncoderConfiguration) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
        """Load and prepare data for AutoEncoder training"""
        print("Loading data...")
        train_df = pd.read_parquet(global_config.train_path)
        test_df = pd.read_parquet(global_config.test_path)
        sample_submission = pd.read_csv(global_config.sample_sub_path)
        
        # Verify feature availability
        available_features = [col for col in config.feature_columns if col in train_df.columns]
        missing_features = set(config.feature_columns) - set(available_features)
        
        if missing_features:
            print(f"Warning: {len(missing_features)} features not found in data")
            print(f"Missing features: {missing_features}")
        
        # Select available features
        train_df = train_df[available_features + ['label']]
        test_df = test_df[available_features]
        
        # Optimize memory usage
        train_df = reduce_memory_usage(train_df, verbose=False)
        test_df = reduce_memory_usage(test_df, verbose=False)
        
        # Extract arrays
        X_full = train_df[available_features].values
        y_full = train_df['label'].values
        X_test = test_df[available_features].values
        
        # Handle invalid values
        X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
        
        print(f"Data loaded - Train: {X_full.shape}, Test: {X_test.shape}")
        print(f"Using {len(available_features)} features")
        
        return X_full, y_full, X_test, sample_submission

class AutoEncoderNeuralNetwork:
    """Neural network components for AutoEncoder architecture"""
    
    def __init__(self, config: AutoEncoderConfiguration):
        self.config = config
        self.device = None
        self.torch = None
        self.nn = None
        self.F = None
    
    def initialize_pytorch(self):
        """Initialize PyTorch modules after environment configuration"""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
            
            self.torch = torch
            self.nn = nn
            self.F = F
            self.device = torch.device('cpu')
            
            print(f"PyTorch version: {torch.__version__}")
            print(f"Using device: {self.device}")
            
            return True
            
        except Exception as e:
            print(f"Error initializing PyTorch: {e}")
            return False
    
    def create_autoencoder(self, input_size: int):
        """Create autoencoder neural network"""
        
        class Swish(self.nn.Module):
            def forward(self, x):
                return x * self.torch.sigmoid(x)
        
        class GaussianNoise(self.nn.Module):
            def __init__(self, std: float = 0.05):
                super().__init__()
                self.std = std
                
            def forward(self, x):
                if self.training:
                    noise = self.torch.randn_like(x) * self.std
                    return x + noise
                return x
        
        class AutoEncoder(self.nn.Module):
            def __init__(self, input_size: int, encoding_size: int, dropout: float):
                super().__init__()
                
                # Encoder layers
                self.encoder = self.nn.Sequential(
                    self.nn.Linear(input_size, input_size // 2),
                    self.nn.BatchNorm1d(input_size // 2),
                    Swish(),
                    self.nn.Dropout(dropout),
                    
                    self.nn.Linear(input_size // 2, input_size // 4),
                    self.nn.BatchNorm1d(input_size // 4),
                    Swish(),
                    self.nn.Dropout(dropout),
                    
                    self.nn.Linear(input_size // 4, encoding_size),
                    self.nn.BatchNorm1d(encoding_size),
                    Swish()
                )
                
                # Decoder layers
                self.decoder = self.nn.Sequential(
                    self.nn.Linear(encoding_size, input_size // 4),
                    self.nn.BatchNorm1d(input_size // 4),
                    Swish(),
                    self.nn.Dropout(dropout),
                    
                    self.nn.Linear(input_size // 4, input_size // 2),
                    self.nn.BatchNorm1d(input_size // 2),
                    Swish(),
                    self.nn.Dropout(dropout),
                    
                    self.nn.Linear(input_size // 2, input_size)
                )
                
            def forward(self, x):
                encoded = self.encoder(x)
                decoded = self.decoder(encoded)
                return encoded, decoded
        
        return AutoEncoder(input_size, self.config.encoding_size, self.config.ae_dropout)
    
    def create_full_model(self, input_size: int):
        """Create complete model with autoencoder and prediction head"""
        
        autoencoder = self.create_autoencoder(input_size)
        config = self.config
        
        class CryptoMLPWithAutoEncoder(self.nn.Module):
            def __init__(self):
                super().__init__()
                
                self.noise_layer = self.nn.Sequential()  # Will add GaussianNoise
                self.autoencoder = autoencoder
                
                combined_input_size = input_size + config.encoding_size
                self.input_bn = self.nn.BatchNorm1d(combined_input_size)
                
                # Initial block
                self.initial_block = self.nn.Sequential(
                    self.nn.Linear(combined_input_size, config.hidden_size),
                    self.nn.BatchNorm1d(config.hidden_size),
                    self.nn.ReLU(),
                    self.nn.Dropout(config.dropout),
                    
                    self.nn.Linear(config.hidden_size, config.hidden_size),
                    self.nn.BatchNorm1d(config.hidden_size),
                    self.nn.ReLU(),
                    self.nn.Dropout(config.dropout),
                    
                    self.nn.Linear(config.hidden_size, config.hidden_size),
                    self.nn.BatchNorm1d(config.hidden_size),
                    self.nn.ReLU()
                )
                
                # Residual blocks
                self.residual_blocks = self.nn.ModuleList()
                for _ in range(config.num_blocks):
                    block = self.nn.Sequential(
                        self.nn.Linear(config.hidden_size, config.hidden_size),
                        self.nn.BatchNorm1d(config.hidden_size),
                        self.nn.ReLU(),
                        self.nn.Dropout(config.dropout),
                        
                        self.nn.Linear(config.hidden_size, config.hidden_size),
                        self.nn.BatchNorm1d(config.hidden_size),
                        self.nn.ReLU()
                    )
                    self.residual_blocks.append(block)
                
                # Output layer
                self.output = self.nn.Linear(config.hidden_size, 1)
                
            def forward(self, x, return_ae_loss=False):
                # Add noise during training
                if self.training:
                    x = x + self.torch.randn_like(x) * config.noise_std
                
                # Autoencoder
                encoded, decoded = self.autoencoder(x)
                
                # Combine original and encoded features
                x_combined = self.torch.cat([x, encoded], dim=1)
                x_combined = self.input_bn(x_combined)
                
                # Initial transformation
                x_hidden = self.initial_block(x_combined)
                
                # Residual connections
                for block in self.residual_blocks:
                    x_residual = block(x_hidden)
                    x_hidden = x_hidden + x_residual
                
                # Output
                output = self.output(x_hidden)
                
                if return_ae_loss:
                    return output, decoded, x
                return output
        
        return CryptoMLPWithAutoEncoder()

class AutoEncoderTrainer:
    """Training logic for AutoEncoder models"""
    
    def __init__(self, config: AutoEncoderConfiguration, nn_builder: AutoEncoderNeuralNetwork):
        self.config = config
        self.nn_builder = nn_builder
    
    def train_fold(self, X_train: np.ndarray, y_train: np.ndarray,
                   X_val: np.ndarray, y_val: np.ndarray,
                   X_test: np.ndarray, fold_idx: int) -> Dict[str, Any]:
        """Train a single fold of the AutoEncoder model"""
        
        torch = self.nn_builder.torch
        nn = self.nn_builder.nn
        device = self.nn_builder.device
        
        # Create data loaders
        from torch.utils.data import Dataset, DataLoader
        
        class CryptoDataset(Dataset):
            def __init__(self, features: np.ndarray, labels: Optional[np.ndarray] = None):
                self.features = torch.FloatTensor(features)
                self.labels = torch.FloatTensor(labels) if labels is not None else None
                
            def __len__(self):
                return len(self.features)
                
            def __getitem__(self, idx):
                if self.labels is not None:
                    return self.features[idx], self.labels[idx]
                return self.features[idx]
        
        train_dataset = CryptoDataset(X_train, y_train)
        val_dataset = CryptoDataset(X_val, y_val)
        test_dataset = CryptoDataset(X_test)
        
        train_loader = DataLoader(train_dataset, batch_size=self.config.batch_size, 
                                shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=self.config.batch_size, 
                              shuffle=False, num_workers=0)
        test_loader = DataLoader(test_dataset, batch_size=self.config.batch_size, 
                               shuffle=False, num_workers=0)
        
        # Create model
        num_features = X_train.shape[1]
        model = self.nn_builder.create_full_model(num_features).to(device)
        
        # Initialize weights
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)
        
        model.apply(init_weights)
        
        # Loss function
        class CombinedLoss(nn.Module):
            def __init__(self):
                super().__init__()
                self.mse = nn.MSELoss()
                
            def forward(self, y_pred, y_true, decoded=None, original=None):
                # Main prediction loss
                mse_loss = self.mse(y_pred, y_true)
                
                # Correlation loss
                y_pred_centered = y_pred - y_pred.mean()
                y_true_centered = y_true - y_true.mean()
                
                correlation = torch.sum(y_pred_centered * y_true_centered) / (
                    torch.sqrt(torch.sum(y_pred_centered ** 2)) * 
                    torch.sqrt(torch.sum(y_true_centered ** 2)) + 1e-8
                )
                
                # Autoencoder reconstruction loss
                ae_loss = self.mse(decoded, original) if decoded is not None else 0
                
                total_loss = (self.config.mse_weight * mse_loss - 
                            self.config.corr_weight * correlation + 
                            self.config.ae_weight * ae_loss)
                
                return total_loss, correlation.item()
        
        criterion = CombinedLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate,
                                   weight_decay=self.config.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                                              patience=5, factor=0.5,
                                                              min_lr=self.config.min_lr)
        
        # Training loop
        best_val_corr = -float('inf')
        best_model_state = None
        early_stop_counter = 0
        
        from tqdm import tqdm
        
        for epoch in range(self.config.num_epochs):
            # Training
            model.train()
            train_losses = []
            
            for batch_features, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False):
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                
                outputs, decoded, original = model(batch_features, return_ae_loss=True)
                loss, _ = criterion(outputs, batch_labels, decoded, original)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    outputs = model(batch_features)
                    
                    val_predictions.extend(outputs.cpu().numpy().flatten())
                    val_targets.extend(batch_labels.numpy().flatten())
            
            # Calculate validation correlation
            val_corr = pearsonr(val_predictions, val_targets)[0]
            scheduler.step(val_corr)
            
            # Early stopping check
            if val_corr > best_val_corr:
                best_val_corr = val_corr
                best_model_state = model.state_dict().copy()
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.config.num_epochs} | Val Corr: {val_corr:.6f}")
            
            if early_stop_counter >= self.config.patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
        
        # Load best model and make predictions
        model.load_state_dict(best_model_state)
        model.eval()
        
        test_predictions = []
        with torch.no_grad():
            for batch_features in tqdm(test_loader, desc="Predicting", leave=False):
                batch_features = batch_features.to(device)
                outputs = model(batch_features)
                test_predictions.append(outputs.cpu().numpy())
        
        test_predictions = np.vstack(test_predictions).flatten()
        
        # Save model state
        model_path = os.path.join(self.config.intermediate_dir, f'fold_{fold_idx}_model.pt')
        torch.save(best_model_state, model_path)
        
        return {
            'fold_idx': fold_idx,
            'best_val_corr': best_val_corr,
            'test_predictions': test_predictions
        }

class AutoEncoderPipeline:
    """Main pipeline orchestrating the AutoEncoder Deep MLP approach"""
    
    def __init__(self, config: Optional[AutoEncoderConfiguration] = None):
        self.config = config or AutoEncoderConfiguration()
        self.data_processor = AutoEncoderDataProcessor()
    
    def create_fallback_predictions(self, X_full: np.ndarray, y_full: np.ndarray, 
                                  X_test: np.ndarray, sample_submission: pd.DataFrame) -> Dict[str, Any]:
        """Create predictions using Ridge regression as fallback"""
        print("\nUsing Ridge regression fallback...")
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_full)
        X_test_scaled = scaler.transform(X_test)
        
        model = Ridge(alpha=1.0, random_state=self.config.seed)
        model.fit(X_train_scaled, y_full)
        
        predictions = model.predict(X_test_scaled)
        
        # Estimate score
        from sklearn.model_selection import cross_val_score
        scores = cross_val_score(model, X_train_scaled, y_full, cv=3,
                               scoring=lambda est, X, y: pearsonr(est.predict(X), y)[0])
        final_score = np.mean(scores)
        
        return {
            'predictions': predictions,
            'score': final_score,
            'method': 'ridge_fallback'
        }
    
    def save_performance_metrics(self, all_results: List[Dict], weights: np.ndarray, final_score: float):
        """Save detailed performance metrics"""
        performance_data = {
            'final_score': float(final_score),
            'method': 'neural_network' if all_results else 'ridge_fallback',
            'n_folds': len(all_results),
            'fold_results': []
        }
        
        if all_results:
            for i, result in enumerate(all_results):
                performance_data['fold_results'].append({
                    'fold': result['fold_idx'],
                    'validation_correlation': float(result['best_val_corr']),
                    'weight': float(weights[i]) if i < len(weights) else 0.0
                })
            
            correlations = [r['best_val_corr'] for r in all_results]
            performance_data['statistics'] = {
                'mean_correlation': float(np.mean(correlations)),
                'std_correlation': float(np.std(correlations)),
                'min_correlation': float(np.min(correlations)),
                'max_correlation': float(np.max(correlations))
            }
        
        with open(self.config.performance_metrics_path, 'w') as f:
            json.dump(performance_data, f, indent=2)
        
        # Register performance metrics with global configuration
        global_config.register_model_output(
            'autoencoder_simple',
            'performance_metrics',
            self.config.performance_metrics_path,
            'analysis',
            metadata=performance_data
        )
    
    def run(self) -> float:
        """Execute the complete AutoEncoder pipeline"""
        print("\nStarting AutoEncoder Deep MLP Pipeline")
        print("="*80)
        
        # Update status for both model variants
        global_config.update_model_status('autoencoder_simple', 'running')
        global_config.update_model_status('autoencoder_weighted', 'running')
        
        # Configure PyTorch environment
        print("Configuring PyTorch environment...")
        TorchEnvironmentManager.configure_environment()
        
        # Load data
        X_full, y_full, X_test, sample_submission = self.data_processor.load_and_prepare_data(self.config)
        
        # Initialize neural network builder
        nn_builder = AutoEncoderNeuralNetwork(self.config)
        
        if not nn_builder.initialize_pytorch():
            print("PyTorch initialization failed, using fallback method")
            fallback_results = self.create_fallback_predictions(X_full, y_full, X_test, sample_submission)
            
            # Save fallback predictions
            submission = sample_submission.copy()
            submission.iloc[:, 1] = fallback_results['predictions']
            submission.to_csv(self.config.simple_submission_path, index=False)
            submission.to_csv(self.config.weighted_submission_path, index=False)
            
            # Register outputs with global configuration
            global_config.register_model_output(
                'autoencoder_simple',
                'ensemble_simple_submission',
                self.config.simple_submission_path,
                'submission',
                metadata={'method': 'ridge_fallback', 'score': fallback_results['score']}
            )
            
            global_config.register_model_output(
                'autoencoder_weighted',
                'ensemble_weighted_submission',
                self.config.weighted_submission_path,
                'submission',
                metadata={'method': 'ridge_fallback', 'score': fallback_results['score']}
            )
            
            return fallback_results['score']
        
        # Create trainer
        trainer = AutoEncoderTrainer(self.config, nn_builder)
        
        # Time series cross-validation
        tss = TimeSeriesSplit(n_splits=self.config.n_splits,
                            max_train_size=self.config.max_train_size,
                            gap=self.config.gap)
        
        all_results = []
        all_test_predictions = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tss.split(X_full)):
            print(f"\nTraining Fold {fold_idx + 1}/{self.config.n_splits}")
            
            RandomSeedManager.set_all_seeds(self.config.seed + fold_idx)
            
            X_train = X_full[train_idx]
            X_val = X_full[val_idx]
            y_train = y_full[train_idx]
            y_val = y_full[val_idx]
            
            print(f"  Train samples: {len(X_train):,}")
            print(f"  Val samples: {len(X_val):,}")
            
            try:
                fold_results = trainer.train_fold(X_train, y_train, X_val, y_val, X_test, fold_idx)
                all_results.append(fold_results)
                all_test_predictions.append(fold_results['test_predictions'])
                
                print(f"  Fold {fold_idx + 1} completed | Best Corr: {fold_results['best_val_corr']:.6f}")
                
            except Exception as e:
                print(f"  Error in fold {fold_idx + 1}: {e}")
                print("  Continuing with remaining folds...")
            
            # Clean up memory
            aggressive_memory_cleanup()
        
        if not all_results:
            print("All folds failed, using fallback method")
            fallback_results = self.create_fallback_predictions(X_full, y_full, X_test, sample_submission)
            
            submission = sample_submission.copy()
            submission.iloc[:, 1] = fallback_results['predictions']
            submission.to_csv(self.config.simple_submission_path, index=False)
            submission.to_csv(self.config.weighted_submission_path, index=False)
            
            # Register outputs with global configuration
            global_config.register_model_output(
                'autoencoder_simple',
                'ensemble_simple_submission',
                self.config.simple_submission_path,
                'submission',
                metadata={'method': 'ridge_fallback', 'score': fallback_results['score']}
            )
            
            global_config.register_model_output(
                'autoencoder_weighted',
                'ensemble_weighted_submission',
                self.config.weighted_submission_path,
                'submission',
                metadata={'method': 'ridge_fallback', 'score': fallback_results['score']}
            )
            
            return fallback_results['score']
        
        # Create ensemble predictions
        print("\nCreating ensemble predictions...")
        
        # Simple average ensemble
        ensemble_predictions = np.mean(all_test_predictions, axis=0)
        
        # Weighted ensemble based on validation scores
        weights = np.array([r['best_val_corr'] for r in all_results])
        weights = np.maximum(weights, 0)
        weights = weights / weights.sum()
        
        weighted_ensemble_predictions = np.average(all_test_predictions, axis=0, weights=weights)
        
        # Save predictions
        submission = sample_submission.copy()
        submission.iloc[:, 1] = ensemble_predictions
        submission.to_csv(self.config.simple_submission_path, index=False)
        
        # Register simple ensemble output
        global_config.register_model_output(
            'autoencoder_simple',
            'ensemble_simple_submission',
            self.config.simple_submission_path,
            'submission',
            metadata={
                'method': 'simple_average',
                'n_folds': len(all_results),
                'score': float(np.mean([r['best_val_corr'] for r in all_results]))
            }
        )
        
        submission = sample_submission.copy()
        submission.iloc[:, 1] = weighted_ensemble_predictions
        submission.to_csv(self.config.weighted_submission_path, index=False)
        
        # Register weighted ensemble output
        global_config.register_model_output(
            'autoencoder_weighted',
            'ensemble_weighted_submission',
            self.config.weighted_submission_path,
            'submission',
            metadata={
                'method': 'weighted_average',
                'n_folds': len(all_results),
                'weights': weights.tolist(),
                'score': float(np.mean([r['best_val_corr'] for r in all_results]))
            }
        )
        
        # Save configuration
        self.config.save()
        
        # Register configuration
        global_config.register_model_output(
            'autoencoder_simple',
            'config',
            self.config.config_path,
            'config',
            metadata={'encoding_size': self.config.encoding_size, 'hidden_size': self.config.hidden_size}
        )
        
        # Calculate final score
        correlations = [r['best_val_corr'] for r in all_results]
        final_score = np.mean(correlations)
        
        # Save performance metrics
        self.save_performance_metrics(all_results, weights, final_score)
        
        print(f"\nAutoEncoder Ensemble Statistics:")
        print(f"  Average Correlation: {final_score:.6f}")
        print(f"  Best Single Fold: {np.max(correlations):.6f}")
        print(f"  Fold Weights: {weights}")
        
        print("\nAutoEncoder pipeline completed successfully")
        
        return final_score

# Main execution
if __name__ == "__main__":
    try:
        print("\nğŸ§  Running AutoEncoder Deep MLP Pipeline")
        print("-"*80)
        
        # Clean memory before starting
        aggressive_memory_cleanup()
        
        # Create configuration
        config = AutoEncoderConfiguration(
            num_epochs=80,
            encoding_size=128,
            hidden_size=256,
            num_blocks=8,
            dropout=0.7,
            noise_std=0.05
        )
        
        # Create and run pipeline
        pipeline = AutoEncoderPipeline(config)
        final_score = pipeline.run()
        
        # Update global configuration with success
        global_config.update_model_status('autoencoder_simple', 'completed', score=final_score)
        global_config.update_model_status('autoencoder_weighted', 'completed', score=final_score)
        print("\nâœ… AutoEncoder pipeline completed successfully")
        
    except Exception as e:
        # Update global configuration with failure
        error_msg = str(e)
        global_config.update_model_status('autoencoder_simple', 'failed', error_message=error_msg)
        global_config.update_model_status('autoencoder_weighted', 'failed', error_message=error_msg)
        print(f"\nâ�Œ AutoEncoder pipeline failed: {error_msg}")
        raise
        
    finally:
        # Clean up memory
        aggressive_memory_cleanup()
        
        # Display current execution status
        print("\n" + "="*80)
        print("Current Execution Status:")
        print(global_config.get_execution_summary())
        print("="*80)


# MARKET MICROSTRUCTURE XGBOOST PIPELINE IMPLEMENTATION
# ===========================================================================================
# ===========================================================================================

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DRW Crypto Market Prediction - Market Microstructure XGBoost Pipeline
This module implements an XGBoost model with extensive market microstructure features
and optional Optuna hyperparameter optimization followed by Ridge ensemble stacking
"""

# ===========================================================================================
# PACKAGE INSTALLATION FOR MARKET MICROSTRUCTURE PIPELINE
# ===========================================================================================

import subprocess
import sys
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Install required packages for this pipeline
print("Installing packages for Market Microstructure XGBoost pipeline...")
packages_to_install = [
    'xgboost==2.0.3',      # Specific version for stability
    'optuna==3.5.0',       # Hyperparameter optimization
    'pandas',
    'numpy',
    'scipy',
    'scikit-learn',
    'matplotlib'
]

for package in packages_to_install:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# Import packages after installation
import xgboost as xgb
from xgboost import XGBRegressor
import optuna

class MarketMicrostructureConfiguration:
    """Configuration for Market Microstructure XGBoost Pipeline with model-specific settings"""
    def __init__(self):
        # Model identification
        self.model_name = "market_microstructure"
        self.model_directory = os.path.join(global_config.base_dir, "market_microstructure_xgboost")
        
        # Register model with global configuration
        global_config.register_model(self.model_name, self.model_directory)
        
        # Data paths from global configuration
        self.train_path = global_config.train_path
        self.test_path = global_config.test_path
        self.sample_sub_path = global_config.sample_sub_path
        
        # Model parameters
        self.target = "label"
        self.n_folds = 5
        self.seed = 42
        
        # Optuna optimization settings
        self.run_optuna = True
        self.n_optuna_trials = 250
        
        # XGBoost parameters (optimized for market microstructure features)
        self.xgb_params = {
            "tree_method": "hist",  # Changed from gpu_hist for better compatibility
            "device": "cpu",
            "colsample_bylevel": 0.7,
            "colsample_bynode": 0.7,
            "colsample_bytree": 0.7,
            "gamma": 1.5,
            "learning_rate": 0.02,
            "max_depth": 15,
            "max_leaves": 20,
            "min_child_weight": 10,
            "n_estimators": 1500,
            "n_jobs": -1,
            "random_state": 42,
            "reg_alpha": 30,
            "reg_lambda": 60,
            "subsample": 0.08,
            "verbosity": 0
        }
        
        # Output paths
        self.intermediate_dir = os.path.join(self.model_directory, "intermediate_predictions")
        self.submission_file = os.path.join(self.model_directory, "submission.csv")
        self.metrics_file = os.path.join(self.model_directory, "model_metrics.csv")
        self.feature_importance_file = os.path.join(self.model_directory, "feature_importance.csv")
        self.optuna_results_file = os.path.join(self.model_directory, "optuna_results.json")
        
        # Model-specific columns to drop (these are typically redundant or problematic features)
        self.cols_to_drop = [
            'X697', 'X698', 'X699', 'X700', 'X701', 'X702', 'X703', 'X704', 'X705', 'X706', 
            'X707', 'X708', 'X709', 'X710', 'X711', 'X712', 'X713', 'X714', 'X715', 'X716',
            'X717', 'X864', 'X867', 'X869', 'X870', 'X871', 'X872', 'X104', 'X110', 'X116',
            'X122', 'X128', 'X134', 'X140', 'X146', 'X152', 'X158', 'X164', 'X170', 'X176',
            'X182', 'X351', 'X357', 'X363', 'X369', 'X375', 'X381', 'X387', 'X393', 'X399',
            'X405', 'X411', 'X417', 'X423', 'X429'
        ]
        
        # Ensure directories exist
        Path(self.model_directory).mkdir(parents=True, exist_ok=True)
        Path(self.intermediate_dir).mkdir(parents=True, exist_ok=True)

class MarketMicrostructureDataProcessor:
    """Data processing utilities for Market Microstructure pipeline"""
    @staticmethod
    def reduce_mem_usage(dataframe: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
        """Reduce memory usage by downcasting numeric types"""
        print(f'Reducing memory usage for: {dataset_name}')
        initial_mem_usage = dataframe.memory_usage().sum() / 1024**2
        
        for col in dataframe.columns:
            col_type = dataframe[col].dtype
            
            if col_type != object:
                c_min = dataframe[col].min()
                c_max = dataframe[col].max()
                
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        dataframe[col] = dataframe[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        dataframe[col] = dataframe[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        dataframe[col] = dataframe[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        dataframe[col] = dataframe[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        dataframe[col] = dataframe[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        dataframe[col] = dataframe[col].astype(np.float32)
                    else:
                        dataframe[col] = dataframe[col].astype(np.float64)
        
        final_mem_usage = dataframe.memory_usage().sum() / 1024**2
        print(f'Memory usage reduced by {100 * (initial_mem_usage - final_mem_usage) / initial_mem_usage:.1f}%')
        
        return dataframe

class MarketMicrostructureFeatureEngineer:
    """Feature engineering for market microstructure analysis"""
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create extensive market microstructure features"""
        df = df.copy()
        
        # Check if required columns exist
        required_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < len(required_cols):
            print(f"Warning: Only {len(available_cols)} of {len(required_cols)} required columns available")
            print(f"Missing columns: {set(required_cols) - set(available_cols)}")
            
            # Create dummy columns if missing
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0
        
        # Interaction features
        df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
        df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
        df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
        df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
        df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
        df['buy_sell_interaction'] = df['buy_qty'] * df['sell_qty']
        
        # Spread indicators
        df['spread_indicator'] = (df['ask_qty'] - df['bid_qty']) / (df['ask_qty'] + df['bid_qty'] + 1e-8)
        
        # Volume-weighted features
        df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
        df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
        df['volume_weighted_bid'] = df['bid_qty'] * df['volume']
        df['volume_weighted_ask'] = df['ask_qty'] * df['volume']
        
        # Ratios
        df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
        df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
        
        # Order flow imbalance
        df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
        
        # Pressure indicators
        df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-8)
        df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
        
        # Liquidity features
        df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
        df['liquidity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_liquidity'] + 1e-8)
        df['relative_spread'] = (df['ask_qty'] - df['bid_qty']) / (df['volume'] + 1e-8)
        
        # Trade intensity
        df['trade_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-8)
        df['avg_trade_size'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
        df['net_trade_flow'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
        
        # Market depth
        df['depth_ratio'] = df['total_liquidity'] / (df['volume'] + 1e-8)
        df['volume_participation'] = (df['buy_qty'] + df['sell_qty']) / (df['total_liquidity'] + 1e-8)
        df['market_activity'] = df['volume'] * df['total_liquidity']
        
        # Spread proxy
        df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
        df['realized_volatility_proxy'] = np.abs(df['order_flow_imbalance']) * df['volume']
        
        # Normalized volumes
        df['normalized_buy_volume'] = df['buy_qty'] / (df['bid_qty'] + 1e-8)
        df['normalized_sell_volume'] = df['sell_qty'] / (df['ask_qty'] + 1e-8)
        
        # Advanced features
        df['liquidity_adjusted_imbalance'] = df['order_flow_imbalance'] * df['depth_ratio']
        df['pressure_spread_interaction'] = df['buying_pressure'] * df['spread_indicator']
        
        # Additional market microstructure indicators
        df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
        df['mid_price_proxy'] = (df['bid_qty'] + df['ask_qty']) / 2
        df['price_pressure'] = df['net_trade_flow'] * df['volume']
        df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-8)
        
        # Volatility and risk proxies
        df['volume_volatility'] = df['volume'] * df['spread_indicator']
        df['liquidity_risk'] = 1 / (df['total_liquidity'] + 1)
        df['execution_risk'] = df['spread_indicator'] * df['liquidity_risk']
        
        # Clean up infinities and NaNs
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        print(f"Created {len(df.columns)} features including engineered features")
        
        return df

class MarketMicrostructurePipeline:
    """Complete Market Microstructure XGBoost Pipeline"""
    def __init__(self, config: Optional[MarketMicrostructureConfiguration] = None):
        self.config = config or MarketMicrostructureConfiguration()
        self.data_processor = MarketMicrostructureDataProcessor()
        self.feature_engineer = MarketMicrostructureFeatureEngineer()
        
    def optimize_ridge_hyperparameters(self, X_ensemble: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize Ridge regression hyperparameters using Optuna"""
        print("Optimizing Ridge hyperparameters with Optuna...")
        
        def objective(trial):
            params = {
                "random_state": self.config.seed,
                "alpha": trial.suggest_float("alpha", 0.001, 100),
                "tol": trial.suggest_float("tol", 1e-6, 1e-2),
                "solver": trial.suggest_categorical("solver", ["auto", "svd", "cholesky", "lsqr"])
            }
            
            scores = []
            kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.seed)
            
            for train_idx, val_idx in kf.split(X_ensemble):
                X_train, X_val = X_ensemble.iloc[train_idx], X_ensemble.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                model = Ridge(**params)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                score = pearsonr(y_val, y_pred)[0]
                scores.append(score)
            
            return np.mean(scores)
        
        # Set Optuna logging level
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        
        sampler = optuna.samplers.TPESampler(seed=self.config.seed, multivariate=True)
        study = optuna.create_study(direction="maximize", sampler=sampler)
        study.optimize(objective, n_trials=self.config.n_optuna_trials, n_jobs=-1, catch=(ValueError,))
        
        best_params = study.best_params
        print(f"Best Ridge parameters: {best_params}")
        print(f"Best cross-validation score: {study.best_value:.6f}")
        
        ridge_params = {
            "random_state": self.config.seed,
            "alpha": best_params["alpha"],
            "tol": best_params["tol"],
            "solver": best_params.get("solver", "auto")
        }
        
        # Save Optuna results
        optuna_results = {
            "best_params": best_params,
            "best_value": study.best_value,
            "n_trials": len(study.trials),
            "optimization_history": [trial.value for trial in study.trials if trial.value is not None]
        }
        
        with open(self.config.optuna_results_file, 'w') as f:
            json.dump(optuna_results, f, indent=2)
        
        # Register Optuna results with global configuration
        global_config.register_model_output(
            self.config.model_name,
            'optuna_results',
            self.config.optuna_results_file,
            'analysis',
            metadata=optuna_results
        )
        
        return ridge_params
    
    def save_metrics(self, scores: Dict[str, float], ridge_params: Dict[str, Any]):
        """Save model metrics and parameters"""
        metrics = {
            "model": "Market Microstructure XGBoost",
            "xgboost_score": scores.get("XGBoost", 0),
            "ridge_ensemble_score": scores.get("ridge_ensemble", 0),
            "ridge_params": ridge_params,
            "n_features": scores.get("n_features", 0),
            "n_folds": self.config.n_folds
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.config.metrics_file, index=False)
        print(f"Metrics saved to {self.config.metrics_file}")
        
        # Register metrics with global configuration
        global_config.register_model_output(
            self.config.model_name,
            'model_metrics',
            self.config.metrics_file,
            'analysis',
            metadata=metrics
        )
    
    def run(self) -> Tuple[np.ndarray, Dict[str, float]]:
        print("\nStarting Market Microstructure XGBoost Pipeline...")
        print("This model focuses on market microstructure features and order flow dynamics")
        
        # Update model status
        global_config.update_model_status(self.config.model_name, "running")
        
        # Load data
        print("\nLoading data...")
        train = pd.read_parquet(self.config.train_path).reset_index(drop=True)
        test = pd.read_parquet(self.config.test_path).reset_index(drop=True)
        
        print(f"Original data shapes - Train: {train.shape}, Test: {test.shape}")
        
        # Drop unnecessary columns
        cols_to_drop_train = [col for col in self.config.cols_to_drop if col in train.columns]
        cols_to_drop_test = [col for col in self.config.cols_to_drop + ["label"] if col in test.columns]
        
        if cols_to_drop_train:
            train = train.drop(columns=cols_to_drop_train)
            print(f"Dropped {len(cols_to_drop_train)} columns from training data")
        
        if cols_to_drop_test:
            test = test.drop(columns=cols_to_drop_test)
            print(f"Dropped {len(cols_to_drop_test)} columns from test data")
        
        # Reduce memory usage
        train = self.data_processor.reduce_mem_usage(train, "train")
        test = self.data_processor.reduce_mem_usage(test, "test")
        
        # Create market microstructure features
        print("\nEngineering market microstructure features...")
        train = self.feature_engineer.create_features(train)
        test = self.feature_engineer.create_features(test)
        
        # Prepare data for modeling
        X = train.drop(self.config.target, axis=1)
        y = train[self.config.target]
        X_test = test
        
        # Clean infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        gc.collect()
        
        print(f"\nFinal data shapes - X: {X.shape}, X_test: {X_test.shape}")
        
        # Initialize storage
        scores = {"n_features": X.shape[1]}
        oof_preds = {}
        test_preds = {}
        feature_importance_list = []
        
        print("\nTraining XGBoost with Market Microstructure Features")
        print(f"Using {self.config.n_folds}-fold cross-validation")
        
        # Train XGBoost with cross-validation
        oof = np.zeros(len(X))
        test_predictions = np.zeros(len(X_test))
        fold_scores = []
        
        kf = KFold(n_splits=self.config.n_folds, shuffle=True, random_state=self.config.seed)
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\nFold {fold + 1}/{self.config.n_folds}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            print(f"  Train samples: {len(X_train):,}, Val samples: {len(X_val):,}")
            
            # Train model
            model = XGBRegressor(**self.config.xgb_params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=50,
                verbose=False
            )
            
            # Make predictions
            y_pred = model.predict(X_val)
            oof[val_idx] = y_pred
            
            # Calculate fold score
            score = pearsonr(y_val, y_pred)[0]
            fold_scores.append(score)
            print(f"  Fold {fold + 1} Score: {score:.6f}")
            
            # Test predictions
            test_predictions += model.predict(X_test) / self.config.n_folds
            
            # Store feature importance
            importance = model.feature_importances_
            feature_importance_list.append(importance)
            
            # Clean up memory
            del X_train, X_val, y_train, y_val, model
            gc.collect()
        
        # Calculate overall score
        overall_score = pearsonr(y, oof)[0]
        
        print(f"\nCross-validation completed:")
        print(f"  Average fold score: {np.mean(fold_scores):.6f}")
        print(f"  Overall OOF score: {overall_score:.6f}")
        
        # Store results
        oof_preds["XGBoost"] = oof
        test_preds["XGBoost"] = test_predictions
        scores["XGBoost"] = overall_score
        
        # Ridge ensemble stacking
        print("\nCreating Ridge Regression Ensemble")
        
        # Prepare ensemble data
        X_ensemble = pd.DataFrame(oof_preds)
        X_test_ensemble = pd.DataFrame(test_preds)
        
        # Optimize Ridge hyperparameters if enabled
        if self.config.run_optuna:
            ridge_params = self.optimize_ridge_hyperparameters(X_ensemble, y)
        else:
            ridge_params = {"random_state": self.config.seed, "alpha": 1.0}
            print("Using default Ridge parameters (Optuna disabled)")
        
        # Train Ridge ensemble with cross-validation
        print("\nTraining Ridge ensemble...")
        ridge_test_preds = np.zeros(len(X_test_ensemble))
        ridge_oof_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_ensemble)):
            X_train, X_val = X_ensemble.iloc[train_idx], X_ensemble.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = Ridge(**ridge_params)
            model.fit(X_train, y_train)
            
            # Validation score
            y_pred_val = model.predict(X_val)
            fold_score = pearsonr(y_val, y_pred_val)[0]
            ridge_oof_scores.append(fold_score)
            
            ridge_test_preds += model.predict(X_test_ensemble) / self.config.n_folds
        
        ridge_ensemble_score = np.mean(ridge_oof_scores)
        print(f"Ridge ensemble average validation score: {ridge_ensemble_score:.6f}")
        scores["ridge_ensemble"] = ridge_ensemble_score
        
        # Save results
        print("\nSaving results...")
        
        # Save submission
        sub = pd.read_csv(self.config.sample_sub_path)
        sub["prediction"] = ridge_test_preds
        sub.to_csv(self.config.submission_file, index=False)
        print(f"Submission saved to {self.config.submission_file}")
        
        # Register submission with global configuration
        global_config.register_model_output(
            self.config.model_name,
            'submission',
            self.config.submission_file,
            'submission',
            metadata={
                'xgboost_score': float(overall_score),
                'ridge_ensemble_score': float(ridge_ensemble_score),
                'n_features': X.shape[1],
                'method': 'ridge_stacked_xgboost'
            }
        )
        
        # Save metrics
        self.save_metrics(scores, ridge_params)
        
        # Save feature importance
        avg_importance = np.mean(feature_importance_list, axis=0)
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': avg_importance
        }).sort_values('importance', ascending=False)
        
        feature_importance.to_csv(self.config.feature_importance_file, index=False)
        
        # Register feature importance with global configuration
        global_config.register_model_output(
            self.config.model_name,
            'feature_importance',
            self.config.feature_importance_file,
            'analysis',
            metadata={
                'top_10_features': feature_importance.head(10)['feature'].tolist(),
                'top_10_importances': feature_importance.head(10)['importance'].tolist()
            }
        )
        
        print("\nMarket Microstructure pipeline completed successfully!")
        print(f"Final model score: {overall_score:.6f}")
        
        return ridge_test_preds, scores

# ===========================================================================================
# EXECUTION BLOCK
# ===========================================================================================

if __name__ == "__main__":
    try:
        print("\nğŸ�›ï¸� Running Market Microstructure XGBoost Pipeline")
        print("-"*80)
        
        # Clean memory before starting
        aggressive_memory_cleanup()
        
        # Create and run pipeline
        market_pipeline = MarketMicrostructurePipeline()
        predictions, scores = market_pipeline.run()
        
        # Extract final score (using XGBoost OOF score as primary metric)
        final_score = scores.get("XGBoost", 0.0)
        
        # Update global configuration with success
        # FIXED: Use market_pipeline.config.model_name instead of self.config.model_name
        global_config.update_model_status(market_pipeline.config.model_name, 'completed', score=final_score)
        print("\nâœ… Market Microstructure pipeline completed successfully")
        
    except Exception as e:
        # Update global configuration with failure
        error_msg = str(e)
        global_config.update_model_status('market_microstructure', 'failed', error_message=error_msg)
        print(f"\nâ�Œ Market Microstructure pipeline failed: {error_msg}")
        raise
        
    finally:
        # Clean up memory
        aggressive_memory_cleanup()
        
        # Display current execution status
        print("\n" + "="*80)
        print("Current Execution Status:")
        print(global_config.get_execution_summary())
        print("="*80)


# LIGHTGBM VOTING ENSEMBLE PIPELINE IMPLEMENTATION

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DRW Crypto Market Prediction - LightGBM Voting Ensemble Pipeline
This module implements a voting ensemble of LightGBM models trained on different
time-based folds with exponential time decay weighting for cryptocurrency prediction
"""

# ===========================================================================================
# PACKAGE INSTALLATION FOR LIGHTGBM VOTING PIPELINE
# ===========================================================================================

import subprocess
import sys
import os
import gc
import warnings
warnings.filterwarnings('ignore')

# Install required packages for this pipeline
print("Installing packages for LightGBM Voting Ensemble pipeline...")
packages_to_install = [
    'lightgbm==4.1.0',  # Specific version for stability
    'pandas',
    'numpy',
    'scipy',
    'scikit-learn',
    'matplotlib'
]

for package in packages_to_install:
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# ===========================================================================================
# ===========================================================================================

import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import json
from sklearn.base import BaseEstimator, RegressorMixin
import matplotlib.pyplot as plt

# Import LightGBM after installation
import lightgbm as lgb

class VotingModel(BaseEstimator, RegressorMixin):
    """Voting ensemble model that averages predictions from multiple estimators"""
    def __init__(self, estimators):
        super().__init__()
        self.estimators = estimators

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        y_preds = [estimator.predict(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

    def predict_proba(self, X):
        y_preds = [estimator.predict_proba(X) for estimator in self.estimators]
        return np.mean(y_preds, axis=0)

class LightGBMVotingConfiguration:
    """Configuration for LightGBM Voting Pipeline with model-specific settings"""
    def __init__(self):
        # Model identification
        self.model_name = "lightgbm_voting"
        self.model_directory = os.path.join(global_config.base_dir, "lightgbm_voting")
        
        # Register model with global configuration
        global_config.register_model(self.model_name, self.model_directory)
        
        # Data paths from global configuration
        self.train_path = global_config.train_path
        self.test_path = global_config.test_path
        self.sample_sub_path = global_config.sample_sub_path
        
        # Model-specific feature selection
        self.feature_names = [
            'X863', 'X856', 'X344', 'X598', 'X862', 'X385', 'X852', 'X603', 'X860', 'X674',
            'X415', 'X345', 'X137', 'X855', 'X174', 'X302', 'X178', 'X532', 'X168', 'X612',
            'bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume',
            'bid_ask_interaction', 'bid_buy_interaction', 'bid_sell_interaction',
            'ask_buy_interaction', 'ask_sell_interaction'
        ]
        
        # LightGBM parameters (optimized for financial time series)
        self.lgb_params = {
            "boosting_type": "gbdt",
            "objective": "regression",
            "metric": "mae",
            "colsample_bytree": 0.55,
            "learning_rate": 0.021,
            "min_child_samples": 32,
            "min_child_weight": 0.15,
            "max_depth": -1,
            "n_jobs": -1,
            "num_leaves": 64,
            "random_state": 42,
            "reg_alpha": 80,
            "reg_lambda": 100,
            "subsample": 0.85,
            "verbosity": 1,
            "device": "cpu"  # Changed from "gpu" for better compatibility
        }
        
        # Training parameters
        self.decay_factor = 0.95
        self.num_boost_round = 150
        self.n_folds = 5
        
        # Time-based fold date ranges
        self.fold_dates = {
            1: ('2023-03-01 00:00:00', '2023-05-01 00:00:00'),
            2: ('2023-05-01 00:00:00', '2023-07-01 00:00:00'),
            3: ('2023-07-01 00:00:00', '2023-09-01 00:00:00'),
            4: ('2023-09-01 00:00:00', '2023-11-01 00:00:00'),
            5: ('2023-11-01 00:00:00', '2024-01-01 00:00:00'),
            6: ('2024-01-01 00:00:00', '2024-03-01 00:00:00')
        }
        
        # Output paths
        self.intermediate_dir = os.path.join(self.model_directory, "fold_models")
        self.submission_file = os.path.join(self.model_directory, "submission.csv")
        self.metrics_file = os.path.join(self.model_directory, "model_metrics.csv")
        self.feature_importance_file = os.path.join(self.model_directory, "feature_importance.csv")
        self.fold_performance_file = os.path.join(self.model_directory, "fold_performance.json")
        
        # Ensure directories exist
        Path(self.model_directory).mkdir(parents=True, exist_ok=True)
        Path(self.intermediate_dir).mkdir(parents=True, exist_ok=True)

class LightGBMDataProcessor:
    """Data processing utilities for LightGBM pipeline"""
    @staticmethod
    def reduce_mem_usage(df: pd.DataFrame) -> pd.DataFrame:
        """Reduce memory usage by downcasting numeric types"""
        start_mem = df.memory_usage().sum() / 1024**2
        print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
        
        for col in df.columns:
            col_type = df[col].dtype
            if str(col_type) == "category":
                continue
                
            if col_type != object:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == "int":
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)
                        
        end_mem = df.memory_usage().sum() / 1024**2
        print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
        print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
        
        return df
    
    @staticmethod
    def create_time_weights(n_samples: int, decay_factor: float = 0.95) -> np.ndarray:
        """Create exponential decay weights giving more importance to recent data"""
        positions = np.arange(n_samples)
        normalized_positions = positions / (n_samples - 1)
        weights = decay_factor ** (1 - normalized_positions)
        weights = weights * n_samples / weights.sum()
        return weights

class LightGBMFeatureEngineer:
    """Feature engineering for LightGBM pipeline"""
    @staticmethod
    def create_features(df: pd.DataFrame) -> pd.DataFrame:
        """Create market microstructure features"""
        # Check if required columns exist
        required_cols = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < len(required_cols):
            print(f"Warning: Only {len(available_cols)} of {len(required_cols)} required columns available")
            # Create dummy columns if missing
            for col in required_cols:
                if col not in df.columns:
                    df[col] = 0
        
        # Interaction features
        df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
        df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
        df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
        df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
        df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
        df['buy_sell_interaction'] = df['buy_qty'] * df['sell_qty']
        
        # Spread indicators
        df['spread_indicator'] = (df['ask_qty'] - df['bid_qty']) / (df['ask_qty'] + df['bid_qty'] + 1e-8)
        
        # Volume-weighted features
        df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
        df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
        df['volume_weighted_bid'] = df['bid_qty'] * df['volume']
        df['volume_weighted_ask'] = df['ask_qty'] * df['volume']
        
        # Ratios
        df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
        df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
        
        # Order flow imbalance
        df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
        
        # Pressure indicators
        df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-8)
        df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
        
        # Liquidity features
        df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
        df['liquidity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_liquidity'] + 1e-8)
        df['relative_spread'] = (df['ask_qty'] - df['bid_qty']) / (df['volume'] + 1e-8)
        
        # Trade intensity
        df['trade_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-8)
        df['avg_trade_size'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-8)
        df['net_trade_flow'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
        
        # Market depth
        df['depth_ratio'] = df['total_liquidity'] / (df['volume'] + 1e-8)
        df['volume_participation'] = (df['buy_qty'] + df['sell_qty']) / (df['total_liquidity'] + 1e-8)
        df['market_activity'] = df['volume'] * df['total_liquidity']
        
        # Spread proxy
        df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
        df['realized_volatility_proxy'] = np.abs(df['order_flow_imbalance']) * df['volume']
        
        # Normalized volumes
        df['normalized_buy_volume'] = df['buy_qty'] / (df['bid_qty'] + 1e-8)
        df['normalized_sell_volume'] = df['sell_qty'] / (df['ask_qty'] + 1e-8)
        
        # Advanced features
        df['liquidity_adjusted_imbalance'] = df['order_flow_imbalance'] * df['depth_ratio']
        df['pressure_spread_interaction'] = df['buying_pressure'] * df['spread_indicator']
        
        # Clean up infinities and NaNs
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.fillna(0)
        
        return df

class LightGBMVotingPipeline:
    """Complete LightGBM Voting Ensemble Pipeline"""
    def __init__(self, config: Optional[LightGBMVotingConfiguration] = None):
        self.config = config or LightGBMVotingConfiguration()
        self.data_processor = LightGBMDataProcessor()
        self.feature_engineer = LightGBMFeatureEngineer()
        
    def assign_time_folds(self, train: pd.DataFrame) -> pd.DataFrame:
        """Assign time-based folds to training data"""
        train['Fold'] = 0
        
        for fold_num, (start_date, end_date) in self.config.fold_dates.items():
            mask = (train.index >= start_date) & (train.index < end_date)
            train.loc[mask, 'Fold'] = fold_num
            fold_size = mask.sum()
            print(f"Fold {fold_num}: {start_date} to {end_date} - {fold_size:,} samples")
        
        return train
    
    def pearsonr_coeff(self, preds, data):
        """Custom evaluation metric for LightGBM"""
        y_true = data.get_label()
        valid_score = pearsonr(y_true, preds)[0]
        return 'pearsonr_coeff_score', valid_score, True
    
    def train_single_model(self, train_data: lgb.Dataset, valid_data: lgb.Dataset) -> Tuple[lgb.Booster, float]:
        """Train a single LightGBM model"""
        print("Training Model...")
        
        model = lgb.train(
            self.config.lgb_params,
            train_data,
            num_boost_round=self.config.num_boost_round,
            valid_sets=[valid_data],
            feval=self.pearsonr_coeff,
            callbacks=[lgb.callback.log_evaluation(period=50)]
        )
        
        valid_pred = model.predict(valid_data.get_data())
        valid_score = pearsonr(valid_data.get_label(), valid_pred)[0]
        print(f"Validation Score: {valid_score:.6f}")
        
        return model, valid_score
    
    def save_metrics(self, valid_scores: List[float], final_score: float):
        """Save pipeline metrics"""
        metrics = {
            "model": "LightGBM Voting Ensemble",
            "n_models": len(valid_scores),
            "fold_scores": valid_scores,
            "average_fold_score": float(np.mean(valid_scores)),
            "std_fold_score": float(np.std(valid_scores)),
            "final_ensemble_score": float(final_score)
        }
        
        metrics_df = pd.DataFrame([metrics])
        metrics_df.to_csv(self.config.metrics_file, index=False)
        print(f"Metrics saved to {self.config.metrics_file}")
        
        # Register metrics with global configuration
        global_config.register_model_output(
            self.config.model_name,
            'model_metrics',
            self.config.metrics_file,
            'analysis',
            metadata=metrics
        )
    
    def save_fold_performance(self, fold_details: List[Dict[str, Any]]):
        """Save detailed fold performance data"""
        with open(self.config.fold_performance_file, 'w') as f:
            json.dump(fold_details, f, indent=2)
        
        # Register fold performance with global configuration
        global_config.register_model_output(
            self.config.model_name,
            'fold_performance',
            self.config.fold_performance_file,
            'analysis',
            metadata={'n_folds': len(fold_details)}
        )
    
    def run(self) -> Tuple[np.ndarray, List[float]]:
        print("\nStarting LightGBM Voting Ensemble Pipeline...")
        print("This model uses time-based cross-validation with exponential decay weighting")
        
        # Update model status
        global_config.update_model_status(self.config.model_name, "running")
        
        # Load data
        print("\nLoading data...")
        train = pd.read_parquet(self.config.train_path)
        test = pd.read_parquet(self.config.test_path)
        
        print(f"Original data shapes - Train: {train.shape}, Test: {test.shape}")
        
        # Reduce memory usage
        train = self.data_processor.reduce_mem_usage(train)
        test = self.data_processor.reduce_mem_usage(test)
        
        # Feature engineering
        print("\nEngineering features...")
        train = self.feature_engineer.create_features(train)
        test = self.feature_engineer.create_features(test)
        
        # Assign time-based folds
        print("\nAssigning time-based folds...")
        train = self.assign_time_folds(train)
        
        # Create time weights
        print("\nCreating time decay weights...")
        train['weight'] = self.data_processor.create_time_weights(len(train), self.config.decay_factor)
        
        # Verify feature availability
        available_features = [f for f in self.config.feature_names if f in train.columns]
        if len(available_features) < len(self.config.feature_names):
            print(f"Warning: Only {len(available_features)} of {len(self.config.feature_names)} features available")
            print(f"Missing features: {set(self.config.feature_names) - set(available_features)}")
        
        # Initialize storage
        models = []
        valid_scores = []
        fold_details = []
        
        # Train on each fold
        print("\nTraining models on time-based folds...")
        for fold in range(1, 6):  # Train 5 models, validate on fold 6
            print(f"\n{'='*50}")
            print(f"Training Fold {fold}/5")
            print(f"{'='*50}")
            
            # Prepare training and validation data
            train_mask = train['Fold'] != fold
            valid_mask = train['Fold'] == 6  # Always validate on the most recent fold
            
            X_train = train[train_mask][available_features]
            w_train = train[train_mask]['weight']
            X_valid = train[valid_mask][available_features]
            w_valid = train[valid_mask]['weight']
            y_train = train[train_mask]['label']
            y_valid = train[valid_mask]['label']
            
            print(f"Train samples: {len(X_train):,}")
            print(f"Valid samples: {len(X_valid):,}")
            print(f"Train time range: {X_train.index.min()} to {X_train.index.max()}")
            print(f"Valid time range: {X_valid.index.min()} to {X_valid.index.max()}")
            
            # Create LightGBM datasets
            train_data = lgb.Dataset(
                X_train, 
                label=y_train, 
                weight=w_train, 
                free_raw_data=False
            ).construct()
            
            valid_data = lgb.Dataset(
                X_valid, 
                label=y_valid, 
                weight=w_valid, 
                reference=train_data, 
                free_raw_data=False
            ).construct()
            
            # Train model
            model, valid_score = self.train_single_model(train_data, valid_data)
            
            # Save model
            model_path = os.path.join(self.config.intermediate_dir, f'fold_{fold}_model.txt')
            model.save_model(model_path)
            print(f"Model saved to {model_path}")
            
            models.append(model)
            valid_scores.append(valid_score)
            
            # Store fold details
            fold_details.append({
                'fold': fold,
                'train_samples': len(X_train),
                'valid_samples': len(X_valid),
                'train_start': str(X_train.index.min()),
                'train_end': str(X_train.index.max()),
                'valid_start': str(X_valid.index.min()),
                'valid_end': str(X_valid.index.max()),
                'validation_score': float(valid_score)
            })
        
        # Save fold performance details
        self.save_fold_performance(fold_details)
        
        # Summary statistics
        print(f"\n{'='*50}")
        print("Training Summary")
        print(f"{'='*50}")
        print(f"Individual fold scores: {[f'{score:.6f}' for score in valid_scores]}")
        print(f"Average validation score: {np.mean(valid_scores):.6f}")
        print(f"Standard deviation: {np.std(valid_scores):.6f}")
        
        # Create voting ensemble
        print("\nCreating voting ensemble...")
        lgbm_voting = VotingModel(models)
        
        # Make predictions
        print("Making predictions on test data...")
        test_features = [f for f in available_features if f in test.columns]
        predictions = lgbm_voting.predict(test[test_features])
        
        # Save submission
        submission = pd.read_csv(self.config.sample_sub_path)
        submission["prediction"] = predictions
        submission.to_csv(self.config.submission_file, index=False)
        print(f"Submission saved to {self.config.submission_file}")
        
        # Register submission with global configuration
        final_score = np.mean(valid_scores)
        global_config.register_model_output(
            self.config.model_name,
            'submission',
            self.config.submission_file,
            'submission',
            metadata={
                'ensemble_score': float(final_score),
                'n_models': len(models),
                'fold_scores': [float(s) for s in valid_scores],
                'method': 'time_based_voting_ensemble'
            }
        )
        
        # Save metrics
        self.save_metrics(valid_scores, final_score)
        
        # Create feature importance summary
        print("\nCalculating feature importance...")
        feature_importance_sum = np.zeros(len(available_features))
        for model in models:
            feature_importance_sum += model.feature_importance(importance_type='gain')
        
        feature_importance_df = pd.DataFrame({
            'feature': available_features,
            'importance': feature_importance_sum / len(models)
        }).sort_values('importance', ascending=False)
        
        feature_importance_df.to_csv(self.config.feature_importance_file, index=False)
        print(f"Feature importance saved to {self.config.feature_importance_file}")
        
        # Register feature importance with global configuration
        global_config.register_model_output(
            self.config.model_name,
            'feature_importance',
            self.config.feature_importance_file,
            'analysis',
            metadata={
                'top_10_features': feature_importance_df.head(10)['feature'].tolist(),
                'top_10_importances': feature_importance_df.head(10)['importance'].tolist()
            }
        )
        
        print(f"\nTop 10 most important features:")
        for idx, row in feature_importance_df.head(10).iterrows():
            print(f"  {row['feature']}: {row['importance']:.2f}")
        
        print("\nLightGBM Voting pipeline completed successfully!")
        print(f"Final ensemble score: {final_score:.6f}")
        
        return predictions, valid_scores

# ===========================================================================================
# EXECUTION BLOCK
# ===========================================================================================

if __name__ == "__main__":
    try:
        print("\nğŸŒ² Running LightGBM Voting Ensemble Pipeline")
        print("-"*80)
        
        # Clean memory before starting
        aggressive_memory_cleanup()
        
        # Create and run pipeline
        lgbm_pipeline = LightGBMVotingPipeline()
        predictions, valid_scores = lgbm_pipeline.run()
        
        # Extract final score
        final_score = np.mean(valid_scores)
        
        # Update global configuration with success
        global_config.update_model_status('lightgbm_voting', 'completed', score=final_score)
        print("\nâœ… LightGBM Voting pipeline completed successfully")
        
    except Exception as e:
        # Update global configuration with failure
        error_msg = str(e)
        global_config.update_model_status('lightgbm_voting', 'failed', error_message=error_msg)
        print(f"\nâ�Œ LightGBM Voting pipeline failed: {error_msg}")
        raise
        
    finally:
        # Clean up memory
        aggressive_memory_cleanup()
        
        # Display current execution status
        print("\n" + "="*80)
        print("Current Execution Status:")
        print(global_config.get_execution_summary())
        print("="*80)


# Final Ensemble Building with Meta-Learning
# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DRW Crypto Market Prediction - Advanced Meta-Learning Ensemble Builder
This module creates the final ensemble using sophisticated meta-learning techniques
including multi-level stacking, dynamic model selection, and adaptive weighting
"""

import subprocess
import sys
import os
import gc
import warnings
import json
import pickle
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import List, Dict, Tuple, Optional, Any, Union
from pathlib import Path
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')

# Install required packages for meta-learning ensemble
print("Installing packages for Meta-Learning Ensemble Builder...")
packages_to_install = [
    'flaml==2.1.1',
    'lightgbm==4.1.0',
    'optuna==3.4.0',
    'scikit-optimize==0.9.0',
    'matplotlib',
    'seaborn'
]

for package in packages_to_install:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])
    except:
        print(f"Warning: Could not install {package}")

# =============================================================================
# Base Meta-Learning Classes
# =============================================================================

class BaseMetaLearner(ABC):
    """Abstract base class for meta-learners"""
    
    @abstractmethod
    def fit(self, predictions: Dict[str, np.ndarray], y: np.ndarray, **kwargs):
        """Fit the meta-learner"""
        pass
    
    @abstractmethod
    def predict(self, predictions: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """Make predictions using the meta-learner"""
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores"""
        pass

# =============================================================================
# Multi-Level Stacking Meta-Learner
# =============================================================================

class MultiLevelStackingMetaLearner(BaseMetaLearner):
    """Multi-level stacking with diverse meta-features and models"""
    
    def __init__(self, n_levels: int = 2, random_state: int = 42):
        self.n_levels = n_levels
        self.random_state = random_state
        self.level_models = {}
        self.feature_generators = {}
        self.feature_names = []
        self.scaler = RobustScaler()
        
    def _create_polynomial_features(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create polynomial interaction features"""
        poly_features = {}
        model_names = list(predictions.keys())
        
        # Quadratic features
        for i, name1 in enumerate(model_names):
            poly_features[f'{name1}_squared'] = predictions[name1] ** 2
            
            for j, name2 in enumerate(model_names[i+1:], i+1):
                poly_features[f'{name1}_x_{name2}'] = predictions[name1] * predictions[name2]
        
        return poly_features
    
    def _create_rank_features(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create rank-based features"""
        rank_features = {}
        pred_array = np.column_stack(list(predictions.values()))
        
        # Rank of each prediction
        for i, name in enumerate(predictions.keys()):
            rank_features[f'{name}_rank'] = np.argsort(np.argsort(predictions[name])) / len(predictions[name])
        
        # Consensus ranking
        mean_ranks = np.mean([rank_features[f'{name}_rank'] for name in predictions.keys()], axis=0)
        rank_features['consensus_rank'] = mean_ranks
        
        return rank_features
    
    def _create_statistical_features(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create statistical aggregation features"""
        pred_array = np.column_stack(list(predictions.values()))
        
        stats_features = {
            'mean': np.mean(pred_array, axis=1),
            'median': np.median(pred_array, axis=1),
            'std': np.std(pred_array, axis=1),
            'min': np.min(pred_array, axis=1),
            'max': np.max(pred_array, axis=1),
            'range': np.ptp(pred_array, axis=1),
            'iqr': np.percentile(pred_array, 75, axis=1) - np.percentile(pred_array, 25, axis=1),
            'skew': np.array([np.mean((pred_array[i] - np.mean(pred_array[i])) ** 3) / 
                             (np.std(pred_array[i]) ** 3 + 1e-8) for i in range(len(pred_array))]),
            'kurtosis': np.array([np.mean((pred_array[i] - np.mean(pred_array[i])) ** 4) / 
                                 (np.std(pred_array[i]) ** 4 + 1e-8) - 3 for i in range(len(pred_array))])
        }
        
        # Coefficient of variation
        stats_features['cv'] = np.where(stats_features['mean'] != 0,
                                       stats_features['std'] / np.abs(stats_features['mean']),
                                       0)
        
        return stats_features
    
    def _create_distance_features(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Create distance-based features"""
        distance_features = {}
        pred_array = np.column_stack(list(predictions.values()))
        
        # Distance from mean
        mean_pred = np.mean(pred_array, axis=1)
        for i, name in enumerate(predictions.keys()):
            distance_features[f'{name}_dist_from_mean'] = np.abs(predictions[name] - mean_pred)
        
        # Pairwise distances
        model_names = list(predictions.keys())
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                distance_features[f'dist_{model_names[i]}_{model_names[j]}'] = \
                    np.abs(predictions[model_names[i]] - predictions[model_names[j]])
        
        return distance_features
    
    def _generate_meta_features(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Generate comprehensive meta-features"""
        all_features = {}
        
        # Base predictions
        all_features.update(predictions)
        
        # Polynomial features
        all_features.update(self._create_polynomial_features(predictions))
        
        # Rank features
        all_features.update(self._create_rank_features(predictions))
        
        # Statistical features
        all_features.update(self._create_statistical_features(predictions))
        
        # Distance features
        all_features.update(self._create_distance_features(predictions))
        
        # Store feature names for later use
        self.feature_names = list(all_features.keys())
        
        # Convert to array
        return np.column_stack([all_features[name] for name in self.feature_names])
    
    def fit(self, predictions: Dict[str, np.ndarray], y: np.ndarray, **kwargs):
        """Fit multi-level stacking models"""
        print("\nTraining Multi-Level Stacking Meta-Learner...")
        
        # Generate meta-features
        X_meta = self._generate_meta_features(predictions)
        X_meta_scaled = self.scaler.fit_transform(X_meta)
        
        # Level 1: Diverse base meta-learners
        print("  Training Level 1 models...")
        
        # Import LightGBM after installation
        try:
            import lightgbm as lgb
            
            self.level_models['level1'] = {
                'rf': RandomForestRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=20,
                    random_state=self.random_state
                ),
                'et': ExtraTreesRegressor(
                    n_estimators=200,
                    max_depth=10,
                    min_samples_split=20,
                    random_state=self.random_state
                ),
                'lgb': lgb.LGBMRegressor(
                    n_estimators=200,
                    max_depth=8,
                    learning_rate=0.05,
                    random_state=self.random_state,
                    verbose=-1
                ),
                'mlp': MLPRegressor(
                    hidden_layer_sizes=(100, 50),
                    activation='relu',
                    solver='adam',
                    alpha=0.01,
                    random_state=self.random_state,
                    max_iter=500
                ),
                'ridge': Ridge(alpha=1.0, random_state=self.random_state),
                'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state),
                'huber': HuberRegressor(epsilon=1.35, alpha=0.01)
            }
        except ImportError:
            print("  Warning: LightGBM not available, using alternative models")
            self.level_models['level1'] = {
                'rf': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=self.random_state),
                'et': ExtraTreesRegressor(n_estimators=200, max_depth=10, random_state=self.random_state),
                'ridge': Ridge(alpha=1.0, random_state=self.random_state),
                'elastic': ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=self.random_state)
            }
        
        # Train level 1 models
        level1_predictions = {}
        for name, model in self.level_models['level1'].items():
            model.fit(X_meta_scaled, y)
            level1_predictions[name] = model.predict(X_meta_scaled)
            score = pearsonr(level1_predictions[name], y)[0]
            print(f"    {name}: correlation = {score:.4f}")
        
        # Level 2: Meta-blender
        if self.n_levels >= 2:
            print("  Training Level 2 blender...")
            
            # Combine level 1 predictions with original predictions
            level2_features = np.column_stack(
                list(level1_predictions.values()) + 
                [predictions[name] for name in predictions.keys()]
            )
            
            # Train final blender
            self.level_models['blender'] = Ridge(alpha=0.1, random_state=self.random_state)
            self.level_models['blender'].fit(level2_features, y)
            
            final_pred = self.level_models['blender'].predict(level2_features)
            final_score = pearsonr(final_pred, y)[0]
            print(f"    Final blender: correlation = {final_score:.4f}")
    
    def predict(self, predictions: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """Make predictions using stacked models"""
        # Generate meta-features
        X_meta = self._generate_meta_features(predictions)
        X_meta_scaled = self.scaler.transform(X_meta)
        
        # Level 1 predictions
        level1_predictions = {}
        for name, model in self.level_models['level1'].items():
            level1_predictions[name] = model.predict(X_meta_scaled)
        
        # Final prediction
        if 'blender' in self.level_models:
            level2_features = np.column_stack(
                list(level1_predictions.values()) + 
                [predictions[name] for name in predictions.keys()]
            )
            return self.level_models['blender'].predict(level2_features)
        else:
            # Simple average if no blender
            return np.mean(list(level1_predictions.values()), axis=0)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get aggregated feature importance"""
        importance_scores = {}
        
        # Get importance from tree-based models
        for name, model in self.level_models['level1'].items():
            if hasattr(model, 'feature_importances_'):
                for i, feat_name in enumerate(self.feature_names):
                    if feat_name not in importance_scores:
                        importance_scores[feat_name] = 0
                    importance_scores[feat_name] += model.feature_importances_[i]
        
        # Normalize
        total = sum(importance_scores.values())
        if total > 0:
            importance_scores = {k: v/total for k, v in importance_scores.items()}
        
        return importance_scores

# =============================================================================
# Dynamic Model Selection Meta-Learner
# =============================================================================

class DynamicModelSelector(BaseMetaLearner):
    """Selects models dynamically based on market regime detection"""
    
    def __init__(self, n_regimes: int = 4, lookback_window: int = 100, random_state: int = 42):
        self.n_regimes = n_regimes
        self.lookback_window = lookback_window
        self.random_state = random_state
        self.regime_detector = None
        self.regime_models = {}
        self.regime_features_scaler = StandardScaler()
        self.model_performance_history = {}
    
    def _extract_regime_features(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Extract features for regime detection"""
        pred_array = np.column_stack(list(predictions.values()))
        
        regime_features = []
        
        # Model disagreement
        model_std = np.std(pred_array, axis=1)
        regime_features.append(model_std)
        
        # Prediction level
        mean_pred = np.mean(pred_array, axis=1)
        regime_features.append(mean_pred)
        
        # Rolling statistics (simulated with expanding windows)
        window_size = min(self.lookback_window, len(mean_pred) // 10)
        if window_size > 1:
            # Rolling volatility proxy
            rolling_std = np.array([
                np.std(mean_pred[max(0, i-window_size):i+1]) 
                for i in range(len(mean_pred))
            ])
            regime_features.append(rolling_std)
            
            # Trend strength proxy
            trend_strength = np.array([
                (mean_pred[i] - mean_pred[max(0, i-window_size)]) / (window_size + 1e-8)
                for i in range(len(mean_pred))
            ])
            regime_features.append(trend_strength)
        
        # Model correlation changes
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                model_names = list(predictions.keys())
                corr_proxy = predictions[model_names[i]] * predictions[model_names[j]]
                regime_features.append(corr_proxy)
        
        return np.column_stack(regime_features)
    
    def _detect_regimes(self, regime_features: np.ndarray) -> np.ndarray:
        """Detect market regimes using clustering"""
        # Apply PCA for dimensionality reduction
        pca = PCA(n_components=min(5, regime_features.shape[1]), random_state=self.random_state)
        regime_features_pca = pca.fit_transform(regime_features)
        
        # Cluster into regimes
        self.regime_detector = KMeans(
            n_clusters=self.n_regimes,
            random_state=self.random_state,
            n_init=10
        )
        regimes = self.regime_detector.fit_predict(regime_features_pca)
        
        return regimes
    
    def _select_best_models_for_regime(self, predictions: Dict[str, np.ndarray], 
                                     y: np.ndarray, regime_mask: np.ndarray) -> Dict[str, float]:
        """Select best performing models for a specific regime"""
        regime_predictions = {name: pred[regime_mask] for name, pred in predictions.items()}
        y_regime = y[regime_mask]
        
        model_scores = {}
        for name, pred in regime_predictions.items():
            if len(pred) > 10:  # Minimum samples
                score = pearsonr(pred, y_regime)[0]
                model_scores[name] = max(0, score)  # Clip negative correlations
            else:
                model_scores[name] = 0
        
        # Normalize scores
        total_score = sum(model_scores.values())
        if total_score > 0:
            model_scores = {k: v/total_score for k, v in model_scores.items()}
        else:
            # Equal weights if all models perform poorly
            model_scores = {k: 1/len(predictions) for k in predictions.keys()}
        
        return model_scores
    
    def fit(self, predictions: Dict[str, np.ndarray], y: np.ndarray, **kwargs):
        """Fit regime-specific models"""
        print("\nTraining Dynamic Model Selector...")
        
        # Extract regime features
        regime_features = self._extract_regime_features(predictions)
        regime_features_scaled = self.regime_features_scaler.fit_transform(regime_features)
        
        # Detect regimes
        regimes = self._detect_regimes(regime_features_scaled)
        
        print(f"  Detected {self.n_regimes} market regimes")
        unique_regimes, regime_counts = np.unique(regimes, return_counts=True)
        for regime, count in zip(unique_regimes, regime_counts):
            print(f"    Regime {regime}: {count} samples ({count/len(regimes)*100:.1f}%)")
        
        # Train regime-specific models
        for regime in unique_regimes:
            regime_mask = regimes == regime
            
            if np.sum(regime_mask) > 20:  # Minimum samples per regime
                # Select best models for this regime
                model_weights = self._select_best_models_for_regime(predictions, y, regime_mask)
                
                # Create weighted ensemble for regime
                self.regime_models[regime] = {
                    'weights': model_weights,
                    'n_samples': np.sum(regime_mask)
                }
                
                # Display regime model selection
                print(f"  Regime {regime} model weights:")
                for model_name, weight in sorted(model_weights.items(), key=lambda x: x[1], reverse=True):
                    if weight > 0.1:  # Only show significant weights
                        print(f"    {model_name}: {weight:.3f}")
    
    def predict(self, predictions: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """Predict using regime-appropriate models"""
        # Extract regime features
        regime_features = self._extract_regime_features(predictions)
        regime_features_scaled = self.regime_features_scaler.transform(regime_features)
        
        # Detect current regimes
        regime_features_pca = PCA(n_components=min(5, regime_features_scaled.shape[1]), 
                                 random_state=self.random_state).fit_transform(regime_features_scaled)
        current_regimes = self.regime_detector.predict(regime_features_pca)
        
        # Make predictions based on regime
        final_predictions = np.zeros(len(next(iter(predictions.values()))))
        
        for regime in np.unique(current_regimes):
            regime_mask = current_regimes == regime
            
            if regime in self.regime_models:
                weights = self.regime_models[regime]['weights']
                regime_pred = np.zeros(np.sum(regime_mask))
                
                for model_name, weight in weights.items():
                    regime_pred += weight * predictions[model_name][regime_mask]
                
                final_predictions[regime_mask] = regime_pred
            else:
                # Fallback to equal weights
                regime_pred = np.mean([pred[regime_mask] for pred in predictions.values()], axis=0)
                final_predictions[regime_mask] = regime_pred
        
        return final_predictions
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get model importance across regimes"""
        importance = {}
        
        for regime, regime_info in self.regime_models.items():
            weights = regime_info['weights']
            n_samples = regime_info['n_samples']
            
            for model_name, weight in weights.items():
                if model_name not in importance:
                    importance[model_name] = 0
                importance[model_name] += weight * n_samples
        
        # Normalize
        total = sum(importance.values())
        if total > 0:
            importance = {k: v/total for k, v in importance.items()}
        
        return importance

# =============================================================================
# Bayesian Model Averaging Meta-Learner
# =============================================================================

class BayesianModelAveraging(BaseMetaLearner):
    """Bayesian approach to model combination with uncertainty quantification"""
    
    def __init__(self, prior_strength: float = 1.0, mcmc_samples: int = 1000, random_state: int = 42):
        self.prior_strength = prior_strength
        self.mcmc_samples = mcmc_samples
        self.random_state = random_state
        self.posterior_weights = None
        self.model_precisions = None
        self.evidence = None
    
    def _compute_model_likelihood(self, predictions: np.ndarray, y: np.ndarray, 
                                 precision: float) -> float:
        """Compute likelihood of data given model predictions"""
        residuals = y - predictions
        log_likelihood = -0.5 * precision * np.sum(residuals ** 2)
        log_likelihood += 0.5 * len(y) * (np.log(precision) - np.log(2 * np.pi))
        return log_likelihood
    
    def _sample_posterior(self, predictions: Dict[str, np.ndarray], y: np.ndarray):
        """Sample from posterior distribution using simple MCMC"""
        n_models = len(predictions)
        model_names = list(predictions.keys())
        pred_matrix = np.column_stack([predictions[name] for name in model_names])
        
        # Initialize
        weights = np.ones(n_models) / n_models
        precisions = np.ones(n_models)
        
        # Storage for samples
        weight_samples = []
        precision_samples = []
        
        # Simple Metropolis-Hastings
        for iteration in range(self.mcmc_samples * 2):  # Extra samples for burn-in
            # Update weights (Dirichlet-like proposal)
            proposed_weights = np.random.dirichlet(weights * 10 + self.prior_strength)
            
            # Compute ensemble predictions
            current_pred = np.dot(pred_matrix, weights)
            proposed_pred = np.dot(pred_matrix, proposed_weights)
            
            # Compute acceptance ratio
            current_likelihood = self._compute_model_likelihood(current_pred, y, np.mean(precisions))
            proposed_likelihood = self._compute_model_likelihood(proposed_pred, y, np.mean(precisions))
            
            accept_prob = np.exp(proposed_likelihood - current_likelihood)
            
            if np.random.rand() < accept_prob:
                weights = proposed_weights
            
            # Update precisions (Gamma-like)
            for i in range(n_models):
                residuals = y - pred_matrix[:, i]
                shape = len(y) / 2 + 1
                scale = 2 / (np.sum(residuals ** 2) + 1e-6)
                precisions[i] = np.random.gamma(shape, scale)
            
            # Store samples after burn-in
            if iteration >= self.mcmc_samples:
                weight_samples.append(weights.copy())
                precision_samples.append(precisions.copy())
        
        # Compute posterior statistics
        self.posterior_weights = np.mean(weight_samples, axis=0)
        self.model_precisions = np.mean(precision_samples, axis=0)
        
        # Store samples for uncertainty quantification
        self.weight_samples = np.array(weight_samples)
        self.precision_samples = np.array(precision_samples)
    
    def fit(self, predictions: Dict[str, np.ndarray], y: np.ndarray, **kwargs):
        """Fit Bayesian model averaging"""
        print("\nTraining Bayesian Model Averaging...")
        
        # Sample from posterior
        self._sample_posterior(predictions, y)
        
        # Display results
        print("  Posterior model weights:")
        for name, weight in zip(predictions.keys(), self.posterior_weights):
            print(f"    {name}: {weight:.3f} (precision: {self.model_precisions[list(predictions.keys()).index(name)]:.2f})")
        
        # Compute model evidence (marginal likelihood approximation)
        pred_matrix = np.column_stack(list(predictions.values()))
        ensemble_pred = np.dot(pred_matrix, self.posterior_weights)
        self.evidence = pearsonr(ensemble_pred, y)[0]
        print(f"  Model evidence (correlation): {self.evidence:.4f}")
    
    def predict(self, predictions: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """Make predictions with uncertainty quantification"""
        pred_matrix = np.column_stack([predictions[name] for name in predictions.keys()])
        
        # Point prediction
        point_prediction = np.dot(pred_matrix, self.posterior_weights)
        
        # Uncertainty quantification if requested
        if kwargs.get('return_uncertainty', False):
            # Sample predictions
            sample_predictions = []
            n_uncertainty_samples = min(100, len(self.weight_samples))
            
            for i in range(n_uncertainty_samples):
                sample_pred = np.dot(pred_matrix, self.weight_samples[i])
                sample_predictions.append(sample_pred)
            
            sample_predictions = np.array(sample_predictions)
            uncertainty = np.std(sample_predictions, axis=0)
            
            return point_prediction, uncertainty
        
        return point_prediction
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get posterior weights as importance"""
        return dict(zip(self.posterior_weights.keys() if hasattr(self.posterior_weights, 'keys') 
                       else range(len(self.posterior_weights)), self.posterior_weights))

# =============================================================================
# Neural Attention Meta-Learner
# =============================================================================

class NeuralAttentionMetaLearner(BaseMetaLearner):
    """Neural network with attention mechanism for model weighting"""
    
    def __init__(self, hidden_size: int = 64, n_heads: int = 4, dropout: float = 0.3, 
                 learning_rate: float = 0.001, n_epochs: int = 100, random_state: int = 42):
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.attention_weights_history = []
    
    def _build_attention_model(self, n_models: int):
        """Build attention-based neural network using sklearn"""
        # Simplified attention mechanism using MLPRegressor
        # In production, you'd use PyTorch/TensorFlow for true attention
        
        self.model = MLPRegressor(
            hidden_layer_sizes=(self.hidden_size * 2, self.hidden_size, self.hidden_size // 2),
            activation='relu',
            solver='adam',
            alpha=0.01,
            learning_rate_init=self.learning_rate,
            max_iter=self.n_epochs,
            random_state=self.random_state,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=10
        )
    
    def _compute_attention_weights(self, predictions: np.ndarray) -> np.ndarray:
        """Compute pseudo-attention weights"""
        # Simplified attention: use correlations between predictions
        n_samples, n_models = predictions.shape
        
        # Compute pairwise similarities
        attention_scores = np.zeros((n_samples, n_models))
        
        for i in range(n_models):
            for j in range(n_models):
                if i != j:
                    # Use rolling correlation as attention score
                    window = min(50, n_samples // 10)
                    for k in range(window, n_samples):
                        if k < window:
                            corr = 0.5
                        else:
                            corr = np.corrcoef(
                                predictions[k-window:k, i],
                                predictions[k-window:k, j]
                            )[0, 1]
                            if np.isnan(corr):
                                corr = 0.5
                        attention_scores[k, i] += corr
        
        # Normalize to create attention weights
        attention_weights = np.exp(attention_scores)
        attention_weights = attention_weights / (np.sum(attention_weights, axis=1, keepdims=True) + 1e-8)
        
        return attention_weights
    
    def fit(self, predictions: Dict[str, np.ndarray], y: np.ndarray, **kwargs):
        """Fit neural attention model"""
        print("\nTraining Neural Attention Meta-Learner...")
        
        # Prepare data
        pred_matrix = np.column_stack(list(predictions.values()))
        n_models = pred_matrix.shape[1]
        
        # Build model
        self._build_attention_model(n_models)
        
        # Create attention-weighted features
        attention_weights = self._compute_attention_weights(pred_matrix)
        self.attention_weights_history = attention_weights
        
        # Create enhanced features
        features = []
        features.append(pred_matrix)  # Original predictions
        features.append(pred_matrix ** 2)  # Squared predictions
        features.append(attention_weights)  # Attention weights
        
        # Weighted predictions
        weighted_preds = pred_matrix * attention_weights
        features.append(weighted_preds)
        
        # Concatenate all features
        X = np.hstack(features)
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Evaluate
        y_pred = self.model.predict(X_scaled)
        score = pearsonr(y_pred, y)[0]
        print(f"  Training correlation: {score:.4f}")
        
        # Analyze attention patterns
        mean_attention = np.mean(attention_weights, axis=0)
        print("  Average attention weights:")
        for name, weight in zip(predictions.keys(), mean_attention):
            print(f"    {name}: {weight:.3f}")
    
    def predict(self, predictions: Dict[str, np.ndarray], **kwargs) -> np.ndarray:
        """Predict using attention model"""
        # Prepare data
        pred_matrix = np.column_stack(list(predictions.values()))
        
        # Compute attention weights
        attention_weights = self._compute_attention_weights(pred_matrix)
        
        # Create features
        features = []
        features.append(pred_matrix)
        features.append(pred_matrix ** 2)
        features.append(attention_weights)
        features.append(pred_matrix * attention_weights)
        
        X = np.hstack(features)
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get average attention weights as importance"""
        if len(self.attention_weights_history) > 0:
            mean_attention = np.mean(self.attention_weights_history, axis=0)
            return dict(enumerate(mean_attention))
        return {}

# =============================================================================
# Advanced Meta-Learning Ensemble Builder
# =============================================================================

class AdvancedMetaLearningEnsemble:
    """Main ensemble builder using multiple meta-learning strategies"""
    
    def __init__(self, strategies: List[str] = None):
        # Base configuration
        self.model_name = "meta_ensemble"
        self.model_directory = os.path.join(global_config.base_dir, "meta_ensemble")
        
        # Register with global configuration
        global_config.register_model(self.model_name, self.model_directory)
        
        # Output paths
        self.final_submission_path = "/kaggle/working/final_submission.csv"
        self.ensemble_analysis_path = os.path.join(self.model_directory, "meta_ensemble_analysis.csv")
        self.model_correlations_path = os.path.join(self.model_directory, "model_correlations.csv")
        self.meta_weights_path = os.path.join(self.model_directory, "meta_weights.json")
        self.visualization_path = os.path.join(self.model_directory, "meta_ensemble_visualization.png")
        self.meta_models_path = os.path.join(self.model_directory, "meta_models.pkl")
        
        # Meta-learning strategies
        if strategies is None:
            strategies = ['stacking', 'dynamic', 'bayesian', 'attention']
        self.strategies = strategies
        
        # Initialize meta-learners
        self.meta_learners = {
            'stacking': MultiLevelStackingMetaLearner(n_levels=2),
            'dynamic': DynamicModelSelector(n_regimes=4),
            'bayesian': BayesianModelAveraging(prior_strength=1.0),
            'attention': NeuralAttentionMetaLearner(hidden_size=64)
        }
        
        # Ensemble parameters
        self.n_folds = 5
        self.random_state = 42
        self.final_blender = None
        
        # Ensure directories exist
        Path(self.model_directory).mkdir(parents=True, exist_ok=True)
    
    def load_base_predictions(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
        """Load predictions from base models"""
        submissions = {}
        predictions = {}
        
        print("\nLoading base model predictions...")
        
        # Define target models
        target_models = {
            'xgboost': ['submission', 'xgboost_submission'],
            'autoencoder_simple': ['ensemble_simple_submission', 'autoencoder_simple'],
            'autoencoder_weighted': ['ensemble_weighted_submission', 'autoencoder_weighted']
        }
        
        # Load from registry
        for model_name, possible_output_names in target_models.items():
            if model_name in global_config.model_registry:
                model_record = global_config.model_registry[model_name]
                
                if model_record.status == 'completed':
                    for output_name in possible_output_names:
                        if output_name in model_record.outputs:
                            output = model_record.outputs[output_name]
                            if os.path.exists(output.file_path):
                                try:
                                    submission_df = pd.read_csv(output.file_path)
                                    submissions[model_name] = submission_df
                                    predictions[model_name] = submission_df.iloc[:, 1].values
                                    print(f"  âœ“ Loaded {model_name}")
                                    break
                                except Exception as e:
                                    print(f"  âœ— Error loading {model_name}: {e}")
        
        # Fallback paths
        fallback_paths = {
            'xgboost': os.path.join(global_config.base_dir, "triple_xgboost", "submission.csv"),
            'autoencoder_simple': os.path.join(global_config.base_dir, "autoencoder_deepmlp", "ensemble_simple_submission.csv"),
            'autoencoder_weighted': os.path.join(global_config.base_dir, "autoencoder_deepmlp", "ensemble_weighted_submission.csv")
        }
        
        for model_name, path in fallback_paths.items():
            if model_name not in submissions and os.path.exists(path):
                try:
                    submission_df = pd.read_csv(path)
                    submissions[model_name] = submission_df
                    predictions[model_name] = submission_df.iloc[:, 1].values
                    print(f"  âœ“ Loaded {model_name} (fallback)")
                except Exception as e:
                    print(f"  âœ— Error: {e}")
        
        print(f"\nLoaded {len(predictions)} models")
        return submissions, predictions
    
    def create_synthetic_labels(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Create synthetic labels for meta-learning"""
        # Use weighted average based on model diversity
        pred_array = np.column_stack(list(predictions.values()))
        
        # Calculate pairwise correlations
        n_models = pred_array.shape[1]
        correlations = np.corrcoef(pred_array.T)
        
        # Weight based on uniqueness (lower correlation with others)
        uniqueness = 1 - (np.sum(correlations, axis=1) - 1) / (n_models - 1)
        weights = uniqueness / np.sum(uniqueness)
        
        # Create synthetic target
        synthetic_y = np.average(pred_array, axis=1, weights=weights)
        
        # Add small noise for regularization
        np.random.seed(self.random_state)
        noise = np.random.normal(0, np.std(synthetic_y) * 0.01, size=len(synthetic_y))
        synthetic_y += noise
        
        return synthetic_y
    
    def train_meta_learners(self, predictions: Dict[str, np.ndarray], y: np.ndarray) -> Dict[str, float]:
        """Train all meta-learners and evaluate performance"""
        scores = {}
        
        print("\nTraining meta-learners...")
        
        # Split data for validation
        split_idx = int(len(y) * 0.8)
        train_predictions = {k: v[:split_idx] for k, v in predictions.items()}
        val_predictions = {k: v[split_idx:] for k, v in predictions.items()}
        y_train = y[:split_idx]
        y_val = y[split_idx:]
        
        # Train each meta-learner
        for strategy in self.strategies:
            if strategy in self.meta_learners:
                print(f"\n--- {strategy.upper()} META-LEARNER ---")
                
                try:
                    # Train
                    self.meta_learners[strategy].fit(train_predictions, y_train)
                    
                    # Validate
                    val_pred = self.meta_learners[strategy].predict(val_predictions)
                    score = pearsonr(val_pred, y_val)[0]
                    scores[strategy] = score
                    
                    print(f"  Validation score: {score:.4f}")
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    scores[strategy] = 0.0
        
        return scores
    
    def create_final_ensemble(self, predictions: Dict[str, np.ndarray], 
                            meta_scores: Dict[str, float]) -> np.ndarray:
        """Create final ensemble from meta-learners"""
        print("\nCreating final ensemble...")
        
        # Get predictions from each meta-learner
        meta_predictions = {}
        
        for strategy in self.strategies:
            if strategy in self.meta_learners and meta_scores.get(strategy, 0) > 0:
                try:
                    pred = self.meta_learners[strategy].predict(predictions)
                    meta_predictions[strategy] = pred
                except:
                    pass
        
        if not meta_predictions:
            # Fallback to simple average
            print("  Warning: No meta-learners succeeded, using simple average")
            return np.mean(list(predictions.values()), axis=0)
        
        # Weight meta-learners by performance
        weights = np.array([meta_scores.get(name, 0) for name in meta_predictions.keys()])
        weights = np.maximum(weights, 0)  # Ensure non-negative
        
        if np.sum(weights) > 0:
            weights = weights / np.sum(weights)
        else:
            weights = np.ones(len(weights)) / len(weights)
        
        print("\n  Meta-learner weights:")
        for name, weight in zip(meta_predictions.keys(), weights):
            print(f"    {name}: {weight:.3f}")
        
        # Create weighted ensemble
        final_predictions = np.average(
            list(meta_predictions.values()),
            axis=0,
            weights=weights
        )
        
        return final_predictions
    
    def save_meta_models(self):
        """Save trained meta-learners"""
        with open(self.meta_models_path, 'wb') as f:
            pickle.dump({
                'meta_learners': self.meta_learners,
                'final_blender': self.final_blender
            }, f)
        
        global_config.register_model_output(
            self.model_name,
            'meta_models',
            self.meta_models_path,
            'model'
        )
    
    def create_advanced_visualization(self, predictions: Dict[str, np.ndarray],
                                    final_predictions: np.ndarray,
                                    meta_scores: Dict[str, float]):
        """Create comprehensive visualization of meta-learning results"""
        try:
            fig = plt.figure(figsize=(16, 12))
            
            # Create grid
            gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
            
            # 1. Base model predictions distribution
            ax1 = fig.add_subplot(gs[0, :2])
            for name, pred in predictions.items():
                ax1.hist(pred, bins=50, alpha=0.5, label=name, density=True)
            ax1.set_xlabel('Prediction Value')
            ax1.set_ylabel('Density')
            ax1.set_title('Base Model Prediction Distributions')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 2. Meta-learner performance
            ax2 = fig.add_subplot(gs[0, 2])
            meta_names = list(meta_scores.keys())
            meta_values = list(meta_scores.values())
            bars = ax2.bar(meta_names, meta_values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            ax2.set_ylabel('Validation Score')
            ax2.set_title('Meta-Learner Performance')
            ax2.set_ylim(0, 1)
            for i, (name, value) in enumerate(zip(meta_names, meta_values)):
                ax2.text(i, value + 0.01, f'{value:.3f}', ha='center', va='bottom')
            
            # 3. Model correlation heatmap
            ax3 = fig.add_subplot(gs[1, 0])
            pred_array = np.column_stack(list(predictions.values()))
            corr_matrix = np.corrcoef(pred_array.T)
            im = ax3.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
            ax3.set_xticks(range(len(predictions)))
            ax3.set_yticks(range(len(predictions)))
            ax3.set_xticklabels(list(predictions.keys()), rotation=45, ha='right')
            ax3.set_yticklabels(list(predictions.keys()))
            ax3.set_title('Model Correlations')
            plt.colorbar(im, ax=ax3)
            
            # 4. Final vs base predictions scatter
            ax4 = fig.add_subplot(gs[1, 1])
            mean_base = np.mean(list(predictions.values()), axis=0)
            ax4.scatter(mean_base, final_predictions, alpha=0.5, s=1)
            ax4.plot([mean_base.min(), mean_base.max()], 
                    [mean_base.min(), mean_base.max()], 
                    'r--', lw=2)
            ax4.set_xlabel('Base Ensemble (Simple Average)')
            ax4.set_ylabel('Meta-Learning Ensemble')
            ax4.set_title('Meta-Learning vs Simple Ensemble')
            ax4.grid(True, alpha=0.3)
            
            # 5. Feature importance from stacking
            ax5 = fig.add_subplot(gs[1, 2])
            if 'stacking' in self.meta_learners:
                importance = self.meta_learners['stacking'].get_feature_importance()
                if importance:
                    # Show top 10 features
                    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
                    feature_names = [f[0] for f in sorted_features]
                    feature_values = [f[1] for f in sorted_features]
                    
                    ax5.barh(range(len(feature_names)), feature_values)
                    ax5.set_yticks(range(len(feature_names)))
                    ax5.set_yticklabels(feature_names)
                    ax5.set_xlabel('Importance')
                    ax5.set_title('Top 10 Stacking Features')
            
            # 6. Final prediction distribution
            ax6 = fig.add_subplot(gs[2, :])
            ax6.hist(final_predictions, bins=100, color='darkgreen', alpha=0.8, edgecolor='black')
            ax6.axvline(np.mean(final_predictions), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(final_predictions):.6f}')
            ax6.axvline(np.median(final_predictions), color='orange', linestyle='--', 
                       label=f'Median: {np.median(final_predictions):.6f}')
            ax6.set_xlabel('Prediction Value')
            ax6.set_ylabel('Count')
            ax6.set_title('Final Meta-Ensemble Predictions')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
            
            # Add text summary
            summary_text = f"""
Meta-Learning Ensemble Summary:
- Models used: {len(predictions)}
- Meta-learners: {', '.join(self.strategies)}
- Best meta-learner: {max(meta_scores.items(), key=lambda x: x[1])[0] if meta_scores else 'N/A'}
- Prediction range: [{np.min(final_predictions):.6f}, {np.max(final_predictions):.6f}]
- Standard deviation: {np.std(final_predictions):.6f}
            """
            fig.text(0.02, 0.02, summary_text, fontsize=10, 
                    bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
            
            plt.suptitle('DRW Crypto Prediction - Meta-Learning Ensemble Analysis', fontsize=16)
            plt.tight_layout()
            plt.savefig(self.visualization_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\nVisualization saved to: {self.visualization_path}")
            
            global_config.register_model_output(
                self.model_name,
                'visualization',
                self.visualization_path,
                'visualization'
            )
            
        except Exception as e:
            print(f"Warning: Could not create visualization: {e}")
    
    def run(self) -> pd.DataFrame:
        """Execute the complete meta-learning ensemble pipeline"""
        print("\n" + "="*80)
        print("ADVANCED META-LEARNING ENSEMBLE BUILDER")
        print("="*80)
        
        # Update status
        global_config.update_model_status(self.model_name, 'running')
        
        try:
            # Load base predictions
            submissions, predictions = self.load_base_predictions()
            
            if len(predictions) < 2:
                raise ValueError(f"Insufficient models for ensemble: {len(predictions)}")
            
            # Create synthetic labels
            synthetic_y = self.create_synthetic_labels(predictions)
            
            # Train meta-learners
            meta_scores = self.train_meta_learners(predictions, synthetic_y)
            
            # Create final ensemble
            final_predictions = self.create_final_ensemble(predictions, meta_scores)
            
            # Post-processing
            final_predictions = self.apply_post_processing(final_predictions, predictions)
            
            # Create submission
            template = next(iter(submissions.values()))
            final_submission = template.copy()
            final_submission.iloc[:, 1] = final_predictions
            
            # Save outputs
            final_submission.to_csv(self.final_submission_path, index=False)
            print(f"\nFinal submission saved to: {self.final_submission_path}")
            
            global_config.register_model_output(
                self.model_name,
                'final_submission',
                self.final_submission_path,
                'submission'
            )
            
            # Save meta-learner weights
            meta_info = {
                'strategies_used': self.strategies,
                'meta_scores': meta_scores,
                'n_base_models': len(predictions),
                'base_models': list(predictions.keys()),
                'prediction_stats': {
                    'mean': float(np.mean(final_predictions)),
                    'std': float(np.std(final_predictions)),
                    'min': float(np.min(final_predictions)),
                    'max': float(np.max(final_predictions))
                }
            }
            
            with open(self.meta_weights_path, 'w') as f:
                json.dump(meta_info, f, indent=2)
            
            global_config.register_model_output(
                self.model_name,
                'meta_weights',
                self.meta_weights_path,
                'config',
                metadata=meta_info
            )
            
            # Save models
            self.save_meta_models()
            
            # Create visualization
            self.create_advanced_visualization(predictions, final_predictions, meta_scores)
            
            # Update status
            best_score = max(meta_scores.values()) if meta_scores else 0.0
            global_config.update_model_status(self.model_name, 'completed', score=best_score)
            
            print("\nâœ… Meta-learning ensemble completed successfully")
            
            return final_submission
            
        except Exception as e:
            error_msg = str(e)
            global_config.update_model_status(self.model_name, 'failed', error_message=error_msg)
            print(f"\nâ�Œ Meta-learning ensemble failed: {error_msg}")
            raise
    
    def apply_post_processing(self, predictions: np.ndarray, 
                            base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Apply sophisticated post-processing"""
        # Calculate bounds from base predictions
        all_base = np.concatenate(list(base_predictions.values()))
        
        # Use robust percentiles
        lower_bound = np.percentile(all_base, 0.1)
        upper_bound = np.percentile(all_base, 99.9)
        
        # Clip predictions
        clipped = np.clip(predictions, lower_bound, upper_bound)
        
        # Smooth extreme outliers
        z_scores = np.abs((clipped - np.median(clipped)) / (1.4826 * np.median(np.abs(clipped - np.median(clipped)))))
        outlier_mask = z_scores > 3.5
        
        if np.sum(outlier_mask) > 0:
            print(f"\nPost-processing {np.sum(outlier_mask)} outliers")
            
            # Use robust location estimate
            robust_center = np.median(clipped[~outlier_mask])
            clipped[outlier_mask] = 0.7 * robust_center + 0.3 * clipped[outlier_mask]
        
        return clipped

# =============================================================================
# Main Execution
# =============================================================================

def create_meta_learning_ensemble():
    """Create the final ensemble using meta-learning techniques"""
    print("\nDRW Crypto Market Prediction - Meta-Learning Ensemble")
    print("="*80)
    
    try:
        # Clean memory
        aggressive_memory_cleanup()
        
        # Choose strategies
        strategies = ['stacking', 'dynamic', 'bayesian', 'attention']
        
        # Create ensemble
        ensemble = AdvancedMetaLearningEnsemble(strategies=strategies)
        
        # Run pipeline
        final_submission = ensemble.run()
        
        return final_submission
        
    except Exception as e:
        print(f"\nError: {e}")
        raise

# Entry point
if __name__ == "__main__":
    final_submission = create_meta_learning_ensemble()
    
    # Display summary
    print("\n" + "="*80)
    print("Pipeline Execution Summary:")
    print(global_config.get_execution_summary())
    print("="*80)


# # Final Ensemble Building
# # !/usr/bin/env python
# # -*- coding: utf-8 -*-
# """
# DRW Crypto Market Prediction - Final Ensemble Builder
# This module creates the final ensemble from previously executed individual model predictions
# Currently configured to work with XGBoost and AutoEncoder models
# """

# import subprocess
# import sys
# import os
# import gc
# import warnings
# import json
# import pandas as pd
# import numpy as np
# from scipy.stats import pearsonr
# from typing import List, Dict, Tuple, Optional, Any
# from pathlib import Path
# from sklearn.model_selection import KFold, GridSearchCV
# from sklearn.linear_model import Ridge
# import matplotlib.pyplot as plt
# import seaborn as sns

# warnings.filterwarnings('ignore')

# # Install required packages for ensemble building
# print("Installing packages for Final Ensemble Builder...")
# packages_to_install = [
#     'flaml==2.1.1',
#     'matplotlib',
#     'seaborn'
# ]

# for package in packages_to_install:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--quiet"])

# class AdvancedFinalEnsembleBuilder:
#     """Advanced ensemble builder with stacking, cross-validation, and AutoML options"""
    
#     def __init__(self, use_automl: bool = True):
#         self.use_automl = use_automl
        
#         # Model configuration
#         self.model_name = "final_ensemble"
#         self.model_directory = os.path.join(global_config.base_dir, "final_ensemble")
        
#         # Register with global configuration
#         global_config.register_model(self.model_name, self.model_directory)
        
#         # Output paths
#         self.final_submission_path = "/kaggle/working/final_submission.csv"
#         self.ensemble_analysis_path = os.path.join(self.model_directory, "ensemble_analysis.csv")
#         self.model_correlations_path = os.path.join(self.model_directory, "model_correlations.csv")
#         self.ensemble_weights_path = os.path.join(self.model_directory, "ensemble_weights.json")
#         self.visualization_path = os.path.join(self.model_directory, "ensemble_analysis.png")
        
#         # Ensemble parameters
#         self.n_folds = 5
#         self.random_state = 42
#         self.test_size = 0.2
        
#         # Ensure directories exist
#         Path(self.model_directory).mkdir(parents=True, exist_ok=True)
    
#     def load_available_submissions(self) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray]]:
#         """Load submissions from successfully completed models using the global registry"""
#         submissions = {}
#         predictions = {}
        
#         print("\nSearching for model submissions in global registry...")
        
#         # Define target models and their expected submission names
#         target_models = {
#             'xgboost': ['submission', 'xgboost_submission'],
#             'autoencoder_simple': ['ensemble_simple_submission', 'autoencoder_simple'],
#             'autoencoder_weighted': ['ensemble_weighted_submission', 'autoencoder_weighted']
#         }
        
#         # First attempt: Check the global registry for registered outputs
#         for model_name, possible_output_names in target_models.items():
#             if model_name in global_config.model_registry:
#                 model_record = global_config.model_registry[model_name]
                
#                 # Check if model completed successfully
#                 if model_record.status == 'completed':
#                     # Look for submission files in the model's outputs
#                     submission_found = False
#                     for output_name in possible_output_names:
#                         if output_name in model_record.outputs:
#                             output = model_record.outputs[output_name]
#                             if os.path.exists(output.file_path):
#                                 try:
#                                     submission_df = pd.read_csv(output.file_path)
#                                     submissions[model_name] = submission_df
#                                     predictions[model_name] = submission_df.iloc[:, 1].values
#                                     print(f"  âœ“ Loaded {model_name} from registry: {output.file_path}")
#                                     submission_found = True
#                                     break
#                                 except Exception as e:
#                                     print(f"  âœ— Error loading {model_name} from {output.file_path}: {e}")
                    
#                     if not submission_found:
#                         print(f"  âš  {model_name} completed but no submission found in outputs")
#                 else:
#                     print(f"  âš  {model_name} status: {model_record.status}")
#             else:
#                 print(f"  âš  {model_name} not found in registry")
        
#         # Fallback: Check standard file paths if registry is incomplete
#         fallback_paths = {
#             'xgboost': os.path.join(global_config.base_dir, "triple_xgboost", "submission.csv"),
#             'autoencoder_simple': os.path.join(global_config.base_dir, "autoencoder_deepmlp", "ensemble_simple_submission.csv"),
#             'autoencoder_weighted': os.path.join(global_config.base_dir, "autoencoder_deepmlp", "ensemble_weighted_submission.csv")
#         }
        
#         for model_name, submission_path in fallback_paths.items():
#             if model_name not in submissions and os.path.exists(submission_path):
#                 try:
#                     submission_df = pd.read_csv(submission_path)
#                     submissions[model_name] = submission_df
#                     predictions[model_name] = submission_df.iloc[:, 1].values
#                     print(f"  âœ“ Loaded {model_name} from fallback path: {submission_path}")
#                 except Exception as e:
#                     print(f"  âœ— Error loading {model_name} from fallback: {e}")
        
#         # Validate loaded predictions
#         print(f"\nLoaded {len(submissions)} model submissions:")
#         for model_name, preds in predictions.items():
#             print(f"  â€¢ {model_name}: {len(preds)} predictions, range [{preds.min():.4f}, {preds.max():.4f}]")
        
#         return submissions, predictions
    
#     def analyze_model_diversity(self, predictions: Dict[str, np.ndarray]) -> pd.DataFrame:
#         """Analyze correlation between model predictions to ensure diversity"""
#         model_names = list(predictions.keys())
#         n_models = len(model_names)
        
#         if n_models < 2:
#             print("Warning: Insufficient models for diversity analysis")
#             return pd.DataFrame()
        
#         correlation_matrix = np.zeros((n_models, n_models))
        
#         for i, model1 in enumerate(model_names):
#             for j, model2 in enumerate(model_names):
#                 if i <= j:
#                     corr = pearsonr(predictions[model1], predictions[model2])[0]
#                     correlation_matrix[i, j] = corr
#                     correlation_matrix[j, i] = corr
        
#         corr_df = pd.DataFrame(
#             correlation_matrix,
#             index=model_names,
#             columns=model_names
#         )
        
#         # Save correlation matrix
#         corr_df.to_csv(self.model_correlations_path)
#         global_config.register_model_output(
#             self.model_name, 
#             'model_correlations', 
#             self.model_correlations_path,
#             'analysis'
#         )
        
#         # Display correlation matrix
#         print("\nModel Correlation Matrix:")
#         print(corr_df.round(4))
        
#         # Calculate diversity metrics
#         off_diagonal_corr = correlation_matrix[np.triu_indices(n_models, k=1)]
        
#         print(f"\nDiversity Metrics:")
#         print(f"  Average inter-model correlation: {np.mean(off_diagonal_corr):.4f}")
#         if len(off_diagonal_corr) > 0:
#             print(f"  Correlation range: [{np.min(off_diagonal_corr):.4f}, {np.max(off_diagonal_corr):.4f}]")
        
#         return corr_df
    
#     def create_stacking_features(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
#         """Create comprehensive feature matrix for stacking ensemble"""
#         feature_list = []
#         feature_names = []
        
#         # Add base predictions
#         for model_name, preds in predictions.items():
#             feature_list.append(preds)
#             feature_names.append(model_name)
        
#         # Add interaction features between different model types
#         model_names = list(predictions.keys())
#         for i in range(len(model_names)):
#             for j in range(i + 1, len(model_names)):
#                 interaction = predictions[model_names[i]] * predictions[model_names[j]]
#                 feature_list.append(interaction)
#                 feature_names.append(f"{model_names[i]}*{model_names[j]}")
        
#         # Statistical features across all predictions
#         predictions_array = np.column_stack(list(predictions.values()))
        
#         # Basic statistics
#         feature_list.extend([
#             np.mean(predictions_array, axis=1),
#             np.std(predictions_array, axis=1),
#             np.min(predictions_array, axis=1),
#             np.max(predictions_array, axis=1),
#             np.median(predictions_array, axis=1)
#         ])
#         feature_names.extend(['mean', 'std', 'min', 'max', 'median'])
        
#         # Advanced statistics
#         feature_list.extend([
#             np.max(predictions_array, axis=1) - np.min(predictions_array, axis=1),  # Range
#             np.where(np.mean(predictions_array, axis=1) != 0, 
#                     np.std(predictions_array, axis=1) / np.abs(np.mean(predictions_array, axis=1)), 
#                     0)  # Coefficient of variation
#         ])
#         feature_names.extend(['range', 'cv'])
        
#         # Stack all features
#         feature_matrix = np.column_stack(feature_list)
        
#         print(f"\nCreated {feature_matrix.shape[1]} stacking features: {', '.join(feature_names)}")
        
#         return feature_matrix
    
#     def train_automl_ensemble(self, X: np.ndarray, y: np.ndarray) -> Tuple[Any, float]:
#         """Train ensemble using FLAML AutoML"""
#         try:
#             from flaml import AutoML
            
#             print("\nTraining FLAML AutoML ensemble...")
            
#             # Split for validation
#             split_idx = int(len(X) * 0.8)
#             X_train, X_val = X[:split_idx], X[split_idx:]
#             y_train, y_val = y[:split_idx], y[split_idx:]
            
#             # Configure AutoML
#             automl = AutoML()
            
#             automl_settings = {
#                 "time_budget": 120,  # 2 minutes
#                 "metric": 'r2',
#                 "task": 'regression',
#                 "n_jobs": -1,
#                 "estimator_list": ['rf', 'xgboost', 'lgbm', 'lrl1', 'lrl2'],
#                 "seed": self.random_state,
#                 "verbose": 0,
#                 "eval_method": "cv",
#                 "n_splits": 3
#             }
            
#             # Train AutoML
#             automl.fit(X_train, y_train, **automl_settings)
            
#             # Evaluate
#             val_pred = automl.predict(X_val)
#             val_score = pearsonr(y_val, val_pred)[0]
            
#             print(f"  FLAML selected model: {automl.best_estimator}")
#             print(f"  Validation correlation: {val_score:.6f}")
            
#             return automl, val_score
            
#         except Exception as e:
#             print(f"  AutoML training failed: {e}")
#             return None, 0.0
    
#     def train_ridge_ensemble(self, X: np.ndarray, y: np.ndarray) -> Tuple[Ridge, float]:
#         """Train Ridge regression ensemble with hyperparameter optimization"""
#         print("\nTraining Ridge regression ensemble...")
        
#         # Split for validation
#         split_idx = int(len(X) * 0.8)
#         X_train, X_val = X[:split_idx], X[split_idx:]
#         y_train, y_val = y[:split_idx], y[split_idx:]
        
#         # Test multiple alpha values
#         alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
#         best_score = -float('inf')
#         best_model = None
#         best_alpha = None
        
#         for alpha in alphas:
#             model = Ridge(alpha=alpha, random_state=self.random_state)
#             model.fit(X_train, y_train)
            
#             val_pred = model.predict(X_val)
#             score = pearsonr(y_val, val_pred)[0]
            
#             if score > best_score:
#                 best_score = score
#                 best_model = model
#                 best_alpha = alpha
        
#         print(f"  Best alpha: {best_alpha}")
#         print(f"  Validation correlation: {best_score:.6f}")
        
#         return best_model, best_score
    
#     def create_ensemble(self, predictions: Dict[str, np.ndarray]) -> np.ndarray:
#         """Create the final ensemble using the best available method"""
#         # Create feature matrix
#         X = self.create_stacking_features(predictions)
        
#         # Create synthetic target based on weighted average
#         # Weight based on individual model performance from registry
#         weights = {}
#         for model_name in predictions.keys():
#             if model_name in global_config.model_registry:
#                 model_score = global_config.model_registry[model_name].score
#                 if model_score is not None and model_score > 0:
#                     weights[model_name] = model_score
#                 else:
#                     weights[model_name] = 0.1  # Default small weight
#             else:
#                 weights[model_name] = 0.1
        
#         # Normalize weights
#         total_weight = sum(weights.values())
#         weights = {k: v/total_weight for k, v in weights.items()}
        
#         print(f"\nModel weights based on individual performance:")
#         for model_name, weight in weights.items():
#             print(f"  â€¢ {model_name}: {weight:.3f}")
        
#         # Create weighted synthetic target
#         y_synthetic = np.zeros(len(next(iter(predictions.values()))))
#         for model_name, preds in predictions.items():
#             y_synthetic += weights[model_name] * preds
        
#         # Add small noise for training stability
#         np.random.seed(self.random_state)
#         noise = np.random.normal(0, np.std(y_synthetic) * 0.02, size=len(y_synthetic))
#         y_synthetic = y_synthetic + noise
        
#         best_model = None
#         best_score = -float('inf')
#         best_method = None
        
#         # Try AutoML if enabled
#         if self.use_automl:
#             automl_model, automl_score = self.train_automl_ensemble(X, y_synthetic)
#             if automl_model is not None and automl_score > best_score:
#                 best_model = automl_model
#                 best_score = automl_score
#                 best_method = 'AutoML'
        
#         # Always try Ridge as baseline
#         ridge_model, ridge_score = self.train_ridge_ensemble(X, y_synthetic)
#         if ridge_score > best_score or best_model is None:
#             best_model = ridge_model
#             best_score = ridge_score
#             best_method = 'Ridge'
        
#         print(f"\nSelected ensemble method: {best_method} (score: {best_score:.6f})")
        
#         # Make final predictions
#         final_predictions = best_model.predict(X)
        
#         # Save ensemble information
#         ensemble_info = {
#             'method': best_method,
#             'validation_score': float(best_score),
#             'n_models': len(predictions),
#             'models_used': list(predictions.keys()),
#             'model_weights': weights
#         }
        
#         with open(self.ensemble_weights_path, 'w') as f:
#             json.dump(ensemble_info, f, indent=2)
        
#         global_config.register_model_output(
#             self.model_name,
#             'ensemble_weights',
#             self.ensemble_weights_path,
#             'config',
#             metadata=ensemble_info
#         )
        
#         return final_predictions
    
#     def apply_post_processing(self, predictions: np.ndarray, 
#                             base_predictions: Dict[str, np.ndarray]) -> np.ndarray:
#         """Apply post-processing to ensure prediction quality"""
#         # Calculate reasonable bounds from base predictions
#         all_base_preds = np.concatenate(list(base_predictions.values()))
        
#         lower_bound = np.percentile(all_base_preds, 0.5)
#         upper_bound = np.percentile(all_base_preds, 99.5)
        
#         # Clip extreme predictions
#         clipped_predictions = np.clip(predictions, lower_bound, upper_bound)
        
#         # Identify and smooth outliers
#         pred_mean = np.mean(clipped_predictions)
#         pred_std = np.std(clipped_predictions)
#         z_scores = np.abs((clipped_predictions - pred_mean) / (pred_std + 1e-8))
        
#         outlier_mask = z_scores > 3
#         n_outliers = np.sum(outlier_mask)
        
#         if n_outliers > 0:
#             print(f"\nPost-processing: Adjusting {n_outliers} outliers ({n_outliers/len(predictions)*100:.2f}%)")
            
#             # Blend outliers toward the mean
#             blend_factor = 0.7  # 70% mean, 30% original
#             clipped_predictions[outlier_mask] = (
#                 blend_factor * pred_mean + 
#                 (1 - blend_factor) * clipped_predictions[outlier_mask]
#             )
        
#         return clipped_predictions
    
#     def create_visualization(self, final_submission: pd.DataFrame, 
#                            predictions: Dict[str, np.ndarray]):
#         """Create comprehensive visualization of ensemble results"""
#         try:
#             fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
#             # Plot 1: Individual model predictions
#             ax1 = axes[0, 0]
#             for model_name, preds in predictions.items():
#                 ax1.hist(preds, bins=50, alpha=0.6, label=model_name, density=True)
#             ax1.set_xlabel('Prediction Value')
#             ax1.set_ylabel('Density')
#             ax1.set_title('Distribution of Individual Model Predictions')
#             ax1.legend()
#             ax1.grid(True, alpha=0.3)
            
#             # Plot 2: Final ensemble distribution
#             ax2 = axes[0, 1]
#             final_preds = final_submission.iloc[:, 1].values
#             ax2.hist(final_preds, bins=50, color='darkgreen', alpha=0.8, density=True, edgecolor='black')
#             ax2.axvline(np.mean(final_preds), color='red', linestyle='--', label=f'Mean: {np.mean(final_preds):.4f}')
#             ax2.set_xlabel('Prediction Value')
#             ax2.set_ylabel('Density')
#             ax2.set_title('Final Ensemble Predictions')
#             ax2.legend()
#             ax2.grid(True, alpha=0.3)
            
#             # Plot 3: Model agreement
#             ax3 = axes[1, 0]
#             predictions_array = np.column_stack(list(predictions.values()))
#             model_std = np.std(predictions_array, axis=1)
#             ax3.hist(model_std, bins=50, color='orange', alpha=0.8, edgecolor='black')
#             ax3.axvline(np.mean(model_std), color='red', linestyle='--', label=f'Mean Std: {np.mean(model_std):.4f}')
#             ax3.set_xlabel('Standard Deviation Across Models')
#             ax3.set_ylabel('Count')
#             ax3.set_title('Model Agreement Analysis')
#             ax3.legend()
#             ax3.grid(True, alpha=0.3)
            
#             # Plot 4: Summary statistics
#             ax4 = axes[1, 1]
#             ax4.axis('off')
            
#             stats_text = f"""Ensemble Summary Statistics
            
# Final Predictions:
#   â€¢ Mean: {np.mean(final_preds):.6f}
#   â€¢ Std Dev: {np.std(final_preds):.6f}
#   â€¢ Range: [{np.min(final_preds):.6f}, {np.max(final_preds):.6f}]
#   â€¢ Median: {np.median(final_preds):.6f}

# Model Agreement:
#   â€¢ Average Std: {np.mean(model_std):.6f}
#   â€¢ Max Disagreement: {np.max(model_std):.6f}
#   â€¢ Models Used: {len(predictions)}"""
            
#             ax4.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center', 
#                     fontfamily='monospace', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray"))
            
#             plt.suptitle('DRW Crypto Prediction - Ensemble Analysis', fontsize=14, fontweight='bold')
#             plt.tight_layout()
#             plt.savefig(self.visualization_path, dpi=150, bbox_inches='tight')
#             plt.close()
            
#             print(f"Visualization saved to: {self.visualization_path}")
            
#             global_config.register_model_output(
#                 self.model_name,
#                 'visualization',
#                 self.visualization_path,
#                 'visualization'
#             )
            
#         except Exception as e:
#             print(f"Warning: Could not create visualization: {e}")
    
#     def run(self) -> pd.DataFrame:
#         """Execute the complete ensemble building process"""
#         print("\nBuilding Advanced Ensemble")
#         print("="*80)
        
#         # Update status
#         global_config.update_model_status(self.model_name, 'running')
        
#         # Load available submissions
#         submissions, predictions = self.load_available_submissions()
        
#         if len(submissions) < 2:
#             # Try single model if available
#             if len(submissions) == 1:
#                 print("\nWarning: Only one model available, using single model predictions")
#                 template = next(iter(submissions.values()))
#                 final_submission = template.copy()
#                 final_submission.to_csv(self.final_submission_path, index=False)
                
#                 global_config.register_model_output(
#                     self.model_name,
#                     'final_submission',
#                     self.final_submission_path,
#                     'submission'
#                 )
                
#                 global_config.update_model_status(self.model_name, 'completed')
#                 return final_submission
#             else:
#                 error_msg = f"No models available for ensemble"
#                 global_config.update_model_status(self.model_name, 'failed', error_message=error_msg)
#                 raise ValueError(error_msg)
        
#         print(f"\nSuccessfully loaded {len(submissions)} model submissions for ensemble")
        
#         # Analyze model diversity
#         correlation_df = self.analyze_model_diversity(predictions)
        
#         # Create ensemble
#         print("\nCreating ensemble predictions...")
#         template_submission = next(iter(submissions.values()))
#         ensemble_predictions = self.create_ensemble(predictions)
        
#         # Apply post-processing
#         final_predictions = self.apply_post_processing(ensemble_predictions, predictions)
        
#         # Create final submission
#         final_submission = template_submission.copy()
#         final_submission.iloc[:, 1] = final_predictions
        
#         # Save results
#         final_submission.to_csv(self.final_submission_path, index=False)
#         print(f"\nFinal submission saved to: {self.final_submission_path}")
        
#         global_config.register_model_output(
#             self.model_name,
#             'final_submission',
#             self.final_submission_path,
#             'submission'
#         )
        
#         # Create analysis dataframe
#         analysis_df = pd.DataFrame({
#             'final_ensemble': final_predictions,
#             'pre_postprocess': ensemble_predictions,
#             **predictions
#         })
#         analysis_df.to_csv(self.ensemble_analysis_path, index=False)
        
#         global_config.register_model_output(
#             self.model_name,
#             'ensemble_analysis',
#             self.ensemble_analysis_path,
#             'analysis'
#         )
        
#         # Create visualization
#         self.create_visualization(final_submission, predictions)
        
#         # Update status
#         global_config.update_model_status(self.model_name, 'completed')
        
#         print("\nEnsemble building completed successfully")
        
#         return final_submission

# # Main execution function
# def create_final_ensemble():
#     """Create the final ensemble from completed model predictions"""
#     print("\nDRW Crypto Market Prediction - Final Ensemble Builder")
#     print("="*80)
    
#     try:
#         # Clean memory before starting
#         aggressive_memory_cleanup()
        
#         # Configure ensemble builder
#         use_automl = True  # Set to False to use only Ridge regression
#         ensemble_builder = AdvancedFinalEnsembleBuilder(use_automl=use_automl)
        
#         # Build ensemble
#         final_submission = ensemble_builder.run()
        
#         print("\nâœ… Final ensemble creation completed successfully")
        
#         return final_submission
        
#     except Exception as e:
#         error_msg = str(e)
#         print(f"\nâ�Œ Ensemble creation failed: {error_msg}")
#         raise

# # Main entry point
# if __name__ == "__main__":
#     final_submission = create_final_ensemble()
    
#     # Display final execution summary
#     print("\n" + "="*80)
#     print("Pipeline Execution Summary:")
#     print(global_config.get_execution_summary())
#     print("="*80)

