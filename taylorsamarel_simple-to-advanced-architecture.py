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


#!/usr/bin/env python3

# Install required packages with specific versions to avoid conflicts
!pip install pandas numpy scikit-learn==1.3.2 imbalanced-learn psutil lightgbm catboost xgboost -q

import pandas as pd
import numpy as np
import gc
import psutil
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple, Type
import hashlib
import json
import itertools
import random
from collections import defaultdict, Counter
from datetime import datetime

# Core imports
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
from sklearn.decomposition import PCA, TruncatedSVD, FastICA, IncrementalPCA
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif, VarianceThreshold, RFE
from sklearn.ensemble import (
    RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier,
    VotingClassifier, StackingClassifier, HistGradientBoostingClassifier, IsolationForest
)
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import f1_score, make_scorer, precision_score, recall_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.impute import SimpleImputer
from sklearn.cluster import MiniBatchKMeans
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import BalancedRandomForestClassifier, BalancedBaggingClassifier

# Advanced model imports
import lightgbm as lgb
from catboost import CatBoostClassifier

import warnings
warnings.filterwarnings('ignore')

# Enhanced configuration
SKIP_HIGH_MEMORY_OPERATIONS = True
MEMORY_THRESHOLD_MB = 12000
MAX_CONFIGURATIONS_TO_TEST = 15
EXPLORATION_RATE = 0.3  
SUBMISSION_DIR = '/kaggle/working/submissions'
ENSEMBLE_TOP_N = 5  

# CRITICAL IMPROVEMENT: Threshold for detecting reversals
REVERSAL_CONFIDENCE_THRESHOLD = {
    'H': 0.20,  # Need 20% confidence to predict High reversal
    'L': 0.20,  # Need 20% confidence to predict Low reversal
}

if not os.path.exists(SUBMISSION_DIR):
    os.makedirs(SUBMISSION_DIR)

# ========== ENHANCED TIME SERIES FEATURE ENGINEERING ==========

def create_advanced_time_series_features(X, y=None, ticker_ids=None, timestamps=None, 
                                        lookback_periods=[5, 10, 20, 50], 
                                        max_features_to_process=50):
    """
    Create comprehensive time series features for reversal detection
    Including pattern tracking, momentum indicators, and regime detection
    """
    
    n_samples, n_features = X.shape
    feature_list = []
    
    print(f"  Creating time series features for {n_samples} samples with {n_features} features...")
    
    # Convert timestamps to datetime if provided
    if timestamps is not None:
        dates = pd.to_datetime(timestamps)
        # Extract temporal features
        time_features = np.column_stack([
            dates.dayofweek.values.astype(np.float32),  # Day of week pattern
            dates.day.values.astype(np.float32),         # Day of month
            dates.month.values.astype(np.float32),       # Seasonality
            (dates.dayofyear.values % 91).astype(np.float32),  # Quarter position
        ])
        feature_list.append(time_features)
    
    if ticker_ids is not None:
        unique_tickers = np.unique(ticker_ids)
        print(f"  Processing {len(unique_tickers)} tickers...")
        
        all_ticker_features = []
        
        for ticker_idx, ticker in enumerate(unique_tickers):
            ticker_mask = ticker_ids == ticker
            ticker_indices = np.where(ticker_mask)[0]
            ticker_data = X[ticker_mask]
            n_ticker_samples = len(ticker_data)
            
            if n_ticker_samples < 20:
                continue
            
            # Process only the most important features to save memory
            num_features_to_process = min(max_features_to_process, n_features)
            
            # Calculate feature importance using variance
            feature_variances = np.var(ticker_data[:, :num_features_to_process], axis=0)
            important_features = np.argsort(feature_variances)[-num_features_to_process:]
            
            ticker_feature_arrays = []
            
            for feat_idx in important_features[:20]:  # Limit to top 20 features per ticker
                series = ticker_data[:, feat_idx].astype(np.float32)
                
                # === ROLLING STATISTICS ===
                for window in [5, 10, 20]:  # Reduced lookback periods
                    if window < n_ticker_samples:
                        # Rolling mean
                        roll_mean = pd.Series(series).rolling(window, min_periods=1).mean().values.astype(np.float32)
                        
                        # Rolling std (volatility)
                        roll_std = pd.Series(series).rolling(window, min_periods=1).std().fillna(0).values.astype(np.float32)
                        
                        # Distance from rolling mean (z-score)
                        z_score = np.where(roll_std != 0, 
                                         (series - roll_mean) / (roll_std + 1e-10), 
                                         0).astype(np.float32)
                        
                        # Rolling min/max
                        roll_min = pd.Series(series).rolling(window, min_periods=1).min().values.astype(np.float32)
                        roll_max = pd.Series(series).rolling(window, min_periods=1).max().values.astype(np.float32)
                        
                        # Position in range (0 to 1)
                        range_val = roll_max - roll_min
                        range_pos = np.where(range_val != 0, 
                                           (series - roll_min) / (range_val + 1e-10), 
                                           0.5).astype(np.float32)
                        
                        ticker_feature_arrays.extend([z_score, range_pos])
                
                # === MOMENTUM INDICATORS ===
                # Rate of change
                for lag in [1, 5]:
                    if lag < n_ticker_samples:
                        roc = np.zeros(n_ticker_samples, dtype=np.float32)
                        roc[lag:] = (series[lag:] - series[:-lag]) / (np.abs(series[:-lag]) + 1e-10)
                        ticker_feature_arrays.append(roc)
                
                # === PATTERN DETECTION ===
                # Simple peak and trough detection
                peaks = np.zeros(n_ticker_samples, dtype=np.float32)
                troughs = np.zeros(n_ticker_samples, dtype=np.float32)
                
                for i in range(2, n_ticker_samples - 2):
                    # Peak: higher than neighbors
                    if (series[i] > series[i-1] and series[i] > series[i+1] and 
                        series[i] > series[i-2] and series[i] > series[i+2]):
                        peaks[i] = 1.0
                    # Trough: lower than neighbors
                    elif (series[i] < series[i-1] and series[i] < series[i+1] and 
                          series[i] < series[i-2] and series[i] < series[i+2]):
                        troughs[i] = 1.0
                
                # Distance from last peak/trough
                dist_from_peak = np.full(n_ticker_samples, fill_value=n_ticker_samples, dtype=np.float32)
                dist_from_trough = np.full(n_ticker_samples, fill_value=n_ticker_samples, dtype=np.float32)
                
                peak_indices = np.where(peaks == 1)[0]
                trough_indices = np.where(troughs == 1)[0]
                
                for i in range(n_ticker_samples):
                    if len(peak_indices) > 0:
                        past_peaks = peak_indices[peak_indices <= i]
                        if len(past_peaks) > 0:
                            dist_from_peak[i] = i - past_peaks[-1]
                    
                    if len(trough_indices) > 0:
                        past_troughs = trough_indices[trough_indices <= i]
                        if len(past_troughs) > 0:
                            dist_from_trough[i] = i - past_troughs[-1]
                
                ticker_feature_arrays.extend([peaks, troughs, 
                                            dist_from_peak / n_ticker_samples,  # Normalize
                                            dist_from_trough / n_ticker_samples])
                
                # === CONSECUTIVE MOVEMENTS ===
                consecutive_ups = np.zeros(n_ticker_samples, dtype=np.float32)
                consecutive_downs = np.zeros(n_ticker_samples, dtype=np.float32)
                
                up_count = 0
                down_count = 0
                for i in range(1, n_ticker_samples):
                    if series[i] > series[i-1]:
                        up_count += 1
                        down_count = 0
                    else:
                        down_count += 1
                        up_count = 0
                    consecutive_ups[i] = up_count
                    consecutive_downs[i] = down_count
                
                ticker_feature_arrays.extend([consecutive_ups / 10,  # Normalize
                                             consecutive_downs / 10])
            
            # Stack all features for this ticker
            if ticker_feature_arrays:
                ticker_feature_matrix = np.column_stack(ticker_feature_arrays)
                
                # Map back to full dataset
                full_ticker_features = np.zeros((n_samples, ticker_feature_matrix.shape[1]), dtype=np.float32)
                full_ticker_features[ticker_mask] = ticker_feature_matrix
                all_ticker_features.append(full_ticker_features)
        
        # Combine all ticker features
        if all_ticker_features:
            combined_ticker_features = np.hstack(all_ticker_features)
            feature_list.append(combined_ticker_features)
            print(f"  Added {combined_ticker_features.shape[1]} ticker-based time series features")
    
    # Global time series features (if no ticker info or as supplement)
    else:
        print("  Creating global time series features...")
        global_features = []
        
        for feat_idx in range(min(20, n_features)):  # Process first 20 features
            series = X[:, feat_idx].astype(np.float32)
            
            # Simple differences
            diff_1 = np.diff(series, prepend=series[0])
            diff_2 = np.diff(diff_1, prepend=diff_1[0])
            
            # Rolling statistics
            roll_mean_5 = pd.Series(series).rolling(5, min_periods=1).mean().values.astype(np.float32)
            roll_std_5 = pd.Series(series).rolling(5, min_periods=1).std().fillna(0).values.astype(np.float32)
            
            global_features.extend([
                diff_1.reshape(-1, 1),
                diff_2.reshape(-1, 1),
                (series - roll_mean_5).reshape(-1, 1),
                roll_std_5.reshape(-1, 1)
            ])
        
        if global_features:
            feature_list.append(np.hstack(global_features))
    
    # Combine all features with original
    if feature_list:
        new_features = np.hstack(feature_list)
        X_enhanced = np.hstack([X, new_features])
        print(f"  Final shape: {X_enhanced.shape} (added {new_features.shape[1]} features)")
        return X_enhanced
    else:
        return X

# Memory Manager Class
class MemoryManager:
    @staticmethod
    def get_memory_usage():
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def get_available_memory():
        return psutil.virtual_memory().available / 1024 / 1024
    
    @staticmethod
    def get_total_memory():
        return psutil.virtual_memory().total / 1024 / 1024
    
    @staticmethod
    def estimate_array_memory(shape, dtype=np.float64):
        bytes_per_element = np.dtype(dtype).itemsize
        total_bytes = np.prod(shape) * bytes_per_element
        return total_bytes / 1024 / 1024
    
    @staticmethod
    def clean_memory():
        gc.collect()
    
    @staticmethod
    def check_memory_for_operation(estimated_mb, safety_factor=2.0):
        available = MemoryManager.get_available_memory()
        required = estimated_mb * safety_factor
        return available > required, available, required

# Data classes
@dataclass
class ProcessingData:
    X: np.ndarray
    y: np.ndarray
    feature_names: Optional[List[str]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def copy(self):
        return ProcessingData(
            X=self.X.copy(),
            y=self.y.copy() if self.y is not None else None,
            feature_names=self.feature_names.copy() if self.feature_names else None,
            metadata=self.metadata.copy()
        )
    
    def get_memory_usage(self):
        x_memory = self.X.nbytes / 1024 / 1024
        y_memory = self.y.nbytes / 1024 / 1024 if self.y is not None else 0
        return x_memory + y_memory

@dataclass
class StepConfig:
    step_type: str
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    memory_limit_mb: Optional[float] = None
    
    def to_dict(self):
        return {
            'type': self.step_type,
            'method': self.method,
            'params': self.params,
            'enabled': self.enabled,
            'memory_limit_mb': self.memory_limit_mb
        }
    
    def __hash__(self):
        return hash((self.step_type, self.method, json.dumps(self.params, sort_keys=True)))

@dataclass
class PipelineConfig:
    steps: List[StepConfig]
    config_id: Optional[str] = None
    skip_high_memory: bool = True
    memory_threshold_mb: float = 1000
    model_type: str = 'logistic'
    model_params: Dict[str, Any] = field(default_factory=dict)
    
    def generate_id(self):
        config_str = json.dumps([s.to_dict() for s in self.steps] + [self.model_type, self.model_params], sort_keys=True)
        self.config_id = hashlib.md5(config_str.encode()).hexdigest()[:12]
        return self.config_id
    
    def __str__(self):
        return ' -> '.join([f"{s.step_type}:{s.method}" for s in self.steps if s.enabled])

# Base preprocessing classes
class BasePreprocessingStep(ABC):
    STEP_TYPE = None
    METHOD_NAME = None
    
    def __init__(self, config: StepConfig):
        self.config = config
        self.is_fitted = False
        self._fitted_params = {}
    
    def estimate_memory_requirement(self, data: ProcessingData) -> float:
        return data.get_memory_usage()
    
    @abstractmethod
    def fit(self, data: ProcessingData) -> 'BasePreprocessingStep':
        pass
    
    @abstractmethod
    def transform(self, data: ProcessingData) -> ProcessingData:
        pass
    
    def fit_transform(self, data: ProcessingData) -> ProcessingData:
        self.fit(data)
        return self.transform(data)

class StepRegistry:
    _registry = defaultdict(dict)
    
    @classmethod
    def register(cls, step_class: Type[BasePreprocessingStep]):
        if step_class.STEP_TYPE and step_class.METHOD_NAME:
            cls._registry[step_class.STEP_TYPE][step_class.METHOD_NAME] = step_class
    
    @classmethod
    def get_step_class(cls, step_type: str, method: str) -> Optional[Type[BasePreprocessingStep]]:
        return cls._registry.get(step_type, {}).get(method)
    
    @classmethod
    def get_available_steps(cls) -> Dict[str, List[str]]:
        return {step_type: list(methods.keys()) 
                for step_type, methods in cls._registry.items()}
    
    @classmethod
    def create_step(cls, config: StepConfig) -> Optional[BasePreprocessingStep]:
        step_class = cls.get_step_class(config.step_type, config.method)
        if step_class:
            return step_class(config)
        return None

def register_step(cls):
    StepRegistry.register(cls)
    return cls

# Preprocessing step implementations
@register_step
class DowncastStep(BasePreprocessingStep):
    STEP_TYPE = "downcast"
    METHOD_NAME = "auto"
    
    def fit(self, data: ProcessingData) -> 'DowncastStep':
        self.is_fitted = True
        return self
    
    def transform(self, data: ProcessingData) -> ProcessingData:
        result = data.copy()
        original_memory = result.get_memory_usage()
        
        # Convert to float32 for memory efficiency
        if result.X.dtype != np.float32:
            result.X = result.X.astype(np.float32)
        
        new_memory = result.get_memory_usage()
        reduction_pct = (1 - new_memory/original_memory) * 100 if original_memory > 0 else 0
        result.metadata['memory_reduction'] = f"{reduction_pct:.1f}%"
        return result

@register_step
class ImputationStep(BasePreprocessingStep):
    STEP_TYPE = "impute"
    METHOD_NAME = "simple"
    
    def fit(self, data: ProcessingData) -> 'ImputationStep':
        strategy = self.config.params.get('strategy', 'median')
        fill_value = self.config.params.get('fill_value', 0)
        X_clean = data.X.copy()
        X_clean = np.where(np.isinf(X_clean), np.nan, X_clean)
        
        if strategy == 'constant':
            self._fitted_params['imputer'] = SimpleImputer(strategy='constant', fill_value=fill_value)
        else:
            self._fitted_params['imputer'] = SimpleImputer(strategy=strategy)
        
        self._fitted_params['imputer'].fit(X_clean)
        self.is_fitted = True
        return self
    
    def transform(self, data: ProcessingData) -> ProcessingData:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        result = data.copy()
        result.X = np.where(np.isinf(result.X), np.nan, result.X)
        result.X = self._fitted_params['imputer'].transform(result.X)
        result.X = np.nan_to_num(result.X, nan=0.0, posinf=0.0, neginf=0.0)
        return result

@register_step
class StandardScalerStep(BasePreprocessingStep):
    STEP_TYPE = "scaling"
    METHOD_NAME = "standard"
    
    def fit(self, data: ProcessingData) -> 'StandardScalerStep':
        X_float = data.X.astype(np.float32)
        self._fitted_params['scaler'] = StandardScaler()
        self._fitted_params['scaler'].fit(X_float)
        self.is_fitted = True
        return self
    
    def transform(self, data: ProcessingData) -> ProcessingData:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        result = data.copy()
        result.X = result.X.astype(np.float32)
        result.X = self._fitted_params['scaler'].transform(result.X)
        result.X = np.nan_to_num(result.X, nan=0.0, posinf=3.0, neginf=-3.0)
        return result

@register_step
class RobustScalerStep(BasePreprocessingStep):
    STEP_TYPE = "scaling"
    METHOD_NAME = "robust"
    
    def fit(self, data: ProcessingData) -> 'RobustScalerStep':
        X_float = data.X.astype(np.float32)
        self._fitted_params['scaler'] = RobustScaler()
        self._fitted_params['scaler'].fit(X_float)
        self.is_fitted = True
        return self
    
    def transform(self, data: ProcessingData) -> ProcessingData:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        result = data.copy()
        result.X = result.X.astype(np.float32)
        result.X = self._fitted_params['scaler'].transform(result.X)
        result.X = np.nan_to_num(result.X, nan=0.0, posinf=3.0, neginf=-3.0)
        return result

@register_step
class VarianceThresholdReduction(BasePreprocessingStep):
    STEP_TYPE = "dim_reduce"
    METHOD_NAME = "variance_threshold"
    
    def estimate_memory_requirement(self, data: ProcessingData) -> float:
        return MemoryManager.estimate_array_memory((data.X.shape[1],), dtype=np.float32)
    
    def fit(self, data: ProcessingData) -> 'VarianceThresholdReduction':
        threshold = self.config.params.get('threshold', 0.01)
        chunk_size = 1000
        variances = np.zeros(data.X.shape[1], dtype=np.float32)
        
        for i in range(0, data.X.shape[1], chunk_size):
            end_idx = min(i + chunk_size, data.X.shape[1])
            variances[i:end_idx] = np.var(data.X[:, i:end_idx].astype(np.float32), axis=0)
        
        self._fitted_params['keep_indices'] = np.where(variances > threshold)[0]
        
        # Ensure we keep at least some features
        if len(self._fitted_params['keep_indices']) < 100:
            # Keep top 100 by variance
            top_indices = np.argsort(variances)[-min(100, len(variances)):]
            self._fitted_params['keep_indices'] = top_indices
        
        self.is_fitted = True
        return self
    
    def transform(self, data: ProcessingData) -> ProcessingData:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        result = data.copy()
        result.X = result.X[:, self._fitted_params['keep_indices']]
        if result.feature_names:
            result.feature_names = [result.feature_names[i] for i in self._fitted_params['keep_indices']]
        return result

@register_step
class SelectKBestReduction(BasePreprocessingStep):
    STEP_TYPE = "dim_reduce"
    METHOD_NAME = "select_k_best"
    
    def estimate_memory_requirement(self, data: ProcessingData) -> float:
        return data.get_memory_usage() * 0.1
    
    def fit(self, data: ProcessingData) -> 'SelectKBestReduction':
        k = min(self.config.params.get('k', 500), data.X.shape[1])
        score_func = self.config.params.get('score_func', 'f_classif')
        
        if score_func == 'mutual_info':
            func = mutual_info_classif
        else:
            func = f_classif
        
        self._fitted_params['selector'] = SelectKBest(score_func=func, k=k)
        X_float = data.X.astype(np.float32)
        self._fitted_params['selector'].fit(X_float, data.y)
        self.is_fitted = True
        return self
    
    def transform(self, data: ProcessingData) -> ProcessingData:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        result = data.copy()
        result.X = result.X.astype(np.float32)
        result.X = self._fitted_params['selector'].transform(result.X)
        if result.feature_names:
            mask = self._fitted_params['selector'].get_support()
            result.feature_names = [name for name, keep in zip(result.feature_names, mask) if keep]
        return result

# Pipeline class
class MemoryAwarePipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.steps = []
        self.mandatory_steps = []
        self._initialize_steps()
    
    def _initialize_steps(self):
        self.mandatory_steps = [
            DowncastStep(StepConfig('downcast', 'auto')),
            ImputationStep(StepConfig('impute', 'simple', {'strategy': 'median'}))
        ]
        
        for step_config in self.config.steps:
            if step_config.enabled:
                step = StepRegistry.create_step(step_config)
                if step:
                    self.steps.append(step)
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        data = ProcessingData(X=X, y=y, feature_names=feature_names)
        
        for step in self.mandatory_steps:
            step.fit(data)
            data = step.transform(data)
            MemoryManager.clean_memory()
        
        for step in self.steps:
            if self.config.skip_high_memory:
                estimated_mem = step.estimate_memory_requirement(data)
                can_proceed, available, required = MemoryManager.check_memory_for_operation(estimated_mem)
                
                if not can_proceed or estimated_mem > self.config.memory_threshold_mb:
                    continue
            
            try:
                step.fit(data)
                data = step.transform(data)
                MemoryManager.clean_memory()
            except Exception as e:
                print(f"    Warning: Step failed - {e}")
                continue
        
        self._final_shape = data.X.shape
        return self
    
    def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, feature_names: Optional[List[str]] = None):
        data = ProcessingData(X=X, y=y if y is not None else np.zeros(X.shape[0]), feature_names=feature_names)
        
        for step in self.mandatory_steps:
            data = step.transform(data)
        
        for step in self.steps:
            try:
                data = step.transform(data)
            except:
                continue
        
        if y is None:
            return data.X
        return data.X, data.y
    
    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None):
        self.fit(X, y, feature_names)
        return self.transform(X, y, feature_names)

# Grid Search Class
class IntelligentGridSearch:
    def __init__(self, X_train, y_train, X_test=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.exploration_rate = EXPLORATION_RATE
        self.memory_threshold_mb = MEMORY_THRESHOLD_MB
        
        self.results = []
        self.config_performance = {}
        self.step_performance = defaultdict(lambda: defaultdict(list))
        self.all_predictions = {}
        
        self.best_config = None
        self.best_score = -np.inf
        self.best_model = None
        self.best_pipeline = None
        
        # Track class distribution
        unique, counts = np.unique(y_train, return_counts=True)
        self.class_distribution = dict(zip(unique, counts))
        print(f"Training class distribution: {self.class_distribution}")
    
    def generate_memory_efficient_configs(self, n_configs=10):
        configs = []
        available_steps = StepRegistry.get_available_steps()
        
        # Step orderings
        step_orderings = [
            ['dim_reduce', 'scaling'],  
            ['scaling', 'dim_reduce'],
            ['dim_reduce'],
            ['scaling'],
            [],  # No preprocessing
        ]
        
        # Model configurations with adjusted weights
        model_configs = [
            ('logistic', {'C': 0.01, 'solver': 'liblinear', 'weight_factor': 10.0}),
            ('logistic', {'C': 0.1, 'solver': 'liblinear', 'weight_factor': 15.0}),
            ('logistic', {'C': 1.0, 'solver': 'saga', 'weight_factor': 20.0}),
            
            ('random_forest', {'n_estimators': 100, 'max_depth': 5, 'min_samples_split': 50, 'weight_factor': 10.0}),
            ('random_forest', {'n_estimators': 200, 'max_depth': 8, 'min_samples_split': 30, 'weight_factor': 15.0}),
            
            ('lightgbm', {'num_leaves': 15, 'learning_rate': 0.03, 'min_child_samples': 50, 'weight_factor': 10.0}),
            ('lightgbm', {'num_leaves': 31, 'learning_rate': 0.05, 'min_child_samples': 30, 'weight_factor': 15.0}),
            
            ('hist_gradient', {'max_iter': 100, 'max_depth': 5, 'min_samples_leaf': 50, 'weight_factor': 10.0}),
            
            ('isolation_forest', {'contamination': 0.06}),
        ]
        
        for _ in range(n_configs):
            ordering_choice = random.choice(step_orderings)
            model_type, model_params = random.choice(model_configs)
            
            steps = []
            for step_type in ordering_choice:
                if step_type == 'dim_reduce' and 'dim_reduce' in available_steps:
                    method = random.choice(['variance_threshold', 'select_k_best'])
                    if method == 'variance_threshold':
                        params = {'threshold': random.choice([0.0001, 0.001, 0.01])}
                    else:
                        params = {'k': random.choice([500, 1000, 2000]), 'score_func': 'f_classif'}
                    steps.append(StepConfig('dim_reduce', method, params, True, self.memory_threshold_mb))
                    
                elif step_type == 'scaling' and 'scaling' in available_steps:
                    method = random.choice(['standard', 'robust'])
                    steps.append(StepConfig('scaling', method, {}, True, self.memory_threshold_mb))
            
            config = PipelineConfig(
                steps=steps,
                skip_high_memory=True,
                memory_threshold_mb=self.memory_threshold_mb,
                model_type=model_type,
                model_params=model_params
            )
            config.generate_id()
            configs.append(config)
        
        return configs
    
    def create_model(self, model_type, model_params, class_weight_dict):
        """Create model with class weight handling"""
        
        weight_factor = model_params.get('weight_factor', 10.0)
        
        # Adjust weights
        adjusted_weights = {}
        max_class = max(self.class_distribution, key=self.class_distribution.get)
        
        for class_idx in class_weight_dict:
            if class_idx == max_class:
                adjusted_weights[class_idx] = 1.0
            else:
                adjusted_weights[class_idx] = weight_factor
        
        print(f"  Adjusted weights: {adjusted_weights}")
        
        if model_type == 'isolation_forest':
            contamination = model_params.get('contamination', 0.06)
            return IsolationForest(contamination=contamination, random_state=42)
            
        elif model_type == 'lightgbm':
            params = {
                'objective': 'multiclass',
                'num_class': 3,
                'boosting_type': 'gbdt',
                'num_leaves': model_params.get('num_leaves', 31),
                'learning_rate': model_params.get('learning_rate', 0.05),
                'feature_fraction': 0.8,
                'bagging_fraction': 0.7,
                'bagging_freq': 5,
                'n_estimators': model_params.get('n_estimators', 100),
                'class_weight': adjusted_weights,
                'random_state': 42,
                'verbose': -1,
                'min_child_samples': model_params.get('min_child_samples', 30),
                'reg_alpha': 0.1,
                'reg_lambda': 0.1,
            }
            return lgb.LGBMClassifier(**params)
            
        elif model_type == 'random_forest':
            params = {
                'n_estimators': model_params.get('n_estimators', 100),
                'max_depth': model_params.get('max_depth', 10),
                'min_samples_split': model_params.get('min_samples_split', 30),
                'min_samples_leaf': model_params.get('min_samples_leaf', 15),
                'class_weight': adjusted_weights,
                'random_state': 42,
                'n_jobs': -1
            }
            return RandomForestClassifier(**params)
            
        elif model_type == 'hist_gradient':
            params = {
                'max_iter': model_params.get('max_iter', 100),
                'max_depth': model_params.get('max_depth', 5),
                'learning_rate': model_params.get('learning_rate', 0.1),
                'random_state': 42,
                'min_samples_leaf': model_params.get('min_samples_leaf', 30),
            }
            return HistGradientBoostingClassifier(**params)
            
        else:  # logistic
            params = {
                'C': model_params.get('C', 1.0),
                'max_iter': 1000,
                'class_weight': adjusted_weights,
                'solver': model_params.get('solver', 'liblinear'),
                'random_state': 42
            }
            return LogisticRegression(**params)
    
    def apply_probability_threshold(self, probabilities, thresholds=None):
        """Apply probability thresholds to convert predictions"""
        if thresholds is None:
            thresholds = REVERSAL_CONFIDENCE_THRESHOLD
        
        n_samples = probabilities.shape[0]
        predictions = []
        
        for i in range(n_samples):
            probs = probabilities[i]
            
            # Map: 0=H, 1=L, 2=None
            if len(probs) == 3:
                if probs[0] > thresholds.get('H', 0.20):  # H class
                    predictions.append(0)
                elif probs[1] > thresholds.get('L', 0.20):  # L class
                    predictions.append(1)
                else:
                    predictions.append(2)  # None class
            else:
                predictions.append(2)  # Default to None
        
        return np.array(predictions)
    
    def evaluate_config(self, config: PipelineConfig, config_index: int):
        try:
            pipeline = MemoryAwarePipeline(config)
            X_processed, y_processed = pipeline.fit_transform(self.X_train, self.y_train)
            
            unique_classes = np.unique(y_processed)
            class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_processed)
            class_weight_dict = {c: w for c, w in zip(unique_classes, class_weights)}
            
            # Create model
            model = self.create_model(config.model_type, config.model_params, class_weight_dict)
            
            # Cross-validation
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            if config.model_type == 'hist_gradient':
                # Manual CV with sample weights
                sample_weights = np.array([class_weight_dict.get(y, 1.0) for y in y_processed])
                scores = []
                for train_idx, val_idx in cv.split(X_processed, y_processed):
                    X_tr, X_val = X_processed[train_idx], X_processed[val_idx]
                    y_tr, y_val = y_processed[train_idx], y_processed[val_idx]
                    w_tr = sample_weights[train_idx]
                    
                    model.fit(X_tr, y_tr, sample_weight=w_tr)
                    y_pred = model.predict(X_val)
                    score = f1_score(y_val, y_pred, average='macro', zero_division=0)
                    scores.append(score)
                cv_scores = np.array(scores)
            else:
                cv_scores = cross_val_score(model, X_processed, y_processed, cv=cv, scoring='f1_macro')
            
            score = cv_scores.mean()
            
            # Fit final model
            if config.model_type == 'hist_gradient':
                sample_weights = np.array([class_weight_dict.get(y, 1.0) for y in y_processed])
                model.fit(X_processed, y_processed, sample_weight=sample_weights)
            else:
                model.fit(X_processed, y_processed)
            
            # Generate test predictions
            if self.X_test is not None:
                X_test_transformed = pipeline.transform(self.X_test)
                
                if config.model_type == 'isolation_forest':
                    anomaly_scores = model.decision_function(X_test_transformed)
                    # Lower scores are anomalies
                    threshold_high = np.percentile(anomaly_scores, 3)  # Bottom 3% as H
                    threshold_low = np.percentile(anomaly_scores, 6)   # Next 3% as L
                    
                    predictions = np.ones(len(anomaly_scores), dtype=int) * 2  # Default None
                    predictions[anomaly_scores < threshold_high] = 0  # H
                    predictions[(anomaly_scores >= threshold_high) & (anomaly_scores < threshold_low)] = 1  # L
                    
                    probabilities = np.zeros((len(predictions), 3))
                    for i, pred in enumerate(predictions):
                        probabilities[i, pred] = 1.0
                else:
                    if hasattr(model, 'predict_proba'):
                        probabilities = model.predict_proba(X_test_transformed)
                        predictions = self.apply_probability_threshold(probabilities)
                    else:
                        predictions = model.predict(X_test_transformed)
                        probabilities = np.zeros((len(predictions), 3))
                        for i, pred in enumerate(predictions):
                            probabilities[i, int(pred)] = 1.0
                
                predictions = np.array(predictions).astype(int)
                
                self.all_predictions[config.config_id] = {
                    'predictions': predictions,
                    'probabilities': probabilities,
                    'score': score,
                    'config': config,
                    'model': model,
                    'pipeline': pipeline,
                    'config_index': config_index,
                    'model_type': config.model_type,
                    'cv_scores': cv_scores
                }
            
            MemoryManager.clean_memory()
            
            result = {
                'config': config,
                'score': score,
                'model': model,
                'pipeline': pipeline,
                'shape': X_processed.shape,
                'memory_used': MemoryManager.get_memory_usage(),
                'model_type': config.model_type,
                'cv_scores': cv_scores
            }
            
            for step in config.steps:
                if step.enabled:
                    self.step_performance[step.step_type][step.method].append(score)
            
            return result
            
        except Exception as e:
            print(f"  Error in config evaluation: {str(e)}")
            MemoryManager.clean_memory()
            return None
    
    def generate_intelligent_ensembles(self, test_ids, class_names):
        """Generate ensemble predictions"""
        
        if len(self.all_predictions) < 2:
            return {}
        
        sorted_results = sorted(self.all_predictions.items(), key=lambda x: x[1]['score'], reverse=True)
        
        ensembles = {}
        
        # Conservative voting ensemble
        if len(sorted_results) >= 3:
            top_n = min(5, len(sorted_results))
            top_n_preds = [sorted_results[i][1]['predictions'] for i in range(top_n)]
            
            ensemble_conservative = []
            for i in range(len(top_n_preds[0])):
                votes = [int(top_n_preds[j][i]) for j in range(top_n)]
                vote_counts = Counter(votes)
                
                # Require strong agreement for minority classes
                if vote_counts.get(0, 0) >= top_n * 0.6:  # H
                    ensemble_conservative.append(0)
                elif vote_counts.get(1, 0) >= top_n * 0.6:  # L
                    ensemble_conservative.append(1)
                else:
                    ensemble_conservative.append(2)  # None
            
            ensembles['conservative_voting'] = [class_names[int(p)] for p in ensemble_conservative]
        
        return ensembles
    
    def run_search(self, n_iterations=None, test_ids=None, class_names=None):
        if n_iterations is None:
            n_iterations = MAX_CONFIGURATIONS_TO_TEST
        
        initial_configs = self.generate_memory_efficient_configs(min(50, n_iterations * 2))
        
        for i in range(n_iterations):
            if i < len(initial_configs):
                config = initial_configs[i]
                strategy = "initial"
            else:
                config = self.generate_memory_efficient_configs(1)[0]
                strategy = "explore"
            
            if config.config_id in self.config_performance:
                continue
            
            print(f"\nConfig {i+1}/{n_iterations} [{strategy}] - Model: {config.model_type}")
            print(f"  Pipeline: {str(config)}")
            
            result = self.evaluate_config(config, i+1)
            
            if result:
                self.results.append(result)
                self.config_performance[config.config_id] = result['score']
                
                print(f"  Score: {result['score']:.4f} (std: {result['cv_scores'].std():.4f}), Shape: {result['shape']}")
                
                # Print prediction distribution
                if config.config_id in self.all_predictions:
                    pred_counts = Counter(self.all_predictions[config.config_id]['predictions'])
                    total_preds = sum(pred_counts.values())
                    print(f"  Predictions: H={pred_counts[0]} ({pred_counts[0]/total_preds:.1%}), " + 
                          f"L={pred_counts[1]} ({pred_counts[1]/total_preds:.1%}), " +
                          f"None={pred_counts[2]} ({pred_counts[2]/total_preds:.1%})")
                
                if result['score'] > self.best_score:
                    self.best_score = result['score']
                    self.best_config = config
                    self.best_model = result['model']
                    self.best_pipeline = result['pipeline']
                    print(f"  *** New best score: {self.best_score:.4f} ***")
                
                if config.config_id in self.all_predictions and test_ids is not None and class_names is not None:
                    pred_data = self.all_predictions[config.config_id]
                    filename = f"config_{i+1:03d}_{pred_data['model_type']}_score_{result['score']:.4f}.csv"
                    self.save_submission(pred_data['predictions'], filename, test_ids, class_names)
        
        if test_ids is not None and class_names is not None and len(self.all_predictions) > 0:
            print("\nGenerating ensemble submissions...")
            ensembles = self.generate_intelligent_ensembles(test_ids, class_names)
            
            for ensemble_name, predictions in ensembles.items():
                submission = pd.DataFrame({
                    'id': test_ids,
                    'class_label': predictions
                })
                filepath = os.path.join(SUBMISSION_DIR, f'ensemble_{ensemble_name}.csv')
                submission.to_csv(filepath, index=False)
                print(f"  Saved ensemble: {ensemble_name}")
                
                dist = Counter(predictions)
                print(f"    Distribution: {dist}")
        
        return self.best_config, self.best_model, self.best_pipeline
    
    def save_submission(self, predictions, filename, test_ids, class_names):
        predictions = np.array(predictions).astype(int)
        
        submission = pd.DataFrame({
            'id': test_ids,
            'class_label': [class_names[int(p)] for p in predictions]
        })
        filepath = os.path.join(SUBMISSION_DIR, filename)
        submission.to_csv(filepath, index=False)
        return filepath

def main():
    print("Loading data...")
    train = pd.read_csv('/kaggle/input/detecting-reversal-points-in-us-equities/competition_data/train.csv')
    test = pd.read_csv('/kaggle/input/detecting-reversal-points-in-us-equities/competition_data/test.csv')
    
    # Map targets
    target_mapping = {'HH': 'H', 'LH': 'H', 'HL': 'L', 'LL': 'L'}
    train['class_label_mapped'] = train['class_label'].map(target_mapping).fillna('None')
    
    # Extract ticker and time information
    ticker_ids = train['ticker_id'].values if 'ticker_id' in train.columns else None
    timestamps = train['t'].values if 't' in train.columns else None
    
    test_ticker_ids = test['ticker_id'].values if 'ticker_id' in test.columns else None
    test_timestamps = test['t'].values if 't' in test.columns else None
    
    metadata_cols = ['ticker_id', 't', 'class_label', 'class_label_mapped', 'train_id', 'id']
    feature_cols = [col for col in train.columns if col not in metadata_cols]
    
    X_train = train[feature_cols].values.astype(np.float32)
    y_train = pd.factorize(train['class_label_mapped'])[0]
    X_test = test[feature_cols].values.astype(np.float32)
    
    # Apply enhanced time series feature engineering
    print("Creating advanced time series features...")
    X_train = create_advanced_time_series_features(X_train, y_train, ticker_ids, timestamps)
    X_test = create_advanced_time_series_features(X_test, None, test_ticker_ids, test_timestamps)
    
    class_names = ['H', 'L', 'None']
    test_ids = test['id'].values if 'id' in test.columns else range(len(test))
    
    print(f"\nFinal data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution: {Counter(y_train)}")
    
    # Calculate expected distribution
    train_dist = Counter(y_train)
    total = sum(train_dist.values())
    print(f"Class percentages: H={train_dist[0]/total:.1%}, L={train_dist[1]/total:.1%}, None={train_dist[2]/total:.1%}")
    
    # Run grid search
    search = IntelligentGridSearch(X_train, y_train, X_test)
    best_config, best_model, best_pipeline = search.run_search(
        n_iterations=MAX_CONFIGURATIONS_TO_TEST,
        test_ids=test_ids,
        class_names=class_names
    )
    
    if best_model is not None:
        X_test_transformed = best_pipeline.transform(X_test)
        
        # Generate final predictions
        if hasattr(best_model, 'predict_proba'):
            probabilities = best_model.predict_proba(X_test_transformed)
            predictions = search.apply_probability_threshold(probabilities)
        else:
            predictions = best_model.predict(X_test_transformed)
        
        predictions = np.array(predictions).astype(int)
        final_predictions = [class_names[int(p)] for p in predictions]
        
        submission = pd.DataFrame({
            'id': test_ids,
            'class_label': final_predictions
        })
        
        submission.to_csv('submission.csv', index=False)
        submission.to_csv(os.path.join(SUBMISSION_DIR, 'FINAL_BEST_submission.csv'), index=False)
        
        print(f"\n{'='*60}")
        print(f"Best model type: {best_config.model_type}")
        print(f"Best score: {search.best_score:.4f}")
        print(f"Final submission distribution: {submission['class_label'].value_counts()}")
        
        # Check distribution
        dist_pct = submission['class_label'].value_counts(normalize=True)
        print(f"Percentages: H={dist_pct.get('H', 0):.1%}, L={dist_pct.get('L', 0):.1%}, None={dist_pct.get('None', 0):.1%}")
        
        if dist_pct.get('None', 0) < 0.8:
            print("WARNING: 'None' class may be under-represented.")
        elif dist_pct.get('None', 0) > 0.98:
            print("WARNING: Model may be too conservative - consider adjusting thresholds.")
        
        print(f"{'='*60}")

if __name__ == "__main__":
    main()


# #!/usr/bin/env python3

# # Install required packages
# !pip install pandas numpy scikit-learn imbalanced-learn -q

# import pandas as pd
# import numpy as np
# from dataclasses import dataclass, field
# from typing import Dict, List, Any, Optional, Tuple
# from abc import ABC, abstractmethod
# import hashlib
# import json
# import itertools
# from sklearn.preprocessing import (
#     StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer,
#     PowerTransformer, Normalizer, MaxAbsScaler, KBinsDiscretizer
# )
# from sklearn.decomposition import PCA, TruncatedSVD, FastICA, NMF, FactorAnalysis
# from sklearn.feature_selection import (
#     VarianceThreshold, SelectKBest, SelectPercentile, 
#     f_classif, mutual_info_classif, chi2, RFE
# )
# from sklearn.linear_model import LogisticRegression, RidgeClassifier
# from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
# from sklearn.model_selection import cross_val_score, StratifiedKFold
# from sklearn.metrics import f1_score
# from sklearn.utils.class_weight import compute_class_weight
# from sklearn.cluster import KMeans
# from imblearn.over_sampling import SMOTE, RandomOverSampler, ADASYN, BorderlineSMOTE
# from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
# from imblearn.combine import SMOTETomek, SMOTEENN
# import warnings
# warnings.filterwarnings('ignore')

# # ============================================================================
# # ENHANCED DATA TYPE HANDLER
# # ============================================================================

# class DataTypeHandler:
#     """Advanced handler for mixed data types"""
    
#     @staticmethod
#     def identify_feature_types(X, threshold=3):
#         """Identify boolean, categorical, and continuous features"""
#         n_samples, n_features = X.shape
#         boolean_mask = np.zeros(n_features, dtype=bool)
#         categorical_mask = np.zeros(n_features, dtype=bool)
        
#         for i in range(n_features):
#             col = X[:, i]
#             try:
#                 col_numeric = pd.to_numeric(col, errors='coerce')
#                 unique_vals = np.unique(col_numeric[~pd.isna(col_numeric)])
                
#                 if len(unique_vals) <= 2:
#                     boolean_mask[i] = True
#                 elif len(unique_vals) <= threshold:
#                     categorical_mask[i] = True
#             except:
#                 pass
                
#         return boolean_mask, categorical_mask
    
#     @staticmethod
#     def safe_variance_threshold(X, threshold=0.01):
#         """Apply variance threshold safely"""
#         n_features = X.shape[1]
#         keep_features = []
        
#         for i in range(n_features):
#             col = X[:, i].astype(float)
#             if np.var(col) > threshold:
#                 keep_features.append(i)
                
#         return np.array(keep_features)

# # ============================================================================
# # EXPANDED PREPROCESSING COMPONENTS
# # ============================================================================

# @dataclass
# class ProcessingInput:
#     X: np.ndarray
#     y: np.ndarray
#     boolean_mask: Optional[np.ndarray] = None
#     categorical_mask: Optional[np.ndarray] = None
#     metadata: Dict[str, Any] = field(default_factory=dict)

# @dataclass
# class ProcessingOutput:
#     X: np.ndarray
#     y: np.ndarray
#     metadata: Dict[str, Any] = field(default_factory=dict)
#     transformation_log: List[str] = field(default_factory=list)

# @dataclass
# class StepConfig:
#     enabled: bool = True
#     method: str = "default"
#     params: Dict[str, Any] = field(default_factory=dict)

# @dataclass
# class PipelineConfig:
#     config_id: str = ""
#     outlier_removal: StepConfig = field(default_factory=lambda: StepConfig(enabled=False))
#     scaling: StepConfig = field(default_factory=lambda: StepConfig(enabled=True, method="standard"))
#     feature_selection: StepConfig = field(default_factory=lambda: StepConfig(enabled=False))
#     dimensionality_reduction: StepConfig = field(default_factory=lambda: StepConfig(enabled=False))
#     balancing: StepConfig = field(default_factory=lambda: StepConfig(enabled=False))
    
#     def to_dict(self):
#         return {
#             'outlier_removal': {'enabled': self.outlier_removal.enabled, 'method': self.outlier_removal.method},
#             'scaling': {'enabled': self.scaling.enabled, 'method': self.scaling.method},
#             'feature_selection': {'enabled': self.feature_selection.enabled, 'method': self.feature_selection.method},
#             'dimensionality_reduction': {'enabled': self.dimensionality_reduction.enabled, 'method': self.dimensionality_reduction.method},
#             'balancing': {'enabled': self.balancing.enabled, 'method': self.balancing.method}
#         }
    
#     def generate_id(self):
#         config_str = json.dumps(self.to_dict(), sort_keys=True)
#         self.config_id = hashlib.md5(config_str.encode()).hexdigest()[:8]
#         return self.config_id

# class ExpandedOutlierRemovalStep:
#     """Expanded outlier removal with multiple strategies"""
    
#     def __init__(self, config: StepConfig):
#         self.config = config
#         self.bounds = {}
#         self.boolean_mask = None
#         self.is_fitted = False
        
#     def fit(self, data: ProcessingInput):
#         if not self.config.enabled:
#             return self
        
#         self.boolean_mask = data.boolean_mask
#         continuous_indices = np.where(~self.boolean_mask)[0] if self.boolean_mask is not None else np.arange(data.X.shape[1])
        
#         method = self.config.method
        
#         if method == "iqr":
#             multiplier = self.config.params.get('multiplier', 1.5)
#             for i in continuous_indices:
#                 col = data.X[:, i].astype(float)
#                 Q1 = np.percentile(col[~np.isnan(col)], 25)
#                 Q3 = np.percentile(col[~np.isnan(col)], 75)
#                 IQR = Q3 - Q1
#                 self.bounds[i] = (Q1 - multiplier * IQR, Q3 + multiplier * IQR)
                
#         elif method == "zscore":
#             threshold = self.config.params.get('threshold', 3)
#             for i in continuous_indices:
#                 col = data.X[:, i].astype(float)
#                 mean = np.nanmean(col)
#                 std = np.nanstd(col)
#                 self.bounds[i] = (mean - threshold * std, mean + threshold * std)
                
#         elif method == "mad":  # Median Absolute Deviation
#             threshold = self.config.params.get('threshold', 3)
#             for i in continuous_indices:
#                 col = data.X[:, i].astype(float)
#                 median = np.nanmedian(col)
#                 mad = np.nanmedian(np.abs(col - median))
#                 self.bounds[i] = (median - threshold * mad, median + threshold * mad)
                
#         elif method == "percentile":
#             lower_percentile = self.config.params.get('lower', 1)
#             upper_percentile = self.config.params.get('upper', 99)
#             for i in continuous_indices:
#                 col = data.X[:, i].astype(float)
#                 self.bounds[i] = (
#                     np.percentile(col[~np.isnan(col)], lower_percentile),
#                     np.percentile(col[~np.isnan(col)], upper_percentile)
#                 )
                
#         elif method == "isolation_forest":
#             # Simple isolation using distance from mean
#             contamination = self.config.params.get('contamination', 0.1)
#             for i in continuous_indices:
#                 col = data.X[:, i].astype(float)
#                 threshold = np.percentile(np.abs(col - np.nanmean(col)), 100 * (1 - contamination))
#                 mean = np.nanmean(col)
#                 self.bounds[i] = (mean - threshold, mean + threshold)
        
#         self.is_fitted = True
#         return self
    
#     def transform(self, data: ProcessingInput):
#         if not self.config.enabled:
#             return ProcessingOutput(X=data.X, y=data.y, metadata=data.metadata,
#                                   transformation_log=["outlier_removal: skipped"])
        
#         X_transformed = data.X.copy()
#         for i, (lower, upper) in self.bounds.items():
#             X_transformed[:, i] = np.clip(X_transformed[:, i].astype(float), lower, upper)
        
#         return ProcessingOutput(
#             X=X_transformed, y=data.y, metadata=data.metadata,
#             transformation_log=[f"outlier_removal: {self.config.method}"]
#         )

# class ExpandedScalingStep:
#     """Expanded scaling with binning and transformation strategies"""
    
#     def __init__(self, config: StepConfig):
#         self.config = config
#         self.scaler = None
#         self.boolean_mask = None
#         self.continuous_indices = None
#         self.is_fitted = False
        
#     def fit(self, data: ProcessingInput):
#         if not self.config.enabled:
#             return self
        
#         self.boolean_mask = data.boolean_mask
#         self.continuous_indices = np.where(~self.boolean_mask)[0] if self.boolean_mask is not None else np.arange(data.X.shape[1])
        
#         if len(self.continuous_indices) > 0:
#             continuous_data = data.X[:, self.continuous_indices].astype(float)
#             method = self.config.method
            
#             if method == "standard":
#                 self.scaler = StandardScaler()
#             elif method == "robust":
#                 self.scaler = RobustScaler()
#             elif method == "minmax":
#                 self.scaler = MinMaxScaler()
#             elif method == "maxabs":
#                 self.scaler = MaxAbsScaler()
#             elif method == "quantile_uniform":
#                 n_quantiles = min(self.config.params.get('n_quantiles', 1000), len(data.X))
#                 self.scaler = QuantileTransformer(output_distribution='uniform', n_quantiles=n_quantiles, random_state=42)
#             elif method == "quantile_normal":
#                 n_quantiles = min(self.config.params.get('n_quantiles', 1000), len(data.X))
#                 self.scaler = QuantileTransformer(output_distribution='normal', n_quantiles=n_quantiles, random_state=42)
#             elif method == "power_yeo":
#                 self.scaler = PowerTransformer(method='yeo-johnson', standardize=True)
#             elif method == "power_box":
#                 # Box-Cox requires positive values
#                 self.scaler = PowerTransformer(method='box-cox', standardize=True)
#             elif method == "normalizer":
#                 norm = self.config.params.get('norm', 'l2')
#                 self.scaler = Normalizer(norm=norm)
#             elif method == "binning_uniform":
#                 n_bins = self.config.params.get('n_bins', 10)
#                 self.scaler = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='uniform')
#             elif method == "binning_quantile":
#                 n_bins = self.config.params.get('n_bins', 10)
#                 self.scaler = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
#             elif method == "binning_kmeans":
#                 n_bins = self.config.params.get('n_bins', 10)
#                 self.scaler = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans')
#             else:
#                 self.scaler = StandardScaler()
            
#             try:
#                 self.scaler.fit(continuous_data)
#             except:
#                 # Fallback to standard scaler if method fails
#                 self.scaler = StandardScaler()
#                 self.scaler.fit(continuous_data)
        
#         self.is_fitted = True
#         return self
    
#     def transform(self, data: ProcessingInput):
#         if not self.config.enabled:
#             return ProcessingOutput(X=data.X, y=data.y, metadata=data.metadata,
#                                   transformation_log=["scaling: skipped"])
        
#         X_transformed = data.X.copy().astype(float)
        
#         if self.scaler is not None and len(self.continuous_indices) > 0:
#             try:
#                 X_transformed[:, self.continuous_indices] = self.scaler.transform(
#                     X_transformed[:, self.continuous_indices]
#                 )
#             except:
#                 pass
        
#         return ProcessingOutput(
#             X=X_transformed, y=data.y, metadata=data.metadata,
#             transformation_log=[f"scaling: {self.config.method}"]
#         )

# class ExpandedFeatureSelectionStep:
#     """Expanded feature selection with multiple strategies"""
    
#     def __init__(self, config: StepConfig):
#         self.config = config
#         self.selected_features = None
#         self.selector = None
#         self.is_fitted = False
        
#     def fit(self, data: ProcessingInput):
#         if not self.config.enabled:
#             return self
        
#         X_float = data.X.astype(float)
#         method = self.config.method
        
#         if method == "variance":
#             threshold = self.config.params.get('threshold', 0.01)
#             self.selected_features = DataTypeHandler.safe_variance_threshold(X_float, threshold)
            
#         elif method == "kbest_f":
#             k = min(self.config.params.get('k', 100), data.X.shape[1])
#             selector = SelectKBest(score_func=f_classif, k=k)
#             selector.fit(X_float, data.y)
#             self.selected_features = np.where(selector.get_support())[0]
            
#         elif method == "kbest_mutual":
#             k = min(self.config.params.get('k', 100), data.X.shape[1])
#             selector = SelectKBest(score_func=mutual_info_classif, k=k)
#             selector.fit(X_float, data.y)
#             self.selected_features = np.where(selector.get_support())[0]
            
#         elif method == "kbest_chi2":
#             # Chi2 requires non-negative features
#             k = min(self.config.params.get('k', 100), data.X.shape[1])
#             X_nonneg = X_float - X_float.min() + 0.001
#             selector = SelectKBest(score_func=chi2, k=k)
#             selector.fit(X_nonneg, data.y)
#             self.selected_features = np.where(selector.get_support())[0]
            
#         elif method == "percentile":
#             percentile = self.config.params.get('percentile', 50)
#             selector = SelectPercentile(score_func=f_classif, percentile=percentile)
#             selector.fit(X_float, data.y)
#             self.selected_features = np.where(selector.get_support())[0]
            
#         elif method == "rfe":
#             n_features = self.config.params.get('n_features', min(100, data.X.shape[1] // 2))
#             estimator = LogisticRegression(max_iter=100, random_state=42)
#             selector = RFE(estimator, n_features_to_select=n_features, step=0.1)
#             selector.fit(X_float, data.y)
#             self.selected_features = np.where(selector.get_support())[0]
            
#         elif method == "tree_based":
#             # Use tree importance for selection
#             clf = ExtraTreesClassifier(n_estimators=50, random_state=42)
#             clf.fit(X_float, data.y)
#             importances = clf.feature_importances_
#             threshold = np.percentile(importances, self.config.params.get('percentile', 50))
#             self.selected_features = np.where(importances >= threshold)[0]
            
#         elif method == "correlation":
#             threshold = self.config.params.get('threshold', 0.95)
#             try:
#                 corr_matrix = np.corrcoef(X_float.T)
#                 corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
#                 upper = np.triu(np.abs(corr_matrix), k=1)
#                 to_drop = np.unique(np.where(upper > threshold)[1])
#                 self.selected_features = np.setdiff1d(np.arange(data.X.shape[1]), to_drop)
#             except:
#                 self.selected_features = np.arange(data.X.shape[1])
        
#         # Ensure minimum features
#         if self.selected_features is None or len(self.selected_features) < 10:
#             self.selected_features = np.arange(min(100, data.X.shape[1]))
        
#         self.is_fitted = True
#         return self
    
#     def transform(self, data: ProcessingInput):
#         if not self.config.enabled or self.selected_features is None:
#             return ProcessingOutput(X=data.X, y=data.y, metadata=data.metadata,
#                                   transformation_log=["feature_selection: skipped"])
        
#         X_transformed = data.X[:, self.selected_features]
        
#         return ProcessingOutput(
#             X=X_transformed, y=data.y, metadata=data.metadata,
#             transformation_log=[f"feature_selection: {self.config.method} ({len(self.selected_features)} features)"]
#         )

# class ExpandedDimensionalityReductionStep:
#     """Expanded dimensionality reduction with multiple strategies"""
    
#     def __init__(self, config: StepConfig):
#         self.config = config
#         self.reducer = None
#         self.is_fitted = False
        
#     def fit(self, data: ProcessingInput):
#         if not self.config.enabled:
#             return self
        
#         n_components = min(
#             self.config.params.get('n_components', 100),
#             min(data.X.shape) - 1
#         )
        
#         if n_components < 2:
#             return self
        
#         X_float = data.X.astype(float)
#         method = self.config.method
        
#         try:
#             if method == "pca":
#                 self.reducer = PCA(n_components=n_components, random_state=42)
#             elif method == "svd":
#                 self.reducer = TruncatedSVD(n_components=n_components, random_state=42)
#             elif method == "ica":
#                 self.reducer = FastICA(n_components=n_components, random_state=42, max_iter=500)
#             elif method == "nmf":
#                 # NMF requires non-negative features
#                 X_nonneg = X_float - X_float.min() + 0.001
#                 self.reducer = NMF(n_components=n_components, random_state=42, max_iter=500)
#                 self.reducer.fit(X_nonneg)
#                 return self
#             elif method == "factor":
#                 self.reducer = FactorAnalysis(n_components=n_components, random_state=42)
#             elif method == "kernel_pca":
#                 from sklearn.decomposition import KernelPCA
#                 kernel = self.config.params.get('kernel', 'rbf')
#                 self.reducer = KernelPCA(n_components=n_components, kernel=kernel, random_state=42)
#             else:
#                 self.reducer = PCA(n_components=n_components, random_state=42)
            
#             if self.reducer and method != "nmf":
#                 self.reducer.fit(X_float)
#         except:
#             self.reducer = None
        
#         self.is_fitted = True
#         return self
    
#     def transform(self, data: ProcessingInput):
#         if not self.config.enabled or self.reducer is None:
#             return ProcessingOutput(X=data.X, y=data.y, metadata=data.metadata,
#                                   transformation_log=["dimensionality_reduction: skipped"])
        
#         try:
#             X_float = data.X.astype(float)
#             if self.config.method == "nmf":
#                 X_float = X_float - X_float.min() + 0.001
#             X_transformed = self.reducer.transform(X_float)
#         except:
#             X_transformed = data.X
        
#         return ProcessingOutput(
#             X=X_transformed, y=data.y, metadata=data.metadata,
#             transformation_log=[f"dimensionality_reduction: {self.config.method}"]
#         )

# class ExpandedBalancingStep:
#     """Expanded balancing with multiple strategies including hybrid methods"""
    
#     def __init__(self, config: StepConfig):
#         self.config = config
#         self.balancer = None
#         self.is_fitted = False
        
#     def fit(self, data: ProcessingInput):
#         if not self.config.enabled:
#             return self
        
#         unique, counts = np.unique(data.y, return_counts=True)
#         min_samples = counts.min()
#         method = self.config.method
        
#         try:
#             if method == "smote":
#                 k_neighbors = min(self.config.params.get('k_neighbors', 5), min_samples - 1)
#                 if k_neighbors > 0:
#                     self.balancer = SMOTE(k_neighbors=k_neighbors, random_state=42)
#                 else:
#                     self.balancer = RandomOverSampler(random_state=42)
                    
#             elif method == "borderline_smote":
#                 k_neighbors = min(self.config.params.get('k_neighbors', 5), min_samples - 1)
#                 if k_neighbors > 0:
#                     self.balancer = BorderlineSMOTE(k_neighbors=k_neighbors, random_state=42)
#                 else:
#                     self.balancer = RandomOverSampler(random_state=42)
                    
#             elif method == "adasyn":
#                 n_neighbors = min(self.config.params.get('n_neighbors', 5), min_samples - 1)
#                 if n_neighbors > 0:
#                     self.balancer = ADASYN(n_neighbors=n_neighbors, random_state=42)
#                 else:
#                     self.balancer = RandomOverSampler(random_state=42)
                    
#             elif method == "oversample":
#                 self.balancer = RandomOverSampler(random_state=42)
                
#             elif method == "undersample":
#                 self.balancer = RandomUnderSampler(random_state=42)
                
#             elif method == "tomek":
#                 self.balancer = TomekLinks()
                
#             elif method == "nearmiss":
#                 self.balancer = NearMiss(version=1)
                
#             elif method == "smote_tomek":
#                 k_neighbors = min(self.config.params.get('k_neighbors', 5), min_samples - 1)
#                 if k_neighbors > 0:
#                     self.balancer = SMOTETomek(smote=SMOTE(k_neighbors=k_neighbors), random_state=42)
#                 else:
#                     self.balancer = RandomOverSampler(random_state=42)
                    
#             elif method == "smote_enn":
#                 k_neighbors = min(self.config.params.get('k_neighbors', 5), min_samples - 1)
#                 if k_neighbors > 0:
#                     self.balancer = SMOTEENN(smote=SMOTE(k_neighbors=k_neighbors), random_state=42)
#                 else:
#                     self.balancer = RandomOverSampler(random_state=42)
#             else:
#                 self.balancer = RandomOverSampler(random_state=42)
#         except:
#             self.balancer = None
        
#         self.is_fitted = True
#         return self
    
#     def transform(self, data: ProcessingInput):
#         if not self.config.enabled or self.balancer is None:
#             return ProcessingOutput(X=data.X, y=data.y, metadata=data.metadata,
#                                   transformation_log=["balancing: skipped"])
        
#         try:
#             X_float = data.X.astype(float)
#             X_balanced, y_balanced = self.balancer.fit_resample(X_float, data.y)
#             return ProcessingOutput(
#                 X=X_balanced, y=y_balanced, metadata=data.metadata,
#                 transformation_log=[f"balancing: {self.config.method}"]
#             )
#         except Exception as e:
#             return ProcessingOutput(X=data.X, y=data.y, metadata=data.metadata,
#                                   transformation_log=[f"balancing: failed"])

# class ComprehensivePipeline:
#     """Pipeline that handles all preprocessing steps"""
    
#     def __init__(self, config: PipelineConfig, boolean_mask=None):
#         self.config = config
#         self.boolean_mask = boolean_mask
#         self.steps = self._initialize_steps()
        
#     def _initialize_steps(self):
#         return [
#             ExpandedOutlierRemovalStep(self.config.outlier_removal),
#             ExpandedScalingStep(self.config.scaling),
#             ExpandedFeatureSelectionStep(self.config.feature_selection),
#             ExpandedDimensionalityReductionStep(self.config.dimensionality_reduction),
#             ExpandedBalancingStep(self.config.balancing)
#         ]
    
#     def fit(self, X: np.ndarray, y: np.ndarray):
#         data = ProcessingInput(X=X, y=y, boolean_mask=self.boolean_mask)
#         for step in self.steps:
#             step.fit(data)
#             output = step.transform(data)
#             if isinstance(step, ExpandedFeatureSelectionStep) and step.selected_features is not None:
#                 if self.boolean_mask is not None:
#                     self.boolean_mask = self.boolean_mask[step.selected_features]
#             data = ProcessingInput(X=output.X, y=output.y, boolean_mask=self.boolean_mask)
#         return self
    
#     def transform(self, X: np.ndarray, y: Optional[np.ndarray] = None):
#         data = ProcessingInput(X=X, y=y if y is not None else np.zeros(X.shape[0]), boolean_mask=self.boolean_mask)
        
#         for i, step in enumerate(self.steps[:-1]):
#             output = step.transform(data)
#             data = ProcessingInput(X=output.X, y=output.y, boolean_mask=self.boolean_mask)
        
#         if y is not None and self.config.balancing.enabled:
#             output = self.steps[-1].transform(data)
#             return output.X, output.y
#         else:
#             return data.X, y
    
#     def fit_transform(self, X: np.ndarray, y: np.ndarray):
#         self.fit(X, y)
#         return self.transform(X, y)

# class GridSearchOrchestrator:
#     """Orchestrates comprehensive grid search across all preprocessing combinations"""
    
#     def __init__(self, X_train, y_train, X_test, boolean_mask):
#         self.X_train = X_train
#         self.y_train = y_train
#         self.X_test = X_test
#         self.boolean_mask = boolean_mask
#         self.results = []
#         self.best_config = None
#         self.best_score = -np.inf
#         self.best_model = None
#         self.best_pipeline = None
        
#     def generate_all_configurations(self):
#         """Generate all possible configurations"""
        
#         outlier_options = [
#             StepConfig(enabled=False),
#             StepConfig(enabled=True, method="iqr", params={'multiplier': 1.5}),
#             StepConfig(enabled=True, method="zscore", params={'threshold': 3}),
#             StepConfig(enabled=True, method="mad", params={'threshold': 3}),
#             StepConfig(enabled=True, method="percentile", params={'lower': 1, 'upper': 99})
#         ]
        
#         scaling_options = [
#             StepConfig(enabled=False),
#             StepConfig(enabled=True, method="standard"),
#             StepConfig(enabled=True, method="robust"),
#             StepConfig(enabled=True, method="minmax"),
#             StepConfig(enabled=True, method="quantile_uniform", params={'n_quantiles': 1000}),
#             StepConfig(enabled=True, method="normalizer", params={'norm': 'l2'}),
#             StepConfig(enabled=True, method="binning_quantile", params={'n_bins': 10})
#         ]
        
#         feature_selection_options = [
#             StepConfig(enabled=False),
#             StepConfig(enabled=True, method="variance", params={'threshold': 0.01}),
#             StepConfig(enabled=True, method="kbest_f", params={'k': 500}),
#             StepConfig(enabled=True, method="kbest_mutual", params={'k': 500}),
#             StepConfig(enabled=True, method="percentile", params={'percentile': 50})
#         ]
        
#         dim_reduction_options = [
#             StepConfig(enabled=False),
#             StepConfig(enabled=True, method="pca", params={'n_components': 100}),
#             StepConfig(enabled=True, method="svd", params={'n_components': 100}),
#             StepConfig(enabled=True, method="ica", params={'n_components': 50})
#         ]
        
#         balancing_options = [
#             StepConfig(enabled=False),
#             StepConfig(enabled=True, method="oversample"),
#             StepConfig(enabled=True, method="smote", params={'k_neighbors': 5}),
#             StepConfig(enabled=True, method="adasyn", params={'n_neighbors': 5}),
#             StepConfig(enabled=True, method="smote_tomek", params={'k_neighbors': 5})
#         ]
        
#         # Generate all combinations
#         all_combinations = list(itertools.product(
#             outlier_options,
#             scaling_options,
#             feature_selection_options,
#             dim_reduction_options,
#             balancing_options
#         ))
        
#         configs = []
#         for combo in all_combinations:
#             config = PipelineConfig(
#                 outlier_removal=combo[0],
#                 scaling=combo[1],
#                 feature_selection=combo[2],
#                 dimensionality_reduction=combo[3],
#                 balancing=combo[4]
#             )
#             config.generate_id()
#             configs.append(config)
        
#         return configs
    
#     def evaluate_configuration(self, config):
#         """Evaluate a single configuration"""
#         try:
#             pipeline = ComprehensivePipeline(config, self.boolean_mask.copy())
#             X_processed, y_processed = pipeline.fit_transform(self.X_train, self.y_train)
            
#             # Calculate class weights
#             unique_classes = np.unique(y_processed)
#             class_weights = compute_class_weight('balanced', classes=unique_classes, y=y_processed)
#             class_weight_dict = {c: w for c, w in zip(unique_classes, class_weights)}
            
#             # Quick evaluation with logistic regression
#             model = LogisticRegression(max_iter=1000, class_weight=class_weight_dict, random_state=42)
#             cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
#             cv_scores = cross_val_score(model, X_processed, y_processed, cv=cv, scoring='f1_macro')
#             cv_mean = cv_scores.mean()
            
#             # Fit model for potential final use
#             model.fit(X_processed, y_processed)
            
#             return {
#                 'config': config,
#                 'cv_score': cv_mean,
#                 'model': model,
#                 'pipeline': pipeline,
#                 'shape': X_processed.shape
#             }
#         except:
#             return None
    
#     def run_grid_search(self, max_configs=10):
#         """Run grid search with specified number of configurations"""
        
#         all_configs = self.generate_all_configurations()
#         total_configs = len(all_configs)
        
#         print(f"\nTotal configuration space: {total_configs} combinations")
#         print(f"Testing {min(max_configs, total_configs)} configurations")
#         print(f"Initial data shape: {self.X_train.shape}")
#         print(f"Class distribution: {dict(zip(*np.unique(self.y_train, return_counts=True)))}\n")
        
#         # Sample configurations if necessary
#         if max_configs < total_configs:
#             # Take strategic sample: some from beginning, middle, end, and random
#             indices = []
#             indices.extend(range(0, min(max_configs//4, total_configs)))  # First quarter
#             indices.extend(range(total_configs//2, min(total_configs//2 + max_configs//4, total_configs)))  # Middle quarter
#             indices.extend(range(max(0, total_configs - max_configs//4), total_configs))  # Last quarter
#             remaining = max_configs - len(indices)
#             if remaining > 0:
#                 random_indices = np.random.choice(total_configs, size=remaining, replace=False)
#                 indices.extend(random_indices)
#             configs_to_test = [all_configs[i] for i in set(indices[:max_configs])]
#         else:
#             configs_to_test = all_configs
        
#         for i, config in enumerate(configs_to_test):
#             print(f"Testing configuration {i+1}/{len(configs_to_test)} (ID: {config.config_id})", end=" ")
            
#             result = self.evaluate_configuration(config)
            
#             if result:
#                 print(f"CV F1: {result['cv_score']:.4f}, Shape: {result['shape']}")
                
#                 if result['cv_score'] > self.best_score:
#                     self.best_score = result['cv_score']
#                     self.best_config = result['config']
#                     self.best_model = result['model']
#                     self.best_pipeline = result['pipeline']
                
#                 self.results.append(result)
#             else:
#                 print("Failed")
        
#         self._print_summary()
#         return self.best_config, self.best_model, self.best_pipeline
    
#     def _print_summary(self):
#         """Print summary of grid search results"""
#         print("\n" + "="*80)
#         print("GRID SEARCH SUMMARY")
#         print("="*80)
        
#         if self.results:
#             print(f"Configurations tested: {len(self.results)}")
#             print(f"Best CV F1 Score: {self.best_score:.4f}")
#             print(f"Best Configuration ID: {self.best_config.config_id}")
            
#             # Show top 5 configurations
#             sorted_results = sorted(self.results, key=lambda x: x['cv_score'], reverse=True)[:5]
#             print("\nTop 5 Configurations:")
#             for i, result in enumerate(sorted_results):
#                 config_desc = []
#                 if result['config'].outlier_removal.enabled:
#                     config_desc.append(f"outlier:{result['config'].outlier_removal.method}")
#                 if result['config'].scaling.enabled:
#                     config_desc.append(f"scale:{result['config'].scaling.method}")
#                 if result['config'].feature_selection.enabled:
#                     config_desc.append(f"feature:{result['config'].feature_selection.method}")
#                 if result['config'].dimensionality_reduction.enabled:
#                     config_desc.append(f"dim:{result['config'].dimensionality_reduction.method}")
#                 if result['config'].balancing.enabled:
#                     config_desc.append(f"balance:{result['config'].balancing.method}")
                
#                 print(f"  {i+1}. CV F1: {result['cv_score']:.4f} | {' | '.join(config_desc)}")

# def main():
#     """Main execution function"""
#     print("Loading data...")
#     train = pd.read_csv('/kaggle/input/detecting-reversal-points-in-us-equities/competition_data/train.csv')
#     test = pd.read_csv('/kaggle/input/detecting-reversal-points-in-us-equities/competition_data/test.csv')
    
#     # Map targets
#     target_mapping = {'HH': 'H', 'LH': 'H', 'HL': 'L', 'LL': 'L'}
#     train['class_label_mapped'] = train['class_label'].map(target_mapping).fillna('None')
    
#     # Identify columns
#     metadata_cols = ['ticker_id', 't', 'class_label', 'class_label_mapped', 'train_id', 'id']
#     feature_cols = [col for col in train.columns if col not in metadata_cols]
    
#     # Prepare data
#     X_train = train[feature_cols].values
#     y_train = pd.factorize(train['class_label_mapped'])[0]
#     X_test = test[feature_cols].values
    
#     # Clean data
#     X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
#     X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
#     # Identify feature types
#     print("Identifying feature types...")
#     boolean_mask, categorical_mask = DataTypeHandler.identify_feature_types(X_train)
#     print(f"Boolean features: {boolean_mask.sum()}, Categorical: {categorical_mask.sum()}, Continuous: {(~boolean_mask & ~categorical_mask).sum()}")
    
#     # Run grid search
#     orchestrator = GridSearchOrchestrator(X_train, y_train, X_test, boolean_mask)
    
#     # Set number of configurations to test
#     NUM_CONFIGS_TO_TEST = 10  # Adjust this value as needed
    
#     best_config, best_model, best_pipeline = orchestrator.run_grid_search(max_configs=NUM_CONFIGS_TO_TEST)
    
#     # Generate predictions
#     if best_model is not None:
#         print("\nGenerating final predictions...")
#         X_test_final, _ = best_pipeline.transform(X_test)
#         predictions = best_model.predict(X_test_final)
        
#         class_names = ['H', 'L', 'None']
#         final_predictions = [class_names[p] for p in predictions]
        
#         submission = pd.DataFrame({
#             'id': test['id'] if 'id' in test.columns else range(len(test)),
#             'class_label': final_predictions
#         })
        
#         submission.to_csv('submission.csv', index=False)
        
#         print(f"\nSubmission saved: {submission.shape}")
#         print(f"Prediction distribution: {submission['class_label'].value_counts().to_dict()}")
#         print("Prediction percentages:")
#         for class_name, pct in submission['class_label'].value_counts(normalize=True).items():
#             print(f"  {class_name}: {pct:.1%}")

# if __name__ == "__main__":
#     main()

