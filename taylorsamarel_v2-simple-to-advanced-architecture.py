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


import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, ClassVar
from enum import Enum
import hashlib
import json
import random
import warnings
import time
from datetime import datetime, timedelta
warnings.filterwarnings('ignore')

from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, 
                             VotingClassifier, StackingClassifier, AdaBoostClassifier,
                             GradientBoostingClassifier, HistGradientBoostingClassifier)
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.cluster import AgglomerativeClustering
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import lightgbm as lgb
import xgboost as xgb
from scipy import stats
import catboost as cb

try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    print("Installing sklearn-extra for KMedoids...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn-extra"])
    from sklearn_extra.cluster import KMedoids

# ============================================================================
# CORE FRAMEWORK (Keep existing structure)
# ============================================================================

class StepType(Enum):
    """Enumeration of preprocessing step types"""
    IMPUTATION = "imputation"
    SCALING = "scaling"
    UNIVARIATE_OUTLIER = "univariate_outlier"
    TIME_SERIES_FEATURES = "time_series_features"
    PATTERN_FEATURES = "pattern_features"
    FEATURE_ENGINEERING = "feature_engineering"
    FEATURE_REDUCTION = "feature_reduction"
    MODELING = "modeling"

@dataclass
class StepConfig:
    """Configuration for a single preprocessing step"""
    step_type: StepType
    method: str
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'type': self.step_type.value,
            'method': self.method,
            'params': self.params
        }
    
    def __hash__(self):
        return hash((self.step_type.value, self.method, json.dumps(self.params, sort_keys=True)))

class BaseStep(ABC):
    """Abstract base class for all preprocessing steps"""
    
    CONFIGURATIONS: ClassVar[List[Dict[str, Any]]] = []
    STEP_TYPE: ClassVar[StepType] = None
    ALLOW_SKIP: ClassVar[bool] = False
    
    def __init__(self, config: StepConfig):
        self.config = config
        self.is_fitted = False
        self._state = {}
        self._cache = {}
    
    @classmethod
    def get_configurations(cls) -> List[StepConfig]:
        """Get all possible configurations for this step"""
        configs = []
        for config_dict in cls.CONFIGURATIONS:
            configs.append(StepConfig(
                step_type=cls.STEP_TYPE,
                method=config_dict['method'],
                params=config_dict.get('params', {})
            ))
        
        if cls.ALLOW_SKIP:
            configs.append(None)
            
        return configs
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'BaseStep':
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        pass
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        self.fit(X, y, **kwargs)
        return self.transform(X, **kwargs)
    
    @property
    def name(self) -> str:
        params_str = "_".join([f"{k}{v}" for k, v in self.config.params.items()])
        if params_str:
            return f"{self.config.step_type.value}_{self.config.method}_{params_str}"
        return f"{self.config.step_type.value}_{self.config.method}"

# [Keep existing Imputation, Scaling, UnivariateOutlier steps as before]

class ImputationStep(BaseStep):
    """Handles null, inf, and -inf values"""
    
    STEP_TYPE = StepType.IMPUTATION
    ALLOW_SKIP = False
    
    CONFIGURATIONS = [
        {'method': 'median'},
        {'method': 'mean'},
        {'method': 'zero'},
    ]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'ImputationStep':
        self._state['fill_values'] = np.zeros(X.shape[1])
        
        for i in range(X.shape[1]):
            col = X[:, i]
            finite_mask = np.isfinite(col)
            if np.any(finite_mask):
                if self.config.method == "median":
                    self._state['fill_values'][i] = np.median(col[finite_mask])
                elif self.config.method == "mean":
                    self._state['fill_values'][i] = np.mean(col[finite_mask])
                elif self.config.method == "zero":
                    self._state['fill_values'][i] = 0
            else:
                self._state['fill_values'][i] = 0
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        X_transformed = X.copy()
        X_transformed = np.where(np.isinf(X_transformed), np.nan, X_transformed)
        
        for i in range(X.shape[1]):
            mask = np.isnan(X_transformed[:, i])
            X_transformed[mask, i] = self._state['fill_values'][i]
        
        return np.nan_to_num(X_transformed, nan=0.0, posinf=0.0, neginf=0.0)

class ScalingStep(BaseStep):
    """Scales features"""
    
    STEP_TYPE = StepType.SCALING
    ALLOW_SKIP = True
    
    CONFIGURATIONS = [
        {'method': 'standard'},
        {'method': 'robust'},
        {'method': 'minmax'},
    ]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'ScalingStep':
        if self.config.method == "standard":
            self._state['scaler'] = StandardScaler()
        elif self.config.method == "robust":
            self._state['scaler'] = RobustScaler()
        elif self.config.method == "minmax":
            self._state['scaler'] = MinMaxScaler()
        
        self._state['scaler'].fit(X)
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        X_transformed = self._state['scaler'].transform(X)
        return np.nan_to_num(X_transformed, nan=0.0, posinf=3.0, neginf=-3.0)

class UnivariateOutlierStep(BaseStep):
    """Handles outliers on a per-feature basis"""
    
    STEP_TYPE = StepType.UNIVARIATE_OUTLIER
    ALLOW_SKIP = True
    
    CONFIGURATIONS = [
        {'method': 'iqr', 'params': {'multiplier': 1.5}},
        {'method': 'percentile', 'params': {'lower': 5, 'upper': 95}},
    ]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'UnivariateOutlierStep':
        self._state['bounds'] = []
        
        if self.config.method == "iqr":
            multiplier = self.config.params.get('multiplier', 1.5)
            
            for i in range(X.shape[1]):
                col = X[:, i]
                Q1 = np.percentile(col, 25)
                Q3 = np.percentile(col, 75)
                IQR = Q3 - Q1
                lower = Q1 - multiplier * IQR
                upper = Q3 + multiplier * IQR
                self._state['bounds'].append((lower, upper))
                
        elif self.config.method == "percentile":
            lower_pct = self.config.params.get('lower', 5)
            upper_pct = self.config.params.get('upper', 95)
            
            for i in range(X.shape[1]):
                col = X[:, i]
                lower = np.percentile(col, lower_pct)
                upper = np.percentile(col, upper_pct)
                self._state['bounds'].append((lower, upper))
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        X_transformed = X.copy()
        
        for i, (lower, upper) in enumerate(self._state['bounds']):
            X_transformed[:, i] = np.clip(X_transformed[:, i], lower, upper)
        
        return X_transformed

# ============================================================================
# ENHANCED TIME SERIES & PATTERN FEATURES
# ============================================================================

class TimeSeriesFeaturesStep(BaseStep):
    """Enhanced time series features for reversal detection"""
    
    STEP_TYPE = StepType.TIME_SERIES_FEATURES
    ALLOW_SKIP = True
    
    CONFIGURATIONS = [
        {'method': 'basic_lags', 'params': {'lags': [1, 2, 3]}},
        {'method': 'rolling_stats', 'params': {'windows': [5, 10, 20]}},
        {'method': 'momentum', 'params': {'periods': [5, 10, 15]}},
        {'method': 'acceleration', 'params': {'periods': [3, 5]}},
    ]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'TimeSeriesFeaturesStep':
        variances = np.var(X, axis=0)
        n_top = min(15, X.shape[1])
        self._state['top_features'] = np.argsort(variances)[-n_top:]
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        features_list = [X]
        X_selected = X[:, self._state['top_features']]
        
        if self.config.method == "basic_lags":
            lags = self.config.params.get('lags', [1, 2, 3])
            new_features = []
            for lag in lags:
                for j in range(min(5, X_selected.shape[1])):
                    lagged = np.zeros(X.shape[0])
                    if lag < X.shape[0]:
                        lagged[lag:] = X_selected[:-lag, j]
                    new_features.append(lagged.reshape(-1, 1))
            if new_features:
                features_list.append(np.hstack(new_features))
        
        elif self.config.method == "rolling_stats":
            windows = self.config.params.get('windows', [5, 10, 20])
            new_features = []
            for window in windows:
                for j in range(min(3, X_selected.shape[1])):
                    rolling_mean = np.zeros(X.shape[0])
                    rolling_std = np.zeros(X.shape[0])
                    for i in range(window, X.shape[0]):
                        window_data = X_selected[i-window:i, j]
                        rolling_mean[i] = np.mean(window_data)
                        rolling_std[i] = np.std(window_data)
                    new_features.append(rolling_mean.reshape(-1, 1))
                    new_features.append(rolling_std.reshape(-1, 1))
            if new_features:
                features_list.append(np.hstack(new_features))
                
        elif self.config.method == "momentum":
            periods = self.config.params.get('periods', [5, 10, 15])
            new_features = []
            for period in periods:
                for j in range(min(5, X_selected.shape[1])):
                    momentum = np.zeros(X.shape[0])
                    if period < X.shape[0]:
                        momentum[period:] = X_selected[period:, j] - X_selected[:-period, j]
                    new_features.append(momentum.reshape(-1, 1))
            if new_features:
                features_list.append(np.hstack(new_features))
                
        elif self.config.method == "acceleration":
            periods = self.config.params.get('periods', [3, 5])
            new_features = []
            for period in periods:
                for j in range(min(3, X_selected.shape[1])):
                    accel = np.zeros(X.shape[0])
                    if period*2 < X.shape[0]:
                        momentum1 = X_selected[period:, j] - X_selected[:-period, j]
                        momentum2 = X_selected[:-period, j] - X_selected[:-period*2, j][:len(momentum1)]
                        accel[period*2:] = momentum1[period:] - momentum2
                    new_features.append(accel.reshape(-1, 1))
            if new_features:
                features_list.append(np.hstack(new_features))
        
        return np.hstack(features_list)

class PatternFeaturesStep(BaseStep):
    """Enhanced pattern features for reversal detection"""
    
    STEP_TYPE = StepType.PATTERN_FEATURES
    ALLOW_SKIP = True
    
    CONFIGURATIONS = [
        {'method': 'extrema'},
        {'method': 'reversal_strength'},
        {'method': 'zigzag'},
    ]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'PatternFeaturesStep':
        variances = np.var(X, axis=0)
        n_top = min(15, X.shape[1])
        self._state['pattern_features'] = np.argsort(variances)[-n_top:]
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        features_list = [X]
        
        if self.config.method == "extrema":
            new_features = []
            for idx in self._state['pattern_features'][:5]:
                for window in [3, 5, 7]:
                    local_max = np.zeros(X.shape[0])
                    local_min = np.zeros(X.shape[0])
                    for i in range(window, X.shape[0] - window):
                        if X[i, idx] == np.max(X[i-window:i+window+1, idx]):
                            local_max[i] = 1
                        if X[i, idx] == np.min(X[i-window:i+window+1, idx]):
                            local_min[i] = 1
                    new_features.append(local_max.reshape(-1, 1))
                    new_features.append(local_min.reshape(-1, 1))
            if new_features:
                features_list.append(np.hstack(new_features))
        
        elif self.config.method == "reversal_strength":
            new_features = []
            for idx in self._state['pattern_features'][:5]:
                reversal_score = np.zeros(X.shape[0])
                window = 10
                for i in range(window*2, X.shape[0]):
                    left_trend = X[i-window, idx] - X[i-window*2, idx]
                    right_trend = X[i, idx] - X[i-window, idx]
                    if left_trend * right_trend < 0:  # Opposite signs = potential reversal
                        reversal_score[i] = abs(left_trend - right_trend)
                new_features.append(reversal_score.reshape(-1, 1))
            if new_features:
                features_list.append(np.hstack(new_features))
                
        elif self.config.method == "zigzag":
            new_features = []
            for idx in self._state['pattern_features'][:5]:
                zigzag = np.zeros(X.shape[0])
                threshold = np.std(X[:, idx]) * 0.1
                last_extreme = X[0, idx]
                last_extreme_idx = 0
                is_rising = True
                
                for i in range(1, X.shape[0]):
                    if is_rising:
                        if X[i, idx] >= last_extreme:
                            last_extreme = X[i, idx]
                            last_extreme_idx = i
                        elif X[i, idx] < last_extreme - threshold:
                            zigzag[last_extreme_idx] = 1  # Mark as high
                            last_extreme = X[i, idx]
                            last_extreme_idx = i
                            is_rising = False
                    else:
                        if X[i, idx] <= last_extreme:
                            last_extreme = X[i, idx]
                            last_extreme_idx = i
                        elif X[i, idx] > last_extreme + threshold:
                            zigzag[last_extreme_idx] = -1  # Mark as low
                            last_extreme = X[i, idx]
                            last_extreme_idx = i
                            is_rising = True
                
                new_features.append(zigzag.reshape(-1, 1))
            if new_features:
                features_list.append(np.hstack(new_features))
        
        return np.hstack(features_list)

class FeatureEngineeringStep(BaseStep):
    """Feature engineering"""
    
    STEP_TYPE = StepType.FEATURE_ENGINEERING
    ALLOW_SKIP = True
    
    CONFIGURATIONS = [
        {'method': 'polynomial'},
        {'method': 'interactions'},
        {'method': 'ratios'},
    ]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'FeatureEngineeringStep':
        variances = np.var(X, axis=0)
        self._state['top_features'] = np.argsort(variances)[-15:]
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        features = [X]
        
        if self.config.method == "polynomial":
            for idx in self._state['top_features'][:7]:
                col_squared = X[:, idx] ** 2
                features.append(col_squared.reshape(-1, 1))
        
        elif self.config.method == "interactions":
            top_indices = self._state['top_features'][:7]
            for i in range(len(top_indices)):
                for j in range(i+1, min(i+3, len(top_indices))):
                    interaction = X[:, top_indices[i]] * X[:, top_indices[j]]
                    features.append(interaction.reshape(-1, 1))
                    
        elif self.config.method == "ratios":
            top_indices = self._state['top_features'][:7]
            for i in range(len(top_indices)):
                for j in range(i+1, min(i+3, len(top_indices))):
                    ratio = X[:, top_indices[i]] / (X[:, top_indices[j]] + 1e-10)
                    ratio = np.clip(ratio, -100, 100)
                    features.append(ratio.reshape(-1, 1))
        
        return np.hstack(features)

# ============================================================================
# ENHANCED FEATURE REDUCTION WITH MULTIPLE MEDOID CONFIGURATIONS
# ============================================================================

class FeatureReductionStep(BaseStep):
    """Enhanced feature reduction with extensive MEDOID grid"""
    
    STEP_TYPE = StepType.FEATURE_REDUCTION
    ALLOW_SKIP = False
    
    CONFIGURATIONS = [
        # Traditional methods
        {'method': 'variance', 'params': {'n_features': 100}},
        {'method': 'variance', 'params': {'n_features': 200}},
        {'method': 'variance', 'params': {'n_features': 500}},
        {'method': 'statistical', 'params': {'n_features': 150}},
        {'method': 'statistical', 'params': {'n_features': 300}},
        {'method': 'tree_importance', 'params': {'n_features': 200}},
        {'method': 'tree_importance', 'params': {'n_features': 400}},
        
        # PCA variations
        {'method': 'pca', 'params': {'n_components': 50}},
        {'method': 'pca', 'params': {'n_components': 100}},
        {'method': 'pca', 'params': {'n_components': 200}},
        
        # EXTENSIVE MEDOIDS GRID
        {'method': 'medoids', 'params': {'n_clusters': 50}},
        {'method': 'medoids', 'params': {'n_clusters': 75}},
        {'method': 'medoids', 'params': {'n_clusters': 100}},
        {'method': 'medoids', 'params': {'n_clusters': 150}},
        {'method': 'medoids', 'params': {'n_clusters': 200}},
        {'method': 'medoids', 'params': {'n_clusters': 250}},
        {'method': 'medoids', 'params': {'n_clusters': 300}},
        {'method': 'medoids', 'params': {'n_clusters': 400}},
        
        # Other methods
        {'method': 'lda', 'params': {'n_components': 2}},
        {'method': 'mutual_info', 'params': {'n_features': 300}},
    ]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None, **kwargs) -> 'FeatureReductionStep':
        n_features = X.shape[1]
        
        if 'n_features' in self.config.params:
            target_features = self.config.params['n_features']
        elif 'n_components' in self.config.params:
            target_features = self.config.params['n_components']
        elif 'n_clusters' in self.config.params:
            target_features = self.config.params['n_clusters']
        else:
            target_features = min(500, n_features)
        
        target_features = min(target_features, n_features)
        
        if self.config.method == "variance":
            variances = np.var(X, axis=0)
            self._state['selected_features'] = np.argsort(variances)[-target_features:]
        
        elif self.config.method == "statistical":
            if y is not None and target_features < n_features:
                selector = SelectKBest(f_classif, k=target_features)
                selector.fit(X, y)
                self._state['selected_features'] = np.where(selector.get_support())[0]
            else:
                variances = np.var(X, axis=0)
                self._state['selected_features'] = np.argsort(variances)[-target_features:]
                
        elif self.config.method == "mutual_info":
            if y is not None and target_features < n_features:
                # Subsample for speed
                subset_size = min(1000, X.shape[0])
                subset_idx = np.random.choice(X.shape[0], subset_size, replace=False)
                selector = SelectKBest(mutual_info_classif, k=target_features)
                selector.fit(X[subset_idx], y[subset_idx])
                self._state['selected_features'] = np.where(selector.get_support())[0]
            else:
                variances = np.var(X, axis=0)
                self._state['selected_features'] = np.argsort(variances)[-target_features:]
        
        elif self.config.method == "tree_importance":
            if y is not None:
                clf = ExtraTreesClassifier(n_estimators=50, random_state=42, n_jobs=-1, max_depth=7)
                clf.fit(X, y)
                importances = clf.feature_importances_
                self._state['selected_features'] = np.argsort(importances)[-target_features:]
            else:
                variances = np.var(X, axis=0)
                self._state['selected_features'] = np.argsort(variances)[-target_features:]
        
        elif self.config.method == "pca":
            n_components = min(target_features, min(X.shape))
            self._state['pca'] = PCA(n_components=n_components, random_state=42)
            self._state['pca'].fit(X)
            
        elif self.config.method == "lda":
            if y is not None:
                n_components = min(target_features, len(np.unique(y)) - 1)
                self._state['lda'] = LinearDiscriminantAnalysis(n_components=n_components)
                self._state['lda'].fit(X, y)
            else:
                # Fallback to PCA
                n_components = min(target_features, min(X.shape))
                self._state['pca'] = PCA(n_components=n_components, random_state=42)
                self._state['pca'].fit(X)
        
        elif self.config.method == "medoids":
            # Enhanced MEDOIDS implementation
            n_clusters = self.config.params.get('n_clusters', 100)
            n_clusters = min(n_clusters, n_features // 2)
            
            # Pre-select features by variance
            variances = np.var(X, axis=0)
            n_candidates = min(2000, n_features)
            top_features = np.argsort(variances)[-n_candidates:]
            
            if n_clusters < 10:
                self._state['selected_features'] = top_features[:target_features]
            else:
                try:
                    # Sample for correlation computation
                    sample_size = min(800, X.shape[0])
                    sample_idx = np.random.choice(X.shape[0], sample_size, replace=False)
                    X_sampled = X[sample_idx][:, top_features]
                    
                    # Compute distance matrix
                    corr_matrix = np.abs(np.corrcoef(X_sampled.T))
                    np.fill_diagonal(corr_matrix, 1.0)  # Ensure diagonal is 1
                    distance_matrix = 1 - corr_matrix
                    distance_matrix = np.clip(distance_matrix, 0, 2)  # Ensure valid distances
                    
                    # K-Medoids clustering with better parameters
                    kmedoids = KMedoids(
                        n_clusters=n_clusters, 
                        metric='precomputed',
                        init='k-medoids++',
                        random_state=42, 
                        max_iter=100
                    )
                    cluster_labels = kmedoids.fit_predict(distance_matrix)
                    
                    # Select best features from each cluster
                    selected = []
                    for cluster_id in range(n_clusters):
                        cluster_mask = cluster_labels == cluster_id
                        cluster_features = top_features[cluster_mask]
                        
                        if len(cluster_features) > 0:
                            # Select feature with highest variance and importance
                            cluster_vars = variances[cluster_features]
                            
                            # If we have labels, also consider correlation with target
                            if y is not None:
                                correlations = np.array([
                                    abs(np.corrcoef(X[:, f], y)[0, 1]) 
                                    for f in cluster_features
                                ])
                                scores = cluster_vars * correlations
                            else:
                                scores = cluster_vars
                            
                            best_idx = cluster_features[np.argmax(scores)]
                            selected.append(best_idx)
                    
                    # Ensure we have enough features
                    selected = list(set(selected))  # Remove duplicates
                    if len(selected) < target_features:
                        remaining = top_features[~np.isin(top_features, selected)]
                        additional_needed = target_features - len(selected)
                        if len(remaining) >= additional_needed:
                            additional = remaining[-additional_needed:]
                        else:
                            additional = remaining
                        selected.extend(additional)
                    
                    self._state['selected_features'] = np.array(selected[:target_features])
                    
                except Exception as e:
                    print(f"  MEDOIDS failed ({str(e)[:50]}), using variance fallback")
                    self._state['selected_features'] = top_features[:target_features]
        
        self.is_fitted = True
        return self
    
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        
        if 'selected_features' in self._state:
            return X[:, self._state['selected_features']]
        elif 'pca' in self._state:
            return self._state['pca'].transform(X)
        elif 'lda' in self._state:
            return self._state['lda'].transform(X)
        else:
            return X

# ============================================================================
# ENHANCED MODELING WITH MLPs AND ADVANCED ENSEMBLES
# ============================================================================

class ModelingStep(BaseStep):
    """Enhanced modeling with neural networks and advanced ensembles"""
    
    STEP_TYPE = StepType.MODELING
    ALLOW_SKIP = False
    
    CONFIGURATIONS = [
        # Tree-based models
        {'method': 'lightgbm', 'params': {
            'num_leaves': 20, 'learning_rate': 0.05, 'n_estimators': 200,
            'class_weight': {0: 15, 1: 15, 2: 1}, 'max_depth': 7,
            'min_child_samples': 30, 'reg_alpha': 0.5, 'reg_lambda': 0.5
        }},
        {'method': 'lightgbm_v2', 'params': {
            'num_leaves': 31, 'learning_rate': 0.03, 'n_estimators': 300,
            'class_weight': {0: 20, 1: 20, 2: 1}, 'max_depth': 10
        }},
        {'method': 'xgboost', 'params': {
            'max_depth': 6, 'learning_rate': 0.05, 'n_estimators': 250,
            'gamma': 0.1, 'subsample': 0.8, 'colsample_bytree': 0.8
        }},
        {'method': 'catboost', 'params': {
            'depth': 6, 'learning_rate': 0.03, 'iterations': 200,
            'class_weights': {0: 15, 1: 15, 2: 1}, 'l2_leaf_reg': 3
        }},
        {'method': 'random_forest', 'params': {
            'n_estimators': 300, 'max_depth': 12, 'min_samples_leaf': 10,
            'class_weight': {0: 15, 1: 15, 2: 1}, 'max_features': 'sqrt'
        }},
        {'method': 'extra_trees', 'params': {
            'n_estimators': 400, 'max_depth': 15, 'min_samples_leaf': 8,
            'class_weight': {0: 12, 1: 12, 2: 1}
        }},
        {'method': 'hist_gradient', 'params': {
            'max_iter': 200, 'learning_rate': 0.05, 'max_depth': 8,
            'class_weight': {0: 15, 1: 15, 2: 1}
        }},
        
        # Neural Networks (MLPs)
        {'method': 'mlp_small', 'params': {
            'hidden_layers': (100, 50), 'activation': 'relu',
            'learning_rate': 0.001, 'max_iter': 500
        }},
        {'method': 'mlp_medium', 'params': {
            'hidden_layers': (200, 100, 50), 'activation': 'relu',
            'learning_rate': 0.001, 'max_iter': 500, 'early_stopping': True
        }},
        {'method': 'mlp_large', 'params': {
            'hidden_layers': (512, 256, 128, 64), 'activation': 'relu',
            'learning_rate': 0.0005, 'max_iter': 1000, 'early_stopping': True
        }},
        {'method': 'mlp_deep', 'params': {
            'hidden_layers': (256, 128, 64, 32, 16), 'activation': 'tanh',
            'learning_rate': 0.001, 'max_iter': 800
        }},
        
        # Advanced Ensembles
        {'method': 'voting_ensemble', 'params': {}},
        {'method': 'stacking_ensemble', 'params': {}},
        {'method': 'mixed_ensemble', 'params': {}},
    ]
    
    def __init__(self, config: StepConfig):
        super().__init__(config)
        self.model = None
        self.cv_score = None
        self.train_score = None
        self.overfitting_gap = 0
        
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'ModelingStep':
        model_params = self.config.params.copy()
        
        # Create the appropriate model
        if 'lightgbm' in self.config.method:
            self.model = lgb.LGBMClassifier(
                objective='multiclass',
                num_leaves=model_params.get('num_leaves', 31),
                learning_rate=model_params.get('learning_rate', 0.05),
                n_estimators=model_params.get('n_estimators', 200),
                max_depth=model_params.get('max_depth', -1),
                class_weight=model_params.get('class_weight'),
                min_child_samples=model_params.get('min_child_samples', 20),
                reg_alpha=model_params.get('reg_alpha', 0),
                reg_lambda=model_params.get('reg_lambda', 0),
                random_state=42,
                verbose=-1,
                n_jobs=-1
            )
            
        elif self.config.method == "xgboost":
            self.model = xgb.XGBClassifier(
                max_depth=model_params.get('max_depth', 6),
                learning_rate=model_params.get('learning_rate', 0.03),
                n_estimators=model_params.get('n_estimators', 200),
                gamma=model_params.get('gamma', 0),
                subsample=model_params.get('subsample', 1.0),
                colsample_bytree=model_params.get('colsample_bytree', 1.0),
                objective='multi:softprob',
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                n_jobs=-1
            )
            
        elif self.config.method == "catboost":
            self.model = cb.CatBoostClassifier(
                depth=model_params.get('depth', 6),
                learning_rate=model_params.get('learning_rate', 0.03),
                iterations=model_params.get('iterations', 150),
                class_weights=model_params.get('class_weights'),
                l2_leaf_reg=model_params.get('l2_leaf_reg', 3),
                random_seed=42,
                verbose=False,
                loss_function='MultiClass'
            )
            
        elif self.config.method == "random_forest":
            self.model = RandomForestClassifier(
                n_estimators=model_params.get('n_estimators', 200),
                max_depth=model_params.get('max_depth', 10),
                min_samples_leaf=model_params.get('min_samples_leaf', 10),
                max_features=model_params.get('max_features', 'sqrt'),
                class_weight=model_params.get('class_weight'),
                random_state=42,
                n_jobs=-1
            )
            
        elif self.config.method == "extra_trees":
            self.model = ExtraTreesClassifier(
                n_estimators=model_params.get('n_estimators', 250),
                max_depth=model_params.get('max_depth', 12),
                min_samples_leaf=model_params.get('min_samples_leaf', 10),
                class_weight=model_params.get('class_weight'),
                random_state=42,
                n_jobs=-1
            )
            
        elif self.config.method == "hist_gradient":
            # Map class weights for HistGradientBoostingClassifier
            class_weight = model_params.get('class_weight', {0: 1, 1: 1, 2: 1})
            sample_weights = np.ones(len(y))
            for class_idx, weight in class_weight.items():
                sample_weights[y == class_idx] = weight
            
            self.model = HistGradientBoostingClassifier(
                max_iter=model_params.get('max_iter', 100),
                learning_rate=model_params.get('learning_rate', 0.1),
                max_depth=model_params.get('max_depth', None),
                random_state=42
            )
            self._sample_weights = sample_weights
            
        elif 'mlp' in self.config.method:
            hidden_layers = model_params.get('hidden_layers', (100,))
            
            self.model = MLPClassifier(
                hidden_layer_sizes=hidden_layers,
                activation=model_params.get('activation', 'relu'),
                solver='adam',
                learning_rate_init=model_params.get('learning_rate', 0.001),
                max_iter=model_params.get('max_iter', 500),
                early_stopping=model_params.get('early_stopping', False),
                validation_fraction=0.1 if model_params.get('early_stopping', False) else 0.1,
                random_state=42
            )
            
        elif self.config.method == "voting_ensemble":
            estimators = [
                ('lgb', lgb.LGBMClassifier(
                    num_leaves=25, n_estimators=150,
                    class_weight={0: 15, 1: 15, 2: 1},
                    random_state=42, verbose=-1
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=150, max_depth=10,
                    class_weight={0: 15, 1: 15, 2: 1},
                    random_state=42, n_jobs=-1
                )),
                ('mlp', MLPClassifier(
                    hidden_layer_sizes=(128, 64),
                    max_iter=500, random_state=42
                ))
            ]
            self.model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
            
        elif self.config.method == "stacking_ensemble":
            base_estimators = [
                ('lgb', lgb.LGBMClassifier(
                    num_leaves=20, n_estimators=100,
                    class_weight={0: 15, 1: 15, 2: 1},
                    random_state=42, verbose=-1
                )),
                ('rf', RandomForestClassifier(
                    n_estimators=100, max_depth=8,
                    class_weight={0: 15, 1: 15, 2: 1},
                    random_state=42, n_jobs=-1
                )),
            ]
            final_estimator = LogisticRegression(
                class_weight={0: 15, 1: 15, 2: 1},
                random_state=42
            )
            self.model = StackingClassifier(
                estimators=base_estimators,
                final_estimator=final_estimator,
                cv=3,
                n_jobs=-1
            )
            
        elif self.config.method == "mixed_ensemble":
            # A more complex ensemble mixing different model types
            estimators = [
                ('lgb', lgb.LGBMClassifier(
                    num_leaves=31, n_estimators=150,
                    class_weight={0: 12, 1: 12, 2: 1},
                    random_state=42, verbose=-1
                )),
                ('xgb', xgb.XGBClassifier(
                    max_depth=6, n_estimators=150,
                    random_state=42, use_label_encoder=False,
                    eval_metric='logloss'
                )),
                ('mlp', MLPClassifier(
                    hidden_layer_sizes=(200, 100),
                    max_iter=500, random_state=42
                )),
                ('et', ExtraTreesClassifier(
                    n_estimators=150, max_depth=10,
                    class_weight={0: 12, 1: 12, 2: 1},
                    random_state=42, n_jobs=-1
                ))
            ]
            self.model = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        cv_scores = []
        train_scores = []
        
        for train_idx, val_idx in cv.split(X, y):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]
            
            # Clone and fit model
            if hasattr(self.model, 'get_params'):
                model_copy = self.model.__class__(**self.model.get_params())
            else:
                # For complex ensembles, recreate
                model_copy = self._create_model_copy()
            
            if self.config.method == "hist_gradient" and hasattr(self, '_sample_weights'):
                model_copy.fit(X_tr, y_tr, sample_weight=self._sample_weights[train_idx])
            else:
                model_copy.fit(X_tr, y_tr)
            
            # Evaluate
            train_pred = model_copy.predict(X_tr)
            val_pred = model_copy.predict(X_val)
            
            train_scores.append(f1_score(y_tr, train_pred, average='macro'))
            cv_scores.append(f1_score(y_val, val_pred, average='macro'))
        
        self.cv_score = np.mean(cv_scores)
        self.train_score = np.mean(train_scores)
        self.overfitting_gap = self.train_score - self.cv_score
        
        # Fit final model
        if self.config.method == "hist_gradient" and hasattr(self, '_sample_weights'):
            self.model.fit(X, y, sample_weight=self._sample_weights)
        else:
            self.model.fit(X, y)
        
        self.is_fitted = True
        return self
    
    def _create_model_copy(self):
        """Helper to recreate complex models"""
        if self.config.method == "voting_ensemble":
            return VotingClassifier(
                estimators=[
                    ('lgb', lgb.LGBMClassifier(num_leaves=25, n_estimators=150,
                                              class_weight={0: 15, 1: 15, 2: 1},
                                              random_state=42, verbose=-1)),
                    ('rf', RandomForestClassifier(n_estimators=150, max_depth=10,
                                                 class_weight={0: 15, 1: 15, 2: 1},
                                                 random_state=42, n_jobs=-1)),
                    ('mlp', MLPClassifier(hidden_layer_sizes=(128, 64),
                                        max_iter=500, random_state=42))
                ],
                voting='soft', n_jobs=-1
            )
        elif self.config.method == "mixed_ensemble":
            return VotingClassifier(
                estimators=[
                    ('lgb', lgb.LGBMClassifier(num_leaves=31, n_estimators=150,
                                              class_weight={0: 12, 1: 12, 2: 1},
                                              random_state=42, verbose=-1)),
                    ('xgb', xgb.XGBClassifier(max_depth=6, n_estimators=150,
                                             random_state=42, use_label_encoder=False,
                                             eval_metric='logloss')),
                    ('mlp', MLPClassifier(hidden_layer_sizes=(200, 100),
                                        max_iter=500, random_state=42)),
                    ('et', ExtraTreesClassifier(n_estimators=150, max_depth=10,
                                               class_weight={0: 12, 1: 12, 2: 1},
                                               random_state=42, n_jobs=-1))
                ],
                voting='soft', n_jobs=-1
            )
        else:
            return self.model.__class__(**self.model.get_params())
    
    def transform(self, X: np.ndarray, **kwargs) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        return self.model.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Must fit before transform")
        return self.model.predict_proba(X)

# [Keep Pipeline, SmartGridSearchEngine, apply_dynamic_thresholds, and main() similar to before]
# But update SmartGridSearchEngine to handle the new models

class Pipeline:
    """Pipeline with caching"""
    
    def __init__(self, steps: List[BaseStep]):
        self.steps = steps
        self.train_score = None
        self.cv_score = None
        self.overfitting_gap = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, df: Optional[pd.DataFrame] = None) -> 'Pipeline':
        X_current = X.copy()
        
        for step in self.steps:
            if isinstance(step, ModelingStep):
                step.fit(X_current, y)
                self.train_score = step.train_score
                self.cv_score = step.cv_score
                self.overfitting_gap = step.overfitting_gap
            else:
                step.fit(X_current, y, df=df)
                X_current = step.transform(X_current, df=df)
        
        return self
    
    def transform(self, X: np.ndarray, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        X_current = X.copy()
        
        for step in self.steps:
            if not isinstance(step, ModelingStep):
                X_current = step.transform(X_current, df=df)
        
        return X_current
    
    def predict_proba(self, X: np.ndarray, df: Optional[pd.DataFrame] = None) -> np.ndarray:
        X_transformed = self.transform(X, df=df)
        
        for step in self.steps:
            if isinstance(step, ModelingStep):
                return step.predict_proba(X_transformed)
        
        raise ValueError("No modeling step in pipeline")
    
    def get_scores(self) -> Dict[str, float]:
        return {
            'train_score': self.train_score or 0,
            'cv_score': self.cv_score or 0,
            'overfitting_gap': self.overfitting_gap or 0
        }

class SmartGridSearchEngine:
    """Enhanced grid search with focus on diverse models"""
    
    def __init__(self, X_train, y_train, X_test=None, train_df=None, test_df=None, max_runtime_minutes=90):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.train_df = train_df
        self.test_df = test_df
        self.results = []
        self.best_pipeline = None
        self.best_score = -np.inf
        self.max_runtime = timedelta(minutes=max_runtime_minutes)
        self.start_time = None
        
        # Register step classes
        self.step_classes = {
            StepType.IMPUTATION: ImputationStep,
            StepType.SCALING: ScalingStep,
            StepType.UNIVARIATE_OUTLIER: UnivariateOutlierStep,
            StepType.TIME_SERIES_FEATURES: TimeSeriesFeaturesStep,
            StepType.PATTERN_FEATURES: PatternFeaturesStep,
            StepType.FEATURE_ENGINEERING: FeatureEngineeringStep,
            StepType.FEATURE_REDUCTION: FeatureReductionStep,
            StepType.MODELING: ModelingStep,
        }
    
    def sample_strategic_configurations(self, step_order, n_samples=50):
        """Sample configurations with focus on diversity"""
        
        # Prioritize different reduction methods and models
        reduction_priority = [
            'medoids_150', 'medoids_200', 'medoids_250', 'medoids_300',
            'pca_100', 'pca_200', 
            'variance_200', 'statistical_300', 
            'tree_importance_200', 'mutual_info_300'
        ]
        
        model_priority = [
            'mlp_medium', 'mlp_large', 'mlp_deep',
            'lightgbm', 'lightgbm_v2',
            'xgboost', 'catboost',
            'mixed_ensemble', 'stacking_ensemble',
            'random_forest', 'extra_trees'
        ]
        
        pipelines = []
        
        # Create combinations
        for reduction_spec in reduction_priority:
            for model_spec in model_priority:
                if len(pipelines) >= n_samples:
                    break
                    
                config = []
                
                for step_type in step_order:
                    step_class = self.step_classes.get(step_type)
                    if not step_class:
                        continue
                    
                    if step_type == StepType.FEATURE_REDUCTION:
                        # Parse reduction spec
                        if 'medoids' in reduction_spec:
                            n_clusters = int(reduction_spec.split('_')[1])
                            matching = [c for c in step_class.CONFIGURATIONS 
                                      if c['method'] == 'medoids' and 
                                      c.get('params', {}).get('n_clusters') == n_clusters]
                        elif 'pca' in reduction_spec:
                            n_components = int(reduction_spec.split('_')[1])
                            matching = [c for c in step_class.CONFIGURATIONS 
                                      if c['method'] == 'pca' and 
                                      c.get('params', {}).get('n_components') == n_components]
                        else:
                            method = reduction_spec.split('_')[0]
                            matching = [c for c in step_class.CONFIGURATIONS 
                                      if c['method'] == method]
                        
                        if matching:
                            chosen = matching[0]
                        else:
                            continue
                            
                    elif step_type == StepType.MODELING:
                        matching = [c for c in step_class.CONFIGURATIONS 
                                  if c['method'] == model_spec or 
                                  (model_spec in c['method'])]
                        if matching:
                            chosen = matching[0]
                        else:
                            continue
                    else:
                        # Random selection for other steps
                        configs = step_class.get_configurations()
                        valid = [c for c in configs if c is not None]
                        if valid:
                            chosen_config = random.choice(valid)
                            chosen = chosen_config.to_dict()
                        else:
                            continue
                    
                    config.append(StepConfig(
                        step_type=step_type,
                        method=chosen['method'],
                        params=chosen.get('params', {})
                    ))
                
                if self._is_valid_pipeline(config):
                    pipelines.append(config)
        
        return pipelines[:n_samples]
    
    def _is_valid_pipeline(self, steps):
        """Check if pipeline has required steps"""
        has_imputation = any(s.step_type == StepType.IMPUTATION for s in steps)
        has_modeling = any(s.step_type == StepType.MODELING for s in steps)
        has_reduction = any(s.step_type == StepType.FEATURE_REDUCTION for s in steps)
        return has_imputation and has_modeling and has_reduction
    
    def evaluate_pipeline(self, config_list):
        """Evaluate a single pipeline configuration"""
        try:
            eval_start = time.time()
            
            steps = []
            for config in config_list:
                step_class = self.step_classes.get(config.step_type)
                if step_class:
                    steps.append(step_class(config))
            
            pipeline = Pipeline(steps)
            pipeline.fit(self.X_train, self.y_train, df=self.train_df)
            
            scores = pipeline.get_scores()
            X_transformed = pipeline.transform(self.X_train, df=self.train_df)
            
            eval_time = time.time() - eval_start
            
            return {
                'pipeline': pipeline,
                'cv_score': scores['cv_score'],
                'train_score': scores['train_score'],
                'overfitting_gap': scores['overfitting_gap'],
                'feature_shape': X_transformed.shape,
                'description': ' â†’ '.join([s.name for s in steps]),
                'eval_time': eval_time
            }
            
        except Exception as e:
            print(f"  âœ— Error: {str(e)[:100]}")
            return None
    
    def run(self, step_order, max_iterations=50):
        """Run grid search"""
        self.start_time = datetime.now()
        
        configs = self.sample_strategic_configurations(step_order, max_iterations)
        
        print(f"Testing {len(configs)} unique pipeline configurations")
        print(f"Initial data shape: {self.X_train.shape}")
        print("="*60)
        
        for i, config_list in enumerate(configs):
            if (datetime.now() - self.start_time) > self.max_runtime:
                print(f"\nâ�° Time limit reached after {i} configurations")
                break
            
            # Short description for monitoring
            reduction_step = [s for s in config_list if s.step_type == StepType.FEATURE_REDUCTION][0]
            model_step = [s for s in config_list if s.step_type == StepType.MODELING][0]
            
            print(f"\n[{i+1}/{len(configs)}] Reduction: {reduction_step.method} | Model: {model_step.method}")
            
            result = self.evaluate_pipeline(config_list)
            
            if result:
                self.results.append(result)
                
                overfitting_warning = "âš ï¸� " if result['overfitting_gap'] > 0.2 else ""
                print(f"  âœ“ CV: {result['cv_score']:.4f} | Train: {result['train_score']:.4f} | "
                      f"Gap: {result['overfitting_gap']:.4f} {overfitting_warning}| "
                      f"Shape: {result['feature_shape']} | Time: {result['eval_time']:.1f}s")
                
                if result['cv_score'] > self.best_score:
                    self.best_score = result['cv_score']
                    self.best_pipeline = result['pipeline']
                    print(f"  â˜… New best CV score!")
        
        total_time = (datetime.now() - self.start_time).total_seconds() / 60
        print(f"\nTotal runtime: {total_time:.1f} minutes")
        
        return self.best_pipeline

def apply_dynamic_thresholds(probabilities, target_h_pct=0.03, target_l_pct=0.029):
    """Apply dynamic thresholds for class balancing"""
    n_samples = len(probabilities)
    
    target_h_count = max(1, int(n_samples * target_h_pct))
    target_l_count = max(1, int(n_samples * target_l_pct))
    
    h_probs = probabilities[:, 0]
    l_probs = probabilities[:, 1]
    
    predictions = np.full(n_samples, 2)  # Default to None
    
    # Select top H predictions
    h_indices = np.argsort(h_probs)[-target_h_count:]
    predictions[h_indices] = 0
    
    # Select top L predictions (excluding already assigned H)
    l_candidates = [(i, l_probs[i]) for i in range(n_samples) if predictions[i] == 2]
    l_candidates.sort(key=lambda x: x[1], reverse=True)
    
    for idx, _ in l_candidates[:target_l_count]:
        predictions[idx] = 1
    
    return predictions

def main():
    """Main execution with enhanced models"""
    
    print("Loading competition data...")
    
    # Load data
    train = pd.read_csv('/kaggle/input/detecting-reversal-points-in-us-equities/competition_data/train.csv')
    test = pd.read_csv('/kaggle/input/detecting-reversal-points-in-us-equities/competition_data/test.csv')
    
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    
    # Map targets
    target_mapping = {'HH': 'H', 'LH': 'H', 'HL': 'L', 'LL': 'L'}
    train['class_label_mapped'] = train['class_label'].map(target_mapping).fillna('None')
    
    # Label encoding
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.array(['H', 'L', 'None'])
    y_train = label_encoder.transform(train['class_label_mapped'])
    
    # Verify encoding
    print("\nLabel encoding verification:")
    unique_vals, counts = np.unique(y_train, return_counts=True)
    for val, count in zip(unique_vals, counts):
        original_label = label_encoder.inverse_transform([val])[0]
        print(f"  {original_label} -> {val}: {count} samples ({count/len(y_train)*100:.1f}%)")
    
    # Extract features
    metadata_cols = ['ticker_id', 't', 'class_label', 'class_label_mapped', 'train_id', 'id']
    feature_cols = [col for col in train.columns if col not in metadata_cols]
    
    X_train = train[feature_cols].values.astype(np.float32)
    X_test = test[feature_cols].values.astype(np.float32)
    
    test_ids = test['id'].values if 'id' in test.columns else range(len(test))
    
    print(f"\nNumber of features: {len(feature_cols)}")
    
    # Keep metadata
    train_metadata = train[['ticker_id', 't']] if 'ticker_id' in train.columns else None
    test_metadata = test[['ticker_id', 't']] if 'ticker_id' in test.columns else None
    
    # Define step order
    step_order = [
        StepType.IMPUTATION,
        StepType.SCALING,
        StepType.UNIVARIATE_OUTLIER,
        StepType.PATTERN_FEATURES,
        StepType.TIME_SERIES_FEATURES,
        StepType.FEATURE_ENGINEERING,
        StepType.FEATURE_REDUCTION,
        StepType.MODELING
    ]
    
    # Run grid search
    print("\n" + "="*60)
    print("ENHANCED GRID SEARCH WITH NEURAL NETWORKS")
    print("="*60)
    
    search = SmartGridSearchEngine(
        X_train, y_train, X_test,
        train_df=train_metadata,
        test_df=test_metadata,
        max_runtime_minutes=90
    )
    
    best_pipeline = search.run(step_order, max_iterations=40)
    
    # Generate predictions
    if best_pipeline and X_test is not None:
        print("\n" + "="*60)
        print("Generating predictions...")
        print("="*60)
        
        probabilities = best_pipeline.predict_proba(X_test, df=test_metadata)
        predictions = apply_dynamic_thresholds(probabilities)
        
        predicted_labels = label_encoder.inverse_transform(predictions)
        
        submission = pd.DataFrame({
            'id': test_ids,
            'class_label': predicted_labels
        })
        
        print("\nSubmission distribution:")
        print(submission['class_label'].value_counts())
        print("\nSubmission percentages:")
        print(submission['class_label'].value_counts(normalize=True))
        
        submission.to_csv('submission.csv', index=False)
        print("\nâœ“ Submission saved to submission.csv")
        
        # Summary
        print("\n" + "="*60)
        print("FINAL RESULTS SUMMARY")
        print("="*60)
        print(f"Best CV score: {search.best_score:.4f}")
        
        # Top configurations
        sorted_results = sorted(search.results, key=lambda x: x['cv_score'], reverse=True)
        
        print("\nTop 10 configurations:")
        for i, result in enumerate(sorted_results[:10]):
            gap_indicator = "ğŸ”´" if result['overfitting_gap'] > 0.3 else "ğŸŸ¡" if result['overfitting_gap'] > 0.15 else "ğŸŸ¢"
            
            # Extract model and reduction method
            parts = result['description'].split(' â†’ ')
            reduction = [p for p in parts if 'feature_reduction' in p]
            model = [p for p in parts if 'modeling' in p]
            
            print(f"\n{i+1}. CV: {result['cv_score']:.4f} | Train: {result['train_score']:.4f} | "
                  f"Gap: {result['overfitting_gap']:.4f} {gap_indicator}")
            
            if reduction:
                print(f"   Reduction: {reduction[0].split('_')[2]}")
            if model:
                print(f"   Model: {model[0].replace('modeling_', '')}")
    
    print("\nDone!")

if __name__ == "__main__":
    main()

