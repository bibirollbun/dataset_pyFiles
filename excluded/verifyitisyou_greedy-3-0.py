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


# Install required packages
!pip install numpy pandas scikit-learn scipy psutil
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install gspread google-auth google-auth-oauthlib google-auth-httplib2
!pip install pyarrow  # For parquet file support
!pip install gputil


#!/usr/bin/env python3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, Subset
from torch.cuda.amp import GradScaler, autocast
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer, MinMaxScaler
from sklearn.model_selection import KFold, TimeSeriesSplit, StratifiedKFold, GroupKFold
from sklearn.decomposition import PCA, TruncatedSVD, FastICA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import pearsonr, spearmanr, rankdata, zscore
from scipy.special import expit, logit
import warnings
import gc
import psutil
import time
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from collections import defaultdict
import random
import json
import hashlib
from datetime import datetime, timedelta
import os
import itertools
from dataclasses import dataclass, field
import traceback
import signal
import sys
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import threading
import math
import pickle
import socket
import uuid
from functools import partial

try:
    import gspread
    from google.oauth2.service_account import Credentials
    GSPREAD_AVAILABLE = True
except ImportError:
    GSPREAD_AVAILABLE = False
    print("Warning: Google Sheets libraries not available. Install with: pip install gspread google-auth")

warnings.filterwarnings('ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def to_python_type(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif torch is not None and isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, list):
        return [to_python_type(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: to_python_type(value) for key, value in obj.items()}
    else:
        return obj

def make_hashable(obj):
    if isinstance(obj, dict):
        return tuple(sorted((k, make_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(make_hashable(item) for item in obj)
    elif isinstance(obj, set):
        return tuple(sorted(make_hashable(item) for item in obj))
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    elif isinstance(obj, tuple):
        return tuple(make_hashable(item) for item in obj)
    else:
        return str(obj)

@dataclass
class NoiseDetectionConfig:
    use_isolation_forest: bool = True
    use_local_outlier_factor: bool = True
    use_statistical_detection: bool = True
    use_label_consistency: bool = True
    
    isolation_contamination: float = 0.05
    lof_contamination: float = 0.05
    lof_novelty: bool = True  # Fixed: Set to True to enable predict()
    statistical_threshold: float = 3.0
    label_consistency_window: int = 100
    label_consistency_threshold: float = 0.1
    
    consensus_threshold: float = 0.5

@dataclass
class FeatureGroupConfig:
    market_features: List[str] = field(default_factory=lambda: [
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
    ])
    
    microstructure_features: List[str] = field(default_factory=lambda: [
        "volume_weighted_sell", "buy_sell_ratio", 
        "selling_pressure", "effective_spread_proxy",
        "bid_ask_imbalance", "flow_toxicity",
        "volume_concentration", "liquidity_consumption",
        "price_pressure", "order_imbalance", "relative_spread",
        "liquidity_ratio", "trade_intensity", "volume_imbalance",
        "price_efficiency", "market_depth_ratio"
    ])
    
    core_proprietary_features: List[str] = field(default_factory=lambda: [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", 
        "X674", "X415", "X345", "X855", "X174", "X302", "X178", "X168", 
        "X612", "X888", "X421", "X333", "X292"
    ])
    
    # Updated dropout rates closer to 0.5
    dropout_rates: Dict[str, float] = field(default_factory=lambda: {
        'market': 0.45,
        'microstructure': 0.5,
        'core_proprietary': 0.55,
        'other_proprietary': 0.6,
        'interactions': 0.5,
        'engineered': 0.5
    })
    
    noise_levels: Dict[str, float] = field(default_factory=lambda: {
        'market': 0.005,
        'microstructure': 0.005,
        'core_proprietary': 0.005,
        'other_proprietary': 0.01,
        'interactions': 0.005,
        'engineered': 0.005
    })
    
    lr_multipliers: Dict[str, float] = field(default_factory=lambda: {
        'market': 1.0,
        'microstructure': 0.8,
        'core_proprietary': 0.5,
        'other_proprietary': 0.3,
        'interactions': 0.6,
        'engineered': 0.7
    })
    
    def get_feature_indices(self, all_features: List[str]) -> Dict[str, List[int]]:
        indices = {}
        
        indices['market'] = [
            i for i, f in enumerate(all_features) 
            if f in self.market_features
        ]
        
        indices['microstructure'] = [
            i for i, f in enumerate(all_features) 
            if f in self.microstructure_features
        ]
        
        indices['core_proprietary'] = [
            i for i, f in enumerate(all_features) 
            if f in self.core_proprietary_features
        ]
        
        indices['other_proprietary'] = [
            i for i, f in enumerate(all_features) 
            if f.startswith('X') and f not in self.core_proprietary_features
        ]
        
        indices['engineered'] = [
            i for i, f in enumerate(all_features)
            if any(x in f for x in ['_ratio_', '_imbalance', '_pressure', '_concentration'])
        ]
        
        return indices

def create_advanced_microstructure_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    eps = 1e-8
    
    df['volume_weighted_sell'] = df['sell_qty'].values * df['volume'].values
    df['buy_sell_ratio'] = df['buy_qty'].values / (df['sell_qty'].values + eps)
    df['selling_pressure'] = df['sell_qty'].values / (df['volume'].values + eps)
    df['effective_spread_proxy'] = np.abs(df['buy_qty'].values - df['sell_qty'].values) / (df['volume'].values + eps)
    
    df['bid_ask_imbalance'] = (
        (df['bid_qty'] - df['ask_qty']) / 
        (df['bid_qty'] + df['ask_qty'] + eps)
    )
    
    df['flow_toxicity'] = df['sell_qty'] / (df['buy_qty'] + eps)
    
    df['volume_concentration'] = (
        (df['buy_qty'] + df['sell_qty']) / (df['volume'] + eps)
    )
    
    df['liquidity_consumption'] = (
        (df['buy_qty'] + df['sell_qty']) / 
        (df['bid_qty'] + df['ask_qty'] + eps)
    )
    
    df['price_pressure'] = (
        (df['buy_qty'] - df['sell_qty']) / (df['volume'] + eps)
    )
    
    df['order_imbalance'] = (
        df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + eps)
    )
    
    df['relative_spread'] = (
        np.abs(df['buy_qty'] - df['sell_qty']) / 
        (df['bid_qty'] + df['ask_qty'] + eps)
    )
    
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + eps)
    df['trade_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + eps)
    df['volume_imbalance'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + eps)
    df['price_efficiency'] = 1 - np.abs(df['order_imbalance'] - 0.5) * 2
    df['market_depth_ratio'] = df['bid_qty'] / (df['ask_qty'] + eps)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df

class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    CREDENTIALS_FILE = '/kaggle/input/greedy3/forward-leaf-464213-u2-489add6f6da1.json'
    SPREADSHEET_URL = 'https://docs.google.com/spreadsheets/d/1nBW-fJxYwtopFNFuTuqHirBsK9QkjojXJU8DmaB8OnI/edit?gid=0#gid=0'
    
    FEATURE_GROUP_CONFIG = FeatureGroupConfig()
    NOISE_DETECTION_CONFIG = NoiseDetectionConfig()
    
    CORE_FEATURES = (
        FEATURE_GROUP_CONFIG.market_features +
        FEATURE_GROUP_CONFIG.microstructure_features +
        FEATURE_GROUP_CONFIG.core_proprietary_features
    )
    
    MAX_BATCH_SIZE_PREDICT = 10000
    CLEAR_MEMORY_INTERVAL = 5
    USE_MIXED_PRECISION = True
    
    USE_MULTI_GPU = False
    DEVICE_IDS = [0]
    
    BASELINE_LAYERS = [256, 128, 64, 1]
    BASELINE_DROPOUT = 0.5
    BASELINE_NOISE = 0.005
    
    MAX_RUNTIME_HOURS = 8.0
    CHECKPOINT_INTERVAL_MINUTES = 30
    STALE_EXPERIMENT_HOURS = 12.0
    
    MIN_MEMORY_GB = 2.0
    MEMORY_CHECK_INTERVAL = 10
    
    MIN_ADDITIONAL_FEATURES = 1
    MAX_ADDITIONAL_FEATURES = 30
    CONFIGS_PER_FEATURE_SET = 50
    
    USE_FULL_DATASET = True
    SAMPLE_SIZE_FOR_TESTING = None
    
    # Updated batch size as requested
    BATCH_SIZE = 1024 * 8 * 4  # 32768
    MAX_EPOCHS = 50
    EARLY_STOPPING_PATIENCE = 7
    
    N_FOLDS = 7
    MIN_FOLDS_FOR_VALID_SCORE = 5
    
    MAX_TRAIN_VAL_GAP = 0.05
    MIN_ACCEPTABLE_VAL_SCORE = 0.001
    MAX_ACCEPTABLE_TRAIN_SCORE = 0.35
    MIN_VAL_STD = 0.0001
    MAX_VAL_STD = 0.04
    CONSISTENCY_THRESHOLD = 0.025
    
    MC_DROPOUT_SAMPLES = 5
    MAX_PREDICTION_UNCERTAINTY = 0.15
    
    INFINITY_STRATEGIES = ['median', 'percentile', 'zero', 'winsorize']
    
    # Removed 'power' transform due to errors
    FEATURE_TRANSFORMS = [
        ['standard'],
        ['rank'],
        ['quantile'],
        ['robust'],
        ['standard', 'rank'],
        ['robust', 'rank']
    ]
    
    LABEL_TRANSFORMS = ['none', 'rank', 'quantile', 'log1p']
    
    # Complex architecture variants as requested
    ARCHITECTURE_VARIANTS = [
        {'name': 'baseline_mlp', 'hidden_dims': [256, 128, 64, 1], 'type': 'standard'},
        {'name': 'deep', 'hidden_dims': [512, 256, 128, 64, 32, 1], 'type': 'deep'},
        {'name': 'wide', 'hidden_dims': [1024, 256, 64, 1], 'type': 'wide'},
        {'name': 'pyramid', 'hidden_dims': [256, 128, 64, 32, 16, 1], 'type': 'pyramid'},
        {'name': 'bottleneck', 'hidden_dims': [512, 64, 512, 1], 'type': 'bottleneck'},
        {'name': 'shallow_wide', 'hidden_dims': [2048, 512, 1], 'type': 'shallow'},
        {'name': 'very_deep', 'hidden_dims': [256, 192, 128, 96, 64, 48, 32, 16, 1], 'type': 'very_deep'},
        {'name': 'funnel', 'hidden_dims': [1024, 256, 64, 16, 1], 'type': 'funnel'},
        {'name': 'ultra_deep', 'hidden_dims': [256, 224, 192, 160, 128, 96, 64, 48, 32, 16, 1], 'type': 'ultra_deep'},
        {'name': 'expanding', 'hidden_dims': [64, 128, 256, 512, 256, 128, 64, 1], 'type': 'expanding'}
    ]
    
    ACTIVATIONS = ['relu', 'tanh', 'leaky_relu', 'elu', 'selu', 'gelu']
    
    # Updated dropout rates closer to 0.5
    DROPOUT_RATES = [0.4, 0.45, 0.5, 0.55, 0.6]
    
    # Updated learning rates closer to 0.001
    LEARNING_RATES = [0.0005, 0.0008, 0.001, 0.0012, 0.0015]
    
    # Updated weight decays
    WEIGHT_DECAYS = [0.0001, 0.0005, 0.001, 0.002, 0.005]
    
    # Updated input noise levels
    INPUT_NOISE_LEVELS = [0.001, 0.003, 0.005, 0.008, 0.01]
    
    CV_STRATEGIES = ['kfold', 'stratified', 'random_split']
    
    USE_ENSEMBLE_UNCERTAINTY = True
    ENSEMBLE_THRESHOLD = 0.05
    
    MAX_CONFIGS_TO_TRY = 2000

@dataclass
class ExperimentConfig:
    feature_list: List[str]
    additional_features: List[str]
    
    infinity_strategy: str = 'median'
    feature_transforms: List[str] = field(default_factory=lambda: ['standard'])
    use_interactions: bool = False
    interaction_features: List[str] = field(default_factory=list)
    interaction_method: str = 'multiply'
    use_clustering: bool = False
    n_clusters: int = 10
    use_pca: bool = False
    pca_components: int = 20
    use_binning: bool = False
    binning_method: str = 'quantile'
    binning_features: List[str] = field(default_factory=list)
    n_bins: int = 10
    
    label_transform: str = 'none'
    label_noise: float = 0.0
    
    architecture_name: str = 'baseline_mlp'
    architecture_type: str = 'standard'
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64, 1])
    activation: str = 'relu'
    dropout_rate: float = 0.5
    dropout_rates_dict: Dict[str, float] = field(default_factory=dict)
    noise_levels_dict: Dict[str, float] = field(default_factory=dict)
    lr_multipliers_dict: Dict[str, float] = field(default_factory=dict)
    use_batch_norm: bool = False
    use_layer_norm: bool = False
    use_residual: bool = False
    use_spectral_norm: bool = False
    use_gradient_penalty: bool = False
    gradient_penalty_weight: float = 0.1
    use_attention: bool = False
    attention_heads: int = 4
    
    optimizer: str = 'adam'
    learning_rate: float = 0.001
    weight_decay: float = 0.001
    use_scheduler: str = 'plateau'
    
    input_noise: float = 0.005
    use_mixup: bool = False
    mixup_alpha: float = 0.2
    use_cutmix: bool = False
    cutmix_alpha: float = 1.0
    gradient_clip: float = 1.0
    label_smoothing: float = 0.0
    
    cv_strategy: str = 'kfold'
    n_ensemble: int = 1
    use_mc_dropout: bool = True
    
    use_noise_detection: bool = False
    noise_detection_method: str = 'ensemble'
    
    def get_hash(self) -> str:
        config_dict = {
            'features': sorted(self.additional_features),
            'infinity': self.infinity_strategy,
            'transforms': sorted(self.feature_transforms),
            'interactions': self.use_interactions,
            'interaction_features': sorted(self.interaction_features) if self.interaction_features else [],
            'interaction_method': self.interaction_method,
            'clustering': self.use_clustering,
            'n_clusters': self.n_clusters,
            'pca': self.use_pca,
            'pca_components': self.pca_components,
            'binning': self.use_binning,
            'binning_method': self.binning_method,
            'binning_features': sorted(self.binning_features) if self.binning_features else [],
            'label_transform': self.label_transform,
            'label_noise': self.label_noise,
            'architecture': self.architecture_name,
            'hidden_dims': self.hidden_dims,
            'activation': self.activation,
            'dropout': self.dropout_rate,
            'batch_norm': self.use_batch_norm,
            'layer_norm': self.use_layer_norm,
            'residual': self.use_residual,
            'spectral_norm': self.use_spectral_norm,
            'gradient_penalty': self.use_gradient_penalty,
            'attention': self.use_attention,
            'optimizer': self.optimizer,
            'lr': self.learning_rate,
            'weight_decay': self.weight_decay,
            'scheduler': self.use_scheduler,
            'input_noise': self.input_noise,
            'mixup': self.use_mixup,
            'cutmix': self.use_cutmix,
            'cv_strategy': self.cv_strategy,
            'n_ensemble': self.n_ensemble,
            'mc_dropout': self.use_mc_dropout,
            'noise_detection': self.use_noise_detection
        }
        config_str = json.dumps(config_dict, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]
    
    def get_description(self) -> str:
        parts = []
        
        feature_group_config = Config.FEATURE_GROUP_CONFIG
        
        n_market = len([f for f in self.feature_list if f in feature_group_config.market_features])
        n_micro = len([f for f in self.feature_list if f in feature_group_config.microstructure_features])
        n_core_prop = len([f for f in self.feature_list if f in feature_group_config.core_proprietary_features])
        n_other_prop = len([f for f in self.additional_features if f.startswith('X')])
        
        parts.append(f"M:{n_market} Mi:{n_micro} CP:{n_core_prop} OP:{n_other_prop}")
        parts.append(f"Total_Add:{len(self.additional_features)}")
        parts.append(f"Inf:{self.infinity_strategy[:3]}")
        parts.append(f"Arch:{self.architecture_name}")
        parts.append(f"Drop:{self.dropout_rate}")
        parts.append(f"LR:{self.learning_rate}")
        parts.append(f"CV:{self.cv_strategy[:4]}")
        
        if self.use_noise_detection:
            parts.append("NoiseDetect")
        
        return " | ".join(parts)

def create_baseline_config() -> ExperimentConfig:
    return ExperimentConfig(
        feature_list=Config.CORE_FEATURES,
        additional_features=[],
        architecture_name='baseline_mlp',
        architecture_type='standard',
        hidden_dims=[256, 128, 64, 1],
        activation='relu',
        dropout_rate=0.5,
        learning_rate=0.001,
        weight_decay=0.001,
        input_noise=0.005,
        optimizer='adam',
        use_batch_norm=False,
        use_layer_norm=False,
        use_spectral_norm=False,
        use_gradient_penalty=False,
        label_transform='none',
        cv_strategy='kfold',
        n_ensemble=1,
        use_mc_dropout=True,
        infinity_strategy='median',
        feature_transforms=['standard'],
        dropout_rates_dict=Config.FEATURE_GROUP_CONFIG.dropout_rates,
        noise_levels_dict=Config.FEATURE_GROUP_CONFIG.noise_levels,
        lr_multipliers_dict=Config.FEATURE_GROUP_CONFIG.lr_multipliers,
        use_scheduler='plateau',
        use_noise_detection=False
    )

class MemoryManager:
    def __init__(self):
        self.gpu_available = torch.cuda.is_available()
        self.device_count = torch.cuda.device_count() if self.gpu_available else 0
        
    def get_memory_info(self) -> Dict[str, float]:
        info = {
            'cpu_percent': psutil.virtual_memory().percent,
            'cpu_available_gb': psutil.virtual_memory().available / 1e9,
            'cpu_used_gb': psutil.virtual_memory().used / 1e9
        }
        
        if self.gpu_available:
            for i in range(self.device_count):
                info[f'gpu_{i}_allocated_gb'] = torch.cuda.memory_allocated(i) / 1e9
                info[f'gpu_{i}_reserved_gb'] = torch.cuda.memory_reserved(i) / 1e9
                info[f'gpu_{i}_free_gb'] = (torch.cuda.get_device_properties(i).total_memory - 
                                           torch.cuda.memory_allocated(i)) / 1e9
        
        return info
    
    def clear_memory(self, device=None):
        gc.collect()
        if self.gpu_available:
            if device is not None:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            else:
                for i in range(self.device_count):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
    
    def check_memory_available(self, required_gb: float = 2.0) -> bool:
        info = self.get_memory_info()
        return info['cpu_available_gb'] > required_gb
    
    def get_optimal_device(self) -> torch.device:
        if not self.gpu_available:
            return torch.device('cpu')
        
        return torch.device('cuda:0')

class NoiseDetector:
    def __init__(self, config: NoiseDetectionConfig):
        self.config = config
        self.detectors = {}
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        print("Fitting noise detectors...")
        
        # Ensure no NaN values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if self.config.use_isolation_forest:
            self.detectors['isolation_forest'] = IsolationForest(
                contamination=self.config.isolation_contamination,
                random_state=42,
                n_jobs=-1
            )
            self.detectors['isolation_forest'].fit(X_clean)
        
        if self.config.use_local_outlier_factor:
            # Fixed: Set novelty=True to enable predict()
            self.detectors['lof'] = LocalOutlierFactor(
                contamination=self.config.lof_contamination,
                n_neighbors=min(20, X_clean.shape[0] - 1),
                novelty=self.config.lof_novelty,
                n_jobs=-1
            )
            self.detectors['lof'].fit(X_clean)
        
        if self.config.use_statistical_detection:
            self.feature_means = np.mean(X_clean, axis=0)
            self.feature_stds = np.std(X_clean, axis=0) + 1e-8
        
        if self.config.use_label_consistency:
            self.label_mean = np.mean(y)
            self.label_std = np.std(y) + 1e-8
    
    def predict(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        n_samples = X.shape[0]
        noise_votes = np.zeros(n_samples)
        n_detectors = 0
        
        # Ensure no NaN values
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        if 'isolation_forest' in self.detectors:
            predictions = self.detectors['isolation_forest'].predict(X_clean)
            noise_votes += (predictions == -1).astype(float)
            n_detectors += 1
        
        if 'lof' in self.detectors:
            # Fixed: Now we can use predict() since novelty=True
            predictions = self.detectors['lof'].predict(X_clean)
            noise_votes += (predictions == -1).astype(float)
            n_detectors += 1
        
        if self.config.use_statistical_detection and hasattr(self, 'feature_means'):
            z_scores = np.abs((X_clean - self.feature_means) / self.feature_stds)
            max_z_scores = np.max(z_scores, axis=1)
            noise_votes += (max_z_scores > self.config.statistical_threshold).astype(float)
            n_detectors += 1
        
        if self.config.use_label_consistency and y is not None and hasattr(self, 'label_mean'):
            label_z_scores = np.abs((y - self.label_mean) / self.label_std)
            noise_votes += (label_z_scores > self.config.statistical_threshold).astype(float)
            n_detectors += 1
        
        if n_detectors > 0:
            noise_fraction = noise_votes / n_detectors
            weights = 1 - noise_fraction
            
            weights[noise_fraction >= self.config.consensus_threshold] *= 0.5
        else:
            weights = np.ones(n_samples)
        
        return weights

class EnhancedSheetsTracker:
    def __init__(self, credentials_file: str, spreadsheet_url: str, worker_id: str = None):
        if not GSPREAD_AVAILABLE:
            raise ImportError("gspread not available")
        
        self.gc = gspread.service_account(filename=credentials_file)
        self.spreadsheet = self.gc.open_by_url(spreadsheet_url)
        
        self.worker_id = worker_id or f"{socket.gethostname()}_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        
        # Updated sheet name to V6
        try:
            self.sheet = self.spreadsheet.worksheet("Experiments_Regression_V6")
        except:
            self.sheet = self.spreadsheet.add_worksheet("Experiments_Regression_V6", 50000, 120)
        
        self.columns = [
            'experiment_id', 'timestamp_start', 'timestamp_end', 'status',
            'worker_id', 'config_hash', 'baseline_similarity_score',
            
            'core_features_list', 'n_core_features',
            'market_features_list', 'n_market_features', 
            'microstructure_features_list', 'n_microstructure_features',
            'additional_x_features_list', 'n_additional_x_features',
            'total_features',
            
            'core_features_transform', 'market_features_transform',
            'x_features_transform', 'global_transforms',
            'infinity_strategy',
            
            'use_noise_detection', 'noise_detection_method', 'noise_samples_filtered',
            
            'use_interactions', 'interaction_pairs', 'n_interactions',
            'interaction_method', 'interaction_features_source',
            
            'mean_train_score', 'mean_val_score', 'std_train_score', 'std_val_score',
            'train_val_gap', 'best_fold_score', 'worst_fold_score',
            'fold_consistency', 'n_valid_folds',
            'improvement_over_baseline', 'overfitting_flag', 'overfitting_severity',
            'mc_uncertainty', 'ensemble_uncertainty',
            
            'architecture_name', 'architecture_category', 'hidden_dims', 
            'total_parameters', 'depth', 'width',
            
            'activation', 'dropout_rates_dict', 'optimizer', 'learning_rate',
            'weight_decay', 'noise_levels_dict', 'batch_size', 'input_noise',
            'lr_multipliers_dict', 'use_spectral_norm', 'use_gradient_penalty',
            'use_attention', 'attention_heads', 'label_smoothing',
            
            'use_clustering', 'n_clusters', 'use_pca', 'pca_components',
            'use_binning', 'binning_details',
            
            'label_transform', 'cv_strategy', 'n_folds', 'n_ensemble',
            'gradient_clip', 'use_mixup', 'use_cutmix',
            'use_batch_norm', 'use_layer_norm', 'use_residual',
            'use_mc_dropout', 'use_scheduler',
            
            'training_time_minutes', 'memory_peak_gb', 'gpu_memory_peak_gb',
            'submission_file', 'error_message',
            
            'fold_scores_json', 'fold_train_scores_json', 'config_full_json'
        ]
        
        self._ensure_headers()
        
        self.config_cache = {}
        self._load_config_cache()
        
        self.experiment_rows = {}
        
        self.baseline_score = None
        
        all_values = self.sheet.get_all_values()
        self.next_row = len(all_values) + 1
        
        print(f"Initialized sheets tracker. Worker ID: {self.worker_id}")
        print(f"Next available row: {self.next_row}")
        
    def _ensure_headers(self):
        try:
            current_headers = self.sheet.row_values(1)
            if current_headers != self.columns:
                self.sheet.update('A1', [self.columns])
                print("Updated spreadsheet headers")
        except:
            self.sheet.update('A1', [self.columns])
            print("Created spreadsheet headers")
    
    def _load_config_cache(self):
        try:
            all_values = self.sheet.get_all_values()
            if len(all_values) > 1:
                header_row = all_values[0]
                
                hash_idx = header_row.index('config_hash')
                status_idx = header_row.index('status')
                timestamp_idx = header_row.index('timestamp_start')
                worker_idx = header_row.index('worker_id') if 'worker_id' in header_row else None
                
                for row in all_values[1:]:
                    if len(row) > hash_idx and row[hash_idx]:
                        status = row[status_idx] if len(row) > status_idx else 'Unknown'
                        timestamp = row[timestamp_idx] if len(row) > timestamp_idx else ''
                        worker_id = row[worker_idx] if worker_idx and len(row) > worker_idx else 'Unknown'
                        
                        self.config_cache[row[hash_idx]] = {
                            'status': status,
                            'timestamp': timestamp,
                            'worker_id': worker_id
                        }
            
            print(f"Loaded {len(self.config_cache)} existing configurations from cache")
            
            self._mark_stale_experiments()
            
        except Exception as e:
            print(f"Warning: Failed to load config cache: {e}")
    
    def _mark_stale_experiments(self):
        now = datetime.now()
        stale_count = 0
        
        for config_hash, info in self.config_cache.items():
            if info['status'] in ['Started', 'Running']:
                try:
                    start_time = datetime.fromisoformat(info['timestamp'])
                    if (now - start_time).total_seconds() / 3600 > Config.STALE_EXPERIMENT_HOURS:
                        info['status'] = 'Did not complete in 12 hours'
                        stale_count += 1
                except:
                    pass
        
        if stale_count > 0:
            print(f"Marked {stale_count} stale experiments")
    
    def can_claim_config(self, config: ExperimentConfig) -> bool:
        config_hash = config.get_hash()
        
        if config_hash not in self.config_cache:
            return True
        
        info = self.config_cache[config_hash]
        
        if info['status'] in ['Did not complete in 12 hours', 'Error']:
            return True
        
        if info['status'] in ['Completed', 'Started', 'Running']:
            try:
                start_time = datetime.fromisoformat(info['timestamp'])
                if (datetime.now() - start_time).total_seconds() / 3600 > Config.STALE_EXPERIMENT_HOURS:
                    return True
            except:
                pass
            return False
        
        return True
    
    def atomic_claim_experiment(self, experiment_id: str, config: ExperimentConfig) -> Tuple[bool, int]:
        config_hash = config.get_hash()
        
        self._load_config_cache()
        
        if not self.can_claim_config(config):
            return False, -1
        
        row_data = self._prepare_row_data(config, {
            'experiment_id': experiment_id,
            'timestamp_start': datetime.now().isoformat(),
            'timestamp_end': '',
            'status': 'Started',
            'config_hash': config_hash,
            'worker_id': self.worker_id,
            'baseline_similarity': 0
        })
        
        row_list = []
        for col in self.columns:
            value = row_data.get(col, '')
            row_list.append(str(value) if value is not None else '')
        
        try:
            self.sheet.append_row(row_list, value_input_option='RAW')
            row_number = self.next_row
            self.next_row += 1
            self.experiment_rows[experiment_id] = row_number
            
            self.config_cache[config_hash] = {
                'status': 'Started',
                'timestamp': datetime.now().isoformat(),
                'worker_id': self.worker_id
            }
            
            print(f"âœ“ Claimed experiment {experiment_id} at row {row_number}")
            return True, row_number
            
        except Exception as e:
            print(f"âœ— Failed to claim experiment: {e}")
            return False, -1
    
    def update_experiment_status(self, experiment_id: str, status: str, results: Dict[str, Any] = None):
        if experiment_id not in self.experiment_rows:
            print(f"Warning: No row found for experiment {experiment_id}")
            return
        
        row_number = self.experiment_rows[experiment_id]
        
        try:
            updates = []
            
            updates.append({
                'range': f'D{row_number}',
                'values': [[status]]
            })
            updates.append({
                'range': f'C{row_number}',
                'values': [[datetime.now().isoformat()]]
            })
            
            if results:
                col_indices = {col: i for i, col in enumerate(self.columns)}
                
                metrics = {
                    'mean_val_score': results.get('mean_val_score', 0),
                    'mean_train_score': results.get('mean_train_score', 0),
                    'train_val_gap': results.get('train_val_gap', 0),
                    'overfitting_flag': results.get('overfitting_flag', False),
                    'overfitting_severity': results.get('overfitting_severity', 'None'),
                    'training_time_minutes': results.get('training_time_minutes', 0),
                    'fold_consistency': results.get('fold_consistency', 0),
                    'n_valid_folds': results.get('n_valid_folds', 0),
                    'mc_uncertainty': results.get('mc_uncertainty', 0),
                    'ensemble_uncertainty': results.get('ensemble_uncertainty', 0),
                    'noise_samples_filtered': results.get('noise_samples_filtered', 0)
                }
                
                for metric, value in metrics.items():
                    if metric in col_indices:
                        col_letter = self._get_column_letter(col_indices[metric] + 1)
                        updates.append({
                            'range': f'{col_letter}{row_number}',
                            'values': [[str(value)]]
                        })
            
            self.sheet.batch_update(updates)
            print(f"âœ“ Updated experiment {experiment_id} status to {status}")
            
        except Exception as e:
            print(f"âœ— Failed to update experiment status: {e}")
            traceback.print_exc()
    
    def log_experiment_complete(self, experiment_id: str, config: ExperimentConfig, results: Dict[str, Any]):
        if experiment_id not in self.experiment_rows:
            print(f"Warning: No row found for experiment {experiment_id}")
            return
        
        row_number = self.experiment_rows[experiment_id]
        
        row_data = self._prepare_row_data(config, results)
        
        row_list = []
        for col in self.columns:
            value = row_data.get(col, '')
            row_list.append(str(value) if value is not None else '')
        
        try:
            range_name = f'A{row_number}:{self._get_column_letter(len(self.columns))}{row_number}'
            self.sheet.update(range_name, [row_list], value_input_option='RAW')
            print(f"âœ“ Updated complete results for experiment {experiment_id}")
        except Exception as e:
            print(f"âœ— Failed to update experiment row: {e}")
            traceback.print_exc()
    
    def _prepare_row_data(self, config: ExperimentConfig, results: Dict[str, Any]) -> Dict[str, Any]:
        config_hash = config.get_hash()
        
        feature_group_config = Config.FEATURE_GROUP_CONFIG
        
        market_features = [f for f in config.feature_list if f in feature_group_config.market_features]
        microstructure_features = [f for f in config.feature_list if f in feature_group_config.microstructure_features]
        core_proprietary = [f for f in config.feature_list if f in feature_group_config.core_proprietary_features]
        additional_x = config.additional_features
        
        improvement = 0
        if self.baseline_score and results.get('mean_val_score'):
            improvement = results['mean_val_score'] - self.baseline_score
        
        total_params = 0
        if config.hidden_dims:
            prev_dim = len(config.feature_list)
            for hidden_dim in config.hidden_dims:
                total_params += prev_dim * hidden_dim + hidden_dim
                prev_dim = hidden_dim
        
        fold_scores = to_python_type(results.get('fold_scores', []))
        fold_train_scores = to_python_type(results.get('fold_train_scores', []))
        
        row_data = {
            'experiment_id': results.get('experiment_id', ''),
            'timestamp_start': results.get('timestamp_start', ''),
            'timestamp_end': results.get('timestamp_end', datetime.now().isoformat()),
            'status': results.get('status', 'completed'),
            'worker_id': results.get('worker_id', self.worker_id),
            'config_hash': config_hash,
            'baseline_similarity_score': results.get('baseline_similarity', 0),
            
            'core_features_list': json.dumps(core_proprietary),
            'n_core_features': len(core_proprietary),
            'market_features_list': json.dumps(market_features),
            'n_market_features': len(market_features),
            'microstructure_features_list': json.dumps(microstructure_features),
            'n_microstructure_features': len(microstructure_features),
            'additional_x_features_list': json.dumps(additional_x),
            'n_additional_x_features': len(additional_x),
            'total_features': len(config.feature_list),
            
            'core_features_transform': 'standard',
            'market_features_transform': 'standard',
            'x_features_transform': json.dumps(config.feature_transforms),
            'global_transforms': json.dumps(config.feature_transforms),
            'infinity_strategy': config.infinity_strategy,
            
            'use_noise_detection': config.use_noise_detection,
            'noise_detection_method': config.noise_detection_method,
            'noise_samples_filtered': results.get('noise_samples_filtered', 0),
            
            'use_interactions': config.use_interactions,
            'interaction_pairs': json.dumps(results.get('interaction_pairs', [])),
            'n_interactions': len(results.get('interaction_pairs', [])),
            'interaction_method': config.interaction_method,
            'interaction_features_source': json.dumps(config.interaction_features),
            
            'mean_train_score': round(float(results.get('mean_train_score', 0)), 6),
            'mean_val_score': round(float(results.get('mean_val_score', 0)), 6),
            'std_train_score': round(float(results.get('std_train_score', 0)), 6),
            'std_val_score': round(float(results.get('std_val_score', 0)), 6),
            'train_val_gap': round(float(results.get('train_val_gap', 0)), 6),
            'best_fold_score': round(float(results.get('best_fold_score', 0)), 6),
            'worst_fold_score': round(float(results.get('worst_fold_score', 0)), 6),
            'fold_consistency': round(float(results.get('fold_consistency', 0)), 6),
            'n_valid_folds': results.get('n_valid_folds', 0),
            'improvement_over_baseline': round(improvement, 6),
            'overfitting_flag': results.get('overfitting_flag', False),
            'overfitting_severity': results.get('overfitting_severity', 'None'),
            'mc_uncertainty': round(float(results.get('mc_uncertainty', 0)), 6),
            'ensemble_uncertainty': round(float(results.get('ensemble_uncertainty', 0)), 6),
            
            'architecture_name': config.architecture_name,
            'architecture_category': config.architecture_type,
            'hidden_dims': json.dumps(config.hidden_dims),
            'total_parameters': total_params,
            'depth': len(config.hidden_dims) if config.hidden_dims else 0,
            'width': max(config.hidden_dims) if config.hidden_dims else 0,
            
            'activation': config.activation,
            'dropout_rates_dict': json.dumps(config.dropout_rates_dict),
            'optimizer': config.optimizer,
            'learning_rate': config.learning_rate,
            'weight_decay': config.weight_decay,
            'noise_levels_dict': json.dumps(config.noise_levels_dict),
            'batch_size': Config.BATCH_SIZE,
            'input_noise': config.input_noise,
            'lr_multipliers_dict': json.dumps(config.lr_multipliers_dict),
            'use_spectral_norm': config.use_spectral_norm,
            'use_gradient_penalty': config.use_gradient_penalty,
            'use_attention': config.use_attention,
            'attention_heads': config.attention_heads,
            'label_smoothing': config.label_smoothing,
            
            'use_clustering': config.use_clustering,
            'n_clusters': config.n_clusters,
            'use_pca': config.use_pca,
            'pca_components': config.pca_components,
            'use_binning': config.use_binning,
            'binning_details': json.dumps({
                'method': config.binning_method,
                'features': config.binning_features,
                'n_bins': config.n_bins
            }),
            
            'label_transform': config.label_transform,
            'cv_strategy': config.cv_strategy,
            'n_folds': results.get('n_folds', Config.N_FOLDS),
            'n_ensemble': config.n_ensemble,
            'gradient_clip': config.gradient_clip,
            'use_mixup': config.use_mixup,
            'use_cutmix': config.use_cutmix,
            'use_batch_norm': config.use_batch_norm,
            'use_layer_norm': config.use_layer_norm,
            'use_residual': config.use_residual,
            'use_mc_dropout': config.use_mc_dropout,
            'use_scheduler': config.use_scheduler,
            
            'training_time_minutes': round(float(results.get('training_time_minutes', 0)), 2),
            'memory_peak_gb': round(float(results.get('memory_peak_gb', 0)), 2),
            'gpu_memory_peak_gb': round(float(results.get('gpu_memory_peak_gb', 0)), 2),
            'submission_file': results.get('submission_file', ''),
            'error_message': results.get('error_message', ''),
            
            'fold_scores_json': json.dumps(fold_scores),
            'fold_train_scores_json': json.dumps(fold_train_scores),
            'config_full_json': json.dumps(config.__dict__, default=str)
        }
        
        return row_data
    
    def _get_column_letter(self, n):
        string = ""
        while n > 0:
            n, remainder = divmod(n - 1, 26)
            string = chr(65 + remainder) + string
        return string
    
    def set_baseline_score(self, score: float):
        self.baseline_score = score
        print(f"Set baseline score: {score:.6f}")

# Fixed SpectralNorm implementation
def spectral_norm(module, name='weight', n_power_iterations=1):
    def _spectral_norm_forward_pre_hook(m, input):
        weight = getattr(m, name)
        if weight.ndim == 1:
            return
        
        with torch.no_grad():
            weight_mat = weight.view(weight.size(0), -1)
            u = getattr(m, name + '_u')
            v = getattr(m, name + '_v')
            
            for _ in range(n_power_iterations):
                v = F.normalize(torch.mv(weight_mat.t(), u), dim=0)
                u = F.normalize(torch.mv(weight_mat, v), dim=0)
            
            sigma = torch.dot(u, torch.mv(weight_mat, v))
            weight.data = weight.data / sigma
    
    weight = getattr(module, name)
    if weight.ndim == 1:
        return module
    
    with torch.no_grad():
        weight_mat = weight.view(weight.size(0), -1)
        h, w = weight_mat.size()
        u = F.normalize(weight.new_empty(h).normal_(0, 1), dim=0)
        v = F.normalize(weight.new_empty(w).normal_(0, 1), dim=0)
    
    delattr(module, name)
    module.register_parameter(name, nn.Parameter(weight.data))
    module.register_buffer(name + '_u', u)
    module.register_buffer(name + '_v', v)
    module.register_forward_pre_hook(_spectral_norm_forward_pre_hook)
    
    return module

class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        
        # Ensure dim is divisible by heads
        assert dim % heads == 0, f"dim {dim} must be divisible by heads {heads}"
        
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out = nn.Linear(dim, dim)
        
    def forward(self, x):
        B, D = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x)  # [B, 3*D]
        qkv = qkv.reshape(B, 3, self.heads, self.head_dim)  # [B, 3, heads, head_dim]
        q, k, v = qkv.unbind(1)  # Each is [B, heads, head_dim]
        
        # Compute attention scores
        attn = torch.einsum('bhd,bhd->bh', q, k) * (self.head_dim ** -0.5)  # [B, heads]
        attn = attn.softmax(dim=-1)  # [B, heads]
        
        # Apply attention to values
        out = torch.einsum('bh,bhd->bhd', attn, v)  # [B, heads, head_dim]
        out = out.reshape(B, D)  # [B, D]
        
        return self.out(out)

class HierarchicalMLP(nn.Module):
    def __init__(self, input_dim: int, config: ExperimentConfig, feature_indices: Dict[str, List[int]]):
        super().__init__()
        
        self.config = config
        self.feature_indices = feature_indices
        
        self.group_processors = nn.ModuleDict()
        
        # Process each feature group
        for group_name, indices in feature_indices.items():
            if len(indices) > 0:
                group_dim = len(indices)
                hidden_dim = min(group_dim * 2, 64)
                
                layers = [
                    nn.Linear(group_dim, hidden_dim),
                    self._get_activation(config.activation),
                    nn.Dropout(config.dropout_rates_dict.get(group_name, 0.5)),
                    nn.Linear(hidden_dim, hidden_dim // 2)
                ]
                
                self.group_processors[group_name] = nn.Sequential(*layers)
        
        # Calculate processed dimension
        processed_dim = sum(
            min(len(indices) * 2, 64) // 2 
            for indices in feature_indices.values() 
            if len(indices) > 0
        )
        
        # Attention layer (if enabled)
        if config.use_attention and processed_dim > 0:
            # Ensure processed_dim is divisible by attention_heads
            if processed_dim % config.attention_heads != 0:
                # Adjust processed_dim to be divisible
                processed_dim = ((processed_dim // config.attention_heads) + 1) * config.attention_heads
                self.projection = nn.Linear(
                    sum(min(len(indices) * 2, 64) // 2 
                        for indices in feature_indices.values() 
                        if len(indices) > 0),
                    processed_dim
                )
            else:
                self.projection = None
            
            self.attention = SelfAttention(processed_dim, config.attention_heads)
            self.attention_dropout = nn.Dropout(config.dropout_rate * 0.5)
        else:
            self.attention = None
            self.projection = None
        
        # Main network
        self.main_network = self._build_main_network(processed_dim, config)
        
        self.noise_levels = config.noise_levels_dict
        
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'selu':
            return nn.SELU()
        elif activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def _build_main_network(self, input_dim: int, config: ExperimentConfig):
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(config.hidden_dims[:-1]):
            linear = nn.Linear(prev_dim, hidden_dim)
            
            if config.use_spectral_norm:
                linear = spectral_norm(linear)
            
            layers.append(linear)
            
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self._get_activation(config.activation))
            
            dropout_rate = min(config.dropout_rate + i * 0.05, 0.9)
            layers.append(nn.Dropout(dropout_rate))
            
            if config.use_residual and prev_dim == hidden_dim:
                block = nn.Sequential(*layers[-4:])
                layers = layers[:-4]
                layers.append(ResidualBlock(block))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, config.hidden_dims[-1]))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        group_outputs = []
        
        # Process each feature group
        for group_name, indices in self.feature_indices.items():
            if len(indices) > 0:
                group_features = x[:, indices]
                
                # Add noise during training
                if self.training:
                    noise_level = self.noise_levels.get(group_name, 0.005)
                    noise = torch.randn_like(group_features) * noise_level
                    group_features = group_features + noise
                
                group_output = self.group_processors[group_name](group_features)
                group_outputs.append(group_output)
        
        if group_outputs:
            combined = torch.cat(group_outputs, dim=1)
            
            # Apply projection if needed for attention
            if self.projection is not None:
                combined = self.projection(combined)
            
            # Apply attention if enabled
            if self.attention is not None:
                attended = self.attention(combined)
                combined = combined + self.attention_dropout(attended)
        else:
            combined = x
        
        # Pass through main network
        output = self.main_network(combined)
        
        return output.squeeze()

class ResidualBlock(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.block = block
    
    def forward(self, x):
        return x + self.block(x)

class CryptoModel(nn.Module):
    def __init__(self, input_dim: int, config: ExperimentConfig, feature_indices: Dict[str, List[int]] = None):
        super().__init__()
        self.config = config
        self.input_noise = config.input_noise
        self.use_mc_dropout = config.use_mc_dropout
        
        if feature_indices and any(len(idx) > 0 for idx in feature_indices.values()):
            self.model = HierarchicalMLP(input_dim, config, feature_indices)
        else:
            self.model = self._build_standard_model(input_dim, config)
        
        self.apply(self._init_weights)
    
    def _get_activation(self, activation: str):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'leaky_relu':
            return nn.LeakyReLU(0.1)
        elif activation == 'elu':
            return nn.ELU()
        elif activation == 'selu':
            return nn.SELU()
        elif activation == 'gelu':
            return nn.GELU()
        else:
            return nn.ReLU()
    
    def _build_standard_model(self, input_dim: int, config: ExperimentConfig):
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(config.hidden_dims[:-1]):
            linear = nn.Linear(prev_dim, hidden_dim)
            
            if config.use_spectral_norm:
                linear = spectral_norm(linear)
            
            layers.append(linear)
            
            if config.use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            elif config.use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            
            layers.append(self._get_activation(config.activation))
            
            dropout_rate = min(config.dropout_rate + i * 0.05, 0.9)
            layers.append(nn.Dropout(dropout_rate))
            
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, config.hidden_dims[-1]))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            if self.config.activation in ['relu', 'leaky_relu', 'elu', 'selu']:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                       nonlinearity='relu' if self.config.activation != 'selu' else 'linear')
            else:
                nn.init.xavier_normal_(m.weight, gain=0.5)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x, enable_dropout=False):
        if self.training and self.input_noise > 0:
            noise = torch.randn_like(x) * self.input_noise
            x = x + noise
        
        if enable_dropout and self.use_mc_dropout:
            self.train()
            output = self.model(x)
            self.eval()
            return output
        else:
            return self.model(x)
    
    def mc_forward(self, x, n_samples=10):
        if not self.use_mc_dropout:
            return self.forward(x), torch.zeros(x.shape[0], device=x.device)
        
        outputs = []
        for _ in range(n_samples):
            outputs.append(self.forward(x, enable_dropout=True))
        
        outputs = torch.stack(outputs)
        mean = outputs.mean(dim=0)
        uncertainty = outputs.std(dim=0)
        
        return mean, uncertainty

class FeatureEngineer:
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.fitted = False
        self.transformers = {}
        self.feature_stats = {}
        
    def fit(self, X: pd.DataFrame, y: np.ndarray):
        for col in X.columns:
            values = X[col].values
            finite_mask = np.isfinite(values)
            if finite_mask.sum() > 0:
                finite_values = values[finite_mask]
                self.feature_stats[col] = {
                    'median': np.median(finite_values),
                    'mean': np.mean(finite_values),
                    'std': np.std(finite_values),
                    'p5': np.percentile(finite_values, 5),
                    'p95': np.percentile(finite_values, 95),
                    'p25': np.percentile(finite_values, 25),
                    'p75': np.percentile(finite_values, 75),
                    'p1': np.percentile(finite_values, 1),
                    'p99': np.percentile(finite_values, 99)
                }
        
        X_clean = self._handle_infinity(X.copy())
        
        if 'standard' in self.config.feature_transforms:
            self.transformers['standard'] = StandardScaler()
            self.transformers['standard'].fit(X_clean)
        
        if 'robust' in self.config.feature_transforms:
            self.transformers['robust'] = RobustScaler()
            self.transformers['robust'].fit(X_clean)
        
        if 'quantile' in self.config.feature_transforms:
            self.transformers['quantile'] = QuantileTransformer(
                n_quantiles=min(1000, len(X_clean)),
                output_distribution='normal'
            )
            self.transformers['quantile'].fit(X_clean)
        
        # Removed PowerTransformer due to errors
        
        self.fitted = True
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted:
            raise ValueError("Must fit before transform")
        
        X_clean = self._handle_infinity(X.copy())
        
        all_features = []
        
        if 'standard' in self.config.feature_transforms:
            all_features.append(self.transformers['standard'].transform(X_clean))
        
        if 'robust' in self.config.feature_transforms and 'robust' in self.transformers:
            all_features.append(self.transformers['robust'].transform(X_clean))
        
        if 'rank' in self.config.feature_transforms:
            rank_features = np.apply_along_axis(
                lambda x: rankdata(x, method='average') / (len(x) + 1), 0, X_clean.values
            )
            all_features.append(rank_features)
        
        if 'quantile' in self.config.feature_transforms and 'quantile' in self.transformers:
            all_features.append(self.transformers['quantile'].transform(X_clean))
        
        if len(all_features) == 0:
            return X_clean.values
        
        return np.hstack(all_features)
    
    def _handle_infinity(self, X: pd.DataFrame) -> pd.DataFrame:
        for col in X.columns:
            if col not in self.feature_stats:
                finite_mask = np.isfinite(X[col])
                if finite_mask.sum() > 0:
                    median_val = np.median(X[col][finite_mask])
                    X.loc[~finite_mask, col] = median_val
                else:
                    X[col] = 0
                continue
            
            stats = self.feature_stats[col]
            
            if self.config.infinity_strategy == 'median':
                X.loc[~np.isfinite(X[col]), col] = stats['median']
            elif self.config.infinity_strategy == 'percentile':
                X.loc[X[col] == np.inf, col] = stats['p95']
                X.loc[X[col] == -np.inf, col] = stats['p5']
                X.loc[np.isnan(X[col]), col] = stats['median']
            elif self.config.infinity_strategy == 'zero':
                X.loc[~np.isfinite(X[col]), col] = 0
            elif self.config.infinity_strategy == 'winsorize':
                X.loc[X[col] > stats['p99'], col] = stats['p99']
                X.loc[X[col] < stats['p1'], col] = stats['p1']
                X.loc[np.isnan(X[col]), col] = stats['median']
        
        return X

class ModelTrainer:
    def __init__(self, config: ExperimentConfig, memory_manager: MemoryManager):
        self.config = config
        self.memory_manager = memory_manager
        
    def train_with_cv(self, X_train: np.ndarray, y_train: np.ndarray, 
                      feature_indices: Dict[str, List[int]] = None,
                      sample_weights: Optional[np.ndarray] = None) -> Dict[str, Any]:
        
        fold_train_scores = []
        fold_val_scores = []
        models = []
        mc_uncertainties = []
        
        noise_detector = None
        noise_samples_filtered = 0
        
        if self.config.use_noise_detection:
            noise_config = Config.NOISE_DETECTION_CONFIG
            noise_detector = NoiseDetector(noise_config)
            noise_detector.fit(X_train, y_train)
            
            if sample_weights is None:
                sample_weights = noise_detector.predict(X_train, y_train)
            else:
                sample_weights *= noise_detector.predict(X_train, y_train)
            
            noise_samples_filtered = np.sum(sample_weights < 0.5)
            print(f"Identified {noise_samples_filtered} potentially noisy samples")
        
        # Choose CV strategy
        if self.config.cv_strategy == 'kfold':
            cv = KFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
            splits = list(cv.split(X_train))
            
        elif self.config.cv_strategy == 'stratified':
            y_bins = pd.qcut(y_train, q=10, labels=False, duplicates='drop')
            cv = StratifiedKFold(n_splits=Config.N_FOLDS, shuffle=True, random_state=42)
            splits = list(cv.split(X_train, y_bins))
            
        elif self.config.cv_strategy == 'random_split':
            n_samples = len(X_train)
            indices = np.arange(n_samples)
            np.random.shuffle(indices)
            splits = []
            
            fold_size = n_samples // Config.N_FOLDS
            for i in range(Config.N_FOLDS):
                val_start = i * fold_size
                val_end = (i + 1) * fold_size if i < Config.N_FOLDS - 1 else n_samples
                val_idx = indices[val_start:val_end]
                train_idx = np.concatenate([indices[:val_start], indices[val_end:]])
                splits.append((train_idx, val_idx))
        
        print(f"\nTraining with {len(splits)} folds using {self.config.cv_strategy} strategy")
        
        valid_folds = 0
        
        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            print(f"\nFold {fold_idx + 1}/{len(splits)}")
            
            if fold_idx > 0 and fold_idx % Config.CLEAR_MEMORY_INTERVAL == 0:
                self.memory_manager.clear_memory()
                print("  Cleared GPU memory")
            
            X_fold_train = X_train[train_idx]
            y_fold_train = y_train[train_idx]
            X_fold_val = X_train[val_idx]
            y_fold_val = y_train[val_idx]
            
            fold_sample_weights = None
            if sample_weights is not None:
                fold_sample_weights = sample_weights[train_idx]
            
            try:
                result = self._train_single_model(
                    X_fold_train, y_fold_train, X_fold_val, y_fold_val,
                    feature_indices=feature_indices,
                    sample_weights=fold_sample_weights,
                    seed=42 + fold_idx
                )
                
                train_score, val_score, model, mc_uncertainty = result
                
                if val_score > Config.MIN_ACCEPTABLE_VAL_SCORE and not np.isnan(val_score):
                    valid_folds += 1
                    fold_train_scores.append(train_score)
                    fold_val_scores.append(val_score)
                    models.append(model)
                    mc_uncertainties.append(mc_uncertainty)
                    
                    print(f"  Train: {train_score:.6f}, Val: {val_score:.6f}, Gap: {train_score - val_score:.6f}")
                    print(f"  MC Uncertainty: {mc_uncertainty:.6f}")
                else:
                    print(f"  Fold invalid - Val score: {val_score:.6f}")
                
            except Exception as e:
                print(f"  Fold failed: {str(e)}")
                continue
            
            if fold_idx + 1 - valid_folds > Config.N_FOLDS - Config.MIN_FOLDS_FOR_VALID_SCORE:
                print("  Too many invalid folds, stopping CV")
                break
        
        if not fold_val_scores:
            print("Warning: No valid folds found")
            fold_val_scores = [0.0]
            fold_train_scores = [0.0]
        
        fold_consistency = 0
        if len(fold_val_scores) > 1:
            fold_consistency = np.max(fold_val_scores) - np.min(fold_val_scores)
        
        results = {
            'fold_train_scores': fold_train_scores,
            'fold_scores': fold_val_scores,
            'mean_train_score': np.mean(fold_train_scores) if fold_train_scores else 0,
            'mean_val_score': np.mean(fold_val_scores) if fold_val_scores else 0,
            'std_train_score': np.std(fold_train_scores) if len(fold_train_scores) > 1 else 0,
            'std_val_score': np.std(fold_val_scores) if len(fold_val_scores) > 1 else 0,
            'train_val_gap': (np.mean(fold_train_scores) - np.mean(fold_val_scores)) if fold_train_scores else 0,
            'best_fold_score': np.max(fold_val_scores) if fold_val_scores else 0,
            'worst_fold_score': np.min(fold_val_scores) if fold_val_scores else 0,
            'fold_consistency': fold_consistency,
            'n_valid_folds': valid_folds,
            'models': models,
            'n_folds': len(fold_val_scores),
            'mc_uncertainty': np.mean(mc_uncertainties) if mc_uncertainties else 0,
            'overfitting_flag': False,
            'overfitting_severity': 'None',
            'noise_samples_filtered': noise_samples_filtered
        }
        
        # Check for overfitting
        if results['mean_train_score'] > 0 and results['mean_val_score'] > 0:
            if results['train_val_gap'] > Config.MAX_TRAIN_VAL_GAP:
                results['overfitting_flag'] = True
                results['overfitting_severity'] = 'High'
            
            if results['mean_train_score'] > Config.MAX_ACCEPTABLE_TRAIN_SCORE:
                results['overfitting_flag'] = True
                results['overfitting_severity'] = 'Very High'
            
            if results['std_val_score'] > Config.MAX_VAL_STD:
                results['overfitting_flag'] = True
                results['overfitting_severity'] = 'Unstable'
            
            if results['fold_consistency'] > Config.CONSISTENCY_THRESHOLD:
                results['overfitting_flag'] = True
                results['overfitting_severity'] = 'Inconsistent'
            
            if results['n_valid_folds'] < Config.MIN_FOLDS_FOR_VALID_SCORE:
                results['overfitting_flag'] = True
                results['overfitting_severity'] = 'Insufficient Valid Folds'
            
            if results['mean_train_score'] > 2 * results['mean_val_score'] and results['mean_val_score'] > 0:
                results['overfitting_flag'] = True
                results['overfitting_severity'] = 'Severe'
            
            if results['mc_uncertainty'] > Config.MAX_PREDICTION_UNCERTAINTY:
                results['overfitting_flag'] = True
                results['overfitting_severity'] = 'High Uncertainty'
        
        return results
    
    def _train_single_model(self, X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           feature_indices: Dict[str, List[int]] = None,
                           sample_weights: Optional[np.ndarray] = None,
                           seed: int = 42) -> Tuple[float, float, nn.Module, float]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        # Transform labels
        y_train_transformed = self._transform_labels(y_train)
        y_val_transformed = self._transform_labels(y_val)
        
        # Create model
        model = CryptoModel(X_train.shape[1], self.config, feature_indices).to(device)
        
        # Set up parameter groups
        param_groups = []
        
        base_model = model
        
        if hasattr(base_model.model, 'group_processors'):
            for group_name, processor in base_model.model.group_processors.items():
                lr_mult = self.config.lr_multipliers_dict.get(group_name, 1.0)
                param_groups.append({
                    'params': processor.parameters(),
                    'lr': self.config.learning_rate * lr_mult,
                    'weight_decay': self.config.weight_decay
                })
            
            param_groups.append({
                'params': base_model.model.main_network.parameters(),
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            })
        else:
            param_groups.append({
                'params': model.parameters(),
                'lr': self.config.learning_rate,
                'weight_decay': self.config.weight_decay
            })
        
        # Create optimizer
        if self.config.optimizer == 'adam':
            optimizer = optim.Adam(param_groups)
        elif self.config.optimizer == 'adamw':
            optimizer = optim.AdamW(param_groups)
        elif self.config.optimizer == 'sgd':
            optimizer = optim.SGD(param_groups, momentum=0.9)
        else:
            optimizer = optim.Adam(param_groups)
        
        # Create scheduler
        if self.config.use_scheduler == 'plateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='max', patience=3, factor=0.5, min_lr=1e-7
            )
        elif self.config.use_scheduler == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=Config.MAX_EPOCHS, eta_min=1e-7
            )
        elif self.config.use_scheduler == 'exponential':
            scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        else:
            scheduler = None
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train_transformed)
        )
        
        if sample_weights is not None:
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.FloatTensor(y_train_transformed),
                torch.FloatTensor(sample_weights)
            )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val),
            torch.FloatTensor(y_val_transformed)
        )
        
        # Create data loaders with smaller batch size if memory issues
        try:
            train_loader = DataLoader(
                train_dataset, 
                batch_size=Config.BATCH_SIZE, 
                shuffle=True,
                num_workers=0,
                pin_memory=True if device.type == 'cuda' else False
            )
        except:
            # If batch size is too large, use smaller one
            train_loader = DataLoader(
                train_dataset, 
                batch_size=min(Config.BATCH_SIZE, 8192), 
                shuffle=True,
                num_workers=0,
                pin_memory=False
            )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=min(Config.BATCH_SIZE * 2, 16384), 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Loss function
        criterion = nn.HuberLoss(delta=5.0)
        
        # Scaler for mixed precision
        scaler = GradScaler() if Config.USE_MIXED_PRECISION and device.type == 'cuda' else None
        
        best_val_score = -np.inf
        patience_counter = 0
        best_model_state = None
        
        # Training loop
        for epoch in range(Config.MAX_EPOCHS):
            model.train()
            train_losses = []
            
            for batch_data in train_loader:
                if sample_weights is not None:
                    batch_X, batch_y, batch_weights = batch_data
                    batch_weights = batch_weights.to(device)
                else:
                    batch_X, batch_y = batch_data
                    batch_weights = None
                
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                
                optimizer.zero_grad()
                
                if scaler is not None:
                    with autocast():
                        outputs = model(batch_X)
                        
                        if self.config.label_smoothing > 0:
                            smooth_targets = batch_y * (1 - self.config.label_smoothing) + \
                                           outputs.detach().mean() * self.config.label_smoothing
                            loss = criterion(outputs, smooth_targets)
                        else:
                            loss = criterion(outputs, batch_y)
                        
                        if batch_weights is not None:
                            loss = (loss * batch_weights).mean()
                    
                    scaler.scale(loss).backward()
                    
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    outputs = model(batch_X)
                    
                    if self.config.label_smoothing > 0:
                        smooth_targets = batch_y * (1 - self.config.label_smoothing) + \
                                       outputs.detach().mean() * self.config.label_smoothing
                        loss = criterion(outputs, smooth_targets)
                    else:
                        loss = criterion(outputs, batch_y)
                    
                    if batch_weights is not None:
                        loss = (loss * batch_weights).mean()
                    
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.gradient_clip)
                    
                    optimizer.step()
                
                train_losses.append(loss.item())
            
            # Validation
            model.eval()
            val_preds = []
            val_targets = []
            val_uncertainties = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X = batch_X.to(device)
                    
                    if self.config.use_mc_dropout:
                        preds, uncertainty = base_model.mc_forward(
                            batch_X, n_samples=Config.MC_DROPOUT_SAMPLES
                        )
                        val_uncertainties.extend(uncertainty.detach().cpu().numpy())
                    else:
                        preds = model(batch_X)
                    
                    val_preds.extend(preds.detach().cpu().numpy())
                    val_targets.extend(batch_y.numpy())
            
            val_preds = np.array(val_preds)
            val_targets = np.array(val_targets)
            mc_uncertainty = np.mean(val_uncertainties) if val_uncertainties else 0
            
            # Calculate validation score
            if len(val_preds) > 1 and len(np.unique(val_preds)) > 1:
                val_score = pearsonr(val_targets, val_preds)[0]
                val_score = 0 if np.isnan(val_score) else val_score
            else:
                val_score = 0
            
            # Calculate train score on subset
            model.eval()
            with torch.no_grad():
                train_subset_size = min(len(X_train), 10000)
                train_subset_idx = np.random.choice(len(X_train), train_subset_size, replace=False)
                train_subset = torch.FloatTensor(X_train[train_subset_idx]).to(device)
                train_preds_subset = model(train_subset).detach().cpu().numpy()
                
                if len(train_preds_subset) > 1 and len(np.unique(train_preds_subset)) > 1:
                    train_score = pearsonr(
                        y_train_transformed[train_subset_idx], 
                        train_preds_subset
                    )[0]
                    train_score = 0 if np.isnan(train_score) else train_score
                else:
                    train_score = 0
            
            # Early stopping checks
            if train_score - val_score > Config.MAX_TRAIN_VAL_GAP * 2:
                print(f"    Epoch {epoch}: Stopping due to overfitting")
                break
            
            if train_score > Config.MAX_ACCEPTABLE_TRAIN_SCORE:
                print(f"    Epoch {epoch}: Train score too high ({train_score:.4f})")
                break
            
            # Update learning rate
            if scheduler is not None:
                if self.config.use_scheduler == 'plateau':
                    scheduler.step(val_score)
                else:
                    scheduler.step()
            
            # Save best model
            if val_score > best_val_score:
                best_val_score = val_score
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= Config.EARLY_STOPPING_PATIENCE:
                    break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Final evaluation
        model.eval()
        with torch.no_grad():
            final_train_preds = []
            for i in range(0, len(X_train), Config.MAX_BATCH_SIZE_PREDICT):
                batch = torch.FloatTensor(
                    X_train[i:i+Config.MAX_BATCH_SIZE_PREDICT]
                ).to(device)
                final_train_preds.extend(model(batch).detach().cpu().numpy())
            
            if len(final_train_preds) > 1 and len(np.unique(final_train_preds)) > 1:
                train_score = pearsonr(y_train_transformed, final_train_preds)[0]
                train_score = 0 if np.isnan(train_score) else train_score
            else:
                train_score = 0
            
            # Calculate MC uncertainty
            if self.config.use_mc_dropout:
                val_uncertainties = []
                for i in range(0, len(X_val), Config.MAX_BATCH_SIZE_PREDICT):
                    batch = torch.FloatTensor(
                        X_val[i:i+Config.MAX_BATCH_SIZE_PREDICT]
                    ).to(device)
                    _, uncertainty = base_model.mc_forward(
                        batch, n_samples=Config.MC_DROPOUT_SAMPLES
                    )
                    val_uncertainties.extend(uncertainty.detach().cpu().numpy())
                mc_uncertainty = np.mean(val_uncertainties)
            else:
                mc_uncertainty = 0
        
        model = model.cpu()
        
        return train_score, best_val_score, model, mc_uncertainty
    
    def _transform_labels(self, y: np.ndarray) -> np.ndarray:
        if self.config.label_transform == 'none':
            return y
        elif self.config.label_transform == 'rank':
            return rankdata(y) / (len(y) + 1)
        elif self.config.label_transform == 'quantile':
            transformer = QuantileTransformer(n_quantiles=min(1000, len(y)), output_distribution='normal')
            return transformer.fit_transform(y.reshape(-1, 1)).ravel()
        elif self.config.label_transform == 'log1p':
            min_val = np.min(y)
            if min_val < 0:
                y_shifted = y - min_val + 1
            else:
                y_shifted = y
            return np.log1p(y_shifted)
        return y

class ExperimentRunner:
    def __init__(self, memory_manager: MemoryManager, sheets_tracker: Optional[EnhancedSheetsTracker] = None):
        self.memory_manager = memory_manager
        self.sheets_tracker = sheets_tracker
        self.feature_group_config = Config.FEATURE_GROUP_CONFIG
        
    def run_experiment(self, train_df: pd.DataFrame, test_df: pd.DataFrame,
                      config: ExperimentConfig) -> Dict[str, Any]:
        start_time = time.time()
        experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{random.randint(1000, 9999)}"
        
        if self.sheets_tracker:
            success, row_num = self.sheets_tracker.atomic_claim_experiment(experiment_id, config)
            if not success:
                print(f"Configuration already being processed or completed")
                return {'success': False, 'reason': 'already_claimed'}
        
        print(f"\n{'='*60}")
        print(f"Starting experiment {experiment_id}")
        print(f"Config: {config.get_description()}")
        print(f"{'='*60}")
        
        try:
            if self.sheets_tracker:
                self.sheets_tracker.update_experiment_status(experiment_id, "Running")
            
            if not self.memory_manager.check_memory_available(Config.MIN_MEMORY_GB):
                raise MemoryError("Insufficient memory available")
            
            # Prepare data
            X_train = train_df[config.feature_list]
            y_train = train_df['label'].values
            X_test = test_df[config.feature_list]
            
            print(f"Training on FULL dataset: {len(X_train)} samples")
            print(f"Features: {len(config.feature_list)} total")
            
            # Feature engineering
            engineer = FeatureEngineer(config)
            engineer.fit(X_train, y_train)
            
            X_train_transformed = engineer.transform(X_train)
            X_test_transformed = engineer.transform(X_test)
            
            print(f"Transformed dimensions: {X_train_transformed.shape[1]}")
            
            # Get feature indices for hierarchical processing
            feature_indices = self.feature_group_config.get_feature_indices(config.feature_list)
            
            # Train model with CV
            trainer = ModelTrainer(config, self.memory_manager)
            cv_results = trainer.train_with_cv(X_train_transformed, y_train, feature_indices)
            
            print(f"\nCV Results:")
            print(f"  Mean train: {cv_results['mean_train_score']:.6f} Â± {cv_results['std_train_score']:.6f}")
            print(f"  Mean val: {cv_results['mean_val_score']:.6f} Â± {cv_results['std_val_score']:.6f}")
            print(f"  Train-val gap: {cv_results['train_val_gap']:.6f}")
            print(f"  Fold consistency: {cv_results['fold_consistency']:.6f}")
            print(f"  Valid folds: {cv_results['n_valid_folds']}/{Config.N_FOLDS}")
            print(f"  MC Uncertainty: {cv_results['mc_uncertainty']:.6f}")
            print(f"  Overfitting: {cv_results['overfitting_severity']}")
            if cv_results['noise_samples_filtered'] > 0:
                print(f"  Noise samples filtered: {cv_results['noise_samples_filtered']}")
            
            if self.sheets_tracker:
                self.sheets_tracker.update_experiment_status(
                    experiment_id, 
                    "Generating predictions",
                    cv_results
                )
            
            # Generate predictions
            device = self.memory_manager.get_optimal_device()
            all_preds = []
            all_uncertainties = []
            
            print("\nGenerating predictions...")
            for model_idx, model in enumerate(cv_results['models']):
                print(f"  Model {model_idx + 1}/{len(cv_results['models'])}")
                model = model.to(device)
                model.eval()
                
                model_preds = []
                model_uncertainties = []
                
                for i in range(0, len(X_test_transformed), Config.MAX_BATCH_SIZE_PREDICT):
                    batch = torch.FloatTensor(
                        X_test_transformed[i:i+Config.MAX_BATCH_SIZE_PREDICT]
                    ).to(device)
                    
                    if config.use_mc_dropout:
                        base_model = model
                        preds, uncertainty = base_model.mc_forward(
                            batch, n_samples=Config.MC_DROPOUT_SAMPLES
                        )
                        model_preds.extend(preds.detach().cpu().numpy())
                        model_uncertainties.extend(uncertainty.detach().cpu().numpy())
                    else:
                        with torch.no_grad():
                            preds = model(batch)
                            model_preds.extend(preds.detach().cpu().numpy())
                
                all_preds.append(model_preds)
                if model_uncertainties:
                    all_uncertainties.append(model_uncertainties)
                
                model = model.cpu()
                self.memory_manager.clear_memory(device)
            
            # Ensemble predictions
            test_preds = np.mean(all_preds, axis=0)
            ensemble_uncertainty = np.std(all_preds, axis=0).mean() if len(all_preds) > 1 else 0
            
            # Save submission
            submission_file = f"submission_{config.get_hash()}.csv"
            submission_df = pd.read_csv(Config.SUBMISSION_PATH)
            submission_df['prediction'] = test_preds
            submission_df.to_csv(submission_file, index=False)
            print(f"Saved predictions to {submission_file}")
            
            # Calculate memory usage
            gpu_memory_peak = 0
            if self.memory_manager.gpu_available:
                for i in range(self.memory_manager.device_count):
                    gpu_memory_peak = max(
                        gpu_memory_peak,
                        torch.cuda.max_memory_allocated(i) / 1e9
                    )
            
            # Prepare results
            results = {
                'success': True,
                'experiment_id': experiment_id,
                'timestamp_start': datetime.fromtimestamp(start_time).isoformat(),
                'timestamp_end': datetime.now().isoformat(),
                'mean_train_score': cv_results['mean_train_score'],
                'mean_val_score': cv_results['mean_val_score'],
                'std_train_score': cv_results['std_train_score'],
                'std_val_score': cv_results['std_val_score'],
                'train_val_gap': cv_results['train_val_gap'],
                'best_fold_score': cv_results['best_fold_score'],
                'worst_fold_score': cv_results['worst_fold_score'],
                'fold_consistency': cv_results['fold_consistency'],
                'n_valid_folds': cv_results['n_valid_folds'],
                'fold_scores': cv_results['fold_scores'],
                'fold_train_scores': cv_results['fold_train_scores'],
                'n_folds': cv_results['n_folds'],
                'training_time_minutes': (time.time() - start_time) / 60,
                'memory_peak_gb': self.memory_manager.get_memory_info()['cpu_used_gb'],
                'gpu_memory_peak_gb': gpu_memory_peak,
                'submission_file': submission_file,
                'baseline_similarity': 0,
                'overfitting_flag': cv_results['overfitting_flag'],
                'overfitting_severity': cv_results['overfitting_severity'],
                'mc_uncertainty': cv_results['mc_uncertainty'],
                'ensemble_uncertainty': ensemble_uncertainty,
                'noise_samples_filtered': cv_results.get('noise_samples_filtered', 0),
                'status': 'Completed',
                'worker_id': self.sheets_tracker.worker_id if self.sheets_tracker else 'local'
            }
            
            if self.sheets_tracker:
                self.sheets_tracker.update_experiment_status(experiment_id, "Completed", results)
                self.sheets_tracker.log_experiment_complete(experiment_id, config, results)
            
            # Clean up
            del cv_results['models']
            self.memory_manager.clear_memory()
            
            print(f"âœ“ Experiment completed in {results['training_time_minutes']:.1f} minutes")
            
            return results
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            print(f"âœ— Experiment failed: {error_msg}")
            traceback.print_exc()
            
            results = {
                'success': False,
                'experiment_id': experiment_id,
                'timestamp_start': datetime.fromtimestamp(start_time).isoformat(),
                'timestamp_end': datetime.now().isoformat(),
                'status': 'Error',
                'error_message': error_msg[:500],
                'training_time_minutes': (time.time() - start_time) / 60,
                'worker_id': self.sheets_tracker.worker_id if self.sheets_tracker else 'local'
            }
            
            if self.sheets_tracker:
                self.sheets_tracker.update_experiment_status(experiment_id, "Error", results)
                self.sheets_tracker.log_experiment_complete(experiment_id, config, results)
            
            self.memory_manager.clear_memory()
            
            return results

class RandomSearchConfigGenerator:
    def __init__(self, core_features: List[str], all_features: List[str]):
        self.core_features = core_features
        self.additional_features = [f for f in all_features if f not in core_features and f.startswith('X')]
        self.additional_features.sort()
        
    def generate_random_configs(self, n_configs: int) -> List[ExperimentConfig]:
        configs = []
        
        for _ in range(n_configs):
            # Random number of additional features
            n_additional = random.randint(Config.MIN_ADDITIONAL_FEATURES, 
                                        min(Config.MAX_ADDITIONAL_FEATURES, len(self.additional_features)))
            
            selected_features = random.sample(self.additional_features, n_additional)
            
            feature_list = self.core_features + selected_features
            
            # Sample hyperparameters
            params = self._sample_hyperparameters()
            
            # Random regularization options
            use_spectral_norm = random.random() < 0.3
            use_gradient_penalty = random.random() < 0.2
            use_batch_norm = random.random() < 0.3 and not use_spectral_norm
            use_layer_norm = random.random() < 0.3 and not use_batch_norm
            use_mc_dropout = random.random() < 0.7
            use_attention = random.random() < 0.2 and params['architecture']['name'] not in ['very_deep', 'bottleneck']
            use_residual = random.random() < 0.3
            
            use_noise_detection = random.random() < 0.4
            
            # Create config
            config = ExperimentConfig(
                feature_list=feature_list,
                additional_features=selected_features,
                infinity_strategy=params['infinity_strategy'],
                feature_transforms=params['feature_transforms'],
                label_transform=params['label_transform'],
                architecture_name=params['architecture']['name'],
                architecture_type=params['architecture']['type'],
                hidden_dims=params['architecture']['hidden_dims'],
                activation=params['activation'],
                dropout_rate=params['dropout_rate'],
                learning_rate=params['learning_rate'],
                weight_decay=params['weight_decay'],
                cv_strategy=params['cv_strategy'],
                optimizer=random.choice(['adam', 'adamw']),
                gradient_clip=random.choice([0.5, 1.0, 2.0, 5.0]),
                input_noise=params['input_noise'],
                dropout_rates_dict=Config.FEATURE_GROUP_CONFIG.dropout_rates,
                noise_levels_dict=Config.FEATURE_GROUP_CONFIG.noise_levels,
                lr_multipliers_dict=Config.FEATURE_GROUP_CONFIG.lr_multipliers,
                use_spectral_norm=use_spectral_norm,
                use_gradient_penalty=use_gradient_penalty,
                gradient_penalty_weight=random.choice([0.05, 0.1, 0.2]) if use_gradient_penalty else 0.1,
                use_batch_norm=use_batch_norm,
                use_layer_norm=use_layer_norm,
                use_mc_dropout=use_mc_dropout,
                use_scheduler=random.choice(['plateau', 'cosine', 'exponential', 'none']),
                use_attention=use_attention,
                attention_heads=random.choice([2, 4, 8]) if use_attention else 4,
                use_residual=use_residual,
                label_smoothing=random.choice([0.0, 0.01, 0.02, 0.05]),
                use_noise_detection=use_noise_detection,
                noise_detection_method='ensemble' if use_noise_detection else 'none'
            )
            
            configs.append(config)
        
        return configs
    
    def _sample_hyperparameters(self) -> Dict:
        return {
            'infinity_strategy': random.choice(Config.INFINITY_STRATEGIES),
            'feature_transforms': random.choice(Config.FEATURE_TRANSFORMS),
            'label_transform': random.choice(Config.LABEL_TRANSFORMS),
            'architecture': random.choice(Config.ARCHITECTURE_VARIANTS),
            'activation': random.choice(Config.ACTIVATIONS),
            'dropout_rate': random.choice(Config.DROPOUT_RATES),
            'learning_rate': random.choice(Config.LEARNING_RATES),
            'weight_decay': random.choice(Config.WEIGHT_DECAYS),
            'input_noise': random.choice(Config.INPUT_NOISE_LEVELS),
            'cv_strategy': random.choice(Config.CV_STRATEGIES)
        }

class EnhancedCryptoPipeline:
    def __init__(self, worker_id: str = None):
        self.start_time = time.time()
        self.memory_manager = MemoryManager()
        self.sheets_tracker = None
        self.experiment_runner = None
        self.baseline_config = create_baseline_config()
        self.baseline_score = None
        self.worker_id = worker_id
        
        if GSPREAD_AVAILABLE and os.path.exists(Config.CREDENTIALS_FILE):
            try:
                self.sheets_tracker = EnhancedSheetsTracker(
                    Config.CREDENTIALS_FILE,
                    Config.SPREADSHEET_URL,
                    worker_id=self.worker_id
                )
                self.experiment_runner = ExperimentRunner(
                    self.memory_manager, self.sheets_tracker
                )
                print("âœ“ Google Sheets tracking initialized")
            except Exception as e:
                print(f"âœ— Failed to initialize sheets tracker: {e}")
                self.experiment_runner = ExperimentRunner(self.memory_manager)
        else:
            print("Running without Google Sheets tracking")
            self.experiment_runner = ExperimentRunner(self.memory_manager)
    
    def run(self):
        print("="*80)
        print("CRYPTO PREDICTION PIPELINE - REGRESSION V6")
        print(f"Worker ID: {self.sheets_tracker.worker_id if self.sheets_tracker else 'local'}")
        print(f"Available GPUs: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")
        print("="*80)
        
        print("\nLoading data...")
        train_df = pd.read_parquet(Config.TRAIN_PATH)
        test_df = pd.read_parquet(Config.TEST_PATH)
        
        print("Creating microstructure features...")
        train_df = create_advanced_microstructure_features(train_df)
        test_df = create_advanced_microstructure_features(test_df)
        
        print(f"Train shape: {train_df.shape}")
        print(f"Test shape: {test_df.shape}")
        
        # Check for missing features
        missing_features = [f for f in Config.CORE_FEATURES if f not in train_df.columns]
        if missing_features:
            print(f"Warning: Missing core features: {missing_features}")
            Config.CORE_FEATURES = [f for f in Config.CORE_FEATURES if f in train_df.columns]
        
        print(f"Using {len(Config.CORE_FEATURES)} core features")
        
        # Get all features
        all_features = [col for col in train_df.columns if col not in ['label', 'timestamp']]
        
        # Create config generator
        config_generator = RandomSearchConfigGenerator(Config.CORE_FEATURES, all_features)
        
        print(f"Found {len(config_generator.additional_features)} additional X features")
        
        # Initialize counters
        n_completed = 0
        n_errors = 0
        n_skipped = 0
        best_score = -np.inf
        best_config = None
        
        # Test baseline model
        print("\n" + "="*60)
        print("Testing baseline model")
        print("="*60)
        
        baseline_result = self.experiment_runner.run_experiment(
            train_df, test_df, self.baseline_config
        )
        
        if baseline_result['success']:
            self.baseline_score = baseline_result['mean_val_score']
            best_score = self.baseline_score
            best_config = self.baseline_config
            n_completed += 1
            
            if self.sheets_tracker:
                self.sheets_tracker.set_baseline_score(self.baseline_score)
            
            print(f"\nâœ“ Baseline score: {self.baseline_score:.6f}")
        else:
            if baseline_result.get('reason') != 'already_claimed':
                print("\nâœ— Failed to run baseline model")
                self.baseline_score = 0.01
            else:
                self.baseline_score = 0.01
                print("Baseline already tested, using default score: 0.01")
                n_skipped += 1
        
        # Start random search
        print("\n" + "="*60)
        print("Starting enhanced random search")
        print("="*60)
        
        configs_tried = 0
        
        while configs_tried < Config.MAX_CONFIGS_TO_TRY:
            elapsed_hours = (time.time() - self.start_time) / 3600
            if elapsed_hours >= Config.MAX_RUNTIME_HOURS:
                print(f"\nReached maximum runtime of {Config.MAX_RUNTIME_HOURS} hours")
                break
            
            # Generate batch of configs
            batch_size = min(Config.CONFIGS_PER_FEATURE_SET, Config.MAX_CONFIGS_TO_TRY - configs_tried)
            random_configs = config_generator.generate_random_configs(batch_size)
            
            for config in random_configs:
                configs_tried += 1
                
                print(f"\n--- Config {configs_tried}/{Config.MAX_CONFIGS_TO_TRY} ---")
                print(f"Features: {len(config.additional_features)} additional")
                print(f"Architecture: {config.architecture_name}, CV: {config.cv_strategy}")
                
                # Run experiment
                result = self.experiment_runner.run_experiment(train_df, test_df, config)
                
                if result['success']:
                    n_completed += 1
                    val_score = result['mean_val_score']
                    
                    # Skip if overfitting is severe
                    if result.get('overfitting_severity') in ['Very High', 'Severe', 'Insufficient Valid Folds']:
                        print(f"  Skipping due to {result['overfitting_severity']} overfitting")
                        continue
                    
                    # Update best score
                    if val_score > best_score:
                        best_score = val_score
                        best_config = config
                        improvement = best_score - self.baseline_score
                        print(f"  ðŸŽ¯ New best score: {best_score:.6f} (improvement: {improvement:.6f})")
                        
                elif result.get('reason') == 'already_claimed':
                    n_skipped += 1
                else:
                    n_errors += 1
                
                # Log progress
                if (n_completed + n_errors + n_skipped) % 10 == 0:
                    self._log_progress(n_completed, n_errors, n_skipped, best_score)
                
                # Check runtime
                if (time.time() - self.start_time) / 3600 >= Config.MAX_RUNTIME_HOURS:
                    break
                
                # Clear memory periodically
                if configs_tried % 5 == 0:
                    self.memory_manager.clear_memory()
                    print("  Cleared memory")
        
        # Save summary
        self._save_summary(n_completed, n_errors, n_skipped, best_score, best_config)
    
    def _log_progress(self, n_completed: int, n_errors: int, n_skipped: int, best_score: float):
        elapsed_hours = (time.time() - self.start_time) / 3600
        improvement = best_score - self.baseline_score if self.baseline_score else 0
        
        print(f"\n--- Progress Update ---")
        print(f"Runtime: {elapsed_hours:.2f} hours")
        print(f"Completed: {n_completed}, Skipped: {n_skipped}, Errors: {n_errors}")
        print(f"Best score: {best_score:.6f} (improvement: {improvement:.6f})")
        
        mem_info = self.memory_manager.get_memory_info()
        print(f"Memory: CPU {mem_info['cpu_used_gb']:.1f}GB / {mem_info['cpu_used_gb'] + mem_info['cpu_available_gb']:.1f}GB")
        if self.memory_manager.gpu_available:
            for i in range(self.memory_manager.device_count):
                if f'gpu_{i}_allocated_gb' in mem_info:
                    print(f"  GPU {i}: {mem_info[f'gpu_{i}_allocated_gb']:.1f}GB allocated")
        print("-" * 23)
    
    def _save_summary(self, n_completed: int, n_errors: int, n_skipped: int, 
                     best_score: float, best_config: Optional[ExperimentConfig]):
        print("\n" + "="*80)
        print("PIPELINE COMPLETE")
        print("="*80)
        print(f"Total runtime: {(time.time() - self.start_time) / 3600:.2f} hours")
        print(f"Experiments completed: {n_completed}")
        print(f"Experiments skipped: {n_skipped}")
        print(f"Experiments errored: {n_errors}")
        print(f"Baseline score: {self.baseline_score:.6f}")
        print(f"Best validation score: {best_score:.6f}")
        print(f"Improvement: {best_score - self.baseline_score:.6f}")
        
        if best_config:
            print(f"\nBest configuration:")
            print(f"  {best_config.get_description()}")
            print(f"  Additional features ({len(best_config.additional_features)}): {best_config.additional_features[:5]}...")
            print(f"  Architecture: {best_config.architecture_name} with {best_config.activation} activation")
            print(f"  Regularization: Spectral={best_config.use_spectral_norm}, "
                  f"GradPenalty={best_config.use_gradient_penalty}, MCDropout={best_config.use_mc_dropout}")
            print(f"  Advanced features: Attention={best_config.use_attention}, "
                  f"NoiseDetection={best_config.use_noise_detection}")
            
            # Save best config
            best_config_file = f"best_config_v6_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(best_config_file, 'w') as f:
                json.dump(best_config.__dict__, f, indent=2, default=str)
            print(f"\nBest configuration saved to: {best_config_file}")

def signal_handler(sig, frame):
    print("\n\nReceived interrupt signal. Shutting down gracefully...")
    sys.exit(0)

def main():
    signal.signal(signal.SIGINT, signal_handler)
    
    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Set memory growth
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    
    # Run pipeline
    pipeline = EnhancedCryptoPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()

