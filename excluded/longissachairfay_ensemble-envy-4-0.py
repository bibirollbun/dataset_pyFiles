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


# Memory-Efficient Crypto Market Prediction Pipeline
import numpy as np
import pandas as pd
import os
import json
import datetime
import math
import gc  # For garbage collection
from pathlib import Path
from typing import Dict, List, Tuple, Optional

# Machine Learning imports
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr, skew, kurtosis, rankdata
from scipy.optimize import minimize

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

# Utilities
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# =========================
# Directory Setup
# =========================
BASE_DIR = Path("/kaggle/working")
MODEL_DIR = BASE_DIR / "models"
PREDICTIONS_DIR = BASE_DIR / "predictions"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
CONFIGS_DIR = BASE_DIR / "configs"
SUBMISSIONS_DIR = BASE_DIR / "submissions"

# Create all directories
for directory in [MODEL_DIR, PREDICTIONS_DIR, CHECKPOINTS_DIR, CONFIGS_DIR, SUBMISSIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {directory}")

# =========================
# Model Configuration
# =========================
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    # Core XGBoost features (before feature engineering)
    CORE_FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612", "bid_qty",
        "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333","X817", 
        "X586",  "X292"
    ]
    
    # Additional feature pools for creating variations
    FEATURE_POOL_1 = [  # ~20 additional features
        "X344", "X137", "X532", "X425", "X132", "X691", "X593", "X377", 
        "X285", "X126", "X419", "X604", "X84", "X138", "X413", "X291", 
        "X40", "X123", "X81", "X853"
    ]
    
    FEATURE_POOL_2 = [  # ~20 more features
        "X854", "X777", "X219", "X776", "X180", "X781", "X445", "X444", 
        "X384", "X466", "X95", "X583", "X272", "X533", "X758", "X279", 
        "X297", "X21", "X20", "X28"
    ]
    
    FEATURE_POOL_3 = [  # ~20 more features
        "X29", "X19", "X27", "X22", "X198", "X89", "X90", "X98", "X96", 
        "X97", "X383", "X427", "X451", "X283", "X753", "X497", "X748", 
        "X820", "X566", "X535"
    ]
    
    FEATURE_POOL_4 = [  # ~20 more features
        "X394", "X618", "X429", "X381", "X387", "X890", "X752", "X375", 
        "X68", "X152", "X110", "X850", "X851", "X481", "X321", "X363", 
        "X405", "X492", "X750", "X751"
    ]
    
    # For memory efficiency, we'll select top features dynamically
    ALL_X_FEATURES = None  # Will be populated as needed
    
    # XGBoost features remain unchanged
    FEATURES = CORE_FEATURES.copy()
    
    # Original features for comparison
    MLP_FEATURES_ORIGINAL = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
    ]
    
    GANDALF_FEATURES_ORIGINAL = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612",
        "X425", "X132", "X691", "X593", "X377", "X285", "X126", "X419", "X604",
        "X84", "X138", "X413", "X291", "X40", "X123", "X81", "X853", "X854",
        "X777", "X219", "X776", "X180", "X781", "X445", "X444", "X384", "X466",
        "X95", "X583", "X272", "X137", "X533", "X758", "X279", "X297",
        "X21", "X20", "X28", "X29", "X19", "X27", "X22", "X198", "X89", "X90",
        "X98", "X96", "X97", "X383", "X427", "X451", "X283",
        "X753", "X497", "X748", "X820", "X566", "X535", "X394", "X618",
        "X429", "X381", "X387", "X890", "X752", "X375", "X68", "X152",
        "X110", "X850", "X851", "X481", "X321", "X363", "X405", "X492",
        "X888", "X421", "X333", "X817", "X586", "X292", "X344", "X532",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
    ]

    LABEL_COLUMN = "label"
    N_FOLDS = 3
    RANDOM_STATE = 42
    OUTLIER_FRACTION = 0.001
    
    # Memory optimization parameters
    MAX_FEATURES_TO_LOAD = 300  # Maximum features to load at once
    FEATURE_SELECTION_SAMPLE_SIZE = 50000  # Sample size for feature selection

def create_feature_variations():
    """Create different feature sets for model variations including large sets"""
    variations = {
        "core": Config.CORE_FEATURES.copy(),
        "core_plus_20": Config.CORE_FEATURES + Config.FEATURE_POOL_1[:20],
        "core_plus_40": Config.CORE_FEATURES + Config.FEATURE_POOL_1 + Config.FEATURE_POOL_2[:20],
        "core_plus_60": Config.CORE_FEATURES + Config.FEATURE_POOL_1 + Config.FEATURE_POOL_2 + Config.FEATURE_POOL_3[:20],
        "core_plus_80": Config.CORE_FEATURES + Config.FEATURE_POOL_1 + Config.FEATURE_POOL_2 + Config.FEATURE_POOL_3 + Config.FEATURE_POOL_4[:20],
        "top_150": None,  # Will be selected based on feature importance
        "top_200": None,  # Will be selected based on feature importance
        "top_300": None,  # Will be selected based on feature importance
        "original": None  # Will use model-specific original features
    }
    
    # Remove duplicates
    for key, features in variations.items():
        if features is not None:
            variations[key] = list(dict.fromkeys(features))  # Preserves order while removing duplicates
    
    return variations

# XGBoost parameters (unchanged)
XGB_PARAMS = {
    "tree_method": "hist",
    "device": "gpu",
    "colsample_bylevel": 0.4778,
    "colsample_bynode": 0.3628,
    "colsample_bytree": 0.7107,
    "gamma": 1.7095,
    "learning_rate": 0.02213,
    "max_depth": 20,
    "max_leaves": 12,
    "min_child_weight": 16,
    "n_estimators": 1667,
    "subsample": 0.06567,
    "reg_alpha": 39.3524,
    "reg_lambda": 75.4484,
    "verbosity": 0,
    "random_state": Config.RANDOM_STATE,
    "n_jobs": -1
}

# =========================
# Memory-Efficient Feature Engineering
# =========================
def add_essential_features(df):
    """Add only the most essential engineered features to save memory"""
    
    # Core features
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['log_volume'] = np.log1p(df['volume'])
    
    # Key ratios
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['order_flow_imbalance'] = df['net_order_flow'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
    # Liquidity measures
    df['liquidity_ratio'] = df['total_depth'] / (df['volume'] + 1e-10)
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['vpin_proxy'] = np.abs(df['net_order_flow']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
    # Market stress
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
    # Price impact
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['amihud_illiquidity'] = np.abs(df['net_order_flow']) / (df['volume'] ** 2 + 1e-10)
    
    # Log transformations
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_total_depth'] = np.log1p(df['total_depth'])
    
    # Squares
    df['order_flow_squared'] = df['net_order_flow'] ** 2
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

def add_advanced_features_selective(df, level='medium'):
    """Add features based on memory constraints"""
    
    if level == 'minimal':
        # Only most essential features
        return add_essential_features(df)
    
    # Start with essential features
    df = add_essential_features(df)
    
    if level == 'medium':
        # Add medium complexity features
        df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
        df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
        df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
        df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
        
        # Interaction features
        df['volume_depth_interaction'] = df['volume'] * df['total_depth']
        df['flow_volume_interaction'] = df['net_order_flow'] * df['volume']
        
        # Additional measures
        df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
        df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
        
    elif level == 'full':
        # This should be used sparingly due to memory constraints
        # Add all the original advanced features
        # [Previous full feature engineering code would go here]
        pass
    
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

# =========================
# Memory-Efficient Feature Selection
# =========================
def select_top_features_memory_efficient(train_path, n_features, sample_size=50000):
    """Select top features using a sample to save memory"""
    
    print(f"Selecting top {n_features} features using sample size {sample_size}...")
    
    # First, get all X feature names
    all_x_features = [f"X{i}" for i in range(1, 891)]
    base_features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
    
    # Load a sample to evaluate features
    sample_df = pd.read_parquet(train_path, 
                               columns=all_x_features + base_features + [Config.LABEL_COLUMN])
    
    # Take a random sample
    if len(sample_df) > sample_size:
        sample_df = sample_df.sample(n=sample_size, random_state=Config.RANDOM_STATE)
    
    # Add engineered features
    sample_df = add_essential_features(sample_df)
    
    # Get all feature columns
    feature_cols = [col for col in sample_df.columns if col != Config.LABEL_COLUMN]
    X = sample_df[feature_cols]
    y = sample_df[Config.LABEL_COLUMN]
    
    # Use mutual information for feature selection
    selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, len(feature_cols)))
    selector.fit(X, y)
    
    # Get selected features
    feature_scores = pd.DataFrame({
        'feature': feature_cols,
        'score': selector.scores_
    }).sort_values('score', ascending=False)
    
    selected_features = feature_scores['feature'].head(n_features).tolist()
    
    # Clean up memory
    del sample_df, X, y
    gc.collect()
    
    return selected_features

# =========================
# Utility Functions
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_time_decay_weights(n: int, decay: float = 0.9) -> np.ndarray:
    positions = np.arange(n)
    normalized = positions / (n - 1)
    weights = decay ** (1.0 - normalized)
    return weights * n / weights.sum()

def detect_outliers_and_adjust_weights(X, y, sample_weights, outlier_fraction=0.001):
    """Detect outliers and adjust weights"""
    rf = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y, sample_weight=sample_weights)
    
    predictions = rf.predict(X)
    residuals = np.abs(y - predictions)
    
    n_outliers = max(1, int(len(residuals) * outlier_fraction))
    threshold = np.sort(residuals)[-n_outliers]
    
    outlier_mask = residuals >= threshold
    adjusted_weights = sample_weights.copy()
    
    if outlier_mask.any():
        outlier_residuals = residuals[outlier_mask]
        min_outlier_res = outlier_residuals.min()
        max_outlier_res = outlier_residuals.max()
        
        if max_outlier_res > min_outlier_res:
            normalized_residuals = (outlier_residuals - min_outlier_res) / (max_outlier_res - min_outlier_res)
        else:
            normalized_residuals = np.ones_like(outlier_residuals)
        
        weight_factors = 0.8 - 0.6 * normalized_residuals
        adjusted_weights[outlier_mask] *= weight_factors
        
        print(f"    Adjusted weights for {n_outliers} outliers ({outlier_fraction*100:.1f}% of data)")
    
    return adjusted_weights

def load_data_memory_efficient(feature_list=None):
    """Load data with only necessary features to save memory"""
    
    if feature_list is None:
        # Load only the predefined features
        feature_list = list(set(
            Config.CORE_FEATURES + 
            Config.FEATURE_POOL_1 + 
            Config.FEATURE_POOL_2 + 
            Config.FEATURE_POOL_3 + 
            Config.FEATURE_POOL_4
        ))
    
    # Always include base features
    base_features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
    feature_list = list(set(feature_list + base_features))
    
    print(f"Loading {len(feature_list)} features...")
    
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=feature_list + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=feature_list)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}")
    
    # Add engineered features
    train_df = add_essential_features(train_df)
    test_df = add_essential_features(test_df)
    
    # Update Config.FEATURES with engineered features
    engineered_features = [
        col for col in train_df.columns 
        if col not in feature_list and col != Config.LABEL_COLUMN
    ]
    
    Config.FEATURES = feature_list + engineered_features
    print(f"Total features after engineering: {len(Config.FEATURES)}")
    
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

def get_model_slices(n_samples: int):
    """Get model training slices"""
    base_slices = [
        {"name": "full_data", "cutoff": 0, "is_oldest": False, "outlier_adjusted": False},
        {"name": "last_90pct", "cutoff": int(0.10 * n_samples), "is_oldest": False, "outlier_adjusted": False},
        {"name": "last_85pct", "cutoff": int(0.15 * n_samples), "is_oldest": False, "outlier_adjusted": False},
        {"name": "last_80pct", "cutoff": int(0.20 * n_samples), "is_oldest": False, "outlier_adjusted": False},
        {"name": "oldest_25pct", "cutoff": int(0.25 * n_samples), "is_oldest": True, "outlier_adjusted": False},
    ]
    
    outlier_adjusted_slices = []
    for slice_info in base_slices:
        adjusted_slice = slice_info.copy()
        adjusted_slice["name"] = f"{slice_info['name']}_outlier_adj"
        adjusted_slice["outlier_adjusted"] = True
        outlier_adjusted_slices.append(adjusted_slice)
    
    return base_slices + outlier_adjusted_slices

def save_predictions(model_name: str, predictions: np.ndarray, metadata: dict = None):
    """Save model predictions with metadata"""
    pred_path = PREDICTIONS_DIR / f"{model_name}_predictions.npy"
    np.save(pred_path, predictions)
    
    # Save metadata
    meta = {
        "model_name": model_name,
        "shape": list(predictions.shape),
        "stats": {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions)),
            "skew": float(skew(predictions)),
            "kurtosis": float(kurtosis(predictions))
        },
        "saved_at": datetime.datetime.now().isoformat()
    }
    
    if metadata:
        # Convert all numpy types to Python types in metadata
        for key, value in metadata.items():
            if isinstance(value, np.ndarray):
                metadata[key] = value.tolist()
            elif isinstance(value, (np.float32, np.float64)):
                metadata[key] = float(value)
            elif isinstance(value, (np.int32, np.int64)):
                metadata[key] = int(value)
        meta.update(metadata)
        
    meta_path = pred_path.with_suffix('.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=4)
        
    print(f"ðŸ’¾ Saved {model_name} predictions to {pred_path}")
    print(f"   Mean: {meta['stats']['mean']:.6f}, Std: {meta['stats']['std']:.6f}")

# =========================
# Deep Learning Components
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_activation_function(name):
    """Return the activation function based on the name."""
    if name == None:
        return None
    name = name.lower()
    if name == 'relu':
        return nn.ReLU()
    elif name == 'tanh':
        return nn.Tanh()
    elif name == 'sigmoid':
        return nn.Sigmoid()
    elif name == 'gelu':
        return nn.GELU()
    elif name == 'selu':
        return nn.SELU()
    elif name == 'elu':
        return nn.ELU()
    else:
        raise ValueError(f"Unsupported activation function: {name}")

def get_dataloaders(X, Y, hparams, device, shuffle=True):
    """Create DataLoader for training and validation datasets."""
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    if Y is not None:
        Y_tensor = torch.tensor(Y.values if hasattr(Y, 'values') else Y, 
                                dtype=torch.float32, device=device).unsqueeze(1)
        dataset = TensorDataset(X_tensor, Y_tensor)
    else:
        dataset = TensorDataset(X_tensor)
    
    dataloader = DataLoader(dataset, batch_size=hparams["batch_size"], shuffle=shuffle, 
                            generator=torch.Generator().manual_seed(hparams["seed"]))
    return dataloader

# Enhanced MLP with multiple regularization strategies
class RegularizedMLP(nn.Module):
    def __init__(self, dropout_rate=0.6, dropout_strategy='standard',
                 layers=[128, 64], activation='relu', last_activation=None,
                 use_batch_norm=True, use_layer_norm=False, 
                 use_weight_norm=False, use_spectral_norm=False):
        super(RegularizedMLP, self).__init__()
        
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.activation = get_activation_function(activation)
        self.last_activation = get_activation_function(last_activation)
        self.dropout_strategy = dropout_strategy
        self.use_batch_norm = use_batch_norm
        self.use_layer_norm = use_layer_norm

        for i in range(len(layers) - 1):
            linear = nn.Linear(layers[i], layers[i + 1])
            
            # Apply weight normalization strategies
            if use_weight_norm:
                linear = nn.utils.weight_norm(linear)
            elif use_spectral_norm:
                linear = nn.utils.spectral_norm(linear)
                
            self.linears.append(linear)
            
            # Add normalization layers
            if i < len(layers) - 2:  # Not on the last layer
                if use_batch_norm:
                    self.norms.append(nn.BatchNorm1d(layers[i + 1]))
                elif use_layer_norm:
                    self.norms.append(nn.LayerNorm(layers[i + 1]))
                else:
                    self.norms.append(None)

        # Different dropout strategies
        if dropout_strategy == 'standard':
            self.dropout = nn.Dropout(dropout_rate)
        elif dropout_strategy == 'alpha':
            self.dropout = nn.AlphaDropout(dropout_rate)
        elif dropout_strategy == 'gaussian':
            self.dropout = lambda x: x + torch.randn_like(x) * dropout_rate
        else:
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for k in range(len(self.linears) - 1):
            x = self.linears[k](x)
            if k < len(self.norms) and self.norms[k] is not None:
                x = self.norms[k](x)
            x = self.activation(x)
            x = self.dropout(x)
            
        x = self.linears[-1](x)
        if self.last_activation is not None:
            x = self.last_activation(x)
        return x

# =========================
# XGBoost Training
# =========================
def train_xgboost(train_df, test_df):
    """Train XGBoost with multiple slices"""
    print("\n=== Training XGBoost Model ===")
    
    n_samples = len(train_df)
    model_slices = get_model_slices(n_samples)

    oof_preds = {s["name"]: np.zeros(n_samples) for s in model_slices}
    test_preds = {s["name"]: np.zeros(len(test_df)) for s in model_slices}

    full_weights = create_time_decay_weights(n_samples)
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)

    for fold, (train_idx, valid_idx) in enumerate(kf.split(train_df), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        X_valid = train_df.iloc[valid_idx][Config.FEATURES]
        y_valid = train_df.iloc[valid_idx][Config.LABEL_COLUMN]

        for s in model_slices:
            cutoff = s["cutoff"]
            slice_name = s["name"]
            is_oldest = s["is_oldest"]
            outlier_adjusted = s["outlier_adjusted"]
            
            if is_oldest:
                subset = train_df.iloc[:cutoff].reset_index(drop=True)
                rel_idx = train_idx[train_idx < cutoff]
                sw = np.ones(len(rel_idx))
            else:
                subset = train_df.iloc[cutoff:].reset_index(drop=True)
                rel_idx = train_idx[train_idx >= cutoff] - cutoff
                sw = create_time_decay_weights(len(subset))[rel_idx] if cutoff > 0 else full_weights[train_idx]

            X_train = subset.iloc[rel_idx][Config.FEATURES]
            y_train = subset.iloc[rel_idx][Config.LABEL_COLUMN]
            
            if outlier_adjusted and len(X_train) > 100:
                sw = detect_outliers_and_adjust_weights(
                    X_train.values, 
                    y_train.values, 
                    sw, 
                    outlier_fraction=Config.OUTLIER_FRACTION
                )

            print(f"  Training slice: {slice_name}, samples: {len(X_train)}")

            # Use fewer trees for memory efficiency if needed
            xgb_params = XGB_PARAMS.copy()
            if len(Config.FEATURES) > 200:
                xgb_params['n_estimators'] = 1000  # Reduce from 1667
                
            model = XGBRegressor(**xgb_params)
            model.fit(X_train, y_train, sample_weight=sw, eval_set=[(X_valid, y_valid)], verbose=False)

            if is_oldest:
                oof_preds[slice_name][valid_idx] = model.predict(
                    train_df.iloc[valid_idx][Config.FEATURES]
                )
            else:
                mask = valid_idx >= cutoff
                if mask.any():
                    idxs = valid_idx[mask]
                    oof_preds[slice_name][idxs] = model.predict(
                        train_df.iloc[idxs][Config.FEATURES]
                    )
                if cutoff > 0 and (~mask).any():
                    base_slice_name = slice_name.replace("_outlier_adj", "")
                    if base_slice_name == slice_name:
                        fallback_slice = "full_data"
                    else:
                        fallback_slice = "full_data_outlier_adj"
                    oof_preds[slice_name][valid_idx[~mask]] = oof_preds[fallback_slice][valid_idx[~mask]]

            test_preds[slice_name] += model.predict(test_df[Config.FEATURES])

    # Normalize test predictions
    for slice_name in test_preds:
        test_preds[slice_name] /= (Config.N_FOLDS - 1)

    # Create XGBoost ensemble
    weights = np.array([
        1.0,   # full_data
        1.0,   # last_90pct
        1.0,   # last_85pct
        1.0,   # last_80pct
        0.25,  # oldest_25pct
        0.9,   # full_data_outlier_adj
        0.9,   # last_90pct_outlier_adj
        0.9,   # last_85pct_outlier_adj
        0.9,   # last_80pct_outlier_adj
        0.2    # oldest_25pct_outlier_adj
    ])
    
    weights = weights / weights.sum()

    oof_weighted = pd.DataFrame(oof_preds).values @ weights
    test_weighted = pd.DataFrame(test_preds).values @ weights
    score_weighted = pearsonr(train_df[Config.LABEL_COLUMN], oof_weighted)[0]
    print(f"\nXGBoost Weighted Ensemble Pearson: {score_weighted:.4f}")

    # Print individual slice scores
    print("\nIndividual slice OOF scores and weights:")
    slice_names = list(oof_preds.keys())
    slice_scores = {}
    for i, slice_name in enumerate(slice_names):
        score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds[slice_name])[0]
        slice_scores[slice_name] = score
        print(f"  {slice_name}: {score:.4f} (weight: {weights[i]:.3f})")

    # Save predictions and metadata
    save_predictions("xgboost", test_weighted, {
        "slice_scores": slice_scores,
        "ensemble_weights": weights.tolist(),
        "oof_score": score_weighted
    })
    
    return test_weighted

# =========================
# MLP Training
# =========================
def train_mlp_single(train_df, test_df, features, model_suffix, hparams, regularization_config):
    """Train a single MLP model with given features and regularization"""
    print(f"\n--- Training MLP {model_suffix} with {len(features)} features ---")
    
    set_seed(hparams["seed"])
    
    # Filter features to only those that exist
    features = [f for f in features if f in train_df.columns]
    
    X_train_full = train_df[features].values
    y_train_full = train_df[Config.LABEL_COLUMN].values
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.2, shuffle=False, random_state=42
    )
    
    # Try different scalers
    if regularization_config.get('scaler_type') == 'quantile':
        scaler = QuantileTransformer(output_distribution='normal')
    elif regularization_config.get('scaler_type') == 'robust':
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()
        
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(test_df[features].values)
    
    # Update layers for different feature sizes
    hidden_dims = regularization_config.get('hidden_dims', [256, 128, 64])
    layers = [len(features)] + hidden_dims + [1]
    
    # Create dataloaders
    train_loader = get_dataloaders(X_train, y_train, hparams, device, shuffle=True)
    val_loader = get_dataloaders(X_val, y_val, hparams, device, shuffle=False)
    test_loader = get_dataloaders(X_test, None, hparams, device, shuffle=False)
    
    model = RegularizedMLP(
        layers=layers,
        dropout_rate=regularization_config.get('dropout_rate', 0.5),
        dropout_strategy=regularization_config.get('dropout_strategy', 'standard'),
        activation=hparams["activation"],
        last_activation=hparams["hidden_activation"],
        use_batch_norm=regularization_config.get('use_batch_norm', True),
        use_layer_norm=regularization_config.get('use_layer_norm', False),
        use_weight_norm=regularization_config.get('use_weight_norm', False),
        use_spectral_norm=regularization_config.get('use_spectral_norm', False),
    ).to(device)
    
    criterion = nn.HuberLoss(delta=hparams["delta"], reduction='mean')
    
    # Different optimizers
    if regularization_config.get('optimizer') == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=hparams["learning_rate"], 
                             momentum=0.9, weight_decay=hparams["weight_decay"])
    elif regularization_config.get('optimizer') == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=hparams["learning_rate"], 
                               weight_decay=hparams["weight_decay"])
    else:
        optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"], 
                              weight_decay=hparams["weight_decay"])
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    checkpoint_path = CHECKPOINTS_DIR / f"best_mlp_{model_suffix}_model.pt"
    best_pearson = -np.inf
    
    # Training loop
    num_epochs = hparams["num_epochs"]
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Dynamic noise injection
            noise_factor = hparams["noise_factor"] * (1 - epoch / num_epochs)
            inputs = inputs + torch.randn_like(inputs) * noise_factor
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Add L1 regularization if specified
            if regularization_config.get('l1_lambda', 0) > 0:
                l1_loss = sum(p.abs().sum() for p in model.parameters())
                loss = loss + regularization_config['l1_lambda'] * l1_loss
            
            loss.backward()
            
            # Gradient clipping
            if regularization_config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), regularization_config['grad_clip'])
            
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            
        running_loss = running_loss / len(train_loader.dataset)
        print(f"Training Loss: {running_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        preds = []
        trues = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                preds.append(outputs.cpu().numpy())
                trues.append(targets.cpu().numpy())

        val_loss /= len(val_loader.dataset)
        preds = np.concatenate(preds).flatten()
        trues = np.concatenate(trues).flatten()
        pearson_coef = pearsonr(preds, trues)[0]
        print(f"Validation Pearson Coef: {pearson_coef:.4f} | Loss: {val_loss:.4f}")
        
        scheduler.step(pearson_coef)

        if pearson_coef > best_pearson:
            best_pearson = pearson_coef
            torch.save(model.state_dict(), checkpoint_path)
            print(f"âœ… New best model saved with Pearson: {best_pearson:.4f}")
    
    # Load best model and make predictions
    model.load_state_dict(torch.load(checkpoint_path))
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs in test_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            predictions.append(outputs.cpu().numpy())

    predictions = np.concatenate(predictions).flatten()
    
    # Save predictions
    save_predictions(f"mlp_{model_suffix}", predictions, {
        "best_val_pearson": best_pearson,
        "n_epochs_trained": num_epochs,
        "n_features": len(features),
        "regularization": regularization_config
    })
    
    return predictions

def train_mlp(train_df, test_df):
    """Train MLP models with multiple feature sets and regularization strategies"""
    print("\n=== Training MLP Models with Multiple Feature Sets and Regularization ===")
    
    # Base hyperparameters
    base_hparams = {
        "seed": 42,
        "num_epochs": 10,
        "batch_size": 2048,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "dropout_rate": 0.5,
        "hidden_activation": None,
        "activation": "relu",
        "delta": 5,
        "noise_factor": 0.005
    }
    
    # Different regularization configurations
    regularization_configs = {
        "standard": {
            "dropout_rate": 0.5,
            "dropout_strategy": "standard",
            "use_batch_norm": True,
            "hidden_dims": [256, 128, 64],
            "optimizer": "adam",
            "scaler_type": "standard",
            "grad_clip": 1.0
        },
        "heavy_reg": {
            "dropout_rate": 0.7,
            "dropout_strategy": "standard",
            "use_batch_norm": True,
            "hidden_dims": [128, 64, 32],
            "optimizer": "adamw",
            "scaler_type": "robust",
            "l1_lambda": 1e-5,
            "grad_clip": 0.5
        }
    }
    
    feature_variations = create_feature_variations()
    
    # For large feature sets, we need to be more careful
    if feature_variations["top_150"] is None:
        # Select top features using the memory-efficient method
        feature_variations["top_150"] = select_top_features_memory_efficient(
            Config.TRAIN_PATH, 150, Config.FEATURE_SELECTION_SAMPLE_SIZE
        )
        feature_variations["top_200"] = select_top_features_memory_efficient(
            Config.TRAIN_PATH, 200, Config.FEATURE_SELECTION_SAMPLE_SIZE
        )
        feature_variations["top_300"] = select_top_features_memory_efficient(
            Config.TRAIN_PATH, 300, Config.FEATURE_SELECTION_SAMPLE_SIZE
        )
    
    all_predictions = {}
    
    # Train models with different feature sets
    for variant_name, features in feature_variations.items():
        if variant_name == "original":
            features = Config.MLP_FEATURES_ORIGINAL
        
        # Skip if features not available in current dataframe
        if not all(f in train_df.columns for f in features):
            print(f"Skipping {variant_name} - features not available in current data")
            continue
        
        # For larger feature sets, use heavier regularization
        if len(features) > 150:
            reg_config = regularization_configs["heavy_reg"]
        else:
            reg_config = regularization_configs["standard"]
            
        predictions = train_mlp_single(
            train_df, test_df, features, 
            f"{variant_name}_reg", 
            base_hparams, reg_config
        )
        all_predictions[variant_name] = predictions
    
    return all_predictions

# =========================
# Advanced Ensemble Creation
# =========================
def create_advanced_ensembles(submission_df, train_df):
    """Create multiple ensemble submissions with different strategies"""
    print("\n=== Creating Advanced Ensemble Submissions ===")
    
    # Collect available model predictions
    predictions = {}
    model_scores = {}
    
    pred_files = list(PREDICTIONS_DIR.glob("*_predictions.npy"))
    
    for pred_file in pred_files:
        model_name = pred_file.stem.replace("_predictions", "")
        predictions[model_name] = np.load(pred_file)
        
        # Load metadata
        meta_file = pred_file.with_suffix('.json')
        if meta_file.exists():
            with open(meta_file, 'r') as f:
                meta = json.load(f)
                if 'best_val_pearson' in meta:
                    model_scores[model_name] = meta['best_val_pearson']
                elif 'oof_score' in meta:
                    model_scores[model_name] = meta['oof_score']
                else:
                    model_scores[model_name] = 0.05
        
        print(f"âœ“ Loaded {model_name} predictions (score: {model_scores.get(model_name, 'N/A'):.4f})")
    
    # Create different ensemble strategies
    ensemble_configs = {
        "xgb_dominant_80": {
            "description": "XGBoost 80% weight, others split equally",
            "weights": lambda models: {
                m: 0.8 if m == "xgboost" else 0.2 / (len(models) - 1)
                for m in models
            }
        },
        
        "xgb_dominant_85": {
            "description": "XGBoost 85% weight, others split equally",
            "weights": lambda models: {
                m: 0.85 if m == "xgboost" else 0.15 / (len(models) - 1)
                for m in models
            }
        },
        
        "xgb_dominant_90": {
            "description": "XGBoost 90% weight, others split equally",
            "weights": lambda models: {
                m: 0.9 if m == "xgboost" else 0.1 / (len(models) - 1)
                for m in models
            }
        },
        
        "performance_weighted": {
            "description": "Weights based on validation performance",
            "weights": lambda models: {
                m: max(0.01, model_scores.get(m, 0.05))
                for m in models
            }
        },
        
        "top_models_only": {
            "description": "Only use models with score > 0.08",
            "weights": lambda models: {
                m: 1.0 if model_scores.get(m, 0) > 0.08 else 0.0
                for m in models
            }
        },
        
        "stable_models": {
            "description": "Favor models with lower std deviation",
            "weights": lambda models: {
                m: 1.0 / (np.std(predictions[m]) + 0.1)
                for m in models
            }
        }
    }
    
    submissions = {}
    
    for config_name, config in ensemble_configs.items():
        print(f"\nðŸ“Š Creating ensemble: {config['description']}")
        
        # Get weights
        weights = config['weights'](list(predictions.keys()))
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items() if v > 0}
        else:
            # Fallback to equal weights
            weights = {k: 1.0/len(predictions) for k in predictions.keys()}
        
        # Create ensemble
        ensemble_pred = np.zeros_like(list(predictions.values())[0])
        
        for model_name, weight in weights.items():
            if weight > 0 and model_name in predictions:
                ensemble_pred += weight * predictions[model_name]
        
        # Post-processing
        if config_name.startswith("xgb_dominant"):
            # Minimal post-processing for XGB-heavy ensembles
            ensemble_pred = np.clip(ensemble_pred, 
                                   np.percentile(ensemble_pred, 0.1),
                                   np.percentile(ensemble_pred, 99.9))
        else:
            # Light post-processing for other ensembles
            mean_pred = ensemble_pred.mean()
            std_pred = ensemble_pred.std()
            ensemble_pred = np.clip(ensemble_pred, mean_pred - 4*std_pred, mean_pred + 4*std_pred)
        
        # Create submission
        submission = submission_df.copy()
        submission["prediction"] = ensemble_pred
        
        # Save submission
        submission_path = SUBMISSIONS_DIR / f"ensemble_{config_name}.csv"
        submission.to_csv(submission_path, index=False)
        submissions[config_name] = submission_path
        
        # Print statistics
        print(f"   Mean: {ensemble_pred.mean():.6f}, Std: {ensemble_pred.std():.6f}")
        print(f"   Min: {ensemble_pred.min():.6f}, Max: {ensemble_pred.max():.6f}")
        
        # Show top weighted models
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:5]
        print("   Top weights:")
        for model_name, weight in sorted_weights:
            print(f"     {model_name}: {weight:.1%}")
    
    return submissions

# =========================
# Main Execution
# =========================
def main():
    """Main execution function"""
    
    # Load data with memory-efficient approach
    print("\n=== Loading Data (Memory Efficient) ===")
    train_df, test_df, submission_df = load_data_memory_efficient()
    
    # Train XGBoost
    xgb_predictions = train_xgboost(train_df, test_df)
    
    # Clean up memory before training neural networks
    gc.collect()
    
    # Train MLP models
    mlp_predictions = train_mlp(train_df, test_df)
    
    # Create multiple ensemble submissions
    ensemble_paths = create_advanced_ensembles(submission_df, train_df)
    
    # Print summary
    print("\n" + "="*60)
    print("EXECUTION SUMMARY")
    print("="*60)
    
    print("\nâœ… Pipeline execution completed successfully!")
    print(f"ðŸ“Š Models trained:")
    print(f"   - XGBoost: 1 model")
    print(f"   - MLP: {len(mlp_predictions)} variations")
    print(f"\nðŸ“ˆ Ensemble submissions created:")
    for name, path in ensemble_paths.items():
        print(f"   - {name}: {path}")

if __name__ == "__main__":
    main()

