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


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

import numpy as np
import pandas as pd
import os
import json
import datetime
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from itertools import combinations
from scipy.optimize import minimize

# Machine Learning imports
from sklearn.model_selection import KFold, train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from xgboost import XGBRegressor
from scipy.stats import pearsonr, spearmanr

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau

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
ENSEMBLE_DIR = BASE_DIR / "ensembles"

# Create all directories
for directory in [MODEL_DIR, PREDICTIONS_DIR, CHECKPOINTS_DIR, CONFIGS_DIR, SUBMISSIONS_DIR, ENSEMBLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)
    print(f"Created directory: {directory}")

# =========================
# Model Configuration
# =========================
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

    FEATURES = [
        "X863", "X856", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X855", "X174", "X302", "X178", "X168", "X612", "bid_qty",
        "ask_qty", "buy_qty", "sell_qty", "volume", "X888", "X421", "X333","X817", 
        "X586",  "X292"
    ]
    
    MLP_FEATURES = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"
    ]
    
    GANDALF_FEATURES = [
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
    
    # All available models
    ALL_MODELS = ["xgboost", "mlp", "gandalf", "simplified_gandalf", "anam", "dcnv2", "tangos", "dofen"]

# XGBoost parameters
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
# Feature Engineering
# =========================
def add_features(df):
    """Add all engineered features"""
    # Original features
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']

    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['log_volume'] = np.log1p(df['volume'])

    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    
    # Price Pressure Indicators
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    
    # Liquidity Depth Measures
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])
    
    # Order Flow Toxicity Proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
    # Market Activity Indicators
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # Microstructure Volatility Proxies
    df['realized_spread_proxy'] = 2 * np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])
    
    # Complex Interaction Terms
    df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['depth_volume_interaction'] = df['total_depth'] * df['volume']
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    
    # Information Asymmetry Measures
    df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + 1e-10)
    df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10) * df['volume']
    
    # Market Efficiency Indicators
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + 1e-10)
    
    # Non-linear Transformations
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['sqrt_depth'] = np.sqrt(df['total_depth'])
    df['volume_squared'] = df['volume'] ** 2
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    
    # Relative Measures
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
    # Market Stress Indicators
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    
    # Directional Indicators
    df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
    df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume']
    
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

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
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
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

def load_data():
    """Load and prepare data"""
    all_features = list(set(Config.FEATURES + Config.MLP_FEATURES + Config.GANDALF_FEATURES))
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=all_features + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=all_features)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    # Update Config.FEATURES with new features
    engineered_features = [
        "log_volume", 'bid_ask_interaction', 'bid_buy_interaction', 'bid_sell_interaction', 
        'ask_buy_interaction', 'ask_sell_interaction', 'net_order_flow', 'normalized_net_flow',
        'buying_pressure', 'volume_weighted_buy', 'total_depth', 'depth_imbalance',
        'relative_spread', 'log_depth', 'kyle_lambda', 'flow_toxicity', 'aggressive_flow_ratio',
        'volume_depth_ratio', 'activity_intensity', 'log_buy_qty', 'log_sell_qty',
        'log_bid_qty', 'log_ask_qty', 'realized_spread_proxy', 'price_impact_proxy',
        'quote_volatility_proxy', 'flow_depth_interaction', 'imbalance_volume_interaction',
        'depth_volume_interaction', 'buy_sell_spread', 'bid_ask_spread', 'trade_informativeness',
        'execution_shortfall_proxy', 'adverse_selection_proxy', 'fill_probability',
        'execution_rate', 'market_efficiency', 'sqrt_volume', 'sqrt_depth', 'volume_squared',
        'imbalance_squared', 'bid_ratio', 'ask_ratio', 'buy_ratio', 'sell_ratio',
        'liquidity_consumption', 'market_stress', 'depth_depletion', 'net_buying_ratio',
        'directional_volume', 'signed_volume'
    ]
    
    Config.FEATURES += engineered_features

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
    
    # Save metadata - convert numpy types to Python types for JSON serialization
    meta = {
        "model_name": model_name,
        "shape": list(predictions.shape),
        "stats": {
            "mean": float(np.mean(predictions)),
            "std": float(np.std(predictions)),
            "min": float(np.min(predictions)),
            "max": float(np.max(predictions))
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
        
    print(f"ğŸ’¾ Saved {model_name} predictions to {pred_path}")
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

class Checkpointer:
    def __init__(self, path="best_model.pt"):
        self.path = CHECKPOINTS_DIR / path
        self.best_pearson = -np.inf

    def load(self, model):
        """Load the best model weights."""
        model.load_state_dict(torch.load(self.path))
        print(f"Model loaded from {self.path} with best Pearson: {self.best_pearson:.4f}")
        return model

    def __call__(self, pearson_coef, model):
        """Save the model if the Pearson coefficient is better than the best one."""
        if pearson_coef > self.best_pearson:
            self.best_pearson = pearson_coef
            torch.save(model.state_dict(), self.path)
            print(f"âœ… New best model saved to {self.path} with Pearson: {pearson_coef:.4f}")

class MLP(nn.Module):
    def __init__(self, dropout_rate=0.6, 
                 layers=[128, 64], activation='relu', last_activation=None):
        super(MLP, self).__init__()
        
        self.linears = nn.ModuleList()
        self.activation = get_activation_function(activation)
        self.last_activation = get_activation_function(last_activation)

        for i in range(len(layers) - 1):
            self.linears.append(nn.Linear(layers[i], layers[i + 1]))

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        for k in range(len(self.linears) - 1):
            x = self.activation(self.linears[k](x))
            x = self.dropout(x)
        x = self.linears[-1](x)
        if self.last_activation is not None:
            x = self.last_activation(x)
        return x

# =========================
# DCN V2 Components
# =========================
class CrossLayerV2(nn.Module):
    """Cross Layer for DCN V2 with low-rank approximation"""
    def __init__(self, input_dim, num_experts=4, low_rank=32):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.low_rank = low_rank
        
        # Expert weights with low-rank decomposition
        self.U = nn.Parameter(torch.randn(num_experts, input_dim, low_rank))
        self.V = nn.Parameter(torch.randn(num_experts, low_rank, input_dim))
        self.C = nn.Parameter(torch.randn(num_experts, low_rank, low_rank))
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Bias
        self.bias = nn.Parameter(torch.zeros(input_dim))
        
        # Initialize
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.C)
        
    def forward(self, x0, x):
        """
        x0: input to the cross layer (batch_size, input_dim)
        x: current layer input (batch_size, input_dim)
        """
        batch_size = x.shape[0]
        
        # Compute gating weights
        gates = self.gate(x)  # (batch_size, num_experts)
        
        # Compute expert outputs
        expert_outputs = []
        for i in range(self.num_experts):
            # W = U * C * V
            W_i = torch.matmul(torch.matmul(self.U[i], self.C[i]), self.V[i])
            # x0 * (W * x + b)
            output_i = x0 * (torch.matmul(x, W_i.t()) + self.bias)
            expert_outputs.append(output_i)
        
        # Stack expert outputs
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (batch_size, num_experts, input_dim)
        
        # Apply gating
        gates = gates.unsqueeze(-1)  # (batch_size, num_experts, 1)
        output = torch.sum(expert_outputs * gates, dim=1)  # (batch_size, input_dim)
        
        return output + x  # Residual connection

class DCNV2(nn.Module):
    """Deep & Cross Network V2"""
    def __init__(self, input_dim, num_cross_layers=3, num_experts=4, 
                 deep_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_cross_layers = num_cross_layers
        
        # Cross network
        self.cross_layers = nn.ModuleList()
        for _ in range(num_cross_layers):
            self.cross_layers.append(
                CrossLayerV2(input_dim, num_experts=num_experts, low_rank=32)
            )
        
        # Deep network
        deep_layers = []
        prev_dim = input_dim
        
        for dim in deep_dims:
            deep_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
            
        self.deep_network = nn.Sequential(*deep_layers)
        self.deep_output_dim = prev_dim
        
        # Combination layer
        self.combination = nn.Sequential(
            nn.Linear(input_dim + self.deep_output_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
    def forward(self, x):
        # Cross network
        x0 = x
        cross_output = x
        for cross_layer in self.cross_layers:
            cross_output = cross_layer(x0, cross_output)
        
        # Deep network
        deep_output = self.deep_network(x)
        
        # Combine cross and deep
        combined = torch.cat([cross_output, deep_output], dim=1)
        output = self.combination(combined)
        
        return output

class DifferentiableDecisionTree(nn.Module):
    """Soft decision tree for GANDALF with improved numerical stability"""
    def __init__(self, input_dim, depth, temperature=1.0):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        
        # Internal nodes with better initialization
        self.internal_nodes = nn.ModuleList()
        for i in range(2 ** depth - 1):
            linear = nn.Linear(input_dim, 1)
            # Xavier initialization scaled down
            nn.init.xavier_uniform_(linear.weight, gain=0.5)
            nn.init.zeros_(linear.bias)
            self.internal_nodes.append(linear)
        
        # Leaf values with small initialization
        self.leaf_values = nn.Parameter(torch.zeros(self.n_leaves))
        nn.init.normal_(self.leaf_values, mean=0, std=0.01)
        
        # Temperature as learnable parameter with constraints
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        # Constrain temperature to reasonable range
        temp = torch.clamp(torch.exp(self.log_temperature), min=0.1, max=10.0)
        
        # Initialize path probabilities
        path_probs = torch.ones(batch_size, 1, device=device)
        
        # Traverse the tree
        for level in range(self.depth):
            n_nodes = 2 ** level
            next_path_probs = []
            
            for node in range(n_nodes):
                node_idx = 2 ** level - 1 + node
                
                if node_idx < len(self.internal_nodes) and node < path_probs.size(1):
                    # Get decision logit
                    logit = self.internal_nodes[node_idx](x).squeeze(-1)
                    
                    # Stable sigmoid with temperature
                    logit_scaled = logit / temp
                    # Clamp to prevent overflow
                    logit_scaled = torch.clamp(logit_scaled, min=-10, max=10)
                    split_prob = torch.sigmoid(logit_scaled)
                    
                    current_prob = path_probs[:, node:node+1]
                    
                    # Left and right probabilities
                    left_prob = current_prob * (1 - split_prob).unsqueeze(1)
                    right_prob = current_prob * split_prob.unsqueeze(1)
                    
                    next_path_probs.append(left_prob)
                    next_path_probs.append(right_prob)
            
            if next_path_probs:
                path_probs = torch.cat(next_path_probs, dim=1)
        
        # Ensure we have the right number of leaf probabilities
        if path_probs.size(1) != self.n_leaves:
            # Pad or truncate as needed
            if path_probs.size(1) < self.n_leaves:
                padding = torch.zeros(batch_size, self.n_leaves - path_probs.size(1), device=device)
                path_probs = torch.cat([path_probs, padding], dim=1)
            else:
                path_probs = path_probs[:, :self.n_leaves]
        
        # Normalize path probabilities for numerical stability
        path_probs = path_probs + 1e-10
        path_probs = path_probs / path_probs.sum(dim=1, keepdim=True)
        
        # Compute output as weighted sum of leaf values
        output = torch.sum(path_probs * self.leaf_values.unsqueeze(0), dim=1)
        
        return output

class GatingNetwork(nn.Module):
    """Gating network for tree selection with improved stability"""
    def __init__(self, input_dim, n_trees, hidden_dim=128, dropout=0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_trees)
        )
        
        # Initialize last layer with smaller weights
        nn.init.xavier_uniform_(self.network[-1].weight, gain=0.1)
        
    def forward(self, x):
        gates = self.network(x)
        # Temperature softmax for more stable gradients
        gates = F.softmax(gates / 2.0, dim=-1)
        return gates

class GANDALF(nn.Module):
    """GANDALF: Gated Additive Neural Decision Additive Forest with improved stability"""
    def __init__(self, config):
        super().__init__()
        
        self.n_trees = config['n_trees']
        self.tree_depth = config['tree_depth']
        self.input_dim = config['input_dim']
        
        # Feature embedding with batch normalization
        embed_layers = []
        prev_dim = self.input_dim
        
        # First layer with batch norm
        embed_layers.extend([
            nn.Linear(prev_dim, config['embed_dims'][0]),
            nn.BatchNorm1d(config['embed_dims'][0]),
            nn.GELU(),
            nn.Dropout(config['feature_dropout'])
        ])
        prev_dim = config['embed_dims'][0]
        
        # Remaining layers
        for dim in config['embed_dims'][1:]:
            embed_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.GELU(),
                nn.Dropout(config['feature_dropout'])
            ])
            prev_dim = dim
            
        self.feature_embedder = nn.Sequential(*embed_layers)
        self.embed_dim = prev_dim
        
        # Decision trees with varying depths
        self.trees = nn.ModuleList()
        for i in range(self.n_trees):
            # Vary tree depth slightly
            tree_depth = self.tree_depth + (i % 3 - 1)
            tree_depth = max(2, min(tree_depth, 6))  # Constrain depth
            
            self.trees.append(
                DifferentiableDecisionTree(
                    self.embed_dim,
                    tree_depth,
                    temperature=config['tree_temperature']
                )
            )
        
        # Gating network
        self.gating_network = GatingNetwork(
            self.input_dim,
            self.n_trees,
            config['gate_hidden_dim'],
            config['gate_dropout']
        )
        
        # Optional neural head
        if config.get('use_nn_head', True):
            head_layers = []
            prev_dim = self.input_dim
            
            for dim in config['head_dims']:
                head_layers.extend([
                    nn.Linear(prev_dim, dim),
                    nn.BatchNorm1d(dim),
                    nn.GELU(),
                    nn.Dropout(config['head_dropout'])
                ])
                prev_dim = dim
                
            head_layers.append(nn.Linear(prev_dim, 1))
            self.nn_head = nn.Sequential(*head_layers)
            
            # Combination weight initialized to favor trees
            self.combination_weight = nn.Parameter(torch.tensor(0.7))
        else:
            self.nn_head = None
            
        # Output scaling parameters
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))
            
    def forward(self, x):
        # Embed features
        embedded = self.feature_embedder(x)
        
        # Get tree outputs
        tree_outputs = []
        for tree in self.trees:
            output = tree(embedded)
            tree_outputs.append(output)
        
        # Stack tree outputs
        tree_outputs = torch.stack(tree_outputs, dim=1)  # (batch_size, n_trees)
        
        # Get gating weights
        gates = self.gating_network(x)  # (batch_size, n_trees)
        
        # Weighted sum of tree outputs
        forest_output = torch.sum(gates * tree_outputs, dim=1, keepdim=True)
        
        # Combine with neural head if available
        if self.nn_head is not None:
            nn_output = self.nn_head(x)
            weight = torch.sigmoid(self.combination_weight)
            final_output = weight * forest_output + (1 - weight) * nn_output
        else:
            final_output = forest_output
            
        # Scale and shift output
        final_output = final_output * self.output_scale + self.output_bias
            
        return final_output

class SimplifiedGANDALF(nn.Module):
    """Simplified GANDALF model with better numerical stability"""
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config['input_dim']
        self.n_estimators = config.get('n_estimators', 20)
        self.tree_dim = config.get('tree_dim', 128)
        self.depth = config.get('depth', 3)
        
        # Feature transformation layers
        self.feature_layers = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.tree_dim),
            nn.BatchNorm1d(self.tree_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Soft decision trees as simple neural networks
        self.trees = nn.ModuleList()
        for _ in range(self.n_estimators):
            tree = nn.Sequential(
                nn.Linear(self.tree_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(32, 1)
            )
            self.trees.append(tree)
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.n_estimators),
            nn.Softmax(dim=1)
        )
        
        # Output combination
        self.output_weight = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Transform features
        tree_features = self.feature_layers(x)
        
        # Get tree outputs
        tree_outputs = []
        for tree in self.trees:
            output = tree(tree_features)
            tree_outputs.append(output)
        
        tree_outputs = torch.cat(tree_outputs, dim=1)  # (batch_size, n_estimators)
        
        # Get gates
        gates = self.gate(x)  # (batch_size, n_estimators)
        
        # Weighted combination
        output = torch.sum(tree_outputs * gates, dim=1, keepdim=True)
        
        # Scale and shift
        output = output * self.output_weight + self.output_bias
        
        return output

class ANAM(nn.Module):
    """Additive Neural Attention Model"""
    def __init__(self, n_features, shape_hidden_dim=64, shape_n_hidden=2, 
                 attention_hidden_dim=128, n_heads=4, dropout=0.2):
        super().__init__()
        
        self.n_features = n_features
        self.n_heads = n_heads
        
        # Shape functions for each feature
        self.shape_functions = nn.ModuleList()
        for _ in range(n_features):
            layers = []
            in_dim = 1
            
            for i in range(shape_n_hidden):
                layers.extend([
                    nn.Linear(in_dim, shape_hidden_dim),
                    nn.LayerNorm(shape_hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.1)
                ])
                in_dim = shape_hidden_dim
            
            layers.append(nn.Linear(shape_hidden_dim, 1))
            self.shape_functions.append(nn.Sequential(*layers))
        
        # Feature embeddings for attention
        self.feature_embeddings = nn.Parameter(torch.randn(n_features, attention_hidden_dim))
        nn.init.xavier_uniform_(self.feature_embeddings)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=attention_hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Context vector for attention
        self.context = nn.Parameter(torch.randn(1, attention_hidden_dim))
        nn.init.xavier_uniform_(self.context)
        
        # Final aggregation
        self.output_layer = nn.Sequential(
            nn.Linear(n_features, attention_hidden_dim),
            nn.LayerNorm(attention_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(attention_hidden_dim, 1)
        )
        
        # Learnable bias
        self.bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x, return_components=False):
        batch_size = x.size(0)
        
        # Apply shape functions to each feature
        shape_outputs = []
        for i in range(self.n_features):
            feature_input = x[:, i:i+1]  # (batch_size, 1)
            shape_output = self.shape_functions[i](feature_input)
            shape_outputs.append(shape_output)
        
        shape_outputs = torch.cat(shape_outputs, dim=1)  # (batch_size, n_features)
        
        # Prepare for attention
        # Expand feature embeddings for batch
        feature_embeds = self.feature_embeddings.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Expand context for batch
        context = self.context.expand(batch_size, -1, -1)
        
        # Apply attention
        attended_features, attention_weights = self.attention(
            query=context,
            key=feature_embeds,
            value=feature_embeds
        )
        
        # Attention weights shape: (batch_size, 1, n_features)
        attention_weights = attention_weights.squeeze(1)  # (batch_size, n_features)
        
        # Apply attention weights to shape outputs
        weighted_outputs = shape_outputs * attention_weights
        
        # Final output
        output = self.output_layer(weighted_outputs) + self.bias
        
        if return_components:
            return output, shape_outputs, attention_weights
        
        return output
    
    def get_feature_importance(self, x):
        """Calculate feature importance based on attention weights"""
        self.eval()
        with torch.no_grad():
            _, _, attention_weights = self.forward(x, return_components=True)
            # Average attention weights across batch
            importance = attention_weights.mean(dim=0)
        return importance

# =========================
# TANGOS Model Components
# =========================
class GatedFeatureExtractor(nn.Module):
    """Gated mechanism for feature extraction with smoothing"""
    def __init__(self, input_dim, hidden_dim, dropout=0.2):
        super().__init__()
        
        # Gate networks
        self.gate_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )
        
        # Feature transformation
        self.feature_transform = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Smoothing layer
        self.smoothing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        # Apply gating
        gates = self.gate_transform(x)
        features = self.feature_transform(x)
        
        # Gated features
        gated_features = gates * features
        
        # Apply smoothing
        smoothed = self.smoothing(gated_features)
        
        # Residual connection
        return smoothed + gated_features

class TemporalInspiredBlock(nn.Module):
    """Temporal-inspired processing block even though this is regression"""
    def __init__(self, input_dim, hidden_dim, n_layers=2, dropout=0.2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(n_layers):
            layer_input_dim = input_dim if i == 0 else hidden_dim
            
            # GRU-inspired gating without sequential dependency
            self.layers.append(nn.Linear(layer_input_dim, hidden_dim * 3))
            self.norms.append(nn.LayerNorm(hidden_dim))
            
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        h = x
        
        for layer, norm in zip(self.layers, self.norms):
            # GRU-style gating
            gates = layer(h)
            i, f, g = gates.chunk(3, dim=-1)
            
            i = torch.sigmoid(i)  # Input gate
            f = torch.sigmoid(f)  # Forget gate  
            g = torch.tanh(g)     # Candidate
            
            # Update hidden state
            if h.shape[-1] == i.shape[-1]:
                h = f * h + i * g
            else:
                h = i * g
                
            h = norm(h)
            h = self.dropout(h)
        
        return self.output_projection(h)

class SmoothingAttention(nn.Module):
    """Attention mechanism with built-in smoothing"""
    def __init__(self, hidden_dim, n_heads=8, dropout=0.2, temperature=1.0):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.head_dim = hidden_dim // n_heads
        self.temperature = temperature
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Smoothing parameters
        self.smooth_alpha = nn.Parameter(torch.ones(1) * 0.1)
        
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention with temperature
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (math.sqrt(self.head_dim) * self.temperature)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply smoothing to attention scores
        attn_weights = F.softmax(scores, dim=-1)
        
        # Smooth the attention weights
        smoothing_kernel = torch.ones(1, 1, 3, device=x.device) / 3
        if attn_weights.dim() == 4 and seq_len > 2:
            # Apply 1D smoothing across sequence dimension
            attn_weights_smooth = F.conv1d(
                attn_weights.view(-1, 1, seq_len),
                smoothing_kernel,
                padding=1
            ).view(batch_size, self.n_heads, seq_len, seq_len)
            
            # Blend original and smoothed weights
            attn_weights = (1 - torch.sigmoid(self.smooth_alpha)) * attn_weights + \
                          torch.sigmoid(self.smooth_alpha) * attn_weights_smooth
        
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        
        output = self.out_proj(context)
        
        return output, attn_weights

class TANGOS(nn.Module):
    """Temporal Attention Networks with Gated Operations and Smoothing"""
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config['input_dim']
        self.hidden_dim = config['hidden_dim']
        self.n_layers = config.get('n_layers', 3)
        self.n_heads = config.get('n_heads', 8)
        self.dropout = config.get('dropout', 0.2)
        self.use_feature_gating = config.get('use_feature_gating', True)
        
        # Input projection with batch norm
        self.input_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.BatchNorm1d(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout)
        )
        
        # Gated feature extractor
        if self.use_feature_gating:
            self.feature_gating = GatedFeatureExtractor(
                self.hidden_dim, 
                self.hidden_dim, 
                dropout=self.dropout
            )
        
        # Temporal-inspired blocks
        self.temporal_blocks = nn.ModuleList()
        for _ in range(self.n_layers):
            self.temporal_blocks.append(
                TemporalInspiredBlock(
                    self.hidden_dim,
                    self.hidden_dim,
                    n_layers=2,
                    dropout=self.dropout
                )
            )
        
        # Smoothing attention layers
        self.attention_layers = nn.ModuleList()
        for _ in range(self.n_layers):
            self.attention_layers.append(
                SmoothingAttention(
                    self.hidden_dim,
                    n_heads=self.n_heads,
                    dropout=self.dropout,
                    temperature=1.5
                )
            )
        
        # Layer normalization after each block
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(self.hidden_dim) for _ in range(self.n_layers)
        ])
        
        # Global pooling strategies
        self.pooling_weight = nn.Parameter(torch.ones(3) / 3)
        
        # Output layers with smoothing
        self.output_smooth = nn.Sequential(
            nn.Linear(self.hidden_dim * 3, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(128, 1)
        )
        
        # Learnable smoothing for final output
        self.output_smooth_param = nn.Parameter(torch.tensor(0.1))
        self.running_mean = None
        self.alpha = 0.95  # EMA alpha
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Input projection
        h = self.input_projection(x)
        
        # Apply feature gating if enabled
        if self.use_feature_gating:
            h = self.feature_gating(h)
        
        # Add sequence dimension for attention
        h = h.unsqueeze(1)  # (batch_size, 1, hidden_dim)
        
        # Process through temporal and attention blocks
        for i in range(self.n_layers):
            # Temporal processing
            h_temp = self.temporal_blocks[i](h.squeeze(1)).unsqueeze(1)
            
            # Attention with residual
            h_attn, _ = self.attention_layers[i](h)
            
            # Combine temporal and attention outputs
            h = h + h_temp + h_attn
            h = self.layer_norms[i](h)
        
        # Multi-scale pooling
        h_squeezed = h.squeeze(1)
        
        # Different pooling strategies
        max_pool = h_squeezed
        avg_pool = h_squeezed
        last_hidden = h_squeezed
        
        # Weighted combination of pooling strategies
        weights = F.softmax(self.pooling_weight, dim=0)
        pooled = torch.cat([
            max_pool * weights[0],
            avg_pool * weights[1],
            last_hidden * weights[2]
        ], dim=-1)
        
        # Final output
        output = self.output_smooth(pooled)
        
        # Apply output smoothing during training
        if self.training and self.running_mean is not None:
            # Exponential moving average smoothing
            smoothed_output = (1 - self.alpha) * output + self.alpha * self.running_mean
            self.running_mean = output.detach().mean()
            
            # Blend raw and smoothed outputs
            smooth_weight = torch.sigmoid(self.output_smooth_param)
            output = (1 - smooth_weight) * output + smooth_weight * smoothed_output
        elif self.training:
            # Initialize running mean
            self.running_mean = output.detach().mean()
        
        return output

# =========================
# DOFEN Model Components
# =========================
class ObliviousDecisionTree(nn.Module):
    """Differentiable Oblivious Decision Tree"""
    def __init__(self, depth, num_features, temperature=1.0):
        super().__init__()
        self.depth = depth
        self.num_features = num_features
        self.temperature = temperature
        self.num_leaves = 2 ** depth
        
        # Internal node parameters
        self.feature_indices = nn.Parameter(torch.randn(depth, num_features))
        self.thresholds = nn.Parameter(torch.randn(depth))
        
        # Leaf values
        self.leaf_values = nn.Parameter(torch.randn(self.num_leaves))
        
        # Initialize parameters
        nn.init.xavier_uniform_(self.feature_indices)
        nn.init.zeros_(self.thresholds)
        nn.init.normal_(self.leaf_values, mean=0, std=0.01)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Soft routing through the tree
        decisions = []
        for d in range(self.depth):
            # Soft feature selection using softmax
            feature_weights = F.softmax(self.feature_indices[d] / self.temperature, dim=0)
            selected_feature = (x * feature_weights.unsqueeze(0)).sum(dim=1)
            
            # Soft threshold comparison
            decision = torch.sigmoid((selected_feature - self.thresholds[d]) / self.temperature)
            decisions.append(decision)
        
        # Compute leaf probabilities (soft routing)
        leaf_probs = torch.ones(batch_size, self.num_leaves, device=x.device)
        for d in range(self.depth):
            decision = decisions[d].unsqueeze(1)
            # Update probabilities for left and right subtrees
            mask = torch.zeros(batch_size, self.num_leaves, device=x.device)
            for leaf in range(self.num_leaves):
                if (leaf >> (self.depth - 1 - d)) & 1:
                    mask[:, leaf] = 1
            leaf_probs = leaf_probs * (mask * decision + (1 - mask) * (1 - decision))
        
        # Normalize probabilities for stability
        leaf_probs = leaf_probs + 1e-10
        leaf_probs = leaf_probs / leaf_probs.sum(dim=1, keepdim=True)
        
        # Weighted sum of leaf values
        output = (leaf_probs * self.leaf_values.unsqueeze(0)).sum(dim=1)
        
        return output

class SparseFeatureSelection(nn.Module):
    """Sparse column selection layer for DOFEN"""
    def __init__(self, num_features, num_selected, temperature=1.0):
        super().__init__()
        self.num_features = num_features
        self.num_selected = num_selected
        self.temperature = temperature
        
        # Feature selection scores
        self.scores = nn.Parameter(torch.randn(num_features))
        nn.init.normal_(self.scores, mean=0, std=0.1)
        
    def forward(self, x):
        # Compute selection probabilities
        if self.training:
            # Add Gumbel noise for exploration during training
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(self.scores) + 1e-10) + 1e-10)
            scores_with_noise = (self.scores + gumbel_noise) / self.temperature
        else:
            scores_with_noise = self.scores / self.temperature
        
        # Get top-k features (soft selection)
        if self.num_selected < self.num_features:
            _, indices = torch.topk(scores_with_noise, self.num_selected)
            mask = torch.zeros_like(self.scores)
            mask.scatter_(0, indices, 1.0)
            
            # Soft mask for gradient flow
            soft_mask = torch.sigmoid(self.scores * 5)
            if not self.training:
                mask = soft_mask
        else:
            mask = torch.ones_like(self.scores)
        
        # Apply mask with small residual connection for stability
        selected_features = x * mask.unsqueeze(0) + x * 0.1
        
        return selected_features, mask

class ObliviousForestLayer(nn.Module):
    """Single layer of oblivious forest"""
    def __init__(self, num_trees, depth, num_features, temperature=1.0):
        super().__init__()
        self.num_trees = num_trees
        self.num_features = num_features
        self.trees = nn.ModuleList([
            ObliviousDecisionTree(depth, num_features, temperature)
            for _ in range(num_trees)
        ])
        
        # Tree weights for ensemble
        self.tree_weights = nn.Parameter(torch.ones(num_trees) / num_trees)
        
    def forward(self, x):
        # Ensemble predictions
        tree_outputs = []
        for tree in self.trees:
            tree_outputs.append(tree(x).unsqueeze(1))
        
        tree_outputs = torch.cat(tree_outputs, dim=1)  # (batch_size, num_trees)
        
        # Weighted ensemble
        weights = F.softmax(self.tree_weights, dim=0)
        output = (tree_outputs * weights.unsqueeze(0)).sum(dim=1)
        
        return output, tree_outputs

class DOFEN(nn.Module):
    """Deep Oblivious Forest ENsemble"""
    def __init__(self, num_features, num_layers=2, trees_per_layer=[50, 25], 
                 tree_depth=4, feature_subset_ratio=0.6, temperature=1.5,
                 use_sparse_selection=True, dropout=0.2):
        super().__init__()
        self.num_features = num_features
        self.num_layers = num_layers
        self.use_sparse_selection = use_sparse_selection
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(num_features)
        
        # Sparse feature selection
        if use_sparse_selection:
            num_selected = int(num_features * feature_subset_ratio)
            self.feature_selector = SparseFeatureSelection(num_features, num_selected, temperature)
        
        # Forest layers
        self.forest_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = ObliviousForestLayer(
                num_trees=trees_per_layer[i],
                depth=tree_depth,
                num_features=num_features,
                temperature=temperature
            )
            self.forest_layers.append(layer)
        
        # Layer normalization
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(1) for _ in range(num_layers)
        ])
        
        # Combination weights
        self.layer_weights = nn.Parameter(torch.ones(num_layers) / num_layers)
        
        # Final projection
        self.final_layer = nn.Sequential(
            nn.Linear(num_layers, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        # Output scaling
        self.output_scale = nn.Parameter(torch.ones(1))
        self.output_bias = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        # Input normalization
        x = self.input_norm(x)
        
        # Feature selection
        if self.use_sparse_selection:
            x_selected, feature_mask = self.feature_selector(x)
        else:
            x_selected = x
        
        # Forward through forest layers
        layer_outputs = []
        
        for i, (layer, norm) in enumerate(zip(self.forest_layers, self.layer_norms)):
            layer_output, _ = layer(x_selected)
            layer_output = norm(layer_output.unsqueeze(1)).squeeze(1)
            layer_outputs.append(layer_output.unsqueeze(1))
        
        # Combine layer outputs
        layer_outputs = torch.cat(layer_outputs, dim=1)  # (batch_size, num_layers)
        
        # Weighted combination
        weights = F.softmax(self.layer_weights, dim=0)
        weighted_output = (layer_outputs * weights.unsqueeze(0)).sum(dim=1, keepdim=True)
        
        # Final prediction through neural layers
        output = self.final_layer(layer_outputs).squeeze()
        
        # Combine neural output with weighted forest output
        final_output = 0.7 * output + 0.3 * weighted_output.squeeze()
        
        # Apply scaling
        final_output = final_output * self.output_scale + self.output_bias
        
        return final_output

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

            model = XGBRegressor(**XGB_PARAMS)
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
    
    # Also save OOF predictions for ensemble analysis
    np.save(PREDICTIONS_DIR / "xgboost_oof.npy", oof_weighted)
    
    return test_weighted, oof_weighted

# =========================
# MLP Training
# =========================
def train_mlp(train_df, test_df):
    """Train MLP model"""
    print("\n=== Training MLP Model ===")
    
    hparams = {
        "seed": 42,
        "num_epochs": 10,
        "batch_size": 1024 * 8 * 4,
        "learning_rate": 0.001,
        "weight_decay": 1e-3,
        "dropout_rate": 0.6,
        "layers": [len(Config.MLP_FEATURES), 256, 64, 1],
        "hidden_activation": None,
        "activation": "relu",
        "delta": 5,
        "noise_factor": 0.005
    }
    
    set_seed(hparams["seed"])
    
    X_train_full = train_df[Config.MLP_FEATURES].values
    y_train_full = train_df[Config.LABEL_COLUMN].values
    
    # Keep track of OOF predictions
    oof_preds = np.zeros(len(train_df))
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    
    # Collect all fold predictions for ensemble
    test_preds_all_folds = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_full), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        
        X_train = X_train_full[train_idx]
        y_train = y_train_full[train_idx]
        X_val = X_train_full[valid_idx]
        y_val = y_train_full[valid_idx]
        
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(test_df[Config.MLP_FEATURES].values)
        
        # Create dataloaders
        train_loader = get_dataloaders(X_train, y_train, hparams, device, shuffle=True)
        val_loader = get_dataloaders(X_val, y_val, hparams, device, shuffle=False)
        test_loader = get_dataloaders(X_test, None, hparams, device, shuffle=False)
        
        model = MLP(
            layers=hparams["layers"],
            dropout_rate=hparams["dropout_rate"],
            activation=hparams["activation"],
            last_activation=hparams["hidden_activation"],
        ).to(device)
        
        criterion = nn.HuberLoss(delta=hparams["delta"], reduction='sum')
        optimizer = optim.Adam(model.parameters(), lr=hparams["learning_rate"], 
                              weight_decay=hparams["weight_decay"])
        
        checkpoint_path = CHECKPOINTS_DIR / f"mlp_fold{fold}_model.pt"
        best_pearson = -np.inf
        
        # Training loop
        num_epochs = hparams["num_epochs"]
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                inputs = inputs + torch.randn_like(inputs) * hparams["noise_factor"]
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                
            running_loss = running_loss / len(train_loader.dataset)

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

            if pearson_coef > best_pearson:
                best_pearson = pearson_coef
                torch.save(model.state_dict(), checkpoint_path)
                
        # Load best model and save OOF predictions
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        val_preds = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_preds.append(outputs.cpu().numpy())
        
        oof_preds[valid_idx] = np.concatenate(val_preds).flatten()
        
        # Make test predictions
        test_preds = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)
                outputs = model(inputs)
                test_preds.append(outputs.cpu().numpy())
        
        test_preds_all_folds.append(np.concatenate(test_preds).flatten())
    
    # Average test predictions across folds
    test_predictions = np.mean(test_preds_all_folds, axis=0)
    
    # Calculate overall OOF score
    oof_score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds)[0]
    print(f"\nMLP Overall OOF Pearson: {oof_score:.4f}")
    
    # Save predictions
    save_predictions("mlp", test_predictions, {
        "oof_score": oof_score,
        "n_folds": Config.N_FOLDS
    })
    
    # Save OOF predictions for ensemble analysis
    np.save(PREDICTIONS_DIR / "mlp_oof.npy", oof_preds)
    
    return test_predictions, oof_preds

# =========================
# DCN V2 Training
# =========================
def train_dcnv2(train_df, test_df):
    """Train DCN V2 model"""
    print("\n=== Training DCN V2 Model ===")
    
    hparams = {
        "seed": 42,
        "num_epochs": 15,
        "batch_size": 512,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "num_cross_layers": 3,
        "num_experts": 4,
        "deep_dims": [256, 128, 64],
        "dropout": 0.2,
        "gradient_clip": 1.0,
        "patience": 5,
        "noise_factor": 0.005
    }
    
    set_seed(hparams["seed"])
    
    # Use MLP features for DCN V2
    X_train_full = train_df[Config.MLP_FEATURES].values
    y_train_full = train_df[Config.LABEL_COLUMN].values
    
    print(f"Using {len(Config.MLP_FEATURES)} features for DCN V2")
    
    # Keep track of OOF predictions
    oof_preds = np.zeros(len(train_df))
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    
    # Collect all fold predictions
    test_preds_all_folds = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_full), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        
        X_train = X_train_full[train_idx]
        y_train = y_train_full[train_idx]
        X_val = X_train_full[valid_idx]
        y_val = y_train_full[valid_idx]
        
        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(test_df[Config.MLP_FEATURES].values)
        
        # Create dataloaders
        train_loader = get_dataloaders(X_train, y_train, hparams, device, shuffle=True)
        val_loader = get_dataloaders(X_val, y_val, hparams, device, shuffle=False)
        test_loader = get_dataloaders(X_test, None, hparams, device, shuffle=False)
        
        # Initialize model
        model = DCNV2(
            input_dim=len(Config.MLP_FEATURES),
            num_cross_layers=hparams["num_cross_layers"],
            num_experts=hparams["num_experts"],
            deep_dims=hparams["deep_dims"],
            dropout=hparams["dropout"]
        ).to(device)
        
        criterion = nn.HuberLoss(delta=1.0, reduction='mean')
        optimizer = optim.AdamW(model.parameters(), lr=hparams["learning_rate"], 
                                weight_decay=hparams["weight_decay"])
        
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-6)
        
        checkpoint_path = CHECKPOINTS_DIR / f"dcnv2_fold{fold}_model.pt"
        best_pearson = -np.inf
        patience_counter = 0
        
        # Training loop
        num_epochs = hparams["num_epochs"]
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Add noise for robustness
                inputs = inputs + torch.randn_like(inputs) * hparams["noise_factor"]
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams["gradient_clip"])
                
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            running_loss = running_loss / len(train_loader.dataset)

            # Validation phase
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

            scheduler.step(pearson_coef)
            
            # Save best model
            if pearson_coef > best_pearson:
                best_pearson = pearson_coef
                torch.save(model.state_dict(), checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= hparams["patience"]:
                    break
        
        # Load best model and save OOF predictions
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        val_preds = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_preds.append(outputs.cpu().numpy())
        
        oof_preds[valid_idx] = np.concatenate(val_preds).flatten()
        
        # Make test predictions
        predictions = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())

        test_preds_all_folds.append(np.concatenate(predictions).flatten())
    
    # Average test predictions across folds
    test_predictions = np.mean(test_preds_all_folds, axis=0)
    
    # Calculate overall OOF score
    oof_score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds)[0]
    print(f"\nDCN V2 Overall OOF Pearson: {oof_score:.4f}")
    
    # Save predictions
    save_predictions("dcnv2", test_predictions, {
        "oof_score": oof_score,
        "n_features": len(Config.MLP_FEATURES)
    })
    
    # Save OOF predictions for ensemble analysis
    np.save(PREDICTIONS_DIR / "dcnv2_oof.npy", oof_preds)
    
    return test_predictions, oof_preds

# =========================
# GANDALF Training
# =========================
def train_gandalf_model(model, train_loader, val_loader, config, device, checkpoint_path):
    """Train GANDALF model with improved stability"""
    
    criterion = nn.HuberLoss(delta=config.get('huber_delta', 1.0))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-6)
    
    best_val_pearson = -np.inf
    patience_counter = 0
    patience = config.get('patience', 10)
    num_epochs = config.get('num_epochs', 30)
    
    saved_checkpoint = False
    
    # Use mixed precision training if available
    use_amp = config.get('use_amp', True) and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Add noise for regularization
            if config.get('noise_factor', 0) > 0:
                noise = torch.randn_like(inputs) * config['noise_factor']
                inputs = inputs + noise
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch+1}, skipping batch")
                    continue
                    
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch+1}, skipping batch")
                    continue
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        if train_batches == 0:
            print(f"No valid batches in epoch {epoch+1}, stopping training")
            break
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_preds.extend(outputs.cpu().numpy().flatten())
                    val_targets.extend(targets.cpu().numpy().flatten())
                    val_batches += 1
        
        if val_batches == 0:
            print(f"No valid validation batches, stopping training")
            break
        
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / val_batches
        
        if len(val_preds) > 0:
            val_pearson = pearsonr(val_targets, val_preds)[0]
            val_spearman = spearmanr(val_targets, val_preds)[0]
        else:
            val_pearson = -np.inf
            val_spearman = -np.inf
        
        print(f"\nTrain Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Val Pearson: {val_pearson:.4f}, Val Spearman: {val_spearman:.4f}")
        
        scheduler.step(val_pearson)
        
        if not np.isnan(val_pearson) and val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            saved_checkpoint = True
            print(f"âœ… New best model saved! Pearson: {best_val_pearson:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model if saved
    if saved_checkpoint and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path))
    
    return model, best_val_pearson

def train_gandalf(train_df, test_df):
    """Train both GANDALF and Simplified GANDALF models"""
    print("\n=== Training GANDALF Models ===")
    
    set_seed(42)
    
    # Prepare features
    gandalf_features = Config.GANDALF_FEATURES.copy()
    engineered_features = [
        "log_volume", 'bid_ask_interaction', 'net_order_flow', 'normalized_net_flow',
        'buying_pressure', 'total_depth', 'depth_imbalance', 'kyle_lambda', 
        'aggressive_flow_ratio', 'volume_depth_ratio', 'log_buy_qty', 'log_sell_qty',
        'price_impact_proxy', 'market_stress', 'liquidity_consumption'
    ]
    
    all_gandalf_features = gandalf_features + engineered_features
    all_gandalf_features = list(set(all_gandalf_features))
    all_gandalf_features = [f for f in all_gandalf_features if f in train_df.columns]
    
    print(f"Using {len(all_gandalf_features)} features for GANDALF")
    
    # Feature selection
    print("\nSelecting features...")
    selector = SelectKBest(score_func=mutual_info_regression, k=min(80, len(all_gandalf_features)))
    X_train_full = train_df[all_gandalf_features].values
    y_train_full = train_df[Config.LABEL_COLUMN].values
    X_train_selected = selector.fit_transform(X_train_full, y_train_full)
    selected_features = [all_gandalf_features[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} features")
    
    # Keep track of OOF predictions for both models
    oof_preds_gandalf = np.zeros(len(train_df))
    oof_preds_simplified = np.zeros(len(train_df))
    
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    
    # Collect all fold predictions
    test_preds_gandalf_folds = []
    test_preds_simplified_folds = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_selected), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        
        X_train = X_train_selected[train_idx]
        y_train = y_train_full[train_idx]
        X_val = X_train_selected[valid_idx]
        y_val = y_train_full[valid_idx]
        
        # Transform data using RobustScaler for better stability
        transformer = RobustScaler()
        X_train_transformed = transformer.fit_transform(X_train)
        X_val_transformed = transformer.transform(X_val)
        
        # Clip extreme values
        clip_value = 5.0
        X_train_transformed = np.clip(X_train_transformed, -clip_value, clip_value)
        X_val_transformed = np.clip(X_val_transformed, -clip_value, clip_value)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train_transformed, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_transformed, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        
        # 1. Train Original GANDALF
        print(f"\n=== Training Original GANDALF - Fold {fold} ===")
        
        gandalf_config = {
            'input_dim': X_train_transformed.shape[1],
            'n_trees': 10,
            'tree_depth': 4,
            'tree_temperature': 1.5,
            'embed_dims': [128, 64],
            'feature_dropout': 0.2,
            'gate_hidden_dim': 64,
            'gate_dropout': 0.2,
            'use_nn_head': True,
            'head_dims': [128, 64],
            'head_dropout': 0.3,
            'learning_rate': 0.0001,
            'weight_decay': 0.01,
            'huber_delta': 1.0,
            'noise_factor': 0.005,
            'grad_clip': 0.5,
            'num_epochs': 15,
            'patience': 5,
            'use_amp': True
        }
        
        gandalf_model = GANDALF(gandalf_config).to(device)
        gandalf_checkpoint = CHECKPOINTS_DIR / f"gandalf_fold{fold}_model.pt"
        gandalf_model, gandalf_score = train_gandalf_model(
            gandalf_model, train_loader, val_loader, gandalf_config, device, gandalf_checkpoint
        )
        
        # 2. Train Simplified GANDALF
        print(f"\n=== Training Simplified GANDALF - Fold {fold} ===")
        
        simplified_config = {
            'input_dim': X_train_transformed.shape[1],
            'n_estimators': 15,
            'tree_dim': 64,
            'depth': 3,
            'learning_rate': 0.0005,
            'weight_decay': 0.01,
            'huber_delta': 1.0,
            'noise_factor': 0.01,
            'grad_clip': 0.5,
            'num_epochs': 15,
            'patience': 5
        }
        
        simplified_model = SimplifiedGANDALF(simplified_config).to(device)
        simplified_checkpoint = CHECKPOINTS_DIR / f"simplified_gandalf_fold{fold}_model.pt"
        simplified_model, simplified_score = train_gandalf_model(
            simplified_model, train_loader, val_loader, simplified_config, device, simplified_checkpoint
        )
        
        # Make validation predictions
        gandalf_model.eval()
        simplified_model.eval()
        
        val_preds_gandalf = []
        val_preds_simplified = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                
                outputs_gandalf = gandalf_model(inputs)
                val_preds_gandalf.extend(outputs_gandalf.cpu().numpy().flatten())
                
                outputs_simplified = simplified_model(inputs)
                val_preds_simplified.extend(outputs_simplified.cpu().numpy().flatten())
        
        oof_preds_gandalf[valid_idx] = np.array(val_preds_gandalf)
        oof_preds_simplified[valid_idx] = np.array(val_preds_simplified)
        
        # Make test predictions
        X_test = test_df[all_gandalf_features].values
        X_test_selected = selector.transform(X_test)
        X_test_transformed = transformer.transform(X_test_selected)
        X_test_transformed = np.clip(X_test_transformed, -clip_value, clip_value)
        
        test_dataset = TensorDataset(torch.tensor(X_test_transformed, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
        
        gandalf_predictions = []
        simplified_predictions = []
        
        with torch.no_grad():
            for (inputs,) in test_loader:
                inputs = inputs.to(device)
                
                outputs_gandalf = gandalf_model(inputs)
                gandalf_predictions.extend(outputs_gandalf.cpu().numpy().flatten())
                
                outputs_simplified = simplified_model(inputs)
                simplified_predictions.extend(outputs_simplified.cpu().numpy().flatten())
        
        test_preds_gandalf_folds.append(np.array(gandalf_predictions))
        test_preds_simplified_folds.append(np.array(simplified_predictions))
    
    # Average test predictions across folds
    test_gandalf_final = np.mean(test_preds_gandalf_folds, axis=0)
    test_simplified_final = np.mean(test_preds_simplified_folds, axis=0)
    
    # Post-processing
    pred_mean = train_df[Config.LABEL_COLUMN].mean()
    pred_std = train_df[Config.LABEL_COLUMN].std()
    
    test_gandalf_final = np.clip(test_gandalf_final, pred_mean - 4 * pred_std, pred_mean + 4 * pred_std)
    test_simplified_final = np.clip(test_simplified_final, pred_mean - 4 * pred_std, pred_mean + 4 * pred_std)
    
    # Calculate OOF scores
    oof_score_gandalf = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds_gandalf)[0]
    oof_score_simplified = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds_simplified)[0]
    
    print(f"\nGANDALF Overall OOF Pearson: {oof_score_gandalf:.4f}")
    print(f"Simplified GANDALF Overall OOF Pearson: {oof_score_simplified:.4f}")
    
    # Save predictions
    save_predictions("gandalf", test_gandalf_final, {
        "oof_score": oof_score_gandalf,
        "n_features": len(selected_features),
        "model_type": "original_differentiable_trees"
    })
    
    save_predictions("simplified_gandalf", test_simplified_final, {
        "oof_score": oof_score_simplified,
        "n_features": len(selected_features),
        "model_type": "simplified_neural_trees"
    })
    
    # Save OOF predictions
    np.save(PREDICTIONS_DIR / "gandalf_oof.npy", oof_preds_gandalf)
    np.save(PREDICTIONS_DIR / "simplified_gandalf_oof.npy", oof_preds_simplified)
    
    return test_gandalf_final, test_simplified_final, oof_preds_gandalf, oof_preds_simplified

# =========================
# ANAM Training
# =========================
def train_anam(train_df, test_df):
    """Train ANAM model"""
    print("\n=== Training ANAM Model ===")
    
    # Hyperparameters
    hparams = {
        "seed": 42,
        "num_epochs": 20,
        "batch_size": 256,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "shape_hidden_dim": 64,
        "shape_n_hidden": 2,
        "attention_hidden_dim": 128,
        "n_heads": 4,
        "dropout": 0.2,
        "gradient_clip": 1.0,
        "warmup_epochs": 5,
        "patience": 5,
        "noise_factor": 0.005
    }
    
    set_seed(hparams["seed"])
    
    # Prepare data for ANAM using MLP features
    X_train_full = train_df[Config.MLP_FEATURES].values
    y_train_full = train_df[Config.LABEL_COLUMN].values
    
    print(f"Using {len(Config.MLP_FEATURES)} features for ANAM")
    
    # Keep track of OOF predictions
    oof_preds = np.zeros(len(train_df))
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    
    # Collect all fold predictions
    test_preds_all_folds = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_full), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        
        X_train = X_train_full[train_idx]
        y_train = y_train_full[train_idx]
        X_val = X_train_full[valid_idx]
        y_val = y_train_full[valid_idx]
        
        # Scale data
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(test_df[Config.MLP_FEATURES].values)
        
        # Create dataloaders
        train_loader = get_dataloaders(X_train, y_train, hparams, device, shuffle=True)
        val_loader = get_dataloaders(X_val, y_val, hparams, device, shuffle=False)
        test_loader = get_dataloaders(X_test, None, hparams, device, shuffle=False)
        
        # Initialize model
        model = ANAM(
            n_features=len(Config.MLP_FEATURES),
            shape_hidden_dim=hparams["shape_hidden_dim"],
            shape_n_hidden=hparams["shape_n_hidden"],
            attention_hidden_dim=hparams["attention_hidden_dim"],
            n_heads=hparams["n_heads"],
            dropout=hparams["dropout"]
        ).to(device)
        
        criterion = nn.HuberLoss(delta=1.0, reduction='mean')
        optimizer = optim.AdamW(model.parameters(), lr=hparams["learning_rate"], 
                                weight_decay=hparams["weight_decay"])
        
        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < hparams["warmup_epochs"]:
                return (epoch + 1) / hparams["warmup_epochs"]
            else:
                return 0.5 * (1 + math.cos(math.pi * (epoch - hparams["warmup_epochs"]) / 
                                          (hparams["num_epochs"] - hparams["warmup_epochs"])))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        checkpoint_path = CHECKPOINTS_DIR / f"anam_fold{fold}_model.pt"
        best_val_pearson = -np.inf
        patience_counter = 0
        
        # Training loop
        num_epochs = hparams["num_epochs"]
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Add noise for robustness
                inputs = inputs + torch.randn_like(inputs) * hparams["noise_factor"]
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams["gradient_clip"])
                
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            scheduler.step()
            
            running_loss = running_loss / len(train_loader.dataset)

            # Validation phase
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

            if pearson_coef > best_val_pearson:
                best_val_pearson = pearson_coef
                torch.save(model.state_dict(), checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= hparams["patience"]:
                    break
        
        # Load best model and save OOF predictions
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        val_preds = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_preds.append(outputs.cpu().numpy())
        
        oof_preds[valid_idx] = np.concatenate(val_preds).flatten()
        
        # Make test predictions
        predictions = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())

        test_preds_all_folds.append(np.concatenate(predictions).flatten())
    
    # Average test predictions across folds
    test_predictions = np.mean(test_preds_all_folds, axis=0)
    
    # Calculate overall OOF score
    oof_score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds)[0]
    print(f"\nANAM Overall OOF Pearson: {oof_score:.4f}")
    
    # Save predictions
    save_predictions("anam", test_predictions, {
        "oof_score": oof_score,
        "n_features": len(Config.MLP_FEATURES)
    })
    
    # Save OOF predictions
    np.save(PREDICTIONS_DIR / "anam_oof.npy", oof_preds)
    
    return test_predictions, oof_preds

# =========================
# TANGOS Training
# =========================
def train_tangos(train_df, test_df):
    """Train TANGOS model"""
    print("\n=== Training TANGOS Model ===")
    
    # Hyperparameters
    hparams = {
        "seed": 42,
        "num_epochs": 15,
        "batch_size": 512,
        "learning_rate": 0.0005,
        "weight_decay": 1e-4,
        "hidden_dim": 256,
        "n_layers": 3,
        "n_heads": 8,
        "dropout": 0.25,
        "gradient_clip": 0.5,
        "patience": 5,
        "noise_factor": 0.01,
        "use_feature_gating": True,
        "warmup_steps": 1000
    }
    
    set_seed(hparams["seed"])
    
    # Prepare features - use GANDALF features for TANGOS
    tangos_features = Config.GANDALF_FEATURES.copy()
    
    # Add select engineered features
    engineered_features = [
        "log_volume", "net_order_flow", "normalized_net_flow",
        "total_depth", "depth_imbalance", "kyle_lambda",
        "volume_depth_ratio", "price_impact_proxy",
        "market_stress", "liquidity_consumption"
    ]
    
    all_tangos_features = tangos_features + engineered_features
    all_tangos_features = list(set(all_tangos_features))
    all_tangos_features = [f for f in all_tangos_features if f in train_df.columns]
    
    print(f"Using {len(all_tangos_features)} features for TANGOS")
    
    # Use recent data with time decay
    X_train_full = train_df[all_tangos_features].values
    y_train_full = train_df[Config.LABEL_COLUMN].values
    
    # Keep track of OOF predictions
    oof_preds = np.zeros(len(train_df))
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    
    # Collect all fold predictions
    test_preds_all_folds = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_full), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        
        X_train = X_train_full[train_idx]
        y_train = y_train_full[train_idx]
        X_val = X_train_full[valid_idx]
        y_val = y_train_full[valid_idx]
        
        # Use RobustScaler for outlier resistance
        scaler = RobustScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(test_df[all_tangos_features].values)
        
        # Clip extreme values
        clip_value = 5.0
        X_train = np.clip(X_train, -clip_value, clip_value)
        X_val = np.clip(X_val, -clip_value, clip_value)
        X_test = np.clip(X_test, -clip_value, clip_value)
        
        # Create dataloaders
        train_loader = get_dataloaders(X_train, y_train, hparams, device, shuffle=True)
        val_loader = get_dataloaders(X_val, y_val, hparams, device, shuffle=False)
        test_loader = get_dataloaders(X_test, None, hparams, device, shuffle=False)
        
        # Initialize model
        config = {
            'input_dim': len(all_tangos_features),
            'hidden_dim': hparams['hidden_dim'],
            'n_layers': hparams['n_layers'],
            'n_heads': hparams['n_heads'],
            'dropout': hparams['dropout'],
            'use_feature_gating': hparams['use_feature_gating']
        }
        
        model = TANGOS(config).to(device)
        
        # Loss and optimizer
        criterion = nn.HuberLoss(delta=1.0, reduction='mean')
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"],
            betas=(0.9, 0.999)
        )
        
        # Warmup scheduler
        def lr_lambda(step):
            if step < hparams["warmup_steps"]:
                return step / hparams["warmup_steps"]
            return 1.0
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        checkpoint_path = CHECKPOINTS_DIR / f"tangos_fold{fold}_model.pt"
        best_pearson = -np.inf
        patience_counter = 0
        
        # Training loop
        global_step = 0
        num_epochs = hparams["num_epochs"]
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Add noise for robustness
                if hparams["noise_factor"] > 0:
                    inputs = inputs + torch.randn_like(inputs) * hparams["noise_factor"]
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams["gradient_clip"])
                
                optimizer.step()
                scheduler.step()
                
                running_loss += loss.item() * inputs.size(0)
                global_step += 1
            
            running_loss = running_loss / len(train_loader.dataset)

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

            # Save best model
            if pearson_coef > best_pearson:
                best_pearson = pearson_coef
                torch.save(model.state_dict(), checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= hparams["patience"]:
                    break
        
        # Load best model and save OOF predictions
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        val_preds = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_preds.append(outputs.cpu().numpy())
        
        oof_preds[valid_idx] = np.concatenate(val_preds).flatten()
        
        # Make test predictions
        predictions = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)
                outputs = model(inputs)
                predictions.append(outputs.cpu().numpy())

        test_preds_all_folds.append(np.concatenate(predictions).flatten())
    
    # Average test predictions across folds
    test_predictions = np.mean(test_preds_all_folds, axis=0)
    
    # Post-process predictions
    pred_mean = train_df[Config.LABEL_COLUMN].mean()
    pred_std = train_df[Config.LABEL_COLUMN].std()
    test_predictions = np.clip(test_predictions, pred_mean - 4 * pred_std, pred_mean + 4 * pred_std)
    
    # Calculate overall OOF score
    oof_score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds)[0]
    print(f"\nTANGOS Overall OOF Pearson: {oof_score:.4f}")
    
    # Save predictions
    save_predictions("tangos", test_predictions, {
        "oof_score": oof_score,
        "n_features": len(all_tangos_features)
    })
    
    # Save OOF predictions
    np.save(PREDICTIONS_DIR / "tangos_oof.npy", oof_preds)
    
    return test_predictions, oof_preds

# =========================
# DOFEN Training
# =========================
def train_dofen(train_df, test_df):
    """Train DOFEN model"""
    print("\n=== Training DOFEN Model ===")
    print("Deep Oblivious Forest ENsemble - State-of-the-art for tabular data")
    
    # Hyperparameters
    hparams = {
        "seed": 42,
        "num_epochs": 20,
        "batch_size": 512,
        "learning_rate": 0.0005,
        "weight_decay": 1e-4,
        "num_layers": 2,
        "trees_per_layer": [50, 25],
        "tree_depth": 4,
        "feature_subset_ratio": 0.6,
        "temperature": 1.5,
        "use_sparse_selection": True,
        "dropout": 0.2,
        "gradient_clip": 0.5,
        "patience": 5,
        "warmup_epochs": 3
    }
    
    set_seed(hparams["seed"])
    
    # Use GANDALF features plus some additional ones for DOFEN
    dofen_features = Config.GANDALF_FEATURES.copy()
    
    # Add specific engineered features that work well with trees
    tree_friendly_features = [
        "log_volume", "net_order_flow", "normalized_net_flow",
        "total_depth", "depth_imbalance", "kyle_lambda",
        "volume_depth_ratio", "price_impact_proxy",
        "order_flow_imbalance", "bid_ask_imbalance",
        "liquidity_ratio", "market_stress", 
        "trade_informativeness", "execution_rate",
        "buy_sell_ratio", "selling_pressure"
    ]
    
    all_dofen_features = dofen_features + tree_friendly_features
    all_dofen_features = list(set(all_dofen_features))
    all_dofen_features = [f for f in all_dofen_features if f in train_df.columns]
    
    print(f"Using {len(all_dofen_features)} features for DOFEN")
    
    # Prepare data
    X_train_full = train_df[all_dofen_features].values
    y_train_full = train_df[Config.LABEL_COLUMN].values
    
    # Keep track of OOF predictions
    oof_preds = np.zeros(len(train_df))
    kf = KFold(n_splits=Config.N_FOLDS, shuffle=False)
    
    # Collect all fold predictions
    test_preds_all_folds = []
    
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train_full), start=1):
        print(f"\n--- Fold {fold}/{Config.N_FOLDS} ---")
        
        X_train = X_train_full[train_idx]
        y_train = y_train_full[train_idx]
        X_val = X_train_full[valid_idx]
        y_val = y_train_full[valid_idx]
        
        # Use QuantileTransformer for tree-based models
        scaler = QuantileTransformer(
            n_quantiles=min(1000, X_train.shape[0]), 
            output_distribution='uniform',
            random_state=42
        )
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(test_df[all_dofen_features].values)
        
        # Handle any NaN values
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=0.0)
        X_val = np.nan_to_num(X_val, nan=0.0, posinf=1.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=1.0, neginf=0.0)
        
        # Create dataloaders
        train_loader = get_dataloaders(X_train, y_train, hparams, device, shuffle=True)
        val_loader = get_dataloaders(X_val, y_val, hparams, device, shuffle=False)
        test_loader = get_dataloaders(X_test, None, hparams, device, shuffle=False)
        
        # Initialize model
        model = DOFEN(
            num_features=len(all_dofen_features),
            num_layers=hparams["num_layers"],
            trees_per_layer=hparams["trees_per_layer"],
            tree_depth=hparams["tree_depth"],
            feature_subset_ratio=hparams["feature_subset_ratio"],
            temperature=hparams["temperature"],
            use_sparse_selection=hparams["use_sparse_selection"],
            dropout=hparams["dropout"]
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.HuberLoss(delta=1.0, reduction='mean')
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        
        # Learning rate scheduler with warmup
        def lr_lambda(epoch):
            if epoch < hparams["warmup_epochs"]:
                return (epoch + 1) / hparams["warmup_epochs"]
            else:
                # Cosine annealing after warmup
                progress = (epoch - hparams["warmup_epochs"]) / (hparams["num_epochs"] - hparams["warmup_epochs"])
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        checkpoint_path = CHECKPOINTS_DIR / f"dofen_fold{fold}_model.pt"
        best_pearson = -np.inf
        patience_counter = 0
        
        # Training loop
        num_epochs = hparams["num_epochs"]
        
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Check for NaN
                if torch.isnan(loss):
                    print(f"NaN loss detected at epoch {epoch+1}, skipping batch")
                    continue
                
                loss.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), hparams["gradient_clip"])
                
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
            
            scheduler.step()
            
            running_loss = running_loss / len(train_loader.dataset)
            
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
                    
                    if not torch.isnan(loss):
                        val_loss += loss.item() * inputs.size(0)
                        preds.extend(outputs.cpu().numpy())
                        trues.extend(targets.cpu().numpy())
            
            val_loss /= len(val_loader.dataset)
            
            if len(preds) > 0:
                preds = np.array(preds)
                trues = np.array(trues)
                pearson_coef = pearsonr(preds, trues)[0]
            else:
                pearson_coef = -np.inf
            
            # Save best model
            if pearson_coef > best_pearson:
                best_pearson = pearson_coef
                torch.save(model.state_dict(), checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= hparams["patience"]:
                    break
        
        # Load best model and save OOF predictions
        model.load_state_dict(torch.load(checkpoint_path))
        model.eval()
        
        val_preds = []
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                val_preds.extend(outputs.cpu().numpy())
        
        oof_preds[valid_idx] = np.array(val_preds)
        
        # Make test predictions
        predictions = []
        with torch.no_grad():
            for inputs in test_loader:
                inputs = inputs[0].to(device)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy())
        
        test_preds_all_folds.append(np.array(predictions))
    
    # Average test predictions across folds
    test_predictions = np.mean(test_preds_all_folds, axis=0)
    
    # Post-process predictions
    pred_mean = train_df[Config.LABEL_COLUMN].mean()
    pred_std = train_df[Config.LABEL_COLUMN].std()
    test_predictions = np.clip(test_predictions, pred_mean - 4 * pred_std, pred_mean + 4 * pred_std)
    
    # Calculate overall OOF score
    oof_score = pearsonr(train_df[Config.LABEL_COLUMN], oof_preds)[0]
    print(f"\nDOFEN Overall OOF Pearson: {oof_score:.4f}")
    
    # Save predictions
    save_predictions("dofen", test_predictions, {
        "oof_score": oof_score,
        "n_features": len(all_dofen_features),
        "n_layers": hparams["num_layers"],
        "trees_per_layer": hparams["trees_per_layer"],
        "tree_depth": hparams["tree_depth"]
    })
    
    # Save OOF predictions
    np.save(PREDICTIONS_DIR / "dofen_oof.npy", oof_preds)
    
    return test_predictions, oof_preds

# =========================
# Ensemble Analysis Functions
# =========================
def optimize_ensemble_weights(predictions_dict, y_true, method='scipy'):
    """Optimize ensemble weights using various methods"""
    
    if method == 'scipy':
        # Use scipy optimization
        def objective(weights):
            weights = weights / weights.sum()
            ensemble = sum(w * pred for w, pred in zip(weights, predictions_dict.values()))
            return -pearsonr(y_true, ensemble)[0]
        
        n_models = len(predictions_dict)
        initial_weights = np.ones(n_models) / n_models
        bounds = [(0, 1) for _ in range(n_models)]
        
        result = minimize(objective, initial_weights, bounds=bounds, method='SLSQP')
        optimal_weights = result.x / result.x.sum()
        
    elif method == 'greedy':
        # Greedy forward selection
        model_names = list(predictions_dict.keys())
        selected_models = []
        remaining_models = model_names.copy()
        weights = []
        
        while remaining_models:
            best_score = -np.inf
            best_model = None
            best_weight = None
            
            for model in remaining_models:
                # Try different weights for the new model
                for w in np.linspace(0.1, 0.9, 9):
                    if selected_models:
                        # Combine with existing ensemble
                        existing_weight = 1 - w
                        existing_weights = np.array(weights) * existing_weight / sum(weights) if sum(weights) > 0 else np.array(weights)
                        
                        ensemble = sum(existing_weights[i] * predictions_dict[selected_models[i]] 
                                     for i in range(len(selected_models)))
                        ensemble += w * predictions_dict[model]
                    else:
                        ensemble = predictions_dict[model]
                    
                    score = pearsonr(y_true, ensemble)[0]
                    if score > best_score:
                        best_score = score
                        best_model = model
                        best_weight = w
            
            if best_model:
                selected_models.append(best_model)
                weights.append(best_weight)
                remaining_models.remove(best_model)
                
                # Normalize weights
                weights = list(np.array(weights) / sum(weights))
                
                print(f"Added {best_model} with weight {best_weight:.3f}, score: {best_score:.4f}")
        
        # Create weight dict
        optimal_weights = np.zeros(len(model_names))
        for i, model in enumerate(model_names):
            if model in selected_models:
                idx = selected_models.index(model)
                optimal_weights[i] = weights[idx]
    
    else:
        # Simple average
        n_models = len(predictions_dict)
        optimal_weights = np.ones(n_models) / n_models
    
    return optimal_weights

def analyze_model_correlations(predictions_dict, save_path=None):
    """Analyze correlations between model predictions"""
    model_names = list(predictions_dict.keys())
    n_models = len(model_names)
    
    # Create correlation matrix
    corr_matrix = np.zeros((n_models, n_models))
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names):
            corr_matrix[i, j] = np.corrcoef(predictions_dict[model1], 
                                           predictions_dict[model2])[0, 1]
    
    # Print correlation matrix
    print("\nğŸ“Š Model Correlation Matrix:")
    print("       ", "  ".join([m[:6] for m in model_names]))
    for i, model in enumerate(model_names):
        print(f"{model[:6]:6s}", " ".join([f"{corr_matrix[i, j]:6.3f}" for j in range(n_models)]))
    
    if save_path:
        np.save(save_path, corr_matrix)
    
    return corr_matrix

def create_ensemble_combinations(predictions_dict, y_true, submission_df):
    """Create and evaluate different ensemble combinations"""
    
    model_names = list(predictions_dict.keys())
    results = []
    
    # 1. Individual models
    print("\n=== Individual Model Scores ===")
    for model in model_names:
        score = pearsonr(y_true, predictions_dict[model])[0]
        results.append({
            'ensemble_name': model,
            'models': [model],
            'weights': {model: 1.0},
            'oof_score': score,
            'n_models': 1
        })
        print(f"{model}: {score:.4f}")
    
    # 2. All possible pairs
    print("\n=== Best Model Pairs ===")
    pair_results = []
    for i, model1 in enumerate(model_names):
        for j, model2 in enumerate(model_names[i+1:], i+1):
            # Try different weight combinations
            best_score = -np.inf
            best_w1 = 0.5
            
            for w1 in np.linspace(0.1, 0.9, 9):
                w2 = 1 - w1
                ensemble = w1 * predictions_dict[model1] + w2 * predictions_dict[model2]
                score = pearsonr(y_true, ensemble)[0]
                
                if score > best_score:
                    best_score = score
                    best_w1 = w1
            
            pair_results.append({
                'ensemble_name': f"{model1}_{model2}",
                'models': [model1, model2],
                'weights': {model1: best_w1, model2: 1-best_w1},
                'oof_score': best_score,
                'n_models': 2
            })
    
    # Sort and print top pairs
    pair_results.sort(key=lambda x: x['oof_score'], reverse=True)
    for result in pair_results[:5]:
        models = result['models']
        weights = result['weights']
        print(f"{models[0]}({weights[models[0]]:.2f}) + {models[1]}({weights[models[1]]:.2f}): {result['oof_score']:.4f}")
    
    results.extend(pair_results)
    
    # 3. Top 3, 4, 5 models
    print("\n=== Multi-Model Ensembles ===")
    
    # Sort models by individual performance
    model_scores = [(model, pearsonr(y_true, predictions_dict[model])[0]) for model in model_names]
    model_scores.sort(key=lambda x: x[1], reverse=True)
    top_models = [m[0] for m in model_scores]
    
    for n in [3, 4, 5, 6, 7, 8]:
        if n > len(model_names):
            break
            
        selected_models = top_models[:n]
        selected_preds = {m: predictions_dict[m] for m in selected_models}
        
        # Optimize weights
        optimal_weights = optimize_ensemble_weights(selected_preds, y_true, method='scipy')
        
        # Create ensemble
        ensemble = sum(w * selected_preds[m] for w, m in zip(optimal_weights, selected_models))
        score = pearsonr(y_true, ensemble)[0]
        
        weight_dict = {m: w for m, w in zip(selected_models, optimal_weights)}
        
        results.append({
            'ensemble_name': f"top_{n}_models",
            'models': selected_models,
            'weights': weight_dict,
            'oof_score': score,
            'n_models': n
        })
        
        print(f"Top {n} models: {score:.4f}")
        for m, w in weight_dict.items():
            if w > 0.01:
                print(f"  {m}: {w:.3f}")
    
    # 4. Custom ensembles based on model types
    print("\n=== Custom Ensembles ===")
    
    # Tree-based ensemble
    tree_models = ['xgboost', 'gandalf', 'simplified_gandalf', 'dofen']
    tree_models = [m for m in tree_models if m in model_names]
    if len(tree_models) > 1:
        tree_preds = {m: predictions_dict[m] for m in tree_models}
        tree_weights = optimize_ensemble_weights(tree_preds, y_true, method='scipy')
        tree_ensemble = sum(w * tree_preds[m] for w, m in zip(tree_weights, tree_models))
        tree_score = pearsonr(y_true, tree_ensemble)[0]
        
        results.append({
            'ensemble_name': 'tree_based_ensemble',
            'models': tree_models,
            'weights': {m: w for m, w in zip(tree_models, tree_weights)},
            'oof_score': tree_score,
            'n_models': len(tree_models)
        })
        print(f"Tree-based ensemble: {tree_score:.4f}")
    
    # Neural network ensemble
    nn_models = ['mlp', 'dcnv2', 'anam', 'tangos']
    nn_models = [m for m in nn_models if m in model_names]
    if len(nn_models) > 1:
        nn_preds = {m: predictions_dict[m] for m in nn_models}
        nn_weights = optimize_ensemble_weights(nn_preds, y_true, method='scipy')
        nn_ensemble = sum(w * nn_preds[m] for w, m in zip(nn_weights, nn_models))
        nn_score = pearsonr(y_true, nn_ensemble)[0]
        
        results.append({
            'ensemble_name': 'neural_network_ensemble',
            'models': nn_models,
            'weights': {m: w for m, w in zip(nn_models, nn_weights)},
            'oof_score': nn_score,
            'n_models': len(nn_models)
        })
        print(f"Neural network ensemble: {nn_score:.4f}")
    
    # 5. Greedy ensemble
    print("\n=== Greedy Forward Selection ===")
    greedy_weights = optimize_ensemble_weights(predictions_dict, y_true, method='greedy')
    greedy_ensemble = sum(w * predictions_dict[m] for w, m in zip(greedy_weights, model_names))
    greedy_score = pearsonr(y_true, greedy_ensemble)[0]
    
    results.append({
        'ensemble_name': 'greedy_selection',
        'models': model_names,
        'weights': {m: w for m, w in zip(model_names, greedy_weights)},
        'oof_score': greedy_score,
        'n_models': sum(1 for w in greedy_weights if w > 0.01)
    })
    
    # Sort all results by score
    results.sort(key=lambda x: x['oof_score'], reverse=True)
    
    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(ENSEMBLE_DIR / "ensemble_analysis.csv", index=False)
    
    # Save detailed results
    with open(ENSEMBLE_DIR / "ensemble_details.json", 'w') as f:
        json.dump(results, f, indent=4)
    
    return results

def create_submission(predictions, submission_df, name):
    """Create submission file"""
    submission = submission_df.copy()
    submission["prediction"] = predictions
    submission_path = SUBMISSIONS_DIR / f"{name}.csv"
    submission.to_csv(submission_path, index=False)
    return submission_path

# =========================
# Main Execution
# =========================
def main():
    """Main execution function"""
    
    # Load data
    print("\n=== Loading Data ===")
    train_df, test_df, submission_df = load_data()
    
    # Dictionary to store all predictions
    test_predictions = {}
    oof_predictions = {}
    
    # Train all models
    print("\n=== Training All Models ===")
    
    # XGBoost
    xgb_test, xgb_oof = train_xgboost(train_df, test_df)
    test_predictions['xgboost'] = xgb_test
    oof_predictions['xgboost'] = xgb_oof
    
    # MLP
    mlp_test, mlp_oof = train_mlp(train_df, test_df)
    test_predictions['mlp'] = mlp_test
    oof_predictions['mlp'] = mlp_oof
    
    # DCN V2
    dcnv2_test, dcnv2_oof = train_dcnv2(train_df, test_df)
    test_predictions['dcnv2'] = dcnv2_test
    oof_predictions['dcnv2'] = dcnv2_oof
    
    # GANDALF models
    gandalf_test, simplified_test, gandalf_oof, simplified_oof = train_gandalf(train_df, test_df)
    test_predictions['gandalf'] = gandalf_test
    test_predictions['simplified_gandalf'] = simplified_test
    oof_predictions['gandalf'] = gandalf_oof
    oof_predictions['simplified_gandalf'] = simplified_oof
    
    # ANAM
    anam_test, anam_oof = train_anam(train_df, test_df)
    test_predictions['anam'] = anam_test
    oof_predictions['anam'] = anam_oof
    
    # TANGOS
    tangos_test, tangos_oof = train_tangos(train_df, test_df)
    test_predictions['tangos'] = tangos_test
    oof_predictions['tangos'] = tangos_oof
    
    # DOFEN
    # dofen_test, dofen_oof = train_dofen(train_df, test_df)
    # test_predictions['dofen'] = dofen_test
    # oof_predictions['dofen'] = dofen_oof
    
    # Analyze model correlations
    print("\n=== Analyzing Model Correlations ===")
    corr_matrix = analyze_model_correlations(oof_predictions, 
                                           ENSEMBLE_DIR / "model_correlations.npy")
    
    # Create and evaluate ensemble combinations
    print("\n=== Creating Ensemble Combinations ===")
    y_true = train_df[Config.LABEL_COLUMN].values
    ensemble_results = create_ensemble_combinations(oof_predictions, y_true, submission_df)
    
    # Create submissions for top ensembles
    print("\n=== Creating Submissions for Top Ensembles ===")
    
    top_n = 5
    for i, result in enumerate(ensemble_results[:top_n]):
        ensemble_name = result['ensemble_name']
        models = result['models']
        weights = result['weights']
        oof_score = result['oof_score']
        
        print(f"\n{i+1}. {ensemble_name} (OOF: {oof_score:.4f})")
        
        # Create test ensemble
        test_ensemble = np.zeros_like(test_predictions[models[0]])
        for model in models:
            if weights[model] > 0.01:  # Only include models with significant weight
                test_ensemble += weights[model] * test_predictions[model]
                print(f"   {model}: {weights[model]:.3f}")
        
        # Create submission
        submission_path = create_submission(test_ensemble, submission_df, 
                                          f"ensemble_{i+1}_{ensemble_name}")
        print(f"   Saved to: {submission_path}")
    
    # Print final recommendations
    print("\n" + "="*60)
    print("ENSEMBLE RECOMMENDATIONS")
    print("="*60)
    
    print("\nğŸ�† TOP 3 ENSEMBLES TO SUBMIT:")
    for i, result in enumerate(ensemble_results[:3]):
        print(f"\n{i+1}. {result['ensemble_name']}")
        print(f"   OOF Score: {result['oof_score']:.4f}")
        print(f"   Models: {', '.join(result['models'])}")
        print(f"   Key weights:")
        for model, weight in result['weights'].items():
            if weight > 0.05:
                print(f"     - {model}: {weight:.1%}")
    
    print("\nğŸ“Š INSIGHTS:")
    print(f"â€¢ Best individual model: {ensemble_results[0]['ensemble_name']} ({ensemble_results[0]['oof_score']:.4f})")
    
    # Find best ensemble
    best_ensemble = max([r for r in ensemble_results if r['n_models'] > 1], 
                       key=lambda x: x['oof_score'])
    print(f"â€¢ Best ensemble: {best_ensemble['ensemble_name']} ({best_ensemble['oof_score']:.4f})")
    print(f"â€¢ Ensemble improvement: {(best_ensemble['oof_score'] - ensemble_results[0]['oof_score'])*100:.2f}%")
    
    # Model diversity
    avg_corr = (corr_matrix.sum() - len(corr_matrix)) / (len(corr_matrix) * (len(corr_matrix) - 1))
    print(f"â€¢ Average model correlation: {avg_corr:.3f}")
    
    print("\nâœ… Pipeline execution completed successfully!")
    print(f"ğŸ“‚ All results saved to: {ENSEMBLE_DIR}")

if __name__ == "__main__":
    main()

