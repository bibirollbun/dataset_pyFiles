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


# Enhanced NODE Implementation for Crypto Market Price Prediction
# Improvements include: advanced feature engineering, better architectures, 
# ensemble methods, and optimized training strategies

import random
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import mutual_info_regression, SelectKBest, f_regression
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import gc
from typing import List, Tuple, Dict, Optional, Union
import math
import pickle
from collections import defaultdict

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts, OneCycleLR
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from scipy.stats import pearsonr, spearmanr, kurtosis, skew
from scipy.special import erfinv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Enhanced Feature Configuration
# =========================
MARKET_FEATURES = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]

# Expanded proprietary features based on importance
PROPRIETARY_FEATURES = [
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
    "X888", "X421", "X333", "X817", "X586", "X292", "X344", "X532"
]

SELECTED_FEATURES = MARKET_FEATURES + PROPRIETARY_FEATURES + ["label"]

# =========================
# Advanced Feature Engineering
# =========================
def add_enhanced_market_features(df):
    """Add comprehensive market microstructure features with advanced calculations"""
    
    eps = 1e-10
    
    # Core microstructure features
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + eps)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + eps)
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + eps)
    
    # Advanced volume analysis
    df['log_volume'] = np.log1p(df['volume'])
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['cbrt_volume'] = np.cbrt(df['volume'])  # Cube root
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + eps)
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + eps)
    
    # Rolling statistics for volume (no future leakage)
    for window in [5, 10, 20]:
        df[f'volume_ma_{window}'] = df['volume'].rolling(window=window, min_periods=1).mean()
        df[f'volume_std_{window}'] = df['volume'].rolling(window=window, min_periods=1).std().fillna(0)
        df[f'volume_zscore_{window}'] = (df['volume'] - df[f'volume_ma_{window}']) / (df[f'volume_std_{window}'] + eps)
        
    # Liquidity metrics
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + eps)
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + eps)
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['bid_qty'] + df['ask_qty'] + eps)
    df['depth_exhaustion'] = df['volume'] / (df['total_depth'] + eps)
    
    # Price pressure and toxicity indicators
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + eps)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + eps)
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + eps)
    df['amihud_illiquidity'] = np.abs(df['order_flow_imbalance']) / (df['volume'] + eps)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * np.log1p(df['volume'])
    
    # Market stress and volatility proxies
    df['volume_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + eps)
    df['depth_volume_interaction'] = df['total_depth'] * np.log1p(df['volume'])
    df['flow_intensity'] = df['net_order_flow'] / (df['total_depth'] + eps)
    df['market_stress'] = df['volume'] / (df['total_depth'] + eps) * np.abs(df['order_flow_imbalance'])
    df['volatility_proxy'] = np.abs(df['net_order_flow']) / (df['total_depth'] + eps) * df['volume']
    
    # Order book shape metrics
    df['book_pressure'] = (df['bid_qty'] - df['ask_qty']) / (df['volume'] + eps)
    df['book_skew'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + eps) * np.sign(df['net_order_flow'])
    df['depth_ratio'] = df['bid_qty'] / (df['ask_qty'] + eps)
    df['inverse_depth_ratio'] = df['ask_qty'] / (df['bid_qty'] + eps)
    
    # Execution quality metrics
    df['execution_probability'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + eps)
    df['fill_rate'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + df['bid_qty'] + df['ask_qty'] + eps)
    df['trade_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['bid_qty'] + df['ask_qty'] + eps)
    
    # Information asymmetry measures
    df['pin_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + eps)
    df['information_share'] = df['net_order_flow'] / (df['volume'] + eps) * np.abs(df['depth_imbalance'])
    df['adverse_selection'] = np.abs(df['net_order_flow']) / (df['bid_qty'] + df['ask_qty'] + eps) * df['volume']
    
    # Complex interaction features
    df['volume_adjusted_imbalance'] = df['order_flow_imbalance'] * np.log1p(df['volume'])
    df['depth_adjusted_flow'] = df['net_order_flow'] / (np.log1p(df['total_depth']) + 1)
    df['liquidity_weighted_pressure'] = df['market_stress'] * df['liquidity_ratio']
    df['toxic_volume_ratio'] = df['flow_toxicity'] / (df['volume'] + eps)
    
    # Non-linear transformations
    df['volume_squared'] = df['volume'] ** 2
    df['volume_cubed'] = df['volume'] ** 3
    df['log_total_activity'] = np.log1p(df['buy_qty'] + df['sell_qty'] + df['volume'])
    df['sqrt_market_activity'] = np.sqrt(df['buy_qty'] + df['sell_qty'] + df['bid_qty'] + df['ask_qty'])
    
    # Ratios and normalized features
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + eps)
    df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + eps)
    df['volume_to_activity_ratio'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + df['bid_qty'] + df['ask_qty'] + eps)
    
    # Market efficiency indicators
    df['price_discovery'] = np.abs(df['net_order_flow']) / (df['total_depth'] + eps)
    df['information_ratio'] = df['flow_toxicity'] / (df['market_stress'] + eps)
    
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values with column-specific strategies
    for col in df.columns:
        if df[col].isna().any():
            if 'ratio' in col or 'imbalance' in col:
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna(df[col].median(), inplace=True)
    
    return df

def create_interaction_features(df, important_features, max_interactions=20):
    """Create interaction features between most important features"""
    interaction_features = []
    feature_pairs = []
    
    # Select top features for interactions
    for i in range(min(10, len(important_features))):
        for j in range(i+1, min(10, len(important_features))):
            if len(feature_pairs) >= max_interactions:
                break
            feature_pairs.append((important_features[i], important_features[j]))
    
    # Create interactions
    for feat1, feat2 in feature_pairs:
        if feat1 in df.columns and feat2 in df.columns:
            # Multiplication interaction
            interaction_name = f"{feat1}_x_{feat2}"
            df[interaction_name] = df[feat1] * df[feat2]
            interaction_features.append(interaction_name)
            
            # Ratio interaction
            ratio_name = f"{feat1}_div_{feat2}"
            df[ratio_name] = df[feat1] / (df[feat2] + 1e-10)
            interaction_features.append(ratio_name)
    
    return df, interaction_features

# =========================
# Advanced Data Transformation
# =========================
class AdvancedTransformer:
    """Advanced feature transformation with multiple methods"""
    
    def __init__(self, method='ensemble'):
        self.method = method
        self.transformers = {}
        self.weights = None
        
    def fit(self, X, y=None):
        if self.method == 'ensemble':
            # Fit multiple transformers
            self.transformers['quantile'] = QuantileTransformer(
                output_distribution='normal', 
                n_quantiles=min(1000, len(X)),
                random_state=42
            )
            self.transformers['robust'] = RobustScaler()
            self.transformers['standard'] = StandardScaler()
            
            for name, transformer in self.transformers.items():
                transformer.fit(X)
                
            # Learn optimal weights if y is provided
            if y is not None:
                self._learn_weights(X, y)
        else:
            # Single transformer
            if self.method == 'quantile':
                self.transformers[self.method] = QuantileTransformer(
                    output_distribution='normal',
                    n_quantiles=min(1000, len(X)),
                    random_state=42
                )
            elif self.method == 'robust':
                self.transformers[self.method] = RobustScaler()
            elif self.method == 'standard':
                self.transformers[self.method] = StandardScaler()
            
            self.transformers[self.method].fit(X)
            
        return self
    
    def transform(self, X):
        if self.method == 'ensemble':
            # Transform with all methods
            transformed_data = []
            
            for name, transformer in self.transformers.items():
                transformed_data.append(transformer.transform(X))
            
            # Weighted average
            if self.weights is not None:
                result = np.zeros_like(transformed_data[0])
                for i, data in enumerate(transformed_data):
                    result += self.weights[i] * data
                return result
            else:
                # Equal weights
                return np.mean(transformed_data, axis=0)
        else:
            return self.transformers[self.method].transform(X)
    
    def _learn_weights(self, X, y):
        """Learn optimal transformation weights"""
        from sklearn.linear_model import Ridge
        
        # Transform data with each method
        transformed_data = []
        for name, transformer in self.transformers.items():
            transformed_data.append(transformer.transform(X))
        
        # Stack transformations
        X_stacked = np.hstack(transformed_data)
        
        # Use Ridge regression to find weights
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_stacked, y)
        
        # Extract weights
        n_features = X.shape[1]
        n_transformers = len(self.transformers)
        
        weights = []
        for i in range(n_transformers):
            start_idx = i * n_features
            end_idx = (i + 1) * n_features
            weight = np.mean(np.abs(ridge.coef_[start_idx:end_idx]))
            weights.append(weight)
        
        # Normalize weights
        self.weights = np.array(weights) / np.sum(weights)

def apply_target_transformation(y, method='rank_gauss'):
    """Apply advanced target transformations"""
    if method == 'rank_gauss':
        # Rank-based inverse normal transformation
        ranks = (y.rank(method='average').values - 0.5) / len(y)
        transformed = np.sqrt(2) * erfinv(2 * ranks - 1)
        # Clip extreme values
        transformed = np.clip(transformed, -5, 5)
        return pd.Series(transformed, index=y.index)
    elif method == 'log':
        # Log transformation with shift
        y_shifted = y - y.min() + 1
        return np.log1p(y_shifted)
    elif method == 'box_cox':
        from scipy.stats import boxcox
        y_positive = y - y.min() + 1
        transformed, _ = boxcox(y_positive)
        return transformed
    else:
        return y

# =========================
# Enhanced NODE Components
# =========================
def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class EnhancedObliviousDecisionTree(nn.Module):
    """Enhanced Oblivious Decision Tree with advanced features"""
    
    def __init__(self, depth, num_features, temperature=1.0, use_entmax=False):
        super().__init__()
        self.depth = depth
        self.num_features = num_features
        self.temperature = temperature
        self.use_entmax = use_entmax
        self.num_leaves = 2 ** depth
        
        # Learnable temperature
        self.temperature_param = nn.Parameter(torch.tensor(temperature))
        
        # Split parameters with better initialization
        self.split_features = nn.Parameter(torch.randn(depth, num_features) * 0.01)
        self.split_thresholds = nn.Parameter(torch.zeros(depth))
        
        # Leaf parameters with residual connections
        self.leaf_values = nn.Parameter(torch.randn(self.num_leaves) * 0.01)
        self.leaf_embeddings = nn.Parameter(torch.randn(self.num_leaves, num_features) * 0.01)
        
        # Feature attention
        self.feature_attention = nn.Sequential(
            nn.Linear(num_features, num_features // 2),
            nn.ReLU(),
            nn.Linear(num_features // 2, num_features),
            nn.Sigmoid()
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(num_features, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Apply feature attention
        attention_weights = self.feature_attention(x)
        x_attended = x * attention_weights
        
        # Compute routing decisions
        routing_decisions = []
        
        for d in range(self.depth):
            if self.use_entmax:
                # Use entmax instead of softmax for sparse feature selection
                from entmax import entmax15
                feature_weights = entmax15(self.split_features[d] / self.temperature_param, dim=0)
            else:
                feature_weights = F.softmax(self.split_features[d] / self.temperature_param, dim=0)
            
            # Select features with attention
            selected_feature = torch.sum(x_attended * feature_weights.unsqueeze(0), dim=1)
            
            # Soft routing with learnable temperature
            decision = torch.sigmoid(
                (selected_feature - self.split_thresholds[d]) / self.temperature_param.clamp(min=0.01)
            )
            routing_decisions.append(decision)
        
        # Compute leaf probabilities with numerical stability
        leaf_log_probs = torch.zeros(batch_size, self.num_leaves, device=x.device)
        
        for leaf_idx in range(self.num_leaves):
            log_prob = 0.0
            for d in range(self.depth):
                if (leaf_idx >> (self.depth - d - 1)) & 1 == 0:
                    log_prob = log_prob + torch.log(1 - routing_decisions[d] + 1e-10)
                else:
                    log_prob = log_prob + torch.log(routing_decisions[d] + 1e-10)
            leaf_log_probs[:, leaf_idx] = log_prob
        
        # Convert to probabilities
        leaf_probs = torch.exp(leaf_log_probs - torch.logsumexp(leaf_log_probs, dim=1, keepdim=True))
        
        # Compute predictions with leaf embeddings
        leaf_features = torch.sum(
            leaf_probs.unsqueeze(2) * self.leaf_embeddings.unsqueeze(0), 
            dim=1
        )
        
        # Final prediction with gating
        base_predictions = torch.sum(leaf_probs * self.leaf_values.unsqueeze(0), dim=1)
        gate_value = self.gate(leaf_features).squeeze()
        
        predictions = base_predictions * gate_value + (1 - gate_value) * x.mean(dim=1)
        
        return predictions, leaf_probs, attention_weights

class ResidualBlock(nn.Module):
    """Residual block for deep feature processing"""
    
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim)
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return x + self.dropout(self.layers(x))

class EnhancedNODE(nn.Module):
    """Enhanced Neural Oblivious Decision Ensembles"""
    
    def __init__(self, config):
        super().__init__()
        
        self.num_features = config['num_features']
        self.num_trees = config['num_trees']
        self.tree_depth = config['tree_depth']
        self.temperature = config['temperature']
        
        # Feature preprocessing layers
        self.feature_processor = nn.Sequential(
            nn.Linear(self.num_features, config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            ResidualBlock(config['hidden_dim'], config['dropout']),
            ResidualBlock(config['hidden_dim'], config['dropout']),
            nn.Linear(config['hidden_dim'], self.num_features),
            nn.LayerNorm(self.num_features)
        )
        
        # Multi-scale tree ensemble
        self.trees = nn.ModuleList()
        tree_depths = config.get('tree_depths', [self.tree_depth])
        
        for i in range(self.num_trees):
            # Vary tree depths for diversity
            depth = tree_depths[i % len(tree_depths)]
            self.trees.append(
                EnhancedObliviousDecisionTree(
                    depth=depth,
                    num_features=self.num_features,
                    temperature=self.temperature,
                    use_entmax=config.get('use_entmax', False)
                )
            )
        
        # Tree attention mechanism
        self.tree_attention = nn.Sequential(
            nn.Linear(self.num_features, config['hidden_dim']),
            nn.ReLU(),
            nn.Linear(config['hidden_dim'], self.num_trees),
            nn.Softmax(dim=1)
        )
        
        # Deep output layers
        self.output_layers = nn.Sequential(
            nn.Linear(self.num_trees + self.num_features, config['hidden_dim']),
            nn.LayerNorm(config['hidden_dim']),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            ResidualBlock(config['hidden_dim'], config['dropout']),
            nn.Linear(config['hidden_dim'], config['hidden_dim'] // 2),
            nn.LayerNorm(config['hidden_dim'] // 2),
            nn.GELU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['hidden_dim'] // 2, 1)
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Custom weight initialization"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Process features
        x_processed = self.feature_processor(x)
        x_combined = x + 0.1 * x_processed  # Residual connection with small weight
        
        # Get tree outputs
        tree_outputs = []
        tree_attentions = []
        
        for tree in self.trees:
            output, leaf_probs, attention = tree(x_combined)
            tree_outputs.append(output)
            tree_attentions.append(attention.mean(dim=1))  # Average attention
        
        tree_outputs = torch.stack(tree_outputs, dim=1)
        
        # Apply tree attention
        tree_weights = self.tree_attention(x_combined)
        weighted_trees = (tree_outputs * tree_weights).sum(dim=1, keepdim=True)
        
        # Combine with original features
        combined_features = torch.cat([tree_outputs, x_combined], dim=1)
        
        # Final prediction
        final_output = self.output_layers(combined_features)
        
        return final_output + 0.1 * weighted_trees  # Residual from trees

class StackedNODE(nn.Module):
    """Stacked NODE models for hierarchical learning"""
    
    def __init__(self, base_config, num_layers=3):
        super().__init__()
        
        self.num_layers = num_layers
        self.models = nn.ModuleList()
        
        # Create stacked models with decreasing complexity
        for i in range(num_layers):
            config = base_config.copy()
            config['num_trees'] = max(20, config['num_trees'] // (i + 1))
            config['tree_depth'] = max(3, config['tree_depth'] - i)
            
            self.models.append(EnhancedNODE(config))
        
        # Combination weights
        self.layer_weights = nn.Parameter(torch.ones(num_layers))
        
        # Final projection
        self.final_projection = nn.Linear(num_layers, 1)
        
    def forward(self, x):
        outputs = []
        
        for model in self.models:
            output = model(x)
            outputs.append(output)
        
        # Stack outputs
        stacked = torch.cat(outputs, dim=1)
        
        # Apply layer weights
        weights = F.softmax(self.layer_weights, dim=0)
        weighted_output = (stacked * weights.unsqueeze(0)).sum(dim=1, keepdim=True)
        
        # Alternative: use learned projection
        projected_output = self.final_projection(stacked)
        
        # Combine both approaches
        return 0.7 * weighted_output + 0.3 * projected_output

# =========================
# Advanced Training Utilities
# =========================
class FocalLoss(nn.Module):
    """Focal loss for handling imbalanced predictions"""
    
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = F.mse_loss(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss
        return focal_loss.mean()

class AdaptiveLoss(nn.Module):
    """Adaptive loss that combines multiple objectives"""
    
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        self.huber = nn.HuberLoss(delta=1.0)
        self.focal = FocalLoss()
        
        # Learnable weights
        self.weights = nn.Parameter(torch.ones(3))
        
    def forward(self, inputs, targets):
        # Compute individual losses
        mse_loss = self.mse(inputs, targets)
        huber_loss = self.huber(inputs, targets)
        focal_loss = self.focal(inputs, targets)
        
        # Combine with learnable weights
        weights = F.softmax(self.weights, dim=0)
        combined_loss = (
            weights[0] * mse_loss + 
            weights[1] * huber_loss + 
            weights[2] * focal_loss
        )
        
        return combined_loss

def create_weighted_sampler(y, num_bins=10):
    """Create weighted sampler for balanced training"""
    # Bin the target values
    y_binned = pd.qcut(y, q=num_bins, labels=False, duplicates='drop')
    
    # Calculate weights
    bin_counts = np.bincount(y_binned)
    weights = 1.0 / (bin_counts[y_binned] + 1)
    weights = weights / weights.sum() * len(weights)
    
    return torch.utils.data.WeightedRandomSampler(
        weights=torch.FloatTensor(weights),
        num_samples=len(weights),
        replacement=True
    )

def train_enhanced_node(model, train_loader, val_loader, config, device):
    """Enhanced training with advanced techniques"""
    
    # Loss function
    if config.get('use_adaptive_loss', True):
        criterion = AdaptiveLoss().to(device)
    else:
        criterion = nn.HuberLoss(delta=config['huber_delta'])
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-8,
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    if config.get('scheduler_type', 'cosine') == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, 
            T_0=10, 
            T_mult=2,
            eta_min=config['learning_rate'] * 0.01
        )
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=config['learning_rate'] * 10,
            epochs=config['num_epochs'],
            steps_per_epoch=len(train_loader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
    
    # Training history
    history = defaultdict(list)
    best_val_score = -np.inf
    patience_counter = 0
    
    # Temperature annealing
    temperature_schedule = lambda epoch: max(0.1, config['temperature'] * (0.95 ** epoch))
    
    for epoch in range(config['num_epochs']):
        # Update temperature
        current_temperature = temperature_schedule(epoch)
        for module in model.modules():
            if hasattr(module, 'temperature_param'):
                module.temperature_param.data.fill_(current_temperature)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_preds = []
        train_targets = []
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Data augmentation
            if config.get('use_mixup', True) and np.random.random() < 0.5:
                # Mixup augmentation
                lam = np.random.beta(0.2, 0.2)
                index = torch.randperm(inputs.size(0)).to(device)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                mixed_targets = lam * targets + (1 - lam) * targets[index]
                
                outputs = model(mixed_inputs)
                loss = criterion(outputs, mixed_targets)
            else:
                # Standard forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # Add regularization
            if config.get('l2_penalty', 0) > 0:
                l2_reg = sum(p.pow(2.0).sum() for p in model.parameters())
                loss = loss + config['l2_penalty'] * l2_reg
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            
            # Update scheduler
            if config.get('scheduler_type') == 'onecycle':
                scheduler.step()
            
            train_loss += loss.item()
            train_preds.extend(outputs.detach().cpu().numpy().flatten())
            train_targets.extend(targets.cpu().numpy().flatten())
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'temp': f'{current_temperature:.3f}',
                'lr': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        # Calculate training metrics
        avg_train_loss = train_loss / len(train_loader)
        train_pearson = pearsonr(train_targets, train_preds)[0]
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_pearson = pearsonr(val_targets, val_preds)[0]
        val_spearman = spearmanr(val_targets, val_preds)[0]
        
        # Update scheduler
        if config.get('scheduler_type') == 'cosine':
            scheduler.step()
        
        # Log metrics
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_pearson'].append(train_pearson)
        history['val_pearson'].append(val_pearson)
        history['val_spearman'].append(val_spearman)
        
        print(f"\nEpoch {epoch+1}:")
        print(f"  Train - Loss: {avg_train_loss:.4f}, Pearson: {train_pearson:.4f}")
        print(f"  Valid - Loss: {avg_val_loss:.4f}, Pearson: {val_pearson:.4f}, Spearman: {val_spearman:.4f}")
        
        # Save best model
        if val_pearson > best_val_score:
            best_val_score = val_pearson
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': config,
                'epoch': epoch,
                'best_score': best_val_score
            }, f"best_enhanced_node_{config.get('model_id', 'default')}.pt")
            print(f"  ✅ New best model saved! Score: {best_val_score:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    checkpoint = torch.load(
        f"best_enhanced_node_{config.get('model_id', 'default')}.pt",
        weights_only=False,
        map_location=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model, history, best_val_score

# =========================
# Feature Selection and Engineering
# =========================
def advanced_feature_selection(X, y, feature_names, method='multi_stage', n_features=150):
    """Multi-stage feature selection combining multiple methods"""
    
    selected_features = set()
    importance_scores = {}
    
    # Stage 1: Mutual Information
    print("Stage 1: Mutual Information...")
    mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=10)
    mi_df = pd.DataFrame({
        'feature': feature_names,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    # Select top MI features
    top_mi_features = mi_df.head(n_features // 3)['feature'].tolist()
    selected_features.update(top_mi_features)
    
    for idx, row in mi_df.iterrows():
        importance_scores[row['feature']] = row['mi_score']
    
    # Stage 2: Random Forest Importance
    print("Stage 2: Random Forest Importance...")
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X, y)
    
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # Select top RF features
    top_rf_features = rf_importance.head(n_features // 3)['feature'].tolist()
    selected_features.update(top_rf_features)
    
    # Stage 3: Correlation with target
    print("Stage 3: Correlation Analysis...")
    correlations = []
    for i, feature in enumerate(feature_names):
        corr = abs(pearsonr(X[:, i], y)[0])
        correlations.append(corr)
        if feature not in importance_scores:
            importance_scores[feature] = 0
        importance_scores[feature] += corr
    
    corr_df = pd.DataFrame({
        'feature': feature_names,
        'correlation': correlations
    }).sort_values('correlation', ascending=False)
    
    # Select top correlation features
    top_corr_features = corr_df.head(n_features // 3)['feature'].tolist()
    selected_features.update(top_corr_features)
    
    # Stage 4: F-statistic
    print("Stage 4: F-statistic Selection...")
    selector = SelectKBest(f_regression, k=min(n_features, X.shape[1]))
    selector.fit(X, y)
    
    f_scores = pd.DataFrame({
        'feature': feature_names,
        'f_score': selector.scores_
    }).sort_values('f_score', ascending=False)
    
    # Add remaining features based on F-score
    for idx, row in f_scores.iterrows():
        if len(selected_features) < n_features:
            selected_features.add(row['feature'])
    
    # Create final importance dataframe
    final_importance = pd.DataFrame([
        {'feature': feat, 'importance': importance_scores.get(feat, 0)}
        for feat in selected_features
    ]).sort_values('importance', ascending=False)
    
    selected_feature_list = final_importance['feature'].tolist()[:n_features]
    
    print(f"Selected {len(selected_feature_list)} features")
    
    return selected_feature_list, final_importance

# =========================
# Main Enhanced Execution
# =========================
def main():
    print("=== Enhanced NODE for Crypto Market Prediction ===")
    print(f"Device: {device}")
    
    # Set random seed
    set_seed(42)
    
    # Load data
    print("\nLoading data...")
    train = pl.scan_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
    train = train.select(SELECTED_FEATURES).collect().to_pandas()
    
    print(f"Initial shape: {train.shape}")
    
    # Use more recent data for better relevance
    train_size = int(0.9 * len(train))
    train = train.iloc[-train_size:].reset_index(drop=True)
    print(f"Using last {train_size} samples")
    
    # Add enhanced features
    print("\nAdding enhanced market features...")
    train = add_enhanced_market_features(train)
    
    # Create interaction features with top proprietary features
    print("Creating interaction features...")
    top_proprietary = ['X863', 'X856', 'X598', 'X862', 'X385']
    train, interaction_features = create_interaction_features(train, top_proprietary)
    
    # Prepare features and target
    all_features = [col for col in train.columns if col != 'label']
    print(f"Total features: {len(all_features)}")
    
    # Target transformation
    print("\nApplying target transformation...")
    y_original = train['label'].copy()
    train['label'] = apply_target_transformation(train['label'], method='rank_gauss')
    
    # Split data
    split_idx = int(0.85 * len(train))
    train_data = train[:split_idx].copy()
    val_data = train[split_idx:].copy()
    
    X_train = train_data.drop('label', axis=1)
    y_train = train_data['label']
    X_val = val_data.drop('label', axis=1)
    y_val = val_data['label']
    
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    
    # Advanced feature selection
    print("\nPerforming advanced feature selection...")
    selected_features, feature_importance = advanced_feature_selection(
        X_train[all_features].values,
        y_train.values,
        all_features,
        method='multi_stage',
        n_features=200
    )
    
    # Filter data
    X_train_selected = X_train[selected_features].values
    X_val_selected = X_val[selected_features].values
    
    # Advanced transformation
    print("\nApplying advanced transformations...")
    transformer = AdvancedTransformer(method='ensemble')
    X_train_transformed = transformer.fit(X_train_selected, y_train).transform(X_train_selected)
    X_val_transformed = transformer.transform(X_val_selected)
    
    print(f"Final train shape: {X_train_transformed.shape}")
    print(f"Final validation shape: {X_val_transformed.shape}")
    
    # Create data loaders with weighted sampling
    print("\nCreating data loaders...")
    sampler = create_weighted_sampler(y_train, num_bins=20)
    
    train_dataset = TensorDataset(
        torch.tensor(X_train_transformed, dtype=torch.float32),
        torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_transformed, dtype=torch.float32),
        torch.tensor(y_val.values, dtype=torch.float32).unsqueeze(1)
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=512,
        sampler=sampler,
        num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=1024,
        shuffle=False,
        num_workers=0
    )
    
    # Model configurations
    base_config = {
        'num_features': X_train_transformed.shape[1],
        'num_trees': 150,
        'tree_depth': 6,
        'tree_depths': [4, 5, 6, 7, 8],  # Multi-scale trees
        'temperature': 1.0,
        'hidden_dim': 384,
        'dropout': 0.2,
        'use_entmax': False,
        'num_epochs': 100,
        'batch_size': 512,
        'learning_rate': 0.0003,
        'weight_decay': 0.01,
        'huber_delta': 1.0,
        'grad_clip': 1.0,
        'patience': 15,
        'l2_penalty': 0.0001,
        'use_adaptive_loss': True,
        'use_mixup': True,
        'scheduler_type': 'cosine'
    }
    
    # Train multiple models
    print("\n=== Training Enhanced NODE Models ===")
    
    ensemble_models = []
    model_configs = [
        {
            'model_id': 'enhanced_node_v1',
            'model_type': 'enhanced',
            'seed': 42
        },
        {
            'model_id': 'stacked_node_v1',
            'model_type': 'stacked',
            'seed': 1337,
            'num_trees': 100
        },
        {
            'model_id': 'enhanced_node_v2',
            'model_type': 'enhanced',
            'seed': 2468,
            'tree_depth': 7,
            'num_trees': 120
        }
    ]
    
    for config_update in model_configs:
        print(f"\n--- Training {config_update['model_id']} ---")
        
        # Update config
        model_config = base_config.copy()
        model_config.update(config_update)
        
        # Set seed
        set_seed(model_config['seed'])
        
        # Create model
        if config_update['model_type'] == 'stacked':
            model = StackedNODE(model_config, num_layers=3).to(device)
        else:
            model = EnhancedNODE(model_config).to(device)
        
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train model
        model, history, best_score = train_enhanced_node(
            model, train_loader, val_loader, model_config, device
        )
        
        print(f"{config_update['model_id']} best validation Pearson: {best_score:.4f}")
        
        # Store model info
        ensemble_models.append({
            'model': model,
            'transformer': transformer,
            'features': selected_features,
            'config': model_config,
            'score': best_score
        })
    
    # Test predictions
    print("\n=== Making Test Predictions ===")
    
    # Load test data
    test = pl.scan_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
    test_features = [f for f in SELECTED_FEATURES if f != "label"]
    test = test.select(test_features).collect().to_pandas()
    
    # Add features
    test = add_enhanced_market_features(test)
    test, _ = create_interaction_features(test, top_proprietary)
    
    # Make ensemble predictions
    all_predictions = []
    weights = []
    
    for model_info in ensemble_models:
        model = model_info['model']
        transformer = model_info['transformer']
        features = model_info['features']
        score = model_info['score']
        
        # Prepare test data
        X_test_selected = test[features].values
        X_test_transformed = transformer.transform(X_test_selected)
        
        # Make predictions
        model.eval()
        test_dataset = TensorDataset(torch.tensor(X_test_transformed, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for (inputs,) in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy().flatten())
        
        all_predictions.append(np.array(predictions))
        weights.append(score)
    
    # Weighted ensemble based on validation scores
    weights = np.array(weights)
    weights = weights / weights.sum()
    
    print(f"\nEnsemble weights: {weights}")
    
    final_predictions = np.zeros_like(all_predictions[0])
    for pred, weight in zip(all_predictions, weights):
        final_predictions += weight * pred
    
    # Inverse transform predictions (approximate)
    # Since we used rank_gauss, we need to map back to original distribution
    train_ranks = (y_original.rank(method='average').values - 0.5) / len(y_original)
    
    # Create mapping from transformed to original
    sorted_indices = np.argsort(train['label'].values[:len(y_original)])
    sorted_original = y_original.values[sorted_indices]
    
    # Map predictions back
    pred_ranks = (final_predictions.argsort().argsort() + 0.5) / len(final_predictions)
    final_predictions_mapped = np.interp(pred_ranks, np.linspace(0, 1, len(sorted_original)), sorted_original)
    
    # Post-processing
    pred_mean = y_original.mean()
    pred_std = y_original.std()
    final_predictions_mapped = np.clip(
        final_predictions_mapped,
        pred_mean - 4 * pred_std,
        pred_mean + 4 * pred_std
    )
    
    # Create submission
    submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
    submission["prediction"] = final_predictions_mapped
    submission.to_csv("submission_enhanced_node.csv", index=False)
    
    # Display results
    print("\n=== Final Results ===")
    print(f"Ensemble size: {len(ensemble_models)} models")
    print(f"Features used: {len(selected_features)}")
    print(f"\nPrediction statistics:")
    print(f"  Mean: {final_predictions_mapped.mean():.6f}")
    print(f"  Std: {final_predictions_mapped.std():.6f}")
    print(f"  Min: {final_predictions_mapped.min():.6f}")
    print(f"  Max: {final_predictions_mapped.max():.6f}")
    print(f"  Skewness: {skew(final_predictions_mapped):.4f}")
    print(f"  Kurtosis: {kurtosis(final_predictions_mapped):.4f}")
    
    # Display feature importance
    print("\nTop 20 most important features:")
    print(feature_importance.head(20))
    
    print("\n✅ Enhanced NODE model training complete!")
    print("Submission saved as 'submission_enhanced_node.csv'")

if __name__ == "__main__":
    main()

