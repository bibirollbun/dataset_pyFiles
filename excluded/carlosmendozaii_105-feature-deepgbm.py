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





!pip install optuna
import random
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import mutual_info_regression
from tqdm import tqdm
import gc
from typing import List, Tuple, Dict
import pickle
import json
from datetime import datetime

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

import xgboost as xgb
import lightgbm as lgb

from scipy.stats import pearsonr, spearmanr
from scipy.stats import kurtosis, skew

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Feature Configuration
# =========================
MARKET_FEATURES = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]

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
# Feature Engineering Functions
# =========================
def add_advanced_market_features(df):
    """Add comprehensive market microstructure features for crypto markets"""
    
    # Core microstructure features
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    
    # Advanced volume analysis
    df['log_volume'] = np.log1p(df['volume'])
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['volume_intensity'] = df['volume'] / (df['volume'].rolling(window=10, min_periods=1).mean() + 1e-10)
    
    # Liquidity metrics
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['depth_exhaustion'] = df['volume'] / (df['total_depth'] + 1e-10)
    
    # Price pressure and toxicity indicators
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['amihud_illiquidity'] = np.abs(df['order_flow_imbalance']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * np.log1p(df['volume'])
    
    # Market stress and volatility proxies
    df['volume_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['depth_volume_interaction'] = df['total_depth'] * np.log1p(df['volume'])
    df['flow_intensity'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['volatility_proxy'] = np.abs(df['net_order_flow']) / (df['total_depth'] + 1e-10) * df['volume']
    
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

def compute_rolling_features(df, windows=[5, 10, 20]):
    """Compute rolling statistics for key features"""
    key_features = ['volume', 'net_order_flow', 'total_depth', 'order_flow_imbalance']
    
    for feature in key_features:
        if feature in df.columns:
            for window in windows:
                df[f'{feature}_ma_{window}'] = df[feature].rolling(window=window, min_periods=1).mean()
                df[f'{feature}_std_{window}'] = df[feature].rolling(window=window, min_periods=1).std().fillna(0)
                df[f'{feature}_skew_{window}'] = df[feature].rolling(window=window, min_periods=3).skew().fillna(0)
                df[f'{feature}_rel_ma_{window}'] = df[feature] / (df[f'{feature}_ma_{window}'] + 1e-10)
    
    return df

# =========================
# Data Transformation Functions
# =========================
def apply_advanced_transformations(X, method='quantile'):
    """Apply advanced transformations to features"""
    if method == 'quantile':
        transformer = QuantileTransformer(output_distribution='normal', random_state=42)
    elif method == 'robust':
        transformer = RobustScaler()
    elif method == 'standard':
        transformer = StandardScaler()
    else:
        raise ValueError(f"Unknown transformation method: {method}")
    
    return transformer.fit_transform(X), transformer

def compute_feature_importance(X, y, feature_names, n_samples=50000):
    """Compute feature importance using mutual information"""
    print(f"Computing feature importance with {n_samples} samples...")
    
    if len(X) <= n_samples:
        X_subset = X
        y_subset = y
    else:
        indices = np.random.choice(len(X), n_samples, replace=False)
        X_subset = X[indices]
        y_subset = y.iloc[indices] if hasattr(y, 'iloc') else y[indices]
    
    mi_scores = mutual_info_regression(X_subset, y_subset, random_state=42, n_neighbors=5)
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mi_score': mi_scores
    }).sort_values('mi_score', ascending=False)
    
    return importance_df

# =========================
# Fixed GBDT Feature Extractor
# =========================
class GBDTFeatureExtractor:
    """Extract features from GBDT models for DeepGBM"""
    def __init__(self, config):
        self.config = config
        self.models = []
        self.max_leaves_per_tree = {}
        
    def train_gbdt_models(self, X_train, y_train, X_val, y_val):
        """Train multiple GBDT models and extract features"""
        
        # XGBoost model
        xgb_params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
            'max_depth': self.config.get('xgb_max_depth', 6),
            'learning_rate': self.config.get('xgb_learning_rate', 0.05),
            'n_estimators': self.config.get('xgb_n_estimators', 200),
            'subsample': self.config.get('xgb_subsample', 0.8),
            'colsample_bytree': self.config.get('xgb_colsample', 0.8),
            'reg_alpha': self.config.get('xgb_reg_alpha', 0.1),
            'reg_lambda': self.config.get('xgb_reg_lambda', 0.1),
            'random_state': 42,
            'n_jobs': -1
        }
        
        print("Training XGBoost model...")
        xgb_model = xgb.XGBRegressor(**xgb_params)
        xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )
        
        # Store max leaves information
        booster = xgb_model.get_booster()
        trees_df = booster.trees_to_dataframe()
        self.max_leaves_per_tree['xgboost'] = int(trees_df['Node'].max()) + 1
        
        self.models.append(('xgboost', xgb_model))
        
        # LightGBM model
        lgb_params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': self.config.get('lgb_max_depth', 6),
            'learning_rate': self.config.get('lgb_learning_rate', 0.05),
            'n_estimators': self.config.get('lgb_n_estimators', 200),
            'num_leaves': self.config.get('lgb_num_leaves', 31),
            'subsample': self.config.get('lgb_subsample', 0.8),
            'colsample_bytree': self.config.get('lgb_colsample', 0.8),
            'reg_alpha': self.config.get('lgb_reg_alpha', 0.1),
            'reg_lambda': self.config.get('lgb_reg_lambda', 0.1),
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        
        print("Training LightGBM model...")
        lgb_model = lgb.LGBMRegressor(**lgb_params)
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        self.models.append(('lightgbm', lgb_model))
        
    def extract_features(self, X):
        """Extract features from trained GBDT models"""
        features = []
        
        for name, model in self.models:
            if name == 'xgboost':
                # Get leaf indices
                leaf_features = model.apply(X)
                n_trees = leaf_features.shape[1]
                
                # Use actual max leaves instead of hardcoded value
                max_leaves = self.max_leaves_per_tree.get('xgboost', 64)
                
                # Create one-hot encoded features
                leaf_features_encoded = np.zeros((X.shape[0], min(n_trees * max_leaves, self.config.get('gbdt_output_dim', 100))))
                
                for i in range(X.shape[0]):
                    for j in range(n_trees):
                        if j * max_leaves + leaf_features[i, j] < leaf_features_encoded.shape[1]:
                            leaf_features_encoded[i, j * max_leaves + leaf_features[i, j]] = 1
                
                features.append(leaf_features_encoded)
                
            elif name == 'lightgbm':
                # Get predictions from individual trees
                tree_preds = []
                n_trees_to_use = min(model.n_estimators_, self.config.get('gbdt_output_dim', 100) // 2)
                
                for i in range(n_trees_to_use):
                    tree_pred = model.predict(X, num_iteration=i+1) - (model.predict(X, num_iteration=i) if i > 0 else 0)
                    tree_preds.append(tree_pred)
                
                if tree_preds:
                    tree_features = np.column_stack(tree_preds)
                    features.append(tree_features)
        
        # Concatenate all features
        if features:
            return np.hstack(features)
        else:
            return np.zeros((X.shape[0], self.config.get('gbdt_output_dim', 100)))

# =========================
# DeepGBM Architecture Components
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

class FeatureTransformer(nn.Module):
    """Feature transformation layer for DeepGBM"""
    def __init__(self, input_dim, output_dim, dropout=0.1):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.dropout(self.activation(self.norm(self.linear(x))))

class CatNN(nn.Module):
    """Categorical-aware Neural Network component of DeepGBM"""
    def __init__(self, config):
        super().__init__()
        
        self.input_dim = config['input_dim']
        self.embedding_dim = config['embedding_dim']
        self.hidden_dims = config['hidden_dims']
        self.output_dim = config['output_dim']
        
        # Feature embedding layers
        self.feature_embedder = nn.Sequential(
            FeatureTransformer(self.input_dim, self.embedding_dim, config['dropout']),
            FeatureTransformer(self.embedding_dim, self.embedding_dim, config['dropout'])
        )
        
        # Deep layers
        layers = []
        prev_dim = self.embedding_dim
        
        for hidden_dim in self.hidden_dims:
            layers.append(FeatureTransformer(prev_dim, hidden_dim, config['dropout']))
            prev_dim = hidden_dim
        
        self.deep_layers = nn.Sequential(*layers)
        
        # Output projection
        self.output_projection = nn.Linear(prev_dim, self.output_dim)
        
        # Attention mechanism for feature interactions
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=self.embedding_dim,
            num_heads=config['num_attention_heads'],
            dropout=config['attention_dropout'],
            batch_first=True
        )
        
        # Gating mechanism
        self.gate = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(self.embedding_dim // 2, self.embedding_dim),
            nn.Sigmoid()
        )
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x):
        # Initial feature embedding
        embedded = self.feature_embedder(x)
        
        # Self-attention for feature interactions
        attended, _ = self.feature_attention(
            embedded.unsqueeze(1), 
            embedded.unsqueeze(1), 
            embedded.unsqueeze(1)
        )
        attended = attended.squeeze(1)
        
        # Gating mechanism
        gate_values = self.gate(embedded)
        features = gate_values * attended + (1 - gate_values) * embedded
        
        # Deep transformation
        deep_features = self.deep_layers(features)
        
        # Output
        output = self.output_projection(deep_features)
        
        return output, deep_features

class DeepGBM(nn.Module):
    """Deep Gradient Boosting Machine combining NN and GBDT"""
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        self.use_gbdt_features = config.get('use_gbdt_features', True)
        
        # CatNN component
        catnn_config = {
            'input_dim': config['input_dim'],
            'embedding_dim': config['catnn_embedding_dim'],
            'hidden_dims': config['catnn_hidden_dims'],
            'output_dim': config['catnn_output_dim'],
            'dropout': config['dropout'],
            'num_attention_heads': config.get('num_attention_heads', 4),
            'attention_dropout': config.get('attention_dropout', 0.1)
        }
        self.catnn = CatNN(catnn_config)
        
        # Fusion layers for combining CatNN and GBDT outputs
        if self.use_gbdt_features:
            fusion_input_dim = config['catnn_output_dim'] + config.get('gbdt_output_dim', 100)
        else:
            fusion_input_dim = config['catnn_output_dim']
            
        self.fusion_layers = nn.Sequential(
            nn.Linear(fusion_input_dim, config['fusion_hidden_dim']),
            nn.LayerNorm(config['fusion_hidden_dim']),
            nn.GELU(),
            nn.Dropout(config['fusion_dropout']),
            nn.Linear(config['fusion_hidden_dim'], config['fusion_hidden_dim'] // 2),
            nn.LayerNorm(config['fusion_hidden_dim'] // 2),
            nn.GELU(),
            nn.Dropout(config['fusion_dropout']),
            nn.Linear(config['fusion_hidden_dim'] // 2, 1)
        )
        
        # Residual connection
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.fusion_layers.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
                    
    def forward(self, x, gbdt_features=None):
        # Get CatNN outputs
        catnn_output, catnn_features = self.catnn(x)
        
        # Combine with GBDT features if available
        if self.use_gbdt_features and gbdt_features is not None:
            combined_features = torch.cat([catnn_features, gbdt_features], dim=-1)
        else:
            combined_features = catnn_features
        
        # Fusion
        fusion_output = self.fusion_layers(combined_features)
        
        # Residual connection
        final_output = fusion_output + self.residual_weight * catnn_output
        
        return final_output

# =========================
# Training Function for Single DeepGBM Model
# =========================
def train_single_deepgbm(config, X_train_transformed, y_train, X_val_transformed, y_val,
                        X_train_raw, X_val_raw, device, verbose=True):
    """Train a single DeepGBM model with given configuration"""
    
    set_seed(config.get('seed', 42))
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_transformed, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_transformed, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], 
                            shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], 
                          shuffle=False, num_workers=0)
    
    # Create model and GBDT extractor
    model = DeepGBM(config).to(device)
    gbdt_extractor = GBDTFeatureExtractor(config)
    
    if verbose:
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Train GBDT models
    if verbose:
        print("Training GBDT components...")
    gbdt_extractor.train_gbdt_models(X_train_raw, y_train, X_val_raw, y_val)
    
    # Extract GBDT features
    train_gbdt_features = gbdt_extractor.extract_features(X_train_raw)
    val_gbdt_features = gbdt_extractor.extract_features(X_val_raw)
    
    # Convert to tensors
    train_gbdt_tensor = torch.tensor(train_gbdt_features, dtype=torch.float32).to(device)
    val_gbdt_tensor = torch.tensor(val_gbdt_features, dtype=torch.float32).to(device)
    
    # Loss and optimizer
    criterion = nn.HuberLoss(delta=config['huber_delta'])
    
    # Optimizer with different learning rates
    param_groups = [
        {'params': model.catnn.parameters(), 'lr': config['learning_rate']},
        {'params': model.fusion_layers.parameters(), 'lr': config['learning_rate'] * 0.5},
        {'params': [model.residual_weight], 'lr': config['learning_rate'] * 0.01}
    ]
    
    optimizer = optim.AdamW(param_groups, weight_decay=config['weight_decay'], eps=1e-8)
    
    # Learning rate scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=3, 
        verbose=False
    )
    
    # Training
    best_pearson = -np.inf
    patience_counter = 0
    
    for epoch in range(config['num_epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        if verbose:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']}")
        else:
            progress_bar = train_loader
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Get corresponding GBDT features
            batch_start = batch_idx * train_loader.batch_size
            batch_end = min((batch_idx + 1) * train_loader.batch_size, len(train_loader.dataset))
            batch_indices = torch.arange(batch_start, batch_end)
            gbdt_batch = train_gbdt_tensor[batch_indices]
            
            # Apply augmentation
            if np.random.random() < 0.8:
                noise_factor = config.get('noise_factor', 0.01)
                noise = torch.randn_like(inputs) * noise_factor
                inputs = inputs + noise
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs, gbdt_batch)
            loss = criterion(outputs, targets)
            
            # Add L2 regularization on outputs
            if config.get('output_penalty', 0) > 0:
                output_penalty = config['output_penalty'] * torch.mean(outputs ** 2)
                loss = loss + output_penalty
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config['grad_clip'])
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            if verbose:
                progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = train_loss / train_batches
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(val_loader):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Get corresponding GBDT features
                batch_start = batch_idx * val_loader.batch_size
                batch_end = min((batch_idx + 1) * val_loader.batch_size, len(val_loader.dataset))
                batch_indices = torch.arange(batch_start, batch_end)
                gbdt_batch = val_gbdt_tensor[batch_indices]
                
                outputs = model(inputs, gbdt_batch)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())
        
        avg_val_loss = val_loss / len(val_loader)
        val_pearson = pearsonr(val_targets, val_preds)[0]
        
        # Update scheduler
        scheduler.step(val_pearson)
        
        if verbose and epoch % 5 == 0:
            print(f"\nEpoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            print(f"Val Pearson: {val_pearson:.4f}")
        
        # Save best model
        if val_pearson > best_pearson:
            best_pearson = val_pearson
            best_model_state = model.state_dict().copy()
            best_gbdt_models = gbdt_extractor.models.copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                if verbose:
                    print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(best_model_state)
    gbdt_extractor.models = best_gbdt_models
    
    return model, gbdt_extractor, best_pearson

# =========================
# Configuration Conversion Function
# =========================
def convert_optuna_params_to_config(optuna_params, input_dim):
    """Convert Optuna parameters to proper config format"""
    config = optuna_params.copy()
    config['input_dim'] = input_dim
    
    # Convert hidden dimensions from individual params to list
    hidden_dims = []
    i = 0
    while f'hidden_dim_{i}' in config:
        hidden_dims.append(config[f'hidden_dim_{i}'])
        del config[f'hidden_dim_{i}']
        i += 1
    
    config['catnn_hidden_dims'] = hidden_dims
    
    # Add any missing required parameters
    config.setdefault('num_epochs', 50)
    config.setdefault('patience', 10)
    config.setdefault('grad_clip', 1.0)
    config.setdefault('use_gbdt_features', True)
    config.setdefault('seed', 42)
    config.setdefault('xgb_reg_alpha', 0.1)
    config.setdefault('xgb_reg_lambda', 0.1)
    config.setdefault('lgb_reg_alpha', 0.1)
    config.setdefault('lgb_reg_lambda', 0.1)
    
    return config

# =========================
# Main Execution
# =========================
def main():
    print("=== DeepGBM for Crypto Market Prediction ===")
    
    # Load training data
    print("\nLoading training data...")
    train = pl.scan_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
    train = train.select(SELECTED_FEATURES).collect().to_pandas()
    
    print(f"Initial training data shape: {train.shape}")
    
    # Use recent data for better relevance
    train_size = int(0.85 * len(train))
    train = train.iloc[-train_size:].reset_index(drop=True)
    print(f"Using last {train_size} samples for training")
    
    # Add market microstructure features
    print("\nAdding advanced market microstructure features...")
    train = add_advanced_market_features(train)
    train = compute_rolling_features(train)
    
    # Get all feature names
    all_features = [col for col in train.columns if col != 'label']
    print(f"Total number of features after engineering: {len(all_features)}")
    
    # Split data
    X_train, X_val = train_test_split(train, test_size=0.2, shuffle=False, random_state=42)
    
    y_train = X_train.pop("label")
    y_val = X_val.pop("label")
    
    # Feature importance analysis
    print("\nComputing feature importance...")
    importance_df = compute_feature_importance(
        X_train[all_features].values,
        y_train,
        all_features,
        n_samples=min(100000, len(X_train))
    )
    
    # Select features with importance threshold
    feature_threshold = 0.01
    selected_features = importance_df[importance_df['mi_score'] > feature_threshold]['feature'].tolist()
    print(f"Features with importance > {feature_threshold}: {len(selected_features)}")
    
    # Prepare data with selected features
    X_train_selected = X_train[selected_features].values
    X_val_selected = X_val[selected_features].values
    
    # Store raw data for GBDT
    X_train_raw = X_train_selected.copy()
    X_val_raw = X_val_selected.copy()
    y_train_values = y_train.values
    y_val_values = y_val.values
    
    # Apply transformations for neural network
    print("\nApplying data transformations...")
    X_train_transformed, transformer = apply_advanced_transformations(X_train_selected, method='quantile')
    X_val_transformed = transformer.transform(X_val_selected)
    
    print(f"Final training shape: {X_train_transformed.shape}")
    print(f"Final validation shape: {X_val_transformed.shape}")
    
    # Define best configuration (example configuration - you would use your Optuna results)
    best_config = {
        'input_dim': X_train_transformed.shape[1],
        'catnn_embedding_dim': 256,
        'catnn_hidden_dims': [256, 128, 64],
        'catnn_output_dim': 64,
        'gbdt_output_dim': 100,
        'fusion_hidden_dim': 256,
        'batch_size': 512,
        'learning_rate': 0.001,
        'dropout': 0.2,
        'fusion_dropout': 0.3,
        'attention_dropout': 0.15,
        'num_attention_heads': 4,
        'weight_decay': 0.01,
        'huber_delta': 1.0,
        'noise_factor': 0.01,
        'output_penalty': 0.001,
        'xgb_max_depth': 6,
        'xgb_learning_rate': 0.05,
        'xgb_n_estimators': 200,
        'xgb_subsample': 0.8,
        'xgb_colsample': 0.8,
        'lgb_max_depth': 6,
        'lgb_num_leaves': 31,
        'lgb_learning_rate': 0.05,
        'lgb_n_estimators': 200,
        'lgb_subsample': 0.8,
        'lgb_colsample': 0.8,
        'num_epochs': 50,
        'patience': 10,
        'grad_clip': 1.0,
        'use_gbdt_features': True,
        'seed': 42
    }
    
    # Train ensemble models
    print("\n=== Training Final Ensemble ===")
    ensemble_models = []
    ensemble_seeds = [42, 123, 456]
    
    for i, seed in enumerate(ensemble_seeds):
        print(f"\n--- Training Ensemble Model {i+1} ---")
        
        model_config = best_config.copy()
        model_config['seed'] = seed
        
        # Train model
        model, gbdt_extractor, best_pearson = train_single_deepgbm(
            model_config, X_train_transformed, y_train_values, X_val_transformed, y_val_values,
            X_train_raw, X_val_raw, device, verbose=True
        )
        
        print(f"Model {i+1} validation Pearson: {best_pearson:.4f}")
        
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'gbdt_models': gbdt_extractor.models,
            'config': model_config,
            'transformer': transformer,
            'selected_features': selected_features
        }, f"deepgbm_ensemble_{i+1}.pt")
        
        ensemble_models.append((model, gbdt_extractor, transformer, selected_features))
    
    # Test prediction
    print("\n=== Making Test Predictions ===")
    
    # Load test data
    test = pl.scan_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
    test_features = [f for f in SELECTED_FEATURES if f != "label"]
    test = test.select(test_features).collect().to_pandas()
    
    # Add features to test data
    test = add_advanced_market_features(test)
    test = compute_rolling_features(test)
    
    # Make ensemble predictions
    all_predictions = []
    
    for i, (model, gbdt_extractor, transformer, features) in enumerate(ensemble_models):
        print(f"Making predictions with model {i+1}...")
        
        # Prepare test data
        X_test_selected = test[features].values
        X_test_raw = X_test_selected.copy()
        X_test_transformed = transformer.transform(X_test_selected)
        
        # Extract GBDT features
        test_gbdt_features = gbdt_extractor.extract_features(X_test_raw)
        test_gbdt_tensor = torch.tensor(test_gbdt_features, dtype=torch.float32).to(device)
        
        # Make predictions
        model.eval()
        test_dataset = TensorDataset(torch.tensor(X_test_transformed, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for batch_idx, (inputs,) in enumerate(test_loader):
                inputs = inputs.to(device)
                
                # Get corresponding GBDT features
                batch_start = batch_idx * test_loader.batch_size
                batch_end = min((batch_idx + 1) * test_loader.batch_size, len(test_dataset))
                batch_indices = torch.arange(batch_start, batch_end)
                gbdt_batch = test_gbdt_tensor[batch_indices]
                
                outputs = model(inputs, gbdt_batch)
                predictions.extend(outputs.cpu().numpy().flatten())
        
        all_predictions.append(np.array(predictions))
    
    # Ensemble predictions
    final_predictions = np.mean(all_predictions, axis=0)
    
    # Post-processing
    pred_mean = y_train.mean()
    pred_std = y_train.std()
    final_predictions = np.clip(final_predictions, 
                               pred_mean - 4 * pred_std, 
                               pred_mean + 4 * pred_std)
    
    # Create submission
    submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
    submission["prediction"] = final_predictions
    submission.to_csv("submission_deepgbm.csv", index=False)
    
    # Display results
    print("\n=== Final Results ===")
    print(f"Ensemble size: {len(ensemble_models)} DeepGBM models")
    print(f"Features used: {len(selected_features)}")
    print(f"\nPrediction statistics:")
    print(f"Mean: {final_predictions.mean():.6f}")
    print(f"Std: {final_predictions.std():.6f}")
    print(f"Min: {final_predictions.min():.6f}")
    print(f"Max: {final_predictions.max():.6f}")
    print(f"Skewness: {skew(final_predictions):.4f}")
    print(f"Kurtosis: {kurtosis(final_predictions):.4f}")
    
    print("\nTop 15 most important features:")
    print(importance_df.head(15)[['feature', 'mi_score']])
    
    print("\nSubmission saved as 'submission_deepgbm.csv'")

if __name__ == "__main__":
    main()

