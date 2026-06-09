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

#!/usr/bin/env python3
"""
Enhanced TANGOS Feature Importance Analysis and Prediction for DRW Crypto Market Prediction
Major improvements:
- Multiple feature combination strategies
- Noise injection for regularization
- Intelligent hyperparameter search
- Comprehensive visualizations
- Detailed analytics and insights
- Correlation analysis
- Feature interaction analysis
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr, pearsonr, rankdata
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import sys
import gc
import random
import json
from typing import List, Tuple, Dict, Optional
from itertools import combinations
import optuna
from datetime import datetime

warnings.filterwarnings("ignore")

# Set random seeds for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# =========================
# Feature Engineering
# =========================
def add_features(df):
    """Add engineered features to the dataframe"""
    print("Adding engineered features...")
    
    # Store original columns to avoid modifying them
    original_cols = df.columns.tolist()
    
    # Basic interactions
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
    
    # Volume-based features
    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['log_volume'] = np.log1p(df['volume'])
    
    # Market microstructure features
    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-10)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    
    # Order flow features
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    
    # Depth features
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])
    
    # Market impact proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
    # Activity indicators
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    
    # Log transformations
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # Volatility proxies
    df['realized_spread_proxy'] = 2 * np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])
    
    # Complex interactions
    df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['depth_volume_interaction'] = df['total_depth'] * df['volume']
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    
    # Information measures
    df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + 1e-10)
    df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10) * df['volume']
    
    # Market efficiency
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + 1e-10)
    
    # Non-linear transformations
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['sqrt_depth'] = np.sqrt(df['total_depth'])
    df['volume_squared'] = df['volume'] ** 2
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    
    # Relative measures
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
    # Market stress
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    
    # Directional indicators
    df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
    df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume']
    
    # Percentile features for better normalization
    for col in ['volume', 'total_depth', 'net_order_flow', 'bid_qty', 'ask_qty']:
        if col in df.columns:
            df[f'{col}_percentile'] = df[col].rank(pct=True)
    
    # Advanced ratio features
    df['bid_ask_volume_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + 1e-10)
    df['trade_imbalance_ratio'] = df['order_flow_imbalance'] * df['depth_imbalance']
    
    # Volume-depth logarithmic ratios
    df['volume_depth_log_ratio'] = np.log1p(df['volume']) / (np.log1p(df['total_depth']) + 1e-10)
    df['sqrt_flow_volume'] = np.sqrt(np.abs(df['net_order_flow'])) * np.sign(df['net_order_flow'])
    
    # Harmonic means
    df['harmonic_mean_depth'] = 2 / (1/(df['bid_qty'] + 1e-10) + 1/(df['ask_qty'] + 1e-10))
    df['harmonic_mean_trade'] = 2 / (1/(df['buy_qty'] + 1e-10) + 1/(df['sell_qty'] + 1e-10))
    
    # NEW: Interaction features between top X features (to be added after feature importance)
    
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"Added {len(df.columns) - len(original_cols)} engineered features")
    
    return df

def add_top_feature_interactions(df, top_features, n_interactions=10):
    """Add interaction features between top features"""
    print(f"Adding interaction features between top {len(top_features)} features...")
    
    # Select top X features for interactions
    interaction_features = top_features[:20]  # Use top 20 for interactions
    
    # Create pairwise interactions
    interactions_added = 0
    for i, feat1 in enumerate(interaction_features):
        for feat2 in interaction_features[i+1:]:
            if interactions_added >= n_interactions:
                break
            
            # Multiplicative interaction
            df[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
            
            # Additive interaction
            df[f'{feat1}_plus_{feat2}'] = df[feat1] + df[feat2]
            
            # Ratio interaction
            df[f'{feat1}_div_{feat2}'] = df[feat1] / (df[feat2] + 1e-10)
            
            interactions_added += 3
        
        if interactions_added >= n_interactions:
            break
    
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    print(f"Added {interactions_added} interaction features")
    
    return df

# =========================
# Model Components
# =========================
class NoiseInjection(nn.Module):
    """Noise injection layer for regularization"""
    def __init__(self, noise_level=0.1):
        super().__init__()
        self.noise_level = noise_level
    
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.noise_level
            return x + noise
        return x

class StableOrthogonalProjection(nn.Module):
    """Stable orthogonal projection using QR decomposition"""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = min(output_dim, input_dim)
        
        self.weight = nn.Parameter(torch.empty(self.output_dim, input_dim))
        nn.init.orthogonal_(self.weight)
        
        self.bias = nn.Parameter(torch.zeros(self.output_dim))
        
    def forward(self, x):
        Q, R = torch.linalg.qr(self.weight.t())
        orthogonal_weight = Q[:, :self.output_dim].t()
        
        return F.linear(x, orthogonal_weight, self.bias)

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout"""
    def __init__(self, dim, dropout_rate=0.3):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.bn1 = nn.BatchNorm1d(dim)
        self.bn2 = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.GELU()
        
    def forward(self, x):
        residual = x
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        return self.activation(out + residual)

class AttentionFeatureFusion(nn.Module):
    """Attention mechanism for feature interactions"""
    def __init__(self, input_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(input_dim)
        
    def forward(self, x):
        # Reshape for attention: (batch, 1, features)
        x_reshaped = x.unsqueeze(1)
        attn_out, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
        attn_out = attn_out.squeeze(1)
        return self.norm(x + attn_out)

class FeatureSelector(nn.Module):
    """Feature selection with learnable gates"""
    def __init__(self, input_dim, temperature=1.0):
        super().__init__()
        self.input_dim = input_dim
        self.temperature = temperature
        
        self.importance_logits = nn.Parameter(torch.randn(input_dim) * 0.01)
        
    def forward(self, x, hard=False):
        gates = torch.sigmoid(self.importance_logits / self.temperature)
        
        if hard:
            gates = (gates > 0.5).float()
        
        selected_features = x * gates.unsqueeze(0)
        
        return selected_features, gates
    
    def get_feature_ranking(self):
        with torch.no_grad():
            scores = torch.sigmoid(self.importance_logits)
            ranking = torch.argsort(scores, descending=True)
            return ranking, scores

class EnhancedTANGOSEncoder(nn.Module):
    """Enhanced encoder with residual connections and attention"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout_rate=0.3, 
                 use_residual=True, use_attention=False, noise_level=0.0):
        super().__init__()
        
        self.use_residual = use_residual
        self.use_attention = use_attention
        
        layers = []
        
        # Add noise injection if specified
        if noise_level > 0:
            layers.append(NoiseInjection(noise_level))
        
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            # Use orthogonal projection for dimension reduction
            if i < len(hidden_dims) - 1 and hidden_dim < prev_dim:
                layers.append(StableOrthogonalProjection(prev_dim, hidden_dim))
            else:
                layers.append(nn.Linear(prev_dim, hidden_dim))
            
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            
            # Add residual block
            if use_residual and i % 2 == 1 and i < len(hidden_dims) - 1:
                layers.append(ResidualBlock(hidden_dim, dropout_rate))
            
            # Add attention layer
            if use_attention and i == len(hidden_dims) // 2:
                layers.append(AttentionFeatureFusion(hidden_dim))
            
            prev_dim = hidden_dim
        
        self.layers = nn.ModuleList(layers)
        self.output_dim = prev_dim
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class EnhancedTANGOS(nn.Module):
    """Enhanced TANGOS with all improvements"""
    def __init__(self, input_dim, encoder_dims=[256, 128, 64], 
                 predictor_dims=[32, 16], dropout_rate=0.3, temperature=1.0, 
                 use_residual=True, use_attention=False, noise_level=0.0):
        super().__init__()
        
        self.input_dim = input_dim
        
        self.feature_selector = FeatureSelector(input_dim, temperature)
        self.encoder = EnhancedTANGOSEncoder(
            input_dim, encoder_dims, dropout_rate, 
            use_residual, use_attention, noise_level
        )
        
        # Predictor with skip connection
        layers = []
        prev_dim = self.encoder.output_dim
        
        for hidden_dim in predictor_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.predictor = nn.Sequential(*layers)
        
        # For gradient tracking
        self.register_buffer('gradient_importance', torch.zeros(input_dim))
        self.register_buffer('gradient_count', torch.tensor(0))
        
    def forward(self, x, track_gradients=False):
        selected_x, gates = self.feature_selector(x)
        encoded = self.encoder(selected_x)
        output = self.predictor(encoded)
        
        if track_gradients and x.requires_grad:
            self.track_feature_gradients(x, output)
        
        return output, gates
    
    def track_feature_gradients(self, x, output):
        if output.requires_grad:
            grad = torch.autograd.grad(
                outputs=output.sum(),
                inputs=x,
                retain_graph=True,
                create_graph=False
            )[0]
            
            self.gradient_importance += torch.abs(grad).mean(dim=0).detach()
            self.gradient_count += 1
    
    def get_gradient_importance(self):
        if self.gradient_count > 0:
            return self.gradient_importance / self.gradient_count
        return torch.zeros(self.input_dim, device=self.gradient_importance.device)

# =========================
# Training Functions
# =========================
def train_model_advanced(model, train_loader, val_loader, config, device='cuda'):
    """Advanced training with multiple improvements"""
    criterion = nn.HuberLoss(delta=config['huber_delta'])
    
    # Optimizer selection
    if config['optimizer'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'RAdam':
        optimizer = torch.optim.RAdam(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
    
    # Learning rate scheduler
    if config['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=5, T_mult=2, eta_min=1e-5
        )
    elif config['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=3
        )
    else:
        scheduler = None
    
    best_val_corr = -float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_corr': []}
    
    for epoch in range(config['n_epochs']):
        # Training
        model.train()
        train_loss = 0.0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['n_epochs']}")
        for i, (inputs, targets) in enumerate(progress_bar):
            inputs = inputs.to(device).requires_grad_(True)
            targets = targets.to(device)
            
            # Mixup augmentation
            if config.get('mixup', False) and np.random.random() < 0.5:
                lam = np.random.beta(0.2, 0.2)
                batch_size = inputs.size(0)
                index = torch.randperm(batch_size).to(device)
                mixed_inputs = lam * inputs + (1 - lam) * inputs[index]
                mixed_targets = lam * targets + (1 - lam) * targets[index]
                
                outputs, gates = model(mixed_inputs, track_gradients=True)
                loss = criterion(outputs, mixed_targets)
            else:
                outputs, gates = model(inputs, track_gradients=True)
                loss = criterion(outputs, targets)
            
            # Add regularization
            sparsity_loss = config['sparsity_weight'] * gates.mean()
            total_loss = loss + sparsity_loss
            
            # Gradient accumulation
            total_loss = total_loss / config['accumulation_steps']
            total_loss.backward()
            
            if (i + 1) % config['accumulation_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
                optimizer.step()
                optimizer.zero_grad()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        # Step scheduler
        if scheduler and config['scheduler'] == 'cosine':
            scheduler.step()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                
                outputs, _ = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_corr = pearsonr(val_targets, val_preds)[0]
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_corr'].append(val_corr)
        
        print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, "
              f"Val Loss={avg_val_loss:.4f}, Val Corr={val_corr:.4f}, "
              f"LR={optimizer.param_groups[0]['lr']:.6f}")
        
        # Step plateau scheduler
        if scheduler and config['scheduler'] == 'plateau':
            scheduler.step(val_corr)
        
        # Save best model
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            torch.save(model.state_dict(), 'best_model.pt')
            patience_counter = 0
        else:
            patience_counter += 1
        
        if patience_counter >= config['patience']:
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(torch.load('best_model.pt'))
    
    return model, history, best_val_corr

# =========================
# Hyperparameter Optimization
# =========================
def objective(trial, X_train, y_train, X_val, y_val, input_dim, device):
    """Optuna objective function for hyperparameter optimization"""
    
    # Suggest hyperparameters
    config = {
        'encoder_dims': trial.suggest_categorical('encoder_dims', 
            [[256, 128, 64], [512, 256, 128], [128, 64, 32], [256, 128, 64, 32]]),
        'predictor_dims': trial.suggest_categorical('predictor_dims', 
            [[32, 16], [64, 32], [16, 8], [32]]),
        'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
        'lr': trial.suggest_float('lr', 1e-4, 1e-2, log=True),
        'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        'temperature': trial.suggest_float('temperature', 0.5, 3.0),
        'noise_level': trial.suggest_float('noise_level', 0.0, 0.2),
        'use_attention': trial.suggest_categorical('use_attention', [True, False]),
        'optimizer': trial.suggest_categorical('optimizer', ['AdamW', 'Adam', 'RAdam']),
        'scheduler': trial.suggest_categorical('scheduler', ['cosine', 'plateau', 'none']),
        'huber_delta': trial.suggest_float('huber_delta', 0.5, 2.0),
        'sparsity_weight': trial.suggest_float('sparsity_weight', 0.001, 0.1, log=True),
        'batch_size': trial.suggest_categorical('batch_size', [256, 512, 1024]),
        'n_epochs': 10,
        'patience': 3,
        'accumulation_steps': 2,
        'grad_clip': 1.0,
        'mixup': trial.suggest_categorical('mixup', [True, False])
    }
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size']*2, shuffle=False)
    
    # Initialize model
    model = EnhancedTANGOS(
        input_dim=input_dim,
        encoder_dims=config['encoder_dims'],
        predictor_dims=config['predictor_dims'],
        dropout_rate=config['dropout_rate'],
        temperature=config['temperature'],
        use_residual=True,
        use_attention=config['use_attention'],
        noise_level=config['noise_level']
    ).to(device)
    
    # Train model
    model, history, best_val_corr = train_model_advanced(
        model, train_loader, val_loader, config, device
    )
    
    return best_val_corr

def optimize_hyperparameters(X_train, y_train, X_val, y_val, input_dim, device, n_trials=20):
    """Run hyperparameter optimization using Optuna"""
    print("\n" + "="*60)
    print("HYPERPARAMETER OPTIMIZATION")
    print("="*60)
    
    study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val, input_dim, device),
        n_trials=n_trials
    )
    
    print(f"\nBest trial:")
    print(f"Value: {study.best_value:.4f}")
    print(f"Params: {study.best_params}")
    
    return study.best_params

# =========================
# Analysis Functions
# =========================
def analyze_feature_correlations(df, features, target='label', top_n=30):
    """Analyze and visualize feature correlations"""
    print("\nAnalyzing feature correlations...")
    
    # Calculate correlations with target
    correlations = []
    for feat in features:
        if feat in df.columns:
            corr = pearsonr(df[feat], df[target])[0]
            correlations.append((feat, corr))
    
    # Sort by absolute correlation
    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top correlations bar plot
    top_features = [x[0] for x in correlations[:top_n]]
    top_corrs = [x[1] for x in correlations[:top_n]]
    
    ax1.barh(range(len(top_features)), top_corrs)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features)
    ax1.set_xlabel('Correlation with Target')
    ax1.set_title(f'Top {top_n} Feature Correlations')
    ax1.grid(True, alpha=0.3)
    
    # Correlation matrix heatmap
    corr_matrix = df[top_features + [target]].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_title('Feature Correlation Matrix')
    
    plt.tight_layout()
    plt.savefig('outputs/feature_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return correlations

def visualize_training_history(history, title="Training History"):
    """Visualize training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Loss plot
    ax1.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax1.plot(history['val_loss'], label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Correlation plot
    ax2.plot(history['val_corr'], label='Val Correlation', linewidth=2, color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Pearson Correlation')
    ax2.set_title('Validation Correlation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f'outputs/{title.lower().replace(" ", "_")}.png', dpi=300, bbox_inches='tight')
    plt.close()

def compute_comprehensive_feature_importance(model, X_sample, feature_names, n_permutations=5):
    """Compute comprehensive feature importance using multiple methods"""
    model.eval()
    
    # 1. Selection scores from gates
    _, selection_scores = model.feature_selector.get_feature_ranking()
    selection_scores = selection_scores.cpu().numpy()
    
    # 2. Gradient importance
    gradient_importance = model.get_gradient_importance().cpu().numpy()
    
    # 3. Permutation importance (simplified)
    perm_importance = np.zeros(len(feature_names))
    
    with torch.no_grad():
        X_tensor = torch.tensor(X_sample, dtype=torch.float32, device=next(model.parameters()).device)
        base_pred, _ = model(X_tensor)
        
        for i in range(len(feature_names)):
            if i % 50 == 0:
                print(f"Computing permutation importance: {i}/{len(feature_names)}")
            
            perm_scores = []
            for _ in range(n_permutations):
                X_perm = X_tensor.clone()
                # Permute feature i
                X_perm[:, i] = X_perm[torch.randperm(X_perm.size(0)), i]
                perm_pred, _ = model(X_perm)
                # Measure change in predictions
                change = torch.abs(base_pred - perm_pred).mean().item()
                perm_scores.append(change)
            
            perm_importance[i] = np.mean(perm_scores)
    
    # Normalize all scores
    def normalize_scores(scores):
        if scores.max() > scores.min():
            return (scores - scores.min()) / (scores.max() - scores.min())
        return np.zeros_like(scores)
    
    selection_norm = normalize_scores(selection_scores)
    gradient_norm = normalize_scores(gradient_importance)
    perm_norm = normalize_scores(perm_importance)
    
    # Combined importance with weights
    combined_importance = (
        0.4 * selection_norm + 
        0.3 * gradient_norm + 
        0.3 * perm_norm
    )
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'combined_importance': combined_importance,
        'selection_score': selection_scores,
        'gradient_importance': gradient_importance,
        'permutation_importance': perm_importance
    })
    
    importance_df = importance_df.sort_values('combined_importance', ascending=False)
    
    # Visualize top features
    fig, ax = plt.subplots(figsize=(10, 8))
    
    top_n = 30
    top_features = importance_df.head(top_n)
    
    y_pos = np.arange(len(top_features))
    ax.barh(y_pos, top_features['combined_importance'], alpha=0.8, label='Combined')
    ax.barh(y_pos, top_features['selection_score'], alpha=0.5, label='Selection')
    ax.barh(y_pos, top_features['gradient_importance'], alpha=0.5, label='Gradient')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Feature Importance Scores')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/feature_importance_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    return importance_df

def create_feature_strategy_dict(x_features, market_features, engineered_features, 
                               importance_df=None):
    """Create different feature selection strategies"""
    strategies = {}
    
    # 1. All features
    strategies['all_features'] = x_features + market_features + engineered_features
    
    # 2. Only X features
    strategies['x_only'] = x_features
    
    # 3. Market + Engineered features
    strategies['market_engineered'] = market_features + engineered_features
    
    # 4. Top K features (if importance_df provided)
    if importance_df is not None:
        for k in [30, 50, 80, 100, 150, 200]:
            strategies[f'top_{k}'] = importance_df.head(k)['feature'].tolist()
        
        # 5. Top X features + all market/engineered
        top_x = [f for f in importance_df.head(100)['feature'].tolist() if f.startswith('X')]
        strategies['top_x_plus_market'] = top_x + market_features + engineered_features
    
    return strategies

def create_ensemble_predictions_advanced(train_df, test_df, feature_strategy, 
                                        strategy_name, best_params=None, 
                                        n_models=5, device='cuda'):
    """Create advanced ensemble predictions with multiple strategies"""
    print(f"\n{'='*60}")
    print(f"Creating ensemble for strategy: {strategy_name}")
    print(f"Number of features: {len(feature_strategy)}")
    print(f"{'='*60}")
    
    predictions_list = []
    model_scores = []
    
    # Different configurations for ensemble diversity
    ensemble_configs = []
    
    # Base configuration (use best params if available)
    if best_params:
        base_config = {
            'encoder_dims': best_params.get('encoder_dims', [256, 128, 64]),
            'predictor_dims': best_params.get('predictor_dims', [32, 16]),
            'dropout_rate': best_params.get('dropout_rate', 0.3),
            'lr': best_params.get('lr', 0.003),
            'weight_decay': best_params.get('weight_decay', 1e-4),
            'temperature': best_params.get('temperature', 1.0),
            'noise_level': best_params.get('noise_level', 0.1),
            'use_attention': best_params.get('use_attention', False),
            'optimizer': best_params.get('optimizer', 'AdamW'),
            'scheduler': best_params.get('scheduler', 'cosine'),
            'huber_delta': best_params.get('huber_delta', 1.0),
            'sparsity_weight': best_params.get('sparsity_weight', 0.01),
            'n_epochs': 20,
            'patience': 5,
            'accumulation_steps': 2,
            'grad_clip': 1.0,
            'mixup': best_params.get('mixup', True)
        }
        ensemble_configs.append(base_config)
    
    # Add variations
    variations = [
        {'encoder_dims': [512, 256, 128], 'predictor_dims': [64, 32], 'dropout_rate': 0.2},
        {'encoder_dims': [128, 64, 32], 'predictor_dims': [16, 8], 'dropout_rate': 0.4},
        {'encoder_dims': [256, 128, 64, 32], 'predictor_dims': [32], 'use_attention': True},
        {'encoder_dims': [384, 192, 96], 'predictor_dims': [48, 24], 'noise_level': 0.15},
        {'encoder_dims': [256, 128], 'predictor_dims': [64, 32, 16], 'optimizer': 'RAdam'}
    ]
    
    for var in variations[:n_models-1]:
        config = base_config.copy() if best_params else {
            'encoder_dims': [256, 128, 64],
            'predictor_dims': [32, 16],
            'dropout_rate': 0.3,
            'lr': 0.003,
            'weight_decay': 1e-4,
            'temperature': 1.0,
            'noise_level': 0.1,
            'use_attention': False,
            'optimizer': 'AdamW',
            'scheduler': 'cosine',
            'huber_delta': 1.0,
            'sparsity_weight': 0.01,
            'n_epochs': 20,
            'patience': 5,
            'accumulation_steps': 2,
            'grad_clip': 1.0,
            'mixup': False
        }
        config.update(var)
        config['batch_size'] = 512 if device.type == 'cuda' else 256
        ensemble_configs.append(config)
    
    # Train ensemble models
    for i in range(n_models):
        print(f"\nTraining model {i+1}/{n_models}")
        
        # Set seed for reproducibility with variation
        seed = RANDOM_SEED + i
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # Get configuration
        config = ensemble_configs[i % len(ensemble_configs)]
        
        # Prepare data
        X_full = train_df[feature_strategy].values
        y_full = train_df['label'].values
        
        # Train/validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X_full, y_full, test_size=0.15, random_state=seed
        )
        
        # Scale features (alternate between scalers)
        if i % 3 == 0:
            scaler = StandardScaler()
        elif i % 3 == 1:
            scaler = RobustScaler()
        else:
            scaler = QuantileTransformer(output_distribution='normal', random_state=seed)
        
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train_scaled, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_scaled, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config.get('batch_size', 512), 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config.get('batch_size', 512)*2, 
            shuffle=False
        )
        
        # Initialize model
        model = EnhancedTANGOS(
            input_dim=len(feature_strategy),
            encoder_dims=config['encoder_dims'],
            predictor_dims=config['predictor_dims'],
            dropout_rate=config['dropout_rate'],
            temperature=config['temperature'],
            use_residual=True,
            use_attention=config.get('use_attention', False),
            noise_level=config.get('noise_level', 0.0)
        ).to(device)
        
        # Train model
        model, history, best_score = train_model_advanced(
            model, train_loader, val_loader, config, device
        )
        
        model_scores.append(best_score)
        
        # Visualize training history for first model
        if i == 0:
            visualize_training_history(history, f"{strategy_name}_model_{i+1}")
        
        # Generate predictions
        test_features = test_df[feature_strategy].values
        test_features_scaled = scaler.transform(test_features)
        test_tensor = torch.tensor(test_features_scaled, dtype=torch.float32, device=device)
        
        model.eval()
        predictions = []
        
        with torch.no_grad():
            batch_size = config.get('batch_size', 512) * 4
            for j in range(0, len(test_tensor), batch_size):
                batch = test_tensor[j:j+batch_size]
                batch_preds, _ = model(batch)
                predictions.extend(batch_preds.cpu().numpy().flatten())
        
        predictions_list.append(np.array(predictions))
        
        # Clean up
        del model
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    
    print(f"\nEnsemble model scores: {[f'{s:.4f}' for s in model_scores]}")
    print(f"Mean score: {np.mean(model_scores):.4f} Â± {np.std(model_scores):.4f}")
    
    return predictions_list, model_scores

def create_final_predictions(predictions_list, model_scores, method='weighted_rank'):
    """Create final predictions using advanced ensemble methods"""
    
    if method == 'weighted_mean':
        # Weight by model performance
        weights = np.array(model_scores)
        weights = weights / weights.sum()
        predictions = np.average(predictions_list, axis=0, weights=weights)
        
    elif method == 'weighted_rank':
        # Rank-based weighted average
        ranked_preds = []
        for pred in predictions_list:
            ranks = rankdata(pred) / len(pred)
            ranked_preds.append(ranks)
        
        weights = np.array(model_scores)
        weights = weights / weights.sum()
        avg_ranks = np.average(ranked_preds, axis=0, weights=weights)
        
        # Convert back to prediction scale
        all_preds = np.concatenate(predictions_list)
        sorted_preds = np.sort(all_preds)
        rank_indices = (avg_ranks * (len(sorted_preds) - 1)).astype(int)
        predictions = sorted_preds[rank_indices]
        
    elif method == 'trimmed_mean':
        # Remove best and worst prediction for each sample
        predictions = np.array(predictions_list)
        trimmed = np.sort(predictions, axis=0)[1:-1]
        predictions = np.mean(trimmed, axis=0)
        
    elif method == 'median':
        predictions = np.median(predictions_list, axis=0)
    
    else:  # 'mean'
        predictions = np.mean(predictions_list, axis=0)
    
    # Post-processing
    # 1. Clip extreme values
    p1, p99 = np.percentile(predictions, [1, 99])
    predictions = np.clip(predictions, p1, p99)
    
    # 2. Smooth slightly
    predictions = gaussian_filter1d(predictions, sigma=0.5)
    
    return predictions

# =========================
# Main Pipeline
# =========================
def main():
    print("="*60)
    print("ENHANCED TANGOS FOR DRW CRYPTO MARKET PREDICTION")
    print("="*60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {device}")
    print("="*60)
    
    # Data paths
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_submission_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Load data
    print("\nğŸ“Š Loading competition data...")
    
    # Define features
    x_features = [f"X{i}" for i in range(1, 891)]
    market_features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
    
    # Load data
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    sample_submission = pd.read_csv(sample_submission_path)
    
    print(f"âœ“ Loaded {len(train_df):,} training samples")
    print(f"âœ“ Loaded {len(test_df):,} test samples")
    
    # Test IDs
    test_ids = np.arange(1, len(test_df) + 1)
    
    # Add engineered features
    train_df = add_features(train_df)
    test_df = add_features(test_df)
    
    # Get all feature names
    engineered_features = [col for col in train_df.columns 
                          if col not in x_features + market_features + ['timestamp', 'label']]
    all_features = x_features + market_features + engineered_features
    
    print(f"\nğŸ“ˆ Feature Summary:")
    print(f"   - Anonymous features (X_): {len(x_features)}")
    print(f"   - Market features: {len(market_features)}")
    print(f"   - Engineered features: {len(engineered_features)}")
    print(f"   - Total features: {len(all_features)}")
    
    # Create outputs directory
    os.makedirs('outputs', exist_ok=True)
    
    # Analyze feature correlations
    correlations = analyze_feature_correlations(
        train_df.sample(n=min(100000, len(train_df)), random_state=42),
        all_features, 
        target='label', 
        top_n=30
    )
    
    print(f"\nğŸ”� Top 10 correlated features with target:")
    for i, (feat, corr) in enumerate(correlations[:10]):
        print(f"   {i+1}. {feat}: {corr:.4f}")
    
    # Feature importance analysis with small sample
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Use sample for initial feature importance
    sample_size = 30000
    train_sample = train_df.sample(n=min(sample_size, len(train_df)), random_state=42)
    
    X_sample = train_sample[all_features].values
    y_sample = train_sample['label'].values
    
    # Split for feature importance
    X_train_fi, X_val_fi, y_train_fi, y_val_fi = train_test_split(
        X_sample, y_sample, test_size=0.2, random_state=42
    )
    
    # Scale features
    scaler_fi = StandardScaler()
    X_train_fi_scaled = scaler_fi.fit_transform(X_train_fi)
    X_val_fi_scaled = scaler_fi.transform(X_val_fi)
    
    # Hyperparameter optimization (quick version for feature importance)
    print("\nğŸ”§ Running hyperparameter optimization...")
    best_params = optimize_hyperparameters(
        X_train_fi_scaled, y_train_fi, 
        X_val_fi_scaled, y_val_fi, 
        len(all_features), device, 
        n_trials=10  # Quick optimization
    )
    
    # Train model for feature importance
    print("\nğŸ�¯ Training model for feature importance...")
    
    # Create config from best params
    fi_config = {
        'encoder_dims': best_params.get('encoder_dims', [256, 128, 64]),
        'predictor_dims': best_params.get('predictor_dims', [32, 16]),
        'dropout_rate': best_params.get('dropout_rate', 0.3),
        'lr': best_params.get('lr', 0.003),
        'weight_decay': best_params.get('weight_decay', 1e-4),
        'temperature': best_params.get('temperature', 1.0),
        'noise_level': best_params.get('noise_level', 0.1),
        'use_attention': best_params.get('use_attention', False),
        'optimizer': best_params.get('optimizer', 'AdamW'),
        'scheduler': best_params.get('scheduler', 'cosine'),
        'huber_delta': best_params.get('huber_delta', 1.0),
        'sparsity_weight': best_params.get('sparsity_weight', 0.01),
        'batch_size': 512 if device.type == 'cuda' else 256,
        'n_epochs': 15,
        'patience': 5,
        'accumulation_steps': 2,
        'grad_clip': 1.0,
        'mixup': best_params.get('mixup', False)
    }
    
    # Create data loaders
    train_dataset_fi = TensorDataset(
        torch.tensor(X_train_fi_scaled, dtype=torch.float32),
        torch.tensor(y_train_fi, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset_fi = TensorDataset(
        torch.tensor(X_val_fi_scaled, dtype=torch.float32),
        torch.tensor(y_val_fi, dtype=torch.float32).unsqueeze(1)
    )
    
    train_loader_fi = DataLoader(train_dataset_fi, batch_size=fi_config['batch_size'], shuffle=True)
    val_loader_fi = DataLoader(val_dataset_fi, batch_size=fi_config['batch_size']*2, shuffle=False)
    
    # Initialize and train model
    model_fi = EnhancedTANGOS(
        input_dim=len(all_features),
        encoder_dims=fi_config['encoder_dims'],
        predictor_dims=fi_config['predictor_dims'],
        dropout_rate=fi_config['dropout_rate'],
        temperature=fi_config['temperature'],
        use_residual=True,
        use_attention=fi_config['use_attention'],
        noise_level=fi_config['noise_level']
    ).to(device)
    
    model_fi, history_fi, _ = train_model_advanced(
        model_fi, train_loader_fi, val_loader_fi, fi_config, device
    )
    
    # Compute comprehensive feature importance
    print("\nğŸ“Š Computing comprehensive feature importance...")
    importance_df = compute_comprehensive_feature_importance(
        model_fi, X_train_fi_scaled[:1000], all_features, n_permutations=3
    )
    
    # Save feature importance
    importance_df.to_csv('outputs/feature_importance_comprehensive.csv', index=False)
    
    # Display top features
    print("\nğŸ�† Top 20 most important features:")
    for i in range(min(20, len(importance_df))):
        row = importance_df.iloc[i]
        print(f"   {i+1:2d}. {row['feature']:25s} {row['combined_importance']:.4f}")
    
    # Add top feature interactions
    top_interaction_features = importance_df.head(50)['feature'].tolist()
    train_df = add_top_feature_interactions(train_df, top_interaction_features, n_interactions=30)
    test_df = add_top_feature_interactions(test_df, top_interaction_features, n_interactions=30)
    
    # Update feature lists
    all_features_with_interactions = [col for col in train_df.columns 
                                     if col not in ['timestamp', 'label']]
    
    # Clean up
    del model_fi, X_train_fi_scaled, X_val_fi_scaled
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Create feature strategies
    print("\n" + "="*60)
    print("CREATING PREDICTIONS WITH MULTIPLE STRATEGIES")
    print("="*60)
    
    feature_strategies = create_feature_strategy_dict(
        x_features, market_features, engineered_features, importance_df
    )
    
    # Add strategy with interactions
    feature_strategies['top_100_interactions'] = (
        importance_df.head(100)['feature'].tolist() + 
        [col for col in all_features_with_interactions if '_x_' in col or '_plus_' in col or '_div_' in col]
    )
    
    # Submissions dictionary
    all_submissions = {}
    strategy_scores = {}
    
    # Process each strategy
    for strategy_name, features in feature_strategies.items():
        print(f"\n{'='*60}")
        print(f"Strategy: {strategy_name}")
        print(f"Number of features: {len(features)}")
        
        # Skip if too many features for available memory
        if len(features) > 500 and device.type == 'cpu':
            print("âš ï¸�  Skipping - too many features for CPU")
            continue
        
        # Create ensemble predictions
        predictions_list, model_scores = create_ensemble_predictions_advanced(
            train_df, test_df, features, strategy_name, 
            best_params=best_params, n_models=5, device=device
        )
        
        strategy_scores[strategy_name] = {
            'mean_score': np.mean(model_scores),
            'std_score': np.std(model_scores),
            'n_features': len(features)
        }
        
        # Create predictions with different ensemble methods
        ensemble_methods = ['weighted_rank', 'weighted_mean', 'trimmed_mean', 'median']
        
        for method in ensemble_methods:
            predictions = create_final_predictions(predictions_list, model_scores, method)
            
            # Create submission
            submission = pd.DataFrame({
                'ID': test_ids,
                'prediction': predictions
            })
            
            # Save submission
            filename = f'submission_{strategy_name}_{method}.csv'
            submission.to_csv(filename, index=False)
            
            # Store in dictionary
            key = f"{strategy_name}_{method}"
            all_submissions[key] = {
                'filename': filename,
                'mean': submission['prediction'].mean(),
                'std': submission['prediction'].std(),
                'strategy': strategy_name,
                'method': method,
                'score': np.mean(model_scores)
            }
            
            print(f"   âœ“ {method}: mean={submission['prediction'].mean():.4f}, "
                  f"std={submission['prediction'].std():.4f}")
    
    # Create final ensemble of best strategies
    print("\n" + "="*60)
    print("CREATING FINAL ENSEMBLE")
    print("="*60)
    
    # Select best submissions
    best_submissions = sorted(all_submissions.items(), 
                            key=lambda x: x[1]['score'], 
                            reverse=True)[:10]
    
    print("\nğŸ�† Best submissions for final ensemble:")
    for i, (name, info) in enumerate(best_submissions):
        print(f"   {i+1}. {name}: score={info['score']:.4f}")
    
    # Load best predictions
    best_predictions = []
    best_scores = []
    
    for name, info in best_submissions:
        df = pd.read_csv(info['filename'])
        best_predictions.append(df['prediction'].values)
        best_scores.append(info['score'])
    
    # Create final ensemble
    final_predictions = create_final_predictions(best_predictions, best_scores, 'weighted_rank')
    
    # Create final submission
    final_submission = pd.DataFrame({
        'ID': test_ids,
        'prediction': final_predictions
    })
    
    final_submission.to_csv('submission.csv', index=False)
    print(f"\nâœ… Final submission saved to 'submission.csv'")
    print(f"   Mean: {final_submission['prediction'].mean():.6f}")
    print(f"   Std: {final_submission['prediction'].std():.6f}")
    
    # Visualize final predictions
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Distribution
    ax1.hist(final_submission['prediction'], bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Predicted Values')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Final Predictions')
    ax1.grid(True, alpha=0.3)
    
    # QQ plot
    from scipy import stats
    stats.probplot(final_submission['prediction'], dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/final_predictions_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Summary report
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    
    print("\nğŸ“Š Strategy Performance:")
    for strategy, scores in strategy_scores.items():
        print(f"   {strategy}: {scores['mean_score']:.4f} Â± {scores['std_score']:.4f} "
              f"({scores['n_features']} features)")
    
    print("\nğŸ“� Files created:")
    print(f"   - Total submissions: {len(all_submissions)}")
    print("   - Feature importance: outputs/feature_importance_comprehensive.csv")
    print("   - Visualizations: outputs/*.png")
    print("   - Main submission: submission.csv")
    
    # Create prioritized submission list
    print("\n" + "="*60)
    print("PRIORITIZED SUBMISSION LIST")
    print("="*60)
    
    prioritized = sorted(all_submissions.items(), 
                        key=lambda x: x[1]['score'], 
                        reverse=True)
    
    print("\nğŸ�¯ Top 15 submissions to try (in order of priority):")
    for i, (name, info) in enumerate(prioritized[:15]):
        print(f"   {i+1:2d}. {info['filename']} "
              f"(score: {info['score']:.4f}, strategy: {info['strategy']})")
    
    # Save submission metadata
    with open('outputs/submission_metadata.json', 'w') as f:
        json.dump(all_submissions, f, indent=2)
    
    print("\nâœ… Analysis completed successfully!")
    
    return final_submission

# =========================
# Execute Analysis
# =========================
if __name__ == "__main__":
    final_submission = main()

