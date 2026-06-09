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


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DRW Crypto Market Prediction - Neural Architecture Search with Feature Selection
===============================================================================
Enhanced implementation combining NAS with intelligent feature selection
"""

import random
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr
import gc

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Configuration
# =========================
class Config:
    # Paths
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Feature selection parameters
    MIN_VARIANCE_THRESHOLD = 0.01  # Remove features with variance below this
    CORRELATION_THRESHOLD = 0.95   # Remove highly correlated features
    MAX_FEATURES_TO_TEST = 300     # Maximum features to consider
    MIN_FEATURES_TO_KEEP = 50      # Minimum features to retain
    
    # NAS parameters
    NAS_ITERATIONS = 30
    NAS_PATIENCE = 10
    EPOCHS_PER_ARCH = 10
    
    # Model defaults
    SEED = 42
    BATCH_SIZE = 1024 * 16
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-3
    DROPOUT_RATE = 0.6
    NOISE_FACTOR = 0.005

# =========================
# Utility Functions
# =========================
def set_seed(seed=Config.SEED):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def shrink_dtypes(df):
    """Optimize polars dataframe memory usage"""
    return df.select(
        pl.col(col).shrink_dtype() for col in df.collect_schema().names()
    )

def get_activation_function(name):
    """Return the activation function based on the name"""
    if name is None:
        return None
    name = name.lower()
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'sigmoid': nn.Sigmoid(),
        'leakyrelu': nn.LeakyReLU(),
        'elu': nn.ELU(),
        'gelu': nn.GELU(),
        'selu': nn.SELU(),
        'prelu': nn.PReLU()
    }
    if name not in activations:
        raise ValueError(f"Unsupported activation function: {name}")
    return activations[name]

# =========================
# Feature Engineering
# =========================
def create_microstructure_features(df):
    """Create comprehensive microstructure features"""
    print("Creating microstructure features...")
    
    # Basic microstructure features
    df = df.with_columns([
        # Order book imbalance
        ((pl.col("bid_qty") - pl.col("ask_qty")) / 
         (pl.col("bid_qty") + pl.col("ask_qty") + 1e-10)).alias("order_book_imbalance"),
        
        # Trade flow imbalance
        ((pl.col("buy_qty") - pl.col("sell_qty")) / 
         (pl.col("buy_qty") + pl.col("sell_qty") + 1e-10)).alias("trade_flow_imbalance"),
        
        # Liquidity consumption ratio
        (pl.col("volume") / 
         (pl.col("bid_qty") + pl.col("ask_qty") + 1e-10)).alias("liquidity_consumption_ratio"),
        
        # Buy/sell pressure
        (pl.col("buy_qty") / (pl.col("bid_qty") + 1e-10)).alias("buy_pressure"),
        (pl.col("sell_qty") / (pl.col("ask_qty") + 1e-10)).alias("sell_pressure"),
        
        # Market depth
        (pl.col("bid_qty") + pl.col("ask_qty")).alias("total_depth"),
        (pl.col("bid_qty") / (pl.col("ask_qty") + 1e-10)).alias("bid_ask_ratio"),
        
        # Trade metrics
        (pl.col("volume") / 
         ((pl.col("buy_qty") + pl.col("sell_qty")) + 1e-10)).alias("avg_trade_size"),
        
        # Kyle's lambda proxy
        ((pl.col("buy_qty") - pl.col("sell_qty")).abs() / 
         (pl.col("volume").sqrt() + 1e-10)).alias("kyle_lambda_proxy"),
        
        # Amihud illiquidity
        ((pl.col("buy_qty") - pl.col("sell_qty")).abs() / 
         (pl.col("volume") + 1e-10)).alias("amihud_proxy"),
    ])
    
    # Log transformations
    log_cols = ["volume", "total_depth", "bid_qty", "ask_qty", "buy_qty", "sell_qty"]
    df = df.with_columns([
        pl.col(col).log1p().alias(f"log_{col}") for col in log_cols
    ])
    
    # Square root transformations
    sqrt_cols = ["volume", "total_depth", "bid_qty", "ask_qty"]
    df = df.with_columns([
        pl.col(col).sqrt().alias(f"sqrt_{col}") for col in sqrt_cols
    ])
    
    # Power transformations
    df = df.with_columns([
        (pl.col("order_book_imbalance") ** 2).alias("order_book_imbalance_sq"),
        (pl.col("trade_flow_imbalance") ** 2).alias("trade_flow_imbalance_sq"),
        pl.col("volume").pow(0.25).alias("volume_pow_quarter"),
    ])
    
    # Statistical distance features
    volume_stats = {
        'mean': df["volume"].mean(),
        'std': df["volume"].std(),
        'median': df["volume"].quantile(0.50)
    }
    
    df = df.with_columns([
        ((pl.col("volume") - volume_stats['mean']) / 
         (volume_stats['std'] + 1e-10)).alias("volume_zscore"),
        (pl.col("volume") / (volume_stats['median'] + 1e-10)).alias("volume_ratio_median"),
    ])
    
    # Interaction features
    df = df.with_columns([
        (pl.col("order_book_imbalance") * 
         pl.col("trade_flow_imbalance")).alias("imbalance_interaction"),
        (pl.col("buy_pressure") * 
         pl.col("volume_ratio_median")).alias("buy_pressure_volume_interaction"),
        (pl.col("kyle_lambda_proxy") * 
         pl.col("amihud_proxy")).alias("impact_illiquidity_interaction"),
    ])
    
    # Clean data
    df = df.fill_null(0)
    for col in df.columns:
        if col not in ["timestamp", "label"]:
            df = df.with_columns(
                pl.col(col).fill_nan(0).alias(col)
            )
    
    return df

# =========================
# Feature Selection
# =========================
class AdvancedFeatureSelector:
    """Advanced feature selection with multiple criteria"""
    
    def __init__(self, min_variance=0.01, correlation_threshold=0.95, 
                 max_features=300, min_features=50):
        self.min_variance = min_variance
        self.correlation_threshold = correlation_threshold
        self.max_features = max_features
        self.min_features = min_features
        self.selected_features = None
        self.feature_scores = None
        
    def fit_transform(self, X, y, feature_names):
        """Select features based on multiple criteria"""
        print("\nPerforming advanced feature selection...")
        n_features_original = X.shape[1]
        
        # Step 1: Remove constant features
        constant_mask = np.var(X, axis=0) > 0
        X = X[:, constant_mask]
        feature_names = [f for f, m in zip(feature_names, constant_mask) if m]
        print(f"Removed {n_features_original - len(feature_names)} constant features")
        
        # Step 2: Remove low variance features
        if X.shape[1] > self.min_features:
            variances = np.var(X, axis=0)
            variance_threshold = np.percentile(variances, 10)  # Keep top 90%
            variance_threshold = max(variance_threshold, self.min_variance)
            
            variance_mask = variances > variance_threshold
            X = X[:, variance_mask]
            feature_names = [f for f, m in zip(feature_names, variance_mask) if m]
            print(f"Removed {sum(~variance_mask)} low variance features")
        
        # Step 3: Remove highly correlated features
        if X.shape[1] > self.min_features:
            correlation_matrix = np.corrcoef(X.T)
            upper_triangle = np.triu(np.ones_like(correlation_matrix, dtype=bool), k=1)
            high_corr_pairs = np.where((np.abs(correlation_matrix) > self.correlation_threshold) & upper_triangle)
            
            features_to_remove = set()
            for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]):
                if i not in features_to_remove and j not in features_to_remove:
                    # Keep feature with higher correlation to target
                    corr_i = abs(pearsonr(X[:, i], y)[0])
                    corr_j = abs(pearsonr(X[:, j], y)[0])
                    if corr_i < corr_j:
                        features_to_remove.add(i)
                    else:
                        features_to_remove.add(j)
            
            keep_mask = np.array([i not in features_to_remove for i in range(X.shape[1])])
            X = X[:, keep_mask]
            feature_names = [f for f, m in zip(feature_names, keep_mask) if m]
            print(f"Removed {len(features_to_remove)} highly correlated features")
        
        # Step 4: Score remaining features
        print("\nScoring remaining features...")
        feature_scores = []
        for i in range(X.shape[1]):
            # Calculate multiple metrics
            variance = np.var(X[:, i])
            correlation = abs(pearsonr(X[:, i], y)[0])
            mutual_info_approx = variance * correlation  # Simplified mutual information proxy
            
            # Combined score
            score = mutual_info_approx
            feature_scores.append((feature_names[i], score, variance, correlation))
        
        # Sort by score
        feature_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top features
        n_select = min(self.max_features, len(feature_scores))
        n_select = max(n_select, self.min_features)
        
        self.selected_features = [f[0] for f in feature_scores[:n_select]]
        self.feature_scores = feature_scores[:n_select]
        
        # Print top features
        print(f"\nSelected {len(self.selected_features)} features from {n_features_original}")
        print("\nTop 15 features:")
        for i, (name, score, var, corr) in enumerate(self.feature_scores[:15]):
            print(f"  {i+1:2d}. {name:25s} score={score:.4f}, var={var:.4f}, corr={corr:.4f}")
        
        return self.selected_features
    
    def transform(self, df, feature_names):
        """Transform dataframe to selected features"""
        if self.selected_features is None:
            raise ValueError("Must call fit_transform first")
        
        # Find indices of selected features
        indices = [i for i, f in enumerate(feature_names) if f in self.selected_features]
        return df[:, indices] if isinstance(df, np.ndarray) else df[self.selected_features]

# =========================
# Neural Network Components
# =========================
class MLP(nn.Module):
    def __init__(self, layers, dropout_rate=0.6, activation='relu', 
                 last_activation=None, use_batch_norm=True):
        """Enhanced MLP with optional batch normalization"""
        super(MLP, self).__init__()
        
        self.layers = nn.ModuleList()
        self.use_batch_norm = use_batch_norm
        
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            
            if i < len(layers) - 2:  # Not the last layer
                if use_batch_norm:
                    self.layers.append(nn.BatchNorm1d(layers[i + 1]))
                self.layers.append(get_activation_function(activation))
                self.layers.append(nn.Dropout(dropout_rate))
        
        if last_activation is not None:
            self.layers.append(get_activation_function(last_activation))
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Checkpointer:
    def __init__(self, path="best_model.pt"):
        self.path = path
        self.best_pearson = -np.inf
        self.patience_counter = 0
    
    def __call__(self, pearson_coef, model, patience=None):
        """Save model if performance improves"""
        if pearson_coef > self.best_pearson:
            self.best_pearson = pearson_coef
            self.patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'pearson': pearson_coef
            }, self.path)
            print(f"âœ… New best model saved with Pearson: {pearson_coef:.4f}")
            return True
        else:
            self.patience_counter += 1
            if patience and self.patience_counter >= patience:
                print(f"Early stopping triggered after {patience} epochs without improvement")
                return False
        return None
    
    def load(self, model):
        """Load the best model weights"""
        checkpoint = torch.load(self.path)
        model.load_state_dict(checkpoint['model_state_dict'])
        self.best_pearson = checkpoint['pearson']
        print(f"Model loaded from {self.path} with Pearson: {self.best_pearson:.4f}")
        return model

# =========================
# Enhanced Neural Architecture Search
# =========================
class EnhancedNeuralArchitectureSearch:
    def __init__(self, input_dim, feature_selector, search_space=None):
        self.input_dim = input_dim
        self.feature_selector = feature_selector
        self.best_architecture = None
        self.best_score = -np.inf
        self.search_history = []
        
        # Enhanced search space
        if search_space is None:
            self.search_space = {
                'num_layers': [3, 4, 5, 6],
                'layer_sizes': [64, 128, 256, 512, 1024],
                'activation': ['relu', 'leakyrelu', 'elu', 'gelu', 'selu'],
                'dropout_rate': [0.3, 0.4, 0.5, 0.6, 0.7],
                'learning_rate': [0.0001, 0.0005, 0.001, 0.002],
                'weight_decay': [1e-4, 5e-4, 1e-3, 5e-3],
                'batch_size': [1024*8, 1024*16, 1024*32],
                'noise_factor': [0, 0.001, 0.005, 0.01],
                'use_batch_norm': [True, False],
                'optimizer_type': ['adam', 'adamw', 'sgd'],
                'scheduler_type': ['none', 'cosine', 'step']
            }
        else:
            self.search_space = search_space
    
    def generate_architecture(self, method='random'):
        """Generate a new architecture"""
        if method == 'random':
            return self._random_architecture()
        elif method == 'evolutionary':
            return self._evolutionary_architecture()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _random_architecture(self):
        """Generate random architecture"""
        arch = {}
        
        # Sample hyperparameters
        for param, values in self.search_space.items():
            if param not in ['layer_sizes', 'num_layers']:
                arch[param] = random.choice(values)
        
        # Generate layer configuration
        arch['num_layers'] = random.choice(self.search_space['num_layers'])
        layers = [self.input_dim]
        
        # Decreasing layer sizes
        prev_size = self.input_dim
        for i in range(arch['num_layers'] - 1):
            candidates = [s for s in self.search_space['layer_sizes'] if s <= prev_size]
            if not candidates:
                candidates = self.search_space['layer_sizes']
            layer_size = random.choice(candidates)
            layers.append(layer_size)
            prev_size = layer_size
        
        layers.append(1)  # Output
        arch['layers'] = layers
        
        return arch
    
    def _evolutionary_architecture(self):
        """Generate architecture using evolutionary strategy"""
        if len(self.search_history) < 5:
            return self._random_architecture()
        
        # Tournament selection
        tournament_size = min(5, len(self.search_history))
        tournament = random.sample(self.search_history, tournament_size)
        parent = max(tournament, key=lambda x: x['score'])['architecture']
        
        # Create child with mutations
        child = parent.copy()
        child['layers'] = parent['layers'].copy()
        
        # Mutate 1-3 parameters
        n_mutations = random.randint(1, 3)
        for _ in range(n_mutations):
            param = random.choice(list(self.search_space.keys()))
            
            if param in ['layer_sizes', 'num_layers']:
                # Mutate architecture
                child['num_layers'] = random.choice(self.search_space['num_layers'])
                layers = [self.input_dim]
                for i in range(child['num_layers'] - 1):
                    if random.random() < 0.7 and i+1 < len(child['layers'])-1:
                        layers.append(child['layers'][i+1])
                    else:
                        layers.append(random.choice(self.search_space['layer_sizes']))
                layers.append(1)
                child['layers'] = layers
            else:
                child[param] = random.choice(self.search_space[param])
        
        return child
    
    def train_and_evaluate(self, architecture, X_train, Y_train, X_val, Y_val, epochs=10):
        """Train and evaluate architecture"""
        set_seed(Config.SEED)
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(Y_train.values, dtype=torch.float32).unsqueeze(1)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(Y_val.values, dtype=torch.float32).unsqueeze(1)
        )
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=architecture['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=architecture['batch_size'], 
            shuffle=False
        )
        
        # Initialize model
        model = MLP(
            layers=architecture['layers'],
            dropout_rate=architecture['dropout_rate'],
            activation=architecture['activation'],
            use_batch_norm=architecture.get('use_batch_norm', True)
        ).to(device)
        
        # Loss and optimizer
        criterion = nn.HuberLoss(delta=5.0)
        
        if architecture.get('optimizer_type', 'adam') == 'adam':
            optimizer = optim.Adam(
                model.parameters(), 
                lr=architecture['learning_rate'],
                weight_decay=architecture['weight_decay']
            )
        elif architecture['optimizer_type'] == 'adamw':
            optimizer = optim.AdamW(
                model.parameters(), 
                lr=architecture['learning_rate'],
                weight_decay=architecture['weight_decay']
            )
        else:
            optimizer = optim.SGD(
                model.parameters(), 
                lr=architecture['learning_rate'],
                weight_decay=architecture['weight_decay'],
                momentum=0.9
            )
        
        # Learning rate scheduler
        if architecture.get('scheduler_type', 'none') == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif architecture['scheduler_type'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
        else:
            scheduler = None
        
        # Training loop
        best_val_pearson = -np.inf
        
        for epoch in range(epochs):
            # Training
            model.train()
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Add noise
                if architecture['noise_factor'] > 0:
                    inputs = inputs + torch.randn_like(inputs) * architecture['noise_factor']
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            if scheduler:
                scheduler.step()
            
            # Validation
            model.eval()
            val_preds = []
            val_true = []
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    val_preds.extend(outputs.cpu().numpy())
                    val_true.extend(targets.cpu().numpy())
            
            val_preds = np.array(val_preds).flatten()
            val_true = np.array(val_true).flatten()
            pearson = pearsonr(val_preds, val_true)[0]
            
            if pearson > best_val_pearson:
                best_val_pearson = pearson
        
        return best_val_pearson, model
    
    def search(self, X_train, Y_train, X_val, Y_val, n_iterations=30, 
               method='evolutionary', epochs_per_arch=10, patience=10):
        """Perform architecture search"""
        print(f"\nStarting Enhanced Neural Architecture Search")
        print(f"Method: {method}, Iterations: {n_iterations}, Patience: {patience}")
        print("="*60)
        
        no_improvement = 0
        
        for i in range(n_iterations):
            # Generate architecture
            if i < 5:  # Start with random exploration
                arch = self.generate_architecture('random')
            else:
                arch = self.generate_architecture(method)
            
            print(f"\nIteration {i+1}/{n_iterations}")
            print(f"Architecture: {arch['num_layers']} layers, "
                  f"sizes={arch['layers'][1:-1]}, "
                  f"activation={arch['activation']}, "
                  f"dropout={arch['dropout_rate']}")
            
            # Train and evaluate
            score, model = self.train_and_evaluate(
                arch, X_train, Y_train, X_val, Y_val, epochs_per_arch
            )
            
            print(f"Validation Pearson: {score:.4f}")
            
            # Update history
            self.search_history.append({
                'architecture': arch,
                'score': score,
                'model': model
            })
            
            # Check for improvement
            if score > self.best_score:
                self.best_score = score
                self.best_architecture = arch
                no_improvement = 0
                print(f"ðŸŽ¯ New best architecture! Score: {score:.4f}")
            else:
                no_improvement += 1
            
            # Early stopping
            if no_improvement >= patience:
                print(f"\nEarly stopping: No improvement for {patience} iterations")
                break
        
        print(f"\nâœ… Search complete. Best score: {self.best_score:.4f}")
        return self.best_architecture, self.best_score

# =========================
# Main Pipeline
# =========================
def main():
    """Main execution pipeline"""
    print("DRW Crypto Market Prediction - Enhanced NAS with Feature Selection")
    print("="*70)
    
    set_seed(Config.SEED)
    
    # Load training data
    print("\nLoading training data...")
    train = pl.scan_parquet(Config.TRAIN_PATH)
    
    # Get all columns except timestamp
    all_columns = train.collect_schema().names()
    feature_columns = [col for col in all_columns if col not in ['timestamp', 'label']]
    
    # Load with all features plus label
    train = shrink_dtypes(train.select(feature_columns + ['label'])).collect()
    print(f"Initial shape: {train.shape}")
    
    # Create microstructure features
    train = create_microstructure_features(train)
    
    # Get updated feature list
    feature_cols = [col for col in train.columns if col != 'label']
    print(f"Total features after engineering: {len(feature_cols)}")
    
    # Convert to pandas
    train_pd = train.to_pandas()
    
    # Store statistics for test data processing
    train_stats = {}
    for col in feature_cols:
        train_stats[col] = {
            'median': train_pd[col].median(),
            'q01': train_pd[col].quantile(0.01),
            'q99': train_pd[col].quantile(0.99)
        }
    
    # Handle outliers
    print("\nHandling outliers...")
    for col in feature_cols:
        train_pd[col] = train_pd[col].replace([np.inf, -np.inf], np.nan)
        train_pd[col] = train_pd[col].fillna(train_stats[col]['median'])
        train_pd[col] = train_pd[col].clip(
            lower=train_stats[col]['q01'], 
            upper=train_stats[col]['q99']
        )
    
    # Split data
    print("\nSplitting data...")
    X_train, X_val = train_test_split(
        train_pd, test_size=0.2, shuffle=False, random_state=Config.SEED
    )
    
    Y_train = X_train.pop("label")
    Y_val = X_val.pop("label")
    
    # Advanced feature selection
    feature_selector = AdvancedFeatureSelector(
        min_variance=Config.MIN_VARIANCE_THRESHOLD,
        correlation_threshold=Config.CORRELATION_THRESHOLD,
        max_features=Config.MAX_FEATURES_TO_TEST,
        min_features=Config.MIN_FEATURES_TO_KEEP
    )
    
    selected_features = feature_selector.fit_transform(
        X_train[feature_cols].values, 
        Y_train.values, 
        feature_cols
    )
    
    # Apply feature selection
    X_train_selected = X_train[selected_features].values
    X_val_selected = X_val[selected_features].values
    
    # Scale features
    print("\nScaling features...")
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    
    print(f"\nFinal training shape: {X_train_scaled.shape}")
    
    # Neural Architecture Search
    nas = EnhancedNeuralArchitectureSearch(
        input_dim=X_train_scaled.shape[1],
        feature_selector=feature_selector
    )
    
    best_arch, best_score = nas.search(
        X_train_scaled, Y_train, X_val_scaled, Y_val,
        n_iterations=Config.NAS_ITERATIONS,
        method='evolutionary',
        epochs_per_arch=Config.EPOCHS_PER_ARCH,
        patience=Config.NAS_PATIENCE
    )
    
    print("\n" + "="*70)
    print("BEST ARCHITECTURE FOUND:")
    print("="*70)
    for key, value in best_arch.items():
        if key != 'layers':
            print(f"{key:20s}: {value}")
    print(f"{'layers':20s}: {best_arch['layers']}")
    print(f"{'best_score':20s}: {best_score:.4f}")
    
    # Train final model with best architecture
    print("\n" + "="*70)
    print("TRAINING FINAL MODEL")
    print("="*70)
    
    # Combine train and validation for final training
    X_full = np.vstack([X_train_scaled, X_val_scaled])
    Y_full = pd.concat([Y_train, Y_val])
    
    # Create data loaders
    full_dataset = TensorDataset(
        torch.tensor(X_full, dtype=torch.float32),
        torch.tensor(Y_full.values, dtype=torch.float32).unsqueeze(1)
    )
    
    train_loader = DataLoader(
        full_dataset,
        batch_size=best_arch['batch_size'],
        shuffle=True
    )
    
    # Initialize final model
    final_model = MLP(
        layers=best_arch['layers'],
        dropout_rate=best_arch['dropout_rate'],
        activation=best_arch['activation'],
        use_batch_norm=best_arch.get('use_batch_norm', True)
    ).to(device)
    
    # Training setup
    criterion = nn.HuberLoss(delta=5.0)
    optimizer = optim.AdamW(
        final_model.parameters(),
        lr=best_arch['learning_rate'],
        weight_decay=best_arch['weight_decay']
    )
    
    checkpointer = Checkpointer(path="final_model.pt")
    
    # Train for more epochs
    print("\nTraining final model...")
    for epoch in range(30):
        final_model.train()
        running_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/30"):
            inputs, targets = inputs.to(device), targets.to(device)
            
            if best_arch['noise_factor'] > 0:
                inputs = inputs + torch.randn_like(inputs) * best_arch['noise_factor']
            
            optimizer.zero_grad()
            outputs = final_model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Generate test predictions
    print("\n" + "="*70)
    print("GENERATING TEST PREDICTIONS")
    print("="*70)
    
    # Load test data
    test = pl.scan_parquet(Config.TEST_PATH)
    test = shrink_dtypes(test.select(feature_columns)).collect()
    
    # Create same features
    test = create_microstructure_features(test)
    test_pd = test.select(feature_cols).to_pandas()
    
    # Apply same preprocessing
    for col in feature_cols:
        test_pd[col] = test_pd[col].replace([np.inf, -np.inf], np.nan)
        test_pd[col] = test_pd[col].fillna(train_stats[col]['median'])
        test_pd[col] = test_pd[col].clip(
            lower=train_stats[col]['q01'],
            upper=train_stats[col]['q99']
        )
    
    # Select and scale features
    X_test = test_pd[selected_features].values
    X_test_scaled = scaler.transform(X_test)
    
    # Create test loader
    test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))
    test_loader = DataLoader(test_dataset, batch_size=best_arch['batch_size'], shuffle=False)
    
    # Generate predictions
    final_model.eval()
    predictions = []
    
    with torch.no_grad():
        for inputs in tqdm(test_loader, desc="Predicting"):
            inputs = inputs[0].to(device)
            outputs = final_model(inputs)
            predictions.extend(outputs.cpu().numpy())
    
    predictions = np.array(predictions).flatten()
    
    # Create submission
    submission = pd.read_csv(Config.SUBMISSION_PATH)
    submission["prediction"] = predictions
    submission.to_csv("submission.csv", index=False)
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    print(f"Selected features: {len(selected_features)}")
    print(f"Best validation score: {best_score:.4f}")
    print(f"Test predictions: {len(predictions)}")
    print(f"\nPrediction statistics:")
    print(f"  Mean: {predictions.mean():.4f}")
    print(f"  Std:  {predictions.std():.4f}")
    print(f"  Min:  {predictions.min():.4f}")
    print(f"  Max:  {predictions.max():.4f}")
    print("\nSubmission saved to: submission.csv")

if __name__ == "__main__":
    main()

