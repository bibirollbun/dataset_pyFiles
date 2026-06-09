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


# Complete GANDALF Implementation for Crypto Market Prediction
# No placeholders - Full working code with all components

import os
import random
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tqdm import tqdm
import gc
from typing import List, Tuple, Dict, Optional, Any
import math
import json
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

from scipy.stats import pearsonr, spearmanr, kurtosis, skew
from scipy.special import erfinv

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create results directory
RESULTS_DIR = Path("gandalf_results")
RESULTS_DIR.mkdir(exist_ok=True)

# =========================
# Configuration
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
# Feature Engineering
# =========================
def add_market_features(df):
    """Add market microstructure features"""
    eps = 1e-10
    
    # Core features
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + eps)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + eps)
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + eps)
    
    # Volume features
    df['log_volume'] = np.log1p(df['volume'])
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + eps)
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + eps)
    
    # Liquidity metrics
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + eps)
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + eps)
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['bid_qty'] + df['ask_qty'] + eps)
    
    # Price pressure
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + eps)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + eps)
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + eps)
    
    # Market stress
    df['market_stress'] = df['volume'] / (df['total_depth'] + eps) * np.abs(df['order_flow_imbalance'])
    df['volatility_proxy'] = np.abs(df['net_order_flow']) / (df['total_depth'] + eps) * df['volume']
    
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df

def set_seed(seed=42):
    """Set random seeds"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# =========================
# GANDALF Components
# =========================
class DifferentiableDecisionTree(nn.Module):
    """Soft decision tree for GANDALF"""
    def __init__(self, input_dim, depth, temperature=1.0):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        
        # Internal nodes
        self.internal_nodes = nn.ModuleList()
        for i in range(2 ** depth - 1):
            self.internal_nodes.append(nn.Linear(input_dim, 1))
        
        # Leaf values
        self.leaf_values = nn.Parameter(torch.zeros(self.n_leaves))
        nn.init.normal_(self.leaf_values, mean=0, std=0.01)
        
        # Temperature
        self.temperature = nn.Parameter(torch.tensor(temperature))
        
    def forward(self, x):
        batch_size = x.size(0)
        device = x.device
        
        # Temperature with lower bound
        temp = F.softplus(self.temperature) + 0.1
        
        # Path probabilities
        path_probs = torch.ones(batch_size, 1, device=device)
        
        # Traverse tree
        for level in range(self.depth):
            n_nodes = 2 ** level
            next_path_probs = []
            
            for node in range(n_nodes):
                node_idx = 2 ** level - 1 + node
                
                if node_idx < len(self.internal_nodes):
                    # Split decision
                    logit = self.internal_nodes[node_idx](x).squeeze(-1)
                    split_prob = torch.sigmoid(logit / temp)
                    
                    # Update paths
                    if node < path_probs.size(1):
                        current_prob = path_probs[:, node:node+1]
                        next_path_probs.append(current_prob * (1 - split_prob).unsqueeze(1))
                        next_path_probs.append(current_prob * split_prob.unsqueeze(1))
            
            if next_path_probs:
                path_probs = torch.cat(next_path_probs, dim=1)
        
        # Ensure correct number of leaves
        if path_probs.size(1) > self.n_leaves:
            path_probs = path_probs[:, :self.n_leaves]
        
        # Weighted sum
        output = torch.sum(path_probs * self.leaf_values.unsqueeze(0), dim=1)
        
        return output

class GatingNetwork(nn.Module):
    """Gating network for tree selection"""
    def __init__(self, input_dim, n_trees, hidden_dim=128, dropout=0.3):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_trees)
        )
        
    def forward(self, x):
        gates = self.network(x)
        return F.softmax(gates, dim=-1)

class GANDALF(nn.Module):
    """GANDALF: Gated Additive Neural Decision Additive Forest"""
    def __init__(self, config):
        super().__init__()
        
        self.n_trees = config['n_trees']
        self.tree_depth = config['tree_depth']
        self.input_dim = config['input_dim']
        
        # Feature embedding
        embed_layers = []
        prev_dim = self.input_dim
        
        for dim in config['embed_dims']:
            embed_layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.GELU(),
                nn.Dropout(config['feature_dropout'])
            ])
            prev_dim = dim
            
        self.feature_embedder = nn.Sequential(*embed_layers)
        self.embed_dim = prev_dim
        
        # Decision trees
        self.trees = nn.ModuleList()
        for i in range(self.n_trees):
            # Vary depth for diversity
            tree_depth = self.tree_depth + (i % 3 - 1)
            tree_depth = max(2, tree_depth)
            
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
                    nn.LayerNorm(dim),
                    nn.GELU(),
                    nn.Dropout(config['head_dropout'])
                ])
                prev_dim = dim
                
            head_layers.append(nn.Linear(prev_dim, 1))
            self.nn_head = nn.Sequential(*head_layers)
            
            # Combination weight
            self.combination_weight = nn.Parameter(torch.tensor(0.5))
        else:
            self.nn_head = None
            
    def forward(self, x):
        # Embed features
        embedded = self.feature_embedder(x)
        
        # Get tree outputs
        tree_outputs = []
        for tree in self.trees:
            output = tree(embedded)
            tree_outputs.append(output)
        
        tree_outputs = torch.stack(tree_outputs, dim=1)
        
        # Get gates
        gates = self.gating_network(x)
        
        # Weighted combination
        forest_output = torch.sum(gates * tree_outputs, dim=1, keepdim=True)
        
        # Combine with neural head if available
        if self.nn_head is not None:
            nn_output = self.nn_head(x)
            weight = torch.sigmoid(self.combination_weight)
            final_output = weight * forest_output + (1 - weight) * nn_output
        else:
            final_output = forest_output
            
        return final_output

# =========================
# Training Functions
# =========================
def train_gandalf(model, train_loader, val_loader, config, device):
    """Train GANDALF model"""
    
    # Loss and optimizer
    criterion = nn.HuberLoss(delta=config.get('huber_delta', 1.0))
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    
    # Training settings
    best_val_pearson = -np.inf
    patience_counter = 0
    patience = config.get('patience', 10)
    num_epochs = config.get('num_epochs', 50)
    
    # Mixed precision
    use_amp = config.get('use_amp', True) and device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Add noise
            if config.get('noise_factor', 0) > 0:
                noise = torch.randn_like(inputs) * config['noise_factor']
                inputs = inputs + noise
            
            optimizer.zero_grad()
            
            if use_amp:
                with autocast():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
                
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.get('grad_clip', 1.0))
                optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
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
        
        # Metrics
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / len(val_loader)
        val_pearson = pearsonr(val_targets, val_preds)[0]
        val_spearman = spearmanr(val_targets, val_preds)[0]
        
        print(f"\nTrain Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Val Pearson: {val_pearson:.4f}, Val Spearman: {val_spearman:.4f}")
        
        # Update scheduler
        scheduler.step(val_pearson)
        
        # Save best model
        if val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            patience_counter = 0
            torch.save(model.state_dict(), RESULTS_DIR / "best_gandalf.pt")
            print(f"✅ New best model saved! Pearson: {best_val_pearson:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load(RESULTS_DIR / "best_gandalf.pt"))
    
    return model, best_val_pearson

# =========================
# Main Execution
# =========================
def main():
    print("=== GANDALF for Crypto Market Prediction ===")
    
    # Set seed
    set_seed(42)
    
    # Load data
    print("\nLoading data...")
    train = pl.scan_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
    train = train.select(SELECTED_FEATURES).collect().to_pandas()
    
    print(f"Initial shape: {train.shape}")
    
    # Use recent data
    train_size = int(0.85 * len(train))
    train = train.iloc[-train_size:].reset_index(drop=True)
    print(f"Using last {train_size} samples")
    
    # Add features
    print("\nAdding market features...")
    train = add_market_features(train)
    
    # Get features
    all_features = [col for col in train.columns if col != 'label']
    print(f"Total features: {len(all_features)}")
    
    # Split data
    split_idx = int(0.8 * len(train))
    train_data = train[:split_idx].copy()
    val_data = train[split_idx:].copy()
    
    y_train = train_data.pop('label').values
    y_val = val_data.pop('label').values
    
    X_train = train_data[all_features].values
    X_val = val_data[all_features].values
    
    # Feature selection
    print("\nSelecting features...")
    selector = SelectKBest(score_func=mutual_info_regression, k=min(150, len(all_features)))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    
    selected_features = [all_features[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} features")
    
    # Transform data
    print("\nTransforming data...")
    transformer = QuantileTransformer(output_distribution='normal', random_state=42)
    X_train_transformed = transformer.fit_transform(X_train_selected)
    X_val_transformed = transformer.transform(X_val_selected)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_transformed, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_transformed, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # Model configurations to try
    configs = [
        {
            'n_trees': 32,
            'tree_depth': 5,
            'tree_temperature': 1.0,
            'embed_dims': [256, 256],
            'feature_dropout': 0.2,
            'gate_hidden_dim': 128,
            'gate_dropout': 0.2,
            'use_nn_head': True,
            'head_dims': [256, 128],
            'head_dropout': 0.3,
            'learning_rate': 0.0003,
            'weight_decay': 0.01,
            'huber_delta': 1.0,
            'noise_factor': 0.01,
            'grad_clip': 1.0,
            'num_epochs': 50,
            'patience': 10,
            'use_amp': True,
            'input_dim': X_train_transformed.shape[1]
        },
        {
            'n_trees': 24,
            'tree_depth': 6,
            'tree_temperature': 1.5,
            'embed_dims': [128, 256, 128],
            'feature_dropout': 0.3,
            'gate_hidden_dim': 256,
            'gate_dropout': 0.2,
            'use_nn_head': True,
            'head_dims': [256, 128, 64],
            'head_dropout': 0.4,
            'learning_rate': 0.0005,
            'weight_decay': 0.001,
            'huber_delta': 0.5,
            'noise_factor': 0.02,
            'grad_clip': 2.0,
            'num_epochs': 50,
            'patience': 10,
            'use_amp': True,
            'input_dim': X_train_transformed.shape[1]
        }
    ]
    
    # Train multiple models
    ensemble_models = []
    ensemble_scores = []
    
    for i, config in enumerate(configs):
        print(f"\n=== Training Model {i+1}/{len(configs)} ===")
        
        # Create model
        model = GANDALF(config).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        model, best_score = train_gandalf(model, train_loader, val_loader, config, device)
        
        ensemble_models.append(model)
        ensemble_scores.append(best_score)
        
        print(f"Model {i+1} best validation Pearson: {best_score:.4f}")
    
    # Test predictions
    print("\n=== Making Test Predictions ===")
    
    # Load test data
    test = pl.scan_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
    test_features = [f for f in SELECTED_FEATURES if f != "label"]
    test = test.select(test_features).collect().to_pandas()
    
    # Add features
    test = add_market_features(test)
    
    # Transform test data
    X_test = test[all_features].values
    X_test_selected = selector.transform(X_test)
    X_test_transformed = transformer.transform(X_test_selected)
    
    # Make predictions
    all_predictions = []
    
    for model in ensemble_models:
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
    
    # Ensemble predictions
    # Weighted average based on validation scores
    weights = np.array(ensemble_scores)
    weights = weights / weights.sum()
    
    final_predictions = np.zeros_like(all_predictions[0])
    for pred, weight in zip(all_predictions, weights):
        final_predictions += weight * pred
    
    # Post-processing
    pred_mean = y_train.mean()
    pred_std = y_train.std()
    final_predictions = np.clip(
        final_predictions,
        pred_mean - 4 * pred_std,
        pred_mean + 4 * pred_std
    )
    
    # Create submission
    submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
    submission["prediction"] = final_predictions
    submission.to_csv("submission_gandalf.csv", index=False)
    
    # Display results
    print("\n=== Final Results ===")
    print(f"Ensemble size: {len(ensemble_models)} models")
    print(f"Ensemble weights: {weights}")
    print(f"Features used: {len(selected_features)}")
    print(f"\nPrediction statistics:")
    print(f"  Mean: {final_predictions.mean():.6f}")
    print(f"  Std: {final_predictions.std():.6f}")
    print(f"  Min: {final_predictions.min():.6f}")
    print(f"  Max: {final_predictions.max():.6f}")
    print(f"  Skewness: {skew(final_predictions):.4f}")
    print(f"  Kurtosis: {kurtosis(final_predictions):.4f}")
    
    # Plot predictions distribution
    plt.figure(figsize=(10, 6))
    plt.subplot(1, 2, 1)
    plt.hist(final_predictions, bins=50, alpha=0.7, edgecolor='black')
    plt.xlabel('Predicted Values')
    plt.ylabel('Frequency')
    plt.title('Test Predictions Distribution')
    
    plt.subplot(1, 2, 2)
    plt.hist(y_train, bins=50, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Target Values')
    plt.ylabel('Frequency')
    plt.title('Training Target Distribution')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'predictions_distribution.png', dpi=300)
    plt.close()
    
    print(f"\nResults saved to: {RESULTS_DIR}")
    print("Submission saved as 'submission_gandalf.csv'")
    print("✅ GANDALF training complete!")

if __name__ == "__main__":
    main()

