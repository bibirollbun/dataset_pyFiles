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


# DOFEN (Deep Oblivious Forest ENsemble) Feature Importance Analysis - FIXED VERSION
# State-of-the-art tree-neural hybrid for tabular data

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import warnings
import os
import math

warnings.filterwarnings("ignore")

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Feature Engineering Function
# =========================
def add_features(df):
    """Add all engineered features with numerical stability"""
    eps = 1e-8
    
    # Make a copy to avoid modifying original
    df = df.copy()
    
    # Original features
    df['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    df['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    df['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    df['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    df['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']

    df['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + eps)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + eps)
    df['log_volume'] = np.log1p(df['volume'])

    df['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + eps)
    df['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + eps)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + eps)
    df['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + eps)
    
    # Price Pressure Indicators
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + eps)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + eps)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    
    # Liquidity Depth Measures
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + eps)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + eps)
    df['log_depth'] = np.log1p(df['total_depth'])
    
    # Order Flow Toxicity Proxies
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + eps)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * np.log1p(df['volume'])
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + eps)
    
    # Market Activity Indicators
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + eps)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + eps)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # Microstructure Volatility Proxies
    df['realized_spread_proxy'] = 2 * np.abs(df['net_order_flow']) / (df['volume'] + eps)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + eps)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])
    
    # Complex Interaction Terms
    df['flow_depth_interaction'] = df['net_order_flow'] * np.log1p(df['total_depth'])
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * np.log1p(df['volume'])
    df['depth_volume_interaction'] = np.log1p(df['total_depth']) * np.log1p(df['volume'])
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    
    # Information Asymmetry Measures
    df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + eps)
    df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + eps)
    df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] + eps) * np.log1p(df['volume'])
    
    # Market Efficiency Indicators
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + eps)
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + eps)
    df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + eps)
    
    # Non-linear Transformations
    df['sqrt_volume'] = np.sqrt(df['volume'] + eps)
    df['sqrt_depth'] = np.sqrt(df['total_depth'] + eps)
    df['volume_squared'] = np.minimum(df['volume'] ** 2, 1e10)
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    
    # Relative Measures
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + eps)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + eps)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + eps)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + eps)
    
    # Market Stress Indicators
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + eps)
    df['market_stress'] = df['volume'] / (df['total_depth'] + eps) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + eps)
    
    # Directional Indicators
    df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + eps)
    df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
    df['signed_volume'] = np.sign(df['net_order_flow']) * np.log1p(df['volume'])
    
    # Clip extreme values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col not in ['label', 'timestamp']:
            df[col] = np.clip(df[col], -1e6, 1e6)
    
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df

# =========================
# FIXED DOFEN Model Components
# =========================

class StableObliviousDecisionTree(nn.Module):
    """Numerically stable Differentiable Oblivious Decision Tree"""
    def __init__(self, depth, num_features, temperature=1.0):
        super().__init__()
        self.depth = depth
        self.num_features = num_features
        self.temperature = max(temperature, 0.1)  # Ensure minimum temperature
        self.num_leaves = 2 ** depth
        
        # Better initialization for stability
        self.feature_indices = nn.Parameter(torch.randn(depth, num_features) * 0.1)
        self.thresholds = nn.Parameter(torch.zeros(depth))
        self.leaf_values = nn.Parameter(torch.randn(self.num_leaves) * 0.1)
        
        # Add batch normalization for stability
        self.feature_norm = nn.BatchNorm1d(num_features)
        
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Normalize features for stability
        x_norm = self.feature_norm(x)
        
        # Compute leaf probabilities with numerical stability
        leaf_log_probs = torch.zeros(batch_size, self.num_leaves, device=x.device)
        
        for d in range(self.depth):
            # Stable feature selection using log-softmax
            feature_log_weights = F.log_softmax(self.feature_indices[d] / self.temperature, dim=0)
            feature_weights = torch.exp(feature_log_weights)
            
            # Select features with stability
            selected_feature = (x_norm * feature_weights.unsqueeze(0)).sum(dim=1)
            
            # Stable sigmoid computation
            decision_logit = (selected_feature - self.thresholds[d]) / self.temperature
            decision_logit = torch.clamp(decision_logit, -10, 10)  # Prevent overflow
            decision = torch.sigmoid(decision_logit)
            
            # Update leaf probabilities in log space for stability
            for leaf in range(self.num_leaves):
                if (leaf >> (self.depth - 1 - d)) & 1:
                    leaf_log_probs[:, leaf] += torch.log(decision + 1e-8)
                else:
                    leaf_log_probs[:, leaf] += torch.log(1 - decision + 1e-8)
        
        # Convert back from log space
        leaf_probs = torch.exp(leaf_log_probs)
        leaf_probs = leaf_probs / (leaf_probs.sum(dim=1, keepdim=True) + 1e-8)
        
        # Compute output with bounded leaf values
        bounded_leaf_values = torch.tanh(self.leaf_values)
        output = (leaf_probs * bounded_leaf_values.unsqueeze(0)).sum(dim=1)
        
        return output
    
    def get_feature_importance(self):
        """Extract feature importance from the tree"""
        importance = torch.zeros(self.num_features)
        for d in range(self.depth):
            feature_probs = F.softmax(self.feature_indices[d] / self.temperature, dim=0)
            importance += feature_probs.detach().cpu()
        return importance.numpy() / self.depth

class SimplifiedDOFEN(nn.Module):
    """Simplified and stable DOFEN model"""
    def __init__(self, num_features, num_trees=20, tree_depth=3, temperature=2.0, dropout=0.3):
        super().__init__()
        self.num_features = num_features
        self.num_trees = num_trees
        self.tree_depth = tree_depth
        
        # Input normalization
        self.input_norm = nn.BatchNorm1d(num_features)
        self.input_dropout = nn.Dropout(dropout)
        
        # Forest of trees
        self.trees = nn.ModuleList([
            StableObliviousDecisionTree(tree_depth, num_features, temperature)
            for _ in range(num_trees)
        ])
        
        # Tree combination
        self.tree_weights = nn.Parameter(torch.ones(num_trees) / num_trees)
        
        # Output head with residual connections
        self.output_head = nn.Sequential(
            nn.Linear(num_trees + 1, 32),  # +1 for residual
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
        # Direct linear path for residual
        self.direct_linear = nn.Linear(num_features, 1)
        
        # Initialize weights properly
        self._init_weights()
        
    def _init_weights(self):
        """Initialize weights for stability"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input normalization and dropout
        x_norm = self.input_norm(x)
        x_norm = self.input_dropout(x_norm)
        
        # Get tree outputs
        tree_outputs = []
        for tree in self.trees:
            tree_out = tree(x_norm)
            tree_outputs.append(tree_out)
        
        # Stack tree outputs
        tree_outputs = torch.stack(tree_outputs, dim=1)  # [batch, num_trees]
        
        # Direct linear path (residual connection)
        direct_out = self.direct_linear(x_norm)
        
        # Combine tree outputs with residual
        combined = torch.cat([tree_outputs, direct_out], dim=1)  # [batch, num_trees + 1]
        
        # Final output
        output = self.output_head(combined)
        
        return output.squeeze(-1)
    
    def get_tree_importance(self):
        """Get feature importance from all trees"""
        importance = np.zeros(self.num_features)
        tree_weights = F.softmax(self.tree_weights, dim=0).detach().cpu().numpy()
        
        for i, tree in enumerate(self.trees):
            tree_importance = tree.get_feature_importance()
            importance += tree_weights[i] * tree_importance
        
        return importance
    
    def get_gradient_importance(self, dataloader, criterion, device, max_batches=50):
        """Compute gradient-based feature importance"""
        self.eval()
        gradient_importance = torch.zeros(self.num_features)
        n_batches = 0
        
        for inputs, targets in dataloader:
            if n_batches >= max_batches:
                break
                
            inputs = inputs.to(device).requires_grad_(True)
            targets = targets.to(device)
            
            # Forward pass
            outputs = self(inputs)
            loss = criterion(outputs, targets.squeeze())
            
            # Backward pass
            loss.backward()
            
            # Accumulate absolute gradients
            if inputs.grad is not None:
                gradient_importance += torch.abs(inputs.grad).mean(dim=0).cpu()
            
            n_batches += 1
            
            # Clear gradients
            if inputs.grad is not None:
                inputs.grad.zero_()
        
        if n_batches > 0:
            gradient_importance /= n_batches
            
        return gradient_importance.numpy()

# =========================
# Training Function
# =========================

def train_dofen_fixed(model, train_loader, val_loader, n_epochs=20, learning_rate=0.001, patience=5):
    """Train DOFEN model with better stability"""
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(n_epochs):
        # Training
        model.train()
        train_loss = 0.0
        n_train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs}")
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device).squeeze()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Skip if loss is NaN
            if torch.isnan(loss):
                continue
            
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            n_train_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        n_val_batches = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device).squeeze()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                if not torch.isnan(loss):
                    val_loss += loss.item()
                    val_preds.extend(outputs.cpu().numpy())
                    val_targets.extend(targets.cpu().numpy())
                    n_val_batches += 1
        
        # Calculate metrics
        avg_train_loss = train_loss / n_train_batches if n_train_batches > 0 else float('inf')
        avg_val_loss = val_loss / n_val_batches if n_val_batches > 0 else float('inf')
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if len(val_preds) > 0 and len(val_targets) > 0:
            val_pearson = pearsonr(val_targets, val_preds)[0]
            if np.isnan(val_pearson):
                val_pearson = 0.0
        else:
            val_pearson = 0.0
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, "
              f"Val Loss = {avg_val_loss:.4f}, Val Pearson = {val_pearson:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_dofen_model.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    # Load best model
    if os.path.exists('best_dofen_model.pt'):
        model.load_state_dict(torch.load('best_dofen_model.pt'))
    
    return model, train_losses, val_losses

# =========================
# Fixed Visualization Functions
# =========================

def create_dofen_visualizations_fixed(importance_results, feature_names, save_prefix="dofen"):
    """Create visualizations with proper NaN handling"""
    
    # Create output directory
    os.makedirs('dofen_outputs', exist_ok=True)
    
    # 1. Combined importance horizontal bar plot
    plt.figure(figsize=(14, 10))
    top_n = 40
    
    combined_importance = importance_results['combined']
    # Handle NaN values
    combined_importance = np.nan_to_num(combined_importance, nan=0.0)
    
    top_indices = np.argsort(combined_importance)[-top_n:][::-1]
    
    top_features = [feature_names[i] for i in top_indices]
    top_importance = combined_importance[top_indices]
    
    # Color coding
    colors = []
    for f in top_features:
        if f.startswith('X'):
            colors.append('#FF6B6B')  # Red for anonymous
        elif f in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']:
            colors.append('#4ECDC4')  # Teal for market
        else:
            colors.append('#45B7D1')  # Blue for engineered
    
    plt.barh(range(len(top_features)), top_importance, color=colors)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Combined Importance Score', fontsize=12)
    plt.title(f'Top {top_n} Most Important Features (DOFEN)', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Anonymous Features (X_)'),
        Patch(facecolor='#4ECDC4', label='Market Features'),
        Patch(facecolor='#45B7D1', label='Engineered Features')
    ]
    plt.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.savefig(f'dofen_outputs/{save_prefix}_top_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Component importance comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Tree importance distribution
    ax = axes[0]
    tree_imp = np.nan_to_num(importance_results['tree_importance'], nan=0.0)
    valid_tree_imp = tree_imp[tree_imp > 0]
    
    if len(valid_tree_imp) > 0:
        ax.hist(valid_tree_imp, bins=30, alpha=0.7, color='forestgreen', edgecolor='black')
        ax.axvline(valid_tree_imp.mean(), color='darkgreen', linestyle='--',
                   label=f'Mean: {valid_tree_imp.mean():.4f}')
        ax.set_xlabel('Tree-based Importance')
        ax.set_ylabel('Frequency')
        ax.set_title('Tree Structure Importance Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Gradient importance distribution
    ax = axes[1]
    grad_imp = np.nan_to_num(importance_results['gradient_importance'], nan=0.0)
    valid_grad_imp = grad_imp[grad_imp > 0]
    
    if len(valid_grad_imp) > 0:
        ax.hist(valid_grad_imp, bins=30, alpha=0.7, color='lightcoral', edgecolor='black')
        ax.axvline(valid_grad_imp.mean(), color='darkred', linestyle='--',
                   label=f'Mean: {valid_grad_imp.mean():.4f}')
        ax.set_xlabel('Gradient-based Importance')
        ax.set_ylabel('Frequency')
        ax.set_title('Gradient Importance Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('DOFEN Feature Importance Components', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'dofen_outputs/{save_prefix}_components.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Feature category analysis
    plt.figure(figsize=(10, 8))
    
    # Categorize features
    anonymous_mask = [f.startswith('X') for f in feature_names]
    market_mask = [f in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume'] for f in feature_names]
    engineered_mask = [not (anonymous_mask[i] or market_mask[i]) for i in range(len(feature_names))]
    
    # Calculate average importance by category
    anonymous_imp = combined_importance[anonymous_mask].mean() if any(anonymous_mask) else 0
    market_imp = combined_importance[market_mask].mean() if any(market_mask) else 0
    engineered_imp = combined_importance[engineered_mask].mean() if any(engineered_mask) else 0
    
    categories = ['Anonymous\nFeatures', 'Market\nFeatures', 'Engineered\nFeatures']
    avg_importances = [anonymous_imp, market_imp, engineered_imp]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars = plt.bar(categories, avg_importances, color=colors, alpha=0.7, edgecolor='black')
    plt.ylabel('Average Importance Score', fontsize=12)
    plt.title('Average Feature Importance by Category', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_importances):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(f'dofen_outputs/{save_prefix}_category_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# =========================
# Main Analysis Function (Fixed)
# =========================

def analyze_dofen_feature_importance_fixed():
    """Fixed DOFEN feature importance analysis"""
    print("=== DOFEN Feature Importance Analysis (Fixed Version) ===")
    print("Deep Oblivious Forest ENsemble - Stabilized Implementation\n")
    
    # Data paths
    data_paths = [
        "/kaggle/input/drw-crypto-market-prediction/train.parquet",
        "../input/drw-crypto-market-prediction/train.parquet",
        "./data/train.parquet"
    ]
    
    train_path = None
    for path in data_paths:
        if os.path.exists(path):
            train_path = path
            break
    
    if train_path is None:
        train_path = data_paths[0]
    
    print(f"Data path: {train_path}")
    
    # Define features
    x_features = [f"X{i}" for i in range(1, 891)]
    market_features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
    
    # Load data
    try:
        print("Loading data...")
        train_df = pd.read_parquet(train_path)
        print(f"Loaded {len(train_df)} samples")
        
        # Check for NaN values in label
        if train_df['label'].isna().sum() > 0:
            print(f"Found {train_df['label'].isna().sum()} NaN values in label, removing...")
            train_df = train_df[~train_df['label'].isna()]
    except:
        print("Could not load data. Generating synthetic data...")
        # Generate synthetic data
        n_samples = 20000
        np.random.seed(42)
        
        data = {}
        # Create correlated features
        base_features = np.random.randn(n_samples, 20)
        
        for i, feat in enumerate(x_features):
            if i < 20:
                # Important features
                data[feat] = base_features[:, i % 20] + np.random.randn(n_samples) * 0.3
            elif i < 50:
                # Interaction features
                idx1, idx2 = i % 20, (i + 5) % 20
                data[feat] = base_features[:, idx1] * base_features[:, idx2] + np.random.randn(n_samples) * 0.5
            else:
                # Noise features
                data[feat] = np.random.randn(n_samples) * 0.5
        
        # Market features
        data['bid_qty'] = np.abs(np.random.lognormal(6, 1, n_samples))
        data['ask_qty'] = np.abs(np.random.lognormal(6, 1, n_samples))
        data['buy_qty'] = np.abs(np.random.lognormal(5, 1, n_samples))
        data['sell_qty'] = np.abs(np.random.lognormal(5, 1, n_samples))
        data['volume'] = data['buy_qty'] + data['sell_qty'] + np.abs(np.random.randn(n_samples) * 100)
        
        # Create label with tree-friendly patterns
        data['label'] = (
            0.3 * (data['X1'] > 0) * data['X1'] + 
            0.2 * (data['X2'] < 0) * data['X2'] + 
            0.15 * np.where(data['X3'] > np.median(data['X3']), data['X3'], -data['X3']) +
            0.1 * np.log1p(data['volume']) +
            0.05 * (data['buy_qty'] - data['sell_qty']) / (data['volume'] + 1) +
            np.random.randn(n_samples) * 0.1
        )
        
        train_df = pd.DataFrame(data)
    
    # Add engineered features
    print("\nAdding engineered features...")
    train_df = add_features(train_df)
    
    # Get all feature names
    engineered_features = [col for col in train_df.columns 
                          if col not in x_features + market_features + ['label', 'timestamp']]
    
    all_features = x_features + market_features + engineered_features
    print(f"Total features: {len(all_features)}")
    
    # Prepare data
    print("\nPreparing data...")
    sample_size = min(30000, int(0.5 * len(train_df)))
    train_data = train_df.iloc[-sample_size:].reset_index(drop=True)
    
    X = train_data[all_features].values
    y = train_data["label"].values
    
    # Handle NaN values
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    y = np.nan_to_num(y, nan=0.0)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Use StandardScaler for stability
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Clip scaled values for additional stability
    X_train_scaled = np.clip(X_train_scaled, -5, 5)
    X_val_scaled = np.clip(X_val_scaled, -5, 5)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    # Initialize simplified DOFEN model
    print("\nInitializing simplified DOFEN model...")
    model = SimplifiedDOFEN(
        num_features=len(all_features),
        num_trees=15,
        tree_depth=3,
        temperature=2.0,
        dropout=0.3
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Architecture: 15 trees, depth 3, with residual connections")
    
    # Train model
    print("\nTraining DOFEN model...")
    model, train_losses, val_losses = train_dofen_fixed(
        model, train_loader, val_loader, n_epochs=15, learning_rate=0.001
    )
    
    # Compute feature importance
    print("\n=== Computing DOFEN Feature Importance ===")
    
    print("\n1. Computing tree-based importance...")
    tree_importance = model.get_tree_importance()
    
    print("\n2. Computing gradient-based importance...")
    criterion = nn.MSELoss()
    gradient_importance = model.get_gradient_importance(val_loader, criterion, device)
    
    # Combine importance scores
    print("\n3. Combining importance scores...")
    
    # Normalize importances
    tree_importance = np.nan_to_num(tree_importance, nan=0.0)
    gradient_importance = np.nan_to_num(gradient_importance, nan=0.0)
    
    if tree_importance.max() > tree_importance.min():
        tree_importance = (tree_importance - tree_importance.min()) / (tree_importance.max() - tree_importance.min())
    
    if gradient_importance.max() > gradient_importance.min():
        gradient_importance = (gradient_importance - gradient_importance.min()) / (gradient_importance.max() - gradient_importance.min())
    
    # Combined importance (weighted average)
    combined_importance = 0.6 * tree_importance + 0.4 * gradient_importance
    
    importance_dict = {
        'tree_importance': tree_importance,
        'gradient_importance': gradient_importance,
        'combined': combined_importance
    }
    
    # Create visualizations
    print("\n=== Creating Visualizations ===")
    create_dofen_visualizations_fixed(importance_dict, all_features)
    
    # Create results dataframe
    results_df = pd.DataFrame({
        'feature': all_features,
        'combined_importance': combined_importance,
        'tree_importance': tree_importance,
        'gradient_importance': gradient_importance
    })
    
    # Sort by combined importance
    results_df = results_df.sort_values('combined_importance', ascending=False)
    
    # Display results
    print("\n" + "="*80)
    print("DOFEN FEATURE IMPORTANCE RESULTS")
    print("="*80)
    
    print("\nğŸ”� Top 30 Most Important Features:")
    print("-" * 60)
    for idx, row in results_df.head(30).iterrows():
        print(f"{row['feature']:30s} {row['combined_importance']:.6f}")
    
    # Category analysis
    print("\nğŸ“Š Feature Importance by Category:")
    print("-" * 60)
    
    # Market features
    market_df = results_df[results_df['feature'].isin(market_features)]
    print(f"\nMarket Features (n={len(market_df)}):")
    for idx, row in market_df.iterrows():
        print(f"  {row['feature']:25s} Combined: {row['combined_importance']:.4f}")
    
    # Top engineered features
    engineered_df = results_df[results_df['feature'].isin(engineered_features)]
    print(f"\nTop 15 Engineered Features (n={len(engineered_df)}):")
    for idx, row in engineered_df.head(15).iterrows():
        print(f"  {row['feature']:25s} Combined: {row['combined_importance']:.4f}")
    
    # Top anonymous features
    x_df = results_df[results_df['feature'].str.startswith('X')]
    print(f"\nTop 15 Anonymous Features (n={len(x_df)}):")
    for idx, row in x_df.head(15).iterrows():
        print(f"  {row['feature']:25s} Combined: {row['combined_importance']:.4f}")
    
    # Save results
    os.makedirs('dofen_outputs', exist_ok=True)
    results_df.to_csv("dofen_outputs/dofen_feature_importance_fixed.csv", index=False)
    print("\nâœ… Results saved to 'dofen_outputs/dofen_feature_importance_fixed.csv'")
    
    # Training curve plot
    if len(train_losses) > 0 and len(val_losses) > 0:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('DOFEN Training Progress', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('dofen_outputs/dofen_training_progress.png', dpi=300)
        plt.show()
    
    return results_df, importance_dict

# =========================
# Execute Analysis
# =========================

if __name__ == "__main__":
    print("Starting Fixed DOFEN Feature Importance Analysis...")
    print("="*80)
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    
    results_df, importance_dict = analyze_dofen_feature_importance_fixed()
    
    print("\nâœ… DOFEN Analysis completed successfully!")
    print("Check the 'dofen_outputs' directory for results and visualizations.")

