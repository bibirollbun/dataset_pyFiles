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
import os
from pathlib import Path
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from sklearn.impute import SimpleImputer
from scipy.stats import pearsonr, spearmanr, skew, kurtosis, entropy
from scipy.special import expit
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import gc

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Features for GANDALF
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
    RANDOM_STATE = 42

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================
# Advanced Feature Engineering
# =========================
def add_advanced_features(df):
    """Add comprehensive feature engineering with interactions and derived variables"""
    eps = 1e-10
    
    # Store original columns to avoid fragmentation warning
    new_features = {}
    
    # 1. Original features with safety
    new_features['bid_ask_interaction'] = df['bid_qty'] * df['ask_qty']
    new_features['bid_buy_interaction'] = df['bid_qty'] * df['buy_qty']
    new_features['bid_sell_interaction'] = df['bid_qty'] * df['sell_qty']
    new_features['ask_buy_interaction'] = df['ask_qty'] * df['buy_qty']
    new_features['ask_sell_interaction'] = df['ask_qty'] * df['sell_qty']
    new_features['buy_sell_interaction'] = df['buy_qty'] * df['sell_qty']

    # 2. Volume-based features
    new_features['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    new_features['volume_weighted_sell'] = df['sell_qty'] * df['volume']
    new_features['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + eps)
    new_features['selling_pressure'] = df['sell_qty'] / (df['volume'] + eps)
    new_features['buying_pressure'] = df['buy_qty'] / (df['volume'] + eps)
    new_features['log_volume'] = np.log1p(df['volume'])
    new_features['sqrt_volume'] = np.sqrt(df['volume'])
    new_features['volume_squared'] = df['volume'] ** 2

    # 3. Order book features
    new_features['effective_spread_proxy'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['volume'] + eps)
    new_features['bid_ask_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + eps)
    new_features['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + eps)
    new_features['liquidity_ratio'] = (df['bid_qty'] + df['ask_qty']) / (df['volume'] + eps)
    new_features['total_depth'] = df['bid_qty'] + df['ask_qty']
    new_features['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (new_features['total_depth'] + eps)
    new_features['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (new_features['total_depth'] + eps)
    new_features['log_depth'] = np.log1p(new_features['total_depth'])
    
    # 4. Microstructure features
    new_features['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    new_features['normalized_net_flow'] = new_features['net_order_flow'] / (df['volume'] + eps)
    new_features['kyle_lambda'] = np.abs(new_features['net_order_flow']) / (df['volume'] + eps)
    new_features['signed_kyle_lambda'] = new_features['net_order_flow'] / (df['volume'] + eps)
    new_features['flow_toxicity'] = np.abs(new_features['order_flow_imbalance']) * df['volume']
    new_features['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (new_features['total_depth'] + eps)
    
    # 5. Ratios and relationships
    new_features['volume_depth_ratio'] = df['volume'] / (new_features['total_depth'] + eps)
    new_features['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + eps)
    new_features['log_buy_qty'] = np.log1p(df['buy_qty'])
    new_features['log_sell_qty'] = np.log1p(df['sell_qty'])
    new_features['log_bid_qty'] = np.log1p(df['bid_qty'])
    new_features['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    # 6. Price impact proxies
    new_features['price_impact_proxy'] = new_features['net_order_flow'] / (new_features['total_depth'] + eps)
    new_features['market_stress'] = df['volume'] / (new_features['total_depth'] + eps) * np.abs(new_features['order_flow_imbalance'])
    new_features['illiquidity_measure'] = np.abs(new_features['net_order_flow']) / (df['volume'] * new_features['total_depth'] + eps)
    
    # 7. Advanced market microstructure features
    new_features['amihud_illiquidity'] = np.abs(new_features['net_order_flow']) / (df['volume'] ** 1.5 + eps)
    new_features['hasbrouck_lambda'] = new_features['net_order_flow'] / np.sqrt(df['volume'] + eps)
    new_features['execution_cost_proxy'] = np.abs(new_features['order_flow_imbalance']) * np.sqrt(df['volume'])
    
    # 8. Liquidity consumption features
    new_features['bid_consumption_rate'] = df['buy_qty'] / (df['bid_qty'] + eps)
    new_features['ask_consumption_rate'] = df['sell_qty'] / (df['ask_qty'] + eps)
    new_features['total_consumption_rate'] = (df['buy_qty'] + df['sell_qty']) / (new_features['total_depth'] + eps)
    
    # 9. Non-linear transformations
    new_features['bid_ask_imbalance_squared'] = new_features['bid_ask_imbalance'] ** 2
    new_features['order_flow_imbalance_squared'] = new_features['order_flow_imbalance'] ** 2
    new_features['bid_ask_imbalance_cubed'] = new_features['bid_ask_imbalance'] ** 3
    new_features['order_flow_imbalance_cubed'] = new_features['order_flow_imbalance'] ** 3
    
    # 10. Interaction features between X features
    important_x_features = ['X863', 'X856', 'X598', 'X862', 'X385', 'X852', 'X603', 'X860', 
                           'X674', 'X415', 'X345', 'X855', 'X174', 'X302']
    
    for i, feat1 in enumerate(important_x_features[:7]):
        if feat1 in df.columns:
            new_features[f'{feat1}_squared'] = df[feat1] ** 2
            new_features[f'{feat1}_x_volume'] = df[feat1] * df['volume']
            new_features[f'{feat1}_x_bid_ask_imb'] = df[feat1] * new_features['bid_ask_imbalance']
            new_features[f'{feat1}_x_order_flow_imb'] = df[feat1] * new_features['order_flow_imbalance']
            
            for feat2 in important_x_features[i+1:i+3]:
                if feat2 in df.columns:
                    new_features[f'{feat1}_x_{feat2}'] = df[feat1] * df[feat2]
    
    # 11. Market regime indicators
    new_features['high_volume_regime'] = (df['volume'] > df['volume'].quantile(0.75)).astype(float)
    new_features['imbalanced_market'] = (np.abs(new_features['bid_ask_imbalance']) > 0.5).astype(float)
    new_features['toxic_flow'] = (np.abs(new_features['order_flow_imbalance']) > 0.5).astype(float)
    
    # 12. Composite scores
    new_features['liquidity_score'] = (
        new_features['total_depth'] / (new_features['total_depth'].mean() + eps) * 0.4 +
        df['volume'] / (df['volume'].mean() + eps) * 0.3 +
        (1 - np.abs(new_features['bid_ask_imbalance'])) * 0.3
    )
    
    new_features['market_quality_score'] = (
        new_features['liquidity_score'] * 0.5 +
        (1 - np.abs(new_features['order_flow_imbalance'])) * 0.3 +
        new_features['total_depth'] / (df['volume'] + eps) * 0.2
    )
    
    # 13. Advanced transformations
    new_features['tanh_bid_ask_imb'] = np.tanh(new_features['bid_ask_imbalance'])
    new_features['tanh_order_flow_imb'] = np.tanh(new_features['order_flow_imbalance'])
    new_features['sigmoid_volume_ratio'] = expit(df['volume'] / (df['volume'].mean() + eps))
    
    # 14. Volatility proxies
    new_features['flow_volatility'] = np.abs(new_features['net_order_flow']) * np.sqrt(df['volume'])
    new_features['depth_volatility'] = np.abs(new_features['depth_imbalance']) * new_features['total_depth']
    
    # 15. Information asymmetry proxies
    new_features['pin_proxy'] = np.abs(new_features['order_flow_imbalance']) / (new_features['activity_intensity'] + eps)
    new_features['information_share'] = new_features['net_order_flow'] ** 2 / (df['volume'] * new_features['total_depth'] + eps)
    
    # Add all new features to dataframe at once
    new_features_df = pd.DataFrame(new_features, index=df.index)
    df = pd.concat([df, new_features_df], axis=1)
    
    # Handle infinities and NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != Config.LABEL_COLUMN:
            lower = df[col].quantile(0.001)
            upper = df[col].quantile(0.999)
            df[col] = df[col].clip(lower, upper)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df

# =========================
# Enhanced GANDALF Components
# =========================
class AttentionGate(nn.Module):
    """Attention mechanism for feature importance"""
    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        weights = self.attention(x)
        return x * weights

class EnhancedDifferentiableDecisionTree(nn.Module):
    """Enhanced soft decision tree with attention"""
    def __init__(self, input_dim, depth, temperature=1.0, use_attention=True):
        super().__init__()
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.use_attention = use_attention
        
        if use_attention:
            self.attention_gate = AttentionGate(input_dim)
        
        self.internal_nodes = nn.ModuleList()
        for i in range(2 ** depth - 1):
            layer = nn.Linear(input_dim, 1)
            nn.init.xavier_uniform_(layer.weight, gain=0.5)
            nn.init.zeros_(layer.bias)
            self.internal_nodes.append(layer)
        
        self.leaf_values = nn.Parameter(torch.zeros(self.n_leaves))
        nn.init.uniform_(self.leaf_values, -0.01, 0.01)
        
        self.log_temperature = nn.Parameter(torch.log(torch.tensor(temperature)))
        
    def forward(self, x):
        if self.use_attention:
            x = self.attention_gate(x)
        
        batch_size = x.size(0)
        device = x.device
        
        temp = torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)
        
        path_probs = torch.ones(batch_size, 1, device=device)
        
        for level in range(self.depth):
            n_nodes = 2 ** level
            next_path_probs = []
            
            for node in range(n_nodes):
                node_idx = 2 ** level - 1 + node
                
                if node_idx < len(self.internal_nodes):
                    logit = self.internal_nodes[node_idx](x).squeeze(-1)
                    logit = torch.clamp(logit, -10, 10)
                    
                    split_prob = torch.sigmoid(logit / temp)
                    split_prob = torch.clamp(split_prob, 1e-7, 1 - 1e-7)
                    
                    if node < path_probs.size(1):
                        current_prob = path_probs[:, node]
                        left_prob = current_prob * (1 - split_prob)
                        right_prob = current_prob * split_prob
                        
                        next_path_probs.append(left_prob.unsqueeze(1))
                        next_path_probs.append(right_prob.unsqueeze(1))
            
            if next_path_probs:
                path_probs = torch.cat(next_path_probs, dim=1)
        
        if path_probs.size(1) != self.n_leaves:
            if path_probs.size(1) < self.n_leaves:
                padding = torch.zeros(batch_size, self.n_leaves - path_probs.size(1), device=device)
                path_probs = torch.cat([path_probs, padding], dim=1)
            else:
                path_probs = path_probs[:, :self.n_leaves]
        
        path_probs = F.normalize(path_probs, p=1, dim=1)
        output = torch.sum(path_probs * self.leaf_values.unsqueeze(0), dim=1)
        
        return output

class EnhancedGatingNetwork(nn.Module):
    """Enhanced gating with fixed attention dimensions"""
    def __init__(self, input_dim, n_trees, hidden_dim=128, n_heads=4, dropout=0.3):
        super().__init__()
        
        # Ensure embed_dim is divisible by num_heads
        self.attention_dim = ((input_dim + n_heads - 1) // n_heads) * n_heads
        
        # Project to attention dimension if needed
        if input_dim != self.attention_dim:
            self.input_projection = nn.Linear(input_dim, self.attention_dim)
        else:
            self.input_projection = None
        
        # Self-attention layer
        self.self_attention = nn.MultiheadAttention(
            embed_dim=self.attention_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Gating network
        self.network = nn.Sequential(
            nn.Linear(self.attention_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, n_trees)
        )
        
        for m in self.network.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # Project to attention dimension if needed
        if self.input_projection is not None:
            x_att = self.input_projection(x)
        else:
            x_att = x
        
        # Self-attention
        x_att = x_att.unsqueeze(1)
        x_att, _ = self.self_attention(x_att, x_att, x_att)
        x_att = x_att.squeeze(1)
        
        # Gating
        gates = self.network(x_att)
        gates = gates / 2.0
        return F.softmax(gates, dim=-1)

class EnhancedGANDALF(nn.Module):
    """Enhanced GANDALF with fixed architecture"""
    def __init__(self, config):
        super().__init__()
        
        self.n_trees = config['n_trees']
        self.tree_depth = config['tree_depth']
        self.input_dim = config['input_dim']
        self.use_attention = config.get('use_attention', True)
        
        # Feature embedder
        self.feature_embedder = self._build_embedder(config)
        self.embed_dim = config['embed_dims'][-1]
        
        # Skip connection
        if self.embed_dim != self.input_dim:
            self.skip_projection = nn.Linear(self.input_dim, self.embed_dim)
        else:
            self.skip_projection = None
        
        # Decision trees ensemble
        self.trees = nn.ModuleList()
        for i in range(self.n_trees):
            tree_depth = self.tree_depth + (i % 3 - 1)
            tree_depth = max(2, min(tree_depth, 7))
            use_attention = self.use_attention and (i % 2 == 0)
            
            self.trees.append(
                EnhancedDifferentiableDecisionTree(
                    self.embed_dim,
                    tree_depth,
                    temperature=config['tree_temperature'],
                    use_attention=use_attention
                )
            )
        
        # Enhanced gating network
        self.gating_network = EnhancedGatingNetwork(
            self.input_dim,
            self.n_trees,
            config['gate_hidden_dim'],
            config.get('n_heads', 4),
            config['gate_dropout']
        )
        
        # Neural network head
        if config.get('use_nn_head', True):
            self.nn_head = self._build_nn_head(config)
            self.combination_weight = nn.Parameter(torch.tensor(0.5))
        else:
            self.nn_head = None
        
        # Final normalization
        self.final_norm = nn.LayerNorm(1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _build_embedder(self, config):
        """Build feature embedder"""
        layers = []
        prev_dim = self.input_dim
        
        for i, dim in enumerate(config['embed_dims']):
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.GELU())
            
            if i < len(config['embed_dims']) - 1:
                layers.append(nn.Dropout(config['feature_dropout']))
            
            prev_dim = dim
        
        return nn.Sequential(*layers)
    
    def _build_nn_head(self, config):
        """Build neural network head"""
        layers = []
        prev_dim = self.input_dim
        
        for dim in config['head_dims']:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LayerNorm(dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(config['head_dropout']))
            prev_dim = dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x):
        x = torch.clamp(x, -10, 10)
        
        # Feature embedding
        embedded = self.feature_embedder(x)
        
        # Skip connection
        if self.skip_projection is not None:
            embedded = embedded + self.skip_projection(x)
        elif self.embed_dim == self.input_dim:
            embedded = embedded + x
        
        embedded = F.layer_norm(embedded, embedded.shape[1:])
        
        # Get tree outputs
        tree_outputs = []
        for tree in self.trees:
            output = tree(embedded)
            output = torch.clamp(output, -10, 10)
            tree_outputs.append(output)
        
        tree_outputs = torch.stack(tree_outputs, dim=1)
        
        # Gating
        gates = self.gating_network(x)
        
        # Weighted combination
        forest_output = torch.sum(gates * tree_outputs, dim=1, keepdim=True)
        
        # Combine with neural head
        if self.nn_head is not None:
            nn_output = self.nn_head(x)
            weight = torch.sigmoid(self.combination_weight)
            final_output = weight * forest_output + (1 - weight) * nn_output
        else:
            final_output = forest_output
        
        # Final normalization
        final_output = self.final_norm(final_output)
        final_output = torch.clamp(final_output, -10, 10)
        
        return final_output

# =========================
# Training Function
# =========================
def train_enhanced_gandalf_model(model, train_loader, val_loader, config, device):
    """Train enhanced GANDALF"""
    
    criterion = nn.SmoothL1Loss()
    
    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        eps=1e-8
    )
    
    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6)
    
    # Training variables
    best_val_pearson = -np.inf
    best_model_state = None
    patience_counter = 0
    patience = config.get('patience', 15)
    num_epochs = config.get('num_epochs', 50)
    
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
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
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
        
        # Calculate metrics
        avg_train_loss = train_loss / train_batches
        avg_val_loss = val_loss / len(val_loader)
        
        val_preds = np.array(val_preds)
        val_targets = np.array(val_targets)
        
        # Remove NaN values
        valid_mask = ~(np.isnan(val_preds) | np.isnan(val_targets))
        if valid_mask.sum() > 10:
            val_pearson = pearsonr(val_targets[valid_mask], val_preds[valid_mask])[0]
            val_spearman = spearmanr(val_targets[valid_mask], val_preds[valid_mask])[0]
        else:
            val_pearson = -1.0
            val_spearman = -1.0
        
        print(f"\nEpoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        print(f"Val Pearson: {val_pearson:.4f}, Val Spearman: {val_spearman:.4f}")
        
        # Update scheduler
        if not np.isnan(val_pearson):
            scheduler.step(val_pearson)
        
        # Save best model
        if not np.isnan(val_pearson) and val_pearson > best_val_pearson:
            best_val_pearson = val_pearson
            best_model_state = model.state_dict().copy()
            patience_counter = 0
            print(f"✅ New best model! Pearson: {best_val_pearson:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, best_val_pearson

def train_enhanced_gandalf(train_df, test_df):
    """Main enhanced GANDALF training function"""
    print("\n=== Training Enhanced GANDALF Model ===")
    
    set_seed(42)
    
    # Get all features
    gandalf_features = Config.GANDALF_FEATURES.copy()
    
    # Get all available engineered features
    all_features = [col for col in train_df.columns if col not in [Config.LABEL_COLUMN, 'timestamp']]
    all_gandalf_features = list(set(gandalf_features + all_features))
    all_gandalf_features = [f for f in all_gandalf_features if f in train_df.columns]
    
    print(f"Using {len(all_gandalf_features)} features for Enhanced GANDALF")
    
    # Use recent data
    train_size = int(0.9 * len(train_df))
    train_data = train_df.iloc[-train_size:].reset_index(drop=True)
    
    # Split for validation
    split_idx = int(0.85 * len(train_data))
    train_split = train_data[:split_idx].copy()
    val_split = train_data[split_idx:].copy()
    
    y_train = train_split[Config.LABEL_COLUMN].values
    y_val = val_split[Config.LABEL_COLUMN].values
    
    X_train = train_split[all_gandalf_features].values
    X_val = val_split[all_gandalf_features].values
    
    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    X_train = imputer.fit_transform(X_train)
    X_val = imputer.transform(X_val)
    
    # Feature selection
    print("\nSelecting features...")
    n_features = min(150, len(all_gandalf_features))
    selector = SelectKBest(score_func=mutual_info_regression, k=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    
    selected_features = [all_gandalf_features[i] for i in selector.get_support(indices=True)]
    print(f"Selected {len(selected_features)} features")
    
    # Scaling
    print("\nScaling data...")
    scaler = RobustScaler(quantile_range=(5, 95))
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_val_scaled = scaler.transform(X_val_selected)
    
    X_train_scaled = np.clip(X_train_scaled, -5, 5)
    X_val_scaled = np.clip(X_val_scaled, -5, 5)
    
    # Create data loaders
    train_dataset = TensorDataset(
        torch.tensor(X_train_scaled, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val_scaled, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    )
    
    # Fixed configurations with proper dimensions
    configs = [
        # Config 1: Simple model
        {
            'config_name': 'simple_stable',
            'n_trees': 12,
            'tree_depth': 4,
            'tree_temperature': 1.0,
            'embed_dims': [128, 96, 64],
            'feature_dropout': 0.1,
            'gate_hidden_dim': 64,
            'gate_dropout': 0.1,
            'n_heads': 2,  # Reduced heads
            'use_attention': True,
            'use_nn_head': True,
            'head_dims': [128, 64],
            'head_dropout': 0.2,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'batch_size': 512,
            'noise_factor': 0.005,
            'grad_clip': 1.0,
            'num_epochs': 30,
            'patience': 10,
            'input_dim': X_train_scaled.shape[1]
        },
        # Config 2: Medium model
        {
            'config_name': 'medium_ensemble',
            'n_trees': 16,
            'tree_depth': 5,
            'tree_temperature': 1.2,
            'embed_dims': [192, 128, 96],
            'feature_dropout': 0.15,
            'gate_hidden_dim': 96,
            'gate_dropout': 0.15,
            'n_heads': 3,  # Adjusted for divisibility
            'use_attention': False,  # Disable attention for stability
            'use_nn_head': True,
            'head_dims': [192, 96],
            'head_dropout': 0.2,
            'learning_rate': 0.0008,
            'weight_decay': 0.02,
            'batch_size': 384,
            'noise_factor': 0.005,
            'grad_clip': 1.0,
            'num_epochs': 25,
            'patience': 8,
            'input_dim': X_train_scaled.shape[1]
        }
    ]
    
    # Train ensemble
    ensemble_models = []
    ensemble_scores = []
    
    for config in configs:
        print(f"\n=== Training Enhanced GANDALF {config['config_name']} ===")
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        
        try:
            model = EnhancedGANDALF(config).to(device)
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Train
            model, best_score = train_enhanced_gandalf_model(
                model, train_loader, val_loader, config, device
            )
            
            if best_score > -0.5:  # Accept models with reasonable scores
                ensemble_models.append(model)
                ensemble_scores.append(max(0.01, best_score))  # Ensure positive weight
                print(f"Model {config['config_name']} best validation Pearson: {best_score:.4f}")
            else:
                print(f"Model {config['config_name']} score too low: {best_score:.4f}")
                
        except Exception as e:
            print(f"Error training {config['config_name']}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
        
        # Clean up memory
        free_memory()
    
    # Check if we have any models
    if len(ensemble_models) == 0:
        print("No models were successfully trained! Creating a simple baseline model.")
        # Create a simple baseline model
        simple_config = configs[0].copy()
        simple_config['n_trees'] = 5
        simple_config['tree_depth'] = 3
        simple_config['use_attention'] = False
        simple_config['num_epochs'] = 10
        
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        
        model = EnhancedGANDALF(simple_config).to(device)
        model, best_score = train_enhanced_gandalf_model(
            model, train_loader, val_loader, simple_config, device
        )
        ensemble_models.append(model)
        ensemble_scores.append(1.0)
    
    # Make test predictions
    print("\n=== Making Enhanced GANDALF Test Predictions ===")
    
    # Prepare test data
    X_test = test_df[all_gandalf_features].values
    X_test = imputer.transform(X_test)
    X_test_selected = selector.transform(X_test)
    X_test_scaled = scaler.transform(X_test_selected)
    X_test_scaled = np.clip(X_test_scaled, -5, 5)
    
    # Make predictions with each model
    all_predictions = []
    
    for model in ensemble_models:
        model.eval()
        test_dataset = TensorDataset(torch.tensor(X_test_scaled, dtype=torch.float32))
        test_loader = DataLoader(test_dataset, batch_size=2048, shuffle=False)
        
        predictions = []
        with torch.no_grad():
            for (inputs,) in test_loader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                predictions.extend(outputs.cpu().numpy().flatten())
        
        all_predictions.append(np.array(predictions))
    
    # Weighted ensemble
    weights = np.array(ensemble_scores)
    weights = weights / weights.sum()
    
    final_predictions = np.zeros_like(all_predictions[0])
    for pred, weight in zip(all_predictions, weights):
        final_predictions += weight * pred
    
    # Post-processing
    pred_mean = train_df[Config.LABEL_COLUMN].mean()
    pred_std = train_df[Config.LABEL_COLUMN].std()
    
    # Clip predictions
    final_predictions = np.clip(
        final_predictions,
        pred_mean - 3 * pred_std,
        pred_mean + 3 * pred_std
    )
    
    # Final NaN check
    final_predictions = np.nan_to_num(final_predictions, nan=pred_mean)
    
    print(f"\nEnhanced GANDALF ensemble weights: {weights}")
    print(f"Prediction stats - Mean: {final_predictions.mean():.6f}, Std: {final_predictions.std():.6f}")
    
    return final_predictions

def load_data():
    """Load data with enhanced features"""
    all_features = Config.GANDALF_FEATURES
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=all_features + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=all_features)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")

    # Add enhanced features
    train_df = add_advanced_features(train_df)
    test_df = add_advanced_features(test_df)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    # Set memory management
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
    
    # Load data with enhanced features
    train_df, test_df, submission_df = load_data()
    
    # Train enhanced GANDALF
    gandalf_predictions = train_enhanced_gandalf(train_df, test_df)
    
    # Save enhanced submission
    enhanced_submission = submission_df.copy()
    enhanced_submission["prediction"] = gandalf_predictions
    enhanced_submission.to_csv("submission_enhanced_gandalf_fixed.csv", index=False)
    print(f"\nSaved: submission_enhanced_gandalf_fixed.csv")
    
    # Show sample predictions
    print("\nSample predictions (first 10 rows):")
    print(enhanced_submission[['ID', 'prediction']].head(10))
    
    # Show final statistics
    print("\n✅ Enhanced GANDALF training completed successfully!")
    print(f"Total features used: {len(train_df.columns) - 1}")
    print(f"Prediction range: [{gandalf_predictions.min():.4f}, {gandalf_predictions.max():.4f}]")
    print(f"Prediction mean: {gandalf_predictions.mean():.6f}")
    print(f"Prediction std: {gandalf_predictions.std():.6f}")

