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
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.feature_selection import mutual_info_regression, SelectKBest
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# =========================
# Configuration
# =========================
class Config:
    TRAIN_PATH = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    TEST_PATH = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    SUBMISSION_PATH = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    
    # Extended features for SAINT
    SAINT_FEATURES = [
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

# =========================
# SAINT Components
# =========================
class FeatureEmbedder(nn.Module):
    """Embeds features with positional encoding"""
    def __init__(self, num_features, embed_dim, dropout=0.1):
        super().__init__()
        self.num_features = num_features
        self.embed_dim = embed_dim
        
        # Feature embedding
        self.feature_embed = nn.Linear(num_features, embed_dim)
        
        # Positional encoding for features
        self.pos_encoding = nn.Parameter(torch.randn(1, num_features, embed_dim))
        nn.init.normal_(self.pos_encoding, std=0.02)
        
        # Feature-wise projections
        self.feature_projections = nn.ModuleList([
            nn.Linear(1, embed_dim) for _ in range(num_features)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # Method 1: Global embedding
        global_embed = self.feature_embed(x)
        
        # Method 2: Feature-wise embeddings
        feature_embeds = []
        for i in range(self.num_features):
            feat = x[:, i:i+1]
            embed = self.feature_projections[i](feat)
            feature_embeds.append(embed)
        
        feature_stack = torch.stack(feature_embeds, dim=1)  # [batch, num_features, embed_dim]
        
        # Add positional encoding
        feature_stack = feature_stack + self.pos_encoding
        
        # Combine global and feature-wise
        combined = global_embed.unsqueeze(1) + feature_stack
        
        # Normalize and dropout
        output = self.norm(combined)
        output = self.dropout(output)
        
        return output, global_embed

class IntersampleAttention(nn.Module):
    """Fixed attention mechanism between different samples in a batch"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Simple self-attention that processes batch dimension
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # x shape: [batch, embed_dim]
        batch_size = x.size(0)
        
        # Simple approach: treat batch as sequence for self-attention
        # Add a dummy sequence dimension
        x_seq = x.unsqueeze(1)  # [batch, 1, embed_dim]
        
        # Self-attention within the batch
        attn_out, _ = self.attention(x_seq, x_seq, x_seq, attn_mask=mask)
        
        # Remove sequence dimension
        attn_out = attn_out.squeeze(1)  # [batch, embed_dim]
        
        # Residual connection
        output = self.norm(x + self.dropout(attn_out))
        
        return output

class MixtureOfExperts(nn.Module):
    """Mixture of Experts layer"""
    def __init__(self, embed_dim, num_experts, expert_dim, dropout=0.1):
        super().__init__()
        self.num_experts = num_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, expert_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(expert_dim, embed_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(embed_dim, num_experts),
            nn.Softmax(dim=-1)
        )
        
        # Top-k selection
        self.top_k = min(2, num_experts)  # Use top-2 experts
        
    def forward(self, x):
        # Get gating weights
        gates = self.gate(x)  # [batch, num_experts]
        
        # Select top-k experts
        top_k_gates, top_k_indices = torch.topk(gates, self.top_k, dim=-1)
        
        # Renormalize top-k gates
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        
        # Apply selected experts
        output = torch.zeros_like(x)
        for i in range(self.top_k):
            for j in range(self.num_experts):
                mask = (top_k_indices[:, i] == j)
                if mask.any():
                    expert_out = self.experts[j](x[mask])
                    gate_values = top_k_gates[mask, i].unsqueeze(-1)
                    output[mask] += gate_values * expert_out
        
        return output

class HierarchicalProcessor(nn.Module):
    """Process features at multiple scales"""
    def __init__(self, embed_dim, num_levels, dropout=0.1):
        super().__init__()
        self.num_levels = num_levels
        
        # Different pooling operations for hierarchy
        self.pooling_ops = nn.ModuleList([
            nn.AdaptiveAvgPool1d(2 ** (num_levels - i - 1))
            for i in range(num_levels)
        ])
        
        # Processing at each level
        self.level_processors = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.LayerNorm(embed_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ) for _ in range(num_levels)
        ])
        
        # Fusion layer
        self.fusion = nn.Linear(embed_dim * num_levels, embed_dim)
        
    def forward(self, x):
        # x shape: [batch, num_features, embed_dim]
        batch_size, num_features, embed_dim = x.size()
        
        # Process at different scales
        level_outputs = []
        for i in range(self.num_levels):
            # Pool features
            x_pooled = self.pooling_ops[i](x.transpose(1, 2)).transpose(1, 2)
            
            # Global pooling at this level
            x_global = x_pooled.mean(dim=1)  # [batch, embed_dim]
            
            # Process
            x_processed = self.level_processors[i](x_global)
            level_outputs.append(x_processed)
        
        # Concatenate all levels
        multi_scale = torch.cat(level_outputs, dim=-1)  # [batch, embed_dim * num_levels]
        
        # Fuse
        output = self.fusion(multi_scale)
        
        return output

class SAINT(nn.Module):
    """SAINT: Self-Attention and INtersample Transformer"""
    def __init__(self, config):
        super().__init__()
        
        self.num_features = config['input_dim']
        self.embed_dim = config['embed_dim']
        self.num_layers = config['num_layers']
        self.use_intersample = config.get('use_intersample_attention', True)
        self.use_moe = config.get('use_mixture_of_experts', True)
        
        # Feature embedding
        self.feature_embedder = FeatureEmbedder(
            self.num_features,
            self.embed_dim,
            config['embed_dropout']
        )
        
        # Self-attention layers
        self.self_attention_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.embed_dim,
                nhead=config['num_heads'],
                dim_feedforward=config['ff_dim'],
                dropout=config['attention_dropout'],
                activation='gelu',
                batch_first=True
            ) for _ in range(self.num_layers)
        ])
        
        # Intersample attention (optional)
        if self.use_intersample:
            self.intersample_attention = IntersampleAttention(
                self.embed_dim,
                config['intersample_heads'],
                config['intersample_dropout']
            )
        
        # Mixture of experts (optional)
        if self.use_moe:
            self.moe = MixtureOfExperts(
                self.embed_dim,
                config['num_experts'],
                config['expert_dim'],
                config['moe_dropout']
            )
        
        # Hierarchical processor
        self.hierarchical = HierarchicalProcessor(
            self.embed_dim,
            config['num_hierarchy_levels'],
            config['hierarchy_dropout']
        )
        
        # Final prediction head
        self.prediction_head = nn.Sequential(
            nn.Linear(self.embed_dim, config['head_dim']),
            nn.LayerNorm(config['head_dim']),
            nn.GELU(),
            nn.Dropout(config['head_dropout']),
            nn.Linear(config['head_dim'], config['head_dim'] // 2),
            nn.GELU(),
            nn.Dropout(config['head_dropout']),
            nn.Linear(config['head_dim'] // 2, 1)
        )
        
    def forward(self, x, return_embeddings=False):
        batch_size = x.size(0)
        
        # Embed features
        feature_embeds, global_embed = self.feature_embedder(x)
        
        # Self-attention over features
        attended_features = feature_embeds
        
        for layer in self.self_attention_layers:
            attended_features = layer(attended_features)
        
        # Pool features
        pooled_features = attended_features.mean(dim=1)  # [batch, embed_dim]
        
        # Intersample attention
        if self.use_intersample and batch_size > 1:
            pooled_features = self.intersample_attention(pooled_features)
        
        # Mixture of experts
        if self.use_moe:
            pooled_features = pooled_features + self.moe(pooled_features)
        
        # Hierarchical processing
        hierarchical_features = self.hierarchical(attended_features)
        
        # Combine all representations
        final_features = pooled_features + hierarchical_features + global_embed
        
        # Predictions
        predictions = self.prediction_head(final_features)
        
        if return_embeddings:
            return predictions, final_features
        
        return predictions

# =========================
# Feature Engineering
# =========================
def add_features(df):
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
    
    # New microstructure features
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-10)
    df['volume_weighted_buy'] = df['buy_qty'] * df['volume']
    
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['log_depth'] = np.log1p(df['total_depth'])
    
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['flow_toxicity'] = np.abs(df['order_flow_imbalance']) * df['volume']
    df['aggressive_flow_ratio'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    df['activity_intensity'] = (df['buy_qty'] + df['sell_qty']) / (df['volume'] + 1e-10)
    df['log_buy_qty'] = np.log1p(df['buy_qty'])
    df['log_sell_qty'] = np.log1p(df['sell_qty'])
    df['log_bid_qty'] = np.log1p(df['bid_qty'])
    df['log_ask_qty'] = np.log1p(df['ask_qty'])
    
    df['realized_spread_proxy'] = 2 * np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    df['quote_volatility_proxy'] = np.abs(df['depth_imbalance'])
    
    df['flow_depth_interaction'] = df['net_order_flow'] * df['total_depth']
    df['imbalance_volume_interaction'] = df['order_flow_imbalance'] * df['volume']
    df['depth_volume_interaction'] = df['total_depth'] * df['volume']
    df['buy_sell_spread'] = np.abs(df['buy_qty'] - df['sell_qty'])
    df['bid_ask_spread'] = np.abs(df['bid_qty'] - df['ask_qty'])
    
    df['trade_informativeness'] = df['net_order_flow'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    df['execution_shortfall_proxy'] = df['buy_sell_spread'] / (df['volume'] + 1e-10)
    df['adverse_selection_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10) * df['volume']
    
    df['fill_probability'] = df['volume'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['execution_rate'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_efficiency'] = df['volume'] / (df['bid_ask_spread'] + 1e-10)
    
    df['sqrt_volume'] = np.sqrt(df['volume'])
    df['sqrt_depth'] = np.sqrt(df['total_depth'])
    df['volume_squared'] = df['volume'] ** 2
    df['imbalance_squared'] = df['order_flow_imbalance'] ** 2
    
    df['bid_ratio'] = df['bid_qty'] / (df['total_depth'] + 1e-10)
    df['ask_ratio'] = df['ask_qty'] / (df['total_depth'] + 1e-10)
    df['buy_ratio'] = df['buy_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['sell_ratio'] = df['sell_qty'] / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    
    df['liquidity_consumption'] = (df['buy_qty'] + df['sell_qty']) / (df['total_depth'] + 1e-10)
    df['market_stress'] = df['volume'] / (df['total_depth'] + 1e-10) * np.abs(df['order_flow_imbalance'])
    df['depth_depletion'] = df['volume'] / (df['bid_qty'] + df['ask_qty'] + 1e-10)
    
    df['net_buying_ratio'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    df['directional_volume'] = df['net_order_flow'] * np.log1p(df['volume'])
    df['signed_volume'] = np.sign(df['net_order_flow']) * df['volume']
    
    # Replace infinities and NaNs
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

def load_data():
    # Load data with all features available
    all_features = Config.SAINT_FEATURES
    train_df = pd.read_parquet(Config.TRAIN_PATH, columns=all_features + [Config.LABEL_COLUMN])
    test_df = pd.read_parquet(Config.TEST_PATH, columns=all_features)
    submission_df = pd.read_csv(Config.SUBMISSION_PATH)
    print(f"Loaded data - Train: {train_df.shape}, Test: {test_df.shape}, Submission: {submission_df.shape}")

    # Add features
    train_df = add_features(train_df)
    test_df = add_features(test_df)

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True), submission_df

# =========================
# SAINT Training
# =========================
def train_saint_model(model, train_loader, val_loader, config, device):
    """Train SAINT model"""
    
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
            torch.save(model.state_dict(), "best_saint.pt")
            print(f"✅ New best model saved! Pearson: {best_val_pearson:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(torch.load("best_saint.pt"))
    
    return model, best_val_pearson

def train_saint(train_df, test_df):
    print("\n=== Training SAINT Model ===")
    
    # Set seed
    set_seed(42)
    
    # Get SAINT features
    saint_features = Config.SAINT_FEATURES.copy()
    
    # Add engineered features to SAINT features
    engineered_features = [
        "log_volume", 'bid_ask_interaction', 'net_order_flow', 'normalized_net_flow',
        'buying_pressure', 'total_depth', 'depth_imbalance', 'kyle_lambda', 
        'aggressive_flow_ratio', 'volume_depth_ratio', 'log_buy_qty', 'log_sell_qty',
        'price_impact_proxy', 'market_stress', 'liquidity_consumption'
    ]
    
    all_saint_features = saint_features + engineered_features
    all_saint_features = list(set(all_saint_features))  # Remove duplicates
    
    # Ensure all features exist
    all_saint_features = [f for f in all_saint_features if f in train_df.columns]
    
    print(f"Using {len(all_saint_features)} features for SAINT")
    
    # Use recent data (last 85%)
    train_size = int(0.85 * len(train_df))
    train_data = train_df.iloc[-train_size:].reset_index(drop=True)
    
    # Split for validation
    split_idx = int(0.8 * len(train_data))
    train_split = train_data[:split_idx].copy()
    val_split = train_data[split_idx:].copy()
    
    y_train = train_split[Config.LABEL_COLUMN].values
    y_val = val_split[Config.LABEL_COLUMN].values
    
    X_train = train_split[all_saint_features].values
    X_val = val_split[all_saint_features].values
    
    # Feature selection
    print("\nSelecting features...")
    selector = SelectKBest(score_func=mutual_info_regression, k=min(105, len(all_saint_features)))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_val_selected = selector.transform(X_val)
    
    selected_features = [all_saint_features[i] for i in selector.get_support(indices=True)]
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
    
    # Try different hyperparameter configurations
    configs = [
        # Config 1: Basic SAINT
        {
            'config_name': 'basic_saint',
            'embed_dim': 128,
            'num_layers': 3,
            'num_heads': 4,
            'ff_dim': 512,
            'embed_dropout': 0.2,
            'attention_dropout': 0.2,
            'use_intersample_attention': True,
            'intersample_heads': 4,
            'intersample_dropout': 0.2,
            'use_mixture_of_experts': False,
            'num_experts': 4,
            'expert_dim': 128,
            'moe_dropout': 0.3,
            'num_hierarchy_levels': 3,
            'hierarchy_dropout': 0.2,
            'head_dim': 256,
            'head_dropout': 0.3,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'batch_size': 512,
            'huber_delta': 1.0,
            'noise_factor': 0.01,
            'grad_clip': 1.0,
            'num_epochs': 30,
            'patience': 10,
            'input_dim': X_train_transformed.shape[1]
        },
        # Config 2: Medium SAINT
        {
            'config_name': 'medium_saint',
            'embed_dim': 192,
            'num_layers': 4,
            'num_heads': 6,
            'ff_dim': 768,
            'embed_dropout': 0.25,
            'attention_dropout': 0.25,
            'use_intersample_attention': True,
            'intersample_heads': 6,
            'intersample_dropout': 0.25,
            'use_mixture_of_experts': True,
            'num_experts': 4,
            'expert_dim': 192,
            'moe_dropout': 0.3,
            'num_hierarchy_levels': 3,
            'hierarchy_dropout': 0.25,
            'head_dim': 384,
            'head_dropout': 0.35,
            'learning_rate': 0.0008,
            'weight_decay': 0.008,
            'batch_size': 256,
            'huber_delta': 0.8,
            'noise_factor': 0.015,
            'grad_clip': 1.5,
            'num_epochs': 25,
            'patience': 8,
            'input_dim': X_train_transformed.shape[1]
        },
        # Config 3: Light SAINT
        {
            'config_name': 'light_saint',
            'embed_dim': 96,
            'num_layers': 2,
            'num_heads': 3,
            'ff_dim': 384,
            'embed_dropout': 0.15,
            'attention_dropout': 0.15,
            'use_intersample_attention': False,
            'intersample_heads': 3,
            'intersample_dropout': 0.15,
            'use_mixture_of_experts': False,
            'num_experts': 3,
            'expert_dim': 96,
            'moe_dropout': 0.2,
            'num_hierarchy_levels': 2,
            'hierarchy_dropout': 0.15,
            'head_dim': 192,
            'head_dropout': 0.25,
            'learning_rate': 0.002,
            'weight_decay': 0.005,
            'batch_size': 1024,
            'huber_delta': 1.0,
            'noise_factor': 0.008,
            'grad_clip': 1.0,
            'num_epochs': 35,
            'patience': 12,
            'input_dim': X_train_transformed.shape[1]
        }
    ]
    
    # Train ensemble of models
    ensemble_models = []
    ensemble_scores = []
    
    for config in configs:
        print(f"\n=== Training SAINT {config['config_name']} ===")
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
        
        # Create model
        model = SAINT(config).to(device)
        print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Train
        model, best_score = train_saint_model(model, train_loader, val_loader, config, device)
        
        ensemble_models.append(model)
        ensemble_scores.append(best_score)
        
        print(f"Model {config['config_name']} best validation Pearson: {best_score:.4f}")
    
    # Make test predictions
    print("\n=== Making SAINT Test Predictions ===")
    
    # Transform test data
    X_test = test_df[all_saint_features].values
    X_test_selected = selector.transform(X_test)
    X_test_transformed = transformer.transform(X_test_selected)
    
    # Make predictions with each model
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
    weights = np.array(ensemble_scores)
    weights = weights / weights.sum()
    
    final_predictions = np.zeros_like(all_predictions[0])
    for pred, weight in zip(all_predictions, weights):
        final_predictions += weight * pred
    
    # Post-processing
    pred_mean = train_df[Config.LABEL_COLUMN].mean()
    pred_std = train_df[Config.LABEL_COLUMN].std()
    final_predictions = np.clip(
        final_predictions,
        pred_mean - 4 * pred_std,
        pred_mean + 4 * pred_std
    )
    
    print(f"\nSAINT ensemble weights: {weights}")
    print(f"SAINT prediction stats - Mean: {final_predictions.mean():.6f}, Std: {final_predictions.std():.6f}")
    
    return final_predictions

# =========================
# Main Execution
# =========================
if __name__ == "__main__":
    # Load data
    train_df, test_df, submission_df = load_data()
    
    # Train SAINT model
    saint_predictions = train_saint(train_df, test_df)
    
    # Save SAINT submission
    saint_submission = submission_df.copy()
    saint_submission["prediction"] = saint_predictions
    saint_submission.to_csv("submission_saint_fixed.csv", index=False)
    print(f"\nSaved: submission_saint_fixed.csv")
    
    # Show sample predictions
    print("\nSample predictions (first 10 rows):")
    print(saint_submission[['ID', 'prediction']].head(10))

