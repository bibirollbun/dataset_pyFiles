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


# Enhanced DCN V2 Feature Importance Analysis with Visualizations
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import itertools
import warnings
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
# DCN V2 Model Components
# =========================
class CrossLayer(nn.Module):
    """Single cross layer in DCN V2 with matrix parameterization"""
    def __init__(self, input_dim, use_bias=True):
        super().__init__()
        self.input_dim = input_dim
        
        # Weight matrix for cross operation
        self.weight = nn.Parameter(torch.empty(input_dim, input_dim))
        nn.init.xavier_normal_(self.weight)
        
        # Bias
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(input_dim))
            
    def forward(self, x0, x):
        # x0: initial input, x: current layer input
        # Cross operation: x0 * (W * x^T) + bias + x
        batch_size = x.shape[0]
        
        # Compute x^T * W
        feature_cross = torch.matmul(x, self.weight)  # (batch_size, input_dim)
        
        # Element-wise multiplication with x0
        cross_output = x0 * feature_cross
        
        if self.use_bias:
            cross_output = cross_output + self.bias
            
        # Residual connection
        output = cross_output + x
        
        return output
    
    def get_cross_weights(self):
        """Get the cross weight matrix for analysis"""
        return self.weight.detach().cpu().numpy()

class MixtureCrossLayer(nn.Module):
    """Mixture of Experts Cross Layer for efficiency"""
    def __init__(self, input_dim, num_experts=4, low_rank=32, use_bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.num_experts = num_experts
        self.low_rank = low_rank
        
        # Expert weights (low-rank factorization)
        self.U = nn.Parameter(torch.empty(input_dim, low_rank, num_experts))
        self.V = nn.Parameter(torch.empty(input_dim, low_rank, num_experts))
        self.C = nn.Parameter(torch.empty(input_dim, low_rank, num_experts))
        
        nn.init.xavier_normal_(self.U)
        nn.init.xavier_normal_(self.V)
        nn.init.xavier_normal_(self.C)
        
        # Gating network
        self.gate = nn.Sequential(
            nn.Linear(input_dim, num_experts),
            nn.Softmax(dim=1)
        )
        
        self.use_bias = use_bias
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(input_dim))
            
    def forward(self, x0, x):
        batch_size = x.shape[0]
        
        # Compute expert gates
        gates = self.gate(x)  # (batch_size, num_experts)
        
        # Compute expert outputs
        expert_outputs = []
        for i in range(self.num_experts):
            # Low-rank cross: x0 * (U * V^T * x) where U, V are expert-specific
            Ui = self.U[:, :, i]  # (input_dim, low_rank)
            Vi = self.V[:, :, i]  # (input_dim, low_rank)
            Ci = self.C[:, :, i]  # (input_dim, low_rank)
            
            # Efficient computation
            v_x = torch.matmul(x, Vi)  # (batch_size, low_rank)
            u_v_x = torch.matmul(v_x, Ui.t())  # (batch_size, input_dim)
            
            # Apply gating implicitly
            c_x = torch.matmul(x, Ci)  # (batch_size, low_rank)
            
            # Cross output
            cross_i = x0 * u_v_x * gates[:, i:i+1]
            expert_outputs.append(cross_i)
        
        # Combine expert outputs
        cross_output = sum(expert_outputs)
        
        if self.use_bias:
            cross_output = cross_output + self.bias
            
        # Residual connection
        output = cross_output + x
        
        return output
    
    def get_expert_importance(self, x):
        """Get importance of each expert"""
        gates = self.gate(x)
        return gates.mean(dim=0).detach().cpu().numpy()

class CrossNetwork(nn.Module):
    """Cross Network with multiple cross layers"""
    def __init__(self, input_dim, num_layers=3, use_mixture=True, num_experts=4):
        super().__init__()
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.use_mixture = use_mixture
        
        self.cross_layers = nn.ModuleList()
        for i in range(num_layers):
            if use_mixture:
                layer = MixtureCrossLayer(input_dim, num_experts=num_experts)
            else:
                layer = CrossLayer(input_dim)
            self.cross_layers.append(layer)
            
    def forward(self, x):
        x0 = x
        for layer in self.cross_layers:
            x = layer(x0, x)
        return x
    
    def get_cross_weights(self):
        """Get cross weights from all layers"""
        weights = []
        for i, layer in enumerate(self.cross_layers):
            if isinstance(layer, CrossLayer):
                weights.append(layer.get_cross_weights())
        return weights
    
    def get_interaction_strengths(self):
        """Compute feature interaction strengths"""
        if not self.use_mixture:
            # For matrix cross layers
            interaction_matrix = np.zeros((self.input_dim, self.input_dim))
            
            for layer in self.cross_layers:
                if isinstance(layer, CrossLayer):
                    W = layer.get_cross_weights()
                    interaction_matrix += np.abs(W)
                    
            return interaction_matrix / self.num_layers
        else:
            # For mixture layers, return None (handled differently)
            return None

class DeepNetwork(nn.Module):
    """Deep network component of DCN V2"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
            
        self.network = nn.Sequential(*layers)
        self.output_dim = prev_dim
        
    def forward(self, x):
        return self.network(x)

class DCNV2(nn.Module):
    """DCN V2: Deep & Cross Network V2"""
    def __init__(self, input_dim, num_cross_layers=3, deep_hidden_dims=[256, 128, 64],
                 use_mixture=True, num_experts=4, combination='parallel'):
        super().__init__()
        
        self.input_dim = input_dim
        self.combination = combination  # 'parallel' or 'stacked'
        
        # Cross Network
        self.cross_network = CrossNetwork(
            input_dim, 
            num_layers=num_cross_layers,
            use_mixture=use_mixture,
            num_experts=num_experts
        )
        
        # Deep Network
        self.deep_network = DeepNetwork(input_dim, deep_hidden_dims)
        
        # Final layers
        if combination == 'parallel':
            # Parallel: concat cross and deep outputs
            final_dim = input_dim + self.deep_network.output_dim
        else:
            # Stacked: cross output feeds into deep
            final_dim = self.deep_network.output_dim
            
        self.final_layer = nn.Sequential(
            nn.Linear(final_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        # Feature importance tracking
        self.register_buffer('feature_gradients', torch.zeros(input_dim))
        self.register_buffer('gradient_count', torch.tensor(0))
        
    def forward(self, x, track_gradients=False):
        if self.combination == 'parallel':
            # Parallel structure
            cross_output = self.cross_network(x)
            deep_output = self.deep_network(x)
            
            # Concatenate
            combined = torch.cat([cross_output, deep_output], dim=1)
        else:
            # Stacked structure
            cross_output = self.cross_network(x)
            deep_output = self.deep_network(cross_output)
            combined = deep_output
            
        # Final prediction
        output = self.final_layer(combined)
        
        if track_gradients and x.requires_grad:
            self.accumulate_gradients(x, output)
            
        return output
    
    def accumulate_gradients(self, x, output):
        """Accumulate gradients for feature importance"""
        grad_outputs = torch.ones_like(output)
        grads = torch.autograd.grad(outputs=output, inputs=x, 
                                   grad_outputs=grad_outputs, 
                                   retain_graph=True, create_graph=True)[0]
        
        self.feature_gradients += torch.abs(grads).mean(dim=0).detach()
        self.gradient_count += 1
    
    def get_gradient_importance(self):
        """Get accumulated gradient importance"""
        if self.gradient_count > 0:
            return (self.feature_gradients / self.gradient_count).cpu().numpy()
        else:
            return np.zeros(self.input_dim)
    
    def reset_gradient_importance(self):
        """Reset gradient accumulator"""
        self.feature_gradients.zero_()
        self.gradient_count.zero_()
    
    def compute_cross_importance(self):
        """Compute importance from cross network weights"""
        if not self.cross_network.use_mixture:
            # Get interaction matrix
            interaction_matrix = self.cross_network.get_interaction_strengths()
            
            # Feature importance = sum of interactions with other features
            cross_importance = interaction_matrix.sum(axis=0) + interaction_matrix.sum(axis=1)
            return cross_importance / 2
        else:
            # For mixture layers, use different approach
            return None
    
    def compute_feature_interactions(self, X, n_samples=3000, top_k=20):
        """Compute pairwise feature interactions through cross network"""
        self.eval()
        
        # Use subset of data
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_subset = X[indices]
        else:
            X_subset = X
            
        X_tensor = torch.tensor(X_subset, dtype=torch.float32, device=device)
        
        # Get baseline predictions
        with torch.no_grad():
            baseline_preds = self.forward(X_tensor).cpu().numpy().flatten()
        
        # Compute pairwise interactions for top features
        # First, get individual feature importance
        individual_importance = np.zeros(self.input_dim)
        
        for i in range(self.input_dim):
            X_zeroed = X_subset.copy()
            X_zeroed[:, i] = 0
            
            with torch.no_grad():
                X_zeroed_tensor = torch.tensor(X_zeroed, dtype=torch.float32, device=device)
                zeroed_preds = self.forward(X_zeroed_tensor).cpu().numpy().flatten()
                
            individual_importance[i] = np.abs(baseline_preds - zeroed_preds).mean()
        
        # Get top k features
        top_features = np.argsort(individual_importance)[-top_k:][::-1]
        
        # Compute pairwise interactions for top features
        interaction_matrix = np.zeros((top_k, top_k))
        
        for idx_i, i in enumerate(tqdm(top_features, desc="Computing interactions")):
            for idx_j, j in enumerate(top_features):
                if i >= j:
                    continue
                    
                # Zero out both features
                X_both_zeroed = X_subset.copy()
                X_both_zeroed[:, i] = 0
                X_both_zeroed[:, j] = 0
                
                with torch.no_grad():
                    X_both_tensor = torch.tensor(X_both_zeroed, dtype=torch.float32, device=device)
                    both_zeroed_preds = self.forward(X_both_tensor).cpu().numpy().flatten()
                
                # Interaction = effect of zeroing both - sum of individual effects
                interaction_effect = np.abs(baseline_preds - both_zeroed_preds).mean()
                expected_effect = individual_importance[i] + individual_importance[j]
                interaction_strength = abs(interaction_effect - expected_effect)
                
                interaction_matrix[idx_i, idx_j] = interaction_strength
                interaction_matrix[idx_j, idx_i] = interaction_strength
                
        return top_features, interaction_matrix, individual_importance
    
    def compute_polynomial_importance(self, X, max_order=3, n_samples=2000):
        """Compute importance of polynomial feature combinations"""
        self.eval()
        
        # Use subset of data
        if len(X) > n_samples:
            indices = np.random.choice(len(X), n_samples, replace=False)
            X_subset = X[indices]
        else:
            X_subset = X
            
        polynomial_importance = defaultdict(float)
        
        # Get top features first
        individual_importance = self.compute_feature_interactions(X_subset, n_samples=1000, top_k=10)[2]
        top_features = np.argsort(individual_importance)[-10:][::-1]
        
        # Test polynomial combinations up to max_order
        for order in range(1, min(max_order + 1, len(top_features) + 1)):
            for feature_combo in itertools.combinations(top_features, order):
                X_poly = X_subset.copy()
                
                # Create polynomial feature
                poly_feature = np.ones(len(X_subset))
                for feat_idx in feature_combo:
                    poly_feature *= X_subset[:, feat_idx]
                
                # Standardize
                poly_feature = (poly_feature - poly_feature.mean()) / (poly_feature.std() + 1e-8)
                
                # Test importance by correlation with predictions
                X_tensor = torch.tensor(X_subset, dtype=torch.float32, device=device)
                with torch.no_grad():
                    preds = self.forward(X_tensor).cpu().numpy().flatten()
                
                correlation = abs(pearsonr(poly_feature, preds)[0])
                polynomial_importance[feature_combo] = correlation
                
        return polynomial_importance

# =========================
# Visualization Functions
# =========================
def create_dcnv2_visualizations(importance_results, feature_names, save_prefix="dcnv2"):
    """Create comprehensive visualizations for DCN V2 feature importance"""
    
    # 1. Combined importance bar plot
    plt.figure(figsize=(12, 8))
    top_n = 30
    
    # Get combined importance
    combined_importance = importance_results['combined_importance']
    top_indices = np.argsort(combined_importance)[-top_n:][::-1]
    
    top_features = [feature_names[i] for i in top_indices]
    top_importance = combined_importance[top_indices]
    
    # Color code by feature type
    colors = ['#FF6B6B' if f.startswith('X') else '#4ECDC4' if f in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume'] else '#45B7D1' for f in top_features]
    
    bars = plt.bar(range(len(top_features)), top_importance, color=colors)
    plt.xticks(range(len(top_features)), top_features, rotation=45, ha='right')
    plt.xlabel('Features')
    plt.ylabel('Combined Importance Score')
    plt.title(f'Top {top_n} Most Important Features (DCN V2)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#FF6B6B', label='Anonymous Features (X_)'),
        Patch(facecolor='#4ECDC4', label='Market Features'),
        Patch(facecolor='#45B7D1', label='Engineered Features')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.savefig(f'{save_prefix}_top_features.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Feature interaction heatmap
    if 'interaction_matrix' in importance_results and importance_results['interaction_matrix'] is not None:
        plt.figure(figsize=(12, 10))
        
        top_features = importance_results['interaction_top_features']
        interaction_matrix = importance_results['interaction_matrix']
        
        # Create labels
        labels = [feature_names[idx] for idx in top_features]
        
        # Plot heatmap
        ax = sns.heatmap(interaction_matrix, 
                        xticklabels=labels,
                        yticklabels=labels,
                        cmap='YlOrRd',
                        cbar_kws={'label': 'Interaction Strength'},
                        square=True)
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.title('Feature Interaction Matrix (Top Features)', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_interaction_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 3. DCN V2 components visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Cross importance
    ax = axes[0, 0]
    if 'cross_importance' in importance_results and importance_results['cross_importance'] is not None:
        cross_importance = importance_results['cross_importance']
        ax.hist(cross_importance, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax.axvline(cross_importance.mean(), color='red', linestyle='--', 
                   label=f'Mean: {cross_importance.mean():.4f}')
        ax.set_xlabel('Cross Network Importance')
        ax.set_ylabel('Frequency')
        ax.set_title('Cross Network Importance Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Gradient importance
    ax = axes[0, 1]
    gradient_importance = importance_results['gradient_importance']
    ax.hist(gradient_importance, bins=50, alpha=0.7, color='lightcoral', edgecolor='black')
    ax.axvline(gradient_importance.mean(), color='darkred', linestyle='--',
               label=f'Mean: {gradient_importance.mean():.4f}')
    ax.set_xlabel('Gradient Importance')
    ax.set_ylabel('Frequency')
    ax.set_title('Gradient Importance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Individual feature effects
    ax = axes[1, 0]
    individual_effects = importance_results['individual_effects']
    ax.hist(individual_effects, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax.axvline(individual_effects.mean(), color='darkgreen', linestyle='--',
               label=f'Mean: {individual_effects.mean():.4f}')
    ax.set_xlabel('Individual Feature Effect')
    ax.set_ylabel('Frequency')
    ax.set_title('Individual Feature Effect Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Top polynomial features
    ax = axes[1, 1]
    if 'polynomial_importance' in importance_results:
        poly_importance = importance_results['polynomial_importance']
        
        # Get top polynomial features
        top_polys = sorted(poly_importance.items(), key=lambda x: x[1], reverse=True)[:10]
        
        poly_names = []
        poly_values = []
        for combo, value in top_polys:
            if len(combo) == 1:
                name = feature_names[combo[0]]
            else:
                name = ' Ã— '.join([feature_names[idx] for idx in combo])
            poly_names.append(name[:30] + '...' if len(name) > 30 else name)
            poly_values.append(value)
        
        y_pos = np.arange(len(poly_names))
        ax.barh(y_pos, poly_values, color='plum')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(poly_names)
        ax.set_xlabel('Correlation with Output')
        ax.set_title('Top Polynomial Features')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.suptitle('DCN V2 Component Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_components.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Cross layer analysis
    if not importance_results.get('use_mixture', True):
        # For matrix cross layers, show weight evolution
        plt.figure(figsize=(10, 6))
        
        # This would show how cross weights evolve through layers
        # (Implementation depends on specific cross weight structure)
        plt.text(0.5, 0.5, 'Cross Layer Weight Analysis\n(Matrix Parameterization)', 
                ha='center', va='center', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(f'{save_prefix}_cross_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # 5. Feature importance by category with interaction strength
    plt.figure(figsize=(12, 6))
    
    # Categorize features
    x_features_idx = [i for i, f in enumerate(feature_names) if f.startswith('X')]
    market_features_idx = [i for i, f in enumerate(feature_names) if f in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']]
    engineered_features_idx = [i for i in range(len(feature_names)) if i not in x_features_idx and i not in market_features_idx]
    
    categories = ['Anonymous (X_)', 'Market', 'Engineered']
    
    # Calculate average importance and interaction strength by category
    avg_importance = [
        combined_importance[x_features_idx].mean(),
        combined_importance[market_features_idx].mean(),
        combined_importance[engineered_features_idx].mean()
    ]
    
    # Create bar plot
    x = np.arange(len(categories))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, avg_importance, width, label='Average Importance', 
                    color=['#FF6B6B', '#4ECDC4', '#45B7D1'])
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom')
    
    ax.set_xlabel('Feature Category')
    ax.set_ylabel('Average Score')
    ax.set_title('Feature Importance by Category', fontsize=16, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_category_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 6. Network architecture visualization
    plt.figure(figsize=(10, 8))
    
    # Simple network architecture diagram
    ax = plt.gca()
    ax.text(0.5, 0.9, 'DCN V2 Architecture', fontsize=18, ha='center', fontweight='bold')
    
    # Input
    ax.text(0.2, 0.7, 'Input Features', fontsize=14, ha='center', 
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    
    # Cross Network
    ax.text(0.2, 0.5, 'Cross Network\n(Explicit Interactions)', fontsize=12, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
    
    # Deep Network
    ax.text(0.5, 0.5, 'Deep Network\n(Implicit Patterns)', fontsize=12, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
    
    # Output
    ax.text(0.35, 0.3, 'Combined Features', fontsize=12, ha='center')
    ax.text(0.35, 0.1, 'Output', fontsize=14, ha='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
    
    # Arrows
    ax.annotate('', xy=(0.2, 0.65), xytext=(0.2, 0.55),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(0.5, 0.65), xytext=(0.5, 0.55),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(0.3, 0.35), xytext=(0.2, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(0.4, 0.35), xytext=(0.5, 0.45),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    ax.annotate('', xy=(0.35, 0.25), xytext=(0.35, 0.15),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))
    
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_architecture.png', dpi=300, bbox_inches='tight')
    plt.show()

# =========================
# Main Feature Importance Analysis
# =========================
def analyze_dcnv2_feature_importance():
    print("=== Enhanced DCN V2 Feature Importance Analysis ===\n")
    
    # Load data
    print("Loading data...")
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    
    # Get all X features
    x_features = [f"X{i}" for i in range(1, 891)]
    market_features = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]
    
    # Load data with all features
    train_df = pd.read_parquet(train_path, columns=x_features + market_features + ["label"])
    print(f"Loaded {len(train_df)} samples")
    
    # Add engineered features
    print("\nAdding engineered features...")
    train_df = add_features(train_df)
    
    # Get all feature names
    all_features = x_features + market_features
    
    # Engineered features
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
        'directional_volume', 'signed_volume', 'volume_weighted_sell', 'buy_sell_ratio',
        'selling_pressure', 'effective_spread_proxy', 'bid_ask_imbalance', 'order_flow_imbalance',
        'liquidity_ratio'
    ]
    
    all_features = all_features + engineered_features
    print(f"Total features: {len(all_features)}")
    
    # Prepare data - Use more data for robustness
    print("\nPreparing data...")
    train_size = int(0.5 * len(train_df))  # Use 50% of data
    train_data = train_df.iloc[-train_size:].reset_index(drop=True)
    
    X = train_data[all_features].values
    y = train_data["label"].values
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale data
    scaler = StandardScaler()
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
    
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1024, shuffle=False)
    
    # Initialize DCN V2 model
    print("\nInitializing DCN V2 model...")
    use_mixture = True  # Use mixture of experts for efficiency
    
    model = DCNV2(
        input_dim=len(all_features),
        num_cross_layers=4,
        deep_hidden_dims=[512, 256, 128],
        use_mixture=use_mixture,
        num_experts=4,
        combination='parallel'
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Using {'Mixture of Experts' if use_mixture else 'Matrix'} Cross Layers")
    
    # Train model
    print("\nTraining DCN V2 model...")
    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Reset gradient importance accumulator
    model.reset_gradient_importance()
    
    for epoch in range(20):  # Train for 20 epochs
        model.train()
        train_loss = 0.0
        
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/20"):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Enable gradient computation for inputs
            inputs.requires_grad = True
            
            optimizer.zero_grad()
            outputs = model(inputs, track_gradients=True)
            loss = criterion(outputs, targets)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs, track_gradients=False)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
                
                val_preds.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(targets.cpu().numpy().flatten())
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_pearson = pearsonr(val_targets, val_preds)[0]
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}, Val Pearson = {val_pearson:.4f}")
        
        scheduler.step(avg_val_loss)
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_dcnv2_importance.pt')
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered")
                break
    
    # Load best model
    model.load_state_dict(torch.load('best_dcnv2_importance.pt'))
    
    # Compute feature importance using multiple methods
    print("\n=== Computing DCN V2 Feature Importance ===")
    
    # 1. Gradient importance (accumulated during training)
    print("\n1. Getting accumulated gradient importance...")
    gradient_importance = model.get_gradient_importance()
    
    # 2. Cross network importance (if using matrix parameterization)
    print("\n2. Computing cross network importance...")
    cross_importance = model.compute_cross_importance()
    
    # 3. Feature interactions
    print("\n3. Computing feature interactions...")
    top_features, interaction_matrix, individual_effects = model.compute_feature_interactions(
        X_val_scaled, n_samples=3000, top_k=20
    )
    
    # 4. Polynomial feature importance
    print("\n4. Computing polynomial feature importance...")
    polynomial_importance = model.compute_polynomial_importance(
        X_val_scaled, max_order=3, n_samples=2000
    )
    
    # Combine all importance scores
    importance_results = {
        'gradient_importance': gradient_importance,
        'cross_importance': cross_importance,
        'individual_effects': individual_effects,
        'interaction_top_features': top_features,
        'interaction_matrix': interaction_matrix,
        'polynomial_importance': polynomial_importance,
        'use_mixture': use_mixture
    }
    
    # Create combined score
    combined_scores = []
    
    # Add gradient importance
    grad_normalized = (gradient_importance - gradient_importance.min()) / \
                     (gradient_importance.max() - gradient_importance.min() + 1e-8)
    combined_scores.append(grad_normalized)
    
    # Add cross importance if available
    if cross_importance is not None:
        cross_normalized = (cross_importance - cross_importance.min()) / \
                          (cross_importance.max() - cross_importance.min() + 1e-8)
        combined_scores.append(cross_normalized)
    
    # Add individual effects
    effects_normalized = (individual_effects - individual_effects.min()) / \
                        (individual_effects.max() - individual_effects.min() + 1e-8)
    combined_scores.append(effects_normalized)
    
    # Average normalized scores
    combined_importance = np.mean(combined_scores, axis=0)
    importance_results['combined_importance'] = combined_importance
    
    # Create all visualizations
    print("\n=== Creating Visualizations ===")
    create_dcnv2_visualizations(importance_results, all_features)
    
    # Create detailed results dataframe
    results_df = pd.DataFrame({
        'feature': all_features,
        'combined_importance': combined_importance,
        'gradient_importance': gradient_importance,
        'individual_effect': individual_effects
    })
    
    # Add cross importance if available
    if cross_importance is not None:
        results_df['cross_importance'] = cross_importance
    
    # Sort by combined importance
    results_df = results_df.sort_values('combined_importance', ascending=False)
    
    # Display results
    print("\n" + "="*80)
    print("DCN V2 FEATURE IMPORTANCE RESULTS")
    print("="*80)
    
    print("\nğŸ”� Top 30 Most Important Features (Combined Score):")
    print("-" * 60)
    for idx, row in results_df.head(30).iterrows():
        print(f"{row['feature']:25s} {row['combined_importance']:.6f}")
    
    # Separate by category
    market_df = results_df[results_df['feature'].isin(market_features)]
    engineered_df = results_df[results_df['feature'].isin(engineered_features)]
    x_features_df = results_df[results_df['feature'].str.startswith('X')]
    
    print("\nğŸ“Š All Market Features Importance:")
    print("-" * 60)
    for idx, row in market_df.iterrows():
        print(f"{row['feature']:25s} Combined: {row['combined_importance']:.4f} | "
              f"Gradient: {row['gradient_importance']:.4f} | Effect: {row['individual_effect']:.4f}")
    
    print("\nğŸ”§ Top 20 Engineered Features:")
    print("-" * 60)
    for idx, row in engineered_df.head(20).iterrows():
        print(f"{row['feature']:25s} Combined: {row['combined_importance']:.4f} | "
              f"Gradient: {row['gradient_importance']:.4f} | Effect: {row['individual_effect']:.4f}")
    
    print("\nğŸ�¯ Top 20 Anonymous Features (X_):")
    print("-" * 60)
    for idx, row in x_features_df.head(20).iterrows():
        print(f"{row['feature']:25s} Combined: {row['combined_importance']:.4f} | "
              f"Gradient: {row['gradient_importance']:.4f} | Effect: {row['individual_effect']:.4f}")
    
    # DCN V2-specific insights
    print("\nğŸ§  DCN V2-Specific Insights:")
    print("-" * 60)
    print(f"Cross Network Type: {'Mixture of Experts' if use_mixture else 'Matrix Parameterization'}")
    print(f"Number of Cross Layers: 4")
    print(f"Deep Network Architecture: [512, 256, 128]")
    
    # Top feature interactions
    print("\nğŸ”— Top Feature Interactions:")
    interaction_pairs = []
    for i in range(len(top_features)):
        for j in range(i+1, len(top_features)):
            if interaction_matrix[i, j] > 0:
                interaction_pairs.append((
                    (top_features[i], top_features[j]),
                    interaction_matrix[i, j]
                ))
    
    interaction_pairs.sort(key=lambda x: x[1], reverse=True)
    
    for (feat1, feat2), strength in interaction_pairs[:10]:
        print(f"  {all_features[feat1]} Ã— {all_features[feat2]}: {strength:.4f}")
    
    # Top polynomial features
    print("\nğŸ“ˆ Top Polynomial Features:")
    top_polys = sorted(polynomial_importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for combo, importance in top_polys:
        if len(combo) == 1:
            poly_name = all_features[combo[0]]
        else:
            poly_name = ' Ã— '.join([all_features[idx] for idx in combo])
        print(f"  {poly_name}: {importance:.4f}")
    
    # Save results
    results_df.to_csv("dcnv2_feature_importance_comprehensive.csv", index=False)
    print("\nâœ… Comprehensive results saved to 'dcnv2_feature_importance_comprehensive.csv'")
    
    # Save interaction analysis
    interaction_df = pd.DataFrame({
        'feature1': [all_features[feat1] for (feat1, feat2), _ in interaction_pairs[:50]],
        'feature2': [all_features[feat2] for (feat1, feat2), _ in interaction_pairs[:50]],
        'interaction_strength': [strength for _, strength in interaction_pairs[:50]]
    })
    interaction_df.to_csv("dcnv2_feature_interactions.csv", index=False)
    print("âœ… Top 50 feature interactions saved to 'dcnv2_feature_interactions.csv'")
    
    # Save polynomial features
    poly_df = pd.DataFrame([
        {
            'features': ' Ã— '.join([all_features[idx] for idx in combo]),
            'order': len(combo),
            'importance': importance
        }
        for combo, importance in sorted(polynomial_importance.items(), 
                                       key=lambda x: x[1], reverse=True)[:100]
    ])
    poly_df.to_csv("dcnv2_polynomial_features.csv", index=False)
    print("âœ… Top 100 polynomial features saved to 'dcnv2_polynomial_features.csv'")
    
    # Save detailed analysis
    with open('dcnv2_feature_analysis.txt', 'w') as f:
        f.write("DCN V2 Feature Importance Analysis\n")
        f.write("="*80 + "\n\n")
        
        f.write("Model Configuration:\n")
        f.write(f"- Cross Network: {'Mixture of Experts' if use_mixture else 'Matrix'}\n")
        f.write(f"- Number of Cross Layers: 4\n")
        f.write(f"- Deep Network: [512, 256, 128]\n")
        f.write(f"- Combination: Parallel\n\n")
        
        # Top features by each method
        methods = ['gradient_importance', 'individual_effect']
        if cross_importance is not None:
            methods.append('cross_importance')
            
        for method in methods:
            f.write(f"\nTop 20 Features by {method}:\n")
            f.write("-"*60 + "\n")
            top_by_method = results_df.nlargest(20, method)[['feature', method]]
            for idx, row in top_by_method.iterrows():
                f.write(f"{row['feature']:25s} {row[method]:.6f}\n")
    
    print("âœ… Detailed analysis saved to 'dcnv2_feature_analysis.txt'")
    
    return results_df, importance_results

if __name__ == "__main__":
    results_df, importance_dict = analyze_dcnv2_feature_importance()

