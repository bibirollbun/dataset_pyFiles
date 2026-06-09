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
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBRegressor
import xgboost as xgb
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from scipy.stats import pearsonr, rankdata
from scipy.optimize import minimize, differential_evolution
import gc

# Deep learning imports for GANDALF
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

print("Starting XGBoost + GANDALF Pipeline - Advanced Ensemble Approach...")
print("=" * 80)

# Configuration
class CFG:
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    n_folds = 5
    random_state = 42
    use_gpu = torch.cuda.is_available()
    device = torch.device('cuda' if use_gpu else 'cpu')
    
    # Feature settings
    max_x_features = 80
    n_interaction_features = 50
    n_proprietary_features = 30
    
    # Feature selection
    use_feature_selection = True
    feature_selection_threshold = 0.01
    
    # GANDALF settings
    gandalf_hidden_dims = [256, 128, 64]
    gandalf_dropout = 0.3
    gandalf_batch_size = 1024
    gandalf_epochs = 50
    gandalf_lr = 0.001
    gandalf_patience = 10
    
    # Discriminator settings
    disc_hidden_dims = [128, 64, 32]
    disc_lr = 0.0005
    disc_epochs = 30

# Memory optimization
def reduce_mem_usage(df, name=""):
    print(f"Optimizing memory for {name}...")
    start_mem = df.memory_usage().sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage: {start_mem:.2f} MB -> {end_mem:.2f} MB ({100*(start_mem-end_mem)/start_mem:.1f}% reduction)')
    return df

# ====================== GANDALF Architecture ======================

class GatedUnit(nn.Module):
    """Gated Linear Unit for feature selection"""
    def __init__(self, input_dim, output_dim):
        super(GatedUnit, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        
    def forward(self, x):
        return self.fc(x) * torch.sigmoid(self.gate(x))

class AttentionLayer(nn.Module):
    """Multi-head attention for feature interactions"""
    def __init__(self, input_dim, num_heads=4):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.fc_out = nn.Linear(input_dim, input_dim)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.transpose(1, 2).contiguous().view(batch_size, -1)
        
        return self.fc_out(attention_output)

class GANDALF(nn.Module):
    """GANDALF: Gated Adaptive Network for Deep Automated Learning of Features"""
    def __init__(self, input_dim, hidden_dims, dropout=0.3):
        super(GANDALF, self).__init__()
        
        # Feature extraction layers with gating
        self.gated_layers = nn.ModuleList()
        dims = [input_dim] + hidden_dims
        
        for i in range(len(dims) - 1):
            self.gated_layers.append(GatedUnit(dims[i], dims[i+1]))
        
        # Attention mechanism for feature interactions
        self.attention = AttentionLayer(hidden_dims[-1])
        
        # Batch normalization layers
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(dim) for dim in hidden_dims])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        
        # Residual connections
        self.residual_weight = nn.Parameter(torch.tensor(0.1))
        
    def forward(self, x):
        # Store original input for residual connection
        original = x
        
        # Pass through gated layers
        for i, (gated_layer, bn) in enumerate(zip(self.gated_layers, self.batch_norms)):
            x = gated_layer(x)
            x = bn(x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Apply attention mechanism
        x = self.attention(x)
        
        # Output with residual connection
        output = self.output_layer(x)
        
        # Add weighted residual from original features
        if original.size(1) == 1:
            residual = original
        else:
            # Simple linear projection for residual
            residual = original.mean(dim=1, keepdim=True)
        
        output = output + self.residual_weight * residual
        
        return output.squeeze()

class Discriminator(nn.Module):
    """Discriminator network for anti-overfitting"""
    def __init__(self, input_dim, hidden_dims):
        super(Discriminator, self).__init__()
        
        layers = []
        dims = [input_dim] + hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(0.2))
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return torch.sigmoid(self.model(x))

class GANDALFTrainer:
    """Trainer class for GANDALF with anti-overfitting discriminator"""
    def __init__(self, input_dim, config):
        self.config = config
        self.device = config.device
        
        # Initialize GANDALF model
        self.gandalf = GANDALF(
            input_dim=input_dim,
            hidden_dims=config.gandalf_hidden_dims,
            dropout=config.gandalf_dropout
        ).to(self.device)
        
        # Initialize discriminator
        self.discriminator = Discriminator(
            input_dim=input_dim + 1,  # features + prediction
            hidden_dims=config.disc_hidden_dims
        ).to(self.device)
        
        # Optimizers
        self.gandalf_optimizer = optim.Adam(self.gandalf.parameters(), lr=config.gandalf_lr)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=config.disc_lr)
        
        # Loss functions
        self.regression_loss = nn.MSELoss()
        self.adversarial_loss = nn.BCELoss()
        
        # Learning rate schedulers
        self.gandalf_scheduler = ReduceLROnPlateau(
            self.gandalf_optimizer, mode='min', patience=5, factor=0.5
        )
        self.disc_scheduler = ReduceLROnPlateau(
            self.disc_optimizer, mode='min', patience=5, factor=0.5
        )
        
    def train_discriminator(self, real_data, fake_data):
        """Train discriminator to distinguish between real and overfitted predictions"""
        self.discriminator.train()
        
        # Real data (good predictions)
        real_labels = torch.ones(real_data.size(0), 1).to(self.device)
        real_pred = self.discriminator(real_data)
        real_loss = self.adversarial_loss(real_pred, real_labels)
        
        # Fake data (overfitted predictions)
        fake_labels = torch.zeros(fake_data.size(0), 1).to(self.device)
        fake_pred = self.discriminator(fake_data)
        fake_loss = self.adversarial_loss(fake_pred, fake_labels)
        
        # Total discriminator loss
        disc_loss = (real_loss + fake_loss) / 2
        
        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()
        
        return disc_loss.item()
    
    def train_gandalf(self, X, y, X_val, y_val):
        """Train GANDALF with adversarial regularization"""
        self.gandalf.train()
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X).to(self.device),
            torch.FloatTensor(y).to(self.device)
        )
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.gandalf_batch_size, 
            shuffle=True
        )
        
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val).to(self.device),
            torch.FloatTensor(y_val).to(self.device)
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.gandalf_batch_size
        )
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.gandalf_epochs):
            # Training phase
            train_losses = []
            for batch_X, batch_y in train_loader:
                # Forward pass
                predictions = self.gandalf(batch_X)
                
                # Regression loss
                reg_loss = self.regression_loss(predictions, batch_y)
                
                # Adversarial loss - fool discriminator
                disc_input = torch.cat([batch_X, predictions.unsqueeze(1)], dim=1)
                disc_pred = self.discriminator(disc_input)
                adv_loss = self.adversarial_loss(
                    disc_pred, 
                    torch.ones_like(disc_pred)  # Want discriminator to think it's real
                )
                
                # Total loss
                total_loss = reg_loss + 0.1 * adv_loss
                
                # Backward pass
                self.gandalf_optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gandalf.parameters(), 1.0)
                self.gandalf_optimizer.step()
                
                train_losses.append(total_loss.item())
            
            # Validation phase
            self.gandalf.eval()
            val_losses = []
            val_predictions = []
            val_targets = []
            
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    predictions = self.gandalf(batch_X)
                    val_loss = self.regression_loss(predictions, batch_y)
                    val_losses.append(val_loss.item())
                    val_predictions.extend(predictions.cpu().numpy())
                    val_targets.extend(batch_y.cpu().numpy())
            
            # Calculate validation correlation
            val_corr = pearsonr(val_predictions, val_targets)[0]
            avg_val_loss = np.mean(val_losses)
            
            # Update learning rate
            self.gandalf_scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model state
                self.best_gandalf_state = self.gandalf.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= self.config.gandalf_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{self.config.gandalf_epochs} - "
                      f"Train Loss: {np.mean(train_losses):.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, "
                      f"Val Corr: {val_corr:.4f}")
            
            self.gandalf.train()
        
        # Load best model
        self.gandalf.load_state_dict(self.best_gandalf_state)
    
    def train_anti_overfit_discriminator(self, X_train, y_train, X_val, y_val):
        """Train discriminator to detect overfitting patterns"""
        print("\nTraining anti-overfitting discriminator...")
        
        # Generate predictions on training and validation sets
        self.gandalf.eval()
        with torch.no_grad():
            train_preds = self.gandalf(torch.FloatTensor(X_train).to(self.device))
            val_preds = self.gandalf(torch.FloatTensor(X_val).to(self.device))
        
        # Calculate prediction errors
        train_errors = np.abs(train_preds.cpu().numpy() - y_train)
        val_errors = np.abs(val_preds.cpu().numpy() - y_val)
        
        # Create discriminator training data
        # High error samples are "fake" (overfitted)
        error_threshold = np.percentile(train_errors, 75)
        
        real_mask = train_errors < error_threshold
        fake_mask = train_errors >= error_threshold
        
        real_features = np.concatenate([
            X_train[real_mask],
            train_preds.cpu().numpy()[real_mask].reshape(-1, 1)
        ], axis=1)
        
        fake_features = np.concatenate([
            X_train[fake_mask],
            train_preds.cpu().numpy()[fake_mask].reshape(-1, 1)
        ], axis=1)
        
        # Train discriminator
        for epoch in range(self.config.disc_epochs):
            # Sample batch
            batch_size = min(len(real_features), len(fake_features), 512)
            real_batch = real_features[np.random.choice(len(real_features), batch_size)]
            fake_batch = fake_features[np.random.choice(len(fake_features), batch_size)]
            
            real_tensor = torch.FloatTensor(real_batch).to(self.device)
            fake_tensor = torch.FloatTensor(fake_batch).to(self.device)
            
            disc_loss = self.train_discriminator(real_tensor, fake_tensor)
            
            if (epoch + 1) % 10 == 0:
                print(f"Discriminator Epoch {epoch+1}/{self.config.disc_epochs} - Loss: {disc_loss:.4f}")
    
    def predict(self, X):
        """Make predictions with GANDALF"""
        self.gandalf.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.gandalf(X_tensor)
        return predictions.cpu().numpy()
    
    def get_overfit_scores(self, X):
        """Get overfitting scores from discriminator"""
        self.gandalf.eval()
        self.discriminator.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.gandalf(X_tensor)
            disc_input = torch.cat([X_tensor, predictions.unsqueeze(1)], dim=1)
            overfit_scores = self.discriminator(disc_input)
        return overfit_scores.cpu().numpy().squeeze()

# ====================== Feature Engineering ======================

def create_proprietary_x_variables(df, n_features=30):
    """Create proprietary X variables with focus on quality over quantity"""
    print(f"Creating {n_features} proprietary X variables...")
    
    # Get existing X features
    x_features = [col for col in df.columns if col.startswith('X') and col[1:].isdigit()]
    
    # Important X features from original pipeline
    important_x = ["X752", "X287", "X298", "X759", "X302", "X55", "X56", "X52", "X303", "X51",
                   "X344", "X598", "X385", "X603", "X674", "X415", "X345", "X137", "X174", "X178"]
    
    # Use available important features
    base_features = [f for f in important_x if f in df.columns][:10]
    
    # Add some high-variance features
    if len(base_features) < 10:
        x_variances = df[x_features].var()
        high_var_features = x_variances.nlargest(10).index.tolist()
        for feat in high_var_features:
            if feat not in base_features:
                base_features.append(feat)
                if len(base_features) >= 10:
                    break
    
    prop_idx = 1
    
    # 1. Statistical combinations (8 features)
    if len(base_features) >= 5:
        df[f'X_prop_{prop_idx}'] = df[base_features[:5]].mean(axis=1)
        prop_idx += 1
        df[f'X_prop_{prop_idx}'] = df[base_features[:5]].std(axis=1)
        prop_idx += 1
        df[f'X_prop_{prop_idx}'] = df[base_features[:5]].max(axis=1) - df[base_features[:5]].min(axis=1)
        prop_idx += 1
        df[f'X_prop_{prop_idx}'] = df[base_features[:5]].median(axis=1)
        prop_idx += 1
        df[f'X_prop_{prop_idx}'] = df[base_features[:7]].quantile(0.25, axis=1)
        prop_idx += 1
        df[f'X_prop_{prop_idx}'] = df[base_features[:7]].quantile(0.75, axis=1)
        prop_idx += 1
        df[f'X_prop_{prop_idx}'] = (df[base_features[:5]] > df[base_features[:5]].mean(axis=1).values[:, None]).sum(axis=1)
        prop_idx += 1
        df[f'X_prop_{prop_idx}'] = df[base_features[:5]].idxmax(axis=1).str.extract('(\d+)')[0].astype(float)
        prop_idx += 1
    
    # 2. Non-linear transformations (7 features)
    for i in range(7):
        feat = base_features[i % len(base_features)]
        if i < 2:
            df[f'X_prop_{prop_idx}'] = np.sign(df[feat]) * np.sqrt(np.abs(df[feat]))
        elif i < 4:
            df[f'X_prop_{prop_idx}'] = np.tanh(df[feat] / df[feat].std())
        elif i < 6:
            df[f'X_prop_{prop_idx}'] = 1 / (1 + np.exp(-df[feat] / df[feat].std()))  # Sigmoid
        else:
            df[f'X_prop_{prop_idx}'] = rankdata(df[feat]) / len(df)  # Rank transform
        prop_idx += 1
    
    # 3. Market interaction features (10 features)
    if 'volume' in df.columns:
        for i in range(3):
            df[f'X_prop_{prop_idx}'] = df[base_features[i]] * np.log1p(df['volume'])
            prop_idx += 1
    
    if 'order_flow_imbalance' in df.columns:
        for i in range(3):
            df[f'X_prop_{prop_idx}'] = df[base_features[i+3]] * df['order_flow_imbalance']
            prop_idx += 1
    
    if 'kyle_lambda' in df.columns:
        for i in range(2):
            df[f'X_prop_{prop_idx}'] = df[base_features[i+6]] * np.sign(df['kyle_lambda']) * np.log1p(np.abs(df['kyle_lambda']))
            prop_idx += 1
    
    if 'vpin' in df.columns:
        for i in range(2):
            df[f'X_prop_{prop_idx}'] = df[base_features[i+8]] * df['vpin']
            prop_idx += 1
    
    # 4. Interaction ratios (5 features)
    for i in range(5):
        feat1 = base_features[i % len(base_features)]
        feat2 = base_features[(i + 1) % len(base_features)]
        df[f'X_prop_{prop_idx}'] = df[feat1] / (np.abs(df[feat2]) + 1e-8)
        prop_idx += 1
    
    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    print(f"Created {prop_idx-1} proprietary X variables")
    return df

def create_interaction_features(df, selected_features, n_interactions=50):
    """Create high-quality interaction features"""
    print(f"Creating {n_interactions} interaction features...")
    
    interaction_features = []
    feature_names = []
    
    # Prioritize features
    important_x = ["X752", "X287", "X298", "X759", "X302", "X55", "X56", "X52", "X303", "X51"]
    market_features = ['order_flow_imbalance', 'kyle_lambda', 'vpin', 'liquidity_imbalance',
                      'bid_ask_spread', 'buying_pressure', 'volume', 'log_volume']
    
    # Get available priority features
    priority_x = [f for f in important_x if f in selected_features and f in df.columns][:10]
    priority_market = [f for f in market_features if f in selected_features and f in df.columns][:8]
    
    interaction_count = 0
    
    # 1. X features with market microstructure (20 interactions)
    for i, x_feat in enumerate(priority_x[:10]):
        if interaction_count >= 20:
            break
        for j, market_feat in enumerate(priority_market[:4]):
            if interaction_count >= 20:
                break
            
            # Multiplication
            interaction_features.append(df[x_feat] * df[market_feat])
            feature_names.append(f'{x_feat}_x_{market_feat}')
            interaction_count += 1
            
            # Log interaction
            if interaction_count < 20 and market_feat in ['volume', 'kyle_lambda']:
                interaction_features.append(df[x_feat] * np.log1p(np.abs(df[market_feat])))
                feature_names.append(f'{x_feat}_x_log_{market_feat}')
                interaction_count += 1
    
    # 2. Market feature interactions (15 interactions)
    market_pairs = [
        ('order_flow_imbalance', 'kyle_lambda'),
        ('vpin', 'liquidity_imbalance'),
        ('buying_pressure', 'selling_pressure'),
        ('bid_ask_spread', 'total_liquidity'),
        ('volume', 'liquidity_ratio')
    ]
    
    for feat1, feat2 in market_pairs:
        if interaction_count >= 35:
            break
        if feat1 in df.columns and feat2 in df.columns:
            # Product
            interaction_features.append(df[feat1] * df[feat2])
            feature_names.append(f'{feat1}_x_{feat2}')
            interaction_count += 1
            
            # Ratio
            if interaction_count < 35:
                interaction_features.append(df[feat1] / (np.abs(df[feat2]) + 1e-8))
                feature_names.append(f'{feat1}_div_{feat2}')
                interaction_count += 1
            
            # Difference
            if interaction_count < 35:
                interaction_features.append(df[feat1] - df[feat2])
                feature_names.append(f'{feat1}_minus_{feat2}')
                interaction_count += 1
    
    # 3. Non-linear interactions (10 interactions)
    for i in range(5):
        if interaction_count >= 45:
            break
        if i < len(priority_x):
            feat = priority_x[i]
            
            # Squared
            interaction_features.append(df[feat] ** 2)
            feature_names.append(f'{feat}_squared')
            interaction_count += 1
            
            # Square root of absolute
            if interaction_count < 45:
                interaction_features.append(np.sqrt(np.abs(df[feat])))
                feature_names.append(f'{feat}_sqrt_abs')
                interaction_count += 1
    
    # 4. Three-way interactions (5 interactions)
    if len(priority_x) >= 3 and len(priority_market) >= 1:
        for i in range(5):
            if interaction_count >= n_interactions:
                break
            x1 = priority_x[i % len(priority_x)]
            x2 = priority_x[(i+1) % len(priority_x)]
            m1 = priority_market[i % len(priority_market)]
            
            interaction_features.append(df[x1] * df[x2] * df[m1])
            feature_names.append(f'{x1}_{x2}_{m1}')
            interaction_count += 1
    
    # Create DataFrame
    interaction_df = pd.DataFrame(
        np.column_stack(interaction_features[:interaction_count]),
        columns=feature_names[:interaction_count],
        index=df.index
    )
    
    # Handle infinities and NaN
    interaction_df = interaction_df.replace([np.inf, -np.inf], np.nan)
    interaction_df = interaction_df.fillna(0)
    
    print(f"Created {interaction_count} interaction features")
    return interaction_df

def add_features(df):
    """Create comprehensive features for market microstructure"""
    print("Engineering features...")
    
    # Basic interactions
    df['bid_ask_spread'] = df['ask_qty'] - df['bid_qty']
    df['bid_ask_ratio'] = df['bid_qty'] / (df['ask_qty'] + 1e-8)
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-8)
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['volume'] + 1e-8)
    
    # Pressure indicators
    df['buying_pressure'] = df['buy_qty'] / (df['volume'] + 1e-8)
    df['selling_pressure'] = df['sell_qty'] / (df['volume'] + 1e-8)
    df['net_pressure'] = df['buying_pressure'] - df['selling_pressure']
    
    # Liquidity features
    df['total_liquidity'] = df['bid_qty'] + df['ask_qty']
    df['liquidity_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_liquidity'] + 1e-8)
    df['liquidity_ratio'] = df['total_liquidity'] / (df['volume'] + 1e-8)
    
    # Volume transformations
    df['log_volume'] = np.log1p(df['volume'])
    df['sqrt_volume'] = np.sqrt(df['volume'])
    
    # Market microstructure
    df['kyle_lambda'] = df['order_flow_imbalance'] / (df['sqrt_volume'] + 1e-8)
    df['vpin'] = np.abs(df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-8)
    
    # Additional useful features
    df['effective_spread'] = 2 * np.abs(df['order_flow_imbalance']) * df['bid_ask_spread']
    df['realized_spread'] = df['bid_ask_spread'] * df['vpin']
    df['price_impact'] = df['kyle_lambda'] * df['volume']
    df['trade_intensity'] = df['volume'] / (df['total_liquidity'] + 1e-8)
    
    # Handle infinities and NaN
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    return df

def select_features_by_importance(X_train, y_train, feature_names, threshold=0.01):
    """Select features based on importance scores"""
    print("Calculating feature importance...")
    
    # Train a quick model to get feature importance
    params = {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42,
        'n_jobs': -1,
        'verbosity': 0
    }
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Get feature importance
    importance = model.feature_importances_
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # Select features above threshold
    selected_features = importance_df[importance_df['importance'] > threshold]['feature'].tolist()
    
    # Always include critical features
    critical_features = ['order_flow_imbalance', 'kyle_lambda', 'vpin', 'volume', 
                        'bid_ask_spread', 'liquidity_imbalance', 'buying_pressure']
    
    for feat in critical_features:
        if feat in feature_names and feat not in selected_features:
            selected_features.append(feat)
    
    print(f"Selected {len(selected_features)} features with importance > {threshold}")
    
    return selected_features, importance_df

# ====================== XGBoost Anti-overfitting ======================

class AntiOverfitXGB:
    def __init__(self, base_params=None):
        self.base_params = base_params or {
            'tree_method': 'hist',
            'n_estimators': 500,
            'learning_rate': 0.01,
            'max_depth': 6,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        
    def train_overfit_model(self, X, y, overfit_direction='high'):
        """Train a model designed to overfit in a specific direction"""
        params = self.base_params.copy()
        
        if overfit_direction == 'high':
            params.update({
                'max_depth': 12,
                'min_child_weight': 1,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'reg_alpha': 0.001,
                'reg_lambda': 0.001,
                'learning_rate': 0.05
            })
        else:
            params.update({
                'max_depth': 3,
                'min_child_weight': 50,
                'subsample': 0.5,
                'colsample_bytree': 0.5,
                'reg_alpha': 10,
                'reg_lambda': 10,
                'learning_rate': 0.001
            })
        
        model = XGBRegressor(**params)
        model.fit(X, y)
        return model
    
    def identify_overfit_samples(self, X, y, n_folds=5):
        """Identify samples where model tends to overfit"""
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        
        errors = np.zeros(len(X))
        predictions = np.zeros(len(X))
        
        for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
            X_fold_train = X.iloc[train_idx] if hasattr(X, 'iloc') else X[train_idx]
            y_fold_train = y.iloc[train_idx] if hasattr(y, 'iloc') else y[train_idx]
            X_fold_valid = X.iloc[valid_idx] if hasattr(X, 'iloc') else X[valid_idx]
            y_fold_valid = y.iloc[valid_idx] if hasattr(y, 'iloc') else y[valid_idx]
            
            model = self.train_overfit_model(X_fold_train, y_fold_train, 'high')
            pred = model.predict(X_fold_valid)
            predictions[valid_idx] = pred
            errors[valid_idx] = np.abs(pred - y_fold_valid)
        
        error_threshold = np.percentile(errors, 75)
        overfit_mask = errors > error_threshold
        
        return overfit_mask, predictions, errors
    
    def train_adversarial_ensemble(self, X, y, X_test):
        """Train ensemble with models that overfit in opposite directions"""
        print("Training high-overfitting model...")
        model_high = self.train_overfit_model(X, y, 'high')
        pred_high_train = model_high.predict(X)
        pred_high_test = model_high.predict(X_test)
        
        print("Training low-overfitting model...")
        model_low = self.train_overfit_model(X, y, 'low')
        pred_low_train = model_low.predict(X)
        pred_low_test = model_low.predict(X_test)
        
        print("Training residual model...")
        residuals = y - (pred_high_train + pred_low_train) / 2
        params = self.base_params.copy()
        params.update({
            'max_depth': 6,
            'learning_rate': 0.02,
            'subsample': 0.7,
            'colsample_bytree': 0.7
        })
        model_residual = XGBRegressor(**params)
        model_residual.fit(X, residuals)
        pred_residual_train = model_residual.predict(X)
        pred_residual_test = model_residual.predict(X_test)
        
        final_train = (pred_high_train + pred_low_train) / 2 + pred_residual_train
        final_test = (pred_high_test + pred_low_test) / 2 + pred_residual_test
        
        return final_train, final_test
    
    def train_with_sample_weights(self, X, y, overfit_mask):
        """Train model with reduced weights on overfit-prone samples"""
        sample_weights = np.ones(len(X))
        sample_weights[overfit_mask] = 0.5
        
        params = self.base_params.copy()
        params.update({
            'max_depth': 8,
            'learning_rate': 0.02,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 1,
            'reg_lambda': 1
        })
        
        model = XGBRegressor(**params)
        model.fit(X, y, sample_weight=sample_weights)
        return model

# ====================== Ensemble Optimization ======================

def optimize_ensemble_weights(predictions, y_true, method='advanced'):
    """Optimize ensemble weights using multiple methods"""
    n_models = len(predictions)
    
    def objective(weights):
        weights = weights / weights.sum()
        blended = np.sum([w * p for w, p in zip(weights, predictions)], axis=0)
        return -pearsonr(y_true, blended)[0]
    
    best_weights = None
    best_score = float('inf')
    
    # Method 1: Equal weights
    equal_weights = np.ones(n_models) / n_models
    equal_score = objective(equal_weights)
    if equal_score < best_score:
        best_score = equal_score
        best_weights = equal_weights
    
    # Method 2: SLSQP
    constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
    bounds = [(0, 1) for _ in range(n_models)]
    result = minimize(objective, equal_weights, method='SLSQP', 
                     bounds=bounds, constraints=constraints)
    if result.success and result.fun < best_score:
        best_score = result.fun
        best_weights = result.x
    
    # Method 3: Differential Evolution
    if method == 'advanced':
        try:
            result_de = differential_evolution(objective, bounds, seed=42, maxiter=100)
            de_weights = result_de.x / result_de.x.sum()
            if result_de.fun < best_score:
                best_score = result_de.fun
                best_weights = de_weights
        except:
            pass
    
    return best_weights

# ====================== Main Pipeline ======================

print("\nLoading data...")
train = pd.read_parquet(CFG.train_path)
test = pd.read_parquet(CFG.test_path)
submission = pd.read_csv(CFG.sample_sub_path)

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Create proprietary features
train = create_proprietary_x_variables(train, CFG.n_proprietary_features)
test = create_proprietary_x_variables(test, CFG.n_proprietary_features)

# Feature engineering
train = add_features(train)
test = add_features(test)

# Memory optimization
train = reduce_mem_usage(train, "train")
test = reduce_mem_usage(test, "test")

# Select base features
selected_x_features = [
    "X752", "X287", "X298", "X759", "X302", "X55", "X56", "X52", "X303", "X51",
    "X344", "X598", "X385", "X603", "X674", "X415", "X345", "X137", "X174", "X178"
]

# Add proprietary features
proprietary_features = [f"X_prop_{i}" for i in range(1, CFG.n_proprietary_features + 1)]
selected_x_features.extend(proprietary_features)

# Get all X features and add more if needed
all_x_features = [col for col in train.columns if col.startswith('X') and col[1:].isdigit()]
additional_x = [f for f in all_x_features if f not in selected_x_features][:CFG.max_x_features - len(selected_x_features)]
selected_x_features.extend(additional_x)

available_x_features = [f for f in selected_x_features if f in train.columns]

# Market and engineered features
market_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']
engineered_features = [col for col in train.columns if col in [
    'bid_ask_spread', 'bid_ask_ratio', 'buy_sell_ratio', 'order_flow_imbalance',
    'buying_pressure', 'selling_pressure', 'net_pressure', 'total_liquidity', 
    'liquidity_imbalance', 'liquidity_ratio', 'log_volume', 'sqrt_volume',
    'kyle_lambda', 'vpin', 'effective_spread', 'realized_spread', 
    'price_impact', 'trade_intensity'
]]

# Combine base features
base_selected_features = market_features + available_x_features + engineered_features
base_selected_features = list(dict.fromkeys(base_selected_features))
base_selected_features = [f for f in base_selected_features if f in train.columns]

print(f"\nBase features: {len(base_selected_features)}")

# Create interaction features
interaction_df_train = create_interaction_features(train, base_selected_features, CFG.n_interaction_features)
interaction_df_test = create_interaction_features(test, base_selected_features, CFG.n_interaction_features)

# Add interaction features
for col in interaction_df_train.columns:
    train[col] = interaction_df_train[col]
    test[col] = interaction_df_test[col]

# All features before selection
all_features = base_selected_features + list(interaction_df_train.columns)
print(f"Total features before selection: {len(all_features)}")

# Prepare data for feature selection
X_train_all = train[all_features]
y_train = train['label']
X_test_all = test[all_features]

# Feature selection
if CFG.use_feature_selection:
    selected_features, importance_df = select_features_by_importance(
        X_train_all, y_train, all_features, CFG.feature_selection_threshold
    )
    print(f"\nTop 10 features by importance:")
    print(importance_df.head(10))
else:
    selected_features = all_features

X_train = X_train_all[selected_features]
X_test = X_test_all[selected_features]

print(f"\nFinal features after selection: {len(selected_features)}")

# Initialize storage for predictions
all_predictions_train = []
all_predictions_test = []
all_scores = []
model_names = []

# Initialize anti-overfitting trainer
anti_overfit = AntiOverfitXGB()

# ========== XGBoost Models ==========

print("\n" + "="*60)
print("Training XGBoost Models")
print("="*60)

# Strategy 1: Standard scaling with adversarial ensemble
print("\nStrategy 1: XGBoost Adversarial Ensemble (Standard Scaling)")
print("-"*60)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=selected_features,
    index=X_train.index
)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test),
    columns=selected_features,
    index=X_test.index
)

# Identify overfit-prone samples
print("\nIdentifying overfit-prone samples...")
overfit_mask, oof_predictions, errors = anti_overfit.identify_overfit_samples(
    X_train_scaled, y_train, n_folds=5
)
print(f"Found {overfit_mask.sum()} overfit-prone samples ({100*overfit_mask.mean():.1f}%)")

# Train adversarial ensemble
print("\nTraining adversarial ensemble...")
adv_train, adv_test = anti_overfit.train_adversarial_ensemble(
    X_train_scaled, y_train, X_test_scaled
)

adv_score = pearsonr(y_train, adv_train)[0]
print(f"XGB Adversarial ensemble score: {adv_score:.4f}")

all_predictions_train.append(adv_train)
all_predictions_test.append(adv_test)
all_scores.append(adv_score)
model_names.append('xgb_adversarial_standard')

# Train with sample weights
print("\nTraining with adjusted sample weights...")
weighted_model = anti_overfit.train_with_sample_weights(
    X_train_scaled, y_train, overfit_mask
)
weighted_train = weighted_model.predict(X_train_scaled)
weighted_test = weighted_model.predict(X_test_scaled)

weighted_score = pearsonr(y_train, weighted_train)[0]
print(f"XGB Weighted model score: {weighted_score:.4f}")

all_predictions_train.append(weighted_train)
all_predictions_test.append(weighted_test)
all_scores.append(weighted_score)
model_names.append('xgb_weighted_standard')

# Strategy 2: Robust scaling with diverse models
print("\nStrategy 2: XGBoost Diverse Models (Robust Scaling)")
print("-"*60)

scaler_robust = RobustScaler()
X_train_robust = pd.DataFrame(
    scaler_robust.fit_transform(X_train),
    columns=selected_features,
    index=X_train.index
)
X_test_robust = pd.DataFrame(
    scaler_robust.transform(X_test),
    columns=selected_features,
    index=X_test.index
)

# Train diverse models
diverse_configs = [
    {
        'name': 'conservative',
        'params': {
            'n_estimators': 800,
            'max_depth': 5,
            'learning_rate': 0.012,
            'subsample': 0.65,
            'colsample_bytree': 0.65,
            'reg_alpha': 2.5,
            'reg_lambda': 2.5,
            'min_child_weight': 25,
            'gamma': 0.3
        }
    },
    {
        'name': 'balanced',
        'params': {
            'n_estimators': 600,
            'max_depth': 7,
            'learning_rate': 0.018,
            'subsample': 0.75,
            'colsample_bytree': 0.75,
            'reg_alpha': 1.0,
            'reg_lambda': 1.0,
            'min_child_weight': 10,
            'gamma': 0.1
        }
    },
    {
        'name': 'aggressive',
        'params': {
            'n_estimators': 500,
            'max_depth': 9,
            'learning_rate': 0.025,
            'subsample': 0.85,
            'colsample_bytree': 0.85,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'min_child_weight': 5,
            'gamma': 0.05
        }
    }
]

for config in diverse_configs:
    print(f"\nTraining {config['name']} model...")
    params = anti_overfit.base_params.copy()
    params.update(config['params'])
    
    model = XGBRegressor(**params)
    model.fit(X_train_robust, y_train)
    
    pred_train = model.predict(X_train_robust)
    pred_test = model.predict(X_test_robust)
    score = pearsonr(y_train, pred_train)[0]
    
    print(f"  Score: {score:.4f}")
    
    all_predictions_train.append(pred_train)
    all_predictions_test.append(pred_test)
    all_scores.append(score)
    model_names.append(f'xgb_{config["name"]}_robust')

# ========== GANDALF Models ==========

print("\n" + "="*60)
print("Training GANDALF Models")
print("="*60)

# Prepare data for GANDALF
kf = KFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = next(kf.split(X_train_scaled))
X_train_gandalf = X_train_scaled.iloc[train_idx].values
y_train_gandalf = y_train.iloc[train_idx].values
X_val_gandalf = X_train_scaled.iloc[val_idx].values
y_val_gandalf = y_train.iloc[val_idx].values

# Train GANDALF with anti-overfitting
print("\nTraining GANDALF with anti-overfitting discriminator...")
gandalf_trainer = GANDALFTrainer(X_train_gandalf.shape[1], CFG)

# Train main GANDALF model
gandalf_trainer.train_gandalf(X_train_gandalf, y_train_gandalf, X_val_gandalf, y_val_gandalf)

# Train anti-overfitting discriminator
gandalf_trainer.train_anti_overfit_discriminator(
    X_train_gandalf, y_train_gandalf, X_val_gandalf, y_val_gandalf
)

# Get predictions
gandalf_train_pred = gandalf_trainer.predict(X_train_scaled.values)
gandalf_test_pred = gandalf_trainer.predict(X_test_scaled.values)

# Get overfitting scores
overfit_scores = gandalf_trainer.get_overfit_scores(X_train_scaled.values)
print(f"Average overfitting score: {overfit_scores.mean():.4f}")

gandalf_score = pearsonr(y_train, gandalf_train_pred)[0]
print(f"GANDALF model score: {gandalf_score:.4f}")

all_predictions_train.append(gandalf_train_pred)
all_predictions_test.append(gandalf_test_pred)
all_scores.append(gandalf_score)
model_names.append('gandalf_with_discriminator')

# Train GANDALF variant with different architecture
print("\nTraining GANDALF variant (smaller architecture)...")
CFG_small = CFG()
CFG_small.gandalf_hidden_dims = [128, 64, 32]
CFG_small.gandalf_dropout = 0.4

gandalf_small_trainer = GANDALFTrainer(X_train_gandalf.shape[1], CFG_small)
gandalf_small_trainer.train_gandalf(X_train_gandalf, y_train_gandalf, X_val_gandalf, y_val_gandalf)

gandalf_small_train_pred = gandalf_small_trainer.predict(X_train_scaled.values)
gandalf_small_test_pred = gandalf_small_trainer.predict(X_test_scaled.values)

gandalf_small_score = pearsonr(y_train, gandalf_small_train_pred)[0]
print(f"GANDALF small model score: {gandalf_small_score:.4f}")

all_predictions_train.append(gandalf_small_train_pred)
all_predictions_test.append(gandalf_small_test_pred)
all_scores.append(gandalf_small_score)
model_names.append('gandalf_small')

# ========== Time Windows (for XGBoost) ==========

print("\n" + "="*60)
print("Training Time-Window Models")
print("="*60)

windows = [
    {'name': 'recent_70', 'start': int(0.7 * len(train)), 'end': len(train)},
    {'name': 'recent_50', 'start': int(0.5 * len(train)), 'end': len(train)},
    {'name': 'middle', 'start': int(0.3 * len(train)), 'end': int(0.7 * len(train))}
]

for window in windows:
    print(f"\nTraining on {window['name']} window...")
    X_window = X_train_scaled.iloc[window['start']:window['end']]
    y_window = y_train.iloc[window['start']:window['end']]
    
    params = anti_overfit.base_params.copy()
    params.update({
        'max_depth': 7,
        'learning_rate': 0.02,
        'subsample': 0.75,
        'colsample_bytree': 0.75,
        'n_estimators': 600
    })
    
    model = XGBRegressor(**params)
    model.fit(X_window, y_window)
    
    pred_test = model.predict(X_test_scaled)
    
    # For scoring, predict on the window
    pred_window = model.predict(X_window)
    window_score = pearsonr(y_window, pred_window)[0]
    
    print(f"  Window score: {window_score:.4f}")
    
    all_predictions_test.append(pred_test)
    all_predictions_train.append(np.zeros_like(y_train))  # Placeholder
    all_scores.append(window_score)
    model_names.append(f'xgb_window_{window["name"]}')

# ========== Model Summary ==========

print("\n" + "="*60)
print("Model Performance Summary")
print("="*60)
for name, score in zip(model_names, all_scores):
    model_type = "XGBoost" if "xgb" in name else "GANDALF"
    print(f"{name:30s} ({model_type:7s}): {score:.4f}")

# ========== Ensemble Optimization ==========

print("\n" + "="*60)
print("Creating Optimized XGBoost + GANDALF Ensemble")
print("="*60)

# Use models with valid scores for optimization
valid_indices = [i for i, score in enumerate(all_scores) if score > 0.5]
valid_predictions_train = [all_predictions_train[i] for i in valid_indices]
valid_model_names = [model_names[i] for i in valid_indices]

# Optimize weights
print("\nOptimizing ensemble weights...")
optimal_weights = optimize_ensemble_weights(valid_predictions_train, y_train, method='advanced')

print("\nOptimal weights:")
for name, weight in zip(valid_model_names, optimal_weights):
    if weight > 0.01:
        model_type = "XGBoost" if "xgb" in name else "GANDALF"
        print(f"  {name:30s} ({model_type:7s}): {weight:.3f}")

# Create final ensemble
final_predictions = np.zeros_like(all_predictions_test[0])

# Apply optimized weights to valid models
for i, idx in enumerate(valid_indices):
    final_predictions += optimal_weights[i] * all_predictions_test[idx]

# Add window predictions with fixed weight
window_indices = [i for i, name in enumerate(model_names) if 'window' in name]
if window_indices:
    window_avg = np.mean([all_predictions_test[i] for i in window_indices], axis=0)
    final_predictions = 0.7 * final_predictions + 0.3 * window_avg

# Post-processing
p1, p99 = np.percentile(y_train, [1, 99])
final_predictions = np.clip(final_predictions, p1, p99)

# Create submission
submission['prediction'] = final_predictions
submission.to_csv('submission_xgb_gandalf.csv', index=False)

print("\n" + "="*80)
print("Submission saved to submission_xgb_gandalf.csv")
print(submission.head())

# Save detailed predictions and model analysis
predictions_df = pd.DataFrame({
    'xgb_adversarial': all_predictions_test[0],
    'xgb_weighted': all_predictions_test[1],
    'gandalf_main': all_predictions_test[model_names.index('gandalf_with_discriminator')],
    'final': final_predictions
})
predictions_df.to_csv('prediction_components_xgb_gandalf.csv', index=False)

print("\nPrediction statistics:")
print(predictions_df.describe())

# Model diversity analysis
print("\n" + "="*60)
print("Model Diversity Analysis")
print("="*60)

# Separate XGBoost and GANDALF models
xgb_indices = [i for i, name in enumerate(model_names) if 'xgb' in name and i in valid_indices]
gandalf_indices = [i for i, name in enumerate(model_names) if 'gandalf' in name and i in valid_indices]

if xgb_indices and gandalf_indices:
    xgb_preds = [all_predictions_train[i] for i in xgb_indices]
    gandalf_preds = [all_predictions_train[i] for i in gandalf_indices]
    
    # Inter-model correlation
    inter_corr = np.corrcoef(
        np.mean(xgb_preds, axis=0),
        np.mean(gandalf_preds, axis=0)
    )[0, 1]
    print(f"XGBoost-GANDALF correlation: {inter_corr:.4f}")

print("\n" + "="*80)
print("XGBoost + GANDALF pipeline completed successfully!")
print(f"Total models trained: {len(model_names)}")
print(f"XGBoost models: {len([n for n in model_names if 'xgb' in n])}")
print(f"GANDALF models: {len([n for n in model_names if 'gandalf' in n])}")
print(f"Effective models in ensemble: {len(valid_indices)}")
print("="*80)

