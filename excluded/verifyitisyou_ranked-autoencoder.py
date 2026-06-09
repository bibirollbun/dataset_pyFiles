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


# Complete Working DRW Crypto Market Prediction Framework
# Fully debugged and tested implementation

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, QuantileTransformer, RobustScaler
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.feature_selection import mutual_info_regression, f_regression
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from scipy.stats import pearsonr, spearmanr
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
import random
import os
import pickle
from pathlib import Path
import gc
from tqdm import tqdm

warnings.filterwarnings('ignore')

# Set all random seeds for reproducibility
def set_random_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Robust data handling
def handle_infinite_values(data, method='clip'):
    """Handle infinite and extreme values in data"""
    data = np.array(data, dtype=np.float32)
    
    # Handle infinities
    if method == 'clip':
        if len(data.shape) == 2:
            for i in range(data.shape[1]):
                col = data[:, i]
                finite_mask = np.isfinite(col)
                if np.sum(finite_mask) > 0:
                    finite_vals = col[finite_mask]
                    p1, p99 = np.percentile(finite_vals, [1, 99])
                    col[np.isposinf(col)] = p99
                    col[np.isneginf(col)] = p1
                    data[:, i] = np.clip(col, p1, p99)
                else:
                    data[:, i] = 0.0
        else:
            finite_mask = np.isfinite(data)
            if np.sum(finite_mask) > 0:
                finite_vals = data[finite_mask]
                p1, p99 = np.percentile(finite_vals, [1, 99])
                data[np.isposinf(data)] = p99
                data[np.isneginf(data)] = p1
                data = np.clip(data, p1, p99)
            else:
                data[:] = 0.0
    
    # Handle NaN values
    if len(data.shape) == 2:
        for i in range(data.shape[1]):
            col_nan_mask = np.isnan(data[:, i])
            if np.any(col_nan_mask):
                finite_vals = data[np.isfinite(data[:, i]), i]
                if len(finite_vals) > 0:
                    data[col_nan_mask, i] = np.median(finite_vals)
                else:
                    data[col_nan_mask, i] = 0.0
    else:
        nan_mask = np.isnan(data)
        if np.any(nan_mask):
            finite_vals = data[np.isfinite(data)]
            if len(finite_vals) > 0:
                data[nan_mask] = np.median(finite_vals)
            else:
                data[nan_mask] = 0.0
    
    return data

# Autoencoder for feature extraction
class Autoencoder(nn.Module):
    def __init__(self, input_dim, encoding_dim=32, dropout=0.2):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, encoding_dim),
            nn.BatchNorm1d(encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    def encode(self, x):
        return self.encoder(x)

# Feature engineering
def create_enhanced_features(df, feature_cols, autoencoder_model=None, scaler=None):
    """Create comprehensive enhanced features"""
    print(f"   Creating features from {len(feature_cols)} base features...")
    
    # Base features
    X_base = df[feature_cols].values.astype(np.float32)
    X_base = handle_infinite_values(X_base)
    
    enhanced_features = []
    feature_names = []
    
    # 1. Base features
    enhanced_features.append(X_base)
    feature_names.extend(feature_cols)
    
    # 2. Microstructure features
    if all(col in df.columns for col in ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']):
        eps = 1e-8
        
        # Order flow features
        order_imbalance = (df['bid_qty'] - df['ask_qty']) / (df['bid_qty'] + df['ask_qty'] + eps)
        buy_sell_imbalance = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + eps)
        
        # Pressure indicators
        bid_pressure = df['bid_qty'] / (df['volume'] + eps)
        ask_pressure = df['ask_qty'] / (df['volume'] + eps)
        buy_pressure = df['buy_qty'] / (df['volume'] + eps)
        sell_pressure = df['sell_qty'] / (df['volume'] + eps)
        
        # Volume features
        volume_log = np.log1p(df['volume'])
        volume_sqrt = np.sqrt(df['volume'])
        
        # Liquidity indicators
        spread_proxy = (df['ask_qty'] + df['bid_qty']) / (df['volume'] + eps)
        depth_imbalance = (df['bid_qty'] + df['ask_qty']) / (df['buy_qty'] + df['sell_qty'] + eps)
        
        microstructure = np.column_stack([
            order_imbalance, buy_sell_imbalance,
            bid_pressure, ask_pressure, buy_pressure, sell_pressure,
            volume_log, volume_sqrt,
            spread_proxy, depth_imbalance
        ])
        
        microstructure = handle_infinite_values(microstructure)
        enhanced_features.append(microstructure)
        
        feature_names.extend([
            'order_imbalance', 'buy_sell_imbalance',
            'bid_pressure', 'ask_pressure', 'buy_pressure', 'sell_pressure',
            'volume_log', 'volume_sqrt',
            'spread_proxy', 'depth_imbalance'
        ])
    
    # 3. Statistical transformations (for top features only)
    top_n = min(10, X_base.shape[1])
    
    stat_features = []
    for i in range(top_n):
        feature = X_base[:, i]
        
        # Log transform
        log_feat = np.sign(feature) * np.log1p(np.abs(feature))
        stat_features.append(log_feat)
        feature_names.append(f'{feature_cols[i]}_log')
        
        # Power transform
        power_feat = np.sign(feature) * np.power(np.abs(feature) + eps, 0.5)
        stat_features.append(power_feat)
        feature_names.append(f'{feature_cols[i]}_sqrt')
    
    if stat_features:
        stat_matrix = np.column_stack(stat_features)
        stat_matrix = handle_infinite_values(stat_matrix)
        enhanced_features.append(stat_matrix)
    
    # 4. Interaction features (limited to top features)
    interaction_features = []
    
    top_indices = list(range(min(5, X_base.shape[1])))
    for i in range(len(top_indices)):
        for j in range(i+1, len(top_indices)):
            # Multiplication
            inter = X_base[:, top_indices[i]] * X_base[:, top_indices[j]]
            interaction_features.append(inter)
            feature_names.append(f'{feature_cols[top_indices[i]]}_x_{feature_cols[top_indices[j]]}')
    
    if interaction_features:
        inter_matrix = np.column_stack(interaction_features)
        inter_matrix = handle_infinite_values(inter_matrix)
        enhanced_features.append(inter_matrix)
    
    # Combine all features
    X_enhanced = np.hstack(enhanced_features)
    X_enhanced = handle_infinite_values(X_enhanced)
    
    # 5. Autoencoder features (if model provided)
    if autoencoder_model is not None and scaler is not None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        autoencoder_model.eval()
        
        # Scale features
        X_scaled = scaler.transform(X_enhanced)
        X_scaled = handle_infinite_values(X_scaled)
        
        # Get encoded features
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_scaled).to(device)
            # Process in batches to avoid memory issues
            batch_size = 4096
            encoded_features = []
            
            for i in range(0, len(X_tensor), batch_size):
                batch = X_tensor[i:i+batch_size]
                encoded = autoencoder_model.encode(batch)
                encoded_features.append(encoded.cpu().numpy())
            
            encoded_features = np.vstack(encoded_features)
        
        # Add encoded features
        X_enhanced = np.hstack([X_enhanced, encoded_features])
        feature_names.extend([f'ae_feature_{i}' for i in range(encoded_features.shape[1])])
    
    return X_enhanced, feature_names

# Feature selection
def advanced_feature_selection(X, y, max_features=100):
    """Select top features using multiple methods"""
    print(f"ğŸ”� Selecting top {max_features} features from {X.shape[1]}...")
    
    X = handle_infinite_values(X)
    y = handle_infinite_values(y.reshape(-1, 1)).flatten()
    
    # Calculate feature importance scores
    scores = []
    
    # 1. Correlation
    corr_scores = []
    for i in range(X.shape[1]):
        try:
            corr = abs(pearsonr(X[:, i], y)[0])
            corr_scores.append(corr if not np.isnan(corr) else 0)
        except:
            corr_scores.append(0)
    
    # 2. Mutual information
    try:
        mi_scores = mutual_info_regression(X, y, random_state=42)
        mi_scores = np.nan_to_num(mi_scores, 0)
    except:
        mi_scores = np.zeros(X.shape[1])
    
    # 3. Random Forest importance
    try:
        from sklearn.ensemble import RandomForestRegressor
        rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        rf_scores = rf.feature_importances_
    except:
        rf_scores = np.zeros(X.shape[1])
    
    # Normalize and combine scores
    def normalize_scores(s):
        if s.max() == s.min():
            return np.ones_like(s) / len(s)
        return (s - s.min()) / (s.max() - s.min() + 1e-8)
    
    corr_norm = normalize_scores(np.array(corr_scores))
    mi_norm = normalize_scores(mi_scores)
    rf_norm = normalize_scores(rf_scores)
    
    # Weighted ensemble of scores
    ensemble_scores = 0.4 * corr_norm + 0.3 * mi_norm + 0.3 * rf_norm
    
    # Select top features
    n_features = min(max_features, X.shape[1])
    top_features = np.argsort(ensemble_scores)[-n_features:]
    
    print(f"   âœ… Selected {len(top_features)} features")
    return top_features, ensemble_scores

# Data transformations
def create_rank_transform(X):
    """Create rank transformation"""
    X_rank = np.zeros_like(X)
    for i in range(X.shape[1]):
        X_rank[:, i] = pd.Series(X[:, i]).rank(pct=True, method='average').fillna(0.5).values
    return handle_infinite_values(X_rank)

# Neural Network Components
class SimpleNN(nn.Module):
    """Simple neural network for regression"""
    def __init__(self, input_dim, hidden_dim=128, dropout=0.5):
        super(SimpleNN, self).__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.BatchNorm1d(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.BatchNorm1d(hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            
            nn.Linear(hidden_dim // 4, 1)
        )
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)

class CryptoDataset(Dataset):
    """PyTorch dataset for crypto data"""
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]

def train_neural_network(X_train, y_train, X_val, y_val, input_dim, epochs=20):
    """Train neural network model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model
    model = SimpleNN(input_dim, hidden_dim=128, dropout=0.5).to(device)
    
    # Create datasets
    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=2048, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4096, shuffle=False)
    
    # Training setup
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    best_val_loss = float('inf')
    best_val_corr = -float('inf')
    patience = 0
    
    # Training loop
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_preds = []
        val_targets = []
        val_loss = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                outputs = model(X_batch).squeeze()
                loss = criterion(outputs, y_batch)
                val_loss += loss.item()
                
                val_preds.extend(outputs.cpu().numpy())
                val_targets.extend(y_batch.cpu().numpy())
        
        # Calculate correlation
        val_corr = pearsonr(val_preds, val_targets)[0]
        if np.isnan(val_corr):
            val_corr = 0
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            patience = 0
        else:
            patience += 1
            if patience >= 10:
                break
    
    return model, best_val_corr

# Gradient Boosting Models
def train_xgboost(X_train, y_train, X_val, y_val):
    """Train XGBoost model"""
    params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'random_state': 42,
        'n_estimators': 1000,
        'early_stopping_rounds': 50
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model

def train_lightgbm(X_train, y_train, X_val, y_val):
    """Train LightGBM model"""
    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'max_depth': 5,
        'learning_rate': 0.02,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1,
        'reg_lambda': 1,
        'random_state': 42,
        'n_estimators': 1000,
        'verbose': -1,
        'force_col_wise': True
    }
    
    model = lgb.LGBMRegressor(**params)
    
    # Use callbacks for early stopping
    callbacks = [
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(0)
    ]
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=callbacks
    )
    
    return model

def train_catboost(X_train, y_train, X_val, y_val):
    """Train CatBoost model"""
    params = {
        'iterations': 1000,
        'depth': 5,
        'learning_rate': 0.02,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False,
        'early_stopping_rounds': 50
    }
    
    model = cb.CatBoostRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    return model

# Main pipeline
def main_pipeline():
    """Main prediction pipeline"""
    print("ğŸš€ Complete DRW Crypto Prediction Pipeline")
    print("=" * 80)
    
    set_random_seeds(42)
    
    # Create output directory
    output_dir = Path("/kaggle/working/drw_predictions")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("\nğŸ“Š Loading data...")
    train_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
    test_df = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
    
    print(f"   Train shape: {train_df.shape}")
    print(f"   Test shape: {test_df.shape}")
    
    # Use recent data (last 70% of training data)
    train_size = int(0.7 * len(train_df))
    train_df = train_df.iloc[-train_size:].reset_index(drop=True)
    print(f"   Using recent {len(train_df)} samples for training")
    
    # Top features from analysis
    top_features = [
        "X612", "X860", "X168", "X174", "X333", "X345", "X385", "X598", 
        "X421", "X852", "X863", "X856", "X344", "X862", "X603", "X674", 
        "X415", "X137", "X855", "X302", "bid_qty", "ask_qty", "buy_qty", 
        "sell_qty", "volume"
    ]
    
    # Filter available features
    feature_cols = [col for col in top_features if col in train_df.columns]
    print(f"   Using {len(feature_cols)} features")
    
    # Extract target
    y_train = train_df['label'].values
    y_train = handle_infinite_values(y_train.reshape(-1, 1)).flatten()
    
    # STEP 1: Create base features (without autoencoder)
    print("\nğŸ”§ Feature Engineering - Step 1: Base features...")
    X_train_base, train_feature_names = create_enhanced_features(
        train_df, feature_cols, autoencoder_model=None, scaler=None
    )
    X_test_base, test_feature_names = create_enhanced_features(
        test_df, [col for col in feature_cols if col != 'label'],
        autoencoder_model=None, scaler=None
    )
    
    # STEP 2: Train autoencoder on base features
    print("\nğŸ¤– Training autoencoder...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Scale features for autoencoder
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_base)
    X_train_scaled = handle_infinite_values(X_train_scaled)
    
    # Train autoencoder
    encoding_dim = 20
    autoencoder = Autoencoder(X_train_base.shape[1], encoding_dim).to(device)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # Create dataset for autoencoder training
    subset_size = min(50000, len(X_train_scaled))
    subset_idx = np.random.choice(len(X_train_scaled), subset_size, replace=False)
    
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train_scaled[subset_idx]))
    loader = DataLoader(dataset, batch_size=2048, shuffle=True)
    
    # Train autoencoder
    autoencoder.train()
    for epoch in range(30):
        total_loss = 0
        for batch in loader:
            data = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = autoencoder(data)
            loss = criterion(reconstructed, data)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"   Epoch {epoch}: Loss = {total_loss/len(loader):.4f}")
    
    # STEP 3: Create features with autoencoder
    print("\nğŸ”§ Feature Engineering - Step 2: Adding autoencoder features...")
    X_train_enhanced, train_feature_names = create_enhanced_features(
        train_df, feature_cols, autoencoder_model=autoencoder, scaler=scaler
    )
    X_test_enhanced, test_feature_names = create_enhanced_features(
        test_df, [col for col in feature_cols if col != 'label'],
        autoencoder_model=autoencoder, scaler=scaler
    )
    
    print(f"   Train features: {X_train_enhanced.shape}")
    print(f"   Test features: {X_test_enhanced.shape}")
    
    # Feature selection
    top_indices, feature_scores = advanced_feature_selection(
        X_train_enhanced, y_train, max_features=100
    )
    
    X_train_selected = X_train_enhanced[:, top_indices]
    X_test_selected = X_test_enhanced[:, top_indices]
    
    # Apply rank transformation
    print("\nğŸ”„ Applying rank transformation...")
    X_train_rank = create_rank_transform(X_train_selected)
    X_test_rank = create_rank_transform(X_test_selected)
    
    # Time series cross-validation
    print("\nğŸ�¯ Training models with time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=3)
    
    all_test_predictions = []
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_rank)):
        print(f"\nğŸ“Š Fold {fold + 1}/3")
        
        X_fold_train = X_train_rank[train_idx]
        X_fold_val = X_train_rank[val_idx]
        y_fold_train = y_train[train_idx]
        y_fold_val = y_train[val_idx]
        
        fold_predictions = []
        fold_scores = {}
        
        # Train Neural Network
        print("   ğŸ§  Training Neural Network...")
        try:
            nn_model, nn_score = train_neural_network(
                X_fold_train, y_fold_train, 
                X_fold_val, y_fold_val,
                input_dim=X_fold_train.shape[1],
                epochs=20
            )
            
            # Make predictions
            nn_model.eval()
            test_dataset = CryptoDataset(X_test_rank)
            test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False)
            
            nn_preds = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    preds = nn_model(batch).squeeze().cpu().numpy()
                    nn_preds.extend(preds)
            
            fold_predictions.append(np.array(nn_preds))
            fold_scores['neural'] = nn_score
            print(f"      Validation correlation: {nn_score:.4f}")
        except Exception as e:
            print(f"      Failed: {e}")
            fold_scores['neural'] = 0
        
        # Train XGBoost
        print("   ğŸŒ³ Training XGBoost...")
        try:
            xgb_model = train_xgboost(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
            xgb_preds = xgb_model.predict(X_test_rank)
            xgb_val_preds = xgb_model.predict(X_fold_val)
            xgb_score = pearsonr(xgb_val_preds, y_fold_val)[0]
            
            fold_predictions.append(xgb_preds)
            fold_scores['xgboost'] = xgb_score
            print(f"      Validation correlation: {xgb_score:.4f}")
        except Exception as e:
            print(f"      Failed: {e}")
            fold_scores['xgboost'] = 0
        
        # Train LightGBM
        print("   ğŸŒ¿ Training LightGBM...")
        try:
            lgb_model = train_lightgbm(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
            lgb_preds = lgb_model.predict(X_test_rank)
            lgb_val_preds = lgb_model.predict(X_fold_val)
            lgb_score = pearsonr(lgb_val_preds, y_fold_val)[0]
            
            fold_predictions.append(lgb_preds)
            fold_scores['lightgbm'] = lgb_score
            print(f"      Validation correlation: {lgb_score:.4f}")
        except Exception as e:
            print(f"      Failed: {e}")
            fold_scores['lightgbm'] = 0
        
        # Train CatBoost
        print("   ğŸ�± Training CatBoost...")
        try:
            cb_model = train_catboost(X_fold_train, y_fold_train, X_fold_val, y_fold_val)
            cb_preds = cb_model.predict(X_test_rank)
            cb_val_preds = cb_model.predict(X_fold_val)
            cb_score = pearsonr(cb_val_preds, y_fold_val)[0]
            
            fold_predictions.append(cb_preds)
            fold_scores['catboost'] = cb_score
            print(f"      Validation correlation: {cb_score:.4f}")
        except Exception as e:
            print(f"      Failed: {e}")
            fold_scores['catboost'] = 0
        
        # Ensemble fold predictions
        if fold_predictions:
            # Weight by squared correlation (emphasize better models)
            weights = []
            for model_name in ['neural', 'xgboost', 'lightgbm', 'catboost']:
                score = fold_scores.get(model_name, 0)
                if score > 0:
                    weights.append(score ** 2)
                else:
                    weights.append(0)
            
            weights = np.array(weights[:len(fold_predictions)])
            
            if weights.sum() > 0:
                weights = weights / weights.sum()
                fold_ensemble = np.average(fold_predictions, axis=0, weights=weights)
            else:
                fold_ensemble = np.mean(fold_predictions, axis=0)
            
            all_test_predictions.append(fold_ensemble)
            valid_scores = [s for s in fold_scores.values() if s > 0]
            cv_scores.append(np.mean(valid_scores) if valid_scores else 0)
            
            print(f"   ğŸ“Š Fold average correlation: {cv_scores[-1]:.4f}")
    
    # Final ensemble across folds
    print("\nğŸ”® Creating final predictions...")
    if all_test_predictions:
        # Weight folds by their average score
        if cv_scores and np.array(cv_scores).sum() > 0:
            fold_weights = np.array(cv_scores)
            fold_weights = fold_weights / fold_weights.sum()
            final_predictions = np.average(all_test_predictions, axis=0, weights=fold_weights)
        else:
            final_predictions = np.mean(all_test_predictions, axis=0)
    else:
        print("âš ï¸� No successful predictions, using baseline")
        final_predictions = np.full(len(test_df), y_train.mean())
    
    # Post-processing: clip extreme values
    p5, p95 = np.percentile(y_train, [5, 95])
    final_predictions = np.clip(final_predictions, p5, p95)
    
    # Create submission
    print("\nğŸ’¾ Creating submission...")
    submission = pd.read_csv("/kaggle/input/drw-crypto-market-prediction/sample_submission.csv")
    submission['label'] = final_predictions
    submission.to_csv(output_dir / 'submission_final.csv', index=False)
    
    print("\nâœ… Pipeline completed successfully!")
    if cv_scores:
        print(f"   Average CV score: {np.mean(cv_scores):.4f}")
    print(f"   Output saved to: {output_dir / 'submission_final.csv'}")
    
    # Clean up
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return submission

# Run the pipeline
if __name__ == "__main__":
    submission = main_pipeline()

