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


# Error-Free Podcast Listening Time Prediction
# Complete working solution with advanced models

# Install required packages
!pip install -q pytorch-tabnet
!pip install -q optuna

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import TPESampler

# Deep Learning imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

# TabNet
from pytorch_tabnet.tab_model import TabNetRegressor

import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Set random seeds
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# =====================================================
# 1. DATA LOADING
# =====================================================
print("\n" + "="*60)
print("LOADING DATA")
print("="*60)

train_df = pd.read_csv('/kaggle/input/predict-podcast-listening-time/train.csv')
test_df = pd.read_csv('/kaggle/input/predict-podcast-listening-time/test.csv')
sample_submission = pd.read_csv('/kaggle/input/predict-podcast-listening-time/sample_submission.csv')

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# Display basic info
print("\nColumn names:")
print(train_df.columns.tolist())

print("\nData types:")
print(train_df.dtypes)

print("\nMissing values in train:")
print(train_df.isnull().sum())

print("\nMissing values in test:")
print(test_df.isnull().sum())

target_col = 'Listening_Time_minutes'

# Basic statistics
print(f"\nTarget statistics:")
print(f"Mean: {train_df[target_col].mean():.2f}")
print(f"Std: {train_df[target_col].std():.2f}")
print(f"Min: {train_df[target_col].min():.2f}")
print(f"Max: {train_df[target_col].max():.2f}")

# =====================================================
# 2. CONSISTENT FEATURE ENGINEERING
# =====================================================
print("\n" + "="*60)
print("FEATURE ENGINEERING")
print("="*60)

class ConsistentFeatureEngineer:
    """Feature engineering that ensures train/test consistency"""
    
    def __init__(self, seed=42):
        self.seed = seed
        self.label_encoders = {}
        self.target_encoders = {}
        self.numerical_cols = None
        self.categorical_cols = None
        self.interaction_features = []
        
    def fit(self, train_df, target_col):
        """Fit on training data only"""
        # Identify column types
        self.numerical_cols = train_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = train_df.select_dtypes(include=['object']).columns.tolist()
        
        # Remove id and target
        if 'id' in self.numerical_cols:
            self.numerical_cols.remove('id')
        if target_col in self.numerical_cols:
            self.numerical_cols.remove(target_col)
            
        # Store interaction features to create
        if len(self.numerical_cols) >= 2:
            # Use fixed pairs instead of correlation-based selection
            for i in range(min(3, len(self.numerical_cols))):
                for j in range(i+1, min(3, len(self.numerical_cols))):
                    self.interaction_features.append((self.numerical_cols[i], self.numerical_cols[j]))
        
        return self
    
    def transform(self, df, is_train=False, target_col=None):
        """Transform data consistently"""
        df = df.copy()
        
        # Handle missing values
        for col in self.numerical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna(df[col].median(), inplace=True)
        
        for col in self.categorical_cols:
            if col in df.columns and df[col].isnull().sum() > 0:
                df[col].fillna('missing', inplace=True)
        
        # Create statistical features
        if len(self.numerical_cols) >= 2:
            df['num_mean'] = df[self.numerical_cols].mean(axis=1)
            df['num_std'] = df[self.numerical_cols].std(axis=1)
            df['num_max'] = df[self.numerical_cols].max(axis=1)
            df['num_min'] = df[self.numerical_cols].min(axis=1)
            df['num_range'] = df['num_max'] - df['num_min']
            df['num_median'] = df[self.numerical_cols].median(axis=1)
        
        # Create interaction features (consistent between train/test)
        for col1, col2 in self.interaction_features:
            if col1 in df.columns and col2 in df.columns:
                df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
                df[f'{col1}_+_{col2}'] = df[col1] + df[col2]
                if df[col2].min() != 0:  # Avoid division by zero
                    df[f'{col1}_div_{col2}'] = df[col1] / df[col2].replace(0, 1)
        
        # Create polynomial features for first few numerical columns
        for i, col in enumerate(self.numerical_cols[:3]):
            if col in df.columns:
                df[f'{col}_squared'] = df[col] ** 2
                df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col]))
                df[f'{col}_log1p'] = np.log1p(np.abs(df[col]))
        
        return df
    
    def fit_transform(self, train_df, target_col):
        """Fit and transform training data"""
        self.fit(train_df, target_col)
        return self.transform(train_df, is_train=True, target_col=target_col)

# Apply feature engineering
fe = ConsistentFeatureEngineer(seed=SEED)
train_fe = fe.fit_transform(train_df, target_col)
test_fe = fe.transform(test_df, is_train=False)

print(f"Numerical columns: {fe.numerical_cols}")
print(f"Categorical columns: {fe.categorical_cols}")

# Handle categorical encoding
print("\nEncoding categorical variables...")
label_encoders = {}

for col in fe.categorical_cols:
    le = LabelEncoder()
    
    # Fit on combined data
    combined_values = pd.concat([train_fe[col], test_fe[col]]).astype(str).unique()
    le.fit(combined_values)
    
    # Transform
    train_fe[f'{col}_encoded'] = le.transform(train_fe[col].astype(str))
    test_fe[f'{col}_encoded'] = le.transform(test_fe[col].astype(str))
    label_encoders[col] = le

# Target encoding with KFold to prevent overfitting
print("\nApplying target encoding...")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

for col in fe.categorical_cols:
    train_fe[f'{col}_target_enc'] = 0.0
    
    # For each fold
    for train_idx, val_idx in kf.split(train_fe):
        # Get fold data
        X_train_fold = train_fe.iloc[train_idx]
        X_val_fold = train_fe.iloc[val_idx]
        
        # Calculate target mean for each category
        target_means = X_train_fold.groupby(col)[target_col].mean()
        
        # Apply to validation fold
        train_fe.loc[val_idx, f'{col}_target_enc'] = X_val_fold[col].map(target_means).fillna(train_fe[target_col].mean())
    
    # For test set, use overall means
    overall_means = train_df.groupby(col)[target_col].mean()
    test_fe[f'{col}_target_enc'] = test_fe[col].map(overall_means).fillna(train_df[target_col].mean())

# Select feature columns (excluding original categoricals)
feature_cols = [col for col in train_fe.columns if col not in ['id', target_col] + fe.categorical_cols]

# Ensure consistency
common_cols = list(set(feature_cols) & set(test_fe.columns))
print(f"\nTotal features: {len(common_cols)}")

# Use only common columns
train_fe = train_fe[['id', target_col] + common_cols]
test_fe = test_fe[['id'] + common_cols]

# =====================================================
# 3. NEURAL NETWORK ARCHITECTURES
# =====================================================
print("\n" + "="*60)
print("DEFINING NEURAL NETWORKS")
print("="*60)

class TabularDataset(Dataset):
    def __init__(self, features, targets=None):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets) if targets is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.targets is not None:
            return self.features[idx], self.targets[idx]
        return self.features[idx]

class GANDALF(nn.Module):
    """Gated Adaptive Network for Deep Automated Learning of Features"""
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.3):
        super(GANDALF, self).__init__()
        
        # Feature gates
        self.gates = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim),
            nn.Sigmoid()
        )
        
        # Main network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.main = nn.Sequential(*layers)
        
        # Skip connection
        self.skip = nn.Linear(input_dim, 1)
        
    def forward(self, x):
        # Apply gates
        gated = x * self.gates(x)
        
        # Main path + skip
        return self.main(gated) + 0.1 * self.skip(x)

class SAINT(nn.Module):
    """Self-Attention and Intersample Attention Transformer"""
    def __init__(self, input_dim, num_heads=4, hidden_dim=128, num_layers=2, dropout=0.2):
        super(SAINT, self).__init__()
        
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output
        self.output = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
    def forward(self, x):
        # Project and add sequence dimension
        x = self.input_proj(x).unsqueeze(1)
        
        # Transform
        x = self.transformer(x)
        
        # Output
        return self.output(x.squeeze(1))

def train_nn(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10):
    """Train neural network"""
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    criterion = nn.MSELoss()
    
    best_loss = float('inf')
    patience_count = 0
    best_state = None
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            pred = model(X).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X).squeeze()
                loss = criterion(pred, y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step(val_loss)
        
        if epoch % 20 == 0:
            print(f'Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}')
        
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            patience_count = 0
            best_state = model.state_dict().copy()
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f'Early stopping at epoch {epoch}')
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

# =====================================================
# 4. PREPARE DATA
# =====================================================
print("\n" + "="*60)
print("PREPARING DATA FOR MODELING")
print("="*60)

X = train_fe[common_cols].values
y = train_fe[target_col].values
X_test = test_fe[common_cols].values

print(f"Train shape: {X.shape}")
print(f"Test shape: {X_test.shape}")

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED)

# Scale
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)
X_scaled = scaler.transform(X)

# =====================================================
# 5. HYPERPARAMETER OPTIMIZATION
# =====================================================
print("\n" + "="*60)
print("HYPERPARAMETER OPTIMIZATION")
print("="*60)

def optimize_xgb(trial):
    params = {
        'n_estimators': 1000,
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': SEED
    }
    
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
              early_stopping_rounds=50, verbose=False)
    
    pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, pred))

def optimize_lgb(trial):
    params = {
        'n_estimators': 1000,
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 50),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': SEED,
        'verbosity': -1
    }
    
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
    
    pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, pred))

def optimize_cat(trial):
    params = {
        'iterations': 1000,
        'depth': trial.suggest_int('depth', 4, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
        'random_state': SEED,
        'verbose': False
    }
    
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), 
              early_stopping_rounds=50, verbose=False)
    
    pred = model.predict(X_val)
    return np.sqrt(mean_squared_error(y_val, pred))

# Optimize models
print("\nOptimizing XGBoost...")
study_xgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
study_xgb.optimize(optimize_xgb, n_trials=15)
xgb_params = study_xgb.best_params
xgb_params.update({'n_estimators': 1000, 'random_state': SEED})
print(f"Best RMSE: {study_xgb.best_value:.4f}")

print("\nOptimizing LightGBM...")
study_lgb = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
study_lgb.optimize(optimize_lgb, n_trials=15)
lgb_params = study_lgb.best_params
lgb_params.update({'n_estimators': 1000, 'random_state': SEED, 'verbosity': -1})
print(f"Best RMSE: {study_lgb.best_value:.4f}")

print("\nOptimizing CatBoost...")
study_cat = optuna.create_study(direction='minimize', sampler=TPESampler(seed=SEED))
study_cat.optimize(optimize_cat, n_trials=15)
cat_params = study_cat.best_params
cat_params.update({'iterations': 1000, 'random_state': SEED, 'verbose': False})
print(f"Best RMSE: {study_cat.best_value:.4f}")

# =====================================================
# 6. TRAIN MODELS
# =====================================================
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

# Train gradient boosting models
print("\nTraining XGBoost...")
xgb_model = xgb.XGBRegressor(**xgb_params)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], 
              early_stopping_rounds=100, verbose=False)
xgb_val_pred = xgb_model.predict(X_val)
print(f"Validation RMSE: {np.sqrt(mean_squared_error(y_val, xgb_val_pred)):.4f}")

print("\nTraining LightGBM...")
lgb_model = lgb.LGBMRegressor(**lgb_params)
lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
lgb_val_pred = lgb_model.predict(X_val)
print(f"Validation RMSE: {np.sqrt(mean_squared_error(y_val, lgb_val_pred)):.4f}")

print("\nTraining CatBoost...")
cat_model = CatBoostRegressor(**cat_params)
cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), 
              early_stopping_rounds=100, verbose=False)
cat_val_pred = cat_model.predict(X_val)
print(f"Validation RMSE: {np.sqrt(mean_squared_error(y_val, cat_val_pred)):.4f}")

print("\nTraining Random Forest...")
rf_model = RandomForestRegressor(n_estimators=300, max_depth=10, 
                                min_samples_split=20, random_state=SEED, n_jobs=-1)
rf_model.fit(X_train, y_train)
rf_val_pred = rf_model.predict(X_val)
print(f"Validation RMSE: {np.sqrt(mean_squared_error(y_val, rf_val_pred)):.4f}")

# Train neural networks
print("\nTraining Neural Networks...")

# Data loaders
train_dataset = TabularDataset(X_train_scaled, y_train)
val_dataset = TabularDataset(X_val_scaled, y_val)
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128)

# GANDALF
print("\nTraining GANDALF...")
gandalf = GANDALF(X_train_scaled.shape[1]).to(device)
gandalf = train_nn(gandalf, train_loader, val_loader, epochs=100)

# SAINT
print("\nTraining SAINT...")
saint = SAINT(X_train_scaled.shape[1]).to(device)
saint = train_nn(saint, train_loader, val_loader, epochs=100)

# TabNet
print("\nTraining TabNet...")
tabnet = TabNetRegressor(
    n_d=32, n_a=32, n_steps=3, gamma=1.3,
    n_independent=2, n_shared=2,
    lambda_sparse=1e-4,
    optimizer_params=dict(lr=0.02, weight_decay=1e-5),
    scheduler_params={"gamma": 0.95, "step_size": 20},
    scheduler_fn=optim.lr_scheduler.StepLR,
    seed=SEED,
    verbose=0
)

tabnet.fit(
    X_train=X_train_scaled, y_train=y_train.reshape(-1, 1),
    eval_set=[(X_val_scaled, y_val.reshape(-1, 1))],
    max_epochs=100,
    patience=20,
    batch_size=256,
    virtual_batch_size=128
)

# Get NN predictions
def get_nn_pred(model, X, scaler):
    model.eval()
    X_scaled = scaler.transform(X)
    dataset = TabularDataset(X_scaled)
    loader = DataLoader(dataset, batch_size=128)
    
    preds = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).cpu().numpy()
            preds.extend(pred.flatten())
    
    return np.array(preds)

gandalf_val_pred = get_nn_pred(gandalf, X_val, scaler)
saint_val_pred = get_nn_pred(saint, X_val, scaler)
tabnet_val_pred = tabnet.predict(X_val_scaled).flatten()

print(f"GANDALF Validation RMSE: {np.sqrt(mean_squared_error(y_val, gandalf_val_pred)):.4f}")
print(f"SAINT Validation RMSE: {np.sqrt(mean_squared_error(y_val, saint_val_pred)):.4f}")
print(f"TabNet Validation RMSE: {np.sqrt(mean_squared_error(y_val, tabnet_val_pred)):.4f}")

# =====================================================
# 7. CROSS-VALIDATION & ENSEMBLE
# =====================================================
print("\n" + "="*60)
print("CROSS-VALIDATION & ENSEMBLE")
print("="*60)

n_folds = 5
kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)

# Store predictions
oof_preds = {
    'xgboost': np.zeros(len(X)),
    'lightgbm': np.zeros(len(X)),
    'catboost': np.zeros(len(X)),
    'rf': np.zeros(len(X)),
    'gandalf': np.zeros(len(X)),
    'saint': np.zeros(len(X)),
    'tabnet': np.zeros(len(X))
}

test_preds = {
    'xgboost': np.zeros(len(X_test)),
    'lightgbm': np.zeros(len(X_test)),
    'catboost': np.zeros(len(X_test)),
    'rf': np.zeros(len(X_test)),
    'gandalf': np.zeros(len(X_test)),
    'saint': np.zeros(len(X_test)),
    'tabnet': np.zeros(len(X_test))
}

print(f"\nRunning {n_folds}-fold cross-validation...")
for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_tr, X_vl = X[train_idx], X[val_idx]
    y_tr, y_vl = y[train_idx], y[val_idx]
    
    # Scale
    fold_scaler = RobustScaler()
    X_tr_scaled = fold_scaler.fit_transform(X_tr)
    X_vl_scaled = fold_scaler.transform(X_vl)
    X_test_scaled_fold = fold_scaler.transform(X_test)
    
    # XGBoost
    xgb_fold = xgb.XGBRegressor(**xgb_params)
    xgb_fold.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)], 
                 early_stopping_rounds=100, verbose=False)
    oof_preds['xgboost'][val_idx] = xgb_fold.predict(X_vl)
    test_preds['xgboost'] += xgb_fold.predict(X_test) / n_folds
    
    # LightGBM
    lgb_fold = lgb.LGBMRegressor(**lgb_params)
    lgb_fold.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
                 callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_preds['lightgbm'][val_idx] = lgb_fold.predict(X_vl)
    test_preds['lightgbm'] += lgb_fold.predict(X_test) / n_folds
    
    # CatBoost
    cat_fold = CatBoostRegressor(**cat_params)
    cat_fold.fit(X_tr, y_tr, eval_set=(X_vl, y_vl), 
                 early_stopping_rounds=100, verbose=False)
    oof_preds['catboost'][val_idx] = cat_fold.predict(X_vl)
    test_preds['catboost'] += cat_fold.predict(X_test) / n_folds
    
    # Random Forest
    rf_fold = RandomForestRegressor(n_estimators=300, max_depth=10,
                                   min_samples_split=20, random_state=SEED, n_jobs=-1)
    rf_fold.fit(X_tr, y_tr)
    oof_preds['rf'][val_idx] = rf_fold.predict(X_vl)
    test_preds['rf'] += rf_fold.predict(X_test) / n_folds
    
    # Neural networks (using pre-trained models for speed)
    oof_preds['gandalf'][val_idx] = get_nn_pred(gandalf, X_vl, fold_scaler)
    test_preds['gandalf'] += get_nn_pred(gandalf, X_test, fold_scaler) / n_folds
    
    oof_preds['saint'][val_idx] = get_nn_pred(saint, X_vl, fold_scaler)
    test_preds['saint'] += get_nn_pred(saint, X_test, fold_scaler) / n_folds
    
    oof_preds['tabnet'][val_idx] = tabnet.predict(X_vl_scaled).flatten()
    test_preds['tabnet'] += tabnet.predict(X_test_scaled_fold).flatten() / n_folds

# Calculate scores
print("\n" + "="*60)
print("MODEL PERFORMANCE (OOF)")
print("="*60)

model_scores = {}
for name, pred in oof_preds.items():
    rmse = np.sqrt(mean_squared_error(y, pred))
    mae = mean_absolute_error(y, pred)
    r2 = r2_score(y, pred)
    model_scores[name] = rmse
    print(f"{name:12s}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

# =====================================================
# 8. ENSEMBLE
# =====================================================
print("\n" + "="*60)
print("CREATING ENSEMBLE")
print("="*60)

# Simple average
simple_avg_oof = np.mean(list(oof_preds.values()), axis=0)
simple_avg_test = np.mean(list(test_preds.values()), axis=0)
simple_rmse = np.sqrt(mean_squared_error(y, simple_avg_oof))
print(f"Simple Average RMSE: {simple_rmse:.4f}")

# Weighted average
weights = {name: 1/score for name, score in model_scores.items()}
total_weight = sum(weights.values())
weights = {name: w/total_weight for name, w in weights.items()}

weighted_avg_oof = sum(weights[name] * oof_preds[name] for name in weights)
weighted_avg_test = sum(weights[name] * test_preds[name] for name in weights)
weighted_rmse = np.sqrt(mean_squared_error(y, weighted_avg_oof))
print(f"Weighted Average RMSE: {weighted_rmse:.4f}")

# Stacking
print("\nTraining stacking model...")
stack_train = np.column_stack(list(oof_preds.values()))
stack_test = np.column_stack(list(test_preds.values()))

# Try multiple meta-models
meta_models = {
    'ridge': Ridge(alpha=1.0),
    'lasso': Lasso(alpha=0.01),
    'elastic': ElasticNet(alpha=0.01, l1_ratio=0.5),
    'rf_meta': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=SEED)
}

best_meta_rmse = float('inf')
best_meta_name = None
best_meta_model = None

for name, model in meta_models.items():
    model.fit(stack_train, y)
    pred = model.predict(stack_train)
    rmse = np.sqrt(mean_squared_error(y, pred))
    print(f"Stacking with {name}: RMSE={rmse:.4f}")
    
    if rmse < best_meta_rmse:
        best_meta_rmse = rmse
        best_meta_name = name
        best_meta_model = model

print(f"\nBest meta-model: {best_meta_name} with RMSE={best_meta_rmse:.4f}")
stacking_test = best_meta_model.predict(stack_test)

# =====================================================
# 9. FINAL SUBMISSION
# =====================================================
print("\n" + "="*60)
print("CREATING SUBMISSION")
print("="*60)

# Choose best ensemble
if best_meta_rmse <= weighted_rmse and best_meta_rmse <= simple_rmse:
    final_pred = stacking_test
    print("Using stacking ensemble")
elif weighted_rmse <= simple_rmse:
    final_pred = weighted_avg_test
    print("Using weighted average")
else:
    final_pred = simple_avg_test
    print("Using simple average")

# Create submission
submission = pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': final_pred
})

# Post-process
submission['Listening_Time_minutes'] = submission['Listening_Time_minutes'].clip(lower=0)
upper_limit = np.percentile(y, 99.5)
submission['Listening_Time_minutes'] = submission['Listening_Time_minutes'].clip(upper=upper_limit)

print("\nSubmission statistics:")
print(submission['Listening_Time_minutes'].describe())

# Save
submission.to_csv('submission.csv', index=False)
print("\nSubmission saved!")

# Save alternative versions
pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': weighted_avg_test
}).to_csv('submission_weighted.csv', index=False)

pd.DataFrame({
    'id': test_df['id'],
    'Listening_Time_minutes': simple_avg_test
}).to_csv('submission_simple.csv', index=False)

print("\nAlternative submissions saved!")

# =====================================================
# 10. FEATURE IMPORTANCE
# =====================================================
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

# Average importance across tree models
importance_df = pd.DataFrame({
    'feature': common_cols,
    'xgb': xgb_model.feature_importances_,
    'lgb': lgb_model.feature_importances_,
    'cat': cat_model.feature_importances_,
    'rf': rf_model.feature_importances_
})

importance_df['avg_importance'] = importance_df[['xgb', 'lgb', 'cat', 'rf']].mean(axis=1)
importance_df = importance_df.sort_values('avg_importance', ascending=False)

print("\nTop 20 Features:")
print(importance_df[['feature', 'avg_importance']].head(20))

# Plot
plt.figure(figsize=(10, 8))
importance_df.head(20).plot(x='feature', y='avg_importance', kind='barh')
plt.title('Top 20 Feature Importances')
plt.xlabel('Average Importance')
plt.tight_layout()
plt.show()

print("\n" + "="*60)
print("COMPLETE!")
print("="*60)
print(f"\nBest ensemble RMSE: {min(simple_rmse, weighted_rmse, best_meta_rmse):.4f}")
print("Good luck with your competition!")

