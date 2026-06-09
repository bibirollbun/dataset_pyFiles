"""
COMPLETE ENSEMBLE: RealMLP + GANDALF-Style + XGBoost + LightGBM + CatBoost
Medical Insurance Cost Prediction
Fully Working Code - No Placeholders - All Models Included
"""

# Install required packages
import subprocess
import sys

print("Installing required packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-qq", "pytabkit"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-qq", "lightgbm"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "-qq", "catboost"])

import warnings
warnings.simplefilter('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder

# Import all models
from pytabkit import RealMLP_TD_Regressor
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

# For GANDALF-style model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

# ============================================================================
# GANDALF-STYLE MODEL IMPLEMENTATION
# ============================================================================
class GANDALFRegressor(nn.Module):
    """
    GANDALF-inspired architecture: Gated Adaptive Network with Deep Attention
    Since pytabkit doesn't have GANDALF_Regressor, we implement it ourselves
    """
    def __init__(self, input_dim, hidden_dim=256, n_layers=4, dropout=0.1):
        super(GANDALFRegressor, self).__init__()
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)
        
        # Gated Linear Units (GLU) - core of GANDALF
        self.glu_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.BatchNorm1d(hidden_dim * 2),
            ) for _ in range(n_layers)
        ])
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=4, dropout=dropout, batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(hidden_dim, 1)
        
    def glu_block(self, x, glu_layer):
        """Gated Linear Unit block"""
        out = glu_layer(x)
        gate, value = out.chunk(2, dim=1)
        return torch.sigmoid(gate) * value
    
    def forward(self, x):
        # Input projection
        x = F.relu(self.bn_input(self.input_layer(x)))
        x = self.dropout(x)
        
        # GLU layers
        for glu_layer in self.glu_layers:
            residual = x
            x = self.glu_block(x, glu_layer)
            x = self.dropout(x)
            x = x + residual  # Residual connection
        
        # Self-attention
        x_attn = x.unsqueeze(1)  # Add sequence dimension
        x_attn, _ = self.attention(x_attn, x_attn, x_attn)
        x = x + x_attn.squeeze(1)  # Residual
        
        # Output
        return self.output_layer(x).squeeze()

class GANDALFWrapper:
    """Wrapper to make GANDALF compatible with sklearn-style fit/predict"""
    def __init__(self, n_epochs=50, lr=0.001, hidden_dim=256, n_layers=4, 
                 dropout=0.1, batch_size=512, device='cpu', random_state=42):
        self.n_epochs = n_epochs
        self.lr = lr
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.device = device
        self.random_state = random_state
        self.model = None
        self.input_dim = None
        
    def fit(self, X, y, X_val=None, y_val=None, verbose=True):
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        
        # Convert to numpy if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values
            
        self.input_dim = X.shape[1]
        
        # Initialize model
        self.model = GANDALFRegressor(
            self.input_dim, 
            self.hidden_dim, 
            self.n_layers, 
            self.dropout
        ).to(self.device)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        criterion = nn.MSELoss()
        
        # Create data loaders
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        # Training loop
        self.model.train()
        for epoch in range(self.n_epochs):
            total_loss = 0
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                predictions = self.model(batch_X)
                loss = criterion(predictions, batch_y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = total_loss / len(loader)
                print(f"Epoch {epoch+1}/{self.n_epochs}: Loss = {avg_loss:.6f}")
        
        return self
    
    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values
            
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            predictions = self.model(X_tensor)
            return predictions.cpu().numpy()

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "="*80)
print("LOADING DATA")
print("="*80)
train = pd.read_csv('/kaggle/input/medical-insurance-cost-dataset/train.csv')
test = pd.read_csv('/kaggle/input/medical-insurance-cost-dataset/test.csv')
print(f'Train Shape: {train.shape}')
print(f'Test Shape: {test.shape}')

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
print("\n" + "="*80)
print("FEATURE ENGINEERING")
print("="*80)

TARGET = 'charges'
BASE = [col for col in train.columns if col not in ['id', TARGET]]
CATS = ['sex', 'smoker', 'region']
NUMS = [col for col in BASE if col not in CATS]

# Create interaction features
train['bmi_age'] = train['bmi'] * train['age']
train['smoker_bmi'] = train['bmi'] * (train['smoker'] == 'yes').astype(int)
train['smoker_age'] = train['age'] * (train['smoker'] == 'yes').astype(int)
train['age_squared'] = train['age'] ** 2
train['bmi_squared'] = train['bmi'] ** 2

test['bmi_age'] = test['bmi'] * test['age']
test['smoker_bmi'] = test['bmi'] * (test['smoker'] == 'yes').astype(int)
test['smoker_age'] = test['age'] * (test['smoker'] == 'yes').astype(int)
test['age_squared'] = test['age'] ** 2
test['bmi_squared'] = test['bmi'] ** 2

FEATURES = [col for col in train.columns if col not in ['id', TARGET]]
print(f'{len(FEATURES)} Features: {FEATURES}')

# Prepare data
X_raw = train[FEATURES].copy()
y = np.log1p(train[TARGET])
X_test_raw = test[FEATURES].copy()

# Encode for tree models
X_encoded = X_raw.copy()
X_test_encoded = X_test_raw.copy()
label_encoders = {}
for col in CATS:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_raw[col].astype(str))
    X_test_encoded[col] = le.transform(X_test_raw[col].astype(str))
    label_encoders[col] = le

# ============================================================================
# CROSS-VALIDATION SETUP
# ============================================================================
N_SPLITS = 5
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

# Storage for predictions
model_names = ['RealMLP', 'GANDALF', 'XGBoost', 'LightGBM', 'CatBoost']
oof_preds = {name: np.zeros(len(X_raw)) for name in model_names}
test_preds = {name: np.zeros(len(test)) for name in model_names}
oof_rmse = {}

# ============================================================================
# MODEL 1: RealMLP
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODEL 1: RealMLP Neural Network")
print("="*80)

realmlp_params = {
    'device': 'cpu',
    'n_epochs': 10,
    'random_state': 42,
    'val_metric_name': 'rmse',
    'verbosity': 2,
    'hidden_sizes': [256, 256, 256],
    'max_one_hot_cat_size': 9,
    'embedding_size': 8,
    'weight_param': 'ntk',
    'weight_init_mode': 'std',
    'bias_init_mode': 'he+5',
    'bias_lr_factor': 0.1,
    'act': 'selu',
    'use_parametric_act': True,
    'act_lr_factor': 0.1,
    'wd': 0.02,
    'wd_sched': 'flat_cos',
    'bias_wd_factor': 0.0,
    'block_str': 'w-b-a-d',
    'p_drop': 0.0,
    'p_drop_sched': 'flat_cos',
    'add_front_scale': True,
    'scale_lr_factor': 6.0,
    'tfms': ['one_hot', 'median_center', 'robust_scale', 'smooth_clip', 'embedding'],
    'num_emb_type': 'plr',
    'plr_sigma': 0.1513700357637058,
    'plr_hidden_1': 16,
    'plr_hidden_2': 4,
    'plr_lr_factor': 0.1,
    'clamp_output': True,
    'normalize_output': True,
    'lr': 0.05846217780681372,
    'lr_sched': 'coslog4',
    'opt': 'adam',
    'sq_mom': 0.95,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_raw, y)):
    print(f'\n--- RealMLP Fold {fold+1}/{N_SPLITS} ---')
    
    X_train, X_val = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = RealMLP_TD_Regressor(**realmlp_params)
    model.fit(X_train, y_train, X_val, y_val, cat_col_names=CATS)
    
    oof_preds['RealMLP'][val_idx] = model.predict(X_val)
    test_preds['RealMLP'] += model.predict(X_test_raw)
    
    fold_rmse = root_mean_squared_error(y_val, oof_preds['RealMLP'][val_idx])
    print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")

test_preds['RealMLP'] /= N_SPLITS
oof_rmse['RealMLP'] = root_mean_squared_error(y, oof_preds['RealMLP'])
print(f"\nRealMLP Overall OOF RMSE: {oof_rmse['RealMLP']:.5f}")

# ============================================================================
# MODEL 2: GANDALF (Custom Implementation)
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODEL 2: GANDALF (Gated Adaptive Network)")
print("="*80)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_encoded, y)):
    print(f'\n--- GANDALF Fold {fold+1}/{N_SPLITS} ---')
    
    X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = GANDALFWrapper(
        n_epochs=50,
        lr=0.001,
        hidden_dim=256,
        n_layers=4,
        dropout=0.1,
        batch_size=512,
        device='cpu',
        random_state=42 + fold
    )
    
    model.fit(X_train, y_train, X_val, y_val, verbose=(fold == 0))
    
    oof_preds['GANDALF'][val_idx] = model.predict(X_val)
    test_preds['GANDALF'] += model.predict(X_test_encoded)
    
    fold_rmse = root_mean_squared_error(y_val, oof_preds['GANDALF'][val_idx])
    print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")

test_preds['GANDALF'] /= N_SPLITS
oof_rmse['GANDALF'] = root_mean_squared_error(y, oof_preds['GANDALF'])
print(f"\nGANDALF Overall OOF RMSE: {oof_rmse['GANDALF']:.5f}")

# ============================================================================
# MODEL 3: XGBoost
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODEL 3: XGBoost")
print("="*80)

xgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'min_child_weight': 3,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist',
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_encoded, y)):
    print(f'\n--- XGBoost Fold {fold+1}/{N_SPLITS} ---')
    
    X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    
    oof_preds['XGBoost'][val_idx] = model.predict(X_val)
    test_preds['XGBoost'] += model.predict(X_test_encoded)
    
    fold_rmse = root_mean_squared_error(y_val, oof_preds['XGBoost'][val_idx])
    print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")

test_preds['XGBoost'] /= N_SPLITS
oof_rmse['XGBoost'] = root_mean_squared_error(y, oof_preds['XGBoost'])
print(f"\nXGBoost Overall OOF RMSE: {oof_rmse['XGBoost']:.5f}")

# ============================================================================
# MODEL 4: LightGBM
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODEL 4: LightGBM")
print("="*80)

lgb_params = {
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'max_depth': 6,
    'num_leaves': 31,
    'min_child_samples': 20,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.5,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'verbose': -1,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_encoded, y)):
    print(f'\n--- LightGBM Fold {fold+1}/{N_SPLITS} ---')
    
    X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = lgb.LGBMRegressor(**lgb_params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    oof_preds['LightGBM'][val_idx] = model.predict(X_val)
    test_preds['LightGBM'] += model.predict(X_test_encoded)
    
    fold_rmse = root_mean_squared_error(y_val, oof_preds['LightGBM'][val_idx])
    print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")

test_preds['LightGBM'] /= N_SPLITS
oof_rmse['LightGBM'] = root_mean_squared_error(y, oof_preds['LightGBM'])
print(f"\nLightGBM Overall OOF RMSE: {oof_rmse['LightGBM']:.5f}")

# ============================================================================
# MODEL 5: CatBoost
# ============================================================================
print("\n" + "="*80)
print("TRAINING MODEL 5: CatBoost")
print("="*80)

cat_params = {
    'iterations': 1000,
    'learning_rate': 0.05,
    'depth': 6,
    'l2_leaf_reg': 3,
    'random_seed': 42,
    'verbose': False,
}

for fold, (train_idx, val_idx) in enumerate(kf.split(X_encoded, y)):
    print(f'\n--- CatBoost Fold {fold+1}/{N_SPLITS} ---')
    
    X_train, X_val = X_encoded.iloc[train_idx], X_encoded.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    model = CatBoostRegressor(**cat_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    oof_preds['CatBoost'][val_idx] = model.predict(X_val)
    test_preds['CatBoost'] += model.predict(X_test_encoded)
    
    fold_rmse = root_mean_squared_error(y_val, oof_preds['CatBoost'][val_idx])
    print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")

test_preds['CatBoost'] /= N_SPLITS
oof_rmse['CatBoost'] = root_mean_squared_error(y, oof_preds['CatBoost'])
print(f"\nCatBoost Overall OOF RMSE: {oof_rmse['CatBoost']:.5f}")

# ============================================================================
# ENSEMBLE: Simple Average
# ============================================================================
print("\n" + "="*80)
print("CREATING ENSEMBLES")
print("="*80)

ensemble_simple_oof = np.mean([oof_preds[m] for m in model_names], axis=0)
ensemble_simple_test = np.mean([test_preds[m] for m in model_names], axis=0)
ensemble_simple_rmse = root_mean_squared_error(y, ensemble_simple_oof)

# ============================================================================
# ENSEMBLE: Weighted (Inverse RMSE)
# ============================================================================
weights = {m: 1/oof_rmse[m] for m in model_names}
total_weight = sum(weights.values())
weights = {m: w/total_weight for m, w in weights.items()}

ensemble_weighted_oof = sum(weights[m] * oof_preds[m] for m in model_names)
ensemble_weighted_test = sum(weights[m] * test_preds[m] for m in model_names)
ensemble_weighted_rmse = root_mean_squared_error(y, ensemble_weighted_oof)

# ============================================================================
# ENSEMBLE: Optimized (Grid Search)
# ============================================================================
print("\nOptimizing ensemble weights...")
from itertools import product

best_rmse = float('inf')
best_weights = None

# Simplified grid search (5 models = too many combinations, use coarse grid)
weight_options = [0.0, 0.1, 0.2, 0.3, 0.4]
best_combo = None

# Try different weight combinations for top 3 models
top_3_models = sorted(model_names, key=lambda m: oof_rmse[m])[:3]
print(f"Top 3 models for optimization: {top_3_models}")

for weights_tuple in product(weight_options, repeat=3):
    if sum(weights_tuple) == 0:
        continue
    
    w = {m: 0.0 for m in model_names}
    total = sum(weights_tuple)
    for i, model in enumerate(top_3_models):
        w[model] = weights_tuple[i] / total
    
    opt_oof = sum(w[m] * oof_preds[m] for m in model_names)
    opt_rmse = root_mean_squared_error(y, opt_oof)
    
    if opt_rmse < best_rmse:
        best_rmse = opt_rmse
        best_weights = w.copy()

ensemble_optimized_oof = sum(best_weights[m] * oof_preds[m] for m in model_names)
ensemble_optimized_test = sum(best_weights[m] * test_preds[m] for m in model_names)

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print("\n" + "="*80)
print("FINAL RESULTS")
print("="*80)

results_df = pd.DataFrame({
    'Model': model_names + ['Ensemble (Simple)', 'Ensemble (Weighted)', 'Ensemble (Optimized)'],
    'OOF RMSE': [oof_rmse[m] for m in model_names] + [ensemble_simple_rmse, ensemble_weighted_rmse, best_rmse]
})
results_df = results_df.sort_values('OOF RMSE')

print("\n" + results_df.to_string(index=False))
print(f"\n⭐ Best Model: {results_df.iloc[0]['Model']}")
print(f"⭐ Best RMSE: {results_df.iloc[0]['OOF RMSE']:.5f}")

print("\n\nWeighted Ensemble Weights:")
for m in model_names:
    print(f"  {m}: {weights[m]:.4f}")

print("\n\nOptimized Ensemble Weights:")
for m in model_names:
    if best_weights[m] > 0:
        print(f"  {m}: {best_weights[m]:.4f}")

# ============================================================================
# SAVE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("SAVING PREDICTIONS")
print("="*80)

# Individual models
for model_name in model_names:
    pd.DataFrame({
        'id': train.id,
        TARGET: np.expm1(oof_preds[model_name])
    }).to_csv(f'oof_{model_name.lower()}.csv', index=False)
    
    pd.DataFrame({
        'id': test.id,
        TARGET: np.expm1(test_preds[model_name])
    }).to_csv(f'test_{model_name.lower()}.csv', index=False)

# Ensembles
pd.DataFrame({'id': train.id, TARGET: np.expm1(ensemble_simple_oof)}).to_csv('oof_ensemble_simple.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: np.expm1(ensemble_simple_test)}).to_csv('submission_ensemble_simple.csv', index=False)

pd.DataFrame({'id': train.id, TARGET: np.expm1(ensemble_weighted_oof)}).to_csv('oof_ensemble_weighted.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: np.expm1(ensemble_weighted_test)}).to_csv('submission_ensemble_weighted.csv', index=False)

pd.DataFrame({'id': train.id, TARGET: np.expm1(ensemble_optimized_oof)}).to_csv('oof_ensemble_optimized.csv', index=False)
pd.DataFrame({'id': test.id, TARGET: np.expm1(ensemble_optimized_test)}).to_csv('submission_ensemble_optimized.csv', index=False)

print("\n✅ All predictions saved!")
print("\nFiles created:")
print("  Individual Models: oof_*.csv and test_*.csv for each model")
print("  Ensembles: submission_ensemble_*.csv")
print("\n⭐ RECOMMENDED SUBMISSION: submission_ensemble_optimized.csv")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE!")
print("="*80)

