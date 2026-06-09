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
# coding: utf-8

"""
Complete VIME Feature Importance Analysis for DRW Crypto Market Prediction
=========================================================================
Full implementation with corrected submission file generation
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import lightgbm as lgb
import xgboost as xgb
from tqdm import tqdm
import gc
import warnings
warnings.filterwarnings("ignore")

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Memory management
def clean_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# =========================
# 1. VIME Model Implementation
# =========================
class VIMEModel(nn.Module):
    """VIME: Value Imputation and Mask Estimation"""
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout_rate=0.2, corruption_rate=0.3):
        super().__init__()
        
        self.input_dim = input_dim
        self.corruption_rate = corruption_rate
        
        # Encoder network
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        self.encoder = nn.Sequential(*encoder_layers)
        self.encoder_output_dim = hidden_dims[-1]
        
        # Mask estimation network
        self.mask_estimator = nn.Sequential(
            nn.Linear(self.encoder_output_dim, self.encoder_output_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.encoder_output_dim, input_dim),
            nn.Sigmoid()
        )
        
        # Feature reconstruction network
        self.feature_reconstructor = nn.Sequential(
            nn.Linear(self.encoder_output_dim, self.encoder_output_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.encoder_output_dim * 2, input_dim)
        )
        
        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(self.encoder_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, 1)
        )
        
        # Feature importance weights
        self.feature_importance_weights = nn.Parameter(torch.ones(input_dim))
        
    def corrupt_data(self, x):
        """Corrupt data for self-supervised pretraining"""
        batch_size = x.size(0)
        
        # Random corruption mask
        mask = torch.bernoulli(torch.full((batch_size, self.input_dim), self.corruption_rate)).to(x.device)
        
        # Corrupt data by mixing with random samples
        corrupted_x = x.clone()
        for i in range(self.input_dim):
            corrupted_indices = mask[:, i] == 1
            if corrupted_indices.any():
                # Swap with random samples from the batch
                num_corrupted = corrupted_indices.sum()
                random_indices = torch.randperm(batch_size)[:num_corrupted]
                corrupted_x[corrupted_indices, i] = x[random_indices, i]
        
        return corrupted_x, mask
    
    def forward(self, x, pretrain=False):
        if pretrain:
            # Self-supervised pretraining
            corrupted_x, mask = self.corrupt_data(x)
            encoded = self.encoder(corrupted_x)
            mask_pred = self.mask_estimator(encoded)
            feature_pred = self.feature_reconstructor(encoded)
            return mask_pred, feature_pred, mask, x
        else:
            # Supervised prediction with feature importance
            weighted_x = x * torch.sigmoid(self.feature_importance_weights)
            encoded = self.encoder(weighted_x)
            output = self.predictor(encoded)
            return output

# =========================
# 2. Data Loading and Preprocessing
# =========================
print("=== VIME Feature Importance Analysis ===\n")
print("1. Loading data...")

# Define features
X_FEATURES = [f"X{i}" for i in range(1, 891)]
MARKET_FEATURES = ["bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume"]

# Load data - use subset for memory efficiency
train_df = pd.read_parquet(
    "/kaggle/input/drw-crypto-market-prediction/train.parquet",
    columns=X_FEATURES[:300] + MARKET_FEATURES + ["label"]  # Use first 300 X features
)

test_df = pd.read_parquet(
    "/kaggle/input/drw-crypto-market-prediction/test.parquet",
    columns=X_FEATURES[:300] + MARKET_FEATURES
)

print(f"Train shape: {train_df.shape}")
print(f"Test shape: {test_df.shape}")

# =========================
# 3. Feature Engineering
# =========================
print("\n2. Engineering features...")

def add_engineered_features(df):
    """Add engineered features"""
    # Order flow features
    df['net_order_flow'] = df['buy_qty'] - df['sell_qty']
    df['order_flow_imbalance'] = (df['buy_qty'] - df['sell_qty']) / (df['buy_qty'] + df['sell_qty'] + 1e-10)
    df['normalized_net_flow'] = df['net_order_flow'] / (df['volume'] + 1e-10)
    
    # Liquidity features
    df['total_depth'] = df['bid_qty'] + df['ask_qty']
    df['depth_imbalance'] = (df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    df['relative_spread'] = np.abs(df['bid_qty'] - df['ask_qty']) / (df['total_depth'] + 1e-10)
    
    # Volume features
    df['log_volume'] = np.log1p(df['volume'])
    df['buy_sell_ratio'] = df['buy_qty'] / (df['sell_qty'] + 1e-10)
    df['volume_depth_ratio'] = df['volume'] / (df['total_depth'] + 1e-10)
    
    # Market microstructure
    df['kyle_lambda'] = np.abs(df['net_order_flow']) / (df['volume'] + 1e-10)
    df['price_impact_proxy'] = df['net_order_flow'] / (df['total_depth'] + 1e-10)
    
    # Simple moving averages
    for col in ['volume', 'net_order_flow', 'order_flow_imbalance']:
        df[f'{col}_ma5'] = df[col].rolling(5, min_periods=1).mean()
        df[f'{col}_ma10'] = df[col].rolling(10, min_periods=1).mean()
    
    # Fill NaN and inf values
    df = df.replace([np.inf, -np.inf], 0).fillna(0)
    
    return df

# Apply feature engineering
train_df = add_engineered_features(train_df)
test_df = add_engineered_features(test_df)

# Get all feature names
feature_cols = [col for col in train_df.columns if col not in ['label']]
print(f"Total features: {len(feature_cols)}")

# =========================
# 4. Prepare Training Data
# =========================
print("\n3. Preparing training data...")

# Use recent 50% of data for training
train_size = int(0.5 * len(train_df))
train_data = train_df.iloc[-train_size:].reset_index(drop=True)

# Clean up memory
del train_df
clean_memory()

# Convert to arrays
X_full = train_data[feature_cols].values.astype(np.float32)
y_full = train_data["label"].values.astype(np.float32)

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=0.2, random_state=RANDOM_SEED
)

print(f"Training samples: {len(X_train)}")
print(f"Validation samples: {len(X_val)}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Create PyTorch datasets
train_dataset = TensorDataset(
    torch.tensor(X_train_scaled, dtype=torch.float32),
    torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
)
val_dataset = TensorDataset(
    torch.tensor(X_val_scaled, dtype=torch.float32),
    torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
)

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)

# =========================
# 5. Train VIME Model
# =========================
print("\n4. Training VIME model...")

# Initialize model
model = VIMEModel(
    input_dim=len(feature_cols),
    hidden_dims=[256, 128],
    dropout_rate=0.2,
    corruption_rate=0.3
).to(device)

print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

# Phase 1: Self-supervised pretraining
print("\nPhase 1: Self-supervised pretraining...")
pretrain_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
mask_criterion = nn.BCELoss()
feature_criterion = nn.MSELoss()

for epoch in range(8):  # 8 epochs for pretraining
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, _) in enumerate(train_loader):
        inputs = inputs.to(device)
        
        pretrain_optimizer.zero_grad()
        
        # Forward pass
        mask_pred, feature_pred, true_mask, original_x = model(inputs, pretrain=True)
        
        # Calculate losses
        mask_loss = mask_criterion(mask_pred, true_mask)
        feature_loss = feature_criterion(feature_pred, original_x)
        total_loss_batch = mask_loss + feature_loss
        
        # Backward pass
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        pretrain_optimizer.step()
        
        total_loss += total_loss_batch.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f"Pretrain Epoch {epoch+1}/8: Loss = {avg_loss:.4f}")

# Phase 2: Supervised fine-tuning
print("\nPhase 2: Supervised fine-tuning...")
finetune_optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.HuberLoss()

best_val_corr = -1
patience = 0
max_patience = 5

for epoch in range(15):  # 15 epochs for fine-tuning
    # Training
    model.train()
    train_loss = 0
    
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        finetune_optimizer.zero_grad()
        outputs = model(inputs, pretrain=False)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        finetune_optimizer.step()
        
        train_loss += loss.item()
    
    # Validation
    model.eval()
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs, pretrain=False)
            val_preds.extend(outputs.cpu().numpy().flatten())
            val_targets.extend(targets.numpy().flatten())
    
    # Calculate metrics
    val_corr = pearsonr(val_targets, val_preds)[0]
    avg_train_loss = train_loss / len(train_loader)
    
    print(f"Epoch {epoch+1}/15: Train Loss = {avg_train_loss:.4f}, Val Correlation = {val_corr:.4f}")
    
    # Early stopping
    if val_corr > best_val_corr:
        best_val_corr = val_corr
        patience = 0
        # Save best model
        torch.save(model.state_dict(), 'best_vime_model.pt')
    else:
        patience += 1
        if patience >= max_patience:
            print("Early stopping triggered")
            break

# Load best model
model.load_state_dict(torch.load('best_vime_model.pt'))

# =========================
# 6. Extract Feature Importance
# =========================
print("\n5. Extracting feature importance...")

# Get learned feature importance weights
with torch.no_grad():
    vime_weights = torch.sigmoid(model.feature_importance_weights).cpu().numpy()

# Compute corruption sensitivity importance
print("Computing corruption sensitivity...")
model.eval()

# Get baseline predictions
X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32, device=device)
with torch.no_grad():
    baseline_preds = model(X_val_tensor, pretrain=False).cpu().numpy().flatten()
baseline_score = pearsonr(y_val, baseline_preds)[0]

# Test corruption sensitivity for each feature
corruption_importance = np.zeros(len(feature_cols))

for feat_idx in tqdm(range(len(feature_cols)), desc="Testing corruption sensitivity"):
    # Create corrupted version
    X_corrupted = X_val_scaled.copy()
    # Shuffle the feature values
    np.random.shuffle(X_corrupted[:, feat_idx])
    
    # Get predictions with corrupted feature
    X_corrupted_tensor = torch.tensor(X_corrupted, dtype=torch.float32, device=device)
    with torch.no_grad():
        corrupted_preds = model(X_corrupted_tensor, pretrain=False).cpu().numpy().flatten()
    
    # Calculate performance drop
    corrupted_score = pearsonr(y_val, corrupted_preds)[0]
    corruption_importance[feat_idx] = baseline_score - corrupted_score

# Combine importance scores
combined_importance = 0.7 * (vime_weights / vime_weights.max()) + \
                    0.3 * (corruption_importance / (corruption_importance.max() + 1e-8))

# Create importance dataframe
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'vime_weight': vime_weights,
    'corruption_sensitivity': corruption_importance,
    'combined_importance': combined_importance
}).sort_values('combined_importance', ascending=False)

print("\nTop 30 most important features:")
print("-" * 60)
for idx, row in importance_df.head(30).iterrows():
    print(f"{row['feature']:30s} {row['combined_importance']:.4f}")

# Save importance results
importance_df.to_csv('vime_feature_importance.csv', index=False)

# =========================
# 7. Train Models on Top Features
# =========================
print("\n6. Training models on top features...")

# Select top features
n_top_features = 150
top_features = importance_df.head(n_top_features)['feature'].tolist()

print(f"Using top {n_top_features} features for final models")

# Prepare data with top features
X_train_top = train_data[top_features].values
y_train_top = train_data['label'].values

X_test_top = test_df[top_features].values

# Scale features
scaler_final = RobustScaler()
X_train_top_scaled = scaler_final.fit_transform(X_train_top)
X_test_top_scaled = scaler_final.transform(X_test_top)

# Split for validation
X_tr, X_vl, y_tr, y_vl = train_test_split(
    X_train_top_scaled, y_train_top, test_size=0.2, random_state=RANDOM_SEED
)

# Train LightGBM
print("\nTraining LightGBM...")
lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.7,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': RANDOM_SEED,
    'n_jobs': -1
}

lgb_train = lgb.Dataset(X_tr, y_tr)
lgb_val = lgb.Dataset(X_vl, y_vl, reference=lgb_train)

lgb_model = lgb.train(
    lgb_params,
    lgb_train,
    valid_sets=[lgb_val],
    num_boost_round=1500,
    callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)]
)

# Evaluate LightGBM
lgb_val_pred = lgb_model.predict(X_vl, num_iteration=lgb_model.best_iteration)
lgb_corr = pearsonr(y_vl, lgb_val_pred)[0]
print(f"LightGBM validation correlation: {lgb_corr:.4f}")

# Train XGBoost
print("\nTraining XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    learning_rate=0.02,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    tree_method='hist',
    n_jobs=-1
)

xgb_model.fit(
    X_tr, y_tr,
    eval_set=[(X_vl, y_vl)],
    early_stopping_rounds=100,
    verbose=False
)

# Evaluate XGBoost
xgb_val_pred = xgb_model.predict(X_vl)
xgb_corr = pearsonr(y_vl, xgb_val_pred)[0]
print(f"XGBoost validation correlation: {xgb_corr:.4f}")

# =========================
# 8. Generate Test Predictions
# =========================
print("\n7. Generating test predictions...")

# Get predictions from both models
lgb_test_pred = lgb_model.predict(X_test_top_scaled, num_iteration=lgb_model.best_iteration)
xgb_test_pred = xgb_model.predict(X_test_top_scaled)

# Ensemble predictions
ensemble_pred = 0.6 * lgb_test_pred + 0.4 * xgb_test_pred

# Get VIME neural network predictions
X_test_nn_scaled = scaler.transform(test_df[feature_cols].values)
X_test_nn_tensor = torch.tensor(X_test_nn_scaled, dtype=torch.float32, device=device)

model.eval()
with torch.no_grad():
    nn_test_pred = model(X_test_nn_tensor, pretrain=False).cpu().numpy().flatten()

# Final ensemble including neural network
final_ensemble = 0.5 * ensemble_pred + 0.3 * lgb_test_pred + 0.2 * nn_test_pred

# =========================
# 9. Create Submission Files - CORRECTED VERSION
# =========================
print("\n8. Creating submission files...")

# Get the number of test samples
n_test_samples = len(test_df)

# Create submission dataframe with correct format
def create_submission(predictions, filename):
    """Create submission file with ID and prediction columns"""
    submission_df = pd.DataFrame({
        'ID': range(1, n_test_samples + 1),  # IDs from 1 to n_test_samples
        'prediction': predictions
    })
    submission_df.to_csv(filename, index=False)
    return submission_df

# Create multiple submissions
submissions = {
    'submission_vime_lgb.csv': lgb_test_pred,
    'submission_vime_xgb.csv': xgb_test_pred,
    'submission_vime_ensemble.csv': ensemble_pred,
    'submission_vime_nn.csv': nn_test_pred,
    'submission_vime_final.csv': final_ensemble
}

# Generate each submission file
for filename, predictions in submissions.items():
    sub_df = create_submission(predictions, filename)
    print(f"Created {filename} - Shape: {sub_df.shape}, Columns: {list(sub_df.columns)}")

# Display sample of final submission
print("\nSample of final submission (submission_vime_final.csv):")
final_sub = pd.read_csv('submission_vime_final.csv')
print(final_sub.head(10))
print(f"\nTotal rows in submission: {len(final_sub)}")

# Verify submission format
print("\nVerifying submission format:")
print(f"- First ID: {final_sub['ID'].iloc[0]}")
print(f"- Last ID: {final_sub['ID'].iloc[-1]}")
print(f"- Number of rows: {len(final_sub)}")
print(f"- Columns: {list(final_sub.columns)}")
print(f"- Prediction range: [{final_sub['prediction'].min():.6f}, {final_sub['prediction'].max():.6f}]")

# =========================
# 10. Summary Report
# =========================
print("\n" + "="*60)
print("VIME FEATURE IMPORTANCE ANALYSIS COMPLETE")
print("="*60)
print(f"Total features analyzed: {len(feature_cols)}")
print(f"Top features selected: {n_top_features}")
print(f"Training samples: {len(X_train)}")
print(f"Test samples: {n_test_samples}")
print(f"\nModel Performance:")
print(f"  VIME NN validation correlation: {best_val_corr:.4f}")
print(f"  LightGBM validation correlation: {lgb_corr:.4f}")
print(f"  XGBoost validation correlation: {xgb_corr:.4f}")

# Feature importance summary
x_features_important = importance_df[importance_df['feature'].str.startswith('X')].head(20)
market_features_important = importance_df[importance_df['feature'].isin(MARKET_FEATURES)]
engineered_features_important = importance_df[~importance_df['feature'].str.startswith('X') & 
                                           ~importance_df['feature'].isin(MARKET_FEATURES)].head(10)

print(f"\nTop Anonymous Features (X_):")
for _, row in x_features_important.head(10).iterrows():
    print(f"  {row['feature']:10s} importance: {row['combined_importance']:.4f}")

print(f"\nMarket Features Ranking:")
for _, row in market_features_important.iterrows():
    print(f"  {row['feature']:10s} importance: {row['combined_importance']:.4f}")

print(f"\nTop Engineered Features:")
for _, row in engineered_features_important.head(5).iterrows():
    print(f"  {row['feature']:25s} importance: {row['combined_importance']:.4f}")

print("\n✅ Analysis complete!")
print("\nAll submission files have been created with the correct format:")
print("- ID column (starting from 1)")
print("- prediction column (model predictions)")
print("\nRecommended submission: submission_vime_final.csv")
print("This combines VIME neural network with tree-based models for best performance.")

# Clean up
clean_memory()

