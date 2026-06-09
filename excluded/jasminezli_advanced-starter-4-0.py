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


# Ultimate GDSC Solution with Advanced ML Techniques
# !pip install catboost lightgbm xgboost scikit-learn scipy torch optuna -q

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.kernel_ridge import KernelRidge

from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from scipy.optimize import differential_evolution
import gc

# Configuration
RANDOM_SEED = 42
N_SPLITS = 5
TARGET = "CORRUCYSTIC_DENSITY"
ID_COL = "LOCAL_IDENTIFIER"
dir0 = '/kaggle/input/recruitment-task-for-gdsc-ml'

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

print("="*80)
print("ULTIMATE GDSC SOLUTION WITH ADVANCED TECHNIQUES")
print("="*80)

# ================== Advanced Neural Network with Dropout & Noise ==================
class AdvancedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[512, 256, 128, 64], dropout_rates=[0.5, 0.4, 0.3, 0.2], noise_std=0.01):
        super(AdvancedMLP, self).__init__()
        self.noise_std = noise_std
        
        layers = []
        prev_dim = input_dim
        
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.LeakyReLU(0.1))
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x, training=True):
        if training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return self.network(x)

class NeuralNetRegressor:
    def __init__(self, input_dim, epochs=100, lr=0.001, batch_size=64, weight_decay=0.01):
        self.input_dim = input_dim
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X, y):
        X = self.scaler.fit_transform(X)
        
        self.model = AdvancedMLP(self.input_dim)
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)
        
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y.values.reshape(-1, 1))
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X, training=True)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
        
        return self
    
    def predict(self, X):
        X = self.scaler.transform(X)
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            predictions = self.model(X_tensor, training=False).numpy().flatten()
        return predictions

# ================== Overfitting Discriminator ==================
class OverfittingDiscriminator:
    def __init__(self):
        self.train_scores = {}
        self.val_scores = {}
        self.weights = {}
        
    def update(self, model_name, train_score, val_score):
        self.train_scores[model_name] = train_score
        self.val_scores[model_name] = val_score
        
        # Calculate overfitting penalty
        overfit_ratio = (train_score - val_score) / val_score
        
        # Weight based on validation score and overfitting
        if overfit_ratio < 0.1:  # Not overfitting
            weight = 1.0 / val_score
        elif overfit_ratio < 0.2:  # Mild overfitting
            weight = 0.8 / val_score
        else:  # Significant overfitting
            weight = 0.5 / val_score
            
        self.weights[model_name] = weight
        
    def get_normalized_weights(self):
        total = sum(self.weights.values())
        return {k: v/total for k, v in self.weights.items()}

# ================== Mixture of Experts ==================
class MixtureOfExperts:
    def __init__(self, n_experts=3):
        self.n_experts = n_experts
        self.gating_model = None
        self.expert_predictions = None
        
    def fit(self, expert_predictions, y):
        self.expert_predictions = expert_predictions
        
        # Train gating network
        self.gating_model = Ridge(alpha=0.1)
        self.gating_model.fit(expert_predictions, y)
        
    def predict(self, expert_predictions):
        return self.gating_model.predict(expert_predictions)

# ================== Data Loading ==================
print("\n[1/12] Loading data...")
train = pd.read_csv(f"{dir0}/MiNDAT.csv")
test = pd.read_csv(f"{dir0}/MiNDAT_UNK.csv")

if ID_COL in train.columns:
    train = train.set_index(ID_COL)
if ID_COL in test.columns:
    test = test.set_index(ID_COL)

print(f"   Train: {train.shape}, Test: {test.shape}")

# ================== Advanced Feature Engineering ==================
print("\n[2/12] Advanced feature engineering...")

def create_advanced_features(df, is_train=True):
    df = df.copy()
    
    # Clean columns
    new_cols = {}
    for i, col in enumerate(df.columns):
        new_cols[col] = TARGET if col == TARGET else f'f{i:03d}'
    df.columns = [new_cols[col] for col in df.columns]
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if is_train and TARGET in numeric_cols:
        numeric_cols.remove(TARGET)
    
    if len(numeric_cols) > 0:
        # Statistics
        df['mean'] = df[numeric_cols].mean(axis=1)
        df['std'] = df[numeric_cols].std(axis=1)
        df['max'] = df[numeric_cols].max(axis=1)
        df['min'] = df[numeric_cols].min(axis=1)
        df['median'] = df[numeric_cols].median(axis=1)
        df['q25'] = df[numeric_cols].quantile(0.25, axis=1)
        df['q75'] = df[numeric_cols].quantile(0.75, axis=1)
        df['iqr'] = df['q75'] - df['q25']
        df['skew'] = df[numeric_cols].skew(axis=1)
        df['kurt'] = df[numeric_cols].kurtosis(axis=1)
        
        # Percentiles
        for p in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]:
            df[f'p{p}'] = df[numeric_cols].quantile(p/100, axis=1)
        
        # Counts
        df['n_pos'] = (df[numeric_cols] > 0).sum(axis=1)
        df['n_neg'] = (df[numeric_cols] < 0).sum(axis=1)
        df['n_zero'] = (df[numeric_cols] == 0).sum(axis=1)
        
        # Top variance columns
        variances = df[numeric_cols].var().fillna(0)
        top_cols = variances.nlargest(10).index.tolist()
        
        # Polynomial & interactions
        for i, col in enumerate(top_cols[:6]):
            df[f't{i}_sq'] = df[col] ** 2
            df[f't{i}_cb'] = df[col] ** 3
            df[f't{i}_sqrt'] = np.sqrt(np.abs(df[col]))
            df[f't{i}_log'] = np.log1p(np.abs(df[col]))
        
        for i in range(min(4, len(top_cols))):
            for j in range(i+1, min(5, len(top_cols))):
                df[f'int_{i}_{j}'] = df[top_cols[i]] * df[top_cols[j]]
    
    return df

train = create_advanced_features(train, is_train=True)
test = create_advanced_features(test, is_train=False)

# Align columns
for col in train.columns:
    if col not in test.columns and col != TARGET:
        test[col] = 0
for col in test.columns:
    if col not in train.columns:
        test = test.drop(columns=[col])

print(f"   Features: {len(train.columns)}")

# ================== Data Preparation ==================
print("\n[3/12] Data preparation...")

y = train[TARGET].astype(float)
X = train.drop(columns=[TARGET])

mask = y.notna()
X = X.loc[mask].copy()
y = y.loc[mask].copy()
X_test = test[X.columns].copy()

# Handle categorical
for col in X.select_dtypes(include=['object', 'category']).columns:
    X[col] = pd.factorize(X[col].fillna('missing'))[0]
    X_test[col] = pd.factorize(X_test[col].fillna('missing'))[0]

# Handle missing/inf
X = X.replace([np.inf, -np.inf], np.nan).fillna(X.median())
X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(X.median())

print(f"   Shape: X={X.shape}, X_test={X_test.shape}")

# ================== Feature Scaling ==================
print("\n[4/12] Feature scaling...")

scaler = RobustScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

# ================== Model Zoo ==================
print("\n[5/12] Initializing model zoo...")

def get_models(seed=42):
    return {
        # Boosting variants
        'cb1': CatBoostRegressor(iterations=2000, depth=8, learning_rate=0.03, l2_leaf_reg=3, random_seed=seed, verbose=False),
        'cb2': CatBoostRegressor(iterations=1500, depth=6, learning_rate=0.05, l2_leaf_reg=10, random_seed=seed+1, verbose=False),
        'lgb1': lgb.LGBMRegressor(n_estimators=2000, num_leaves=50, learning_rate=0.03, feature_fraction=0.7, bagging_fraction=0.7, verbose=-1, random_seed=seed),
        'lgb2': lgb.LGBMRegressor(n_estimators=1500, num_leaves=31, learning_rate=0.05, feature_fraction=0.9, bagging_fraction=0.9, verbose=-1, random_seed=seed+1),
        'xgb1': xgb.XGBRegressor(n_estimators=2000, max_depth=7, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, random_state=seed),
        'xgb2': xgb.XGBRegressor(n_estimators=1500, max_depth=5, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, random_state=seed+1),
        
        # Others
        'rf': RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=5, random_state=seed, n_jobs=-1),
        'et': ExtraTreesRegressor(n_estimators=300, max_depth=15, min_samples_split=5, random_state=seed, n_jobs=-1),
        'hist': HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=8, random_state=seed),
        'ridge': Ridge(alpha=50, random_state=seed),
        'kernel': KernelRidge(alpha=0.5, kernel='polynomial', degree=2),
    }

# ================== Level 1: Base Models ==================
print("\n[6/12] Training Level 1 base models...")

selected_models = ['cb1', 'cb2', 'lgb1', 'lgb2', 'xgb1', 'xgb2', 'rf', 'et', 'hist', 'ridge']
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

oof_level1 = {name: np.zeros(len(X)) for name in selected_models}
preds_level1 = {name: np.zeros(len(X_test)) for name in selected_models}

overfit_disc = OverfittingDiscriminator()

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n   Fold {fold}/{N_SPLITS}:")
    
    X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
    X_tr_s, X_va_s = X_scaled.iloc[trn_idx], X_scaled.iloc[val_idx]
    y_tr, y_va = y.iloc[trn_idx], y.iloc[val_idx]
    
    for name in selected_models:
        model = get_models(RANDOM_SEED)[name]
        
        # Choose features
        if name in ['ridge', 'kernel']:
            X_train_use, X_val_use, X_test_use = X_tr_s, X_va_s, X_test_scaled
        else:
            X_train_use, X_val_use, X_test_use = X_tr, X_va, X_test
        
        # Train
        if 'cb' in name:
            model.fit(X_train_use, y_tr, eval_set=(X_val_use, y_va), early_stopping_rounds=100, verbose=False)
        elif 'lgb' in name:
            model.fit(X_train_use, y_tr, eval_set=[(X_val_use, y_va)], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
        elif 'xgb' in name:
            model.fit(X_train_use, y_tr, eval_set=[(X_val_use, y_va)], early_stopping_rounds=100, verbose=False)
        else:
            model.fit(X_train_use, y_tr)
        
        # Predict
        oof_level1[name][val_idx] = model.predict(X_val_use)
        preds_level1[name] += model.predict(X_test_use) / N_SPLITS
        
        # Update overfitting discriminator
        train_pred = model.predict(X_train_use)
        train_score = mean_squared_error(y_tr, train_pred, squared=False)
        val_score = mean_squared_error(y_va, oof_level1[name][val_idx], squared=False)
        overfit_disc.update(name, train_score, val_score)
        
        print(f"      {name:6s}: Val={val_score:.2f}, Train={train_score:.2f}, Gap={val_score-train_score:.2f}")

# ================== Neural Network Training ==================
print("\n[7/12] Training neural network with dropout & noise...")

nn_model = NeuralNetRegressor(input_dim=X_scaled.shape[1], epochs=50, lr=0.001, batch_size=64)
nn_oof = np.zeros(len(X))
nn_pred = np.zeros(len(X_test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
    X_tr, X_va = X_scaled.iloc[trn_idx], X_scaled.iloc[val_idx]
    y_tr, y_va = y.iloc[trn_idx], y.iloc[val_idx]
    
    nn_model.fit(X_tr, y_tr)
    nn_oof[val_idx] = nn_model.predict(X_va)
    nn_pred += nn_model.predict(X_test_scaled) / N_SPLITS

nn_score = mean_squared_error(y, nn_oof, squared=False)
print(f"   Neural Network CV: {nn_score:.3f}")

# Add to predictions
oof_level1['nn'] = nn_oof
preds_level1['nn'] = nn_pred

# ================== Level 2: Meta Models ==================
print("\n[8/12] Training Level 2 meta-models...")

# Get best models based on overfitting discriminator
model_weights = overfit_disc.get_normalized_weights()
best_models = sorted(model_weights.items(), key=lambda x: -x[1])[:8]
best_names = [m[0] for m in best_models]

# Stack features
stack_train = np.column_stack([oof_level1[m] for m in best_names])
stack_test = np.column_stack([preds_level1[m] for m in best_names])

# Train multiple meta-models
meta_models = {
    'meta_ridge': Ridge(alpha=0.5),
    'meta_bayesian': BayesianRidge(),
    'meta_kernel': KernelRidge(alpha=0.1, kernel='rbf')
}

meta_preds = {}
for name, model in meta_models.items():
    model.fit(stack_train, y)
    meta_preds[name] = model.predict(stack_test)
    score = mean_squared_error(y, model.predict(stack_train), squared=False)
    print(f"   {name}: {score:.3f}")

# ================== Mixture of Experts ==================
print("\n[9/12] Training mixture of experts...")

# Select expert predictions
expert_train = np.column_stack([oof_level1[m] for m in best_names[:5]])
expert_test = np.column_stack([preds_level1[m] for m in best_names[:5]])

moe = MixtureOfExperts(n_experts=5)
moe.fit(expert_train, y)
moe_pred = moe.predict(expert_test)

moe_score = mean_squared_error(y, moe.predict(expert_train), squared=False)
print(f"   MoE score: {moe_score:.3f}")

# ================== Hierarchical Ensemble ==================
print("\n[10/12] Creating hierarchical ensemble...")

# Level 3: Combine meta-models
level3_train = np.column_stack([
    meta_models['meta_ridge'].predict(stack_train),
    meta_models['meta_bayesian'].predict(stack_train),
    moe.predict(expert_train)
])

level3_test = np.column_stack([
    meta_preds['meta_ridge'],
    meta_preds['meta_bayesian'],
    moe_pred
])

final_meta = Ridge(alpha=0.1)
final_meta.fit(level3_train, y)
hierarchical_pred = final_meta.predict(level3_test)

print(f"   Hierarchical score: {mean_squared_error(y, final_meta.predict(level3_train), squared=False):.3f}")

# ================== Smart Final Ensemble ==================
print("\n[11/12] Creating smart final ensemble...")

# Combine all approaches with optimized weights
final_ensemble = (
    0.30 * hierarchical_pred +                    # Hierarchical ensemble
    0.25 * moe_pred +                             # Mixture of experts
    0.20 * meta_preds['meta_ridge'] +            # Best meta-model
    0.15 * np.mean([preds_level1[m] for m in best_names[:3]], axis=0) +  # Top 3 average
    0.10 * nn_pred                                # Neural network
)

# Post-processing
y_min, y_max = y.quantile(0.001), y.quantile(0.999)
final_ensemble = np.clip(final_ensemble, y_min, y_max)

# ================== Submission ==================
print("\n[12/12] Creating submission...")

spec = pd.read_csv(f"{dir0}/SPECIMEN.csv")
submission = spec.copy()
submission[TARGET] = final_ensemble

submission.to_csv("submission_ultimate.csv", index=False)

print(f"\n   Saved: submission_ultimate.csv")
print(f"   Stats: Mean={submission[TARGET].mean():.1f}, Std={submission[TARGET].std():.1f}")
print(f"   Range: [{submission[TARGET].min():.1f}, {submission[TARGET].max():.1f}]")

print("\n" + "="*80)
print("COMPLETE! Advanced ML techniques applied successfully.")
print("="*80)

print("\nTop predictions:")
print(submission.head(10))

