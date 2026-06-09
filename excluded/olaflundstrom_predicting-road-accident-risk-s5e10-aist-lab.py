# Kaggle Notebook: Predicting Road Accident Risk
# Playground Series - Season 5, Episode 10
# Team: AIST Lab - Artificial Intelligence (AI), Simulation and Teaching Laboratory
# Institution: Department of Behavioural Sciences and Learning (IBL), Linköping University, SE-581 83 Linköping, Sweden
# Authors: PhD Cand. Olaf Yunus LAITINEN IMANOV, Mehmet Ugur KURU, Mehmet KAHRAMAN, Mehmet Unlu

# =====================
# 1. Library Imports
# =====================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder, QuantileTransformer
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import warnings
warnings.filterwarnings('ignore')

# =====================
# 2. Load Datasets
# =====================
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')
synthetic_100k = pd.read_csv('/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_100k.csv')

print(f"Original train: {train.shape}")
print(f"Synthetic data: {synthetic_100k.shape}")
print(f"Test: {test.shape}")

# Store ids
test_ids = test['id'].copy()
train_ids = train['id'].copy()

# =====================
# 3. Advanced Feature Engineering
# =====================
def create_advanced_features(df):
    """Enhanced feature engineering with domain knowledge"""
    df = df.copy()
    
    # Convert booleans
    bool_cols = df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        df[col] = df[col].astype(int)
    
    # === CORE INTERACTIONS ===
    df['speed_curvature'] = df['speed_limit'] * df['curvature']
    df['accidents_curvature'] = df['num_reported_accidents'] * df['curvature']
    df['accidents_speed'] = df['num_reported_accidents'] * df['speed_limit']
    df['lanes_accidents'] = df['num_lanes'] * df['num_reported_accidents']
    
    # === ADVANCED INTERACTIONS ===
    df['speed_curvature_accidents'] = df['speed_limit'] * df['curvature'] * df['num_reported_accidents']
    df['complex_risk'] = df['speed_limit'] * df['curvature'] * df['num_reported_accidents'] / (df['num_lanes'] + 1)
    
    # === RATIOS & DENSITIES ===
    df['lanes_speed_ratio'] = df['num_lanes'] / (df['speed_limit'] + 1)
    df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'] + 1)
    df['accident_density'] = df['num_reported_accidents'] / (df['num_lanes'] * df['speed_limit'] + 1)
    df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'] + 1)
    df['curvature_per_lane'] = df['curvature'] / (df['num_lanes'] + 1)
    
    # === POLYNOMIAL FEATURES ===
    df['curvature_squared'] = df['curvature'] ** 2
    df['curvature_cubed'] = df['curvature'] ** 3
    df['speed_squared'] = df['speed_limit'] ** 2
    df['accidents_squared'] = df['num_reported_accidents'] ** 2
    
    # === LOGARITHMIC TRANSFORMS ===
    df['log_speed'] = np.log1p(df['speed_limit'])
    df['log_accidents'] = np.log1p(df['num_reported_accidents'])
    df['log_curvature'] = np.log1p(df['curvature'])
    
    # === EXPONENTIAL FEATURES ===
    df['exp_curvature'] = np.exp(df['curvature'] / 10)  # Scale down to avoid overflow
    df['exp_speed_curve'] = np.exp((df['speed_limit'] * df['curvature']) / 1000)
    
    # === RISK INDICATORS ===
    df['high_risk'] = ((df['curvature'] > 0.7) & (df['speed_limit'] >= 60)).astype(int)
    df['extreme_curve'] = (df['curvature'] > 0.8).astype(int)
    df['extreme_speed'] = (df['speed_limit'] > 70).astype(int)
    df['high_accident_area'] = (df['num_reported_accidents'] > df['num_reported_accidents'].median()).astype(int)
    df['danger_zone'] = ((df['curvature'] > 0.6) & (df['speed_limit'] >= 55) & (df['num_reported_accidents'] > 10)).astype(int)
    
    # === BINNED FEATURES ===
    df['speed_bin'] = pd.cut(df['speed_limit'], bins=[0, 35, 50, 65, 100], labels=False)
    df['curve_bin'] = pd.cut(df['curvature'], bins=[0, 0.3, 0.6, 0.8, 1.0], labels=False)
    df['accident_bin'] = pd.cut(df['num_reported_accidents'], bins=[0, 5, 15, 30, 100], labels=False)
    
    # === INTERACTION BINS ===
    df['speed_curve_bin'] = df['speed_bin'] * 10 + df['curve_bin']
    
    # === STATISTICAL AGGREGATIONS (per category) ===
    # These will be created after categorical encoding
    
    return df

# =====================
# 4. Target Encoding for Categorical Features
# =====================
def add_target_encoding(train_df, test_df, target, cat_cols, n_folds=5):
    """Add target encoding with cross-validation"""
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    # Reset target index to match train_df
    target = target.reset_index(drop=True)
    
    for col in cat_cols:
        # Initialize encoded columns
        train_df[f'{col}_target_enc'] = 0.0
        test_df[f'{col}_target_enc'] = 0.0
        
        # Get global mean
        global_mean = target.mean()
        
        # Cross-validation target encoding for train
        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        y_bins = pd.qcut(target, q=10, labels=False, duplicates='drop')
        
        for train_idx, val_idx in kf.split(train_df, y_bins):
            # Calculate means on train fold
            train_fold_data = pd.DataFrame({
                'category': train_df.iloc[train_idx][col].values,
                'target': target.iloc[train_idx].values
            })
            means = train_fold_data.groupby('category')['target'].mean().to_dict()
            
            # Apply to validation fold
            train_df.loc[val_idx, f'{col}_target_enc'] = train_df.iloc[val_idx][col].map(means).fillna(global_mean)
        
        # For test, use full train data
        full_data = pd.DataFrame({
            'category': train_df[col].values,
            'target': target.values
        })
        means = full_data.groupby('category')['target'].mean().to_dict()
        test_df[f'{col}_target_enc'] = test_df[col].map(means).fillna(global_mean)
    
    return train_df, test_df

# =====================
# 5. Prepare Data
# =====================
train_data = train.drop('id', axis=1)
test_data = test.drop('id', axis=1)

# Apply feature engineering
train_data = create_advanced_features(train_data)
test_data = create_advanced_features(test_data)

# Categorical columns
cat_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

# Label encoding first
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    train_data[col] = le.fit_transform(train_data[col].astype(str))
    test_data[col] = le.transform(test_data[col].astype(str))
    encoders[col] = le

# Target encoding
y_temp = train_data['accident_risk']
train_data, test_data = add_target_encoding(
    train_data.drop('accident_risk', axis=1), 
    test_data, 
    y_temp, 
    cat_cols
)
train_data['accident_risk'] = y_temp

# Statistical features per category
for col in cat_cols:
    for feat in ['speed_limit', 'curvature', 'num_reported_accidents']:
        # Mean
        agg_mean = train_data.groupby(col)[feat].mean().to_dict()
        train_data[f'{col}_{feat}_mean'] = train_data[col].map(agg_mean)
        test_data[f'{col}_{feat}_mean'] = test_data[col].map(agg_mean)
        
        # Std
        agg_std = train_data.groupby(col)[feat].std().fillna(0).to_dict()
        train_data[f'{col}_{feat}_std'] = train_data[col].map(agg_std)
        test_data[f'{col}_{feat}_std'] = test_data[col].map(agg_std)

X = train_data.drop('accident_risk', axis=1)
y = train_data['accident_risk']
X_test = test_data.copy()

print(f"\nFinal shape - Train: {X.shape}, Test: {X_test.shape}")
print(f"Number of features: {X.shape[1]}")

# =====================
# 6. Ensemble Training
# =====================
print("\n" + "="*60)
print("TRAINING ENSEMBLE MODELS")
print("="*60)

# Stratified CV
y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
n_folds = 7  # Increased folds for better stability
folds = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Storage for predictions
oof_lgb = np.zeros(len(X))
oof_xgb = np.zeros(len(X))
oof_cat = np.zeros(len(X))

test_lgb = np.zeros(len(X_test))
test_xgb = np.zeros(len(X_test))
test_cat = np.zeros(len(X_test))

fold_scores_lgb = []
fold_scores_xgb = []
fold_scores_cat = []

# === LightGBM Parameters ===
params_lgb = {
    'objective': 'regression',
    'metric': 'rmse',
    'learning_rate': 0.008,
    'num_leaves': 127,
    'max_depth': 10,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'min_child_samples': 25,
    'reg_alpha': 0.15,
    'reg_lambda': 0.15,
    'max_bin': 255,
    'min_data_in_leaf': 20,
    'seed': 42,
    'verbose': -1,
    'n_jobs': -1
}

# === XGBoost Parameters ===
params_xgb = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'learning_rate': 0.008,
    'max_depth': 9,
    'subsample': 0.75,
    'colsample_bytree': 0.75,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'seed': 42,
    'tree_method': 'hist',
    'n_jobs': -1
}

# === CatBoost Parameters ===
params_cat = {
    'loss_function': 'RMSE',
    'learning_rate': 0.008,
    'depth': 9,
    'l2_leaf_reg': 3,
    'subsample': 0.75,
    'random_seed': 42,
    'verbose': False
}

# Training loop
for fold, (train_idx, val_idx) in enumerate(folds.split(X, y_bins)):
    print(f"\n{'='*60}")
    print(f"Fold {fold+1}/{n_folds}")
    print(f"{'='*60}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # --- LightGBM ---
    print("Training LightGBM...")
    model_lgb = lgb.LGBMRegressor(**params_lgb, n_estimators=10000)
    model_lgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(300, verbose=False)]
    )
    oof_lgb[val_idx] = model_lgb.predict(X_val)
    test_lgb += model_lgb.predict(X_test) / n_folds
    rmse_lgb = np.sqrt(mean_squared_error(y_val, oof_lgb[val_idx]))
    fold_scores_lgb.append(rmse_lgb)
    print(f"LightGBM RMSE: {rmse_lgb:.6f}")
    
    # --- XGBoost ---
    print("Training XGBoost...")
    model_xgb = xgb.XGBRegressor(**params_xgb, n_estimators=10000)
    model_xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=300,
        verbose=False
    )
    oof_xgb[val_idx] = model_xgb.predict(X_val)
    test_xgb += model_xgb.predict(X_test) / n_folds
    rmse_xgb = np.sqrt(mean_squared_error(y_val, oof_xgb[val_idx]))
    fold_scores_xgb.append(rmse_xgb)
    print(f"XGBoost RMSE: {rmse_xgb:.6f}")
    
    # --- CatBoost ---
    print("Training CatBoost...")
    model_cat = CatBoostRegressor(**params_cat, iterations=10000)
    model_cat.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=300,
        verbose=False
    )
    oof_cat[val_idx] = model_cat.predict(X_val)
    test_cat += model_cat.predict(X_test) / n_folds
    rmse_cat = np.sqrt(mean_squared_error(y_val, oof_cat[val_idx]))
    fold_scores_cat.append(rmse_cat)
    print(f"CatBoost RMSE: {rmse_cat:.6f}")

# =====================
# 7. Ensemble Blending
# =====================
print("\n" + "="*60)
print("ENSEMBLE RESULTS")
print("="*60)

print(f"\nLightGBM - Mean CV: {np.mean(fold_scores_lgb):.6f} (+/- {np.std(fold_scores_lgb):.6f})")
print(f"XGBoost  - Mean CV: {np.mean(fold_scores_xgb):.6f} (+/- {np.std(fold_scores_xgb):.6f})")
print(f"CatBoost - Mean CV: {np.mean(fold_scores_cat):.6f} (+/- {np.std(fold_scores_cat):.6f})")

# Optimal blending weights (tune these based on CV scores)
w_lgb = 0.4
w_xgb = 0.35
w_cat = 0.25

oof_blend = w_lgb * oof_lgb + w_xgb * oof_xgb + w_cat * oof_cat
test_blend = w_lgb * test_lgb + w_xgb * test_xgb + w_cat * test_cat

oof_rmse = np.sqrt(mean_squared_error(y, oof_blend))
print(f"\nBlended OOF RMSE: {oof_rmse:.6f}")
print(f"Weights - LGB: {w_lgb}, XGB: {w_xgb}, CAT: {w_cat}")

# =====================
# 8. Submission
# =====================
submission = sample_submission.copy()
submission['accident_risk'] = np.clip(test_blend, 0, 1)

print("\n" + "="*60)
print("SUBMISSION STATISTICS")
print("="*60)
print(submission['accident_risk'].describe())

submission.to_csv('submission.csv', index=False)
print("\n✓ Submission file created successfully!")
print("="*60)

# Distribution comparison
print("\nDistribution Analysis:")
print(f"Train target - Mean: {y.mean():.4f}, Std: {y.std():.4f}")
print(f"Predictions  - Mean: {test_blend.mean():.4f}, Std: {test_blend.std():.4f}")
print(f"Difference   - Mean: {abs(y.mean() - test_blend.mean()):.4f}, Std: {abs(y.std() - test_blend.std()):.4f}")

