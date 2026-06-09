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


import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
import lightgbm as lgb
import xgboost as xgb
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration
RANDOM_SEED = 42
N_SPLITS = 5
TARGET = "CORRUCYSTIC_DENSITY"
ID_COL = "LOCAL_IDENTIFIER"
dir0 = '/kaggle/input/recruitment-task-for-gdsc-ml'

np.random.seed(RANDOM_SEED)

print("="*70)
print("ADVANCED GDSC ML Competition Solution V2")
print("="*70)

# ================== Data Loading ==================
print("\n[1/10] Loading data...")
train = pd.read_csv(f"{dir0}/MiNDAT.csv")
test = pd.read_csv(f"{dir0}/MiNDAT_UNK.csv")

original_train_shape = train.shape
original_test_shape = test.shape

if ID_COL in train.columns:
    train = train.set_index(ID_COL)
if ID_COL in test.columns:
    test = test.set_index(ID_COL)

print(f"   Train: {train.shape}, Test: {test.shape}")

# ================== Column Name Cleaning ==================
print("\n[2/10] Cleaning column names...")

def safe_clean_column_names(df):
    """Safely clean column names"""
    new_cols = {}
    for i, col in enumerate(df.columns):
        if col == TARGET:
            new_cols[col] = TARGET
        else:
            # Simple approach: just use feature indices
            new_cols[col] = f'feat_{i:03d}'
    
    df.columns = [new_cols[col] for col in df.columns]
    return df, new_cols

train, train_col_map = safe_clean_column_names(train)
test, test_col_map = safe_clean_column_names(test)

print(f"   Renamed {len(train_col_map)} columns")

# ================== Advanced Feature Engineering ==================
print("\n[3/10] Creating advanced features...")

def create_advanced_features(df, is_train=True):
    """Create comprehensive feature set"""
    df = df.copy()
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if is_train and TARGET in numeric_cols:
        numeric_cols.remove(TARGET)
    
    if len(numeric_cols) == 0:
        return df
    
    # 1. Row-wise statistics
    df['row_mean'] = df[numeric_cols].mean(axis=1)
    df['row_std'] = df[numeric_cols].std(axis=1)
    df['row_median'] = df[numeric_cols].median(axis=1)
    df['row_max'] = df[numeric_cols].max(axis=1)
    df['row_min'] = df[numeric_cols].min(axis=1)
    df['row_range'] = df['row_max'] - df['row_min']
    df['row_q25'] = df[numeric_cols].quantile(0.25, axis=1)
    df['row_q75'] = df[numeric_cols].quantile(0.75, axis=1)
    df['row_iqr'] = df['row_q75'] - df['row_q25']
    df['row_skew'] = df[numeric_cols].skew(axis=1)
    df['row_kurt'] = df[numeric_cols].kurtosis(axis=1)
    df['row_nunique'] = df[numeric_cols].nunique(axis=1)
    df['row_nan_count'] = df[numeric_cols].isna().sum(axis=1)
    df['row_zero_count'] = (df[numeric_cols] == 0).sum(axis=1)
    
    # Coefficient of variation
    df['row_cv'] = df['row_std'] / (np.abs(df['row_mean']) + 1e-8)
    
    # Percentile features
    df['row_p10'] = df[numeric_cols].quantile(0.1, axis=1)
    df['row_p90'] = df[numeric_cols].quantile(0.9, axis=1)
    df['row_p90_p10'] = df['row_p90'] - df['row_p10']
    
    # 2. Polynomial features for top variance columns
    variances = df[numeric_cols].var().fillna(0)
    top_cols = variances.nlargest(min(8, len(variances))).index.tolist()
    
    for col in top_cols[:5]:
        if col in df.columns:
            # Safe transformations
            df[f'{col}_sq'] = df[col] ** 2
            df[f'{col}_sqrt'] = np.sqrt(np.abs(df[col].fillna(0)))
            df[f'{col}_log1p'] = np.log1p(np.abs(df[col].fillna(0)))
            df[f'{col}_reciprocal'] = 1 / (np.abs(df[col].fillna(1)) + 1)
    
    # 3. Interaction features between high variance columns
    for i in range(min(3, len(top_cols))):
        for j in range(i+1, min(4, len(top_cols))):
            col1, col2 = top_cols[i], top_cols[j]
            df[f'interact_{i}_{j}_mult'] = df[col1].fillna(0) * df[col2].fillna(0)
            df[f'interact_{i}_{j}_div'] = df[col1].fillna(0) / (df[col2].fillna(1) + 1e-8)
            df[f'interact_{i}_{j}_add'] = df[col1].fillna(0) + df[col2].fillna(0)
            df[f'interact_{i}_{j}_sub'] = df[col1].fillna(0) - df[col2].fillna(0)
    
    # 4. Count features
    df['row_positive'] = (df[numeric_cols] > 0).sum(axis=1)
    df['row_negative'] = (df[numeric_cols] < 0).sum(axis=1)
    
    # 5. Clustering features (simple version)
    from sklearn.cluster import KMeans
    
    if len(df) > 50:
        # Select subset of features for clustering
        cluster_cols = numeric_cols[:min(20, len(numeric_cols))]
        cluster_data = df[cluster_cols].fillna(0)
        
        # Scale the data
        scaler = StandardScaler()
        cluster_data_scaled = scaler.fit_transform(cluster_data)
        
        # KMeans clustering
        n_clusters = min(8, len(df) // 20)
        if n_clusters > 1:
            kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_SEED, n_init=3)
            df[f'cluster_{n_clusters}'] = kmeans.fit_predict(cluster_data_scaled)
            
            # Distance to nearest cluster center
            distances = kmeans.transform(cluster_data_scaled)
            df['min_cluster_dist'] = distances.min(axis=1)
            df['mean_cluster_dist'] = distances.mean(axis=1)
    
    return df

# Apply feature engineering
train_fe = create_advanced_features(train, is_train=True)
test_fe = create_advanced_features(test, is_train=False)

# Ensure column consistency
train_cols = set(train_fe.columns) - {TARGET}
test_cols = set(test_fe.columns)

# Add missing columns to test
for col in train_cols - test_cols:
    test_fe[col] = 0

# Remove extra columns from test
for col in test_cols - train_cols:
    test_fe = test_fe.drop(columns=[col])

print(f"   Created {len(train_fe.columns) - len(train.columns)} new features")

# ================== Target Analysis ==================
print("\n[4/10] Analyzing target variable...")

y = train_fe[TARGET].astype(float)
X = train_fe.drop(columns=[TARGET])

# Remove NaN targets
mask = y.notna()
X = X.loc[mask].copy()
y = y.loc[mask].copy()

print(f"   Samples after removing NaN targets: {len(y)}")
print(f"   Target - Mean: {y.mean():.2f}, Std: {y.std():.2f}, Skew: {y.skew():.3f}")

# ================== Data Preparation ==================
print("\n[5/10] Preparing final datasets...")

X_test = test_fe[X.columns].copy()

# Handle categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"   Categorical columns: {len(cat_cols)}")

# Simple label encoding for categorical columns
for col in cat_cols:
    X[col] = X[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')
    
    # Create mapping
    unique_vals = sorted(list(set(X[col].unique()) | set(X_test[col].unique())))
    mapping = {val: i for i, val in enumerate(unique_vals)}
    
    X[col] = X[col].map(mapping)
    X_test[col] = X_test[col].map(mapping).fillna(-1).astype(int)

# Handle infinite and missing values
X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

# Fill missing values with median
for col in X.columns:
    if X[col].isna().any() or X_test[col].isna().any():
        median_val = X[col].median() if not X[col].isna().all() else 0
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# Convert to float32 for efficiency
X = X.astype(np.float32)
X_test = X_test.astype(np.float32)
y = y.astype(np.float32)

print(f"   Final shapes - X: {X.shape}, X_test: {X_test.shape}")

# ================== Model Definitions ==================
print("\n[6/10] Initializing models...")

def get_models(seed=42):
    """Get dictionary of models"""
    models = {
        'catboost': CatBoostRegressor(
            iterations=1500, depth=6, learning_rate=0.04,
            l2_leaf_reg=8, random_seed=seed, verbose=False,
            early_stopping_rounds=50
        ),
        'lightgbm': lgb.LGBMRegressor(
            n_estimators=1500, num_leaves=40, learning_rate=0.04,
            feature_fraction=0.7, bagging_fraction=0.7,
            min_child_samples=20, reg_alpha=0.5, reg_lambda=0.5,
            verbose=-1, seed=seed
        ),
        'xgboost': xgb.XGBRegressor(
            n_estimators=1500, max_depth=6, learning_rate=0.04,
            subsample=0.7, colsample_bytree=0.7,
            reg_alpha=0.5, reg_lambda=0.5,
            seed=seed
        ),
        'rf': RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt',
            random_state=seed, n_jobs=-1
        ),
        'et': ExtraTreesRegressor(
            n_estimators=300, max_depth=12, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt',
            random_state=seed, n_jobs=-1
        ),
        'ridge': Ridge(alpha=50.0, random_state=seed),
        'huber': HuberRegressor(epsilon=1.5, alpha=10.0, max_iter=200)
    }
    return models

# ================== Cross-Validation Training ==================
print("\n[7/10] Training models with cross-validation...")

models = get_models(RANDOM_SEED)
selected_models = ['catboost', 'lightgbm', 'xgboost', 'rf', 'ridge']

# Storage for predictions
oof_preds = {name: np.zeros(len(X)) for name in selected_models}
test_preds = {name: np.zeros(len(X_test)) for name in selected_models}
fold_scores = {name: [] for name in selected_models}

kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n   Fold {fold}/{N_SPLITS}:")
    
    X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[trn_idx], y.iloc[val_idx]
    
    # For XGBoost compatibility
    X_tr_reset = X_tr.reset_index(drop=True)
    X_va_reset = X_va.reset_index(drop=True)
    X_test_reset = X_test.reset_index(drop=True)
    
    for model_name in selected_models:
        model = get_models(RANDOM_SEED)[model_name]
        
        try:
            if model_name == 'catboost':
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
                val_pred = model.predict(X_va)
                test_pred = model.predict(X_test)
                
            elif model_name == 'lightgbm':
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                         callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
                val_pred = model.predict(X_va)
                test_pred = model.predict(X_test)
                
            elif model_name == 'xgboost':
                model.fit(X_tr_reset, y_tr, eval_set=[(X_va_reset, y_va)],
                         early_stopping_rounds=50, verbose=False)
                val_pred = model.predict(X_va_reset)
                test_pred = model.predict(X_test_reset)
                
            else:
                model.fit(X_tr, y_tr)
                val_pred = model.predict(X_va)
                test_pred = model.predict(X_test)
            
            # Store predictions
            oof_preds[model_name][val_idx] = val_pred
            test_preds[model_name] += test_pred / N_SPLITS
            
            # Calculate score
            score = mean_squared_error(y_va, val_pred, squared=False)
            fold_scores[model_name].append(score)
            print(f"      {model_name:10s}: {score:.3f}")
            
        except Exception as e:
            print(f"      {model_name:10s}: Error - {str(e)[:40]}")
            oof_preds[model_name][val_idx] = y_va.mean()
            test_preds[model_name] += y.mean() / N_SPLITS

# ================== Model Evaluation ==================
print("\n[8/10] Evaluating models...")

cv_scores = {}
for model_name in selected_models:
    cv_score = mean_squared_error(y, oof_preds[model_name], squared=False)
    cv_scores[model_name] = cv_score
    mean_fold = np.mean(fold_scores[model_name])
    print(f"   {model_name:10s}: CV={cv_score:.3f}, Avg_fold={mean_fold:.3f}")

# ================== Ensemble Creation ==================
print("\n[9/10] Creating ensemble...")

# Get valid models (not failed)
valid_models = [m for m in selected_models if cv_scores[m] < 500]

# 1. Simple average
oof_simple = np.mean([oof_preds[m] for m in valid_models], axis=0)
test_simple = np.mean([test_preds[m] for m in valid_models], axis=0)
score_simple = mean_squared_error(y, oof_simple, squared=False)

# 2. Weighted average (by CV score)
weights = np.array([1/cv_scores[m] for m in valid_models])
weights = weights / weights.sum()

oof_weighted = sum(w * oof_preds[m] for w, m in zip(weights, valid_models))
test_weighted = sum(w * test_preds[m] for w, m in zip(weights, valid_models))
score_weighted = mean_squared_error(y, oof_weighted, squared=False)

# 3. Top-3 models
top3 = sorted(cv_scores.items(), key=lambda x: x[1])[:3]
top3_models = [m[0] for m in top3]
oof_top3 = np.mean([oof_preds[m] for m in top3_models], axis=0)
test_top3 = np.mean([test_preds[m] for m in top3_models], axis=0)
score_top3 = mean_squared_error(y, oof_top3, squared=False)

print(f"\n   Ensemble scores:")
print(f"      Simple average: {score_simple:.3f}")
print(f"      Weighted:       {score_weighted:.3f}")
print(f"      Top-3:          {score_top3:.3f}")

# Select best ensemble
ensemble_scores = {
    'simple': (score_simple, test_simple),
    'weighted': (score_weighted, test_weighted),
    'top3': (score_top3, test_top3)
}

best_name = min(ensemble_scores.keys(), key=lambda x: ensemble_scores[x][0])
best_score, best_preds = ensemble_scores[best_name]

print(f"\n   Best ensemble: {best_name} (RMSE: {best_score:.3f})")

# ================== Create Submission ==================
print("\n[10/10] Creating submission...")

# Post-processing: clip to reasonable range
y_min = y.quantile(0.002)
y_max = y.quantile(0.998)
best_preds_clipped = np.clip(best_preds, y_min, y_max)

# Load submission template
spec = pd.read_csv(f"{dir0}/SPECIMEN.csv")
submission = spec.copy()
submission[TARGET] = best_preds_clipped

# Save
submission.to_csv("submission.csv", index=False)

print(f"\n   Submission saved!")
print(f"   Stats - Mean: {submission[TARGET].mean():.1f}, Std: {submission[TARGET].std():.1f}")
print(f"   Range: [{submission[TARGET].min():.1f}, {submission[TARGET].max():.1f}]")

print("\n" + "="*70)
print(f"COMPLETE! Expected LB score: ~{best_score:.0f} RMSE")
print("="*70)

print("\nFirst 10 predictions:")
print(submission.head(10))

