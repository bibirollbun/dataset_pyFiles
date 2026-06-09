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


# Install required packages (uncomment if needed)
!pip install catboost lightgbm xgboost scikit-learn scipy numpy pandas -q

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Linear Models
from sklearn.linear_model import (
    Ridge, Lasso, ElasticNet, BayesianRidge, 
    HuberRegressor, Lars, LassoLars,
    PassiveAggressiveRegressor, RANSACRegressor
)

# Tree-based Models
from sklearn.ensemble import (
    RandomForestRegressor, ExtraTreesRegressor, 
    GradientBoostingRegressor, AdaBoostRegressor, 
    BaggingRegressor, HistGradientBoostingRegressor
)

# Boosting Libraries
from catboost import CatBoostRegressor
import lightgbm as lgb
import xgboost as xgb

# Neural Networks
from sklearn.neural_network import MLPRegressor

# Other Models
from sklearn.svm import SVR, LinearSVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge

# Optimization
from scipy.optimize import minimize

# Configuration
RANDOM_SEED = 42
N_SPLITS = 5
TARGET = "CORRUCYSTIC_DENSITY"
ID_COL = "LOCAL_IDENTIFIER"
dir0 = '/kaggle/input/recruitment-task-for-gdsc-ml'

np.random.seed(RANDOM_SEED)

print("="*80)
print("GDSC ML COMPETITION - COMPLETE MODEL ZOO SOLUTION")
print("="*80)

# ================== Data Loading ==================
print("\n[1/10] Loading data...")
train = pd.read_csv(f"{dir0}/MiNDAT.csv")
test = pd.read_csv(f"{dir0}/MiNDAT_UNK.csv")

if ID_COL in train.columns:
    train = train.set_index(ID_COL)
if ID_COL in test.columns:
    test = test.set_index(ID_COL)

print(f"   Train: {train.shape}, Test: {test.shape}")

# ================== Feature Engineering ==================
print("\n[2/10] Engineering features...")

def create_features(df, is_train=True):
    df = df.copy()
    
    # Clean column names - simple approach
    new_cols = {}
    for i, col in enumerate(df.columns):
        if col == TARGET:
            new_cols[col] = TARGET
        else:
            new_cols[col] = f'feat_{i:03d}'
    df.columns = [new_cols[col] for col in df.columns]
    
    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if is_train and TARGET in numeric_cols:
        numeric_cols.remove(TARGET)
    
    if len(numeric_cols) > 0:
        # Basic statistics
        df['row_mean'] = df[numeric_cols].mean(axis=1)
        df['row_std'] = df[numeric_cols].std(axis=1)
        df['row_max'] = df[numeric_cols].max(axis=1)
        df['row_min'] = df[numeric_cols].min(axis=1)
        df['row_median'] = df[numeric_cols].median(axis=1)
        df['row_skew'] = df[numeric_cols].skew(axis=1)
        df['row_kurt'] = df[numeric_cols].kurtosis(axis=1)
        df['row_q25'] = df[numeric_cols].quantile(0.25, axis=1)
        df['row_q75'] = df[numeric_cols].quantile(0.75, axis=1)
        df['row_iqr'] = df['row_q75'] - df['row_q25']
        df['row_range'] = df['row_max'] - df['row_min']
        
        # Additional features
        df['row_sum'] = df[numeric_cols].sum(axis=1)
        df['row_nunique'] = df[numeric_cols].nunique(axis=1)
        df['row_count_zeros'] = (df[numeric_cols] == 0).sum(axis=1)
        df['row_count_pos'] = (df[numeric_cols] > 0).sum(axis=1)
        df['row_count_neg'] = (df[numeric_cols] < 0).sum(axis=1)
        df['row_count_nan'] = df[numeric_cols].isna().sum(axis=1)
        
        # Coefficient of variation
        df['row_cv'] = df['row_std'] / (np.abs(df['row_mean']) + 1e-8)
        
        # Variance
        df['row_var'] = df[numeric_cols].var(axis=1)
        
        # Polynomial features for high variance columns
        variances = df[numeric_cols].var().fillna(0)
        top_cols = variances.nlargest(min(8, len(variances))).index.tolist()
        
        for i, col in enumerate(top_cols[:5]):
            if col in df.columns:
                df[f'poly_{i}_sq'] = df[col] ** 2
                df[f'poly_{i}_sqrt'] = np.sqrt(np.abs(df[col].fillna(0)))
                df[f'poly_{i}_log'] = np.log1p(np.abs(df[col].fillna(0)))
                df[f'poly_{i}_inv'] = 1 / (np.abs(df[col].fillna(1)) + 1)
        
        # Interaction features
        for i in range(min(3, len(top_cols))):
            for j in range(i+1, min(4, len(top_cols))):
                col1, col2 = top_cols[i], top_cols[j]
                df[f'inter_{i}_{j}_mult'] = df[col1].fillna(0) * df[col2].fillna(0)
                df[f'inter_{i}_{j}_add'] = df[col1].fillna(0) + df[col2].fillna(0)
                df[f'inter_{i}_{j}_sub'] = np.abs(df[col1].fillna(0) - df[col2].fillna(0))
                
                # Safe division
                denominator = df[col2].fillna(1).replace(0, 1)
                df[f'inter_{i}_{j}_div'] = df[col1].fillna(0) / denominator
    
    return df

train = create_features(train, is_train=True)
test = create_features(test, is_train=False)

# Ensure column consistency
train_cols = set(train.columns) - {TARGET}
test_cols = set(test.columns)

for col in train_cols - test_cols:
    test[col] = 0

for col in test_cols - train_cols:
    test = test.drop(columns=[col])

print(f"   Created {len(train.columns) - 47} new features")

# ================== Data Preparation ==================
print("\n[3/10] Preparing data...")

y = train[TARGET].astype(float)
X = train.drop(columns=[TARGET])

# Remove NaN targets
mask = y.notna()
X = X.loc[mask].copy()
y = y.loc[mask].copy()

X_test = test[X.columns].copy()

# Handle categorical columns
cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
for col in cat_cols:
    X[col] = X[col].astype(str).fillna('missing')
    X_test[col] = X_test[col].astype(str).fillna('missing')
    
    unique_vals = sorted(list(set(X[col]) | set(X_test[col])))
    mapping = {val: i for i, val in enumerate(unique_vals)}
    
    X[col] = X[col].map(mapping)
    X_test[col] = X_test[col].map(mapping).fillna(-1).astype(int)

# Handle infinite and missing values
X = X.replace([np.inf, -np.inf], np.nan)
X_test = X_test.replace([np.inf, -np.inf], np.nan)

for col in X.columns:
    if X[col].isna().any() or X_test[col].isna().any():
        median_val = X[col].median() if not X[col].isna().all() else 0
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

# Convert to float32 for memory efficiency
X = X.astype(np.float32)
X_test = X_test.astype(np.float32)
y = y.astype(np.float32)

print(f"   Final shape: X={X.shape}, X_test={X_test.shape}")
print(f"   Target samples: {len(y)}")

# ================== Model Definitions ==================
print("\n[4/10] Initializing model zoo...")

def get_models(seed=42):
    """Initialize all models"""
    return {
        # Linear models
        'ridge': Ridge(alpha=20.0, random_state=seed),
        'lasso': Lasso(alpha=0.01, random_state=seed, max_iter=1000),
        'elastic': ElasticNet(alpha=0.01, l1_ratio=0.5, random_state=seed, max_iter=1000),
        'bayesian_ridge': BayesianRidge(),
        'huber': HuberRegressor(epsilon=1.35, max_iter=100),
        'lars': Lars(n_nonzero_coefs=50),
        'lasso_lars': LassoLars(alpha=0.01, max_iter=100),
        
        # Tree-based models
        'rf': RandomForestRegressor(
            n_estimators=100, max_depth=12, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt',
            random_state=seed, n_jobs=-1
        ),
        'et': ExtraTreesRegressor(
            n_estimators=100, max_depth=12, min_samples_split=10,
            min_samples_leaf=5, max_features='sqrt',
            random_state=seed, n_jobs=-1
        ),
        'gb': GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            subsample=0.8, random_state=seed
        ),
        'hist_gb': HistGradientBoostingRegressor(
            max_iter=100, learning_rate=0.1, random_state=seed
        ),
        
        # Boosting models
        'catboost': CatBoostRegressor(
            iterations=500, depth=6, learning_rate=0.05,
            l2_leaf_reg=5, random_seed=seed, verbose=False
        ),
        'lightgbm': lgb.LGBMRegressor(
            n_estimators=500, num_leaves=31, learning_rate=0.05,
            feature_fraction=0.8, bagging_fraction=0.8,
            min_child_samples=20, reg_alpha=0.5, reg_lambda=0.5,
            verbose=-1, random_seed=seed
        ),
        'xgboost': xgb.XGBRegressor(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.5, reg_lambda=0.5,
            random_state=seed
        ),
        
        # Other models
        'knn': KNeighborsRegressor(n_neighbors=15, weights='distance', n_jobs=-1),
        'kernel_ridge': KernelRidge(alpha=1.0, kernel='rbf'),
        
        # Neural network
        'mlp': MLPRegressor(
            hidden_layer_sizes=(100, 50), activation='relu',
            solver='adam', alpha=0.01, batch_size=32,
            learning_rate_init=0.001, max_iter=200,
            random_state=seed, early_stopping=True
        ),
    }

# Select models to use
selected_models = [
    'catboost', 'lightgbm', 'xgboost',
    'rf', 'et', 'hist_gb',
    'ridge', 'huber', 'bayesian_ridge',
    'knn'
]

print(f"   Selected {len(selected_models)} models")

# ================== Cross-Validation Training ==================
print("\n[5/10] Training models with {}-fold CV...".format(N_SPLITS))

models = get_models(RANDOM_SEED)
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)

# Storage for predictions
oof_preds = {name: np.zeros(len(X)) for name in selected_models}
test_preds = {name: np.zeros(len(X_test)) for name in selected_models}
cv_scores = {name: [] for name in selected_models}

for fold, (trn_idx, val_idx) in enumerate(kf.split(X, y), 1):
    print(f"\n   Fold {fold}/{N_SPLITS}:")
    
    X_tr, X_va = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_va = y.iloc[trn_idx], y.iloc[val_idx]
    
    # Reset index for XGBoost
    X_tr_reset = X_tr.reset_index(drop=True)
    X_va_reset = X_va.reset_index(drop=True)
    X_test_reset = X_test.reset_index(drop=True)
    
    for model_name in selected_models:
        try:
            model = get_models(RANDOM_SEED)[model_name]
            
            if model_name == 'catboost':
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                         early_stopping_rounds=50, verbose=False)
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
            
            oof_preds[model_name][val_idx] = val_pred
            test_preds[model_name] += test_pred / N_SPLITS
            
            score = mean_squared_error(y_va, val_pred, squared=False)
            cv_scores[model_name].append(score)
            print(f"      {model_name:15s}: {score:.3f}")
            
        except Exception as e:
            print(f"      {model_name:15s}: Error - {str(e)[:40]}")
            oof_preds[model_name][val_idx] = y_va.mean()
            test_preds[model_name] += y.mean() / N_SPLITS

# ================== Model Evaluation ==================
print("\n[6/10] Evaluating models...")

model_scores = {}
for model_name in selected_models:
    cv_score = mean_squared_error(y, oof_preds[model_name], squared=False)
    model_scores[model_name] = cv_score

print("\n   CV Scores (sorted):")
for model, score in sorted(model_scores.items(), key=lambda x: x[1]):
    avg_fold = np.mean(cv_scores[model])
    print(f"      {model:15s}: CV={score:.3f}, Avg_fold={avg_fold:.3f}")

# ================== Stacking ==================
print("\n[7/10] Creating stacked models...")

# Get top models
top_models = sorted(model_scores.items(), key=lambda x: x[1])[:6]
top_model_names = [m[0] for m in top_models]

# Create stacking features
stack_train = np.column_stack([oof_preds[m] for m in top_model_names])
stack_test = np.column_stack([test_preds[m] for m in top_model_names])

# Train meta-models
meta_models = {
    'meta_ridge': Ridge(alpha=1.0),
    'meta_huber': HuberRegressor(),
}

stacked_preds = {}
for meta_name, meta_model in meta_models.items():
    meta_model.fit(stack_train, y)
    stacked_preds[meta_name] = meta_model.predict(stack_test)
    
    oof_meta = meta_model.predict(stack_train)
    score = mean_squared_error(y, oof_meta, squared=False)
    print(f"   {meta_name}: {score:.3f}")

# ================== Ensemble Creation ==================
print("\n[8/10] Creating ensembles...")

# 1. Simple average
ensemble_simple = np.mean([test_preds[m] for m in top_model_names[:3]], axis=0)

# 2. Weighted average
weights = np.array([1/model_scores[m] for m in top_model_names[:3]])
weights = weights / weights.sum()
ensemble_weighted = sum(w * test_preds[m] for w, m in zip(weights, top_model_names[:3]))

# 3. All models average
ensemble_all = np.mean([test_preds[m] for m in selected_models 
                        if model_scores[m] < 500], axis=0)

print("   Created 3 ensemble strategies")

# ================== Optimization ==================
print("\n[9/10] Optimizing weights...")

def rmse_objective(weights, predictions, y_true):
    blend = np.zeros(len(y_true))
    for w, pred in zip(weights, predictions):
        blend += w * pred
    return mean_squared_error(y_true, blend, squared=False)

# Optimize for top 3 models
top3_oof = [oof_preds[m] for m in top_model_names[:3]]
initial_weights = np.ones(3) / 3

result = minimize(
    lambda w: rmse_objective(w / w.sum(), top3_oof, y),
    initial_weights,
    method='Nelder-Mead'
)

optimal_weights = result.x / result.x.sum()
ensemble_optimized = sum(w * test_preds[m] for w, m in zip(optimal_weights, top_model_names[:3]))

print(f"   Optimal weights: {[f'{w:.3f}' for w in optimal_weights]}")

# ================== Final Submission ==================
print("\n[10/10] Creating submission...")

# Use optimized ensemble
final_predictions = ensemble_optimized

# Clip to reasonable range
y_min = y.quantile(0.002)
y_max = y.quantile(0.998)
final_predictions = np.clip(final_predictions, y_min, y_max)

# Create submission
spec = pd.read_csv(f"{dir0}/SPECIMEN.csv")
submission = spec.copy()
submission[TARGET] = final_predictions

# Save
submission.to_csv("submission.csv", index=False)

print(f"\n   Submission saved!")
print(f"   Stats - Mean: {submission[TARGET].mean():.1f}, Std: {submission[TARGET].std():.1f}")
print(f"   Range: [{submission[TARGET].min():.1f}, {submission[TARGET].max():.1f}]")

print("\n" + "="*80)
print("COMPLETE! Submission ready for upload.")
print("="*80)

print("\nFirst 10 predictions:")
print(submission.head(10))

