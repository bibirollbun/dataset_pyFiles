# Playground Series S5E10 - Road Accident Risk Prediction

# This notebook provides a comprehensive solution using ensemble modeling with XGBoost, LightGBM, and CatBoost.


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
import warnings
warnings.filterwarnings('ignore')
import os
from glob import glob

# Set random seed for reproducibility
SEED = 42
np.random.seed(SEED)


import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Load the datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e10/sample_submission.csv')

print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"\nTrain columns:\n{train.columns.tolist()}")
print(f"\nFirst few rows of train data:")
train.head()


# Basic statistics
print("Train data info:")
print(train.info())
print("\n" + "="*50)
print("\nTarget variable statistics:")
print(train['accident_risk'].describe())

# Check for missing values
print("\n" + "="*50)
print("\nMissing values in train:")
print(train.isnull().sum().sort_values(ascending=False).head(10))
print("\nMissing values in test:")
print(test.isnull().sum().sort_values(ascending=False).head(10))


# Visualize target distribution
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.hist(train['accident_risk'], bins=50, edgecolor='black')
plt.title('Distribution of Accident Risk')
plt.xlabel('Accident Risk')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.boxplot(train['accident_risk'])
plt.title('Boxplot of Accident Risk')
plt.ylabel('Accident Risk')

plt.tight_layout()
plt.show()


def feature_engineering(df):
    """Create additional features from existing data"""
    df = df.copy()
    
    # Identify categorical and numerical columns
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove id and target from feature lists
    if 'id' in num_cols:
        num_cols.remove('id')
    if 'accident_risk' in num_cols:
        num_cols.remove('accident_risk')
    
    # Statistical features for numerical columns
    if len(num_cols) > 1:
        df['num_mean'] = df[num_cols].mean(axis=1)
        df['num_std'] = df[num_cols].std(axis=1)
        df['num_min'] = df[num_cols].min(axis=1)
        df['num_max'] = df[num_cols].max(axis=1)
        df['num_range'] = df['num_max'] - df['num_min']
    
    # Interactions (robust, low-cost)
    if all(c in df.columns for c in ['num_lanes','speed_limit']):
        df['lanes_speed'] = df['num_lanes'] * df['speed_limit']
        df['speed_per_lane'] = df['speed_limit'] / (df['num_lanes'].replace(0, np.nan))
        df['speed_per_lane'] = df['speed_per_lane'].fillna(0)
    if all(c in df.columns for c in ['curvature','speed_limit']):
        df['curve_speed'] = df['curvature'] * df['speed_limit']
    if all(c in df.columns for c in ['num_reported_accidents','num_lanes']):
        df['accidents_per_lane'] = df['num_reported_accidents'] / (df['num_lanes'].replace(0, np.nan))
        df['accidents_per_lane'] = df['accidents_per_lane'].fillna(0)
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in cat_cols:
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
    
    # Frequency encoding for categorical variables
    for col in cat_cols:
        freq = df[col].value_counts(normalize=True)
        df[f'{col}_freq'] = df[col].map(freq).astype(float)
    
    return df

# Apply feature engineering
train_fe = feature_engineering(train)
test_fe = feature_engineering(test)

print(f"Train shape after feature engineering: {train_fe.shape}")
print(f"Test shape after feature engineering: {test_fe.shape}")


# Prepare features and target
target = 'accident_risk'

# Exclude original categorical columns to avoid non-numeric median issues
cat_cols = train.select_dtypes(include=['object']).columns.tolist()
drop_cols = set(['id', target] + cat_cols)
features = [col for col in train_fe.columns if col not in drop_cols]

X = train_fe[features]
y = train_fe[target]
X_test = test_fe[features]

# Handle any remaining missing values (numeric only now)
X = X.fillna(X.median())
X_test = X_test.fillna(X_test.median())

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Test features shape: {X_test.shape}")
print(f"Dropped categorical columns: {len(cat_cols)}")


# Define cross-validation strategy (Stratified by target bins for stability)
n_folds = 10
y_bins = pd.qcut(y, q=20, labels=False, duplicates='drop')
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)

# Storage for predictions
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

# Track scores
cv_scores = []

print("Training XGBoost with Cross-Validation...")


# XGBoost Model
# Ensure required matrices exist (handles out-of-order execution)
if 'X' not in globals() or 'y' not in globals() or 'X_test' not in globals():
    assert 'train_fe' in globals() and 'test_fe' in globals(), "Please run the feature engineering cell first."
    target = 'accident_risk'
    # Exclude original categorical columns here as well
    cat_cols = train.select_dtypes(include=['object']).columns.tolist()
    drop_cols = set(['id', target] + cat_cols)
    features = [c for c in train_fe.columns if c not in drop_cols]
    X = train_fe[features].copy()
    y = train_fe[target].copy()
    X_test = test_fe[features].copy()
    X = X.fillna(X.median())
    X_test = X_test.fillna(X_test.median())

xgb_test_preds = np.zeros(len(X_test))

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_bins)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # XGBoost (slightly slower LR, more rounds)
    xgb_model = xgb.XGBRegressor(
        n_estimators=1500,
        learning_rate=0.035,
        max_depth=7,
        min_child_weight=2,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.2,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=SEED,
        tree_method='hist',
        early_stopping_rounds=100
    )
    
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    # Predictions
    val_preds = xgb_model.predict(X_val)
    oof_preds[val_idx] = val_preds
    xgb_test_preds += xgb_model.predict(X_test) / n_folds
    
    # Score
    fold_score = np.sqrt(mean_squared_error(y_val, val_preds))
    cv_scores.append(fold_score)
    print(f"Fold {fold + 1} RMSE: {fold_score:.5f}")

# Overall CV score
overall_cv = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\n{'='*50}")
print(f"XGBoost Overall CV RMSE: {overall_cv:.5f}")
print(f"XGBoost Mean CV RMSE: {np.mean(cv_scores):.5f} (+/- {np.std(cv_scores):.5f})")


# LightGBM Model
print("\n" + "="*50)
print("Training LightGBM with Cross-Validation...")

lgb_oof_preds = np.zeros(len(X))
lgb_test_preds = np.zeros(len(X_test))
lgb_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_bins)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # LightGBM (tuned a bit stronger)
    lgb_model = lgb.LGBMRegressor(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=7,
        num_leaves=63,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.2,
        reg_lambda=1.5,
        random_state=SEED,
        verbose=-1
    )
    
    lgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
    )
    
    # Predictions
    val_preds = lgb_model.predict(X_val)
    lgb_oof_preds[val_idx] = val_preds
    lgb_test_preds += lgb_model.predict(X_test) / n_folds
    
    # Score
    fold_score = np.sqrt(mean_squared_error(y_val, val_preds))
    lgb_cv_scores.append(fold_score)
    print(f"Fold {fold + 1} RMSE: {fold_score:.5f}")

# Overall CV score
lgb_overall_cv = np.sqrt(mean_squared_error(y, lgb_oof_preds))
print(f"\n{'='*50}")
print(f"LightGBM Overall CV RMSE: {lgb_overall_cv:.5f}")
print(f"LightGBM Mean CV RMSE: {np.mean(lgb_cv_scores):.5f} (+/- {np.std(lgb_cv_scores):.5f})")


# CatBoost Model
print("\n" + "="*50)
print("Training CatBoost with Cross-Validation...")

cb_oof_preds = np.zeros(len(X))
cb_test_preds = np.zeros(len(X_test))
cb_cv_scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y_bins)):
    print(f"\nFold {fold + 1}/{n_folds}")
    
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    # CatBoost (slightly deeper, stronger L2)
    cb_model = cb.CatBoostRegressor(
        iterations=2000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=6,
        subsample=0.8,
        random_state=SEED,
        verbose=False,
        early_stopping_rounds=100
    )
    
    cb_model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False
    )
    
    # Predictions
    val_preds = cb_model.predict(X_val)
    cb_oof_preds[val_idx] = val_preds
    cb_test_preds += cb_model.predict(X_test) / n_folds
    
    # Score
    fold_score = np.sqrt(mean_squared_error(y_val, val_preds))
    cb_cv_scores.append(fold_score)
    print(f"Fold {fold + 1} RMSE: {fold_score:.5f}")

# Overall CV score
cb_overall_cv = np.sqrt(mean_squared_error(y, cb_oof_preds))
print(f"\n{'='*50}")
print(f"CatBoost Overall CV RMSE: {cb_overall_cv:.5f}")
print(f"CatBoost Mean CV RMSE: {np.mean(cb_cv_scores):.5f} (+/- {np.std(cb_cv_scores):.5f})")


# Optimized weighted ensemble across 3 models (XGB, LGB, CAT) with fallback if scipy not available
import numpy as np
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def normalize(weights):
    w = np.abs(np.array(weights, dtype=float))
    s = w.sum()
    return (w / s) if s > 0 else np.array([1/3, 1/3, 1/3])

def eval_weights(w):
    w = normalize(w)
    pred = w[0]*oof_preds + w[1]*lgb_oof_preds + w[2]*cb_oof_preds
    return rmse(y, pred)

best_w = np.array([1/3, 1/3, 1/3])
best_score = eval_weights(best_w)

try:
    from scipy.optimize import minimize
    res = minimize(lambda w: eval_weights(w), x0=np.array([0.2, 0.4, 0.4]), method='Nelder-Mead', options={'maxiter': 1500})
    w_opt = normalize(res.x)
    score_opt = eval_weights(w_opt)
    if score_opt < best_score:
        best_w, best_score = w_opt, score_opt
except Exception as _:
    # Random simplex fallback
    rng = np.random.default_rng(SEED)
    for _ in range(500):
        w = rng.random(3); w = w / w.sum()
        s = eval_weights(w)
        if s < best_score:
            best_w, best_score = w, s

ensemble_weights = {
    'xgb': float(best_w[0]),
    'lgb': float(best_w[1]),
    'cb':  float(best_w[2])
}

ensemble_oof = (
    ensemble_weights['xgb'] * oof_preds +
    ensemble_weights['lgb'] * lgb_oof_preds +
    ensemble_weights['cb']  * cb_oof_preds
)

ensemble_test = (
    ensemble_weights['xgb'] * xgb_test_preds +
    ensemble_weights['lgb'] * lgb_test_preds +
    ensemble_weights['cb']  * cb_test_preds
)

ensemble_cv = rmse(y, ensemble_oof)

print("="*50)
print("FINAL MODEL COMPARISON")
print("="*50)
print(f"XGBoost CV RMSE:     {overall_cv:.5f}")
print(f"LightGBM CV RMSE:    {lgb_overall_cv:.5f}")
print(f"CatBoost CV RMSE:    {cb_overall_cv:.5f}")
print(f"Ensemble CV RMSE:    {ensemble_cv:.5f}")
print("-"*50)
print("Optimal Weights:")
print(f"  XGB: {ensemble_weights['xgb']:.4f}")
print(f"  LGB: {ensemble_weights['lgb']:.4f}")
print(f"  CAT: {ensemble_weights['cb']:.4f}")
print("="*50)


# Stacking (NNLS) + Rank-averaged blend for 3 models
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))

# Base OOF and TEST matrices (XGB, LGB, CAT)
A_oof = np.vstack([oof_preds, lgb_oof_preds, cb_oof_preds]).T
A_tst = np.vstack([xgb_test_preds, lgb_test_preds, cb_test_preds]).T

# 1) NNLS weights (non-negative least squares)
try:
    from scipy.optimize import nnls
    w_nnls, _ = nnls(A_oof, y.values if hasattr(y,'values') else np.array(y))
    w_nnls = w_nnls / (w_nnls.sum() + 1e-12)
except Exception:
    w_nnls = np.array([1/3,1/3,1/3])

ens_oof_nnls = A_oof.dot(w_nnls)
ens_tst_nnls = A_tst.dot(w_nnls)
cv_nnls = rmse(y, ens_oof_nnls)

# 2) Rank-average ensemble (scale-free)
def rank01(v):
    s = pd.Series(v)
    return (s.rank(method='average').values - 1) / (len(s)-1)

r_oof = np.vstack([rank01(oof_preds), rank01(lgb_oof_preds), rank01(cb_oof_preds)]).T
r_tst = np.vstack([
    rank01(xgb_test_preds),
    rank01(lgb_test_preds),
    rank01(cb_test_preds)
]).T
ens_oof_rank = r_oof.mean(axis=1)
ens_tst_rank = r_tst.mean(axis=1)
cv_rank = rmse(y, ens_oof_rank)

# 3) Compare with previous optimized weighted ensemble
prev_oof = ensemble_oof
prev_tst = ensemble_test
cv_prev = rmse(y, prev_oof)

candidates = {
    'prev': (prev_oof, prev_tst, cv_prev),
    'nnls': (ens_oof_nnls, ens_tst_nnls, cv_nnls),
    'rank': (ens_oof_rank, ens_tst_rank, cv_rank)
}

# Line search blend between best pair
labels = list(candidates.keys())
best_key = min(labels, key=lambda k: candidates[k][2])
best_oof, best_tst, best_cv = candidates[best_key]

for a_key in labels:
    for b_key in labels:
        if a_key == b_key: continue
        oof_a, tst_a, cv_a = candidates[a_key]
        oof_b, tst_b, cv_b = candidates[b_key]
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            mix_oof = alpha*oof_a + (1-alpha)*oof_b
            mix_cv = rmse(y, mix_oof)
            if mix_cv < best_cv - 1e-7:
                best_cv = mix_cv
                best_oof = mix_oof
                best_tst = alpha*tst_a + (1-alpha)*tst_b
                best_key = f"blend({a_key},{b_key},{alpha:.2f})"

best_ensemble_oof = best_oof
best_ensemble_test = best_tst
print("Ensemble CVs:")
print(f"  prev: {cv_prev:.5f}")
print(f"  nnls: {cv_nnls:.5f}")
print(f"  rank: {cv_rank:.5f}")
print(f"Best: {best_key} -> {best_cv:.5f}")


# Calibration + shrinkage on best ensemble to squeeze extra CV
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

def rmse(a,b):
    return np.sqrt(mean_squared_error(a,b))

assert 'best_ensemble_oof' in globals() and 'best_ensemble_test' in globals(), "Run the stacking/ensemble cells first."

y_true = y.values if hasattr(y,'values') else np.array(y)
base_oof = best_ensemble_oof.copy()
base_tst = best_ensemble_test.copy()
base_cv = rmse(y_true, base_oof)
gmean = float(np.mean(y_true))

best_oof_cal = base_oof
best_tst_cal = base_tst
best_cv_cal = base_cv
best_tag = 'base'

# 1) Global shrinkage towards mean
for alpha in np.linspace(0.0, 0.3, 7):  # 0.00 .. 0.30
    oof_s = (1.0 - alpha) * base_oof + alpha * gmean
    cv_s = rmse(y_true, oof_s)
    if cv_s + 1e-9 < best_cv_cal:
        best_cv_cal = cv_s
        best_oof_cal = oof_s
        best_tst_cal = (1.0 - alpha) * base_tst + alpha * gmean
        best_tag = f'shrink(alpha={alpha:.2f})'

# 2) Isotonic calibration (monotonic, clamp to [0,1])
try:
    from sklearn.isotonic import IsotonicRegression
    ir = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds='clip')
    ir.fit(base_oof, y_true)
    oof_iso = ir.predict(base_oof)
    tst_iso = ir.predict(base_tst)
    cv_iso = rmse(y_true, oof_iso)
    if cv_iso + 1e-9 < best_cv_cal:
        best_cv_cal = cv_iso
        best_oof_cal = oof_iso
        best_tst_cal = tst_iso
        best_tag = 'isotonic'

    # 2b) Blend isotonic with base
    for beta in [0.25, 0.50, 0.75]:
        oof_b = beta * oof_iso + (1.0 - beta) * base_oof
        cv_b = rmse(y_true, oof_b)
        if cv_b + 1e-9 < best_cv_cal:
            best_cv_cal = cv_b
            best_oof_cal = oof_b
            best_tst_cal = beta * tst_iso + (1.0 - beta) * base_tst
            best_tag = f'isotonic_blend(beta={beta:.2f})'
except Exception:
    pass

# 3) Final clamp and expose best calibrated ensemble
best_ensemble_oof = np.clip(best_oof_cal, 0.0, 1.0)
best_ensemble_test = np.clip(best_tst_cal, 0.0, 1.0)
best_cv = rmse(y_true, best_ensemble_oof)

print("Calibration summary:")
print(f"  Base CV:      {base_cv:.5f}")
print(f"  Best tag:     {best_tag}")
print(f"  Best Cal CV:  {best_cv:.5f}")


# Advanced calibration sweep: quadratic, quantile mapping, extended shrink, clipping
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression

def rmse(a, b):
    return np.sqrt(mean_squared_error(a, b))

assert 'best_ensemble_oof' in globals() and 'best_ensemble_test' in globals(), 'Run ensemble cells first.'

y_true = y.values if hasattr(y, 'values') else np.array(y)
base_oof = np.asarray(best_ensemble_oof).copy()
base_tst = np.asarray(best_ensemble_test).copy()
base_cv = rmse(y_true, base_oof)
gmean = float(np.mean(y_true))

best_oof = base_oof
best_tst = base_tst
best_cv_pp = base_cv
best_tag = 'base'

# 1) Extended global shrinkage to mean
for alpha in np.linspace(0.0, 0.5, 11):  # 0.00..0.50
    oof_s = (1.0 - alpha) * base_oof + alpha * gmean
    tst_s = (1.0 - alpha) * base_tst + alpha * gmean
    cv_s = rmse(y_true, oof_s)
    if cv_s + 1e-9 < best_cv_pp:
        best_cv_pp = cv_s; best_oof = oof_s; best_tst = tst_s; best_tag = f'shrink(alpha={alpha:.2f})'

# 2) Quadratic regression calibration: y ~ a*p + b*p^2 + c
try:
    P = base_oof.reshape(-1, 1)
    Xq = np.column_stack([P[:,0], P[:,0]**2, np.ones_like(P[:,0])])
    lr = LinearRegression(fit_intercept=False)
    lr.fit(Xq, y_true)
    oof_q = lr.predict(Xq)
    tst_q = lr.predict(np.column_stack([base_tst, base_tst**2, np.ones_like(base_tst)]))
    cv_q = rmse(y_true, oof_q)
    if cv_q + 1e-9 < best_cv_pp:
        best_cv_pp = cv_q; best_oof = oof_q; best_tst = tst_q; best_tag = 'quadratic'
    # Blends
    for beta in (0.25, 0.50, 0.75):
        oof_b = beta * oof_q + (1.0 - beta) * base_oof
        tst_b = beta * tst_q + (1.0 - beta) * base_tst
        cv_b = rmse(y_true, oof_b)
        if cv_b + 1e-9 < best_cv_pp:
            best_cv_pp = cv_b; best_oof = oof_b; best_tst = tst_b; best_tag = f'quad_blend(beta={beta:.2f})'
except Exception:
    pass

# 3) Quantile mapping (QMAP) calibration
try:
    q = np.linspace(0.0, 1.0, 101)
    p_q = np.quantile(base_oof, q)
    y_q = np.quantile(y_true, q)
    # Ensure monotonic p_q
    p_q = np.maximum.accumulate(p_q)
    oof_qmap = np.interp(base_oof, p_q, y_q)
    tst_qmap = np.interp(base_tst, p_q, y_q)
    cv_qmap = rmse(y_true, oof_qmap)
    if cv_qmap + 1e-9 < best_cv_pp:
        best_cv_pp = cv_qmap; best_oof = oof_qmap; best_tst = tst_qmap; best_tag = 'qmap'
    # Blends
    for gamma in (0.25, 0.50, 0.75):
        oof_b = gamma * oof_qmap + (1.0 - gamma) * base_oof
        tst_b = gamma * tst_qmap + (1.0 - gamma) * base_tst
        cv_b = rmse(y_true, oof_b)
        if cv_b + 1e-9 < best_cv_pp:
            best_cv_pp = cv_b; best_oof = oof_b; best_tst = tst_b; best_tag = f'qmap_blend(gamma={gamma:.2f})'
except Exception:
    pass

# 4) Percentile clipping sweep (upper cap improves RMSE outliers)
for up in (0.995, 0.992, 0.99, 0.98, 0.975):
    cap = float(np.quantile(best_oof, up))
    oof_c = np.clip(best_oof, 0.0, cap)
    tst_c = np.clip(best_tst, 0.0, cap)
    cv_c = rmse(y_true, oof_c)
    if cv_c + 1e-9 < best_cv_pp:
        best_cv_pp = cv_c; best_oof = oof_c; best_tst = tst_c; best_tag = f'clip(p{int(up*1000)/10:.1f})'

# Final clamp to [0,1] and expose
best_ensemble_oof = np.clip(best_oof, 0.0, 1.0)
best_ensemble_test = np.clip(best_tst, 0.0, 1.0)
best_cv = rmse(y_true, best_ensemble_oof)

print('Post-processing calibration summary:')
print(f'  Started CV: {base_cv:.5f}')
print(f'  Best tag:   {best_tag}')
print(f'  Final CV:   {best_cv:.5f}')


# Clip predictions to valid range [0, 1] and save best ensemble
use_test = best_ensemble_test if 'best_ensemble_test' in globals() else ensemble_test
use_cv = best_cv if 'best_cv' in globals() else ensemble_cv
use_test = np.clip(use_test, 0, 1)

# Create submission file
submission = pd.DataFrame({
    'id': test_fe['id'],
    'accident_risk': use_test
})

# Save default submission
submission.to_csv('submission.csv', index=False)

# Also save a traceable file with CV and (if available) current weights
xw = ensemble_weights['xgb'] if 'ensemble_weights' in globals() else 0.0
lw = ensemble_weights['lgb'] if 'ensemble_weights' in globals() else 0.0
cw = ensemble_weights['cb']  if 'ensemble_weights' in globals() else 0.0
fname = f"submission_best_cv{use_cv:.5f}_x{xw:.3f}_l{lw:.3f}_c{cw:.3f}.csv"
submission.to_csv(fname, index=False)

# Predict LB using observed CV→LB gap (~0.00044 better on LB)
pred_lb = use_cv - 0.00044
print("Submission files created successfully!")
print(f"Saved: submission.csv and {fname}")
print(f"Predicted Public LB: ~{pred_lb:.5f}")

# Quick recommendation
if pred_lb < 0.05550:
    print("Recommendation: SUBMIT (likely < 0.05550)")
elif pred_lb <= 0.05580:
    print("Recommendation: SUBMIT (near/better than previous best)")
else:
    print("Recommendation: Hold (not clearly better)")

# Basic stats
print(f"\nSubmission shape: {submission.shape}")
if 'ensemble_weights' in globals():
    print("Weights:", {k: round(v, 4) for k, v in ensemble_weights.items()})
print(f"Best Ensemble CV: {use_cv:.5f}")
print("\nPrediction statistics:")
print(submission['accident_risk'].describe())

