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


import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesRegressor
import scipy.stats

# ===========================
# CONFIG & UTILS
# ===========================
SEED = 42
FOLDS = 7
MODEL_NAME = "ExtraTrees"
np.random.seed(SEED)

def timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def title_prefix(cv_rmse: float):
    return f"{timestamp()}_{MODEL_NAME}_CV{cv_rmse:.6f}"

# ===========================
# LOAD DATA
# ===========================
print("Loading data...")
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
test["accident_risk"] = 0.5

# Load synthetic data
orig = []
for k in [2, 10, 100]:
    df = pd.read_csv(f"/kaggle/input/simulated-roads-accident-data/synthetic_road_accidents_{k}k.csv")
    orig.append(df)
orig = pd.concat(orig, axis=0)
orig["id"] = np.arange(len(orig)) + test["id"].max() + 1
orig = orig[train.columns]

print(f"Train: {train.shape}, Test: {test.shape}, Synthetic: {orig.shape}")

# Combine for feature engineering
combine = pd.concat([train, test, orig], axis=0, ignore_index=True)
print(f"Combined: {combine.shape}")

# ===========================
# BAYESIAN FEATURE
# ===========================
FEATURES = list(orig.columns[1:-1])
TARGET = orig.columns[-1]

def f(X):
    return \
        0.3 * X["curvature"] + \
        0.2 * (X["lighting"] == "night").astype(int) + \
        0.1 * (X["weather"] != "clear").astype(int) + \
        0.2 * (X["speed_limit"] >= 60).astype(int) + \
        0.1 * (X["num_reported_accidents"] > 2).astype(int)

def clip(g):
    def clip_f(X):
        sigma = 0.05
        mu = g(X)
        a, b = -mu / sigma, (1 - mu) / sigma
        Phi_a, Phi_b = scipy.stats.norm.cdf(a), scipy.stats.norm.cdf(b)
        phi_a, phi_b = scipy.stats.norm.pdf(a), scipy.stats.norm.pdf(b)
        return mu * (Phi_b - Phi_a) + sigma * (phi_a - phi_b) + 1 - Phi_b
    return clip_f

z = clip(f)(combine)
combine["y"] = z.values

# Identify categorical and numerical features
CATS, NUMS = [], []
for c in FEATURES:
    if combine[c].dtype == "object":
        CATS.append(c)
    else:
        NUMS.append(c)

# ===========================
# FEATURE ENGINEERING
# ===========================
print("\nCreating features...")
NEW_FEATURES = FEATURES + ["y"]
NUMS_NEW = NUMS + ["y"]

# Pre-create needed interactions
combine["curv_plus_speed"] = combine["curvature"] + combine["speed_limit"]
combine["all_features_sum"] = combine["curvature"] + combine["speed_limit"] + combine["num_reported_accidents"]
combine["curvature_times_speed"] = combine["curvature"] * combine["speed_limit"]

# Risk score
combine["risk_score"] = (
    combine["curvature"] * 0.3 +
    (combine["lighting"] == "night").astype(int) * 0.2 +
    (combine["weather"] != "clear").astype(int) * 0.1 +
    (combine["speed_limit"] >= 60).astype(int) * 0.2 +
    (combine["num_reported_accidents"] > 2).astype(int) * 0.1 +
    (combine["holiday"] == True).astype(int) * 0.05
)
combine["risk_score_sq"] = combine["risk_score"] ** 2

NEW_FEATURES.extend(["curv_plus_speed", "all_features_sum", "curvature_times_speed", "risk_score", "risk_score_sq"])
NUMS_NEW.extend(["curv_plus_speed", "all_features_sum", "curvature_times_speed", "risk_score", "risk_score_sq"])

# Polynomial features
for col in ["y", "curvature", "num_reported_accidents"]:
    combine[f"{col}_sq"] = combine[col] ** 2
    combine[f"{col}_cube"] = combine[col] ** 3
    NEW_FEATURES.extend([f"{col}_sq", f"{col}_cube"])
    NUMS_NEW.extend([f"{col}_sq", f"{col}_cube"])

# Ratios
key_cols = ["curvature", "num_reported_accidents", "risk_score", "speed_limit", "curv_plus_speed"]
for col in key_cols:
    combine[f"y_div_{col}"] = combine["y"] / (combine[col] + 1e-8)
    combine[f"{col}_div_y"] = combine[col] / (combine["y"] + 1e-8)
    NEW_FEATURES.extend([f"y_div_{col}", f"{col}_div_y"])
    NUMS_NEW.extend([f"y_div_{col}", f"{col}_div_y"])

# More interactions
combine["risk_times_curvature"] = combine["risk_score"] * combine["curvature"]
combine["y_risk_interaction"] = combine["y"] * combine["risk_score"]
combine["y_curvature_interaction"] = combine["y"] * combine["curvature"]
NEW_FEATURES.extend(["risk_times_curvature", "y_risk_interaction", "y_curvature_interaction"])
NUMS_NEW.extend(["risk_times_curvature", "y_risk_interaction", "y_curvature_interaction"])

# Log transforms
for col in ["curvature", "num_reported_accidents", "speed_limit"]:
    combine[f"{col}_log"] = np.log1p(combine[col])
    NEW_FEATURES.append(f"{col}_log")
    NUMS_NEW.append(f"{col}_log")

# Binning
combine["curvature_bin"] = pd.cut(combine["curvature"], bins=10, labels=False).astype(float)
combine["speed_limit_bin"] = pd.cut(combine["speed_limit"], bins=8, labels=False).astype(float)
combine["num_accidents_bin"] = pd.cut(combine["num_reported_accidents"], bins=6, labels=False).astype(float)
NEW_FEATURES.extend(["curvature_bin", "speed_limit_bin", "num_accidents_bin"])
NUMS_NEW.extend(["curvature_bin", "speed_limit_bin", "num_accidents_bin"])

# Label encode categoricals
for c in CATS:
    le = LabelEncoder()
    combine[c] = le.fit_transform(combine[c].astype(str))

# Split back
train_v = combine.iloc[:len(train)].copy()
test_v = combine.iloc[len(train):len(train) + len(test)].copy()
orig_v = combine.iloc[-len(orig):].copy()

# ===========================
# TARGET ENCODING
# ===========================
print("Creating target encodings...")
TE_FEATURES = [
    "y", "curvature", "num_reported_accidents", "risk_score", "risk_score_sq",
    "all_features_sum", "curv_plus_speed", "curvature_bin", "speed_limit_bin", 
    "num_accidents_bin", "curvature_times_speed", "risk_times_curvature",
    "y_risk_interaction", "y_curvature_interaction"
]

TE = []
for c in TE_FEATURES:
    if c in NEW_FEATURES:
        tmp = orig_v.groupby(c)[TARGET].mean()
        n = f"TE_{c}"
        tmp.name = n
        train_v = train_v.merge(tmp, on=c, how="left")
        test_v = test_v.merge(tmp, on=c, how="left")
        TE.append(n)

# Fill NaNs
global_mean = orig_v[TARGET].mean()
for te_col in TE:
    train_v[te_col].fillna(global_mean, inplace=True)
    test_v[te_col].fillna(global_mean, inplace=True)

FINAL_FEATURES = NEW_FEATURES + TE
print(f"Total features: {len(FINAL_FEATURES)}")

# ===========================
# BEST PARAMETERS FROM OPTUNA
# ===========================
BEST_PARAMS = {
    "n_estimators": 543,
    "max_depth": 16,
    "min_samples_split": 39,
    "min_samples_leaf": 11,
    "max_features": 0.6336965827544817,
    "max_leaf_nodes": 1000,
    "min_weight_fraction_leaf": 0.0,
    "ccp_alpha": 0.0,
    "bootstrap": False,
    "n_jobs": -1,
    "random_state": SEED,
    "criterion": "squared_error",
}

print("\nBest parameters from Optuna:")
for k, v in BEST_PARAMS.items():
    if k not in ["n_jobs", "random_state", "criterion"]:
        print(f"  {k}: {v}")

# ===========================
# TRAIN WITH CROSS-VALIDATION
# ===========================
print(f"\n{'='*60}")
print(f"TRAINING {FOLDS}-FOLD CROSS-VALIDATION")
print('='*60)

kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
oof_preds = np.zeros(len(train_v), dtype=float)
test_preds = np.zeros(len(test_v), dtype=float)
fold_scores = []

X_test = test_v[FINAL_FEATURES]

for fold_num, (tr_idx, va_idx) in enumerate(kf.split(train_v), 1):
    print(f"\nFold {fold_num}:")
    X_tr = train_v.iloc[tr_idx][FINAL_FEATURES]
    y_tr = train_v.iloc[tr_idx][TARGET] - train_v.iloc[tr_idx]["y"]
    
    X_va = train_v.iloc[va_idx][FINAL_FEATURES]
    y_va_true = train_v.iloc[va_idx][TARGET].values
    y_va_base = train_v.iloc[va_idx]["y"].values
    
    model = ExtraTreesRegressor(**BEST_PARAMS)
    model.fit(X_tr, y_tr)
    
    va_pred = model.predict(X_va) + y_va_base
    oof_preds[va_idx] = va_pred
    rmse = np.sqrt(mean_squared_error(y_va_true, va_pred))
    fold_scores.append(rmse)
    print(f"  RMSE: {rmse:.6f}")
    
    test_fold_pred = model.predict(X_test) + test_v["y"].values
    test_preds += test_fold_pred / FOLDS
    
    if fold_num == 1:
        importances = model.feature_importances_
        idxs = np.argsort(importances)[::-1][:20]
        print("\n  Top 20 Features:")
        for j, idx in enumerate(idxs):
            print(f"    {j+1}. {FINAL_FEATURES[idx]}: {importances[idx]:.4f}")

cv_mean = float(np.mean(fold_scores))
cv_std = float(np.std(fold_scores))

print(f"\n{'='*60}")
print("SUMMARY")
print('='*60)
print(f"Fold scores: {[f'{s:.6f}' for s in fold_scores]}")
print(f"Mean CV RMSE: {cv_mean:.6f} (+/- {cv_std:.6f})")

# ===========================
# SAVE OUTPUTS
# ===========================
prefix = title_prefix(cv_mean)

submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": test_preds
})

submission.to_csv("submission.csv", index=False)
np.save(f"{prefix}_oof.npy", oof_preds)
np.save(f"{prefix}_test.npy", test_preds)

print(f"\n✓ Saved: submission.csv")
print(f"✓ Saved: {prefix}_oof.npy")
print(f"✓ Saved: {prefix}_test.npy")
print(f"\n{'='*60}")
print("TRAINING COMPLETE!")
print('='*60)


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import glob

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*80)
print("MODEL DIVERSITY ANALYSIS")
print("ExtraTrees vs XGBoost vs LightGBM")
print("="*80 + "\n")

# ===========================
# LOAD DATA
# ===========================
print("Loading data...")
train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
TARGET = 'accident_risk'
y_true = train[TARGET].values
n_samples = len(y_true)
print(f"✓ Ground truth: {n_samples} samples\n")

# ===========================
# LOAD MODEL PREDICTIONS
# ===========================
MODELS = {
    'ExtraTrees': {
        'path': '/kaggle/working',
        'color': '#2E86AB',
        'marker': 'o'
    },
    'XGBoost': {
        'path': '/kaggle/input/feature-rich-single-xgb-cv-0-05594',
        'color': '#A23B72',
        'marker': 's'
    },
    'LightGBM': {
        'path': '/kaggle/input/bayesian-residual-lightgbm-cv-0-05595',
        'color': '#F18F01',
        'marker': '^'
    }
}

predictions = {}

# Load ExtraTrees (most recent file in working directory)
et_files = glob.glob('/kaggle/working/*_oof.npy')
if et_files:
    et_files.sort()
    et_path = et_files[-1]
    print(f"Loading ExtraTrees from: {et_path}")
    predictions['ExtraTrees'] = np.load(et_path)
    rmse = np.sqrt(mean_squared_error(y_true, predictions['ExtraTrees']))
    print(f"  ✓ RMSE: {rmse:.6f}\n")

# Load XGBoost
xgb_files = glob.glob('/kaggle/input/feature-rich-single-xgb-cv-0-05594/*oof*.npy')
if xgb_files:
    xgb_files.sort()
    xgb_path = xgb_files[-1]
    print(f"Loading XGBoost from: {xgb_path}")
    predictions['XGBoost'] = np.load(xgb_path)
    rmse = np.sqrt(mean_squared_error(y_true, predictions['XGBoost']))
    print(f"  ✓ RMSE: {rmse:.6f}\n")

# Load LightGBM
lgb_files = glob.glob('/kaggle/input/bayesian-residual-lightgbm-cv-0-05595/*oof*.npy')
if lgb_files:
    lgb_files.sort()
    lgb_path = lgb_files[-1]
    print(f"Loading LightGBM from: {lgb_path}")
    predictions['LightGBM'] = np.load(lgb_path)
    rmse = np.sqrt(mean_squared_error(y_true, predictions['LightGBM']))
    print(f"  ✓ RMSE: {rmse:.6f}\n")

model_names = list(predictions.keys())
n_models = len(model_names)
print(f"Analyzing {n_models} models: {', '.join(model_names)}\n")

# ===========================
# 1. PERFORMANCE METRICS
# ===========================
print("="*80)
print("1. PERFORMANCE METRICS")
print("="*80 + "\n")

metrics_data = []
for model_name in model_names:
    preds = predictions[model_name]
    
    rmse = np.sqrt(mean_squared_error(y_true, preds))
    mae = mean_absolute_error(y_true, preds)
    r2 = r2_score(y_true, preds)
    
    residuals = y_true - preds
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    max_err = np.max(np.abs(residuals))
    
    metrics_data.append({
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2,
        'Mean_Residual': mean_res,
        'Std_Residual': std_res,
        'Max_Error': max_err
    })
    
    print(f"{model_name}:")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  MAE: {mae:.6f}")
    print(f"  R²: {r2:.6f}")
    print(f"  Max Error: {max_err:.6f}\n")

metrics_df = pd.DataFrame(metrics_data)

# ===========================
# 2. CORRELATION ANALYSIS
# ===========================
print("="*80)
print("2. CORRELATION ANALYSIS")
print("="*80 + "\n")

print("Pearson Correlation:")
pearson_matrix = np.zeros((n_models, n_models))
for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        corr, _ = pearsonr(predictions[m1], predictions[m2])
        pearson_matrix[i, j] = corr
        if i < j:
            print(f"  {m1:15s} vs {m2:15s}: {corr:.6f}")

print("\nSpearman Correlation:")
spearman_matrix = np.zeros((n_models, n_models))
for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        corr, _ = spearmanr(predictions[m1], predictions[m2])
        spearman_matrix[i, j] = corr
        if i < j:
            print(f"  {m1:15s} vs {m2:15s}: {corr:.6f}")

avg_pearson = np.mean([pearson_matrix[i, j] for i in range(n_models) for j in range(i+1, n_models)])
avg_spearman = np.mean([spearman_matrix[i, j] for i in range(n_models) for j in range(i+1, n_models)])

print(f"\nAverage Pearson: {avg_pearson:.6f}")
print(f"Average Spearman: {avg_spearman:.6f}")
print(f"Diversity Score: {1 - avg_pearson:.6f}")

# ===========================
# 3. ERROR AGREEMENT
# ===========================
print("\n" + "="*80)
print("3. ERROR AGREEMENT")
print("="*80 + "\n")

residuals = {name: y_true - predictions[name] for name in model_names}

print("Residual Correlation:")
residual_corr = np.zeros((n_models, n_models))
for i, m1 in enumerate(model_names):
    for j, m2 in enumerate(model_names):
        corr, _ = pearsonr(residuals[m1], residuals[m2])
        residual_corr[i, j] = corr
        if i < j:
            print(f"  {m1:15s} vs {m2:15s}: {corr:.6f}")

avg_residual_corr = np.mean([residual_corr[i, j] for i in range(n_models) for j in range(i+1, n_models)])
print(f"\nAverage Residual Correlation: {avg_residual_corr:.6f}")

# ===========================
# 4. PREDICTION DISTRIBUTIONS
# ===========================
print("\n" + "="*80)
print("4. PREDICTION DISTRIBUTIONS")
print("="*80 + "\n")

for model_name in model_names:
    preds = predictions[model_name]
    print(f"{model_name}:")
    print(f"  Mean: {np.mean(preds):.6f}")
    print(f"  Std: {np.std(preds):.6f}")
    print(f"  Min: {np.min(preds):.6f}")
    print(f"  Max: {np.max(preds):.6f}")
    print(f"  Median: {np.median(preds):.6f}\n")

# ===========================
# VISUALIZATIONS
# ===========================
print("="*80)
print("GENERATING VISUALIZATIONS")
print("="*80 + "\n")

fig = plt.figure(figsize=(20, 12))

# Plot 1: Performance Comparison
ax1 = plt.subplot(2, 3, 1)
rmse_values = [metrics_df[metrics_df['Model'] == name]['RMSE'].values[0] for name in model_names]
colors = [MODELS[name]['color'] for name in model_names]
bars = ax1.bar(model_names, rmse_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax1.set_ylabel('RMSE', fontsize=12, fontweight='bold')
ax1.set_title('Performance Comparison', fontsize=14, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, rmse_values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.00002, 
             f'{val:.6f}', ha='center', va='bottom', fontweight='bold', fontsize=10)

# Plot 2: R² Comparison
ax2 = plt.subplot(2, 3, 2)
r2_values = [metrics_df[metrics_df['Model'] == name]['R²'].values[0] for name in model_names]
bars = ax2.bar(model_names, r2_values, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax2.set_ylabel('R²', fontsize=12, fontweight='bold')
ax2.set_title('R² Comparison', fontsize=14, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for bar, val in zip(bars, r2_values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.002, 
             f'{val:.6f}', ha='center', va='top', fontweight='bold', fontsize=10)

# Plot 3: Prediction Distribution
ax3 = plt.subplot(2, 3, 3)
for model_name in model_names:
    ax3.hist(predictions[model_name], bins=50, alpha=0.5, 
             label=model_name, color=MODELS[model_name]['color'], edgecolor='black')
ax3.axvline(np.mean(y_true), color='red', linestyle='--', linewidth=2, label='True Mean')
ax3.set_xlabel('Prediction Value', fontsize=11, fontweight='bold')
ax3.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax3.set_title('Prediction Distributions', fontsize=12, fontweight='bold')
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Plot 4: Residual Distribution
ax4 = plt.subplot(2, 3, 4)
for model_name in model_names:
    ax4.hist(residuals[model_name], bins=50, alpha=0.5, 
             label=model_name, color=MODELS[model_name]['color'], edgecolor='black')
ax4.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero')
ax4.set_xlabel('Residual', fontsize=11, fontweight='bold')
ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
ax4.set_title('Residual Distributions', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(alpha=0.3)

# Plot 5: Actual vs Predicted
ax5 = plt.subplot(2, 3, 5)
for model_name in model_names:
    ax5.scatter(y_true, predictions[model_name], 
                alpha=0.3, s=2, label=model_name,
                color=MODELS[model_name]['color'])
ax5.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
         'r--', linewidth=2, label='Perfect')
ax5.set_xlabel('True Values', fontsize=11, fontweight='bold')
ax5.set_ylabel('Predicted Values', fontsize=11, fontweight='bold')
ax5.set_title('Actual vs Predicted', fontsize=12, fontweight='bold')
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

# Plot 6: Error by Range
ax6 = plt.subplot(2, 3, 6)
bins = np.linspace(y_true.min(), y_true.max(), 20)
bin_centers = (bins[:-1] + bins[1:]) / 2

for model_name in model_names:
    bin_errors = []
    for i in range(len(bins)-1):
        mask = (y_true >= bins[i]) & (y_true < bins[i+1])
        if mask.sum() > 0:
            bin_rmse = np.sqrt(mean_squared_error(y_true[mask], predictions[model_name][mask]))
            bin_errors.append(bin_rmse)
        else:
            bin_errors.append(np.nan)
    
    ax6.plot(bin_centers, bin_errors, marker=MODELS[model_name]['marker'], 
             label=model_name, color=MODELS[model_name]['color'], linewidth=2, markersize=6)

ax6.set_xlabel('True Value Range', fontsize=11, fontweight='bold')
ax6.set_ylabel('RMSE', fontsize=11, fontweight='bold')
ax6.set_title('Error by Prediction Range', fontsize=12, fontweight='bold')
ax6.legend(fontsize=9)
ax6.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('model_diversity_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_diversity_analysis.png")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

