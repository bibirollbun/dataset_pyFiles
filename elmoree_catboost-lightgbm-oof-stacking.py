import numpy as np
import pandas as pd
import gc
import os
import warnings

import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")
%matplotlib inline

from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

import lightgbm as lgb
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor, Pool

warnings.filterwarnings('ignore')


RANDOM_STATE = 42
TARGET = "accident_risk"
IDCOL = "id"

cat_cols  = ["road_type","lighting","weather","time_of_day"]
bool_cols = ["road_signs_present","public_road","holiday","school_season"]
num_cols  = ["num_lanes","curvature","speed_limit","num_reported_accidents"]

train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")

for c in bool_cols:
    train[c] = train[c].astype(int)
    test[c]  = test[c].astype(int)

y = train[TARGET].values



plt.figure(figsize=(14, 5))

# 1. Distribution of the target variable
plt.subplot(1, 2, 1)
sns.histplot(train[TARGET], kde=True, bins=50, color='teal')
plt.title(f'Distribution of Target: {TARGET}')

# 2. Numerical feature correlation matrix
plt.subplot(1, 2, 2)
corr_matrix = train[num_cols + [TARGET]].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix (Numerical Features)')

plt.tight_layout()
plt.show()

# 3. The relationship between category features and the mean of the target
fig, axes = plt.subplots(2, 2, figsize=(14, 8))
for i, col in enumerate(cat_cols):
    ax = axes[i//2, i%2]
    grouped = train.groupby(col)[TARGET].mean().sort_values()
    sns.barplot(x=grouped.index, y=grouped.values, ax=ax, palette='viridis')
    ax.set_title(f'Mean Accident Risk by {col}')
    ax.tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.show()


print("--- [Visual Analysis] Data Distribution ---")
plt.figure(figsize=(15, 5))

# Distribution of the target variable
plt.subplot(1, 2, 1)
sns.histplot(y, kde=True, bins=50, color='teal')
plt.title(f'Distribution of Target: {TARGET}')
plt.xlabel('Accident Risk')

# Heatmap of Correlation of Numerical Features
plt.subplot(1, 2, 2)
corr_matrix = train[num_cols + [TARGET]].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix (Numerical Features)')

plt.tight_layout()
plt.show()


def oof_target_encode(train_df, test_df, y, cols, n_splits=5, seed=42, alpha=20.0, noise=0.01):
    """
    OOF target encoding with smoothing:
    enc = (sum + prior*alpha) / (count + alpha)
    """
    train_enc = train_df.copy()
    test_enc  = test_df.copy()
    prior = float(np.mean(y))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    rng = np.random.default_rng(seed)

    for c in cols:
        tr_col = np.zeros(len(train_enc), dtype=float)

        for tr_idx, va_idx in kf.split(train_enc):
            g = pd.DataFrame({c: train_enc.iloc[tr_idx][c], "y": y[tr_idx]}) \
                    .groupby(c)["y"].agg(["sum","count"])
            enc_map = (g["sum"] + prior*alpha) / (g["count"] + alpha)

            tr_col[va_idx] = train_enc.iloc[va_idx][c].map(enc_map).fillna(prior).values

        # Test with full-scale statistics.
        g_full = pd.DataFrame({c: train_enc[c], "y": y}).groupby(c)["y"].agg(["sum","count"])
        enc_full = (g_full["sum"] + prior*alpha) / (g_full["count"] + alpha)
        te_col = test_enc[c].map(enc_full).fillna(prior).values

        # Add a bit of noise to the TE of the train to enhance generalization.
        if noise and noise > 0:
            tr_col = tr_col * (1.0 + noise * rng.normal(0.0, 1.0, size=len(tr_col)))

        train_enc[c + "_te"] = tr_col
        test_enc[c + "_te"]  = te_col

    return train_enc, test_enc



train_viz, _ = oof_target_encode(train, test, y, cols=cat_cols, n_splits=5, seed=42)

sample_df = train_viz.sample(n=2000, random_state=42)
sample_y = train.loc[sample_df.index, TARGET]

plt.figure(figsize=(14, 4))
for i, col in enumerate(cat_cols[:3]): # fisrt 3
    plt.subplot(1, 3, i+1)
    sns.scatterplot(x=sample_df[col + "_te"], y=sample_y, alpha=0.3, color='purple')
    plt.title(f'TE Feature: {col} vs Target')
    plt.xlabel(f'{col}_te')
    plt.ylabel('Accident Risk')
plt.tight_layout()
plt.show()


def run_oof_cat_lgbm(train, test, y, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # 1) Do TE (for LGBM only)
    tr_te, te_te = oof_target_encode(train, test, y, cols=cat_cols, n_splits=n_splits, seed=seed, alpha=20.0, noise=0.01)

    # LGBM Features:  TE + bool + num + Your own interaction features 
    X_lgb = pd.concat(
        [tr_te[[c+"_te" for c in cat_cols]],
         tr_te[bool_cols].astype(int),
         tr_te[num_cols]],
        axis=1
    )
    T_lgb = pd.concat(
        [te_te[[c+"_te" for c in cat_cols]],
         te_te[bool_cols].astype(int),
         te_te[num_cols]],
        axis=1
    )

    # CatBoost Feature: Original category + bool + num
    feat_cat = cat_cols + bool_cols + num_cols
    X_cat = train[feat_cat].copy()
    T_cat = test[feat_cat].copy()
    cat_idx = [feat_cat.index(c) for c in cat_cols]

    oof_lgb = np.zeros(len(train))
    oof_cat = np.zeros(len(train))
    pred_lgb = np.zeros(len(test))
    pred_cat = np.zeros(len(test))

    lgb_params = dict(
        objective="regression",
        learning_rate=0.05,
        n_estimators=5000,
        num_leaves=128,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=seed,
        n_jobs=-1,
    )

    cat_params = dict(
        loss_function="RMSE",
        learning_rate=0.06,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=seed,
        iterations=8000,
        od_type="Iter",
        od_wait=300,
        verbose=False,
    )

    for fold, (tr_idx, va_idx) in enumerate(kf.split(train), 1):
        # LGBM
        Xtr, Xva = X_lgb.iloc[tr_idx], X_lgb.iloc[va_idx]
        ytr, yva = y[tr_idx], y[va_idx]

        mdl_lgb = LGBMRegressor(**lgb_params)
        mdl_lgb.fit(
            Xtr, ytr,
            eval_set=[(Xva, yva)],
            eval_metric="rmse",
            callbacks=[lgb.early_stopping(200, verbose=False)],
        )
        oof_lgb[va_idx] = mdl_lgb.predict(Xva, num_iteration=mdl_lgb.best_iteration_)
        pred_lgb += mdl_lgb.predict(T_lgb, num_iteration=mdl_lgb.best_iteration_) / n_splits

        #  CatBoost 
        pool_tr = Pool(X_cat.iloc[tr_idx], label=ytr, cat_features=cat_idx)
        pool_va = Pool(X_cat.iloc[va_idx], label=yva, cat_features=cat_idx)
        mdl_cat = CatBoostRegressor(**cat_params)
        mdl_cat.fit(pool_tr, eval_set=pool_va)

        oof_cat[va_idx] = mdl_cat.predict(pool_va)
        pred_cat += mdl_cat.predict(Pool(T_cat, cat_features=cat_idx)) / n_splits

        rmse_lgb = mean_squared_error(yva, oof_lgb[va_idx], squared=False)
        rmse_cat = mean_squared_error(yva, oof_cat[va_idx], squared=False)
        print(f"Fold {fold}: LGBM RMSE={rmse_lgb:.6f} | Cat RMSE={rmse_cat:.6f}")

    rmse_lgb_all = mean_squared_error(y, oof_lgb, squared=False)
    rmse_cat_all = mean_squared_error(y, oof_cat, squared=False)
    print("OOF LGBM RMSE:", rmse_lgb_all)
    print("OOF Cat  RMSE:", rmse_cat_all)

    return oof_lgb, oof_cat, pred_lgb, pred_cat



oof_lgb, oof_cat, pred_lgb, pred_cat = run_oof_cat_lgbm(train, test, y, n_splits=5, seed=RANDOM_STATE)


plt.figure(figsize=(14, 5))

# 1. True values vs.   Model OOF Comparison of the distribution of predicted values
plt.subplot(1, 2, 1)
sns.kdeplot(y, label='True Target', color='black', linewidth=2)
sns.kdeplot(oof_lgb, label='LGBM OOF', color='blue', alpha=0.5)
sns.kdeplot(oof_cat, label='CatBoost OOF', color='orange', alpha=0.5)
plt.title('Distribution: True vs Predicted Values')
plt.legend()

# 2. LGBM & CatBoost   Dependency of predicted values
plt.subplot(1, 2, 2)

indices = np.random.choice(len(oof_lgb), 2000, replace=False)
sns.scatterplot(x=oof_lgb[indices], y=oof_cat[indices], alpha=0.2, color='green')

min_val = min(oof_lgb.min(), oof_cat.min())
max_val = max(oof_lgb.max(), oof_cat.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--')
plt.title(f'LGBM vs CatBoost Correlation: {np.corrcoef(oof_lgb, oof_cat)[0,1]:.4f}')
plt.xlabel('LGBM Predictions')
plt.ylabel('CatBoost Predictions')

plt.tight_layout()
plt.show()


print("\n--- [Visual Analysis] Model Correlations & Predictions ---")
plt.figure(figsize=(15, 6))

# correlation 
plt.subplot(1, 2, 1)
idx = np.random.choice(len(oof_lgb), 2000, replace=False)
sns.scatterplot(x=oof_lgb[idx], y=oof_cat[idx], alpha=0.3, color='purple')

min_val, max_val = min(oof_lgb.min(), oof_cat.min()), max(oof_lgb.max(), oof_cat.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Correlation')
plt.xlabel('LGBM OOF Predictions')
plt.ylabel('CatBoost OOF Predictions')
plt.title(f'LGBM vs CatBoost Correlation: {np.corrcoef(oof_lgb, oof_cat)[0,1]:.4f}')
plt.legend()

# Comparison of predicted value distributions
plt.subplot(1, 2, 2)
sns.kdeplot(y, label='True Target', color='black', fill=True, alpha=0.1)
sns.kdeplot(oof_lgb, label='LGBM OOF', color='blue')
sns.kdeplot(oof_cat, label='CatBoost OOF', color='orange')
plt.title('Distribution: True vs Predicted Values')
plt.xlabel('Accident Risk')
plt.legend()

plt.tight_layout()
plt.show()


stack_X = np.vstack([oof_lgb, oof_cat]).T
meta = Ridge(alpha=1.0)
meta.fit(stack_X, y)

# OOF Evaluation
oof_stack = meta.predict(stack_X)
rmse_stack = mean_squared_error(y, oof_stack, squared=False)
print("OOF Stacking RMSE:", rmse_stack)

# Prediction Test
stack_T = np.vstack([pred_lgb, pred_cat]).T
pred_stack = meta.predict(stack_T)

# Clip to [0,1]
pred_stack = np.clip(pred_stack, 0.0, 1.0)


print("\n--- [Visual Analysis] Stacking Weights & Residuals ---")
plt.figure(figsize=(15, 5))

# Ridge regression Coefficients
plt.subplot(1, 2, 1)
models = ['LightGBM', 'CatBoost']
weights = meta.coef_
sns.barplot(x=models, y=weights, palette='magma')
for i, v in enumerate(weights):
    plt.text(i, v, f"{v:.3f}", ha='center', va='bottom', fontsize=12)
plt.title('Stacking Model Weights (Ridge Coefficients)')
plt.ylabel('Coefficient Value')
plt.ylim(0, max(weights)*1.2)

# Final residual distribution
plt.subplot(1, 2, 2)
residuals = y - oof_stack
sns.histplot(residuals, bins=50, kde=True, color='crimson')
plt.title('Residuals Distribution (True - Stacked Pred)')
plt.xlabel('Residual')

plt.tight_layout()
plt.show()



plt.figure(figsize=(10, 4))

# Display the weights given to each model by Ridge Regression
weights = meta.coef_
model_names = ['LightGBM', 'CatBoost']

plt.subplot(1, 2, 1)
sns.barplot(x=model_names, y=weights, palette='magma')
plt.title('Stacking Model Weights (Coefficients)')
plt.ylabel('Weight')
for i, v in enumerate(weights):
    plt.text(i, v, f"{v:.3f}", ha='center', va='bottom')

# Display the residual distribution after Stacking
residuals = y - oof_stack
plt.subplot(1, 2, 2)
sns.histplot(residuals, bins=50, kde=True, color='red')
plt.title('Residuals Distribution (True - Stacked Pred)')
plt.xlabel('Residual')

plt.tight_layout()
plt.show()


sub = sample.copy()
sub[TARGET] = pred_stack
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("\nSubmission saved to submission.csv")
sub.head()

