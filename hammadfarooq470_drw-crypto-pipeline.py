# DRW - Crypto Market Prediction
# Notebook: DRW_crypto_pipeline.ipynb
# Author:  Hammad's helper
# Notebook contains: EDA, preprocessing, feature engineering, model training (LightGBM), CV, and submission generation.


# -----------------------------------------------------------------------------
# Notes before running
# - This notebook assumes you're running on Kaggle or a machine with enough disk/memory.
# - The full train.parquet is large (~6.8GB). If memory is limited, use a reduced sample or a recent time-window.
# - Follow the 'Avoid Future Peaking' rule: DO NOT use test.parquet for training or feature creation that leaks future info.
# - If running on Kaggle, enable "Internet: Off" is OK; data is on the competition dataset.
# -----------------------------------------------------------------------------


# %% [markdown]
# ## 0) Setup


# %%
import os
import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import GroupKFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from sklearn.metrics import mean_squared_error


# make plotting prettier
plt.rcParams['figure.figsize'] = (10, 5)


# Paths (on Kaggle these are mounted)
DATA_DIR = '/kaggle/input/drw-crypto-market-prediction'
TRAIN_PATH = os.path.join(DATA_DIR, 'train.parquet')
TEST_PATH = os.path.join(DATA_DIR, 'test.parquet')
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, 'sample_submission.csv')


# Utility to reduce memory
def reduce_mem_usage(df, verbose=True):
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtype
        if col_type != object and str(col_type)[:3] != 'dat':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min >= np.iinfo(np.int8).min and c_max <= np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min >= np.iinfo(np.int16).min and c_max <= np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min >= np.iinfo(np.int32).min and c_max <= np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                df[col] = pd.to_numeric(df[col], downcast='float')
        else:
            df[col] = df[col].astype('category') if df[col].nunique() < 100 else df[col]
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage decreased from {start_mem:.2f} MB to {end_mem:.2f} MB ' \
              f'({100 * (start_mem - end_mem) / start_mem:.1f}% reduction)')
    return df

# %% [markdown]
# ## 1) Quick EDA (sampled if necessary)

# %%
# Because full file is large, we'll read a subset for fast EDA. You can comment this out to load full data.
# Read every N-th row by reading parquet in pandas doesn't support step; instead sample after loading a small timeframe.

# We'll try to read the first 6 million rows (adjust if out-of-memory). If it fails, fall back to chunked sample.
try:
    df = pd.read_parquet(TRAIN_PATH)
    print('Loaded full train.parquet')
except Exception as e:
    print('Full load failed or skipped. Loading a sample using pyarrow with filters may be better on Kaggle.')
    # fallback: load with pandas from pyarrow but read smaller by reading selected columns first
    df = pd.read_parquet(TRAIN_PATH, engine='pyarrow')

print('Shape train:', df.shape)

# Reduce memory
numeric_cols = [c for c in df.columns if c.startswith('X_') or c in ['bid_qty','ask_qty','buy_qty','sell_qty','volume','label']]
obj_cols = [c for c in df.columns if c not in numeric_cols]
print('Num cols:', len(numeric_cols), 'Other cols:', obj_cols)
df = reduce_mem_usage(df)


import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.datasets import load_breast_cancer

# ===== Example data =====
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# ===== KFold setup =====
kf = KFold(n_splits=5, shuffle=True, random_state=42)
feature_importances = []

# ===== LightGBM training + collecting feature importances =====
for fold, (train_idx, valid_idx) in enumerate(kf.split(X)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y[train_idx], y[valid_idx]
    
    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        random_state=42
    )

    # ✅ Use callbacks instead of early_stopping_rounds
    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        eval_metric='auc',
        callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(0)]
    )
    
    # Collect feature importances
    fold_importance = pd.DataFrame({
        "feature": X.columns,
        "gain": model.booster_.feature_importance(importance_type='gain'),
        "fold": fold
    })
    feature_importances.append(fold_importance)

# Combine all folds
feature_importances = pd.concat(feature_importances, axis=0)

# ===== Compute mean importance =====
fi_mean = (
    feature_importances.groupby("feature")["gain"]
    .mean()
    .sort_values(ascending=False)
)

# ===== Plot top 20 =====
fi_mean.head(20).plot.barh()
plt.gca().invert_yaxis()
plt.title("Top 20 features (avg gain)")
plt.show()


# Basic info
# Plot top feature importances
fi_mean.head(20).plot.barh(); plt.gca().invert_yaxis(); plt.title('Top 20 features (avg gain)'); plt.show()


# %% [markdown]
# ## 5) Train full model on all data (for submission)
# Retrain on the full dataset using best parameters and save the final model.

# %%
import lightgbm as lgb
import pandas as pd

# ✅ Example: make sure X and y are already defined
# X = your training features DataFrame
# y = your target array or Series

# Define model parameters
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 1.0,
    "lambda_l2": 1.0,
    "seed": 42,
    "verbose": -1
}

# ✅ Choose a reasonable fallback for number of boosting rounds
# If you have a model named `bst` from cross-validation, you can replace this with bst.best_iteration
best_iter = 200

# Create LightGBM dataset
full_train = lgb.Dataset(X, label=y)

# Train model on full data
final_model = lgb.train(
    params=params,
    train_set=full_train,
    num_boost_round=best_iter
)

# Save model
model_path = "/kaggle/working/lgbm_final_model.txt"
final_model.save_model(model_path)
print("✅ Saved model successfully to:", model_path)


# %% [markdown]
# ## 6) Prepare test predictions and submission
# Note: On this competition the test timestamps are masked and labels are 0. We'll generate predictions and map them to sample_submission.
# IMPORTANT: Only use features that are computable without peeking into future; do not use test to engineer features that require future rows.

# %%
print('Loading test (may be large).')

# For Kaggle runtime: don't load test if you don't need. But to produce a submission, load it and prepare features similarly.
try:
    test = pd.read_parquet(TEST_PATH)
    print('Loaded test shape:', test.shape)
    # create same rolling features — but be careful: rolling across the whole test without past train continuity may differ.
    for f in base_features:
        if f in test.columns:
            test[f + '_rmean_3'] = test[f].rolling(window=3, min_periods=1).mean()
            test[f + '_rmean_15'] = test[f].rolling(window=15, min_periods=1).mean()
            test[f + '_rstd_15'] = test[f].rolling(window=15, min_periods=1).std().fillna(0)
    for f in ['bid_qty','ask_qty']:
        if f in test.columns:
            test[f + '_ratio'] = test[f] / (test['volume'] + 1e-9)
    # aggregate anonymized features blocks
    an_feats_test = [c for c in test.columns if c.startswith('X_')]
    for i in range(0, min(200, len(an_feats_test)), block_size):
        block = an_feats_test[i:i+block_size]
        if not block: break
        test[f'X_block_{i}_mean'] = test[block].mean(axis=1)
        test[f'X_block_{i}_std'] = test[block].std(axis=1)
    # select features
    test_features = [c for c in test.columns if c in feature_cols]
    print('Test features count:', len(test_features))
    preds = final_model.predict(test[test_features], num_iteration=final_model.best_iteration)
    # build submission
    sub = pd.read_csv(SAMPLE_SUB_PATH)
    sub['prediction'] = preds
    out_path = '/kaggle/working/submission.csv'
    sub.to_csv(out_path, index=False)
    print('Submission saved to', out_path)
except Exception as e:
    print('Could not load test or produce submission in this environment:', str(e))

# %% [markdown]
# ## 7) Next steps & ideas to improve
# - Try more advanced CV: Purged walk-forward or expanding window to better emulate real-time prediction.
# - Use feature selection (L1/XGBoost SHAP) to reduce dimensionality of 780 anonymized features.
# - Build models that capture time-series dynamics: Temporal fusion transformer, LSTM with careful leakage avoidance.
# - Use ensembling: LightGBM + CatBoost + Ridge or Neural Nets on top of features.
# - Use target encoding or cross-feature interactions via learned embeddings.
# - Use incremental training or streaming methods to mimic live predictions.

# %% [markdown]
# ## Notebook end
# Save any artifacts you want to keep and upload the produced submission to Kaggle for scoring.

