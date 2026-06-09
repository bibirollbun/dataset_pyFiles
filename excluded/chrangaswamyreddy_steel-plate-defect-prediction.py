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


# =============================================================
# ğŸ§  Steel Plate Defect Prediction with Graphs
# =============================================================

# === 1ï¸�âƒ£ Import Libraries ===
import numpy as np
import pandas as pd
import os
import gc
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
from lightgbm import early_stopping, log_evaluation

sns.set(style="whitegrid", palette="Set2")
plt.rcParams["figure.figsize"] = (10,6)

# === 2ï¸�âƒ£ Load Dataset ===
DATA_DIR = "/kaggle/input/playground-series-s4e3"

train = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))

print("âœ… Data loaded successfully!")
print(f"Train shape: {train.shape}, Test shape: {test.shape}")

# === 3ï¸�âƒ£ Identify ID and Target Columns ===
id_col = "id"
possible_targets = ["Pastry", "Z_Scratch", "K_Scatch", "K_Scratch", "Stains", "Dirtiness", "Bumps", "Other_Faults"]
target_cols = [c for c in possible_targets if c in train.columns]

print(f"\nğŸªª ID column: {id_col}")
print(f"ğŸ�¯ Detected target columns ({len(target_cols)}): {target_cols}")

# === 4ï¸�âƒ£ Define Features ===
feature_cols = [c for c in train.columns if c not in [id_col] + target_cols]
feature_cols = [c for c in feature_cols if c in test.columns]
print(f"âœ… Number of features used: {len(feature_cols)}")

# === 5ï¸�âƒ£ Prepare Data ===
X = train[feature_cols].copy()
X_test = test[feature_cols].copy()
y = train[target_cols].copy()

# Handle missing values
for df in [X, X_test]:
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].fillna("Missing").astype(str)
            df[col] = pd.factorize(df[col])[0]
        else:
            df[col] = df[col].fillna(df[col].median())

print("\nâœ… Preprocessing complete!")

# === 6ï¸�âƒ£ Visualize Target Distributions ===
plt.figure(figsize=(12,6))
for i, target in enumerate(target_cols, 1):
    plt.subplot(2, 4, i)
    sns.countplot(y[target])
    plt.title(f"{target} Distribution")
plt.tight_layout()
plt.show()

# === 7ï¸�âƒ£ Correlation Heatmap ===
plt.figure(figsize=(12,10))
corr = train[target_cols].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.title("Target Correlation Heatmap")
plt.show()

# === 8ï¸�âƒ£ LightGBM Parameters ===
lgb_params = {
    "objective": "binary",
    "boosting_type": "gbdt",
    "metric": "auc",
    "learning_rate": 0.02,
    "num_leaves": 31,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 4,
    "seed": 42,
    "verbose": -1,
    "n_jobs": -1,
}

# === 9ï¸�âƒ£ Cross-Validation Setup ===
N_FOLDS = 5
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
X_values = X.values
X_test_values = X_test.values

oof = np.zeros((len(X), len(target_cols)))
preds_test = np.zeros((len(X_test), len(target_cols)))
scores = []
feature_importances = pd.DataFrame()

# === ğŸ”Ÿ Model Training Loop ===
for tidx, target in enumerate(target_cols):
    print(f"\n==============================")
    print(f"Training target [{tidx+1}/{len(target_cols)}]: {target}")
    print("==============================")

    y_col = y[target].values

    if np.isnan(y_col).all() or len(np.unique(y_col[~np.isnan(y_col)])) == 1:
        print(f"Skipping {target}")
        continue

    fold_scores = []
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X_values, y_col), 1):
        X_tr, X_val = X_values[tr_idx], X_values[val_idx]
        y_tr, y_val = y_col[tr_idx], y_col[val_idx]

        ltrain = lgb.Dataset(X_tr, label=y_tr)
        lvalid = lgb.Dataset(X_val, label=y_val, reference=ltrain)

        clf = lgb.train(
            params=lgb_params,
            train_set=ltrain,
            valid_sets=[ltrain, lvalid],
            valid_names=["train", "valid"],
            num_boost_round=5000,
            callbacks=[
                early_stopping(stopping_rounds=100),
                log_evaluation(period=200)
            ]
        )

        oof[val_idx, tidx] = clf.predict(X_val, num_iteration=clf.best_iteration)
        preds_test[:, tidx] += clf.predict(X_test_values, num_iteration=clf.best_iteration) / N_FOLDS
        fold_scores.append(roc_auc_score(y_val, oof[val_idx, tidx]))

        # Collect feature importances
        fold_importance = pd.DataFrame()
        fold_importance["feature"] = feature_cols
        fold_importance["importance"] = clf.feature_importance()
        fold_importance["fold"] = fold
        fold_importance["target"] = target
        feature_importances = pd.concat([feature_importances, fold_importance], axis=0)

        del clf, ltrain, lvalid
        gc.collect()

    mean_auc = np.mean(fold_scores)
    scores.append(mean_auc)
    print(f"âœ… Target {target} mean CV AUC: {mean_auc:.6f}")

# === 1ï¸�âƒ£1ï¸�âƒ£ Overall Performance ===
print("\n==============================")
print(f"Average CV AUC across all targets: {np.mean(scores):.6f}")
print("==============================")

# === 1ï¸�âƒ£2ï¸�âƒ£ Feature Importance Plot ===
plt.figure(figsize=(12,8))
imp_mean = feature_importances.groupby("feature")["importance"].mean().sort_values(ascending=False).head(20)
sns.barplot(x=imp_mean.values, y=imp_mean.index)
plt.title("Top 20 Feature Importances (averaged across targets)")
plt.show()

# === 1ï¸�âƒ£3ï¸�âƒ£ Submission ===
sub = sample_sub.copy()
for col in target_cols:
    if col in sub.columns:
        sub[col] = preds_test[:, target_cols.index(col)]
sub.to_csv("/kaggle/working/submission.csv", index=False)
print("\nğŸ’¾ Submission saved to /kaggle/working/submission.csv")
print(sub.head())





