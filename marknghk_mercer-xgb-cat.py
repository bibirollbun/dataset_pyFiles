# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import gc
import time
import warnings
import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ====================================================
# 1. CONFIGURATION & CUSTOM METRIC
# ====================================================
INPUT_DIR = "/kaggle/input/mercor-cheating-detection"
N_FOLDS = 5
SEED = 42

def calculate_mercor_cost(y_true, y_pred_probs, t_low=None, t_high=None):
    """
    Calculates the Mercor Cost. If thresholds are None, it searches for optimal ones.
    """
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs)
    
    # If no thresholds provided, we optimize (simple grid search)
    if t_low is None or t_high is None:
        best_cost = float('inf')
        # Search percentiles to avoid arbitrary scale issues
        thresholds = np.unique(np.percentile(y_pred_probs, np.linspace(0, 100, 50)))
        
        for low in thresholds:
            # We only check highs that are strictly greater than low
            possible_highs = thresholds[thresholds > low]
            for high in possible_highs:
                # Calculate cost for this pair
                # 0=Pass, 1=Review, 2=Block
                decisions = np.zeros(len(y_true))
                decisions[y_pred_probs >= low] = 1
                decisions[y_pred_probs >= high] = 2
                
                cost = 0
                # False Negative (Cheater passes) -> $600
                cost += np.sum((decisions == 0) & (y_true == 1)) * 600
                # False Positive Block (Clean blocked) -> $300
                cost += np.sum((decisions == 2) & (y_true == 0)) * 300
                # False Positive Review (Clean reviewed) -> $150
                cost += np.sum((decisions == 1) & (y_true == 0)) * 150
                # True Positive Review (Cheater reviewed) -> $5
                cost += np.sum((decisions == 1) & (y_true == 1)) * 5
                
                if cost < best_cost:
                    best_cost = cost
                    t_low = low
                    t_high = high
        
        return best_cost, t_low, t_high

    # If thresholds provided, just calc cost
    decisions = np.zeros(len(y_true))
    decisions[y_pred_probs >= t_low] = 1
    decisions[y_pred_probs >= t_high] = 2
    
    cost = 0
    cost += np.sum((decisions == 0) & (y_true == 1)) * 600
    cost += np.sum((decisions == 2) & (y_true == 0)) * 300
    cost += np.sum((decisions == 1) & (y_true == 0)) * 150
    cost += np.sum((decisions == 1) & (y_true == 1)) * 5
    
    return cost


# ====================================================
# 2. DATA LOADING & PREPROCESSING
# ====================================================

print("Loading Data...")
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
graph_df = pd.read_csv(os.path.join(INPUT_DIR, "social_graph.csv"), names=["source", "target"])

train['is_train'] = 1
test['is_train'] = 0
test['is_cheating'] = np.nan
all_data = pd.concat([train, test], axis=0).reset_index(drop=True)

# Map Users
unique_users = all_data['user_hash'].unique()
user_to_idx = {u: i for i, u in enumerate(unique_users)}
all_data['user_idx'] = all_data['user_hash'].map(user_to_idx)

print(f"Total Users: {len(unique_users)}")




# 2. ROBUST GRAPH FEATURES
print("Computing Graph Features...")
graph_df['source_idx'] = graph_df['source'].map(user_to_idx).fillna(-1).astype(int)
graph_df['target_idx'] = graph_df['target'].map(user_to_idx).fillna(-1).astype(int)
graph_df = graph_df[(graph_df['source_idx'] >= 0) & (graph_df['target_idx'] >= 0)]

n_users = len(unique_users)
rows = np.concatenate([graph_df['source_idx'], graph_df['target_idx']])
cols = np.concatenate([graph_df['target_idx'], graph_df['source_idx']])
data = np.ones(len(rows))
adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_users, n_users))

# A. Degree
degrees = np.array(adj_matrix.sum(axis=1)).flatten()
all_data['graph_degree'] = all_data['user_idx'].map(lambda x: degrees[x])

# B. Neighbor Cheat Ratio (Leakage Free)
all_data['neighbor_cheat_ratio'] = np.nan
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train_df_only = all_data[all_data['is_train'] == 1]
y_global = np.zeros(n_users)
train_indices = all_data[all_data['is_train'] == 1].index
y_global[all_data.loc[train_indices, 'user_idx']] = all_data.loc[train_indices, 'is_cheating'].fillna(0)

for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df_only, train_df_only['is_cheating'].fillna(0))):
    global_tr = train_df_only.iloc[tr_idx].index
    global_val = train_df_only.iloc[val_idx].index
    y_fold = np.zeros(n_users)
    y_fold[all_data.loc[global_tr, 'user_idx']] = all_data.loc[global_tr, 'is_cheating'].fillna(0)
    nbr_sums = adj_matrix.dot(y_fold)
    val_u = all_data.loc[global_val, 'user_idx'].values
    all_data.loc[global_val, 'neighbor_cheat_ratio'] = nbr_sums[val_u] / (degrees[val_u] + 1e-6)

y_full = np.zeros(n_users)
y_full[all_data.loc[train_indices, 'user_idx']] = all_data.loc[train_indices, 'is_cheating'].fillna(0)
nbr_sums_test = adj_matrix.dot(y_full)
test_idx = all_data[all_data['is_train'] == 0].index
test_u = all_data.loc[test_idx, 'user_idx'].values
all_data.loc[test_idx, 'neighbor_cheat_ratio'] = nbr_sums_test[test_u] / (degrees[test_u] + 1e-6)
all_data['neighbor_cheat_ratio'] = all_data['neighbor_cheat_ratio'].fillna(0)

# C. Propagated Risk
print("   Propagating Risk...")
d_inv = 1.0 / (degrees + 1e-6)
P = sp.diags(d_inv).dot(adj_matrix)
all_data['propagated_risk'] = np.nan
global_mean = train['is_cheating'].mean()

for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df_only, train_df_only['is_cheating'].fillna(0))):
    global_tr = train_df_only.iloc[tr_idx].index
    global_val = train_df_only.iloc[val_idx].index
    Y_init = np.zeros(n_users) + global_mean
    tr_u = all_data.loc[global_tr, 'user_idx'].values
    Y_init[tr_u] = all_data.loc[global_tr, 'is_cheating'].values
    Y_curr = Y_init.copy()
    for _ in range(15):
        Y_curr = P.dot(Y_curr)
        Y_curr[tr_u] = Y_init[tr_u]
    val_u = all_data.loc[global_val, 'user_idx'].values
    all_data.loc[global_val, 'propagated_risk'] = Y_curr[val_u]

Y_init = np.zeros(n_users) + global_mean
tr_u = all_data.loc[train_indices, 'user_idx'].values
Y_init[tr_u] = all_data.loc[train_indices, 'is_cheating'].values
Y_curr = Y_init.copy()
for _ in range(15):
    Y_curr = P.dot(Y_curr)
    Y_curr[tr_u] = Y_init[tr_u]
all_data.loc[test_idx, 'propagated_risk'] = Y_curr[test_u]


from catboost import CatBoostClassifier

# 3. HIGH-RES TRAINING
feature_cols = [c for c in train.columns if c.startswith("feature_")]
feature_cols += ['graph_degree', 'neighbor_cheat_ratio', 'propagated_risk']

print(f"Training on {len(feature_cols)} features.")

X = all_data[all_data['is_train'] == 1].dropna(subset=['is_cheating']).reset_index(drop=True)
y = X['is_cheating'].astype(int)
X_test = all_data[all_data['is_train'] == 0].reset_index(drop=True)

xgb_oof, xgb_test = np.zeros(len(X)), np.zeros(len(X_test))
cat_oof, cat_test = np.zeros(len(X)), np.zeros(len(X_test))

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print("Starting Slow-Rate Training (This may take 2-3 mins)...")

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, y_tr = X.loc[train_idx, feature_cols], y[train_idx]
    X_val, y_val = X.loc[val_idx, feature_cols], y[val_idx]
    
    # XGBoost: Lower LR, Higher Estimators
    xgb = XGBClassifier(
        n_estimators=2500,        # Increased
        learning_rate=0.02,       # Slowed down
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=2.0,     # The Safety Net
        random_state=SEED,
        eval_metric='logloss',
        early_stopping_rounds=100
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    xgb_oof[val_idx] = xgb.predict_proba(X_val)[:, 1]
    xgb_test += xgb.predict_proba(X_test[feature_cols])[:, 1] / N_FOLDS
    
    # CatBoost: Lower LR, Higher Estimators
    cat = CatBoostClassifier(
        iterations=2500,          # Increased
        learning_rate=0.02,       # Slowed down
        depth=6,
        scale_pos_weight=2.0,     # The Safety Net
        eval_metric='Logloss',
        random_seed=SEED,
        verbose=False,
        early_stopping_rounds=100
    )
    cat.fit(X_tr, y_tr, eval_set=(X_val, y_val))
    cat_oof[val_idx] = cat.predict_proba(X_val)[:, 1]
    cat_test += cat.predict_proba(X_test[feature_cols])[:, 1] / N_FOLDS
    
    print(f"Fold {fold+1} Done.")


# ====================================================
# [Paste this at the bottom of your notebook]
# ====================================================

# 1. BLEND THE MODELS
# Average the predictions from XGBoost and CatBoost
final_oof = 0.4 * xgb_oof + 0.6 * cat_oof
final_test = 0.4 * xgb_test + 0.6 * cat_test

# 2. OFFICIAL FAST SCORING FUNCTION
# This is the exact math from the competition hosts
def fast_mercor_score(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Sort by prediction (low to high)
    sort_order = np.argsort(y_pred)
    y_true_sorted = y_true[sort_order]
    
    # Calculate optimal thresholds using Cumulative Sum (Fast & Exact)
    c1 = (y_true_sorted * 745 - 150).cumsum()
    c2 = (y_true_sorted * 155 - 150).cumsum()
    base_cost = (y_true == 0).sum() * 300
    
    min_total_cost = c1.min() + c2.min() + base_cost
    return min_total_cost

# 3. CALCULATE & PRINT EXACT COST
real_cost = fast_mercor_score(y, final_oof)

print(f"\n=============================================")
print(f"✅ EXACT LOCAL COST: ${real_cost:,.2f}")
print(f"=============================================")

# 4. SAVE SUBMISSION
sub = pd.DataFrame({'user_hash': X_test['user_hash'], 'prediction': final_test})
sub.to_csv("submission.csv", index=False)
print("Saved submission.csv")

