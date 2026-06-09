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
print("Loading data...")
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
graph_df = pd.read_csv(os.path.join(INPUT_DIR, "social_graph.csv"), names=["source", "target"])

# Combine for feature engineering
train['is_train'] = 1
test['is_train'] = 0
test['is_cheating'] = np.nan
all_data = pd.concat([train, test], axis=0).reset_index(drop=True)

# Map user_hash to integer IDs for sparse matrix operations
unique_users = all_data['user_hash'].unique()
user_to_idx = {u: i for i, u in enumerate(unique_users)}
all_data['user_idx'] = all_data['user_hash'].map(user_to_idx)

print(f"Total Users: {len(unique_users)}")


# ====================================================
# 3. GRAPH CONSTRUCTION & STATIC FEATURES
# ====================================================
print("Building Graph...")
# Convert graph hashes to indices
graph_df['source_idx'] = graph_df['source'].map(user_to_idx)
graph_df['target_idx'] = graph_df['target'].map(user_to_idx)
# Filter edges where users exist in our data (just in case)
graph_df = graph_df.dropna(subset=['source_idx', 'target_idx'])
graph_df['source_idx'] = graph_df['source_idx'].astype(int)
graph_df['target_idx'] = graph_df['target_idx'].astype(int)

# Create Adjacency Matrix
n_users = len(unique_users)
# Undirected graph: add reverse edges
rows = np.concatenate([graph_df['source_idx'], graph_df['target_idx']])
cols = np.concatenate([graph_df['target_idx'], graph_df['source_idx']])
data = np.ones(len(rows))

adj_matrix = sp.coo_matrix((data, (rows, cols)), shape=(n_users, n_users))

# Feature: Degree (Number of connections)
degrees = np.array(adj_matrix.sum(axis=1)).flatten()
all_data['graph_degree'] = all_data['user_idx'].map(lambda x: degrees[x])

# Feature: PageRank (Fast approximation)
# We convert to nx graph only for PageRank, then delete
G = nx.from_scipy_sparse_array(adj_matrix)
pagerank = nx.pagerank(G, alpha=0.85, max_iter=50) # Reduced iter for speed
all_data['graph_pagerank'] = all_data['user_idx'].map(lambda x: pagerank.get(x, 0))
del G
gc.collect()


# ====================================================
# 4. LEAKAGE-FREE NEIGHBOR FEATURES
# ====================================================
print("Computing Neighbor Features (Leakage Free)...")

# We need to calculate "Ratio of Neighbors Who Cheat".
# BUT for Train set, we must not use the row's own label or future labels.
# We use K-Fold Target Encoding.

# Create a global label vector (NaN for test)
y_global = np.zeros(n_users) * np.nan
train_indices = all_data[all_data['is_train'] == 1].index
y_global[all_data.loc[train_indices, 'user_idx']] = all_data.loc[train_indices, 'is_cheating']

# Initialize feature column
all_data['neighbor_cheat_ratio'] = np.nan

# --- A. For Training Data: Use K-Fold to prevent leakage ---
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
train_df_only = all_data[all_data['is_train'] == 1]

for fold, (inner_train_idx, inner_val_idx) in enumerate(kf.split(train_df_only, train_df_only['is_cheating'].fillna(0))):
    # Map back to global indices
    global_train_idx = train_df_only.iloc[inner_train_idx].index
    global_val_idx = train_df_only.iloc[inner_val_idx].index
    
    # Create a label vector using ONLY the inner training set
    y_fold = np.zeros(n_users)
    # Fill known cheaters with 1, known clean with 0. 
    # Unlabeled (NaN) are treated as 0 risk for this calculation (conservative)
    # or you could ignore them. Here we treat NaN as 0.
    
    # Get user_indices for the inner train set
    train_user_idxs = all_data.loc[global_train_idx, 'user_idx'].values
    train_labels = all_data.loc[global_train_idx, 'is_cheating'].fillna(0).values
    
    y_fold[train_user_idxs] = train_labels
    
    # Compute sum of neighbor labels
    # A * y_fold gives sum of neighbors' labels for every node
    neighbor_sums = adj_matrix.dot(y_fold)
    
    # For the validation set users, compute the ratio
    val_user_idxs = all_data.loc[global_val_idx, 'user_idx'].values
    
    # ratio = sum_of_bad_neighbors / degree
    # Add epsilon to avoid div/0
    ratios = neighbor_sums[val_user_idxs] / (degrees[val_user_idxs] + 1e-6)
    
    all_data.loc[global_val_idx, 'neighbor_cheat_ratio'] = ratios

# --- B. For Test Data: Use Full Training Set ---
# Create label vector with ALL training data
y_full = np.zeros(n_users)
train_user_idxs = all_data.loc[train_indices, 'user_idx'].values
train_labels = all_data.loc[train_indices, 'is_cheating'].fillna(0).values # Treat unlabeled as 0
y_full[train_user_idxs] = train_labels

neighbor_sums_test = adj_matrix.dot(y_full)
test_indices = all_data[all_data['is_train'] == 0].index
test_user_idxs = all_data.loc[test_indices, 'user_idx'].values

test_ratios = neighbor_sums_test[test_user_idxs] / (degrees[test_user_idxs] + 1e-6)
all_data.loc[test_indices, 'neighbor_cheat_ratio'] = test_ratios

# Fill any remaining NaNs (isolated nodes)
all_data['neighbor_cheat_ratio'] = all_data['neighbor_cheat_ratio'].fillna(0)


# ====================================================
# 4b. PART 3: LABEL PROPAGATION (CORRECTED - NO LEAKAGE)
# ====================================================
print("Running Graph Label Propagation (Leakage-Free)...")

# 1. Normalize the Adjacency Matrix
# D_inv = 1 / degrees
d_inv = 1.0 / (degrees + 1e-6)
D_inv_mat = sp.diags(d_inv)
P = D_inv_mat.dot(adj_matrix)

# Initialize the new feature column
all_data['propagated_risk'] = np.nan
global_mean = train['is_cheating'].mean()

# --- A. For Training Data (Strict Out-of-Fold) ---
# We must NOT use a user's own label to calculate their risk score.
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
train_df = all_data[all_data['is_train'] == 1]

print("   Propagating for Train Folds...", end="")
for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df, train_df['is_cheating'].fillna(0))):
    # Get Global Indices for this fold
    global_tr = train_df.iloc[tr_idx].index
    global_val = train_df.iloc[val_idx].index
    
    # Initialize Graph with Global Mean
    Y_init = np.zeros(n_users) + global_mean
    
    # "Clamp" (Lock) the labels for the TRAINING part of this fold only
    tr_user_idxs = all_data.loc[global_tr, 'user_idx'].values
    tr_labels = all_data.loc[global_tr, 'is_cheating'].values
    Y_init[tr_user_idxs] = tr_labels
    
    # Propagate Risk (Iterative averaging)
    Y_curr = Y_init.copy()
    for i in range(20): # 20 hops
        Y_curr = P.dot(Y_curr)
        # Re-clamp training labels so they don't fade
        Y_curr[tr_user_idxs] = tr_labels
        
    # Read the result for the VALIDATION part (the ones we masked)
    val_user_idxs = all_data.loc[global_val, 'user_idx'].values
    all_data.loc[global_val, 'propagated_risk'] = Y_curr[val_user_idxs]
    print(f" {fold+1}", end="")
print(" Done.")

# --- B. For Test Data (Use Full Training Set) ---
print("   Propagating for Test Set...")
# Now we can use ALL training labels to help the test set
Y_init = np.zeros(n_users) + global_mean
train_indices = all_data[all_data['is_train'] == 1].index

tr_user_idxs = all_data.loc[train_indices, 'user_idx'].values
tr_labels = all_data.loc[train_indices, 'is_cheating'].values
Y_init[tr_user_idxs] = tr_labels

Y_curr = Y_init.copy()
for i in range(20):
    Y_curr = P.dot(Y_curr)
    Y_curr[tr_user_idxs] = tr_labels

# Assign to Test users
test_indices = all_data[all_data['is_train'] == 0].index
test_user_idxs = all_data.loc[test_indices, 'user_idx'].values
all_data.loc[test_indices, 'propagated_risk'] = Y_curr[test_user_idxs]

print("Propagation Complete. Feature 'propagated_risk' created safely.")


# ====================================================
# 5. MODEL TRAINING (CORRECTED)
# ====================================================

# 1. Define Base Features
feature_cols = [c for c in train.columns if c.startswith("feature_")]

# 2. Add Graph Features
# We manually add the new features we created
graph_features = ['graph_degree', 'graph_pagerank', 'neighbor_cheat_ratio', 'propagated_risk']

# Combine them
feature_cols = feature_cols + graph_features

print(f"Training on {len(feature_cols)} features.")
print(f"Verified Graph Features: {graph_features}")

# 3. Prepare Data
X = all_data[all_data['is_train'] == 1].dropna(subset=['is_cheating']).reset_index(drop=True)
y = X['is_cheating'].astype(int)
X_test = all_data[all_data['is_train'] == 0].reset_index(drop=True)

# 4. Training Loop
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

print("\nStarting Training...")
for fold, (train_idx, val_idx) in enumerate(cv.split(X, y)):
    X_tr, y_tr = X.loc[train_idx, feature_cols], y[train_idx]
    X_val, y_val = X.loc[val_idx, feature_cols], y[val_idx]
    
    # XGBoost
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        eval_metric='logloss',
        early_stopping_rounds=50
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    val_pred = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_pred
    test_preds += model.predict_proba(X_test[feature_cols])[:, 1] / N_FOLDS
    
    # Check if new feature is being used (Only need to print for first fold)
    if fold == 0:
        imp = pd.DataFrame({'f': feature_cols, 'imp': model.feature_importances_})
        risk_imp = imp[imp['f'] == 'propagated_risk']['imp'].values[0]
        print(f"   [DEBUG] 'propagated_risk' Importance in Fold 1: {risk_imp:.5f}")

    auc = roc_auc_score(y_val, val_pred)
    print(f"Fold {fold+1} AUC: {auc:.4f}")


# ====================================================
# 6. EVALUATION & OPTIMIZATION
# ====================================================
print("\n--- Final Evaluation ---")
total_auc = roc_auc_score(y, oof_preds)
print(f"Overall OOF AUC: {total_auc:.5f}")

print("Optimizing Thresholds for Mercor Cost...")
best_cost, t_low, t_high = calculate_mercor_cost(y, oof_preds)

print(f"Minimum Cost: ${best_cost:,.2f}")
print(f"Optimal Thresholds: Auto-Pass < {t_low:.4f} | Manual | Auto-Block > {t_high:.4f}")

# ====================================================
# 7. SUBMISSION
# ====================================================
submission = pd.DataFrame({
    'user_hash': X_test['user_hash'],
    'prediction': test_preds
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved successfully.")

