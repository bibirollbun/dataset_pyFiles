# =========================
# Mercor Cheating Detection
# LightGBM + OOF + Graph Features (2-stage)
# Generates: submission.csv with columns [user_hash, prediction]
# =========================
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

# -------------------------
# Paths
# -------------------------
DATA_PATH = "/kaggle/input/mercor-cheating-detection"

train_df = pd.read_csv(f"{DATA_PATH}/train.csv")
test_df  = pd.read_csv(f"{DATA_PATH}/test.csv")
graph_df = pd.read_csv(f"{DATA_PATH}/social_graph.csv")

print("train:", train_df.shape, "test:", test_df.shape, "graph:", graph_df.shape)

# -------------------------
# Features + labeled data
# -------------------------
features = [c for c in train_df.columns if c.startswith("feature_")]

labeled_df = train_df[train_df["is_cheating"].notna()].copy()
X_base = labeled_df[features].reset_index(drop=True)
y = labeled_df["is_cheating"].astype(int).values
X_test_base = test_df[features].reset_index(drop=True)

train_users = labeled_df["user_hash"].values
test_users  = test_df["user_hash"].values

# -------------------------
# Cost metric (vectorized)
# -------------------------
COST_FN = 600
COST_FP_BLOCK = 300
COST_FP_REVIEW = 150
COST_TP_REVIEW = 5

def compute_cost_vectorized(y_true, y_prob, t_low, t_high):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)

    auto_pass = y_prob < t_low
    manual    = (y_prob >= t_low) & (y_prob < t_high)
    auto_block= y_prob >= t_high

    cost = 0
    cost += np.sum(auto_pass & (y_true == 1)) * COST_FN
    cost += np.sum(manual & (y_true == 1)) * COST_TP_REVIEW
    cost += np.sum(manual & (y_true == 0)) * COST_FP_REVIEW
    cost += np.sum(auto_block & (y_true == 0)) * COST_FP_BLOCK
    return int(cost)

def find_best_thresholds(y_true, y_prob,
                         t_low_grid=None,
                         t_high_grid=None):
    if t_low_grid is None:
        t_low_grid = np.linspace(0.0, 0.4, 81)
    if t_high_grid is None:
        t_high_grid = np.linspace(0.4, 1.0, 121)

    best_cost = np.inf
    best_t_low, best_t_high = None, None

    for t_low in tqdm(t_low_grid, desc="t_low"):
        for t_high in t_high_grid:
            if t_low >= t_high:
                continue
            cost = compute_cost_vectorized(y_true, y_prob, t_low, t_high)
            if cost < best_cost:
                best_cost = cost
                best_t_low = t_low
                best_t_high = t_high

    return {
        "best_cost": float(best_cost),
        "best_t_low": float(best_t_low),
        "best_t_high": float(best_t_high),
        "leaderboard_score": -float(best_cost),
    }

# -------------------------
# Build adjacency (undirected)
# -------------------------
adj = defaultdict(set)
for u, v in zip(graph_df["user_a"].values, graph_df["user_b"].values):
    adj[u].add(v)
    adj[v].add(u)

# -------------------------
# Stage 1: Base OOF preds (for graph features)
# -------------------------
SEED = 42
N_FOLDS = 5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

oof_base = np.zeros(len(X_base))
test_base = np.zeros(len(X_test_base))

base_params = dict(
    n_estimators=2500,
    learning_rate=0.015,
    num_leaves=96,
    min_child_samples=200,
    min_split_gain=0.02,
    reg_lambda=2.0,
    reg_alpha=0.5,
    subsample=0.85,
    colsample_bytree=0.85,
    class_weight={0: 1, 1: 5},
)

print("\n=== Stage 1: Base OOF ===")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_base, y), 1):
    print(f"Fold {fold}/{N_FOLDS}")
    X_tr, X_val = X_base.iloc[tr_idx], X_base.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**base_params, random_state=SEED + fold)
    model.fit(X_tr, y_tr)

    oof_base[val_idx] = model.predict_proba(X_val)[:, 1]
    test_base += model.predict_proba(X_test_base)[:, 1] / N_FOLDS

print("\nStage 1 OOF cost (rough indicator):")
print(find_best_thresholds(y, oof_base))

plt.figure(figsize=(8,4))
plt.hist(oof_base, bins=100)
plt.title("Stage 1 OOF Probability Distribution (Base)")
plt.xlabel("p")
plt.ylabel("count")
plt.show()

# -------------------------
# Graph feature engineering (from base predictions)
# -------------------------
train_pred_map = dict(zip(train_users, oof_base))
test_pred_map  = dict(zip(test_users, test_base))

def graph_features_for_user(user, pred_map, adj_dict):
    nbrs = adj_dict.get(user, None)
    if not nbrs:
        return 0.0, 0.0, 0.0

    vals = [pred_map[n] for n in nbrs if n in pred_map]
    if len(vals) == 0:
        return 0.0, 0.0, float(len(nbrs))

    vals = np.asarray(vals, dtype=float)
    return float(vals.mean()), float(vals.max()), float(len(vals))

graph_cols = ["nbr_risk_mean", "nbr_risk_max", "nbr_known_cnt"]

train_graph_feats = np.array([graph_features_for_user(u, train_pred_map, adj) for u in train_users])
test_graph_feats  = np.array([graph_features_for_user(u, test_pred_map, adj) for u in test_users])

train_graph_df = pd.DataFrame(train_graph_feats, columns=graph_cols)
test_graph_df  = pd.DataFrame(test_graph_feats,  columns=graph_cols)

X_graph = pd.concat([X_base, train_graph_df], axis=1)
X_test_graph = pd.concat([X_test_base, test_graph_df], axis=1)

print("\nGraph-augmented shapes:", X_graph.shape, X_test_graph.shape)
print(X_graph[graph_cols].describe().T)

# -------------------------
# Stage 2: OOF on augmented features -> final test preds
# -------------------------
oof_final = np.zeros(len(X_graph))
test_final = np.zeros(len(X_test_graph))

# Slightly more regularized for stability (optional but usually helpful)
final_params = dict(
    n_estimators=3000,
    learning_rate=0.012,
    num_leaves=96,
    min_child_samples=250,
    min_split_gain=0.02,
    reg_lambda=3.0,
    reg_alpha=0.7,
    subsample=0.85,
    colsample_bytree=0.85,
    class_weight={0: 1, 1: 5},
)

print("\n=== Stage 2: Graph-Augmented OOF ===")
for fold, (tr_idx, val_idx) in enumerate(skf.split(X_graph, y), 1):
    print(f"Fold {fold}/{N_FOLDS}")
    X_tr, X_val = X_graph.iloc[tr_idx], X_graph.iloc[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMClassifier(**final_params, random_state=SEED + 100 + fold)
    model.fit(X_tr, y_tr)

    oof_final[val_idx] = model.predict_proba(X_val)[:, 1]
    test_final += model.predict_proba(X_test_graph)[:, 1] / N_FOLDS

print("\nStage 2 OOF cost (closer to LB):")
print(find_best_thresholds(y, oof_final))

plt.figure(figsize=(8,4))
plt.hist(oof_final, bins=100)
plt.title("Stage 2 OOF Probability Distribution (Graph-Augmented)")
plt.xlabel("p")
plt.ylabel("count")
plt.show()

# -------------------------
# Submission
# -------------------------
submission = pd.DataFrame({
    "user_hash": test_df["user_hash"].values,
    "prediction": test_final.astype(float)
})

# Hard checks
assert submission.shape[0] == test_df.shape[0]
assert submission["prediction"].between(0, 1).all()
assert submission.isna().sum().sum() == 0
assert submission["user_hash"].duplicated().sum() == 0

submission.to_csv("submission.csv", index=False)
print("\nSaved: submission.csv")
print(submission.head())




import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
# FORCE write submission file
out_path = "/kaggle/working/submission.csv"

submission = pd.DataFrame({
    "user_hash": test_df["user_hash"].values,
    "prediction": test_final.astype(float)
})

submission.to_csv(out_path, index=False)

print("Saved to:", out_path)
print("Exists:", os.path.exists(out_path))
print("File size (bytes):", os.path.getsize(out_path))

submission.head()





