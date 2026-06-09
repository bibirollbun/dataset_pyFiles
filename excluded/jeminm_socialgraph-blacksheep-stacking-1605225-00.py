import os
import subprocess
import warnings
import gc
from time import time

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Models
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except Exception:
    CATBOOST_AVAILABLE = False

# -------------------------
# GPU detection
# -------------------------
def list_gpus():
    try:
        res = subprocess.run(["nvidia-smi", "-L"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=3)
        out = res.stdout.strip()
        if not out:
            return []
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        return lines
    except Exception:
        return []

gpus = list_gpus()
NUM_GPUS = len(gpus)
print("Detected GPUs:", gpus)
print("NUM_GPUS =", NUM_GPUS)

# -------------------------
# Paths & load data
# -------------------------
INPUT = "/kaggle/input/mercor-cheating-detection"
train = pd.read_csv(os.path.join(INPUT, "train.csv"))
test  = pd.read_csv(os.path.join(INPUT, "test.csv"))
graph = pd.read_csv(os.path.join(INPUT, "social_graph.csv"), names=["source", "target"])

feature_cols = [c for c in train.columns if c.startswith("feature_")]
print("Base features:", len(feature_cols))
print("Train rows:", train.shape[0], "Test rows:", test.shape[0], "Graph rows:", graph.shape[0])

# -------------------------
# 1. Neighbor Feature Aggregation (Homophily)
# -------------------------
t0 = time()
print("Computing neighbor aggregations (Mean/Std)...")

# Combine train/test features for mapping
all_users_feat = pd.concat([
    train[["user_hash"] + feature_cols], 
    test[["user_hash"] + feature_cols]
]).drop_duplicates("user_hash").set_index("user_hash")

# Create bi-directional edge list for aggregation
rev_graph = graph.rename(columns={"source": "target", "target": "source"})
full_edges = pd.concat([graph, rev_graph], ignore_index=True)

# Merge features onto the 'target' (neighbor)
full_edges = full_edges.merge(all_users_feat, left_on="target", right_index=True, how="left")

# Group by 'source' (the user) and compute stats of their neighbors
agg_stats = full_edges.groupby("source")[feature_cols].agg(["mean", "std"])
agg_stats.columns = [f"nbr_{c[0]}_{c[1]}" for c in agg_stats.columns]

# Cleanup memory
del all_users_feat, full_edges, rev_graph
gc.collect()

# Merge back to train/test
train = train.merge(agg_stats, left_on="user_hash", right_index=True, how="left")
test = test.merge(agg_stats, left_on="user_hash", right_index=True, how="left")

# Fill NaNs (Isolates have 0 neighbors)
neighbor_cols = list(agg_stats.columns)
train[neighbor_cols] = train[neighbor_cols].fillna(0)
test[neighbor_cols]  = test[neighbor_cols].fillna(0)

print(f"Added {len(neighbor_cols)} neighbor features.")
print("Neighbor agg time:", round(time() - t0, 2), "s")

# -------------------------
# 1.5 LOCAL SUSPICION FEATURES (NEW ADDITION)
# -------------------------
# This captures the "Black Sheep" effect: users who are vastly different from their friends.
print("Computing Local Suspicion (Ratio/Diff)...")
for col in feature_cols:
    nbr_mean = f"nbr_{col}_mean"
    if nbr_mean in train.columns:
        # Ratio: Am I 10x better than my friends?
        train[f"{col}_ratio"] = train[col] / (train[nbr_mean] + 1e-5)
        test[f"{col}_ratio"]  = test[col]  / (test[nbr_mean] + 1e-5)
        
        # Diff: Am I +50 points above my friends?
        train[f"{col}_diff"] = train[col] - train[nbr_mean]
        test[f"{col}_diff"]  = test[col]  - test[nbr_mean]

print("Suspicion features added.")

# -------------------------
# 2. Build graph (CPU Fast Mode)
# -------------------------
t0 = time()
print("Building graph topology (CPU Fast Mode)...")
import networkx as nx 
G = nx.from_pandas_edgelist(graph, "source", "target", create_using=nx.Graph())

# 1. Degree
print("Computing Degree...")
degree_map = dict(G.degree())

# 2. PageRank (Optimized for speed)
print("Computing PageRank...")
try:
    pagerank_map = nx.pagerank(G, alpha=0.85, max_iter=20, tol=1e-4)
except Exception as e:
    print("pagerank failed:", e)
    pagerank_map = {n: 0.0 for n in G.nodes()}

# 3. Components
print("Computing Components...")
component_map = {}
for c in nx.connected_components(G):
    size = len(c)
    for node in c:
        component_map[node] = size

# 4. Clustering (Skipped for speed)
clustering_map = {n: 0.0 for n in G.nodes()}

# Map features to DataFrames
print("Mapping features to Train/Test...")
for df in (train, test):
    df["degree"] = df["user_hash"].map(degree_map).fillna(0.0)
    df["pagerank"] = df["user_hash"].map(pagerank_map).fillna(0.0)
    df["clustering"] = df["user_hash"].map(clustering_map).fillna(0.0)
    df["component_size"] = df["user_hash"].map(component_map).fillna(1.0)
    
    # Log Transforms
    df["degree"] = np.log1p(df["degree"])
    df["component_size"] = np.log1p(df["component_size"])

topo_cols = ["degree", "pagerank", "clustering", "component_size"]

print(f"Graph topology done.")
print("Time taken:", round(time() - t0, 2), "s")
 
# -------------------------
# 2.5 FEATURE PRUNING (NEW ADDITION)
# -------------------------
# Only keep features that exist in both sets and are numeric.
print("Pruning Features...")
cols = [c for c in train.columns if c in test.columns and c not in ["user_hash", "is_cheating"]]
# Numeric Only
all_features = [c for c in cols if train[c].dtype in [np.float64, np.float32, np.int64]]

print(f"Selected {len(all_features)} Robust Features for Training.")

# -------------------------
# Scale features
# -------------------------
print("Scaling features...")
scaler = StandardScaler()
train[all_features] = scaler.fit_transform(train[all_features])
test[all_features]  = scaler.transform(test[all_features])

# -------------------------
# Prepare labeled data
# -------------------------
labeled = train[train["is_cheating"].notnull()].copy()
X = labeled[all_features].reset_index(drop=True)
y = labeled["is_cheating"].astype(int).reset_index(drop=True)
test_X = test[all_features].reset_index(drop=True)

print("Labeled rows:", len(X), "Positive rate:", y.mean())

# -------------------------
# KFold
# -------------------------
NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# -------------------------
# Device assignment
# -------------------------
def assign_device_for_model(model_index):
    if NUM_GPUS == 0:
        return None
    return model_index % NUM_GPUS

# -------------------------
# Base models config
# -------------------------
use_gpu_flag = NUM_GPUS > 0

# LightGBM
lgb_params_gpu = {"n_estimators": 800, "learning_rate": 0.02, "num_leaves": 64}
lgb_params_cpu = {"n_estimators": 600, "learning_rate": 0.02, "num_leaves": 64, "n_jobs": -1}

# XGBoost
xgb_params_gpu = {
    "n_estimators": 800, "learning_rate": 0.02, "max_depth": 7, 
    "tree_method": "gpu_hist", "predictor": "gpu_predictor", "verbosity": 0
}
xgb_params_cpu = {
    "n_estimators": 600, "learning_rate": 0.02, "max_depth": 7, 
    "verbosity": 0
}

# CatBoost
cat_params_gpu = {
    "iterations": 800, "learning_rate": 0.02, "depth": 7, 
    "task_type": "GPU", "verbose": 0
}
cat_params_cpu = {
    "iterations": 600, "learning_rate": 0.02, "depth": 7, 
    "verbose": 0
}

base_sequence = []
base_sequence.append(("lgbm", lgb_params_gpu, lgb_params_cpu))
base_sequence.append(("xgb",  xgb_params_gpu, xgb_params_cpu))

if CATBOOST_AVAILABLE:
    base_sequence.append(("catboost", cat_params_gpu, cat_params_cpu))

print("Base models sequence:", [b[0] for b in base_sequence])

# -------------------------
# Storage for OOF/test
# -------------------------
n_models = len(base_sequence)
oof_mat = np.zeros((len(X), n_models))
test_mat = np.zeros((len(test_X), n_models))
model_names = []

# -------------------------
# Train loop
# -------------------------
for i, (name, gpu_params, cpu_params) in enumerate(base_sequence):
    model_names.append(name)
    dev = assign_device_for_model(i)
    print(f"\n=== Training {name} (Device: {dev}) ===")

    if dev is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(dev)
    else:
        if "CUDA_VISIBLE_DEVICES" in os.environ:
            del os.environ["CUDA_VISIBLE_DEVICES"]

    use_gpu_for_model = (dev is not None) and use_gpu_flag

    # Instantiate
    if name == "lgbm":
        if use_gpu_for_model:
            try:
                model = LGBMClassifier(**gpu_params)
            except:
                model = LGBMClassifier(**cpu_params)
        else:
            model = LGBMClassifier(**cpu_params)

    elif name == "xgb":
        params = gpu_params if use_gpu_for_model else cpu_params
        model = XGBClassifier(**params)

    elif name == "catboost":
        params = gpu_params if use_gpu_for_model else cpu_params
        try:
            model = CatBoostClassifier(**params)
        except:
            model = CatBoostClassifier(**cpu_params)

    # Fold Loop
    fold_idx = 0
    for tr_idx, va_idx in skf.split(X, y):
        fold_idx += 1
        print(f"  Fold {fold_idx}...", end="", flush=True)
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        try:
            if name == "lgbm":
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="auc", callbacks=[])
            elif name == "xgb":
                model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
            elif name == "catboost":
                model.fit(X_tr, y_tr, eval_set=(X_va, y_va), verbose=False)
            else:
                model.fit(X_tr, y_tr)
        except Exception as e:
            print(f"  FAILED (fallback CPU) {e}")
            break

        oof_mat[va_idx, i] = model.predict_proba(X_va)[:, 1]
        test_mat[:, i] += model.predict_proba(test_X)[:, 1] / NFOLDS
        print(" done.")

    score = roc_auc_score(y, oof_mat[:, i])
    print(f"  OOF AUC [{name}]: {score:.5f}")

# Reset Env
if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

# -------------------------
# Meta-model (STACKING)
# -------------------------
print("\nTraining Meta-Model (Stacking)...")
meta_gpu = 0 if NUM_GPUS > 0 else None
if meta_gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(meta_gpu)

meta_params = {
    "n_estimators": 1000, 
    "learning_rate": 0.03, 
    "max_depth": 4, 
    "subsample": 0.8,
    "colsample_bytree": 0.8
}
if NUM_GPUS > 0:
    meta_params.update({"tree_method": "gpu_hist", "predictor": "gpu_predictor", "verbosity": 0})

meta_model = XGBClassifier(**meta_params)
meta_model.fit(oof_mat, y)

meta_oof = meta_model.predict_proba(oof_mat)[:, 1]
meta_test = meta_model.predict_proba(test_mat)[:, 1]

if "CUDA_VISIBLE_DEVICES" in os.environ:
    del os.environ["CUDA_VISIBLE_DEVICES"]

print("Final Meta OOF AUC:", roc_auc_score(y, meta_oof))

# -------------------------
# Submission
# -------------------------
submission = pd.DataFrame({
    "user_hash": test["user_hash"].values,
    "prediction": meta_test
})
submission.to_csv("submission.csv", index=False)
print("Saved submission.csv")

