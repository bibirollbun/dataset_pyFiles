# =========================================
# IMPORT LIBRARIES
# =========================================
import os
import warnings
import gc
from time import time

import numpy as np
import pandas as pd
import networkx as nx

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Models (CPU)
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

# =========================================
# LOAD DATA
# =========================================
INPUT_DIR = "/kaggle/input/mercor-cheating-detection"

train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test  = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
graph = pd.read_csv(os.path.join(INPUT_DIR, "social_graph.csv"),
                    names=["source", "target"])

feature_cols = [c for c in train.columns if c.startswith("feature_")]
print(f"Train: {train.shape}, Test: {test.shape}, Graph: {graph.shape}")

# =========================================
# FEATURE ENGINEERING: NEIGHBORS
# =========================================
t0 = time()

all_users_feat = pd.concat([
    train[["user_hash"] + feature_cols],
    test[["user_hash"] + feature_cols]
]).drop_duplicates("user_hash").set_index("user_hash")

rev_graph = graph.rename(columns={"source": "target", "target": "source"})
edges = pd.concat([graph, rev_graph], ignore_index=True)
edges = edges.merge(all_users_feat, left_on="target",
                    right_index=True, how="left")

agg = edges.groupby("source")[feature_cols].agg(["mean", "std"])
agg.columns = [f"nbr_{c[0]}_{c[1]}" for c in agg.columns]

del edges, rev_graph, all_users_feat
gc.collect()

train = train.merge(agg, left_on="user_hash", right_index=True, how="left")
test  = test.merge(agg, left_on="user_hash", right_index=True, how="left")

nbr_cols = list(agg.columns)
train[nbr_cols] = train[nbr_cols].fillna(0)
test[nbr_cols]  = test[nbr_cols].fillna(0)

for col in feature_cols:
    m = f"nbr_{col}_mean"
    if m in train.columns:
        train[f"{col}_ratio"] = train[col] / (train[m] + 1e-5)
        test[f"{col}_ratio"]  = test[col]  / (test[m] + 1e-5)
        train[f"{col}_diff"] = train[col] - train[m]
        test[f"{col}_diff"]  = test[col]  - test[m]

print(f"Neighbor features done in {time()-t0:.2f}s")

# =========================================
# GRAPH TOPOLOGY FEATURES
# =========================================
t0 = time()
G = nx.from_pandas_edgelist(graph, "source", "target")

degree = dict(G.degree())

try:
    pagerank = nx.pagerank(G, alpha=0.85)
except:
    pagerank = {n: 0.0 for n in G.nodes()}

component = {}
for c in nx.connected_components(G):
    size = len(c)
    for n in c:
        component[n] = size

for df in (train, test):
    df["degree"] = np.log1p(df["user_hash"].map(degree).fillna(0))
    df["pagerank"] = df["user_hash"].map(pagerank).fillna(0)
    df["component_size"] = np.log1p(
        df["user_hash"].map(component).fillna(1)
    )

print(f"Graph features done in {time()-t0:.2f}s")

# =========================================
# SCALING
# =========================================
cols = [c for c in train.columns
        if c in test.columns and c not in ["user_hash", "is_cheating"]]

num_cols = [c for c in cols
            if train[c].dtype in [np.float64, np.float32, np.int64]]

scaler = StandardScaler()
train[num_cols] = scaler.fit_transform(train[num_cols])
test[num_cols]  = scaler.transform(test[num_cols])

labeled = train[train["is_cheating"].notnull()]
X = labeled[num_cols]
y = labeled["is_cheating"].astype(int)
X_test = test[num_cols]

# =========================================
# CPU MODEL CONFIGS
# =========================================
lgb_params = {
    "n_estimators": 800,
    "learning_rate": 0.02,
    "num_leaves": 64,
    "n_jobs": -1
}

xgb_params = {
    "n_estimators": 800,
    "learning_rate": 0.02,
    "max_depth": 7,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "auc",
    "n_jobs": -1
}

cat_params = {
    "iterations": 800,
    "learning_rate": 0.02,
    "depth": 7,
    "verbose": 0
}

models = [
    ("lgbm", LGBMClassifier(**lgb_params)),
    ("xgb", XGBClassifier(**xgb_params))
]

if CATBOOST_AVAILABLE:
    models.append(("catboost", CatBoostClassifier(**cat_params)))

# =========================================
# TRAINING (CPU STACKING)
# =========================================
NFOLDS = 5
skf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

oof = np.zeros((len(X), len(models)))
test_pred = np.zeros((len(X_test), len(models)))

for i, (name, model) in enumerate(models):
    print(f"\nTraining {name} (CPU)")
    for fold, (tr, val) in enumerate(skf.split(X, y)):
        model.fit(X.iloc[tr], y.iloc[tr])
        oof[val, i] = model.predict_proba(X.iloc[val])[:, 1]
        test_pred[:, i] += model.predict_proba(X_test)[:, 1] / NFOLDS
        print(f"  Fold {fold+1} done")

    print(f"  OOF AUC [{name}]: {roc_auc_score(y, oof[:, i]):.5f}")

# =========================================
# META MODEL (CPU)
# =========================================
meta = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    n_jobs=-1
)

meta.fit(oof, y)
meta_test = meta.predict_proba(test_pred)[:, 1]

final_pred = 0.95 * meta_test + 0.05 * test_pred.mean(axis=1)

# =========================================
# SUBMISSION
# =========================================
submission = pd.DataFrame({
    "user_hash": test["user_hash"],
    "prediction": final_pred
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved (CPU ONLY)")


