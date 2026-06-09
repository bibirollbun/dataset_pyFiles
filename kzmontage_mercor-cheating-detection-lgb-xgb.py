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


import numpy as np
import pandas as pd
import gc
import warnings
warnings.filterwarnings("ignore")

import networkx as nx
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb
import lightgbm as lgb



BASE_PATH = "/kaggle/input/mercor-cheating-detection"

train = pd.read_csv(f"{BASE_PATH}/train.csv")
test  = pd.read_csv(f"{BASE_PATH}/test.csv")
graph = pd.read_csv(f"{BASE_PATH}/social_graph.csv")

print("Train:", train.shape)
print("Test :", test.shape)
print("Graph:", graph.shape)


feature_cols = [c for c in train.columns if c.startswith("feature_")]

# Build reverse edges for fast join
edges_rev = graph.rename(columns={"user_a": "user_b", "user_b": "user_a"})
edges_full = pd.concat([graph, edges_rev], ignore_index=True)

# Attach target user features
user_feats = pd.concat([
    train[["user_hash"] + feature_cols],
    test[["user_hash"] + feature_cols]
]).drop_duplicates("user_hash").set_index("user_hash")

edges_full = edges_full.merge(
    user_feats,
    left_on="user_b",
    right_index=True,
    how="left"
)

# Aggregate neighbor means
nbr_mean = edges_full.groupby("user_a")[feature_cols].mean()
nbr_mean.columns = [f"nbr_mean_{c}" for c in feature_cols]

# Merge back
train = train.merge(nbr_mean, left_on="user_hash", right_index=True, how="left")
test  = test.merge(nbr_mean, left_on="user_hash", right_index=True, how="left")

train.fillna(0, inplace=True)
test.fillna(0, inplace=True)

gc.collect()


eps = 1e-5
rel_features = []

for c in feature_cols:
    m = f"nbr_mean_{c}"
    
    ratio = f"{c}_rel_ratio"
    diff  = f"{c}_rel_diff"
    
    train[ratio] = train[c] / (train[m] + eps)
    test[ratio]  = test[c]  / (test[m] + eps)
    
    train[diff] = train[c] - train[m]
    test[diff]  = test[c]  - test[m]
    
    rel_features.extend([ratio, diff])


deg_a = graph["user_a"].value_counts()
deg_b = graph["user_b"].value_counts()

degree = deg_a.add(deg_b, fill_value=0)

train["degree"] = train["user_hash"].map(degree).fillna(0)
test["degree"]  = test["user_hash"].map(degree).fillna(0)


def fast_label_propagation(edges, seeds, all_users, n_iter=3):
    adj = {}
    for _, r in edges.iterrows():
        adj.setdefault(r["user_a"], []).append(r["user_b"])
        adj.setdefault(r["user_b"], []).append(r["user_a"])
    
    scores = {u: 0.5 for u in all_users}
    scores.update(seeds.to_dict())
    
    for _ in range(n_iter):
        new_scores = {}
        for u in all_users:
            if u in seeds:
                new_scores[u] = seeds[u]
            else:
                nbrs = adj.get(u, [])
                if nbrs:
                    new_scores[u] = 0.5 * scores[u] + 0.5 * np.mean([scores[n] for n in nbrs])
                else:
                    new_scores[u] = scores[u]
        scores = new_scores
    
    return pd.Series(scores)


labeled = train[train["is_cheating"].notna()].reset_index(drop=True)
y = labeled["is_cheating"].astype(int).values

high_conf_clean = train[train["high_conf_clean"] == 1]
clean_seeds = pd.Series(0.0, index=high_conf_clean["user_hash"])

all_users = pd.concat([
    graph["user_a"], graph["user_b"],
    train["user_hash"], test["user_hash"]
]).unique()

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_risk = pd.Series(index=labeled["user_hash"], dtype=float)

for fold, (tr, va) in enumerate(kf.split(labeled, y)):
    print(f"LP Fold {fold+1}/5")
    
    tr_df = labeled.iloc[tr]
    seeds = tr_df.set_index("user_hash")["is_cheating"]
    seeds = pd.concat([seeds, clean_seeds])
    
    scores = fast_label_propagation(graph, seeds, all_users, n_iter=3)
    
    val_users = labeled.iloc[va]["user_hash"]
    oof_risk.loc[val_users] = scores.loc[val_users]
    
    gc.collect()

train["risk_score"] = train["user_hash"].map(oof_risk).fillna(0.5)

# test risk
all_seeds = labeled.set_index("user_hash")["is_cheating"]
all_seeds = pd.concat([all_seeds, clean_seeds])

test_scores = fast_label_propagation(graph, all_seeds, all_users, n_iter=3)
test["risk_score"] = test["user_hash"].map(test_scores).fillna(0.5)


base_features = [c for c in train.columns if c.startswith("feature_")]

graph_features = [
    "degree",
    "risk_score"
]

all_features = base_features + graph_features + rel_features

print("Total features:", len(all_features))


labeled = train[train["is_cheating"].notna()].reset_index(drop=True)

X = labeled[all_features].values
y = labeled["is_cheating"].astype(int).values

X_test = test[all_features].values

print("Train shape:", X.shape)
print("Positive rate:", y.mean())


xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "max_depth": 6,
    "eta": 0.02,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 20,
    "lambda": 2.0,
    "alpha": 1.0,
    "seed": 42
}


NFOLDS = 5
kf = StratifiedKFold(n_splits=NFOLDS, shuffle=True, random_state=42)

oof_pred = np.zeros(len(X))
test_pred = np.zeros(len(X_test))

for fold, (tr, va) in enumerate(kf.split(X, y)):
    print(f"Fold {fold+1}/{NFOLDS}")
    
    dtr = xgb.DMatrix(X[tr], label=y[tr])
    dva = xgb.DMatrix(X[va])
    dte = xgb.DMatrix(X_test)
    
    model = xgb.train(
        xgb_params,
        dtr,
        num_boost_round=900,
        verbose_eval=False
    )
    
    oof_pred[va] = model.predict(dva)
    test_pred += model.predict(dte) / NFOLDS


auc = roc_auc_score(y, oof_pred)
print("OOF AUC:", auc)


def competition_cost(y_true, y_prob, t1, t2):
    cost = 0
    for yt, yp in zip(y_true, y_prob):
        if yp < t1:
            if yt == 1:
                cost += 600
        elif yp < t2:
            if yt == 0:
                cost += 150
            else:
                cost += 5
        else:
            if yt == 0:
                cost += 300
    return cost


best_cost = 1e18
best_t1, best_t2 = None, None

for t1 in np.linspace(0.05, 0.45, 40):
    for t2 in np.linspace(t1 + 0.05, 0.95, 40):
        c = competition_cost(y, oof_pred, t1, t2)
        if c < best_cost:
            best_cost = c
            best_t1, best_t2 = t1, t2

print("Best cost:", best_cost)
print("Best thresholds:", best_t1, best_t2)


submission = pd.DataFrame({
    "user_hash": test["user_hash"],
    "prediction": test_pred
})

submission.to_csv("submission.csv", index=False)
print("✅ submission.csv saved")




