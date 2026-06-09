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


from  collections import defaultdict
import os 
import time
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import networkx as nx
from sklearn.model_selection import KFold
# Models

from xgboost import XGBClassifier, XGBRegressor,callback
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
import lightgbm as lgb
from catboost import CatBoostClassifier
warnings.filterwarnings("ignore")


class config:
    base_url="/kaggle/input/mercor-cheating-detection"
    train_csv =os.path.join(base_url,"train.csv")
    test_csv = os.path.join(base_url,"test.csv")
    social_graph = os.path.join(base_url,"social_graph.csv")
    sample_submission = os.path.join(base_url,"sample_submission.csv")


train=pd.read_csv(config.train_csv)
feature_cols = [c for c in train.columns if c.startswith("feature_")]
def train_stacked_primary(
    X, y, test_X, features,
    seed=42, n_folds=5,CATBOOST_AVAILABLE=None
):
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed
    )

    model_configs = [
        ("xgb", XGBClassifier(
            n_estimators=722,
            learning_rate=0.03,
            max_depth=9,
            subsample=0.9,
            colsample_bytree=0.8,
            gamma=0.7,
            min_child_weight=5,
            reg_alpha=2,
            reg_lambda=1.7,
            random_state=seed,
            verbosity=0
        )),
        ("lgbm", LGBMClassifier(
            n_estimators=761,
            learning_rate=0.03,
            num_leaves=105,
            max_depth=9,
            min_data_in_leaf=123,
            lambda_l1=0.02,
            lambda_l2=0.005,
            feature_fraction=0.47,
            bagging_fraction=0.81,
            bagging_freq=1,
            random_state=seed,
            verbose=-1
        ))
    ]

    if CATBOOST_AVAILABLE:
        model_configs.append(
            ("cat", CatBoostClassifier(
                iterations=700,
                learning_rate=0.03,
                depth=9,
                verbose=False,
                random_seed=seed
            ))
        )

    oof_preds = []
    test_preds = []

    for name, model in model_configs:
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(test_X))

        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx][features], X.iloc[va_idx][features]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            if name == "xgb":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    #callback=[callback.EarlyStopping(rounds=50)],
                    verbose=0
                )
            elif name == "lgbm":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="logloss",
                    callbacks=[early_stopping(50, verbose=False)]
                )
            else:  # cat
                model.fit(
                    X_tr, y_tr,
                    eval_set=(X_va, y_va),
                    early_stopping_rounds=50,
                    verbose=False
                )

            oof[va_idx] = model.predict_proba(X_va)[:, 1]
            test_pred += model.predict_proba(test_X[features])[:, 1] / n_folds

        oof_preds.append(oof)
        test_preds.append(test_pred)

    # Meta learner
    oof_stack = np.column_stack(oof_preds)
    test_stack = np.column_stack(test_preds)

    meta_oof = np.zeros(len(X))
    meta_test = np.zeros(len(test_X))

    for tr_idx, va_idx in skf.split(oof_stack, y):
        meta = LogisticRegression(
            random_state=seed,
            max_iter=1000
        )
        meta.fit(oof_stack[tr_idx], y.iloc[tr_idx])
        meta_oof[va_idx] = meta.predict_proba(oof_stack[va_idx])[:, 1]
        meta_test += meta.predict_proba(test_stack)[:, 1] / n_folds

    auc = roc_auc_score(y, meta_oof)
    print(f"Seed {seed} | Primary OOF AUC: {auc:.5f}")

    return meta_oof, meta_test


test=pd.read_csv(config.test_csv)
test.head()


all_users_feat =pd.concat([
    train[["user_hash"] + feature_cols], 
    test[["user_hash"] + feature_cols]]).drop_duplicates("user_hash").set_index("user_hash")


graph=pd.read_csv(config.social_graph,
                 names=["source", "target"])
graph.head()


reverse_graph =graph.rename(columns={"source": "target", "target": "source"})


full_edges =pd.concat([graph, reverse_graph], ignore_index=True).\
merge(all_users_feat, left_on="target", right_index=True, how="left")


full_edges.head()


agg_stats = full_edges.groupby("source")[feature_cols].agg(["mean", "std"])


agg_stats.columns


agg_stats.columns = [f"nbr_{c[0]}_{c[1]}" for c in agg_stats.columns]


neighbor_agg_cols = list(agg_stats.columns)
del all_users_feat, full_edges, reverse_graph



train=train.merge(agg_stats, left_on="user_hash", right_index=True, how="left")
test = test.merge(agg_stats, left_on="user_hash", right_index=True, how="left")


train[neighbor_agg_cols]=train[neighbor_agg_cols].fillna(0)
test[neighbor_agg_cols]=test[neighbor_agg_cols].fillna(0)


new_relative_cols = []
eps = 1e-5 

for col in feature_cols:
    nbr_mean = f"nbr_{col}_mean"
    if nbr_mean in train.columns:
        col_ratio = f"{col}_ratio"
        train[col_ratio] = train[col] / (train[nbr_mean] + eps)
        test[col_ratio]  = test[col]  / (test[nbr_mean] + eps)
        new_relative_cols.append(col_ratio)

        col_diff = f"{col}_diff"
        train[col_diff] = train[col] - train[nbr_mean]
        test[col_diff]  = test[col]  - test[nbr_mean]
        new_relative_cols.append(col_diff)


train.head()


test.head()


G=nx.from_pandas_edgelist(graph, "source", "target", create_using=nx.Graph())



degree_map =dict(G.degree())
comp_size_map = {}
for comp in nx.connected_components(G):
    size = len(comp)
    for node in comp:
        comp_size_map[node] = size


user_to_label=train.set_index("user_hash")["is_cheating"].dropna().to_dict()
neighbor_cheat_ratio = {}
num_labeled_neighbors = {}

for node in G.nodes():
    nbrs = list(G.neighbors(node))
    labeled_nbrs = [nbr for nbr in nbrs if nbr in user_to_label]
    if labeled_nbrs:
        cheat_ratio = np.mean([user_to_label[nbr] for nbr in labeled_nbrs])
        neighbor_cheat_ratio[node] = cheat_ratio
        num_labeled_neighbors[node] = len(labeled_nbrs)
    else:
        neighbor_cheat_ratio[node] = 0.0
        num_labeled_neighbors[node] = 0
        
pagerank_map = nx.pagerank(G, alpha=0.85)


for df in [train, test]:
    df["degree"] = df["user_hash"].map(degree_map).fillna(0)
    df["component_size"] = df["user_hash"].map(comp_size_map).fillna(1)
    df["neighbor_cheat_ratio"] = df["user_hash"].map(neighbor_cheat_ratio).fillna(0)
    df["num_labeled_neighbors"] = df["user_hash"].map(num_labeled_neighbors).fillna(0)
    df["pagerank"] = df["user_hash"].map(pagerank_map)

graph_feature_cols = [
    "degree",
    "component_size",
    "neighbor_cheat_ratio",
    "num_labeled_neighbors",
    "pagerank"
]

# -------------------------
# specia features
# -------------------------
for df in [train, test]:
    df["f012_is_too_fast"] = (df["feature_012"] > df["feature_012"].quantile(0.95)).astype(int)
    df["f012_bin"] = pd.qcut(df["feature_012"], q=5, duplicates='drop').cat.codes
    df["f015_bin"] = pd.qcut(df["feature_015"], q=7, duplicates='drop').cat.codes
    df["f016_is_high"] = (df["feature_016"] > df["feature_016"].median()).astype(int)
    df["f012_in_risky_time"] = df["feature_012"] * (1 - df["feature_014"])
    
special_features = [
    "f012_is_too_fast",
    "f012_bin",
    "f015_bin",
    "f016_is_high",
    "f012_in_risky_time"
]

feature_cols = [c for c in train.columns if c.startswith("feature_")]

all_features = feature_cols + graph_feature_cols + special_features + neighbor_agg_cols
base_features = feature_cols + graph_feature_cols
print(f"Total features: {len(all_features)}")


labeled =train[train["is_cheating"].notnull()].reset_index(drop=True)
X = labeled[all_features].reset_index(drop=True)
y = labeled["is_cheating"].astype(int).reset_index(drop=True)
test_X = test[all_features].reset_index(drop=True)
print(f"Labeled samples: {len(X)} | Positive rate: {y.mean():.3f}")


NFOLDS = 5

SEEDS = [42]

all_oof = []
all_test = []

for seed in SEEDS:
    print(f"\n========== Training seed {seed} ==========")
    oof, test_pred = train_stacked_primary(
        X=X,
        y=y,
        test_X=test_X,
        features=all_features,
        seed=seed,
        n_folds=NFOLDS,
        CATBOOST_AVAILABLE=True
    )
    all_oof.append(oof)
    all_test.append(test_pred)

final_oof = np.mean(all_oof, axis=0)
final_test = np.mean(all_test, axis=0)

final_oof = np.clip(final_oof, 0, 1)
final_test = np.clip(final_test, 0, 1)

print(f"\nFinal Multi-Seed OOF AUC: {roc_auc_score(y, final_oof):.5f}")


submission = pd.DataFrame({
    "user_hash": test["user_hash"],
    "prediction": final_test
})
submission.to_csv("submission3.csv", index=False)

print(f"\n✅ Saved final submission.")

