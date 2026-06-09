# %env NX_CUGRAPH_AUTOCONFIG=True


import os
import warnings
import gc
import logging
import sys
warnings.filterwarnings("ignore")
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', force=True)

import numpy as np
import pandas as pd
import time
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import networkx as nx
from sklearn.model_selection import KFold
# Models
from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor, early_stopping
import lightgbm as lgb
try:
    from catboost import CatBoostClassifier, CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

# -------------------------
# Load data
# -------------------------
INPUT = "/kaggle/input/mercor-cheating-detection"
train = pd.read_csv(os.path.join(INPUT, "train.csv"))
test = pd.read_csv(os.path.join(INPUT, "test.csv"))
graph = pd.read_csv(os.path.join(INPUT, "social_graph.csv"), names=["source", "target"])

feature_cols = [c for c in train.columns if c.startswith("feature_")]
print(f"Features: {len(feature_cols)} | Train: {len(train)} | Test: {len(test)} | Graph edges: {len(graph)}")


import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# -------------------------
# 1) Model: Logistic Regression (linear -> logit)
# -------------------------
class TorchLogReg(nn.Module):
    def __init__(self, n_features:int):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        out = F.sigmoid(self.linear(x))
        return out.squeeze(-1)


class TorchMLP(nn.Module):
    def __init__(self, n_features:int, hidden_layer_dim1:int=3):
        super().__init__()
        self.f1 = nn.Linear(n_features, hidden_layer_dim1)
        self.f2 = nn.Linear(hidden_layer_dim1, 1)

    def forward(self, x):
        x = F.gelu(self.f1(x))
        out = F.sigmoid(self.f2(x))
        return out.squeeze(-1)

# -------------------------
# 2) Soft gating (continuous region design) + cost-based expected loss
# -------------------------
class CostSensitive3RegionLoss(nn.Module):
    """
    Regions by thresholds t1 < t2:
      pass   : low risk
      review : medium risk
      block  : high risk

    Soft gating (recommended continuous form):
      g_pass   = sigmoid((t1 - p)/tau)
      g_block  = sigmoid((p - t2)/tau)
      g_review = sigmoid((p - t1)/tau) - sigmoid((p - t2)/tau)   >= 0

    Costs:
      FN (cheat passes) = 600
      FP in block       = 300
      FP in review      = 150
      TP needing review = 5
      correct pass/block= 0
    """
    def __init__(
        self,
        t1_init=0.30,
        t2_init=0.70,
        tau_init=0.05,
        learn_thresholds=True,
        costs=None,
        eps=1e-6,
        bce_weight:float|None=0.5,
    ):
        super().__init__()
        self.eps = eps

        if costs is None:
            costs = dict(
                fn_pass=600.0,     # y=1 and action=pass
                fp_review=150.0,   # y=0 and action=review
                fp_block=300.0,    # y=0 and action=block
                tp_review=5.0,     # y=1 and action=review
            )
        self.costs = costs

        # Parameterization to ensure 0 < t1 < t2 < 1
        # t1 = sigmoid(a)
        # t2 = t1 + (1 - t1) * sigmoid(b)  => always >= t1 and < 1
        a0 = torch.logit(torch.tensor(float(t1_init)).clamp(1e-4, 1-1e-4))
        # Solve rough b0 so that t2 matches t2_init
        t1_ = float(torch.sigmoid(a0))
        frac = (float(t2_init) - t1_) / max(1e-6, (1.0 - t1_))
        frac = min(max(frac, 1e-4), 1-1e-4)
        b0 = torch.logit(torch.tensor(frac))

        self.a = nn.Parameter(a0) if learn_thresholds else a0
        self.b = nn.Parameter(b0) if learn_thresholds else b0

        # tau must be > 0, use softplus param
        tau0 = torch.tensor(float(tau_init))
        # self.tau_unconstrained = nn.Parameter(torch.log(torch.exp(tau0) - 1.0))  # inverse softplus
        self.tau_unconstrained = torch.log(torch.exp(tau0) - 1.0)  # inverse softplus (Not trained)
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight

    def get_thresholds_and_tau(self):
        if isinstance(self.a, nn.Parameter):
            t1 = torch.sigmoid(self.a)
            frac = torch.sigmoid(self.b)
        else:
            t1 = torch.sigmoid(self.a)
            frac = torch.sigmoid(self.b)
        t2 = t1 + (1.0 - t1) * frac
        tau = F.softplus(self.tau_unconstrained) + self.eps
        return t1, t2, tau

    def forward(self, logits, y_true, separate_loss=False):
        """
        logits: (B,) raw logits
        y_true: (B,) float tensor in {0,1}
        """
        y = y_true.float()
        p = logits.clamp(self.eps, 1 - self.eps)

        t1, t2, tau = self.get_thresholds_and_tau()

        # Soft gating (continuous region design)
        g_pass = torch.sigmoid((t1 - p) / tau)
        s1 = torch.sigmoid((p - t1) / tau)
        s2 = torch.sigmoid((p - t2) / tau)
        g_review = (s1 - s2)  # guaranteed >= 0
        g_block = s2

        # (Optional) numerical safety: renormalize so sums are ~1
        g_sum = (g_pass + g_review + g_block).clamp_min(self.eps)
        g_pass, g_review, g_block = g_pass / g_sum, g_review / g_sum, g_block / g_sum

        # Expected cost per sample
        # pass: y=1 -> FN cost, y=0 -> 0
        cost_pass = self.costs["fn_pass"] * y

        # review: y=1 -> TP review cost, y=0 -> FP review cost
        cost_review = self.costs["tp_review"] * y + self.costs["fp_review"] * (1.0 - y)

        # block: y=0 -> FP block cost, y=1 -> 0
        cost_block = self.costs["fp_block"] * (1.0 - y)

        expected_cost = g_pass * cost_pass + g_review * cost_review + g_block * cost_block
        cost_loss = expected_cost.mean() / 600.  # normalization

        # bce loss
        bce_loss = self.bce(p, y)
        
        if self.bce_weight is not None:
            # Sum loss
            loss = self.bce_weight * bce_loss + (1.0 - self.bce_weight) * cost_loss
        else: 
            loss = cost_loss
        
        if separate_loss:
            return loss, bce_loss, cost_loss
        
        return loss


def pytorch_train(
    model, 
    X_train, 
    y_train, 
    X_val, 
    y_val, 
    loss_fn, 
    optimizer, 
    epochs=50, 
    device="cpu", 
    patience_limit=10,
):
    model = model.to(device).train()
    loss_fn = loss_fn.to(device)
    # tensors
    X_train = torch.as_tensor(np.array(X_train), dtype=torch.float32)
    y_train = torch.as_tensor(np.array(y_train), dtype=torch.float32)
    X_val = torch.as_tensor(np.array(X_val), dtype=torch.float32)
    y_val = torch.as_tensor(np.array(y_val), dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=8192,
        shuffle=True,
        drop_last=False
    )
    best_loss = 10 ** 9
    for epoch in range(1, epochs+1):
        total_loss = 0.
        n=0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * inputs.size(0)
            n += inputs.size(0)

        with torch.no_grad():
            outputs = model(X_val.to(device))
            val_loss, val_bce_loss, val_cost_loss = loss_fn(outputs, y_val.to(device), separate_loss=True)
        logging.debug(
            f"epoch {epoch}: train_loss={total_loss/n:.5f}, " 
            + f"val_loss={val_loss:.5f}, "
            + f"val_bce_loss={val_bce_loss:.5f}, "
            + f"val_cost_loss={val_cost_loss:.5f}"
        )

        ### Early stopping logic ###
        if val_loss > best_loss:
            patience_check += 1
            if patience_check >= patience_limit:
                logging.debug("Stopped training by early stopping condition")
                break
        else:
            best_loss = val_loss
            patience_check = 0
        
    return model.to("cpu")


# from sklearn.linear_model import LogisticRegression

def train_stacked_primary(
    X, y, test_X, features,
    seed=42, n_folds=5, device="cpu",
    meta_lr=5e-2, meta_weight_decay=1e-2, meta_epochs=50, meta_bce_weight=0.5,
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
            verbosity=0,
            early_stopping_rounds=50,
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
        logging.info(f"Training {name} model...")
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(test_X))

        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx][features], X.iloc[va_idx][features]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

            if name == "xgb":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    verbose=0
                )
            elif name == "lgbm":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    eval_metric="logloss",
                    callbacks=[early_stopping(50, verbose=False)]
                )
            elif name == "cat":
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

    logging.info("Training meta model...(logistic regresssion with soft cost loss)")
    for tr_idx, va_idx in skf.split(oof_stack, y):
        # meta = LogisticRegression(
        #     random_state=seed,
        #     max_iter=1000
        # )
        # meta.fit(oof_stack[tr_idx], y.iloc[tr_idx])
        # meta_oof[va_idx] = meta.predict_proba(oof_stack[va_idx])[:, 1]
        # meta_test += meta.predict_proba(test_stack)[:, 1] / n_folds
        
        torch.manual_seed(seed)
        meta = TorchLogReg(oof_stack.shape[-1])
        # meta = TorchMLP(oof_stack.shape[-1])  # Nonlinearity doesn't help 
        loss_fn = CostSensitive3RegionLoss(
            t1_init=0.30,
            t2_init=0.70,
            tau_init=0.05,
            learn_thresholds=True,
            bce_weight=meta_bce_weight,
        )
        # Train model params and thresholds together
        params = list(meta.parameters()) + [p for p in loss_fn.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=meta_lr, weight_decay=meta_weight_decay)
        
        meta = pytorch_train(
            model=meta, 
            X_train=oof_stack[tr_idx], 
            y_train=y.iloc[tr_idx], 
            X_val=oof_stack[va_idx],
            y_val=y.iloc[va_idx],
            loss_fn=loss_fn, 
            optimizer=opt,
            epochs=meta_epochs,
            device=device,
        )

        with torch.no_grad():
            meta_oof[va_idx] = meta(torch.as_tensor(oof_stack[va_idx], dtype=torch.float32))
            meta_test += np.array(meta(torch.as_tensor(test_stack, dtype=torch.float32))) / n_folds        

    auc = roc_auc_score(y, meta_oof)
    logging.info(f"Seed {seed} | Primary OOF AUC: {auc:.5f}")

    return meta_oof, meta_test


def run_fast_label_propagation(social_graph, seeds, all_users, n_iter=3):
    """
    Fast label propagation with minimal iterations.
    """
    # Create adjacency dictionary (much faster than DataFrame operations)
    adj = {}
    for _, row in social_graph.iterrows():
        adj.setdefault(row['source'], []).append(row['target'])
        adj.setdefault(row['target'], []).append(row['source'])
    
    # Initialize scores
    scores = {user: 0.5 for user in all_users}
    scores.update(seeds.to_dict())
    
    # Run propagation
    for _ in range(n_iter):
        new_scores = {}
        for user in all_users:
            if user in seeds:
                new_scores[user] = seeds[user]
            else:
                neighbors = adj.get(user, [])
                if neighbors:
                    neighbor_scores = [scores[n] for n in neighbors]
                    new_scores[user] = 0.5 * scores[user] + 0.5 * np.mean(neighbor_scores)
                else:
                    new_scores[user] = scores[user]
        scores = new_scores
    
    return pd.Series(scores)


t0_fe = time.time()

all_users_feat = pd.concat([
    train[["user_hash"] + feature_cols], 
    test[["user_hash"] + feature_cols]]).drop_duplicates("user_hash").set_index("user_hash")

rev_graph = graph.rename(columns={"source": "target", "target": "source"})
full_edges = pd.concat([graph, rev_graph], ignore_index=True)

full_edges = full_edges.merge(all_users_feat, left_on="target", right_index=True, how="left")

# 피쳐 추가: 이웃의 피쳐별 mean, std
agg_stats = full_edges.groupby("source")[feature_cols].agg(["mean", "std"])
agg_stats.columns = [f"nbr_{c[0]}_{c[1]}" for c in agg_stats.columns]
neighbor_agg_cols = list(agg_stats.columns)

del all_users_feat, full_edges, rev_graph
gc.collect()

train = train.merge(agg_stats, left_on="user_hash", right_index=True, how="left")
test = test.merge(agg_stats, left_on="user_hash", right_index=True, how="left")

train[neighbor_agg_cols] = train[neighbor_agg_cols].fillna(0)
test[neighbor_agg_cols]  = test[neighbor_agg_cols].fillna(0)


logging.info("Creating relative features...")
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

# -------------------------
# Build graph
# -------------------------
G = nx.from_pandas_edgelist(graph, "source", "target", create_using=nx.Graph())

# -------------------------
# Compute graph feature dictionaries
# -------------------------
degree_map = dict(G.degree())

comp_size_map = {}
for comp in nx.connected_components(G):
    size = len(comp)
    for node in comp:
        comp_size_map[node] = size

# Get labeled user -> cheating status from FULL train (including NaNs)
user_to_label = train.set_index("user_hash")["is_cheating"].dropna().to_dict()

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

logging.info("Running pagerank...")
pagerank_map = nx.pagerank(G, alpha=0.85)

# logging.info("Counting triangles...")
# triangle_map = nx.triangles(G)

# logging.info("Running clustering coefficient...")
# clustering_coeff_map = nx.clustering(G)

# logging.info("Running eigen centrality...")  # Takes too long
# eigen_centrality_map = nx.eigenvector_centrality(G, max_iter=100, tol=1e-04)

# logging.info("Running betweenness centrality...")  # Used nx-cugraph to extract this feature.
# betweenness_centrality_map = nx.betweenness_centrality(G, k=100)


high_conf_clean = train[train['high_conf_clean'] == 1].set_index('user_hash')
clean_seeds = pd.Series(0.0, index=high_conf_clean.index)

labeled_train = train[train['is_cheating'].notna()]
y = labeled_train['is_cheating'].values

oof_risk = pd.Series(index=labeled_train['user_hash'], dtype=float)
all_users = pd.concat([graph['source'], graph['target'], 
                          train['user_hash'], test['user_hash']]).unique()

kf_feat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
for fold, (train_idx, val_idx) in enumerate(kf_feat.split(labeled_train, y)):
    logging.info(f"LP Fold {fold+1}/3...")
    
    train_fold = labeled_train.iloc[train_idx]
    train_seeds = train_fold.set_index('user_hash')['is_cheating']
    seeds = pd.concat([train_seeds, clean_seeds])
    
    # Run fast LP with only 3 iterations
    scores = run_fast_label_propagation(graph, seeds, all_users, n_iter=3)
    
    val_users = labeled_train.iloc[val_idx]['user_hash']
    oof_risk.loc[val_users] = scores.loc[val_users]
    
    gc.collect()

train['risk_score'] = train['user_hash'].map(oof_risk).fillna(0.5)

# Test scores using all labeled data
print("  Generating test risk scores...")
all_labeled_seeds = labeled_train.set_index('user_hash')['is_cheating']
seeds = pd.concat([all_labeled_seeds, clean_seeds])
test_scores = run_fast_label_propagation(graph, seeds, all_users, n_iter=3)
test['risk_score'] = test['user_hash'].map(test_scores).fillna(0.5)

# -------------------------
# Add graph features to train and test
# -------------------------
for df in [train, test]:
    df["degree"] = df["user_hash"].map(degree_map).fillna(0)
    df["component_size"] = df["user_hash"].map(comp_size_map).fillna(1)
    df["neighbor_cheat_ratio"] = df["user_hash"].map(neighbor_cheat_ratio).fillna(0)
    df["num_labeled_neighbors"] = df["user_hash"].map(num_labeled_neighbors).fillna(0)
    df["pagerank"] = df["user_hash"].map(pagerank_map)
    # df["triangle"] = df["user_hash"].map(triangle_map).fillna(0)
    # df["clustering_coefficient"] = df["user_hash"].map(clustering_coeff_map).fillna(0.)
    # df["eigen_centrality"] = df["user_hash"].map(eigen_centrality_map).fillna(0.)
    # df["betweenness_centrality"] = df["user_hash"].map(betweenness_centrality_map).fillna(0.)

graph_feature_cols = [
    'risk_score',
    "degree",
    "component_size",
    "neighbor_cheat_ratio",
    "num_labeled_neighbors",
    "pagerank",
    # "triangle",
    # "clustering_coefficient",
    # "eigen_centrality",
    # "betweenness_centrality",
]

# -------------------------
# 피쳐 추가: specia features
# -------------------------
for df in [train, test]:
    df["f012_is_too_fast"] = (df["feature_012"] > df["feature_012"].quantile(0.95)).astype(int)
    df["f012_bin"] = pd.qcut(df["feature_012"], q=5, duplicates='drop').cat.codes
    df["f015_bin"] = pd.qcut(df["feature_015"], q=7, duplicates='drop').cat.codes
    df["f016_is_high"] = (df["feature_016"] > df["feature_016"].median()).astype(int)
    df["f012_in_risky_time"] = df["feature_012"] * (1 - df["feature_014"])    
    df["danger_f004"] = df["feature_004"].isin([0.0, 3.0, np.nan]).astype(int)
    df["missing_count"] = df[feature_cols].isin([np.nan]).sum(axis=1)
    
special_features = [
    "f012_is_too_fast",
    "f012_bin",
    "f015_bin",
    "f016_is_high",
    "f012_in_risky_time",
    "danger_f004",
    "missing_count"
]

feature_cols = [c for c in train.columns if c.startswith("feature_")]

all_features = (
    feature_cols 
    + graph_feature_cols 
    + special_features 
    + neighbor_agg_cols
)
base_features = (
    feature_cols 
    + graph_feature_cols
)
print(f"Total features: {len(all_features)}")

# -------------------------
# Scale features
# -------------------------
# scaler = StandardScaler()
# train[all_features] = scaler.fit_transform(train[all_features])
# test[all_features] = scaler.transform(test[all_features])

# -------------------------
# NOW extract labeled data (after features are added!)
# -------------------------
labeled = train[train["is_cheating"].notnull()].reset_index(drop=True)
X = labeled[all_features].reset_index(drop=True)
y = labeled["is_cheating"].astype(int).reset_index(drop=True)
test_X = test[all_features].reset_index(drop=True)
print(f"Labeled samples: {len(X)} | Positive rate: {y.mean():.3f}")


# -------------------------
# CV setup
# -------------------------
NFOLDS = 5

# SEEDS = [42, 52, 62]  # version 2
SEEDS = [42]

all_oof = []
all_test = []

for seed in SEEDS:
    print(f"\n========== Training seed {seed} ==========")
    oof, test_pred = train_stacked_primary(
        X=X,
        y=y,
        # X=X.iloc[:1000],  # debug
        # y=y[:1000], # debug
        test_X=test_X,
        features=all_features,
        seed=seed,
        n_folds=NFOLDS,
        device="cpu",  # cuda or cpu
        meta_lr=5e-2,
        meta_epochs=1000,
        meta_bce_weight=None,
    )
    all_oof.append(oof)
    all_test.append(test_pred)

final_oof = np.mean(all_oof, axis=0)
final_test = np.mean(all_test, axis=0)

final_oof = np.clip(final_oof, 0, 1)
final_test = np.clip(final_test, 0, 1)

print(f"\nFinal Multi-Seed OOF AUC: {roc_auc_score(y, final_oof):.5f}")


# ==================================================
# FINAL: Use Primary Stacked Model Only
# ==================================================
submission_pdf = pd.DataFrame({
    "user_hash": test["user_hash"],
    "prediction": final_test
})
submission_pdf.to_csv("submission_before_propagation.csv", index=False)

print(f"\n✅ Saved final submission.")


import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.preprocessing import normalize
import gc

# ==========================================
# BALANCED CONFIGURATION
# ==========================================
SOCIAL_PATH = '/kaggle/input/mercor-cheating-detection/social_graph.csv'
# TEST_PRED_PATH = '/kaggle/input/submission-lgb-xgb-cat-softcostloss-v3/submission_lgb_xgb_cat_SoftCostLoss.csv' 

# TUNING: THE GOLDILOCKS ZONE
ALPHA = 0.15       
MAX_ITER = 50

def run_balanced_LP(df_test):
    print("1. Reading Submission...")
    # df_test = pd.read_csv(TEST_PRED_PATH)
    test_user_set = set(df_test['user_hash'])

    print("2. Reading Social Graph (Test <-> Test)...")

    df_social = pd.read_csv(SOCIAL_PATH)
    mask = df_social['user_a'].isin(test_user_set) & df_social['user_b'].isin(test_user_set)
    df_edges = df_social[mask].copy()

    # Build Adjacency
    user_to_idx = {u: i for i, u in enumerate(df_test['user_hash'])}
    
    row = df_edges['user_a'].map(user_to_idx).values
    col = df_edges['user_b'].map(user_to_idx).values
    data = np.ones(len(row))
    
    num_users = len(df_test)
    
    adj_matrix = sp.coo_matrix((data, (row, col)), shape=(num_users, num_users))
    adj_matrix = adj_matrix + adj_matrix.T
    
    # L1 Normalize (Row Stochastic)
    adj_norm = normalize(adj_matrix, norm='l1', axis=1)
    
    # Clean up
    del df_social, df_edges, row, col, data
    gc.collect()

    print(f"3. Running Balanced LP (Alpha={ALPHA}, Iter={MAX_ITER})...")
    
    y_init = df_test['prediction'].values
    y_current = y_init.copy()

    node_degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    has_neighbor_mask = node_degrees > 0

    # Iterative Propagation
    for i in range(MAX_ITER):
        neighbor_avg = adj_norm.dot(y_current)
        y_current[has_neighbor_mask] = (
            ALPHA * neighbor_avg[has_neighbor_mask] + 
            (1 - ALPHA) * y_init[has_neighbor_mask]
        )

        if (i+1) % 10 == 0:
            print(f"   Iteration {i+1}/{MAX_ITER} complete")
        
    submission = df_test.copy()
    submission['prediction'] = y_current
    return submission

if __name__ == "__main__":
    propagated_submission = run_balanced_LP(submission_pdf)
    propagated_submission.to_csv('submission.csv', index=False)
    print("Balanced Submission Saved!")

    propagated_train_submission = run_balanced_LP(pd.DataFrame({
        "user_hash": labeled["user_hash"],
        "prediction": final_oof
    }))


"""
Custom Cost-Based Evaluation Metric for Mercor's Kaggle Cheating Detection Competition
"""

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype

def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
    """
    Calculate cost-based score for cheating detection.
    
    Finds optimal decision thresholds that divide predictions into three regions:
    - Auto-pass (low confidence cheating): $0 if correct, $600 if missed cheating
    - Manual review (medium confidence cheating): $5 if cheating, $150 if wasted on legitimate user
    - Auto-block (high confidence cheating): $0 if correct, $300 if wrongly blocked legitimate user

    """
    merged = solution.merge(submission, on=row_id_column_name, how='inner')
    
    if merged.empty:
        raise ValueError("No matching IDs between solution and submission")
    
    if 'prediction' not in merged.columns:
        raise ValueError("Submission must have 'prediction' column")
    
    target_candidates = [c for c in merged.columns 
                        if c not in [row_id_column_name, 'prediction', 'Usage']]
    
    if len(target_candidates) != 1:
        raise ValueError(f"Cannot identify target column. Found: {target_candidates}")
    
    target_col = target_candidates[0]
    
    y_true = merged[target_col].values
    y_pred = merged['prediction'].values
    
    if not is_numeric_dtype(merged['prediction']):
        raise ValueError("Predictions must be numeric")
    if not np.all((y_pred >= 0) & (y_pred <= 1)):
        raise ValueError("Predictions must be between 0 and 1")
    
    y_true = np.array(y_true, copy=False)
    sort_order = np.argsort(y_pred)
    y_true = y_true[sort_order]
    
    c1 = (y_true * 745 - 150).cumsum()
    
    c2 = (y_true * 155 - 150).cumsum()
    
    total_cost = c1.min() + c2.min() + (y_true == 0).sum() * 300
    
    return float(-total_cost)


# 전체 train에 대한 비용 스코어
train_solution = pd.DataFrame({"user_hash":labeled["user_hash"], "answer":y}) 
train_submission = pd.DataFrame({"user_hash":labeled["user_hash"], "prediction":final_oof})
score(solution=train_solution, submission=train_submission, row_id_column_name="user_hash")


# 전체 train에 대한 비용 스코어 (after propagation)
train_solution = pd.DataFrame({"user_hash":labeled["user_hash"], "answer":y}) 
score(solution=train_solution, submission=propagated_train_submission, row_id_column_name="user_hash")


# 기존과 xgboost 버전이 달라짐. (2.0.3 -> 3.1.0)
# !pip list | grep xgboost


# oof_stack = np.random.rand(len(y), 5)
# tr_idx = list(range(10000))
# va_idx = list(range(10000, 11000))

# torch.manual_seed(seed)
# meta = TorchLogReg(oof_stack.shape[-1])
# loss_fn = CostSensitive3RegionLoss(
#     t1_init=0.30,
#     t2_init=0.70,
#     tau_init=0.05,
#     learn_thresholds=True,
# )
# # Train model params and thresholds together
# params = list(meta.parameters()) + [p for p in loss_fn.parameters() if p.requires_grad]
# opt = torch.optim.AdamW(params, lr=1e-2, weight_decay=1e-2)

# meta, loss_fn = pytorch_train(
#     model=meta, 
#     X_train=oof_stack[tr_idx], 
#     y_train=y.iloc[tr_idx], 
#     X_val=oof_stack[va_idx],
#     y_val=y.iloc[va_idx],
#     loss_fn=loss_fn, 
#     optimizer=opt,
#     epochs=50,
#     device="cuda",
# )

