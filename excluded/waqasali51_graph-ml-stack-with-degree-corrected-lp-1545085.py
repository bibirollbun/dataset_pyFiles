%%capture
!pip uninstall -y xgboost xgboost-gpu
!pip install xgboost==2.0.3


import os
import warnings
import gc
import time
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import networkx as nx
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

# Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', 100)

# -------------------------
# Configuration
# -------------------------
INPUT = "/kaggle/input/mercor-cheating-detection"
SEED = 42
NFOLDS = 5
LP_ITERATIONS = 3
BALANCED_LP_ITER = 50
ALPHA_LP = 0.15

print("ğŸŒ± Mercor Cheating Detection Solution - Graph-ML Stack with Degree Corrected LP")
print("="*50)


# -------------------------
# Load Data
# -------------------------
print("ğŸ“¥ Loading data...")
t0 = time.time()
train = pd.read_csv(os.path.join(INPUT, "train.csv"))
test = pd.read_csv(os.path.join(INPUT, "test.csv"))
graph = pd.read_csv(os.path.join(INPUT, "social_graph.csv"), names=["source", "target"])
print(f"âœ… Data loaded in {time.time()-t0:.2f}s")
print(f"ğŸ“Š Train: {len(train)} | Test: {len(test)} | Graph edges: {len(graph)}")

feature_cols = [c for c in train.columns if c.startswith("feature_")]
print(f"ğŸ“ˆ Features: {len(feature_cols)}")


# -------------------------
# Feature Engineering Functions
# -------------------------
def create_neighbor_aggregates(train_df, test_df, graph_df, feature_cols):
    """Create neighbor aggregate features"""
    print("ğŸ”§ Creating neighbor aggregates...")
    t0 = time.time()
    
    all_users_feat = pd.concat([
        train_df[["user_hash"] + feature_cols], 
        test_df[["user_hash"] + feature_cols]
    ]).drop_duplicates("user_hash").set_index("user_hash")
    
    rev_graph = graph_df.rename(columns={"source": "target", "target": "source"})
    full_edges = pd.concat([graph_df, rev_graph], ignore_index=True)
    full_edges = full_edges.merge(all_users_feat, left_on="target", right_index=True, how="left")
    
    agg_stats = full_edges.groupby("source")[feature_cols].agg(["mean", "std"])
    agg_stats.columns = [f"nbr_{c[0]}_{c[1]}" for c in agg_stats.columns]
    neighbor_agg_cols = list(agg_stats.columns)
    
    train_df = train_df.merge(agg_stats, left_on="user_hash", right_index=True, how="left")
    test_df = test_df.merge(agg_stats, left_on="user_hash", right_index=True, how="left")
    
    train_df[neighbor_agg_cols] = train_df[neighbor_agg_cols].fillna(0)
    test_df[neighbor_agg_cols] = test_df[neighbor_agg_cols].fillna(0)
    
    # Relative features
    new_relative_cols = []
    eps = 1e-5 
    for col in feature_cols:
        nbr_mean = f"nbr_{col}_mean"
        if nbr_mean in train_df.columns:
            col_ratio = f"{col}_ratio"
            train_df[col_ratio] = train_df[col] / (train_df[nbr_mean] + eps)
            test_df[col_ratio] = test_df[col] / (test_df[nbr_mean] + eps)
            new_relative_cols.append(col_ratio)

            col_diff = f"{col}_diff"
            train_df[col_diff] = train_df[col] - train_df[nbr_mean]
            test_df[col_diff] = test_df[col] - test_df[nbr_mean]
            new_relative_cols.append(col_diff)
    
    print(f"âœ… Neighbor aggregates created in {time.time()-t0:.2f}s")
    return train_df, test_df, neighbor_agg_cols, new_relative_cols


def fold_neighbor_cheat_ratio(G, train_df, train_idx):
    """Compute neighbor cheating ratio using only training indices (no leakage)."""
    fold_users = set(train_df.iloc[train_idx]["user_hash"])
    labels = train_df.iloc[train_idx].set_index("user_hash")["is_cheating"]

    ratio = {}
    for node in G.nodes():
        try:
            nbrs = [nbr for nbr in G.neighbors(node) if nbr in fold_users]
            if nbrs:
                ratio[node] = labels.loc[nbrs].mean()
            else:
                ratio[node] = 0.0
        except (nx.NetworkXError, KeyError):
            # Node not in graph or labels
            ratio[node] = 0.0
    return ratio


def create_graph_features(train_df, test_df, graph_df, G=None, neighbor_cheat_map=None):
    """Create graph-based features"""
    print("ğŸ•¸ï¸� Creating graph features...")
    t0 = time.time()
    
    if G is None:
        G = nx.from_pandas_edgelist(graph_df, "source", "target", create_using=nx.Graph())
    
    # Basic graph features
    degree_map = dict(G.degree())
    pagerank_map = nx.pagerank(G, alpha=0.85)
    
    # Component sizes
    comp_size_map = {}
    for comp in nx.connected_components(G):
        size = len(comp)
        for node in comp:
            comp_size_map[node] = size
    
    # Apply to dataframes
    for df in [train_df, test_df]:
        # Skip empty dataframes
        if len(df) == 0:
            continue
            
        df["degree"] = df["user_hash"].map(degree_map).fillna(0)
        df["component_size"] = df["user_hash"].map(comp_size_map).fillna(1)
        df["pagerank"] = df["user_hash"].map(pagerank_map).fillna(0)
        
        # Use precomputed neighbor cheat ratio if provided (for CV folds)
        if neighbor_cheat_map is not None:
            df["neighbor_cheat_ratio"] = df["user_hash"].map(neighbor_cheat_map).fillna(0)
            # Compute num_labeled_neighbors safely
            num_labeled_neighbors = {}
            for user in df["user_hash"]:
                try:
                    if user in G:
                        nbrs = [nbr for nbr in G.neighbors(user) if nbr in neighbor_cheat_map]
                        num_labeled_neighbors[user] = len(nbrs)
                    else:
                        num_labeled_neighbors[user] = 0
                except (nx.NetworkXError, KeyError):
                    num_labeled_neighbors[user] = 0
            df["num_labeled_neighbors"] = df["user_hash"].map(num_labeled_neighbors).fillna(0)
        else:
            # For test set, use all available labels
            user_to_label = train_df.set_index("user_hash")["is_cheating"].dropna().to_dict()
            neighbor_cheat_ratio = {}
            num_labeled_neighbors = {}
            
            for node in G.nodes():
                try:
                    nbrs = list(G.neighbors(node))
                    labeled_nbrs = [nbr for nbr in nbrs if nbr in user_to_label]
                    if labeled_nbrs:
                        cheat_ratio = np.mean([user_to_label[nbr] for nbr in labeled_nbrs])
                        neighbor_cheat_ratio[node] = cheat_ratio
                        num_labeled_neighbors[node] = len(labeled_nbrs)
                    else:
                        neighbor_cheat_ratio[node] = 0.0
                        num_labeled_neighbors[node] = 0
                except (nx.NetworkXError, KeyError):
                    neighbor_cheat_ratio[node] = 0.0
                    num_labeled_neighbors[node] = 0
            
            df["neighbor_cheat_ratio"] = df["user_hash"].map(neighbor_cheat_ratio).fillna(0)
            df["num_labeled_neighbors"] = df["user_hash"].map(num_labeled_neighbors).fillna(0)
    
    graph_feature_cols = [
        'degree', 'component_size', 'neighbor_cheat_ratio', 
        'num_labeled_neighbors', 'pagerank'
    ]
    
    print(f"âœ… Graph features created in {time.time()-t0:.2f}s")
    return train_df, test_df, graph_feature_cols, G


def create_special_features(train_df, test_df):
    """Create domain-specific features"""
    print("ğŸ�¯ Creating special features...")
    t0 = time.time()
    
    for df in [train_df, test_df]:
        df["f012_is_too_fast"] = (df["feature_012"] > df["feature_012"].quantile(0.95)).astype(int)
        df["f012_bin"] = pd.qcut(df["feature_012"], q=5, duplicates='drop').cat.codes
        df["f015_bin"] = pd.qcut(df["feature_015"], q=7, duplicates='drop').cat.codes
        df["f016_is_high"] = (df["feature_016"] > df["feature_016"].median()).astype(int)
        df["f012_in_risky_time"] = df["feature_012"] * (1 - df["feature_014"])    
        df["danger_f004"] = df["feature_004"].isin([0.0, 3.0, np.nan]).astype(int)
        df["missing_count"] = df[feature_cols].isna().sum(axis=1)
    
    special_features = [
        "f012_is_too_fast", "f012_bin", "f015_bin", "f016_is_high",
        "f012_in_risky_time", "danger_f004", "missing_count"
    ]
    
    print(f"âœ… Special features created in {time.time()-t0:.2f}s")
    return train_df, test_df, special_features


# -------------------------
# Label Propagation
# -------------------------
def run_fast_label_propagation(social_graph, seeds, all_users, n_iter=3):
    """Fast degree-corrected label propagation"""
    print(f"ğŸ”„ Running degree-corrected label propagation ({n_iter} iterations)...")
    t0 = time.time()
    
    # Create adjacency dictionary
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
                new_scores[user] = scores[user]
            else:
                neighbors = adj.get(user, [])
                if neighbors:
                    # Degree-corrected aggregation
                    neighbor_scores = [
                        scores[n] / np.sqrt(len(adj.get(n, [])) + 1e-6)
                        for n in neighbors
                    ]
                    avg_score = np.sum(neighbor_scores) / (np.sqrt(len(neighbors)) + 1e-6)
                    new_scores[user] = 0.5 * scores[user] + 0.5 * avg_score
                else:
                    new_scores[user] = scores[user]
        scores = new_scores
    
    print(f"âœ… Degree-corrected label propagation completed in {time.time()-t0:.2f}s")
    return pd.Series(scores)


# -------------------------
# Model Training
# -------------------------
def train_stacked_primary(X, y, test_X, features, seed=42, n_folds=5):
    """Train stacked ensemble with XGBoost, LightGBM, and CatBoost"""
    print(f"Training a stacking ensemble (Seed: {seed})...")
    t0 = time.time()
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    
    model_configs = [
        ("xgb", XGBClassifier(
            n_estimators=722, learning_rate=0.03, max_depth=9,
            subsample=0.9, colsample_bytree=0.8, gamma=0.7,
            min_child_weight=5, reg_alpha=2, reg_lambda=1.7,
            random_state=seed, verbosity=0
        )),
        ("lgbm", LGBMClassifier(
            n_estimators=761, learning_rate=0.03, num_leaves=105,
            max_depth=9, min_data_in_leaf=123, lambda_l1=0.02,
            lambda_l2=0.005, feature_fraction=0.47, bagging_fraction=0.81,
            bagging_freq=1, random_state=seed, verbose=-1
        ))
    ]
    
    if CATBOOST_AVAILABLE:
        model_configs.append(
            ("cat", CatBoostClassifier(
                iterations=700, learning_rate=0.03, depth=9,
                verbose=False, random_seed=seed
            ))
        )
    
    oof_preds = []
    test_preds = []
    
    for name, model in model_configs:
        print(f"  Training {name.upper()}...")
        oof = np.zeros(len(X))
        test_pred = np.zeros(len(test_X))
        
        for tr_idx, va_idx in skf.split(X, y):
            X_tr, X_va = X.iloc[tr_idx][features], X.iloc[va_idx][features]
            y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]
            
            if name == "xgb":
                model.fit(
                    X_tr, y_tr,
                    eval_set=[(X_va, y_va)],
                    early_stopping_rounds=50,
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
        auc = roc_auc_score(y, oof)
        print(f"    {name.upper()} OOF AUC: {auc:.5f}")
    
    # Meta learner
    print("  Training meta-learner...")
    oof_stack = np.column_stack(oof_preds)
    test_stack = np.column_stack(test_preds)
    
    meta_oof = np.zeros(len(X))
    meta_test = np.zeros(len(test_X))
    
    for tr_idx, va_idx in skf.split(oof_stack, y):
        meta = LogisticRegression(random_state=seed, max_iter=1000)
        meta.fit(oof_stack[tr_idx], y.iloc[tr_idx])
        meta_oof[va_idx] = meta.predict_proba(oof_stack[va_idx])[:, 1]
        meta_test += meta.predict_proba(test_stack)[:, 1] / n_folds
    
    auc = roc_auc_score(y, meta_oof)
    print(f"  ğŸ”� Stacked OOF AUC: {auc:.5f}")
    print(f"âœ… Stacking completed in {time.time()-t0:.2f}s")
    
    return meta_oof, meta_test


# -------------------------
# Balanced Label Propagation
# -------------------------
def run_balanced_LP(test_df, graph_df, predictions, alpha=0.15, max_iter=50):
    """Run logit-space balanced label propagation on test set"""
    print(f"âš–ï¸� Running logit-space balanced LP (Î±={alpha}, iter={max_iter})...")
    t0 = time.time()
    
    test_user_set = set(test_df['user_hash'])
    mask = graph_df['source'].isin(test_user_set) & graph_df['target'].isin(test_user_set)
    df_edges = graph_df[mask].copy()
    
    # Build adjacency matrix
    user_to_idx = {u: i for i, u in enumerate(test_df['user_hash'])}
    row = df_edges['source'].map(user_to_idx).values
    col = df_edges['target'].map(user_to_idx).values
    data = np.ones(len(row))
    
    num_users = len(test_df)
    adj_matrix = sp.coo_matrix((data, (row, col)), shape=(num_users, num_users))
    adj_matrix = adj_matrix + adj_matrix.T
    adj_norm = normalize(adj_matrix, norm='l1', axis=1)
    
    # Logit transformation
    logit = lambda p: np.log(p / (1 - p + 1e-6) + 1e-6)
    sigmoid = lambda x: 1 / (1 + np.exp(-np.clip(x, -10, 10)))
    
    # Propagation in logit space
    y_init = logit(predictions)
    y_current = y_init.copy()
    node_degrees = np.array(adj_matrix.sum(axis=1)).flatten()
    has_neighbor_mask = node_degrees > 0
    
    for i in range(max_iter):
        neighbor_avg = adj_norm.dot(y_current)
        y_current[has_neighbor_mask] = (
            alpha * neighbor_avg[has_neighbor_mask] + 
            (1 - alpha) * y_init[has_neighbor_mask]
        )
        
        if (i+1) % 10 == 0:
            print(f"    Iteration {i+1}/{max_iter} completed")
    
    final_pred = sigmoid(y_current)
    print(f"âœ… Logit-space balanced LP completed in {time.time()-t0:.2f}s")
    return final_pred


# -------------------------
# Main Execution
# -------------------------
def main():
    global train, test
    
    # 1. Neighbor aggregates
    train, test, neighbor_agg_cols, relative_cols = create_neighbor_aggregates(
        train, test, graph, feature_cols
    )
    
    # 2. Graph features (initial - will be recomputed per fold)
    _, _, graph_feature_cols, G = create_graph_features(train, test, graph)
    
    # 3. Special features
    train, test, special_features = create_special_features(train, test)
    
    # 4. All features
    all_features = feature_cols + graph_feature_cols + special_features + neighbor_agg_cols + relative_cols
    print(f"ğŸ“Š Total features: {len(all_features)}")
    
    # 5. Label propagation for risk scores
    labeled_train = train[train['is_cheating'].notnull()]
    y = labeled_train['is_cheating'].values
    oof_risk = pd.Series(index=labeled_train['user_hash'], dtype=float)
    all_users = pd.concat([graph['source'], graph['target'], 
                          train['user_hash'], test['user_hash']]).unique()
    
    # High confidence clean samples
    high_conf_clean = train[train['high_conf_clean'] == 1].set_index('user_hash')
    clean_seeds = pd.Series(0.0, index=high_conf_clean.index)
    
    # Cross-validation for OOF risk scores with graph-aware CV
    kf_feat = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    for fold, (train_idx, val_idx) in enumerate(kf_feat.split(labeled_train, y)):
        print(f"ğŸ”„ LP Fold {fold+1}/3...")
        train_fold = labeled_train.iloc[train_idx]
        train_seeds = train_fold.set_index('user_hash')['is_cheating']
        seeds = pd.concat([train_seeds, clean_seeds])
        scores = run_fast_label_propagation(graph, seeds, all_users, n_iter=LP_ITERATIONS)
        val_users = labeled_train.iloc[val_idx]['user_hash']
        oof_risk.loc[val_users] = scores.loc[val_users]
        gc.collect()
    
    train['risk_score'] = train['user_hash'].map(oof_risk).fillna(0.5)
    
    # Test risk scores
    print("ğŸ”„ Generating test risk scores...")
    all_labeled_seeds = labeled_train.set_index('user_hash')['is_cheating']
    seeds = pd.concat([all_labeled_seeds, clean_seeds])
    test_scores = run_fast_label_propagation(graph, seeds, all_users, n_iter=LP_ITERATIONS)
    test['risk_score'] = test['user_hash'].map(test_scores).fillna(0.5)
    
    # 6. Prepare final datasets with fold-pure graph features
    labeled = train[train["is_cheating"].notnull()].reset_index(drop=True)
    
    # Recompute graph features per fold for OOF
    print("ğŸ”„ Recomputing graph features with fold-pure neighbor cheat ratios...")
    oof_graph_features = []
    
    for fold, (train_idx, val_idx) in enumerate(kf_feat.split(labeled, y)):
        print(f"  Fold {fold+1}/3...")
        # Compute neighbor cheat ratio for this fold only
        neighbor_cheat = fold_neighbor_cheat_ratio(G, labeled, train_idx)
        
        # Apply to validation set
        val_fold = labeled.iloc[val_idx].copy()
        # Create a dummy test_df with user_hash column to avoid errors
        dummy_test_df = pd.DataFrame(columns=['user_hash'])
        val_fold, _, _, _ = create_graph_features(
            val_fold, dummy_test_df, graph, G=G, neighbor_cheat_map=neighbor_cheat
        )
        oof_graph_features.append(val_fold)
    
    # Combine OOF graph features
    oof_graph_df = pd.concat(oof_graph_features).sort_index()
    labeled = labeled.sort_index()
    
    # Update graph features in labeled dataset
    for col in graph_feature_cols:
        labeled[col] = oof_graph_df[col].values
    
    X = labeled[all_features].reset_index(drop=True)
    y = labeled["is_cheating"].astype(int).reset_index(drop=True)
    test_X = test[all_features].reset_index(drop=True)
    
    print(f"ğŸ“Š Labeled samples: {len(X)} | Positive rate: {y.mean():.3f}")
    
    # 7. Model training
    oof, test_pred = train_stacked_primary(
        X=X, y=y, test_X=test_X,
        features=all_features, seed=SEED, n_folds=NFOLDS
    )
    
    # 8. Balanced label propagation in logit space
    final_test = run_balanced_LP(
        test, graph, test_pred, 
        alpha=ALPHA_LP, max_iter=BALANCED_LP_ITER
    )
    
    # 9. Final evaluation
    final_auc = roc_auc_score(y, oof)
    print(f"ğŸ�† Final OOF AUC: {final_auc:.5f}")
    
    # 10. Submission
    submission = pd.DataFrame({
        "user_hash": test["user_hash"],
        "prediction": final_test
    })
    submission.to_csv("submission.csv", index=False)
    print("âœ… Final submission saved!")

if __name__ == "__main__":
    main()

