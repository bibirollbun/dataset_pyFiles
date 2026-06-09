# =========================================================================================
# ğŸ�† MERCOR CHEATING DETECTION - 2-HOP + FEATURE SELECTION (THE POLISHED GEM)
# =========================================================================================
import os
import gc
import warnings
import numpy as np
import pandas as pd
import networkx as nx
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier, early_stopping
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")

# =========================================================================================
# âš™ï¸� CONFIGURATION
# =========================================================================================
INPUT_DIR = "/kaggle/input/mercor-cheating-detection"
N_FOLDS = 5
# Upgrading to 5 SEEDS for the 2-Hop Model (Previous best was 3)
SEEDS = [42, 2024, 777, 101, 999] 

print("âœ… Configuration Set: Polished 2-Hop Ensemble")

# =========================================================================================
# ğŸ› ï¸� HELPER FUNCTIONS
# =========================================================================================
def mercor_cost_metric(y_true, y_pred_ranks):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred_ranks)
    sort_order = np.argsort(y_pred)
    y_true_sorted = y_true[sort_order]
    
    c1 = (y_true_sorted * 745 - 150).cumsum()
    c2 = (y_true_sorted * 155 - 150).cumsum()
    base_block_cost = (y_true == 0).sum() * 300
    
    return -(c1.min() + c2.min() + base_block_cost)

def run_fast_label_propagation(social_graph, seeds, all_users, n_iter=3):
    adj = {}
    for _, row in social_graph.iterrows():
        adj.setdefault(row['source'], []).append(row['target'])
        adj.setdefault(row['target'], []).append(row['source'])
    scores = {user: 0.5 for user in all_users}
    scores.update(seeds.to_dict())
    for _ in range(n_iter):
        new_scores = {}
        for user in all_users:
            if user in seeds:
                new_scores[user] = seeds[user]
            else:
                neighbors = adj.get(user, [])
                if neighbors:
                    vals = [scores.get(n, 0.5) for n in neighbors]
                    new_scores[user] = 0.5 * scores[user] + 0.5 * np.mean(vals)
                else:
                    new_scores[user] = scores[user]
        scores = new_scores
    return pd.Series(scores)

# =========================================================================================
# ğŸ“¥ DATA LOADING & FEATURE ENGINEERING (2-HOP)
# =========================================================================================
print("\n--- 1. Loading & Processing Data ---")
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
test = pd.read_csv(os.path.join(INPUT_DIR, "test.csv"))
graph = pd.read_csv(os.path.join(INPUT_DIR, "social_graph.csv"), names=["source", "target"])
feature_cols = [c for c in train.columns if c.startswith("feature_")]

# 1. 1-Hop Neighbor Aggregations
print("   > Generating 1-Hop Neighbor Aggregations...")
all_users_feat = pd.concat([train[["user_hash"] + feature_cols], test[["user_hash"] + feature_cols]])
all_users_feat = all_users_feat.drop_duplicates("user_hash").set_index("user_hash")

rev_graph = graph.rename(columns={"source": "target", "target": "source"})
full_edges = pd.concat([graph, rev_graph], ignore_index=True)
full_edges = full_edges.merge(all_users_feat, left_on="target", right_index=True, how="left")

agg_stats = full_edges.groupby("source")[feature_cols].agg(["mean", "std"])
agg_stats.columns = [f"nbr_{c[0]}_{c[1]}" for c in agg_stats.columns]
neighbor_agg_cols = list(agg_stats.columns)

del full_edges
gc.collect()

train = train.merge(agg_stats, left_on="user_hash", right_index=True, how="left")
test = test.merge(agg_stats, left_on="user_hash", right_index=True, how="left")
for col in neighbor_agg_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

# 2. 2-Hop Neighbor Aggregations
print("   > Generating 2-Hop Neighbor Aggregations...")
all_users_1hop = pd.concat([train[["user_hash"] + neighbor_agg_cols], test[["user_hash"] + neighbor_agg_cols]])
all_users_1hop = all_users_1hop.drop_duplicates("user_hash").set_index("user_hash")

full_edges_2hop = pd.concat([graph, rev_graph], ignore_index=True)
full_edges_2hop = full_edges_2hop.merge(all_users_1hop, left_on="target", right_index=True, how="left")

# Select only the 'mean' columns for 2-hop to save memory/complexity
mean_cols = [c for c in neighbor_agg_cols if 'mean' in c]
agg_stats_2hop = full_edges_2hop.groupby("source")[mean_cols].agg("mean")
agg_stats_2hop.columns = [f"nbr_{c}" for c in agg_stats_2hop.columns]
neighbor_2hop_cols = list(agg_stats_2hop.columns)

del full_edges_2hop, all_users_1hop, rev_graph
gc.collect()

train = train.merge(agg_stats_2hop, left_on="user_hash", right_index=True, how="left")
test = test.merge(agg_stats_2hop, left_on="user_hash", right_index=True, how="left")
for col in neighbor_2hop_cols:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

# 3. Relative Features
print("   > Generating Relative Features...")
eps = 1e-5
for col in feature_cols:
    nbr_mean = f"nbr_{col}_mean"
    if nbr_mean in train.columns:
        train[f"{col}_ratio"] = train[col] / (train[nbr_mean] + eps)
        test[f"{col}_ratio"]  = test[col]  / (test[nbr_mean] + eps)
        train[f"{col}_diff"] = train[col] - train[nbr_mean]
        test[f"{col}_diff"]  = test[col]  - test[nbr_mean]

# 4. Graph Features
print("   > Generating Graph Features...")
G = nx.from_pandas_edgelist(graph, "source", "target", create_using=nx.Graph())
degree_map = dict(G.degree())
pagerank_map = nx.pagerank(G, alpha=0.85)
comp_size_map = {node: len(comp) for comp in nx.connected_components(G) for node in comp}

user_to_label = train.set_index("user_hash")["is_cheating"].dropna().to_dict()
neighbor_cheat_ratio = {}
for node in G.nodes():
    nbrs = list(G.neighbors(node))
    labeled_nbrs = [nbr for nbr in nbrs if nbr in user_to_label]
    neighbor_cheat_ratio[node] = np.mean([user_to_label[n] for n in labeled_nbrs]) if labeled_nbrs else 0.0

# 5. Feature-Level LP
print("   > Running Feature-Level LP...")
labeled_train = train[train['is_cheating'].notna()]
clean_seeds = pd.Series(0.0, index=train[train['high_conf_clean'] == 1]['user_hash'])
all_users = pd.concat([graph['source'], graph['target'], train['user_hash'], test['user_hash']]).unique()
kf_feat = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
oof_risk = pd.Series(index=labeled_train['user_hash'], dtype=float)
for fold, (train_idx, val_idx) in enumerate(kf_feat.split(labeled_train, labeled_train['is_cheating'])):
    train_fold = labeled_train.iloc[train_idx]
    train_seeds = train_fold.set_index('user_hash')['is_cheating']
    seeds = pd.concat([train_seeds, clean_seeds])
    scores = run_fast_label_propagation(graph, seeds, all_users, n_iter=3)
    val_users = labeled_train.iloc[val_idx]['user_hash']
    oof_risk.loc[val_users] = scores.reindex(val_users).fillna(0.5).values
train['risk_score'] = train['user_hash'].map(oof_risk).fillna(0.5)
all_labeled_seeds = labeled_train.set_index('user_hash')['is_cheating']
seeds = pd.concat([all_labeled_seeds, clean_seeds])
test_scores = run_fast_label_propagation(graph, seeds, all_users, n_iter=3)
test['risk_score'] = test['user_hash'].map(test_scores).fillna(0.5)

# 6. Mapping & Finalizing
print("   > Finalizing Features...")
for df in [train, test]:
    df["degree"] = df["user_hash"].map(degree_map).fillna(0)
    df["component_size"] = df["user_hash"].map(comp_size_map).fillna(1)
    df["neighbor_cheat_ratio"] = df["user_hash"].map(neighbor_cheat_ratio).fillna(0)
    df["pagerank"] = df["user_hash"].map(pagerank_map).fillna(0)
    df["f012_is_too_fast"] = (df["feature_012"] > df["feature_012"].quantile(0.95)).astype(int)
    df["f012_in_risky_time"] = df["feature_012"] * (1 - df["feature_014"])    
    df["missing_count"] = df[feature_cols].isin([np.nan]).sum(axis=1)

base_cols = [c for c in train.columns if c.startswith("feature_") or c.endswith("_ratio") or c.endswith("_diff")]
graph_cols = ['risk_score', 'degree', 'component_size', 'neighbor_cheat_ratio', 'pagerank']
special_cols = ['f012_is_too_fast', 'f012_in_risky_time', 'missing_count']
# Initial Feature Set
all_features = list(set(base_cols + graph_cols + special_cols + neighbor_agg_cols + neighbor_2hop_cols))

print(f"âœ… Initial Features: {len(all_features)}")

# =========================================================================================
# âœ‚ï¸� FEATURE SELECTION (The Magic Step)
# =========================================================================================
print("\n--- 2. Feature Selection (Removing Noise) ---")
# Train a quick LightGBM to identify useless features
X = train[train['is_cheating'].notna()][all_features]
y = train[train['is_cheating'].notna()]['is_cheating'].astype(int)

lgb_sel = LGBMClassifier(n_estimators=500, random_state=42, verbose=-1, device='gpu')
lgb_sel.fit(X, y)

# Get Importance
imp_df = pd.DataFrame({'feature': all_features, 'importance': lgb_sel.feature_importances_})
imp_df = imp_df.sort_values('importance', ascending=False)

# Keep top 85% of features (Drop the bottom tail which is just noise)
keep_count = int(len(all_features) * 0.85)
selected_features = imp_df.head(keep_count)['feature'].tolist()

print(f"   > Dropped {len(all_features) - keep_count} features.")
print(f"   > Selected Features: {len(selected_features)}")

# =========================================================================================
# ğŸ§  5-SEED TRAINING (Robust)
# =========================================================================================
print(f"\n--- 3. Training with {len(SEEDS)} Seeds on Selected Features ---")
X = train[train['is_cheating'].notna()][selected_features]
y = train[train['is_cheating'].notna()]['is_cheating'].astype(int)
X_test = test[selected_features]

oof_accum = {'xgb': np.zeros(len(X)), 'lgb': np.zeros(len(X)), 'cat': np.zeros(len(X))}
test_accum = {'xgb': np.zeros(len(X_test)), 'lgb': np.zeros(len(X_test)), 'cat': np.zeros(len(X_test))}

for i, seed in enumerate(SEEDS):
    print(f"\nğŸŒ± SEED {seed} ({i+1}/{len(SEEDS)})")
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
        X_va, y_va = X.iloc[val_idx], y.iloc[val_idx]
        
        # XGBoost
        xgb = XGBClassifier(n_estimators=1000, learning_rate=0.03, max_depth=8, subsample=0.8,
                            colsample_bytree=0.8, random_state=seed, verbosity=0, tree_method='gpu_hist')
        xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False, early_stopping_rounds=50)
        
        # LightGBM
        lgbm = LGBMClassifier(n_estimators=1000, learning_rate=0.03, num_leaves=64, subsample=0.8,
                              colsample_bytree=0.8, random_state=seed, verbose=-1, device='gpu')
        lgbm.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[early_stopping(50, verbose=False)])

        # CatBoost
        cat = CatBoostClassifier(iterations=1000, learning_rate=0.03, depth=8, random_seed=seed, 
                                 verbose=False, allow_writing_files=False, task_type='GPU')
        cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
        
        # Accumulate
        oof_accum['xgb'][val_idx] += xgb.predict_proba(X_va)[:, 1] / len(SEEDS)
        oof_accum['lgb'][val_idx] += lgbm.predict_proba(X_va)[:, 1] / len(SEEDS)
        oof_accum['cat'][val_idx] += cat.predict_proba(X_va)[:, 1] / len(SEEDS)
        test_accum['xgb'] += xgb.predict_proba(X_test)[:, 1] / (len(SEEDS) * N_FOLDS)
        test_accum['lgb'] += lgbm.predict_proba(X_test)[:, 1] / (len(SEEDS) * N_FOLDS)
        test_accum['cat'] += cat.predict_proba(X_test)[:, 1] / (len(SEEDS) * N_FOLDS)

print("\nâœ… Training Complete")

# =========================================================================================
# âš–ï¸� OPTIMIZATION & SUBMISSION
# =========================================================================================
print("\n--- 4. Final Optimization ---")
oof_ranks = pd.DataFrame({
    'xgb': pd.Series(oof_accum['xgb']).rank(pct=True),
    'lgb': pd.Series(oof_accum['lgb']).rank(pct=True),
    'cat': pd.Series(oof_accum['cat']).rank(pct=True)
})

def objective(weights):
    w = np.array(weights)
    w = np.maximum(w, 0)
    w /= w.sum()
    blend = (w[0]*oof_ranks['xgb'] + w[1]*oof_ranks['lgb'] + w[2]*oof_ranks['cat'])
    return -mercor_cost_metric(y, blend)

res = minimize(objective, [0.33, 0.33, 0.33], method='Nelder-Mead', tol=1e-5)
best_weights = res.x / res.x.sum()
print(f"ğŸ�† Best Weights: XGB={best_weights[0]:.3f}, LGB={best_weights[1]:.3f}, CAT={best_weights[2]:.3f}")

test_ranks = pd.DataFrame({
    'xgb': pd.Series(test_accum['xgb']).rank(pct=True),
    'lgb': pd.Series(test_accum['lgb']).rank(pct=True),
    'cat': pd.Series(test_accum['cat']).rank(pct=True)
})
final_pred = (best_weights[0]*test_ranks['xgb'] + best_weights[1]*test_ranks['lgb'] + best_weights[2]*test_ranks['cat'])

sub = pd.DataFrame({"user_hash": test["user_hash"], "prediction": final_pred})
sub.to_csv("submission_polished.csv", index=False)
print("âœ… Saved: submission_polished.csv")

