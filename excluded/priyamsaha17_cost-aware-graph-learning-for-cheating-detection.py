# =========================++++++++++++++++++++++=========================
# Label-Induced Graph + LightGBM
# 80:20 split
# ========================================================================

import numpy as np
import pandas as pd
import networkx as nx
import lightgbm as lgb
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split

# -----------------------------
# 0. LOAD DATA
# -----------------------------
print("Loading CSVs...")
train = pd.read_csv("/kaggle/input/mercor-cheating-detection/train.csv")
test  = pd.read_csv("/kaggle/input/mercor-cheating-detection/test.csv")
graph_df = pd.read_csv("/kaggle/input/mercor-cheating-detection/social_graph.csv")

feature_cols = [c for c in train.columns if c.startswith("feature_")]

# -----------------------------
# 1. KEEP ONLY LABELED DATA
# -----------------------------
labeled = train[train.is_cheating.notna()].copy().reset_index(drop=True)
labeled["is_cheating"] = labeled["is_cheating"].astype(int)

y = labeled.is_cheating.values
users = labeled.user_hash.values
labeled_set = set(users)

print(f"Labeled samples: {len(labeled)} | Positive rate: {y.mean():.4f}")

# -----------------------------
# 2. BUILD LABEL-INDUCED GRAPH
# -----------------------------
print("Building labeled-induced graph...")
G = nx.Graph()

for _, r in tqdm(graph_df.iterrows(), total=len(graph_df), desc="Filtering edges"):
    if r.user_a in labeled_set and r.user_b in labeled_set:
        G.add_edge(r.user_a, r.user_b)

print("Nodes:", G.number_of_nodes(), "Edges:", G.number_of_edges())

# -----------------------------
# 3. GRAPH FEATURES (incl. PPR)
# -----------------------------
print("Computing graph features...")

pagerank = nx.pagerank(G, max_iter=100)
core = nx.core_number(G)
deg = dict(G.degree())

label_map = dict(zip(users, y))
personalization = {u: float(label_map.get(u, 0)) for u in G.nodes()}
if sum(personalization.values()) == 0:
    personalization = None

ppr_cheater = nx.pagerank(G, personalization=personalization, max_iter=100)

rows = []
for u in tqdm(users, desc="Graph feature extraction"):
    if u in G:
        nbrs = list(G.neighbors(u))
        lbls = [label_map[n] for n in nbrs]
        rows.append({
            "user_hash": u,
            "lab_degree": len(nbrs),
            "lab_pagerank": pagerank.get(u, 0.0),
            "lab_core": core.get(u, 0),
            "lab_nbr_cheat_rate": np.mean(lbls) if lbls else 0.0,
            "lab_nbr_cnt": len(lbls),
            "ppr_cheater": ppr_cheater.get(u, 0.0)
        })
    else:
        rows.append({
            "user_hash": u,
            "lab_degree": 0,
            "lab_pagerank": 0.0,
            "lab_core": 0,
            "lab_nbr_cheat_rate": 0.0,
            "lab_nbr_cnt": 0,
            "ppr_cheater": 0.0
        })

graph_feat = pd.DataFrame(rows).set_index("user_hash")

# -----------------------------
# 4. TABULAR FEATURES
# -----------------------------
X_tab = labeled[["user_hash"] + feature_cols].copy()
X_tab["mean"] = X_tab[feature_cols].mean(axis=1)
X_tab["std"] = X_tab[feature_cols].std(axis=1)
X_tab["miss_cnt"] = X_tab[feature_cols].isna().sum(axis=1)
X_tab = X_tab.fillna(-999).set_index("user_hash")

# -----------------------------
# 5. MERGE FEATURES
# -----------------------------
X_full = X_tab.join(graph_feat, how="left").fillna(0.0).reset_index()
X = X_full.drop(columns=["user_hash"])
assert len(X) == len(y)

# -----------------------------
# 6. 80:20 TRAIN / VALIDATION SPLIT
# -----------------------------
X_tr, X_va, y_tr, y_va, u_tr, u_va = train_test_split(
    X, y, users, test_size=0.2, stratify=y, random_state=42
)

# -----------------------------------
# 7. METRIC to induce cost-awareness
# -----------------------------------
def mercor_cost(y_true, decisions):
    auto_pass = decisions == 0
    manual = decisions == 1
    auto_block = decisions == 2
    cost = 0
    cost += np.sum((y_true == 1) & auto_pass) * 600
    cost += np.sum((y_true == 1) & manual) * 5
    cost += np.sum((y_true == 0) & manual) * 150
    cost += np.sum((y_true == 0) & auto_block) * 300
    return cost

# -----------------------------
# 8. TRAIN WITH METRIC-DRIVEN SELECTION
# -----------------------------
params = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "min_data_in_leaf": 50,
    "verbosity": -1,
    "seed": 42
}

dtrain = lgb.Dataset(X_tr, label=y_tr)
dvalid = lgb.Dataset(X_va, label=y_va)

num_rounds = 1200
eval_every = 50

best_cost = 1e18
best_model = None
best_policy = None

print("Training LightGBM + evaluating Mercor metric...")

for it in tqdm(range(eval_every, num_rounds + 1, eval_every), desc="Boosting checkpoints"):
    model = lgb.train(
        params,
        dtrain,
        num_boost_round=it
    )

    probs = model.predict(X_va)

    # policy grid (small but effective)
    best_local = (1e18, None)
    for t1 in np.linspace(0.05, 0.5, 30):
        for t2 in np.linspace(0.5, 0.95, 30):
            dec = np.ones(len(probs), dtype=int)
            dec[probs <= t1] = 0
            dec[probs > t2] = 2
            cost = mercor_cost(y_va, dec)
            if cost < best_local[0]:
                best_local = (cost, (t1, t2))

    if best_local[0] < best_cost:
        best_cost = best_local[0]
        best_model = model
        best_policy = best_local[1]

    tqdm.write(f"Rounds {it} | Best cost so far: {best_cost} | Mercor score: {-best_cost}")

print("Best policy (t_low, t_high):", best_policy)
print("Final Mercor validation score:", -best_cost)

# -----------------------------
# 9. FINAL TRAIN ON ALL LABELED
# -----------------------------
print("Training final model...")
final_model = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=best_model.current_iteration())

# -----------------------------
# 10. BUILD TEST FEATURES (MATCH TRAIN)
# -----------------------------
print("Building test features...")
test_rows = []

for u in tqdm(test.user_hash.values, desc="Test graph features"):
    if u in G:
        nbrs = list(G.neighbors(u))
        lbls = [label_map.get(n, 0) for n in nbrs]
        test_rows.append({
            "user_hash": u,
            "lab_degree": len(nbrs),
            "lab_pagerank": pagerank.get(u, 0.0),
            "lab_core": core.get(u, 0),
            "lab_nbr_cheat_rate": np.mean(lbls) if lbls else 0.0,
            "lab_nbr_cnt": len(lbls),
            "ppr_cheater": ppr_cheater.get(u, 0.0)
        })
    else:
        test_rows.append({
            "user_hash": u,
            "lab_degree": 0,
            "lab_pagerank": 0.0,
            "lab_core": 0,
            "lab_nbr_cheat_rate": 0.0,
            "lab_nbr_cnt": 0,
            "ppr_cheater": 0.0
        })

test_graph = pd.DataFrame(test_rows).set_index("user_hash")

X_test_tab = test[["user_hash"] + feature_cols].copy()
X_test_tab["mean"] = X_test_tab[feature_cols].mean(axis=1)
X_test_tab["std"] = X_test_tab[feature_cols].std(axis=1)
X_test_tab["miss_cnt"] = X_test_tab[feature_cols].isna().sum(axis=1)
X_test_tab = X_test_tab.fillna(-999).set_index("user_hash")

X_test = X_test_tab.join(test_graph, how="left").fillna(0.0)

# -----------------------------
# 11. APPLY POLICY + SUBMISSION
# -----------------------------
print("Predicting on test and applying policy...")
test_probs = final_model.predict(X_test)

'''
t1, t2 = best_policy
final_preds = np.where(
    test_probs <= t1, 0.01,
    np.where(test_probs > t2, 0.99, 0.5)
)
'''

submission = pd.DataFrame({
    "user_hash": test.user_hash.values,
    "prediction": test_probs
})

submission.to_csv("submission.csv", index=False)
print("submission.csv saved")
print(submission.head())

