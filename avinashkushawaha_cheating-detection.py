import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb
from collections import defaultdict



train = pd.read_csv("/kaggle/input/mercor-cheating-detection/train.csv")
test = pd.read_csv("/kaggle/input/mercor-cheating-detection/test.csv")
sample = pd.read_csv("/kaggle/input/mercor-cheating-detection/sample_submission.csv")
graph = pd.read_csv("/kaggle/input/mercor-cheating-detection/social_graph.csv")

print(train.shape, test.shape, graph.shape)



print(train["is_cheating"].value_counts(dropna=False))
print(train["high_conf_clean"].value_counts(dropna=False))



# Labeled data
labeled = train[train["is_cheating"].notna()].copy()

# High-confidence clean (weak labels)
weak_clean = train[
    (train["is_cheating"].isna()) & (train["high_conf_clean"] == 1)
].copy()

# Targets
labeled["target"] = labeled["is_cheating"].astype(int)
weak_clean["target"] = 0

# Sample weights
labeled["weight"] = 1.0
weak_clean["weight"] = 0.3

# Combine
train_final = pd.concat([labeled, weak_clean], axis=0)

print(train_final.shape)
print(train_final["target"].value_counts())




# Feature columns
feature_cols = [c for c in train.columns if c.startswith("feature_")]

# Add missing indicators
for col in feature_cols:
    train_final[col + "_missing"] = train_final[col].isna().astype(int)
    test[col + "_missing"] = test[col].isna().astype(int)

final_features = feature_cols + [c + "_missing" for c in feature_cols]

X = train_final[final_features]
y = train_final["target"]
weights = train_final["weight"]

X_test = test[final_features]

print(X.shape, X_test.shape)



graph.columns = ["user_a", "user_b"]

adj = defaultdict(set)
for _, row in graph.iterrows():
    adj[row["user_a"]].add(row["user_b"])
    adj[row["user_b"]].add(row["user_a"])

print("Graph ready")



known_cheaters = set(
    labeled[labeled["is_cheating"] == 1]["user_hash"]
)

print("Known cheaters:", len(known_cheaters))



def compute_graph_features(user_id):
    neighbors = adj.get(user_id, set())
    degree = len(neighbors)

    if degree == 0:
        return 0, 0, 0.0

    cheater_neighbors = sum(1 for n in neighbors if n in known_cheaters)
    cheater_ratio = cheater_neighbors / degree

    return degree, cheater_neighbors, cheater_ratio



# Train
train_graph_feats = train_final["user_hash"].apply(compute_graph_features)
train_graph_feats = pd.DataFrame(
    train_graph_feats.tolist(),
    columns=["degree", "cheater_neighbors", "cheater_ratio"],
    index=train_final.index
)

train_final = pd.concat([train_final, train_graph_feats], axis=1)

# Test
test_graph_feats = test["user_hash"].apply(compute_graph_features)
test_graph_feats = pd.DataFrame(
    test_graph_feats.tolist(),
    columns=["degree", "cheater_neighbors", "cheater_ratio"],
    index=test.index
)

test = pd.concat([test, test_graph_feats], axis=1)



graph_features = ["degree", "cheater_neighbors", "cheater_ratio"]
all_features = final_features + graph_features

X_graph = train_final[all_features]
y_graph = train_final["target"]
weights_graph = train_final["weight"]

X_test_graph = test[all_features]

print(X_graph.shape, X_test_graph.shape)



final_model = lgb.LGBMClassifier(
    objective="binary",
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=31,
    max_depth=-1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)

final_model.fit(
    X_graph,
    y_graph,
    sample_weight=weights_graph
)

final_preds = final_model.predict_proba(X_test_graph)[:, 1]

print(final_preds.min(), final_preds.max())



final_submission = sample.copy()
final_submission["is_cheating"] = final_preds

final_submission.to_csv("mercor_cheating_final.csv", index=False)

print(final_submission.shape)
print(final_submission.columns)
final_submission.head()





