import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

# -----------------------------
# Load submissions
# -----------------------------
df1 = pd.read_csv("/kaggle/input/fake-or-real-the-impostor-hunt-in-texts-blend/submission_0.90456.csv")
df2 = pd.read_csv("/kaggle/input/fake-or-real-distilbert-jcl/submission.csv")
df3 = pd.read_csv("/kaggle/input/0-87759-fake-or-real-bert-pca-randomforest/submission.csv")
df4 = pd.read_csv("/kaggle/input/combining-feature-extraction-bert/submission.csv")
df5 = pd.read_csv("/kaggle/input/fake-or-real-using-lgbm-deberta/submission.csv")
df6 = pd.read_csv("/kaggle/input/fake-or-real-using-lgbm-deberta/submission.csv")
df7 = pd.read_csv("/kaggle/input/kaggle-project/submission.csv")

dfs = [df1, df2, df3, df4, df5, df6, df7]
for df in dfs:
    df.sort_values("id", inplace=True)
    df.reset_index(drop=True, inplace=True)

ids = df1["id"].values

# -----------------------------
# 1. Weighted Voting Ensemble
# -----------------------------
# Use LB scores as weights
lb_scores = [0.87967, 0.87966, 0.87759, 0.84232, 0.84233, 0.82157, 0.90456]
weights = np.array(lb_scores) / sum(lb_scores)

score_1 = np.zeros(len(df1))
score_2 = np.zeros(len(df1))

for df, w in zip(dfs, weights):
    score_1 += (df["real_text_id"] == 1) * w
    score_2 += (df["real_text_id"] == 2) * w

final_pred_vote = np.where(score_1 >= score_2, 1, 2).astype(int)

submission_vote = pd.DataFrame({"id": ids, "real_text_id": final_pred_vote})
submission_vote.to_csv("submission_weighted.csv", index=False)
print("Weighted voting submission saved → submission_weighted.csv")

# -----------------------------
# 2. Stacking Meta-Model
# -----------------------------
X = np.column_stack([df["real_text_id"] for df in dfs])

# Use weighted voting result as pseudo-labels
y_pseudo = final_pred_vote

meta_model = LogisticRegression(max_iter=500, class_weight="balanced")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
for tr_idx, val_idx in skf.split(X, y_pseudo):
    meta_model.fit(X[tr_idx], y_pseudo[tr_idx])
    oof_preds[val_idx] = meta_model.predict(X[val_idx])

print("Stacking CV Macro-F1:", f1_score(y_pseudo, oof_preds, average="macro"))

# Retrain on full data
meta_model.fit(X, y_pseudo)
final_pred_stack = meta_model.predict(X)

submission_stack = pd.DataFrame({"id": ids, "real_text_id": final_pred_stack.astype(int)})
submission_stack.to_csv("submission_stacking.csv", index=False)
print("Stacking submission saved → submission_stacking.csv")


