# ===============================================================

# A = Backbone freeze
# B = Ultra-conservative rank blend UCB

# ===============================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings; warnings.filterwarnings("ignore")

TARGET = "diagnosed_diabetes"
SEED = 42
np.random.seed(SEED)

# backbone 
BACKBONE_PATH = "/kaggle/input/ps-s5e12-0-70644/submission.csv"


OTHER_PATHS = [
    "/kaggle/input/ps-s5e12-lightgbm-xgboost-catboost/submission_optimized.csv",
    "/kaggle/input/s5e12-exploring-fe-optuna-ensemble/submission.csv",
    "/kaggle/input/s5e12-xgb-sample-weight/submission.csv",
    "/kaggle/input/diabetes-161225/0.70191.csv",
]


RANK_WEIGHT = 0.015  



sample = pd.read_csv("/kaggle/input/playground-series-s5e12/sample_submission.csv")

def load_pred(path):
    return pd.read_csv(path)[TARGET].values

backbone = load_pred(BACKBONE_PATH)
others = [load_pred(p) for p in OTHER_PATHS]

# ---------------- SUBMISSION A (FREEZE) ----------------
sub_A = sample.copy()
sub_A[TARGET] = backbone
sub_A.to_csv("submission.csv", index=False)

print("OK Saved submission.csv")

# ---------------- SUBMISSION B ----------------

rank_preds = [pd.Series(p).rank(pct=True).values for p in [backbone] + others]
rank_blend = np.mean(rank_preds, axis=0)

final_B = (1 - RANK_WEIGHT) * backbone + RANK_WEIGHT * rank_blend
final_B = np.clip(final_B, 0, 1)

sub_B = sample.copy()
sub_B[TARGET] = final_B
sub_B.to_csv("submission_B_rank_blend_safe.csv", index=False)

print("OK Saved submission_B_rank_blend_safe.csv")

# ---------------- PLOT ----------------
plt.figure(figsize=(10,5))
sns.kdeplot(backbone, label="Backbone", linewidth=1)
sns.kdeplot(final_B, label="Rank blend (safe)", linewidth=2, linestyle="--")
plt.title("KDE comparison (must be almost identical)")
plt.xlabel("Predicted value")
plt.ylabel("Density")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


diff = np.abs(final_B - backbone)
print("Mean abs diff :", diff.mean())
print("Max abs diff  :", diff.max())


