import os, random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from scipy.stats import rankdata

warnings.filterwarnings('ignore')

# ---------------------------
# Load input files
# ---------------------------
p = '/kaggle/input/'

test_df = pd.read_csv(p + "playground-series-s5e8/test.csv")
sub1    = pd.read_csv(p + "ps-s5e8-lightgb-model-add-original-dataset/submission.csv")      
sub2    = pd.read_csv(p + "train-more-xgb-nn-lb-0-9774/submission_ensemble_train_more.csv") 
sub3    = pd.read_csv(p + "21-august-2025-ps-s5e8/submission 0.977621.csv")
sub4    = pd.read_csv(p + "21-august-2025-ps-s5e8/submission arti_18.csv")

# ---------------------------
# Extract prediction columns
# ---------------------------
r1 = sub1['y']  # 0.97541
r2 = sub2['y']  # 0.97742
r3 = sub3['y']  # 0.977621
r4 = sub4['y']  # artificial

# ---------------------------
# Blending (manual weights)
# ---------------------------
r = 0.25
r12 = r * r1 + (1 - r) * r2

sub123  = 0.631 * r3 + 0.369 * r12  # current top ~0.97763
sub1234 = 0.05 * r4 + 0.95 * sub123

# Save manual weighted blend
submission = pd.DataFrame({"id": test_df["id"], "y": sub123})
submission.to_csv("submission.csv", index=False)

# ---------------------------
# Rank averaging trick
# ---------------------------
def rank_average(preds):
    ranked = [rankdata(p) / len(p) for p in preds]
    return np.mean(ranked, axis=0)

# Rank average across all four models
sub_rank = rank_average([r1, r2, r3, r4])
submission_rank = pd.DataFrame({"id": test_df["id"], "y": sub_rank})
submission_rank.to_csv("submission_rankavg.csv", index=False)

# ---------------------------
# Hybrid trick: rank average strong models, mix in r1
# ---------------------------
sub_rank_strong = rank_average([r2, r3, r4])
sub_hybrid = 0.9 * sub_rank_strong + 0.1 * r1
submission_hybrid = pd.DataFrame({"id": test_df["id"], "y": sub_hybrid})
submission_hybrid.to_csv("submission_hybrid.csv", index=False)

# ---------------------------
# Visualization helper
# ---------------------------
def compare_preds(a, b, label_a="A", label_b="B"):
    df1 = pd.DataFrame({"id": test_df["id"], "y": a})
    df2 = pd.DataFrame({"id": test_df["id"], "y": b})
    df = pd.merge(df1, df2, on='id', suffixes=('_1', '_2'))

    fig, ax = plt.subplots(1,2, figsize=(12,4))
    sns.kdeplot(df['y_1'], label=label_a, fill=True, ax=ax[0])
    sns.kdeplot(df['y_2'], label=label_b, fill=True, ax=ax[0])
    ax[0].set_title('Raw Probability Densities')

    sns.scatterplot(x='y_1', y='y_2', data=df.sample(5000), ax=ax[1], alpha=0.3)
    ax[1].plot([0,1],[0,1],'r--')
    ax[1].set_title('Pairwise Probability Scatter')
    plt.show()

# ---------------------------
# Pairwise comparisons
# ---------------------------
compare_preds(sub123, sub_rank, "manual_blend", "rank_avg")
compare_preds(sub123, sub_hybrid, "manual_blend", "hybrid")
compare_preds(r1, r2, "r1", "r2")
compare_preds(r2, r3, "r2", "r3")
compare_preds(r2, r4, "r2", "r4")

