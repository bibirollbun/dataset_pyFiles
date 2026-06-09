import numpy as np
import pandas as pd
from functools import reduce

ID_COL = "id"
PRED_COL = "diagnosed_diabetes"

def load_preds_table(paths, id_col=ID_COL, pred_col=PRED_COL, prefix="S"):
    dfs = []
    for i, p in enumerate(paths):
        df = pd.read_csv(p)[[id_col, pred_col]].rename(columns={pred_col: f"{prefix}{i}"})
        dfs.append(df)
    m = reduce(lambda l, r: l.merge(r, on=id_col, how="inner"), dfs)
    return m

def normalize_weights(weights, k):
    if weights is None:
        return np.ones(k, dtype=float) / k
    w = np.asarray(weights, dtype=float)
    if w.shape[0] != k:
        raise ValueError(f"weights length {len(w)} != number of files {k}")
    s = w.sum()
    if abs(s) < 1e-12:
        # if sum is ~0, fall back to equal weights
        return np.ones(k, dtype=float) / k
    return w / s

def weighted_mean_blend(paths, weights=None, clip_eps=1e-12):
    m = load_preds_table(paths)
    cols = [c for c in m.columns if c != ID_COL]
    P = m[cols].to_numpy(float)
    P = np.clip(P, clip_eps, 1.0 - clip_eps)

    w = normalize_weights(weights, P.shape[1])
    blend = P @ w

    out = pd.DataFrame({ID_COL: m[ID_COL].values, PRED_COL: blend})
    return out

def rank_average_blend(paths):
    m = load_preds_table(paths)
    cols = [c for c in m.columns if c != ID_COL]
    P = m[cols].to_numpy(float)

    ranks = np.zeros_like(P)
    for j in range(P.shape[1]):
        order = np.argsort(P[:, j])
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(order), dtype=float)
        ranks[:, j] = r

    blend = ranks.mean(axis=1)
    blend = (blend - blend.min()) / (blend.max() - blend.min() + 1e-12)

    out = pd.DataFrame({ID_COL: m[ID_COL].values, PRED_COL: blend})
    return out

def power_mean_blend(paths, weights=None, p=8, clip_eps=1e-12):
    m = load_preds_table(paths)
    cols = [c for c in m.columns if c != ID_COL]
    P = m[cols].to_numpy(float)
    P = np.clip(P, clip_eps, 1.0 - clip_eps)

    w = normalize_weights(weights, P.shape[1])

    s = (P ** p) @ w
    s = np.clip(s, clip_eps, None)
    blend = np.power(s, 1.0 / p)

    out = pd.DataFrame({ID_COL: m[ID_COL].values, PRED_COL: blend})
    return out

# -----------------------------
# USAGE
# -----------------------------
paths = [
    "/kaggle/input/s5e12-blending-nyoba/submission_auc (1).csv",
    "/kaggle/input/s5e12-blending-nyoba/submission_auc (2).csv",
    "/kaggle/input/s5e12-blending-nyoba/submission_auc (3).csv",
    "/kaggle/input/s5e12-blending-nyoba/submission_auc.csv",
]

# 1) Rank average (recommended first try for AUC)
rank_out = rank_average_blend(paths)
rank_out.to_csv("submission_rankavg.csv", index=False)
print("✅ saved submission_rankavg.csv")

# 2) Weighted mean (simple & stable)
weights_mean = [0.4, 0.3, 0.2, 0.1]  
mean_out = weighted_mean_blend(paths, weights=weights_mean)
mean_out.to_csv("submission_wmean.csv", index=False)
print("✅ saved submission_wmean.csv")

# 3) Power mean (a bit “agresive”)
weights_power = [0.55, 0.25, 0.15, 0.05]  
power_out = power_mean_blend(paths, weights=weights_power, p=8)
power_out.to_csv("submission_powermean.csv", index=False)
print("✅ saved submission_powermean.csv")


