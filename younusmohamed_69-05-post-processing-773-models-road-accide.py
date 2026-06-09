import os, gc, re, json, math, glob, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import Ridge

# ---- YOUR ROOTS (edit only if your structure changes) ----
ROOT_DIR = Path("/kaggle/input/s05e10-outputs") / "S05E10 - Outputs" / "S05E10 - Outputs"
RESULTS_PATH = Path("/kaggle/input/s05e10-outputs") / "S05E10 - Outputs" / "Results.xlsx"

# output folders
OUT_DIR = Path("/kaggle/working")
ENS_SUB_DIR = OUT_DIR / "ensembles"
ENS_SUB_DIR.mkdir(parents=True, exist_ok=True)

# competition target
ID_COL = "id"
TARGET  = "accident_risk"

print("[INFO] ROOT_DIR:", ROOT_DIR)
print("[INFO] RESULTS_PATH:", RESULTS_PATH)


def _norm_key_from_fname(p: Path) -> str:
    # model key = file stem without common prefixes
    s = p.stem
    s = re.sub(r"^(oof_|submission_)", "", s, flags=re.I)
    return s

def discover_files(root: Path):
    # OOF folders: accept names like 'oof', 'OOF*', 'OOF Predictions - Folder*'
    oof_globs = [
        "**/oof/*.csv",
        "**/OOF/*.csv",
        "**/OOF*/*.csv",
        "**/OOF Predictions*/*.csv",
    ]
    # Submission folders: 'submissions', 'Submission Files*'
    sub_globs = [
        "**/submissions/*.csv",
        "**/Submissions/*.csv",
        "**/Submission Files*/*.csv",
        "**/Submission*/*.csv",
    ]
    oof_files = []
    for pat in oof_globs:
        oof_files += list(root.glob(pat))
    sub_files = []
    for pat in sub_globs:
        sub_files += list(root.glob(pat))

    # unique & sort for reproducibility
    oof_files = sorted(set(oof_files))
    sub_files = sorted(set(sub_files))

    print(f"[DISCOVER] OOF files: {len(oof_files)} | SUB files: {len(sub_files)}")
    return oof_files, sub_files

def load_oof_map(oof_files):
    m = {}
    for p in oof_files:
        try:
            df = pd.read_csv(p)
            # Expect columns: id, accident_risk (true), oof_pred
            cols = {c.lower(): c for c in df.columns}
            if "oof_pred" not in cols and "oof" in cols:
                df = df.rename(columns={cols["oof"]: "oof_pred"})
            # sanity
            if not {"oof_pred"}.issubset(set(df.columns)):
                continue
            k = _norm_key_from_fname(p)
            m[k] = df[[ID_COL, TARGET, "oof_pred"]].copy()
        except Exception:
            pass
    print(f"[LOAD] OOF map loaded: {len(m)} keys")
    return m

def load_sub_map(sub_files):
    m = {}
    for p in sub_files:
        try:
            df = pd.read_csv(p)
            # Expect columns: id, accident_risk
            if not {ID_COL, TARGET}.issubset(df.columns):
                continue
            k = _norm_key_from_fname(p)
            m[k] = df[[ID_COL, TARGET]].copy()
        except Exception:
            pass
    print(f"[LOAD] SUB map loaded: {len(m)} keys")
    return m

oof_files, sub_files = discover_files(ROOT_DIR)
oof_map = load_oof_map(oof_files)
sub_map = load_sub_map(sub_files)

common_keys = sorted(set(oof_map.keys()) & set(sub_map.keys()))
print(f"[COMMON] models with BOTH OOF & SUB: {len(common_keys)}")
if len(common_keys) == 0:
    raise RuntimeError("No overlapping OOF and submission files found. Check ROOT_DIR or folder names.")


def load_results_table(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        print("[WARN] Results.xlsx not found; build from discovered keys only.")
        rows = [{"model_key": k} for k in sorted(set(oof_map.keys()) | set(sub_map.keys()))]
        return pd.DataFrame(rows)

    if path.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    std = {c.lower().strip(): c for c in df.columns}
    rename = {}
    if "model" in std:           rename[std["model"]] = "model_key"
    if "rmse" in std:            rename[std["rmse"]] = "rmse"
    if "mae" in std:             rename[std["mae"]] = "mae"
    if "r2" in std:              rename[std["r2"]] = "r2"
    if "submission" in std:      rename[std["submission"]] = "public"  # public LB score
    if "rmse rank" in std:       rename[std["rmse rank"]] = "rmse_rank"
    if "mae rank" in std:        rename[std["mae rank"]] = "mae_rank"
    if "r2 rank" in std:         rename[std["r2 rank"]] = "r2_rank"
    if "submission rank" in std: rename[std["submission rank"]] = "public_rank"
    if "fe_key" in std:          rename[std["fe_key"]] = "fe_key"
    if "prep" in std:            rename[std["prep"]] = "prep"
    if "folds" in std:           rename[std["folds"]] = "folds"
    if "time_min" in std:        rename[std["time_min"]] = "time_min"
    if "trial (hp)" in std:      rename[std["trial (hp)"]] = "trial_hp"

    df = df.rename(columns=rename)
    keep_cols = [
        "model_key","rmse","mae","r2","public",
        "rmse_rank","mae_rank","r2_rank","public_rank",
        "fe_key","prep","folds","time_min","trial_hp"
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # coerce numerics
    for c in ["rmse","mae","r2","public","rmse_rank","mae_rank","r2_rank","public_rank","folds","time_min","trial_hp"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["model_key"]).drop_duplicates(subset=["model_key"]).reset_index(drop=True)
    return df

results_df = load_results_table(RESULTS_PATH)
print("[RESULTS] rows:", len(results_df))
results_df.head(3)


def rmse(y, p): return mean_squared_error(y, p, squared=False)

def evaluate(y, p):
    return {
        "rmse": rmse(y, p),
        "mae": mean_absolute_error(y, p),
        "r2": r2_score(y, p)
    }

# get ground truth y from any OOF file
first_key = next(iter(oof_map.keys()))
y_df = oof_map[first_key][[ID_COL, TARGET]].copy().sort_values(ID_COL).reset_index(drop=True)
y_true = y_df[TARGET].values
ids_train = y_df[ID_COL].values

# get test id universe from first submission
first_sub_key = next(iter(sub_map.keys()))
test_ids = sub_map[first_sub_key][ID_COL].values

def get_oof_vec(model_key):
    df = oof_map[model_key][[ID_COL, "oof_pred"]].copy().sort_values(ID_COL).reset_index(drop=True)
    # align to ids_train
    if not np.array_equal(df[ID_COL].values, ids_train):
        df = df.set_index(ID_COL).reindex(ids_train).reset_index()
    return df["oof_pred"].values

def get_sub_vec(model_key):
    df = sub_map[model_key][[ID_COL, TARGET]].copy().sort_values(ID_COL).reset_index(drop=True)
    if not np.array_equal(df[ID_COL].values, test_ids):
        df = df.set_index(ID_COL).reindex(test_ids).reset_index()
    return df[TARGET].values

def save_submission(tag: str, preds: np.ndarray):
    out = pd.DataFrame({ID_COL: test_ids, TARGET: np.clip(preds, 0.0, 1.0)})
    path = ENS_SUB_DIR / f"submission_{tag}.csv"
    out.to_csv(path, index=False)
    return str(path)

def blend_weighted(preds_mat: np.ndarray, weights: np.ndarray):
    w = np.asarray(weights, dtype=float)
    w = np.where(np.isfinite(w), w, 0.0)
    if w.sum() <= 0:
        w = np.ones_like(w)
    w = w / w.sum()
    return np.dot(preds_mat, w)


# keep only models present in (oof & sub)
results_df = results_df[results_df["model_key"].isin(common_keys)].reset_index(drop=True)
print("[FILTERED results] rows:", len(results_df))

def topk_by(col, k_list=(10,20,30,50,100), ascending=True):
    out = {}
    col_ok = results_df[col].dropna()
    if col_ok.empty:
        return out
    df = results_df.dropna(subset=[col]).copy()
    df = df.sort_values(col, ascending=ascending)
    for k in k_list:
        out[k] = df.head(k)["model_key"].tolist() if len(df) >= k else df["model_key"].tolist()
    return out

topk_rmse = topk_by("rmse", ascending=True)
topk_mae  = topk_by("mae",  ascending=True)
topk_r2   = topk_by("r2",   ascending=False)
topk_pub  = topk_by("public", ascending=True)  # if 'submission' is score (lower is better); flip if higher-better


# matrices for fast blending
keys_all = sorted(common_keys)

# Train OOF matrix (N_train x M)
OOF_MAT = np.column_stack([get_oof_vec(k) for k in keys_all])
# Test SUB matrix  (N_test x M)
SUB_MAT = np.column_stack([get_sub_vec(k) for k in keys_all])

print("[MATS] OOF:", OOF_MAT.shape, "SUB:", SUB_MAT.shape, "Models:", len(keys_all))


summary_rows = []

# 1) simple mean of ALL
p_oof = OOF_MAT.mean(axis=1)
p_sub = SUB_MAT.mean(axis=1)
m = evaluate(y_true, p_oof)
path = save_submission("mean_all", p_sub)
summary_rows.append({"method":"mean_all","n_models":len(keys_all), **m, "file":path})

# 2) weight by 1/rmse and 1/rmse^2
rmse_map = results_df.set_index("model_key")["rmse"].to_dict()
w1 = np.array([1.0/max(rmse_map.get(k, np.nan), 1e-12) for k in keys_all])
w2 = np.array([1.0/(max(rmse_map.get(k, np.nan), 1e-12)**2) for k in keys_all])

p_oof = blend_weighted(OOF_MAT, w1)
p_sub = blend_weighted(SUB_MAT, w1)
m = evaluate(y_true, p_oof)
path = save_submission("w_rmse_inv", p_sub)
summary_rows.append({"method":"w_rmse_inv","n_models":len(keys_all), **m, "file":path})

p_oof = blend_weighted(OOF_MAT, w2)
p_sub = blend_weighted(SUB_MAT, w2)
m = evaluate(y_true, p_oof)
path = save_submission("w_rmse_inv2", p_sub)
summary_rows.append({"method":"w_rmse_inv2","n_models":len(keys_all), **m, "file":path})


def mean_of_subset(model_list, tag):
    if not model_list:
        return None
    idx = [keys_all.index(k) for k in model_list if k in keys_all]
    if not idx:
        return None
    p_oof = OOF_MAT[:, idx].mean(axis=1)
    p_sub = SUB_MAT[:, idx].mean(axis=1)
    m = evaluate(y_true, p_oof)
    path = save_submission(tag, p_sub)
    summary_rows.append({"method":tag, "n_models":len(idx), **m, "file":path})

for k, lst in topk_rmse.items():
    mean_of_subset(lst, f"mean_top{k}_rmse")

for k, lst in topk_mae.items():
    mean_of_subset(lst, f"mean_top{k}_mae")

for k, lst in topk_r2.items():
    mean_of_subset(lst, f"mean_top{k}_r2")

for k, lst in topk_pub.items():
    mean_of_subset(lst, f"mean_top{k}_public")


# fit ridge on OOF (features = model oofs) to y_true
alpha = 1.0
ridge = Ridge(alpha=alpha, fit_intercept=True)
ridge.fit(OOF_MAT, y_true)
p_oof = ridge.predict(OOF_MAT)
p_sub = ridge.predict(SUB_MAT)

m = evaluate(y_true, p_oof)
path = save_submission(f"ridge_stack_a{alpha}", p_sub)
summary_rows.append({"method":f"ridge_stack_a{alpha}","n_models":len(keys_all), **m, "file":path})

# also ridge on **top-50 by rmse** if available
if 50 in topk_rmse and len(topk_rmse[50]) > 0:
    idx = [keys_all.index(k) for k in topk_rmse[50] if k in keys_all]
    Xoof = OOF_MAT[:, idx]; Xsub = SUB_MAT[:, idx]
    ridge2 = Ridge(alpha=0.5, fit_intercept=True)
    ridge2.fit(Xoof, y_true)
    p_oof = ridge2.predict(Xoof)
    p_sub = ridge2.predict(Xsub)
    m = evaluate(y_true, p_oof)
    path = save_submission("ridge_stack_top50_rmse_a0.5", p_sub)
    summary_rows.append({"method":"ridge_stack_top50_rmse_a0.5","n_models":len(idx), **m, "file":path})


# Hill Climbing (Greedy Forward Selection, Equal Weights)
# Seeds with best single model (by OOF RMSE), then adds models if they strictly improve OOF RMSE.
# Efficient via running sum and batched evaluation to avoid huge temporary matrices.

# ------------- Settings -------------
BATCH_SIZE = 25       # evaluate candidates in chunks to keep memory in check
TOL = 1e-8            # strict improvement tolerance
MAX_ADD = 300         # hard cap on how many models to include (safety)
VERBOSE = True

n_models = len(keys_all)
assert OOF_MAT.shape[1] == n_models and SUB_MAT.shape[1] == n_models, "Matrix/model alignment error."

# --- Step 1: singleton scores to find the best seed ---
if VERBOSE: print("[HC] Scoring single models to choose the seed...")
single_rmse = np.empty(n_models, dtype=np.float64)
for i in range(n_models):
    single_rmse[i] = rmse(y_true, OOF_MAT[:, i])

seed_idx = int(np.argmin(single_rmse))
selected = [seed_idx]
remaining = [i for i in range(n_models) if i != seed_idx]

# running sums (use float64 for numerical stability)
sum_oof = OOF_MAT[:, seed_idx].astype(np.float64).copy()
sum_sub = SUB_MAT[:, seed_idx].astype(np.float64).copy()
best_rmse = float(single_rmse[seed_idx])

if VERBOSE:
    print(f"[HC] Seed: {keys_all[seed_idx]} | RMSE={best_rmse:.9f}")

# --- Step 2: greedy forward adds with batching ---
while remaining and len(selected) < MAX_ADD:
    best_cand = None
    best_cand_rmse = best_rmse
    k = len(selected)  # current pool size

    # Evaluate all remaining candidates; compute (sum + col)/(k+1) over OOF in batches
    for start in range(0, len(remaining), BATCH_SIZE):
        chunk = remaining[start:start + BATCH_SIZE]
        # new OOF means if each candidate in chunk is added
        # shape: (N_train, len(chunk))
        new_means = (sum_oof[:, None] + OOF_MAT[:, chunk]) / (k + 1)

        # compute RMSE for each candidate in the chunk (vectorized)
        diffs = new_means - y_true[:, None]
        mse = np.mean(diffs * diffs, axis=0)
        rmse_chunk = np.sqrt(mse)

        # check best in this chunk
        min_j = int(np.argmin(rmse_chunk))
        cand_rmse = float(rmse_chunk[min_j])
        if cand_rmse + TOL < best_cand_rmse:
            best_cand_rmse = cand_rmse
            best_cand = chunk[min_j]

    # Accept the best candidate if it strictly improves
    if best_cand is not None:
        # update running sums with the accepted candidate
        sum_oof += OOF_MAT[:, best_cand]
        sum_sub += SUB_MAT[:, best_cand]
        best_rmse = best_cand_rmse

        selected.append(best_cand)
        remaining.remove(best_cand)

        if VERBOSE:
            print(f"[HC] + {keys_all[best_cand]:<40} | k={len(selected):>3} | RMSE={best_rmse:.9f}")
    else:
        if VERBOSE:
            print("[HC] No further improvement. Stopping.")
        break

# --- Step 3: finalize predictions and save ---
if selected:
    p_oof = (sum_oof / len(selected)).astype(np.float32)
    p_sub = (sum_sub / len(selected)).astype(np.float32)

    m = evaluate(y_true, p_oof)
    tag = f"hillclimb_eqw_{len(selected)}"
    path = save_submission(tag, p_sub)

    summary_rows.append({
        "method": tag,
        "n_models": len(selected),
        **m,
        "file": path
    })

    print(f"[HC] DONE | models={len(selected)} | OOF RMSE={m['rmse']:.9f} | saved -> {path}")
    # Optional: also print selected model names
    if VERBOSE:
        print("[HC] Selected models (in order):")
        for i, idx in enumerate(selected, 1):
            print(f"  {i:>3}. {keys_all[idx]}")
else:
    print("[HC] No selection could be made (unexpected). Skipping hill climb.")


from typing import List, Tuple

# --- helpers ---
def _compute_single_rmse_all() -> np.ndarray:
    """Compute OOF RMSE for each model column in OOF_MAT."""
    n_models = OOF_MAT.shape[1]
    out = np.empty(n_models, dtype=np.float64)
    for i in range(n_models):
        out[i] = rmse(y_true, OOF_MAT[:, i])
    return out

def _topk_indices_by_rmse(K: int) -> List[int]:
    """
    Return indices into keys_all for the top-K models by RMSE.
    Prefer results_df['rmse'] when available; otherwise compute from OOF_MAT.
    """
    # Use results_df if it has rmse for the current models
    df = results_df.copy()
    df = df[df["model_key"].isin(keys_all)]
    if "rmse" in df.columns and df["rmse"].notna().any():
        # align to keys_all order with rmse values; fallback to computed if missing
        rmse_map = df.set_index("model_key")["rmse"].to_dict()
        computed_rmse = _compute_single_rmse_all()
        rmse_vec = np.array([rmse_map.get(k, np.nan) for k in keys_all], dtype=float)
        # fill NaNs with computed rmse
        nan_mask = ~np.isfinite(rmse_vec)
        rmse_vec[nan_mask] = computed_rmse[nan_mask]
    else:
        # compute entirely from OOF_MAT
        rmse_vec = _compute_single_rmse_all()

    order = np.argsort(rmse_vec)  # ascending (lower rmse first)
    K_eff = min(K, len(order))
    return list(order[:K_eff])

def _weights_from_rmse(idxs: List[int]) -> np.ndarray:
    """Weights = 1/RMSE for the selected indices (computed from OOF if needed)."""
    # Try to use results_df first
    rmse_series = None
    if "rmse" in results_df.columns and results_df["rmse"].notna().any():
        rmse_map = results_df.set_index("model_key")["rmse"].to_dict()
        # map indices -> model_key -> rmse
        vals = []
        for i in idxs:
            key = keys_all[i]
            v = rmse_map.get(key, np.nan)
            vals.append(v)
        rmse_series = np.array(vals, dtype=float)

    # Fill missing with computed RMSE if needed
    if rmse_series is None or not np.isfinite(rmse_series).all():
        comp = _compute_single_rmse_all()
        if rmse_series is None:
            rmse_series = comp[idxs]
        else:
            nan_mask = ~np.isfinite(rmse_series)
            rmse_series[nan_mask] = comp[np.array(idxs)[nan_mask]]

    # avoid division by zero
    rmse_series = np.clip(rmse_series, 1e-12, None)
    w = 1.0 / rmse_series
    w = w / w.sum()
    return w

def _rank_0_1(vec: np.ndarray) -> np.ndarray:
    """
    Rank a 1D vector to [0,1] using average ranks for ties.
    """
    s = pd.Series(vec)
    r = s.rank(method="average").to_numpy()  # 1..N
    return (r - 1.0) / (len(s) - 1.0) if len(s) > 1 else np.zeros_like(r)

def _rank_avg(oof_sub_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rank-average across columns for both OOF and SUB matrices of shape (N, m).
    Returns (p_oof, p_sub), each in [0,1].
    """
    # OOF
    oof_rank_cols = []
    for j in range(oof_sub_mat[0].shape[1] if isinstance(oof_sub_mat, tuple) else oof_sub_mat.shape[1]):  # compatibility
        pass

    # OOF
    ranks_oof = np.column_stack([_rank_0_1(OOF_sel[:, j]) for j in range(OOF_sel.shape[1])])
    p_oof = ranks_oof.mean(axis=1)

    # SUB
    ranks_sub = np.column_stack([_rank_0_1(SUB_sel[:, j]) for j in range(SUB_sel.shape[1])])
    p_sub = ranks_sub.mean(axis=1)

    return p_oof, p_sub

def _rank_avg_for_indices(idxs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Apply rank average only on the selected indices."""
    OOF_sel = OOF_MAT[:, idxs]
    SUB_sel = SUB_MAT[:, idxs]
    ranks_oof = np.column_stack([_rank_0_1(OOF_sel[:, j]) for j in range(OOF_sel.shape[1])])
    p_oof = ranks_oof.mean(axis=1)
    ranks_sub = np.column_stack([_rank_0_1(SUB_sel[:, j]) for j in range(SUB_sel.shape[1])])
    p_sub = ranks_sub.mean(axis=1)
    return p_oof, p_sub

def _mean_for_indices(idxs: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    OOF_sel = OOF_MAT[:, idxs]
    SUB_sel = SUB_MAT[:, idxs]
    return OOF_sel.mean(axis=1), SUB_sel.mean(axis=1)

def _weighted_for_indices(idxs: List[int], w: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    OOF_sel = OOF_MAT[:, idxs]
    SUB_sel = SUB_MAT[:, idxs]
    # weights are length m; compute dot across columns
    p_oof = np.dot(OOF_sel, w)
    p_sub = np.dot(SUB_sel, w)
    return p_oof, p_sub

# --- main: build top-K sets and write submissions ---
K_LIST = [2, 3, 5, 10]
for K in K_LIST:
    idxs = _topk_indices_by_rmse(K)
    if len(idxs) < 2:
        print(f"[TopK] Only {len(idxs)} model(s) available for K={K}; skipping.")
        continue

    # 1) Simple mean
    p_oof, p_sub = _mean_for_indices(idxs)
    m = evaluate(y_true, p_oof)
    tag = f"mean_top{K}_rmse"
    path = save_submission(tag, p_sub)
    summary_rows.append({"method": tag, "n_models": len(idxs), **m, "file": path})
    print(f"[TopK:{K}] mean -> RMSE={m['rmse']:.6f} | saved {path}")

    # 2) Weighted by inverse RMSE (computed over OOF)
    w = _weights_from_rmse(idxs)
    p_oof, p_sub = _weighted_for_indices(idxs, w)
    m = evaluate(y_true, p_oof)
    tag = f"w1rmse_top{K}_rmse"
    path = save_submission(tag, p_sub)
    summary_rows.append({"method": tag, "n_models": len(idxs), **m, "file": path})
    print(f"[TopK:{K}] w(1/rmse) -> RMSE={m['rmse']:.6f} | saved {path}")

    # 3) Ranked average
    p_oof, p_sub = _rank_avg_for_indices(idxs)
    m = evaluate(y_true, p_oof)
    tag = f"rankavg_top{K}_rmse"
    path = save_submission(tag, p_sub)
    summary_rows.append({"method": tag, "n_models": len(idxs), **m, "file": path})
    print(f"[TopK:{K}] rankavg -> RMSE={m['rmse']:.6f} | saved {path}")


summary_df = pd.DataFrame(summary_rows).sort_values("rmse").reset_index(drop=True)
cmp_path = OUT_DIR / "ensemble_comparison.csv"
summary_df.to_csv(cmp_path, index=False)
summary_df.head(20)


plt.figure(figsize=(9,6))
top = summary_df.head(15)
plt.barh(range(len(top)), top["rmse"].values)
plt.yticks(range(len(top)), top["method"].values, fontsize=9)
plt.gca().invert_yaxis()
plt.title("Top ensembles by OOF RMSE")
plt.xlabel("OOF RMSE")
plt.tight_layout()
plt.show()

print("[SAVED] comparison:", cmp_path)
print("[SAVED] submissions in:", ENS_SUB_DIR)




