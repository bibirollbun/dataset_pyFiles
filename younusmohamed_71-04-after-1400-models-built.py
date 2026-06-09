import os, glob, json, math, warnings, re
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.isotonic import IsotonicRegression
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD

# optional deps
try:
    from scipy.optimize import nnls, minimize
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False

try:
    import lightgbm as lgb
    LGB_OK = True
except Exception:
    LGB_OK = False

print("SciPy available:", SCIPY_OK)
print("LightGBM available:", LGB_OK)


# Competition data
TRAIN_PATH  = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH   = "/kaggle/input/playground-series-s5e12/test.csv"
SAMPLE_SUB  = "/kaggle/input/playground-series-s5e12/sample_submission.csv"

# (Optional) original dataset (not needed for ensembling)
ORIG_PATH   = "/kaggle/input/diabetes-health-indicators-dataset/diabetes_dataset.csv"

# Root folder containing outputs
ROOT_DIR    = "/kaggle/input/s05e12-outputs-diabetes-prediction"

# Subfolders
OOF_DIRS    = [f"{ROOT_DIR}/OOF Files/OOF Files"]
SUB_DIRS    = [f"{ROOT_DIR}/Submission Files/Submission Files"]
RESULT_DIRS = [f"{ROOT_DIR}/Result Files/Result Files"]
FI_DIRS     = [f"{ROOT_DIR}/Feature Importances/Feature Importances"]

# Excel combined results (with cv_auc_mean etc.)
COMBINED_RESULTS_XLSX = f"{ROOT_DIR}/Combined Results (1408 Models).csv.xlsx"
COMBINED_SHEET_NAME   = "Combined Results (1408 Models)"

# Columns / constants
ID_COL     = "id"
TARGET_COL = "diagnosed_diabetes"

# Output location
OUT_DIR = "/kaggle/working/ensemble_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# Ensembling knobs
TOPK_LIST             = [3, 5, 10, 20, 50, 100]
CORR_DUP_CUTOFF       = 0.9997   # a tad stricter than 0.9999 to drop more clones
LOW_VAR_EPS           = 1e-12
DIVERSE_MAX_CORR      = 0.95
TOP_PCT_FOR_SHORTLIST = 0.20

# Switches & shortlist safety
PRUNE_NEAR_DUPES = True
MIN_SHORTLIST    = 50
RELAX_CORR_STEPS = [0.98, 0.99, 0.999]

# New knobs for upgraded sections
CLUSTER_K                = 60     # number of diversity clusters to sample reps from
RIDGE_L2                 = 0.03   # L2 for ridge meta
STACK_FOLDS              = 5      # out-of-sample meta folds
SFFS_MAX_K               = 50     # cap members chosen by SFFS
RANK_RIDGE_L2            = 0.05
SVD_COMPONENTS           = 64

# Heavy solver throttles
MAX_COLS_FOR_WEIGHTS    = 200
ROW_SAMPLE_FOR_WEIGHTS  = 200_000
N_SPLITS_STACK_LITE     = 3

np.set_printoptions(suppress=True, precision=6)
print("Configured.")


train  = pd.read_csv(TRAIN_PATH)
test   = pd.read_csv(TEST_PATH)
sample = pd.read_csv(SAMPLE_SUB)

y = train[[ID_COL, TARGET_COL]].set_index(ID_COL).sort_index().squeeze()
test_ids = test[[ID_COL]].set_index(ID_COL).sort_index().index

def _glob_many(dirs, patterns):
    paths = []
    for d in dirs:
        for p in patterns:
            paths.extend(glob.glob(os.path.join(d, p)))
    return sorted(list(set(paths)))

oof_files = _glob_many(OOF_DIRS, ["*.csv"])
sub_files = _glob_many(SUB_DIRS, ["*.csv"])
res_files = _glob_many(RESULT_DIRS, ["*.csv"])
fi_files  = _glob_many(FI_DIRS, ["*.csv"])

print("#files -> OOF:", len(oof_files), "| SUB:", len(sub_files), "| RES:", len(res_files), "| FI:", len(fi_files))


# Try to read meta sheet if present
try:
    meta_raw = pd.read_excel(
        COMBINED_RESULTS_XLSX,
        sheet_name=COMBINED_SHEET_NAME,
        engine="openpyxl"
    )
    if "cv_auc_mean" not in meta_raw.columns:
        meta_raw["cv_auc_mean"] = np.nan
    META_OK = True
except Exception:
    meta_raw = pd.DataFrame({"cv_auc_mean": []})
    META_OK = False


def _classify_oof_or_sub(path):
    try:
        df = pd.read_csv(path, nrows=5)
        cols = set(df.columns.str.lower())
        if "oof_pred" in cols: return "oof"
        if TARGET_COL.lower() in cols: return "sub"
        b = os.path.basename(path).lower()
        if "oof" in b: return "oof"
        if ("sub" in b) or ("submission" in b): return "sub"
    except Exception:
        pass
    return None

def read_oof_series(path, id_col="id", oof_col="oof_pred"):
    df = pd.read_csv(path)
    lc = {c.lower(): c for c in df.columns}
    idc = lc.get(id_col, lc.get("id"))
    if idc is None:
        raise ValueError(f"{path}: no id column")
    if oof_col not in lc:
        cand = [c for c in df.columns if ("oof" in c.lower() and "pred" in c.lower())]
        if not cand:
            raise ValueError(f"{path}: no oof_pred col")
        oofc = cand[0]
    else:
        oofc = lc[oof_col]
    s = df[[idc, oofc]].copy()
    s.columns = [ID_COL, "oof_pred"]
    s = s.set_index(ID_COL).squeeze().astype(float)
    return s.reindex(y.index)

def read_sub_series(path, id_col="id", target_col=TARGET_COL):
    df = pd.read_csv(path)
    lc = {c.lower(): c for c in df.columns}
    idc = lc.get(id_col, lc.get("id"))
    tgc = lc.get(target_col.lower())
    if (idc is None) or (tgc is None):
        # fallback: last non-id column—guard it
        others = [c for c in df.columns if c.lower() != "id"]
        assert len(others)>0, f"{path}: cannot find target column"
        tgc = others[-1]
        idc = idc or "id"
    s = df[[idc, tgc]].copy()
    s.columns = [ID_COL, "pred"]
    s = s.set_index(ID_COL).squeeze().astype(float)
    return s.reindex(test_ids)

def stem(p): return os.path.splitext(os.path.basename(p))[0]

subs_by_stem = {stem(p): p for p in sub_files if _classify_oof_or_sub(p) == "sub"}

def best_match_sub(oof_stem):
    if oof_stem in subs_by_stem:
        return subs_by_stem[oof_stem]
    ta = set(re.split(r"[_\-\.]+", oof_stem.lower()))
    best = (0.0, None)
    for st, path in subs_by_stem.items():
        tb = set(re.split(r"[_\-\.]+", st.lower()))
        if not ta or not tb: 
            continue
        j = len(ta & tb) / max(1, len(ta | tb))
        if j > best[0]:
            best = (j, path)
    return best[1]


rows = []
oof_matrix = {}
skipped = 0

for p in oof_files:
    if _classify_oof_or_sub(p) != "oof":
        continue
    try:
        s = read_oof_series(p)
        if s.isnull().all():
            continue
        auc_oof = roc_auc_score(y.values.astype(int), s.values.astype(float))
        mid = stem(p)
        sp = best_match_sub(mid)
        rows.append({"model_id": mid, "oof_path": p, "sub_path": sp, "oof_auc": auc_oof})
        oof_matrix[mid] = s.values.astype(float)
    except Exception:
        skipped += 1
        continue

registry_df = pd.DataFrame(rows).sort_values("oof_auc", ascending=False).reset_index(drop=True)
print("Registry size:", registry_df.shape, "| skipped:", skipped)

# Attach cv_auc_mean if available (tokenized fuzzy map)
if META_OK and ("cv_auc_mean" in meta_raw.columns):
    meta_text_cols = [c for c in meta_raw.columns if meta_raw[c].dtype == object]
    fused = {}
    bags = []
    for i, r in meta_raw.iterrows():
        parts = []
        for c in meta_text_cols:
            v = r[c]
            if pd.isna(v): continue
            parts.append(str(v))
        bags.append((" ".join(parts).lower(), i))
    for mid in registry_df["model_id"]:
        tokens = set(re.split(r"[_\-\.\s]+", mid.lower()))
        best = (-1, None)
        for bag, idx in bags:
            overlap = len(tokens & set(bag.split()))
            if overlap > best[0]:
                best = (overlap, idx)
        if best[1] is not None and best[0] >= 1:
            fused[mid] = meta_raw.loc[best[1], "cv_auc_mean"]
    registry_df["cv_auc_mean_meta"] = registry_df["model_id"].map(fused)
else:
    registry_df["cv_auc_mean_meta"] = np.nan

display(registry_df.head(8))


model_ids = registry_df["model_id"].tolist()
X_oof = np.vstack([oof_matrix[mid] for mid in model_ids]).T  # (n_train, n_models)

# low-variance prune
var = X_oof.var(axis=0)
keep = var > LOW_VAR_EPS
if keep.sum() < len(keep):
    print("Dropping low-var models:", (~keep).sum())
X_oof = X_oof[:, keep]
registry_df = registry_df.loc[keep].reset_index(drop=True)
model_ids = registry_df["model_id"].tolist()

# near-duplicate prune
if PRUNE_NEAR_DUPES:
    C = np.corrcoef(X_oof.T)
    to_drop = set()
    for i in range(len(model_ids)):
        if i in to_drop: 
            continue
        for j in range(i+1, len(model_ids)):
            if j in to_drop or np.isnan(C[i, j]): 
                continue
            if C[i, j] >= CORR_DUP_CUTOFF:
                to_drop.add(j)
    if to_drop:
        mask = np.array([False if i in to_drop else True for i in range(len(model_ids))])
        print("Dropping near-duplicates:", len(to_drop))
        X_oof = X_oof[:, mask]
        registry_df = registry_df.loc[mask].reset_index(drop=True)
        model_ids = registry_df["model_id"].tolist()

print("Post-prune OOF shape:", X_oof.shape, "| models:", len(model_ids))


score = registry_df["cv_auc_mean_meta"].fillna(registry_df["oof_auc"])
ranked = registry_df.assign(score=score).sort_values("score", ascending=False).reset_index(drop=True)

top_n = max(MIN_SHORTLIST, int(len(ranked) * TOP_PCT_FOR_SHORTLIST))
seed = ranked.head(top_n).copy()

def make_diverse_shortlist(seed_df, max_corr):
    pool_idx, pool_vecs = [], []
    for idx, row in seed_df.iterrows():
        col_idx = ranked.index[ranked["model_id"] == row["model_id"]][0]
        vec = X_oof[:, col_idx]
        if not pool_vecs:
            pool_idx.append(idx); pool_vecs.append(vec); continue
        corrs = [np.corrcoef(vec, v)[0,1] for v in pool_vecs]
        if np.nanmax(corrs) <= max_corr:
            pool_idx.append(idx); pool_vecs.append(vec)
    return seed_df.loc[pool_idx].reset_index(drop=True)

shortlist_df = make_diverse_shortlist(seed, DIVERSE_MAX_CORR)
if len(shortlist_df) < MIN_SHORTLIST:
    for cap in RELAX_CORR_STEPS:
        shortlist_df = make_diverse_shortlist(seed, cap)
        if len(shortlist_df) >= MIN_SHORTLIST:
            print(f"Relaxed diversity cap to {cap} to reach MIN_SHORTLIST={MIN_SHORTLIST}.")
            break

if len(shortlist_df) < MIN_SHORTLIST:
    have = set(shortlist_df["model_id"])
    for _, r in seed.iterrows():
        if r["model_id"] not in have:
            shortlist_df = pd.concat([shortlist_df, r.to_frame().T], ignore_index=True)
            have.add(r["model_id"])
        if len(shortlist_df) >= MIN_SHORTLIST:
            print("Padded shortlist with best remaining to hit MIN_SHORTLIST.")
            break

shortlist_ids  = shortlist_df["model_id"].tolist()
shortlist_cols = [ranked.index[ranked["model_id"] == mid][0] for mid in shortlist_ids]
X_oof_short    = X_oof[:, shortlist_cols]

print("Shortlist size:", len(shortlist_ids))
display(shortlist_df[["model_id","oof_auc","cv_auc_mean_meta"]].head(12))


def read_sub_matrix(ids):
    cols = []
    ok_ids = []
    for mid in ids:
        spath = registry_df.loc[registry_df["model_id"] == mid, "sub_path"].values
        spath = spath[0] if len(spath) else None
        if not spath or not os.path.exists(spath): 
            continue
        try:
            s = read_sub_series(spath).values.astype(float)
            cols.append(s); ok_ids.append(mid)
        except Exception:
            pass
    if not cols: 
        return None, []
    return np.vstack(cols).T, ok_ids

X_test_short, shortlist_ids_sub = read_sub_matrix(shortlist_ids)
assert X_test_short is not None, "No matching SUB files for shortlist—check SUB_DIRS structure"

mask = [mid in shortlist_ids_sub for mid in shortlist_ids]
X_oof_short = X_oof_short[:, mask]
shortlist_df = shortlist_df.loc[mask].reset_index(drop=True)
shortlist_ids = shortlist_df["model_id"].tolist()

print("Aligned shortlist -> OOF:", X_oof_short.shape, "| TEST:", X_test_short.shape, "| k:", len(shortlist_ids))


y_true = y.values.astype(int)

def auc_safe(y_true, y_score):
    m = np.isfinite(y_score)
    if m.sum() == 0: 
        return np.nan
    return roc_auc_score(y_true[m], y_score[m])

def save_submission(tag, preds):
    sub = sample.copy()
    sub[TARGET_COL] = preds
    path = os.path.join(OUT_DIR, f"{tag}.csv")
    sub.to_csv(path, index=False)
    return path

def to_rank01(M):
    R = np.zeros_like(M, dtype=float)
    for j in range(M.shape[1]):
        r = pd.Series(M[:,j]).rank(method="average").values
        R[:,j] = (r - r.min())/max(1e-12, (r.max()-r.min()))
    return R

def cv_weighted_blend(X_oof, X_test, y, base="ridge", l2=1.0, nonneg=False, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    o_pred = np.zeros(len(y))
    t_pred = np.zeros(X_test.shape[0])
    for tr, va in skf.split(X_oof, y):
        Xtr, Xva = X_oof[tr], X_oof[va]
        ytr = y[tr]
        if base == "ridge":
            mdl = Ridge(alpha=l2, fit_intercept=False, positive=nonneg)
            mdl.fit(Xtr, ytr)
            o_pred[va] = mdl.predict(Xva)
            t_pred += mdl.predict(X_test)/n_splits
        elif base == "lr":
            lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
            lr.fit(Xtr, ytr)
            o_pred[va] = lr.predict_proba(Xva)[:,1]
            t_pred += lr.predict_proba(X_test)[:,1]/n_splits
        else:
            raise ValueError("Unknown base")
    return o_pred, t_pred

def cluster_diverse(X, k=40, random_state=42):
    # distance ~ sqrt(1 - corr)
    C = np.corrcoef(X, rowvar=False)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    D = np.sqrt(np.clip(1.0 - C, 0, 1))
    # KMeans on distance rows (simple proxy embedding)
    km = KMeans(n_clusters=min(k, D.shape[0]), n_init=10, random_state=random_state)
    labels = km.fit_predict(D)
    reps = []
    # pick best AUC within each cluster
    base_aucs = np.array([auc_safe(y_true, X[:, j]) for j in range(X.shape[1])])
    for c in range(km.n_clusters):
        idx = np.where(labels==c)[0]
        if len(idx)==0: 
            continue
        best = idx[np.argmax(base_aucs[idx])]
        reps.append(best)
    return sorted(set(reps))

def sffs_mean(X, y, max_k=50):
    chosen, pred = [], None
    best = -1.0
    remaining = list(range(X.shape[1]))
    improved = True
    while improved and len(chosen) < max_k:
        improved = False
        # forward
        best_gain, best_j, best_pred = 0, None, None
        for j in remaining:
            cand = X[:, chosen+[j]].mean(axis=1) if chosen else X[:, j]
            a = auc_safe(y, cand)
            gain = a - (best if pred is not None else 0.0)
            if gain > best_gain:
                best_gain, best_j, best_pred = gain, j, cand
        if best_j is not None and best_gain > 0:
            chosen.append(best_j); remaining.remove(best_j)
            pred = best_pred; best += best_gain; improved = True
        # backward
        if len(chosen) > 2:
            rm_gain, rm_idx, rm_pred = 0, None, None
            for k in range(len(chosen)):
                subset = [c for i, c in enumerate(chosen) if i != k]
                cand = X[:, subset].mean(axis=1)
                a = auc_safe(y, cand)
                gain = a - best
                if gain > rm_gain:
                    rm_gain, rm_idx, rm_pred = gain, k, cand
            if rm_idx is not None and rm_gain > 0:
                del chosen[rm_idx]; pred = rm_pred; best += rm_gain; improved = True
    return chosen, best


blend_rows = []
for K in TOPK_LIST:
    K = min(K, X_oof_short.shape[1])
    cols = np.arange(K)
    oK, tK = X_oof_short[:, cols], X_test_short[:, cols]

    # mean
    o, t = oK.mean(axis=1), tK.mean(axis=1)
    a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_mean", t)
    blend_rows.append(["mean", K, a, p])

    # median
    o, t = np.median(oK, axis=1), np.median(tK, axis=1)
    a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_median", t)
    blend_rows.append(["median", K, a, p])

    # rank average
    def rank_average(oof_mat, test_mat):
        oof_r, test_r = to_rank01(oof_mat), to_rank01(test_mat)
        return oof_r.mean(axis=1), test_r.mean(axis=1)
    o, t = rank_average(oK, tK)
    a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_rankavg", t)
    blend_rows.append(["rankavg", K, a, p])

blend_df = pd.DataFrame(blend_rows, columns=["blend","K","oof_auc","submission_path"]).sort_values("oof_auc", ascending=False)
display(blend_df.head(10))


rep_idx = cluster_diverse(X_oof_short, k=CLUSTER_K, random_state=42)
print("Cluster reps:", len(rep_idx))
Xo, Xt = X_oof_short[:, rep_idx], X_test_short[:, rep_idx]

o_cv_ridge, t_cv_ridge = cv_weighted_blend(
    Xo, Xt, y_true, base="ridge", l2=RIDGE_L2, nonneg=True, n_splits=STACK_FOLDS, seed=42
)
print("CV-stacked Ridge (NN) AUC:", auc_safe(y_true, o_cv_ridge))
path_ridge = save_submission(f"blend_cluster{CLUSTER_K}_cv_ridge_nn", t_cv_ridge)
path_ridge


Xo_r, Xt_r = to_rank01(Xo), to_rank01(Xt)

o_cv_rank_lr, t_cv_rank_lr = cv_weighted_blend(
    Xo_r, Xt_r, y_true, base="lr", n_splits=STACK_FOLDS, seed=42
)
print("CV-stacked Rank-LR AUC:", auc_safe(y_true, o_cv_rank_lr))
path_ranklr = save_submission(f"blend_cluster{CLUSTER_K}_rank_cv_lr", t_cv_rank_lr)

# Blend the two metas (tune 0.6/0.4 if needed)
alpha = 0.6
t_meta_combo = alpha * t_cv_ridge + (1 - alpha) * t_cv_rank_lr
save_submission(f"blend_cluster{CLUSTER_K}_meta_combo_a{alpha:.2f}", t_meta_combo)


chosen_sffs, best_auc_sffs = sffs_mean(X_oof_short, y_true, max_k=SFFS_MAX_K)
print("SFFS chosen:", len(chosen_sffs), "AUC:", best_auc_sffs)
t_sffs = X_test_short[:, chosen_sffs].mean(axis=1)
save_submission(f"blend_sffs_mean_k{len(chosen_sffs)}", t_sffs)


n_comp = min(SVD_COMPONENTS, X_oof_short.shape[1]-1)
if n_comp >= 2:
    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    Xo_s = svd.fit_transform(X_oof_short)
    Xt_s = svd.transform(X_test_short)
    o_cv_s, t_cv_s = cv_weighted_blend(Xo_s, Xt_s, y_true, base="ridge", l2=0.05, nonneg=False, n_splits=STACK_FOLDS)
    print("SVD+Ridge AUC:", auc_safe(y_true, o_cv_s))
    save_submission(f"blend_svd{n_comp}_cv_ridge", t_cv_s)
else:
    print("SVD skipped (not enough columns).")


# Choose one strong base to calibrate (e.g., ridge meta above)
base_oof = o_cv_ridge
base_test = t_cv_ridge

# K-fold isotonic: fit on train folds, predict on val fold; then fit on all to transform TEST
skf = StratifiedKFold(n_splits=STACK_FOLDS, shuffle=True, random_state=42)
oof_iso = np.zeros_like(base_oof)
for tr, va in skf.split(base_oof.reshape(-1,1), y_true):
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(base_oof[tr], y_true[tr])
    oof_iso[va] = iso.predict(base_oof[va])

# report calibrated OOF AUC (may not improve AUC; improves calibration)
print("Isotonic OOF AUC:", auc_safe(y_true, oof_iso))

# fit final isotonic on all OOF to transform TEST
iso_full = IsotonicRegression(out_of_bounds="clip").fit(base_oof, y_true)
test_iso = iso_full.predict(base_test)
save_submission(f"{os.path.splitext(os.path.basename(path_ridge))[0]}_iso", test_iso)


# Build full OOF/TEST dfs for consistent column filtering
OOF_df = pd.DataFrame(X_oof, index=y.index, columns=registry_df["model_id"])
TEST_cols, mats = [], []
for mid in registry_df["model_id"]:
    sp = registry_df.loc[registry_df["model_id"]==mid, "sub_path"].values
    if len(sp)==0 or not os.path.exists(sp[0]):
        continue
    try:
        s = read_sub_series(sp[0])
        mats.append(s); TEST_cols.append(mid)
    except Exception:
        pass

TEST_df = pd.concat(mats, axis=1) if mats else None
if TEST_df is None:
    print("Heavier solvers skipped: no TEST_df")
else:
    def top_cols_for_weights(k):
        order = (registry_df.sort_values("cv_auc_mean_meta", ascending=False)["model_id"].tolist()
                 if registry_df["cv_auc_mean_meta"].notna().any()
                 else registry_df.sort_values("oof_auc", ascending=False)["model_id"].tolist())
        cols = [c for c in order if (c in OOF_df.columns and c in TEST_df.columns)]
        return cols[:min(k, len(cols))]

    def stratified_rows(y_series, n_keep):
        n = len(y_series)
        if n_keep >= n:
            return np.arange(n)
        k = max(2, int(np.ceil(n / n_keep)))
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        for _, idx in skf.split(np.zeros(n), y_series.values):
            return idx[:n_keep]

    colsW = top_cols_for_weights(MAX_COLS_FOR_WEIGHTS)
    if colsW:
        idx_rows = stratified_rows(y, ROW_SAMPLE_FOR_WEIGHTS)
        X_sub = OOF_df.loc[y.index[idx_rows], colsW].values.astype(np.float64)
        b_sub = y.values[idx_rows].astype(np.float64)
        T_mat = TEST_df[colsW].values.astype(np.float64)

        # NNLS
        if SCIPY_OK:
            try:
                w_nnls, _ = nnls(X_sub, b_sub)
                s = w_nnls.sum()
                if s > 0:
                    w_nnls = w_nnls / s
                    save_submission("sub_nnls_subsampled", T_mat.dot(w_nnls))
            except Exception:
                pass

            # Constrained LS (sum=1, w>=0)
            try:
                m = X_sub.shape[1]
                w0 = np.ones(m)/m
                bounds = [(0.0, None)]*m
                cons = ({'type': 'eq','fun': lambda w: np.sum(w)-1.0},)
                def obj(w):
                    r = X_sub.dot(w) - b_sub
                    return float(np.dot(r, r))
                res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
                               options={'maxiter':200, 'ftol':1e-9})
                w_c = res.x if (res.success and np.isfinite(res.x).all()) else w0
                w_c = w_c / max(w_c.sum(), 1e-12)
                save_submission("sub_constrained_subsampled", T_mat.dot(w_c))
            except Exception:
                pass

        # Lite stacking on subsample columns/rows
        def stacked_oof_test_lite(OOF, TEST, y, colsW, base="lr", C=1.0, n_splits=3, random_state=42):
            OOF_sel = OOF[colsW]
            TEST_sel = TEST[colsW]
            idx_rows_local = stratified_rows(y, ROW_SAMPLE_FOR_WEIGHTS)
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
            test_meta = np.zeros(TEST_sel.shape[0])
            y_sub = y.iloc[idx_rows_local]
            O_sub = OOF_sel.iloc[idx_rows_local]
            for tr, va in skf.split(O_sub, y_sub):
                Xtr, Xva = O_sub.iloc[tr], O_sub.iloc[va]
                ytr, yva = y_sub.iloc[tr], y_sub.iloc[va]
                if base == "lr":
                    clf = LogisticRegression(max_iter=2000, C=C)
                    clf.fit(Xtr, ytr)
                    test_meta += clf.predict_proba(TEST_sel)[:,1] / n_splits
                elif base == "lgb" and LGB_OK:
                    params = dict(objective="binary", metric="auc",
                                  learning_rate=0.05, num_leaves=31,
                                  n_estimators=800, subsample=0.8, colsample_bytree=0.8,
                                  random_state=42, verbosity=-1)
                    clf = lgb.LGBMClassifier(**params)
                    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)],
                            eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
                    test_meta += clf.predict_proba(TEST_sel)[:,1] / n_splits
                else:
                    raise ValueError("Unknown base or LightGBM not available")
            return test_meta

        try:
            sub_st_lr = stacked_oof_test_lite(OOF_df, TEST_df, y, colsW, base="lr", C=1.0, n_splits=N_SPLITS_STACK_LITE, random_state=42)
            save_submission("sub_stack_lr_subsampled", sub_st_lr)
        except Exception:
            pass

        if LGB_OK:
            try:
                sub_st_lgb = stacked_oof_test_lite(OOF_df, TEST_df, y, colsW, base="lgb", n_splits=N_SPLITS_STACK_LITE, random_state=42)
                save_submission("sub_stack_lgb_subsampled", sub_st_lgb)
            except Exception:
                pass
    else:
        print("Heavier solvers: no eligible columns.")


if fi_files:
    fis = []
    for p in fi_files:
        try:
            df = pd.read_csv(p)
            cols = [c.lower() for c in df.columns]
            if "feature" not in cols:
                df = df.rename(columns={df.columns[0]:"feature", df.columns[1]:"importance"})
            else:
                imp_col = None
                for c in df.columns:
                    if "import" in c.lower():
                        imp_col = c; break
                if imp_col is None: 
                    continue
                df = df[["feature", imp_col]].rename(columns={imp_col:"importance"})
            df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
            df["source_file"] = os.path.basename(p)
            fis.append(df[["feature","importance","source_file"]])
        except Exception:
            pass
    if fis:
        F = pd.concat(fis, ignore_index=True)
        agg = (F.groupby("feature", as_index=False)["importance"]
                 .sum()
                 .sort_values("importance", ascending=False))
        agg.to_csv(os.path.join(OUT_DIR, "feature_importance_aggregate.csv"), index=False)
        display(agg.head(30))
        print("Saved FI aggregate.")
else:
    print("No FI files detected.")


for f in sorted(glob.glob(os.path.join(OUT_DIR, "*.csv"))):
    print(os.path.basename(f))


# y_true = y.values.astype(int)

# def save_submission(tag, preds):
#     sub = sample.copy()
#     sub[TARGET_COL] = preds
#     path = os.path.join(OUT_DIR, f"{tag}.csv")
#     sub.to_csv(path, index=False)
#     return path

# blend_rows = []

# for K in TOPK_LIST:
#     K = min(K, X_oof_short.shape[1])
#     cols = np.arange(K)
#     oK, tK = X_oof_short[:, cols], X_test_short[:, cols]

#     # mean
#     o, t = oK.mean(axis=1), tK.mean(axis=1)
#     a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_mean", t)
#     blend_rows.append(["mean", K, a, p])

#     # median
#     o, t = np.median(oK, axis=1), np.median(tK, axis=1)
#     a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_median", t)
#     blend_rows.append(["median", K, a, p])

#     # rank average
#     o, t = rank_average(oK, tK)
#     a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_rankavg", t)
#     blend_rows.append(["rankavg", K, a, p])

#     # power mean p=2
#     o, t = power_mean(oK, tK, p=2.0)
#     a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_pmean2", t)
#     blend_rows.append(["powermean_p2", K, a, p])

#     # trimmed 10%
#     o, t = trimmed_mean(oK, tK, trim=0.10)
#     a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_trim10", t)
#     blend_rows.append(["trimmed10", K, a, p])

#     # NNLS
#     if SCIPY_OK:
#         w = nnls_weights(oK, y_true)
#         if w is not None and np.isfinite(w).all() and w.sum() > 0:
#             o = (oK * w).sum(axis=1); t = (tK * w).sum(axis=1)
#             a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_nnls", t)
#             blend_rows.append(["nnls", K, a, p])

#     # Logistic meta
#     try:
#         meta = fit_logreg_meta(oK, y_true, C=1.0)
#         o = meta.predict_proba(oK)[:,1]; t = meta.predict_proba(tK)[:,1]
#         a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_logreg", t)
#         blend_rows.append(["logreg_meta", K, a, p])
#     except Exception:
#         pass

#     # LGB meta
#     if LGB_OK:
#         try:
#             meta = fit_lgb_meta(oK, y_true)
#             o = meta.predict(oK); t = meta.predict(tK)
#             a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_lgbmeta", t)
#             blend_rows.append(["lgb_meta", K, a, p])
#         except Exception:
#             pass

# # Greedy forward selection on shortlist
# chosen, gauc = greedy_forward_selection(X_oof_short, y_true, max_k=min(50, X_oof_short.shape[1]))
# if chosen:
#     o = X_oof_short[:, chosen].mean(axis=1)
#     t = X_test_short[:, chosen].mean(axis=1)
#     a = auc_safe(y_true, o); p = save_submission(f"blend_greedy_mean_k{len(chosen)}", t)
#     blend_rows.append(["greedy_mean", len(chosen), a, p])

# blend_df = pd.DataFrame(blend_rows, columns=["blend","K","oof_auc","submission_path"]).sort_values("oof_auc", ascending=False)
# display(blend_df.head(20))
# blend_df.to_csv(os.path.join(OUT_DIR, "blend_summary.csv"), index=False)
# print("Saved:", os.path.join(OUT_DIR, "blend_summary.csv"))


# topN = 6
# best_rows = blend_df.head(topN).copy()
# calib_rows = []

# def reconstruct_oof_from_tag(tag):
#     # tag examples: blend_top10_mean, blend_top20_rankavg, blend_greedy_mean_kXX
#     if "top" in tag:
#         # parse K
#         m = re.search(r"top(\d+)", tag)
#         if not m: return None, None
#         K = int(m.group(1))
#         cols = np.arange(min(K, X_oof_short.shape[1]))
#         if "mean" in tag:
#             return X_oof_short[:, cols].mean(axis=1), X_test_short[:, cols].mean(axis=1)
#         if "median" in tag:
#             return np.median(X_oof_short[:, cols], axis=1), np.median(X_test_short[:, cols], axis=1)
#         if "rankavg" in tag:
#             return rank_average(X_oof_short[:, cols], X_test_short[:, cols])
#         if "pmean2" in tag:
#             return power_mean(X_oof_short[:, cols], X_test_short[:, cols], p=2.0)
#         if "trim10" in tag:
#             return trimmed_mean(X_oof_short[:, cols], X_test_short[:, cols], trim=0.10)
#     if "greedy_mean" in tag:
#         return X_oof_short[:, chosen].mean(axis=1), X_test_short[:, chosen].mean(axis=1)
#     return None, None

# for _, r in best_rows.iterrows():
#     tag = os.path.splitext(os.path.basename(r["submission_path"]))[0]
#     oof_pred, test_pred = reconstruct_oof_from_tag(tag)
#     if oof_pred is None: continue
#     try:
#         iso = IsotonicRegression(out_of_bounds="clip")
#         iso.fit(oof_pred, y_true)
#         oof_cal = iso.predict(oof_pred)
#         test_cal= iso.predict(test_pred)
#         a = auc_safe(y_true, oof_cal)
#         p = save_submission(f"{tag}_iso", test_cal)
#         calib_rows.append([f"{tag}_iso", a, p])
#     except Exception:
#         pass

# calib_df = pd.DataFrame(calib_rows, columns=["blend_calibrated","oof_auc","submission_path"]).sort_values("oof_auc", ascending=False)
# display(calib_df.head(10))
# calib_df.to_csv(os.path.join(OUT_DIR, "blend_calibrated_summary.csv"), index=False)
# print("Saved:", os.path.join(OUT_DIR, "blend_calibrated_summary.csv"))


# def corr_penalized_weights(oof_mat, y_true, lam=0.35):
#     m = oof_mat.shape[1]
#     if m < 2:
#         return np.array([1.0])
#     perf = np.array([auc_safe(y_true, oof_mat[:, j]) for j in range(m)])
#     C = np.corrcoef(oof_mat.T)
#     if np.ndim(C) < 2:
#         avgc = np.zeros(m)
#     else:
#         with np.errstate(invalid="ignore"):
#             avgc = (np.nansum(C, axis=1) - 1.0) / np.maximum(1, m-1)
#     raw = np.clip(perf, 0, 1) / (1.0 + lam * np.clip(avgc, 0, 1))
#     raw[~np.isfinite(raw)] = 0.0
#     return raw / max(raw.sum(), 1e-12)

# pen_rows = []
# for K in TOPK_LIST:
#     K = min(K, X_oof_short.shape[1])
#     cols = np.arange(K)
#     if K < 1:
#         continue
#     if K == 1:
#         o = X_oof_short[:, cols].ravel()
#         t = X_test_short[:, cols].ravel()
#         a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_corrpen", t)
#         pen_rows.append(["corr_penalized", K, a, p, np.array([1.0])])
#     else:
#         w = corr_penalized_weights(X_oof_short[:, cols], y_true, lam=0.35)
#         o = (X_oof_short[:, cols] * w).sum(axis=1)
#         t = (X_test_short[:, cols] * w).sum(axis=1)
#         a = auc_safe(y_true, o); p = save_submission(f"blend_top{K}_corrpen", t)
#         pen_rows.append(["corr_penalized", K, a, p, w])

# pen_df = pd.DataFrame(pen_rows, columns=["blend","K","oof_auc","submission_path","weights"]).sort_values("oof_auc", ascending=False)
# display(pen_df.head(10))
# pen_df.drop(columns=["weights"]).to_csv(os.path.join(OUT_DIR, "blend_corrpen_summary.csv"), index=False)


# # 12. Full Matrix Ensembles  (SAFE + SCALABLE)

# # --- knobs to keep heavy steps fast ---
# MAX_COLS_FOR_WEIGHTS = 200      # cap number of base models in weight solvers
# ROW_SAMPLE_FOR_WEIGHTS = 200_000  # stratified sample size for NNLS/SLSQP/stacking
# N_SPLITS_STACK = 3              # lighter CV for stacking

# # Build TEST dataframe for all surviving models (may still be large)
# def build_test_df(ids):
#     mats = []
#     cols = []
#     for mid in ids:
#         sp = registry_df.loc[registry_df["model_id"] == mid, "sub_path"].values
#         if len(sp) == 0 or not os.path.exists(sp[0]): 
#             continue
#         try:
#             s = read_sub_series(sp[0]).reindex(test_ids)
#             mats.append(s); cols.append(mid)
#         except Exception:
#             pass
#     if not mats: 
#         return None
#     T = pd.concat(mats, axis=1)
#     T.columns = cols
#     return T

# # OOF / TEST dataframes (full)
# OOF_df = pd.DataFrame(X_oof, index=y.index, columns=model_ids)
# TEST_df = build_test_df(model_ids)

# def auc_series(s): 
#     return roc_auc_score(y.values, s.values)

# def zscore_df(df):
#     mu = df.mean(0)
#     sd = df.std(0).replace(0, 1.0)
#     return (df - mu)/sd

# def pct_rank_df(df): 
#     return df.rank(pct=True, axis=0)

# # helper: pick top columns by meta score (or oof_auc) and align OOF/TEST
# def top_cols_for_weights(k):
#     order = (registry_df.sort_values("cv_auc_mean_meta", ascending=False)["model_id"].tolist()
#              if registry_df["cv_auc_mean_meta"].notna().any()
#              else registry_df.sort_values("oof_auc", ascending=False)["model_id"].tolist())
#     cols = [c for c in order if (c in OOF_df.columns and (TEST_df is None or c in TEST_df.columns))]
#     return cols[:min(k, len(cols))]

# # helper: stratified row indices
# def stratified_rows(n_target, n_keep):
#     # n_target is y (Series)
#     n = len(n_target)
#     if n_keep >= n:
#         return np.arange(n)
#     skf = StratifiedKFold(n_splits=max(2, int(np.ceil(n / n_keep))), shuffle=True, random_state=42)
#     # take the first split's validation fold as a roughly stratified sample
#     for _, idx in skf.split(np.zeros(n), n_target.values):
#         return idx[:n_keep]  # guard in case fold is slightly larger

# if TEST_df is not None:
#     # Quick blends (cheap, keep them full)
#     order = (registry_df.sort_values("cv_auc_mean_meta", ascending=False)["model_id"].tolist()
#              if registry_df["cv_auc_mean_meta"].notna().any()
#              else registry_df["model_id"].tolist())
#     for K in TOPK_LIST:
#         K = min(K, len(order))
#         colsK = [c for c in order[:K] if c in TEST_df.columns]
#         if not colsK:
#             continue
#         o_mean = OOF_df[colsK].mean(1); t_mean = TEST_df[colsK].mean(1)
#         save_submission(f"sub_quick_mean_top{K}", t_mean.values)

#         o_rank = pct_rank_df(OOF_df[colsK]).mean(1); t_rank = pct_rank_df(TEST_df[colsK]).mean(1)
#         save_submission(f"sub_quick_rank_top{K}", t_rank.values)

#         o_z = zscore_df(OOF_df[colsK]).mean(1); t_z = zscore_df(TEST_df[colsK]).mean(1)
#         save_submission(f"sub_quick_zmean_top{K}", t_z.values)

#     # Greedy mean selection on FULL (still cheap incremental means)
#     def greedy_mean_select(OOF, y, limit=200):
#         chosen, current_auc, current_pred = [], -1.0, None
#         remaining = list(OOF.columns)
#         for _ in range(min(limit, len(remaining))):
#             best_gain, best_col, best_pred = 0.0, None, None
#             for c in remaining:
#                 cand = OOF[chosen + [c]].mean(1) if chosen else OOF[c]
#                 val = auc_series(cand)
#                 gain = val - (current_auc if current_pred is not None else 0.0)
#                 if gain > best_gain:
#                     best_gain, best_col, best_pred = gain, c, cand
#             if best_col is None or best_gain <= 0: 
#                 break
#             chosen.append(best_col); remaining.remove(best_col)
#             current_pred = best_pred; current_auc = auc_series(current_pred)
#         return chosen, current_pred, current_auc

#     chosen_full, oof_g_full, auc_g_full = greedy_mean_select(OOF_df, y, limit=200)
#     if chosen_full:
#         save_submission(f"sub_greedy_full_k{len(chosen_full)}", TEST_df[chosen_full].mean(1).values)

#     # ======= HEAVY SOLVERS (safe mode: cap columns + subsample rows) =======
#     colsW = top_cols_for_weights(MAX_COLS_FOR_WEIGHTS)
#     if colsW:
#         idx_rows = stratified_rows(y, ROW_SAMPLE_FOR_WEIGHTS)
#         X_sub = OOF_df.loc[y.index[idx_rows], colsW].values.astype(np.float64)
#         b_sub = y.values[idx_rows].astype(np.float64)
#         T_mat = TEST_df[colsW].values.astype(np.float64)

#         if SCIPY_OK:
#             # NNLS on subsample/columns
#             try:
#                 w_nnls, _ = nnls(X_sub, b_sub)
#                 s = w_nnls.sum()
#                 if s > 0:
#                     w_nnls = w_nnls / s
#                     save_submission("sub_nnls_subsampled", T_mat.dot(w_nnls))
#             except Exception:
#                 pass

#             # Constrained LS (SLSQP) on subsample/columns
#             try:
#                 m = X_sub.shape[1]
#                 w0 = np.ones(m)/m
#                 bounds = [(0.0, None)]*m
#                 cons = ({'type': 'eq','fun': lambda w: np.sum(w)-1.0},)
#                 def obj(w):
#                     r = X_sub.dot(w) - b_sub
#                     return float(np.dot(r, r))
#                 res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons,
#                                options={'maxiter':200, 'ftol':1e-9})
#                 w_c = res.x if (res.success and np.isfinite(res.x).all()) else w0
#                 w_c = w_c / max(w_c.sum(), 1e-12)
#                 save_submission("sub_constrained_subsampled", T_mat.dot(w_c))
#             except Exception:
#                 pass

#         # Stacking (LR / LGB) on subsample/columns, lighter CV
#         def stacked_oof_test_lite(OOF, TEST, y, base="lr", C=1.0, n_splits=3, random_state=42):
#             # only use selected columns and subsampled rows for fitting, predict on full TEST
#             OOF_sel = OOF[colsW]
#             TEST_sel = TEST[colsW]
#             idx_rows_local = stratified_rows(y, ROW_SAMPLE_FOR_WEIGHTS)
#             skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
#             oof_meta = np.zeros(len(idx_rows_local))
#             test_meta = np.zeros(TEST_sel.shape[0])
#             y_sub = y.iloc[idx_rows_local]
#             O_sub = OOF_sel.iloc[idx_rows_local]
#             for tr, va in skf.split(O_sub, y_sub):
#                 Xtr, Xva = O_sub.iloc[tr], O_sub.iloc[va]
#                 ytr, yva = y_sub.iloc[tr], y_sub.iloc[va]
#                 if base == "lr":
#                     clf = LogisticRegression(max_iter=2000, C=C)
#                     clf.fit(Xtr, ytr)
#                     # oof_meta only used for internal AUC check if desired
#                     oof_meta[va] = clf.predict_proba(Xva)[:,1]
#                     test_meta += clf.predict_proba(TEST_sel)[:,1] / n_splits
#                 elif base == "lgb" and LGB_OK:
#                     params = dict(objective="binary", metric="auc",
#                                   learning_rate=0.05, num_leaves=31,
#                                   n_estimators=800, subsample=0.8, colsample_bytree=0.8,
#                                   random_state=42, verbosity=-1)
#                     clf = lgb.LGBMClassifier(**params)
#                     clf.fit(Xtr, ytr, eval_set=[(Xva, yva)],
#                             eval_metric="auc", callbacks=[lgb.early_stopping(50, verbose=False)])
#                     oof_meta[va] = clf.predict_proba(Xva)[:,1]
#                     test_meta += clf.predict_proba(TEST_sel)[:,1] / n_splits
#                 else:
#                     raise ValueError("Unknown base or LightGBM not available")
#             return test_meta

#         try:
#             sub_st_lr = stacked_oof_test_lite(OOF_df, TEST_df, y, base="lr", C=1.0, n_splits=N_SPLITS_STACK, random_state=42)
#             save_submission("sub_stack_lr_subsampled", sub_st_lr)
#         except Exception:
#             pass

#         if LGB_OK:
#             try:
#                 sub_st_lgb = stacked_oof_test_lite(OOF_df, TEST_df, y, base="lgb", n_splits=N_SPLITS_STACK, random_state=42)
#                 save_submission("sub_stack_lgb_subsampled", sub_st_lgb)
#             except Exception:
#                 pass

# else:
#     print("TEST_df build skipped (not all models had SUB files).")


# # We can only do family blends for models present in TEST_df
# if ('TEST_df' in locals()) and (TEST_df is not None) and META_OK and (len(meta_raw) > 0):
#     # build a meta table keyed by model_id if possible — fuzzy again via tokens
#     meta_map = {}
#     meta_text_cols = [c for c in meta_raw.columns if meta_raw[c].dtype == object]
#     bags = []
#     for i, r in meta_raw.iterrows():
#         parts = []
#         for c in meta_text_cols:
#             v = r[c]
#             if pd.isna(v): continue
#             parts.append(str(v))
#         bags.append((" ".join(parts).lower(), i))

#     for mid in TEST_df.columns:
#         tokens = set(re.split(r"[_\-\.\s]+", mid.lower()))
#         best = (-1, None)
#         for bag, idx in bags:
#             overlap = len(tokens & set(bag.split()))
#             if overlap > best[0]:
#                 best = (overlap, idx)
#         if best[1] is not None and best[0] >= 1:
#             meta_map[mid] = meta_raw.loc[best[1]]

#     meta_tab = pd.DataFrame(meta_map).T  # rows = model_id
#     def _family_blends(col):
#         if col not in meta_tab.columns:
#             print(f"'{col}' not present; skip.")
#             return []
#         rows = []
#         for val, mids in meta_tab.groupby(col).groups.items():
#             mids = list(mids)
#             mids = [m for m in mids if m in TEST_df.columns and m in OOF_df.columns]
#             if len(mids) < 2: continue
#             o = OOF_df[mids].mean(1); t = TEST_df[mids].mean(1)
#             a = roc_auc_score(y.values, o.values)
#             fn = f"sub_family_{col}={val}_mean.csv"
#             pd.DataFrame({ID_COL: test_ids, TARGET_COL: t.values}).to_csv(os.path.join(OUT_DIR, fn), index=False)
#             rows.append((col, val, len(mids), a, fn))
#         return rows

#     fam_rows = []
#     for col in ["model", "preprocessor", "feature_set", "combo_name"]:
#         if col in meta_raw.columns:
#             fam_rows += _family_blends(col)

#     if fam_rows:
#         fam_df = pd.DataFrame(fam_rows, columns=["group","value","count","oof_auc","file"]).sort_values("oof_auc", ascending=False)
#         display(fam_df.head(20))
#         fam_df.to_csv(os.path.join(OUT_DIR, "family_blends_summary.csv"), index=False)

# else:
#     print("Family blend skipped")


# if fi_files:
#     fis = []
#     for p in fi_files:
#         try:
#             df = pd.read_csv(p)
#             cols = [c.lower() for c in df.columns]
#             if "feature" not in cols:
#                 df = df.rename(columns={df.columns[0]:"feature", df.columns[1]:"importance"})
#             else:
#                 imp_col = None
#                 for c in df.columns:
#                     if "import" in c.lower():
#                         imp_col = c; break
#                 if imp_col is None: continue
#                 df = df[["feature", imp_col]].rename(columns={imp_col:"importance"})
#             df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
#             df["source_file"] = os.path.basename(p)
#             fis.append(df[["feature","importance","source_file"]])
#         except Exception:
#             pass
#     if fis:
#         F = pd.concat(fis, ignore_index=True)
#         agg = (F.groupby("feature", as_index=False)["importance"]
#                  .sum()
#                  .sort_values("importance", ascending=False))
#         agg.to_csv(os.path.join(OUT_DIR, "feature_importance_aggregate.csv"), index=False)
#         display(agg.head(30))
#         print("Saved FI aggregate.")
# else:
#     print("No FI files detected.")


# if fi_files:
#     fis = []
#     for p in fi_files:
#         try:
#             df = pd.read_csv(p)
#             cols = [c.lower() for c in df.columns]
#             if "feature" not in cols:
#                 df = df.rename(columns={df.columns[0]:"feature", df.columns[1]:"importance"})
#             else:
#                 imp_col = None
#                 for c in df.columns:
#                     if "import" in c.lower():
#                         imp_col = c; break
#                 if imp_col is None: continue
#                 df = df[["feature", imp_col]].rename(columns={imp_col:"importance"})
#             df["importance"] = pd.to_numeric(df["importance"], errors="coerce").fillna(0.0)
#             df["source_file"] = os.path.basename(p)
#             fis.append(df[["feature","importance","source_file"]])
#         except Exception:
#             pass
#     if fis:
#         F = pd.concat(fis, ignore_index=True)
#         agg = (F.groupby("feature", as_index=False)["importance"]
#                  .sum()
#                  .sort_values("importance", ascending=False))
#         agg.to_csv(os.path.join(OUT_DIR, "feature_importance_aggregate.csv"), index=False)
#         display(agg.head(30))
#         print("Saved FI aggregate.")
# else:
#     print("No FI files detected.")




