# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# %% [markdown]
# # NeurIPS – Open Polymer Prediction 2025
# ## Leakage-Safe TF-IDF(SMILES) + Stacking + Residual Calibration (Per-Target, NaN-safe)
#
# **Goal.** Predict 5 polymer properties (`Tg`, `FFV`, `Tc`, `Density`, `Rg`) for the competition’s **wMAE** metric.
#
# **Headline.** Avoids “look-alike” leakage, builds chemistry-aware text features from SMILES **without RDKit**,
# and combines multiple models with **OOF stacking, residual correction, and calibration**. Now **per-target** to be NaN-safe.
#
# **Highlights**
# - **SMILES → TF-IDF → SVD**: high-signal, RDKit-free features (char n-grams 2–6 + 400 SVD dims).
# - **Canonical SMILES features**: rings, branch depth, aromaticity, repeat hints.
# - **Leakage-safe CV**: clusters in SVD space → **GroupKFold**.
# - **OOF KNN label features**: distance-weighted neighbors in SVD space (OOF on train; full on test), computed **per target**.
# - **Base models**: LGBM / XGBoost / CatBoost / RandomForest (MAE-aligned where possible).
# - **Stacking**: OOF predictions → RidgeCV per target.
# - **Residual calibration**: KNN residual correction + isotonic regression + quantile clipping.
# - **wMAE-aware choices**: per-target post-processing tuned with train-based weights.
#
# > No internet. No external installs. Safe for Kaggle’s hidden reruns.

# %% [markdown]
# ## Pipeline at a glance
#
# SMILES ──► TF-IDF(char 2–6) ──► SVD(400) ─┐
#                                           ├─► [Feature set]
# Engineered features (train_features.csv) ─┘
#                     + Canonical SMILES signals (rings/branches/aromatics/atoms)
#
# Per target t:
#   + OOF KNN label feature (in SVD space, distance-weighted)
#   + GroupKFold (scaffold-aware)      ──► Base models ──► OOF preds
#   + RidgeCV stack (OOF-only)         ──► blended OOF/test
#   + KNN residual correction + isotonic + quantile clipping (wMAE-aware knob pick)

# %% [markdown]
# ## Metric (wMAE) — what matters for choices
# We don’t know the hidden test weights, so we compute **train-based weights** (robust inverse range × inverse √frequency) and
# use them **only** to choose post-processing knobs (residual strength, neighbor count, clipping). Predictions remain per-target.

# =========================
# CONFIG (tune speed vs score)
# =========================
SEEDS            = [42, 2025, 123]  # add/remove for speed vs stability
NFOLDS           = 5
N_CLUSTERS       = 10
LGB_N_EST        = 3500
XGB_N_EST        = 2000
CAT_ITERS        = 2000
RF_N_EST         = 1000
EARLY_STOP_ROUNDS= 150
NGRAM_RANGE      = (2, 6)            # char n-grams
TFIDF_MAX_FEAT   = 60000
N_SVD            = 400
RANDOM_STATE     = 42
LOG_SKEW_TH      = 0.3               # flexible log transform
CLIP_CANDS       = [(1,99),(2,98),(5,95)]
RES_ALPHAS       = [0.0,0.25,0.5,0.75,1.0]
RES_K_CANDS      = [20,30,40]

# =========================
# Imports (no internet)
# =========================
import os, random, gc, re
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error

def set_seeds(seed=42):
    os.environ["PYTHONHASHSEED"]=str(seed); random.seed(seed); np.random.seed(seed)
set_seeds(RANDOM_STATE)

# =========================
# 1) Load Data
# =========================
train_feat = pd.read_csv("/kaggle/input/pre-trained/train_features.csv")
test_feat  = pd.read_csv("/kaggle/input/pre-trained/test_features.csv")
train_df   = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/train.csv")
test_df    = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/test.csv")
sample_sub = pd.read_csv("/kaggle/input/neurips-open-polymer-prediction-2025/sample_submission.csv")

TARGETS = ["Tg","FFV","Tc","Density","Rg"]

# =========================
# 2) SMILES → TF-IDF + SVD
# =========================
def build_tfidf_svd(train_smiles, test_smiles, ngram_range=NGRAM_RANGE, max_features=TFIDF_MAX_FEAT, n_svd=N_SVD):
    """Convert SMILES to char n-grams and compress with TruncatedSVD."""
    tr_sm = train_smiles.fillna("")
    te_sm = test_smiles.fillna("")
    all_sm = pd.concat([tr_sm, te_sm], axis=0)

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range, max_features=max_features)
    X_all = vec.fit_transform(all_sm.values)

    max_svd = max(2, min(n_svd, X_all.shape[1]-1, X_all.shape[0]-1))
    svd = TruncatedSVD(n_components=max_svd, random_state=RANDOM_STATE)
    X_svd = svd.fit_transform(X_all)

    cols = [f"sm_svd_{i:03d}" for i in range(max_svd)]
    X_svd_df = pd.DataFrame(X_svd, columns=cols, index=all_sm.index)
    tf_tr = X_svd_df.iloc[:len(tr_sm)].reset_index(drop=True)
    tf_te = X_svd_df.iloc[len(tr_sm):].reset_index(drop=True)
    return tf_tr, tf_te

tf_tr, tf_te = build_tfidf_svd(train_df["SMILES"], test_df["SMILES"])
print(f"✅ SVD dims: {tf_tr.shape[1]}")

# =========================
# 3) Canonical SMILES features (cheap chemistry priors)
# =========================
def canonical_smiles_features(smiles: pd.Series) -> pd.DataFrame:
    s = smiles.fillna("")
    feats = pd.DataFrame(index=s.index)
    feats['smiles_len']   = s.str.len()
    feats['ring_count']   = s.str.count(r'\d')
    feats['aromatic_cnt'] = s.str.count(r'[a-z]')
    feats['atom_cnt']     = s.str.count(r'[A-Za-z]')
    feats['repeat_ind']   = s.str.count(r'\[\w+\]')
    def branch_depth(x):
        md,d=0,0
        for ch in x:
            if ch=='(': d+=1; md=max(md,d)
            elif ch==')': d-=1
        return md
    feats['max_branch_depth'] = s.apply(branch_depth)
    def branch_var(x):
        pos=[m.start() for m in re.finditer(r'\(', x)]
        return np.var(pos) if pos else 0
    feats['branch_var'] = s.apply(branch_var)
    return feats.replace([np.inf,-np.inf],0).fillna(0)

sm_tr = canonical_smiles_features(train_df["SMILES"])
sm_te = canonical_smiles_features(test_df["SMILES"])

# =========================
# 4) Merge Features + Cleaning (base — target-agnostic)
# =========================
train_all = train_df[["id"]].merge(train_feat, on="id", how="left")
test_all  = test_df[["id"]].merge(test_feat,  on="id", how="left")
train_all = pd.concat([train_all.reset_index(drop=True), tf_tr, sm_tr.reset_index(drop=True)], axis=1)
test_all  = pd.concat([test_all.reset_index(drop=True),  tf_te, sm_te.reset_index(drop=True)], axis=1)

EXCL = set(["id"]+TARGETS)
feat_cols = [c for c in train_all.columns if c not in EXCL and c in test_all.columns]
X_base   = train_all[feat_cols].replace([np.inf,-np.inf],0).fillna(0).reset_index(drop=True)
X_test   = test_all[feat_cols].replace([np.inf,-np.inf],0).fillna(0).reset_index(drop=True)
y        = train_df[TARGETS].copy()

print(f"✅ Base features: {len(feat_cols)} | Train {X_base.shape} | Test {X_test.shape}")
assert len(X_test)==len(test_df)

# =========================
# 5) Train-based weights (proxy for leaderboard wMAE)
# =========================
def approx_weight(ycol: pd.Series):
    n_i = ycol.notna().sum()
    r_i = float(ycol.quantile(0.99)-ycol.quantile(0.01))
    r_i = max(r_i,1e-6)
    return (1.0/r_i) * (1.0/np.sqrt(max(n_i,1)))

raw_w = np.array([approx_weight(y[t]) for t in TARGETS])
w_train = len(TARGETS)*(raw_w/raw_w.sum())
W_BY_TGT = {t: float(w) for t,w in zip(TARGETS, w_train)}
print("≈ train-based weights:", W_BY_TGT)

# =========================
# 6) Scaffold-aware groups (clusters in SVD space)
# =========================
def make_groups(S_train_df, n_clusters=N_CLUSTERS, seed=RANDOM_STATE):
    try:
        km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        return km.fit_predict(S_train_df.values)
    except Exception:
        try:
            try:
                ac = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", affinity="cosine")
            except TypeError:
                ac = AgglomerativeClustering(n_clusters=n_clusters, linkage="average", metric="cosine")
            return ac.fit_predict(S_train_df.values)
        except Exception:
            rng = np.random.RandomState(seed)
            return rng.randint(0, n_clusters, size=len(S_train_df))

groups_all = make_groups(tf_tr, n_clusters=N_CLUSTERS, seed=RANDOM_STATE)

# =========================
# 7) Per-target KNN label feature (OOF, leak-safe)
# =========================
def knn_label_feature_per_target(S_tr_valid, S_te, y_valid, groups_valid, n_neighbors=30):
    """
    Build a single KNN label feature for ONE target:
    - Only uses rows with target present (no NaNs).
    - OOF within GroupKFold: neighbors fit on train folds, averaged (distance-weighted).
    - Test uses neighbors fit on ALL valid rows.
    Returns: oof_feat (len valid), te_feat (len test)
    """
    S_tr = S_tr_valid.values
    S_te = S_te.values

    gkf = GroupKFold(n_splits=NFOLDS)
    oof = np.zeros(len(S_tr))
    for tr_idx, va_idx in gkf.split(S_tr, y_valid.values, groups=groups_valid):
        nn = NearestNeighbors(n_neighbors=min(n_neighbors, len(tr_idx)), metric="cosine")
        nn.fit(S_tr[tr_idx])
        dist, ind = nn.kneighbors(S_tr[va_idx], return_distance=True)
        y_tr = y_valid.values[tr_idx]  # no NaNs by construction
        for i in range(len(va_idx)):
            neigh = ind[i]; d = dist[i]
            w = 1.0/(d+1e-6)
            oof[i] = np.average(y_tr[neigh], weights=w) if len(neigh)>0 else np.mean(y_tr)
        oof[va_idx] = oof[:len(va_idx)] if len(va_idx)==len(oof) else oof[va_idx]  # safe assign

    # test on full valid set
    nn_full = NearestNeighbors(n_neighbors=min(n_neighbors, len(S_tr)), metric="cosine")
    nn_full.fit(S_tr)
    dist_te, ind_te = nn_full.kneighbors(S_te, return_distance=True)
    te = np.zeros(len(S_te))
    for i in range(len(S_te)):
        neigh = ind_te[i]; d = dist_te[i]
        w = 1.0/(d+1e-6)
        te[i] = np.average(y_valid.values[neigh], weights=w) if len(neigh)>0 else np.mean(y_valid.values)

    return oof, te

# =========================
# 8) Base model factories (per-target; MAE-aligned when possible)
# =========================
def make_lgb(seed):
    return lgb.LGBMRegressor(
        objective="regression_l1", metric="mae",
        boosting_type="gbdt",
        n_estimators=LGB_N_EST, learning_rate=0.01,
        num_leaves=128, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=5,
        min_data_in_leaf=20, random_state=seed,
    )

def make_xgb(seed):
    # 'reg:absoluteerror' may not exist on some images; fallback handled below.
    return xgb.XGBRegressor(
        objective="reg:absoluteerror",
        n_estimators=XGB_N_EST, learning_rate=0.02,
        max_depth=6, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0, random_state=seed, tree_method="hist",
    )

def make_cat(seed):
    return CatBoostRegressor(
        iterations=CAT_ITERS, depth=6, learning_rate=0.03,
        l2_leaf_reg=3.0, loss_function="MAE", bagging_temperature=0.2,
        random_seed=seed, verbose=False
    )

def make_rf(seed):
    return RandomForestRegressor(
        n_estimators=RF_N_EST, max_depth=10, random_state=seed, n_jobs=-1
    )

def maybe_log1p(y_series):
    ys = y_series.dropna()
    if len(ys)==0: return (lambda x:x, lambda x:x)
    return ((np.log1p,np.expm1) if (ys.min()>0 and ys.skew()>LOG_SKEW_TH) else (lambda x:x, lambda x:x))

# =========================
# 9) Train per target: OOF, Ridge stack, tuned residuals, calibration, clipping
# =========================
final_test = {}
cv_report  = {}

for tgt in TARGETS:
    print(f"\n=== {tgt} ===")
    mask = ~y[tgt].isna()
    y_tgt  = y.loc[mask, tgt].astype(float).reset_index(drop=True)
    X_t    = X_base.loc[mask].reset_index(drop=True)
    X_te   = X_test.copy().reset_index(drop=True)
    S_tr_v = tf_tr.loc[mask].reset_index(drop=True)
    groups = groups_all[mask.values]

    # per-target KNN label feature (OOF for train, full for test) — leak-safe
    knn_oof, knn_te = knn_label_feature_per_target(S_tr_v, tf_te, y_tgt, groups, n_neighbors=30)
    X_t = pd.concat([X_t, pd.DataFrame({"knn_label": knn_oof})], axis=1)
    X_te_aug = pd.concat([X_te, pd.DataFrame({"knn_label": knn_te})], axis=1)

    fwd, inv = maybe_log1p(y_tgt)
    y_tr = fwd(y_tgt)

    # Fixed GroupKFold splits (reuse for all models/seeds)
    gkf = GroupKFold(n_splits=NFOLDS)
    folds = list(gkf.split(X_t, y_tr.values, groups=groups))

    # Collect OOF/test preds for each base model
    base_names = ["lgb","xgb","cat","rf"]
    oof_base = {m: np.zeros(len(X_t)) for m in base_names}
    te_base  = {m: np.zeros(len(X_te_aug)) for m in base_names}

    for seed in SEEDS:
        for name, factory in [("lgb",make_lgb),("xgb",make_xgb),("cat",make_cat),("rf",make_rf)]:
            oof = np.zeros(len(X_t))
            te_folds = []

            for tr_idx, va_idx in folds:
                X_tr, X_va = X_t.iloc[tr_idx], X_t.iloc[va_idx]
                y_tr_f, y_va_f = y_tr.iloc[tr_idx], y_tr.iloc[va_idx]
                model = factory(seed)

                # safe XGB fallback if needed
                try:
                    if name=="lgb":
                        model.fit(
                            X_tr, y_tr_f,
                            eval_set=[(X_va, y_va_f)],
                            eval_metric="mae",
                            callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOP_ROUNDS),
                                       lgb.log_evaluation(period=0)]
                        )
                        bi = getattr(model,"best_iteration_",None)
                        pv = model.predict(X_va, num_iteration=bi)
                        pt = model.predict(X_te_aug, num_iteration=bi)
                    elif name=="xgb":
                        model.fit(X_tr, y_tr_f, eval_set=[(X_va, y_va_f)],
                                  verbose=False, early_stopping_rounds=EARLY_STOP_ROUNDS)
                        pv = model.predict(X_va); pt = model.predict(X_te_aug)
                    elif name=="cat":
                        model.fit(X_tr, y_tr_f, eval_set=(X_va, y_va_f), verbose=False)
                        pv = model.predict(X_va); pt = model.predict(X_te_aug)
                    else: # rf
                        model.fit(X_tr, y_tr_f)
                        pv = model.predict(X_va); pt = model.predict(X_te_aug)
                except Exception as e:
                    if name=="xgb":
                        model = xgb.XGBRegressor(
                            objective="reg:squarederror",
                            n_estimators=XGB_N_EST, learning_rate=0.02,
                            max_depth=6, subsample=0.8, colsample_bytree=0.8,
                            reg_alpha=0.1, reg_lambda=1.0, random_state=seed, tree_method="hist",
                        )
                        model.fit(X_tr, y_tr_f, eval_set=[(X_va, y_va_f)], verbose=False,
                                  early_stopping_rounds=EARLY_STOP_ROUNDS)
                        pv = model.predict(X_va); pt = model.predict(X_te_aug)
                    else:
                        raise

                oof[va_idx] = pv
                te_folds.append(pt)

            # back to natural scale
            oof_base[name] += inv(oof) / len(SEEDS)
            te_base[name]  += inv(np.mean(te_folds, axis=0)) / len(SEEDS)

            del oof, te_folds, model; gc.collect()

    # Ridge stack
    P_tr = np.vstack([oof_base[m] for m in base_names]).T
    P_te = np.vstack([te_base[m]  for m in base_names]).T

    meta = RidgeCV(alphas=[0.1,1.0,10.0])
    meta.fit(P_tr, y_tgt.values)
    oof_blend = meta.predict(P_tr)
    te_blend  = meta.predict(P_te)
    print(f"{tgt} stack coefs:", meta.coef_.round(3))
    print(f"OOF MAE (stacked) {tgt}: {mean_absolute_error(y_tgt.values, oof_blend):.5f}")

    # Residual correction in SVD space (distance-weighted), isotonic, wMAE-aware clipping
    S_te = tf_te.copy().reset_index(drop=True)
    # Precompute neighbor indices once
    Kmax = max(RES_K_CANDS) + 1
    nn_tr = NearestNeighbors(n_neighbors=min(Kmax, len(S_tr_v)), metric="cosine")
    nn_tr.fit(S_tr_v.values)
    dist_tr, ind_tr = nn_tr.kneighbors(S_tr_v.values, return_distance=True)

    nn_te = NearestNeighbors(n_neighbors=min(max(RES_K_CANDS), len(S_tr_v)), metric="cosine")
    nn_te.fit(S_tr_v.values)
    dist_te, ind_te = nn_te.kneighbors(S_te.values, return_distance=True)

    base_resid = (y_tgt.values - oof_blend).astype(float)
    wi = W_BY_TGT[tgt]

    best_score, best_pred, best_combo = 1e18, None, None
    for k in RES_K_CANDS:
        # OOF local residuals
        oof_res = np.zeros(len(S_tr_v))
        for i in range(len(S_tr_v)):
            neigh = ind_tr[i]; dst = dist_tr[i]
            if len(neigh)>0 and neigh[0]==i:
                neigh, dst = neigh[1:], dst[1:]
            neigh, dst = neigh[:k], dst[:k]
            w = 1.0/(dst+1e-6)
            oof_res[i] = np.average(base_resid[neigh], weights=w) if len(neigh)>0 else 0.0

        # TEST local residuals
        te_res = np.zeros(len(S_te))
        for i in range(len(S_te)):
            neigh = ind_te[i][:k]; dst = dist_te[i][:k]
            w = 1.0/(dst+1e-6)
            te_res[i] = np.average(base_resid[neigh], weights=w) if len(neigh)>0 else 0.0

        for alpha in RES_ALPHAS:
            oof_rc = oof_blend + alpha * oof_res
            te_rc  = te_blend  + alpha * te_res

            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(oof_rc, y_tgt.values)
            te_cal = iso.predict(te_rc)

            for (qlo,qhi) in CLIP_CANDS:
                lo, hi = np.percentile(y_tgt.dropna(), [qlo,qhi])
                oof_clip = np.clip(iso.predict(oof_rc), lo, hi)
                score = wi * mean_absolute_error(y_tgt.values, oof_clip)
                if score < best_score:
                    best_score = score
                    best_pred  = np.clip(te_cal, lo, hi)
                    best_combo = (k, alpha, (qlo,qhi))

    print(f"{tgt} best (wMAE-proxy): k={best_combo[0]}, alpha={best_combo[1]}, clip={best_combo[2]}")
    final_test[tgt] = best_pred

# =========================
# 10) Submission (strict checks)
# =========================
sub = pd.DataFrame({
    "id": test_df["id"],
    "Tg":      final_test["Tg"],
    "FFV":     final_test["FFV"],
    "Tc":      final_test["Tc"],
    "Density": final_test["Density"],
    "Rg":      final_test["Rg"],
}).replace([np.inf,-np.inf],0).fillna(0)

assert sub.shape[0]==test_df.shape[0], "Row count mismatch!"
assert list(sub.columns)==list(sample_sub.columns), "Column mismatch!"
for c in TARGETS: assert np.isfinite(sub[c]).all(), f"Non-finite in {c}"

sub.to_csv("submission.csv", index=False)
print("✅ submission.csv created:", sub.shape)


