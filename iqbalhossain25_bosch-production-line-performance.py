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


# ============================================================
# Bosch: Cost-Sensitive Gradient Boosting + Feature Hashing
# Models: XGBoost, LightGBM, (optional) CatBoost + Ensemble
# Adds: PR/ROC plots, Threshold–Cost curve, Confusion Matrix,
#       Model metric bar charts, Warnings silenced
# Kaggle-ready (reads zipped CSVs from /kaggle/input/)
# ============================================================

import os, gc, math, json, time, warnings
from typing import Dict, Any
import numpy as np
import pandas as pd
from scipy import sparse as sp

# ML / metrics
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score, confusion_matrix,
    precision_recall_curve, roc_curve
)
from sklearn.feature_extraction import FeatureHasher
from joblib import dump

# Plots
import matplotlib.pyplot as plt

# Silencing noisy warnings (optional)
warnings.filterwarnings("ignore")

# ---- Boosted tree libs ----
HAVE_XGB = True
try:
    from xgboost import XGBClassifier
except Exception:
    HAVE_XGB = False

HAVE_LGBM = True
try:
    import lightgbm as lgb
except Exception:
    HAVE_LGBM = False

HAVE_CAT = True
try:
    from catboost import CatBoostClassifier
except Exception:
    HAVE_CAT = False


# =========================
# Config
# =========================
DATA_DIR = "/kaggle/input/bosch-production-line-performance"
WORK_DIR = "/kaggle/working"
ARTIFACT_DIR = os.path.join(WORK_DIR, "artifacts")
PLOTS_DIR = os.path.join(WORK_DIR, "plots")
os.makedirs(ARTIFACT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

TRAIN_NUM_ZIP = os.path.join(DATA_DIR, "train_numeric.csv.zip")
TRAIN_CAT_ZIP = os.path.join(DATA_DIR, "train_categorical.csv.zip")
TEST_NUM_ZIP  = os.path.join(DATA_DIR, "test_numeric.csv.zip")
TEST_CAT_ZIP  = os.path.join(DATA_DIR, "test_categorical.csv.zip")

# ---- Speed toggles ----
FAST_PREVIEW = True      # দ্রুত টেস্টিং; ফুল রান চাইলে False দিন
USE_GPU_XGB  = True      # Kaggle Accelerator: GPU দিলে True রাখুন

if FAST_PREVIEW:
    N_ROWS_TRAIN = 100_000
    N_ROWS_TEST  = None
    N_HASHED_FEATURES = 2**17     # 131,072
    CHUNKSIZE_CAT     = 100_000
    N_SPLITS          = 3
    EARLY_STOP        = 100
    XGB_N_ESTIMATORS  = 1500
    LGBM_N_ESTIMATORS = 1500
    CAT_ITERATIONS    = 2000
    SPARSE_COL_MIN_NN_RATE = 0.001  # 0.1% non-null না হলে কলাম ড্রপ
else:
    N_ROWS_TRAIN = None
    N_ROWS_TEST  = None
    N_HASHED_FEATURES = 2**18     # 262,144
    CHUNKSIZE_CAT     = 200_000
    N_SPLITS          = 5
    EARLY_STOP        = 200
    XGB_N_ESTIMATORS  = 5000
    LGBM_N_ESTIMATORS = 5000
    CAT_ITERATIONS    = 5000
    SPARSE_COL_MIN_NN_RATE = 0.0

# কোন মডেল চালাবেন
ENABLE_XGB        = True
ENABLE_LGBM_GBDT  = True
ENABLE_LGBM_DART  = False if FAST_PREVIEW else True
ENABLE_CATBOOST   = False if FAST_PREVIEW else True

# Cost matrix (FN >> FP)
COST_FP = 1.0
COST_FN = 20.0
RANDOM_STATE = 42

def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# =========================
# Threshold tuner (cost)
# =========================
def tune_threshold_for_cost(y_true: np.ndarray, p: np.ndarray,
                            cost_fp: float, cost_fn: float):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t, best_cost = 0.5, float("inf")
    curve = []
    for t in thresholds:
        yhat = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yhat, labels=[0,1]).ravel()
        c = cost_fn*fn + cost_fp*fp
        curve.append({"t": float(t), "cost": int(c), "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn)})
        if c < best_cost:
            best_cost, best_t = c, t
    return best_t, best_cost, {"curve": curve}

# =========================
# Data loading (zips OK)
# =========================
def read_train_numeric(n_rows=None):
    log(f"Reading numeric (train): {TRAIN_NUM_ZIP}")
    df = pd.read_csv(TRAIN_NUM_ZIP, nrows=n_rows, low_memory=False)
    assert "Id" in df.columns and "Response" in df.columns
    df.sort_values("Id", inplace=True)
    ids = df["Id"].astype(np.int64).values
    y   = df["Response"].astype(np.int8).values
    Xn  = df.drop(columns=["Id", "Response"])
    for c in Xn.columns:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce", downcast="float")
    X_num = sp.csr_matrix(Xn.values)
    log(f"Numeric loaded: rows={X_num.shape[0]}, cols={X_num.shape[1]}")
    del df, Xn; gc.collect()
    return ids, y, X_num

def read_test_numeric(n_rows=None):
    log(f"Reading numeric (test): {TEST_NUM_ZIP}")
    df = pd.read_csv(TEST_NUM_ZIP, nrows=n_rows, low_memory=False)
    assert "Id" in df.columns
    df.sort_values("Id", inplace=True)
    ids = df["Id"].astype(np.int64).values
    Xn  = df.drop(columns=["Id"])
    for c in Xn.columns:
        Xn[c] = pd.to_numeric(Xn[c], errors="coerce", downcast="float")
    X_num = sp.csr_matrix(Xn.values)
    log(f"Numeric test loaded: rows={X_num.shape[0]}, cols={X_num.shape[1]}")
    del df, Xn; gc.collect()
    return ids, X_num

def hashed_cat_matrix(cat_zip_path: str,
                      id_to_row_index: Dict[int, int],
                      n_rows_total: int,
                      n_features: int = N_HASHED_FEATURES,
                      chunksize: int = CHUNKSIZE_CAT) -> sp.csr_matrix:
    """
    Subset-aware categorical hashing:
    - Only rows whose Id is in id_to_row_index are processed (no KeyError)
    - Optional rare-column pruning by non-null rate
    """
    log(f"Hashing categorical (chunked, subset-aware): {cat_zip_path}")
    hasher = FeatureHasher(n_features=n_features, input_type="string", alternate_sign=False)

    known_ids = set(id_to_row_index.keys())
    data_all, rows_all, cols_all = [], [], []

    for chunk in pd.read_csv(cat_zip_path, chunksize=chunksize, low_memory=False):
        if "Id" not in chunk.columns:
            raise ValueError("Expected 'Id' column in categorical file.")
        chunk["Id"] = chunk["Id"].astype(np.int64)

        # keep only ids that exist in numeric subset
        mask = chunk["Id"].map(known_ids.__contains__).values
        if not np.any(mask):
            continue
        chunk = chunk.loc[mask]

        # drop all-NaN columns
        chunk = chunk.dropna(axis=1, how="all")

        cat_cols = [c for c in chunk.columns if c != "Id"]
        if not cat_cols:
            continue

        # (optional) rare column pruning by non-null rate
        if SPARSE_COL_MIN_NN_RATE > 0:
            nn_rate = chunk[cat_cols].notna().mean()
            keep = nn_rate[nn_rate > SPARSE_COL_MIN_NN_RATE].index.tolist()
            if keep:
                cat_cols = keep
            if not cat_cols:
                continue

        ids_chunk = chunk["Id"].values

        # Build tokens
        tokens_per_row = []
        for row in chunk[cat_cols].itertuples(index=False, name=None):
            toks = [f"{c}={v}" for c, v in zip(cat_cols, row) if pd.notna(v)]
            tokens_per_row.append(toks)

        Xh = hasher.transform(tokens_per_row).tocoo()
        if Xh.nnz == 0:
            continue

        global_rows_for_chunk = np.fromiter((id_to_row_index[int(i)] for i in ids_chunk),
                                            dtype=np.int64, count=len(ids_chunk))
        global_rows = global_rows_for_chunk[Xh.row]

        data_all.append(Xh.data); rows_all.append(global_rows); cols_all.append(Xh.col)

        del chunk, Xh, tokens_per_row, global_rows_for_chunk, global_rows
        gc.collect()

    if not data_all:
        return sp.csr_matrix((n_rows_total, n_features))

    data_all = np.concatenate(data_all)
    rows_all = np.concatenate(rows_all)
    cols_all = np.concatenate(cols_all)

    X_cat = sp.coo_matrix((data_all, (rows_all, cols_all)),
                          shape=(n_rows_total, n_features)).tocsr()
    log(f"Categorical hashed: rows={X_cat.shape[0]}, hashed_bins={X_cat.shape[1]}")
    return X_cat

def load_design_matrix_train(n_rows=N_ROWS_TRAIN):
    ids, y, X_num = read_train_numeric(n_rows=n_rows)
    id_to_row = {int(i): r for r, i in enumerate(ids)}
    X_cat = hashed_cat_matrix(TRAIN_CAT_ZIP, id_to_row, len(ids),
                              n_features=N_HASHED_FEATURES, chunksize=CHUNKSIZE_CAT)
    X = sp.hstack([X_num, X_cat], format="csr")
    log(f"TRAIN matrix: rows={X.shape[0]}, cols={X.shape[1]}")
    return ids, y, X

def load_design_matrix_test(n_rows=N_ROWS_TEST):
    ids, X_num = read_test_numeric(n_rows=n_rows)
    id_to_row = {int(i): r for r, i in enumerate(ids)}
    X_cat = hashed_cat_matrix(TEST_CAT_ZIP, id_to_row, len(ids),
                              n_features=N_HASHED_FEATURES, chunksize=CHUNKSIZE_CAT)
    X = sp.hstack([X_num, X_cat], format="csr")
    log(f"TEST matrix: rows={X.shape[0]}, cols={X.shape[1]}")
    return ids, X

# =========================
# Model builders
# =========================
def build_xgb():
    if not HAVE_XGB:
        raise ImportError("xgboost not installed.")
    # XGBoost 2.x: GPU-এর জন্য tree_method নয়, device="cuda" ব্যবহার করুন
    params = dict(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="aucpr",
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=EARLY_STOP,
        device=("cuda" if USE_GPU_XGB else "cpu")
    )
    return XGBClassifier(**params)

def build_lgbm(boosting_type="gbdt"):
    if not HAVE_LGBM:
        raise ImportError("lightgbm not installed.")
    # নোট: GPU চাইলে Kaggle ইমেজে সাপোর্ট সাধারণত থাকে; verbosity কমানো হলো
    return lgb.LGBMClassifier(
        objective="binary",
        boosting_type=boosting_type,
        n_estimators=LGBM_N_ESTIMATORS,
        learning_rate=0.03,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=-1
    )

def build_cat():
    if not HAVE_CAT:
        raise ImportError("catboost not installed.")
    return CatBoostClassifier(
        task_type="GPU" if USE_GPU_XGB else "CPU",
        devices="0" if USE_GPU_XGB else None,
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=CAT_ITERATIONS,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=3.0,
        random_seed=RANDOM_STATE,
        od_type="Iter",
        od_wait=EARLY_STOP,
        verbose=False,
        allow_writing_files=False
    )

# =========================
# CV runner
# =========================
def run_cv(model_name: str, model_builder, X: sp.csr_matrix, y: np.ndarray,
           cost_fp: float, cost_fn: float, early_stop_rounds: int = EARLY_STOP) -> Dict[str, Any]:
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_pred = np.zeros_like(y, dtype=float)
    pr_aucs, roc_aucs, costs, best_ts, best_iters = [], [], [], [], []
    fold = 0

    for tr_idx, va_idx in skf.split(X, y):
        fold += 1
        log(f"[{model_name}] Fold {fold}/{N_SPLITS}")
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        w_tr = np.where(y_tr == 1, cost_fn, cost_fp)
        w_va = np.where(y_va == 1, cost_fn, cost_fp)

        model = model_builder()

        if model_name.startswith("lgbm"):
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                callbacks=[lgb.early_stopping(stopping_rounds=early_stop_rounds, verbose=False)]
            )
            bi = getattr(model, "best_iteration_", None)
        elif model_name.startswith("xgb"):
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                sample_weight_eval_set=[w_va],
                verbose=False
            )
            bi = getattr(model, "best_iteration", None)
        elif model_name.startswith("cat"):
            model.fit(
                X_tr, y_tr,
                sample_weight=w_tr,
                eval_set=[(X_va, y_va)],
                use_best_model=True
            )
            bi = model.get_best_iteration()
        else:
            model.fit(X_tr, y_tr, sample_weight=w_tr); bi = None

        p = model.predict_proba(X_va)[:, 1]
        oof_pred[va_idx] = p

        pr = average_precision_score(y_va, p)
        ra = roc_auc_score(y_va, p)
        t_best, cost_best, _ = tune_threshold_for_cost(y_va, p, cost_fp, cost_fn)

        pr_aucs.append(pr); roc_aucs.append(ra)
        costs.append(cost_best); best_ts.append(float(t_best))
        if bi is not None:
            best_iters.append(int(bi))

        dump(model, os.path.join(ARTIFACT_DIR, f"{model_name}_fold{fold}.joblib"))
        log(f"[{model_name}] Fold {fold}: PR-AUC={pr:.5f}, ROC-AUC={ra:.5f}, t*={t_best:.2f}, cost={cost_best}"
            + (f", best_iter={bi}" if bi is not None else ""))

        del model, X_tr, X_va, y_tr, y_va, w_tr, w_va, p; gc.collect()

    t_global, cost_global, details = tune_threshold_for_cost(y, oof_pred, cost_fp, cost_fn)

    result = {
        "model": model_name,
        "pr_auc_mean": float(np.mean(pr_aucs)),
        "pr_auc_std": float(np.std(pr_aucs)),
        "roc_auc_mean": float(np.mean(roc_aucs)),
        "roc_auc_std": float(np.std(roc_aucs)),
        "cv_cost_mean": float(np.mean(costs)),
        "cv_cost_std": float(np.std(costs)),
        "t_cv_mean": float(np.mean(best_ts)),
        "t_global": float(t_global),
        "oof_pred_path": os.path.join(ARTIFACT_DIR, f"oof_{model_name}.npy"),
        "threshold_cost_curve": os.path.join(ARTIFACT_DIR, f"threshold_cost_curve_{model_name}.json"),
        "best_iterations": best_iters,
        "recommended_n_estimators": int(np.mean(best_iters)) if best_iters else None
    }
    np.save(result["oof_pred_path"], oof_pred)
    with open(result["threshold_cost_curve"], "w") as f:
        json.dump(details, f)

    log(f"[{model_name}] OOF: PR-AUC={result['pr_auc_mean']:.5f}±{result['pr_auc_std']:.5f} | "
        f"ROC-AUC={result['roc_auc_mean']:.5f}±{result['roc_auc_std']:.5f} | "
        f"t_global={result['t_global']:.2f} | mean_cost={result['cv_cost_mean']:.1f}")
    return result


# =========================
# Plot helpers
# =========================
def plot_pr_roc(y_true, p, prefix):
    pr, rc, th = precision_recall_curve(y_true, p)
    fpr, tpr, _ = roc_curve(y_true, p)

    # PR
    plt.figure(figsize=(6,4))
    plt.plot(rc, pr, lw=2)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR Curve ({prefix})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"{prefix}_pr_curve.png")); plt.close()

    # ROC
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, lw=2)
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC Curve ({prefix})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"{prefix}_roc_curve.png")); plt.close()

def plot_threshold_cost(details_json_path, prefix):
    with open(details_json_path, "r") as f:
        data = json.load(f)["curve"]
    ts = [d["t"] for d in data]; cs = [d["cost"] for d in data]
    plt.figure(figsize=(6,4))
    plt.plot(ts, cs, lw=2)
    plt.xlabel("Threshold"); plt.ylabel("Expected Cost")
    plt.title(f"Threshold–Cost Curve ({prefix})"); plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"{prefix}_threshold_cost.png")); plt.close()

def plot_confmat(cm, prefix):
    tn, fp, fn, tp = cm.ravel()
    fig, ax = plt.subplots(figsize=(4.5,4))
    im = ax.imshow([[tn, fp],[fn, tp]], cmap="Blues")
    for (i,j), val in np.ndenumerate([[tn, fp],[fn, tp]]):
        ax.text(j, i, int(val), ha="center", va="center", fontsize=12)
    ax.set_xticks([0,1]); ax.set_xticklabels(["Pred 0","Pred 1"])
    ax.set_yticks([0,1]); ax.set_yticklabels(["True 0","True 1"])
    ax.set_title(f"Confusion Matrix ({prefix})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, f"{prefix}_confusion.png")); plt.close()

def plot_model_bars(results):
    if not results: return
    df = pd.DataFrame([{
        "model": r["model"], "PR-AUC": r["pr_auc_mean"], "ROC-AUC": r["roc_auc_mean"]
    } for r in results])
    # PR-AUC
    plt.figure(figsize=(6,4))
    order = df.sort_values("PR-AUC", ascending=False)
    plt.bar(order["model"], order["PR-AUC"])
    plt.xticks(rotation=20); plt.ylabel("PR-AUC"); plt.title("Model PR-AUC (OOF mean)")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "models_pr_auc_bar.png")); plt.close()
    # ROC-AUC
    plt.figure(figsize=(6,4))
    order = df.sort_values("ROC-AUC", ascending=False)
    plt.bar(order["model"], order["ROC-AUC"])
    plt.xticks(rotation=20); plt.ylabel("ROC-AUC"); plt.title("Model ROC-AUC (OOF mean)")
    plt.tight_layout(); plt.savefig(os.path.join(PLOTS_DIR, "models_roc_auc_bar.png")); plt.close()


# =========================
# Final refit for test (optional)
# =========================
def refit_full(model_name: str, model_builder, X: sp.csr_matrix, y: np.ndarray, n_estimators_hint: int = None):
    model = model_builder()
    if n_estimators_hint is not None:
        if model_name.startswith("lgbm"):
            model.set_params(n_estimators=int(n_estimators_hint))
        elif model_name.startswith("xgb"):
            model.set_params(n_estimators=int(n_estimators_hint))
        elif model_name.startswith("cat"):
            model.set_params(iterations=int(n_estimators_hint))
    w = np.where(y == 1, COST_FN, COST_FP)
    model.fit(X, y, sample_weight=w)
    dump(model, os.path.join(ARTIFACT_DIR, f"{model_name}_FULL.joblib"))
    return model


# =========================
# Main
# =========================
if __name__ == "__main__":
    log("Loading TRAIN matrix ...")
    ids_tr, y, X = load_design_matrix_train(n_rows=N_ROWS_TRAIN)

    with open(os.path.join(ARTIFACT_DIR, "run_meta.json"), "w") as f:
        json.dump({
            "rows_train": int(X.shape[0]),
            "cols_total": int(X.shape[1]),
            "hashed_bins": int(N_HASHED_FEATURES),
            "splits": int(N_SPLITS),
            "cost_fp": float(COST_FP),
            "cost_fn": float(COST_FN),
            "random_state": int(RANDOM_STATE),
            "fast_preview": FAST_PREVIEW,
            "gpu_xgb": USE_GPU_XGB
        }, f, indent=2)

    results = []

    if ENABLE_XGB and HAVE_XGB:
        results.append(run_cv("xgb_gbdt", build_xgb, X, y, COST_FP, COST_FN))
    if ENABLE_LGBM_GBDT and HAVE_LGBM:
        results.append(run_cv("lgbm_gbdt", lambda: build_lgbm("gbdt"), X, y, COST_FP, COST_FN))
    if ENABLE_LGBM_DART and HAVE_LGBM:
        results.append(run_cv("lgbm_dart", lambda: build_lgbm("dart"), X, y, COST_FP, COST_FN))
    if ENABLE_CATBOOST and HAVE_CAT:
        results.append(run_cv("cat_gbdt", build_cat, X, y, COST_FP, COST_FN))

    # Save CV summary
    with open(os.path.join(ARTIFACT_DIR, "cv_summary.json"), "w") as f:
        json.dump(results, f, indent=2)

    # ---- Create visualizations ----
    plot_model_bars(results)  # bar charts for PR/ROC

    # For each model: threshold–cost curve plot
    for r in results:
        plot_threshold_cost(r["threshold_cost_curve"], r["model"])

    # If at least two models: ensemble OOF
    oof_paths = [r["oof_pred_path"] for r in results if os.path.exists(r["oof_pred_path"])]
    if len(oof_paths) >= 2:
        oofs = [np.load(p) for p in oof_paths]
        ens_oof = np.mean(np.vstack(oofs), axis=0)
        ens_pr  = average_precision_score(y, ens_oof)
        ens_roc = roc_auc_score(y, ens_oof)
        ens_t, ens_cost, ens_details = tune_threshold_for_cost(y, ens_oof, COST_FP, COST_FN)
        np.save(os.path.join(ARTIFACT_DIR, "oof_ensemble_mean.npy"), ens_oof)
        with open(os.path.join(ARTIFACT_DIR, "threshold_cost_curve_ensemble.json"), "w") as f:
            json.dump(ens_details, f, indent=2)
        with open(os.path.join(ARTIFACT_DIR, "ensemble_summary.json"), "w") as f:
            json.dump({
                "model": "ensemble_mean",
                "pr_auc": float(ens_pr),
                "roc_auc": float(ens_roc),
                "t_global": float(ens_t),
                "cost": float(ens_cost),
                "members": [os.path.basename(p)[4:-4] for p in oof_paths]
            }, f, indent=2)

        # Ensemble PR/ROC + Confusion at t*
        plot_pr_roc(y, ens_oof, "ensemble")
        yhat = (ens_oof >= ens_t).astype(int)
        cm = confusion_matrix(y, yhat, labels=[0,1])
        plot_confmat(cm, "ensemble")
        # Ensemble threshold–cost
        plot_threshold_cost(os.path.join(ARTIFACT_DIR, "threshold_cost_curve_ensemble.json"), "ensemble")
    else:
        # Single best model plots from its OOF
        best = sorted(results, key=lambda d: d["pr_auc_mean"], reverse=True)[0]
        p = np.load(best["oof_pred_path"])
        plot_pr_roc(y, p, best["model"])
        yhat = (p >= best["t_global"]).astype(int)
        cm = confusion_matrix(y, yhat, labels=[0,1])
        plot_confmat(cm, best["model"])

    log("Done. Artifacts in /kaggle/working/artifacts/ and plots in /kaggle/working/plots/")


