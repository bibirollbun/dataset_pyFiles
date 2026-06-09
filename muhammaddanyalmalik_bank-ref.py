import warnings
import logging
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

warnings.filterwarnings("ignore", category=FutureWarning)


sns.set_theme(style="whitegrid", font_scale=1.05)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

EPS = 1e-15


def safe_clip(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, EPS, 1 - EPS)


def load_predictions_list(file_paths: List[str], pred_col: str = "y") -> pd.DataFrame:
    """
    Load multiple prediction CSVs and merge on 'id'. 
    Each file's prediction column is renamed to y_<stem>.
    """
    dfs = []
    for p in file_paths:
        pth = Path(p)
        if not pth.exists():
            raise FileNotFoundError(f"File not found: {p}")
        df = pd.read_csv(p)
        if "id" not in df.columns:
            raise ValueError(f"'id' column missing in {p}")
        if pred_col not in df.columns:
            raise ValueError(f"Prediction column '{pred_col}' missing in {p}")
        dfs.append(df[["id", pred_col]].rename(columns={pred_col: f"y_{pth.stem}"}))
    # Merge all on id (inner join)
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="id", how="inner")
    if len(merged) == 0:
        raise RuntimeError("No common IDs found across prediction files.")
    return merged


# ---------- Blend functions for N models ----------
def weighted_average_multi(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return np.dot(preds, weights)


def geometric_mean_multi(preds: np.ndarray, **_kw) -> np.ndarray:
    logs = np.log(safe_clip(preds))
    gm = np.exp(logs.mean(axis=1))
    return safe_clip(gm)


def rank_blend_multi(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    ranks = np.column_stack([pd.Series(preds[:, i]).rank(pct=True).values for i in range(preds.shape[1])])
    blended = np.dot(ranks, weights)
    return safe_clip(blended)


def probit_blend_multi(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    n = np.column_stack([stats.norm.ppf(safe_clip(preds[:, i])) for i in range(preds.shape[1])])
    blended = np.dot(n, weights)
    return safe_clip(stats.norm.cdf(blended))


def logit_avg_multi(preds: np.ndarray, weights: np.ndarray) -> np.ndarray:
    def logit(p): return np.log(p) - np.log1p(-p)
    L = np.column_stack([logit(safe_clip(preds[:, i])) for i in range(preds.shape[1])])
    blended = np.dot(L, weights)
    inv_logit = lambda x: 1 / (1 + np.exp(-x))
    return safe_clip(inv_logit(blended))


# Map method name to function (all take preds, weights)
BLEND_MULTI = {
    "weighted_average": weighted_average_multi,
    "geometric_mean": lambda preds, weights: geometric_mean_multi(preds),
    "rank_blend": rank_blend_multi,
    "probit": probit_blend_multi,
    "logit_avg": logit_avg_multi,
}


def compute_multi_diagnostics(preds: np.ndarray, y_true: Optional[np.ndarray] = None) -> Dict:
    """
    preds: (n_samples, n_models)
    """
    diag = {}
    n_models = preds.shape[1]
    for i in range(n_models):
        p = safe_clip(preds[:, i])
        diag[f"mean_{i+1}"] = float(p.mean())
        diag[f"std_{i+1}"] = float(p.std())
    corr = np.corrcoef(preds.T)
    diag["pairwise_corr_matrix"] = corr
    if y_true is not None:
        y = np.asarray(y_true)
        for i in range(n_models):
            p = safe_clip(preds[:, i])
            diag[f"logloss_{i+1}"] = float(log_loss(y, p))
            diag[f"brier_{i+1}"] = float(brier_score_loss(y, p))
            try:
                diag[f"auc_{i+1}"] = float(roc_auc_score(y, p))
            except Exception:
                diag[f"auc_{i+1}"] = None
    return diag


# ---------- optimization of weights ----------
def optimize_weights(preds: np.ndarray, y_true: np.ndarray, method: str = "weighted_average", metric: str = "logloss") -> np.ndarray:
    """
    Optimize weights (non-negative, sum to 1) to minimize metric.
    metric: 'logloss' (default), 'brier', 'auc' (maximize auc -> minimize -auc).
    """
    n_models = preds.shape[1]
    assert method in BLEND_MULTI, f"Unknown method {method}"

    def objective(w):
        w = np.array(w)
        if np.all(w <= 1e-12):
            return 1e6
        # compute blended
        blended = BLEND_MULTI[method](preds, w)
        if metric == "logloss":
            return log_loss(y_true, safe_clip(blended))
        elif metric == "brier":
            return brier_score_loss(y_true, safe_clip(blended))
        elif metric == "auc":
            try:
                return -roc_auc_score(y_true, safe_clip(blended))
            except Exception:
                return 1e6
        else:
            raise ValueError("Unsupported metric")

    # constraints: sum(w) == 1
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},)
    bounds = [(0.0, 1.0)] * n_models
    x0 = np.ones(n_models) / n_models

    res = minimize(objective, x0=x0, bounds=bounds, constraints=cons, method="SLSQP", options={"ftol": 1e-6, "maxiter": 200})
    if not res.success:
        logger.warning("Weight optimization failed: %s. Falling back to equal weights.", res.message)
        return x0
    w_opt = np.maximum(res.x, 0.0)
    # normalize to sum to 1
    if w_opt.sum() == 0:
        return x0
    return w_opt / w_opt.sum()

def plot_multi(preds: np.ndarray, blended: np.ndarray, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    n_models = preds.shape[1]

    plt.figure(figsize=(4 * (min(n_models, 3) + 1), 4))
    for i in range(min(n_models, 3)):
        plt.subplot(1, min(n_models, 3) + 1, i + 1)
        sns.kdeplot(preds[:, i], fill=True, alpha=0.5)
        plt.title(f"Model {i+1} dist")
    plt.subplot(1, min(n_models, 3) + 1, min(n_models, 3) + 1)
    sns.kdeplot(blended, fill=True, alpha=0.5)
    plt.title("Blended dist")
    save_path = out_dir / "multi_blend_dists.png"
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    logger.info("Saved plot to %s", save_path)


def top_disagreements(df_ids: np.ndarray, preds: np.ndarray, top_n: int = 10) -> pd.DataFrame:
    """
    Returns top N rows with largest std deviation across model predictions (i.e. most disagreement).
    """
    stds = preds.std(axis=1)
    idx = np.argsort(-stds)[:top_n]
    rows = []
    for i in idx:
        row = {"id": df_ids[i], "std": float(stds[i])}
        for m in range(preds.shape[1]):
            row[f"model_{m+1}"] = float(preds[i, m])
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- top-level function ----------
def blend_multiple_and_save(
    pred_files: List[str],
    out_submission: str = "/kaggle/working/submission.csv",
    method: str = "weighted_average",
    weights: Optional[List[float]] = None,
    true_labels_file: Optional[str] = None,
    pred_col_name: str = "y",
    metric_for_opt: str = "logloss",
    save_plots: bool = True,
    top_k_disagreements: int = 10,
):
    """
    pred_files: list of CSV files (each must have 'id' and prediction column)
    method: blending method name
    weights: if provided, list of length n_models summing to 1 (will be normalized)
    true_labels_file: optional CSV with columns ['id','y'] used for weight optimization
    metric_for_opt: 'logloss'|'brier'|'auc'
    """
    # Load and merge predictions
    merged = load_predictions_list(pred_files, pred_col=pred_col_name)
    ids = merged["id"].values
    pred_cols = [c for c in merged.columns if c != "id"]
    preds = merged[pred_cols].values.astype(float)
    preds = safe_clip(preds)

    # optionally attach true labels
    y_true = None
    if true_labels_file:
        ydf = pd.read_csv(true_labels_file)
        if "id" not in ydf.columns or "y" not in ydf.columns:
            raise ValueError("True labels file must contain 'id' and 'y' columns")
        merged2 = merged.merge(ydf[["id", "y"]], on="id", how="left")
        if merged2["y"].isna().any():
            logger.warning("Some merged rows are missing true labels; dropping those for optimization")
            merged2 = merged2.dropna(subset=["y"]).reset_index(drop=True)
        ids = merged2["id"].values
        preds = merged2[pred_cols].values.astype(float)
        preds = safe_clip(preds)
        y_true = merged2["y"].values.astype(int)

    # diagnostics
    diag = compute_multi_diagnostics(preds, y_true)
    logger.info("Diagnostics summary:")
    # print simple diagnostics
    n_models = preds.shape[1]
    for i in range(n_models):
        logger.info(" Model %d: mean=%.6f std=%.6f", i + 1, diag[f"mean_{i+1}"], diag[f"std_{i+1}"])
    logger.info(" Pairwise correlation matrix (shape %s):\n%s", diag["pairwise_corr_matrix"].shape, np.round(diag["pairwise_corr_matrix"], 4))

    # decide weights
    if weights is not None:
        w = np.array(weights, dtype=float)
        if w.size != n_models:
            raise ValueError(f"weights length ({w.size}) does not match number of models ({n_models})")
        if (w < 0).any():
            raise ValueError("weights must be non-negative")
        # normalize
        w = w / w.sum()
        logger.info("Using provided weights (normalized): %s", np.round(w, 4).tolist())
    else:
        if y_true is not None and method in BLEND_MULTI and method != "geometric_mean":
            logger.info("Optimizing weights using metric=%s and method=%s", metric_for_opt, method)
            try:
                w = optimize_weights(preds, y_true, method=method, metric=metric_for_opt)
                logger.info("Optimized weights: %s", np.round(w, 4).tolist())
            except Exception as e:
                logger.warning("Optimization failed: %s. Falling back to equal weights.", e)
                w = np.ones(n_models) / n_models
        else:
            
            w = np.ones(n_models) / n_models
            logger.info("Using equal weights: %s", np.round(w, 4).tolist())

    # blend
    if method not in BLEND_MULTI:
        raise ValueError(f"method must be one of {list(BLEND_MULTI.keys())}")
    blended = BLEND_MULTI[method](preds, w)

    # save submission
    out_df = pd.DataFrame({"id": ids, "y": safe_clip(blended)})
    out_path = Path(out_submission)
    out_df.to_csv(out_path, index=False)
    logger.info("Saved blended submission to %s", out_path)

    if save_plots:
        plot_multi(preds, blended, out_dir=out_path.parent)
    if top_k_disagreements > 0:
        top_df = top_disagreements(ids, preds, top_n=top_k_disagreements)
        logger.info("Top %d disagreement rows (highest std across models):", top_k_disagreements)
        logger.info("\n%s", top_df.to_string(index=False))
    
    if y_true is not None:
        final_logloss = log_loss(y_true, blended)
        final_brier = brier_score_loss(y_true, blended)
        try:
            final_auc = roc_auc_score(y_true, blended)
        except Exception:
            final_auc = None
        logger.info("Final metrics: logloss=%.6f brier=%.6f auc=%s", final_logloss, final_brier, final_auc)

    return out_path, w

# ---------------- Example usage ----------------
pred_files = [
    "/kaggle/input/ps-s5e8-blend-xgb-lgb/submission.csv",
    "/kaggle/input/train-more-xgb-nn-lb-0-9774/submission_ensemble_train_more.csv",
    "/kaggle/input/ps-s5e8-h-blend-bokeh-only-public-ml-solutions/submission.csv",  
    "/kaggle/input/top-1-solution-0-97754-esay-is-all-you-need/submission.csv",
    "/kaggle/input/ps-s5e8-h-blend-bokeh-only-public-ml-solutions/submission_v101.csv",
    "/kaggle/input/ps-s5e8-h-blend-bokeh-only-public-ml-solutions/submission_v102.csv",
    "/kaggle/input/ps-s5e8-h-blend-bokeh-only-public-ml-solutions/submission_v104.csv",
    "/kaggle/input/train-more-xgb-nn-lb-0-9774/submission_nn_train_more.csv",
    "/kaggle/input/train-more-xgb-nn-lb-0-9774/submission_xgb_train_more.csv",
    "/kaggle/input/playground-series-s5e8/sample_submission.csv"
]

out_submission = "/kaggle/working/submission.csv"


method = "weighted_average"

weights = None #np.array([0.45, 0.35, 0.25, 0.15, 0.45, 0.45, 0.35, 0.55, 0.65, 0.25])


true_labels_file = None  


metric_for_opt = "logloss"

# Run blending
out_path, used_weights = blend_multiple_and_save(
    pred_files=pred_files,
    out_submission=out_submission,
    method=method,
    weights=weights,
    true_labels_file=true_labels_file,
    pred_col_name="y",
    metric_for_opt=metric_for_opt,
    save_plots=True,
    top_k_disagreements=10,
)

print("Done. Output:", out_path)
print("Weights used:", used_weights)


# pred_files = [
#     "/kaggle/input/ps-s5e8-blend-xgb-lgb/submission.csv",
#     "/kaggle/input/train-more-xgb-nn-lb-0-9774/submission_ensemble_train_more.csv",
#     "/kaggle/input/me-on-25-upvote-and-copy/submission.csv",
#     "/kaggle/input/ps-s5e8-h-blend-bokeh-only-public-ml-solutions/submission.csv",
#     "/kaggle/input/bank-dataset-classification-s5e8/submission.csv",
#     "/kaggle/input/top-1-solution-0-97754-esay-is-all-you-need/submission.csv",
# ]

# out_submission = "/kaggle/working/submission.csv"


# method = "weighted_average"

# weights = [0.20, 0.15, 0.15, 0.10, 0.20, 0.20]


# import os

# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))


# import numpy as np
# import pandas as pd

# test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")
# sub1 = pd.read_csv("/kaggle/input/ps-s5e8-blend-xgb-lgb/submission.csv")
# sub2 = pd.read_csv("/kaggle/input/train-more-xgb-nn-lb-0-9774/submission_ensemble_train_more.csv")
# sub3 = pd.read_csv("/kaggle/input/ps-s5e8-h-blend-bokeh-only-public-ml-solutions/submission.csv")
# sub4 = pd.read_csv("/kaggle/input/top-1-solution-0-97754-esay-is-all-you-need/submission.csv")
# sub5 = pd.read_csv("/kaggle/input/me-on-25-upvote-and-copy/submission.csv")
# sub6 = pd.read_csv("/kaggle/input/bank-dataset-classification-s5e8/submission.csv")


# # Extract prediction columns
# r1 = sub1['y']
# r2 = sub2['y']
# r3 = sub3['y']
# r4 = sub4['y']
# r5 = sub5['y']
# r6 = sub6['y']

# # Define weights (should sum to 1)
# w1, w2, w3, w4, w5, w6 = 0.1, 0.5, 0.1, 0.25, 0.20, 0.10

# # Final blend (like your style)
# sub = w1*r1 + w2*r2 + w3*r3 + w4*r4 + w5*r5 + w6*r6
# # Save final submission
# submission = pd.DataFrame({"id": test_df["id"], "y": sub})
# submission.to_csv("submission.csv", index=False)
# submission.head()

