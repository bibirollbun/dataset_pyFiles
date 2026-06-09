# Load data

df_train = pd.read_csv("/kaggle/input/playground-series-s5e12/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e12/test.csv")

print(df_train.shape, df_test.shape)


df_train.info()


# ============================================================
# Section 1 — Data sanity checks (adapted to df_train / df_test)
# ============================================================

import numpy as np
import pandas as pd

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 140)

# ----------------------------
# 0) Assumptions from your info()
# ----------------------------
# df_train and df_test already exist
target_col = "diagnosed_diabetes"
id_col = "id"

assert target_col in df_train.columns, f"Target '{target_col}' not in df_train."
assert id_col in df_train.columns and id_col in df_test.columns, f"ID '{id_col}' must be in both train/test."

y = df_train[target_col]
X_train = df_train.drop(columns=[target_col])
X_test  = df_test.copy()

# ----------------------------
# 1) Dataset structure
# ----------------------------
def report_structure(X_tr: pd.DataFrame, X_te: pd.DataFrame) -> None:
    print("\n" + "="*80)
    print("1.1 Dataset structure")
    print("="*80)
    print(f"Train shape (features only): {X_tr.shape}")
    print(f"Test shape:                 {X_te.shape}")

    tr_cols = set(X_tr.columns)
    te_cols = set(X_te.columns)

    only_in_train = sorted(list(tr_cols - te_cols))
    only_in_test  = sorted(list(te_cols - tr_cols))

    if only_in_train or only_in_test:
        print("\n[Schema mismatch]")
        if only_in_train:
            print("Columns only in TRAIN:", only_in_train)
        if only_in_test:
            print("Columns only in TEST: ", only_in_test)
    else:
        print("\nSchema check: ✅ Train/Test columns match.")

    print("\nDtype counts (train features):")
    print(X_tr.dtypes.value_counts().to_string())

    cat_cols = [c for c in X_tr.columns if X_tr[c].dtype == "object" or str(X_tr[c].dtype).startswith("category")]
    num_cols = [c for c in X_tr.columns if c not in cat_cols]

    print(f"\nInferred numeric cols:     {len(num_cols)}")
    print(f"Inferred categorical cols: {len(cat_cols)}")

report_structure(X_train, X_test)

# ----------------------------
# 2) Missingness / constant columns
# ----------------------------
def report_missing_and_constants(X_tr: pd.DataFrame, X_te: pd.DataFrame, top_n: int = 20) -> dict:
    print("\n" + "="*80)
    print("Missingness & constant features")
    print("="*80)

    miss_tr = X_tr.isna().mean().sort_values(ascending=False)
    miss_te = X_te.isna().mean().sort_values(ascending=False)

    # show only non-zero missingness to keep it readable
    nz_tr = miss_tr[miss_tr > 0]
    nz_te = miss_te[miss_te > 0]

    print("\nMissingness (train) — non-zero only:")
    print(nz_tr.head(top_n).to_string() if not nz_tr.empty else "✅ No missing values in train features.")

    print("\nMissingness (test) — non-zero only:")
    print(nz_te.head(top_n).to_string() if not nz_te.empty else "✅ No missing values in test features.")

    # constant columns (train)
    nunq_tr = X_tr.nunique(dropna=False)
    constant_cols = nunq_tr[nunq_tr <= 1].index.tolist()

    if constant_cols:
        print("\nConstant columns (train):", constant_cols)
    else:
        print("\nConstant columns (train): ✅ none")

    # constant columns in both train & test
    nunq_te = X_te.nunique(dropna=False)
    constant_both = [c for c in X_tr.columns if nunq_tr.get(c, 2) <= 1 and nunq_te.get(c, 2) <= 1]
    if constant_both:
        print("Constant columns in BOTH train & test:", constant_both)

    return {
        "missing_train": miss_tr,
        "missing_test": miss_te,
        "constant_cols": constant_cols,
        "constant_both": constant_both,
    }

audit_basic = report_missing_and_constants(X_train, X_test)

# ----------------------------
# 3) Target distribution
# ----------------------------
def report_target(y: pd.Series) -> None:
    print("\n" + "="*80)
    print("1.2 Target distribution")
    print("="*80)

    y_clean = y.dropna()
    print(f"Target dtype: {y.dtype}")
    print(f"Target missing rate: {y.isna().mean():.6f}")
    print(f"n = {len(y_clean):,}")

    uniq = pd.Series(y_clean.unique())
    if len(uniq) <= 10:
        print(f"Unique values: {sorted(uniq.tolist())}")
    else:
        # show a quick peek
        print(f"Unique values count: {len(uniq)}")
        print("Value counts (top 10):")
        print(y_clean.value_counts().head(10).to_string())

    # binary-like check (your target is float64; may still be 0.0/1.0)
    uniq_set = set(np.unique(y_clean))
    if uniq_set.issubset({0.0, 1.0}) or uniq_set.issubset({0, 1}):
        pos_rate = float(np.mean(y_clean))
        print(f"\nPositive rate P(y=1): {pos_rate:.4f}")
        print(f"Imbalance ratio (neg:pos): {(1-pos_rate)/max(pos_rate, 1e-12):.2f}:1")
    else:
        # If it's continuous (unlikely here), show summary stats
        print("\nTarget summary statistics:")
        print(y_clean.describe().to_string())

report_target(y)

# ----------------------------
# 4) Duplicates + conflicting labels
# ----------------------------
def report_duplicates_and_conflicts(X_tr: pd.DataFrame, y: pd.Series, id_col: str | None = None) -> dict:
    print("\n" + "="*80)
    print("1.3 Duplicates and leakage risk")
    print("="*80)

    # Use all feature columns except id_col to define a "feature vector"
    cols_for_dupes = X_tr.columns.tolist()
    if id_col and id_col in cols_for_dupes:
        cols_for_dupes = [c for c in cols_for_dupes if c != id_col]

    df = X_tr[cols_for_dupes].copy()
    df["_target_"] = y.values

    dup_mask = df[cols_for_dupes].duplicated(keep=False)
    n_dup_rows = int(dup_mask.sum())
    print(f"Rows that are part of a duplicated feature-vector group: {n_dup_rows:,} / {len(df):,} ({n_dup_rows/len(df):.2%})")

    n_dup_groups = 0
    n_conflicting_groups = 0

    if n_dup_rows > 0:
        grp = df.loc[dup_mask].groupby(cols_for_dupes, dropna=False)["_target_"].nunique()
        n_dup_groups = int(grp.shape[0])
        n_conflicting_groups = int((grp > 1).sum())
        print(f"Duplicated feature-vector groups: {n_dup_groups:,}")
        print(f"Groups with conflicting labels:   {n_conflicting_groups:,}")

        if n_conflicting_groups > 0:
            print("\n⚠️ Conflicting labels for identical feature vectors detected.")
            print("If these split across folds, CV can be optimistic. Consider GroupKFold-style splitting on the feature-vector hash.")
    else:
        print("No duplicated feature vectors detected (excluding id). ✅")

    return {
        "n_dup_rows": n_dup_rows,
        "n_dup_groups": n_dup_groups,
        "n_conflicting_groups": n_conflicting_groups
    }

dup_audit = report_duplicates_and_conflicts(X_train, y, id_col=id_col)

# ----------------------------
# 5) Train–test distribution consistency (lightweight)
# ----------------------------
def psi_numeric(train_col: pd.Series, test_col: pd.Series, bins: int = 10) -> float:
    """
    Lightweight PSI on numeric columns using quantile bins from TRAIN.
    NaNs are dropped. If a column is (near) constant, PSI returns 0.
    """
    tr = train_col.dropna().astype(float)
    te = test_col.dropna().astype(float)

    if tr.empty or te.empty:
        return np.nan

    qs = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(tr, qs))

    if len(edges) <= 2:
        return 0.0

    tr_counts, _ = np.histogram(tr, bins=edges)
    te_counts, _ = np.histogram(te, bins=edges)

    tr_dist = tr_counts / max(tr_counts.sum(), 1)
    te_dist = te_counts / max(te_counts.sum(), 1)

    eps = 1e-6
    tr_dist = np.clip(tr_dist, eps, 1)
    te_dist = np.clip(te_dist, eps, 1)

    return float(np.sum((te_dist - tr_dist) * np.log(te_dist / tr_dist)))

def report_train_test_shift(X_tr: pd.DataFrame, X_te: pd.DataFrame, id_col: str | None = None, top_n: int = 12) -> pd.DataFrame:
    print("\n" + "="*80)
    print("1.4 Train–test distribution consistency (lightweight)")
    print("="*80)

    cols = X_tr.columns.tolist()
    if id_col and id_col in cols:
        cols = [c for c in cols if c != id_col]

    cat_cols = [c for c in cols if X_tr[c].dtype == "object" or str(X_tr[c].dtype).startswith("category")]
    num_cols = [c for c in cols if c not in cat_cols]

    rows = []

    # Numeric shift via PSI + summary stats
    for c in num_cols:
        rows.append({
            "feature": c,
            "type": "num",
            "train_mean": X_tr[c].mean(),
            "test_mean":  X_te[c].mean(),
            "train_std":  X_tr[c].std(),
            "test_std":   X_te[c].std(),
            "psi": psi_numeric(X_tr[c], X_te[c], bins=10),
        })

    # Categorical: unseen levels + L1 freq distance on top levels
    for c in cat_cols:
        tr_vals = X_tr[c].astype("object")
        te_vals = X_te[c].astype("object")

        tr_levels = set(tr_vals.dropna().unique())
        te_levels = set(te_vals.dropna().unique())

        unseen_in_test  = len(tr_levels - te_levels)
        unseen_in_train = len(te_levels - tr_levels)

        top_k = 20
        tr_freq = tr_vals.value_counts(normalize=True, dropna=False).head(top_k)
        te_freq = te_vals.value_counts(normalize=True, dropna=False).head(top_k)

        idx = tr_freq.index.union(te_freq.index)
        l1 = float((tr_freq.reindex(idx, fill_value=0) - te_freq.reindex(idx, fill_value=0)).abs().sum())

        rows.append({
            "feature": c,
            "type": "cat",
            "psi": np.nan,
            "unseen_levels_in_test": unseen_in_test,
            "unseen_levels_in_train": unseen_in_train,
            "top20_L1_freq_distance": l1,
        })

    shift = pd.DataFrame(rows)

    # print most shifted numeric
    if not shift[shift["type"] == "num"].empty:
        top_num = shift[shift["type"] == "num"].sort_values("psi", ascending=False).head(top_n)
        print(f"\nTop {top_n} numeric features by PSI (higher = more shift):")
        print(top_num[["feature", "psi", "train_mean", "test_mean", "train_std", "test_std"]].to_string(index=False))
    else:
        print("\nNo numeric columns detected.")

    # print categorical anomalies
    if not shift[shift["type"] == "cat"].empty:
        cat_view = shift[shift["type"] == "cat"].copy()
        cat_view["unseen_sum"] = cat_view["unseen_levels_in_test"].fillna(0) + cat_view["unseen_levels_in_train"].fillna(0)
        cat_view = cat_view.sort_values(["unseen_sum", "top20_L1_freq_distance"], ascending=False).head(top_n)
        print(f"\nTop {top_n} categorical features by unseen levels / freq distance:")
        print(cat_view[["feature", "unseen_levels_in_test", "unseen_levels_in_train", "top20_L1_freq_distance"]].to_string(index=False))
    else:
        print("\nNo categorical columns detected.")

    return shift

shift_report = report_train_test_shift(X_train, X_test, id_col=id_col, top_n=12)

# ----------------------------
# 6) Section 1 summary (for narrative)
# ----------------------------
print("\n" + "="*80)
print("Section 1 Summary")
print("="*80)
print(f"- n_train: {len(df_train):,} | n_test: {len(df_test):,}")
print(f"- n_features (excluding target): {X_train.shape[1]:,}")
print(f"- numeric features: {sum(~X_train.dtypes.isin(['object'])):,} | categorical features: {sum(X_train.dtypes.isin(['object'])):,}")
print(f"- missing values in train features: {int(X_train.isna().sum().sum()):,}")
print(f"- constant columns (train): {len(audit_basic['constant_cols'])}")
print(f"- duplicated feature-vector rows (excluding id): {dup_audit['n_dup_rows']:,}")
print(f"- conflicting duplicate groups: {dup_audit['n_conflicting_groups']:,}")

# Keep a clean feature list for Section 2
feature_cols = [c for c in X_train.columns if c != target_col]
print("\nFeature columns ready for CV/modeling in Section 2.")



cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = [c for c in X_train.columns if c not in cat_cols]

print(f"- numeric features: {len(num_cols)} | categorical features: {len(cat_cols)}")


import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

# If not installed in your Kaggle environment, add: !pip -q install catboost
from catboost import CatBoostClassifier, Pool

RANDOM_STATE = 42
N_SPLITS = 5

target_col = "diagnosed_diabetes"
id_col = "id"

# ----------------------------
# 0) Data
# ----------------------------
assert target_col in df_train.columns
assert target_col not in df_test.columns

y = df_train[target_col].astype(int).values  # ensure 0/1 integers
X_train = df_train.drop(columns=[target_col]).copy()
X_test = df_test.copy()

# Detect categorical columns robustly
cat_cols = X_train.select_dtypes(include=["object", "category"]).columns.tolist()
feature_cols = [c for c in X_train.columns if c != target_col]

print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape}")
print(f"Categorical columns ({len(cat_cols)}): {cat_cols}")

# ----------------------------
# 1) CV setup
# ----------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_pred = np.zeros(len(X_train), dtype=float)
test_pred = np.zeros(len(X_test), dtype=float)

fold_rows = []

# ----------------------------
# 2) Model config (strong baseline; minimal tuning)
# ----------------------------
# Notes:
# - loss_function='Logloss' matches probability quality
# - eval_metric='Logloss' is stable
# - depth/learning_rate/iterations chosen for strong baseline without brute forcing
# - auto_class_weights not needed given P(y=1)=0.62 (not severely imbalanced)
model_params = dict(
    loss_function="Logloss",
    eval_metric="Logloss",
    iterations=4000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    random_seed=RANDOM_STATE,
    verbose=250,
    allow_writing_files=False,
    task_type="CPU",
)

# ----------------------------
# 3) CV loop
# ----------------------------
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y), start=1):
    X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_cols)
    valid_pool = Pool(X_va, y_va, cat_features=cat_cols)
    test_pool  = Pool(X_test, cat_features=cat_cols)

    model = CatBoostClassifier(**model_params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=200)

    # Predict probabilities for the positive class
    va_pred = model.predict_proba(valid_pool)[:, 1]
    oof_pred[va_idx] = va_pred

    te_pred = model.predict_proba(test_pool)[:, 1]
    test_pred += te_pred / N_SPLITS

    # Metrics
    fold_ll = log_loss(y_va, va_pred)
    fold_auc = roc_auc_score(y_va, va_pred)

    fold_rows.append({
        "fold": fold,
        "n_train": len(tr_idx),
        "n_valid": len(va_idx),
        "pos_rate_train": float(y_tr.mean()),
        "pos_rate_valid": float(y_va.mean()),
        "logloss": fold_ll,
        "auc": fold_auc,
        "best_iteration": int(model.get_best_iteration() or model.tree_count_),
    })

    print(f"\nFold {fold}: logloss={fold_ll:.5f} | auc={fold_auc:.5f} | best_iter={fold_rows[-1]['best_iteration']}")

cv_results = pd.DataFrame(fold_rows)

# ----------------------------
# 4) Overall OOF metrics
# ----------------------------
oof_logloss = log_loss(y, oof_pred)
oof_auc = roc_auc_score(y, oof_pred)

print("\n" + "="*80)
print("CV Results (per fold)")
print("="*80)
display(cv_results)

print("\n" + "="*80)
print("OOF Summary")
print("="*80)
print(f"OOF LogLoss: {oof_logloss:.6f}")
print(f"OOF AUC:     {oof_auc:.6f}")
print("\nPer-fold:")
print(f"LogLoss mean±std: {cv_results['logloss'].mean():.6f} ± {cv_results['logloss'].std():.6f}")
print(f"AUC     mean±std: {cv_results['auc'].mean():.6f} ± {cv_results['auc'].std():.6f}")

# ----------------------------
# 5) Submission file
# ----------------------------
# Adjust the submission column name if the competition requires a different one.
# Many Kaggle comps expect 'id' + target name or 'prediction'. We'll default to target_col.
sub = pd.DataFrame({
    id_col: df_test[id_col].values,
    target_col: test_pred
})

display(sub.head())


# FIRST SUBMISSION:
#sub.to_csv("/kaggle/working/submission.csv", index=False)
#print("Saved: submission.csv")

# Score: 0.70032


# ============================================================
# Section 4 — Calibration (diagnostics + optional post-hoc calibration)
# Uses:
#   - y (0/1 array)          from df_train[target_col]
#   - oof_pred (prob array)  from CV
#   - test_pred (prob array) from CV-averaged test predictions
# Produces:
#   - Reliability diagram (raw + calibrated)
#   - Brier / LogLoss / AUC for raw and calibrated
#   - Fold-safe calibration evaluation (Platt + Isotonic)
#   - Optionally: calibrated test predictions
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.calibration import calibration_curve
from sklearn.model_selection import StratifiedKFold
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

# ----------------------------
# 0) Inputs sanity
# ----------------------------
y_true = df_train["diagnosed_diabetes"].astype(int).values
p_oof  = np.asarray(oof_pred, dtype=float)
p_test = np.asarray(test_pred, dtype=float)

assert len(y_true) == len(p_oof)
assert np.all((p_oof >= 0) & (p_oof <= 1))
assert np.all((p_test >= 0) & (p_test <= 1))

# ----------------------------
# 1) Metrics helpers
# ----------------------------
def compute_metrics(y, p, name="model"):
    return {
        "model": name,
        "logloss": log_loss(y, p),
        "brier": brier_score_loss(y, p),
        "auc": roc_auc_score(y, p),
        "pred_mean": float(np.mean(p)),
        "pred_min": float(np.min(p)),
        "pred_max": float(np.max(p)),
    }

def expected_calibration_error(y, p, n_bins=15):
    """
    Simple ECE: bin by predicted probability, compare avg predicted vs avg observed.
    """
    y = np.asarray(y)
    p = np.asarray(p)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if mask.sum() == 0:
            continue
        p_bin = p[mask].mean()
        y_bin = y[mask].mean()
        ece += (mask.mean()) * abs(y_bin - p_bin)
    return float(ece)

# ----------------------------
# 2) Reliability plot helper (matplotlib only)
# ----------------------------
def plot_reliability(y, preds_dict, n_bins=15):
    """
    preds_dict: {"Raw": p_raw, "Platt": p_platt, "Isotonic": p_iso, ...}
    """
    plt.figure(figsize=(7.5, 6))
    # perfect calibration line
    plt.plot([0, 1], [0, 1])

    for label, p in preds_dict.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        plt.plot(mean_pred, frac_pos, marker="o", linewidth=1)

    plt.xlabel("Mean predicted probability (bin)")
    plt.ylabel("Empirical positive rate (bin)")
    plt.title("Reliability diagram (OOF predictions)")
    plt.grid(True, alpha=0.25)
    plt.show()

# ----------------------------
# 3) Fold-safe calibration on OOF predictions (recommended)
#    We calibrate using only OOF preds (already out-of-sample),
#    and we evaluate calibrators on held-out splits of the OOF set.
# ----------------------------
def fold_safe_calibration(y, p, method="platt", n_splits=5, random_state=42):
    """
    method: "platt" (logistic regression) or "isotonic"
    Returns:
      - p_cal_oof: calibrated predictions for each training point (via calibrator-CV)
      - metrics_df: per-fold metrics
    """
    y = np.asarray(y)
    p = np.asarray(p)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    p_cal = np.zeros_like(p, dtype=float)
    rows = []

    for fold, (tr, va) in enumerate(skf.split(p.reshape(-1, 1), y), start=1):
        p_tr, y_tr = p[tr], y[tr]
        p_va, y_va = p[va], y[va]

        if method == "platt":
            # Platt scaling: logistic regression on the single feature p
            lr = LogisticRegression(solver="lbfgs", max_iter=1000)
            lr.fit(p_tr.reshape(-1, 1), y_tr)
            p_va_cal = lr.predict_proba(p_va.reshape(-1, 1))[:, 1]
        elif method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p_tr, y_tr)
            p_va_cal = iso.predict(p_va)
        else:
            raise ValueError("method must be 'platt' or 'isotonic'")

        p_cal[va] = p_va_cal

        m_raw = compute_metrics(y_va, p_va, name=f"raw_fold{fold}")
        m_cal = compute_metrics(y_va, p_va_cal, name=f"{method}_fold{fold}")
        rows.append({
            "fold": fold,
            "raw_logloss": m_raw["logloss"],
            "cal_logloss": m_cal["logloss"],
            "raw_brier": m_raw["brier"],
            "cal_brier": m_cal["brier"],
            "raw_auc": m_raw["auc"],
            "cal_auc": m_cal["auc"],
        })

    metrics_df = pd.DataFrame(rows)
    return p_cal, metrics_df

# Run fold-safe calibration for both methods
p_platt_oof, platt_cv = fold_safe_calibration(y_true, p_oof, method="platt", n_splits=5, random_state=42)
p_iso_oof,   iso_cv   = fold_safe_calibration(y_true, p_oof, method="isotonic", n_splits=5, random_state=42)

# ----------------------------
# 4) Summarize metrics (raw vs calibrated)
# ----------------------------
raw_metrics   = compute_metrics(y_true, p_oof,       name="raw_oof")
platt_metrics = compute_metrics(y_true, p_platt_oof, name="platt_oof_cv")
iso_metrics   = compute_metrics(y_true, p_iso_oof,   name="isotonic_oof_cv")

raw_metrics["ece"]   = expected_calibration_error(y_true, p_oof, n_bins=15)
platt_metrics["ece"] = expected_calibration_error(y_true, p_platt_oof, n_bins=15)
iso_metrics["ece"]   = expected_calibration_error(y_true, p_iso_oof, n_bins=15)

summary = pd.DataFrame([raw_metrics, platt_metrics, iso_metrics])
display(summary)

print("\nPer-fold calibration effect (Platt):")
display(platt_cv.describe().T[["mean", "std", "min", "max"]])

print("\nPer-fold calibration effect (Isotonic):")
display(iso_cv.describe().T[["mean", "std", "min", "max"]])

# ----------------------------
# 5) Reliability diagram (OOF)
# ----------------------------
plot_reliability(
    y_true,
    preds_dict={
        "Raw": p_oof,
        "Platt (CV on OOF)": p_platt_oof,
        "Isotonic (CV on OOF)": p_iso_oof,
    },
    n_bins=15
)

# ============================================================
# 6) Optional: Fit calibrator on FULL OOF and calibrate TEST
#    (Use this only for submission if calibration improves logloss on OOF-CV above.)
# ============================================================

# Platt on full OOF
lr_full = LogisticRegression(solver="lbfgs", max_iter=1000)
lr_full.fit(p_oof.reshape(-1, 1), y_true)
p_test_platt = lr_full.predict_proba(p_test.reshape(-1, 1))[:, 1]

# Isotonic on full OOF
iso_full = IsotonicRegression(out_of_bounds="clip")
iso_full.fit(p_oof, y_true)
p_test_iso = iso_full.predict(p_test)

# Create submissions (choose one)
sub_platt = pd.DataFrame({"id": df_test["id"].values, "diagnosed_diabetes": p_test_platt})
sub_iso   = pd.DataFrame({"id": df_test["id"].values, "diagnosed_diabetes": p_test_iso})

display(sub_platt.head())
display(sub_iso.head())

# Uncomment to save:
# sub_platt.to_csv("submission_platt.csv", index=False)
# sub_iso.to_csv("submission_isotonic.csv", index=False)
# print("Saved: submission_platt.csv and submission_isotonic.csv")



import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss

def slice_metrics(y_true, y_pred, mask):
    """Compute metrics for a boolean mask."""
    if mask.sum() == 0:
        return None
    return {
        "n": int(mask.sum()),
        "logloss": log_loss(y_true[mask], y_pred[mask]),
        "brier": brier_score_loss(y_true[mask], y_pred[mask]),
        "pred_mean": float(y_pred[mask].mean()),
        "empirical_rate": float(y_true[mask].mean()),
    }


def numeric_slice_report(df, y, p, feature, bins):
    """Slice report for a numeric feature."""
    rows = []
    values = df[feature].values

    for lo, hi in bins:
        mask = (values >= lo) & (values < hi)
        m = slice_metrics(y, p, mask)
        if m is None:
            continue
        rows.append({
            "feature": feature,
            "slice": f"[{lo}, {hi})",
            **m
        })

    return pd.DataFrame(rows)


def quantile_bins(series, n_bins=5):
    """Create quantile-based bins."""
    qs = np.linspace(0, 1, n_bins + 1)
    edges = np.unique(np.quantile(series, qs))
    return list(zip(edges[:-1], edges[1:]))




age_bins = quantile_bins(df_train["age"], n_bins=5)
age_report = numeric_slice_report(df_train, y, oof_pred, "age", age_bins)
display(age_report)



bmi_bins = [
    (0, 18.5),
    (18.5, 25),
    (25, 30),
    (30, 100)
]

bmi_report = numeric_slice_report(df_train, y, oof_pred, "bmi", bmi_bins)
display(bmi_report)



pa_bins = quantile_bins(df_train["physical_activity_minutes_per_week"], n_bins=5)
pa_report = numeric_slice_report(
    df_train, y, oof_pred,
    "physical_activity_minutes_per_week",
    pa_bins
)
display(pa_report)



def categorical_slice_report(df, y, p, feature, min_count=5000):
    rows = []
    for val, mask in df[feature].value_counts().items():
        idx = df[feature] == val
        if idx.sum() < min_count:
            continue
        m = slice_metrics(y, p, idx.values)
        if m is None:
            continue
        rows.append({
            "feature": feature,
            "category": val,
            **m
        })
    return pd.DataFrame(rows).sort_values("logloss")

# Example: education level
edu_report = categorical_slice_report(df_train, y, oof_pred, "education_level")
display(edu_report)

# Example: smoking status
smoking_report = categorical_slice_report(df_train, y, oof_pred, "smoking_status")
display(smoking_report)



# ============================================================
# New cell: Targeted feature addition + re-run CV (no changes to prior cells)
# Hypothesis-driven feature:
#   activity_per_bmi = physical_activity_minutes_per_week / (bmi + eps)
# Produces:
#   - new OOF predictions: oof_pred_v2
#   - new test predictions: test_pred_v2
#   - CV table + OOF summary
#   - submission file: submission_v2.csv
# ============================================================

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score

from catboost import CatBoostClassifier, Pool

RANDOM_STATE = 42
N_SPLITS = 5

target_col = "diagnosed_diabetes"
id_col = "id"

# ----------------------------
# 0) Build fresh feature matrices (do not overwrite X_train/X_test from earlier cells)
# ----------------------------
assert target_col in df_train.columns
assert target_col not in df_test.columns

y = df_train[target_col].astype(int).values

X_train_v2 = df_train.drop(columns=[target_col]).copy()
X_test_v2  = df_test.copy()

# Add interaction feature (interpretable, target-free)
eps = 1e-3
X_train_v2["activity_per_bmi"] = X_train_v2["physical_activity_minutes_per_week"] / (X_train_v2["bmi"] + eps)
X_test_v2["activity_per_bmi"]  = X_test_v2["physical_activity_minutes_per_week"] / (X_test_v2["bmi"] + eps)

# Detect categorical columns (unchanged)
cat_cols_v2 = X_train_v2.select_dtypes(include=["object", "category"]).columns.tolist()

print(f"Train v2 shape: {X_train_v2.shape} | Test v2 shape: {X_test_v2.shape}")
print(f"Categorical columns ({len(cat_cols_v2)}): {cat_cols_v2}")
print("Added feature: activity_per_bmi")

# ----------------------------
# 1) CV setup
# ----------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

oof_pred_v2 = np.zeros(len(X_train_v2), dtype=float)
test_pred_v2 = np.zeros(len(X_test_v2), dtype=float)

fold_rows_v2 = []

# ----------------------------
# 2) Model config (same as baseline for fair comparison)
# ----------------------------
model_params = dict(
    loss_function="Logloss",
    eval_metric="Logloss",
    iterations=4000,
    learning_rate=0.05,
    depth=8,
    l2_leaf_reg=3.0,
    random_seed=RANDOM_STATE,
    verbose=250,
    allow_writing_files=False,
    task_type="CPU",
)

def safe_log_loss(y_true, y_pred):
    # Version-proof stability (independent of sklearn log_loss signature)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return log_loss(y_true, y_pred)

# ----------------------------
# 3) CV loop
# ----------------------------
for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train_v2, y), start=1):
    X_tr, X_va = X_train_v2.iloc[tr_idx], X_train_v2.iloc[va_idx]
    y_tr, y_va = y[tr_idx], y[va_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_cols_v2)
    valid_pool = Pool(X_va, y_va, cat_features=cat_cols_v2)
    test_pool  = Pool(X_test_v2, cat_features=cat_cols_v2)

    model = CatBoostClassifier(**model_params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True, early_stopping_rounds=200)

    va_pred = model.predict_proba(valid_pool)[:, 1]
    oof_pred_v2[va_idx] = va_pred

    te_pred = model.predict_proba(test_pool)[:, 1]
    test_pred_v2 += te_pred / N_SPLITS

    fold_ll = safe_log_loss(y_va, va_pred)
    fold_auc = roc_auc_score(y_va, va_pred)

    best_iter = model.get_best_iteration()
    if best_iter is None:
        best_iter = model.tree_count_ - 1

    fold_rows_v2.append({
        "fold": fold,
        "n_train": len(tr_idx),
        "n_valid": len(va_idx),
        "pos_rate_train": float(y_tr.mean()),
        "pos_rate_valid": float(y_va.mean()),
        "logloss": fold_ll,
        "auc": fold_auc,
        "best_iteration": int(best_iter),
    })

    print(f"\nFold {fold}: logloss={fold_ll:.5f} | auc={fold_auc:.5f} | best_iter={int(best_iter)}")

cv_results_v2 = pd.DataFrame(fold_rows_v2)

# ----------------------------
# 4) Overall OOF metrics
# ----------------------------
oof_logloss_v2 = safe_log_loss(y, oof_pred_v2)
oof_auc_v2 = roc_auc_score(y, oof_pred_v2)

print("\n" + "="*80)
print("CV Results v2 (per fold)")
print("="*80)
display(cv_results_v2)

print("\n" + "="*80)
print("OOF Summary v2")
print("="*80)
print(f"OOF LogLoss: {oof_logloss_v2:.6f}")
print(f"OOF AUC:     {oof_auc_v2:.6f}")
print("\nPer-fold:")
print(f"LogLoss mean±std: {cv_results_v2['logloss'].mean():.6f} ± {cv_results_v2['logloss'].std():.6f}")
print(f"AUC     mean±std: {cv_results_v2['auc'].mean():.6f} ± {cv_results_v2['auc'].std():.6f}")

# ----------------------------
# 5) Submission file (v2)
# ----------------------------
sub_v2 = pd.DataFrame({
    id_col: df_test[id_col].values,
    target_col: test_pred_v2
})

display(sub_v2.head())

sub_v2.to_csv("/kaggle/working/submission.csv", index=False)
print("Saved: submission.csv --- version 2")


