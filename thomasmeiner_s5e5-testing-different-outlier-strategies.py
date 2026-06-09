"""
Outlier Experiments + Final Inference Pipelines (Duration‑only winning features)
--------------------------------------------------------------------------
This single script contains:
1. Exhaustive subset experiments (unchanged).
2. **Final inference pipelines** adjusted so that:
   • IQR‑append uses only the ``Duration`` column to compute the outlier score.
   • Isolation‑Forest‑append uses only the ``Duration`` column to compute the outlier score.
   • Other previous pipelines (LOF removal & do‑nothing baseline) are preserved.
"""

import itertools
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_log_error
from scipy import stats

# ---------------------------------------------------------------------
# 1. EXPERIMENT SECTION  (identical to previous version)
# ---------------------------------------------------------------------
train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv').drop('id', axis=1)
X = train.drop(columns=['Calories'])
y = train['Calories']
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# --- helper functions (remove_outliers, compute_scores, cv_removal, cv_append) ---
# ... [unchanged code omitted for brevity; same as previous version] ...

# ---------------------------------------------------------------------
# 2. FINAL INFERENCE PIPELINES (UPDATED)
# ---------------------------------------------------------------------
print("\n================  FINAL INFERENCE PIPELINES  ================")

# --------------------------
# Common data prep
# --------------------------
train_full = train.copy()
train_full['Sex'] = train_full['Sex'].map({'male': 0, 'female': 1})

test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv').drop('id', axis=1)
test['Sex'] = test['Sex'].map({'male': 0, 'female': 1})

sample_sub = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')
TARGET = 'Calories'
X_train_full = train_full.drop(columns=[TARGET])
y_train_full = train_full[TARGET]
X_test_full = test.copy()

# Standard scaler fit on ALL numeric features for scoring purposes
scaler_full = StandardScaler().fit(X_train_full)

# --------------------------
# Utility: IQR score for a *single* column Series
# --------------------------
def iqr_score_series(s: pd.Series, factor: float = 1.5) -> pd.Series:
    q1 = s.quantile(0.25)
    q3 = s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - factor * iqr
    upper = q3 + factor * iqr
    # Severity: distance outside bounds divided by IQR; 0 inside bounds
    lower_dev = (lower - s).clip(lower=0)
    upper_dev = (s - upper).clip(lower=0)
    severity = (lower_dev + upper_dev) / (iqr + 1e-9)
    return severity

# --------------------------
# Pipeline: Duration‑only IQR APPEND (winner)
# --------------------------

def run_duration_iqr_append_pipeline():
    print("\n▶ Building Duration‑only IQR‑append model …")

    # Scale numeric features (needed only for score, not CatBoost)
    X_train_scaled = pd.DataFrame(scaler_full.transform(X_train_full),
                                  index=X_train_full.index,
                                  columns=X_train_full.columns)
    X_test_scaled = pd.DataFrame(scaler_full.transform(X_test_full),
                                 index=X_test_full.index,
                                 columns=X_test_full.columns)

    # Append IQR score *computed on Duration only*
    X_train_aug = X_train_full.copy()
    X_test_aug = X_test_full.copy()

    X_train_aug['iqr_duration'] = iqr_score_series(X_train_scaled['Duration'])
    X_test_aug['iqr_duration'] = iqr_score_series(X_test_scaled['Duration'])

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42,
        loss_function='RMSE'
    )
    model.fit(X_train_aug, np.log1p(y_train_full))

    preds = np.expm1(model.predict(X_test_aug))
    sub = sample_sub.copy()
    sub[TARGET] = np.clip(preds, 0, None)
    sub.to_csv('submission_duration_iqr.csv', index=False)
    print("✅ Saved 'submission_duration_iqr.csv'")

# --------------------------
# Pipeline: Duration‑only IsolationForest APPEND (runner‑up)
# --------------------------

def run_duration_iforest_append_pipeline():
    print("\n▶ Building Duration‑only IsolationForest‑append model …")

    # Prepare Duration as 2‑D array for IsolationForest
    dur_train = X_train_full[['Duration']].to_numpy()
    dur_test = X_test_full[['Duration']].to_numpy()

    iso = IsolationForest(contamination=0.01, random_state=42)
    iso.fit(dur_train)
    train_score = -iso.decision_function(dur_train)
    test_score = -iso.decision_function(dur_test)

    X_train_aug = X_train_full.copy()
    X_test_aug = X_test_full.copy()

    X_train_aug['iforest_duration'] = train_score
    X_test_aug['iforest_duration'] = test_score

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42,
        loss_function='RMSE'
    )
    model.fit(X_train_aug, np.log1p(y_train_full))

    preds = np.expm1(model.predict(X_test_aug))
    sub = sample_sub.copy()
    sub[TARGET] = np.clip(preds, 0, None)
    sub.to_csv('submission_duration_iforest.csv', index=False)
    print("✅ Saved 'submission_duration_iforest.csv'")

# --------------------------
# Pipeline: LOF removal (unchanged baseline winner in earlier NB)
# --------------------------

def run_lof_removal_pipeline():
    print("\n▶ Building LOF‑removal model …")
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_train_full),
                            index=X_train_full.index,
                            columns=X_train_full.columns)
    lof = LocalOutlierFactor(n_neighbors=20, contamination=0.01)
    mask = lof.fit_predict(X_scaled.to_numpy()) == 1

    X_train_clean = X_train_full.loc[mask]
    y_train_clean = y_train_full.loc[mask]

    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42,
        loss_function='RMSE'
    )
    model.fit(X_train_clean, np.log1p(y_train_clean))

    preds = np.expm1(model.predict(X_test_full))
    sub = sample_sub.copy()
    sub[TARGET] = np.clip(preds, 0, None)
    sub.to_csv('submission_lof_removal.csv', index=False)
    print("✅ Saved 'submission_lof_removal.csv'")

# --------------------------
# Pipeline: Do‑nothing baseline
# --------------------------

def run_do_nothing_pipeline():
    print("\n▶ Building baseline CatBoost model …")
    model = CatBoostRegressor(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        verbose=0,
        random_state=42,
        loss_function='RMSE'
    )
    model.fit(X_train_full, np.log1p(y_train_full))
    preds = np.expm1(model.predict(X_test_full))
    sub = sample_sub.copy()
    sub[TARGET] = np.clip(preds, 0, None)
    sub.to_csv('submission_do_nothing.csv', index=False)
    print("✅ Saved 'submission_do_nothing.csv')")

# --------------------------
# Execute desired pipelines
# --------------------------
if __name__ == "__main__":
    run_duration_iqr_append_pipeline()
    run_duration_iforest_append_pipeline()
    # Uncomment to run others
    # run_lof_removal_pipeline()
    # run_do_nothing_pipeline()

