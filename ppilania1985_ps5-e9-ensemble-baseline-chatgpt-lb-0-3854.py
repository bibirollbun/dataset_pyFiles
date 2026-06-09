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


import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.isotonic import IsotonicRegression

# --------------------
# Load data
# --------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
sample_sub = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
target_col = sample_sub.columns[-1]
y_true = train[target_col].values

# --------------------
# Define ensemble OOF + submission files
# --------------------
ensemble_files = {
    # Random Forest models (use y_pred)
    "fe_rf": {
        "oof": "/kaggle/input/ps5e9-ensemble/oof_fe_rf.csv",
        "sub": "/kaggle/input/ps5e9-ensemble/submission_fe_rf.csv",
        "col": "y_pred",
    },
    "best": {
        "oof": "/kaggle/input/ps5e9-ensemble/oof_best.csv",
        "sub": "/kaggle/input/ps5e9-ensemble/submission_best.csv",
        "col": "y_pred",
    },
    # Gradient boosting & other models (use oof_pred)
    "fe_gbm": {
        "oof": "/kaggle/input/ps5e9-ensemble/oof_FE_gbm.csv",
        "sub": "/kaggle/input/ps5e9-ensemble/submission_cat.csv",
        "col": "oof_pred",
    },
    "fe_gbm_v16": {
        "oof": "/kaggle/input/ps5e9-ensemble/oof_fe_gbm_v16.csv",
        "sub": "/kaggle/input/ps5e9-ensemble/submission_fe_gbm_v16.csv",
        "col": "oof_pred",
    },
    "fe_gbm_v15": {
        "oof": "/kaggle/input/ps5e9-ensemble/oof_fe_v15.csv",
        "sub": "/kaggle/input/ps5e9-ensemble/submission_fe_gbm_v15.csv",
        "col": "oof_pred",
    },
    "xgb": {
        "oof": "/kaggle/input/ps5e9-ensemble/oof_xgb.csv",
        "sub": "/kaggle/input/ps5e9-ensemble/submission_xgb.csv",
        "col": "oof_pred",
    },
    "lgb": {
        "oof": "/kaggle/input/ps5e9-ensemble/oof_lgb.csv",
        "sub": "/kaggle/input/ps5e9-ensemble/submission_lgb.csv",
        "col": "oof_pred",
    },
    "fe_xgb_v27": {
        "oof": "/kaggle/input/ps5e9-ensemble/oof_fe_xgb_v27.csv",
        "sub": "/kaggle/input/ps5e9-ensemble/submission_fe_xgb_v27.csv",
        "col": "oof_pred",
    },
    "fe_xgb": {
        "oof": "/kaggle/input/ps5e9-ensemble/oof_fe_xgb.csv",
        "sub": "/kaggle/input/ps5e9-ensemble/submission_fe_xgb.csv",
        "col": "oof_pred",
    },
    "fe_lgb": {
        "oof": "/kaggle/input/ps5e9-ensemble/oof_fe_v15.csv",
        "sub": "/kaggle/input/ps5e9-ensemble/submission_fe_lgb.csv",
        "col": "oof_pred",
    },
    "cat": {
        "oof": "/kaggle/input/ps5e9-ensemble/oof_cat.csv",
        "sub": "/kaggle/input/ps5e9-ensemble/submission_cat.csv",
        "col": "oof_pred",
    },
}

# --------------------
# Load OOF preds + submissions
# --------------------
model_names, oof_preds, subs = [], [], []

for name, paths in ensemble_files.items():
    df_oof = pd.read_csv(paths["oof"])
    df_sub = pd.read_csv(paths["sub"])
    oof_preds.append(df_oof[paths["col"]].values)
    subs.append(df_sub.iloc[:, 1].values)
    model_names.append(name)

oof_preds = np.vstack(oof_preds).T
subs = np.vstack(subs).T

# --------------------
# Helpers
# --------------------
def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)

results = {}
final_submissions = {}

# --------------------
# 1. Simple Average
# --------------------
avg_pred = np.mean(oof_preds, axis=1)
results["Simple Avg"] = rmse(y_true, avg_pred)
final_submissions["Simple Avg"] = np.mean(subs, axis=1)

# --------------------
# 2. Weighted Average (random search over Dirichlet weights)
# --------------------
best_rmse, best_w = 1e9, None
for _ in range(2000):
    w = np.random.dirichlet(np.ones(oof_preds.shape[1]), size=1)
    blend = np.dot(oof_preds, w.T).ravel()
    score = rmse(y_true, blend)
    if score < best_rmse:
        best_rmse, best_w = score, w
results["Weighted Avg"] = best_rmse
best_weights = best_w
final_submissions["Weighted Avg"] = np.dot(subs, best_weights.T).ravel()

# --------------------
# 3. RidgeCV
# --------------------
ridge = RidgeCV(alphas=np.logspace(-6, 3, 20), cv=5)
ridge.fit(oof_preds, y_true)
results["RidgeCV"] = rmse(y_true, ridge.predict(oof_preds))
final_submissions["RidgeCV"] = ridge.predict(subs)

# --------------------
# 4. Linear Regression
# --------------------
linreg = LinearRegression()
linreg.fit(oof_preds, y_true)
results["Linear Regression"] = rmse(y_true, linreg.predict(oof_preds))
final_submissions["Linear Regression"] = linreg.predict(subs)

# --------------------
# 5. Hill Climbing Ensemble
# --------------------
remaining = list(range(oof_preds.shape[1]))
selected, current_pred = [], np.zeros_like(y_true)
best_score = 1e9

while remaining:
    improved = False
    for i in remaining:
        candidate = (current_pred * len(selected) + oof_preds[:, i]) / (len(selected) + 1)
        score = rmse(y_true, candidate)
        if score < best_score:
            best_score = score
            best_model = i
            best_candidate = candidate
            improved = True
    if improved:
        selected.append(best_model)
        remaining.remove(best_model)
        current_pred = best_candidate
    else:
        break
results["Hill Climb"] = best_score
selected_models = [model_names[i] for i in selected]
final_submissions["Hill Climb"] = np.mean(subs[:, [model_names.index(m) for m in selected_models]], axis=1)

# --------------------
# Calibration with Isotonic Regression
# --------------------
for method, preds in list(final_submissions.items()):
    # Fit isotonic on OOF
    if method == "Weighted Avg":
        oof_pred = np.dot(oof_preds, best_weights.T).ravel()
    elif method == "RidgeCV":
        oof_pred = ridge.predict(oof_preds)
    elif method == "Linear Regression":
        oof_pred = linreg.predict(oof_preds)
    elif method == "Hill Climb":
        idxs = [model_names.index(m) for m in selected_models]
        oof_pred = np.mean(oof_preds[:, idxs], axis=1)
    else:  # Simple Avg
        oof_pred = np.mean(oof_preds, axis=1)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof_pred, y_true)

    calibrated_sub = iso.predict(preds)
    calibrated_rmse = rmse(y_true, iso.predict(oof_pred))

    results[f"{method} + Isotonic"] = calibrated_rmse
    final_submissions[f"{method} + Isotonic"] = calibrated_sub

# --------------------
# Print results
# --------------------
print("\nğŸ”¹ Ensemble Leaderboard (OOF RMSE):")
for k, v in sorted(results.items(), key=lambda x: x[1]):
    print(f"{k:>25}: {v:.6f}")

print("\nâœ… Best weights (Weighted Avg):", dict(zip(model_names, best_weights.ravel())))
print("âœ… Selected models (Hill Climb):", selected_models)

# --------------------
# Save ALL submissions
# --------------------
for method, preds in final_submissions.items():
    submission = sample_sub.copy()
    submission[target_col] = preds
    filename = f"submission_{method.replace(' ', '_').lower()}.csv"
    submission.to_csv(filename, index=False)
    print(f"ğŸ’¾ Saved {filename}")

