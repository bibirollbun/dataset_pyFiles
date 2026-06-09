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


import os, gc, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
import lightgbm as lgb

warnings.filterwarnings("ignore")


TRAIN_PATH = "/kaggle/input/playground-series-s5e9/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e9/test.csv"
SAMPLE_SUB = "/kaggle/input/playground-series-s5e9/sample_submission.csv"

ID_COL = "id"
TARGET = "BeatsPerMinute"
RANDOM_STATE = 42
N_FOLDS = 10
USE_CLUSTER = True
N_CLUSTERS = 5


LGB_PARAMS = {
    "n_estimators": 20000,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "min_child_samples": 30,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.8,
    "objective": "regression",
    "metric": "rmse",
    "verbosity": -1,
    "n_jobs": -1,
    "random_state": RANDOM_STATE
}

EARLY_STOPPING_ROUNDS = 500
LOG_PERIOD = 0

def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred, squared=False)


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "TrackDurationMs" in df.columns:
        df["LogTrackDuration"] = np.log1p(df["TrackDurationMs"].astype(float))

    # Interactions
    df["mood_energy_interaction"] = df["MoodScore"] * df["Energy"]
    df["loudness_vocal_interaction"] = df["AudioLoudness"] * df["VocalContent"]
    df["acoustic_instrumental_ratio"] = df["AcousticQuality"] / (df["InstrumentalScore"] + 1e-6)
    df["Rhythm_Energy_Ratio"] = df["RhythmScore"] / (df["Energy"] + 1e-6)
    df["Energy_to_Loudness"] = df["Energy"] / (df["AudioLoudness"] + 1e-6)
    df["Rhythm_to_Duration"] = df["RhythmScore"] / (df["TrackDurationMs"] + 1e-6)
    df["MoodEnergyRhythm"] = df["MoodScore"] * df["Energy"] * df["RhythmScore"]

    # Polynomial features (degree 2 for selected vars)
    poly_cols = ["Energy", "RhythmScore", "MoodScore"]
    for c in poly_cols:
        if c in df.columns:
            df[f"{c}_sq"] = df[c].astype(float) ** 2

    # Null count
    df["null_count"] = df.isnull().sum(axis=1)

    return df


train = pd.read_csv(TRAIN_PATH)
test  = pd.read_csv(TEST_PATH)

print("Train shape:", train.shape, "| Test shape:", test.shape)

train = create_features(train)
test  = create_features(test)

# Candidate features
features = [c for c in train.columns if c not in [ID_COL, TARGET]]

# Drop single-value cols
single_cols = [c for c in features if train[c].nunique(dropna=False) <= 1]
if single_cols:
    print("Dropping single-value columns:", single_cols)
    train.drop(columns=single_cols, inplace=True, errors="ignore")
    test.drop(columns=single_cols, inplace=True, errors="ignore")
    features = [c for c in train.columns if c not in [ID_COL, TARGET]]

# Ensure test has same features
for c in features:
    if c not in test.columns:
        test[c] = train[c].median()


imputer = SimpleImputer(strategy="median")
scaler = StandardScaler()

X = train[features].copy()
y = train[TARGET].values
X_test = test[features].copy()

X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)

X_scaled = pd.DataFrame(scaler.fit_transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test_imputed), columns=X_test_imputed.columns, index=X_test_imputed.index)

# Optional cluster feature
if USE_CLUSTER:
    combined = pd.concat([X_scaled, X_test_scaled], axis=0)
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(combined)
    X_scaled["cluster"] = clusters[:len(X_scaled)]
    X_test_scaled["cluster"] = clusters[len(X_scaled):]

features_final = X_scaled.columns.tolist()
print("Final feature count:", len(features_final))

X_arr = X_scaled[features_final].values
X_test_arr = X_test_scaled[features_final].values


bins = pd.qcut(y, q=10, duplicates="drop")
kf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros(X_arr.shape[0])
test_preds = np.zeros(X_test_arr.shape[0])
models = []
best_iterations = []
fold_metrics = []

print(f"\nStarting {N_FOLDS}-fold Stratified CV (LGBM)...")
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_arr, bins.codes), 1):
    print(f"Fold {fold}/{N_FOLDS}")
    X_tr, X_val = X_arr[tr_idx], X_arr[val_idx]
    y_tr, y_val = y[tr_idx], y[val_idx]

    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False),
            lgb.log_evaluation(period=LOG_PERIOD)
        ]
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration_)
    test_fold_pred = model.predict(X_test_arr, num_iteration=model.best_iteration_)

    oof_preds[val_idx] = val_pred
    test_preds += test_fold_pred / N_FOLDS

    models.append(model)
    best_iterations.append(model.best_iteration_ if model.best_iteration_ is not None else LGB_PARAMS["n_estimators"])

    # Metrics
    fold_rmse = rmse(y_val, val_pred)
    fold_mae = mean_absolute_error(y_val, val_pred)
    fold_r2 = r2_score(y_val, val_pred)
    fold_metrics.append({"fold": fold, "rmse": fold_rmse, "mae": fold_mae, "r2": fold_r2, "best_iter": model.best_iteration_})
    print(f" Fold {fold} RMSE={fold_rmse:.6f}, MAE={fold_mae:.6f}, R2={fold_r2:.6f}, BestIter={model.best_iteration_}")



oof_rmse = rmse(y, oof_preds)
oof_mae = mean_absolute_error(y, oof_preds)
oof_r2 = r2_score(y, oof_preds)

print("\nOOF results:")
print(f" OOF RMSE: {oof_rmse:.6f}")
print(f" OOF MAE : {oof_mae:.6f}")
print(f" OOF R2  : {oof_r2:.6f}")
print(" Fold summary:")
print(pd.DataFrame(fold_metrics))


oof_df = pd.DataFrame({ID_COL: train[ID_COL], "actual": y, "predicted": oof_preds})
oof_df.to_csv("oof_predictions.csv", index=False)
print("Saved oof_predictions.csv")

# Clip predictions to realistic BPM range
test_preds = np.clip(test_preds, 40, 250)

sub_cv = pd.DataFrame({ID_COL: test[ID_COL], TARGET: test_preds})
sub_cv.to_csv("submission_lgb_cv.csv", index=False)
print("Saved submission_lgb_cv.csv (CV-averaged predictions)")



oof_rmse = rmse(y, oof_preds)
oof_mae = mean_absolute_error(y, oof_preds)
oof_r2 = r2_score(y, oof_preds)

print("\nOOF results:")
print(f" OOF RMSE: {oof_rmse:.6f}")
print(f" OOF MAE : {oof_mae:.6f}")
print(f" OOF R2  : {oof_r2:.6f}")
print(" Fold summary:")
print(pd.DataFrame(fold_metrics))

# Save OOF predictions
oof_df = pd.DataFrame({ID_COL: train[ID_COL], "actual": y, "predicted": oof_preds})
oof_df.to_csv("oof_predictions.csv", index=False)
print("Saved oof_predictions.csv")


avg_best_iter = int(np.mean([it for it in best_iterations if it is not None]) * 1.1)
if avg_best_iter < 10:
    avg_best_iter = LGB_PARAMS["n_estimators"]

print(f"Training final LGBM on full data with n_estimators={avg_best_iter} ...")
final = lgb.LGBMRegressor(**{**LGB_PARAMS, "n_estimators": avg_best_iter})
final.fit(X_arr, y)
final_test_pred = final.predict(X_test_arr)
final_test_pred = np.clip(final_test_pred, 40, 250)

sub_full = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final_test_pred})
sub_full.to_csv("submission_lgb_full.csv", index=False)
print("Saved submission_lgb_full.csv (full-data model predictions)")


blend_pred = 0.7 * test_preds + 0.3 * final_test_pred
sub_blend = pd.DataFrame({ID_COL: test[ID_COL], TARGET: blend_pred})
sub_blend.to_csv("submission_lgb_blend.csv", index=False)
print("Saved submission_lgb_blend.csv (blended predictions)")


fi_dfs = []
for i, m in enumerate(models):
    fi = pd.DataFrame({"feature": features_final, "importance": m.feature_importances_, "fold": i+1})
    fi_dfs.append(fi)
fi_df = pd.concat(fi_dfs).groupby("feature").agg({"importance":["mean","sum"]})
fi_df.columns = ["importance_mean", "importance_sum"]
fi_df = fi_df.reset_index().sort_values("importance_mean", ascending=False)
fi_df.to_csv("feature_importance_lgb_cv.csv", index=False)
print("Saved feature_importance_lgb_cv.csv")



print("\n--- DONE ---")
print(f"OOF RMSE: {oof_rmse:.6f}")
print("Files created: oof_predictions.csv, submission_lgb_cv.csv, submission_lgb_full.csv, submission_lgb_blend.csv, feature_importance_lgb_cv.csv")

