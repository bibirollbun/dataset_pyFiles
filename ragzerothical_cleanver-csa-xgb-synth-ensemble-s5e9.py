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


# enhanced_csa_xgb_synth_ensemble.py
import numpy as np
import pandas as pd
import math
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor, XGBClassifier
import lightgbm as lgb
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt

# -----------------------
# Config
# -----------------------
TRAIN_CSV = "/kaggle/input/playground-series-s5e9/train.csv"
TEST_CSV = "/kaggle/input/playground-series-s5e9/test.csv"
OUTPUT_CSV = "submission_output.csv"

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Synthetic multipliers
SYN_MUL_LOW  = 3
SYN_MUL_MID  = 1.0  # slightly more mid-range
SYN_MUL_HIGH = 3

# Ensemble / training params
N_ENSEMBLE_XGB = 3
N_ENSEMBLE_LGB = 2
N_ENSEMBLE_CAT = 2

XGB_PARAMS = {
    "n_estimators": 800,
    "max_depth": 9,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "reg_alpha": 0.2,
    "tree_method": "hist",
    "device": "cuda",
    "n_jobs": -1,
    "random_state": RANDOM_STATE
}

LGB_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "reg_alpha": 0.2,
    "reg_lambda": 1.0,
    "random_state": RANDOM_STATE,
    "n_jobs": -1
}

CAT_PARAMS = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 8,
    "l2_leaf_reg": 3,
    "loss_function": "RMSE",
    "random_seed": RANDOM_STATE,
    "verbose": 0
}

LOW_TH = 100
HIGH_TH = 140

# -----------------------
# Feature Engineering
# -----------------------
def add_features(df):
    df = df.copy()
    eps = 1e-6

    # -------------------------
    # Original features
    # -------------------------
    df["Amplitude"] = 10 ** (df["AudioLoudness"] / 20)
    df["Frequency_Hz"] = (df["RhythmScore"] * df["Energy"]) / (df["TrackDurationMs"].replace(0, eps)/1000 + eps)

    # Polynomial & log transforms
    for col in ["VocalContent", "Energy", "AcousticQuality", "TrackDurationMs", "AudioLoudness"]:
        if col in df.columns:
            df[f"{col}^2"] = df[col] ** 2
            df[f"{col}^3"] = df[col] ** 3
            df[f"log_{col}"] = np.log1p(np.abs(df[col])) * np.sign(df[col])

    # Interactions (original ones)
    interactions = [
        ("VocalContent", "Energy"),
        ("AcousticQuality", "MoodScore"),
        ("TrackDurationMs", "MoodScore"),
        ("Energy", "AcousticQuality"),
        ("VocalContent", "MoodScore"),
        ("Amplitude", "Energy")
    ]
    for a,b in interactions:
        if a in df.columns and b in df.columns:
            df[f"{a}_x_{b}"] = df[a] * df[b]

    # -------------------------
    # Extra engineered features (cover missing top-10)
    # -------------------------

    # Rhythm-based
    if ("RhythmScore" in df.columns) and ("Energy" in df.columns):
        df["Rhythm_Energy"] = df["RhythmScore"] * df["Energy"]
    if ("RhythmScore" in df.columns) and ("AudioLoudness" in df.columns):
        df["Rhythm_Loudness"] = df["RhythmScore"] * df["AudioLoudness"]

    # Duration-related
    if "TrackDurationMs" in df.columns:
        df["Duration_Minutes"] = df["TrackDurationMs"] / 60000.0
        df["Log_Duration"] = np.log1p(df["TrackDurationMs"])

    if ("TrackDurationMs" in df.columns) and ("Energy" in df.columns):
        df["Duration_Energy_Ratio"] = df["TrackDurationMs"] / (df["Energy"] * 10000 + 1)

    # Non-linear (some overlap with above but safe to add)
    if "RhythmScore" in df.columns:
        df["RhythmScore_Squared"] = df["RhythmScore"] ** 2
    if "Energy" in df.columns:
        df["Energy_Squared"] = df["Energy"] ** 2

    # Musical character
    if "AcousticQuality" in df.columns and "InstrumentalScore" in df.columns:
        df["Acoustic_Instrumental_Ratio"] = df["AcousticQuality"] / (df["InstrumentalScore"] + 0.01)
    if "VocalContent" in df.columns and "Energy" in df.columns:
        df["Vocal_Energy"] = df["VocalContent"] * df["Energy"]

    # Performance & mood
    if "LivePerformanceLikelihood" in df.columns and "Energy" in df.columns:
        df["Live_Energy"] = df["LivePerformanceLikelihood"] * df["Energy"]
    if "MoodScore" in df.columns and "RhythmScore" in df.columns:
        df["Mood_Rhythm"] = df["MoodScore"] * df["RhythmScore"]

    # Composite metrics
    if "Energy" in df.columns and "AudioLoudness" in df.columns:
        df["Audio_Intensity"] = (df["Energy"] * np.abs(df["AudioLoudness"])) / 10
        df["Energy_Loudness_Ratio"] = df["Energy"] / (np.abs(df["AudioLoudness"]) + 0.01)

    if "LivePerformanceLikelihood" in df.columns and "MoodScore" in df.columns:
        df["Performance_Character"] = (df["LivePerformanceLikelihood"] + df["MoodScore"]) / 2

    # Ratios
    if "RhythmScore" in df.columns and "Duration_Minutes" in df.columns:
        df["Rhythm_Duration_Density"] = df["RhythmScore"] / (df["Duration_Minutes"] + 1e-6)

    return df


# Mixup + Gaussian noise
def mixup_augment_range(X_range, y_range, n_new, alpha=0.2):
    if n_new <= 0 or len(X_range) < 2:
        return pd.DataFrame(columns=X_range.columns), pd.Series([], dtype=float)
    i_idx = np.random.randint(0, len(X_range), size=n_new)
    j_idx = np.random.randint(0, len(X_range), size=n_new)
    lam = np.random.beta(alpha, alpha, size=n_new).reshape(-1,1)
    X_arr = X_range.values
    X_new = lam * X_arr[i_idx] + (1 - lam) * X_arr[j_idx]
    # add small Gaussian noise (per-column std)
    col_std = np.std(X_arr, axis=0, ddof=0)
    noise = np.random.normal(0, 0.01 * (col_std + 1e-6), X_new.shape)
    X_new += noise
    y_new = (lam.flatten() * y_range.values[i_idx] + (1 - lam.flatten()) * y_range.values[j_idx])
    X_new_df = pd.DataFrame(X_new, columns=X_range.columns)
    y_new_sr = pd.Series(y_new)
    return X_new_df, y_new_sr


def gen_count(orig_count, multiplier):
    # ensure integer and non-negative
    return int(round(orig_count * max(0.0, multiplier)))


# -----------------------
# Read data, features, outlier remove
# -----------------------
df = pd.read_csv(TRAIN_CSV)
df = add_features(df)

# keep only columns that exist and are safe for modeling
features = [c for c in df.columns if c not in ["id", "BeatsPerMinute"]]
X = df[features].copy()
y = df["BeatsPerMinute"].copy()

# outlier removal (1% tails)
low_val, high_val = np.percentile(y, [1, 99])
mask = (y >= low_val) & (y <= high_val)
X = X[mask].reset_index(drop=True)
y = y[mask].reset_index(drop=True)

print("Total samples after outlier removal:", len(y))
print("Feature count:", len(features))

# -----------------------
# Split original data FIRST (real validation only)
# -----------------------
X_train_orig, X_val, y_train_orig, y_val = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)
print("Original train size:", len(y_train_orig), "Validation size (original):", len(y_val))

# -----------------------
# Augment ONLY the training split (by range)
# -----------------------
low_mask  = y_train_orig <= LOW_TH
mid_mask  = (y_train_orig > LOW_TH) & (y_train_orig <= HIGH_TH)
high_mask = y_train_orig > HIGH_TH

synth_parts_X = []
synth_parts_y = []

n_low_new  = gen_count(low_mask.sum(), SYN_MUL_LOW)
n_mid_new  = gen_count(mid_mask.sum(), SYN_MUL_MID)
n_high_new = gen_count(high_mask.sum(), SYN_MUL_HIGH)
print("Synthetic targets (low, mid, high):", n_low_new, n_mid_new, n_high_new)

if low_mask.sum() > 1 and n_low_new > 0:
    X_low = X_train_orig[low_mask].reset_index(drop=True)
    y_low = y_train_orig[low_mask].reset_index(drop=True)
    X_syn_low, y_syn_low = mixup_augment_range(X_low, y_low, n_low_new, alpha=0.3)
    synth_parts_X.append(X_syn_low); synth_parts_y.append(y_syn_low)

if mid_mask.sum() > 1 and n_mid_new > 0:
    X_mid = X_train_orig[mid_mask].reset_index(drop=True)
    y_mid = y_train_orig[mid_mask].reset_index(drop=True)
    X_syn_mid, y_syn_mid = mixup_augment_range(X_mid, y_mid, n_mid_new, alpha=0.1)
    synth_parts_X.append(X_syn_mid); synth_parts_y.append(y_syn_mid)

if high_mask.sum() > 1 and n_high_new > 0:
    X_high = X_train_orig[high_mask].reset_index(drop=True)
    y_high = y_train_orig[high_mask].reset_index(drop=True)
    X_syn_high, y_syn_high = mixup_augment_range(X_high, y_high, n_high_new, alpha=0.3)
    synth_parts_X.append(X_syn_high); synth_parts_y.append(y_syn_high)

# Combine original training + synthetic
if synth_parts_X:
    X_train = pd.concat([X_train_orig] + synth_parts_X, ignore_index=True)
    y_train = pd.concat([y_train_orig] + synth_parts_y, ignore_index=True)
else:
    X_train = X_train_orig.copy()
    y_train = y_train_orig.copy()

print("Training size after augmentation:", len(y_train))

# -----------------------
# Domain classifier for CSA weighting
# (train vs val: zeros for train, ones for val)
# -----------------------
X_domain = pd.concat([X_train, X_val], ignore_index=True)
y_domain = np.concatenate([np.zeros(len(X_train)), np.ones(len(X_val))])

domain_clf = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    tree_method="hist",
    device="cuda",
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1
)
print("Training domain classifier (train vs val)...")
domain_clf.fit(X_domain, y_domain)

# compute importance weights for training samples (on augmented train)
p_val_proba = domain_clf.predict_proba(X_train)[:, 1]  # probability of being validation-like
p_train_proba = 1 - p_val_proba
sample_weights = p_val_proba / (p_train_proba + 1e-6)
# smooth & clip weights to avoid extremes
sample_weights = 50.0 / (1.0 + np.exp(-sample_weights))
sample_weights = np.clip(sample_weights, 1e-3, 50.0)
print("Sample-weights summary — min/max/mean:", sample_weights.min(), sample_weights.max(), sample_weights.mean())

# -----------------------
# Train ensemble of regressors
# -----------------------
ensemble_models = []
print(f"Training ensemble: XGB x{N_ENSEMBLE_XGB}, LGB x{N_ENSEMBLE_LGB}, CAT x{N_ENSEMBLE_CAT} ...")

# XGB
for i in range(N_ENSEMBLE_XGB):
    params = XGB_PARAMS.copy(); params["random_state"] = RANDOM_STATE + i*7
    model = XGBRegressor(**params)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    ensemble_models.append(model)
    print(f" - trained XGB model {i+1}")

# LightGBM
for i in range(N_ENSEMBLE_LGB):
    model = lgb.LGBMRegressor(**LGB_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    ensemble_models.append(model)
    print(f" - trained LGB model {i+1}")

# CatBoost
for i in range(N_ENSEMBLE_CAT):
    model = CatBoostRegressor(**CAT_PARAMS)
    model.fit(X_train, y_train, sample_weight=sample_weights)
    ensemble_models.append(model)
    print(f" - trained CatBoost model {i+1}")

def ensemble_predict(models, Xq):
    preds = np.column_stack([m.predict(Xq) for m in models])
    return preds.mean(axis=1)

# -----------------------
# Residual correction for low/mid/high using original training examples only
# (we compute residuals on TRAIN ORIG portion to avoid synthetic residual artifacts)
# -----------------------
# compute predictions for original training rows only
# find indices in X_train that correspond to the original training records:
# The first len(X_train_orig) rows in our concatenation are the original train rows
n_orig_train = len(X_train_orig)
y_train_pred_on_orig = ensemble_predict(ensemble_models, X_train.iloc[:n_orig_train])
residuals_orig = y_train_orig.reset_index(drop=True) - y_train_pred_on_orig

resid_models = {}
# build masks based on y_train_orig
for name, mask_cond in [("low", y_train_orig <= LOW_TH),
                        ("mid", (y_train_orig > LOW_TH) & (y_train_orig <= HIGH_TH)),
                        ("high", y_train_orig > HIGH_TH)]:
    idxs = np.where(mask_cond)[0]
    if len(idxs) >= 10:
        model = Ridge(alpha=1.0)
        model.fit(X_train_orig.iloc[idxs], residuals_orig.iloc[idxs])
        resid_models[name] = model
        print(f" - trained residual model for '{name}' (n={len(idxs)})")
    else:
        print(f" - skipping residual model for '{name}' (n={len(idxs)})")

# -----------------------
# Validation evaluation (validation set is original-only)
# -----------------------
y_val_pred = ensemble_predict(ensemble_models, X_val)
y_val_final = y_val_pred.copy()

# apply residual corrections using resid_models if available (use resid models trained on original-train)
for name, cond in [("low", y_val <= LOW_TH),
                   ("mid", (y_val > LOW_TH) & (y_val <= HIGH_TH)),
                   ("high", y_val > HIGH_TH)]:
    mask_idx = np.where(cond)[0]
    if name in resid_models and mask_idx.size > 0:
        y_val_final[mask_idx] += resid_models[name].predict(X_val.iloc[mask_idx])

val_rmse = math.sqrt(mean_squared_error(y_val, y_val_final))
print("Validation RMSE after residual correction (on original val):", val_rmse)

# -----------------------
# Save models & artifacts
# -----------------------
joblib.dump(domain_clf, "domain_clf_synth_enhanced.pkl")
joblib.dump(ensemble_models, "ensemble_models_enhanced.pkl")
for name, model in resid_models.items():
    joblib.dump(model, f"resid_{name}_ridge.pkl")
print("Saved models: domain_clf_synth_enhanced.pkl, ensemble_models_enhanced.pkl, resid_*.pkl")

# -----------------------
# Test set prediction
# -----------------------
df_test = pd.read_csv(TEST_CSV)
df_test = add_features(df_test)
X_test = df_test[features].copy()
# align columns (fill missing)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

test_preds = ensemble_predict(ensemble_models, X_test)

# apply residual correction using the resid_models (based on predicted range)
for name, cond in [("low", test_preds <= LOW_TH),
                   ("mid", (test_preds > LOW_TH) & (test_preds <= HIGH_TH)),
                   ("high", test_preds > HIGH_TH)]:
    idxs = np.where(cond)[0]
    if name in resid_models and idxs.size > 0:
        test_preds[idxs] += resid_models[name].predict(X_test.iloc[idxs])

# Clip to valid BPM range (tunable)
test_preds = np.clip(test_preds, 60, 200)

submission = pd.DataFrame({"id": df_test["id"], "BPM": test_preds})
submission.to_csv(OUTPUT_CSV, index=False)
print("Submission saved to", OUTPUT_CSV)

# -----------------------
# Optional: plot validation
# -----------------------
plt.figure(figsize=(8,6))
plt.scatter(y_val, y_val_final, alpha=0.6, edgecolor="k")
plt.plot([y_val.min(), y_val.max()], [y_val.min(), y_val.max()], "r--")
plt.xlabel("Actual BPM")
plt.ylabel("Predicted BPM (final)")
plt.title(f"Validation: final preds vs actual (RMSE={val_rmse:.3f})")
plt.grid(True, alpha=0.3)
plt.show()


