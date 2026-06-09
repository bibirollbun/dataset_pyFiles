!pip install "optuna-integration[xgboost]"


import pandas as pd
import numpy as np
import os 
import time 
import seaborn as sns
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.integration import XGBoostPruningCallback
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, train_test_split
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import joblib
import optuna
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


train = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
original = pd.read_csv('/kaggle/input/bpm-prediction-challenge/Train.csv')


train = pd.concat([train, original], axis=0, ignore_index=True)
tran = train.drop_duplicates().reset_index(drop=True)


train.head()


train.info()


train.describe()


epsilon = 1e-6

# --- Base conversions ---
train['TrackDurationMin'] = train['TrackDurationMs'] / 60000

# --- Ratios & balances ---
train['Energy_Acoustic_Ratio'] = train['Energy'] / (train['AcousticQuality'] + epsilon)
train['Vocal_Instrument_Balance'] = train['VocalContent'] / (train['InstrumentalScore'] + epsilon)
train['Vocal_share'] = train['VocalContent'] / (train['VocalContent'] + train['InstrumentalScore'] + epsilon)
train['Instrumental_to_Total'] = train['InstrumentalScore'] / (train['VocalContent'] + train['InstrumentalScore'] + epsilon)
train['Acoustic_to_Energy'] = train['AcousticQuality'] / (train['Energy'] + epsilon)

# --- Multiplicative interactions ---
train['MoodRhythm'] = train['MoodScore'] * train['RhythmScore']
train['RhythmEnergy'] = train['RhythmScore'] * train['Energy']
train['MoodAcoustic'] = train['MoodScore'] * train['AcousticQuality']
train['Vocal_Energy_Interaction'] = train['VocalContent'] * train['Energy']
train['PerformanceIntensity'] = train['LivePerformanceLikelihood'] * train['AudioLoudness']
train['Electronic_Proxy'] = (1 - train['AcousticQuality']) * train['Energy'] * train['RhythmScore']
train['Ambient_Proxy'] = train['AcousticQuality'] * (1 - train['Energy']) * (1 - train['RhythmScore'])
train['Ballad_Proxy'] = train['VocalContent'] * train['AcousticQuality'] * (1 - train['Energy']) * (1 - train['RhythmScore'])
train['Instrumental_Intensity_Proxy'] = train['InstrumentalScore'] * train['Energy'] * train['AudioLoudness']
train['Mood_Rhythm_Energy'] = train['MoodScore'] * train['RhythmScore'] * train['Energy']

# --- Differences ---
train['Energy_Mood_diff'] = train['Energy'] - train['MoodScore']
train['Energy_Mood_absdiff'] = np.abs(train['Energy_Mood_diff'])
train['Rhythm_Instrument_diff'] = train['RhythmScore'] - train['InstrumentalScore']
train['Vocal_Instrument_diff'] = train['VocalContent'] - train['InstrumentalScore']

# --- Log / nonlinear transforms ---
train['TrackDurationMin_log'] = np.log1p(train['TrackDurationMin'])
train['Rhythm_sqrt'] = np.sqrt(train['RhythmScore'])
train['AudioLoudness_sq'] = train['AudioLoudness'] ** 2
train['Energy_cbrt'] = np.cbrt(train['Energy'])
train['log_Rhythm_over_Acoustic'] = np.log1p(train['RhythmScore'] / (train['AcousticQuality'] + epsilon))
train['log_Instrumental_Energy'] = np.log1p(train['InstrumentalScore'] * train['Energy'])

# --- Composite interactions ---
train['Rhythm_Energy_Loudness'] = train['RhythmScore'] * train['Energy'] * (train['AudioLoudness'] + 20)
train['Instrumental_Energy'] = train['InstrumentalScore'] * train['Energy']
train['Vocal_Rhythm_Energy'] = train['VocalContent'] * train['RhythmScore'] * train['Energy']
train['RhythmEnergy_log'] = np.log1p(train['RhythmScore'] * train['Energy'])
train['RhythmEnergy_over_Instrument'] = (train['RhythmScore'] * train['Energy']) / (train['InstrumentalScore'] + epsilon)

# --- Proxies with sigmoid (soft categories) ---
def sigmoid(x): return 1 / (1 + np.exp(-x))

x = 3*(train['Energy'] - 0.5) + 2*(train['RhythmScore'] - 0.5) + 0.2*(train['AudioLoudness'] + 8)
train['Dance_Proxy'] = sigmoid(x)

y = 4*(train['VocalContent']) + 3*(train['AcousticQuality']) - 5*(train['Energy']) - 4*(train['RhythmScore'])
train['Ballad_Proxy_Score'] = sigmoid(y)

# --- Relative rankings ---
train['Rhythm_rank'] = train['RhythmScore'].rank(pct=True)
train['Energy_rank'] = train['Energy'].rank(pct=True)
train['RhythmEnergy_rank_prod'] = train['Rhythm_rank'] * train['Energy_rank']

# --- Ratios with loudness ---
train['Rhythm_Loudness_Ratio'] = train['RhythmScore'] / (train['AudioLoudness'] + epsilon)
train['Rhythm_Loudness_Ratio_clipped'] = np.clip(train['RhythmScore'] / (train['AudioLoudness'] + 20 + epsilon), -100, 100)

# --- Duration transforms ---
train['TrackDurationMin_inv'] = 1 / (train['TrackDurationMin'] + epsilon)

# --- NEW: Statistical moment features ---
# Calculate rolling statistics for key features
for col in ['Energy', 'RhythmScore', 'AudioLoudness', 'MoodScore']:
    train[f'{col}_zscore'] = (train[col] - train[col].mean()) / train[col].std()
    train[f'{col}_skewness'] = (train[col] - train[col].mean())**3 / train[col].std()**3

# --- NEW: Polynomial features ---
for col in ['Energy', 'RhythmScore', 'AudioLoudness']:
    train[f'{col}_squared'] = train[col] ** 2
    train[f'{col}_cubed'] = train[col] ** 3

# --- NEW: Audio properties combinations ---
train['Loudness_Energy_Ratio'] = train['AudioLoudness'] / (train['Energy'] + epsilon)
train['Dynamic_Range_Proxy'] = train['AudioLoudness'].max() - train['AudioLoudness'].min()  # This might need adjustment

# --- NEW: Genre-like proxies based on statistics ---
train['Mellow_Proxy'] = train['AcousticQuality'] * train['MoodScore'] * (1 - train['Energy'])

# --- NEW: Vocal prominence score ---
train['Vocal_Prominence'] = train['VocalContent'] * (1 - train['InstrumentalScore']) * train['AudioLoudness']

# --- NEW: Performance quality indicators ---
train['Live_Energy_Balance'] = train['LivePerformanceLikelihood'] * train['Energy']
train['Studio_Polish_Proxy'] = (1 - train['LivePerformanceLikelihood']) * train['AudioLoudness']

# --- NEW: Complex interaction terms ---
train['Full_Production_Score'] = (
    train['Energy'] * train['AudioLoudness'] * 
    (1 - train['AcousticQuality']) * train['RhythmScore']
)

train = train.replace([np.inf, -np.inf], np.nan)


epsilon = 1e-6

# --- Base conversions ---
test['TrackDurationMin'] = test['TrackDurationMs'] / 60000

# --- Ratios & balances ---
test['Energy_Acoustic_Ratio'] = test['Energy'] / (test['AcousticQuality'] + epsilon)
test['Vocal_Instrument_Balance'] = test['VocalContent'] / (test['InstrumentalScore'] + epsilon)
test['Vocal_share'] = test['VocalContent'] / (test['VocalContent'] + test['InstrumentalScore'] + epsilon)
test['Instrumental_to_Total'] = test['InstrumentalScore'] / (test['VocalContent'] + test['InstrumentalScore'] + epsilon)
test['Acoustic_to_Energy'] = test['AcousticQuality'] / (test['Energy'] + epsilon)

# --- Multiplicative interactions ---
test['MoodRhythm'] = test['MoodScore'] * test['RhythmScore']
test['RhythmEnergy'] = test['RhythmScore'] * test['Energy']
test['MoodAcoustic'] = test['MoodScore'] * test['AcousticQuality']
test['Vocal_Energy_Interaction'] = test['VocalContent'] * test['Energy']
test['PerformanceIntensity'] = test['LivePerformanceLikelihood'] * test['AudioLoudness']
test['Electronic_Proxy'] = (1 - test['AcousticQuality']) * test['Energy'] * test['RhythmScore']
test['Ambient_Proxy'] = test['AcousticQuality'] * (1 - test['Energy']) * (1 - test['RhythmScore'])
test['Ballad_Proxy'] = test['VocalContent'] * test['AcousticQuality'] * (1 - test['Energy']) * (1 - test['RhythmScore'])
test['Instrumental_Intensity_Proxy'] = test['InstrumentalScore'] * test['Energy'] * test['AudioLoudness']
test['Mood_Rhythm_Energy'] = test['MoodScore'] * test['RhythmScore'] * test['Energy']

# --- Differences ---
test['Energy_Mood_diff'] = test['Energy'] - test['MoodScore']
test['Energy_Mood_absdiff'] = np.abs(test['Energy_Mood_diff'])
test['Rhythm_Instrument_diff'] = test['RhythmScore'] - test['InstrumentalScore']
test['Vocal_Instrument_diff'] = test['VocalContent'] - test['InstrumentalScore']

# --- Log / nonlinear transforms ---
test['TrackDurationMin_log'] = np.log1p(test['TrackDurationMin'])
test['Rhythm_sqrt'] = np.sqrt(test['RhythmScore'])
test['AudioLoudness_sq'] = test['AudioLoudness'] ** 2
test['Energy_cbrt'] = np.cbrt(test['Energy'])
test['log_Rhythm_over_Acoustic'] = np.log1p(test['RhythmScore'] / (test['AcousticQuality'] + epsilon))
test['log_Instrumental_Energy'] = np.log1p(test['InstrumentalScore'] * test['Energy'])

# --- Composite interactions ---
test['Rhythm_Energy_Loudness'] = test['RhythmScore'] * test['Energy'] * (test['AudioLoudness'] + 20)
test['Instrumental_Energy'] = test['InstrumentalScore'] * test['Energy']
test['Vocal_Rhythm_Energy'] = test['VocalContent'] * test['RhythmScore'] * test['Energy']
test['RhythmEnergy_log'] = np.log1p(test['RhythmScore'] * test['Energy'])
test['RhythmEnergy_over_Instrument'] = (test['RhythmScore'] * test['Energy']) / (test['InstrumentalScore'] + epsilon)

# --- Proxies with sigmoid (soft categories) ---
def sigmoid(x): return 1 / (1 + np.exp(-x))

x = 3*(test['Energy'] - 0.5) + 2*(test['RhythmScore'] - 0.5) + 0.2*(test['AudioLoudness'] + 8)
test['Dance_Proxy'] = sigmoid(x)

y = 4*(test['VocalContent']) + 3*(test['AcousticQuality']) - 5*(test['Energy']) - 4*(test['RhythmScore'])
test['Ballad_Proxy_Score'] = sigmoid(y)

# --- Relative rankings ---
test['Rhythm_rank'] = test['RhythmScore'].rank(pct=True)
test['Energy_rank'] = test['Energy'].rank(pct=True)
test['RhythmEnergy_rank_prod'] = test['Rhythm_rank'] * test['Energy_rank']

# --- Ratios with loudness ---
test['Rhythm_Loudness_Ratio'] = test['RhythmScore'] / (test['AudioLoudness'] + epsilon)
test['Rhythm_Loudness_Ratio_clipped'] = np.clip(test['RhythmScore'] / (test['AudioLoudness'] + 20 + epsilon), -100, 100)

# --- Duration transforms ---
test['TrackDurationMin_inv'] = 1 / (test['TrackDurationMin'] + epsilon)

# --- NEW: Statistical moment features ---
# Calculate rolling statistics for key features
for col in ['Energy', 'RhythmScore', 'AudioLoudness', 'MoodScore']:
    test[f'{col}_zscore'] = (test[col] - test[col].mean()) / test[col].std()
    test[f'{col}_skewness'] = (test[col] - test[col].mean())**3 / test[col].std()**3

# --- NEW: Polynomial features ---
for col in ['Energy', 'RhythmScore', 'AudioLoudness']:
    test[f'{col}_squared'] = test[col] ** 2
    test[f'{col}_cubed'] = test[col] ** 3

# --- NEW: Audio properties combinations ---
test['Loudness_Energy_Ratio'] = test['AudioLoudness'] / (test['Energy'] + epsilon)
test['Dynamic_Range_Proxy'] = test['AudioLoudness'].max() - test['AudioLoudness'].min()  # This might need adjustment

# --- NEW: Genre-like proxies based on statistics ---
test['Mellow_Proxy'] = test['AcousticQuality'] * test['MoodScore'] * (1 - test['Energy'])

# --- NEW: Vocal prominence score ---
test['Vocal_Prominence'] = test['VocalContent'] * (1 - test['InstrumentalScore']) * test['AudioLoudness']

# --- NEW: Performance quality indicators ---
test['Live_Energy_Balance'] = test['LivePerformanceLikelihood'] * test['Energy']
test['Studio_Polish_Proxy'] = (1 - test['LivePerformanceLikelihood']) * test['AudioLoudness']

# --- NEW: Complex interaction terms ---
test['Full_Production_Score'] = (
    test['Energy'] * test['AudioLoudness'] * 
    (1 - test['AcousticQuality']) * test['RhythmScore']
)

test = test.replace([np.inf, -np.inf], np.nan)


train.head()


X = train.drop(columns=["id", "BeatsPerMinute"])
y = train["BeatsPerMinute"]
X_test = test.drop(columns=["id"])


# Train and test data
X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Test and val data
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)


RANDOM_SEED = 42
N_TRIALS = 100                # adjust as needed (e.g., 100 or 200 for more thorough tuning)
N_ESTIMATORS_MAX = 10000     # maximum number of trees (as you requested)
EARLY_STOPPING_ROUNDS = 50   # stop early if no improvement for this many rounds
VERBOSE = False   

GPU_PARAMS = {
    "tree_method": "gpu_hist",
    "predictor": "gpu_predictor",
    "gpu_id": 0,
}


def objective(trial):
    # Parameter search space
    params = {
        # learning rate (eta)
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 20),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        # objective + eval metric are fixed for regression
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "verbosity": 0,
        # optionally we can set booster type, but keep default 'gbtree'
    }

    # Combine GPU params with trial params
    all_params = {**params, **GPU_PARAMS}

    # Build model with a very large n_estimators; we'll rely on early stopping
    model = XGBRegressor(
        n_estimators=N_ESTIMATORS_MAX,
        random_state=RANDOM_SEED,
        n_jobs=1,  # GPU does parallelism internally; keep this small to avoid conflicts
        **all_params
    )

    # Use Optuna pruning callback that monitors validation rmse ("validation_0-rmse")
    pruning_callback = XGBoostPruningCallback(trial, "validation_0-rmse")

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose=False,
        callbacks=[pruning_callback],
    )

    # Save best_iteration for later use (if early stopping occurred)
    try:
        best_iter = model.get_booster().best_iteration
    except Exception:
        best_iter = None

    # Record best iteration in trial user attrs
    trial.set_user_attr("best_iteration", int(best_iter) if best_iter is not None else None)

    # Predict on validation set and return RMSE
    preds_val = model.predict(X_val)
    rmse = mean_squared_error(y_val, preds_val, squared=False)

    return rmse


#sampler = TPESampler(seed=RANDOM_SEED)
#pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=5)  
#study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)


#print(f"Starting Optuna study with {N_TRIALS} trials (this may take a while depending on trials & GPU).")
#study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)


#best_trial = study.best_trial
#print("\n=== Best trial ===")
#print("Value (RMSE):", best_trial.value)
#print("Params:")
#for k, v in best_trial.params.items():
#    print(f"  {k}: {v}")
#best_iteration = best_trial.user_attrs.get("best_iteration", None)
#print("Best iteration (from early stopping):", best_iteration)


RANDOM_SEED = 42
USE_GPU = True               # set False if no GPU
OUTPUT_DIR = "final_model_cv_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Best params from your Optuna run
best_params = {
    "learning_rate": 0.11936460732799715,
    "max_depth": 3,
    "min_child_weight": 2,
    "subsample": 0.5606560470740858,
    "colsample_bytree": 0.7661167052009941,
    "gamma": 0.631002480673827,
    "reg_alpha": 6.385721875114712,
    "reg_lambda": 1.673685454892917e-06,
    # fixed:
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "verbosity": 0,
}
GPU_PARAMS = {"tree_method": "gpu_hist", "predictor": "gpu_predictor", "gpu_id": 0}
if USE_GPU:
    best_params.update(GPU_PARAMS)

# Best iteration from your tuning
best_iteration = 16
if best_iteration is None or int(best_iteration) < 1:
    # fallback if not available, set a reasonable number
    n_estimators_final = 10000
else:
    n_estimators_final = int(best_iteration)

# CV settings
N_FOLDS = 10
shuffle_cv = True

# ---------------- Load data ----------------
# Try to use existing variables in the environment, otherwise load CSVs
try:
    train  # noqa: F821
    test   # noqa: F821
    y      # noqa: F821
    env_has_data = True
except NameError:
    env_has_data = False

if not env_has_data:
    if os.path.exists("train.csv") and os.path.exists("test.csv"):
        train = pd.read_csv("train.csv")
        test = pd.read_csv("test.csv")
        if "BeatsPerMinute" not in train.columns:
            raise KeyError("train.csv must contain target column 'BeatsPerMinute'")
        y = train["BeatsPerMinute"].copy()
    else:
        raise RuntimeError("Dataframes `train`/`test` not found and train.csv/test.csv missing.")

# Recreate splits exactly like your earlier code (train/val/test)
X = train.drop(columns=[c for c in ["id", "BeatsPerMinute"] if c in train.columns])
X_train_val, X_test_local, y_train_val, y_test_local = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_SEED
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.2, random_state=RANDOM_SEED
)

# Use external test if present, otherwise the held-out portion
if "BeatsPerMinute" in test.columns:
    X_test = test.drop(columns=[c for c in ["id", "BeatsPerMinute"] if c in test.columns])
    y_test = test["BeatsPerMinute"].copy()
else:
    X_test = X_test_local
    y_test = y_test_local

# Combine train+val to perform stratified K-fold CV
X_trainval = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
y_trainval = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

print(f"Train+Val shape: {X_trainval.shape}, Test shape: {X_test.shape}")

# ---------------- Create stratify bins for regression ----------------
def make_stratify_bins(y_series, n_bins=5):
    """Create quantile-based bins for stratified k-fold on a regression target.
       Falls back to uniform binning if qcut fails due to duplicate edges."""
    try:
        y_binned, bins = pd.qcut(y_series, q=n_bins, labels=False, retbins=True, duplicates="drop")
        # qcut may produce fewer bins if there are many duplicates; ensure labels are ints
        y_binned = y_binned.astype(int)
        return y_binned
    except Exception:
        # fallback: simple uniform discretization
        y_min, y_max = y_series.min(), y_series.max()
        if y_max == y_min:
            return np.zeros(len(y_series), dtype=int)
        scaled = (y_series - y_min) / (y_max - y_min + 1e-12)
        bins = np.floor(scaled * n_bins).astype(int)
        bins[bins == n_bins] = n_bins - 1
        return bins

y_bins = make_stratify_bins(y_trainval, n_bins=N_FOLDS)

# If number of unique bins < n_folds, reduce folds or adjust
unique_bins = np.unique(y_bins)
if unique_bins.shape[0] < N_FOLDS:
    # reduce n_splits to number of unique bins
    effective_folds = int(unique_bins.shape[0])
    print(f"Warning: only {effective_folds} unique bins found for stratification; using {effective_folds} folds.")
else:
    effective_folds = N_FOLDS

skf = StratifiedKFold(n_splits=effective_folds, shuffle=shuffle_cv, random_state=RANDOM_SEED)

# ---------------- CV training ----------------
test_preds_folds = []        # store per-fold test predictions
oof_preds = np.zeros(len(X_trainval))
fold_rmse_list = []
models = []

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_trainval, y_bins)):
    print(f"\n----- Fold {fold + 1}/{effective_folds} -----")
    X_tr, X_val_fold = X_trainval.iloc[tr_idx], X_trainval.iloc[val_idx]
    y_tr, y_val_fold = y_trainval.iloc[tr_idx], y_trainval.iloc[val_idx]

    # Create model (use final n_estimators determined earlier)
    model = XGBRegressor(
        n_estimators=n_estimators_final,
        random_state=RANDOM_SEED + fold,
        n_jobs=1,
        **best_params,
    )

    # Fit model: because n_estimators is small (best_iteration), we skip early stopping here.
    # If you'd rather keep early stopping, set n_estimators to a large number and pass eval_set & early_stopping_rounds.
    model.fit(X_tr, y_tr, verbose=False)

    # Predict val and test
    preds_val_fold = model.predict(X_val_fold)
    oof_preds[val_idx] = preds_val_fold
    rmse_fold = mean_squared_error(y_val_fold, preds_val_fold, squared=False)
    fold_rmse_list.append(rmse_fold)
    print(f"Fold {fold+1} RMSE: {rmse_fold:.6f}")

    preds_test_fold = model.predict(X_test)
    test_preds_folds.append(preds_test_fold)

    # Save per-fold model
    model_path = os.path.join(OUTPUT_DIR, f"xgb_fold{fold+1}.joblib")
    joblib.dump(model, model_path)
    print(f"Saved model to {model_path}")
    models.append(model)

# ---------------- Aggregate CV results ----------------
cv_rmse = mean_squared_error(y_trainval, oof_preds, squared=False)
print("\n=== CV results ===")
for i, r in enumerate(fold_rmse_list, 1):
    print(f" Fold {i} RMSE: {r:.6f}")
print(f"OOF (train+val) RMSE: {cv_rmse:.6f}")
print(f"Mean fold RMSE: {np.mean(fold_rmse_list):.6f} (std {np.std(fold_rmse_list):.6f})")

# Average test predictions across folds
test_preds_avg = np.mean(np.column_stack(test_preds_folds), axis=1)
test_rmse_avg = mean_squared_error(y_test, test_preds_avg, squared=False)
print(f"Test RMSE (averaged over {len(test_preds_folds)} folds): {test_rmse_avg:.6f}")

# Save averaged test predictions and OOF predictions
pd.DataFrame({"prediction": test_preds_avg}).to_csv(os.path.join(OUTPUT_DIR, "test_predictions_avg.csv"), index=False)
pd.DataFrame({"oof_prediction": oof_preds, "target": y_trainval}).to_csv(os.path.join(OUTPUT_DIR, "oof_preds.csv"), index=False)
print(f"Saved averaged test predictions and OOF predictions to {OUTPUT_DIR}")

# ---------------- Optional: feature importance from last fold (or average later) ----------------
booster = models[-1].get_booster()
importance = booster.get_score(importance_type="gain")
if len(importance) == 0:
    print("No feature importance found.")
else:
    fi = pd.DataFrame(importance.items(), columns=["feature", "gain"]).sort_values("gain", ascending=True)
    plt.figure(figsize=(8, max(4, 0.2 * len(fi))))
    plt.barh(fi["feature"], fi["gain"])
    plt.xlabel("Gain")
    plt.title("Feature importance (gain) - last fold")
    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, "feature_importance_gain_last_fold.png")
    plt.savefig(fig_path)
    print(f"Saved feature importance plot to {fig_path}")
    plt.show()


# ---------------- Train final model on full training data ----------------
X_full = X  # all training features
y_full = y  # all targets

final_model = XGBRegressor(
    n_estimators=n_estimators_final,
    random_state=RANDOM_SEED,
    n_jobs=1,
    **best_params,
)

print("Training final model on FULL data...")
final_model.fit(X_full, y_full, verbose=False)

# Predict on the FULL Kaggle test set (not split test)
X_test_full = test.drop(columns=["id"])
test_preds_final = final_model.predict(X_test_full)

# ---------------- Create submission file ----------------
submission = pd.DataFrame({
    "id": test["id"],
    "BeatsPerMinute": test_preds_final
})

sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)
print(f"✅ Submission file saved to: {sub_path}")

