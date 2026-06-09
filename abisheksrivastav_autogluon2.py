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


!pip install xgboost lightgbm scipy joblib --quiet


# Run this cell (code cell, not shell) in Kaggle. It can take a few minutes.
!pip install autogluon==1.1.1 --quiet



# AutoGluon 5-seed ensemble + fine shrink grid + linear calibration
# Run after AutoGluon is installed and kernel restarted

import os, time, joblib, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# ---------------- USER SETTINGS ----------------
TRAIN_CSV = "/kaggle/input/playground-series-s5e9/train.csv"
TEST_CSV  = "/kaggle/input/playground-series-s5e9/test.csv"
SAMPLE_SUB = "/kaggle/input/playground-series-s5e9/sample_submission.csv"

# Seeds and time budget
SEEDS = [42, 7, 99, 123, 2023]    # 5 seeds (you can change or add)
AG_TIME_LIMIT = 900               # seconds per AutoGluon run (increase if you can)
NUM_BAG_FOLDS = 5                 # must be >=1 for OOF
AG_PRESET = "medium_quality"      # or "best_quality" if you have more time

OUT_DIR = "/kaggle/working/ag_seed5"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "BeatsPerMinute"
IDCOL = "id"

# fine shrink grid including ultra-small values
SHRINK_GRID = [0.00025, 0.0005, 0.001, 0.0015, 0.002, 0.003, 0.005, 0.01, 0.02]

# safety
MAX_SEEDS = 10
if len(SEEDS) > MAX_SEEDS:
    raise RuntimeError("Too many seeds — reduce SEEDS to <= 10 for safety.")

# ---------------- verify AutoGluon import ----------------
try:
    import autogluon
    from autogluon.tabular import TabularPredictor
    print("AutoGluon version:")
except Exception as e:
    raise RuntimeError("AutoGluon not installed/importable. Run: !pip install autogluon==1.1.1 --quiet and restart kernel.")

# ---------------- load data & minimal FE ----------------
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)
sample   = pd.read_csv(SAMPLE_SUB)

def fe_min(df):
    df = df.copy()
    if "TrackDurationMs" in df.columns:
        df["TrackDurationMin"] = df["TrackDurationMs"].astype(float) / 60000.0
    if "Energy" in df.columns and "RhythmScore" in df.columns:
        df["energy_rhythm"] = df["Energy"].astype(float) * df["RhythmScore"].astype(float)
    if "AudioLoudness" in df.columns and "Energy" in df.columns:
        df["loudness_norm"] = df["AudioLoudness"].astype(float) / (1.0 + df["Energy"].astype(float))
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    return df

train_fe = fe_min(train_df)
test_fe  = fe_min(test_df)
features = [c for c in train_fe.columns if c not in [IDCOL, TARGET]]

n_train = train_fe.shape[0]
n_test  = test_fe.shape[0]
print("Train rows:", n_train, "Test rows:", n_test, "Num features:", len(features))

# ---------------- run AutoGluon for each seed ----------------
oof_list = []
preds_list = []
oof_expected_len = n_train

for seed in SEEDS:
    run_dir = os.path.join(OUT_DIR, f"ag_seed{seed}")
    os.makedirs(run_dir, exist_ok=True)
    print("\n" + "="*60)
    print(f"Running AutoGluon seed {seed} -> path: {run_dir} (time_limit={AG_TIME_LIMIT}s)")
    start_t = time.time()
    predictor = TabularPredictor(label=TARGET, problem_type="regression", eval_metric="rmse", path=run_dir)
    predictor.fit(
        train_data=train_fe[features + [TARGET]],
        presets=AG_PRESET,
        time_limit=AG_TIME_LIMIT,
        num_bag_folds=NUM_BAG_FOLDS,
        ag_args_fit={"seed": seed, "save_bag_folds": True},
        verbosity=2
    )
    elapsed = time.time() - start_t
    print(f"AutoGluon seed {seed} done in {elapsed:.1f}s")

    # get OOF
    try:
        oof_series = predictor.predict_oof()
        oof_arr = np.asarray(oof_series).ravel()
    except Exception as e:
        # fallback: get_oof_pred on feature matrix (older versions)
        try:
            oof_arr = np.asarray(predictor.get_oof_pred(train_fe[features])).ravel()
        except Exception as e2:
            raise RuntimeError(f"Could not obtain OOF for seed {seed}: {e} | {e2}")

    # get test preds
    test_preds = np.asarray(predictor.predict(test_fe[features])).ravel()

    # sanity checks
    if oof_arr.shape[0] != oof_expected_len:
        raise RuntimeError(f"OOF length mismatch for seed {seed}: got {oof_arr.shape[0]} expected {oof_expected_len}")
    if test_preds.shape[0] != n_test:
        raise RuntimeError(f"Test preds length mismatch for seed {seed}: got {test_preds.shape[0]} expected {n_test}")

    # save artifacts
    np.save(os.path.join(OUT_DIR, f"oof_ag_seed{seed}.npy"), oof_arr)
    np.save(os.path.join(OUT_DIR, f"preds_ag_seed{seed}_test.npy"), test_preds)
    joblib.dump({"seed": seed, "path": run_dir, "elapsed": elapsed, "models": predictor.model_names}, os.path.join(OUT_DIR, f"meta_seed{seed}.joblib"))

    rmse_oof = mean_squared_error(train_fe[TARGET].to_numpy(), oof_arr, squared=False)
    print(f" Seed {seed} OOF RMSE: {rmse_oof:.6f}")

    oof_list.append(oof_arr)
    preds_list.append(test_preds)

# ---------------- build seed-ensemble mean OOF + test preds ----------------
oof_stack = np.vstack(oof_list)   # (n_seeds, n_train)
pred_stack = np.vstack(preds_list) # (n_seeds, n_test)

oof_mean = oof_stack.mean(axis=0)
pred_mean = pred_stack.mean(axis=0)
rmse_oof_mean = mean_squared_error(train_fe[TARGET].to_numpy(), oof_mean, squared=False)
print("\nSeed-ensemble (mean) OOF RMSE:", rmse_oof_mean)

# save stacks
np.save(os.path.join(OUT_DIR, "oof_stack_seeds.npy"), oof_stack)
np.save(os.path.join(OUT_DIR, "pred_stack_seeds.npy"), pred_stack)
np.save(os.path.join(OUT_DIR, "oof_ag_mean.npy"), oof_mean)
np.save(os.path.join(OUT_DIR, "preds_ag_mean_test.npy"), pred_mean)

# ---------------- linear calibration on OOF mean ----------------
y = train_fe[TARGET].to_numpy().reshape(-1,1)
X_cal = oof_mean.reshape(-1,1)
lr = LinearRegression()
lr.fit(X_cal, y)
a = float(lr.coef_[0][0]); b = float(lr.intercept_[0])
print(f"\nLinear calibration: y = a*pred + b -> a={a:.6f}, b={b:.6f}")

oof_cal = (a * oof_mean) + b
pred_cal = (a * pred_mean) + b
rmse_oof_cal = mean_squared_error(y.ravel(), oof_cal, squared=False)
print("OOF RMSE after calibration:", rmse_oof_cal)

# ---------------- shrink-to-mean sweep (fine grid) ----------------
mean_y = float(y.mean())
best = {"beta": None, "rmse": 1e9, "oof": None, "pred": None}
for beta in SHRINK_GRID:
    oof_shrunk = (1 - beta) * oof_cal + beta * mean_y
    rm = mean_squared_error(y.ravel(), oof_shrunk, squared=False)
    print(f" beta={beta:.6f} -> OOF RMSE={rm:.6f}")
    if rm < best["rmse"]:
        best["beta"] = beta
        best["rmse"] = rm
        best["oof"] = oof_shrunk.copy()
        best["pred"] = (1 - beta) * pred_cal + beta * mean_y

print("\nBest shrink beta:", best["beta"], "-> OOF RMSE:", best["rmse"])

# ---------------- save best submission (calibrated + shrunk) ----------------
final_pred = np.clip(best["pred"], 20, 240)
sub = sample.copy()
sub[TARGET] = final_pred
best_path = "/kaggle/working/submission_ag_best_calibrated_shrunk.csv"
sub.to_csv(best_path, index=False)
print("Saved best submission:", best_path)

# also save calibrated (no shrink) and uncalibrated mean for reference
sub_cal_path = "/kaggle/working/submission_ag_mean_calibrated.csv"
sub_cal = sample.copy(); sub_cal[TARGET] = np.clip(pred_cal, 20, 240); sub_cal.to_csv(sub_cal_path, index=False)
sub_unc_path = "/kaggle/working/submission_ag_mean_uncalibrated.csv"
sub_unc = sample.copy(); sub_unc[TARGET] = np.clip(pred_mean, 20, 240); sub_unc.to_csv(sub_unc_path, index=False)
print("Saved reference submissions:", sub_unc_path, sub_cal_path)

# ---------------- diagnostics summary ----------------
print("\nSUMMARY DIAGNOSTICS:")
for i,seed in enumerate(SEEDS):
    rm = mean_squared_error(train_fe[TARGET].to_numpy(), oof_stack[i], squared=False)
    print(f" seed {seed} -> OOF RMSE: {rm:.6f}")
print("seed-ensemble mean OOF RMSE:", rmse_oof_mean)
print("after calibration OOF RMSE:", rmse_oof_cal)
print("best beta:", best["beta"], "best OOF RMSE:", best["rmse"])

print("\nFiles written to /kaggle/working/ and artifacts to", OUT_DIR)
print("Submit submission_ag_best_calibrated_shrunk.csv and tell me the public LB score.")


