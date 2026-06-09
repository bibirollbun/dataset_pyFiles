# Check for GPU availability
import subprocess

try:
    subprocess.check_output(["nvidia-smi"])
    gpu_available = True
    print("GPU detected.")
except Exception as e:
    gpu_available = False
    print("No GPU detected, running in CPU mode.")

# Try to load cuDF-Pandas if GPU is available
if gpu_available:
    try:
        get_ipython().run_line_magic('load_ext', 'cudf.pandas')
        print("Loaded cuDF-Pandas extension for GPU acceleration.")
    except Exception as e:
        print("Could not load cuDF-Pandas extension. Falling back to Pandas. Error:", e)
else:
    print("Skipping cuDF-Pandas extension since no GPU is available.")

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)

VER = 1

# ---------------------------
# Load Data
# ---------------------------
train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
print("Train shape", train.shape)
train.head()

train2 = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
print("Extra Train shape", train2.shape)
train2.head()

train = pd.concat([train, train2], axis=0, ignore_index=True)
print("Combined Train shape", train.shape)

test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
print("Test shape", test.shape)
test.head()

# ---------------------------
# Feature Engineer Columns
# ---------------------------
CATS = list(train.columns[1:-2])
print(f"There are {len(CATS)} categorical columns:")
print(CATS)
print("There are 1 numerical column:")
print(["Weight Capacity (kg)"])

COMBO = []
for i, c in enumerate(CATS):
    combine = pd.concat([train[c], test[c]], axis=0)
    combine, _ = pd.factorize(combine)
    train[c] = combine[:len(train)]
    test[c] = combine[len(train):]
    n = f"{c}_wc"
    train[n] = train[c] * 100 + train["Weight Capacity (kg)"]
    test[n] = test[c] * 100 + test["Weight Capacity (kg)"]
    COMBO.append(n)
print()
print(f"We engineer {len(COMBO)} new columns!")
print(COMBO)

FEATURES = CATS + ["Weight Capacity (kg)"] + COMBO
print(f"We now have {len(FEATURES)} columns:")
print(FEATURES)

# ---------------------------
# XGBoost with Feature Engineer GroupBy
# ---------------------------
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import xgboost as xgb
print("XGBoost version", xgb.__version__)

# STATISTICS TO AGGREGATE FOR OUR FEATURE GROUPS
STATS = ["mean", "std", "count", "nunique", "median", "min", "max", "skew"]
STATS2 = ["mean", "std"]

# Set device parameter based on GPU availability
device_param = "cuda" if gpu_available else "cpu"

# ---------------------------
# Begin Timing (optional)
# ---------------------------
import time
start_time = time.time()

FOLDS = 7
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=42)

oof = np.zeros(len(train))
pred = np.zeros(len(test))

# OUTER K-FOLD
for i, (train_index, test_index) in enumerate(kf.split(train)):
    print(f"### OUTER Fold {i+1} ###")
    
    X_train = train.loc[train_index, FEATURES + ['Price']].reset_index(drop=True).copy()
    y_train = train.loc[train_index, 'Price']

    X_valid = train.loc[test_index, FEATURES].reset_index(drop=True).copy()
    y_valid = train.loc[test_index, 'Price']

    X_test = test[FEATURES].reset_index(drop=True).copy()

    # INNER K-FOLD (TO PREVENT LEAKAGE WHEN USING PRICE)
    kf2 = KFold(n_splits=FOLDS, shuffle=True, random_state=42)   
    for j, (train_index2, test_index2) in enumerate(kf2.split(X_train)):
        print(f" ## INNER Fold {j+1} (outer fold {i+1}) ##")

        X_train2 = X_train.loc[train_index2, FEATURES + ['Price']].copy()
        X_valid2 = X_train.loc[test_index2, FEATURES].copy()

        ### FEATURE SET 1 (uses price) ###
        col = "Weight Capacity (kg)"
        tmp = X_train2.groupby(col).Price.agg(STATS)
        tmp.columns = [f"TE1_wc_{s}" for s in STATS]
        X_valid2 = X_valid2.merge(tmp, on=col, how="left")
        for c in tmp.columns:
            X_train.loc[test_index2, c] = X_valid2[c].values

        ### FEATURE SET 2 (uses price) ###
        for col in COMBO:
            tmp = X_train2.groupby(col).Price.agg(STATS2)
            tmp.columns = [f"TE2_{col}_{s}" for s in STATS2]
            X_valid2 = X_valid2.merge(tmp, on=col, how="left")
            for c in tmp.columns:
                X_train.loc[test_index2, c] = X_valid2[c].values

    ### FEATURE SET 1 (uses price) ###
    col = "Weight Capacity (kg)"
    tmp = X_train.groupby(col).Price.agg(STATS)
    tmp.columns = [f"TE1_wc_{s}" for s in STATS]
    X_valid = X_valid.merge(tmp, on=col, how="left")
    X_test = X_test.merge(tmp, on=col, how="left")

    ### FEATURE SET 2 (uses price) ###
    for col in COMBO:
        tmp = X_train.groupby(col).Price.agg(STATS2)
        tmp.columns = [f"TE2_{col}_{s}" for s in STATS2]
        X_valid = X_valid.merge(tmp, on=col, how="left")
        X_test = X_test.merge(tmp, on=col, how="left")

    ### FEATURE SET 3 (does not use price) ###
    for col in CATS:
        col2 = "Weight Capacity (kg)"
        tmp = X_train.groupby(col)[col2].agg(STATS2)
        tmp.columns = [f"FE3_{col}_wc_{s}" for s in STATS2]
        X_train = X_train.merge(tmp, on=col, how="left")
        X_valid = X_valid.merge(tmp, on=col, how="left")
        X_test = X_test.merge(tmp, on=col, how="left")

    # CONVERT TO CATS SO XGBOOST RECOGNIZES THEM
    X_train[CATS] = X_train[CATS].astype("category")
    X_valid[CATS] = X_valid[CATS].astype("category")
    X_test[CATS] = X_test[CATS].astype("category")

    # DROP PRICE THAT WAS USED FOR TARGET ENCODING
    X_train = X_train.drop(['Price'], axis=1)

    # BUILD MODEL
    model = XGBRegressor(
        device=device_param,
        max_depth=6,  
        colsample_bytree=0.5, 
        subsample=0.8,  
        n_estimators=10000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=10,
        early_stopping_rounds=100,
    )
    
    # TRAIN MODEL
    COLS = X_train.columns
    model.fit(
        X_train[COLS], y_train,
        eval_set=[(X_valid[COLS], y_valid)],  
        verbose=300,
    )

    # PREDICT OOF AND TEST
    oof[test_index] = model.predict(X_valid[COLS])
    pred += model.predict(X_test[COLS])

pred /= FOLDS
print("Total time: {:.2f} seconds".format(time.time() - start_time))

# ---------------------------
# Evaluation and Saving Results
# ---------------------------
true = train.Price.values
s = np.sqrt(np.mean((oof - true) ** 2.0))
print(f"=> Overall CV Score = {s}")

# SAVE OOF TO DISK FOR ENSEMBLES
np.save(f"oof_v{VER}.npy", oof)
print("Saved OOF predictions to disk.")

# Print Feature Names
print(f"\nIn total, we used {len(COLS)} features, Wow!\n")
print(list(COLS))

# ---------------------------
# XGB Feature Importance
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 20))
xgb.plot_importance(model, max_num_features=100, importance_type='gain', ax=ax)
plt.title("Top 100 Feature Importances (XGBoost)")
plt.show()

# ---------------------------
# Make Submission CSV and Plot Predictions
# ---------------------------
sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub.Price = pred
sub.to_csv(f"submission_v{VER}.csv", index=False)
print(sub.head())

plt.figure(figsize=(6,4))
plt.hist(sub.Price, bins=100)
plt.title("Test Predictions")
plt.show()


