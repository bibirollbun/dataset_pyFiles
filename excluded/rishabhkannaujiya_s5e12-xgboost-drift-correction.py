import numpy as np
import pandas as pd
import os
import gc
import random
import warnings

from joblib import Parallel, delayed

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

import xgboost as xgb

SEED = 42

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

seed_everything(SEED)

warnings.filterwarnings("ignore")

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

print("Block 1 completed âœ”")
print("XGBoost version:", xgb.__version__)



# ====================================================
# BLOCK 2: Data Loading & Sanity Checks
# ====================================================

TRAIN_PATH = "/kaggle/input/playground-series-s5e12/train.csv"
TEST_PATH  = "/kaggle/input/playground-series-s5e12/test.csv"

train_df = pd.read_csv(TRAIN_PATH)
test_df  = pd.read_csv(TEST_PATH)


# ====================================================
# BLOCK 2.5: Drift Visualization
# ====================================================
import matplotlib.pyplot as plt
import seaborn as sns

feature = 'physical_activity_minutes_per_week'

plt.figure(figsize=(15, 6))

plt.plot(
    train_df['id'], 
    train_df[feature].rolling(window=5000).mean(), 
    label=f'Train {feature} (Rolling Mean)', 
    color='blue'
)

test_mean = test_df[feature].mean()
plt.axhline(y=test_mean, color='red', linestyle='--', label=f'Test Mean ({test_mean:.2f})')

plt.title(f'Drift Analysis: {feature} vs ID')
plt.xlabel('ID')
plt.ylabel('Rolling Mean Value')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)


TARGET = "diagnosed_diabetes"

print("\nTarget distribution:")
print(train_df[TARGET].value_counts(normalize=True))


print("\nMissing values (top 10):")
display(train_df.isnull().sum().sort_values(ascending=False).head(10))


print("\nColumns:")
print(train_df.columns.tolist())


feature_cols = [c for c in train_df.columns if c not in [TARGET]]

assert TARGET not in test_df.columns, "Target leaked into test set!"

print("\nID statistics (train):")
display(train_df["id"].describe())

print("\nID statistics (test):")
display(test_df["id"].describe())


def add_features(df):
    df = df.copy()

    df["ldl_hdl_ratio"] = df["ldl_cholesterol"] / (df["hdl_cholesterol"] + 1e-6)
    df["triglycerides_hdl_ratio"] = df["triglycerides"] / (df["hdl_cholesterol"] + 1e-6)

    df["pulse_pressure"] = df["systolic_bp"] - df["diastolic_bp"]
    df["mean_arterial_pressure"] = (
        df["diastolic_bp"] + (df["pulse_pressure"] / 3.0)
    )

    df["bmi_waist_ratio"] = df["bmi"] * df["waist_to_hip_ratio"]

    df["lifestyle_load"] = (
        df["screen_time_hours_per_day"] /
        (df["physical_activity_minutes_per_week"] + 1.0)
    )

    df["metabolic_risk_score"] = (
        df["bmi"] *
        df["ldl_hdl_ratio"]
    )
    
    df["physical_activity_bin"] = pd.qcut(
        df["physical_activity_minutes_per_week"], 
        q=10, 
        labels=False, 
        duplicates='drop'
    )
    
    df["triglycerides_bin"] = pd.qcut(
        df["triglycerides"], 
        q=10, 
        labels=False, 
        duplicates='drop'
    )

    return df


cat_cols = [
    "gender",
    "ethnicity",
    "education_level",
    "income_level",
    "smoking_status",
    "employment_status"
]

def encode_categoricals(train_df, test_df, cat_cols):
    train_df = train_df.copy()
    test_df = test_df.copy()

    for col in cat_cols:
        combined = pd.concat([train_df[col], test_df[col]], axis=0)

        codes, uniques = pd.factorize(combined, sort=True)

        train_df[col] = codes[:len(train_df)]
        test_df[col] = codes[len(train_df):]

        train_df[col] = train_df[col].astype("int32")
        test_df[col] = test_df[col].astype("int32")

    return train_df, test_df


train_df, test_df = encode_categoricals(train_df, test_df, cat_cols)

print("Categorical dtypes after encoding:")
display(train_df[cat_cols].dtypes)


sample_weights = np.ones(len(train_df))

print("Sample weighting disabled. All weights set to 1.0.")


# ====================================================
# BLOCK 5: Stratified K-Fold Setup
# ====================================================

N_FOLDS = 5

skf = StratifiedKFold(
    n_splits=N_FOLDS,
    shuffle=True,
    random_state=SEED
)

X = train_df.drop(columns=[TARGET])
y = train_df[TARGET].values

oof_preds = np.zeros(len(train_df))

print(f"Using {N_FOLDS}-Fold Stratified CV")


# ====================================================
# BLOCK 6: XGBoost Anchor Configuration (Optuned)
# ====================================================

xgb_params = {
    "objective": "binary:logistic",
    "eval_metric": "auc",

    "max_depth": 5,
    "min_child_weight": 9,
    "gamma": 0.3290038823015833,

    "subsample": 0.8220147154680112,
    "colsample_bytree": 0.5328892343926531,

    "reg_alpha": 9.751720788060178,
    "reg_lambda": 2.7502551002838533,

    "learning_rate": 0.013197332259121087,

    "n_estimators": 5000,
    
    "tree_method": "hist",
    "device": "cuda",

    "random_state": SEED,
    "verbosity": 0
}

print("XGBoost optimized parameters set âœ”")


# ====================================================
# BLOCK 7: Fold-wise Training & OOF Evaluation (2 GPUs Parallel)
# ====================================================

def train_fold(fold, train_idx, val_idx, X, y, weights, params):
    gpu_id = fold % 2

    fold_params = params.copy()
    fold_params["device"] = f"cuda:{gpu_id}"
    fold_params["early_stopping_rounds"] = 50

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    w_train, w_val = weights[train_idx], weights[val_idx]

    model = xgb.XGBClassifier(**fold_params)

    model.fit(
        X_train,
        y_train,
        sample_weight=w_train,
        eval_set=[(X_val, y_val)],
        sample_weight_eval_set=[w_val],
        verbose=False
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    score = roc_auc_score(y_val, val_preds)

    print(f"Fold {fold + 1} finished | GPU {gpu_id} | AUC: {score:.6f}")
    return val_idx, val_preds, score


print(f"Starting training on {N_FOLDS} folds using 2 GPUs...")

results = Parallel(n_jobs=2, backend="threading")(
    delayed(train_fold)(fold, tr, val, X, y, sample_weights, xgb_params)
    for fold, (tr, val) in enumerate(skf.split(X, y))
)

oof_preds = np.zeros(len(X))
fold_aucs = []

for val_idx, preds, score in results:
    oof_preds[val_idx] = preds
    fold_aucs.append(score)

cv_auc = roc_auc_score(y, oof_preds)

print("\n===================================")
print(f"Mean Fold AUC : {np.mean(fold_aucs):.6f}")
print(f"OOF CV AUC    : {cv_auc:.6f}")
print(f"Std Fold AUC  : {np.std(fold_aucs):.6f}")
print("===================================")
print("Block 7 completed âœ”")



# ====================================================
# BLOCK 8: Final Training on Full Data
# ====================================================

FINAL_N_ESTIMATORS = 5000
print(f"Using FINAL_N_ESTIMATORS = {FINAL_N_ESTIMATORS}")

final_xgb_params = xgb_params.copy()
final_xgb_params["n_estimators"] = FINAL_N_ESTIMATORS

final_model = xgb.XGBClassifier(**final_xgb_params)

final_model.fit(
    X,
    y,
    sample_weight=sample_weights,
    verbose=False
)

print("Final anchor model trained âœ”")


# ====================================================
# BLOCK 9: Submission
# ====================================================

X_test = test_df[X.columns]

test_preds = final_model.predict_proba(X_test)[:, 1]

assert not np.isnan(test_preds).any(), "NaNs in predictions!"
assert (test_preds >= 0).all() and (test_preds <= 1).all(), "Preds out of range!"

submission = pd.DataFrame({
    "id": test_df["id"],
    TARGET: test_preds
})

SUBMISSION_PATH = "submission.csv"
submission.to_csv(SUBMISSION_PATH, index=False)

print("Submission file created âœ”")
submission.head()





