# -----------------------------  LightGBM – full pipeline  -----------------------------
from __future__ import annotations
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# 1 · Config & paths
# ---------------------------------------------------------------------------
DATA_DIR = Path("/kaggle/input/playground-series-s5e5")  # adjust if running locally
SEED     = 42
TEST_SZ  = 0.20        # validation share
EARLY_STOP = 400       # early-stopping rounds

# ---------------------------------------------------------------------------
# 2 · Feature engineering helper
# ---------------------------------------------------------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add BMI and a few domain-inspired interaction / ratio features."""
    df = df.copy()
    df["BMI"]            = df["Weight"] / (df["Height"] / 100) ** 2
    df["Weight_Height"]  = df["Weight"] * df["Height"]
    df["Duration_HR"]    = df["Duration"] * df["Heart_Rate"]
    df["Age_HR_ratio"]   = df["Heart_Rate"] / df["Age"]
    df["Age_BT_ratio"]   = df["Body_Temp"] / df["Age"]
    df["HR_BT_ratio"]    = df["Heart_Rate"] / df["Body_Temp"]
    return df

# ---------------------------------------------------------------------------
# 3 · Load data & create train/validation split
# ---------------------------------------------------------------------------
train = pd.read_csv(DATA_DIR / "train.csv")
test  = pd.read_csv(DATA_DIR / "test.csv")

train = add_features(train)
test  = add_features(test)

# Separate target and features
X_full = train.drop(columns=["Calories"])
y_full = train["Calories"]

# Simple random split (stratification not needed for regression)
X_tr, X_val, y_tr, y_val = train_test_split(
    X_full, y_full, test_size=TEST_SZ, random_state=SEED
)

# ---------------------------------------------------------------------------
# 4 · Categorical preprocessing
# ---------------------------------------------------------------------------
# 4-a  Identify object columns
obj_cols = X_tr.select_dtypes(include="object").columns.tolist()

# 4-b  Binary map for 'Sex' if present
if "Sex" in obj_cols:
    sex_map = {"male": 0, "female": 1}
    for df in (X_tr, X_val, test):
        df["Sex"] = df["Sex"].map(sex_map)
    obj_cols.remove("Sex")               # handled, no longer an object col

# 4-c  One-hot encode any *remaining* object cols
if obj_cols:
    X_tr  = pd.get_dummies(X_tr,  columns=obj_cols)
    X_val = pd.get_dummies(X_val, columns=obj_cols)
    test  = pd.get_dummies(test,  columns=obj_cols)

    # Align columns so train / val / test match exactly
    X_tr,  X_val = X_tr.align(X_val, join="left", axis=1, fill_value=0)
    X_tr,  test  = X_tr.align(test,  join="left", axis=1, fill_value=0)

# 4-d  LightGBM categorical column list
cat_cols = ["Sex"] if "Sex" in X_tr.columns else []   # names, not indices

# ---------------------------------------------------------------------------
# 5 · Build LightGBM datasets (note: log1p target stabilises variance)
# ---------------------------------------------------------------------------
lgb_tr  = lgb.Dataset(X_tr,  label=np.log1p(y_tr), categorical_feature=cat_cols or "auto")
lgb_val = lgb.Dataset(X_val, label=np.log1p(y_val), categorical_feature=cat_cols or "auto")

# ---------------------------------------------------------------------------
# 6 · Hyper-parameters & training
# ---------------------------------------------------------------------------
params = {
    "objective"      : "regression",
    "metric"         : "rmse",
    "verbosity"      : -1,
    "boosting_type"  : "gbdt",
    "learning_rate"  : 0.05,
    "num_leaves"     : 31,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq"   : 1,
    "random_state"   : SEED,
}

model = lgb.train(
    params,
    lgb_tr,
    num_boost_round=8000,
    valid_sets=[lgb_val],
    callbacks=[
        lgb.early_stopping(EARLY_STOP, verbose=False),
        lgb.log_evaluation(period=200),
    ],
)

# ---------------------------------------------------------------------------
# 7 · Validation score (RMSLE in original scale)
# ---------------------------------------------------------------------------
val_pred_log = model.predict(X_val, num_iteration=model.best_iteration)
rmsle = np.sqrt(
    mean_squared_log_error(y_val, np.expm1(val_pred_log))
)
print(f"Validation RMSLE: {rmsle:.5f}  ·  best_iter = {model.best_iteration}")

# ---------------------------------------------------------------------------
# 8 · Predict test set & create Kaggle submission
# ---------------------------------------------------------------------------
test_pred = np.expm1(model.predict(test, num_iteration=model.best_iteration))

submission = pd.DataFrame({"id": test["id"].astype(int), "Calories": test_pred})
submission.to_csv("submission.csv", index=False, float_format="%.6f")
print("Saved submission.csv ✅")


