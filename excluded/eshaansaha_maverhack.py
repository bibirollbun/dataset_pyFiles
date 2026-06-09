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


import os
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# =============================================================
# PureÂ XGBoost lapâ€‘time predictor (single split, early stopping)
# =============================================================
!pip install --quiet xgboost==2.0.3 pandas==2.2.2

import pandas as pd, numpy as np, os, xgboost as xgb, time, gc
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# â”€â”€ 1.  Paths & column names â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
DATA_DIR = "/kaggle/input/burnout-datathon-ieeecsmuj"   
TARGET   = "Lap_Time_Seconds"                # â¬…ï¸� lapâ€‘time column
ID_COL   = "Unique ID"                       # â¬…ï¸� ID column for submission

train_df = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_df  = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
print("Train shape:", train_df.shape, "Â Â Test shape:", test_df.shape)



# â”€â”€ 2.  Minimal feature engineering (checks each column) â”€â”€â”€â”€
def add_features(df):
    if {"Circuit_Length_km", "Avg_Speed_kmh"}.issubset(df.columns):
        df["secs_per_km"] = (df["Circuit_Length_km"] / df["Avg_Speed_kmh"]) * 3600
    if {"Track_Temperature_Celsius", "Ambient_Temperature_Celsius"}.issubset(df.columns):
        df["track_minus_air"] = df["Track_Temperature_Celsius"] - df["Ambient_Temperature_Celsius"]
    if "Track_Condition" in df.columns:
        df["is_wet"] = df["Track_Condition"].str.contains("Wet", case=False, na=False).astype(int)
    if {"Tire_Compound_Front", "Tire_Compound_Rear"}.issubset(df.columns):
        df["tire_combo"] = df["Tire_Compound_Front"].fillna("") + "_" + df["Tire_Compound_Rear"].fillna("")
    return df

train_df = add_features(train_df);  test_df = add_features(test_df)



# â”€â”€ 3.  Convert position numeric â†’ categorical string (if present) â”€â”€
if "position" in train_df.columns:
    train_df["Position_cat"] = train_df["position"].astype(str)
    test_df["Position_cat"]  = test_df["position"].astype(str)
    train_df.drop(columns=["position"], inplace=True)
    test_df.drop(columns=["position"],  inplace=True)


# â”€â”€ 4.  Identify categorical & numeric columns robustly â”€â”€â”€â”€â”€â”€
cat_cols = [c for c in train_df.columns
            if train_df[c].dtype == "object" or train_df[c].dtype.name == "category"]
num_cols = [c for c in train_df.columns if c not in cat_cols + [TARGET]]

# Keep only cats present in both frames
cat_cols = [c for c in cat_cols if c in test_df.columns]

# Fill NaNs
train_df[cat_cols] = train_df[cat_cols].fillna("missing")
test_df[cat_cols]  = test_df[cat_cols].fillna("missing")
for col in num_cols:
    med = train_df[col].median()
    train_df[col].fillna(med, inplace=True)
    test_df[col].fillna(med,  inplace=True)

# Cast categorical dtype for XGB native cats
for c in cat_cols:
    train_df[c] = train_df[c].astype("category")
    test_df[c]  = test_df[c].astype("category")

feature_cols = cat_cols + num_cols
print(f"Using {len(feature_cols)} features ({len(cat_cols)} categorical).")


# â”€â”€ 5.  Trainâ€‘validation split â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
X_train, X_val, y_train, y_val = train_test_split(
    train_df[feature_cols], train_df[TARGET],
    test_size=0.2, random_state=42
)

dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dval   = xgb.DMatrix(X_val,   label=y_val,   enable_categorical=True)
dtest  = xgb.DMatrix(test_df[feature_cols], enable_categorical=True)


# â”€â”€ 6.  XGBoost parameters & training â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
params = {
    "objective": "reg:squarederror",
    "eval_metric": "rmse",
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 10,
    "lambda": 3.0,
    "tree_method": "hist",          # â�¡ â€œgpu_histâ€� if you enabled GPU runtime
    "enable_categorical": True,
    "seed": 42,
}

start = time.time()
model = xgb.train(
    params,
    dtrain,
    num_boost_round=5000,
    evals=[(dval, "val")],
    early_stopping_rounds=200,
    verbose_eval=250
)
print(f"â�±ï¸�  Training time: {time.time()-start:.1f}â€¯s")
print("ğŸ”� Best iteration:", model.best_iteration)
print("ğŸ�� BestÂ valÂ RMSE:", model.best_score)


# â”€â”€ 7.  Predict on test & create submission â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
test_preds = model.predict(dtest, iteration_range=(0, model.best_iteration + 1))
submission = pd.DataFrame({ID_COL: test_df[ID_COL], TARGET: test_preds})
submission.to_csv("submission_xgb.csv", index=False)
print("âœ… submission_xgb.csv saved:", submission.shape)
submission.head()


import os
print("Files in /kaggle/working/:", os.listdir("/kaggle/working/"))



import os

# List all files in the current directory
print("Files in working directory:", os.listdir())



import os

# Rename the file from old name to new name
os.rename("submission_xgb.csv", "solution.csv")

# Check to confirm
print("âœ… Renamed file:")
print(os.listdir("/kaggle/working/"))





