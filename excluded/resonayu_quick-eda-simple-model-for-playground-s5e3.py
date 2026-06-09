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
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
import catboost as cb
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

# Find the dataset path
data_dir = None
for root, dirs, files in os.walk("/kaggle/input"):
    if "train.csv" in files and "test.csv" in files:
        data_dir = root
        break

if data_dir is None:
    raise FileNotFoundError("train.csv and test.csv not found in /kaggle/input")

# Load dataset
train_path = os.path.join(data_dir, "train.csv")
test_path = os.path.join(data_dir, "test.csv")
train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

# Check target column
if "target" not in train.columns:
    if "rainfall" in train.columns:
        train.rename(columns={"rainfall": "target"}, inplace=True)
    else:
        raise KeyError("Neither 'target' nor 'rainfall' column found in the dataset.")

# Feature engineering
def feature_engineering(df):
    df["day_of_week"] = df["day"] % 7
    df["month"] = (df["day"] // 30) % 12
    df["season"] = (df["month"] // 3) % 4

    # Handle NaN values before using sin/cos
    df["winddirection"] = df["winddirection"].fillna(0)
    df["wind_sin"] = np.sin(df["winddirection"] * np.pi / 180)
    df["wind_cos"] = np.cos(df["winddirection"] * np.pi / 180)

    df["humidity_dewpoint"] = df["humidity"] * df["dewpoint"]
    return df

train = feature_engineering(train)
test = feature_engineering(test)

# Define features and target
drop_cols = ["id", "target"]
X = train.drop(columns=drop_cols)
y = train["target"]
X_test = test.drop(columns=["id"])

# Scale features
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

# KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
lgb_preds = np.zeros(len(X_test))
xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))

for train_idx, valid_idx in kf.split(X):
    X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

    # LightGBM (Use CPU)
    lgb_model = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.02, max_depth=10, num_leaves=60,
        min_data_in_leaf=40, feature_fraction=0.8, bagging_fraction=0.8, 
        bagging_freq=5, random_state=42, device="cpu"  # Changed to CPU
    )
    lgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], eval_metric="binary_error", callbacks=[
        lgb.early_stopping(50), lgb.log_evaluation(50)])
    lgb_preds += lgb_model.predict_proba(X_test)[:, 1] / kf.n_splits

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=1000, learning_rate=0.02, max_depth=8, random_state=42, 
        use_label_encoder=False, eval_metric="logloss", tree_method="hist"  # Changed to CPU
    )
    xgb_model.fit(X_train, y_train)
    xgb_preds += xgb_model.predict_proba(X_test)[:, 1] / kf.n_splits

    # CatBoost
    cat_model = cb.CatBoostClassifier(
        n_estimators=1000, learning_rate=0.02, depth=8, random_state=42,
        verbose=0, task_type="CPU"  # Changed to CPU
    )
    cat_model.fit(X_train, y_train)
    cat_preds += cat_model.predict_proba(X_test)[:, 1] / kf.n_splits

# Ensemble predictions
final_preds = (lgb_preds * 0.3) + (xgb_preds * 0.45) + (cat_preds * 0.25)
final_preds = (final_preds > 0.5).astype(int)

# Create submission file
submission = pd.DataFrame({"id": test["id"], "target": final_preds})
submission.to_csv("submission.csv", index=False)


