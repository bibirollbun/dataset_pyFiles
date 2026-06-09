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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
print("Train shape:", train.shape, " | Test shape:", test.shape)


train.head()


# Target Distribution
plt.figure(figsize=(6,4))
sns.histplot(train["accident_risk"], bins=50, kde=True)
plt.title("Target Distribution - Accident Risk")
plt.xlabel("accident_risk")
plt.ylabel("Count")
plt.show()


missing = train.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    print("Missing values:\n", missing)
else:
    print("No missing values ✅")


# Feature overview
print("\nNumerical columns:", train.select_dtypes(include=np.number).columns.tolist()[:10])
print("Categorical columns:", train.select_dtypes(exclude=np.number).columns.tolist()[:10])


TARGET = "accident_risk"
IDCOL = "id"


def add_basic_features(df):
    df = df.copy()
    df["risk_per_lane"] = df["num_reported_accidents"] / (df["num_lanes"] + 1)
    df["speed_curvature_ratio"] = df["speed_limit"] / (df["curvature"] + 0.01)
    df["is_high_speed"] = (df["speed_limit"] >= 60).astype(int)
    df["is_night"] = (df["time_of_day"] == "night").astype(int)
    df["rush_hour"] = df["time_of_day"].isin(["morning", "evening"]).astype(int)
    df["school_zone_active"] = (df["school_season"].astype(int) * df["public_road"].astype(int)).astype(int)
    df["curvature_category"] = pd.cut(df["curvature"], bins=[-1, 0.15, 0.4, 1.0], labels=["low", "medium", "high"])
    return df


train_fe = add_basic_features(train)
test_fe = add_basic_features(test)


group_cols = [("road_type",), ("weather",), ("lighting",), ("road_type","weather"), ("lighting","weather")]
for gs in group_cols:
    name = "_".join(gs)
    col = f"avg_risk_by_{name}"
    means = train_fe.groupby(list(gs))[TARGET].mean()
    train_fe[col] = train_fe.set_index(list(gs)).index.map(means).values
    test_fe[col]  = test_fe.set_index(list(gs)).index.map(means).values
    train_fe[col].fillna(train_fe[TARGET].mean(), inplace=True)
    test_fe[col].fillna(train_fe[TARGET].mean(), inplace=True)


train_fe["speed_bucket"] = (train_fe["speed_limit"] // 10).astype(str)
test_fe["speed_bucket"]  = (test_fe["speed_limit"] // 10).astype(str)
train_fe["speed_weather"] = train_fe["speed_bucket"] + "_" + train_fe["weather"].astype(str)
test_fe["speed_weather"]  = test_fe["speed_bucket"] + "_" + test_fe["weather"].astype(str)


cat_cols = ["road_type", "lighting", "weather", "time_of_day", "curvature_category", "speed_weather"]
for c in cat_cols:
    train_fe[c] = train_fe[c].astype(str)
    test_fe[c]  = test_fe[c].astype(str)

exclude = [IDCOL, TARGET] + cat_cols
num_cols = [c for c in train_fe.columns if c not in exclude]

X = train_fe.drop(columns=[IDCOL, TARGET])
y = train_fe[TARGET]
X_test = test_fe.drop(columns=[IDCOL])

preprocessor = ColumnTransformer([
    ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False), cat_cols),
    ("num", "passthrough", num_cols)
])

preprocessor.fit(X)
X_all = preprocessor.transform(X)
X_test_all = preprocessor.transform(X_test)
print("Preprocessing complete")


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


from lightgbm import early_stopping, log_evaluation

n_splits = 5
try:
    y_bins = pd.qcut(y, q=10, labels=False, duplicates='drop')
except:
    y_bins = pd.cut(y, bins=10, labels=False)

skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

oof_preds = np.zeros((len(X), 3))
test_preds_folds = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_bins), 1):
    print(f"\n--- Fold {fold}/{n_splits} ---")
    X_tr_df, X_val_df = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
    
    X_tr, X_val = preprocessor.transform(X_tr_df), preprocessor.transform(X_val_df)
    
    # --- LightGBM ---
    lgbm = LGBMRegressor(
        n_estimators=1200,
        learning_rate=0.03,
        num_leaves=127,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    lgbm.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric="rmse",
        callbacks=[early_stopping(100), log_evaluation(period=0)]
    )
    lgb_val = lgbm.predict(X_val)
    
    # --- XGBoost ---
    xgb = XGBRegressor(
        n_estimators=1000,
        learning_rate=0.03,
        max_depth=8,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:squarederror',
        random_state=RANDOM_STATE,
        n_jobs=4,
        verbosity=0
    )
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=100, verbose=False)
    xgb_val = xgb.predict(X_val)
    
    # --- CatBoost ---
    cat = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.03,
        depth=8,
        random_seed=RANDOM_STATE,
        verbose=0,
        early_stopping_rounds=100
    )
    cat.fit(X_tr_df, y_tr, cat_features=cat_cols, eval_set=(X_val_df, y_val))
    cat_val = cat.predict(X_val_df)
    
    # Store OOF and test predictions
    oof_preds[val_idx, :] = np.vstack([lgb_val, xgb_val, cat_val]).T
    test_preds_folds.append(np.vstack([
        lgbm.predict(X_test_all),
        xgb.predict(X_test_all),
        cat.predict(X_test)
    ]).T)
    
    print(f"Fold RMSEs: "
          f"LGB {mean_squared_error(y_val,lgb_val,squared=False):.4f} | "
          f"XGB {mean_squared_error(y_val,xgb_val,squared=False):.4f} | "
          f"CAT {mean_squared_error(y_val,cat_val,squared=False):.4f}")


oof_df = pd.DataFrame(oof_preds, columns=["lgb","xgb","cat"])
meta = RidgeCV(alphas=np.logspace(-3,3,13), scoring="neg_root_mean_squared_error", cv=5)
meta.fit(oof_df, y)

meta_oof = meta.predict(oof_df)
meta_rmse = mean_squared_error(y, meta_oof, squared=False)
print("Meta-model RMSE:", meta_rmse)

coefs = meta.coef_ / meta.coef_.sum()
print("Model Weights (normalized):", dict(zip(["LGB","XGB","CAT"], coefs.round(3))))

test_mean = np.mean(test_preds_folds, axis=0)
test_df = pd.DataFrame(test_mean, columns=["lgb","xgb","cat"])
final_preds = np.clip(meta.predict(test_df), 0, 1)

submission = pd.DataFrame({"id": test["id"], "accident_risk": final_preds})
submission.to_csv("/kaggle/working/submission.csv", index=False)
print("Saved submission.csv")
submission.head()


# Feature importance (from LGBM)
importances = pd.Series(lgbm.feature_importances_, index=preprocessor.get_feature_names_out())
top_features = importances.sort_values(ascending=False).head(15)

plt.figure(figsize=(8,4))
sns.barplot(x=top_features.values, y=top_features.index)
plt.title("Top 15 Important Features (LGBM)")
plt.show()




