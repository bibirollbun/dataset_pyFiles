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


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Ridge
from scipy.stats import rankdata
from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
from xgboost import XGBRegressor


SEED  = 42
FOLDS = 5
np.random.seed(SEED)

def rmse(y_true, y_pred): 
    return mean_squared_error(y_true, y_pred, squared=False)

def clip01(a): 
    return np.clip(a, 0, 1)


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

y = train["accident_risk"]
X = train.drop(["id", "accident_risk"], axis=1)
X_test = test.drop(["id"], axis=1)

print("Train shape:", X.shape)
print("Test shape:", X_test.shape)


print("\nMissing values (train):\n", X.isnull().sum()[X.isnull().sum() > 0])


sns.histplot(y, bins=40, kde=True)
plt.title("Distribution of Accident Risk")
plt.show()


num_cols = X.select_dtypes(include=["float64", "int64"]).columns
if len(num_cols) > 0:
    plt.figure(figsize=(8,6))
    sns.heatmap(pd.concat([X[num_cols], y], axis=1).corr(), annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Heatmap (Numeric Features)")
    plt.show()


cat_cols = X.select_dtypes(include=["object", "bool", "category"]).columns.tolist()
for c in cat_cols:
    X[c] = X[c].astype(str)
    X_test[c] = X_test[c].astype(str)


for col in X.columns:
    if col not in cat_cols:
        X[col] = pd.to_numeric(X[col], errors="ignore")
        X_test[col] = pd.to_numeric(X_test[col], errors="ignore")


if "road_type" in X.columns and "lighting" in X.columns:
    X["road_lighting"] = X["road_type"] + "_" + X["lighting"]
    X_test["road_lighting"] = X_test["road_type"] + "_" + X_test["lighting"]


bad_weather = {"rain", "snow", "fog", "storm"}
if "weather" in X.columns:
    X["bad_weather_flag"] = X["weather"].str.lower().isin(bad_weather).astype(int)
    X_test["bad_weather_flag"] = X_test["weather"].str.lower().isin(bad_weather).astype(int)

if set(["holiday","school_season"]).issubset(X.columns):
    X["holiday_school_flag"] = ((X["holiday"] == "True") | (X["school_season"] == "True")).astype(int)
    X_test["holiday_school_flag"] = ((X_test["holiday"] == "True") | (X_test["school_season"] == "True")).astype(int)



def _is_floatable(v):
    try:
        float(v); return True
    except:
        return False

if "time_of_day" in X.columns:
    numeric_like_time = X["time_of_day"].map(_is_floatable).mean() > 0.9
    if numeric_like_time:
        X["time_of_day"] = X["time_of_day"].astype(float)
        X_test["time_of_day"] = X_test["time_of_day"].astype(float)
        X["time_sin"] = np.sin(2 * np.pi * X["time_of_day"]/24)
        X["time_cos"] = np.cos(2 * np.pi * X["time_of_day"]/24)
        X_test["time_sin"] = np.sin(2 * np.pi * X_test["time_of_day"]/24)
        X_test["time_cos"] = np.cos(2 * np.pi * X_test["time_of_day"]/24)
        X["rush_hour_flag"] = ((X["time_of_day"].between(7, 9)) | (X["time_of_day"].between(16, 19))).astype(int)
        X_test["rush_hour_flag"] = ((X_test["time_of_day"].between(7, 9)) | (X_test["time_of_day"].between(16, 19))).astype(int)


if "speed_limit" in X.columns and "weather" in X.columns:
    try:
        X["speed_limit"] = X["speed_limit"].astype(float)
        X_test["speed_limit"] = X_test["speed_limit"].astype(float)
        bw_tr = X["weather"].str.lower().isin(bad_weather).astype(int)
        bw_te = X_test["weather"].str.lower().isin(bad_weather).astype(int)
        X["weather_risk"] = X["speed_limit"] * bw_tr
        X_test["weather_risk"] = X_test["speed_limit"] * bw_te
    except:
        pass


if set(["road_type","weather","lighting"]).issubset(X.columns):
    X["danger_combo"] = (X["road_type"] + "_" + X["weather"] + "_" + X["lighting"])
    X_test["danger_combo"] = (X_test["road_type"] + "_" + X_test["weather"] + "_" + X_test["lighting"])


cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
print("Categorical columns used:", cat_cols)
print("Total features after FE:", X.shape[1])



kf = KFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
cat_idx = [X.columns.get_loc(c) for c in cat_cols]

cb_params = dict(
    iterations=10000,
    learning_rate=0.018,
    depth=9,
    l2_leaf_reg=7,
    bagging_temperature=0.4,
    random_strength=0.9,
    loss_function="RMSE",
    random_seed=SEED,
    verbose=500,
    early_stopping_rounds=500
)

oof_cb = np.zeros(len(X))
test_pred_folds_cb = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X, y), 1):
    X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx)
    valid_pool = Pool(X_va, y_va, cat_features=cat_idx)
    test_pool  = Pool(X_test,      cat_features=cat_idx)

    model_cb = CatBoostRegressor(**cb_params)
    model_cb.fit(train_pool, eval_set=valid_pool)

    oof_cb[va_idx] = model_cb.predict(valid_pool)
    print(f"[CatBoost Fold {fold}] RMSE: {rmse(y_va, oof_cb[va_idx]):.5f}")

    test_pred_folds_cb.append(model_cb.predict(test_pool))

oof_cb = clip01(oof_cb)
cv_rmse_cb = rmse(y, oof_cb)
print(f"\n[CatBoost] OOF CV RMSE: {cv_rmse_cb:.5f}")


print("\nRunning LightGBM CV...")

X_lgb = X.copy()
X_test_lgb = X_test.copy()
for c in cat_cols:
    X_lgb[c] = X_lgb[c].astype("category")
    X_test_lgb[c] = X_test_lgb[c].astype("category")

lgb_params = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.02,
    n_estimators=10000,
    num_leaves=80,
    feature_fraction=0.90,
    bagging_fraction=0.90,
    bagging_freq=1,
    reg_alpha=0.3,
    reg_lambda=0.6,
    random_state=SEED,
    verbose=-1
)

oof_lgb = np.zeros(len(X_lgb))
test_pred_folds_lgb = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_lgb, y), 1):
    X_tr, y_tr = X_lgb.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X_lgb.iloc[va_idx], y.iloc[va_idx]

    model_lgb = lgb.LGBMRegressor(**lgb_params)
    model_lgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(stopping_rounds=600, verbose=False)]
    )

    oof_lgb[va_idx] = model_lgb.predict(X_va, num_iteration=model_lgb.best_iteration_)
    print(f"[LightGBM Fold {fold}] RMSE: {rmse(y_va, oof_lgb[va_idx]):.5f}")

    test_pred_folds_lgb.append(model_lgb.predict(X_test_lgb, num_iteration=model_lgb.best_iteration_))

oof_lgb = clip01(oof_lgb)
cv_rmse_lgb = rmse(y, oof_lgb)
print(f"\n[LightGBM] OOF CV RMSE: {cv_rmse_lgb:.5f}")



print("\nRunning XGBoost CV...")

X_xgb = pd.get_dummies(X, columns=cat_cols, drop_first=False)
X_test_xgb = pd.get_dummies(X_test, columns=cat_cols, drop_first=False)
X_xgb, X_test_xgb = X_xgb.align(X_test_xgb, join="left", axis=1, fill_value=0)

xgb_params = dict(
    n_estimators=12000,
    learning_rate=0.028,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=SEED,
    tree_method="hist"
)

oof_xgb = np.zeros(len(X_xgb))
test_pred_folds_xgb = []

for fold, (tr_idx, va_idx) in enumerate(kf.split(X_xgb, y), 1):
    X_tr, y_tr = X_xgb.iloc[tr_idx], y.iloc[tr_idx]
    X_va, y_va = X_xgb.iloc[va_idx], y.iloc[va_idx]

    model_xgb = XGBRegressor(**xgb_params)
    model_xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
        early_stopping_rounds=800
    )

    oof_xgb[va_idx] = model_xgb.predict(X_va)
    print(f"[XGBoost Fold {fold}] RMSE: {rmse(y_va, oof_xgb[va_idx]):.5f}")

    test_pred_folds_xgb.append(model_xgb.predict(X_test_xgb))

oof_xgb = clip01(oof_xgb)
cv_rmse_xgb = rmse(y, oof_xgb)
print(f"\n[XGBoost] OOF CV RMSE: {cv_rmse_xgb:.5f}")



def blend2(a, b, w):          # 2-way linear
    return clip01(w*a + (1-w)*b)

def blend3(a, b, c, w1, w2):  # 3-way linear
    w3 = 1.0 - w1 - w2
    return clip01(w1*a + w2*b + w3*c)

def rank_blend(preds_list, weights=None):  # rank-average (robust)
    ranked = np.vstack([rankdata(p) for p in preds_list]).T
    ranked = ranked / ranked.max(axis=0)
    if weights is None:
        return clip01(np.mean(ranked, axis=1))
    else:
        w = np.array(weights) / np.sum(weights)
        return clip01(ranked @ w)

# refined 2-way sweep (CB vs LGB)
best2, bestw2 = 1e9, 0.5
for w in np.arange(0.35, 0.71, 0.02):
    sc = rmse(y, blend2(oof_cb, oof_lgb, w))
    if sc < best2:
        best2, bestw2 = sc, w
print(f"[CB/LGB] Best 2-way: w={bestw2:.2f}, OOF={best2:.5f}")

# refined 3-way sweep around plausible region
best3, best_w = 1e9, (1/3, 1/3, 1/3)
for w_cb in np.arange(0.30, 0.61, 0.02):
    for w_lgb in np.arange(0.30, 0.61, 0.02):
        if w_cb + w_lgb >= 0.95:
            continue
        oof_bl3 = blend3(oof_cb, oof_lgb, oof_xgb, w_cb, w_lgb)
        sc = rmse(y, oof_bl3)
        if sc < best3:
            best3, best_w = sc, (w_cb, w_lgb, 1.0 - w_cb - w_lgb)
print(f"[CB/LGB/XGB] Best 3-way weights: {best_w} | OOF={best3:.5f}")

# stacking meta-model on OOF preds
stack_train = np.vstack([oof_cb, oof_lgb, oof_xgb]).T
test_pred_cb   = clip01(np.mean(np.column_stack(test_pred_folds_cb),  axis=1))
test_pred_lgb  = clip01(np.mean(np.column_stack(test_pred_folds_lgb), axis=1))
test_pred_xgb  = clip01(np.mean(np.column_stack(test_pred_folds_xgb), axis=1))
stack_test  = np.vstack([test_pred_cb, test_pred_lgb, test_pred_xgb]).T

meta = Ridge(alpha=1.0, random_state=SEED)
meta.fit(stack_train, y)
meta_oof  = clip01(meta.predict(stack_train))
print(f"[Stacking Ridge] OOF RMSE: {rmse(y, meta_oof):.5f}")

# rank-based OOF
rank_oof  = rank_blend([oof_cb, oof_lgb, oof_xgb], weights=[0.4,0.3,0.3])
print(f"[RankBlend OOF] RMSE: {rmse(y, rank_oof):.5f}")



pred_blend2 = blend2(test_pred_cb, test_pred_lgb, bestw2)
w_cb, w_lgb, w_xgb = best_w
pred_blend3 = blend3(test_pred_cb, test_pred_lgb, test_pred_xgb, w_cb, w_lgb)
pred_meta = clip01(meta.predict(stack_test))
pred_rank = rank_blend([test_pred_cb, test_pred_lgb, test_pred_xgb], weights=[0.4,0.3,0.3])

pd.DataFrame({"id": test["id"], "accident_risk": test_pred_cb}).to_csv("submission_catboost_cv.csv", index=False)
pd.DataFrame({"id": test["id"], "accident_risk": test_pred_lgb}).to_csv("submission_lightgbm_cv.csv", index=False)
pd.DataFrame({"id": test["id"], "accident_risk": test_pred_xgb}).to_csv("submission_xgboost_cv.csv", index=False)
pd.DataFrame({"id": test["id"], "accident_risk": pred_blend2}).to_csv("submission_blend_cb_lgb.csv", index=False)
pd.DataFrame({"id": test["id"], "accident_risk": pred_blend3}).to_csv("submission_blend_cb_lgb_xgb.csv", index=False)
pd.DataFrame({"id": test["id"], "accident_risk": pred_meta}).to_csv("submission_meta_ridge.csv", index=False)
pd.DataFrame({"id": test["id"], "accident_risk": pred_rank}).to_csv("submission_rankblend.csv", index=False)

print("\nSaved:")
print(" - submission_catboost_cv.csv")
print(" - submission_lightgbm_cv.csv")
print(" - submission_xgboost_cv.csv")
print(" - submission_blend_cb_lgb.csv")
print(" - submission_blend_cb_lgb_xgb.csv")
print(" - submission_meta_ridge.csv")
print(" - submission_rankblend.csv")

