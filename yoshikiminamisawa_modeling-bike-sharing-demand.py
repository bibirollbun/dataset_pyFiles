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

from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import TimeSeriesSplit

# 1. データ読み込み
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv",
                    parse_dates=["datetime"])

train = train.sort_values("datetime").reset_index(drop=True)

# 2. 特徴量作成
for df in [train]:
    df["hour"]    = df["datetime"].dt.hour
    df["day"]     = df["datetime"].dt.day
    df["month"]   = df["datetime"].dt.month
    df["year"]    = df["datetime"].dt.year
    df["weekday"] = df["datetime"].dt.weekday

    # peak_reg（registered用ピーク）
    df["peak_reg"] = 0
    mask_work = (df["workingday"] == 1)
    df.loc[mask_work & df["hour"].isin([7, 8, 9, 16, 17, 18]), "peak_reg"] = 1

    # peak_cas（casual用ピーク）
    df["peak_cas"] = 0
    df.loc[df["hour"].between(11, 17), "peak_cas"] = 1


# 3. windspeed == 0 の補正
from sklearn.ensemble import RandomForestRegressor

def fill_windspeed(df):
    df_w0    = df[df["windspeed"] == 0]
    df_wnot0 = df[df["windspeed"] != 0]

    if len(df_w0) == 0:
        return df

    model_wind = RandomForestRegressor(
        random_state=42, n_estimators=100, n_jobs=-1
    )
    cols = ["season", "weather", "humidity", "temp", "atemp", "month", "year"]
    model_wind.fit(df_wnot0[cols], df_wnot0["windspeed"])
    df_w0.loc[:, "windspeed"] = model_wind.predict(df_w0[cols])

    df = pd.concat([df_wnot0, df_w0]).sort_index()
    return df

train = fill_windspeed(train)

# 4. 特徴量
base_features = [
    "season", "holiday", "workingday", "weather",
    "temp", "atemp", "humidity", "windspeed",
    "year", "hour", "day", "month", "weekday"
]

features_casual = base_features + ["peak_cas"]
features_reg    = base_features + ["peak_reg"]

X_casual = train[features_casual]
X_reg    = train[features_reg]

y_casual_log = np.log1p(train["casual"])
y_reg_log    = np.log1p(train["registered"])

# TimeSeriesSplit（時系列を保った5分割）
tscv = TimeSeriesSplit(n_splits=5)

# RMSLE
def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))

# casual / registered を別々に学習してから合算して RMSLE を計算
def evaluate_pair(model_casual, model_reg, name):
    rmsles = []

    for train_idx, val_idx in tscv.split(train):
        # ====== 学習・検証分割 ======
        Xc_tr, Xc_val = X_casual.iloc[train_idx], X_casual.iloc[val_idx]
        Xr_tr, Xr_val = X_reg.iloc[train_idx],    X_reg.iloc[val_idx]

        yc_tr, yc_val = y_casual_log.iloc[train_idx], y_casual_log.iloc[val_idx]
        yr_tr, yr_val = y_reg_log.iloc[train_idx],    y_reg_log.iloc[val_idx]

        # ====== casual モデル学習 ======
        model_casual.fit(Xc_tr, yc_tr)

        # ====== registered モデル学習 ======
        model_reg.fit(Xr_tr, yr_tr)

        # ====== 予測 ======
        pred_casual     = np.expm1(model_casual.predict(Xc_val))
        pred_registered = np.expm1(model_reg.predict(Xr_val))

        y_true = np.expm1(yc_val) + np.expm1(yr_val)
        y_pred = np.clip(pred_casual + pred_registered, 0, None)

        rmsles.append(rmsle(y_true, y_pred))

    score = np.mean(rmsles)
    print(f"{name:20s}  RMSLE = {score:.5f}")

    return score


from sklearn.ensemble import RandomForestRegressor

rf_casual = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

rf_reg = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    random_state=42,
    n_jobs=-1
)

evaluate_pair(rf_casual, rf_reg, "RandomForest")


from sklearn.ensemble import GradientBoostingRegressor

gb_casual = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

gb_reg = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

evaluate_pair(gb_casual, gb_reg, "GradientBoosting")


from sklearn.ensemble import ExtraTreesRegressor

et_casual = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

et_reg = ExtraTreesRegressor(
    n_estimators=500,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

evaluate_pair(et_casual, et_reg, "ExtraTrees")



from sklearn.neighbors import KNeighborsRegressor

knn_casual = KNeighborsRegressor(
    n_neighbors=10,
    weights="distance",
    n_jobs=-1
)

knn_reg = KNeighborsRegressor(
    n_neighbors=10,
    weights="distance",
    n_jobs=-1
)

evaluate_pair(knn_casual, knn_reg, "KNN")



from sklearn.linear_model import LinearRegression

lr_casual = LinearRegression()
lr_reg    = LinearRegression()

evaluate_pair(lr_casual, lr_reg, "LinearRegression")



from xgboost import XGBRegressor

xgb_casual = XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    verbosity=0
)

xgb_reg = XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.0,
    reg_lambda=1.0,
    objective="reg:squarederror",
    random_state=42,
    verbosity=0
)

evaluate_pair(xgb_casual, xgb_reg, "XGBoost")



import lightgbm as lgb

lgb_casual = lgb.LGBMRegressor(
    objective="rmse",        # log1p した y を学習して RMSE → RMSLE に近似
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

lgb_reg = lgb.LGBMRegressor(
    objective="rmse",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

evaluate_pair(lgb_casual, lgb_reg, "LightGBM")



from catboost import CatBoostRegressor

cb_casual = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    random_state=42,
    verbose=0
)

cb_reg = CatBoostRegressor(
    iterations=500,
    depth=6,
    learning_rate=0.05,
    loss_function="RMSE",
    random_state=42,
    verbose=0
)

evaluate_pair(cb_casual, cb_reg, "CatBoost")



from itertools import product
import lightgbm as lgb
import pandas as pd

#========================================================
# 1. 基本パラメータ
#========================================================
base_params = dict(
    objective="rmse",
    num_leaves=31,
    learning_rate=0.05,
    n_estimators=500,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.0,
    reg_lambda=0.1,
    random_state=42,
    verbose=-1    # ← LightGBM のログを抑制
)

#========================================================
# 2. 試すパラメータの候補
#========================================================
num_leaves_list = [8, 12, 20, 24]
learning_rate_list = [0.03, 0.04, 0.05]
n_estimators_map = {
    0.03: 1500,
    0.04: 1000,
    0.05: 500
}
min_child_list = [20, 30]
reg_lambda_list = [0.1, 0.3]

# 特別追加
special_param = dict(
    num_leaves=64,
    learning_rate=0.03,
    n_estimators=1500,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.0,
    reg_lambda=0.1,
    random_state=42,
    objective="rmse",
    verbose=-1
)

#========================================================
# 3. グリッド生成
#========================================================
param_grid = []

for nl, lr, mcs, rl in product(num_leaves_list, learning_rate_list, min_child_list, reg_lambda_list):
    params = base_params.copy()
    params["num_leaves"] = nl
    params["learning_rate"] = lr
    params["n_estimators"] = n_estimators_map[lr]
    params["min_child_samples"] = mcs
    params["reg_lambda"] = rl
    param_grid.append(params)

param_grid.append(special_param)

print(f"総パラメータセット: {len(param_grid)} 個\n")


#========================================================
# 4. 全パラメータ評価
#========================================================
results = []

for i, params in enumerate(param_grid):
    print(f"({i+1}/{len(param_grid)}) evaluating ...")

    model_casual = lgb.LGBMRegressor(**params)
    model_reg = lgb.LGBMRegressor(**params)

    score = evaluate_pair(model_casual, model_reg, f"LGB {i+1}")

    # None（評価失敗）はスキップ
    results.append((score, params))


#========================================================
# 5. スコア順に並べて上位表示
#========================================================
clean_results = [r for r in results if r[0] is not None]
sorted_results = sorted(clean_results, key=lambda x: x[0])

print("\n===== BEST RESULT =====")
print("Score =", sorted_results[0][0])
print("Params =", sorted_results[0][1])


#========================================================
# 6. 全結果を表形式で出力
#========================================================
df = pd.DataFrame([
    {
        "score": score,
        **params
    }
    for score, params in clean_results
])

print("\n===== 全パラメータとスコア一覧（スコア昇順） =====")
df_sorted = df.sort_values("score").reset_index(drop=True)
print(df_sorted)



from itertools import product
import lightgbm as lgb
import pandas as pd

#========================================================
# 1. 基本パラメータ（固定部分）
#========================================================
base_params = dict(
    objective="rmse",
    learning_rate=0.05,
    random_state=42,
    verbose=-1
)

#========================================================
# 2. 探索パラメータ（序盤データ不足を考慮して拡張）
#========================================================

num_leaves_list = [12, 20, 24]                       # 小〜中規模
learning_rate_list = [0.03, 0.04]
n_estimators_map = {0.03: 1500, 0.04: 1000}

subsample_list = [0.8, 0.9, 1.0]
colsample_list = [0.8, 0.9, 1.0]

min_child_list = [10, 20, 30]                       
reg_alpha_list = [0.0, 0.1]
reg_lambda_list = [0.0, 0.1, 0.3]

#========================================================
# 3. グリッド生成
#========================================================
param_grid = []

for nl, lr, ss, cs, mcs, ra, rl in product(
    num_leaves_list,
    learning_rate_list,
    subsample_list,
    colsample_list,
    min_child_list,
    reg_alpha_list,
    reg_lambda_list
):
    params = base_params.copy()
    params.update({
        "num_leaves": nl,
        "learning_rate": lr,
        "n_estimators": n_estimators_map[lr],
        "subsample": ss,
        "colsample_bytree": cs,
        "min_child_samples": mcs,
        "reg_alpha": ra,
        "reg_lambda": rl,
    })
    param_grid.append(params)

print(f"総パラメータセット: {len(param_grid)} 個\n")

#========================================================
# 4. 全パラメータを評価（★最小修正：パラメータを表示）
#========================================================
results = []

for i, params in enumerate(param_grid):
    print(f"({i+1}/{len(param_grid)}) evaluating ...")
    print("  Params:", params)

    model_casual = lgb.LGBMRegressor(**params)
    model_reg    = lgb.LGBMRegressor(**params)

    score = evaluate_pair(model_casual, model_reg, f"LGB {i+1}")

    results.append((score, params))


#========================================================
# 5. ベストスコア表示
#========================================================
clean_results = [r for r in results if r[0] is not None]
sorted_results = sorted(clean_results, key=lambda x: x[0])

print("\n===== BEST RESULT =====")
print("Score =", sorted_results[0][0])
print("Params =", sorted_results[0][1])

#========================================================
# 6. 全結果を DataFrame 出力
#========================================================
df = pd.DataFrame([
    {"score": score, **params}
    for score, params in clean_results
])

print("\n===== 全パラメータとスコア一覧（昇順） =====")
df_sorted = df.sort_values("score").reset_index(drop=True)
print(df_sorted)



from catboost import CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd

# --------------------------------------------------------
# 0. LightGBM / CatBoost の最適パラメータ
# --------------------------------------------------------

lgb_best_params = {
    "objective": "rmse",
    "learning_rate": 0.04,
    "num_leaves": 20,
    "n_estimators": 1000,
    "subsample": 1.0,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.0,
    "reg_lambda": 0.0,
    "random_state": 42,
    "verbose": -1
}

cat_params = {
    "learning_rate": 0.05,
    "depth": 7,
    "l2_leaf_reg": 3,
    "iterations": 600,
    "loss_function": "RMSE",
    "random_seed": 42,
    "verbose": False
}

# --------------------------------------------------------
# 1. 試すアンサンブル比率
# --------------------------------------------------------
ratio_list = [
    (0.90, 0.10),
    (0.85, 0.15),
    (0.80, 0.20),
]

# --------------------------------------------------------
# 2. モデル学習用関数（casual / registered を別々に）
# --------------------------------------------------------

def train_lgb_model(X, y):
    model = lgb.LGBMRegressor(**lgb_best_params)
    model.fit(X, y)
    return model

def train_cat_model(X, y):
    model = CatBoostRegressor(**cat_params)
    model.fit(X, y)
    return model

# --------------------------------------------------------
# 3. RMSLE 関数
# --------------------------------------------------------
from sklearn.metrics import mean_squared_log_error

def rmsle(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(y_true, y_pred))


# --------------------------------------------------------
# 4. アンサンブルを評価
# --------------------------------------------------------

def evaluate_ensemble(weight_lgb, weight_cat, name="ensemble"):
    rmsles = []

    for train_idx, val_idx in tscv.split(train):
        # casual
        Xtr_c, Xval_c = X_casual.iloc[train_idx], X_casual.iloc[val_idx]
        ytr_c, yval_c = y_casual_log.iloc[train_idx], y_casual_log.iloc[val_idx]

        model_lgb_c = train_lgb_model(Xtr_c, ytr_c)
        model_cat_c = train_cat_model(Xtr_c, ytr_c)

        pred_lgb_c = np.expm1(model_lgb_c.predict(Xval_c))
        pred_cat_c = np.expm1(model_cat_c.predict(Xval_c))

        pred_c = weight_lgb * pred_lgb_c + weight_cat * pred_cat_c

        # registered
        Xtr_r, Xval_r = X_reg.iloc[train_idx], X_reg.iloc[val_idx]
        ytr_r, yval_r = y_reg_log.iloc[train_idx], y_reg_log.iloc[val_idx]

        model_lgb_r = train_lgb_model(Xtr_r, ytr_r)
        model_cat_r = train_cat_model(Xtr_r, ytr_r)

        pred_lgb_r = np.expm1(model_lgb_r.predict(Xval_r))
        pred_cat_r = np.expm1(model_cat_r.predict(Xval_r))

        pred_r = weight_lgb * pred_lgb_r + weight_cat * pred_cat_r

        # 合算
        y_true = np.expm1(yval_c) + np.expm1(yval_r)
        y_pred = np.clip(pred_c + pred_r, 0, None)

        rmsles.append(rmsle(y_true, y_pred))

    score = np.mean(rmsles)
    print(f"{name:15s}  RMSLE = {score:.5f}")
    return score


# --------------------------------------------------------
# 5. 全比率を評価して結果まとめ
# --------------------------------------------------------

results = []

for (wl, wc) in ratio_list:
    name = f"LGB {wl:.2f} + CAT {wc:.2f}"
    print(f"\n=== Evaluating {name} ===")
    score = evaluate_ensemble(wl, wc, name)
    results.append((score, wl, wc))

# --------------------------------------------------------
# 6. 結果を DataFrame にまとめる
# --------------------------------------------------------

df = pd.DataFrame([
    {"score": s, "weight_lgb": wl, "weight_cat": wc}
    for (s, wl, wc) in results
])

print("\n===== アンサンブル比較（昇順） =====")
print(df.sort_values("score").reset_index(drop=True))


