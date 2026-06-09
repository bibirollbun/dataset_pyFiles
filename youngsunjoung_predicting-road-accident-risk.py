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


train = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')

train


train.isna().sum()


test.isna().sum()


num_cols = train.select_dtypes(include="number").columns.to_list()
cat_cols = train.select_dtypes(exclude="number").columns.to_list()


train.corr(numeric_only=True)['accident_risk'].sort_values(ascending=False)


from scipy.stats import f_oneway
for c in cat_cols:
    if train[c].nunique() > 1:
        groups = [g['accident_risk'].values for _, g in train.groupby(c)]
        stat, p = f_oneway(*groups)
        print(c, p)


features = [
    'curvature', 'speed_limit', 'num_reported_accidents',
    'road_type', 'lighting', 'weather', 'public_road', 
    'time_of_day', 'holiday'
]

train["curv_speed"] = train["curvature"] * train["speed_limit"]
test["curv_speed"]  = test["curvature"]  * test["speed_limit"]

train["is_night"] = ((train["time_of_day"]=="Night") & (train["lighting"]=="None")).astype(int)
test["is_night"]  = ((test["time_of_day"]=="Night")  & (test["lighting"]=="None")).astype(int)

features += ['curv_speed', 'is_night']


from sklearn.preprocessing import LabelEncoder

cat_cols = ["road_type","lighting","weather","public_road","time_of_day","holiday"]
for c in cat_cols:
    le = LabelEncoder()
    train[c] = le.fit_transform(train[c].astype(str))
    test[c]  = le.transform(test[c].astype(str))

train.head()


from sklearn.model_selection import KFold
import xgboost as xgb

print(f"XGBoost version {xgb.__version__}")


X = train[features]
y = train["accident_risk"]
X_test = test[features]


from sklearn.metrics import mean_squared_error

kf = KFold(n_splits=5, shuffle=True, random_state=42)

params = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=42,
    objective="reg:squarederror",
    tree_method="hist",
)

oof = np.zeros(len(train))
preds = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[trn_idx], y.iloc[val_idx]

    model = xgb.XGBRegressor(**params)
    model.fit(X_tr, y_tr, verbose=False)

    oof[val_idx] = model.predict(X_val)
    preds += model.predict(X_test) / kf.n_splits

cv_rmse = mean_squared_error(y, oof)
print(f"CV RMSE: {cv_rmse:.6f}")


residual = y - oof

kf = KFold(n_splits=5, shuffle=True, random_state=42)

params2 = dict(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    random_state=2025,
    objective="reg:squarederror",
    tree_method="hist",
)

oof_res = np.zeros(len(train))
preds_res = np.zeros(len(test))

for fold, (trn_idx, val_idx) in enumerate(kf.split(X), 1):
    X_tr, X_val = X.iloc[trn_idx], X.iloc[val_idx]
    r_tr, r_val = residual.iloc[trn_idx], residual.iloc[val_idx]

    model2 = xgb.XGBRegressor(**params2)
    model2.fit(X_tr, r_tr, verbose=False)

    oof_res[val_idx] = model2.predict(X_val)
    preds_res += model2.predict(X_test) / kf.n_splits

final_oof = oof + oof_res
final_preds = preds + preds_res

rmse2 = mean_squared_error(y, final_oof)
print(f"Stage2 CV RMSE: {rmse2:.6f}")
print(f"Gain from residual learning: {np.sqrt(((oof - y)**2).mean()) - rmse2:+.6f}")


# 제출용 DataFrame 생성
submission = pd.DataFrame({
    "id": test["id"],
    "accident_risk": final_preds
})

# CSV로 저장
submission.to_csv("submission.csv", index=False)
print(submission.head())

