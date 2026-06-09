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


import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error
from catboost import CatBoostRegressor, Pool

BASE_PATH = "/kaggle/input/rent-prediction-2025"

train = pd.read_csv(f"{BASE_PATH}/airbnb_train.csv")
test = pd.read_csv(f"{BASE_PATH}/airbnb_test.csv")
sample_sub = pd.read_csv(f"{BASE_PATH}/airbnb_sample_submission.csv")


def add_features(df):
    df = df.copy()
    bool_map = {"t":1,"f":0,"True":1,"False":0,"TRUE":1,"FALSE":0}
    bool_cols = ["host_is_superhost","host_has_profile_pic","host_identity_verified","has_availability"]
    for c in bool_cols:
        if c in df.columns:
            df[c] = df[c].map(bool_map).astype("float32")
    for c in ["host_response_rate","host_acceptance_rate"]:
        if c in df.columns:
            df[c] = df[c].str.rstrip("%").replace("",np.nan).astype(float)/100
    text_cols = ["name","description","neighborhood_overview","host_about"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].fillna("")
            df[f"{c}_len"] = df[c].str.len()
            df[f"{c}_words"] = df[c].str.split().str.len().fillna(0)
    if "amenities" in df.columns:
        am = df["amenities"].fillna("")
        df["amenities_count"] = np.where(am=="",0,am.str.count(",")+1)
        keys = ["Wifi","Kitchen","Air conditioning","Free parking","Washer","Dryer","TV","Heating","Pool","Breakfast"]
        for k in keys:
            df["amenity_"+k.lower().replace(" ","_")] = am.str.contains(k,case=False,na=False).astype(int)
    if "host_verifications" in df.columns:
        hv = df["host_verifications"].fillna("")
        df["host_verifications_count"] = np.where(hv=="",0,hv.str.count(",")+1)
        for k in ["email","phone","identity","work_email"]:
            df["verification_"+k] = hv.str.contains(k,case=False,na=False).astype(int)
    date_cols = ["host_since","first_review","last_review"]
    for c in date_cols:
        if c in df.columns:
            dt = pd.to_datetime(df[c],errors="coerce")
            df[f"{c}_year"] = dt.dt.year
            df[f"{c}_month"] = dt.dt.month
            df[f"{c}_day"] = dt.dt.day
            mx = dt.max()
            df[f"{c}_recency_days"] = (mx - dt).dt.days
    return df


full = pd.concat([train.drop(columns=["price"]), test], axis=0, ignore_index=True)
full_fe = add_features(full)
train_fe = full_fe.iloc[:len(train)].copy()
test_fe = full_fe.iloc[len(train):].copy()
train_fe["price"] = train["price"].astype(float)

TARGET = "price"
text_cols = ["name","description","neighborhood_overview","host_about","amenities","host_verifications"]
text_cols = [c for c in text_cols if c in train_fe.columns]
feature_cols = [c for c in train_fe.columns if c != TARGET]
cat_cols = [c for c in feature_cols if train_fe[c].dtype=="object" and c not in text_cols]
num_cols = [c for c in feature_cols if c not in cat_cols + text_cols]


for c in num_cols:
    train_fe[c] = train_fe[c].astype(float)
    test_fe[c] = test_fe[c].astype(float)
    m = train_fe[c].median()
    train_fe[c] = train_fe[c].fillna(m)
    test_fe[c] = test_fe[c].fillna(m)

for c in cat_cols + text_cols:
    train_fe[c] = train_fe[c].fillna("NA").astype(str)
    test_fe[c] = test_fe[c].fillna("NA").astype(str)


X = train_fe[feature_cols]
X_test = test_fe[feature_cols]
y = train_fe[TARGET].values
y_log = np.log1p(y)

cat_idx = [X.columns.get_loc(c) for c in cat_cols]
text_idx = [X.columns.get_loc(c) for c in text_cols]

N_SPLITS = 5
groups = train["city"].astype("category").cat.codes.values
gkf = GroupKFold(n_splits=N_SPLITS)

oof_log = np.zeros(len(train))
test_pred_log = np.zeros(len(test))

for f, (tr_idx, val_idx) in enumerate(gkf.split(X, y_log, groups), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y_log[tr_idx], y_log[val_idx]

    train_pool = Pool(X_tr, y_tr, cat_features=cat_idx, text_features=text_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx, text_features=text_idx)
    test_pool = Pool(X_test, cat_features=cat_idx, text_features=text_idx)

    model = CatBoostRegressor(
        loss_function="RMSE",
        eval_metric="RMSE",
        learning_rate=0.05,
        depth=8,
        l2_leaf_reg=4,
        random_state=42+f,
        iterations=5000,
        od_type="Iter",
        od_wait=200,
        task_type="CPU",
        verbose=200
    )

    model.fit(train_pool, eval_set=val_pool)
    val_pred_log = model.predict(val_pool)
    oof_log[val_idx] = val_pred_log
    fold_rmse = np.sqrt(mean_squared_error(np.expm1(y_val), np.expm1(val_pred_log)))
    print("Fold", f, "RMSE:", fold_rmse)
    test_pred_log += model.predict(test_pool) / N_SPLITS

oof = np.expm1(oof_log)
cv_rmse = np.sqrt(mean_squared_error(y, oof))
print("Overall CV RMSE:", cv_rmse)


test_price = np.expm1(test_pred_log)
test_price = np.clip(test_price, 1, 1000)

submission = pd.DataFrame({"id": test["id"], "price": test_price})
submission.to_csv("submission.csv", index=False)
submission.head()




