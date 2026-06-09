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


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
sample = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")


train.shape, test.shape


train.columns


train.describe()


print(f"Target range: [{train['accident_risk'].min()}, {train['accident_risk'].max()}]")
print(f"Target mean: {train['accident_risk'].mean()}")
print(f"Missing ; Train: {train.isnull().sum().sum()}, Test: {test.isnull().sum().sum()}")


train = train.drop('id', axis=1)
test = test.drop('id', axis=1)


num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = train.select_dtypes(include=['object']).columns.tolist()

num_cols, cat_cols


num_cols.remove('accident_risk')


from sklearn.preprocessing import LabelEncoder
# encoding cat. feat.
train_tmp = train.copy()
test_tmp = test.copy()

for col in cat_cols:
    le = LabelEncoder()
    combined = pd.concat([train_tmp[col], test_tmp[col]]).astype(str)
    le.fit(combined)
    train_tmp[col] = le.transform(train_tmp[col].astype(str)).astype(int)
    test_tmp[col] = le.transform(test_tmp[col].astype(str)).astype(int)

features = num_cols + cat_cols
Xtr = train_tmp[features]
ytr = train_tmp['accident_risk']
Xte = test_tmp[features]


X_tr, X_val, y_tr, y_val = train_test_split(Xtr, ytr, test_size=0.15, random_state=42)


# XGBoost
xgb_model = XGBRegressor(
    n_estimators=900,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    tree_method='hist',
)
xgb_model.fit(X_tr, y_tr)
pred_val_xgb = xgb_model.predict(X_val)


lgb_model = LGBMRegressor(
    n_estimators=800,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
)
lgb_model.fit(X_tr, y_tr)
pred_val_lgb = lgb_model.predict(X_val)


catb_model = CatBoostRegressor(
    iterations=700,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,  
    rsm=0.8,        
    random_state=42,
    verbose=0
)
catb_model.fit(X_tr, y_tr)
pred_val_catb = catb_model.predict(X_val)


ensemble = (pred_val_lgb + pred_val_xgb + pred_val_catb) / 3
ensemble = np.clip(ensemble, 0, 1)

err = mean_squared_error(y_val, ensemble, squared=False)
err


lgb_model.fit(Xtr, ytr)
xgb_model.fit(Xtr, ytr)
catb_model.fit(Xtr, ytr)

preds_test = (
    lgb_model.predict(Xte) +  
    xgb_model.predict(Xte) +
    catb_model.predict(Xte)) / 3  

preds_test = np.clip(preds_test, 0, 1)


preds_test


sample["accident_risk"] = preds_test
sample.to_csv("submission_ensemble.csv", index=False)

print("Submission saved as 'submission.csv'")
print(sample.head(10))

