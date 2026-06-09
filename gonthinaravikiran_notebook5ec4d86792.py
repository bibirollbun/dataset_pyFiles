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
import re
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
import lightgbm as lgb
import warnings
warnings.filterwarnings("ignore")

# Load data
train = pd.read_csv("/kaggle/input/car-becho-paisa-paao/train.csv")
test = pd.read_csv("/kaggle/input/car-becho-paisa-paao/test.csv")
sample_sub = pd.read_csv("/kaggle/input/car-becho-paisa-paao/sample_submission.csv")
test['price'] = np.nan
data = pd.concat([train, test], ignore_index=True)

# Feature extraction
def extract_engine_features(row):
    s = row.lower()
    hp = re.search(r'(\d+(\.\d+)?)hp', s)
    eng = re.search(r'(\d+(\.\d+)?)l', s)
    cyl = re.search(r'(\d+)\s*(cylinder|v\d|straight \d)', s)
    fuel = re.findall(r'(gasoline|diesel|electric|flex fuel)', s)
    return pd.Series([
        float(hp.group(1)) if hp else np.nan,
        float(eng.group(1)) if eng else np.nan,
        int(cyl.group(1)) if cyl else np.nan,
        fuel[-1] if fuel else 'Unknown'
    ])
data[['hp', 'engine_size', 'cylinders', 'fuel_type_extracted']] = data['engine'].fillna("").apply(extract_engine_features)

# Feature engineering
data['milage'] = data['milage'].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)
data['milage'] = data['milage'].clip(upper=data['milage'].quantile(0.99)).fillna(data['milage'].median())
data['engine_size'] = data['engine_size'].fillna(data['engine_size'].median())
data['hp'] = data['hp'].fillna(data['hp'].median())
data['cylinders'] = data['cylinders'].fillna(data['cylinders'].median())
data['car_age'] = data['model_year'].max() - data['model_year']
data['brand_avg_price'] = data.groupby('brand')['price'].transform('mean')
data['brand_age_interaction'] = data['car_age'] * data.groupby('brand')['car_age'].transform('mean')
data['model_freq'] = data['model'].map(data['model'].value_counts())
data['color_match'] = (data['ext_col'] == data['int_col']).astype(int)
data['is_auto_trans'] = data['transmission'].str.lower().str.contains('auto').astype(int)
data['age_bin'] = pd.cut(data['car_age'], bins=[-1, 3, 7, 12, 25], labels=False)
common_colors = ['black', 'white', 'grey', 'silver', 'blue', 'red']
for color in common_colors:
    data[f'ext_col_{color}'] = data['ext_col'].str.lower().str.contains(color).astype(int)

# Encode categorical
cat_cols = ['brand', 'model', 'fuel_type', 'fuel_type_extracted', 'transmission',
            'ext_col', 'int_col', 'accident', 'clean_title']
for col in cat_cols:
    data[col] = data[col].astype(str).fillna("Missing")
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])

# Split back
train_data = data[~data['price'].isna()].copy()
test_data = data[data['price'].isna()].copy()
features = [col for col in train_data.columns if col not in ['id', 'price', 'engine']]
X = train_data[features]
y = train_data['price']
X_test = test_data[features]

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
xgb_oof = np.zeros(len(train_data))
cat_oof = np.zeros(len(train_data))
hist_oof = np.zeros(len(train_data))
lgb_oof = np.zeros(len(train_data))

xgb_preds = np.zeros(len(X_test))
cat_preds = np.zeros(len(X_test))
hist_preds = np.zeros(len(X_test))
lgb_preds = np.zeros(len(X_test))

# Train base models
for train_idx, val_idx in kf.split(X):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    xgb = XGBRegressor(n_estimators=1000, learning_rate=0.09, max_depth=8,
                       subsample=0.8, colsample_bytree=0.4, random_state=42)
    xgb.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
    xgb_oof[val_idx] = xgb.predict(X_val)
    xgb_preds += xgb.predict(X_test) / kf.n_splits

    cat = CatBoostRegressor(iterations=1000, depth=9, learning_rate=0.07,
                            verbose=False, random_seed=42)
    cat.fit(X_tr, y_tr, eval_set=(X_val, y_val), use_best_model=True)
    cat_oof[val_idx] = cat.predict(X_val)
    cat_preds += cat.predict(X_test) / kf.n_splits

    hist = HistGradientBoostingRegressor(max_iter=1000, learning_rate=0.08, max_depth=8, random_state=42)
    hist.fit(X_tr, y_tr)
    hist_oof[val_idx] = hist.predict(X_val)
    hist_preds += hist.predict(X_test) / kf.n_splits

    lgbm = lgb.LGBMRegressor(n_estimators=1000, learning_rate=0.07, max_depth=8, random_state=42)
    lgbm.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], early_stopping_rounds=20, verbose=False)
    lgb_oof[val_idx] = lgbm.predict(X_val)
    lgb_preds += lgbm.predict(X_test) / kf.n_splits

# Weighted stacking
stacked_train = np.vstack([
    xgb_oof * 0.2,
    cat_oof * 0.3,
    hist_oof * 0.3,
    lgb_oof * 0.2
]).T

stacked_test = np.vstack([
    xgb_preds * 0.2,
    cat_preds * 0.3,
    hist_preds * 0.3,
    lgb_preds * 0.2
]).T

# Meta-model
meta_model = Ridge(alpha=1.0)
meta_model.fit(stacked_train, y)

val_preds = meta_model.predict(stacked_train)
final_preds = meta_model.predict(stacked_test)

cv_rmse = mean_squared_error(y, val_preds, squared=False)
print("Stacking RMSE (2:3:3:2 + Ridge):", round(cv_rmse, 2))

sample_sub['price'] = final_preds
sample_sub.to_csv("submission.csv", index=False)
print("Submission file created!")


