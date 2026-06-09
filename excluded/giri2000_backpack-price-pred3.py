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


!python --version


# %%time
# !pip install --target=/kaggle/working --extra-index-url=https://pypi.nvidia.com "cudf-cu12==25.2.*" "cuml-cu12==25.2.*"
# !rm -rf /kaggle/working/numpy*


%load_ext cudf.pandas

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_columns', 500)


train = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
train.head()


train.shape


train2 = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")
train2.head()


train2.shape


train = pd.concat([train, train2], axis=0)


train.shape


train['Compartments'].unique()


test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
test.head()


test.shape


CATS = train.columns[1:-2]
CATS





COMBO = []
for i, c in enumerate(CATS):
    combine = pd.concat([train[c], test[c]], axis=0)
    combine,_ = pd.factorize(combine)
    train[c] = combine[:len(train)]
    test[c] = combine[len(train):]
    n = f"{c}_wc"
    train[n] = train[c] * 100 + train["Weight Capacity (kg)"]
    test[n] = test[c] * 100 + test["Weight Capacity (kg)"]
    COMBO.append(n)
print("New Features: ")
print(COMBO)


train.head()


train.isna().sum()


FEATURES = list(CATS) + ['Weight Capacity (kg)'] + list(COMBO)
print(f"Number of features:  {len(FEATURES)}")
print(FEATURES)


from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import xgboost as xgb
print(f"XGBoost version",xgb.__version__)


# STATISTICS TO AGGEGATE FOR OUR FEATURE GROUPS
STATS = ["count","nunique","median","min","max","skew"]
STATS2 = ["mean","std"]


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


X = train.drop(['Price', 'id'], axis=1)
y = train['Price']


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=10)


X_train.shape


X_val.shape


X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=10)


X_val.shape


X_test.shape


X_train.head()





X_train.fillna(X_train.mean(), inplace=True)
X_val.fillna(X_train.mean(), inplace=True)
X_test.fillna(X_train.mean(), inplace=True)


CATS


for col in CATS:
    col2 = "Weight Capacity (kg)"
    tmp = X_train.groupby(col)[col2].agg(STATS2)
    tmp.columns = [f"FE1_{col}_wc_{s}" for s in STATS2]
    X_train = X_train.merge(tmp, on=col, how="left")
    X_val = X_val.merge(tmp, on=col, how="left")
    X_test = X_test.merge(tmp, on=col, how="left")


X_train.shape


X_train.columns


X_val.shape


X_test.shape


model = XGBRegressor(
        device="cuda",
        max_depth=6,  
        colsample_bytree=0.5, 
        subsample=0.8,  
        n_estimators=10_000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=10,
        early_stopping_rounds=100,
    )
model.fit(X_train, y_train, eval_set=[(X_val, y_val)],  
        verbose=300,)


val_prediction = model.predict(X_val)


def root_mean_square_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


print("RMSE for validation data: ",root_mean_square_error(y_val, val_prediction))


test_prediction = model.predict(X_test)


print("RMSE for test data: ",root_mean_square_error(y_test, test_prediction))


test.head()


test.isna().sum()


test.fillna(X_train.mean(), inplace=True)


test.isna().sum()


for col in CATS:
    col2 = "Weight Capacity (kg)"
    tmp = X_train.groupby(col)[col2].agg(STATS2)
    tmp.columns = [f"FE1_{col}_wc_{s}" for s in STATS2]
    test = test.merge(tmp, on=col, how="left")


test.shape


test.drop('id', axis=1, inplace=True)


test_prediction_xg = model.predict(test)


VER = 1


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub.Price = test_prediction_xg
sub.to_csv(f"submission_v{VER}.csv",index=False)
sub.head()


import xgboost as xgb
fig, ax = plt.subplots(figsize=(10, 20))
xgb.plot_importance(model, max_num_features=100, importance_type='gain',ax=ax)
plt.title("Top 100 Feature Importances (XGBoost)")
plt.show()


plt.figure(figsize=(6,4))
plt.hist(sub.Price,bins=100)
plt.title("Test Predictions")
plt.show()


print("previously number of columns: ", len(X_train.columns))
X_train.columns


STATS = ["count","nunique","median","min","max","skew"]


for col in CATS:
    col2 = "Weight Capacity (kg)"
    tmp = X_train.groupby(col)[col2].agg(STATS)
    tmp.columns = [f"FE2_{col}_wc_{s}" for s in STATS]
    X_train = X_train.merge(tmp, on=col, how="left")
    X_val = X_val.merge(tmp, on=col, how="left")
    X_test = X_test.merge(tmp, on=col, how="left")


X_train.shape


X_val.shape


X_test.shape


X_train.head()


model_2 = XGBRegressor(
        device="cuda",
        max_depth=6,  
        colsample_bytree=0.5, 
        subsample=0.8,  
        n_estimators=10_000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=10,
        early_stopping_rounds=100,
    )
model_2.fit(X_train, y_train, eval_set=[(X_val, y_val)],  
        verbose=300,)


train_prediction2 = model_2.predict(X_train)


print("RMSE for train data: ",root_mean_square_error(y_train, train_prediction2))


val_prediction2 = model_2.predict(X_val)


print("RMSE for validation data: ",root_mean_square_error(y_val, val_prediction2))


test_prediction2 = model_2.predict(X_test)


print("RMSE for test data: ",root_mean_square_error(y_test, test_prediction2))


test.shape


for col in CATS:
    col2 = "Weight Capacity (kg)"
    tmp = X_train.groupby(col)[col2].agg(STATS)
    tmp.columns = [f"FE2_{col}_wc_{s}" for s in STATS]
    test = test.merge(tmp, on=col, how="left")


test.shape


test_prediction_xg2 = model_2.predict(test)


VER = 2


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub.Price = test_prediction_xg2
sub.to_csv(f"submission_v{VER}.csv",index=False)
sub.head()


plt.figure(figsize=(6,4))
plt.hist(sub.Price,bins=100)
plt.title("Test Predictions")
plt.show()


STATS = ["mean","std", "count","nunique","median","min","max","skew"]
STATS2 = ["mean","std"]


train.shape


train.head()


train.isna().sum()


X = train.drop(['id', 'Price'], axis=1)
y = train['Price']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


X_train.fillna(X_train.mean(), inplace=True)


X_train.isna().sum()


X_test.fillna(X_train.mean(), inplace=True)


X_test.isna().sum()


X_test, X_encript, y_test, y_encript = train_test_split(X_test, y_test, test_size=0.30, random_state=42)


X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


X_encript.shape


y_encript.shape


encripts = pd.concat([X_encript, y_encript], axis=1)


STATS


for col in CATS:
    col2 = "Price"
    tmp = encripts.groupby(col)[col2].agg(STATS)
    tmp.columns = [f"FE1_{col}_wc_{s}" for s in STATS]
    X_train = X_train.merge(tmp, on=col, how="left")
    X_val = X_val.merge(tmp, on=col, how="left")
    X_test = X_test.merge(tmp, on=col, how="left")


STATS2


for col in CATS:
    col2 = "Weight Capacity (kg)"
    tmp = encripts.groupby(col)[col2].agg(STATS2)
    tmp.columns = [f"FE2_{col}_wc_{s}" for s in STATS2]
    X_train = X_train.merge(tmp, on=col, how="left")
    X_val = X_val.merge(tmp, on=col, how="left")
    X_test = X_test.merge(tmp, on=col, how="left")


X_train.shape


X_val.shape


X_test.shape


model_3 = XGBRegressor(
        device="cuda",
        max_depth=6,  
        colsample_bytree=0.5, 
        subsample=0.8,  
        n_estimators=10_000,  
        learning_rate=0.02,  
        enable_categorical=True,
        min_child_weight=10,
        early_stopping_rounds=100,
    )
model_3.fit(X_train, y_train, eval_set=[(X_val, y_val)],  
        verbose=300,)


train_prediction3 = model_3.predict(X_train)


print("RMSE for train data: ",root_mean_square_error(y_train, train_prediction3))


val_prediction3 = model_3.predict(X_val)


print("RMSE for val data: ",root_mean_square_error(y_val, val_prediction3))


test_prediction3 = model_3.predict(X_test)


print("RMSE for test data: ",root_mean_square_error(y_test, test_prediction3))


test = test[['Brand', 'Material', 'Size', 'Compartments', 'Laptop Compartment',
       'Waterproof', 'Style', 'Color', 'Weight Capacity (kg)', 'Brand_wc',
       'Material_wc', 'Size_wc', 'Compartments_wc', 'Laptop Compartment_wc',
       'Waterproof_wc', 'Style_wc', 'Color_wc']]


for col in CATS:
    col2 = "Price"
    tmp = encripts.groupby(col)[col2].agg(STATS)
    tmp.columns = [f"FE1_{col}_wc_{s}" for s in STATS]
    test = test.merge(tmp, on=col, how="left")


for col in CATS:
    col2 = "Weight Capacity (kg)"
    tmp = encripts.groupby(col)[col2].agg(STATS2)
    tmp.columns = [f"FE2_{col}_wc_{s}" for s in STATS2]
    test = test.merge(tmp, on=col, how="left")


test.shape


test_prediction_xg3 = model_3.predict(test)


VER = 3


sub = pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
sub.Price = test_prediction_xg3
sub.to_csv(f"submission_v{VER}.csv",index=False)
sub.head()


plt.figure(figsize=(6,4))
plt.hist(sub.Price,bins=100)
plt.title("Test Predictions")
plt.show()




