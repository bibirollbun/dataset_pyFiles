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


import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error

df_train, df_test, sample = (pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv"),
                            pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv"),
                             pd.read_csv("/kaggle/input/playground-series-s5e2/sample_submission.csv")
                            )
sample


df_train.isnull().sum(), df_test.isnull().sum()


df_train.iloc[:,1].unique()


# check the consistency between tarin and test column
for col in df_test.columns:
    result = df_train[col].unique().size == df_test[col].unique().size
    print(f"{col} of train & test are consistent. -------------{result} {df_train[col].dtype}")


# Notice the last col of weigths, they are just continuous values
# We focus on the cat cols and prepare the labelencoder

train, test = df_train.copy(), df_test.copy()
cat_lbes = []
cat_cols = df_train.select_dtypes(include=['object'])
for col in cat_cols:
    lbe = LabelEncoder()
    train[col] = lbe.fit_transform(df_train[col])
    test[col] = lbe.transform(df_test[col])
    cat_lbes.append(lbe)
    print(f"Current handled col is {col}.")
print('finished.')


import warnings
warnings.filterwarnings('ignore')
train['Weight Capacity (kg)'].fillna(0, inplace=True)
test['Weight Capacity (kg)'].fillna(0, inplace=True)


train


from sklearn.metrics import mean_squared_error as mse


# Evaluate the performance based on the train
X, y = train.iloc[:, 1:-1], train.iloc[:, -1]

#def kf_eval(model, features, values):

rfr = RandomForestRegressor(n_jobs=-1)
kf = KFold(5)  
mses = []
for fold, (train_idx, test_idx) in enumerate(kf.split(X,y)):
    X_train, X_test = X.iloc[train_idx,:], X.iloc[test_idx, :]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    rfr.fit(X_train, y_train)
    pred = rfr.predict(X_test)
    score = mse(y_test, pred)
    mses.append(score)
    print(f"{fold} Fold score: {score}")
    


from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.pipeline import make_pipeline
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

knn = KNeighborsRegressor()
rknn = RadiusNeighborsRegressor()
knn = make_pipeline(StandardScaler(), knn)
rknn = make_pipeline(StandardScaler(), rknn)
hgbr = HistGradientBoostingRegressor()
lgbm =  LGBMRegressor()
lr =  make_pipeline(StandardScaler(), LinearRegression())


sample = sample.set_index('id')
# rf
rfr.fit(X,y)
pred = rfr.predict(test.iloc[:,1:])
sample['Price'] = pred
sample.to_csv('submit_rf.csv')

# rknn
rknn.fit(X,y)
pred = rknn.predict(test.iloc[:,1:])
sample['Price'] = pred
sample.to_csv('submit_rknn.csv')

# knn_pipe
knn.fit(X,y)
pred = knn.predict(test.iloc[:,1:])
sample['Price'] = pred
sample.to_csv('submit_knn.csv')

# hgbr
hgbr.fit(X,y)
pred = hgbr.predict(test.iloc[:,1:])
sample['Price'] = pred
sample.to_csv('submit_hgbr.csv')

# lgbm
lgbm.fit(X,y)
pred = lgbm.predict(test.iloc[:,1:])
sample['Price'] = pred
sample.to_csv('submit_lgbm.csv')

# lr
lr.fit(X,y)
pred = lr.predict(test.iloc[:,1:])
sample['Price'] = pred
sample.to_csv('submit_lr.csv')

