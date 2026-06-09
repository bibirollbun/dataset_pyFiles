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

import warnings
warnings.filterwarnings('ignore')

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


train = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv").set_index('id')
test = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv").set_index('id')


train.info()


train.head()


train.describe()


train.duplicated().sum()


for i in train.columns:
    sns.displot(train[i], kde=True, bins=30)
    plt.show()


outlier_rows = set()

for col in train.columns.to_list():
    Q1 = train[col].quantile(0.25)
    Q3 = train[col].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = train[(train[col] < lower_bound) | (train[col] > upper_bound)].index
    outlier_rows.update(outliers)
    
    print(f"{col}: {len(outliers)} outlier")

print(f"\nTotal outlier row: {len(outlier_rows)}")

train_clean = train.drop(outlier_rows)

print(f"Cleaned data size: {train_clean.shape}")
print(f"Number of deleted rows: {len(outlier_rows)}")


train = train_clean.copy()

print("✅ Outliers cleared, variable 'df' is up to date!")


train.info()


for i in train.columns:
    sns.displot(train[i], kde=True, bins=30)
    plt.show()


X = train.drop(columns='BeatsPerMinute', axis=1)
y = train['BeatsPerMinute']


scaler = StandardScaler()


X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)


X.head()


X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.8, random_state=42)



from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import numpy as np

lgbm = LGBMRegressor(device='gpu')
xgb = XGBRegressor(tree_method='gpu_hist', predictor='gpu_predictor')
cat = CatBoostRegressor(task_type='GPU', verbose=0)

# Training
lgbm.fit(X_train, y_train)
xgb.fit(X_train, y_train)
cat.fit(X_train, y_train)

# Prediksi
pred_lgbm = lgbm.predict(X_val)
pred_xgb = xgb.predict(X_val)
pred_cat = cat.predict(X_val)

# Evaluasi RMSE
rmse_lgbm = mean_squared_error(y_val, pred_lgbm, squared=False)
rmse_xgb = mean_squared_error(y_val, pred_xgb, squared=False)
rmse_cat = mean_squared_error(y_val, pred_cat, squared=False)

print("LGBM RMSE:", rmse_lgbm)
print("XGBoost RMSE:", rmse_xgb)
print("CatBoost RMSE:", rmse_cat)



scaled_test = pd.DataFrame(scaler.transform(test), columns=test.columns)


y_pred = lgbm.predict(scaled_test)


my_sub = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv').set_index('id')
my_sub['BeatsPerMinute'] = y_pred
my_sub.to_csv('submission.csv')




