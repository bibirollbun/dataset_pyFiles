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


df1 = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
df2 = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')


df1.head(3)


from sklearn.preprocessing import LabelEncoder, MinMaxScaler 


enc = LabelEncoder()
def encode_(df1):
    for i in df1.columns:
        if df1[i].dtype in ['object', 'bool']:
            df1[i] = enc.fit_transform(df1[i])
    return df1.head(3)

encode_(df1)


encode_(df2)


scale = MinMaxScaler(feature_range=(0,1))

X = pd.DataFrame(scale.fit_transform(df1.drop(['id', 'accident_risk'], axis=1)))
X.head(2)


X_test = pd.DataFrame(scale.fit_transform(df2.drop('id', axis=1)))


y = df1.accident_risk


from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


X_train, X_val, y_train, y_val = train_test_split(X, y, random_state=10)


clf = RandomForestRegressor(n_estimators=100)
clf.fit(X_train, y_train)

clf_mod = clf.predict(X_val)
RMSE = np.sqrt(mean_squared_error(y_val, clf_mod))
print(f"RMSE: {RMSE:.4F}")


from catboost import CatBoostRegressor

m_cat = CatBoostRegressor(iterations=500, 
                          learning_rate=0.1, 
                          depth=6, 
                          loss_function='RMSE', 
                          random_state=10,
                          verbose=0)
m_cat.fit(X_train, y_train)

cat_mod = m_cat.predict(X_val)
RMSE_cat = np.sqrt(mean_squared_error(y_val, cat_mod))
print(f"RMSE: {RMSE_cat:.4F}")


preds_tst = m_cat.predict(X_test)
submission = pd.DataFrame({'id': df2['id'], 'accident_risk': preds_tst})
submission.to_csv("submission.csv", index=False)

