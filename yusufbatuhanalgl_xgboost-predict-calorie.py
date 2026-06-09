# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LinearRegression 
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_train = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
df_sample = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")


df_train = pd.get_dummies(df_train, columns=["Sex"], drop_first=True)
df_test = pd.get_dummies(df_test, columns=["Sex"], drop_first=True)


x = df_train.drop(columns=["Calories"])
y = df_train["Calories"]


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


xgb = XGBRegressor(n_estimators=100)


xgb.fit(X_train, y_train,)



xgb.score(X_test,y_test)


missing_cols = set(x.columns) - set(df_test.columns)
for col in missing_cols:
    df_test[col] = 0
df_test = df_test[x.columns]


pred = xgb.predict(df_test)
pred = np.maximum(pred, 0)



submission = pd.DataFrame({
    'id': df_test['id'],
    'Calories': pred
})


submission.columns = submission.columns.str.strip()


submission.to_csv("submission.csv", index=False)




