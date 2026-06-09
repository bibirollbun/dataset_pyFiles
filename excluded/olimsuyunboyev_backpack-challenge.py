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


import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.pipeline import Pipeline
# from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error


df = pd.read_csv("/kaggle/input/playground-series-s5e2/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e2/test.csv")
# train_extra = pd.read_csv("/kaggle/input/playground-series-s5e2/training_extra.csv")


# df = pd.concat([train, train_extra])


test.fillna(method='ffill', inplace=True)


df.head()


df.isnull().sum()


df.info()


df = df.dropna()


test_ids = test['id']


# df = df.drop('id', axis=1)
# test = test.drop('id', axis=1)


df.head()


df.isnull().sum()/df.shape[0]


df.info()


label = LabelEncoder()
df['Brand'] = label.fit_transform(df['Brand'].values)
df['Material'] = label.fit_transform(df['Material'].values)
df['Size'] = label.fit_transform(df['Size'].values)
df['Laptop Compartment'] = label.fit_transform(df['Laptop Compartment'].values)
df['Waterproof'] = label.fit_transform(df['Waterproof'].values)
df['Style'] = label.fit_transform(df['Style'].values)
df['Color'] = label.fit_transform(df['Color'].values)


label = LabelEncoder()
test['Brand'] = label.fit_transform(test['Brand'].values)
test['Material'] = label.fit_transform(test['Material'].values)
test['Size'] = label.fit_transform(test['Size'].values)
test['Laptop Compartment'] = label.fit_transform(test['Laptop Compartment'].values)
test['Waterproof'] = label.fit_transform(test['Waterproof'].values)
test['Style'] = label.fit_transform(test['Style'].values)
test['Color'] = label.fit_transform(test['Color'].values)


df.info()


df.corrwith(df['Price']).abs().sort_values(ascending=False)


X = df.drop('Price', axis=1)
y = df['Price']


scaler = MinMaxScaler()
X = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state=11)


# from sklearn.model_selection import GridSearchCV

# param_grid = {
#     'n_estimators': [250, 300, 350],
#     'max_depth': [10, 15, 20],
#     'min_samples_split': [2, 4, 6],
#     'min_samples_leaf': [1, 2, 3]
# }

# grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
# grid_search.fit(X_train, y_train)

# print("Yakuniy eng yaxshi parametrlar:", grid_search.best_params_)



# Modelni yaratish va o'qitish
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=15,
    max_features='sqrt',
    bootstrap=True,
    random_state=42)

model.fit(X_train, y_train)


# Test datasetida bashorat qilish
predictions = model.predict(X_test)


rmse = np.sqrt(mean_squared_error(y_test, predictions))
print("RMSE:", rmse)


test_prediction = model.predict(test)


submission = pd.DataFrame({
    'id': test_ids,
    'Price': test_prediction
})

submission.to_csv('_sample_submission.csv', index=False)

print("Submission fayli muvaffaqiyatli saqlandi!")


print(f"X_test uzunligi: {len(X_test)}")
print(f"predictions uzunligi: {len(predictions)}")
print(f"test uzunligi: {len(test)}")

