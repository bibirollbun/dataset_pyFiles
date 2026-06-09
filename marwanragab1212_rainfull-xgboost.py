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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier


train= pd.read_csv('/kaggle/input/playground-series-s5e3/train.csv')
test= pd.read_csv('/kaggle/input/playground-series-s5e3/test.csv')



train.head(5)


test.head(5)


def search_nan(data):
  return data.isnull().sum()
print(search_nan(train))
test["winddirection"].fillna(train["winddirection"].mean(), inplace=True)


def info(data):
  return data.info()


print(info(train))
print(info(test))


def drop_id(data):
  return data.drop('id',axis=1,inplace=True)
drop_id(train)
drop_id(test)


print(train.columns)
print(test.columns)


train.describe().transpose()



train["temp_diff"] = train["maxtemp"] - train["mintemp"]
test["temp_diff"] = test["maxtemp"] - test["mintemp"]


train["humidity_temp_ratio"] = train["humidity"] / (train["temparature"] + 1e-6)
test["humidity_temp_ratio"] = test["humidity"] / (test["temparature"] + 1e-6)


train["wind_index"] = train["windspeed"] * train["winddirection"]
test["wind_index"] = test["windspeed"] * test["winddirection"]



plt.figure(figsize=(12, 10))
sns.heatmap(train.corr(), annot=True, cmap='coolwarm')
plt.show()


plt.figure(figsize=(12, 10))
sns.countplot(x='rainfall', data=train)
plt.show()


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(train.drop('rainfall', axis=1), train['rainfall'])

train_balanced = pd.DataFrame(X_resampled, columns=train.drop('rainfall', axis=1).columns)
train_balanced['rainfall'] = y_resampled

print(train_balanced['rainfall'].value_counts())


from sklearn.model_selection import train_test_split

X = train_balanced.drop('rainfall', axis=1)
y = train_balanced['rainfall']


sc=StandardScaler()
X=sc.fit_transform(X)
test_scaled=sc.transform(test)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

param_grid = {
    'n_estimators': [100, 200, 300,500],
    'max_depth': [3, 5, 7, 8],
    'learning_rate': [0.01, 0.1, 0.2,0.22],
    'subsample': [0.7, 0.8, 1.0, 0.2],
    'colsample_bytree': [0.7, 0.8, 1.0 ,0.2]
}

grid_search = GridSearchCV(model, param_grid, cv=3, scoring='roc_auc', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)


best_model = grid_search.best_estimator_



y_p=best_model.predict(X_test)
accuracy_score(y_test,y_p)


# y_probs1 = best_model.predict_proba(test_scaled)[:, 1]


y_probs2 = best_model.predict_proba(X_test)[:, 1]


auc_score = roc_auc_score(y_test, y_probs2)

print(f"AUC Score: {auc_score:.4f}")

