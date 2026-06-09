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


df_train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
df_sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")


df_train


df_train.info()


df_train['gender'] = df_train['gender'].apply(lambda x: 1 if x=='Male' else 0)
df_test['gender'] = df_test['gender'].apply(lambda x: 1 if x=='Male' else 0)


df_train = pd.get_dummies(df_train, columns = ['marital_status', 'education_level', 'employment_status', 'loan_purpose'])
df_test = pd.get_dummies(df_test, columns = ['marital_status', 'education_level', 'employment_status', 'loan_purpose'])


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsRegressor

X = df_train.drop(columns=['grade_subgrade', 'loan_paid_back'],axis=1)
y = df_train['loan_paid_back']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'Random Forest': RandomForestRegressor(),
    'Gradient Boosting': GradientBoostingRegressor(),
    'Kneighbors': KNeighborsRegressor()
}

for name, model in models.items():
    print("="*50)
    print(name)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(f'auc: {roc_auc_score(y_test, y_pred)}')


from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import HistGradientBoostingRegressor

params = {
    'learning_rate': [0.1, 0.3, 0.6, 1],
    'max_iter': [100, 125, 150],
    'max_depth': [None, 10, 20, 30]
}

gcv = GridSearchCV(HistGradientBoostingRegressor(), params, cv=5, scoring='roc_auc')

gcv.fit(X_train, y_train)
print(f"Best cross-validation score: {gcv.best_score_:.4f}")
print(f"Best parameters: {gcv.best_params_}")


from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

X = df_train.drop(columns=['grade_subgrade', 'loan_paid_back'],axis=1)
y = df_train['loan_paid_back']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = HistGradientBoostingRegressor(learning_rate=0.1, max_depth=30, max_iter=150)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(roc_auc_score(y_test, y_pred))


X = df_test.drop(columns=['grade_subgrade'],axis=1)

final_prediction = model.predict(X)
submission = df_sample_submission.copy()
submission["loan_paid_back"] = final_prediction
submission.to_csv("submission.csv", index=False)

