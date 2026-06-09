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
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')


cat_cols = ['gender', 'ethnicity', 'education_level', 'income_level', 'smoking_status', 'employment_status']
df_final = pd.get_dummies(df, columns=cat_cols, drop_first=True)

X = df_final.drop('diagnosed_diabetes', axis=1)
y = df_final['diagnosed_diabetes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42, stratify = y)

df_test_final = pd.get_dummies(df_test, columns = cat_cols, drop_first = True)
model_columns = X_train.columns.tolist()

df_test_final = df_test_final.reindex(columns=model_columns, fill_value=0)


xgb_model = XGBClassifier(
    n_estimators=1000, 
    learning_rate=0.01, 
    max_depth=7, 
    subsample=0.8, 
    colsample_bytree=0.8,
    tree_method='hist', 
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train,y_train)


y_pred = xgb_model.predict(X_test)

print(f"Final Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))


test_predictions = xgb_model.predict(df_test_final)


submission = pd.DataFrame({
    'id': df_test['id'],
    'diagnosed_diabetes': test_predictions
})

print(submission.head())

submission.to_csv('diabetes_predict.csv', index=False)

