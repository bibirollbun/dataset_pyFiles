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


df_train = pd.read_csv("/kaggle/input/playground-series-s4e10/train.csv") 
df_test = pd.read_csv("/kaggle/input/playground-series-s4e10/test.csv")


df_train.head()


df_train = df_train.drop("id", axis=1)
df_test = df_test.drop("id", axis=1)


def transform_category(df):
    for column in df.columns:
        if df[column].dtype == "object":
            df[column] = df[column].astype("category")

transform_category(df_train)
transform_category(df_test)


from sklearn.model_selection import train_test_split

X = df_train.drop("loan_status", axis=1)
y = df_train["loan_status"]

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Number of training examples:{len(X_train)}")
print(f"Number of testing examples:{len(X_val)}")


(df_train["loan_status"] == 0).sum(), (df_train["loan_status"] == 1).sum()


from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

param_items = {
    "n_estimators": randint(100, 1000),
    "learning_rate": uniform(0.01, 1),
    "alpha": uniform(0.01, 1),
    "colsample_bytree": uniform(0.5, 0.5)
}

xgbmodel = XGBClassifier(enable_categorical=True, n_jobs=-1)

search_result = RandomizedSearchCV(xgbmodel, param_items, n_jobs=-1, cv=5, n_iter=20, scoring="roc_auc")

search_result.fit(X_train, y_train)


print("Best parameters found:", search_result.best_params_)
print(f"ROC-AUC score of the best model: {search_result.best_score_:.3f}")
xgbmodel = search_result.best_estimator_


from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
y_hat = xgbmodel.predict(X_train)
accuracy = accuracy_score(y_hat, y_train)
print(f"Model accuracy on training set: {accuracy:.3f}")

y_hat = xgbmodel.predict(X_val)
accuracy = accuracy_score(y_hat, y_val)
print(f"Model accuracy on validation set: {accuracy:.3f}")

cm = confusion_matrix(y_hat, y_val)
disp = ConfusionMatrixDisplay(cm)
disp.plot()


subm_pred = xgbmodel.predict(df_test)
subm = pd.read_csv("/kaggle/input/playground-series-s4e10/sample_submission.csv")
subm["loan_status"] = subm_pred
subm.to_csv('/kaggle/working/submission.csv', index=False)
subm.head()

