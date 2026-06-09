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
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier


df = pd.read_csv("/kaggle/input/playground-series-s4e1/train.csv")
df.head()


df.describe()


df = df.drop(["CustomerId", "Surname"], axis=1)
df = pd.get_dummies(df)


def categorise(df, c):
    df = df.copy()
    t1 = np.percentile(np.array(df[c]), 0)
    t2 = np.percentile(np.array(df[c]), 25)
    t3 = np.percentile(np.array(df[c]), 50)
    t4 = np.percentile(np.array(df[c]), 75)
    t5 = np.percentile(np.array(df[c]), 100)
    bins = [t1, t2, t3, t4, t5]
    new_values = pd.cut(df[c],
                        bins, 
                        labels=[1, 2, 3, 4],
                        duplicates='drop')
    return new_values


df.head()


boolean_dict = {True : 1, False : 0}
cols = ["Geography_Germany", "Geography_France", "Geography_Spain", "Gender_Female", "Gender_Male"]
for col in cols:
    df[col] = df[col].map(boolean_dict)


df['combined_info'] = (df['NumOfProducts']+df['HasCrCard'])*df['IsActiveMember']


df["BalancePerProd"] = df["Balance"] / df["NumOfProducts"] 
X, y = df.drop("Exited", axis=1), df["Exited"]


to_cat = ["CreditScore", "EstimatedSalary", "Age"]
for c in to_cat:
    df[c+"_cat"] = categorise(df, c)


colsToScale = ["CreditScore", "Balance", "EstimatedSalary", "BalancePerProd"]
scaler = StandardScaler()
for c in colsToScale:
    
    X[c+"_scaled"] = scaler.fit_transform(X[[c]])
X = X.drop(colsToScale, axis=1)





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)


rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)


score = roc_auc_score(y_test, y_pred)
print(f"The ROC area under the curve score is {score}")


test = pd.read_csv("/kaggle/input/playground-series-s4e1/test.csv")
test.head()


id = test["id"]


test = test.drop(["CustomerId", "Surname"], axis=1)
test = pd.get_dummies(test)
test['combined_info'] = (test['NumOfProducts']+test['HasCrCard'])*test['IsActiveMember']
for col in cols:
    test[col] = test[col].map(boolean_dict)
test["BalancePerProd"] = test["Balance"] / test["NumOfProducts"] 
for c in colsToScale:
    
    test[c+"_scaled"] = scaler.fit_transform(test[[c]])

test = test.drop(colsToScale, axis=1)
Exited = rf.predict(test)


submission_dict = {"id" : id, "Exited" : Exited}
submission = pd.DataFrame(data=submission_dict)


submission.to_csv("submission.csv", index=False)


import xgboost as xgb 


s = y_train.value_counts()
percentage = s[1] / (s[0]+s[1])


percentage


xgb_class = xgb.XGBClassifier(num_estimators=10000, learning_rate=0.2, scale_pos_weight=int(1/percentage))
xgb_class.fit(X_train, y_train)
y_xgb = xgb_class.predict(X_test)
score_xgb = roc_auc_score(y_test, y_xgb)
print(f"The ROC area under the curve score is {score_xgb}")


y_xgb_train = xgb_class.predict(X_train)
y_rf_train = rf.predict(X_train)


lr = LogisticRegression()
lr.fit(X_train, y_train)
y_log = lr.predict(X_test)
score_lr = roc_auc_score(y_test, y_log)
print(f"The ROC area under the curve score is {score_lr}")


y_log_train = lr.predict(X_train)


weights = {0:percentage, 1:1-percentage}


Exited_xgb = xgb_class.predict(test)
submission_xgb = pd.DataFrame({"id" : id, "Exited" : Exited_xgb})
submission_xgb.to_csv("submission.csv", index=False)


cat = CatBoostClassifier(class_weights=weights, eval_metric='AUC', learning_rate=0.022, verbose=False)
cat.fit(X_train, y_train)
y_cat = cat.predict(X_test)
y_train_cat = cat.predict(X_train)
score_cat = roc_auc_score(y_test, y_cat)
print(f"The ROC area under the curve score is {score_cat}")





Exited_cat = cat.predict(test)
submission_cat = pd.DataFrame({"id" : id, "Exited" : Exited_cat})
submission_cat.to_csv("submission.csv", index=False)

