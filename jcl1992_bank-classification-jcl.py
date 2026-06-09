import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

pd.options.mode.chained_assignment = None


CROSS_VALIDATION = False


df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")

df


df.info()


df["y"].value_counts()


df["pcontacted"] = df["pdays"].apply(lambda x: "yes" if x > 0 else "no")
df_test["pcontacted"] = df_test["pdays"].apply(lambda x: "yes" if x > 0 else "no")

df["day"] = df["day"].astype(str)
df_test["day"] = df_test["day"].astype(str)


categorical = df.dtypes[df.dtypes=="object"].index.to_list()
numeric = df.dtypes[df.dtypes!="object"].index.to_list()
numeric = [col for col in numeric if col not in ["id", "y"]]
print("Categorical features :",categorical)
print("Numeric features :", numeric)


X = df[categorical + numeric]
X_test = df_test[categorical + numeric]

X[categorical] = X[categorical].astype('category')
X_test[categorical] = X_test[categorical].astype('category')

#X_cat = pd.get_dummies(df[categorical], dtype=int)
#X_num = df[numeric]
#X = pd.concat([X_num, X_cat], axis=1)

y = df["y"]

X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2)

X_train


if CROSS_VALIDATION:

    param_grid = {'max_depth': [3, 4, 5, 6, 7],
                  'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5]}
    skf = StratifiedKFold(n_splits=5)
    xgb = XGBClassifier(enable_categorical=True)
    clf = GridSearchCV(xgb, param_grid, cv=skf, refit=True, scoring='roc_auc', n_jobs=-1)
    %time clf.fit(X, y)
    print(clf.best_params_)
    print(clf.best_score_)
    xgb = clf.best_estimator_
    y_proba = clf.predict_proba(X_test)[:,1]
    display(pd.DataFrame(clf.cv_results_)[["params", "mean_test_score"]])

else:

    xgb = XGBClassifier(n_estimators=100, 
                    max_depth=7, 
                    learning_rate=0.2, 
                    enable_categorical=True)
    %time xgb.fit(X_train, y_train)
    y_proba = xgb.predict_proba(X_val)[:,1]
    print("ROC AUC score :", roc_auc_score(y_val, y_proba))
    %time xgb.fit(X, y)
    y_proba = xgb.predict_proba(X_test)[:,1]


feature_importance = xgb.get_booster().get_score(importance_type='gain')
feature_importance = pd.Series(feature_importance, name="importance").sort_values()
feature_importance.plot(kind="barh");


df_submission = pd.read_csv("/kaggle/input/playground-series-s5e8/sample_submission.csv")

df_submission["y"] = y_proba

df_submission


df_submission.to_csv("submission.csv", index=False)




