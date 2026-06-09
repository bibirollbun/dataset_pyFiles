# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Pré-traitement et métriques : 
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.model_selection import train_test_split, KFold

# Classifiers : 
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC

from lightgbm.sklearn import LGBMRegressor, LGBMClassifier
from catboost import CatBoostRegressor, CatBoostClassifier, Pool
import xgboost as xgb

import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


df_test = pd.read_csv("/kaggle/input/playground-series-s5e3/test.csv")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e3/train.csv")
sub = pd.read_csv("/kaggle/input/playground-series-s5e3/sample_submission.csv")


df_test.info()


LABEL = "rainfall"
FEATURES = [c for c in df_train.columns if c not in [LABEL, "id"]]

DO = 3


df_test = df_test.fillna(0)


df_train.columns


X = df_train[FEATURES]
y = df_train[LABEL]
X_test_competition = df_test[FEATURES]


X.shape, y.shape


X.info()


y


X_temporary, X_test, y_temporary, y_test = train_test_split(X, y, test_size=100, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temporary, y_temporary, test_size=100, random_state=42)


model = CatBoostClassifier(iterations=100, loss_function='Logloss') 
# model = RandomForestClassifier(n_estimators=100) 


if DO == 1:
    model.fit(X_train, y_train)
    prediction_test = model.predict(X_test)
    score_test = accuracy_score(y_test, prediction_test)

    print("Score test", score_test)
elif DO == 2:
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
    
    prediction_test = model.predict(X_test)
    score_test = accuracy_score(y_test, prediction_test)

    prediction_val = model.predict(X_val)
    score_val = accuracy_score(y_val, prediction_val)

    print("Score val", score_val, "Score test", score_test)
elif DO == 3:
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], use_best_model=True)

    prediction_test = model.predict(X_test)    

    prediction_test = model.predict(X_test)
    score_test = accuracy_score(y_test, prediction_test)

    prediction_val = model.predict(X_val)
    score_val = accuracy_score(y_val, prediction_val)

    print("Score val", score_val, "Score test", score_test)



# Si on veut participer à la compétition : 

if DO > 0:
    sub[LABEL] = model.predict_proba(X_test_competition)[:,1]
    sub.to_csv("submission.csv", index=False)


if DO > 0:
    prediction = model.predict_proba(X_test_competition)
    print(prediction)




