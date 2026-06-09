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


# !pip install xgboost
!pip install lightgbm


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import RidgeCV
# from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
# from xgboost import XGBRegressor
import lightgbm as lgb

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor
from sklearn.metrics import mean_squared_error, r2_score


train = pd.read_csv('/kaggle/input/eda-predict-road-accident-risk/eda_predict_road_accident_risk_v2_feature_importance.csv', index_col=False )


test = pd.read_csv('/kaggle/input/eda-predict-road-accident-risk/test_eda_predict_road_accident_risk_v2_feature_importance.csv', index_col=False)


test.shape


train.shape


train.columns


test.columns


# Remover colunas que não ajudam no modelo
if 'Unnamed' in test.columns:
    test = test.drop(columns=["Unnamed: 0"])


# Remover colunas que não ajudam no modelo
if 'Unnamed' in train.columns:
    train = train.drop(columns=["Unnamed: 0"])


train


# Remover colunas que não ajudam no modelo
if 'Unnamed: 0' in train.columns:
    train = train.drop(columns=["Unnamed: 0"])


# Remover colunas que não ajudam no modelo
if 'Unnamed: 0' in test.columns:
    test = test.drop(columns=["Unnamed: 0"])


train


# Definir alvo (exemplo: prever accident_risk)
y = train["accident_risk"]
X = train.drop(columns=["accident_risk"])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Modelos base
rf = RandomForestRegressor(n_estimators=100, random_state=42)
# gb = GradientBoostingRegressor(n_estimators=100, random_state=42)
# xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1, random_state=42)
lgb_model = lgb.LGBMRegressor(objective='regression', num_leaves=31, learning_rate=0.05, n_estimators=100)
# enc = ElasticNetCV(random_state=42)
srv = LinearSVR(random_state=42)
dt = DecisionTreeRegressor(random_state=42)
pm = MLPRegressor(max_iter=1000, random_state=42)
# kr = KNeighborsRegressor()


estimators = [
    ('rf',rf),
    # ('gb',gb),
    # ('xgb', xgb),
    # ('enc',enc)
    ('lgb_model',lgb_model),
    ('srv',srv),
    ('dt',dt),
    ('pm',pm),
    # ('kr',kr)
]


# Define a range of alpha values to try
alphas = [0.1, 1.0, 10.0]


# Define meta-learner
# final_estimator = LinearRegression()
# final_estimator = RidgeCV()
final_estimator = RidgeCV(alphas=alphas, cv=5)


# Ensemble
stacking_clf = StackingRegressor(estimators=estimators, final_estimator=final_estimator, cv=5)


X_train = X_train.iloc[:,1:len(train.columns)]


# Train the stacking classifier
stacking_clf.fit(X_train, y_train)


X_train.head()


X_test


X_test = X_test.iloc[:,1:len(X_test.columns)]


# Evaluate performance
accuracy = stacking_clf.score(X_test, y_test)
print(f"Stacking Classifier Accuracy: {accuracy}")


# Assuming X_test is your test features
y_pred_stacked = stacking_clf.predict(X_test)


# Avaliação
rmse = np.sqrt(mean_squared_error(y_test, y_pred_stacked))
r2 = r2_score(y_test, y_pred_stacked)


print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.3f}")


test


train


id = test['id']


test = test.drop(columns=['id'])


# Assuming X_test is your test features
y_test_stacked = stacking_clf.predict(test)


#Submission must be done within the standard requested by the challenge. 
#    In this case, the depression column must be submitted with 0 and 1.
#    This is why it is necessary to convert.

submission = y_test_stacked
submission


test.shape


submission.shape


## Submit notebooks to the challenge. Final


submission_final = pd.DataFrame({

        "id":id,

        "Premium Amount":submission

    })

submission_final.to_csv('submission.csv', index=False)


print(" Arquivo submission.csv pronto ")

