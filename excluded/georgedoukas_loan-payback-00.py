import os
import numpy as np
import pandas as pd

from sklearn.preprocessing import OrdinalEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from lightgbm import early_stopping

from sklearn.compose import ColumnTransformer

import sklearn

from sklearn.pipeline import Pipeline

from category_encoders.target_encoder import TargetEncoder
from category_encoders.woe import WOEEncoder
from category_encoders.woe import OrdinalEncoder
from category_encoders.one_hot import OneHotEncoder
from category_encoders.quantile_encoder import QuantileEncoder

from sklearn.feature_selection import SelectKBest

import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


train=pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
ypo=pd.read_csv("/kaggle/input/playground-series-s5e11/sample_submission.csv")
test=pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")


train.head()


train.isnull().sum()


test.head()


test.isnull().sum()


train.shape, test.shape


train=train.drop("id", axis=1)
test=test.drop("id", axis=1)


train.describe(include="all")


test.describe(include="all")


np.unique(train["grade_subgrade"],return_counts=True)


np.unique(test["grade_subgrade"],return_counts=True)


train["grade_subgrade_let"]=list(map(lambda x:x[0],train["grade_subgrade"] ))


train["grade_subgrade_num"]=list(map(lambda x:int(x[1]),train["grade_subgrade"] ))


test["grade_subgrade_let"]=list(map(lambda x:x[0],test["grade_subgrade"] ))
test["grade_subgrade_num"]=list(map(lambda x:int(x[1]),test["grade_subgrade"] ))



features_categorical = [c for c in train.columns if (train[c].dtypes == "O" or  train[c].dtypes == "bool")]
features_categorical


target="loan_paid_back"
features_numerical = [
    c for c in train.columns if c not in features_categorical and c != target
]
features_numerical


X=train.copy().drop(target,axis=1)
X_pred=test.copy()
y=train[target]


numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])

categorical_transformer = Pipeline(steps=[("encoder", OrdinalEncoder())])


preprocessor = ColumnTransformer(
    transformers=[
        ("numerical", numeric_transformer, features_numerical),
        ("categorical", categorical_transformer, features_categorical),
    ]
)


import lightgbm
from lightgbm import LGBMClassifier


model=LGBMClassifier(n_estimators=200, learning_rate=0.1,verbose=1)


pipe = Pipeline(
 steps=[("preprocessor", preprocessor), ("model", model)])


param_grid = {
    "preprocessor__numerical": [MinMaxScaler(), StandardScaler(), RobustScaler()],
    "preprocessor__categorical": [OrdinalEncoder(),OneHotEncoder(),TargetEncoder(),QuantileEncoder()],
    "model__learning_rate": [0.1,0.05, 0.15]
}


skf=StratifiedKFold(3)


grid_search = GridSearchCV(pipe, param_grid, cv=skf, n_jobs=-1, scoring="roc_auc",verbose=3)


#grid_search.fit(X, y)


#grid_search.best_estimator_


#grid_search.best_params_


#grid_search.best_score_


preprocessor1 = ColumnTransformer(
    transformers=[
        ("numerical", MinMaxScaler(), features_numerical),
        ("categorical", TargetEncoder(), features_categorical),
    ]
)


preprocessor1.fit(X,y)


X1=preprocessor1.transform(X)
X_pred_1=preprocessor1.transform(X_pred)


!pip install flaml


from flaml import AutoML
automl = AutoML()

params={"task":"classification",
"time_budget": 4*3600,
"metric": "roc_auc",
"estimator_list":["xgboost","lgbm","histgb","extra_tree" ],#,"xgboost","catboost","xgb_limitdepth" ,"xgboost","xgb_limitdepth"],     #"rf","extra_tree", "histgb"
"eval_method":"cv",
"n_splits":5
       }

automl.fit(X1, y,**params)


print(automl.model.estimator)


preds=automl.predict_proba(X_pred_1)
preds


ypo[target]=preds[:,1]
ypo.to_csv('submission.csv', index=False)
ypo.head()

