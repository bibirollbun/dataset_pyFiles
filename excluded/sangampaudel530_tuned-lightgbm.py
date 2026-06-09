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


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    OrdinalEncoder
)
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier


train_df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
test_df  = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')



train_df.sample(5)


test_df.sample(5)


TARGET = 'diagnosed_diabetes'
ID_COL = 'id'

X = train_df.drop(columns=[TARGET, ID_COL])
y = train_df[TARGET]

X_test = test_df.drop(columns=[ID_COL])



ordinal_cols = ['education_level', 'income_level']

nominal_cols = [
    'gender',
    'ethnicity',
    'smoking_status',
    'employment_status'
]

binary_cols = [
    'family_history_diabetes',
    'hypertension_history',
    'cardiovascular_history'
]

num_cols = [
    col for col in X.columns
    if col not in ordinal_cols + nominal_cols + binary_cols
]



education_map = {
    'Primary': 0,
    'Highschool': 1,
    'Graduate': 2,
    'Postgraduate': 3
}

income_map = {
    'Low': 0,
    'Middle': 1,
    'Upper-Middle': 2,
    'High': 3
}



def lgbm_preprocess(df):
    df = df.copy()

    # Ordinal encoding
    df['education_level'] = df['education_level'].map(education_map)
    df['income_level'] = df['income_level'].map(income_map)

    # Nominal → category codes
    for col in nominal_cols:
        df[col] = df[col].astype('category').cat.codes

    # Binary columns (ensure int)
    for col in binary_cols:
        df[col] = df[col].astype(int)

    # Numerical missing values
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    return df



X_lgb = lgbm_preprocess(X)
X_test_lgb = lgbm_preprocess(X_test)



categorical_features = ordinal_cols + nominal_cols



    # Train–validation split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_lgb, y,
        test_size=0.15,
        stratify=y,
        random_state=42
    )


final_model = LGBMClassifier(
    n_estimators=5000,   # let early stopping decide
    learning_rate=0.04421848919944907,
    max_depth=5,
    num_leaves=32,
    min_child_samples=40,
    subsample=0.6500763999746155,
    colsample_bytree=0.5705021871344044,
    reg_alpha=0.2,
    reg_lambda=0.2,
    objective='binary',
    metric='auc',
    random_state=42,
    n_jobs=-1
)

final_model.fit(
    X_tr, y_tr,
    eval_set=[(X_val, y_val)],
    eval_metric="auc",
    categorical_feature=categorical_features,
    callbacks=[early_stopping(200)]
)









# import optuna
# from lightgbm import LGBMClassifier, early_stopping
# from sklearn.metrics import roc_auc_score
# from sklearn.model_selection import train_test_split
# from lightgbm import log_evaluation

# def objective(trial):

#     params = {
#         "n_estimators": 5000,   # large, early stopping will decide
#         "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
#         "max_depth": trial.suggest_int("max_depth", 3, 8),
#         "num_leaves": trial.suggest_int("num_leaves", 16, 64),
#         "min_child_samples": trial.suggest_int("min_child_samples", 20, 100),
#         "subsample": trial.suggest_float("subsample", 0.5, 1.0),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
#         "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
#         "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 1.0),
#         "objective": "binary",
#         "metric": "auc",
#         "random_state": 42,
#         "n_jobs": -1
#     }

#     model = LGBMClassifier(**params)
    


#     model.fit(
#         X_tr, y_tr,
#         eval_set=[(X_val, y_val)],
#         eval_metric="auc",
#         categorical_feature=categorical_features,
#         callbacks=[early_stopping(200)]
#     )

#     preds = model.predict_proba(X_val)[:, 1]
#     auc = roc_auc_score(y_val, preds)

#     return auc



# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=50)



# print("Best AUC:", study.best_value)
# print("Best params:", study.best_params)



# from sklearn.model_selection import train_test_split
# from lightgbm import early_stopping, log_evaluation

# # Train–validation split
# X_tr, X_val, y_tr, y_val = train_test_split(
#     X_lgb, y,
#     test_size=0.15,
#     stratify=y,
#     random_state=42
# )

# # Fit with callbacks (LightGBM v4+)
# model.fit(
#     X_tr, y_tr,
#     eval_set=[(X_val, y_val)],
#     eval_metric='auc',
#     categorical_feature=categorical_features,
#     callbacks=[
#         early_stopping(stopping_rounds=200),
#         log_evaluation(100)
#     ]
# )



# best_params = study.best_params

# final_model = LGBMClassifier(
#     **best_params,
#     n_estimators=5000,
#     objective="binary",
#     metric="auc",
#     random_state=42,
#     n_jobs=-1
# )

# final_model.fit(
#     X_lgb, y,
#     categorical_feature=categorical_features
# )



# final_model.fit(
#     X_lgb,
#     y,
#     categorical_feature=categorical_features
# )



test_preds = final_model.predict_proba(X_test_lgb)[:, 1]

submission = pd.DataFrame({
    'id': test_df['id'],
    'diagnosed_diabetes': test_preds
})

submission.to_csv('submission.csv', index=False)





