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
import zipfile
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()


# Reading the train df
df_train = pd.read_csv('/kaggle/input/playground-series-s4e6/train.csv')
df_train['Target'] = label_encoder.fit_transform(df_train['Target'])
# Reading the test df
df_test = pd.read_csv('/kaggle/input/playground-series-s4e6/test.csv')


# The shapes of train and test dataframes

df_train.shape, df_test.shape


# Sample of df_train dataset

df_train.head()


test_ids= df_test['id']

# Droping 'id' from dfs
df_train.drop('id', axis=1, inplace=True)
df_test.drop('id', axis=1, inplace=True)


# Checking the null values

df_train.isnull().sum()


# Checking the null values

df_test.isnull().sum()


## Feature Engineering

## Feature Engineering

def feature_engineering(df):
    
    df['Total_Curricular_Units_Enrolled'] = df['Curricular units 1st sem (enrolled)'] + df['Curricular units 2nd sem (enrolled)']
    df['Total_Curricular_Units_Evaluations'] = df['Curricular units 1st sem (evaluations)'] + df['Curricular units 2nd sem (evaluations)']
    df['Total_Curricular_Units_Approved'] = df['Curricular units 1st sem (approved)'] + df['Curricular units 2nd sem (approved)']
    df['Average_Curricular_Units_Grade'] = (df['Curricular units 1st sem (grade)'] + df['Curricular units 2nd sem (grade)']) / 2
    
    return df

df_train = feature_engineering(df_train)
df_test = feature_engineering(df_test)




import lightgbm

from sklearn.metrics import mean_squared_log_error
from lightgbm import LGBMClassifier
from math import sqrt
import optuna


from sklearn.model_selection import KFold, StratifiedKFold 


# Independed variables (Features)
X = df_train.drop('Target', axis = 1)

# Depended variables (Targets)
y = df_train['Target']



from sklearn.metrics import accuracy_score


from sklearn.utils.class_weight import compute_class_weight


# # Defineing the objective function for Optuna optimization
# def objective(trial, X, y):
#     """
#     This function is created to find the best parameters using the OPTUNA (a hyperparameter optimization framework) for LightGBM.
#     """
#     # Defining parameters and their ranges
#     param = {
#            'objective': 'multiclass',
#         'num_class': len(y.unique()),
        
#         #"metric": "auc",
#         "verbosity": -1,
#         "boosting_type": "gbdt",
#         "random_state": 42,
# #         "class_weight": {
# #             cls: weight for cls, weight in zip(np.unique(y), compute_class_weight('balanced', classes=np.unique(y), y=y))
# #         },
        
#         "learning_rate": trial.suggest_float("learning_rate", 0.009, 0.02),
#         "n_estimators": trial.suggest_int("n_estimators", 880, 1400),
#         "lambda_l1": trial.suggest_float("lambda_l1", 0.01, 0.55),
#         "lambda_l2": trial.suggest_float("lambda_l2", 0.01, 0.55),
#         "max_depth": trial.suggest_int("max_depth", 3, 8),
#         "num_leaves": trial.suggest_int("num_leaves", 20, 80),
#         "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 0.95),
#         "subsample": trial.suggest_float("subsample", 0.15, 1.0),
#         "min_child_samples": trial.suggest_int("min_child_samples", 2, 75),
#     }

#     # Createing an instance of LGBMClassifier with the suggested parameters
#     lgbm_classifier = LGBMClassifier(**param)

#    # Perform cross-validation
#      #kf = KFold(n_splits=5, shuffle=True, random_state=42)
#     kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
#     scores = []
#     for train_index, test_index in kf.split(X, y):
#         X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
#         y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

#         # Fit the regressor on the training data
#         lgbm_classifier.fit(X_train_fold, y_train_fold)

#         # Evaluate the regressor on the test data
#         y_pred_lgb = lgbm_classifier.predict(X_test_fold)

#         score = accuracy_score(y_test_fold, y_pred_lgb)
#         scores.append(score)

#     return np.mean(scores)

# # Setting up the sampler for Optuna optimization
# sampler = optuna.samplers.TPESampler(seed=42)  # Using Tree-structured Parzen Estimator sampler for optimization

# # Createing a study object for Optuna optimization
# study = optuna.create_study(direction="maximize", sampler=sampler)

# # Running the optimization process
# study.optimize(lambda trial: objective(trial, X, y), n_trials=50) # n_trials=25

# # Getting the best parameters after optimization
# best_params_lgb = study.best_params
# best_score_lgb = study.best_value

# print("Best score:", best_score_lgb )
# print("Best parameters:", best_params_lgb)


lgb_best_parameters = {
    'objective': 'multiclass',
        'num_class': len(y.unique()),
        
        "verbosity": -1,
        "boosting_type": "gbdt",
  "random_state": 42,
    
  'learning_rate': 0.015,
  'n_estimators': 10000,
 'lambda_l1': 0.195,
 'lambda_l2': 0.2088,
 'max_depth': 6,
 'num_leaves': 62,
 'colsample_bytree': 0.359,
 'subsample': 0.94763,
 'min_child_samples': 47
    
}


proba = np.zeros((len(df_test),5))
proba.shape



# Createing an instance of LGBMClassifier with the suggested parameters
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
lgbm_model = LGBMClassifier(**lgb_best_parameters)

proba = np.zeros((len(df_test),3))
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=20)
scores = []
predictions = []
i = 0
for train_index, test_index in kf.split(X, y):
    X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
    y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

    # Fit the regressor on the training data
    lgbm_model.fit(
        X_train_fold, y_train_fold,
        eval_set=[(X_test_fold, y_test_fold)],
        eval_metric='multi_logloss', # Ou 'merror' pour l'erreur de classification
        callbacks=[
            early_stopping(stopping_rounds=100, verbose=False),
            log_evaluation(period=0) # Pour ne pas spammer la console
        ]
    )

    # Evaluate the regressor on the test data
    y_pred_lgb = lgbm_model.predict(X_test_fold)
    
    # Evaluate the regressor on the test data
    y_pred_test = lgbm_model.predict(df_test)
    
    score = accuracy_score(y_test_fold, y_pred_lgb)
    scores.append(score)
    predictions.append(y_pred_test)
    proba+=lgbm_model.predict_proba(df_test)
    
    print('Fold Score', score)
    i = i+1
    
    
print('--------')
print('Total Score:', np.mean(scores))



predictions


from scipy import stats


test_targets = stats.mode(predictions,keepdims = True).mode[0]


test_targets


test_targets = test_targets.astype(int)
test_targets = label_encoder.inverse_transform(test_targets)


test_prediction = pd.DataFrame(test_targets, columns = ['Target'])
submission = pd.concat([test_ids, test_prediction],axis = 1)
submission


submission.to_csv('submission.csv', index = False)

