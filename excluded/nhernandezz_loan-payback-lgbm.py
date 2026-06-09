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


from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna 


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
gender_encoder = LabelEncoder()
gender_encoder.fit(df['gender'])
df['gender'] = gender_encoder.transform(df['gender'])
marital_encoder = LabelEncoder()
marital_encoder.fit(df['marital_status'])
df['marital_status'] = marital_encoder.transform(df['marital_status'])
categorical_columns = ['education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded_variables = encoder.fit_transform(df[categorical_columns])
categorical_df = pd.DataFrame(encoded_variables, columns=encoder.get_feature_names_out(categorical_columns))
full_df = pd.concat([df.drop(columns = categorical_columns).reset_index(drop=True),
                      categorical_df.reset_index(drop=True)], axis=1)
full_df.drop(columns = "id", inplace = True)
scaler_income = StandardScaler()
scaler_income.fit(df[['annual_income']])
credit_score_scaler = StandardScaler()
credit_score_scaler.fit(df[['credit_score']])
debt_to_income_ratio_scaler = StandardScaler()
debt_to_income_ratio_scaler.fit(df[['debt_to_income_ratio']])
loan_amount_scaler = StandardScaler()
loan_amount_scaler.fit(df[['loan_amount']])
interest_rate_scaler = StandardScaler()
interest_rate_scaler.fit(df[['interest_rate']])

full_df['annual_income'] = scaler_income.transform(full_df[['annual_income']])
full_df['credit_score'] = credit_score_scaler.transform(full_df[['credit_score']])
full_df['debt_to_income_ratio'] = debt_to_income_ratio_scaler.transform(full_df[['debt_to_income_ratio']])
full_df['loan_amount'] = loan_amount_scaler.transform(full_df[['loan_amount']])
full_df['interest_rate'] = interest_rate_scaler.transform(full_df[['interest_rate']])



X = full_df.drop(columns = ['loan_paid_back'])
y = full_df['loan_paid_back']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


def objective(trial):
    params = {
        'objective': 'binary',           
        'metric': 'auc',                 
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_float('learning_rate', 1e-3, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 200),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-5, 10.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-5, 10.0),
        'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0.0, 1.0)
    }

    # Entrenamiento
    model = lgb.LGBMClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='auc'
    )

    
    model.fit(X_train, y_train)     
    y_score = model.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_score)
    return auc


#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=10)


#paramsLGBM = study.best_params
#paramsLGBM


'''
paramsLR = study.best_params
lgbm_model = lgb.LGBMClassifier(**paramsLR)
X_train_full = pd.concat([X_train, X_val])
y_train_full = pd.concat([y_train, y_val])
lgbm_model.fit(X_train_full, y_train_full)     
y_score = lgbm_model.predict_proba(X_test)[:,1]
auc = roc_auc_score(y_test, y_score)
auc
'''


X_train_full = pd.concat([X_train, X_val, X_test])
y_train_full = pd.concat([y_train, y_val, y_test])
lgbm_model = lgb.LGBMClassifier(**{'learning_rate': 0.03564502491535402,
 'num_leaves': 68,
 'max_depth': 10,
 'min_data_in_leaf': 132,
 'subsample': 0.5986089261404222,
 'colsample_bytree': 0.5508694645338765,
 'n_estimators': 421,
 'reg_alpha': 2.4896070799725423,
 'reg_lambda': 4.290605892567426,
 'min_gain_to_split': 0.8898447345310345,
 'random_state':42})
lgbm_model.fit(X_train_full, y_train_full) 


test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
test['gender'] = gender_encoder.transform(test['gender'])
test['marital_status'] = marital_encoder.transform(test['marital_status'])
categorical_columns = ['education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
encoder = OneHotEncoder(sparse_output=False, drop=None)
encoded_variables = encoder.fit_transform(test[categorical_columns])
categorical_df = pd.DataFrame(encoded_variables, columns=encoder.get_feature_names_out(categorical_columns))
full_df = pd.concat([test.drop(columns = categorical_columns).reset_index(drop=True),
                      categorical_df.reset_index(drop=True)], axis=1)
full_df.drop(columns = "id", inplace = True)

full_df['annual_income'] = scaler_income.transform(full_df[['annual_income']])
full_df['credit_score'] = credit_score_scaler.transform(full_df[['credit_score']])
full_df['debt_to_income_ratio'] = debt_to_income_ratio_scaler.transform(full_df[['debt_to_income_ratio']])
full_df['loan_amount'] = loan_amount_scaler.transform(full_df[['loan_amount']])
full_df['interest_rate'] = interest_rate_scaler.transform(full_df[['interest_rate']])


X = full_df
y_score = lgbm_model.predict_proba(X)[:,1]
test['loan_paid_back'] = y_score
submit = test[['id','loan_paid_back']]
submit = submit.set_index('id')
submit.to_csv('submission.csv')
submit

