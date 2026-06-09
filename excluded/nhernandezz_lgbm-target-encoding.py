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
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import optuna 


df = pd.read_csv('/kaggle/input/playground-series-s5e11/train.csv')
df.head()


#New features
df['loan_charge'] = df['loan_amount']/df['annual_income']
df['total_credit'] = df['interest_rate']*df['loan_amount']
df['DTI/LA'] = df['debt_to_income_ratio']*df['loan_amount']
df['risk_sign'] = df['loan_amount']/df['credit_score']
df.head()


#Label encoders
gender_encoder = LabelEncoder()
gender_encoder.fit(df['gender'])
df['gender'] = gender_encoder.transform(df['gender'])
marital_encoder = LabelEncoder()
marital_encoder.fit(df['marital_status'])
df['marital_status'] = marital_encoder.transform(df['marital_status'])


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

df['annual_income'] = scaler_income.transform(df[['annual_income']])
df['credit_score'] = credit_score_scaler.transform(df[['credit_score']])
df['debt_to_income_ratio'] = debt_to_income_ratio_scaler.transform(df[['debt_to_income_ratio']])
df['loan_amount'] = loan_amount_scaler.transform(df[['loan_amount']])
df['interest_rate'] = interest_rate_scaler.transform(df[['interest_rate']])



X = df.drop(columns = ['loan_paid_back', 'id'])
y = df['loan_paid_back']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)


te = TargetEncoder(cols=["education_level", "employment_status", "loan_purpose", "grade_subgrade"], smoothing=0.3)

# Ajustar SOLO en train
X_train_te = te.fit_transform(X_train, y_train)

# Transformar val y test
X_val_te = te.transform(X_val)
X_test_te = te.transform(X_test)


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
        X_train_te, y_train,
        eval_set=[(X_val_te, y_val)],
        eval_metric='auc'
    )

    
    model.fit(X_train_te, y_train)     
    y_score = model.predict_proba(X_val_te)[:,1]
    auc = roc_auc_score(y_val, y_score)
    return auc


#study = optuna.create_study(direction="maximize")
#study.optimize(objective, n_trials=20)


#paramsLGBM = study.best_params
#paramsLGBM


'''
paramsLR = study.best_params
lgbm_model = lgb.LGBMClassifier(**paramsLR)
X_train_full = pd.concat([X_train_te, X_val_te])
y_train_full = pd.concat([y_train, y_val])
lgbm_model.fit(X_train_full, y_train_full)     
y_score = lgbm_model.predict_proba(X_test_te)[:,1]
auc = roc_auc_score(y_test, y_score)
auc
'''


X_train_full = pd.concat([X_train_te, X_val_te, X_test_te])
y_train_full = pd.concat([y_train, y_val, y_test])
lgbm_model = lgb.LGBMClassifier(**{'learning_rate': 0.08133332392325474,
 'num_leaves': 103,
 'max_depth': 12,
 'min_data_in_leaf': 197,
 'subsample':0.6042016570161177,
 'colsample_bytree': 0.5011950995843608,
 'n_estimators': 646,
 'reg_alpha': 0.3621293934921277,
 'reg_lambda': 4.254583202282918,
 'min_gain_to_split': 0.8396616036784496,
 'random_state':42})
lgbm_model.fit(X_train_full, y_train_full) 


X_test


test = pd.read_csv('/kaggle/input/playground-series-s5e11/test.csv')
test['loan_charge'] = test['loan_amount']/test['annual_income']
test['total_credit'] = test['interest_rate']*test['loan_amount']
test['DTI/LA'] = test['debt_to_income_ratio']*test['loan_amount']
test['risk_sign'] = test['loan_amount']/test['credit_score']
test['gender'] = gender_encoder.transform(test['gender'])
test['marital_status'] = marital_encoder.transform(test['marital_status'])
test['annual_income'] = scaler_income.transform(test[['annual_income']])
test['credit_score'] = credit_score_scaler.transform(test[['credit_score']])
test['debt_to_income_ratio'] = debt_to_income_ratio_scaler.transform(test[['debt_to_income_ratio']])
test['loan_amount'] = loan_amount_scaler.transform(test[['loan_amount']])
test['interest_rate'] = interest_rate_scaler.transform(test[['interest_rate']])

full = test.drop(columns = ['id'])
full = te.transform(full)
full.head()


X = full
y_score = lgbm_model.predict_proba(X)[:,1]
test['loan_paid_back'] = y_score
submit = test[['id','loan_paid_back']]
submit = submit.set_index('id')
submit.to_csv('submission.csv')
submit




