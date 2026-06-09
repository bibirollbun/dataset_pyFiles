import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)
import optuna


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df = pd.read_csv('/kaggle/input/playground-series-s5e12/train.csv')
numeric_df = df.select_dtypes(include=["number"])
numeric_df["ldl_hdl_ratio"] = numeric_df["ldl_cholesterol"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["cholesterol_ratio"] = numeric_df["cholesterol_total"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["activity_bmi"] = numeric_df["physical_activity_minutes_per_week"] / (numeric_df["bmi"] + 1e-6)
numeric_df["age_bmi"] = numeric_df["age"] * numeric_df["bmi"]
numeric_df["age_activity"] = numeric_df["age"] * numeric_df["physical_activity_minutes_per_week"]
numeric_df["age_triglycerides"] = numeric_df["age"] * numeric_df["triglycerides"]
numeric_df["high_bmi"] = (numeric_df["bmi"] > 30).astype(int)
numeric_df["high_triglycerides"] = (numeric_df["triglycerides"] > 150).astype(int)
numeric_df.head()


features = [
    "family_history_diabetes",
    "physical_activity_minutes_per_week",
    "activity_bmi",
    "age_bmi",
    "age_triglycerides",
    "age_activity",
    "age",
    "ldl_hdl_ratio",
    "triglycerides",
    "cholesterol_ratio",
    "bmi"
]

X = numeric_df[features]
y = numeric_df["diagnosed_diabetes"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, 
                                                random_state=42,stratify=y_test)


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
#study.optimize(objective, n_trials=50)


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
lgbm_model = lgb.LGBMClassifier(**{'learning_rate': 0.07183521969292042,
 'num_leaves': 69,
 'max_depth': 5,
 'min_data_in_leaf': 155,
 'subsample': 0.6038402866038635,
 'colsample_bytree': 0.7556756642569674,
 'n_estimators': 762,
 'reg_alpha': 1.7469018618683667,
 'reg_lambda': 5.229358219612677,
 'min_gain_to_split': 0.11624493766312131,
 'random_state':42})
lgbm_model.fit(X_train_full, y_train_full) 


test = pd.read_csv('/kaggle/input/playground-series-s5e12/test.csv')
numeric_df = test.select_dtypes(include=["number"])
numeric_df["ldl_hdl_ratio"] = numeric_df["ldl_cholesterol"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["cholesterol_ratio"] = numeric_df["cholesterol_total"] / (numeric_df["hdl_cholesterol"] + 1e-6)
numeric_df["activity_bmi"] = numeric_df["physical_activity_minutes_per_week"] / (numeric_df["bmi"] + 1e-6)
numeric_df["age_bmi"] = numeric_df["age"] * numeric_df["bmi"]
numeric_df["age_activity"] = numeric_df["age"] * numeric_df["physical_activity_minutes_per_week"]
numeric_df["age_triglycerides"] = numeric_df["age"] * numeric_df["triglycerides"]
numeric_df["high_bmi"] = (numeric_df["bmi"] > 30).astype(int)
numeric_df["high_triglycerides"] = (numeric_df["triglycerides"] > 150).astype(int)
full = numeric_df


X = full[features]
y_score = lgbm_model.predict_proba(X)[:,1]
test['diagnosed_diabetes'] = y_score
submit = test[['id','diagnosed_diabetes']]
submit = submit.set_index('id')
submit.to_csv('submission.csv')
submit







