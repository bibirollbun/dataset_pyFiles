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


#### Import the required python libraries
import pandas as pd
import time
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import optuna
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#### Import data
train = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
print(train)


!pip install optuna-integration[xgboost]


#### View data types and shapes
print(train.dtypes)
#You can see that each record is of type object except for id, Compartments, Weight, Price(Target)

#Shape
print(train.shape)
print(test.shape)

train_object=train.select_dtypes(include=['object'])


#### View unique values for “object” type data
for col in list(train_object.columns):
    print(f'{col} includes:{train_object[col].unique()}')
# Simply, you can see that there are null values for each object type data, which need to be processed . 
# And there are not many categories of unique values for each feature, you can consider doing dummy variable processing


#### Check the status of vacancy values
train.isnull().sum()
print(pd.concat([train.dtypes,train.isnull().sum()],axis=1,keys=['data_type','null_values']).sort_values(by='null_values',ascending=False))#1按列合并,0按行合并

# The data feature of type obeject has more null values, try to set it to “Else” for null values, and the numeric feature-'weight' also has some null values, but the percentage is very small, 
# you can see that the 'Compartments' column does not have a null value, when there are more 'compartments', intuitively, the backpack's 'weight' will be larger, so here we use the average of all records with the same number of 'compartments' for 'weight' to supplement the null value of 'weight'. 
# Therefore, we use the average of all 'weight' records with the same number of 'compartments' to supplement the null value of 'weight'.


def data_processing(data):
    mean_weight_by_compartment = data.groupby('Compartments')['Weight Capacity (kg)'].transform('median')
    data['Weight Capacity (kg)'] = data['Weight Capacity (kg)'].fillna(mean_weight_by_compartment)
    
    object_data=data.select_dtypes(include=['object'])
    object_columns = list(object_data.columns)
    
    for col in object_columns:
        data[col] = data[col].fillna('Else')
        data[col] = data[col].astype('category')
    data = pd.get_dummies(data, columns=object_columns, drop_first=True)
    return data


train = data_processing(train)
# View null values after processing
pd.concat([train.dtypes,train.isnull().sum()],axis=1,keys=['data_type','null_values']).sort_values(by='null_values',ascending=False)#1 merged by columns, 0 merged by rows


test = data_processing(test)
# View null values after processing
pd.concat([test.dtypes,test.isnull().sum()],axis=1,keys=['data_type','null_values']).sort_values(by='null_values',ascending=False)#1按列合并,0按行合并


#### Creat datasets for subsequent model training, tuning
X = train.drop(columns=['id','Price'])
y = train['Price']
X_test = test.drop(columns=['id'])
X.shape,X_test.shape


# Set the actual training data and validation data
X_train,X_valid,y_train,y_valid = train_test_split(X,y,test_size=0.2,random_state=100)


#Converting data to DMatrix format - the format recommended by the XGB documentation for data processing, which helps accelerate model training
d_Xtrain = xgb.DMatrix(data=X_train, label=y_train, enable_categorical=False)
d_Xvalid = xgb.DMatrix(data=X_valid, label=y_valid, enable_categorical=False)
d_Xtest  = xgb.DMatrix(data=X_test, enable_categorical=False)
d_Xalltrain = xgb.DMatrix(data=X, label=y,enable_categorical=False)


## Define trial and use optuna for parameter combination tests

# XGB parameters tunning refered to the following documents:
# 1.XGBoost Parameters : https://xgboost.readthedocs.io/en/stable/parameter.html
# 2.https://randomrealizations.com/posts/xgboost-parameter-tuning-with-optuna/

def objective_xgb(trial):
    params = {
        #'tree_method': trial.suggest_categorical('tree_method', ['approx', 'hist']),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 200),
        'subsample': trial.suggest_float('subsample', 0.8, 1.0),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.5, 1.0),
        #'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        #'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.5, 1.0),
        'gamma':trial.suggest_float('gamma',0,3),
        'reg_lambda': trial.suggest_float('reg_lambda',0,3),
        'reg_alpha': trial.suggest_float('reg_alpha',0,3),
        'learning_rate': trial.suggest_float('learning_rate',0.01,0.3),
        #'learning_rate': 0.3,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method':'hist'
        #'n_estimators':trial.suggest_int('n_estimators',50,500)
    }
    num_boost_round = 5000
    pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f"valid-rmse")
    ##pruning_callback = optuna.integration.XGBoostPruningCallback(trial, f"validation_{fold_idx}-rmse")
    model = xgb.train(params=params, dtrain=d_Xtrain, 
                          num_boost_round=num_boost_round,
                          evals=[(d_Xtrain, 'train'), (d_Xvalid, 'valid')],
                          early_stopping_rounds=20,
                          verbose_eval=0,
                          callbacks=[pruning_callback])
    trial.set_user_attr('best_iteration', model.best_iteration)
    return model.best_score


# Create Optuna study object with the goal of minimizing RMSE
sampler_xgb = optuna.samplers.TPESampler(seed=100)
study_xgb = optuna.create_study(direction="minimize",sampler=sampler_xgb)

# 200 parameter combination trials
study_xgb.optimize(objective_xgb, n_trials=200)

## Perform 2-minute parameter combination trials
# tic = time.time()
# while time.time() - tic < 120:
#     study_xgb.optimize(objective_xgb, n_trials=1)

#Input best_params、best_rmse_value、best_trial
print(study_xgb.best_params)
print(study_xgb.best_value)
print(study_xgb.best_trial)


params = {}
params.update(study_xgb.best_params)

model_best_params= xgb.train(
    params=params,
    dtrain = d_Xtrain,
    num_boost_round= 5000,
    evals=[(d_Xtrain,'train'),(d_Xvalid,'valid')],
    early_stopping_rounds=20,
    verbose_eval=0
    )
print(model_best_params.best_score)
print(model_best_params.best_iteration)

final_model = xgb.train(
    params=params,
    dtrain = d_Xalltrain,
    num_boost_round=model_best_params.best_iteration,
    verbose_eval=0
    )
y_test_pre = final_model.predict(d_Xtest)

sample_submission['Price']=y_test_pre
sample_submission.to_csv("submission_xgb.csv",index=False)


# Data Transformation - Boosting the Speed of Model Training
all_train_data = lgb.Dataset(X,label=y)
train_data= lgb.Dataset(X_train,label=y_train,params={'feature_pre_filter': False})
valid_data = lgb.Dataset(X_valid,label=y_valid,params={'feature_pre_filter': False})
test_data= lgb.Dataset(X_test)


## Define trial and use optuna for parameter combination tests

# LGBM parameters tunning refered to the following documents
# 1.https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html
# 2.https://lightgbm.readthedocs.io/en/stable/Parameters.html
# 3.https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html#lightgbm.early_stopping

#According to the information provided in the second URL (Parameters Tuning document provided by the official website), 'num_leaves' should be set equal to 2^(max_depth), which should be less than n2^(max_depth) in practical applications.
#but when I try to add a limit to 'num_leaves' in objective_lgb myself, the model doesn't work well????

def objective_lgb(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 4, 10),
        'num_leaves': trial.suggest_int('num_leaves',15,60),
        'min_data_in_leaf': trial.suggest_categorical('min_data_in_leaf', [100,200,300,400,500]),
        'num_iterations': trial.suggest_int('num_iterations', 100, 500),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'reg_alpha': trial.suggest_float('lambda_l1', 0, 3),
        'reg_lambda': trial.suggest_float('lambda_l2', 0, 3),
        'early_stopping_round': 20,
        'objective': 'regression',
        'num_threads': 4,
        'device_type': 'cpu',
        'random_state': 100,
        'verbosity': -1,
        'metric': 'rmse',
        #'bagging_fraction': trial.suggest_float('bagging_fraction',0,1),
        #'n_estimators': trial.suggest_int('n_estimators',100,300),
        #'feature_fraction': trial.suggest_float('feature_fraction',0,1),
    }
    lgb_model = lgb.train(
        params=params,
        train_set=train_data,
        valid_sets=valid_data,
        callbacks=[lgb.early_stopping(stopping_rounds=20)]
    )
    y_valid_pre=lgb_model.predict(X_valid)
    return np.sqrt(mean_squared_error(y_valid,y_valid_pre)) #Each trial will return the RMSE of the validation set


## Create Optuna study object with the goal of minimizing RMSE
sampler_lgb = optuna.samplers.TPESampler(seed=100)
study_lgb = optuna.create_study(direction="minimize",sampler=sampler_lgb)

# 200 parameter combination trials
study_lgb.optimize(objective_lgb, n_trials=200)

#best_params、best_rmse_value、best_trial
print(study_lgb.best_params)
print(study_lgb.best_value)
print(study_lgb.best_trial)


best_params_lgb= study_lgb.best_params
final_model=lgb.train(
    best_params_lgb,
    train_set=all_train_data
)

y_test_pre=final_model.predict(X_test)
sample_submission['Price']=y_test_pre
sample_submission.to_csv("submission_lgb.csv",index=False)

