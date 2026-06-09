!pip install -qq scikit-learn==1.6.1


#
# Libraries
#

# General
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os, string, re, random, gc, pickle, math,warnings
import json
from itertools import *
from datetime import date
from tqdm.keras import TqdmCallback
from tqdm import tqdm
import h5py
from catboost import CatBoostRegressor
import optuna

# Sklearn
from sklearn.model_selection import *
from sklearn.feature_extraction import *
from sklearn.metrics import *
from sklearn.metrics import pairwise
from sklearn.preprocessing import *
from sklearn.utils import *
from sklearn.pipeline import *
from sklearn.compose import *

# Stats
import scipy
from scipy.stats import *
from scipy.sparse import csr_matrix

# Setting
pd.set_option('max_colwidth',None)
seed = 505
warnings.simplefilter('ignore')

data_path = None

for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename.endswith('h5'):
            data_path = os.path.join(dirname, filename)


#
# Load data
#

# train data
with h5py.File(data_path,'r') as f:
    train_spots = f['spots/Train']
    train_spot_tables = {slide_name: pd.DataFrame(np.array(train_spots[slide_name])) for slide_name in train_spots.keys()}

train_ds = pd.concat(train_spot_tables.values(), ignore_index=True)

# test data
with h5py.File(data_path,'r') as f:
    test_spots = f['spots/Test']
    test_spot_tables = {slide_name: pd.DataFrame(np.array(test_spots[slide_name])) for slide_name in test_spots.keys()}

test_ds = pd.concat(test_spot_tables.values(), ignore_index=True)


# view
train_ds.head()


#
# Preprocessing
#

# feature engineering
x = train_ds[['x', 'y']]  # Use spatial coordinates
y = train_ds.iloc[:, 2:]  # All other columns are cell-type labels

# train-test split
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.02, random_state=seed)

# scale 
scaler = MinMaxScaler()
scaled_val = scaler.fit_transform(x_train)
x_train_sc = pd.DataFrame(scaled_val,columns=x_train.columns)

scaled_val = scaler.fit_transform(x_valid)
x_valid_sc = pd.DataFrame(scaled_val,columns=x_valid.columns)

# shape info
print(f"  Training shape: {x_train_sc.shape} | Validation shape: {x_valid.shape}")


#
# Training, Prediction & Evaluation - Baseline
#
catbr = CatBoostRegressor(verbose=False,objective='MultiRMSE')
catbr.fit(x_train_sc, y_train,early_stopping_rounds=100)

# prediction
pred = catbr.predict(x_valid)

# rmse
rmse = root_mean_squared_error(y_valid,pred)

print('RMSE: {:.5}'.format(rmse))


#
# Optuna 
#

def objective(trial):
    param = {}
    param['learning_rate'] = trial.suggest_discrete_uniform("learning_rate", 0.01, 0.03, 0.01)
    # param['iterations'] = trial.suggest_int('iterations', 1000, 2000,1000)
    param['depth'] = trial.suggest_int('depth', 3, 5)
    param['l2_leaf_reg'] = trial.suggest_discrete_uniform('l2_leaf_reg', 1.0, 5.5, 0.5)
    param['min_child_samples'] = trial.suggest_categorical('min_child_samples', [1, 4, 8, 16, 32])
    param['grow_policy'] = 'Depthwise'
    param['iterations'] = 1000
    param['objective'] = 'MultiRMSE'
    # param['od_type'] = 'iter'
    param['od_wait'] = 20
    param['random_state'] = seed
    param['logging_level'] = 'Silent'
    #param['task_type'] ='GPU'
    
    regressor = CatBoostRegressor(**param)
    regressor.fit(x_train_sc, y_train,early_stopping_rounds=100)
    #loss = mean_squared_error(y_valid, regressor.predict(x_valid_sc))
    rmse = root_mean_squared_error(y_valid, regressor.predict(x_valid_sc))
    
    return rmse

study = optuna.create_study(study_name=f'catboost-seed{seed}')
study.optimize(objective, n_trials=20, n_jobs=-1, timeout=24000)

trial = study.best_trial

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))


# 
# Prediction with best params
#

# best params & adding new items
best_param = study.best_params
best_param['logging_level'] = 'Silent'
best_param['objective'] = 'MultiRMSE'

# test data
x_tst = test_ds[['x', 'y']]  

# scale 
scaler = MinMaxScaler()
scaled_val = scaler.fit_transform(x_tst)
x_tst_sc = pd.DataFrame(scaled_val,columns=x_tst.columns)

# training
catbr_bst = CatBoostRegressor(**best_param)
catbr_bst.fit(x_train_sc, y_train,early_stopping_rounds=100)

# prediction
pred_tst = catbr_bst.predict(x_tst_sc)

# prepare submission file
sub_df = pd.DataFrame(pred_tst, columns=y.columns)
sub_df.insert(0, 'ID', test_ds.index)
sub_df.to_csv("/kaggle/working/submission.csv", index=False)
print("Submission file 'submission.csv' created!")

