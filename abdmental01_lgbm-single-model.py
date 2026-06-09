%%time

!pip install -qq lifelines

import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from colorama import Fore
from IPython.display import clear_output

from sklearn.model_selection import *
from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier
import catboost as cb
from lightgbm import LGBMRegressor
import lightgbm as lgb
from tqdm import tqdm
import numpy as np


%%time

SEED = 114514
n_splits = 5

!git clone https://github.com/muhammadabdullah0303/AbdML

import sys
sys.path.append('/kaggle/working/repository')

from AbdML.main import AbdBase

train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
sample = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')
train = train.dropna(subset=['num_sold'])

train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

cat_c = ['country', 'store', 'product','month_name','day_of_week']

ohe_cols = {'cat_c': cat_c}

base = AbdBase(train_data=train, test_data=test, target_column='num_sold',gpu=False,handle_date=True,
                 problem_type="regression", metric="mape", seed=SEED,
                 n_splits=n_splits,early_stop=True,num_classes=0,cat_features=None,ohe_fe=ohe_cols,
                 fold_type='GKF')


%%time

base.X_train.head()


%%time

base.X_test.head()


%%time

params = {'n_estimators': 848, 'max_depth': 4, 'colsample_bytree': 0.40922204719271094, 
          'subsample': 0.5185247148796622, 'learning_rate': 0.08812296173534281, 'min_child_samples': 91}

rLGBM = base.Train_ML(params,'LGBM',e_stop=200, y_log=True, g_col='group')  


%%time

sample["num_sold"] = rLGBM[1]
sample.to_csv("submission.csv", index=False)
print("Sub shape:", sample.shape)
sample.head()

