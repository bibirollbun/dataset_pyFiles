%%time

!pip install -qq lifelines
!pip install -qq hillclimbers

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

SEED = 42
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
                 n_splits=n_splits,early_stop=False,num_classes=0,cat_features=None,ordinal_encoder=ohe_cols,
                 fold_type='GKF')

base.X_train = base.X_train.fillna(-1)
base.X_test = base.X_test.fillna(-1)


%%time

base.X_train.head()


%%time

base.X_test.head()


%%time

params =  {'n_estimators': 200, 'max_depth': 5, 'colsample_bytree': 0.5359752614980476,
            'subsample': 0.7271274739921461, 'learning_rate': 0.011247656752870117, 'min_child_weight': 74}

XGBresult = base.Train_ML(params,'XGB', y_log=True, g_col='group')


%%time

sample["num_sold"] = XGBresult[1]
sample.to_csv("submission.csv", index=False)
print("Sub shape:", sample.shape)
sample.head()

