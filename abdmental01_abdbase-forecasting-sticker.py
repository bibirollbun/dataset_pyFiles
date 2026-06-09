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

rLGBM = base.Train_ML(params,'LGBM',e_stop=200, y_log=True, g_col='group')  # 0.07531 LB


%%time

params1 = {
    'n_estimators': 802,
    'max_depth': 4,
    'colsample_bytree': 0.4087726844027313,
    'subsample': 0.5150029934968837,
    'learning_rate': 0.0885280505784011,
    'min_child_samples': 98
}

rLGBM1 = base.Train_ML(params1, 'LGBM', e_stop=100, y_log=True, g_col='group') # 0.07704 LB


%%time

from hillclimbers import climb_hill, partial

oof_t= np.vstack([rLGBM[0], rLGBM1[0]])
oof_t_df = pd.DataFrame(oof_t.T, columns=['LGBM','LGBM1'])

test_m = np.vstack([rLGBM[1], rLGBM1[1]])
test_m_df = pd.DataFrame(test_m.T, columns=['LGBM','LGBM1'])

from sklearn.metrics import mean_absolute_percentage_error

hc_test_pred_probs, hc_oof_pred_probs = climb_hill(
    train=pd.concat([base.X_train, base.train_data['num_sold']], axis=1),
    oof_pred_df=oof_t_df, 
    test_pred_df=test_m_df,
    target='num_sold',
    objective='minimize', 
    eval_metric = partial(mean_absolute_percentage_error), 
    negative_weights=False, 
    precision=0.001, 
    plot_hill=False, 
    plot_hist=False,
    return_oof_preds=True,
)


%%time

sample["num_sold"] = hc_test_pred_probs
sample.to_csv("submission.csv", index=False)
print("Sub shape:", sample.shape)
sample.head()

