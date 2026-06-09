# Install sklearn
!pip install scikit-learn==1.5.2 -q
# Install TabPFN
!pip install tabpfn -q

# TabPFN Community installs optional functionalities around the TabPFN model
# These include post-hoc ensembles, interpretability tools, and more
!git clone https://github.com/PriorLabs/tabpfn-extensions
!pip install -e tabpfn-extensions[post_hoc_ensembles,interpretability,hpo] -q
!pip install datasets


import os

# Setup Imports
import pandas as pd
import numpy as np

from sklearn.datasets import load_breast_cancer, load_diabetes, load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.datasets import fetch_openml
from sklearn.preprocessing import LabelEncoder
from IPython.display import display, Markdown, Latex

# Baseline Imports
from xgboost import XGBClassifier, XGBRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from catboost import CatBoostClassifier, CatBoostRegressor

import torch

from tabpfn import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor

if not torch.cuda.is_available():
    raise SystemError('GPU device not found. For fast training, please enable GPU. See section above for instructions.')


import numpy as np
import pandas as pd
from copy import deepcopy
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor, DMatrix



inventory = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/inventory.csv').drop(['warehouse','product_unique_id'],axis=1)
inventory.head()


calendar = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/calendar.csv', parse_dates=['date'])
calendar.loc[calendar['holiday_name'].isna(), 'holiday'] = 0 # V3
calendar['last_holiday_date'] = calendar['date']
calendar['next_holiday_date'] = calendar['date']
calendar.loc[calendar['holiday'] == 0, ['last_holiday_date','next_holiday_date']] = np.nan
calendar['last_holiday_date'] = calendar.sort_values('date').groupby('warehouse')['last_holiday_date'].ffill()
calendar['next_holiday_date'] = calendar.sort_values('date').groupby('warehouse')['next_holiday_date'].bfill()
calendar['days_since_last_holiday'] = ((calendar['date'] - calendar['last_holiday_date']).dt.days)
calendar['days_to_next_holiday'] = ((calendar['next_holiday_date'] - calendar['date']).dt.days)
calendar['day_before_holiday'] = calendar['days_to_next_holiday'] == 1
calendar['day_after_holiday'] = calendar['days_since_last_holiday'] == 1
calendar.drop(['last_holiday_date','next_holiday_date'],axis=1,inplace=True)
calendar.drop(['days_since_last_holiday','days_to_next_holiday'],axis=1,inplace=True)
calendar.drop(['shops_closed','winter_school_holidays','school_holidays','holiday_name','day_before_holiday','day_after_holiday'],axis=1,inplace=True)
calendar


train = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_train.csv', parse_dates=['date'])
train['id'] = train['unique_id'].astype('str') + '_' + train['date'].astype('str')
train.set_index('id',inplace=True)
train = train[~train['sales'].isna()]
train = train.reset_index().merge(inventory, on='unique_id').set_index('id').loc[train.index]
train = train.reset_index().merge(calendar, on=['date','warehouse']).set_index('id').loc[train.index]

test = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/sales_test.csv', parse_dates=['date'])
test['id'] = test['unique_id'].astype('str') + '_' + test['date'].astype('str')
test.set_index('id',inplace=True)
test = test.reset_index().merge(inventory, on='unique_id').set_index('id').loc[test.index]
test = test.reset_index().merge(calendar, on=['date','warehouse']).set_index('id')




X_train = train.drop('sales',axis=1)
y_train = train['sales']
train_availability = X_train['availability']
X_train.drop('availability',inplace=True,axis=1)
weights = pd.read_csv('/kaggle/input/rohlik-sales-forecasting-challenge-v2/test_weights.csv').set_index('unique_id')
X_train_weights = X_train['unique_id'].map(weights['weight'])


cat_cols = ['unique_id'] + list(X_train.columns[X_train.dtypes == 'object'])


# Cutting all data prior to 2022 seems to help. This could be due to COVID effects, and also the fact that there is little data from the Germany warehouses before 2022.
X_train = X_train[X_train['date'] >= '2023-01-01']
y_train = y_train.loc[X_train.index]
X_train_weights = X_train_weights.loc[X_train.index]

# all_data[cat_cols] = all_data[cat_cols].astype('str').astype('category')


X_train.shape


lr = .1
es = 10
n_est = round(5000/lr)
seed = 2
base_params = {
    'n_estimators':n_est
    ,'learning_rate':lr
    ,'verbosity':0
    ,'enable_categorical':True
    ,'early_stopping_rounds':es
    ,'random_state':seed
    ,'objective':'reg:squarederror'
    ,'eval_metric':'rmse'
    ,'device':'cuda'
    ,'reg_lambda':0
    ,'min_child_weight':1
}



class BlockingTimeSeriesSplit():
    def __init__(self, n_splits):
        self.n_splits = n_splits
    
    def get_n_splits(self, X, y, groups):
        return self.n_splits
    
    def split(self, X, y=None, groups=None):
        n_samples = len(X)
        k_fold_size = n_samples // self.n_splits
        indices = np.arange(n_samples)

        margin = 0
        for i in range(self.n_splits):
            start = i * k_fold_size
            stop = start + k_fold_size
            mid = int(0.5 * (stop - start)) + start
            yield indices[start: mid], indices[mid + margin: stop]


import gc
# Inference is memory intensive and large workloads must be processed in batches
def batch_predict(model, X_test, batch_size=1000):
    predictions = []
    for i in range(0, len(X_test), batch_size):
        torch.cuda.empty_cache()
        batch = X_test[i:i + batch_size]
        batch_predictions = model.predict(batch)
        predictions.extend(batch_predictions)
        gc.collect()
        print(i)
    return np.array(predictions)


import warnings
warnings.simplefilter("ignore")

drop_cols = ['date','name','L1_category_name_en',
             # '', 'warehouse',
             'name', 'L2_category_name_en', 'L3_category_name_en', 'L4_category_name_en']
oof_preds = []
test_preds = []
pow_trans=True
pow_degree=.5


kf_params = {
    'n_splits':3
    ,'n_repeats':1
    ,'random_state':seed
}


# kf = RepeatedKFold(**kf_params)
kf = BlockingTimeSeriesSplit(n_splits=90)
X,y = deepcopy(X_train),deepcopy(y_train)
# X=X[:10000]
X = X.dropna()
X.drop(drop_cols,axis=1,inplace=True)
X = pd.get_dummies(X)
test_copy = deepcopy(test)
test_copy.drop(drop_cols,axis=1,inplace=True)
test_copy = pd.get_dummies(test_copy)

oof_pred_df = pd.DataFrame(index=X.index, columns=[
    'Pred_{0}'.format(i) for i in range(kf_params['n_repeats'])])

tabPFN = TabPFNRegressor(random_state=42, device="cuda")
for i, (idx_t, idx_v) in enumerate(kf.split(X)):
    print(f'CV {i}: {len(idx_t)}')
    X_t, X_v = X.iloc[idx_t], X.iloc[idx_v]        
    y_t, y_v = y.loc[X_t.index], y.loc[X_v.index]
    if pow_trans:
        y_t, y_v = np.power(y_t, pow_degree), np.power(y_v, pow_degree)
    tabPFN.fit(X_t, y_t)
    # Make predictions
    if i % 10 == 9 or i in [85,86,87,88,89]:
        predictions_test = batch_predict(tabPFN, test_copy, batch_size=10000)
        model_test_preds = np.power(predictions_test.clip(0), 1/pow_degree) if pow_trans else predictions_test.clip(0)
        test_preds.append(model_test_preds)



test_pred_df = pd.DataFrame(np.transpose(test_preds), index=test.index)
test_sub = test_pred_df.mean(axis=1)
test_sub.name = 'sales_hat'
test_sub.to_csv('submission_tabpdn_woFE.csv')

