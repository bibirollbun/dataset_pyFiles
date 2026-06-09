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


!pip install /kaggle/input/pip-install-lifelines/autograd-1.7.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/autograd-gamma-0.5.0.tar.gz
!pip install /kaggle/input/pip-install-lifelines/interface_meta-1.3.0-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/formulaic-1.0.2-py3-none-any.whl
!pip install /kaggle/input/pip-install-lifelines/lifelines-0.30.0-py3-none-any.whl


import pandas as pd
import numpy as np
import lifelines
from lifelines import KaplanMeierFitter, NelsonAalenFitter
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from lifelines.utils import concordance_index
import lightgbm as lgb
from termcolor import colored
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv', index_col='ID')
test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv', index_col='ID')
sub = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv', index_col='ID')
data_description = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/data_dictionary.csv')


# Categorical & Numeric columns
cat_cols = []
num_cols = []
for v, t in data_description[['variable', 'type']].values:
    if t == 'Categorical' and v != 'efs':
        cat_cols.append(v)
    elif not v in ['efs_time', 'efs']:
        num_cols.append(v)


naf = NelsonAalenFitter()
naf.fit(train['efs_time'], train['efs'])
train['naf_label'] = -naf.cumulative_hazard_at_times(train['efs_time']).values
train.loc[train.efs == 0, 'naf_label'] -= 0.15


kmf = KaplanMeierFitter()
kmf.fit(train['efs_time'], train['efs'])
train['km_label'] = kmf.survival_function_at_times(train['efs_time']).values
train.loc[train.efs == 0, 'km_label'] -= 0.15


def cindex_xgb(preds, dtrain):
    labels = dtrain.get_label()
    times = dtrain.get_weight()
    c_index = concordance_index(times, preds, labels)
    return 'cindex', -c_index


# XGBoost Parameters
xgb_naf_params = {
    #　'enable_categorical': True,
    "objective": "reg:squarederror",
    "verbosity": 0,
    # 'n_estimators': 9400,
    # 'learning_rate': 0.01462545658882346,
    # 'max_depth': 4,
    # 'subsample': 0.8427706960687078,
    # 'colsample_bytree': 0.2630880900000106,
    # 'min_child_weight': 50,
    'reg_lambda': 29.0,
    "max_depth":3,
    "colsample_bytree":0.7129400756425178,
    "subsample":0.8185881823156917,
    "n_estimators":20_000,
    "learning_rate" :0.04425768131771064,
    "eval_metric":"auc",
    "feval":cindex_xgb,
    # "early_stopping_rounds":50,
    "scale_pos_weight":1.5379160847615545,
    "min_child_weight":4,
    "enable_categorical":True,
    "gamma":3.1330719334577584
}

xgb_km_params = {
    #　'enable_categorical': True,
    "objective": "reg:squarederror",
    "verbosity": 0,
    # 'n_estimators': 9400,
    # 'learning_rate': 0.01462545658882346,
    # 'max_depth': 4,
    # 'subsample': 0.8427706960687078,
    # 'colsample_bytree': 0.2630880900000106,
    # 'min_child_weight': 50,
    'reg_lambda': 29.0,
    "max_depth":3,
    "colsample_bytree":0.7129400756425178,
    "subsample":0.8185881823156917,
    "n_estimators":20_000,
    "learning_rate" :0.04425768131771064,
    "eval_metric":"auc",
    "feval":cindex_xgb,
    # "early_stopping_rounds":50,
    "scale_pos_weight":1.5379160847615545,
    "min_child_weight":4,
    "enable_categorical":True,
    "gamma":3.1330719334577584
}

# LightGBM Parameters
lgbm_naf_params = {
    'objective': 'regression',
    "bagging_freq": 1,
    'n_estimators': 9800,
    'learning_rate': 0.0025562611410098906,
    'max_depth': 11,
    'subsample': 0.7000302358347922,
    'colsample_bytree': 0.34454787171802054,
    'min_data_in_leaf': 52,
    'reg_lambda': 0.01478462915287414,
    'random_state': 43,
    'verbose': -1,
    'metric':'concodance_index'
    
}

lgbm_km_params = {
    'objective': 'regression',
    'verbose': -1,
    "bagging_freq": 1,
    'n_estimators': 9800,
    'learning_rate': 0.0025562611410098906,
    'max_depth': 11,
    'subsample': 0.7000302358347922,
    'colsample_bytree': 0.34454787171802054,
    'min_data_in_leaf': 52,
    'metric' : 'concordance_index',
    'reg_lambda': 0.01478462915287414,
    'random_state': 43,
}



def cindex_xgb(preds, dtrain):
    labels = dtrain.get_label()
    times = dtrain.get_weight()
    c_index = concordance_index(times, preds, labels)
    return 'cindex', -c_index



train_T = train.T

train_T_null_series = train_T.isnull().sum()

train['Null_number'] = train_T_null_series

train.isnull().sum()


test_T = test.T

test_T_null_series = test_T.isnull().sum()

test['Null_number'] = test_T_null_series

test.isnull().sum()



target_cols = ['efs', 'efs_time', 'km_label', 'naf_label']
all_preds = []
all_efs = []
all_efs_time = []
scores = []
cv = KFold(n_splits=15, shuffle = True,  random_state=42)



for i, (train_indexes, val_indexes) in enumerate(cv.split(train)):
    train_data = train.iloc[train_indexes]
    val_data = train.iloc[val_indexes]
    cat_cols = train_data.drop(columns=target_cols).select_dtypes(include=object).columns.values.tolist()

    train_data[cat_cols] = train_data[cat_cols].astype('category')
    val_data[cat_cols] = val_data[cat_cols].astype('category')
    all_efs += list(val_data['efs'].values)
    all_efs_time += list(val_data['efs_time'].values)
    
    train_lgb_naf = lgb.Dataset(train_data.drop(columns=target_cols), label=train_data['naf_label'], categorical_feature=cat_cols)
    train_lgb_km = lgb.Dataset(train_data.drop(columns=target_cols), label=train_data['km_label'], categorical_feature=cat_cols)

    # best_naf = lgb.train(lgbm_naf_params, train_lgb_naf, 1000, valid_sets=[train_lgb_naf])
    # best_km = lgb.train(lgbm_km_params, train_lgb_km, 1000, valid_sets=[train_lgb_km])
    # best_naf = lgbm_naf_params['metric'] = 'concordance_index'
    best_naf = lgb.train(lgbm_naf_params, train_lgb_naf, 1000, valid_sets=[train_lgb_naf])
    # best_km = lgbm_km_params['metric'] = 'concordance_index'
    best_km = lgb.train(lgbm_km_params, train_lgb_km, 1000, valid_sets=[train_lgb_km])
    
    d_train_naf = xgb.DMatrix(train_data.drop(columns=target_cols), label=train_data['naf_label'], weight=train_data['efs_time'], enable_categorical=True)
    d_train_km = xgb.DMatrix(train_data.drop(columns=target_cols), label=train_data['km_label'], weight=train_data['efs_time'], enable_categorical=True)

    xgb_naf = xgb.XGBRegressor(**xgb_naf_params,
                               # eval_metric=cindex_xgb,
                               eval_set=[(val_data.drop(columns=target_cols), val_data['naf_label'])], verbose=False)
    xgb_km = xgb.XGBRegressor(**xgb_km_params, 
                              # eval_metric=cindex_xgb,
                              eval_set=[(val_data.drop(columns=target_cols), val_data['km_label'])], verbose=False)

    # xgb_naf.fit(train_data.drop(columns=target_cols), train_data['naf_label'])
    
    # xgb_naf.fit(train_data.drop(columns=target_cols), train_data['naf_label'], eval_set=[(val_data.drop(columns=target_cols), val_data['naf_label'])], eval_metric=cindex_xgb, verbose=False)

    xgb_naf.fit(train_data.drop(columns=target_cols), train_data['naf_label'])
    xgb_km.fit(train_data.drop(columns=target_cols), train_data['km_label'])

    preds_lgb_naf = best_naf.predict(val_data.drop(columns=target_cols))
    preds_lgb_km = best_km.predict(val_data.drop(columns=target_cols))
    preds_xgb_naf = xgb_naf.predict(val_data.drop(columns=target_cols))
    preds_xgb_km = xgb_km.predict(val_data.drop(columns=target_cols))

    preds = (preds_lgb_naf*0.3 + preds_lgb_km*0.3 + preds_xgb_naf*0.2 + preds_xgb_km*0.2)
    all_preds += list(preds)
    score = concordance_index(val_data['efs_time'], -preds, val_data['efs'])
    scores.append(score)

    print(f'Fold #{i} C-index_2: {score}')

print(f'Mean C-index: {sum(scores) / cv.n_splits}\tFull C-index: {concordance_index(np.array(all_efs_time), -np.array(all_preds), np.array(all_efs))}')


cat_cols = train.drop(columns=target_cols).select_dtypes(include=object).columns.values.tolist()

train[cat_cols] = train[cat_cols].astype('category')
test[cat_cols] = test[cat_cols].astype('category')

train_lgb_naf = lgb.Dataset(train.drop(columns=target_cols), label=train['naf_label'], categorical_feature=cat_cols)
train_lgb_km = lgb.Dataset(train.drop(columns=target_cols), label=train['km_label'], categorical_feature=cat_cols)

# best_naf = lgbm_naf_params['metric'] = 'concordance_index'
best_naf = lgb.train(lgbm_naf_params, train_lgb_naf, 1000, valid_sets=[train_lgb_naf])
# best_km = lgbm_km_params['metric'] = 'concordance_index'
best_km = lgb.train(lgbm_km_params, train_lgb_km, 1000, valid_sets=[train_lgb_km])

xgb_naf = xgb.XGBRegressor(**xgb_naf_params,
                           # eval_metric=cindex_xgb, 
                           eval_set=[(val_data.drop(columns=target_cols), val_data['naf_label'])], verbose=False)
xgb_km = xgb.XGBRegressor(**xgb_km_params,
                          # eval_metric=cindex_xgb
                         eval_set=[(val_data.drop(columns=target_cols), val_data['km_label'])], verbose=False)

xgb_naf.fit(train.drop(columns=target_cols), train['naf_label'])
xgb_km.fit(train.drop(columns=target_cols), train['km_label'])


preds_lgb_naf = best_naf.predict(test)
preds_lgb_km = best_km.predict(test)
preds_xgb_naf = xgb_naf.predict(test)
preds_xgb_km = xgb_km.predict(test)


final_preds_test = (preds_lgb_naf + preds_lgb_km + preds_xgb_naf + preds_xgb_km)/4


sub['prediction'] = final_preds_test
sub.to_csv('submission.csv')

sub






















