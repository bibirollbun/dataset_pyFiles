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

# Target encoding using KaplanMeierFitter and NelsonAalenFitter
naf = NelsonAalenFitter()
naf.fit(train['efs_time'], train['efs'])
train['naf_label'] = -naf.cumulative_hazard_at_times(train['efs_time']).values
train.loc[train['efs'] == 0, 'naf_label'] -= 0.1

kmf = KaplanMeierFitter()
kmf.fit(train['efs_time'], train['efs'])
train['km_label'] = kmf.survival_function_at_times(train['efs_time']).values
train.loc[train['efs'] == 0, 'km_label'] -= 0.1


# XGBoost Parameters
xgb_naf_params = {
    'max_depth': 2,
    'learning_rate': 0.009806810287436414,
    'n_estimators': 9110,
    'reg_lambda': 0.16957442536602274,
    'random_state': 12,
    'objective': 'reg:squarederror',
    'enable_categorical': True
}

xgb_km_params = {
    'max_depth': 2,
    'learning_rate': 0.012887726635046637,
    'n_estimators': 5759,
    'reg_lambda': 0.014550241891247515,
    'random_state': 25,
    'objective': 'reg:squarederror',
    'enable_categorical': True
}

# LightGBM Parameters
lgbm_naf_params = {
    'max_depth': 3,
    'learning_rate': 0.03251780857602963,
    'n_estimators': 1999,
    'reg_lambda': 0.01478462915287414,
    'random_state': 53,
    'verbose': -1
}

lgbm_km_params = {
    'max_depth': 2,
    'learning_rate': 0.020089200208762432,
    'n_estimators': 3757,
    'reg_lambda': 0.004744665699048939,
    'random_state': 0,
    'verbose': -1
}


target_cols = ['efs', 'efs_time', 'km_label', 'naf_label']
all_preds = []
all_efs = []
all_efs_time = []
scores = []
cv = KFold(n_splits=5)

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

    best_naf = lgb.train(lgbm_naf_params, train_lgb_naf, 1000, valid_sets=[train_lgb_naf])
    best_km = lgb.train(lgbm_km_params, train_lgb_km, 1000, valid_sets=[train_lgb_km])

    xgb_naf = xgb.XGBRegressor(**xgb_naf_params)
    xgb_km = xgb.XGBRegressor(**xgb_km_params)

    xgb_naf.fit(train_data.drop(columns=target_cols), train_data['naf_label'])
    xgb_km.fit(train_data.drop(columns=target_cols), train_data['km_label'])

    preds_lgb_naf = best_naf.predict(val_data.drop(columns=target_cols))
    preds_lgb_km = best_km.predict(val_data.drop(columns=target_cols))
    preds_xgb_naf = xgb_naf.predict(val_data.drop(columns=target_cols))
    preds_xgb_km = xgb_km.predict(val_data.drop(columns=target_cols))

    preds = (preds_lgb_naf + preds_lgb_km + preds_xgb_naf + preds_xgb_km) / 4
    all_preds += list(preds)
    score = concordance_index(val_data['efs_time'], -preds, val_data['efs'])
    scores.append(score)

    print(f'Fold #{i} C-index: {score}')

print(f'Mean C-index: {sum(scores) / cv.n_splits}\tFull C-index: {concordance_index(np.array(all_efs_time), -np.array(all_preds), np.array(all_efs))}')


cat_cols = train.drop(columns=target_cols).select_dtypes(include=object).columns.values.tolist()

train[cat_cols] = train[cat_cols].astype('category')
test[cat_cols] = test[cat_cols].astype('category')

train_lgb_naf = lgb.Dataset(train.drop(columns=target_cols), label=train['naf_label'], categorical_feature=cat_cols)
train_lgb_km = lgb.Dataset(train.drop(columns=target_cols), label=train['km_label'], categorical_feature=cat_cols)

best_naf = lgb.train(lgbm_naf_params, train_lgb_naf, 1000, valid_sets=[train_lgb_naf])
best_km = lgb.train(lgbm_km_params, train_lgb_km, 1000, valid_sets=[train_lgb_km])

xgb_naf = xgb.XGBRegressor(**xgb_naf_params)
xgb_km = xgb.XGBRegressor(**xgb_km_params)

xgb_naf.fit(train.drop(columns=target_cols), train['naf_label'])
xgb_km.fit(train.drop(columns=target_cols), train['km_label'])


preds_lgb_naf = best_naf.predict(test)
preds_lgb_km = best_km.predict(test)
preds_xgb_naf = xgb_naf.predict(test)
preds_xgb_km = xgb_km.predict(test)


final_preds_test = (preds_lgb_naf + preds_lgb_km + preds_xgb_naf + preds_xgb_km) / 4


sub['prediction'] = final_preds_test
sub.to_csv('/kaggle/working/submission.csv')

print("Submission file has been saved.")

