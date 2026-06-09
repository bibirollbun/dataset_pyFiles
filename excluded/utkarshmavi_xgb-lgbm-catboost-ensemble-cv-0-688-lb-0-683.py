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
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold, GridSearchCV
from lifelines.utils import concordance_index
import lightgbm as lgb
from catboost import CatBoostRegressor

import warnings
warnings.filterwarnings('ignore')


df_train = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/train.csv')
df_test = pd.read_csv('/kaggle/input/equity-post-HCT-survival-predictions/test.csv')


df_train.head()


df_train.shape


df_test.shape


cat_cols = []
num_cols = []

for col in df_train.columns:
    if df_train[col].dtypes == 'object' and col not in ('efs','efs_time'):
        cat_cols.append(col)
    elif df_train[col].dtypes != 'object' and col not in ('efs','efs_time','ID'):
        num_cols.append(col)


print("Number of categorical columns: ", len(cat_cols))
print("Number of numerical columns: ", len(num_cols))


for col in df_train.select_dtypes(include='object').columns:
    df_train[col] = df_train[col].astype('category')

for col in df_test.select_dtypes(include='object').columns:
    df_test[col] = df_test[col].astype('category')


# Making Target variable using KaplanMeierFitter (kmf) and NelsonAalenFitter (naf)

kmf = KaplanMeierFitter()
kmf.fit(df_train['efs_time'], df_train['efs'])

df_train['y_kmf'] = kmf.survival_function_at_times(df_train['efs_time']).values
df_train.loc[df_train['efs']==0, 'y_kmf'] -= 0.1


# NelsonAalenFitter

naf = NelsonAalenFitter()
naf.fit(df_train['efs_time'], df_train['efs'])

df_train['y_naf'] = -naf.cumulative_hazard_at_times(df_train['efs_time']).values
df_train.loc[df_train['efs'] == 0, 'y_naf'] -= 0.1





kf = KFold(n_splits = 5, shuffle = True, random_state = 42)


xgb_kmf_final_param = {
    'booster': 'gbtree', 
    'enable_categorical': True, 
    'learning_rate': 0.02, 
    'max_depth': 2, 
    'n_estimators': 5000, 
    'objective': 'reg:squarederror',
    'random_state': 42, 
    'reg_lambda': 0.015
}

xgb_naf_final_param = {
    'booster': 'gbtree', 
    'enable_categorical': True, 
    'learning_rate': 0.01, 
    'max_depth': 2, 
    'n_estimators': 10000, 
    'objective': 'reg:squarederror',
    'random_state': 42, 
    'reg_lambda': 0.1
}

lgbm_final_param_kmf = {

    'max_depth': 2 ,
    'learning_rate': 0.03 ,
    'n_estimators': 4000 ,
    'reg_lambda': 0.001 ,
    'random_state': 42,
    'verbose': -1
    
}

lgbm_final_param_naf = {

    'max_depth': 3,
    'learning_rate': 0.03,
    'n_estimators': 2000,
    'reg_lambda': 0.02 ,
    'random_state': 42,
    'verbose': -1
    
}

cb_param_kmf = {

    'bootstrap_type' : 'Bernoulli',
    'learning_rate': 0.03,
    'num_trees': 8000,
    'subsample': 0.85,
    'reg_lambda': 8.0,
    'depth': 8,
    'verbose' : 2000
    
}



all_preds = []
all_efs = []
all_efs_time = []
scores = []

for i, (train_indexes, val_indexes) in enumerate(kf.split(df_train)):
    train_data = df_train.iloc[train_indexes]
    val_data = df_train.iloc[val_indexes]
    train_data_cb = df_train.iloc[train_indexes]
    val_data_cb = df_train.iloc[val_indexes]

    for col in train_data_cb.select_dtypes(include='category').columns:
        train_data_cb[col] = train_data_cb[col].astype('str')

    for col in val_data_cb.select_dtypes(include='category').columns:
        val_data_cb[col] = val_data_cb[col].astype('str')

    cat_cols = train_data.drop(columns=['ID','efs','efs_time','y_kmf', 'y_naf']).select_dtypes(include='category').columns.values.tolist()

    cat_cols_cb = train_data_cb.drop(columns=['ID','efs','efs_time','y_kmf', 'y_naf']).select_dtypes(include='object').columns.values.tolist()
    
    xgb_kmf = XGBRegressor(**xgb_kmf_final_param)
    xgb_kmf.fit(train_data.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']), train_data['y_kmf'])

    xgb_naf = XGBRegressor(**xgb_naf_final_param)
    xgb_naf.fit(train_data.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']), train_data['y_naf'])

    lgbm_kmf = lgb.LGBMRegressor(**lgbm_final_param_kmf)
    lgbm_kmf.fit(train_data.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']), train_data['y_kmf'])

    lgbm_naf = lgb.LGBMRegressor(**lgbm_final_param_naf)
    lgbm_naf.fit(train_data.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']), train_data['y_naf'])

    cb_kmf = CatBoostRegressor(**cb_param_kmf)
    cb_kmf.fit(train_data_cb.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']), train_data['y_kmf'], cat_features=cat_cols_cb)
    
    
    preds_xgb_kmf = xgb_kmf.predict(val_data.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']))    
    preds_xgb_naf = xgb_naf.predict(val_data.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']))    
    preds_lgbm_kmf = lgbm_kmf.predict(val_data.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']))    
    preds_lgbm_naf = lgbm_naf.predict(val_data.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']))    
    preds_cb_kmf = cb_kmf.predict(val_data_cb.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']))    
    
    preds =  (preds_xgb_kmf + preds_xgb_naf + preds_lgbm_kmf + preds_lgbm_naf + preds_cb_kmf) / 5

    all_preds += list(preds)
    
    score = concordance_index(val_data['efs_time'], -preds, val_data['efs'])
    
    scores.append(score)

    print(f'Fold #{i} C-index: {score}')

print("Mean Concordance index (using XGB & LGBM): ", sum(scores) / len(scores))


df_train.shape


xgb_kmf_final_param = {
    'booster': 'gbtree', 
    'enable_categorical': True, 
    'learning_rate': 0.02, 
    'max_depth': 2, 
    'n_estimators': 5000, 
    'objective': 'reg:squarederror',
    'random_state': 42, 
    'reg_lambda': 0.015
}

xgb_naf_final_param = {
    'booster': 'gbtree', 
    'enable_categorical': True, 
    'learning_rate': 0.01, 
    'max_depth': 2, 
    'n_estimators': 10000, 
    'objective': 'reg:squarederror',
    'random_state': 42, 
    'reg_lambda': 0.1
}

lgbm_final_param_kmf = {

    'max_depth': 2 ,
    'learning_rate': 0.03 ,
    'n_estimators': 4000 ,
    'reg_lambda': 0.001 ,
    'random_state': 42,
    'verbose': -1
    
}

lgbm_final_param_naf = {

    'max_depth': 3,
    'learning_rate': 0.03,
    'n_estimators': 2000,
    'reg_lambda': 0.02 ,
    'random_state': 42,
    'verbose': -1
    
}

cb_param_kmf = {

    'bootstrap_type' : 'Bernoulli',
    'learning_rate': 0.03,
    'num_trees': 8000,
    'subsample': 0.85,
    'reg_lambda': 8.0,
    'depth': 8,
    'verbose' : 2000
    
}



xgb_kmf = XGBRegressor(**xgb_kmf_final_param)
xgb_kmf.fit(df_train.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']), df_train['y_kmf'])

xgb_naf = XGBRegressor(**xgb_naf_final_param)
xgb_naf.fit(df_train.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']), df_train['y_naf'])

lgbm_kmf = lgb.LGBMRegressor(**lgbm_final_param_kmf)
lgbm_kmf.fit(df_train.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']), df_train['y_kmf'])

lgbm_naf = lgb.LGBMRegressor(**lgbm_final_param_naf)
lgbm_naf.fit(df_train.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']), df_train['y_naf'])



X_train_cb = df_train.copy()

for col in X_train_cb.select_dtypes(include='category').columns:
    X_train_cb[col] = X_train_cb[col].astype('str')

X_test_cb = df_test.copy()
for col in X_test_cb.select_dtypes(include='category').columns:
    X_test_cb[col] = X_test_cb[col].astype('str')


cat_cols_cb = X_train_cb.drop(columns=['ID','efs','efs_time','y_kmf', 'y_naf']).select_dtypes(include='object').columns.values.tolist()

cb_kmf = CatBoostRegressor(**cb_param_kmf)
cb_kmf.fit(X_train_cb.drop(columns = ['ID', 'efs', 'efs_time', 'y_kmf','y_naf']), X_train_cb['y_kmf'], cat_features=cat_cols_cb)


preds_xgb_kmf = xgb_kmf.predict(df_test.drop(columns = ['ID']))    
preds_xgb_naf = xgb_naf.predict(df_test.drop(columns = ['ID']))    
preds_lgbm_kmf = lgbm_kmf.predict(df_test.drop(columns = ['ID']))    
preds_lgbm_naf = lgbm_naf.predict(df_test.drop(columns = ['ID']))    
preds_cb_kmf = cb_kmf.predict(X_test_cb.drop(columns = ['ID']))    


final_prediction = (preds_xgb_kmf + preds_xgb_naf + preds_lgbm_kmf + preds_lgbm_naf + preds_cb_kmf) / 5
# final_prediction_2 = (preds_xgb_kmf + preds_xgb_naf + preds_lgbm_kmf + preds_lgbm_naf) / 4


df_test['prediction'] = final_prediction
# df_test['prediction_2'] = final_prediction_2



sub = df_test[['ID','prediction']]


sub.to_csv('submission.csv')




