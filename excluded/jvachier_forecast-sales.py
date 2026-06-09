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


import warnings
import xgboost as xgb
import lightgbm as lgb

from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.model_selection import GroupKFold


warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)


def to_category(df: pd.DataFrame)->pd.DataFrame:
    categoric_c = df.select_dtypes(include=['object']).columns.tolist()
    df[categoric_c] = df[categoric_c].astype('category')
    return df

def expand_time(df: pd.DataFrame) -> pd.DataFrame:
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df["is Weekend"] = (df['date'].dt.dayofweek > 4).astype(int)
    return df

def evaluate_model(model, X_train, y_train, X_test, y_test) -> None:
    train_pred = model.predict(X_train)
    print('----------------------------------------------\n')
    print('Train MAPE: ', mean_absolute_percentage_error(y_true=y_train, y_pred=train_pred))
    print('----------------------------------------------\n')
    test_pred = model.predict(X_test)
    print('Test MAPE: ', mean_absolute_percentage_error(y_true=y_test, y_pred=test_pred))


train = pd.read_csv('/kaggle/input/playground-series-s5e1/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e1/test.csv')
submission = pd.read_csv('/kaggle/input/playground-series-s5e1/sample_submission.csv')


train['date'] = pd.to_datetime(train['date'])
test['date'] = pd.to_datetime(test['date'])


train = expand_time(train)
test = expand_time(test)


train.drop(columns=['id','date'], inplace=True)
test.drop(columns=['id','date'], inplace=True)


train = to_category(train)
test = to_category(test)


numeric_c_int = train.select_dtypes(include=['int64']).columns.tolist()
cat_columns = test.select_dtypes(include=['category']).columns.tolist()


train.dropna(inplace=True)
train[numeric_c_int] = train[numeric_c_int].astype('int32')


X = train.drop('num_sold', axis=1)
y = train['num_sold']

X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, y, test_size=0.2, random_state=42)


param_cat = {
    'learning_rate': 0.258441682206043,
    'depth': 9,
    'random_strength': 3.5323176999235493,
    'subsample': 0.8955520227225029,
    'colsample_bylevel': 0.7625974626682459,
    'bagging_temperature': 0.15342763612551344,
    'border_count': 13,
    'l2_leaf_reg': 7,
    'min_data_in_leaf': 5,
    'loss_function': 'MAPE',
    'iterations': 1000,
    'silent':True,
    'random_state':42,
    'thread_count':3,
} 

param_xgb = {
    "eval_metric": "mape",
    'objective': 'reg:squarederror',
    'tree_method': 'hist',
    "iterations": 1000,
    'learning_rate': 0.07740093446686303,
    'colsample_bytree': 0.9674916240049458,
    'max_depth': 12,
    'min_child_weight': 21,
    'subsample': 0.9395385388586801,
    'reg_lambda': 2.8019853417187974,
    "verbosity": 0,
    'random_state':42,
    'nthread':3,
}

param_lgb = {
    'reg_alpha': 0.0031093033284386186,
    'reg_lambda': 0.08311722872685912,
    'colsample_bytree': 0.9833172441176441,
    'subsample': 0.8398200271186018,
    'learning_rate': 0.040753313370108936,
    'max_depth': 8,
    'num_leaves': 84,
    'min_child_samples': 41,
    'cat_smooth': 66,
    'objective': 'regression',
    'metric': 'mape',
    'n_estimators': 1000,
    'force_col_wise':True,
    'random_state':42,
    'num_threads':3,
    'verbose':-1
}


model_XGBoost = xgb.XGBRegressor(
    **param_xgb, 
    enable_categorical=True, 
)
model_LGMB = lgb.LGBMRegressor(
    **param_lgb, 
)
model_Cat = CatBoostRegressor(
    **param_cat,
    cat_features=cat_columns, 
)


param_elasticnet= {
    'l1_ratio': 0.11671459624530144,
    'random_state': 42, 
    'max_iter': 1000,
    'selection': 'random',
}


estimators = [
    ('lgb', model_LGMB),
    ('cat', model_Cat),
    ('xgb', model_XGBoost),
]


stack = StackingRegressor(
    estimators=estimators, 
    final_estimator=ElasticNet(**param_elasticnet),
    n_jobs=5,
)


group_col=X_Train['year']
groupkfold = GroupKFold(
    n_splits=5, 
).split(X_Train, Y_Train,groups=group_col)
for i, (train_index, test_index) in enumerate(groupkfold):
    print(i)
    stack.fit(X_Train.iloc[train_index], Y_Train.iloc[train_index])
    evaluate_model(
        stack, 
        X_Train.iloc[train_index], 
        Y_Train.iloc[train_index], 
        X_Train.iloc[test_index], 
        Y_Train.iloc[test_index]
    )


pred_stack = stack.predict(test)
Sub = pd.DataFrame({
    'id': submission.id,
    'num_sold': pred_stack
})
Sub


Sub.to_parquet(
    './submission_xgboost_catboost_stackingregressor_elasticnet2.parquet', 
    engine='pyarrow', 
    index=False
)

