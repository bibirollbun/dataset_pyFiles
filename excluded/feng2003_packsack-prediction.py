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


from sklearn.experimental import enable_iterative_imputer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.compose import ColumnTransformer
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import log_evaluation, early_stopping, LGBMRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
original_train_df = train_df.copy()
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
sub = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')
train_extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e2/sample_submission.csv')


train_extra_df.head(30)


train_columns = ['id', 'brand', 'material', 'size', 'compartments', 'laptop_compartment','waterproof', 'style', 'color', 'weight_capacity', 'price']
test_columns = ['id', 'brand', 'material', 'size', 'compartments', 'laptop_compartment','waterproof', 'style', 'color', 'weight_capacity']
train_df.columns = train_columns
test_df.columns = test_columns
train_extra_df.columns = train_columns
train_df.columns
train_extra_df.columns


extra_nan_count_per_row = train_extra_df.isna().sum(axis = 1)
rows_with_more_than_two_nans = train_extra_df[extra_nan_count_per_row >= 2]
train_extra_df = train_extra_df[extra_nan_count_per_row < 2]

train_nan_count_per_row = train_df.isna().sum(axis = 1)
train_df = train_df[train_nan_count_per_row<2]



# fillna_columns = [ 'brand', 'material', 'size', 'laptop_compartment','waterproof', 'style', 'color','weight_capacity']
# for column in fillna_columns:
#     if train_df[column].dtype == 'object':
#         train_df[column] = train_df[column].fillna(train_df[column].mode()[0])
#         test_df[column] = test_df[column].fillna(test_df[column].mode()[0])
#         train_extra_df[column] = train_extra_df[column].fillna(train_extra_df[column].mode()[0])
#     else:
#         train_df[column] = train_df[column].fillna(train_df[column].median())
#         test_df[column] = test_df[column].fillna(test_df[column].median())
#         train_extra_df[column] = train_extra_df[column].fillna(train_extra_df[column].median())


# transform_columns = ['size', 'laptop_compartment', 'waterproof']
# le = LabelEncoder()
# for column in transform_columns:
#     train_df[column] = le.fit_transform(train_df[column])
#     test_df[column] = le.fit_transform(test_df[column])
#     train_extra_df[column] = le.fit_transform(train_extra_df[column])
# train_df  = pd.get_dummies(train_df, columns = ['style'], drop_first = False)
# test_df = pd.get_dummies(test_df, columns = ['style'], drop_first = False)
# train_extra_df = pd.get_dummies(train_extra_df, columns = ['style'], drop_first = False)


def TargetEncoding(df):
    for column in df.select_dtypes(include = ['object']):
        groups = df.groupby(column).groups
        groups_dict = {}
        for index,single in enumerate(groups):
            groups_dict[single] = len(groups[single].tolist()) / len(df)
        df[f'TE_{column}'] = df[column].map(groups_dict)
    return df
    


train_df = TargetEncoding(train_df)
test_df = TargetEncoding(test_df)
train_extra_df = TargetEncoding(train_extra_df)
train_df['weight_capacity_square'] = train_df['weight_capacity'] ** 2
train_df['compartments_twice'] = train_df['compartments'] * 2
train_extra_df['weight_capacity_square'] = train_df['weight_capacity'] ** 2
train_extra_df['compartments_twice'] = train_extra_df['compartments'] * 2
test_df['weight_capacity_square'] = test_df['weight_capacity'] ** 2
test_df['compartments_twice'] = test_df['compartments'] * 2
train_extra_df = TargetEncoding(train_extra_df)

test_df.head(10)


for column in train_df.select_dtypes(include = ['object']):
    train_df[column] = train_df[column].astype('category')
    test_df[column] = test_df[column].astype('category')
    train_extra_df[column] = train_extra_df[column].astype('category')
    print(train_df[column].unique())

print('\n\ntrain_null_sum : ')
print(train_df.isna().sum())
print('\n\ntest_null_sum : ')
print(test_df.isna().sum())
print('\n\ntrain_extra_null_sum : ')
print(train_extra_df.isna().sum())
print('\n\ntrain_sum : ')
print(len(train_df['id']))
print('\n\ntrain_extra_sum : ')
print(len(train_extra_df['id']))


# Y = train_df['price']
# X = train_df.drop(columns = ['price','id'])
test_df = test_df.drop(columns = ['id'])
Y = train_extra_df['price']
X = train_extra_df.drop(columns = ['price', 'id'])


print(train_df.weight_capacity.min())
print(train_df.weight_capacity.max())
print(train_df.compartments.min())
print(train_df.compartments.max())



train_df.head(10)



# # 计算皮尔逊相关系数
# correlation = train_df[[ 'size', 'laptop_compartment','waterproof','weight_capacity', 'price']].corr()

# print(correlation)



# spearman_corr = train_df[['compartments', 'weight_capacity', 'price']].corr(method='spearman')
# kendall_corr = train_df[['compartments', 'weight_capacity', 'price']].corr(method='kendall')

# print("Spearman correlation:\n", spearman_corr)
# print("Kendall correlation:\n", kendall_corr)



xgb_params = {
    'device':'cuda',
    'max_depth':7,
    'colsample_bytree':0.8831890304605191,
    'subsample':0.699208776105851,
    'n_estimators':2500,
    'learning_rate':0.015616900063639172,
    'min_child_weight':51,
    'enable_categorical':True,
    'random_state' : 42,
    'early_stopping_rounds':True,
    
}
# 'max_depth': 3, 'colsample_bytree': 0.8070768632011154, 'subsample': 0.8475678788257452, 'learning_rate': 0.019565337795049015, 'min_child_weight': 54
# {'max_depth': 6, 'colsample_bytree': 0.9724991142531788, 'subsample': 0.7041468982329704, 'learning_rate': 0.025130121799742722, 'min_child_weight': 71}
# {'max_depth': 7, 'colsample_bytree': 0.8831890304605191, 'subsample': 0.699208776105851, 'learning_rate': 0.015616900063639172, 'min_child_weight': 51}
# {'max_depth': 5, 'colsample_bytree': 0.9879192537461129, 'subsample': 0.813437105015594, 'learning_rate': 0.0230512179506565, 'min_child_weight': 55}
# {'max_depth': 8, 'colsample_bytree': 0.8061890295388814, 'subsample': 0.8550017043283502, 'learning_rate': 0.011372513214763385, 'min_child_weight': 53
# {'max_depth': 5, 'colsample_bytree': 0.5368590376873121, 'subsample': 0.9414624534965425, 'learning_rate': 0.02500388999754255, 'min_child_weight': 76}
lgb_params = {
    'objective':'regression_l1',
    'metric':'rmse',
    'max_depth':7,
    'min_child_weight':66,
    'colsample_bytree': 0.548291893721376,
    'reg_alpha':0.08728855480055722,
    'reg_lambda':0.6899135066600413,
    'random_state':42,
    'early_stopping_round':100,
    'verbose':-1,
    'boosting_type':'gbdt',
    'n_estimators':2500,
    'learning_rate':0.011801241013987857,
    'min_child_samples':25,
    'num_leaves':169
    
}
# {'max_depth': 5, 'min_child_weight': 57, 'colsamplt_bytree': 0.502590313526927, 
# 'reg_alpha': 0.04394011571500413, 'reg_lambda': 0.5947706884040629, 
# 'learning_rate': 0.017300631010969462, 'min_child_samples': 44, 'num_leaves': 64}
#Best hyperparameters: {'max_depth': 5, 'min_child_weight': 64, 'colsamplt_bytree': 0.5668041095616804,
# 'reg_alpha': 0.05179033125263544, 'reg_lambda': 0.7286134337794834, 'learning_rate': 0.01782991579457247,
# 'min_child_samples': 43, 'num_leaves': 119}

# 'max_depth': 6, 'min_child_weight': 54, 'colsamplt_bytree': 0.5535350618554497, 'reg_alpha': 0.08483014848205175, 'reg_lambda': 0.6362540220255315, 
# 'learning_rate': 0.03501735139969201, 'min_child_samples': 50, 'num_leaves': 135

# {'max_depth': 7, 'min_child_weight': 66, 'colsamplt_bytree': 0.548291893721376, 'reg_alpha': 0.08728855480055722, 'reg_lambda': 0.6899135066600413, 
# 'learning_rate': 0.011801241013987857, 'min_child_samples': 25, 'num_leaves': 169}


import optuna
def objective(trial):
    try_xgb_params = {
        'device':'cuda',
        'max_depth':trial.suggest_int('max_depth',3,10),
        'colsample_bytree':trial.suggest_float('colsample_bytree', 0.5,1.0),
        'subsample':trial.suggest_float('subsample',0.5,1.0),
        'n_estimators':2500,
        'learning_rate':trial.suggest_float('learning_rate',0.01,0.03),
        'min_child_weight':trial.suggest_int('min_child_weight',50,80),
        'enable_categorical':True,
        'random_state':42,
        'early_stopping_rounds':True,
        'verbose':0
    }
    best_rmse = float('inf')
    n_folds = 5
    oof_preds = np.zeros(len(X))
    cv = KFold(n_splits = n_folds, shuffle = False)
    score = 0
    for fold,(train_idx,val_idx) in enumerate(cv.split(X,Y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        Y_train, Y_valid = Y.iloc[train_idx], Y.iloc[val_idx]
        model = XGBRegressor(**try_xgb_params)
        model.fit(
            X_train,Y_train,
            eval_set = [(X_valid, Y_valid)]
        )
        oof_preds[val_idx] = model.predict(X_valid)
        rmse = np.sqrt(mean_squared_error(Y_valid, oof_preds[val_idx]))
        print(f'XGB_Validation_RMSE:{rmse}')
        best_rmse = min(best_rmse, rmse)
    return best_rmse
study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials = 10)
print(study.best_params)


import optuna
def objective(trial):
    try_lgb_params = {
        'objective':'regression_l1',
        'metric':'rmse',
        'max_depth':trial.suggest_int('max_depth', 5,7),
        'min_child_weight':trial.suggest_int('min_child_weight',50, 70),
        'colsample_bytree':trial.suggest_float('colsamplt_bytree', 0.5,0.7),
        'reg_alpha':trial.suggest_float('reg_alpha',0.03,0.1),
        'reg_lambda':trial.suggest_float('reg_lambda',0.5,1),
        'random_state':42,
        'boosting_type':'gbdt',
        'n_estimators':2500,
        'learning_rate':trial.suggest_float('learning_rate',0.01,0.05),
        'min_child_samples':trial.suggest_int('min_child_samples',20,50),
        'num_leaves':trial.suggest_int('num_leaves',63,255),
        'early_stopping_round':100,
        'decive':'gpu'
    }
    lgb_best_rmse = float('inf')
    lgb_n_folds = 5
    lgb_oof_preds = np.zeros(len(X))
    lgb_cv = KFold(n_splits = lgb_n_folds, shuffle = False)
    score = 0
    for fold,(train_idx,val_idx) in enumerate(lgb_cv.split(X,Y)):
        X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
        Y_train, Y_valid = Y.iloc[train_idx], Y.iloc[val_idx]
        lgb_model = LGBMRegressor(**try_lgb_params)
        lgb_model.fit(
            X_train,Y_train,
            eval_set = [(X_valid, Y_valid)]
        )
        lgb_oof_preds[val_idx] = lgb_model.predict(X_valid)
        lgb_rmse = np.sqrt(mean_squared_error(Y_valid, lgb_oof_preds[val_idx]))
        print(f'LGB_Validation RMSE : {lgb_rmse}')
        lgb_best_rmse = min(lgb_best_rmse,lgb_rmse)
    return lgb_best_rmse
study = optuna.create_study(direction = 'minimize')
study.optimize(objective, n_trials = 10)
print(f"Best hyperparameters: {study.best_params}")



best_model = None
best_test_preds = 0
best_rmse = float('inf')
n_folds = 5
test_preds = np.zeros(len(test_df))
oof_preds = np.zeros(len(X))
cv = KFold(n_splits = n_folds, shuffle = False)
score = 0
for fold,(train_idx, val_idx) in enumerate(cv.split(X,Y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
    Y_train, Y_valid = Y.iloc[train_idx], Y.iloc[val_idx]
    model = XGBRegressor(**xgb_params)
    model.fit(
        X_train,Y_train,
        eval_set = [(X_valid, Y_valid)],
        verbose = 100
    )
    oof_preds[val_idx] = model.predict(X_valid)
    test_preds += model.predict(test_df) / n_folds

    rmse = np.sqrt(mean_squared_error(Y_valid, oof_preds[val_idx]))
    print(f'Validation RMSE:{rmse}')
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_test_preds = test_preds



lgb_best_model = None
lgb_test_preds = 0
lgb_best_rmse = float('inf')
lgb_best_preds = 0
lgb_n_folds = 5
lgb_test_preds = np.zeros(len(test_df))
lgb_oof_preds = np.zeros(len(X))
lgb_cv = KFold(n_splits = lgb_n_folds, shuffle = False)
score = 0
for fold,(train_idx,val_idx) in enumerate(lgb_cv.split(X,Y)):
    X_train, X_valid = X.iloc[train_idx], X.iloc[val_idx]
    Y_train, Y_valid = Y.iloc[train_idx], Y.iloc[val_idx]
    lgb_model = LGBMRegressor(**lgb_params)
    lgb_model.fit(
        X_train,Y_train,
        eval_set = [(X_valid, Y_valid)]
    )
    lgb_oof_preds[val_idx] = lgb_model.predict(X_valid)
    lgb_test_preds += lgb_model.predict(test_df) / lgb_n_folds
    lgb_rmse = np.sqrt(mean_squared_error(Y_valid, lgb_oof_preds[val_idx]))
    print(f'LGB_Validation RMSE : {lgb_rmse}')
    if lgb_rmse < lgb_best_rmse:
        lgb_best_rmse = lgb_rmse
        lgb_best_model = lgb_model
        lgb_best_preds = lgb_model.predict(test_df)


# test_result = (lgb_test_preds + test_preds) / 2.0
sub = pd.DataFrame({'id':sample_submission_df.id, 'Price':test_preds})
sub.to_csv('submission.csv', index = False)
sub.head()

