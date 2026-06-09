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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e2/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e2/test.csv')
train_extra_df = pd.read_csv('/kaggle/input/playground-series-s5e2/training_extra.csv')


train_df.info()


num_columns = ['Compartments', 'Weight Capacity (kg)']
object_columns = ['Brand', 'Material', 'Size', 'Laptop Compartment', 'Waterproof', 'Style', 'Color']

train_df['Weight Capacity (kg)'].fillna(train_df['Weight Capacity (kg)'].mean(), inplace=True)
train_df['Brand'].fillna(train_df['Brand'].mode()[0], inplace=True)
train_df['Material'].fillna(train_df['Material'].mode()[0], inplace=True)
train_df['Size'].fillna(train_df['Size'].mode()[0], inplace=True)
train_df['Laptop Compartment'].fillna(train_df['Laptop Compartment'].mode()[0], inplace=True)
train_df['Waterproof'].fillna(train_df['Waterproof'].mode()[0], inplace=True)
train_df['Style'].fillna(train_df['Style'].mode()[0], inplace=True)
train_df['Color'].fillna(train_df['Color'].mode()[0], inplace=True)

test_df['Weight Capacity (kg)'].fillna(test_df['Weight Capacity (kg)'].mean(), inplace=True)
test_df['Brand'].fillna(test_df['Brand'].mode()[0], inplace=True)
test_df['Material'].fillna(test_df['Material'].mode()[0], inplace=True)
test_df['Size'].fillna(test_df['Size'].mode()[0], inplace=True)
test_df['Laptop Compartment'].fillna(test_df['Laptop Compartment'].mode()[0], inplace=True)
test_df['Waterproof'].fillna(test_df['Waterproof'].mode()[0], inplace=True)
test_df['Style'].fillna(test_df['Style'].mode()[0], inplace=True)
test_df['Color'].fillna(test_df['Color'].mode()[0], inplace=True)



# 遍历所有object类型的字段，查看这些字段的unique()值
for column in train_df.select_dtypes(include=['object']).columns:
    unique_values = train_df[column].unique()
    print(f"Unique values in '{column}': {unique_values}")


train = pd.get_dummies(train_df, columns=object_columns, drop_first=True, dtype=int)
test  = pd.get_dummies( test_df, columns=object_columns, drop_first=True, dtype=int)


train.shape,test.shape


from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
import numpy as np

# 数据准备
x = train.drop(columns=['Price','id'])
y = train['Price']
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ================== 定义三个模型的超参数优化函数 ==================

# LightGBM优化函数
def objective_lgb(trial):
    params = {
        'objective': 'regression',
        'boosting_type': 'gbdt',
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'num_leaves': trial.suggest_int('num_leaves', 20, 150),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.05),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 50.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 50.0),
        'random_state': 42
    }
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, pred))

# XGBoost优化函数
def objective_xgb(trial):
    params = {
        'objective': 'reg:squarederror',
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'alpha': trial.suggest_float('alpha', 1e-2, 50.0),
        'lambda': trial.suggest_float('lambda', 1e-2, 50.0),
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'random_state': 42
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, pred))

# CatBoost优化函数
def objective_cat(trial):
    params = {
        'loss_function': 'RMSE',
        'depth': trial.suggest_int('depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 50.0),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'verbose': False,
        'random_state': 42
    }
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return np.sqrt(mean_squared_error(y_test, pred))

# ================== 执行超参数优化 ==================
# 优化LightGBM
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=50)
best_lgb = study_lgb.best_params
best_lgb['random_state'] = 42

# 优化XGBoost
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=50)
best_xgb = study_xgb.best_params
best_xgb['random_state'] = 42

# 优化CatBoost
study_cat = optuna.create_study(direction='minimize')
study_cat.optimize(objective_cat, n_trials=50)
best_cat = study_cat.best_params
best_cat['verbose'] = False
best_cat['random_state'] = 42

# ================== 基模型定义 ==================
lgb_model = lgb.LGBMRegressor(**best_lgb)
xgb_model = xgb.XGBRegressor(**best_xgb)
cat_model = CatBoostRegressor(**best_cat)

# ================== 生成堆叠特征 ==================
def generate_stacking_features(model, X_train, y_train, X_test, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    oof_train = np.zeros(X_train.shape[0])
    oof_test = np.zeros(X_test.shape[0])
    
    for train_idx, val_idx in kf.split(X_train):
        X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
        X_val = X_train.iloc[val_idx]
        
        model.fit(X_tr, y_tr)
        oof_train[val_idx] = model.predict(X_val)
        oof_test += model.predict(X_test)
    
    oof_test /= n_splits
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# 生成各模型的OOF特征
lgb_oof, lgb_test = generate_stacking_features(lgb_model, X_train, y_train, X_test)
xgb_oof, xgb_test = generate_stacking_features(xgb_model, X_train, y_train, X_test)
cat_oof, cat_test = generate_stacking_features(cat_model, X_train, y_train, X_test)

# 堆叠特征合并
stacked_X_train = np.concatenate([lgb_oof, xgb_oof, cat_oof], axis=1)
stacked_X_test = np.concatenate([lgb_test, xgb_test, cat_test], axis=1)

# ================== 训练元模型 ==================
meta_model = LinearRegression()
meta_model.fit(stacked_X_train, y_train)

# ================== 评估堆叠模型 ==================
stacked_pred = meta_model.predict(stacked_X_test)
rmse = np.sqrt(mean_squared_error(y_test, stacked_pred))

print("LightGBM最佳参数:", best_lgb)
print("XGBoost最佳参数:", best_xgb)
print("CatBoost最佳参数:", best_cat)
print(f"堆叠模型的RMSE: {rmse:.5f}")


# from sklearn.model_selection import train_test_split, KFold
# from sklearn.metrics import mean_squared_error
# from sklearn.linear_model import LinearRegression
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.svm import SVR
# import lightgbm as lgb
# import xgboost as xgb
# from catboost import CatBoostRegressor
# import optuna
# import numpy as np

# # 数据准备
# x = train.drop(columns=['Price', 'id'])
# y = train['Price']
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # ================== 定义五个模型的超参数优化函数 ==================

# # LightGBM优化函数
# def objective_lgb(trial):
#     params = {
#         'objective': 'regression',
#         'boosting_type': 'gbdt',
#         'max_depth': trial.suggest_int('max_depth', 3, 8),
#         'num_leaves': trial.suggest_int('num_leaves', 20, 150),
#         'min_child_samples': trial.suggest_int('min_child_samples', 20, 200),
#         'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.05),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 50.0),
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 50.0),
#         'random_state': 42
#     }
#     model = lgb.LGBMRegressor(**params)
#     model.fit(X_train, y_train)
#     pred = model.predict(X_test)
#     return np.sqrt(mean_squared_error(y_test, pred))

# # XGBoost优化函数
# def objective_xgb(trial):
#     params = {
#         'objective': 'reg:squarederror',
#         'max_depth': trial.suggest_int('max_depth', 3, 8),
#         'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
#         'alpha': trial.suggest_float('alpha', 1e-2, 50.0),
#         'lambda': trial.suggest_float('lambda', 1e-2, 50.0),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'random_state': 42
#     }
#     model = xgb.XGBRegressor(**params)
#     model.fit(X_train, y_train)
#     pred = model.predict(X_test)
#     return np.sqrt(mean_squared_error(y_test, pred))

# # CatBoost优化函数
# def objective_cat(trial):
#     params = {
#         'loss_function': 'RMSE',
#         'depth': trial.suggest_int('depth', 3, 8),
#         'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1),
#         'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-2, 50.0),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'iterations': trial.suggest_int('iterations', 100, 1000),
#         'verbose': False,
#         'random_state': 42
#     }
#     model = CatBoostRegressor(**params)
#     model.fit(X_train, y_train)
#     pred = model.predict(X_test)
#     return np.sqrt(mean_squared_error(y_test, pred))

# # RandomForest优化函数
# def objective_rf(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
#         'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
#         'random_state': 42
#     }
#     model = RandomForestRegressor(**params)
#     model.fit(X_train, y_train)
#     pred = model.predict(X_test)
#     return np.sqrt(mean_squared_error(y_test, pred))

# # GradientBoosting优化函数
# def objective_gb(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 1e-4, 0.1),
#         'subsample': trial.suggest_float('subsample', 0.6, 1.0),
#         'random_state': 42
#     }
#     model = GradientBoostingRegressor(**params)
#     model.fit(X_train, y_train)
#     pred = model.predict(X_test)
#     return np.sqrt(mean_squared_error(y_test, pred))

# # SVR优化函数
# def objective_svr(trial):
#     params = {
#         'C': trial.suggest_float('C', 0.1, 10.0),
#         'epsilon': trial.suggest_float('epsilon', 0.01, 0.1),
#         'kernel': trial.suggest_categorical('kernel', ['linear', 'rbf']),
#         'random_state': 42
#     }
#     model = SVR(**params)
#     model.fit(X_train, y_train)
#     pred = model.predict(X_test)
#     return np.sqrt(mean_squared_error(y_test, pred))

# # ================== 执行超参数优化 ==================
# # 优化LightGBM
# study_lgb = optuna.create_study(direction='minimize')
# study_lgb.optimize(objective_lgb, n_trials=50)
# best_lgb = study_lgb.best_params
# best_lgb['random_state'] = 42

# # 优化XGBoost
# study_xgb = optuna.create_study(direction='minimize')
# study_xgb.optimize(objective_xgb, n_trials=50)
# best_xgb = study_xgb.best_params
# best_xgb['random_state'] = 42

# # 优化CatBoost
# study_cat = optuna.create_study(direction='minimize')
# study_cat.optimize(objective_cat, n_trials=50)
# best_cat = study_cat.best_params
# best_cat['verbose'] = False
# best_cat['random_state'] = 42

# # 优化RandomForest
# study_rf = optuna.create_study(direction='minimize')
# study_rf.optimize(objective_rf, n_trials=10)
# best_rf = study_rf.best_params
# best_rf['random_state'] = 42

# # 优化GradientBoosting
# study_gb = optuna.create_study(direction='minimize')
# study_gb.optimize(objective_gb, n_trials=50)
# best_gb = study_gb.best_params
# best_gb['random_state'] = 42

# # 优化SVR
# study_svr = optuna.create_study(direction='minimize')
# study_svr.optimize(objective_svr, n_trials=20)
# best_svr = study_svr.best_params
# best_svr['random_state'] = 42

# # ================== 基模型定义 ==================
# lgb_model = lgb.LGBMRegressor(**best_lgb)
# xgb_model = xgb.XGBRegressor(**best_xgb)
# cat_model = CatBoostRegressor(**best_cat)
# rf_model = RandomForestRegressor(**best_rf)
# gb_model = GradientBoostingRegressor(**best_gb)
# svr_model = SVR(**best_svr)

# # ================== 生成堆叠特征 ==================
# def generate_stacking_features(model, X_train, y_train, X_test, n_splits=5):
#     kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
#     oof_train = np.zeros(X_train.shape[0])
#     oof_test = np.zeros(X_test.shape[0])
    
#     for train_idx, val_idx in kf.split(X_train):
#         X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
#         X_val = X_train.iloc[val_idx]
        
#         model.fit(X_tr, y_tr)
#         oof_train[val_idx] = model.predict(X_val)
#         oof_test += model.predict(X_test)
    
#     oof_test /= n_splits
#     return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)

# # 生成各模型的OOF特征
# lgb_oof, lgb_test = generate_stacking_features(lgb_model, X_train, y_train, X_test)
# xgb_oof, xgb_test = generate_stacking_features(xgb_model, X_train, y_train, X_test)
# cat_oof, cat_test = generate_stacking_features(cat_model, X_train, y_train, X_test)
# rf_oof, rf_test = generate_stacking_features(rf_model, X_train, y_train, X_test)
# svr_oof, svr_test = generate_stacking_features(svr_model, X_train, y_train, X_test)

# # 堆叠特征合并
# stacked_X_train = np.concatenate([lgb_oof, xgb_oof, cat_oof,rf_oof,svr_oof], axis=1)
# stacked_X_test = np.concatenate([lgb_test, xgb_test, cat_test,rf_test,svr_test], axis=1)



# # ================== 训练元模型 ==================
# meta_model = LinearRegression()
# meta_model.fit(stacked_X_train, y_train)

# # ================== 评估堆叠模型 ==================
# stacked_pred = meta_model.predict(stacked_X_test)
# rmse = np.sqrt(mean_squared_error(y_test, stacked_pred))

# print("LightGBM最佳参数:", best_lgb)
# print("XGBoost最佳参数:", best_xgb)
# print("CatBoost最佳参数:", best_cat)
# print("RandomForest最佳参数:", best_rf)
# print("svm最佳参数:", best_svr)

# print(f"堆叠模型的RMSE: {rmse:.5f}")


test2 = test.drop(columns=['id'])


# 使用堆叠模型进行预测
# 生成test2数据的OOF特征（需要对test2数据使用训练好的模型进行预测）
lgb_oof_test2, lgb_test2 = generate_stacking_features(lgb_model, X_train, y_train, test2)
xgb_oof_test2, xgb_test2 = generate_stacking_features(xgb_model, X_train, y_train, test2)
cat_oof_test2, cat_test2 = generate_stacking_features(cat_model, X_train, y_train, test2)


# 堆叠特征合并
stacked_X_test2 = np.concatenate([lgb_test2, xgb_test2, cat_test2], axis=1)

# 使用训练好的元模型进行预测
predictions = meta_model.predict(stacked_X_test2)

# 获取test2的id列
ids = test['id'].copy()

# 创建一个 DataFrame，将预测结果和 id 组合在一起
result = pd.DataFrame({
    'id': ids,
    'Price': predictions
})

# 将结果保存到CSV文件中
result.to_csv('prediction_results.csv', index=False)

# 返回结果
result


