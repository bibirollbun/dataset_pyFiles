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


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from scipy.optimize import minimize


train = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
print(f'训练集维度 / train shape:{train.shape}\n')

print('训练集概况 / train info:')
print(train.info())

print('训练集特征统计 / train statistics:')
display(train.describe())

print('\n训练集前五行数据展示 / preview of first 5 training rows:')
display(train.head())


test = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
print(f'测试集维度 / test shape:{test.shape}\n')

print('测试集概况 / test info:')
print(test.info())

print('测试集特征统计 / test statistics:')
display(test.describe())

print('\n测试集前五行数据展示 / preview of first 5 test rows:')
display(test.head())


le = LabelEncoder()
train['Sex'] = le.fit_transform(train['Sex'])
test['Sex'] = le.transform(test['Sex'])


# 绘制各个特征与Calories的散点图
# Scatter plot between each feature and Calories
old_features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'Calories']

# 去掉Calories，避免与它自己画图 
# Exclude Calories to aviod plotting it against itself
for feature in old_features[:-1]:   
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=train[feature], y=train['Calories'], alpha=0.5)
    sns.regplot(x=train[feature], y=train['Calories'], scatter=False, color='red')
    plt.title(f'{feature} vs. Calories')
    plt.xlabel(feature)
    plt.ylabel('Calories')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 绘制特征相关热力图
# Plot feature correlation heatmsp
plt.figure(figsize=(7, 6))
sns.heatmap(train[old_features[1:]].corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm')
plt.title('feature heatmap')
plt.show()


# 绘制Sex和Calories的箱型图
# Box plot of Sex vs. Calories
plt.figure(figsize=(7, 6))
sns.boxplot(x='Sex', y='Calories', data=train)
plt.title('Sex vs. Calories')
plt.grid(True)
plt.tight_layout()
plt.show()


# 绘制Calories的KDE图
# KDE plot for Calories
plt.figure(figsize=(6, 5))
sns.kdeplot(train['Calories'], shade=True, color='purple')
plt.title('Calories KDE')
plt.xlabel('Calories')
plt.ylabel('density')
plt.grid(True)
plt.tight_layout()
plt.show()


features = ['Sex', 'Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp']
           
X = train[features]
y = np.log1p(train['Calories'])   # 标签对数转换 / Log transform the target variable
X_test = test[features]

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# xgb = XGBRegressor(random_state=42)

# param_grid = {
#     'n_estimators': [500, 1000],
#     'learning_rate': [0.01, 0.05],
#     'max_depth': [3, 5],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }

# grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid,
#                            cv=3,scoring='neg_root_mean_squared_error', verbose=2)

# grid_search.fit(X_train, y_train)

# print('XGB最优参数 / XGB best parameters:', grid_search.best_params_)
# print('XGB最佳RMSE / XGB best RMSE :', -grid_search.best_score_)

# # 用最优参数训练XGB模型
# # Train XGB with best parameters
# best_params = grid_search.best_params_
xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42)
xgb.fit(X_train, y_train)


# lgb = LGBMRegressor(random_state=42, verbose=-1)
# param_grid_lgb = {
#     'n_estimators': [100, 500],
#     'learning_rate': [0.01, 0.05],
#     'max_depth': [3, 5, 7],
#     'subsample': [0.8, 1.0],
#     'colsample_bytree': [0.8, 1.0]
# }

# grid_lgb = GridSearchCV(estimator=lgb, param_grid=param_grid_lgb,
#                        cv=3, scoring='neg_root_mean_squared_error', verbose=2)

# grid_lgb.fit(X_train, y_train)

# print('LGB最优参数 / LGB best parameters:', grid_lgb.best_params_)
# print('LGB最佳RMSE / LGB best RMSE:', -grid_lgb.best_score_)

# # 用最优参数训练LGB模型
# # train LGB with best parameters
# best_params_lgb = grid_lgb.best_params_
lgb = LGBMRegressor(n_estimators=500, learning_rate=0.05, max_depth=7, subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
lgb.fit(X_train, y_train)


# cat = CatBoostRegressor(random_seed=42, verbose=0)

# param_grid_cat = {
#     'iterations': [500, 1000],
#     'learning_rate': [0.01, 0.05],
#     'depth': [4, 5, 6]
# }

# grid_cat = GridSearchCV(cat, param_grid_cat, cv=3, scoring='neg_root_mean_squared_error', verbose=2)
# grid_cat.fit(X_train, y_train)

# print('Cat最优参数 / Cat best parameters:', grid_cat.best_params_)
# print('Cat最佳RMSE / Cat best RMSE:', -grid_cat.best_score_)

# # 用最优参数重新训练模型
# # train Cat with best parameters
# best_params_cat = grid_cat.best_params_
cat = CatBoostRegressor(iterations=1000, learning_rate=0.05, depth=6, random_seed=42, verbose=0)
cat.fit(X_train, y_train)


# 各模型预测
# Model predictions
pred_xgb = xgb.predict(X_valid)
pred_lgb = lgb.predict(X_valid)
pred_cat = cat.predict(X_valid)

# 各模型RMSE
# Model RMSE
rmse_xgb = np.sqrt(mean_squared_error(y_valid, pred_xgb))
rmse_lgb = np.sqrt(mean_squared_error(y_valid, pred_lgb))
rmse_cat = np.sqrt(mean_squared_error(y_valid, pred_cat))

print(f'XGB RMSE:{rmse_xgb:.6f}')
print(f'LGB RMSE:{rmse_lgb:.6f}')
print(f'Cat RMSE:{rmse_cat:.6f}')


# best_rmse = float('inf')
# best_weights = (0, 0, 0)

# step = 0.05
# max_xgb = 0.2

# for w_xgb in np.arange(0, 1 + step, step):
#     for w_lgb in np.arange(0, 1 - w_xgb + step, step):
#         w_cat = 1 - w_xgb - w_lgb
#         if w_cat < 0:
#             continue
            
#         ensemble_pred = w_xgb * pred_xgb + w_lgb * pred_lgb + w_cat * pred_cat
#         rmse = np.sqrt(mean_squared_error(y_valid, ensemble_pred))
        
#         if rmse < best_rmse:
#             best_rmse = rmse
#             best_weights = (w_xgb, w_lgb, w_cat)

# print(f'最优权重 / Best weight :XGB={best_weights[0]:.2f}, LGB={best_weights[1]:.2f}, Cat={best_weights[2]:.2f}')
# print(f'对应的RMSE / Corresponding RMSE:{best_rmse:.6f}')


ensemble_pred = 0.20 * pred_xgb + 0.15 * pred_lgb + 0.65 * pred_cat
mse = mean_squared_error(y_valid, ensemble_pred)
rmse = np.sqrt(mse)
print(f'融合模型均方误差 / Ensemble MSE: {mse:.6f}')
print(f'融合模型均方根误差 / Ensemble RMSE: {rmse:.6f}')


test_pred_xgb = xgb.predict(X_test)
test_pred_lgb = lgb.predict(X_test)
test_pred_cat = cat.predict(X_test)

final_pred = 0.20 * test_pred_xgb + 0.15 * test_pred_lgb + 0.65 * test_pred_cat
final_pred = np.expm1(final_pred)

submission = pd.DataFrame({
    'id': test['id'],
    'Calories': final_pred
})

submission.to_csv('/kaggle/working/submission.csv', index=False)




