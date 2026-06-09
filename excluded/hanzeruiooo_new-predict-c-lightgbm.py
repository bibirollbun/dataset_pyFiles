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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e5/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e5/test.csv')
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e5/sample_submission.csv')


train_df.info()


test_df.info()


train_df.head()


object_columns = ['Sex']
train = pd.get_dummies(train_df, columns=object_columns, drop_first=True, dtype=int)
test  = pd.get_dummies(test_df, columns=object_columns, drop_first=True, dtype=int)
 
 
train.shape,test.shape


train.head()


train.describe()


train['Sex_male'].value_counts()


test.describe()


import matplotlib.pyplot as plt
import seaborn as sns

# 排除 'id' 列
columns_to_plot = [col for col in train.columns if col != 'id']

plt.figure(figsize=(12, 10))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    sns.histplot(train[column], kde=True, bins=10)
    plt.title(f'Distribution of {column}')
    plt.tight_layout()

plt.show()



# 查看重复值情况
train.duplicated().sum()


# 计算交互项
train['Age_Weight'] = train['Age'] * train['Weight']
train['Height_Heart_Rate'] = train['Height'] * train['Heart_Rate']
train['Weight_Duration'] = train['Weight'] * train['Duration']
train['Age_Height'] = train['Age'] * train['Height']
train['Heart_Rate_Body_Temp_Ratio'] = train['Heart_Rate'] / train['Body_Temp']
train['Body_Temp_Age_Diff'] = train['Body_Temp'] - train['Age']
# 计算交互项
test['Age_Weight'] = test['Age'] * test['Weight']
test['Height_Heart_Rate'] = test['Height'] * test['Heart_Rate']
test['Weight_Duration'] = test['Weight'] * test['Duration']
test['Age_Height'] = test['Age'] * test['Height']
test['Heart_Rate_Body_Temp_Ratio'] = test['Heart_Rate'] / test['Body_Temp']
test['Body_Temp_Age_Diff'] = test['Body_Temp'] - test['Age']



train.head()


test.head()


train_data = train.drop(columns = ['Calories','id'])

label = train['Calories']


# from sklearn.metrics import mean_squared_error
# import numpy as np
# import lightgbm as lgb
# from sklearn.model_selection import train_test_split
# import optuna

# # 数据准备
# x = train_data
# y = label
# X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# # ================== 定义LightGBM的超参数优化函数 ==================
# def objective_lgb(trial):
#     params = {
#         'objective': 'regression',
#         'boosting_type': 'gbdt',
#         'max_depth': trial.suggest_int('max_depth', 2, 15),  # 扩大搜索范围
#         'num_leaves': trial.suggest_int('num_leaves', 30, 300),  # 扩大搜索范围
#         'min_child_samples': trial.suggest_int('min_child_samples', 10, 500),  # 扩大搜索范围
#         'learning_rate': trial.suggest_float('learning_rate', 1e-5, 0.1),  # 扩大搜索范围
#         'subsample': trial.suggest_float('subsample', 0.4, 1.0),  # 扩大搜索范围
#         'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0),  # 扩大搜索范围
#         'reg_alpha': trial.suggest_float('reg_alpha', 1e-2, 100.0),  # 扩大搜索范围
#         'reg_lambda': trial.suggest_float('reg_lambda', 1e-2, 100.0),  # 扩大搜索范围
#         'random_state': 42,
#         'device': 'gpu',  # 启用 GPU 加速
#         'gpu_platform_id': 2,  # GPU 平台 ID (如果有多个 GPU，可以选择)
#         'gpu_device_id': 2,  # GPU 设备 ID (如果有多个设备，可以选择)
#     }
    
#     model = lgb.LGBMRegressor(**params)
    
#     # 训练模型
#     model.fit(X_train, y_train)
    
#     # 预测并计算 RMSE
#     pred = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, pred))
    
#     return rmse  # 返回 RMSE 作为目标函数

# # ================== 执行超参数优化 ==================
# study_lgb = optuna.create_study(direction='minimize')
# study_lgb.optimize(objective_lgb, n_trials=50)

# # 输出最优参数
# best_lgb = study_lgb.best_params
# best_lgb['random_state'] = 42
# best_lgb['device'] = 'gpu'  # 确保最佳超参数包含 GPU 设置

# # 输出最优参数
# print("最优参数: ", best_lgb)

# # ================== 定义LightGBM模型并训练 ==================
# lgb_model = lgb.LGBMRegressor(**best_lgb)
# lgb_model.fit(X_train, y_train)

# # ================== 评估原始模型 ==================
# pred = lgb_model.predict(X_test)
# initial_rmse = np.sqrt(mean_squared_error(y_test, pred))  # 计算 RMSE

# print("测试集 RMSE: ", initial_rmse)



from sklearn.metrics import mean_squared_error
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold
import joblib  # 用于保存最佳模型

# 数据准备
x = train_data
y = label
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# 最优参数
best_params = {
    'max_depth': 15,
    'num_leaves': 207,
    'min_child_samples': 21,
    'learning_rate': 0.07673095235608113,
    'subsample': 0.6809107669348211,
    'colsample_bytree': 0.7566787396639643,
    'reg_alpha': 0.5112472482344856,
    'reg_lambda': 23.443756905245287,
    'random_state': 42,
    'device': 'gpu'
}

# 使用5折交叉验证训练模型
kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
best_rmse = float('inf')  # 初始设置一个非常大的 RMSE 值
best_model = None  # 保存最好的模型

for fold, (train_idx, valid_idx) in enumerate(kf.split(X_train), 1):
    X_train_fold, X_valid_fold = X_train.iloc[train_idx], X_train.iloc[valid_idx]
    y_train_fold, y_valid_fold = y_train.iloc[train_idx], y_train.iloc[valid_idx]
    
    model = lgb.LGBMRegressor(**best_params)
    
    # 训练模型
    model.fit(X_train_fold, y_train_fold)
    
    # 预测并计算 RMSE
    pred = model.predict(X_valid_fold)
    rmse = np.sqrt(mean_squared_error(y_valid_fold, pred))
    rmse_scores.append(rmse)
    
    # 输出当前折的 RMSE
    print(f'Fold {fold} RMSE: {rmse}')
    
    # 如果当前折的 RMSE 更好，保存该模型
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model

# 输出5折交叉验证的平均 RMSE
average_rmse = np.mean(rmse_scores)
print(f'5折交叉验证的平均 RMSE: {average_rmse}')

# 保存最好的模型
if best_model:
    joblib.dump(best_model, 'best_model.joblib')  # 保存模型到文件
    print("最佳模型已保存！")



# # ================== 特征回溯过程 ==================
# rmse_results = {}

# for feature in X_train.columns:
#     X_backtest = X_train.drop(columns=[feature])
    
#     # 确保X_train和y_train的样本数一致
#     X_train_backtest, X_test_backtest, y_train_backtest, y_test_backtest = train_test_split(
#         X_backtest, y_train, test_size=0.3, random_state=42
#     )
    
#     # 在去除一个特征后重新训练模型
#     lgb_model.fit(X_train_backtest, y_train_backtest)
#     y_pred_backtest = lgb_model.predict(X_test_backtest)
    
#     # 计算去除特征后的RMSE
#     rmse_backtest = np.sqrt(mean_squared_error(y_test_backtest, y_pred_backtest))
#     rmse_results[feature] = rmse_backtest

# # 输出每个特征去除后的RMSE变化
# print("\n特征去除后的RMSE变化：")
# for feature, rmse_value in rmse_results.items():
#     print(f'去除特征 {feature} 后，RMSE: {rmse_value}, RMSE变化: {rmse_value - initial_rmse}')


import joblib  # 用于加载保存的模型

# 加载最佳模型
best_model = joblib.load('best_model.joblib')

# 预处理测试数据
test_data = test.drop(columns=['id'])

# 使用加载的模型进行预测
predictions = best_model.predict(test_data)

# 获取 id 列
ids = test['id'].copy()

# 创建一个 DataFrame，将预测结果和 id 组合在一起
result = pd.DataFrame({
    'id': ids,
    'Calories': predictions
})

# 将结果保存到 CSV 文件中
result.to_csv('prediction_results.csv', index=False)

# 返回预测结果
result


