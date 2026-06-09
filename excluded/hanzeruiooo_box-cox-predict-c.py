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


test_df['BMI'] = test_df['Weight'] / (test_df['Height'] ** 2)
test_df['Age_Group'] = pd.cut(test_df['Age'], bins=[0, 18, 30, 40, 50, 60, 100], labels=["<18", "18-30", "30-40", "40-50", "50-60", "60+"])
test_df['Intensity'] = test_df['Duration'] * test_df['Heart_Rate']
test_df['Temp_Diff'] = abs(test_df['Body_Temp'] - 37)
test_df['Height_Weight_Ratio'] = test_df['Height'] / test_df['Weight']
test_df['Heart_Rate_Temperature'] = test_df['Heart_Rate'] * test_df['Body_Temp']
test_df = pd.get_dummies(test_df, columns=['Age_Group'], drop_first=True)

train_df['BMI'] = train_df['Weight'] / (train_df['Height'] ** 2)
train_df['Age_Group'] = pd.cut(train_df['Age'], bins=[0, 18, 30, 40, 50, 60, 100], labels=["<18", "18-30", "30-40", "40-50", "50-60", "60+"])
train_df['Intensity'] = train_df['Duration'] * train_df['Heart_Rate']
train_df['Temp_Diff'] = abs(train_df['Body_Temp'] - 37)
train_df['Height_Weight_Ratio'] = train_df['Height'] / train_df['Weight']
train_df['Heart_Rate_Temperature'] = train_df['Heart_Rate'] * train_df['Body_Temp']
train_df = pd.get_dummies(train_df, columns=['Age_Group'], drop_first=True)



test_df.info()


train_df.info()


object_columns = ['Sex']
train = pd.get_dummies(train_df, columns=object_columns, drop_first=True, dtype=int)
test  = pd.get_dummies(test_df, columns=object_columns, drop_first=True, dtype=int)
 
 
train.shape,test.shape


train.columns


test.columns


train.describe()


train['Sex_male'].value_counts()


test.describe()


# import matplotlib.pyplot as plt
# import seaborn as sns

# # 排除 'id' 列
# columns_to_plot = [col for col in train.columns if col != 'id' and col != 'Age_Group']

# plt.figure(figsize=(12, 10))
# for i, column in enumerate(columns_to_plot, 1):
#     plt.subplot(3, 3, i)
#     sns.histplot(train[column], kde=True, bins=10)
#     plt.title(f'Distribution of {column}')
#     plt.tight_layout()

# plt.show()



# 查看重复值情况
train.duplicated().sum()


import numpy as np
from scipy.stats import boxcox

columns_to_boxcox_transform = ['Age', 'Weight', 'Body_Temp']

df_boxcox_transformed = train.copy()

for column in columns_to_boxcox_transform:
    # 对每一列进行Box-Cox变换
    # Box-Cox变换要求数据为正数，如果数据可能包含零或负数，可以对数据进行平移（加上一个常数）以确保所有值都为正
    df_boxcox_transformed[column], _ = boxcox(df_boxcox_transformed[column])  # +1是为了避免负值或零

df_boxcox_transformed.info()


columns_to_boxcox_transform = ['Age', 'Weight', 'Body_Temp']

df_boxcox_transformed_test = test.copy()

for column in columns_to_boxcox_transform:
    df_boxcox_transformed_test[column], _ = boxcox(df_boxcox_transformed_test[column] + 1)  

df_boxcox_transformed_test.info()


import matplotlib.pyplot as plt
import seaborn as sns

# 排除 'id' 列
columns_to_plot = ['Age', 'Weight', 'Body_Temp']

plt.figure(figsize=(12, 10))

# 使用 enumerate 获取索引和值
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(3, 3, i)
    sns.histplot(df_boxcox_transformed[column], kde=True, bins=10)  # 使用单列数据
    plt.title(f'Distribution of {column}')

# 调整布局
plt.tight_layout()
plt.show()



train_data = df_boxcox_transformed.drop(columns = ['Calories','id'])

label = df_boxcox_transformed['Calories']


from sklearn.metrics import mean_squared_error
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
import optuna

# 数据准备
x = train_data
y = label
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# ================== 定义LightGBM的超参数优化函数 ==================
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
    return np.sqrt(mean_squared_error(y_test, pred))  # 这里返回 RMSE

# ================== 执行超参数优化 ==================
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=50)
best_lgb = study_lgb.best_params
best_lgb['random_state'] = 42

# ================== 定义LightGBM模型并训练 ==================
lgb_model = lgb.LGBMRegressor(**best_lgb)
lgb_model.fit(X_train, y_train)

# ================== 评估原始模型 ==================
pred = lgb_model.predict(X_test)
initial_rmse = np.sqrt(mean_squared_error(y_test, pred))  # 计算 RMSE

initial_rmse


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


test_data = df_boxcox_transformed_test.drop(columns = ['id'])
predictions = lgb_model.predict(test_data)

ids = df_boxcox_transformed_test['id'].copy()

# 创建一个 DataFrame，将预测结果和 id 组合在一起
result = pd.DataFrame({
    'id': ids,
    'Calories': predictions
})

# 将结果保存到CSV文件中
result.to_csv('prediction_results.csv', index=False)

# 返回结果
result

