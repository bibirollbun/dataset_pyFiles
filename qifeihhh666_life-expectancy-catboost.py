from sklearn.model_selection import train_test_split  #划分数据集
from sklearn.metrics import mean_squared_error   #评估模型效果 MSE ≥ 0，值越小表示模型越好。
from catboost import CatBoostRegressor

import numpy as np
import pandas as pd
import re
import time
import math

import warnings
warnings.filterwarnings("ignore")

print("ok")


def fill_missing_values(df_):
    columns = df_.columns
    for column in columns:
        if df_[column].dtype == 'object':  # 如果是字符列
            mode_value = df_[column].mode()[0]  # 获取出现次数最多的字符
            df_[column].fillna(mode_value, inplace=True)
        else:  # 如果是数值列
            mean_value = df_[column].mean()  # 计算平均值
            df_[column].fillna(mean_value, inplace=True)
    return df_


train = pd.read_csv('/kaggle/input/xdu-hic-math-2025/train.csv')  
train=fill_missing_values(train)
train['Status'] = train['Status'].map({'Developing': 1, 'Developed': 2})

table_feature=['Year', 'Adult Mortality','Status',
       'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
       'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
       'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
       ' thinness  1-19 years', ' thinness 5-9 years',
       'Income composition of resources', 'Schooling']


X = train[table_feature]
y = train['Life expectancy ']

# 划分训练集 测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("ok")


# CatBoost（超参数调优）
cat_params = {
    'iterations': 5000,      # 树的数量
    'learning_rate': 0.05,   # 学习率
    'depth': 6,              # 树深度
    'l2_leaf_reg': 3,        # L2正则化系数
    'random_state': 42,
    'verbose': 0
}
 
cat_optimized = CatBoostRegressor(**cat_params)
cat_optimized.fit(
    X,y,
    #X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=100,  # 提前停止
)


y_pred = cat_optimized.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE score: {mse}")



#预测测试集 
test = pd.read_csv('/kaggle/input/xdu-hic-math-2025/test.csv')
test=fill_missing_values(test)
test['Status'] = test['Status'].map({'Developing': 1, 'Developed': 2})
y_pred_test = cat_optimized.predict(test[table_feature])

#保存结果
sample_submission = pd.read_csv('/kaggle/input/xdu-hic-math-2025/sample_submission.csv')
sample_submission['Life expectancy ']=y_pred_test
sample_submission.to_csv('CatBoost_params.csv',index=False)
sample_submission.head()

