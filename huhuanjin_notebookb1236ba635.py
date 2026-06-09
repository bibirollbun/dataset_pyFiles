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


# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 检查数据集目录
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 加载数据
train_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

# 查看数据的基本信息
print("训练数据的基本信息：")
print(train_data.info())
print("\n测试数据的基本信息：")
print(test_data.info())



# 检查缺失值
print("训练数据的缺失值情况：")
print(train_data.isnull().sum())
print("\n测试数据的缺失值情况：")
print(test_data.isnull().sum())

# 检查列名
print("\n训练数据的列名：")
print(train_data.columns)

# 查看数据结构
print("\n训练数据的前五行：")
print(train_data.head())



# 简单的数据可视化
# 目标变量的分布
plt.figure(figsize=(10, 6))
sns.histplot(train_data['label'], bins=50, kde=True)
plt.title('目标变量（label）的分布')
plt.xlabel('目标值')
plt.ylabel('频率')
plt.show()

# 特征X1的分布
plt.figure(figsize=(10, 6))
sns.histplot(train_data['X1'], bins=50, kde=True)
plt.title('特征X1的分布')
plt.xlabel('X1值')
plt.ylabel('频率')
plt.show()

# 时间戳与目标变量的关系（抽样数据）
sample_data = train_data.sample(n=1000, random_state=42)
plt.figure(figsize=(12, 6))
plt.scatter(sample_data.index, sample_data['label'], alpha=0.5)
plt.title('时间戳与目标变量的关系（抽样数据）')
plt.xlabel('时间戳（索引）')
plt.ylabel('目标值')
plt.show()

# 特征相关性分析
selected_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'X1', 'label']
corr_matrix = train_data[selected_features].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)
plt.title('部分特征的相关性矩阵')
plt.show()


# 导入必要的库
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score


train_data['timestamp'] = pd.to_datetime(train_data.index)
test_data['timestamp'] = pd.to_datetime(test_data.index)
# 构造小时、天、星期几特征
train_data['hour'] = train_data['timestamp'].dt.hour
train_data['day'] = train_data['timestamp'].dt.day
train_data['day_of_week'] = train_data['timestamp'].dt.dayofweek

test_data['hour'] = test_data['timestamp'].dt.hour
test_data['day'] = test_data['timestamp'].dt.day
test_data['day_of_week'] = test_data['timestamp'].dt.dayofweek



# 查看构造的新特征
print("\n训练数据的新特征：")
print(train_data[['timestamp', 'hour', 'day', 'day_of_week']].head())


# 创建分类目标（如果原始目标是连续的）
# 例如：1表示价格上涨，0表示价格下跌
target_column = 'label'
train_data['target_class'] = np.where(train_data[target_column] > 0, 1, 0)

# 分割特征和目标变量
# 特征选择 - 包括新构造的时间特征
features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'X1', 'hour', 'day', 'day_of_week']
X = train_data[features]
y = train_data['target_class']
# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)


# 初始化线性回归模型
lr_model = LinearRegression()

# 训练线性回归模型
lr_model.fit(X_train_scaled, y_train)

# 在训练集上进行预测
y_train_pred_lr = lr_model.predict(X_train_scaled)

# 计算训练集上的均方误差（MSE）
train_mse_lr = mean_squared_error(y_train, y_train_pred_lr)
print(f'线性回归 - 训练集上的均方误差（MSE）：{train_mse_lr}')

# 在验证集上进行预测
y_val_pred_lr = lr_model.predict(X_val_scaled)

# 计算验证集上的均方误差（MSE）
val_mse_lr = mean_squared_error(y_val, y_val_pred_lr)
print(f'线性回归 - 验证集上的均方误差（MSE）：{val_mse_lr}')


# 特征重要性分析（线性回归的系数）
feature_importance_lr = pd.DataFrame({
    '特征': features,
    '系数': lr_model.coef_
}).sort_values(by='系数', ascending=False)

print("\n线性回归 - 特征系数：")
print(feature_importance_lr)



# 对测试集进行预测并提交
X_test = test_data[features]
X_test_scaled = scaler.transform(X_test)

test_pred_lr = lr_model.predict(X_test_scaled)

# 创建提交文件
submission = pd.DataFrame({
    'ID': test_data.index,
    'prediction': test_pred_lr
})
submission.to_csv('submission.csv', index=False)
print("逻辑回归 - 提交文件已保存为 'submission.csv'")

