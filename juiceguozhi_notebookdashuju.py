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
import pandas as pd       # 用于数据处理和分析
import numpy as np        # 用于数值计算
import matplotlib.pyplot as plt  # 用于数据可视化
import seaborn as sns    # 基于matplotlib的高级可视化库
import os                # 用于文件和目录操作

# 检查数据集目录
print("数据集目录结构：")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# 加载数据（从Parquet格式文件读取）
print("\n开始加载训练数据和测试数据...")
train_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

# 查看数据的基本信息
print("\n训练数据的基本信息：")
print(train_data.info())
print("\n测试数据的基本信息：")
print(test_data.info())



# 数据质量检查：缺失值统计
print("训练数据的缺失值情况：")
print(train_data.isnull().sum())  # isnull()检测缺失值，sum()按列汇总缺失数量

print("\n测试数据的缺失值情况：")
print(test_data.isnull().sum())   # 对测试集执行相同的缺失值检查

# 特征工程准备：检查列名
print("\n训练数据的列名：")
print(train_data.columns)         # 输出列名列表，用于确认特征一致性

# 数据探查：查看数据结构和前几行
print("\n训练数据的前五行：")
print(train_data.head())          # 展示数据集行数、列名、数据类型和前5行内容


# 目标变量分布可视化 - 理解预测目标的基本统计特性
plt.figure(figsize=(10, 6))  # 设置图表大小
sns.histplot(train_data['label'], bins=50, kde=True)  # 绘制直方图并添加核密度估计曲线
plt.title('目标变量（label）的分布')
plt.xlabel('目标值')
plt.ylabel('频率')
plt.show()

# 特征分布可视化 - 分析单个特征的数据特性
plt.figure(figsize=(10, 6))
sns.histplot(train_data['X1'], bins=50, kde=True)  # 对特征X1进行分布分析
plt.title('特征X1的分布')
plt.xlabel('X1值')
plt.ylabel('频率')
plt.show()

# 时间序列趋势可视化 - 抽取1000个样本点分析目标变量随时间的变化趋势
sample_data = train_data.sample(n=1000, random_state=42)  # 固定随机种子确保结果可复现
plt.figure(figsize=(12, 6))
plt.scatter(sample_data.index, sample_data['label'], alpha=0.5)  # 使用散点图展示时间序列关系
plt.title('时间戳与目标变量的关系（抽样数据）')
plt.xlabel('时间戳（索引）')
plt.ylabel('目标值')
plt.show()

# 特征相关性分析 - 计算特征间的皮尔逊相关系数并可视化
selected_features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'X1', 'label']  # 选择需要分析的特征
corr_matrix = train_data[selected_features].corr()  # 计算相关系数矩阵
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', square=True)  # 使用热力图可视化相关系数
plt.title('部分特征的相关性矩阵')
plt.show()


# 导入必要的机器学习工具库
from sklearn.model_selection import train_test_split  # 用于数据集划分
from sklearn.preprocessing import StandardScaler  # 用于特征标准化
from sklearn.linear_model import LinearRegression, LogisticRegression  # 线性回归和逻辑回归模型
from sklearn.metrics import mean_squared_error, accuracy_score  # 评估指标：均方误差和准确率

# 特征工程：选择用于模型训练的特征和目标变量
features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'X1']  # 选择的特征列
X = train_data[features]  # 特征矩阵
y = train_data['label']  # 目标变量

# 数据集划分：将数据分为训练集(80%)和验证集(20%)
# random_state固定随机种子确保结果可复现
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 特征标准化处理：使用Z-score方法将特征缩放到均值为0，标准差为1的分布
scaler = StandardScaler()  # 初始化标准化器
X_train_scaled = scaler.fit_transform(X_train)  # 计算训练集的均值和标准差并应用标准化
X_val_scaled = scaler.transform(X_val)  # 使用训练集的统计参数对验证集进行标准化


# 初始化线性回归模型 - 创建线性回归对象
lr_model = LinearRegression()  # 默认参数配置，适用于连续型目标变量预测

# 模型训练阶段 - 使用标准化后的训练数据拟合模型
lr_model.fit(X_train_scaled, y_train)  # 学习特征与目标变量之间的线性关系

# 训练集预测 - 应用训练好的模型对训练数据进行预测
y_train_pred_lr = lr_model.predict(X_train_scaled)  # 生成训练集预测结果

# 训练集评估 - 计算模型在训练数据上的均方误差(MSE)
train_mse_lr = mean_squared_error(y_train, y_train_pred_lr)  # 评估模型对已知数据的拟合能力
print(f'线性回归 - 训练集上的均方误差（MSE）：{train_mse_lr}')

# 验证集预测 - 使用训练好的模型对验证数据进行预测
y_val_pred_lr = lr_model.predict(X_val_scaled)  # 生成验证集预测结果

# 验证集评估 - 计算模型在验证数据上的均方误差(MSE)
val_mse_lr = mean_squared_error(y_val, y_val_pred_lr)  # 评估模型的泛化能力
print(f'线性回归 - 验证集上的均方误差（MSE）：{val_mse_lr}')
print(f'线性回归 - 验证集上的均方误差（MSE）：{val_mse_lr}')


# 特征重要性分析 - 基于线性回归模型的系数
feature_importance_lr = pd.DataFrame({
    '特征': features,                 # 特征名称列表
    '系数': lr_model.coef_            # 线性回归模型学习到的系数
}).sort_values(by='系数', ascending=False)  # 按系数降序排列

print("\n线性回归 - 特征系数：")
print(feature_importance_lr)               # 打印特征重要性排序结果



import pandas as pd                  # 数据处理库，用于DataFrame操作
import numpy as np                   # 数值计算库，提供数组操作和数学函数
from sklearn.model_selection import train_test_split  # 用于数据集划分
from sklearn.preprocessing import StandardScaler     # 特征标准化工具
from sklearn.linear_model import LogisticRegression  # 逻辑回归分类模型
from sklearn.metrics import accuracy_score           # 分类模型评估指标

# 设置随机种子以保证结果可复现
np.random.seed(42)  # 固定随机数生成器种子，确保实验结果可重复

# 加载数据 - 从Parquet文件读取训练集和测试集
train_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

# 数据探查：查看训练数据的列名
print("训练数据的列名：")
print(train_data.columns)  # 输出所有列名，确认特征和目标变量的存在性

# 数据预处理：缺失值处理
train_data.fillna(method='ffill', inplace=True)  # 使用前向填充法处理缺失值，用前一个有效值填充

# 目标变量处理：将连续值转换为二分类变量
target_column = 'label'  # 定义目标变量列名，需根据实际数据调整
# 创建分类目标：当label>0时标记为1（上涨），否则为0（下跌）
train_data['target_class'] = np.where(train_data[target_column] > 0, 1, 0)

# 特征工程：选择用于建模的特征
features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'X1']  # 选择的特征列
X = train_data[features]  # 特征矩阵
y = train_data['target_class']  # 分类目标变量

# 数据集划分：将数据分为训练集(80%)和验证集(20%)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42)  # 固定随机种子保证划分一致性

# 特征标准化：将特征缩放到相同尺度
scaler = StandardScaler()  # 初始化标准化器
X_train_scaled = scaler.fit_transform(X_train)  # 训练集标准化（计算均值和标准差并应用）
X_val_scaled = scaler.transform(X_val)  # 验证集标准化（使用训练集的统计量）


# 初始化逻辑回归模型 - 设置最大迭代次数为1000以确保收敛
logr_model = LogisticRegression(max_iter=1000)  # 创建逻辑回归分类器，适用于二分类问题

# 模型训练 - 使用标准化后的训练数据拟合模型
logr_model.fit(X_train_scaled, y_train)  # 学习特征与目标类别之间的逻辑关系

# 训练集预测 - 应用训练好的模型对训练数据进行预测
y_train_pred_logr = logr_model.predict(X_train_scaled)  # 生成训练集类别预测结果

# 训练集评估 - 计算模型在训练数据上的准确率
train_acc_logr = accuracy_score(y_train, y_train_pred_logr)  # 评估模型对已知数据的分类能力
print(f'\n逻辑回归 - 训练集上的准确率（Accuracy）：{train_acc_logr:.4f}')

# 验证集预测 - 使用训练好的模型对验证数据进行预测
y_val_pred_logr = logr_model.predict(X_val_scaled)  # 生成验证集类别预测结果

# 验证集评估 - 计算模型在验证数据上的准确率
val_acc_logr = accuracy_score(y_val, y_val_pred_logr)  # 评估模型的泛化能力
print(f'逻辑回归 - 验证集上的准确率（Accuracy）：{val_acc_logr:.4f}')

# 特征重要性分析 - 基于逻辑回归模型的系数
feature_importance_logr = pd.DataFrame({
    '特征': features,                 # 特征名称列表
    '系数': logr_model.coef_[0]       # 逻辑回归模型学习到的系数（取第一行，因为是二分类）
}).sort_values(by='系数', ascending=False)  # 按系数降序排列

print("\n逻辑回归 - 特征系数：")
print(feature_importance_logr)               # 打印特征重要性排序结果


# 准备测试集特征 - 选择与训练集相同的特征列
X_test = test_data[features]  # 使用预定义的特征列表从测试数据中提取特征

# 特征标准化 - 使用训练集的标准化参数处理测试集
X_test_scaled = scaler.transform(X_test)  # 使用训练集拟合的scaler进行标准化，避免数据泄露

# 模型预测 - 对测试集进行类别预测
test_pred_logr = logr_model.predict(X_test_scaled)  # 生成测试集的类别预测结果

# 创建提交文件 - 按照竞赛要求的格式构建结果DataFrame
submission_logr = pd.DataFrame({
    'ID': test_data.index,          # 使用测试数据的索引作为ID
    'label': test_pred_logr         # 预测的类别标签
})

# 保存提交文件 - 输出为CSV格式
submission_logr.to_csv('submission_logr.csv', index=False)  # 不保存行索引
print("逻辑回归 - 提交文件已保存为 'submission_logr.csv'")


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, accuracy_score

# 设置随机种子以保证结果可复现
np.random.seed(42)

# 加载数据 - 确保train_data被正确定义
train_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/train.parquet')
test_data = pd.read_parquet('/kaggle/input/drw-crypto-market-prediction/test.parquet')

# 查看数据的列名
print("训练数据的列名：")
print(train_data.columns)

# 数据预处理步骤（根据实际情况调整）
# 例如：处理缺失值、特征工程、时间特征提取等
# 这里添加简单的缺失值处理
train_data.fillna(method='ffill', inplace=True)

# 假设目标列名为'label'，而不是'target'，我们需要根据实际情况进行调整
target_column = 'label'  # 根据实际情况修改这里

# 创建分类目标（如果原始目标是连续的）
# 例如：1表示价格上涨，0表示价格下跌
train_data['target_class'] = np.where(train_data[target_column] > 0, 1, 0)

# 分割特征和目标变量
features = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume', 'X1']
X = train_data[features]
y_reg = train_data[target_column]  # 回归目标
y_cls = train_data['target_class']  # 分类目标

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y_reg, test_size=0.2, random_state=42)
X_train_cls, X_val_cls, y_train_cls, y_val_cls = train_test_split(X, y_cls, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_train_cls_scaled = scaler.fit_transform(X_train_cls)
X_val_cls_scaled = scaler.transform(X_val_cls)

# 初始化线性回归模型
lr_model = LinearRegression()

# 训练线性回归模型
lr_model.fit(X_train_scaled, y_train)

# 在训练集上进行预测
y_train_pred_lr = lr_model.predict(X_train_scaled)

# 计算训练集上的均方误差（MSE）
train_mse_lr = mean_squared_error(y_train, y_train_pred_lr)

# 在验证集上进行预测
y_val_pred_lr = lr_model.predict(X_val_scaled)

# 计算验证集上的均方误差（MSE）
val_mse_lr = mean_squared_error(y_val, y_val_pred_lr)

# 初始化逻辑回归模型
logr_model = LogisticRegression(max_iter=1000)

# 训练逻辑回归模型
logr_model.fit(X_train_cls_scaled, y_train_cls)

# 在训练集上进行预测
y_train_pred_logr = logr_model.predict(X_train_cls_scaled)

# 计算训练集上的准确率
train_acc_logr = accuracy_score(y_train_cls, y_train_pred_logr)

# 在验证集上进行预测
y_val_pred_logr = logr_model.predict(X_val_cls_scaled)

# 计算验证集上的准确率
val_acc_logr = accuracy_score(y_val_cls, y_val_pred_logr)


# 评估模型的拟合情况
print("\n模型评估：")
print(f"线性回归 - 训练集 MSE: {train_mse_lr:.4f}")
print(f"线性回归 - 验证集 MSE: {val_mse_lr:.4f}")
print(f"逻辑回归 - 训练集 Accuracy: {train_acc_logr:.4f}")
print(f"逻辑回归 - 验证集 Accuracy: {val_acc_logr:.4f}")


# 检查模型是否过拟合或欠拟合
if val_mse_lr > train_mse_lr * 1.5:
    print("\n线性回归 - 警告：模型可能存在过拟合现象。")
elif val_mse_lr < train_mse_lr * 0.8:
    print("\n线性回归 - 警告：模型可能存在欠拟合现象。")
else:
    print("\n线性回归 - 模型拟合情况良好。")

if val_acc_logr < train_acc_logr * 0.8:
    print("\n逻辑回归 - 警告：模型可能存在过拟合现象。")
elif val_acc_logr > train_acc_logr * 1.2:
    print("\n逻辑回归 - 警告：模型可能存在欠拟合现象。")
else:
    print("\n逻辑回归 - 模型拟合情况良好。")


# 对测试集进行预测并提交
X_test = test_data[features]
X_test_scaled = scaler.transform(X_test)

test_pred_logr = logr_model.predict(X_test_scaled)

# 创建提交文件
submission_logr = pd.DataFrame({
    'ID': test_data.index,
    'prediction': test_pred_logr  # 修改列名为 'prediction'
})
submission_logr.to_csv('submission_logr.csv', index=False)
print("逻辑回归 - 提交文件已保存为 'submission_logr.csv'")

