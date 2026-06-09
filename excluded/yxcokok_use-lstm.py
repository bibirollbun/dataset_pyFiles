# 导入必要的库
import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

# 加载数据
print("正在加载数据文件...")
sales_data = pd.read_csv('../input/sales_train.csv')  # 销售数据
test_data = pd.read_csv('../input/test.csv')  # 测试数据

# 数据预处理
print("\n正在处理销售数据...")
# 将日期列转换为datetime格式
sales_data['date'] = pd.to_datetime(sales_data['date'], format='%d.%m.%Y')

# 创建透视表：按月统计每个商店-商品组合的销量总和
print("\n创建销售数据透视表...")
dataset = sales_data.pivot_table(
    index=['shop_id', 'item_id'],  # 行索引：商店ID和商品ID
    values=['item_cnt_day'],  # 值：每日销量
    columns=['date_block_num'],  # 列：月份编号
    fill_value=0,  # 缺失值填充为0
    aggfunc='sum'  # 聚合函数：求和
)

# 重置索引以便后续操作
dataset.reset_index(inplace=True)
print("\n透视表前5行：")
print(dataset.head())

# 合并测试数据
print("\n合并测试数据集...")
dataset = pd.merge(
    test_data,  # 测试数据
    dataset,  # 透视表数据
    on=['shop_id', 'item_id'],  # 合并键
    how='left'  # 左连接
)

# 清理不需要的列
print("\n清理不需要的列...")
dataset.drop(['ID'], inplace=True, axis=1)
print("\n处理后数据集前5行：")
print(dataset.head())

# 检查数据完整性
if dataset.isnull().values.any():
    print("数据中存在缺失值，进行填充...")
    dataset.fillna(0, inplace=True)

# 准备训练和测试数据
print("\n准备训练和测试数据...")
# 训练数据：所有列除了最后一列作为特征
X_train = np.expand_dims(dataset.iloc[:-1, :-1].values, axis=2)
# 训练标签：最后一列作为目标值
y_train = dataset.iloc[:-1, -1].values

# 测试数据：所有列作为特征
X_test = np.expand_dims(dataset.iloc[-1:, :-1].values, axis=2)

print("\n数据形状：")
print(f"训练特征：{X_train.shape}，训练标签：{y_train.shape}，测试特征：{X_test.shape}")

# 构建LSTM模型
print("\n构建LSTM模型...")
# 初始化序贯模型
my_model = Sequential()

# 添加LSTM层
my_model.add(LSTM(
    units=16,  # 减少LSTM单元数量
    input_shape=(X_train.shape[1], 1)  # 输入形状：时间步数，每个时间步1个特征
))

# 添加Dropout层防止过拟合
my_model.add(Dropout(0.2))  # 减少Dropout比例

# 添加全连接输出层
my_model.add(Dense(1))  # 输出1个值（预测销量）

# 编译模型
my_model.compile(
    loss='mse',  # 使用均方误差作为损失函数
    optimizer='adam',  # 使用Adam优化器
    metrics=['mean_squared_error']  # 评估指标：均方误差
)

# 输出模型摘要
print("\n模型结构摘要：")
my_model.summary()

# 训练模型
print("\n开始训练模型...")
history = my_model.fit(
    X_train,  # 训练特征
    y_train,  # 训练标签
    batch_size=4096,  # 批量大小
    epochs=5,  # 减少训练轮数
    validation_split=0.2  # 使用20%的数据作为验证集
)

# 生成预测结果
print("\n生成预测结果...")
submission_pfs = my_model.predict(X_test)  # 在测试集上进行预测

# 将预测结果限制在0-20之间（比赛要求）
submission_pfs = submission_pfs.clip(0, 20)

# 输出预测结果
print("\n预测结果：")
print(submission_pfs)

print("\n全部流程完成！")

