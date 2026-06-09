# 导入必要的库
import numpy as np
import pandas as pd 
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.regularizers import l2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

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
X_train = np.expand_dims(dataset.values[:, :-1], axis=2)
# 训练标签：最后一列作为目标值
y_train = dataset.values[:, -1]

# 测试数据：所有列作为特征（我们需要预测所有测试数据）
X_test = np.expand_dims(dataset.values[:, :-1], axis=2)

# 特征缩放
print("\n进行特征缩放...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[1])).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[1])).reshape(X_test.shape)

print("\n数据形状：")
print(f"训练特征：{X_train_scaled.shape}，训练标签：{y_train.shape}，测试特征：{X_test_scaled.shape}")

# 构建LSTM模型
print("\n构建优化后的LSTM模型...")
# 初始化序贯模型
my_model = Sequential()

# 添加第一个LSTM层
my_model.add(LSTM(
    units=64,  # 增加单元数量
    input_shape=(X_train_scaled.shape[1], 1),
    return_sequences=True,  # 返回完整序列以便堆叠
    kernel_regularizer=l2(0.01)  # 添加L2正则化
))
my_model.add(Dropout(0.3))  # 适当增加Dropout比例

# 添加第二个LSTM层
my_model.add(LSTM(
    units=32,  # 第二层单元数量
    return_sequences=False,  # 不返回序列
    kernel_regularizer=l2(0.01)  # 添加L2正则化
))
my_model.add(Dropout(0.3))

# 添加全连接层
my_model.add(Dense(16, activation='relu'))  # 添加一个隐藏层
my_model.add(Dropout(0.2))

# 输出层
my_model.add(Dense(1))

# 编译模型
print("\n编译模型...")
optimizer = Adam(lr=0.001)  # 注意这里使用lr而不是learning_rate
my_model.compile(
    loss='mse',
    optimizer=optimizer,
    metrics=['mean_squared_error']
)

# 添加回调函数
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5),  # 早停
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3)  # 动态学习率
]

# 输出模型摘要
print("\n模型结构摘要：")
my_model.summary()

# 训练模型
print("\n开始训练模型...")
history = my_model.fit(
    X_train_scaled,  # 使用缩放后的数据
    y_train,
    batch_size=2048,  # 调整批量大小
    epochs=30,  # 增加epoch数量，配合早停使用
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# 生成预测结果
print("\n生成预测结果...")
submission_pfs = my_model.predict(X_test_scaled)  # 在缩放后的测试集上进行预测

# 将预测结果限制在0-20之间（比赛要求）
submission_pfs = submission_pfs.clip(0, 20)

# 创建提交文件
print("\n创建提交文件...")
# 重新加载测试数据以获取ID列
test_data = pd.read_csv('../input/test.csv')
submission = pd.DataFrame({
    'ID': test_data['ID'],
    'item_cnt_month': submission_pfs.flatten()  # 将预测结果展平为一维数组
})

# 保存为CSV文件
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)
print(f"\n提交文件已保存为: {submission_file}")

# 显示提交文件的前几行
print("\n提交文件前5行：")
print(submission.head())

print("\n全部流程完成！")

