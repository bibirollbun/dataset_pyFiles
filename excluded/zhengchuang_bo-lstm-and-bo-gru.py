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


import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler



train_df = pd.read_csv(r'/kaggle/input/playground-series-s5e1/train.csv')
test_df = pd.read_csv(r'/kaggle/input/playground-series-s5e1/test.csv')


train_df.head()


test_df.head()


print(train_df.info())


print(test_df.info())


train_df = train_df.dropna()


# 确保日期列为 datetime 类型
train_df['date'] = pd.to_datetime(train_df['date'])

# 设置 Seaborn 样式
sns.set_style('whitegrid')

# 创建绘图
plt.figure(figsize=(20, 8))  # 设置图表大小

# 使用 Seaborn 绘制折线图
sns.lineplot(x='date', y='num_sold', data=train_df, color='blue',errorbar=None, linewidth=0.8)

# 标记每年的 1 月 1 日
key_dates = pd.date_range(start=train_df['date'].min(), end=train_df['date'].max(), freq='12MS')  # 每年的 1 月 1 日
for date in key_dates:
    plt.axvline(date, color='red', linestyle='--', linewidth=0.8, alpha=0.7)  # 绘制竖线标记

# 设置标题和轴标签
plt.title('Total Sales Over Time', fontsize=16)
plt.xlabel('Date', fontsize=14)
plt.ylabel('num_sold', fontsize=14)

# 美化 x 轴刻度
plt.xticks(fontsize=12, rotation=45)
plt.yticks(fontsize=12)

# 显示图表
plt.tight_layout()

# 显示图表
plt.show()



# 计算每个分类值的频率
product_counts = train_df['product'].value_counts()

# 绘制条形图
product_counts.plot(kind='bar', figsize=(8, 6), color='skyblue')
plt.title('Frequency of Products')
plt.xlabel('Product')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# 计算每个分类值的频率
product_counts = train_df['country'].value_counts()

# 绘制条形图
product_counts.plot(kind='bar', figsize=(8, 6), color='skyblue')
plt.title('Frequency of country')
plt.xlabel('country')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# 计算每个分类值的频率
product_counts = train_df['store'].value_counts()

# 绘制条形图
product_counts.plot(kind='bar', figsize=(8, 6), color='skyblue')
plt.title('Frequency of store')
plt.xlabel('store')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# 提取日期
train_df['weekday'] = train_df['date'].dt.dayofweek
train_df['month'] = train_df['date'].dt.month
train_df['day'] = train_df['date'].dt.day
train_df['day_of_year'] = train_df['date'].dt.dayofyear

train_df['is_weekend'] = train_df['date'].dt.weekday.isin([5, 6])

train_df['is_year_start'] = train_df['date'].dt.dayofyear == 1
# 获取年份总天数（365 或 366）
train_df['is_year_end'] = train_df['date'].dt.dayofyear == train_df['date'].dt.is_leap_year.apply(lambda is_leap: 366 if is_leap else 365)

# 修正的季节分配函数
def get_season(date):
    if (date.month == 12 and date.day >= 21) or (date.month in [1, 2]) or (date.month == 3 and date.day <= 20):
        return 'Winter'
    elif (date.month == 3 and date.day >= 21) or (date.month in [4, 5]) or (date.month == 6 and date.day <= 20):
        return 'Spring'
    elif (date.month == 6 and date.day >= 21) or (date.month in [7, 8]) or (date.month == 9 and date.day <= 22):
        return 'Summer'
    elif (date.month == 9 and date.day >= 23) or (date.month in [10, 11]) or (date.month == 12 and date.day <= 20):
        return 'Fall'

# 添加季节列
train_df['season'] = train_df['date'].apply(get_season)

# 修正季节开始标记逻辑
def is_season_start(date):
    # 定义季节开始日期
    season_starts = {
        'Winter': [(12, 21)],  # 冬季开始
        'Spring': [(3, 21)],   # 春季开始
        'Summer': [(6, 21)],   # 夏季开始
        'Fall': [(9, 23)]      # 秋季开始
    }
    # 获取月份和日期
    month, day = date.month, date.day
    
    # 检查当前日期是否是季节开始日期
    for season, starts in season_starts.items():
        if (month, day) in starts:
            return True
    return False
# 应用函数，标记季节开始
train_df['is_season_start'] = train_df['date'].apply(is_season_start)

#周期性
train_df['weekday_sin'] = np.sin(2 * np.pi * train_df['weekday'] % 7 + 1)
train_df['weekday_cos'] = np.cos(2 * np.pi * train_df['weekday'] / 7 + 1)
train_df['month_sin'] = np.sin(2 * np.pi * train_df['month'] % 12 + 1)
train_df['month_cos'] = np.cos(2 * np.pi * train_df['month'] % 12 + 1)
train_df['day_sin'] = np.sin(2 * np.pi * train_df['day'] / 31)
train_df['day_cos'] = np.cos(2 * np.pi * train_df['day'] / 31)
train_df['day_of_year_sin'] = np.sin(2 * np.pi * train_df['day_of_year'] / 365)
train_df['day_of_year_cos'] = np.cos(2 * np.pi * train_df['day_of_year'] / 365)



bool_columns = ['is_year_start', 'is_year_end', 'is_season_start', 'is_weekend']
for col in bool_columns:
    train_df[col] = train_df[col].astype(int)

season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
train_df['season'] = train_df['season'].map(season_mapping)

train_df['store'] = train_df['store'].astype('category').cat.codes
train_df['product'] = train_df['product'].astype('category').cat.codes
train_df['country'] = train_df['country'].astype('category').cat.codes


test_df['date'] = pd.to_datetime(test_df['date'])
# 提取日期
test_df['weekday'] = test_df['date'].dt.dayofweek
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df['day_of_year'] = test_df['date'].dt.dayofyear

test_df['is_weekend'] = test_df['date'].dt.weekday.isin([5, 6])

test_df['is_year_start'] = test_df['date'].dt.dayofyear == 1
# 获取年份总天数（365 或 366）
test_df['is_year_end'] = test_df['date'].dt.dayofyear == test_df['date'].dt.is_leap_year.apply(lambda is_leap: 366 if is_leap else 365)

# 修正的季节分配函数
def get_season(date):
    if (date.month == 12 and date.day >= 21) or (date.month in [1, 2]) or (date.month == 3 and date.day <= 20):
        return 'Winter'
    elif (date.month == 3 and date.day >= 21) or (date.month in [4, 5]) or (date.month == 6 and date.day <= 20):
        return 'Spring'
    elif (date.month == 6 and date.day >= 21) or (date.month in [7, 8]) or (date.month == 9 and date.day <= 22):
        return 'Summer'
    elif (date.month == 9 and date.day >= 23) or (date.month in [10, 11]) or (date.month == 12 and date.day <= 20):
        return 'Fall'

# 添加季节列
test_df['season'] = test_df['date'].apply(get_season)

# 修正季节开始标记逻辑
def is_season_start(date):
    # 定义季节开始日期
    season_starts = {
        'Winter': [(12, 21)],  # 冬季开始
        'Spring': [(3, 21)],   # 春季开始
        'Summer': [(6, 21)],   # 夏季开始
        'Fall': [(9, 23)]      # 秋季开始
    }
    # 获取月份和日期
    month, day = date.month, date.day
    
    # 检查当前日期是否是季节开始日期
    for season, starts in season_starts.items():
        if (month, day) in starts:
            return True
    return False
# 应用函数，标记季节开始
test_df['is_season_start'] = test_df['date'].apply(is_season_start)

#周期性
test_df['weekday_sin'] = np.sin(2 * np.pi * test_df['weekday'] % 7 + 1)
test_df['weekday_cos'] = np.cos(2 * np.pi * test_df['weekday'] / 7 + 1)
test_df['month_sin'] = np.sin(2 * np.pi * test_df['month'] % 12 + 1)
test_df['month_cos'] = np.cos(2 * np.pi * test_df['month'] % 12 + 1)
test_df['day_sin'] = np.sin(2 * np.pi * test_df['day'] / 31)
test_df['day_cos'] = np.cos(2 * np.pi * test_df['day'] / 31)
test_df['day_of_year_sin'] = np.sin(2 * np.pi * test_df['day_of_year'] / 365)
test_df['day_of_year_cos'] = np.cos(2 * np.pi * test_df['day_of_year'] / 365)


bool_columns = ['is_year_start', 'is_year_end', 'is_season_start', 'is_weekend']
train_df['num_sold'] = np.log1p(train_df['num_sold'])
for col in bool_columns:
    test_df[col] = test_df[col].astype(int)

season_mapping = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3}
test_df['season'] = test_df['season'].map(season_mapping)

test_df['store'] = test_df['store'].astype('category').cat.codes
test_df['product'] = test_df['product'].astype('category').cat.codes
test_df['country'] = test_df['country'].astype('category').cat.codes


# 数据准备
x = train_df.drop(['num_sold', 'id', 'date'], axis=1)
y = train_df['num_sold']

# 分割数据集
x_train, x_valid, y_train, y_valid = train_test_split(x, y, test_size=0.2, random_state=42)

# 标准化数据
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_valid = scaler.transform(x_valid)  

# 标准化目标变量
target_scaler = StandardScaler()
y_train_scaled = target_scaler.fit_transform(y_train.values.reshape(-1, 1))
y_valid_scaled = target_scaler.transform(y_valid.values.reshape(-1, 1))

# 重塑数据
X_train_reshaped = x_train.reshape(x_train.shape[0], 1, x_train.shape[1])
X_valid_reshaped = x_valid.reshape(x_valid.shape[0], 1, x_valid.shape[1])




from tensorflow.keras.layers import LSTM
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(LSTM(50, input_shape=(1, X_train_reshaped.shape[2])))
model.compile(optimizer='adam', loss='mse')
history = model.fit(X_train_reshaped, y_train_scaled, epochs=30, batch_size=32, validation_data=(X_valid_reshaped, y_valid_scaled))


from tensorflow.keras.layers import GRU
gru_model = Sequential()

gru_model.add(GRU(32, input_shape=(1, X_train_reshaped.shape[2])))
gru_model.compile(optimizer='adam', loss='mse')
history = gru_model.fit(X_train_reshaped, y_train_scaled, epochs=30, batch_size=32, validation_data=(X_valid_reshaped, y_valid_scaled))



import warnings
warnings.filterwarnings('ignore')



'''
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

#目标函数：返回验证集上的损失值
def lstm_evaluate(learning_rate, units, dropout, num_layers):
    # 构建 LSTM 模型
    model = Sequential()
    for i in range(int(num_layers)):
        if i == int(num_layers) - 1:
            model.add(LSTM(int(units), activation='tanh', return_sequences=False, input_shape=(1, X_train_reshaped.shape[2])))
        else:
            model.add(LSTM(int(units), activation='tanh', return_sequences=True, input_shape=(1, X_train_reshaped.shape[2])))
        model.add(Dropout(dropout))
    model.add(Dense(1))

    #编译模型
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # 训练模型并获取历史对象
    history = model.fit(X_train_reshaped, y_train_scaled, epochs=1, batch_size=32, verbose=0, validation_data=(X_valid_reshaped, y_valid_scaled))

    #验证损失
    # 从历史对象中获取验证集损失
    loss = history.history['val_loss'][-1]
    return -loss # 贝叶斯优化会寻找最大值，所以取负值

#定义参数范围
pbounds = {
    'learning_rate': (0.001, 0.1), # 学习率范围
    'units': (10, 128),            # LSTM 隐藏单元数范围
    'dropout': (0.1, 0.5),
    'num_layers': (1,5)            # LSTM 层数范围
}


from bayes_opt import BayesianOptimization
optimizer = BayesianOptimization(
    f=lstm_evaluate,   # 优化目标函数
    pbounds = pbounds,  #参数搜索范围
    random_state = 42  #随机种子
)

#开始优化
optimizer.maximize(init_points=4, n_iter=20)

#打印最优参数
print("最佳参数", optimizer.max)
# 获取最优参数
best_params = optimizer.max['params']
'''
        


from tensorflow.keras.layers import LSTM, Dropout, Dense
from tensorflow.keras.optimizers import Adam
# 使用最优参数构建模型
final_model = Sequential()

# 需要用整数构造循环次数
for i in range(2):  # 修改为整数 2，代表 2 层
    if i == 1:  # 最后一层
        final_model.add(LSTM(37, activation='tanh', return_sequences=False, input_shape=(1, X_train_reshaped.shape[2])))  # 修改 units 为 37
    else:
        final_model.add(LSTM(37, activation='tanh', return_sequences=True, input_shape=(1, X_train_reshaped.shape[2])))  # 修改 units 为 37
    final_model.add(Dropout(0.274))  # 修改 dropout 为 0.274

final_model.add(Dense(1))

# 编译和训练模型
final_model.compile(optimizer=Adam(learning_rate=0.0178), loss='mse')  # 修改学习率为 0.0178
history = final_model.fit(X_train_reshaped, y_train_scaled, epochs=30, batch_size=32, validation_data=(X_valid_reshaped, y_valid_scaled))



'''from bayes_opt import BayesianOptimization
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
import numpy as np

# 定义模型训练函数
def train_gru(units_1, units_2, units_3,units_4, units_5, dropout_rate, learning_rate, num_layers):
    units_list = [int(units_1), int(units_2), int(units_3),int(units_4),int(units_5)]  # 确保神经元数是整数
    num_layers = int(num_layers)  # 层数也需要是整数
    model = Sequential()
    
    # 添加GRU层，单位数根据units_list动态调整，层数由num_layers决定
    for i in range(num_layers):
        units = units_list[i % len(units_list)]  # 循环使用 units_list 中的值
        model.add(GRU(units=units, activation='tanh', return_sequences=True if i < num_layers - 1 else False))
        model.add(Dropout(dropout_rate))

    # 添加输出层
    model.add(Dense(1, activation='sigmoid'))

   # 编译模型
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # 训练模型并获取历史对象
    history = model.fit(X_train_reshaped, y_train_scaled, epochs=1, batch_size=32, verbose=0, validation_data=(X_valid_reshaped, y_valid_scaled))

    #验证损失
    # 从历史对象中获取验证集损失
    loss = history.history['val_loss'][-1]
    return -loss # 贝叶斯优化会寻找最大值，所以取负值

# 定义贝叶斯优化的搜索空间
pbounds = {
    'units_1': (1, 150),  # 第1层神经元数的范围
    'units_2': (1, 150),  # 第2层神经元数的范围
    'units_3': (1, 150),  # 第3层神经元数的范围
    'units_4': (1, 150),  # 第4层神经元数的范围
    'dropout_rate': (0.1, 0.5),  # Dropout率范围
    'learning_rate': (0.0001, 0.01),  # 学习率范围
    'num_layers': (2, 4),  # 层数范围，设置最小为1，最大为5层
}

# 创建贝叶斯优化对象
optimizer = BayesianOptimization(
    f=train_gru,  # 目标函数
    pbounds=pbounds,  # 超参数搜索空间
    random_state=42
)

# 执行贝叶斯优化
optimizer.maximize(init_points=5, n_iter=10)  # 初始化探索5个点，进行10次迭代

# 输出最优参数
print("Best Parameters:", optimizer.max['params'])'''



'''
from bayes_opt import BayesianOptimization
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
import numpy as np

# 定义模型训练函数，固定3层GRU网络
def train_gru(units_1, units_2, units_3, dropout_rate, learning_rate):
    units_list = [int(units_1), int(units_2), int(units_3)]  # 固定为3层，设置不同的神经元数
    model = Sequential()
    
    # 添加3层GRU层，单位数根据units_list动态调整
    for i in range(3):  # 固定3层
        model.add(GRU(units=units_list[i], activation='tanh', return_sequences=True if i < 2 else False))
        model.add(Dropout(dropout_rate))

    model.add(Dense(1))

    #编译模型
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

    # 训练模型并获取历史对象
    history = model.fit(X_train_reshaped, y_train_scaled, epochs=15, batch_size=32, verbose=0, validation_data=(X_valid_reshaped, y_valid_scaled))

    #验证损失
    # 从历史对象中获取验证集损失
    loss = history.history['val_loss'][-1]
    return -loss # 贝叶斯优化会寻找最大值，所以取负值


# 定义贝叶斯优化的搜索空间
pbounds = {
    'units_1': (1, 150),  # 第1层神经元数的范围
    'units_2': (1, 150),  # 第2层神经元数的范围
    'units_3': (1, 150),  # 第3层神经元数的范围
    'dropout_rate': (0.1, 0.5),  # Dropout率范围
    'learning_rate': (0.0001, 0.01),  # 学习率范围
}

# 创建贝叶斯优化对象
optimizer = BayesianOptimization(
    f=train_gru,  # 目标函数
    pbounds=pbounds,  # 超参数搜索空间
    random_state=42
)

# 执行贝叶斯优化
optimizer.maximize(init_points=5, n_iter=20)  # 初始化探索5个点，进行10次迭代

# 输出最优参数
print("Best Parameters:", optimizer.max['params'])
'''


from bayes_opt import BayesianOptimization
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense
from keras.optimizers import Adam
from sklearn.model_selection import cross_val_score
import numpy as np



units_list = [int(130), int(96), int(106)]  # 固定为3层，设置不同的神经元数
model = Sequential()
    
# 添加3层GRU层，单位数根据units_list动态调整
for i in range(3):  # 固定3层
    model.add(GRU(units=units_list[i], activation='tanh', return_sequences=True if i < 2 else False))
    model.add(Dropout(0.1624))

model.add(Dense(1))

    #编译模型
model.compile(optimizer=Adam(learning_rate=0.000675), loss='mse')

    # 训练模型并获取历史对象
history = model.fit(X_train_reshaped, y_train_scaled, epochs=15, batch_size=32, validation_data=(X_valid_reshaped, y_valid_scaled))



'''from bayes_opt import BayesianOptimization
from keras.models import Sequential
from keras.layers import GRU, Dropout, Dense
from keras.optimizers import Adam

# 定义模型训练函数
def train_gru(units_1, units_2, units_3, units_4, dropout_rate, learning_rate, num_layers):
    units_list = []
    for i in range(int(num_layers)):  # 根据num_layers动态构建单位列表
        if i == 0:
            units_list.append(int(units_1))
        elif i == 1:
            units_list.append(int(units_2))
        elif i == 2:
            units_list.append(int(units_3))
        elif i == 3:
            units_list.append(int(units_4))
    
    model = Sequential()
    
    # 添加GRU层，单位数根据units_list动态调整
    for i in range(int(num_layers)):
        model.add(GRU(units=units_list[i], activation='tanh', return_sequences=True if i < int(num_layers) - 1 else False))
        model.add(Dropout(dropout_rate))

    # 添加输出层
    model.add(Dense(1, activation='sigmoid'))

    # 编译模型
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    
    # 使用训练数据进行拟合
    history = model.fit(X_train_reshaped, y_train_scaled, epochs=1, batch_size=32, verbose=0, validation_data=(X_valid_reshaped, y_valid_scaled))
    
    loss = history.history['val_loss'][-1]
    return -loss # 贝叶斯优化会寻找最大值，所以取负值
    
# 定义贝叶斯优化的搜索空间
pbounds = {
    'units_1': (1, 150),  # 第1层神经元数的范围
    'units_2': (1, 150),  # 第2层神经元数的范围
    'units_3': (1, 150),  # 第3层神经元数的范围
    'units_4': (1, 150),  # 第4层神经元数的范围
    'dropout_rate': (0.1, 0.5),  # Dropout率范围
    'learning_rate': (0.0001, 0.01),  # 学习率范围
    'num_layers': (2, 4),  # 层数范围，设置最小为2，最大为4层
}

# 创建贝叶斯优化对象
optimizer = BayesianOptimization(
    f=train_gru,  # 目标函数
    pbounds=pbounds,  # 超参数搜索空间
    random_state=42
)

# 执行贝叶斯优化
optimizer.maximize(init_points=5, n_iter=10)  # 初始化探索5个点，进行10次迭代

# 输出最优参数
print("Best Parameters:", optimizer.max['params'])
'''


loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Traiing and validation loss')
plt.legend()

plt.show()


from sklearn.metrics import mean_absolute_percentage_error
# 对验证集的预测
y_pred_scaled = model.predict(X_valid_reshaped)

# 逆标准化
y_pred = target_scaler.inverse_transform(y_pred_scaled)
y_valid_original = target_scaler.inverse_transform(y_valid_scaled)

# 计算 MAPE
mape = mean_absolute_percentage_error(y_valid_original, y_pred)
print(f"Validation MAPE: {mape * 100:.2f}%")


# 预处理测试集
features_test = test_df.drop(['id', 'date'], axis=1)  # 删除不需要的列
features_test_scaled = scaler.transform(features_test)  # 使用训练时的规则进行归一化

# 将测试集调整为 LSTM模型的输入形状
features_test_reshaped = features_test_scaled.reshape(features_test_scaled.shape[0], 1, features_test_scaled.shape[1])

#预测
y_pred_scaled=model.predict(features_test_reshaped)

# 逆标准化
y_pred_log = target_scaler.inverse_transform(y_pred_scaled)

#逆对数变换
predictions = np.expm1(y_pred_log)  

# 检查预测值形状
print(f"Predictions shape: {predictions.shape}")

# 创建输出文件
submission = pd.DataFrame({'id': test_df['id'], 'num_sold': predictions.ravel()})  # 保证预测值为一维
submission.to_csv('submission.csv', index=False)

# 提示保存成功
print("Your submission was successfully saved!")



submission




