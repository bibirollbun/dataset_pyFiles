import pandas as pd
import numpy as np
import os
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, LSTM, Conv1D, Dropout, Bidirectional, Multiply, Permute, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import keras.backend as K


import tensorflow as tf
print("GPU is", "available" if tf.config.list_physical_devices('GPU') else "NOT AVAILABLE")
print(tf.config.list_physical_devices('GPU'))


train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
test_df = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')


train_nht.head(5)


test_df


# --------------------------
#  1. 数据加载与预处理 
# --------------------------
print("--- 1. Loading and Preprocessing Data ---")

def build_month_codes():
    """返回月份缩写到数字的映射字典。"""
    return {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }

def add_time_and_sector_fields(df, month_codes):
    """解析id或month列，添加'time'和'sector_id'列。"""
    df_copy = df.copy()
    
    # 如果有 'id' 列 (说明是测试集), 就解析id
    if 'id' in df_copy.columns:
        parts = df_copy['id'].str.split('_', expand=True)
        df_copy['month_str'] = parts[0]
        df_copy['sector'] = parts[1]
    else:
        df_copy['month_str'] = df_copy['month']

    if 'sector' in df_copy.columns:
         df_copy['sector_id'] = df_copy['sector'].str.slice(7, None).astype(int)

    df_copy['year'] = df_copy['month_str'].str.slice(0, 4).astype(int)
    df_copy['month_num'] = df_copy['month_str'].str.slice(5, None).map(month_codes)
    # 'time'列表示从2019年1月开始的第几个月 (从0开始)
    df_copy['time'] = (df_copy['year'] - 2019) * 12 + df_copy['month_num'] - 1
    
    return df_copy.drop(columns=['month_str', 'year'], errors='ignore')

def build_amount_matrix(train_nht, month_codes):
    """创建 [time(时间步长) x sector_id] 的交易额时间序列矩阵。"""
    train_nht_processed = add_time_and_sector_fields(train_nht, month_codes)
    pivot = train_nht_processed.set_index(['time', 'sector_id']).amount_new_house_transactions.unstack()
    pivot = pivot.fillna(0)
    
    # 确保所有96个板块都存在
    all_sectors = np.arange(1, 97)
    for s in all_sectors:
        if s not in pivot.columns:
            pivot[s] = 0
            
    return pivot[all_sectors]

# 加载数据
try:
    train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
    test_df = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')
except FileNotFoundError:
    print("请确保 new_house_transactions.csv 和 test.csv 文件在当前目录下")
    # 如果在Kaggle环境中，请使用下面的路径
    # train_nht = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/train/new_house_transactions.csv')
    # test_df = pd.read_csv('/kaggle/input/china-real-estate-demand-prediction/test.csv')

month_codes = build_month_codes()
a_tr = build_amount_matrix(train_nht, month_codes)
print("Data matrix shape:", a_tr.shape)
print("Data matrix preview:\n", a_tr.tail())





# --------------------------
#  2. 为深度学习构建特征和标签 (修正版)
# --------------------------
print("\n--- 2. Feature Engineering for Deep Learning Model ---")

from sklearn.preprocessing import MinMaxScaler
import joblib # 用于保存scaler

# 定义超参数
LOOKBACK = 12
# 特征数现在是2个：归一化后的交易额, 月份
# 我们不再将sector_id作为输入特征，而是为每个sector训练一个归一化器
N_FEATURES = 2 

# 创建一个字典来为每个板块保存一个MinMaxScaler
scalers = {}

def create_nn_dataset_v2(data_matrix):
    """将 [time x sector] 矩阵转换为 [样本数, 时间步, 特征数] 格式 (使用全局归一化)"""
    X, y = [], []
    
    # data_matrix.values.T 使得我们可以按板块进行遍历 (96, 67)
    for sector_idx, sector_ts in enumerate(tqdm(data_matrix.values.T, desc="Generating Features")):
        sector_id = data_matrix.columns[sector_idx]
        
        # 对当前板块的时间序列进行归一化
        # reshape(-1, 1) 是因为scaler要求输入是二维的
        scaler = MinMaxScaler(feature_range=(0, 1))
        ts_scaled = scaler.fit_transform(sector_ts.reshape(-1, 1)).flatten()
        scalers[sector_id] = scaler # 保存这个板块的scaler，预测时需要用
        
        # 为每个板块创建滑动窗口数据
        for i in range(len(ts_scaled) - LOOKBACK):
            sequence = ts_scaled[i : i + LOOKBACK]
            target = ts_scaled[i + LOOKBACK]
            
            # 构建特征
            features = []
            for j in range(LOOKBACK):
                time_step = data_matrix.index[i+j]
                month_feature = (time_step % 12) / 11.0  # 月份特征 (归一化到0-1)
                
                features.append([sequence[j], month_feature])
            
            X.append(features)
            y.append(target)
            
    return np.array(X), np.array(y)

X_train, y_train = create_nn_dataset_v2(a_tr)

# 保存scalers字典以备预测时使用
joblib.dump(scalers, 'scalers.pkl')

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")


X_train[1]


# --------------------------
#  3. 模型定义
# --------------------------
print("\n--- 3. Defining the Attention-based Deep Learning Model ---")

def attention_3d_block(inputs):
    """自定义注意力机制层"""
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Dense(LOOKBACK, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def attention_model(lstm_units=64, drop=0.2):
    """构建 CNN + BiLSTM + Attention 模型"""
    inputs = Input(shape=(LOOKBACK, N_FEATURES))

    # CNN层，提取局部特征
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = Dropout(drop)(x)

    # BiLSTM层，学习长期依赖
    lstm_out = Bidirectional(LSTM(lstm_units, return_sequences=True))(x)
    lstm_out = Dropout(drop)(lstm_out)
    
    # Attention层，关注重要时间步
    attention_mul = attention_3d_block(lstm_out)
    attention_mul = Flatten()(attention_mul)

    # 输出层
    output = Dense(1, activation='linear')(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model

# 实例化模型并查看结构
model = attention_model()
model.summary()


# --------------------------
#  4. 模型训练 
# --------------------------
print("\n--- 4. Training the Model ---")

# 配置模型
model.compile(optimizer='adam', loss='mae') # 使用MAE作为损失函数

# 设置回调函数
model_path = '/kaggle/working/best_attention_model.h5'
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, mode='min', restore_best_weights=True)
# model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', save_best_only=True, mode='min', verbose=1)

# 训练模型
history = model.fit(
    X_train, 
    y_train,
    epochs=50,          # 增加训练周期
    batch_size=256,       # 增大批大小以加快训练
    validation_split=0.1, # 使用10%数据作为验证集
    callbacks=[early_stopping],
    verbose=1
)


# --------------------------
#  5. 预测与生成提交文件 (修正版)
# --------------------------
print("\n--- 5. Generating Predictions ---")

# 重新加载之前保存的scalers
import joblib
scalers = joblib.load('scalers.pkl')

def create_submission_file(a_pred, test_df, month_codes, filename):
    """将宽格式的预测矩阵转换为最终提交的CSV文件 (源自你的脚本)"""
    test_processed = add_time_and_sector_fields(test_df, month_codes)
    pred_long = a_pred.stack().rename('new_house_transaction_amount').reset_index()
    # 在merge前确保sector_id类型一致
    pred_long['sector_id'] = pred_long['sector_id'].astype(int) 
    
    submission = pd.merge(test_processed[['id', 'time', 'sector_id']], pred_long, on=['time', 'sector_id'], how='left')
    submission['new_house_transaction_amount'] = submission['new_house_transaction_amount'].fillna(0.0)
    
    # 保证金额为非负
    submission['new_house_transaction_amount'] = submission['new_house_transaction_amount'].clip(lower=0)
    
    submission[['id', 'new_house_transaction_amount']].to_csv(filename, index=False)
    print(f"Successfully generated submission file: {filename}")

# 创建一个DataFrame来存储预测结果
a_pred_nn = pd.DataFrame(index=np.arange(67, 79), columns=a_tr.columns, dtype=float)
a_pred_nn.index.name = 'time'

# 循环对每个板块进行预测
current_data = a_tr.values.T.copy() # shape: (96, 67)
num_sectors = current_data.shape[0]

for t in tqdm(range(67, 79), desc="Predicting Horizon"):
    X_pred_list = []
    
    for sector_idx in range(num_sectors):
        sector_id = a_tr.columns[sector_idx]
        
        # 提取该板块最近LOOKBACK个月的数据
        sequence = current_data[sector_idx, -LOOKBACK:]
        
        # --- 这是关键的修改 ---
        # 使用对应板块的scaler进行归一化
        scaler = scalers[sector_id]
        norm_sequence = scaler.transform(sequence.reshape(-1, 1)).flatten()
        
        # 构建特征 (确保只有2个特征)
        features = []
        for j in range(LOOKBACK):
            time_step = t - LOOKBACK + j
            month_feature = (time_step % 12) / 11.0
            # 移除 sector_feature
            features.append([norm_sequence[j], month_feature]) 
        # --- 修改结束 ---
        
        X_pred_list.append(features)
        
    X_pred = np.array(X_pred_list)
    
    # 进行预测 (一次性预测所有板块)
    predictions_norm = model.predict(X_pred, verbose=0).flatten()
    
    # 反归一化
    predictions = []
    for sector_idx in range(num_sectors):
        sector_id = a_tr.columns[sector_idx]
        scaler = scalers[sector_id]
        pred_val = scaler.inverse_transform(predictions_norm[sector_idx].reshape(1, -1)).flatten()[0]
        predictions.append(pred_val)
    
    # 存储预测结果
    a_pred_nn.loc[t] = predictions
    
    # 更新数据，用于下一步预测 (滚动预测)
    predictions = np.array(predictions).reshape(-1, 1) # (96, 1)
    current_data = np.hstack([current_data, predictions])

print("Prediction matrix preview:\n", a_pred_nn)
create_submission_file(a_pred_nn, test_df, month_codes, 'submission.csv')

print("\n--- 任务完成！---")
print("你可以提交 'submission.csv' 文件了。")


sub = pd.read_csv("/kaggle/working/submission.csv")
sub




