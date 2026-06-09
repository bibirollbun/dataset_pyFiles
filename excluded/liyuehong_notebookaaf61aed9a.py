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



import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv1D, LSTM, GlobalAveragePooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split


import pandas as pd
from datetime import datetime

# 1. 加载数据（注意列名修正）
train_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")

# 2. 从labels的ID列提取target_id（假设ID格式为"T1001_chainA"）
train_labels['target_id'] = train_labels['ID'].str.split('_').str[0]  # 关键修正！

# 3. 时间字段转换
train_df['temporal_cutoff'] = pd.to_datetime(train_df['temporal_cutoff'])

# 4. 按时间划分训练/验证集（示例：以2023-06-01为分界线）
cutoff_date = datetime(2023, 6, 1)
phase1_train = train_df[train_df['temporal_cutoff'] < cutoff_date]
phase1_valid = train_df[train_df['temporal_cutoff'] >= cutoff_date]

# 5. 关联标签数据（使用提取后的target_id）
phase1_train_labels = train_labels[train_labels['target_id'].isin(phase1_train['target_id'])]
phase1_valid_labels = train_labels[train_labels['target_id'].isin(phase1_valid['target_id'])]

print("训练集样本数:", len(phase1_train))
print("验证集样本数:", len(phase1_valid))


import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# 1. 数据预处理函数
def encode_sequence(sequence, max_len=200):
    # 在 encoding 字典中添加对 'X' 和 '-' 字符的编码
    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1], '-': [0, 0, 0, 0], 'X': [0, 0, 0, 0]}
    encoded = np.array([encoding[base] for base in sequence])
    if len(encoded) < max_len:
        encoded = np.pad(encoded, ((0, max_len - len(encoded)), (0, 0)), 'constant')
    return encoded[:max_len]

# 假设的 process_labels 函数，需要根据实际情况修改
def process_labels(train_labels, target_id):
    # 这里简单返回一个随机的形状为 (200, 3) 的数组，实际需要根据标签数据进行处理
    return np.random.rand(200, 3)

# 2. 模型构建
def build_model(input_dim=4, max_len=200):
    inputs = Input(shape=(max_len, input_dim))

    x = Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = Conv1D(256, 3, activation='relu', padding='same')(x)

    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256, return_sequences=True)(x)

    outputs = Dense(3, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model

# 3. 训练函数
def train_model(train_sequences, train_labels, epochs=50):
    X_train = np.array([encode_sequence(seq) for seq in train_sequences])
    y_train = np.array([process_labels(train_labels, tid) for tid in train_sequences.index])

    model = build_model()

    # 保存最佳模型，修改文件后缀为 .keras
    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=16,
              callbacks=[checkpoint])

    return model

# 4. 预测函数
def predict_structures(model, sequence, max_len=200, num_models=5):
    encoded = encode_sequence(sequence, max_len)
    X = np.expand_dims(encoded, axis=0)

    predictions = []
    for _ in range(num_models):
        tf.random.set_seed(_)
        pred = model.predict(X)  # 形状：(1, max_len, 3)
        predictions.append(pred)

    combined = np.concatenate(predictions, axis=2)  # 形状：(1, max_len, 15)
    return combined.reshape(1, max_len, 5, 3)  # 形状：(1, max_len, 5, 3)

# 5. 生成提交文件
def create_submission(test_sequences, predictions, output_file='submission.csv'):
    rows = []
    for idx, (target_id, seq) in enumerate(zip(test_sequences['target_id'], test_sequences['sequence'])):
        residues = min(len(seq), 200)  # 确保不会超出预测数组的长度
        for res in range(residues):
            row = {
                'ID': target_id,
                'resname': seq[res],
                'resid': res + 1
            }
            for m in range(5):
                row[f'x_{m + 1}'] = predictions[idx][0, res, m, 0]
                row[f'y_{m + 1}'] = predictions[idx][0, res, m, 1]
                row[f'z_{m + 1}'] = predictions[idx][0, res, m, 2]
            rows.append(row)

    submission_df = pd.DataFrame(rows)
    submission_df.to_csv(output_file, index=False)

# 主程序入口
if __name__ == "__main__":
    # 加载数据（示例路径）
    train_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
    train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")

    # 数据关联（假设ID格式为"T1001_chainA"）
    train_labels['target_id'] = train_labels['ID'].str.split('_').str[0]

    # 简单训练（示例使用全部数据，实际应划分验证集）
    model = train_model(train_df['sequence'], train_labels)

    # 加载测试数据
    test_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")

    # 生成预测
    predictions = []
    for seq in test_df['sequence']:
        pred = predict_structures(model, seq)
        predictions.append(pred)

    # 创建提交文件
    create_submission(test_df, predictions)
    print("提交文件已生成：submission.csv")



import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

# 1. 数据预处理函数，动态处理序列长度，容错处理未知字符，并进行填充
def encode_sequence(sequence, max_len=None):
    encoding = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'U': [0, 0, 0, 1]}
    encoded = np.array([encoding.get(base, [0, 0, 0, 0]) for base in sequence])
    if max_len is not None:
        if len(encoded) < max_len:
            encoded = np.pad(encoded, ((0, max_len - len(encoded)), (0, 0)), 'constant')
        elif len(encoded) > max_len:
            encoded = encoded[:max_len]
    return encoded

# 假设的 process_labels 函数，需要根据实际情况修改
def process_labels(train_df, train_labels, target_id):
    # 从 train_df 中获取序列信息
    seq = train_df[train_df.index == target_id]['sequence'].values[0]
    seq_length = len(seq)
    return np.random.rand(seq_length, 3)

# 2. 模型构建
def build_model(input_dim=4):
    inputs = Input(shape=(None, input_dim))

    x = Conv1D(128, 3, activation='relu', padding='same')(inputs)
    x = Conv1D(256, 3, activation='relu', padding='same')(x)

    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(256, return_sequences=True)(x)

    outputs = Dense(3, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=1e-4), loss='mse')
    return model

# 3. 训练函数
def train_model(train_df, train_sequences, train_labels, epochs=50):
    max_len = max([len(seq) for seq in train_sequences])
    X_train = np.array([encode_sequence(seq, max_len) for seq in train_sequences])
    y_train = []
    for tid in train_sequences.index:
        label = process_labels(train_df, train_labels, tid)
        if len(label) < max_len:
            label = np.pad(label, ((0, max_len - len(label)), (0, 0)), 'constant')
        elif len(label) > max_len:
            label = label[:max_len]
        y_train.append(label)
    y_train = np.array(y_train)

    model = build_model()

    # 保存最佳模型，修改文件后缀为 .keras
    checkpoint = ModelCheckpoint(
        'best_model.keras',
        monitor='loss',
        verbose=1,
        save_best_only=True,
        mode='min'
    )

    model.fit(X_train, y_train,
              epochs=epochs,
              batch_size=16,
              callbacks=[checkpoint])

    return model

# 4. 预测函数，支持任意长度序列
def predict_structures(model, sequence, num_models=5):
    encoded = encode_sequence(sequence)
    X = np.expand_dims(encoded, axis=0)

    predictions = []
    for _ in range(num_models):
        tf.random.set_seed(_)
        pred = model.predict(X)  # 形状：(1, 序列长度, 3)
        predictions.append(pred)

    combined = np.concatenate(predictions, axis=2)  # 形状：(1, 序列长度, 15)
    return combined.reshape(1, len(sequence), 5, 3)

# 5. 生成提交文件，确保格式正确
def create_submission(test_sequences, predictions, output_file='submission.csv'):
    rows = []
    for idx, (target_id, seq) in enumerate(zip(test_sequences['target_id'], test_sequences['sequence'])):
        residues = len(seq)
        for res in range(residues):
            row = {
                'ID': target_id,
                'resname': seq[res],
                'resid': res + 1
            }
            for m in range(5):
                row[f'x_{m + 1}'] = predictions[idx][0, res, m, 0]
                row[f'y_{m + 1}'] = predictions[idx][0, res, m, 1]
                row[f'z_{m + 1}'] = predictions[idx][0, res, m, 2]
            rows.append(row)

    submission_df = pd.DataFrame(rows)
    submission_df = submission_df.round(4)  # 确保坐标精度为4位小数
    submission_df.to_csv(output_file, index=False)

# 主程序入口
if __name__ == "__main__":
    # 加载数据（示例路径）
    train_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
    train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")

    # 数据关联（假设ID格式为"T1001_chainA"）
    train_labels['target_id'] = train_labels['ID'].str.split('_').str[0]

    # 简单训练（示例使用全部数据，实际应划分验证集）
    model = train_model(train_df, train_df['sequence'], train_labels)

    # 加载测试数据
    test_df = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")

    # 生成预测
    predictions = []
    for seq in test_df['sequence']:
        pred = predict_structures(model, seq)
        predictions.append(pred)

    # 创建提交文件
    create_submission(test_df, predictions)
    print("提交文件已生成：submission.csv")

