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


import matplotlib.pyplot as plt
import seaborn as sns
# Load the datasets
train_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_sequences.csv')
train_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/train_labels.csv')
validation_sequences = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv')
validation_labels = pd.read_csv('/kaggle/input/stanford-rna-3d-folding/validation_labels.csv')
test_sequence = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")


train_sequences['sequence_length'] = train_sequences['sequence'].apply(len)
validation_sequences['sequence_length'] = validation_sequences['sequence'].apply(len)

plt.figure(figsize=(10, 6))
sns.histplot(train_sequences['sequence_length'], bins=50, kde=True, label='Train Sequences', color='blue')
sns.histplot(validation_sequences['sequence_length'], bins=50, kde=True, label='Validation Sequences', color='orange')
plt.title('Distribution of Sequence Lengths')
plt.xlabel('Sequence Length')
plt.ylabel('Frequency')
plt.legend()
plt.show()


from collections import Counter
def nucleotide_composition(sequence):
    return dict(Counter(sequence))

train_sequences['nucleotide_composition'] = train_sequences['sequence'].apply(nucleotide_composition)
validation_sequences['nucleotide_composition'] = validation_sequences['sequence'].apply(nucleotide_composition)

train_nucleotide_counts = pd.DataFrame(train_sequences['nucleotide_composition'].tolist()).fillna(0).sum()
validation_nucleotide_counts = pd.DataFrame(validation_sequences['nucleotide_composition'].tolist()).fillna(0).sum()

plt.figure(figsize=(10, 6))
train_nucleotide_counts.plot(kind='bar', color='blue', label='Train Sequences')
validation_nucleotide_counts.plot(kind='bar', color='orange', label='Validation Sequences', alpha=0.7)
plt.title('Nucleotide Composition')
plt.xlabel('Nucleotide')
plt.ylabel('Count')
plt.legend()
plt.show()


train_sequences['temporal_cutoff'] = pd.to_datetime(train_sequences['temporal_cutoff'])
validation_sequences['temporal_cutoff'] = pd.to_datetime(validation_sequences['temporal_cutoff'])

train_sequences['year'] = train_sequences['temporal_cutoff'].dt.year
validation_sequences['year'] = validation_sequences['temporal_cutoff'].dt.year

plt.figure(figsize=(10, 6))
sns.histplot(train_sequences['temporal_cutoff'], bins=50, kde=True, label='Train Sequences')
sns.histplot(validation_sequences['temporal_cutoff'], bins=50, kde=True, label='Validation Sequences', color='orange')
plt.title('Temporal Distribution of Sequences')
plt.xlabel('Temporal Cutoff')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# Analyze the distribution of 3D coordinates in train_labels
coordinate_columns = [col for col in train_labels.columns if col.startswith(('x_', 'y_', 'z_'))]
train_labels['num_structures'] = train_labels[coordinate_columns].count(axis=1) // 3

# Plot number of structures per target
plt.figure(figsize=(10, 6))
sns.histplot(train_labels['num_structures'], bins=20, kde=True)
plt.title('Distribution of Number of Structures per Target')
plt.xlabel('Number of Structures')
plt.ylabel('Frequency')
plt.show()

# Plot distribution of coordinates
plt.figure(figsize=(15, 5))
for i, coord in enumerate(['x_1', 'y_1', 'z_1']):
    plt.subplot(1, 3, i+1)
    sns.histplot(train_labels[coord].dropna(), bins=50, kde=True)
    plt.title(f'Distribution of {coord}')
    plt.xlabel(coord)
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()


duplicate_sequences = train_sequences[train_sequences.duplicated('sequence', keep=False)]
print(f"Number of duplicate sequences in train_sequences: {len(duplicate_sequences)}")

# Check for duplicate target_ids in train_labels
duplicate_targets = train_labels[train_labels.duplicated('ID', keep=False)]
print(f"Number of duplicate targets in train_labels: {len(duplicate_targets)}")


# Extract coordinates for a sample target
sample_target = train_labels
x = sample_target['x_1'].values
y = sample_target['y_1'].values
z = sample_target['z_1'].values

# Plot 3D structure
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='blue', marker='o')
ax.set_title('3D RNA Structure')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()



# 检查描述中是否提及配体
train_sequences['has_ligand'] = train_sequences['description'].str.contains('ligand', case=False)

# 比较配体结合与未结合序列的序列长度
plt.figure(figsize=(10, 6))
sns.boxplot(x='has_ligand', y='sequence_length', data=train_sequences)
plt.title('Sequence Length for Ligand-Bound vs. Unbound Sequences') #配体结合与未结合序列的长度
plt.xlabel('Has Ligand') #是否有配体
plt.ylabel('Sequence Length')   #序列长度
plt.show()


from sklearn.feature_extraction.text import CountVectorizer
# 将序列转换为k-mer计数
vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
kmer_counts = vectorizer.fit_transform(train_sequences['sequence'])
from sklearn.cluster import KMeans
# 执行k-means聚类
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(kmer_counts)

# 将聚类结果添加到数据框
train_sequences['cluster'] = clusters

# 绘制聚类分布图
plt.figure(figsize=(10, 6))
sns.countplot(x='cluster', data=train_sequences)
plt.title('Sequence Clusters') # 序列聚类分布
plt.xlabel('Cluster') #聚类
plt.ylabel('Count') # 计数
plt.show()


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
coordinate_data = train_labels[['x_1', 'y_1', 'z_1']].dropna().values

# 标准化坐标数据
scaler = StandardScaler()
scaled_coordinates = scaler.fit_transform(coordinate_data)

# 执行PCA降维
pca = PCA(n_components=2)
pca_result = pca.fit_transform(scaled_coordinates)

# 绘制前两个主成分
plt.figure(figsize=(10, 6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.scatter(pca_result[:, 0], pca_result[:, 1], alpha=0.5)
plt.title('PCA on 3D Coordinates of RNA Sequences') # RNA序列的3D坐标PCA
plt.xlabel('Principal Component 1')# 主成分1
plt.ylabel('Principal Component 2')
plt.colorbar(label='Target ID')
plt.show()


 # Check for missing values
print("Missing values in train_sequences:")
print(train_sequences.isnull().sum())

print("\nMissing values in train_labels:")
print(train_labels.isnull().sum())


# Load Data
train_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
train_sequence = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/train_sequences.csv")
val_labels = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
val_sequence = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv")
test_sequence = pd.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")

# Fill missing values
train_labels.fillna(0, inplace=True)
validation_labels.fillna(0, inplace=True)


# Sequence Encoding
seq_dict = {'A': 1, 'C': 2, 'G': 3, 'U': 4}
def seq_map(seq):
    return [seq_dict.get(char, 0) for char in seq]

train_sequence['encoded_seq'] = train_sequence['sequence'].apply(seq_map)
test_sequence['encoded_seq'] = test_sequence['sequence'].apply(seq_map)
val_sequence['encoded_seq'] = val_sequence['sequence'].apply(seq_map)


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Conv1D, BatchNormalization,
    Concatenate, UpSampling1D, MaxPooling1D, Layer, Dropout  
)
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import mean_squared_error
def generate_label_coord(df):
    result = {}
    df["label"] = df.ID.str.rsplit('_', n=1, expand=True).iloc[:,0]
    for _, row in df.iterrows():
        label = row['label']
        resid = row['resid']
        if label not in result:
            result[label] = []
        
        # 检查是否所有必需的列都存在
        if all(col in row for col in ['x_1', 'y_1', 'z_1']):
            # 如果只有 x_1, y_1, z_1 可用，则将它们复制为 x_2, y_2, z_2 等
            coords = np.array([
                [row['x_1'], row['y_1'], row['z_1']],
                [row['x_1'], row['y_1'], row['z_1']],  # 复制为 x_2, y_2, z_2
                [row['x_1'], row['y_1'], row['z_1']],  # 复制为 x_3, y_3, z_3
                [row['x_1'], row['y_1'], row['z_1']],  # 复制为 x_4, y_4, z_4
                [row['x_1'], row['y_1'], row['z_1']]   # 复制为 x_5, y_5, z_5
            ], dtype=np.float32)
        else:
            # 如果没有坐标可用，则使用零
            # coords = np.zeros(5, 3, dtype=np.float32)
            coords = np.zeros((5, 3), dtype=np.float32)
        result[label].append((resid, coords))
    
    for key in result:
        coords = np.stack([c for r, c in result[key]])
        result[key] = coords
    
    return result

train_stacked_coords = generate_label_coord(train_labels)
val_stacked_coords = generate_label_coord(val_labels)

def generate_dataset(seq, stacked_coords):
    X, y, tids = [], [], []
    for idx, row in seq.iterrows():
        tid = row['target_id']
        if tid in stacked_coords:
            X.append(row['encoded_seq'])
            y.append(stacked_coords[tid])
            tids.append(tid)
    return X, y, tids

train_X, train_y, train_tids = generate_dataset(train_sequence, train_stacked_coords)
val_X, val_y, val_tids = generate_dataset(val_sequence, val_stacked_coords)

# 填充序列和坐标
max_len = max(len(seq) for seq in train_X)
train_X_pad = pad_sequences(train_X, maxlen=max_len, padding='post', value=0)
val_X_pad = pad_sequences(val_X, maxlen=max_len, padding='post', value=0)
test_X = test_sequence['encoded_seq'].tolist()
test_X_pad = pad_sequences(test_X, maxlen=max_len, padding='post', value=0)

def pad_coords(coords, max_len):
    L = coords.shape[0]
    if L < max_len:
        pad_width = ((0, max_len-L), (0, 0), (0, 0))
        return np.pad(coords, pad_width, mode='constant', constant_values=0)
    else:
        return coords

train_y_pad = np.array([pad_coords(y, max_len) for y in train_y])
val_y_pad = np.array([pad_coords(y, max_len) for y in val_y])

# CNN模型
def build_cnn_model(max_len):
    input_seq = Input(shape=(max_len,), name='input_seq')
    x = Embedding(input_dim=5, output_dim=16, mask_zero=False, name='embedding')(input_seq)
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv1')(x)
    x = BatchNormalization(name='norm1')(x)
    x = Dropout(0.2, name='drop1')(x)
    x = Conv1D(filters=64, kernel_size=3, padding='same', activation='relu', name='conv2')(x)
    x = BatchNormalization(name='norm2')(x)
    x = Dropout(0.2, name='drop2')(x)
    # 每个残基输出15个值（5组x、y、z坐标）
    x = Conv1D(filters=15, kernel_size=1, padding='same', activation='linear', name='predicted_coords')(x)
    model = Model(inputs=input_seq, outputs=x)
    model.compile(optimizer='adam', loss='mae')
    return model

class BasicUNet(Layer):
    """1D版简化UNet替代原DiffusionLayer"""
    def __init__(self, filters=64, **kwargs):
        super().__init__(**kwargs)
        # 编码器
        self.conv1 = Conv1D(filters, 3, padding='same', activation='relu')
        self.pool1 = MaxPooling1D(2)
        self.conv2 = Conv1D(filters*2, 3, padding='same', activation='relu')
        
        # 解码器
        self.up1 = UpSampling1D(2)
        self.deconv1 = Conv1D(filters, 3, padding='same', activation='relu')
        
        # 跳跃连接
        self.concat = Concatenate(axis=-1)
        
        # 最终输出
        self.final_conv = Conv1D(filters, 1, activation='relu')

    def call(self, inputs):
        # 编码路径
        x1 = self.conv1(inputs)
        p1 = self.pool1(x1)
        x2 = self.conv2(p1)
        
        # 解码路径
        u1 = self.up1(x2)
        u1 = self.deconv1(u1)
        
        # 跳跃连接
        c1 = self.concat([u1, x1])
        
        # 最终输出
        return self.final_conv(c1)

def build_diffusion_model(max_len):
    input_seq = Input(shape=(max_len,), name='input_seq')
    
    # 嵌入层保持不变
    x = Embedding(input_dim=5, output_dim=16, mask_zero=False)(input_seq)
    
    # 用BasicUNet替换原DiffusionLayer
    x = BasicUNet(filters=64)(x)  # 主要修改点
    
    # 后续层保持原结构
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(15, activation='linear')(x)
    
    model = Model(inputs=input_seq, outputs=x)
    model.compile(optimizer='adam', loss='mae')
    return model

# 训练CNN
cnn_model = build_cnn_model(max_len)
cnn_history = cnn_model.fit(
    train_X_pad, train_y_pad.reshape(train_y_pad.shape[0], train_y_pad.shape[1], -1),
    validation_data=(val_X_pad, val_y_pad.reshape(val_y_pad.shape[0], val_y_pad.shape[1], -1)),
    epochs=50, batch_size=16, verbose=1
)


# 训练扩散模型
diffusion_model = build_diffusion_model(max_len)
diffusion_history = diffusion_model.fit(
    train_X_pad, train_y_pad.reshape(train_y_pad.shape[0], train_y_pad.shape[1], -1),
    validation_data=(val_X_pad, val_y_pad.reshape(val_y_pad.shape[0], val_y_pad.shape[1], -1)),
    epochs=50, batch_size=16, verbose=1
)

def evaluate_model(model, X, y):
    preds = model.predict(X)
    preds = preds.reshape(preds.shape[0], preds.shape[1], 5, 3)  # 重新形状为 [batch_size, seq_len, 5, 3]
    rmse = np.sqrt(mean_squared_error(y.reshape(-1), preds.reshape(-1)))
    return rmse

cnn_rmse = evaluate_model(cnn_model, val_X_pad, val_y_pad)
diffusion_rmse = evaluate_model(diffusion_model, val_X_pad, val_y_pad)
print(f"CNN验证集RMSE: {cnn_rmse}")
print(f"扩散模型验证集RMSE: {diffusion_rmse}")

cnn_preds = cnn_model.predict(test_X_pad)
diffusion_preds = diffusion_model.predict(test_X_pad)

# 集成预测（加权平均）
ensemble_preds = 0.7 * cnn_preds + 0.3 * diffusion_preds
ensemble_preds = ensemble_preds.reshape(ensemble_preds.shape[0], ensemble_preds.shape[1], 5, 3)


submission_rows = []
for idx, row in test_sequence.iterrows():
    target_id = row['target_id']
    coords = ensemble_preds[idx]  # Shape: [sequence_length, 5, 3]
    seq_length = len(row['encoded_seq'])
    coords = coords[:seq_length, :, :]  # Trim to the actual sequence length
    for i in range(seq_length):
        x_coords = coords[i, :, 0]  # x_1, x_2, x_3, x_4, x_5
        y_coords = coords[i, :, 1]  # y_1, y_2, y_3, y_4, y_5
        z_coords = coords[i, :, 2]  # z_1, z_2, z_3, z_4, z_5
        submission_rows.append({
            'ID': f"{target_id}_{i+1}",
            'resname': row['sequence'][i],
            'resid': i+1,
            'x_1': x_coords[0], 'x_2': x_coords[1], 'x_3': x_coords[2], 'x_4': x_coords[3], 'x_5': x_coords[4],
            'y_1': y_coords[0], 'y_2': y_coords[1], 'y_3': y_coords[2], 'y_4': y_coords[3], 'y_5': y_coords[4],
            'z_1': z_coords[0], 'z_2': z_coords[1], 'z_3': z_coords[2], 'z_4': z_coords[3], 'z_5': z_coords[4]})

submission = pd.DataFrame(submission_rows)
submission.to_csv("submission.csv", index=False)
print("Submission file created: submission.csv") 

