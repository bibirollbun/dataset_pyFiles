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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子，保证结果可复现
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 检查GPU配置
print(f"CUDA可用: {torch.cuda.is_available()}")
print(f"GPU数量: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"当前GPU: {torch.cuda.get_device_name(0)}")
    if torch.cuda.device_count() > 1:
        print(f"第二GPU: {torch.cuda.get_device_name(1)}")

# 定义设备，优先使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ----------------------
# 1. 数据加载与详细探索
# ----------------------
def load_and_explore_data():
    """加载数据并进行详细探索，为后续处理提供依据"""
    print("加载数据...")
    try:
        train = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
        test = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
        train_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
        test_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
        print("尝试使用备选路径...")
        train = pd.read_csv('train.csv')
        test = pd.read_csv('test.csv')
        train_demo = pd.read_csv('train_demographics.csv')
        test_demo = pd.read_csv('test_demographics.csv')
    
    # 详细数据探索
    print("\n===== 数据探索 =====")
    print(f"训练集形状: {train.shape}")
    print(f"测试集形状: {test.shape}")
    
    print("\n训练集列名及数据类型:")
    print(train.dtypes)
    print("\n测试集列名及数据类型:")
    print(test.dtypes)
    
    # 检查特征差异
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    train_only_cols = train_cols - test_cols
    test_only_cols = test_cols - train_cols
    print(f"\n训练集特有列: {len(train_only_cols)}个 - {sorted(train_only_cols)[:5]}...")
    print(f"测试集特有列: {len(test_only_cols)}个 - {sorted(test_only_cols)[:5]}...")
    
    # 检查关键列是否存在
    required_columns = ['sequence_id', 'row_id']
    for col in required_columns:
        if col not in train.columns:
            print(f"警告: 训练集缺少必要列 '{col}'")
        if col not in test.columns:
            print(f"警告: 测试集缺少必要列 '{col}'")
    
    # 检查是否有共同的键用于合并人口统计学数据
    merge_key = 'subject' if 'subject' in train.columns and 'subject' in train_demo.columns else None
    if merge_key:
        print(f"\n将使用 '{merge_key}' 合并主数据和人口统计学数据")
    else:
        print("\n警告: 未找到合适的合并键，将不合并人口统计学数据")
    
    return train, test, train_demo, test_demo


# ----------------------
# 2. 数据预处理
# ----------------------
def preprocess_data(train, test, train_demo, test_demo):
    """数据预处理：灵活处理不同结构的数据"""
    print("\n===== 数据预处理 =====")
    
    # 尝试合并人口统计学数据
    merge_key = 'subject' if 'subject' in train.columns and 'subject' in train_demo.columns else None
    if merge_key:
        print(f"使用 '{merge_key}' 合并数据...")
        train = train.merge(train_demo, on=merge_key, how='left')
        test = test.merge(test_demo, on=merge_key, how='left')
    else:
        print("无法合并人口统计学数据，跳过此步骤")
    
    # 确保训练集和测试集有相同的特征列
    train_cols = set(train.columns)
    test_cols = set(test.columns)
    
    # 找出双方共有的列
    common_cols = train_cols.intersection(test_cols)
    
    # 训练集特有列（如目标列）只保留在训练集中
    train_only_cols = ['gesture', 'gesture_encoded', 'sequence_type']
    train_only_cols = [col for col in train_only_cols if col in train_cols]
    print(f"仅在训练集中保留的列: {train_only_cols}")
    
    # 为训练集添加特有列
    train_features = list(common_cols) + train_only_cols
    
    # 测试集只使用共同列
    test_features = list(common_cols)
    
    # 确保必要的ID列被包含
    for col in ['sequence_id', 'row_id', 'subject']:
        if col in train_cols and col not in train_features:
            train_features.append(col)
        if col in test_cols and col not in test_features:
            test_features.append(col)
    
    # 应用特征选择
    train = train[train_features]
    test = test[test_features]
    
    print(f"训练集特征列数量: {len(train_features)}")
    print(f"测试集特征列数量: {len(test_features)}")
    
    # 检查并处理缺失值
    print(f"训练集缺失值比例: {train.isnull().mean().mean():.4f}")
    print(f"测试集缺失值比例: {test.isnull().mean().mean():.4f}")
    
    # 识别传感器类型列
    imu_cols = [col for col in train.columns if col.startswith(('acc_', 'rot_', 'gyro_', 'mag_'))]
    thm_cols = [col for col in train.columns if col.startswith(('thm_', 'temp_', 'thermal_'))]
    tof_cols = [col for col in train.columns if col.startswith(('tof_', 'distance_'))]
    
    print(f"找到 {len(imu_cols)} 个IMU传感器列")
    print(f"找到 {len(thm_cols)} 个热传感器列")
    print(f"找到 {len(tof_cols)} 个飞行时间传感器列")
    
    # 对不同类型的传感器使用不同的缺失值填充策略
    if imu_cols:
        train[imu_cols] = train.groupby('sequence_id')[imu_cols].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both'))
        test[imu_cols] = test.groupby('sequence_id')[imu_cols].transform(
            lambda x: x.interpolate(method='linear', limit_direction='both'))
    
    if thm_cols:
        train[thm_cols] = train.groupby('sequence_id')[thm_cols].transform(
            lambda x: x.fillna(x.mean()))
        test[thm_cols] = test.groupby('sequence_id')[thm_cols].transform(
            lambda x: x.fillna(x.mean()))
    
    if tof_cols:
        train[tof_cols] = train.groupby('sequence_id')[tof_cols].transform(
            lambda x: x.fillna(x.mean()))
        test[tof_cols] = test.groupby('sequence_id')[tof_cols].transform(
            lambda x: x.fillna(x.mean()))
    
    # 人口统计学数据处理
    demo_cols = ['age', 'height', 'height_cm', 'weight', 'shoulder_to_wrist', 
                'shoulder_to_wrist_cm', 'elbow_to_wrist', 'elbow_to_wrist_cm']
    demo_cols = [col for col in demo_cols if col in train.columns]
    print(f"找到 {len(demo_cols)} 个人口统计学列: {demo_cols}")
    
    if demo_cols:
        train[demo_cols] = train[demo_cols].fillna(train[demo_cols].median())
        test[demo_cols] = test[demo_cols].fillna(test[demo_cols].median())
    
    # 编码目标变量
    if 'gesture' not in train.columns:
        raise ValueError("训练集必须包含'gesture'列作为目标变量")
    
    gesture_le = LabelEncoder()
    train['gesture_encoded'] = gesture_le.fit_transform(train['gesture'])
    print(f"已编码 {len(gesture_le.classes_)} 种手势类别")
    
    return train, test, imu_cols, thm_cols, tof_cols, demo_cols, gesture_le


# ----------------------
# 修复：时序统计特征提取函数（处理NaN和非数值问题）
# ----------------------
def extract_temporal_stats(group, feature_cols):
    """为每个序列提取时序统计特征，确保输出为纯标量数值"""
    stats = {}
    # 基础统计量
    stats['mean'] = group[feature_cols].mean()
    stats['std'] = group[feature_cols].std()
    stats['max'] = group[feature_cols].max()
    stats['min'] = group[feature_cols].min()
    stats['median'] = group[feature_cols].median()
    stats['ptp'] = group[feature_cols].max() - group[feature_cols].min()  # 峰峰值
    stats['skew'] = group[feature_cols].skew()  # 偏度
    stats['kurt'] = group[feature_cols].kurt()  # 峰度
    
    # 自相关系数（滞后1步，捕捉短期依赖）
    for col in feature_cols:
        try:
            # 处理可能的自相关计算错误
            autocorr = group[col].autocorr(lag=1)
            # 检查是否为NaN或无穷大
            if pd.isna(autocorr) or not np.isfinite(autocorr):
                autocorr = 0.0  # 用0替代无效值
            stats[f'{col}_autocorr_1'] = autocorr
        except:
            stats[f'{col}_autocorr_1'] = 0.0  # 出错时用0替代
    
    # 拼接为DataFrame（每行对应一个序列的统计特征）
    stats_df = pd.DataFrame([stats])
    
    # 修复：确保所有值都是数值类型，并填充任何NaN
    for col in stats_df.columns:
        # 强制转换为数值类型
        stats_df[col] = pd.to_numeric(stats_df[col], errors='coerce')
        # 填充任何剩余的NaN值
        stats_df[col] = stats_df[col].fillna(0.0)
    
    # 重命名列（避免与原始特征冲突）
    stats_df.columns = [f'{col}_stat' for col in stats_df.columns]
    return stats_df

# ----------------------
# 修复：prepare_sequences函数（确保特征维度正确）
# ----------------------
def prepare_sequences(train, test, max_seq_length=100):
    print("\n===== 准备序列数据（含时序统计特征） =====")
    
    # 1. 保留原逻辑：获取共同特征列
    if 'sequence_id' not in train.columns or 'sequence_id' not in test.columns:
        raise ValueError("数据必须包含'sequence_id'列以区分不同序列")
    
    train_exclude_cols = ['sequence_id', 'row_id', 'gesture', 'gesture_encoded', 'sequence_type', 'subject']
    test_exclude_cols = ['sequence_id', 'row_id', 'subject']
    train_exclude_cols = [col for col in train_exclude_cols if col in train.columns]
    test_exclude_cols = [col for col in test_exclude_cols if col in test.columns]
    
    train_feature_cols = [col for col in train.columns if col not in train_exclude_cols]
    test_feature_cols = [col for col in test.columns if col not in test_exclude_cols]
    common_feature_cols = list(set(train_feature_cols).intersection(set(test_feature_cols)))
    common_feature_cols.sort()
    print(f"共同特征列: {len(common_feature_cols)}")
    
    if len(common_feature_cols) == 0:
        raise ValueError("未找到共同的特征列，无法继续")
    
    # 2. 提取序列级统计特征（修复版本）
    print("提取时序统计特征...")
    # 训练集统计特征
    try:
        train_stats = train.groupby('sequence_id').apply(
            lambda x: extract_temporal_stats(x, common_feature_cols)
        ).reset_index(drop=True)
        # 确保索引正确对应
        unique_train_ids = train['sequence_id'].unique()
        if len(train_stats) != len(unique_train_ids):
            print(f"警告：训练集统计特征数量({len(train_stats)})与序列数量({len(unique_train_ids)})不匹配，重新对齐")
            train_stats = train_stats.iloc[:len(unique_train_ids)]
        train_stats.index = unique_train_ids
    except Exception as e:
        print(f"训练集统计特征提取出错: {e}")
        # 创建默认的统计特征
        train_stats = pd.DataFrame(0.0, index=train['sequence_id'].unique(), 
                                  columns=[f'feat_{i}_stat' for i in range(10)])
    
    # 测试集统计特征
    try:
        test_stats = test.groupby('sequence_id').apply(
            lambda x: extract_temporal_stats(x, common_feature_cols)
        ).reset_index(drop=True)
        # 确保索引正确对应
        unique_test_ids = test['sequence_id'].unique()
        if len(test_stats) != len(unique_test_ids):
            print(f"警告：测试集统计特征数量({len(test_stats)})与序列数量({len(unique_test_ids)})不匹配，重新对齐")
            test_stats = test_stats.iloc[:len(unique_test_ids)]
        test_stats.index = unique_test_ids
    except Exception as e:
        print(f"测试集统计特征提取出错: {e}")
        # 创建默认的统计特征
        test_stats = pd.DataFrame(0.0, index=test['sequence_id'].unique(), 
                                 columns=[f'feat_{i}_stat' for i in range(10)])
    
    # 3. 处理训练序列（拼接统计特征）
    train_sequences = []
    train_gestures = []
    train_binary = []
    train_ids = []
    
    print("处理训练序列...")
    for seq_id, group in tqdm(train.groupby('sequence_id'), desc="处理训练序列"):
        try:
            # 原始时序特征
            seq_data = group[common_feature_cols].copy()
            # 填充/转换数值类型
            for col in seq_data.columns:
                try:
                    seq_data[col] = pd.to_numeric(seq_data[col], errors='coerce').fillna(seq_data[col].median())
                except:
                    le = LabelEncoder()
                    seq_data[col] = le.fit_transform(seq_data[col].astype(str))
            
            # 获取当前序列的统计特征
            if seq_id in train_stats.index:
                seq_stat = train_stats.loc[seq_id].values.reshape(1, -1)  # (1, n_stats)
            else:
                # 如果找不到对应的统计特征，使用0填充
                seq_stat = np.zeros((1, train_stats.shape[1]))
                print(f"警告：序列 {seq_id} 未找到统计特征，使用默认值")
            
            # 确保统计特征是有效的数值数组
            seq_stat = np.nan_to_num(seq_stat.astype(np.float32))
            
            # 广播到每个时间步
            seq_stat_broadcast = np.repeat(seq_stat, len(seq_data), axis=0)  # (seq_len, n_stats)
            
            # 确保原始特征是有效的数值数组
            seq_data_values = np.nan_to_num(seq_data.values.astype(np.float32))
            
            # 拼接特征
            seq_values = np.hstack([seq_data_values, seq_stat_broadcast])
            
            # 填充/截断到固定长度
            if len(seq_values) < max_seq_length:
                pad_length = max_seq_length - len(seq_values)
                seq_values = np.pad(seq_values, ((0, pad_length), (0, 0)), mode='edge')
            else:
                seq_values = seq_values[:max_seq_length]
                
            train_sequences.append(seq_values)
            train_gestures.append(group['gesture_encoded'].iloc[0])
            train_binary.append(1 if 'sequence_type' in group.columns and group['sequence_type'].iloc[0] == 'target' else 0)
            train_ids.append(seq_id)
        except Exception as e:
            print(f"处理训练序列 {seq_id} 时出错: {e}")
            # 创建一个默认序列避免中断
            default_seq = np.zeros((max_seq_length, len(common_feature_cols) + train_stats.shape[1]), dtype=np.float32)
            train_sequences.append(default_seq)
            train_gestures.append(0)
            train_binary.append(0)
            train_ids.append(seq_id)
    
    # 4. 处理测试序列
    test_sequences = []
    test_ids = []
    
    print("处理测试序列...")
    for seq_id, group in tqdm(test.groupby('sequence_id'), desc="处理测试序列"):
        try:
            seq_data = group[common_feature_cols].copy()
            for col in seq_data.columns:
                try:
                    seq_data[col] = pd.to_numeric(seq_data[col], errors='coerce').fillna(seq_data[col].median())
                except:
                    le = LabelEncoder()
                    seq_data[col] = le.fit_transform(seq_data[col].astype(str))
            
            # 获取当前序列的统计特征
            if seq_id in test_stats.index:
                seq_stat = test_stats.loc[seq_id].values.reshape(1, -1)  # (1, n_stats)
            else:
                seq_stat = np.zeros((1, test_stats.shape[1]))
                print(f"警告：测试序列 {seq_id} 未找到统计特征，使用默认值")
            
            # 确保统计特征是有效的数值数组
            seq_stat = np.nan_to_num(seq_stat.astype(np.float32))
            
            # 广播到每个时间步
            seq_stat_broadcast = np.repeat(seq_stat, len(seq_data), axis=0)  # (seq_len, n_stats)
            
            # 确保原始特征是有效的数值数组
            seq_data_values = np.nan_to_num(seq_data.values.astype(np.float32))
            
            # 拼接特征
            seq_values = np.hstack([seq_data_values, seq_stat_broadcast])
            
            # 填充/截断
            if len(seq_values) < max_seq_length:
                pad_length = max_seq_length - len(seq_values)
                seq_values = np.pad(seq_values, ((0, pad_length), (0, 0)), mode='edge')
            else:
                seq_values = seq_values[:max_seq_length]
                
            test_sequences.append(seq_values)
            test_ids.append(seq_id)
        except Exception as e:
            print(f"处理测试序列 {seq_id} 时出错: {e}")
            # 创建一个默认序列避免中断
            default_seq = np.zeros((max_seq_length, len(common_feature_cols) + test_stats.shape[1]), dtype=np.float32)
            test_sequences.append(default_seq)
            test_ids.append(seq_id)
    
    # 5. 后续处理
    train_sequences = np.array(train_sequences, dtype=np.float32)
    train_gestures = np.array(train_gestures)
    train_binary = np.array(train_binary)
    test_sequences = np.array(test_sequences, dtype=np.float32)
    
    print(f"训练序列形状: {train_sequences.shape}")
    print(f"测试序列形状: {test_sequences.shape}")
    
    # 特征标准化
    if len(train_sequences) > 0 and train_sequences.ndim == 3 and train_sequences.shape[2] > 0:
        scaler = StandardScaler()
        n_samples, n_timesteps, n_features = train_sequences.shape
        train_reshaped = train_sequences.reshape(n_samples * n_timesteps, n_features)
        train_reshaped = scaler.fit_transform(train_reshaped)
        train_sequences = train_reshaped.reshape(n_samples, n_timesteps, n_features)
        
        if len(test_sequences) > 0 and test_sequences.ndim == 3 and test_sequences.shape[2] > 0:
            n_samples_test, n_timesteps_test, n_features_test = test_sequences.shape
            test_reshaped = test_sequences.reshape(n_samples_test * n_timesteps_test, n_features_test)
            test_reshaped = scaler.transform(test_reshaped)
            test_sequences = test_reshaped.reshape(n_samples_test, n_timesteps_test, n_features_test)
    
    n_features = train_sequences.shape[2] if (len(train_sequences.shape) > 2 and train_sequences.shape[2] > 0) else 0
    print(f"最终特征数量（原始+统计）: {n_features}")
    
    return (train_sequences, train_gestures, train_binary, train_ids,
            test_sequences, test_ids, n_features)



# ----------------------
# 修改：SensorSequenceDataset（加入数据增强）
# ----------------------
class SensorSequenceDataset(Dataset):
    def __init__(self, sequences, gestures=None, binary_labels=None, is_train=False, aug_prob=0.5):
        self.sequences = sequences
        self.gestures = gestures
        self.binary_labels = binary_labels
        self.is_train = is_train  # 标记是否为训练集（仅训练时增强）
        self.aug_prob = aug_prob  # 增强概率
    
    def __len__(self):
        return len(self.sequences)
    
    def _augment_sequence(self, seq):
        """时序数据增强逻辑"""
        seq = seq.copy()
        n_timesteps, n_features = seq.shape
        
        # 1. 高斯噪声注入（概率50%，噪声强度0.05倍标准差）
        if np.random.random() < self.aug_prob:
            noise = np.random.normal(0, 0.05 * seq.std(axis=0), seq.shape)
            seq += noise
        
        # 2. 时间轴拉伸/压缩（概率40%，缩放比例0.8-1.2）
        if np.random.random() < self.aug_prob * 0.8:
            scale = np.random.uniform(0.8, 1.2)
            new_timesteps = int(n_timesteps * scale)
            # 线性插值调整长度
            seq_stretched = np.zeros((new_timesteps, n_features))
            for i in range(n_features):
                seq_stretched[:, i] = np.interp(
                    np.linspace(0, 1, new_timesteps),
                    np.linspace(0, 1, n_timesteps),
                    seq[:, i]
                )
            # 恢复原长度（截断/填充）
            if new_timesteps > n_timesteps:
                seq = seq_stretched[:n_timesteps]
            else:
                seq = np.pad(seq_stretched, ((0, n_timesteps - new_timesteps), (0, 0)), mode='edge')
        
        # 3. 常数偏移（概率30%，偏移量±0.1倍均值）
        if np.random.random() < self.aug_prob * 0.6:
            offset = np.random.uniform(-0.1, 0.1) * seq.mean(axis=0)
            seq += offset
        
        return seq.astype(np.float32)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        # 训练时应用增强
        if self.is_train:
            sequence = self._augment_sequence(sequence)
        
        sequence = torch.FloatTensor(sequence)
        
        if self.gestures is not None and self.binary_labels is not None:
            gesture = torch.LongTensor([self.gestures[idx]])[0]
            binary_label = torch.FloatTensor([self.binary_labels[idx]])[0]
            return sequence, gesture, binary_label
        else:
            return sequence


# ----------------------
# 替换：CNN+BiLSTM+Self-Attention模型
# ----------------------
class CNNBiLSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, num_gestures, dropout=0.3):
        super(CNNBiLSTMAttentionModel, self).__init__()
        
        # 1. 1D CNN层（提取局部特征）
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim//2, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim//2),  # BatchNorm加速训练
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(in_channels=hidden_dim//2, out_channels=hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. 双向LSTM层（输入维度=CNN输出维度）
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # 3. Self-Attention层（关注关键时间步）
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # 输入：BiLSTM输出（2*hidden_dim）
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)  # 对时间步加权
        )
        
        # 4. 全连接层 - 手势分类
        self.fc_gesture = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_gestures)
        )
        
        # 5. 全连接层 - 二元分类
        self.fc_binary = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        batch_size = x.size(0)
        
        # 1. CNN前向（需转置：(batch, input_dim, seq_len) → 符合Conv1d输入格式）
        x_cnn = x.permute(0, 2, 1)  # (batch, input_dim, seq_len)
        x_cnn = self.cnn(x_cnn)      # (batch, hidden_dim, seq_len)
        x_cnn = x_cnn.permute(0, 2, 1)  # 转回：(batch, seq_len, hidden_dim)
        
        # 2. BiLSTM前向
        lstm_out, _ = self.lstm(x_cnn)  # (batch, seq_len, 2*hidden_dim)
        
        # 3. Self-Attention：对每个时间步加权
        attn_weights = self.attention(lstm_out)  # (batch, seq_len, 1)
        attn_weights = attn_weights.repeat(1, 1, lstm_out.size(-1))  # (batch, seq_len, 2*hidden_dim)
        attn_out = torch.mul(lstm_out, attn_weights).sum(dim=1)  # (batch, 2*hidden_dim) → 加权求和
        
        # 4. 分类输出
        gesture_out = self.fc_gesture(attn_out)  # 手势分类
        binary_out = self.fc_binary(attn_out).squeeze(1)  # 二元分类
        
        return gesture_out, binary_out


# ----------------------
# 6. 训练和验证函数
# ----------------------
def train_epoch(model, train_loader, criterion_gesture, criterion_binary, optimizer, device):
    model.train()
    total_loss = 0
    total_gesture_correct = 0
    total_binary_correct = 0
    total_samples = 0
    
    for sequences, gestures, binary_labels in tqdm(train_loader, desc="训练"):
        sequences = sequences.to(device)
        gestures = gestures.to(device)
        binary_labels = binary_labels.to(device)
        
        optimizer.zero_grad()
        gesture_out, binary_logits = model(sequences)  # 接收logits而非概率
        
        # 计算损失（BCEWithLogitsLoss会自动应用sigmoid）
        loss_gesture = criterion_gesture(gesture_out, gestures)
        loss_binary = criterion_binary(binary_logits, binary_labels)  # 直接使用logits
        loss = loss_gesture + loss_binary
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * sequences.size(0)
        _, predicted_gesture = torch.max(gesture_out.data, 1)
        total_gesture_correct += (predicted_gesture == gestures).sum().item()
        
        # 预测时才应用sigmoid和阈值
        predicted_binary = (torch.sigmoid(binary_logits) > 0.5).float()
        total_binary_correct += (predicted_binary == binary_labels).sum().item()
        
        total_samples += sequences.size(0)
    
    avg_loss = total_loss / total_samples
    gesture_acc = total_gesture_correct / total_samples
    binary_acc = total_binary_correct / total_samples
    
    return avg_loss, gesture_acc, binary_acc

def validate(model, val_loader, criterion_gesture, criterion_binary, device):
    model.eval()
    total_loss = 0
    total_gesture_correct = 0
    total_binary_correct = 0
    total_samples = 0
    
    all_gesture_preds = []
    all_gesture_labels = []
    all_binary_preds = []
    all_binary_labels = []
    
    with torch.no_grad():
        for sequences, gestures, binary_labels in tqdm(val_loader, desc="验证"):
            sequences = sequences.to(device)
            gestures = gestures.to(device)
            binary_labels = binary_labels.to(device)
            
            gesture_out, binary_out = model(sequences)
            
            loss_gesture = criterion_gesture(gesture_out, gestures)
            loss_binary = criterion_binary(binary_out, binary_labels)
            loss = loss_gesture + loss_binary
            
            total_loss += loss.item() * sequences.size(0)
            _, predicted_gesture = torch.max(gesture_out.data, 1)
            total_gesture_correct += (predicted_gesture == gestures).sum().item()
            
            predicted_binary = (binary_out > 0.5).float()
            total_binary_correct += (predicted_binary == binary_labels).sum().item()
            
            total_samples += sequences.size(0)
            
            all_gesture_preds.extend(predicted_gesture.cpu().numpy())
            all_gesture_labels.extend(gestures.cpu().numpy())
            all_binary_preds.extend(predicted_binary.cpu().numpy())
            all_binary_labels.extend(binary_labels.cpu().numpy())
    
    avg_loss = total_loss / total_samples
    gesture_acc = total_gesture_correct / total_samples
    binary_acc = total_binary_correct / total_samples
    
    macro_f1 = f1_score(all_gesture_labels, all_gesture_preds, average='macro')
    binary_f1 = f1_score(all_binary_labels, all_binary_preds)
    avg_f1 = (macro_f1 + binary_f1) / 2
    
    return avg_loss, gesture_acc, binary_acc, macro_f1, binary_f1, avg_f1, \
           all_gesture_preds, all_gesture_labels, all_binary_preds, all_binary_labels


# ----------------------
# 新增：LabelSmoothingCrossEntropyLoss
# ----------------------
class LabelSmoothingCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.1, weight=None):
        super(LabelSmoothingCrossEntropyLoss, self).__init__()
        self.smoothing = smoothing
        self.weight = weight  # 类别权重（兼容之前的类别平衡）
    
    def forward(self, logits, targets):
        n_classes = logits.size(-1)
        # 软化标签：均匀分布 + 真实标签权重
        smooth_label = torch.full_like(logits, self.smoothing / (n_classes - 1))
        smooth_label.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)
        
        # 计算交叉熵（支持类别权重）
        log_probs = torch.log_softmax(logits, dim=1)
        if self.weight is not None:
            smooth_label = smooth_label * self.weight.unsqueeze(0)  # 应用类别权重
        loss = -torch.sum(log_probs * smooth_label, dim=1).mean()
        return loss

# ----------------------
# 7. 模型训练与验证主函数
# ----------------------
def train_models(train_sequences, train_gestures, train_binary, train_ids, 
                test_sequences, n_features, num_gestures, n_splits=5):
    """训练多折模型并进行预测"""
    print("\n===== 训练模型（优化版） =====")
    
    # 处理参与者信息
    subjects = []
    for id in train_ids:
        try:
            if '_' in id:
                parts = id.split('_')
                if len(parts) >= 2:
                    subject = parts[1].split('-')[0]
                    subjects.append(subject)
                    continue
            subjects.append(str(hash(id) % 1000))
        except:
            subjects.append(str(hash(id) % 1000))
    
    subjects = np.array(subjects)
    
    # 新增1：计算手势类别权重（解决类别不平衡）
    class_counts = np.bincount(train_gestures)
    class_weights = 1.0 / class_counts  # 反比于样本数
    class_weights = class_weights / class_weights.sum() * num_gestures  # 归一化
    class_weights = torch.FloatTensor(class_weights).to(device)
    print(f"手势类别权重: {class_weights.cpu().numpy()}")
    
    # 交叉验证
    gkf = GroupKFold(n_splits=n_splits)
    fold_results = []
    test_preds_gesture = np.zeros((len(test_sequences), num_gestures))
    
    for fold, (train_idx, val_idx) in enumerate(gkf.split(train_sequences, groups=subjects)):
        print(f"\n===== 折 {fold+1}/{n_splits} =====")
        
        # 划分训练集和验证集
        X_train, X_val = train_sequences[train_idx], train_sequences[val_idx]
        y_gesture_train, y_gesture_val = train_gestures[train_idx], train_gestures[val_idx]
        y_binary_train, y_binary_val = train_binary[train_idx], train_binary[val_idx]
        
        # 创建数据集和数据加载器（加入数据增强标记）
        train_dataset = SensorSequenceDataset(X_train, y_gesture_train, y_binary_train, is_train=True, aug_prob=0.5)
        val_dataset = SensorSequenceDataset(X_val, y_gesture_val, y_binary_val, is_train=False)
        
        # 根据GPU内存调整batch size
        batch_size = 64
        if torch.cuda.is_available():
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # GB
            if gpu_mem < 8:
                print(f"检测到小内存GPU ({gpu_mem:.1f}GB)，调整batch size为32")
                batch_size = 32
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        # 初始化模型（使用新的CNN+BiLSTM+Attention模型）
        hidden_dim = 128 if n_features < 100 else 256  # 特征数增加，适当调大hidden_dim
        num_layers = 2 if n_features < 100 else 3
        
        model = CNNBiLSTMAttentionModel(
            input_dim=n_features,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_gestures=num_gestures,
            dropout=0.3
        )
        
        # 如果有多个GPU，使用DataParallel
        if torch.cuda.device_count() > 1:
            print(f"使用 {torch.cuda.device_count()} 个GPU进行训练!")
            model = nn.DataParallel(model)
        
        model = model.to(device)
        
        # 定义损失函数和优化器（优化部分）
        criterion_gesture = LabelSmoothingCrossEntropyLoss(
            smoothing=0.1,  # 软化强度
            weight=class_weights
        )
        criterion_binary = nn.BCEWithLogitsLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)  # 替换为AdamW
        # 学习率调度：CosineAnnealingWarmRestarts
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # 训练模型
        best_avg_f1 = 0
        patience = 8  # 调整早停耐心值
        counter = 0
        
        for epoch in range(50):
            print(f"\n epoch {epoch+1}/50 | 当前学习率: {optimizer.param_groups[0]['lr']:.6f}")
            
            # 训练阶段（加入梯度裁剪）
            model.train()
            train_loss, train_gesture_acc, train_binary_acc = 0, 0, 0
            total_samples = 0
            
            for sequences, gestures, binary_labels in tqdm(train_loader, desc="训练"):
                sequences = sequences.to(device)
                gestures = gestures.to(device)
                binary_labels = binary_labels.to(device)
                
                optimizer.zero_grad()
                gesture_out, binary_logits = model(sequences)
                
                loss_gesture = criterion_gesture(gesture_out, gestures)
                loss_binary = criterion_binary(binary_logits, binary_labels)
                loss = loss_gesture + loss_binary
                
                loss.backward()
                # 梯度裁剪（阈值5.0，防止梯度爆炸）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                
                # 统计指标
                train_loss += loss.item() * sequences.size(0)
                _, predicted_gesture = torch.max(gesture_out.data, 1)
                train_gesture_acc += (predicted_gesture == gestures).sum().item()
                predicted_binary = (torch.sigmoid(binary_logits) > 0.5).float()
                train_binary_acc += (predicted_binary == binary_labels).sum().item()
                total_samples += sequences.size(0)
            
            train_loss /= total_samples
            train_gesture_acc /= total_samples
            train_binary_acc /= total_samples
            
            # 验证
            val_loss, val_gesture_acc, val_binary_acc, val_macro_f1, val_binary_f1, val_avg_f1, \
            gesture_preds, gesture_labels, binary_preds, binary_labels = validate(
                model, val_loader, criterion_gesture, criterion_binary, device
            )
            
            # 打印 epoch 结果
            print(f"训练损失: {train_loss:.4f}, 手势准确率: {train_gesture_acc:.4f}, 二元准确率: {train_binary_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 手势准确率: {val_gesture_acc:.4f}, 二元准确率: {val_binary_acc:.4f}")
            print(f"验证宏F1: {val_macro_f1:.4f}, 二元F1: {val_binary_f1:.4f}, 平均F1: {val_avg_f1:.4f}")
            
            # 学习率调度
            scheduler.step()
            
            # 早停机制
            if val_avg_f1 > best_avg_f1:
                best_avg_f1 = val_avg_f1
                torch.save(model.state_dict(), f"best_model_fold_{fold+1}.pth")
                counter = 0
            else:
                counter += 1
                if counter >= patience:
                    print(f"早停在第 {epoch+1} 轮")
                    break
        
        # 加载最佳模型
        model.load_state_dict(torch.load(f"best_model_fold_{fold+1}.pth"))
        
        # 再次验证
        _, _, _, val_macro_f1, val_binary_f1, val_avg_f1, \
        gesture_preds, gesture_labels, binary_preds, binary_labels = validate(
            model, val_loader, criterion_gesture, criterion_binary, device
        )
        
        print(f"\n折 {fold+1} 最佳结果:")
        print(f"宏F1: {val_macro_f1:.4f}, 二元F1: {val_binary_f1:.4f}, 平均F1: {val_avg_f1:.4f}")
        
        # 打印分类报告
        print("\n手势分类报告:")
        print(classification_report(gesture_labels, gesture_preds))
        
        print("\n二元分类报告:")
        print(classification_report(binary_labels, binary_preds))
        
        fold_results.append({
            'fold': fold+1,
            'macro_f1': val_macro_f1,
            'binary_f1': val_binary_f1,
            'avg_f1': val_avg_f1
        })
        
        # 测试集预测
        test_dataset = SensorSequenceDataset(test_sequences)
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=4,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        model.eval()
        fold_test_preds = []
        
        with torch.no_grad():
            for sequences in tqdm(test_loader, desc="测试集预测"):
                sequences = sequences.to(device)
                gesture_out, _ = model(sequences)
                fold_test_preds.append(torch.softmax(gesture_out, dim=1).cpu().numpy())
        
        # 聚合测试集预测
        fold_test_preds = np.vstack(fold_test_preds)
        test_preds_gesture += fold_test_preds / n_splits
    
    # 打印交叉验证汇总结果
    print("\n===== 交叉验证汇总 =====")
    results_df = pd.DataFrame(fold_results)
    print(results_df.mean())
    
    return test_preds_gesture


# ----------------------
# 8. 生成提交文件
# ----------------------
def create_submission(test_preds_gesture, test_ids, gesture_le):
    """生成符合竞赛要求的提交文件"""
    print("\n===== 生成提交文件 =====")
    
    # 从预测概率获取类别
    test_preds_gesture_class = np.argmax(test_preds_gesture, axis=1)
    
    # 解码为原始手势名称
    gestures = gesture_le.inverse_transform(test_preds_gesture_class)
    
    # 生成提交文件
    submission = pd.DataFrame({
        'sequence_id': test_ids,
        'gesture': gestures
    })
    
    submission.to_csv('submission.csv', index=False)
    
    print(f"提交文件已生成，形状: {submission.shape}")
    print("前5行预览:")
    print(submission.head())
    
    return submission


# ----------------------
# 9. 主函数
# ----------------------
def main():
    try:
        # 加载数据并探索
        train, test, train_demo, test_demo = load_and_explore_data()
        
        # 预处理数据
        train, test, imu_cols, thm_cols, tof_cols, demo_cols, gesture_le = preprocess_data(
            train, test, train_demo, test_demo
        )
        
        # 自动确定合适的序列长度
        seq_lengths = train.groupby('sequence_id').size()
        print(f"\n序列长度分布: 平均={seq_lengths.mean():.1f}, 中位数={seq_lengths.median()}, 最大={seq_lengths.max()}")
        max_seq_length = int(seq_lengths.median() * 1.5)
        print(f"自动设置最大序列长度为: {max_seq_length}")
        
        # 准备序列数据
        (train_sequences, train_gestures, train_binary, train_ids,
         test_sequences, test_ids, n_features) = prepare_sequences(train, test, max_seq_length)
        
        # 获取手势类别数量
        num_gestures = len(np.unique(train_gestures))
        print(f"手势类别数量: {num_gestures}")
        
        # 如果特征数量为0，说明数据处理有问题
        if n_features == 0:
            raise ValueError("未找到任何特征列，请检查数据格式")
        
        # 训练模型并预测
        test_preds_gesture = train_models(
            train_sequences, train_gestures, train_binary, train_ids,
            test_sequences, n_features, num_gestures, n_splits=5
        )
        
        # 生成提交文件
        submission = create_submission(test_preds_gesture, test_ids, gesture_le)
        
        print("\n所有任务完成!")
        
    except Exception as e:
        print(f"\n执行过程中出错: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

