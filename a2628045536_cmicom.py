import os, json, joblib, numpy as np, pandas as pd
import random, math
import matplotlib.pyplot as plt
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from timm.scheduler import CosineLRScheduler
from scipy.signal import firwin
from metric import CompetitionMetric

import polars as pl


# 配置
# TRAIN = False                   # 训练时将其设置为 True
TRAIN = True
RAW_DIR = Path("../input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/cmi3-models-p") # 在 TRAIN=False 时使用
EXPORT_DIR = Path("./")         # 输出目录
BATCH_SIZE = 64                 # 批处理大小
PAD_PERCENTILE = 100            # 序列填充百分比
maxlen = PAD_PERCENTILE         # 最大序列长度
LR_INIT = 1e-3                  # 初始学习率
WD = 3e-3                       # 权重衰减
PATIENCE = 40                   # 早停耐心值
FOLDS = 5                       # 交叉验证折数
random_state = 42               # 随机种子
epochs_warmup = 20              # 预热周期数
warmup_lr_init = 1.822126131809773e-05  # 预热初始学习率
lr_min = 3.810323058740104e-09  # 最小学习率
EPOCHS = 125                    # 训练轮次

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"device: {device}")

mean = torch.tensor([
    0,  0, 0, 0, 0,
    0,  9.0319e-03,  1.0849e+00, -2.6186e-03,  3.7651e-03,
    -5.3660e-03, -2.8177e-03,  1.3318e-03, -1.5876e-04,  6.3495e-01,
     6.2877e-01,  6.0607e-01,  6.2142e-01,  6.3808e-01,  6.5420e-01,
     7.4102e-03, -3.4159e-03, -7.5237e-03, -2.6034e-02,  2.9704e-02,
    -3.1546e-02, -2.0610e-03, -4.6986e-03, -4.7216e-03, -2.6281e-02,
     1.5799e-02,  1.0016e-02
], dtype=torch.float32).view(1, -1, 1).to(device)         

std = torch.tensor([
    1, 1, 1, 1, 1, 1, 0.2067, 0.8583, 0.3162,
    0.2668, 0.2917, 0.2341, 0.3023, 0.3281, 1.0264, 0.8838, 0.8686, 1.0973,
    1.0267, 0.9018, 0.4658, 0.2009, 0.2057, 1.2240, 0.9535, 0.6655, 0.2941,
    0.3421, 0.8156, 0.6565, 1.1034, 1.5577
], dtype=torch.float32).view(1, -1, 1).to(device) + 1e-8  

if TRAIN:
    print("训练模式")
else:
    print("推理模式")


print("正在加载数据...")
train_df = pd.read_csv(RAW_DIR / "train.csv")
test_df = pd.read_csv(RAW_DIR / "test.csv")
demographics_df = pd.read_csv(RAW_DIR / "train_demographics.csv")
test_demo_df = pd.read_csv(RAW_DIR / "test_demographics.csv")

print(f"训练集形状: {train_df.shape}")
print(f"测试集形状: {test_df.shape}")
print(f"训练集人口统计信息形状: {demographics_df.shape}")
print(f"测试集人口统计信息形状: {test_demo_df.shape}")

print("\n人口统计信息")
print(demographics_df.describe())

print("\n=== Sequence ID 分析 ===")
train_sequence_counts = train_df['sequence_id'].value_counts()
print(f"真实样本数量（Sequence ID数）: {train_df['sequence_id'].nunique()}")
print(f"样本时间序列长度统计:")
print(f"  最小值: {train_sequence_counts.min()}")
print(f"  最大值: {train_sequence_counts.max()}")
print(f"\n样本时间序列长度前10的sequence_id:")
print(train_sequence_counts.head(10))

count_frequency = train_sequence_counts.value_counts().sort_index()

plt.figure(figsize=(12, 6))
plt.bar(count_frequency.index, count_frequency.values, alpha=0.7, color='steelblue')
plt.xlabel('Sample Time Series Length', fontsize=12)
plt.ylabel('Frequency (Number of Samples)', fontsize=12)
plt.title('Distribution of Sample Time Series Lengths', fontsize=14)
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(EXPORT_DIR / 'sequence_length_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n=== 手势标签分布 ===")
gesture_counts = train_df['gesture'].value_counts()
print("\n标签总数量\n", gesture_counts)

gesture_sequence_counts = train_df.groupby('sequence_id')['gesture'].first().value_counts()
print("\n标签对应样本数量\n", gesture_sequence_counts)

print("\n=== 手势标签的平均序列长度 ===")
sequence_lengths = train_df.groupby('sequence_id').size().reset_index(name='length')
sequence_gestures = train_df.groupby('sequence_id')['gesture'].first().reset_index()
sequence_info = pd.merge(sequence_lengths, sequence_gestures, on='sequence_id')

gesture_avg_length = sequence_info.groupby('gesture')['length'].agg(['mean', 'count']).reset_index()
gesture_avg_length.columns = ['gesture', '平均长度', '序列数量']
gesture_avg_length = gesture_avg_length.sort_values('平均长度', ascending=False)

print("每个手势标签的平均序列长度:")
print(gesture_avg_length.to_string(index=False, float_format="%.2f"))

print("\n=== Gesture 和 Orientation 组合统计 ===")

sequence_gesture_orientation = train_df.groupby('sequence_id')[['gesture', 'orientation']].first()
gesture_orientation_counts = sequence_gesture_orientation.groupby(['gesture', 'orientation']).size().reset_index(name='sequence_count')
gesture_orientation_counts = gesture_orientation_counts.sort_values('gesture')

print(f"\n总组合数(实际可能需要分辨的标签数): {len(gesture_orientation_counts)}")
print("每个(gesture, orientation)组合的序列数量:")
print(gesture_orientation_counts.to_string(index=False))

print("\n=== 缺失值分析 ===")
train_missing = train_df.isnull().sum()
train_missing_pct = (train_missing / len(train_df)) * 100

test_missing = test_df.isnull().sum()
test_missing_pct = (test_missing / len(test_df)) * 100

missing_summary = pd.DataFrame({
    '训练集缺失值': train_missing,
    '训练集缺失比例(%)': train_missing_pct.round(2),
    '测试集缺失值': test_missing,
    '测试集缺失比例(%)': test_missing_pct.round(2)
})

missing_summary = missing_summary[(missing_summary['训练集缺失值'] > 0) | 
                                 (missing_summary['测试集缺失值'] > 0)]

print("缺失值汇总（仅显示有缺失的列）:")
print(missing_summary)

print("\n=== 传感器数据分布 ===")
imu_cols = [col for col in train_df.columns if col.startswith(('acc_', 'rot_'))]
print(f"IMU传感器特征数量: {len(imu_cols)}")

thm_cols = [col for col in train_df.columns if col.startswith('thm_')]
print(f"温度传感器特征数量: {len(thm_cols)}")

tof_cols = [col for col in train_df.columns if col.startswith('tof_')]
print(f"ToF传感器特征数量: {len(tof_cols)}")




# 序列预处理函数
def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    """规范化并清理时间序列序列"""
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values     # 填充缺失值
    return scaler.transform(mat).astype('float32')     # 标准化

# 时间序列处理函数
def pad_sequences_torch(sequences, maxlen, padding='post', truncating='post', value=0.0):
    result = []
    for seq in sequences:
        if len(seq) >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:  # 'pre'
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        result.append(seq)
    return np.array(result, dtype=np.float32)

# 自定义数据集类
class CMI3Dataset(Dataset):
    def __init__(self,
                 X_list,
                 y_list,
                 maxlen,
                 mode="train",
                 imu_dim=7,
                 augment=None):
        self.X_list = X_list
        self.mode = mode
        self.y_list = y_list
        self.maxlen = maxlen
        self.imu_dim = imu_dim     
        self.augment = augment     # 数据增强

    def pad_sequences_torch(self, seq, maxlen, padding='post', truncating='post', value=0.0):
        # 序列填充或截断
        if seq.shape[0] >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:  # 'pre'
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - seq.shape[0]
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        return seq  
        
    def __getitem__(self, index):
        X = self.X_list[index]
        y = self.y_list[index]

        # 数据增强（仅训练模式）
        if self.mode == "train" and self.augment is not None:
            X = self.augment(X, self.imu_dim)     

        X = self.pad_sequences_torch(X, self.maxlen, 'pre', 'pre')
        return X, y
    
    def __len__(self):
        return len(self.X_list)



# 数据增强类
class Augment:
    def __init__(self,
                 p_jitter=0.8, sigma=0.02, scale_range=[0.9,1.1],
                 p_dropout=0.3,
                 p_moda=0.5,          
                 drift_std=0.005,     
                 drift_max=0.25):      
        self.p_jitter  = p_jitter  # 抖动概率
        self.sigma     = sigma     # 噪声标准差
        self.scale_min, self.scale_max = scale_range  # 缩放范围
        self.p_dropout = p_dropout  # Dropout概率
        self.p_moda    = p_moda     # 运动漂移概率
        self.drift_std = drift_std  # 漂移标准差
        self.drift_max = drift_max  # 最大漂移值

    # 抖动与缩放增强
    def jitter_scale(self, x: np.ndarray) -> np.ndarray:
        noise  = np.random.randn(*x.shape) * self.sigma
        scale  = np.random.uniform(self.scale_min,
                                   self.scale_max,
                                   size=(1, x.shape[1]))
        return (x + noise) * scale

    # 传感器随机屏蔽
    def sensor_dropout(self,
                       x: np.ndarray,
                       imu_dim: int) -> np.ndarray:

        if random.random() < self.p_dropout:
            x[:, imu_dim:] = 0.0    # 将非IMU传感器数据置零
        return x

    # 运动漂移增强
    def motion_drift(self, x: np.ndarray, imu_dim: int) -> np.ndarray:

        T = x.shape[0]

        # 生成漂移序列
        drift = np.cumsum(
            np.random.normal(scale=self.drift_std, size=(T, 1)),
            axis=0
        )
        drift = np.clip(drift, -self.drift_max, self.drift_max)   

        # 应用漂移到IMU数据
        x[:, :6] += drift

        if imu_dim > 6:
            x[:, 6:imu_dim] += drift     
        return x
    
    def __call__(self,
                 x: np.ndarray,
                 imu_dim: int) -> np.ndarray:
        if random.random() < self.p_jitter:
            x = self.jitter_scale(x)    # 应用抖动和缩放

        if random.random() < self.p_moda:
            x = self.motion_drift(x, imu_dim)    # 应用运动漂移
 
        x = self.sensor_dropout(x, imu_dim)   # 应用传感器Dropout
        return x




# IMU特征提取器类
class ImuFeatureExtractor(nn.Module):
    def __init__(self, fs=100., add_quaternion=False):
        super().__init__()
        self.fs = fs
        self.add_quaternion = add_quaternion

        k = 15
        # 低通滤波器卷积层
        self.lpf = nn.Conv1d(6, 6, kernel_size=k, padding=k//2,
                             groups=6, bias=False)
        nn.init.kaiming_uniform_(self.lpf.weight, a=math.sqrt(5))

        # 加速度计和陀螺仪的低通滤波器
        self.lpf_acc  = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
        self.lpf_gyro = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)

    def forward(self, imu):
        #分离加速度计和陀螺仪数据
        B, C, T = imu.shape
        acc  = imu[:, 0:3, :]                 # acc_x, acc_y, acc_z
        gyro = imu[:, 3:6, :]                 # gyro_x, gyro_y, gyro_z
        extra = imu[:, 6:, :]                 

        # 1) 计算幅度
        acc_mag  = torch.norm(acc,  dim=1, keepdim=True)          # (B,1,T)
        gyro_mag = torch.norm(gyro, dim=1, keepdim=True)

        # 2) 计算变化率（加速度变化率和角速度变化率）
        jerk = F.pad(acc[:, :, 1:] - acc[:, :, :-1], (1,0))       # (B,3,T)
        gyro_delta = F.pad(gyro[:, :, 1:] - gyro[:, :, :-1], (1,0))

        # 3) 计算能量（平方）
        acc_pow  = acc ** 2
        gyro_pow = gyro ** 2

        # 4) LPF / HPF (低通滤波器 / 高通滤波器)
        acc_lpf  = self.lpf_acc(acc)
        acc_hpf  = acc - acc_lpf
        gyro_lpf = self.lpf_gyro(gyro)
        gyro_hpf = gyro - gyro_lpf

        # 合并所有特征
        features = [
            acc, gyro,
            acc_mag, gyro_mag,
            jerk, gyro_delta,
            acc_pow, gyro_pow,
            acc_lpf, acc_hpf,
            gyro_lpf, gyro_hpf,
        ]
        return torch.cat(features, dim=1)  # (B, C_out, T)


# 压缩激励块（SEBlock）
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)  # 全局平均池化
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()  
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)  # 重新校准特征

# 残差SE-CNN块
class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        
        # 第一个卷积块
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # 第二个卷积块
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # SE块
        self.se = SEBlock(out_channels)
        
        # 快捷连接
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.pool = nn.MaxPool1d(pool_size)   
        self.dropout = nn.Dropout(dropout)  
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        # 首次卷积
        out = F.relu(self.bn1(self.conv1(x)))
        # 第二次卷积
        out = self.bn2(self.conv2(out))
        
        # SE 块
        out = self.se(out)
        
        # 添加快捷连接
        out += shortcut
        out = F.relu(out)
        
        # 池化和Dropout
        out = self.pool(out)
        out = self.dropout(out)
        
        return out

# 注意力层
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)    # 注意力线性层
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context

# 双分支模型（IMU和TOF/Thermal）
class TwoBranchModel(nn.Module):
    def __init__(self, pad_len, imu_dim_raw, tof_dim, n_classes, dropouts=[0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.3], feature_engineering=True, **kwargs):
        super().__init__()
        self.feature_engineering = feature_engineering
        if feature_engineering:
            self.imu_fe = ImuFeatureExtractor(**kwargs)    # IMU特征工程
            imu_dim = 32            
        else:
            self.imu_fe = nn.Identity()    # 恒等映射

            imu_dim = imu_dim_raw   
            
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim

        self.fir_nchan = 7    # FIR滤波器通道数

        weight_decay = 3e-3

        # FIR滤波器初始化
        numtaps = 33  
        fir_coef = firwin(numtaps, cutoff=1.0, fs=10.0, pass_zero=False)
        fir_kernel = torch.tensor(fir_coef, dtype=torch.float32).view(1, 1, -1)
        fir_kernel = fir_kernel.repeat(7, 1, 1)  # (imu_dim, 1, numtaps)
        self.register_buffer("fir_kernel", fir_kernel)
        
        # IMU 深度分支
        self.imu_block1 = ResidualSECNNBlock(imu_dim, 64, 3, dropout=dropouts[0], weight_decay=weight_decay)
        self.imu_block2 = ResidualSECNNBlock(64, 128, 5, dropout=dropouts[1], weight_decay=weight_decay)
        
        # TOF/Thermal 轻量分支
        self.tof_conv1 = nn.Conv1d(tof_dim, 64, 3, padding=1, bias=False)
        self.tof_bn1 = nn.BatchNorm1d(64)
        self.tof_pool1 = nn.MaxPool1d(2)
        self.tof_drop1 = nn.Dropout(dropouts[2])
        
        self.tof_conv2 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        self.tof_bn2 = nn.BatchNorm1d(128)
        self.tof_pool2 = nn.MaxPool1d(2)
        self.tof_drop2 = nn.Dropout(dropouts[3])
        
        # BiLSTM层
        self.bilstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropouts[4])
        
        # 注意力层
        self.attention = AttentionLayer(256)  # 128*2 for bidirectional
        
        # 全连接层
        self.dense1 = nn.Linear(256, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropouts[5])
        
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropouts[6])
        
        self.classifier = nn.Linear(128, n_classes)    # 分类器
        
    def forward(self, x):
        # 分割输入数据        
        imu = x[:, :, :self.fir_nchan].transpose(1, 2)  # (batch, imu_dim, seq_len)
        tof = x[:, :, self.fir_nchan:].transpose(1, 2)  # (batch, tof_dim, seq_len)

        imu = self.imu_fe(imu)   # (B, imu_dim, T)
        # 应用FIR滤波器
        filtered = F.conv1d(
            imu[:, :self.fir_nchan, :],        # (B,7,T)
            self.fir_kernel,
            padding=self.fir_kernel.shape[-1] // 2,
            groups=self.fir_nchan,
        )
        
        imu = torch.cat([filtered, imu[:, self.fir_nchan:, :]], dim=1)  
        imu = (imu - mean) / std    # 标准化
        # IMU 分支处理
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)
        
        # TOF 分支处理
        x2 = F.relu(self.tof_bn1(self.tof_conv1(tof)))
        x2 = self.tof_drop1(self.tof_pool1(x2))
        x2 = F.relu(self.tof_bn2(self.tof_conv2(x2)))
        x2 = self.tof_drop2(self.tof_pool2(x2))
        
        # 合并分支
        merged = torch.cat([x1, x2], dim=1).transpose(1, 2)  # (batch, seq_len, 256)
        
        # BiLSTM处理
        lstm_out, _ = self.bilstm(merged)
        lstm_out = self.lstm_dropout(lstm_out)
        
        # 注意力机制
        attended = self.attention(lstm_out)
        
        # 全连接层处理
        x = F.relu(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        
        # 分类输出
        logits = (self.classifier(x))
        return logits



# 早停
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


# 指数移动平均（EMA）类
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

# 设置随机种子
def set_seed(seed: int = 42):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 


# 训练模式下的数据处理
if TRAIN:
    print("TRAIN MODE – loading dataset")
    df = pd.read_csv(RAW_DIR / "train.csv")

    # 标签编码
    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

    # 特征列表
    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # 分离IMU和TOF/THM特征
    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]
    print(f"  IMU {len(imu_cols)} | TOF/THM {len(tof_cols)} | total {len(feature_cols)} features")

    # 全局缩放器
    scaler = StandardScaler().fit(df[feature_cols].ffill().bfill().fillna(0).values)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")

    # 构建序列
    seq_gp = df.groupby('sequence_id')
    X_list, y_list, id_list = [], [], []
    for seq_id, seq in seq_gp:
        mat = preprocess_sequence(seq, feature_cols, scaler)
        X_list.append(mat)
        y_list.append(seq['gesture_int'].iloc[0])    # 获取序列的标签
        id_list.append(seq_id)
        # lens.append(len(mat))
    
    pad_len = PAD_PERCENTILE#int(np.percentile(lens, PAD_PERCENTILE))
    print(pad_len)
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(feature_cols))
    id_list = np.array(id_list)
    X_list_all = pad_sequences_torch(X_list, maxlen=pad_len, padding='pre', truncating='pre')
    y_list_all = np.eye(len(le.classes_))[y_list].astype(np.float32)  # One-hot 编码

    # 数据增强器
    augmenter = Augment(
        p_jitter=0.9844818619033621, 
        sigma=0.03291295776089293, 
        scale_range=(0.7542342630597011,1.1625052821731077),
        p_dropout=0.41782786013520684,
        p_moda=0.3910622476959722, 
        drift_std=0.0040285239353308015, 
        drift_max=0.3929358950258158    
    )


# 数据集划分统计
if TRAIN:
    print("\n=== 数据集划分统计 ===\n")
    
    # 预先创建序列到元数据的映射
    sequence_metadata = {}
    for seq_id in id_list:
        seq_data = df[df['sequence_id'] == seq_id].iloc[0]
        sequence_metadata[seq_id] = {
            'subject': seq_data['subject'],
            'gesture': seq_data['gesture'],
            'gesture_int': seq_data['gesture_int']
        }
    
    # 获取所有受试者ID
    all_subjects = demographics_df['subject'].unique()
    
    # 分层K折交叉验证
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=random_state)
    
    # 获取序列对应的标签（用于分层划分）
    sequence_labels = np.array([sequence_metadata[seq_id]['gesture_int'] for seq_id in id_list])
    
    # 定义BFRB样手势
    bfrb_gestures = [
        'Eyebrow - pull hair', 'Eyelash - pull hair', 'Above ear - pull hair',
        'Forehead - pull hairline', 'Neck - scratch', 'Neck - pinch skin',
        'Forehead - scratch', 'Cheek - pinch skin'
    ]
    
    # 获取所有手势类别（在训练模式下使用le.classes_）
    if TRAIN:
        all_gesture_classes = le.classes_
    else:
        all_gesture_classes = gesture_classes
    
    # 对于第一个fold进行统计
    for fold, (train_idx, val_idx) in enumerate(skf.split(id_list, sequence_labels)):
        if fold == 0:  # 只统计第一个fold
            # 获取训练集和验证集的序列ID
            train_sequences = id_list[train_idx]
            val_sequences = id_list[val_idx]
            
            # 统计序列数量
            train_seq_count = len(train_sequences)
            val_seq_count = len(val_sequences)
            total_seq_count = len(id_list)
            
            # 统计受试者数量（使用集合去重）
            train_subjects = set(sequence_metadata[seq_id]['subject'] for seq_id in train_sequences)
            val_subjects = set(sequence_metadata[seq_id]['subject'] for seq_id in val_sequences)
            
            train_subject_count = len(train_subjects)
            val_subject_count = len(val_subjects)
            total_subject_count = len(all_subjects)
            
            # 统计BFRB样手势样本数
            train_bfrb_count = sum(1 for seq_id in train_sequences 
                                 if sequence_metadata[seq_id]['gesture'] in bfrb_gestures)
            val_bfrb_count = sum(1 for seq_id in val_sequences 
                               if sequence_metadata[seq_id]['gesture'] in bfrb_gestures)
            total_bfrb_count = sum(1 for seq_id in id_list 
                                 if sequence_metadata[seq_id]['gesture'] in bfrb_gestures)
            
            # 统计非BFRB样手势样本数
            train_non_bfrb_count = train_seq_count - train_bfrb_count
            val_non_bfrb_count = val_seq_count - val_bfrb_count
            total_non_bfrb_count = total_seq_count - total_bfrb_count
            
            # 统计类别数量
            train_classes = set(sequence_metadata[seq_id]['gesture'] for seq_id in train_sequences)
            val_classes = set(sequence_metadata[seq_id]['gesture'] for seq_id in val_sequences)
            
            train_class_count = len(train_classes)
            val_class_count = len(val_classes)
            total_class_count = len(all_gesture_classes)
            
            # 使用pandas DataFrame输出统计结果
            stats_data = {
                '统计指标': ['序列数量', '受试者数量', 'BFRB样手势样本数', '非BFRB样手势样本数', '类别数量'],
                '训练集 (Train Set)': [train_seq_count, train_subject_count, train_bfrb_count, train_non_bfrb_count, train_class_count],
                '验证集 (Validation Set)': [val_seq_count, val_subject_count, val_bfrb_count, val_non_bfrb_count, val_class_count],
                '全体数据集 (Total)': [total_seq_count, total_subject_count, total_bfrb_count, total_non_bfrb_count, total_class_count]
            }
            
            stats_df = pd.DataFrame(stats_data)
            print("数据集划分统计表:")
            print("=" * 80)
            print(stats_df.to_string(index=False))
            print("=" * 80)
            break


# 训练循环
if TRAIN:
    # 分层K折交叉验证
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=random_state)

    # 为每个fold创建图表
    for fold, (train_idx, val_idx) in enumerate(skf.split(id_list, np.argmax(y_list_all, axis=1))):
        print(f"\n{'='*50}")
        print(f"Starting training for fold {fold+1}/{FOLDS}")
        print(f"{'='*50}")

        # 初始化存储训练历史的列表
        train_losses = []
        val_losses = []
        train_accs = []
        val_accs = []
        
        train_list = X_list_all[train_idx]
        train_y_list = y_list_all[train_idx]
        val_list = X_list_all[val_idx]
        val_y_list = y_list_all[val_idx]
        
        # 数据加载器
        train_dataset = CMI3Dataset(train_list, train_y_list, maxlen, mode="train", imu_dim=len(imu_cols),
                                augment=augmenter)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4,drop_last=True)
    
        val_dataset = CMI3Dataset(val_list, val_y_list, maxlen, mode="val")
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4,drop_last=True)
    
        # 模型初始化
        model = TwoBranchModel(maxlen, len(imu_cols), len(tof_cols), 
                      len(le.classes_)).to(device)
        ema = EMA(model, decay=0.999)   # 指数移动平均
        
        # 优化器和学习率调度器
        optimizer = Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
        steps_per_epoch = len(train_loader)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5*steps_per_epoch)
        
        steps_per_epoch = len(train_loader)
        nbatch = len(train_loader)
        warmup = epochs_warmup * nbatch
        nsteps = EPOCHS * nbatch
        scheduler = CosineLRScheduler(optimizer,
                          warmup_t=warmup, warmup_lr_init=warmup_lr_init, warmup_prefix=True,
                          t_initial=(nsteps - warmup), lr_min=lr_min) 
    
        early_stopping = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        val_best_acc = 0.0
        i_scheduler = 0
        
        # 训练循环
        for epoch in range(EPOCHS):
            model.train()
            train_preds = []
            train_targets = []
            train_loss_epoch = 0.0
            
            for X, y in (train_loader):  
                X, y = X.float().to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(X)
    
                loss = -torch.sum(F.log_softmax(logits, dim=1) * y, dim=1).mean()    # 交叉熵损失
                loss.backward()
                optimizer.step()
                ema.update(model)    # 更新EMA 
                train_preds.extend(logits.argmax(dim=1).cpu().numpy())
                train_targets.extend(y.argmax(dim=1).cpu().numpy())
                scheduler.step(i_scheduler)
                i_scheduler +=1
    
                train_loss_epoch += loss.item()
                
            # 计算训练损失和准确率
            train_loss_epoch /= len(train_loader)
            train_acc_epoch = CompetitionMetric().calculate_hierarchical_f1(
                pd.DataFrame({'gesture': le.classes_[train_targets]}),
                pd.DataFrame({'gesture': le.classes_[train_preds]}))
            
            model.eval()
            val_preds = []
            val_targets = []
            val_loss_epoch = 0.0
            with torch.inference_mode():
                for X, y in (val_loader):  
                    half = BATCH_SIZE // 2         

                    # 数据增强：部分TOF数据置零
                    x_front = X[:half]               
                    x_back  = X[half:].clone()      
                    
                    x_back[:, :, 7:] = 0.0    
                    X = torch.cat([x_front, x_back], dim=0)  # (B, C, T)
                    X, y = X.float().to(device), y.to(device)
                    
                    logits = model(X)
                    val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    val_targets.extend(y.argmax(dim=1).cpu().numpy())
                    
                    loss = F.cross_entropy(logits, y)
                    val_loss_epoch += loss.item()

            # 计算验证损失和准确率
            val_loss_epoch /= len(val_loader)
            val_acc_epoch = CompetitionMetric().calculate_hierarchical_f1(
                pd.DataFrame({'gesture': le.classes_[val_targets]}),
                pd.DataFrame({'gesture': le.classes_[val_preds]}))
            
            # 存储历史记录
            train_losses.append(train_loss_epoch)
            val_losses.append(val_loss_epoch)
            train_accs.append(train_acc_epoch)
            val_accs.append(val_acc_epoch)
            
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0 or epoch == EPOCHS - 1:
                print(f"Epoch {epoch+1:3d}/{EPOCHS} - "
                      f"Train Loss: {train_loss_epoch:.4f}, Train Acc: {train_acc_epoch:.4f} - "
                      f"Val Loss: {val_loss_epoch:.4f}, Val Acc: {val_acc_epoch:.4f}")
            
            # 早停检查
            if early_stopping(val_loss_epoch, model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # 保存模型
        torch.save({
            'model_state_dict': model.state_dict(),
            'imu_dim': len(imu_cols),
            'tof_dim': len(tof_cols),
            'n_classes': len(le.classes_),
            'pad_len': pad_len
        }, EXPORT_DIR / f"gesture_two_branch_fold{fold}.pth")
        
        # 绘制训练历史图表
        plt.figure(figsize=(12, 5))
        
        # 损失图表
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.title(f'Fold {fold+1} - Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # 准确率图表
        plt.subplot(1, 2, 2)
        plt.plot(train_accs, label='Train Accuracy')
        plt.plot(val_accs, label='Validation Accuracy')
        plt.title(f'Fold {fold+1} - Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(EXPORT_DIR / f'training_history_fold{fold}.png')
        plt.show()
        plt.close()
        
        print(f"Fold {fold+1} completed - Best Val Acc: {max(val_accs):.4f}")
        print("Training history plot saved as", EXPORT_DIR / f'training_history_fold{fold}.png')

    print("Training done - all artefacts saved in", EXPORT_DIR)


# 推理模式 
if TRAIN == False:
    # 推理模式：加载预训练模型和预处理工具
    print("INFERENCE MODE – loading artefacts from", PRETRAINED_DIR)
    feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
    scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
    gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]

    
    # 加载所有折的模型
    MODELS = [f'gesture_two_branch_fold{i}.pth' for i in range(5)]
    
    models = []
    for path in MODELS:
        checkpoint = torch.load(PRETRAINED_DIR / path, map_location=device)
        
        model = TwoBranchModel(
            checkpoint['pad_len'], 
            checkpoint['imu_dim'], 
            checkpoint['tof_dim'], 
            checkpoint['n_classes']
            ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        models.append(model)

    print("  model, scaler, pads loaded – ready for evaluation")

# 确保gesture_classes在两个模式下都存在
if TRAIN:
    gesture_classes = le.classes_



# 预测函数
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    global gesture_classes
    if gesture_classes is None:
        gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    df_seq = sequence.to_pandas()
    mat = preprocess_sequence(df_seq, feature_cols, scaler)   # 预处理
    pad = pad_sequences_torch([mat], maxlen=pad_len, padding='pre', truncating='pre')    # 填充
    
    with torch.no_grad():
        x = torch.FloatTensor(pad).to(device)
        outputs = None
        # 模型集成预测
        for model in models:
            model.eval()
            p = torch.softmax(model(x), dim=1)
            if outputs is None: outputs = p
            else: outputs += p
        outputs /= len(models)    # 平均预测结果
        
        idx = int(outputs.argmax(dim=1)[0].cpu().numpy())    # 获取预测类别
    
    return str(gesture_classes[idx])   # 返回手势名称



# Kaggle竞赛接口
import kaggle_evaluation.cmi_inference_server
if TRAIN:
    print("训练完成")
else:
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
    
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            data_paths=(
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
            )
        )
    print("预测完成")

