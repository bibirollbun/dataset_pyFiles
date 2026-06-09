# 2025-07-09 11:25:00 [INFO] Average F1 Score: 0.7858
# 2025-07-09 11:25:00 [INFO] 各Fold分数详情:
# 2025-07-09 11:25:00 [INFO]   Fold 1: 0.7734
# 2025-07-09 11:25:00 [INFO]   Fold 2: 0.8004
# 2025-07-09 11:25:00 [INFO]   Fold 3: 0.7856
# 2025-07-09 11:25:00 [INFO]   Fold 4: 0.7701
# 2025-07-09 11:25:00 [INFO]   Fold 5: 0.7994


import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from pathlib import Path
import pickle
from scipy.spatial.transform import Rotation as R
import warnings
import torch.nn as nn
import torch.optim as optim
import time
import torch.nn.functional as F
from copy import deepcopy
import gc
import random, math
import warnings 
from torch.optim import Adam, AdamW, Adamax
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from timm.scheduler import CosineLRScheduler
from scipy.signal import firwin
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class ToFScaler:
    def __init__(self):
        self.means_full = np.zeros(320)  # 320维均值向量
        self.stds_full = np.ones(320)    # 320维标准差向量
        self.failure_counts = {s: 0 for s in range(1, 6)}
        # 预计算列索引
        self.col_indices = {}
        for sensor in range(1, 6):
            start = (sensor-1)*64
            end = sensor*64
            self.col_indices[sensor] = (start, end)
    
    def fit(self, X_tof):
        # 一次性提取所有TOF数据
        tof_data = X_tof.values if isinstance(X_tof, pd.DataFrame) else X_tof
        
        # 重塑为(samples, sensor, values)
        if tof_data.shape[1] != 320:
            raise ValueError(f"TOF数据应有320列，实际有{tof_data.shape[1]}列")
        
        tof_data = tof_data.reshape(-1, 5, 64)
        
        for sensor in range(5):
            sensor_idx = sensor + 1
            sensor_data = tof_data[:, sensor, :]
            # 故障检测
            failure_mask = (sensor_data == 0)
            self.failure_counts[sensor_idx] = failure_mask.sum()
            # 有效值掩码
            valid_mask = (sensor_data != -1) & (sensor_data != 0)
            valid_data = sensor_data[valid_mask]
            # 计算均值和标准差
            if len(valid_data) > 0:
                mean_val = valid_data.mean()
                std_val = valid_data.std()
            else:
                mean_val = 0
                std_val = 1
            # 更新全量向量
            start, end = self.col_indices[sensor_idx]
            self.means_full[start:end] = mean_val
            self.stds_full[start:end] = std_val if std_val > 1e-7 else 1.0
        return self

    def transform(self, X_tof):
        # Get data
        if isinstance(X_tof, pd.DataFrame):
            tof_cols = X_tof.columns
            tof_values = X_tof.values.copy()
        else:
            tof_values = X_tof.copy()
        # Create masks for the entire data at once
        valid_mask = (tof_values != -1) & (tof_values != 0)
        no_signal_mask = (tof_values == -1)
        failure_mask = (tof_values == 0)
        # Normalize all valid values in one operation
        normalized = np.zeros_like(tof_values)
        normalized[valid_mask] = (tof_values[valid_mask] - self.means_full[np.where(valid_mask)[1]]) / self.stds_full[np.where(valid_mask)[1]]
        # Set special values
        normalized[no_signal_mask] = 10     # -1 → 10
        normalized[failure_mask]   = -10    # 0 → -10
        # Return result
        if isinstance(X_tof, pd.DataFrame):
            return pd.DataFrame(normalized, columns=tof_cols, index=X_tof.index)
        return normalized



class GestureDataset(Dataset):
    def __init__(self, sequence_dir, label_dir, metadata_df, max_len=None, 
                 is_train=False, scale_path=None, use_augmentation=False,
                 drift_std=0.01, drift_max=0.05):
        """
        参数:
            sequence_dir: 序列数据根目录
            label_dir: 标签数据根目录
            metadata_df: 包含序列元数据的DataFrame
            max_len: 序列最大长度
            train_scalers: 训练集的归一化统计量（现在是非TOF和TOF归一化器的元组）
        """
        self.sequence_dir = sequence_dir
        self.label_dir = label_dir
        self.metadata = metadata_df
        self.sequence_ids = self.metadata['sequence_id'].tolist()
        self.max_len = max_len
        self.is_train = is_train
        self.scale_path = scale_path
        self.use_augmentation = use_augmentation and is_train # 增强只在训练时使用
        self.drift_std = drift_std
        self.drift_max = drift_max

        # 传感器组定义
        self.thm_tof_dim = 325  # 温度(5) + TOF(320)
        self.tof_dim = 320  # TOF特征
        self.demo_dim = 7    # 人口统计特征
        
        # 手势编码器
        self.gesture_encoder = LabelEncoder()
        self.gesture_encoder.fit(self.metadata['gesture'])
        
        # 归一化统计量
        self.non_tof_scaler = StandardScaler()
        self.tof_scaler = ToFScaler()
        
        # 加载序列数据
        self.sequence_data = self._load_sequences()
        
        # 如果是训练集且需要计算归一化统计量
        if is_train and scale_path is not None:
            self._fit_scalers()
            self._save_scalers(scale_path)
            # 归一化并缓存所有序列
            self._normalize_and_cache_sequences()
        elif not is_train and scale_path is not None:
            # 如果是验证集或测试集，加载预先计算的归一化统计量
            if os.path.exists(os.path.join(scale_path, 'train_scalers.pkl')):
                with open(os.path.join(scale_path, 'train_scalers.pkl'), 'rb') as f:
                    self.non_tof_scaler, self.tof_scaler = pickle.load(f)
                # 归一化并缓存所有序列
                self._normalize_and_cache_sequences()
            else:
                warnings.warn("未找到训练集归一化统计量，载入归一化器失败")
        else:
            warnings.warn("未提供归一化统计量路径，将无法进行归一化处理")
            # 即使没有归一化器，也缓存当前序列（未归一化）
            self.cached_sequences = self.sequence_data.copy()

        # 如果没有提供最大长度，则使用最长序列长度
        print(f"数据集初始化完成，包含 {len(self)} 个序列")
        print(f"序列长度统计: 平均={np.mean([len(s) for s in self.cached_sequences]):.1f}, " 
              f"最大={np.max([len(s) for s in self.cached_sequences])}")
        

    def _normalize_and_cache_sequences(self):
        """归一化所有序列并缓存结果"""
        self.cached_sequences = []
        for seq in tqdm(self.sequence_data, desc="归一化序列数据", leave=False):
            # 分离非TOF和TOF特征
            non_tof_part = seq[:, :-self.tof_dim]
            tof_part = seq[:, -self.tof_dim:]
            # 分别归一化
            norm_non_tof = self.non_tof_scaler.transform(non_tof_part)
            norm_tof = self.tof_scaler.transform(tof_part)
            # 拼接归一化后的特征
            normalized_seq = np.concatenate([norm_non_tof, norm_tof], axis=1)
            self.cached_sequences.append(normalized_seq)
        # 释放原始数据内存
        self.sequence_data = None


    def _load_sequences(self):
        """加载所有序列数据"""
        sequence_data = []
        for seq_id in tqdm(self.sequence_ids, desc="加载序列数据", leave=False):
            subject_id = self.metadata[self.metadata['sequence_id'] == seq_id]['subject'].iloc[0]
            seq_path = os.path.join(self.sequence_dir, str(subject_id), f"{seq_id}.npy")
            if os.path.exists(seq_path):
                seq = np.load(seq_path)  # 加载原始序列数据
                seq = self._feature_engineering(seq)
                sequence_data.append(seq)
        return sequence_data


    def _remove_gravity_from_acc(self, acc_values, quat_values):
        num_samples = acc_values.shape[0]
        linear_accel = np.zeros_like(acc_values)
        gravity_world = np.array([0, 0, 9.81])
        for i in range(num_samples):
            if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)):
                linear_accel[i, :] = acc_values[i, :] 
                continue
            try:
                rotation = R.from_quat(quat_values[i])
                gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
                linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
            except ValueError:
                linear_accel[i, :] = acc_values[i, :]  
        return linear_accel
    def _calculate_angular_velocity_from_quat(self, quat_values, time_delta=1/10):
        num_samples = quat_values.shape[0]
        angular_vel = np.zeros((num_samples, 3))
        for i in range(num_samples - 1):
            q_t = quat_values[i]
            q_t_plus_dt = quat_values[i+1]
            if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
            np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
                continue
            try:
                rot_t = R.from_quat(q_t)
                rot_t_plus_dt = R.from_quat(q_t_plus_dt)
                delta_rot = rot_t.inv() * rot_t_plus_dt
                angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
            except ValueError:
                pass
        return angular_vel
    def _calculate_angular_distance(self, quat_values):
        num_samples = quat_values.shape[0]
        angular_dist = np.zeros(num_samples)
        for i in range(num_samples - 1):
            q1 = quat_values[i]
            q2 = quat_values[i+1]
            if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
            np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
                angular_dist[i] = 0
                continue
            try:
                r1 = R.from_quat(q1)
                r2 = R.from_quat(q2)
                relative_rotation = r1.inv() * r2
                angle = np.linalg.norm(relative_rotation.as_rotvec())
                angular_dist[i] = angle
            except ValueError:
                angular_dist[i] = 0
                pass
        return angular_dist



    def _feature_engineering(self, seq):
        imu = seq[:,:7]
        other_features = seq[:, 7:]  # 除IMU外的其他特征
        acc = imu[:,0:3]                                                                       # x, y, z
        rot = imu[:,3:7]                                                                       # w, x, y, z
        acc_mag = np.sqrt(acc[:,0]**2 + acc[:,1]**2 + acc[:,2]**2)                             # 1,计算加速度模长
        rot_angle = 2 * np.arccos(rot[:,0].clip(-1, 1))                                        # 1,计算四元数旋转角度（弧度）
        acc_mag_jerk = np.diff(acc_mag, prepend=acc_mag[0])                                    # 1,计算加速度模长的jerk（差分，首位补0）
        rot_angle_vel = np.diff(rot_angle, prepend=rot_angle[0])                               # 1,计算旋转角度的速度（差分，首位补0）
        linear_acc = self._remove_gravity_from_acc(acc, rot[:, [1, 2, 3, 0]])                  # 3,去除重力后的线性加速度（输入四元数格式为[x, y, z, w]）
        linear_acc_mag = np.sqrt(linear_acc[:,0]**2 + linear_acc[:,1]**2 + linear_acc[:,2]**2) # 1,线性加速度模长
        linear_acc_mag_jerk = np.diff(linear_acc_mag, prepend=linear_acc_mag[0])               # 1,线性加速度模长的jerk（差分，首位补0）
        angular_vel = self._calculate_angular_velocity_from_quat(rot[:, [1, 2, 3, 0]])         # 3,计算角速度（输入四元数格式为[x, y, z, w]）
        angular_distance = self._calculate_angular_distance(rot[:, [1, 2, 3, 0]])              # 1,计算相邻帧的角距离（弧度）
        # 拼接所有特征
        new_imu = np.concatenate([
            acc,                          # 3
            rot,                          # 4
            acc_mag[:, None],             # 1
            rot_angle[:, None],           # 1
            acc_mag_jerk[:, None],        # 1
            rot_angle_vel[:, None],       # 1
            linear_acc,                   # 3
            linear_acc_mag[:, None],      # 1
            linear_acc_mag_jerk[:, None], # 1
            angular_vel,                  # 3
            angular_distance[:, None]     # 1
        ], axis=1)
        thm = other_features[:, :5]  # 温度特征
        tof = other_features[:, 5:5+320]  # TOF特征
        demo = other_features[:, 5+320:]  # 人口统计特征
        # 拼接所有特征: IMU(20) + 温度(5) + 人口统计(7) + TOF(320)
        features = np.concatenate([new_imu, thm,demo,tof], axis=1)
        return features

    def _fit_scalers(self):
        """在整个训练集上计算归一化统计量 (优化版)"""
        all_sequences = np.vstack(self.sequence_data)
        # 分离非TOF和TOF特征
        non_tof_features = all_sequences[:, :-self.tof_dim]
        tof_features = all_sequences[:, -self.tof_dim:]
        # 计算归一化统计量
        self.non_tof_scaler.fit(non_tof_features)
        self.tof_scaler.fit(tof_features)  # 直接传入数组
        print("训练集归一化统计量计算完成")

    def _save_scalers(self, scale_path):
        """保存归一化器到文件"""
        Path(scale_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(scale_path, 'train_scalers.pkl'), 'wb') as f:
            # 保存两个归一化器组成的元组
            pickle.dump((self.non_tof_scaler, self.tof_scaler), f)
        print(f"保存训练集归一化统计量到 {scale_path}/train_scalers.pkl")

    def normalize_features(self, sequence):
        """归一化特征"""
        # 分离非TOF和TOF特征
        non_tof_part = sequence[:, :-self.tof_dim]
        tof_part = sequence[:, -self.tof_dim:]  # TOF(320)
        # 分别归一化 (TOF部分直接使用数组)
        norm_non_tof = self.non_tof_scaler.transform(non_tof_part)
        norm_tof = self.tof_scaler.transform(tof_part)  # 返回数组
        # 拼接归一化后的特征
        return np.concatenate([norm_non_tof, norm_tof], axis=1)


    # --- 数据增强方法 (作用于归一化数据) ---
    def _jitter(self, sequence, sigma=0.1):
        return sequence + np.random.normal(loc=0., scale=sigma, size=sequence.shape)
    def _time_mask(self, sequence, max_mask_size=25):
        seq_len = sequence.shape[0]
        mask_size = np.random.randint(1, max_mask_size)
        start = np.random.randint(0, max(1, seq_len - mask_size))
        sequence[start : start + mask_size] = 0
        return sequence
    def _feature_mask(self, sequence, max_mask_size=10):
        num_features = sequence.shape[1]
        mask_size = np.random.randint(1, max_mask_size)
        masked_features = np.random.choice(num_features, mask_size, replace=False)
        sequence[:, masked_features] = 0
        return sequence
    def _motion_drift(self, imu_features: np.ndarray) -> np.ndarray:
        """在IMU特征上模拟传感器漂移"""
        T = imu_features.shape[0]
        # 生成漂移信号
        drift = np.cumsum(np.random.normal(scale=self.drift_std, size=(T, 1)),axis=0)
        drift = np.clip(drift, -self.drift_max, self.drift_max)   
        # 将漂移应用到加速度和角速度相关的特征上
        # acc (cols 0-2), linear_acc (cols 10-12), angular_vel (cols 15-17)
        imu_features[:, 0:3] += drift
        imu_features[:, 10:13] += drift
        imu_features[:, 15:18] += drift
        return imu_features
    def _apply_augmentations(self, sequence):
        """在归一化后的序列上应用增强"""
        # 增强1: Jitter
        if np.random.rand() < 0.7:
            sequence = self._jitter(sequence, sigma=0.05)
        # 增强2 & 3: Time and Feature Masking
        if np.random.rand() < 0.5:
            sequence = self._time_mask(sequence, max_mask_size=20)
        if np.random.rand() < 0.5:
            sequence = self._feature_mask(sequence, max_mask_size=15)
        # 增强4: Motion Drift (只作用于IMU部分)
        if np.random.rand() < 0.5:
            imu_features = sequence[:, :20]
            other_features = sequence[:, 20:]
            augmented_imu = self._motion_drift(imu_features)
            sequence = np.concatenate([augmented_imu, other_features], axis=1)
        return sequence


    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        # 直接使用缓存的归一化序列
        sequence = self.cached_sequences[idx].copy()
        if self.use_augmentation:
            sequence = self._apply_augmentations(sequence)
        seq_id = self.sequence_ids[idx]
        subject_id = self.metadata[self.metadata['sequence_id'] == seq_id]['subject'].iloc[0]
        # 加载阶段标签
        label_path = os.path.join(self.label_dir, str(subject_id), f"{seq_id}.npy")
        phase_labels = np.load(label_path).astype(np.float32)
        # 截断序列
        if self.max_len and len(sequence) > self.max_len:
            sequence = sequence[-self.max_len:]
            phase_labels = phase_labels[-self.max_len:]
        # 分割特征
        # 归一化后特征顺序: 
        #   [0:20] - IMU特征
        #   [20:25] - 温度特征
        #   [25:32] - 人口统计特征 (只在第一帧使用)
        #   [32:352] - TOF特征
        tof_features = sequence[:, -self.tof_dim:]  # TOF特征
        imu_thm_demo_features = sequence[:, :-self.tof_dim]
        demo_features = imu_thm_demo_features[0, -7:]  # 人口统计数据在整个序列中相同
        thm_features = imu_thm_demo_features[:, :-7][:, -5:]  # 温度特征
        imu_features = imu_thm_demo_features[:, :-7][:, :-5] # IMU特征
        thm_tof_features = np.concatenate([thm_features, tof_features], axis=1)
        # 获取元数据
        meta = self.metadata[self.metadata['sequence_id'] == seq_id].iloc[0]
        gesture_label = self.gesture_encoder.transform([meta['gesture']])[0]
        # 转换为张量
        imu_features = torch.tensor(imu_features, dtype=torch.float32)
        thm_tof_features = torch.tensor(thm_tof_features, dtype=torch.float32)
        demo_features = torch.tensor(demo_features, dtype=torch.float32)
        phase_labels = torch.tensor(phase_labels, dtype=torch.long)
        
        return {
            'imu': imu_features,
            'thm_tof': thm_tof_features,
            'demo': demo_features,
            'phase_labels': phase_labels,
            'length': len(imu_features),
            'sequence_id': seq_id,
            'gesture_label': gesture_label
        }


def collate_fn(batch):
    """自定义批处理函数"""
    # 按序列长度排序
    batch.sort(key=lambda x: x['length'], reverse=True)
    # 提取不同特征
    imu_features = [item['imu'] for item in batch]
    thm_tof_features = [item['thm_tof'] for item in batch]
    demo_features = torch.stack([item['demo'] for item in batch])
    phase_labels = [item['phase_labels'] for item in batch]
    lengths = torch.tensor([item['length'] for item in batch], dtype=torch.long)
    sequence_ids = [item['sequence_id'] for item in batch]
    gesture_labels = torch.tensor([item['gesture_label'] for item in batch], dtype=torch.long)
    # 填充序列
    imu_padded = pad_sequence(imu_features, batch_first=True, padding_value=0)
    thm_tof_padded = pad_sequence(thm_tof_features, batch_first=True, padding_value=0)
    phase_padded = pad_sequence(phase_labels, batch_first=True, padding_value=3)
    # 创建掩码
    max_len = imu_padded.size(1)
    mask = torch.arange(max_len)[None, :] < lengths[:, None]
    mask = mask.float()
    
    return {
        'imu': imu_padded,
        'thm_tof': thm_tof_padded,
        'demo': demo_features,
        'phase_labels': phase_padded,
        'mask': mask,
        'lengths': lengths,
        'gesture_labels': gesture_labels,
        'sequence_ids': sequence_ids
    }


def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


import pandas as pd
from sklearn.metrics import f1_score
class ParticipantVisibleError(Exception):
    """Errors raised here will be shown directly to the competitor."""
    pass
class CompetitionMetric:
    """Hierarchical macro F1 for the CMI 2025 challenge."""
    def __init__(self):
        self.target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
        self.non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in tray and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]
        self.all_classes = self.target_gestures + self.non_target_gestures
    def calculate_hierarchical_f1(
        self,
        sol: pd.DataFrame,
        sub: pd.DataFrame
    ) -> float:
        invalid_types = {i for i in sub['gesture'].unique() if i not in self.all_classes}
        if invalid_types:
            raise ParticipantVisibleError(
                f"Invalid gesture values in submission: {invalid_types}"
            )
        y_true_bin = sol['gesture'].isin(self.target_gestures).values
        y_pred_bin = sub['gesture'].isin(self.target_gestures).values
        f1_binary = f1_score(
            y_true_bin,
            y_pred_bin,
            pos_label=True,
            zero_division=0,
            average='binary'
        )
        y_true_mc = sol['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        y_pred_mc = sub['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        f1_macro = f1_score(
            y_true_mc,
            y_pred_mc,
            average='macro',
            zero_division=0
        )
        return 0.5 * f1_binary + 0.5 * f1_macro, f1_binary, f1_macro
def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str
) -> float:
    for col in (row_id_column_name, 'gesture'):
        if col not in solution.columns:
            raise ParticipantVisibleError(f"Solution file missing required column: '{col}'")
        if col not in submission.columns:
            raise ParticipantVisibleError(f"Submission file missing required column: '{col}'")
    metric = CompetitionMetric()
    return metric.calculate_hierarchical_f1(solution, submission)


import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
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
        return x * y.expand_as(x)
    

class CoordAttention(nn.Module):
    """
    Coordinate Attention for Sequences.
    Input Dimension: (B, T, C)
    Output Dimension: (B, T, C)
    """
    def __init__(self, channels, reduction=8):
        super(CoordAttention, self).__init__()
        self.mid_channels = max(8, channels // reduction)

        self.compression = nn.Sequential(
            nn.Conv1d(channels, self.mid_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(self.mid_channels),
            nn.SiLU(inplace=True)
        )
        # Attention branches
        self.time_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)  
        self.channel_conv = nn.Conv1d(self.mid_channels, channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = x.permute(0, 2, 1)
        # x: (B, T, C)
        x_p = x.permute(0, 2, 1)  # (B, C, T)
        f   = self.compression(x_p)  # (B, rC, T)
        ## Time Attention (B, 1, T)
        f_t = f.mean(dim=1, keepdim=True)      
        time_attn = self.sigmoid(self.time_conv(f_t))  
        ## Channel Attention (B, C, 1)
        f_c = f.mean(dim=2, keepdim=True)      
        channel_attn = self.sigmoid(self.channel_conv(f_c)) 
        ## (B, T, C)
        out = (x_p * time_attn * channel_attn).permute(0,2,1)
        return out.permute(0, 2, 1)


class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, Model, reduction=8, pool_size=2, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # attention block
        self.attention = Model(out_channels, reduction)
        # Shortcut connection
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
        # First conv
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv
        out = self.bn2(self.conv2(out))
        # attention block
        out = self.attention(out)
        # Add shortcut
        out += shortcut
        out = F.relu(out)
        # Pool and dropout
        out = self.pool(out)
        out = self.dropout(out)
        
        return out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context

class MLPAttention(nn.Module):
    def __init__(self, feature_dim):
        super(MLPAttention, self).__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//8),
            nn.SiLU(inplace=True),
            nn.Linear(feature_dim//8, 1)
        )
    def forward(self, x):
        # inputs shape: (B, T, C)
        weights = self.attn(x)  # (B, T, 1)
        weights = F.softmax(weights, dim=1)  # (B, T, 1)
        context = (x * weights).sum(dim=1)  # (B, C)
        return context


class IMUOnlyModel(nn.Module):
    def __init__(self, imu_dim, tof_dim, n_classes, weight_decay=1e-4):
        super().__init__()
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        # IMU deep branch
        self.imu_block1 = ResidualCNNBlock(imu_dim, 64, 3, dropout=0.3, Model=CoordAttention,  weight_decay=weight_decay)
        self.imu_block2 = ResidualCNNBlock(64, 128, 5, dropout=0.3, Model=CoordAttention, weight_decay=weight_decay)
        # BiGRU
        self.bigru = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        self.gru_dropout = nn.Dropout(0.4)
        # Attention
        self.attention = AttentionLayer(256)  # 128*2 for bidirectional
        self.mlp_attention = MLPAttention(256)  # MLP attention for final context aggregation
        
        # Dense layers
        self.dense1 = nn.Linear(256, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)
        
    def forward(self, imu, thm_tof, demo):
        imu = imu.transpose(1, 2)  # (batch, imu_dim, seq_len)
        # IMU branch
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)
        merged = x1.transpose(1, 2)  # (batch, seq_len, 128)
        # BiGRU 
        gru_out, _ = self.bigru(merged)
        gru_out = self.gru_dropout(gru_out)
        # Attention
        attended = self.mlp_attention(gru_out)
        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        # Classification
        logits = self.classifier(x)
        return logits


def apply_label_smoothing(targets, num_classes, smoothing_factor=0.1):
    """
    Apply label smoothing to convert hard targets to soft targets.
    Args:
        targets (torch.Tensor): Hard label indices
        num_classes (int): Total number of classes
        smoothing_factor (float): Smoothing factor, typically a small value like 0.1 
    Returns:
        torch.Tensor: Smoothed label distribution (batch_size, num_classes)
    """
    # Create a tensor of zeros with shape [batch_size, num_classes]
    smoothed_labels = torch.zeros(targets.size(0), num_classes, device=targets.device)
    # Fill in the tensor with the smoothing value
    smoothed_labels.fill_(smoothing_factor / (num_classes - 1))
    # Set the correct class with the main probability mass
    smoothed_labels.scatter_(1, targets.unsqueeze(1), 1.0 - smoothing_factor)
    return smoothed_labels


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, num_classes,
                   warmup_epochs=5, total_epochs=100, label_smoothing=0.1, eta_min=1e-6):
    """
    训练一个 epoch，支持标签平滑和两阶段学习率调度（线性warmup + 余弦退火）
    :param warmup_epochs: warmup阶段的总epoch数
    :param total_epochs: 整个训练过程的总epoch数
    :param label_smoothing: 标签平滑系数，设为 0 则不使用标签平滑
    :param eta_min: 余弦退火的最小学习率
    """    
    model.train()
    running_loss = 0.0
    train_preds = []
    train_targets = []
    
    # 根据是否使用标签平滑选择不同的损失函数处理
    use_label_smoothing = label_smoothing > 0
    
    # 获取基础学习率
    base_lr = optimizer.defaults['lr']  # 基础学习率
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)):
        # ======== 学习率调整 (batch 级别) ========
        # 计算当前训练进度（以epoch为单位，包含小数部分）
        current_iter = epoch * len(dataloader) + batch_idx
        current_epoch = current_iter / len(dataloader)  # 转换为小数形式的epoch
        # 阶段1: 线性warmup
        if current_epoch < warmup_epochs:
            # 线性warmup：从0.001 * base_lr 增加到 base_lr
            lr = base_lr * (0.001 + (current_epoch / warmup_epochs) * 0.999)
        # 阶段2: 余弦退火
        else:
            # 余弦退火公式
            cos_factor = 0.5 * (1 + math.cos(math.pi * (current_epoch - warmup_epochs) / (total_epochs - warmup_epochs)))
            lr = eta_min + (base_lr - eta_min) * cos_factor
        # 应用计算得到的学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # ======== 训练步骤 ========
        imu = batch['imu'].to(device)
        thm_tof = batch['thm_tof'].to(device)
        demo = batch['demo'].to(device)
        labels = batch['gesture_labels'].to(device)
        phase_labels = batch['phase_labels'].to(device)  # 阶段标签
        
        optimizer.zero_grad(set_to_none=True)  # 使用set_to_none=True更高效
        outputs = model(imu, thm_tof, demo)
        # ======== 应用标签平滑 ========
        if use_label_smoothing:
            # 将硬标签转换为软标签
            soft_labels = apply_label_smoothing(labels, num_classes, label_smoothing)
            # 使用交叉熵损失函数 (对于软标签，通常直接使用 log_softmax + sum)
            loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(outputs, dim=1),
                soft_labels,
                reduction='batchmean'
            )
        else:
            # 使用原始损失函数
            loss = criterion(outputs, labels)
        loss.backward()
        # 添加梯度裁剪防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        # ======== 收集训练指标 ========
        # 显式分离张量，减少内存占用
        preds = outputs.argmax(dim=1).detach().cpu().numpy()
        targets = labels.cpu().numpy()
        train_preds.extend(preds)
        train_targets.extend(targets)
        running_loss += loss.item()
    
    # ======== 计算训练指标 ========
    avg_loss = running_loss / len(dataloader)
    f1, f1_binary, f1_macro = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': dataloader.dataset.gesture_encoder.inverse_transform(train_targets)}),
        pd.DataFrame({'gesture': dataloader.dataset.gesture_encoder.inverse_transform(train_preds)}))
    
    return avg_loss, f1, f1_binary, f1_macro


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    val_preds = []
    val_targets = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Val]", leave=False):
            imu = batch['imu'].to(device)
            thm_tof = batch['thm_tof'].to(device)
            demo = batch['demo'].to(device)
            labels = batch['gesture_labels'].to(device)
            
            outputs = model(imu, thm_tof, demo)
            loss = criterion(outputs, labels)
            
            # 显式分离张量
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = labels.cpu().numpy()
            val_preds.extend(preds)
            val_targets.extend(targets)
            running_loss += loss.item()

    
    avg_loss = running_loss / len(dataloader)
    f1, f1_binary, f1_macro = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': dataloader.dataset.gesture_encoder.inverse_transform(val_targets)}),
        pd.DataFrame({'gesture': dataloader.dataset.gesture_encoder.inverse_transform(val_preds)}))
    
    return avg_loss, f1, f1_binary, f1_macro


TRAIN = False

if TRAIN == True:
    # 日志设置
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "imutrain.log")
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    
    metadata = pd.read_csv('/kaggle/input/cmi-data/data/train_labels.csv')
    NUM_FOLDS = 5
    SEED = 2
    scale_path='CMI_data/data/scalers'
    BATCH_SIZE = 32
    sequence_dir='/kaggle/input/cmi-data/data/train_sequences'
    label_dir='/kaggle/input/cmi-data/data/phase_labels'
    max_len = 256
    LR = 1e-3
    LR_min = 1e-6
    EPOCHS = 100
    patience = 10
    early_stopping_patience = 25  # 新增：早停策略的轮数
    model_path = "IMUMODEL"
    num_workers = 4
    warmup_epochs = 3  # 新增：warmup阶段的轮数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    print(f"▶ imports ready · pytorch {torch.__version__} · device: {device}")
    set_seed(SEED)  # 设置随机种子
    # 记录训练种子到日志
    logging.info(f"Training with seed: {SEED}")
    print(f"Training with seed: {SEED}")
    sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    Path(model_path).mkdir(parents=True, exist_ok=True)
    labels = metadata['gesture'].values
    sequences = metadata['sequence_id'].tolist()
    subjects = metadata['subject'].tolist()
    splits = []
    
    all_fold_f1_scores = []  # 存储每个fold的最佳验证F1分数
    
    for fold, (train_idx, val_idx) in enumerate(sgkf.split(sequences, labels, subjects)):
        print(f"\n=== Starting Fold {fold+1}/{NUM_FOLDS} ===")
        logging.info(f"=== Starting Fold {fold+1}/{NUM_FOLDS} ===")
    
    
        # 获取当前fold的训练集和验证集元数据
        train_metadata = metadata.iloc[train_idx].reset_index(drop=True)
        val_metadata = metadata.iloc[val_idx].reset_index(drop=True)
    
        # 创建数据集
        train_dataset = GestureDataset(sequence_dir=sequence_dir, label_dir=label_dir,
            metadata_df=train_metadata,max_len=max_len,is_train=True, scale_path=scale_path,use_augmentation=True
            )    
        val_dataset = GestureDataset(sequence_dir=sequence_dir, label_dir=label_dir,
            metadata_df=val_metadata,max_len=max_len,is_train=False, scale_path=scale_path
            )
        # 创建数据加载器
        train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True,
            collate_fn=collate_fn,num_workers=num_workers,pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False,
            collate_fn=collate_fn,num_workers=num_workers, pin_memory=True, drop_last=False)
        
        model = IMUOnlyModel(20, 325, n_classes=len(train_dataset.gesture_encoder.classes_))
        optimizer = Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()
        
        model.to(device)
        best_val_f1 = 0.0
        epochs_without_improvement = 0  # 用于跟踪没有改善的轮数
        # 计算当前 fold 的总 warmup 步数
        total_warmup_steps = warmup_epochs * len(train_loader)
        global_step = 0  # 全局步数计数器
    
        for epoch in range(EPOCHS):
            train_loss, train_f1, train_f1_binary, train_f1_macro = \
                train_one_epoch(model, train_loader, optimizer, criterion, device, epoch,
                num_classes=len(train_dataset.gesture_encoder.classes_),
                warmup_epochs=warmup_epochs, total_epochs=EPOCHS,
                label_smoothing=0.1, eta_min=LR_min)
            val_loss, val_f1, val_f1_binary, val_f1_macro = \
                validate(model, val_loader, criterion, device)
            logging.info(
                f"Fold {fold+1}, Epoch {epoch+1}/{EPOCHS}, LR: {optimizer.param_groups[0]['lr']:.6f}\n"
                f"Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}, Train F1 binary: {train_f1_binary:.4f}, Train F1 macro: {train_f1_macro:.4f}\n"
                f"Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}, Val F1 binary: {val_f1_binary:.4f}, Val F1 macro: {val_f1_macro:.4f}"
            )
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                epochs_without_improvement = 0  # 重置计数器
                print(f"Fold {fold+1}, Epoch {epoch+1}: 新的最佳验证F1分数: {best_val_f1:.4f}")
                logging.info(f"Fold {fold+1}, Epoch {epoch+1}: 新的最佳验证F1分数: {best_val_f1:.4f}")
                # 保存最佳模型
                model_save_path = os.path.join(model_path, f"model_fold{fold+1}_best.pth")
                torch.save(model.state_dict(), model_save_path)
            else:
                epochs_without_improvement += 1
                logging.info(f"Epochs without improvement: {epochs_without_improvement}")
            
            # 早停策略检查
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping triggered! No improvement for {early_stopping_patience} epochs.")
                logging.info(f"Early stopping triggered after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)")
                break
            # 保存模型
            model_save_path = os.path.join(model_path, f"model_fold{fold+1}_last.pth")
            torch.save(model.state_dict(), model_save_path)
        # ======== 存储当前fold的最佳分数 ========
        all_fold_f1_scores.append(best_val_f1)
        print(f"Fold {fold+1} 完成! 最佳验证F1: {best_val_f1:.4f}")
        logging.info(f"Fold {fold+1} 完成! 最佳验证F1: {best_val_f1:.4f}")
    
    
        # 删除所有相关引用
        del model, optimizer, train_loader, val_loader, train_dataset, val_dataset
        gc.collect()
        torch.cuda.empty_cache()  # 清空GPU缓存
        time.sleep(2)  # 等待资源释放
    
    # ======== 所有fold完成后计算平均分数 ========
    if all_fold_f1_scores:
        average_f1 = sum(all_fold_f1_scores) / len(all_fold_f1_scores)
        print(f"\n===== 最终结果 =====")
        print(f"平均F1分数: {average_f1:.4f}")
        print("各Fold分数详情:")
        for i, score in enumerate(all_fold_f1_scores):
            print(f"  Fold {i+1}: {score:.4f}")
        logging.info(f"\n===== 交叉验证最终结果 =====")
        logging.info(f"平均F1分数: {average_f1:.4f}")
        logging.info("各Fold分数详情:")
        for i, score in enumerate(all_fold_f1_scores):
            logging.info(f"  Fold {i+1}: {score:.4f}")


# 2025-07-09 11:25:00 [INFO] 平均F1分数: 0.7858
# 2025-07-09 11:25:00 [INFO] 各Fold分数详情:
# 2025-07-09 11:25:00 [INFO]   Fold 1: 0.7734
# 2025-07-09 11:25:00 [INFO]   Fold 2: 0.8004
# 2025-07-09 11:25:00 [INFO]   Fold 3: 0.7856
# 2025-07-09 11:25:00 [INFO]   Fold 4: 0.7701
# 2025-07-09 11:25:00 [INFO]   Fold 5: 0.7994





import numpy as np
import torch
import polars as pl
import threading
import pandas as pd  # 加载csv文件时需要
import glob
import os

# 全局变量
DM = None
IMU_MODELS = []
TOF_MODELS = []
GESTURE_CLASSES = None
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
TRAIN_DS = None
INITIALIZED = False
INIT_LOCK = threading.Lock()

def load_models(models_dir, model_class, n_classes, device):
    """从指定文件夹加载所有模型文件，返回模型列表"""
    model_paths = sorted(glob.glob(os.path.join(models_dir, "model_fold*_best.pth")))
    if not model_paths:
        raise FileNotFoundError(f"No models found in {models_dir}")
    models = []
    for path in model_paths:
        model = model_class(20, 325, n_classes=n_classes).to(device)
        model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
        model.eval()
        models.append(model)
    return models




def init_infer_module(
    sequence_dir='/kaggle/input/cmi-data/data/train_sequences',
    label_dir='/kaggle/input/cmi-data/data/phase_labels',
    metadata_path='/kaggle/input/cmi-data/data/train_labels.csv',
    scale_path='cmi-data/data/scalers',
    imu_models_dir='IMUMODEL',
):

    
    global DM, IMU_MODELS, TOF_MODELS, GESTURE_CLASSES, TRAIN_DS, INITIALIZED

    if TRAIN == False:
        imu_models_dir='/kaggle/input/imuonly/pytorch/imu-only-model/1'
    
    if INITIALIZED:
        return

    with INIT_LOCK:
        if INITIALIZED:
            return
        print("Initializing inference module...")

        # 数据集准备
        TRAIN_DS = GestureDataset(
            sequence_dir=sequence_dir, 
            label_dir=label_dir,
            metadata_df=pd.read_csv(metadata_path),
            max_len=256,
            is_train=True,
            scale_path=scale_path
        )
        GESTURE_CLASSES = len(TRAIN_DS.gesture_encoder.classes_)

        # 加载IMU模型
        IMU_MODELS = load_models(
            imu_models_dir, 
            IMUOnlyModel, 
            n_classes=GESTURE_CLASSES, 
            device=DEVICE
        )


        print(f"Inference module initialized: {len(IMU_MODELS)} IMU models.")
        INITIALIZED = True

# 特征列表
TIME_SERIES_FEATURES = [
    'acc_x', 'acc_y', 'acc_z',
    'rot_w', 'rot_x', 'rot_y', 'rot_z',
    'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'
]

TOF_FEATURES = [f'tof_{sensor}_v{pixel}' for sensor in range(1, 6) for pixel in range(64)]

STATIC_FEATURES = [
    'adult_child', 'age', 'sex', 'handedness',
    'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'
]


# ============================================================================
# Interpolation methods matching the training pipeline (updated)
# ============================================================================

def interpolate_thm_column(col):
    """
    Temperature sensor (thm) interpolation matching training pipeline:
    1. Apply linear interpolation (both directions)
    2. Fill remaining NaNs with 0
    """
    # 创建pandas Series以利用内置插值功能
    s = pd.Series(col)
    # 应用线性插值（双向）,剩余NaN填充0
    s = s.interpolate(method='linear', limit_direction='both').fillna(0)
    return s.values

def interpolate_imu_column(col):
    """
    IMU sensor (acc, rot) interpolation matching training pipeline:
    1. Forward fill (ffill)
    2. Backward fill (bfill)
    3. Fill remaining NaNs with 0
    """
    # 创建pandas Series以利用内置填充功能
    s = pd.Series(col)
    # 前向填充,后向填充,剩余NaN填充0
    s = s.ffill().bfill().fillna(0)
    return s.values

def interpolate_tof_column(col):
    """
    TOF sensor interpolation matching training pipeline:
    """
    # 创建pandas Series以利用内置填充功能
    s = pd.Series(col)
    # 前向填充,后向填充,剩余NaN填充0
    s = s.ffill().bfill().fillna(0)
    return s.values

def interpolate_sequence_polars(sequence, imu_features, thm_features, tof_features):
    """Apply consistent interpolation to a Polars DataFrame, matching training pipeline"""
    # Convert to pandas for processing
    df_pd = sequence.to_pandas()
    # 应用IMU传感器插值
    for feat in imu_features:
        if feat in df_pd.columns:
            df_pd[feat] = interpolate_imu_column(df_pd[feat])
    # 应用温度传感器插值
    for feat in thm_features:
        if feat in df_pd.columns:
            df_pd[feat] = interpolate_thm_column(df_pd[feat])
    # 应用TOF传感器插值
    for feat in tof_features:
        if feat in df_pd.columns:
            df_pd[feat] = interpolate_tof_column(df_pd[feat])
    # Convert back to polars
    return pl.from_pandas(df_pd)

# ============================================================================
# Modified prediction function with consistent interpolation
# ============================================================================

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    sequence: Polars DataFrame, single time series data with all dynamic features
    demographics: Polars DataFrame, single record with demographic features
    return: predicted gesture label (string)
    """
    init_infer_module()  # Initialize inference module

    # Define feature groups
    IMU_FEATURES = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    THM_FEATURES = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
    TOF_FEATURES = [f'tof_{sensor}_v{pixel}' for sensor in range(1, 6) for pixel in range(64)]

    # 判断原始数据的thm和tof是否全为nan（未插值前）
    thm_all_nan = sequence.select(THM_FEATURES).to_pandas().isna().all().all()
    tof_all_nan = sequence.select(TOF_FEATURES).to_pandas().isna().all().all()

    # 1. Apply interpolation consistent with training pipeline
    sequence = interpolate_sequence_polars(
        sequence, 
        imu_features=IMU_FEATURES,
        thm_features=THM_FEATURES,
        tof_features=TOF_FEATURES
    )

    # 2. Process static features (fill nulls)
    demographics = demographics.fill_null(0)
    sequence = sequence.select(IMU_FEATURES + THM_FEATURES + TOF_FEATURES)
    demographics = demographics.select(STATIC_FEATURES)
    # 3. Convert to numpy arrays
    seq_np = sequence.to_numpy().astype(np.float32)
    demo_np = demographics.to_numpy().astype(np.float32)[0]
    # 4. Concatenate static features to each time step
    full_seq = np.concatenate([seq_np, np.tile(demo_np, (len(seq_np), 1))], axis=1)
    # 5. Feature engineering and normalization
    features = TRAIN_DS._feature_engineering(full_seq)
    norm_features = TRAIN_DS.normalize_features(features)
    # 5.1 分离特征
    tof_features = norm_features[:, -TRAIN_DS.tof_dim:]  # TOF特征
    imu_thm_demo_features = norm_features[:, :-TRAIN_DS.tof_dim]
    demo_features = imu_thm_demo_features[0, -7:]  # 人口统计数据在整个序列中相同
    thm_features = imu_thm_demo_features[:, :-7][:, -5:]  # 温度特征
    imu_features = imu_thm_demo_features[:, :-7][:, :-5] # IMU特征
    thm_tof_features = np.concatenate([thm_features, tof_features], axis=1)
    # 6. Convert to tensor
    imu_tensor = torch.tensor(imu_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    thm_tof_tensor = torch.tensor(thm_tof_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    demo_tensor = torch.tensor(demo_features, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    pad_len = 40
    imu_tensor = torch.cat([imu_tensor, torch.zeros(imu_tensor.shape[0], pad_len, imu_tensor.shape[2], device=imu_tensor.device)], dim=1)
    thm_tof_tensor = torch.cat([thm_tof_tensor, torch.zeros(thm_tof_tensor.shape[0], pad_len, thm_tof_tensor.shape[2], device=thm_tof_tensor.device)], dim=1)

    models = IMU_MODELS  # 使用IMU模型

    # 8. Model inference (ensemble prediction)
    with torch.no_grad():
        all_preds = []
        for model in models:
            outputs = model(imu_tensor, thm_tof_tensor, demo_tensor)
            probs = F.softmax(outputs, dim=1)
            all_preds.append(probs)
        
        avg_pred = torch.stack(all_preds).mean(dim=0)
        pred_idx = avg_pred.argmax(dim=1).item()
        pred_label = TRAIN_DS.gesture_encoder.inverse_transform([pred_idx])[0]
    
    print(f"Predicted gesture: {pred_label}")
    return str(pred_label)


import kaggle_evaluation.cmi_inference_server

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

