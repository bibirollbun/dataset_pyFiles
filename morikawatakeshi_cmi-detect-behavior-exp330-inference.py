import os
import random
import pickle
import polars as pl
import pandas as pd
import numpy as np

from scipy.spatial.transform import Rotation as R
from scipy.stats import skew, kurtosis


from sklearn.metrics import f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

import lightgbm as lgb
from catboost import CatBoostClassifier

import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from torch.optim import lr_scheduler, Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import warnings
warnings.filterwarnings('ignore')

# save
def save(model=None, model_name=None):
    with open(model_name, 'wb') as f:
        pickle.dump(model, f)
        
# load
def load(model_path=None):
    with open(model_path, mode='rb') as f:
            model = pickle.load(f)
    return model

def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

exp_paths = {
    # imu
    'exp068': '/kaggle/input/cmi-detect-behavior-exp068-train/',
    'exp070': '/kaggle/input/cmi-detect-behavior-exp070-train/',
    'exp092': '/kaggle/input/cmi-detect-behavior-exp092-train/',
    'exp093': '/kaggle/input/cmi-detect-behavior-exp093-train/',
    'exp104': '/kaggle/input/cmi-detect-behavior-exp104-train/',
    'exp117': '/kaggle/input/cmi-detect-behavior-exp117-train/',
    'exp326': '/kaggle/input/cmi-detect-behavior-exp326-train/',
    'exp329': '/kaggle/input/cmi-detect-behavior-exp329-train/',

    # imu,tof,thm
    'exp086': '/kaggle/input/cmi-detect-behavior-exp086-train/',
    'exp137': '/kaggle/input/cmi-detect-behavior-exp137-train/',
    'exp179': '/kaggle/input/cmi-detect-behavior-exp179-train/',
    'exp226': '/kaggle/input/cmi-detect-behavior-exp226-train/'
}

configs = {name: load(f'{path}config.pkl') for name, path in exp_paths.items()}


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

device


def calculate_angular_speed(angular_velocity):
    return np.linalg.norm(angular_velocity, axis=1)

def calculate_angular_acceleration(angular_velocity):
    angular_acc = np.diff(angular_velocity, axis=0)
    return angular_acc

def calculate_euler_angle_diff(quat_data):
    valid_mask = ~np.all(np.isclose(quat_data, 0), axis=1)

    cleaned_quat = quat_data.copy()
    for i in range(len(cleaned_quat)):
        if not valid_mask[i]:
            cleaned_quat[i] = cleaned_quat[i-1] if i > 0 else [0, 0, 0, 1]

    euler_angles = R.from_quat(cleaned_quat).as_euler('xyz', degrees=False)

    euler_diff = np.diff(euler_angles, axis=0)

    return euler_diff
    
def remove_gravity_from_acc(acc_data, rot_data):
    """
    Удаляет компоненту гравитации из данных акселерометра, используя данные кватерниона.
    acc_data: pd.DataFrame или np.array с колонками/столбцами ['acc_x', 'acc_y', 'acc_z']
    rot_data: pd.DataFrame или np.array с колонками/столбцами ['rot_x', 'rot_y', 'rot_z', 'rot_w'] (порядок важен для Scipy: x, y, z, w)
    Возвращает: np.array с линейным ускорением [linear_acc_x, linear_acc_y, linear_acc_z]
    """
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data # предполагаем, что это уже numpy array [N, 3]

    if isinstance(rot_data, pd.DataFrame):
        # Scipy ожидает кватернионы в порядке [x, y, z, w]
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data # предполагаем, что это уже numpy array [N, 4] в нужном порядке

    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    
    # Вектор гравитации в глобальной системе координат (приблизительно)
    # Предполагаем, что Z - это "вверх", поэтому гравитация действует вдоль -Z
    # или +Z в зависимости от системы координат IMU.
    # Если IMU дает ускорение свободного падения ~9.81 по Z в состоянии покоя, 
    # то гравитация в мировой системе [0,0,-9.81] будет преобразована в [0,0,9.81] в системе датчика.
    # Будем считать, что гравитация в системе датчика, когда он лежит плашмя, это [0,0,g]
    # и мы хотим ее вычесть.
    gravity_world = np.array([0, 0, 9.81]) # Стандартное значение g

    for i in range(num_samples):
        if np.all(np.isnan(quat_values[i])) or np.all(np.isclose(quat_values[i], 0)): # Проверка на NaN или нулевой кватернион
            # Если кватернион невалидный, просто используем сырое ускорение
            # или можно присвоить NaN/0, но это может потребовать доп. обработки
            linear_accel[i, :] = acc_values[i, :] 
            continue

        try:
            # Создаем объект Rotation из кватерниона
            rotation = R.from_quat(quat_values[i])
            # Вращаем вектор гравитации из мировой системы в систему координат сенсора
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True) # или inverse=False, зависит от конвенции кватерниона
            # Вычитаем компоненту гравитации
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except ValueError: # Например, если кватернион не нормирован
             linear_accel[i, :] = acc_values[i, :] # Фоллбэк
             
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200): # Assuming 200Hz sampling rate
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

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

            # Calculate the relative rotation
            delta_rot = rot_t.inv() * rot_t_plus_dt
            
            # Convert delta rotation to angular velocity vector
            # The rotation vector (Euler axis * angle) scaled by 1/dt
            # is a good approximation for small delta_rot
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            # If quaternion is invalid, angular velocity remains zero
            pass
            
    return angular_vel

def calculate_angular_distance(rot_data):
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = quat_values[i]
        q2 = quat_values[i+1]

        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0 # Или np.nan, в зависимости от желаемого поведения
            continue
        try:
            # Преобразование кватернионов в объекты Rotation
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            # Вычисление углового расстояния: 2 * arccos(|real(p * q*)|)
            # где p* - сопряженный кватернион q
            # В scipy.spatial.transform.Rotation, r1.inv() * r2 дает относительное вращение.
            # Угол этого относительного вращения - это и есть угловое расстояние.
            relative_rotation = r1.inv() * r2
            
            # Угол rotation vector соответствует угловому расстоянию
            # Норма rotation vector - это угол в радианах
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0 # В случае недействительных кватернионов
            pass
            
    return angular_dist
    
def pad_sequences_torch(sequences, maxlen, padding='post', truncating='post', value=0.0):
    """PyTorch equivalent of Keras pad_sequences"""
    result = []
    for seq in sequences:
        seq = np.array(seq)
        T, D = seq.shape
        if T == maxlen:
            result.append(seq)
            
        elif T < maxlen:
            pad_len = maxlen - T
            pad = np.full((pad_len, D), value)
            seq_padded = np.concatenate([seq, pad], axis=0)
            result.append(seq_padded)
        else:  # T > maxlen → downsample by average
            indices = np.linspace(0, T, num=maxlen + 1, dtype=int)
            compressed = []
            for i in range(maxlen):
                segment = seq[indices[i]:indices[i+1]]
                compressed.append(np.mean(segment, axis=0))
            result.append(np.stack(compressed))
            
    return np.array(result, dtype=np.float32)

def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    """Normalizes and cleans the time series sequence"""
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

class MixupDataset(Dataset):
    """Dataset with MixUp augmentation"""
    def __init__(self, X, y, alpha=0.2):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.alpha = alpha
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        
        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            rand_idx = np.random.randint(0, len(self.X))
            x_rand, y_rand = self.X[rand_idx], self.y[rand_idx]
            
            x = lam * x + (1 - lam) * x_rand
            y = lam * y + (1 - lam) * y_rand
            
        return x, y



def Imu_FE(df):
    linear_accel_list = []
    for _, group in df.groupby('sequence_id'):
        acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']] # Порядок x,y,z,w для Scipy
        linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
        linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))
    
    df_linear_accel = pd.concat(linear_accel_list)
    df = pd.concat([df, df_linear_accel], axis=1)

    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))

    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)  

    angular_vel_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group)
        angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))


    df_angular_vel = pd.concat(angular_vel_list)
    df = pd.concat([df, df_angular_vel], axis=1)


    angular_distance_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_dist_group = calculate_angular_distance(rot_data_group)
        angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))
    
    df_angular_distance = pd.concat(angular_distance_list)
    df = pd.concat([df, df_angular_distance], axis=1)

    angular_speed_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']]
        angular_dist_group = calculate_angular_speed(rot_data_group)
        angular_speed_list.append(pd.DataFrame(angular_dist_group, columns=['angular_speed'], index=group.index))
    
    df_angular_speed = pd.concat(angular_speed_list)
    df = pd.concat([df, df_angular_speed], axis=1)

    angular_acc_list = []

    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']]
        angular_dist_group = calculate_angular_acceleration(rot_data_group)

        new_index = group.index[1:]

        acc_df = pd.DataFrame(
            angular_dist_group,
            columns=['ang_acc_x', 'ang_acc_y', 'ang_acc_z'],
            index=new_index
        )

        angular_acc_list.append(acc_df)

    df_angular_acc = pd.concat(angular_acc_list)
    df_angular_acc = df_angular_acc.reindex(df.index)
    df = pd.concat([df, df_angular_acc], axis=1)

    df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
    df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)

    df['linear_accx_jerk'] = df.groupby('sequence_id')['linear_acc_x'].diff().fillna(0)
    df['linear_accy_jerk'] = df.groupby('sequence_id')['linear_acc_y'].diff().fillna(0)
    df['linear_accz_jerk'] = df.groupby('sequence_id')['linear_acc_z'].diff().fillna(0)

    df['linear_accx_rolling_mean5'] = df.groupby('sequence_id')['linear_acc_x'].rolling(5).mean().reset_index(level=0, drop=True).fillna(0)
    df['linear_accy_rolling_mean5'] = df.groupby('sequence_id')['linear_acc_y'].rolling(5).mean().reset_index(level=0, drop=True).fillna(0)
    df['linear_accz_rolling_mean5'] = df.groupby('sequence_id')['linear_acc_z'].rolling(5).mean().reset_index(level=0, drop=True).fillna(0)

    
    df['linear_accx_rolling_std3'] = df.groupby('sequence_id')['linear_acc_x'].rolling(3).std().reset_index(level=0, drop=True).fillna(0)
    df['linear_accy_rolling_std3'] = df.groupby('sequence_id')['linear_acc_y'].rolling(3).std().reset_index(level=0, drop=True).fillna(0)
    df['linear_accz_rolling_std3'] = df.groupby('sequence_id')['linear_acc_z'].rolling(3).std().reset_index(level=0, drop=True).fillna(0)

    df['linear_accx_rolling_min3'] = df.groupby('sequence_id')['linear_acc_x'].rolling(3).min().reset_index(level=0, drop=True).fillna(0)
    df['linear_accy_rolling_min3'] = df.groupby('sequence_id')['linear_acc_y'].rolling(3).min().reset_index(level=0, drop=True).fillna(0)
    df['linear_accz_rolling_min3'] = df.groupby('sequence_id')['linear_acc_z'].rolling(3).min().reset_index(level=0, drop=True).fillna(0)
    
    df['linear_accx_rolling_max3'] = df.groupby('sequence_id')['linear_acc_x'].rolling(3).max().reset_index(level=0, drop=True).fillna(0)
    df['linear_accy_rolling_max3'] = df.groupby('sequence_id')['linear_acc_y'].rolling(3).max().reset_index(level=0, drop=True).fillna(0)
    df['linear_accz_rolling_max3'] = df.groupby('sequence_id')['linear_acc_z'].rolling(3).max().reset_index(level=0, drop=True).fillna(0)
    
    df['angular_vel_x_rolling_mean3'] = df.groupby('sequence_id')['angular_vel_x'].rolling(3).mean().reset_index(level=0, drop=True).fillna(0)
    df['angular_vel_y_rolling_mean3'] = df.groupby('sequence_id')['angular_vel_y'].rolling(3).mean().reset_index(level=0, drop=True).fillna(0)
    df['angular_vel_z_rolling_mean3'] = df.groupby('sequence_id')['angular_vel_z'].rolling(3).mean().reset_index(level=0, drop=True).fillna(0)
    df['angular_distance_rolling_mean3'] = df.groupby('sequence_id')['angular_distance'].rolling(3).mean().reset_index(level=0, drop=True).fillna(0)
    
    df['angular_vel_x_rolling_std3'] = df.groupby('sequence_id')['angular_vel_x'].rolling(3).std().reset_index(level=0, drop=True).fillna(0)
    df['angular_vel_y_rolling_std3'] = df.groupby('sequence_id')['angular_vel_y'].rolling(3).std().reset_index(level=0, drop=True).fillna(0)
    df['angular_vel_z_rolling_std3'] = df.groupby('sequence_id')['angular_vel_z'].rolling(3).std().reset_index(level=0, drop=True).fillna(0)
    df['angular_distance_rolling_std3'] = df.groupby('sequence_id')['angular_distance'].rolling(3).std().reset_index(level=0, drop=True).fillna(0)

    df['angular_vel_x_dominance'] = df['angular_vel_x'] / (df['angular_vel_x'] + df['angular_vel_y'] + df['angular_vel_z'] + 1e-8)
    df['angular_vel_y_dominance'] = df['angular_vel_y'] / (df['angular_vel_x'] + df['angular_vel_y'] + df['angular_vel_z'] + 1e-8)
    df['angular_vel_z_dominance'] = df['angular_vel_z'] / (df['angular_vel_x'] + df['angular_vel_y'] + df['angular_vel_z'] + 1e-8)

    df['angular_vel_x_rolling_min3'] = df.groupby('sequence_id')['angular_vel_x'].rolling(3).min().reset_index(level=0, drop=True)
    df['angular_vel_y_rolling_min3'] = df.groupby('sequence_id')['angular_vel_y'].rolling(3).min().reset_index(level=0, drop=True)
    df['angular_vel_z_rolling_min3'] = df.groupby('sequence_id')['angular_vel_z'].rolling(3).min().reset_index(level=0, drop=True)
    df['angular_distance_rolling_min3'] = df.groupby('sequence_id')['angular_distance'].rolling(3).min().reset_index(level=0, drop=True)

    df['angular_vel_x_rolling_max3'] = df.groupby('sequence_id')['angular_vel_x'].rolling(3).max().reset_index(level=0, drop=True)
    df['angular_vel_y_rolling_max3'] = df.groupby('sequence_id')['angular_vel_y'].rolling(3).max().reset_index(level=0, drop=True)
    df['angular_vel_z_rolling_max3'] = df.groupby('sequence_id')['angular_vel_z'].rolling(3).max().reset_index(level=0, drop=True)
    df['angular_distance_rolling_max3'] = df.groupby('sequence_id')['angular_distance'].rolling(3).max().reset_index(level=0, drop=True)

    df['angular_vel_x_diff'] = df.groupby('sequence_id')['angular_vel_x'].diff().fillna(0)
    df['angular_vel_y_diff'] = df.groupby('sequence_id')['angular_vel_y'].diff().fillna(0)
    df['angular_vel_z_diff'] = df.groupby('sequence_id')['angular_vel_z'].diff().fillna(0)
    
    df['linear_accxy_mag'] = np.sqrt(df['linear_acc_x']**2+df['linear_acc_y']**2)
    df['linear_accyz_mag'] = np.sqrt(df['linear_acc_y']**2+df['linear_acc_z']**2)
    df['linear_acczx_mag'] = np.sqrt(df['linear_acc_x']**2+df['linear_acc_z']**2)

    df['linear_accxy_jerk_mag'] = np.sqrt(df['linear_accx_jerk']**2+df['linear_accy_jerk']**2)
    df['linear_accyz_jerk_mag'] = np.sqrt(df['linear_accy_jerk']**2+df['linear_accz_jerk']**2)
    df['linear_acczx_jerk_mag'] = np.sqrt(df['linear_accx_jerk']**2+df['linear_accz_jerk']**2)

    df['angular_vel_xy_mag'] = np.sqrt(df['angular_vel_x']**2+df['angular_vel_y']**2)
    df['angular_vel_yz_mag'] = np.sqrt(df['angular_vel_y']**2+df['angular_vel_z']**2)
    df['angular_vel_zx_mag'] = np.sqrt(df['angular_vel_x']**2+df['angular_vel_z']**2)

    df['ang_acc_xy_ang'] = np.sqrt(df['ang_acc_x']**2+df['ang_acc_y']**2)
    df['ang_acc_yz_ang'] = np.sqrt(df['ang_acc_y']**2+df['ang_acc_z']**2)
    df['ang_acc_zx_ang'] = np.sqrt(df['ang_acc_x']**2+df['ang_acc_z']**2)

    df['linear_accxy_rolling_mean5_mag'] = np.sqrt(df['linear_accx_rolling_mean5']**2+df['linear_accy_rolling_mean5']**2)
    df['linear_accyz_rolling_mean5_mag'] = np.sqrt(df['linear_accy_rolling_mean5']**2+df['linear_accz_rolling_mean5']**2)
    df['linear_acczx_rolling_mean5_mag'] = np.sqrt(df['linear_accx_rolling_mean5']**2+df['linear_accz_rolling_mean5']**2)
    df['linear_acc_rolling_mean5_mag'] = np.sqrt(df['linear_accx_rolling_mean5']**2+df['linear_accy_rolling_mean5']**2+df['linear_accz_rolling_mean5']**2)

    df['linear_accxy_rolling_std3_mag'] = np.sqrt(df['linear_accx_rolling_std3']**2+df['linear_accy_rolling_std3']**2)
    df['linear_accyz_rolling_std3_mag'] = np.sqrt(df['linear_accy_rolling_std3']**2+df['linear_accz_rolling_std3']**2)
    df['linear_acczx_rolling_std3_mag'] = np.sqrt(df['linear_accz_rolling_std3']**2+df['linear_accx_rolling_std3']**2)
    df['linear_acc_rolling_std3_mag'] = np.sqrt(df['linear_accx_rolling_std3']**2+df['linear_accy_rolling_std3']**2+df['linear_accz_rolling_std3']**2)

    df['linear_accxy_rolling_min3_mag'] = np.sqrt(df['linear_accx_rolling_min3']**2+df['linear_accy_rolling_min3']**2)  
    df['linear_accyz_rolling_min3_mag'] = np.sqrt(df['linear_accy_rolling_min3']**2+df['linear_accz_rolling_min3']**2)  
    df['linear_acczx_rolling_min3_mag'] = np.sqrt(df['linear_accz_rolling_min3']**2+df['linear_accx_rolling_min3']**2)  
    df['linear_acc_rolling_min3_mag'] = np.sqrt(df['linear_accx_rolling_min3']**2+df['linear_accy_rolling_min3']**2+df['linear_accz_rolling_min3']**2)  

    total = df['linear_acc_x']+df['linear_acc_y']+df['linear_acc_z']+1e-8
    df['linear_acc_x_div'] = df['linear_acc_x'] / total
    df['linear_acc_y_div'] = df['linear_acc_y'] / total
    df['linear_acc_z_div'] = df['linear_acc_z'] / total

    total = df['ang_acc_x']+df['ang_acc_y']+df['ang_acc_z']+1e-8
    df['ang_acc_x_div'] = df['ang_acc_x'] / total
    df['ang_acc_y_div'] = df['ang_acc_y'] / total
    df['ang_acc_z_div'] = df['ang_acc_z'] / total

    total = df['angular_vel_x_diff']+df['angular_vel_y_diff']+df['angular_vel_z_diff']+1e-8
    df['angular_vel_x_diff_div'] = df['angular_vel_x_diff'] / total
    df['angular_vel_y_diff_div'] = df['angular_vel_y_diff'] / total
    df['angular_vel_z_diff_div'] = df['angular_vel_z_diff'] / total
    return df

def ImuTofThm_FE(df):

    linear_accel_list = []
    for _, group in df.groupby('sequence_id'):
        acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']] # Порядок x,y,z,w для Scipy
        linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
        linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))
    
    df_linear_accel = pd.concat(linear_accel_list)
    df = pd.concat([df, df_linear_accel], axis=1)

    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))

    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)  

    angular_vel_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group)
        angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))


    df_angular_vel = pd.concat(angular_vel_list)
    df = pd.concat([df, df_angular_vel], axis=1)

    angular_distance_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_dist_group = calculate_angular_distance(rot_data_group)
        angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))
    
    df_angular_distance = pd.concat(angular_distance_list)
    df = pd.concat([df, df_angular_distance], axis=1)

    angular_speed_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']]
        angular_dist_group = calculate_angular_speed(rot_data_group)
        angular_speed_list.append(pd.DataFrame(angular_dist_group, columns=['angular_speed'], index=group.index))
    
    df_angular_speed = pd.concat(angular_speed_list)
    df = pd.concat([df, df_angular_speed], axis=1)

    angular_acc_list = []

    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']]
        angular_dist_group = calculate_angular_acceleration(rot_data_group)

        new_index = group.index[1:]

        acc_df = pd.DataFrame(
            angular_dist_group,
            columns=['ang_acc_x', 'ang_acc_y', 'ang_acc_z'],
            index=new_index
        )

        angular_acc_list.append(acc_df)

    df_angular_acc = pd.concat(angular_acc_list)
    df_angular_acc = df_angular_acc.reindex(df.index)
    df = pd.concat([df, df_angular_acc], axis=1)

    df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
    df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)

    df['linear_accx_jerk'] = df.groupby('sequence_id')['linear_acc_x'].diff().fillna(0)
    df['linear_accy_jerk'] = df.groupby('sequence_id')['linear_acc_y'].diff().fillna(0)
    df['linear_accz_jerk'] = df.groupby('sequence_id')['linear_acc_z'].diff().fillna(0)

    df['linear_accx_rolling_mean5'] = df.groupby('sequence_id')['linear_acc_x'].rolling(5).mean().reset_index(level=0, drop=True).fillna(0)
    df['linear_accy_rolling_mean5'] = df.groupby('sequence_id')['linear_acc_y'].rolling(5).mean().reset_index(level=0, drop=True).fillna(0)
    df['linear_accz_rolling_mean5'] = df.groupby('sequence_id')['linear_acc_z'].rolling(5).mean().reset_index(level=0, drop=True).fillna(0)

    df['linear_accx_rolling_std3'] = df.groupby('sequence_id')['linear_acc_x'].rolling(3).std().reset_index(level=0, drop=True).fillna(0)
    df['linear_accy_rolling_std3'] = df.groupby('sequence_id')['linear_acc_y'].rolling(3).std().reset_index(level=0, drop=True).fillna(0)
    df['linear_accz_rolling_std3'] = df.groupby('sequence_id')['linear_acc_z'].rolling(3).std().reset_index(level=0, drop=True).fillna(0)

    df['linear_accx_rolling_min3'] = df.groupby('sequence_id')['linear_acc_x'].rolling(3).min().reset_index(level=0, drop=True).fillna(0)
    df['linear_accy_rolling_min3'] = df.groupby('sequence_id')['linear_acc_y'].rolling(3).min().reset_index(level=0, drop=True).fillna(0)
    df['linear_accz_rolling_min3'] = df.groupby('sequence_id')['linear_acc_z'].rolling(3).min().reset_index(level=0, drop=True).fillna(0)

    df['linear_accx_rolling_max3'] = df.groupby('sequence_id')['linear_acc_x'].rolling(3).max().reset_index(level=0, drop=True).fillna(0)
    df['linear_accy_rolling_max3'] = df.groupby('sequence_id')['linear_acc_y'].rolling(3).max().reset_index(level=0, drop=True).fillna(0)
    df['linear_accz_rolling_max3'] = df.groupby('sequence_id')['linear_acc_z'].rolling(3).max().reset_index(level=0, drop=True).fillna(0)

    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        tof_data = df[pixel_cols].replace(-1, np.nan)
    
        df[f'tof_{i}_mean'] = tof_data.mean(axis=1)
        df[f'tof_{i}_std'] = tof_data.std(axis=1)
        df[f'tof_{i}_min'] = tof_data.min(axis=1)
        df[f'tof_{i}_max'] = tof_data.max(axis=1)

        tof_mode = 16
        region_size = 64 // tof_mode
        for r in range(tof_mode):
            region_data = tof_data.iloc[:, r*region_size : (r+1)*region_size]
            df[f'tof{tof_mode}_{i}_region_{r}_mean'] = region_data.mean(axis=1)
            df[f'tof{tof_mode}_{i}_region_{r}_std'] = region_data.std(axis=1)
            df[f'tof{tof_mode}_{i}_region_{r}_min'] = region_data.min(axis=1)
            df[f'tof{tof_mode}_{i}_region_{r}_max'] = region_data.max(axis=1)
    return df

def gbdt_feature_engineering(df):
    linear_accel_list = []
    for _, group in df.groupby('sequence_id'):
        acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']] # Порядок x,y,z,w для Scipy
        linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
        linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))
    
    df_linear_accel = pd.concat(linear_accel_list)
    df = pd.concat([df, df_linear_accel], axis=1)

    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))

    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)  

    angular_vel_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group)
        angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))


    df_angular_vel = pd.concat(angular_vel_list)
    df = pd.concat([df, df_angular_vel], axis=1)


    angular_distance_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_dist_group = calculate_angular_distance(rot_data_group)
        angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))
    
    df_angular_distance = pd.concat(angular_distance_list)
    df = pd.concat([df, df_angular_distance], axis=1)

    angular_speed_list = []
    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']]
        angular_dist_group = calculate_angular_speed(rot_data_group)
        angular_speed_list.append(pd.DataFrame(angular_dist_group, columns=['angular_speed'], index=group.index))
    
    df_angular_speed = pd.concat(angular_speed_list)
    df = pd.concat([df, df_angular_speed], axis=1)

    angular_acc_list = []

    for _, group in df.groupby('sequence_id'):
        rot_data_group = group[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']]
        angular_dist_group = calculate_angular_acceleration(rot_data_group)

        new_index = group.index[1:]

        acc_df = pd.DataFrame(
            angular_dist_group,
            columns=['ang_acc_x', 'ang_acc_y', 'ang_acc_z'],
            index=new_index
        )

        angular_acc_list.append(acc_df)

    df_angular_acc = pd.concat(angular_acc_list)
    df_angular_acc = df_angular_acc.reindex(df.index)
    df = pd.concat([df, df_angular_acc], axis=1)
    return df

def GBDT_IMU_FE(df, train=True):

    df = pl.from_pandas(df)

    agg_exprs = []
    for c in ['acc_x', 'acc_y','acc_z', 'rot_w', 'rot_x', 'rot_y', 
              'rot_z', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'acc_mag', 'rot_angle',
              'acc_mag_jerk', 'rot_angle_vel', 'angular_vel_x', 'angular_vel_y',
              'angular_vel_z', 'angular_distance', 'angular_speed', 'ang_acc_x',
              'ang_acc_y', 'ang_acc_z']:
        agg_exprs += [
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).var().alias(f"{c}_var"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).median().alias(f"{c}_q50"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).min().alias(f"{c}_min"),
            pl.col(c).first().alias(f"{c}_first"),
            pl.col(c).last().alias(f"{c}_last"),
            pl.col(c).quantile(0.25, "nearest").alias(f"{c}_t25"),
            pl.col(c).quantile(0.75, "nearest").alias(f"{c}_t75"),
            (pl.col(c).last() - pl.col(c).first()).alias(f"{c}_delta"),
            pl.corr("sequence_counter", c).alias(f"{c}_corr_time"),
            pl.col(c).diff().mean().alias(f"{c}_diff_mean"),
            pl.col(c).diff().std().alias(f"{c}_diff_std"),
            pl.col(c).skew().alias(f"{c}_skew"),
            pl.col(c).kurtosis().alias(f"{c}_kurt"),
            pl.col(c).diff().abs().gt(0).sum().alias(f"{c}_n_changes")
        ]
        agg_exprs += [
            pl.when(pl.col("sequence_counter") < 0.1 * pl.max("sequence_counter"))
              .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg1_mean"),
            pl.when(pl.col("sequence_counter") > 0.9 * pl.max("sequence_counter"))
              .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg3_mean"),
        ]

        for part_name, part_expr in [
            ('first_30pct', pl.col('sequence_counter') < pl.max('sequence_counter') * 0.3),
            ('middle_40pct', (pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.3) & (pl.col('sequence_counter') < pl.max('sequence_counter') * 0.7)),
            ('last_30pct', pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.7),
        ]:
            agg_exprs.extend([
                (pl.when(part_expr).then(pl.col(c))).mean().alias(f'{c}_mean_{part_name}'),
                (pl.when(part_expr).then(pl.col(c))).std().alias(f'{c}_std_{part_name}'),
                (pl.when(part_expr).then(pl.col(c))).diff().mean().alias(f'{c}_diff_mean_{part_name}'),
                (pl.when(part_expr).then(pl.col(c))).diff().std().alias(f'{c}_diff_std_{part_name}'),
                (pl.when(part_expr).then(pl.col(c))).quantile(0.25, "nearest").alias(f'{c}_t25_{part_name}'),
                (pl.when(part_expr).then(pl.col(c))).quantile(0.75, "nearest").alias(f'{c}_t75_{part_name}'),
            ])

    if train:
        agg_exprs += [pl.col("gesture").first().alias("gesture")]
    
    agg_exprs += [pl.col("subject").first().alias("subject")]
    agg_exprs += [pl.col("sequence_counter").max().alias('sequence_counter')]

    if train:
        agg_exprs += [pl.col('row_id').first().alias('row_id')]
        agg_exprs += [pl.col('orientation').first().alias('orientation')]
        agg_exprs += [pl.col("sequence_type").first().alias("sequence_type")]

    df = (
        df
        .group_by("sequence_id", maintain_order=True)
        .agg(agg_exprs)
        .to_pandas()
    )

    return df

def GBDT_IMU_TOF_THM_FE(df, train=True):

    df = pl.from_pandas(df)

    agg_exprs = []
    for c in ['acc_x', 'acc_y','acc_z', 'rot_w', 'rot_x', 'rot_y', 
              'rot_z', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'acc_mag', 'rot_angle',
              'acc_mag_jerk', 'rot_angle_vel', 'angular_vel_x', 'angular_vel_y',
              'angular_vel_z', 'angular_distance', 'angular_speed', 'ang_acc_x',
              'ang_acc_y', 'ang_acc_z'] + ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'] + [f'tof_{i+1}_v{j}' for i in range(5) for j in range(64)]:
        agg_exprs += [
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).var().alias(f"{c}_var"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).median().alias(f"{c}_q50"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).min().alias(f"{c}_min"),
            pl.col(c).first().alias(f"{c}_first"),
            pl.col(c).last().alias(f"{c}_last"),
            pl.col(c).quantile(0.25, "nearest").alias(f"{c}_t25"),
            pl.col(c).quantile(0.75, "nearest").alias(f"{c}_t75"),
            (pl.col(c).last() - pl.col(c).first()).alias(f"{c}_delta"),
            pl.corr("sequence_counter", c).alias(f"{c}_corr_time"),
            pl.col(c).diff().mean().alias(f"{c}_diff_mean"),
            pl.col(c).diff().std().alias(f"{c}_diff_std"),
            pl.col(c).skew().alias(f"{c}_skew"),
            pl.col(c).kurtosis().alias(f"{c}_kurt"),
            pl.col(c).diff().abs().gt(0).sum().alias(f"{c}_n_changes")
        ]
        agg_exprs += [
            pl.when(pl.col("sequence_counter") < 0.1 * pl.max("sequence_counter"))
              .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg1_mean"),
            pl.when(pl.col("sequence_counter") > 0.9 * pl.max("sequence_counter"))
              .then(pl.col(c)).otherwise(None).mean().alias(f"{c}_seg3_mean"),
        ]

        for part_name, part_expr in [
            ('first_30pct', pl.col('sequence_counter') < pl.max('sequence_counter') * 0.3),
            ('middle_40pct', (pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.3) & (pl.col('sequence_counter') < pl.max('sequence_counter') * 0.7)),
            ('last_30pct', pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.7),
        ]:
            agg_exprs.extend([
                (pl.when(part_expr).then(pl.col(c))).mean().alias(f'{c}_mean_{part_name}'),
                (pl.when(part_expr).then(pl.col(c))).std().alias(f'{c}_std_{part_name}'),
                (pl.when(part_expr).then(pl.col(c))).diff().mean().alias(f'{c}_diff_mean_{part_name}'),
                (pl.when(part_expr).then(pl.col(c))).diff().std().alias(f'{c}_diff_std_{part_name}'),
                (pl.when(part_expr).then(pl.col(c))).quantile(0.25, "nearest").alias(f'{c}_t25_{part_name}'),
                (pl.when(part_expr).then(pl.col(c))).quantile(0.75, "nearest").alias(f'{c}_t75_{part_name}'),
            ])

    
    agg_exprs += [pl.col("subject").first().alias("subject")]
    agg_exprs += [pl.col("sequence_counter").max().alias('sequence_counter')]


    df = (
        df
        .group_by("sequence_id", maintain_order=True)
        .agg(agg_exprs)
        .to_pandas()
    )
    return df


pad_len=127
# imu
exp068_scaler = load(f'{exp_paths["exp068"]}scaler.pkl')   
exp070_scaler = load(f'{exp_paths["exp070"]}scaler.pkl')
exp092_scaler = load(f'{exp_paths["exp092"]}scaler.pkl')
exp093_scaler = load(f'{exp_paths["exp093"]}scaler.pkl')
exp104_scaler = load(f'{exp_paths["exp104"]}scaler.pkl')
exp326_scaler = load(f'{exp_paths["exp326"]}scaler.pkl')
exp329_scaler = load(f'{exp_paths["exp329"]}scaler.pkl')

# imu, tof, thm
exp086_scaler = load(f'{exp_paths["exp086"]}scaler.pkl')
exp137_scaler = load(f'{exp_paths["exp137"]}scaler.pkl')

le = load('/kaggle/input/cmi-detect-behavior-gesture-encoder/Encoder_model.pickle')


import sys
sys.path.append("/kaggle/input/cmi-detect-behavior-model-structure")

# imu
from exp068 import exp068_model
from exp070 import exp070_model
from exp092 import exp092_model
from exp093 import exp093_model
from exp104 import exp104_model
from exp326 import exp326_model
from exp329 import exp329_model

# imu, tof, thm
from exp086 import exp086_model
from exp137 import exp137_model


def load_models(config, model_class, model_path, pad_len, device, extra_args=None):
    """
    共通のモデル読み込み関数

    config: モデル設定辞書
    model_class: モデルクラス
    model_path: 学習済みモデルファイルのパス（fold番号は後で追加）
    pad_len: シーケンス長
    device: 推論デバイス
    extra_args: 追加のキーワード引数（辞書）
    """
    models = []
    extra_args = extra_args or {}

    for i in range(config['n_splits']):
        # モデルインスタンス生成
        model = model_class(
            pad_len=pad_len,
            acc_dim=len(config['acc_cols']),
            rot_dim=len(config['rot_cols']),
            n_classes=len(config['target_le'].classes_),
            weight_decay=config['WD'],
            **extra_args
        )

        # 特殊な入力次元（imu_dim / tof_dim / thm_tof_cols）に対応
        if 'features' in config:
            model.imu_dim = len(config['features'])
        if 'thm_tof_cols' in config:
            model.tof_dim = len(config['thm_tof_cols'])

        # DataParallel & 重み読み込み
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(f"{model_path}{config['model_name']}_fold{i}.bin"))
        model.to(device)
        model.eval()

        models.append(model)

    return models


# === 実際の読み込み ===
# imu
exp068_models = load_models(
    config=configs['exp068'],
    model_class=exp068_model,
    model_path=exp_paths['exp068'],
    pad_len=pad_len,
    device=device,
    extra_args={
        'imu_dim': len(configs['exp068']['features']),
        'model_args': configs['exp068']['model_args']
    }
)

exp070_models = load_models(
    config=configs['exp070'],
    model_class=exp070_model,
    model_path=exp_paths['exp070'],
    pad_len=pad_len,
    device=device,
    extra_args={
        'imu_dim': len(configs['exp070']['features']),
        'model_args': configs['exp070']['model_args']
    }
)

exp092_models = load_models(
    config=configs['exp092'],
    model_class=exp092_model,
    model_path=exp_paths['exp092'],
    pad_len=pad_len,
    device=device,
    extra_args={
        'imu_dim': len(configs['exp092']['features']),
        'model_args': configs['exp092']['model_args']
    }
)

exp093_models = load_models(
    config=configs['exp093'],
    model_class=exp093_model,
    model_path=exp_paths['exp093'],
    pad_len=pad_len,
    device=device,
    extra_args={
        'imu_dim': len(configs['exp093']['features']),
        'model_args': configs['exp093']['model_args']
    }
)



exp104_models = []
for i in range(configs['exp104']['n_splits']):
    model = exp104_model(
        in_channels=len(configs['exp104']['features']),
        num_classes=len(configs['exp104']['target_le'].classes_),
        args=configs['exp104']['model_args']
    )
    
    model = nn.DataParallel(model)
    model.load_state_dict(
        torch.load(f'{exp_paths["exp104"]}{configs["exp104"]["model_name"]}_fold{i}.bin')
    )
    model.to(configs['exp104']['device'])
    model.eval()
    exp104_models.append(model)

exp117_features = np.load(f'{exp_paths["exp117"]}columns.npy', allow_pickle=True)
exp117_models = [load(f'{exp_paths["exp117"]}{configs["exp117"]["model_name"]}_fold{i}.pkl') for i in range(configs["exp117"]["n_splits"])]

exp326_models = load_models(
    config=configs['exp326'],
    model_class=exp326_model,
    model_path=exp_paths['exp326'],
    pad_len=pad_len,
    device=device,
    extra_args={
        'imu_dim': len(configs['exp326']['features'])
    } 
)

exp329_models = load_models(
    config=configs['exp329'],
    model_class=exp329_model,
    model_path=exp_paths['exp329'],
    pad_len=pad_len,
    device=device,
    extra_args={
        'imu_dim': len(configs['exp329']['features']),
        'model_args': configs['exp329']['model_args']
    }
)

# imu, tof, thm
exp086_models = load_models(
    config=configs['exp086'],
    model_class=exp086_model,
    model_path=exp_paths['exp086'],
    pad_len=pad_len,
    device=device,
    extra_args={
        'tof_dim': len(configs['exp086']['thm_tof_cols']),
        'args': configs['exp086']['model_args']
    }
)

exp137_models = load_models(
    config=configs['exp137'],
    model_class=exp137_model,
    model_path=exp_paths['exp137'],
    pad_len=pad_len,
    device=device,
    extra_args={
        'tof_dim': len(configs['exp137']['thm_tof_cols']),
        'args': configs['exp137']['model_args']
    }
)


exp179_features = np.load(f'{exp_paths["exp179"]}columns.npy', allow_pickle=True)
exp179_models = [load(f'{exp_paths["exp179"]}{configs["exp179"]["model_name"]}_fold{i}.pkl') for i in range(configs["exp179"]["n_splits"])]

exp226_features = np.load(f'{exp_paths["exp226"]}columns.npy', allow_pickle=True)
exp226_models = [load(f'{exp_paths["exp226"]}{configs["exp226"]["model_name"]}_fold{i}.pkl') for i in range(configs["exp226"]["n_splits"])]


tof_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'] + \
           [f'tof_{i+1}_v{j}' for i in range(5) for j in range(64)]

def run_inference(df_seq, fe_func, config, scaler, models):
    df_seq = fe_func(df_seq)
    mat = preprocess_sequence(df_seq, config['features'], scaler)

    pad = pad_sequences_torch([mat], maxlen=pad_len, padding='post', truncating='post')

    # inference
    preds = []
    for model in models:
        model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(pad).to(device)
            logits = model(x)
            logits = nn.Softmax(dim=1)(logits)
            preds.append(logits.detach().cpu().numpy())

    preds = np.array(preds)
    pred = preds.mean(axis=0)
    return pred

def gbdt_test_data(df_seq, df_demo, fe_fuc0, fe_fuc1):
    df_seq = fe_fuc0(df_seq)
    df_seq = fe_fuc1(df_seq, train=False)
    test = pd.merge(df_seq, df_demo, on='subject', how='left')
    return test
    
   
def gbdt_inferenc(test, features, models):
    data = test[features]

    for c in data.columns:
        if data[c].dtype == "float64":
            data[c] = data[c].astype("float32")
        if data[c].dtype == "int64":
            data[c] = data[c].astype("int32")

    pred = []
    for model in models:
        pred.append(model.predict_proba(data))

        
    pred = np.mean(pred, axis=0)
    return pred
    
    
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Prediction function for Kaggle competition"""

    df_seq = sequence.to_pandas()
    df_demo = demographics.to_pandas()
    is_imu_only = df_seq[tof_cols].isnull().all(axis=1).all()
    
    if is_imu_only:
        exp068_pred = run_inference(
                df_seq,
                fe_func=Imu_FE,
                config=configs['exp068'],
                scaler=exp068_scaler,
                models=exp068_models
        )

        exp070_pred = run_inference(
            df_seq,
            fe_func=Imu_FE,
            config=configs['exp070'],
            scaler=exp070_scaler,
            models=exp070_models
        )

        exp092_pred = run_inference(
            df_seq,
            fe_func=Imu_FE,
            config=configs['exp092'],
            scaler=exp092_scaler,
            models=exp092_models
        )

        exp093_pred = run_inference(
            df_seq,
            fe_func=Imu_FE,
            config=configs['exp093'],
            scaler=exp093_scaler,
            models=exp093_models
        )

        exp104_pred = run_inference(
            df_seq,
            fe_func=Imu_FE,
            config=configs['exp104'],
            scaler=exp104_scaler,
            models=exp104_models
        )

        exp326_pred = run_inference(
            df_seq,
            fe_func=Imu_FE,
            config=configs['exp326'],
            scaler=exp326_scaler,
            models=exp326_models
        )

        exp329_pred = run_inference(
            df_seq,
            fe_func=Imu_FE,
            config=configs['exp329'],
            scaler=exp329_scaler,
            models=exp329_models
        )



        gbdt_test = gbdt_test_data(
            df_seq,
            df_demo,
            fe_fuc0=gbdt_feature_engineering,
            fe_fuc1=GBDT_IMU_FE
        )

        exp117_pred = gbdt_inferenc(
            test=gbdt_test,
            features=exp117_features,
            models=exp117_models
        )    

        pred = (
            exp068_pred
            + exp070_pred
            + exp092_pred
            + exp093_pred
            + exp104_pred
            + exp117_pred
            + exp326_pred
            + exp329_pred
        ) / 8

    else:
        exp068_pred = run_inference(
                df_seq,
                fe_func=Imu_FE,
                config=configs['exp068'],
                scaler=exp068_scaler,
                models=exp068_models
        )
        
        exp086_pred = run_inference(
            df_seq,
            fe_func=ImuTofThm_FE,
            config=configs['exp086'],
            scaler=exp086_scaler,
            models=exp086_models
        )

        exp137_pred = run_inference(
            df_seq,
            fe_func=ImuTofThm_FE,
            config=configs['exp137'],
            scaler=exp137_scaler,
            models=exp137_models
        )

        exp326_pred = run_inference(
            df_seq,
            fe_func=Imu_FE,
            config=configs['exp326'],
            scaler=exp326_scaler,
            models=exp326_models
        )

        gbdt_test = gbdt_test_data(
            df_seq,
            df_demo,
            fe_fuc0=gbdt_feature_engineering,
            fe_fuc1=GBDT_IMU_TOF_THM_FE,
        )


        exp179_pred = gbdt_inferenc(
            test=gbdt_test,
            features=exp179_features,
            models=exp179_models
        )

        exp226_pred = gbdt_inferenc(
            test=gbdt_test,
            features=exp226_features,
            models=exp226_models
        )


        pred = (
            exp086_pred
            + exp137_pred
            + exp068_pred
            + exp179_pred
            + exp226_pred
            + exp326_pred
        ) / 6

    
    
    pred = pred.argmax(axis=1)
    pred = le.inverse_transform(pred)
    
    
    return pred[0]
    
# Kaggle competition interface
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


pd.read_parquet('submission.parquet')


def custom_metrics(y_true, y_pred):
    target_gestures = [
        'Above ear - pull hair',
        'Cheek - pinch skin',
        'Eyebrow - pull hair',
        'Eyelash - pull hair',
        'Forehead - pull hairline',
        'Forehead - scratch',
        'Neck - pinch skin',
        'Neck - scratch',
    ]

    non_target_gestures = [
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

    all_classes = target_gestures + non_target_gestures

    invalid_types = {i for i in y_pred.unique() if i not in all_classes}
    if invalid_types:
        raise ParticipantVisibleError(
                f"Invalid gesture values in submission: {invalid_types}"
        )

    y_true_bin = y_true.isin(target_gestures).values
    y_pred_bin = y_pred.isin(target_gestures).values
    f1_binary = f1_score(
            y_true_bin,
            y_pred_bin,
            pos_label=True,
            zero_division=0,
            average='binary'
    )

    y_true_mc = y_true.apply(lambda x: x if x in target_gestures else 'non_target')
    y_pred_mc = y_pred.apply(lambda x: x if x in target_gestures else 'non_target')

    f1_macro = f1_score(
            y_true_mc,
            y_pred_mc,
            average='macro',
            zero_division=0
    )
    return f1_binary, f1_macro, 0.5 * f1_binary + 0.5 * f1_macro


exp068 = pd.read_csv('/kaggle/input/cmi-detect-behavior-exp068-train/lstm_imu_imu_only_train.csv')
exp070 = pd.read_csv('/kaggle/input/cmi-detect-behavior-exp070-train/lstm_imu_imu_only_train.csv')
exp092 = pd.read_csv('/kaggle/input/cmi-detect-behavior-exp092-train/lstm_imu_imu_only_train.csv')
exp093 = pd.read_csv('/kaggle/input/cmi-detect-behavior-exp093-train/lstm_imu_imu_only_train.csv')
exp104 = pd.read_csv('/kaggle/input/cmi-detect-behavior-exp104-train/gru_imu_imu_only_train.csv')
exp117 = pd.read_csv('/kaggle/input/cmi-detect-behavior-exp117-train/oof_cat_imu.csv')
exp326 = pd.read_csv('/kaggle/input/cmi-detect-behavior-exp326-train/gru_imu_imu_only_train.csv')
exp329 = pd.read_csv('/kaggle/input/cmi-detect-behavior-exp329-train/lstm_imu_imu_only_train.csv')

le = load('/kaggle/input/cmi-detect-behavior-gesture-encoder/Encoder_model.pickle')
pred = (exp068.iloc[:, 3:].values
         + exp070.iloc[:, 3:].values
         + exp104.iloc[:, 3:].values
         + exp092.iloc[:, 3:].values
         + exp093.iloc[:, 3:].values
         + exp117.iloc[:, -18:].values
         + exp326.iloc[:, 3:].values
         + exp329.iloc[:, 3:].values
        ) / 8
pred = np.argmax(pred, axis=1)
pred = le.inverse_transform(pred)
pred = pd.Series(pred)
custom_metrics(exp326['gesture'], pred)


exp086 = pd.read_csv('/kaggle/input/cmi-detect-behavior-exp086-train/gru_imu_tof_thm_imu_only_train.csv')
exp137 = pd.read_csv('/kaggle/input/cmi-detect-behavior-exp137-train/lstm_imu_tof_thm_imu_only_train.csv')
exp179 = pd.read_csv('/kaggle/input/cmi-detect-behavior-exp179-train/oof_cat_imu_tof_thm.csv')
exp226 = pd.read_csv('/kaggle/input/cmi-detect-behavior-exp226-train/oof_xgb_imu_tof_thm.csv')

index = exp326['sequence_id'].isin(exp086['sequence_id'])
pred = (exp086.iloc[:, 3:].values
        +exp137.iloc[:, 3:].values
        + exp326.loc[index].iloc[:, 3:].values
        + exp068.loc[index].iloc[:, 3:].values
        + exp179.iloc[:, -18:].values
        + exp226.iloc[:, -18:].values
        ) / 6
pred = np.argmax(pred, axis=1)
pred = le.inverse_transform(pred)
pred = pd.Series(pred)
custom_metrics(exp086['gesture'], pred)

