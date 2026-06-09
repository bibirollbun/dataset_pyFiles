import os, json, joblib, numpy as np, pandas as pd
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
import polars as pl
from tqdm import tqdm
import torchmetrics
import random
from scipy.spatial.transform import Rotation as R
from scipy.fft import fft, dct



from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import polars as pl
from tqdm import tqdm
import torchmetrics
import random


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"â–¶ imports ready Â· pytorch {torch.__version__} Â· device: {device}")


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(42)


FEATURE_NAMES = [
    'acc_x', 'acc_y', 'acc_z',
    'rot_w', 'rot_x', 'rot_y', 'rot_z',
    'acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel',
    'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag', 'linear_acc_mag_jerk',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
    'angular_distance',
]
CATEGORICAL_FEATURES = []
NUMERICAL_FEATURES = [f for f in FEATURE_NAMES if f not in CATEGORICAL_FEATURES]
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
maps = {}
for k,cl in enumerate(all_classes):
    maps[cl] = k
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from scipy.spatial.transform import Rotation as R
def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    """Normalizes and cleans the time series sequence"""
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

def pad_sequences_torch(sequences, maxlen, padding='post', truncating='post', value=0.0):
    """PyTorch equivalent of Keras pad_sequences"""
    result = []
    for seq in sequences:
        if len(seq) >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        result.append(seq)
    return np.array(result, dtype=np.float32)


from scipy.spatial.transform import Rotation as R
os.environ['NO_ALBUMENTATIONS_UPDATE'] = '1'
def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
seed_everything(42)
def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    """Normalizes and cleans the time series sequence"""
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

def pad_sequences_torch(sequences, maxlen, padding='post', truncating='post', value=0.0):
    """PyTorch equivalent of Keras pad_sequences"""
    result = []
    for seq in sequences:
        if len(seq) >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        result.append(seq)
    return np.array(result, dtype=np.float32)
# ä»�accä¸­ç§»é™¤é‡�åŠ›
def remove_gravity_from_acc(acc_data, rot_data):
    if isinstance(acc_data, pd.DataFrame):
        acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    else:
        acc_values = acc_data
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

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

# è®¡ç®—è§’åº¦
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
    
# è®¡ç®—è§’åº¦è·�ç¦»
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
            angular_dist[i] = 0 # Ğ˜Ğ»Ğ¸ np.nan, Ğ² Ğ·Ğ°Ğ²Ğ¸Ñ�Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸ Ğ¾Ñ‚ Ğ¶ĞµĞ»Ğ°ĞµĞ¼Ğ¾Ğ³Ğ¾ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ�
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
def feature_engineering(train_df):
    # IMU magnitude
    train_df['acc_mag'] = np.sqrt(train_df['acc_x']**2 + train_df['acc_y']**2 + train_df['acc_z']**2)
    # IMU angle
    train_df['rot_angle'] = 2 * np.arccos(train_df['rot_w'].clip(-1, 1))
    # IMU jerk, angular velocity
    train_df['acc_mag_jerk'] = train_df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    train_df['rot_angle_vel'] = train_df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    # Remove gravity
    def get_linear_accel(df):
        res = remove_gravity_from_acc(
            df[['acc_x', 'acc_y', 'acc_z']],
            df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        )
        res = pd.DataFrame(res, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=df.index)
        return res
    linear_accel_df = train_df.groupby('sequence_id').apply(get_linear_accel, include_groups=False)
    linear_accel_df = linear_accel_df.droplevel('sequence_id')
    train_df = train_df.join(linear_accel_df)
    train_df['linear_acc_mag'] = np.sqrt(train_df['linear_acc_x']**2 + train_df['linear_acc_y']**2 + train_df['linear_acc_z']**2)
    train_df['linear_acc_mag_jerk'] = train_df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)

    # Calc angular velocity
    def calc_angular_velocity(df):
        res = calculate_angular_velocity_from_quat( df[['rot_x', 'rot_y', 'rot_z', 'rot_w']] )
        res = pd.DataFrame(res, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=df.index)
        return res
    angular_velocity_df = train_df.groupby('sequence_id').apply(calc_angular_velocity, include_groups=False)
    angular_velocity_df = angular_velocity_df.droplevel('sequence_id')
    train_df = train_df.join(angular_velocity_df)

    # Calculating angular distance
    def calc_angular_distance(df):
        res = calculate_angular_distance(df[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
        res = pd.DataFrame(res, columns=['angular_distance'], index=df.index)
        return res
    angular_distance_df = train_df.groupby('sequence_id').apply(calc_angular_distance, include_groups=False)
    angular_distance_df = angular_distance_df.droplevel('sequence_id')
    train_df = train_df.join(angular_distance_df)
    train_df[FEATURE_NAMES] = train_df[FEATURE_NAMES].ffill().bfill().fillna(0).values.astype('float32')
    return train_df


def get_fix_imu(df):
    acc_body = df[['acc_x', 'acc_y', 'acc_z']].values
    quats = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    num_points = len(acc_body)
    gravity = np.array([0, 0, 9.81])  # é‡�åŠ›å�‘é‡� (Zè½´å�‘ä¸Š)
    positions = np.zeros((num_points, 3))
    for i in range(num_points):
        if quats[i][0]==0:
            continue
        q = quats[i] / np.linalg.norm(quats[i])
        rotation = R.from_quat([ q[0], q[1], q[2], q[3]])  # æ³¨æ„�scipyçš„wxyzé¡ºåº�
        # # æœ¬ä½“åŠ é€Ÿåº¦è½¬ä¸–ç•Œå��æ ‡ç³»å¹¶å�»é™¤é‡�åŠ›
        positions[i] = rotation.apply(acc_body[i]) - gravity
    res = pd.DataFrame(positions, columns=['remove_g_x', 'remove_g_y', 'remove_g_z'], index=df.index)
    return res


# è®¡ç®—è§’åº¦
def calculate_angular_velocity_from_quat_ori(df): # Assuming 200Hz sampling rate
    if isinstance(df, pd.DataFrame):
        quat_values = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))
    q_t = quat_values[0]
    for i in range(1, num_samples):
        q_t_plus_dt = quat_values[i]

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
            angular_vel[i, :] = delta_rot.as_rotvec()
        except ValueError:
            # If quaternion is invalid, angular velocity remains zero
            pass
    res = pd.DataFrame(angular_vel, columns=['ang_ori_x', 'ang_ori_y', 'ang_ori_z'], index=df.index)
    return res


# convert eular
import numpy as np
from scipy.spatial.transform import Rotation as R
def calculate_euler_angles_from_quat(rot_data, euler_order='xyz'):
    # å¤„ç�†è¾“å…¥
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data

    num_samples = quat_values.shape[0]
    euler_angles = np.zeros((num_samples, 3))

    for i in range(num_samples):
        q = quat_values[i]
        # æ£€æŸ¥å››å…ƒæ•°æ˜¯å�¦æœ‰æ•ˆ
        if np.all(np.isnan(q)) or np.all(np.isclose(q, 0)):
            continue
        try:
            # å°†å››å…ƒæ•°è½¬æ�¢ä¸ºæ—‹è½¬å¯¹è±¡
            rot = R.from_quat(q)
            # è½¬æ�¢ä¸ºæ¬§æ‹‰è§’ï¼ˆè§’åº¦åˆ¶ï¼‰
            euler_angles[i, :] = rot.as_euler(euler_order, degrees=True)
        except ValueError:
            # å¦‚æ�œå››å…ƒæ•°æ— æ•ˆï¼Œä¿�æŒ�è§’åº¦ä¸ºé›¶
            pass
    res = pd.DataFrame(euler_angles, columns=['eular_vel_x', 'eular_vel_y', 'eular_vel_z'], index=rot_data.index)
    return res


def get_rot(df):
    acc_body = df[['acc_x', 'acc_y', 'acc_z']].values
    quats = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    num_points = len(acc_body)
    gravity = np.array([0, 0, 9.81])  # é‡�åŠ›å�‘é‡� (Zè½´å�‘ä¸Š)
    rotvecs = np.zeros((num_points, 4))
    for i in range(num_points-1):
        q_t = quats[i]
        q_t_plus_dt = quats[i+1]

        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue
        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt     
            
            # q = quats[i] / np.linalg.norm(quats[i])
            # rotation = R.from_quat([ q[0], q[1], q[2], q[3]])  # æ³¨æ„�scipyçš„wxyzé¡ºåº�
            
            rotvecs[i][:3] = delta_rot.as_rotvec()
            rotvecs[i][3] = np.linalg.norm(delta_rot.as_rotvec())
        except ValueError:
            # å¦‚æ�œå››å…ƒæ•°æ— æ•ˆï¼Œä¿�æŒ�è§’åº¦ä¸ºé›¶
            pass
    res = pd.DataFrame(rotvecs, columns=['rot_new_x', 'rot_new_y', 'rot_new_z','rot_new_norm'], index=df.index)
    return res


def get_imu_dct(df):
    acc_body = df[['acc_x', 'acc_y', 'acc_z']].values
    quats = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    num_points = len(acc_body)
    positions = np.zeros((num_points, 7))
    for k, col in enumerate(['acc_x', 'acc_y', 'acc_z','rot_x', 'rot_y', 'rot_z', 'rot_w']):
        dct_data = dct(df[col].values, type=2, norm='ortho')
        positions[:,k] = dct_data
    res = pd.DataFrame(positions, columns=['acc_x_dct', 'acc_y_dct', 'acc_z_dct','rot_x_dct', 'rot_y_dct', 'rot_z_dct', 'rot_w_dct'], index=df.index)
    return res


import sys
sys.path.append("/kaggle/input/cmimodel")


import torch
import torch.nn as nn
import torch.nn.functional as F
# from transformers import BertConfig, BertModel
# ä¸€æ ·
from torch.autograd import Function

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


train = pd.read_csv(f"/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv",nrows=100)


train = feature_engineering(train)
linear_accel_df = train.groupby('sequence_id').apply(get_rot, include_groups=False)
linear_accel_df = linear_accel_df.droplevel('sequence_id')
train = train.join(linear_accel_df)
linear_accel_df = train.groupby('sequence_id').apply(calculate_angular_velocity_from_quat_ori, include_groups=False)
linear_accel_df = linear_accel_df.droplevel('sequence_id')
train = train.join(linear_accel_df)
train['ang_diff_x'] = train.groupby('sequence_id')['angular_vel_x'].diff().fillna(0)
train['ang_diff_y'] = train.groupby('sequence_id')['angular_vel_y'].diff().fillna(0)
train['ang_diff_z'] = train.groupby('sequence_id')['angular_vel_z'].diff().fillna(0)
train['ang_diff_the'] = train.groupby('sequence_id')['rot_angle_vel'].diff().fillna(0)


def create_gaussian_kernel(size: int, channels: int):
    """Create gaussian kernel for smoothing"""
    kernel = torch.tensor([np.exp(-(i - size//2)**2/2) for i in range(size)], dtype=torch.float32)
    kernel = kernel / kernel.sum()
    kernel = kernel.repeat(channels, 1, 1)  # (out_channels, in_channels/groups, kernel_size)
    return kernel

k = 15
grouped = train.groupby('sequence_id')
for fe in (['rot',"angular_vel"]):
    for dir in ('x', 'y', 'z'):
        col_name = f'{fe}_{dir}'
        weight = create_gaussian_kernel(k, 1)  # 1 channel, cause process 1 column per iteration
        lpf_results = []
        for _, group in grouped:
            # convert to tensor and add dimentions (batch, channel, length)
            data = torch.tensor(group[col_name].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # apply convolution
            lpf = F.conv1d(data, weight, padding=k//2)
            lpf_results.append(lpf.squeeze().numpy())
        
        # concatenate results
        lpf_series = pd.concat([pd.Series(x, index=group.index) for x, (_, group) in zip(lpf_results, grouped)])
        train[f'{fe}_lpf_{dir}'] = lpf_series
        train[f'{fe}_hpf_{dir}'] = train[col_name] - train[f'{fe}_lpf_{dir}']


grouped = train.groupby('sequence_id')

for fe in (['acc','linear_acc']):
    for dir in ('x', 'y', 'z', 'mag'):
        col_name = f'{fe}_{dir}'
        weight = create_gaussian_kernel(k, 1)  # 1 channel, cause process 1 column per iteration
        lpf_results = []
        for _, group in grouped:
            # convert to tensor and add dimentions (batch, channel, length)
            data = torch.tensor(group[col_name].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # apply convolution
            lpf = F.conv1d(data, weight, padding=k//2)
            lpf_results.append(lpf.squeeze().numpy())
        
        # concatenate results
        lpf_series = pd.concat([pd.Series(x, index=group.index) for x, (_, group) in zip(lpf_results, grouped)])
        train[f'{fe}_lpf_{dir}'] = lpf_series
        train[f'{fe}_hpf_{dir}'] = train[col_name] - train[f'{fe}_lpf_{dir}']


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


gesture_classes = {}
for i,k in enumerate(all_classes):
    gesture_classes[i] = k


def pad_sequences_torch(sequences, maxlen, padding='post', truncating='post',value=0.0):
    """PyTorch equivalent of Keras pad_sequences"""
    result = []
    for seq in sequences:
        if len(seq) >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        result.append(seq)
    return np.array(result, dtype=np.float32)


meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation', "orientation_id", "fold", 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
feature_cols = [c for c in train.columns if c not in meta_cols]
imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
tof_cols = [c for c in feature_cols if c.startswith('thm_') or c.startswith('tof_')]
print(f"  IMU {len(imu_cols)} | TOF/THM {len(tof_cols)} | total {len(feature_cols)} features")


imu_cols=['acc_x',
 'acc_y',
 'acc_z',
 'rot_w',
 'rot_x',
 'rot_y',
 'rot_z', # 7
          
'acc_mag',
'acc_mag_jerk',
'rot_angle',
'rot_angle_vel',  # 4

 
 'linear_acc_x',
 'linear_acc_y',
 'linear_acc_z',
 'linear_acc_mag',
 'linear_acc_mag_jerk', # 5
 
 'angular_vel_x',
 'angular_vel_y',
 'angular_vel_z',
 'angular_distance', # 4
 
 'ang_diff_x',
 'ang_diff_y',
 'ang_diff_z',
 'ang_diff_the',
 'ang_ori_x',
 'ang_ori_y',
 'ang_ori_z',
 'rot_new_x',
 'rot_new_y',
 'rot_new_z',
 'rot_new_norm', # 11
          
 'rot_hpf_x',
 'rot_hpf_y',
 'rot_hpf_z',
 'angular_vel_hpf_x',
 'angular_vel_hpf_y',
 'angular_vel_hpf_z',
 'acc_hpf_x',
 'acc_hpf_y',
 'acc_hpf_z',
 'acc_hpf_mag',
 'linear_acc_hpf_x',
 'linear_acc_hpf_y',
 'linear_acc_hpf_z',
 'linear_acc_hpf_mag', # 14 

]


feature_cols = imu_cols


# Make sure gesture_classes exists in both modes
count = 0
pad_len = 256
n_classes = 18
imu_dim = 20
tof_dim = 325


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context
        
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=1):
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
        
# ä¸€æ · åŠ äº†ä¸€ä¸ªmaxpoolå’Œdropout

class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        
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
        # SE block
        out = self.se(out)
        
        # Add shortcut
        out += shortcut
        out = F.relu(out)
        # Pool and dropout
        # out = self.pool(out)
        out = self.dropout(out)
        return out


class IMUModel(nn.Module):
    def __init__(self, imu_ind, n_classes, weight_decay=1e-4):
        super().__init__()
        self.imu_ind = imu_ind
        self.imu_dim = len(imu_ind)
        self.tof_dim = 320
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        indim = 128

        self.pool = nn.AdaptiveAvgPool1d(1)
        # ä¿®æ”¹ conv lstm conv lstm
        self.num_branches = len(imu_ind)
        self.imu_blocks = nn.ModuleList()
        self.grus = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.classf = nn.ModuleList()

        self.imu_blocks2 = nn.ModuleList()
        self.grus2 = nn.ModuleList()
        self.dropouts2 = nn.ModuleList()
        self.attentions2 = nn.ModuleList()
        self.classf2 = nn.ModuleList()
        

        self.imu_blocks3 = nn.ModuleList()
        self.grus3 = nn.ModuleList()
        self.dropouts3 = nn.ModuleList()
        self.attentions3 = nn.ModuleList()
        self.classf3 = nn.ModuleList()
        
        self.allch = np.sum(self.imu_ind)

        ############################ALL#########################
        for in_channels in self.imu_ind:
            self.imu_blocks.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts.append(nn.Dropout(0.5))
            self.attentions.append(AttentionLayer(indim))
            self.classf.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        self.allcls =  nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
        )
        ############################Trans#########################
        for in_channels in self.imu_ind:
            self.imu_blocks2.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus2.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts2.append(nn.Dropout(0.5))
            self.attentions2.append(AttentionLayer(indim))
            # self.exp.append(Expert(indim, indim, n_classes))
            self.classf2.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        ############################Gesture#########################
        for in_channels in self.imu_ind:
            self.imu_blocks3.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus3.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts3.append(nn.Dropout(0.5))
            self.attentions3.append(AttentionLayer(indim))
            self.classf3.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
            
        # all block
        self.fn_block = nn.Sequential(
            ResidualSECNNBlock(self.allch, indim, 3, dropout=0.3),
            ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.fn_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.fn_dropouts = nn.Dropout(0.5)
        self.fn_atten = AttentionLayer(indim)

        ### classification
        self.dense1 = nn.Linear(len(self.imu_ind) * indim * 3  + indim, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

        self.finalallcls = nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.transcls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.gescls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.final = nn.Conv1d(indim, 1, kernel_size=1)

    def forward_mask(self, x):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        all_x = x[:,:self.allch,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        phasemask = self.final(out)  # (B, 1, 256)
        return phasemask
    
    def forward(self, x, phasemask, isfe=False):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []
        

        all_x = x[:,:self.allch,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        # phasemask = self.final(out)  # (B, 1, 256)
        out = out.transpose(1, 2)# (B, T, D)
        out,_= self.fn_gru(out)              # GRU
        out = self.fn_dropouts(out)             # Dropout
        attended = self.fn_atten(out)
        clssf.append(self.allcls(attended))
        attended_list.append(attended)

        # !!!!!!!!!!!phase   000111000
        sigac1 = phasemask.sigmoid() # (B, 1, 256)
        sigac2 = 1 - sigac1
        ges = x * sigac1
        trans =  x * sigac2   
        
        for i in range(self.num_branches):
            imu_i = imu_splits[i]
            out = self.imu_blocks[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            pre.append(out)
            out,_= self.grus[i](out)              # GRU
            out = self.dropouts[i](out)             # Dropout
            attended = self.attentions[i](out)      # Attention
            clssf.append(self.classf[i](attended))
            attended_list.append(attended)

        attended_ges = []
        ii = 0
        for i, k in enumerate(self.imu_ind):
            imu_i = ges[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks2[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus2[i](out)              # GRU
            out = self.dropouts2[i](out)             # Dropout
            attended = self.attentions2[i](out)      # Attention
            clssf.append(self.classf2[i](attended))
            attended_ges.append(attended)

        ii = 0
        attended_trans = []
        for i, k in enumerate(self.imu_ind):
            imu_i = trans[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks3[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus3[i](out)              # GRU
            out = self.dropouts3[i](out)             # Dropout
            attended = self.attentions3[i](out)      # Attention
            clssf.append(self.classf3[i](attended))
            attended_trans.append(attended)

        attended_one = torch.cat(attended_list, dim=-1)
        attended_trans = torch.cat(attended_trans, dim=-1)
        attended_ges = torch.cat(attended_ges, dim=-1)


        finalatten = torch.cat([attended_ges, attended_trans, attended_one], dim=-1)

        clssf.append(self.finalallcls(attended_trans))
        clssf.append(self.transcls(attended_trans))
        clssf.append(self.gescls(attended_ges))
        
        x = F.relu(self.bn_dense1(self.dense1(finalatten)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        if isfe:
            return attended_one, logits, clssf, pre
        return logits, clssf, phasemask


modeldet = IMUModel([7,4,5,4,11],18).to(device)
modeldet.load_state_dict(torch.load("/kaggle/input/cmimodel/bets_phasedet.pt"))
modeldet.eval();


class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context
        
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=1):
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
        
# ä¸€æ · åŠ äº†ä¸€ä¸ªmaxpoolå’Œdropout

class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        
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
        # SE block
        out = self.se(out)
        
        # Add shortcut
        out += shortcut
        out = F.relu(out)
        # Pool and dropout
        # out = self.pool(out)
        out = self.dropout(out)
        return out
class IMUModel(nn.Module):
    def __init__(self, imu_ind, n_classes, weight_decay=1e-4):
        super().__init__()
        self.imu_ind = imu_ind
        self.imu_dim = len(imu_ind)
        self.tof_dim = 320
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        indim = 128

        self.pool = nn.AdaptiveAvgPool1d(1)
        # ä¿®æ”¹ conv lstm conv lstm
        self.num_branches = len(imu_ind)
        self.imu_blocks = nn.ModuleList()
        self.grus = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.classf = nn.ModuleList()

        self.imu_blocks2 = nn.ModuleList()
        self.grus2 = nn.ModuleList()
        self.dropouts2 = nn.ModuleList()
        self.attentions2 = nn.ModuleList()
        self.classf2 = nn.ModuleList()
        

        self.imu_blocks3 = nn.ModuleList()
        self.grus3 = nn.ModuleList()
        self.dropouts3 = nn.ModuleList()
        self.attentions3 = nn.ModuleList()
        self.classf3 = nn.ModuleList()


        self.imu_blocks4 = nn.ModuleList()
        self.grus4 = nn.ModuleList()
        self.dropouts4 = nn.ModuleList()
        self.attentions4 = nn.ModuleList()
        self.classf4 = nn.ModuleList()
        
        self.allch = np.sum(self.imu_ind)

        #############one channel loss############################
        for i in range(self.allch):
            self.imu_blocks4.append(nn.Sequential(
                ResidualSECNNBlock(1, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus4.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts4.append(nn.Dropout(0.5))
            self.attentions4.append(AttentionLayer(indim))
            self.classf4.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )        
        ############################ALL#########################
        for in_channels in self.imu_ind:
            self.imu_blocks.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts.append(nn.Dropout(0.5))
            self.attentions.append(AttentionLayer(indim))
            self.classf.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        self.allcls =  nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
        )
        ############################Trans#########################
        for in_channels in self.imu_ind:
            self.imu_blocks2.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus2.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts2.append(nn.Dropout(0.5))
            self.attentions2.append(AttentionLayer(indim))
            # self.exp.append(Expert(indim, indim, n_classes))
            self.classf2.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        ############################Gesture#########################
        for in_channels in self.imu_ind:
            self.imu_blocks3.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus3.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts3.append(nn.Dropout(0.5))
            self.attentions3.append(AttentionLayer(indim))
            self.classf3.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
            
        # all block
        self.fn_block = nn.Sequential(
            ResidualSECNNBlock(20, indim, 3, dropout=0.3),
            ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.fn_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.fn_dropouts = nn.Dropout(0.5)
        self.fn_atten = AttentionLayer(indim)

        ### classification
        self.dense1 = nn.Linear(len(self.imu_ind) * indim * 3  + indim + indim*4 , 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

        self.finalallcls = nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim + indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.transcls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.gescls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.onecls =  nn.Sequential(
                nn.Linear(4 * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.precls = nn.Sequential(
                nn.Linear(self.allch * indim, 4 * indim, bias=False),
                nn.BatchNorm1d(4 * indim),
                nn.Dropout(0.5),
                )
        
        
        self.final = nn.Conv1d(indim, 1, kernel_size=1)

    def forward_mask(self, x):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        all_x = x[:,:self.allch,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        phasemask = self.final(out)  # (B, 1, 256)
        return phasemask
    
    def forward(self, x, phasemask, isfe=False):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        oneattended_list = []
        
        ###########One Channel##################
        for i in range(self.allch):
            imu_i = x[:, i:i+1, :]
            out = self.imu_blocks4[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus4[i](out)              # GRU
            out = self.dropouts4[i](out)             # Dropout
            attended = self.attentions4[i](out)      # Attention
            clssf.append(self.classf4[i](attended))
            oneattended_list.append(attended)

        oneattended_list = torch.cat(oneattended_list, dim=-1)
        oneattended_list = self.precls(oneattended_list)
        
        all_x = x[:,:20,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        # phasemask = self.final(out)  # (B, 1, 256)
        out = out.transpose(1, 2)# (B, T, D)
        out,_= self.fn_gru(out)              # GRU
        out = self.fn_dropouts(out)             # Dropout
        attended = self.fn_atten(out)
        clssf.append(self.allcls(attended))
        attended_list.append(attended)

        # !!!!!!!!!!!phase   000111000
        sigac1 = phasemask.sigmoid() # (B, 1, 256)
        sigac2 = 1 - sigac1
        ges = x * sigac1
        trans =  x * sigac2   
        
        for i in range(self.num_branches):
            imu_i = imu_splits[i]
            out = self.imu_blocks[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            pre.append(out)
            out,_= self.grus[i](out)              # GRU
            out = self.dropouts[i](out)             # Dropout
            attended = self.attentions[i](out)      # Attention
            clssf.append(self.classf[i](attended))
            attended_list.append(attended)

        attended_ges = []
        ii = 0
        for i, k in enumerate(self.imu_ind):
            imu_i = ges[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks2[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus2[i](out)              # GRU
            out = self.dropouts2[i](out)             # Dropout
            attended = self.attentions2[i](out)      # Attention
            clssf.append(self.classf2[i](attended))
            attended_ges.append(attended)

        ii = 0
        attended_trans = []
        for i, k in enumerate(self.imu_ind):
            imu_i = trans[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks3[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus3[i](out)              # GRU
            out = self.dropouts3[i](out)             # Dropout
            attended = self.attentions3[i](out)      # Attention
            clssf.append(self.classf3[i](attended))
            attended_trans.append(attended)

        attended_one = torch.cat(attended_list, dim=-1)
        attended_trans = torch.cat(attended_trans, dim=-1)
        attended_ges = torch.cat(attended_ges, dim=-1)
        # oneattended_list = torch.cat(oneattended_list, dim=-1)

        finalatten = torch.cat([attended_ges, attended_trans, attended_one, oneattended_list], dim=-1)

        clssf.append(self.finalallcls(attended_one))
        clssf.append(self.transcls(attended_trans))
        clssf.append(self.gescls(attended_ges))
        clssf.append(self.onecls(oneattended_list))
        
        x = F.relu(self.bn_dense1(self.dense1(finalatten)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        if isfe:
            return attended_one, logits, clssf, pre
        return logits, clssf, phasemask


model_imus = []
device = "cuda:0"
md = [
    "/kaggle/input/cmimodel/bets_0_837.pt",
    "/kaggle/input/cmimodel/bets_1_837.pt",
    "/kaggle/input/cmimodel/bets_2_837.pt",
    "/kaggle/input/cmimodel/bets_3_837.pt",
    "/kaggle/input/cmimodel/bets_4_837.pt"
]
for fold in range(5):
    checkpoint = torch.load(md[fold], map_location=device)
    model_imu = IMUModel([7,4,5,4,11,14],18).to(device)
    model_imu.load_state_dict(checkpoint)
    model_imu.eval();
    model_imus.append(model_imu)


class IMUModel(nn.Module):
    def __init__(self, imu_ind, n_classes, weight_decay=1e-4):
        super().__init__()
        self.imu_ind = imu_ind
        self.imu_dim = len(imu_ind)
        self.tof_dim = 320
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        indim = 128

        self.pool = nn.AdaptiveAvgPool1d(1)
        # ä¿®æ”¹ conv lstm conv lstm
        self.num_branches = len(imu_ind)
        self.imu_blocks = nn.ModuleList()
        self.grus = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.classf = nn.ModuleList()

        self.imu_blocks2 = nn.ModuleList()
        self.grus2 = nn.ModuleList()
        self.dropouts2 = nn.ModuleList()
        self.attentions2 = nn.ModuleList()
        self.classf2 = nn.ModuleList()
        

        self.imu_blocks3 = nn.ModuleList()
        self.grus3 = nn.ModuleList()
        self.dropouts3 = nn.ModuleList()
        self.attentions3 = nn.ModuleList()
        self.classf3 = nn.ModuleList()


        self.imu_blocks4 = nn.ModuleList()
        self.grus4 = nn.ModuleList()
        self.dropouts4 = nn.ModuleList()
        self.attentions4 = nn.ModuleList()
        self.classf4 = nn.ModuleList()
        
        self.allch = np.sum(self.imu_ind)

        #############one channel loss############################
        for i in range(self.allch):
            self.imu_blocks4.append(nn.Sequential(
                ResidualSECNNBlock(1, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus4.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts4.append(nn.Dropout(0.5))
            self.attentions4.append(AttentionLayer(indim))
            self.classf4.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )        
        ############################ALL#########################
        for in_channels in self.imu_ind:
            self.imu_blocks.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts.append(nn.Dropout(0.5))
            self.attentions.append(AttentionLayer(indim))
            self.classf.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        self.allcls =  nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
        )
        ############################Trans#########################
        for in_channels in self.imu_ind:
            self.imu_blocks2.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus2.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts2.append(nn.Dropout(0.5))
            self.attentions2.append(AttentionLayer(indim))
            # self.exp.append(Expert(indim, indim, n_classes))
            self.classf2.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        ############################Gesture#########################
        for in_channels in self.imu_ind:
            self.imu_blocks3.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus3.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts3.append(nn.Dropout(0.5))
            self.attentions3.append(AttentionLayer(indim))
            self.classf3.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
            
        # all block
        self.fn_block = nn.Sequential(
            ResidualSECNNBlock(20, indim, 3, dropout=0.3),
            ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.fn_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.fn_dropouts = nn.Dropout(0.5)
        self.fn_atten = AttentionLayer(indim)

        ### classification
        self.dense1 = nn.Linear(len(self.imu_ind) * indim * 3  + indim + indim*4 , 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

        self.finalallcls = nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim + indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.transcls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.gescls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.onecls =  nn.Sequential(
                nn.Linear(4 * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.precls = nn.Sequential(
                nn.Linear(self.allch * indim, 4 * indim, bias=False),
                nn.BatchNorm1d(4 * indim),
                nn.Dropout(0.5),
                )
        
        
        self.final = nn.Conv1d(indim, 1, kernel_size=1)

    def forward_mask(self, x):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        all_x = x[:,:self.allch,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        phasemask = self.final(out)  # (B, 1, 256)
        return phasemask
    
    def forward(self, x, phasemask, isfe=False):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        oneattended_list = []
        
        ###########One Channel##################
        for i in range(self.allch):
            imu_i = x[:, i:i+1, :]
            out = self.imu_blocks4[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus4[i](out)              # GRU
            out = self.dropouts4[i](out)             # Dropout
            attended = self.attentions4[i](out)      # Attention
            clssf.append(self.classf4[i](attended))
            oneattended_list.append(attended)

        oneattended_list = torch.cat(oneattended_list, dim=-1)
        oneattended_list = self.precls(oneattended_list)
        
        all_x = x[:,:20,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        # phasemask = self.final(out)  # (B, 1, 256)
        out = out.transpose(1, 2)# (B, T, D)
        out,_= self.fn_gru(out)              # GRU
        out = self.fn_dropouts(out)             # Dropout
        attended = self.fn_atten(out)
        clssf.append(self.allcls(attended))
        attended_list.append(attended)

        # !!!!!!!!!!!phase   000111000
        sigac1 = phasemask.sigmoid() # (B, 1, 256)
        sigac2 = 1 - sigac1
        ges = x * sigac1
        trans =  x * sigac2   
        
        for i in range(self.num_branches):
            imu_i = imu_splits[i]
            out = self.imu_blocks[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            pre.append(out)
            out,_= self.grus[i](out)              # GRU
            out = self.dropouts[i](out)             # Dropout
            attended = self.attentions[i](out)      # Attention
            clssf.append(self.classf[i](attended))
            attended_list.append(attended)

        attended_ges = []
        ii = 0
        for i, k in enumerate(self.imu_ind):
            imu_i = ges[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks2[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus2[i](out)              # GRU
            out = self.dropouts2[i](out)             # Dropout
            attended = self.attentions2[i](out)      # Attention
            clssf.append(self.classf2[i](attended))
            attended_ges.append(attended)

        ii = 0
        attended_trans = []
        for i, k in enumerate(self.imu_ind):
            imu_i = trans[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks3[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus3[i](out)              # GRU
            out = self.dropouts3[i](out)             # Dropout
            attended = self.attentions3[i](out)      # Attention
            clssf.append(self.classf3[i](attended))
            attended_trans.append(attended)

        attended_one = torch.cat(attended_list, dim=-1)
        attended_trans = torch.cat(attended_trans, dim=-1)
        attended_ges = torch.cat(attended_ges, dim=-1)
        # oneattended_list = torch.cat(oneattended_list, dim=-1)

        finalatten = torch.cat([attended_ges, attended_trans, attended_one, oneattended_list], dim=-1)

        clssf.append(self.finalallcls(attended_one))
        clssf.append(self.transcls(attended_trans))
        clssf.append(self.gescls(attended_ges))
        clssf.append(self.onecls(oneattended_list))
        
        x = F.relu(self.bn_dense1(self.dense1(finalatten)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        if isfe:
            return finalatten, logits, clssf, phasemask
        return logits, clssf, phasemask
        
class TwoBranchModel_IMU_THM_TOF(nn.Module):
    def __init__(self, imu_ind, n_classes, fold=0):
        super().__init__()
        self.imu_dim = 31
        self.tof_dim = 320
        self.imu_ind = imu_ind
        the_dim = 5
        self.the_dim = the_dim
        self.n_classes = n_classes
        
        indim = 128
        self.num_branches = len(imu_ind)
        
        self.allch = np.sum(self.imu_ind)

        
        self.model_imu = IMUModel(self.imu_ind,18)

        ################################TOF BLOCK#########################
        # self.tof_block = nn.Sequential(
        #         ResidualSECNNBlock(320 + 31 + 5, indim, 3, dropout=0.3),
        #         ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        # )
        # self.tof_block2 = nn.Sequential(
        #         ResidualSECNNBlock(64 + self.allch, indim, 3, dropout=0.3),
        #         ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        # )        
        # self.tof_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        # self.tof_gru_dropout = nn.Dropout(0.4)
        # self.tof_attention = AttentionLayer(indim)
        # self.tof_cls_conv1d = nn.Sequential(
        #         nn.Linear(indim, 256, bias=False),
        #         nn.BatchNorm1d(256),
        #         nn.Dropout(0.5),
        #         nn.Linear(256, n_classes)
        # )
        
        self.tof_encoder = nn.Sequential(
            nn.Conv3d(5, 32, kernel_size=(3, 3, 3), padding=1),  # [B, 32, 256, 8, 8]
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # [B, 32, 128, 4, 4]

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),  # [B, 64, 128, 4, 4]
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # [B, 64, 64, 2, 2]

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),  # [B, 128, 64, 2, 2]
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # [B, 128, 1, 1, 1]
            nn.Flatten()
        )
        
        # Thermal
        self.the_block = nn.Sequential(
                ResidualSECNNBlock(the_dim, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.the_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.the_gru_dropout = nn.Dropout(0.4)
        self.the_attention = AttentionLayer(indim)

        
        self.tof_cls = nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.the_cls = nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        self.imuprecls = nn.Sequential(
            nn.Linear(len(self.imu_ind) * indim * 3  + indim + indim*4 , 4*indim, bias=False),
            nn.BatchNorm1d(4*indim),
            nn.Dropout(0.5),
        )
        
        self.imucls = nn.Sequential(
            nn.Linear(4*indim , 2*indim, bias=False),
            nn.BatchNorm1d(2*indim),
            nn.Dropout(0.5),
            nn.Linear(2*indim, n_classes)
        )
        
        # Dense layers
        self.dense1 = nn.Linear(4*indim + indim + indim , 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, allx, phasemask):
        
        B, T, D = allx.shape
        imux = allx[:,:,:self.allch]
        tofx = allx[:,:,self.allch:]
        
        # IMU=======================================================
        finalatten, logits, clssf, phasemask = self.model_imu(imux[:,:,:self.allch], phasemask, True)
        finalatten = self.imuprecls(finalatten)
        
        # THE======================================================
        the = tofx[:,:,:5]
        the = the.transpose(1, 2)  # (batch, imu_dim, seq_len)
        x3 = self.the_block(the)
        x3 = x3.transpose(1, 2)
        x3, _ = self.the_gru(x3)
        x3 = self.the_gru_dropout(x3)
        attendedthe = self.the_attention(x3)
        clssf.append(self.the_cls(attendedthe))

        # TOF=====================================================
        tof = tofx[:,:,5:]
        # tof1d = tofx[:,:,5:].transpose(1,2)
        tof = tof.view(B,T, 5, 8, 8)
        tof = tof.permute(0,2,1,3,4)
        attendedtof = self.tof_encoder(tof)
        clssf.append(self.tof_cls(attendedtof))
        clssf.append(self.imucls(finalatten))
        
        attendedfinal = torch.cat([finalatten, attendedtof, attendedthe],dim=1)
        # åˆ†ç±»
        x = F.relu(self.bn_dense1(self.dense1(attendedfinal)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        return logits, clssf
        # clssf.append(logits)
        # weightclsemb = torch.cat(clssf, dim=-1)
        # return logits, clssf, attendedfinal


model_alls = []
device = "cuda:0"
md = [
    "/kaggle/input/cmimodel/bets_0_710.pt",
    "/kaggle/input/cmimodel/bets_1_710.pt",
    "/kaggle/input/cmimodel/bets_2_710.pt",
    "/kaggle/input/cmimodel/bets_3_710.pt",
    "/kaggle/input/cmimodel/bets_4_710.pt"
]
for fold in range(5):
    checkpoint = torch.load(md[fold], map_location=device)
    model_all = TwoBranchModel_IMU_THM_TOF([7,4,5,4,11,14],18).to(device)
    model_all.load_state_dict(checkpoint)
    model_all.eval();
    model_alls.append(model_all)


model_alls3 = []
device = "cuda:0"
md = [
    "/kaggle/input/cmimodel/bets_0_775.pt",
    "/kaggle/input/cmimodel/bets_1_775.pt",
    "/kaggle/input/cmimodel/bets_2_775.pt",
    "/kaggle/input/cmimodel/bets_3_775.pt",
    "/kaggle/input/cmimodel/bets_4_775.pt"
]
for fold in range(5):
    checkpoint = torch.load(md[fold], map_location=device)
    model_all = TwoBranchModel_IMU_THM_TOF([7,4,5,4,11,14],18).to(device)
    model_all.load_state_dict(checkpoint)
    model_all.eval();
    model_alls3.append(model_all)


class IMUModel(nn.Module):
    def __init__(self, imu_ind, n_classes, weight_decay=1e-4):
        super().__init__()
        self.imu_ind = imu_ind
        self.imu_dim = len(imu_ind)
        self.tof_dim = 320
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        indim = 128

        self.pool = nn.AdaptiveAvgPool1d(1)
        # ä¿®æ”¹ conv lstm conv lstm
        self.num_branches = len(imu_ind)
        self.imu_blocks = nn.ModuleList()
        self.grus = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.classf = nn.ModuleList()

        self.imu_blocks2 = nn.ModuleList()
        self.grus2 = nn.ModuleList()
        self.dropouts2 = nn.ModuleList()
        self.attentions2 = nn.ModuleList()
        self.classf2 = nn.ModuleList()
        

        self.imu_blocks3 = nn.ModuleList()
        self.grus3 = nn.ModuleList()
        self.dropouts3 = nn.ModuleList()
        self.attentions3 = nn.ModuleList()
        self.classf3 = nn.ModuleList()


        self.imu_blocks4 = nn.ModuleList()
        self.grus4 = nn.ModuleList()
        self.dropouts4 = nn.ModuleList()
        self.attentions4 = nn.ModuleList()
        self.classf4 = nn.ModuleList()
        
        self.allch = np.sum(self.imu_ind)

        #############one channel loss############################
        for i in range(self.allch):
            self.imu_blocks4.append(nn.Sequential(
                ResidualSECNNBlock(1, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus4.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts4.append(nn.Dropout(0.5))
            self.attentions4.append(AttentionLayer(indim))
            self.classf4.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )        
        ############################ALL#########################
        for in_channels in self.imu_ind:
            self.imu_blocks.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts.append(nn.Dropout(0.5))
            self.attentions.append(AttentionLayer(indim))
            self.classf.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        self.allcls =  nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
        )
        ############################Trans#########################
        for in_channels in self.imu_ind:
            self.imu_blocks2.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus2.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts2.append(nn.Dropout(0.5))
            self.attentions2.append(AttentionLayer(indim))
            # self.exp.append(Expert(indim, indim, n_classes))
            self.classf2.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        ############################Gesture#########################
        for in_channels in self.imu_ind:
            self.imu_blocks3.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus3.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts3.append(nn.Dropout(0.5))
            self.attentions3.append(AttentionLayer(indim))
            self.classf3.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
            
        # all block
        self.fn_block = nn.Sequential(
            ResidualSECNNBlock(20, indim, 3, dropout=0.3),
            ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.fn_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.fn_dropouts = nn.Dropout(0.5)
        self.fn_atten = AttentionLayer(indim)

        ### classification
        self.dense1 = nn.Linear(len(self.imu_ind) * indim * 3  + indim + indim*4 , 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

        self.finalallcls = nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim + indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.transcls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.gescls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.onecls =  nn.Sequential(
                nn.Linear(4 * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.precls = nn.Sequential(
                nn.Linear(self.allch * indim, 4 * indim, bias=False),
                nn.BatchNorm1d(4 * indim),
                nn.Dropout(0.5),
                )
        
        
        self.final = nn.Conv1d(indim, 1, kernel_size=1)

        self.embeddingcls = nn.Sequential(
                nn.Linear(self.allch * indim + indim + self.num_branches * indim, 2*indim, bias=False),
                nn.BatchNorm1d(2*indim),
                nn.Dropout(0.5),
                )
    def forward_mask(self, x):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        all_x = x[:,:self.allch,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        phasemask = self.final(out)  # (B, 1, 256)
        return phasemask
    
    def forward(self, x, phasemask, isfe=False):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        attended_list = []

        oneattended_list = []

        pre = []
        ###########One Channel##################
        for i in range(self.allch):
            imu_i = x[:, i:i+1, :]
            out = self.imu_blocks4[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            pre.append(out)
            out,_= self.grus4[i](out)              # GRU
            out = self.dropouts4[i](out)             # Dropout
            attended = self.attentions4[i](out)      # Attention
            clssf.append(self.classf4[i](attended))
            oneattended_list.append(attended)

        oneattended_list = torch.cat(oneattended_list, dim=-1)
        oneattended_list = self.precls(oneattended_list)
        
        all_x = x[:,:20,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        # phasemask = self.final(out)  # (B, 1, 256)
        out = out.transpose(1, 2)# (B, T, D)

        
        # all_embedding = out
        pre.append(out)
        out,_= self.fn_gru(out)              # GRU
        out = self.fn_dropouts(out)             # Dropout
        attended = self.fn_atten(out)
        clssf.append(self.allcls(attended))
        attended_list.append(attended)

        # !!!!!!!!!!!phase   000111000
        sigac1 = phasemask.sigmoid() # (B, 1, 256)
        sigac2 = 1 - sigac1
        ges = x * sigac1
        trans =  x * sigac2   
        
        for i in range(self.num_branches):
            imu_i = imu_splits[i]
            out = self.imu_blocks[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            pre.append(out)
            out,_= self.grus[i](out)              # GRU
            out = self.dropouts[i](out)             # Dropout
            attended = self.attentions[i](out)      # Attention
            clssf.append(self.classf[i](attended))
            attended_list.append(attended)

        attended_ges = []
        ii = 0
        for i, k in enumerate(self.imu_ind):
            imu_i = ges[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks2[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus2[i](out)              # GRU
            out = self.dropouts2[i](out)             # Dropout
            attended = self.attentions2[i](out)      # Attention
            clssf.append(self.classf2[i](attended))
            attended_ges.append(attended)

        ii = 0
        attended_trans = []
        for i, k in enumerate(self.imu_ind):
            imu_i = trans[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks3[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus3[i](out)              # GRU
            out = self.dropouts3[i](out)             # Dropout
            attended = self.attentions3[i](out)      # Attention
            clssf.append(self.classf3[i](attended))
            attended_trans.append(attended)


        
        attended_one = torch.cat(attended_list, dim=-1)
        attended_trans = torch.cat(attended_trans, dim=-1)
        attended_ges = torch.cat(attended_ges, dim=-1)
        # oneattended_list = torch.cat(oneattended_list, dim=-1)

        finalatten = torch.cat([attended_ges, attended_trans, attended_one, oneattended_list], dim=-1)

        clssf.append(self.finalallcls(attended_one))
        clssf.append(self.transcls(attended_trans))
        clssf.append(self.gescls(attended_ges))
        clssf.append(self.onecls(oneattended_list))

        all_embedding = torch.cat(pre, dim=-1) # B * T * D
        all_embedding = self.embeddingcls(all_embedding)
        
        x = F.relu(self.bn_dense1(self.dense1(finalatten)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        if isfe:
            return finalatten, logits, clssf, phasemask,all_embedding
        return logits, clssf, phasemask


class TwoBranchModel_IMU_THM_TOF(nn.Module):
    def __init__(self, imu_ind, n_classes, fold=0):
        super().__init__()
        self.imu_dim = 31
        self.tof_dim = 320
        self.imu_ind = imu_ind
        the_dim = 5
        self.the_dim = the_dim
        self.n_classes = n_classes
        
        indim = 128
        self.num_branches = len(imu_ind)
        
        self.allch = np.sum(self.imu_ind)

        
        self.model_imu = IMUModel(self.imu_ind,18)

        ################################TOF BLOCK#########################
        # self.tof_block = nn.Sequential(
        #         ResidualSECNNBlock(320 + 31 + 5, indim, 3, dropout=0.3),
        #         ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        # )
        # self.tof_block2 = nn.Sequential(
        #         ResidualSECNNBlock(64 + self.allch, indim, 3, dropout=0.3),
        #         ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        # )        
        # self.tof_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        # self.tof_gru_dropout = nn.Dropout(0.4)
        # self.tof_attention = AttentionLayer(indim)
        # self.tof_cls_conv1d = nn.Sequential(
        #         nn.Linear(indim, 256, bias=False),
        #         nn.BatchNorm1d(256),
        #         nn.Dropout(0.5),
        #         nn.Linear(256, n_classes)
        # )
        
        self.tof_encoder = nn.Sequential(
            nn.Conv3d(5, 32, kernel_size=(3, 3, 3), padding=1),  # [B, 32, 256, 8, 8]
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # [B, 32, 128, 4, 4]

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),  # [B, 64, 128, 4, 4]
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # [B, 64, 64, 2, 2]

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),  # [B, 128, 64, 2, 2]
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # [B, 128, 1, 1, 1]
            nn.Flatten()
        )
        
        # Thermal
        self.the_block = nn.Sequential(
                ResidualSECNNBlock(the_dim, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.the_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.the_gru_dropout = nn.Dropout(0.4)
        self.the_attention = AttentionLayer(indim)

        
        self.tof_cls = nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.the_cls = nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        self.imuprecls = nn.Sequential(
            nn.Linear(len(self.imu_ind) * indim * 3  + indim + indim*4 , 4*indim, bias=False),
            nn.BatchNorm1d(4*indim),
            nn.Dropout(0.5),
        )
        
        self.imucls = nn.Sequential(
            nn.Linear(4*indim , 2*indim, bias=False),
            nn.BatchNorm1d(2*indim),
            nn.Dropout(0.5),
            nn.Linear(2*indim, n_classes)
        )
        
        # Dense layers
        self.dense1 = nn.Linear(4*indim + indim + indim +indim, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

        ######
        ## tof encoder 2d
        self.tof_encoder_2d =  nn.Sequential(
            nn.Conv2d(5, 32, kernel_size=(3, 3), padding=1),  # [B, 256, 32, 8, 8]
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),  # [B, 256, 32, 4, 4]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),  # [B, 256, 32, 2, 2]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),  # [B, 256, 128, 1, 1]
        )

        
        self.imu_tof_gru = nn.GRU(indim*(1+2), indim, bidirectional=False, batch_first=True)
        self.imu_tof_gru_drop = nn.Dropout(0.5)
        self.imu_tof_gru_drop_attention = AttentionLayer(indim)
        self.imu_tof_cls = nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
        )
        
    def forward(self, allx, phasemask):
        
        B, T, D = allx.shape
        imux = allx[:,:,:self.allch]
        tofx = allx[:,:,self.allch:]
        
        # IMU=======================================================
        finalatten, logits, clssf, phasemask,all_embedding = self.model_imu(imux[:,:,:self.allch], phasemask, True)
        finalatten = self.imuprecls(finalatten)
        
        # THE======================================================
        the = tofx[:,:,:5]
        the = the.transpose(1, 2)  # (batch, imu_dim, seq_len)
        x3 = self.the_block(the)
        x3 = x3.transpose(1, 2)
        x3, _ = self.the_gru(x3)
        x3 = self.the_gru_dropout(x3)
        attendedthe = self.the_attention(x3)
        clssf.append(self.the_cls(attendedthe))

        # TOF=====================================================
        tof = tofx[:,:,5:]
        tof_2d = tof.view(B,T, 5, 8, 8)
        tof = tof_2d.permute(0,2,1,3,4)
        attendedtof = self.tof_encoder(tof)
        clssf.append(self.tof_cls(attendedtof))
        clssf.append(self.imucls(finalatten))

        #==================================
        tof_2d = tof_2d.view(B*T, 5,8,8)
        tof_2d_embedding = self.tof_encoder_2d(tof_2d) # B*256  * 128 * 1 * 1
        tof_2d_embedding = tof_2d_embedding.view(B,T,128)
        tof_2d_embedding = torch.cat([all_embedding, tof_2d_embedding], dim=-1) # B* T * 256
        tof_2d_embedding = self.imu_tof_gru(tof_2d_embedding)[0]
        tof_2d_embedding = self.imu_tof_gru_drop(tof_2d_embedding)
        tof_2d_embedding = self.imu_tof_gru_drop_attention(tof_2d_embedding)
        clssf.append(self.imu_tof_cls(tof_2d_embedding))
        
        attendedfinal = torch.cat([tof_2d_embedding, finalatten, attendedtof, attendedthe],dim=1)
        # åˆ†ç±»
        x = F.relu(self.bn_dense1(self.dense1(attendedfinal)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        return logits, clssf
        # clssf.append(logits)
        # weightclsemb = torch.cat(clssf, dim=-1)
        # return logits, clssf, attendedfinal


model_alls2 = []
device = "cuda:0"
md = [
    "/kaggle/input/cmimodel/bets_0_876.pt",
    "/kaggle/input/cmimodel/bets_1_876.pt",
    "/kaggle/input/cmimodel/bets_2_876.pt",
    "/kaggle/input/cmimodel/bets_3_876.pt",
    "/kaggle/input/cmimodel/bets_4_876.pt"
]
for fold in range(5):
    checkpoint = torch.load(md[fold], map_location=device)
    model_all = TwoBranchModel_IMU_THM_TOF([7,4,5,4,11,14],18).to(device)
    model_all.load_state_dict(checkpoint)
    model_all.eval();
    model_alls2.append(model_all)


class IMUModel(nn.Module):
    def __init__(self, imu_ind, n_classes, weight_decay=1e-4):
        super().__init__()
        self.imu_ind = imu_ind
        self.imu_dim = len(imu_ind)
        self.tof_dim = 320
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        indim = 128

        self.pool = nn.AdaptiveAvgPool1d(1)
        # ä¿®æ”¹ conv lstm conv lstm
        self.num_branches = len(imu_ind)
        self.imu_blocks = nn.ModuleList()
        self.grus = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.classf = nn.ModuleList()

        self.imu_blocks2 = nn.ModuleList()
        self.grus2 = nn.ModuleList()
        self.dropouts2 = nn.ModuleList()
        self.attentions2 = nn.ModuleList()
        self.classf2 = nn.ModuleList()
        

        self.imu_blocks3 = nn.ModuleList()
        self.grus3 = nn.ModuleList()
        self.dropouts3 = nn.ModuleList()
        self.attentions3 = nn.ModuleList()
        self.classf3 = nn.ModuleList()


        self.imu_blocks4 = nn.ModuleList()
        self.grus4 = nn.ModuleList()
        self.dropouts4 = nn.ModuleList()
        self.attentions4 = nn.ModuleList()
        self.classf4 = nn.ModuleList()
        
        self.allch = np.sum(self.imu_ind)

        #############one channel loss############################
        for i in range(self.allch):
            self.imu_blocks4.append(nn.Sequential(
                ResidualSECNNBlock(1, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus4.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts4.append(nn.Dropout(0.5))
            self.attentions4.append(AttentionLayer(indim))
            self.classf4.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )        
        ############################ALL#########################
        for in_channels in self.imu_ind:
            self.imu_blocks.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts.append(nn.Dropout(0.5))
            self.attentions.append(AttentionLayer(indim))
            self.classf.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        self.allcls =  nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
        )
        ############################Trans#########################
        for in_channels in self.imu_ind:
            self.imu_blocks2.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus2.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts2.append(nn.Dropout(0.5))
            self.attentions2.append(AttentionLayer(indim))
            # self.exp.append(Expert(indim, indim, n_classes))
            self.classf2.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        ############################Gesture#########################
        for in_channels in self.imu_ind:
            self.imu_blocks3.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus3.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts3.append(nn.Dropout(0.5))
            self.attentions3.append(AttentionLayer(indim))
            self.classf3.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
            
        # all block
        self.fn_block = nn.Sequential(
            ResidualSECNNBlock(20, indim, 3, dropout=0.3),
            ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.fn_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.fn_dropouts = nn.Dropout(0.5)
        self.fn_atten = AttentionLayer(indim)

        ### classification
        self.dense1 = nn.Linear(len(self.imu_ind) * indim * 3  + indim + indim*4 , 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

        self.finalallcls = nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim + indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.transcls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.gescls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.onecls =  nn.Sequential(
                nn.Linear(4 * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.precls = nn.Sequential(
                nn.Linear(self.allch * indim, 4 * indim, bias=False),
                nn.BatchNorm1d(4 * indim),
                nn.Dropout(0.5),
                )
        
        
        self.final = nn.Conv1d(indim, 1, kernel_size=1)

    def forward_mask(self, x):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        all_x = x[:,:self.allch,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        phasemask = self.final(out)  # (B, 1, 256)
        return phasemask
    
    def forward(self, x, phasemask, isfe=False):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        oneattended_list = []
        
        ###########One Channel##################
        for i in range(self.allch):
            imu_i = x[:, i:i+1, :]
            out = self.imu_blocks4[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus4[i](out)              # GRU
            out = self.dropouts4[i](out)             # Dropout
            attended = self.attentions4[i](out)      # Attention
            clssf.append(self.classf4[i](attended))
            oneattended_list.append(attended)

        oneattended_list = torch.cat(oneattended_list, dim=-1)
        oneattended_list = self.precls(oneattended_list)
        
        all_x = x[:,:20,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        # phasemask = self.final(out)  # (B, 1, 256)
        out = out.transpose(1, 2)# (B, T, D)
        out,_= self.fn_gru(out)              # GRU
        out = self.fn_dropouts(out)             # Dropout
        attended = self.fn_atten(out)
        clssf.append(self.allcls(attended))
        attended_list.append(attended)

        # !!!!!!!!!!!phase   000111000
        sigac1 = phasemask.sigmoid() # (B, 1, 256)
        sigac2 = 1 - sigac1
        ges = x * sigac1
        trans =  x * sigac2   
        
        for i in range(self.num_branches):
            imu_i = imu_splits[i]
            out = self.imu_blocks[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            pre.append(out)
            out,_= self.grus[i](out)              # GRU
            out = self.dropouts[i](out)             # Dropout
            attended = self.attentions[i](out)      # Attention
            clssf.append(self.classf[i](attended))
            attended_list.append(attended)

        attended_ges = []
        ii = 0
        for i, k in enumerate(self.imu_ind):
            imu_i = ges[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks2[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus2[i](out)              # GRU
            out = self.dropouts2[i](out)             # Dropout
            attended = self.attentions2[i](out)      # Attention
            clssf.append(self.classf2[i](attended))
            attended_ges.append(attended)

        ii = 0
        attended_trans = []
        for i, k in enumerate(self.imu_ind):
            imu_i = trans[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks3[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus3[i](out)              # GRU
            out = self.dropouts3[i](out)             # Dropout
            attended = self.attentions3[i](out)      # Attention
            clssf.append(self.classf3[i](attended))
            attended_trans.append(attended)

        attended_one = torch.cat(attended_list, dim=-1)
        attended_trans = torch.cat(attended_trans, dim=-1)
        attended_ges = torch.cat(attended_ges, dim=-1)
        # oneattended_list = torch.cat(oneattended_list, dim=-1)

        finalatten = torch.cat([attended_ges, attended_trans, attended_one, oneattended_list], dim=-1)

        clssf.append(self.finalallcls(attended_one))
        clssf.append(self.transcls(attended_trans))
        clssf.append(self.gescls(attended_ges))
        clssf.append(self.onecls(oneattended_list))
        
        x = F.relu(self.bn_dense1(self.dense1(finalatten)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        if isfe:
            return finalatten, logits, clssf, phasemask
        return logits, clssf, phasemask
        
class TwoBranchModel_IMU_THM_TOF(nn.Module):
    def __init__(self, imu_ind, n_classes, fold=0):
        super().__init__()
        self.imu_dim = 31
        self.tof_dim = 320
        self.imu_ind = imu_ind
        the_dim = 5
        self.the_dim = the_dim
        self.n_classes = n_classes
        
        indim = 128
        self.num_branches = len(imu_ind)
        
        self.allch = np.sum(self.imu_ind)

        
        self.model_imu = IMUModel(self.imu_ind,18)

        ################################TOF BLOCK#########################
        # self.tof_block = nn.Sequential(
        #         ResidualSECNNBlock(320 + 31 + 5, indim, 3, dropout=0.3),
        #         ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        # )
        # self.tof_block2 = nn.Sequential(
        #         ResidualSECNNBlock(64 + self.allch, indim, 3, dropout=0.3),
        #         ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        # )        
        # self.tof_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        # self.tof_gru_dropout = nn.Dropout(0.4)
        # self.tof_attention = AttentionLayer(indim)
        # self.tof_cls_conv1d = nn.Sequential(
        #         nn.Linear(indim, 256, bias=False),
        #         nn.BatchNorm1d(256),
        #         nn.Dropout(0.5),
        #         nn.Linear(256, n_classes)
        # )
        
        self.tof_encoder = nn.Sequential(
            nn.Conv3d(5, 32, kernel_size=(3, 3, 3), padding=1),  # [B, 32, 256, 8, 8]
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # [B, 32, 128, 4, 4]

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),  # [B, 64, 128, 4, 4]
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # [B, 64, 64, 2, 2]

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),  # [B, 128, 64, 2, 2]
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # [B, 128, 1, 1, 1]
            nn.Flatten()
        )
        
        # Thermal
        self.the_block = nn.Sequential(
                ResidualSECNNBlock(the_dim, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.the_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.the_gru_dropout = nn.Dropout(0.4)
        self.the_attention = AttentionLayer(indim)

        
        self.tof_cls = nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.the_cls = nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        self.imuprecls = nn.Sequential(
            nn.Linear(len(self.imu_ind) * indim * 3  + indim + indim*4 , 4*indim, bias=False),
            nn.BatchNorm1d(4*indim),
            nn.Dropout(0.5),
        )
        
        self.imucls = nn.Sequential(
            nn.Linear(4*indim , 2*indim, bias=False),
            nn.BatchNorm1d(2*indim),
            nn.Dropout(0.5),
            nn.Linear(2*indim, n_classes)
        )
        
        # Dense layers
        self.dense1 = nn.Linear(4*indim + indim + indim , 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, allx, phasemask):
        
        B, T, D = allx.shape
        imux = allx[:,:,:self.allch]
        tofx = allx[:,:,self.allch:]
        
        # IMU=======================================================
        finalatten, logits, clssf, phasemask = self.model_imu(imux[:,:,:self.allch], phasemask, True)
        finalatten = self.imuprecls(finalatten)
        
        # THE======================================================
        the = tofx[:,:,:5]
        the = the.transpose(1, 2)  # (batch, imu_dim, seq_len)
        x3 = self.the_block(the)
        x3 = x3.transpose(1, 2)
        x3, _ = self.the_gru(x3)
        x3 = self.the_gru_dropout(x3)
        attendedthe = self.the_attention(x3)
        clssf.append(self.the_cls(attendedthe))

        # TOF=====================================================
        tof = tofx[:,:,5:]
        # tof1d = tofx[:,:,5:].transpose(1,2)
        tof = tof.view(B,T, 5, 8, 8)
        tof = tof.permute(0,2,1,3,4)
        attendedtof = self.tof_encoder(tof)
        clssf.append(self.tof_cls(attendedtof))
        clssf.append(self.imucls(finalatten))
        
        attendedfinal = torch.cat([finalatten, attendedtof, attendedthe],dim=1)
        # åˆ†ç±»
        x = F.relu(self.bn_dense1(self.dense1(attendedfinal)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        return logits, clssf


model_alls4 = []
device = "cuda:0"
md = [
    "/kaggle/input/cminew/bets_0_875_addmask.pt",
    "/kaggle/input/cminew/bets_1_875_addmask.pt",
    "/kaggle/input/cminew/bets_2_875_addmask.pt",
    "/kaggle/input/cminew/bets_3_875_addmask.pt",
    "/kaggle/input/cminew/bets_4_875_addmask.pt"
]
for fold in range(5):
    checkpoint = torch.load(md[fold], map_location=device)
    model_all = TwoBranchModel_IMU_THM_TOF([7,4,5,4,11,14],18).to(device)
    model_all.load_state_dict(checkpoint)
    model_all.eval();
    model_alls4.append(model_all)


class IMUModel(nn.Module):
    def __init__(self, imu_ind, n_classes, weight_decay=1e-4):
        super().__init__()
        self.imu_ind = imu_ind
        self.imu_dim = len(imu_ind)
        self.tof_dim = 320
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        indim = 128

        self.pool = nn.AdaptiveAvgPool1d(1)
        # ä¿®æ”¹ conv lstm conv lstm
        self.num_branches = len(imu_ind)
        self.imu_blocks = nn.ModuleList()
        self.grus = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.classf = nn.ModuleList()

        self.imu_blocks2 = nn.ModuleList()
        self.grus2 = nn.ModuleList()
        self.dropouts2 = nn.ModuleList()
        self.attentions2 = nn.ModuleList()
        self.classf2 = nn.ModuleList()
        

        self.imu_blocks3 = nn.ModuleList()
        self.grus3 = nn.ModuleList()
        self.dropouts3 = nn.ModuleList()
        self.attentions3 = nn.ModuleList()
        self.classf3 = nn.ModuleList()


        self.imu_blocks4 = nn.ModuleList()
        self.grus4 = nn.ModuleList()
        self.dropouts4 = nn.ModuleList()
        self.attentions4 = nn.ModuleList()
        self.classf4 = nn.ModuleList()
        
        self.allch = np.sum(self.imu_ind)

        #############one channel loss############################
        for i in range(self.allch):
            self.imu_blocks4.append(nn.Sequential(
                ResidualSECNNBlock(1, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus4.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts4.append(nn.Dropout(0.5))
            self.attentions4.append(AttentionLayer(indim))
            self.classf4.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )        
        ############################ALL#########################
        for in_channels in self.imu_ind:
            self.imu_blocks.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts.append(nn.Dropout(0.5))
            self.attentions.append(AttentionLayer(indim))
            self.classf.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        self.allcls =  nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
        )
        ############################Trans#########################
        for in_channels in self.imu_ind:
            self.imu_blocks2.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus2.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts2.append(nn.Dropout(0.5))
            self.attentions2.append(AttentionLayer(indim))
            # self.exp.append(Expert(indim, indim, n_classes))
            self.classf2.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        ############################Gesture#########################
        for in_channels in self.imu_ind:
            self.imu_blocks3.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus3.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts3.append(nn.Dropout(0.5))
            self.attentions3.append(AttentionLayer(indim))
            self.classf3.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
            
        # all block
        self.fn_block = nn.Sequential(
            ResidualSECNNBlock(20, indim, 3, dropout=0.3),
            ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.fn_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.fn_dropouts = nn.Dropout(0.5)
        self.fn_atten = AttentionLayer(indim)

        ### classification
        self.dense1 = nn.Linear(len(self.imu_ind) * indim * 3  + indim + indim*4 , 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

        self.finalallcls = nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim + indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.transcls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.gescls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.onecls =  nn.Sequential(
                nn.Linear(4 * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.precls = nn.Sequential(
                nn.Linear(self.allch * indim, 4 * indim, bias=False),
                nn.BatchNorm1d(4 * indim),
                nn.Dropout(0.5),
                )
        
        
        self.final = nn.Conv1d(indim, 1, kernel_size=1)

    def forward_mask(self, x):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        all_x = x[:,:self.allch,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        phasemask = self.final(out)  # (B, 1, 256)
        return phasemask
    
    def forward(self, x, phasemask, isfe=False):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        oneattended_list = []
        
        ###########One Channel##################
        for i in range(self.allch):
            imu_i = x[:, i:i+1, :]
            out = self.imu_blocks4[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus4[i](out)              # GRU
            out = self.dropouts4[i](out)             # Dropout
            attended = self.attentions4[i](out)      # Attention
            clssf.append(self.classf4[i](attended))
            oneattended_list.append(attended)

        oneattended_list = torch.cat(oneattended_list, dim=-1)
        oneattended_list = self.precls(oneattended_list)
        
        all_x = x[:,:20,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        # phasemask = self.final(out)  # (B, 1, 256)
        out = out.transpose(1, 2)# (B, T, D)
        out,_= self.fn_gru(out)              # GRU
        out = self.fn_dropouts(out)             # Dropout
        attended = self.fn_atten(out)
        clssf.append(self.allcls(attended))
        attended_list.append(attended)

        # !!!!!!!!!!!phase   000111000
        sigac1 = phasemask.sigmoid() # (B, 1, 256)
        sigac2 = 1 - sigac1
        ges = x * sigac1
        trans =  x * sigac2   
        
        for i in range(self.num_branches):
            imu_i = imu_splits[i]
            out = self.imu_blocks[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            pre.append(out)
            out,_= self.grus[i](out)              # GRU
            out = self.dropouts[i](out)             # Dropout
            attended = self.attentions[i](out)      # Attention
            clssf.append(self.classf[i](attended))
            attended_list.append(attended)

        attended_ges = []
        ii = 0
        for i, k in enumerate(self.imu_ind):
            imu_i = ges[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks2[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus2[i](out)              # GRU
            out = self.dropouts2[i](out)             # Dropout
            attended = self.attentions2[i](out)      # Attention
            clssf.append(self.classf2[i](attended))
            attended_ges.append(attended)

        ii = 0
        attended_trans = []
        for i, k in enumerate(self.imu_ind):
            imu_i = trans[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks3[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus3[i](out)              # GRU
            out = self.dropouts3[i](out)             # Dropout
            attended = self.attentions3[i](out)      # Attention
            clssf.append(self.classf3[i](attended))
            attended_trans.append(attended)

        attended_one = torch.cat(attended_list, dim=-1)
        attended_trans = torch.cat(attended_trans, dim=-1)
        attended_ges = torch.cat(attended_ges, dim=-1)
        # oneattended_list = torch.cat(oneattended_list, dim=-1)

        finalatten = torch.cat([attended_ges, attended_trans, attended_one, oneattended_list], dim=-1)

        clssf.append(self.finalallcls(attended_one))
        clssf.append(self.transcls(attended_trans))
        clssf.append(self.gescls(attended_ges))
        clssf.append(self.onecls(oneattended_list))
        
        x = F.relu(self.bn_dense1(self.dense1(finalatten)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        if isfe:
            return attended_one, logits, clssf, pre
        return logits, clssf, phasemask


model_imu2 = []
device = "cuda:0"
md = [
    "/kaggle/input/cmimodel/bets_0_0.838.pt",
    "/kaggle/input/cmimodel/bets_1_0.838.pt",
    "/kaggle/input/cmimodel/bets_2_0.838.pt",
    "/kaggle/input/cmimodel/bets_3_0.838.pt",
    "/kaggle/input/cmimodel/bets_4_0.838.pt"
]
for fold in range(5):
    checkpoint = torch.load(md[fold], map_location=device)
    model_all = IMUModel([7,4,5,4,11,14],18).to(device)
    model_all.load_state_dict(checkpoint)
    model_all.eval();
    model_imu2.append(model_all)


model_imu3 = []
device = "cuda:0"
md = [
    "/kaggle/input/cminew/0_842.pt",
    "/kaggle/input/cminew/1_842.pt",
    "/kaggle/input/cminew/2_842.pt",
    "/kaggle/input/cminew/3_842.pt",
    "/kaggle/input/cminew/4_842.pt"
]
for fold in range(5):
    checkpoint = torch.load(md[fold], map_location=device)
    model_all = IMUModel([7,4,5,4,11,14],18).to(device)
    model_all.load_state_dict(checkpoint)
    model_all.eval();
    model_imu3.append(model_all)


class IMUModel(nn.Module):
    def __init__(self, imu_ind, n_classes, weight_decay=1e-4):
        super().__init__()
        self.imu_ind = imu_ind
        self.imu_dim = len(imu_ind)
        self.tof_dim = 320
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        indim = 128

        self.pool = nn.AdaptiveAvgPool1d(1)
        # ä¿®æ”¹ conv lstm conv lstm
        self.num_branches = len(imu_ind)
        self.imu_blocks = nn.ModuleList()
        self.grus = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.classf = nn.ModuleList()

        self.imu_blocks2 = nn.ModuleList()
        self.grus2 = nn.ModuleList()
        self.dropouts2 = nn.ModuleList()
        self.attentions2 = nn.ModuleList()
        self.classf2 = nn.ModuleList()
        

        self.imu_blocks3 = nn.ModuleList()
        self.grus3 = nn.ModuleList()
        self.dropouts3 = nn.ModuleList()
        self.attentions3 = nn.ModuleList()
        self.classf3 = nn.ModuleList()


        self.imu_blocks4 = nn.ModuleList()
        self.grus4 = nn.ModuleList()
        self.dropouts4 = nn.ModuleList()
        self.attentions4 = nn.ModuleList()
        self.classf4 = nn.ModuleList()
        
        self.allch = np.sum(self.imu_ind)

        #############one channel loss############################
        for i in range(self.allch):
            self.imu_blocks4.append(nn.Sequential(
                ResidualSECNNBlock(1, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus4.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts4.append(nn.Dropout(0.5))
            self.attentions4.append(AttentionLayer(indim))
            self.classf4.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )        
        ############################ALL#########################
        for in_channels in self.imu_ind:
            self.imu_blocks.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts.append(nn.Dropout(0.5))
            self.attentions.append(AttentionLayer(indim))
            self.classf.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        self.allcls =  nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
        )
        ############################Trans#########################
        for in_channels in self.imu_ind:
            self.imu_blocks2.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus2.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts2.append(nn.Dropout(0.5))
            self.attentions2.append(AttentionLayer(indim))
            # self.exp.append(Expert(indim, indim, n_classes))
            self.classf2.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        ############################Gesture#########################
        for in_channels in self.imu_ind:
            self.imu_blocks3.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus3.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts3.append(nn.Dropout(0.5))
            self.attentions3.append(AttentionLayer(indim))
            self.classf3.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
            
        # all block
        self.fn_block = nn.Sequential(
            ResidualSECNNBlock(20, indim, 3, dropout=0.3),
            ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.fn_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.fn_dropouts = nn.Dropout(0.5)
        self.fn_atten = AttentionLayer(indim)

        ### classification
        self.dense1 = nn.Linear(len(self.imu_ind) * indim * 3  + indim + indim*4 + indim, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

        self.finalallcls = nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim + indim + indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.transcls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.gescls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.onecls =  nn.Sequential(
                nn.Linear(4 * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.precls = nn.Sequential(
                nn.Linear(self.allch * indim, 4 * indim, bias=False),
                nn.BatchNorm1d(4 * indim),
                nn.Dropout(0.5),
                )
        
        
        self.final = nn.Conv1d(indim, 1, kernel_size=1)

        self.pre_dim = nn.Sequential(
                nn.Linear(indim, 1, bias=False),
                nn.BatchNorm1d(indim),
                nn.Dropout(0.5),
                )
        # mid fea
        self.mid_block = nn.Sequential(
            ResidualSECNNBlock(indim * self.num_branches, indim*2, 3, dropout=0.3),
            ResidualSECNNBlock(indim * 2, indim, 3, dropout=0.3),
        )
        self.mid_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.mid_dropouts = nn.Dropout(0.5)
        self.mid_atten = AttentionLayer(indim)
        self.mid_cls = nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
    def forward_mask(self, x):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        all_x = x[:,:self.allch,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        phasemask = self.final(out)  # (B, 1, 256)
        return phasemask
    
    def forward(self, x, phasemask, isfe=False):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        
        clssf = []
        pre = []
        attended_list = []

        oneattended_list = []
        
        ###########One Channel##################
        for i in range(self.allch):
            imu_i = x[:, i:i+1, :]
            out = self.imu_blocks4[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            pre.append(out)
            out,_= self.grus4[i](out)              # GRU
            out = self.dropouts4[i](out)             # Dropout
            attended = self.attentions4[i](out)      # Attention
            clssf.append(self.classf4[i](attended))
            oneattended_list.append(attended)

        oneattended_list = torch.cat(oneattended_list, dim=-1)
        oneattended_list = self.precls(oneattended_list)

        pre = torch.stack(pre,dim=1) # B * CH * 256 * 128
        # pre = self.pre_dim(pre)[0] # B * CH * 256
        pre = pre.mean(dim=-1)  # (B, CH, 256)
        
        all_x = pre[:,:20,:]
        out = self.fn_block(all_x)         # CNN block
        out = out.transpose(1, 2)# (B, T, D)
        out,_= self.fn_gru(out)              # GRU
        out = self.fn_dropouts(out)             # Dropout
        attended = self.fn_atten(out)
        clssf.append(self.allcls(attended))
        attended_list.append(attended)

        # !!!!!!!!!!!phase   000111000
        sigac1 = phasemask.sigmoid() # (B, 1, 256)
        sigac2 = 1 - sigac1
        ges = x * sigac1
        trans =  x * sigac2   

        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(pre[:, ct:ct+k, :])
            ct = ct + k

        midfe = []
        for i in range(self.num_branches):
            imu_i = imu_splits[i]
            out = self.imu_blocks[i](imu_i)         # CNN block
            out = out.transpose(1, 2)            # (B, T, D)
            midfe.append(out)
            out,_= self.grus[i](out)              # GRU
            out = self.dropouts[i](out)             # Dropout
            attended = self.attentions[i](out)      # Attention
            clssf.append(self.classf[i](attended))
            attended_list.append(attended)

        midfe = torch.stack(midfe,dim=1) # B * N * 256 * 128
        bb,nn,tt,ff = midfe.shape
        midfe = midfe.permute(0,1,3,2)
        midfe = midfe.reshape(bb,nn*ff,tt) # B * D * T
        midfe = self.mid_block(midfe)
        midfe = midfe.transpose(1, 2)# (B, T, D)
        midfe,_ = self.mid_gru(midfe)              # GRU
        midfe = self.mid_dropouts(midfe)             # Dropout
        midfe = self.mid_atten(midfe)
        clssf.append(self.mid_cls(midfe))
        attended_list.append(midfe)
        
        attended_ges = []
        ii = 0
        for i, k in enumerate(self.imu_ind):
            imu_i = ges[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks2[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus2[i](out)              # GRU
            out = self.dropouts2[i](out)             # Dropout
            attended = self.attentions2[i](out)      # Attention
            clssf.append(self.classf2[i](attended))
            attended_ges.append(attended)

        ii = 0
        attended_trans = []
        for i, k in enumerate(self.imu_ind):
            imu_i = trans[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks3[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus3[i](out)              # GRU
            out = self.dropouts3[i](out)             # Dropout
            attended = self.attentions3[i](out)      # Attention
            clssf.append(self.classf3[i](attended))
            attended_trans.append(attended)

        attended_one = torch.cat(attended_list, dim=-1)
        attended_trans = torch.cat(attended_trans, dim=-1)
        attended_ges = torch.cat(attended_ges, dim=-1)
        # oneattended_list = torch.cat(oneattended_list, dim=-1)

        finalatten = torch.cat([attended_ges, attended_trans, attended_one, oneattended_list], dim=-1)

        clssf.append(self.finalallcls(attended_one))
        clssf.append(self.transcls(attended_trans))
        clssf.append(self.gescls(attended_ges))
        clssf.append(self.onecls(oneattended_list))
        
        x = F.relu(self.bn_dense1(self.dense1(finalatten)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        if isfe:
            return attended_one, logits, clssf, pre
        return logits, clssf, phasemask


model_imu4 = []
device = "cuda:0"
md = [
    "/kaggle/input/cmimodel/bets_0_831_imumodelchange.pt",
    "/kaggle/input/cmimodel/bets_1_831_imumodelchange.pt",
    "/kaggle/input/cmimodel/bets_2_831_imumodelchange.pt",
    "/kaggle/input/cmimodel/bets_3_831_imumodelchange.pt",
    "/kaggle/input/cmimodel/bets_4_831_imumodelchange.pt"
]
for fold in range(5):
    checkpoint = torch.load(md[fold], map_location=device)
    model_all = IMUModel([7,4,5,4,11,14],18).to(device)
    model_all.load_state_dict(checkpoint)
    model_all.eval();
    model_imu4.append(model_all)


class IMUModel(nn.Module):
    def __init__(self, imu_ind, n_classes, weight_decay=1e-4):
        super().__init__()
        self.imu_ind = imu_ind
        self.imu_dim = len(imu_ind)
        self.tof_dim = 320
        self.n_classes = n_classes
        self.weight_decay = weight_decay
        indim = 128

        self.pool = nn.AdaptiveAvgPool1d(1)
        # ä¿®æ”¹ conv lstm conv lstm
        self.num_branches = len(imu_ind)
        self.imu_blocks = nn.ModuleList()
        self.grus = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.attentions = nn.ModuleList()
        self.classf = nn.ModuleList()

        self.imu_blocks2 = nn.ModuleList()
        self.grus2 = nn.ModuleList()
        self.dropouts2 = nn.ModuleList()
        self.attentions2 = nn.ModuleList()
        self.classf2 = nn.ModuleList()
        

        self.imu_blocks3 = nn.ModuleList()
        self.grus3 = nn.ModuleList()
        self.dropouts3 = nn.ModuleList()
        self.attentions3 = nn.ModuleList()
        self.classf3 = nn.ModuleList()


        self.imu_blocks4 = nn.ModuleList()
        self.grus4 = nn.ModuleList()
        self.dropouts4 = nn.ModuleList()
        self.attentions4 = nn.ModuleList()
        self.classf4 = nn.ModuleList()
        
        self.allch = np.sum(self.imu_ind)

        #############one channel loss############################
        for i in range(self.allch):
            self.imu_blocks4.append(nn.Sequential(
                ResidualSECNNBlock(1, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus4.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts4.append(nn.Dropout(0.5))
            self.attentions4.append(AttentionLayer(indim))
            self.classf4.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )        
        ############################ALL#########################
        for in_channels in self.imu_ind:
            self.imu_blocks.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts.append(nn.Dropout(0.5))
            self.attentions.append(AttentionLayer(indim))
            self.classf.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        self.allcls =  nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
        )
        ############################Trans#########################
        for in_channels in self.imu_ind:
            self.imu_blocks2.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus2.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts2.append(nn.Dropout(0.5))
            self.attentions2.append(AttentionLayer(indim))
            # self.exp.append(Expert(indim, indim, n_classes))
            self.classf2.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
        ############################Gesture#########################
        for in_channels in self.imu_ind:
            self.imu_blocks3.append(nn.Sequential(
                ResidualSECNNBlock(in_channels, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
            ))
            self.grus3.append(nn.GRU(indim, indim, bidirectional=False, batch_first=True))
            self.dropouts3.append(nn.Dropout(0.5))
            self.attentions3.append(AttentionLayer(indim))
            self.classf3.append(
                nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
            )
            
        # all block
        self.fn_block = nn.Sequential(
            ResidualSECNNBlock(20, indim, 3, dropout=0.3),
            ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.fn_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.fn_dropouts = nn.Dropout(0.5)
        self.fn_atten = AttentionLayer(indim)

        ### classification
        self.dense1 = nn.Linear(len(self.imu_ind) * indim * 3  + indim + indim*4 , 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

        self.finalallcls = nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim + indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.transcls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.gescls =  nn.Sequential(
                nn.Linear(len(self.imu_ind) * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.onecls =  nn.Sequential(
                nn.Linear(4 * indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )

        self.precls = nn.Sequential(
                nn.Linear(self.allch * indim, 4 * indim, bias=False),
                nn.BatchNorm1d(4 * indim),
                nn.Dropout(0.5),
                )
        
        
        self.final = nn.Conv1d(indim, 1, kernel_size=1)

    def forward_mask(self, x):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        all_x = x[:,:self.allch,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        phasemask = self.final(out)  # (B, 1, 256)
        return phasemask
    
    def forward(self, x, phasemask, isfe=False):
        x = x.transpose(1, 2)  # (batch, imu_dim, seq_len)
        imu_splits = []
        ct = 0
        for k in self.imu_ind:
            imu_splits.append(x[:, ct:ct+k, :])
            ct = ct + k
        clssf = []
        pre = []
        attended_list = []

        oneattended_list = []
        
        ###########One Channel##################
        for i in range(self.allch):
            imu_i = x[:, i:i+1, :]
            out = self.imu_blocks4[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus4[i](out)              # GRU
            out = self.dropouts4[i](out)             # Dropout
            attended = self.attentions4[i](out)      # Attention
            clssf.append(self.classf4[i](attended))
            oneattended_list.append(attended)

        oneattended_list = torch.cat(oneattended_list, dim=-1)
        oneattended_list = self.precls(oneattended_list)
        
        all_x = x[:,:20,:]
        out = self.fn_block(all_x)         # CNN block
        # é¢„æµ‹phasemask
        # phasemask = self.final(out)  # (B, 1, 256)
        out = out.transpose(1, 2)# (B, T, D)
        out,_= self.fn_gru(out)              # GRU
        out = self.fn_dropouts(out)             # Dropout
        attended = self.fn_atten(out)
        clssf.append(self.allcls(attended))
        attended_list.append(attended)

        # !!!!!!!!!!!phase   000111000
        sigac1 = phasemask.sigmoid() # (B, 1, 256)
        sigac2 = 1 - sigac1
        ges = x * sigac1
        trans =  x * sigac2   
        
        for i in range(self.num_branches):
            imu_i = imu_splits[i]
            out = self.imu_blocks[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            pre.append(out)
            out,_= self.grus[i](out)              # GRU
            out = self.dropouts[i](out)             # Dropout
            attended = self.attentions[i](out)      # Attention
            clssf.append(self.classf[i](attended))
            attended_list.append(attended)

        attended_ges = []
        ii = 0
        for i, k in enumerate(self.imu_ind):
            imu_i = ges[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks2[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus2[i](out)              # GRU
            out = self.dropouts2[i](out)             # Dropout
            attended = self.attentions2[i](out)      # Attention
            clssf.append(self.classf2[i](attended))
            attended_ges.append(attended)

        ii = 0
        attended_trans = []
        for i, k in enumerate(self.imu_ind):
            imu_i = trans[:,ii:ii+k,:]
            ii = ii + k
            out = self.imu_blocks3[i](imu_i)         # CNN block
            out = out.transpose(1, 2)# (B, T, D)
            out,_= self.grus3[i](out)              # GRU
            out = self.dropouts3[i](out)             # Dropout
            attended = self.attentions3[i](out)      # Attention
            clssf.append(self.classf3[i](attended))
            attended_trans.append(attended)

        attended_one = torch.cat(attended_list, dim=-1)
        attended_trans = torch.cat(attended_trans, dim=-1)
        attended_ges = torch.cat(attended_ges, dim=-1)
        # oneattended_list = torch.cat(oneattended_list, dim=-1)

        finalatten = torch.cat([attended_ges, attended_trans, attended_one, oneattended_list], dim=-1)

        clssf.append(self.finalallcls(attended_one))
        clssf.append(self.transcls(attended_trans))
        clssf.append(self.gescls(attended_ges))
        clssf.append(self.onecls(oneattended_list))
        
        x = F.relu(self.bn_dense1(self.dense1(finalatten)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        if isfe:
            return finalatten, logits, clssf, phasemask
        return logits, clssf, phasemask


class TwoBranchModel_IMU_THM_TOF(nn.Module):
    def __init__(self, imu_ind, n_classes, fold=0):
        super().__init__()
        self.imu_dim = 31
        self.tof_dim = 320
        self.imu_ind = imu_ind
        the_dim = 5
        self.the_dim = the_dim
        self.n_classes = n_classes
        
        indim = 128
        self.num_branches = len(imu_ind)
        
        self.allch = np.sum(self.imu_ind)

        
        self.model_imu = IMUModel(self.imu_ind,18)

        ################################TOF BLOCK#########################
        # self.tof_block = nn.Sequential(
        #         ResidualSECNNBlock(320 + 31 + 5, indim, 3, dropout=0.3),
        #         ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        # )
        # self.tof_block2 = nn.Sequential(
        #         ResidualSECNNBlock(64 + self.allch, indim, 3, dropout=0.3),
        #         ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        # )        
        # self.tof_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        # self.tof_gru_dropout = nn.Dropout(0.4)
        # self.tof_attention = AttentionLayer(indim)
        # self.tof_cls_conv1d = nn.Sequential(
        #         nn.Linear(indim, 256, bias=False),
        #         nn.BatchNorm1d(256),
        #         nn.Dropout(0.5),
        #         nn.Linear(256, n_classes)
        # )
        
        self.tof_encoder = nn.Sequential(
            nn.Conv3d(5, 32, kernel_size=(3, 3, 3), padding=1),  # [B, 32, 256, 8, 8]
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # [B, 32, 128, 4, 4]

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),  # [B, 64, 128, 4, 4]
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # [B, 64, 64, 2, 2]

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),  # [B, 128, 64, 2, 2]
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # [B, 128, 1, 1, 1]
            nn.Flatten()
        )
        
        # Thermal
        self.the_block = nn.Sequential(
                ResidualSECNNBlock(the_dim, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.the_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.the_gru_dropout = nn.Dropout(0.4)
        self.the_attention = AttentionLayer(indim)

        
        self.tof_cls = nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.the_cls = nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        self.imuprecls = nn.Sequential(
            nn.Linear(len(self.imu_ind) * indim * 3  + indim + indim*4 , 4*indim, bias=False),
            nn.BatchNorm1d(4*indim),
            nn.Dropout(0.5),
        )
        
        self.imucls = nn.Sequential(
            nn.Linear(4*indim , 2*indim, bias=False),
            nn.BatchNorm1d(2*indim),
            nn.Dropout(0.5),
            nn.Linear(2*indim, n_classes)
        )
        
        # Dense layers
        self.dense1 = nn.Linear(4*indim + indim + indim , 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, allx, phasemask):
        
        B, T, D = allx.shape
        imux = allx[:,:,:self.allch]
        tofx = allx[:,:,self.allch:]
        
        # IMU=======================================================
        finalatten, logits, clssf, phasemask = self.model_imu(imux[:,:,:self.allch], phasemask, True)
        finalatten = self.imuprecls(finalatten)
        
        # THE======================================================
        the = tofx[:,:,:5]
        the = the.transpose(1, 2)  # (batch, imu_dim, seq_len)
        x3 = self.the_block(the)
        x3 = x3.transpose(1, 2)
        x3, _ = self.the_gru(x3)
        x3 = self.the_gru_dropout(x3)
        attendedthe = self.the_attention(x3)
        clssf.append(self.the_cls(attendedthe))

        # TOF=====================================================
        tof = tofx[:,:,5:]
        # tof1d = tofx[:,:,5:].transpose(1,2)
        tof = tof.view(B,T, 5, 8, 8)
        tof = tof.permute(0,2,1,3,4)
        attendedtof = self.tof_encoder(tof)
        clssf.append(self.tof_cls(attendedtof))
        clssf.append(self.imucls(finalatten))
        
        attendedfinal = torch.cat([finalatten, attendedtof, attendedthe],dim=1)
        # åˆ†ç±»
        x = F.relu(self.bn_dense1(self.dense1(attendedfinal)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        return logits, clssf


model_alls5 = []
device = "cuda:0"
md = [
    "/kaggle/input/cminew/bets_0_794.pt",
    "/kaggle/input/cminew/bets_1_794.pt",
    "/kaggle/input/cminew/bets_2_794.pt",
    "/kaggle/input/cminew/bets_3_794.pt",
    "/kaggle/input/cminew/bets_4_794.pt"
]
for fold in range(5):
    checkpoint = torch.load(md[fold], map_location=device)
    model_all = TwoBranchModel_IMU_THM_TOF([7,4,5,4,11,14],18).to(device)
    model_all.load_state_dict(checkpoint)
    model_all.eval();
    model_alls5.append(model_all)


model_alls6 = []
device = "cuda:0"
md = [
    "/kaggle/input/cminew/bets_0_879.pt",
    "/kaggle/input/cminew/bets_1_879.pt",
    "/kaggle/input/cminew/bets_2_879.pt",
    "/kaggle/input/cminew/bets_3_879.pt",
    "/kaggle/input/cminew/bets_4_879.pt"
]
for fold in range(5):
    checkpoint = torch.load(md[fold], map_location=device)
    model_all = TwoBranchModel_IMU_THM_TOF([7,4,5,4,11,14],18).to(device)
    model_all.load_state_dict(checkpoint)
    model_all.eval();
    model_alls6.append(model_all)


class TwoBranchModel_IMU_THM_TOF(nn.Module):
    def __init__(self, imu_ind, n_classes, fold=0):
        super().__init__()
        self.imu_dim = 31
        self.tof_dim = 320
        self.imu_ind = imu_ind
        the_dim = 5
        self.the_dim = the_dim
        self.n_classes = n_classes
        
        indim = 128
        self.num_branches = len(imu_ind)
        
        self.allch = np.sum(self.imu_ind)

        
        self.model_imu = IMUModel(self.imu_ind,18)

        ################################TOF BLOCK#########################
        # self.tof_block = nn.Sequential(
        #         ResidualSECNNBlock(320 + 31 + 5, indim, 3, dropout=0.3),
        #         ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        # )
        # self.tof_block2 = nn.Sequential(
        #         ResidualSECNNBlock(64 + self.allch, indim, 3, dropout=0.3),
        #         ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        # )        
        # self.tof_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        # self.tof_gru_dropout = nn.Dropout(0.4)
        # self.tof_attention = AttentionLayer(indim)
        # self.tof_cls_conv1d = nn.Sequential(
        #         nn.Linear(indim, 256, bias=False),
        #         nn.BatchNorm1d(256),
        #         nn.Dropout(0.5),
        #         nn.Linear(256, n_classes)
        # )
        
        self.tof_encoder = nn.Sequential(
            nn.Conv3d(5, 32, kernel_size=(3, 3, 3), padding=1),  # [B, 32, 256, 8, 8]
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # [B, 32, 128, 4, 4]

            nn.Conv3d(32, 64, kernel_size=(3, 3, 3), padding=1),  # [B, 64, 128, 4, 4]
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d((2, 2, 2)),  # [B, 64, 64, 2, 2]

            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=1),  # [B, 128, 64, 2, 2]
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((1, 1, 1)),  # [B, 128, 1, 1, 1]
            nn.Flatten()
        )
        
        # Thermal
        self.the_block = nn.Sequential(
                ResidualSECNNBlock(the_dim, indim, 3, dropout=0.3),
                ResidualSECNNBlock(indim, indim, 3, dropout=0.3),
        )
        self.the_gru = nn.GRU(indim, indim, bidirectional=False, batch_first=True)
        self.the_gru_dropout = nn.Dropout(0.4)
        self.the_attention = AttentionLayer(indim)

        
        self.tof_cls = nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        
        self.the_cls = nn.Sequential(
                nn.Linear(indim, 256, bias=False),
                nn.BatchNorm1d(256),
                nn.Dropout(0.5),
                nn.Linear(256, n_classes)
                )
        self.imuprecls = nn.Sequential(
            nn.Linear(len(self.imu_ind) * indim * 3  + indim + indim*4 , 4*indim, bias=False),
            nn.BatchNorm1d(4*indim),
            nn.Dropout(0.5),
        )
        
        self.imucls = nn.Sequential(
            nn.Linear(4*indim , 2*indim, bias=False),
            nn.BatchNorm1d(2*indim),
            nn.Dropout(0.5),
            nn.Linear(2*indim, n_classes)
        )
        
        # Dense layers
        self.dense1 = nn.Linear(4*indim + indim + indim , 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

        self.classaux = nn.Sequential(
            nn.Linear(4*indim + indim + indim , 256, bias=False),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1)
        )
        
    def forward(self, allx, phasemask):
        
        B, T, D = allx.shape
        imux = allx[:,:,:self.allch]
        tofx = allx[:,:,self.allch:]
        
        # IMU=======================================================
        finalatten, logits, clssf, phasemask = self.model_imu(imux[:,:,:self.allch], phasemask, True)
        finalatten = self.imuprecls(finalatten)
        
        # THE======================================================
        the = tofx[:,:,:5]
        the = the.transpose(1, 2)  # (batch, imu_dim, seq_len)
        x3 = self.the_block(the)
        x3 = x3.transpose(1, 2)
        x3, _ = self.the_gru(x3)
        x3 = self.the_gru_dropout(x3)
        attendedthe = self.the_attention(x3)
        clssf.append(self.the_cls(attendedthe))

        # TOF=====================================================
        tof = tofx[:,:,5:]
        # tof1d = tofx[:,:,5:].transpose(1,2)
        tof = tof.view(B,T, 5, 8, 8)
        tof = tof.permute(0,2,1,3,4)
        attendedtof = self.tof_encoder(tof)
        clssf.append(self.tof_cls(attendedtof))
        clssf.append(self.imucls(finalatten))
        
        attendedfinal = torch.cat([finalatten, attendedtof, attendedthe],dim=1)
        # åˆ†ç±»
        x = F.relu(self.bn_dense1(self.dense1(attendedfinal)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        logits = self.classifier(x)
        return logits, clssf


model_alls7 = []
device = "cuda:0"
md = [
    "/kaggle/input/cminew/bets_0_8809.pt",
    "/kaggle/input/cminew/bets_1_8809.pt",
    "/kaggle/input/cminew/bets_2_8809.pt",
    "/kaggle/input/cminew/bets_3_8809.pt",
    "/kaggle/input/cminew/bets_4_8809.pt"
]
for fold in range(5):
    checkpoint = torch.load(md[fold], map_location=device)
    model_all = TwoBranchModel_IMU_THM_TOF([7,4,5,4,11,14],18).to(device)
    model_all.load_state_dict(checkpoint)
    model_all.eval();
    model_alls7.append(model_all)


max_len = 256
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Prediction function for Kaggle competition"""    
    df_seq = sequence.to_pandas()
    ratio = df_seq["thm_1"].isna().sum() / len(df_seq)
    train = df_seq
    train = feature_engineering(train)
    linear_accel_df = train.groupby('sequence_id').apply(get_rot, include_groups=False)
    linear_accel_df = linear_accel_df.droplevel('sequence_id')
    train = train.join(linear_accel_df)
    linear_accel_df = train.groupby('sequence_id').apply(calculate_angular_velocity_from_quat_ori, include_groups=False)
    linear_accel_df = linear_accel_df.droplevel('sequence_id')
    train = train.join(linear_accel_df)    
    train['ang_diff_x'] = train.groupby('sequence_id')['angular_vel_x'].diff().fillna(0)
    train['ang_diff_y'] = train.groupby('sequence_id')['angular_vel_y'].diff().fillna(0)
    train['ang_diff_z'] = train.groupby('sequence_id')['angular_vel_z'].diff().fillna(0)
    train['ang_diff_the'] = train.groupby('sequence_id')['rot_angle_vel'].diff().fillna(0)

    k = 15
    grouped = train.groupby('sequence_id')
    for fe in (['rot',"angular_vel"]):
        for dir in ('x', 'y', 'z'):
            col_name = f'{fe}_{dir}'
            weight = create_gaussian_kernel(k, 1)  # 1 channel, cause process 1 column per iteration
            lpf_results = []
            for _, group in grouped:
                # convert to tensor and add dimentions (batch, channel, length)
                data = torch.tensor(group[col_name].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                # apply convolution
                lpf = F.conv1d(data, weight, padding=k//2)
                lpf_results.append(lpf.squeeze().numpy())
            
            # concatenate results
            lpf_series = pd.concat([pd.Series(x, index=group.index) for x, (_, group) in zip(lpf_results, grouped)])
            train[f'{fe}_lpf_{dir}'] = lpf_series
            train[f'{fe}_hpf_{dir}'] = train[col_name] - train[f'{fe}_lpf_{dir}']

    grouped = train.groupby('sequence_id')
    for fe in (['acc','linear_acc']):
        for dir in ('x', 'y', 'z', 'mag'):
            col_name = f'{fe}_{dir}'
            weight = create_gaussian_kernel(k, 1)  # 1 channel, cause process 1 column per iteration
            lpf_results = []
            for _, group in grouped:
                # convert to tensor and add dimentions (batch, channel, length)
                data = torch.tensor(group[col_name].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                # apply convolution
                lpf = F.conv1d(data, weight, padding=k//2)
                lpf_results.append(lpf.squeeze().numpy())
            
            # concatenate results
            lpf_series = pd.concat([pd.Series(x, index=group.index) for x, (_, group) in zip(lpf_results, grouped)])
            train[f'{fe}_lpf_{dir}'] = lpf_series
            train[f'{fe}_hpf_{dir}'] = train[col_name] - train[f'{fe}_lpf_{dir}']
            
    train[tof_cols] = train[tof_cols].ffill().bfill().fillna(0).values
    feature_cols = imu_cols+tof_cols
    with torch.no_grad():
        mat = train[imu_cols].values
        pad = pad_sequences_torch([mat], maxlen=pad_len, padding='post', truncating='post')
        x = torch.FloatTensor(pad).to(device)
        phasemask = modeldet.forward_mask(x)

        fnpred = []
        fnlog = []
        if ratio>0.5:
            for fold in range(3,5):
                model = model_imus[fold]
                logits, cls, _ = model(x, phasemask)
                fnlog.append(logits.softmax(dim=1))
                fnpred.append(logits.argmax(dim=1))
            for fold in range(3,5):
                model = model_imu2[fold]
                logits, cls, _ = model(x, phasemask)
                fnlog.append(logits.softmax(dim=1))
                fnpred.append(logits.argmax(dim=1))
            for fold in range(3,5):
                model = model_imu3[fold]
                logits, cls, _ = model(x, phasemask)
                fnlog.append(logits.softmax(dim=1))
                fnpred.append(logits.argmax(dim=1))
            for fold in range(3,5):
                model = model_imu4[fold]
                logits, cls, _ = model(x, phasemask)
                fnlog.append(logits.softmax(dim=1))
                fnpred.append(logits.argmax(dim=1))
        else:
            mat = train[feature_cols].values
            pad = pad_sequences_torch([mat], maxlen=pad_len, padding='post', truncating='post')
            x = torch.FloatTensor(pad).to(device)
            for fold in range(3,5):
                model_all = model_alls[fold]
                logits, cls = model_all(x,phasemask)
                fnlog.append(logits.softmax(dim=1))
                fnpred.append(logits.argmax(dim=1))

            for fold in range(3,5):
                model_all = model_alls2[fold]
                logits, cls = model_all(x,phasemask)
                fnlog.append(logits.softmax(dim=1))
                fnpred.append(logits.argmax(dim=1))

            for fold in range(3,5):
                model_all = model_alls3[fold]
                logits, cls = model_all(x,phasemask)
                fnlog.append(logits.softmax(dim=1))
                fnpred.append(logits.argmax(dim=1))

            for fold in range(3,5):
                model_all = model_alls4[fold]
                logits, cls = model_all(x,phasemask)
                fnlog.append(logits.softmax(dim=1))
                fnpred.append(logits.argmax(dim=1))

            for fold in range(3,5):
                model_all = model_alls5[fold]
                logits, cls = model_all(x,phasemask)
                fnlog.append(logits.softmax(dim=1))
                fnpred.append(logits.argmax(dim=1))

            for fold in range(3,5):
                model_all = model_alls6[fold]
                logits, cls = model_all(x,phasemask)
                fnlog.append(logits.softmax(dim=1))
                fnpred.append(logits.argmax(dim=1))

            for fold in range(3,5):
                model_all = model_alls7[fold]
                logits, cls = model_all(x,phasemask)
                fnlog.append(logits.softmax(dim=1))
                fnpred.append(logits.argmax(dim=1))

        fnlog = torch.stack(fnlog, dim=0)   # [5, B, C]
        preds = torch.stack(fnpred, dim=0)   # [5, B]

        # B, C = fnlog.size(1), fnlog.size(2)
        # final_preds = []
        # for i in range(B):  # é��å�† batch
        #     votes = torch.bincount(preds[:, i], minlength=C)  # [C] å�„ç±»åˆ«ç¥¨æ•°
        #     max_vote = votes.max()
        #     print(votes)
        #     candidates = (votes == max_vote).nonzero(as_tuple=True)[0]  # å¹³ç¥¨ç±»åˆ«
        #     if len(candidates) == 1:
        #         # âœ… æœ‰å”¯ä¸€æœ€é«˜ç¥¨ï¼Œç›´æ�¥é€‰
        #         final_preds.append(candidates.item())
        #     else:
        #         # ğŸ¤� å¹³ç¥¨ï¼Œä½¿ç”¨å¹³å�‡ softmax æ¦‚ç�‡æ�¥æ‰“ç ´å¹³å±€
        #         avg_prob = fnlog[:, i, :].mean(dim=0)  # [C]
        #         chosen = avg_prob[candidates].argmax().item()
        #         final_preds.append(candidates[chosen].item())
        
        # final_preds = torch.tensor(final_preds)  # [B]
        fnlog = fnlog.mean(dim=0)
        print(fnlog)
        idx = int(fnlog[0].argmax().cpu().numpy())
        # idx = int(final_preds.cpu().numpy()[0])
    print(str(gesture_classes[idx]))
    return str(gesture_classes[idx])
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







