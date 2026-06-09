import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""      # è®“æ‰€æœ‰æ¡†æ�¶éƒ½çœ‹ä¸�åˆ° GPU
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"     # å�¯é�¸ï¼šå®‰é�œä¸€é»�

import kagglehub
import os
import json
import joblib
import numpy as np
import pandas as pd
import polars as pl
import random
import math
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
from collections import defaultdict

# Sklearn imports
from sklearn.model_selection import train_test_split, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GRU, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate, GaussianNoise
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam as AdamTF
from tensorflow.keras.optimizers.schedules import CosineDecay
from tensorflow.keras.utils import Sequence, to_categorical, pad_sequences
from tensorflow.keras.callbacks import EarlyStopping

# PyTorch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam as AdamTorch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.amp import autocast

# Scipy imports
from scipy.spatial.transform import Rotation as R
from scipy.signal import firwin

# Other imports
from tqdm.notebook import tqdm
from transformers import BertConfig, BertModel

# Competition metric (optional)
try:
    from cmi_2025_metric_copy_for_import import CompetitionMetric
except ImportError:
    CompetitionMetric = None
    print("CompetitionMetric could not be imported. OOF/CV score will not be calculated.")

def seed_everything(seed=42):
    """è¨­ç½®æ‰€æœ‰éš¨æ©Ÿç¨®å­�ä»¥ç¢ºä¿�çµ�æ�œå�¯é‡�ç�¾"""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# è¨­ç½®éš¨æ©Ÿç¨®å­�å’Œè­¦å‘Š
seed_everything(seed=42)
warnings.filterwarnings("ignore")

# é…�ç½®å¸¸æ•¸
TRAIN = False
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/cmi-d-111")
EXPORT_DIR = Path("./")
BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 5e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 160
PATIENCE = 40

print("â–¶ imports ready Â· tensorflow", tf.__version__)




cmi_detect_behavior_with_sensor_data_path = kagglehub.competition_download('cmi-detect-behavior-with-sensor-data')
hideyukizushi_cmi25_imu_thmtof_tf_bilstm_gru_attentionlb_xx_path = kagglehub.dataset_download('hideyukizushi/cmi25-imu-thmtof-tf-bilstm-gru-attentionlb-xx')
hideyukizushi_20250627_cmi_b_102_b_105_path = kagglehub.dataset_download('hideyukizushi/20250627-cmi-b-102-b-105')
myso1987_cmi3_models_p_path = kagglehub.dataset_download('myso1987/cmi3-models-p')
hideyukizushi_cmi_d_111_path = kagglehub.dataset_download('hideyukizushi/cmi-d-111')
kerta27_cmi_data_gated_gru_path = kagglehub.dataset_download('kerta27/cmi-data-gated-gru')
wasupandceacar_deterministic_path = kagglehub.package_import('wasupandceacar/deterministic')
hideyukizushi_lb_0_78_quaternions_tf_bilstm_gru_attention_path = kagglehub.notebook_output_download('hideyukizushi/lb-0-78-quaternions-tf-bilstm-gru-attention')
wasupandceacar_cmi_metric_path = kagglehub.package_import('wasupandceacar/cmi-metric')
wasupandceacar_cmi_precompute_pytorch_all_1_path = kagglehub.model_download('wasupandceacar/cmi-precompute/PyTorch/all/1')
wasupandceacar_cmi_models_public_pytorch_train_fold_model05_tof16_raw_1_path = kagglehub.model_download('wasupandceacar/cmi-models-public/PyTorch/train_fold_model05_tof16_raw/1')

print('Data source import complete.')


def to_device(*xs, device=torch.device("cpu")):
    out = []
    for x in xs:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        out.append(x.to(device).float() if isinstance(x, torch.Tensor) else x)
    return out if len(out) > 1 else out[0]
# ç”¨æ³•ï¼šimu, thm, tof = to_device(imu), to_device(thm), to_device(tof)



class CMIFeDataset(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.init_feature_names(data_path)
        df = self.generate_features(pd.read_csv(data_path, usecols=set(self.base_cols+self.feature_cols)))
        self.generate_dataset(df)
    def init_feature_names(self, data_path):
        self.imu_engineered_features = ['acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel', 'linear_acc_mag', 'linear_acc_mag_jerk', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance']
        self.tof_mode = self.config.get("tof_mode", "stats")
        self.tof_region_stats = ['mean', 'std', 'min', 'max']
        self.tof_cols = self.generate_tof_feature_names()
        columns = pd.read_csv(data_path, nrows=0).columns.tolist()
        imu_cols_base = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z']
        imu_cols_base.extend([c for c in columns if c.startswith('rot_') and c not in ['rot_angle', 'rot_angle_vel']])
        self.imu_cols = list(dict.fromkeys(imu_cols_base + self.imu_engineered_features))
        self.thm_cols = [c for c in columns if c.startswith('thm_')]
        self.feature_cols = self.imu_cols + self.thm_cols + self.tof_cols
        self.imu_dim = len(self.imu_cols)
        self.thm_dim = len(self.thm_cols)
        self.tof_dim = len(self.tof_cols)
        self.base_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w', 'sequence_id', 'subject', 'sequence_type', 'gesture', 'orientation'] + [c for c in columns if c.startswith('thm_')] + [f"tof_{i}_v{p}" for i in range(1, 6) for p in range(64)]
        self.fold_cols = ['subject', 'sequence_type', 'gesture', 'orientation']
    def generate_tof_feature_names(self):
        features = []
        if self.config.get("tof_raw", False):
            for i in range(1, 6):
                features.extend([f"tof_{i}_v{p}" for p in range(64)])
        for i in range(1, 6):
            if self.tof_mode != 0:
                for stat in self.tof_region_stats:
                    features.append(f'tof_{i}_{stat}')
                if self.tof_mode > 1:
                    for r in range(self.tof_mode):
                        for stat in self.tof_region_stats:
                            features.append(f'tof{self.tof_mode}_{i}_region_{r}_{stat}')
                if self.tof_mode == -1:
                    for mode in [2, 4, 8, 16, 32]:
                        for r in range(mode):
                            for stat in self.tof_region_stats:
                                features.append(f'tof{mode}_{i}_region_{r}_{stat}')
        return features
    def compute_features(self, df):
        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
        df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
        df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
        linear_accel_list = []
        for _, group in df.groupby('sequence_id'):
            acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
            linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))
        df_linear_accel = pd.concat(linear_accel_list)
        df = pd.concat([df, df_linear_accel], axis=1)
        df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
        df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)
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
        if self.tof_mode != 0:
            new_columns = {}
            for i in range(1, 6):
                pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
                tof_data = df[pixel_cols].replace(-1, np.nan)
                new_columns.update({f'tof_{i}_mean': tof_data.mean(axis=1), f'tof_{i}_std': tof_data.std(axis=1), f'tof_{i}_min': tof_data.min(axis=1), f'tof_{i}_max': tof_data.max(axis=1)})
                if self.tof_mode > 1:
                    region_size = 64 // self.tof_mode
                    for r in range(self.tof_mode):
                        region_data = tof_data.iloc[:, r*region_size : (r+1)*region_size]
                        new_columns.update({f'tof{self.tof_mode}_{i}_region_{r}_mean': region_data.mean(axis=1), f'tof{self.tof_mode}_{i}_region_{r}_std': region_data.std(axis=1), f'tof{self.tof_mode}_{i}_region_{r}_min': region_data.min(axis=1), f'tof{self.tof_mode}_{i}_region_{r}_max': region_data.max(axis=1)})
                if self.tof_mode == -1:
                    for mode in [2, 4, 8, 16, 32]:
                        region_size = 64 // mode
                        for r in range(mode):
                            region_data = tof_data.iloc[:, r*region_size : (r+1)*region_size]
                            new_columns.update({f'tof{mode}_{i}_region_{r}_mean': region_data.mean(axis=1), f'tof{mode}_{i}_region_{r}_std': region_data.std(axis=1), f'tof{mode}_{i}_region_{r}_min': region_data.min(axis=1), f'tof{mode}_{i}_region_{r}_max': region_data.max(axis=1)})
            df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        return df
    def generate_features(self, df):
        self.le = LabelEncoder()
        df['gesture_int'] = self.le.fit_transform(df['gesture'])
        self.class_num = len(self.le.classes_)
        if all(c in df.columns for c in self.imu_engineered_features) and all(c in df.columns for c in self.tof_cols):
            print("Have precomputed, skip compute.")
        else:
            print("Not precomputed, do compute.")
            df = self.compute_features(df)
        if self.config.get("save_precompute", False):
            df.to_csv(self.config.get("save_filename", "train.csv"))
        return df
    def scale(self, data_unscaled):
        scaler_function = self.config.get("scaler_function", StandardScaler())
        scaler = scaler_function.fit(np.concatenate(data_unscaled, axis=0))
        return [scaler.transform(x) for x in data_unscaled], scaler
    def pad(self, data_scaled, cols):
        pad_data = np.zeros((len(data_scaled), self.pad_len, len(cols)), dtype='float32')
        for i, seq in enumerate(data_scaled):
            seq_len = min(len(seq), self.pad_len)
            pad_data[i, :seq_len] = seq[:seq_len]
        return pad_data
    def get_nan_value(self, data, ratio):
        max_value = data.max().max()
        nan_value = -max_value * ratio
        return nan_value
    def generate_dataset(self, df):
        seq_gp = df.groupby('sequence_id')
        imu_unscaled, thm_unscaled, tof_unscaled = [], [], []
        classes, lens = [], []
        self.imu_nan_value = self.get_nan_value(df[self.imu_cols], self.config["nan_ratio"]["imu"])
        self.thm_nan_value = self.get_nan_value(df[self.thm_cols], self.config["nan_ratio"]["thm"])
        self.tof_nan_value = self.get_nan_value(df[self.tof_cols], self.config["nan_ratio"]["tof"])
        self.fold_feats = defaultdict(list)
        for seq_id, seq_df in seq_gp:
            imu_data = seq_df[self.imu_cols]
            if self.config["fbfill"]["imu"]:
                imu_data = imu_data.ffill().bfill()
            imu_unscaled.append(imu_data.fillna(self.imu_nan_value).values.astype('float32'))
            thm_data = seq_df[self.thm_cols]
            if self.config["fbfill"]["thm"]:
                thm_data = thm_data.ffill().bfill()
            thm_unscaled.append(thm_data.fillna(self.thm_nan_value).values.astype('float32'))
            tof_data = seq_df[self.tof_cols]
            if self.config["fbfill"]["tof"]:
                tof_data = tof_data.ffill().bfill()
            tof_unscaled.append(tof_data.fillna(self.tof_nan_value).values.astype('float32'))
            classes.append(seq_df['gesture_int'].iloc[0])
            lens.append(len(imu_data))
            for col in self.fold_cols:
                self.fold_feats[col].append(seq_df[col].iloc[0])
        self.dataset_indices = classes
        self.pad_len = int(np.percentile(lens, self.config.get("percent", 95)))
        if self.config.get("one_scale", True):
            x_unscaled = [np.concatenate([imu, thm, tof], axis=1) for imu, thm, tof in zip(imu_unscaled, thm_unscaled, tof_unscaled)]
            x_scaled, self.x_scaler = self.scale(x_unscaled)
            x = self.pad(x_scaled, self.imu_cols+self.thm_cols+self.tof_cols)
            self.imu = x[..., :self.imu_dim]
            self.thm = x[..., self.imu_dim:self.imu_dim+self.thm_dim]
            self.tof = x[..., self.imu_dim+self.thm_dim:self.imu_dim+self.thm_dim+self.tof_dim]
        else:
            imu_scaled, self.imu_scaler = self.scale(imu_unscaled)
            thm_scaled, self.thm_scaler = self.scale(thm_unscaled)
            tof_scaled, self.tof_scaler = self.scale(tof_unscaled)
            self.imu = self.pad(imu_scaled, self.imu_cols)
            self.thm = self.pad(thm_scaled, self.thm_cols)
            self.tof = self.pad(tof_scaled, self.tof_cols)
        self.precompute_scaled_nan_values()
        self.class_ = F.one_hot(torch.from_numpy(np.array(classes)).long(), num_classes=len(self.le.classes_)).float().numpy()
        self.class_weight = torch.FloatTensor(compute_class_weight('balanced', classes=np.arange(len(self.le.classes_)), y=classes))
    def precompute_scaled_nan_values(self):
        dummy_df = pd.DataFrame(np.array([[self.imu_nan_value]*len(self.imu_cols) + [self.thm_nan_value]*len(self.thm_cols) + [self.tof_nan_value]*len(self.tof_cols)]), columns=self.imu_cols + self.thm_cols + self.tof_cols)
        if self.config.get("one_scale", True):
            scaled = self.x_scaler.transform(dummy_df)
            self.imu_scaled_nan = scaled[0, :self.imu_dim].mean()
            self.thm_scaled_nan = scaled[0, self.imu_dim:self.imu_dim+self.thm_dim].mean()
            self.tof_scaled_nan = scaled[0, self.imu_dim+self.thm_dim:self.imu_dim+self.thm_dim+self.tof_dim].mean()
        else:
            self.imu_scaled_nan = self.imu_scaler.transform(dummy_df[self.imu_cols])[0].mean()
            self.thm_scaled_nan = self.thm_scaler.transform(dummy_df[self.thm_cols])[0].mean()
            self.tof_scaled_nan = self.tof_scaler.transform(dummy_df[self.tof_cols])[0].mean()
    def get_scaled_nan_tensors(self, imu, thm, tof):
        return torch.full(imu.shape, self.imu_scaled_nan, device=imu.device), torch.full(thm.shape, self.thm_scaled_nan, device=thm.device), torch.full(tof.shape, self.tof_scaled_nan, device=tof.device)
    def inference_process(self, sequence):
        df_seq = sequence.to_pandas().copy()
        if not all(c in df_seq.columns for c in self.imu_engineered_features):
            df_seq['acc_mag'] = np.sqrt(df_seq['acc_x']**2 + df_seq['acc_y']**2 + df_seq['acc_z']**2)
            df_seq['rot_angle'] = 2 * np.arccos(df_seq['rot_w'].clip(-1, 1))
            df_seq['acc_mag_jerk'] = df_seq['acc_mag'].diff().fillna(0)
            df_seq['rot_angle_vel'] = df_seq['rot_angle'].diff().fillna(0)
            if all(col in df_seq.columns for col in ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']):
                linear_accel = remove_gravity_from_acc(df_seq[['acc_x', 'acc_y', 'acc_z']], df_seq[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
                df_seq[['linear_acc_x', 'linear_acc_y', 'linear_acc_z']] = linear_accel
            else:
                df_seq['linear_acc_x'] = df_seq.get('acc_x', 0)
                df_seq['linear_acc_y'] = df_seq.get('acc_y', 0)
                df_seq['linear_acc_z'] = df_seq.get('acc_z', 0)
            df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 + df_seq['linear_acc_y']**2 + df_seq['linear_acc_z']**2)
            df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)
            if all(col in df_seq.columns for col in ['rot_x', 'rot_y', 'rot_z', 'rot_w']):
                angular_vel = calculate_angular_velocity_from_quat(df_seq[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
                df_seq[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']] = angular_vel
            else:
                df_seq[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']] = 0
            if all(col in df_seq.columns for col in ['rot_x', 'rot_y', 'rot_z', 'rot_w']):
                df_seq['angular_distance'] = calculate_angular_distance(df_seq[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
            else:
                df_seq['angular_distance'] = 0
        if self.tof_mode != 0:
            new_columns = {}
            for i in range(1, 6):
                pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
                tof_data = df_seq[pixel_cols].replace(-1, np.nan)
                new_columns.update({f'tof_{i}_mean': tof_data.mean(axis=1), f'tof_{i}_std': tof_data.std(axis=1), f'tof_{i}_min': tof_data.min(axis=1), f'tof_{i}_max': tof_data.max(axis=1)})
                if self.tof_mode > 1:
                    region_size = 64 // self.tof_mode
                    for r in range(self.tof_mode):
                        region_data = tof_data.iloc[:, r*region_size : (r+1)*region_size]
                        new_columns.update({f'tof{self.tof_mode}_{i}_region_{r}_mean': region_data.mean(axis=1), f'tof{self.tof_mode}_{i}_region_{r}_std': region_data.std(axis=1), f'tof{self.tof_mode}_{i}_region_{r}_min': region_data.min(axis=1), f'tof{self.tof_mode}_{i}_region_{r}_max': region_data.max(axis=1)})
                if self.tof_mode == -1:
                    for mode in [2, 4, 8, 16, 32]:
                        region_size = 64 // mode
                        for r in range(mode):
                            region_data = tof_data.iloc[:, r*region_size : (r+1)*region_size]
                            new_columns.update({f'tof{mode}_{i}_region_{r}_mean': region_data.mean(axis=1), f'tof{mode}_{i}_region_{r}_std': region_data.std(axis=1), f'tof{mode}_{i}_region_{r}_min': region_data.min(axis=1), f'tof{mode}_{i}_region_{r}_max': region_data.max(axis=1)})
            df_seq = pd.concat([df_seq, pd.DataFrame(new_columns)], axis=1)
        imu_unscaled = df_seq[self.imu_cols]
        if self.config["fbfill"]["imu"]:
            imu_unscaled = imu_unscaled.ffill().bfill()
        imu_unscaled = imu_unscaled.fillna(self.imu_nan_value).values.astype('float32')
        thm_unscaled = df_seq[self.thm_cols]
        if self.config["fbfill"]["thm"]:
            thm_unscaled = thm_unscaled.ffill().bfill()
        thm_unscaled = thm_unscaled.fillna(self.thm_nan_value).values.astype('float32')
        tof_unscaled = df_seq[self.tof_cols]
        if self.config["fbfill"]["tof"]:
            tof_unscaled = tof_unscaled.ffill().bfill()
        tof_unscaled = tof_unscaled.fillna(self.tof_nan_value).values.astype('float32')
        if self.config.get("one_scale", True):
            x_unscaled = np.concatenate([imu_unscaled, thm_unscaled, tof_unscaled], axis=1)
            x_scaled = self.x_scaler.transform(x_unscaled)
            imu_scaled = x_scaled[..., :self.imu_dim]
            thm_scaled = x_scaled[..., self.imu_dim:self.imu_dim+self.thm_dim]
            tof_scaled = x_scaled[..., self.imu_dim+self.thm_dim:self.imu_dim+self.thm_dim+self.tof_dim]
        else:
            imu_scaled = self.imu_scaler.transform(imu_unscaled)
            thm_scaled = self.thm_scaler.transform(thm_unscaled)
            tof_scaled = self.tof_scaler.transform(tof_unscaled)
        combined = np.concatenate([imu_scaled, thm_scaled, tof_scaled], axis=1)
        padded = np.zeros((self.pad_len, combined.shape[1]), dtype='float32')
        seq_len = min(combined.shape[0], self.pad_len)
        padded[:seq_len] = combined[:seq_len]
        imu = padded[..., :self.imu_dim]
        thm = padded[..., self.imu_dim:self.imu_dim+self.thm_dim]
        tof = padded[..., self.imu_dim+self.thm_dim:self.imu_dim+self.thm_dim+self.tof_dim]
        return torch.from_numpy(imu).float().unsqueeze(0), torch.from_numpy(thm).float().unsqueeze(0), torch.from_numpy(tof).float().unsqueeze(0)
    def __getitem__(self, idx):
        return self.imu[idx], self.thm[idx], self.tof[idx], self.class_[idx]
    def __len__(self):
        return len(self.class_)


from sklearn.model_selection import StratifiedKFold
class CMIFoldDataset:
    def __init__(self, data_path, config, full_dataset_function, n_folds=5, random_seed=0):
        self.full_dataset = full_dataset_function(data_path=data_path, config=config)
        self.imu_dim = self.full_dataset.imu_dim
        self.thm_dim = self.full_dataset.thm_dim
        self.tof_dim = self.full_dataset.tof_dim
        self.le = self.full_dataset.le
        self.class_names = self.full_dataset.le.classes_
        self.class_weight = self.full_dataset.class_weight
        self.n_folds = n_folds
        self.skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        self.folds = list(self.skf.split(np.arange(len(self.full_dataset)), np.array(self.full_dataset.dataset_indices)))
    def get_fold_datasets(self, fold_idx):
        if self.folds is None or fold_idx >= self.n_folds:
            return None, None
        fold_train_idx, fold_valid_idx = self.folds[fold_idx]
        return Subset(self.full_dataset, fold_train_idx), Subset(self.full_dataset, fold_valid_idx)
    def print_fold_stats(self):
        def get_label_counts(subset):
            counts = {name: 0 for name in self.class_names}
            if subset is None:
                return counts
            for idx in subset.indices:
                label_idx = self.full_dataset.dataset_indices[idx]
                counts[self.class_names[label_idx]] += 1
            return counts
        print("\nCross-validation fold statistics:")
        for fold_idx in range(self.n_folds):
            train_fold, valid_fold = self.get_fold_datasets(fold_idx)
            train_counts = get_label_counts(train_fold)
            valid_counts = get_label_counts(valid_fold)
            print(f"\nFold {fold_idx + 1}:")
            print(f"{'Category':<50} {'Training Set':<10} {'Validation Set':<10}")
            for name in self.class_names:
                print(f"{name:<50} {train_counts[name]:<10} {valid_counts[name]:<10}")
CUDA0 = "cuda:0"
seed = 0
batch_size = 64
num_workers = 4
n_folds = 5
universe_csv_path = Path("/kaggle/input/cmi-precompute/pytorch/all/1/tof-1_raw.csv")
deterministic = kagglehub.package_import('wasupandceacar/deterministic').deterministic
deterministic.init_all(seed)


def init_dataset():
    dataset_config = {"percent": 95, "scaler_function": StandardScaler(), "nan_ratio": {"imu": 0, "thm": 0, "tof": 0}, "fbfill": {"imu": True, "thm": True, "tof": True}, "one_scale": True, "tof_raw": True, "tof_mode": 16, "save_precompute": False}
    dataset = CMIFoldDataset(universe_csv_path, dataset_config, n_folds=n_folds, random_seed=seed, full_dataset_function=CMIFeDataset)
    dataset.print_fold_stats()
    return dataset
def get_fold_dataset(dataset, fold):
    _, valid_dataset = dataset.get_fold_datasets(fold)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return valid_loader
dataset = init_dataset()



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

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
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
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            pass
    return angular_vel


def pad_sequences_torch3(sequences, maxlen, padding='post', truncating='post', value=0.0):
    result = []
    for seq in sequences:
        if len(seq) >= maxlen: seq = seq[:maxlen] if truncating == 'post' else seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            pad_array = np.full((pad_len, seq.shape[1]), value)
            seq = np.concatenate([seq, pad_array]) if padding == 'post' else np.concatenate([pad_array, seq])
        result.append(seq)
    return np.array(result, dtype=np.float32)
def remove_gravity_from_acc3(acc_data, rot_data):
    acc_values = acc_data[['acc_x', 'acc_y', 'acc_z']].values
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])
    for i in range(len(acc_values)):
        if np.all(np.isnan(quat_values[i])):
            linear_accel[i, :] = acc_values[i, :]
            continue
        try:
            rotation = R.from_quat(quat_values[i])
            gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
            linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
        except (ValueError, IndexError):
            linear_accel[i, :] = acc_values[i, :]
    return linear_accel
def calculate_angular_velocity_from_quat3(rot_data, time_delta=1/200):
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_vel = np.zeros((len(quat_values), 3))
    for i in range(len(quat_values) - 1):
        q_t, q_t_plus_dt = quat_values[i], quat_values[i+1]
        if np.all(np.isnan(q_t)) or np.all(np.isnan(q_t_plus_dt)): continue
        try:
            rot_t = R.from_quat(q_t)
            rot_t_plus_dt = R.from_quat(q_t_plus_dt)
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except (ValueError, IndexError): pass
    return angular_vel
def calculate_angular_distance3(rot_data):
    quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_dist = np.zeros(len(quat_values))
    for i in range(len(quat_values) - 1):
        q1, q2 = quat_values[i], quat_values[i+1]
        if np.all(np.isnan(q1)) or np.all(np.isnan(q2)): continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angular_dist[i] = np.linalg.norm(relative_rotation.as_rotvec())
        except (ValueError, IndexError): pass
    return angular_dist








import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # é—œæ�‰ GPU å�¯è¦‹æ€§ï¼ˆå°� TF/PyTorch éƒ½ç”Ÿæ•ˆï¼‰



# model1===============================
def time_sum(x):
    return K.sum(x, axis=1)

def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

def se_block(x, reduction=8):
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    shortcut = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False,
                   kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False,
                          kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x

def attention_layer(inputs):
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context

# Load necessary preprocessing components
final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")

# Register custom Keras components
custom_objs = {
    'time_sum': time_sum,
    'squeeze_last_axis': squeeze_last_axis,
    'expand_last_axis': expand_last_axis,
    'se_block': se_block,
    'residual_se_cnn_block': residual_se_cnn_block,
    'attention_layer': attention_layer,
}

# Load Keras ensemble models
models1 = []
print(f"Loading models for ensemble inference...")

for fold in range(10):
    MODEL_DIR = "/kaggle/input/cmi-d-111"
    model_path = f"{MODEL_DIR}/D-111_{fold}.h5"
    print(">>> LoadModel >>>", model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models1.append(model)

print("-" * 50)

for fold in range(10):
    MODEL_DIR = "/kaggle/input/cmi-d-111"
    model_path = f"{MODEL_DIR}/v0629_{fold}.h5"
    print(">>> LoadModel >>>", model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models1.append(model)

print("-" * 50)
print(f"[INFO] NumUseModels: {len(models1)}")



# model2 ========================
# --- å¼·åˆ¶å�ªç”¨ CPUï¼ˆæ”¾åœ¨æœ€ä¸Šæ–¹ï¼Œimport torch å‰�ä¹Ÿå�¯ï¼‰ ---
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # è®“ PyTorch/TF éƒ½çœ‹ä¸�åˆ° GPU

import torch, torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import numpy as np

DEVICE = torch.device("cpu")


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = F.adaptive_avg_pool1d(x, 1).squeeze(-1)
        se = F.relu(self.fc1(se), inplace=True)
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)
        return x * se


class ResNetSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, wd=1e-4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return self.relu(out)


class CMIModel(nn.Module):
    def __init__(self, imu_dim, thm_dim, tof_dim, n_classes, **kwargs):
        super().__init__()

        self.imu_branch = nn.Sequential(
            self.residual_se_cnn_block(imu_dim, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_se_cnn_block(kwargs["imu1_channels"], kwargs["feat_dim"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )

        self.thm_branch = nn.Sequential(
            nn.Conv1d(thm_dim, kwargs["thm1_channels"], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(kwargs["thm1_channels"]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Dropout(kwargs["thm1_dropout"]),
            nn.Conv1d(kwargs["thm1_channels"], kwargs["feat_dim"], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(kwargs["feat_dim"]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Dropout(kwargs["thm2_dropout"])
        )

        self.tof_branch = nn.Sequential(
            nn.Conv1d(tof_dim, kwargs["tof1_channels"], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(kwargs["tof1_channels"]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Dropout(kwargs["tof1_dropout"]),
            nn.Conv1d(kwargs["tof1_channels"], kwargs["feat_dim"], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(kwargs["feat_dim"]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True),
            nn.Dropout(kwargs["tof2_dropout"])
        )

        self.cls_token = nn.Parameter(torch.zeros(1, 1, kwargs["feat_dim"]))

        self.bert = BertModel(
            BertConfig(
                hidden_size=kwargs["feat_dim"],
                num_hidden_layers=kwargs["bert_layers"],
                num_attention_heads=kwargs["bert_heads"],
                intermediate_size=kwargs["feat_dim"] * 4
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(kwargs["feat_dim"], kwargs["cls1_channels"], bias=False),
            nn.BatchNorm1d(kwargs["cls1_channels"]),
            nn.ReLU(inplace=True),
            nn.Dropout(kwargs["cls1_dropout"]),
            nn.Linear(kwargs["cls1_channels"], kwargs["cls2_channels"], bias=False),
            nn.BatchNorm1d(kwargs["cls2_channels"]),
            nn.ReLU(inplace=True),
            nn.Dropout(kwargs["cls2_dropout"]),
            nn.Linear(kwargs["cls2_channels"], n_classes)
        )

    def residual_se_cnn_block(self, in_channels, out_channels, num_layers, pool_size=2, drop=0.3, wd=1e-4):
        return nn.Sequential(
            *[ResNetSEBlock(in_channels=in_channels, out_channels=in_channels) for _ in range(num_layers)],
            ResNetSEBlock(in_channels, out_channels, wd=wd),
            nn.MaxPool1d(pool_size),
            nn.Dropout(drop)
        )

    def forward(self, imu, thm, tof):
        imu_feat = self.imu_branch(imu.permute(0, 2, 1))
        thm_feat = self.thm_branch(thm.permute(0, 2, 1))
        tof_feat = self.tof_branch(tof.permute(0, 2, 1))
        bert_input = torch.cat([imu_feat, thm_feat, tof_feat], dim=-1).permute(0, 2, 1)
        cls_token = self.cls_token.expand(bert_input.size(0), -1, -1)
        bert_input = torch.cat([cls_token, bert_input], dim=1)
        outputs = self.bert(inputs_embeds=bert_input)
        pred_cls = outputs.last_hidden_state[:, 0, :]
        return self.classifier(pred_cls)


# ========== Load Model2 Ensemble ==========

model_function = CMIModel
model_args = {
    "feat_dim": 500,
    "imu1_channels": 219,
    "imu1_dropout": 0.2946731587132302,
    "imu2_dropout": 0.2697745571929592,
    "imu1_weight_decay": 0.0014824054650601245,
    "imu2_weight_decay": 0.002742543773142381,
    "imu1_layers": 0,
    "imu2_layers": 0,
    "thm1_channels": 82,
    "thm1_dropout": 0.2641274454844602,
    "thm2_dropout": 0.302896343020985,
    "tof1_channels": 82,
    "tof1_dropout": 0.2641274454844602,
    "tof2_dropout": 0.3028963430209852,
    "bert_layers": 8,
    "bert_heads": 10,
    "cls1_channels": 937,
    "cls2_channels": 303,
    "cls1_dropout": 0.2281834512100508,
    "cls2_dropout": 0.22502521933558461
}
model_args.update({
    "imu_dim": dataset.full_dataset.imu_dim,
    "thm_dim": dataset.full_dataset.thm_dim,
    "tof_dim": dataset.full_dataset.tof_dim,
    "n_classes": dataset.full_dataset.class_num
})

model_dir = Path("/kaggle/input/cmi-models-public/pytorch/train_fold_model05_tof16_raw/1")
model_dicts = [
    {
        "model_function": model_function,
        "model_args": model_args,
        "model_path": model_dir / f"fold{fold}/best_ema.pt"
    }
    for fold in range(n_folds)
]

models2 = []
for md in model_dicts:
    m = md["model_function"](**md["model_args"]).to(DEVICE)
    # ä¸€å¾‹ map_location=CPUï¼Œé�¿å…�æŠŠæ¬Šé‡�è¼‰åˆ° GPU
    sd = torch.load(md["model_path"], map_location=DEVICE)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=False)
    m.eval()
    models2.append(m)

# ========== Metric & Utility ==========

metric_package = kagglehub.package_import('wasupandceacar/cmi-metric')
metric = metric_package.Metric()
imu_only_metric = metric_package.Metric()

def to_cuda(*tensors):
    return [tensor.to(CUDA0) for tensor in tensors]

def avg_predict(models, imu, thm, tof):
    outputs = []
    with autocast(device_type='cuda'):
        for model in models:
            logits = model(imu, thm, tof)
            outputs.append(logits)
    return torch.mean(torch.stack(outputs), dim=0)



# model3
TRAIN = False
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
YOUR_MODELS_DIR = Path("/kaggle/input/cmi-data-gated-gru")
PUBLIC_TF_MODEL_DIR = Path("/kaggle/input/lb-0-78-quaternions-tf-bilstm-gru-attention")
PUBLIC_PT_MODEL_DIR = Path("/kaggle/input/cmi3-models-p")
EXPORT_DIR = Path("./")
BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 4e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 360
PATIENCE = 50
N_SPLITS = 10
MASKING_PROB = 0.25
GATE_LOSS_WEIGHT = 0.2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mean_pt = torch.tensor([0, 0, 0, 0, 0, 0, 9.0319e-03, 1.0849e+00, -2.6186e-03, 3.7651e-03, -5.3660e-03, -2.8177e-03, 1.3318e-03, -1.5876e-04, 6.3495e-01, 6.2877e-01, 6.0607e-01, 6.2142e-01, 6.3808e-01, 6.5420e-01, 7.4102e-03, -3.4159e-03, -7.5237e-03, -2.6034e-02, 2.9704e-02, -3.1546e-02, -2.0610e-03, -4.6986e-03, -4.7216e-03, -2.6281e-02, 1.5799e-02, 1.0016e-02], dtype=torch.float32).view(1, -1, 1).to(device)
std_pt = torch.tensor([1, 1, 1, 1, 1, 1, 0.2067, 0.8583, 0.3162, 0.2668, 0.2917, 0.2341, 0.3023, 0.3281, 1.0264, 0.8838, 0.8686, 1.0973, 1.0267, 0.9018, 0.4658, 0.2009, 0.2057, 1.2240, 0.9535, 0.6655, 0.2941, 0.3421, 0.8156, 0.6565, 1.1034, 1.5577], dtype=torch.float32).view(1, -1, 1).to(device) + 1e-8
class ImuFeatureExtractor(nn.Module):
    def __init__(self, fs=100., add_quaternion=False):
        super().__init__()
        self.fs = fs
        self.add_quaternion = add_quaternion
        k = 15
        self.lpf = nn.Conv1d(6, 6, kernel_size=k, padding=k//2, groups=6, bias=False)
        nn.init.kaiming_uniform_(self.lpf.weight, a=math.sqrt(5))
        self.lpf_acc = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
        self.lpf_gyro = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
    def forward(self, imu):
        acc = imu[:, 0:3, :]
        gyro = imu[:, 3:6, :]
        acc_mag = torch.norm(acc, dim=1, keepdim=True)
        gyro_mag = torch.norm(gyro, dim=1, keepdim=True)
        jerk = F.pad(acc[:, :, 1:] - acc[:, :, :-1], (1,0))
        gyro_delta = F.pad(gyro[:, :, 1:] - gyro[:, :, :-1], (1,0))
        acc_pow = acc ** 2
        gyro_pow = gyro ** 2
        acc_lpf = self.lpf_acc(acc)
        acc_hpf = acc - acc_lpf
        gyro_lpf = self.lpf_gyro(gyro)
        gyro_hpf = gyro - gyro_lpf
        features = [acc, gyro, acc_mag, gyro_mag, jerk, gyro_delta, acc_pow, gyro_pow, acc_lpf, acc_hpf, gyro_lpf, gyro_hpf]
        return torch.cat(features, dim=1)
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True), nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid())
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)
class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, bias=False), nn.BatchNorm1d(out_channels))
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        return self.dropout(self.pool(F.relu(out)))
class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
    def forward(self, x):
        scores = torch.tanh(self.attention(x))
        weights = F.softmax(scores.squeeze(-1), dim=1)
        return torch.sum(x * weights.unsqueeze(-1), dim=1)

class PublicTwoBranchModel(nn.Module):
    def __init__(self, pad_len, imu_dim_raw, tof_dim, n_classes, dropouts=[0.3, 0.3, 0.3, 0.3, 0.4, 0.5, 0.3], feature_engineering=True, **kwargs):
        super().__init__()
        self.feature_engineering = feature_engineering
        imu_dim = 32 if feature_engineering else imu_dim_raw
        self.imu_fe = ImuFeatureExtractor(**kwargs) if feature_engineering else nn.Identity()
        self.fir_nchan = 7
        numtaps = 33
        fir_kernel = torch.tensor(firwin(numtaps, cutoff=1.0, fs=10.0, pass_zero=False), dtype=torch.float32).view(1, 1, -1).repeat(self.fir_nchan, 1, 1)
        self.register_buffer("fir_kernel", fir_kernel)
        self.imu_block1 = ResidualSECNNBlock(imu_dim, 64, 3, dropout=dropouts[0])
        self.imu_block2 = ResidualSECNNBlock(64, 128, 5, dropout=dropouts[1])
        self.tof_conv1 = nn.Conv1d(tof_dim, 64, 3, padding=1, bias=False)
        self.tof_bn1, self.tof_pool1, self.tof_drop1 = nn.BatchNorm1d(64), nn.MaxPool1d(2), nn.Dropout(dropouts[2])
        self.tof_conv2 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        self.tof_bn2, self.tof_pool2, self.tof_drop2 = nn.BatchNorm1d(128), nn.MaxPool1d(2), nn.Dropout(dropouts[3])
        self.bilstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.lstm_dropout = nn.Dropout(dropouts[4])
        self.attention = AttentionLayer(256)
        self.dense1, self.bn_dense1, self.drop1 = nn.Linear(256, 256, bias=False), nn.BatchNorm1d(256), nn.Dropout(dropouts[5])
        self.dense2, self.bn_dense2, self.drop2 = nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128), nn.Dropout(dropouts[6])
        self.classifier = nn.Linear(128, n_classes)
    def forward(self, x):
        imu_raw = x[:, :, :self.fir_nchan].transpose(1, 2)
        tof = x[:, :, self.fir_nchan:].transpose(1, 2)
        imu_fe = self.imu_fe(imu_raw)
        filtered = F.conv1d(imu_fe[:, :self.fir_nchan, :], self.fir_kernel, padding=self.fir_kernel.shape[-1] // 2, groups=self.fir_nchan)
        imu = (torch.cat([filtered, imu_fe[:, self.fir_nchan:, :]], dim=1) - mean_pt) / std_pt
        x1 = self.imu_block1(imu); x1 = self.imu_block2(x1)
        x2 = self.tof_drop1(self.tof_pool1(F.relu(self.tof_bn1(self.tof_conv1(tof)))))
        x2 = self.tof_drop2(self.tof_pool2(F.relu(self.tof_bn2(self.tof_conv2(x2)))))
        merged = torch.cat([x1, x2], dim=1).transpose(1, 2)
        lstm_out, _ = self.bilstm(merged); lstm_out = self.lstm_dropout(lstm_out)
        attended = self.attention(lstm_out)
        x = self.drop1(F.relu(self.bn_dense1(self.dense1(attended))))
        x = self.drop2(F.relu(self.bn_dense2(self.dense2(x))))
        return self.classifier(x)
def time_sum(x): return K.sum(x, axis=1)
def squeeze_last_axis(x): return tf.squeeze(x, axis=-1)
def expand_last_axis(x): return tf.expand_dims(x, axis=-1)

def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    shortcut = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False, kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x
def attention_layer(inputs):
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context


print("â–¶ Inference mode start â€“ loading trained models and artifacts...")
print(" Loading model group A (Custom 5-Fold Gated GRU model)...")
final_feature_cols_A = np.load(YOUR_MODELS_DIR / "final_feature_cols.npy", allow_pickle=True).tolist()
pad_len_A = int(np.load(YOUR_MODELS_DIR / "sequence_maxlen.npy"))
scaler_A = joblib.load(YOUR_MODELS_DIR / "scaler.pkl")
gesture_classes = np.load(YOUR_MODELS_DIR / "gesture_classes.npy", allow_pickle=True)
custom_objs_A = {'time_sum': time_sum, 'squeeze_last_axis': squeeze_last_axis, 'expand_last_axis': expand_last_axis, 'se_block': se_block, 'residual_se_cnn_block': residual_se_cnn_block, 'attention_layer': attention_layer}
models_A = [load_model(YOUR_MODELS_DIR / f"final_model_fold_{f}.h5", compile=False, custom_objects=custom_objs_A) for f in range(N_SPLITS)]
print(f" > Loaded {len(models_A)} models successfully.")
print("\n Loading model group B (Public TF/Keras model)...")
final_feature_cols_B = np.load(PUBLIC_TF_MODEL_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len_B = int(np.load(PUBLIC_TF_MODEL_DIR / "sequence_maxlen.npy"))
scaler_B = joblib.load(PUBLIC_TF_MODEL_DIR / "scaler.pkl")
custom_objs_B = custom_objs_A
model_B = load_model(PUBLIC_TF_MODEL_DIR / "gesture_two_branch_mixup.h5", compile=False, custom_objects=custom_objs_B)
print(" > Loaded 1 model successfully.")
print("\n Loading model group C (Public PyTorch model)...")
final_feature_cols_C = np.load(PUBLIC_PT_MODEL_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len_C = int(np.load(PUBLIC_PT_MODEL_DIR / "sequence_maxlen.npy"))
scaler_C = joblib.load(PUBLIC_PT_MODEL_DIR / "scaler.pkl")
pt_models = []
for f in range(5):
    checkpoint = torch.load(PUBLIC_PT_MODEL_DIR / f"gesture_two_branch_fold{f}.pth", map_location=device)
    cfg = {'pad_len': checkpoint['pad_len'], 'imu_dim_raw': checkpoint['imu_dim'], 'tof_dim': checkpoint['tof_dim'], 'n_classes': checkpoint['n_classes']}
    m = PublicTwoBranchModel(**cfg).to(device)
    m.load_state_dict(checkpoint['model_state_dict'])
    m.eval()
    pt_models.append(m)
print(f" > Loaded {len(pt_models)} models successfully.")


def predict1(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_seq = sequence.to_pandas()
    linear_accel = remove_gravity_from_acc(df_seq, df_seq)
    df_seq['linear_acc_x'], df_seq['linear_acc_y'], df_seq['linear_acc_z'] = linear_accel[:, 0], linear_accel[:, 1], linear_accel[:, 2]
    df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 + df_seq['linear_acc_y']**2 + df_seq['linear_acc_z']**2)
    df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)
    angular_vel = calculate_angular_velocity_from_quat(df_seq)
    df_seq['angular_vel_x'], df_seq['angular_vel_y'], df_seq['angular_vel_z'] = angular_vel[:, 0], angular_vel[:, 1], angular_vel[:, 2]
    df_seq['angular_distance'] = calculate_angular_distance(df_seq)

    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]; tof_data = df_seq[pixel_cols].replace(-1, np.nan)
        df_seq[f'tof_{i}_mean'], df_seq[f'tof_{i}_std'], df_seq[f'tof_{i}_min'], df_seq[f'tof_{i}_max'] = tof_data.mean(axis=1), tof_data.std(axis=1), tof_data.min(axis=1), tof_data.max(axis=1)

    mat_unscaled = df_seq[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')
    mat_scaled = scaler.transform(mat_unscaled)
    pad_input = pad_sequences([mat_scaled], maxlen=pad_len, padding='post', truncating='post', dtype='float32')
    all_preds = [model.predict(pad_input, verbose=0)[0] for model in models1]
    avg_pred = np.mean(all_preds, axis=0)
    return avg_pred

def predict2(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    imu, thm, tof = dataset.full_dataset.inference_process(sequence)
    with torch.no_grad():
        imu, thm, tof = to_cuda(imu, thm, tof)
        logits = avg_predict(models2, imu, thm, tof)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
    return probabilities

import numpy as np
import torch
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict3(sequence: pl.DataFrame, demographics: pl.DataFrame):
    # ---- å¼·åˆ¶å�ªç”¨ CPU ----
    device = torch.device("cpu")

    df_seq_orig = sequence.to_pandas() if hasattr(sequence, "to_pandas") else sequence.copy()

    # ================= A branch =================
    dfA = df_seq_orig.copy()
    accA  = dfA[['acc_x','acc_y','acc_z']].to_numpy(np.float32)
    quatA = dfA[['rot_x','rot_y','rot_z','rot_w']].to_numpy(np.float32)

    linear_accel_A = remove_gravity_from_acc3(accA, quatA)
    dfA['linear_acc_x'], dfA['linear_acc_y'], dfA['linear_acc_z'] = linear_accel_A[:,0], linear_accel_A[:,1], linear_accel_A[:,2]
    dfA['linear_acc_mag'] = np.linalg.norm(linear_accel_A, axis=1)
    dfA['linear_acc_mag_jerk'] = dfA['linear_acc_mag'].diff().fillna(0)

    angular_vel_A = calculate_angular_velocity_from_quat3(quatA)
    dfA['angular_vel_x'], dfA['angular_vel_y'], dfA['angular_vel_z'] = angular_vel_A[:,0], angular_vel_A[:,1], angular_vel_A[:,2]
    dfA['angular_distance'] = calculate_angular_distance3(quatA)

    for col in ['rot_x','rot_y','rot_z','rot_w']:
        dfA[f'{col}_diff'] = dfA[col].diff().fillna(0)
    for col in ['linear_acc_mag','linear_acc_mag_jerk','angular_distance']:
        dfA[f'{col}_skew'] = dfA[col].skew()
        dfA[f'{col}_kurt'] = dfA[col].kurtosis()

    for i in range(1,6):
        pix = [f"tof_{i}_v{p}" for p in range(64) if f"tof_{i}_v{p}" in dfA.columns]
        if pix:
            t = dfA[pix].replace(-1, np.nan)
            dfA[f'tof_{i}_mean'], dfA[f'tof_{i}_std'], dfA[f'tof_{i}_min'], dfA[f'tof_{i}_max'] = \
                t.mean(axis=1), t.std(axis=1), t.min(axis=1), t.max(axis=1)

    tof_mean_cols=[f'tof_{i}_mean' for i in range(1,6) if f'tof_{i}_mean' in dfA.columns]
    if tof_mean_cols:
        dfA['tof_std_across_sensors']   = dfA[tof_mean_cols].std(axis=1)
        dfA['tof_range_across_sensors'] = dfA[tof_mean_cols].max(axis=1) - dfA[tof_mean_cols].min(axis=1)
    thm_cols=[f'thm_{i}' for i in range(1,6) if f'thm_{i}' in dfA.columns]
    if thm_cols:
        dfA['thm_std_across_sensors']   = dfA[thm_cols].std(axis=1)
        dfA['thm_range_across_sensors'] = dfA[thm_cols].max(axis=1) - dfA[thm_cols].min(axis=1)

    mat_A = dfA[final_feature_cols_A].ffill().bfill().fillna(0).values.astype('float32')
    mat_A = scaler_A.transform(mat_A)
    pad_A = pad_sequences([mat_A], maxlen=pad_len_A, padding='post', truncating='post', dtype='float32')
    preds_A = [m.predict(pad_A, verbose=0)[0] for m in models_A] if isinstance(models_A, (list, tuple)) else [models_A.predict(pad_A, verbose=0)[0]]
    avg_pred_A = np.mean(preds_A, axis=0, dtype=np.float32)[None, :]  # (1,C)

    # ================= B branch =================
    dfB   = df_seq_orig.copy()
    accB  = dfB[['acc_x','acc_y','acc_z']].to_numpy(np.float32)
    quatB = dfB[['rot_x','rot_y','rot_z','rot_w']].to_numpy(np.float32)

    dfB['acc_mag']       = np.sqrt(dfB['acc_x']**2+dfB['acc_y']**2+dfB['acc_z']**2)
    dfB['rot_angle']     = 2*np.arccos(dfB['rot_w'].clip(-1,1))
    dfB['acc_mag_jerk']  = dfB['acc_mag'].diff().fillna(0)
    dfB['rot_angle_vel'] = dfB['rot_angle'].diff().fillna(0)

    linear_accel_B = remove_gravity_from_acc3(accB, quatB)
    dfB['linear_acc_x'], dfB['linear_acc_y'], dfB['linear_acc_z'] = linear_accel_B[:,0], linear_accel_B[:,1], linear_accel_B[:,2]
    dfB['linear_acc_mag']      = np.sqrt(dfB['linear_acc_x']**2+dfB['linear_acc_y']**2+dfB['linear_acc_z']**2)
    dfB['linear_acc_mag_jerk'] = dfB['linear_acc_mag'].diff().fillna(0)

    angular_vel_B = calculate_angular_velocity_from_quat3(quatB)
    dfB['angular_vel_x'], dfB['angular_vel_y'], dfB['angular_vel_z'] = angular_vel_B[:,0], angular_vel_B[:,1], angular_vel_B[:,2]
    dfB['angular_distance'] = calculate_angular_distance3(quatB)

    for i in range(1,6):
        pix = [f"tof_{i}_v{p}" for p in range(64) if f"tof_{i}_v{p}" in dfB.columns]
        if pix:
            t = dfB[pix].replace(-1, np.nan)
            dfB[f"tof_{i}_mean"], dfB[f"tof_{i}_std"], dfB[f"tof_{i}_min"], dfB[f"tof_{i}_max"] = \
                t.mean(axis=1), t.std(axis=1), t.min(axis=1), t.max(axis=1)

    mat_B = dfB[final_feature_cols_B].ffill().bfill().fillna(0).values.astype('float32')
    mat_B = scaler_B.transform(mat_B)
    pad_B = pad_sequences([mat_B], maxlen=pad_len_B, padding='post', truncating='post', dtype='float32')
    pred_B = model_B.predict(pad_B, verbose=0)
    if isinstance(pred_B, list): pred_B = pred_B[0]
    avg_pred_B = np.asarray(pred_B, dtype=np.float32)  # (1,C)

    # ================= C branch (PyTorch, CPU only) =================
    dfC  = df_seq_orig.copy()
    mat_C = dfC[final_feature_cols_C].ffill().bfill().fillna(0).values.astype('float32')
    mat_C = scaler_C.transform(mat_C)
    pad_C = pad_sequences_torch3([mat_C], maxlen=pad_len_C, padding='pre', truncating='pre')  # (1,T,F)

    with torch.no_grad():
        xC = torch.from_numpy(pad_C).float().to(device)
        # ç¢ºä¿�æ¨¡å�‹åœ¨ CPU
        models_cpu = []
        for m in pt_models:
            m = m.to(device)
            m.eval()
            models_cpu.append(m)
        logits = torch.mean(torch.stack([m(xC) for m in models_cpu], dim=0), dim=0)  # (1,C)
        avg_pred_C = torch.softmax(logits, dim=1).cpu().numpy()  # (1,C)

    # ================= è��å�ˆ =================
    w = {'A': 0.50, 'B': 0.20, 'C': 0.30}
    final_pred = (w['A']*avg_pred_A + w['B']*avg_pred_B + w['C']*avg_pred_C)
    final_pred = final_pred / (final_pred.sum(axis=1, keepdims=True) + 1e-9)
    return final_pred






# main predict function ========================================
#sequnce will load test.csv file ,demographics will load test_demographics file
# lps->original prediction weighted
#lps ->acend/decend in predict
#-> mix other weighted
#-> ascend/descend weighted mix
#-> pred*lps=wps

def predict(sequence, demographics):
    import copy
    pred0 = predict1(sequence, demographics)[0]
    pred1 = predict2(sequence, demographics)[0]
    pred2 = predict_model3_folds(sequence, demographics,pad_len_C,scaler_C,)[0]
    preds = []
    main_wts = np.asarray([0.271, 0.347, 0.382])
    correct_wts = [+0.0021, -0.0007, -0.0014]
    asc_desc_wts = [0.70, 0.30]
    for a,b,c in zip(pred0,pred1,pred2):
        l_abc = [{ 'wts':main_wts[0], 'pred':a, 'n':'p0', 'result':0 }, { 'wts':main_wts[1], 'pred':b, 'n':'p1', 'result':0 }, { 'wts':main_wts[2], 'pred':c, 'n':'p2', 'result':0 }]
        lps_asc = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=False)
        lps_desc = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=True)
        for asc,correct_wt in zip(lps_asc, correct_wts): asc ['wts'] += correct_wt
        for desc,correct_wt in zip(lps_desc, correct_wts): desc['wts'] += correct_wt
        for asc in lps_asc: asc ['result'] = asc ['pred'] * asc ['wts']
        for desc in lps_desc: desc['result'] = desc['pred'] * desc['wts']
        result_asc = sum([asc ['result'] for asc in lps_asc])
        result_desc = sum([desc['result'] for asc in lps_desc])
        result = result_asc * asc_desc_wts[0] + result_desc * asc_desc_wts[1]
        preds.append(result)
    avg_pred = np.asarray(preds)
    return dataset.le.classes_[avg_pred.argmax()]






import glob; print(glob.glob('/kaggle/input/*'))



# ===== KFold è¨­å®š =====
N_FOLDS   = 5
NUM_CLASS = 7
SEED      = 42

# è¼¸å‡ºç›®éŒ„ï¼ˆKaggle ä¸Šå�¯æ”¹æˆ� /kaggle/workingï¼‰
ART_DIR = "./working_artifacts"

# ç”¢ç‰©å‘½å��ï¼ˆå›ºå®šè¦�å‰‡ï¼Œæ�¨è«–è¼‰å…¥æœƒç”¨ï¼‰
# model1ï¼ˆKerasï¼‰
M1_SCALER_PKL = f"{ART_DIR}/m1_scaler.pkl"
M1_PAD_JSON   = f"{ART_DIR}/m1_padlen.json"
M1_FOLD_W     = f"{ART_DIR}/m1_fold{0}_weights.keras"   # ç”¨ .format(i) ç”Ÿæ¯�æŠ˜

# model2ï¼ˆPyTorchï¼‰
M2_FOLD_W     = f"{ART_DIR}/m2_fold{0}.pt"

# model3-Aï¼ˆKerasï¼‰ï¼Œ3-Bï¼ˆKerasï¼‰ï¼Œ3-Cï¼ˆPyTorchï¼‰
M3A_SCALER_PKL= f"{ART_DIR}/m3a_scaler.pkl"
M3A_PAD_JSON  = f"{ART_DIR}/m3a_padlen.json"
M3A_FOLD_W    = f"{ART_DIR}/m3a_fold{0}.keras"

M3B_SCALER_PKL= f"{ART_DIR}/m3b_scaler.pkl"
M3B_PAD_JSON  = f"{ART_DIR}/m3b_padlen.json"
M3B_FOLD_W    = f"{ART_DIR}/m3b_fold{0}.keras"

M3C_PAD_JSON  = f"{ART_DIR}/m3c_padlen.json"
M3C_FOLD_W    = f"{ART_DIR}/m3c_fold{0}.pt"
import os, json, joblib, glob, random
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

os.makedirs(ART_DIR, exist_ok=True)
rng = np.random.default_rng(SEED)
random.seed(SEED)

def save_json(path, obj):
    with open(path, "w") as f: json.dump(obj, f)

def load_json(path):
    with open(path, "r") as f: return json.load(f)

# ä½ è‹¥æœ‰ participant/user çš„æ¬„ä½�ï¼Œå»ºè­°æ”¹æˆ� GroupKFold/StratifiedGroupKFold é˜²æ´©æ¼�
def make_folds(y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(np.arange(len(y)), y))

# æ±ºå®š pad_lenï¼ˆä¾‹å¦‚å�– 95% åˆ†ä½�ï¼‰
def decide_pad_len(lengths, q=0.95, cap=None):
    L = int(np.quantile(lengths, q))
    if cap: L = min(L, cap)
    return max(8, L)

# Keras çš„ pad_sequencesï¼ˆä½ å·²æœ‰å�Œå��å�¯æ²¿ç”¨ï¼‰
from tensorflow.keras.preprocessing.sequence import pad_sequences



import os, glob, json, joblib, numpy as np, pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

# === å»ºè­°ï¼šç°¡å–®çš„æ¬„ä½�æª¢æŸ¥å°�å·¥å…· ===
def _as_pd(df):
    if hasattr(df, "to_pandas"): return df.to_pandas()
    if isinstance(df, pd.DataFrame): return df.copy()
    raise TypeError(f"sequence must be Polars or pandas DataFrame, got {type(df)}")

def _safe_tof_block(df, i):
    cols = [f"tof_{i}_v{p}" for p in range(64) if f"tof_{i}_v{p}" in df.columns]
    if not cols:
        return None
    t = df[cols].replace(-1, np.nan)
    return t

# =============== ç‰¹å¾µï¼šç­‰å�Œä½  predict1 çš„æµ�ç¨‹ï¼ˆåŠ å…¥å®¹éŒ¯ï¼‰ ===============
def fe_m1_from_sequence(df_seq: pd.DataFrame):
    df = df_seq.copy()

    # é€™ä¸‰å€‹å‡½å¼�ä½ å�Ÿæœ¬æ˜¯ç”¨æ•´å€‹ dfï¼›å¦‚è¦�æ›´åš´è¬¹å�¯æ”¹æˆ�æ¬„ä½�åˆ‡ç‰‡ç‰ˆæœ¬ï¼š
    # linear_accel = remove_gravity_from_acc(df[['acc_x','acc_y','acc_z']], df[['rot_x','rot_y','rot_z','rot_w']])
    linear_accel = remove_gravity_from_acc(df, df)
    df['linear_acc_x'], df['linear_acc_y'], df['linear_acc_z'] = linear_accel[:, 0], linear_accel[:, 1], linear_accel[:, 2]
    df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
    df['linear_acc_mag_jerk'] = df['linear_acc_mag'].diff().fillna(0)

    angular_vel = calculate_angular_velocity_from_quat(df)   # å¦‚éœ€æ›´åš´è¬¹å�¯æ”¹æˆ�æ¬„ä½�åˆ‡ç‰‡ç‰ˆ
    df['angular_vel_x'], df['angular_vel_y'], df['angular_vel_z'] = angular_vel[:, 0], angular_vel[:, 1], angular_vel[:, 2]
    df['angular_distance'] = calculate_angular_distance(df)

    # ToF çµ±è¨ˆï¼šå�ªå°�å­˜åœ¨çš„åƒ�ç´ æ¬„ä½�å�šï¼Œé�¿å…� KeyError
    for i in range(1, 6):
        tof_block = _safe_tof_block(df, i)
        if tof_block is None:
            continue
        df[f'tof_{i}_mean'] = tof_block.mean(axis=1)
        df[f'tof_{i}_std']  = tof_block.std(axis=1)
        df[f'tof_{i}_min']  = tof_block.min(axis=1)
        df[f'tof_{i}_max']  = tof_block.max(axis=1)

    # å�–å‡ºä½ å®šç¾©çš„ç‰¹å¾µæ¬„ä½�
    mat = df[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')
    return mat  # (T, F)

# =============== æ¨¡å�‹ï¼ˆå�¯æ›¿æ�›ç‚ºä½ å�Ÿæœ¬çš„çµ�æ§‹ï¼‰ ===============
def build_model1(input_shape):
    inp = tf.keras.Input(shape=input_shape)           # (pad_len, F)
    x = tf.keras.layers.Masking()(inp)
    x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    out = tf.keras.layers.Dense(NUM_CLASS, activation='softmax')(x)
    return tf.keras.Model(inp, out)

# =============== K-fold è¨“ç·´ ===============
def train_model1_kfold(train_sequences, train_demographics, y):
    # 1) ç‰¹å¾µæŠ½å�–
    X_list = []
    for seq in train_sequences:
        df = seq.to_pandas() if hasattr(seq, "to_pandas") else seq
        X_list.append(fe_m1_from_sequence(df))
    lengths = [x.shape[0] for x in X_list]
    pad_len = decide_pad_len(lengths, q=0.95)
    save_json(M1_PAD_JSON, {"pad_len": pad_len})

    # 2) scalerï¼ˆç”¨å…¨éƒ¨ time-steps fitï¼‰
    scaler = StandardScaler()
    X_all = np.vstack(X_list)                    # (sumT, F)
    scaler.fit(X_all)
    joblib.dump(scaler, M1_SCALER_PKL)

    # 3) å�šæŠ˜
    folds = make_folds(np.asarray(y))
    for fi, (tr_idx, va_idx) in enumerate(folds):
        def pack(indices):
            mats = [scaler.transform(X_list[i]) for i in indices]
            X = pad_sequences(mats, maxlen=pad_len, padding="post",
                              truncating="post", dtype="float32")   # (N, pad_len, F)
            Y = tf.keras.utils.to_categorical(np.asarray([y[i] for i in indices]),
                                              num_classes=NUM_CLASS)
            return X, Y

        Xtr, Ytr = pack(tr_idx)
        Xva, Yva = pack(va_idx)

        model = build_model1((pad_len, Xtr.shape[-1]))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        cb = [
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True,
                                             monitor="val_accuracy"),
            tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5,
                                                 monitor="val_accuracy"),
        ]
        model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
                  epochs=60, batch_size=128, callbacks=cb, verbose=2)
        model.save(M1_FOLD_W.format(fi))         # e.g. ".../m1_fold0.keras"

# =============== æ�¨è«–è¼‰å…¥ï¼ˆä¿®æ­£ glob patternï¼‰ ===============
def _load_model1_for_infer():
    global models1, scaler, pad_len
    if "models1" in globals():  # å·²è¼‰å…¥å°±è·³é��
        return
    scaler  = joblib.load(M1_SCALER_PKL)
    pad_len = load_json(M1_PAD_JSON)["pad_len"]
    # ç›´æ�¥ format("*") â†’ ".../m1_fold*.keras"
    paths = sorted(glob.glob(M1_FOLD_W.format("*")))
    assert len(paths) > 0, f"No fold weights found with pattern: {M1_FOLD_W.format('*')}"
    models1 = [tf.keras.models.load_model(p) for p in paths]

# =============== æ�¨è«–ï¼ˆå›�å‚³ (1, C)ï¼‰ ===============
def predict1_kfold(sequence: pl.DataFrame, demographics: pl.DataFrame):
    _load_model1_for_infer()
    df_seq = _as_pd(sequence)

    # === å�Œ predict1 çš„ç‰¹å¾µæµ�ç¨‹ ===
    linear_accel = remove_gravity_from_acc(df_seq, df_seq)
    df_seq['linear_acc_x'], df_seq['linear_acc_y'], df_seq['linear_acc_z'] = linear_accel[:, 0], linear_accel[:, 1], linear_accel[:, 2]
    df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 + df_seq['linear_acc_y']**2 + df_seq['linear_acc_z']**2)
    df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)

    angular_vel = calculate_angular_velocity_from_quat(df_seq)
    df_seq['angular_vel_x'], df_seq['angular_vel_y'], df_seq['angular_vel_z'] = angular_vel[:, 0], angular_vel[:, 1], angular_vel[:, 2]
    df_seq['angular_distance'] = calculate_angular_distance(df_seq)

    for i in range(1, 6):
        tof_block = _safe_tof_block(df_seq, i)
        if tof_block is None:
            continue
        df_seq[f'tof_{i}_mean'] = tof_block.mean(axis=1)
        df_seq[f'tof_{i}_std']  = tof_block.std(axis=1)
        df_seq[f'tof_{i}_min']  = tof_block.min(axis=1)
        df_seq[f'tof_{i}_max']  = tof_block.max(axis=1)

    mat_unscaled = df_seq[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')
    mat_scaled = scaler.transform(mat_unscaled)
    pad_input = pad_sequences([mat_scaled], maxlen=pad_len, padding='post',
                              truncating='post', dtype='float32')   # (1, pad_len, F)

    # æ¯�æŠ˜è¼¸å‡ºéƒ½æ˜¯ (1, C)ï¼›å�– [0] è®Š (C,) å†�å¹³å�‡
    fold_probs = [m.predict(pad_input, verbose=0)[0] for m in models1]   # list[(C,)]
    avg_pred = np.mean(fold_probs, axis=0, dtype=np.float32)            # (C,)
    return avg_pred[None, :]   # â†� (1, C) æ–¹ä¾¿å¤–å±¤ç”¨ [...][0] å�–åˆ°å�‘é‡�






# =========================
# Model2 â€” PyTorch K-fold
# =========================
import os, glob, json, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F

# ç”¢ç‰©è·¯å¾‘ï¼ˆæ²¿ç”¨ä½ å‰�é�¢å®šç¾©çš„å¸¸æ•¸ï¼‰
M2_FOLD_W      = f"{ART_DIR}/m2_fold{{}}.pt"
M2_META_JSON   = f"{ART_DIR}/m2_meta.json"
DEVICE         = torch.device("cpu")  # Kaggle è©•åˆ†ç”¨ CPU è¼ƒç©©

# ---------- å°�å·¥å…· ----------
def _to_tensor(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return x  # å·²ç¶“æ˜¯ Tensor

def _ensure_batch_time_feat(x):
    """æŠŠè¼¸å…¥æ•´ç�†æˆ� (B, T, F)ï¼›å…�è¨± (T, F) æˆ– (1, T, F)ã€‚"""
    x = _to_tensor(x).float()
    if x.dim() == 2:         # (T, F) -> (1, T, F)
        x = x.unsqueeze(0)
    elif x.dim() == 3:       # (B, T, F) ç›´æ�¥ç”¨
        pass
    else:
        raise ValueError(f"expected (T,F) or (B,T,F), got {tuple(x.shape)}")
    return x

def _feat_dim(x):
    x = _ensure_batch_time_feat(x)
    return x.shape[-1]

# ---------- æ¨¡å�‹ ----------
class SimpleFusionNet(nn.Module):
    """
    æ¯�å€‹æ¨¡æ…‹ï¼šæ™‚é–“å¹³å�‡ -> ç·šæ€§ -> ReLUï¼›ä¸‰æ¨¡æ…‹æ‹¼æ�¥ -> MLP -> logits
    æ�¥å�—ä¸�å�Œ Tï¼Œå› ç‚ºæˆ‘å€‘å�š mean pooling
    """
    def __init__(self, imu_dim, thm_dim, tof_dim, hidden=128, num_classes=NUM_CLASS):
        super().__init__()
        self.imu_fc = nn.Linear(imu_dim, hidden)
        self.thm_fc = nn.Linear(thm_dim, hidden)
        self.tof_fc = nn.Linear(tof_dim, hidden)

        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden*3, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, num_classes),
        )

    def _pool(self, x):         # x: (B, T, F)
        return x.mean(dim=1)    # (B, F)

    def forward(self, imu, thm, tof):   # æ”¯æ�´ (T,F) æˆ– (B,T,F)
        imu = _ensure_batch_time_feat(imu)
        thm = _ensure_batch_time_feat(thm)
        tof = _ensure_batch_time_feat(tof)

        imu_f = torch.relu(self.imu_fc(self._pool(imu)))  # (B, H)
        thm_f = torch.relu(self.thm_fc(self._pool(thm)))
        tof_f = torch.relu(self.tof_fc(self._pool(tof)))
        z = torch.cat([imu_f, thm_f, tof_f], dim=-1)
        logits = self.head(z)                                  # (B, C)
        return logits

def build_model2(imu_dim, thm_dim, tof_dim):
    return SimpleFusionNet(imu_dim, thm_dim, tof_dim, hidden=128, num_classes=NUM_CLASS)

# ---------- è¨“ç·´ ----------
def train_model2_kfold(train_sequences, train_demographics, y):
    y = np.asarray(y, dtype=np.int64)
    folds = make_folds(y)

    # å…ˆæŠŠæ‰€æœ‰æ¨£æœ¬å�šå‰�è™•ç�†ï¼Œä»¥å…�æ¯�æŠ˜é‡�å�š
    all_imus, all_thms, all_tofs = [], [], []
    for seq in train_sequences:
        imu, thm, tof = dataset.full_dataset.inference_process(seq)
        all_imus.append(_ensure_batch_time_feat(imu)[0].cpu().numpy())  # å­˜ (T,F)
        all_thms.append(_ensure_batch_time_feat(thm)[0].cpu().numpy())
        all_tofs.append(_ensure_batch_time_feat(tof)[0].cpu().numpy())

    # å¾�ç¬¬ä¸€ç­†è³‡æ–™å�µæ¸¬å�„æ¨¡æ…‹ç‰¹å¾µç¶­åº¦ï¼Œå¯«åˆ° meta ä¾›æ�¨è«–å»ºæ¨¡ç”¨
    imu_dim = all_imus[0].shape[-1]
    thm_dim = all_thms[0].shape[-1]
    tof_dim = all_tofs[0].shape[-1]
    with open(M2_META_JSON, "w") as f:
        json.dump({"imu_dim": int(imu_dim), "thm_dim": int(thm_dim), "tof_dim": int(tof_dim)}, f)

    for fi, (tr_idx, va_idx) in enumerate(folds):
        model = build_model2(imu_dim, thm_dim, tof_dim).to(DEVICE)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        best_va = float("inf")
        bad, patience = 0, 5

        for epoch in range(50):
            # ---- train ----
            model.train()
            idxs = np.random.permutation(tr_idx)
            for i in idxs:
                imu = _ensure_batch_time_feat(all_imus[i]).to(DEVICE)
                thm = _ensure_batch_time_feat(all_thms[i]).to(DEVICE)
                tof = _ensure_batch_time_feat(all_tofs[i]).to(DEVICE)
                yi  = torch.tensor([y[i]], dtype=torch.long, device=DEVICE)

                opt.zero_grad()
                logits = model(imu, thm, tof)   # (1, C)
                loss   = crit(logits, yi)
                loss.backward()
                opt.step()

            # ---- valid ----
            model.eval()
            with torch.no_grad():
                va_losses = []
                for i in va_idx:
                    imu = _ensure_batch_time_feat(all_imus[i]).to(DEVICE)
                    thm = _ensure_batch_time_feat(all_thms[i]).to(DEVICE)
                    tof = _ensure_batch_time_feat(all_tofs[i]).to(DEVICE)
                    yi  = torch.tensor([y[i]], dtype=torch.long, device=DEVICE)
                    lv  = crit(model(imu, thm, tof), yi).item()
                    va_losses.append(lv)
                va_loss = float(np.mean(va_losses)) if va_losses else 0.0

            if va_loss < best_va:
                best_va = va_loss
                bad = 0
                torch.save(model.state_dict(), M2_FOLD_W.format(fi))
            else:
                bad += 1
                if bad >= patience:
                    break

        # è‹¥æ—©å�œå‰�æ²’å­˜æˆ�åŠŸï¼Œè‡³å°‘å­˜æœ€å¾Œä¸€æ¬¡
        if not os.path.exists(M2_FOLD_W.format(fi)):
            torch.save(model.state_dict(), M2_FOLD_W.format(fi))

# ---------- è¼‰å…¥æ�¨è«– ----------
def _load_model2_for_infer():
    global models2
    if "models2" in globals():
        return
    # è®€æ¨¡æ…‹ç¶­åº¦
    with open(M2_META_JSON, "r") as f:
        meta = json.load(f)
    imu_dim, thm_dim, tof_dim = meta["imu_dim"], meta["thm_dim"], meta["tof_dim"]

    paths = sorted(glob.glob(M2_FOLD_W.format("*")))
    assert len(paths) > 0, f"No model2 fold weights found at pattern: {M2_FOLD_W.format('*')}"
    models2 = []
    for p in paths:
        m = build_model2(imu_dim, thm_dim, tof_dim).to(DEVICE)
        m.load_state_dict(torch.load(p, map_location=DEVICE))
        m.eval()
        models2.append(m)

# ---------- æ�¨è«–ï¼ˆå›�å‚³ (1, NUM_CLASS)ï¼‰ ----------
def predict2_kfold(sequence: pl.DataFrame, demographics: pl.DataFrame):
    _load_model2_for_infer()
    imu, thm, tof = dataset.full_dataset.inference_process(sequence)
    imu = _ensure_batch_time_feat(imu).to(DEVICE)
    thm = _ensure_batch_time_feat(thm).to(DEVICE)
    tof = _ensure_batch_time_feat(tof).to(DEVICE)

    with torch.no_grad():
        # å¹³å�‡ logits å†� softmaxï¼Œè¼ƒç©©
        logits_list = [m(imu, thm, tof) for m in models2]     # list[(1, C)]
        logits_avg  = torch.mean(torch.stack(logits_list, dim=0), dim=0)  # (1, C)
        probs = F.softmax(logits_avg, dim=1).cpu().numpy()    # (1, C)
    return probs



# =========================
# K-FOLD TRAINING FOR MODEL3 (A/B: Keras, C: PyTorch)
# Keep predict3 I/O unchanged
# =========================
import os, json, glob, joblib, numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import torch, torch.nn as nn
import torch.nn.functional as F

# --------- constants ---------
NUM_CLASS = 7
N_FOLDS   = 5
SEED      = 42
ART_DIR   = "./working_artifacts"
os.makedirs(ART_DIR, exist_ok=True)

# A
M3A_SCALER_PKL = f"{ART_DIR}/m3a_scaler.pkl"
M3A_PAD_JSON   = f"{ART_DIR}/m3a_padlen.json"
M3A_FOLD_W     = f"{ART_DIR}/m3a_fold{{}}.keras"

# B
M3B_SCALER_PKL = f"{ART_DIR}/m3b_scaler.pkl"
M3B_PAD_JSON   = f"{ART_DIR}/m3b_padlen.json"
M3B_FOLD_W     = f"{ART_DIR}/m3b_fold{{}}.keras"

# C
M3C_PAD_JSON   = f"{ART_DIR}/m3c_padlen.json"
M3C_FOLD_W     = f"{ART_DIR}/m3c_fold{{}}.pt"

rng = np.random.default_rng(SEED)

def save_json(path, obj):
    with open(path, "w") as f: json.dump(obj, f)
def load_json(path):
    with open(path, "r") as f: return json.load(f)

def make_folds(y):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    return list(skf.split(np.arange(len(y)), y))

def decide_pad_len(lengths, q=0.95, cap=None):
    L = int(np.quantile(lengths, q))
    if cap is not None:
        L = min(L, cap)
    return max(8, L)

# ---------- FEATURES (same as your predict3) ----------
# Expect the following globals already defined in your file:
#   final_feature_cols_A, final_feature_cols_B, final_feature_cols_C
#   remove_gravity_from_acc3, calculate_angular_velocity_from_quat3, calculate_angular_distance3
def _as_pd(df):
    import pandas as pd
    if hasattr(df, "to_pandas"):
        return df.to_pandas()
    if isinstance(df, pd.DataFrame):
        return df.copy()
    raise TypeError(f"sequence must be Polars or pandas DataFrame, got {type(df)}")

def _require_cols(df, cols, ctx=""):
    miss = [c for c in cols if c not in df.columns]
    if miss:
        raise ValueError(f"[{ctx}] missing columns: {miss}. Available: {list(df.columns)[:10]}...")

def fe_m3A(df_seq_orig: pd.DataFrame):
    df = df_seq_orig.copy()
    lin = remove_gravity_from_acc3(df[['acc_x','acc_y','acc_z']], df[['rot_x','rot_y','rot_z','rot_w']])
    df['linear_acc_x'], df['linear_acc_y'], df['linear_acc_z'] = lin[:,0], lin[:,1], lin[:,2]
    df['linear_acc_mag'] = np.linalg.norm(lin, axis=1)
    df['linear_acc_mag_jerk'] = df['linear_acc_mag'].diff().fillna(0)
    ang = calculate_angular_velocity_from_quat3(df[['rot_x','rot_y','rot_z','rot_w']])
    df['angular_vel_x'], df['angular_vel_y'], df['angular_vel_z'] = ang[:,0], ang[:,1], ang[:,2]
    df['angular_distance'] = calculate_angular_distance3(df[['rot_x','rot_y','rot_z','rot_w']])
    for col in ['rot_x','rot_y','rot_z','rot_w']:
        df[f'{col}_diff'] = df[col].diff().fillna(0)
    for col in ['linear_acc_mag','linear_acc_mag_jerk','angular_distance']:
        df[f'{col}_skew'] = df[col].skew()
        df[f'{col}_kurt'] = df[col].kurtosis()
    for i in range(1,6):
        cols = [f"tof_{i}_v{p}" for p in range(64) if f"tof_{i}_v{p}" in df.columns]
        if cols:
            t = df[cols].replace(-1, np.nan)
            df[f'tof_{i}_mean'], df[f'tof_{i}_std'], df[f'tof_{i}_min'], df[f'tof_{i}_max'] = \
                t.mean(axis=1), t.std(axis=1), t.min(axis=1), t.max(axis=1)
    tof_mean_cols=[f'tof_{i}_mean' for i in range(1,6) if f'tof_{i}_mean' in df.columns]
    if tof_mean_cols:
        df['tof_std_across_sensors']   = df[tof_mean_cols].std(axis=1)
        df['tof_range_across_sensors'] = df[tof_mean_cols].max(axis=1) - df[tof_mean_cols].min(axis=1)
    thm_cols=[f'thm_{i}' for i in range(1,6) if f'thm_{i}' in df.columns]
    if thm_cols:
        df['thm_std_across_sensors']   = df[thm_cols].std(axis=1)
        df['thm_range_across_sensors'] = df[thm_cols].max(axis=1) - df[thm_cols].min(axis=1)
    mat = df[final_feature_cols_A].ffill().bfill().fillna(0).values.astype('float32')
    return mat

def fe_m3B(df_seq_orig: pd.DataFrame):
    df = df_seq_orig.copy()
    _require_cols(df, ['acc_x','acc_y','acc_z','rot_x','rot_y','rot_z','rot_w'], "fe_m3B")

    df['acc_mag'] = np.sqrt(df['acc_x']**2+df['acc_y']**2+df['acc_z']**2)
    df['rot_angle'] = 2*np.arccos(df['rot_w'].clip(-1,1))
    df['acc_mag_jerk'] = df['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df['rot_angle'].diff().fillna(0)

    lin = remove_gravity_from_acc3(df[['acc_x','acc_y','acc_z']],
                                   df[['rot_x','rot_y','rot_z','rot_w']])
    df['linear_acc_x'], df['linear_acc_y'], df['linear_acc_z'] = lin[:,0], lin[:,1], lin[:,2]
    df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2+df['linear_acc_y']**2+df['linear_acc_z']**2)
    df['linear_acc_mag_jerk'] = df['linear_acc_mag'].diff().fillna(0)

    ang = calculate_angular_velocity_from_quat3(df[['rot_x','rot_y','rot_z','rot_w']])
    df['angular_vel_x'], df['angular_vel_y'], df['angular_vel_z'] = ang[:,0], ang[:,1], ang[:,2]
    df['angular_distance'] = calculate_angular_distance3(df[['rot_x','rot_y','rot_z','rot_w']])

    for i in range(1,6):
        cols=[f"tof_{i}_v{p}" for p in range(64) if f"tof_{i}_v{p}" in df.columns]
        if cols:
            t=df[cols].replace(-1,np.nan)
            df[f"tof_{i}_mean"],df[f"tof_{i}_std"],df[f"tof_{i}_min"],df[f"tof_{i}_max"]= \
                t.mean(axis=1),t.std(axis=1),t.min(axis=1),t.max(axis=1)

    mat = df[final_feature_cols_B].ffill().bfill().fillna(0).values.astype('float32')
    return mat


def fe_m3C(df_seq_orig: pd.DataFrame):
    # ä½  C è·¯å�Ÿæœ¬å°±æ˜¯å�– final_feature_cols_C ç„¶å¾Œå‰�ç½® scalerï¼Œå†� pad åˆ°å·¦å�´
    df = df_seq_orig.copy()
    mat = df[final_feature_cols_C].ffill().bfill().fillna(0).values.astype('float32')
    return mat  # (T, F)
def build_model3A(input_shape):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Masking()(inp)
    x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu')(x)
    x = tf.keras.layers.GlobalAveragePooling1D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    out = tf.keras.layers.Dense(NUM_CLASS, activation='softmax')(x)
    return tf.keras.Model(inp, out)

def build_model3B(input_shape):
    inp = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Masking()(inp)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(96, return_sequences=True))(x)
    x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(64))(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    out = tf.keras.layers.Dense(NUM_CLASS, activation='softmax')(x)
    return tf.keras.Model(inp, out)
class M3CNet(nn.Module):
    # ç°¡å–®ç©©å®šï¼šæ™‚é–“å¹³å�‡ + MLP
    def __init__(self, in_dim, hidden=128, num_classes=NUM_CLASS):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):          # x: (B, T, F)
        x = x.mean(dim=1)          # (B, F)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)            # (B, C) logits
        return x

def build_model3C(in_dim):
    return M3CNet(in_dim=in_dim, hidden=128, num_classes=NUM_CLASS)
def _pad_np_pre(mats, pad_len):
    # mats: list of (T_i, F) -> (N, pad_len, F), pre-padding
    Fdim = mats[0].shape[-1]
    out = np.zeros((len(mats), pad_len, Fdim), dtype=np.float32)
    for i, m in enumerate(mats):
        t = m.shape[0]
        if t >= pad_len:
            out[i] = m[-pad_len:]                  # keep last pad_len
        else:
            out[i, -t:] = m                        # right align (pre-padding zeros)
    return out

def train_model3C_kfold(train_sequences, y):
    X_list = [fe_m3C(seq.to_pandas() if hasattr(seq,"to_pandas") else seq) for seq in train_sequences]
    lengths = [x.shape[0] for x in X_list]
    pad_len = decide_pad_len(lengths, q=0.95)
    save_json(M3C_PAD_JSON, {"pad_len": pad_len})

    folds = make_folds(np.asarray(y))
    device = torch.device("cpu")
    for fi, (tr, va) in enumerate(folds):
        def pack(idx):
            mats = [X_list[i] for i in idx]
            X = _pad_np_pre(mats, pad_len)                 # (N, pad_len, F)
            Y = np.asarray([y[i] for i in idx], dtype=np.int64)
            return X, Y

        Xtr, Ytr = pack(tr); Xva, Yva = pack(va)
        Fdim = Xtr.shape[-1]
        model = build_model3C(in_dim=Fdim).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()

        best_va = 1e9
        patience, bad = 6, 0
        for epoch in range(60):
            # train
            model.train()
            idxs = np.arange(len(Xtr))
            rng.shuffle(idxs)
            for s in range(0, len(idxs), 128):
                j = idxs[s:s+128]
                xb = torch.from_numpy(Xtr[j]).to(device)
                yb = torch.from_numpy(Ytr[j]).to(device)
                opt.zero_grad()
                logits = model(xb)
                loss = crit(logits, yb)
                loss.backward(); opt.step()

            # valid
            model.eval()
            with torch.no_grad():
                xv = torch.from_numpy(Xva).to(device)
                yv = torch.from_numpy(Yva).to(device)
                lv = crit(model(xv), yv).item()
            if lv < best_va:
                best_va = lv; bad = 0
                torch.save(model.state_dict(), M3C_FOLD_W.format(fi))
            else:
                bad += 1
                if bad >= patience:
                    break

        # è‹¥æ²’è¢«æ—©å�œå­˜ä¸‹ï¼Œè‡³å°‘å­˜æœ€å¾Œä¸€ç‰ˆ
        if not os.path.exists(M3C_FOLD_W.format(fi)):
            torch.save(model.state_dict(), M3C_FOLD_W.format(fi))
# =============== A è·¯ï¼šK-fold è¨“ç·´ï¼ˆKerasï¼‰ ===============
def train_model3A_kfold(train_sequences, y):
    # 1) æŠ½ç‰¹å¾µ
    X_list = [fe_m3A(seq.to_pandas() if hasattr(seq, "to_pandas") else seq)
              for seq in train_sequences]
    lengths = [x.shape[0] for x in X_list]
    pad_len = decide_pad_len(lengths, q=0.95)
    save_json(M3A_PAD_JSON, {"pad_len": pad_len})

    # 2) scaler ç”¨å…¨éƒ¨æ¨£æœ¬çš„ time-steps fit
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_list))
    joblib.dump(scaler, M3A_SCALER_PKL)

    # 3) K-fold
    folds = make_folds(np.asarray(y))

    for fi, (tr_idx, va_idx) in enumerate(folds):
        def pack(idxs):
            mats = [scaler.transform(X_list[i]) for i in idxs]
            X = pad_sequences(mats, maxlen=pad_len, padding="post",
                              truncating="post", dtype="float32")  # (N, pad_len, F)
            Y = tf.keras.utils.to_categorical(np.asarray([y[i] for i in idxs]),
                                              num_classes=NUM_CLASS)
            return X, Y

        Xtr, Ytr = pack(tr_idx)
        Xva, Yva = pack(va_idx)

        model = build_model3A((pad_len, Xtr.shape[-1]))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                             patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",
                                                 patience=4, factor=0.5),
        ]
        model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
                  epochs=60, batch_size=128, callbacks=cbs, verbose=2)

        model.save(M3A_FOLD_W.format(fi))


# =============== B è·¯ï¼šK-fold è¨“ç·´ï¼ˆKerasï¼‰ ===============
def train_model3B_kfold(train_sequences, y):
    # 1) æŠ½ç‰¹å¾µ
    X_list = [fe_m3B(seq.to_pandas() if hasattr(seq, "to_pandas") else seq)
              for seq in train_sequences]
    lengths = [x.shape[0] for x in X_list]
    pad_len = decide_pad_len(lengths, q=0.95)
    save_json(M3B_PAD_JSON, {"pad_len": pad_len})

    # 2) scaler
    scaler = StandardScaler()
    scaler.fit(np.vstack(X_list))
    joblib.dump(scaler, M3B_SCALER_PKL)

    # 3) K-fold
    folds = make_folds(np.asarray(y))

    for fi, (tr_idx, va_idx) in enumerate(folds):
        def pack(idxs):
            mats = [scaler.transform(X_list[i]) for i in idxs]
            X = pad_sequences(mats, maxlen=pad_len, padding="post",
                              truncating="post", dtype="float32")
            Y = tf.keras.utils.to_categorical(np.asarray([y[i] for i in idxs]),
                                              num_classes=NUM_CLASS)
            return X, Y

        Xtr, Ytr = pack(tr_idx)
        Xva, Yva = pack(va_idx)

        model = build_model3B((pad_len, Xtr.shape[-1]))
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                      loss="categorical_crossentropy",
                      metrics=["accuracy"])
        cbs = [
            tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",
                                             patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_accuracy",
                                                 patience=4, factor=0.5),
        ]
        model.fit(Xtr, Ytr, validation_data=(Xva, Yva),
                  epochs=60, batch_size=128, callbacks=cbs, verbose=2)

        model.save(M3B_FOLD_W.format(fi))


# =============== ä¸€é�µè¨“ç·´ï¼ˆA/B/C å…¨éƒ¨ï¼‰ ===============
def train_model3_all_kfold(train_sequences, y):
    train_model3A_kfold(train_sequences, y)
    train_model3B_kfold(train_sequences, y)
    train_model3C_kfold(train_sequences, y)
# ---------- Inference loader ----------
def _load_model3_for_infer():
    # A
    global models_A, scaler_A, pad_len_A
    if "models_A" not in globals():
        scaler_A = joblib.load(M3A_SCALER_PKL)
        pad_len_A = load_json(M3A_PAD_JSON)["pad_len"]
        paths_A = sorted(glob.glob(M3A_FOLD_W.format("*")))
        assert paths_A, f"No A-fold weights: {M3A_FOLD_W.format('*')}"
        models_A = [tf.keras.models.load_model(p) for p in paths_A]

    # Bï¼ˆå¤šæŠ˜ï¼‰
    global models_B, scaler_B, pad_len_B
    if "models_B" not in globals():
        scaler_B = joblib.load(M3B_SCALER_PKL)
        pad_len_B = load_json(M3B_PAD_JSON)["pad_len"]
        paths_B = sorted(glob.glob(M3B_FOLD_W.format("*")))
        assert paths_B, f"No B-fold weights: {M3B_FOLD_W.format('*')}"
        models_B = [tf.keras.models.load_model(p) for p in paths_B]

    # Cï¼ˆPyTorch å¤šæŠ˜ï¼‰
    global pt_models, pad_len_C, device
    if "pt_models" not in globals():
        pad_len_C = load_json(M3C_PAD_JSON)["pad_len"]
        device = torch.device("cpu")
        paths_C = sorted(glob.glob(M3C_FOLD_W.format("*")))
        assert paths_C, f"No C-fold weights: {M3C_FOLD_W.format('*')}"
        # ä»¥ç¬¬ä¸€å€‹æ¨£æœ¬çš„ç‰¹å¾µç¶­åº¦å»ºç«‹æ¨¡å�‹ï¼ˆæ�¨è«–æ™‚è¨ˆç®—ï¼‰
        # å…ˆæš«å­˜ï¼ŒçœŸæ­£å»ºæ¨¡åœ¨ predict3_kfold å…§æ ¹æ“š feat_dim æ±ºå®š
        pt_models = paths_C  # å…ˆæ”¾è·¯å¾‘ï¼Œç¨�å¾Œå¯¦ä¾‹åŒ–

# ---------- predict3_kfold (I/O èˆ‡ä½ å�Ÿ predict3 ç›¸å�Œï¼šå›�å‚³ (1, NUM_CLASS)) ----------
def predict3_kfold(sequence: pl.DataFrame, demographics: pl.DataFrame):
    _load_model3_for_infer()
    df_seq_orig = _as_pd(sequence)

    # ---- A è·¯ ----
    mat_A = fe_m3A(df_seq_orig)                                    # (T, F_A)
    mat_A = scaler_A.transform(mat_A)
    pad_input_A = pad_sequences([mat_A], maxlen=pad_len_A,
                                padding='post', truncating='post', dtype='float32')
    preds_A = [m.predict(pad_input_A, verbose=0)[0] for m in models_A]   # list[(C,)]
    avg_pred_A = np.mean(preds_A, axis=0, dtype=np.float32)[None, :]     # (1, C)

    # ---- B è·¯ï¼ˆå¤šæŠ˜å¹³å�‡ï¼‰----
    mat_B = fe_m3B(df_seq_orig)
    mat_B = scaler_B.transform(mat_B)
    pad_input_B = pad_sequences([mat_B], maxlen=pad_len_B,
                                padding='post', truncating='post', dtype='float32')
    preds_B = [m.predict(pad_input_B, verbose=0)[0] for m in models_B]
    avg_pred_B = np.mean(preds_B, axis=0, dtype=np.float32)[None, :]     # (1, C)

    # ---- C è·¯ï¼ˆPyTorchï¼Œå¤šæŠ˜ logits å¹³å�‡ï¼‰----
    mat_C = fe_m3C(df_seq_orig)                                        # (T, F_C)
    pad_np_C = _pad_np_pre([mat_C], pad_len_C)                          # (1, pad_len, F_C)

    # ä¾� F_C å»ºç«‹æ¨¡å�‹ï¼ˆå�ªå�šä¸€æ¬¡ï¼‰
    global pt_models
    if isinstance(pt_models[0], str):  # é‚„æ˜¯è·¯å¾‘ â†’ è½‰æˆ�å·²è¼‰å…¥çš„æ¨¡å�‹
        Fdim = pad_np_C.shape[-1]
        loaded = []
        for p in pt_models:
            m = build_model3C(in_dim=Fdim).to(device)
            m.load_state_dict(torch.load(p, map_location=device))
            m.eval()
            loaded.append(m)
        pt_models = loaded

    with torch.no_grad():
        xC = torch.from_numpy(pad_np_C).to(device)
        logits_list = [m(xC) for m in pt_models]                        # list[(1, C)]
        logits_avg  = torch.mean(torch.stack(logits_list, dim=0), dim=0)
        avg_pred_C  = torch.softmax(logits_avg, dim=1).cpu().numpy()    # (1, C)

    # ---- è��å�ˆï¼ˆç¶­æŒ�ä½ çš„æ¬Šé‡�ï¼‰----
    weights = {'A': 0.50, 'B': 0.20, 'C': 0.30}
    final_pred_proba = (weights['A'] * avg_pred_A +
                        weights['B'] * avg_pred_B +
                        weights['C'] * avg_pred_C)                      # (1, C)
    return final_pred_proba



def predict(sequence, demographics):
    import copy
    pred0 = predict1_kfold(sequence, demographics)[0]
    pred1 = predict2_kfold(sequence, demographics)[0]
    pred2 = predict3(sequence, demographics)[0]
    preds = []
    main_wts = np.asarray([0.271, 0.347, 0.382])
    correct_wts = [+0.0021, -0.0007, -0.0014]
    asc_desc_wts = [0.70, 0.30]
    for a,b,c in zip(pred0,pred1,pred2):
        l_abc = [{ 'wts':main_wts[0], 'pred':a, 'n':'p0', 'result':0 }, { 'wts':main_wts[1], 'pred':b, 'n':'p1', 'result':0 }, { 'wts':main_wts[2], 'pred':c, 'n':'p2', 'result':0 }]
        lps_asc = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=False)
        lps_desc = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=True)
        for asc,correct_wt in zip(lps_asc, correct_wts): asc ['wts'] += correct_wt
        for desc,correct_wt in zip(lps_desc, correct_wts): desc['wts'] += correct_wt
        for asc in lps_asc: asc ['result'] = asc ['pred'] * asc ['wts']
        for desc in lps_desc: desc['result'] = desc['pred'] * desc['wts']
        result_asc = sum([asc ['result'] for asc in lps_asc])
        result_desc = sum([d['result'] for d in lps_desc])
        result = result_asc * asc_desc_wts[0] + result_desc * asc_desc_wts[1]
        preds.append(result)
    avg_pred = np.asarray(preds)
    return dataset.le.classes_[avg_pred.argmax()]



import warnings
warnings.simplefilter("ignore")

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










