import os
import torch
import kagglehub
from pathlib import Path
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from collections import defaultdict
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.notebook import tqdm
from torch.amp import autocast
import pandas as pd
import polars as pl
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from transformers import BertConfig, BertModel

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
            delta_rot = rot_t.inv() * rot_t_plus_dt
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
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
            angular_dist[i] = 0
            continue
        try:
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)
            relative_rotation = r1.inv() * r2
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0 # Ğ’ Ñ�Ğ»ÑƒÑ‡Ğ°Ğµ Ğ½ĞµĞ´ĞµĞ¹Ñ�Ñ‚Ğ²Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ñ‹Ñ… ĞºĞ²Ğ°Ñ‚ĞµÑ€Ğ½Ğ¸Ğ¾Ğ½Ğ¾Ğ²
            pass
    return angular_dist

class CMIFeDataset(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.init_feature_names(data_path)
        df = self.generate_features(pd.read_csv(data_path, usecols=set(self.base_cols+self.feature_cols)))
        self.generate_dataset(df)

    def init_feature_names(self, data_path):
        self.imu_engineered_features = [
            'acc_mag', 'rot_angle',
            'acc_mag_jerk', 'rot_angle_vel',
            'linear_acc_mag', 'linear_acc_mag_jerk',
            'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
            'angular_distance'
        ]

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
        self.base_cols = ['acc_x', 'acc_y', 'acc_z',
                          'rot_x', 'rot_y', 'rot_z', 'rot_w',
                          'sequence_id', 'subject', 
                          'sequence_type', 'gesture', 'orientation'] + [c for c in columns if c.startswith('thm_')] + [f"tof_{i}_v{p}" for i in range(1, 6) for p in range(64)]
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
                new_columns.update({
                    f'tof_{i}_mean': tof_data.mean(axis=1),
                    f'tof_{i}_std': tof_data.std(axis=1),
                    f'tof_{i}_min': tof_data.min(axis=1),
                    f'tof_{i}_max': tof_data.max(axis=1)
                })
                if self.tof_mode > 1:
                    region_size = 64 // self.tof_mode
                    for r in range(self.tof_mode):
                        region_data = tof_data.iloc[:, r*region_size : (r+1)*region_size]
                        new_columns.update({
                            f'tof{self.tof_mode}_{i}_region_{r}_mean': region_data.mean(axis=1),
                            f'tof{self.tof_mode}_{i}_region_{r}_std': region_data.std(axis=1),
                            f'tof{self.tof_mode}_{i}_region_{r}_min': region_data.min(axis=1),
                            f'tof{self.tof_mode}_{i}_region_{r}_max': region_data.max(axis=1)
                        })
                if self.tof_mode == -1:
                    for mode in [2, 4, 8, 16, 32]:
                        region_size = 64 // mode
                        for r in range(mode):
                            region_data = tof_data.iloc[:, r*region_size : (r+1)*region_size]
                            new_columns.update({
                                f'tof{mode}_{i}_region_{r}_mean': region_data.mean(axis=1),
                                f'tof{mode}_{i}_region_{r}_std': region_data.std(axis=1),
                                f'tof{mode}_{i}_region_{r}_min': region_data.min(axis=1),
                                f'tof{mode}_{i}_region_{r}_max': region_data.max(axis=1)
                            })
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
        dummy_df = pd.DataFrame(
            np.array([[self.imu_nan_value]*len(self.imu_cols) + 
                     [self.thm_nan_value]*len(self.thm_cols) +
                     [self.tof_nan_value]*len(self.tof_cols)]),
            columns=self.imu_cols + self.thm_cols + self.tof_cols
        )
        
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
        return torch.full(imu.shape, self.imu_scaled_nan, device=imu.device), \
            torch.full(thm.shape, self.thm_scaled_nan, device=thm.device), \
            torch.full(tof.shape, self.tof_scaled_nan, device=tof.device)

    def inference_process(self, sequence):
        df_seq = sequence.to_pandas().copy()
        if not all(c in df_seq.columns for c in self.imu_engineered_features):
            df_seq['acc_mag'] = np.sqrt(df_seq['acc_x']**2 + df_seq['acc_y']**2 + df_seq['acc_z']**2)
            df_seq['rot_angle'] = 2 * np.arccos(df_seq['rot_w'].clip(-1, 1))
            df_seq['acc_mag_jerk'] = df_seq['acc_mag'].diff().fillna(0)
            df_seq['rot_angle_vel'] = df_seq['rot_angle'].diff().fillna(0)
            if all(col in df_seq.columns for col in ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']):
                linear_accel = remove_gravity_from_acc(
                    df_seq[['acc_x', 'acc_y', 'acc_z']], 
                    df_seq[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
                )
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
                new_columns.update({
                    f'tof_{i}_mean': tof_data.mean(axis=1),
                    f'tof_{i}_std': tof_data.std(axis=1),
                    f'tof_{i}_min': tof_data.min(axis=1),
                    f'tof_{i}_max': tof_data.max(axis=1)
                })
                if self.tof_mode > 1:
                    region_size = 64 // self.tof_mode
                    for r in range(self.tof_mode):
                        region_data = tof_data.iloc[:, r*region_size : (r+1)*region_size]
                        new_columns.update({
                            f'tof{self.tof_mode}_{i}_region_{r}_mean': region_data.mean(axis=1),
                            f'tof{self.tof_mode}_{i}_region_{r}_std': region_data.std(axis=1),
                            f'tof{self.tof_mode}_{i}_region_{r}_min': region_data.min(axis=1),
                            f'tof{self.tof_mode}_{i}_region_{r}_max': region_data.max(axis=1)
                        })
                if self.tof_mode == -1:
                    for mode in [2, 4, 8, 16, 32]:
                        region_size = 64 // mode
                        for r in range(mode):
                            region_data = tof_data.iloc[:, r*region_size : (r+1)*region_size]
                            new_columns.update({
                                f'tof{mode}_{i}_region_{r}_mean': region_data.mean(axis=1),
                                f'tof{mode}_{i}_region_{r}_std': region_data.std(axis=1),
                                f'tof{mode}_{i}_region_{r}_min': region_data.min(axis=1),
                                f'tof{mode}_{i}_region_{r}_max': region_data.max(axis=1)
                            })
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
        
        print("\nCross_Validation fold:")
        for fold_idx in range(self.n_folds):
            train_fold, valid_fold = self.get_fold_datasets(fold_idx)
            train_counts = get_label_counts(train_fold)
            valid_counts = get_label_counts(valid_fold)
                
            print(f"\nFold {fold_idx + 1}:")
            print(f"{'category':<50} {'training_set':<10} {'validation_set':<10}")
            for name in self.class_names:
                print(f"{name:<50} {train_counts[name]:<10} {valid_counts[name]:<10}")




class SEBlock(nn.Module):
    def __init__(self, channels, reduction = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        se = F.adaptive_avg_pool1d(x, 1).squeeze(-1)      # -> (B, C)
        se = F.relu(self.fc1(se), inplace=True)          # -> (B, C//r)
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)    # -> (B, C, 1)
        return x * se                

class ResNetSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, wd = 1e-4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels,
                               kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        # SE
        self.se = SEBlock(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) :
        identity = self.shortcut(x)              # (B, out, L)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)                       # (B, out, L)
        out = out + identity
        return self.relu(out)

class CMIModel(nn.Module):
    def __init__(self, imu_dim, thm_dim, tof_dim, n_classes, **kwargs):
        super().__init__()
        self.imu_branch = nn.Sequential(
            self.residual_se_cnn_block(imu_dim, kwargs["imu1_channels"], kwargs["imu1_layers"],
                                       drop=kwargs["imu1_dropout"]),
            self.residual_se_cnn_block(kwargs["imu1_channels"], kwargs["feat_dim"], kwargs["imu2_layers"],
                                       drop=kwargs["imu2_dropout"])
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
        self.bert = BertModel(BertConfig(
            hidden_size=kwargs["feat_dim"],
            num_hidden_layers=kwargs["bert_layers"],
            num_attention_heads=kwargs["bert_heads"],
            intermediate_size=kwargs["feat_dim"]*4
        ))
        
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
            *[ResNetSEBlock(in_channels=in_channels, out_channels=in_channels) for i in range(num_layers)],
            ResNetSEBlock(in_channels, out_channels, wd=wd),
            nn.MaxPool1d(pool_size),
            nn.Dropout(drop)
        )
    
    def forward(self, imu, thm, tof):
        imu_feat = self.imu_branch(imu.permute(0, 2, 1))
        thm_feat = self.thm_branch(thm.permute(0, 2, 1))
        tof_feat = self.tof_branch(tof.permute(0, 2, 1))
        
        bert_input = torch.cat([imu_feat, thm_feat, tof_feat], dim=-1).permute(0, 2, 1)
        cls_token = self.cls_token.expand(bert_input.size(0), -1, -1)  # (B,1,H)
        bert_input = torch.cat([cls_token, bert_input], dim=1)  # (B,T+1,H)
        outputs = self.bert(inputs_embeds=bert_input)
        pred_cls = outputs.last_hidden_state[:, 0, :]

        return self.classifier(pred_cls)

CUDA0 = "cuda:0"
seed = 0
batch_size = 64
num_workers = 4
n_folds = 5

universe_csv_path = Path("/kaggle/input/cmi-precompute/pytorch/all/1/tof-1_raw.csv")

deterministic = kagglehub.package_import('wasupandceacar/deterministic').deterministic
deterministic.init_all(seed)

def init_dataset():
    dataset_config = {
        "percent": 95,
        "scaler_function": StandardScaler(),
        "nan_ratio": {
            "imu": 0,
            "thm": 0,
            "tof": 0,
        },
        "fbfill": {
            "imu": True,
            "thm": True,
            "tof": True,
        },
        "one_scale": True,
        "tof_raw": True,
        "tof_mode": 16,
        "save_precompute": False,
    }
    dataset = CMIFoldDataset(universe_csv_path, dataset_config,
                             n_folds=n_folds, random_seed=seed, full_dataset_function=CMIFeDataset)
    dataset.print_fold_stats()
    return dataset

def get_fold_dataset(dataset, fold):
    _, valid_dataset = dataset.get_fold_datasets(fold)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return valid_loader

dataset = init_dataset()

model_function = CMIModel
model_args = {"feat_dim": 500,
              "imu1_channels": 219, "imu1_dropout": 0.2946731587132302, "imu2_dropout": 0.2697745571929592,
              "imu1_weight_decay": 0.0014824054650601245, "imu2_weight_decay": 0.002742543773142381,
              "imu1_layers": 0, "imu2_layers": 0,
              "thm1_channels": 82, "thm1_dropout": 0.2641274454844602, "thm2_dropout": 0.302896343020985, 
              "tof1_channels": 82, "tof1_dropout": 0.2641274454844602, "tof2_dropout": 0.3028963430209852, 
              "bert_layers": 8, "bert_heads": 10,
              "cls1_channels": 937, "cls2_channels": 303, "cls1_dropout": 0.2281834512100508, "cls2_dropout": 0.22502521933558461}
model_args.update({
    "imu_dim": dataset.full_dataset.imu_dim, 
    "thm_dim": dataset.full_dataset.thm_dim,
    "tof_dim": dataset.full_dataset.tof_dim,
    "n_classes": dataset.full_dataset.class_num})
model_dir = Path("/kaggle/input/cmi-models-public/pytorch/train_fold_model05_tof16_raw/1")

model_dicts = [
    {
        "model_function": model_function,
        "model_args": model_args,
        "model_path": model_dir / f"fold{fold}/best_ema.pt",
    } for fold in range(n_folds)
]

models = list()
for model_dict in model_dicts:
    model_function = model_dict["model_function"]
    model_args = model_dict["model_args"]
    model_path = model_dict["model_path"]
    model = model_function(**model_args).to(CUDA0)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in torch.load(model_path).items()}
    model.load_state_dict(state_dict)
    model = model.eval()
    models.append(model)

metric_package = kagglehub.package_import('wasupandceacar/cmi-metric')

metric = metric_package.Metric()
imu_only_metric = metric_package.Metric()

def to_cuda(*tensors):
    return [tensor.to(CUDA0) for tensor in tensors]

def predict_valid(model, imu, thm, tof):
    pred = model(imu, thm, tof)
    return pred

def valid(model, valid_bar):
    with torch.no_grad():
        for imu, thm, tof, y in valid_bar:
            imu, thm, tof, y = to_cuda(imu, thm, tof, y)
            with autocast(device_type='cuda', dtype=torch.bfloat16): 
                logits = predict_valid(model, imu, thm, tof)
            metric.add(dataset.le.classes_[y.argmax(dim=1).cpu()], dataset.le.classes_[logits.argmax(dim=1).cpu()])
            _, thm, tof = dataset.full_dataset.get_scaled_nan_tensors(imu, thm, tof)
            with autocast(device_type='cuda', dtype=torch.bfloat16): 
                logits = model(imu, thm, tof)
            imu_only_metric.add(dataset.le.classes_[y.argmax(dim=1).cpu()], dataset.le.classes_[logits.argmax(dim=1).cpu()])

for fold, model in enumerate(models):
    valid_loader = get_fold_dataset(dataset, fold)
    valid_bar = tqdm(valid_loader, desc=f"Valid", position=0, leave=False)
    valid(model, valid_bar)

print(f"""
Normal score: {metric.score()}
IMU only score: {imu_only_metric.score()}
""")

def avg_predict(models, imu, thm, tof):
    outputs = []
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        for model in models:
            logits = model(imu, thm, tof)
        outputs.append(logits)
    return torch.mean(torch.stack(outputs), dim=0)

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    imu, thm, tof = dataset.full_dataset.inference_process(sequence)
    with torch.no_grad():
        imu, thm, tof = to_cuda(imu, thm, tof)
        logits = avg_predict(models, imu, thm, tof)
    return dataset.le.classes_[logits.argmax(dim=1).cpu()]

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

