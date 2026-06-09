!pip install /kaggle/input/cmi-2025-bfrb-prediction-pretrained-models/packages/scikit_learn-1.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl


from tensorflow.keras.layers import (
    Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda
)
from tensorflow.keras.utils import to_categorical, pad_sequences
from tensorflow.keras.saving import register_keras_serializable
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from scipy.spatial.transform import Rotation as R
from scipy.special import logit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import kaggle_evaluation.cmi_inference_server
import torch.nn.functional as F
from transformers import BertConfig, BertModel
from collections import defaultdict
import matplotlib.pyplot as plt
import torch.nn as nn
import tensorflow as tf
import seaborn as sns
import polars as pl
import pandas as pd
import numpy as np
import warnings
import torch
import joblib
import scipy
import glob
import gc
import os

warnings.filterwarnings("ignore")


class CFG:
    train_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
    train_metadata_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
    
    test_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
    test_metadata_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
    
    data_path = "/kaggle/input/cmi-2025-bfrb-prediction-pretrained-models/data"
    models_path = "/kaggle/input/cmi-2025-bfrb-prediction-pretrained-models/models"
    
    n_folds = 5
    seed = 42

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


num_to_label = {
    0: "Above ear - pull hair",
    1: "Cheek - pinch skin",
    2: "Eyebrow - pull hair",
    3: "Eyelash - pull hair",
    4: "Forehead - pull hairline",
    5: "Forehead - scratch",
    6: "Neck - pinch skin",
    7: "Neck - scratch",
    8: "Drink from bottle/cup",
    9: "Feel around in tray and pull out an object",
    10: "Glasses on/off",
    11: "Pinch knee/leg skin",
    12: "Pull air toward your face",
    13: "Scratch knee/leg skin",
    14: "Text on phone",
    15: "Wave hello",
    16: "Write name in air",
    17: "Write name on leg"
}

label_to_num = {v: k for k, v in num_to_label.items()}


class NNPreprocessorPT:
    def __init__(self, train_path=None, data_path=None):
        self.train_path = train_path
        self.data_path = data_path
        
        self.init_feature_names()

    def init_feature_names(self):
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

        self.imu_cols = [
            'rot_x', 'rot_y', 'rot_z', 'rot_w', 
            'acc_mag', 'acc_mag_jerk', 
            'rot_angle', 'rot_angle_vel', 
            'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag', 'linear_acc_mag_jerk', 
            'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance'
        ]
        self.imu_channel_keys = defaultdict(list)
        self.imu_channel_keys["acc"] = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z']
        self.imu_channel_keys["rot"] = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
        self.imu_channel_keys["other"] = ['acc_mag', 'acc_mag_jerk', 'linear_acc_mag', 'linear_acc_mag_jerk', 'rot_angle', 'rot_angle_vel', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance']

        self.thm_cols = [f"thm_{i}" for i in range(1, 6)]
        self.thm_channel_keys = {k: [f"thm_{k}"] for k in range(1, 6)}
        
        self.tof_cols = []
        self.tof_channel_keys = defaultdict(list)
        for i in range(1, 6):
            self.tof_cols.extend([f"tof_{i}_v{p}" for p in range(64)])
            self.tof_channel_keys[i].extend([f"tof_{i}_v{p}" for p in range(64)])
            
            for stat in ['mean', 'std', 'min', 'max']:
                self.tof_cols.append(f'tof_{i}_{stat}')
                self.tof_channel_keys[i].append(f'tof_{i}_{stat}')

            for r in range(16):
                for stat in ['mean', 'std', 'min', 'max']:
                    self.tof_cols.append(f'tof_16_{i}_region_{r}_{stat}')
                    self.tof_channel_keys[i].append(f'tof_16_{i}_region_{r}_{stat}')

        self.imu_dim = len(self.imu_cols)
        self.thm_dim = len(self.thm_cols)
        self.tof_dim = len(self.tof_cols)
        
        self.global_imu_indices = {k: sorted([self.imu_cols.index(feat) for feat in feats]) for k, feats in self.imu_channel_keys.items()}
        self.global_thm_indices = {k: sorted([self.thm_cols.index(key) for key in self.thm_channel_keys[k]]) for k in range(1, 6)}
        self.global_tof_indices = {k: sorted([self.tof_cols.index(key) for key in self.tof_channel_keys[k]]) for k in range(1, 6)}
        
    def remove_gravity_from_acc(self, acc_data, rot_data):
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

    def calculate_angular_velocity_from_quat(self, rot_data, time_delta=1/200):
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

    def calculate_angular_distance(self, rot_data):
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
        
    def generate_features(self, df, is_test=False):
        if not is_test:
            df['gesture_int'] = df["gesture"].map(label_to_num)
            
        self.targets = np.array([label_to_num[name] for name in self.target_gestures])
        self.non_targets = np.array([label_to_num[name] for name in self.non_target_gestures])
        
        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
        df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
        df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
            
        linear_accel_list = []
        for _, group in df.groupby('sequence_id'):
            acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            linear_accel_group = self.remove_gravity_from_acc(acc_data_group, rot_data_group)
            linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))
       
        df_linear_accel = pd.concat(linear_accel_list)
        df = pd.concat([df, df_linear_accel], axis=1)
        del linear_accel_list, df_linear_accel
        gc.collect()
       
        df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
        df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)
    
        angular_vel_list = []
        for _, group in df.groupby('sequence_id'):
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            angular_vel_group = self.calculate_angular_velocity_from_quat(rot_data_group)
            angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))
       
        df_angular_vel = pd.concat(angular_vel_list)
        df = pd.concat([df, df_angular_vel], axis=1)
        del angular_vel_list, df_angular_vel
        gc.collect()
        
        angular_distance_list = []
        for _, group in df.groupby('sequence_id'):
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            angular_dist_group = self.calculate_angular_distance(rot_data_group)
            angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))
        
        
        df_angular_distance = pd.concat(angular_distance_list)
        df = pd.concat([df, df_angular_distance], axis=1)
        del angular_distance_list, df_angular_distance
        gc.collect()

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
            region_size = 64 // 16
            for r in range(16):
                region_data = tof_data.iloc[:, r*region_size : (r+1)*region_size]
                new_columns.update({
                    f'tof_16_{i}_region_{r}_mean': region_data.mean(axis=1),
                    f'tof_16_{i}_region_{r}_std': region_data.std(axis=1),
                    f'tof_16_{i}_region_{r}_min': region_data.min(axis=1),
                    f'tof_16_{i}_region_{r}_max': region_data.max(axis=1)
                })

        return pd.concat([df, pd.DataFrame(new_columns)], axis=1)

    def scale(self, data_unscaled, is_test=False, scaler_path=None, scaler_prefix=""):
        if is_test:
            scaler = joblib.load(scaler_path)
        else:
            scaler = StandardScaler()
            scaler = scaler.fit(np.concatenate(data_unscaled, axis=0))
            joblib.dump(scaler, f"{self.data_path}/{scaler_prefix}_scaler.pkl")
            
        return [scaler.transform(x) for x in data_unscaled]
        

    def pad(self, data_scaled, cols, pad_len):
        pad_data = np.zeros((len(data_scaled), pad_len, len(cols)), dtype='float32')
        for i, seq in enumerate(data_scaled):
            seq_len = min(len(seq), pad_len)
            pad_data[i, :seq_len] = seq[:seq_len]
        return pad_data

    def process_train(self):
        if os.path.exists(f"{self.data_path}/X_imu.npy"):
            X_imu = np.load(f"{self.data_path}/X_imu.npy", mmap_mode='r')
            X_thm = np.load(f"{self.data_path}/X_thm.npy", mmap_mode='r')
            X_tof = np.load(f"{self.data_path}/X_tof.npy", mmap_mode='r')
            targets_int = np.load(f"{self.data_path}/targets_int.npy", mmap_mode='r')
            targets_ohe = np.load(f"{self.data_path}/targets_ohe.npy", mmap_mode='r')
            groups = np.load(f"{self.data_path}/groups.npy", mmap_mode='r')
            pad_len = joblib.load(f"{self.data_path}/pad_len.pkl")
            
            return X_imu, X_thm, X_tof, targets_int, targets_ohe, groups, pad_len
        
        os.makedirs(self.data_path, exist_ok=True)
        
        df = pd.read_csv(self.train_path)
        df = self.generate_features(df, is_test=False)
        
        targets_int, lens, groups = [], [], []
        imu_unscaled, thm_unscaled, tof_unscaled = [], [], []
        for i in range(2):
            for _, seq_df in df.groupby('sequence_id') :
                imu_data = seq_df[self.imu_cols]
                thm_data = seq_df[self.thm_cols]
                tof_data = seq_df[self.tof_cols]
                
                imu_data = imu_data.ffill().bfill().fillna(0)
                imu_unscaled.append(imu_data.values.astype('float32'))
                if i == 0:
                    thm_data = thm_data.ffill().bfill().fillna(0)
                    thm_unscaled.append(thm_data.values.astype('float32'))

                    tof_data = tof_data.ffill().bfill().fillna(0)
                    tof_unscaled.append(tof_data.values.astype('float32'))
                else:
                    thm_unscaled.append(np.zeros((len(thm_data), len(self.thm_cols))))
                    tof_unscaled.append(np.zeros((len(tof_data), len(self.tof_cols))))
                
                targets_int.append(seq_df['gesture_int'].iloc[0])
                groups.append(seq_df['subject'].iloc[0])
                lens.append(len(imu_data))
            
        pad_len = int(np.percentile(lens, 99))

        imu_scaled = self.scale(imu_unscaled, False, None, "imu")
        thm_scaled = self.scale(thm_unscaled, False, None, "thm")
        tof_scaled = self.scale(tof_unscaled, False, None, "tof")
        
        X_imu = self.pad(imu_scaled, self.imu_cols, pad_len)
        X_thm = self.pad(thm_scaled, self.thm_cols, pad_len)
        X_tof = self.pad(tof_scaled, self.tof_cols, pad_len)

        targets_ohe = F.one_hot(torch.from_numpy(np.array(targets_int)).long(), num_classes=18).float().numpy()
        groups = np.array(groups)
        targets_int = np.array(targets_int)
        targets_ohe = np.array(targets_ohe)
        
        np.save(f"{self.data_path}/X_imu.npy", X_imu)
        np.save(f"{self.data_path}/X_thm.npy", X_thm)
        np.save(f"{self.data_path}/X_tof.npy", X_tof)
        np.save(f"{self.data_path}/targets_int.npy", targets_int)
        np.save(f"{self.data_path}/targets_ohe.npy", targets_ohe)
        np.save(f"{self.data_path}/groups.npy", groups)
        joblib.dump(pad_len, f"{self.data_path}/pad_len.pkl")
        
        return X_imu, X_thm, X_tof, targets_int, targets_ohe, groups, pad_len
    

    def process_test(self, sequence, pad_len, scaler_paths):
        df = sequence.to_pandas()
        df = self.generate_features(df, is_test=True)
        
        imu_unscaled, thm_unscaled, tof_unscaled = [], [], []
        for _, seq_df in df.groupby('sequence_id'):
            imu_data = seq_df[self.imu_cols]
            imu_data = imu_data.ffill().bfill().fillna(0)
            imu_unscaled.append(imu_data.values.astype('float32'))
            
            thm_data = seq_df[self.thm_cols]
            thm_data = thm_data.ffill().bfill().fillna(0)
            thm_unscaled.append(thm_data.values.astype('float32'))
            
            tof_data = seq_df[self.tof_cols]
            tof_data = tof_data.ffill().bfill().fillna(0)
            tof_unscaled.append(tof_data.values.astype('float32'))
            
        imu_scaled = self.scale(imu_unscaled, True, scaler_paths["imu"])
        thm_scaled = self.scale(thm_unscaled, True, scaler_paths["thm"])
        tof_scaled = self.scale(tof_unscaled, True, scaler_paths["tof"])
        
        X_imu = self.pad(imu_scaled, self.imu_cols, pad_len)
        X_thm = self.pad(thm_scaled, self.thm_cols, pad_len)
        X_tof = self.pad(tof_scaled, self.tof_cols, pad_len)
        
        return X_imu, X_thm, X_tof


class NNPreprocessorTF:
    def __init__(self, train_path=None, data_path=None):
        self.train_path = train_path
        self.data_path = data_path
        
        self.init_feature_names()
        
    def init_feature_names(self):
        self.imu_cols = [
            'acc_x', 'acc_y', 'acc_z', 
            'rot_x', 'rot_y', 'rot_z', 'rot_w', 
            'acc_mag', 'acc_mag_jerk', 
            'rot_angle', 'rot_angle_vel', 
            'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag', 'linear_acc_mag_jerk', 
            'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance'
        ]
        
        base_thm_cols = [f'thm_{i}' for i in range(1, 6)]
        self.advanced_thm_cols = [
            'thm_gradient_1_2',   
        ]
        
        self.thm_cols = base_thm_cols + self.advanced_thm_cols

        base_tof_cols = []
        for i in range(1, 6):
            base_tof_cols.extend([f"tof_{i}_v{p}" for p in range(64)])
            
            for stat in ['mean', 'std', 'min', 'max']:
                base_tof_cols.append(f'tof_{i}_{stat}')
                
            for r in range(16):
                for stat in ['mean', 'std', 'min', 'max']:
                    base_tof_cols.append(f'tof_16_{i}_region_{r}_{stat}')
        
        self.advanced_tof_cols = [
            'tof_3_center_mass_x',
            'tof_5_spatial_gradient', 
            'tof_4_center_mass_x', 
        ]
        
        self.tof_cols = base_tof_cols + self.advanced_tof_cols
        
        self.feature_cols = self.imu_cols + self.thm_cols + self.tof_cols
        
        self.imu_dim = len(self.imu_cols)
        self.thm_dim = len(self.thm_cols)
        self.tof_dim = len(self.tof_cols)
        
    def remove_gravity_from_acc(self, acc_data, rot_data):
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

    def calculate_angular_velocity_from_quat(self, rot_data, time_delta=1/200):
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

    def calculate_angular_distance(self, rot_data):
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

    def calculate_thermal_features(self, df):
        thm_gradient_1_2 = np.abs(df['thm_1'].fillna(0) - df['thm_2'].fillna(0))
        thm_gradient_2_3 = np.abs(df['thm_2'].fillna(0) - df['thm_3'].fillna(0))
        thm_gradient_3_4 = np.abs(df['thm_3'].fillna(0) - df['thm_4'].fillna(0))
        thm_gradient_4_5 = np.abs(df['thm_4'].fillna(0) - df['thm_5'].fillna(0))
        
        thm_cols = ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5']
        thm_change_rate_mean = df[thm_cols].diff().abs().mean(axis=1).fillna(0)
        
        thm_mean = df[thm_cols].mean(axis=1)
        thm_std = df[thm_cols].std(axis=1)
        thm_distribution_pattern = (thm_std / (thm_mean + 1e-6)).fillna(0)
        
        return (thm_gradient_1_2, thm_gradient_2_3, thm_gradient_3_4, thm_gradient_4_5,
                thm_change_rate_mean, thm_distribution_pattern)

    def calculate_tof_spatial_features(self, df):
        spatial_features = {}
        
        for sensor_id in range(1, 6):
            tof_pixels = df[[f"tof_{sensor_id}_v{p}" for p in range(64)]].fillna(0)
            
            spatial_gradients = []
            center_mass_x_list = []
            center_mass_y_list = []
            
            for idx, row in tof_pixels.iterrows():
                grid = row.values.reshape(8, 8)
                
                grad_x = np.gradient(grid, axis=1)
                grad_y = np.gradient(grid, axis=0)
                gradient_mag = np.sqrt(grad_x**2 + grad_y**2).mean()
                spatial_gradients.append(gradient_mag)
                
                if np.sum(grid) > 0:
                    y_coords, x_coords = np.mgrid[0:8, 0:8]
                    center_x = np.sum(x_coords * grid) / np.sum(grid)
                    center_y = np.sum(y_coords * grid) / np.sum(grid)
                else:
                    center_x, center_y = 4.0, 4.0
                    
                center_mass_x_list.append(center_x)
                center_mass_y_list.append(center_y)
            
            spatial_features[f'tof_{sensor_id}_spatial_gradient'] = pd.Series(spatial_gradients, index=df.index)
            spatial_features[f'tof_{sensor_id}_center_mass_x'] = pd.Series(center_mass_x_list, index=df.index)
            spatial_features[f'tof_{sensor_id}_center_mass_y'] = pd.Series(center_mass_y_list, index=df.index)
        
        return spatial_features
    
    def generate_features(self, df, is_test=False):     
        if not is_test:
            df['gesture_int'] = df['gesture'].map(label_to_num)

        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
        df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
        df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

        linear_accel_list = []
        for _, group in df.groupby('sequence_id'):
            acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            linear_accel_group = self.remove_gravity_from_acc(acc_data_group, rot_data_group)
            linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))

        df_linear_accel = pd.concat(linear_accel_list)
        df = pd.concat([df, df_linear_accel], axis=1)
        del df_linear_accel, linear_accel_list
        gc.collect()
        
        df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
        df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)

        angular_vel_list = []
        for _, group in df.groupby('sequence_id'):
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            angular_vel_group = self.calculate_angular_velocity_from_quat(rot_data_group)
            angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))
        
        df_angular_vel = pd.concat(angular_vel_list)
        df = pd.concat([df, df_angular_vel], axis=1)
        del angular_vel_list, df_angular_vel
        gc.collect()

        angular_distance_list = []
        for _, group in df.groupby('sequence_id'):
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            angular_dist_group = self.calculate_angular_distance(rot_data_group)
            angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))
            
        df_angular_distance = pd.concat(angular_distance_list)
        df = pd.concat([df, df_angular_distance], axis=1)
        del angular_distance_list, df_angular_distance
        gc.collect()

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

            region_size = 64 // 16
            for r in range(16):
                region_data = tof_data.iloc[:, r*region_size : (r+1)*region_size]
                new_columns.update({
                    f'tof_16_{i}_region_{r}_mean': region_data.mean(axis=1),
                    f'tof_16_{i}_region_{r}_std': region_data.std(axis=1),
                    f'tof_16_{i}_region_{r}_min': region_data.min(axis=1),
                    f'tof_16_{i}_region_{r}_max': region_data.max(axis=1)
                })
                del region_data

            del tof_data
        
        df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)
        del new_columns
        gc.collect()

        advanced_features = {}
        
        for seq_id, group in df.groupby('sequence_id'):                        
            (thm_gradient_1_2, thm_gradient_2_3, thm_gradient_3_4, thm_gradient_4_5,
             thm_change_rate_mean, thm_distribution_pattern) = self.calculate_thermal_features(group)
            advanced_features.update({
                ('thm_gradient_1_2', seq_id): thm_gradient_1_2,
                ('thm_gradient_2_3', seq_id): thm_gradient_2_3,
                ('thm_gradient_3_4', seq_id): thm_gradient_3_4,
                ('thm_gradient_4_5', seq_id): thm_gradient_4_5,
                ('thm_change_rate_mean', seq_id): thm_change_rate_mean,
                ('thm_distribution_pattern', seq_id): thm_distribution_pattern
            })
            
            tof_spatial_features = self.calculate_tof_spatial_features(group)
            for feature_name, feature_data in tof_spatial_features.items():
                advanced_features[(feature_name, seq_id)] = feature_data
        
        thm_feature_names = self.advanced_thm_cols
        tof_feature_names = self.advanced_tof_cols
        
        for feature_name in thm_feature_names:
            feature_data = []
            for seq_id in df['sequence_id'].unique():
                feature_data.append(advanced_features[(feature_name, seq_id)])
            df[feature_name] = pd.concat(feature_data)
        
        for feature_name in tof_feature_names:
            feature_data = []
            for seq_id in df['sequence_id'].unique():
                feature_data.append(advanced_features[(feature_name, seq_id)])
            df[feature_name] = pd.concat(feature_data)
        
        return df
    
    def scale(self, data_unscaled, is_test=False, scaler_path=None):
        if is_test:
            scaler = joblib.load(scaler_path)
        else:
            scaler = StandardScaler()
            scaler = scaler.fit(np.concatenate(data_unscaled, axis=0))
            joblib.dump(scaler, f"{self.data_path}/scaler.pkl")
        
        return [scaler.transform(x) for x in data_unscaled]

    def pad(self, data_scaled, cols, pad_len):
        pad_data = np.zeros((len(data_scaled), pad_len, len(cols)), dtype='float32')
        for i, seq in enumerate(data_scaled):
            seq_len = min(len(seq), pad_len)
            pad_data[i, :seq_len] = seq[:seq_len]
        return pad_data  
    
    def process_train(self):
        if os.path.exists(f"{self.data_path}/X_imu.npy"):
            X_imu = np.load(f"{self.data_path}/X_imu.npy", mmap_mode='r')
            X_thm = np.load(f"{self.data_path}/X_thm.npy", mmap_mode='r')
            X_tof = np.load(f"{self.data_path}/X_tof.npy", mmap_mode='r')
            targets_int = np.load(f"{self.data_path}/targets_int.npy", mmap_mode='r')
            targets_ohe = np.load(f"{self.data_path}/targets_ohe.npy", mmap_mode='r')
            groups = np.load(f"{self.data_path}/groups.npy", mmap_mode='r')
            pad_len = joblib.load(f"{self.data_path}/pad_len.pkl")
            
            return X_imu, X_thm, X_tof, targets_int, targets_ohe, groups, pad_len

        os.makedirs(self.data_path, exist_ok=True)

        df = pd.read_csv(CFG.train_path)
        df = self.generate_features(df, is_test=False)

        targets_int, lens, groups = [], [], []
        imu_unscaled, thm_unscaled, tof_unscaled = [], [], []
        for i in range(2):
            for _, seq_df in df.groupby('sequence_id'):
                imu_data = seq_df[self.imu_cols]
                thm_data = seq_df[self.thm_cols]
                tof_data = seq_df[self.tof_cols]

                imu_data = imu_data.ffill().bfill().fillna(0)
                imu_unscaled.append(imu_data.values.astype('float32'))
                if i == 0:
                    thm_data = thm_data.ffill().bfill().fillna(0)
                    thm_unscaled.append(thm_data.values.astype('float32'))

                    tof_data = tof_data.ffill().bfill().fillna(0)
                    tof_unscaled.append(tof_data.values.astype('float32'))
                else:
                    thm_unscaled.append(np.zeros((len(thm_data), len(self.thm_cols))))
                    tof_unscaled.append(np.zeros((len(tof_data), len(self.tof_cols))))
                
                targets_int.append(seq_df['gesture_int'].iloc[0])
                groups.append(seq_df['subject'].iloc[0])
                lens.append(len(imu_data))

        pad_len = int(np.percentile(lens, 99))
        X_unscaled = [np.concatenate([imu, thm, tof], axis=1) for imu, thm, tof in zip(imu_unscaled, thm_unscaled, tof_unscaled)]
        X_scaled = self.scale(X_unscaled, is_test=False)
        X_padded = self.pad(X_scaled, self.imu_cols+self.thm_cols+self.tof_cols, pad_len)
        
        X_imu = X_padded[..., :self.imu_dim]
        X_thm = X_padded[..., self.imu_dim:self.imu_dim+self.thm_dim]
        X_tof = X_padded[..., self.imu_dim+self.thm_dim:self.imu_dim+self.thm_dim+self.tof_dim]

        targets_ohe = F.one_hot(torch.from_numpy(np.array(targets_int)).long(), num_classes=18).float().numpy()
        groups = np.array(groups)
        targets_int = np.array(targets_int)
        targets_ohe = np.array(targets_ohe)

        np.save(f"{self.data_path}/X_imu.npy", X_imu)
        np.save(f"{self.data_path}/X_thm.npy", X_thm)
        np.save(f"{self.data_path}/X_tof.npy", X_tof)
        np.save(f"{self.data_path}/targets_int.npy", targets_int)
        np.save(f"{self.data_path}/targets_ohe.npy", targets_ohe)
        np.save(f"{self.data_path}/groups.npy", groups)
        joblib.dump(pad_len, f"{self.data_path}/pad_len.pkl")
        
        return X_imu, X_thm, X_tof, targets_int, targets_ohe, groups, pad_len
    
    def process_test(self, sequence, pad_len, scaler_path):
        df = sequence.to_pandas()
        df = self.generate_features(df, is_test=True)
        
        imu_unscaled, thm_unscaled, tof_unscaled = [], [], []
        for _, seq_df in df.groupby('sequence_id'):
            imu_data = seq_df[self.imu_cols]
            imu_data = imu_data.ffill().bfill().fillna(0)
            imu_unscaled.append(imu_data.values.astype('float32'))

            thm_data = seq_df[self.thm_cols]
            thm_data = thm_data.ffill().bfill().fillna(0)
            thm_unscaled.append(thm_data.values.astype('float32'))
            
            tof_data = seq_df[self.tof_cols]
            tof_data = tof_data.ffill().bfill().fillna(0)
            tof_unscaled.append(tof_data.values.astype('float32'))
            
        X_unscaled = [np.concatenate([imu, thm, tof], axis=1) for imu, thm, tof in zip(imu_unscaled, thm_unscaled, tof_unscaled)]
        X_scaled = self.scale(X_unscaled, is_test=True, scaler_path=scaler_path)
        X_padded = self.pad(X_scaled, self.imu_cols+self.thm_cols+self.tof_cols, pad_len)

        X_imu = X_padded[..., :self.imu_dim]
        X_thm = X_padded[..., self.imu_dim:self.imu_dim+self.thm_dim]
        X_tof = X_padded[..., self.imu_dim+self.thm_dim:self.imu_dim+self.thm_dim+self.tof_dim]
        
        return X_imu, X_thm, X_tof


class GBMPreprocessor:
    def remove_gravity_from_acc(self, acc_data, rot_data):
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

    def calculate_angular_velocity_from_quat(self, rot_data, time_delta=1/200):
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

    def calculate_angular_distance(self, rot_data):
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

    def add_linear_acc_to_dataset(self, dataset):
        dataset = dataset.to_pandas()
        
        linear_accel_list = []
        for _, group in dataset.groupby('sequence_id'):
            acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            linear_accel_group = self.remove_gravity_from_acc(acc_data_group, rot_data_group)
            linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))    
        
        linear_accel_df = pd.concat(linear_accel_list)
        dataset = pd.concat([dataset, linear_accel_df], axis=1)
        
        return  pl.from_pandas(dataset)
        
    def add_angular_vel_to_dataset(self, dataset):
        dataset = dataset.to_pandas()

        angular_vel_list = []
        for _, group in dataset.groupby('sequence_id'):
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            angular_vel_group = self.calculate_angular_velocity_from_quat(rot_data_group)
            angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))
        
        angular_vel_df = pd.concat(angular_vel_list)
        dataset = pd.concat([dataset, angular_vel_df], axis=1)
        
        return  pl.from_pandas(dataset)

    def add_angular_distance_to_dataset(self, dataset):
        dataset = dataset.to_pandas()
        
        angular_distance_list = []
        for _, group in dataset.groupby('sequence_id'):
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            angular_dist_group = self.calculate_angular_distance(rot_data_group)
            angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))
        
        angular_distance_df = pd.concat(angular_distance_list)
        dataset = pd.concat([dataset, angular_distance_df], axis=1)
        
        return pl.from_pandas(dataset)

    def get_imu_data(self, dataset):
        imu_cols = [col for col in dataset.columns if 'acc_' in col or 'rot_' in col]
        
        dataset = self.add_linear_acc_to_dataset(dataset)
        dataset = self.add_angular_vel_to_dataset(dataset)
        dataset = self.add_angular_distance_to_dataset(dataset)
        
        imu_cols.extend([
            'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 
            'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 
            'angular_distance'
        ])
        
        imu_aggs = []    
        for col in imu_cols:        
            imu_aggs.extend([
                pl.mean(col).alias(f'{col}_mean'),
                pl.std(col).alias(f'{col}_std'),
                pl.max(col).alias(f'{col}_max'),
                pl.min(col).alias(f'{col}_min'),
                pl.quantile(col, 0.1).alias(f'{col}_q10'),
                pl.quantile(col, 0.9).alias(f'{col}_q90'),
                pl.median(col).alias(f'{col}_median'),
                (pl.col(col).diff().fill_null(0)).std().alias(f'{col}_diff_std'),
                (pl.col(col).diff().fill_null(0)).max().alias(f'{col}_diff_max'),
                (pl.col(col).diff().fill_null(0)).min().alias(f'{col}_diff_min'),
                (pl.col(col).diff(2).fill_null(0)).mean().alias(f'{col}_diff2_mean'),
                (pl.col(col).skew()).alias(f'{col}_skew'),
                (pl.col(col).kurtosis()).alias(f'{col}_kurtosis'),
                (pl.col(col) - pl.col(col).shift(1)).abs().mean().alias(f'{col}_mad'),
                (pl.col(col).max() - pl.col(col).min()).alias(f'{col}_range'),
                (pl.col(col).quantile(0.75) - pl.col(col).quantile(0.25)).alias(f'{col}_iqr'),
                pl.col(col).rolling_mean(window_size=5).std().alias(f'{col}_rolling5_std'),
                pl.col(col).rolling_mean(window_size=10).std().alias(f'{col}_rolling10_std'),
                pl.col(col).rolling_max(window_size=5).mean().alias(f'{col}_rolling5_max_mean'),
                pl.col(col).rolling_min(window_size=5).mean().alias(f'{col}_rolling5_min_mean'),
                (pl.col(col) > 0).sum().alias(f'{col}_positive_count'),
                (pl.col(col).abs() > pl.col(col).abs().mean()).sum().alias(f'{col}_above_mean_count'),
                ((pl.col(col) > 0) & (pl.col(col).shift(1) <= 0) | (pl.col(col) <= 0) & (pl.col(col).shift(1) > 0)).sum().alias(f'{col}_zero_crossings'),
                (pl.col(col).rolling_max(window_size=3) == pl.col(col)).sum().alias(f'{col}_local_maxima_count'),
                (pl.col(col).rolling_min(window_size=3) == pl.col(col)).sum().alias(f'{col}_local_minima_count'),
                (pl.col(col).abs() > pl.col(col).abs().std()).sum().alias(f'{col}_above_std_count'),
                (pl.col(col) - pl.col(col).median()).abs().median().alias(f'{col}_mad_median'),
                pl.col(col).quantile(0.95).alias(f'{col}_q95'),
                pl.col(col).quantile(0.05).alias(f'{col}_q05'),
                (pl.col(col).shift(1) * pl.col(col)).mean().alias(f'{col}_autocorr_lag1'),
                (pl.col(col).shift(2) * pl.col(col)).mean().alias(f'{col}_autocorr_lag2'),
                (pl.col(col).shift(3) * pl.col(col)).mean().alias(f'{col}_autocorr_lag3'),
                (pl.col(col).shift(5) * pl.col(col)).mean().alias(f'{col}_autocorr_lag5'),
                (pl.col(col).ewm_mean(alpha=0.1)).std().alias(f'{col}_trend_std'),
                (pl.col(col) - pl.col(col).ewm_mean(alpha=0.1)).abs().mean().alias(f'{col}_detrend_mad'),
                ((pl.col(col) - pl.col(col).shift(1)).abs() > pl.col(col).std()).sum().alias(f'{col}_spike_count'),
                ((pl.col(col) > pl.col(col).mean()) & 
                (pl.col(col).shift(1) <= pl.col(col).mean()) |
                (pl.col(col) <= pl.col(col).mean()) & 
                (pl.col(col).shift(1) > pl.col(col).mean())).sum().alias(f'{col}_mean_crossings'),
                (pl.col(col).diff().fill_null(0).abs()).max().alias(f'{col}_max_velocity'),
            ])
            
            if col != 'rot_w':
                imu_aggs.extend([
                    (pl.col(col).abs()).max().alias(f'{col}_abs_max'),
                    (pl.col(col) < 0).sum().alias(f'{col}_negative_count'),
                ])

        
        magnitude_expr = (pl.col('acc_x')**2 + pl.col('acc_y')**2 + pl.col('acc_z')**2).sqrt()
        imu_aggs.extend([
            magnitude_expr.mean().alias('acc_magnitude_mean'),
            magnitude_expr.max().alias('acc_magnitude_max'),
            magnitude_expr.quantile(0.25).alias('acc_magnitude_q25'),
            magnitude_expr.diff().fill_null(0).std().alias('acc_magnitude_diff_std'),
            magnitude_expr.diff().fill_null(0).abs().mean().alias('acc_magnitude_jerk'),
            
            (pl.col('acc_x') * pl.col('acc_y')).mean().alias('acc_xy_correlation'),
            (pl.col('acc_x') * pl.col('acc_z')).mean().alias('acc_xz_correlation'),
            (pl.col('acc_y') * pl.col('acc_z')).mean().alias('acc_yz_correlation'),
            
            (pl.col('acc_x')**2 / (magnitude_expr**2 + 0.001)).mean().alias('acc_x_energy_ratio'),
            (pl.col('acc_z')**2 / (magnitude_expr**2 + 0.001)).mean().alias('acc_z_energy_ratio'),
            
            (magnitude_expr - magnitude_expr.rolling_mean(window_size=10)).abs().mean().alias('acc_dynamic_component'),
            magnitude_expr.rolling_mean(window_size=10).std().alias('acc_gravity_variation'),
            
            (pl.col('acc_y').abs() > pl.col('acc_z').abs()).mean().alias('acc_y_dominance_ratio'),
            (pl.col('acc_z').abs() > pl.col('acc_x').abs()).mean().alias('acc_z_dominance_ratio'),
            
            (pl.col('acc_x').abs().mean() / (pl.col('acc_y').abs().mean() + 0.001)).alias('acc_x_y_ratio'),
            (pl.col('acc_x').abs().mean() / (pl.col('acc_z').abs().mean() + 0.001)).alias('acc_x_z_ratio'),
            (pl.col('acc_y').abs().mean() / (pl.col('acc_z').abs().mean() + 0.001)).alias('acc_y_z_ratio'),
            
            (pl.col('acc_z') / (magnitude_expr + 0.001)).mean().alias('acc_z_direction_mean'),
            
            (pl.col('acc_x').var() / (pl.col('acc_x').var() + pl.col('acc_y').var() + pl.col('acc_z').var() + 0.001)).alias('acc_x_variance_ratio'),
            (pl.col('acc_y').var() / (pl.col('acc_x').var() + pl.col('acc_y').var() + pl.col('acc_z').var() + 0.001)).alias('acc_y_variance_ratio'),
            (pl.col('acc_z').var() / (pl.col('acc_x').var() + pl.col('acc_y').var() + pl.col('acc_z').var() + 0.001)).alias('acc_z_variance_ratio'),
        ])
        
        linear_acc_magnitude_expr = (pl.col('linear_acc_x')**2 + pl.col('linear_acc_y')**2 + pl.col('linear_acc_z')**2).sqrt()
        imu_aggs.extend([
            linear_acc_magnitude_expr.diff().fill_null(0).std().alias('linear_acc_magnitude_diff_std'),
            
            (pl.col('linear_acc_x') * pl.col('linear_acc_z')).mean().alias('linear_acc_xz_correlation'),
            (pl.col('linear_acc_y') * pl.col('linear_acc_z')).mean().alias('linear_acc_yz_correlation'),
            
            (pl.col('linear_acc_x')**2 / (linear_acc_magnitude_expr**2 + 0.001)).mean().alias('linear_acc_x_energy_ratio'),
            (pl.col('linear_acc_z')**2 / (linear_acc_magnitude_expr**2 + 0.001)).mean().alias('linear_acc_z_energy_ratio'),
            
            (pl.col('linear_acc_y').abs() > pl.col('linear_acc_z').abs()).mean().alias('linear_acc_y_dominance_ratio'),
            (pl.col('linear_acc_z').abs() > pl.col('linear_acc_x').abs()).mean().alias('linear_acc_z_dominance_ratio'),
            
            (pl.col('linear_acc_x').abs().mean() / (pl.col('linear_acc_y').abs().mean() + 0.001)).alias('linear_acc_x_y_ratio'),
            (pl.col('linear_acc_x').abs().mean() / (pl.col('linear_acc_z').abs().mean() + 0.001)).alias('linear_acc_x_z_ratio'),
            (pl.col('linear_acc_y').abs().mean() / (pl.col('linear_acc_z').abs().mean() + 0.001)).alias('linear_acc_y_z_ratio'),
            
            (pl.col('linear_acc_z') / (linear_acc_magnitude_expr + 0.001)).mean().alias('linear_acc_z_direction_mean'),
            
            (pl.col('linear_acc_z').var() / (pl.col('linear_acc_x').var() + pl.col('linear_acc_y').var() + pl.col('linear_acc_z').var() + 0.001)).alias('linear_acc_z_variance_ratio'),
        ])
        
        quat_magnitude = (pl.col('rot_w')**2 + pl.col('rot_x')**2 + pl.col('rot_y')**2 + pl.col('rot_z')**2).sqrt()    
        imu_aggs.extend([        
            quat_magnitude.mean().alias('rot_magnitude_mean'),
            quat_magnitude.diff().fill_null(0).std().alias('rot_magnitude_diff_std'),
            
            (2 * (pl.col('rot_w') * pl.col('rot_z') + pl.col('rot_x') * pl.col('rot_y'))).mean().alias('rot_euler_z_mean'),
            (2 * (pl.col('rot_w') * pl.col('rot_x') + pl.col('rot_y') * pl.col('rot_z'))).std().alias('rot_euler_x_std'),
            (2 * (pl.col('rot_w') * pl.col('rot_y') - pl.col('rot_z') * pl.col('rot_x'))).std().alias('rot_euler_y_std'),
            (2 * (pl.col('rot_w') * pl.col('rot_z') + pl.col('rot_x') * pl.col('rot_y'))).std().alias('rot_euler_z_std'),
            
            (pl.col('rot_w') * pl.col('rot_x')).pow(2).mean().alias('rot_wx_interaction'),
            (pl.col('rot_w') * pl.col('rot_z')).pow(2).mean().alias('rot_wz_interaction'),
            
            (pl.col('rot_y')**2 / (pl.col('rot_x')**2 + pl.col('rot_y')**2 + pl.col('rot_z')**2 + 0.001)).mean().alias('rot_y_axis_dominance'),
        ])

        part_configs = [
            ('first_30pct', pl.col('sequence_counter') < pl.max('sequence_counter') * 0.3),
            ('middle_40pct', (pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.3) & (pl.col('sequence_counter') < pl.max('sequence_counter') * 0.7)),
            ('last_30pct', pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.7),
            ('first_half', pl.col('sequence_counter') < pl.max('sequence_counter') * 0.5),
            ('last_half', pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.5),
            ('first_10pct', pl.col('sequence_counter') < pl.max('sequence_counter') * 0.1),
            ('last_10pct', pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.9),
        ]
        for part_name, part_expr in part_configs:
            for col in imu_cols:
                is_rot_w = 'rot_w' in col
                imu_aggs.extend([
                    (pl.when(part_expr).then(pl.col(col))).mean().alias(f'{col}_mean_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).std().alias(f'{col}_std_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).max().alias(f'{col}_max_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).min().alias(f'{col}_min_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).var().alias(f'{col}_var_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).quantile(0.05).alias(f'{col}_q05_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).quantile(0.10).alias(f'{col}_q10_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).quantile(0.25).alias(f'{col}_q25_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).quantile(0.75).alias(f'{col}_q75_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).quantile(0.90).alias(f'{col}_q90_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).entropy().alias(f'{col}_entropy_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).median().alias(f'{col}_median_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).kurtosis().alias(f'{col}_kurtosis_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).skew().alias(f'{col}_skew_{part_name}'),
                ])
                
                if not is_rot_w:
                    imu_aggs.extend([
                        (pl.when(part_expr).then(pl.col(col).abs())).mean().alias(f'{col}_abs_mean_{part_name}'),
                    ])

        for col in imu_cols:
            imu_aggs.extend([
                ((pl.when(pl.col('sequence_counter') < pl.max('sequence_counter') * 0.2).then(pl.col(col))).mean() -
                (pl.when(pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.8).then(pl.col(col))).mean()).alias(f'{col}_early_late_diff'),
            ])
            
        imu_data = dataset.group_by('sequence_id', maintain_order=True).agg(imu_aggs).fill_nan(None).fill_null(strategy="forward").fill_null(strategy="forward").fill_null(0)
        
        return imu_data

    def get_thm_data(self, dataset):
        thm_cols = [col for col in dataset.columns if 'thm_' in col]
        
        thm_aggs = []    
        part_configs = [
            ('middle_40pct', (pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.3) & (pl.col('sequence_counter') < pl.max('sequence_counter') * 0.7)),
            ('last_30pct', pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.7),
            ('last_half', pl.col('sequence_counter') >= pl.max('sequence_counter') * 0.5)
        ]
        for part_name, part_expr in part_configs:
            for col in thm_cols:
                thm_aggs.extend([
                    (pl.when(part_expr).then(pl.col(col))).max().alias(f'{col}_max_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).quantile(0.90).alias(f'{col}_q90_{part_name}'),
                    (pl.when(part_expr).then(pl.col(col))).median().alias(f'{col}_median_{part_name}'),
                ])

        thm_data = dataset.group_by('sequence_id', maintain_order=True).agg(thm_aggs).fill_nan(None).fill_null(strategy="forward").fill_null(strategy="forward").fill_null(0)                 
        
        return thm_data

    def get_tof_data(self, dataset):
        dataset = dataset.to_pandas()
        
        features = {
            "sequence_id": []          
        }
        for i in [2, 3, 4]:
            features[f"tof_{i}_median"] = []
            features[f"tof_{i}_q25"] = []
            features[f"tof_{i}_cv"] = []
            features[f"tof_{i}_skew"] = []
            features[f"tof_{i}_depth_grad"] = []
            features[f"tof_{i}_depth_smoothness"] = []
            
        for seq_id, seq in dataset.groupby('sequence_id'):
            seq_df = seq.copy()
            features["sequence_id"].append(seq_id)
            for i in [2, 3, 4]:
                pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
                tof_sensor_data = seq_df[pixel_cols_tof].replace(-1, np.nan).values.flatten()
                tof_sensor_data = tof_sensor_data[~np.isnan(tof_sensor_data)]       
                features[f"tof_{i}_median"].append(np.median(tof_sensor_data) if len(tof_sensor_data) > 0 else np.nan)
                features[f"tof_{i}_q25"].append(np.percentile(tof_sensor_data, 25) if len(tof_sensor_data) > 0 else np.nan)
                features[f"tof_{i}_cv"].append(tof_sensor_data.std() / tof_sensor_data.mean() if len(tof_sensor_data) > 0 and tof_sensor_data.mean() != 0 else np.nan)
                features[f"tof_{i}_skew"].append(scipy.stats.skew(tof_sensor_data) if len(tof_sensor_data) > 0 else np.nan)
                features[f"tof_{i}_depth_grad"].append(np.mean(np.abs(np.gradient(tof_sensor_data))) if len(tof_sensor_data) > 1 else np.nan)
                features[f"tof_{i}_depth_smoothness"].append(np.mean(np.abs(np.diff(tof_sensor_data, 2))) if len(tof_sensor_data) > 2 else np.nan)
                
        return pl.DataFrame(features).fill_nan(None).fill_null(strategy="forward").fill_null(strategy="forward").fill_null(0)
        
    def process_train(self):
        self.features = joblib.load(f"{CFG.data_path}/tabular-models/features.pkl")
        
        if os.path.exists(f"{CFG.data_path}/tabular-models"):
            X = joblib.load(f"{CFG.data_path}/tabular-models/X.pkl")[self.features]
            y = joblib.load(f"{CFG.data_path}/tabular-models/y.pkl")
            groups = joblib.load(f"{CFG.data_path}/tabular-models/groups.pkl")
            return X, y, groups

        def _process(dataset):                 
            imu_data = self.get_imu_data(dataset)
            thm_data = self.get_thm_data(dataset)    
            tof_data = self.get_tof_data(dataset)
            data = imu_data.join(thm_data, on="sequence_id").join(tof_data, on="sequence_id")
        
            targets = dataset.group_by('sequence_id', maintain_order=True).agg([
                pl.col("gesture").first(),
                pl.col("subject").first(),
                pl.col("sequence_type").first()
            ])
            
            data = data.join(targets, on="sequence_id", how="left")
            
            X = data.drop(['sequence_id', 'gesture', 'subject', 'sequence_type'])
            y = data.select(pl.col('gesture').map_elements(lambda x: label_to_num[x], return_dtype=pl.Int32))
            groups = data.select('subject')
            
            del dataset, targets
            gc.collect()
            
            return X.to_pandas(), y.to_pandas().values.flatten(), groups.to_pandas().values.flatten()

        train = pl.read_csv(CFG.train_path)
        train_metadata = pl.read_csv(CFG.train_metadata_path)
        X_1, y_1, groups_1 = _process(train)
        
        thm_tof_cols = [c for c in train.columns if 'thm_' in c or 'tof_' in c]
        train = train.with_columns([pl.lit(np.nan).alias(c) for c in thm_tof_cols])
        X_2, y_2, groups_2 = _process(train)
        
        X = pd.concat([X_1, X_2]).reset_index(drop=True)[self.features]
        y = np.concatenate([y_1, y_2])
        groups = np.concatenate([groups_1, groups_2])

        os.makedirs("data/tabular-models", exist_ok=True)
        joblib.dump(X, "data/tabular-models/X.pkl")
        joblib.dump(y, "data/tabular-models/y.pkl")
        joblib.dump(groups, "data/tabular-models/groups.pkl")
        
        del train, train_metadata, X_1, X_2, y_1, y_2, groups_1, groups_2
        gc.collect()
        
        return X, y, groups
    
    def process_test(self, dataset, metadata):
        imu_data = self.get_imu_data(dataset)
        thm_data = self.get_thm_data(dataset)    
        tof_data = self.get_tof_data(dataset)
        data = imu_data.join(thm_data, on="sequence_id").join(tof_data, on="sequence_id")
        
        subjects = dataset.group_by('sequence_id', maintain_order=True).agg(pl.col("subject").first())
        data = data.join(subjects, on="sequence_id", how="left")
        X_test = data[self.features]
        
        del dataset, metadata
        gc.collect()
        
        return X_test.to_pandas()


gbm_preprocessor = GBMPreprocessor()
X, y, groups = gbm_preprocessor.process_train()

nn_preprocessor_pt = NNPreprocessorPT(CFG.train_path, f"{CFG.data_path}/neural-networks/neural-networks-pt")
X_imu, X_thm, X_tof, y_int, y_ohe, groups, pad_len = nn_preprocessor_pt.process_train()

nn_preprocessor_tf = NNPreprocessorTF(CFG.train_path, f"{CFG.data_path}/neural-networks/neural-networks-tf")
X_imu, X_thm, X_tof, y_int, y_ohe, groups, pad_len = nn_preprocessor_tf.process_train()


def get_scores(y_true, y_pred_probs):
    y_preds = np.argmax(y_pred_probs, axis=1)
    
    binary_f1 = f1_score(
        np.where(y_true  <= 7, 1, 0),
        np.where(y_preds <= 7, 1, 0),
        zero_division = 0.0,
        average = "binary"
    )

    macro_f1 = f1_score(
        np.where(y_true   <= 7, y_true, 99),
        np.where(y_preds  <= 7, y_preds, 99),
        zero_division = 0.0,
        average = "macro"
    )

    score = (binary_f1 + macro_f1) / 2
    
    return binary_f1, macro_f1, score


@register_keras_serializable()
def se_block(x, reduction=8):
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

@register_keras_serializable()
def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, l2_1=1e-4, l2_2=1e-4):
    shortcut = x
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False, kernel_regularizer=l2(l2_1))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False, kernel_regularizer=l2(l2_2))(shortcut)
        shortcut = BatchNormalization()(shortcut)
        
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    
    return x

@register_keras_serializable()
def expand_last_axis(x):
    return tf.expand_dims(x, axis=-1)

@register_keras_serializable()
def squeeze_last_axis(x):
    return tf.squeeze(x, axis=-1)

@register_keras_serializable()
def time_sum(x):
    return K.sum(x, axis=1)

@register_keras_serializable()
def attention_layer(inputs):
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    
    return context


class SEBlockPT(nn.Module):
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

class ResNetSEBlockPT(nn.Module):
    def __init__(self, in_channels, out_channels, wd=1e-4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlockPT(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) :
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return self.relu(out)

class NNModelPT(nn.Module):
    def __init__(self, imu_dim, thm_dim, tof_dim, n_classes, **kwargs):
        super().__init__()
        
        self.imu_branch = nn.Sequential(
            self.residual_se_cnn_block(
                imu_dim, kwargs["imu1_channels"], 
                kwargs["imu1_layers"],
                drop=kwargs["imu1_dropout"]
            ),
            self.residual_se_cnn_block(
                kwargs["imu1_channels"],
                kwargs["feat_dim"], 
                kwargs["imu2_layers"],
                drop=kwargs["imu2_dropout"]
            )
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
            *[
                ResNetSEBlockPT(
                    in_channels=in_channels, 
                    out_channels=in_channels) 
                for _ in range(num_layers)
            ],
            ResNetSEBlockPT(in_channels, out_channels, wd=wd),
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


class SEBlockPT(nn.Module):
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

class ResNetSEBlockPT(nn.Module):
    def __init__(self, in_channels, out_channels, wd=1e-4):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlockPT(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x) :
        identity = self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = out + identity
        return self.relu(out)

class AttentionLayerPT(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.score_fn = nn.Linear(feature_dim, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        score = torch.tanh(self.score_fn(x)) 
        weights = self.softmax(score.squeeze(-1))
        weights = weights.unsqueeze(-1)    
        context = x * weights              
        return context.sum(dim=1)      

class GaussianNoisePT(nn.Module):
    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev
    
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x

class CMIBackbone(nn.Module):
    def __init__(self, imu_dim, thm_dim, tof_dim, **kwargs):
        super().__init__()
        self.imu_acc_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )
        self.imu_rot_branch = nn.Sequential(
            self.residual_feature_block(4, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )
        self.imu_other_branch = nn.Sequential(
            self.residual_feature_block(imu_dim-7, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )

        self.thm_branch1, self.tof_branch1 = self.init_thm_tof_branch(thm_dim//5, tof_dim//5, **kwargs)
        self.thm_branch2, self.tof_branch2 = self.init_thm_tof_branch(thm_dim//5, tof_dim//5, **kwargs)
        self.thm_branch3, self.tof_branch3 = self.init_thm_tof_branch(thm_dim//5, tof_dim//5, **kwargs)
        self.thm_branch4, self.tof_branch4 = self.init_thm_tof_branch(thm_dim//5, tof_dim//5, **kwargs)
        self.thm_branch5, self.tof_branch5 = self.init_thm_tof_branch(thm_dim//5, tof_dim//5, **kwargs)

        self.imu_proj = ResNetSEBlockPT(in_channels=3*kwargs["imu2_channels"], out_channels=kwargs["imu2_channels"])
        self.thm_proj = ResNetSEBlockPT(in_channels=5*kwargs["thm2_channels"], out_channels=kwargs["thm2_channels"])
        self.tof_proj = ResNetSEBlockPT(in_channels=5*kwargs["tof2_channels"], out_channels=kwargs["tof2_channels"])

        self.lstm = nn.LSTM(
            input_size=kwargs['imu2_channels']+kwargs['thm2_channels']+kwargs['tof2_channels'],
            hidden_size=kwargs['lstm_hidden_size'],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.gru = nn.GRU(
            input_size=kwargs['imu2_channels']+kwargs['thm2_channels']+kwargs['tof2_channels'],
            hidden_size=kwargs['gru_hidden_size'],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.noise = GaussianNoisePT(kwargs['gaussian_noise_rate'])
        self.dense = nn.Sequential(
            nn.Linear(kwargs['imu2_channels']+kwargs['thm2_channels']+kwargs['tof2_channels'], kwargs['dense_channels']),
            nn.ELU()
        )
        
        self.attn = AttentionLayerPT(feature_dim=(kwargs['lstm_hidden_size']+kwargs['gru_hidden_size'])*2+kwargs['dense_channels'])

    def feature_block(self, in_channels, out_channels, num_layers, pool_size=2, drop=0.3):
        return nn.Sequential(
            *[ResNetSEBlockPT(in_channels=in_channels, out_channels=in_channels) for i in range(num_layers)],
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size, ceil_mode=True),
            nn.Dropout(drop)
        )

    def residual_feature_block(self, in_channels, out_channels, num_layers, pool_size=2, drop=0.3):
        return nn.Sequential(
            *[ResNetSEBlockPT(in_channels=in_channels, out_channels=in_channels) for i in range(num_layers)],
            ResNetSEBlockPT(in_channels, out_channels, wd=1e-4),
            nn.MaxPool1d(pool_size, ceil_mode=True),
            nn.Dropout(drop)
        )

    def init_thm_tof_branch(self, thm_dim, tof_dim, **kwargs):
        thm_branch = nn.Sequential(
            self.feature_block(thm_dim, kwargs["thm1_channels"], kwargs["thm1_layers"], drop=kwargs["thm1_dropout"]),
            self.feature_block(kwargs["thm1_channels"], kwargs["thm2_channels"], kwargs["thm2_layers"], drop=kwargs["thm2_dropout"]),
        )
        tof_branch = nn.Sequential(
            self.feature_block(tof_dim, kwargs["tof1_channels"], kwargs["tof1_layers"], drop=kwargs["tof1_dropout"]),
            self.feature_block(kwargs["tof1_channels"], kwargs["tof2_channels"], kwargs["tof2_layers"], drop=kwargs["tof2_dropout"]),
        )
        return thm_branch, tof_branch
    
    def forward(self, imus, thms, tofs):
        imu_acc, imu_rot, imu_other = imus
        imu_acc_feat = self.imu_acc_branch(imu_acc.permute(0, 2, 1))
        imu_rot_feat = self.imu_rot_branch(imu_rot.permute(0, 2, 1))
        imu_other_feat = self.imu_other_branch(imu_other.permute(0, 2, 1))
        imu_feat = self.imu_proj(torch.cat([imu_acc_feat, imu_rot_feat, imu_other_feat], dim=1))
        
        thm1, thm2, thm3, thm4, thm5 = thms
        tof1, tof2, tof3, tof4, tof5 = tofs
        
        thm1_feat = self.thm_branch1(thm1.permute(0, 2, 1))
        thm2_feat = self.thm_branch2(thm2.permute(0, 2, 1))
        thm3_feat = self.thm_branch3(thm3.permute(0, 2, 1))
        thm4_feat = self.thm_branch4(thm4.permute(0, 2, 1))
        thm5_feat = self.thm_branch5(thm5.permute(0, 2, 1))
        thm_feat = self.thm_proj(torch.cat([thm1_feat, thm2_feat, thm3_feat, thm4_feat, thm5_feat], dim=1))
        
        tof1_feat = self.tof_branch1(tof1.permute(0, 2, 1))
        tof2_feat = self.tof_branch2(tof2.permute(0, 2, 1))
        tof3_feat = self.tof_branch3(tof3.permute(0, 2, 1))
        tof4_feat = self.tof_branch4(tof4.permute(0, 2, 1))
        tof5_feat = self.tof_branch5(tof5.permute(0, 2, 1))
        tof_feat = self.tof_proj(torch.cat([tof1_feat, tof2_feat, tof3_feat, tof4_feat, tof5_feat], dim=1))
        
        feat = torch.cat([imu_feat, thm_feat, tof_feat], dim=1).permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat)
        gru_out, _ = self.gru(feat)
        dense_out = self.dense(self.noise(feat))
        
        return self.attn(torch.cat([lstm_out, gru_out, dense_out], dim=-1))
        

class NNModelPT2(nn.Module):
    def __init__(self, imu_dim, thm_dim, tof_dim, imu_indices, thm_indices, tof_indices, **kwargs):
        super().__init__()
        self.imu_indices = imu_indices
        self.thm_indices = thm_indices
        self.tof_indices = tof_indices
        
        self.backbone = CMIBackbone(imu_dim, thm_dim, tof_dim, **kwargs)
        self.target_classifier = nn.Sequential(
            nn.Linear((kwargs['lstm_hidden_size']+kwargs['gru_hidden_size'])*2+kwargs['dense_channels'], kwargs["cls_channels1"]),
            nn.BatchNorm1d(kwargs["cls_channels1"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout1"]),
            nn.Linear(kwargs["cls_channels1"], kwargs["cls_channels2"]),
            nn.BatchNorm1d(kwargs["cls_channels2"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout2"]),
            nn.Linear(kwargs["cls_channels2"], 8)
        )
        self.non_target_classifier = nn.Sequential(
            nn.Linear((kwargs['lstm_hidden_size']+kwargs['gru_hidden_size'])*2+kwargs['dense_channels'], kwargs["cls_channels1"]),
            nn.BatchNorm1d(kwargs["cls_channels1"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout1"]),
            nn.Linear(kwargs["cls_channels1"], kwargs["cls_channels2"]),
            nn.BatchNorm1d(kwargs["cls_channels2"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout2"]),
            nn.Linear(kwargs["cls_channels2"], 10)
        )
    
    def forward(self, imu, thm, tof):        
        imu = [imu[:, :, self.imu_indices[k]] for k in self.imu_indices]
        thm = [thm[:, :, self.thm_indices[k]] for k in range(1, 6)]
        tof = [tof[:, :, self.tof_indices[k]] for k in range(1, 6)]
        
        feat = self.backbone(imu, thm, tof)
        targets_y = self.target_classifier(feat)
        non_targets_y = self.non_target_classifier(feat)
        return torch.cat([targets_y, non_targets_y], dim=1)


model_config_pt_1 = {
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
    "cls2_dropout": 0.22502521933558461,
    "imu_dim": nn_preprocessor_pt.imu_dim, 
    "thm_dim": nn_preprocessor_pt.thm_dim,
    "tof_dim": nn_preprocessor_pt.tof_dim,
    "n_classes": 18
}

model_config_pt_2 = {
    "imu_dim": nn_preprocessor_pt.imu_dim, 
    "thm_dim": nn_preprocessor_pt.thm_dim,
    "tof_dim": nn_preprocessor_pt.tof_dim,
    "imu1_channels": 128, 
    "imu2_channels": 256, 
    "imu1_dropout": 0.3,
    "imu2_dropout": 0.25,
    "imu1_layers": 0, 
    "imu2_layers": 0, 
    "thm1_channels": 32, 
    "thm2_channels": 64,
    "thm1_dropout": 0.25, 
    "thm2_dropout": 0.2,
    "thm1_layers": 0,
    "thm2_layers": 0, 
    "tof1_channels": 256, 
    "tof2_channels": 512, 
    "tof1_dropout": 0.4,
    "tof2_dropout": 0.3,
    "tof1_layers": 0, 
    "tof2_layers": 0, 
    "lstm_hidden_size": 128,
    "gru_hidden_size": 128, 
    "gaussian_noise_rate": 0.1, 
    "dense_channels": 32,
    "cls_channels1": 256, 
    "cls_dropout1": 0.2,
    "cls_channels2": 128,
    "cls_dropout2": 0.2,
    "target_classes_num": 8, 
    "non_target_classes_num": 10,
    "thm_indices": nn_preprocessor_pt.global_thm_indices,
    "tof_indices": nn_preprocessor_pt.global_tof_indices,
    "imu_indices": nn_preprocessor_pt.global_imu_indices,
}


class FX:
    def get_final_error(self, error, weight):
        return error
    
    def is_max_optimal(self):
        return True
    
    def evaluate(self, approxes, target, weight):
        def softmax(x):
            exp_x = np.exp(x - np.max(x))
            return exp_x / np.sum(exp_x)
        
        n_samples = len(target)
        n_classes = len(approxes)
        
        y_pred_probs = np.zeros((n_samples, n_classes))
        for i in range(n_samples):
            raw_preds = [approxes[class_idx][i] for class_idx in range(n_classes)]
            y_pred_probs[i] = softmax(raw_preds)
        
        y_preds = np.argmax(y_pred_probs, axis=1)
        y_true = np.array(target)
        
        binary_f1 = f1_score(
            np.where(y_true <= 7, 1, 0),
            np.where(y_preds <= 7, 1, 0),
            zero_division=0.0,
            average="binary"
        )
        
        macro_f1 = f1_score(
            np.where(y_true <= 7, y_true, 99),
            np.where(y_preds <= 7, y_preds, 99),
            zero_division=0.0,
            average="macro"
        )
        
        score = (binary_f1 + macro_f1) / 2
        
        return score, 1.0


fold_scores = {}
nn_base_models = {}
tabular_base_models = {}

nn_model_paths = glob.glob(f"{CFG.models_path}/neural-networks/**")
tabular_model_paths = glob.glob(f"{CFG.models_path}/tabular-models/*")
model_paths = nn_model_paths + tabular_model_paths

for model_path in model_paths:
    model_name = model_path.split("/")[-1]
    
    temp_scores = {
        "binary_f1": [],
        "macro_f1": [],
        "score": []
    }

    oof_file = joblib.load(f"{model_path}/oof_pred_probs.pkl")
    if "neural-networks-pt" not in model_name:
        oof_file = logit(oof_file)
    
    splits = StratifiedGroupKFold(n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed).split(X, y, groups)
    for _, val_idx in splits:
        y_val = y[val_idx]
        y_preds = oof_file[val_idx]
        
        scores = get_scores(y_val, y_preds)
        
        temp_scores["binary_f1"].append(scores[0])
        temp_scores["macro_f1"].append(scores[1])
        temp_scores["score"].append(scores[2])
        
    fold_scores[model_name] = temp_scores

    if "neural-networks" in model_name:
        models = []
        for i in range(CFG.n_folds):
            if model_name == "neural-networks-pt-1":
                checkpoint = torch.load(f"{model_path}/models/fold_{i}.pth")
                model = NNModelPT(**model_config_pt_1).to(CFG.device)
                model.load_state_dict(checkpoint)
                model.eval()
                models.append(model)

            elif model_name == "neural-networks-pt-2":
                checkpoint = torch.load(f"{model_path}/models/fold_{i}.pth")
                model = NNModelPT2(**model_config_pt_2).to(CFG.device)
                model.load_state_dict(checkpoint)
                model.eval()
                models.append(model)
    
            else:
                custom_objs = {
                    'time_sum': time_sum,
                    'squeeze_last_axis': squeeze_last_axis,
                    'expand_last_axis': expand_last_axis,
                    'se_block': se_block,
                    'residual_se_cnn_block': residual_se_cnn_block,
                    'attention_layer': attention_layer,
                }
                model = load_model(f"{model_path}/models/fold_{i}.keras", compile=False, custom_objects=custom_objs, safe_mode=False)
                models.append(model)
    
        nn_base_models[model_name] = models
    else:
        tabular_base_models[model_name] = joblib.load(f"{model_path}/models.pkl")


scaler_paths_pt = {
    "imu": CFG.data_path + "/neural-networks/neural-networks-pt/imu_scaler.pkl",
    "thm": CFG.data_path + "/neural-networks/neural-networks-pt/thm_scaler.pkl",
    "tof": CFG.data_path + "/neural-networks/neural-networks-pt/tof_scaler.pkl"
}

scaler_path_tf = CFG.data_path + "/neural-networks/neural-networks-tf/scaler.pkl"


def get_nn_models_predictions(sequence, demographics):
    X_test_imu_tf, X_test_thm_tf, X_test_tof_tf = nn_preprocessor_tf.process_test(sequence, pad_len, scaler_path_tf)
    X_test_tf = np.concatenate([X_test_imu_tf, X_test_thm_tf, X_test_tof_tf], axis=-1)  

    X_test_imu_pt, X_test_thm_pt, X_test_tof_pt = nn_preprocessor_pt.process_test(sequence, pad_len, scaler_paths_pt)
    X_test_imu_pt = torch.from_numpy(X_test_imu_pt).float().to(CFG.device)
    X_test_thm_pt = torch.from_numpy(X_test_thm_pt).float().to(CFG.device)
    X_test_tof_pt = torch.from_numpy(X_test_tof_pt).float().to(CFG.device)
    
    test_pred_probs = {}
    for model_name, pretrained_models in nn_base_models.items(): 
        temp_test_pred_probs = np.zeros((len(X_test_tf), 18))
        
        if "neural-networks-pt" in model_name:
            with torch.no_grad():        
                outputs = []
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    for model in pretrained_models:
                        logits = model(X_test_imu_pt, X_test_thm_pt, X_test_tof_pt)
                        outputs.append(logits)
            temp_test_pred_probs = torch.mean(torch.stack(outputs), dim=0).cpu().detach().numpy()
        else:
            for model in pretrained_models:
                temp_test_pred_probs += model.predict(X_test_tf, verbose=False) / len(pretrained_models)
            temp_test_pred_probs = logit(temp_test_pred_probs)

        test_pred_probs[model_name] = temp_test_pred_probs
        del temp_test_pred_probs
        gc.collect()

    return test_pred_probs


def get_tabular_models_predictions(sequence, demographics):
    X_test = gbm_preprocessor.process_test(sequence, demographics)
    
    test_pred_probs = {}
    for model_name, pretrained_models in tabular_base_models.items(): 
        temp_test_pred_probs = np.zeros((len(X_test), 18))
        for model in pretrained_models:
            temp_test_pred_probs += model.predict_proba(X_test) / len(pretrained_models)
            
        temp_test_pred_probs = logit(temp_test_pred_probs)

        test_pred_probs[model_name] = temp_test_pred_probs
        del temp_test_pred_probs
        gc.collect()

    return test_pred_probs


weights = joblib.load("/kaggle/input/cmi-train-weightedensemble-1/weighted-ensemble/weights.pkl")
fold_scores["weighted-ensemble"] = joblib.load("/kaggle/input/cmi-train-weightedensemble-1/weighted-ensemble/scores_0.840973.pkl")["weighted-ensemble"]


def predict(sequence, demographics):        
    test_pred_probs = {
        **get_nn_models_predictions(sequence, demographics),
        **get_tabular_models_predictions(sequence, demographics)
    }

    best_weights = [weights[model] for model in test_pred_probs.keys()]
    best_weights /= np.sum(best_weights)
    
    weighted_test_preds = np.zeros((sequence.shape[0], 18))
    for model, weight in zip(test_pred_probs.keys(), best_weights):
        weighted_test_preds += test_pred_probs[model] * weight

    test_preds = np.argmax(weighted_test_preds, axis=1)
    final_preds = [num_to_label[p] for p in test_preds]
    
    return str(final_preds[0])


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            CFG.test_path,
            CFG.test_metadata_path,
        )
    )


def plot_scores(scores, name):
    mean_scores = scores.mean().sort_values(ascending=False)
    order = scores.mean().sort_values(ascending=False).index.tolist()

    min_score = mean_scores.min()
    max_score = mean_scores.max()
    padding = (max_score - min_score) * 0.5
    lower_limit = min_score - padding
    upper_limit = max_score + padding

    fig, axs = plt.subplots(1, 2, figsize=(15, scores.shape[1] * 0.5))

    boxplot = sns.boxplot(data=scores, order=order, ax=axs[0], orient="h", color="grey")
    axs[0].set_title(f"Fold {name}")
    axs[0].set_xlabel("")
    axs[0].set_ylabel("")

    barplot = sns.barplot(x=mean_scores.values, y=mean_scores.index, ax=axs[1], color="grey")
    axs[1].set_title(f"Mean {name}")
    axs[1].set_xlabel("")
    axs[1].set_xlim(left=lower_limit, right=upper_limit)
    axs[1].set_ylabel("")

    for i, (score, model) in enumerate(zip(mean_scores.values, mean_scores.index)):
        color = "cyan" if "ensemble" in model.lower() or "logistic" in model.lower() else "grey"
        barplot.patches[i].set_facecolor(color)
        boxplot.patches[i].set_facecolor(color)
        barplot.text(score, i, round(score, 6), va="center")

    plt.tight_layout()
    plt.show()


comp_scores = {}
macro_f1_scores = {}
binary_f1_scores = {}

for model in fold_scores.keys():
    comp_scores[model] = fold_scores[model]["score"]
    macro_f1_scores[model] = fold_scores[model]["macro_f1"]
    binary_f1_scores[model] = fold_scores[model]["binary_f1"]
    
comp_scores = pd.DataFrame(comp_scores)
macro_f1_scores = pd.DataFrame(macro_f1_scores)
binary_f1_scores = pd.DataFrame(binary_f1_scores)


plot_scores(comp_scores, "Competition Score")


plot_scores(macro_f1_scores, "Macro F1")


plot_scores(binary_f1_scores, "Binary F1")

