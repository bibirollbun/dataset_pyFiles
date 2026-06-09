import os
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.notebook import tqdm
from torch.amp import autocast
import pandas as pd
import polars as pl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import fft
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import f1_score
import random
import math
from multiprocessing import freeze_support
#freeze_support() # for num_workers>0, but still not working on my machine


TRAIN = False #True #

root_dir = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
universe_csv_path = Path("/kaggle/input/cmi-precompute/pytorch/all/1/tof-1_raw.csv")
model_dir = Path("/kaggle/input/cmi-fold5-tf/pytorch/default/1")

seed = 0
batch_size = 64
num_workers = 0 #4
n_folds = 5
CUDA0 = "cuda:0"


model_args = {"imu1_channels": 128, "imu2_channels": 256, "imu1_dropout": 0.3, "imu2_dropout": 0.25,
              "imu1_layers": 0, "imu2_layers": 0,
              "thm1_channels": 32, "thm2_channels": 64, "thm1_dropout": 0.25, "thm2_dropout": 0.2,
              "thm1_layers": 0, "thm2_layers": 0,
              "tof1_channels": 256, "tof2_channels": 512, "tof1_dropout": 0.4, "tof2_dropout": 0.3,
              "tof1_layers": 0, "tof2_layers": 0,
              "lstm_hidden_size": 128, "gru_hidden_size": 128, "gaussian_noise_rate": 0.1, "dense_channels": 32,
              "cls_channels1": 256, "cls_dropout1": 0.2, "cls_channels2": 128, "cls_dropout2": 0.2,
              "target_classes_num": 8, "non_target_classes_num": 10, }



imu_only = False


def fix_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # NumPy
    np.random.seed(seed)
    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # # when using multiple GPUs
    # CUDA deterministic mode (may reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Disable GPU atomic operation randomness (PyTorch 2.0+)
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        import cv2
        from transformers import set_seed
        set_seed(seed)  # override internal randomness
        cv2.setRNGSeed(seed)
        print("case 2")
    except:
        print("case 1")


fix_seed(seed)


def init_dataset():
    dataset_config = {
        "percent": 99,
        "scaler_config": StandardScaler(),
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
        "one_scale": False,
        "tof_raw": True,
        "tof_mode": 16,
        "save_precompute": False,
        "fold_y": "gesture",
        "fold_groups": "subject",
    }

    dataset = CMIFoldDataset(universe_csv_path, dataset_config, full_dataset_function=CMIFeDataset, n_folds=n_folds, random_seed=seed)
    dataset.print_fold_stats()
    return dataset


def get_fold_dataset(dataset, fold):
    _, valid_dataset = dataset.get_fold_datasets(fold)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
    return valid_loader


#dataset = init_dataset()


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


def calculate_angular_velocity_from_quat(rot_data, time_delta=1 / 200):  # Assuming 200Hz sampling rate
    if isinstance(rot_data, pd.DataFrame):
        quat_values = rot_data[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    else:
        quat_values = rot_data
    num_samples = quat_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))
    for i in range(num_samples - 1):
        q_t = quat_values[i]
        q_t_plus_dt = quat_values[i + 1]
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
        q2 = quat_values[i + 1]
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
            angular_dist[i] = 0  # В случае недействительных кватернионов
            pass
    return angular_dist


class CMIFeDataset(Dataset):
    def __init__(self, data_path, config):
        self.config = config
        self.init_feature_names(data_path)
        df = self.generate_features(pd.read_csv(data_path, usecols=set(self.use_cols) & set(self.raw_columns)))
        self.generate_dataset(df)

    def init_feature_names(self, data_path):
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

        self.acc_features = ['acc_mag', 'acc_mag_jerk', 'linear_acc_mag', 'linear_acc_mag_jerk']
        self.rot_features = ['rot_angle', 'rot_angle_vel', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
                             'angular_distance']
        self.old_imu_features = [
            'acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel',
            'linear_acc_mag', 'linear_acc_mag_jerk',
            'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance'
        ]

        self.extra_imu_features = self.config.get("imu_feats", [])
        self.imu_features = self.extra_imu_features.copy()
        if self.config.get("add_imu_feat_default", True):
            if self.config.get("old_imu_feat", True):
                self.imu_features.extend(self.old_imu_features)
            else:
                self.imu_features.extend(self.acc_features)
                self.imu_features.extend(self.rot_features)
        self.er1_fearues = ["er_x", "er_y", "er_z"]
        self.er2_fearues = ['er_r_xy', 'er_r_xz', 'er_r_yz', 'er_c_xy', 'er_c_xz', 'er_c_yz']
        self.er_fearues = self.er1_fearues + self.er2_fearues
        self.tof_mode = self.config.get("tof_mode", "stats")
        self.tof_region_stats = ['mean', 'std', 'min', 'max']
        self.tof_cols = self.generate_tof_feature_names()

        self.raw_columns = pd.read_csv(data_path, nrows=0).columns.tolist()
        self.imu_acc_cols_base = ['acc_x', 'acc_y', 'acc_z', 'linear_acc_x', 'linear_acc_y',
                                  'linear_acc_z'] if self.config.get("add_raw_acc", False) else ['linear_acc_x',
                                                                                                 'linear_acc_y',
                                                                                                 'linear_acc_z']
        self.imu_rot_cols_base = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
        self.imu_cols_base = self.imu_acc_cols_base + self.imu_rot_cols_base
        self.imu_cols = list()
        self.imu_channel_keys = defaultdict(list)
        if self.config.get("add_imu_base", True):
            self.imu_cols.extend(self.imu_cols_base)
            self.imu_channel_keys["acc"] = self.imu_acc_cols_base
            self.imu_channel_keys["rot"] = self.imu_rot_cols_base
        if self.config.get("add_imu_feats", True):
            self.imu_cols.extend(self.imu_features)
            if self.config.get("split_imu_feat", False):
                if self.config.get("old_imu_feat", True):
                    assert False, "split_imu_feat=True and old_imu_feat=True not supported"
                self.imu_channel_keys["acc_feat"] = self.acc_features
                self.imu_channel_keys["rot_feat"] = self.rot_features
            else:
                if self.config.get("old_imu_feat", True):
                    self.imu_channel_keys["other"].extend(self.old_imu_features)
                else:
                    self.imu_channel_keys["other"].extend(self.acc_features)
                    self.imu_channel_keys["other"].extend(self.rot_features)
        if self.config.get("add_imu_er_feats", False):
            self.imu_cols.extend(self.er_fearues)
            if self.config.get("split_imu_feat", False):
                self.imu_channel_keys["er1_feat"] = self.er1_fearues
                self.imu_channel_keys["er2_feat"] = self.er2_fearues
            else:
                self.imu_channel_keys["other"].extend(self.er1_fearues)
                self.imu_channel_keys["other"].extend(self.er2_fearues)
        self.flip_imu_cols = [f"{col}_flip" for col in self.imu_cols]
        self.imu_channel_keys = {k: sorted(v) for k, v in self.imu_channel_keys.items()}
        self.thm_cols = [c for c in self.raw_columns if c.startswith('thm_')]
        self.thm_channel_keys = {k: [f"thm_{k}"] for k in range(1, 6)}
        self.feature_cols = self.imu_cols + self.thm_cols + self.tof_cols
        self.imu_dim = len(self.imu_cols)
        self.thm_dim = len(self.thm_cols)
        self.tof_dim = len(self.tof_cols)
        self.base_cols = (['acc_x', 'acc_y', 'acc_z',
                          'rot_x', 'rot_y', 'rot_z', 'rot_w',
                          'sequence_id', 'subject',
                          'sequence_type', 'gesture', 'orientation']
                          + [c for c in self.raw_columns if c.startswith('thm_')]
                          + [f"tof_{i}_v{p}" for i in range(1, 6) for p in range(64)])
        self.use_cols = self.base_cols + self.feature_cols
        if self.config.get("return_flip_imu", False):
            self.use_cols.extend(self.flip_imu_cols)
        self.fold_cols = ['subject', 'sequence_type', 'gesture', 'orientation', 'sequence_id']
        if self.config.get("use_dg", False):
            self.dg_cols = ['adult_child', 'age', 'sex', 'handedness', 'shoulder_to_wrist_height', 'elbow_to_wrist_height']
        self.global_imu_indices = {k: sorted([self.imu_cols.index(feat) for feat in feats]) for k, feats in self.imu_channel_keys.items()}
        self.global_thm_indices = {k: sorted([self.thm_cols.index(key) for key in self.thm_channel_keys[k]]) for k in range(1, 6)}
        self.global_tof_indices = {k: sorted([self.tof_cols.index(key) for key in self.tof_channel_keys[k]]) for k in range(1, 6)}

    def generate_tof_feature_names(self):
        features = list()
        self.tof_channel_keys = defaultdict(list)
        if self.config.get("tof_raw", False):
            for i in range(1, 6):
                features.extend([f"tof_{i}_v{p}" for p in range(64)])
                self.tof_channel_keys[i].extend([f"tof_{i}_v{p}" for p in range(64)])
        for i in range(1, 6):
            if self.tof_mode != 0:
                for stat in self.tof_region_stats:
                    features.append(f'tof_{i}_{stat}')
                    self.tof_channel_keys[i].append(f'tof_{i}_{stat}')
                if self.tof_mode > 1:
                    for r in range(self.tof_mode):
                        for stat in self.tof_region_stats:
                            features.append(f'tof{self.tof_mode}_{i}_region_{r}_{stat}')
                            self.tof_channel_keys[i].append(f'tof{self.tof_mode}_{i}_region_{r}_{stat}')
                if self.tof_mode == -1:
                    for mode in [2, 4, 8, 16, 32]:
                        for r in range(mode):
                            for stat in self.tof_region_stats:
                                features.append(f'tof{mode}_{i}_region_{r}_{stat}')
                                self.tof_channel_keys[i].append(f'tof{mode}_{i}_region_{r}_{stat}')
        return features

    def compute_cross_axis_energy(self, df):
        axes = ['x', 'y', 'z']
        features = {}
        for axis in axes:
            fft_result = fft(df[f'acc_{axis}'].values)
            energy = np.sum(np.abs(fft_result) ** 2)
            features[f"er_{axis}"] = energy
        for i, axis1 in enumerate(axes):
            for axis2 in axes[i + 1:]:
                features[f'er_r_{axis1}{axis2}'] = features[f'er_{axis1}'] / (features[f'er_{axis2}'] + 1e-6)
        for i, axis1 in enumerate(axes):
            for axis2 in axes[i + 1:]:
                features[f'er_c_{axis1}{axis2}'] = \
                    np.corrcoef(np.abs(fft(df[f'acc_{axis1}'].values)), np.abs(fft(df[f'acc_{axis2}'].values)))[0, 1]
        return {k: v for k, v in features.items() if k in self.er_fearues}

    def compute_imu_features(self, df):
        if self.config.get("rot_fillna", False):
            df['rot_w'] = df['rot_w'].fillna(1)
            df[['rot_x', 'rot_y', 'rot_z']] = df[['rot_x', 'rot_y', 'rot_z']].fillna(0)
        df['acc_mag'] = np.sqrt(df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2)
        df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))
        df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
        df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)

        linear_accel_list = []
        for _, group in df.groupby('sequence_id'):
            acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
            linear_accel_list.append(
                pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'],
                             index=group.index))
        df_linear_accel = pd.concat(linear_accel_list)
        df = pd.concat([df, df_linear_accel], axis=1)
        df['linear_acc_mag'] = np.sqrt(df['linear_acc_x'] ** 2 + df['linear_acc_y'] ** 2 + df['linear_acc_z'] ** 2)
        df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)

        angular_vel_list = []
        for _, group in df.groupby('sequence_id'):
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group)
            angular_vel_list.append(
                pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'],
                             index=group.index))
        df_angular_vel = pd.concat(angular_vel_list)
        df = pd.concat([df, df_angular_vel], axis=1)

        angular_distance_list = []
        for _, group in df.groupby('sequence_id'):
            rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
            angular_dist_group = calculate_angular_distance(rot_data_group)
            angular_distance_list.append(
                pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))
        df_angular_distance = pd.concat(angular_distance_list)
        df = pd.concat([df, df_angular_distance], axis=1)
        return df

    def compute_flip_features(self, df):
        flip_df = df[['sequence_id', 'acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']].copy()
        flip_df[['acc_x', 'acc_y', 'rot_x', 'rot_y']] *= -1
        flip_df = self.compute_imu_features(flip_df)
        for col in flip_df.columns:
            if col != 'sequence_id':
                df[f"{col}_flip"] = flip_df[col]
        return df

    def compute_features(self, df):
        df = self.compute_imu_features(df)
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
                        region_data = tof_data.iloc[:, r * region_size: (r + 1) * region_size]
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
                            region_data = tof_data.iloc[:, r * region_size: (r + 1) * region_size]
                            new_columns.update({
                                f'tof{mode}_{i}_region_{r}_mean': region_data.mean(axis=1),
                                f'tof{mode}_{i}_region_{r}_std': region_data.std(axis=1),
                                f'tof{mode}_{i}_region_{r}_min': region_data.min(axis=1),
                                f'tof{mode}_{i}_region_{r}_max': region_data.max(axis=1)
                            })
            df = pd.concat([df, pd.DataFrame(new_columns)], axis=1)

        def _calc_features(group):
            return pd.DataFrame(self.compute_cross_axis_energy(group), index=[group.index[0]])

        features_df = df.groupby('sequence_id', group_keys=False).apply(_calc_features)
        df = df.join(features_df, how='left')
        df[features_df.columns] = df.groupby('sequence_id')[features_df.columns].ffill()

        return df

    def generate_features(self, df):
        self.le = LabelEncoder()
        if self.config.get("one_neg", False):
            neg_other = "Write name on leg"
            df['gesture'] = df['gesture'].apply(lambda x: x if x in self.target_gestures else neg_other)
        df['gesture_int'] = self.le.fit_transform(df['gesture'])
        self.class_num = len(self.le.classes_)
        self.target_ints = np.array([self.le.classes_.tolist().index(name) for name in self.target_gestures])
        self.non_target_ints = np.array([self.le.classes_.tolist().index(name) for name in self.non_target_gestures])

        if all(c in df.columns for c in self.feature_cols):
            print("Features have precomputed, skip compute.")
        else:
            print("Features not precomputed, do compute.")
            df = self.compute_features(df)

        if self.config.get("return_flip_imu", False):
            if all(c in df.columns for c in self.flip_imu_cols):
                print("Flip have precomputed, skip compute.")
            else:
                print("Flip not precomputed, do compute.")
                df = self.compute_flip_features(df)

        if self.config.get("use_dg", False):
            dg_df = pd.read_csv(self.config["dg_path"])
            df = pd.merge(df, dg_df, how='left', on='subject')
            df['age'] /= 100
            df['shoulder_to_wrist_height'] = df['shoulder_to_wrist_cm'] / df['height_cm']
            df['elbow_to_wrist_height'] = df['elbow_to_wrist_cm'] / df['height_cm']

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
        print(f"Max: {max_value}, set nan to {nan_value}")
        return nan_value

    def generate_dataset(self, df):
        seq_gp = df.groupby('sequence_id')
        imu_unscaled, thm_unscaled, tof_unscaled = list(), list(), list()
        if self.config.get("return_flip_imu", False): flip_imu_unscaled = list()
        classes, lens = list(), list()
        self.imu_nan_value = self.get_nan_value(df[self.imu_cols], self.config["nan_ratio"]["imu"])
        self.thm_nan_value = self.get_nan_value(df[self.thm_cols], self.config["nan_ratio"]["thm"])
        self.tof_nan_value = self.get_nan_value(df[self.tof_cols], self.config["nan_ratio"]["tof"])
        if self.config.get("use_dg", False):
            self.dg = list()

        self.fold_feats = defaultdict(list)
        for seq_id, seq_df in seq_gp:
            imu_data = seq_df[self.imu_cols]
            if self.config["fbfill"]["imu"]:
                imu_data = imu_data.ffill().bfill()
            imu_unscaled.append(imu_data.fillna(self.imu_nan_value).values.astype('float32'))

            if self.config.get("return_flip_imu", False):
                flip_imu_data = seq_df[self.flip_imu_cols]
                if self.config["fbfill"]["imu"]:
                    flip_imu_data = flip_imu_data.ffill().bfill()
                flip_imu_unscaled.append(flip_imu_data.fillna(self.imu_nan_value).values.astype('float32'))

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

            if self.config.get("use_dg", False):
                self.dg.append(seq_df[self.dg_cols].iloc[0].values.astype('float32'))

        self.dataset_indices = classes
        self.pad_len = int(np.percentile(lens, self.config.get("percent", 95)))
        if self.config.get("one_scale", True):
            x_unscaled = [np.concatenate([imu, thm, tof], axis=1) for imu, thm, tof in
                          zip(imu_unscaled, thm_unscaled, tof_unscaled)]
            x_scaled, self.x_scaler = self.scale(x_unscaled)
            x = self.pad(x_scaled, self.imu_cols + self.thm_cols + self.tof_cols)
            self.imu = x[..., :self.imu_dim]
            self.thm = x[..., self.imu_dim:self.imu_dim + self.thm_dim]
            self.tof = x[..., self.imu_dim + self.thm_dim:self.imu_dim + self.thm_dim + self.tof_dim]

            if self.config.get("return_flip_imu", False):
                flip_x_unscaled = [np.concatenate([flip_imu, thm, tof], axis=1) for flip_imu, thm, tof in
                                   zip(flip_imu_unscaled, thm_unscaled, tof_unscaled)]
                flip_x_scaled = [self.x_scaler.transform(x) for x in flip_x_unscaled]
                flip_x = self.pad(flip_x_scaled, self.imu_cols + self.thm_cols + self.tof_cols)
                self.flip_imu = flip_x[..., :self.imu_dim]
        else:
            imu_scaled, self.imu_scaler = self.scale(imu_unscaled)
            thm_scaled, self.thm_scaler = self.scale(thm_unscaled)
            tof_scaled, self.tof_scaler = self.scale(tof_unscaled)
            self.imu = self.pad(imu_scaled, self.imu_cols)
            self.thm = self.pad(thm_scaled, self.thm_cols)
            self.tof = self.pad(tof_scaled, self.tof_cols)

            if self.config.get("return_flip_imu", False):
                flip_imu_scaled = [self.imu_scaler.transform(x) for x in flip_imu_unscaled]
                self.flip_imu = self.pad(flip_imu_scaled, self.imu_cols)
        self.precompute_scaled_nan_values()
        self.class_ = F.one_hot(torch.from_numpy(np.array(classes)).long(),
                                num_classes=len(self.le.classes_)).float().numpy()
        self.binary_class_ = np.isin(np.array(classes), self.target_ints).astype(np.float32)
        self.class_weight = torch.FloatTensor(
            compute_class_weight('balanced', classes=np.arange(len(self.le.classes_)), y=classes))

    def precompute_scaled_nan_values(self):
        dummy_df = pd.DataFrame(
            np.array([[self.imu_nan_value] * len(self.imu_cols) +
                      [self.thm_nan_value] * len(self.thm_cols) +
                      [self.tof_nan_value] * len(self.tof_cols)]),
            columns=self.imu_cols + self.thm_cols + self.tof_cols
        )

        if self.config.get("one_scale", True):
            scaled = self.x_scaler.transform(dummy_df)
            self.imu_scaled_nan = scaled[0, :self.imu_dim].mean()
            self.thm_scaled_nan = scaled[0, self.imu_dim:self.imu_dim + self.thm_dim].mean()
            self.tof_scaled_nan = scaled[0, self.imu_dim + self.thm_dim:self.imu_dim + self.thm_dim + self.tof_dim].mean()
        else:
            self.imu_scaled_nan = self.imu_scaler.transform(dummy_df[self.imu_cols])[0].mean()
            self.thm_scaled_nan = self.thm_scaler.transform(dummy_df[self.thm_cols])[0].mean()
            self.tof_scaled_nan = self.tof_scaler.transform(dummy_df[self.tof_cols])[0].mean()

    def get_scaled_nan_tensors(self, imu, thm, tof):
        return torch.full(imu.shape, self.imu_scaled_nan, device=imu.device), \
            torch.full(thm.shape, self.thm_scaled_nan, device=thm.device), \
            torch.full(tof.shape, self.tof_scaled_nan, device=tof.device)

    def inference_process(self, sequence, demographics=None, reverse=False):
        if self.config.get("use_dg", False):
            assert demographics is not None, "Demographics needed"
            df_dg = demographics.to_pandas().copy()
            df_dg['age'] /= 100
            df_dg['shoulder_to_wrist_height'] = df_dg['shoulder_to_wrist_cm'] / df_dg['height_cm']
            df_dg['elbow_to_wrist_height'] = df_dg['elbow_to_wrist_cm'] / df_dg['height_cm']
        df_seq = sequence.to_pandas().copy()
        if reverse:
            df_seq[['acc_x', 'acc_y', 'rot_x', 'rot_y']] *= -1
        if self.config.get("rot_fillna", False):
            df_seq['rot_w'] = df_seq['rot_w'].fillna(1)
            df_seq[['rot_x', 'rot_y', 'rot_z']] = df_seq[['rot_x', 'rot_y', 'rot_z']].fillna(0)
        if not all(c in df_seq.columns for c in self.imu_features):
            df_seq['acc_mag'] = np.sqrt(df_seq['acc_x'] ** 2 + df_seq['acc_y'] ** 2 + df_seq['acc_z'] ** 2)
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
            df_seq['linear_acc_mag'] = np.sqrt(
                df_seq['linear_acc_x'] ** 2 + df_seq['linear_acc_y'] ** 2 + df_seq['linear_acc_z'] ** 2)
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
                        region_data = tof_data.iloc[:, r * region_size: (r + 1) * region_size]
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
                            region_data = tof_data.iloc[:, r * region_size: (r + 1) * region_size]
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
            thm_scaled = x_scaled[..., self.imu_dim:self.imu_dim + self.thm_dim]
            tof_scaled = x_scaled[..., self.imu_dim + self.thm_dim:self.imu_dim + self.thm_dim + self.tof_dim]
        else:
            imu_scaled = self.imu_scaler.transform(imu_unscaled)
            thm_scaled = self.thm_scaler.transform(thm_unscaled)
            tof_scaled = self.tof_scaler.transform(tof_unscaled)

        combined = np.concatenate([imu_scaled, thm_scaled, tof_scaled], axis=1)
        padded = np.zeros((self.pad_len, combined.shape[1]), dtype='float32')
        seq_len = min(combined.shape[0], self.pad_len)
        padded[:seq_len] = combined[:seq_len]
        imu = padded[..., :self.imu_dim]
        thm = padded[..., self.imu_dim:self.imu_dim + self.thm_dim]
        tof = padded[..., self.imu_dim + self.thm_dim:self.imu_dim + self.thm_dim + self.tof_dim]

        ret = [torch.from_numpy(imu).float().unsqueeze(0), torch.from_numpy(thm).float().unsqueeze(0),
               torch.from_numpy(tof).float().unsqueeze(0)]
        if self.config.get("use_dg", False):
            dg = df_dg[self.dg_cols].values.astype('float32')
            ret.append(torch.from_numpy(dg).float())
        return ret

    def split5(self, imu, thm, tof):
        imus = [imu[:, :, self.global_imu_indices[k]] for k in self.global_imu_indices]
        thms = [thm[:, :, self.global_thm_indices[k]] for k in range(1, 6)]
        tofs = [tof[:, :, self.global_tof_indices[k]] for k in range(1, 6)]
        return imus, thms, tofs

    def slide(self, imu, thm, tof, ratio=1.0):
        def slide_tensor(tensor, nan_value, ratio):
            b, l, d = tensor.shape
            length = int(l * ratio)
            if length > l:
                pad = torch.full((b, length - l, d), nan_value, device=tensor.device)
                tensor = torch.cat([tensor, pad], dim=1)
            elif length < l:
                tensor = tensor[:, :length, :]
            return tensor

        return slide_tensor(imu, self.imu_scaled_nan, ratio), slide_tensor(thm, self.thm_scaled_nan, ratio), slide_tensor(tof, self.tof_scaled_nan, ratio)

    def __getitem__(self, idx):
        ret = [self.imu[idx], self.thm[idx], self.tof[idx], self.class_[idx], self.binary_class_[idx]]
        if self.config.get("return_extra", False):
            fold_feat_info = [self.fold_feats[col][idx] for col in self.fold_cols]
            ret.append((idx, fold_feat_info))
        if self.config.get("use_dg", False):
            ret.append(self.dg[idx])
        if self.config.get("return_flip_imu", False):
            ret.append(self.flip_imu[idx])
        return ret

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
        self.sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        self.fold_y = np.array(self.full_dataset.fold_feats[config.get("fold_y", "sequence_type")])
        self.fold_groups = np.array(self.full_dataset.fold_feats[config.get("fold_groups", "subject")])
        self.folds = list(self.sgkf.split(X=np.arange(len(self.full_dataset)), y=self.fold_y, groups=self.fold_groups))
        self.exclude_subjects = set(config.get("exclude_subjects", []))

    def get_fold_datasets(self, fold_idx):
        if self.folds is None or fold_idx >= self.n_folds:
            return None, None
        fold_train_idx, fold_valid_idx = self.folds[fold_idx]
        subjects = np.array(self.full_dataset.fold_feats["subject"])
        train_subjects, valid_subjects = subjects[fold_train_idx], subjects[fold_valid_idx]
        train_mask, valid_mask = ~np.isin(train_subjects, list(self.exclude_subjects)), ~np.isin(valid_subjects, list(self.exclude_subjects))
        return Subset(self.full_dataset, np.array(fold_train_idx)[train_mask].tolist()), Subset(self.full_dataset, np.array(fold_valid_idx)[valid_mask].tolist())

    def print_fold_stats(self):
        def get_label_counts(subset):
            counts = {name: 0 for name in self.class_names}
            if subset is None: return counts
            for idx in subset.indices:
                label_idx = self.full_dataset.dataset_indices[idx]
                counts[self.class_names[label_idx]] += 1
            return counts

        print("\nCross-validation fold statistics:")
        for fold_idx, (train_idx, val_idx) in enumerate(self.folds):
            train_subjects = set(self.fold_groups[train_idx])
            val_subjects = set(self.fold_groups[val_idx])
            print(f"\nFold {fold_idx + 1}:")
            print("Train subjects:", train_subjects)
            print("Valid subjects:", val_subjects)

        self.print_filtered_stats()

    def print_filtered_stats(self):
        original_counts = defaultdict(int)
        filtered_counts = defaultdict(int)

        for fold_idx in range(self.n_folds):
            train_idx, val_idx = self.folds[fold_idx]
            for idx in train_idx:
                original_counts['train'] += 1
            for idx in val_idx:
                original_counts['valid'] += 1
            train_set, val_set = self.get_fold_datasets(fold_idx)
            filtered_counts['train'] += len(train_set)
            filtered_counts['valid'] += len(val_set)

        print(f"\nDataset size changes after excluding subjects {self.exclude_subjects}:")
        print(f"Original train samples: {original_counts['train']}")
        print(f"Filtered train samples: {filtered_counts['train']}")
        print(f"Original valid samples: {original_counts['valid']}")
        print(f"Filtered valid samples: {filtered_counts['valid']}")


#-----------------------------------------------


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(torch.tensor(10000.0)) / dim))
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, dim]

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]



class TransformerBlock_CL(nn.Module):
    def __init__(self, dim, heads=8, ff_mult=4, dropout=0.1, depth=1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        #self.attn = DFattn(dim, heads, dropout=dropout, depth=depth)
        self.norm1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * ff_mult, dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)  # bidirectional: no masking
        #attn_out = self.attn(x)  # DF_ATTN self
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x


class TransformerEncoder_CL(nn.Module): # classic type
    def __init__(self, input_dim, num_layers=6, heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock_CL(input_dim, heads=heads)
            #TransformerBlock(input_dim, heads=heads, depth=i+1)
            for i in range(num_layers)
        ])

    def forward(self, x):  # x: [B, T, D]
        for layer in self.layers:
            x = layer(x)
        return x  # [B, T, D]


class TimeSeriesTransformer_CL(nn.Module):
    def __init__(self, dim_in, num_layers, heads=8):
        super().__init__()
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_in))
        self.num_register_tokens = 4
        self.reg_token = nn.Parameter(torch.randn(1, self.num_register_tokens, dim_in))
        self.pos_enc = PositionalEncoding(dim_in)
        self.encoder = TransformerEncoder_CL(dim_in, num_layers, heads=heads)

    def forward(self, x):  # x: [B, T, D]
        B = x.size(0)
        cls_tokens = self.cls_token.expand(B, 1, -1)  # [B, 1, D]
        #reg_tokens = self.reg_token.expand(B, self.num_register_tokens, -1)  # [B, 4, D]
        x = torch.cat([cls_tokens, x], dim=1)         # [B, T+1, D]
        #x = torch.cat([x, cls_tokens, reg_tokens], dim=1)         # [B, T+1, D]
        x = self.pos_enc(x)
        x = self.encoder(x)
        #return x[:, 0]  # return CLS token # torch.Size([4, 128])??
        #return x  # return ENTIRE OUTPUT
        return x[:, 1:]  # return EXCEPT CLS
        #return x[:,:-self.num_register_tokens]

#-----------------------------------------------


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, L)
        se = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # -> (B, C)
        se = F.relu(self.fc1(se), inplace=True)  # -> (B, C//r)
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)  # -> (B, C, 1)
        return x * se


class ResNetSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, wd=1e-4):
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

    def forward(self, x):
        identity = self.shortcut(x)  # (B, out, L)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)  # (B, out, L)
        out = out + identity
        return self.relu(out)


class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.score_fn = nn.Linear(feature_dim, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (B, L, F)
        score = torch.tanh(self.score_fn(x))  # (B, L, 1)
        weights = self.softmax(score.squeeze(-1))  # (B, L)
        weights = weights.unsqueeze(-1)  # (B, L, 1)
        context = x * weights  # (B, L, F)
        return context.sum(dim=1)  # (B, F)


class GaussianNoise(nn.Module):
    """Add Gaussian noise to input tensor"""

    def __init__(self, stddev):
        super().__init__()
        self.stddev = stddev

    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.stddev
            return x + noise
        return x


class CMIBackbone(nn.Module): # LSTM + GRU + DTF
    def __init__(self, imu_dim, thm_dim, tof_dim, **kwargs):
        super().__init__()
        self.imu_acc_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"],
                                        drop=kwargs["imu2_dropout"])
        )
        self.imu_rot_branch = nn.Sequential(
            self.residual_feature_block(4, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"],
                                        drop=kwargs["imu2_dropout"])
        )
        self.imu_other_branch = nn.Sequential(
            self.residual_feature_block(imu_dim - 7, kwargs["imu1_channels"], kwargs["imu1_layers"],
                                        drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"],
                                        drop=kwargs["imu2_dropout"])
        )

        self.thm_branch1, self.tof_branch1 = self.init_thm_tof_branch(thm_dim // 5, tof_dim // 5, **kwargs)
        self.thm_branch2, self.tof_branch2 = self.init_thm_tof_branch(thm_dim // 5, tof_dim // 5, **kwargs)
        self.thm_branch3, self.tof_branch3 = self.init_thm_tof_branch(thm_dim // 5, tof_dim // 5, **kwargs)
        self.thm_branch4, self.tof_branch4 = self.init_thm_tof_branch(thm_dim // 5, tof_dim // 5, **kwargs)
        self.thm_branch5, self.tof_branch5 = self.init_thm_tof_branch(thm_dim // 5, tof_dim // 5, **kwargs)

        self.imu_proj = ResNetSEBlock(in_channels=3 * kwargs["imu2_channels"], out_channels=kwargs["imu2_channels"])
        self.thm_proj = ResNetSEBlock(in_channels=5 * kwargs["thm2_channels"], out_channels=kwargs["thm2_channels"])
        self.tof_proj = ResNetSEBlock(in_channels=5 * kwargs["tof2_channels"], out_channels=kwargs["tof2_channels"])

        self.lstm = nn.LSTM(
            input_size=kwargs['imu2_channels'] + kwargs['thm2_channels'] + kwargs['tof2_channels'],
            hidden_size=kwargs['lstm_hidden_size'],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.gru = nn.GRU(
            input_size=kwargs['imu2_channels'] + kwargs['thm2_channels'] + kwargs['tof2_channels'],
            hidden_size=kwargs['gru_hidden_size'],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )

        self.noise = GaussianNoise(kwargs['gaussian_noise_rate'])
        self.dense = nn.Sequential(
            nn.Linear(kwargs['imu2_channels'] + kwargs['thm2_channels'] + kwargs['tof2_channels'], kwargs['dense_channels']),
            nn.ELU()
        )

        # final_output = self.attn(torch.cat([lstm_out, gru_out, dense_out], dim=-1))
        #self.dtf = TimeSeriesTransformer((kwargs['lstm_hidden_size'] + kwargs['gru_hidden_size']) * 2 + kwargs['dense_channels'], num_layers=6, heads=8)
        #self.dtf = TimeSeriesTransformer_CL((kwargs['lstm_hidden_size'] + kwargs['gru_hidden_size']) * 2 + kwargs['dense_channels'], num_layers=6, heads=2)

        #self.featconversion = nn.Linear(54, 53)
        self.dtf = TimeSeriesTransformer_CL(kwargs['dense_channels'], num_layers=6, heads=2)
        #print(f"kwargs['imu2_channels'] + kwargs['thm2_channels'] + kwargs['tof2_channels'] = {kwargs['imu2_channels'] + kwargs['thm2_channels'] + kwargs['tof2_channels']}")
        #print(f"kwargs['(kwargs['lstm_hidden_size'] + kwargs['gru_hidden_size']) * 2 + kwargs['dense_channels'] = {(kwargs['lstm_hidden_size'] + kwargs['gru_hidden_size']) * 2 + kwargs['dense_channels']}")

        self.attn = AttentionLayer(feature_dim=(kwargs['lstm_hidden_size'] + kwargs['gru_hidden_size']) * 2 + kwargs['dense_channels'] * 2)  # lstm + gru + dense

    def feature_block(self, in_channels, out_channels, num_layers, pool_size=2, drop=0.3):
        return nn.Sequential(
            *[ResNetSEBlock(in_channels=in_channels, out_channels=in_channels) for i in range(num_layers)],
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size, ceil_mode=True),
            nn.Dropout(drop)
        )

    def residual_feature_block(self, in_channels, out_channels, num_layers, pool_size=2, drop=0.3):
        return nn.Sequential(
            *[ResNetSEBlock(in_channels=in_channels, out_channels=in_channels) for i in range(num_layers)],
            ResNetSEBlock(in_channels, out_channels, wd=1e-4),
            nn.MaxPool1d(pool_size, ceil_mode=True),
            nn.Dropout(drop)
        )

    def init_thm_tof_branch(self, thm_dim, tof_dim, **kwargs):
        thm_branch = nn.Sequential(
            self.feature_block(thm_dim, kwargs["thm1_channels"], kwargs["thm1_layers"], drop=kwargs["thm1_dropout"]),
            self.feature_block(kwargs["thm1_channels"], kwargs["thm2_channels"], kwargs["thm2_layers"],
                               drop=kwargs["thm2_dropout"]),
        )
        tof_branch = nn.Sequential(
            self.feature_block(tof_dim, kwargs["tof1_channels"], kwargs["tof1_layers"], drop=kwargs["tof1_dropout"]),
            self.feature_block(kwargs["tof1_channels"], kwargs["tof2_channels"], kwargs["tof2_layers"],
                               drop=kwargs["tof2_dropout"]),
        )
        return thm_branch, tof_branch

    def forward(self, imus, thms, tofs):
        #print(imus)
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

        tf_out = self.dtf(dense_out)

        #print(f"shapes : {feat.shape},{lstm_out.shape},{gru_out.shape},{dense_out.shape}")
        # shapes : torch.Size([64, 53, 832]),torch.Size([64, 53, 256]),torch.Size([64, 53, 256]),torch.Size([64, 53, 32])

        #tf_out = self.featconversion(dense_out.transpose(-1,-2)).transpose(-1,-2)

        #dtf_out = self.dtf(torch.cat([lstm_out, gru_out, dense_out], dim=-1))
        final_output = self.attn(torch.cat([lstm_out, gru_out, dense_out, tf_out], dim=-1))
        #final_output = self.attn(dtf_out)

        return final_output
        #return self.attn(torch.cat([lstm_out, gru_out, dense_out], dim=-1))


class CMIModel(nn.Module):
    def __init__(self, target_classes_num, non_target_classes_num, **kwargs):
        super().__init__()
        self.backbone = CMIBackbone(dataset.imu_dim, dataset.thm_dim, dataset.tof_dim, **kwargs)
        self.target_classifier = nn.Sequential(
            nn.Linear((kwargs['lstm_hidden_size'] + kwargs['gru_hidden_size']) * 2 + kwargs['dense_channels'] * 2, kwargs["cls_channels1"]),
            nn.BatchNorm1d(kwargs["cls_channels1"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout1"]),
            nn.Linear(kwargs["cls_channels1"], kwargs["cls_channels2"]),
            nn.BatchNorm1d(kwargs["cls_channels2"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout2"]),
            nn.Linear(kwargs["cls_channels2"], target_classes_num)
        )
        self.non_target_classifier = nn.Sequential(
            nn.Linear((kwargs['lstm_hidden_size'] + kwargs['gru_hidden_size']) * 2 + kwargs['dense_channels'] * 2, kwargs["cls_channels1"]),
            nn.BatchNorm1d(kwargs["cls_channels1"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout1"]),
            nn.Linear(kwargs["cls_channels1"], kwargs["cls_channels2"]),
            nn.BatchNorm1d(kwargs["cls_channels2"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout2"]),
            nn.Linear(kwargs["cls_channels2"], non_target_classes_num)
        )

    def forward(self, imu, thm, tof):
        feat = self.backbone(imu, thm, tof)
        targets_y = self.target_classifier(feat)
        non_targets_y = self.non_target_classifier(feat)
        return torch.cat([targets_y, non_targets_y], dim=1)


def replace(k):
    k = k.replace("_orig_mod.", "")
    return k

# ---------------------------
# METRICS
class ParticipantVisibleError(Exception):
    """Errors raised here will be shown directly to the competitor."""
    pass


class Metric:
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
        self.gts = dict()
        self.preds = dict()
        self.gts_p = dict()
        self.preds_p = dict()
        self.auto_idx = 0
        self.auto_idx_p = 0

    def add(self, gts, preds, ids=None):
        if ids is None:
            ids = range(self.auto_idx, self.auto_idx + len(gts))
            self.auto_idx += len(gts)
        for gt, pred, id_ in zip(gts, preds, ids):
            if torch.is_tensor(id_):
                id_ = id_.item()
            self.gts[id_] = gt
            self.preds[id_] = pred

    def add_p(self, gts_p, preds_p, ids=None):
        if ids is None:
            ids = range(self.auto_idx_p, self.auto_idx_p + len(gts_p))
            self.auto_idx_p += len(gts_p)
        for gt_p, pred_p, id_ in zip(gts_p, preds_p, ids):
            if torch.is_tensor(id_):
                id_ = id_.item()
            self.gts_p[id_] = gt_p
            self.preds_p[id_] = pred_p

    def merge_p(self, *metrics):
        for metric in metrics:
            assert set(self.gts_p.keys()) == set(metric.gts_p.keys()), "GT not same, plz check"
            for id_ in self.gts_p.keys():
                assert np.allclose(self.gts_p[id_], metric.gts_p[id_]), f"GT values not same for ID {id_}"

        all_ids = list(self.preds_p.keys())
        num_classes = len(next(iter(self.preds_p.values())))
        combined = np.zeros((len(metrics) + 1, len(all_ids), num_classes))
        for i, id_ in enumerate(all_ids):
            combined[0, i] = self.preds_p[id_]
        for m_idx, metric in enumerate(metrics, 1):
            for i, id_ in enumerate(all_ids):
                combined[m_idx, i] = metric.preds_p[id_]
        avg_preds = combined.mean(axis=0)
        for i, id_ in enumerate(all_ids):
            self.preds_p[id_] = avg_preds[i].tolist()

    def apply_weight(self, weight):
        all_preds = np.array(list(self.preds_p.values()))
        weighted = all_preds * weight
        for id_, pred in zip(self.preds_p.keys(), weighted):
            self.preds_p[id_] = pred.tolist()

    def merge_vote(self, *metrics):
        for metric in metrics:
            assert set(self.gts.keys()) == set(metric.gts.keys()), "GT IDs not same, please check"
            for id_ in self.gts.keys():
                assert self.gts[id_] == metric.gts[id_], f"GT values not same for ID {id_}"

        id_to_preds = defaultdict(list)
        for metric in [self] + list(metrics):
            id_to_preds.update(metric.preds)
        vote_results = {}
        for id_, preds in id_to_preds.items():
            from collections import Counter
            counts = Counter(preds)
            max_count = max(counts.values())
            candidates = [k for k, v in counts.items() if v == max_count]
            vote_results[id_] = candidates[0] if len(candidates) == 1 else preds[0]
        self.preds.update(vote_results)

    def clear(self):
        self.gts.clear()
        self.preds.clear()
        self.gts_p.clear()
        self.preds_p.clear()
        self.auto_idx = 0
        self.auto_idx_p = 0

    def score(self):
        sorted_ids = sorted(self.gts.keys())
        gts_list = [self.gts[id_] for id_ in sorted_ids]
        preds_list = [self.preds[id_] for id_ in sorted_ids]
        gts_pd = pd.Series(gts_list)
        preds_pd = pd.Series(preds_list)
        y_true_bin = gts_pd.isin(self.target_gestures).values
        y_pred_bin = preds_pd.isin(self.target_gestures).values
        f1_binary = f1_score(y_true_bin, y_pred_bin, pos_label=True, zero_division=0, average='binary')
        y_true_mc = gts_pd.apply(lambda x: x if x in self.target_gestures else 'non_target')
        y_pred_mc = preds_pd.apply(lambda x: x if x in self.target_gestures else 'non_target')
        f1_macro = f1_score(y_true_mc, y_pred_mc, average='macro', zero_division=0)
        return 0.5 * f1_binary + 0.5 * f1_macro

    def acc(self):
        sorted_ids = sorted(self.gts.keys())
        gts = np.array([self.gts[id_] for id_ in sorted_ids])
        preds = np.array([self.preds[id_] for id_ in sorted_ids])
        return np.sum(preds == gts) / len(gts)

    def score_p(self, classes):
        sorted_ids = sorted(self.gts_p.keys())
        gts_list = [classes[np.argmax(self.gts_p[id_])] for id_ in sorted_ids]
        preds_list = [classes[np.argmax(self.preds_p[id_])] for id_ in sorted_ids]
        gts_pd = pd.Series(gts_list)
        preds_pd = pd.Series(preds_list)
        y_true_bin = gts_pd.isin(self.target_gestures).values
        y_pred_bin = preds_pd.isin(self.target_gestures).values
        f1_binary = f1_score(y_true_bin, y_pred_bin, pos_label=True, zero_division=0, average='binary')
        y_true_mc = gts_pd.apply(lambda x: x if x in self.target_gestures else 'non_target').values
        y_pred_mc = preds_pd.apply(lambda x: x if x in self.target_gestures else 'non_target').values
        f1_macro = f1_score(y_true_mc, y_pred_mc, average='macro', zero_division=0)
        return 0.5 * f1_binary + 0.5 * f1_macro

    def calculate_hierarchical_f1(self, sol: pd.DataFrame, sub: pd.DataFrame) -> float:
        # Validate gestures
        invalid_types = {i for i in sub['gesture'].unique() if i not in self.all_classes}
        if invalid_types:
            raise ParticipantVisibleError(f"Invalid gesture values in submission: {invalid_types}")

        # Compute binary F1 (Target vs Non-Target)
        y_true_bin = sol['gesture'].isin(self.target_gestures).values
        y_pred_bin = sub['gesture'].isin(self.target_gestures).values
        f1_binary = f1_score(
            y_true_bin,
            y_pred_bin,
            pos_label=True,
            zero_division=0,
            average='binary'
        )

        # Build multi-class labels for gestures
        y_true_mc = sol['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        y_pred_mc = sub['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')

        # Compute macro F1 over all gesture classes
        f1_macro = f1_score(
            y_true_mc,
            y_pred_mc,
            average='macro',
            zero_division=0
        )

        # return 0.5 * f1_binary + 0.5 * f1_macro
        return f1_binary, f1_macro


# ---------------------------

def to_cuda(*tensors):
    return [tensor.to(CUDA0) for tensor in tensors]


def inference(model, imu, thm, tof):
    imus, thms, tofs = dataset.full_dataset.split5(imu, thm, tof)
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        pred_y = model(imus, thms, tofs)
    return pred_y


def valid(model, valid_bar):
    with torch.no_grad():
        for imu, thm, tof, y, b in valid_bar:
            imu, thm, tof, y = to_cuda(imu, thm, tof, y)
            pred_y = inference(model, imu, thm, tof)
            metric.add(dataset.le.classes_[y.argmax(dim=1).cpu()], dataset.le.classes_[pred_y.argmax(dim=1).cpu()])
            _, thm, tof = dataset.full_dataset.get_scaled_nan_tensors(imu, thm, tof)
            pred_y = inference(model, imu, thm, tof)
            imu_only_metric.add(dataset.le.classes_[y.argmax(dim=1).cpu()], dataset.le.classes_[pred_y.argmax(dim=1).cpu()])



# ---------------------------


if not TRAIN:
    dataset = init_dataset()
    metric = Metric()
    imu_only_metric = Metric()

    model_function = CMIModel
    model_dicts = [
        {
            "model_function": model_function,
            "model_args": model_args,
            "model_path": model_dir / f"model_fold{fold}.pt",
        } for fold in range(n_folds)
    ]
    models = list()

    for model_dict in model_dicts:
        model_function = model_dict["model_function"]
        model_args = model_dict["model_args"]
        model_path = model_dict["model_path"]
        model = model_function(**model_args).to(CUDA0)
        state_dict = {replace(k): v for k, v in torch.load(model_path).items()}
        model.load_state_dict(state_dict)
        model = model.eval()
        models.append(model)


print("DONE!")


def avg_predict(models, imu, thm, tof):
    outputs = []
    with autocast(device_type='cuda', dtype=torch.bfloat16):
        for model in models:
            pred_y = inference(model, imu, thm, tof)
            outputs.append(pred_y)
    return torch.mean(torch.stack(outputs), dim=0)


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    imu, thm, tof = dataset.full_dataset.inference_process(sequence)
    with torch.no_grad():
        imu, thm, tof = to_cuda(imu, thm, tof)
        if imu_only:
            _, thm, tof = dataset.full_dataset.get_scaled_nan_tensors(imu, thm, tof)
        pred_y = avg_predict(models, imu, thm, tof)
    return dataset.le.classes_[pred_y.argmax(dim=1).cpu()]


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

