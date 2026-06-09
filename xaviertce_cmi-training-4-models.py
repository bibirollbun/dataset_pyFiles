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
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from numpy.fft import fft
from transformers import DebertaV2Model, DebertaV2Config
from transformers import BertConfig, BertModel
from transformers import DebertaV2Model, DebertaV2Config, AdamW, get_linear_schedule_with_warmup
from transformers import RobertaModel, RobertaConfig
import warnings
import gc
import torch
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


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
            angular_dist[i] = 0 # В случае недействительных кватернионов
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
        self.rot_features = ['rot_angle', 'rot_angle_vel', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance']
        self.old_imu_features = [
            'acc_mag', 'rot_angle','acc_mag_jerk', 'rot_angle_vel',
            'linear_acc_mag', 'linear_acc_mag_jerk',
            'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance'
        ]

        self.acc_features = ['acc_mag', 'acc_mag_jerk', 'linear_acc_mag', 'linear_acc_mag_jerk']
        self.rot_features = ['rot_angle', 'rot_angle_vel', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
                             'angular_distance']
        self.old_imu_features = [
            'acc_mag', 'rot_angle', 'acc_mag_jerk', 'rot_angle_vel',
            'linear_acc_mag', 'linear_acc_mag_jerk',
            'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance'
        ]
        # "acc_xy_magnitude","acc_xz_magnitude","acc_yz_magnitude",'acc_xy_sum'
        self.sum_imu_features = ['acc_yz_sum','acc_xy_sum','acc_xz_sum']
        # 'acc_xz_sum','acc_xy_diff','acc_yz_diff','acc_xz_diff','acc_xy_z_net','acc_yz_x_net',   'acc_xz_y_net'
        # rot_xy_z_net  rot_xw_y_net_md rot_xw_z_net_md rot_yw_x_net_md rot_xy_z_net_md rot_yz_x_net_md rot_zw_y_net_md rot_xz_over_yw rot_xw_over_yz
        self.ms_rot_features = ["rot_yz_x_net","rot_xz_y_net","rot_xw_z_net"]
        # "rot_yw_x_net","rot_xw_y_net",'rot_xw_z_net_ms',"rot_yw_x_net_ms","rot_xw_y_net_ms","rot_xy_z_net_ms","rot_yz_x_net_ms","rot_zw_y_net_ms"
        self.md_rot_features = ["rot_xy_plus_zw","rot_xz_plus_yw","rot_xw_plus_yz"]

        self.jerk = ["angular_jerk_x",
                    "angular_jerk_y",
                    "angular_jerk_z"]

        self.rot_magnitude = [
				"rot_xyz_magnitude",
				"rot_xyw_magnitude",
				"rot_xzw_magnitude",
				"rot_yzw_magnitude",
				"rot_full_magnitude"
				]

        self.rolling_mean = [
				"acc_mag_smooth_mean",
				"linear_acc_mean",
				"rot_full_magnitude_mean"]

        self.rot_jerk = [
                        "rot_w_vel",
                        "rot_x_vel",
                        "rot_y_vel",
                        "rot_z_vel"
                            ]
        self.rot_xyz = ["rot_xy_over_zw","rot_xz_over_yw","rot_xw_over_yz"]        

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
        self.imu_acc_cols_base = ['acc_x', 'acc_y', 'acc_z', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z'] if self.config.get("add_raw_acc", False) else ['linear_acc_x', 'linear_acc_y', 'linear_acc_z']
        self.imu_rot_cols_base = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
        self.imu_cols_base = self.imu_acc_cols_base + self.imu_rot_cols_base + self.sum_imu_features +\
                                self.ms_rot_features + self.md_rot_features + self.jerk + self.rot_magnitude +\
                                self.rolling_mean + self.rot_jerk + self.rot_xyz
        self.imu_cols = list()
        self.imu_channel_keys = defaultdict(list)
        if self.config.get("add_imu_base", True): 
            self.imu_cols.extend(self.imu_cols_base)
            self.imu_channel_keys["acc"] = self.imu_acc_cols_base
            self.imu_channel_keys["rot"] = self.imu_rot_cols_base
            self.imu_channel_keys["sum"] = self.sum_imu_features
            self.imu_channel_keys["ms"] = self.ms_rot_features
            self.imu_channel_keys["md"] = self.md_rot_features
            self.imu_channel_keys["jerk"] = self.jerk
            self.imu_channel_keys["rot_magnitude"] = self.rot_magnitude
            self.imu_channel_keys["roll_mean"] = self.rolling_mean
            self.imu_channel_keys["angular_jerk"] = self.rot_jerk
            self.imu_channel_keys["rot_xyz"] = self.rot_xyz
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
        self.thm_cols = [c for c in self.raw_columns if c.startswith('thm_')] + ['thm_right_chain','thm_left_chain','thm_center_chain','thm_height_chain','thm_weight_chain']
        self.thm_cols1 = [c for c in self.raw_columns if c.startswith('thm_')] #+ ['thm_right_chain','thm_left_chain','thm_center_chain','thm_height_chain','thm_weight_chain']
        self.thm_cols2 = ['thm_right_chain','thm_left_chain','thm_center_chain','thm_height_chain','thm_weight_chain']
        self.thm_channel_keys = {k+1: [a,b] for k,(a,b) in enumerate(zip(self.thm_cols1,self.thm_cols2))}  #{k: [f"thm_{k}"] for k in range(1, 6)}
        self.feature_cols = self.imu_cols + self.thm_cols + self.tof_cols
        self.imu_dim = len(self.imu_cols)
        self.thm_dim = len(self.thm_cols)
        self.tof_dim = len(self.tof_cols)
        self.base_cols = ['acc_x', 'acc_y', 'acc_z',
                          'rot_x', 'rot_y', 'rot_z', 'rot_w',
                          'sequence_id', 'subject', 
                          'sequence_type', 'gesture', 'orientation'] + [c for c in self.raw_columns if c.startswith('thm_')] + [f"tof_{i}_v{p}" for i in range(1, 6) for p in range(64)]
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
        axes=['x', 'y', 'z']
        features = {}
        for axis in axes:
            fft_result = fft(df[f'acc_{axis}'].values)
            energy = np.sum(np.abs(fft_result)**2)
            features[f"er_{axis}"] = energy
        for i, axis1 in enumerate(axes):
            for axis2 in axes[i+1:]:
                features[f'er_r_{axis1}{axis2}'] = features[f'er_{axis1}'] / (features[f'er_{axis2}'] + 1e-6)
        for i, axis1 in enumerate(axes):
            for axis2 in axes[i+1:]:
                features[f'er_c_{axis1}{axis2}'] = np.corrcoef(np.abs(fft(df[f'acc_{axis1}'].values)), np.abs(fft(df[f'acc_{axis2}'].values)))[0, 1]
        return {k: v for k, v in features.items() if k in self.er_fearues}

    def compute_imu_features(self, df):
        if self.config.get("rot_fillna", False):
            df['rot_w'] = df['rot_w'].fillna(1)
            df[['rot_x', 'rot_y', 'rot_z']] = df[['rot_x', 'rot_y', 'rot_z']].fillna(0)
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

        # linear_acc_x linear_acc_y linear_acc_z
        ##################################modified features#####################################################
        df["acc_xy_magnitude"] = np.sqrt(df["linear_acc_x"] ** 2 + df["linear_acc_y"] ** 2)
        df["acc_xz_magnitude"] = np.sqrt(df["linear_acc_x"] ** 2 + df["linear_acc_y"] ** 2)
        df["acc_yz_magnitude"] = np.sqrt(df["linear_acc_y"] ** 2 + df["linear_acc_z"] ** 2)

        # Additive combinations
        df["acc_xy_sum"] = df["linear_acc_x"] + df["linear_acc_y"]
        df["acc_yz_sum"] = df["linear_acc_y"] + df["linear_acc_z"]
        df["acc_xz_sum"] = df["linear_acc_x"] + df["linear_acc_z"]

        # Subtractive combinations
        df["acc_xy_diff"] = df["linear_acc_x"] - df["linear_acc_y"]
        df["acc_yz_diff"] = df["linear_acc_y"] - df["linear_acc_z"]
        df["acc_xz_diff"] = df["linear_acc_x"] - df["linear_acc_z"]

        # Triple-axis interactions
        df["acc_xy_z_net"] = df["linear_acc_x"] + df["linear_acc_y"] - df["linear_acc_z"]
        df["acc_yz_x_net"] = df["linear_acc_y"] + df["linear_acc_z"] - df["linear_acc_x"]
        df["acc_xz_y_net"] = df["linear_acc_x"] + df["linear_acc_z"] - df["linear_acc_y"]

####################################################################################
        # df["acc_xy_magnitude"] = np.sqrt(df["acc_x"] ** 2 + df["acc_y"] ** 2)
        # df["acc_xz_magnitude"] = np.sqrt(df["acc_x"] ** 2 + df["acc_z"] ** 2)
        # df["acc_yz_magnitude"] = np.sqrt(df["acc_y"] ** 2 + df["acc_z"] ** 2)

        # # Additive combinations
        # df["acc_xy_sum"] = df["acc_x"] + df["acc_y"]
        # df["acc_yz_sum"] = df["acc_y"] + df["acc_z"]
        # df["acc_xz_sum"] = df["acc_x"] + df["acc_z"]

        # # Subtractive combinations
        # df["acc_xy_diff"] = df["acc_x"] - df["acc_y"]
        # df["acc_yz_diff"] = df["acc_y"] - df["acc_z"]
        # df["acc_xz_diff"] = df["acc_x"] - df["acc_z"]

        # # Triple-axis interactions
        # df["acc_xy_z_net"] = df["acc_x"] + df["acc_y"] - df["acc_z"]
        # df["acc_yz_x_net"] = df["acc_y"] + df["acc_z"] - df["acc_x"]
        # df["acc_xz_y_net"] = df["acc_x"] + df["acc_z"] - df["acc_y"]
###############################################################################        

        # Linear combinations 
        df["rot_xy_z_net"] = df["rot_x"] + df["rot_y"] - df["rot_z"] 
        df["rot_yz_x_net"] = df["rot_y"] + df["rot_z"] - df["rot_x"]
        df["rot_xz_y_net"] = df["rot_x"] + df["rot_z"] - df["rot_y"]

        df["rot_xw_z_net"] = df["rot_x"] + df["rot_w"] - df["rot_z"]
        df["rot_yw_x_net"] = df["rot_y"] + df["rot_w"] - df["rot_x"]
        df["rot_xw_y_net"] = df["rot_x"] + df["rot_w"] - df["rot_y"]

        # Multiplicative + subtractive
        df["rot_xw_z_net_ms"] = df["rot_x"] * df["rot_w"] - df["rot_z"]
        df["rot_yw_x_net_ms"] = df["rot_y"] * df["rot_w"] - df["rot_x"]
        df["rot_xw_y_net_ms"] = df["rot_x"] * df["rot_w"] - df["rot_y"]

        df["rot_xy_z_net_ms"] = df["rot_x"] * df["rot_y"] - df["rot_z"]
        df["rot_yz_x_net_ms"] = df["rot_y"] * df["rot_z"] - df["rot_x"]
        df["rot_zw_y_net_ms"] = df["rot_z"] * df["rot_w"] - df["rot_y"]

        # Division + subtractive
        df["rot_xw_z_net_md"] = df["rot_x"] / df["rot_w"] - df["rot_z"]
        df["rot_yw_x_net_md"] = df["rot_y"] / df["rot_w"] - df["rot_x"]
        df["rot_xw_y_net_md"] = df["rot_x"] / df["rot_w"] - df["rot_y"]

        df["rot_xy_z_net_md"] = df["rot_x"] / df["rot_y"] - df["rot_z"]
        df["rot_yz_x_net_md"] = df["rot_y"] / df["rot_z"] - df["rot_x"]
        df["rot_zw_y_net_md"] = df["rot_z"] / df["rot_w"] - df["rot_y"]

        # Pairwise interactions
        df["rot_xy_plus_zw"] = df["rot_x"] * df["rot_y"] + df["rot_z"] * df["rot_w"]
        df["rot_xz_plus_yw"] = df["rot_x"] * df["rot_z"] + df["rot_y"] * df["rot_w"]
        df["rot_xw_plus_yz"] = df["rot_x"] * df["rot_w"] + df["rot_y"] * df["rot_z"]

        # Ratios with numerical stability
        df["rot_xy_over_zw"] = (df["rot_x"] + df["rot_y"]) / (df["rot_z"] + df["rot_w"] + 1e-6)
        df["rot_xz_over_yw"] = (df["rot_x"] + df["rot_z"]) / (df["rot_y"] + df["rot_w"] + 1e-6)
        df["rot_xw_over_yz"] = (df["rot_x"] + df["rot_w"]) / (df["rot_y"] + df["rot_z"] + 1e-6)

        # Angular jerk (difference over sequence_id)
        df["angular_jerk_x"] = df.groupby("sequence_id")["angular_vel_x"].diff()
        df["angular_jerk_y"] = df.groupby("sequence_id")["angular_vel_y"].diff()
        df["angular_jerk_z"] = df.groupby("sequence_id")["angular_vel_z"].diff()
        
        # Rotation magnitudes
        df["rot_xyz_magnitude"] = np.sqrt(df["rot_x"]**2 + df["rot_y"]**2 + df["rot_z"]**2)
        df["rot_xyw_magnitude"] = np.sqrt(df["rot_x"]**2 + df["rot_y"]**2 + df["rot_w"]**2)
        df["rot_xzw_magnitude"] = np.sqrt(df["rot_x"]**2 + df["rot_z"]**2 + df["rot_w"]**2)
        df["rot_yzw_magnitude"] = np.sqrt(df["rot_y"]**2 + df["rot_z"]**2 + df["rot_w"]**2)
        df["rot_full_magnitude"] = np.sqrt(df["rot_x"]**2 + df["rot_y"]**2 + df["rot_z"]**2 + df["rot_w"]**2)
        
        # Rolling means over sequence_id
        df["acc_mag_smooth_mean"] = df.groupby("sequence_id")["acc_mag"].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        df["linear_acc_mean"] = df.groupby("sequence_id")["linear_acc_mag"].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
        df["rot_full_magnitude_mean"] = df.groupby("sequence_id")["rot_full_magnitude"].transform(lambda x: x.rolling(window=5, min_periods=1).mean()) 

        # df["angular_jerk_x"] = df.groupby("sequence_id")["angular_vel_x"].diff()
        # df["angular_jerk_y"] = df.groupby("sequence_id")["angular_vel_y"].diff()
        # df["angular_jerk_z"] = df.groupby("sequence_id")["angular_vel_z"].diff()

        df["rot_xy_over_zw"] = ((df["rot_x"] + df["rot_y"]) / (df["rot_z"] + df["rot_w"]+ 1e-6)).clip(-47.0799630, 48.1292149256)
        
        # rot_xz_over_yw
        df["rot_xz_over_yw"] = ((df["rot_x"] + df["rot_z"]) / (df["rot_y"] + df["rot_w"]+ 1e-6)).clip(-47.0799630, 48.1292149256)
        
        # rot_xw_over_yz
        df["rot_xw_over_yz"] = ((df["rot_x"] + df["rot_w"]) / (df["rot_y"] + df["rot_z"]+ 1e-6)).clip(-47.0799630, 48.1292149256) 

        df["rot_w_vel"] = df.groupby("sequence_id")["rot_w"].diff()
        df["rot_x_vel"] = df.groupby("sequence_id")["rot_x"].diff()
        df["rot_y_vel"] = df.groupby("sequence_id")["rot_y"].diff()
        df["rot_z_vel"] = df.groupby("sequence_id")["rot_z"].diff()

        def mean_abs_chain(diffs):
            return sum(d.abs() for d in diffs) / len(diffs)
        
        df['thm_right_chain'] = mean_abs_chain([
            df['thm_3'] - df['thm_2'],
            df['thm_3'] - df['thm_1'],
            df['thm_3'] - df['thm_4'],
        ])
        
        df['thm_left_chain'] = mean_abs_chain([
            df['thm_5'] - df['thm_2'],
            df['thm_5'] - df['thm_1'],
            df['thm_5'] - df['thm_4'],
        ])
        
        df['thm_center_chain'] = mean_abs_chain([
            df['thm_1'] - df['thm_2'],
            df['thm_1'] - df['thm_3'],
            df['thm_1'] - df['thm_4'],
            df['thm_1'] - df['thm_5'],
        ])
        
        df['thm_height_chain'] = mean_abs_chain([
            df['thm_2'] - df['thm_1'],
            df['thm_1'] - df['thm_4'],
        ])
        
        df['thm_weight_chain'] = mean_abs_chain([
            df['thm_4'] - df['thm_1'],
            df['thm_3'] - df['thm_2'],
        ])
        
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
            x_unscaled = [np.concatenate([imu, thm, tof], axis=1) for imu, thm, tof in zip(imu_unscaled, thm_unscaled, tof_unscaled)]
            x_scaled, self.x_scaler = self.scale(x_unscaled)
            x = self.pad(x_scaled, self.imu_cols+self.thm_cols+self.tof_cols)
            self.imu = x[..., :self.imu_dim]
            self.thm = x[..., self.imu_dim:self.imu_dim+self.thm_dim]
            self.tof = x[..., self.imu_dim+self.thm_dim:self.imu_dim+self.thm_dim+self.tof_dim]

            if self.config.get("return_flip_imu", False):
                flip_x_unscaled = [np.concatenate([flip_imu, thm, tof], axis=1) for flip_imu, thm, tof in zip(flip_imu_unscaled, thm_unscaled, tof_unscaled)]
                flip_x_scaled = [self.x_scaler.transform(x) for x in flip_x_unscaled]
                flip_x = self.pad(flip_x_scaled, self.imu_cols+self.thm_cols+self.tof_cols)
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
        self.class_ = F.one_hot(torch.from_numpy(np.array(classes)).long(), num_classes=len(self.le.classes_)).float().numpy()
        self.binary_class_ = np.isin(np.array(classes), self.target_ints).astype(np.float32)
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

        ret = [torch.from_numpy(imu).float().unsqueeze(0), torch.from_numpy(thm).float().unsqueeze(0), torch.from_numpy(tof).float().unsqueeze(0)]
        if self.config.get("use_dg", False):
            dg = df_dg[self.dg_cols].values.astype('float32')
            ret.append(torch.from_numpy(dg).float())
        return ret

    def split5(self, imu, thm, tof):
        imus = [imu[:, self.global_imu_indices[k]] for k in self.global_imu_indices]
        thms = [thm[:, self.global_thm_indices[k]] for k in range(1, 6)]
        tofs = [tof[:, self.global_tof_indices[k]] for k in range(1, 6)]
        return imus, thms, tofs

    def slide(self, imu, thm, tof, ratio=1.0):
        def slide_tensor(tensor, nan_value, ratio):
            b, l, d = tensor.shape
            length = int(l * ratio)
            if length > l:
                pad = torch.full((b, length-l, d), nan_value, device=tensor.device)
                tensor = torch.cat([tensor, pad], dim=1)
            elif length < l:
                tensor = tensor[:, :length, :] 
            return tensor
        return slide_tensor(imu, self.imu_scaled_nan, ratio), slide_tensor(thm, self.thm_scaled_nan, ratio), slide_tensor(tof, self.tof_scaled_nan, ratio)

    def __getitem__(self, idx):
        imus, thms, tofs = self.split5(self.imu[idx], self.thm[idx], self.tof[idx])
        ret = [imus, thms, tofs, self.class_[idx], self.binary_class_[idx]]
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
        if self.folds is None or fold_idx >= self.n_folds: return None, None
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
        
        print("\n交叉验证折叠统计:")
        for fold_idx in range(self.n_folds):
            train_fold, valid_fold = self.get_fold_datasets(fold_idx)
            train_counts = get_label_counts(train_fold)
            valid_counts = get_label_counts(valid_fold)
            print(f"\nFold {fold_idx + 1}:")
            print(f"{'类别':<50} {'训练集':<10} {'验证集':<10}")
            for name in self.class_names:
                print(f"{name:<50} {train_counts[name]:<10} {valid_counts[name]:<10}")

        for fold_idx, (train_idx, val_idx) in enumerate(self.folds):
            train_subjects = set(self.fold_groups[train_idx])
            val_subjects = set(self.fold_groups[val_idx])
            print(f"\nFold {fold_idx + 1}:")
            print("训练集受试者:", train_subjects)
            print("验证集受试者:", val_subjects)

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
        
        print(f"\n排除subject {self.exclude_subjects} 后的数据量变化:")
        print(f"原始训练集样本: {original_counts['train']}")
        print(f"过滤后训练集样本: {filtered_counts['train']}")
        print(f"原始验证集样本: {original_counts['valid']}") 
        print(f"过滤后验证集样本: {filtered_counts['valid']}")


CUDA0 = "cuda:0"
seed = 0
batch_size = 64
num_workers = 4
n_folds = 5

root_dir = Path("./")
universe_csv_path = Path("./train.csv")


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

dataset = init_dataset()





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

config = DebertaV2Config(
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    intermediate_size=1024,
    max_position_embeddings=512,
    relative_attention=True,
    position_biased_input=False,
    position_buckets=256,
    norm_rel_ebd="layer_norm",
    share_att_key=True,
    pos_att_type="p2c|c2p",
    layer_norm_eps=1e-7,
)

     
class CMIModel_deberta_dp_aux(nn.Module): 
    def __init__(self,**kwargs):
        super().__init__()
        self.imu_branch_linear = nn.Sequential(
            self.residual_se_cnn_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"],
                                       drop=kwargs["imu1_dropout"]),
            self.residual_se_cnn_block(kwargs["imu1_channels"], kwargs["feat_dim"], kwargs["imu2_layers"],
                                       drop=kwargs["imu2_dropout"])
        )

        self.imu_branch_rot = nn.Sequential(
            self.residual_se_cnn_block(4, kwargs["imu1_channels"], kwargs["imu1_layers"],
                                       drop=kwargs["imu1_dropout"]),
            self.residual_se_cnn_block(kwargs["imu1_channels"], kwargs["feat_dim"], kwargs["imu2_layers"],
                                       drop=kwargs["imu2_dropout"])
        )

        self.imu_branch_other = nn.Sequential(
            self.residual_se_cnn_block(10, kwargs["imu1_channels"], kwargs["imu1_layers"],
                                       drop=kwargs["imu1_dropout"]),
            self.residual_se_cnn_block(kwargs["imu1_channels"], kwargs["feat_dim"], kwargs["imu2_layers"],
                                       drop=kwargs["imu2_dropout"])
        )        
    
        self.thm_branch = nn.Sequential(
            nn.Conv1d(kwargs["thm_dim"], kwargs["thm1_channels"], kernel_size=3, padding=1, bias=False),
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
            nn.Conv1d(kwargs["tof_dim"], kwargs["tof1_channels"], kernel_size=3, padding=1, bias=False),
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

        self.deberta = DebertaV2Model(config)

        self.aux_classifier = nn.Linear(kwargs["feat_dim"], 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(kwargs["feat_dim"]*2, kwargs["cls1_channels"], bias=False),
            nn.BatchNorm1d(kwargs["cls1_channels"]),
            nn.ReLU(inplace=True),
            nn.Dropout(kwargs["cls1_dropout"]),
            nn.Linear(kwargs["cls1_channels"], kwargs["cls2_channels"], bias=False),
            nn.BatchNorm1d(kwargs["cls2_channels"]),
            nn.ReLU(inplace=True),
            nn.Dropout(kwargs["cls2_dropout"]),
            nn.Linear(kwargs["cls2_channels"], kwargs["n_classes"])
        )

        
    
    def residual_se_cnn_block(self, in_channels, out_channels, num_layers, pool_size=2, drop=0.3, wd=1e-4):
        return nn.Sequential(
            *[ResNetSEBlock(in_channels=in_channels, out_channels=in_channels) for i in range(num_layers)],
            ResNetSEBlock(in_channels, out_channels, wd=wd),
            nn.MaxPool1d(pool_size),
            nn.Dropout(drop)
        )
    
    def forward(self, imu, thm, tof):
        # imu = torch.cat(imu, dim=-1)
        imu_acc, imu_rot, imu_sum, imu_ms, imu_md, imu_jerk, imu_rot_mag, imu_roll_mean,imu_rot_jerk,imu_rot_xyz, imu_others = imu
        thm = torch.cat(thm, dim=-1)
        tof = torch.cat((tof), dim=-1)
        imu_feat1 = self.imu_branch_linear(imu_acc.permute(0, 2, 1))
        imu_feat2 = self.imu_branch_rot(imu_rot.permute(0, 2, 1))
        imu_feat3 = self.imu_branch_other(imu_others.permute(0, 2, 1))
        # print("imu_feat, thm_feat, tof_feat.shape 111",torch.cat([imu_feat1,imu_feat2,imu_feat3, thm_feat, tof_feat], dim=1).shape)
        # print("imu_feat.shape",imu_feat1.shape)
        thm_feat = self.thm_branch(thm.permute(0, 2, 1))
        tof_feat = self.tof_branch(tof.permute(0, 2, 1))
        # print(imu_feat.shape,thm_feat.shape,thm_feat.shape)
        # print("imu_feat, thm_feat, tof_feat.shape",torch.cat([imu_feat1,imu_feat2,imu_feat3, thm_feat, tof_feat], dim=-1).shape)
        bert_input = torch.cat([imu_feat1,imu_feat2,imu_feat3, thm_feat, tof_feat], dim=-1).permute(0, 2, 1)
        # print('bert_input -->',bert_input.shape)
        cls_token = self.cls_token.expand(bert_input.size(0), -1, -1)  # (B,1,H)
        # print("cls_token",cls_token.shape)
        # print("cls_token concat",torch.cat([cls_token, bert_input], dim=1).shape)
        bert_input = torch.cat([cls_token, bert_input], dim=1)  # (B,T+1,H)
        outputs = self.deberta(inputs_embeds=bert_input)
        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        mean_pool = last_hidden.mean(dim=1)
        pred_cls = torch.cat([cls_token, mean_pool], dim=-1)  # (B, 2H)
        return self.classifier(pred_cls),self.aux_classifier(cls_token).squeeze(-1)  


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

config = DebertaV2Config(
    hidden_size=256,
    num_hidden_layers=4,
    num_attention_heads=8,
    intermediate_size=1024,
    max_position_embeddings=512,
    relative_attention=True,
    position_biased_input=False,
    position_buckets=256,
    norm_rel_ebd="layer_norm",
    share_att_key=True,
    pos_att_type="p2c|c2p",
    layer_norm_eps=1e-7,
)

     

class CMIModel_roberta_dp_aux(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.imu_branch_linear = nn.Sequential(
            self.residual_se_cnn_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"],
                                       drop=kwargs["imu1_dropout"]),
            self.residual_se_cnn_block(kwargs["imu1_channels"], kwargs["feat_dim"], kwargs["imu2_layers"],
                                       drop=kwargs["imu2_dropout"])
        )

        self.imu_branch_rot = nn.Sequential(
            self.residual_se_cnn_block(4, kwargs["imu1_channels"], kwargs["imu1_layers"],
                                       drop=kwargs["imu1_dropout"]),
            self.residual_se_cnn_block(kwargs["imu1_channels"], kwargs["feat_dim"], kwargs["imu2_layers"],
                                       drop=kwargs["imu2_dropout"])
        )

        self.imu_branch_other = nn.Sequential(
            self.residual_se_cnn_block(10, kwargs["imu1_channels"], kwargs["imu1_layers"],
                                       drop=kwargs["imu1_dropout"]),
            self.residual_se_cnn_block(kwargs["imu1_channels"], kwargs["feat_dim"], kwargs["imu2_layers"],
                                       drop=kwargs["imu2_dropout"])
        )        
    
        self.thm_branch = nn.Sequential(
            nn.Conv1d(kwargs["thm_dim"], kwargs["thm1_channels"], kernel_size=3, padding=1, bias=False),
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
            nn.Conv1d(kwargs["tof_dim"], kwargs["tof1_channels"], kernel_size=3, padding=1, bias=False),
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

        self.roberta = RobertaModel(RobertaConfig(
                        hidden_size=kwargs["feat_dim"],                 # e.g., 768 or 1024
                        num_hidden_layers=kwargs["bert_layers"],        # e.g., 12 (base) or 24 (large)
                        num_attention_heads=kwargs["bert_heads"],       # e.g., 12 (base) or 16 (large)
                        intermediate_size=kwargs["feat_dim"] * 4,       # feed-forward dimension
                        max_position_embeddings=514,                        # default for RoBERTa
                        vocab_size=50265,                                   # RoBERTa tokenizer vocab size
                        type_vocab_size=1,                                  # RoBERTa doesn’t use token_type_ids
                        pad_token_id=1,                                     # RoBERTa’s padding ID
                        layer_norm_eps=1e-5,
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1
                    ))

        self.aux_classifier = nn.Linear(kwargs["feat_dim"], 1)
        
        self.classifier = nn.Sequential(
            nn.Linear(kwargs["feat_dim"]*2, kwargs["cls1_channels"], bias=False),
            nn.BatchNorm1d(kwargs["cls1_channels"]),
            nn.ReLU(inplace=True),
            nn.Dropout(kwargs["cls1_dropout"]),
            nn.Linear(kwargs["cls1_channels"], kwargs["cls2_channels"], bias=False),
            nn.BatchNorm1d(kwargs["cls2_channels"]),
            nn.ReLU(inplace=True),
            nn.Dropout(kwargs["cls2_dropout"]),
            nn.Linear(kwargs["cls2_channels"], kwargs["n_classes"])
        )

        
    
    def residual_se_cnn_block(self, in_channels, out_channels, num_layers, pool_size=2, drop=0.3, wd=1e-4):
        return nn.Sequential(
            *[ResNetSEBlock(in_channels=in_channels, out_channels=in_channels) for i in range(num_layers)],
            ResNetSEBlock(in_channels, out_channels, wd=wd),
            nn.MaxPool1d(pool_size),
            nn.Dropout(drop)
        )
    
    def forward(self, imu, thm, tof):
        # imu = torch.cat(imu, dim=-1)
        imu_acc, imu_rot, imu_sum, imu_ms, imu_md, imu_jerk, imu_rot_mag, imu_roll_mean,imu_rot_jerk,imu_rot_xyz, imu_others = imu
        thm = torch.cat(thm, dim=-1)
        tof = torch.cat((tof), dim=-1)
        imu_feat1 = self.imu_branch_linear(imu_acc.permute(0, 2, 1))
        imu_feat2 = self.imu_branch_rot(imu_rot.permute(0, 2, 1))
        imu_feat3 = self.imu_branch_other(imu_others.permute(0, 2, 1))
        thm_feat = self.thm_branch(thm.permute(0, 2, 1))
        tof_feat = self.tof_branch(tof.permute(0, 2, 1))

        bert_input = torch.cat([imu_feat1,imu_feat2,imu_feat3, thm_feat, tof_feat], dim=-1).permute(0, 2, 1)
        cls_token = self.cls_token.expand(bert_input.size(0), -1, -1)  # (B,1,H)
        bert_input = torch.cat([cls_token, bert_input], dim=1)  # (B,T+1,H)
        outputs = self.roberta(inputs_embeds=bert_input)
        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        mean_pool = last_hidden.mean(dim=1)
        pred_cls = torch.cat([cls_token, mean_pool], dim=-1)  # (B, 2H)
        return self.classifier(pred_cls),self.aux_classifier(cls_token).squeeze(-1)  


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
        #out = out + identity
        #return self.relu(out)
        out = out + identity
        out = self.relu(out)
        out = F.layer_norm(out, out.shape[1:])
        return out

class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.score_fn = nn.Linear(feature_dim, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (B, L, F)
        score = torch.tanh(self.score_fn(x))     # (B, L, 1)
        weights = self.softmax(score.squeeze(-1))# (B, L)
        weights = weights.unsqueeze(-1)          # (B, L, 1)
        context = x * weights                    # (B, L, F)
        return context.sum(dim=1)                # (B, F)

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


class CMIBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.imu_acc_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )
        self.imu_rot_branch = nn.Sequential(
            self.residual_feature_block(4, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )
        self.imu_sum_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )
        self.imu_ms_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )
        self.imu_md_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )      
        self.imu_other_branch = nn.Sequential(
            self.residual_feature_block(10, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )    

        self.imu_jerk_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )    

        self.imu_rotmag_branch = nn.Sequential(
            self.residual_feature_block(5, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )    

        self.imu_rollmean_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )            
        self.imu_rotkerk_branch = nn.Sequential(
            self.residual_feature_block(4, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )   
        self.imu_rotxyz_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )   
        # kwargs["thm_dim"]  kwargs["tof_dim"]
        self.thm_branch1, self.tof_branch1 = self.init_thm_tof_branch(**kwargs)
        self.thm_branch2, self.tof_branch2 = self.init_thm_tof_branch(**kwargs)
        self.thm_branch3, self.tof_branch3 = self.init_thm_tof_branch(**kwargs)
        self.thm_branch4, self.tof_branch4 = self.init_thm_tof_branch(**kwargs)
        self.thm_branch5, self.tof_branch5 = self.init_thm_tof_branch(**kwargs)

        self.imu_proj1 = ResNetSEBlock(in_channels=3*kwargs["imu2_channels"], out_channels=kwargs["imu2_channels"])
        self.imu_proj2 = ResNetSEBlock(in_channels=4*kwargs["imu2_channels"], out_channels=kwargs["imu2_channels"])
        self.thm_proj = ResNetSEBlock(in_channels=5*kwargs["thm2_channels"], out_channels=kwargs["thm2_channels"])
        self.tof_proj = ResNetSEBlock(in_channels=5*kwargs["tof2_channels"], out_channels=kwargs["tof2_channels"])

        self.lstm = nn.LSTM(
            input_size=2*kwargs['imu2_channels']+kwargs['thm2_channels']+kwargs['tof2_channels'],
            hidden_size=kwargs['lstm_hidden_size'],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.gru = nn.GRU(
            input_size=2*kwargs['imu2_channels']+kwargs['thm2_channels']+kwargs['tof2_channels'],
            hidden_size=kwargs['gru_hidden_size'],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.noise = GaussianNoise(kwargs['gaussian_noise_rate'])
        self.dense = nn.Sequential(
            nn.Linear(2*kwargs['imu2_channels']+kwargs['thm2_channels']+kwargs['tof2_channels'], kwargs['dense_channels']),
            nn.ELU()
        )
        
        self.attn = AttentionLayer(feature_dim=(kwargs['lstm_hidden_size']+kwargs['gru_hidden_size'])*2+kwargs['dense_channels'])  # lstm + gru + dense

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

    def init_thm_tof_branch(self, **kwargs):
        thm_branch = nn.Sequential(
            self.feature_block(kwargs["thm_dim"]//5, kwargs["thm1_channels"], kwargs["thm1_layers"], drop=kwargs["thm1_dropout"]),
            self.feature_block(kwargs["thm1_channels"], kwargs["thm2_channels"], kwargs["thm2_layers"], drop=kwargs["thm2_dropout"]),
        )
        tof_branch = nn.Sequential(
            self.feature_block(kwargs["tof_dim"]//5, kwargs["tof1_channels"], kwargs["tof1_layers"], drop=kwargs["tof1_dropout"]),
            self.feature_block(kwargs["tof1_channels"], kwargs["tof2_channels"], kwargs["tof2_layers"], drop=kwargs["tof2_dropout"]),
        )
        return thm_branch, tof_branch
    
    def forward(self, imus, thms, tofs):
        imu_acc, imu_rot, imu_sum, imu_ms, imu_md, imu_jerk, imu_rot_mag, imu_roll_mean,imu_rot_jerk,imu_rot_xyz, imu_others = imus
        imu_acc_feat = self.imu_acc_branch(imu_acc.permute(0, 2, 1))
        imu_rot_feat = self.imu_rot_branch(imu_rot.permute(0, 2, 1))
        imu_sum_feat = self.imu_sum_branch(imu_sum.permute(0, 2, 1))
        imu_ms_feat = self.imu_ms_branch(imu_ms.permute(0, 2, 1))
        imu_md_feat = self.imu_md_branch(imu_md.permute(0, 2, 1))
        imu_others_feat = self.imu_other_branch(imu_others.permute(0, 2, 1))
        imu_jerk_feat = self.imu_jerk_branch(imu_jerk.permute(0, 2, 1))
        imu_rotmag_feat = self.imu_rotmag_branch(imu_rot_mag.permute(0, 2, 1))
        imu_rollmean_feat = self.imu_rollmean_branch(imu_roll_mean.permute(0, 2, 1))

        imu_rotkerk_feat = self.imu_rotkerk_branch(imu_rot_jerk.permute(0, 2, 1))
        imu_rotxyz_feat = self.imu_rotxyz_branch(imu_rot_xyz.permute(0, 2, 1))

        imu_feat1 = self.imu_proj1(torch.cat([imu_acc_feat, imu_rot_feat, imu_others_feat], dim=1)) #
        imu_feat2 = self.imu_proj2(torch.cat([imu_md_feat, imu_jerk_feat,imu_ms_feat,imu_sum_feat], dim=1)) # imu_rotmag_feat

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
        
        feat = torch.cat([imu_feat1,imu_feat2, thm_feat, tof_feat], dim=1).permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat)
        gru_out, _ = self.gru(feat)
        dense_out = self.dense(self.noise(feat))
        
        return self.attn(torch.cat([lstm_out, gru_out, dense_out], dim=-1))

class CMIModel_combined_model(nn.Module):
    def __init__(self, target_classes_num, non_target_classes_num, **kwargs):
        super().__init__()
        self.backbone = CMIBackbone(**kwargs)
        
        self.imu_branch_linear = nn.Sequential(
        self.residual_se_cnn_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"],
                                   drop=kwargs["imu1_dropout"]),
        self.residual_se_cnn_block(kwargs["imu1_channels"], kwargs["feat_dim"], kwargs["imu2_layers"],
                                       drop=kwargs["imu2_dropout"])
        )

        self.imu_branch_rot = nn.Sequential(
            self.residual_se_cnn_block(4, kwargs["imu1_channels"], kwargs["imu1_layers"],
                                       drop=kwargs["imu1_dropout"]),
            self.residual_se_cnn_block(kwargs["imu1_channels"], kwargs["feat_dim"], kwargs["imu2_layers"],
                                       drop=kwargs["imu2_dropout"])
        )

        self.imu_branch_other = nn.Sequential(
            self.residual_se_cnn_block(10, kwargs["imu1_channels"], kwargs["imu1_layers"],
                                       drop=kwargs["imu1_dropout"]),
            self.residual_se_cnn_block(kwargs["imu1_channels"], kwargs["feat_dim"], kwargs["imu2_layers"],
                                       drop=kwargs["imu2_dropout"])
        )        
    
        self.thm_branch = nn.Sequential(
            nn.Conv1d(kwargs["thm_dim"], kwargs["thm1_channels"], kernel_size=3, padding=1, bias=False),
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
            nn.Conv1d(kwargs["tof_dim"], kwargs["tof1_channels"], kernel_size=3, padding=1, bias=False),
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

        self.roberta = RobertaModel(RobertaConfig(
                        hidden_size=kwargs["feat_dim"],                 # e.g., 768 or 1024
                        num_hidden_layers=kwargs["bert_layers"],        # e.g., 12 (base) or 24 (large)
                        num_attention_heads=kwargs["bert_heads"],       # e.g., 12 (base) or 16 (large)
                        intermediate_size=kwargs["feat_dim"] * 4,       # feed-forward dimension
                        max_position_embeddings=514,                        # default for RoBERTa
                        vocab_size=50265,                                   # RoBERTa tokenizer vocab size
                        type_vocab_size=1,                                  # RoBERTa doesn’t use token_type_ids
                        pad_token_id=1,                                     # RoBERTa’s padding ID
                        layer_norm_eps=1e-5,
                        hidden_dropout_prob=0.1,
                        attention_probs_dropout_prob=0.1
                    ))

        self.aux_classifier = nn.Sequential(
            nn.Linear((kwargs['lstm_hidden_size']+kwargs['gru_hidden_size'])*2+kwargs['dense_channels']+kwargs["feat_dim"]*2, kwargs["cls1_channels"], bias=False),
            nn.BatchNorm1d(kwargs["cls1_channels"]),
            nn.ReLU(inplace=True),
            nn.Dropout(kwargs["cls1_dropout"]),
            nn.Linear(kwargs["cls1_channels"], kwargs["cls2_channels"], bias=False),
            nn.BatchNorm1d(kwargs["cls2_channels"]),
            nn.ReLU(inplace=True),
            nn.Dropout(kwargs["cls2_dropout"]),
            nn.Linear(kwargs["cls2_channels"], 1)
        )        
   
        self.classifier = nn.Sequential(
            nn.Linear((kwargs['lstm_hidden_size']+kwargs['gru_hidden_size'])*2+kwargs['dense_channels']+kwargs["feat_dim"]*2, kwargs["cls1_channels"], bias=False),
            nn.BatchNorm1d(kwargs["cls1_channels"]),
            nn.ReLU(inplace=True),
            nn.Dropout(kwargs["cls1_dropout"]),
            nn.Linear(kwargs["cls1_channels"], kwargs["cls2_channels"], bias=False),
            nn.BatchNorm1d(kwargs["cls2_channels"]),
            nn.ReLU(inplace=True),
            nn.Dropout(kwargs["cls2_dropout"]),
            nn.Linear(kwargs["cls2_channels"], kwargs["n_classes"])
        )

    def residual_se_cnn_block(self, in_channels, out_channels, num_layers, pool_size=2, drop=0.3, wd=1e-4):
        return nn.Sequential(
        *[ResNetSEBlock(in_channels=in_channels, out_channels=in_channels) for i in range(num_layers)],
        ResNetSEBlock(in_channels, out_channels, wd=wd),
        nn.MaxPool1d(pool_size),
        nn.Dropout(drop))        



    def forward(self, imu, thm, tof):
        feat1 = self.backbone(imu, thm, tof)
        imu_acc, imu_rot, imu_sum, imu_ms, imu_md, imu_jerk, imu_rot_mag, imu_roll_mean,imu_rot_jerk,imu_rot_xyz, imu_others = imu
        thm = torch.cat(thm, dim=-1)
        tof = torch.cat((tof), dim=-1)
        imu_feat1 = self.imu_branch_linear(imu_acc.permute(0, 2, 1))
        imu_feat2 = self.imu_branch_rot(imu_rot.permute(0, 2, 1))
        imu_feat3 = self.imu_branch_other(imu_others.permute(0, 2, 1))
        thm_feat = self.thm_branch(thm.permute(0, 2, 1))
        tof_feat = self.tof_branch(tof.permute(0, 2, 1))
        bert_input = torch.cat([imu_feat1,imu_feat2,imu_feat3, thm_feat, tof_feat], dim=-1).permute(0, 2, 1)
        cls_token = self.cls_token.expand(bert_input.size(0), -1, -1)  # (B,1,H)
        bert_input = torch.cat([cls_token, bert_input], dim=1)  # (B,T+1,H)
        outputs = self.roberta(inputs_embeds=bert_input)
        last_hidden = outputs.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        mean_pool = last_hidden.mean(dim=1)
        feat2 = torch.cat([cls_token, mean_pool], dim=-1)  # (B, 2H)
        combined = torch.cat([feat1, feat2], dim=-1)
        aux = self.aux_classifier(combined).squeeze(-1)
        targets = self.classifier(combined)
        return targets,aux


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
        #out = out + identity
        #return self.relu(out)
        out = out + identity
        out = self.relu(out)
        out = F.layer_norm(out, out.shape[1:])
        return out

class AttentionLayer(nn.Module):
    def __init__(self, feature_dim):
        super().__init__()
        self.score_fn = nn.Linear(feature_dim, 1, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # x: (B, L, F)
        score = torch.tanh(self.score_fn(x))     # (B, L, 1)
        weights = self.softmax(score.squeeze(-1))# (B, L)
        weights = weights.unsqueeze(-1)          # (B, L, 1)
        context = x * weights                    # (B, L, F)
        return context.sum(dim=1)                # (B, F)

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


class CMIBackbone(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.imu_acc_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )
        self.imu_rot_branch = nn.Sequential(
            self.residual_feature_block(4, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )
        self.imu_sum_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )
        self.imu_ms_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )
        self.imu_md_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )      
        self.imu_other_branch = nn.Sequential(
            self.residual_feature_block(10, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )    

        self.imu_jerk_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )    

        self.imu_rotmag_branch = nn.Sequential(
            self.residual_feature_block(5, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )    

        self.imu_rollmean_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )            
        self.imu_rotkerk_branch = nn.Sequential(
            self.residual_feature_block(4, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )   
        self.imu_rotxyz_branch = nn.Sequential(
            self.residual_feature_block(3, kwargs["imu1_channels"], kwargs["imu1_layers"], drop=kwargs["imu1_dropout"]),
            self.residual_feature_block(kwargs["imu1_channels"], kwargs["imu2_channels"], kwargs["imu2_layers"], drop=kwargs["imu2_dropout"])
        )   

        self.thm_branch1, self.tof_branch1 = self.init_thm_tof_branch(**kwargs)
        self.thm_branch2, self.tof_branch2 = self.init_thm_tof_branch(**kwargs)
        self.thm_branch3, self.tof_branch3 = self.init_thm_tof_branch(**kwargs)
        self.thm_branch4, self.tof_branch4 = self.init_thm_tof_branch(**kwargs)
        self.thm_branch5, self.tof_branch5 = self.init_thm_tof_branch(**kwargs)

        self.imu_proj1 = ResNetSEBlock(in_channels=3*kwargs["imu2_channels"], out_channels=kwargs["imu2_channels"])
        self.imu_proj2 = ResNetSEBlock(in_channels=4*kwargs["imu2_channels"], out_channels=kwargs["imu2_channels"])
        self.thm_proj = ResNetSEBlock(in_channels=5*kwargs["thm2_channels"], out_channels=kwargs["thm2_channels"])
        self.tof_proj = ResNetSEBlock(in_channels=5*kwargs["tof2_channels"], out_channels=kwargs["tof2_channels"])

        self.lstm = nn.LSTM(
            input_size=2*kwargs['imu2_channels']+kwargs['thm2_channels']+kwargs['tof2_channels'],
            hidden_size=kwargs['lstm_hidden_size'],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.gru = nn.GRU(
            input_size=2*kwargs['imu2_channels']+kwargs['thm2_channels']+kwargs['tof2_channels'],
            hidden_size=kwargs['gru_hidden_size'],
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        self.noise = GaussianNoise(kwargs['gaussian_noise_rate'])
        self.dense = nn.Sequential(
            nn.Linear(2*kwargs['imu2_channels']+kwargs['thm2_channels']+kwargs['tof2_channels'], kwargs['dense_channels']),
            nn.ELU()
        )
        
        self.attn = AttentionLayer(feature_dim=(kwargs['lstm_hidden_size']+kwargs['gru_hidden_size'])*2+kwargs['dense_channels'])  # lstm + gru + dense

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

    def init_thm_tof_branch(self,**kwargs):
        thm_branch = nn.Sequential(
            self.feature_block(kwargs["thm_dim"]//5, kwargs["thm1_channels"], kwargs["thm1_layers"], drop=kwargs["thm1_dropout"]),
            self.feature_block(kwargs["thm1_channels"], kwargs["thm2_channels"], kwargs["thm2_layers"], drop=kwargs["thm2_dropout"]),
        )
        tof_branch = nn.Sequential(
            self.feature_block(kwargs["tof_dim"]//5, kwargs["tof1_channels"], kwargs["tof1_layers"], drop=kwargs["tof1_dropout"]),
            self.feature_block(kwargs["tof1_channels"], kwargs["tof2_channels"], kwargs["tof2_layers"], drop=kwargs["tof2_dropout"]),
        )
        return thm_branch, tof_branch
    
    def forward(self, imus, thms, tofs):
        imu_acc, imu_rot, imu_sum, imu_ms, imu_md, imu_jerk, imu_rot_mag, imu_roll_mean,imu_rot_jerk,imu_rot_xyz, imu_others = imus
        imu_acc_feat = self.imu_acc_branch(imu_acc.permute(0, 2, 1))
        imu_rot_feat = self.imu_rot_branch(imu_rot.permute(0, 2, 1))
        imu_sum_feat = self.imu_sum_branch(imu_sum.permute(0, 2, 1))
        imu_ms_feat = self.imu_ms_branch(imu_ms.permute(0, 2, 1))
        imu_md_feat = self.imu_md_branch(imu_md.permute(0, 2, 1))
        imu_others_feat = self.imu_other_branch(imu_others.permute(0, 2, 1))
        imu_jerk_feat = self.imu_jerk_branch(imu_jerk.permute(0, 2, 1))
        imu_rotmag_feat = self.imu_rotmag_branch(imu_rot_mag.permute(0, 2, 1))
        imu_rollmean_feat = self.imu_rollmean_branch(imu_roll_mean.permute(0, 2, 1))

        imu_rotkerk_feat = self.imu_rotkerk_branch(imu_rot_jerk.permute(0, 2, 1))
        imu_rotxyz_feat = self.imu_rotxyz_branch(imu_rot_xyz.permute(0, 2, 1))

        imu_feat1 = self.imu_proj1(torch.cat([imu_acc_feat, imu_rot_feat, imu_others_feat], dim=1)) #
        imu_feat2 = self.imu_proj2(torch.cat([imu_md_feat, imu_jerk_feat,imu_ms_feat,imu_sum_feat], dim=1)) # imu_rotmag_feat

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
        
        feat = torch.cat([imu_feat1,imu_feat2, thm_feat, tof_feat], dim=1).permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat)
        gru_out, _ = self.gru(feat)
        dense_out = self.dense(self.noise(feat))
        
        return self.attn(torch.cat([lstm_out, gru_out, dense_out], dim=-1))

class CMIModel_cnn_model_normal(nn.Module):
    def __init__(self, target_classes_num, non_target_classes_num, **kwargs):
        super().__init__()
        self.backbone = CMIBackbone(**kwargs)
        self.target_classifier = nn.Sequential(
            nn.Linear((kwargs['lstm_hidden_size']+kwargs['gru_hidden_size'])*2+kwargs['dense_channels'], kwargs["cls_channels1"]),
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
            nn.Linear((kwargs['lstm_hidden_size']+kwargs['gru_hidden_size'])*2+kwargs['dense_channels'], kwargs["cls_channels1"]),
            nn.BatchNorm1d(kwargs["cls_channels1"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout1"]),
            nn.Linear(kwargs["cls_channels1"], kwargs["cls_channels2"]),
            nn.BatchNorm1d(kwargs["cls_channels2"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout2"]),
            nn.Linear(kwargs["cls_channels2"], non_target_classes_num)
        )

        self.aux_classifier = nn.Sequential(
            nn.Linear((kwargs['lstm_hidden_size']+kwargs['gru_hidden_size'])*2+kwargs['dense_channels'], kwargs["cls_channels1"]),
            nn.BatchNorm1d(kwargs["cls_channels1"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout1"]),
            nn.Linear(kwargs["cls_channels1"], kwargs["cls_channels2"]),
            nn.BatchNorm1d(kwargs["cls_channels2"]),
            nn.ReLU(),
            nn.Dropout(kwargs["cls_dropout2"]),
            nn.Linear(kwargs["cls_channels2"], 1)
        )
    def forward(self, imu, thm, tof):
        feat = self.backbone(imu, thm, tof)
        targets_y = self.target_classifier(feat)
        non_targets_y = self.non_target_classifier(feat)
        aux = self.aux_classifier(feat).squeeze(-1)
        return torch.cat([targets_y, non_targets_y], dim=1),aux





def label_smoothing_loss(pred, target, smoothing=0.1):
    """Label smoothing loss"""
    confidence = 1.0 - smoothing
    log_probs = F.log_softmax(pred, dim=-1)
    nll_loss = -log_probs.gather(dim=-1, index=target.unsqueeze(1))
    nll_loss = nll_loss.squeeze(1)
    smooth_loss = -log_probs.mean(dim=-1)
    loss = confidence * nll_loss + smoothing * smooth_loss
    return loss.mean()


def get_optimizer_with_llrd(model, base_lr, weight_decay=0.01, lr_decay=0.9):
    """
    Assign smaller learning rates to lower layers, larger to top layers.
    """
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = []

    layers = [model.deberta.embeddings] + list(model.deberta.encoder.layer)
    layers.append(model.classifier)

    lr = base_lr
    for layer in reversed(layers):  # top layers get full lr
        params = list(layer.named_parameters())
        lr_layer = lr
        for n, p in params:
            if any(nd in n for nd in no_decay):
                optimizer_grouped_parameters.append(
                    {"params": [p], "weight_decay": 0.0, "lr": lr_layer}
                )
            else:
                optimizer_grouped_parameters.append(
                    {"params": [p], "weight_decay": weight_decay, "lr": lr_layer}
                )
        lr *= lr_decay  # decay as we go down

    optimizer = AdamW(optimizer_grouped_parameters, lr=base_lr)
    return optimizer

class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, dim=-1):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        return torch.mean(torch.sum(-true_dist * pred, dim=self.dim))    


def imu_only_augment(imus, thms, tofs, p):
    """
    Randomly selects B * p rows in a batch and replaces them with IMU-only tensors.
    
    Parameters:
    imu (Tensor): IMU data tensor of shape (B, ...)
    thm (Tensor): THM data tensor of shape (B, ...)
    tof (Tensor): TOF data tensor of shape (B, ...)
    p (float): Proportion of the batch to convert to IMU-only
    
    Returns:
    Tuple of augmented (imu, thm, tof) tensors
    """
    B = imus[0].size(0)
    num_imu_only = int(B * p)
    
    # Generate random indices for IMU-only rows
    indices = torch.randperm(B)[:num_imu_only]
    
    # Create copies to avoid modifying original tensors
    thm_aug = []
    tof_aug = []
    
    # Zero out THM and TOF data for selected indices
    
    for idx, thm in enumerate(thms):
        thm_ = thm.clone()
        thm_[indices] = 0
        thm_aug.append(thm_)

    for idx, tof in enumerate(tofs):
        tof_ = tof.clone()
        tof_[indices] = 0
        tof_aug.append(tof_)
    
    return imus, thm_aug, tof_aug

def mixup_augment(imus, thms, tofs, labels, alpha=0.2, prob=0.5):
    """
    Applies Mixup augmentation to IMU, THM, TOF batches and labels together.

    Parameters:
    imus (list of Tensors): IMU data tensors, each with shape (B, ...)
    thms (list of Tensors): THM data tensors, each with shape (B, ...)
    tofs (list of Tensors): TOF data tensors, each with shape (B, ...)
    labels (Tensor): Shape (B,) for class indices or (B, num_classes) for one-hot
    alpha (float): Beta distribution parameter for mix ratio
    prob (float): Probability of applying mixup

    Returns:
    Tuple: (aug_imus, aug_thms, aug_tofs, aug_labels)
    """
    if random.random() > prob:
        return imus, thms, tofs, labels

    B = imus[0].size(0)
    lambda_ = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(B)

    aug_imus = [(lambda_ * imu + (1 - lambda_) * imu[index]) for imu in imus]
    aug_thms = [(lambda_ * thm + (1 - lambda_) * thm[index]) for thm in thms]
    aug_tofs = [(lambda_ * tof + (1 - lambda_) * tof[index]) for tof in tofs]
    aug_labels = lambda_ * labels + (1 - lambda_) * labels[index]

    return aug_imus, aug_thms, aug_tofs, aug_labels


infer = False
training = True

import copy
import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class WarmupCosineScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, final_lr=2e-5, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.final_lr = final_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            return [self.base_lr * (self.last_epoch + 1) / self.warmup_epochs for _ in self.optimizer.param_groups]
        else:
            decay_epoch = self.last_epoch - self.warmup_epochs
            decay_total = self.total_epochs - self.warmup_epochs
            cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_epoch / decay_total))
            return [self.final_lr + (self.base_lr - self.final_lr) * cosine_decay for _ in self.optimizer.param_groups]

# === Metric ===
class CompetitionMetric:
    def __init__(self):
        self.target_gestures = [
            'Above ear - pull hair', 'Cheek - pinch skin', 'Eyebrow - pull hair',
            'Eyelash - pull hair', 'Forehead - pull hairline', 'Forehead - scratch',
            'Neck - pinch skin', 'Neck - scratch'
        ]
        self.non_target_gestures = [
            'Write name on leg', 'Wave hello', 'Glasses on/off', 'Text on phone',
            'Write name in air', 'Feel around in tray and pull out an object',
            'Scratch knee/leg skin', 'Pull air toward your face',
            'Drink from bottle/cup', 'Pinch knee/leg skin'
        ]

    def calculate_hierarchical_f1(self, sol: pd.DataFrame, sub: pd.DataFrame) -> float:
        y_true_bin = sol['gesture'].isin(self.target_gestures).values
        y_pred_bin = sub['gesture'].isin(self.target_gestures).values
        f1_binary = f1_score(y_true_bin, y_pred_bin, pos_label=True, zero_division=0, average='binary')
        y_true_mc = sol['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        y_pred_mc = sub['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        f1_macro = f1_score(y_true_mc, y_pred_mc, average='macro', zero_division=0)
        return 0.5 * f1_binary + 0.5 * f1_macro

def plot_lr_schedule(optimizer, scheduler, total_epochs):
    lrs = []
    for epoch in range(total_epochs):
        scheduler.step()
        lrs.append(optimizer.param_groups[0]['lr'])
    plt.plot(range(total_epochs), lrs)
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule with Warmup and Cosine Decay")
    plt.grid(True)
    plt.show()

def check_tensor(tensor, name="tensor"):
    if torch.isnan(tensor).any():
        print(f"⚠️ NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"⚠️ Inf detected in {name}")

class ModelEMA:
    def __init__(self, model, decay=0.9999):
        self.ema_model = copy.deepcopy(model).eval()
        self.decay = decay
        self.ema_model.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            msd = model.state_dict()
            for k, ema_v in self.ema_model.state_dict().items():
                model_v = msd[k].detach()
                if model_v.dtype.is_floating_point:
                    ema_v.copy_(ema_v * self.decay + (1. - self.decay) * model_v)
                else:
                    ema_v.copy_(model_v)

# === Training ===
def train_model(config, dataset, fold_idx, num_epochs,model_name,folderpath):
    patience = 22
    model_lr = 5e-4
    model_weight_decay = 3e-3
    model_warmup_steps = 30
    min_lr = 2e-5
    min_lr_mode = False
    bad_epochs = 0
    alpha = 0.3
    train_set, val_set = dataset.get_fold_datasets(fold_idx)
    train_idx,valid_idx = dataset.folds[fold_idx]
    

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if model_name == "deberta":
        print('CMIModel_deberta_dp_aux running!!!')
        model = CMIModel_deberta_dp_aux(
            **config
        ).to(CUDA0)
    elif model_name == "roberta":
        print('CMIModel_roberta_dp_aux running!!!')
        model = CMIModel_roberta_dp_aux(
            **config
        ).to(CUDA0)
    elif model_name == "combined":
        print('CMIModel_combined_model running!!!')
        model = CMIModel_combined_model(
            **config
        ).to(CUDA0)
    elif model_name == "cnn":
        print('CMIModel_cnn_model_normal running!!!')
        model = CMIModel_cnn_model_normal(
            **config
        ).to(CUDA0)    
    
    ema = ModelEMA(model, decay=0.999)

    optimizer = optim.Adam(model.parameters(), lr=model_lr) #optim.Adam(model.parameters(), lr=model_lr) get_optimizer_with_llrd(model,base_lr=model_lr)
    # scheduler_dummy = WarmupCosineScheduler(optimizer, warmup_epochs=model_warmup_steps, total_epochs=200, base_lr=model_lr)
    scheduler_dummy = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.9875881 ** epoch)
    criterion = nn.CrossEntropyLoss(weight = dataset.full_dataset.class_weight.to(CUDA0))  #nn.CrossEntropyLoss()  label_smoothing_loss #LabelSmoothingLoss(classes=18, smoothing=0.05) #nn.CrossEntropyLoss()
    aux_criterion = nn.BCEWithLogitsLoss()

    metric = CompetitionMetric()

    plot_lr_schedule(optimizer, scheduler_dummy, total_epochs=num_epochs)
    scheduler = WarmupCosineScheduler(optimizer, warmup_epochs=model_warmup_steps, total_epochs=200, base_lr=model_lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.9875881 ** epoch)

    #print("H-F1 values from here out are actually ACC!")
    ACC_NOT_F1 = False
    best_hf1 = 0
    for epoch in range(num_epochs):
        for m in model.modules():
            if isinstance(m, (nn.LSTM, nn.GRU)):
                m.flatten_parameters()
        for m in ema.ema_model.modules():
            if isinstance(m, (nn.LSTM, nn.GRU)):
                m.flatten_parameters()
                
        model.train()
        train_loss = 0
        train_preds, train_targets = [], []

        for imus, thms, tofs, labels, aux in train_loader:
            for idx, imu in enumerate(imus):
                imus[idx] = imu.to(CUDA0)
            for idx, thm in enumerate(thms):
                thms[idx] = thm.to(CUDA0)
            for idx, tof in enumerate(tofs):
                tofs[idx] = tof.to(CUDA0)
            labels = labels.to(CUDA0)
            aux = aux.to(CUDA0)
            imus, thms, tofs = imu_only_augment(imus, thms, tofs, p=0.3)
            imus, thms, tofs, labels = mixup_augment(imus, thms, tofs, labels)
            
            #check_tensor(imu, "imu")
            #check_tensor(thm, "thm")
            #check_tensor(tof, "tof")
            #check_tensor(labels, "labels")
        
            labels_cls = labels.argmax(dim=1)

            optimizer.zero_grad()
            outputs,aux_logits = model(imus, thms, tofs)
            loss_main = criterion(outputs, labels_cls)
            loss_aux = aux_criterion(aux_logits, aux)
            loss = loss_main + alpha * loss_aux
            loss.backward()
            check_tensor(loss, "loss")
            check_tensor(outputs, "outputs")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            if torch.isnan(loss):
                print(f"⚠️ NaN detected in loss at epoch {epoch+1}")
                break  # Stop training if NaN is detected

            optimizer.step()
            ema.update(model)

            train_loss += loss.item()
            train_preds.extend(outputs.argmax(1).cpu().numpy())
            train_targets.extend(labels_cls.cpu().numpy())

        train_df = pd.DataFrame({'gesture': [dataset.class_names[p] for p in train_preds]})
        train_target_df = pd.DataFrame({'gesture': [dataset.class_names[t] for t in train_targets]})
        train_hf1 = metric.calculate_hierarchical_f1(train_target_df, train_df)
        #train_hf1 = accuracy_score(train_targets, train_preds)

        #model.eval()
        ema.ema_model.eval()
        val_preds_full, val_targets,val_preds_full_classs = [], [], []
        with torch.no_grad():
            for imus, thms, tofs, labels, _ in val_loader:
                for idx, imu in enumerate(imus):
                    imus[idx] = imu.to(CUDA0)
                for idx, thm in enumerate(thms):
                    thms[idx] = thm.to(CUDA0)
                for idx, tof in enumerate(tofs):
                    tofs[idx] = tof.to(CUDA0)
                labels = labels.to(CUDA0)
                labels_cls = labels.argmax(dim=1)

                #outputs = model(imus, thms, tofs)
                outputs,_ = ema.ema_model(imus, thms, tofs)
                val_preds_full.extend(outputs.argmax(1).cpu().numpy())
                val_preds_full_classs.extend(outputs.cpu().numpy())
                val_targets.extend(labels_cls.cpu().numpy())

        val_df = pd.DataFrame({'gesture': [dataset.class_names[p] for p in val_preds_full]})
        val_target_df = pd.DataFrame({'gesture': [dataset.class_names[t] for t in val_targets]})
        val_hf1_full = metric.calculate_hierarchical_f1(val_target_df, val_df)

        val_preds_imu, val_targets, val_preds_imu_classs = [], [], []
        with torch.no_grad():
            for imus, thms, tofs, labels, _ in val_loader:
                for idx, imu in enumerate(imus):
                    imus[idx] = imu.to(CUDA0)
                for idx, thm in enumerate(thms):
                    thms[idx] = thm.to(CUDA0)
                for idx, tof in enumerate(tofs):
                    tofs[idx] = tof.to(CUDA0)
                imus, thms, tofs = imu_only_augment(imus, thms, tofs, p=1)
                labels = labels.to(CUDA0)
                labels_cls = labels.argmax(dim=1)

                #outputs = model(imus, thms, tofs) robertav3/
                outputs,_ = ema.ema_model(imus, thms, tofs)
                val_preds_imu.extend(outputs.argmax(1).cpu().numpy())
                val_preds_imu_classs.extend(outputs.cpu().numpy())
                val_targets.extend(labels_cls.cpu().numpy())

        val_df = pd.DataFrame({'gesture': [dataset.class_names[p] for p in val_preds_imu]})
        val_target_df = pd.DataFrame({'gesture': [dataset.class_names[t] for t in val_targets]})
        val_hf1_imu = metric.calculate_hierarchical_f1(val_target_df, val_df)
        val_hf1 = (val_hf1_full + val_hf1_imu) / 2

        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train H-F1: {train_hf1:.4f}, Val H-F1 full: {val_hf1_full:.4f}, Val H-F1 imu: {val_hf1_imu:.4f}, Val H-F1: {val_hf1:.4f}, LR: {current_lr:.8f}")

        if val_hf1 > best_hf1:
            best_hf1 = val_hf1
            bad_epochs = 0
            # oof_preds_deberta2[valid_idx] = labels.cpu().numpy()
            #torch.save(model.state_dict(), f"best_model_fold{fold_idx}.pt")   debertav2
            torch.save(ema.ema_model.state_dict(), f"./{folderpath}/best_model_fold{fold_idx}.pt")
            # print(np.concatenate(val_preds_full).shape)
            # print(np.array(val_preds_full_classs).shape)
            oof_preds_deberta2_full[valid_idx] = np.array(val_preds_full_classs)
            oof_preds_deberta2_imu[valid_idx] = np.array(val_preds_imu_classs)
            oof_preds_deberta2_org[valid_idx] = np.array(val_targets)
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
               print(f"Early stopping triggered after {epoch+1} epochs.")
               break
        scheduler.step()


if training:
    model_args_dbv2 = {"feat_dim": 256,
                  "imu1_channels": 219, "imu1_dropout": 0.2946731587132302, "imu2_dropout": 0.2697745571929592,
                  "imu1_weight_decay": 0.0014824054650601245, "imu2_weight_decay": 0.002742543773142381,
                  "imu1_layers": 0, "imu2_layers": 0,
                  "thm1_channels": 82, "thm1_dropout": 0.2641274454844602, "thm2_dropout": 0.302896343020985, 
                  "tof1_channels": 82, "tof1_dropout": 0.2641274454844602, "tof2_dropout": 0.3028963430209852, 
                  "bert_layers": 4, "bert_heads": 8,"target_classes_num": 8, "non_target_classes_num": 10,
                  "cls1_channels": 937, "cls2_channels": 303, "cls1_dropout": 0.2281834512100508, "cls2_dropout": 0.22502521933558461,
                  "imu_dim": 44, 
                  "thm_dim": 10,
                  "tof_dim": 660,
                  "n_classes": 18,
                 "imu1_channels": 128, "imu2_channels": 256, "imu1_dropout": 0.3, "imu2_dropout": 0.25,
                  "imu1_layers": 0, "imu2_layers": 0, 
                  "thm1_channels": 32, "thm2_channels": 64, "thm1_dropout": 0.25, "thm2_dropout": 0.2,
                  "thm1_layers": 0, "thm2_layers": 0, 
                  "tof1_channels": 256, "tof2_channels": 512, "tof1_dropout": 0.4, "tof2_dropout": 0.3,
                  "tof1_layers": 0, "tof2_layers": 0, 
                  "lstm_hidden_size": 128, "gru_hidden_size": 128, "gaussian_noise_rate": 0.1, "dense_channels": 32,
                  "cls_channels1": 256, "cls_dropout1": 0.2, "cls_channels2": 128, "cls_dropout2": 0.2,
                  "target_classes_num": 8, "non_target_classes_num": 10}   
    model_args_roberta_deberta = {"feat_dim": 256,
              "imu1_channels": 219, "imu1_dropout": 0.2946731587132302, "imu2_dropout": 0.2697745571929592,
              "imu1_weight_decay": 0.0014824054650601245, "imu2_weight_decay": 0.002742543773142381,
              "imu1_layers": 0, "imu2_layers": 0,
              "thm1_channels": 82, "thm1_dropout": 0.2641274454844602, "thm2_dropout": 0.302896343020985, 
              "tof1_channels": 82, "tof1_dropout": 0.2641274454844602, "tof2_dropout": 0.3028963430209852, 
              "bert_layers": 4, "bert_heads": 8,"target_classes_num": 8, "non_target_classes_num": 10,
              "cls1_channels": 937, "cls2_channels": 303, "cls1_dropout": 0.2281834512100508, "cls2_dropout": 0.22502521933558461,
                 "imu_dim": dataset.full_dataset.imu_dim, 
                "thm_dim": dataset.full_dataset.thm_dim,
                "tof_dim": dataset.full_dataset.tof_dim,
                "n_classes": dataset.full_dataset.class_num} 

    import random
    import numpy as np
    oof_preds_deberta2_full = np.zeros((len(dataset.full_dataset), 18))
    oof_preds_deberta2_imu = np.zeros((len(dataset.full_dataset), 18))
    oof_preds_deberta2_org = np.zeros((len(dataset.full_dataset)))
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    train_model(model_args_roberta_deberta, dataset, fold_idx=0, num_epochs=130,model_name="deberta",folderpath = "CMIModel_deberta_dp_aux")
    train_model(model_args_roberta_deberta, dataset, fold_idx=1, num_epochs=130,model_name="deberta",folderpath = "CMIModel_deberta_dp_aux")
    train_model(model_args_roberta_deberta, dataset, fold_idx=2, num_epochs=130,model_name="deberta",folderpath = "CMIModel_deberta_dp_aux")
    train_model(model_args_roberta_deberta, dataset, fold_idx=3, num_epochs=130,model_name="deberta",folderpath = "CMIModel_deberta_dp_aux")
    train_model(model_args_roberta_deberta, dataset, fold_idx=4, num_epochs=130,model_name="deberta",folderpath = "CMIModel_deberta_dp_aux")
    np.save('./CMIModel_deberta_dp_aux/oof_preds_deberta2_full.npy', oof_preds_deberta2_full)
    np.save('./CMIModel_deberta_dp_aux/oof_preds_deberta2_imu.npy', oof_preds_deberta2_imu)
    np.save('./CMIModel_deberta_dp_aux/oof_preds_deberta2_org.npy', oof_preds_deberta2_org)


torch.cuda.empty_cache()
gc.collect()


oof_preds_deberta2_full = np.zeros((len(dataset.full_dataset), 18))
oof_preds_deberta2_imu = np.zeros((len(dataset.full_dataset), 18))
oof_preds_deberta2_org = np.zeros((len(dataset.full_dataset)))
train_model(model_args_roberta_deberta, dataset, fold_idx=0, num_epochs=130,model_name="roberta",folderpath = "CMIModel_roberta_dp_aux")
train_model(model_args_roberta_deberta, dataset, fold_idx=1, num_epochs=130,model_name="roberta",folderpath = "CMIModel_roberta_dp_aux")
train_model(model_args_roberta_deberta, dataset, fold_idx=2, num_epochs=130,model_name="roberta",folderpath = "CMIModel_roberta_dp_aux")
train_model(model_args_roberta_deberta, dataset, fold_idx=3, num_epochs=130,model_name="roberta",folderpath = "CMIModel_roberta_dp_aux")
train_model(model_args_roberta_deberta, dataset, fold_idx=4, num_epochs=130,model_name="roberta",folderpath = "CMIModel_roberta_dp_aux")
np.save('./CMIModel_roberta_dp_aux/oof_preds_deberta2_full.npy', oof_preds_deberta2_full)
np.save('./CMIModel_roberta_dp_aux/oof_preds_deberta2_imu.npy', oof_preds_deberta2_imu)
np.save('./CMIModel_roberta_dp_aux/oof_preds_deberta2_org.npy', oof_preds_deberta2_org)


torch.cuda.empty_cache()
gc.collect()


oof_preds_deberta2_full = np.zeros((len(dataset.full_dataset), 18))
oof_preds_deberta2_imu = np.zeros((len(dataset.full_dataset), 18))
oof_preds_deberta2_org = np.zeros((len(dataset.full_dataset)))
train_model(model_args_dbv2, dataset, fold_idx=0, num_epochs=130,model_name="combined",folderpath = "CMIModel_combined_model")
train_model(model_args_dbv2, dataset, fold_idx=1, num_epochs=130,model_name="combined",folderpath = "CMIModel_combined_model")
train_model(model_args_dbv2, dataset, fold_idx=2, num_epochs=130,model_name="combined",folderpath = "CMIModel_combined_model")
train_model(model_args_dbv2, dataset, fold_idx=3, num_epochs=130,model_name="combined",folderpath = "CMIModel_combined_model")
train_model(model_args_dbv2, dataset, fold_idx=4, num_epochs=130,model_name="combined",folderpath = "CMIModel_combined_model")
np.save('./CMIModel_combined_model/oof_preds_deberta2_full.npy', oof_preds_deberta2_full)
np.save('./CMIModel_combined_model/oof_preds_deberta2_imu.npy', oof_preds_deberta2_imu)
np.save('./CMIModel_combined_model/oof_preds_deberta2_org.npy', oof_preds_deberta2_org)


torch.cuda.empty_cache()
gc.collect()


oof_preds_deberta2_full = np.zeros((len(dataset.full_dataset), 18))
oof_preds_deberta2_imu = np.zeros((len(dataset.full_dataset), 18))
oof_preds_deberta2_org = np.zeros((len(dataset.full_dataset)))
train_model(model_args_dbv2, dataset, fold_idx=0, num_epochs=130,model_name="cnn",folderpath = "CMIModel_cnn_model_normal")
train_model(model_args_dbv2, dataset, fold_idx=1, num_epochs=130,model_name="cnn",folderpath = "CMIModel_cnn_model_normal")
train_model(model_args_dbv2, dataset, fold_idx=2, num_epochs=130,model_name="cnn",folderpath = "CMIModel_cnn_model_normal")
train_model(model_args_dbv2, dataset, fold_idx=3, num_epochs=130,model_name="cnn",folderpath = "CMIModel_cnn_model_normal")
train_model(model_args_dbv2, dataset, fold_idx=4, num_epochs=130,model_name="cnn",folderpath = "CMIModel_cnn_model_normal")
np.save('./CMIModel_cnn_model_normal/oof_preds_deberta2_full.npy', oof_preds_deberta2_full)
np.save('./CMIModel_cnn_model_normal/oof_preds_deberta2_imu.npy', oof_preds_deberta2_imu)
np.save('./CMIModel_cnn_model_normal/oof_preds_deberta2_org.npy', oof_preds_deberta2_org)








import os
import zipfile

def zip_folder(folder_path, output_zip_path):
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(folder_path):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, folder_path)
                zipf.write(abs_path, rel_path)

# Example usage
main_folder = "./CMIModel_cnn_model_normal"             # Replace with your folder path
output_zip = "./CMIModel_cnn_model_normal.zip"   # Replace with desired zip file path

zip_folder(main_folder, output_zip)
print(f"Zipped '{main_folder}' to '{output_zip}'")




