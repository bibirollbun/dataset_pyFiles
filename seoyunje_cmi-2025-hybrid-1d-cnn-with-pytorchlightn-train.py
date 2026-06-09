import os
import random
import gc, ctypes
import time
from tqdm import tqdm
import copy
from collections import defaultdict

import pandas as pd 
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from scipy.spatial.transform import Rotation as R


import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from colorama import Fore, Style
c_ = Fore.BLUE
sr_ = Style.BRIGHT

import warnings
warnings.filterwarnings('ignore')


import wandb 

try: 
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("wandb")
    wandb.login(key=api_key)

    anonymous = None
except:
    anonymous = "must"
    print(f'{c_}{sr_}To use your W&B account, \n Go to Add-ons -> Secrets and provide your W&B access token')


class CFG:
    save_dir = 'runs/'
    seed = 2025
    debug = False # Full Training
    sequence_len = 128
    project = 'CMI_2025'
    model_name = 'WaveNet'
    comment = "Multi-Class"
    label_smoothing = 0.05
    mix_up_prob = 0.15
    mix_up_alpha = 0.4
    num_classes = 18
    log_model = 'all' # when offline=False or 'all'
    offline = False
    kernel_size = 3
    train_bs = 32
    valid_bs = 2 * train_bs
    epochs= 20
    lr = 5e-4
    min_lr = lr*1e-3
    wd = 1e-4
    scheduler = 'CosineAnnealingLR'
    T_mult = 1
    warmup_epochs = int(epochs * 0.2)
    n_accumulate = 1
    n_fold = 5
    n_splits = 5
    fold = 0

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    device_count = torch.cuda.device_count()

os.makedirs(CFG.save_dir, exist_ok=True)
print(f"{sr_}{c_} => Device is {CFG.device}")
print(f"{sr_}{c_} => Num GPU of machine is ", CFG.device_count)


def clean_memory():
    ctypes.CDLL('libc.so.6').malloc_trim(0)
    gc.collect()


def seed_everything(SEED):
    np.random.seed(SEED)
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    """When running CuDNN backend, two further options must be set"""
    """Pytorch, TensorFlow Framework are both using CuDNN backend"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    os.environ['PYTHONHASHSEED'] = str(SEED)

seed_everything(CFG.seed)
pl.seed_everything(CFG.seed)


from scipy import interpolate

def inter_signal(x):
    original_len = len(x)
    interp_func = interpolate.interp1d(np.linspace(0, 1, original_len), x, kind='linear')
    x_interp = interp_func(np.linspace(0, 1, CFG.sequence_len))
    return x_interp


def prepare_loader(fold, train_transform=True, valid_transform=False, collate_fn = torch.utils.data._utils.collate.default_collate):
    train_df = df[df['fold'] != fold].reset_index(drop=True)
    valid_df = df[df['fold'] == fold].reset_index(drop=True)

    if CFG.debug:
        train_sp = np.random.choice(train_df['sequence_id'].unique(), 500)
        train_df = train_df[train_df['sequence_id'].isin(train_sp)]
    
        valid_sp = np.random.choice(valid_df['sequence_id'].unique(), 250)
        valid_df = valid_df[valid_df['sequence_id'].isin(valid_sp)]


    train_ds = CMI_Dataset(train_df, col=total_feature, transforms=train_transform, mix_up=True)
    valid_ds = CMI_Dataset(valid_df, col=total_feature, transforms=valid_transform, mix_up=False)

    train_loader = DataLoader(train_ds, shuffle=True, batch_size=CFG.train_bs,
                              num_workers=4, pin_memory=True, drop_last = True,
                              collate_fn = collate_fn)
    valid_loader = DataLoader(valid_ds, shuffle=False, batch_size=CFG.valid_bs,
                              num_workers=4, pin_memory=True, drop_last = False,
                              collate_fn = torch.utils.data._utils.collate.default_collate)

    return train_loader, valid_loader


df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
print(f'{sr_}{c_} Shape of DataFrame: {df.shape}')


behavior_list = ['SEQ_011975'] # only 2 behaviors in sequence
tmp_pause_list = ['SEQ_012617'] # Huge Variations in pause(TMP Device)
tof_pause_list = ['SEQ_047121','SEQ_030299'] # Huge Variations in pause(TOF Device)

delete_list = []
delete_list.extend(behavior_list)
delete_list.extend(tmp_pause_list)
delete_list.extend(tof_pause_list)


df = df[~df['sequence_id'].isin(delete_list)].reset_index(drop=True)


tmp = df.groupby(['sequence_id']).agg(
    rot_x_na_ratio = ('rot_x', lambda x: x.isna().mean()*100),
    rot_y_na_ratio = ('rot_y', lambda x: x.isna().mean()*100),
    rot_z_na_ratio = ('rot_z', lambda x: x.isna().mean()*100),
    rot_w_na_ratio = ('rot_w', lambda x: x.isna().mean()*100),

).reset_index()

rot_x_na_list = tmp[tmp['rot_x_na_ratio'] > 50]['sequence_id'].values
rot_y_na_list = tmp[tmp['rot_y_na_ratio'] > 50]['sequence_id'].values
rot_z_na_list = tmp[tmp['rot_z_na_ratio'] > 50]['sequence_id'].values
rot_w_na_list = tmp[tmp['rot_w_na_ratio'] > 50]['sequence_id'].values

all_ids = np.concatenate([rot_x_na_list, rot_y_na_list, rot_z_na_list, rot_w_na_list])
rot_na_list = np.unique(all_ids)


df = df[~df['sequence_id'].isin(rot_na_list)].reset_index(drop=True)


tmp = df.groupby(['sequence_id']).agg(
    thm_1_na_ratio = ('thm_1', lambda x: x.isna().mean()*100),
    thm_2_na_ratio = ('thm_2', lambda x: x.isna().mean()*100),
    thm_3_na_ratio = ('thm_3', lambda x: x.isna().mean()*100),
    thm_4_na_ratio = ('thm_4', lambda x: x.isna().mean()*100),
    thm_5_na_ratio = ('thm_5', lambda x: x.isna().mean()*100),

).reset_index()


thm_1_na_list = tmp[tmp['thm_1_na_ratio'] > 50]['sequence_id'].values
thm_2_na_list = tmp[tmp['thm_2_na_ratio'] > 50]['sequence_id'].values
thm_3_na_list = tmp[tmp['thm_3_na_ratio'] > 50]['sequence_id'].values
thm_4_na_list = tmp[tmp['thm_4_na_ratio'] > 50]['sequence_id'].values
thm_5_na_list = tmp[tmp['thm_5_na_ratio'] > 50]['sequence_id'].values

all_ids = np.concatenate([thm_1_na_list, thm_2_na_list, thm_3_na_list, thm_4_na_list, thm_5_na_list])
thm_na_list = np.unique(all_ids)
len(thm_na_list)


df = df[~df['sequence_id'].isin(thm_na_list)].reset_index(drop=True)


tof_list = [col for col in df.columns if 'tof' in col ]

tof_1_list = tof_list[:64]
tof_2_list = tof_list[64:128]
tof_3_list = tof_list[128:192]
tof_4_list = tof_list[192:256]
tof_5_list = tof_list[256:320]


def nan_ratio_mean(df, cols):
    return df[cols].isna().mean(axis=1).groupby(df['sequence_id']).mean() * 100

tof_1_na = nan_ratio_mean(df, tof_1_list).rename('tof_1_na_ratio')
tof_2_na = nan_ratio_mean(df, tof_2_list).rename('tof_2_na_ratio')
tof_3_na = nan_ratio_mean(df, tof_3_list).rename('tof_3_na_ratio')
tof_4_na = nan_ratio_mean(df, tof_4_list).rename('tof_4_na_ratio')
tof_5_na = nan_ratio_mean(df, tof_5_list).rename('tof_5_na_ratio')

tmp = pd.concat([tof_1_na, tof_2_na, tof_3_na, tof_4_na, tof_5_na], axis=1).reset_index()


tof_1_na_list = tmp[tmp['tof_1_na_ratio'] > 50]['sequence_id'].values
tof_2_na_list = tmp[tmp['tof_2_na_ratio'] > 50]['sequence_id'].values
tof_3_na_list = tmp[tmp['tof_3_na_ratio'] > 50]['sequence_id'].values
tof_4_na_list = tmp[tmp['tof_4_na_ratio'] > 50]['sequence_id'].values
tof_5_na_list = tmp[tmp['tof_5_na_ratio'] > 50]['sequence_id'].values

all_ids = np.concatenate([tof_1_na_list, tof_2_na_list, tof_3_na_list, tof_4_na_list, tof_5_na_list])
tof_na_list = np.unique(all_ids)
len(tof_na_list)


df = df[~df['sequence_id'].isin(tof_na_list)].reset_index(drop=True)


tmp = df.groupby(['subject','sequence_id'])['behavior'].value_counts().unstack().reset_index().fillna(0)
tmp = tmp.rename(columns={'Hand at target location':'Pause_len', 'Performs gesture': 'Gesture_len'})

df = df.merge(tmp[['sequence_id','Gesture_len','Pause_len']], on='sequence_id', how='left')


class CMI_Transform:
    def __init__(self, always_apply=False, p=0.5):
        self.always_apply = always_apply
        self.p = p

    def __call__(self, y):
        if self.always_apply:
            return self.apply(y)
        else:
            if np.random.rand() < self.p:
                return self.apply(y)
            else:
                return y

    def apply(self, y):
        raise NotImplementedError

class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, y):
        for trans in self.transforms:
            y = trans(y)

        return y

class OneOf:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, y):
        n_trans = len(self.transforms)
        trans_idx = np.random.choice(n_trans)
        trans = self.transforms[trans_idx]
        return trans(y)


class Phase_Locked_Time_Shift(CMI_Transform):
    def __init__(self, start, end, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.start = int(start)
        self.end = int(end)
        
    def apply(self, x):
        L = x.shape[0]
        trans_shift = int(np.random.uniform(-0.15*L, 0.15*L, size=1))
        pause_shift = int(np.random.uniform(-0.15*L, 0.15*L, size=1))
        gesture_shift = int(np.random.uniform(-0.05*L, 0.05*L, size=1))

        x[:self.start] = np.roll(x[:self.start], trans_shift, axis = 0) # Transition Phase 
        x[self.start:self.end] = np.roll(x[self.start:self.end], pause_shift, axis = 0) # Pause Phase
        x[self.end:] = np.roll(x[self.end:], gesture_shift, axis = 0)
        
        return x


class Jitter_Scale(CMI_Transform):
    def __init__(self, always_apply=False, p=0.5, max_noise_amplitude=0.05, scale=0.1):
        super().__init__(always_apply, p)
        self.max_noise_amplitude = max_noise_amplitude
        self.scale=0.1

    def apply(self, x):
        noise  = np.random.randn(*x.shape) * self.max_noise_amplitude
        scale  = np.random.uniform(1-self.scale,
                                   1+self.scale,
                                   size=(1, x.shape[1]))

        return (x + noise) * scale



raw_acc_feature = ['acc_x','acc_y','acc_z']
raw_rot_feature = ['rot_x','rot_y','rot_z','rot_w']


def gravity_remove(imu_sequence):
    
    acc_values = imu_sequence[raw_acc_feature].values
    rot_values = imu_sequence[raw_rot_feature].values

    gravity_world = np.array([0,0,9.81])
    
    rotation = R.from_quat(rot_values)
    gravity_sensor = rotation.apply(gravity_world, inverse=True)

    linear_acc = acc_values - gravity_sensor
    
    return linear_acc


def compute_angular_velocity(quat_sequence, sample_rate=128):
    
    time_steps = quat_sequence.shape[0]
    delta_t = 1.0 / sample_rate
    angular_velocity = np.zeros((time_steps, 3))
    
    for i in range(time_steps - 1):
        q_current = quat_sequence[i]
        q_next = quat_sequence[i + 1]

        rot_current = R.from_quat(q_current)
        rot_next = R.from_quat(q_next)

        delta_rot = rot_current.inv() * rot_next

        angular_velocity[i, :] = delta_rot.as_rotvec() / delta_t

    angular_velocity[-1, :] = angular_velocity[-2, :] 

    return angular_velocity



def compute_angular_distance(quat_sequence):
 
    seq_length = quat_sequence.shape[0]
    angular_distance = np.zeros(seq_length)
    
    for i in range(seq_length - 1):
        q_current = quat_sequence[i]
        q_next = quat_sequence[i + 1]

        r_current = R.from_quat(q_current)
        r_next = R.from_quat(q_next)

        delta_rot = r_current.inv() * r_next
        angular_distance[i] = np.linalg.norm(delta_rot.as_rotvec())

    angular_distance[-1] = angular_distance[-2]

    return angular_distance


def IMU_Extractor(df):

    # df[raw_acc_feature] = df[raw_acc_feature].apply(
    # lambda row: row.fillna(
    #     row.dropna().mean() if row.notna().any() else 0
    # ),
    # axis=1
    # )

    # df[raw_rot_feature] = df[raw_rot_feature].apply(
    # lambda row: row.fillna(
    #     row.dropna().mean() if row.notna().any() else 0
    # ),
    # axis=1
    # )
    
    
    ## Gravity Remove
    linear_acc_df = df.groupby('sequence_id', group_keys=False).apply(
        lambda df: pd.DataFrame(
            gravity_remove(df[raw_acc_feature+raw_rot_feature]),
            columns = raw_acc_feature,
            index = df.index,
        )
    )
    ## Acc Feature
    df = df.drop(columns=raw_acc_feature)
    df[raw_acc_feature] = linear_acc_df

    for axis in ['x', 'y', 'z']:
        df[f'acc_{axis}_jerk'] = df.groupby('sequence_id')[f'acc_{axis}'].diff().fillna(0)

    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['acc_mag_snap'] = df.groupby('sequence_id')['acc_mag_jerk'].diff().fillna(0)

    ## Rot Feature

    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1,1))
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    df['rot_angle_snap'] = df.groupby('sequence_id')['rot_angle_vel'].diff().fillna(0)


    ## Angular Velocity
    angular_velocity_df = df.groupby('sequence_id', group_keys=False).apply(
                lambda df: pd.DataFrame(
                compute_angular_velocity(df[raw_rot_feature].to_numpy()),
                columns=['angular_vel_x','angular_vel_y','angular_vel_z'],
            index=df.index
            )
        )
    df = df.join(angular_velocity_df)

    for axis in ['x', 'y', 'z']:
        df[f'angular_vel_{axis}_jerk'] = df.groupby('sequence_id')[f'angular_vel_{axis}'].diff().fillna(0)

    df['angular_vel_mag'] = np.sqrt(df['angular_vel_x']**2 + df['angular_vel_y']**2 + df['angular_vel_z']**2)
    df['angular_vel_mag_jerk'] = df.groupby('sequence_id')['angular_vel_mag'].diff().fillna(0)
    df['angular_vel_mag_snap'] = df.groupby('sequence_id')['angular_vel_mag_jerk'].diff().fillna(0)
  
    ## Angular Distance
    angular_distance_df = df.groupby('sequence_id', group_keys=False).apply(
            lambda df: pd.DataFrame(
                compute_angular_distance(df[raw_rot_feature].to_numpy()),
                columns=['angular_distance'],
                index=df.index,
            )
    )
    df = df.join(angular_distance_df)

    ## Hybrid Feature
    df['acc_angular_vel'] = df['acc_mag'] * df['angular_vel_mag']
    df['acc_angular_vel_ratio'] = df['acc_mag'] / (df['angular_vel_mag'] + 1e-6)
    df['acc_angular_distance'] = df['acc_mag'] * df['angular_distance']
    df['acc_angular_distance_ratio'] = df['acc_mag'] / (df['angular_distance'] + 1e-6)

    return df


acc_feature = ['acc_x','acc_y','acc_z',
               'acc_x_jerk', 'acc_y_jerk', 'acc_z_jerk',
               'acc_mag','acc_mag_jerk','acc_mag_snap',
               'acc_angular_vel', 'acc_angular_vel_ratio',  'acc_angular_distance', 'acc_angular_distance_ratio']


rot_feature = ['rot_x','rot_y','rot_z','rot_w',
               'rot_angle', 'rot_angle_vel', 'rot_angle_snap',
               'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 
               'angular_vel_x_jerk', 'angular_vel_y_jerk', 'angular_vel_z_jerk',
               'angular_vel_mag', 'angular_vel_mag_jerk', 'angular_vel_mag_snap', 
               'angular_distance']


%%time

df = IMU_Extractor(df)


raw_thm_feature = ['thm_1','thm_2','thm_3','thm_4','thm_5']


def mean_abs_chain(diffs):
    return sum(d.abs() for d in diffs) / len(diffs)


def THM_Extractor(df):

    mask = df['thm_3'] == 0
    df.loc[mask, 'thm_3'] = np.mean(df.loc[mask, 'thm_2'])
    
    # df[raw_thm_feature] = df[raw_thm_feature].apply(
    # lambda row: row.fillna(
    #     row.dropna().mean() if row.notna().any() else 0
    # ),
    # axis=1
    # )
    
    ## Chain Feature
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

    
    ## Chaine Expansion Feature
    df['thm_right_left_diff'] = df['thm_3'] - df['thm_5']
    df['thm_top_bottom_diff'] = df[['thm_1','thm_2']].mean(axis=1) - df['thm_4']

    
    ## Stats Feature
    df['thm_top_mean'] = df[['thm_1','thm_2']].mean(axis=1)
    df['thm_middle_mean'] =df[['thm_5','thm_3']].mean(axis=1)
    df['thm_mean'] = df[raw_thm_feature].mean(axis=1)
    df['thm_std'] = df[raw_thm_feature].std(axis=1)
    df['thm_range'] = df[raw_thm_feature].max(axis=1) - df[raw_thm_feature].min(axis=1)
    
    return df 


thm_feature = ['thm_1','thm_2','thm_3','thm_4','thm_5',
               'thm_right_chain','thm_left_chain','thm_center_chain','thm_height_chain',
               'thm_top_bottom_diff', 'thm_right_left_diff',
               'thm_top_mean','thm_middle_mean','thm_mean','thm_std','thm_range'
                   ]


%%time 

df = THM_Extractor(df)


raw_tof_feature = ['tof_1_mean','tof_2_mean','tof_3_mean','tof_4_mean','tof_5_mean']


tof_top_left = []; tof_top_right = [];
tof_bottom_left = []; tof_bottom_right = []

tof_list = [tof_1_list, tof_2_list, tof_3_list, tof_4_list, tof_5_list]

for list_ in tof_list: 

    tof_top_left.extend([list_[8*i + k] for i in range(0, 4) for k in range(0, 4)])
    tof_top_right.extend([list_[8*i + k + 4] for i in range(0, 4) for k in range(0, 4)])
    tof_bottom_left.extend([list_[8*i + k + 32] for i in range(0, 4) for k in range(0, 4)])
    tof_bottom_right.extend([list_[8*i + k + 36] for i in range(0, 4) for k in range(0, 4)])


def TOF_Extractor(df):

    df[tof_1_list] = df[tof_1_list].replace(-1, 0)
    df[tof_2_list] = df[tof_2_list].replace(-1, 0)
    df[tof_3_list] = df[tof_3_list].replace(-1, 0)
    df[tof_4_list] = df[tof_4_list].replace(-1, 0)
    df[tof_5_list] = df[tof_5_list].replace(-1, 0)
    
    df['tof_1_mean'] = df[tof_1_list].fillna(0).mean(axis=1)
    df['tof_2_mean'] = df[tof_2_list].fillna(0).mean(axis=1)
    df['tof_3_mean'] = df[tof_3_list].fillna(0).mean(axis=1)
    df['tof_4_mean'] = df[tof_4_list].fillna(0).mean(axis=1)
    df['tof_5_mean'] = df[tof_5_list].fillna(0).mean(axis=1)

    df['tof_1_std'] = df[tof_1_list].fillna(0).std(axis=1)
    df['tof_2_std'] = df[tof_2_list].fillna(0).std(axis=1)
    df['tof_3_std'] = df[tof_3_list].fillna(0).std(axis=1)
    df['tof_4_std'] = df[tof_4_list].fillna(0).std(axis=1)
    df['tof_5_std'] = df[tof_5_list].fillna(0).std(axis=1)

    df['tof_1_min'] = df[tof_1_list].fillna(0).min(axis=1)
    df['tof_2_min'] = df[tof_2_list].fillna(0).min(axis=1)
    df['tof_3_min'] = df[tof_3_list].fillna(0).min(axis=1)
    df['tof_4_min'] = df[tof_4_list].fillna(0).min(axis=1)
    df['tof_5_min'] = df[tof_5_list].fillna(0).min(axis=1)

    df['tof_1_max'] = df[tof_1_list].fillna(0).max(axis=1)
    df['tof_2_max'] = df[tof_2_list].fillna(0).max(axis=1)
    df['tof_3_max'] = df[tof_3_list].fillna(0).max(axis=1)
    df['tof_4_max'] = df[tof_4_list].fillna(0).max(axis=1)
    df['tof_5_max'] = df[tof_5_list].fillna(0).max(axis=1)

    # df[raw_tof_feature] = df[raw_tof_feature].apply(
    # lambda row: row.where(row != 0, row[row != 0].mean() if (row != 0).any() else 0),
    # axis=1
    # )

    df['tof_mean'] = df[raw_tof_feature].mean(axis=1)

    df['tof_top_left_mean'] = df[tof_top_left].fillna(0).mean(axis=1)
    df['tof_top_right_mean'] = df[tof_top_right].fillna(0).mean(axis=1)
    df['tof_bottom_left_mean'] = df[tof_bottom_left].fillna(0).mean(axis=1)
    df['tof_bottom_right_mean'] = df[tof_bottom_right].fillna(0).mean(axis=1)


    df['tof_top_diff'] = df['tof_top_left_mean'] - df['tof_top_right_mean']
    df['tof_bottom_diff'] = df['tof_bottom_left_mean'] - df['tof_bottom_right_mean']
    df['tof_vertical_diff'] = (df['tof_top_left_mean'] + df['tof_top_right_mean']) - (df['tof_bottom_left_mean'] + df['tof_bottom_right_mean'])
    df['tof_horizontal_diff'] = (df['tof_top_left_mean'] + df['tof_bottom_left_mean']) - (df['tof_top_right_mean'] + df['tof_bottom_right_mean'])

    df['tof_top_bottom_ratio'] = (df['tof_top_left_mean'] + df['tof_top_right_mean']) / (df['tof_bottom_left_mean'] + df['tof_bottom_right_mean'] + 1e-6)

    df['tof_left_right_ratio'] = (df['tof_top_left_mean'] + df['tof_bottom_left_mean']) / (df['tof_top_right_mean'] + df['tof_bottom_right_mean'] + 1e-6)

    return df


tof_feature = ['tof_1_mean','tof_2_mean','tof_3_mean','tof_4_mean','tof_5_mean',
               'tof_1_std','tof_2_std','tof_3_std','tof_4_std','tof_5_std',
               'tof_1_min','tof_2_min','tof_3_min','tof_4_min','tof_5_min',
               'tof_1_max','tof_2_max','tof_3_max','tof_4_max','tof_5_max',
               'tof_top_left_mean','tof_top_right_mean','tof_bottom_left_mean','tof_bottom_right_mean',
               'tof_mean',
               'tof_top_diff','tof_bottom_diff','tof_vertical_diff','tof_horizontal_diff',
               'tof_top_bottom_ratio', 'tof_left_right_ratio'
              ]
tof_feature.extend(tof_1_list)
tof_feature.extend(tof_2_list)
tof_feature.extend(tof_3_list)
tof_feature.extend(tof_4_list)
tof_feature.extend(tof_5_list)


%%time

df = TOF_Extractor(df)


total_feature = acc_feature + rot_feature + thm_feature + tof_feature

print(f"{c_}{sr_}=> Acc Feature Count is {len(acc_feature)}\n")
print(f"{c_}{sr_}=> Rot Feature Count is {len(rot_feature)}\n")
print(f"{c_}{sr_}=> Thm Feature Count is {len(thm_feature)}\n")
print(f"{c_}{sr_}=> Tof Feature Count is {len(tof_feature)}\n")


label_to_num = {
    'Above ear - pull hair': 0,  # < ------- TARGETS START
    'Cheek - pinch skin': 1,
    'Eyebrow - pull hair': 2,
    'Eyelash - pull hair': 3,
    'Forehead - pull hairline': 4,
    'Forehead - scratch': 5,
    'Neck - pinch skin': 6,
    'Neck - scratch': 7,  # < ------- TARGETS END
    'Drink from bottle/cup': 8,  # < ------- NON-TARGETS START
    'Feel around in tray and pull out an object': 9,
    'Glasses on/off': 10,
    'Pinch knee/leg skin': 11,
    'Pull air toward your face': 12,
    'Scratch knee/leg skin': 13,
    'Text on phone': 14,
    'Wave hello': 15,
    'Write name in air': 16,
    'Write name on leg': 17  # < ------- NON-TARGETS END
}

num_to_label = {k: v for k, v in enumerate(label_to_num)}


# df['gesture'] = df['gesture'].map(label_to_num)

# one hot encoding
label_list = []

for label, _ in label_to_num.items():
    df[label] = np.where(df['gesture'] == label, 1.0 - CFG.label_smoothing, CFG.label_smoothing / (CFG.num_classes - 1)).astype('float32')
    label_list.append(label)

print(label_list)


sgkf = StratifiedGroupKFold(n_splits=CFG.n_splits, shuffle=True, random_state=CFG.seed)

for i, (_, val_idx) in enumerate(sgkf.split(df, df['gesture'], groups=df['subject'])):
    df.loc[val_idx, 'fold'] = i


class CMI_Dataset(Dataset):
    def __init__(self, data, col, transforms=False, mix_up=False):
        super(CMI_Dataset, self).__init__()
        self.data = data
        self.seq_list = data['sequence_id'].unique()
        self.col = col
        self.transforms = transforms
        self.mix_up = mix_up

    def __len__(self):
        return len(self.seq_list)

    def __getitem__(self, index):

        X, y= self.__data_generation(index)

        if self.mix_up and (np.random.rand() < CFG.mix_up_prob):
            mix_index = np.random.randint(len(self.seq_list))
            mix_X, mix_y = self.__data_generation(mix_index)
            
            lam = np.random.beta(CFG.mix_up_alpha, CFG.mix_up_alpha)

            X = lam * X + (1 - lam) * mix_X
            y = lam * y + (1 - lam) * mix_y
        

        X = torch.tensor(X, dtype=torch.float32).permute(1,0)
        y = torch.tensor(y, dtype=torch.float32)

        return X, y

    def __data_generation(self, index):
        X = np.zeros((CFG.sequence_len,len(self.col)), dtype='float32')
        y = np.zeros(CFG.num_classes, dtype='float32')

        seq_id = self.seq_list[index]
        row = self.data[self.data['sequence_id'] == seq_id]

        inter_ratio = CFG.sequence_len / len(row)
        end = int((CFG.sequence_len - row['Gesture_len'].values[0] * inter_ratio))
        start = int((CFG.sequence_len - (row['Gesture_len'].values[0]+row['Pause_len'].values[0]) * inter_ratio))

        for i, col in enumerate(self.col):
            signal = inter_signal(row[col].values)

            # normalization
            if col not in raw_rot_feature:
                mean = np.mean(signal)
                eps = 1e-9
                std = np.std(signal) + eps
                signal = (signal - mean) / std

            X[:,i] = signal

        ## Augmentation
        if self.transforms:
            augment = Compose([
                Phase_Locked_Time_Shift(start=start, end=end, p=0.5),
                Jitter_Scale(max_noise_amplitude=0.02, scale=0.1, p=0.5),

            ])

            X = augment(X)

        ## THM, TOF 
        if np.random.rand() < 0.3:
            X[:,len(acc_feature+rot_feature):] = 0


        y = row[label_list].values[0]

        return X, y


_, augmented_valid_loader = prepare_loader(fold=CFG.fold, valid_transform=True)
_, valid_loader = prepare_loader(fold=CFG.fold, valid_transform=False)

augmented_seqs, tars = next(iter(augmented_valid_loader))
seqs, _ = next(iter(valid_loader))

del valid_loader


ROWS = 2; COLS = 8

fig, axes = plt.subplots(ROWS, COLS, figsize=(2*COLS, 2*ROWS))
axes = axes.flatten()


for j in range(COLS):

    seq = seqs[j].permute(1,0).cpu().numpy()
    augmented_seq = augmented_seqs[j].permute(1,0).cpu().numpy()

    for k in range(len(raw_acc_feature)):
        axes[j].plot(seq[:,k])
        axes[j].set_title('Before Augmented')

    for k in range(len(raw_acc_feature)):
        axes[COLS+j].plot(augmented_seq[:,k])
        axes[COLS+j].set_title('After Augmented')
        

plt.tight_layout()
plt.show()


class Wave_Block(nn.Module):
    def __init__(self, in_channels, out_channels, dilation_rates, kernel_size=3):
        super(Wave_Block, self).__init__()
        self.num_rates = dilation_rates
        self.convs = nn.ModuleList()
        self.filter_convs = nn.ModuleList()
        self.gate_convs = nn.ModuleList()
        self.convs.append(nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True))
        dilation_rates = [2 ** i for i in range(dilation_rates)]

        for dilation_rate in dilation_rates:
            self.filter_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate)
            )
            self.gate_convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                          padding=int((dilation_rate*(kernel_size-1))/2), dilation=dilation_rate)
            )
            self.convs.append(
                nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=True)
            )

        for i in range(len(self.convs)):
            nn.init.xavier_uniform_(self.convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.convs[i].bias)
            
        for i in range(len(self.filter_convs)):
            nn.init.xavier_uniform_(self.filter_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.filter_convs[i].bias)
            
        for i in range(len(self.gate_convs)):
            nn.init.xavier_uniform_(self.gate_convs[i].weight, gain=nn.init.calculate_gain('relu'))
            nn.init.zeros_(self.gate_convs[i].bias)

    def forward(self,x):
        x = self.convs[0](x)
        res = x # skip_connection
        for i in range(self.num_rates):
            tanh_out = torch.tanh(self.filter_convs[i](x))
            sigmoid_out = torch.sigmoid(self.gate_convs[i](x))
            x = tanh_out * sigmoid_out
            x = self.convs[i+1](x)
            res = res + x
    
        return res


class WaveNet(nn.Module):
    def __init__(self, kernel_size, in_channels = 1):
        super(WaveNet, self).__init__()

        self.model = nn.Sequential(
            Wave_Block(in_channels, 64, 2, kernel_size),
             nn.MaxPool1d(2),
             nn.Dropout(0.4),
             Wave_Block(64, 128, 2, kernel_size),
             nn.MaxPool1d(2),
             nn.Dropout(0.2),
        )

    def forward(self, x):
        output = self.model(x)
        return output


class TemporalAttentionPooling(nn.Module):
    def __init__(self, input_channels, hidden_dim=64):
        super(TemporalAttentionPooling, self).__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(hidden_dim, 1, kernel_size=1)
        )

    def forward(self, x):
        attn_scores = self.attn(x)
        attn_weights = F.softmax(attn_scores, dim=-1)

        weighted = x * attn_weights
        pooled = weighted.sum(dim=-1)

        return pooled


class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool1d(1)

        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels, in_channels//reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels//reduction, in_channels, 1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):

        scale = self.avgpool(input)
        scale = self.mlp(scale)
        

        return input * scale

class Residual_Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Residual_Block, self).__init__()

        self.model = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding='same', bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU()
        )

        self.se_block = SEBlock(out_channels)

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, padding='same', bias=False),
            nn.BatchNorm1d(out_channels)
        )

        self.dropout = nn.Dropout(0.2)  
        self.maxpool = nn.MaxPool1d(2)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        out = self.model(x)
        out = self.se_block(out)
        out = out + shortcut
        out = F.relu(out)
        out = self.maxpool(out)
        out = self.dropout(out)

        return out


class Res_CNN_GRU_Attention(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(Res_CNN_GRU_Attention, self).__init__()

        
        self.wave_model = WaveNet(kernel_size, in_channels=in_channels)

        self.res_model1 = Residual_Block(in_channels=in_channels, out_channels=out_channels//2, kernel_size=kernel_size)
        self.res_model2 = Residual_Block(in_channels=out_channels//2, out_channels=out_channels, kernel_size=kernel_size+2)

        self.gru = nn.ModuleList([
            nn.GRU(input_size=out_channels, hidden_size=out_channels//2, num_layers=1, 
                          batch_first=True, bidirectional=True),
            nn.GRU(input_size=out_channels, hidden_size=out_channels//2, num_layers=1, 
                          batch_first=True, bidirectional=True),
        
            ])
        self.attn_pool = nn.ModuleList([
            TemporalAttentionPooling(input_channels=128, hidden_dim=64),
            TemporalAttentionPooling(input_channels=128, hidden_dim=64),
        ])
            
         
     

    def forward(self, x):
        
        x_trans, x_ges = x[:, :, :CFG.sequence_len//2], x[:, :, CFG.sequence_len//2:]

        ## wave_branch
        x_trans = self.wave_model(x_trans)
        gru_trans, _ = self.gru[0](x_trans.permute(0,2,1))
        trans_out = self.attn_pool[0](gru_trans.permute(0,2,1))

        ## cnn_branch
        x_ges = self.res_model1(x_ges)
        x_ges = self.res_model2(x_ges)
        gru_ges, _ = self.gru[1](x_ges.permute(0,2,1))
        ges_out = self.attn_pool[1](gru_ges.permute(0,2,1))

        return trans_out, ges_out
         


class CNN_GRU_Attention(nn.Module):
    def __init__(self, device_feature):
        super(CNN_GRU_Attention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Wave
        self.wave_feature = nn.Sequential(
             Wave_Block(device_feature, 32, 2, 3),
             nn.MaxPool1d(2),
             nn.Dropout(0.2),
             Wave_Block(32, 64, 2, 3),
             nn.MaxPool1d(2),
             nn.Dropout(0.2),
        )

        # 1D CNN
        self.conv1 = nn.Conv1d(in_channels=device_feature, out_channels=64, kernel_size=3, 
                               padding='same', bias=False)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.max_pool1 = nn.MaxPool1d(2)
        self.max_pool2 = nn.MaxPool1d(2)
        self.dropout1 = nn.Dropout(0.2)

        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, 
                               padding='same', bias=False)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(0.2)
        

        # GRU
        self.gru = nn.ModuleList([
            nn.GRU(input_size=64, hidden_size=32, num_layers=1, 
                          batch_first=True, bidirectional=True),
            nn.GRU(input_size=128, hidden_size=64, num_layers=1, 
                          batch_first=True, bidirectional=True),
            ])
        
        self.attn_pool = nn.ModuleList([
            TemporalAttentionPooling(input_channels=64, hidden_dim=32),
            TemporalAttentionPooling(input_channels=128, hidden_dim=64),
        ])
            

    def forward(self, x):
        
        x_trans, x_ges = x[:, :, :CFG.sequence_len//2], x[:, :, CFG.sequence_len//2:]

        # wave branch with transition part
        x_trans = self.wave_feature(x_trans)

        gru_trans, _ = self.gru[0](x_trans.permute(0,2,1))
        trans_out = self.attn_pool[0](gru_trans.permute(0,2,1))

        # cnn_branch with gesture part
        x_ges = self.conv1(x_ges)
        x_ges = self.batch_norm1(x_ges)
        x_ges = self.relu(x_ges)
        x_ges = self.max_pool1(x_ges)
        x_ges = self.dropout1(x_ges)

        x_ges = self.conv2(x_ges)
        x_ges = self.batch_norm2(x_ges)
        x_ges = self.relu(x_ges)
        x_ges = self.max_pool2(x_ges)
        x_ges = self.dropout2(x_ges)

        gru_ges, _ = self.gru[1](x_ges.permute(0,2,1))
        ges_out = self.attn_pool[1](gru_ges.permute(0,2,1))

        return trans_out, ges_out


class Multi_Model(nn.Module):
    def __init__(self, kernel_size):
        super(Multi_Model, self).__init__()
    
        self.imu_model = Res_CNN_GRU_Attention(in_channels=len(acc_feature+rot_feature), out_channels=128, kernel_size=3)

        self.thm_tof_model = CNN_GRU_Attention(device_feature = len(thm_feature+tof_feature))

        self.dropout = nn.Dropout(0.2)

    def forward(self, x):

        # acc
        imu_trans, imu_ges = self.imu_model(x[:,0:30,:])

        # thm + tof
        thm_tof_trans, thm_tof_ges = self.thm_tof_model(x[:,30:,:])

        thm_tof_trans = thm_tof_trans
        thm_tof_ges   = thm_tof_ges
        
        trans_out = torch.cat([imu_trans, thm_tof_trans], dim=1)
        trans_out = self.dropout(trans_out)

        ges_out = torch.cat([imu_ges, thm_tof_ges], dim=1)
        ges_out = self.dropout(ges_out)

        y = torch.cat([trans_out, ges_out], dim=1)

        return y


!pip install -q torchinfo
from torchinfo import summary

summary(
        Multi_Model(kernel_size=CFG.kernel_size).cpu(), 
        input_size=(1, len(total_feature), CFG.sequence_len), 
        col_names=["input_size", "output_size", "num_params", "mult_adds"],
        depth=2,
    )


def SoftCrossEntropy(pred, target, class_weight=None):
    log_probs = torch.log_softmax(pred, dim=1)
    if class_weight is not None:
        loss = -(target * log_probs * class_weight).sum(dim=1).mean()
    else:
        loss = -(target * log_probs).sum(dim=1).mean()
    return loss


class_weight = torch.tensor([1.2]*8 + [0.8]*10).to(CFG.device)
class_weight


class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (B, feature_dim)
        # labels: (B,)
        device = features.device
        batch_size = features.shape[0]

        features = F.normalize(features, dim=1)  # L2 normalization
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(device)  # (B, B)

        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )

        # negative mask
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)
        mask = mask * logits_mask  # remove self-comparison

        # compute log_prob
        exp_logits = torch.exp(anchor_dot_contrast) * logits_mask
        log_prob = anchor_dot_contrast - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + 1e-12)

        loss = -mean_log_prob_pos.mean()
        return loss


from torchmetrics.classification import MulticlassF1Score, BinaryF1Score

class Final_Model(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.feature_extractor = Multi_Model(kernel_size=CFG.kernel_size)

        dummy_input = torch.randn(1, len(total_feature), CFG.sequence_len)
        with torch.no_grad():
            dummy_out = self.feature_extractor(dummy_input)
        feature_dim = dummy_out.shape[1]

        self.head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim//2, bias=False),
            nn.BatchNorm1d(feature_dim//2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(feature_dim//2, CFG.num_classes)
        )
        self.supcon_criterion = SupConLoss(temperature=0.2)

        self.train_f1 = MulticlassF1Score(num_classes=CFG.num_classes, average='macro')
        self.valid_f1 = MulticlassF1Score(num_classes=CFG.num_classes, average='macro')

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.head(out)
        return out

    def training_step(self, batch, batch_idx):

        seqs, tars = batch
        features = self.feature_extractor(seqs)

        outs = self.head(features)

        loss = SoftCrossEntropy(outs, tars, class_weight=class_weight)

        preds = outs.argmax(dim=1)
        hard_tars = tars.argmax(dim=1)
        
        supcon_loss = self.supcon_criterion(features, hard_tars)

        self.train_f1.update(preds, hard_tars)

        self.log('Train Loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('Train Supcon Loss', supcon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('Train F1-Score', self.train_f1, on_step=False, on_epoch=True, prog_bar=True)
        
        
        total_loss = loss * 1.0 + supcon_loss * 0.1

        return loss

    def validation_step(self, batch, batch_idx):

        seqs, tars = batch
        features = self.feature_extractor(seqs)
        outs = self.head(features)

        loss = SoftCrossEntropy(outs, tars, class_weight=class_weight)
        
        preds = outs.argmax(dim=1)
        hard_tars = tars.argmax(dim=1)
        supcon_loss = self.supcon_criterion(features, hard_tars)

        self.valid_f1.update(preds, hard_tars)

        self.log('Valid Loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('Valid Supcon Loss', supcon_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('Valid F1-Score', self.valid_f1, on_step=False, on_epoch=True, prog_bar=True)


    def on_train_epoch_start(self):
        self.train_f1.reset()

    def on_validation_epoch_start(self):
        self.valid_f1.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=CFG.lr, weight_decay=CFG.wd)
        warmup_epochs = CFG.warmup_epochs
        cosine_epochs = max(1, CFG.epochs - warmup_epochs)

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cosine_epochs, eta_min=CFG.min_lr
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs]
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
            'scheduler': scheduler,
            'interval': 'epoch',
            'frequency': 1
            }
        }


from pytorch_lightning.callbacks import StochasticWeightAveraging

swa_callback = StochasticWeightAveraging(
    swa_epoch_start=int(CFG.epochs * 0.75),      
    swa_lrs= 3e-4,          
    annealing_epochs=int(CFG.epochs * 0.25),
    annealing_strategy='cos',
)


from pytorch_lightning.loggers import WandbLogger

for fold in range(CFG.n_fold):
    print(f"Training fold {fold}...")


    train_loader, valid_loader = prepare_loader(fold=fold)

    wandb_logger = WandbLogger(
        offline = CFG.offline,
        log_model= CFG.log_model,
        project=CFG.project,
        name=f"{CFG.model_name}_debug_{CFG.debug}",
        group=CFG.comment,
        anonymous = anonymous,
        config={
            'cv': 'sgkf',
            'fold': CFG.fold,
            'seq_length': CFG.sequence_len,
            'epoch': CFG.epochs,
            'batch_size': CFG.train_bs,
            'model_name': CFG.model_name,
            'base_lr': CFG.lr,
            'min_lr': CFG.min_lr,
            'optimizer': 'AdamW',
            'weight_decay': CFG.wd,
            'scheduler': CFG.scheduler,
        }
        )

    trainer = pl.Trainer(
        max_epochs=CFG.epochs,
        accelerator='gpu',
        devices=1,
        precision='16-mixed',
        callbacks = [LearningRateMonitor(logging_interval='epoch'),
        # swa_callback,
             ],

        logger=wandb_logger
        )

    cls_model = Final_Model();

    trainer.fit(model = cls_model, train_dataloaders = train_loader,
           val_dataloaders = valid_loader)
    trainer.save_checkpoint(f"{CFG.save_dir}last_fold{fold}.ckpt")

    del cls_model, trainer, train_loader, valid_loader
    torch.cuda.empty_cache()

wandb.finish()


wandb_logger.experiment ## Click Display W&B


ckpt_path = f"{CFG.save_dir}last_fold0.ckpt"

cls_model = Final_Model.load_from_checkpoint(ckpt_path)
cls_model.eval();

_, valid_loader = prepare_loader(fold=0)

feature_list = []
labels = []

with torch.no_grad():
    for batch in tqdm(valid_loader):
        seqs, tars = batch
        feature = cls_model.feature_extractor(seqs)

        feature_list.append(feature)
        labels.append(tars)

feature_list = torch.concat(feature_list, dim=0).cpu().numpy()
labels = torch.concat(labels, dim=0).cpu().numpy()

del cls_model; clean_memory()


import umap

reducer = umap.UMAP(n_components=2, random_state=42)
embedding = reducer.fit_transform(feature_list)

plt.figure(figsize=(8,6))
sns.scatterplot(
    x=embedding[:,0], y=embedding[:,1],
    hue=np.minimum(labels.argmax(axis=1), 8),
    palette='tab10',
    s=50,
    alpha=0.8
)
plt.title('UMAP of Feature Extractor Output')
plt.show()


from sklearn.metrics import f1_score, accuracy_score

all_val_pred = []; all_val_true = []


for fold in range(CFG.n_fold):
    print(f"Evaluating fold {fold}...")

    ckpt_path = f"{CFG.save_dir}last_fold{fold}.ckpt"

    cls_model = Final_Model.load_from_checkpoint(ckpt_path)
    cls_model.eval(); cls_model.to(CFG.device)

    _, valid_loader = prepare_loader(fold=fold)

    val_pred = []; val_true = []

    with torch.no_grad():
        for batch in tqdm(valid_loader):
            seqs, tars = batch
            seqs = seqs.to(CFG.device)
            outs = cls_model(seqs)

            val_pred.append(outs.argmax(dim=1).cpu().numpy())
            val_true.append(tars.argmax(dim=1).cpu().numpy())

    val_pred = np.concatenate(val_pred, axis=0)
    val_true = np.concatenate(val_true, axis=0)

    binary_f1 = f1_score((val_true < 8).astype(int), (val_pred < 8).astype(int), pos_label=True, zero_division=0, average='binary')
    macro_f1 = f1_score(np.where(val_true >= 8, 8, val_true), np.where(val_pred >= 8, 8, val_pred), average='macro', zero_division=0)
    total_f1 = binary_f1 * 0.5 + macro_f1 * 0.5

    print(f'{c_}{sr_}#'*25)
    print(f"Fold {fold}")
    print(f"binary f1: {binary_f1}")
    print(f"macro f1: {macro_f1}")
    print(f"total f1: {total_f1}")
    print('#'*25)

    all_val_pred.append(val_pred)
    all_val_true.append(val_true)

all_val_pred = np.concatenate(all_val_pred, axis=0)
all_val_true = np.concatenate(all_val_true, axis=0)


import plotly.graph_objects as go
from plotly.subplots import make_subplots

cls_pred_df = pd.DataFrame(all_val_pred, columns=['pred'])
cls_true_df = pd.DataFrame(all_val_true, columns=['true'])

acc = accuracy_score((cls_true_df['true'] < 8).astype(int), (cls_pred_df['pred'] < 8).astype(int))
f1 = f1_score((cls_true_df['true'] < 8).astype(int), (cls_pred_df['pred'] < 8).astype(int))

x_min = 0
x_max = 1.001
bin_size = 0.05

trace1 = go.Histogram(
    x=cls_pred_df['pred'],
    opacity=0.5,
    name='Predicted',
    marker=dict(color='blue'),
    xbins=dict(
        start=x_min,
        end=x_max,
        size=bin_size
    )
)

trace2 = go.Histogram(
    x=cls_true_df['true'],
    opacity=0.5,
    name='True',
    marker=dict(color='pink'),
    xbins=dict(
        start=x_min,
        end=x_max,
        size=bin_size
    )
)

data = [trace1, trace2]

layout = go.Layout(
    title=f"Target(1) vs Non-Target(0): Accuracy is {acc*100:.1f}, F1-Score is {f1*100:.1f}%",
    barmode='overlay',
    legend=dict(
        bordercolor='black',
        borderwidth=1
    ),
    xaxis_title="Value",
    yaxis_title="Count"
)

fig = go.Figure(data=data, layout=layout)

fig.show(renderer="iframe")

