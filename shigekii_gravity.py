from tqdm import tqdm
import pandas as pd
import sys
folder_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data"
#sys.path.append(folder_path)
sys.path.append("/kaggle/input/utils-010")

row_train_demo = pd.read_csv(f"{folder_path}/train_demographics.csv")
row_train = pd.read_csv(f"{folder_path}/train.csv")


import matplotlib.pyplot as plt
import numpy as np


import torch
import numpy as np
import pandas as pd
import random
import os

def set_seed(seed):
    """
    主要なライブラリの乱数シードを固定する関数。
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 複数のGPUを使用する場合
        # CUDNN関連のシードも固定して、決定論的挙動を保証
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"乱数シードを {seed} に設定しました。")

SEED = 42
set_seed(SEED)


train=row_train.copy()
train_demo = row_train_demo.copy()


train = train.iloc[:, :-325]
#train = train[train["behavior"] == "Hand at target location"]


import pandas as pd
import numpy as np

def calculate_dynamic_acceleration(train):
    """
    クォータニオンから回転行列を導出し、重力ベクトルを減算して動的加速度を計算します。
    
    引数:
        train (pd.DataFrame): 'acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z'
                              の各列を含むDataFrame。
    
    戻り値:
        pd.DataFrame: 動的加速度の3軸 ('acc_x_dynamic', 'acc_y_dynamic', 'acc_z_dynamic')
                      と、回転後の重力ベクトル ('gravity_x', 'gravity_y', 'gravity_z') 
                      を追加したDataFrame。
    """
    quaternions = train[['rot_w', 'rot_x', 'rot_y', 'rot_z']].values.astype(np.float64)
    accelerations = train[['acc_x', 'acc_y', 'acc_z']].values.astype(np.float64)
    
    # 単位クォータニオンであることを保証するために正規化
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions /= norms

    w, x, y, z = quaternions.T

    R = np.zeros((len(train), 3, 3))
    R[:, 0, 0] = 1 - 2*y**2 - 2*z**2
    R[:, 0, 1] = 2*x*y - 2*z*w
    R[:, 0, 2] = 2*x*z + 2*y*w
    R[:, 1, 0] = 2*x*y + 2*z*w
    R[:, 1, 1] = 1 - 2*x**2 - 2*z**2
    R[:, 1, 2] = 2*y*z - 2*x*w
    R[:, 2, 0] = 2*x*z - 2*y*w
    R[:, 2, 1] = 2*y*z + 2*x*w
    R[:, 2, 2] = 1 - 2*x**2 - 2*y**2

    # Z軸が上方向と仮定
    gravity_vector = np.array([0, 0, 9.8])
    
    rotated_gravity = np.einsum('nij, j->ni', R, gravity_vector)

    dynamic_acceleration = accelerations - rotated_gravity
    
    result_df = train.copy()
    result_df['acc_x_dynamic'] = dynamic_acceleration[:, 0]
    result_df['acc_y_dynamic'] = dynamic_acceleration[:, 1]
    result_df['acc_z_dynamic'] = dynamic_acceleration[:, 2]

    result_df['gravity_x'] = rotated_gravity[:, 0]
    result_df['gravity_y'] = rotated_gravity[:, 1]
    result_df['gravity_z'] = rotated_gravity[:, 2]

    return result_df


train = calculate_dynamic_acceleration(train)


def plot_sequence(data):
    rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
    acc_cols = ['acc_x', 'acc_y', 'acc_z']
    grav_cols = ['gravity_x', 'gravity_y', 'gravity_z']
    dynamic_cols = ['acc_x_dynamic', 'acc_y_dynamic', 'acc_z_dynamic']

    unique_behaviors = data['behavior'].unique()
    bg_colors = plt.cm.get_cmap('Paired', len(unique_behaviors))
    
    fig, axes = plt.subplots(4, 1, figsize=(15, 20), sharex=True)
    ax_acc, ax_rot, ax_grav, ax_dyn = axes.flatten()

    fig.suptitle(
        f"Sequence ID: {data['sequence_id'].iloc[0]}, Orientation: {data['orientation'].iloc[0]}, Gesture: {data['gesture'].iloc[0]}, Sequence Type: {data['sequence_type'].iloc[0]}",
        fontsize=16
    )

    for i, (behavior, group) in enumerate(data.groupby('behavior')):
        ax_acc.axvspan(group['sequence_counter'].min(), group['sequence_counter'].max(), 
                       color=bg_colors(i), alpha=0.2, label=f'Behavior: {behavior}')
        ax_rot.axvspan(group['sequence_counter'].min(), group['sequence_counter'].max(), 
                       color=bg_colors(i), alpha=0.2)
        ax_grav.axvspan(group['sequence_counter'].min(), group['sequence_counter'].max(), 
                        color=bg_colors(i), alpha=0.2)
        ax_dyn.axvspan(group['sequence_counter'].min(), group['sequence_counter'].max(), 
                       color=bg_colors(i), alpha=0.2)

    for col in acc_cols:
        ax_acc.plot(data['sequence_counter'], data[col], label=col, linestyle='-')
    ax_acc.set_ylabel('Acceleration (m/s²)')
    ax_acc.legend(loc='upper left')
    ax_acc.grid(True, linestyle='--', alpha=0.6)

    for col in rot_cols:
        ax_rot.plot(data['sequence_counter'], data[col], label=col, linestyle='-')
    ax_rot.set_ylabel('Rotation (quaternion)')
    ax_rot.legend(loc='upper left')
    ax_rot.grid(True, linestyle='--', alpha=0.6)

    for col in grav_cols:
        ax_grav.plot(data['sequence_counter'], data[col], label=col, linestyle='-')
    ax_grav.set_ylabel('Gravity Vector (m/s²)')
    ax_grav.legend(loc='upper left')
    ax_grav.grid(True, linestyle='--', alpha=0.6)

    for col in dynamic_cols:
        ax_dyn.plot(data['sequence_counter'], data[col], label=col, linestyle='-')
    ax_dyn.set_ylabel('Dynamic Acceleration (m/s²)')
    ax_dyn.legend(loc='upper left')
    ax_dyn.grid(True, linestyle='--', alpha=0.6)
    
    ax_dyn.set_xlabel('sequence_counter')
    
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()


# DatasetとDataLoaderのインポート
from torch.utils.data import DataLoader
from dataset import SubjectDataset, custom_collate_fn

train_dataset = SubjectDataset(train_demo, train, drop_cols=False)


count = 0
for demo, data, label in train_dataset:
    if np.random.rand() < 0.5:
        plot_sequence(data)
        count += 1
        if count >=4:
            break


train=row_train.copy()
train_demo = row_train_demo.copy()


train = train.iloc[:, :-325]


# DatasetとDataLoaderのインポート
from torch.utils.data import DataLoader
from dataset import SubjectDataset, custom_collate_fn

train_dataset = SubjectDataset(train_demo, train, drop_cols=False)


import pandas as pd
import numpy as np
from scipy.optimize import minimize_scalar

def calculate_dynamic_acceleration2(train):
    quaternions = train[['rot_w', 'rot_x', 'rot_y', 'rot_z']].values.astype(np.float64)
    accelerations = train[['acc_x', 'acc_y', 'acc_z']].values.astype(np.float64)
    
    norms = np.linalg.norm(quaternions, axis=1, keepdims=True)
    quaternions /= norms

    w, x, y, z = quaternions.T

    R_sensor = np.zeros((len(train), 3, 3))
    R_sensor[:, 0, 0] = 1 - 2*y**2 - 2*z**2
    R_sensor[:, 0, 1] = 2*x*y - 2*z*w
    R_sensor[:, 0, 2] = 2*x*z + 2*y*w
    R_sensor[:, 1, 0] = 2*x*y + 2*z*w
    R_sensor[:, 1, 1] = 1 - 2*x**2 - 2*z**2
    R_sensor[:, 1, 2] = 2*y*z - 2*x*w
    R_sensor[:, 2, 0] = 2*x*z - 2*y*w
    R_sensor[:, 2, 1] = 2*y*z + 2*x*w
    R_sensor[:, 2, 2] = 1 - 2*x**2 - 2*y**2
    
    gravity_vector = np.array([0, 0, 9.8])
    rotated_gravity = np.einsum('nij, j->ni', R_sensor, gravity_vector)

    def objective_function(phi, acc_all, grav_rot_all):
        n = len(acc_all)
        trim_size = int(n * 0.3)
        if trim_size == 0:
            return 0.0
            
        cos_phi, sin_phi = np.cos(phi), np.sin(phi)
        R_z = np.array([[cos_phi, -sin_phi, 0],
                        [sin_phi,  cos_phi, 0],
                        [      0,        0, 1]])

        acc_rotated = np.einsum('ij, kj->ki', R_z, acc_all)
        dynamic_acc = acc_rotated - grav_rot_all
        
        last_30_percent_sum = np.sum(np.abs(dynamic_acc[-trim_size:]))
        return last_30_percent_sum
    
    # ここからループを削除した修正
    acc_vector = train[['acc_x', 'acc_y', 'acc_z']].values
    
    if len(acc_vector) == 0 or len(acc_vector) < 2:
        return train.copy()
            
    optimized_phi = minimize_scalar(
        objective_function,
        bounds=(0, 2 * np.pi),
        args=(acc_vector, rotated_gravity)
    ).x
    
    cos_phi, sin_phi = np.cos(optimized_phi), np.sin(optimized_phi)
    R_z_opt = np.array([[cos_phi, -sin_phi, 0],
                        [sin_phi,  cos_phi, 0],
                        [      0,        0, 1]])
    
    acc_rotated = np.einsum('ij, kj->ki', R_z_opt, acc_vector)
    dynamic_acceleration = acc_rotated - rotated_gravity
    
    result_df = train.copy()
    result_df['acc_x_dynamic'] = dynamic_acceleration[:, 0]
    result_df['acc_y_dynamic'] = dynamic_acceleration[:, 1]
    result_df['acc_z_dynamic'] = dynamic_acceleration[:, 2]

    result_df['gravity_x'] = rotated_gravity[:, 0]
    result_df['gravity_y'] = rotated_gravity[:, 1]
    result_df['gravity_z'] = rotated_gravity[:, 2]

    return result_df


count = 0
for demo, data, label in train_dataset:
    data = calculate_dynamic_acceleration2(data)
    if np.random.rand() < 0.5:
        plot_sequence(data)
        count += 1
        if count >= 4:
            break

