import os, numpy as np, pandas as pd
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")

import torch
import torch.nn.functional as F



# Configuration

PROCESS_DATASET = __name__ == '__main__'

if PROCESS_DATASET:

    #RAW_DIR = Path("data")
    RAW_DIR = Path("/kaggle/input/bfrb-dataset")
    #EXPORT_DIR = RAW_DIR
    EXPORT_DIR = Path("/kaggle/working")
    FILENAME = 'train.csv'

    assert RAW_DIR.exists(), f"Directory not found: {RAW_DIR}"
    if not os.path.exists(EXPORT_DIR):
        os.mkdir(EXPORT_DIR)


# Remove Gravity

from scipy.spatial.transform import Rotation as R

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




#################################################################################################################
####################################              FE               ##############################################
#################################################################################################################

def feature_engineering(df: pd.DataFrame):
    """
    creates new features unsing imu cols in input dataframe,
    other cols will be ignored

    Args:
        df (pd.DataFrame): df just like train.csv. *Note:* this \
        function will change it.

    Returns:
        df (pd.DataFrame): input dataframe with extra features in \
        propriate columns order: `meta_cols + imu_cols + new_cols \
        + thm_cols + tof_cols`
    """

    #######################
    ### Feature list ######
    #######################
    meta_labels = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                   'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    meta_cols = [c for c in df.columns if c in meta_labels]
    
    feature_cols = [c for c in df.columns if c not in meta_cols]

    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    thm_cols = [c for c in feature_cols if c.startswith('thm_')]
    tof_cols = [c for c in feature_cols if c.startswith('tof_')]
    #print(f"  IMU {len(imu_cols)} |  THM {len(thm_cols)}|  TOF {len(tof_cols)} | total {len(feature_cols)} features")



    #######################
    #### ace & gyro #######
    #######################
    # IMU magnitude
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_mag'] = np.sqrt(df['rot_x']**2 + df['rot_y']**2 + df['rot_z']**2)



    #######################
    #### linear_ace #######
    #######################
    # Remove gravity
    def get_linear_accel(df):
        res = remove_gravity_from_acc(
            df[['acc_x', 'acc_y', 'acc_z']],
            df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        )
        res = pd.DataFrame(res, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=df.index)
        return res
    
    linear_accel_df = df.groupby('sequence_id').apply(get_linear_accel, include_groups=False)
    linear_accel_df = linear_accel_df.droplevel('sequence_id')
    df = df.join(linear_accel_df)
    
    df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)



    #######################
    #### angular_vel ######
    #######################
    # Calc angular velocity
    def calc_angular_velocity(df):
        res = calculate_angular_velocity_from_quat( df[['rot_x', 'rot_y', 'rot_z', 'rot_w']] )
        res = pd.DataFrame(res, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=df.index)
        return res
    
    angular_velocity_df = df.groupby('sequence_id').apply(calc_angular_velocity, include_groups=False)
    angular_velocity_df = angular_velocity_df.droplevel('sequence_id')
    df = df.join(angular_velocity_df)

    df['angular_vel_mag'] = np.sqrt(df['angular_vel_x']**2 + df['angular_vel_y']**2 + df['angular_vel_z']**2)
    


    #########################
    ### apply jerk & pow ####
    #########################
    grouped_df = df.groupby('sequence_id')
    for fe in ('acc', 'rot', 'linear_acc', 'angular_vel'):
        for dir in ('x', 'y', 'z', 'mag'):
            # IMU jerks [acc/gyro/linear_acc]_jerk_[x/y/z/mag]
            df[fe + '_jerk_' + dir] = grouped_df[fe + '_' + dir].diff().fillna(0)

            # IMU Energy (=pow) [acc/gyro/linear_acc]_pow_[x/y/z/mag]
            df[fe + '_pow_' + dir] = df[fe + '_' + dir] ** 2



    ###################################
    ### angle & ang_vel & ang_dist ####
    ###################################
    # IMU angle
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))

    # IMU angular velocity
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    
    # Calculating angular distance
    def calc_angular_distance(df):
        res = calculate_angular_distance(df[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
        res = pd.DataFrame(res, columns=['angular_distance'], index=df.index)
        return res
    
    angular_distance_df = df.groupby('sequence_id').apply(calc_angular_distance, include_groups=False)
    angular_distance_df = angular_distance_df.droplevel('sequence_id')
    df = df.join(angular_distance_df)



    #########################
    ####### LPF & HPF #######
    #########################
    def create_gaussian_kernel(size: int, channels: int):
        """Create gaussian kernel for smoothing"""
        kernel = torch.tensor([np.exp(-(i - size//2)**2/2) for i in range(size)], dtype=torch.float32)
        kernel = kernel / kernel.sum()
        kernel = kernel.repeat(channels, 1, 1)  # (out_channels, in_channels/groups, kernel_size)
        return kernel
    
    k = 15
    grouped = df.groupby('sequence_id')
    
    for fe in ('acc', 'rot', 'linear_acc', 'angular_vel'):
        for dir in ('x', 'y', 'z', 'mag'):
            col_name = f'{fe}_{dir}'
            weight = create_gaussian_kernel(k, 1)  # 1 channel, cause process 1 column per iteration
            
            # apply convolution to each group
            lpf_results = []
            for _, group in grouped:
                # convert to tensor and add dimentions (batch, channel, length)
                data = torch.tensor(group[col_name].values, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                # apply convolution
                lpf = F.conv1d(data, weight, padding=k//2)
                lpf_results.append(lpf.squeeze().numpy())
            
            # concatenate results
            lpf_series = pd.concat([pd.Series(x, index=group.index) for x, (_, group) in zip(lpf_results, grouped)])
            df[f'{fe}_lpf_{dir}'] = lpf_series
            df[f'{fe}_hpf_{dir}'] = df[col_name] - df[f'{fe}_lpf_{dir}']




    new_cols = [ c for c in df.columns if (c not in meta_cols) and (c not in feature_cols) ]
    df[imu_cols + new_cols] = df[imu_cols + new_cols].ffill().bfill().fillna(0).values.astype('float32')
    
    return df[meta_cols + imu_cols + new_cols + thm_cols + tof_cols]


if PROCESS_DATASET:
    
    print("▶ loading dataset …")
    df = pd.read_csv(RAW_DIR / FILENAME)


if PROCESS_DATASET:
    df = feature_engineering(df)


if PROCESS_DATASET:
    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                    'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    thm_cols = [c for c in feature_cols if c.startswith('thm_')]
    tof_cols = [c for c in feature_cols if c.startswith('tof_')]
    print(f"  IMU {len(imu_cols)} | THM {len(thm_cols)} | TOF {len(tof_cols)} | total {len(feature_cols)} features")



if PROCESS_DATASET:
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Матрица корреляций
    corr_matrix = df.loc[:20000, imu_cols].corr()
    sns.heatmap(corr_matrix, annot=False)
    plt.show()




if PROCESS_DATASET:
    df.to_csv(EXPORT_DIR / f'featured_{FILENAME}', index=False)

