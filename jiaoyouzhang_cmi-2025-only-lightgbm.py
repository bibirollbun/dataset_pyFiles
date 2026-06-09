import numpy as np
import polars as pl
import pandas as pd
import joblib
from scipy.spatial.transform import Rotation as R

models=[]
for fold in range(3):
    model = joblib.load(f'/kaggle/input/cmi-2025-only-lightgbm-training/model_lgb{fold}.joblib')
    models.append(model)
le = joblib.load('/kaggle/input/cmi-2025-only-lightgbm-training/le.joblib')


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


def feature_engineering_imu(data:pl.DataFrame):
    data = data.to_pandas()
    data['acc_mag'] = np.sqrt(data['acc_x']**2 + data['acc_y']**2 + data['acc_z']**2)
    data['rot_angle'] = 2 * np.arccos(data['rot_w'].clip(-1, 1))
    data['acc_mag_jerk'] = data.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    data['rot_angle_vel'] = data.groupby('sequence_id')['rot_angle'].diff().fillna(0)

    linear_accel_list = []
    for _, group in data.groupby('sequence_id'):
        acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
        linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))
    
    df_linear_accel = pd.concat(linear_accel_list)
    data = pd.concat([data, df_linear_accel], axis=1)
    data['linear_acc_mag'] = np.sqrt(data['linear_acc_x']**2 + data['linear_acc_y']**2 + data['linear_acc_z']**2)
    data['linear_acc_mag_jerk'] = data.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)
    angular_vel_list = []
    for _, group in data.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group)
        angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))
    
    df_angular_vel = pd.concat(angular_vel_list)
    data = pd.concat([data, df_angular_vel], axis=1)
    
    print("  Calculating angular distance between successive quaternions...")
    angular_distance_list = []
    for _, group in data.groupby('sequence_id'):
        rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        angular_dist_group = calculate_angular_distance(rot_data_group)
        angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))
    
    df_angular_distance = pd.concat(angular_distance_list)
    data = pd.concat([data, df_angular_distance], axis=1)
    data = pl.from_pandas(data)
    return data


import kaggle_evaluation.cmi_inference_server
import os

def feature_engineering_inf(data:pl.DataFrame):

    
    # All numeric sensor columns (everything except id, demo, target)
    stat_cols = [
        c for c in data.columns
        if c not in ["sequence_id", "row_id","sequence_counter","subject"]
    ]
    
    # Build aggregation expressions
    agg_exprs = []
    
    # full-stats bundle for sensor columns
    for c in stat_cols:
        agg_exprs += [
            pl.col(c).mean().alias(f"{c}_mean"),
            pl.col(c).std().alias(f"{c}_std"),
            pl.col(c).var().alias(f"{c}_var"),
            # pl.col(c).mode().list.first().alias(f"{c}_mode"),
            pl.col(c).quantile(0.25).alias(f"{c}_q25"),
            pl.col(c).median().alias(f"{c}_q50"),
            pl.col(c).quantile(0.75).alias(f"{c}_q75"),
            pl.col(c).max().alias(f"{c}_max"),
            pl.col(c).min().alias(f"{c}_min"),
        ]
    
    
    # Group-by and aggregate
    cleaned_data = (
        data
        .group_by("sequence_id", maintain_order=True)
        .agg(agg_exprs)
    )
    return cleaned_data


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    data = sequence
    data = feature_engineering_imu(data)
    data = feature_engineering_inf(data)
    
    pdf = data.to_pandas().drop(columns=["sequence_id"])
    # Predict using ensemble of CV models
    predictions = []
    for model in models:
        pred = model.predict(pdf)
        # Ensure we get a scalar value
        if isinstance(pred, np.ndarray):
            pred = pred[0]
        predictions.append(int(pred))
    
    # Use majority vote or most confident prediction
    predicted_label_id = max(set(predictions), key=predictions.count)
    
    # Convert back to gesture string
    predicted_gesture_str = le.inverse_transform([predicted_label_id])[0]
    
    return predicted_gesture_str


import os
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

