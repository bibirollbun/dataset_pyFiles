# Suppress pip install output
!pip install --no-index --find-links=/kaggle/input/my-offline-wheels/kaggle_wheels_v2 scikit-learn==1.6.1 > /dev/null 2>&1
!pip install --no-index --find-links=/kaggle/input/my-offline-wheels/kaggle_wheels_v2 imbalanced-learn==0.13.0 > /dev/null 2>&1

# Verify imports
from imblearn.over_sampling import SMOTE
print("SMOTE imported successfully")

import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.utils import to_categorical, pad_sequences, Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv1D, BatchNormalization, Activation, MaxPooling1D, Dropout, Bidirectional, LSTM, Dense, Concatenate, GaussianNoise, Layer
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import CosineDecayRestarts
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import polars as pl
from scipy.stats import skew
from scipy.spatial.transform import Rotation as R
print("Imports successful")


# Fix seed
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed_everything(42)


# Configuration
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
EXPORT_DIR = Path("/kaggle/working")
BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 5e-4
EPOCHS = 99
PATIENCE = 30
dt = 0.001
MIXUP_ALPHA = 0.2


# Remove gravity from acceleration
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


# Calculate angular velocity from quaternions
def calculate_angular_velocity_from_quat(rot_data, time_delta=0.001): # Match dt=0.001 (1000 Hz)
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


# Calculate angular distance between quaternions
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
    return angular_dist


# MixUp Generator
class MixupGenerator(Sequence):
    def __init__(self, X, y, batch_size, alpha=0.2):
        self.X, self.y = X, y
        self.batch = batch_size
        self.alpha = alpha
        self.indices = np.arange(len(X))
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))
    def __getitem__(self, i):
        idx = self.indices[i*self.batch:(i+1)*self.batch]
        Xb, yb = self.X[idx], self.y[idx]
        lam = np.random.beta(self.alpha, self.alpha)
        perm = np.random.permutation(len(Xb))
        X_mix = lam * Xb + (1-lam) * Xb[perm]
        y_mix = lam * yb + (1-lam) * yb[perm]
        return X_mix, y_mix
    def on_epoch_end(self):
        np.random.shuffle(self.indices)


# Simple Attention Layer
class SimpleAttention(Layer):
    def __init__(self, **kwargs):
        super(SimpleAttention, self).__init__(**kwargs)
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight',
                                 shape=(input_shape[-1], 1),
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(SimpleAttention, self).build(input_shape)
    def call(self, inputs):
        e = tf.matmul(inputs, self.W)  # (None, time_steps, 1)
        alpha = tf.nn.softmax(e, axis=1)  # (None, time_steps, 1)
        context = inputs * alpha  # (None, time_steps, features)
        context = tf.reduce_sum(context, axis=1)  # (None, features)
        return context


# Two-branch model with Gaussian noise
def build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes):
    inp = Input(shape=(pad_len, imu_dim + tof_dim))
    x = GaussianNoise(stddev=0.05)(inp)  # Add Gaussian noise to input
    imu = x[:, :, :imu_dim]
    tof = x[:, :, imu_dim:]
    # IMU branch
    x1 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(0.01))(imu)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Dropout(0.2)(x1)
    x1 = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(0.01))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Activation('relu')(x1)
    x1 = MaxPooling1D(2)(x1)
    x1 = Dropout(0.4)(x1)
    x1 = Bidirectional(LSTM(64, return_sequences=True, kernel_regularizer=l2(0.01)))(x1)
    x1 = SimpleAttention()(x1)
    # TOF/Thermal branch
    x2 = Conv1D(32, 3, padding='same', use_bias=False, kernel_regularizer=l2(0.01))(tof)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Dropout(0.2)(x2)
    x2 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(0.01))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2)
    x2 = Dropout(0.4)(x2)
    x2 = Bidirectional(LSTM(32, return_sequences=True, kernel_regularizer=l2(0.01)))(x2)
    x2 = SimpleAttention()(x2)
    # Merge
    x = Concatenate()([x1, x2])
    x = Dense(128, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.4)(x)
    out = Dense(n_classes, activation='softmax')(x)
    return Model(inp, out)


# Random noise injection
def augment_imu(df, noise_std=0.05):
    print("Applying random noise injection to IMU features")
    df = df.copy()
    imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']
    noise = np.random.normal(0, noise_std, df[imu_cols].shape)
    df[imu_cols] += noise
    # Normalize quaternions after noise
    rot_cols = ['rot_x', 'rot_y', 'rot_z', 'rot_w']
    rot_norm = np.sqrt((df[rot_cols]**2).sum(axis=1)).clip(lower=1e-6)
    df[rot_cols] = df[rot_cols].div(rot_norm, axis=0).fillna(0)
    return df


# SpecAugment for IMU features
def spec_augment(X, freq_mask=2, time_mask=10, n_freq=1, n_time=1):
    print("Applying SpecAugment to IMU features")
    X_aug = X.copy()
    n_samples, seq_len, n_features = X.shape
    # Frequency mask
    for i in range(n_samples):
        for _ in range(n_freq):
            f = np.random.randint(0, n_features - freq_mask + 1)
            X_aug[i, :, f:f+freq_mask] = 0
        # Time mask
        for _ in range(n_time):
            t = np.random.randint(0, seq_len - time_mask + 1)
            X_aug[i, t:t+time_mask, :] = 0
    return X_aug


# Training pipeline
print("Loading dataset")
df = pd.read_csv(RAW_DIR / "train.csv")


# Apply noise injection
df = augment_imu(df, noise_std=0.05)


le = LabelEncoder()
df['gesture_int'] = le.fit_transform(df['gesture'])
#np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)
gesture_classes = le.classes_


# Feature engineering
print("Calculating engineered IMU features")
df['acc_energy'] = df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2
df['rot_mag'] = np.sqrt(df['rot_x']**2 + df['rot_y']**2 + df['rot_z']**2)
df['acc_mag'] = np.sqrt(df['acc_energy'])
df['acc_skew'] = df.groupby('sequence_id')['acc_mag'].transform(lambda x: skew(x.fillna(0)))
df['rot_smooth'] = df.groupby('sequence_id')['rot_mag'].transform(lambda x: x.rolling(3, min_periods=1).mean().fillna(x.mean()))

# New feature: Jerk
df[['jerk_x', 'jerk_y', 'jerk_z']] = df.groupby('sequence_id')[['acc_x', 'acc_y', 'acc_z']].transform(lambda x: x.diff().fillna(0) / dt)

# New features: Linear acceleration
print("Calculating linear acceleration features")
linear_accel_list = []
for _, group in df.groupby('sequence_id'):
    acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
    rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
    linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))
df_linear_accel = pd.concat(linear_accel_list)
df = pd.concat([df, df_linear_accel], axis=1)
df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)
df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].transform(lambda x: x.diff().fillna(0) / dt)

# New features: Angular velocity, jerk, and snap
print("Calculating angular velocity, jerk, and snap from quaternion derivatives")
angular_vel_list = []
angular_jerk_list = []
angular_snap_list = []
for _, group in df.groupby('sequence_id'):
    rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    angular_vel_group = calculate_angular_velocity_from_quat(rot_data_group, time_delta=dt)
    angular_jerk_group = pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z']).diff().fillna(0).values / dt
    angular_snap_group = pd.DataFrame(angular_jerk_group, columns=['angular_jerk_x', 'angular_jerk_y', 'angular_jerk_z']).diff().fillna(0).values / dt
    angular_vel_list.append(pd.DataFrame(angular_vel_group, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=group.index))
    angular_jerk_list.append(pd.DataFrame(angular_jerk_group, columns=['angular_jerk_x', 'angular_jerk_y', 'angular_jerk_z'], index=group.index))
    angular_snap_list.append(pd.DataFrame(angular_snap_group, columns=['angular_snap_x', 'angular_snap_y', 'angular_snap_z'], index=group.index))
df_angular_vel = pd.concat(angular_vel_list)
df_angular_jerk = pd.concat(angular_jerk_list)
df_angular_snap = pd.concat(angular_snap_list)
df = pd.concat([df, df_angular_vel, df_angular_jerk, df_angular_snap], axis=1)

# New features: Angular distance
print("Calculating angular distance between successive quaternions")
angular_distance_list = []
for _, group in df.groupby('sequence_id'):
    rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
    angular_dist_group = calculate_angular_distance(rot_data_group)
    angular_distance_list.append(pd.DataFrame(angular_dist_group, columns=['angular_distance'], index=group.index))
df_angular_distance = pd.concat(angular_distance_list)
df = pd.concat([df, df_angular_distance], axis=1)

# TOF features
print("Calculating TOF features")
for i in range(1, 6):
    pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
    tof_sensor_data = df[pixel_cols_tof].replace(-1, np.nan)
    df[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1)
    df[f'tof_{i}_std'] = tof_sensor_data.std(axis=1)
    df[f'tof_{i}_min'] = tof_sensor_data.min(axis=1)
    df[f'tof_{i}_max'] = tof_sensor_data.max(axis=1)

# Feature columns
meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation', 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z', 'acc_energy', 'rot_mag', 'acc_skew', 'rot_smooth', 
            'jerk_x', 'jerk_y', 'jerk_z', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag', 
            'linear_acc_mag_jerk', 'angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_distance',
            'angular_jerk_x', 'angular_jerk_y', 'angular_jerk_z', 'angular_snap_x', 'angular_snap_y', 'angular_snap_z']
thm_cols = [f'thm_{i}' for i in range(1, 6)]
tof_cols = [f'tof_{i}_{stat}' for i in range(1, 6) for stat in ['mean', 'std', 'min', 'max']]
final_feature_cols = imu_cols + thm_cols + tof_cols
sensor_cols = thm_cols + [f'tof_{i}_v{p}' for i in range(1, 6) for p in range(64)]
imu_dim_final = len(imu_cols)  # 29
tof_thm_dim_final = len(thm_cols) + len(tof_cols)  # 5 + 20 = 25
print(f"IMU features: {imu_dim_final} | THM + TOF: {tof_thm_dim_final} | Total: {len(final_feature_cols)} features")
#np.save(EXPORT_DIR / "feature_cols.npy", np.array(final_feature_cols))


print("Building sequences with aggregated TOF")
seq_gp = df.groupby('sequence_id')
X_list_unscaled, y_list_int, lens = [], [], []

for seq_id, seq_df_orig in seq_gp:
    seq_df = seq_df_orig.copy()
    for i in range(1, 6):
        pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
        tof_data = seq_df[pixel_cols_tof].replace(-1, np.nan)
        seq_df[f'tof_{i}_mean'] = tof_data.mean(axis=1)
        seq_df[f'tof_{i}_std'] = tof_data.std(axis=1)
        seq_df[f'tof_{i}_min'] = tof_data.min(axis=1)
        seq_df[f'tof_{i}_max'] = tof_data.max(axis=1)
    mat_unscaled = seq_df[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')
    X_list_unscaled.append(mat_unscaled)
    y_list_int.append(seq_df['gesture_int'].iloc[0])
    lens.append(len(mat_unscaled))


print("Fitting StandardScaler")
all_steps = np.concatenate(X_list_unscaled, axis=0)
scaler = StandardScaler().fit(all_steps)
#joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")
del all_steps


print("Scaling and padding sequences")
X_scaled_list = [scaler.transform(x_seq) for x_seq in X_list_unscaled]
del X_list_unscaled
pad_len = int(np.percentile(lens, PAD_PERCENTILE))
#np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
X = pad_sequences(X_scaled_list, maxlen=pad_len, padding='post', truncating='post', dtype='float32')
del X_scaled_list
y_int = np.array(y_list_int)


print("Applying SpecAugment")
X = spec_augment(X, freq_mask=2, time_mask=10, n_freq=1, n_time=1)


print("Applying SMOTE")
n_samples, seq_len, n_features = X.shape
X_flat = X.reshape(n_samples, seq_len * n_features)
smote = SMOTE(random_state=42)
X_flat_smote, y_int_smote = smote.fit_resample(X_flat, y_int)
X_smote = X_flat_smote.reshape(-1, seq_len, n_features)
y_smote = to_categorical(y_int_smote, num_classes=len(gesture_classes))


print("Splitting data")
X_tr, X_val, y_tr, y_val = train_test_split(X_smote, y_smote, test_size=0.1, random_state=42, stratify=y_int_smote)
del X_smote, y_smote

cw_vals = compute_class_weight('balanced', classes=np.arange(len(gesture_classes)), y=y_int_smote)
class_weight = dict(enumerate(cw_vals))


print("Building and training model")
model = build_two_branch_model(pad_len, imu_dim_final, tof_thm_dim_final, len(gesture_classes))
steps = len(X_tr) // BATCH_SIZE

train_gen = MixupGenerator(X_tr, y_tr, batch_size=BATCH_SIZE, alpha=MIXUP_ALPHA)
#cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1, monitor='val_accuracy')
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=5, min_lr=1e-6)

model.compile(optimizer=Adam(learning_rate=LR_INIT),
              loss=CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

model.fit(train_gen, 
          steps_per_epoch=len(train_gen),
          epochs=EPOCHS,
          validation_data=(X_val, y_val),
          class_weight=class_weight,
          callbacks=[reduce_lr],
          verbose=1)
#model.save(EXPORT_DIR / "gesture_two_branch.h5")
print("Training done – model saved")


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_seq = sequence.to_pandas()
    
    # Feature engineering
    df_seq['acc_energy'] = df_seq['acc_x']**2 + df_seq['acc_y']**2 + df_seq['acc_z']**2
    df_seq['rot_mag'] = np.sqrt(df_seq['rot_x']**2 + df_seq['rot_y']**2 + df_seq['rot_z']**2)
    df_seq['acc_mag'] = np.sqrt(df_seq['acc_energy'])
    df_seq['acc_skew'] = skew(df_seq['acc_mag'].fillna(0))
    df_seq['rot_smooth'] = df_seq['rot_mag'].rolling(3, min_periods=1).mean().fillna(df_seq['rot_mag'].mean())
    df_seq[['jerk_x', 'jerk_y', 'jerk_z']] = df_seq[['acc_x', 'acc_y', 'acc_z']].diff().fillna(0) / dt
    
    # Linear acceleration
    linear_accel = remove_gravity_from_acc(df_seq[['acc_x', 'acc_y', 'acc_z']], df_seq[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
    df_seq[['linear_acc_x', 'linear_acc_y', 'linear_acc_z']] = linear_accel
    df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 + df_seq['linear_acc_y']**2 + df_seq['linear_acc_z']**2)
    df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)
    
    # Angular velocity, jerk, and snap
    angular_vel = calculate_angular_velocity_from_quat(df_seq[['rot_x', 'rot_y', 'rot_z', 'rot_w']], time_delta=dt)
    df_seq[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']] = angular_vel
    df_seq[['angular_jerk_x', 'angular_jerk_y', 'angular_jerk_z']] = pd.DataFrame(angular_vel, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z']).diff().fillna(0) / dt
    df_seq[['angular_snap_x', 'angular_snap_y', 'angular_snap_z']] = df_seq[['angular_jerk_x', 'angular_jerk_y', 'angular_jerk_z']].diff().fillna(0) / dt
    
    # Angular distance
    df_seq['angular_distance'] = 2 * np.arccos(df_seq['rot_w'].clip(-1, 1))
    
    # TOF features
    for i in range(1, 6):
        pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
        tof_data = df_seq[pixel_cols_tof].replace(-1, np.nan)
        df_seq[f'tof_{i}_mean'] = tof_data.mean(axis=1)
        df_seq[f'tof_{i}_std'] = tof_data.std(axis=1)
        df_seq[f'tof_{i}_min'] = tof_data.min(axis=1)
        df_seq[f'tof_{i}_max'] = tof_data.max(axis=1)
    
    df_seq_reordered = pd.DataFrame(columns=final_feature_cols)
    for col in final_feature_cols:
        if col in df_seq.columns:
            df_seq_reordered[col] = df_seq[col]
        else:
            df_seq_reordered[col] = 0
    mat_unscaled = df_seq_reordered.ffill().bfill().fillna(0).values.astype('float32')
    mat_scaled = scaler.transform(mat_unscaled)
    pad_input = pad_sequences([mat_scaled], maxlen=pad_len, padding='post', truncating='post', dtype='float32')
    
    idx = int(np.argmax(model.predict(pad_input, verbose=0)[0]))
    gesture = str(gesture_classes[idx])
    print(f"Predicted gesture: {gesture}")
    return gesture


import kaggle_evaluation.cmi_inference_server

# Submit
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

