import os, json, joblib, numpy as np, pandas as pd
import random
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight

from tensorflow.keras.utils import Sequence, to_categorical, pad_sequences
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, LSTM, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate, GRU, GaussianNoise
)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K
import tensorflow as tf
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold
from scipy.spatial.transform import Rotation as R

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_everything(seed=42)
# (Competition metric will only be imported when TRAINing)
TRAIN = False                     # â†� set to True when you want to train
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/cmi-d-111")
EXPORT_DIR = Path("./")                                    # artefacts will be saved here
BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 5e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 160
PATIENCE = 40

print("â–¶ imports ready Â· tensorflow", tf.__version__)

#Tensor Manipulations
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

# Residual CNN Block with SE
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

# Normalizes and cleans the time series sequence. 

def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

# MixUp the data argumentation in order to regularize the neural network. 

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
            angular_dist[i] = 0 # Ğ˜Ğ»Ğ¸ np.nan, Ğ² Ğ·Ğ°Ğ²Ğ¸Ñ�Ğ¸Ğ¼Ğ¾Ñ�Ñ‚Ğ¸ Ğ¾Ñ‚ Ğ¶ĞµĞ»Ğ°ĞµĞ¼Ğ¾Ğ³Ğ¾ Ğ¿Ğ¾Ğ²ĞµĞ´ĞµĞ½Ğ¸Ñ�
            continue
        try:
            # ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ ĞºĞ²Ğ°Ñ‚ĞµÑ€Ğ½Ğ¸Ğ¾Ğ½Ğ¾Ğ² Ğ² Ğ¾Ğ±ÑŠĞµĞºÑ‚Ñ‹ Rotation
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            # Ğ’Ñ‹Ñ‡Ğ¸Ñ�Ğ»ĞµĞ½Ğ¸Ğµ ÑƒĞ³Ğ»Ğ¾Ğ²Ğ¾Ğ³Ğ¾ Ñ€Ğ°Ñ�Ñ�Ñ‚Ğ¾Ñ�Ğ½Ğ¸Ñ�: 2 * arccos(|real(p * q*)|)
            # Ğ³Ğ´Ğµ p* - Ñ�Ğ¾Ğ¿Ñ€Ñ�Ğ¶ĞµĞ½Ğ½Ñ‹Ğ¹ ĞºĞ²Ğ°Ñ‚ĞµÑ€Ğ½Ğ¸Ğ¾Ğ½ q
            # Ğ’ scipy.spatial.transform.Rotation, r1.inv() * r2 Ğ´Ğ°ĞµÑ‚ Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾Ğµ Ğ²Ñ€Ğ°Ñ‰ĞµĞ½Ğ¸Ğµ.
            # Ğ£Ğ³Ğ¾Ğ» Ñ�Ñ‚Ğ¾Ğ³Ğ¾ Ğ¾Ñ‚Ğ½Ğ¾Ñ�Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ğ¾Ğ³Ğ¾ Ğ²Ñ€Ğ°Ñ‰ĞµĞ½Ğ¸Ñ� - Ñ�Ñ‚Ğ¾ Ğ¸ ĞµÑ�Ñ‚ÑŒ ÑƒĞ³Ğ»Ğ¾Ğ²Ğ¾Ğµ Ñ€Ğ°Ñ�Ñ�Ñ‚Ğ¾Ñ�Ğ½Ğ¸Ğµ.
            relative_rotation = r1.inv() * r2
            
            # Ğ£Ğ³Ğ¾Ğ» rotation vector Ñ�Ğ¾Ğ¾Ñ‚Ğ²ĞµÑ‚Ñ�Ñ‚Ğ²ÑƒĞµÑ‚ ÑƒĞ³Ğ»Ğ¾Ğ²Ğ¾Ğ¼Ñƒ Ñ€Ğ°Ñ�Ñ�Ñ‚Ğ¾Ñ�Ğ½Ğ¸Ñ�
            # Ğ�Ğ¾Ñ€Ğ¼Ğ° rotation vector - Ñ�Ñ‚Ğ¾ ÑƒĞ³Ğ¾Ğ» Ğ² Ñ€Ğ°Ğ´Ğ¸Ğ°Ğ½Ğ°Ñ…
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0 # Ğ’ Ñ�Ğ»ÑƒÑ‡Ğ°Ğµ Ğ½ĞµĞ´ĞµĞ¹Ñ�Ñ‚Ğ²Ğ¸Ñ‚ĞµĞ»ÑŒĞ½Ñ‹Ñ… ĞºĞ²Ğ°Ñ‚ĞµÑ€Ğ½Ğ¸Ğ¾Ğ½Ğ¾Ğ²
            pass
            
    return angular_dist

def build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    inp = Input(shape=(pad_len, imu_dim+tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    # IMU deep branch
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)

    # TOF/Thermal lighter branch
    x2 = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)
    x2 = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x2)
    x2 = BatchNormalization()(x2); x2 = Activation('relu')(x2)
    x2 = MaxPooling1D(2)(x2); x2 = Dropout(0.2)(x2)

    merged = Concatenate()([x1, x2])

    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xc = GaussianNoise(0.09)(merged)
    xc = Dense(16, activation='elu')(xc)
    
    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    x = attention_layer(x)

    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x); x = Activation('relu')(x)
        x = Dropout(drop)(x)

    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    return Model(inp, out)

tmp_model = build_two_branch_model(127,7,325,18)
print("â–¶ INFERENCE MODE â€“ loading artefacts from", PRETRAINED_DIR)
final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len        = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
scaler         = joblib.load(PRETRAINED_DIR / "scaler.pkl")
gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)


custom_objs = {
    'time_sum': time_sum, 'squeeze_last_axis': squeeze_last_axis, 'expand_last_axis': expand_last_axis,
    'se_block': se_block, 'residual_se_cnn_block': residual_se_cnn_block, 'attention_layer': attention_layer,
}

# ----------------------------------------------------------------- #
# Load any Models
# * is 2 Train Model Load
# ----------------------------------------------------------------- #

models1 = []
print(f"  Loading models for ensemble inference...")
for fold in range(10):
    MODEL_DIR = "/kaggle/input/cmi-d-111"
    
    model_path = f"{MODEL_DIR}/D-111_{fold}.h5"
    print(">>>LoadModel>>>",model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models1.append(model)
print("-"*50)

for fold in range(10):
    MODEL_DIR = "/kaggle/input/cmi-d-111"
    
    model_path = f"{MODEL_DIR}/v0629_{fold}.h5"
    print(">>>LoadModel>>>",model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models1.append(model)
print("-"*50)
print(f"[INFO]NumUseModels:{len(models1)}")


# predict_1

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
    
    all_preds = [model.predict(pad_input, verbose=0)[0] for model in models1] # ä¸»å‡ºåŠ›ã�®ã�¿å�–å¾—
    avg_pred = np.mean(all_preds, axis=0)
    return avg_pred
    # return str(gesture_classes[avg_pred.argmax()])


import os
import torch
import kagglehub
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.notebook import tqdm
from torch.amp import autocast
import pandas as pd
import polars as pl
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial.transform import Rotation as R
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from collections import defaultdict
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold

# %% [markdown]
# # Dataset
# Compared to last notebook I release, change skf to sgkf and set percent=99.

# %% [code] {"jupyter":{"source_hidden":true},"_kg_hide-input":true,"execution":{"iopub.status.busy":"2025-08-11T18:05:34.488471Z","iopub.execute_input":"2025-08-11T18:05:34.488909Z","iopub.status.idle":"2025-08-11T18:05:34.499979Z","shell.execute_reply.started":"2025-08-11T18:05:34.488886Z","shell.execute_reply":"2025-08-11T18:05:34.499185Z"}}
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

# %% [code] {"jupyter":{"source_hidden":true},"_kg_hide-input":true,"execution":{"iopub.status.busy":"2025-08-11T18:05:34.500904Z","iopub.execute_input":"2025-08-11T18:05:34.501162Z","iopub.status.idle":"2025-08-11T18:05:34.686241Z","shell.execute_reply.started":"2025-08-11T18:05:34.501138Z","shell.execute_reply":"2025-08-11T18:05:34.685443Z"}}
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
        imus = [imu[:, :, self.global_imu_indices[k]] for k in self.global_imu_indices]
        thms = [thm[:, :, self.global_thm_indices[k]] for k in range(1, 6)]
        tofs = [tof[:, :, self.global_tof_indices[k]] for k in range(1, 6)]
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

# %% [code] {"jupyter":{"source_hidden":true},"_kg_hide-input":true,"execution":{"iopub.status.busy":"2025-08-11T18:05:34.687882Z","iopub.execute_input":"2025-08-11T18:05:34.688197Z","iopub.status.idle":"2025-08-11T18:05:34.701186Z","shell.execute_reply.started":"2025-08-11T18:05:34.688172Z","shell.execute_reply":"2025-08-11T18:05:34.700543Z"}}
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
        
        print("\näº¤å�‰éªŒè¯�æŠ˜å� ç»Ÿè®¡:")
        for fold_idx in range(self.n_folds):
            train_fold, valid_fold = self.get_fold_datasets(fold_idx)
            train_counts = get_label_counts(train_fold)
            valid_counts = get_label_counts(valid_fold)
            print(f"\nFold {fold_idx + 1}:")
            print(f"{'ç±»åˆ«':<50} {'è®­ç»ƒé›†':<10} {'éªŒè¯�é›†':<10}")
            for name in self.class_names:
                print(f"{name:<50} {train_counts[name]:<10} {valid_counts[name]:<10}")

        for fold_idx, (train_idx, val_idx) in enumerate(self.folds):
            train_subjects = set(self.fold_groups[train_idx])
            val_subjects = set(self.fold_groups[val_idx])
            print(f"\nFold {fold_idx + 1}:")
            print("è®­ç»ƒé›†å�—è¯•è€…:", train_subjects)
            print("éªŒè¯�é›†å�—è¯•è€…:", val_subjects)

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
        
        print(f"\næ�’é™¤subject {self.exclude_subjects} å��çš„æ•°æ�®é‡�å�˜åŒ–:")
        print(f"å�Ÿå§‹è®­ç»ƒé›†æ ·æœ¬: {original_counts['train']}")
        print(f"è¿‡æ»¤å��è®­ç»ƒé›†æ ·æœ¬: {filtered_counts['train']}")
        print(f"å�Ÿå§‹éªŒè¯�é›†æ ·æœ¬: {original_counts['valid']}") 
        print(f"è¿‡æ»¤å��éªŒè¯�é›†æ ·æœ¬: {filtered_counts['valid']}")

# %% [markdown]
# # Model
# Main changes here. The idea is: the dataset has many nan values, but count of nans is not the same in different sensors.
# 
# For example, in thm data, the count of nans for thm1-thm5:
# 
# (6987, 7638, 6472, 6224, 33286)
# 
# So I consider it's not good to just combine all thm data and return thm_feat. Instead, make every thm sensor has its own feature layer and combine it at last.
# 
# Imu and tof data have same condition in nan counts. For imu, acc has no nan but rot does have. For tof, almost same as thm but a little different. Also mention that thm5 and tof5 have obviously more nans than other sensors.

# %% [code] {"jupyter":{"source_hidden":true},"_kg_hide-input":true,"execution":{"iopub.status.busy":"2025-08-11T18:05:34.701835Z","iopub.execute_input":"2025-08-11T18:05:34.702028Z","iopub.status.idle":"2025-08-11T18:05:34.717038Z","shell.execute_reply.started":"2025-08-11T18:05:34.702012Z","shell.execute_reply":"2025-08-11T18:05:34.716391Z"}}
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

# %% [code] {"_kg_hide-input":true,"execution":{"iopub.status.busy":"2025-08-11T18:05:34.717679Z","iopub.execute_input":"2025-08-11T18:05:34.717896Z","iopub.status.idle":"2025-08-11T18:05:34.735113Z","shell.execute_reply.started":"2025-08-11T18:05:34.717871Z","shell.execute_reply":"2025-08-11T18:05:34.73441Z"}}
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

        self.imu_proj = ResNetSEBlock(in_channels=3*kwargs["imu2_channels"], out_channels=kwargs["imu2_channels"])
        self.thm_proj = ResNetSEBlock(in_channels=5*kwargs["thm2_channels"], out_channels=kwargs["thm2_channels"])
        self.tof_proj = ResNetSEBlock(in_channels=5*kwargs["tof2_channels"], out_channels=kwargs["tof2_channels"])

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
        
        self.noise = GaussianNoise(kwargs['gaussian_noise_rate'])
        self.dense = nn.Sequential(
            nn.Linear(kwargs['imu2_channels']+kwargs['thm2_channels']+kwargs['tof2_channels'], kwargs['dense_channels']),
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

# %% [markdown]
# # Settings

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T18:05:34.735896Z","iopub.execute_input":"2025-08-11T18:05:34.736143Z","iopub.status.idle":"2025-08-11T18:05:41.887745Z","shell.execute_reply.started":"2025-08-11T18:05:34.736117Z","shell.execute_reply":"2025-08-11T18:05:41.88714Z"}}
CUDA0 = "cuda:0"
seed = 0
batch_size = 64
num_workers = 4
n_folds = 5

root_dir = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
universe_csv_path = Path("/kaggle/input/cmi-precompute/pytorch/all/1/tof-1_raw.csv")

imu_only = False

deterministic = kagglehub.package_import('wasupandceacar/deterministic').deterministic
deterministic.init_all(seed)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T18:05:41.888435Z","iopub.execute_input":"2025-08-11T18:05:41.888864Z","iopub.status.idle":"2025-08-11T18:07:56.15879Z","shell.execute_reply.started":"2025-08-11T18:05:41.888845Z","shell.execute_reply":"2025-08-11T18:07:56.15791Z"}}
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

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T18:07:56.159862Z","iopub.execute_input":"2025-08-11T18:07:56.160447Z","iopub.status.idle":"2025-08-11T18:07:59.825605Z","shell.execute_reply.started":"2025-08-11T18:07:56.160423Z","shell.execute_reply":"2025-08-11T18:07:59.824879Z"}}
class CMIModel(nn.Module):
    def __init__(self, target_classes_num, non_target_classes_num, **kwargs):
        super().__init__()
        self.backbone = CMIBackbone(dataset.imu_dim, dataset.thm_dim, dataset.tof_dim, **kwargs)
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
    
    def forward(self, imu, thm, tof):
        feat = self.backbone(imu, thm, tof)
        targets_y = self.target_classifier(feat)
        non_targets_y = self.non_target_classifier(feat)
        return torch.cat([targets_y, non_targets_y], dim=1)

model_function = CMIModel
model_args = {"imu1_channels": 128, "imu2_channels": 256, "imu1_dropout": 0.3, "imu2_dropout": 0.25,
              "imu1_layers": 0, "imu2_layers": 0, 
              "thm1_channels": 32, "thm2_channels": 64, "thm1_dropout": 0.25, "thm2_dropout": 0.2,
              "thm1_layers": 0, "thm2_layers": 0, 
              "tof1_channels": 256, "tof2_channels": 512, "tof1_dropout": 0.4, "tof2_dropout": 0.3,
              "tof1_layers": 0, "tof2_layers": 0, 
              "lstm_hidden_size": 128, "gru_hidden_size": 128, "gaussian_noise_rate": 0.1, "dense_channels": 32,
              "cls_channels1": 256, "cls_dropout1": 0.2, "cls_channels2": 128, "cls_dropout2": 0.2,
              "target_classes_num": 8, "non_target_classes_num": 10,}
model_dir = Path("/kaggle/input/cmi-models-public/pytorch/base04/1")

model_dicts = [
    {
        "model_function": model_function,
        "model_args": model_args,
        "model_path": model_dir / f"fold{fold}/best_ema.pt",
    } for fold in range(n_folds)
]

def replace(k):
    k = k.replace("_orig_mod.", "")
    return k

models2 = list()
for model_dict in model_dicts:
    model_function = model_dict["model_function"]
    model_args = model_dict["model_args"]
    model_path = model_dict["model_path"]
    model = model_function(**model_args).to(CUDA0)
    state_dict = {replace(k): v for k,v in torch.load(model_path).items()}
    model.load_state_dict(state_dict)
    model = model.eval()
    models2.append(model)

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T18:07:59.827444Z","iopub.execute_input":"2025-08-11T18:07:59.827645Z","iopub.status.idle":"2025-08-11T18:10:11.433727Z","shell.execute_reply.started":"2025-08-11T18:07:59.827629Z","shell.execute_reply":"2025-08-11T18:10:11.432997Z"}}
metric_package = kagglehub.package_import('wasupandceacar/cmi-metric')

metric = metric_package.Metric()
imu_only_metric = metric_package.Metric()

def to_cuda(*tensors):
    return [tensor.to(CUDA0) for tensor in tensors]

def inference(model, imu, thm, tof):
    imus, thms, tofs = dataset.full_dataset.split5(imu, thm, tof)
    with autocast(device_type='cuda'):
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
'''
for fold, model in enumerate(models2):
    valid_loader = get_fold_dataset(dataset, fold)
    valid_bar = tqdm(valid_loader, desc=f"Valid", leave=False)
    valid(model, valid_bar)

print(f"""
Normal score: {metric.score()}
IMU only score: {imu_only_metric.score()}
""")
'''

# %% [code] {"execution":{"iopub.status.busy":"2025-08-11T18:10:11.434715Z","iopub.execute_input":"2025-08-11T18:10:11.435036Z","iopub.status.idle":"2025-08-11T18:10:13.765355Z","shell.execute_reply.started":"2025-08-11T18:10:11.435009Z","shell.execute_reply":"2025-08-11T18:10:13.764736Z"}}
def avg_predict(models, imu, thm, tof):
    outputs = []
    with autocast(device_type='cuda'):
        for model in models:
            pred_y = inference(model, imu, thm, tof)
            outputs.append(pred_y)
    return torch.mean(torch.stack(outputs), dim=0)



# predict_2

def predict2(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    imu, thm, tof = dataset.full_dataset.inference_process(sequence)
    with torch.no_grad():
        imu, thm, tof = to_cuda(imu, thm, tof)
        logits = avg_predict(models2, imu, thm, tof)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
    return probabilities # logits.cpu().numpy()
    # return dataset.le.classes_[logits.argmax(dim=1).cpu()]


# -*- coding: utf-8 -*-
"""gated-gru-hybrid-ensemble-v02.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/15f-PUIU6Tc6qYWYP6g7trekz1LypFFwW
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import random
import math
import matplotlib.pyplot as plt
import polars as pl
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, add, MaxPooling1D, Dropout,
    Bidirectional, GRU, GlobalAveragePooling1D, Dense, Multiply, Reshape,
    Lambda, Concatenate
)
from tensorflow.keras.optimizers import Adam as AdamTF
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import Sequence, to_categorical, pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers.schedules import CosineDecay

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam as AdamTorch
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from scipy.spatial.transform import Rotation as R
from scipy.signal import firwin

# è©•ä¾¡ãƒ¡ãƒˆãƒªã‚¯ã‚¹ã�¯ãƒ­ãƒ¼ã‚«ãƒ«æ¤œè¨¼/å­¦ç¿’æ™‚ã�«ã�®ã�¿ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆ
try:
    from cmi_2025_metric_copy_for_import import CompetitionMetric
except ImportError:
    CompetitionMetric = None
    print("CompetitionMetric could not be imported. OOF/CV score will not be calculated.")

def seed_everything(seed=42):
    """
    å®Ÿè¡Œç’°å¢ƒã�®ä¹±æ•°ã‚·ãƒ¼ãƒ‰ã‚’çµ±ä¸€çš„ã�«è¨­å®šã�™ã‚‹é–¢æ•°ã€‚
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(2025)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # torch.backends.cudnn.deterministic = True # ãƒ‘ãƒ•ã‚©ãƒ¼ãƒ�ãƒ³ã‚¹ã�Œä½�ä¸‹ã�™ã‚‹å�¯èƒ½æ€§ã�Œã�‚ã‚‹ã�Ÿã‚�ã‚³ãƒ¡ãƒ³ãƒˆã‚¢ã‚¦ãƒˆ
    # torch.backends.cudnn.benchmark = False

seed_everything(seed=42)
warnings.filterwarnings("ignore")

TRAIN = False

# --- ãƒ‘ã‚¹è¨­å®š ---
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
# YOUR_MODELS_DIRã�¯è‡ªåˆ†ã�®å­¦ç¿’æ¸ˆã�¿ãƒ¢ãƒ‡ãƒ«ã�Œæ ¼ç´�ã�•ã‚Œã�¦ã�„ã‚‹Kaggleãƒ‡ãƒ¼ã‚¿ã‚»ãƒƒãƒˆã�®ãƒ‘ã‚¹ã�«è¨­å®šã�—ã�¦ã��ã� ã�•ã�„
YOUR_MODELS_DIR = Path("/kaggle/input/cmi-data-gated-gru") # â˜…â˜…â˜… è‡ªåˆ†ã�®ãƒ¢ãƒ‡ãƒ«ãƒ‘ã‚¹ã�«å¤‰æ›´ â˜…â˜…â˜…
PUBLIC_TF_MODEL_DIR = Path("/kaggle/input/lb-0-78-quaternions-tf-bilstm-gru-attention")
PUBLIC_PT_MODEL_DIR = Path("/kaggle/input/cmi3-models-p")
EXPORT_DIR = Path("./") # å­¦ç¿’æ¸ˆã�¿ãƒ¢ãƒ‡ãƒ«ã‚„ã‚¢ãƒ¼ãƒ†ã‚£ãƒ•ã‚¡ã‚¯ãƒˆã�®ä¿�å­˜å…ˆ

# --- ãƒ¢ãƒ‡ãƒ«å­¦ç¿’ãƒ�ã‚¤ãƒ‘ãƒ¼ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿ ---
BATCH_SIZE = 64          # ãƒ�ãƒƒãƒ�ã‚µã‚¤ã‚º
PAD_PERCENTILE = 95      # ã‚·ãƒ¼ã‚±ãƒ³ã‚¹é•·ã�®ãƒ‘ãƒ‡ã‚£ãƒ³ã‚°ã‚’æ±ºã‚�ã‚‹ã�Ÿã‚�ã�®ãƒ‘ãƒ¼ã‚»ãƒ³ã‚¿ã‚¤ãƒ«å€¤
LR_INIT = 4e-4           # å­¦ç¿’ç�‡ã�®åˆ�æœŸå€¤ (å¾®èª¿æ•´)
WD = 3e-3                # Weight Decayï¼ˆL2æ­£å‰‡åŒ–ï¼‰ã�®ä¿‚æ•°
MIXUP_ALPHA = 0.4        # Mixupã�®Î±å€¤
EPOCHS = 360             # æœ€å¤§ã‚¨ãƒ�ãƒƒã‚¯æ•° (å¢—åŠ )
PATIENCE = 50            # EarlyStoppingã�®patience (å¢—åŠ )
N_SPLITS = 10             # ã‚¯ãƒ­ã‚¹ãƒ�ãƒªãƒ‡ãƒ¼ã‚·ãƒ§ãƒ³ã�®åˆ†å‰²æ•°
MASKING_PROB = 0.25      # å­¦ç¿’æ™‚ã�«TOF/THMãƒ‡ãƒ¼ã‚¿ã‚’ãƒ�ã‚¹ã‚¯ã�™ã‚‹ç¢ºç�‡
GATE_LOSS_WEIGHT = 0.2   # Gatedãƒ¢ãƒ‡ãƒ«ã�®ã‚²ãƒ¼ãƒˆæ��å¤±ã�«å¯¾ã�™ã‚‹é‡�ã�¿

print(f"â–¶ ãƒ©ã‚¤ãƒ–ãƒ©ãƒªã�®ã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆå®Œäº†")
print(f"  - TensorFlow: {tf.__version__}")
print(f"  - PyTorch: {torch.__version__}")
print(f"â–¶ TRAINãƒ¢ãƒ¼ãƒ‰: {TRAIN}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PyTorchãƒ¢ãƒ‡ãƒ«ç”¨ã�®æ¨™æº–åŒ–ãƒ‘ãƒ©ãƒ¡ãƒ¼ã‚¿
mean_pt = torch.tensor([
    0, 0, 0, 0, 0, 0, 9.0319e-03, 1.0849e+00, -2.6186e-03, 3.7651e-03,
    -5.3660e-03, -2.8177e-03, 1.3318e-03, -1.5876e-04, 6.3495e-01,
    6.2877e-01, 6.0607e-01, 6.2142e-01, 6.3808e-01, 6.5420e-01,
    7.4102e-03, -3.4159e-03, -7.5237e-03, -2.6034e-02, 2.9704e-02,
    -3.1546e-02, -2.0610e-03, -4.6986e-03, -4.7216e-03, -2.6281e-02,
    1.5799e-02, 1.0016e-02
], dtype=torch.float32).view(1, -1, 1).to(device)

std_pt = torch.tensor([
    1, 1, 1, 1, 1, 1, 0.2067, 0.8583, 0.3162,
    0.2668, 0.2917, 0.2341, 0.3023, 0.3281, 1.0264, 0.8838, 0.8686, 1.0973,
    1.0267, 0.9018, 0.4658, 0.2009, 0.2057, 1.2240, 0.9535, 0.6655, 0.2941,
    0.3421, 0.8156, 0.6565, 1.1034, 1.5577
], dtype=torch.float32).view(1, -1, 1).to(device) + 1e-8

class ImuFeatureExtractor(nn.Module):
    """
    â˜…â˜…â˜… PyTorchãƒ¢ãƒ‡ãƒ«ç”¨ã�®ç‰¹å¾´é‡�æŠ½å‡ºå™¨ â˜…â˜…â˜…
    å…¬é–‹ãƒ¢ãƒ‡ãƒ«ã�®é‡�ã�¿ã�¨ä¸€è‡´ã�•ã�›ã‚‹ã�Ÿã‚�ã€�å…ƒã�®æ­£ã�—ã�„å®šç¾©ã�«ä¿®æ­£ã€‚
    """
    def __init__(self, fs=100., add_quaternion=False):
        super().__init__()
        self.fs = fs
        self.add_quaternion = add_quaternion

        k = 15

        # â–¼â–¼â–¼ã€�ã�“ã�“ã�Œä¿®æ­£ç‚¹ã€‘â–¼â–¼â–¼
        # å…¬é–‹ãƒ¢ãƒ‡ãƒ«ã�®é‡�ã�¿ãƒ•ã‚¡ã‚¤ãƒ«ã�«å­˜åœ¨ã�™ã‚‹ 'self.lpf' å±¤ã‚’å†�åº¦è¿½åŠ ã�™ã‚‹
        self.lpf = nn.Conv1d(6, 6, kernel_size=k, padding=k//2,
                                 groups=6, bias=False)
        nn.init.kaiming_uniform_(self.lpf.weight, a=math.sqrt(5))
        # â–²â–²â–²ã€�ã�“ã�“ã�¾ã�§ã�Œä¿®æ­£ç‚¹ã€‘â–²â–²â–²

        self.lpf_acc  = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
        self.lpf_gyro = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)

    def forward(self, imu):
        acc  = imu[:, 0:3, :]
        gyro = imu[:, 3:6, :]

        # 1) magnitude
        acc_mag  = torch.norm(acc,  dim=1, keepdim=True)
        gyro_mag = torch.norm(gyro, dim=1, keepdim=True)

        # 2) jerk
        jerk = F.pad(acc[:, :, 1:] - acc[:, :, :-1], (1,0))
        gyro_delta = F.pad(gyro[:, :, 1:] - gyro[:, :, :-1], (1,0))

        # 3) energy
        acc_pow  = acc ** 2
        gyro_pow = gyro ** 2

        # 4) LPF / HPF
        # self.lpf ã�¯ forwardãƒ‘ã‚¹ã�§ã�¯ä½¿ã‚�ã‚Œã�¦ã�„ã�ªã�„ã�Œã€�é‡�ã�¿èª­ã�¿è¾¼ã�¿ã�®ã�Ÿã‚�ã�«å®šç¾©ã�Œå¿…è¦�
        acc_lpf  = self.lpf_acc(acc)
        acc_hpf  = acc - acc_lpf
        gyro_lpf = self.lpf_gyro(gyro)
        gyro_hpf = gyro - gyro_lpf

        features = [
            acc, gyro,
            acc_mag, gyro_mag,
            jerk, gyro_delta,
            acc_pow, gyro_pow,
            acc_lpf, acc_hpf,
            gyro_lpf, gyro_hpf,
        ]
        return torch.cat(features, dim=1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False), nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False), nn.Sigmoid()
        )
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

class TwoBranchModel(nn.Module):
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

class PublicTwoBranchModel(nn.Module):
    """
    â˜…â˜…â˜… å…¬é–‹ã�•ã‚Œã�¦ã�„ã‚‹PyTorchãƒ¢ãƒ‡ãƒ«ï¼ˆãƒ¢ãƒ‡ãƒ«ç¾¤Cï¼‰ã‚’èª­ã�¿è¾¼ã‚€ã�Ÿã‚�ã�®ã€�å…ƒã�®ã‚¢ãƒ¼ã‚­ãƒ†ã‚¯ãƒ�ãƒ£ã‚’æŒ�ã�¤ã‚¯ãƒ©ã‚¹ â˜…â˜…â˜…
    """
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
        self.bilstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True) # GRUã�§ã�¯ã�ªã��LSTM
        self.lstm_dropout = nn.Dropout(dropouts[4])
        self.attention = AttentionLayer(256) # 128*2 for bidirectional
        self.dense1, self.bn_dense1, self.drop1 = nn.Linear(256, 256, bias=False), nn.BatchNorm1d(256), nn.Dropout(dropouts[5])
        self.dense2, self.bn_dense2, self.drop2 = nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128), nn.Dropout(dropouts[6])
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, x):
        imu_raw = x[:, :, :self.fir_nchan].transpose(1, 2)
        tof = x[:, :, self.fir_nchan:].transpose(1, 2)
        imu_fe = self.imu_fe(imu_raw)
        filtered = F.conv1d(imu_fe[:, :self.fir_nchan, :], self.fir_kernel, padding=self.fir_kernel.shape[-1] // 2, groups=self.fir_nchan)
        # mean_pt, std_pt ã�¯äº‹å‰�ã�«å®šç¾©ã�•ã‚Œã�¦ã�„ã‚‹ã‚°ãƒ­ãƒ¼ãƒ�ãƒ«å¤‰æ•°
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

# =============================================================================
# ## ç‰¹å¾´é‡�ã‚¨ãƒ³ã‚¸ãƒ‹ã‚¢ãƒªãƒ³ã‚°é–¢æ•°
# =============================================================================
def remove_gravity_from_acc3(acc_data, rot_data):
    """åŠ é€Ÿåº¦ãƒ‡ãƒ¼ã‚¿ã�‹ã‚‰é‡�åŠ›æˆ�åˆ†ã‚’é™¤å�»ã�™ã‚‹"""
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
    """ã‚¯ã‚©ãƒ¼ã‚¿ãƒ‹ã‚ªãƒ³ã�‹ã‚‰è§’é€Ÿåº¦ã‚’è¨ˆç®—ã�™ã‚‹"""
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
    """ã‚¯ã‚©ãƒ¼ã‚¿ãƒ‹ã‚ªãƒ³ã�‹ã‚‰è§’è·�é›¢ã‚’è¨ˆç®—ã�™ã‚‹"""
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

def time_sum(x): return K.sum(x, axis=1)
def squeeze_last_axis(x): return tf.squeeze(x, axis=-1)
def expand_last_axis(x): return tf.expand_dims(x, axis=-1)

def se_block(x, reduction=8):
    """Squeeze-and-Excitationãƒ–ãƒ­ãƒƒã‚¯"""
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    """Residual SE-CNNãƒ–ãƒ­ãƒƒã‚¯"""
    shortcut = x
    # 2å±¤ã�®Conv1D
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    # SEãƒ–ãƒ­ãƒƒã‚¯
    x = se_block(x)
    # ã‚·ãƒ§ãƒ¼ãƒˆã‚«ãƒƒãƒˆæ�¥ç¶š
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False, kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x

def attention_layer(inputs):
    """ã‚¢ãƒ†ãƒ³ã‚·ãƒ§ãƒ³å±¤"""
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context

class GatedMixupGenerator(Sequence):
    """Mixupã�¨ã‚»ãƒ³ã‚µãƒ¼ãƒ�ã‚¹ã‚­ãƒ³ã‚°ã‚’é�©ç”¨ã�™ã‚‹ãƒ‡ãƒ¼ã‚¿ã‚¸ã‚§ãƒ�ãƒ¬ãƒ¼ã‚¿"""
    def __init__(self, X, y, batch_size, imu_dim, class_weight=None, alpha=0.2, masking_prob=0.0):
        self.X, self.y, self.batch, self.imu_dim = X, y, batch_size, imu_dim
        self.class_weight, self.alpha, self.masking_prob = class_weight, alpha, masking_prob
        self.indices = np.arange(len(X))

    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch))

    def __getitem__(self, i):
        idx = self.indices[i*self.batch:(i+1)*self.batch]
        Xb, yb = self.X[idx].copy(), self.y[idx].copy()

        sample_weights = np.ones(len(Xb), dtype='float32')
        if self.class_weight:
            sample_weights = np.array([self.class_weight.get(i, 1.0) for i in yb.argmax(axis=1)])

        gate_target = np.ones(len(Xb), dtype='float32')
        if self.masking_prob > 0:
            for j in range(len(Xb)):
                if np.random.rand() < self.masking_prob:
                    Xb[j, :, self.imu_dim:] = 0
                    gate_target[j] = 0.0

        if self.alpha > 0:
            lam = np.random.beta(self.alpha, self.alpha)
            perm = np.random.permutation(len(Xb))
            X_mix = lam * Xb + (1 - lam) * Xb[perm]
            y_mix = lam * yb + (1 - lam) * yb[perm]
            gate_target_mix = lam * gate_target + (1 - lam) * gate_target[perm]
            sample_weights_mix = lam * sample_weights + (1 - lam) * sample_weights[perm]
            return X_mix, {'main_output': y_mix, 'tof_gate': gate_target_mix}, sample_weights_mix

        return Xb, {'main_output': yb, 'tof_gate': gate_target}, sample_weights

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

def build_gated_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    """
    è‡ªä½œã�®Gated Two-Branchãƒ¢ãƒ‡ãƒ«ã‚’æ§‹ç¯‰ã�™ã‚‹é–¢æ•°ã€‚
    [æ”¹è‰¯ç‚¹] LSTMã‚’GRUã�«å¤‰æ›´ã€�å…¨çµ�å�ˆå±¤ã‚’1å±¤è¿½åŠ ã€‚
    """
    inp = Input(shape=(pad_len, imu_dim + tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    # IMUãƒ–ãƒ©ãƒ³ãƒ� (Deep)
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)

    # TOF/THMãƒ–ãƒ©ãƒ³ãƒ� (Light) with Gating
    x2_base = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2_base = BatchNormalization()(x2_base); x2_base = Activation('relu')(x2_base)
    x2_base = MaxPooling1D(2)(x2_base); x2_base = Dropout(0.2)(x2_base)
    x2_base = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x2_base)
    x2_base = BatchNormalization()(x2_base); x2_base = Activation('relu')(x2_base)
    x2_base = MaxPooling1D(2)(x2_base); x2_base = Dropout(0.2)(x2_base)

    # Gatingæ©Ÿæ§‹
    gate_input = GlobalAveragePooling1D()(tof)
    gate_input = Dense(16, activation='relu')(gate_input)
    gate = Dense(1, activation='sigmoid', name='tof_gate')(gate_input)
    x2 = Multiply()([x2_base, gate])

    # ãƒ–ãƒ©ãƒ³ãƒ�ã�®ãƒ�ãƒ¼ã‚¸ã�¨å¾Œç¶šå±¤
    merged = Concatenate()([x1, x2])
    # â˜…æ”¹è‰¯ç‚¹: LSTM -> GRU
    x = Bidirectional(GRU(256, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    x = Dropout(0.45)(x)
    x = attention_layer(x)

    # â˜…æ”¹è‰¯ç‚¹: å…¨çµ�å�ˆå±¤ã‚’1å±¤è¿½åŠ ã�—ã�¦è¡¨ç�¾åŠ›ã‚’å�‘ä¸Š
    for units, drop in [(512, 0.5), (256, 0.4), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(drop)(x)

    out = Dense(n_classes, activation='softmax', name='main_output', kernel_regularizer=l2(wd))(x)

    return Model(inputs=inp, outputs=[out, gate])

# -----------------------------------------------------------------------------
# ### æ�¨è«–ãƒ¢ãƒ¼ãƒ‰ (`TRAIN = False`)
# -----------------------------------------------------------------------------

print("â–¶ æ�¨è«–ãƒ¢ãƒ¼ãƒ‰é–‹å§‹ â€“ å­¦ç¿’æ¸ˆã�¿ãƒ¢ãƒ‡ãƒ«ã�¨ã‚¢ãƒ¼ãƒ†ã‚£ãƒ•ã‚¡ã‚¯ãƒˆã‚’èª­ã�¿è¾¼ã�¿ã�¾ã�™...")

# --- ãƒ¢ãƒ‡ãƒ«ç¾¤A (è‡ªä½œTF/Kerasãƒ¢ãƒ‡ãƒ«) ã�®èª­ã�¿è¾¼ã�¿ ---
print("  ãƒ¢ãƒ‡ãƒ«ç¾¤A (è‡ªä½œ5-Fold Gated GRUãƒ¢ãƒ‡ãƒ«) ã‚’èª­ã�¿è¾¼ã�¿ä¸­...")
final_feature_cols_A = np.load(YOUR_MODELS_DIR / "final_feature_cols.npy", allow_pickle=True).tolist()
pad_len_A = int(np.load(YOUR_MODELS_DIR / "sequence_maxlen.npy"))
scaler_A = joblib.load(YOUR_MODELS_DIR / "scaler.pkl")
gesture_classes = np.load(YOUR_MODELS_DIR / "gesture_classes.npy", allow_pickle=True)
custom_objs_A = {'time_sum': time_sum, 'squeeze_last_axis': squeeze_last_axis, 'expand_last_axis': expand_last_axis,
                 'se_block': se_block, 'residual_se_cnn_block': residual_se_cnn_block, 'attention_layer': attention_layer}
models_A = [load_model(YOUR_MODELS_DIR / f"final_model_fold_{f}.h5", compile=False, custom_objects=custom_objs_A) for f in range(N_SPLITS)]
print(f"  > {len(models_A)}å€‹ã�®ãƒ¢ãƒ‡ãƒ«ã‚’æ­£å¸¸ã�«èª­ã�¿è¾¼ã�¿ã�¾ã�—ã�Ÿã€‚")

# --- ãƒ¢ãƒ‡ãƒ«ç¾¤B (å…¬é–‹TF/Kerasãƒ¢ãƒ‡ãƒ«) ã�®èª­ã�¿è¾¼ã�¿ ---
print("\n  ãƒ¢ãƒ‡ãƒ«ç¾¤B (å…¬é–‹TF/Kerasãƒ¢ãƒ‡ãƒ«) ã‚’èª­ã�¿è¾¼ã�¿ä¸­...")
final_feature_cols_B = np.load(PUBLIC_TF_MODEL_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len_B = int(np.load(PUBLIC_TF_MODEL_DIR / "sequence_maxlen.npy"))
scaler_B = joblib.load(PUBLIC_TF_MODEL_DIR / "scaler.pkl")
custom_objs_B = custom_objs_A # public modelã‚‚å�Œã�˜ã‚«ã‚¹ã‚¿ãƒ ã‚ªãƒ–ã‚¸ã‚§ã‚¯ãƒˆã‚’ä½¿ç”¨
model_B = load_model(PUBLIC_TF_MODEL_DIR / "gesture_two_branch_mixup.h5", compile=False, custom_objects=custom_objs_B)
print("  > 1å€‹ã�®ãƒ¢ãƒ‡ãƒ«ã‚’æ­£å¸¸ã�«èª­ã�¿è¾¼ã�¿ã�¾ã�—ã�Ÿã€‚")

# --- ãƒ¢ãƒ‡ãƒ«ç¾¤C (å…¬é–‹PyTorchãƒ¢ãƒ‡ãƒ«) ã�®èª­ã�¿è¾¼ã�¿ ---
print("\n  ãƒ¢ãƒ‡ãƒ«ç¾¤C (å…¬é–‹PyTorchãƒ¢ãƒ‡ãƒ«) ã‚’èª­ã�¿è¾¼ã�¿ä¸­...")
final_feature_cols_C = np.load(PUBLIC_PT_MODEL_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len_C = int(np.load(PUBLIC_PT_MODEL_DIR / "sequence_maxlen.npy"))
scaler_C = joblib.load(PUBLIC_PT_MODEL_DIR / "scaler.pkl")

pt_models = []
for f in range(5):
    checkpoint = torch.load(PUBLIC_PT_MODEL_DIR / f"gesture_two_branch_fold{f}.pth", map_location=device)
    cfg = {'pad_len': checkpoint['pad_len'], 'imu_dim_raw': checkpoint['imu_dim'],
           'tof_dim': checkpoint['tof_dim'], 'n_classes': checkpoint['n_classes']}
    m = PublicTwoBranchModel(**cfg).to(device)
    m.load_state_dict(checkpoint['model_state_dict'])
    m.eval()
    pt_models.append(m)
print(f"  > {len(pt_models)}å€‹ã�®ãƒ¢ãƒ‡ãƒ«ã‚’æ­£å¸¸ã�«èª­ã�¿è¾¼ã�¿ã�¾ã�—ã�Ÿã€‚")


# predict_3

# --- `predict`é–¢æ•°ã�®å®šç¾© ---
def predict3(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_seq_orig = sequence.to_pandas()
    df_seq_A = df_seq_orig.copy()
    
    linear_accel_A = remove_gravity_from_acc3(df_seq_A[['acc_x','acc_y','acc_z']], df_seq_A[['rot_x','rot_y','rot_z','rot_w']])
    df_seq_A['linear_acc_x'], df_seq_A['linear_acc_y'], df_seq_A['linear_acc_z'] = linear_accel_A[:,0], linear_accel_A[:,1], linear_accel_A[:,2]
    df_seq_A['linear_acc_mag'] = np.linalg.norm(linear_accel_A, axis=1)
    df_seq_A['linear_acc_mag_jerk'] = df_seq_A['linear_acc_mag'].diff().fillna(0)
    angular_vel_A = calculate_angular_velocity_from_quat3(df_seq_A[['rot_x','rot_y','rot_z','rot_w']])
    df_seq_A['angular_vel_x'], df_seq_A['angular_vel_y'], df_seq_A['angular_vel_z'] = angular_vel_A[:,0], angular_vel_A[:,1], angular_vel_A[:,2]
    df_seq_A['angular_distance'] = calculate_angular_distance3(df_seq_A[['rot_x','rot_y','rot_z','rot_w']])
    for col in ['rot_x', 'rot_y', 'rot_z', 'rot_w']:
        df_seq_A[f'{col}_diff'] = df_seq_A[col].diff().fillna(0)
    cols_for_stats=['linear_acc_mag','linear_acc_mag_jerk','angular_distance']
    for col in cols_for_stats:
        df_seq_A[f'{col}_skew'], df_seq_A[f'{col}_kurt'] = df_seq_A[col].skew(), df_seq_A[col].kurtosis()
    for i in range(1,6):
        if f'tof_{i}_v0' in df_seq_A.columns:
            pixel_cols=[f"tof_{i}_v{p}" for p in range(64)]; tof_data=df_seq_A[pixel_cols].replace(-1,np.nan)
            df_seq_A[f'tof_{i}_mean'], df_seq_A[f'tof_{i}_std'], df_seq_A[f'tof_{i}_min'], df_seq_A[f'tof_{i}_max'] = tof_data.mean(axis=1),tof_data.std(axis=1),tof_data.min(axis=1),tof_data.max(axis=1)
    tof_mean_cols=[f'tof_{i}_mean' for i in range(1,6) if f'tof_{i}_mean' in df_seq_A.columns]
    if tof_mean_cols:
        df_seq_A['tof_std_across_sensors']=df_seq_A[tof_mean_cols].std(axis=1)
        df_seq_A['tof_range_across_sensors']=df_seq_A[tof_mean_cols].max(axis=1)-df_seq_A[tof_mean_cols].min(axis=1)
    thm_cols=[f'thm_{i}' for i in range(1,6) if f'thm_{i}' in df_seq_A.columns]
    if thm_cols:
        df_seq_A['thm_std_across_sensors']=df_seq_A[thm_cols].std(axis=1)
        df_seq_A['thm_range_across_sensors']=df_seq_A[thm_cols].max(axis=1)-df_seq_A[thm_cols].min(axis=1)
    # (æ�¨è«– A)
    mat_A = df_seq_A[final_feature_cols_A].ffill().bfill().fillna(0).values.astype('float32')
    mat_A = scaler_A.transform(mat_A)
    pad_input_A = pad_sequences([mat_A], maxlen=pad_len_A, padding='post', dtype='float32')
    preds_A_folds = [model.predict(pad_input_A, verbose=0)[0] for model in models_A]
    avg_pred_A = np.mean(preds_A_folds, axis=0)

    # --- 2. ãƒ¢ãƒ‡ãƒ«ç¾¤B (å…¬é–‹TFãƒ¢ãƒ‡ãƒ«) ã�®äºˆæ¸¬ ---
    df_seq_B = df_seq_orig.copy()
    # (ç‰¹å¾´é‡�ç”Ÿæˆ� B)
    df_seq_B['acc_mag']=np.sqrt(df_seq_B['acc_x']**2+df_seq_B['acc_y']**2+df_seq_B['acc_z']**2)
    df_seq_B['rot_angle']=2*np.arccos(df_seq_B['rot_w'].clip(-1,1))
    df_seq_B['acc_mag_jerk']=df_seq_B['acc_mag'].diff().fillna(0)
    df_seq_B['rot_angle_vel']=df_seq_B['rot_angle'].diff().fillna(0)
    linear_accel_B=remove_gravity_from_acc3(df_seq_B,df_seq_B)
    df_seq_B['linear_acc_x'],df_seq_B['linear_acc_y'],df_seq_B['linear_acc_z']=linear_accel_B[:,0],linear_accel_B[:,1],linear_accel_B[:,2]
    df_seq_B['linear_acc_mag']=np.sqrt(df_seq_B['linear_acc_x']**2+df_seq_B['linear_acc_y']**2+df_seq_B['linear_acc_z']**2)
    df_seq_B['linear_acc_mag_jerk']=df_seq_B['linear_acc_mag'].diff().fillna(0)
    angular_vel_B=calculate_angular_velocity_from_quat3(df_seq_B)
    df_seq_B['angular_vel_x'],df_seq_B['angular_vel_y'],df_seq_B['angular_vel_z']=angular_vel_B[:,0],angular_vel_B[:,1],angular_vel_B[:,2]
    df_seq_B['angular_distance']=calculate_angular_distance3(df_seq_B)
    for i in range(1,6):
        if f'tof_{i}_v0' in df_seq_B.columns:
            pixel_cols=[f"tof_{i}_v{p}" for p in range(64)]; tof_data=df_seq_B[pixel_cols].replace(-1,np.nan)
            df_seq_B[f"tof_{i}_mean"],df_seq_B[f"tof_{i}_std"],df_seq_B[f"tof_{i}_min"],df_seq_B[f"tof_{i}_max"]=tof_data.mean(axis=1),tof_data.std(axis=1),tof_data.min(axis=1),tof_data.max(axis=1)
    # (æ�¨è«– B)
    mat_B = df_seq_B[final_feature_cols_B].ffill().bfill().fillna(0).values.astype('float32')
    mat_B = scaler_B.transform(mat_B)
    pad_input_B = pad_sequences([mat_B], maxlen=pad_len_B, padding='post', dtype='float32')
    pred_B = model_B.predict(pad_input_B, verbose=0)
    if isinstance(pred_B, list): pred_B = pred_B[0]

    # --- 3. ãƒ¢ãƒ‡ãƒ«ç¾¤C (å…¬é–‹PyTorchãƒ¢ãƒ‡ãƒ«) ã�®äºˆæ¸¬ ---
    df_seq_C = df_seq_orig.copy() # Cã�¯ç‰¹å¾´é‡�ç”Ÿæˆ�ã�Œä¸�è¦�ã�ªã�Ÿã‚�ã€�ã‚³ãƒ”ãƒ¼ã�®ã�¿
    mat_C = df_seq_C[final_feature_cols_C].ffill().bfill().fillna(0).values.astype('float32')
    mat_C = scaler_C.transform(mat_C)
    pad_input_C = pad_sequences_torch3([mat_C], maxlen=pad_len_C, padding='pre', truncating='pre')
    with torch.no_grad():
        pt_input = torch.from_numpy(pad_input_C).to(device)
        preds_C_folds = [model(pt_input) for model in pt_models]
        avg_pred_C_logits = torch.mean(torch.stack(preds_C_folds), dim=0)
        avg_pred_C = torch.softmax(avg_pred_C_logits, dim=1).cpu().numpy()

    # --- 4. åŠ é‡�å¹³å�‡ã�«ã‚ˆã‚‹æœ€çµ‚æ±ºå®š ---

    #weights = {'A': 0.60, 'B': 0.15, 'C': 0.25}
    #weights = {'A': 0.50, 'B': 0.20, 'C': 0.30}
    #weights = {'A': 0.45, 'B': 0.22, 'C': 0.33}

    weights = {'A': 0.50, 'B': 0.20, 'C': 0.30}

    final_pred_proba = (weights['A'] * avg_pred_A + weights['B'] * pred_B + weights['C'] * avg_pred_C)

    return final_pred_proba


# help func.1

pred0,pred1,pred2, ws, cws, aws = 1,2,3, [0.274,0.342,0.382], [+0.011, -0.004, -0.007], [0.99, 0.01]

lp = [{ 'w':ws[0], 'p':pred0, 'n':'p0' },
      { 'w':ws[1], 'p':pred1, 'n':'p1' },
      { 'w':ws[2], 'p':pred2, 'n':'p2' }] 

lps_asc  = [{'w':p['w'], 'p':p['p'], 'n':p['n']} for p in lp]
lps_desc = [{'w':p['w'], 'p':p['p'], 'n':p['n']} for p in lp]

lps_asc  = sorted(lps_asc,  key=lambda k:k['p'],reverse=False)
lps_desc = sorted(lps_desc, key=lambda k:k['p'],reverse=True)

print(lps_asc, "\n\n", lps_desc, "") #------------------------

for p,cw in zip(lps_asc,  cws): p['w'] += cw
for p,cw in zip(lps_desc, cws): p['w'] += cw
    
print("-"*11)
print(lps_asc, "\n\n", lps_desc)     #------------------------

lps_asc  = sorted(lps_asc,  key=lambda k:k['n'],reverse=False)
lps_desc = sorted(lps_desc, key=lambda k:k['n'],reverse=False)

print("-"*11)
print(lps_asc, "\n\n", lps_desc)     #------------------------

lps = []

for a,d in zip(lps_asc, lps_desc):
    one_dict = {
        'w':a['w']* aws[0]+aws[1] *d['w'],
        'p':a['p'],
        'n':a['n']
    }
    lps.append(one_dict)

print("-"*11)                        #------------------------
print(lps)

wps = [ps["w"]*ps["p"] for ps in lps]

print("-"*11)                        #------------------------
print(wps)


# help func.2

def equ(_a,_b,_c,_k):
    _o = 0.999985
    if _a == _b and _a != _c: return [_k, _k, _o]
    if _a == _c and _b != _c: return [_k, _o, _k]
    if _b == _c and _a != _b: return [_o, _k, _k]
    return [1,1,1]

def corr(li):
    v2,v1 = min(li),     max(li)
    j2,j1 = li.index(v2),li.index(v1)                   
    v3,j3 = 1-(v1-1/v2), 3-(j1+j2)
    a = [_,_,_];
    a[j1],a[j2],a[j3] = v1,v2,v3
    return np.asarray(a)

s,_,f = 0.9974, 1, 1.0041

li = [f,s,_]

v2,v1 = min(li),max(li)
j2 = li.index(v2)
j1 = li.index(v1)
print(f'value.s = {v2}, index.s = {j2}')
print(f'value.f = {v1}, index.f = {j1}')

'''
                       2     1     0     : index.(j3)
indexes.(j1,j2) :  0,1,    0, 2    ,1,2

j3 = 3 - (j2 + j1)
''' 

j3 = 3 - (j2 + j1)

li = corr([f,s,_])

v3 = li[j3]

print(f'value.v3 = {v3}, index.v3 = {j3}',"   ", li)


# 1

# import numpy as np

# def predict(sequence, demographics):

#     pred0 = predict1(sequence, demographics)[0]
#     pred1 = predict2(sequence, demographics)[0]
#     pred2 = predict3(sequence, demographics)[0]                                

#     wts1, s,_,f = np.asarray([0.271, 0.349, 0.380]),    0.9974,  1, 1.0041
#     #--------------------------------------------
#     c1_123 = corr([f,s,_])
#     c1_132 = corr([f,_,s])
#     #--------------------------------------------
#     c1_213 = corr([s,f,_])
#     c1_231 = corr([_,f,s])
#     #--------------------------------------------
#     c1_312 = corr([s,_,f])
#     c1_321 = corr([_,s,f])
#     #--------------------------------------------
                                        
#     wts2, s,_,f = np.asarray([0.2705, 0.3495, 0.380]),  0.99744, 1, 1.00404 
#     #--------------------------------------------
#     c2_123 = corr([f,s,_])
#     c2_132 = corr([f,_,s])
#     #--------------------------------------------
#     c2_213 = corr([s,f,_])
#     c2_231 = corr([_,f,s])
#     #--------------------------------------------
#     c2_312 = corr([s,_,f])
#     c2_321 = corr([_,s,f])
#     #--------------------------------------------

#     # r =       5
#     # k = 1.00005
#     #-------------
#     _r  =         7
#     _k  = 1.0000007 

#     def equ(_a,_b,_c,_k=_k):
#         if _a == _b and _a != _c: return [_k, _k,  1]
#         if _a == _c and _b != _c: return [_k,  1, _k]
#         if _b == _c and _a != _b: return [ 1, _k, _k]
#         return [1,1,1]

#     preds = []
    
#     for _a,_b,_c in zip(pred0,pred1,pred2):
        
#         a,b,c = round(_a,_r),round(_b,_r),round(_c,_r)
        
#         if   a <= b <= c: _wts1 = wts1 * c1_123
#         elif a <= c <= b: _wts1 = wts1 * c1_132
#         elif b <= a <= c: _wts1 = wts1 * c1_213
#         elif b <= c <= a: _wts1 = wts1 * c1_231
#         elif c <= a <= b: _wts1 = wts1 * c1_312
#         elif c <= b <= a: _wts1 = wts1 * c1_321

#         _equ = equ(a,b,c)

#         if equ == [1,1,1]:

#             if   a <  b <  c: _wts2 = wts2 * c2_123
#             elif a <  c <  b: _wts2 = wts2 * c2_132
#             elif b <  a <  c: _wts2 = wts2 * c2_213
#             elif b <  c <  a: _wts2 = wts2 * c2_231
#             elif c <  a <  b: _wts2 = wts2 * c2_312
#             elif c <  b <  a: _wts2 = wts2 * c2_321
#             else:             _wts2 = wts2;

#             __wts = _wts2 *0.7 +\
#                     _wts1 *0.3
#         else:
#             __wts = _wts1
        
#         p = _a *__wts[0] *_equ[0] +\
#             _b *__wts[1] *_equ[1] +\
#             _c *__wts[2] *_equ[2]
        
#         preds.append(p)
        

#     avg_pred =  np.asarray(preds)
                
#     return dataset.le.classes_[avg_pred.argmax()]


# 2

# def predict(sequence, demographics):

#     pred0 = predict1(sequence, demographics)[0]
#     pred1 = predict2(sequence, demographics)[0]
#     pred2 = predict3(sequence, demographics)[0]    

#     #wts1, s,_,f = np.asarray([0.271,  0.349,  0.380 ]),  0.9974,  1, 1.0041   # Lb=0.850 # v11
#     #wts1, s,_,f = np.asarray([0.2711, 0.3494, 0.3795]),  0.99744, 1, 1.0039   # Lb=0.850 # v12
#     â„–wts1, s,_,f = np.asarray([0.271,  0.349,  0.380 ]),  0.9974,  1, 1.00405  # Lb=0.850 v13
#     wts1, s,_,f = np.asarray([0.271,  0.349,  0.380 ]),  0.99855, 1, 1.00404  # Lb=0.850 v14
    
#     #----------------------
#     c1_123 = corr([f,s,_])
#     c1_132 = corr([f,_,s])
#     #----------------------
#     c1_213 = corr([s,f,_])
#     c1_231 = corr([_,f,s])
#     #----------------------
#     c1_312 = corr([s,_,f])
#     c1_321 = corr([_,s,f])
#     #----------------------

#     r =      4
#     k = 1.0004     

#     preds = []
    
#     for _a,_b,_c in zip(pred0,pred1,pred2):
        
#         a,b,c = round(_a,r),round(_b,r),round(_c,r)
        
#         if   a <= b <= c: _wts1 = wts1 * c1_123
#         elif a <= c <= b: _wts1 = wts1 * c1_132
#         elif b <= a <= c: _wts1 = wts1 * c1_213
#         elif b <= c <= a: _wts1 = wts1 * c1_231
#         elif c <= a <= b: _wts1 = wts1 * c1_312
#         elif c <= b <= a: _wts1 = wts1 * c1_321

#         _equ = equ(a,b,c, k)

#         __wts = _wts1
        
#         p = _a *__wts[0] *_equ[0] +\
#             _b *__wts[1] *_equ[1] +\
#             _c *__wts[2] *_equ[2]

#         # p = a *__wts[0] *_equ[0] +\
#         #     b *__wts[1] *_equ[1] +\
#         #     c *__wts[2] *_equ[2]
        
#         preds.append(p)
        

#     avg_pred =  np.asarray(preds)
                
#     return dataset.le.classes_[avg_pred.argmax()]


# Japan solution

# def predict(sequence, demographics):
#     import numpy as np

#     # --- 3ãƒ¢ãƒ‡ãƒ«ã�®ç¢ºç�‡ ---
#     pred0 = predict1(sequence, demographics)[0]
#     pred1 = predict2(sequence, demographics)[0]
#     pred2 = predict3(sequence, demographics)[0]

#     # --- â‘  æ¸©åº¦ã‚·ãƒ£ãƒ¼ãƒ—ãƒ‹ãƒ³ã‚°ï¼ˆã�»ã‚“ã�®å°‘ã�—å°–ã‚‰ã�›ã‚‹ï¼‰ ---
#     def _sharpen(p, gamma=1.10, eps=1e-12):
#         p = np.clip(p, eps, 1.0)
#         p = p ** gamma
#         return p / p.sum()

#     pred0 = _sharpen(pred0, 1.10)
#     pred1 = _sharpen(pred1, 1.10)
#     pred2 = _sharpen(pred2, 1.10)

#     # --- â‘¡ è‡ªä¿¡ï¼ˆã‚¨ãƒ³ãƒˆãƒ­ãƒ”ãƒ¼ï¼‰ã�§ãƒ™ãƒ¼ã‚¹é‡�ã�¿ã‚’å¾®èª¿æ•´ ---
#     def _entropy(p, eps=1e-12):
#         p = np.clip(p, eps, 1.0)
#         return -np.sum(p * np.log(p))

#     H0, H1, H2 = _entropy(pred0), _entropy(pred1), _entropy(pred2)
#     conf = np.exp(-np.array([H0, H1, H2]))   # ä½�H=é«˜è‡ªä¿¡â†’å¤§ã��ã�„
#     conf = conf / conf.mean()                 # å¹³å�‡1ã�«æ­£è¦�åŒ–
#     beta = 0.35                               # åŠ¹ã��å…·å�ˆï¼ˆ0.2ã€œ0.5ã�§è»½ã��CVï¼‰
#     w_base = np.asarray([0.271, 0.347, 0.382])  # å›ºå®šãƒ™ãƒ¼ã‚¹
#     wts = (w_base * (conf ** beta))
#     wts = wts / wts.sum()                     # å�ˆè¨ˆ1ã�«

#     # --- â‘¢ ãƒ©ãƒ³ã‚¯ã�«å¿œã�˜ã�Ÿå¾®èª¿æ•´ä¿‚æ•° ---
#     c123 = np.asarray([1.0041, 0.9974, 0.9985])
#     c132 = np.asarray([1.0041, 0.9985, 0.9974])
#     c213 = np.asarray([0.9974, 1.0041, 0.9985])
#     c231 = np.asarray([0.9985, 1.0041, 0.9974])
#     c312 = np.asarray([0.9974, 0.9985, 1.0041])
#     c321 = np.asarray([0.9985, 0.9974, 1.0041])

#     # --- â‘£ ã‚¯ãƒ©ã‚¹ã�”ã�¨ã�®åŠ é‡�ãƒ­ã‚°å�ˆç®—ï¼ˆå¹¾ä½•å¹³å�‡ã�®é‡�ã�¿ä»˜ã��ç‰ˆï¼‰ ---
#     scores = []
#     eps = 1e-12
#     for a, b, c in zip(pred0, pred1, pred2):
#         if   a <= b <= c: _w = c123 * wts
#         elif a <= c <= b: _w = c132 * wts
#         elif b <= a <= c: _w = c213 * wts
#         elif b <= c <= a: _w = c231 * wts
#         elif c <= a <= b: _w = c312 * wts
#         else:             _w = c321 * wts  # c <= b <= a

#         s = _w[0]*np.log(a + eps) + _w[1]*np.log(b + eps) + _w[2]*np.log(c + eps)
#         scores.append(s)

#     scores = np.asarray(scores)
#     return dataset.le.classes_[int(np.argmax(scores))]

# # 3 + Japanese added

# import numpy as np

# def predict(sequence, demographics):

#     pred0 = predict1(sequence, demographics)[0]
#     pred1 = predict2(sequence, demographics)[0]
#     pred2 = predict3(sequence, demographics)[0]

#     # --- â‘  Temperature sharpening (slightly sharpen) ---
#     def _sharpen(p, gamma=1.10, eps=1e-12):
#         p = np.clip(p, eps, 1.0)
#         p = p ** gamma
#         return p / p.sum()

    
#     pred0 = _sharpen(pred0, 1.10)
#     pred1 = _sharpen(pred1, 1.10)
#     pred2 = _sharpen(pred2, 1.10)


#     # --- â‘¡ Fine-tune base weights with confidence (entropy) ---
#     def _entropy(p, eps=1e-12):
#         p = np.clip(p, eps, 1.0)
#         return -np.sum(p * np.log(p))


#     H0, H1, H2 = _entropy(pred0), _entropy(pred1), _entropy(pred2)
    
#     conf = np.exp(-np.array([H0, H1, H2]))          # Low H = high confidence â†’ large
#     conf = conf / conf.mean()                       # Normalized to mean 1
#     beta = 0.35                                     # Effectiveness (light CV between 0.2 and 0.5)
    
#     w_base = np.asarray([0.271,  0.347,  0.382])    # Fixed base
#     wts = (w_base * (conf ** beta))
#     wts_J1 = wts / wts.sum()                        # Total to 1

#     w_base2 = np.asarray([0.2705, 0.3474, 0.3821])  # Fixed base
#     wts2 = (w_base2 * (conf ** 0.374))
#     wts_J2 = wts2 / wts2.sum()                      # Total to 1
    

#     wts1, s,_,f = wts_J1,  0.9974, 1, 1.0041
#     #--------------------------------------------------
#     c1_123 = corr([f,s,_])
#     c1_132 = corr([f,_,s])
#     #----------------------------------------
#     c1_213 = corr([s,f,_])
#     c1_231 = corr([_,f,s])
#     #------------------------------
#     c1_312 = corr([s,_,f])
#     c1_321 = corr([_,s,f])
#     #----------------------
                                        
#     wts2, s,_,f = wts_J2,  0.9977, 1, 1.0038
#     #--------------------------------------------------
#     c2_123 = corr([f,s,_])
#     c2_132 = corr([f,_,s])
#     #----------------------------------------
#     c2_213 = corr([s,f,_])
#     c2_231 = corr([_,f,s])
#     #------------------------------
#     c2_312 = corr([s,_,f])
#     c2_321 = corr([_,s,f])
#     #----------------------

#     _r1,k1 = 7,1.000001
#     _r2,k2 = 7,1.000002

#     # --- â‘£ Weighted log sum by class (weighted version of geometric mean) ---

#     preds = []
#     eps = 1e-12
#     for _a,_b,_c in zip(pred0,pred1,pred2):
        
#         a1,b1,c1 = _a,_b,_c
#         _a1,_b1,_c1 = round(_a,_r1),round(_b,_r1),round(_c,_r1)
#         if   a1 <= b1 <= c1: _w1 = wts1 * c1_123
#         elif a1 <= c1 <= b1: _w1 = wts1 * c1_132
#         elif b1 <= a1 <= c1: _w1 = wts1 * c1_213
#         elif b1 <= c1 <= a1: _w1 = wts1 * c1_231
#         elif c1 <= a1 <= b1: _w1 = wts1 * c1_312
#         else:                _w1 = wts1 * c1_321 # c1 <= b1 <= a1:
            
#         _equ1 = equ(a1,b1,c1,k1)    
        
#         p1 = _w1[0]*np.log(a + eps) *_equ1[0] +\
#              _w1[1]*np.log(b + eps) *_equ1[1] +\
#              _w1[2]*np.log(c + eps) *_equ1[2]
#         # -----------------------------------------

#         a2,b2,c2 = _a,_b,_c
#         _a2,_b2,_c2 = round(_a,_r2),round(_b,_r2),round(_c,_r2)
#         if   c2 <= a2 <= b2: _w2 = wts2 * c2_312
#         elif c2 <= b2 <= a2: _wts2 = wts2 * c2_321
#         elif b2 <= a2 <= c2: _wts2 = wts2 * c2_213
#         elif b2 <= c2 <= a2: _wts2 = wts2 * c2_231
#         elif a2 <= b2 <= c2: _wts2 = wts2 * c2_123
#         else               : _wts2 = wts2 * c2_132
            
#         _equ2 = equ(a2,b2,c2,k2)
        
#         p2 = _w2[0]*np.log(a2 + eps) *_equ2[0] +\
#              _w2[1]*np.log(b2 + eps) *_equ2[1] +\
#              _w2[2]*np.log(c2 + eps) *_equ2[2]
#         # -----------------------------------------

#         p = (p1 + p2) / 2
        
#         preds.append(p)
        

#     avg_pred =  np.asarray(preds)
                
#     return dataset.le.classes_[avg_pred.argmax()]


# # 4 Japanese added code -> LB increase -> 0.850 -> 0.851

# # # Let's try adding h-blend to this 'common kitchen'.. 

# import copy

# def predict(sequence, demographics):
#     import numpy as np

#     # --- Probability of 3 models ---
#     pred0 = predict1(sequence, demographics)[0]
#     pred1 = predict2(sequence, demographics)[0]
#     pred2 = predict3(sequence, demographics)[0]

#     # --- Temperature sharpening (slightly sharpening) ---
#     def _sharpen(p, gamma=1.10, eps=1e-12):
#         p = np.clip(p, eps, 1.0)
#         p = p ** gamma
#         return p / p.sum()

#     pred0 = _sharpen(pred0, 1.10)
#     pred1 = _sharpen(pred1, 1.10)
#     pred2 = _sharpen(pred2, 1.10)

#     # --- Fine-tune base weights with confidence (entropy) ---
#     def _entropy(p, eps=1e-12):
#         p = np.clip(p, eps, 1.0)
#         return -np.sum(p * np.log(p))

#     H0, H1, H2 = _entropy(pred0), _entropy(pred1), _entropy(pred2)
#     conf = np.exp(-np.array([H0, H1, H2]))          # Low H = high confidence â†’ large
#     conf = conf / conf.mean()                       # Normalized to mean 1
#     # ---------------------------------------------------------------------------------------------
#     beta = 0.35                                     # Effectiveness (light CV between 0.2 and 0.5)
#     w_base = np.asarray([0.271, 0.347, 0.382])      # Fixed base
#     wts = (w_base * (conf ** beta))                 #                         wts_J1
#     wts_J1 = wts / wts.sum()                        # Total to 1
#     # ---------------------------------------------------------------------------------------------
#     beta = 0.374                                    # Effectiveness (light CV between 0.2 and 0.5)
#     w_base2 = np.asarray([0.2705, 0.3474, 0.3821])  # Fixed base
#     wts2 = (w_base2 * (conf ** beta))               #                         wts_J2
#     wts_J2 = wts2 / wts2.sum()                      # Total to 1


#     wts1, s,_,f = wts_J1,  0.9974, 1, 1.0041
#     #-------------------------------------------------
#     c1_123 = corr([f,s,_])
#     c1_132 = corr([f,_,s])
#     #---------------------------------------
#     c1_213 = corr([s,f,_])
#     c1_231 = corr([_,f,s])
#     #-----------------------------
#     c1_312 = corr([s,_,f])
#     c1_321 = corr([_,s,f])
#     #----------------------

#     wts2, s,_,f = wts_J2,  0.9977, 1, 1.0038
#     #-------------------------------------------------
#     c2_123 = corr([f,s,_])
#     c2_132 = corr([f,_,s])
#     #---------------------------------------
#     c2_213 = corr([s,f,_])
#     c2_231 = corr([_,f,s])
#     #-----------------------------
#     c2_312 = corr([s,_,f])
#     c2_321 = corr([_,s,f])
#     #----------------------

#     correct_wts_1, asc_desc_wts_1 = [+0.0021, -0.0007, -0.0014], [0.70, 0.30]
#     correct_wts_2, asc_desc_wts_2 = [+0.0027, -0.0009, -0.0018], [0.74, 0.26]
    
#     preds = []
#     eps = 1e-12
#     for _a, _b, _c in zip(pred0, pred1, pred2):
#         a1,b1,c1 = _a,_b,_c
#         if   a1 <= b1 <= c1: _w1 = wts1 * c1_123
#         elif a1 <= c1 <= b1: _w1 = wts1 * c1_132
#         elif b1 <= a1 <= c1: _w1 = wts1 * c1_213
#         elif b1 <= c1 <= a1: _w1 = wts1 * c1_231
#         elif c1 <= a1 <= b1: _w1 = wts1 * c1_312
#         else:                _w1 = wts1 * c1_321
        
#         # p1 = _w1[0]*np.log(a1 + eps) +\
#         #      _w1[1]*np.log(b1 + eps) +\
#         #      _w1[2]*np.log(c1 + eps)
#         # ----------------------------------------- h-blend:
#         l_abc = [
#             { 'wts':_w1[0], 'pred':a1, 'result':0, 'n':'pred0' },
#             { 'wts':_w1[1], 'pred':b1, 'result':0, 'n':'pred1' },
#             { 'wts':_w1[2], 'pred':c1, 'result':0, 'n':'pred2' }]
#         lps_asc  = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=False)
#         lps_desc = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=True)
#         for asc,correct_wt  in zip(lps_asc,  correct_wts_1): asc ['wts'] += correct_wt
#         for desc,correct_wt in zip(lps_desc, correct_wts_1): desc['wts'] += correct_wt
#         for asc  in lps_asc:  asc ['result'] = np.log(asc ['pred'] +eps) * asc ['wts']
#         for desc in lps_desc: desc['result'] = np.log(desc['pred'] +eps) * desc['wts']
#         result_asc  = sum([asc ['result'] for asc in lps_asc])
#         result_desc = sum([desc['result'] for asc in lps_desc])
#         result_1 =\
#             result_asc  * asc_desc_wts_1[0] + \
#             result_desc * asc_desc_wts_1[1]
#         # =========================================

#         a2,b2,c2 = _a,_b,_c
#         if   c2 <= a2 <= b2: _w2 = wts2 * c2_312
#         elif c2 <= b2 <= a2: _w2 = wts2 * c2_321
#         elif b2 <= a2 <= c2: _w2 = wts2 * c2_213
#         elif b2 <= c2 <= a2: _w2 = wts2 * c2_231
#         elif a2 <= b2 <= c2: _w2 = wts2 * c2_123
#         else               : _w2 = wts2 * c2_132
        
#         # p2 = _w2[0]*np.log(a2 + eps) +\
#         #      _w2[1]*np.log(b2 + eps) +\
#         #      _w2[2]*np.log(c2 + eps)
#         # ----------------------------------------- h-blend
#         l_abc = [
#             { 'wts':_w2[0], 'pred':a2, 'result':0, 'n':'pred0' },
#             { 'wts':_w2[1], 'pred':b2, 'result':0, 'n':'pred1' },
#             { 'wts':_w2[2], 'pred':c2, 'result':0, 'n':'pred2' }]
#         lps_asc  = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=False)
#         lps_desc = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=True)
#         for asc,correct_wt  in zip(lps_asc,  correct_wts_2): asc ['wts'] += correct_wt
#         for desc,correct_wt in zip(lps_desc, correct_wts_2): desc['wts'] += correct_wt
#         for asc  in lps_asc:  asc ['result'] = np.log(asc ['pred'] +eps) * asc ['wts']
#         for desc in lps_desc: desc['result'] = np.log(desc['pred'] +eps) * desc['wts']
#         result_asc  = sum([asc ['result'] for asc in lps_asc])
#         result_desc = sum([desc['result'] for asc in lps_desc])
#         result_2 =\
#             result_asc  * asc_desc_wts_2[0] + \
#             result_desc * asc_desc_wts_2[1]
#         # =========================================

#         result = (result_1 + result_2) / 2
        
#         preds.append(result)

#     avg_pred = np.asarray(preds)

#     return dataset.le.classes_[avg_pred.argmax()]


# 5
'''
import copy

def predict(sequence, demographics):
    
    pred0 = predict1(sequence, demographics)[0]
    pred1 = predict2(sequence, demographics)[0]
    pred2 = predict3(sequence, demographics)[0]

    m_w,da_w,c_w = [0.271, 0.347, 0.382], [0.70, 0.30], [+0.0021,-0.0007,-0.0014]
    
    s,_,f = 0.9974, 1, 1.0041
    #-------------------------
    c1_123 = corr([f,s,_])
    c1_132 = corr([f,_,s])
    #------------------------
    c1_213 = corr([s,f,_])
    c1_231 = corr([_,f,s])
    #-----------------------
    c1_312 = corr([s,_,f])
    c1_321 = corr([_,s,f])
    #----------------------
    
    wts1, preds = np.asarray(m_w), []
    
    for _a, _b, _c in zip(pred0, pred1, pred2):
        a1,b1,c1 = _a,_b,_c
        if   a1 <= b1 <= c1: _w1 = wts1 * c1_123
        elif a1 <= c1 <= b1: _w1 = wts1 * c1_132
        elif b1 <= a1 <= c1: _w1 = wts1 * c1_213
        elif b1 <= c1 <= a1: _w1 = wts1 * c1_231
        elif c1 <= a1 <= b1: _w1 = wts1 * c1_312
        elif c1 <= b1 <= a1: _w1 = wts1 * c1_321

        l_abc = [
            { 'wts':_w1[0], 'pred':a1, 'res':0 },
            { 'wts':_w1[1], 'pred':b1, 'res':0 },
            { 'wts':_w1[2], 'pred':c1, 'res':0 },
        ]
        l_asc  = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=False)
        l_desc = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=True)
        
        for asc, c_wts in zip(l_asc, c_w): asc ['res'] = asc ['pred'] * (asc ['wts'] +c_wts)
        for desc,c_wts in zip(l_desc,c_w): desc['res'] = desc['pred'] * (desc['wts'] +c_wts)

        result_asc  = sum([asc ['res'] for asc in l_asc])
        result_desc = sum([desc['res'] for asc in l_desc])

        result = result_asc * da_w[0] + da_w[1] * result_desc
 
        preds.append(result)
    avg_pred = np.asarray(preds)

    return dataset.le.classes_[avg_pred.argmax()]
'''
import numpy as np

def predict(sequence, demographics):

    import copy
    
    pred0 = predict1(sequence, demographics)[0]
    pred1 = predict2(sequence, demographics)[0]
    pred2 = predict3(sequence, demographics)[0]
    
    preds = []

    # ------------------------------------------------------------------- v.21
                                          
    main_wts   = np.asarray(  [0.271, 0.347, 0.382]                   )
    
    correct_wts =             [+0.0021, -0.0007, -0.0014]
    asc_desc_wts =                                         [0.70, 0.30]

    # =================================================================== Lb=?


    for a,b,c in zip(pred0,pred1,pred2):
        
        l_abc = [
            { 'wts':main_wts[0], 'pred':a, 'n':'p0', 'result':0 },
            { 'wts':main_wts[1], 'pred':b, 'n':'p1', 'result':0 },
            { 'wts':main_wts[2], 'pred':c, 'n':'p2', 'result':0 }]

        lps_asc  = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=False)
        lps_desc = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=True)
        
        for asc,correct_wt  in zip(lps_asc,  correct_wts): asc ['wts'] += correct_wt
        for desc,correct_wt in zip(lps_desc, correct_wts): desc['wts'] += correct_wt
        for asc  in lps_asc:  asc ['result'] = asc ['pred'] * asc ['wts']
        for desc in lps_desc: desc['result'] = desc['pred'] * desc['wts']

        result_asc  = sum([asc ['result'] for asc in lps_asc])
        result_desc = sum([desc['result'] for asc in lps_desc])

        result =\
            result_asc  * asc_desc_wts[0] + \
            result_desc * asc_desc_wts[1]
        
        preds.append(result)
        
    avg_pred =  np.asarray(preds)
                
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


if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print(pd.read_parquet("submission.parquet"))

