import os, glob, pprint, random, numpy as np
CFG = {
    "INPUT_ROOT": "/kaggle/input",
    "DATASET_DIR": "/kaggle/input/cmi-detect-behavior-with-sensor-data",
    "OOF_DIR": "/kaggle/input/cmi-oof",
    "CHECKPOINT_GLOBS": [
        "/kaggle/input/**/best*.pt",
        "/kaggle/input/**/model*.pt",
        "/kaggle/input/**/best*.h5",
        "/kaggle/input/**/model*.h5",
    ],
    "USE_FP16": True,
    "DETERMINISTIC": True,
    "TTA_ENABLE": True,
    "V3_ENABLE_CONTEXT_NORM": True,
    "V3_WINSOR_P": 0.01,
    "V3_ARM_BIN_THRESHOLD": 52.0,
    "V3_USE_GROUP_TS": True,
    # Ø¢Ø¤Ù¹ Ù¾Ù¹ Ú©Û’ Ù„ÛŒÛ’
    "SUBMISSION_NAME": "submission.csv",
}
pprint.pprint(CFG)

# Ø°ÛŒÙ„ Ù…ÛŒÚº Ø§Ø³ Ú©ÙˆÚˆ Ú©Û’ Ù„ÛŒÛ’ Ù…Ø·Ù„ÙˆØ¨Û� ÚˆÛŒÙ¹Ø§ Ø³ÛŒÙ¹
required = [
    f'{CFG["DATASET_DIR"]}/test.csv',
    f'{CFG["DATASET_DIR"]}/test_demographics.csv',
]
print("\n== Ø¶Ø±ÙˆØ±ÛŒ Û�Û’Û” ==")
for p in required:
    print(p, "ğŸ˜ƒ" if os.path.exists(p) else "ğŸ˜­")

# Ø§Ø®ØªÛŒØ§Ø±ÛŒ
optional = [
    f'{CFG["OOF_DIR"]}/oof_predict1.csv',
    f'{CFG["OOF_DIR"]}/oof_predict2.csv',
    f'{CFG["OOF_DIR"]}/oof_predict3.csv',
    f'{CFG["DATASET_DIR"]}/train_demographics.csv',
    f'{CFG["DATASET_DIR"]}/train_labels.csv',
]
print("\n== Ø§Ø®ØªÛŒØ§Ø±ÛŒ ==")
for p in optional:
    print(p, "ğŸ˜ƒ" if os.path.exists(p) else "ğŸ˜­")

print("\n== Ú†ÛŒÚ© Ù¾ÙˆØ§Ø¦Ù†Ù¹ Ú©Ø§ Ø§Ù†Ø¯Ø§Ø²Û� ==")
hits = []
for pat in CFG["CHECKPOINT_GLOBS"]:
    found = glob.glob(pat, recursive=True)[:6]
    if found:
        print(pat, "->", len(found), "found; sample:", found[:3])
        hits.extend(found)
if not hits:
    print("No checkpoints found under /kaggle/input/** (attach your weights via 'Add data').")

# ØªØ¹ÛŒÙ† Ø§ÙˆØ± Ø¯Ø±Ø³ØªÚ¯ÛŒ
try:
    import torch
    if CFG["DETERMINISTIC"]:
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
        except Exception:
            pass
    if CFG["USE_FP16"]:
        print("[INFO] FP16/BF16 ØªØ§Ø¦ÛŒØ¯ Û�ÙˆÙ†Û’ Ù¾Ø± Ø§Ù†Ø¯Ø§Ø²Û� Ú©ÛŒ Ø§Ø¬Ø§Ø²Øª Ø¯ÛŒ Ø¬Ø§ØªÛŒ Û�Û’Û”.")
except Exception as e:
    print("[WARN] Ù¹Ø§Ø±Ú† Ø³ÛŒÙ¹ Ø§Ù¾:", e)

# Ù¾Ø±Ø§Ù†Û’ Ø®Ù„ÛŒÙˆÚº Ú©Û’ Ù„ÛŒÛ’ Ø¨Ø±Ø¬ Ù¹ÙˆÚ¯Ù„
TTA_ENABLE = bool(CFG.get("TTA_ENABLE", True))
V3_ENABLE_CONTEXT_NORM = bool(CFG.get("V3_ENABLE_CONTEXT_NORM", True))
V3_WINSOR_P = float(CFG.get("V3_WINSOR_P", 0.01))
V3_ARM_BIN_THRESHOLD = float(CFG.get("V3_ARM_BIN_THRESHOLD", 52.0))
V3_USE_GROUP_TS = bool(CFG.get("V3_USE_GROUP_TS", True))


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
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import backend as K
import tensorflow as tf
import polars as pl
from sklearn.model_selection import StratifiedGroupKFold
from scipy.spatial.transform import Rotation as R

# Ø±ÛŒ Ù¾Ø±ÙˆÚˆÛŒÙˆØ³ Ø§ÛŒØ¨Ù„Ù¹ÛŒ Ú©Û’ Ù„Ø¦Û’ Ø³ÛŒÚˆ Ø³ÛŒÙ¹ Ú©Ø±Ù†Ø§

def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed_everything(seed=42)

# (Ù…Ù‚Ø§Ø¨Ù„Û� Ù…ÛŒÙ¹Ø±Ú© ØµØ±Ù� Ù¹Ø±ÛŒÙ†Ù†Ú¯ Ú©Û’ ÙˆÙ‚Øª Ø¯Ø±Ø¢Ù…Ø¯ Ú©ÛŒØ§ Ø¬Ø§Ø¦Û’ Ú¯Ø§Û”)
TRAIN = False           # Ø¬Ø¨ Ø¢Ù¾ ØªØ±Ø¨ÛŒØª Ú©Ø±Ù†Ø§ Ú†Ø§Û�ØªÛ’ Û�ÛŒÚº ØªÙˆ Ø¯Ø±Ø³Øª Ù¾Ø± Ø³ÛŒÙ¹ Ú©Ø±ÛŒÚºÛ”
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/cmi-d-111")
EXPORT_DIR = Path("./")
BATCH_SIZE = 64  
PAD_PERCENTILE = 95
LR_INIT = 5e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 160
PATIENCE = 40

print("Ø¯Ø±Ø¢Ù…Ø¯Ø§Øª ØªÛŒØ§Ø± Û�ÛŒÚºÛ” Â· tensorflow", tf.__version__)

# Ù¹ÛŒÙ†Ø³Ø± Û�ÛŒØ±Ø§ Ù¾Ú¾ÛŒØ±ÛŒ
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

# CNN Block with SE
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

# Ù¹Ø§Ø¦Ù… Ø³ÛŒØ±ÛŒØ² Ú©ÛŒ ØªØ±ØªÛŒØ¨ Ú©Ùˆ Ù…Ø¹Ù…ÙˆÙ„ Ø¨Ù†Ø§ØªØ§ Ø§ÙˆØ± ØµØ§Ù� Ú©Ø±ØªØ§ Û�Û’Û” 

def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')


# Ù†ÛŒÙˆØ±Ù„ Ù†ÛŒÙ¹ ÙˆØ±Ú© Ú©Ùˆ Ø¨Ø§Ù‚Ø§Ø¹Ø¯Û� Ø¨Ù†Ø§Ù†Û’ Ú©Û’ Ù„ÛŒÛ’ ÚˆÛŒÙ¹Ø§ Ø¢Ø±Ú¯ÙˆÙ…ÛŒÙ†Ù¹ÛŒØ´Ù† Ú©Ùˆ Ù…Ú©Ø³ Ø§Ù¾ Ú©Ø±ÛŒÚºÛ” 
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

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
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
            angular_dist[i] = 0
            pass
            
    return angular_dist

def build_two_branch_model(pad_len, imu_dim, tof_dim, n_classes, wd=1e-4):
    inp = Input(shape=(pad_len, imu_dim+tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    # IMU Ú¯Û�Ø±ÛŒ Ø´Ø§Ø®
    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)
    
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

custom_objs = {
    'time_sum': time_sum, 'squeeze_last_axis': squeeze_last_axis, 'expand_last_axis': expand_last_axis,
    'se_block': se_block, 'residual_se_cnn_block': residual_se_cnn_block, 'attention_layer': attention_layer,
}
# Load any Models
PRETRAINED_DIR = Path("/kaggle/input/cmi-d-111")
print("INFERENCE MODE 1,2 â€“ loading artefacts from", PRETRAINED_DIR)
final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

models1 = []
print(f"  Loading models for ensemble inference...")
for fold in range(10):
    model_path = f"{PRETRAINED_DIR}/D-111_{fold}.h5"
    print(">>> Ù„ÙˆÚˆ Ù…Ø§ÚˆÙ„ >>>",model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models1.append(model)
print("-"*60)

for fold in range(10):
    model_path = f"{PRETRAINED_DIR}/v0629_{fold}.h5"
    print(">>> Ù„ÙˆÚˆ Ù…Ø§ÚˆÙ„ >>>",model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models1.append(model)
print("-"*60)
print(f"[INFO]NumUseModels:{len(models1)}")


PRETRAINED_DIR = Path("/kaggle/input/n-splits-10")
print("INFERENCE MODE 3 â€“ loading artefacts from", PRETRAINED_DIR)
final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len        = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
scaler         = joblib.load(PRETRAINED_DIR / "scaler.pkl")
gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)
for fold in range(10):
    model_path = f"{PRETRAINED_DIR}/gesture_model_fold_{fold}.h5"
    print(">>> Ù„ÙˆÚˆ Ù…Ø§ÚˆÙ„ >>>",model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models1.append(model)
print("-"*60)
print(f"[INFO]NumUseModels:{len(models1)}")

for fold in range(10):
    MODEL_DIR = "/kaggle/input/cmi-data-tensorflow-train"
    
    model_path = f"{MODEL_DIR}/gesture_model_fold_{fold}.h5"
    print(">>> Ù„ÙˆÚˆ Ù…Ø§ÚˆÙ„ >>>",model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models1.append(model)
print("-"*70)
print(f"[INFO]NumUseModels:{len(models1)}")


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
    
    all_preds = [model.predict(pad_input, verbose=0)[0] for model in models1]
    avg_pred = np.median(all_preds, axis=0) 
    return avg_pred


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
            angular_dist[i] = 0
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
        imu_mask, thm_mask, tof_mask = [], [], []
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
        all_indices = np.arange(len(self.full_dataset))
        self.n_folds = n_folds
        self.skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        self.folds = list(self.skf.split(all_indices, np.array(self.full_dataset.dataset_indices)))
    
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
        
        print("\Cross-validation fold statistics:")
        for fold_idx in range(self.n_folds):
            train_fold, valid_fold = self.get_fold_datasets(fold_idx)
            train_counts = get_label_counts(train_fold)
            valid_counts = get_label_counts(valid_fold)
                
            print(f"\nFold {fold_idx + 1}:")
            print(f"{'Category':<50} {'Training set':<10} {'Validation set':<10}")
            for name in self.class_names:
                print(f"{name:<50} {train_counts[name]:<10} {valid_counts[name]:<10}")


class SEBlock(nn.Module):
    def __init__(self, channels, reduction = 8):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = F.adaptive_avg_pool1d(x, 1).squeeze(-1)      
        se = F.relu(self.fc1(se), inplace=True)          
        se = self.sigmoid(self.fc2(se)).unsqueeze(-1)   
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
        identity = self.shortcut(x)              
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)                       
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

root_dir = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data") # dataset path
universe_csv_path = Path("/kaggle/input/cmi-precompute/pytorch/all/1/tof-1_raw.csv") # dataset in CSV form

deterministic = kagglehub.package_import('wasupandceacar/deterministic').deterministic
deterministic.init_all(seed)
def init_dataset():
    dataset_config = {
        "percent": 95,
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

models2 = list()
for model_dict in model_dicts:
    model_function = model_dict["model_function"]
    model_args = model_dict["model_args"]
    model_path = model_dict["model_path"]
    model = model_function(**model_args).to(CUDA0)
    state_dict = {k.replace("_orig_mod.", ""): v for k,v in torch.load(model_path).items()}
    model.load_state_dict(state_dict)
    model = model.eval()
    models2.append(model)


metric_package = kagglehub.package_import('wasupandceacar/cmi-metric')

metric = metric_package.Metric()
imu_only_metric = metric_package.Metric()

def to_cuda(*tensors):
    return [tensor.to(CUDA0) for tensor in tensors]

def predict_fold(model, imu, thm, tof):
    pred = model(imu, thm, tof)
    return pred

def valid(model, valid_bar):
    with torch.no_grad():
        for imu, thm, tof, y in valid_bar:
            imu, thm, tof, y = to_cuda(imu, thm, tof, y)
            with autocast(device_type='cuda', dtype=torch.bfloat16): 
                logits = predict_fold(model, imu, thm, tof)
            metric.add(dataset.le.classes_[y.argmax(dim=1).cpu()], dataset.le.classes_[logits.argmax(dim=1).cpu()])
            _, thm, tof = dataset.full_dataset.get_scaled_nan_tensors(imu, thm, tof)
            with autocast(device_type='cuda', dtype=torch.bfloat16): 
                logits = model(imu, thm, tof)
            imu_only_metric.add(dataset.le.classes_[y.argmax(dim=1).cpu()], dataset.le.classes_[logits.argmax(dim=1).cpu()])

def avg_predict(models, imu, thm, tof):
    outputs = []
    with autocast(device_type='cuda'):
        for model in models:
            logits = model(imu, thm, tof)
        outputs.append(logits)
    return torch.mean(torch.stack(outputs), dim=0)

def predict2(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    imu, thm, tof = dataset.full_dataset.inference_process(sequence)
    with torch.no_grad():
        imu, thm, tof = to_cuda(imu, thm, tof)
        logits = avg_predict(models2, imu, thm, tof)
        probabilities = F.softmax(logits, dim=1).cpu().numpy()
    return probabilities


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

# Evaluation Certificate/Study Time Certificate
try:
    from cmi_2025_metric_copy_for_import import CompetitionMetric
except ImportError:
    CompetitionMetric = None
    print("CompetitionMetric could not be imported.")

def seed_everything(seed=42):
    """
    A function to uniformly set the random number seed for the execution environment.
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

seed_everything(seed=42)
warnings.filterwarnings("ignore")

TRAIN = False
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
YOUR_MODELS_DIR = Path("/kaggle/input/cmi-data-gated-gru")
PUBLIC_TF_MODEL_DIR = Path("/kaggle/input/lb-0-78-quaternions-tf-bilstm-gru-attention")
PUBLIC_PT_MODEL_DIR = Path("/kaggle/input/cmi3-models-p")
EXPORT_DIR = Path("./") # Where to store trained models
BATCH_SIZE = 64        
PAD_PERCENTILE = 95      
LR_INIT = 4e-4           
WD = 3e-3                
MIXUP_ALPHA = 0.4        
EPOCHS = 360             
PATIENCE = 50            # Patience for EarlyStopping on "50"
N_SPLITS = 10             
MASKING_PROB = 0.25      
GATE_LOSS_WEIGHT = 0.2   

print(f"Library import complete")
print(f"ğŸ˜‡ TensorFlow: {tf.__version__}")
print(f"ğŸ˜‡ PyTorch: {torch.__version__}")
print(f"ğŸ˜‡ TRAIN: {TRAIN}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    PyTorch
    Corrected to the original correct definition to match the published model weights.
    """
    def __init__(self, fs=100., add_quaternion=False):
        super().__init__()
        self.fs = fs
        self.add_quaternion = add_quaternion
        k = 15
        self.lpf = nn.Conv1d(6, 6, kernel_size=k, padding=k//2,
                                 groups=6, bias=False)
        nn.init.kaiming_uniform_(self.lpf.weight, a=math.sqrt(5))
        self.lpf_acc  = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
        self.lpf_gyro = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)

    def forward(self, imu):
        acc  = imu[:, 0:3, :]
        gyro = imu[:, 3:6, :]
        acc_mag  = torch.norm(acc,  dim=1, keepdim=True)
        gyro_mag = torch.norm(gyro, dim=1, keepdim=True)
        jerk = F.pad(acc[:, :, 1:] - acc[:, :, :-1], (1,0))
        gyro_delta = F.pad(gyro[:, :, 1:] - gyro[:, :, :-1], (1,0))
        acc_pow  = acc ** 2
        gyro_pow = gyro ** 2
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
    A class with the architecture for loading publicly available PyTorch models.
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
        self.bilstm = nn.LSTM(256, 128, bidirectional=True, batch_first=True) # GRU< >LSTM
        self.lstm_dropout = nn.Dropout(dropouts[4])
        self.attention = AttentionLayer(256) # bidirectional at least 128*2
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
    
# Feature Engineering Functions
def remove_gravity_from_acc3(acc_data, rot_data):
    """Remove the gravity component from the acceleration data"""
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
    """Calculating angular velocity from quaternions"""
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
    """Calculating angular distance from a quaternion"""
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
    """Squeeze Excitation"""
    ch = x.shape[-1]
    se = GlobalAveragePooling1D()(x)
    se = Dense(ch // reduction, activation='relu')(se)
    se = Dense(ch, activation='sigmoid')(se)
    se = Reshape((1, ch))(se)
    return Multiply()([x, se])

def residual_se_cnn_block(x, filters, kernel_size, pool_size=2, drop=0.3, wd=1e-4):
    """Residual-CNN"""
    shortcut = x
    # 2 Conv1D
    for _ in range(2):
        x = Conv1D(filters, kernel_size, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
    x = se_block(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same', use_bias=False, kernel_regularizer=l2(wd))(shortcut)
        shortcut = BatchNormalization()(shortcut)
    x = add([x, shortcut])
    x = Activation('relu')(x)
    x = MaxPooling1D(pool_size)(x)
    x = Dropout(drop)(x)
    return x

def attention_layer(inputs):
    """attention layer"""
    score = Dense(1, activation='tanh')(inputs)
    score = Lambda(squeeze_last_axis)(score)
    weights = Activation('softmax')(score)
    weights = Lambda(expand_last_axis)(weights)
    context = Multiply()([inputs, weights])
    context = Lambda(time_sum)(context)
    return context
class GatedMixupGenerator(Sequence):
    """Mixup and a data generator that applies sensor masking"""
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
    Two-Branch LSTM GRU
    """
    inp = Input(shape=(pad_len, imu_dim + tof_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:])(inp)

    x1 = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x1 = residual_se_cnn_block(x1, 128, 5, drop=0.1, wd=wd)

    x2_base = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x2_base = BatchNormalization()(x2_base); x2_base = Activation('relu')(x2_base)
    x2_base = MaxPooling1D(2)(x2_base); x2_base = Dropout(0.2)(x2_base)
    x2_base = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x2_base)
    x2_base = BatchNormalization()(x2_base); x2_base = Activation('relu')(x2_base)
    x2_base = MaxPooling1D(2)(x2_base); x2_base = Dropout(0.2)(x2_base)

    gate_input = GlobalAveragePooling1D()(tof)
    gate_input = Dense(16, activation='relu')(gate_input)
    gate = Dense(1, activation='sigmoid', name='tof_gate')(gate_input)
    x2 = Multiply()([x2_base, gate])
    
    merged = Concatenate()([x1, x2])
    # LSTM > GRU
    x = Bidirectional(GRU(256, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    x = Dropout(0.45)(x)
    x = attention_layer(x)
    for units, drop in [(512, 0.5), (256, 0.4), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(drop)(x)

    out = Dense(n_classes, activation='softmax', name='main_output', kernel_regularizer=l2(wd))(x)

    return Model(inputs=inp, outputs=[out, gate])

print("Enter inference mode â€“ Load the trained model and artifacts")

print("5-Fold Gated GRU")
final_feature_cols_A = np.load(YOUR_MODELS_DIR / "final_feature_cols.npy", allow_pickle=True).tolist()
pad_len_A = int(np.load(YOUR_MODELS_DIR / "sequence_maxlen.npy"))
scaler_A = joblib.load(YOUR_MODELS_DIR / "scaler.pkl")
gesture_classes = np.load(YOUR_MODELS_DIR / "gesture_classes.npy", allow_pickle=True)
custom_objs_A = {'time_sum': time_sum, 'squeeze_last_axis': squeeze_last_axis, 'expand_last_axis': expand_last_axis,
                 'se_block': se_block, 'residual_se_cnn_block': residual_se_cnn_block, 'attention_layer': attention_layer}
models_A = [load_model(YOUR_MODELS_DIR / f"final_model_fold_{f}.h5", compile=False, custom_objects=custom_objs_A) for f in range(N_SPLITS)]

final_feature_cols_B = np.load(PUBLIC_TF_MODEL_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len_B = int(np.load(PUBLIC_TF_MODEL_DIR / "sequence_maxlen.npy"))
scaler_B = joblib.load(PUBLIC_TF_MODEL_DIR / "scaler.pkl")
custom_objs_B = custom_objs_A
model_B = load_model(PUBLIC_TF_MODEL_DIR / "gesture_two_branch_mixup.h5", compile=False, custom_objects=custom_objs_B)

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
print(f"  > {len(pt_models)}models loaded successfully.")


def enumerate_weights(i):
    import random
    
    # Scheme 1
    weights00 = {'A': 0.5285, 'B': 0.1770, 'C': 0.2945}
    weights01 = {'A': 0.5290, 'B': 0.1780, 'C': 0.2930}
    weights02 = {'A': 0.5295, 'B': 0.1785, 'C': 0.2920}
    weights03 = {'A': 0.5305, 'B': 0.1790, 'C': 0.2905}
    weights04 = {'A': 0.5310, 'B': 0.1795, 'C': 0.2895}

    # Scheme 2
    weights05 = {'A': 0.533, 'B': 0.174, 'C': 0.293}
    weights06 = {'A': 0.534, 'B': 0.176, 'C': 0.290}
    weights07 = {'A': 0.535, 'B': 0.177, 'C': 0.288}
    weights08 = {'A': 0.536, 'B': 0.178, 'C': 0.286}
    weights09 = {'A': 0.537, 'B': 0.179, 'C': 0.284}

    # Scheme 3
    weights10 = {'A': 0.534, 'B': 0.158,  'C': 0.308}
    weights11 = {'A': 0.534, 'B': 0.1586, 'C': 0.3074}
    weights12 = {'A': 0.535, 'B': 0.159,  'C': 0.386}
    weights13 = {'A': 0.536, 'B': 0.1595, 'C': 0.3045}
    weights14 = {'A': 0.537, 'B': 0.160,  'C': 0.303}

    # Scheme 4
    weights15 = {'A': 0.527, 'B': 0.185,  'C': 0.288}
    weights16 = {'A': 0.526, 'B': 0.190,  'C': 0.284}
    weights17 = {'A': 0.525, 'B': 0.195,  'C': 0.280}
    weights18 = {'A': 0.524, 'B': 0.200,  'C': 0.276}
    weights19 = {'A': 0.523, 'B': 0.205,  'C': 0.272}

    import random
    weights_1 = random.choice([weights00,weights01,weights02,weights03,weights04])
    weights_2 = random.choice([weights05,weights06,weights07,weights08,weights09])
    weights_3 = random.choice([weights10,weights11,weights12,weights13,weights14])
    weights_4 = random.choice([weights15,weights16,weights17,weights18,weights19])
    if i == 1: return weights_1
    if i == 2: return weights_2
    if i == 3: return weights_3
    if i == 4: return weights_4

    return {'A': 0.53, 'B': 0.18, 'C': 0.29}
    
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
    mat_A = df_seq_A[final_feature_cols_A].ffill().bfill().fillna(0).values.astype('float32')
    mat_A = scaler_A.transform(mat_A)
    pad_input_A = pad_sequences([mat_A], maxlen=pad_len_A, padding='post', dtype='float32')
    preds_A_folds = [model.predict(pad_input_A, verbose=0)[0] for model in models_A]

    df_seq_B = df_seq_orig.copy()
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
    mat_B = df_seq_B[final_feature_cols_B].ffill().bfill().fillna(0).values.astype('float32')
    mat_B = scaler_B.transform(mat_B)
    pad_input_B = pad_sequences([mat_B], maxlen=pad_len_B, padding='post', dtype='float32')
    df_seq_C = df_seq_orig.copy()
    mat_C = df_seq_C[final_feature_cols_C].ffill().bfill().fillna(0).values.astype('float32')
    mat_C = scaler_C.transform(mat_C)
    pad_input_C = pad_sequences_torch3([mat_C], maxlen=pad_len_C, padding='pre', truncating='pre')
   
    with torch.no_grad():
        pt_input = torch.from_numpy(pad_input_C).to(device)
        preds_C_folds = [model(pt_input) for model in pt_models]

    avg_pred_A = np.mean(preds_A_folds, axis=0)
    avg_pred_C_logits = torch.median(torch.stack(preds_C_folds), dim=0).values
    avg_pred_C = torch.softmax(avg_pred_C_logits, dim=1).cpu().numpy()
    pred_B = model_B.predict(pad_input_B, verbose=0)
    if isinstance(pred_B, list): pred_B = pred_B[0]
    weights = enumerate_weights(2)
    final_pred_proba = (weights['A'] * avg_pred_A + weights['B'] * pred_B + weights['C'] * avg_pred_C)

    return final_pred_proba


import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

EPS = 1e-12

def _prob_to_logits(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    if p.ndim == 1:
        p = p[None, :]
    p /= p.sum(axis=1, keepdims=True)
    return np.log(p)

def _logits_to_prob(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def load_oof(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception:
            pass
    return None

CAND_OOF = [
    "/kaggle/input/cmi-oof/oof_predict1.csv",
    "/kaggle/input/cmi-oof/oof_predict2.csv",
    "/kaggle/input/cmi-oof/oof_predict3.csv",
]
oof_list = [load_oof(p) for p in CAND_OOF]
oof_list = [df for df in oof_list if df is not None]

temperatures = {"p1": 1.0, "p2": 1.0, "p3": 1.0}
if len(oof_list) == 3:
    for key, df in zip(["p1","p2","p3"], oof_list):
        if "y" in df.columns:
            y = df["y"].values
            prob_cols = [c for c in df.columns if c.startswith("class_")]
            P = df[prob_cols].values
            logits = _prob_to_logits(P)
            def nll(T):
                z = logits / T
                q = _logits_to_prob(z)
                q = np.clip(q, EPS, 1.0-EPS)
                return -np.mean(np.log(q[np.arange(len(y)), y]))
            Ts = np.linspace(0.5, 5.0, 46)
            vals = [nll(T) for T in Ts]
            temperatures[key] = float(Ts[int(np.argmin(vals))])

meta_model = None
if len(oof_list) == 3 and all("y" in df.columns for df in oof_list):
    y = oof_list[0]["y"].values
    feats = []
    for key, df in zip(["p1","p2","p3"], oof_list):
        prob_cols = [c for c in df.columns if c.startswith("class_")]
        P = df[prob_cols].values
        z = _prob_to_logits(P) / temperatures[key]
        feats.append(z)
    X_meta = np.concatenate(feats, axis=1)
    try:
        meta_model = LogisticRegression(max_iter=200, multi_class="multinomial")
        meta_model.fit(X_meta, y)
    except Exception:
        meta_model = None

demo_prior_model = None
demo_cols_cat, demo_cols_num = [], []
try:
    demo_train_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
    label_train_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_labels.csv"
    if os.path.exists(demo_train_path) and os.path.exists(label_train_path):
        df_demo = pd.read_csv(demo_train_path)
        df_y = pd.read_csv(label_train_path)
        df = df_demo.merge(df_y, on="sequence_id", how="inner")
        for c in df.columns:
            if c in ["sequence_id","subject_id","label"]:
                continue
            if df[c].dtype == "object":
                demo_cols_cat.append(c)
            else:
                demo_cols_num.append(c)
        if len(demo_cols_cat)+len(demo_cols_num) > 0:
            pre = ColumnTransformer([
                ("cat", OneHotEncoder(handle_unknown="ignore"), demo_cols_cat),
                ("num", StandardScaler(with_mean=True, with_std=True), demo_cols_num),
            ])
            demo_prior_model = Pipeline([("pre", pre),
                                         ("clf", LogisticRegression(max_iter=200, multi_class="multinomial"))])
            demo_prior_model.fit(df[demo_cols_cat+demo_cols_num], df["label"])
except Exception:
    demo_prior_model = None

def apply_demographics_prior(proba, demographics_df):
    if demo_prior_model is None or demographics_df is None:
        return proba
    try:
        if hasattr(demographics_df, "to_pandas"):
            dp = demographics_df.to_pandas()
        else:
            dp = demographics_df
        import pandas as _pd
        if isinstance(dp, _pd.DataFrame):
            x = dp[demo_cols_cat+demo_cols_num].copy()
            for c in demo_cols_num:
                if c in x.columns:
                    x[c] = x[c].astype(float)
            if len(x) > 1:
                x_num = x[demo_cols_num].mean(axis=0) if demo_cols_num else _pd.Series([], dtype=float)
                x_cat = x[demo_cols_cat].iloc[[0]] if demo_cols_cat else _pd.DataFrame(index=[0])
                x = _pd.concat([x_cat.reset_index(drop=True), _pd.DataFrame([x_num]).reset_index(drop=True)], axis=1)
            prior = demo_prior_model.predict_proba(x)
            prior = np.clip(prior, EPS, 1.0)
            prior = prior / prior.sum(axis=1, keepdims=True)
            out = proba * prior
            out = out / out.sum(axis=1, keepdims=True)
            return out
    except Exception:
        pass
    return proba

def predict(sequence, demographics):
    p1 = predict1(sequence, demographics)[0]
    p2 = predict2(sequence, demographics)[0]
    p3 = predict3(sequence, demographics)[0]

    z1 = _prob_to_logits(p1[None, :]) / temperatures["p1"]
    z2 = _prob_to_logits(p2[None, :]) / temperatures["p2"]
    z3 = _prob_to_logits(p3[None, :]) / temperatures["p3"]

    if meta_model is not None:
        Xq = np.concatenate([z1, z2, z3], axis=1)
        pq = meta_model.predict_proba(Xq)
    else:
        w = np.array([0.30, 0.35, 0.35], dtype=float)
        pq = _logits_to_prob(w[0]*z1 + w[1]*z2 + w[2]*z3)

    pq2 = apply_demographics_prior(pq, demographics)

    try:
        cls_idx = int(np.argmax(pq2, axis=1)[0])
        return dataset.le.classes_[cls_idx]
    except Exception:
        return int(np.argmax(pq2, axis=1)[0])

print("Enhanced predict with scaling, stacking, and demographics active.")


import numpy as np

def predict(sequence, demographics):
    import copy
    pred0 = predict1(sequence, demographics)[0]
    pred1 = predict2(sequence, demographics)[0]
    pred2 = predict3(sequence, demographics)[0]
    
    m_w,da_w,c_w = [0.273, 0.345, 0.382], [0.60, 0.40], [+0.0021,-0.0007,-0.0014]

    m_wts, preds = np.asarray(m_w), []
    
    for a,b,c in zip(pred0,pred1,pred2):    
        l_abc = [
            { 'wts':m_wts[0], 'pred':a, 'res':0 },
            { 'wts':m_wts[1], 'pred':b, 'res':0 },
            { 'wts':m_wts[2], 'pred':c, 'res':0 },
        ]
        l_asc  = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=False)
        l_desc = sorted(copy.deepcopy(l_abc), key=lambda _:_['pred'],reverse=True)
        
        for asc, c_wts in zip(l_asc, c_w): asc ['res'] = asc ['pred'] * (asc ['wts'] +c_wts)
        for desc,c_wts in zip(l_desc,c_w): desc['res'] = desc['pred'] * (desc['wts'] +c_wts)

        result_asc  = sum([asc ['res'] for asc in l_asc])
        result_desc = sum([desc['res'] for asc in l_desc])

        result = result_asc * da_w[0] + da_w[1] * result_desc
 
        preds.append(result)
        
    avg_pred =  np.asarray(preds)
                
    return dataset.le.classes_[avg_pred.argmax()]


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
    
import numpy as np
import pandas as pd
import re
try:
    import polars as pl
except Exception:
    pl = None

TTA_ENABLE = True
SMOOTH_WIN = 3
EPS = 1e-12

def _entropy(p):
    p = np.clip(p, EPS, 1.0)
    p = p / p.sum(axis=-1, keepdims=True)
    return -np.sum(p * np.log(p), axis=-1)

def _median_roll(a, win=3):
    s = pd.Series(a)
    out = s.rolling(win, min_periods=1).median()
    out = out.fillna(method="bfill").fillna(method="ffill")
    return out.values

def _smooth_sequence(sequence, win=3):
    if hasattr(sequence, "to_pandas"):
        df = sequence.to_pandas().copy()
    else:
        df = sequence.copy()
    pat_tof = re.compile(r"^tof_\d+_v\d+$")
    pat_thm = re.compile(r"^(thm_|thermal_)")
    for col in list(df.columns):
        if getattr(df[col], "dtype", None) is not None and str(df[col].dtype).startswith(("float","int")):
            if pat_tof.match(col) or pat_thm.match(col):
                try:
                    df[col] = _median_roll(df[col].values, win=win)
                except Exception:
                    pass
    if pl is not None:
        try:
            return pl.from_pandas(df)
        except Exception:
            return df
    return df

def _prob_to_logits(p):
    p = np.clip(p, EPS, 1.0 - EPS)
    if p.ndim == 1:
        p = p[None, :]
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(p)

def _logits_to_prob(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def predict(sequence, demographics):
    p1 = predict1(sequence, demographics)[0]
    p2 = predict2(sequence, demographics)[0]
    p3 = predict3(sequence, demographics)[0]

    if TTA_ENABLE:
        seq_s = _smooth_sequence(sequence, win=SMOOTH_WIN)
        p1_s = predict1(seq_s, demographics)[0]
        p2_s = predict2(seq_s, demographics)[0]
        p3_s = predict3(seq_s, demographics)[0]
        p1 = 0.5*(p1 + p1_s)
        p2 = 0.5*(p2 + p2_s)
        p3 = 0.5*(p3 + p3_s)

    z1 = _prob_to_logits(p1) / temperatures.get("p1", 1.0)
    z2 = _prob_to_logits(p2) / temperatures.get("p2", 1.0)
    z3 = _prob_to_logits(p3) / temperatures.get("p3", 1.0)

    if 'meta_model' in globals() and meta_model is not None:
        Xq = np.concatenate([z1, z2, z3], axis=1)
        pq = meta_model.predict_proba(Xq)
    else:
        base = np.array([0.30, 0.35, 0.35], dtype=float)
        e1, e2, e3 = float(_entropy(p1)), float(_entropy(p2)), float(_entropy(p3))
        inv = np.array([1.0/(e1+1e-6), 1.0/(e2+1e-6), 1.0/(e3+1e-6)], dtype=float)
        inv = inv / inv.sum()
        w = 0.5*base + 0.5*inv
        pq = _logits_to_prob(w[0]*z1 + w[1]*z2 + w[2]*z3)

    if 'apply_demographics_prior' in globals():
        pq2 = apply_demographics_prior(pq, demographics)
    else:
        pq2 = pq

    try:
        cls_idx = int(np.argmax(pq2, axis=1)[0])
        return dataset.le.classes_[cls_idx]
    except Exception:
        return int(np.argmax(pq2, axis=1)[0])

# Improve prediction!!

import numpy as np, pandas as pd, os, re
def _num_cols(df):
    cols = []
    for c in df.columns:
        try:
            if pd.api.types.is_numeric_dtype(df[c]):
                cols.append(c)
        except Exception:
            pass
    return cols

_pat_tof = re.compile(r"^(tof|tof_)\b|^tof_\d+(_mean|_std|_min|_max)?$")
_pat_thm = re.compile(r"^(thm_|thermal_)")

def _winsorize_inplace(df, cols, p=0.01):
    for c in cols:
        try:
            lo = df[c].quantile(p)
            hi = df[c].quantile(1-p)
            df[c] = df[c].clip(lo, hi)
        except Exception:
            pass

def _sequence_baseline(df, cols, k=10):
    base = {}
    head = df.head(k)
    for c in cols:
        try:
            base[c] = float(head[c].median(skipna=True))
        except Exception:
            base[c] = 0.0
    return base

def _context_normalize(sequence, demographics):
    if not V3_ENABLE_CONTEXT_NORM:
        return sequence
    if hasattr(sequence, "to_pandas"):
        df = sequence.to_pandas().copy()
        to_polars = True
    else:
        df = sequence.copy()
        to_polars = False
    numc = _num_cols(df)
    tof_cols = [c for c in numc if _pat_tof.search(c)]
    thm_cols = [c for c in numc if _pat_thm.search(c)]
    _winsorize_inplace(df, tof_cols + thm_cols, p=V3_WINSOR_P)
    base_tof = _sequence_baseline(df, tof_cols, k=10) if len(tof_cols) else {}
    base_thm = _sequence_baseline(df, thm_cols, k=10) if len(thm_cols) else {}
    for c in tof_cols:
        try: df[c] = df[c] - base_tof.get(c, 0.0)
        except Exception: pass
    for c in thm_cols:
        try: df[c] = df[c] - base_thm.get(c, 0.0)
        except Exception: pass

    arm_len = None
    try:
        if demographics is not None:
            if hasattr(demographics, "to_pandas"):
                demo = demographics.to_pandas()
            else:
                demo = demographics
            for cand in ["shoulder_to_wrist_cm", "elbow_to_wrist_cm"]:
                if cand in demo.columns and pd.notnull(demo[cand]).any():
                    val = float(pd.to_numeric(demo[cand], errors="coerce").dropna().mean())
                    if val > 0: arm_len = val; break
    except Exception:
        arm_len = None
    if arm_len is not None and len(tof_cols):
        s = max(arm_len, 1e-3)
        for c in tof_cols:
            try: df[c] = df[c] / s
            except Exception: pass

    if 'polars' in globals() and polars is not None and to_polars:
        try:
            import polars as pl
            return pl.from_pandas(df)
        except Exception:
            return df
    return df

def _prob_to_logits(p, eps=1e-12):
    p = np.clip(p, eps, 1.0 - eps)
    if p.ndim == 1: p = p[None, :]
    p = p / p.sum(axis=1, keepdims=True)
    return np.log(p)

def _logits_to_prob(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = np.exp(z); return ez / ez.sum(axis=1, keepdims=True)

def _entropy(p, eps=1e-12):
    p = np.clip(p, eps, 1.0); p = p / p.sum(axis=-1, keepdims=True)
    return -np.sum(p * np.log(p), axis=-1)

# learn group-wise temperatures
temperatures_by_group = None
if V3_USE_GROUP_TS:
    try:
        oof_paths = [
            f'{CFG["OOF_DIR"]}/oof_predict1.csv',
            f'{CFG["OOF_DIR"]}/oof_predict2.csv',
            f'{CFG["OOF_DIR"]}/oof_predict3.csv',
        ]
        oofs = []
        for p in oof_paths:
            if os.path.exists(p):
                df = pd.read_csv(p)
                if 'sequence_id' in df.columns and 'y' in df.columns:
                    oofs.append(df)
        demo_path = f'{CFG["DATASET_DIR"]}/train_demographics.csv'
        if len(oofs)==3 and os.path.exists(demo_path):
            demo = pd.read_csv(demo_path)[["sequence_id","adult_child","shoulder_to_wrist_cm"]]
            merged = []
            for i,df in enumerate(oofs,1):
                prob_cols = [c for c in df.columns if c.startswith("class_")]
                tmp = df[['sequence_id','y']+prob_cols].merge(demo, on="sequence_id", how="left")
                tmp['arm_bin'] = (pd.to_numeric(tmp['shoulder_to_wrist_cm'], errors='coerce')>V3_ARM_BIN_THRESHOLD).astype('Int64')
                tmp['model'] = f"p{i}"; merged.append(tmp)
            full = pd.concat(merged, ignore_index=True)
            def _best_T_from_oof(P, y, Ts=np.linspace(0.5,5.0,46)):
                logits = _prob_to_logits(P)
                n = len(y); bestT, bestNLL = 1.0, 1e9
                for T in Ts:
                    q = _logits_to_prob(logits / T)
                    nll = -np.mean(np.log(np.clip(q[np.arange(n), y], 1e-12, 1.0)))
                    if nll < bestNLL: bestNLL, bestT = nll, float(T)
                return bestT
            temperatures_by_group = {}
            for key, sub in full.groupby(['model','adult_child','arm_bin'], dropna=False):
                ch = sub.dropna(subset=[c for c in sub.columns if c.startswith("class_")]+['y'])
                if len(ch) > 50:
                    P = ch[[c for c in ch.columns if c.startswith("class_")]].values
                    y = ch['y'].astype(int).values
                    Tbest = _best_T_from_oof(P, y)
                    temperatures_by_group[key] = Tbest
            print("[INFO] Learned group-wise T:", len(temperatures_by_group or {}))
    except Exception as e:
        temperatures_by_group = None
        print("[WARN] Group-wise T disabled:", e)

def _select_T(model_key, demographics):
    if temperatures_by_group is not None:
        try:
            if hasattr(demographics, "to_pandas"):
                d = demographics.to_pandas()
            else:
                d = demographics
            ac = d.get("adult_child")
            if isinstance(ac, pd.Series): ac = ac.iloc[0]
            ac = int(ac) if pd.notnull(ac) else None
            ab = None
            if "shoulder_to_wrist_cm" in d:
                val = pd.to_numeric(d["shoulder_to_wrist_cm"], errors='coerce')
                val = val.iloc[0] if isinstance(val, pd.Series) else val
                if pd.notnull(val): ab = int(float(val) > V3_ARM_BIN_THRESHOLD)
            key = (model_key, ac, ab)
            if key in temperatures_by_group:
                return float(temperatures_by_group[key])
        except Exception:
            pass
    return float(temperatures.get(model_key, 1.0))

def predict(sequence, demographics):
    # normalize
    seqN = _context_normalize(sequence, demographics)
    p1 = predict1(seqN, demographics)[0]
    p2 = predict2(seqN, demographics)[0]
    p3 = predict3(seqN, demographics)[0]

    if TTA_ENABLE and 'SMOOTH_WIN' in globals() and '_smooth_sequence' in globals():
        try:
            seq_s = _smooth_sequence(seqN, win=SMOOTH_WIN)
            p1 = 0.5*(p1 + predict1(seq_s, demographics)[0])
            p2 = 0.5*(p2 + predict2(seq_s, demographics)[0])
            p3 = 0.5*(p3 + predict3(seq_s, demographics)[0])
        except Exception:
            pass
    z1 = _prob_to_logits(p1) / _select_T("p1", demographics)
    z2 = _prob_to_logits(p2) / _select_T("p2", demographics)
    z3 = _prob_to_logits(p3) / _select_T("p3", demographics)
    if 'meta_model' in globals() and meta_model is not None:
        Xq = np.concatenate([z1, z2, z3], axis=1)
        pq = meta_model.predict_proba(Xq)
    else:
        base = np.array([0.30, 0.35, 0.35], dtype=float)
        e1, e2, e3 = float(_entropy(p1)), float(_entropy(p2)), float(_entropy(p3))
        inv = np.array([1.0/(e1+1e-6), 1.0/(e2+1e-6), 1.0/(e3+1e-6)], dtype=float)
        inv = inv / inv.sum()
        w = 0.5*base + 0.5*inv
        pq = _logits_to_prob(w[0]*z1 + w[1]*z2 + w[2]*z3)
    if 'apply_demographics_prior' in globals():
        pq2 = apply_demographics_prior(pq, demographics)
    else:
        pq2 = pq

    try:
        cls_idx = int(np.argmax(pq2, axis=1)[0])
        return dataset.le.classes_[cls_idx]
    except Exception:
        return int(np.argmax(pq2, axis=1)[0])

print("ğŸ˜�ğŸ˜�ğŸ˜� Improved predict ready ğŸ˜�ğŸ˜�ğŸ˜�")


import pandas as pd

def _infer_sequences_and_demo():
    seqs, demo = None, None
    # Dataset
    try:
        if 'dataset' in globals():
            ds = dataset
            if hasattr(ds, "get_test_sequences"):
                seqs = ds.get_test_sequences()
            elif hasattr(ds, "full_dataset") and hasattr(ds.full_dataset, "get_test_sequences"):
                seqs = ds.full_dataset.get_test_sequences()
            elif hasattr(ds, "test_sequences"):
                seqs = ds.test_sequences
            if hasattr(ds, "test_demographics"):
                demo = ds.test_demographics
            elif hasattr(ds, "full_dataset") and hasattr(ds.full_dataset, "test_demographics"):
                demo = ds.full_dataset.test_demographics
    except Exception as e:
        print("[WARN] dataset probing:", e)
    return seqs, demo

def _safe_predict_df(test_sequences, test_demographics):
    rows = []
    for sid, seq in test_sequences.items():
        demo = None
        if test_demographics is not None:
            try:
                if 'pl' in globals():
                    demo = test_demographics.filter(pl.col("sequence_id")==sid)
                else:
                    demo = test_demographics[test_demographics['sequence_id']==sid]
            except Exception:
                demo = None
        y = predict(seq, demo)
        subject_id = 0
        try:
            if hasattr(seq, "to_pandas"):
                df = seq.to_pandas()
            else:
                df = seq
            if "subject_id" in df.columns:
                subject_id = int(pd.to_numeric(df["subject_id"], errors="coerce").dropna().iloc[0])
        except Exception:
            pass
        rows.append({"sequence_id": sid, "subject_id": subject_id, "label": y})
    return pd.DataFrame(rows)

seqs, demo = _infer_sequences_and_demo()
if seqs is None:
    print("Test sequences not found")
else:
    sub = _safe_predict_df(seqs, demo)
    sub.to_csv(CFG.get("SUBMISSION_NAME","submission.csv"), index=False)
    print(sub.head())
    print("ğŸ¥°ğŸ¥° Saved ğŸ¥°ğŸ¥°", CFG.get("SUBMISSION_NAME","submission.csv"))


import os as _os, json as _json
import numpy as _np
import torch as _torch
import torch.nn.functional as _F

_CFG_PATCH = {
    "TTA_ENABLE": True,
    "TTA_RUNS": 6,
    "TTA_JITTER_STD": 0.01,
    "TTA_MAX_SHIFT": 3,
    "TTA_MASK_PROB": 0.05,
    "MC_DROPOUT_RUNS": 0,    
    "BLEND_JSON_CANDIDATES": [
        "./blend_and_temp.json",
        "/kaggle/working/blend_and_temp.json",
        "/kaggle/input/blend-and-temp/blend_and_temp.json",
        "/kaggle/input/blend_and_temp/blend_and_temp.json"
    ],
    "DEFAULT_WEIGHTS": [0.35, 0.32, 0.33],
    "DEFAULT_TEMP": 1.0
}
def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    ez = _np.exp(z)
    return ez / ez.sum(axis=1, keepdims=True)

def _to_logits_from_probs(P, eps=1e-12):
    P = _np.clip(P, eps, 1 - eps)
    return _np.log(P)

def _load_blend_and_temp():
    for p in _CFG_PATCH["BLEND_JSON_CANDIDATES"]:
        try:
            if _os.path.exists(p):
                with open(p, "r") as f:
                    cfg = _json.load(f)
                w = _np.array(cfg.get("weights", _CFG_PATCH["DEFAULT_WEIGHTS"]), dtype=float)
                T = float(cfg.get("temp", _CFG_PATCH["DEFAULT_TEMP"]))
                if _np.all(w >= 0) and w.sum() > 0:
                    w = w / w.sum()
                else:
                    w = _np.array(_CFG_PATCH["DEFAULT_WEIGHTS"], dtype=float); w = w / w.sum()
                return w, T, p
        except Exception:
            pass
    w = _np.array(_CFG_PATCH["DEFAULT_WEIGHTS"], dtype=float); w = w / w.sum()
    return w, float(_CFG_PATCH["DEFAULT_TEMP"]), None

def _maybe_to_pandas(df_like):
    try:
        import polars as _pl
        if isinstance(df_like, _pl.DataFrame):
            return df_like.to_pandas(), "pl"
    except Exception:
        pass
    try:
        import pandas as _pd
        if isinstance(df_like, _pd.DataFrame):
            return df_like.copy(), "pd"
    except Exception:
        pass
    return None, None

def _back_from_pandas(df_pd, kind):
    if kind == "pl":
        try:
            import polars as _pl
            return _pl.from_pandas(df_pd)
        except Exception:
            return df_pd
    return df_pd

def _jitter(arr, std):
    return arr + _np.random.normal(0.0, std, size=arr.shape).astype(arr.dtype, copy=False)

def _shift(arr, max_shift):
    if max_shift <= 0: return arr
    s = int(_np.random.randint(-max_shift, max_shift+1))
    if s == 0: return arr
    if s > 0:
        return _np.concatenate([_np.zeros_like(arr[:s]), arr[:-s]], axis=0)
    else:
        return _np.concatenate([arr[-s:], _np.zeros_like(arr[: -s])], axis=0)

def _mask(arr, prob):
    if prob <= 0: return arr
    m = _np.random.rand(*arr.shape) < prob
    out = arr.copy()
    out[m] = 0
    return out

def _augment_sequence(df_like, jitter_std, max_shift, mask_prob):
    df_pd, kind = _maybe_to_pandas(df_like)
    if df_pd is None:
        return df_like
    import pandas as _pd
    num_cols = [c for c in df_pd.columns if _pd.api.types.is_numeric_dtype(df_pd[c])]
    if not num_cols:
        return _back_from_pandas(df_pd, kind)
    mat = df_pd[num_cols].to_numpy()
    mat = _jitter(mat, jitter_std)
    mat = _shift(mat, max_shift)
    mat = _mask(mat, mask_prob)
    df_pd[num_cols] = mat
    return _back_from_pandas(df_pd, kind)

def _set_models_dropout_train(enable=True):
    try:
        groups = []
        if "models1" in globals(): groups.append(models1)
        if "models2" in globals(): groups.append(models2)
        for ms in groups:
            for m in ms:
                for module in m.modules():
                    if "dropout" in module.__class__.__name__.lower():
                        module.train(enable)
    except Exception:
        pass
        
def _predict2_late(sequence, demographics):
    """Late fusion via models2/avg_predict (original)."""
    imu, thm, tof = dataset.full_dataset.inference_process(sequence)
    with _torch.no_grad():
        imu, thm, tof = to_cuda(imu, thm, tof)
        logits = avg_predict(models2, imu, thm, tof)
        probabilities = _F.softmax(logits, dim=1).cpu().numpy()
    return probabilities

def _predict1_imu(sequence, demographics):
    """IMU-only: zero out THM/TOF; reuse models2 head to keep flow stable."""
    imu, thm, tof = dataset.full_dataset.inference_process(sequence)
    try:
        import numpy as _np
        thm = _np.zeros_like(thm, dtype=thm.dtype)
        tof = _np.zeros_like(tof, dtype=tof.dtype)
    except Exception:
        pass
    with _torch.no_grad():
        imu, thm, tof = to_cuda(imu, thm, tof)
        try:
            thm.zero_(); tof.zero_()
        except Exception:
            pass
        logits = avg_predict(models2, imu, thm, tof)
        probabilities = _F.softmax(logits, dim=1).cpu().numpy()
    return probabilities

_predict3_early = predict3


def _predict_with_tta(base_fn, sequence, demographics):
    if not _CFG_PATCH["TTA_ENABLE"] or _CFG_PATCH["TTA_RUNS"] <= 1:
        return base_fn(sequence, demographics)

    preds = []
    preds.append(base_fn(sequence, demographics))
    for _ in range(_CFG_PATCH["TTA_RUNS"] - 1):
        seq_aug = _augment_sequence(
            sequence,
            _CFG_PATCH["TTA_JITTER_STD"],
            _CFG_PATCH["TTA_MAX_SHIFT"],
            _CFG_PATCH["TTA_MASK_PROB"]
        )
        preds.append(base_fn(seq_aug, demographics))
    P = _np.stack(preds, axis=0).astype("float64")
    P = _np.clip(P, 1e-8, 1-1e-8)
    return P.mean(axis=0)

def predict1(sequence, demographics):
    if _CFG_PATCH["MC_DROPOUT_RUNS"] and _CFG_PATCH["MC_DROPOUT_RUNS"] > 0:
        _set_models_dropout_train(True)
        ps = []
        for _ in range(int(_CFG_PATCH["MC_DROPOUT_RUNS"])):
            ps.append(_predict_with_tta(_predict1_imu, sequence, demographics))
        _set_models_dropout_train(False)
        return _np.mean(_np.stack(ps, 0), 0)
    return _predict_with_tta(_predict1_imu, sequence, demographics)

def predict2(sequence, demographics):
    if _CFG_PATCH["MC_DROPOUT_RUNS"] and _CFG_PATCH["MC_DROPOUT_RUNS"] > 0:
        _set_models_dropout_train(True)
        ps = []
        for _ in range(int(_CFG_PATCH["MC_DROPOUT_RUNS"])):
            ps.append(_predict_with_tta(_predict2_late, sequence, demographics))
        _set_models_dropout_train(False)
        return _np.mean(_np.stack(ps, 0), 0)
    return _predict_with_tta(_predict2_late, sequence, demographics)

def predict3(sequence, demographics):
    if _CFG_PATCH["MC_DROPOUT_RUNS"] and _CFG_PATCH["MC_DROPOUT_RUNS"] > 0:
        _set_models_dropout_train(True)
        ps = []
        for _ in range(int(_CFG_PATCH["MC_DROPOUT_RUNS"])):
            ps.append(_predict_with_tta(_predict3_early, sequence, demographics))
        _set_models_dropout_train(False)
        return _np.mean(_np.stack(ps, 0), 0)
    return _predict_with_tta(_predict3_early, sequence, demographics)

print("TTA ready.")


def predict(sequence, demographics):
    p1 = predict1(sequence, demographics)
    p2 = predict2(sequence, demographics)
    p3 = predict3(sequence, demographics)

    def _to_2d(p):
        p = _np.asarray(p)
        if p.ndim == 1:
            p = p[None, :]
        return p
    p1 = _to_2d(p1); p2 = _to_2d(p2); p3 = _to_2d(p3)

    w, T, src = _load_blend_and_temp()
    if src:
        print(f"[PATCH++] Using blend/temp from: {src} -> w={w.tolist()}, T={T:.3f}")
    else:
        print(f"[PATCH++] Using default blend/temp -> w={w.tolist()}, T={T:.3f}")

    Z1, Z2, Z3 = _to_logits_from_probs(p1), _to_logits_from_probs(p2), _to_logits_from_probs(p3)
    Z = w[0]*Z1 + w[1]*Z2 + w[2]*Z3
    P = _softmax(Z / max(T, 1e-3))
    return P

