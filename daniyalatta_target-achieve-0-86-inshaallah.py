# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import kagglehub
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Download latest version
path = kagglehub.dataset_download("kerta27/cmi-data-tensorflow-train")
path = kagglehub.dataset_download("hideyukizushi/cmi25-imu-thmtof-tf-bilstm-gru-attentionlb-xx")
path = kagglehub.dataset_download("kerta27/cmi-data-gated-gru")
path = kagglehub.dataset_download("hideyukizushi/20250627-cmi-b-102-b-105")
path = kagglehub.dataset_download("hideyukizushi/cmi-d-111")
path = kagglehub.dataset_download("myso1987/cmi3-models-p")

print("Path to dataset files:", path)
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# https://www.kaggle.com/code/wasupandceacar/deterministic
# https://www.kaggle.com/code/hideyukizushi/lb-0-78-quaternions-tf-bilstm-gru-attention
# https://www.kaggle.com/code/majiaqi111/n-splits-10
# https://www.kaggle.com/code/hideyukizushi/cmi25-imu-thm-tof-tf-blendingmodel-lb-82
# https://www.kaggle.com/code/wasupandceacar/cmi-metric


import tensorflow as tf
print("TensorFlow Version:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
from tensorflow.python.platform import build_info as tf_build_info
print("cuDNN version:", tf_build_info.build_info['cudnn_version'])


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
TRAIN = False                     # ← set to True when you want to train
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

print("▶ imports ready · tensorflow", tf.__version__)

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

# ----------------------------------------------------------------- #
# Load any Models
# * is 2 Train Model Load
# ----------------------------------------------------------------- #

PRETRAINED_DIR = Path("/kaggle/input/cmi-d-111")
print("▶ INFERENCE MODE 1,2 – loading artefacts from", PRETRAINED_DIR)
final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len        = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
scaler         = joblib.load(PRETRAINED_DIR / "scaler.pkl")
gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

models1 = []
print(f"  Loading models for ensemble inference...")
for fold in range(10):
    model_path = f"{PRETRAINED_DIR}/D-111_{fold}.h5"
    print(">>>LoadModel>>>",model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models1.append(model)
print("-"*50)

for fold in range(10):
    model_path = f"{PRETRAINED_DIR}/v0629_{fold}.h5"
    print(">>>LoadModel>>>",model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models1.append(model)
print("-"*50)
print(f"[INFO]NumUseModels:{len(models1)}")

PRETRAINED_DIR = Path("/kaggle/input/n-splits-10")
print("▶ INFERENCE MODE 3 – loading artefacts from", PRETRAINED_DIR)
final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len        = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
scaler         = joblib.load(PRETRAINED_DIR / "scaler.pkl")
gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)
for fold in range(10):
    model_path = f"{PRETRAINED_DIR}/gesture_model_fold_{fold}.h5"
    print(">>>LoadModel>>>",model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models1.append(model)
print("-"*50)
print(f"[INFO]NumUseModels:{len(models1)}")

for fold in range(10):
    MODEL_DIR = "/kaggle/input/cmi-data-tensorflow-train"
    model_path = f"{MODEL_DIR}/gesture_model_fold_{fold}.h5"
    print(">>>LoadModel>>>",model_path)
    model = load_model(model_path, compile=False, custom_objects=custom_objs)
    models1.append(model)
print("-"*50)
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


import polars as pl
from pathlib import Path

RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")

# load CSVs
train_df = pl.read_csv(RAW_DIR / "train.csv")
demo_df = pl.read_csv(RAW_DIR / "train_demographics.csv")

# merge on "subject" instead of "sequence_id"
train_seq = train_df.join(demo_df, on="subject", how="left")

# save parquet
train_seq.write_parquet("/kaggle/working/train_sequences.parquet")

print("✅ train_sequences.parquet saved in /kaggle/working")



import polars as pl
import numpy as np
from pathlib import Path

RAW_DIR = Path('/kaggle/working/')

try:
    # Try loading training data first
    train_seq = pl.read_parquet(RAW_DIR / "train_sequences.parquet")
    feature_cols = [col for col in train_seq.columns if col.startswith(('acc_', 'rot_', 'tof_'))]
    data = train_seq[feature_cols].to_pandas().ffill().bfill().fillna(0).values
    print("Using train_sequences.parquet for mean/std calculation")
except FileNotFoundError:
    print("Warning: train_sequences.parquet not found. Using test_sequences.parquet for mean/std estimation.")
    try:
        test_seq = pl.read_parquet(RAW_DIR / "test_sequences.parquet")
        feature_cols = [col for col in test_seq.columns if col.startswith(('acc_', 'rot_', 'tof_'))]
        data = test_seq[feature_cols].to_pandas().ffill().bfill().fillna(0).values
    except FileNotFoundError:
        print("Error: Neither train_sequences.parquet nor test_sequences.parquet found. Using default mean=0 and std=1.")
        data = np.zeros((1, 332))  # 332 = 7 IMU + 5*64 ToF + 5*5 derived ToF features

# Calculate mean and std
mean = np.mean(data, axis=0)
std = np.std(data, axis=0) + 1e-6  # Avoid division by zero

# Save to files
np.save('/kaggle/working/mean.npy', mean)
np.save('/kaggle/working/std.npy', std)
print("Generated mean.npy and std.npy in /kaggle/working")


def predict3(sequence: pl.DataFrame, demographics: pl.DataFrame) -> np.ndarray:
    import numpy as np
    
    pred1 = predict1(sequence, demographics)  # Model 1 (~0.820)
    pred2 = predict2(sequence, demographics)  # Model 2 (~0.829)
    
    base_weights = {'A': 0.40, 'B': 0.30, 'C': 0.30}
    
    perturbation = np.random.uniform(-0.02, 0.02, 3)
    weights = {
        'A': max(0.1, min(0.8, base_weights['A'] + perturbation[0])),
        'B': max(0.1, min(0.8, base_weights['B'] + perturbation[1])),
        'C': max(0.1, min(0.8, base_weights['C'] + perturbation[2]))
    }
    total = sum(weights.values())
    weights = {k: v/total for k, v in weights.items()}
    
    pred = weights['A'] * pred1 + weights['B'] * pred2 + weights['C'] * pred1
    
    max_prob = np.max(pred)
    if max_prob < 0.30:  # Lowered threshold
        pred = np.zeros_like(pred)
        pred[np.argmax(pred1)] = 1.0
    
    return pred


import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path

def softmax_with_temperature(pred, T=1.0):
    pred = pred / T
    exp_pred = np.exp(pred - np.max(pred))  # Subtract max for numerical stability
    return exp_pred / np.sum(exp_pred)

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    pred = predict3(sequence, demographics)
    
    # Apply softmax temperature scaling
    pred = softmax_with_temperature(pred, T=0.75)  # Sharper predictions
    
    # Class-specific correction weights for imbalance
    class_weights = np.ones(len(gesture_classes))  # Default weights
    # Boost weights for underrepresented classes (e.g., "Drink from bottle/cup")
    underrepresented = ["Drink from bottle/cup"]  # Add other known underrepresented classes
    for cls in underrepresented:
        if cls in gesture_classes:
            idx = np.where(gesture_classes == cls)[0][0]
            class_weights[idx] = 1.05  # 5% boost for underrepresented classes
    
    # Apply correction weights
    c_w = np.array([+0.0025, -0.0005, -0.0012])
    pred = pred * (1 + c_w[0]) * class_weights + c_w[1] + np.random.normal(0, abs(c_w[2]), pred.shape)
    
    # Ensure probabilities are non-negative and normalized
    pred = np.clip(pred, 0, None)
    pred = pred / pred.sum()
    
    return gesture_classes[np.argmax(pred)]

# # Submission
# RAW_DIR = Path('/kaggle/input/cmi-detect-behavior-with-sensor-data')
# sub = pl.read_csv(RAW_DIR / "sample_submission.csv")
# test_seq = pl.read_parquet(RAW_DIR / "test_sequences.parquet")
# test_demo = pl.read_parquet(RAW_DIR / "test_demographics.parquet")

# submission = []
# for sid in sub["sequence_id"].to_numpy():
#     sequence = test_seq.filter(pl.col("sequence_id") == sid)
#     demographics = test_demo.filter(pl.col("sequence_id") == sid)
#     pred = predict(sequence, demographics)
#     submission.append({"sequence_id": sid, "gesture": pred})

# submission_df = pd.DataFrame(submission)
# submission_df.to_csv("submission.csv", index=False)
# print("Submission file created: submission.csv")









