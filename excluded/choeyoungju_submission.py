# IMPORTANT: SOME KAGGLE DATA SOURCES ARE PRIVATE
# RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES.

# import kagglehub
# kagglehub.login()



# IMPORTANT: RUN THIS CELL IN ORDER TO IMPORT YOUR KAGGLE DATA SOURCES,
# THEN FEEL FREE TO DELETE THIS CELL.
# NOTE: THIS NOTEBOOK ENVIRONMENT DIFFERS FROM KAGGLE'S PYTHON
# ENVIRONMENT SO THERE MAY BE MISSING LIBRARIES USED BY YOUR
# NOTEBOOK.

# cmi_detect_behavior_with_sensor_data_path = kagglehub.competition_download('cmi-detect-behavior-with-sensor-data')
# richolson_cmi_2025_metric_copy_for_import_path = kagglehub.utility_script_install('richolson/cmi-2025-metric-copy-for-import')

# print('Data source import complete.')



import os, json, joblib, numpy as np, pandas as pd
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

from pathlib import Path
from tensorflow.keras.layers import Attention
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import Softmax
from scipy.signal import find_peaks


from scipy.stats import mode
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


from tensorflow.keras.layers import Layer, Dense, Softmax, Multiply
import tensorflow as tf


from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Activation, Add, GlobalAveragePooling1D, Dense,
    Dropout, LayerNormalization, MultiHeadAttention, Lambda
)
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint


#local
# RAW_DIR = Path("./drive/MyDrive/CMI")

#public
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")


# from google.colab import drive
# drive.mount('/content/drive')


"""
Hierarchical macro F1 metric for the CMI 2025 Challenge.

This script defines a single entry point `score(solution, submission, row_id_column_name)`
that the Kaggle metrics orchestrator will call.
It performs validation on submission IDs and computes a combined binary & multiclass F1 score.
"""

import pandas as pd
from sklearn.metrics import f1_score


class ParticipantVisibleError(Exception):
    """Errors raised here will be shown directly to the competitor."""
    pass


class CompetitionMetric:
    """Hierarchical macro F1 for the CMI 2025 challenge."""
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

    def calculate_hierarchical_f1(
        self,
        sol: pd.DataFrame,
        sub: pd.DataFrame
    ) -> float:

        # Validate gestures
        invalid_types = {i for i in sub['gesture'].unique() if i not in self.all_classes}
        if invalid_types:
            raise ParticipantVisibleError(
                f"Invalid gesture values in submission: {invalid_types}"
            )

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

        return 0.5 * f1_binary + 0.5 * f1_macro


def score(
    solution: pd.DataFrame,
    submission: pd.DataFrame,
    row_id_column_name: str
) -> float:
    """
    Compute hierarchical macro F1 for the CMI 2025 challenge.

    Expected input:
      - solution and submission as pandas.DataFrame
      - Column 'sequence_id': unique identifier for each sequence
      - 'gesture': one of the eight target gestures or "Non-Target"

    This metric averages:
    1. Binary F1 on SequenceType (Target vs Non-Target)
    2. Macro F1 on gesture (mapping non-targets to "Non-Target")

    Raises ParticipantVisibleError for invalid submissions,
    including invalid SequenceType or gesture values.


    Examples
    --------
    >>> import pandas as pd
    >>> row_id_column_name = "id"
    >>> solution = pd.DataFrame({'id': range(4), 'gesture': ['Eyebrow - pull hair']*4})
    >>> submission = pd.DataFrame({'id': range(4), 'gesture': ['Forehead - pull hairline']*4})
    >>> score(solution, submission, row_id_column_name=row_id_column_name)
    0.5
    >>> submission = pd.DataFrame({'id': range(4), 'gesture': ['Text on phone']*4})
    >>> score(solution, submission, row_id_column_name=row_id_column_name)
    0.0
    >>> score(solution, solution, row_id_column_name=row_id_column_name)
    1.0
    """
    # Validate required columns
    for col in (row_id_column_name, 'gesture'):
        if col not in solution.columns:
            raise ParticipantVisibleError(f"Solution file missing required column: '{col}'")
        if col not in submission.columns:
            raise ParticipantVisibleError(f"Submission file missing required column: '{col}'")

    metric = CompetitionMetric()
    return metric.calculate_hierarchical_f1(solution, submission)


import random
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.experimental.numpy.random.seed(seed)
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
seed_everything(seed=42)


TRAIN = False
PRETRAINED_DIR = Path("/kaggle/input/predict-v7")
EXPORT_DIR = Path("./")
BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 5e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 160
PATIENCE = 40


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


# 스케일링
def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list[str], scaler: StandardScaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')


# 데이터 mixup
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


# 데이터 중력값 없애기
# acc, rot는 가속도 관련 데이터 - 중력값을 빼버리면서 더 사용하기 좋은 데이터셋으로 변경
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


def build_three_branch_model(pad_len, imu_dim, tof_dim, thm_dim, n_classes, wd=1e-4):
    inp = Input(shape=(pad_len, imu_dim + tof_dim + thm_dim))
    imu = Lambda(lambda t: t[:, :, :imu_dim])(inp)
    tof = Lambda(lambda t: t[:, :, imu_dim:imu_dim + tof_dim])(inp)
    thm = Lambda(lambda t: t[:, :, imu_dim + tof_dim:])(inp)

    x_imu = residual_se_cnn_block(imu, 64, 3, drop=0.1, wd=wd)
    x_imu = residual_se_cnn_block(x_imu, 128, 5, drop=0.1, wd=wd)

    x_tof = Conv1D(64, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(tof)
    x_tof = BatchNormalization()(x_tof); x_tof = Activation('relu')(x_tof)
    x_tof = MaxPooling1D(2)(x_tof); x_tof = Dropout(0.2)(x_tof)
    x_tof = Conv1D(128, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(x_tof)
    x_tof = BatchNormalization()(x_tof); x_tof = Activation('relu')(x_tof)
    x_tof = MaxPooling1D(2)(x_tof); x_tof = Dropout(0.2)(x_tof)

    x_thm = Conv1D(32, 3, padding='same', use_bias=False, kernel_regularizer=l2(wd))(thm)
    x_thm = BatchNormalization()(x_thm); x_thm = Activation('relu')(x_thm)
    x_thm = GlobalAveragePooling1D()(x_thm)
    x_thm = Dense(64, activation='relu')(x_thm)

    merged = Concatenate()([x_imu, x_tof])
    xa = Bidirectional(LSTM(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xb = Bidirectional(GRU(128, return_sequences=True, kernel_regularizer=l2(wd)))(merged)
    xc = GaussianNoise(0.09)(merged)
    xc = Dense(16, activation='elu')(xc)

    x = Concatenate()([xa, xb, xc])
    x = Dropout(0.4)(x)
    x = attention_layer(x)

    x = Concatenate()([x, x_thm])

    for units, drop in [(256, 0.5), (128, 0.3)]:
        x = Dense(units, use_bias=False, kernel_regularizer=l2(wd))(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Dropout(drop)(x)

    out = Dense(n_classes, activation='softmax', kernel_regularizer=l2(wd))(x)
    return Model(inp, out)


if TRAIN:
  df = pd.read_csv(RAW_DIR / "train.csv")
  train_dem_df = pd.read_csv(RAW_DIR / "train_demographics.csv")


# if TRAIN:
#   df_sequence = df['sequence_id'].unique()[:300]
#   df = df[df['sequence_id'].isin(df_sequence)].reset_index(drop=True)


if TRAIN:
  df_for_groups = pd.merge(df.copy(), train_dem_df, on='subject', how='left')


if TRAIN:
  le = LabelEncoder()
  df['gesture_int'] = le.fit_transform(df['gesture'])
  gesture_classes = le.classes_

  # 파일 저장
  np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)


def extract_periodicity_features(signal, max_lag=100):
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]

    peaks = find_peaks(autocorr[:max_lag])[0]

    return {
        'dominant_period': peaks[0] if len(peaks) > 0 else 0,
        'periodicity_strength': np.max(autocorr[1:max_lag]) if len(autocorr) > 1 else 0,
        'num_significant_periods': len(peaks)
    }


# if TRAIN:
#   tof_cols = [col for col in df.columns if col.startswith('tof_')]

#   sensor_groups = {
#       'acc': ['acc_x', 'acc_y', 'acc_z'],
#       'thm': ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'],
#       'tof': tof_cols,
#       'rot': ['rot_w', 'rot_x', 'rot_y', 'rot_z']
#   }

#   features_list = []
#   sequence_ids = df['sequence_id'].unique()

#   for seq_id in sequence_ids:
#       seq_df = df[df['sequence_id'] == seq_id]
#       features = {'sequence_id': seq_id}

#       for group_name, axes in sensor_groups.items():
#           for axis in axes:
#               if axis in seq_df.columns:
#                   res = extract_periodicity_features(seq_df[axis].values)
#                   features[f'dominant_period_{axis}'] = res['dominant_period']
#                   features[f'periodicity_strength_{axis}'] = res['periodicity_strength']
#                   features[f'num_significant_periods_{axis}'] = res['num_significant_periods']
#       features_list.append(features)

#   features_df = pd.DataFrame(features_list)

#   features_unique = features_df.drop_duplicates(subset=['sequence_id']).reset_index(drop=True)



if TRAIN:
    tof_cols = [col for col in df.columns if col.startswith('tof_')]

    sensor_groups = {
        'acc': ['acc_x', 'acc_y', 'acc_z'],
        'thm': ['thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5'],
        'tof': tof_cols,
        'rot': ['rot_w', 'rot_x', 'rot_y', 'rot_z']
    }

    features_list = []
    sequence_ids = df['sequence_id'].unique()

    for seq_id in sequence_ids:
        seq_df = df[df['sequence_id'] == seq_id]
        features = {'sequence_id': seq_id}

        for group_name, axes in sensor_groups.items():
            for axis in axes:
                if axis in seq_df.columns:
                    res = extract_periodicity_features(seq_df[axis].values)
                    features[f'dominant_period_{axis}'] = res['dominant_period']
                    features[f'periodicity_strength_{axis}'] = res['periodicity_strength']
                    features[f'num_significant_periods_{axis}'] = res['num_significant_periods']
        features_list.append(features)

    features_df = pd.DataFrame(features_list)
    features_unique = features_df.drop_duplicates(subset=['sequence_id']).reset_index(drop=True)

    # 모든 값이 0인 컬럼 찾기
    zero_cols = [col for col in features_unique.columns if col != 'sequence_id' and (features_unique[col] == 0).all()]

    zero_cols = [
    'dominant_period_thm_1', 'num_significant_periods_thm_1', 'periodicity_strength_thm_1',
    'dominant_period_thm_2', 'num_significant_periods_thm_2', 'periodicity_strength_thm_2',
    'dominant_period_thm_3', 'num_significant_periods_thm_3', 'periodicity_strength_thm_3',
    'dominant_period_thm_4', 'num_significant_periods_thm_4', 'periodicity_strength_thm_4',
    'dominant_period_thm_5', 'num_significant_periods_thm_5', 'periodicity_strength_thm_5'
    ]
    print("삭제할 0 컬럼들:", zero_cols)

    features_unique = features_unique.drop(columns=zero_cols)


if TRAIN:
  df = df.merge(features_df, on='sequence_id', how='left')


if TRAIN:
  df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
  df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))


if TRAIN:
  df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
  df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)


if TRAIN:
  linear_accel_list = []
  for _, group in df.groupby('sequence_id'):
      acc_data_group = group[['acc_x', 'acc_y', 'acc_z']]
      rot_data_group = group[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
      linear_accel_group = remove_gravity_from_acc(acc_data_group, rot_data_group)
      linear_accel_list.append(pd.DataFrame(linear_accel_group, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=group.index))

  df_linear_accel = pd.concat(linear_accel_list)
  df = pd.concat([df, df_linear_accel], axis=1)


if TRAIN:
  df['linear_acc_mag'] = np.sqrt(df['linear_acc_x']**2 + df['linear_acc_y']**2 + df['linear_acc_z']**2)


if TRAIN:
  df['linear_acc_mag_jerk'] = df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)


if TRAIN:
  meta_cols = { ... }

  imu_cols_base = ['linear_acc_x', 'linear_acc_y', 'linear_acc_z']
  imu_cols_base.extend([c for c in df.columns if c.startswith('rot_') and c not in ['rot_angle', 'rot_angle_vel']])

  imu_engineered_features = [
      'acc_mag', 'rot_angle',
      'acc_mag_jerk', 'rot_angle_vel',
      'linear_acc_mag', 'linear_acc_mag_jerk'
  ]

  imu_periodic_features = []
  for axis in ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']:
      imu_periodic_features.extend([
          f'dominant_period_{axis}',
          f'periodicity_strength_{axis}',
          f'num_significant_periods_{axis}'
      ])


  imu_cols = imu_cols_base + imu_engineered_features + imu_periodic_features
  imu_cols = list(dict.fromkeys(imu_cols))


if TRAIN:
  tof_cols = []
  for i in range(1, 6):
      tof_cols.extend([f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max'])

  tof_periodic_features = []
  for col in tof_cols:
      tof_periodic_features.extend([
          f'dominant_period_{col}',
          f'periodicity_strength_{col}',
          f'num_significant_periods_{col}'
      ])

  tof_cols = tof_cols + tof_periodic_features

  thm_cols = [c for c in df.columns if c.startswith('thm_')]

  thm_periodic_features = []
  for col in thm_cols:
      thm_periodic_features.extend([
          f'dominant_period_{col}',
          f'periodicity_strength_{col}',
          f'num_significant_periods_{col}'
      ])

  thm_diff_cols = [f"{c}_diff" for c in thm_cols]
  thm_rollmean_cols = [f"{c}_rollmean" for c in thm_cols]
  thm_rollstd_cols = [f"{c}_rollstd" for c in thm_cols]


  thm_cols = thm_cols + thm_diff_cols + thm_rollmean_cols + thm_rollstd_cols + thm_periodic_features


  # zero_cols - periodic이 모두 0인 컬럼
  # 해당 컬럼들 제거
  imu_cols = [col for col in imu_cols if col not in zero_cols]
  tof_cols = [col for col in tof_cols if col not in zero_cols]
  thm_cols = [col for col in thm_cols if col not in zero_cols]

  imu_dim_final = len(imu_cols)
  tof_dim_final = len(tof_cols)
  thm_dim_final = len(thm_cols)


  final_feature_cols = imu_cols + tof_cols + thm_cols


if TRAIN:
  np.save(EXPORT_DIR / "feature_cols.npy", np.array(final_feature_cols))


if TRAIN:
  seq_gp = df.groupby('sequence_id')

  all_steps_for_scaler_list = []
  X_list_unscaled, y_list_int_for_stratify, lens = [], [], []

  for seq_id, seq_df_orig in seq_gp:
      seq_df = seq_df_orig.copy()

      # TOF별 periodicity feature 저장 리스트 초기화 (1~5)
      periodicity_dominant_period = {i: [] for i in range(1, 6)}
      periodicity_strength = {i: [] for i in range(1, 6)}
      periodicity_num_sig = {i: [] for i in range(1, 6)}

      for i in range(1, 6):
          pixel_cols_tof = [f"tof_{i}_v{p}" for p in range(64)]
          tof_sensor_data = seq_df[pixel_cols_tof].replace(-1, np.nan)

          # 기존 TOF 요약 통계 계산
          seq_df[f'tof_{i}_mean'] = tof_sensor_data.mean(axis=1)
          seq_df[f'tof_{i}_std']  = tof_sensor_data.std(axis=1)
          seq_df[f'tof_{i}_min']  = tof_sensor_data.min(axis=1)
          seq_df[f'tof_{i}_max']  = tof_sensor_data.max(axis=1)

          # TOF 64개 축 각각에 대해 periodicity 추출
          for col in pixel_cols_tof:
              if col in seq_df.columns:
                  signal = seq_df[col].values
                  res = extract_periodicity_features(signal)

                  periodicity_dominant_period[i].append(res['dominant_period'])
                  periodicity_strength[i].append(res['periodicity_strength'])
                  periodicity_num_sig[i].append(res['num_significant_periods'])

          # 축별 periodicity 결과에 대해 mean, std, min, max 계산하여 seq_df 컬럼에 추가
          seq_df[f'dominant_period_tof_{i}_mean'] = np.nanmean(periodicity_dominant_period[i])
          seq_df[f'dominant_period_tof_{i}_std'] = np.nanstd(periodicity_dominant_period[i])
          seq_df[f'dominant_period_tof_{i}_min'] = np.nanmin(periodicity_dominant_period[i])
          seq_df[f'dominant_period_tof_{i}_max'] = np.nanmax(periodicity_dominant_period[i])

          seq_df[f'periodicity_strength_tof_{i}_mean'] = np.nanmean(periodicity_strength[i])
          seq_df[f'periodicity_strength_tof_{i}_std'] = np.nanstd(periodicity_strength[i])
          seq_df[f'periodicity_strength_tof_{i}_min'] = np.nanmin(periodicity_strength[i])
          seq_df[f'periodicity_strength_tof_{i}_max'] = np.nanmax(periodicity_strength[i])

          seq_df[f'num_significant_periods_tof_{i}_mean'] = np.nanmean(periodicity_num_sig[i])
          seq_df[f'num_significant_periods_tof_{i}_std'] = np.nanstd(periodicity_num_sig[i])
          seq_df[f'num_significant_periods_tof_{i}_min'] = np.nanmin(periodicity_num_sig[i])
          seq_df[f'num_significant_periods_tof_{i}_max'] = np.nanmax(periodicity_num_sig[i])

      # THM 관련 feature 추가 (기존 코드 유지)
      for col in thm_cols:
          seq_df[f'{col}_diff'] = seq_df[col].diff().fillna(0)
          seq_df[f'{col}_rollmean'] = seq_df[col].rolling(window=5, min_periods=1).mean()
          seq_df[f'{col}_rollstd'] = seq_df[col].rolling(window=5, min_periods=1).std().fillna(0)

      # 최종 피처 행렬 생성 (결측치는 앞뒤로 보간 후 0으로 채움)
      mat_unscaled = seq_df[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')

      all_steps_for_scaler_list.append(mat_unscaled)
      X_list_unscaled.append(mat_unscaled)
      y_list_int_for_stratify.append(seq_df['gesture_int'].iloc[0])
      lens.append(len(mat_unscaled))


if TRAIN:
  for col in thm_cols:
      # 온도 변화율 (차분)
      seq_df[f'{col}_diff'] = seq_df[col].diff().fillna(0)
      # 이동 평균 (윈도우 크기 예: 5)
      seq_df[f'{col}_rollmean'] = seq_df[col].rolling(window=5, min_periods=1).mean()
      # 이동 표준편차
      seq_df[f'{col}_rollstd'] = seq_df[col].rolling(window=5, min_periods=1).std().fillna(0)

  mat_unscaled = seq_df[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32')

  all_steps_for_scaler_list.append(mat_unscaled)
  X_list_unscaled.append(mat_unscaled)
  y_list_int_for_stratify.append(seq_df['gesture_int'].iloc[0])
  lens.append(len(mat_unscaled))


if TRAIN:
  all_steps_concatenated = np.concatenate(all_steps_for_scaler_list, axis=0)
  scaler = StandardScaler().fit(all_steps_concatenated)
  joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")
  del all_steps_for_scaler_list, all_steps_concatenated

  X_scaled_list = [scaler.transform(x_seq) for x_seq in X_list_unscaled]
  del X_list_unscaled


if TRAIN:
  pad_len = int(np.percentile(lens, PAD_PERCENTILE))

  np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)


if TRAIN:
  X = pad_sequences(X_scaled_list, maxlen=pad_len, padding='post', truncating='post', dtype='float32')
  del X_scaled_list


if TRAIN:
  y_int_for_stratify = np.array(y_list_int_for_stratify)
  y = to_categorical(y_int_for_stratify, num_classes=len(le.classes_))


if TRAIN:
  X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.2, random_state=82, stratify=y_int_for_stratify)


if TRAIN:
  cw_vals = compute_class_weight('balanced', classes=np.arange(len(le.classes_)), y=y_int_for_stratify)
  class_weight = dict(enumerate(cw_vals))


if TRAIN:
    model = build_three_branch_model(pad_len, imu_dim_final, tof_dim_final, thm_dim_final, len(le.classes_), wd=WD)

    steps = len(X_tr) // BATCH_SIZE
    lr_sched = tf.keras.optimizers.schedules.CosineDecayRestarts(5e-4, first_decay_steps=15 * steps)

    model.compile(optimizer=Adam(lr_sched),
                  loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
                  metrics=['accuracy'])

    train_gen = MixupGenerator(X_tr, y_tr, batch_size=BATCH_SIZE, alpha=MIXUP_ALPHA)
    cb = EarlyStopping(patience=PATIENCE, restore_best_weights=True, verbose=1, monitor='val_accuracy', mode='max')

    checkpoint = ModelCheckpoint(
      filepath=str(EXPORT_DIR / "gesture_two_branch_mixup.h5"),
      monitor='val_accuracy',
      verbose=1,
      save_best_only=True,
      mode='max'
    )
    print("  Starting model training...")
    model.fit(train_gen, epochs=EPOCHS, validation_data=(X_val, y_val),
              class_weight=class_weight, callbacks=[cb, checkpoint], verbose=1)

    # model.save(EXPORT_DIR / "gesture_two_branch_mixup.h5")


else:
    print("▶ INFERENCE MODE – loading artefacts from", PRETRAINED_DIR)
    final_feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
    pad_len        = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
    scaler         = joblib.load(PRETRAINED_DIR / "scaler.pkl")
    gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

    temp_imu_cols = [c for c in final_feature_cols if c.startswith('acc_') or c.startswith('rot_')]
    imu_dim_final = len(temp_imu_cols)
    tof_thm_aggregated_dim_final = len(final_feature_cols) - imu_dim_final

    custom_objs = {
        'time_sum': time_sum,
        'squeeze_last_axis': squeeze_last_axis,
        'expand_last_axis': expand_last_axis,
        'se_block': se_block,
        'residual_se_cnn_block': residual_se_cnn_block,
        'attention_layer': attention_layer,
    }
    model = load_model(PRETRAINED_DIR / "gesture_two_branch_mixup.h5",
                       compile=False, custom_objects=custom_objs)
    print("  Model, scaler, feature_cols, pad_len loaded – ready for evaluation")


# # 예측
# y_pred = model.predict(X_val)
# y_pred_class = np.argmax(y_pred, axis=1)

# # 라벨 인코더로 문자열 변환
# gesture_pred = le.inverse_transform(y_pred_class)
# gesture_true = le.inverse_transform(y_val.argmax(axis=1))

# # id 부여
# ids = list(range(len(y_val)))

# # DataFrame 준비
# submission_df = pd.DataFrame({'id': ids, 'gesture': gesture_pred})
# solution_df = pd.DataFrame({'id': ids, 'gesture': gesture_true})

# # 점수 계산
# final_score = score(solution_df, submission_df, row_id_column_name='id')
# print("검증 데이터 점수:", final_score)


    # from cmi_2025_metric_copy_for_import import CompetitionMetric
    # preds_val = model.predict(X_val).argmax(1)
    # true_val_int  = y_val.argmax(1)

    # h_f1 = CompetitionMetric().calculate_hierarchical_f1(
    #     pd.DataFrame({'gesture': le.classes_[true_val_int]}),
    #     pd.DataFrame({'gesture': le.classes_[preds_val]}))
    # print("Hold‑out H‑F1 =", round(h_f1, 4))


def extract_periodicity_features(signal, max_lag=100):
    autocorr = np.correlate(signal, signal, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    peaks = find_peaks(autocorr[:max_lag])[0]
    return {
        'dominant_period': peaks[0] if len(peaks) > 0 else 0,
        'periodicity_strength': np.max(autocorr[1:max_lag]) if len(autocorr) > 1 else 0,
        'num_significant_periods': len(peaks)
    }

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_seq = sequence.to_pandas()

    # 기본 IMU 파생 특징
    df_seq['acc_mag'] = np.sqrt(df_seq['acc_x']**2 + df_seq['acc_y']**2 + df_seq['acc_z']**2)
    df_seq['rot_angle'] = 2 * np.arccos(df_seq['rot_w'].clip(-1, 1))
    df_seq['acc_mag_jerk'] = df_seq['acc_mag'].diff().fillna(0)
    df_seq['rot_angle_vel'] = df_seq['rot_angle'].diff().fillna(0)

    # 중력 제거 가속도
    if all(col in df_seq.columns for col in ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']):
        acc_data_seq = df_seq[['acc_x', 'acc_y', 'acc_z']]
        rot_data_seq = df_seq[['rot_w', 'rot_x', 'rot_y', 'rot_z']]
        linear_accel_seq_arr = remove_gravity_from_acc(acc_data_seq, rot_data_seq)
        df_seq['linear_acc_x'], df_seq['linear_acc_y'], df_seq['linear_acc_z'] = linear_accel_seq_arr.T
    else:
        for axis in ['x', 'y', 'z']:
            df_seq[f'linear_acc_{axis}'] = df_seq.get(f'acc_{axis}', 0)

    df_seq['linear_acc_mag'] = np.sqrt(df_seq['linear_acc_x']**2 + df_seq['linear_acc_y']**2 + df_seq['linear_acc_z']**2)
    df_seq['linear_acc_mag_jerk'] = df_seq['linear_acc_mag'].diff().fillna(0)

    # TOF 처리 및 periodicity feature
    for i in range(1, 6):
        pixel_cols = [f"tof_{i}_v{p}" for p in range(64)]
        if not all(col in df_seq.columns for col in pixel_cols):
            for stat in ['mean', 'std', 'min', 'max']:
                df_seq[f'tof_{i}_{stat}'] = 0
                for p_col in ['dominant_period', 'periodicity_strength', 'num_significant_periods']:
                    df_seq[f'{p_col}_tof_{i}_{stat}'] = 0
            continue

        sensor_data = df_seq[pixel_cols].replace(-1, np.nan)
        df_seq[f'tof_{i}_mean'] = sensor_data.mean(axis=1)
        df_seq[f'tof_{i}_std'] = sensor_data.std(axis=1)
        df_seq[f'tof_{i}_min'] = sensor_data.min(axis=1)
        df_seq[f'tof_{i}_max'] = sensor_data.max(axis=1)

        doms, strs, nums = [], [], []
        for col in pixel_cols:
            sig = df_seq[col].values
            res = extract_periodicity_features(sig)
            doms.append(res['dominant_period'])
            strs.append(res['periodicity_strength'])
            nums.append(res['num_significant_periods'])

        for stat, arr_func in zip(['mean', 'std', 'min', 'max'],
                             [np.nanmean, np.nanstd, np.nanmin, np.nanmax]):
            df_seq[f'dominant_period_tof_{i}_{stat}'] = arr_func(doms)
            df_seq[f'periodicity_strength_tof_{i}_{stat}'] = arr_func(strs)
            df_seq[f'num_significant_periods_tof_{i}_{stat}'] = arr_func(nums)

    # THM 처리
    for i in range(1, 6):
        col = f'thm_{i}'
        if col in df_seq.columns:
            df_seq[f'{col}_diff'] = df_seq[col].diff().fillna(0)
            df_seq[f'{col}_rollmean'] = df_seq[col].rolling(5, min_periods=1).mean()
            df_seq[f'{col}_rollstd'] = df_seq[col].rolling(5, min_periods=1).std().fillna(0)
        else:
            df_seq[f'{col}_diff'] = 0
            df_seq[f'{col}_rollmean'] = 0
            df_seq[f'{col}_rollstd'] = 0

    # IMU periodicity 추가
    imu_axes = ['acc_x', 'acc_y', 'acc_z', 'rot_x', 'rot_y', 'rot_z', 'rot_w']
    for axis in imu_axes:
        if axis in df_seq.columns:
            res = extract_periodicity_features(df_seq[axis].values)
            df_seq[f'dominant_period_{axis}'] = res['dominant_period']
            df_seq[f'periodicity_strength_{axis}'] = res['periodicity_strength']
            df_seq[f'num_significant_periods_{axis}'] = res['num_significant_periods']
        else:
            df_seq[f'dominant_period_{axis}'] = 0
            df_seq[f'periodicity_strength_{axis}'] = 0
            df_seq[f'num_significant_periods_{axis}'] = 0

    # 센서간 범위 비교
    if 'tof_range_across_sensors' in final_feature_cols:
        tof_means = [f'tof_{i}_mean' for i in range(1, 6) if f'tof_{i}_mean' in df_seq.columns]
        thm_vals = [f'thm_{i}' for i in range(1, 6) if f'thm_{i}' in df_seq.columns]

        if tof_means:
            df_seq['tof_range_across_sensors'] = df_seq[tof_means].max(axis=1) - df_seq[tof_means].min(axis=1)
            df_seq['tof_std_across_sensors'] = df_seq[tof_means].std(axis=1)
        else:
            df_seq['tof_range_across_sensors'] = 0
            df_seq['tof_std_across_sensors'] = 0

        if thm_vals:
            df_seq['thm_range_across_sensors'] = df_seq[thm_vals].max(axis=1) - df_seq[thm_vals].min(axis=1)
            df_seq['thm_std_across_sensors'] = df_seq[thm_vals].std(axis=1)
        else:
            df_seq['thm_range_across_sensors'] = 0
            df_seq['thm_std_across_sensors'] = 0

    # 삭제할 의미 없는 0 컬럼 리스트 정의
    zero_cols = [
        'dominant_period_thm_1', 'num_significant_periods_thm_1', 'periodicity_strength_thm_1',
        'dominant_period_thm_2', 'num_significant_periods_thm_2', 'periodicity_strength_thm_2',
        'dominant_period_thm_3', 'num_significant_periods_thm_3', 'periodicity_strength_thm_3',
        'dominant_period_thm_4', 'num_significant_periods_thm_4', 'periodicity_strength_thm_4',
        'dominant_period_thm_5', 'num_significant_periods_thm_5', 'periodicity_strength_thm_5'
    ]
    # final_feature_cols 는 전역에서 정의된 리스트 (예: imu_cols + tof_cols + thm_cols)
    filtered_feature_cols = [col for col in final_feature_cols if col not in zero_cols]

    # 최종 입력 데이터프레임 생성 (필터링된 컬럼만 사용)
    df_seq_final_features = pd.DataFrame(index=df_seq.index)
    for col in filtered_feature_cols:
        if col in df_seq.columns:
            df_seq_final_features[col] = df_seq[col]
        else:
            print(f"CRITICAL ERROR IN PREDICT: Missing feature '{col}' — filling with 0")
            df_seq_final_features[col] = 0

    mat_unscaled = df_seq_final_features.ffill().bfill().fillna(0).values.astype('float32')
    mat_scaled = scaler.transform(mat_unscaled)
    pad_input = pad_sequences([mat_scaled], maxlen=pad_len, padding='post', truncating='post', dtype='float32')

    idx = int(model.predict(pad_input, verbose=0).argmax(1)[0])
    return str(gesture_classes[idx])


# if TRAIN:
#   test_df = pd.read_csv(RAW_DIR / "test.csv")
#   test_dem_df = pd.read_csv(RAW_DIR / "test_demographics.csv")


# subject_id = test_df['subject'].iloc[0]  # 예: 첫 번째 시퀀스의 subject 추출

# # 시퀀스 데이터에서 해당 subject 데이터만 추출 (또는 sequence_id 기준이 다르면 sequence_id로)
# test_seq_df = pl.from_pandas(test_df[test_df['subject'] == subject_id])

# # demographics에서 subject 기준 추출
# test_demo_df = pl.from_pandas(test_dem_df[test_dem_df['subject'] == subject_id])

# # predict 호출
# result = predict(test_seq_df, test_demo_df)
# print(result)


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




