import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch import nn
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torchvision.models import resnet50, efficientnet_b5, resnet101
from sklearn.utils import class_weight
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedGroupKFold
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import f1_score
from transformers import BertConfig, BertModel
import copy
import cv2
import re
import librosa
from typing import Tuple, Optional, Sequence
from contextlib import nullcontext
from PIL import Image
import joblib
import random
import os
import polars as pl
from collections import defaultdict, deque
from scipy.spatial.transform import Rotation as R
import kaggle_evaluation.cmi_inference_server


train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
train_dem = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
test_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
test_dem = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')


set(train_df.columns) - set(test_df.columns)


train_only = train_df[list(set(train_df.columns) - set(test_df))]


train_only


train_df.head()


missing_ratio = train_df.isnull().mean()

missing_ratio.sort_values(ascending=False)


missing_ratio = test_df.isnull().mean()

missing_ratio.sort_values(ascending=False)


train_df.groupby('sequence_id')['gesture'].unique()


lens = train_df.groupby('sequence_id')['sequence_counter'].count()
pad_ratio = 1 - (lens.sum() / (len(lens)*max(lens)))
print(f"Pad waste: {pad_ratio:.1%}")


train_df.info()


test_df.info()


train_df['gesture'].unique()


lens = [len(x) for _, x in train_df.groupby('sequence_id')['gesture']]
pad_len = int(np.percentile(lens, 95))
print(f"Pad length: {pad_len}")


class CFG:
    # -------- data / training --------
    SEED: int = 17
    WIN_LEN: int = 130
    STRIDE: int = 100
    BATCH: int = 64
    EPOCHS: int = 40
    
    # -------- optimizer --------
    LR: float = 1e-3
    WD: float = 1e-4
    USE_AMP: bool = True # mixed precision training
    
    
    # -------- scheduler --------
    WARMUP_FRAC: float = 0.2
    ETA_MIN: float = 1e-6 # min LR in cosine
    
    # -------- grad clip --------
    GRAD_CLIP: float = 1.0
    
    # Core dims
    D_MODEL = 256
    N_HEAD = 16
    N_LAYER = 6
    SEQ_LEN = 1024 # max window length (for FreqMix)
    
    # Inception/TCN
    INC_KS = (3, 5, 7)
    INC_DILATIONS = (1, 2, 4)
    INC_BOTTLENECK = 128
    INC_DROP = 0.10
    USE_ECA = True

    # MiniRocket (IMU tower)
    USE_MINIROCKET = True
    MR_DILS    = (1, 2, 4, 8, 16, 32, 64)
    MR_N_BIAS  = 9
    MR_DROP    = 0.05
    MR_MAX_WINDOWS = 7000
        
    # Encoder/Pooling/Head
    POOL_NQ = 6
    POOL_OUT = 512
    POOL_DROP = 0.10
    HEAD_HIDDEN = (1024, 512,)
    HEAD_DROPS = (0.20, 0.10,)
    ENC_DROP = 0.10
    DROPPATH_P = 0.10

    # -------- K-fold --------
    N_SPLIT:    int   = 5

    # -------- classes --------
    N_CLASSES:  int   = 18
    TARGET_IDS: set   = set(range(N_CLASSES))


def set_seed(seed=CFG.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # reproducibility (chậm hơn một chút)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()


TARGETS = [
    'Above ear - pull hair',
    'Cheek - pinch skin',
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Neck - pinch skin',
    'Neck - scratch',
]
NON_TARGETS = [
    'Write name on leg',
    'Wave hello',
    'Glasses on/off',
    'Text on phone',
    'Write name in air',
    'Feel around in tray and pull out an object',
    'Scratch knee/leg skin',
    'Pull air toward your face',
    'Drink from bottle/cup',
    'Pinch knee/leg skin',
]
ALL_CLASSES = set(TARGETS) | set(NON_TARGETS)

def _score(
    sol: pd.DataFrame,
    sub: pd.DataFrame,
) -> tuple[float, float, float]:
    bad = set(sub['gesture']) - ALL_CLASSES

    y_true = sol['gesture'].values
    y_pred = sub['gesture'].values

    is_t_true = np.isin(y_true, TARGETS)
    is_t_pred = np.isin(y_pred, TARGETS)
    f1_bin = f1_score(is_t_true, is_t_pred, zero_division=0)

    def multiclass(a):
        return np.where(np.isin(a, TARGETS), a, 'non_target')

    f1_mac = f1_score(
        multiclass(y_true),
        multiclass(y_pred),
        average='macro',
        zero_division=0
    )

    hier = 0.5 * f1_bin + 0.5 * f1_mac
    return hier, f1_bin, f1_mac


train_df = train_df.dropna(subset = ['acc_x', 'acc_y', 'acc_z', 'rot_x','rot_y','rot_z','rot_w'])


def remove_gravity_from_acc(df):
    acc_vals = df[['acc_x','acc_y','acc_z']].values
    quat_vals = df[['rot_x','rot_y','rot_z','rot_w']].values

    N = len(df)
    linear_acc = np.zeros((N,3), dtype=float)
    gravity_world = np.array([0, 0, 9.81])

    for i in range(N):
        q = quat_vals[i]
        rot = R.from_quat(q)
        g_sensor = rot.apply(gravity_world, inverse=True)
        linear_acc[i] = acc_vals[i] - g_sensor

    return linear_acc


def calc_angular_velocity(df: pd.DataFrame, delta_t: float = 1/200) -> pd.DataFrame:
    quat_vals = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_velocity = np.zeros((len(quat_vals), 3))

    for i in range(0, len(quat_vals) - 1):
        q1 = quat_vals[i + 1]
        q2 = quat_vals[i]
        rot1 = R.from_quat(q1)
        rot2 = R.from_quat(q2)
        delta_rot = rot2.inv() * rot1
        angular_velocity[i] = delta_rot.as_rotvec() / delta_t

    angular_velocity[-1] = angular_velocity[-2]

    return pd.DataFrame(
        angular_velocity,
        columns=['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'],
        index=df.index
    )


def calc_angular_dist(df: pd.DataFrame) -> pd.DataFrame:
    quat_vals = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    angular_dist = np.zeros((len(quat_vals), 3))

    for i in range(0, len(quat_vals) - 1):
        q1 = quat_vals[i + 1]
        q2 = quat_vals[i]
        rot1 = R.from_quat(q1)
        rot2 = R.from_quat(q2)
        delta_rot = rot2.inv() * rot1
        angular_dist[i] = np.linalg.norm(delta_rot.as_rotvec())

    angular_dist[-1] = angular_dist[-2]

    return pd.DataFrame(
        angular_dist,
        columns=['angular_dist_x', 'angular_dist_y', 'angular_dist_z'],
        index=df.index
    )


def create_spectrogram(signal, sr=200, n_fft=64, hop_length=32):
    D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length, center=True)
    S = np.abs(D)**2
    log_S = librosa.power_to_db(S, ref=np.max)
    return log_S.astype(np.float32)

def normalize_spectrogram(spec):
    spec_min, spec_max = spec.min(), spec.max()
    denom = spec_max - spec_min
    eps = 1e-6
    spec_norm = (spec - spec_min) / (denom + eps)
    spec_uint8 = np.clip(np.round(spec_norm * 255), 0, 255).astype(np.uint8)
    return spec

def resize_spectrogram_image(spec, size=(150, 150)):
    img = Image.fromarray(spec)
    img = img.resize(size)
    return img


class DataPreprocessing:
    def __init__(self, df,
                 *,
                 tof_region_modes=(4, 8, 16),
                 include_raw_tof=True,
                 mr_include_diff=True,
                 imu_presence_ratio_threshold=0.02,
                 thm_presence_ratio_threshold=0.05,
                 tof_presence_ratio_threshold=0.15):
        """
        include_raw_tof=True  -> dùng cả raw 64*5 kênh TOF + region stats
        include_raw_tof=False -> chỉ dùng region stats để giảm chiều

        mr_include_diff=True  -> X_roc (cho MiniRocket) sẽ gồm cả sai phân bậc 1
        imu_presence_ratio_threshold: tỷ lệ tối thiểu (per-seq) để coi IMU là "có mặt"
        """
        self.df = df.copy()

        # ----------------- Cột theo tiền tố -----------------
        imu_pref = ('acc_', 'rot_', 'angular_velocity_')
        self.imu_cols = [c for c in df.columns if c.startswith(imu_pref)]
        self.thm_cols = [c for c in df.columns if c.startswith('thm_')]

        # --- TOF RAW: đủ 5 cảm biến nếu có ---
        def _tof_raw_cols_present(i):
            return [c for c in [f"tof_{i}_v{p}" for p in range(64)] if c in df.columns]

        self.tof_cols = []
        for i in range(1, 6):
            self.tof_cols.extend(_tof_raw_cols_present(i))

        # ----------------- Cấu hình -----------------
        self.tof_region_modes = tuple(int(m) for m in tof_region_modes)
        self.include_raw_tof = bool(include_raw_tof)
        self.mr_include_diff = bool(mr_include_diff)

        self.imu_presence_ratio_threshold = float(imu_presence_ratio_threshold)
        self.thm_presence_ratio_threshold = float(thm_presence_ratio_threshold)
        self.tof_presence_ratio_threshold = float(tof_presence_ratio_threshold)

        # encoders / imputers / scalers
        self.label_encoder = LabelEncoder()

        # imputers (giữ để tương thích — không khuyến nghị dùng sau sentinel)
        self.knn_imu = KNNImputer(n_neighbors=5)
        self.sim_thm = SimpleImputer(missing_values=np.nan, strategy='mean')
        self.sim_tof = SimpleImputer(missing_values=np.nan, strategy='mean')

        self.scale_imu = StandardScaler()
        self.scale_thm = StandardScaler()
        self.scale_tof = StandardScaler()

        # ===== Missing strategy =====
        self.use_seq_fbfill = True
        self.use_sentinel_fill = True
        # hệ số sentinel theo modality (âm lớn = - max_abs * ratio)
        self.sentinel_ratio = {"imu": 0.2, "thm": 0.3, "tof": 0.3}
        self.keep_imputers_after_sentinel = False

        self._sentinel_values = {"imu": None, "thm": None, "tof": None}
        self.classes_ = None

    # ---------------------- TOF region stats ----------------------
    def _build_tof_region_stats(self, df):
        feat = {}
        for i in range(1, 6):
            base_cols = [f"tof_{i}_v{p}" for p in range(64)]
            if not all(c in df.columns for c in base_cols):
                continue

            seg = df[base_cols].to_numpy(dtype=np.float32)  # [N,64]
            for mode in self.tof_region_modes:
                if 64 % mode != 0:
                    continue
                rsize = 64 // mode
                for r in range(mode):
                    region = seg[:, r * rsize:(r + 1) * rsize]  # [N,rsize]
                    base = f"tof{i}_reg{mode}_{r}"
                    feat[f"{base}_mean"] = region.mean(axis=1)
                    feat[f"{base}_std"]  = region.std(axis=1)
                    feat[f"{base}_min"]  = region.min(axis=1)
                    feat[f"{base}_max"]  = region.max(axis=1)

        if not feat:
            region_df = pd.DataFrame(index=df.index); region_cols = []
        else:
            region_df = pd.DataFrame(feat, index=df.index)
            region_cols = list(region_df.columns)
        return region_df, region_cols

    def _current_tof_feature_cols(self, df):
        region_df, region_cols = self._build_tof_region_stats(df)
        feat_cols = []
        if self.include_raw_tof:
            feat_cols.extend([c for c in self.tof_cols if c in df.columns])
        feat_cols.extend(region_cols)
        return region_df, feat_cols

    # === helper: ffill/bfill + sentinel per modality ===
    def _fill_modality_missing(self, df, cols, key):
        if not cols:
            return
        # Inf -> NaN
        df.loc[:, cols] = df[cols].replace([np.inf, -np.inf], np.nan)

        # ffill/bfill theo sequence
        if self.use_seq_fbfill:
            df.loc[:, cols] = (
                df.groupby('sequence_id', group_keys=False)[cols]
                  .ffill().bfill()
            )
        # sentinel âm lớn cho phần vẫn còn NaN
        if self.use_sentinel_fill:
            arr = df[cols].to_numpy(np.float32, copy=False)
            if arr.size == 0:
                return
            max_abs = np.nanmax(np.abs(arr))
            if not np.isfinite(max_abs) or max_abs == 0:
                max_abs = 1.0
            nan_value = -float(max_abs) * float(self.sentinel_ratio[key])
            df.loc[:, cols] = df[cols].fillna(nan_value)
            self._sentinel_values[key] = nan_value

    # ---------------------- Orientation helpers ----------------------
    @staticmethod
    def _tilt_cos_from_quat(qxyzw):
        """
        cos(tilt) = z-component của trục Z thiết bị trong world frame.
        q in (x,y,z,w) order as in df columns (rot_x, rot_y, rot_z, rot_w).
        """
        rots = R.from_quat(qxyzw)  # (N,4)
        # thiết bị trục Z (0,0,1) chiếu sang world
        z_axes_world = rots.apply(np.tile(np.array([0., 0., 1.]), (len(qxyzw), 1)))
        return z_axes_world[:, 2]  # cos góc so với trục Z thế giới

    # ---------------------- fit/transform ----------------------
    def fit(self, all_df):
        if 'gesture' not in self.df.columns:
            raise RuntimeError("fit() expects the stored df to contain 'gesture' column.")

        self.df = self.df.copy()
        self.label_encoder.fit(all_df['gesture'])
        self.classes_ = self.label_encoder.classes_

        # vẫn fit các imputers để phòng khi bật keep_imputers_after_sentinel
        if self.imu_cols:
            self.knn_imu.fit(self.df[self.imu_cols])
        if self.thm_cols:
            self.sim_thm.fit(self.df[self.thm_cols])
        if self.tof_cols:
            self.sim_tof.fit(self.df[self.tof_cols])
        return self

    def transform(self, df, is_train=0):
        """
        Returns:
          X_imu, X_thm, X_tof, X_roc,
          y, subjects, type_seqs, seq_ids,
          masks_imu, masks_thm, masks_tof,
          sensor_flags
        """
        df = df.copy()
        if self.classes_ is None:
            raise RuntimeError('Call .fit() before transform().')

        # --- presence (RAW) ---
        presence_imu = df[self.imu_cols].notna().any(axis=1) if self.imu_cols else pd.Series(False, index=df.index)
        presence_thm = df[self.thm_cols].notna().any(axis=1) if self.thm_cols else pd.Series(False, index=df.index)
        presence_tof = df[self.tof_cols].notna().any(axis=1) if self.tof_cols else pd.Series(False, index=df.index)

        # --- sequence-level presence (ngưỡng cập nhật) ---
        seq_presence_local = {}
        thm_thr = self.thm_presence_ratio_threshold
        tof_thr = self.tof_presence_ratio_threshold
        imu_thr = self.imu_presence_ratio_threshold

        for sid, g in df.groupby('sequence_id'):
            # IMU
            if self.imu_cols:
                nonnull_imu = g[self.imu_cols].notna().any(axis=1).sum()
                ratio_imu = nonnull_imu / max(1, len(g))
                has_imu = (nonnull_imu > 0) and (ratio_imu >= imu_thr)
            else:
                has_imu = False
            # THM
            if self.thm_cols:
                nonnull_thm = g[self.thm_cols].notna().any(axis=1).sum()
                ratio_thm = nonnull_thm / max(1, len(g))
                has_thm = (nonnull_thm > 0) and (ratio_thm >= thm_thr)
            else:
                has_thm = False
            # TOF
            if self.tof_cols:
                nonnull_tof = g[self.tof_cols].notna().any(axis=1).sum()
                ratio_tof = nonnull_tof / max(1, len(g))
                has_tof = (nonnull_tof > 0) and (ratio_tof >= tof_thr)
            else:
                has_tof = False

            seq_presence_local[sid] = {'imu': bool(has_imu), 'thm': bool(has_thm), 'tof': bool(has_tof)}

        # ====== Missing value strategy ======
        self._fill_modality_missing(df, self.imu_cols, "imu")
        self._fill_modality_missing(df, self.thm_cols, "thm")
        self._fill_modality_missing(df, self.tof_cols, "tof")

        if self.keep_imputers_after_sentinel:
            if self.imu_cols:
                df[self.imu_cols] = self.knn_imu.transform(df[self.imu_cols])
            if self.thm_cols:
                df[self.thm_cols] = self.sim_thm.transform(df[self.thm_cols])
            if self.tof_cols:
                df[self.tof_cols] = self.sim_tof.transform(df[self.tof_cols])

        # ----------------- IMU feature engineering -----------------
        # 1) Linear acceleration (sensor frame) via your gravity removal
        linear_acc = remove_gravity_from_acc(df)  # (N,3)
        df = df.join(pd.DataFrame({
            'liner_acc_x': linear_acc[:, 0],
            'liner_acc_y': linear_acc[:, 1],
            'liner_acc_z': linear_acc[:, 2]
        }, index=df.index))

        # 2) Magnitudes & orientation invariants
        df['acc_mag'] = np.sqrt(df['liner_acc_x']**2 + df['liner_acc_y']**2 + df['liner_acc_z']**2)
        df['acc_mag_median'] = df.groupby('sequence_id')['acc_mag'].transform('median')

        # Rotation angle from quaternion (0..pi)
        df['rot_ang'] = 2 * np.arccos(np.clip(df['rot_w'].astype(float), -1.0, 1.0))

        # Angular velocity (rad/s) from quaternion delta (không truyền include_groups)
        angular_velocity_df = (
            df.groupby('sequence_id', group_keys=False)
              .apply(calc_angular_velocity, include_groups=False)
        )
        df = df.join(angular_velocity_df)

        # Angular distance (replicate to 3 cols for backward-compat)
        angular_dist_df = calc_angular_dist(df)  # Nx3 (the norm replicated)
        df = df.join(angular_dist_df)

        # Gyro magnitude + median
        df['angular_velocity_mag'] = np.sqrt(
            df['angular_velocity_x']**2 + df['angular_velocity_y']**2 + df['angular_velocity_z']**2
        )
        df['angular_velocity_mag_median'] = df.groupby('sequence_id')['angular_velocity_mag'].transform('median')

        # 3) Tilt cos wrt world Z (orientation-invariant cue) — no try/except
        quat_vals = df[['rot_x','rot_y','rot_z','rot_w']].to_numpy(np.float64, copy=False)
        df['tilt_cos'] = self._tilt_cos_from_quat(quat_vals)

        # 4) Temporal derivatives (jerk / angular accel)
        delta_t = 1/200
        for col in ['liner_acc_x', 'liner_acc_y', 'liner_acc_z']:
            df[f'{col}_jerk'] = df.groupby('sequence_id')[col].diff().fillna(0) / delta_t

        for col in ['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z']:
            df[f'{col}_jerk'] = df.groupby('sequence_id')[col].diff().fillna(0) / delta_t

        df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0) / delta_t
        df['rot_ang_vel']  = df.groupby('sequence_id')['rot_ang'].diff().fillna(0) / delta_t
        df['angular_velocity_mag_jerk'] = df.groupby('sequence_id')['angular_velocity_mag'].diff().fillna(0) / delta_t

        # ----------------- TOF region features -----------------
        region_df, tof_feature_cols = self._current_tof_feature_cols(df)
        df_feat = pd.concat([df, region_df], axis=1)

        # ----------------- labels -----------------
        has_label = 'gesture' in df_feat.columns
        if has_label and 'gesture_lbl' not in df_feat.columns:
            df_feat = df_feat.copy()
            df_feat['gesture_lbl'] = self.label_encoder.transform(df_feat['gesture'])

        # ----------------- containers -----------------
        imu_arrays, thm_arrays, tof_arrays = [], [], []
        roc_arrays = []  # for MiniRocket input (RO(C)ket)
        labels, subjects, type_seqs, seq_ids = [], [], [], []
        masks_imu, masks_thm, masks_tof = [], [], []
        sensor_flags = []

        # ----------------- window hoá per sequence -----------------
        # IMU feature set (nâng cấp, thêm tilt_cos)
        imu_cols_block = [
            # raw acc
            'acc_x', 'acc_y', 'acc_z',
            # linear acc (sau khử gravity)
            'liner_acc_x', 'liner_acc_y', 'liner_acc_z',
            'acc_mag', 'rot_ang', 'tilt_cos',
            # gyro
            'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z',
            # quaternion
            'rot_x', 'rot_y', 'rot_z', 'rot_w',
            # derivatives
            'liner_acc_x_jerk', 'liner_acc_y_jerk', 'liner_acc_z_jerk',
            'angular_velocity_x_jerk', 'angular_velocity_y_jerk', 'angular_velocity_z_jerk',
            'angular_velocity_mag', 'acc_mag_jerk',
            'rot_ang_vel', 'angular_velocity_mag_jerk',
            'acc_mag_median', 'angular_velocity_mag_median',
            # backward-compat (angular_dist replicated)
            'angular_dist_x', 'angular_dist_y', 'angular_dist_z'
        ]
        # ROC base channels (clean IMU) -> (acc_lin + gyro)
        roc_base_cols = [
            'liner_acc_x', 'liner_acc_y', 'liner_acc_z',
            'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'
            'acc_mag',                 
            'angular_velocity_mag', 
            'rot_ang_vel',         
            'tilt_cos',    
        ]

        for sid, sub_df in df_feat.groupby('sequence_id'):
            sub_df = sub_df.sort_values('sequence_counter')
            idx = sub_df.index

            # presence per-row
            pres_imu_sub = presence_imu.loc[idx].to_numpy(dtype=np.float32) if len(presence_imu) > 0 else np.zeros(len(sub_df), dtype=np.float32)
            pres_thm_sub = presence_thm.loc[idx].to_numpy(dtype=np.float32) if len(presence_thm) > 0 else np.zeros(len(sub_df), dtype=np.float32)
            pres_tof_sub = presence_tof.loc[idx].to_numpy(dtype=np.float32) if len(presence_tof) > 0 else np.zeros(len(sub_df), dtype=np.float32)

            # seq-level presence
            thm_present = seq_presence_local[sid]['thm']
            tof_present = seq_presence_local[sid]['tof']

            # arrays
            imu_exist_cols = [c for c in imu_cols_block if c in sub_df.columns]
            imu_arr = sub_df[imu_exist_cols].to_numpy(np.float32)

            thm_arr = sub_df[self.thm_cols].to_numpy(np.float32) if self.thm_cols else np.empty((len(sub_df), 0), np.float32)

            if tof_feature_cols:
                cols_exist = [c for c in tof_feature_cols if c in sub_df.columns]
                tof_arr = sub_df[cols_exist].to_numpy(np.float32) if cols_exist else np.empty((len(sub_df), 0), np.float32)
            else:
                tof_arr = np.empty((len(sub_df), 0), np.float32)

            # --- MiniRocket input (IMU clean + diff) -> X_roc ---
            roc_cols_exist = [c for c in roc_base_cols if c in sub_df.columns]
            roc_base = sub_df[roc_cols_exist].to_numpy(np.float32) if roc_cols_exist else np.empty((len(sub_df), 0), np.float32)

            if self.mr_include_diff and roc_base.size:
                roc_diff = np.vstack([np.zeros((1, roc_base.shape[1]), np.float32),
                                      np.diff(roc_base, axis=0).astype(np.float32)])
                roc_full = np.concatenate([roc_base, roc_diff], axis=1)  # [T, 12]
            else:
                roc_full = roc_base  # [T, 6]

            # --- labels/meta ---
            label = sub_df['gesture_lbl'].iloc[0] if has_label else None
            subject = sub_df['subject'].iloc[0]
            type_seq = sub_df['sequence_type'].iloc[0] if 'sequence_type' in sub_df.columns else None

            # --- windowing ---
            T = len(imu_arr)
            if T <= CFG.WIN_LEN:
                starts = [0]
            else:
                starts = list(range(0, T - CFG.WIN_LEN + 1, CFG.STRIDE))
                if starts[-1] != T - CFG.WIN_LEN:
                    starts.append(T - CFG.WIN_LEN)

            for s in starts:
                e = min(T, s + CFG.WIN_LEN)

                imu_seg = imu_arr[s:e]
                thm_seg = thm_arr[s:e]
                tof_seg = tof_arr[s:e] if tof_feature_cols else np.empty((e - s, 0), np.float32)
                roc_seg = roc_full[s:e] if roc_full.size else np.empty((e - s, 0), np.float32)

                # zero-out theo sequence-level presence
                if not thm_present:
                    thm_seg = np.zeros((e - s, thm_seg.shape[1]), np.float32) if thm_seg.size else np.zeros((e - s, 0), np.float32)
                if not tof_present:
                    tof_seg = np.zeros((e - s, tof_seg.shape[1]), np.float32) if tof_seg.size else np.zeros((e - s, 0), np.float32)

                imu_mask_seg = pres_imu_sub[s:e].astype(np.float32)
                thm_mask_seg = (pres_thm_sub[s:e].astype(np.float32)
                                if self.thm_cols else np.zeros((e - s,), np.float32))
                tof_mask_seg = (pres_tof_sub[s:e].astype(np.float32)
                                if self.tof_cols else np.zeros((e - s,), np.float32))

                # pad tới WIN_LEN
                cur_len = e - s
                if cur_len < CFG.WIN_LEN:
                    pad_len = CFG.WIN_LEN - cur_len

                    imu_seg = np.vstack([imu_seg, np.zeros((pad_len, imu_seg.shape[1]), np.float32)])

                    if thm_seg.size:
                        thm_seg = np.vstack([thm_seg, np.zeros((pad_len, thm_seg.shape[1]), np.float32)])
                    else:
                        thm_seg = np.zeros((CFG.WIN_LEN, 0), np.float32)

                    if tof_feature_cols:
                        if tof_seg.size:
                            tof_seg = np.vstack([tof_seg, np.zeros((pad_len, tof_seg.shape[1]), np.float32)])
                        else:
                            tof_seg = np.zeros((CFG.WIN_LEN, 0), np.float32)
                    else:
                        tof_seg = np.zeros((CFG.WIN_LEN, 0), np.float32)

                    if roc_seg.size:
                        roc_seg = np.vstack([roc_seg, np.zeros((pad_len, roc_seg.shape[1]), np.float32)])
                    else:
                        roc_seg = np.zeros((CFG.WIN_LEN, 0), np.float32)

                    imu_mask_seg = np.concatenate([imu_mask_seg, np.zeros(pad_len, np.float32)])
                    thm_mask_seg = np.concatenate([thm_mask_seg, np.zeros(pad_len, np.float32)])
                    tof_mask_seg = np.concatenate([tof_mask_seg, np.zeros(pad_len, np.float32)])

                imu_arrays.append(imu_seg)
                thm_arrays.append(thm_seg)
                tof_arrays.append(tof_seg)
                roc_arrays.append(roc_seg)

                seq_ids.append(sid)
                subjects.append(subject)
                masks_imu.append(imu_mask_seg.astype(bool))
                masks_thm.append(thm_mask_seg.astype(bool))
                masks_tof.append(tof_mask_seg.astype(bool))

                sensor_flags.append([bool(seq_presence_local[sid]['imu']),
                                     bool(thm_present),
                                     bool(tof_present)])
                if has_label:
                    type_seqs.append(type_seq)
                    labels.append(label)

        # ----------------- Stack & Scale -----------------
        if len(imu_arrays) == 0:
            raise RuntimeError("No windows were produced (check sequence lengths / CFG).")

        X_imu = np.stack(imu_arrays)  # [S, T, F_imu]
        X_thm = (np.stack(thm_arrays) if any(a.size for a in thm_arrays)
                 else np.empty((X_imu.shape[0], X_imu.shape[1], 0), np.float32))
        X_tof = (np.stack(tof_arrays) if any(a.size for a in tof_arrays)
                 else np.empty((X_imu.shape[0], X_imu.shape[1], 0), np.float32))
        X_roc = (np.stack(roc_arrays) if any(a.size for a in roc_arrays)
                 else np.empty((X_imu.shape[0], X_imu.shape[1], 0), np.float32))

        # IMU scaler
        S, Tt, F_imu = X_imu.shape
        X2d_imu = X_imu.reshape(-1, F_imu)
        if is_train == 1:
            self.scale_imu.fit(X2d_imu)
        X_imu = self.scale_imu.transform(X2d_imu).reshape(S, Tt, F_imu).astype(np.float32)

        # THM scaler
        if X_thm.size:
            _, _, F_thm = X_thm.shape
            X2d_thm = X_thm.reshape(-1, F_thm)
            if is_train == 1:
                self.scale_thm.fit(X2d_thm)
            X_thm = self.scale_thm.transform(X2d_thm).reshape(S, Tt, F_thm).astype(np.float32)
        else:
            X_thm = np.empty((S, Tt, 0), np.float32)

        # TOF scaler
        if X_tof.size:
            _, _, F_tof = X_tof.shape
            X2d_tof = X_tof.reshape(-1, F_tof)
            if is_train == 1:
                self.scale_tof.fit(X2d_tof)
            X_tof = self.scale_tof.transform(X2d_tof).reshape(S, Tt, F_tof).astype(np.float32)
        else:
            X_tof = np.empty((S, Tt, 0), np.float32)

        # X_roc (MiniRocket input) — KHÔNG scale (thường để thô cho PPV)

        y = np.array(labels, dtype=np.int64) if has_label else None
        subjects = np.array(subjects)
        type_seqs = np.array(type_seqs) if has_label else None
        seq_ids = np.array(seq_ids)

        masks_imu = np.stack(masks_imu).astype(bool)
        masks_thm = (np.stack(masks_thm).astype(bool)
                     if len(masks_thm) else np.zeros((X_imu.shape[0], X_imu.shape[1]), dtype=bool))
        masks_tof = (np.stack(masks_tof).astype(bool)
                     if len(masks_tof) else np.zeros((X_imu.shape[0], X_imu.shape[1]), dtype=bool))

        return (
            X_imu, X_thm, X_tof, X_roc,
            y, subjects, type_seqs, seq_ids,
            masks_imu, masks_thm, masks_tof,
            sensor_flags
        )


classes = ['Above ear - pull hair', 'Cheek - pinch skin',
       'Drink from bottle/cup', 'Eyebrow - pull hair',
       'Eyelash - pull hair',
       'Feel around in tray and pull out an object',
       'Forehead - pull hairline', 'Forehead - scratch', 'Glasses on/off',
       'Neck - pinch skin', 'Neck - scratch', 'Pinch knee/leg skin',
       'Pull air toward your face', 'Scratch knee/leg skin',
       'Text on phone', 'Wave hello', 'Write name in air',
       'Write name on leg']


train_df['gesture']


y_train = train_df['gesture']
y_train_collapsed = np.where(
    np.isin(y_train, TARGETS),
    y_train,
    "non_target"
)

n_tot       = len(y_train)
n_target    = np.isin(y_train, TARGETS).sum()
n_nontarget = n_tot - n_target

weight_binary = np.zeros(len(classes), dtype=float)
for i, c in enumerate(classes):
    if c in TARGETS:
        weight_binary[i] = 0.5 * n_tot / n_target
    else:
        weight_binary[i] = 0.5 * n_tot / n_nontarget

classes_macro = np.array(TARGETS + ["non_target"])
class_weights_macro = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=classes_macro,
    y=y_train_collapsed
)

weight_macro = np.zeros(len(classes), dtype=float)
for i, c in enumerate(classes):
    if c in TARGETS:
        idx = np.where(classes_macro == c)[0][0]
    else:
        idx = np.where(classes_macro == "non_target")[0][0]
    weight_macro[i] = class_weights_macro[idx]

alpha = 0.5
beta  = 0.5
weights = alpha * weight_binary + beta * weight_macro


weights


# X1, X2, y, subj, type_seq, seq_ids = dp_transform

# plt.show()
# for i in range(3):
#     plt.imshow(X2[1][:, :, i])
#     plt.show()


def _eca_kernel(channels: int, gamma: int = 2, b: int = 1) -> int:
    k = int(abs((math.log2(max(1, channels)) + b) / gamma))
    return max(3, k | 1)

class ECABlock1D(nn.Module):
    def __init__(self, d: int, k_size: Optional[int] = None):
        super().__init__()
        k = _eca_kernel(d) if k_size is None else k_size
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=k // 2, bias=False)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.mean(dim=1, keepdim=False).unsqueeze(1)
        y = torch.sigmoid(self.conv(y)).squeeze(1)
        return x * y.unsqueeze(1)


class FreqMix1D(nn.Module):
    def __init__(self, d: int, max_len: int, share_across_d: bool = False, use_fp32_fft: bool = True):
        super().__init__()
        self.use_fp32_fft = use_fp32_fft
        self.share = share_across_d
        Lr = int(max_len) // 2 + 1
        if self.share:
            self.mag   = nn.Parameter(torch.ones(1, Lr))
            self.phase = nn.Parameter(torch.zeros(1, Lr))
        else:
            self.mag   = nn.Parameter(torch.ones(d, Lr))
            self.phase = nn.Parameter(torch.zeros(d, Lr))
        self.gate = nn.Parameter(torch.tensor(0.5))
    @staticmethod
    def _resize_1d(x: torch.Tensor, target_len: int) -> torch.Tensor:
        if x.size(-1) == target_len: return x
        return F.interpolate(x.unsqueeze(0), size=target_len, mode="linear", align_corners=False).squeeze(0)
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, L, D = x.shape
        X = x.transpose(1, 2)
        comp = torch.float32 if (self.use_fp32_fft and X.dtype in (torch.float16, torch.bfloat16)) else X.dtype
        Z = torch.fft.rfft(X.to(comp), dim=2)
        Lr = Z.size(-1)
        mag = self.mag if self.share else self.mag[:D]
        ph  = self.phase if self.share else self.phase[:D]
        if mag.size(-1) != Lr:
            mag = self._resize_1d(mag, Lr)
            ph  = self._resize_1d(ph,  Lr)
        H = torch.polar(mag.clamp_min(0.0).to(comp), ph.to(comp)).unsqueeze(0).expand(B, -1, -1)
        Y = torch.fft.irfft(Z * H, n=L, dim=2).to(X.dtype).transpose(1, 2)
        out = x + torch.sigmoid(self.gate) * Y
        if padding_mask is not None:
            out = out * padding_mask.unsqueeze(-1).to(out.dtype)
        return out


class AttnPool(nn.Module):
    def __init__(self, d: int, n_q: int, h: int, out_dim: int, drop: float):
        super().__init__()
        self.q = nn.Parameter(torch.randn(1, n_q, d) * 0.02)
        self.mha = nn.MultiheadAttention(d, h, batch_first=True, dropout=drop)
        self.proj = nn.Linear(n_q * d, out_dim)
    def forward(self, x: torch.Tensor, padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B = x.size(0)
        q = self.q.expand(B, -1, -1)
        key_pad = (padding_mask == 0) if padding_mask is not None else None
        out, _ = self.mha(q, x, x, key_padding_mask=key_pad)
        return self.proj(out.reshape(B, -1))

class MeanStdPool(nn.Module):
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if mask is not None:
            w = mask.to(x.dtype).unsqueeze(-1)
            m = (x*w).sum(1) / w.sum(1).clamp_min(1e-6)
            v = ((x-m.unsqueeze(1))**2 * w).sum(1) / w.sum(1).clamp_min(1e-6)
        else:
            m = x.mean(1); v = x.var(1, unbiased=False)
        return torch.cat([m, torch.sqrt(v+1e-6)], -1)


class HBlend3Q(nn.Module):
    def __init__(self, d=128, hidden=256, temperature=1.0):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(3*d+6, hidden), nn.GELU(), nn.Linear(hidden, 3))
        self.t = float(temperature)
    def forward(self, f1, f2, f3, present, qual):
        logits = self.gate(torch.cat([f1, f2, f3, present.float(), qual.float()], 1)) / self.t
        logits = logits.masked_fill(~present.bool(), float('-inf'))
        alpha = torch.softmax(logits, 1)
        fused = alpha[:,0:1]*f1 + alpha[:,1:2]*f2 + alpha[:,2:3]*f3
        return fused, alpha


def _torch_quantile_linear(x: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    try:
        return torch.quantile(x, q, method="linear")
    except TypeError:
        return torch.quantile(x, q, interpolation="linear")

class MiniRocketPPV(nn.Module):
    def __init__(self, in_ch: int, dilations=(1,2,4,8,16,32), n_bias: int = 6, quantiles=None):
        super().__init__()
        self.in_ch   = int(in_ch)
        self.dils    = tuple(int(d) for d in dilations)
        self.n_bias  = int(n_bias)
        if quantiles is None:
            qs = torch.linspace(0.1, 0.9, steps=self.n_bias)
        else:
            qs = torch.as_tensor(quantiles, dtype=torch.float32)
            assert qs.numel() == self.n_bias
        self.register_buffer("qs", qs)
        w = torch.zeros(self.in_ch, 1, 2, dtype=torch.float32); w[:,0,0]=1.0; w[:,0,1]=-1.0
        self.register_buffer("w", w)
        self.register_buffer("biases", torch.zeros(self.in_ch, len(self.dils), self.n_bias))
        self.register_buffer("ones2", torch.ones(1, 1, 2))
        self.fitted = False
    @torch.no_grad()
    def fit_biases(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None, max_windows: int = 4000):
        X = X.to(self.w.device).to(torch.float32)
        N, L, C = X.shape
        assert C == self.in_ch
        if mask is None:
            mask = torch.ones(N, L, dtype=torch.bool, device=X.device)
        else:
            mask = mask.to(dtype=torch.bool, device=X.device)
        if N > max_windows:
            idx = torch.randperm(N, device=X.device)[:max_windows]
            X = X[idx]; mask = mask[idx]; N = X.size(0)
        X_ = X.transpose(1, 2)
        biases = torch.zeros(self.in_ch, len(self.dils), self.n_bias, device=X.device)
        for j, d in enumerate(self.dils):
            conv_out = F.conv1d(X_, self.w.to(dtype=X.dtype, device=X.device),
                                padding=d, dilation=d, groups=self.in_ch)
            m = mask.unsqueeze(1).to(dtype=conv_out.dtype, device=conv_out.device)
            den = F.conv1d(m, self.ones2.to(dtype=conv_out.dtype, device=conv_out.device),
                           padding=d, dilation=d)
            valid = (den >= 2 - 1e-6)
            if not valid.any(): continue
            z = conv_out.masked_select(valid.expand_as(conv_out))
            if z.numel() == 0: continue
            qs = _torch_quantile_linear(z, self.qs.to(dtype=z.dtype, device=z.device))
            biases[:, j, :] = qs.unsqueeze(0).expand(self.in_ch, -1)
        self.biases.copy_(biases)
        self.fitted = True
    def forward(self, X: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        assert self.fitted, "Call fit_biases(...) before forward."
        B, L, C = X.shape
        X_ = X.transpose(1, 2)
        if mask is None:
            mask = torch.ones(B, L, dtype=torch.bool, device=X.device)
        m = mask.unsqueeze(1).to(dtype=X.dtype, device=X.device)
        feats = []
        for j, d in enumerate(self.dils):
            conv_out = F.conv1d(X_, self.w.to(dtype=X.dtype, device=X.device),
                                padding=d, dilation=d, groups=C)
            den = F.conv1d(m, self.ones2.to(dtype=X.dtype, device=X.device),
                           padding=d, dilation=d)
            valid = (den >= 2 - 1e-6)
            denom = valid.sum(2).clamp_min(1)
            b = self.biases[:, j, :].to(dtype=conv_out.dtype, device=conv_out.device)
            z = (conv_out.unsqueeze(-1) - b.view(1, C, 1, self.n_bias))
            z = (z > 0).to(conv_out.dtype)
            z = z * valid.unsqueeze(-1)
            ppv = z.sum(2) / denom.unsqueeze(1)
            feats.append(ppv.reshape(B, C * self.n_bias))
        return torch.cat(feats, 1)


class InceptionLite1D(nn.Module):
    def __init__(self, d: int, ks=(3,5,7), dilations=(1,2,4), bottleneck=64, drop=0.1, use_eca=True):
        super().__init__()
        b = max(1, min(bottleneck, d))
        self.reduce = nn.Conv1d(d, b, 1)
        branches = []
        for k in ks:
            for di in dilations:
                pad = (k // 2) * di
                branches.append(nn.Sequential(
                    nn.Conv1d(b, b, k, padding=pad, dilation=di, groups=b, bias=False),
                    nn.Conv1d(b, d, 1, bias=False),
                    nn.GLU(dim=1)
                ))
        self.branches = nn.ModuleList(branches)
        self.proj = nn.Conv1d(d * len(branches) // 2, d, 1, bias=False)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(drop)
        self.gamma = nn.Parameter(1e-2 * torch.ones(1, 1, d))
        self.eca = ECABlock1D(d) if use_eca else nn.Identity()
    def forward(self, x: torch.Tensor, pm: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.reduce(x.transpose(1,2))
        z = torch.cat([br(y) for br in self.branches], 1)
        z = self.proj(z).transpose(1,2)
        z = self.eca(z)
        z = x + self.drop(self.gamma * z)
        if pm is not None:
            z = z * pm.unsqueeze(-1).to(z.dtype)
        return self.norm(z)


class Hybrid3Modal(nn.Module):
    def __init__(self, c_imu: int, c_thm: int, c_tof: int, win_len: int, n_cls: int, CFG_mod=None):
        super().__init__()
        C = CFG if CFG_mod is None else CFG_mod
        self.D = int(getattr(C, "D_MODEL", 128))
        self.seq_len = int(win_len)
        n_blocks = int(getattr(C, "N_LAYER", 3))
        inc_ks = tuple(getattr(C, "INC_KS", (3,5,7)))
        inc_dils = tuple(getattr(C, "INC_DILATIONS", (1,2,4)))
        inc_bottleneck = int(getattr(C, "INC_BOTTLENECK", 64))
        inc_drop = float(getattr(C, "INC_DROP", 0.10))
        use_eca = bool(getattr(C, "USE_ECA", True))
        pool_nq = int(getattr(C, "POOL_NQ", 2))
        pool_out = int(getattr(C, "POOL_OUT", 256))
        pool_drop = float(getattr(C, "POOL_DROP", 0.10))
        self.use_fmix = bool(getattr(C, "USE_FREQMIX", True))
        self.use_minirocket = bool(getattr(C, "USE_MINIROCKET", True))

        self.imu_proj = nn.Sequential(nn.Linear(c_imu, self.D), nn.LayerNorm(self.D), nn.GELU())
        self.thm_proj = nn.Sequential(nn.Linear(c_thm, self.D), nn.LayerNorm(self.D), nn.GELU()) if c_thm>0 else None
        self.tof_proj = nn.Sequential(nn.Linear(c_tof, self.D), nn.LayerNorm(self.D), nn.GELU()) if c_tof>0 else None

        self.blocks = nn.ModuleList([
            InceptionLite1D(self.D, ks=inc_ks, dilations=inc_dils, bottleneck=inc_bottleneck, drop=inc_drop, use_eca=use_eca)
            for _ in range(n_blocks)
        ])

        if self.use_fmix:
            self.fmix = FreqMix1D(self.D, max_len=self.seq_len, share_across_d=False, use_fp32_fft=True)
            self.fmix_alpha = nn.Parameter(torch.tensor(0.5))

        self.pool_attn = AttnPool(self.D, n_q=pool_nq, h=int(getattr(C, "N_HEAD", 4)), out_dim=pool_out, drop=pool_drop)
        self.pool_mstd = MeanStdPool()
        self.pool_mstd_proj = nn.Linear(2 * self.D, pool_out)
        self.fc_128 = nn.Sequential(nn.LayerNorm(2*pool_out), nn.Linear(2*pool_out, 128), nn.GELU())

        self.hblend = HBlend3Q(d=128, hidden=int(getattr(C, "HEAD_HIDDEN", (256,))[0]) if len(getattr(C, "HEAD_HIDDEN", (256,)))>0 else 256, temperature=1.0)
        self.head_full = nn.Linear(128, n_cls)
        self.head_imu  = nn.Linear(128, n_cls)

        if self.use_minirocket:
            mr_dils = tuple(getattr(C, "MR_DILS", (1,2,4,8,16,32)))
            mr_nb   = int(getattr(C, "MR_N_BIAS", 6))
            mr_drop = float(getattr(C, "MR_DROP", 0.10))
            self.mr = MiniRocketPPV(in_ch=c_imu, dilations=mr_dils, n_bias=mr_nb)
            mr_dim = c_imu * len(mr_dils) * mr_nb
            self.mr_head = nn.Sequential(nn.LayerNorm(mr_dim), nn.Linear(mr_dim, 256), nn.GELU(), nn.Dropout(mr_drop), nn.Linear(256, 128))
            self.mr_alpha = nn.Parameter(torch.tensor(-1.0))
            self.imu_merge = nn.Sequential(nn.Linear(128+128, 128), nn.GELU())

        self.recon_head = nn.Linear(self.D, c_imu)

    def _encode_backbone(self, X: torch.Tensor, pmask: Optional[torch.Tensor]):
        z = X
        for blk in self.blocks: z = blk(z, pmask)
        if hasattr(self, "fmix"): z = z + torch.sigmoid(self.fmix_alpha) * self.fmix(z, pmask)
        a = self.pool_attn(z, padding_mask=pmask)
        ms = self.pool_mstd(z, mask=pmask)
        msp = self.pool_mstd_proj(ms)
        f128 = self.fc_128(torch.cat([a, msp], 1))
        if pmask is not None:
            w = pmask.to(z.dtype).unsqueeze(-1)
            mean = (z*w).sum(1) / w.sum(1).clamp_min(1e-6)
            var  = ((z-mean.unsqueeze(1))**2 * w).sum(1) / w.sum(1).clamp_min(1e-6)
        else:
            var = z.var(1, unbiased=False)
        q = torch.sigmoid(var.mean(1))
        return f128, z, q

    def _encode_modality(self, proj, X, pm, device):
        if (proj is None) or (X is None) or (X.shape[-1] == 0):
            B, L = (X.shape[0], X.shape[1]) if X is not None else (pm.shape[0], pm.shape[1])
            return (torch.zeros(B, 128, device=device),
                    torch.zeros(B, L, self.D, device=device),
                    torch.zeros(B, device=device),
                    torch.zeros(B, dtype=torch.bool, device=device))
        present = (pm.sum(1) > 0) if pm is not None else torch.ones(X.size(0), dtype=torch.bool, device=device)
        B, L = X.shape[:2]
        f = torch.zeros(B, 128, device=device)
        z = torch.zeros(B, L, self.D, device=device)
        q = torch.zeros(B, device=device)
        if present.any():
            idx = present.nonzero(as_tuple=True)[0]
            fin, zin, qin = self._encode_backbone(proj(X[idx]), pm[idx] if pm is not None else None)
            f[idx] = fin; z[idx] = zin; q[idx] = qin
        return f, z, q, present

    @staticmethod
    def _fallback_logits_choice(logits_full, logits_imu, present_thm, present_tof):
        only_imu = (~present_thm & ~present_tof)
        return torch.where(only_imu.unsqueeze(1), logits_imu, logits_full)

    def forward(self, X_imu, X_thm=None, X_tof=None, mask_imu=None, mask_thm=None, mask_tof=None, imu_raw=None, force_mode: Optional[str]=None):
        device = X_imu.device
        B, L, _ = X_imu.shape
        assert L == self.seq_len, f"seq_len mismatch: {L} vs {self.seq_len}"

        f_imu, seq_imu, q_imu = self._encode_backbone(self.imu_proj(X_imu), mask_imu)

        if self.use_minirocket and (imu_raw is not None):
            mr_feat = self.mr_head(self.mr(imu_raw, mask_imu))
            mr_gate = torch.sigmoid(self.mr_alpha)
            f_imu = self.imu_merge(torch.cat([f_imu, mr_gate*mr_feat], 1))

        f_thm, _, q_thm, present_thm = self._encode_modality(self.thm_proj, X_thm, mask_thm, device)
        f_tof, _, q_tof, present_tof = self._encode_modality(self.tof_proj, X_tof, mask_tof, device)

        present = torch.stack([torch.ones(B, dtype=torch.bool, device=device), present_thm, present_tof], 1)
        qual = torch.stack([q_imu, q_thm, q_tof], 1).clamp(0, 1)

        fused, alpha = self.hblend(f_imu, f_thm, f_tof, present, qual)
        logits_full = self.head_full(fused)
        logits_imu  = self.head_imu(f_imu)

        if force_mode == 'imu':
            logits = logits_imu
        elif force_mode == 'full':
            logits = logits_full
        else:
            logits = self._fallback_logits_choice(logits_full, logits_imu, present[:,1], present[:,2])

        recon = self.recon_head(seq_imu)
        return logits, logits_full, logits_imu, recon, alpha, present


# ====== cấu hình mặc định ======
AUG_DEFAULTS = dict(
    p_noise=0.5,          noise_sigma=(0.003, 0.02), per_channel=True,
    p_time_mask=0.35,     time_mask_max_frac=0.20,   time_mask_num=(1, 3),
    p_feat_mask=0.35,     feat_mask_max_frac=0.30,
    p_drift=0.30,         drift_std=(0.001, 0.01),   drift_max=(0.03, 0.12),
    p_scale=0.30,         scale_range=(0.9, 1.10),   scale_per_channel=True,
    p_shift=0.30,         shift_max_frac=0.20,       shift_circular=False,
    p_speed=0.25,         speed_range=(0.90, 1.10),
    p_rot3d=0.25,         rot_deg_std=10.0,
)

GROUPS_3D_DEFAULT = {
    'acc':            (0, 1, 2),
    'lin_acc':        (3, 4, 5),
    'ang_vel':        (9, 10, 11),
    'lin_acc_jerk':   (16, 17, 18),
    'ang_vel_jerk':   (19, 20, 21),
    'angular_dist':   (28, 29, 30),
}

def _rand_in_range(lo, hi, shape=(), device='cpu', dtype=torch.float32):
    return torch.empty(shape, device=device, dtype=dtype).uniform_(lo, hi)

def _choose_int(lo, hi, device):               # inclusive randint
    return torch.randint(lo, hi + 1, (), device=device)

def _per_sample_sigmas(B, lo, hi, device):
    return _rand_in_range(lo, hi, (B, 1, 1), device=device)

def _per_sample_scales(B, lo, hi, device):
    return _rand_in_range(lo, hi, (B, 1, 1), device=device)

# ---------- atomic augs ----------
def aug_noise(X, lo=0.003, hi=0.02, per_channel=True):
    B, T, D = X.shape
    if per_channel:
        sigma = _rand_in_range(lo, hi, (B, 1, D), device=X.device, dtype=X.dtype)
    else:
        sigma = _per_sample_sigmas(B, lo, hi, X.device).to(dtype=X.dtype)
    return X + torch.randn_like(X) * sigma

def aug_time_mask(X, max_frac=0.2, num=(1,3)):
    B, T, D = X.shape
    Xo = X.clone()
    m_lo, m_hi = num
    for b in range(B):
        m = int(_choose_int(m_lo, m_hi, X.device))
        for _ in range(m):
            L = max(1, int(math.floor(T * _rand_in_range(1e-6, max_frac, device=X.device).item())))
            s = int(_choose_int(0, max(0, T - L), X.device))
            Xo[b, s:s+L, :] = 0.0
    return Xo

def _int_clamp(v, lo, hi):                     # clamp kiểu Python
    return max(lo, min(hi, int(v)))

def aug_feat_mask(X, max_frac=0.2):
    B, T, D = X.shape
    Xo = X.clone()
    k_max = _int_clamp(D * max_frac, 1, D)
    for b in range(B):
        k = random.randint(1, k_max)
        idx = torch.randperm(D, device=X.device)[:k]
        Xo[b, :, idx] = 0.0
    return Xo

def aug_drift(X, std=(0.001, 0.01), maxv=(0.03, 0.12)):
    B, T, D = X.shape
    Xo = X.clone()
    sigmas = _per_sample_sigmas(B, *std, X.device).to(dtype=X.dtype).squeeze(-1)  # [B,1]
    maxs   = _per_sample_sigmas(B, *maxv, X.device).to(dtype=X.dtype).squeeze(-1) # [B,1]
    for b in range(B):
        inc = torch.randn(T, device=X.device, dtype=X.dtype) * sigmas[b,0]
        drift = torch.cumsum(inc, dim=0).clamp_(-maxs[b,0], maxs[b,0])            # [T]
        Xo[b] = Xo[b] + drift.unsqueeze(1)
    return Xo

def aug_scale(X, lo=0.9, hi=1.1, per_channel=True):
    B, T, D = X.shape
    if per_channel:
        s = _rand_in_range(lo, hi, (B, 1, D), device=X.device, dtype=X.dtype)
    else:
        s = _per_sample_scales(B, lo, hi, X.device).to(dtype=X.dtype)
    return X * s

def aug_time_shift(X, max_frac=0.2, circular=False):
    """
    Non-overlapping implementation: dùng torch.roll rồi zero phần chèn.
    """
    B, T, D = X.shape
    Xo = X.clone()
    max_shift = max(1, int(T * max_frac))
    for b in range(B):
        s = int(_choose_int(-max_shift, max_shift, X.device))
        if s == 0:
            continue
        if circular:
            Xo[b] = torch.roll(Xo[b], shifts=s, dims=0)
        else:
            if s > 0:                     # dịch sang phải
                Xo[b] = torch.roll(Xo[b], shifts=s, dims=0)
                Xo[b, :s] = 0.0
            else:                         # dịch sang trái
                k = -s
                Xo[b] = torch.roll(Xo[b], shifts=-k, dims=0)
                Xo[b, -k:] = 0.0
    return Xo

def aug_speed_perturb(X, speed_range=(0.9, 1.1)):
    """
    Resample theo thời gian rồi pad/crop về T (không overlap write).
    """
    B, T, D = X.shape
    Xo = torch.empty_like(X)
    for b in range(B):
        s = float(_rand_in_range(speed_range[0], speed_range[1], device=X.device).item())
        M = max(2, int(round(T * s)))
        # [1,D,T] để dùng interpolate 1D theo trục T
        xb = X[b].transpose(0,1).unsqueeze(0)                           # [1,D,T]
        xb_rs = F.interpolate(xb, size=M, mode='linear', align_corners=False)
        xb_rs = xb_rs.squeeze(0).transpose(0,1)                          # [M,D]
        if M == T:
            Xo[b] = xb_rs
        elif M < T:
            pad = T - M
            left = pad // 2
            right = pad - left
            Xo[b] = F.pad(xb_rs, (0,0,left,right), mode='constant', value=0.0)
        else:  # M > T
            start = (M - T) // 2
            Xo[b] = xb_rs[start:start+T]
    return Xo

def _rot_mat_from_euler(rx, ry, rz, dtype, device):
    cx, sx = torch.cos(rx), torch.sin(rx)
    cy, sy = torch.cos(ry), torch.sin(ry)
    cz, sz = torch.cos(rz), torch.sin(rz)
    Rz = torch.tensor([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]], device=device, dtype=dtype)
    Rz = torch.stack([torch.stack([cz, -sz, torch.zeros_like(cz)], -1),
                      torch.stack([sz,  cz, torch.zeros_like(cz)], -1),
                      torch.tensor([0.0,0.0,1.0], device=device, dtype=dtype)], -2)
    Ry = torch.stack([torch.stack([cy, torch.zeros_like(cy), sy], -1),
                      torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)], -1),
                      torch.stack([-sy, torch.zeros_like(cy), cy], -1)], -2).to(dtype)
    Rx = torch.stack([torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)], -1),
                      torch.stack([torch.zeros_like(cx), cx, -sx], -1),
                      torch.stack([torch.zeros_like(cx), sx,  cx], -1)], -2).to(dtype)
    return Rz @ Ry @ Rx

def aug_rot3d(X, groups=GROUPS_3D_DEFAULT, deg_std=10.0):
    B, T, D = X.shape
    Xo = X.clone()
    rad_std = deg_std * math.pi / 180.0
    for b in range(B):
        rx = torch.randn((), device=X.device, dtype=X.dtype) * rad_std
        ry = torch.randn((), device=X.device, dtype=X.dtype) * rad_std
        rz = torch.randn((), device=X.device, dtype=X.dtype) * rad_std
        R = _rot_mat_from_euler(rx, ry, rz, X.dtype, X.device)  # [3,3]
        for g in groups.values():
            if max(g) < D:
                Xo[b, :, list(g)] = torch.matmul(Xo[b, :, list(g)], R.T)
    return Xo

# ---------- Orchestrator ----------
def augment(X, y, mask=None, cfg=AUG_DEFAULTS, groups_3d=GROUPS_3D_DEFAULT):
    """
    X: [B,T,D], y: [B]
    mask: [B,T] (bool). Nếu cung cấp, cuối cùng sẽ zero lại các timestep pad.
    """
    X_aug = X.clone()

    if torch.rand((), device=X.device) < cfg['p_noise']:
        X_aug = aug_noise(X_aug, *cfg['noise_sigma'], per_channel=cfg['per_channel'])

    if torch.rand((), device=X.device) < cfg['p_time_mask']:
        X_aug = aug_time_mask(X_aug, max_frac=cfg['time_mask_max_frac'], num=cfg['time_mask_num'])

    if torch.rand((), device=X.device) < cfg['p_feat_mask']:
        X_aug = aug_feat_mask(X_aug, max_frac=cfg['feat_mask_max_frac'])

    if torch.rand((), device=X.device) < cfg['p_drift']:
        X_aug = aug_drift(X_aug, std=cfg['drift_std'], maxv=cfg['drift_max'])

    if torch.rand((), device=X.device) < cfg['p_scale']:
        X_aug = aug_scale(X_aug, lo=cfg['scale_range'][0], hi=cfg['scale_range'][1], per_channel=cfg['scale_per_channel'])

    if torch.rand((), device=X.device) < cfg['p_shift']:
        X_aug = aug_time_shift(X_aug, max_frac=cfg['shift_max_frac'], circular=cfg['shift_circular'])

    if torch.rand((), device=X.device) < cfg['p_speed']:
        X_aug = aug_speed_perturb(X_aug, speed_range=cfg['speed_range'])

    if torch.rand((), device=X.device) < cfg['p_rot3d']:
        X_aug = aug_rot3d(X_aug, groups=groups_3d, deg_std=cfg['rot_deg_std'])

    # Tôn trọng pad nếu có mask
    if mask is not None:
        X_aug = X_aug * mask.unsqueeze(-1).to(X_aug.dtype)

    return X_aug, y


def mixup_augmenter(x, y, alpha=0.2, device='cpu'):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    x_mix = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return x_mix, x, x[index], y_a, y_b, lam, index


class CMILoss(nn.Module):
    """
    Loss cho CMI:
      L = CE_la_focal + λ_f1 * (softF1_binary + softF1_macro_grouped)/2
          + [exp(-log_sigma_rec)*MSE + log_sigma_rec] (nếu có nhánh rec)

    - CE_la_focal: Cross-Entropy với Logit-Adjust + Focal (+ Label Smoothing)
    - softF1_binary: BFRB vs non-BFRB (theo mask các lớp BFRB)
    - softF1_macro_grouped: Macro-F1 trên K'+1 lớp (mỗi BFRB 1 lớp, toàn bộ non-target gộp 1 lớp)
    """

    def __init__(self,
                 num_classes: int,
                 bfrb_classes: Sequence[int],       # chỉ số các lớp BFRB (len = 8)
                 class_counts: Optional[Sequence[int]] = None,
                 log_prior: Optional[torch.Tensor] = None,  # tensor [C] log(prior) nếu đã có
                 tau: float = 1.0,                  # hệ số logit-adjust
                 gamma: float = 1.0,                # focal gamma (0 => tắt focal)
                 label_smoothing: float = 0.05,     # LS nhẹ
                 lambda_softf1: float = 0.2,        # trọng số soft-F1
                 use_logit_adjust: bool = True,
                 allow_both: bool = False,          # cho phép kèm class weights (không khuyến khích)
                 class_weights: Optional[Sequence[float]] = None,
                 rec_init_log_sigma: float = 0.0,   # nếu dùng nhánh reconstruction
                 use_reconstruction: bool = True):
        super().__init__()
        self.C = int(num_classes)
        self.tau = float(tau)
        self.gamma = float(gamma)
        self.ls = float(label_smoothing)
        self.lambda_softf1 = float(lambda_softf1)
        self.use_logit_adjust = bool(use_logit_adjust)
        self.allow_both = bool(allow_both)
        self.use_reconstruction = bool(use_reconstruction)

        # --- BFRB mask
        bfrb_idx = torch.as_tensor(bfrb_classes, dtype=torch.long)
        if bfrb_idx.numel() == 0:
            raise ValueError("bfrb_classes rỗng.")
        mask = torch.zeros(self.C, dtype=torch.bool)
        mask[bfrb_idx] = True
        self.register_buffer("bfrb_mask", mask)

        # --- log-prior (đăng ký buffer DÙ có/không có -> tránh KeyError ở state_dict)
        lp = None
        if log_prior is not None:
            lp = torch.as_tensor(log_prior, dtype=torch.float32)
        elif class_counts is not None:
            counts = np.asarray(class_counts, dtype=np.float64)
            counts[counts <= 0] = 1.0
            prior = counts / counts.sum()
            lp = torch.from_numpy(np.log(np.clip(prior, 1e-12, 1.0)).astype(np.float32))
        self.register_buffer("log_prior", lp)  # có thể là None

        # --- class weights (nếu dùng; mặc định tắt khi có logit-adjust)
        if self.use_logit_adjust and (lp is not None) and (not self.allow_both):
            class_weights = None
        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            if cw.numel() != self.C:
                raise ValueError(f"class_weights phải có độ dài = num_classes ({self.C}), hiện = {cw.numel()}.")
        else:
            cw = None
        self.register_buffer("class_weights", cw)  # có thể là None

        # --- learnable log-variance cho nhánh reconstruction
        if self.use_reconstruction:
            self.log_sigma_rec = nn.Parameter(torch.tensor(rec_init_log_sigma, dtype=torch.float32))
        else:
            self.register_parameter("log_sigma_rec", None)

    @torch.no_grad()
    def adjust_logits(self, logits: torch.Tensor) -> torch.Tensor:
        if self.use_logit_adjust and (self.log_prior is not None):
            return logits - self.tau * self.log_prior.to(logits.device, logits.dtype)
        return logits

    def _ce_la_focal(self, logits: torch.Tensor, targets: torch.LongTensor) -> torch.Tensor:
        # logit-adjust
        if self.use_logit_adjust and (self.log_prior is not None):
            logits = logits - self.tau * self.log_prior.to(logits.device, logits.dtype)

        log_probs = F.log_softmax(logits, dim=1)               # [N, C]
        probs = log_probs.exp()
        N, C = log_probs.size()

        # label smoothing
        if self.ls > 0:
            y = torch.zeros_like(log_probs).scatter_(1, targets.unsqueeze(1), 1.0)
            y = y * (1 - self.ls) + self.ls / C
            ce_per_sample = -(y * log_probs).sum(dim=1)        # [N]
            pt = (probs * y).sum(dim=1).clamp(min=1e-8)
        else:
            ce_per_sample = F.nll_loss(log_probs, targets, reduction='none')
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-8)

        # focal modulation
        if self.gamma > 0.0:
            ce_per_sample = ((1 - pt) ** self.gamma) * ce_per_sample

        # class weights (nếu có)
        if self.class_weights is not None:
            w = self.class_weights.to(log_probs.device, log_probs.dtype)
            ce_per_sample = ce_per_sample * w.gather(0, targets)

        return ce_per_sample.mean()

    def _soft_f1_binary(self, probs: torch.Tensor, targets_onehot: torch.Tensor) -> torch.Tensor:
        """Soft-F1 nhị phân: positive = bất kỳ lớp BFRB nào."""
        pos_mask = self.bfrb_mask.to(probs.device)
        p_pos = probs[:, pos_mask].sum(dim=1)      # [N]
        y_pos = targets_onehot[:, pos_mask].sum(dim=1)
        eps = 1e-8
        tp = (p_pos * y_pos).sum()
        precision = tp / (p_pos.sum() + eps)
        recall = tp / (y_pos.sum() + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        return 1.0 - f1

    def _soft_f1_macro_grouped(self, probs: torch.Tensor, targets_onehot: torch.Tensor) -> torch.Tensor:
        """Macro soft-F1 trên K'+1 lớp (BFRB tách riêng, Non-Target gộp 1)."""
        device = probs.device
        bmask = self.bfrb_mask.to(device)
        p_groups = torch.cat([probs[:, bmask], probs[:, ~bmask].sum(dim=1, keepdim=True)], dim=1)
        y_groups = torch.cat([targets_onehot[:, bmask], targets_onehot[:, ~bmask].sum(dim=1, keepdim=True)], dim=1)

        eps = 1e-8
        tp = (p_groups * y_groups).sum(dim=0)             # [G]
        precision = tp / (p_groups.sum(dim=0) + eps)
        recall = tp / (y_groups.sum(dim=0) + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)    # [G]
        macro_f1 = f1.mean()
        return 1.0 - macro_f1

    def forward(self,
                logits: torch.Tensor,
                targets: torch.LongTensor,
                rec_pred: Optional[torch.Tensor] = None,
                rec_target: Optional[torch.Tensor] = None) -> torch.Tensor:

        # 1) CE với logit-adjust + focal (+ LS)
        loss_ce = self._ce_la_focal(logits, targets)

        # 2) Soft-F1 regularizer (align metric)
        probs_adj = F.softmax(self.adjust_logits(logits), dim=1)
        targets_onehot = F.one_hot(targets, num_classes=self.C).to(probs_adj.dtype)

        loss_f1_bin = self._soft_f1_binary(probs_adj, targets_onehot)
        loss_f1_mac = self._soft_f1_macro_grouped(probs_adj, targets_onehot)
        loss = loss_ce + self.lambda_softf1 * 0.5 * (loss_f1_bin + loss_f1_mac)

        # 3) (tuỳ chọn) Reconstruction auxiliary
        if self.use_reconstruction and (rec_pred is not None) and (rec_target is not None):
            mse = F.mse_loss(rec_pred, rec_target, reduction='mean')
            precision_rec = torch.exp(-self.log_sigma_rec)
            loss = loss + (precision_rec * mse + self.log_sigma_rec)

        return loss


def mixup_criterion(loss_fn: CMILoss,
                    logits: torch.Tensor,
                    y_a: torch.LongTensor,
                    y_b: torch.LongTensor,
                    rec_pred: Optional[torch.Tensor],
                    rec_targ_a: Optional[torch.Tensor],
                    rec_targ_b: Optional[torch.Tensor],
                    lam: float) -> torch.Tensor:
    """
    CE + softF1 đều là batch-mean -> có thể nội suy tuyến tính theo lam.
    Reconstruction (nếu dùng) cũng đi theo từng nhánh tương ứng.
    """
    return (lam * loss_fn(logits, y_a, rec_pred, rec_targ_a)
            + (1 - lam) * loss_fn(logits, y_b, rec_pred, rec_targ_b))


@torch.inference_mode()
def compute_val_metrics_seq(
    model,
    loader,
    device,
    seq_ids_val,
    label_encoder=None,
    use_amp=True,
    alpha=1.0,
    beta=1.0,
    t_model=1.0,
    force_imu_only: bool = False,
    X_imu_va=None  # optional numpy IMU (chỉ dùng nếu không có X_roc trong loader)
):
    """
    Gộp dự đoán theo sequence với trọng số: w = coverage^alpha * confidence^beta
    - force_imu_only=True: ép dùng head_imu (tắt THM/TOF)
    - force_imu_only=False: để model tự router/blend theo present + CFG.ROUTER_WEIGHT
    """
    model.eval()

    # Suy ra số lớp từ head_full/head_imu
    if hasattr(model, "head_full"):
        C = model.head_full.out_features
    elif hasattr(model, "head_imu"):
        C = model.head_imu.out_features
    else:
        raise AssertionError("Cannot infer number of classes: model lacks head_full/head_imu.")

    seq_logp_sum = defaultdict(lambda: torch.zeros(C, dtype=torch.float32))
    seq_w_sum    = defaultdict(float)
    seq_true     = {}

    ptr   = 0
    eps   = 1e-8
    logC  = math.log(C + 1e-12)
    is_cuda = (torch.cuda.is_available() and str(device).startswith("cuda"))

    for batch in loader:
        # batch có thể gồm 8 (có X_roc) hoặc 7 (không có X_roc) tensor
        if len(batch) == 8:
            xb_imu, xb_thm, xb_tof, xb_roc, yb, m_imu, m_thm, m_tof = batch
        elif len(batch) == 7:
            xb_imu, xb_thm, xb_tof, yb, m_imu, m_thm, m_tof = batch
            xb_roc = None
        else:
            raise RuntimeError(f"Unexpected val batch length {len(batch)}; expected 7 or 8.")

        xb_imu = xb_imu.to(device, non_blocking=True)
        xb_thm = xb_thm.to(device, non_blocking=True)
        xb_tof = xb_tof.to(device, non_blocking=True)
        yb     = yb.to(device, non_blocking=True)
        m_imu  = m_imu.to(device, non_blocking=True)
        m_thm  = m_thm.to(device, non_blocking=True)
        m_tof  = m_tof.to(device, non_blocking=True)
        if xb_roc is not None:
            xb_roc = xb_roc.to(device, non_blocking=True)

        B = xb_imu.size(0)
        batch_seq_ids = seq_ids_val[ptr:ptr + B]

        # Chọn đầu vào cho MiniRocket:
        # 1) nếu có X_roc trong loader -> dùng xb_roc
        # 2) nếu không có nhưng có X_imu_va (numpy) -> slice theo ptr
        # 3) fallback dùng xb_imu (đã scale) — chỉ khi MiniRocket khởi tạo theo IMU
        if xb_roc is not None:
            imu_raw = xb_roc
        elif X_imu_va is not None:
            imu_raw = torch.from_numpy(X_imu_va[ptr:ptr+B]).to(device, non_blocking=True)
        else:
            imu_raw = xb_imu

        # Ép IMU-only: tắt THM/TOF và dùng head_imu
        if force_imu_only:
            if xb_thm.shape[-1] > 0:
                xb_thm = torch.zeros_like(xb_thm)
                m_thm  = torch.zeros_like(m_thm)
            if xb_tof.shape[-1] > 0:
                xb_tof = torch.zeros_like(xb_tof)
                m_tof  = torch.zeros_like(m_tof)
            force_mode = 'imu'
        else:
            force_mode = None  # để model tự router/blend

        # Forward
        with torch.amp.autocast('cuda', enabled=use_amp and is_cuda):
            logits, logits_full, logits_imu, *_ = model(
                xb_imu,
                None if (force_imu_only or xb_thm.shape[-1] == 0) else xb_thm,
                None if (force_imu_only or xb_tof.shape[-1] == 0) else xb_tof,
                mask_imu=m_imu, mask_thm=m_thm, mask_tof=m_tof,
                imu_raw=imu_raw,
                force_mode=force_mode
            )

        # Cập nhật con trỏ (phải sau khi đã dùng X_imu_va slice theo ptr)
        ptr += B

        # Softmax + entropy-based confidence
        logp = F.log_softmax(logits / float(t_model), dim=1)   # [B,C]
        p    = logp.exp()
        ent  = -(p * logp).sum(dim=1) / logC
        conf = (1.0 - ent).clamp(0.0, 1.0)

        # coverage (m_imu True ratio trong cửa sổ)
        cov  = m_imu.float().mean(dim=1)

        # Trọng số theo coverage^alpha * confidence^beta
        w_win   = (cov + eps).pow(float(alpha)) * (conf + eps).pow(float(beta))
        w_cpu   = w_win.detach().float().cpu()
        logp_c  = logp.detach().float().cpu()
        yb_cpu  = yb.detach().cpu().numpy()

        # Tích lũy theo sequence
        for j, sid in enumerate(batch_seq_ids):
            wj = float(w_cpu[j].item())
            if wj <= 0:
                continue
            seq_logp_sum[sid] += wj * logp_c[j]
            seq_w_sum[sid]    += wj
            if sid not in seq_true:
                seq_true[sid] = int(yb_cpu[j])

    # Gộp theo sequence với trọng số
    seq_ids_unique = list(seq_true.keys())
    pred_ids, true_ids = [], []
    for sid in seq_ids_unique:
        denom = max(seq_w_sum[sid], eps)
        fused_logp = seq_logp_sum[sid] / denom
        pred_ids.append(int(fused_logp.argmax().item()))
        true_ids.append(seq_true[sid])

    # Tính điểm (HF1, bin_f1, macro_f1) qua _score (định nghĩa sẵn ở repo của bạn)
    if label_encoder is not None:
        df_sol = pd.DataFrame({
            "sequence_id": seq_ids_unique,
            "gesture": label_encoder.inverse_transform(np.asarray(true_ids))
        })
        df_sub = pd.DataFrame({
            "sequence_id": seq_ids_unique,
            "gesture": label_encoder.inverse_transform(np.asarray(pred_ids))
        })
    else:
        df_sol = pd.DataFrame({"sequence_id": seq_ids_unique, "gesture": np.asarray(true_ids)})
        df_sub = pd.DataFrame({"sequence_id": seq_ids_unique, "gesture": np.asarray(pred_ids)})

    return _score(df_sol, df_sub)


# =========================
# EMA an toàn cho mọi dtype
# =========================
class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = float(decay)
        # lưu shadow trên CPU để tiết kiệm VRAM (an toàn khi torch.save)
        self.shadow = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        # cập nhật từ tham số hiện tại của model (trên device nào cũng được)
        cur = model.state_dict()
        for k, v in cur.items():
            v_cpu = v.detach().cpu()
            if not torch.is_floating_point(v_cpu):
                # buffer không-float (vd num_batches_tracked): copy thẳng
                self.shadow[k] = v_cpu.clone()
            else:
                self.shadow[k].mul_(self.decay).add_(v_cpu, alpha=1.0 - self.decay)

    def copy_to(self, model):
        # chép EMA params vào model; strict=False đề phòng sai khác minor
        model.load_state_dict(self.shadow, strict=False)


def derive_bfrb_classes(label_encoder=None, id2label=None):
    import re
    idxs = set()

    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        classes = list(label_encoder.classes_)  # list[str] đúng thứ tự mã hoá 0..C-1
        name_to_idx = {name: i for i, name in enumerate(classes)}
        # 1) map cứng theo tên chuẩn (nếu trùng khớp tuyệt đối)
        for name in TARGETS:
            if name in name_to_idx:
                idxs.add(name_to_idx[name])
        # 2) fallback mềm bằng pattern (phòng khi tên khác biệt nhẹ)
        if len(idxs) < 8:
            for i, n in enumerate(classes):
                n_low = n.lower()
                if ("pull hair" in n_low) or ("pinch skin" in n_low) or re.search(r"(forehead|neck)\s*-\s*scratch", n_low):
                    idxs.add(i)

    elif id2label is not None:  # dict[int] -> str
        for i, n in id2label.items():
            n_low = n.lower()
            if (n in TARGETS) or ("pull hair" in n_low) or ("pinch skin" in n_low) or ("- scratch" in n_low):
                idxs.add(int(i))

    b = sorted(idxs)
    if len(b) != 8:
        print(f"[WARN] Expected 8 BFRB classes, got {len(b)} -> {b}. Kiểm tra lại tên nhãn!")
    return torch.as_tensor(b, dtype=torch.long)


def Train(
    X_imu_tr, X_thm_tr, X_tof_tr, X_roc_tr, y_tr, m_imu_tr, m_thm_tr, m_tof_tr,
    X_imu_va, X_thm_va, X_tof_va, X_roc_va, y_va, m_imu_va, m_thm_va, m_tof_va, seq_ids_val,
    cfg, ckpt_dir, save_name, weights,
    imu_only_prob=0.5, use_mixup=True, mixup_prob=0.5, use_augment=True, use_amp=True,
    label_encoder=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_cuda_amp = (device == "cuda") and bool(use_amp)
    amp_ctx = (lambda: torch.amp.autocast('cuda', enabled=True)) if use_cuda_amp else nullcontext
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda_amp)

    def to_tensor(x, dtype):
        return torch.from_numpy(x).to(dtype)

    # ---------------- Dataloaders ----------------
    tr_ds = TensorDataset(
        to_tensor(X_imu_tr, torch.float32),
        to_tensor(X_thm_tr, torch.float32),
        to_tensor(X_tof_tr, torch.float32),
        to_tensor(X_roc_tr, torch.float32),   # raw IMU cho MiniRocket
        to_tensor(y_tr,    torch.int64),
        to_tensor(m_imu_tr, torch.bool),
        to_tensor(m_thm_tr, torch.bool),
        to_tensor(m_tof_tr, torch.bool),
    )
    nworkers = 2
    tr_loader = DataLoader(
        tr_ds, batch_size=cfg.BATCH, shuffle=True,
        num_workers=nworkers, pin_memory=(device=="cuda"),
        persistent_workers=(nworkers > 0)
    )

    va_loader = None
    if y_va is not None:
        va_ds = TensorDataset(
            to_tensor(X_imu_va, torch.float32),
            to_tensor(X_thm_va, torch.float32),
            to_tensor(X_tof_va, torch.float32),
            to_tensor(X_roc_va, torch.float32),
            to_tensor(y_va,    torch.int64),
            to_tensor(m_imu_va, torch.bool),
            to_tensor(m_thm_va, torch.bool),
            to_tensor(m_tof_va, torch.bool),
        )
        va_loader = DataLoader(
            va_ds, batch_size=cfg.BATCH, shuffle=False,
            num_workers=nworkers, pin_memory=(device=="cuda"),
            persistent_workers=(nworkers > 0)
        )

    # ---------------- Model ----------------
    c_imu, c_thm, c_tof = X_imu_tr.shape[-1], X_thm_tr.shape[-1], X_tof_tr.shape[-1]
    # dùng bản Dual-head mới
    model = Hybrid3Modal(c_imu, c_thm, c_tof, CFG.WIN_LEN, cfg.N_CLASSES, CFG).to(device)
    
    # ---- (Re)Init MiniRocket theo kênh của X_roc ----
    C_roc = int(X_roc_tr.shape[-1])
    if getattr(cfg, "USE_MINIROCKET", True) and C_roc > 0:
        dils   = tuple(getattr(cfg, "MR_DILS", (1,2,4,8,16,32)))
        n_bias = int(getattr(cfg, "MR_N_BIAS", 6))

        # gán lại module MiniRocket để khớp số kênh X_roc
        model.mr = MiniRocketPPV(C_roc, dilations=dils, n_bias=n_bias)
        mr_dim   = C_roc * len(dils) * n_bias
        model.mr_head = nn.Sequential(
            nn.LayerNorm(mr_dim),
            nn.Linear(mr_dim, 256), nn.GELU(), nn.Dropout(getattr(cfg, "MR_DROP", 0.1)),
            nn.Linear(256, 128)
        )
        model.use_minirocket = True

        model.mr.to(device); model.mr_head.to(device)

        # Fit biases từ một subset train
        with torch.no_grad():
            n_fit = min(int(getattr(cfg, "MR_MAX_WINDOWS", 4000)), len(X_roc_tr))
            X_fit = torch.from_numpy(X_roc_tr[:n_fit]).to(device)
            M_fit = torch.from_numpy(m_imu_tr[:n_fit]).to(device)
            model.mr.fit_biases(X_fit, mask=M_fit, max_windows=n_fit)
    else:
        model.use_minirocket = False

    # ---------------- Optim & Scheduler ----------------
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WD)
    steps_per_epoch = max(1, len(tr_loader))
    total_steps     = cfg.EPOCHS * steps_per_epoch
    warmup_steps    = int(getattr(cfg, "WARMUP_FRAC", 0.1) * total_steps)
    eta_min         = float(getattr(cfg, "ETA_MIN", 1e-6))

    warmup  = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=max(1, warmup_steps))
    cosine  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, total_steps - warmup_steps), eta_min=eta_min)
    scheduler = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[warmup_steps])

    # ---------------- Loss ----------------
    bfrb_classes = derive_bfrb_classes(label_encoder=label_encoder)
    num_classes  = int(getattr(cfg, "N_CLASSES", 18))
    class_counts = np.bincount(y_tr, minlength=num_classes).astype(np.int64)
    # bfrb_classes: danh sách chỉ số các lớp BFRB theo mapping của bạn, ví dụ [1,2,3,4]
    criterion = CMILoss(
        num_classes=num_classes,
        bfrb_classes=bfrb_classes,
        class_counts=class_counts,  # dùng để tính log_prior
        tau=1.0,
        gamma=3.0,                  # focal nhẹ
        label_smoothing=0.05,
        lambda_softf1=0.2,          # regularizer bám sát metric
        use_logit_adjust=True,
        allow_both=False,
        class_weights=None,
        rec_init_log_sigma=0.0,
        use_reconstruction=True
    ).to(device)

    # ---------------- EMA ----------------
    ema = EMA(model, decay=float(getattr(cfg, "EMA_DECAY", 0.995)))

    # ---------------- Seed ----------------
    np.random.seed(cfg.SEED); torch.manual_seed(cfg.SEED)
    if device == "cuda":
        torch.cuda.manual_seed_all(cfg.SEED)

    memo, memo_loss = [], []
    best_hf1_ema   = -1.0; best_ema_state = None
    best_hf1_raw   = -1.0; best_raw_state = None
    best_hf1_imu   = -1.0; best_imu_state = None
    best_hf1_combo = -1.0; best_combo_state = None

    global_step = 0
    for epoch in range(cfg.EPOCHS):
        model.train()
        losses = []

        for i, batch in enumerate(tr_loader):
            xb_imu, xb_thm, xb_tof, xb_roc, yb, m_imu, m_thm, m_tof = batch
            xb_imu = xb_imu.to(device, non_blocking=True)
            xb_thm = xb_thm.to(device, non_blocking=True)
            xb_tof = xb_tof.to(device, non_blocking=True)
            xb_roc = xb_roc.to(device, non_blocking=True)
            yb     = yb.to(device, non_blocking=True)
            m_imu  = m_imu.to(device, non_blocking=True)
            m_thm  = m_thm.to(device, non_blocking=True)
            m_tof  = m_tof.to(device, non_blocking=True)

            # --------- Chọn chế độ train: IMU-only vs Full ---------
            # Nếu chọn IMU-only: zero THM/TOF + bắt buộc train head_imu
            batch_imu_only = (np.random.rand() < imu_only_prob)
            if batch_imu_only:
                if xb_thm.shape[-1] > 0: xb_thm.zero_(); m_thm.zero_()
                if xb_tof.shape[-1] > 0: xb_tof.zero_(); m_tof.zero_()

            if use_augment:
                xb_imu, yb = augment(xb_imu, yb, mask=m_imu)

            opt.zero_grad(set_to_none=True)
            with amp_ctx():
                force_mode = 'imu' if batch_imu_only else 'full'
                use_rec = (global_step <= warmup_steps)  # dùng nhánh reconstruction TRƯỚC hoặc ĐẾN mốc
            
                if use_mixup and (np.random.rand() < mixup_prob):
                    # Mixup chỉ trên IMU/raw-IMU (THM/TOF giữ nguyên)
                    xb_mix, xb_a, xb_b, y_a, y_b, lam, perm = mixup_augmenter(xb_imu, yb)
                    roc_mix = lam * xb_roc + (1 - lam) * xb_roc[perm]
            
                    logits, logits_full, logits_imu, rec, alpha, present = model(
                        xb_mix,
                        xb_thm if xb_thm.shape[-1] > 0 else None,
                        xb_tof if xb_tof.shape[-1] > 0 else None,
                        mask_imu=m_imu, mask_thm=m_thm, mask_tof=m_tof,
                        imu_raw=roc_mix,
                        force_mode=force_mode
                    )
            
                    # Nếu chưa qua mốc => thêm rec; sau mốc => bỏ rec
                    rec_pred_in = rec if use_rec else None
                    targ_a_in   = xb_a if use_rec else None
                    targ_b_in   = xb_b if use_rec else None
            
                    loss = mixup_criterion(
                        criterion,          # CMILoss
                        logits,             # logits đã router theo force_mode
                        y_a, y_b,
                        rec_pred_in,
                        targ_a_in, targ_b_in,
                        lam
                    )
            
                else:
                    logits, logits_full, logits_imu, rec, alpha, present = model(
                        xb_imu,
                        xb_thm if xb_thm.shape[-1] > 0 else None,
                        xb_tof if xb_tof.shape[-1] > 0 else None,
                        mask_imu=m_imu, mask_thm=m_thm, mask_tof=m_tof,
                        imu_raw=xb_roc,
                        force_mode=force_mode
                    )
            
                    rec_in  = rec if use_rec else None
                    targ_in = xb_imu if use_rec else None
                    loss = criterion(logits, yb, rec_in, targ_in)

            scaler.scale(loss).backward()
            if use_cuda_amp:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(getattr(cfg, "GRAD_CLIP", 1.0)))
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            scheduler.step()

            losses.append(loss.detach().item())
            global_step += 1

        memo_loss.append(float(np.mean(losses)) if len(losses) else float('nan'))

        # ---------------- Validation ----------------
        if va_loader is not None:
            model.eval()
            with torch.no_grad():
                # RAW (model hiện tại, router/blend theo CFG.ROUTER_WEIGHT)
                hf1_raw, bin_f1_raw, macro_f1_raw = compute_val_metrics_seq(
                    model, va_loader, device,
                    seq_ids_val=seq_ids_val,
                    label_encoder=label_encoder,
                    use_amp=use_cuda_amp,
                    alpha=1.0, beta=1.0, t_model=1.0,
                    force_imu_only=False
                )
                # EMA
                raw_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                ema.copy_to(model)
                hf1_ema, bin_f1_ema, macro_f1_ema = compute_val_metrics_seq(
                    model, va_loader, device,
                    seq_ids_val=seq_ids_val,
                    label_encoder=label_encoder,
                    use_amp=use_cuda_amp,
                    alpha=1.0, beta=1.0, t_model=1.0,
                    force_imu_only=False
                )
                # EMA - IMU only (đánh giá chế độ IMU-only)
                hf1_imu, bin_f1_imu, macro_f1_imu = compute_val_metrics_seq(
                    model, va_loader, device,
                    seq_ids_val=seq_ids_val,
                    label_encoder=label_encoder,
                    use_amp=use_cuda_amp,
                    alpha=1.0, beta=1.0, t_model=1.0,
                    force_imu_only=True
                )
                model.load_state_dict(raw_state, strict=False)

            if hf1_raw > best_hf1_raw:
                best_hf1_raw = hf1_raw; best_raw_state = raw_state
            if hf1_ema > best_hf1_ema:
                best_hf1_ema = hf1_ema
                best_ema_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
            if hf1_imu > best_hf1_imu:
                best_hf1_imu = hf1_imu
                best_imu_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
            combo_now = 0.5 * (hf1_ema + hf1_imu)
            if combo_now > best_hf1_combo:
                best_hf1_combo = combo_now
                best_combo_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}

            memo.append({
                "RAW":   (hf1_raw, bin_f1_raw,   macro_f1_raw),
                "EMA":   (hf1_ema, bin_f1_ema,   macro_f1_ema),
                "EMA_IMU": (hf1_imu, bin_f1_imu, macro_f1_imu),
                "COMBO": (
                    0.5*(hf1_ema + hf1_imu),
                    0.5*(bin_f1_ema + bin_f1_imu),
                    0.5*(macro_f1_ema + macro_f1_imu)
                )
            })

        if (epoch + 1) % 5 == 0:
            cur_lr = opt.param_groups[0]['lr']
            msg = f"Epoch {epoch+1}/{cfg.EPOCHS} | Loss {memo_loss[-1]:.4f} | "
            if va_loader is not None:
                msg += f"Best EMA HF1 {best_hf1_ema:.4f} | Best IMU HF1 {best_hf1_imu:.4f} | Best COMBO {best_hf1_combo:.4f} | "
            msg += f"LR {cur_lr:.2e}"
            print(msg)

    # ---------------- Save checkpoints ----------------
    # if best_ema_state is None:
    #     best_ema_state = {k: v.detach().cpu().clone() for k, v in ema.shadow.items()}
    # torch.save(best_ema_state, os.path.join(ckpt_dir, save_name))
    if best_imu_state is not None:
        torch.save(best_imu_state, os.path.join(ckpt_dir, save_name.replace(".pt", "_imu.pt")))
    if best_combo_state is not None:
        torch.save(best_combo_state, os.path.join(ckpt_dir, save_name.replace(".pt", "_combo.pt")))
    # if best_raw_state is not None:
    #     torch.save(best_raw_state, os.path.join(ckpt_dir, save_name.replace(".pt", "_raw.pt")))

    return (best_hf1_ema if best_hf1_ema >= 0 else None), memo, memo_loss


def CrossKFolds(train, cfg, ckpt_dir="chkpts", weights=None, plot=True):
    """
    Trả về: list EMA HF1 theo fold; in thêm EMA-IMU và COMBO.
    """
    sgkf = StratifiedGroupKFold(n_splits=cfg.N_SPLIT, shuffle=True, random_state=cfg.SEED)
    os.makedirs(ckpt_dir, exist_ok=True)

    scores_ema, scores_imu, scores_combo = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(
        sgkf.split(train, train['gesture'], groups=train['subject']), start=1
    ):
        print(f"Start fold {fold}")

        dp = DataPreprocessing(train.iloc[tr_idx])
        dp.fit(train)

        (X_imu_tr, X_thm_tr, X_tof_tr, X_roc_tr, y_tr,
         subjects_tr, type_seqs_tr, seq_ids_tr,
         masks_imu_tr, masks_thm_tr, masks_tof_tr,
         sensor_flags_tr) = dp.transform(train.iloc[tr_idx], is_train=1)

        torch.save(dp, os.path.join(ckpt_dir, f'dp_fold{fold}.pt'))

        (X_imu_va, X_thm_va, X_tof_va, X_roc_va, y_va,
         subjects_va, type_seqs_va, seq_ids_va,
         masks_imu_va, masks_thm_va, masks_tof_va,
         sensor_flags_va) = dp.transform(train.iloc[va_idx], is_train=0)

        print("Train X_imu, mask:", X_imu_tr.shape, masks_imu_tr.shape)
        print("Num seq (train/val):", len(np.unique(seq_ids_tr)), len(np.unique(seq_ids_va)))

        best_hf1_ema, memo, memo_loss = Train(
            X_imu_tr, X_thm_tr, X_tof_tr, X_roc_tr, y_tr,
            masks_imu_tr, masks_thm_tr, masks_tof_tr,
            X_imu_va, X_thm_va, X_tof_va, X_roc_va, y_va,
            masks_imu_va, masks_thm_va, masks_tof_va,
            seq_ids_va, cfg, ckpt_dir, f'fold{fold}.pt',
            weights,
            imu_only_prob=0.5, use_mixup=True, mixup_prob=0.5,
            use_augment=True, use_amp=False, label_encoder=dp.label_encoder
        )

        if plot and len(memo_loss) > 0:
            plt.figure(figsize=(10, 5))
            plt.plot(memo_loss, label='Loss')
            plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title(f'Loss over Time | fold {fold}')
            plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

        keys_avail = [k for k in ['RAW', 'EMA', 'EMA_IMU', 'COMBO'] if len(memo)>0 and (k in memo[0])]
        for key in keys_avail:
            if plot:
                hier_f1 = [float(m[key][0]) for m in memo]
                bin_f1  = [float(m[key][1]) for m in memo]
                macro_f1= [float(m[key][2]) for m in memo]
                plt.figure(figsize=(10, 5))
                plt.plot(hier_f1, label=f'{key} Hierarchical F1')
                plt.plot(bin_f1,  label=f'{key} Binary F1')
                plt.plot(macro_f1,label=f'{key} Macro F1')
                plt.xlabel('Val check (per epoch)'); plt.ylabel('F1')
                plt.title(f'F1 over Time | {key} | fold {fold}')
                plt.legend(); plt.grid(True); plt.tight_layout(); plt.show()

        best_ema = float(best_hf1_ema)
        scores_ema.append(best_ema)

        best_imu = None
        if 'EMA_IMU' in keys_avail:
            best_imu = max(float(m['EMA_IMU'][0]) for m in memo)
        scores_imu.append(best_imu if best_imu is not None else np.nan)

        best_combo, best_combo_idx = None, None
        if 'COMBO' in keys_avail:
            for i in range(len(memo)):
                combo_i = float(memo[i]['COMBO'][0])
                if (best_combo is None) or (combo_i > best_combo):
                    best_combo, best_combo_idx = combo_i, i
        scores_combo.append(best_combo if best_combo is not None else np.nan)

        print(f'Best EMA HF1 (all-modal): {best_ema:.4f}')
        if best_imu is not None: print(f'Best EMA HF1 (IMU-only): {best_imu:.4f}')
        if best_combo is not None: print(f'Best COMBO HF1 (avg EMA & EMA-IMU): {best_combo:.4f} @epoch_idx={best_combo_idx}')

        ck_all   = os.path.join(ckpt_dir, f'fold{fold}.pt')
        ck_imu   = os.path.join(ckpt_dir, f'fold{fold}_imu.pt')
        ck_combo = os.path.join(ckpt_dir, f'fold{fold}_combo.pt')
        if os.path.exists(ck_all):   print(f'  ✓ Saved ALL-modal EMA model: {ck_all}')
        if os.path.exists(ck_imu):   print(f'  ✓ Saved IMU-only EMA model: {ck_imu}')
        if os.path.exists(ck_combo): print(f'  ✓ Saved COMBO-best EMA model: {ck_combo}')

    mean_ema   = np.nanmean(scores_ema)   if len(scores_ema)   else float('nan')
    mean_imu   = np.nanmean(scores_imu)   if np.isfinite(scores_imu).any()   else float('nan')
    mean_combo = np.nanmean(scores_combo) if np.isfinite(scores_combo).any() else float('nan')

    print("Mean Hierarchical F1 (EMA)      =", mean_ema)
    if np.isfinite(mean_imu):   print("Mean Hierarchical F1 (EMA-IMU) =", mean_imu)
    if np.isfinite(mean_combo): print("Mean Hierarchical F1 (COMBO)   =", mean_combo)

    return scores_ema


# scores = CrossKFolds(train_df, CFG, ckpt_dir="chkpts", weights = weights)


# !zip -r chkpts.zip /kaggle/working/chkpts


# !pip install kaggle


# # tạo folder chứa package dataset và copy file
# !mkdir -p /kaggle/working/dataset_pkg
# !cp /kaggle/working/chkpts.zip /kaggle/working/dataset_pkg/



# %%bash
# set -e
# mkdir -p ~/.kaggle
# cp "/kaggle/input/hshsgf/kaggle (3).json" ~/.kaggle/kaggle.json
# chmod 600 ~/.kaggle/kaggle.json
# ls -l ~/.kaggle


# import json, datetime, os
# p = "/kaggle/working/dataset_pkg/dataset-metadata.json"
# data = {}
# if os.path.exists(p):
#     data = json.load(open(p))
# ts = datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")  # chỉ lấy ngày
# username = "vphmhunhlong"
# slug = f"chkpts-backup-{ts}".lower()
# data['title'] = slug
# data['id'] = f"{username}/{slug}"
# data['licenses'] = [{"name":"CC0-1.0"}]
# open(p,'w').write(json.dumps(data, indent=2))
# print("Wrote:", p)
# print("New id:", data['id'])



# !kaggle datasets create -p /kaggle/working/dataset_pkg



# (Fallback) ensure_consistent_classes nếu chưa có trong môi trường của bạn
try:
    ensure_consistent_classes
except NameError:
    def ensure_consistent_classes(dps):
        if not dps:
            return
        base = tuple(dps[0].label_encoder.classes_)
        for i, dp in enumerate(dps[1:], start=2):
            cur = tuple(dp.label_encoder.classes_)
            if cur != base:
                raise RuntimeError(
                    f"Label encoders mismatch giữa các folds (fold#{i}). "
                    "Hãy đồng bộ classes_ khi train hoặc tự cài đặt ensure_consistent_classes."
                )

# ============================================================
# 1) LOAD ENSEMBLE (combo + imu) theo từng fold
#    Tự phát hiện file: fold{f}_combo.pt, fold{f}_imu.pt
#    và dp_fold{f}.pt
# ============================================================
def _init_model_for_shape(dp, sequence_for_shape, ckpt_path, device):
    """
    Dựng model đúng shape bằng cách chạy dp.transform để suy kênh,
    rồi nạp checkpoint.
    """
    (X_imu, X_thm, X_tof, X_roc, _y,
     _subjects, _type_seqs, _seq_ids,
     M_imu, M_thm, M_tof, _sensor_flags) = dp.transform(sequence_for_shape, is_train=0)

    c_imu, c_thm, c_tof = X_imu.shape[-1], X_thm.shape[-1], X_tof.shape[-1]
    C_roc = X_roc.shape[-1]

    m = Hybrid3Modal(c_imu, c_thm, c_tof, CFG.WIN_LEN, CFG.N_CLASSES, CFG).to(device)

    # ---- (Re)Init MiniRocket theo kênh X_roc ----
    if getattr(CFG, "USE_MINIROCKET", True) and C_roc > 0:
        dils   = tuple(getattr(CFG, "MR_DILS", (1,2,4,8,16,32)))
        n_bias = int(getattr(CFG, "MR_N_BIAS", 6))
        m.mr = MiniRocketPPV(int(C_roc), dilations=dils, n_bias=n_bias)
        mr_dim = int(C_roc) * len(dils) * n_bias
        m.mr_head = nn.Sequential(
            nn.LayerNorm(mr_dim),
            nn.Linear(mr_dim, 256), nn.GELU(), nn.Dropout(getattr(CFG, "MR_DROP", 0.1)),
            nn.Linear(256, 128)
        )
        m.use_minirocket = True
        m.mr.to(device); m.mr_head.to(device)
    else:
        m.use_minirocket = False

    # ---- Load state dict ----
    state = torch.load(ckpt_path, map_location=device)
    m.load_state_dict(state, strict=False)
    m.eval()

    xshape = {"c_imu": c_imu, "c_thm": c_thm, "c_tof": c_tof, "c_roc": C_roc}
    masks_example = {"M_imu": M_imu, "M_thm": M_thm, "M_tof": M_tof}
    return m, xshape, masks_example


def load_multi_ensemble(ckpt_dir: str, folds=(1,2,3,4,5), sequence_for_shape=None, device=None):
    """
    Tải cả 2 loại mô hình cho mỗi fold nếu tồn tại:
      - fold{f}_combo.pt (đa modal)
      - fold{f}_imu.pt   (chỉ IMU)
    và tiền xử lý tương ứng: dp_fold{f}.pt

    Return:
      models: list dict { "model": nn.Module, "type": "combo"/"imu", "fold": f }
      dps:    list DataPreprocessing theo fold (thứ tự khớp folds)
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if sequence_for_shape is None:
        raise ValueError("Cần 'sequence_for_shape' (DataFrame 1 sequence) để suy kênh đúng như lúc train.")

    models, dps = [], []
    for f in folds:
        # 1) Load DP
        dp_path = os.path.join(ckpt_dir, f"dp_fold{f}.pt")
        if not os.path.exists(dp_path):
            raise FileNotFoundError(f"Không thấy {dp_path}")
        dp = torch.load(dp_path, weights_only=False)
        dps.append(dp)

        # Đồng bộ class encoder giữa các dp
        ensure_consistent_classes(dps)

        # 2) Tải từng loại checkpoint nếu có
        for mtype in ("combo", "imu"):
            ckpt_path = os.path.join(ckpt_dir, f"fold{f}_{mtype}.pt")
            if not os.path.exists(ckpt_path):
                continue

            # Dựng model đúng shape & nạp weights
            m, _xshape, _ = _init_model_for_shape(dp, sequence_for_shape, ckpt_path, device)

            # Fit biases MiniRocket (nếu chưa fit) bằng chính sequence_for_shape
            if getattr(m, "use_minirocket", False) and not bool(getattr(m.mr, "fitted", False)):
                with torch.no_grad():
                    (X_imu_fit, X_thm_fit, X_tof_fit, X_roc_fit, _y_fit,
                     _sub_fit, _typ_fit, _sid_fit,
                     M_imu_fit, _M_thm_fit, _M_tof_fit, _sf_fit) = dp.transform(sequence_for_shape, is_train=0)
                    if X_roc_fit.shape[-1] > 0:
                        n_fit = min(int(getattr(CFG, "MR_MAX_WINDOWS", 4000)), len(X_roc_fit))
                        X_fit = torch.from_numpy(X_roc_fit[:n_fit]).to(device).float()
                        M_fit = torch.from_numpy(M_imu_fit[:n_fit]).to(device)
                        m.mr.fit_biases(X_fit, mask=M_fit, max_windows=n_fit)

            models.append({"model": m, "type": mtype, "fold": f})

    if not models:
        raise RuntimeError("Không tìm thấy checkpoint combo/imu nào trong thư mục đã cho.")

    return models, dps


# ============================================================
# 2) PREDICT với kết hợp động combo vs imu theo coverage cảm biến
# ============================================================
def _window_weight(logits, mi_b, alpha=1.0, beta=1.0, t_model=1.0):
    """
    Tính trọng số cho từng window: w = cov^alpha * conf^beta
    """
    logp = F.log_softmax(logits / t_model, dim=1).float().cpu()
    p    = logp.exp()
    ent  = -(p * logp).sum(dim=1) / math.log(p.size(1) + 1e-12)
    conf = (1.0 - ent).clamp(0.0, 1.0)                     # [B]
    cov  = mi_b.float().mean(dim=1).cpu()                  # [B]
    w    = (cov + 1e-8).pow(alpha) * (conf + 1e-8).pow(beta)
    return logp, w


def _mix_weight(cov_imu, cov_thm, cov_tof,
                coef_imu=0.5, coef_other=0.5,
                gamma=1.3, w_min=0.05, w_max=0.95):
    """
    Trọng số "combo" mượt:
      score_full = coef_imu * cov_imu  +  coef_other * max(cov_thm, cov_tof)
      w_combo    = clamp( score_full ** gamma, w_min, w_max )
      w_imu      = 1 - w_combo
    - Khi chỉ có IMU (cov_thm≈cov_tof≈0)  -> w_combo nhỏ -> ưu tiên mô hình imu
    - Khi non-IMU tốt                     -> w_combo lớn -> ưu tiên mô hình combo
    """
    other = max(float(cov_thm), float(cov_tof))
    score = (coef_imu * float(cov_imu) + coef_other * other) / max(coef_imu + coef_other, 1e-8)
    w_combo = float(np.clip(score ** float(gamma), w_min, w_max))
    w_imu   = 1.0 - w_combo
    return w_combo, w_imu


def predict_ensemble(sequence, models, dps,
            alpha=1.0, beta=1.0, t_model=1.0,
            mix_cfg=None):
    """
    sequence: pandas.DataFrame hoặc polars.DataFrame của 1 sequence
    models:  list[{model, type, fold}]
    dps:     list các dp (theo fold)
    Trả về: (label_str, fused_logp)  — fused_logp là torch.tensor[C]
    """
    import pandas as pd
    try:
        import polars as pl
        is_pl = isinstance(sequence, pl.DataFrame)
    except Exception:
        is_pl = False

    seq_df = sequence.to_pandas() if is_pl else sequence.copy()
    if 'sequence_id' not in seq_df.columns:
        seq_df = seq_df.copy(); seq_df['sequence_id'] = 0
    if 'sequence_counter' in seq_df.columns:
        seq_df = seq_df.sort_values('sequence_counter')

    # Số lớp C
    if hasattr(models[0]["model"], "head_full"):
        C = int(models[0]["model"].head_full.out_features)
    elif hasattr(models[0]["model"], "head_imu"):
        C = int(models[0]["model"].head_imu.out_features)
    else:
        C = int(getattr(CFG, "N_CLASSES", len(getattr(dps[0].label_encoder, "classes_", [])) or 1))

    # Khởi tạo per_fold
    folds_present = sorted({m["fold"] for m in models})
    per_fold = {f: {"combo": None, "imu": None, "covs": None} for f in folds_present}

    # Tính logp hợp nhất trong từng model, sau đó trộn combo/imu trong cùng fold
    for entry in models:
        model, mtype, f = entry["model"], entry["type"], entry["fold"]
        device = next(model.parameters()).device
        model.eval()

        # dp theo fold (giả sử folds bắt đầu từ 1 và dps theo thứ tự đó)
        if not (1 <= f <= len(dps)):
            raise IndexError(f"Fold index {f} không khớp danh sách dps (len={len(dps)}).")
        dp = dps[f-1]

        (X_imu, X_thm, X_tof, X_roc, _y,
         _subjects, _type_seqs, _seq_ids,
         M_imu, M_thm, M_tof, _sensor_flags) = dp.transform(seq_df, is_train=0)

        # MiniRocket fit nếu cần
        if getattr(model, "use_minirocket", False) and hasattr(model, "mr"):
            in_ch = int(X_roc.shape[-1])
            if in_ch > 0:
                if getattr(model.mr, "in_ch", in_ch) != in_ch:
                    raise RuntimeError(f"MiniRocket in_ch mismatch: model={getattr(model.mr,'in_ch',None)}, X_roc={in_ch}.")
                if not bool(getattr(model.mr, "fitted", False)):
                    with torch.no_grad():
                        n_fit = min(int(getattr(CFG, "MR_MAX_WINDOWS", 2000)), len(X_roc))
                        X_fit = torch.from_numpy(X_roc[:n_fit]).to(device).float()
                        M_fit = torch.from_numpy(M_imu[:n_fit]).to(device)
                        model.mr.fit_biases(X_fit, mask=M_fit, max_windows=n_fit)

        xi = torch.from_numpy(X_imu).to(device).float()
        xt = torch.from_numpy(X_thm).to(device).float() if X_thm.shape[-1] > 0 else None
        xo = torch.from_numpy(X_tof).to(device).float() if X_tof.shape[-1] > 0 else None
        xr = torch.from_numpy(X_roc).to(device).float() if X_roc.shape[-1] > 0 else None

        mi = torch.from_numpy(M_imu).to(device)
        mt = torch.from_numpy(M_thm).to(device) if M_thm.size else None
        mo = torch.from_numpy(M_tof).to(device) if M_tof.size else None

        # Hợp nhất theo cửa sổ (w = cov^alpha * conf^beta)
        B, bs = xi.size(0), 256
        w_sum = 0.0
        logp_sum = torch.zeros(C, dtype=torch.float32)
        with torch.no_grad():
            for s in range(0, B, bs):
                e = min(B, s + bs)
                xi_b = xi[s:e]
                xt_b = xt[s:e] if xt is not None else None
                xo_b = xo[s:e] if xo is not None else None
                xr_b = xr[s:e] if xr is not None else None
                mi_b = mi[s:e]
                mt_b = mt[s:e] if mt is not None else None
                mo_b = mo[s:e] if mo is not None else None

                logits, logits_full, logits_imu, rec, alpha_w, present = model(
                    xi_b,
                    xt_b if (xt_b is not None and xt_b.shape[-1] > 0) else None,
                    xo_b if (xo_b is not None and xo_b.shape[-1] > 0) else None,
                    mask_imu=mi_b, mask_thm=mt_b, mask_tof=mo_b,
                    imu_raw=xr_b,
                    force_mode=None
                )

                logp_b, w_b = _window_weight(logits, mi_b, alpha=alpha, beta=beta, t_model=t_model)
                logp_sum += (w_b.unsqueeze(1) * logp_b).sum(dim=0)
                w_sum += float(w_b.sum().item())

        fused = logp_sum / max(w_sum, 1e-8)  # [C]

        # Lưu theo loại
        per_fold[f][mtype] = fused

        # Lưu coverage (dùng để mix combo vs imu trong fold này)
        if per_fold[f]["covs"] is None:
            cov_imu = float(M_imu.mean())
            cov_thm = float(M_thm.mean()) if M_thm.size else 0.0
            cov_tof = float(M_tof.mean()) if M_tof.size else 0.0
            per_fold[f]["covs"] = (cov_imu, cov_thm, cov_tof)

    # Trộn combo vs imu theo coverage trong từng fold
    mix_cfg = mix_cfg or {}
    fold_logp_sum = torch.zeros(C, dtype=torch.float32)
    for f, payload in per_fold.items():
        covs = payload["covs"]
        if covs is None:
            continue
        cov_imu, cov_thm, cov_tof = covs

        # Tính trọng số mềm
        w_combo, w_imu = _mix_weight(
            cov_imu, cov_thm, cov_tof,
            coef_imu =  mix_cfg.get("coef_imu", 0.5),
            coef_other = mix_cfg.get("coef_other", 0.5),
            gamma =      mix_cfg.get("gamma", 1.3),
            w_min =      mix_cfg.get("w_min", 0.05),
            w_max =      mix_cfg.get("w_max", 0.95),
        )

        # Nếu thiếu 1 loại model trong fold đó, dồn trọng số cho loại còn lại
        if payload["combo"] is None and payload["imu"] is not None:
            w_combo, w_imu = 0.0, 1.0
        if payload["imu"] is None and payload["combo"] is not None:
            w_combo, w_imu = 1.0, 0.0

        # Hợp nhất loại trong fold
        logp_fold = torch.zeros(C, dtype=torch.float32)
        if payload["combo"] is not None:
            logp_fold += float(w_combo) * payload["combo"]
        if payload["imu"] is not None:
            logp_fold += float(w_imu)   * payload["imu"]

        fold_logp_sum += logp_fold

    fused_ens = fold_logp_sum / max(len(per_fold), 1)
    pred_id = int(torch.argmax(fused_ens).item())
    label = dps[0].label_encoder.inverse_transform(np.array([pred_id]))[0]
    return str(label), fused_ens


# 1) Tải ensemble
models, dps = load_multi_ensemble(
    ckpt_dir="/kaggle/input/chkpts-backup-20250822155857/kaggle/working/chkpts",
    folds=range(1, CFG.N_SPLIT+1),
    sequence_for_shape=train_df.iloc[:2],  # 1 sequence nhỏ bất kỳ
    device=("cuda" if torch.cuda.is_available() else "cpu")
)

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Hàm dự đoán 1 sequence theo format yêu cầu của nền tảng:
    - Input:
        sequence: pl.DataFrame (hoặc pandas.DataFrame) gồm 1 sequence
        demographics: pl.DataFrame (không dùng trong pipeline hiện tại)
    - Output:
        label (str)
    Yêu cầu: đã có sẵn các hàm/ lớp:
        - load_multi_ensemble(...)
        - predict_ensemble(...)  # hoặc đổi tên từ hàm predict hợp nhất bạn đã có
        - CFG, Hybrid3Modal, MiniRocketPPV, ensure_consistent_classes
    """
    import os
    import numpy as np
    import torch
    import pandas as pd
    try:
        import polars as pl
        _is_pl = isinstance(sequence, pl.DataFrame)
    except Exception:
        _is_pl = False

    # 1) Chuẩn hóa input
    seq_df = sequence.to_pandas() if _is_pl else (sequence.copy() if hasattr(sequence, "copy") else sequence)
    if 'sequence_id' not in seq_df.columns:
        seq_df = seq_df.copy()
        seq_df['sequence_id'] = 0
    if 'sequence_counter' in seq_df.columns:
        seq_df = seq_df.sort_values('sequence_counter')

    # 2) Lazy-load ensemble nếu chưa có trong scope
    global models, dps
    need_load = False
    if 'models' not in globals() or 'dps' not in globals():
        need_load = True
    else:
        need_load = (models is None) or (dps is None) or (len(models) == 0) or (len(dps) == 0)

    if need_load:
        ckpt_dir = os.getenv(
            "CKPT_DIR",
            "/kaggle/input/chkpts-backup-20250822155857/kaggle/working/chkpts"
        )
        folds = range(1, int(getattr(CFG, "N_SPLIT", 5)) + 1)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # sequence_for_shape: lấy 2 dòng đầu của sequence hiện tại (đủ để suy kênh)
        seq_for_shape = seq_df.iloc[:2].copy()
        models, dps = load_multi_ensemble(
            ckpt_dir=ckpt_dir,
            folds=folds,
            sequence_for_shape=seq_for_shape,
            device=device
        )

    # 3) Suy nhãn bằng ensemble
    #    Bạn có thể tinh chỉnh mix_cfg/alpha/beta/t_model tùy ý.
    mix_cfg = dict(coef_imu=0.5, coef_other=0.5, gamma=3.0, w_min=0.05, w_max=0.95)
    label, _fused = predict_ensemble(
        sequence=seq_df,
        models=models,
        dps=dps,
        alpha=1.0,
        beta=1.0,
        t_model=1.0,
        mix_cfg=mix_cfg
    )
    # 4) Trả về string label
    return str(label)



# for i, seq in train_df.groupby('sequence_id'):
#     print(i, ' ', predict(pl.from_pandas(seq), None))


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

