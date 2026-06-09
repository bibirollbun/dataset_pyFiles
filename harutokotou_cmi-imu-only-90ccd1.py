# %% [1] Install & Imports & Utils
!pip install -q pywavelets

import os, gc, math, time, glob, shutil, random, pickle, logging, threading
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from scipy.signal import butter, filtfilt
import pywt
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

# Repro
def set_seed(seed: int = 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", DEVICE)


# %% [2] Metric & Scaler & Dataset (feature engineering含む)
from sklearn.metrics import f1_score

class ParticipantVisibleError(Exception):
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

    def calculate_hierarchical_f1(self, sol: pd.DataFrame, sub: pd.DataFrame):
        invalid_types = {i for i in sub['gesture'].unique() if i not in self.all_classes}
        if invalid_types:
            raise ParticipantVisibleError(f"Invalid gesture values in submission: {invalid_types}")
        y_true_bin = sol['gesture'].isin(self.target_gestures).values
        y_pred_bin = sub['gesture'].isin(self.target_gestures).values
        f1_binary = f1_score(y_true_bin, y_pred_bin, pos_label=True, zero_division=0, average='binary')
        y_true_mc = sol['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        y_pred_mc = sub['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        f1_macro = f1_score(y_true_mc, y_pred_mc, average='macro', zero_division=0)
        return 0.5 * f1_binary + 0.5 * f1_macro, f1_binary, f1_macro


class ToFScaler:
    """-1(無信号)→固定値, 0(故障)→固定値, それ以外は標準化（センサー横一律平均/分散）"""
    def __init__(self):
        self.means_full = np.zeros(320)
        self.stds_full  = np.ones(320)
        self.col_indices = {s: ((s-1)*64, s*64) for s in range(1, 6)}

    def fit(self, X_tof):
        tof_values = X_tof.values if isinstance(X_tof, pd.DataFrame) else X_tof
        if tof_values.shape[1] != 320:
            raise ValueError(f"TOF列は320想定、実際は{tof_values.shape[1]}")
        tof = tof_values.reshape(-1, 5, 64)
        for sensor in range(5):
            valid = (tof[:, sensor, :] != -1) & (tof[:, sensor, :] != 0)
            v = tof[:, sensor, :][valid]
            m = v.mean() if len(v) > 0 else 0.0
            s = v.std()  if len(v) > 0 else 1.0
            s = 1.0 if s < 1e-7 else s
            st, ed = self.col_indices[sensor+1]
            self.means_full[st:ed] = m
            self.stds_full[st:ed]  = s
        return self

    def transform(self, X_tof):
        arr = X_tof.values.copy() if isinstance(X_tof, pd.DataFrame) else X_tof.copy()
        valid = (arr != -1) & (arr != 0)
        no_sig = (arr == -1)
        failure = (arr == 0)
        out = np.zeros_like(arr, dtype=float)
        idx = np.where(valid)[1]
        out[valid] = (arr[valid] - self.means_full[idx]) / self.stds_full[idx]
        out[no_sig] = 10.0
        out[failure] = -10.0
        if isinstance(X_tof, pd.DataFrame):
            return pd.DataFrame(out, columns=X_tof.columns, index=X_tof.index)
        return out


class GestureDataset(Dataset):
    def __init__(self, sequence_dir, label_dir, metadata_df, max_len=None,
                 is_train=False, scale_path=None, use_augmentation=False,
                 drift_std=0.01, drift_max=0.05):
        self.sequence_dir = sequence_dir
        self.label_dir = label_dir
        self.metadata = metadata_df.reset_index(drop=True)
        self.sequence_ids = self.metadata['sequence_id'].tolist()
        self.max_len = max_len
        self.is_train = is_train
        self.scale_path = scale_path
        self.use_augmentation = use_augmentation and is_train
        self.drift_std = drift_std
        self.drift_max = drift_max

        # 定数
        self.tof_dim = 320

        # ラベルエンコーダ
        self.gesture_encoder = LabelEncoder()
        self.gesture_encoder.fit(self.metadata['gesture'])

        # スケーラ（IMU+THM、DEMO、TOF）
        self.non_tof_scaler = StandardScaler()  # IMU+THM 用
        self.demo_scaler    = StandardScaler()  # DEMO(7) 用
        self.tof_scaler     = ToFScaler()       # TOF(320) 用

        # シーケンス読み込み（raw -> FE済みの feature 行列の list）
        self.sequence_data = self._load_sequences()

        # スケーラの fit/保存 or 読み込み → 正規化キャッシュ
        if scale_path is not None:
            if is_train:
                self._fit_scalers()
                self._save_scalers(scale_path)
                self._normalize_and_cache_sequences()
            else:
                with open(os.path.join(scale_path, 'train_scalers.pkl'), 'rb') as f:
                    obj = pickle.load(f)
                # 後方互換：2タプル(旧) or 3タプル(新)
                if isinstance(obj, tuple) and len(obj) == 3:
                    self.non_tof_scaler, self.tof_scaler, self.demo_scaler = obj
                elif isinstance(obj, tuple) and len(obj) == 2:
                    self.non_tof_scaler, self.tof_scaler = obj
                    self.demo_scaler = self._identity_scaler(7)  # DEMOは恒等スケール
                else:
                    raise ValueError("train_scalers.pkl is not a (2|3)-tuple.")
                self._normalize_and_cache_sequences()
        else:
            # スケールしない（そのまま FE 結果を保持）
            self.cached_sequences = self.sequence_data.copy()

        print(f"Dataset ready: {len(self)} sequences")
        lens = [len(s) for s in self.cached_sequences]
        if len(lens) > 0:
            print(f"len mean={np.mean(lens):.1f}, max={np.max(lens)}")

    # ===== Utils (filters & wavelet) =====
    def butter_lowpass(self, data, cutoff, fs, order=4):
        b, a = butter(order, cutoff/(0.5*fs), btype='low');  return filtfilt(b, a, data)
    def butter_highpass(self, data, cutoff, fs, order=4):
        b, a = butter(order, cutoff/(0.5*fs), btype='high'); return filtfilt(b, a, data)
    def butter_bandpass(self, data, lowcut, highcut, fs, order=4):
        b, a = butter(order, [lowcut/(0.5*fs), highcut/(0.5*fs)], btype='band'); return filtfilt(b, a, data)
    def wavelet_denoise(self, data, wavelet='db4', level=1):
        coeffs = pywt.wavedec(data, wavelet, mode='symmetric')
        coeffs[1:] = [np.zeros_like(v) for v in coeffs[1:]]
        return pywt.waverec(coeffs, wavelet, mode='symmetric')[:len(data)]

    # ===== IMU 補助 =====
    def _remove_gravity_from_acc(self, acc_values, quat_values):
        T = acc_values.shape[0]
        out = np.zeros_like(acc_values)
        g_world = np.array([0, 0, 9.81])
        for i in range(T):
            q = quat_values[i]
            if np.all(np.isnan(q)) or np.all(np.isclose(q, 0)):
                out[i] = acc_values[i]; continue
            try:
                rot = R.from_quat(q)
                g_sensor = rot.apply(g_world, inverse=True)
                out[i] = acc_values[i] - g_sensor
            except ValueError:
                out[i] = acc_values[i]
        return out

    def _calculate_angular_velocity_from_quat(self, quat_values, time_delta=1/10):
        T = quat_values.shape[0]
        ang = np.zeros((T, 3))
        for i in range(T-1):
            q1, q2 = quat_values[i], quat_values[i+1]
            if (np.all(np.isnan(q1)) or np.all(np.isclose(q1,0)) or
                np.all(np.isnan(q2)) or np.all(np.isclose(q2,0))):
                continue
            try:
                r1, r2 = R.from_quat(q1), R.from_quat(q2)
                delta = r1.inv() * r2
                ang[i] = delta.as_rotvec() / time_delta
            except ValueError:
                pass
        return ang

    def _calculate_angular_distance(self, quat_values):
        T = quat_values.shape[0]
        ang = np.zeros(T)
        for i in range(T-1):
            q1, q2 = quat_values[i], quat_values[i+1]
            if (np.all(np.isnan(q1)) or np.all(np.isclose(q1,0)) or
                np.all(np.isnan(q2)) or np.all(np.isclose(q2,0))):
                ang[i] = 0; continue
            try:
                r1, r2 = R.from_quat(q1), R.from_quat(q2)
                angle = np.linalg.norm((r1.inv()*r2).as_rotvec())
                ang[i] = angle
            except ValueError:
                ang[i] = 0
        return ang

    # ===== Feature Engineering =====
    def _feature_engineering(self, seq):
        """
        RAW seq 形式の想定:
          [acc_x, acc_y, acc_z, rot_w, rot_x, rot_y, rot_z, thm(5), tof(320), demo(7)]
        返却featuresは（いったん）:
          [IMU派生(36), thm(5), tof(320), demo(7)]
        ※ 後段の正規化で [IMU+THM][DEMO][TOF] に並べ替えてキャッシュします
        """
        fs = 50
        imu = seq[:, :7]         # acc(3) + rot(wxyz)(4)
        acc = imu[:, :3]
        rot = imu[:, 3:7]        # wxyz

        acc_lp = np.stack([self.butter_lowpass(acc[:,i], 2, fs) for i in range(3)], axis=1)
        acc_hp = np.stack([self.butter_highpass(acc[:,i], 0.5, fs) for i in range(3)], axis=1)
        acc_bp = np.stack([self.butter_bandpass(acc[:,i], 0.5, 5, fs) for i in range(3)], axis=1)
        acc_wv = np.stack([self.wavelet_denoise(acc[:,i], 'db4', 1) for i in range(3)], axis=1)

        acc_mag = np.sqrt((acc**2).sum(1))
        rot_angle = 2 * np.arccos(np.clip(rot[:,0], -1, 1))
        acc_mag_jerk = np.diff(acc_mag, prepend=acc_mag[0])
        rot_angle_vel = np.diff(rot_angle, prepend=rot_angle[0])
        linear_acc = self._remove_gravity_from_acc(acc, rot[:, [1,2,3,0]])  # x,y,z,w
        linear_acc_mag = np.sqrt((linear_acc**2).sum(1))
        linear_acc_mag_jerk = np.diff(linear_acc_mag, prepend=linear_acc_mag[0])
        angular_vel = self._calculate_angular_velocity_from_quat(rot[:, [1,2,3,0]])
        angular_distance = self._calculate_angular_distance(rot[:, [1,2,3,0]])

        # 36ch の IMU派生特徴
        new_imu = np.concatenate([
            acc, 
            seq[:, 3:7],          # 生 rot(wxyz)
            rot,                  # rot 再掲（合計で +4ch）
            acc_mag[:,None], rot_angle[:,None],
            acc_mag_jerk[:,None], rot_angle_vel[:,None],
            linear_acc, linear_acc_mag[:,None], linear_acc_mag_jerk[:,None],
            angular_vel, angular_distance[:,None],
            acc_lp, acc_hp, acc_bp, acc_wv,
        ], axis=1)  # -> 36

        # thm(5)+tof(320)+demo(7)
        other = seq[:, 7:]
        features = np.concatenate([new_imu, other], axis=1)
        return features  # 返却時点では [IMU(36), thm(5), tof(320), demo(7)]

    # ===== IO / scaler =====
    def _load_sequences(self):
        out = []
        for sid in tqdm(self.sequence_ids, desc="Load seq", leave=False):
            subject = self.metadata.loc[self.metadata['sequence_id']==sid, 'subject'].iloc[0]
            p = os.path.join(self.sequence_dir, str(subject), f"{sid}.npy")
            if os.path.exists(p):
                raw = np.load(p)
                feat = self._feature_engineering(raw)
                out.append(feat)
        return out

    def _identity_scaler(self, n):
        s = StandardScaler()
        s.mean_ = np.zeros(n, dtype=float)
        s.var_ = np.ones(n, dtype=float)
        s.scale_ = np.ones(n, dtype=float)
        s.n_features_in_ = n
        s.n_samples_seen_ = np.array([1], dtype=np.int64)
        return s

    def _fit_scalers(self):
        """
        FE後 features の並びは [IMU(36), thm(5), tof(320), demo(7)] を想定。
        ここで DEMO を独立スケールし、キャッシュ時に [IMU+THM][DEMO][TOF] に組み直す。
        """
        all_seq = np.vstack(self.sequence_data)
        D = all_seq.shape[1]
        demo = all_seq[:, -7:]                                   # (.., 7)
        tof  = all_seq[:, D - (self.tof_dim + 7) : D - 7]        # (.., 320)
        imu_thm = all_seq[:, : D - (self.tof_dim + 7)]           # (.., 36+5)

        self.non_tof_scaler.fit(imu_thm)
        self.demo_scaler.fit(demo)
        self.tof_scaler.fit(tof)

    def _save_scalers(self, scale_path):
        Path(scale_path).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(scale_path, 'train_scalers.pkl'), 'wb') as f:
            # 3タプル保存（後方互換のためTOFを2番目に保持）
            pickle.dump((self.non_tof_scaler, self.tof_scaler, self.demo_scaler), f)

    def _normalize_and_cache_sequences(self):
        """
        features([IMU, thm, tof, demo]) → 正規化 → 最終的に
        [IMU+THM][DEMO][TOF] の順にキャッシュ（以降の __getitem__ の分割ロジックと一致）
        """
        self.cached_sequences = []
        for seq in tqdm(self.sequence_data, desc="Normalize", leave=False):
            D = seq.shape[1]
            demo = seq[:, -7:]
            tof  = seq[:, D - (self.tof_dim + 7) : D - 7]
            imu_thm = seq[:, : D - (self.tof_dim + 7)]

            n_imu_thm = self.non_tof_scaler.transform(imu_thm)
            n_demo    = self.demo_scaler.transform(demo)
            n_tof     = self.tof_scaler.transform(tof)

            # 最終構成を [IMU+THM][DEMO][TOF] に統一
            self.cached_sequences.append(
                np.concatenate([n_imu_thm, n_demo, n_tof], axis=1)
            )
        self.sequence_data = None  # free

    # ===== Augment (train only) =====
    def _jitter(self, x, sigma=0.05):
        return x + np.random.normal(0., sigma, size=x.shape)

    def _time_mask(self, x, max_len=20):
        L = x.shape[0]
        m = np.random.randint(1, max_len+1)
        st = np.random.randint(0, max(1, L-m))
        x[st:st+m] = 0
        return x

    def _feature_mask(self, x, max_k=15):
        C = x.shape[1]
        k = np.random.randint(1, max_k+1)
        idx = np.random.choice(C, k, replace=False)
        x[:, idx] = 0
        return x

    def _motion_drift(self, imu_part):
        T = imu_part.shape[0]
        drift = np.cumsum(np.random.normal(scale=self.drift_std, size=(T,1)), axis=0)
        drift = np.clip(drift, -self.drift_max, self.drift_max)
        # 物理チャネル例（accや一部の派生に疑似ドリフト）
        imu_part[:, 0:3]   += drift   # acc
        imu_part[:, 10:13] += drift
        imu_part[:, 15:18] += drift
        return imu_part

    def _apply_augmentations(self, x):
        """
        x は正規化後・キャッシュ後の配列で [IMU+THM][DEMO7][TOF320]
        DEMO 列は保護（統計量を壊さない）
        """
        D = x.shape[1]
        demo_st = D - self.tof_dim - 7
        demo_ed = D - self.tof_dim
        # DEMO 以外の列インデックス
        aug_cols = np.r_[0:demo_st, demo_ed:D]

        if np.random.rand() < 0.7:
            noise = np.zeros_like(x)
            noise[:, aug_cols] = np.random.normal(0., 0.05, size=(x.shape[0], len(aug_cols)))
            x = x + noise
        if np.random.rand() < 0.5:
            L = x.shape[0]
            m = np.random.randint(1, 21)
            st = np.random.randint(0, max(1, L-m))
            x[st:st+m, aug_cols] = 0
        if np.random.rand() < 0.5:
            C = len(aug_cols)
            k = np.random.randint(1, 16)
            idx = np.random.choice(aug_cols, k, replace=False)
            x[:, idx] = 0
        if np.random.rand() < 0.5:
            # IMU派生を主対象に疑似ドリフト（先頭36chがIMU派生）
            imu_end = 36
            imu = x[:, :imu_end]
            oth = x[:, imu_end:]
            imu = self._motion_drift(imu)
            x = np.concatenate([imu, oth], axis=1)
        return x

    # ===== PyTorch Dataset =====
    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        """
        ここでは [IMU+THM][DEMO][TOF] を前提に分割する。
          - TOF: 末尾320列
          - DEMO: TOF直前の7列
          - IMU+THM: それ以前（IMU=先頭~(末尾-7-5)、THM=その直前5列）
        """
        seq = self.cached_sequences[idx].copy()
        if self.use_augmentation:
            seq = self._apply_augmentations(seq)

        # 長さ制限
        if self.max_len and len(seq) > self.max_len:
            seq = seq[-self.max_len:]

        # 分割
        D = seq.shape[1]
        tof = seq[:, -self.tof_dim:]                   # (T, 320)
        imu_thm_demo = seq[:, :-self.tof_dim]          # (T, *)
        demo = imu_thm_demo[0, -7:]                    # (7,)
        imu_thm = imu_thm_demo[:, :-7]
        thm  = imu_thm[:, -5:]                         # (T, 5)
        imu  = imu_thm[:, :-5]                         # (T, 36)
        thm_tof = np.concatenate([thm, tof], axis=1)   # (T, 325)

        # フェーズラベル
        sid = self.sequence_ids[idx]
        subject = self.metadata.loc[self.metadata['sequence_id']==sid, 'subject'].iloc[0]
        p_lab = os.path.join(self.label_dir, str(subject), f"{sid}.npy")
        phase_labels = np.load(p_lab).astype(np.float32)
        if self.max_len and len(phase_labels) > len(seq):
            phase_labels = phase_labels[-len(seq):]

        # クラスラベル
        meta = self.metadata.iloc[idx]
        y = self.gesture_encoder.transform([meta['gesture']])[0]

        return {
            'imu': torch.tensor(imu, dtype=torch.float32),
            'thm_tof': torch.tensor(thm_tof, dtype=torch.float32),
            'demo': torch.tensor(demo, dtype=torch.float32),
            'phase_labels': torch.tensor(phase_labels, dtype=torch.long),
            'length': len(imu),
            'sequence_id': sid,
            'gesture_labels': torch.tensor(y, dtype=torch.long),
        }

def collate_fn(batch):
    batch.sort(key=lambda x: x['length'], reverse=True)
    imu = [b['imu'] for b in batch]
    thm_tof = [b['thm_tof'] for b in batch]
    demo = torch.stack([b['demo'] for b in batch])
    phase = [b['phase_labels'] for b in batch]
    lengths = torch.tensor([b['length'] for b in batch], dtype=torch.long)
    seq_ids = [b['sequence_id'] for b in batch]
    y = torch.stack([b['gesture_labels'] for b in batch])

    imu_p = pad_sequence(imu, batch_first=True, padding_value=0)
    thm_tof_p = pad_sequence(thm_tof, batch_first=True, padding_value=0)
    phase_p = pad_sequence(phase, batch_first=True, padding_value=3)
    mask = (torch.arange(imu_p.size(1))[None, :] < lengths[:, None]).float()

    return {'imu': imu_p, 'thm_tof': thm_tof_p, 'demo': demo,
            'phase_labels': phase_p, 'mask': mask, 'lengths': lengths,
            'gesture_labels': y, 'sequence_ids': seq_ids}


# %% [3] Model (二段ヘッド) + Blocks
class FiLM(nn.Module):
    def __init__(self, feat_dim: int, cond_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, hidden), nn.SiLU(inplace=True),
            nn.Linear(hidden, feat_dim * 2)
        )
    def forward(self, feat, cond):               # feat: (B, D), cond: (B, C)
        gamma, beta = self.net(cond).chunk(2, dim=-1)
        return feat * (1 + torch.tanh(gamma)) + beta

class CoordAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.mid = max(8, channels // reduction)
        self.comp = nn.Sequential(
            nn.Conv1d(channels, self.mid, 1, bias=False),
            nn.BatchNorm1d(self.mid),
            nn.SiLU(inplace=True)
        )
        self.time_conv = nn.Conv1d(1, 1, kernel_size=5, padding=2, bias=False)
        self.channel_conv = nn.Conv1d(self.mid, channels, kernel_size=1, bias=False)
        self.sig = nn.Sigmoid()

    def forward(self, x):  # x: (B, C, T)
        f = self.comp(x)                   # (B, mid, T)
        t_attn = self.sig(self.time_conv(f.mean(1, keepdim=True)))   # (B,1,T)
        c_attn = self.sig(self.channel_conv(f.mean(2, keepdim=True)))# (B,C,1)
        return x * t_attn * c_attn

class ResidualCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, Model,
                 reduction=8, pool_size=2, dropout=0.3, **_):  # ★ 余剰キーワードを受け流す
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.attention = Model(out_channels, reduction)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        sc = self.shortcut(x)
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        y = self.attention(y)
        y = F.relu(y + sc)
        y = self.pool(y)
        y = self.dropout(y)
        return y

class MLPAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(d, d//8), nn.SiLU(inplace=True), nn.Linear(d//8, 1))
    def forward(self, x):  # (B,T,C)
        w = F.softmax(self.attn(x), dim=1)  # (B,T,1)
        return (x * w).sum(1)

class DilatedTCNBlock(nn.Module):
    def __init__(self, channels=128, kernel_size=3, dilations=(1,2,4), dropout=0.1):
        super().__init__()
        layers = []
        for d in dilations:
            pad = (kernel_size - 1) * d // 2  # 長さ保持
            layers += [
                nn.Conv1d(channels, channels, kernel_size,
                          padding=pad, dilation=d, bias=False),
                nn.BatchNorm1d(channels),
                nn.SiLU(inplace=True),
                nn.Dropout(dropout),
            ]
        self.net = nn.Sequential(*layers)
        self.proj = nn.Identity()  # チャネル不変なので恒等
    def forward(self, x):  # x: (B,C,T)
        y = self.net(x)
        return y + self.proj(x)  # 残差

class IMUOnlyHierModel(nn.Module):
    def __init__(self, imu_dim, tof_dim, n_classes, weight_decay=1e-4):
        super().__init__()
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim
        self.n_classes = n_classes
        self.weight_decay = weight_decay

        # --- IMU branch: CNN 2段 ---
        self.imu_block1 = ResidualCNNBlock(imu_dim, 64, 3, dropout=0.3, Model=CoordAttention, weight_decay=weight_decay)
        self.imu_block2 = ResidualCNNBlock(64, 128, 5, dropout=0.3, Model=CoordAttention, weight_decay=weight_decay)

        # --- ★ 追加: 軽量TCN（長さ保持） ---
        self.tcn = DilatedTCNBlock(channels=128, kernel_size=3, dilations=(1,2,4), dropout=0.1)

        # --- RNN + Attention ---
        self.bigru = nn.GRU(128, 128, bidirectional=True, batch_first=True)
        self.gru_dropout = nn.Dropout(0.4)
        self.mlp_attention = MLPAttention(256)

        # --- demographics(7) で feat(256) をFiLM ---
        self.film = FiLM(feat_dim=256, cond_dim=7)

        # --- Head ---
        self.dense1 = nn.Linear(256, 256, bias=False); self.bn_dense1 = nn.BatchNorm1d(256); self.drop1 = nn.Dropout(0.5)
        self.dense2 = nn.Linear(256, 128, bias=False); self.bn_dense2 = nn.BatchNorm1d(128); self.drop2 = nn.Dropout(0.3)
        self.classifier = nn.Linear(128, n_classes)

    def forward(self, imu, thm_tof, demo):
        x = imu.transpose(1, 2)        # (B, C, T)
        x = self.imu_block1(x)         # (B, 64, T/2)
        x = self.imu_block2(x)         # (B,128, T/4)
        x = self.tcn(x)                # (B,128, T/4)
        x = x.transpose(1, 2)          # (B, T/4, 128)

        x, _ = self.bigru(x)           # (B, T/4, 256)
        x = self.gru_dropout(x)
        feat = self.mlp_attention(x)   # (B, 256)

         # --- FiLM で条件付け ---
        demo = demo.float()                        # (B,7)
        feat = self.film(feat, demo)

        h = F.relu(self.bn_dense1(self.dense1(feat))); h = self.drop1(h)
        h = F.relu(self.bn_dense2(self.dense2(h)));    h = self.drop2(h)
        logits = self.classifier(h)
        return logits


# %% [4] Losses (Logit-Adjusted CE, hierarchical loss) & train_one_epoch

# ==== HierLoss toggle ====
USE_HIER = False   # ← 階層ロスを使わない
HIER_ALPHA = 0.0   # 一応ゼロで固定
HIER_BETA  = 0.0

class LogitAdjustedCE(nn.Module):
    def __init__(self, class_priors: torch.Tensor, tau: float = 1.0):
        super().__init__()
        self.register_buffer('bias', (-tau * torch.log(class_priors.clamp_min(1e-6))).float())
    def forward(self, logits, target):
        return F.cross_entropy(logits + self.bias.to(logits.device), target)

def hierarchical_loss(logits, bin_logit, target, target_mask, la_ce, alpha=0.5, beta=0.5):
    # まず通常の多クラス損失
    loss_mc = la_ce(logits, target)

    # ★ ここを追加：alpha=beta=0 のときは即返す（余計な計算をしない）
    if (alpha <= 0) and (beta <= 0):
        return loss_mc

    device = logits.device
    tmask = target_mask.to(device)
    y_bin = tmask[target].float()
    loss_bin = F.binary_cross_entropy_with_logits(bin_logit.squeeze(-1), y_bin)

    p = F.softmax(logits, dim=1)
    p_target_sum = p[:, tmask].sum(dim=1)
    p_bin = torch.sigmoid(bin_logit.squeeze(-1))
    loss_cons = F.mse_loss(p_bin, p_target_sum)

    return loss_mc + alpha*loss_bin + beta*loss_cons

def bin_logit_from_logits(logits: torch.Tensor, target_mask: torch.Tensor) -> torch.Tensor:
    """
    logits: (B, C)
    target_mask: (C,) bool  — True が targetクラス
    返り値: (B, 1) の二値ロジット（target vs non-target）
    """
    device = logits.device
    tmask = target_mask.to(device)
    if tmask.dtype != torch.bool:
        tmask = tmask.bool()

    B, C = logits.shape
    n_t  = int(tmask.sum().item())
    n_nt = C - n_t

    # 通常経路：target群とnon-target群のlog-sum-exp差 = log-oddsに一致
    if n_t > 0 and n_nt > 0:
        lse_t  = torch.logsumexp(logits[:, tmask], dim=1)   # (B,)
        lse_nt = torch.logsumexp(logits[:, ~tmask], dim=1)  # (B,)
        bl = lse_t - lse_nt                                  # (B,)
    else:
        # 念のための保険（全クラスが片側に寄る異常系）：確率からlogit再構成
        p  = torch.softmax(logits, dim=1)                   # (B, C)
        p_t = p[:, tmask].sum(dim=1) if n_t > 0 else torch.zeros(B, device=device)
        p_t = p_t.clamp(1e-6, 1-1e-6)
        bl = torch.log(p_t / (1 - p_t))                     # (B,)

    return bl.unsqueeze(-1)  # (B,1)


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    la_ce: torch.nn.Module,
    target_mask_tensor: torch.Tensor,
    warmup_epochs: int = 3,
    eta_min: float = 1e-6,
    grad_clip: float = 1.0,
    scheduler=None,  # ★ 追加：外部スケジューラ（例：CosineAnnealingWarmRestarts）
):
    """
    scheduler を渡した場合：スケジューラ側でLR管理（各バッチ後に step）。
    渡さない場合：従来どおり warmup + cosine で手動更新。
    """
    model.train()
    running_loss = 0.0
    preds_all, t_all = [], []

    base_lr = optimizer.defaults.get('lr', optimizer.param_groups[0]['lr'])
    num_batches = max(1, len(dataloader))

    for bi, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1} [Train]", leave=False)):
        # ---------- LR 更新 ----------
        if scheduler is None:
            # 手動: warmup + cosine
            cur_iter = epoch * num_batches + bi
            cur_frac = cur_iter / max(1, num_batches)
            if cur_frac < warmup_epochs:
                lr = base_lr * (0.001 + (cur_frac / max(1e-12, warmup_epochs)) * 0.999)
            else:
                num = (cur_frac - warmup_epochs)
                den = max(1e-12, (total_epochs - warmup_epochs))
                cos = 0.5 * (1 + math.cos(math.pi * (num / den)))
                lr = eta_min + (base_lr - eta_min) * cos
            for pg in optimizer.param_groups:
                pg['lr'] = lr
        # scheduler ありの場合は optimizer の lr は scheduler が管理（ここでは変更しない）

        # ---------- forward ----------
        imu = batch['imu'].to(device)
        thm_tof = batch['thm_tof'].to(device)
        demo = batch['demo'].to(device)
        y = batch['gesture_labels'].to(device)

        optimizer.zero_grad(set_to_none=True)

        out = model(imu, thm_tof, demo)
        if isinstance(out, tuple) and len(out) == 2:
            logits, bin_logit = out
        else:
            logits = out
            # モデルに二値ヘッドが無い場合のフォールバック（target_mask_tensor から作る）
            bin_logit = bin_logit_from_logits(logits, target_mask_tensor)

        # ---------- loss / backward ----------
        loss = hierarchical_loss(logits, bin_logit, y, target_mask_tensor, la_ce, alpha=0.5, beta=0.5)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # ---------- scheduler step（スケジューラ使用時のみ） ----------
        if scheduler is not None:
            # CosineAnnealingWarmRestarts などは fractional epoch で step するのが定石
            scheduler.step(epoch + bi / num_batches)

        running_loss += float(loss.item())
        with torch.no_grad():
            preds_all.extend(logits.argmax(1).detach().cpu().tolist())
            t_all.extend(y.detach().cpu().tolist())

    avg_loss = running_loss / max(1, len(dataloader))
    inv = dataloader.dataset.gesture_encoder.inverse_transform
    f1, f1_bin, f1_mac = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': inv(np.array(t_all, dtype=int))}),
        pd.DataFrame({'gesture': inv(np.array(preds_all, dtype=int))})
    )
    return avg_loss, f1, f1_bin, f1_mac


# %% [5] Multi-crop validate, crops helper
def make_crops(x: np.ndarray, L=256, K=5):
    T = len(x)
    if T <= L: return [x[-L:]]
    starts = np.linspace(0, T-L, num=K, dtype=int)
    return [x[s:s+L] for s in starts]


# ---- TTA (time-shift + jitter) ----
TTA_SHIFTS = (-16, 0, +16)   # 0.32秒相当（50Hz想定）
TTA_SIGMA  = 0.01            # IMUへの微小ノイズ
USE_TTA_INFER = True   # 推論でTTAを使わない


def shift_pad(x: np.ndarray, shift: int) -> np.ndarray:
    """時系列(T,C)をゼロパディングで時間シフト（循環しない）"""
    T, C = x.shape
    y = np.zeros_like(x)
    if shift == 0:
        return x
    if shift > 0:
        y[shift:] = x[:T-shift]
    else:
        s = -shift
        y[:T-s] = x[s:]
    return y

def tta_variants_per_crop(imu_crop: np.ndarray, thm_tof_crop: np.ndarray):
    """各cropからTTA 3本（±16, 0）を生成。JitterはIMUのみ."""
    outs = []
    for sh in TTA_SHIFTS:
        imu_s  = shift_pad(imu_crop, sh)
        thm_s  = shift_pad(thm_tof_crop, sh)
        # 微小Jitter（IMUのみ）
        if TTA_SIGMA > 0:
            imu_s  = imu_s + np.random.normal(0.0, TTA_SIGMA, imu_s.shape).astype(imu_s.dtype)
        outs.append((imu_s, thm_s))
    return outs

def validate(model, dataloader, device, temps=None):
    model.eval()
    val_preds, val_targets = [], []
    Tmc = float(temps.get('T_mc', 1.0)) if temps is not None else 1.0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[Val]", leave=False):
            imu_p = batch['imu'].cpu().numpy()
            thmtof_p = batch['thm_tof'].cpu().numpy()
            demo = batch['demo'].to(device)
            lengths = batch['lengths'].cpu().numpy()
            labels = batch['gesture_labels'].cpu().numpy()

            for i in range(len(lengths)):
                T = lengths[i]
                imu_np = imu_p[i, :T, :]
                thm_np = thmtof_p[i, :T, :]
                imu_cs = make_crops(imu_np, L=256, K=5)
                thm_cs = make_crops(thm_np, L=256, K=5)

                # --- cropごとに TTAの logits を平均 ---
                crop_logits = []
                for ic, tc in zip(imu_cs, thm_cs):
                    tta_logits = []
                    for (ic2, tc2) in tta_variants_per_crop(ic, tc):
                        it = torch.tensor(ic2, dtype=torch.float32).unsqueeze(0).to(device)
                        tt = torch.tensor(tc2, dtype=torch.float32).unsqueeze(0).to(device)
                        dt = demo[i].unsqueeze(0)
                        out = model(it, tt, dt)
                        logits = out[0] if isinstance(out, tuple) else out
                        tta_logits.append(logits.cpu())
                    # TTA logits 平均
                    logits_mean = torch.stack(tta_logits).mean(0)  # (1,C)
                    crop_logits.append(logits_mean)

                # --- crop 平均 → 温度 → softmax ---
                logits_cropmean = torch.stack(crop_logits).mean(0)       # (1,C)
                logits_cropmean = logits_cropmean / max(Tmc, 1e-3)
                prob = F.softmax(logits_cropmean, dim=1).cpu()

                val_preds.append(int(prob.argmax(1).item()))
                val_targets.append(int(labels[i]))

    inv = dataloader.dataset.gesture_encoder.inverse_transform
    f1, f1_bin, f1_mac = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': inv(np.array(val_targets, dtype=int))}),
        pd.DataFrame({'gesture': inv(np.array(val_preds,   dtype=int))})
    )
    return 0.0, f1, f1_bin, f1_mac


# %% [6] Inference helpers: load models+scalers+temps, temperature calibration
def load_models_and_scalers(models_dir, model_class, n_classes, device):
    model_paths = sorted(glob.glob(os.path.join(models_dir, "model_fold*_best.pth")))
    if not model_paths:
        raise FileNotFoundError(f"No models found in {models_dir}")
    models, scalers, temps = [], [], []
    for p in model_paths:
        k = os.path.basename(p).split('_best.pth')[0].split('fold')[-1]
        model = model_class(36, 325, n_classes).to(device)
        model.load_state_dict(torch.load(p, map_location=device, weights_only=False))
        model.eval()
        models.append(model)
        with open(os.path.join(models_dir, f"scaler_fold{k}.pkl"), 'rb') as f:
            scalers.append(pickle.load(f))  # (non_tof_scaler, tof_scaler)
        t_path = os.path.join(models_dir, f"temp_fold{k}.pkl")
        if os.path.exists(t_path):
            with open(t_path, 'rb') as f:
                temps.append(pickle.load(f))  # {'T_mc': float, 'T_bin': float}
        else:
            temps.append({'T_mc': 1.0, 'T_bin': 1.0})
    return models, scalers, temps

def calibrate_temperature(
    model,
    val_loader,
    device,
    init_T: float = 1.0,
    max_iter: int = 20,
    max_samples: int = 2000,   # バリデーションから使う上限サンプル数
):
    """
    温度スケーリング（多クラス側のみ）を安定・高速に実行。
    - モデル出力logitsを事前にno_gradでキャッシュ → 逆伝播はTのみに流れる
    - LBFGSでTを更新（logits/TでCE最小化）
    """
    model.eval()

    # 1) 検証logitsをキャッシュ（no_grad & detach）
    cached_logits, cached_targets, seen = [], [], 0
    with torch.no_grad():
        for batch in val_loader:
            if seen >= max_samples:
                break
            imu = batch['imu'].to(device)
            thm_tof = batch['thm_tof'].to(device)
            demo = batch['demo'].to(device)
            y = batch['gesture_labels'].to(device)

            out = model(imu, thm_tof, demo)
            logits = out[0] if isinstance(out, tuple) else out
            cached_logits.append(logits.detach())  # ← detach が重要
            cached_targets.append(y.detach())
            seen += y.size(0)

    if not cached_logits:
        # データが無い場合はスキップ
        return 1.0

    logits_cat = torch.cat(cached_logits, 0).to(device)
    targets_cat = torch.cat(cached_targets, 0).to(device)

    # 2) Tのみ最適化
    T = torch.tensor([init_T], dtype=torch.float32, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([T], lr=0.1, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        # Tの暴走防止（微小値クリップ）
        T_pos = T.clamp_min(1e-3)
        loss = F.cross_entropy(logits_cat / T_pos, targets_cat)
        loss.backward()  # 勾配はTのみに流れる（logitsは定数）
        return loss

    opt.step(closure)
    return float(T.detach().cpu().item())



# %% [7] Train (5-fold) with fold別scaler保存 & 温度スケーリング保存
TRAIN = False  # Kaggle提出ノートブックでは False に

if TRAIN:
    # Logging
    Path("logs").mkdir(exist_ok=True)
    logging.basicConfig(
        filename="logs/imutrain.log",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    metadata_path = '/kaggle/input/cmi-data/data/train_labels.csv'
    sequence_dir  = '/kaggle/input/cmi-data/data/train_sequences'
    label_dir     = '/kaggle/input/cmi-data/data/phase_labels'

    NUM_FOLDS = 5
    SEED = 2
    BATCH_SIZE = 32
    MAX_LEN = 256
    LR = 1e-3
    LR_MIN = 1e-6
    EPOCHS = 100
    EARLY_STOP = 30  # ← 伸ばしたい場合は 50 に（任意）
    MODEL_DIR = "IMUMODEL"
    NUM_WORKERS = 4
    WARMUP_E = 3
    SCALE_ROOT = "CMI_data/data/scalers"

    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    device = DEVICE

    meta = pd.read_csv(metadata_path)
    sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)

    labels_all = meta['gesture'].values
    seqs_all   = meta['sequence_id'].tolist()
    subs_all   = meta['subject'].tolist()

    all_fold_f1 = []

    for fold, (tr_idx, va_idx) in enumerate(sgkf.split(seqs_all, labels_all, subs_all), start=1):
        print(f"\n=== Fold {fold}/{NUM_FOLDS} ===")
        tr_meta = meta.iloc[tr_idx].reset_index(drop=True)
        va_meta = meta.iloc[va_idx].reset_index(drop=True)

        scale_path = f'{SCALE_ROOT}/fold{fold}'
        tr_ds = GestureDataset(sequence_dir, label_dir, tr_meta, max_len=MAX_LEN, is_train=True,  scale_path=scale_path, use_augmentation=True)
        va_ds = GestureDataset(sequence_dir, label_dir, va_meta, max_len=MAX_LEN, is_train=False, scale_path=scale_path)

        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=True)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn,
                               num_workers=NUM_WORKERS, pin_memory=True, drop_last=False)

        n_classes = len(tr_ds.gesture_encoder.classes_)
        model = IMUOnlyHierModel(imu_dim=36, tof_dim=325, n_classes=n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)

        # ★ 追加：CosineAnnealingWarmRestarts を fold ごとに作成
        #   - T_0 は最初の周期（例: 20epoch）
        #   - もっと頻繁に再スタートさせたいなら T_0=10 も有効
        scheduler = CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=1, eta_min=LR_MIN
        )  # ★

        # prior & target mask
        train_idx_labels = tr_ds.gesture_encoder.transform(tr_meta['gesture'])
        counts = np.bincount(train_idx_labels, minlength=n_classes)
        priors = torch.tensor(counts / counts.sum(), dtype=torch.float32)
        la_ce  = LogitAdjustedCE(priors, tau=1.0)

        target_gestures = CompetitionMetric().target_gestures
        all_classes = list(tr_ds.gesture_encoder.classes_)
        target_idx = [all_classes.index(g) for g in target_gestures]
        target_mask_tensor = torch.zeros(n_classes, dtype=torch.bool)
        target_mask_tensor[target_idx] = True

        best_f1 = -1.0
        no_improve = 0

        for epoch in range(EPOCHS):
            tr_loss, tr_f1, tr_f1b, tr_f1m = train_one_epoch(
                model, tr_loader, optimizer, device, epoch, EPOCHS,
                la_ce, target_mask_tensor, warmup_epochs=WARMUP_E, eta_min=LR_MIN,
                scheduler=scheduler  # ★ 追加：scheduler を渡す
            )
            va_loss, va_f1, va_f1b, va_f1m = validate(model, va_loader, device, temps=None)

            print(f"Fold{fold} Ep{epoch+1:03d} | "
                  f"LR={optimizer.param_groups[0]['lr']:.6f} | "
                  f"Train loss={tr_loss:.4f} F1={tr_f1:.4f} | "
                  f"Val F1={va_f1:.4f}")

            if va_f1 > best_f1:
                best_f1 = va_f1
                no_improve = 0
                best_path = os.path.join(MODEL_DIR, f"model_fold{fold}_best.pth")
                torch.save(model.state_dict(), best_path)
                # fold scaler をモデル横へコピー
                shutil.copyfile(os.path.join(scale_path, 'train_scalers.pkl'),
                                os.path.join(MODEL_DIR, f"scaler_fold{fold}.pkl"))
            else:
                no_improve += 1

            # last also saved
            torch.save(model.state_dict(), os.path.join(MODEL_DIR, f"model_fold{fold}_last.pth"))

            if no_improve >= EARLY_STOP:
                print(f"Early stopping at epoch {epoch+1}")
                break

        # 温度スケーリング（ベスト重みで校正）
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f"model_fold{fold}_best.pth"),
                                         map_location=device, weights_only=False))
        model.eval()
        T_mc = calibrate_temperature(model, va_loader, device)
        with open(os.path.join(MODEL_DIR, f"temp_fold{fold}.pkl"), 'wb') as f:
            pickle.dump({'T_mc': T_mc, 'T_bin': 1.0}, f)
        print(f"[Fold {fold}] Calibrated T_mc={T_mc:.3f}")

        all_fold_f1.append(best_f1)
        # cleanup
        del model, optimizer, tr_loader, va_loader, tr_ds, va_ds, scheduler  # ★ scheduler も解放
        gc.collect(); torch.cuda.empty_cache(); time.sleep(1)

    if all_fold_f1:
        print("\n===== CV Result =====")
        for i, s in enumerate(all_fold_f1, 1):
            print(f"Fold {i}: {s:.4f}")
        print(f"Mean F1: {np.mean(all_fold_f1):.4f}")



# %% [8] Inference module (fold別scaler+温度+マルチクロップ) — 安定版
import os, glob, re, pickle, threading
import numpy as np
import polars as pl
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder

# ==== Globals ====
IMU_MODELS = []
FOLD_SCALERS = []  # list of tuples: (non_tof_scaler, tof_scaler)
FOLD_TEMPS = []    # list of dicts:  {'T_mc': float, 'T_bin': float}
GESTURE_CLASSES = None
TRAIN_DS = None    # FE-only dummy (uses _feature_engineering of your class)
INITIALIZED = False
INIT_LOCK = threading.Lock()

# モデル/スケーラ配置先（提出ノートでは Add Data の実パスに上書き）
IMU_MODELS_DIR = "/kaggle/input/imumodel"  # 例: /kaggle/input/<your-dataset-slug>
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==== Columns ====
TIME_SERIES_FEATURES = [
    'acc_x','acc_y','acc_z','rot_w','rot_x','rot_y','rot_z',
    'thm_1','thm_2','thm_3','thm_4','thm_5'
]
TOF_FEATURES = [f'tof_{sensor}_v{pixel}' for sensor in range(1,6) for pixel in range(64)]
STATIC_FEATURES = ['adult_child','age','sex','handedness','height_cm','shoulder_to_wrist_cm','elbow_to_wrist_cm']
IMU_FEATS = ['acc_x','acc_y','acc_z','rot_w','rot_x','rot_y','rot_z']
THM_FEATS = ['thm_1','thm_2','thm_3','thm_4','thm_5']

# コンペの全クラス名（フォールバック用）
COMP_ALL_CLASSES = [
    'Above ear - pull hair', 'Cheek - pinch skin', 'Eyebrow - pull hair',
    'Eyelash - pull hair', 'Forehead - pull hairline', 'Forehead - scratch',
    'Neck - pinch skin', 'Neck - scratch',
    'Write name on leg', 'Wave hello', 'Glasses on/off', 'Text on phone',
    'Write name in air', 'Feel around in tray and pull out an object',
    'Scratch knee/leg skin', 'Pull air toward your face', 'Drink from bottle/cup',
    'Pinch knee/leg skin'
]

class _IdentityScaler:
    def transform(self, X):
        return X

# ==== Interpolation (train-pipeline consistent) ====
def interpolate_thm_column(col):
    s = pd.Series(col)
    return s.interpolate('linear', limit_direction='both').fillna(0).values

def interpolate_imu_column(col):
    s = pd.Series(col)
    return s.ffill().bfill().fillna(0).values

def interpolate_tof_column(col):
    s = pd.Series(col)
    return s.ffill().bfill().fillna(0).values

def interpolate_sequence_polars(sequence, imu_features, thm_features, tof_features):
    df = sequence.to_pandas()
    for f in imu_features:
        if f in df.columns:
            df[f] = interpolate_imu_column(df[f])
    for f in thm_features:
        if f in df.columns:
            df[f] = interpolate_thm_column(df[f])
    for f in tof_features:
        if f in df.columns:
            df[f] = interpolate_tof_column(df[f])
    return pl.from_pandas(df)

# ==== Utility: 推論時のクラス数とラベル順 ====
def infer_num_classes_from_ckpt(models_dir: str) -> int:
    """checkpointの分類ヘッドの出力次元からクラス数を推定"""
    # 最初に見つかったモデルでOK
    paths = sorted(glob.glob(os.path.join(models_dir, "model_fold*_best.pth")))
    if not paths:
        raise FileNotFoundError(f"No model files found under {models_dir}")
    state = torch.load(paths[0], map_location='cpu', weights_only=False)
    # ヘッド候補のキー
    head_keys = [
        'cls_head.weight', 'classifier.weight', 'head.weight', 
        'fc.weight', 'final.weight'
    ]
    for k in head_keys:
        if k in state:
            return state[k].shape[0]
    # 次善策：weightで out_features を探す
    for k, v in state.items():
        if k.endswith('.weight') and v.ndim == 2:
            # 最も小さな out_features をクラス数候補に（ヘッドは通常最小）
            # ただし 1や2 の場合は除外
            out_features = v.shape[0]
            if 3 <= out_features <= 64:
                return out_features
    raise RuntimeError("Could not infer number of classes from checkpoint")

def ensure_label_encoder(imu_models_dir: str, n_classes: int) -> LabelEncoder:
    """LabelEncoderを用意（classes.npy優先／無ければコンペのクラス名を辞書順）"""
    le = LabelEncoder()
    classes_npy = os.path.join(imu_models_dir, "classes.npy")
    if os.path.exists(classes_npy):
        classes = np.load(classes_npy, allow_pickle=True)
        if len(classes) != n_classes:
            print(f"[warn] classes.npy has {len(classes)} classes, but checkpoint expects {n_classes}. "
                  f"Proceeding with classes.npy; verify mapping.")
        le.classes_ = np.array(list(classes))
        return le
    # フォールバック：コンペの全クラス名を辞書順に
    classes = sorted(COMP_ALL_CLASSES)
    if len(classes) != n_classes:
        raise RuntimeError(
            f"Fallback classes length {len(classes)} != inferred {n_classes}. "
            f"Please include classes.npy in your weights dataset."
        )
    le.fit(classes)
    return le

# ==== Loader (models + scalers + temps) ====
def load_models_and_scalers(models_dir, model_class, n_classes, device):
    model_paths = sorted(glob.glob(os.path.join(models_dir, "model_fold*_best.pth")))
    if not model_paths:
        raise FileNotFoundError(
            f"No model files found under {models_dir}. "
            f"Attach your weights dataset and ensure files like model_fold1_best.pth exist."
        )
    models, scalers, temps = [], [], []
    for p in model_paths:
        m = re.search(r"model_fold(\d+)_best\.pth$", os.path.basename(p))
        if not m:
            continue
        k = int(m.group(1))
        # 1) model（クラス数は推定済みの n_classes を使用）
        model = model_class(imu_dim=36, tof_dim=325, n_classes=n_classes).to(device)
        state = torch.load(p, map_location=device, weights_only=False)
        model.load_state_dict(state)
        model.eval()
        models.append(model)
        # 2) scaler
        sc_path = os.path.join(models_dir, f"scaler_fold{k}.pkl")
        if not os.path.exists(sc_path):
            raise FileNotFoundError(f"Missing scaler for fold{k}: {sc_path}")
        # load_models_and_scalers(...):
        with open(sc_path, "rb") as f:
            sc_obj = pickle.load(f)
            if isinstance(sc_obj, tuple) and len(sc_obj) == 3:
                scalers.append(sc_obj)  # (non_tof, tof, demo)
            else:
                # 後方互換（旧: (non_tof, tof)）— demo は non_tof と同じ基準に
                non_tof, tof = sc_obj
                from sklearn.preprocessing import StandardScaler
                dummy = StandardScaler()
                dummy.mean_ = np.zeros(7); dummy.scale_ = np.ones(7); dummy.n_features_in_ = 7
                scalers.append((non_tof, tof, dummy))

        # 3) temperature (optional)
        tp_path = os.path.join(models_dir, f"temp_fold{k}.pkl")
        if os.path.exists(tp_path):
            with open(tp_path, "rb") as f:
                temps.append(pickle.load(f))
        else:
            temps.append({'T_mc': 1.0, 'T_bin': 1.0})
    return models, scalers, temps

# ==== Multi-crop fallback ====
if 'make_crops' not in globals():
    def make_crops(x, L=256, K=5):
        """
        x: (T, C)
        均等にK箇所の開始位置をサンプリングし、長さLで切り出し（不足はゼロパディング）。
        """
        T, C = x.shape
        crops = []
        if T <= L:
            pad = np.zeros((L - T, C), dtype=x.dtype)
            crops = [np.vstack([x, pad])]
        else:
            starts = np.linspace(0, T - L, num=K, dtype=int)
            for s in starts:
                crops.append(x[s:s+L])
        return crops

# ==== Lightweight init ====
def init_infer_module(
    imu_models_dir=IMU_MODELS_DIR,
    metadata_path='/kaggle/input/cmi-detect-behavior-with-sensor-data/train_labels.csv',
    model_class=None  # 例: IMUOnlyHierModel を渡す
):
    """
    - 学習データのロード/fitは一切行わない
    - GestureDataset.__init__を呼ばず、FE用のダミーを作って _feature_engineering を流用
    - クラス数は checkpoint から推定。classes.npy があれば順序も再現。
    """
    from __main__ import GestureDataset  # あなたの定義を参照
    global IMU_MODELS, FOLD_SCALERS, FOLD_TEMPS, GESTURE_CLASSES, TRAIN_DS, INITIALIZED

    if INITIALIZED:
        return
    with INIT_LOCK:
        if INITIALIZED:
            return
        print("Initializing inference module...")

        # 0) 推論クラス数を checkpoint から推定
        n_classes = infer_num_classes_from_ckpt(imu_models_dir)

        # 1) FE専用の軽量インスタンス（__init__は呼ばない）
        TRAIN_DS = GestureDataset.__new__(GestureDataset)
        TRAIN_DS.tof_dim = 320  # 定数
        TRAIN_DS.gesture_encoder = ensure_label_encoder(imu_models_dir, n_classes)
        GESTURE_CLASSES = n_classes

        # 2) モデルクラスの選択（Hier→普通の順で探す）
        if model_class is None:
            try:
                from __main__ import IMUOnlyHierModel as DefaultModelClass
            except Exception:
                from __main__ import IMUOnlyModel as DefaultModelClass
            model_class = DefaultModelClass

        # 3) モデル/スケーラ/温度をロード
        models, scalers, temps = load_models_and_scalers(
            imu_models_dir, model_class, n_classes=GESTURE_CLASSES, device=DEVICE
        )
        IMU_MODELS[:] = models
        FOLD_SCALERS[:] = scalers
        FOLD_TEMPS[:] = temps

        print(f"Inference init: models={len(IMU_MODELS)} · classes={GESTURE_CLASSES}")
        INITIALIZED = True

# ==== Predict ====
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    try:
        init_infer_module()  # 必要なら model_class=IMUOnlyHierModel を明示

        # 0) 必須列チェック（不足なら即エラーで場所特定）
        required = IMU_FEATS + THM_FEATS + TOF_FEATURES
        miss = [c for c in required if c not in sequence.columns]
        if miss:
            raise RuntimeError(f"Missing columns in sequence: {miss[:10]} ... (total {len(miss)})")
        miss_d = [c for c in STATIC_FEATURES if c not in demographics.columns]
        if miss_d:
            raise RuntimeError(f"Missing columns in demographics: {miss_d}")

        # 1) 補間
        sequence = interpolate_sequence_polars(sequence, IMU_FEATS, THM_FEATS, TOF_FEATURES)
        demographics = demographics.fill_null(0)
        sequence = sequence.select(IMU_FEATS + THM_FEATS + TOF_FEATURES)
        demographics = demographics.select(STATIC_FEATURES)

        # 2) 特徴生成（1回のみ）
        seq_np = sequence.to_numpy().astype(np.float32)
        demo_np = demographics.to_numpy().astype(np.float32)[0]
        full_seq = np.concatenate([seq_np, np.tile(demo_np, (len(seq_np), 1))], axis=1)

        # 学習時と同じ関数を使う（GestureDataset のメソッド）
        base_features = TRAIN_DS._feature_engineering(full_seq)

        # 3) foldごとに正規化→マルチクロップ→温度→確率算出（ロジット平均版）
                # 3) foldごとに正規化→マルチクロップ→温度→ロジット算出（logits平均）
        all_fold_logits = []

        with torch.no_grad():
            for (model, scaler_pack, temps) in zip(IMU_MODELS, FOLD_SCALERS, FOLD_TEMPS):
                # --- スケーラの取り出し（3タプル/2タプル両対応）
                if isinstance(scaler_pack, tuple):
                    if len(scaler_pack) == 3:
                        non_tof_scaler, tof_scaler, demo_scaler = scaler_pack
                    elif len(scaler_pack) == 2:
                        non_tof_scaler, tof_scaler = scaler_pack
                        demo_scaler = _IdentityScaler()  # 旧フォーマット互換
                    else:
                        raise RuntimeError(f"Unexpected scaler tuple length: {len(scaler_pack)}")
                else:
                    raise RuntimeError("Scaler pack must be a tuple.")

                # --- base_features: [IMU(36), THM(5), TOF(320), DEMO(7)]
                D = base_features.shape[1]
                tof  = base_features[:, D - TRAIN_DS.tof_dim - 7 : D - 7]   # 320
                demo = base_features[:, -7:]                                 # 7
                imu_thm = base_features[:, : D - (TRAIN_DS.tof_dim + 7)]     # 36+5

                # --- スケール適用
                n_imu_thm = non_tof_scaler.transform(imu_thm)
                n_demo    = demo_scaler.transform(demo)
                n_tof     = tof_scaler.transform(tof)

                # --- 学習と同じ最終並びに統一: [IMU+THM][DEMO][TOF]
                norm_features = np.concatenate([n_imu_thm, n_demo, n_tof], axis=1)

                # --- モデル入力の分割（学習と同じロジック）
                tof_features   = norm_features[:, -TRAIN_DS.tof_dim:]        # 末尾320
                imu_thm_demo   = norm_features[:, :-TRAIN_DS.tof_dim]        # 残り
                demo_vec       = imu_thm_demo[0, -7:]                        # DEMO 7
                thm            = imu_thm_demo[:, :-7][:, -5:]                # THM 5
                imu            = imu_thm_demo[:, :-7][:, :-5]                # IMU 36
                thm_tof        = np.concatenate([thm, tof_features], axis=1) # 325

                # --- マルチクロップ
                imu_cs = make_crops(imu, L=256, K=5)
                thm_cs = make_crops(thm_tof, L=256, K=5)

                # --- cropごとに TTA ロジットを平均
                crop_logits = []
                for ic, tc in zip(imu_cs, thm_cs):
                    tta_logits = []
                    for (ic2, tc2) in tta_variants_per_crop(ic, tc):
                        it = torch.tensor(ic2, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        tt = torch.tensor(tc2, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        dt = torch.tensor(demo_vec, dtype=torch.float32).unsqueeze(0).to(DEVICE)
                        out = model(it, tt, dt)
                        logits = out[0] if isinstance(out, tuple) else out
                        tta_logits.append(logits.cpu())
                    crop_logits.append(torch.stack(tta_logits).mean(0))  # (1,C)

                # --- crop 平均 → 温度
                fold_logits = torch.stack(crop_logits).mean(0)  # (1,C)
                Tmc = float(temps.get('T_mc', 1.0))
                fold_logits = fold_logits / max(Tmc, 1e-3)

                all_fold_logits.append(fold_logits)

        # --- 全foldのログイット平均 → softmax 1回
        logits_total = torch.stack(all_fold_logits).mean(0)  # (1, C)
        avg_prob = F.softmax(logits_total, dim=1)

        
        pred_idx = int(avg_prob.argmax(dim=1).item())
        pred_label = TRAIN_DS.gesture_encoder.inverse_transform([pred_idx])[0]
        return str(pred_label)


    except Exception as e:
        # Gateway でも見えるように stderr へ詳細を出す
        import sys, traceback
        print("PREDICT_ERROR:", e, file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        raise



# %% [9] Kaggle Inference Server
import kaggle_evaluation.cmi_inference_server

inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    # ローカルゲートウェイ（デバッグ用）
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )




