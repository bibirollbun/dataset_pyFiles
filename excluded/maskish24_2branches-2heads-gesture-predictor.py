import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from scipy.optimize import minimize_scalar
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedGroupKFold, GroupKFold
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import f1_score, confusion_matrix
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import optuna
from optuna.samplers import TPESampler
import json
import random

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
acc_cols = ['acc_x', 'acc_y', 'acc_z']
lin_acc_cols = ['lin_acc_x', 'lin_acc_y', 'lin_acc_z']
rot_cols = ['rot_w', 'rot_x', 'rot_y', 'rot_z']
ang_vel_cols = ['ang_vel_x', 'ang_vel_y', 'ang_vel_z']
norm_cols = ['acc_norm', 'lin_acc_norm', 'ang_vel_norm']
delta_cols = ['acc_delta', 'ang_vel_delta', 'lin_acc_delta', 'rot_delta']
additional_cols = ['angular_distance', 'acc_mag', 'acc_mag_jerk', 'rot_angle', 'rot_angle_vel']
thm_cols = [f'thm_{i}' for i in range(1, 6)]
tof_pixel_cols = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
modes = [4, 8, 16]
tof_region_cols = [
    f"tof{mode}_{i}_region_{r}_{stat}"
    for mode in modes
    for i in range(1, 6)
    for r in range(mode)
    for stat in ["mean", "std", "min", "max"]
]

tof_cols = tof_pixel_cols + tof_region_cols
imu_cols =  acc_cols + lin_acc_cols + rot_cols + ang_vel_cols + norm_cols + delta_cols + additional_cols
input_features = imu_cols + thm_cols + tof_cols


def remove_gravity_from_acc(acc_values, rot_values):
    num_samples = acc_values.shape[0]
    linear_accel = np.zeros_like(acc_values)
    gravity_world = np.array([0, 0, 9.81])

    for i in range(num_samples):
        if np.all(np.isnan(rot_values[i])) or np.all(np.isclose(rot_values[i], 0)):
            linear_accel[i, :] = acc_values[i, :]
            continue
        rotation = R.from_quat(rot_values[i])
        gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
        linear_accel[i, :] = acc_values[i, :] - gravity_sensor_frame
    return linear_accel

def calculate_angular_velocity_from_quat(rot_values, time_delta=1/200): # Assuming 200Hz sampling rate
    num_samples = rot_values.shape[0]
    angular_vel = np.zeros((num_samples, 3))

    for i in range(num_samples - 1):
        q_t = rot_values[i]
        q_t_plus_dt = rot_values[i+1]

        if np.all(np.isnan(q_t)) or np.all(np.isclose(q_t, 0)) or \
           np.all(np.isnan(q_t_plus_dt)) or np.all(np.isclose(q_t_plus_dt, 0)):
            continue
        rot_t = R.from_quat(q_t)
        rot_t_plus_dt = R.from_quat(q_t_plus_dt)

        delta_rot = rot_t.inv() * rot_t_plus_dt

        angular_vel[i, :] = delta_rot.as_rotvec() / time_delta

    return angular_vel

def calculate_angular_distance(rot_values):
    num_samples = rot_values.shape[0]
    angular_dist = np.zeros(num_samples)

    for i in range(num_samples - 1):
        q1 = rot_values[i]
        q2 = rot_values[i+1]

        if np.all(np.isnan(q1)) or np.all(np.isclose(q1, 0)) or \
           np.all(np.isnan(q2)) or np.all(np.isclose(q2, 0)):
            angular_dist[i] = 0
            continue
        r1 = R.from_quat(q1)
        r2 = R.from_quat(q2)
        relative_rotation = r1.inv() * r2
        angle = np.linalg.norm(relative_rotation.as_rotvec())
        angular_dist[i] = angle
    return angular_dist


def compute_multiscale_tof_stats(df, modes):
    new_cols = {}
    for i in range(1, 6):  # tof_1 to tof_5
        base_cols = [f"tof_{i}_v{j}" for j in range(64)]
        tof_data = df[base_cols].replace(-1, np.nan).copy()

        for mode in modes:
            region_size = 64 // mode
            for r in range(mode):
                region_cols = tof_data.iloc[:, r*region_size:(r+1)*region_size]
                new_cols[f"tof{mode}_{i}_region_{r}_mean"] = region_cols.mean(axis=1)
                new_cols[f"tof{mode}_{i}_region_{r}_std"]  = region_cols.std(axis=1)
                new_cols[f"tof{mode}_{i}_region_{r}_min"]  = region_cols.min(axis=1)
                new_cols[f"tof{mode}_{i}_region_{r}_max"]  = region_cols.max(axis=1)

    return pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

def add_new_features(df):
    df[lin_acc_cols] = remove_gravity_from_acc(df[acc_cols].values, df[rot_cols].values)
    df[ang_vel_cols] = calculate_angular_velocity_from_quat(df[rot_cols].values)
    all_ang_dists = []
    for _, g in df.groupby('sequence_id', sort=False):
        ang_dist = calculate_angular_distance(g[rot_cols].values)
        all_ang_dists.extend(ang_dist)
    df['angular_distance'] = all_ang_dists
    df["acc_norm"] = np.linalg.norm(df[acc_cols].values, axis=1)
    df['lin_acc_norm'] = np.linalg.norm(df[lin_acc_cols], axis=1)
    df['ang_vel_norm'] = np.linalg.norm(df[ang_vel_cols], axis=1)
    df["acc_delta"]     = df["acc_norm"].diff().fillna(0)
    df["ang_vel_delta"] = df["ang_vel_norm"].diff().fillna(0)
    df["lin_acc_delta"] = df["lin_acc_norm"].diff().fillna(0)
    df["rot_delta"]     = df[rot_cols].diff().pow(2).sum(axis=1).pow(0.5).fillna(0)
    df['acc_mag'] = np.linalg.norm(df[acc_cols].values, axis=1)
    df['rot_angle'] = 2 * np.arccos(np.clip(df['rot_w'], -1, 1))
    df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    df = compute_multiscale_tof_stats(df, modes)
    return df

def build_tensor(df, feature_cols, sequence_col="sequence_id", cap_len=None, mode="train", rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    seq_tensors, seq_ids = [], []
    for sid, g in df.groupby(sequence_col, sort=False):  # keep original order
        X = g[feature_cols].to_numpy(dtype=np.float32)
        if cap_len is not None and len(X) > cap_len:
            if mode == "train":
                start = rng.integers(0, len(X) - cap_len + 1)
                X = X[start:start+cap_len]
            else:  # eval: deterministic
                X = X[:cap_len]
        seq_tensors.append(torch.from_numpy(X))
        seq_ids.append(sid)
    padded = torch.nn.utils.rnn.pad_sequence(seq_tensors, batch_first=True)  # [N, T, D]
    return padded, np.array(seq_ids)


def group_normalize(df, feature_cols, group_col='sequence_id'):
    return df.groupby(group_col)[feature_cols].transform(
        lambda x: (x - x.mean()) / (x.std() + 1e-6)
    )

def make_train_data():
    train = pd.read_csv('cmi-detect-behavior-with-sensor-data/train.csv')
    train = add_new_features(train)
    train[input_features] = train[input_features].interpolate(method="linear", limit_direction="both", axis=0)
    train[input_features] = train[input_features].fillna(0.0)

    scaler = StandardScaler()
    train[input_features] = scaler.fit_transform(train[input_features])
    os.makedirs("cmi-checkpoint", exist_ok=True)

    joblib.dump(scaler, "cmi-checkpoint/scaler.pkl")
    train_group = train.copy()
    train_group[input_features] = group_normalize(train_group, input_features)
    train_global_tensor, train_seq_order  = build_tensor(train, input_features, cap_len=120, mode='train')
    train_group_tensor,  train_seq_order2 = build_tensor(train_group, input_features, cap_len=120, mode='train')
    X_train = torch.cat([train_global_tensor, train_group_tensor], dim=2)  # [N, T, 2*D]

    seq_label_series = (
    train.groupby('sequence_id', sort=False)['gesture']
        .apply(lambda s: s.iloc[0])
    )
    seq_labels_ordered = seq_label_series.reindex(train_seq_order).values
    target_labels = sorted(train.loc[train['sequence_type'] == 'Target', 'gesture'].unique())
    seq_labels_ordered = np.where(
        np.isin(seq_labels_ordered, target_labels),
        seq_labels_ordered,
        -1
    )

    le = LabelEncoder()
    le.fit(target_labels)
    joblib.dump(le, "cmi-checkpoint/label_encoder.pkl")
    mask = seq_labels_ordered != -1
    y_train = np.full(len(seq_labels_ordered), -1)  # Initialize all as -1
    y_train[mask] = le.transform(seq_labels_ordered[mask])
    y_train = torch.tensor(y_train).long()
    groups = train_seq_order
    
    return X_train, y_train, groups, le

class ResidualSEBlock(nn.Module):
    def __init__(self, Cin, Cout, kernel_size, dropout, activation, se_reduction, drop_prob):
        super().__init__()
        self.conv1 = nn.Conv1d(Cin, Cout, kernel_size, padding=kernel_size // 2)
        self.norm1 = nn.LayerNorm(Cout)
        self.conv2 = nn.Conv1d(Cout, Cout, kernel_size, padding=kernel_size // 2)
        self.norm2 = nn.LayerNorm(Cout)
        self.activation = activation
        self.skip = nn.Conv1d(Cin, Cout, 1) if Cin != Cout else nn.Identity()
        self.se_fc = nn.Sequential(
            nn.Linear(Cout, Cout // se_reduction, bias=False),
            activation,
            nn.Linear(Cout // se_reduction, Cout, bias=False),
            nn.Sigmoid()
        )
        self.se_pool = nn.AdaptiveAvgPool1d(1)

        self.drop_prob = drop_prob

    def stochastic_depth(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        mask = torch.rand(shape, dtype=x.dtype, device=x.device) < keep_prob
        return x * mask / keep_prob

    def se_block(self, x):
        b, c, _ = x.size()
        y = self.se_pool(x).view(b, c)
        y = self.se_fc(y).view(b, c, 1)
        return x * y

    def forward(self, x):  # [B, C, T]
        residual = self.skip(x)
        out = self.conv1(x)
        out = self.norm1(out.transpose(1, 2)).transpose(1, 2)  # LayerNorm over channels
        out = self.activation(out)

        out = self.conv2(out)
        out = self.norm2(out.transpose(1, 2)).transpose(1, 2)

        out = self.se_block(out)

        out = self.stochastic_depth(out)

        return self.activation(out + residual)


class NoiseDenseBranch(nn.Module):
    def __init__(self, input_dim, cfg):
        super().__init__()
        hidden_dim = cfg.get("hidden_dim", 8)
        dropout = cfg.get("dropout", 0.1)
        self.noise_std = cfg.get("noise_std", 0.09)

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):  # x: [B, T, D]
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(x) * self.noise_std
            x = x + noise
        return self.block(x)



class MultiQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim // num_heads)  # shared KV
        self.v_proj = nn.Linear(embed_dim, embed_dim // num_heads)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):  # x: [B, T, D]
        B, T, D = x.size()
        H = self.num_heads
        head_dim = D // H

        q = self.q_proj(x).view(B, T, H, head_dim).transpose(1, 2)  # [B, H, T, d]
        k = self.k_proj(x).mean(dim=1, keepdim=True)                # [B, 1, d]
        v = self.v_proj(x).mean(dim=1, keepdim=True)                # [B, 1, d]

        k = k.unsqueeze(1).expand(-1, H, -1, -1)  # [B, H, 1, d]
        v = v.unsqueeze(1).expand(-1, H, -1, -1)  # [B, H, 1, d]

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (head_dim ** 0.5)  # [B, H, T, 1]
        attn_weights = torch.softmax(attn_scores, dim=-2)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, v)  # [B, H, T, d]
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)

        return self.out_proj(attn_output)  # [B, T, D]


class MultiQueryTransformerBlock(nn.Module):
    def __init__(self, input_dim, cfg, activation):
        super().__init__()
        num_heads = cfg.get("num_heads", 4)
        embed_dim = cfg.get("embed_dim", 512)
        attn_dropout = cfg.get("attn_dropout", 0.1)
        trans_dropout = cfg.get("trans_dropout", 0.2)

        self.norm0 = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, embed_dim) if input_dim != embed_dim else nn.Identity()
        self.attn = MultiQueryAttention(embed_dim, num_heads, dropout=attn_dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.dropout1 = nn.Dropout(trans_dropout)

        ff_hidden_dim = cfg.get("ff_hidden_dim", input_dim * 2)
        ff_num_layers = cfg.get("ff_num_layers", 1)
        ff_dropout = cfg.get("ff_dropout", 0.1)


        def make_ff(input_dim, num_layers, hidden_dim, dropout, activation):
            layers = []
            in_dim = input_dim
            for _ in range(num_layers):
                layers += [
                    nn.Linear(in_dim, hidden_dim),
                    activation,
                    nn.Dropout(dropout)
                ]
                in_dim = hidden_dim
            layers.append(nn.Linear(hidden_dim, input_dim))
            return nn.Sequential(*layers)

        self.ffn = make_ff(embed_dim, ff_num_layers, ff_hidden_dim, ff_dropout, activation)

        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout2 = nn.Dropout(trans_dropout)

        self.reverse_mask_drop_prob = cfg.get("reverse_mask_drop_prob", 0.15)

        self.attn_score = nn.Linear(embed_dim, 1)
        self.gate = nn.Parameter(torch.zeros(embed_dim))  # initialized equally

    def reverse_mask_attention(self, x, drop_prob=0.0):
        if not self.training or drop_prob == 0.0:
            return x
        B, T, D = x.shape
        mask = torch.rand(B, T, device=x.device) > drop_prob
        return x * mask.unsqueeze(-1)

    def forward(self, x):  # x: [B, T, D]
        x = self.input_proj(self.norm0(x))
        x = x + self.dropout1(self.attn(self.norm1(x)))
        x = x + self.dropout2(self.ffn(self.norm2(x)))

        masked_out = self.reverse_mask_attention(x, drop_prob=self.reverse_mask_drop_prob)
        attn_weights = torch.softmax(self.attn_score(masked_out), dim=1)  # [B, T, 1]
        weighted = torch.sum(x * attn_weights, dim=1)  # [B, D]
        residual = self.norm2(x).mean(dim=1)  # [B, D]
        pooled = torch.sigmoid(self.gate) * weighted + (1 - torch.sigmoid(self.gate)) * residual

        return pooled  # [B, D]


class CNNTwoBranchModel(nn.Module):
    def __init__(self, imu_dim, tof_thm_dim, imu_cnn_cfg, tof_thm_cnn_cfg, noise_dense_branch_cfg,
                 lstm_cfg, gru_cfg, trans_cfg, gesture_head_cfg,
                 binary_head_cfg, activation_cfg, num_classes, training=True):
        super().__init__()
        self.imu_dim = imu_dim
        self.tof_thm_dim = tof_thm_dim
        self.training = training

        def get_activation(name):
            if name is None:
                return nn.SiLU()
            return {
                "silu": nn.SiLU(),
                "gelu": nn.GELU(),
                "relu": nn.ReLU()
            }[name]

        def make_cnn(Cin, cfg, activation):
            hidden_dim = cfg.get("hidden_dim", 512)
            kernel_size = cfg.get("kernel_size", 5)
            num_layers = cfg.get("num_layers", 2)
            se_reduction = cfg.get("se_reduction", 8)
            drop_prob = cfg.get("drop_prob", 0.0)
            base_dropout = 0.1

            layers = []
            for i in range(num_layers):
                in_dim = Cin if i == 0 else hidden_dim
                dropout = base_dropout if i == num_layers - 1 else 0.0
                layers.append(
                    ResidualSEBlock(
                        in_dim, hidden_dim, kernel_size,
                        dropout, activation, se_reduction, drop_prob
                    )
                )

            return nn.Sequential(*layers)

        cnn_activation = get_activation(activation_cfg["cnn"])
        self.imu_cnn = make_cnn(imu_dim, imu_cnn_cfg, cnn_activation)
        self.tof_cnn = make_cnn(tof_thm_dim, tof_thm_cnn_cfg, cnn_activation)

        concat_cnn_out_dim = imu_cnn_cfg["hidden_dim"] * 2 + tof_thm_cnn_cfg["hidden_dim"] * 2

        self.noise_dense_branch = NoiseDenseBranch(concat_cnn_out_dim, noise_dense_branch_cfg)

        rnn_input_dim = concat_cnn_out_dim + noise_dense_branch_cfg["hidden_dim"]

        self.bilstm = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=lstm_cfg["hidden_dim"],
            num_layers=lstm_cfg["num_layers"],
            dropout=lstm_cfg["dropout"] if lstm_cfg["num_layers"] > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )

        self.bigru = nn.GRU(
            input_size=rnn_input_dim,
            hidden_size=gru_cfg["hidden_dim"],
            num_layers=gru_cfg["num_layers"],
            dropout=gru_cfg["dropout"] if gru_cfg["num_layers"] > 1 else 0.0,
            bidirectional=True,
            batch_first=True
        )

        concat_rnn_dense_dim = (
            lstm_cfg["hidden_dim"] * 2 +
            gru_cfg["hidden_dim"] * 2
        )
        ffn_activation = get_activation(activation_cfg["ffn"])
        self.trans = MultiQueryTransformerBlock(concat_rnn_dense_dim, trans_cfg, ffn_activation)

        def make_head(input_dim, head_cfg, num_classes, activation):
            layers = []
            hidden_dim = head_cfg.get("hidden_dim", 128)
            num_layers = head_cfg.get("num_layers", 1)
            dropout = head_cfg.get("dropout", 0.1)
            for _ in range(num_layers - 1):
                layers += [
                    nn.Linear(input_dim, hidden_dim),
                    activation,
                    nn.Dropout(dropout)
                ]
                input_dim = hidden_dim
            layers.append(nn.Linear(input_dim, num_classes))
            return nn.Sequential(*layers)

        head_activation = get_activation(activation_cfg["head"])
        head_input_dim = trans_cfg['embed_dim']
        self.gesture_head = make_head(head_input_dim , gesture_head_cfg, num_classes, head_activation)
        self.binary_head = make_head(head_input_dim , binary_head_cfg, 1, head_activation)

    def forward(self, x):  # x: [B, T, 2*(imu_dim + tof_dim)]
        D_imu, D_tof = self.imu_dim, self.tof_thm_dim
        total_dim = 2 * (D_imu + D_tof)

        x_imu_global = x[..., :D_imu]                            # [B, T, D_imu]
        x_tof_global = x[..., D_imu:D_imu+D_tof]                 # [B, T, D_tof]
        x_imu_group  = x[..., D_imu+D_tof:D_imu*2+D_tof]         # [B, T, D_imu]
        x_tof_group  = x[..., D_imu*2+D_tof:]                    # [B, T, D_tof]

        imu_global_out = self.imu_cnn(x_imu_global.permute(0, 2, 1))  # [B, H, T]
        imu_group_out  = self.imu_cnn(x_imu_group.permute(0, 2, 1))   # same weights

        tof_global_out = self.tof_cnn(x_tof_global.permute(0, 2, 1))  # [B, H, T]
        tof_group_out  = self.tof_cnn(x_tof_group.permute(0, 2, 1))   # same weights

        x_cat_cnn = torch.cat([imu_global_out, imu_group_out, tof_global_out, tof_group_out], dim=1)             # [B, 4H, T]
        x_seq_cnn = x_cat_cnn.permute(0, 2, 1)                           # [B, T, 4H]

        x_noise   = self.noise_dense_branch(x_seq_cnn)  # [B, T, 16]
        x_seq_cnn = torch.cat([x_seq_cnn, x_noise], dim=2)
        x_lstm, _ = self.bilstm(x_seq_cnn)
        x_gru, _  = self.bigru(x_seq_cnn)
        x_cat_rnn = torch.cat([x_lstm, x_gru], dim=2)

        x_trans = self.trans(x_cat_rnn)       # [B, total_dim]

        gesture_logits = self.gesture_head(x_trans)  # [B, num_classes]
        binary_logits  = self.binary_head(x_trans)   # [B, 1]

        return gesture_logits, binary_logits


class EarlyStopper():
    def __init__(self, patience=5, min_delta=1e-4, alpha=0.6):
        self.patience = patience
        self.min_delta = min_delta
        self.alpha = alpha
        self.best = -float("inf")
        self.wait = 0
        self.best_state = None
        self.ema_score = None

    def step(self, score, model):
        self.ema_score = score if self.ema_score is None else self.alpha * score + (1 - self.alpha) * self.ema_score
        if self.ema_score > self.best + self.min_delta:
            self.best = self.ema_score
            self.wait = 0
            self.best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            return False
        self.wait += 1
        return self.wait >= self.patience


class SpecAugment(nn.Module):
    def __init__(self, max_time_width=10, max_feature_width=5, prob=0.3):
        super().__init__()
        self.max_time_width = max_time_width
        self.max_feature_width = max_feature_width
        self.prob = prob

    def forward(self, x):
        if not self.training or np.random.rand() > self.prob:
            return x

        B, T, D = x.shape
        x_aug = x.clone()

        for b in range(B):
            if self.max_time_width > 0:
                t0 = np.random.randint(0, T)
                t_width = np.random.randint(1, self.max_time_width + 1)
                t1 = min(T, t0 + t_width)
                x_aug[b, t0:t1, :] = 0.0

            if self.max_feature_width > 0:
                f0 = np.random.randint(0, D)
                f_width = np.random.randint(1, self.max_feature_width + 1)
                f1 = min(D, f0 + f_width)
                x_aug[b, :, f0:f1] = 0.0

        return x_aug

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, input, target):
        logp = F.log_softmax(input, dim=1)
        p = logp.exp()
        logp = logp.gather(1, target.unsqueeze(1)).squeeze(1)
        p = p.gather(1, target.unsqueeze(1)).squeeze(1)

        loss = -(1 - p) ** self.gamma * logp
        if self.weight is not None:
            w = self.weight[target]
            loss = loss * w

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def create_loader(X, y, batch_size, shuffle=True):
    ds = TensorDataset(X, y)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def mixup_data(x, y, alpha=0.4):
    if alpha > 0.0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def competition_f1(y_true, y_pred):
    y_true_bin = np.array([1 if y != -1 else 0 for y in y_true])
    y_pred_bin = np.array([1 if y != -1 else 0 for y in y_pred])
    binary_f1 = f1_score(y_true_bin, y_pred_bin)

    all_labels = np.unique(np.concatenate([y_true, y_pred]))
    macro_f1 = f1_score(y_true, y_pred, labels=all_labels, average="macro")

    return 0.5 * binary_f1 + 0.5 * macro_f1, binary_f1, macro_f1

def stratified_split_with_nontarget(
    X, y, groups, n_splits=5, random_state=42
):
    mask_target = y != -1
    mask_nontarget = ~mask_target

    target_indices = np.arange(len(X))[mask_target]
    nontarget_indices = np.arange(len(X))[mask_nontarget]

    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    gkf  = GroupKFold(n_splits=n_splits)
    target_folds = list(sgkf.split(target_indices, y[mask_target].numpy(), groups[mask_target]))
    nontarget_folds = list(gkf.split(nontarget_indices, np.zeros(len(nontarget_indices)), groups[mask_nontarget]))

    for fold in range(n_splits):
        target_train_idx, target_val_idx = target_folds[fold]
        nt_train_idx, nt_val_idx = nontarget_folds[fold]
        target_train_idx = target_indices[target_train_idx]
        target_val_idx   = target_indices[target_val_idx]
        nt_train_idx     = nontarget_indices[nt_train_idx]
        nt_val_idx       = nontarget_indices[nt_val_idx]
        train_idx = np.concatenate([target_train_idx, nt_train_idx])
        val_idx   = np.concatenate([target_val_idx, nt_val_idx])

        yield train_idx, val_idx

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    mask_a = y_a != -1
    mask_b = y_b != -1
    mask = mask_a & mask_b
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred.device)
    return lam * criterion(pred[mask], y_a[mask]) + (1 - lam) * criterion(pred[mask], y_b[mask])

def dual_margin_loss(logits, targets, pos_margin=0.3, neg_margin=0.1):
    B, C = logits.shape
    true_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]

    one_hot = F.one_hot(targets, num_classes=C).bool()
    logits_masked = logits.masked_fill(one_hot, float('-inf'))

    max_other_logits = logits_masked.max(dim=1).values
    inter_margin_loss = F.relu(pos_margin - (true_logits - max_other_logits))

    mean_other_logits = logits_masked.masked_fill(torch.isnan(logits_masked), 0).mean(dim=1)
    intra_margin_loss = F.relu(neg_margin - (true_logits - mean_other_logits))

    return inter_margin_loss.mean() + 0.05 * intra_margin_loss.mean()

def optimize_binary_threshold(binary_probs, gesture_probs, labels):
    thresholds = np.linspace(0.05, 0.95, 50)
    best_score, best_thresh = -1, 0.5
    for th in thresholds:
        preds = np.full(len(labels), -1)
        mask = binary_probs >= th
        preds[mask] = np.argmax(gesture_probs[mask], axis=1)
        score, _, _ = competition_f1(labels, preds)
        if score > best_score:
            best_score, best_thresh = score, th
    return best_thresh


def create_model():
    X_train, y_train, groups, le = make_train_data()
    num_classes = len(le.classes_)
    rng = np.random.default_rng(seed=42)   
    unique_groups = np.unique(groups)
    tune_groups = rng.choice(unique_groups, size=int(len(unique_groups)*0.3), replace=False)
    tune_mask = np.isin(groups, tune_groups)
    X_tune, y_tune, groups_tune = X_train[tune_mask], y_train[tune_mask], groups[tune_mask]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sampler = TPESampler(seed=42, multivariate=True, warn_independent_sampling=False)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=0)
    study_name = "2branch_2head_study"
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///optuna_{study_name}.db",
        direction="minimize",
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner
    )
    study.optimize(objective, n_trials=100, n_jobs=1)
    print("Best params:", study.best_params)
    print("Best CV correlation:", study.best_value)
    best_trial = study.best_trial
    best_params = {
        "imu_cnn_cfg": {
            "hidden_dim": best_trial.params["imu_cnn_hidden_dim"],
            "kernel_size": best_trial.params["imu_cnn_kernel_size"],
            "num_layers": best_trial.params["imu_cnn_num_layers"],
            "se_reduction": best_trial.params["imu_cnn_se_reduction"],
            "drop_prob": best_trial.params["imu_cnn_drop_prob"]
        },
        "tof_thm_cnn_cfg": {
            "hidden_dim": best_trial.params["tof_thm_cnn_hidden_dim"],
            "kernel_size": best_trial.params["tof_thm_cnn_kernel_size"],
            "num_layers": best_trial.params["tof_thm_cnn_num_layers"],
            "se_reduction": best_trial.params["tof_thm_se_reduction"],
            "drop_prob": best_trial.params["tof_thm_cnn_drop_prob"]
        },
        "noise_dense_branch_cfg": {
            "hidden_dim": best_trial.params["noise_dense_hidden_dim"],
            "dropout": best_trial.params["noise_dense_dropout"],
            "noise_std": best_trial.params["noise_dense_noise_std"]
        },
        "lstm_cfg": {
            "hidden_dim": best_trial.params["lstm_hidden_dim"],
            "num_layers": best_trial.params["lstm_num_layers"],
            "dropout": best_trial.params["lstm_dropout"]
        },
        "gru_cfg": {
            "hidden_dim": best_trial.params["gru_hidden_dim"],
            "num_layers": best_trial.params["gru_num_layers"],
            "dropout": best_trial.params["gru_dropout"]
        },
        "trans_cfg": {
            "num_heads": best_trial.params["attn_num_heads"],
            "embed_dim": best_trial.params["attn_embed_dim"],
            "attn_dropout": best_trial.params["attn_dropout"],
            "trans_dropout": best_trial.params["trans_dropout"],
            "ff_hidden_dim": best_trial.params["ff_hidden_dim"],
            "ff_num_layers": best_trial.params["ff_num_layers"],
            "ff_dropout": best_trial.params["ff_dropout"],
            "reverse_mask_drop_prob": best_trial.params["attn_reverse_mask_drop_prob"]
        },
        "gesture_head_cfg": {
            "hidden_dim": best_trial.params["gesture_hidden_dim"],
            "num_layers": best_trial.params["gesture_num_layers"],
            "dropout": best_trial.params["gesture_dropout"]
        },
        "binary_head_cfg": {
            "hidden_dim": best_trial.params["binary_hidden_dim"],
            "num_layers": best_trial.params["binary_num_layers"],
            "dropout": best_trial.params["binary_dropout"]
        },
        "activation_cfg": {
            "cnn": "silu",
            "ffn": "relu",
            "head": "gelu"
        },
        "specaug_cfg": {
            "max_time_width": best_trial.params["specaug_time_width"],
            "max_feature_width": best_trial.params["specaug_feature_width"],
            "prob": best_trial.params["specaug_prob"]
        },
        "batch_size": best_trial.params["batch_size"],
        "lr": best_trial.params["lr"],
        "weight_decay": best_trial.params["weight_decay"],
        "focal_gamma": best_trial.params["focal_gamma"],
        "epochs": best_trial.params["epochs"],
        "mixup_alpha": best_trial.params["mixup_alpha"],
        "gesture_logit_margin": best_trial.params["gesture_logit_margin"],
        "gesture_neg_margin": best_trial.params["gesture_neg_margin"],
        "margin_weight": best_trial.params["margin_weight"],
        "penalty_weight": best_trial.params["penalty_weight"],
        "conf_penalty_update_epoch": best_trial.params["conf_penalty_update_epoch"],
        "gesture_ratio": best_trial.params["gesture_ratio"]
        }
    with open(os.path.join(save_dir, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)
    train_and_save_model_per_fold(best_params, X_train, y_train, groups, save_dir="/kaggle/working/fold_models", n_splits=5)
    train_full_model(best_params, X_train, y_train, save_dir="/kaggle/working/full_models")

    def objective(trial):
        imu_cnn_cfg = {
            "hidden_dim": trial.suggest_int("imu_cnn_hidden_dim", 256, 1280, step=256),
            "kernel_size": trial.suggest_int("imu_cnn_kernel_size", 3, 9, step=2),
            "num_layers": trial.suggest_int("imu_cnn_num_layers", 2, 4),
            "se_reduction": trial.suggest_int("imu_cnn_se_reduction", 4, 18, step=2),
            "drop_prob": trial.suggest_float("imu_cnn_drop_prob", 0.10, 0.40)
        }
        tof_thm_cnn_cfg = {
            "hidden_dim": trial.suggest_int("tof_thm_cnn_hidden_dim", 256, 1024, step=256),
            "kernel_size": trial.suggest_int("tof_thm_cnn_kernel_size", 3, 9, step=2),
            "num_layers": trial.suggest_int("tof_thm_cnn_num_layers", 1, 3),
            "se_reduction": trial.suggest_int("tof_thm_se_reduction", 2, 10, step=2),
            "drop_prob": trial.suggest_float("tof_thm_cnn_drop_prob", 0.10, 0.40)
        }
        noise_dense_branch_cfg = {
            "hidden_dim": trial.suggest_int("noise_dense_hidden_dim", 8, 32, step=8),
            "dropout": trial.suggest_float("noise_dense_dropout", 0.05, 0.2),
            "noise_std": trial.suggest_float("noise_dense_noise_std", 0.05, 0.15),
        }
        lstm_cfg = {
            "hidden_dim": trial.suggest_int("lstm_hidden_dim", 128, 512, step=128),
            "num_layers": trial.suggest_int("lstm_num_layers", 1, 3),
            "dropout": trial.suggest_float("lstm_dropout", 0.15, 0.45)
        }
        gru_cfg = {
            "hidden_dim": trial.suggest_int("gru_hidden_dim", 128, 512, step=128),
            "num_layers": trial.suggest_int("gru_num_layers", 1, 3),
            "dropout": trial.suggest_float("gru_dropout", 0.1, 0.35)
        }

        trans_cfg = {
            "num_heads": trial.suggest_categorical("attn_num_heads", [2, 4, 8, 16]),
            "embed_dim" : trial.suggest_int("attn_embed_dim", 256, 1024, step=256),
            "attn_dropout": trial.suggest_float("attn_dropout", 0.05, 0.4),
            "trans_dropout": trial.suggest_float("trans_dropout", 0.1, 0.4),
            "ff_hidden_dim": trial.suggest_int("ff_hidden_dim", 256, 1024, step=256),
            "ff_num_layers": trial.suggest_int("ff_num_layers", 3, 8),
            "ff_dropout": trial.suggest_float("ff_dropout", 0.05, 0.4),
            "reverse_mask_drop_prob": trial.suggest_float("attn_reverse_mask_drop_prob", 0.0, 0.3)
        }
        gesture_head_cfg = {
            "hidden_dim": trial.suggest_int("gesture_hidden_dim", 256, 1024, step=256),
            "num_layers": trial.suggest_int("gesture_num_layers", 1, 4),
            "dropout": trial.suggest_float("gesture_dropout", 0.15, 0.5)
        }
        binary_head_cfg = {
            "hidden_dim": trial.suggest_int("binary_hidden_dim", 128, 256, step=64),
            "num_layers": trial.suggest_int("binary_num_layers", 1, 3),
            "dropout": trial.suggest_float("binary_dropout", 0.05, 0.25)
        }
        activation_cfg = {
            "cnn": trial.suggest_categorical("cnn_activation", ["relu", "silu"]),
            "ffn": trial.suggest_categorical("ff_activation", ["gelu", "silu"]),
            "head": trial.suggest_categorical("head_activation", ["relu", "gelu", "silu"])
        }
        batch_size    = trial.suggest_categorical("batch_size", [64, 128, 256])
        lr            = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        weight_decay  = trial.suggest_float("weight_decay", 1e-7, 1e-4, log=True)
        max_epochs    = trial.suggest_int("epochs", 45, 65)
        specaug_cfg = {
            "max_time_width": trial.suggest_int("specaug_time_width", 3, 15, step=3),
            "max_feature_width": trial.suggest_int("specaug_feature_width", 2, 10, step=2),
            "prob": trial.suggest_float("specaug_prob", 0.1, 0.5)
        }
        focal_gamma = trial.suggest_float("focal_gamma", 1.0, 3.0)
        mixup_alpha = trial.suggest_float("mixup_alpha", 0.0, 0.3, step=0.01)
        gesture_margin = trial.suggest_float("gesture_logit_margin", 0.2, 0.5)
        neg_margin = trial.suggest_float("gesture_neg_margin", 0.05, 0.25)
        margin_weight = trial.suggest_float("margin_weight", 0.1, 0.5)
        penalty_weight = trial.suggest_float("penalty_weight", 0.2, 0.5)
        N = trial.suggest_int("conf_penalty_update_epoch", 5, 10)
        gesture_ratio = trial.suggest_float("gesture_ratio", 0.8, 0.95)

        fold_scores = []
        fold_binary_f1s = []
        fold_macro_f1s = []

        for fold, (train_idx, val_idx) in enumerate(
            stratified_split_with_nontarget(X_tune, y_tune, groups_tune, n_splits=3)
        ):
            model = CNNTwoBranchModel(
                imu_dim=len(imu_cols),
                tof_thm_dim=len(tof_cols+thm_cols),
                imu_cnn_cfg=imu_cnn_cfg,
                tof_thm_cnn_cfg=tof_thm_cnn_cfg,
                noise_dense_branch_cfg=noise_dense_branch_cfg,
                lstm_cfg=lstm_cfg,
                gru_cfg=gru_cfg,
                trans_cfg=trans_cfg,
                gesture_head_cfg=gesture_head_cfg,
                binary_head_cfg=binary_head_cfg,
                activation_cfg=activation_cfg,
                num_classes=num_classes
                ).to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
            stopper = EarlyStopper(patience=8, min_delta=1e-4)
            best_fold_score = -float("inf")

            train_X_gpu = X_tune[train_idx].to(device)
            train_y_gpu = y_tune[train_idx].to(device)
            val_X_gpu   = X_tune[val_idx].to(device)
            val_y_gpu   = y_tune[val_idx].to(device)

            train_loader = create_loader(train_X_gpu, train_y_gpu, batch_size)
            val_loader   = create_loader(val_X_gpu, val_y_gpu, batch_size, shuffle=False)
            train_gesture_labels = train_y_gpu[train_y_gpu != -1].cpu().numpy()
            class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_gesture_labels)
            class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
            gesture_criterion = FocalLoss(gamma=focal_gamma, weight=class_weights_tensor)
            binary_criterion = nn.BCEWithLogitsLoss()
            global_step_base = fold * max_epochs
            specaugment = SpecAugment(**specaug_cfg).to(device)
            conf_penalty = None
            for epoch in range(max_epochs):
                model.train()
                for xb, yb in train_loader:
                    xb_full, yb_full = xb, yb
                    mask = yb != -1
                    xb_gesture, yb_gesture = xb[mask], yb[mask]
                    xb_gesture = specaugment(xb_gesture)

                    inputs, targets_a, targets_b, lam = mixup_data(xb_gesture, yb_gesture, alpha=mixup_alpha)
                    optimizer.zero_grad()
                    gesture_logits, binary_logits = model(xb_full)
                    def mixup_criterion(criterion, pred, y_a, y_b, lam):
                        mask_a = y_a != -1
                        mask_b = y_b != -1
                        mask = mask_a & mask_b
                        if mask.sum() == 0:
                            return torch.tensor(0.0, device=pred.device)
                        return lam * criterion(pred[mask], y_a[mask]) + (1 - lam) * criterion(pred[mask], y_b[mask])

                    def dual_margin_loss(logits, targets, pos_margin=0.3, neg_margin=0.1):
                        B, C = logits.shape
                        true_logits = logits.gather(1, targets.unsqueeze(1)).squeeze(1)  # [B]

                        one_hot = F.one_hot(targets, num_classes=C).bool()
                        logits_masked = logits.masked_fill(one_hot, float('-inf'))

                        # Inter-class margin: max other
                        max_other_logits = logits_masked.max(dim=1).values
                        inter_margin_loss = F.relu(pos_margin - (true_logits - max_other_logits))

                        # Intra-class spread: mean of other logits
                        mean_other_logits = logits_masked.masked_fill(torch.isnan(logits_masked), 0).mean(dim=1)
                        intra_margin_loss = F.relu(neg_margin - (true_logits - mean_other_logits))

                        return inter_margin_loss.mean() + 0.05 * intra_margin_loss.mean()


                    ce_loss = mixup_criterion(gesture_criterion, gesture_logits[mask], targets_a, targets_b, lam)
                    margin_loss = dual_margin_loss(gesture_logits[mask], yb_gesture, pos_margin=gesture_margin, neg_margin=neg_margin)
                    if conf_penalty is None:
                        penalty = torch.tensor(0.0, device=device)
                    else:
                        gesture_probs = torch.softmax(gesture_logits[mask], dim=1)
                        penalty = (conf_penalty[yb_gesture] * gesture_probs).sum(dim=1).mean()
                    gesture_loss = ce_loss + margin_weight * margin_loss + penalty_weight * penalty

                    binary_labels = (yb_full != -1).float().unsqueeze(1)
                    binary_loss = binary_criterion(binary_logits, binary_labels)
                    gesture_weight = gesture_ratio
                    binary_weight = 1 - gesture_ratio
                    loss = gesture_weight * gesture_loss + binary_weight * binary_loss + penalty_weight * penalty

                    loss.backward()
                    optimizer.step()

                model.eval()
                all_gesture_probs = []
                all_binary_probs = []
                all_labels = []

                with torch.no_grad():
                    for xb, yb in val_loader:
                        gesture_logits, binary_logits = model(xb)

                        gesture_probs = torch.softmax(gesture_logits, dim=1)
                        binary_probs = torch.sigmoid(binary_logits).squeeze(-1)

                        all_gesture_probs.append(gesture_probs.cpu().numpy())
                        all_binary_probs.append(binary_probs.cpu().numpy())
                        all_labels.append(yb.cpu().numpy())

                all_gesture_probs = np.concatenate(all_gesture_probs, axis=0)
                all_binary_probs = np.concatenate(all_binary_probs, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)

                def optimize_binary_threshold(binary_probs, gesture_probs, labels):
                    thresholds = np.linspace(0.05, 0.95, 50)
                    best_score, best_thresh = -1, 0.5
                    for th in thresholds:
                        preds = np.full(len(labels), -1)
                        mask = binary_probs >= th
                        preds[mask] = np.argmax(gesture_probs[mask], axis=1)
                        score, _, _ = competition_f1(labels, preds)
                        if score > best_score:
                            best_score, best_thresh = score, th
                    return best_thresh

                best_thresh = optimize_binary_threshold(all_binary_probs, all_gesture_probs, all_labels)
                preds = np.full(len(all_labels), -1)
                mask = all_binary_probs >= best_thresh
                preds[mask] = np.argmax(all_gesture_probs[mask], axis=1)

                score, binary_f1, macro_f1 = competition_f1(all_labels, preds)

                warmup = 5
                if epoch >= warmup and epoch % N == 0:
                    cm = confusion_matrix(all_labels, preds, labels=np.arange(num_classes))
                    cm = cm.astype(np.float32)
                    conf_penalty = torch.tensor(cm / (cm.sum(axis=1, keepdims=True) + 1e-8), device=device)

                trial.report(score, global_step_base + epoch)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
                if stopper.step(score, model):
                    break
                best_fold_score = max(best_fold_score, score)
                model.train()

            if stopper.best_state is not None:
                model.load_state_dict(stopper.best_state)
            fold_scores.append(score)
            fold_binary_f1s.append(binary_f1)
            fold_macro_f1s.append(macro_f1)


        binary_str = " | ".join([f"B{i+1}={b:.4f}" for i, b in enumerate(fold_binary_f1s)])
        macro_str  = " | ".join([f"M{i+1}={m:.4f}" for i, m in enumerate(fold_macro_f1s)])

        mean_binary_f1 = np.mean(fold_binary_f1s)
        mean_macro_f1  = np.mean(fold_macro_f1s)

        log_msg = (f"Trial {trial.number} | "
                   f"{binary_str} || MeanB={mean_binary_f1:.4f} || "
                   f"{macro_str} || MeanM={mean_macro_f1:.4f}"
                    )
        optuna.logging.get_logger("optuna").info(log_msg)
        return -np.mean(fold_scores)



import polars as pl
import kaggle_evaluation.cmi_inference_server


save_dir = '/kaggle/input/cmi-checkpoint/'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load resources once
scaler = joblib.load(f"{save_dir}/scaler.pkl")
label_encoder = joblib.load(f"{save_dir}/label_encoder.pkl")
valid_labels = set(label_encoder.classes_) | {"non_target"}
num_classes = len(label_encoder.classes_)

with open(os.path.join(save_dir, "thresholds.json"), "r") as f:
    thresholds = json.load(f)
    thresholds = {int(k): v for k, v in thresholds.items()}

with open(os.path.join(save_dir, "best_params.json"), "r") as f:
    best_params = json.load(f)

# Build model
model = CNNTwoBranchModel(
    imu_dim=len(imu_cols),
    tof_thm_dim=len(tof_cols + thm_cols),
    imu_cnn_cfg=best_params["imu_cnn_cfg"],
    tof_thm_cnn_cfg=best_params["tof_thm_cnn_cfg"],
    noise_dense_branch_cfg=best_params["noise_dense_branch_cfg"],
    lstm_cfg=best_params["lstm_cfg"],
    gru_cfg=best_params["gru_cfg"],
    trans_cfg=best_params["trans_cfg"],
    gesture_head_cfg=best_params["gesture_head_cfg"],
    binary_head_cfg=best_params["binary_head_cfg"],
    activation_cfg=best_params["activation_cfg"],
    num_classes=num_classes
).to(device)


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    try:
        use_full_model = False

        df = sequence.to_pandas()
        if df.empty:
            return "Text on phone" 

        df = add_new_features(df)
        df[imu_cols + thm_cols + tof_cols] = (
            df[imu_cols + thm_cols + tof_cols]
            .interpolate(method="linear", limit_direction="both", axis=0)
            .fillna(0.0)
        )
        df[imu_cols + thm_cols + tof_cols] = scaler.transform(df[imu_cols + thm_cols + tof_cols])

        df_group = df.copy()
        df_group[imu_cols + thm_cols + tof_cols] = group_normalize(df_group, imu_cols + thm_cols + tof_cols)

        global_tensor, _ = build_tensor(df, imu_cols + thm_cols + tof_cols, cap_len=120, mode="eval")
        group_tensor, _ = build_tensor(df_group, imu_cols + thm_cols + tof_cols, cap_len=120, mode="eval")

        x = torch.cat([global_tensor, group_tensor], dim=2).to(device)

        gesture_outputs = []
        binary_outputs = []

        for fold in range(5):
            model.load_state_dict(torch.load(os.path.join(save_dir, f"fold{fold}_model.pt"), map_location=device))
            model.eval()
            with torch.no_grad():
                gesture_logits, binary_logits = model(x)
                gesture_probs = torch.softmax(gesture_logits, dim=1).cpu().numpy().squeeze()
                binary_prob = torch.sigmoid(binary_logits).cpu().numpy().squeeze()
                gesture_outputs.append(gesture_probs)
                binary_outputs.append(binary_prob)

        avg_gesture_probs = np.mean(gesture_outputs, axis=0)
        avg_binary_prob = np.mean(binary_outputs)
        avg_threshold = np.mean([thresholds[i] for i in range(5)])

        if avg_binary_prob < avg_threshold:
            return "Text on phone" 

        pred_class_idx = avg_gesture_probs.argmax()
        pred_class = label_encoder.inverse_transform([pred_class_idx])[0]

        return str(pred_class) if pred_class in valid_labels else "non_target"

    except Exception as e:
        print(f"Exception in predict(): {e}")
        import traceback
        traceback.print_exc()
        return "Text on phone"


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

