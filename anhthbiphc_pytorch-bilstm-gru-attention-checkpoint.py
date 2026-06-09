# =========================
# CMI Inference - Template
# =========================
import os
import numpy as np
import pandas as pd
import polars as pl
import joblib
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------- 1) PATHS: sửa theo tên dataset output của bạn ----------
CKPT_DIR = "/kaggle/input/cmi-two-branch-bigr-u-checkpoints"  # <-- đổi cho đúng
RAW_DIR  = "/kaggle/input/cmi-detect-behavior-with-sensor-data"

CKPT_FILE = "best_checkpoint.pth"  # hoặc "gesture_two_branch_mixup_pytorch.pth"
SCALER_FILE = "scaler.pkl"
FEATS_FILE = "feature_cols.npy"
MAXLEN_FILE = "sequence_maxlen.npy"
CLASSES_FILE = "gesture_classes.npy"

# --------- 2) MODEL CLASS: dán NGUYÊN si class dùng khi train ----------
# Ví dụ nếu bạn dùng TwoBranchHARModel hoặc TwoBranchModel => dán lại ở đây.
# (BẮT BUỘC class phải khớp kiến trúc với checkpoint)
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Residual IMU block: phải có conv_block, se_block, shortcut_conv ---
class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, channels)
    def forward(self, x):                     # x: (B,C,T)
        s = x.mean(dim=2)                     # (B,C)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s)).unsqueeze(2)  # (B,C,1)
        return x * s

class ResidualSECNNBlock(nn.Module):
    """
    PHẢI ĐÚNG TÊN: conv_block, se_block, shortcut_conv
    """
    def __init__(self, in_ch, out_ch, k=3, drop=0.1):
        super().__init__()
        pad = k // 2
        self.conv_block = nn.Sequential(             # <- tên này phải đúng
            nn.Conv1d(in_ch, out_ch, k, padding=pad, bias=False),  # [0]
            nn.BatchNorm1d(out_ch),                                  # [1]
            nn.ReLU(inplace=True),                                   # [2]
            nn.Conv1d(out_ch, out_ch, k, padding=pad, bias=False),   # [3]
            nn.BatchNorm1d(out_ch),                                  # [4]
        )
        self.se_block = SEBlock(out_ch, reduction=8) # <- đúng tên
        if in_ch != out_ch:
            self.shortcut_conv = nn.Sequential(      # <- đúng tên
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut_conv = nn.Identity()

        self.pool  = nn.MaxPool1d(2)
        self.drop  = nn.Dropout(drop)

    def forward(self, x):
        sc = self.shortcut_conv(x)
        y  = self.conv_block(x)
        y  = self.se_block(y)
        y  = F.relu(y + sc, inplace=True)
        y  = self.pool(y)
        y  = self.drop(y)
        return y

# --- ToF block: PHẢI đặt tên "block" để khớp 'tof_branch_blockX.block.*' ---
class SimpleCNNBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, pool_size=2, drop=0.2):
        super().__init__()
        pad = kernel_size // 2
        self.block = nn.Sequential(                          # <- đúng tên
            nn.Conv1d(in_filters, out_filters, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(out_filters),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size),
            nn.Dropout(drop)
        )
    def forward(self, x):
        return self.block(x)

# --- Attention: PHẢI có thuộc tính 'score_dense' ---
class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.score_dense = nn.Linear(input_dim, 1)           # <- đúng tên
    def forward(self, x):     # x: (B,T,D)
        s = torch.tanh(self.score_dense(x)).squeeze(2)       # (B,T)
        a = torch.softmax(s, dim=1).unsqueeze(2)             # (B,T,1)
        return (x * a).sum(1)                                # (B,D)

class TwoBranchModel(nn.Module):
    def __init__(self, pad_len, imu_dim, tof_dim, n_classes):
        super().__init__()
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim

        # IMU branch: tên block phải trùng
        self.imu_branch_block1 = ResidualSECNNBlock(imu_dim, 64, 3, drop=0.1)
        self.imu_branch_block2 = ResidualSECNNBlock(64, 128, 5, drop=0.1)

        # ToF branch: tên block phải trùng
        self.tof_branch_block1 = SimpleCNNBlock(tof_dim, 64, drop=0.2)
        self.tof_branch_block2 = SimpleCNNBlock(64, 128, drop=0.2)

        merged_cnn_features = 128 + 128
        self.recurrent_layer = nn.LSTM(
            input_size=merged_cnn_features,
            hidden_size=128,
            bidirectional=True,
            batch_first=True
        )

        self.attention_layer = AttentionLayer(input_dim=128*2)

        self.classifier_head = nn.Sequential(  # checkpoint có 'classifier_head.*'
            nn.Linear(256, 256, bias=False),   # [0]
            nn.BatchNorm1d(256),               # [1]
            nn.ReLU(inplace=True),             # [2]
            nn.Dropout(0.5),                   # [3]
            nn.Linear(256, 128, bias=False),   # [4]
            nn.BatchNorm1d(128),               # [5]
            nn.ReLU(inplace=True),             # [6]
            nn.Dropout(0.3),                   # [7]
        )
        self.output_layer = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, pad_len, F)
        imu = x[:, :, :self.imu_dim]
        tof = x[:, :, self.imu_dim:]
        imu = imu.permute(0, 2, 1)   # (B,C,T)
        tof = tof.permute(0, 2, 1)

        x1 = self.imu_branch_block1(imu)   # (B,64,T/2)
        x1 = self.imu_branch_block2(x1)    # (B,128,T/4)

        x2 = self.tof_branch_block1(tof)   # (B,64,T/2)
        x2 = self.tof_branch_block2(x2)    # (B,128,T/4)

        # concat theo channel
        merged = torch.cat([x1, x2], dim=1)   # (B,256,T/4)
        merged = merged.permute(0, 2, 1)      # (B,T/4,256)

        rnn_out, _ = self.recurrent_layer(merged)  # (B,T/4,256)
        att = self.attention_layer(rnn_out)        # (B,256)
        h = self.classifier_head(att)              # (B,128)
        logits = self.output_layer(h)              # (B,n_classes)
        return logits


# --------- 3) LOAD ARTEFACTS + MODEL ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_cols = np.load(os.path.join(CKPT_DIR, FEATS_FILE), allow_pickle=True).tolist()
pad_len      = int(np.load(os.path.join(CKPT_DIR, MAXLEN_FILE)))
classes      = np.load(os.path.join(CKPT_DIR, CLASSES_FILE), allow_pickle=True)
scaler       = joblib.load(os.path.join(CKPT_DIR, SCALER_FILE))

imu_dim  = len([c for c in feature_cols if c.startswith(('acc_', 'rot_'))])
ttf_dim  = len(feature_cols) - imu_dim
num_cls  = len(classes)

model = TwoBranchModel(
    pad_len=pad_len,
    imu_dim=imu_dim,
    tof_dim=ttf_dim,
    n_classes=num_cls
).to(device)


ckpt_path = os.path.join(CKPT_DIR, CKPT_FILE)

# PyTorch 2.6: cần weights_only=False nếu checkpoint là dict pickled
ckpt_path = os.path.join(CKPT_DIR, CKPT_FILE)  # ví dụ 'best_checkpoint.pth'
ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state = ckpt['model_state_dict'] if isinstance(ckpt, dict) and 'model_state_dict' in ckpt else ckpt
# nếu từng torch.compile
state = {k.replace('_orig_mod.', ''): v for k, v in state.items()}
# trước đây:
# model.load_state_dict(state, strict=True)

# thay bằng:
missing, unexpected = model.load_state_dict(state, strict=False)
if missing:
    print("Missing keys:", missing)
if unexpected:
    print("Unexpected keys:", unexpected)

model.eval()




# ====== (A) Rebuild engineered features exactly like train ======
def add_engineered_features(seq_df: pd.DataFrame) -> pd.DataFrame:
    df = seq_df.copy()

    # IMU engineered features (an toàn khi thiếu cột)
    for col in ['acc_x','acc_y','acc_z','rot_w']:
        if col not in df.columns:
            df[col] = 0.0

    # acc_mag, rot_angle
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(np.clip(df['rot_w'].astype(float), -1, 1))

    # group theo sequence_id nếu có, nếu không thì diff theo index
    if 'sequence_id' in df.columns:
        df['acc_mag_jerk'] = df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
        df['rot_angle_vel'] = df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    else:
        df['acc_mag_jerk'] = df['acc_mag'].diff().fillna(0)
        df['rot_angle_vel'] = df['rot_angle'].diff().fillna(0)

    # ToF aggregates: mean/std/min/max trên 64 pixel cho từng cảm biến 1..5
    # Nếu không có đủ cột tof_*_v* thì bỏ qua silently (để tránh crash sample lạ)
    # ToF aggregates: mean/std/min/max trên 64 pixel cho từng cảm biến 1..5
    for i in range(1, 6):
        pix_cols = [f"tof_{i}_v{p}" for p in range(64)]
        if not any(col in df.columns for col in pix_cols):
        # Không có cột ToF -> tạo cột 0
            df[f'tof_{i}_mean'] = 0.0
            df[f'tof_{i}_std']  = 0.0
            df[f'tof_{i}_min']  = 0.0
            df[f'tof_{i}_max']  = 0.0
            continue

    # Lấy những cột tồn tại; nếu thiếu vài cột vẫn chạy được
        exist_cols = [c for c in pix_cols if c in df.columns]
        block = df[exist_cols].replace(-1, np.nan).to_numpy(dtype=float)  # (T, K<=64)

    # Tính với NaN-safe; sau đó NaN -> 0 để tránh warning & giữ shape ổn định
        with np.errstate(all='ignore'):
            m = np.nanmean(block, axis=1)
            s = np.nanstd(block,  axis=1)  # ddof=0 mặc định
            mn = np.nanmin(block,  axis=1)
            mx = np.nanmax(block,  axis=1)

        m  = np.nan_to_num(m,  nan=0.0)
        s  = np.nan_to_num(s,  nan=0.0)
        mn = np.nan_to_num(mn, nan=0.0)
        mx = np.nan_to_num(mx, nan=0.0)

        df[f'tof_{i}_mean'] = m
        df[f'tof_{i}_std']  = s
        df[f'tof_{i}_min']  = mn
        df[f'tof_{i}_max']  = mx


    return df

# ====== (B) Preprocess khớp TwoBranchModel: (1, pad_len, F) ======
from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

def preprocess_sequence_two_branch(seq_df: pd.DataFrame) -> torch.Tensor:
    # 1) rebuild engineered features
    df_eng = add_engineered_features(seq_df)

    # 2) đảm bảo đủ cột theo feature_cols.npy (thứ tự PHẢI khớp)
    missing = [c for c in feature_cols if c not in df_eng.columns]
    if missing:
        # tạo cột trống cho mọi cột còn thiếu (trường hợp thm_* v.v. vắng)
        for c in missing:
            df_eng[c] = 0.0

    # 3) chọn và sắp xếp đúng thứ tự
    X = df_eng[feature_cols].ffill().bfill().fillna(0).to_numpy(dtype=float)   # (T, F)

    # 4) scale theo scaler đã fit khi train
    X = scaler.transform(X)

    # 5) pad/truncate về pad_len
    Xpad = keras_pad_sequences([X], maxlen=pad_len, dtype='float32',
                               padding='post', truncating='post')[0]           # (pad_len, F)

    # 6) tensor cho TwoBranchModel: (1, pad_len, F)
    return torch.from_numpy(Xpad).unsqueeze(0).float()

# ====== (C) predict() cho Inference Server ======
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    seq_df = sequence.to_pandas()

    # nếu lúc train bạn KHÔNG merge demographics -> không cần merge ở đây
    # nếu có merge demographics khi train, bạn phải merge tương tự tại đây.

    x = preprocess_sequence_two_branch(seq_df).to(device)
    with torch.no_grad():
        logits = model(x)
        idx = int(torch.argmax(logits, dim=1).item())
    return str(classes[idx])


# --------- 5) START CMI INFERENCE SERVER ----------
import kaggle_evaluation.cmi_inference_server

server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    server.serve()
else:
    server.run_local_gateway(
        data_paths=(
            f"{RAW_DIR}/test.csv",
            f"{RAW_DIR}/test_demographics.csv",
        )
    )


