import os, json, joblib, numpy as np, pandas as pd
import random, math
from pathlib import Path
import warnings 
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from timm.scheduler import CosineLRScheduler
from scipy.signal import firwin
import polars as pl
from tqdm import tqdm
from transformers import BertModel, BertConfig

import numpy as np
from scipy.fft import rfft, rfftfreq

# ================================
import kagglehub
metric = kagglehub.package_import('jiazhuang/cmi-2025-metric')

def get_competition_score(true, pred):
    assert len(true) == len(pred)
    N = len(true)
    true = pd.DataFrame({'id': range(N), 'gesture': true})
    pred = pd.DataFrame({'id': range(N), 'gesture': pred})
    return metric.score(true, pred, 'id')

# ================================
# TTA Helpers
# ================================
def tta_jitter(x, sigma=0.01):
    noise = np.random.randn(*x.shape) * sigma
    return x + noise


# Configuration
# è‡ªåŠ¨æ£€æµ‹æ˜¯å�¦æœ‰é¢„è®­ç»ƒæ–‡ä»¶ï¼Œå¦‚æ�œæ²¡æœ‰åˆ™è®­ç»ƒ
PRETRAINED_DIR = Path("/kaggle/input/cmiyyh")
required_files = [
    PRETRAINED_DIR / "feature_cols.npy",
    PRETRAINED_DIR / "sequence_maxlen.npy", 
    PRETRAINED_DIR / "scaler.pkl",
    PRETRAINED_DIR / "gesture_classes.npy"
]

# æ£€æŸ¥æ˜¯å�¦å­˜åœ¨æ‰€æœ‰å¿…éœ€çš„é¢„è®­ç»ƒæ–‡ä»¶
has_pretrained = all(f.exists() for f in required_files)
TRAIN = False  # å¦‚æ�œæ²¡æœ‰é¢„è®­ç»ƒæ–‡ä»¶ï¼Œåˆ™è¿›è¡Œè®­ç»ƒ

print(f"ğŸ”� Pretrained files check: {has_pretrained}")
print(f"ğŸ�¯ Mode: {'TRAINING' if TRAIN else 'INFERENCE'}")

RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
EXPORT_DIR = Path("./")                                    # artefacts will be saved here
BATCH_SIZE = 64
PAD_PERCENTILE = 100
maxlen = PAD_PERCENTILE
LR_INIT = 1e-3
WD = 3e-3
# MIXUP_ALPHA = 0.4
PATIENCE = 40
FOLDS = 5
random_state = 42
epochs_warmup = 20
warmup_lr_init = 1.822126131809773e-05
lr_min = 3.810323058740104e-09

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"â–¶ imports ready Â· pytorch {torch.__version__} Â· device: {device}")

# ================================
# Model Components
# ================================
mean = torch.tensor([
    0,  0, 0, 0, 0,
    0,  9.0319e-03,  1.0849e+00, -2.6186e-03,  3.7651e-03,
    -5.3660e-03, -2.8177e-03,  1.3318e-03, -1.5876e-04,  6.3495e-01,
     6.2877e-01,  6.0607e-01,  6.2142e-01,  6.3808e-01,  6.5420e-01,
     7.4102e-03, -3.4159e-03, -7.5237e-03, -2.6034e-02,  2.9704e-02,
    -3.1546e-02, -2.0610e-03, -4.6986e-03, -4.7216e-03, -2.6281e-02,
     1.5799e-02,  1.0016e-02,
     0, 0, 0, 1  # rot_x, rot_y, rot_z, rot_w
], dtype=torch.float32).view(1, -1, 1).to(device)         

std = torch.tensor([
    1, 1, 1, 1, 1, 1, 0.2067, 0.8583, 0.3162,
    0.2668, 0.2917, 0.2341, 0.3023, 0.3281, 1.0264, 0.8838, 0.8686, 1.0973,
    1.0267, 0.9018, 0.4658, 0.2009, 0.2057, 1.2240, 0.9535, 0.6655, 0.2941,
    0.3421, 0.8156, 0.6565, 1.1034, 1.5577,
    1, 1, 1, 1  # rot_x, rot_y, rot_z, rot_w
], dtype=torch.float32).view(1, -1, 1).to(device) + 1e-8  

class ImuFeatureExtractor(nn.Module):
    def __init__(self, fs=100., add_quaternion=False, K=10):
        super().__init__()
        self.fs = fs
        self.add_quaternion = add_quaternion
        self.K = K  # Number of frequency components to keep

        k = 15
        self.lpf = nn.Conv1d(6, 6, kernel_size=k, padding=k//2,
                             groups=6, bias=False)
        nn.init.kaiming_uniform_(self.lpf.weight, a=math.sqrt(5))

        self.lpf_acc  = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)
        self.lpf_gyro = nn.Conv1d(3, 3, k, padding=k//2, groups=3, bias=False)

    def forward(self, imu):
        B, C, T = imu.shape
        acc  = imu[:, 0:3, :]                 # acc_x, acc_y, acc_z
        gyro = imu[:, 3:6, :]                 # gyro_x, gyro_y, gyro_z
        quat = imu[:, 6:10, :]                # rot_x, rot_y, rot_z, rot_w
        extra = imu[:, 10:, :]                 

        # 1) magnitude
        acc_mag  = torch.norm(acc,  dim=1, keepdim=True)          # (B,1,T)
        gyro_mag = torch.norm(gyro, dim=1, keepdim=True)

        # 2) jerk 
        jerk = F.pad(acc[:, :, 1:] - acc[:, :, :-1], (1,0))       # (B,3,T)
        gyro_delta = F.pad(gyro[:, :, 1:] - gyro[:, :, :-1], (1,0))

        # 3) energy
        acc_pow  = acc ** 2
        gyro_pow = gyro ** 2

        # 4) LPF / HPF 
        acc_lpf  = self.lpf_acc(acc)
        acc_hpf  = acc - acc_lpf
        gyro_lpf = self.lpf_gyro(gyro)
        gyro_hpf = gyro - gyro_lpf

        # 5) FFT Amplitude Spectrum
        acc_fft = torch.from_numpy(np.abs(rfft(acc.cpu().numpy(), axis=-1)))[:, :, :self.K].to(acc.device)
        gyro_fft = torch.from_numpy(np.abs(rfft(gyro.cpu().numpy(), axis=-1)))[:, :, :self.K].to(gyro.device)

        # 6) Spectral Entropy
        def spectral_entropy(signal):
            psd = np.abs(rfft(signal, axis=-1)) ** 2
            psd_norm = psd / np.sum(psd, axis=-1, keepdims=True)
            se = -np.sum(psd_norm * np.log(psd_norm + 1e-12), axis=-1)
            return torch.from_numpy(se).to(signal.device)

        acc_entropy = spectral_entropy(acc.cpu().numpy())
        gyro_entropy = spectral_entropy(gyro.cpu().numpy())

        # 7) Spectral Centroid
        def spectral_centroid(signal):
            freqs = rfftfreq(signal.shape[-1], 1/self.fs)
            centroid = np.sum(freqs * np.abs(rfft(signal, axis=-1)), axis=-1) / np.sum(np.abs(rfft(signal, axis=-1)), axis=-1)
            return torch.from_numpy(centroid).to(signal.device)

        acc_centroid = spectral_centroid(acc.cpu().numpy())
        gyro_centroid = spectral_centroid(gyro.cpu().numpy())

        # 8) Dominant Frequency
        def dominant_frequency(signal):
            freqs = rfftfreq(signal.shape[-1], 1/self.fs)
            dom_freq = freqs[np.argmax(np.abs(rfft(signal, axis=-1)), axis=-1)]
            return torch.from_numpy(dom_freq).to(signal.device)

        acc_dom_freq = dominant_frequency(acc.cpu().numpy())
        gyro_dom_freq = dominant_frequency(gyro.cpu().numpy())

        features = [
            acc, gyro, quat,
            acc_mag, gyro_mag,
            jerk, gyro_delta,
            acc_pow, gyro_pow,
            acc_lpf, acc_hpf,
            gyro_lpf, gyro_hpf,
            acc_fft, gyro_fft,
            acc_entropy.unsqueeze(1), gyro_entropy.unsqueeze(1),
            acc_centroid.unsqueeze(1), gyro_centroid.unsqueeze(1),
            acc_dom_freq.unsqueeze(1), gyro_dom_freq.unsqueeze(1)
        ]
        return torch.cat(features, dim=1)  # (B, C_out, T)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3, weight_decay=1e-4):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        # First conv
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv
        out = self.bn2(self.conv2(out))
        
        # SE block
        out = self.se(out)
        
        # Add shortcut
        out += shortcut
        out = F.relu(out)
        
        # Pool and dropout
        out = self.pool(out)
        out = self.dropout(out)
        
        return out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context

class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()

        # Spatial attention
        self.conv_spatial = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        b, c, t = x.size()
        avg_out = self.shared_mlp(self.avg_pool(x).view(b, c))
        max_out = self.shared_mlp(self.max_pool(x).view(b, c))
        channel_att = self.sigmoid_channel(avg_out + max_out).view(b, c, 1)
        x = x * channel_att.expand_as(x)

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(torch.cat([avg_out, max_out], dim=1)))
        x = x * spatial_att.expand_as(x)

        return x

class MultiModalModel(nn.Module):
    def __init__(self, imu_dim, tof_dim, thm_dim, d_model, n_classes):
        super().__init__()
        self.imu_dim = imu_dim
        self.tof_dim = tof_dim
        self.thm_dim = thm_dim
        self.d_model = d_model
        self.n_classes = n_classes

        # IMU deep branch
        self.imu_block1 = ResidualSECNNBlock(imu_dim, 64, 3)
        self.imu_block2 = ResidualSECNNBlock(64, 128, 5)
        
        # TOF lighter branch - åŠ å¼ºCNNç»“æ�„
        # æ›´æ–°TOFåˆ†æ”¯ä»¥é€‚åº”å¢�åŠ çš„ç»Ÿè®¡ç‰¹å¾�ç»´åº¦
        self.tof_branch = nn.Sequential(
            # ç¬¬ä¸€å±‚å�·ç§¯ï¼šå°†TOFç‰¹å¾�æ•°æ˜ å°„åˆ°ä¸­é—´é€šé�“æ•°
            nn.Conv1d(tof_dim, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),  # æ‰¹å½’ä¸€åŒ–
            nn.ReLU(inplace=True),                    # æ¿€æ´»å‡½æ•°
            nn.MaxPool1d(2, ceil_mode=True),          # ä¸‹é‡‡æ ·ï¼ˆå‡�å�Šæ—¶é—´ç»´åº¦ï¼‰
            nn.Dropout(0.3),       # é˜²æ­¢è¿‡æ‹Ÿå�ˆ

            # ç¬¬äºŒå±‚å�·ç§¯ï¼šä¿�æŒ�é€šé�“æ•°ï¼Œè¿›ä¸€æ­¥æ��å�–ç‰¹å¾�
            nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2, ceil_mode=True),          # å†�æ¬¡ä¸‹é‡‡æ ·
            nn.Dropout(0.3)
        )

        # THM lighter branch
        self.thm_conv1 = nn.Conv1d(thm_dim, 64, 3, padding=1, bias=False)
        self.thm_bn1 = nn.BatchNorm1d(64)
        self.thm_pool1 = nn.MaxPool1d(2)
        self.thm_drop1 = nn.Dropout(0.3)
        
        self.thm_conv2 = nn.Conv1d(64, 128, 3, padding=1, bias=False)
        self.thm_bn2 = nn.BatchNorm1d(128)
        self.thm_pool2 = nn.MaxPool1d(2)
        self.thm_drop2 = nn.Dropout(0.3)

        # CBAM Block for each modality
        self.cbam_imu = CBAMBlock(128)
        self.cbam_tof = CBAMBlock(128)
        self.cbam_thm = CBAMBlock(128)
        
        # BERT for fusion
        bert_config = BertConfig(hidden_size=384, num_hidden_layers=4, num_attention_heads=8, intermediate_size=512)  # 128*3 for three modalities
        self.bert = BertModel(bert_config)
        
        # Attentionjiapin
        self.attention = AttentionLayer(384)  # 128*3 for three modalities
        
        # Dense layers
        self.dense1 = nn.Linear(384, 256, bias=False)
        self.bn_dense1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.3)
        
        self.dense2 = nn.Linear(256, 128, bias=False)
        self.bn_dense2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        
        self.classifier = nn.Linear(128, n_classes)
        
    def forward(self, imu, tof, thm):
        # IMU branch
        x1 = self.imu_block1(imu)
        x1 = self.imu_block2(x1)
        x1 = self.cbam_imu(x1)
        
        # TOF branch - ä½¿ç”¨åŠ å¼ºçš„CNNç»“æ�„
        x2 = self.tof_branch(tof)
        x2 = self.cbam_tof(x2)

        # THM branch
        x3 = F.relu(self.thm_bn1(self.thm_conv1(thm)))
        x3 = self.thm_drop1(self.thm_pool1(x3))
        x3 = F.relu(self.thm_bn2(self.thm_conv2(x3)))
        x3 = self.thm_drop2(self.thm_pool2(x3))
        x3 = self.cbam_thm(x3)

        # Concatenate branches
        merged = torch.cat([x1, x2, x3], dim=1).transpose(1, 2)  # (batch, seq_len, 384)
        
        # BERT for fusion
        bert_output = self.bert(inputs_embeds=merged).last_hidden_state
        
        # Attention
        attended = self.attention(bert_output)
        
        # Dense layers
        x = F.relu(self.bn_dense1(self.dense1(attended)))
        x = self.drop1(x)
        x = F.relu(self.bn_dense2(self.dense2(x)))
        x = self.drop2(x)
        
        # Classification
        logits = (self.classifier(x))
        return logits


# ================================
# Data Handling
# ================================

def pad_sequences_torch(sequences, maxlen, padding='post', truncating='post', value=0.0):
    result = []
    for seq in sequences:
        if len(seq) >= maxlen:
            seq = seq[:maxlen] if truncating=='post' else seq[-maxlen:]
        else:
            pad_len = maxlen - len(seq)
            pad_array = np.full((pad_len, seq.shape[1]), value)
            seq = np.concatenate([seq, pad_array], axis=0) if padding=='post' else np.concatenate([pad_array, seq], axis=0)
        result.append(seq)
    return np.array(result, dtype=np.float32)

# æ·»åŠ TOFç»Ÿè®¡ç‰¹å¾�æ��å�–å‡½æ•°
def extract_tof_statistics(tof_data, tof_cols):
    """
    ä»�å�Ÿå§‹TOFæ•°æ�®ä¸­æ��å�–ç»Ÿè®¡ç‰¹å¾�
    åŒ…æ‹¬å…¨å±€ç»Ÿè®¡å’ŒåŒºåŸŸç»Ÿè®¡
    """
    B, T, D = tof_data.shape
    tof_reshaped = tof_data.reshape(B*T, D)  # (B*T, D)
    
    # å…¨å±€ç»Ÿè®¡ç‰¹å¾�: å�‡å€¼ã€�æ ‡å‡†å·®ã€�æœ€å°�å€¼ã€�æœ€å¤§å€¼
    tof_mean = np.mean(tof_reshaped, axis=1, keepdims=True)  # (B*T, 1)
    tof_std = np.std(tof_reshaped, axis=1, keepdims=True)    # (B*T, 1)
    tof_min = np.min(tof_reshaped, axis=1, keepdims=True)    # (B*T, 1)
    tof_max = np.max(tof_reshaped, axis=1, keepdims=True)    # (B*T, 1)
    
    # åŒºåŸŸç»Ÿè®¡ç‰¹å¾�: æ ¹æ�®å®�é™…çš„TOFç»´åº¦(320)åˆ’åˆ†ä¸ºå�ˆé€‚åŒºåŸŸ
    # å°†320ä¸ªåƒ�ç´ åˆ’åˆ†ä¸º20ä¸ªåŒºåŸŸï¼Œæ¯�ä¸ªåŒºåŸŸ16ä¸ªåƒ�ç´ 
    num_regions = 20
    pixels_per_region = D // num_regions  # 320 // 20 = 16
    
    tof_regions = tof_reshaped.reshape(B*T, num_regions, pixels_per_region)  # (B*T, 20, 16)
    region_means = np.mean(tof_regions, axis=2)     # (B*T, 20)
    region_stds = np.std(tof_regions, axis=2)       # (B*T, 20)
    region_mins = np.min(tof_regions, axis=2)       # (B*T, 20)
    region_maxs = np.max(tof_regions, axis=2)       # (B*T, 20)
    
    # å�ˆå¹¶æ‰€æœ‰ç»Ÿè®¡ç‰¹å¾�
    stats_features = np.concatenate([
        tof_mean, tof_std, tof_min, tof_max,           # å…¨å±€ç»Ÿè®¡ (4)
        region_means, region_stds, region_mins, region_maxs  # åŒºåŸŸç»Ÿè®¡ (20*4=80)
    ], axis=1)  # (B*T, 84)
    
    # é‡�å¡‘å›�(B, T, 84)
    stats_features = stats_features.reshape(B, T, 84)
    
    return stats_features.astype(np.float32)

def preprocess_sequence(df_seq, feature_cols: list, scaler: StandardScaler):
    # ç¡®ä¿�è¾“å…¥æ˜¯pandas DataFrame
    if hasattr(df_seq, 'to_pandas'):
        df_seq = df_seq.to_pandas()
    
    # å¤„ç�†ç¼ºå¤±å€¼
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

class CMI3Dataset(Dataset):
    def __init__(self,
                 X_list,
                 y_list,
                 maxlen,
                 mode="train",
                 imu_dim=7,
                 tof_dim=64,  # æ·»åŠ TOFç»´åº¦å�‚æ•°
                 augment=None):
        self.X_list = X_list
        self.mode = mode
        self.y_list = y_list
        self.maxlen = maxlen
        self.imu_dim = imu_dim     
        self.tof_dim = tof_dim    # å­˜å‚¨TOFç»´åº¦
        self.augment = augment   

    def pad_sequences_torch(self, seq, maxlen, padding='post', truncating='post', value=0.0):

        if seq.shape[0] >= maxlen:
            if truncating == 'post':
                seq = seq[:maxlen]
            else:  # 'pre'
                seq = seq[-maxlen:]
        else:
            pad_len = maxlen - seq.shape[0]
            if padding == 'post':
                seq = np.concatenate([seq, np.full((pad_len, seq.shape[1]), value)])
            else:  # 'pre'
                seq = np.concatenate([np.full((pad_len, seq.shape[1]), value), seq])
        return seq  
        
    def __getitem__(self, index):
        X = self.X_list[index]
        y = self.y_list[index]

        # ---------- (A)  Augmentation ----------
        if self.mode == "train" and self.augment is not None:
            X = self.augment(X, self.imu_dim)     

        X = self.pad_sequences_torch(X, self.maxlen, 'pre', 'pre')
        return X, y
    
    def __len__(self):
        return len(self.X_list)


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False
    
    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

def set_seed(seed: int = 42):
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True)

class Augment:
    def __init__(self,
                 p_jitter=0.8, sigma=0.02, scale_range=[0.9,1.1],
                 p_dropout=0.3,
                 p_moda=0.5,          
                 drift_std=0.005,     
                 drift_max=0.25):      
        self.p_jitter  = p_jitter
        self.sigma     = sigma
        self.scale_min, self.scale_max = scale_range
        self.p_dropout = p_dropout
        self.p_moda    = p_moda
        self.drift_std = drift_std
        self.drift_max = drift_max

    # ---------- Jitter & Scaling ----------
    def jitter_scale(self, x: np.ndarray) -> np.ndarray:
        noise  = np.random.randn(*x.shape) * self.sigma
        scale  = np.random.uniform(self.scale_min,
                                   self.scale_max,
                                   size=(1, x.shape[1]))
        return (x + noise) * scale

    # ---------- Sensor Drop-out ----------
    def sensor_dropout(self,
                       x: np.ndarray,
                       imu_dim: int) -> np.ndarray:

        if random.random() < self.p_dropout:
            x[:, imu_dim:] = 0.0
        return x

    def motion_drift(self, x: np.ndarray, imu_dim: int) -> np.ndarray:

        T = x.shape[0]

        drift = np.cumsum(
            np.random.normal(scale=self.drift_std, size=(T, 1)),
            axis=0
        )
        drift = np.clip(drift, -self.drift_max, self.drift_max)   

        x[:, :6] += drift

        if imu_dim > 6:
            x[:, 6:imu_dim] += drift     
        return x
    
    def random_erasing(self, x: np.ndarray, p_erase=0.5, erase_ratio=0.1) -> np.ndarray:
        if random.random() < p_erase:
            T, D = x.shape
            erase_size = int(T * erase_ratio)
            start_idx = random.randint(0, T - erase_size)
            x[start_idx:start_idx + erase_size, :] = 0.0
        return x

    # ---------- master call ----------
    def __call__(self,
                 x: np.ndarray,
                 imu_dim: int) -> np.ndarray:
        if random.random() < self.p_jitter:
            x = self.jitter_scale(x)

        if random.random() < self.p_moda:
            x = self.motion_drift(x, imu_dim)

        x = self.sensor_dropout(x, imu_dim)

        # Apply random erasing
        x = self.random_erasing(x)

        return x
# ================================
if TRAIN:
    print("â–¶ TRAIN MODE â€“ loading dataset â€¦")
    df = pd.read_csv(RAW_DIR / "train.csv")

    # Label encoding
    le = LabelEncoder()
    df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

    # Feature list
    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('tof_')]
    thm_cols = [c for c in feature_cols if c.startswith('thm_')]
    print(f"  IMU {len(imu_cols)} | TOF {len(tof_cols)} | THM {len(thm_cols)} | total {len(feature_cols)} features")

    # Global scaler
    scaler = StandardScaler().fit(df[feature_cols].ffill().bfill().fillna(0).values)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")

    # Build sequences
    seq_gp = df.groupby('sequence_id')
    X_list, y_list, id_list = [], [], []
    lens = []  # æ·»åŠ è¿™è¡Œæ�¥æ”¶é›†é•¿åº¦ä¿¡æ�¯
    for seq_id, seq in seq_gp:
        mat = preprocess_sequence(seq, feature_cols, scaler)
        X_list.append(mat)
        y_list.append(seq['gesture_int'].iloc[0])
        id_list.append(seq_id)
        lens.append(len(mat))  # æ”¶é›†æ¯�ä¸ªåº�åˆ—çš„é•¿åº¦
    
    pad_len = PAD_PERCENTILE#int(np.percentile(lens, PAD_PERCENTILE))
    print(pad_len)
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(feature_cols))
    id_list = np.array(id_list)
    
    # ä¿®æ”¹è¿™é‡Œï¼šå…ˆä½¿ç”¨pad_sequences_torchå¤„ç�†X_listï¼Œç„¶å��å†�è¿›è¡ŒTOFç»Ÿè®¡ç‰¹å¾�æ��å�–
    X_list_padded = pad_sequences_torch(X_list, maxlen=pad_len, padding='pre', truncating='pre')
    
    # åœ¨è¿™é‡Œæ·»åŠ TOFç»Ÿè®¡ç‰¹å¾�
    original_tof_dim = len(tof_cols)
    
    # æ��å�–TOFç»Ÿè®¡ç‰¹å¾�
    tof_data = X_list_padded[:, :, len(imu_cols):len(imu_cols)+len(tof_cols)]  # (N, T, 320)
    tof_stats = extract_tof_statistics(tof_data, tof_cols)  # (N, T, 84)
    
    # æ›¿æ�¢å�Ÿå§‹TOFæ•°æ�®ä¸ºç»Ÿè®¡ç‰¹å¾�
    other_data = np.concatenate([
        X_list_padded[:, :, :len(imu_cols)],  # IMUæ•°æ�®
        X_list_padded[:, :, len(imu_cols)+len(tof_cols):]  # THMæ•°æ�®
    ], axis=2)  # (N, T, D-len(tof_cols))
    
    # ç»„å�ˆæ–°ç‰¹å¾�ï¼šIMU + TOFç»Ÿè®¡ + THM
    X_list_enhanced = np.concatenate([other_data, tof_stats], axis=2)
    
    X_list_all = X_list_enhanced  # å·²ç»�æ˜¯å¡«å……å��çš„æ•°ç»„
    y_list_all = np.eye(len(le.classes_))[y_list].astype(np.float32)  # One-hot encoding

    augmenter = Augment(
        p_jitter=0.9844818619033621, sigma=0.03291295776089293, scale_range=(0.7542342630597011,1.1625052821731077),
        p_dropout=0.41782786013520684,
        p_moda=0.3910622476959722, drift_std=0.0040285239353308015, drift_max=0.3929358950258158    
    )  
EPOCHS = 145
if TRAIN:
    # Split
    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=random_state)
    models = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(id_list, np.argmax(y_list_all, axis=1))):

        train_list= X_list_all[train_idx]
        train_y_list= y_list_all[train_idx]
        val_list = X_list_all[val_idx]
        val_y_list= y_list_all[val_idx]

        
        # Data loaders
        train_dataset = CMI3Dataset(train_list, train_y_list, maxlen, mode="train", imu_dim=len(imu_cols),
                                augment=augmenter)
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True)
    
        val_dataset = CMI3Dataset(val_list, val_y_list, maxlen, mode="val")
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, drop_last=True)

    
        # Model
        model = MultiModalModel(
            imu_dim=len(imu_cols),
            tof_dim=84,  # æ›´æ–°ä¸ºæ–°çš„TOFç»Ÿè®¡ç‰¹å¾�ç»´åº¦
            thm_dim=len(thm_cols),  # Now using actual THM data
            d_model=64,
            n_classes=len(le.classes_)
        ).to(device)
        ema = EMA(model, decay=0.999)
        # Optimizer and scheduler
        optimizer = Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
        
        steps_per_epoch = len(train_loader)
        nbatch = len(train_loader)
        warmup = epochs_warmup * nbatch
        nsteps = EPOCHS * nbatch
        scheduler = CosineLRScheduler(optimizer,
                  warmup_t=warmup, warmup_lr_init=warmup_lr_init, warmup_prefix=True,
                  t_initial=(nsteps - warmup), lr_min=lr_min) 
    
        early_stopping = EarlyStopping(patience=PATIENCE, restore_best_weights=True)
    
        train_loss = 0.0
        train_acc = None
        val_acc = None
        val_best_acc = 0.0
        i_scheduler = 0
        val_loss = 0.0
        
        # Training loop
        print("â–¶ Starting training...")
        for epoch in range(EPOCHS):
            model.train()
            train_preds = []
            train_targets = []
            train_loss = 0.0
            with tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}") as t:
                for X, y in t:
                    X, y = X.float().to(device),y.float().to(device)
                    optimizer.zero_grad()
                    
                    # Ensure input data is correctly shaped
                    B, T, D = X.shape
                    imu = X[:, :, :len(imu_cols)].permute(0, 2, 1)  # [B, C, T]
                    # ä¿®æ­£TOFå’ŒTHMæ•°æ�®çš„ç´¢å¼•æ–¹å¼�ï¼Œç¡®ä¿�TOFå�ªä½¿ç”¨ç»Ÿè®¡ç‰¹å¾�
                    tof_start_idx = len(imu_cols)
                    tof_end_idx = len(imu_cols) + 84  # TOFç»Ÿè®¡ç‰¹å¾�ç»´åº¦ä¸º84
                    tof = X[:, :, tof_start_idx:tof_end_idx].permute(0, 2, 1)  # TOFç»Ÿè®¡ç‰¹å¾� (84 channels)
                    thm = X[:, :, -len(thm_cols):].permute(0, 2, 1)  # THMæ•°æ�®
                    
                    logits = model(imu, tof, thm)

                    loss = -torch.sum(F.log_softmax(logits, dim=1) * y, dim=1).mean()
                    loss.backward()
                    optimizer.step()
                    ema.update(model)
                    train_preds.extend(logits.argmax(dim=1).cpu().numpy())
                    train_targets.extend(y.argmax(dim=1).cpu().numpy())
                    scheduler.step(i_scheduler)
                    i_scheduler +=1

                    train_loss += loss.item()
                    t.set_postfix(loss=train_loss/len(train_loader))

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_preds = []
            val_targets = []
            with torch.no_grad():
                with tqdm(val_loader, desc="Validation") as t:
                    for X, y in t:
                        X, y = X.float().to(device),y.float().to(device)
                        
                        # Ensure input data is correctly shaped
                        B, T, D = X.shape
                        imu = X[:, :, :len(imu_cols)].permute(0, 2, 1)  # [B, C, T]
                        # ä¿®æ­£TOFå’ŒTHMæ•°æ�®çš„ç´¢å¼•æ–¹å¼�ï¼Œç¡®ä¿�TOFå�ªä½¿ç”¨ç»Ÿè®¡ç‰¹å¾�
                        tof_start_idx = len(imu_cols)
                        tof_end_idx = len(imu_cols) + 84  # TOFç»Ÿè®¡ç‰¹å¾�ç»´åº¦ä¸º84
                        tof = X[:, :, tof_start_idx:tof_end_idx].permute(0, 2, 1)  # TOFç»Ÿè®¡ç‰¹å¾� (84 channels)
                        thm = X[:, :, -len(thm_cols):].permute(0, 2, 1)  # THMæ•°æ�®
                        
                        logits = model(imu, tof, thm)
                        loss = -torch.sum(F.log_softmax(logits, dim=1) * y, dim=1).mean()  # ä¿®æ”¹ä¸ºä¸�è®­ç»ƒä¸€è‡´çš„æ�Ÿå¤±è®¡ç®—
                        val_loss += loss.item()
                        val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                        val_targets.extend(y.argmax(dim=1).cpu().numpy())
                        t.set_postfix(loss=val_loss/len(val_loader))

            val_loss /= len(val_loader)  # ç§»åˆ°å¾ªç�¯å¤–

            # Convert numeric predictions to class labels for competition metric
            train_preds_labels = [le.classes_[pred] for pred in train_preds]
            train_targets_labels = [le.classes_[target] for target in train_targets]
            val_preds_labels = [le.classes_[pred] for pred in val_preds]
            val_targets_labels = [le.classes_[target] for target in val_targets]
            
            # Calculate competition scores using only kaggle metric
            train_f1 = get_competition_score(train_targets_labels, train_preds_labels)
            val_f1 = get_competition_score(val_targets_labels, val_preds_labels)

            print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {train_loss/len(train_loader):.4f} - Train F1: {train_f1:.4f} - Val Loss: {val_loss:.4f} - Val F1: {val_f1:.4f}")

        model.eval()
        with torch.inference_mode():
            val_preds = []
            val_targets = []
            for X, y in (val_loader):  
                half = BATCH_SIZE // 2         

                x_front = X[:half]               
                x_back  = X[half:].clone()      
                
                x_back[:, :, 7:] = 0.0    
                X = torch.cat([x_front, x_back], dim=0)  # (B, C, T)
                X, y = X.float().to(device), y.to(device)
                
                # Ensure input data is correctly shaped
                B, T, D = X.shape
                imu = X[:, :, :len(imu_cols)].permute(0, 2, 1)  # [B, C, T]
                # ä¿®æ­£TOFå’ŒTHMæ•°æ�®çš„ç´¢å¼•æ–¹å¼�ï¼Œç¡®ä¿�TOFå�ªä½¿ç”¨ç»Ÿè®¡ç‰¹å¾�
                tof_start_idx = len(imu_cols)
                tof_end_idx = len(imu_cols) + 84  # TOFç»Ÿè®¡ç‰¹å¾�ç»´åº¦ä¸º84
                tof = X[:, :, tof_start_idx:tof_end_idx].permute(0, 2, 1)  # TOFç»Ÿè®¡ç‰¹å¾� (84 channels)
                thm = X[:, :, -len(thm_cols):].permute(0, 2, 1)  # THMæ•°æ�®
                
                logits = model(imu, tof, thm)
                val_preds.extend(logits.argmax(dim=1).cpu().numpy())
                val_targets.extend(y.argmax(dim=1).cpu().numpy())
                
                loss = F.cross_entropy(logits, y)
                val_loss += loss.item()
    
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
        models.append(model)
        # Save model
        torch.save({
            'model_state_dict': model.state_dict(),
            'imu_dim': len(imu_cols),
            'tof_dim': len(tof_cols),
            'thm_dim': len(thm_cols),  # Save THM dimension
            'n_classes': len(le.classes_),
            'pad_len': pad_len
        }, EXPORT_DIR / f"gesture_two_branch_fold{fold}.pth")
        # è®¡ç®—éªŒè¯�æŒ‡æ ‡
        val_preds = [le.classes_[pred] for pred in val_preds]
        val_targets = [le.classes_[target] for target in val_targets]
        val_f1 = get_competition_score(val_targets, val_preds)
        
        # Print validation accuracy and F1 score for the current fold
        print(f"fold: {fold} val_f1: {val_f1:.4f}")
        print(f"âœ” Training done - Final F1 Score: {val_f1:.4f} - artefacts saved in {EXPORT_DIR}")

else:
    print("â–¶ INFERENCE MODE â€“ loading artefacts from", PRETRAINED_DIR)

    # 1) å…ˆåŠ è½½é…�å¥—æ–‡ä»¶ï¼ˆfeature_cols, pad_len, scaler, gesture_classesï¼‰
    feature_cols_path = PRETRAINED_DIR / "feature_cols.npy"
    pad_len_path = PRETRAINED_DIR / "sequence_maxlen.npy"
    scaler_path = PRETRAINED_DIR / "scaler.pkl"
    gesture_classes_path = PRETRAINED_DIR / "gesture_classes.npy"

    # æ›´è¯¦ç»†çš„æ–‡ä»¶æ£€æŸ¥å’Œé”™è¯¯ä¿¡æ�¯
    missing_files = []
    if not feature_cols_path.exists():
        missing_files.append("feature_cols.npy")
    if not pad_len_path.exists():
        missing_files.append("sequence_maxlen.npy")
    if not scaler_path.exists():
        missing_files.append("scaler.pkl")
    if not gesture_classes_path.exists():
        missing_files.append("gesture_classes.npy")
    
    if missing_files:
        print(f"â�Œ Missing required files: {missing_files}")
        print(f"ğŸ“� Available files in {PRETRAINED_DIR}:")
        try:
            available_files = list(PRETRAINED_DIR.glob("*"))
            for f in available_files:
                print(f"   - {f.name}")
        except:
            print("   Could not list directory contents")
        
        raise FileNotFoundError(f"Required files not found in {PRETRAINED_DIR}: {missing_files}")
    
    print("âœ… All required files found")

    feature_cols = np.load(feature_cols_path, allow_pickle=True).tolist()
    pad_len = int(np.load(pad_len_path))
    scaler = joblib.load(scaler_path)
    gesture_classes = np.load(gesture_classes_path, allow_pickle=True)

    imu_cols = [c for c in feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
    tof_cols = [c for c in feature_cols if c.startswith('tof_')]
    thm_cols = [c for c in feature_cols if c.startswith('thm_')]

    # 2) åŠ è½½ 5 æŠ˜æ¨¡å�‹ï¼Œä½†ä¸�è¦�ä¿¡ä»» checkpoint['tof_dim'] å­—æ®µ â€”â€” ä»� state_dict æ�¨æ–­å®�é™…çš„ tof_dim
    MODELS = [f'gesture_two_branch_fold{i}.pth' for i in range(5)]
    models = []
    inferred_tof_dim = None
    for path in MODELS:
        ckpt_path = PRETRAINED_DIR / path
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")

        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # å°�è¯•ä»� state_dict è¯»å�– conv çš„ in_channelsï¼ˆä¼˜å…ˆï¼‰
        tof_dim_from_state = None
        # å¸¸è§� key å��ï¼š 'tof_branch.0.weight'ï¼ˆä¸�ä½ æ¨¡å�‹å®šä¹‰ä¸­ç¬¬ä¸€å±‚å�·ç§¯ä½�ç½®ä¸€è‡´ï¼‰
        possible_keys = [
            'tof_branch.0.weight',
            'tof_branch.0.0.weight',  # å¦‚æ�œæ˜¯ Sequential åŒ…è£¹çš„æƒ…å†µï¼ˆä¿�é™©èµ·è§�ï¼‰
            'module.tof_branch.0.weight',
            'module.tof_branch.0.0.weight'
        ]
        for k in possible_keys:
            if k in state_dict:
                w = state_dict[k]
                # w shape: (out_channels, in_channels, kernel_size)
                tof_dim_from_state = int(w.shape[1])
                break

        # å¦‚æ�œæ²¡æœ‰åœ¨ state_dict ä¸­å�‘ç�°ï¼Œå›�é€€åˆ° checkpoint å­—æ®µæˆ– len(tof_cols)ï¼ˆæœ€å��çš„å��å¤‡ï¼‰
        if tof_dim_from_state is None:
            tof_dim_from_state = checkpoint.get('tof_dim', None)
            if tof_dim_from_state is None:
                tof_dim_from_state = len(tof_cols)  # æœ€å��é€€å›�ï¼ˆä½†é€šå¸¸ä¸�æ˜¯æœŸæœ›çš„ï¼‰

        # è®°å½•ç¬¬ä¸€æ¬¡æ�¨æ–­åˆ°çš„ tof_dimï¼Œå¹¶ä½œä¸ºå…¨å±€åˆ‡ç‰‡å®½åº¦
        if inferred_tof_dim is None:
            inferred_tof_dim = int(tof_dim_from_state)
        else:
            # å¦‚æ�œå��ç»­æŸ�æŠ˜å’Œç¬¬ä¸€æŠ˜ä¸�ä¸€è‡´ï¼ŒæŠ›å‡ºè­¦å‘Šï¼ˆé€šå¸¸ä¸�åº”å�‘ç”Ÿï¼‰
            if int(tof_dim_from_state) != inferred_tof_dim:
                print(f"Warning: inferred_tof_dim inconsistent across folds: {inferred_tof_dim} vs {tof_dim_from_state}")

        # imu_dim, thm_dim ä¼˜å…ˆä½¿ç”¨ checkpoint å­—æ®µï¼ˆæœ‰ä¿�å­˜çš„è¯�ï¼‰
        imu_dim = int(checkpoint.get('imu_dim', len(imu_cols)))
        thm_dim = int(checkpoint.get('thm_dim', len(thm_cols)))
        n_classes = int(checkpoint.get('n_classes', None))  # å¿…é¡»æœ‰

        if n_classes is None:
            # å°�è¯•ä»� classifier æ�ƒé‡�æ�¨æ–­ç±»åˆ«æ•°
            cls_keys = ['classifier.weight', 'classifier.0.weight', 'module.classifier.weight']
            n_classes_infer = None
            for k in cls_keys:
                if k in state_dict:
                    n_classes_infer = int(state_dict[k].shape[0])
                    break
            if n_classes_infer is None:
                raise RuntimeError("Cannot determine n_classes from checkpoint; please ensure checkpoint includes 'n_classes'")
            n_classes = n_classes_infer

        # æ�„å»ºæ¨¡å�‹ï¼šä½¿ç”¨ä»� state_dict æ�¨æ–­åˆ°çš„ tof_dimï¼ˆinferred_tof_dimï¼‰
        model = MultiModalModel(
            imu_dim=imu_dim,
            tof_dim=inferred_tof_dim,
            thm_dim=thm_dim,
            d_model=64,
            n_classes=n_classes
        ).to(device)

        # åŠ è½½æ�ƒé‡�ï¼ˆstrict=Trueï¼‰ï¼Œè‹¥æŠ¥é”™å�¯å°�è¯• strict=False å¹¶æ‰“å�°å·®å¼‚ï¼ˆä½†ä¸¥æ ¼åŠ è½½æ›´å®‰å…¨ï¼‰
        try:
            model.load_state_dict(state_dict)
        except RuntimeError as e:
            # å°�è¯•å¤„ç�† DataParallel å‰�ç¼€å·®å¼‚
            # ç§»é™¤ possible 'module.' å‰�ç¼€å†�è¯•
            new_state = {}
            for k, v in state_dict.items():
                new_key = k
                if k.startswith('module.'):
                    new_key = k[len('module.'):]
                new_state[new_key] = v
            model.load_state_dict(new_state)  # è‹¥ä»�æŠ¥é”™ï¼Œæ­¤å¤„ä¼šæŠ›å¼‚å¸¸
        model.eval()
        models.append(model)

    if inferred_tof_dim is None:
        raise RuntimeError("Failed to infer TOF input channels from checkpoints; please check checkpoints")

    TOF_DIM = int(inferred_tof_dim)
    print(f"  Loaded {len(models)} models. Inferred TOF_DIM = {TOF_DIM}")
    print("  model, scaler, pads loaded â€“ ready for evaluation")

# Make sure gesture_classes exists in both modes
if TRAIN:
    gesture_classes = le.classes_
else:
    # åœ¨æ�¨ç�†æ¨¡å¼�ä¸‹åŠ è½½ gesture_classes
    gesture_classes = np.load(gesture_classes_path, allow_pickle=True)

# ================================
# predict å‡½æ•°ï¼ˆå�ªç”¨ tta_jitterï¼‰
# ================================
# predict function for Kaggle API
# ================================



# è¯»å�–æµ‹è¯•æ•°æ�®
print("â–¶ Loading test data...")
try:
    test_df = pd.read_csv(RAW_DIR / "test.csv")
    print(f"âœ… Loaded test data: {test_df.shape}")
except Exception as e:
    print(f"â�Œ ERROR loading test data: {e}")
    # å°�è¯•å…¶ä»–å�¯èƒ½çš„è·¯å¾„
    try:
        test_df = pd.read_csv("test.csv")
        print(f"âœ… Loaded test data from current directory: {test_df.shape}")
    except Exception as e2:
        print(f"â�Œ ERROR loading test data from current directory: {e2}")
        raise

# ================================
# Kaggle competition interface
# ================================
import os
import kaggle_evaluation.cmi_inference_server

def predict(sequence_data, demographics=None):
    """
    Predict a single sequence - this is what the Kaggle API expects
    """
    try:
        # ç¡®ä¿�è¾“å…¥æ•°æ�®ç±»å�‹æ­£ç¡®
        if hasattr(sequence_data, 'to_pandas'):
            # å¦‚æ�œæ˜¯polars DataFrameï¼Œè½¬æ�¢ä¸ºpandas
            sequence_data = sequence_data.to_pandas()
        elif not isinstance(sequence_data, pd.DataFrame):
            # å¦‚æ�œä¸�æ˜¯DataFrameï¼Œå°�è¯•è½¬æ�¢
            print(f"Warning: Unexpected data type: {type(sequence_data)}")
            if isinstance(sequence_data, (list, tuple)):
                sequence_data = pd.DataFrame(sequence_data)
            else:
                raise ValueError(f"Unsupported data type: {type(sequence_data)}")
        
        # é¢„å¤„ç�†å�•ä¸ªåº�åˆ—æ•°æ�®
        mat = preprocess_sequence(sequence_data, feature_cols, scaler)
        print(f"ğŸ”� Debug - mat shape: {mat.shape}")
        
        # å¡«å……åº�åˆ—
        mat_padded = pad_sequences_torch([mat], maxlen=pad_len, padding='pre', truncating='pre')[0]
        print(f"ğŸ”� Debug - mat_padded shape: {mat_padded.shape}")
        print(f"ğŸ”� Debug - imu_cols: {len(imu_cols)}, tof_cols: {len(tof_cols)}, thm_cols: {len(thm_cols)}")
        
        # æ��å�–TOFç»Ÿè®¡ç‰¹å¾�
        # æ­£ç¡®çš„åˆ‡ç‰‡ï¼šä»�ç¬¬7ä¸ªç‰¹å¾�å¼€å§‹ï¼Œå�–320ä¸ªTOFç‰¹å¾�
        tof_data = mat_padded[:, len(imu_cols):len(imu_cols)+len(tof_cols)]
        print(f"ğŸ”� Debug - tof_data shape before reshape: {tof_data.shape}")
        
        # ç¡®ä¿�TOFæ•°æ�®æ˜¯æ­£ç¡®çš„å½¢çŠ¶
        if len(tof_data.shape) == 1:
            # å¦‚æ�œæ˜¯ä¸€ç»´æ•°ç»„ï¼Œéœ€è¦�reshapeä¸º(1, 1, D)
            tof_data = tof_data.reshape(1, 1, len(tof_cols))
        else:
            # å¦‚æ�œæ˜¯äºŒç»´æ•°ç»„ï¼Œéœ€è¦�reshapeä¸º(1, T, D)
            tof_data = tof_data.reshape(1, tof_data.shape[0], len(tof_cols))
        
        print(f"ğŸ”� Debug - tof_data shape after reshape: {tof_data.shape}")
        tof_stats = extract_tof_statistics(tof_data, tof_cols)[0]
        print(f"ğŸ”� Debug - tof_stats shape: {tof_stats.shape}")
        
        # ç»„å�ˆç‰¹å¾�ï¼šIMU + TOFç»Ÿè®¡ + THM
        imu_data = mat_padded[:, :len(imu_cols)]
        thm_data = mat_padded[:, len(imu_cols)+len(tof_cols):]
        print(f"ğŸ”� Debug - imu_data shape: {imu_data.shape}, thm_data shape: {thm_data.shape}")
        
        other_data = np.concatenate([imu_data, thm_data], axis=1)
        print(f"ğŸ”� Debug - other_data shape: {other_data.shape}")
        
        # ç¡®ä¿�TOFç»Ÿè®¡ç‰¹å¾�çš„æ—¶é—´ç»´åº¦ä¸�IMU+THMä¸€è‡´
        if tof_stats.shape[0] != other_data.shape[0]:
            # å¦‚æ�œæ—¶é—´ç»´åº¦ä¸�åŒ¹é…�ï¼Œéœ€è¦�è°ƒæ•´TOFç»Ÿè®¡ç‰¹å¾�
            if tof_stats.shape[0] == 1:
                # TOFç»Ÿè®¡æ˜¯å…¨å±€ç‰¹å¾�ï¼Œéœ€è¦�æ‰©å±•åˆ°æ¯�ä¸ªæ—¶é—´æ­¥
                tof_stats = np.tile(tof_stats, (other_data.shape[0], 1))
            else:
                # æˆªæ–­æˆ–å¡«å……åˆ°ç›¸å�Œé•¿åº¦
                target_length = other_data.shape[0]
                if tof_stats.shape[0] > target_length:
                    tof_stats = tof_stats[:target_length]
                else:
                    # å¡«å……åˆ°ç›®æ ‡é•¿åº¦
                    padding = np.zeros((target_length - tof_stats.shape[0], tof_stats.shape[1]))
                    tof_stats = np.vstack([tof_stats, padding])
        
        enhanced_data = np.concatenate([other_data, tof_stats], axis=1)
        print(f"ğŸ”� Debug - enhanced_data shape: {enhanced_data.shape}")
        
        # ä½¿ç”¨æ¨¡å�‹é¢„æµ‹
        pred_gesture = predict_ensemble(enhanced_data)
        return pred_gesture
        
    except Exception as e:
        print(f"Error in predict function: {e}")
        print(f"Input data type: {type(sequence_data)}")
        if hasattr(sequence_data, 'shape'):
            print(f"Input data shape: {sequence_data.shape}")
        if hasattr(sequence_data, 'columns'):
            print(f"Input data columns: {sequence_data.columns}")
        raise

# é‡�å‘½å��å�Ÿæ�¥çš„predictå‡½æ•°ä¸ºpredict_ensemble
def predict_ensemble(sequence, demographics: pl.DataFrame = None, n_tta: int = 8) -> str:
    global gesture_classes, models, TOF_DIM, feature_cols, pad_len, scaler, imu_cols, thm_cols

    if gesture_classes is None:
        gesture_classes = np.load(
            PRETRAINED_DIR / "gesture_classes.npy",
            allow_pickle=True
        )

    # 1. å¤„ç�†è¾“å…¥æ•°æ�®
    if isinstance(sequence, pl.DataFrame):
        # å¦‚æ�œæ˜¯ polars DataFrameï¼Œéœ€è¦�é‡�æ–°é¢„å¤„ç�†
        df_seq = sequence.to_pandas()
        mat = preprocess_sequence(df_seq, feature_cols, scaler)
    else:
        # å¦‚æ�œå·²ç»�æ˜¯é¢„å¤„ç�†è¿‡çš„ numpy æ•°ç»„ï¼Œç›´æ�¥ä½¿ç”¨
        mat = sequence

    # 2. ä»…ä½¿ç”¨ tta_jitter
    tta_samples = []
    for _ in range(n_tta):
        x_aug = tta_jitter(mat.copy(), sigma=0.01)
        tta_samples.append(x_aug)

    # 3. Pad å¹¶å †å� æˆ� Tensor
    pads = pad_sequences_torch(
        tta_samples,
        maxlen=pad_len,
        padding='pre',
        truncating='pre'
    )
    X = torch.FloatTensor(pads).to(device)  # (n_tta, T, C)

    # 4. å¤šæ¨¡å�‹ ensemble + TTA é¢„æµ‹
    with torch.no_grad():
        probs_accum = []
        for m in models:
            # æ•°æ�®å·²ç»�æ˜¯é¢„å¤„ç�†è¿‡çš„æ ¼å¼�ï¼šIMU + TOFç»Ÿè®¡ + THM
            # æ ¹æ�®å®�é™…çš„æ•°æ�®ç»´åº¦è¿›è¡Œåˆ‡ç‰‡
            total_features = X.shape[2]
            imu_features = len(imu_cols)
            thm_features = len(thm_cols)
            tof_features = total_features - imu_features - thm_features
            
            # åˆ‡ç‰‡æ•°æ�®
            imu = X[:, :, :imu_features].permute(0, 2, 1)  # [B, C, T]
            tof = X[:, :, imu_features:imu_features+tof_features].permute(0, 2, 1)  # TOFç»Ÿè®¡ç‰¹å¾�
            thm = X[:, :, -thm_features:].permute(0, 2, 1)  # THM æ•°æ�®

            # è°ƒè¯•ä¿¡æ�¯ï¼ˆå�ªåœ¨ç¬¬ä¸€æ¬¡è¿�è¡Œæ—¶æ‰“å�°ï¼‰
            if len(probs_accum) == 0:
                print(f"ğŸ”� Debug - Input shapes: IMU {imu.shape}, TOF {tof.shape}, THM {thm.shape}")

            logits = m(imu, tof, thm)               # (n_tta, n_classes)
            probs  = torch.softmax(logits, dim=1)
            probs_accum.append(probs)

        # æŒ‰æ¨¡å�‹å¹³å�‡ â†’ (n_tta, n_classes)
        avg_per_tta = torch.stack(probs_accum).mean(0)
        # å¯¹æ‰€æœ‰ TTA æ ·æœ¬å¹³å�‡ â†’ (n_classes,)
        final_prob = avg_per_tta.mean(0)
        idx = int(final_prob.argmax().cpu().numpy())

    return str(gesture_classes[idx])

# åˆ›å»ºæ�¨ç�†æœ�åŠ¡å™¨
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

# æ ¹æ�®ç�¯å¢ƒå�˜é‡�å†³å®šè¿�è¡Œæ¨¡å¼�
if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    # åœ¨Kaggleæ¯”èµ›ä¸­è¿�è¡Œ
    print("ğŸš€ Running in Kaggle competition mode...")
    inference_server.serve()
else:
    # æœ¬åœ°æµ‹è¯•æ¨¡å¼�
    print("ğŸ§ª Running in local test mode...")
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )


