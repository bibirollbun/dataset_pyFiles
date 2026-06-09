# Install required dependencies
!pip install polars numpy scipy



import gc
import kaggle_evaluation.cmi_inference_server
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import polars as pl
import plotly.express as px
import plotly.io as pio
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from scipy.fft import fft
from scipy.stats import skew, kurtosis, entropy



# Create output directory
!mkdir -p output
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pio.renderers.default = 'iframe'

class config:
    BATCH_SIZE_TEST = 16
    NUM_WORKERS = 2
    PRINT_FREQ = 20
    SEED = 20
    MAX_LENGTH = 200



class paths:
    OUTPUT_DIR = "/kaggle/working/output"
    TEST_CSV = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
    TEST_DEMOGRAPHICS = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
    TRAIN_CSV = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
    TRAIN_DEMOGRAPHICS = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"

def format_for_scoring(df_preds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]: 
    solution = df_preds[["sequence_id", "y_true"]].copy()
    solution.columns = ["id", "gesture"]
    solution["gesture"] = solution["gesture"].map(num_to_label)
    submission = df_preds[["sequence_id", "y_pred"]].copy()
    submission.columns = ["id", "gesture"]
    submission["gesture"] = submission["gesture"].map(num_to_label)
    return solution, submission

def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)



def sep():
    print("—"*100)

label_to_num = {
    'Above ear - pull hair': 0,
    'Cheek - pinch skin': 1,
    'Eyebrow - pull hair': 2,
    'Eyelash - pull hair': 3,
    'Forehead - pull hairline': 4,
    'Forehead - scratch': 5,
    'Neck - pinch skin': 6,
    'Neck - scratch': 7,
    'Drink from bottle/cup': 8,
    'Feel around in tray and pull out an object': 8,
    'Glasses on/off': 8,
    'Pinch knee/leg skin': 8,
    'Pull air toward your face': 8,
    'Scratch knee/leg skin': 8,
    'Text on phone': 8,
    'Wave hello': 8,
    'Write name in air': 8,
    'Write name on leg': 8
}
type_to_num = {"Target": 1, "Non-Target": 0}
num_to_label = {v: k for k, v in label_to_num.items()}
num_to_type = {v: k for k, v in type_to_num.items()}
seed_everything(config.SEED)

# Load data
df_test = pd.read_csv(paths.TEST_CSV)
df_test_demographics = pd.read_csv(paths.TEST_DEMOGRAPHICS)
print(f"Test dataframe shape: {df_test.shape}"), sep()
print(f"Test demographics dataframe shape: {df_test_demographics.shape}")
display(df_test.head())
display(df_test_demographics.head())



def robust_scale(arr: np.ndarray) -> np.ndarray:
    scaler = RobustScaler()
    arr = scaler.fit_transform(arr)
    arr = np.where(np.isnan(arr), 0.0, arr)
    return arr

def time_warp(arr: np.ndarray, sigma=0.2) -> np.ndarray:
    """Apply time warping augmentation"""
    n = arr.shape[0]
    time_points = np.linspace(0, 1, n)
    random_warps = np.random.normal(loc=1.0, scale=sigma, size=n)
    warped_time = np.cumsum(random_warps)
    warped_time = warped_time / warped_time[-1] * n
    indices = np.clip(np.round(warped_time).astype(int), 0, n-1)
    return arr[indices]

def jitter(arr: np.ndarray, noise_level=0.01) -> np.ndarray:
    """Add jittering augmentation"""
    noise = np.random.normal(0, noise_level, arr.shape)
    return arr + noise




def pad_or_truncate(arr: np.ndarray, max_length: int = config.MAX_LENGTH, pad_value: float = 0.0) -> np.ndarray:
    L, D = arr.shape
    if L > max_length:
        start_idx = np.random.randint(0, L - max_length + 1)
        return arr[start_idx:start_idx + max_length, :]
    elif L < max_length:
        pad_width = max_length - L
        padding = np.full((pad_width, D), pad_value)
        return np.vstack((arr, padding))
    return arr

def extract_freq_features(arr, max_length=config.MAX_LENGTH):
    """Extract FFT-based frequency features for all 7 channels"""
    freq_features = []
    for i in range(arr.shape[1]):  # For all 7 channels
        fft_result = np.abs(fft(arr[:, i]))
        freq_features.append(fft_result[:max_length//2].mean())  # Mean of first half of FFT
    return np.array(freq_features)

imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
X_test = []
for sequence_id in tqdm(df_test.sequence_id.unique()):
    ds = df_test[df_test["sequence_id"] == sequence_id]
    X = ds[imu_cols].values
    if np.random.rand() < 0.5:
        X = time_warp(X)
        X = jitter(X, noise_level=0.01)
    X = pad_or_truncate(X)
    X_scaled = X  # Use original 7 channels
    freq_feats = extract_freq_features(X_scaled, config.MAX_LENGTH)  # 7 features
    # Pad to 11 features to align with intended 18 channels
    freq_feats = np.pad(freq_feats, (0, 11 - len(freq_feats)), mode='constant')
    freq_feats_expanded = np.tile(freq_feats, (X_scaled.shape[0], 1))  # (200, 11)
    X_test.append(np.hstack((X_scaled, freq_feats_expanded)))  # (200, 18)

X_test = np.array(X_test)




class CustomDataset(Dataset):
    def __init__(self, config, df: pd.DataFrame, X: np.ndarray):
        self.config = config
        self.df = df
        self.X = X
        self.indexes = self.df.sequence_id.unique()
        
    def __len__(self):
        return len(self.indexes)
        
    def __getitem__(self, index):
        sequence_id = self.indexes[index]
        return {
            "X": torch.tensor(self.X[index], dtype=torch.float32),
            "sequence_id": sequence_id
        }

test_dataset = CustomDataset(config, df_test, X_test)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE_TEST,
    shuffle=False,
    num_workers=config.NUM_WORKERS,
    pin_memory=True,
    prefetch_factor=2,
    drop_last=False
)




class EnhancedSEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat([avg_y, max_y], dim=1)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)




class MultiScaleConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, ks, padding=ks//2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ) for ks in kernel_sizes
        ])
        
    def forward(self, x):
        outputs = [conv(x) for conv in self.convs]
        return torch.cat(outputs, dim=1)




class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = EnhancedSEBlock(out_channels, reduction=8)
        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels)
        ) if in_channels != out_channels else nn.Sequential()
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += shortcut
        out = F.relu(out)
        out = self.pool(out)
        out = self.dropout(out)
        return out





class MetaFeatureExtractor(nn.Module):
    def forward(self, x):
        # Ensure x is on the correct device
        x = x.to(device)
        mean = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        max_val, _ = torch.max(x, dim=1)
        min_val, _ = torch.min(x, dim=1)
        seq_len = x.size(1)
        slope = (x[:, -1, :] - x[:, 0, :]) / (seq_len - 1) if seq_len > 1 else torch.zeros_like(x[:, 0, :]).to(device)
        # Move to CPU for NumPy operations, then back to device
        x_cpu = x.cpu()
        skewness = torch.tensor(skew(x_cpu.numpy(), axis=1), dtype=torch.float32).to(device)
        kurt = torch.tensor(kurtosis(x_cpu.numpy(), axis=1), dtype=torch.float32).to(device)
        ent = torch.tensor([entropy(np.histogram(x_cpu[:, i].numpy(), bins=10)[0]) for i in range(x.size(2))], dtype=torch.float32).to(device).unsqueeze(0)
        return torch.cat([mean, std, max_val, min_val, slope, skewness, kurt, ent], dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads=4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x):
        batch, seq_len, hidden = x.size()
        q = self.query(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.key(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.value(x).view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(batch, seq_len, self.hidden_dim)
        return self.out(out)


class ModelVariant_GRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        num_channels = 18  # 7 IMU + 11 freq features
        self.meta_extractor = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(8 * num_channels, 64),  # 8 features per channel * 18 channels = 144 -> 64
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.branches = nn.ModuleList([
            nn.Sequential(
                MultiScaleConv1d(1, 12, kernel_sizes=[3, 5, 7]),  # Output: 36 channels
                ResidualSEBlock(36, 48, 3, dropout=0.3),         # Output: 48 channels
                ResidualSEBlock(48, 48, 3, dropout=0.3),         # Output: 48 channels
            ) for _ in range(num_channels)
        ])
        self.bigru = nn.GRU(
            input_size=48 * num_channels,  # 48 * 18 = 864
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.attention = MultiHeadAttention(hidden_dim=512, num_heads=4)  # Expects 512 input
        self.attention_pooling = nn.Linear(512, 1)  # Reduces to (batch_size, 1)
        self.head_1 = nn.Sequential(
            nn.Linear(512 + 64, 512),  # 512 (pooled) + 64 (meta) = 576
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, num_classes),
        )
        self.head_2 = nn.Sequential(
            nn.Linear(512 + 64, 512),  # 512 (pooled) + 64 (meta) = 576
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(512, 1),
        )

    def forward(self, x: torch.Tensor):
        x = x.to(device)
        meta = self.meta_extractor(x).to(device)
        meta_proj = self.meta_dense(meta)  # Shape: (batch_size, 64)
        branch_outputs = []
        for i in range(x.shape[2]):
            channel_input = x[:, :, i].unsqueeze(1).to(device)
            processed = self.branches[i](channel_input)  # Shape: (batch_size, 48, seq_len)
            branch_outputs.append(processed.transpose(1, 2))  # To (batch_size, seq_len, 48)
        combined = torch.cat(branch_outputs, dim=2).to(device)  # Shape: (batch_size, seq_len, 864)
        gru_out, _ = self.bigru(combined)  # Shape: (batch_size, seq_len, 512)
        attn_out = self.attention(gru_out)  # Shape: (batch_size, seq_len, 512)
        pooled_output = torch.tanh(self.attention_pooling(attn_out)).squeeze(-1)  # Shape: (batch_size, seq_len)
        # Ensure pooled_output is reduced to (batch_size, 512) by applying mean over seq_len
        pooled_output = torch.mean(attn_out, dim=1)  # Shape: (batch_size, 512)
        fused = torch.cat([pooled_output, meta_proj], dim=1)  # Shape: (batch_size, 576)
        z1 = self.head_1(fused)  # Shape: (batch_size, num_classes)
        z2 = self.head_2(fused)  # Shape: (batch_size, 1)
        return z1, z2


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_test = sequence.to_pandas()
    X_test = []
    
    for sequence_id in tqdm(df_test.sequence_id.unique()):
        ds = df_test[df_test["sequence_id"] == sequence_id]
        X = ds[imu_cols].values
        if np.random.rand() < 0.5:
            X = time_warp(X)
            X = jitter(X, noise_level=0.01)
        X = pad_or_truncate(X)
        X_scaled = X  # Use original 7 channels
        freq_feats = extract_freq_features(X_scaled, config.MAX_LENGTH)  # 7 features
        # Pad to 11 features to align with intended 18 channels
        freq_feats = np.pad(freq_feats, (0, 11 - len(freq_feats)), mode='constant')
        freq_feats_expanded = np.tile(freq_feats, (X_scaled.shape[0], 1))  # (200, 11)
        X_test.append(np.hstack((X_scaled, freq_feats_expanded)))  # (200, 18)
    
    X_test = np.array(X_test)
    test_dataset = CustomDataset(config, df_test, X_test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE_TEST,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2,
        drop_last=False
    )
    
    model_paths = glob("/kaggle/input/mixup-gru-cv0-7358/*.pth")[:5]
    all_preds = []
    model_weights = [1.0 / len(model_paths)] * len(model_paths)
    for model_path in model_paths:
        model = ModelVariant_GRU(num_classes=9)
        checkpoint = torch.load(model_path, map_location=device)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.to(device)
        model.eval()
        softmax = nn.Softmax(dim=1)
        
        mc_preds = []
        for _ in range(5):
            with tqdm(test_loader, unit="test_batch", desc='Test') as tqdm_test_loader:
                for batch in tqdm_test_loader:
                    X = batch["X"].to(device)
                    with torch.no_grad():
                        y_preds, y_preds_hard = model(X)
                    # Apply softmax on GPU, then move to CPU and convert to numpy
                    y_preds = softmax(y_preds).cpu().numpy()
                    mc_preds.append(y_preds)
        all_preds.append(np.mean(mc_preds, axis=0))
    
    weighted_preds = np.average(all_preds, axis=0, weights=model_weights)
    threshold = 0.6
    final_pred = np.argmax(weighted_preds, axis=1)[0] if weighted_preds[0].max() > threshold else 8
    prediction = num_to_label[final_pred]
    return prediction

# Launch inference server
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




