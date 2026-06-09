!pip install polars 
!pip install plotly


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
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

!mkdir output
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
pio.renderers.default = 'iframe'


class config:
    BATCH_SIZE_TEST = 1
    NUM_WORKERS = 0 # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20

class paths:
    OUTPUT_DIR = "/kaggle/working/output"
    TEST_CSV = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
    TEST_DEMOGRAPHICS = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
    TRAIN_CSV = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
    TRAIN_DEMOGRAPHICS = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"


def format_for_scoring(df_preds: pd.DataFrame) ->tuple[pd.DataFrame, pd.DataFrame]: 
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
    os.environ['PYTHONHASHSEED'] = str(seed) 
    

def sep():
    print("—"*100)


label_to_num = {
    'Above ear - pull hair': 0,  # < ------- TARGETS START
    'Cheek - pinch skin': 1,
    'Eyebrow - pull hair': 2,
    'Eyelash - pull hair': 3,
    'Forehead - pull hairline': 4,
    'Forehead - scratch': 5,
    'Neck - pinch skin': 6,
    'Neck - scratch': 7,  # < ------- TARGETS END
    'Drink from bottle/cup': 8,  # < ------- NON-TARGETS START
    'Feel around in tray and pull out an object': 8,
    'Glasses on/off': 8,
    'Pinch knee/leg skin': 8,
    'Pull air toward your face': 8,
    'Scratch knee/leg skin': 8,
    'Text on phone': 8,
    'Wave hello': 8,
    'Write name in air': 8,
    'Write name on leg': 8  # < ------- NON-TARGETS END
}
type_to_num = {"Target": 1, "Non-Target":0}
num_to_label = {v: k for k, v in label_to_num.items()}
num_to_type = {v: k for k, v in type_to_num.items()}
seed_everything(config.SEED)



df_test = pd.read_csv(paths.TEST_CSV)
df_test_demographics = pd.read_csv(paths.TEST_DEMOGRAPHICS)

print(f"Test dataframe shape: {df_test.shape}"), sep()
print(f"Test demographics dataframe shape: {df_test_demographics.shape}")

display(df_test.head())
display(df_test_demographics.head())


def min_max_scale(arr: np.ndarray) -> np.ndarray:
    min_vals = np.nanmin(arr, axis=0)
    max_vals = np.nanmax(arr, axis=0)
    ranges = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
    scaled = (arr - min_vals) / ranges
    return scaled


def standard_scale(arr: np.ndarray) -> np.ndarray:
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    stds = np.where(stds == 0, 1, stds)  # Prevent division by zero for constant columns
    scaled = (arr - means) / stds
    return scaled


def pad_or_truncate(
    arr: np.ndarray,
    max_length: int = 200,
    pad_value: int = 0,
    mode: str = "random"  # "regular" or "random"
) -> np.ndarray:
    L, D = arr.shape

    if L > max_length:
        return arr[:max_length, :]

    elif L < max_length:
        if mode == "regular":
            padding = np.full((max_length - L, D), pad_value)
            return np.vstack((arr, padding))
        
        elif mode == "random":
            total_padding = max_length - L
            pad_start = np.random.randint(0, total_padding + 1)
            pad_end = total_padding - pad_start

            start_padding = np.full((pad_start, D), pad_value)
            end_padding = np.full((pad_end, D), pad_value)

            return np.vstack((start_padding, arr, end_padding))
        
        else:
            raise ValueError(f"Unknown mode: {mode}. Use 'regular' or 'random'.")

    else:
        return arr

imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
X_test  = []

for sequence_id in tqdm(df_test.sequence_id.unique()):
    ds = df_test[df_test["sequence_id"] == sequence_id]
    X = ds[imu_cols].values
    X = pad_or_truncate(X)
    X = np.concatenate((standard_scale(X[:, 0:3]), X[:, 3:]), axis=1)
    X = np.where(np.isnan(X), 0.0, X)  # fill NaNs
    X_test.append(X)

X_test = np.array(X_test)


class CustomDataset(Dataset):
    def __init__(
        self, config, df: pd.DataFrame, X: np.ndarray
    ): 
        
        self.config = config
        self.df = df
        self.X = X
        self.indexes = self.df.sequence_id.unique()
        
    def __len__(self):
        """
        Length of dataset.
        """
        return len(self.indexes)
        
    def __getitem__(self, index):
        """
        Get one item.
        """
        sequence_id = self.indexes[index]
        X = self.X[index]
        output = {
            "X": torch.tensor(X, dtype=torch.float32),
            "sequence_id": sequence_id
        }
        return output 


test_dataset = CustomDataset(config, df_test, X_test)
test_loader = DataLoader(
    test_dataset,
    batch_size=config.BATCH_SIZE_TEST,
    shuffle=False,
    num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True
)
idx = np.random.choice(range(0, len(test_dataset)))
cols = imu_cols
X = test_dataset[idx]["X"]
sequence_id = test_dataset[idx]["sequence_id"]
N = X.shape[0]
df = pd.DataFrame(X, columns=cols)
df['step'] = range(N)
df_melted = df.melt(id_vars='step', var_name='sequence', value_name='value')

fig = px.line(
    df_melted,
    x='step',
    y='value',
    color='sequence',
    title=f"Sequences for {sequence_id}",
    template='plotly_dark',
    width=800,
    height=600
)

fig.show()


import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.SiLU(inplace=True),  # Using SiLU (swish) as in TF reference
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
    """Multi-scale temporal convolution block"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[3, 5, 7]):
        super().__init__()
        self.convs = nn.ModuleList()
        for ks in kernel_sizes:
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, ks, padding=ks//2, bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ))
        
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
        
        # Use the new EnhancedSEBlock
        self.se = EnhancedSEBlock(out_channels, reduction=8)
        
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
        # x shape: (B, L, C)
        mean = torch.mean(x, dim=1)
        std = torch.std(x, dim=1)
        max_val, _ = torch.max(x, dim=1)
        min_val, _ = torch.min(x, dim=1)
        
        # Calculate slope: (last - first) / seq_len
        seq_len = x.size(1)
        if seq_len > 1:
            slope = (x[:, -1, :] - x[:, 0, :]) / (seq_len - 1)
        else:
            slope = torch.zeros_like(x[:, 0, :])
        
        return torch.cat([mean, std, max_val, min_val, slope], dim=1)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))
        weights = F.softmax(scores.squeeze(-1), dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context

class CustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        num_channels = 7 # Hardcoded for IMU data which has 7 channels
        
        # Meta feature extractor
        self.meta_extractor = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * num_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Create multiple parallel branches for IMU channels
        self.branches = nn.ModuleList([
            nn.Sequential(
                MultiScaleConv1d(1, 16, kernel_sizes=[3, 5, 7]), # out: 16*3=48 channels
                ResidualSEBlock(48, 64, 3, dropout=0.3),
                ResidualSEBlock(64, 64, 3, dropout=0.3)
            ) for _ in range(num_channels)
        ])
        
        # Bidirectional LSTM layer
        self.bilstm = nn.LSTM(
            input_size=64 * num_channels,
            hidden_size=128,
            num_layers=2, # Increased layers for more capacity
            batch_first=True,
            bidirectional=True,
            dropout=0.2 # Add regularization
        )
        
        # Multi-Head Attention layer after BiLSTM for enhanced temporal feature extraction
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=256,  # 2 * hidden_size for bidirectional
            num_heads=8,
            dropout=0.2,
            batch_first=True
        )
        
        # Attention pooling layer to aggregate sequence information
        self.attention_pooling = AttentionLayer(256)
        
        # Output heads with meta feature injection
        self.head_1 = nn.Sequential(
            nn.Linear(256 + 32, 512),  # Attention pooled output + meta features
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
        self.head_2 = nn.Sequential(
            nn.Linear(256 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # For regression task
        )
        
    def forward(self, x: torch.Tensor):
        
        # Extract and process meta features
        meta = self.meta_extractor(x)
        meta_proj = self.meta_dense(meta)
        
        branch_outputs = []
        for i in range(x.shape[2]):
            # Process each channel through its dedicated branch
            channel_input = x[:, :, i].unsqueeze(1)  # (B, 1, L)
            processed = self.branches[i](channel_input)      # (B, 64, L/k)
            branch_outputs.append(processed.transpose(1, 2)) # (B, L/k, 64)
        
        # Concatenate all branch outputs along feature dimension
        combined = torch.cat(branch_outputs, dim=2)  # (B, L/k, 64*C)
        
        # Process through BiLSTM
        bilstm_out, _ = self.bilstm(combined)  # (B, L/k, 256)
        
        # Apply Multi-Head Attention
        attn_output, _ = self.multihead_attn(bilstm_out, bilstm_out, bilstm_out)
        
        # Add residual connection for stability
        attended_bilstm = bilstm_out + attn_output

        # Apply attention pooling to get a fixed-size vector
        pooled_output = self.attention_pooling(attended_bilstm)  # (B, 256)
        
        # Combine with meta features
        fused = torch.cat([pooled_output, meta_proj], dim=1)  # (B, 256 + 32)
        
        # Final predictions
        z1 = self.head_1(fused)  # Classification
        z2 = self.head_2(fused)  # Regression
        return z1, z2


import torch
import torch.nn as nn
import torch.nn.functional as F


# EnhancedSEBlock, MetaFeatureExtractor, AttentionLayer


class ModelVariant_GRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        num_channels = 7  # Hardcoded for IMU data

        # 1. 
        self.meta_extractor = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * num_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 2. 
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    MultiScaleConv1d(1, 12, kernel_sizes=[3, 5, 7]),
                    ResidualSEBlock(36, 48, 3, dropout=0.3),
                    ResidualSEBlock(48, 48, 3, dropout=0.3),
                )
                for _ in range(num_channels)
            ]
        )

        # 3. MultiHeadAttention
        self.bigru = nn.GRU(
            input_size=48 * num_channels,
            hidden_size=128,  # hidden_size
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # 4. Attention Pooling
        self.attention_pooling = AttentionLayer(256)  # 128 * 2 for bidirectional

        # 5. 
        self.head_1 = nn.Sequential(
            nn.Linear(256 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

        self.head_2 = nn.Sequential(
            nn.Linear(256 + 32, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),  # For regression task
        )

    def forward(self, x: torch.Tensor):
        # Meta features
        meta = self.meta_extractor(x)
        meta_proj = self.meta_dense(meta)

        # CNN branches
        branch_outputs = []
        for i in range(x.shape[2]):
            channel_input = x[:, :, i].unsqueeze(1)
            processed = self.branches[i](channel_input)
            branch_outputs.append(processed.transpose(1, 2))

        combined = torch.cat(branch_outputs, dim=2)

        # BiGRU processing
        gru_out, _ = self.bigru(combined)  # (B, L/k, 256)

        # Attention pooling
        pooled_output = self.attention_pooling(gru_out)  # (B, 256)

        # Combine with meta features
        fused = torch.cat([pooled_output, meta_proj], dim=1)

        # Final predictions
        z1 = self.head_1(fused)
        z2 = self.head_2(fused)
        return z1, z2


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_test = sequence.to_pandas()
    imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
    X_test, y_test, y_hard_test = [], [], []
    
    for sequence_id in tqdm(df_test.sequence_id.unique()):
        ds = df_test[df_test["sequence_id"] == sequence_id]
        X = ds[imu_cols].values
        X = pad_or_truncate(X)
        X = np.concatenate((standard_scale(X[:, 0:3]), X[:, 3:]), axis=1)
        X = np.where(np.isnan(X), 0.0, X)  # fill NaNs
        X_test.append(X)

    X_test = np.array(X_test)
    model_paths = glob("/kaggle/input/mixup-gru-cv0-7358/*.pth")
    all_preds = []
    for model_path in model_paths:
        test_dataset = valid_dataset = CustomDataset(config, df_test, X_test)
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE_TEST,
            shuffle=False,
            num_workers=config.NUM_WORKERS, 
            pin_memory=True, drop_last=False
        )
        model = ModelVariant_GRU(num_classes=9)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint)
        model.to(device)
        model.eval()
        softmax = nn.Softmax(dim=1)
        
        with tqdm(test_loader, unit="test_batch", desc='Test') as tqdm_test_loader:
            for step, batch in enumerate(tqdm_test_loader):
                X = batch.pop("X").to(device)
                batch_size = X.size(0)
                with torch.no_grad():
                    y_preds, y_preds_hard = model(X)
                y_preds = softmax(y_preds).to('cpu').numpy()
                all_preds.append(y_preds)
    all_preds = np.concatenate(all_preds)
    all_preds = np.argmax(all_preds.mean(axis=0)).item()
    prediction = num_to_label[all_preds]
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

