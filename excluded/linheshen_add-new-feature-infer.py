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
    BATCH_SIZE_TEST = 32
    NUM_WORKERS = 4 # multiprocessing.cpu_count()
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
    'Feel around in tray and pull out an object': 9,
    'Glasses on/off': 10,
    'Pinch knee/leg skin': 11,
    'Pull air toward your face': 12,
    'Scratch knee/leg skin': 13,
    'Text on phone': 14,
    'Wave hello': 15,
    'Write name in air': 16,
    'Write name on leg': 17  # < ------- NON-TARGETS END
}
type_to_num = {"Target": 1, "Non-Target":0}
num_to_label = {v: k for k, v in label_to_num.items()}
num_to_type = {v: k for k, v in type_to_num.items()}
seed_everything(config.SEED)
df_test = pd.read_csv(paths.TEST_CSV)
df_test_demographics = pd.read_csv(paths.TEST_DEMOGRAPHICS)


from scipy.spatial.transform import Rotation as R

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

            # Calculate the relative rotation
            delta_rot = rot_t.inv() * rot_t_plus_dt
            
            # Convert delta rotation to angular velocity vector
            # The rotation vector (Euler axis * angle) scaled by 1/dt
            # is a good approximation for small delta_rot
            angular_vel[i, :] = delta_rot.as_rotvec() / time_delta
        except ValueError:
            # If quaternion is invalid, angular velocity remains zero
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
            angular_dist[i] = 0 # Или np.nan, в зависимости от желаемого поведения
            continue
        try:
            # Преобразование кватернионов в объекты Rotation
            r1 = R.from_quat(q1)
            r2 = R.from_quat(q2)

            # Вычисление углового расстояния: 2 * arccos(|real(p * q*)|)
            # где p* - сопряженный кватернион q
            # В scipy.spatial.transform.Rotation, r1.inv() * r2 дает относительное вращение.
            # Угол этого относительного вращения - это и есть угловое расстояние.
            relative_rotation = r1.inv() * r2
            
            # Угол rotation vector соответствует угловому расстоянию
            # Норма rotation vector - это угол в радианах
            angle = np.linalg.norm(relative_rotation.as_rotvec())
            angular_dist[i] = angle
        except ValueError:
            angular_dist[i] = 0 # В случае недействительных кватернионов
            pass
            
    return angular_dist
def feature_engineering(train_df):
    # IMU magnitude
    train_df['acc_mag'] = np.sqrt(train_df['acc_x']**2 + train_df['acc_y']**2 + train_df['acc_z']**2)
    
    # IMU angle
    train_df['rot_angle'] = 2 * np.arccos(train_df['rot_w'].clip(-1, 1))
    
    # IMU jerk, angular velocity
    train_df['acc_mag_jerk'] = train_df.groupby('sequence_id')['acc_mag'].diff().fillna(0)
    train_df['rot_angle_vel'] = train_df.groupby('sequence_id')['rot_angle'].diff().fillna(0)
    
    # Remove gravity
    def get_linear_accel(df):
        res = remove_gravity_from_acc(
            df[['acc_x', 'acc_y', 'acc_z']],
            df[['rot_x', 'rot_y', 'rot_z', 'rot_w']]
        )
        res = pd.DataFrame(res, columns=['linear_acc_x', 'linear_acc_y', 'linear_acc_z'], index=df.index)
        return res
    
    linear_accel_df = train_df.groupby('sequence_id').apply(get_linear_accel, include_groups=False)
    linear_accel_df = linear_accel_df.droplevel('sequence_id')
    train_df = train_df.join(linear_accel_df)
    
    train_df['linear_acc_mag'] = np.sqrt(train_df['linear_acc_x']**2 + train_df['linear_acc_y']**2 + train_df['linear_acc_z']**2)
    train_df['linear_acc_mag_jerk'] = train_df.groupby('sequence_id')['linear_acc_mag'].diff().fillna(0)

    # Calc angular velocity
    def calc_angular_velocity(df):
        res = calculate_angular_velocity_from_quat( df[['rot_x', 'rot_y', 'rot_z', 'rot_w']] )
        res = pd.DataFrame(res, columns=['angular_vel_x', 'angular_vel_y', 'angular_vel_z'], index=df.index)
        return res
    
    angular_velocity_df = train_df.groupby('sequence_id').apply(calc_angular_velocity, include_groups=False)
    angular_velocity_df = angular_velocity_df.droplevel('sequence_id')
    train_df = train_df.join(angular_velocity_df)

    # Calculating angular distance
    def calc_angular_distance(df):
        res = calculate_angular_distance(df[['rot_x', 'rot_y', 'rot_z', 'rot_w']])
        res = pd.DataFrame(res, columns=['angular_distance'], index=df.index)
        return res
    
    angular_distance_df = train_df.groupby('sequence_id').apply(calc_angular_distance, include_groups=False)
    angular_distance_df = angular_distance_df.droplevel('sequence_id')
    train_df = train_df.join(angular_distance_df)

    train_df[imu_cols] = train_df[imu_cols].ffill().bfill().fillna(0).values.astype('float32')
    
    return train_df
def standard_scale(arr: np.ndarray) -> np.ndarray:
    means = np.nanmean(arr, axis=0)
    stds = np.nanstd(arr, axis=0)
    stds = np.where(stds == 0, 1, stds)  # Prevent division by zero for constant columns
    scaled = (arr - means) / stds
    return scaled


def pad_or_truncate(
    arr: np.ndarray,
    max_length: int = 128,
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

imu_cols = [
    'acc_x', 'acc_y', 'acc_z','acc_mag','acc_mag_jerk', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 
    'linear_acc_mag', 'linear_acc_mag_jerk',
    'rot_w', 'rot_x', 'rot_y', 'rot_z',
    'rot_angle', 'rot_angle_vel',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
    'angular_distance'
]
df_test=feature_engineering(df_test)
X_test  = []

for sequence_id in tqdm(df_test.sequence_id.unique()):
    ds = df_test[df_test["sequence_id"] == sequence_id]
    X = ds[imu_cols].values
    X = pad_or_truncate(X)
    X = np.concatenate((standard_scale(X[:, 0:3]),X[:, 3:]), axis=1)
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
    num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False
)


import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedSEBlock(nn.Module):
    """
    An enhanced Squeeze-and-Excitation block that uses both average and max pooling,
    inspired by the reference implementation.
    """
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
    """Extract statistical meta-features from input sequence"""
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
    """Pools the output of a sequence-based layer (like LSTM or Attention) over the time dimension."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))
        weights = F.softmax(scores.squeeze(-1), dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)
        return context


import torch
import torch.nn as nn
import torch.nn.functional as F


# EnhancedSEBlock, MetaFeatureExtractor, AttentionLayer


class ModelVariant_GRU(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        num_channels = 20  # Hardcoded for IMU data

        # 1. 
        self.meta_extractor = MetaFeatureExtractor()
        self.meta_dense = nn.Sequential(
            nn.Linear(5 * num_channels, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # 2. 
        self.branches_0 = nn.Sequential(
                    # 输出通道数减少，从16*3=48 -> 12*3=36
                    MultiScaleConv1d(10, 36, kernel_sizes=[3, 5, 7]),
                    # 输出通道数相应调整
                    ResidualSEBlock(108, 192, 3, dropout=0.3),
                    ResidualSEBlock(192, 192, 3, dropout=0.3),
                )
        self.branches_1 = nn.Sequential(
                    # 输出通道数减少，从16*3=48 -> 12*3=36
                    MultiScaleConv1d(10, 36, kernel_sizes=[3, 5, 7]),
                    # 输出通道数相应调整
                    ResidualSEBlock(108, 192, 3, dropout=0.3),
                    ResidualSEBlock(192, 192, 3, dropout=0.3),
                )

        # 3. 序列核心：使用BiGRU替换BiLSTM，并移除MultiHeadAttention
        self.bigru = nn.GRU(
            input_size=192 * 2,
            hidden_size=128,  # 保持hidden_size
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )

        # 4. Attention Pooling层保持不变
        self.attention_pooling = AttentionLayer(256)  # 128 * 2 for bidirectional

        # 5. 
        self.head_1 = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

        self.head_2 = nn.Sequential(
            nn.Linear(256 + 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # For regression task
        )

    def forward(self, x: torch.Tensor):
        # Meta features
        meta = self.meta_extractor(x)
        meta_proj = self.meta_dense(meta)

        # CNN branches
        branch_outputs = []
        channel_input = x[:, :, :10].transpose(1, 2)
        processed = self.branches_0(channel_input)
        branch_outputs.append(processed.transpose(1, 2))
        channel_input = x[:, :, 10:].transpose(1, 2)
        processed = self.branches_1(channel_input)
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
    imu_cols = [
    'acc_x', 'acc_y', 'acc_z','acc_mag','acc_mag_jerk', 'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 
    'linear_acc_mag', 'linear_acc_mag_jerk',
    'rot_w', 'rot_x', 'rot_y', 'rot_z',
    'rot_angle', 'rot_angle_vel',
    'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
    'angular_distance'
     ]
    X_test, y_test, y_hard_test = [], [], []
    df_test=feature_engineering(df_test)
    
    for sequence_id in tqdm(df_test.sequence_id.unique()):
        ds = df_test[df_test["sequence_id"] == sequence_id]
        X = ds[imu_cols].values
        X = pad_or_truncate(X)
        X = np.concatenate((standard_scale(X[:, 0:3]),X[:, 3:]), axis=1)
        X = np.where(np.isnan(X), 0.0, X)  # fill NaNs
        X_test.append(X)

    X_test = np.array(X_test)
    model_paths = glob("/kaggle/input/weight00/weight/*.pth")
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
        model = ModelVariant_GRU(num_classes=18)
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

