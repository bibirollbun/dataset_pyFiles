import numpy as np 
import pandas as pd 
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from scipy.ndimage import gaussian_filter1d
import pickle
from scipy.interpolate import interp1d
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
import warnings
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import time
import joblib
from sklearn.model_selection import train_test_split
from tqdm import tqdm


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
    # Validate required columns
    for col in (row_id_column_name, 'gesture'):
        if col not in solution.columns:
            raise ParticipantVisibleError(f"Solution file missing required column: '{col}'")
        if col not in submission.columns:
            raise ParticipantVisibleError(f"Submission file missing required column: '{col}'")

    metric = CompetitionMetric()
    return metric.calculate_hierarchical_f1(solution, submission)


df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
df_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')


def preprocess_data_bfill(
    df, 
    sequence_col, 
    columns_to_fill, 
    value_to_replace=-1, 
    final_fill_value=0,
):
    df = df.copy()

    # Replace known missing indicator with NaN
    df[columns_to_fill] = df[columns_to_fill].replace(value_to_replace, np.nan)
    # Then fill missing values forward/backward within sequence groups
    df[columns_to_fill] = (
        df.groupby(sequence_col)[columns_to_fill]
          .ffill()
          .bfill()
    )
    # Final fill for any remaining NaNs
    df[columns_to_fill] = df[columns_to_fill].fillna(final_fill_value)
    return df



def create_thm_statistics(df):
    """
    Calculates statistical features across the 5 thermopile sensors for each time step.
    This should be run after nulls have been handled.
    """
    thm_cols = [c for c in df.columns if c.startswith('thm_')]

    if not thm_cols:
        return df

    df_with_stats = df.assign(
        thm_mean = df[thm_cols].mean(axis=1),
        thm_std = df[thm_cols].std(axis=1),
        thm_range = df[thm_cols].max(axis=1) - df[thm_cols].min(axis=1)
    ).fillna(0)  # Ensures the added stats columns don't contain NaN

    return df_with_stats



def add_mag_features_fast(df):
    """Optimized magnitude feature calculation."""
    acc_vectors = df[['acc_x', 'acc_y', 'acc_z']].values
    rot_vectors = df[['rot_x', 'rot_y', 'rot_z']].values
    df['acc_mag'] = np.linalg.norm(acc_vectors, axis=1)
    df['rot_mag'] = np.linalg.norm(rot_vectors, axis=1)
    df[['acc_mag', 'rot_mag']] = df[['acc_mag', 'rot_mag']].fillna(0)
    return df


def add_jerk_features(acc_mag, rot_mag):
    """Calculates jerk features from magnitude arrays using NumPy."""
    acc_jerk = np.insert(np.diff(acc_mag), 0, 0)
    rot_jerk = np.insert(np.diff(rot_mag), 0, 0)
    return acc_jerk, rot_jerk


def remove_gravity_from_acc(acc_data, rot_data):
    gravity_world = np.array([0, 0, 9.81])
    # Handle NaNs or zero quaternions by masking
    valid_mask = ~(np.isnan(rot_data).any(axis=1) | np.all(np.isclose(rot_data, 0), axis=1))
    
    linear_accel = acc_data.copy()
    
    if valid_mask.any():
        # Create Rotation object for all valid quaternions
        rotations = R.from_quat(rot_data[valid_mask])
        # Apply inverse rotation to gravity vector in batch
        gravity_sensor = rotations.apply(gravity_world, inverse=True)
        # Subtract gravity from acc_data only for valid indices
        linear_accel[valid_mask] = acc_data[valid_mask] - gravity_sensor
    
    return linear_accel


def create_tof_statistics_np_no_loop(df):
    """
    Fully vectorized ToF feature extractor with no Python loops.
    Assumes ToF columns are named 'tof_[1-5]_v[0-63]' and ordered consistently.
    """
    df = df.copy()
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    tof_cols_sorted = sorted(tof_cols, key=lambda x: (int(x.split('_')[1]), int(x.split('_')[2][1:])))
    
    tof_data = df[tof_cols_sorted].replace(-1, np.nan).values  # shape: (N, 320)

    N = tof_data.shape[0]
    num_sensors = 5
    num_pixels_per_sensor = 64

    if tof_data.shape[1] != num_sensors * num_pixels_per_sensor:
        raise ValueError("Unexpected number of ToF columns")

    # Reshape: (N, 5, 64)
    tof_reshaped = tof_data.reshape(N, num_sensors, num_pixels_per_sensor)

    # Compute stats along the 64-pixel axis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        tof_mean = np.nanmean(tof_reshaped, axis=2)
        tof_std = np.nanstd(tof_reshaped, axis=2)
        tof_min = np.nanmin(tof_reshaped, axis=2)
        tof_max = np.nanmax(tof_reshaped, axis=2)

    # Assign new columns without a loop
    sensor_indices = np.arange(1, num_sensors + 1)
    stat_names = ['mean', 'std', 'min', 'max']
    stats = [tof_mean, tof_std, tof_min, tof_max]

    for stat_array, stat_name in zip(stats, stat_names):
        for i, sensor_idx in enumerate(sensor_indices):
            df[f'tof_{sensor_idx}_{stat_name}'] = np.nan_to_num(stat_array[:, i])

    created_cols = [f'tof_{i}_{stat}' for i in sensor_indices for stat in stat_names]
    return df, created_cols


def add_rolling_features(array, window):
    """Calculates rolling mean and std for a 1D NumPy array efficiently."""
    padded_array = np.pad(array, (window - 1, 0), mode='edge')
    
    cumsum = np.cumsum(padded_array)
    rolling_sum = cumsum[window - 1:] - np.concatenate(([0], cumsum[:-window]))
    rolling_mean = rolling_sum / window
    
    cumsum_sq = np.cumsum(padded_array**2)
    rolling_sum_sq = cumsum_sq[window - 1:] - np.concatenate(([0], cumsum_sq[:-window]))
    rolling_var = (rolling_sum_sq / window) - (rolling_mean**2)
    rolling_std = np.sqrt(np.maximum(rolling_var, 0))

    for i in range(window - 1):
        rolling_mean[i] = np.mean(array[:i+1])
        rolling_std[i] = np.std(array[:i+1])
        
    return rolling_mean, rolling_std


def run_feature_engineering_fast(df):
    """A new, fast feature engineering function that avoids all slow pandas operations."""
    # We work on a copy to avoid modifying the original DataFrame in the loop
    df = df.copy() 
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    thm_cols = [col for col in df.columns if col.startswith('thm_')]
    rot_cols = [col for col in df.columns if col.startswith('rot_')]
    acc_cols = [col for col in df.columns if col.startswith('acc_') and not col.startswith('acc_lin_')]
    initial_cols = set(df.columns)
    acc_data = df[acc_cols].values
    rot_data = df[rot_cols].values
    acc_lin = remove_gravity_from_acc(acc_data, rot_data)

    lin_acc_cols = [col.replace('acc_', 'acc_lin_') for col in acc_cols]
    acc_lin_df = pd.DataFrame(acc_lin, columns=lin_acc_cols, index=df.index)
    df.drop(columns=lin_acc_cols, inplace=True, errors='ignore')  # optional safety
    df = pd.concat([df, acc_lin_df], axis=1)
    # === FAST VECTORIZED OPERATIONS ===
    # These functions are already fast
    df = add_mag_features_fast(df)
    df = create_thm_statistics(df)
    df, tof_created_list = create_tof_statistics_np_no_loop(df)

    # === USE NUMPY HELPERS FOR SLOW PARTS ===
    # Jerk Features
    acc_jerk, rot_jerk = add_jerk_features(df['acc_mag'].values, df['rot_mag'].values)
    df['acc_jerk'] = acc_jerk
    df['rot_jerk'] = rot_jerk

    # Rolling Features
    cols_to_roll = ['acc_mag', 'rot_mag', 'acc_jerk']
    for col in cols_to_roll:
        rolling_mean, rolling_std = add_rolling_features(df[col].values, window=10)
        df[f'{col}_mean_10'] = rolling_mean
        df[f'{col}_std_10'] = rolling_std

    # The bfill/ffill logic is slow with groupby. 
    # For a single sequence, it's simpler and faster to do this.
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.fillna(0, inplace=True)
    cols_to_drop = ['orientation', 'behavior', 'phase', 'subject', 'sequence_type', 'row_id']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    new_cols = list(set(df.columns) - initial_cols)
    return df, new_cols


def pad_and_truncate_sequence(data, max_len):
    """
    Pads/truncates along the first dimension (time).
    Works for 2D (seq_len, feat_dim) or higher dims like (seq_len, 5, 8, 8).
    """
    if data.shape[0] == 0:
        # If empty, return all zeros with the right shape
        return np.zeros((max_len, *data.shape[1:]), dtype=data.dtype)

    seq_len = data.shape[0]

    if seq_len > max_len:
        return data[-max_len:]   # truncate along time
    elif seq_len < max_len:
        pad_shape = (max_len - seq_len, *data.shape[1:])
        pad = np.zeros(pad_shape, dtype=data.dtype)
        return np.concatenate([pad, data], axis=0)
    else:
        return data


def simulate_motion_drift(data, max_drift = 0.2):
    drift = np.linspace(0, np.random.uniform(-max_drift, max_drift), data.shape[0])
    return data + drift[:, None]


def time_warp(data, sigma = 0.2):
    orig_steps = np.arange(data.shape[0])
    random_curve = np.cumsum(np.random.normal(0, sigma, data.shape[0]))
    warped_steps = orig_steps + random_curve
    warped_steps = np.clip(warped_steps, 0, data.shape[0] - 1)
    interp_func = interp1d(orig_steps, data, axis = 0, fill_value = 'extrapolate')
    return interp_func(warped_steps)


def temperature_drift(data):
    drift = np.linspace(0, np.random.uniform(-1, 1), data.shape[0])
    return data + drift[:, None]


def temporal_smoothing(data, sigma = 1):
    return gaussian_filter1d(data, sigma = sigma, axis = 0)


def channel_dropout(data, drop_rate=0.3):
    num_channels = data.shape[1]
    drop_indices = np.random.choice(num_channels, size=int(num_channels * drop_rate), replace=False)
    data[:, drop_indices] = 0  # or np.nan
    return data


def add_noise(data, noise_level=0.01):
    """Adds Gaussian noise to the data."""
    noise = np.random.normal(0, noise_level, data.shape)
    return data + noise


def apply_imu_augmentations(data):
    data = add_noise(data, 0.03)
    data = random_axis_flip_swap(data)
    data = simulate_motion_drift(data)
    data = time_warp(data)
    return data


def random_axis_flip_swap(data):
    if np.random.rand() <0.5:
        data[:, [0, 1]] = data[:, [1, 0]]
    if np.random.rand() < 0.5:
        data[:, 2] *= -1
    return data


def apply_thermopile_augmentations(data):
    data = temperature_drift(data)
    data = channel_dropout(data, drop_rate=0.3)
    data = temporal_smoothing(data, sigma=1)
    return data


def augment_tof_batch(tof_batch, max_val=254.0, missing_prob=0.05, block_prob=0.3, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    aug = tof_batch.copy()

    # --- Step 1: Missing pixel simulation ---
    random_mask = rng.random(size=aug.shape) < missing_prob
    aug[random_mask] = -1

    # --- Step 1b: Contiguous block occlusion ---
    B = aug.shape[0]
    for i in range(B):
        if rng.random() < block_prob:
            sensor_id = rng.integers(0, 5)
            h = rng.integers(2, 5)
            w = rng.integers(2, 5)
            y = rng.integers(0, 8 - h + 1)
            x = rng.integers(0, 8 - w + 1)
            aug[i, sensor_id, y:y+h, x:x+w] = -1

    # --- Step 2: Sensor bias ---
    bias = rng.normal(0, 2.0, size=(B, 5, 1, 1))
    gain = 1.0 + rng.normal(0, 0.01, size=(B, 5, 1, 1))

    valid_mask = (aug != -1)

    # Broadcast gain & bias
    aug = aug * gain + bias
    aug[~valid_mask] = -1  # restore missing pixels

    # --- Step 3: Noise injection ---
    noise_std = (0.01 + 0.02 * np.clip(aug, 0, max_val) / max_val)
    noise = rng.normal(0, noise_std)
    aug[valid_mask] += noise[valid_mask]

    # --- Step 4: Clamp to sensor range ---
    aug[valid_mask] = np.clip(aug[valid_mask], 0.0, max_val)

    return aug



def evaluate_model(model, val_loader, idx_to_label, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    seq_ids = []

    with torch.no_grad():
        for imu_data, thm_data, other_data, tof_data, tof_stats, labels, gate_target, ids in val_loader:
            imu_data = imu_data.to(device)
            thm_data = thm_data.to(device)
            other_data = other_data.to(device)
            tof_data = tof_data.to(device)
            
            if tof_stats is not None:
                tof_stats = tof_stats.to(device)

            labels = labels.to(device)
            val_outputs = model(imu_data, thm_data, other_data, tof_data, tof_stats)
            
            val_loss += criterion(val_outputs, labels).item() * labels.size(0)
            preds = torch.argmax(val_outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            seq_ids.extend(ids)

    avg_val_loss = val_loss / total
    val_acc = correct / total

    df_true = pd.DataFrame({'sequence_id': seq_ids, 'gesture': [idx_to_label[i] for i in y_true]})
    df_pred = pd.DataFrame({'sequence_id': seq_ids, 'gesture': [idx_to_label[i] for i in y_pred]})

    hierarchical_f1 = score(df_true, df_pred, row_id_column_name='sequence_id')

    return avg_val_loss, val_acc, hierarchical_f1



class Conv1DBranch(nn.Module):
    """
    1D CNN branch for IMU data.
    MODIFIED: This now outputs a sequence of features, not a flattened vector.
    """
    def __init__(self, input_features, num_filters=64, dropout_rate=0.4):
        super(Conv1DBranch, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels=input_features, out_channels=num_filters, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_filters),
            nn.LeakyReLU(0.1)
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)
        
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=3, padding='same'),
            nn.BatchNorm1d(num_filters * 2),
            nn.LeakyReLU(0.1)
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, sequence_length, features).
        Returns:
            Tensor: Output sequence of shape (batch_size, new_sequence_length, features).
        """
        # Conv1d expects (batch_size, features, sequence_length)
        x = x.permute(0, 2, 1)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.pool2(x)
        # Permute back to (batch_size, sequence_length, features) for the GRU
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return x


class Conv2DBranch(nn.Module):
    """
    2D CNN branch for ToF data.
    This architecture assumes the ToF data per time step is a (C, H, W) image.
    MODIFIED: Applies 2D CNN and then 1D pooling to match sequence lengths.
    """
    def __init__(self, in_channels=5, num_filters=64, dropout_rate=0.5):
        super(Conv2DBranch, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2) # 8x8 -> 4x4
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters * 2),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2) # 4x4 -> 2x2
        )
        # --- FIX: Add 1D pooling to reduce sequence length ---
        self.pool_seq1 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.pool_seq2 = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, x):
        """
        Args:
            x (Tensor): Input sequence of shape (batch_size, sequence_length, channels, height, width).
        Returns:
            Tensor: Output sequence of shape (batch_size, new_sequence_length, features).
        """
        batch_size, seq_len, C, H, W = x.shape
        # Combine batch and sequence dimensions to apply 2D CNN
        x = x.view(batch_size * seq_len, C, H, W)
        
        x = self.block1(x)
        x = self.block2(x)
        
        # Flatten the spatial dimensions but keep the sequence dimension
        x = x.view(batch_size, seq_len, -1)
        
        # --- FIX: Apply temporal pooling to match IMU branch ---
        # Permute to (batch_size, features, sequence_length) for MaxPool1d
        x = x.permute(0, 2, 1)
        x = self.pool_seq1(x)
        x = self.pool_seq2(x)
        # Permute back to (batch_size, sequence_length, features)
        x = x.permute(0, 2, 1)
        
        x = self.dropout(x)
        return x


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, gru_output):
        energy = torch.tanh(self.attention(gru_output))  # (B, T, 1)
        attention_weights = F.softmax(energy, dim=1)     # (B, T, 1)
        context_vector = torch.sum(attention_weights * gru_output, dim=1)  # (B, H)
        return context_vector, attention_weights


class CombinedGRUModel(nn.Module):
    def __init__(self, imu_input, thm_input, other_feat_input, num_classes, gru_hidden_size=128, gru_num_layers=1, tof_stats_dim=None):
        super(CombinedGRUModel, self).__init__()
        
        total_input_features = imu_input + thm_input + other_feat_input
        self.feature_branch = Conv1DBranch(input_features=total_input_features, num_filters=128)

        # ToF branch stays separate
        self.tof_branch = Conv2DBranch(in_channels=5, num_filters=64)

        # Output sizes from CNN branches
        feature_cnn_features = 256
        tof_cnn_features = 512
        cnn_output_features = feature_cnn_features + tof_cnn_features

        # BiLSTM
        self.bilstm = nn.LSTM(
            input_size=cnn_output_features,
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=True
        )

        # GRU on top of BiLSTM
        self.gru = nn.GRU(
            input_size=gru_hidden_size * 2,  # BiLSTM is bidirectional
            hidden_size=gru_hidden_size,
            num_layers=gru_num_layers,
            batch_first=True,
            dropout=0.5,
            bidirectional=False
        )

        # Attention now on GRU output
        self.attention = Attention(hidden_size=gru_hidden_size)

        self.use_tof_stats = tof_stats_dim is not None

        # Classifier input
        classifier_input_dim = gru_hidden_size
        if self.use_tof_stats:
            self.tof_stats_proj = nn.Sequential(
                nn.Linear(tof_stats_dim, 64),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.4)
            )
            classifier_input_dim += 64

        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, 128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.gru_dropout = nn.Dropout(0.3)
        self.gru_norm = nn.LayerNorm(gru_hidden_size)
        self.attn_dropout = nn.Dropout(0.3)

    def forward(self, imu_x, thm_x, other_x, tof_x, tof_stats=None):
        feature_seq = self.feature_branch(torch.cat((imu_x, thm_x, other_x), dim=2))
        tof_features_seq = self.tof_branch(tof_x)

        # Align sequence lengths
        min_seq_len = min(feature_seq.size(1), tof_features_seq.size(1))
        feature_seq = feature_seq[:, :min_seq_len, :]
        tof_features_seq = tof_features_seq[:, :min_seq_len, :]

        merged_seq = torch.cat((feature_seq, tof_features_seq), dim=2)
        
        bilstm_out, _ = self.bilstm(merged_seq)
        gru_out, _ = self.gru(bilstm_out)  # feed BiLSTM output into GRU
        gru_out = self.gru_norm(gru_out)
        gru_out = self.gru_dropout(gru_out)
                
        context_vector, _ = self.attention(gru_out)
        context_vector = self.attn_dropout(context_vector)  

        if self.use_tof_stats and tof_stats is not None:
            tof_stats_embed = self.tof_stats_proj(tof_stats)
            context_vector = torch.cat([context_vector, tof_stats_embed], dim=1)

        return self.classifier(context_vector)



class GestureDatasetNumpy(Dataset):
    def __init__(self, df, sequence_ids, 
                 imu_scaler, thm_scaler, other_scaler, tof_scaler, tof_stats_scaler,
                 imu_cols, thm_cols, tof_cols, other_feat_cols, tof_stats_cols, imu_cols_raw, thm_cols_raw, tof_cols_raw,
                 labels=None, seq_slices=None, max_seq_len=128, 
                 is_train=False, use_mixup=False, mixup_alpha=0.4, masking_prob=0.15, noise_level=0.01,
                 run_feature_engineering=None):
        """
        Dataset that applies augmentation -> feature engineering -> scaling on the fly.
        """
        self.df = df
        self.sequence_ids = sequence_ids
        self.imu_scaler = imu_scaler
        self.thm_scaler = thm_scaler
        self.other_scaler = other_scaler
        self.tof_scaler = tof_scaler
        self.tof_stats_scaler = tof_stats_scaler
        self.imu_cols = imu_cols
        self.thm_cols = thm_cols
        self.tof_cols = tof_cols
        self.other_feat_cols = other_feat_cols
        self.tof_stats_cols = tof_stats_cols
        self.imu_cols_raw = imu_cols_raw
        self.thm_cols_raw = thm_cols_raw
        self.tof_cols_raw = tof_cols_raw
        self.labels = labels
        self.seq_slices = seq_slices
        self.max_seq_len = max_seq_len
        self.is_train = is_train
        self.has_labels = labels is not None
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha
        self.masking_prob = masking_prob
        self.noise_level = noise_level
        self.run_feature_engineering = run_feature_engineering

    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        seq_id = self.sequence_ids.iloc[idx]
        start, end = self.seq_slices[seq_id]
        df_seq = self.df.iloc[start:end].copy()

        # ---- Augmentation first ----
        if self.is_train:
            df_seq[self.imu_cols_raw] = apply_imu_augmentations(df_seq[self.imu_cols_raw].values)
            df_seq[self.thm_cols_raw] = apply_thermopile_augmentations(df_seq[self.thm_cols_raw].values)
            tof_arr = df_seq[self.tof_cols_raw].values
            B = tof_arr.shape[0]  # sequence length
            tof_arr = tof_arr.reshape(B, 5, 8, 8)  # make sure this matches your sensors and spatial dims
            augmented = augment_tof_batch(tof_arr)
            # flatten back if you need to assign to dataframe
            df_seq[self.tof_cols_raw] = augmented.reshape(B, -1)

        # ---- Feature engineering ----
        if self.run_feature_engineering is not None:
            df_seq, _ = self.run_feature_engineering(df_seq)

        # ---- Scaling ----
        imu_feat   = self.imu_scaler.transform(df_seq[self.imu_cols])
        thm_feat   = self.thm_scaler.transform(df_seq[self.thm_cols])
        other_feat = self.other_scaler.transform(df_seq[self.other_feat_cols])
        tof_feat   = self.tof_scaler.transform(df_seq[self.tof_cols])

        # Aggregate ToF stats per sequence
        tof_stats_feat = df_seq[self.tof_stats_cols].mean().to_numpy().reshape(1, -1)
        tof_stats_feat = self.tof_stats_scaler.transform(tof_stats_feat).squeeze(0)

        # Reshape ToF if needed
        if tof_feat.ndim in (2, 3):
            seq_len = tof_feat.shape[0]
            tof_feat = tof_feat.reshape(seq_len, 5, 8, 8)

        # ---- Pad/truncate ----
        imu_feat   = pad_and_truncate_sequence(imu_feat, self.max_seq_len)
        thm_feat   = pad_and_truncate_sequence(thm_feat, self.max_seq_len)
        other_feat = pad_and_truncate_sequence(other_feat, self.max_seq_len)
        tof_feat   = pad_and_truncate_sequence(tof_feat, self.max_seq_len)

        # ---- Convert to tensors ----
        imu_tensor   = torch.tensor(imu_feat, dtype=torch.float32)
        thm_tensor   = torch.tensor(thm_feat, dtype=torch.float32)
        other_tensor = torch.tensor(other_feat, dtype=torch.float32)
        tof_tensor   = torch.tensor(tof_feat, dtype=torch.float32)
        tof_stats_tensor = torch.tensor(tof_stats_feat, dtype=torch.float32)

        # ---- Mixup / masking ----
        gate_target = torch.ones(1, dtype=torch.float32)
        if self.is_train and self.use_mixup:
            mask_tof = np.random.rand() < self.masking_prob
            if mask_tof:
                tof_tensor = torch.zeros_like(tof_tensor)
                gate_target = torch.zeros(1, dtype=torch.float32)

            mix_idx = np.random.randint(0, len(self))
            if mix_idx != idx:
                seq_id2 = self.sequence_ids.iloc[mix_idx]
                start2, end2 = self.seq_slices[seq_id2]
                df_seq2 = self.df.iloc[start2:end2].copy()
                df_seq2, _ = self.run_feature_engineering(df_seq2)

                imu2   = torch.tensor(pad_and_truncate_sequence(self.imu_scaler.transform(df_seq2[self.imu_cols]), self.max_seq_len), dtype=torch.float32)
                thm2   = torch.tensor(pad_and_truncate_sequence(self.thm_scaler.transform(df_seq2[self.thm_cols]), self.max_seq_len), dtype=torch.float32)
                other2 = torch.tensor(pad_and_truncate_sequence(self.other_scaler.transform(df_seq2[self.other_feat_cols]), self.max_seq_len), dtype=torch.float32)
                tof2   = torch.tensor(pad_and_truncate_sequence(self.tof_scaler.transform(df_seq2[self.tof_cols]), self.max_seq_len), dtype=torch.float32)

                lam = float(np.clip(np.random.beta(self.mixup_alpha, self.mixup_alpha), 0.2, 0.8))
                imu_tensor   = lam * imu_tensor + (1 - lam) * imu2
                thm_tensor   = lam * thm_tensor + (1 - lam) * thm2
                other_tensor = lam * other_tensor + (1 - lam) * other2
                if not mask_tof:
                    tof_tensor = lam * tof_tensor + (1 - lam) * tof2

        if self.has_labels:
            label_tensor = torch.tensor(self.labels[seq_id], dtype=torch.long)
            return imu_tensor, thm_tensor, other_tensor, tof_tensor, tof_stats_tensor, label_tensor, gate_target, seq_id
        else:
            return imu_tensor, thm_tensor, other_tensor, tof_tensor, tof_stats_tensor, gate_target, seq_id



def preprocess_and_scale(df, run_feature_engineering, fold, MAX_LEN, le, fit=True,
                         scaler_imu=None, scaler_thermopile=None, scaler_other=None, 
                         scaler_tof_array=None, scaler_tof_stats=None):
    """
    Preprocess features and fit/return scalers + column names.
    No transformation here – scaling happens inside the Dataset.
    """
    # Feature engineering
    rot_cols = [c for c in df.columns if c.startswith('rot_')]
    acc_cols = [c for c in df.columns if c.startswith('acc_') and not c.startswith('acc_lin_')]
    imu_cols_raw = rot_cols + acc_cols
    thermopile_cols_raw = [c for c in df.columns if c.startswith('thm_') 
                           and not any(feat in c for feat in ['eng', 'feat'])]
    tof_cols_raw = [c for c in df.columns if c.startswith('tof_')]
    df_processed, new_feature_names = run_feature_engineering(df)

    # Identify columns
    rot_cols = [c for c in df_processed.columns if c.startswith('rot_')]
    acc_cols = [c for c in df_processed.columns if c.startswith('acc_') and not c.startswith('acc_lin_')]
    imu_cols = rot_cols + acc_cols

    thermopile_cols = [c for c in df_processed.columns if c.startswith('thm_') 
                           and not any(feat in c for feat in ['eng', 'feat'])]

    stat_names = ['mean', 'std', 'min', 'max']
    tof_cols = [c for c in df_processed.columns if c.startswith('tof_') and not any(stat in c for stat in stat_names)]
    tof_stats = [c for c in df_processed.columns if c.startswith('tof_') and any(stat in c for stat in stat_names)]

    other_feat_cols = [c for c in new_feature_names if c not in tof_stats]

    # ===== Fit scalers only =====
    if fit:
        scaler_imu = StandardScaler().fit(df_processed[imu_cols])
        scaler_thermopile = StandardScaler().fit(df_processed[thermopile_cols])
        scaler_other = StandardScaler().fit(df_processed[other_feat_cols])
        scaler_tof_array = StandardScaler().fit(df_processed[tof_cols])
        tof_stats_per_seq = df_processed.groupby('sequence_id')[tof_stats].mean()
        scaler_tof_stats = StandardScaler().fit(tof_stats_per_seq)

        # Save scalers
        for name, scaler in zip(
            ['imu', 'thermopile', 'other', 'tof_array', 'tof_stats'],
            [scaler_imu, scaler_thermopile, scaler_other, scaler_tof_array, scaler_tof_stats]
        ):
            with open(f'scaler_{name}_{fold}.pkl', 'wb') as f:
                pickle.dump(scaler, f)

        # Save inference-time metadata only once
        if fold == 0:
            columns_for_inference = {
                'imu_cols_raw': imu_cols_raw,
                'thm_cols_raw': thermopile_cols_raw,
                'tof_cols_raw': tof_cols_raw,
                'imu_cols': imu_cols,
                'thermopile_cols': thermopile_cols,
                'tof_cols': tof_cols,
                'tof_stats': tof_stats,
                'other_feat_cols': other_feat_cols,
                'max_seq_len': MAX_LEN,
                'label_classes': le.classes_.tolist()
            }
            with open('columns_for_inference.pkl', 'wb') as f:
                pickle.dump(columns_for_inference, f)

    return imu_cols_raw, thermopile_cols_raw, tof_cols_raw, imu_cols, thermopile_cols, tof_cols, tof_stats, other_feat_cols, \
           scaler_imu, scaler_thermopile, scaler_other, scaler_tof_array, scaler_tof_stats



from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau

def get_warmup_scheduler(optimizer, warmup_epochs, base_lr):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # linearly increase from small LR to base_lr
            return float(epoch + 1) / float(warmup_epochs)
        else:
            return 1.0
    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def process_one_fold(model, train_loader, val_loader, device, le, y_train_fold,
                     num_epochs=100, patience=7, fold=0):
    
    y_train_fold_encoded = le.transform(y_train_fold)
    class_weights = compute_class_weight(
        'balanced', classes=np.unique(y_train_fold_encoded), y=y_train_fold_encoded
    )
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(class_weights, dtype=torch.float32)
    ).to(device)
    base_lr = 1e-3
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    warmup_epochs = 3
    warmup_scheduler = get_warmup_scheduler(optimizer, warmup_epochs, base_lr)
    main_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs - warmup_epochs)
    best_val_f1 = -1
    patience_counter = 0
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for imu, thm, other, tof, tof_stats, labels, gate_target, seq_id in loop:
            imu, thm, other, tof, tof_stats, labels = (
                imu.to(device), thm.to(device), other.to(device),
                tof.to(device), tof_stats.to(device), labels.to(device)
            )
            optimizer.zero_grad()

            outputs = model(imu, thm, other, tof, tof_stats=tof_stats)
            loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print("❌ NaN loss detected! Stopping training for this fold.")
                return 0

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * labels.size(0)
            loop.set_postfix(train_loss=total_train_loss / len(train_loader.dataset))

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        idx_to_label = {i: c for i, c in enumerate(le.classes_)}
        avg_val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, idx_to_label, criterion, device)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Batch LR: {current_lr:.6f}")
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f'best_model_{fold}.pth')
            print(f"  -> New best model saved with F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # Load best model
    model.load_state_dict(torch.load(f'best_model_{fold}.pth'))
    model.eval()
    val_preds_all, val_labels_all, seq_ids_all = [], [], []

    with torch.no_grad():
        for imu, thm, other, tof, tof_stats, labels, gate_target, seq_ids in val_loader:
            imu, thm, other, tof, tof_stats = (
                imu.to(device), thm.to(device), other.to(device),
                tof.to(device), tof_stats.to(device)
            )
            outputs = model(imu, thm, other, tof, tof_stats=tof_stats)
            val_preds_all.append(outputs.cpu().numpy())
            val_labels_all.append(labels.cpu().numpy())
            seq_ids_all.extend(seq_ids)

    val_preds_all = np.concatenate(val_preds_all)
    val_labels_all = np.concatenate(val_labels_all)
    print(f"--- Training Finished. Best Val F1: {best_val_f1:.4f} ---")

    return best_val_f1, val_preds_all, val_labels_all, seq_ids_all



def build_seq_slices(df, sequence_ids):
    seq_lengths = df['sequence_id'].value_counts(sort=False).to_dict()
    seq_slices = {}
    start_idx = 0
    for seq_id in sequence_ids:  # preserves order of provided IDs
        seq_len = seq_lengths[seq_id]
        end_idx = start_idx + seq_len
        seq_slices[seq_id] = (start_idx, end_idx)
        start_idx = end_idx
    return seq_slices


def process_fold(train_idx, val_idx, df, X, y, fold, le, labels_dict):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LEN = 128

    # Split sequences
    train_ids, val_ids = X.iloc[train_idx], X.iloc[val_idx]
    df_train = df[df['sequence_id'].isin(train_ids)].copy()
    df_val   = df[df['sequence_id'].isin(val_ids)].copy()

    train_seq_slices = build_seq_slices(df_train, train_ids)
    val_seq_slices   = build_seq_slices(df_val, val_ids)

    # === Fit scalers on train ===
    imu_cols_raw, thm_cols_raw, tof_cols_raw, imu_cols, thm_cols, tof_cols, tof_stats_cols, other_cols, \
    scaler_imu, scaler_thm, scaler_other, scaler_tof, scaler_tof_stats = \
        preprocess_and_scale(
            df_train,
            run_feature_engineering_fast,
            fold,
            MAX_LEN,
            le,
            fit=True
        )

    # === For validation we don’t fit again, just ensure col alignment ===
    preprocess_and_scale(
        df_val,
        run_feature_engineering_fast,
        fold,
        MAX_LEN,
        le,
        fit=False,
        scaler_imu=scaler_imu,
        scaler_thermopile=scaler_thm,
        scaler_other=scaler_other,
        scaler_tof_array=scaler_tof,
        scaler_tof_stats=scaler_tof_stats,

    )

    # === Dataset objects (datasets handle scaling internally) ===
    train_dataset = GestureDatasetNumpy(
        df=df_train,
        sequence_ids=train_ids,
        imu_scaler=scaler_imu,
        thm_scaler=scaler_thm,
        other_scaler=scaler_other,
        tof_scaler=scaler_tof,
        tof_stats_scaler=scaler_tof_stats,
        imu_cols=imu_cols,
        thm_cols=thm_cols,
        tof_cols=tof_cols,
        imu_cols_raw = imu_cols_raw, 
        thm_cols_raw = thm_cols_raw,
        tof_cols_raw = tof_cols_raw,
        other_feat_cols=other_cols,
        tof_stats_cols=tof_stats_cols,
        labels=labels_dict,
        seq_slices=train_seq_slices,
        max_seq_len=MAX_LEN,
        is_train=True,
        run_feature_engineering=run_feature_engineering_fast
    )

    val_dataset = GestureDatasetNumpy(
        sequence_ids=val_ids,
        df=df_val,
        imu_scaler=scaler_imu,
        thm_scaler=scaler_thm,
        other_scaler=scaler_other,
        tof_scaler=scaler_tof,
        tof_stats_scaler=scaler_tof_stats,
        imu_cols=imu_cols,
        thm_cols=thm_cols,
        tof_cols=tof_cols,
        imu_cols_raw = imu_cols_raw, 
        thm_cols_raw = thm_cols_raw,
        tof_cols_raw = tof_cols_raw,
        other_feat_cols=other_cols,
        tof_stats_cols=tof_stats_cols,
        labels=labels_dict,
        seq_slices=val_seq_slices,
        max_seq_len=MAX_LEN,
        is_train=False,
        run_feature_engineering=run_feature_engineering_fast
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    # === Build model input sizes dynamically from col lengths ===
    model = CombinedGRUModel(
        imu_input=len(imu_cols),
        thm_input=len(thm_cols),
        other_feat_input=len(other_cols),
        num_classes=len(le.classes_),
        tof_stats_dim=len(tof_stats_cols)
    )

    # === Train & evaluate fold ===
    y_train_fold = y.iloc[train_idx].values
    fold_score, val_preds, val_labels, val_seq_ids = process_one_fold(
        model, train_loader, val_loader, device, le,
        y_train_fold, fold=fold
    )

    return fold_score, val_preds, val_labels, val_seq_ids



sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=42)
sequence_level_df = df.groupby('sequence_id').first().reset_index()
X = sequence_level_df['sequence_id']
y = sequence_level_df['gesture']
groups = sequence_level_df['subject']
le = LabelEncoder()
le.fit(sequence_level_df['gesture'])
y_encoded = le.transform(sequence_level_df['gesture'])
labels_dict = {seq_id: int_label for seq_id, int_label in zip(sequence_level_df['sequence_id'], y_encoded)}
with open('/kaggle/working/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)


start_fold = 0
all_oof_preds = []
all_oof_labels = []
all_oof_seq_ids = []
fold_f1_scores = []

for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):

    if fold < start_fold:
        print(f"--- Skipping Fold {fold+1} (already completed) ---")
        continue  # 'continue' jumps to the next iteration of the loop.
    print(f"\n{'='*25}")
    print(f"====== FOLD {fold+1} / 10 ======")
    print(f"{'='*25}")
    fold_score, val_preds, val_labels, val_seq_ids = process_fold(train_idx, val_idx, df, X, y, fold, le, labels_dict)
    
    fold_f1_scores.append(fold_score)
    all_oof_preds.append(val_preds)
    all_oof_labels.append(val_labels)
    all_oof_seq_ids.extend(val_seq_ids)
    print(f"\nBest F1 Score for Fold {fold+1}: {fold_score:.4f}")

all_oof_preds = np.concatenate(all_oof_preds)
all_oof_labels = np.concatenate(all_oof_labels)
print(f"\n{'='*40}")
print("Cross-Validation Finished.")
print(f"Average F1 Score across all folds: {np.mean(fold_f1_scores):.4f}")
print(f"{'='*40}")




