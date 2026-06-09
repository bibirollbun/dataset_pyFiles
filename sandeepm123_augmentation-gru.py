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
from sklearn.model_selection import StratifiedGroupKFold
from scipy.spatial.transform import Rotation as R
from sklearn.metrics import classification_report, accuracy_score, recall_score, f1_score, precision_score, confusion_matrix
from scipy.ndimage import gaussian_filter1d
import pickle
from scipy.interpolate import interp1d


df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
df_demo = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')


def preprocess_data_bfill(
    df, 
    sequence_col, 
    columns_to_fill, 
    value_to_replace=-1, 
    final_fill_value=0,
    add_missingness_flags=True
):
    df = df.copy()

    # Replace known missing indicator with NaN
    df[columns_to_fill] = df[columns_to_fill].replace(value_to_replace, np.nan)

    new_columns = {}

    tof_columns = [col for col in columns_to_fill if col.startswith('tof_')]
    if add_missingness_flags and tof_columns:
        new_columns['is_tof_missing'] = df[tof_columns].isna().any(axis=1).astype(int)

    # Then fill missing values forward/backward within sequence groups
    df[columns_to_fill] = (
        df.groupby(sequence_col)[columns_to_fill]
          .ffill()
          .bfill()
    )

    # Final fill for any remaining NaNs
    df[columns_to_fill] = df[columns_to_fill].fillna(final_fill_value)

    # Add all new columns at once
    if new_columns:
        df = pd.concat([df, pd.DataFrame(new_columns, index=df.index)], axis=1)

    return df



def create_thm_statistics(df):
    """
    Calculates statistical features across the 5 thermopile sensors for each time step.
    This should be run after nulls have been handled.
    """
    thm_cols = [c for c in df.columns if c.startswith('thm_')]

    if not thm_cols:
        return df

    # --- FIX ---
    # Replace sequential assignment with the efficient .assign() method
    df_with_stats = df.assign(
        thm_mean = df[thm_cols].mean(axis=1),
        thm_std = df[thm_cols].std(axis=1),
        thm_range = df[thm_cols].max(axis=1) - df[thm_cols].min(axis=1)
    )
    
    return df_with_stats


def add_mag_features(df):
    df_with_mag = df.assign(
        acc_mag = np.sqrt((df['acc_x'] ** 2) + (df['acc_y'] ** 2) + (df['acc_z'] ** 2)),
        rot_mag = np.sqrt((df['rot_x'] ** 2) + (df['rot_y'] ** 2) + (df['rot_z'] ** 2))
    )
    return df_with_mag


def add_jerk_features(df):
    df_with_jerk = df.assign(
        acc_jerk = df.groupby('sequence_id')['acc_mag'].diff().fillna(0),
        rot_jerk = df.groupby('sequence_id')['rot_mag'].diff().fillna(0)
    )
    return df_with_jerk


def separate_gravity_and_body_acc(df):
    """
    Uses the orientation quaternion to separate the gravity component.
    Includes a robust temporary fill for missing quaternion data.
    """
    rot_cols = ['rot_x', 'rot_y', 'rot_z', 'rot_w']
    
    # Your NaN handling logic is fine
    temp_rot_df = df[rot_cols].groupby(df['sequence_id']).ffill().bfill()
    temp_rot_df = temp_rot_df.fillna(0)

    # Your rotation logic is fine
    rotations = R.from_quat(temp_rot_df.values)
    gravity_in_device_frame = rotations.inv().apply([0, 0, -9.81])
    
    # --- FIX ---
    # Replace sequential assignment with the efficient .assign() method
    df_with_features = df.assign(
        g_x = gravity_in_device_frame[:, 0],
        g_y = gravity_in_device_frame[:, 1],
        g_z = gravity_in_device_frame[:, 2]
    ).assign(
        body_acc_x = lambda x: x['acc_x'] - x['g_x'],
        body_acc_y = lambda x: x['acc_y'] - x['g_y'],
        body_acc_z = lambda x: x['acc_z'] - x['g_z']
    )
    
    return df_with_features


def create_tof_statistics(df):
    """
    Vectorized: Computes statistical features for each of the 5 ToF sensors,
    treating -1 as missing, and drops raw ToF columns.
    """
    df = df.copy()  # <== THIS FIXES THE WARNING

    all_raw_tof_cols = []
    cols_list = []
    # Create a NaN-masked copy for stats calculation
    tof_all_cols = [c for c in df.columns if c.startswith('tof_')]
    tof_df = df[tof_all_cols].replace(-1, np.nan)

    for i in range(1, 6):
        sensor_cols = [col for col in tof_all_cols if col.startswith(f'tof_{i}_')]
        if not sensor_cols:
            continue

        all_raw_tof_cols.extend(sensor_cols)

        df.loc[:, f'tof_{i}_mean'] = tof_df[sensor_cols].mean(axis=1)
        df.loc[:, f'tof_{i}_std']  = tof_df[sensor_cols].std(axis=1)
        df.loc[:, f'tof_{i}_min']  = tof_df[sensor_cols].min(axis=1)
        df.loc[:, f'tof_{i}_max']  = tof_df[sensor_cols].max(axis=1)
        cols_list.append([f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max'])
    # Fill any NaNs in the new stat columns
    stat_cols = [c for c in df.columns if c.startswith('tof_') and any(k in c for k in ['mean', 'std', 'min', 'max'])]
    df.loc[:, stat_cols] = df[stat_cols].fillna(0)

    return df, cols_list


def add_orientation_angles(df):
    rot_cols = ['rot_x', 'rot_y', 'rot_z', 'rot_w']
    rot_quats = df[rot_cols].fillna(0).values
    euler = R.from_quat(rot_quats).as_euler('xyz', degrees=False)
    
    df['rot_roll'] = euler[:, 0]
    df['rot_pitch'] = euler[:, 1]
    df['rot_yaw'] = euler[:, 2]
    return df


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


def add_rolling_features(df, cols, window=10):
    for col in cols:
        df[f'{col}_mean_{window}'] = df.groupby('sequence_id')[col].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f'{col}_std_{window}']  = df.groupby('sequence_id')[col].transform(lambda x: x.rolling(window, min_periods=1).std().fillna(0))
    return df


def run_feature_engineering(df):
    tof_cols = [col for col in df.columns if col.startswith('tof_')]  # All ToF columns including grids
    thm_cols = [col for col in df.columns if col.startswith('thm_')]
    rot_cols = [col for col in df.columns if col.startswith('rot_')]
    acc_cols = [col for col in df.columns if col.startswith('acc_')]
    
    imu_cols = thm_cols + rot_cols + acc_cols 
    
    all_cols = rot_cols + acc_cols + thm_cols + tof_cols
    initial_cols = set(df.columns)
    df = preprocess_data_bfill(df, 'sequence_id', all_cols)
    df = create_thm_statistics(df)
    df = add_mag_features(df)
    df = add_jerk_features(df)
    df = separate_gravity_and_body_acc(df)
    df, tof_created_list = create_tof_statistics(df)   # Adds mean, std, etc for ToF
    
    df = add_orientation_angles(df)
    df = add_rolling_features(df, ['acc_mag', 'rot_mag', 'acc_jerk'], window=10)
    
    # Drop metadata columns only, keep raw ToF grids for Conv2D input!
    df = df.drop(columns=['orientation', 'behavior', 'phase', 'subject', 'sequence_type', 'row_id'])
    new_cols = list(set(df.columns) - initial_cols)
    return df, new_cols


try_df, tof_created_cols = run_feature_engineering(df.head())


try_df.columns


tof_created_cols


thm_cols = [col for col in try_df.columns if col.startswith('thm_')]
rot_cols = [col for col in try_df.columns if col.startswith('rot_')]
acc_cols = [col for col in try_df.columns if col.startswith('acc_')]
imu_cols = thm_cols + rot_cols + acc_cols 
tof_cols = [col for col in try_df.columns if col.startswith('tof_')]
tof_created_cols


imu_cols





def pad_and_truncate_sequence(data, max_len):
    """
    Pads short sequences at the beginning (pre-padding) and truncates 
    long sequences from the beginning (pre-truncating).
    """
    # Check for empty data to avoid errors with .shape
    if data.shape[0] == 0:
        return np.zeros((max_len, data.shape[1]), dtype=data.dtype)
        
    seq_len, num_features = data.shape
    
    if seq_len > max_len:
        # Truncate from the beginning
        return data[-max_len:, :]
    elif seq_len < max_len:
        # Pad at the beginning
        pad_len = max_len - seq_len
        pad = np.zeros((pad_len, num_features), dtype=data.dtype)
        return np.concatenate([pad, data], axis=0)
    else:
        # If length is already correct
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


class GestureDataset(Dataset):
    def __init__(self, df, imu_cols, feat_eng_cols, tof_cols, thermopile_cols,
                 label_encoder=None, max_seq_len=128, is_train=False):
        self.df = df
        self.imu_cols = imu_cols
        self.feat_eng_cols = feat_eng_cols
        self.tof_cols = tof_cols
        self.thermopile_cols = thermopile_cols
        self.sequence_ids = df['sequence_id'].unique()
        self.label_encoder = label_encoder
        self.max_seq_len = max_seq_len
        self.is_train = is_train

        self.has_labels = 'gesture' in df.columns
        self.label_indices = None
        
        if self.has_labels:
            if self.label_encoder is None:
                raise ValueError("A fitted label_encoder must be provided when labels are present.")
            labels = df.groupby('sequence_id')['gesture'].first().reindex(self.sequence_ids)
            self.label_indices = self.label_encoder.transform(labels)

    def __len__(self):
        return len(self.sequence_ids)

    def __getitem__(self, idx):
        seq_id = self.sequence_ids[idx]
        seq_df = self.df[self.df['sequence_id'] == seq_id]

        imu_data = seq_df[self.imu_cols].values
        feat_eng_data = seq_df[self.feat_eng_cols].values
        tof_data = seq_df[self.tof_cols].values
        thermopile_data = seq_df[self.thermopile_cols].values

        if self.is_train:
            # ✅ FIX: Call augmentation correctly
            imu_data = apply_imu_augmentations(imu_data)
            thermopile_data = apply_thermopile_augmentations(thermopile_data)

        # Pad/truncate
        imu_padded = pad_and_truncate_sequence(imu_data, self.max_seq_len)
        feat_eng_padded = pad_and_truncate_sequence(feat_eng_data, self.max_seq_len)
        tof_padded = pad_and_truncate_sequence(tof_data, self.max_seq_len)
        thermopile_padded = pad_and_truncate_sequence(thermopile_data, self.max_seq_len)

        tof_padded = tof_padded.reshape(self.max_seq_len, 5, 8, 8)
        combined_features = np.concatenate([imu_padded, feat_eng_padded, thermopile_padded], axis=1)
        combined_tensor = torch.tensor(combined_features, dtype=torch.float32)
        tof_tensor = torch.tensor(tof_padded, dtype=torch.float32)

        if self.has_labels:
            label = torch.tensor(self.label_indices[idx], dtype=torch.long)
            return combined_tensor, tof_tensor, label, seq_id
        else:
            return combined_tensor, tof_tensor, seq_id


def evaluate_model(model, val_loader, idx_to_label, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    y_true = []
    y_pred = []
    seq_ids = [] # ✅ FIX: Initialize list to store sequence IDs

    with torch.no_grad():
        # ✅ FIX: Swap 'sensor_data' and 'tof_data' to match dataset output order
        for sensor_data, tof_data, labels, ids in val_loader:
            tof_data = tof_data.to(device)
            sensor_data = sensor_data.to(device)
            labels = labels.to(device)
            
            # ✅ FIX: Pass arguments in the correct order
            val_outputs = model(sensor_data, tof_data)
            
            val_loss += criterion(val_outputs, labels).item() * labels.size(0)

            preds = torch.argmax(val_outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            seq_ids.extend(ids) # Collect the actual sequence IDs

    avg_val_loss = val_loss / total
    val_acc = correct / total

    # ✅ FIX: Use the actual sequence IDs for scoring instead of a generated range
    df_true = pd.DataFrame({'sequence_id': seq_ids, 'gesture': [idx_to_label[i] for i in y_true]})
    df_pred = pd.DataFrame({'sequence_id': seq_ids, 'gesture': [idx_to_label[i] for i in y_pred]})

    hierarchical_f1 = score(df_true, df_pred, row_id_column_name='sequence_id')

    return avg_val_loss, val_acc, hierarchical_f1


class Conv1DBranch(nn.Module):
    """
    1D CNN branch for IMU data.
    MODIFIED: This now outputs a sequence of features, not a flattened vector.
    """
    def __init__(self, input_features, num_filters=64, dropout_rate=0.2):
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
    def __init__(self, in_channels=5, num_filters=64, dropout_rate=0.3):
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
    """
    This model combines two CNN branches and feeds their output into 
    a Bidirectional GRU followed by a simple pooling (no attention).
    """
    def __init__(self, imu_input, num_classes, gru_hidden_size=128, gru_num_layers=2):
        super(CombinedGRUModel, self).__init__()
        
        # --- CNN Branches ---
        self.imu_branch = Conv1DBranch(input_features=imu_input, num_filters=64)
        self.tof_branch = Conv2DBranch(in_channels=5, num_filters=64)

        imu_cnn_features = 64 * 2
        tof_cnn_features = (64 * 2) * (2 * 2)  # Adjust according to CNN output shape
        gru_input_size = imu_cnn_features + tof_cnn_features
        
        # --- BiGRU Layer ---
        self.gru = nn.GRU(
            input_size=gru_input_size, 
            hidden_size=gru_hidden_size, 
            num_layers=gru_num_layers,
            batch_first=True, 
            dropout=0.5,
            bidirectional=True
        )
        
        # --- Classifier ---
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_size * 2, 128),  # BiGRU output size
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        self.attention = Attention(hidden_size=gru_hidden_size * 2)  # BiGRU is bidirectional


    def forward(self, imu_x, tof_x):
        imu_features_seq = self.imu_branch(imu_x)
        tof_features_seq = self.tof_branch(tof_x)

        min_seq_len = min(imu_features_seq.size(1), tof_features_seq.size(1))
        imu_features_seq = imu_features_seq[:, :min_seq_len, :]
        tof_features_seq = tof_features_seq[:, :min_seq_len, :]

        merged_seq = torch.cat((imu_features_seq, tof_features_seq), dim=2)

        gru_out, _ = self.gru(merged_seq)

        context_vector, _ = self.attention(gru_out)
        output = self.classifier(context_vector)
        return output


def process_one_fold(model, train_loader, val_loader, device, le, y_train_fold, num_epochs=50, patience=5, fold=0, use_mixup=True):
    y_train_fold_encoded = le.transform(y_train_fold)
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train_fold_encoded), y=y_train_fold_encoded)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    best_val_f1 = -1
    patience_counter = 0
    model.to(device)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        # ✅ FIX: Swap 'sensor' and 'tof' to match dataset output order.
        # The dataloader returns (combined_tensor, tof_tensor, label)
        # We also add '_' to ignore the sequence_id that the dataset now returns.
        for batch_idx, (sensor, tof, labels, seq_id) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}, Batch {batch_idx+1}/{len(train_loader)}")
            imu = sensor.to(device) # This is now the correct 3D tensor
            tof = tof.to(device)    # This is now the correct 5D tensor
            labels = labels.to(device)
            optimizer.zero_grad()
            
            if use_mixup:
                lam = np.random.beta(0.4, 0.4)
                lam = np.clip(lam, 0.2, 0.8)
                index = torch.randperm(labels.size(0)).to(device)
                mixed_tof = lam * tof + (1 - lam) * tof[index]
                mixed_imu = lam * imu + (1 - lam) * imu[index]
                labels_a, labels_b = labels, labels[index]
                
                # ✅ FIX: Pass arguments to the model in the correct order.
                outputs = model(mixed_imu, mixed_tof)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(outputs, labels_b)
            else:
                outputs = model(imu, tof)
                loss = criterion(outputs, labels)

            if torch.isnan(loss):
                print("NaN loss detected! Stopping training for this fold.")
                return 0

            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * labels.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        idx_to_label = {i: c for i, c in enumerate(le.classes_)}
        avg_val_loss, val_acc, val_f1 = evaluate_model(model, val_loader, idx_to_label, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")

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

    print(f"--- Training Finished. Best Val F1: {best_val_f1:.4f} ---")
    return best_val_f1


def process_fold(train_idx, val_idx, df, X, y, fold, le):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MAX_LEN = 128
    
    # --- Create train/validation dataframes for the fold ---
    train_ids, val_ids = X.iloc[train_idx], X.iloc[val_idx]
    df_train = df[df['sequence_id'].isin(train_ids)].copy()
    df_val = df[df['sequence_id'].isin(val_ids)].copy()

    # --- Run Feature Engineering ---
    print("Running Feature Engineering...")
    # Use the full df for column definitions to ensure consistency
    df_processed_schema, new_feature_names = run_feature_engineering(df.head(1))
    df_train_processed, _ = run_feature_engineering(df_train)
    df_val_processed, _ = run_feature_engineering(df_val)
    print("Feature Engineering complete.")

    # --- Define Column Groups for the Dataset ---
    
    # ✅ FIX: Define IMU columns (accel + gyro only) and Thermopile columns separately
    # to prevent overlap.
    rot_cols = [c for c in df_processed_schema.columns if c.startswith('rot_')]
    acc_cols = [c for c in df_processed_schema.columns if c.startswith('acc_')]
    imu_cols_raw = rot_cols + acc_cols

    thermopile_cols_raw = [c for c in df_processed_schema.columns if c.startswith('thm_')]
    
    # ToF grid columns for the 2D CNN branch
    tof_cols_grid = [c for c in df.columns if c.startswith('tof_') and '_' in c and 'mean' not in c and 'std' not in c]

    # All other feature-engineered columns for the 1D CNN branch
    # ✅ FIX: Exclude lists are now correct and not redundant.
    exclude_cols = set(imu_cols_raw) | set(thermopile_cols_raw) | set(tof_cols_grid)
    feat_eng_cols = [c for c in new_feature_names if c not in exclude_cols]
    
    # --- Create Datasets and DataLoaders ---
    train_dataset = GestureDataset(
        df_train_processed, 
        imu_cols=imu_cols_raw, 
        feat_eng_cols=feat_eng_cols, 
        tof_cols=tof_cols_grid, 
        thermopile_cols=thermopile_cols_raw,
        label_encoder=le, 
        max_seq_len=MAX_LEN, 
        is_train=True
    )
    val_dataset = GestureDataset(
        df_val_processed, 
        imu_cols=imu_cols_raw, 
        feat_eng_cols=feat_eng_cols, 
        tof_cols=tof_cols_grid, 
        thermopile_cols=thermopile_cols_raw,
        label_encoder=le, 
        max_seq_len=MAX_LEN, 
        is_train=False
    )
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    num_1d_features = len(imu_cols_raw) + len(feat_eng_cols) + len(thermopile_cols_raw)
    
    model = CombinedGRUModel(
        imu_input=num_1d_features,
        num_classes=len(le.classes_)
    )

    # --- Train and Evaluate ---
    y_train_fold = y.iloc[train_idx].values
    best_f1 = process_one_fold(model, train_loader, val_loader, device, le, y_train_fold, fold = fold, use_mixup = True )
    return best_f1


sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
sequence_level_df = df.groupby('sequence_id').first().reset_index()
X = sequence_level_df['sequence_id']
y = sequence_level_df['gesture']
groups = sequence_level_df['subject']
le = LabelEncoder()
le.fit(sequence_level_df['gesture'])
with open('/kaggle/working/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)



start_fold = 0

# 2. (Optional but good practice) Pre-load scores for completed folds.
#    The score for FOLD 1 (index 0) was 0.8247.
fold_f1_scores = []

# 3. Let the loop run from the beginning, but skip the completed folds.
for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
    
    # This 'if' statement is the key.
    # It checks if the current fold index is less than your desired start.
    if fold < start_fold:
        print(f"--- Skipping Fold {fold+1} (already completed) ---")
        continue  # 'continue' jumps to the next iteration of the loop.
    
    # The rest of your code will only run for fold >= start_fold
    print(f"\n{'='*25}")
    print(f"====== FOLD {fold+1} / 5 ======")
    print(f"{'='*25}")
    fold_score = process_fold(train_idx, val_idx, df, X, y, fold, le)
    
    fold_f1_scores.append(fold_score)
    print(f"\nBest F1 Score for Fold {fold+1}: {fold_score:.4f}")

print(f"\n{'='*40}")
print("Cross-Validation Finished.")
print(f"Average F1 Score across all folds: {np.mean(fold_f1_scores):.4f}")
print(f"{'='*40}")




