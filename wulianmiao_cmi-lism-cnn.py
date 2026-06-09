# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import os
import json
import joblib
import numpy as np
import pandas as pd
import random
from pathlib import Path
import warnings
import shutil
warnings.filterwarnings("ignore")

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR

import polars as pl

# Set Seed
def seed_everything(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(seed=42)

# Configuration
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
EXPORT_DIR = Path("./")                 

BATCH_SIZE = 64
PAD_PERCENTILE = 95
LR_INIT = 5e-4
WD = 3e-3
MIXUP_ALPHA = 0.4
EPOCHS = 160
PATIENCE = 40

print("▶ imports ready · pytorch", torch.__version__)

# Utility Functions
def time_sum(x): return torch.sum(x, dim=1)
def squeeze_last_axis(x): return x.squeeze(-1)
def expand_last_axis(x): return x.unsqueeze(-1)

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _ = x.size()
        se = self.avg_pool(x).view(b, c)
        se = self.fc1(se)
        se = self.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se).view(b, c, 1)
        return x * se

class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, pool_size=2, drop=0.3):
        super(ResidualSECNNBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv1d(in_filters, out_filters, kernel_size, padding='same', bias=False),
            nn.BatchNorm1d(out_filters), nn.ReLU(inplace=True),
            nn.Conv1d(out_filters, out_filters, kernel_size, padding='same', bias=False),
            nn.BatchNorm1d(out_filters), nn.ReLU(inplace=True)
        )
        self.se_block = SEBlock(out_filters)
        self.shortcut_conv = nn.Sequential(
            nn.Conv1d(in_filters, out_filters, 1, padding='same', bias=False),
            nn.BatchNorm1d(out_filters)
        ) if in_filters != out_filters else None
        self.max_pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(drop)

    def forward(self, x):
        shortcut = self.shortcut_conv(x) if self.shortcut_conv else x
        x = self.conv_block(x)
        x = self.se_block(x)
        x = x + shortcut
        x = F.relu(x)
        x = self.max_pool(x)
        x = self.dropout(x)
        return x

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.score_dense = nn.Linear(input_dim, 1)
    def forward(self, inputs):
        score = self.score_dense(inputs)
        score = torch.tanh(score)
        score = squeeze_last_axis(score)
        weights = F.softmax(score, dim=1)
        weights = expand_last_axis(weights)
        context = inputs * weights
        context = time_sum(context)
        return context

class MixupDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
        self.X, self.y, self.alpha = X, y, alpha
    def __len__(self): return len(self.X)
    def __getitem__(self, idx: int):
        x1, y1 = self.X[idx], self.y[idx]
        rand_idx = random.randint(0, len(self.X) - 1)
        x2, y2 = self.X[rand_idx], self.y[rand_idx]
        lam = np.random.beta(self.alpha, self.alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        return x_mixed, y_mixed

class SimpleCNNBlock(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size=3, pool_size=2, drop=0.2):
        super(SimpleCNNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_filters, out_filters, kernel_size, padding='same', bias=False),
            nn.BatchNorm1d(out_filters), nn.ReLU(inplace=True),
            nn.MaxPool1d(pool_size), nn.Dropout(drop)
        )
    def forward(self, x): return self.block(x)

class TwoBranchModel(nn.Module):
    def __init__(self, pad_len, imu_dim, tof_dim, n_classes):
        super(TwoBranchModel, self).__init__()
        self.imu_dim, self.tof_dim = imu_dim, tof_dim
        self.imu_branch = nn.Sequential(
            ResidualSECNNBlock(imu_dim, 64, 3, drop=0.1),
            ResidualSECNNBlock(64, 128, 5, drop=0.1)
        )
        self.tof_branch = nn.Sequential(
            SimpleCNNBlock(tof_dim, 64, drop=0.2),
            SimpleCNNBlock(64, 128, drop=0.2)
        )
        self.recurrent_layer = nn.LSTM(256, 128, bidirectional=True, batch_first=True)
        self.attention_layer = AttentionLayer(128 * 2)
        self.classifier_head = nn.Sequential(
            nn.Linear(128 * 2, 256, bias=False), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(256, 128, bias=False), nn.BatchNorm1d(128),
            nn.ReLU(inplace=True), nn.Dropout(0.3)
        )
        self.output_layer = nn.Linear(128, n_classes)

    def forward(self, x):
        imu = x[:, :, :self.imu_dim].permute(0, 2, 1)
        tof = x[:, :, self.imu_dim:].permute(0, 2, 1)
        x1 = self.imu_branch(imu)
        x2 = self.tof_branch(tof)
        merged = torch.cat([x1, x2], dim=1).permute(0, 2, 1)
        recurrent_out, _ = self.recurrent_layer(merged)
        attention_out = self.attention_layer(recurrent_out)
        classified = self.classifier_head(attention_out)
        return self.output_layer(classified)

def save_checkpoint(state, is_best, directory, filename="latest_checkpoint.pth", best_filename="best_checkpoint.pth"):
    filepath = directory / filename
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, directory / best_filename)

# This global dict will hold the artifacts
artifacts = {}

def fe_and_prepare_data(is_train):
    global artifacts
    if is_train:
        if EXPORT_DIR.exists() and any(EXPORT_DIR.iterdir()):
            print(f"--- Cleaning up existing files in {EXPORT_DIR} ---")
            for item in EXPORT_DIR.iterdir():
                if item.is_file(): item.unlink()
                elif item.is_dir(): shutil.rmtree(item)
        EXPORT_DIR.mkdir(exist_ok=True)
        
        print("▶ Loading and preparing training data...")
        df = pd.read_csv(RAW_DIR / "train.csv")
        le = LabelEncoder()
        df['gesture_int'] = le.fit_transform(df['gesture'])
        np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)
        artifacts['gesture_classes'] = le.classes_
    else:
        # In inference, we don't have a big CSV, so this part is handled in predict
        return

    print("  Calculating engineered features...")
    # 1. Basic Kinematics
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))

    # 2. Advanced Time-Derivative Features
    for col in ['acc_x', 'acc_y', 'acc_z', 'acc_mag', 'rot_x', 'rot_y', 'rot_z', 'rot_angle']:
        df[f'{col}_vel'] = df.groupby('sequence_id')[col].diff().fillna(0)
        df[f'{col}_acc'] = df.groupby('sequence_id')[f'{col}_vel'].diff().fillna(0)

    # 3. Rolling Window Features
    window_sizes = [5, 10]
    feature_cols_for_rolling = ['acc_mag', 'acc_mag_vel', 'acc_mag_acc']
    for window in window_sizes:
        for col in feature_cols_for_rolling:
            df[f'{col}_roll_mean_{window}'] = df.groupby('sequence_id')[col].rolling(window, min_periods=1, center=True).mean().reset_index(level=0, drop=True)
            df[f'{col}_roll_std_{window}'] = df.groupby('sequence_id')[col].rolling(window, min_periods=1, center=True).std().reset_index(level=0, drop=True)

    print("  Calculating ToF features with vectorized NumPy...")
    tof_pixel_cols = [f"tof_{i}_v{p}" for i in range(1, 6) for p in range(64)]
    tof_data_np = df[tof_pixel_cols].replace(-1, np.nan).to_numpy()
    reshaped_tof = tof_data_np.reshape(len(df), 5, 64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean_vals, std_vals = np.nanmean(reshaped_tof, axis=2), np.nanstd(reshaped_tof, axis=2)
        min_vals, max_vals = np.nanmin(reshaped_tof, axis=2), np.nanmax(reshaped_tof, axis=2)
    tof_agg_cols = []
    for i in range(1, 6):
        df[f'tof_{i}_mean'], df[f'tof_{i}_std'] = mean_vals[:, i-1], std_vals[:, i-1]
        df[f'tof_{i}_min'], df[f'tof_{i}_max'] = min_vals[:, i-1], max_vals[:, i-1]
        tof_agg_cols.extend([f'tof_{i}_mean', f'tof_{i}_std', f'tof_{i}_min', f'tof_{i}_max'])
    
    imu_cols = [c for c in df.columns if c.startswith(('acc_', 'rot_'))]
    thm_cols = [c for c in df.columns if c.startswith('thm_')]
    final_feature_cols = imu_cols + thm_cols + tof_agg_cols
    artifacts['final_feature_cols'] = final_feature_cols
    artifacts['imu_dim_final'] = len(imu_cols)
    artifacts['tof_thm_aggregated_dim_final'] = len(thm_cols) + len(tof_agg_cols)
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(final_feature_cols))

    print("  Building, scaling, and padding sequences...")
    X_list, y_list, lens = [], [], []
    for _, seq_df in df.groupby('sequence_id'):
        X_list.append(seq_df[final_feature_cols].ffill().bfill().fillna(0).values.astype('float32'))
        y_list.append(seq_df['gesture_int'].iloc[0])
        lens.append(len(seq_df))
        
    feature_scaler = StandardScaler().fit(np.concatenate(X_list, axis=0))
    joblib.dump(feature_scaler, EXPORT_DIR / "scaler.pkl")
    artifacts['feature_scaler'] = feature_scaler
    X_scaled_list = [feature_scaler.transform(x) for x in X_list]
    
    pad_len = int(np.percentile(lens, PAD_PERCENTILE))
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    artifacts['pad_len'] = pad_len
    
    X_padded = np.zeros((len(X_scaled_list), pad_len, len(final_feature_cols)), dtype='float32')
    for i, seq in enumerate(X_scaled_list):
        X_padded[i, :min(len(seq), pad_len)] = seq[:min(len(seq), pad_len)]
    y_np = np.array(y_list)
    y_one_hot = F.one_hot(torch.from_numpy(y_np), num_classes=len(artifacts['gesture_classes'])).float().numpy()
    
    X_tr, X_val, y_tr_oh, y_val_oh, y_tr_int, _ = train_test_split(
        X_padded, y_one_hot, y_np, test_size=0.1, random_state=82, stratify=y_np
    )
    
    artifacts['train_loader'] = DataLoader(MixupDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr_oh), alpha=MIXUP_ALPHA), batch_size=BATCH_SIZE, shuffle=True, num_workers=os.cpu_count()//2 or 1, pin_memory=True)
    artifacts['val_loader'] = DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val_oh)), batch_size=BATCH_SIZE, shuffle=False, num_workers=os.cpu_count()//2 or 1, pin_memory=True)
    
    cw_vals = compute_class_weight('balanced', classes=np.arange(len(artifacts['gesture_classes'])), y=y_tr_int)
    artifacts['class_weight'] = torch.from_numpy(cw_vals).float()

# --- Main Execution Block ---

# 1. Always train the model first
print("--- Starting Training Phase ---")
fe_and_prepare_data(is_train=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class_weight = artifacts['class_weight'].to(device)
model = TwoBranchModel(
    artifacts['pad_len'], artifacts['imu_dim_final'], 
    artifacts['tof_thm_aggregated_dim_final'], len(artifacts['gesture_classes'])
)
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
lr_scheduler = LambdaLR(optimizer, lambda step: 0.5 * (1 + np.cos(np.pi * step / (EPOCHS * len(artifacts['train_loader'])))))
grad_scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

start_epoch, best_val_accuracy, epochs_no_improve = 0, -1.0, 0

print("Skipping torch.compile() due to GPU hardware incompatibility.")

print("  Starting model training...")
for epoch in range(start_epoch, EPOCHS):
    model.train()
    running_loss = 0
    for data, targets_one_hot in artifacts['train_loader']:
        data, targets_one_hot = data.to(device), targets_one_hot.to(device)
        targets_idx = torch.argmax(targets_one_hot, dim=1)
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(data)
            smoothed_targets = targets_one_hot * (1 - 0.1) + (0.1 / len(artifacts['gesture_classes']))
            per_sample_loss = F.cross_entropy(outputs, smoothed_targets, reduction='none')
            loss = (per_sample_loss * class_weight[targets_idx]).mean()
        optimizer.zero_grad(set_to_none=True)
        grad_scaler.scale(loss).backward()
        grad_scaler.step(optimizer)
        grad_scaler.update()
        lr_scheduler.step()
        running_loss += loss.item() * data.size(0)
    
    epoch_loss = running_loss / len(artifacts['train_loader'].dataset)

    model.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for data, targets_one_hot in artifacts['val_loader']:
            data, targets_one_hot = data.to(device), targets_one_hot.to(device)
            targets_idx = torch.argmax(targets_one_hot, dim=1)
            with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                outputs = model(data)
                v_loss = F.cross_entropy(outputs, targets_idx, weight=class_weight)
            val_loss += v_loss.item() * data.size(0)
            _, predicted = torch.max(outputs.data, 1)
            val_total += targets_idx.size(0)
            val_correct += (predicted == targets_idx).sum().item()
    
    val_epoch_loss = val_loss / val_total
    val_epoch_accuracy = val_correct / val_total

    print(f"E {epoch+1}/{EPOCHS} | L: {epoch_loss:.4f} | VL: {val_epoch_loss:.4f} VAcc: {val_epoch_accuracy:.4f}")

    is_best = val_epoch_accuracy > best_val_accuracy
    if is_best:
        best_val_accuracy = val_epoch_accuracy
        epochs_no_improve = 0
        print(f"  New best val acc: {best_val_accuracy:.4f}. Saving model.")
        model_state = model.state_dict()
        checkpoint_state = {
            'epoch': epoch, 'model_state_dict': model_state,
            'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': lr_scheduler.state_dict(),
            'scaler_state_dict': grad_scaler.state_dict(), 'best_val_accuracy': best_val_accuracy
        }
        save_checkpoint(checkpoint_state, is_best=True, directory=EXPORT_DIR)
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= PATIENCE:
        print(f"Early stopping after {PATIENCE} epochs without improvement.")
        break
print("✔ Training done.")

# --- Fallback save ---
# Always save the latest checkpoint at the end of training, regardless of performance.
# This ensures that a model file is always available for submission.
print("Saving final model checkpoint...")
final_model_state = model.state_dict()
final_checkpoint_state = {
    'epoch': epoch, 'model_state_dict': final_model_state,
    'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': lr_scheduler.state_dict(),
    'scaler_state_dict': grad_scaler.state_dict(), 'best_val_accuracy': best_val_accuracy
}
save_checkpoint(final_checkpoint_state, is_best=False, directory=EXPORT_DIR, filename="latest_checkpoint.pth")
# Also save it as best_checkpoint.pth if no best model was ever saved
if not (EXPORT_DIR / "best_checkpoint.pth").exists():
    print("No best model was saved during training. Saving last model as best_checkpoint.pth.")
    save_checkpoint(final_checkpoint_state, is_best=True, directory=EXPORT_DIR)

print("\n--- Training Script Finished, Preparing Submission ---")


# 2. Immediately load the trained model and prepare for inference
print("\n--- Preparing for Submission Phase ---")
print("  Loading artifacts from training output...")
# The artifacts dictionary is already populated from the training phase.
# We just need to load the model.

print("  Loading best model from training...")
# The device is already defined from the training phase.
inference_model = TwoBranchModel(
    artifacts['pad_len'], artifacts['imu_dim_final'], 
    artifacts['tof_thm_aggregated_dim_final'], len(artifacts['gesture_classes'])
)
# In PyTorch >= 2.6, weights_only defaults to True. We must set it to False 
# to load the full checkpoint which includes optimizer state and other python objects.
checkpoint = torch.load(EXPORT_DIR / "best_checkpoint.pth", map_location=device, weights_only=False)
inference_model.load_state_dict(checkpoint['model_state_dict'])
inference_model.to(device)
inference_model.eval()
print("✔ Artifacts loaded. Ready for evaluation.")


# 3. Define the prediction function using the loaded model
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """Predicts a gesture from a sequence, applying the same feature engineering as in training."""
    df_seq = sequence.to_pandas()
    
    # --- Feature Engineering (must be identical to training) ---
    df_seq['acc_mag'] = np.sqrt(df_seq['acc_x']**2 + df_seq['acc_y']**2 + df_seq['acc_z']**2)
    df_seq['rot_angle'] = 2 * np.arccos(df_seq['rot_w'].clip(-1, 1))
    for col in ['acc_x', 'acc_y', 'acc_z', 'acc_mag', 'rot_x', 'rot_y', 'rot_z', 'rot_angle']:
        df_seq[f'{col}_vel'] = df_seq[col].diff().fillna(0)
        df_seq[f'{col}_acc'] = df_seq[f'{col}_vel'].diff().fillna(0)

    window_sizes = [5, 10]
    feature_cols_for_rolling = ['acc_mag', 'acc_mag_vel', 'acc_mag_acc']
    for window in window_sizes:
        for col in feature_cols_for_rolling:
            df_seq[f'{col}_roll_mean_{window}'] = df_seq[col].rolling(window, min_periods=1, center=True).mean()
            df_seq[f'{col}_roll_std_{window}'] = df_seq[col].rolling(window, min_periods=1, center=True).std()

    tof_pixel_cols = [f"tof_{i}_v{p}" for i in range(1, 6) for p in range(64)]
    tof_data_np = df_seq[tof_pixel_cols].replace(-1, np.nan).to_numpy()
    reshaped_tof = tof_data_np.reshape(len(df_seq), 5, 64)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=RuntimeWarning)
        mean_vals, std_vals = np.nanmean(reshaped_tof, axis=2), np.nanstd(reshaped_tof, axis=2)
        min_vals, max_vals = np.nanmin(reshaped_tof, axis=2), np.nanmax(reshaped_tof, axis=2)
    for i in range(1, 6):
        df_seq[f'tof_{i}_mean'], df_seq[f'tof_{i}_std'] = mean_vals[:, i-1], std_vals[:, i-1]
        df_seq[f'tof_{i}_min'], df_seq[f'tof_{i}_max'] = min_vals[:, i-1], max_vals[:, i-1]
    
    # --- Final Processing ---
    mat_unscaled = df_seq[artifacts['final_feature_cols']].ffill().bfill().fillna(0).values.astype('float32')
    mat_scaled = artifacts['feature_scaler'].transform(mat_unscaled)
    padded_sequence = np.zeros((artifacts['pad_len'], len(artifacts['final_feature_cols'])), dtype='float32')
    seq_len = min(len(mat_scaled), artifacts['pad_len'])
    padded_sequence[:seq_len] = mat_scaled[:seq_len]
    model_input = torch.from_numpy(padded_sequence).float().unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = inference_model(model_input)
        predicted_idx = torch.argmax(outputs, dim=1).item()
        
    return str(artifacts['gesture_classes'][predicted_idx])


# 4. Start the inference server for submission
print("\n--- Starting Inference Server for Submission ---")
try:
    import kaggle_evaluation.cmi_inference_server
    # The official submission environment uses KAGGLE_IS_COMPETITION_RERUN.
    # We run the server if this script is executed in that environment.
    # As requested, removing the compatibility check and forcing the submission server to start.
    print("Starting the inference server for submission.")
    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)
    inference_server.serve()
    print("--- Server finished ---")
except ImportError:
    print("Could not import Kaggle environment, skipping server.")
except Exception as e:
    print(f"An error occurred during submission server setup: {e}")

print("\n--- Submission Script Finished ---")

