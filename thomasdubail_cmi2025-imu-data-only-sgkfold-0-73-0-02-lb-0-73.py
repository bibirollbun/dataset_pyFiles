import random
import numpy as np
import torch
import os

def seed_everything(seed=42):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.use_deterministic_algorithms(True, warn_only=True)

SEED = 42
seed_everything(seed=SEED)

import pandas as pd
import polars as pl
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import GroupKFold
from sklearn.utils.class_weight import compute_class_weight
import joblib
from tqdm import tqdm

from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

import kaggle_evaluation.cmi_inference_server
from matplotlib import pyplot as plt


print("Loading datasets...")
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
train_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")
test_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

# Encode labels
label_encoder = LabelEncoder()
train_df['gesture'] = label_encoder.fit_transform(train_df['gesture'].astype(str))
gesture_classes = label_encoder.classes_

bfrb_gestures = [
    'Above ear - pull hair',
    'Forehead - pull hairline',
    'Forehead - scratch',
    'Eyebrow - pull hair',
    'Eyelash - pull hair',
    'Neck - pinch skin',
    'Neck - scratch',
    'Cheek - pinch skin'
]
bfrb_indices = label_encoder.transform(bfrb_gestures)

imu_cols = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
tof_thm_cols = [c for c in train_df.columns if c.startswith('thm_') or c.startswith('tof_')]

# Reorder so that IMU features come first
feature_cols = imu_cols + tof_thm_cols
imu_dim = len(imu_cols)
tof_thm_dim = len(tof_thm_cols)

print(f"IMU features: {imu_dim}, TOF/Thermal features: {tof_thm_dim}, Total features: {len(feature_cols)}")

# Check for missing values
nan_counts = train_df[feature_cols].isna().sum().sum()
print("Total NaNs in train features:", nan_counts)


# to remove hand dependency in IMU data
# im not sure if the rotation is on the x axis but this give me the best CV
def apply_symmetry(data):
    transformed = data.copy()
    transformed['acc_z'] = -transformed['acc_z']
    transformed['acc_y'] = -transformed['acc_y']
    
    transformed['rot_w'] = transformed['rot_w']
    transformed['rot_x'] = transformed['rot_x']
    transformed['rot_y'] = -transformed['rot_y']
    transformed['rot_z'] = -transformed['rot_z']
    return transformed


train_df = train_df.merge(
    train_dem_df,
    on='subject',
    how='left',
    validate='many_to_one'
)

right_handed_mask = train_df['handedness'] == 1
train_df.loc[right_handed_mask, imu_cols] = apply_symmetry(train_df.loc[right_handed_mask, imu_cols])


sequences = train_df.groupby('sequence_id')
X_list = []
lengths = []
y_list = []

sequence_info = []
for i, (seq_id, seq) in enumerate(sequences):
    seq_data = seq[feature_cols].ffill().bfill().fillna(0).values
    X_list.append(seq_data)
    lengths.append(seq_data.shape[0])
    sequence_info.append({
        'sequence_id': seq_id,
        'subject': seq['subject'].iloc[0],
        'gesture': seq['gesture'].iloc[0]
    })

pad_len = int(np.percentile(lengths, 90))
print(f"Pad/truncate all sequences to length {pad_len} (90th percentile).")

seq_df = pd.DataFrame(sequence_info)
X_array = keras_pad_sequences(
    X_list,
    maxlen=pad_len,
    dtype='float32',
    padding='post',
    truncating='post'
)  # shape: (n_samples, pad_len, total_features)

y_array = seq_df['gesture'].values  # shape: (n_samples,)

num_classes = len(np.unique(y_array))
y_array = np.eye(num_classes)[y_array]  # shape: (n_samples, num_classes)

# Transpose to (n_samples, features, seq_len) for PyTorch
X_array = np.transpose(X_array, (0, 2, 1))


class SequenceDataset(Dataset):
    def __init__(self, X, y=None):
        """
        X: np.ndarray of shape (n_samples, features, seq_len)
        y: np.ndarray of shape (n_samples, num_classes) or None for test
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(channels // reduction, channels, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (batch, channels, seq_len)
        # Squeeze: global average pooling over time dimension
        se = x.mean(dim=2)                      # (batch, channels)
        se = self.relu(self.fc1(se))            # (batch, channels//reduction)
        se = self.sigmoid(self.fc2(se))         # (batch, channels)
        se = se.unsqueeze(2)                    # (batch, channels, 1)
        return x * se                           # scale channels

class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2, dropout_rate=0.3):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock(out_channels, reduction=8)

        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # x: (batch, in_channels, seq_len)
        shortcut = self.shortcut(x)                                 # (batch, out_channels, seq_len)
        out = self.conv1(x)                                          # (batch, out_channels, seq_len)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)                                        # (batch, out_channels, seq_len)
        out = self.bn2(out)

        out = self.se(out)                                           # SE scaling

        out = out + shortcut                                         # skip connection
        out = self.relu(out)

        out = self.pool(out)                                         # (batch, out_channels, seq_len//pool_size)
        out = self.dropout(out)
        return out

class Attention(nn.Module):
    def __init__(self, input_dim):
        super(Attention, self).__init__()
        self.score_fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        # x: (batch, seq_len, features)
        scores = torch.tanh(self.score_fc(x))            # (batch, seq_len, 1)
        scores = scores.squeeze(2)                       # (batch, seq_len)
        weights = F.softmax(scores, dim=1)               # (batch, seq_len)
        weights = weights.unsqueeze(2)                   # (batch, seq_len, 1)
        weighted = x * weights                           # (batch, seq_len, features)
        context = weighted.sum(dim=1)                    # (batch, features)
        return context
        
class IMU_HARModel(nn.Module):
    def __init__(self, total_features, imu_dim, pad_len, num_classes):
        super(IMU_HARModel, self).__init__()
        # IMU branch
        self.resblock1 = ResidualSEBlock(imu_dim, 64, kernel_size=3, pool_size=2, dropout_rate=0.1)
        self.resblock2 = ResidualSEBlock(64, 128, kernel_size=3, pool_size=2, dropout_rate=0.1)

        # After pooling twice, seq_len reduced by factor of 4
        reduced_len = pad_len // 4
        merged_channels = 128 #+ 128  # from IMU and TTF here we use IMU only

        self.bigru = nn.GRU(
            input_size=merged_channels,
            hidden_size=merged_channels//2,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.1,
        )

        # Attention
        self.attention = Attention(input_dim=merged_channels)

        # Dense head
        self.fc1 = nn.Linear(merged_channels, 128, bias=True)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.drop_fc1 = nn.Dropout(0.1)

        self.fc2 = nn.Linear(128, 64, bias=True)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.drop_fc2 = nn.Dropout(0.1)

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: (batch, total_features, seq_len)
        x_imu = x[:, :imu_dim, :]           # (batch, imu_dim, seq_len)
        x_ttf = x[:, imu_dim:, :]           # (batch, rest_dim, seq_len)

        # IMU branch
        b1 = self.resblock1(x_imu)          # (batch, 64, seq_len/2)
        b1 = self.resblock2(b1)             # (batch, 128, seq_len/4)

        # b2 is reserved for tof branch when will be using it

        # Concatenate branches along channel dimension
        merged = b1 # torch.cat([b1, b2], dim=1)  # (batch, 256, seq_len/4)

        # Prepare for GRU: (batch, seq_len/4, 256)
        merged = merged.permute(0, 2, 1)

        # BiGRU
        lstm_out, _ = self.bigru(merged)       # (batch, seq_len/4, 256)

        # Attention
        context = self.attention(lstm_out)    # (batch, 256)

        # Dense head
        x = self.fc1(context)                 # (batch, 256)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.drop_fc1(x)

        x = self.fc2(x)                       # (batch, 128)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.drop_fc2(x)

        out = self.out(x)                     # (batch, num_classes)
        return out

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


def soft_cross_entropy(pred, soft_targets):
    """
    pred: (batch, num_classes) raw scores (no softmax)
    soft_targets: (batch, num_classes) probabilities
    """
    log_probs = F.log_softmax(pred, dim=1)
    loss = -torch.sum(soft_targets * log_probs, dim=1).mean()
    return loss

def mixup_data(x, y, alpha=0.2):
    """
    Return mixed inputs and mixed targets (one-hot) for mixup.
    x: Tensor of shape (batch_size, features, seq_len)
    y: Tensor of shape (batch_size, num_classes)
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    return mixed_x, mixed_y


seed_everything(seed=SEED)

criterion = soft_cross_entropy

n_splits = 5
batch_size = 128
gkf = GroupKFold(n_splits=n_splits)

fold_metrics = []
best_fold_metrics = []
best_models = []

for fold, (train_idx, val_idx) in enumerate(gkf.split(X_array, y_array, groups=seq_df['subject'])):
    print(f"\n{'='*50}")
    print(f"Fold {fold + 1}/{n_splits}")
    print(f"Train subjects: {len(np.unique(seq_df.iloc[train_idx]['subject']))}")
    print(f"Val subjects: {len(np.unique(seq_df.iloc[val_idx]['subject']))}")
    print(f"{'='*50}")
    
    train_dataset = SequenceDataset(X_array[train_idx], y_array[train_idx])
    val_dataset = SequenceDataset(X_array[val_idx], y_array[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    seed_everything(seed=SEED + fold)
    model = IMU_HARModel(
        total_features=len(feature_cols),
        imu_dim=imu_dim,
        pad_len=pad_len,
        num_classes=num_classes,
    ).to(device)
    
    # Optimizer et scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    steps_per_epoch = len(train_loader)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=5 * steps_per_epoch,
        T_mult=2,
        eta_min=1e-5,
    )
    
    # Early stopping
    best_metric = -np.inf
    best_binary_f1 = -np.inf
    best_macro_f1 = -np.inf
    patience = 15
    epochs_no_improve = 0
    num_epochs = 100
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        model.train()
        train_loss = 0.0
        total = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
    
            # Apply mixup
            mixed_x, mixed_y = mixup_data(batch_x, batch_y, alpha=0.2)
    
            optimizer.zero_grad()
            outputs = model(mixed_x)
            loss = criterion(outputs, mixed_y)
            loss.backward()
            optimizer.step()
            scheduler.step()
    
            train_loss += loss.item() * batch_x.size(0)
            total += batch_x.size(0)
        train_loss /= total
    
        # Validation phase
        model.eval()
        val_loss = 0.0
        total = 0
        all_true = []
        all_pred = []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)
                total += batch_x.size(0)
                
                # Get predicted class indices
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                # Get true class indices from one-hot
                trues = torch.argmax(batch_y, dim=1).cpu().numpy()
                
                all_true.append(trues)
                all_pred.append(preds)
        
        val_loss /= total
        all_true = np.concatenate(all_true)
        all_pred = np.concatenate(all_pred)
        
        # Compute competition metrics
        # Binary classification: BFRB (1) vs non-BFRB (0)
        binary_true = np.isin(all_true, bfrb_indices).astype(int)
        binary_pred = np.isin(all_pred, bfrb_indices).astype(int)
        binary_f1 = f1_score(binary_true, binary_pred)
        
        # Collapse non-BFRB gestures into a single class
        collapsed_true = np.where(
            np.isin(all_true, bfrb_indices),
            all_true,
            len(bfrb_gestures)  # Single non-BFRB class
        )
        collapsed_pred = np.where(
            np.isin(all_pred, bfrb_indices),
            all_pred,
            len(bfrb_gestures)  # Single non-BFRB class
        )
        
        # Macro F1 on collapsed classes
        macro_f1 = f1_score(collapsed_true, collapsed_pred, average='macro')
        final_metric = (binary_f1 + macro_f1) / 2
        
        print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
        print(f"  Binary F1 = {binary_f1:.4f}, Macro F1 = {macro_f1:.4f}, Final Metric = {final_metric:.4f}")
        
        if final_metric > best_metric:
            best_metric = final_metric
            best_binary_f1 = binary_f1
            best_macro_f1 = macro_f1
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print(f"  New best metric! Saving model...")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                model.load_state_dict(best_model_state)
                break
    
    torch.save(best_model_state, f"best_model_fold{fold}.pth")
    best_models.append(best_model_state)
    
    fold_metrics.append({
        'binary_f1': binary_f1,
        'macro_f1': macro_f1,
        'final_metric': final_metric
    })
    
    best_fold_metrics.append({
        'binary_f1': best_binary_f1,
        'macro_f1': best_macro_f1,
        'final_metric': best_metric
    })
    
    print(f"\nFold {fold + 1} completed.")
    print(f"Final validation metrics - Binary F1: {binary_f1:.4f}, Macro F1: {macro_f1:.4f}, Final: {final_metric:.4f}")
    print(f"Best validation metrics - Binary F1: {best_binary_f1:.4f}, Macro F1: {best_macro_f1:.4f}, Final: {best_metric:.4f}")

print("\n" + "="*50)
print("Cross-Validation Results")
print("="*50)

# Statistiques pour les meilleures métriques
best_binary_f1 = [m['binary_f1'] for m in best_fold_metrics]
best_macro_f1 = [m['macro_f1'] for m in best_fold_metrics]
best_metrics = [m['final_metric'] for m in best_fold_metrics]

print("\nBest Fold-wise Metrics:")
for i, (bf1, mf1, fm) in enumerate(zip(best_binary_f1, best_macro_f1, best_metrics)):
    print(f"Fold {i+1}: Binary F1 = {bf1:.4f}, Macro F1 = {mf1:.4f}, Final = {fm:.4f}")

print("\nGlobal Statistics (Best Metrics):")
print(f"Mean Best Final Metric: {np.mean(best_metrics):.4f} ± {np.std(best_metrics):.4f}")
print(f"Mean Best Binary F1: {np.mean(best_binary_f1):.4f} ± {np.std(best_binary_f1):.4f}")
print(f"Mean Best Macro F1: {np.mean(best_macro_f1):.4f} ± {np.std(best_macro_f1):.4f}")


model_ensemble = []
for fold in range(5):
    model = IMU_HARModel(
        total_features=len(feature_cols),
        imu_dim=imu_dim,
        pad_len=pad_len,
        num_classes=num_classes,
    ).to(device)
    checkpoint = torch.load(f"best_model_fold{fold}.pth", map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    model_ensemble.append(model)


def preprocess_sequence(df_seq: pd.DataFrame):
    """
    Process a single sequence DataFrame (pandas):
    - Forward/backward fill missing
    - Scale using loaded scaler
    - Pad/truncate to pad_len
    - Return torch.Tensor of shape (1, features, seq_len)
    """
    data = df_seq[feature_cols].ffill().bfill().fillna(0).values
    # Pad/truncate
    padded = keras_pad_sequences(
        [data],
        maxlen=pad_len,
        dtype='float32',
        padding='post',
        truncating='post'
    )[0]  # (pad_len, total_features)
    # Transpose to (features, pad_len)
    tensor = torch.from_numpy(padded.T).unsqueeze(0).float()  # (1, features, pad_len)
    return tensor
    
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Kaggle evaluation API will call this for each sequence.
    sequence: polars DataFrame for a single sequence
    demographics: unused in this model
    Returns: predicted gesture string
    """
    df_seq = sequence.to_pandas()
    df_demo = demographics.to_pandas()
    
    df_seq = df_seq.merge(
    df_demo,
    on='subject',
    how='left',
    validate='many_to_one',
    )
    right_handed_mask = df_seq['handedness'] == 1
    df_seq.loc[right_handed_mask, imu_cols] = apply_symmetry(df_seq.loc[right_handed_mask, imu_cols])

    x_tensor = preprocess_sequence(df_seq).to(device)
    
    all_outputs = []
    with torch.no_grad():
        for model in model_ensemble:
            outputs = model(x_tensor).softmax(dim=-1)
            all_outputs.append(outputs)

    avg_outputs = torch.mean(torch.stack(all_outputs), dim=0)
    pred_idx = torch.argmax(avg_outputs, dim=1).item()
    
    return str(gesture_classes[pred_idx])


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




