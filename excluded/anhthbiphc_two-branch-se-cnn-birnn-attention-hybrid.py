from pathlib import Path

# (Competition metric will only be imported when TRAINing)
TRAIN = False                     # â†� set to True when you want to train
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
PRETRAINED_DIR = Path("/kaggle/input/yoga-dist-quat")  # used when TRAIN=False
EXPORT_DIR = Path("./")                                    # artefacts will be saved here
BATCH_SIZE = 64


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D
import os
import polars as pl
import kaggle_evaluation.cmi_inference_server
from sklearn.metrics import accuracy_score, f1_score
import joblib
from scipy.spatial.transform import Rotation as R


# Táº£i táº­p dá»¯ liá»‡u
#Dá»¯ liá»‡u cáº£m biáº¿n cho táº­p huáº¥n luyá»‡n (Train Sensor Data): 
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")

#ThÃ´ng tin nhÃ¢n kháº©u há»�c cá»§a Ä‘á»‘i tÆ°á»£ng trong táº­p huáº¥n luyá»‡n (giá»›i tÃ­nh, tuá»•i, ...): 
train_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

#Dá»¯ liá»‡u cáº£m biáº¿n cho táº­p kiá»ƒm tra:
test_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv")

# ThÃ´ng tin nhÃ¢n kháº©u há»�c cá»§a Ä‘á»‘i tÆ°á»£ng trong táº­p kiá»ƒm tra:
test_dem_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv")





import os
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import joblib

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import kaggle_evaluation.cmi_inference_server

# ----------------------------------------------------------------------
# 1. Utility functions
# ----------------------------------------------------------------------

def mixup_data(x, y, alpha=0.5):
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

# ----------------------------------------------------------------------
# 2. Load & preprocess data
# ----------------------------------------------------------------------

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
np.save('gesture_classes.npy', gesture_classes)

# Exclude metadata columns
excluded_cols = {
    'gesture', 'sequence_type', 'behavior', 'orientation',
    'row_id', 'subject', 'phase',
    'sequence_id', 'sequence_counter'
}
all_feature_cols = [c for c in train_df.columns if c not in excluded_cols]

# Split feature columns into IMU vs. TOF/Thermal
imu_cols = [c for c in all_feature_cols if not (c.startswith('thm_') or c.startswith('tof_'))]
tof_thm_cols = [c for c in all_feature_cols if c.startswith('thm_') or c.startswith('tof_')]

# Reorder so that IMU features come first
feature_cols = imu_cols + tof_thm_cols
imu_dim = len(imu_cols)
tof_thm_dim = len(tof_thm_cols)
print(f"IMU features: {imu_dim}, TOF/Thermal features: {tof_thm_dim}, Total features: {len(feature_cols)}")

# Check for missing values
nan_counts = train_df[feature_cols].isna().sum().sum()
print("Total NaNs in train features:", nan_counts)

# Fit StandardScaler on all training data
print("Fitting StandardScaler on train data...")
all_values = train_df[feature_cols].ffill().bfill().fillna(0).values
scaler = StandardScaler().fit(all_values)
joblib.dump(scaler, 'global_scaler.pkl')

# Group sequences and build padded arrays
print("Building sequences...")
sequences = train_df.groupby('sequence_id')
X_list = []
lengths = []
y_list = []

for i, (seq_id, seq) in enumerate(sequences):
    seq_data = seq[feature_cols].ffill().bfill().fillna(0).values
    scaled = scaler.transform(seq_data)
    X_list.append(scaled)
    lengths.append(scaled.shape[0])
    y_list.append(seq['gesture'].iloc[0])
    if i % 500 == 0 and i > 0:
        print(f"  Processed {i} sequences...")

# Determine pad length (90th percentile)
pad_len = int(np.percentile(lengths, 90))
print(f"Pad/truncate all sequences to length {pad_len} (90th percentile).")
np.save("sequence_maxlen.npy", pad_len)

# Pad/truncate sequences
print("Padding/truncating sequences...")
from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences
X = keras_pad_sequences(
    X_list,
    maxlen=pad_len,
    dtype='float32',
    padding='post',
    truncating='post'
)  # shape: (n_samples, pad_len, total_features)

y = np.array(y_list)  # shape: (n_samples,)

# One-hot encode labels for mixup
num_classes = len(np.unique(y))
y_cat = np.eye(num_classes)[y]  # shape: (n_samples, num_classes)

# Split into train/validation
X_train_np, X_val_np, y_train_np, y_val_np = train_test_split(
    X, y_cat, test_size=0.2, random_state=42, stratify=y
)
print("Train/Val shapes:", X_train_np.shape, X_val_np.shape, y_train_np.shape, y_val_np.shape)

# Transpose to (n_samples, features, seq_len) for PyTorch
X_train_np = np.transpose(X_train_np, (0, 2, 1))
X_val_np = np.transpose(X_val_np, (0, 2, 1))

# Compute class weights on integer labels
labels_train = np.argmax(y_train_np, axis=1)
class_weights_values = compute_class_weight('balanced',
                                            classes=np.unique(labels_train),
                                            y=labels_train)
class_weights = torch.tensor(class_weights_values, dtype=torch.float)

# ----------------------------------------------------------------------
# 3. Dataset and DataLoader
# ----------------------------------------------------------------------

batch_size = 128

train_dataset = SequenceDataset(X_train_np, y_train_np)
val_dataset = SequenceDataset(X_val_np, y_val_np)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)




# ----------------------------------------------------------------------
# 4. Model definition (Two-branch: SE-CNN + BiRNN + Attention Hybrid Architecture.)
# ----------------------------------------------------------------------

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

class TwoBranchHARModel(nn.Module):
    def __init__(self, total_features, imu_dim, tof_thm_dim, pad_len, num_classes, wd=1e-4):
        super(TwoBranchHARModel, self).__init__()
        # IMU branch
        self.resblock1 = ResidualSEBlock(imu_dim, 64, kernel_size=3, pool_size=2, dropout_rate=0.3)
        self.resblock2 = ResidualSEBlock(64, 128, kernel_size=5, pool_size=2, dropout_rate=0.3)
        self.se_ttf = SEBlock(128, reduction=8)


        # TOF/Thermal branch
        self.conv1_ttf = nn.Conv1d(tof_thm_dim, 64, kernel_size=3, padding=1, bias=False)
        self.bn1_ttf = nn.BatchNorm1d(64)
        self.pool1_ttf = nn.MaxPool1d(kernel_size=2)
        self.drop1_ttf = nn.Dropout(0.3)

        self.conv2_ttf = nn.Conv1d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn2_ttf = nn.BatchNorm1d(128)
        self.pool2_ttf = nn.MaxPool1d(kernel_size=2)
        self.drop2_ttf = nn.Dropout(0.3)

        # After pooling twice, seq_len reduced by factor of 4
        reduced_len = pad_len // 4
        merged_channels = 128 + 128  # from IMU and TTF

        # BiLSTM --> thay báº±ng GRU (self.ltms =nn.LTMS())
        self.rnn = nn.GRU(
            input_size=256, #merged_channels,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.drop_rnn = nn.Dropout(0.4)

        # Attention
        self.attention = Attention(input_dim=512)

        # Dense head
        self.fc1 = nn.Linear(512, 256, bias=False)

        self.bn_fc1 = nn.BatchNorm1d(256)
        self.drop_fc1 = nn.Dropout(0.5)

        self.fc2 = nn.Linear(256, 128, bias=False)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.drop_fc2 = nn.Dropout(0.3)

        self.out = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, total_features, seq_len)
        x_imu = x[:, :imu_dim, :]           # (batch, imu_dim, seq_len)
        x_ttf = x[:, imu_dim:, :]           # (batch, tof_thm_dim, seq_len)

        # IMU branch
        b1 = self.resblock1(x_imu)          # (batch, 64, seq_len/2)
        b1 = self.resblock2(b1)             # (batch, 128, seq_len/4)
        #b1 = self.resblock3(b1)  # NEW

        # TTF branch
        b2 = self.conv1_ttf(x_ttf)          # (batch, 64, seq_len)
        b2 = self.bn1_ttf(b2)
        b2 = F.relu(b2)
        b2 = self.pool1_ttf(b2)             # (batch, 64, seq_len/2)
        b2 = self.drop1_ttf(b2)

        b2 = self.conv2_ttf(b2)             # (batch, 128, seq_len/2)
        b2 = self.bn2_ttf(b2)
        b2 = F.relu(b2)
        b2 = self.pool2_ttf(b2)             # (batch, 128, seq_len/4)
        b2 = self.drop2_ttf(b2)
        b2 = self.se_ttf(b2)  # NEW

        # Concatenate branches along channel dimension
        merged = torch.cat([b1, b2], dim=1)  # (batch, 256, seq_len/4)

        # Prepare for LSTM: (batch, seq_len/4, 256)
        merged = merged.permute(0, 2, 1)

        rnn_out, _ = self.rnn(merged)
        rnn_out = self.drop_rnn(rnn_out)
        context = self.attention(rnn_out)


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

# Instantiate model
input_shape = (len(feature_cols), pad_len)
model = TwoBranchHARModel(
    total_features=len(feature_cols),
    imu_dim=imu_dim,
    tof_thm_dim=tof_thm_dim,
    pad_len=pad_len,
    num_classes=num_classes,
    wd=1e-4
).to(device)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model parameters: {count_parameters(model)}")




# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Instantiate model
input_shape = (len(feature_cols), pad_len)
model = TwoBranchHARModel(
    total_features=len(feature_cols),
    imu_dim=imu_dim,
    tof_thm_dim=tof_thm_dim,
    pad_len=pad_len,
    num_classes=num_classes,
    #wd=1e-4
).to(device)

# Count parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Model parameters: {count_parameters(model)}")


# ==== TRAINING BLOCK (guarded for Kaggle) ====
# YÃªu cáº§u: cÃ¡c biáº¿n Ä‘Ã£ tá»“n táº¡i tá»« trÆ°á»›c: model, device, train_loader, val_loader, mixup_data
# (náº¿u báº¡n Ä‘áº·t tÃªn khÃ¡c, nhá»› Ä‘á»•i láº¡i cho khá»›p)

import os
IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))  # True khi SUBMIT (rerun áº©n)
print(f"[TRAIN GUARD] TRAIN={TRAIN} | IS_RERUN={IS_RERUN}")

if TRAIN and not IS_RERUN:
    # Optimizer
    import numpy as np
    import torch
    import torch.nn.functional as F
    import torch.optim as optim
    from sklearn.metrics import classification_report, f1_score

    lr = 1e-4
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)

    # Scheduler: Giáº£m LR khi val_loss khÃ´ng giáº£m sau 5 epoch
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=8, verbose=True, min_lr=1e-6
    )

    # Loss: soft_cross_entropy (dÃ¹ng vá»›i mixup)
    def soft_cross_entropy(pred, soft_targets):
        log_probs = F.log_softmax(pred, dim=1)
        loss = -torch.sum(soft_targets * log_probs, dim=1).mean()
        return loss

    # Early stopping
    patience = 20
    best_val_loss = np.inf
    epochs_no_improve = 0
    num_epochs = 250
    best_model_state = None
    use_mixup = True  # cÃ³ thá»ƒ báº­t/táº¯t Mixup á»Ÿ Ä‘Ã¢y

    mixup_alpha = 0.5  # TÄƒng nháº¹ alpha mixup

    print("Starting training...")
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            if use_mixup:
                mixed_x, mixed_y = mixup_data(batch_x, batch_y, alpha=mixup_alpha)
            else:
                mixed_x, mixed_y = batch_x, batch_y

            optimizer.zero_grad()
            outputs = model(mixed_x)
            loss = soft_cross_entropy(outputs, mixed_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)

        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                outputs = model(batch_x)
                loss = soft_cross_entropy(outputs, batch_y)
                val_loss += loss.item() * batch_x.size(0)

                # Accuracy
                preds = outputs.argmax(dim=1)
                targets = batch_y.argmax(dim=1)
                correct += (preds == targets).sum().item()
                total += targets.size(0)

        val_loss /= len(val_loader.dataset)
        val_acc = correct / total * 100

        print(f"Epoch {epoch:02d}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}, Val Acc = {val_acc:.2f}%")

        scheduler.step(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            best_model_state = model.state_dict()
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Restoring best model.")
                model.load_state_dict(best_model_state)
                break

    # Save model
    torch.save(best_model_state, "gesture_two_branch_mixup_pytorch.pth")
    print("Training complete and model saved.")
else:
    print("Skip training (either TRAIN=False or SUBMIT rerun).")



# ==== VALIDATION METRICS (guarded) ====
import os, torch
from sklearn.metrics import f1_score, classification_report

IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN'))  # True khi SUBMIT

if TRAIN and not IS_RERUN and 'best_model_state' in globals() and best_model_state is not None:
    # Load best checkpoint vÃ o model Ä‘á»ƒ Ä‘Ã¡nh giÃ¡
    model.load_state_dict(best_model_state)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_x)
            preds = outputs.argmax(dim=1).cpu().numpy()
            targets = batch_y.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets)

    # F1-score
    val_f1_macro = f1_score(all_targets, all_preds, average='macro')
    val_f1_weighted = f1_score(all_targets, all_preds, average='weighted')

    print(f"\nğŸ”� Final Validation Results on Best Model:")
    print(f"Macro F1-score:    {val_f1_macro:.4f}")
    print(f"Weighted F1-score: {val_f1_weighted:.4f}")

    # In classification report (náº¿u cÃ³ gesture_classes)
    try:
        print(classification_report(all_targets, all_preds, target_names=gesture_classes))
    except Exception:
        print(classification_report(all_targets, all_preds))
else:
    print("Skip validation metrics (either TRAIN=False, SUBMIT rerun, hoáº·c thiáº¿u best_model_state).")



# LuÃ´n import á»Ÿ Ã´ launcher nÃ y (Ä‘á»«ng Ä‘á»ƒ trong Ã´ TRAIN)
import kaggle_evaluation.cmi_inference_server
from pathlib import Path
import os



# ----------------------------------------------------------------------
# Inference-only for Kaggle evaluation (no training dependency)
# ----------------------------------------------------------------------
import os
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
import joblib
import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def _preprocess_sequence(df_seq: pd.DataFrame) -> torch.Tensor:
    """ffill/bfill -> scale -> pad/truncate -> (1, features, seq_len) float32 tensor."""
    data = df_seq[_feature_cols].ffill().bfill().fillna(0).values  # (T, F)
    data = scaler.transform(data)

    T, F = data.shape
    if T >= pad_len:
        data = data[:pad_len, :]
    else:
        pad = np.zeros((pad_len - T, F), dtype=data.dtype)
        data = np.vstack([data, pad])

    # (1, features, seq_len)
    data = data.T[None, :, :].astype("float32")
    return torch.from_numpy(data)

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """
    Kaggle evaluation API will call this for each sequence.
    Returns: predicted gesture string
    """
    df_seq = sequence.to_pandas()
    _ensure_model(list(df_seq.columns))
    with torch.no_grad():
        x = _preprocess_sequence(df_seq).to(device)
        logits = _model(x)
        pred_idx = int(torch.argmax(logits, dim=1).item())
    return str(gesture_classes[pred_idx])



# ----------------------------------------------------------------------
# 8. Launch inference server / run local gateway  (robust last cell)
# ----------------------------------------------------------------------
from pathlib import Path
import os

print("KAGGLE_IS_COMPETITION_RERUN =", os.getenv('KAGGLE_IS_COMPETITION_RERUN'))

# ==== Safe wrapper: always provide a working predict entry ====
# Load gesture_classes if missing (for fallback)
try:
    gesture_classes
except NameError:
    import numpy as np
    from pathlib import Path
    PRETRAINED_DIR = Path(os.getenv("PRETRAINED_DIR", "/kaggle/input/two-branch-pretrained"))
    gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

def _predict_entry(sequence, demographics):
    """
    Robust entry point for server and manual writer.
    - Use your real predict() if available.
    - If predict/_ensure_model missing, fall back to constant class to keep pipeline alive.
    """
    try:
        return predict(sequence, demographics)  # <-- real one (from your Inference cell)
    except NameError as e:
        # Fallback: constant class (very low score, but produces a valid file)
        return str(gesture_classes[0])

# 1) Try local gateway (only when Create Version)
try:
    import kaggle_evaluation.cmi_inference_server as cmis
    inference_server = cmis.CMIInferenceServer(_predict_entry)
    if not os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        # Create Version path: ask gateway to write submission.parquet
        inference_server.run_local_gateway(
            data_paths=(
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
            )
        )
except Exception as e:
    print("Gateway run failed -> will fallback to manual writer:", e)

# 2) Fallback: write submission.parquet manually if still missing
from pathlib import Path
if not Path('submission.parquet').exists():
    print("submission.parquet missing -> writing manually...")
    import polars as pl
    import pandas as pd

    test = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
    demo = pl.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')

    seq_ids = test.select('sequence_id').unique().to_series().to_list()
    preds = []
    for sid in seq_ids:
        seq_df = test.filter(pl.col('sequence_id') == sid)
        label = _predict_entry(seq_df, demo)  # <-- use robust wrapper
        preds.append((sid, label))

    sub = pd.DataFrame(preds, columns=['sequence_id', 'gesture'])
    sub.to_parquet('submission.parquet', index=False)
    print("Wrote submission.parquet with", len(sub), "rows")

assert Path('submission.parquet').exists(), "submission.parquet still missing!"
print("OK âœ“ Found submission.parquet")


