import os
import numpy as np
import pandas as pd
import polars as pl
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

# Configuration
config = {
    "batch_size": 128,
    "lr": 1e-3,
    "num_epochs": 100,
    "patience": 10,
    "mixup_alpha": 0.2,
    "use_gru": False,
    "augmentations": True,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load and preprocess training data
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
label_encoder = LabelEncoder()
train_df['gesture'] = label_encoder.fit_transform(train_df['gesture'].astype(str))
gesture_classes = label_encoder.classes_
np.save("gesture_classes.npy", gesture_classes)

# Define feature columns
excluded = {'gesture', 'sequence_type', 'behavior', 'orientation', 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
feature_cols = [c for c in train_df.columns if c not in excluded]
imu_cols = [c for c in feature_cols if not (c.startswith("thm_") or c.startswith("tof_"))]
tof_cols = [c for c in feature_cols if c.startswith("thm_") or c.startswith("tof_")]
imu_dim, tof_dim = len(imu_cols), len(tof_cols)

# Scale features
scaler = StandardScaler().fit(train_df[feature_cols].ffill().bfill().fillna(0).values)
joblib.dump(scaler, "global_scaler.pkl")

# Group by sequence and pad
X_list, y_list, lengths = [], [], []
for _, seq in train_df.groupby("sequence_id"):
    arr = scaler.transform(seq[feature_cols].ffill().bfill().fillna(0).values)
    X_list.append(arr)
    y_list.append(seq['gesture'].iloc[0])
    lengths.append(len(arr))

pad_len = int(np.percentile(lengths, 90))
np.save("sequence_maxlen.npy", pad_len)

X = keras_pad_sequences(X_list, maxlen=pad_len, dtype='float32', padding='post', truncating='post')
y = np.eye(len(np.unique(y_list)))[np.array(y_list)]
X = np.transpose(X, (0, 2, 1))

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y_list)
class_weights = torch.tensor(compute_class_weight('balanced', classes=np.unique(y_list), y=np.array(y_list)), dtype=torch.float)

# Augmentation functions
def jitter(x, sigma=0.01):
    return x + sigma * torch.randn_like(x)

def time_mask(x, mask_ratio=0.1):
    if x.dim() == 2:
        C, T = x.shape
        num_mask = int(T * mask_ratio)
        mask_indices = torch.randint(0, T, (num_mask,))
        x[:, mask_indices] = 0
    elif x.dim() == 3:
        B, C, T = x.shape
        num_mask = int(T * mask_ratio)
        mask_indices = torch.randint(0, T, (B, num_mask))
        for i in range(B):
            x[i, :, mask_indices[i]] = 0
    return x

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[index], lam * y + (1 - lam) * y[index]

# Dataset class
class SequenceDataset(Dataset):
    def __init__(self, X, y=None, augment=False):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float() if y is not None else None
        self.augment = augment

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.augment:
            x = jitter(x)
            x = time_mask(x)
        return (x, self.y[idx]) if self.y is not None else x

# Create data loaders
train_loader = DataLoader(SequenceDataset(X_train, y_train, augment=config["augmentations"]), batch_size=config["batch_size"], shuffle=True, drop_last=True)
val_loader = DataLoader(SequenceDataset(X_val, y_val), batch_size=config["batch_size"])


class SEBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels // 8)
        self.fc2 = nn.Linear(channels // 8, channels)

    def forward(self, x):
        se = x.mean(dim=2)
        se = F.relu(self.fc1(se))
        se = torch.sigmoid(self.fc2(se)).unsqueeze(2)
        return x * se

class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, p=2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=k//2)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=k//2)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.down = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.pool = nn.MaxPool1d(p)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        identity = self.down(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = F.relu(out + identity)
        return self.drop(self.pool(out))

class Attention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.score = nn.Linear(dim, 1)

    def forward(self, x):
        weights = F.softmax(self.score(x).squeeze(2), dim=1).unsqueeze(2)
        return (x * weights).sum(dim=1)

class HARModel(nn.Module):
    def __init__(self, imu_dim, tof_dim, seq_len, num_classes):
        super().__init__()
        self.imu_branch = nn.Sequential(ResidualSEBlock(imu_dim, 64), ResidualSEBlock(64, 128))
        self.tof_branch = nn.Sequential(
            nn.Conv1d(tof_dim, 64, 3, padding=1), nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3),
            nn.Conv1d(64, 128, 3, padding=1), nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(2), nn.Dropout(0.3)
        )
        self.rnn = (nn.GRU if config["use_gru"] else nn.LSTM)(256, 128, batch_first=True, bidirectional=True)
        self.attn = Attention(256)
        self.fc = nn.Sequential(nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, num_classes))

    def forward(self, x):
        x_imu, x_tof = x[:, :imu_dim], x[:, imu_dim:]
        b1 = self.imu_branch(x_imu)
        b2 = self.tof_branch(x_tof)
        x = torch.cat([b1, b2], dim=1).permute(0, 2, 1)
        x, _ = self.rnn(x)
        return self.fc(self.attn(x))


# Initialize model, optimizer, and scheduler
model = HARModel(imu_dim, tof_dim, pad_len, len(gesture_classes)).to(device)
opt = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=1e-4)
sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=5*len(train_loader), T_mult=2)

def soft_ce(pred, target):
    return -(F.log_softmax(pred, dim=1) * target).sum(dim=1).mean()

# Training loop
best_loss = np.inf
for epoch in range(config["num_epochs"]):
    model.train()
    train_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        xb, yb = mixup_data(xb, yb, config["mixup_alpha"])
        opt.zero_grad()
        loss = soft_ce(model(xb), yb)
        loss.backward()
        opt.step()
        sched.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)

    model.eval()
    val_loss, all_preds, all_true = 0, [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            val_loss += soft_ce(preds, yb).item() * xb.size(0)
            all_preds.extend(preds.argmax(1).cpu().numpy())
            all_true.extend(yb.argmax(1).cpu().numpy())
    val_loss /= len(val_loader.dataset)
    f1 = f1_score(all_true, all_preds, average="macro")
    acc = accuracy_score(all_true, all_preds)
    print(f"Epoch {epoch+1}: Train={train_loss:.4f}, Val={val_loss:.4f}, F1={f1:.4f}, Acc={acc:.4f}")
    if val_loss < best_loss:
        best_loss = val_loss
        best_model = model.state_dict()
    else:
        if epoch >= config["patience"]:
            print("Early stopping.")
            break

torch.save(best_model, "gesture_model.pt")
print("Training complete. Model saved as gesture_model.pt")


# Check for Kaggle environment
try:
    import kaggle_evaluation.cmi_inference_server as cmi_server
    is_kaggle = True
except ImportError:
    print("[WARNING] kaggle_evaluation module not found. Running outside of Kaggle environment.")
    is_kaggle = False

# Load
gesture_classes = np.load("/kaggle/working/gesture_classes.npy", allow_pickle=True)
pad_len = int(np.load("/kaggle/working/sequence_maxlen.npy"))
scaler = joblib.load("/kaggle/working/global_scaler.pkl")

# Define feature columns (same as training)
temp_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv", nrows=1)
feature_cols = [c for c in temp_df.columns if c not in excluded]
imu_cols = [c for c in feature_cols if not (c.startswith("thm_") or c.startswith("tof_"))]
tof_cols = [c for c in feature_cols if c.startswith("thm_") or c.startswith("tof_")]
imu_dim, tof_dim = len(imu_cols), len(tof_cols)

# Load model
model = HARModel(imu_dim, tof_dim, pad_len, len(gesture_classes)).to(device)
model.load_state_dict(torch.load("/kaggle/working/gesture_model.pt", map_location=device))
model.eval()

# Inference helper
def preprocess_sequence(df):
    arr = scaler.transform(df[feature_cols].ffill().bfill().fillna(0).values)
    padded = keras_pad_sequences([arr], maxlen=pad_len, dtype='float32', padding='post', truncating='post')[0]
    return torch.from_numpy(padded.T).unsqueeze(0).float()

def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    x = preprocess_sequence(sequence.to_pandas()).to(device)
    with torch.no_grad():
        out = model(x)
        return str(gesture_classes[out.argmax().item()])

# Launch server (Kaggle only)
if is_kaggle:
    inference_server = cmi_server.CMIInferenceServer(predict)
    if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
        inference_server.serve()
    else:
        inference_server.run_local_gateway(
            data_paths=(
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
                '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'
            )
        )
else:
    print("[INFO] Not running in Kaggle evaluation mode. Skipping server startup.")

