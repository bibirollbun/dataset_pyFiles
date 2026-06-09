import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# CONFIG
TRAIN = True
RAW_DIR = Path("/kaggle/input/cmi-detect-behavior-with-sensor-data")
EXPORT_DIR = Path("./")
BATCH_SIZE = 128
PAD_PERCENTILE = 90
LR_INIT = 1e-3
WD = 1e-4
MIXUP_ALPHA = 0.2
EPOCHS = 160
PATIENCE = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def preprocess_sequence(df_seq: pd.DataFrame, feature_cols: list, scaler: StandardScaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

def pad_sequences(X_list, pad_len):
    X_pad = np.zeros((len(X_list), pad_len, X_list[0].shape[1]), dtype=np.float32)
    for i, x in enumerate(X_list):
        l = min(len(x), pad_len)
        X_pad[i, :l, :] = x[:l]
    return X_pad

class SensorDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx], dtype=torch.float32),
                torch.tensor(self.y[idx], dtype=torch.float32))

class MixupSensorDataset(Dataset):
    def __init__(self, X, y, alpha=0.2):
        self.X, self.y = X, y
        self.alpha = alpha

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        lam = np.random.beta(self.alpha, self.alpha)
        j = np.random.randint(0, len(self.X))
        X1, y1 = self.X[i], self.y[i]
        X2, y2 = self.X[j], self.y[j]
        X = lam * X1 + (1 - lam) * X2
        y = lam * y1 + (1 - lam) * y2
        return (torch.tensor(X, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32))

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        y = self.pool(x).squeeze(-1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).unsqueeze(-1)
        return x * y

class ResidualSEBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, pool_size=2, drop=0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.se = SEBlock(out_ch)
        self.pool = nn.MaxPool1d(pool_size)
        self.drop = nn.Dropout(drop)
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.se(x)
        x = x + shortcut
        x = F.relu(x)
        x = self.pool(x)
        x = self.drop(x)
        return x

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        score = torch.tanh(self.fc(x)).squeeze(-1)
        weights = F.softmax(score, dim=1).unsqueeze(-1)
        context = (x * weights).sum(dim=1)
        return context

class IMUOnlyHAR(nn.Module):
    def __init__(self, pad_len, imu_dim, n_classes):
        super().__init__()
        self.imu_dim = imu_dim
        self.imu_res1 = ResidualSEBlock(imu_dim, 64, 3)
        self.imu_res2 = ResidualSEBlock(64, 128, 5)
        self.lstm = nn.LSTM(128, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = AttentionLayer(256)
        self.fc1 = nn.Linear(256, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):
        imu = x.permute(0,2,1)
        imu = self.imu_res1(imu)
        imu = self.imu_res2(imu)
        imu = imu.permute(0,2,1)
        lstm_out, _ = self.lstm(imu)
        attn_out = self.attn(lstm_out)
        x = F.relu(self.bn1(self.fc1(attn_out)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        out = self.out(x)
        return out

if TRAIN:
    print("▶ TRAIN MODE – loading dataset …")
    df = pd.read_csv(RAW_DIR / "train.csv")
    df_demo = pd.read_csv(RAW_DIR / "train_demographics.csv")
    df = pd.merge(df, df_demo, how='left', on='subject')
    le = LabelEncoder(); df['gesture_int'] = le.fit_transform(df['gesture'])
    np.save(EXPORT_DIR / "gesture_classes.npy", le.classes_)

    meta_cols = {'gesture', 'gesture_int', 'sequence_type', 'behavior', 'orientation',
                 'row_id', 'subject', 'phase', 'sequence_id', 'sequence_counter'}
    demo_cols = {'adult_child', 'age', 'sex', 'handedness', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'}
    # Remove demographics and other unwanted columns!
    feature_cols = [c for c in df.columns if c not in meta_cols and c not in demo_cols
                    and not c.startswith('thm_') and not c.startswith('tof_')]
    imu_cols = feature_cols

    scaler = StandardScaler().fit(df[imu_cols].ffill().bfill().fillna(0).values)
    joblib.dump(scaler, EXPORT_DIR / "scaler.pkl")
    np.save(EXPORT_DIR / "feature_cols.npy", np.array(imu_cols))

    seq_gp = df.groupby('sequence_id')
    X_list, y_list, lens = [], [], []
    for seq_id, seq in seq_gp:
        mat = preprocess_sequence(seq, imu_cols, scaler)
        X_list.append(mat)
        y_list.append(seq['gesture_int'].iloc[0])
        lens.append(len(mat))
    pad_len = int(np.percentile(lens, PAD_PERCENTILE))
    np.save(EXPORT_DIR / "sequence_maxlen.npy", pad_len)
    X = pad_sequences(X_list, pad_len)
    y = np.eye(len(le.classes_))[y_list]

    # Split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y_list
    )
    train_data = MixupSensorDataset(X_tr, y_tr, alpha=MIXUP_ALPHA)
    val_data = SensorDataset(X_val, y_val)
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # Class weights
    cw_vals = compute_class_weight('balanced', classes=np.arange(len(le.classes_)), y=y_list)
    class_weight = torch.tensor(cw_vals, dtype=torch.float32).to(DEVICE)

    model = IMUOnlyHAR(pad_len, len(imu_cols), len(le.classes_)).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT, weight_decay=WD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5*len(train_loader))
    criterion = nn.CrossEntropyLoss(weight=class_weight)

    best_acc, bad_epochs = 0, 0
    for epoch in range(EPOCHS):
        model.train()
        tr_loss, tr_count = 0, 0
        for xb, yb in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb.argmax(1))
            loss.backward()
            optimizer.step()
            scheduler.step(epoch + tr_count / len(train_loader))
            tr_loss += loss.item() * xb.size(0)
            tr_count += xb.size(0)
        tr_loss /= tr_count

        model.eval()
        val_loss, val_acc, val_count = 0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                out = model(xb)
                loss = criterion(out, yb.argmax(1))
                val_loss += loss.item() * xb.size(0)
                preds = out.argmax(1).cpu().numpy()
                labels = yb.argmax(1).cpu().numpy()
                val_acc += (preds == labels).sum()
                val_count += xb.size(0)
        val_loss /= val_count
        val_acc /= val_count
        print(f"Epoch {epoch+1}, Loss: {tr_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            bad_epochs = 0
            torch.save(model.state_dict(), EXPORT_DIR / "gesture_imu_only.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= PATIENCE:
                print("Early stopping.")
                break

    print("✔ Training done – model saved in", EXPORT_DIR)



import joblib
import numpy as np
import torch
from pathlib import Path
import polars as pl

PRETRAINED_DIR = Path("/kaggle/working/")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

feature_cols = np.load(PRETRAINED_DIR / "feature_cols.npy", allow_pickle=True).tolist()
pad_len = int(np.load(PRETRAINED_DIR / "sequence_maxlen.npy"))
scaler = joblib.load(PRETRAINED_DIR / "scaler.pkl")
gesture_classes = np.load(PRETRAINED_DIR / "gesture_classes.npy", allow_pickle=True)

class IMUOnlyHAR(nn.Module):
    def __init__(self, pad_len, imu_dim, n_classes):
        super().__init__()
        self.imu_dim = imu_dim
        self.imu_res1 = ResidualSEBlock(imu_dim, 64, 3)
        self.imu_res2 = ResidualSEBlock(64, 128, 5)
        self.lstm = nn.LSTM(128, 128, num_layers=1, batch_first=True, bidirectional=True)
        self.attn = AttentionLayer(256)
        self.fc1 = nn.Linear(256, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128, bias=False)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(0.3)
        self.out = nn.Linear(128, n_classes)

    def forward(self, x):
        imu = x.permute(0,2,1)
        imu = self.imu_res1(imu)
        imu = self.imu_res2(imu)
        imu = imu.permute(0,2,1)
        lstm_out, _ = self.lstm(imu)
        attn_out = self.attn(lstm_out)
        x = F.relu(self.bn1(self.fc1(attn_out)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        out = self.out(x)
        return out

model = IMUOnlyHAR(pad_len, len(feature_cols), len(gesture_classes)).to(DEVICE)
model.load_state_dict(torch.load(PRETRAINED_DIR / "gesture_imu_only.pt", map_location=DEVICE))
model.eval()

def preprocess_sequence(df_seq, feature_cols, scaler):
    mat = df_seq[feature_cols].ffill().bfill().fillna(0).values
    return scaler.transform(mat).astype('float32')

def pad_sequences(X_list, pad_len):
    X_pad = np.zeros((len(X_list), pad_len, X_list[0].shape[1]), dtype=np.float32)
    for i, x in enumerate(X_list):
        l = min(len(x), pad_len)
        X_pad[i, :l, :] = x[:l]
    return X_pad

# Accept demographics for Kaggle compatibility but IGNORE it
def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_seq = sequence.to_pandas()
    mat = preprocess_sequence(df_seq, feature_cols, scaler)
    pad = pad_sequences([mat], pad_len)
    x = torch.tensor(pad, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        out = model(x)
        idx = int(out.argmax(1).cpu().numpy()[0])
    return gesture_classes[idx]



import kaggle_evaluation.cmi_inference_server
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

