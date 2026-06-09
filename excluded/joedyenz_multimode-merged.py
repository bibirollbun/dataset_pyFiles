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


def engineer_therm_features(tmp_array):
    """
    tmp_array: numpy array or torch tensor of shape (..., 5)
        expected order: [TMP1, TMP2, TMP3, TMP4, TMP5]
    returns:
        engineered features array (..., 8) = 5 raw + 3 engineered
    """

    TMP1 = tmp_array[..., 0]
    TMP2 = tmp_array[..., 1]
    TMP3 = tmp_array[..., 2]
    TMP4 = tmp_array[..., 3]
    TMP5 = tmp_array[..., 4]

    # engineered features
    hand_proximity = TMP2 - TMP1
    left_right = TMP3 - TMP5
    arm_delta = TMP1 - TMP4

    engineered = torch.stack([
        TMP1, TMP2, TMP3, TMP4, TMP5,
        hand_proximity,
        left_right,
        arm_delta,
    ], dim=-1)

    return engineered

import torch
import torch.nn as nn
import torch.nn.functional as F

class ThermEncoder(nn.Module):
    def __init__(self, k_classes=8):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, k_classes)  # logits in latent k-space
        )
        
    def forward(self, x):
        """
        x: (batch, 5) raw therm readings
        returns logits: (batch, k_classes)
        """
        # convert raw → engineered
        x = engineer_therm_features(x)
        
        logits = self.mlp(x)
        return logits

import torch
from torch.utils.data import Dataset
import numpy as np

class ThermWindowDataset(Dataset):
    def __init__(
        self,
        df,
        window_size=20,
        stride=5,
        feature_cols=("thm_1", "thm_1", "thm_1", "thm_1", "thm_1"),
        label_col="gesture_id"
    ):
        self.samples = []

        for seq_id, seq_df in df.groupby("sequence_id"):
            data = seq_df[feature_cols].values   # (L, 5)
            labels = seq_df[label_col].values

            L = len(seq_df)
            for start in range(0, L - window_size + 1, stride):
                end = start + window_size

                x = data[start:end].T                  # (5, T)
                y = labels[start:end]

                target = np.bincount(y).argmax()

                self.samples.append((
                    torch.tensor(x, dtype=torch.float32),
                    torch.tensor(target, dtype=torch.long)
                ))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ============================================================
# FLEXIBLE HYBRID CNN + TRANSFORMER MODEL FOR KAGGLE
# ============================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.preprocessing import LabelEncoder
from scipy.spatial.transform import Rotation as R
import time
from sklearn.preprocessing import StandardScaler

# ============================================================
# CONFIGURABLE VARIABLES
# ============================================================

CSV_PATH = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
WINDOW_BEFORE = 10
WINDOW_AFTER = 10
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
D_MODEL = 64
AXIS_HIDDEN_DIM = 32
USE_AXISCNN = True
USE_TRANSFORMER = True
FUSION = 'concat'  # 'concat' or 'gated'
DROPOUT = 0.1
USE_ACC = True
USE_ROT = True
VAL_RATIO = 0.2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def handle_quaternion_missing_values(rot_data: np.ndarray) -> np.ndarray:
    rot_cleaned = rot_data.copy()
    for i in range(len(rot_data)):
        row = rot_data[i]
        missing_count = np.isnan(row).sum()
        if missing_count == 0:
            norm = np.linalg.norm(row)
            rot_cleaned[i] = row / norm if norm > 1e-8 else [1.0,0.0,0.0,0.0]
        elif missing_count == 1:
            missing_idx = np.where(np.isnan(row))[0][0]
            valid_values = row[~np.isnan(row)]
            sum_squares = np.sum(valid_values**2)
            if sum_squares <= 1.0:
                missing_value = np.sqrt(max(0,1.0-sum_squares))
                if i>0 and not np.isnan(rot_cleaned[i-1,missing_idx]):
                    if rot_cleaned[i-1,missing_idx]<0: missing_value=-missing_value
                rot_cleaned[i,missing_idx]=missing_value
                rot_cleaned[i,~np.isnan(row)] = valid_values
            else: rot_cleaned[i]=[1.0,0.0,0.0,0.0]
        else:
            rot_cleaned[i]=[1.0,0.0,0.0,0.0]
    return rot_cleaned

def compute_world_acceleration(acc: np.ndarray, rot: np.ndarray) -> np.ndarray:
    try:
        rot_scipy = rot[:,[1,2,3,0]]
        norms = np.linalg.norm(rot_scipy, axis=1)
        mask = norms<1e-8
        rot_scipy[mask] = [0.0,0.0,0.0,1.0]
        r = R.from_quat(rot_scipy)
        return r.apply(acc)
    except:
        return acc.copy()

def remove_gravity_from_acc(acc_data, rot_data):
    num_samples = acc_data.shape[0]
    linear_accel = np.zeros_like(acc_data)
    gravity_world = np.array([0,0,9.81])
    for i in range(num_samples):
        if np.all(np.isnan(rot_data[i])) or np.all(np.isclose(rot_data[i],0)):
            linear_accel[i,:] = acc_data[i,:]
            continue
        try:
            rotation = R.from_quat(rot_data[i])
            gravity_sensor_frame = rotation.apply(gravity_world,inverse=True)
            linear_accel[i,:] = acc_data[i,:]-gravity_sensor_frame
        except:
            linear_accel[i,:] = acc_data[i,:]
    return linear_accel

def calculate_angular_velocity_from_quat(rot_data, time_delta=1/200):
    num_samples = rot_data.shape[0]
    angular_vel = np.zeros((num_samples,3))
    for i in range(num_samples-1):
        q_t = rot_data[i]; q_t_plus = rot_data[i+1]
        if np.any(np.isnan(q_t)) or np.any(np.isnan(q_t_plus)): continue
        try:
            r1 = R.from_quat(q_t); r2 = R.from_quat(q_t_plus)
            delta_rot = r1.inv()*r2
            angular_vel[i,:] = delta_rot.as_rotvec()/time_delta
        except: continue
    return angular_vel

def calculate_angular_distance(rot_data):
    num_samples = rot_data.shape[0]
    angular_dist = np.zeros(num_samples)
    for i in range(num_samples-1):
        q1=rot_data[i]; q2=rot_data[i+1]
        if np.any(np.isnan(q1)) or np.any(np.isnan(q2)): continue
        try:
            r1=R.from_quat(q1); r2=R.from_quat(q2)
            rel_rot = r1.inv()*r2
            angular_dist[i]=np.linalg.norm(rel_rot.as_rotvec())
        except: continue
    return angular_dist

# ============================================================
# DATASET
# ============================================================
class AccRotHybridDataset(Dataset):
    ACC_COLS = ["acc_x", "acc_y", "acc_z"]
    ROT_COLS = ["rot_w", "rot_x", "rot_y", "rot_z"]

    def __init__(self, df, window_before=10, window_after=10, normalize_tof=True):
        #self.df = pd.read_csv(csv_path)
        self.df = df
        self.window_before = window_before
        self.window_after = window_after
        self.window_len = window_before + window_after + 1
        self.normalize_tof = normalize_tof

        # -----------------------
        # Gesture encoding
        # -----------------------
        self.df["gesture"] = self.df["gesture"].astype(str)
        self.le = LabelEncoder()
        self.df["gesture_id"] = self.le.fit_transform(self.df["gesture"])
        self.gesture_classes = self.le.classes_

        # -----------------------
        # ToF columns
        # -----------------------
        self.tof_cols = [c for c in self.df.columns if c.startswith("tof_")]
        self.tof_dim = len(self.tof_cols)  # This is important!

        # -----------------------
        # Optional ToF normalization
        tof_data = self.df[self.tof_cols]
        X = tof_data.values.astype(np.float32)
        # -----------------------
        if self.normalize_tof:
            self.tof_scaler = StandardScaler()
            X = self.tof_scaler.fit_transform(X)
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        self.df[self.tof_cols] = X

        # -----------------------
        # Therm
        # -----------------------
        self.therm_cols = [c for c in self.df.columns if c.startswith("thm_")]
        
        # -----------------------
        # Build windows
        # -----------------------
        self.sequences = list(self.df.groupby("sequence_id"))
        self.windows = []

        for _, seq in self.sequences:
            seq = seq.reset_index(drop=True)

            gesture_rows = seq[seq["phase"] == "Gesture"]
            if gesture_rows.empty:
                continue

            center = gesture_rows.index[0]
            start = center - self.window_before
            end = center + self.window_after + 1

            start_c = max(0, start)
            end_c = min(len(seq), end)
            
            # -----------------------
            # ACC / ROT
            # -----------------------
            acc = seq.iloc[start_c:end_c][self.ACC_COLS].values.astype(np.float32)
            rot = seq.iloc[start_c:end_c][self.ROT_COLS].values.astype(np.float32)
            tof = seq.iloc[start_c:end_c][self.tof_cols].values.astype(np.float32)

            rot = handle_quaternion_missing_values(rot)
            norms = np.linalg.norm(rot, axis=1, keepdims=True)
            norms[norms < 1e-6] = 1.0
            rot = rot / norms

            acc_world = compute_world_acceleration(acc, rot)
            acc_linear = remove_gravity_from_acc(acc, rot)
            angular_vel = calculate_angular_velocity_from_quat(rot)
            angular_dist = calculate_angular_distance(rot)

            acc_features = np.concatenate(
                [acc_world, acc_linear, angular_vel, angular_dist[:, None]], axis=1
            )

            therm = seq.iloc[start_c:end_c][self.therm_cols].values.astype(np.float32)
            
            # -----------------------
            # Padding
            # -----------------------
            acc_pad = np.zeros((self.window_len, acc_features.shape[1]), dtype=np.float32)
            rot_pad = np.zeros((self.window_len, 4), dtype=np.float32)
            tof_pad = np.zeros((self.window_len, self.tof_dim), dtype=np.float32)
            mask = np.zeros(self.window_len, dtype=np.bool_)

            insert = start_c - start
            actual_len = end_c - start_c
            
            acc_pad[insert:insert + len(acc_features)] = acc_features
            rot_pad[insert:insert + len(rot)] = rot
            mask[insert:insert + len(acc_features)] = True
            tof_pad[insert:insert + actual_len] = tof  # Pad ToF too

            therm_pad = np.zeros((self.window_len, 5), dtype=np.float32)
            therm_pad[insert:insert + actual_len] = therm

            # -----------------------
            # Label
            # -----------------------
            label = seq["gesture_id"].mode().iloc[0]

            #self.windows.append((acc_pad, rot_pad, tof_pad, mask, label))
            self.windows.append((acc_pad, rot_pad, tof_pad, therm_pad, mask, label))

        print(f"Created {len(self.windows)} multimodal windows")
        print(f"ToF dimension: {self.tof_dim}")  # Debug print

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        acc, rot, tof, therm, mask, label = self.windows[idx]

        if not USE_ACC:
            acc = np.zeros_like(acc, dtype=np.float32)
        if not USE_ROT:
            rot = np.zeros_like(rot, dtype=np.float32)

        return {
            "acc": torch.tensor(acc, dtype=torch.float32),
            "rot": torch.tensor(rot, dtype=torch.float32),
            "tof": torch.tensor(tof, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.bool),
            "label": torch.tensor(label, dtype=torch.long),
            "therm": torch.tensor(therm, dtype=torch.float32)
        }

# ============================================================
# MODEL
# ============================================================

class AxisCNN(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Conv1d(1, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )
    def forward(self,x):
        x = self.net(x)
        x = x.mean(dim=-1)
        return x

class EnhancedCNNEncoder(nn.Module):
    def __init__(self, input_dim, d_model=64, axis_hidden_dim=32):
        super().__init__()
        self.axis_cnns = nn.ModuleList([AxisCNN(hidden_dim=axis_hidden_dim) for _ in range(input_dim)])
        self.axis_proj = nn.Linear(input_dim*axis_hidden_dim, d_model)
        self.temporal_cnn = nn.Sequential(
            nn.Conv1d(d_model,d_model,3,padding=1), nn.BatchNorm1d(d_model), nn.ReLU(),
            nn.Conv1d(d_model,d_model,3,padding=1), nn.BatchNorm1d(d_model), nn.ReLU()
        )
        self.ln = nn.LayerNorm(d_model)
    def forward(self,x):
        B,T,F = x.shape
        feats = []
        for i in range(F):
            axis_data = x[:,:,i:i+1].transpose(1,2)
            feats.append(self.axis_cnns[i](axis_data))
        projected = self.axis_proj(torch.cat(feats,dim=1))
        expanded = projected.unsqueeze(1).repeat(1,T,1)
        expanded = self.temporal_cnn(expanded.transpose(1,2)).transpose(1,2)
        return self.ln(expanded)
        
# class ToFAutoencoder(nn.Module):
#     def __init__(self, input_dim, latent_dim=8, hidden_dim=128):
#         super().__init__()

#         self.encoder = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, latent_dim),
#         )

#         self.decoder = nn.Sequential(
#             nn.Linear(latent_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, input_dim),
#         )

#     def forward(self, x):
#         z = self.encoder(x)
#         recon = self.decoder(z)
#         return recon, z
class TemporalToFAutoencoder(nn.Module):
    def __init__(self, input_dim, seq_len, latent_dim=8, hidden_dim=128):
        super().__init__()
        
        # Encoder: [B, T, F] -> [B, latent_dim]
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(seq_len * input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, latent_dim),
        )
        
        # Decoder: [B, latent_dim] -> [B, T, F]
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, seq_len * input_dim),
        )
        self.seq_len = seq_len
        self.input_dim = input_dim
        
    def forward(self, x):
        B, T, F = x.shape
        z = self.encoder(x)  # [B, latent_dim]
        recon_flat = self.decoder(z)  # [B, T*F]
        recon = recon_flat.view(B, self.seq_len, self.input_dim)  # [B, T, F]
        return recon, z        
class TransformerEncoder(nn.Module):
    def __init__(self,d_model=64,nhead=4,layers=2):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model,nhead=nhead,batch_first=True,dim_feedforward=d_model*4,activation='gelu')
        self.encoder = nn.TransformerEncoder(enc_layer,layers)
    def forward(self,x,mask):
        x = self.encoder(x, src_key_padding_mask=~mask)
        m = mask.unsqueeze(-1).float()
        return (x*m).sum(dim=1)/m.sum(dim=1).clamp(min=1)

class HybridFusedModel(nn.Module):
    def __init__(self, num_classes, acc_feat_dim, rot_dim=4,
                 tof_dim=None, tof_latent_dim=8, d_model=64):
        super().__init__()
        self.use_acc = USE_ACC
        self.use_rot = USE_ROT
        self.use_tof = USE_TOF and (tof_dim is not None)  # Fix: check USE_TOF flag
        
        self.acc_encoder = EnhancedCNNEncoder(acc_feat_dim, d_model) if USE_ACC else None
        self.rot_encoder = EnhancedCNNEncoder(rot_dim, d_model) if USE_ROT else None
        self.acc_trans = TransformerEncoder(d_model) if USE_TRANSFORMER and USE_ACC else None
        self.rot_trans = TransformerEncoder(d_model) if USE_TRANSFORMER and USE_ROT else None
        self.therm_encoder = ThermEncoder(k_classes=8)
        self.therm_trans = TransformerEncoder(d_model)
        k_classes=8
        self.therm_proj = nn.Linear(k_classes, d_model)   # d_model = transformer dim
        
        # ToF autoencoder for temporal data
        if self.use_tof:
            self.tof_ae = TemporalToFAutoencoder(
                input_dim=tof_dim, 
                seq_len=WINDOW_BEFORE + WINDOW_AFTER + 1,  # window_len
                latent_dim=tof_latent_dim
            )
            self.tof_proj = nn.Linear(tof_latent_dim, d_model)
            self.tof_scale = nn.Parameter(torch.tensor(0.1))
        else:
            self.tof_ae = None
            self.tof_proj = None
            self.tof_scale = None

        # ------------------
        # Fusion + Head
        # ------------------
        # Count only ACC and ROT outputs (ToF is fused into them)
        #fusion_dim = d_model * (USE_ACC + USE_ROT)
        fusion_dim = d_model * (USE_ACC + USE_ROT + 1)   # +1 for therm
        
        self.fusion = nn.Sequential(
            nn.LayerNorm(fusion_dim),
            nn.Linear(fusion_dim, d_model),
            nn.GELU(),
            nn.Dropout(DROPOUT)
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(d_model // 2, num_classes)
        )

    def forward(self, acc, rot, therm, mask, tof=None):
        feats = []

        # ======================
        # ToF → token
        # ======================
        if self.use_tof and tof is not None:
            _, tof_latent = self.tof_ae(tof)              # [B, L]
            tof_token = self.tof_proj(tof_latent)         # [B, D]
            tof_token = tof_token.unsqueeze(1)            # [B, 1, D]
            tof_token = self.tof_scale * tof_token        # soft conditioning
        else:
            tof_token = None

        # ======================
        # ACC branch
        # ======================
        if self.use_acc:
            acc_feat = self.acc_encoder(acc)              # [B, T, D]

            if tof_token is not None:
                acc_feat = torch.cat([tof_token, acc_feat], dim=1)
                acc_mask = torch.cat(
                    [torch.ones(mask.size(0), 1, device=mask.device, dtype=torch.bool), mask],
                    dim=1
                )
            else:
                acc_mask = mask

            acc_out = self.acc_trans(acc_feat, acc_mask)
            feats.append(acc_out)

        # ======================
        # ROT branch
        # ======================
        if self.use_rot:
            rot_feat = self.rot_encoder(rot)

            if tof_token is not None:
                rot_feat = torch.cat([tof_token, rot_feat], dim=1)
                rot_mask = torch.cat(
                    [torch.ones(mask.size(0), 1, device=mask.device, dtype=torch.bool), mask],
                    dim=1
                )
            else:
                rot_mask = mask

            rot_out = self.rot_trans(rot_feat, rot_mask)
            feats.append(rot_out)

        # ======================
        # Therm
        # ======================
        #if self.use_therm and therm is not None:
        # therm: [B, T, 5]
        #therm_feat = self.therm_encoder(therm)     # [B, T, D] or [B, D]
        #therm_out = self.therm_trans(therm_feat)   # if transformer-based
        #feats.append(therm_out)

        # therm : [B, 5, T]
        #therm = therm.permute(0, 2, 1) # [B, T, 5]
        B, T, _ = therm.shape
        
        therm_flat = therm.reshape(B*T, 5)
        therm_logits = self.therm_encoder(therm_flat)
        therm_feat = therm_logits.reshape(B, T, -1)   # [B, T, k]
        therm_feat = self.therm_proj(therm_feat)      # [B, T, D]
        therm_mask = mask
        therm_out = self.therm_trans(therm_feat, therm_mask)
        
        feats.append(therm_out)

        # ======================
        # Fusion
        # ======================
        if len(feats) > 1:
            z = self.fusion(torch.cat(feats, dim=1))
        else:
            z = feats[0]

        return self.head(z)
        
# ============================================================
# TRAINING
# ============================================================
WINDOW_BEFORE = 20
WINDOW_AFTER = 20
EPOCHS = 20
USE_AXISCNN = False
USE_TRANSFORMER = True
FUSION = 'concat'  # 'concat' or 'gated'
USE_ACC = True
USE_ROT = True
USE_TOF = True

torch.manual_seed(SEED)
df = pd.read_csv(CSV_PATH)
therm_cols = ["thm_1","thm_2","thm_3","thm_4","thm_5"]
df[therm_cols] = df[therm_cols].fillna(0.0)
dataset = AccRotHybridDataset(df,WINDOW_BEFORE,WINDOW_AFTER)
num_classes = len(dataset.gesture_classes)
acc_feat_dim = dataset[0]["acc"].shape[1] if USE_ACC else 0
tof_feat_dim = dataset.tof_dim if USE_TOF else 0

train_size = int((1-VAL_RATIO)*len(dataset)); test_size=len(dataset)-train_size
train_ds, test_ds = random_split(dataset,[train_size,test_size])

train_loader = DataLoader(train_ds,batch_size=BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_ds,batch_size=BATCH_SIZE)

model = HybridFusedModel(
    num_classes=num_classes,
    acc_feat_dim=acc_feat_dim,
    tof_dim=tof_feat_dim ).to(DEVICE)
optimizer = torch.optim.AdamW(model.parameters(),lr=LR)
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train(); total_loss=0
    for batch in train_loader:
        acc=batch["acc"].to(DEVICE) if USE_ACC else None
        rot=batch["rot"].to(DEVICE) if USE_ROT else None
        tof=batch["tof"].to(DEVICE) if USE_TOF else None
        therm = batch["therm"].to(DEVICE)
        mask=batch["mask"].to(DEVICE); label=batch["label"].to(DEVICE)
        
        optimizer.zero_grad()
        logits = model(acc,rot,therm,mask,tof)
        loss = criterion(logits,label)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
    print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f}")

# ============================================================
# EVALUATION
# ============================================================


model.eval()
all_preds, all_labels=[],[]
with torch.no_grad():
    for batch in test_loader:
        acc=batch["acc"].to(DEVICE) if USE_ACC else None
        rot=batch["rot"].to(DEVICE) if USE_ROT else None
        tof=batch["tof"].to(DEVICE) if USE_TOF else None
        mask=batch["mask"].to(DEVICE)
        
        logits = model(acc,rot,therm,mask,tof)
        preds = logits.argmax(dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch["label"].numpy())

all_preds=np.array(all_preds); all_labels=np.array(all_labels)
overall_acc = (all_preds==all_labels).mean()
print(f"\nOverall accuracy: {overall_acc:.3f}")


