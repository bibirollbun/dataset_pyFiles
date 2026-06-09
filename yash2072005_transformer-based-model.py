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


# --------------------------------------------
# Full Transformer-based Model Pipeline for CMI
# --------------------------------------------

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from kaggle_evaluation.cmi_inference_server import CMIInferenceServer


# -------------------------------
# Configuration
# -------------------------------
SEQUENCE_LENGTH = 300
BATCH_SIZE = 32
EPOCHS = 40
LEARNING_RATE = 1e-4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------------
# Features to Use
# -------------------------------
features = [
    'acc_x', 'acc_y', 'acc_z',
    'rot_w', 'rot_x', 'rot_y', 'rot_z',
    'thm_1', 'thm_2', 'thm_3', 'thm_4', 'thm_5',
] + [f"tof_{i}_v{j}" for i in range(1, 6) for j in range(64)]  # Optional: comment out ToF for faster training


# -------------------------------
# Load and Prepare Train Data
# -------------------------------
train = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
ALL_GESTURES = sorted(train["gesture"].dropna().unique())
le = LabelEncoder()
le.fit(ALL_GESTURES)


# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)


# -------------------------------
# Dataset Class
# -------------------------------
class SequenceDataset(Dataset):
    def __init__(self, df, label_encoder):
        self.sequences = []
        self.labels = []
        for sequence_id, group in df.groupby("sequence_id"):
            X = group[features].fillna(0).to_numpy(dtype=np.float32)
            if X.shape[0] < SEQUENCE_LENGTH:
                pad_len = SEQUENCE_LENGTH - X.shape[0]
                X = np.pad(X, ((pad_len, 0), (0, 0)), mode="constant")
            else:
                X = X[-SEQUENCE_LENGTH:]
            self.sequences.append(X)
            self.labels.append(group["gesture"].iloc[0])
        self.labels = label_encoder.transform(self.labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx]), torch.tensor(self.labels[idx])


# -------------------------------
# Transformer Model Definition
# -------------------------------
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.attn = nn.Linear(d_model, 1)

    def forward(self, x):
        weights = torch.softmax(self.attn(x), dim=1)  # [B, T, 1]
        return (x * weights).sum(dim=1)  # Weighted sum


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 128)
        self.pos_encoder = PositionalEncoding(128)  # ✅ Added here
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=num_heads, batch_first=True, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.attn_pool = AttentionPooling(128)  # ✅ Attention pooling
        self.classifier = nn.Sequential(
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoder(x)  # ✅ Now it exists
        x = self.transformer(x)
        x = self.attn_pool(x)    # ✅ Smarter pooling
        return self.classifier(x)



# -------------------------------
# Training
# -------------------------------
from torch.amp import autocast, GradScaler  # For mixed precision

scaler = GradScaler()  # Helps stability in mixed precision

train_dataset = SequenceDataset(train, le)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = TransformerClassifier(
    input_dim=len(features), 
    num_classes=len(le.classes_)
).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)  # AdamW > Adam for transformers
criterion = nn.CrossEntropyLoss()

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad(set_to_none=True)  # Slight speedup
        
        # Mixed precision training
        with autocast(device_type=device.type, dtype=torch.float16):
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
        
        scaler.scale(loss).backward()
        
        # Clip gradients for stability
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss/len(train_loader):.4f}")

# Save final model
torch.save(model.state_dict(), "transformer_model.pt")



# Prediction function
def predict(sequence: pd.DataFrame, demographics: pd.DataFrame) -> str:
    if not isinstance(sequence, pd.DataFrame):
        sequence = sequence.to_pandas()

    # Select features and fill missing
    for col in features:
        if col not in sequence.columns:
            sequence[col] = 0
    X = sequence[features].fillna(0).to_numpy(dtype=np.float32)

    # Pad or truncate
    if len(X) < SEQUENCE_LENGTH:
        pad_len = SEQUENCE_LENGTH - len(X)
        X = np.pad(X, ((pad_len, 0), (0, 0)), mode="constant")
    else:
        X = X[-SEQUENCE_LENGTH:]

    # Predict
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(X_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_label = le.inverse_transform([pred_idx])[0]

    return str(pred_label)


# Inference Server
inference_server = CMIInferenceServer(predict)

if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
            "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"
        )
    )










