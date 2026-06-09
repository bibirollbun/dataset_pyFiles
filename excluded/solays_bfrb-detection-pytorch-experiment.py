import numpy as np
import pandas as pd
import polars as pl
import math
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)


# Paths to data
train_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv"
test_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv"
demographic_train_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv"
demographic_test_path = "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv"

# Read data
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
demographic_train_df = pd.read_csv(demographic_train_path)
demographic_test_df = pd.read_csv(demographic_test_path)


# join both datasets
train_df = train_df.merge(demographic_train_df, on="subject", how="left")
test_df = test_df.merge(demographic_test_df, on="subject", how="left")


train_df.filter(items=['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']).describe()


train_df.filter(like='tof_').describe()


train_df.filter(like='thm_').describe()


# Create a mask for rows where any 'thm_' column is below the threshold
mask = (train_df.filter(like='thm_') < 10).any(axis=1)
filtered_df = train_df[mask]

# Get all unique sequence IDs with values below the threshold
seqs_with_low_values = list(filtered_df['sequence_id'].unique())

print(f"There are {len(seqs_with_low_values)} sequences with erroneous thermopile values.")


# Select 4 random sequence_ids
sampled_seqs = random.sample(seqs_with_low_values, 4)

fig, axes = plt.subplots(2, 2, figsize=(10, 4))
axes = axes.flatten()

for idx, seq_id in enumerate(sampled_seqs):
    ax = axes[idx]
    seq_df = train_df[train_df['sequence_id'] == seq_id]
    
    for i in range(1, 6):
        ax.plot(seq_df['sequence_counter'], seq_df[f'thm_{i}'], label=f'thm_{i}', linewidth=0.8)
    
    ax.set_title(f"Seq {seq_id}")
    ax.set_xlabel("sequence_counter")
    ax.set_ylabel("thm values")
    ax.grid(True)

# Hide any unused axes (if fewer than 4)
for j in range(len(sampled_seqs), 4):
    axes[j].axis('off')

fig.tight_layout()
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
plt.show()


class CMISequenceDataset(Dataset):
    """
    Dataset for CMI sensor data sequences. Expects a CSV with columns including:
    'sequence_id', sensor features, and label.
    """
    def __init__(self, df, is_train=True):
        """
        df: DataFrame containing all rows (time-steps) with columns including 'sequence_id'.
        is_train: whether labels are present.
        """
        self.is_train = is_train

        # Determine feature cols
        self.feature_cols = [c for c in df.columns if c not in ['sequence_id', 'gesture']]
        
        # Determine label column
        label_col = 'gesture'
        
        # Group rows by sequence_id
        self.seqs = []
        self.labels = []
        for seq_id, group in df.groupby('sequence_id'):            
            # Extract feature array for this sequence
            seq_features = group[self.feature_cols].values.astype(np.float32)
            self.seqs.append(seq_features)
            
            # If training, get the label for this sequence (assumed constant per sequence)
            if self.is_train:
                label = group[label_col].iloc[0]
                self.labels.append(label)
        
    def __len__(self):
        return len(self.seqs)
    
    def __getitem__(self, idx):
        seq = self.seqs[idx]  # shape (time_steps, num_features)
        
        if np.isnan(seq).any():
            print(f"NaN found in sequence index {idx}")

        if self.is_train:
            label = self.labels[idx]
            return torch.from_numpy(seq), torch.tensor(label, dtype=torch.long)
        else:
            return torch.from_numpy(seq), None


def collate_fn(batch):
    """
    Collate function to pad sequences to the same length within a batch.
    Returns:
        padded_seqs: tensor of shape (batch_size, max_len, num_features)
        labels: tensor of shape (batch_size) or None for test
        mask: boolean mask (batch_size, max_len) with True at padded positions
    """
    sequences, labels = zip(*batch)
    lengths = [seq.shape[0] for seq in sequences]
    max_len = max(lengths)
    # Number of features (all sequences have same feature count)
    num_features = sequences[0].shape[1]
    batch_size = len(sequences)
    padded = torch.zeros(batch_size, max_len, num_features, dtype=torch.float32)
    mask = torch.ones(batch_size, max_len, dtype=torch.bool)  # True indicates padding
    for i, seq in enumerate(sequences):
        length = seq.shape[0]
        padded[i, :length, :] = seq
        mask[i, :length] = False  # False for actual data
    if labels[0] is not None:
        labels = torch.tensor(labels, dtype=torch.long)
    else:
        labels = None
    return padded, labels, mask


def pre_processing(df, is_train=True, encoders=None, scaler=None):
    # Drop irrelevant columns
    cols_to_drop = ['row_id', 'sequence_type', 'behavior', 'subject', 'orientation', 'phase']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Section to deal with the missing values ------------------------------------------------------
    # Replace NaNs with -1 for all tof_* columns
    tof_cols = [c for c in df.columns if c.startswith("tof_")]
    df[tof_cols] = df[tof_cols].fillna(-1)
    
    # thm columns also have nan, we will replace with a mean value
    thm_cols = [c for c in df.columns if c.startswith("thm_")]
    df[thm_cols] = df[thm_cols].fillna(df[thm_cols].mean())
    
    # rot columns have nan, we will fill with the a null quaternion {1, 0, 0, 0}
    rot_cols = [c for c in df.columns if c.startswith("rot_")]
    rot_fill = {'rot_w': 1.0, 'rot_x': 0.0, 'rot_y': 0.0, 'rot_z': 0.0}
    df[rot_cols] = df[rot_cols].fillna(rot_fill)
    
    # Section to deal with the erroneous sensor values ---------------------------------------------
    # Identify sequences with any thm values < 10
    threshold = 10
    mask = (df.filter(like='thm_') < threshold).any(axis=1)
    filtered_df = df[mask]
    seqs_with_low_values = list(filtered_df['sequence_id'].unique())

    # Fix thm_3 if all values are 0 in the sequence
    for seq_id in seqs_with_low_values:
        seq_mask = df['sequence_id'] == seq_id
        if (df.loc[seq_mask, 'thm_3'] < threshold).all():
            df.loc[seq_mask, 'thm_3'] = df.loc[seq_mask, 'thm_2']
        if (df.loc[seq_mask, 'thm_1'] < threshold).all():
            df.loc[seq_mask, 'thm_1'] = df.loc[seq_mask, 'thm_2']
        
    # Find categorical columns to encode
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    if is_train:
        encoders = {}
        for col in categorical_cols:
            if col == 'sequence_id':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            else:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])        
                encoders[col] = le
    elif encoders is not None and not is_train:
        for col in categorical_cols:
            if col == 'sequence_id':
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col])
            else:
                le = encoders[col]
                df[col] = le.transform(df[col])
    
    # Identify sensor columns (customise this pattern if needed)
    sensor_prefixes = ['acc_', 'thm_', 'tof_', 'rot_']
    cols_to_scale = [c for c in df.columns if any(c.startswith(p) for p in sensor_prefixes)]
    cols_to_scale.extend(['age', 'height_cm', 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm'])

    # Compute all new features
    acc_mag = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    rot_mag = np.sqrt(df['rot_x']**2 + df['rot_y']**2 + df['rot_z']**2)
    
    w, x, y, z = df['rot_w'], df['rot_x'], df['rot_y'], df['rot_z']
    euler_roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x**2 + y**2))
    euler_pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1.0, 1.0))
    euler_yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y**2 + z**2))
    
    # Grouped angular velocity (recommended)
    angular_velocity_x = df.groupby('sequence_id')['rot_x'].diff().fillna(0)
    angular_velocity_y = df.groupby('sequence_id')['rot_y'].diff().fillna(0)
    angular_velocity_z = df.groupby('sequence_id')['rot_z'].diff().fillna(0)
    
    # Combine all new features into one DataFrame
    new_features = pd.DataFrame({
        'acc_mag': acc_mag,
        'rot_mag': rot_mag,
        'euler_roll': euler_roll,
        'euler_pitch': euler_pitch,
        'euler_yaw': euler_yaw,
        'angular_velocity_x': angular_velocity_x,
        'angular_velocity_y': angular_velocity_y,
        'angular_velocity_z': angular_velocity_z
    })

    cols_to_scale.extend(['acc_mag', 'rot_mag', 'euler_roll', 'euler_pitch', 'euler_yaw', 'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z'])

    # Concatenate all at once
    df = pd.concat([df, new_features], axis=1).copy()

    # Fit scaler on training split only
    if is_train:
        scaler = StandardScaler()
        scaler.fit(df[cols_to_scale])
    
    # Apply to all splits
    df[cols_to_scale] = scaler.transform(df[cols_to_scale])
            
    return df, encoders, scaler


train_df, encoders, scaler = pre_processing(train_df)
test_df, _, _ = pre_processing(test_df, is_train=False, encoders=encoders, scaler=scaler)


# Split train into train/validation sets (e.g., 80/20 split)
train_df_part, val_df = train_test_split(train_df, test_size=0.2, random_state=42)


# Create datasets and data loaders
train_dataset = CMISequenceDataset(train_df_part, is_train=True)
# Use the same label encoder for validation and test to ensure consistent class mapping
val_dataset = CMISequenceDataset(val_df, is_train=True)
# For test set, we do not have labels
test_dataset = CMISequenceDataset(test_df, is_train=False)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                          collate_fn=collate_fn, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                        collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                         collate_fn=collate_fn)


class CNNTransformer(nn.Module):
    """
    A lightweight CNN + Transformer model for time-series classification.
    """
    def __init__(self, input_dim, num_classes, hidden_dim=128, n_heads=4, n_layers=2, dropout=0.3):
        super(CNNTransformer, self).__init__()
        # 1D CNN layers to preprocess the time-series (fold time steps).
        # Input shape to CNN: (batch, input_dim, seq_len)
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, 
                               kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        # Transformer Encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=n_heads, 
                                                   dim_feedforward=hidden_dim*4, 
                                                   activation='relu')
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        # Classification head
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, src_key_padding_mask=None):
        """
        x: tensor of shape (batch_size, seq_len, input_dim)
        src_key_padding_mask: bool mask (batch_size, seq_len) True for padding
        """
        # Prepare for conv: (batch, input_dim, seq_len)
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))  # downsample time by ~2
        x = self.dropout(x)
        # x now (batch, hidden_dim, new_seq_len)
        # Prepare for transformer: (new_seq_len, batch, hidden_dim)
        x = x.permute(2, 0, 1)
        # Adjust mask for new length
        if src_key_padding_mask is not None:
            # Original mask (batch, seq_len) -> after conv pooling, we create a new mask.
            # For simplicity, create no further mask (this is an approximation).
            # One could recompute mask by checking if the entire pooled input was padding.
            new_mask = None
        else:
            new_mask = None
        # Transformer encoding
        x = self.transformer(x, src_key_padding_mask=new_mask)
        # x shape: (new_seq_len, batch, hidden_dim) -> (batch, new_seq_len, hidden_dim)
        x = x.permute(1, 0, 2)
        # Mean pooling over time dimension (ignoring any padding by ~mask)
        # For simplicity, we'll take the average of all time steps (more elaborate masking possible).
        x = x.mean(dim=1)  # shape (batch, hidden_dim)
        logits = self.classifier(x)  # shape (batch, num_classes)
        return logits


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    for batch in loader:
        seqs, labels, mask = batch
        seqs = seqs.to(device)
        labels = labels.to(device)
        # Forward pass
        logits = model(seqs)
        loss = criterion(logits, labels)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * seqs.size(0)
        # Track training metrics
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            seqs, labels, mask = batch
            seqs = seqs.to(device)
            labels = labels.to(device)
            logits = model(seqs)
            loss = criterion(logits, labels)
            total_loss += loss.item() * seqs.size(0)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.cpu().numpy().tolist())
    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    # Also compute F1 as an example (macro)
    f1 = f1_score(all_labels, all_preds, average='macro')
    return avg_loss, acc, f1


# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_dim = len(train_dataset.feature_cols)
num_classes = len(set(train_dataset.labels))
model = CNNTransformer(input_dim=input_dim, num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# Optionally, add a learning rate scheduler
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

num_epochs = 30
best_val_f1 = 0.0
for epoch in range(num_epochs):
    train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
    val_loss, val_acc, val_f1 = validate(model, val_loader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}: "
          f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
          f"Val Loss={val_loss:.4f}, Acc={val_acc:.4f}, F1={val_f1:.4f}")
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
    scheduler.step()

# Inference on test set
model.eval()
sequence_ids = []
preds_labels = []
with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        seqs, _, mask = batch
        seqs = seqs.to(device)
        logits = model(seqs, src_key_padding_mask=mask.to(device))
        preds = logits.argmax(dim=1).cpu().numpy()
        preds_labels.extend(preds.tolist())
        # Collect sequence IDs for this batch
        # We need to map batch order to sequence IDs; here we assume test_dataset.seqs preserves order of test_df
        batch_ids = test_df['sequence_id'].unique()[batch_idx*test_loader.batch_size : (batch_idx+1)*test_loader.batch_size]
        sequence_ids.extend(batch_ids.tolist())


# Map numeric predictions back to gesture names
inv_label_map = {i: lbl for i, lbl in enumerate(encoders['gesture'].classes_)}
predicted_gestures = [inv_label_map.get(label, "Non-Target") for label in preds_labels]
submission = pd.DataFrame({
    "sequence_id": sequence_ids,
    "gesture": predicted_gestures
})

submission


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    # Convert polars to pandas
    seq_df = sequence.to_pandas()
    demographics_df = demographics.to_pandas()

    seq_df = seq_df.merge(demographics_df, on="subject", how="left")

    X, _, _ = pre_processing(seq_df, is_train=False, encoders=encoders, scaler=scaler)
    
    # drop unnecessary columns
    cols_to_drop = ['sequence_id']
    X = X.drop(columns=cols_to_drop)

    X = X.values.astype(np.float32)
    
    # Convert to torch tensor
    x_tensor = torch.tensor(X).unsqueeze(0).to(device)
    mask = torch.zeros(1, x_tensor.shape[1], dtype=torch.bool).to(device)
    
    # Inference
    model.eval()
    with torch.no_grad():
        logits = model(x_tensor, src_key_padding_mask=mask)
        pred_idx = torch.argmax(logits, dim=1).item()
    
    # Map back to label string
    pred_class = encoders['gesture'].inverse_transform([pred_idx])[0]
    return pred_class


import os
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

