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



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.manifold import TSNE
import warnings
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for Kaggle
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


def load_and_prepare_data():
    """Loads and preprocesses data, returning dataframes and imputation values."""
    print("\n--- Loading and Preparing Data ---")
    
    # Load raw data
    train_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv')
    train_demographics = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv')
    test_df = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv')
    test_demographics = pd.read_csv('/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv')
 
    print(f"Initial train shape: {train_df.shape}")
    print(f"Initial test shape: {test_df.shape}")
    
    # Merge demographics
    train_df = train_df.merge(train_demographics, on='subject', how='left')
    test_df = test_df.merge(test_demographics, on='subject', how='left')
    
    # Prepare imputation dictionary from training data
    demo_cols = ['adult_child', 'age', 'sex', 'handedness', 'height_cm', 
                 'shoulder_to_wrist_cm', 'elbow_to_wrist_cm']
    imputation_values = {}
    for col in demo_cols:
        if col in train_df.columns:
            if col in ['adult_child', 'sex', 'handedness']:
                mode_val = train_df[col].mode()[0] if not train_df[col].mode().empty else 0
                imputation_values[col] = mode_val
            else:
                median_val = train_df[col].median()
                imputation_values[col] = median_val

    # Apply imputation to both train and test sets
    for df in [train_df, test_df]:
        for col, value in imputation_values.items():
            if col in df.columns:
                df[col].fillna(value, inplace=True)
    
    # Handle sensor data missing values
    sensor_cols = [col for col in train_df.columns if any(col.startswith(x) for x in ['acc_', 'rot_', 'thm_', 'tof_'])]
    
    for df in [train_df, test_df]:
        for col in sensor_cols:
            if col in df.columns:
                if col.startswith('tof_'):
                    df[col].fillna(-1, inplace=True)  # ToF: -1 for no signal
                else:
                    # IMU and Thermopile: forward/backward fill within each sequence
                    df[col] = df.groupby('sequence_id')[col].fillna(method='ffill').fillna(method='bfill')
                    df[col].fillna(0, inplace=True)
    
    print("Data loading and preprocessing complete.")
    return train_df, test_df, imputation_values

# Load the data
train_df, test_df, imputation_values = load_and_prepare_data()




def create_sequences(df, max_length=150, min_length=5):
    """Creates time-series sequences from the dataframe."""
    sequences = []
    labels = []
    sequence_ids = []
    
    imu_cols = [col for col in df.columns if col.startswith(('acc_', 'rot_'))]
    thermal_cols = [col for col in df.columns if col.startswith('thm_')]
    tof_cols = [col for col in df.columns if col.startswith('tof_')]
    sensor_cols = imu_cols + thermal_cols + tof_cols
    
    print(f"\n--- Creating Sequences ---")
    print(f"Sensor features: IMU={len(imu_cols)}, Thermal={len(thermal_cols)}, ToF={len(tof_cols)}")
    
    for seq_id in df['sequence_id'].unique():
        seq_data = df[df['sequence_id'] == seq_id].sort_values('sequence_counter')
        
        if len(seq_data) < min_length:
            continue
            
        sensor_values = seq_data[sensor_cols].values.astype(np.float32)
        sensor_values = np.nan_to_num(sensor_values, nan=0.0)
        sensor_values[sensor_values == -1] = 0
        
        if len(sensor_values) > max_length:
            sensor_values = sensor_values[:max_length]
        
        sequences.append(sensor_values)
        sequence_ids.append(seq_id)
        
        if 'gesture' in seq_data.columns:
            labels.append(seq_data['gesture'].iloc[0])
    
    print(f"Created {len(sequences)} sequences.")
    return sequences, labels, sequence_ids, len(imu_cols), len(thermal_cols), len(tof_cols)

# Create train and test sequences
train_sequences, train_labels, train_seq_ids, n_imu, n_thermal, n_tof = create_sequences(train_df)
# For test sequences, we don't have a minimum length
test_sequences, _, test_seq_ids, _, _, _ = create_sequences(test_df, min_length=1)




class ConvBlock(nn.Module):
    """A 1D convolutional block with BatchNorm and activation."""
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, sequence, features)
        x = x.permute(0, 2, 1)  # -> (batch, features, sequence)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x.permute(0, 2, 1)  # -> (batch, sequence, features)

class TemporalAttentionModel(nn.Module):
    """A temporal attention model with 1D Convolutions for sensor fusion."""
    
    def __init__(self, n_imu, n_thermal, n_tof, num_classes, hidden_dim=128, conv_out_dim=64):
        super().__init__()
        self.n_imu = n_imu
        self.n_thermal = n_thermal  
        self.n_tof = n_tof
        
        # 1D Convolutional Blocks to extract features
        self.imu_conv = ConvBlock(n_imu, conv_out_dim)
        self.thermal_conv = ConvBlock(n_thermal, conv_out_dim)
        self.tof_conv = ConvBlock(n_tof, conv_out_dim)
        
        # LSTMs to process sequences of convolved features
        self.imu_encoder = nn.LSTM(conv_out_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.thermal_encoder = nn.LSTM(conv_out_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        self.tof_encoder = nn.LSTM(conv_out_dim, hidden_dim//2, batch_first=True, bidirectional=True)
        
        # New learnable fusion layer
        self.fusion_layer = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=True)
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim//2, num_classes)
        )
        
    def forward(self, x, mask=None, return_embedding=False):
        imu_data = x[:, :, :self.n_imu]
        thermal_data = x[:, :, self.n_imu:self.n_imu+self.n_thermal]
        tof_data = x[:, :, self.n_imu+self.n_thermal:self.n_imu+self.n_thermal+self.n_tof]
        
        # Apply convolutions
        imu_conv_out = self.imu_conv(imu_data)
        thermal_conv_out = self.thermal_conv(thermal_data)
        tof_conv_out = self.tof_conv(tof_data)

        # Apply LSTMs
        imu_out, _ = self.imu_encoder(imu_conv_out)
        thermal_out, _ = self.thermal_encoder(thermal_conv_out) 
        tof_out, _ = self.tof_encoder(tof_conv_out)
        
        # Concatenate sensor outputs instead of averaging
        concatenated_out = torch.cat((imu_out, thermal_out, tof_out), dim=2)
        
        # Apply the new fusion layer
        fused = self.fusion_layer(concatenated_out)
        
        attended, _ = self.attention(fused, fused, fused, key_padding_mask=mask)
        
        if mask is not None:
            mask_expanded = (~mask).unsqueeze(-1).float()
            pooled = (attended * mask_expanded).sum(dim=1) / mask_expanded.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = attended.mean(dim=1)
        
        output = self.classifier(pooled)

        if return_embedding:
            return output, pooled
        return output

print("\Model success.")



MAX_LENGTH = 150

class CMIDataset(Dataset):
    """Custom PyTorch dataset for CMI data."""
    def __init__(self, sequences, labels=None, max_length=MAX_LENGTH):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = len(seq)
        
        if seq_len < self.max_length:
            padding = np.zeros((self.max_length - seq_len, seq.shape[1]), dtype=np.float32)
            seq = np.vstack([seq, padding])
            mask = np.concatenate([np.zeros(seq_len), np.ones(self.max_length - seq_len)])
        else:
            seq = seq[:self.max_length]
            mask = np.zeros(self.max_length)
        
        result = {
            'sequence': torch.tensor(seq, dtype=torch.float32),
            'mask': torch.tensor(mask, dtype=torch.bool)
        }
        
        if self.labels is not None:
            result['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
            
        return result

# Prepare label encoder
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(train_labels)
num_classes = len(label_encoder.classes_)
print(f"\nFound {num_classes} gesture classes.")

# Create datasets and dataloaders
train_dataset = CMIDataset(train_sequences, encoded_labels)
train_idx, val_idx = train_test_split(range(len(train_dataset)), test_size=0.2, random_state=42, stratify=encoded_labels)
train_subset = torch.utils.data.Subset(train_dataset, train_idx)
val_subset = torch.utils.data.Subset(train_dataset, val_idx)

train_loader = DataLoader(train_subset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
print(f"Train samples: {len(train_subset)}, Validation samples: {len(val_subset)}")




train_subset_labels = [encoded_labels[i] for i in train_idx]
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_subset_labels),
    y=train_subset_labels
)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
print(f"\nCalculated class weights to handle data imbalance.")

model = TemporalAttentionModel(n_imu, n_thermal, n_tof, num_classes).to(device)
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.02)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for batch in loader:
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences, masks)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    return total_loss / len(loader), 100.0 * correct / total

def validate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in loader:
            sequences = batch['sequence'].to(device)
            labels = batch['label'].to(device)
            masks = batch['mask'].to(device)
            
            outputs = model(sequences, masks)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    f1 = f1_score(all_labels, all_preds, average='weighted')
    return total_loss / len(loader), f1

print("\n--- Starting Training ---")
best_val_loss = float('inf')
patience, no_improve = 10, 0

# Lists to store metrics for plotting
history = {
    'train_loss': [],
    'val_loss': [],
    'val_f1': []
}

for epoch in range(50):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    val_loss, val_f1 = validate(model, val_loader, criterion)
    scheduler.step()

    history['train_loss'].append(train_loss)
    history['val_loss'].append(val_loss)
    history['val_f1'].append(val_f1)
    
    print(f'Epoch {epoch+1:02d}/50 | Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f}, F1: {val_f1:.4f}')
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_f1_at_min_loss = val_f1
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'  -> Validation loss decreased. Model saved. Val F1: {val_f1:.4f}')
        no_improve = 0
    else:
        no_improve += 1
        
    if no_improve >= patience:
        print(f"Early stopping after {patience} epochs with no improvement in validation loss.")
        break
print(f"Finished training. Best validation loss: {best_val_loss:.4f}, with F1 score: {best_f1_at_min_loss:.4f}")




print("\n--- Generating Analysis and Visualizations ---")

# --- Plotting Training History ---
epochs_ran = len(history['train_loss'])
plt.figure(figsize=(12, 5))

# Loss Curve
plt.subplot(1, 2, 1)
plt.plot(range(1, epochs_ran + 1), history['train_loss'], label='Train Loss')
plt.plot(range(1, epochs_ran + 1), history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# F1 Score Curve
plt.subplot(1, 2, 2)
plt.plot(range(1, epochs_ran + 1), history['val_f1'], label='Validation F1 Score', color='orange')
plt.title('Validation F1 Score')
plt.xlabel('Epochs')
plt.ylabel('F1 Score')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('training_history.png')
print("Saved training history plot to training_history.png")
plt.close()


# --- Detailed Report on Best Model ---
print("\n--- Detailed Report on Best Model on Validation Set ---")
# Load the best model
model.load_state_dict(torch.load('best_model.pth'))
model.eval()

# Get predictions on the validation set
all_preds = []
all_labels = []
all_embeddings = []
with torch.no_grad():
    for batch in val_loader:
        sequences = batch['sequence'].to(device)
        labels = batch['label'].to(device)
        masks = batch['mask'].to(device)
        outputs, embeddings = model(sequences, masks, return_embedding=True)
        _, predicted = outputs.max(1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_embeddings.append(embeddings.cpu().numpy())

all_embeddings = np.concatenate(all_embeddings, axis=0)

# Print Classification Report
class_names = label_encoder.classes_
print(classification_report(all_labels, all_preds, target_names=class_names))

# Plot Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title('Confusion Matrix for Best Model')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.savefig('confusion_matrix.png')
print("Saved confusion matrix plot to confusion_matrix.png")
plt.close()

# --- t-SNE Visualization ---
print("\n--- Generating t-SNE visualization of embeddings ---")
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
tsne_results = tsne.fit_transform(all_embeddings)

plt.figure(figsize=(16, 12))
scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=all_labels, cmap='viridis', alpha=0.7)
plt.title('t-SNE Visualization of Validation Set Embeddings')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
# Create a legend
legend1 = plt.legend(handles=scatter.legend_elements(num=len(class_names))[0], labels=list(class_names), title="Classes")
plt.gca().add_artist(legend1)
plt.grid(True)
plt.savefig('tsne_embeddings.png')
print("Saved t-SNE plot to tsne_embeddings.png")
plt.close()




import polars as pl
import kaggle_evaluation.cmi_inference_server
import sys
import traceback

print("\n--- Setting up Inference Server ---")

# Load the best model trained in the previous steps
model.load_state_dict(torch.load('best_model.pth'))
model.to(device)
model.eval()
print("Best model loaded for inference.")

# Get the list of sensor columns used during training
sensor_cols = [col for col in train_df.columns if any(col.startswith(x) for x in ['acc_', 'rot_', 'thm_', 'tof_'])]

def preprocess_for_inference(df_seq, df_demo):
    """Preprocesses a single sequence for inference, mirroring the training logic."""
    df = df_seq.merge(df_demo, on='subject', how='left')

    # Fill demo NaNs using pre-calculated values from the training set
    for col, value in imputation_values.items():
        if col in df.columns:
            df[col].fillna(value, inplace=True)
            
    # Fill any remaining NaNs (if columns were missing in original test_demo)
    demo_cols_from_train = list(imputation_values.keys())
    for col in demo_cols_from_train:
        if col in df.columns:
            df[col].fillna(0, inplace=True)

    # Fill sensor NaNs
    for col in sensor_cols:
        if col in df.columns:
            df[col] = df[col].ffill().bfill() # ffill/bfill for single sequence
            df[col].fillna(0, inplace=True)
        else:
            # If a sensor column from training is missing, add it with zeros
            # matching the length of the current sequence.
            df[col] = np.zeros(len(df_seq))

    # Extract sensor data in the correct order
    sensor_data_list = [df[col].values for col in sensor_cols]
    sensor_data = np.stack(sensor_data_list, axis=1).astype(np.float32)
    sensor_data = np.nan_to_num(sensor_data, nan=0.0)
    sensor_data[sensor_data == -1] = 0

    # Padding / Truncating
    seq_len = len(sensor_data)
    mask = np.zeros(MAX_LENGTH, dtype=bool)

    if seq_len > MAX_LENGTH:
        sensor_data = sensor_data[:MAX_LENGTH]
    elif seq_len < MAX_LENGTH:
        padding = np.zeros((MAX_LENGTH - seq_len, sensor_data.shape[1]), dtype=np.float32)
        sensor_data = np.vstack([sensor_data, padding])
        mask[seq_len:] = True

    return torch.tensor(sensor_data, dtype=torch.float32).unsqueeze(0), torch.tensor(mask, dtype=torch.bool).unsqueeze(0)


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    """The main prediction function for the Kaggle server.
    This function is wrapped in a try/except block to be robust against all errors.
    """
    try:
        df_seq = sequence.to_pandas()
        df_demo = demographics.to_pandas()

        sequence_tensor, mask_tensor = preprocess_for_inference(df_seq, df_demo)
        sequence_tensor = sequence_tensor.to(device)
        mask_tensor = mask_tensor.to(device)

        with torch.no_grad():
            outputs = model(sequence_tensor, mask_tensor)
            _, predicted_idx = outputs.max(1)
            prediction = label_encoder.inverse_transform(predicted_idx.cpu().numpy())[0]
        
        return str(prediction)

    except Exception as e:
        # This block makes the function robust. In case of any failure,
        # it prints the error to stderr (for debugging if visible) and
        # returns a default valid gesture. This prevents the gateway from crashing.
        print("--- PREDICTION FAILED ---", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        print("--------------------------", file=sys.stderr)
        return "Drink"


# --- Start Server ---
# This part of the code will only run when submitted to Kaggle.
# For local testing, it will run a local gateway.
inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    print("Competition environment detected. Starting server...")
    inference_server.serve()
else:
    print("Local environment detected. Running local gateway for testing...")
    # The local gateway simulates the Kaggle environment with the provided test files.
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )
    print("Local gateway finished.")

