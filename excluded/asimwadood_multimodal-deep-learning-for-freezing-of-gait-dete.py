import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



# List all files in the competition directory
competition_files = os.listdir('../input/tlvmc-parkinsons-freezing-gait-prediction')
print("Files available in the competition dataset:")
print(competition_files)


def sliding_window(data, window_size, overlap):
    """
    Splits the data into overlapping windows.
    data: numpy array with shape (num_samples, num_features)
    window_size: number of samples per window.
    overlap: fraction overlap between consecutive windows (e.g., 0.5 for 50% overlap).
    """
    step = int(window_size * (1 - overlap))
    windows = []
    for start in range(0, len(data) - window_size + 1, step):
        windows.append(data[start:start + window_size])
    return np.array(windows)
class FoGDatasetFromDF(Dataset):
    def __init__(self, dataframe, window_size, overlap, sensor_columns, label_column, transform=None):
        # Drop rows with missing label values
        self.data = dataframe.dropna(subset=[label_column])
        self.window_size = window_size
        self.overlap = overlap
        self.transform = transform
        self.sensor_columns = sensor_columns
        self.label_column = label_column
        
        sensor_data = self.data[self.sensor_columns].values
        labels = self.data[self.label_column].values
        
        scaler = StandardScaler()
        sensor_data = scaler.fit_transform(sensor_data)
        
        self.windows = sliding_window(sensor_data, window_size, overlap)
        label_windows = sliding_window(labels, window_size, overlap)
        
        window_labels = []
        for lw in label_windows:
            unique, counts = np.unique(lw, return_counts=True)
            majority_label = unique[np.argmax(counts)]
            window_labels.append(majority_label)
        window_labels = np.array(window_labels)
        
        # Filter out any windows where label is NaN
        valid_idx = ~np.isnan(window_labels)
        self.windows = self.windows[valid_idx]
        self.window_labels = window_labels[valid_idx]
        
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, idx):
        x = self.windows[idx]
        y = int(self.window_labels[idx])  # Now guaranteed not to be NaN
        x = torch.tensor(x, dtype=torch.float32).permute(1, 0)
        y = torch.tensor(y, dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y



class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std=0.05):
        self.mean = mean
        self.std = std
        
    def __call__(self, tensor):
        noise = torch.randn(tensor.size()) * self.std + self.mean
        return tensor + noise

# Example transform; you can modify the std value as needed.
transform = AddGaussianNoise(mean=0.0, std=0.05)



# Set Kaggle dataset paths
import glob
import pandas as pd

# Define file patterns for the training data
DEF_TRAIN_PATH = "/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/defog/*.csv"
TDCS_TRAIN_PATH = "/kaggle/input/tlvmc-parkinsons-freezing-gait-prediction/train/tdcsfog/*.csv"

def load_files(pattern, label):
    """
    Loads all CSV files matching the given pattern, assigns a fixed label,
    and concatenates them into one DataFrame.
    
    Parameters:
      pattern (str): File pattern to search for (e.g., using glob)
      label (int): The label to assign to all rows in these files.
      
    Returns:
      DataFrame: Combined data with an added "label" column.
    """
    files = glob.glob(pattern)
    df_list = []
    for file in files:
        df = pd.read_csv(file)
        df["label"] = label  # assign the provided label
        df_list.append(df)
    if df_list:
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame()

# For example, assume that:
# - Files from the "defog" folder represent FoG events (label = 1)
# - Files from the "tdcsfog" folder represent non-FoG events (label = 0)
df_defog = load_files(DEF_TRAIN_PATH, label=1)
df_tdcs  = load_files(TDCS_TRAIN_PATH, label=0)

# Combine the two DataFrames into one training DataFrame
train_df = pd.concat([df_defog, df_tdcs], ignore_index=True)
print("Combined training data shape:", train_df.shape)
print(train_df.head())
MODEL_PATH = '/kaggle/working/defog_detection_model.h5'



# Update sensor column names based on your dataset structure.
sensor_columns = ["AccV","AccML","AccAP",'StartHesitation',
            'Turn', 'Walking']
label_column = 'Valid'

window_size = 100
overlap = 0.5

dataset = FoGDatasetFromDF(train_df, window_size, overlap, sensor_columns, label_column, transform=transform)

print("Dataset sample:")
print(dataset.data.head())

# Split dataset indices (70% train, 15% validation, 15% test)
indices = np.arange(len(dataset))
train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42)
train_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42)

train_dataset = Subset(dataset, train_idx)
val_dataset = Subset(dataset, val_idx)
test_dataset = Subset(dataset, test_idx)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Total windows: {len(dataset)}")
print(f"Training windows: {len(train_dataset)}")
print(f"Validation windows: {len(val_dataset)}")
print(f"Testing windows: {len(test_dataset)}")



# CNN Block for spatial feature extraction
class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=False)  # ensure not in-place
        self.res_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else None
        
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        if self.res_conv:
            residual = self.res_conv(residual)
        out = out + residual  # use out-of-place addition
        out = self.relu(out)
        return out


# Temporal Module using BiLSTM
class TemporalModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(TemporalModule, self).__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size, num_layers=num_layers, 
                              batch_first=True, dropout=dropout, bidirectional=True)
        
    def forward(self, x):
        out, _ = self.bilstm(x)
        return out  # Shape: (batch, seq_len, 2*hidden_size)

# Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, lstm_outputs):
        attn_weights = torch.tanh(self.attn(lstm_outputs))
        attn_weights = self.v(attn_weights).squeeze(-1)
        attn_weights = F.softmax(attn_weights, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_outputs).squeeze(1)
        return context, attn_weights

# Classification Layer
class ClassificationLayer(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ClassificationLayer, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        return self.fc(x)

# Complete HTSAN Model integrating all components
class HTSAN(nn.Module):
    def __init__(self, cnn_in_channels, cnn_out_channels, cnn_kernel_size,
                 lstm_hidden_size, lstm_num_layers, num_classes):
        super(HTSAN, self).__init__()
        self.cnn = CNNBlock(cnn_in_channels, cnn_out_channels, cnn_kernel_size, padding=cnn_kernel_size//2)
        self.temporal = TemporalModule(input_size=cnn_out_channels, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers)
        self.attention = Attention(lstm_hidden_size)
        self.classifier = ClassificationLayer(input_dim=2*lstm_hidden_size, num_classes=num_classes)
        
    def forward(self, x):
        # x shape: (batch, channels, seq_length)
        spatial_features = self.cnn(x)
        # Permute to (batch, seq_length, feature_dim) for LSTM
        temporal_input = spatial_features.permute(0, 2, 1)
        lstm_out = self.temporal(temporal_input)
        context, attn_weights = self.attention(lstm_out)
        logits = self.classifier(context)
        return logits, attn_weights

# Instantiate the model with appropriate parameters.
model = HTSAN(cnn_in_channels=6,       # For example: 3-axis accelerometer + 3-axis gyroscope
              cnn_out_channels=64,
              cnn_kernel_size=3,
              lstm_hidden_size=128,
              lstm_num_layers=2,
              num_classes=2)           # FoG vs. non-FoG

print(model)



torch.autograd.set_detect_anomaly(True)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 50
best_val_loss = float('inf')
patience = 5
trigger_times = 0

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs, _ = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _ = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        trigger_times = 0
        torch.save(model.state_dict(), 'best_model.pth')
    else:
        trigger_times += 1
        if trigger_times >= patience:
            print("Early stopping triggered.")
            break



model.load_state_dict(torch.load('best_model.pth'))
model.eval()

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        outputs, _ = model(inputs)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())


accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, zero_division=0)
recall = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
auc = roc_auc_score(all_labels, all_probs)

print("Evaluation Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC:       {auc:.4f}")

all_probs = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs, _ = model(inputs)
        probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs)

auc = roc_auc_score(all_labels, all_probs)
print(f"Test AUC: {auc:.4f}")


