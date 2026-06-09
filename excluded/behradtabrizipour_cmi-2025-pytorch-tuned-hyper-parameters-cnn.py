import os
import random
from tqdm.auto import tqdm
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, StandardScaler

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader, random_split

from torchmetrics.aggregation import MeanMetric

from prettytable import PrettyTable

import warnings
warnings.simplefilter('ignore')


from cmi_2025_metric_copy_for_import import CompetitionMetric
import kaggle_evaluation.cmi_inference_server


def calc_parameters(model, scale=1e6):
    return sum([p.numel() for p in model.parameters() if p.requires_grad]) / scale


def train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler=None, epoch=None):
    model.train()  # Set the model to training mode
    loss_train = MeanMetric()  # Initialize a metric to track average training loss
    all_preds, all_targets = [], []  # Lists to collect predictions and targets for accuracy

    # Create a progress bar for the training loop
    with tqdm(train_loader, unit='batch') as tepoch:
        for inputs, targets in tepoch:
            if epoch is not None:
                tepoch.set_description(f'Epoch {epoch}')  # Show current epoch in progress bar

            inputs = inputs.to(cfg.device)   # Move inputs to device (CPU/GPU)
            targets = targets.to(cfg.device) # Move targets to device

            logits = model(inputs)  # Forward pass to get raw model outputs

            loss = loss_fn(logits, targets)  # Compute loss between predictions and ground truth

            loss.backward()  # Backpropagation to compute gradients
            
            optimizer.step()  # Update model parameters
            optimizer.zero_grad()  # Clear gradients for the next iteration

            if scheduler is not None:
                scheduler.step()  # Update learning rate if scheduler is provided

            loss_train.update(loss.item(), inputs.shape[0])  # Update running average of loss

            # Display the current average loss in the progress bar
            tepoch.set_postfix(loss=loss_train.compute().item())

            # Save predictions and ground truth labels for accuracy calculation
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Compute hierarchical F1 score using the competition's custom metric
    metric_train = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': label_encoder.classes_[all_targets]}),
        pd.DataFrame({'gesture': label_encoder.classes_[all_preds]})
    )

    return model, loss_train.compute().item(), metric_train  # Return average loss and F1 for the epoch


def evaluate(model, valid_loader):
    model.eval()  # Set the model to evaluation mode (disables dropout, etc.)
    loss_eval = MeanMetric()  # Initialize a metric to track average validation loss
    all_preds, all_targets = [], []  # Lists to collect predictions and targets for accuracy

    with torch.inference_mode():  # Disable gradient computation for faster inference and less memory usage
        for inputs, targets in valid_loader:
            inputs = inputs.to(cfg.device)   # Move inputs to device (CPU/GPU)
            targets = targets.to(cfg.device) # Move targets to device

            logits = model(inputs)  # Forward pass to get model outputs

            loss = F.cross_entropy(logits, targets)  # Compute cross-entropy loss
            loss_eval.update(loss.item(), inputs.shape[0])  # Update running average of loss

            # Save predicted classes and ground truth labels
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Compute hierarchical F1 score using the competition's custom metric
    metric_eval = CompetitionMetric().calculate_hierarchical_f1(
        pd.DataFrame({'gesture': label_encoder.classes_[all_targets]}),
        pd.DataFrame({'gesture': label_encoder.classes_[all_preds]})
    )

    return loss_eval.compute().item(), metric_eval  # Return average validation loss and F1


@dataclass
class Config:
    data_dir: str
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    input_channels: int = 12
    num_classes: int = 18


cfg = Config(
    data_dir = "/kaggle/input/cmi-detect-behavior-with-sensor-data"

)
cfg


df_train = pd.read_csv(f"{cfg.data_dir}/train.csv")
df_train.shape


class SensorImputer(BaseEstimator, TransformerMixin):
    def __init__(self, method='median', thm_threshold=10.0, selected_sensors=None):
        """
        method: 'mean' or 'median'
        thm_threshold: threshold below which thermal values are considered invalid
        selected_sensors: list of sensor types to include, e.g., ['acc', 'rot']
        """
        self.method = method
        self.thm_threshold = thm_threshold
        self.selected_sensors = selected_sensors  # ['acc', 'rot', 'thm', 'tof']
        self.acc_columns = []
        self.rot_columns = []
        self.thm_columns = []
        self.tof_columns = []
        self.column_stats = {}

    def fit(self, X, y=None):
        # Detect columns
        self.acc_columns = [col for col in X.columns if col.startswith("acc_")]
        self.rot_columns = [col for col in X.columns if col.startswith("rot_")]
        self.thm_columns = [col for col in X.columns if col.startswith("thm_")]
        self.tof_columns = [col for col in X.columns if col.startswith("tof_")]

        # Determine which columns to keep based on selected_sensors
        all_groups = {
            'acc': self.acc_columns,
            'rot': self.rot_columns,
            'thm': self.thm_columns,
            'tof': self.tof_columns
        }

        if self.selected_sensors is not None:
            self.sensor_columns = []
            for sensor in self.selected_sensors:
                self.sensor_columns += all_groups.get(sensor, [])
        else:
            self.sensor_columns = sum(all_groups.values(), [])

        # Compute statistics for selected sensors
        for col in self.sensor_columns:
            values = X[col].copy()
            if col in self.thm_columns:
                values[values < self.thm_threshold] = np.nan
            if col in self.tof_columns:
                values = values[values != -1]
            if self.method == 'mean':
                self.column_stats[col] = values.mean()
            elif self.method == 'median':
                self.column_stats[col] = values.median()
            else:
                raise ValueError("method must be either 'mean' or 'median'")

        return self

    def transform(self, X):
        X = X.copy()

        # Preprocessing for thermal
        if 'thm' in self.selected_sensors or self.selected_sensors is None:
            for col in self.thm_columns:
                if col in self.sensor_columns:
                    X.loc[X[col] < self.thm_threshold, col] = np.nan

        for col in self.sensor_columns:
            if col in self.tof_columns:
                X[col] = X[col].replace(-1, np.nan)
            X[col] = X[col].fillna(self.column_stats[col])

        return X[self.sensor_columns]


# Build the full pipeline
pipeline = Pipeline([
    ('imputer', SensorImputer(method='median', thm_threshold=15.0, selected_sensors=['acc', 'rot', 'thm'])),
    ('scaler', StandardScaler()),
])
pipeline

# Apply pipeline to full data
X_processed = pipeline.fit_transform(df_train)
X_processed.shape


# Add back sequence_id so we can group
feature_columns = pipeline['imputer'].sensor_columns
df_processed = pd.DataFrame(X_processed, columns=feature_columns)
df_processed["sequence_id"] = df_train["sequence_id"].values
df_processed['gesture'] = df_train['gesture'].values

df_processed


# Group by sequence
grouped = df_processed.groupby('sequence_id')
total_seqs = grouped.ngroups

# Build tensors
X = []
for seq_id, seq in tqdm(grouped, total=total_seqs, desc="Building sequences"):
    seq = torch.tensor(seq[feature_columns].values, dtype=torch.float32)
    X.append(seq)

print(len(X), X[0].shape)


label_encoder = LabelEncoder()
df_processed['gesture'] = label_encoder.fit_transform(df_processed['gesture'])

# Create sequence-level labels by aggregating gesture per sequence
labels = (
    df_processed.groupby("sequence_id")["gesture"]
    .agg(lambda x: x.mode()[0])  # Use mode in case of repeated labels
)

# Convert the label Series to a PyTorch LongTensor
y = torch.LongTensor(labels.values)

print(y, y.shape)


class CMIDataset(Dataset):
    """IMU dataset with optional preprocessing and onâ€‘theâ€‘fly augmentation.
    """

    def __init__(self, seqs, labels=None, transform=None):
        self.seqs = seqs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, idx):
        x = self.seqs[idx]

        # Transform
        if self.transform is not None:
            x = self.transform(x)
        
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        
        return x

    def __len__(self):
        return len(self.seqs)

    def __repr__(self):
        return "CMIDataset"


def pad_collate(batch):
    """Pad variableâ€‘length sequences to the max length in batch."""
    if isinstance(batch[0], tuple):
        xs, ys = zip(*batch)
    else:
        xs, ys = batch, None

    xs_padded = pad_sequence(xs, batch_first=True).transpose(1, 2)  # (B, C, T_max)
        
    if ys is None:
        return xs_padded
    
    return xs_padded, torch.as_tensor(ys)


class ConvBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_kernel=2):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class TimeSeriesCNN(nn.Module):
    def __init__(self, input_channels=1, num_classes=1, conv_channels=[32, 64, 128], dropout = 0.3):
        super().__init__()
        
        self.feature_extractor = nn.Sequential()
        in_ch = input_channels
        for i, out_ch in enumerate(conv_channels):
            self.feature_extractor.add_module(f'convblock_{i}', ConvBlock1D(in_ch, out_ch))
            in_ch = out_ch  # update for next layer
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(conv_channels[-1], 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, channels, time_steps)
        x = self.feature_extractor(x)
        x = self.global_pool(x)      # shape: (B, C, 1)
        x = x.squeeze(-1)            # shape: (B, C)

        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Split the sequence list and labels together
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=labels
)

train_set = CMIDataset(X_train, y_train, transform=None)

valid_set = CMIDataset(X_valid, y_valid, transform=None)

train_loader = DataLoader(train_set, batch_size=8, shuffle=True, collate_fn=pad_collate)

valid_loader = DataLoader(valid_set, batch_size=16, shuffle=False, collate_fn=pad_collate)


model = TimeSeriesCNN(cfg.input_channels, cfg.num_classes, conv_channels=[32, 64, 128]).to(cfg.device)
calc_parameters(model, scale=1e3)


# loss check

x_batch, y_batch = next(iter(train_loader))
print(x_batch.shape, y_batch.shape, end="\n\n")

with torch.no_grad():
    logits = model(x_batch.to(cfg.device))

loss0 = F.cross_entropy(logits, y_batch.to(cfg.device))
print(f"Initial Loss: {loss0} | Expected Loss: {-np.log(1/18)}") 


counts = Counter(y_train.tolist())
print(counts)
class_counts = torch.tensor([counts.get(i, 0) for i in range(cfg.num_classes)], dtype=torch.float32)
print(class_counts)
class_weights = 1.0 / (class_counts + 1e-8)  # Inverse frequency
class_weights = class_weights / class_weights.sum()  # Normalize to keep loss scale stable (optional but recommended)

loss_fn = nn.CrossEntropyLoss(weight=class_weights).to(cfg.device)


learning_rate   = 3e-4
weight_decay    = 0.01
use_fused       = True

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay,
    fused=use_fused
)


num_epochs = 100
steps_per_epoch = len(train_loader)

# OneCycleLR scheduler: warm-up + cosine annealing
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=learning_rate,              # Peak learning rate after warm-up
    epochs=num_epochs,                 # Total number of epochs (no +1)
    steps_per_epoch=steps_per_epoch,   # Number of optimizer steps per epoch
    pct_start=0.05,                    # % of total steps used for warm-up
    anneal_strategy='cos',             # Use cosine annealing after warm-up
    div_factor=25,                     # initial_lr = max_lr / div_factor
    final_div_factor=1e9               # final_lr  = max_lr / final_div_factor
)

# model = TimeSeriesCNN(cfg.input_channels, cfg.num_classes, conv_channels=[32, 64, 128]).to(cfg.device)


loss_train_hist = [loss0.item()]
loss_valid_hist = [loss0.item()]

metric_train_hist = [0]
metric_valid_hist = [0]

best_loss_valid = torch.inf
epoch_counter = 0

checkpoint_path = "best-model.pt"


torch.manual_seed(1337)

for epoch in range(1, num_epochs+1):
    # Train
    model, loss_train, metric_train = train_one_epoch(model, train_loader, loss_fn, optimizer, scheduler, epoch)
    print(f'ðŸ”¸ Train: Loss={loss_train:.4} | Metric={metric_train:.4}')

    # Validation
    loss_valid, metric_valid = evaluate(model, valid_loader)
    print(f'ðŸ”¹ Valid: Loss={loss_valid:.4} | Metric={metric_valid:.4}')

    loss_train_hist.append(loss_train)
    loss_valid_hist.append(loss_valid)

    metric_train_hist.append(metric_train)
    metric_valid_hist.append(metric_valid)

    #  Save best model
    if loss_valid < best_loss_valid:
        best_loss_valid = loss_valid
        torch.save(model.state_dict(), checkpoint_path)
        print(f"âœ… Saved new best model (epoch {epoch})")

    print()
    
    epoch_counter += 1


n = len(loss_train_hist)

plt.figure(figsize=(8, 6))

plt.plot(range(n), loss_train_hist, 'r-', label='Train')
plt.plot(range(n), loss_valid_hist, 'b-', label='Validation')

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend();


plt.figure(figsize=(8, 6))

plt.plot(range(n), metric_train_hist, 'r-', label='Train')
plt.plot(range(n), metric_valid_hist, 'b-', label='Validation')

plt.xlabel('Epoch')
plt.ylabel('Metric')
plt.grid(True)
plt.legend();


model = TimeSeriesCNN(cfg.input_channels, cfg.num_classes, conv_channels=[32, 64, 128]).to(cfg.device)
state_dict = torch.load("best-model.pt", map_location=cfg.device)
model.load_state_dict(state_dict)
model.eval()


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    df_seq = sequence.to_pandas()
    # Preprocess
    data = pipeline.transform(df_seq)
    data = torch.tensor(data.T, dtype=torch.float32).unsqueeze(0)
    with torch.inference_mode():
        logits = model(data.to(cfg.device))
    cls = torch.argmax(logits, keepdims=True).cpu().numpy()
    cls = label_encoder.inverse_transform(cls[0])
    return cls[0]


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







