import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, accuracy_score, recall_score, f1_score, precision_score,
    roc_auc_score, cohen_kappa_score, log_loss, classification_report,
    balanced_accuracy_score, matthews_corrcoef, roc_curve, auc,
    precision_recall_curve, average_precision_score, brier_score_loss
)
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import time
import cv2
from datetime import datetime
from collections import defaultdict
import psutil
import gc
from torch.profiler import profile, record_function, ProfilerActivity
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
warnings.filterwarnings('ignore')
import torch.nn.functional as F
import numpy as np
from scipy.spatial.distance import jensenshannon

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

start_time = time.time()
start_date = datetime.now()
print(f"Training started at: {start_date}")

BASE_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"

brain_activities = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']
activity_mapping = {activity: idx for idx, activity in enumerate(brain_activities)}

# # Load and split data
# df = pd.read_csv(f"{BASE_DIR}train.csv")
# # df = df.sample(frac=0.01, random_state=42)  # Uncomment for quick testing

# # Split 80% Train, 20% Temp (Validation + Test)
# train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
# # train_df = train_df.head(60000)

# # Split 10% Validation, 10% Test from Temp
# val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_df = pd.read_csv("/kaggle/input/datasets/ashoksingh5972/mixed-subject-split/train.csv")
train_df = train_df.sample(frac=0.2, random_state=42)

val_df = pd.read_csv("/kaggle/input/datasets/ashoksingh5972/mixed-subject-split/val.csv")
test_df = pd.read_csv("/kaggle/input/datasets/ashoksingh5972/mixed-subject-split/test.csv")

print("Splitting done with balanced training data!")
print("Train:", len(train_df), "Val:", len(val_df), "Test:", len(test_df))




# class ChunkedBrainActivityDataset(Dataset):
#     def __init__(self, csv_file, base_dir, activity_mapping, md):
#         self.df = csv_file
#         self.base_dir = base_dir
#         self.activity_mapping = activity_mapping
#         self.resize_transform = transforms.Resize((224, 224))
#         self.md = md

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         spect_id, label, offset = self.df.iloc[idx][["spectrogram_id", "expert_consensus", "spectrogram_label_offset_seconds"]]

#         temp_df = pd.read_parquet(f'{self.base_dir}/train_spectrograms/{spect_id}.parquet')
#         temp_df.drop(['time'], axis=1, inplace=True)

#         start = int(offset) // 2
#         temp_df = temp_df[start:start+300]
#         temp_df = np.log1p(temp_df)
#         temp_df /= temp_df.max()
#         temp_arr = np.nan_to_num(temp_df.to_numpy(), nan=1e-4)

#         # Use OpenCV to apply a colormap and convert to RGB
#         temp_arr_uint8 = np.uint8(255 * temp_arr)
#         rgb_image = cv2.applyColorMap(temp_arr_uint8, cv2.COLORMAP_JET)

#         # Normalize to [0, 1] and convert to tensor
#         rgb_image = rgb_image.astype(np.float32) / 255.0
#         rgb_image_tensor = torch.tensor(rgb_image).permute(2, 0, 1)  # (C, H, W)
#         rgb_image_tensor = self.resize_transform(rgb_image_tensor)
            
#         y = self.activity_mapping[label]
#         y_tensor = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=6).float()
        
#         return rgb_image_tensor, y_tensor


# # Create datasets and data loaders
# train_dataset = ChunkedBrainActivityDataset(csv_file=train_df, base_dir=BASE_DIR, activity_mapping=activity_mapping, md="lr")
# val_dataset = ChunkedBrainActivityDataset(csv_file=val_df, base_dir=BASE_DIR, activity_mapping=activity_mapping, md="lr")
# test_dataset = ChunkedBrainActivityDataset(csv_file=test_df, base_dir=BASE_DIR, activity_mapping=activity_mapping, md="lr")

# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2)
# val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)




import numpy as np

def extract_middle_10sec(eeg, fs):
    total_samples = eeg.shape[0]
    total_duration_sec = total_samples / fs
    
    start_time = (total_duration_sec / 2) - 5
    end_time = (total_duration_sec / 2) + 5
    start_idx = int(start_time * fs)
    end_idx = int(end_time * fs)
    
    return eeg[start_idx:end_idx]


import numpy as np
from scipy.signal import stft

def compute_stft_spectrogram(eeg_10sec, fs):

    # If multi-channel, process each channel separately and stack results
    if eeg_10sec.ndim == 2:
        # Example: Use mean across channels, or adapt as needed
        eeg_input = eeg_10sec.mean(axis=1)
    else:
        eeg_input = eeg_10sec

    # Compute STFT: nperseg=256 window size, 50% overlap
    f, t, Zxx = stft(eeg_input, fs=fs, nperseg=256, noverlap=128)

    # Power spectral density
    Sxx = np.abs(Zxx) ** 2

    # Log scaling and normalization
    Sxx = np.log1p(Sxx)
    Sxx /= (Sxx.max() + 1e-8)
    return Sxx  # shape: (freq bins, time bins)


import pandas as pd
import numpy as np
import os

EEG_CHANNELS = ['Fp1', 'F3', 'C3', 'P3', 'F7', 'T3', 'T5', 'O1',
                'Fz', 'Cz', 'Pz', 'Fp2', 'F4', 'C4', 'P4', 'F8',
                'T4', 'T6', 'O2']

def load_eeg_signal(row):
    BASE_DIR = "/kaggle/input/hms-harmful-brain-activity-classification/"
    spect_id = row['eeg_id']
    eeg_path = os.path.join(BASE_DIR, "train_eegs", f"{spect_id}.parquet")
    df = pd.read_parquet(eeg_path)
    # Load as a [n_samples, n_channels] array
    eeg = df[EEG_CHANNELS].to_numpy()
    return eeg  # shape: (samples, 19)

import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms

class Middle10SecSpectrogramDataset(Dataset):
    def __init__(self, df, eeg_loader_func, activity_mapping, fs):
        self.df = df  # DataFrame with EEG info and labels
        self.eeg_loader_func = eeg_loader_func  # function to load EEG array by row info
        self.activity_mapping = activity_mapping
        self.fs = fs
        self.resize_transform = transforms.Resize((224, 224))
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        eeg = self.eeg_loader_func(row)  # Returns (samples,) array

        # 1. Extract middle 10sec
        eeg_10sec = extract_middle_10sec(eeg, self.fs)

        # 2. Spectrogram
        Sxx = compute_stft_spectrogram(eeg_10sec, self.fs)

        # 3. Make into RGB image
        Sxx_uint8 = np.uint8(255 * Sxx)
        rgb_img = cv2.applyColorMap(Sxx_uint8, cv2.COLORMAP_JET)  # H,W,3
        rgb_img = rgb_img.astype(np.float32) / 255.0

        # 4. Tensor and resize
        img_tensor = torch.tensor(rgb_img).permute(2,0,1)  # C,H,W
        img_tensor = self.resize_transform(img_tensor)

        # 5. Label
        y = self.activity_mapping[row["expert_consensus"]]
        y_tensor = torch.nn.functional.one_hot(torch.tensor(y), num_classes=len(self.activity_mapping)).float()

        return img_tensor, y_tensor

fs = 200

# Create datasets with the new class and required arguments
train_dataset = Middle10SecSpectrogramDataset(df=train_df, eeg_loader_func=load_eeg_signal, activity_mapping=activity_mapping, fs=fs)
val_dataset = Middle10SecSpectrogramDataset(df=val_df, eeg_loader_func=load_eeg_signal, activity_mapping=activity_mapping, fs=fs)
test_dataset = Middle10SecSpectrogramDataset(df=test_df, eeg_loader_func=load_eeg_signal, activity_mapping=activity_mapping, fs=fs)

# Create DataLoaders similarly
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True, prefetch_factor=2)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=2)


def calculate_kl_divergence(y_true, y_pred_probs, epsilon=1e-8):
    """
    Calculate KL divergence between true and predicted probability distributions.
    
    Args:
        y_true: True labels (numpy array or tensor)
        y_pred_probs: Predicted probabilities (numpy array or tensor)
        epsilon: Small value to avoid log(0)
    
    Returns:
        dict: Contains various KL divergence metrics
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred_probs):
        y_pred_probs = y_pred_probs.cpu().numpy()
    
    num_classes = y_pred_probs.shape[1]
    
    # Convert labels to one-hot encoding
    y_true_onehot = np.eye(num_classes)[y_true]
    
    # Add epsilon to avoid log(0)
    y_pred_probs_safe = np.clip(y_pred_probs, epsilon, 1.0)
    y_true_onehot_safe = np.clip(y_true_onehot, epsilon, 1.0)
    
    # Calculate KL divergence for each sample
    # KL(P||Q) = sum(P * log(P/Q))
    kl_divs = []
    reverse_kl_divs = []
    js_divs = []
    
    for i in range(len(y_true)):
        # Forward KL: KL(true || pred)
        kl_forward = np.sum(y_true_onehot_safe[i] * np.log(y_true_onehot_safe[i] / y_pred_probs_safe[i]))
        kl_divs.append(kl_forward)
        
        # Reverse KL: KL(pred || true)
        kl_reverse = np.sum(y_pred_probs_safe[i] * np.log(y_pred_probs_safe[i] / y_true_onehot_safe[i]))
        reverse_kl_divs.append(kl_reverse)
        
        # Jensen-Shannon divergence (symmetric)
        js_div = jensenshannon(y_true_onehot_safe[i], y_pred_probs_safe[i]) ** 2
        js_divs.append(js_div)
    
    # Calculate class-wise KL divergence
    class_kl_divs = []
    for class_idx in range(num_classes):
        class_mask = (y_true == class_idx)
        if np.sum(class_mask) > 0:
            class_true = y_true_onehot_safe[class_mask]
            class_pred = y_pred_probs_safe[class_mask]
            
            # Average KL divergence for this class
            class_kl = np.mean([
                np.sum(class_true[j] * np.log(class_true[j] / class_pred[j]))
                for j in range(len(class_true))
            ])
            class_kl_divs.append(class_kl)
        else:
            class_kl_divs.append(np.nan)
    
    # Calculate distribution-level KL divergence
    # Compare overall class distributions
    true_dist = np.bincount(y_true, minlength=num_classes) / len(y_true)
    pred_dist = np.mean(y_pred_probs, axis=0)
    
    # Add epsilon to distributions
    true_dist_safe = np.clip(true_dist, epsilon, 1.0)
    pred_dist_safe = np.clip(pred_dist, epsilon, 1.0)
    
    # Normalize to ensure they sum to 1
    true_dist_safe = true_dist_safe / np.sum(true_dist_safe)
    pred_dist_safe = pred_dist_safe / np.sum(pred_dist_safe)
    
    dist_kl_forward = np.sum(true_dist_safe * np.log(true_dist_safe / pred_dist_safe))
    dist_kl_reverse = np.sum(pred_dist_safe * np.log(pred_dist_safe / true_dist_safe))
    dist_js = jensenshannon(true_dist_safe, pred_dist_safe) ** 2
    
    return {
        'mean_kl_divergence': np.mean(kl_divs),
        'std_kl_divergence': np.std(kl_divs),
        'median_kl_divergence': np.median(kl_divs),
        'mean_reverse_kl_divergence': np.mean(reverse_kl_divs),
        'mean_js_divergence': np.mean(js_divs),
        'class_kl_divergences': class_kl_divs,
        'distribution_kl_forward': dist_kl_forward,
        'distribution_kl_reverse': dist_kl_reverse,
        'distribution_js_divergence': dist_js,
        'sample_kl_divergences': kl_divs,
        'sample_reverse_kl_divergences': reverse_kl_divs,
        'sample_js_divergences': js_divs
    }

def calculate_kl_divergence_pytorch(y_true, y_pred_logits, temperature=1.0):
    """
    Calculate KL divergence using PyTorch (useful for differentiable operations).
    
    Args:
        y_true: True labels (tensor)
        y_pred_logits: Predicted logits (tensor)
        temperature: Temperature scaling parameter
    
    Returns:
        dict: KL divergence metrics as tensors
    """
    # Apply temperature scaling
    y_pred_probs = F.softmax(y_pred_logits / temperature, dim=1)
    
    # Convert labels to one-hot
    num_classes = y_pred_logits.shape[1]
    y_true_onehot = F.one_hot(y_true, num_classes=num_classes).float()
    
    # Calculate KL divergence
    log_pred = F.log_softmax(y_pred_logits / temperature, dim=1)
    
    # KL(true || pred) = sum(true * log(true / pred))
    # Since true is one-hot, this simplifies to -log(pred[true_class])
    kl_div = F.kl_div(log_pred, y_true_onehot, reduction='none').sum(dim=1)
    
    # Reverse KL: KL(pred || true)
    log_true = torch.log(y_true_onehot + 1e-8)
    reverse_kl_div = F.kl_div(log_true, y_pred_probs, reduction='none').sum(dim=1)
    
    return {
        'mean_kl_divergence': kl_div.mean(),
        'std_kl_divergence': kl_div.std(),
        'mean_reverse_kl_divergence': reverse_kl_div.mean(),
        'sample_kl_divergences': kl_div,
        'sample_reverse_kl_divergences': reverse_kl_div
    }

def plot_kl_divergence_analysis(kl_results, class_names):
    """
    Plot KL divergence analysis results.
    
    Args:
        kl_results: Results from calculate_kl_divergence function
        class_names: List of class names
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sample-wise KL divergence distribution
    axes[0, 0].hist(kl_results['sample_kl_divergences'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(kl_results['mean_kl_divergence'], color='red', linestyle='--', 
                       label=f'Mean: {kl_results["mean_kl_divergence"]:.4f}')
    axes[0, 0].axvline(kl_results['median_kl_divergence'], color='green', linestyle='--',
                       label=f'Median: {kl_results["median_kl_divergence"]:.4f}')
    axes[0, 0].set_title('Distribution of Sample-wise KL Divergences')
    axes[0, 0].set_xlabel('KL Divergence')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Class-wise KL divergence
    valid_class_kl = [kl for kl in kl_results['class_kl_divergences'] if not np.isnan(kl)]
    valid_class_names = [name for i, name in enumerate(class_names) 
                        if not np.isnan(kl_results['class_kl_divergences'][i])]
    
    axes[0, 1].bar(valid_class_names, valid_class_kl, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Class-wise KL Divergences')
    axes[0, 1].set_xlabel('Classes')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Comparison of different divergence measures
    divergence_types = ['KL (Forward)', 'KL (Reverse)', 'Jensen-Shannon']
    divergence_values = [
        kl_results['mean_kl_divergence'],
        kl_results['mean_reverse_kl_divergence'],
        kl_results['mean_js_divergence']
    ]
    
    axes[1, 0].bar(divergence_types, divergence_values, 
                   color=['blue', 'red', 'green'], alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Comparison of Divergence Measures')
    axes[1, 0].set_ylabel('Divergence Value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # KL divergence vs prediction confidence
    max_probs = np.max(np.array(kl_results['sample_kl_divergences']).reshape(-1, 1), axis=1) \
                if len(np.array(kl_results['sample_kl_divergences']).shape) == 1 else \
                np.max(y_pred_probs, axis=1)  # This would need y_pred_probs from calling context
    
    # For now, create a scatter plot of KL vs sample index
    sample_indices = range(len(kl_results['sample_kl_divergences']))
    axes[1, 1].scatter(sample_indices, kl_results['sample_kl_divergences'], 
                       alpha=0.6, s=10, color='purple')
    axes[1, 1].set_title('Sample-wise KL Divergence Pattern')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('KL Divergence')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()




class ResNet101EncoderLogisticRegression(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet101EncoderLogisticRegression, self).__init__()
        # Load pretrained ResNet101
        self.encoder = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        
        # Get the feature size from the last layer (fc layer)
        n_features = self.encoder.fc.in_features
        
        # Remove the original classifier head
        self.encoder.fc = nn.Identity()
        
        # Add a logistic regression layer for classification
        self.logistic_regression = nn.Linear(n_features, num_classes)

    def forward(self, x):
        features = self.encoder(x)  # Extract features
        logits = self.logistic_regression(features)  # Apply classifier
        return logits

    def get_features(self, x):
        """Extract features for t-SNE visualization"""
        with torch.no_grad():
            features = self.encoder(x)
        return features


# Early Stopping Class
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Instantiate the model and move it to the appropriate device
num_classes = 6
model = ResNet101EncoderLogisticRegression(num_classes=num_classes).to(device)

# Calculate model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

param_count = count_parameters(model)
print(f"Model parameters: {param_count / 1e6:.2f}M")

# Calculate model size
def get_model_size_mb(model):
    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb

model_size_mb = get_model_size_mb(model)
print(f"Model size: {model_size_mb:.2f} MB")

print("ResNet101 model with logistic regression classifier loaded successfully!")

# Define the loss function and optimizer
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# Advanced loss functions
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# Initialize early stopping
early_stopping = EarlyStopping(patience=5, min_delta=0.001)

# Training history
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': [],
    'learning_rate': []
}

# Enhanced training loop with validation and early stopping
num_epochs = 100
best_val_acc = 0.0

print("Starting training...")

for epoch in range(num_epochs):
    epoch_start_time = time.time()
    
    # Training phase
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
        images = images.to(device)
        targets = targets.to(device)
        labels = torch.argmax(targets, dim=1)
        
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(logits, dim=1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
    
    train_loss = running_loss / total
    train_acc = 100 * correct / total
    
    # Validation phase
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
            images = images.to(device)
            targets = targets.to(device)
            labels = torch.argmax(targets, dim=1)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            val_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
            val_total += labels.size(0)
            val_correct += (preds == labels).sum().item()
    
    val_loss = val_loss / val_total
    val_acc = 100 * val_correct / val_total
    
    # Step the scheduler
    scheduler.step()
    current_lr = optimizer.param_groups[0]['lr']
    
    # Save training history
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    history['learning_rate'].append(current_lr)
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_resnet101_model.pth')
    
    epoch_time = time.time() - epoch_start_time
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Time: {epoch_time:.2f}s")
    print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"Val - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
    print(f"Learning Rate: {current_lr:.6f}")
    print("-" * 50)
    
    # Early stopping check
    if early_stopping(val_loss, model):
        print(f"Early stopping triggered after epoch {epoch+1}")
        break

# Load best model for evaluation
model.load_state_dict(torch.load('best_resnet101_model.pth'))

# Plot training curves
def plot_training_curves(history):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss curves
    axes[0, 0].plot(history['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].plot(history['val_loss'], label='Validation Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy curves
    axes[0, 1].plot(history['train_acc'], label='Train Accuracy', color='blue')
    axes[0, 1].plot(history['val_acc'], label='Validation Accuracy', color='red')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Learning rate
    axes[1, 0].plot(history['learning_rate'], label='Learning Rate', color='green')
    axes[1, 0].set_title('Learning Rate Schedule')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    axes[1, 0].set_yscale('log')
    
    # Combined loss and accuracy
    ax1 = axes[1, 1]
    ax2 = ax1.twinx()
    
    ax1.plot(history['train_loss'], 'b-', label='Train Loss')
    ax1.plot(history['val_loss'], 'r-', label='Val Loss')
    ax2.plot(history['train_acc'], 'b--', label='Train Acc')
    ax2.plot(history['val_acc'], 'r--', label='Val Acc')
    
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss', color='black')
    ax2.set_ylabel('Accuracy (%)', color='black')
    ax1.set_title('Combined Loss and Accuracy')
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax1.grid(True)
    
    plt.tight_layout()
    plt.show()

plot_training_curves(history)


def calculate_kl_divergence(y_true, y_pred_probs, epsilon=1e-8):
    """
    Calculate KL divergence between true and predicted probability distributions.
    
    Args:
        y_true: True labels (numpy array or tensor)
        y_pred_probs: Predicted probabilities (numpy array or tensor)
        epsilon: Small value to avoid log(0)
    
    Returns:
        dict: Contains various KL divergence metrics
    """
    # Convert to numpy if needed
    if torch.is_tensor(y_true):
        y_true = y_true.cpu().numpy()
    if torch.is_tensor(y_pred_probs):
        y_pred_probs = y_pred_probs.cpu().numpy()
    
    num_classes = y_pred_probs.shape[1]
    
    # Convert labels to one-hot encoding
    y_true_onehot = np.eye(num_classes)[y_true]
    
    # Add epsilon to avoid log(0)
    y_pred_probs_safe = np.clip(y_pred_probs, epsilon, 1.0)
    y_true_onehot_safe = np.clip(y_true_onehot, epsilon, 1.0)
    
    # Calculate KL divergence for each sample
    # KL(P||Q) = sum(P * log(P/Q))
    kl_divs = []
    reverse_kl_divs = []
    js_divs = []
    
    for i in range(len(y_true)):
        # Forward KL: KL(true || pred)
        kl_forward = np.sum(y_true_onehot_safe[i] * np.log(y_true_onehot_safe[i] / y_pred_probs_safe[i]))
        kl_divs.append(kl_forward)
        
        # Reverse KL: KL(pred || true)
        kl_reverse = np.sum(y_pred_probs_safe[i] * np.log(y_pred_probs_safe[i] / y_true_onehot_safe[i]))
        reverse_kl_divs.append(kl_reverse)
        
        # Jensen-Shannon divergence (symmetric)
        js_div = jensenshannon(y_true_onehot_safe[i], y_pred_probs_safe[i]) ** 2
        js_divs.append(js_div)
    
    # Calculate class-wise KL divergence
    class_kl_divs = []
    for class_idx in range(num_classes):
        class_mask = (y_true == class_idx)
        if np.sum(class_mask) > 0:
            class_true = y_true_onehot_safe[class_mask]
            class_pred = y_pred_probs_safe[class_mask]
            
            # Average KL divergence for this class
            class_kl = np.mean([
                np.sum(class_true[j] * np.log(class_true[j] / class_pred[j]))
                for j in range(len(class_true))
            ])
            class_kl_divs.append(class_kl)
        else:
            class_kl_divs.append(np.nan)
    
    # Calculate distribution-level KL divergence
    # Compare overall class distributions
    true_dist = np.bincount(y_true, minlength=num_classes) / len(y_true)
    pred_dist = np.mean(y_pred_probs, axis=0)
    
    # Add epsilon to distributions
    true_dist_safe = np.clip(true_dist, epsilon, 1.0)
    pred_dist_safe = np.clip(pred_dist, epsilon, 1.0)
    
    # Normalize to ensure they sum to 1
    true_dist_safe = true_dist_safe / np.sum(true_dist_safe)
    pred_dist_safe = pred_dist_safe / np.sum(pred_dist_safe)
    
    dist_kl_forward = np.sum(true_dist_safe * np.log(true_dist_safe / pred_dist_safe))
    dist_kl_reverse = np.sum(pred_dist_safe * np.log(pred_dist_safe / true_dist_safe))
    dist_js = jensenshannon(true_dist_safe, pred_dist_safe) ** 2
    
    return {
        'mean_kl_divergence': np.mean(kl_divs),
        'std_kl_divergence': np.std(kl_divs),
        'median_kl_divergence': np.median(kl_divs),
        'mean_reverse_kl_divergence': np.mean(reverse_kl_divs),
        'mean_js_divergence': np.mean(js_divs),
        'class_kl_divergences': class_kl_divs,
        'distribution_kl_forward': dist_kl_forward,
        'distribution_kl_reverse': dist_kl_reverse,
        'distribution_js_divergence': dist_js,
        'sample_kl_divergences': kl_divs,
        'sample_reverse_kl_divergences': reverse_kl_divs,
        'sample_js_divergences': js_divs
    }

def calculate_kl_divergence_pytorch(y_true, y_pred_logits, temperature=1.0):
    """
    Calculate KL divergence using PyTorch (useful for differentiable operations).
    
    Args:
        y_true: True labels (tensor)
        y_pred_logits: Predicted logits (tensor)
        temperature: Temperature scaling parameter
    
    Returns:
        dict: KL divergence metrics as tensors
    """
    # Apply temperature scaling
    y_pred_probs = F.softmax(y_pred_logits / temperature, dim=1)
    
    # Convert labels to one-hot
    num_classes = y_pred_logits.shape[1]
    y_true_onehot = F.one_hot(y_true, num_classes=num_classes).float()
    
    # Calculate KL divergence
    log_pred = F.log_softmax(y_pred_logits / temperature, dim=1)
    
    # KL(true || pred) = sum(true * log(true / pred))
    # Since true is one-hot, this simplifies to -log(pred[true_class])
    kl_div = F.kl_div(log_pred, y_true_onehot, reduction='none').sum(dim=1)
    
    # Reverse KL: KL(pred || true)
    log_true = torch.log(y_true_onehot + 1e-8)
    reverse_kl_div = F.kl_div(log_true, y_pred_probs, reduction='none').sum(dim=1)
    
    return {
        'mean_kl_divergence': kl_div.mean(),
        'std_kl_divergence': kl_div.std(),
        'mean_reverse_kl_divergence': reverse_kl_div.mean(),
        'sample_kl_divergences': kl_div,
        'sample_reverse_kl_divergences': reverse_kl_div
    }

def plot_kl_divergence_analysis(kl_results, class_names):
    """
    Plot KL divergence analysis results.
    
    Args:
        kl_results: Results from calculate_kl_divergence function
        class_names: List of class names
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sample-wise KL divergence distribution
    axes[0, 0].hist(kl_results['sample_kl_divergences'], bins=50, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvline(kl_results['mean_kl_divergence'], color='red', linestyle='--', 
                       label=f'Mean: {kl_results["mean_kl_divergence"]:.4f}')
    axes[0, 0].axvline(kl_results['median_kl_divergence'], color='green', linestyle='--',
                       label=f'Median: {kl_results["median_kl_divergence"]:.4f}')
    axes[0, 0].set_title('Distribution of Sample-wise KL Divergences')
    axes[0, 0].set_xlabel('KL Divergence')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Class-wise KL divergence
    valid_class_kl = [kl for kl in kl_results['class_kl_divergences'] if not np.isnan(kl)]
    valid_class_names = [name for i, name in enumerate(class_names) 
                        if not np.isnan(kl_results['class_kl_divergences'][i])]
    
    axes[0, 1].bar(valid_class_names, valid_class_kl, color='skyblue', edgecolor='black')
    axes[0, 1].set_title('Class-wise KL Divergences')
    axes[0, 1].set_xlabel('Classes')
    axes[0, 1].set_ylabel('KL Divergence')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Comparison of different divergence measures
    divergence_types = ['KL (Forward)', 'KL (Reverse)', 'Jensen-Shannon']
    divergence_values = [
        kl_results['mean_kl_divergence'],
        kl_results['mean_reverse_kl_divergence'],
        kl_results['mean_js_divergence']
    ]
    
    axes[1, 0].bar(divergence_types, divergence_values, 
                   color=['blue', 'red', 'green'], alpha=0.7, edgecolor='black')
    axes[1, 0].set_title('Comparison of Divergence Measures')
    axes[1, 0].set_ylabel('Divergence Value')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # KL divergence vs prediction confidence
    max_probs = np.max(np.array(kl_results['sample_kl_divergences']).reshape(-1, 1), axis=1) \
                if len(np.array(kl_results['sample_kl_divergences']).shape) == 1 else \
                np.max(y_pred_probs, axis=1)  # This would need y_pred_probs from calling context
    
    # For now, create a scatter plot of KL vs sample index
    sample_indices = range(len(kl_results['sample_kl_divergences']))
    axes[1, 1].scatter(sample_indices, kl_results['sample_kl_divergences'], 
                       alpha=0.6, s=10, color='purple')
    axes[1, 1].set_title('Sample-wise KL Divergence Pattern')
    axes[1, 1].set_xlabel('Sample Index')
    axes[1, 1].set_ylabel('KL Divergence')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


# Comprehensive evaluation functions
def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculate Expected Calibration Error"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return ece


def reliability_diagram(y_true, y_prob, n_bins=10):
    """Plot reliability diagram"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
    
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    plt.plot(confidences, accuracies, 'o-', label='Model')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Reliability Diagram')
    plt.legend()
    plt.grid(True)
    plt.show()


def geometric_mean_score(y_true, y_pred):
    """Calculate geometric mean of class-wise recalls"""
    cm = confusion_matrix(y_true, y_pred)
    recalls = []
    for i in range(cm.shape[0]):
        if cm[i, :].sum() > 0:
            recall = cm[i, i] / cm[i, :].sum()
            recalls.append(recall)
    return np.prod(recalls) ** (1.0 / len(recalls))


def calculate_specificity(y_true, y_pred, num_classes):
    """Calculate class-wise specificity"""
    cm = confusion_matrix(y_true, y_pred)
    specificities = []
    for i in range(num_classes):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificities.append(specificity)
    return specificities


def add_noise_and_evaluate(model, test_loader, device, noise_levels=[0.1, 0.2, 0.3]):
    """Evaluate model robustness to noise"""
    results = {}
    model.eval()
    
    for noise_level in noise_levels:
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, targets in test_loader:
                images = images.to(device)
                labels = torch.argmax(targets, dim=1).to(device)
                
                # Add Gaussian noise
                noise = torch.randn_like(images) * noise_level
                noisy_images = images + noise
                noisy_images = torch.clamp(noisy_images, 0, 1)
                
                outputs = model(noisy_images)
                _, preds = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (preds == labels).sum().item()
        
        accuracy = 100 * correct / total
        results[noise_level] = accuracy
        print(f"Noise level {noise_level}: {accuracy:.2f}%")
    
    return results


def measure_inference_time(model, test_loader, device, num_samples=100):
    """Measure average inference time"""
    model.eval()
    times = []
    
    with torch.no_grad():
        for i, (images, _) in enumerate(test_loader):
            if i >= num_samples // images.size(0):
                break
                
            images = images.to(device)
            
            start_time = time.time()
            _ = model(images)
            torch.cuda.synchronize() if device.type == 'cuda' else None
            end_time = time.time()
            
            batch_time = (end_time - start_time) * 1000  # Convert to ms
            per_sample_time = batch_time / images.size(0)
            times.extend([per_sample_time] * images.size(0))
    
    return np.mean(times), np.std(times)


def comprehensive_evaluation(model, data_loader, device, class_names):
    """Calculate all evaluation metrics"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    all_features = []
    
    # Memory usage before inference
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        memory_before = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    else:
        memory_before = psutil.virtual_memory().used / 1024 / 1024
    
    with torch.no_grad():
        for images, targets in tqdm(data_loader, desc="Evaluating"):
            images = images.to(device)
            targets = targets.to(device)
            labels = torch.argmax(targets, dim=1)
            
            # Forward pass
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            # Get features for t-SNE
            features = model.get_features(images)
            
            _, preds = torch.max(outputs, dim=1)
            
            # Collect predictions, labels, probabilities, and features
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_features.extend(features.cpu().numpy())
    
    # Memory usage after inference
    if device.type == 'cuda':
        memory_after = torch.cuda.memory_allocated() / 1024 / 1024  # MB
    else:
        memory_after = psutil.virtual_memory().used / 1024 / 1024
    
    inference_memory = memory_after - memory_before
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    all_features = np.array(all_features)
    
    results = {}
    
    # Basic metrics
    results['accuracy'] = accuracy_score(all_labels, all_preds)
    results['precision_macro'] = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    results['recall_macro'] = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    results['f1_macro'] = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    results['balanced_accuracy'] = balanced_accuracy_score(all_labels, all_preds)
    results['cohen_kappa'] = cohen_kappa_score(all_labels, all_preds)
    results['matthews_corrcoef'] = matthews_corrcoef(all_labels, all_preds)
    results['geometric_mean'] = geometric_mean_score(all_labels, all_preds)
    
    # Multiclass metrics
    try:
        results['auc_roc_macro'] = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        results['log_loss'] = log_loss(all_labels, all_probs)
    except ValueError as e:
        print(f"Warning: Could not calculate AUC-ROC or log loss: {e}")
        results['auc_roc_macro'] = np.nan
        results['log_loss'] = np.nan
    
    # Brier score (for multiclass, we'll use the average)
    brier_scores = []
    for i in range(len(class_names)):
        y_true_binary = (all_labels == i).astype(int)
        y_prob_binary = all_probs[:, i]
        brier_scores.append(brier_score_loss(y_true_binary, y_prob_binary))
    results['brier_score'] = np.mean(brier_scores)
    
    # Expected Calibration Error
    max_probs = np.max(all_probs, axis=1)
    predicted_correctly = (all_preds == all_labels).astype(int)
    results['ece'] = calculate_ece(predicted_correctly, max_probs)
    
    # Class-wise metrics
    cm = confusion_matrix(all_labels, all_preds)
    class_precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    class_recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    class_f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    class_specificity = calculate_specificity(all_labels, all_preds, len(class_names))
    
    # Class-wise AUC-ROC and AUC-PRC
    class_auc_roc = []
    class_auc_prc = []
    class_avg_precision = []
    
    for i in range(len(class_names)):
        try:
            y_true_binary = (all_labels == i).astype(int)
            y_score_binary = all_probs[:, i]
            
            # AUC-ROC
            fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
            auc_roc = auc(fpr, tpr)
            class_auc_roc.append(auc_roc)
            
            # AUC-PRC
            precision_curve, recall_curve, _ = precision_recall_curve(y_true_binary, y_score_binary)
            auc_prc = auc(recall_curve, precision_curve)
            class_auc_prc.append(auc_prc)
            
            # Average Precision
            avg_precision = average_precision_score(y_true_binary, y_score_binary)
            class_avg_precision.append(avg_precision)
            
        except ValueError:
            class_auc_roc.append(np.nan)
            class_auc_prc.append(np.nan)
            class_avg_precision.append(np.nan)
    
    # Store class-wise results
    results['class_precision'] = class_precision
    results['class_recall'] = class_recall
    results['class_f1'] = class_f1
    results['class_specificity'] = class_specificity
    results['class_auc_roc'] = class_auc_roc
    results['class_auc_prc'] = class_auc_prc
    results['class_avg_precision'] = class_avg_precision
    
    # Top-k accuracy (top-3 for 6 classes)
    top_k = min(3, len(class_names))
    top_k_preds = np.argsort(all_probs, axis=1)[:, -top_k:]
    top_k_correct = np.any(top_k_preds == all_labels.reshape(-1, 1), axis=1)
    results[f'top_{top_k}_accuracy'] = np.mean(top_k_correct)
    
    # Macro/Micro/Weighted averages
    results['precision_micro'] = precision_score(all_labels, all_preds, average='micro', zero_division=0)
    results['precision_weighted'] = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    results['recall_micro'] = recall_score(all_labels, all_preds, average='micro', zero_division=0)
    results['recall_weighted'] = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    results['f1_micro'] = f1_score(all_labels, all_preds, average='micro', zero_division=0)
    results['f1_weighted'] = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    # Additional metrics
    results['confusion_matrix'] = cm
    results['classification_report'] = classification_report(all_labels, all_preds, target_names=class_names)
    results['inference_memory_mb'] = inference_memory
    
    # Store data for visualizations
    results['all_labels'] = all_labels
    results['all_preds'] = all_preds  
    results['all_probs'] = all_probs
    results['all_features'] = all_features

    # Calculate KL divergence
    print("Calculating KL divergence...")
    kl_results = calculate_kl_divergence(all_labels, all_probs)
    
    # Store KL divergence results
    results['kl_divergence_mean'] = kl_results['mean_kl_divergence']
    results['kl_divergence_std'] = kl_results['std_kl_divergence']
    results['kl_divergence_median'] = kl_results['median_kl_divergence']
    results['reverse_kl_divergence_mean'] = kl_results['mean_reverse_kl_divergence']
    results['js_divergence_mean'] = kl_results['mean_js_divergence']
    results['distribution_kl_forward'] = kl_results['distribution_kl_forward']
    results['distribution_kl_reverse'] = kl_results['distribution_kl_reverse']
    results['class_kl_divergences'] = kl_results['class_kl_divergences']
    results['kl_results_full'] = kl_results

    
    return results





# Perform comprehensive evaluation
print("Performing comprehensive evaluation...")

class_names = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']
test_results = comprehensive_evaluation(model, test_loader, device, class_names)

# Measure inference time and throughput
print("Measuring inference performance...")
avg_inference_time, std_inference_time = measure_inference_time(model, test_loader, device)
throughput = 1000 / avg_inference_time  # samples per second

# Noise robustness
print("Testing noise robustness...")
noise_results = add_noise_and_evaluate(model, test_loader, device)

# Calculate training time
end_time = time.time()
total_training_time = end_time - start_time
end_date = datetime.now()

print(f"Training ended at: {end_date}")

# Print comprehensive results
print("\n" + "="*80)
print("COMPREHENSIVE EVALUATION RESULTS")
print("="*80)

print(f"\nğŸ“Š BASIC METRICS:")
print(f"Accuracy: {test_results['accuracy']:.4f}")
print(f"Precision (Macro): {test_results['precision_macro']:.4f}")
print(f"Recall (Macro): {test_results['recall_macro']:.4f}")
print(f"F1 Score (Macro): {test_results['f1_macro']:.4f}")
print(f"AUC-ROC (Macro): {test_results['auc_roc_macro']:.4f}")
print(f"Cohen's Kappa: {test_results['cohen_kappa']:.4f}")
print(f"Log Loss: {test_results['log_loss']:.4f}")
print(f"Balanced Accuracy: {test_results['balanced_accuracy']:.4f}")
print(f"Matthews Correlation Coefficient: {test_results['matthews_corrcoef']:.4f}")
print(f"Geometric Mean Score: {test_results['geometric_mean']:.4f}")

print(f"\nğŸ“ˆ CALIBRATION METRICS:")
print(f"Expected Calibration Error (ECE): {test_results['ece']:.4f}")
print(f"Brier Score: {test_results['brier_score']:.4f}")

print(f"\nâš¡ PERFORMANCE METRICS:")
print(f"Avg Inference Time: {avg_inference_time:.2f} Â± {std_inference_time:.2f} ms/sample")
print(f"Throughput: {throughput:.2f} samples/sec")
print(f"Parameter Count: {param_count / 1e6:.2f}M")
print(f"Model Size: {model_size_mb:.2f} MB")
print(f"Training Time: {total_training_time/3600:.2f} hours")
print(f"Inference Memory: {test_results['inference_memory_mb']:.2f} MB")

print(f"\nğŸ�¯ MULTI-AVERAGE METRICS:")
print(f"Precision - Micro: {test_results['precision_micro']:.4f}, Weighted: {test_results['precision_weighted']:.4f}")
print(f"Recall - Micro: {test_results['recall_micro']:.4f}, Weighted: {test_results['recall_weighted']:.4f}")
print(f"F1 Score - Micro: {test_results['f1_micro']:.4f}, Weighted: {test_results['f1_weighted']:.4f}")
print(f"Top-3 Accuracy: {test_results['top_3_accuracy']:.4f}")

print(f"\nğŸ”� CLASS-WISE METRICS:")
for i, class_name in enumerate(class_names):
    print(f"{class_name}:")
    print(f"  Precision: {test_results['class_precision'][i]:.4f}")
    print(f"  Recall: {test_results['class_recall'][i]:.4f}")
    print(f"  F1 Score: {test_results['class_f1'][i]:.4f}")
    print(f"  Specificity: {test_results['class_specificity'][i]:.4f}")
    print(f"  AUC-ROC: {test_results['class_auc_roc'][i]:.4f}")
    print(f"  AUC-PRC: {test_results['class_auc_prc'][i]:.4f}")
    print(f"  Avg Precision: {test_results['class_avg_precision'][i]:.4f}")

print(f"\nğŸ›¡ï¸� ROBUSTNESS METRICS:")
print("Noise Robustness:")
for noise_level, accuracy in noise_results.items():
    print(f"  Noise Level {noise_level}: {accuracy:.2f}%")

print(f"\nğŸ“… TIMING INFORMATION:")
print(f"Start Date: {start_date}")
print(f"End Date: {end_date}")
print(f"Total Duration: {total_training_time/60:.2f} minutes")

# Detailed Classification Report
print(f"\nğŸ“‹ CLASSIFICATION REPORT:")
print(test_results['classification_report'])

# Plot confusion matrix
def plot_confusion_matrix(cm, class_names):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix - ResNet101 + Logistic Regression')
    plt.tight_layout()
    plt.show()

print("\nğŸ”¥ CONFUSION MATRIX:")
plot_confusion_matrix(test_results['confusion_matrix'], class_names)

# Reliability diagram
print("\nğŸ“Š RELIABILITY DIAGRAM:")
max_probs = np.max(test_results['all_probs'], axis=1)
predicted_correctly = (test_results['all_preds'] == test_results['all_labels']).astype(int)
reliability_diagram(predicted_correctly, max_probs)

print(f"\nğŸ“Š KL DIVERGENCE METRICS:")
print(f"Mean KL Divergence: {test_results['kl_divergence_mean']:.4f}")
print(f"Std KL Divergence: {test_results['kl_divergence_std']:.4f}")
print(f"Median KL Divergence: {test_results['kl_divergence_median']:.4f}")
print(f"Mean Reverse KL Divergence: {test_results['reverse_kl_divergence_mean']:.4f}")
print(f"Mean Jensen-Shannon Divergence: {test_results['js_divergence_mean']:.4f}")
print(f"Distribution KL (Forward): {test_results['distribution_kl_forward']:.4f}")
print(f"Distribution KL (Reverse): {test_results['distribution_kl_reverse']:.4f}")

print(f"\nğŸ”� CLASS-WISE KL DIVERGENCES:")
for i, (class_name, kl_div) in enumerate(zip(class_names, test_results['class_kl_divergences'])):
    if not np.isnan(kl_div):
        print(f"  {class_name}: {kl_div:.4f}")

# Plot KL divergence analysis
print("\nğŸ“ˆ KL DIVERGENCE ANALYSIS:")
plot_kl_divergence_analysis(test_results['kl_results_full'], class_names)


# ROC Curves for each class
def plot_multiclass_roc_curves(y_true, y_probs, class_names):
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_score_binary = y_probs[:, i]
        
        fpr, tpr, _ = roc_curve(y_true_binary, y_score_binary)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Each Class')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\nğŸ“ˆ ROC CURVES:")
plot_multiclass_roc_curves(test_results['all_labels'], test_results['all_probs'], class_names)

# Precision-Recall Curves
def plot_multiclass_pr_curves(y_true, y_probs, class_names):
    plt.figure(figsize=(12, 8))
    
    for i, class_name in enumerate(class_names):
        y_true_binary = (y_true == i).astype(int)
        y_score_binary = y_probs[:, i]
        
        precision, recall, _ = precision_recall_curve(y_true_binary, y_score_binary)
        pr_auc = auc(recall, precision)
        avg_precision = average_precision_score(y_true_binary, y_score_binary)
        
        plt.plot(recall, precision, label=f'{class_name} (AP = {avg_precision:.3f})')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves for Each Class')
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

print("\nğŸ“Š PRECISION-RECALL CURVES:")
plot_multiclass_pr_curves(test_results['all_labels'], test_results['all_probs'], class_names)

# t-SNE Visualization
def plot_tsne(features, labels, class_names, title="t-SNE Visualization"):
    print("Computing t-SNE... This may take a while...")
    
    # Subsample for faster computation if dataset is large
    if len(features) > 2000:
        indices = np.random.choice(len(features), 2000, replace=False)
        features = features[indices]
        labels = labels[indices]
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
    features_2d = tsne.fit_transform(features)
    
    # Plot
    plt.figure(figsize=(12, 8))
    colors = plt.cm.Set3(np.linspace(0, 1, len(class_names)))
    
    for i, (class_name, color) in enumerate(zip(class_names, colors)):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], 
                   c=[color], label=class_name, alpha=0.7, s=20)
    
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

print("\nğŸ—ºï¸� t-SNE VISUALIZATION:")
plot_tsne(test_results['all_features'], test_results['all_labels'], class_names, 
          "t-SNE Visualization of Learned Features")

# Feature importance heatmap (using gradients)
def plot_feature_importance_heatmap(model, test_loader, device, class_names):
    model.eval()
    
    # Get one batch for gradient analysis
    images, targets = next(iter(test_loader))
    images = images.to(device)
    images.requires_grad_(True)
    
    # Forward pass
    outputs = model(images)
    
    # Calculate gradients for each class
    gradients = []
    for class_idx in range(len(class_names)):
        model.zero_grad()
        class_output = outputs[:, class_idx].sum()
        class_output.backward(retain_graph=True)
        
        # Get gradient magnitude
        grad = torch.abs(images.grad).mean(dim=(0, 1)).cpu().numpy()
        gradients.append(grad)
    
    # Plot heatmap
    gradients = np.array(gradients)
    plt.figure(figsize=(12, 8))
    sns.heatmap(gradients, xticklabels=False, yticklabels=class_names, 
                cmap='viridis', cbar=True)
    plt.title('Feature Importance Heatmap (Gradient-based)')
    plt.xlabel('Input Features')
    plt.ylabel('Classes')
    plt.tight_layout()
    plt.show()

print("\nğŸ”¥ FEATURE IMPORTANCE:")
try:
    plot_feature_importance_heatmap(model, test_loader, device, class_names)
except Exception as e:
    print(f"Could not generate feature importance heatmap: {e}")

# Learning curve analysis
def plot_detailed_learning_curves(history):
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss comparison
    axes[0, 0].plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training vs Validation Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Accuracy comparison
    axes[0, 1].plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Training vs Validation Accuracy')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning rate schedule
    axes[0, 2].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
    axes[0, 2].set_title('Learning Rate Schedule')
    axes[0, 2].set_xlabel('Epoch')
    axes[0, 2].set_ylabel('Learning Rate')
    axes[0, 2].set_yscale('log')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Loss difference (overfitting indicator)
    loss_diff = np.array(history['val_loss']) - np.array(history['train_loss'])
    axes[1, 0].plot(epochs, loss_diff, 'purple', linewidth=2)
    axes[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 0].set_title('Validation - Training Loss (Overfitting Indicator)')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Loss Difference')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Accuracy difference
    acc_diff = np.array(history['train_acc']) - np.array(history['val_acc'])
    axes[1, 1].plot(epochs, acc_diff, 'orange', linewidth=2)
    axes[1, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[1, 1].set_title('Training - Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy Difference (%)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Combined metrics
    ax = axes[1, 2]
    ax2 = ax.twinx()
    
    line1 = ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    line2 = ax2.plot(epochs, history['val_acc'], 'b-', label='Val Accuracy', linewidth=2)
    
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss', color='red')
    ax2.set_ylabel('Validation Accuracy (%)', color='blue')
    ax.set_title('Validation Metrics Combined')
    
    # Combine legends
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, loc='center right')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

print("\nğŸ“Š DETAILED LEARNING CURVES:")
plot_detailed_learning_curves(history)

# Performance summary table
def create_performance_summary():
    summary_data = {
        'Metric': [
            'Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)',
            'AUC-ROC (Macro)', 'Cohen\'s Kappa', 'Matthews Corr Coef', 'Balanced Accuracy',
            'Top-3 Accuracy', 'Expected Calibration Error', 'Brier Score',
            'Avg Inference Time (ms)', 'Throughput (samples/sec)', 'Model Size (MB)',
            'Training Time (hrs)', 'Parameter Count (M)'
        ],
        'Value': [
            f"{test_results['accuracy']:.4f}",
            f"{test_results['precision_macro']:.4f}",
            f"{test_results['recall_macro']:.4f}",
            f"{test_results['f1_macro']:.4f}",
            f"{test_results['auc_roc_macro']:.4f}",
            f"{test_results['cohen_kappa']:.4f}",
            f"{test_results['matthews_corrcoef']:.4f}",
            f"{test_results['balanced_accuracy']:.4f}",
            f"{test_results['top_3_accuracy']:.4f}",
            f"{test_results['ece']:.4f}",
            f"{test_results['brier_score']:.4f}",
            f"{avg_inference_time:.2f}",
            f"{throughput:.2f}",
            f"{model_size_mb:.2f}",
            f"{total_training_time/3600:.2f}",
            f"{param_count/1e6:.2f}"
        ]
    }
    
    summary_df = pd.DataFrame(summary_data)
    print("\nğŸ“‹ PERFORMANCE SUMMARY TABLE:")
    print(summary_df.to_string(index=False))
    
    return summary_df

summary_df = create_performance_summary()

# Class-wise performance table
def create_classwise_summary():
    classwise_data = {
        'Class': class_names,
        'Precision': [f"{p:.4f}" for p in test_results['class_precision']],
        'Recall': [f"{r:.4f}" for r in test_results['class_recall']],
        'F1-Score': [f"{f:.4f}" for f in test_results['class_f1']],
        'Specificity': [f"{s:.4f}" for s in test_results['class_specificity']],
        'AUC-ROC': [f"{a:.4f}" for a in test_results['class_auc_roc']],
        'AUC-PRC': [f"{a:.4f}" for a in test_results['class_auc_prc']],
        'Avg Precision': [f"{a:.4f}" for a in test_results['class_avg_precision']]
    }
    
    classwise_df = pd.DataFrame(classwise_data)
    print("\nğŸ“Š CLASS-WISE PERFORMANCE TABLE:")
    print(classwise_df.to_string(index=False))
    
    return classwise_df

classwise_df = create_classwise_summary()

# Save results to files
print("\nğŸ’¾ SAVING RESULTS...")

# Save performance summary
summary_df.to_csv('performance_summary.csv', index=False)
classwise_df.to_csv('classwise_performance.csv', index=False)

# Save detailed results as JSON
import json
results_dict = {
    'basic_metrics': {
        'accuracy': float(test_results['accuracy']),
        'precision_macro': float(test_results['precision_macro']),
        'recall_macro': float(test_results['recall_macro']),
        'f1_macro': float(test_results['f1_macro']),
        'auc_roc_macro': float(test_results['auc_roc_macro']) if not np.isnan(test_results['auc_roc_macro']) else None,
        'cohen_kappa': float(test_results['cohen_kappa']),
        'log_loss': float(test_results['log_loss']) if not np.isnan(test_results['log_loss']) else None,
        'balanced_accuracy': float(test_results['balanced_accuracy']),
        'matthews_corrcoef': float(test_results['matthews_corrcoef']),
        'geometric_mean': float(test_results['geometric_mean'])
    },
    'calibration_metrics': {
        'ece': float(test_results['ece']),
        'brier_score': float(test_results['brier_score'])
    },
    'performance_metrics': {
        'avg_inference_time_ms': float(avg_inference_time),
        'throughput_samples_per_sec': float(throughput),
        'parameter_count_M': float(param_count / 1e6),
        'model_size_MB': float(model_size_mb),
        'training_time_hours': float(total_training_time / 3600),
        'inference_memory_MB': float(test_results['inference_memory_mb'])
    },
    'multi_average_metrics': {
        'precision_micro': float(test_results['precision_micro']),
        'precision_weighted': float(test_results['precision_weighted']),
        'recall_micro': float(test_results['recall_micro']),
        'recall_weighted': float(test_results['recall_weighted']),
        'f1_micro': float(test_results['f1_micro']),
        'f1_weighted': float(test_results['f1_weighted']),
        'top_3_accuracy': float(test_results['top_3_accuracy'])
    },
    'noise_robustness': {f'noise_{k}': float(v) for k, v in noise_results.items()},
    'timing': {
        'start_date': start_date.isoformat(),
        'end_date': end_date.isoformat(),
        'total_duration_minutes': float(total_training_time / 60)
    }
}

with open('detailed_results.json', 'w') as f:
    json.dump(results_dict, f, indent=2)

# Save training history
history_df = pd.DataFrame(history)
history_df.to_csv('training_history.csv', index=False)

print("âœ… All results saved successfully!")
print("\nFiles created:")
print("- performance_summary.csv")
print("- classwise_performance.csv") 
print("- detailed_results.json")
print("- training_history.csv")
print("- best_resnet101_model.pth")

print("\nğŸ�‰ EVALUATION COMPLETE!")
print("="*80)




