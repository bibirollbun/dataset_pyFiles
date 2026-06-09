import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import warnings
from collections import defaultdict
import random
import uuid

warnings.filterwarnings('ignore')

# Configuration
TEST_PATH = "/kaggle/working/"
BASE_PATH = "/kaggle/input/hms-harmful-brain-activity-classification"
PREPROCESSED_PATH = "/kaggle/input/preprocessing/preprocessed/eeg"
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train.csv")
MODEL_OUTPUT_PATH = os.path.join(TEST_PATH, "models")
FEATURE_CACHE_PATH = os.path.join(TEST_PATH, "feature_cache")

os.makedirs(MODEL_OUTPUT_PATH, exist_ok=True)
os.makedirs(FEATURE_CACHE_PATH, exist_ok=True)

CLASSES = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
N_CLASSES = len(CLASSES)
BATCH_SIZE = 32
EPOCHS = 20  # Increased for better convergence
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# KL Divergence function
def kl_divergence_torch(y_true, y_pred_proba, epsilon=1e-7):  # Slightly larger epsilon for stability
    y_pred_proba = torch.clamp(y_pred_proba, epsilon, 1 - epsilon)
    y_true = torch.clamp(y_true, epsilon, 1.0)
    kl_div = torch.sum(y_true * torch.log(y_true / y_pred_proba), dim=1)
    return torch.mean(kl_div).item()

# Load and preprocess training labels
try:
    train_df = pd.read_csv(TRAIN_LABELS_PATH)
    print(f"Loaded {len(train_df)} annotations with {len(train_df['eeg_id'].unique())} unique EEG IDs")
except Exception as e:
    print(f"Error loading training labels: {e}")
    exit()

# Encode labels
label_encoder = LabelEncoder()
label_encoder.fit(CLASSES)
train_df['label'] = label_encoder.transform(train_df['expert_consensus'])

# Display original class distribution
print("Original class distribution:")
print(train_df['expert_consensus'].value_counts(normalize=True))

# Custom Dataset for EEG data
class EEGDataset(Dataset):
    def __init__(self, eeg_ids, labels, data_path):
        self.eeg_ids = eeg_ids
        self.labels = labels
        self.data_path = data_path
    
    def __len__(self):
        return len(self.eeg_ids)
    
    def __getitem__(self, idx):
        eeg_id = self.eeg_ids[idx]
        eeg_path = os.path.join(self.data_path, f"{eeg_id}.npy")
        try:
            eeg_data = np.load(eeg_path).astype(np.float32)  # Shape: (19, 2500)
            # Impute NaN/Inf with channel mean
            if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
                print(f"Warning: NaN or Inf in EEG data for {eeg_id}, imputing with channel mean")
                for ch in range(eeg_data.shape[0]):
                    channel = eeg_data[ch, :]
                    mask = np.isnan(channel) | np.isinf(channel)
                    if np.any(mask):
                        channel[mask] = np.nanmean(channel)
            # Standardize each channel
            eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / (np.std(eeg_data, axis=1, keepdims=True) + 1e-7)
            # Transpose to (2500, 19) for LSTM: (timesteps, features)
            eeg_data = eeg_data.T
            label = self.labels[idx]
            return torch.tensor(eeg_data, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading EEG data for {eeg_id}: {e}")
            return torch.zeros((2500, 19), dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

# Load data
print("Loading data...")
success_file_path = os.path.join(os.path.dirname(PREPROCESSED_PATH), "success.csv")
if os.path.exists(success_file_path):
    try:
        success_df = pd.read_csv(success_file_path)
        success_ids = set(success_df['eeg_id'].astype(str).tolist())
        print(f"Found success file with {len(success_ids)} successful preprocessing entries")
    except Exception as e:
        print(f"Error loading success file: {e}")
        success_ids = set(train_df['eeg_id'].astype(str).tolist())
else:
    print("Success file not found, using all available EEG IDs")
    success_ids = set(train_df['eeg_id'].astype(str).tolist())

# Filter samples based on success_ids
valid_samples = train_df[train_df['eeg_id'].astype(str).isin(success_ids)]
eeg_ids = valid_samples['eeg_id'].astype(str).tolist()
labels = valid_samples['label'].values
print(f"Processing {len(eeg_ids)} valid samples")

# Train-validation split
train_ids, val_ids, train_labels, val_labels = train_test_split(
    eeg_ids, labels, test_size=0.2, stratify=labels, random_state=42
)
print(f"Training samples: {len(train_ids)}, Validation samples: {len(val_ids)}")

# Verify class distribution in training set
train_class_counts = pd.Series(train_labels).value_counts(normalize=True)
train_class_counts.index = [CLASSES[i] for i in train_class_counts.index]
print("\nClass distribution in training set:")
print(train_class_counts)

# Create datasets and dataloaders
train_dataset = EEGDataset(train_ids, train_labels, PREPROCESSED_PATH)
val_dataset = EEGDataset(val_ids, val_labels, PREPROCESSED_PATH)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# LSTM Model with enhancements
class EEG_LSTM(nn.Module):
    def __init__(self, input_size=19, hidden_size=128, num_layers=3, output_size=N_CLASSES):
        super(EEG_LSTM, self).__init__()
        self.conv1d = nn.Conv1d(in_channels=19, out_channels=32, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(32, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.4)  # Increased dropout
    
    def forward(self, x, temperature=1.5):  # Temperature for probability calibration
        x = self.conv1d(x.transpose(1, 2)).transpose(1, 2)  # Conv1D: (batch, 2500, 32)
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        out = self.dropout(last_out)
        out = self.fc(out) / temperature
        return out

# Initialize model, loss, and optimizer
class_counts = train_df['label'].value_counts().sort_index().values
class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(DEVICE)
model = EEG_LSTM(input_size=19, hidden_size=128, num_layers=3, output_size=N_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)  # Label smoothing added
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # L2 regularization
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

# Training loop
print("Training LSTM model...")
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
train_kls = []
val_kls = []
lrs = []
best_val_kl = float('inf')
best_model_path = os.path.join(MODEL_OUTPUT_PATH, "lstm_best_model.pth")
patience = 3
counter = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    train_kl_sum = 0.0
    
    for batch_data, batch_labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        batch_data, batch_labels = batch_data.to(DEVICE), batch_labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(batch_data, temperature=1.5)  # Temperature scaling
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * batch_data.size(0)
        _, predicted = torch.max(outputs, 1)
        train_total += batch_labels.size(0)
        train_correct += (predicted == batch_labels).sum().item()
        
        # Compute KL divergence
        pred_proba = torch.softmax(outputs, dim=1)
        true_one_hot = torch.nn.functional.one_hot(batch_labels, N_CLASSES).float()
        train_kl_sum += kl_divergence_torch(true_one_hot, pred_proba) * batch_data.size(0)
    
    train_loss /= len(train_dataset)
    train_acc = train_correct / train_total
    train_kl = train_kl_sum / len(train_dataset)
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    val_kl_sum = 0.0
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch_data, batch_labels in val_loader:
            batch_data, batch_labels = batch_data.to(DEVICE), batch_labels.to(DEVICE)
            outputs = model(batch_data, temperature=1.5)
            loss = criterion(outputs, batch_labels)
            
            val_loss += loss.item() * batch_data.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += batch_labels.size(0)
            val_correct += (predicted == batch_labels).sum().item()
            
            # Collect predictions for confusion matrix
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(batch_labels.cpu().numpy())
            
            # Compute KL divergence
            pred_proba = torch.softmax(outputs, dim=1)
            true_one_hot = torch.nn.functional.one_hot(batch_labels, N_CLASSES).float()
            val_kl_sum += kl_divergence_torch(true_one_hot, pred_proba) * batch_data.size(0)
    
    val_loss /= len(val_dataset)
    val_acc = val_correct / val_total
    val_kl = val_kl_sum / len(val_dataset)
    
    # Record learning rate
    current_lr = optimizer.param_groups[0]['lr']
    lrs.append(current_lr)
    scheduler.step(val_kl)  # Schedule based on KL divergence
    
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    train_accuracies.append(train_acc)
    val_accuracies.append(val_acc)
    train_kls.append(train_kl)
    val_kls.append(val_kl)
    
    print(f"Epoch {epoch+1}/{EPOCHS}")
    print(f"Train Loss: {train_loss:.6f}, Train Accuracy: {train_acc:.6f}, Train KL Divergence: {train_kl:.6f}")
    print(f"Val Loss: {val_loss:.6f}, Val Accuracy: {val_acc:.6f}, Val KL Divergence: {val_kl:.6f}")
    
    # Early stopping based on validation KL divergence
    if val_kl < best_val_kl:
        best_val_kl = val_kl
        counter = 0
        torch.save(model.state_dict(), best_model_path)
        print(f"Saved best model at epoch {epoch+1} with val KL: {val_kl:.6f}")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered")
            break

# Plot training and validation metrics
plt.figure(figsize=(15, 10))

# Loss plot
plt.subplot(2, 2, 1)
plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Accuracy plot
plt.subplot(2, 2, 2)
plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Val Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# KL Divergence plot
plt.subplot(2, 2, 3)
plt.plot(range(1, len(train_kls)+1), train_kls, label='Train KL Divergence')
plt.plot(range(1, len(val_kls)+1), val_kls, label='Val KL Divergence')
plt.title('Training and Validation KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('KL Divergence')
plt.legend()

# Learning Rate vs Epochs
plt.subplot(2, 2, 4)
plt.plot(range(1, len(lrs)+1), lrs, label='Learning Rate')
plt.title('Learning Rate vs Epoch')
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'training_metrics.png'), dpi=100, bbox_inches='tight')
plt.close()

# Confusion Matrix
cm = confusion_matrix(val_true, val_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix (Validation Set)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(MODEL_OUTPUT_PATH, 'confusion_matrix.png'), dpi=100, bbox_inches='tight')
plt.close()

# Load best model for submission
model.load_state_dict(torch.load(best_model_path))
model.eval()

# Submission file creation
def create_prediction_file():
    test_eeg_path = os.path.join(BASE_PATH, "test_eegs")
    if not os.path.exists(test_eeg_path):
        print("Warning: Test EEG path not found at", test_eeg_path)
        print("Creating dummy submission")
        submission_df = pd.DataFrame({'eeg_id': ['dummy_1', 'dummy_2']})
        for cls in CLASSES:
            submission_df[cls] = 1.0 / N_CLASSES
        submission_path = os.path.join(TEST_PATH, "submission.csv")
        submission_df.to_csv(submission_path, index=False)
        print("Dummy submission file saved to:", submission_path)
        return submission_df
    
    test_files = [f.replace(".parquet", "") for f in os.listdir(test_eeg_path) if f.endswith(".parquet")]
    if len(test_files) == 0:
        print("No test files found in", test_eeg_path)
        return None

    submission_df = pd.DataFrame({'eeg_id': test_files})
    for cls in CLASSES:
        submission_df[cls] = 0.0

    print("Generating predictions for test data...")
    predictions_made = 0
    
    for eeg_id in tqdm(test_files[:100]):
        try:
            eeg_path = os.path.join(PREPROCESSED_PATH, f"{eeg_id}.npy")
            if not os.path.exists(eeg_path):
                print(f"Test EEG file not found for {eeg_id} at {eeg_path}")
                raise FileNotFoundError
            
            eeg_data = np.load(eeg_path).astype(np.float32)  # Shape: (19, 2500)
            if np.any(np.isnan(eeg_data)) or np.any(np.isinf(eeg_data)):
                print(f"Warning: NaN or Inf in test EEG data for {eeg_id}, imputing with channel mean")
                for ch in range(eeg_data.shape[0]):
                    channel = eeg_data[ch, :]
                    mask = np.isnan(channel) | np.isinf(channel)
                    if np.any(mask):
                        channel[mask] = np.nanmean(channel)
            eeg_data = (eeg_data - np.mean(eeg_data, axis=1, keepdims=True)) / (np.std(eeg_data, axis=1, keepdims=True) + 1e-7)
            eeg_data = eeg_data.T  # Shape: (2500, 19)
            eeg_data = torch.tensor(eeg_data, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # Shape: (1, 2500, 19)
            
            with torch.no_grad():
                outputs = model(eeg_data, temperature=1.5)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
            
            for i, cls in enumerate(CLASSES):
                submission_df.loc[submission_df['eeg_id'] == eeg_id, cls] = probs[i]
            predictions_made += 1
            
        except Exception as e:
            print(f"Error processing test file {eeg_id}: {e}")
            uniform_prob = 1.0 / N_CLASSES
            for cls in CLASSES:
                submission_df.loc[submission_df['eeg_id'] == eeg_id, cls] = uniform_prob

    submission_path = os.path.join(TEST_PATH, "submission.csv")
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission file saved to: {submission_path}")
    print(f"Made predictions for {predictions_made}/{len(test_files)} test files")
    return submission_df

# Create submission file
try:
    submission_df = create_prediction_file()
except Exception as e:
    print(f"Error creating submission file: {e}")

print("\nTraining completed!")
print(f"Best validation KL divergence: {best_val_kl:.6f}")
print(f"Last epoch train accuracy: {train_accuracies[-1]:.6f}, validation accuracy: {val_accuracies[-1]:.6f}")
print(f"Last epoch train KL divergence: {train_kls[-1]:.6f}, validation KL divergence: {val_kls[-1]:.6f}")

