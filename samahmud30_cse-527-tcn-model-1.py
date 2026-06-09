pip uninstall -y scikit-learn


pip install scikit-learn


pip uninstall scikit-learn imblearn --yes



pip install scikit-learn==1.2.2 imbalanced-learn==0.10.1



# ==================== TCN ON 50K SAMPLES ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm

print("="*80)
print("â�±ï¸�  TCN TRAINING ON 50,000 SAMPLES")
print("="*80)

# 1. DATA LOADING
print("\nğŸ“‚ LOADING DATA...")
csv_path = '/kaggle/input/hms-harmful-brain-activity-classification/train.csv'
train_csv = pd.read_csv(csv_path)
print(f"âœ… Loaded {len(train_csv):,} samples")

# Process labels - FIXED: Handle all-zero rows properly
label_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
votes = train_csv[label_cols].values.astype(float)

# Check for rows with all zeros
zero_rows = np.all(votes == 0, axis=1)
if zero_rows.any():
    print(f"âš ï¸�  Found {zero_rows.sum():,} rows with all zero votes. Setting to uniform distribution.")
    votes[zero_rows] = 1.0  # Set to uniform distribution

row_sums = votes.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1  # Avoid division by zero
probabilities = votes / row_sums
train_csv['dominant_class'] = np.argmax(probabilities, axis=1)

# Select 50K samples
n_samples = 50000
if len(train_csv) > n_samples:
    _, sample_indices = train_test_split(
        range(len(train_csv)),
        test_size=n_samples,
        random_state=42,
        stratify=train_csv['dominant_class']
    )
    selected_df = train_csv.iloc[sample_indices].copy()
else:
    selected_df = train_csv.copy()
    n_samples = len(selected_df)

print(f"Selected {n_samples:,} samples")

# 2. SEQUENCE DATA EXTRACTION
print("\nğŸ”§ EXTRACTING SEQUENCE DATA...")

DATA_DIR = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs'

def extract_tcn_sequence(eeg_data):
    """Extract sequence data for TCN (channels first)"""
    if eeg_data.size == 0 or eeg_data.shape[0] == 0:
        return np.zeros((4, 100), dtype=np.float32)
    
    # Downsample to 100 timesteps
    if eeg_data.shape[0] > 100:
        indices = np.linspace(0, eeg_data.shape[0]-1, 100).astype(int)
        eeg_data = eeg_data[indices, :]
    elif eeg_data.shape[0] < 100:
        # Pad if shorter than 100
        pad_width = ((0, 100 - eeg_data.shape[0]), (0, 0))
        eeg_data = np.pad(eeg_data, pad_width, mode='constant')
    
    # Use first 4 channels
    if eeg_data.shape[1] > 4:
        eeg_data = eeg_data[:, :4]
    elif eeg_data.shape[1] < 4:
        eeg_data = np.pad(eeg_data, ((0, 0), (0, 4 - eeg_data.shape[1])), mode='constant')
    
    # Normalize per channel with robust normalization
    for i in range(eeg_data.shape[1]):
        channel = eeg_data[:, i]
        # Check for constant channel
        if np.std(channel) < 1e-8:
            eeg_data[:, i] = np.random.normal(0, 1e-8, size=channel.shape)
        else:
            # Robust normalization
            median = np.median(channel)
            iqr = np.percentile(channel, 75) - np.percentile(channel, 25)
            if iqr < 1e-8:
                eeg_data[:, i] = (channel - median) / (np.std(channel) + 1e-8)
            else:
                eeg_data[:, i] = (channel - median) / (iqr + 1e-8)
    
    # Transpose to [channels, timesteps] for TCN
    return eeg_data.T.astype(np.float32)

# Process samples
sequence_list = []
labels_list = []
eeg_ids_list = []
failed = 0

processing_start = time.time()

for idx in tqdm(range(n_samples), desc="Processing EEGs"):
    try:
        sample_id = selected_df.iloc[idx]['eeg_id']
        eeg_path = f"{DATA_DIR}/{sample_id}.parquet"
        
        if os.path.exists(eeg_path):
            eeg_df = pd.read_parquet(eeg_path)
            if not eeg_df.empty and eeg_df.shape[0] > 10:  # Reduced minimum length
                eeg_data = eeg_df.values.astype(np.float32)
                
                # Handle NaN values
                if np.any(np.isnan(eeg_data)):
                    # Fill NaN with forward fill then backward fill
                    eeg_df_clean = eeg_df.fillna(method='ffill').fillna(method='bfill').fillna(0)
                    eeg_data = eeg_df_clean.values.astype(np.float32)
                
                sequence = extract_tcn_sequence(eeg_data)
                
                # Check if sequence is valid
                if not np.any(np.isnan(sequence)) and not np.any(np.isinf(sequence)):
                    sequence_list.append(sequence)
                    labels_list.append(selected_df.iloc[idx]['dominant_class'])
                    eeg_ids_list.append(sample_id)
                else:
                    failed += 1
            else:
                failed += 1
        else:
            failed += 1
    except Exception as e:
        failed += 1
        # Uncomment for debugging
        # print(f"Error processing sample {sample_id}: {e}")

processing_time = time.time() - processing_start
print(f"\nâœ… Processed {len(sequence_list):,}/{n_samples} sequences")
print(f"   Failed: {failed:,}")
print(f"   Time: {processing_time:.1f}s")

# Check if we have enough data
if len(sequence_list) < 1000:
    raise ValueError(f"Only {len(sequence_list)} sequences processed successfully. Need at least 1000.")

# 3. DATA PREPARATION
y = np.array(labels_list, dtype=np.int64)  # Changed to int64 for PyTorch compatibility

# Split data
train_idx, val_idx = train_test_split(
    range(len(sequence_list)),
    test_size=0.1,
    random_state=42,
    stratify=y
)

print(f"\nğŸ“Š DATASET SIZES:")
print(f"   Training: {len(train_idx):,} sequences")
print(f"   Validation: {len(val_idx):,} sequences")
print(f"   Sequence shape: {sequence_list[0].shape}")

# 4. TCN MODEL
class EEG_TCN(nn.Module):
    """Enhanced TCN with residual connections and stability improvements"""
    def __init__(self, input_channels=4, num_classes=6, num_filters=64):
        super().__init__()
        
        # First residual block
        self.tcn1 = nn.Conv1d(input_channels, num_filters, kernel_size=3, padding=1, bias=False)
        self.tcn2 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(num_filters)
        self.bn2 = nn.BatchNorm1d(num_filters)
        self.downsample1 = nn.Conv1d(input_channels, num_filters, kernel_size=1, bias=False) if input_channels != num_filters else nn.Identity()
        
        # Second residual block
        self.tcn3 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=2, dilation=2, bias=False)
        self.tcn4 = nn.Conv1d(num_filters, num_filters, kernel_size=3, padding=2, dilation=2, bias=False)
        self.bn3 = nn.BatchNorm1d(num_filters)
        self.bn4 = nn.BatchNorm1d(num_filters)
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classifier with better stability
        self.classifier = nn.Sequential(
            nn.Linear(num_filters, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # First residual block
        identity = x
        out = F.relu(self.bn1(self.tcn1(x)), inplace=True)
        out = self.bn2(self.tcn2(out))
        
        identity = self.downsample1(identity)
        out += identity
        out = F.relu(out, inplace=True)
        
        # Second residual block
        identity = out
        out = F.relu(self.bn3(self.tcn3(out)), inplace=True)
        out = self.bn4(self.tcn4(out))
        out += identity
        out = F.relu(out, inplace=True)
        
        # Global pooling
        out = self.global_pool(out).squeeze(2)
        
        # Classifier
        out = self.classifier(out)
        return F.log_softmax(out, dim=1)

# 5. TRAINING WITH LOSS CURVES
print(f"\n{'='*60}")
print("â�±ï¸�  TRAINING TCN WITH LOSS CURVES")
print(f"{'='*60}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create datasets with validation
class TCNDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # Add small noise for stability during training
        seq = self.sequences[idx].copy()
        label = self.labels[idx]
        return torch.FloatTensor(seq), torch.tensor(label, dtype=torch.long)

train_dataset = TCNDataset(
    [sequence_list[i] for i in train_idx],
    [labels_list[i] for i in train_idx]
)
val_dataset = TCNDataset(
    [sequence_list[i] for i in val_idx],
    [labels_list[i] for i in val_idx]
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True
)

# Initialize model
model = EEG_TCN(input_channels=4, num_classes=6, num_filters=64)
model = model.to(device)

# Weighted loss for class imbalance - FIXED: Handle class counts properly
all_labels = torch.tensor(labels_list, dtype=torch.long)
class_counts = torch.bincount(all_labels)
print(f"Class counts: {class_counts.tolist()}")

# Calculate weights with smoothing
class_weights = 1.0 / (class_counts.float() + 1e-8)
class_weights = class_weights / class_weights.sum()
print(f"Class weights: {class_weights.tolist()}")
criterion = nn.NLLLoss(weight=class_weights.to(device))

# Use gradient clipping and weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

# Track metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
learning_rates = []

num_epochs = 15
best_val_acc = 0
patience_counter = 0
max_patience = 5

print(f"\nTraining for {num_epochs} epochs...")

start_time = time.time()

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    for sequences, labels in train_pbar:
        sequences, labels = sequences.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        # Update progress bar
        train_pbar.set_postfix({'loss': loss.item(), 'acc': train_correct/train_total})
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = train_correct / train_total
    train_losses.append(avg_train_loss)
    train_accuracies.append(train_acc)
    
    # Validation phase
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    all_preds = []
    all_labels_val = []
    
    val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
    with torch.no_grad():
        for sequences, labels in val_pbar:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels_val.extend(labels.cpu().numpy())
            
            # Update progress bar
            val_pbar.set_postfix({'loss': loss.item(), 'acc': val_correct/val_total})
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    # Update scheduler
    scheduler.step(val_acc)
    
    print(f"\n  Epoch {epoch+1}:")
    print(f"    Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"    Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"    LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Check for NaN
    if np.isnan(avg_train_loss) or np.isnan(avg_val_loss):
        print(f"  âš ï¸�  NaN detected! Stopping training.")
        break
    
    # Early stopping
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        # Save best model
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
        }, 'best_tcn_model.pth')
        print(f"    ğŸ’¾ Saved best model with val_acc: {val_acc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"  â�¹ï¸�  Early stopping at epoch {epoch+1}")
            break

total_time = time.time() - start_time

# Load best model
if os.path.exists('best_tcn_model.pth'):
    checkpoint = torch.load('best_tcn_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded best model from epoch {checkpoint['epoch']} with val_acc: {checkpoint['val_acc']:.4f}")

# Final evaluation
model.eval()
all_preds = []
all_labels_final = []

with torch.no_grad():
    for sequences, labels in val_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels_final.extend(labels.cpu().numpy())

final_val_acc = accuracy_score(all_labels_final, all_preds)

print(f"\nâœ… TCN Training Complete!")
print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
print(f"   Total Training Time: {total_time:.1f}s")

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'config': {
        'input_channels': 4,
        'num_classes': 6,
        'num_filters': 64
    },
    'val_acc': final_val_acc
}, 'tcn_50k_model.pth')
print("   Model saved as 'tcn_50k_model.pth'")

# 6. CLASSIFICATION REPORT
print(f"\n{'='*60}")
print("ğŸ“Š CLASSIFICATION REPORT")
print(f"{'='*60}")

class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

print("\nğŸ“‹ Detailed Classification Report:")
report = classification_report(all_labels_final, all_preds, target_names=class_names, digits=4)
print(report)

# Confusion Matrix
cm = confusion_matrix(all_labels_final, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('TCN - Confusion Matrix (50K Samples)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('tcn_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. LOSS CURVES VISUALIZATION
print(f"\n{'='*60}")
print("ğŸ“ˆ LOSS AND ACCURACY CURVES")
print(f"{'='*60}")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Loss Curves
epochs_range = range(1, len(train_losses) + 1)

ax1 = axes[0, 0]
ax1.plot(epochs_range, train_losses, 'o-', linewidth=2, markersize=6, label='Training Loss', color='blue')
ax1.plot(epochs_range, val_losses, 's-', linewidth=2, markersize=6, label='Validation Loss', color='red')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Accuracy Curves
ax2 = axes[0, 1]
ax2.plot(epochs_range, train_accuracies, 'o-', linewidth=2, markersize=6, label='Training Accuracy', color='blue')
ax2.plot(epochs_range, val_accuracies, 's-', linewidth=2, markersize=6, label='Validation Accuracy', color='red')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Learning Rate Curve
ax3 = axes[1, 0]
ax3.plot(epochs_range, learning_rates[:len(epochs_range)], '^-', linewidth=2, markersize=8, label='Learning Rate', color='green')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Learning Rate', fontsize=12)
ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')
ax3.legend()

# Training Summary
ax4 = axes[1, 1]
ax4.axis('off')
summary_text = f"""
Training Summary:
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Best Val Acc: {best_val_acc:.4f}
Final Val Acc: {final_val_acc:.4f}
Total Epochs: {len(train_losses)}
Total Time: {total_time:.1f}s
â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
Class Distribution:
{chr(10).join([f'{class_names[i]}: {class_counts[i].item()}' for i in range(6)])}
"""
ax4.text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('tcn_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\n{'='*80}")
print("ğŸ�¯ TCN TRAINING COMPLETE!")
print(f"{'='*80}")

