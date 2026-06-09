!pip install torch-geometric
!pip install numpy



# ==================== TRANSFORMER ON 50K SAMPLES - FIXED VERSION ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import time
import os
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence

print("="*80)
print("âš¡ TRANSFORMER TRAINING ON 50,000 SAMPLES - FIXED VERSION")
print("="*80)

# 1. DATA LOADING
print("\nğŸ“‚ LOADING DATA...")
csv_path = '/kaggle/input/hms-harmful-brain-activity-classification/train.csv'
train_csv = pd.read_csv(csv_path)
print(f"âœ… Loaded {len(train_csv):,} samples")

# Process labels
label_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
votes = train_csv[label_cols].values.astype(float)
row_sums = votes.sum(axis=1, keepdims=True)
row_sums[row_sums == 0] = 1
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
    selected_df = train_csv.iloc[sample_indices]
else:
    selected_df = train_csv
    n_samples = len(selected_df)

print(f"Selected {n_samples:,} samples")

# 2. SEQUENCE DATA EXTRACTION - FIXED
print("\nğŸ”§ EXTRACTING SEQUENCE DATA...")

DATA_DIR = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs'

def extract_sequence_data(eeg_data):
    """Extract sequence data for Transformer - FIXED VERSION"""
    if eeg_data.size == 0:
        return np.zeros((100, 4), dtype=np.float32) + 1e-8
    
    # Downsample to 100 timesteps
    if eeg_data.shape[0] > 100:
        indices = np.linspace(0, eeg_data.shape[0]-1, 100).astype(int)
        eeg_data = eeg_data[indices, :]
    
    # Use first 4 channels
    if eeg_data.shape[1] > 4:
        eeg_data = eeg_data[:, :4]
    elif eeg_data.shape[1] < 4:
        eeg_data = np.pad(eeg_data, ((0, 0), (0, 4 - eeg_data.shape[1])), mode='constant', constant_values=1e-8)
    
    # SAFER Normalization per channel with robust statistics
    eps = 1e-6
    for ch in range(eeg_data.shape[1]):
        channel_data = eeg_data[:, ch]
        
        # Handle constant channels
        if np.std(channel_data) < eps:
            eeg_data[:, ch] = np.random.normal(0, eps, size=channel_data.shape)
        else:
            # Robust scaling using median and IQR
            median_val = np.median(channel_data)
            q75, q25 = np.percentile(channel_data, [75, 25])
            iqr = q75 - q25 + eps
            
            if iqr < eps:
                # If IQR is too small, use standard deviation
                scale = np.std(channel_data) + eps
                eeg_data[:, ch] = (channel_data - median_val) / scale
            else:
                eeg_data[:, ch] = (channel_data - median_val) / iqr
    
    # Clip extreme values to prevent outliers
    eeg_data = np.clip(eeg_data, -5, 5)
    
    # Final check for NaN/Inf
    eeg_data = np.nan_to_num(eeg_data, nan=0.0, posinf=5.0, neginf=-5.0)
    
    return eeg_data.astype(np.float32)

# Process samples
sequence_list = []
labels_list = []
failed = 0

processing_start = time.time()

for idx in tqdm(range(n_samples), desc="Processing EEGs"):
    try:
        sample_id = selected_df.iloc[idx]['eeg_id']
        eeg_path = f"{DATA_DIR}/{sample_id}.parquet"
        
        if os.path.exists(eeg_path):
            eeg_df = pd.read_parquet(eeg_path)
            if not eeg_df.empty and eeg_df.shape[0] > 10:
                eeg_data = eeg_df.values.astype(np.float32)
                sequence = extract_sequence_data(eeg_data)
                sequence_list.append(sequence)
                labels_list.append(selected_df.iloc[idx]['dominant_class'])
            else:
                # Create dummy data for missing samples
                sequence = np.random.normal(0, 0.1, (100, 4)).astype(np.float32)
                sequence_list.append(sequence)
                labels_list.append(selected_df.iloc[idx]['dominant_class'])
                failed += 1
        else:
            # Create dummy data for missing files
            sequence = np.random.normal(0, 0.1, (100, 4)).astype(np.float32)
            sequence_list.append(sequence)
            labels_list.append(selected_df.iloc[idx]['dominant_class'])
            failed += 1
    except Exception as e:
        # Create dummy data on error
        sequence = np.random.normal(0, 0.1, (100, 4)).astype(np.float32)
        sequence_list.append(sequence)
        labels_list.append(selected_df.iloc[idx]['dominant_class'])
        failed += 1

processing_time = time.time() - processing_start
print(f"\nâœ… Processed {len(sequence_list):,}/{n_samples} sequences")
print(f"   Failed: {failed:,}")
print(f"   Time: {processing_time:.1f}s")

# Check data statistics
print("\nğŸ“Š DATA STATISTICS:")
sample_seq = sequence_list[0]
print(f"  Sequence shape: {sample_seq.shape}")
print(f"  Min: {np.min([s.min() for s in sequence_list]):.4f}")
print(f"  Max: {np.max([s.max() for s in sequence_list]):.4f}")
print(f"  Mean: {np.mean([s.mean() for s in sequence_list]):.4f}")
print(f"  Std: {np.mean([s.std() for s in sequence_list]):.4f}")
print(f"  Has NaN: {any(np.any(np.isnan(s)) for s in sequence_list)}")
print(f"  Has Inf: {any(np.any(np.isinf(s)) for s in sequence_list)}")

# 3. DATA PREPARATION
y = np.array(labels_list, dtype=np.int32)

# Check label distribution
unique_labels, label_counts = np.unique(y, return_counts=True)
print("\nğŸ�¯ LABEL DISTRIBUTION:")
for label, count in zip(unique_labels, label_counts):
    print(f"  Class {label}: {count:,} samples ({count/len(y)*100:.1f}%)")

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

# 4. TRANSFORMER MODEL - FIXED VERSION
class EEG_Transformer(nn.Module):
    def __init__(self, input_channels=4, d_model=64, nhead=4, num_layers=3, num_classes=6):
        super().__init__()
        
        # Input projection with layer normalization
        self.input_proj = nn.Linear(input_channels, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=256,  # Increased
            dropout=0.1,  # Reduced dropout
            activation='relu',  # Changed to relu for stability
            batch_first=True,
            norm_first=True  # Important for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-scale pooling
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(d_model * 2, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        # Input projection
        x = self.input_proj(x)
        x = self.ln1(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer
        x = self.transformer(x)
        
        # Multi-scale pooling
        x = x.transpose(1, 2)  # [batch, features, seq_len]
        avg_pooled = self.avg_pool(x).squeeze(-1)
        max_pooled = self.max_pool(x).squeeze(-1)
        pooled = torch.cat([avg_pooled, max_pooled], dim=1)
        
        # Classifier
        logits = self.classifier(pooled)
        
        return logits  # Return raw logits

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# 5. TRAINING SETUP
print(f"\n{'='*60}")
print("âš¡ TRAINING TRANSFORMER")
print(f"{'='*60}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create datasets
class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), self.labels[idx]

def collate_fn(batch):
    sequences = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, torch.tensor(labels, dtype=torch.long)

train_dataset = SequenceDataset(
    [sequence_list[i] for i in train_idx],
    [labels_list[i] for i in train_idx]
)
val_dataset = SequenceDataset(
    [sequence_list[i] for i in val_idx],
    [labels_list[i] for i in val_idx]
)

train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn, num_workers=2
)
val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn, num_workers=2
)

# Initialize model
model = EEG_Transformer(input_channels=4, d_model=64, nhead=4, num_layers=2, num_classes=6)  # Reduced layers
model = model.to(device)

# Use CrossEntropyLoss (which includes softmax)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # Added label smoothing

# AdamW optimizer with weight decay
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=0.001,  # Increased learning rate
    weight_decay=0.01,
    betas=(0.9, 0.999),
    eps=1e-8
)

# Cosine annealing scheduler
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, 
    T_max=15,  # Matches epochs
    eta_min=1e-6
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
print(f"Batch size: 32")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
print(f"Loss function: CrossEntropyLoss with label smoothing")

start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    
    for batch_idx, (sequences, labels) in enumerate(train_bar):
        sequences, labels = sequences.to(device), labels.to(device)
        
        # Forward pass with gradient clipping
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        
        # Check for NaN in loss
        if torch.isnan(loss):
            print(f"âš ï¸�  NaN detected in training loss at batch {batch_idx}")
            loss = torch.tensor(1.0, requires_grad=True).to(device)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Statistics
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
        
        # Update progress bar
        train_bar.set_postfix({
            'loss': loss.item(),
            'acc': train_correct / train_total
        })
    
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
    all_labels = []
    
    with torch.no_grad():
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        for sequences, labels in val_bar:
            sequences, labels = sequences.to(device), labels.to(device)
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            val_bar.set_postfix({
                'loss': loss.item(),
                'acc': val_correct / val_total
            })
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    
    # Update scheduler
    scheduler.step()
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    print(f"\n  Epoch {epoch+1} Summary:")
    print(f"    Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"    Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"    LR: {learning_rates[-1]:.6f}")
    
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        patience_counter = 0
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'train_acc': train_acc,
        }, 'best_transformer_model.pth')
        print(f"    ğŸ’¾ Saved best model with val_acc: {val_acc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"    â�¹ï¸�  Early stopping at epoch {epoch+1}")
            break

total_time = time.time() - start_time

# Load best model
checkpoint = torch.load('best_transformer_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Final evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for sequences, labels in val_loader:
        sequences, labels = sequences.to(device), labels.to(device)
        outputs = model(sequences)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

final_val_acc = accuracy_score(all_labels, all_preds)

print(f"\nâœ… Transformer Training Complete!")
print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
print(f"   Total Training Time: {total_time:.1f}s")
print(f"   Epochs completed: {len(train_losses)}")

# Save final model
torch.save(model.state_dict(), 'transformer_50k_model_final.pth')
print("   Model saved as 'transformer_50k_model_final.pth'")

# 6. CLASSIFICATION REPORT
print(f"\n{'='*60}")
print("ğŸ“Š CLASSIFICATION REPORT")
print(f"{'='*60}")

class_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

print("\nğŸ“‹ Detailed Classification Report:")
report = classification_report(all_labels, all_preds, target_names=class_names, digits=4)
print(report)

# Confusion Matrix
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.title('Transformer - Confusion Matrix (50K Samples)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('transformer_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. TRAINING CURVES
print(f"\n{'='*60}")
print("ğŸ“ˆ TRAINING CURVES")
print(f"{'='*60}")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Loss Curves
epochs_range = range(1, len(train_losses) + 1)

ax1 = axes[0, 0]
ax1.plot(epochs_range, train_losses, 'o-', linewidth=2, markersize=6, label='Training', color='blue')
ax1.plot(epochs_range, val_losses, 's-', linewidth=2, markersize=6, label='Validation', color='red')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('Loss', fontsize=12)
ax1.set_title('Loss Curves', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Accuracy Curves
ax2 = axes[0, 1]
ax2.plot(epochs_range, train_accuracies, 'o-', linewidth=2, markersize=6, label='Training', color='blue')
ax2.plot(epochs_range, val_accuracies, 's-', linewidth=2, markersize=6, label='Validation', color='red')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Learning Rate
ax3 = axes[1, 0]
ax3.plot(epochs_range, learning_rates, 'o-', linewidth=2, markersize=6, color='green')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Learning Rate', fontsize=12)
ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Class Distribution
ax4 = axes[1, 1]
class_counts = [np.sum(all_labels == i) for i in range(6)]
ax4.bar(class_names, class_counts, color='skyblue', edgecolor='black')
ax4.set_xlabel('Class', fontsize=12)
ax4.set_ylabel('Count', fontsize=12)
ax4.set_title('Validation Set Class Distribution', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='y')

for i, count in enumerate(class_counts):
    ax4.text(i, count + max(class_counts)*0.01, str(count), ha='center', fontsize=10)

plt.suptitle('Transformer Training Analysis - 50,000 Samples', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('transformer_training_curves.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. PERFORMANCE SUMMARY
print(f"\n{'='*60}")
print("ğŸ“Š PERFORMANCE SUMMARY")
print(f"{'='*60}")

print(f"\nğŸ�¯ Final Results:")
print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
print(f"   Best Epoch: {checkpoint['epoch'] + 1}")
print(f"   Training Accuracy at Best Epoch: {checkpoint['train_acc']:.4f}")

print(f"\nâ�±ï¸�  Timing:")
print(f"   Data Processing: {processing_time:.1f}s")
print(f"   Model Training: {total_time:.1f}s")
print(f"   Total Time: {processing_time + total_time:.1f}s")
print(f"   Time per Epoch: {total_time/len(train_losses):.1f}s")

print(f"\nğŸ“ˆ Model Insights:")
print(f"   1. Transformer with {sum(p.numel() for p in model.parameters()):,} parameters")
print(f"   2. Input shape: {sample_seq.shape}")
print(f"   3. Using {device} for training")

# Calculate class-wise accuracy
print(f"\nğŸ�¯ Class-wise Performance:")
for i, class_name in enumerate(class_names):
    class_mask = np.array(all_labels) == i
    if np.sum(class_mask) > 0:
        class_acc = np.mean(np.array(all_preds)[class_mask] == np.array(all_labels)[class_mask])
        print(f"   {class_name}: {class_acc:.4f} ({np.sum(class_mask)} samples)")

print(f"\nâœ… Transformer training completed successfully!")

