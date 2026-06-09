!pip install torch-geometric
!pip install numpy


# ==================== GCN ON 50K SAMPLES - FIXED VERSION ====================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, DataLoader
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
print("ğŸ§  GCN TRAINING ON 50,000 SAMPLES - FIXED VERSION")
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

# 2. IMPROVED GRAPH DATA CREATION
print("\nğŸ”§ CREATING GRAPH DATA...")

DATA_DIR = '/kaggle/input/hms-harmful-brain-activity-classification/train_eegs'

def robust_normalize(data):
    """Robust normalization that handles outliers"""
    # Clip extreme values first
    data = np.clip(data, np.percentile(data, 1), np.percentile(data, 99))
    
    # Z-score normalization with epsilon
    mean = np.mean(data)
    std = np.std(data)
    if std < 1e-8:
        std = 1.0
    
    normalized = (data - mean) / std
    
    # Ensure no extreme values remain
    normalized = np.clip(normalized, -5, 5)
    return normalized

def create_graph_data(eeg_data, label):
    """Create graph data from EEG with robust preprocessing"""
    if eeg_data.size == 0 or eeg_data.shape[0] < 10:
        return None
    
    try:
        # Downsample to 100 timesteps (slightly more for better features)
        if eeg_data.shape[0] > 100:
            indices = np.linspace(0, eeg_data.shape[0]-1, 100).astype(int)
            eeg_data = eeg_data[indices, :]
        
        # Use first 8 channels (better representation)
        num_channels = min(eeg_data.shape[1], 8)
        if eeg_data.shape[1] > num_channels:
            eeg_data = eeg_data[:, :num_channels]
        elif eeg_data.shape[1] < num_channels:
            # Pad with zeros if needed
            padding = num_channels - eeg_data.shape[1]
            eeg_data = np.pad(eeg_data, ((0, 0), (0, padding)), mode='constant')
        
        # Robust normalization per channel
        for ch in range(num_channels):
            eeg_data[:, ch] = robust_normalize(eeg_data[:, ch])
        
        # Extract richer node features (10 features per channel)
        node_features = []
        for ch in range(num_channels):
            channel_data = eeg_data[:, ch]
            
            # Basic statistical features
            mean_val = float(np.mean(channel_data))
            std_val = float(np.std(channel_data))
            min_val = float(np.min(channel_data))
            max_val = float(np.max(channel_data))
            median_val = float(np.median(channel_data))
            
            # Additional features
            mad = float(np.mean(np.abs(channel_data - mean_val)))  # Mean absolute deviation
            energy = float(np.sum(channel_data ** 2))  # Signal energy
            skewness = float(pd.Series(channel_data).skew())  # Skewness
            kurt = float(pd.Series(channel_data).kurt())  # Kurtosis
            pos_ratio = float(np.sum(channel_data > 0) / len(channel_data))  # Positive ratio
            
            node_features.append([mean_val, std_val, min_val, max_val, median_val,
                                 mad, energy, skewness, kurt, pos_ratio])
        
        # Create edges (fully connected for simplicity)
        num_nodes = len(node_features)
        edges = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i != j:
                    edges.append([i, j])
        
        if edges:
            edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
            
            # Add self-loops
            self_loops = torch.tensor([[i, i] for i in range(num_nodes)], dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edges, self_loops], dim=1)
            
            graph_data = Data(
                x=torch.tensor(node_features, dtype=torch.float32),
                edge_index=edge_index,
                y=torch.tensor([label], dtype=torch.long)
            )
            return graph_data
        
        return None
    except Exception as e:
        #print(f"Error creating graph: {e}")
        return None

# Process samples - increased to 50K
print(f"Processing {n_samples:,} samples...")
graph_data_list = []
labels_list = []
failed = 0

processing_start = time.time()

# Process in chunks to manage memory
chunk_size = 5000
num_chunks = (n_samples + chunk_size - 1) // chunk_size

for chunk_idx in range(num_chunks):
    start_idx = chunk_idx * chunk_size
    end_idx = min((chunk_idx + 1) * chunk_size, n_samples)
    
    print(f"\nProcessing chunk {chunk_idx + 1}/{num_chunks} (samples {start_idx}-{end_idx})...")
    
    for idx in tqdm(range(start_idx, end_idx), desc=f"Chunk {chunk_idx + 1}"):
        try:
            sample_id = selected_df.iloc[idx]['eeg_id']
            eeg_path = f"{DATA_DIR}/{sample_id}.parquet"
            
            if os.path.exists(eeg_path):
                eeg_df = pd.read_parquet(eeg_path)
                if not eeg_df.empty and eeg_df.shape[0] > 10:
                    # Select only numeric columns and handle NaNs
                    eeg_df = eeg_df.select_dtypes(include=[np.number])
                    eeg_df = eeg_df.fillna(method='ffill').fillna(0)
                    
                    eeg_data = eeg_df.values.astype(np.float32)
                    label = selected_df.iloc[idx]['dominant_class']
                    graph_data = create_graph_data(eeg_data, label)
                    
                    if graph_data is not None:
                        graph_data_list.append(graph_data)
                        labels_list.append(label)
                    else:
                        failed += 1
                else:
                    failed += 1
            else:
                failed += 1
        except Exception as e:
            #print(f"Error processing sample {idx}: {e}")
            failed += 1
    
    print(f"  Chunk completed: {len(graph_data_list):,} graphs so far, {failed:,} failed")

processing_time = time.time() - processing_start
print(f"\nâœ… Created {len(graph_data_list):,} graphs from {n_samples:,} samples")
print(f"   Success rate: {len(graph_data_list)/n_samples*100:.1f}%")
print(f"   Failed: {failed:,}")
print(f"   Time: {processing_time:.1f}s")

# Check if we have enough data
if len(graph_data_list) < 1000:
    print("â�Œ Not enough valid graphs created. Exiting...")
    exit()

# 3. DATA PREPARATION
# Split data
train_idx, val_idx = train_test_split(
    range(len(graph_data_list)),
    test_size=0.15,
    random_state=42,
    stratify=labels_list
)

train_dataset = [graph_data_list[i] for i in train_idx]
val_dataset = [graph_data_list[i] for i in val_idx]

print(f"\nğŸ“Š DATASET SIZES:")
print(f"   Training graphs: {len(train_dataset):,}")
print(f"   Validation graphs: {len(val_dataset):,}")
print(f"   Node features: {graph_data_list[0].x.shape[1]}")
print(f"   Nodes per graph: {graph_data_list[0].x.shape[0]}")

# 4. IMPROVED GCN MODEL
class EEG_GCN(nn.Module):
    def __init__(self, num_node_features=10, hidden_dim=128, num_classes=6):
        super().__init__()
        
        # GCN layers with residual connections
        self.conv1 = GCNConv(num_node_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classifier with gradient clipping friendly layers
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )
        
        # Initialize weights properly
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, GCNConv):
                nn.init.xavier_uniform_(m.lin.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Layer 1 with residual
        x1 = F.relu(self.bn1(self.conv1(x, edge_index)))
        x1 = F.dropout(x1, p=0.3, training=self.training)
        
        # Layer 2 with residual
        x2 = F.relu(self.bn2(self.conv2(x1, edge_index)))
        x2 = F.dropout(x2, p=0.3, training=self.training)
        
        # Layer 3
        x3 = F.relu(self.bn3(self.conv3(x2, edge_index)))
        
        # Skip connection
        x = x1 + x2 + x3
        
        # Graph-level pooling
        graph_embedding = global_mean_pool(x, batch)
        
        # Attention weighting
        attention_weights = torch.softmax(self.attention(graph_embedding), dim=0)
        attended = graph_embedding * attention_weights
        
        # Classification - use logits instead of log_softmax
        logits = self.classifier(attended)
        
        return logits  # Return raw logits

# 5. TRAINING WITH IMPROVED SETTINGS
print(f"\n{'='*60}")
print("ğŸ§  TRAINING GCN WITH IMPROVED SETTINGS")
print(f"{'='*60}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Create data loaders with appropriate batch size
batch_size = 32 if len(train_dataset) > 10000 else 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)

# Initialize model
model = EEG_GCN(num_node_features=10, hidden_dim=128, num_classes=6)
model = model.to(device)

# Calculate class weights for balanced loss
class_counts = torch.bincount(torch.tensor(labels_list))
class_weights = (1.0 / class_counts.float()) * len(class_counts) / 2.0
class_weights = class_weights / class_weights.sum()
print(f"Class weights: {class_weights.tolist()}")

# Use CrossEntropyLoss instead of NLLLoss (combines log_softmax + NLL)
criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

# Gradient clipping and weight decay
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=1e-5)

# Track metrics
train_losses = []
val_losses = []
train_accuracies = []
val_accuracies = []
learning_rates = []

num_epochs = 30
best_val_acc = 0
patience_counter = 0
max_patience = 7

print(f"\nTraining for {num_epochs} epochs...")
print(f"Training set: {len(train_dataset):,} graphs")
print(f"Validation set: {len(val_dataset):,} graphs")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {optimizer.param_groups[0]['lr']}")

start_time = time.time()

for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
    
    for batch in train_loader_tqdm:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        outputs = model(batch)
        loss = criterion(outputs, batch.y)
        
        # Gradient clipping to prevent NaN
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        train_total += batch.y.size(0)
        train_correct += (predicted == batch.y).sum().item()
        
        # Update progress bar
        train_loader_tqdm.set_postfix({
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
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        
        for batch in val_loader_tqdm:
            batch = batch.to(device)
            outputs = model(batch)
            loss = criterion(outputs, batch.y)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += batch.y.size(0)
            val_correct += (predicted == batch.y).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(batch.y.cpu().numpy())
            
            val_loader_tqdm.set_postfix({
                'loss': loss.item(),
                'acc': val_correct / val_total
            })
    
    avg_val_loss = val_loss / len(val_loader)
    val_acc = val_correct / val_total
    val_losses.append(avg_val_loss)
    val_accuracies.append(val_acc)
    learning_rates.append(optimizer.param_groups[0]['lr'])
    
    # Update scheduler
    scheduler.step()
    
    print(f"\n  Epoch {epoch+1}:")
    print(f"    Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"    Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"    LR: {optimizer.param_groups[0]['lr']:.6f}")
    
    # Check for NaN
    if np.isnan(avg_train_loss) or np.isnan(avg_val_loss):
        print(f"  âš ï¸�  NaN detected in loss! Adjusting learning rate...")
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.1
        continue
    
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
        }, 'best_gcn_model.pth')
        print(f"    ğŸ’¾ Saved best model with val_acc: {val_acc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= max_patience:
            print(f"  â�¹ï¸�  Early stopping at epoch {epoch+1}")
            break
    
    print(f"    Best val_acc so far: {best_val_acc:.4f}")
    print(f"    Patience counter: {patience_counter}/{max_patience}")

total_time = time.time() - start_time

# Load best model
checkpoint = torch.load('best_gcn_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
best_epoch = checkpoint['epoch']

# Final evaluation
model.eval()
all_preds = []
all_labels = []
val_probs = []

with torch.no_grad():
    for batch in val_loader:
        batch = batch.to(device)
        outputs = model(batch)
        probs = F.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(batch.y.cpu().numpy())
        val_probs.extend(probs.cpu().numpy())

final_val_acc = accuracy_score(all_labels, all_preds)

print(f"\nâœ… GCN Training Complete!")
print(f"   Best Validation Accuracy: {best_val_acc:.4f} (epoch {best_epoch + 1})")
print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
print(f"   Total Training Time: {total_time:.1f}s ({total_time/60:.1f} minutes)")

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'node_features': graph_data_list[0].x.shape[1],
    'hidden_dim': 128,
    'num_classes': 6,
    'val_acc': final_val_acc
}, 'gcn_50k_model_fixed.pth')
print("   Model saved as 'gcn_50k_model_fixed.pth'")

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
plt.title(f'GCN - Confusion Matrix ({len(graph_data_list):,} Samples)', fontsize=14, fontweight='bold')
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.tight_layout()
plt.savefig('gcn_confusion_matrix_fixed.png', dpi=300, bbox_inches='tight')
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
ax2.axhline(y=1/6, color='gray', linestyle='--', alpha=0.5, label='Random Chance')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Accuracy Curves', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

# Learning Rate Schedule
ax3 = axes[1, 0]
ax3.plot(epochs_range, learning_rates, 'o-', linewidth=2, markersize=6, color='green')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Learning Rate', fontsize=12)
ax3.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.set_yscale('log')

# Convergence Analysis
ax4 = axes[1, 1]
# Calculate convergence metric (train_acc - val_acc gap)
convergence_gap = [t - v for t, v in zip(train_accuracies, val_accuracies)]
ax4.plot(epochs_range, convergence_gap, 'o-', linewidth=2, markersize=6, color='purple')
ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Train-Val Accuracy Gap', fontsize=12)
ax4.set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

# Add optimal point marker
if convergence_gap:
    best_epoch_gap = np.argmin(np.abs(convergence_gap))
    ax4.scatter(best_epoch_gap + 1, convergence_gap[best_epoch_gap], color='red', s=200, zorder=5)
    ax4.annotate(f'Best Balance\nGap: {convergence_gap[best_epoch_gap]:.4f}', 
                 (best_epoch_gap + 1, convergence_gap[best_epoch_gap]),
                 xytext=(10, 10), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.5))

plt.suptitle(f'GCN Training Analysis - {len(graph_data_list):,} Valid Samples', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('gcn_loss_curves_fixed.png', dpi=300, bbox_inches='tight')
plt.show()

# 8. PERFORMANCE SUMMARY
print(f"\n{'='*60}")
print("ğŸ“Š PERFORMANCE SUMMARY")
print(f"{'='*60}")

print(f"\nğŸ�¯ Training Statistics:")
print(f"   Total Samples Processed: {n_samples:,}")
print(f"   Valid Graphs Created: {len(graph_data_list):,}")
print(f"   Success Rate: {len(graph_data_list)/n_samples*100:.1f}%")
print(f"   Best Epoch: {best_epoch + 1}")
print(f"   Best Validation Accuracy: {best_val_acc:.4f}")
print(f"   Final Validation Accuracy: {final_val_acc:.4f}")
print(f"   Random Chance Baseline: {1/6:.4f}")
print(f"   Improvement over Random: {final_val_acc - 1/6:.4f}")

if convergence_gap:
    print(f"   Final Overfitting Gap: {convergence_gap[-1]:.4f}")
    print(f"   Minimum Overfitting Gap: {min(convergence_gap):.4f}")

print(f"\nâ�±ï¸�  Timing Statistics:")
print(f"   Graph Creation: {processing_time:.1f}s ({processing_time/60:.1f} minutes)")
print(f"   Model Training: {total_time:.1f}s ({total_time/60:.1f} minutes)")
print(f"   Total Time: {processing_time + total_time:.1f}s ({ (processing_time + total_time)/60:.1f} minutes)")
print(f"   Epochs Completed: {len(train_losses)}")
print(f"   Time per Epoch: {total_time/len(train_losses):.1f}s")
print(f"   Graphs per Second: {len(graph_data_list)/processing_time:.1f}")

print(f"\nğŸ“ˆ Learning Insights:")
print(f"   1. Model converged at epoch {best_epoch + 1}")
print(f"   2. Best validation accuracy: {best_val_acc:.4f}")
print(f"   3. Improvement over random chance: {(final_val_acc - 1/6)*100:.1f}%")
print(f"   4. Final learning rate: {learning_rates[-1]:.6f}")

print(f"\nğŸ”§ Model Architecture:")
print(f"   Node features: {graph_data_list[0].x.shape[1]}")
print(f"   Hidden dimension: 128")
print(f"   Number of GCN layers: 3")
print(f"   Classifier layers: 3")
print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")

print(f"\nâœ… GCN training on {len(graph_data_list):,} valid samples completed successfully!")

