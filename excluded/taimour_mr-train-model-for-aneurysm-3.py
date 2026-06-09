# Let's import the tools we need (like bringing tools to a workshop)
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import random
from scipy.ndimage import rotate, zoom, gaussian_filter

# Check if we can use GPU (makes training faster)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Reproducibility (so results are consistent)
torch.manual_seed(42)


# Path to our preprocessed data (from the previous notebook)
PREPROCESSED_DIR = "/kaggle/input/preprocess-files-for-aneurysm-3/preprocessed_data"
TRAIN_CSV_PATH = "/kaggle/input/rsna-intracranial-aneurysm-detection/train.csv"

# Load metadata to see what we have
with open(os.path.join(PREPROCESSED_DIR, 'metadata.json'), 'r') as f:
    metadata = json.load(f)

print(f"Found {metadata['series_count']} successfully processed brain scans")
print(f"Failed to process: {metadata['failed_series_count']} scans")

# Load the training labels (which scans have aneurysms)
train_df = pd.read_csv(TRAIN_CSV_PATH)

# Keep only the series we successfully processed
processed_ids = metadata['series_ids']
train_df = train_df[train_df['SeriesInstanceUID'].isin(processed_ids)].copy()

# Filter out only CT
processed_ids = [sid for sid, mod in metadata['modalities'].items() if mod == "MR"]
train_df = train_df[train_df['SeriesInstanceUID'].isin(processed_ids)].copy()
print(f"Using {len(train_df)} scans for training")

# Define our target columns (the aneurysm locations)
LABEL_COLS = [
    'Left Infraclinoid Internal Carotid Artery',
    'Right Infraclinoid Internal Carotid Artery',
    'Left Supraclinoid Internal Carotid Artery',
    'Right Supraclinoid Internal Carotid Artery',
    'Left Middle Cerebral Artery',
    'Right Middle Cerebral Artery',
    'Anterior Communicating Artery',
    'Left Anterior Cerebral Artery',
    'Right Anterior Cerebral Artery',
    'Left Posterior Communicating Artery',
    'Right Posterior Communicating Artery',
    'Basilar Tip',
    'Other Posterior Circulation',
    'Aneurysm Present',  # This is the main label we care about most
]


# Create a simple dataset class (like a photo album for our brain scans)
class BrainScanDataset(Dataset):
    def __init__(self, series_ids, labels_df, data_dir):
        self.series_ids = series_ids
        self.labels_df = labels_df.set_index('SeriesInstanceUID')
        self.data_dir = data_dir
    
    def __len__(self):
        return len(self.series_ids)
    
    def __getitem__(self, idx):
        series_id = self.series_ids[idx]
        
        # Load the preprocessed brain scan
        volume = np.load(os.path.join(self.data_dir, f"{series_id}.npy"))
        
        # Get the labels (which locations have aneurysms)
        labels = self.labels_df.loc[series_id][LABEL_COLS].values.astype(np.float32)
        
        # Convert to PyTorch tensors (the format our model needs)
        volume_tensor = torch.FloatTensor(volume).unsqueeze(0)  # Add channel dimension
        labels_tensor = torch.FloatTensor(labels)
        
        return volume_tensor, labels_tensor

# Split data into training and validation sets (80% for learning, 20% for testing)
train_ids, val_ids = train_test_split(
    processed_ids, 
    test_size=0.2, 
    random_state=42,
    stratify=train_df['Aneurysm Present'].values  # Keep same proportion of positive cases
)

# Create datasets
train_dataset = BrainScanDataset(train_ids, train_df, PREPROCESSED_DIR)
val_dataset = BrainScanDataset(val_ids, train_df, PREPROCESSED_DIR)

# Create data loaders (like a conveyor belt bringing data to our model)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2)

print(f"Training set: {len(train_dataset)} scans")
print(f"Validation set: {len(val_dataset)} scans")


class ResidualBlock3D(nn.Module):
    """Properly handles channel dimension changes in residual connections"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock3D, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        # Projection shortcut if dimensions change
        self.projection = None
        if stride != 1 or in_channels != out_channels:
            self.projection = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.projection is not None:
            identity = self.projection(x)
            
        out += identity
        out = self.relu(out)
        
        return out

class Improved3DModel(nn.Module):
    """Fixed architecture with proper residual connections"""
    def __init__(self, num_labels=14):
        super(Improved3DModel, self).__init__()
        
        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True)
        )
        
        # Residual blocks with proper downsampling
        self.res1 = ResidualBlock3D(32, 32, stride=1)
        self.pool1 = nn.MaxPool3d(2)
        
        self.res2 = ResidualBlock3D(32, 64, stride=1)
        self.pool2 = nn.MaxPool3d(2)
        
        self.res3 = ResidualBlock3D(64, 128, stride=1)
        self.pool3 = nn.MaxPool3d(2)
        
        self.res4 = ResidualBlock3D(128, 256, stride=1)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.dropout2 = nn.Dropout(0.3)
        
        # Separate heads for main label and locations
        self.main_head = nn.Linear(128, 1)
        self.location_head = nn.Linear(128, 13)
    
    def forward(self, x):
        x = self.initial(x)
        x = self.res1(x)
        x = self.pool1(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.res3(x)
        x = self.pool3(x)
        x = self.res4(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        
        main_pred = self.main_head(x)
        location_preds = self.location_head(x)
        
        return torch.cat([location_preds, main_pred], dim=1)

# Create and move improved model to GPU
model = Improved3DModel(num_labels=len(LABEL_COLS)).to(device)

# Show how many parameters our model has (like counting brain cells)
total_params = sum(p.numel() for p in model.parameters())
print(f"Model created with {total_params:,} parameters")


# # Set up training parameters
learning_rate = 0.001
num_epochs = 50  # We'll keep it short for this beginner example

# Competition-weighted loss function
# Aneurysm Present gets 13x more weight than other locations
weights = torch.ones(len(LABEL_COLS)).to(device)
weights[-1] = 13.0  # Last label is "Aneurysm Present"
criterion = nn.BCEWithLogitsLoss(pos_weight=weights)

# Optimizer with weight decay for regularization
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=3, verbose=True
)

# Training with early stopping
best_score = 0.0
patience = 5
no_improve = 0
train_losses = []
val_scores = []

print("\nStarting training with improved model...")
for epoch in range(num_epochs):
    # Training phase
    model.train()
    running_loss = 0.0
    
    train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
    for volumes, labels in train_bar:
        volumes = volumes.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(volumes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_bar.set_postfix(loss=f"{running_loss/len(train_bar):.4f}")
    
    # Validation phase
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for volumes, labels in val_loader:
            volumes = volumes.to(device)
            labels = labels.to(device)
            
            outputs = model(volumes)
            probs = torch.sigmoid(outputs)
            
            all_preds.append(probs.cpu())
            all_labels.append(labels.cpu())
    
    # Calculate competition metric
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    
    # Calculate weighted AUC
    ap_auc = roc_auc_score(labels[:, -1], preds[:, -1])  # Aneurysm Present
    other_auc = np.mean([roc_auc_score(labels[:, i], preds[:, i]) for i in range(13)])
    competition_score = 0.5 * (ap_auc + other_auc)
    
    # Track metrics
    avg_train_loss = running_loss / len(train_loader)
    train_losses.append(avg_train_loss)
    val_scores.append(competition_score)
    
    print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | "
          f"Competition Score: {competition_score:.4f} | "
          f"Main AUC: {ap_auc:.4f}, Other AUC: {other_auc:.4f}")
    
    # Learning rate adjustment
    scheduler.step(competition_score)
    
    # Early stopping
    if competition_score > best_score:
        best_score = competition_score
        torch.save(model.state_dict(), "/kaggle/working/best_model.pth")
        print(f"\n Epoch for best model = {epoch+1}")
        no_improve = 0
    else:
        no_improve += 1
        if no_improve >= patience:
            print(f"\nEarly stopping triggered after epoch {epoch+1}")
            break

print(f"\nTraining complete! Best competition score: {best_score:.4f}")


# Let's see how well our model did
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for volumes, labels in val_loader:
        volumes = volumes.to(device)
        
        # Get predictions
        outputs = model(volumes)
        probs = torch.sigmoid(outputs)  # Convert to probabilities (0 to 1)
        
        all_preds.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

# Combine all predictions and labels
preds = np.vstack(all_preds)
labels = np.vstack(all_labels)

# Calculate AUC for each label (how well we can tell positive from negative)
auc_scores = []
for i, col in enumerate(LABEL_COLS):
    try:
        auc = roc_auc_score(labels[:, i], preds[:, i])
        auc_scores.append(auc)
        print(f"{col[:30]:<30} AUC: {auc:.4f}")
    except:
        print(f"{col[:30]:<30} AUC: Not enough samples")

# Calculate the competition metric (special weighted average)
ap_auc = auc_scores[-1]  # AUC for "Aneurysm Present" (last column)
other_auc = np.mean(auc_scores[:-1])  # Average of other 13 locations
competition_score = 0.5 * (ap_auc + other_auc)

print(f"\nCompetition Metric Score: {competition_score:.4f}")
print(f"(Main label AUC: {ap_auc:.4f}, Average other locations: {other_auc:.4f})")


# Plot training progress
plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_scores, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.legend()
plt.grid(True)
plt.savefig('/kaggle/working/training_curve.png')
plt.show()

# Show some example predictions
sample_idx = 0
print("\nExample Prediction:")
for i, col in enumerate(LABEL_COLS):
    print(f"{col}: Actual={labels[sample_idx, i]:.0f}, Predicted={preds[sample_idx, i]:.4f}")


# Save the final model
torch.save(model.state_dict(), "/kaggle/working/final_model.pth")
print("Final model saved to /kaggle/working/final_model.pth")

# Save evaluation results
results = {
    "competition_score": float(competition_score),
    "main_label_auc": float(ap_auc),
    "average_other_auc": float(other_auc),
    "label_aucs": {col: float(auc) for col, auc in zip(LABEL_COLS, auc_scores)}
}

with open('/kaggle/working/results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Training results saved to /kaggle/working/results.json")


# ===========================================================
# FINAL TRAINING ON FULL DATASET (for submission model)
# ===========================================================
print("\n" + "="*50)
print("RETRAINING ON FULL DATASET FOR FINAL SUBMISSION")
print("="*50)

# Create dataset with ALL available data
full_dataset = BrainScanDataset(processed_ids, train_df, PREPROCESSED_DIR)
full_loader = DataLoader(full_dataset, batch_size=4, shuffle=True, num_workers=2)

print(f"Training final model on {len(full_dataset)} total samples (no validation split)")

# Recreate the model
final_model = Improved3DModel(num_labels=len(LABEL_COLS)).to(device)

# WARM-START FROM OUR BEST MODEL (critical improvement)
try:
    # Load the best model we found during development
    final_model.load_state_dict(torch.load("/kaggle/working/best_model.pth", map_location=device))
    print("âœ… Warm-starting from best development model")
except Exception as e:
    print(f"âš ï¸�  Could not load best model: {str(e)}")
    print("Starting from random initialization instead")

# Use a smaller learning rate for fine-tuning
optimizer = optim.AdamW(final_model.parameters(), lr=learning_rate/10, weight_decay=1e-4)
final_model.train()

# Train for fewer epochs (just fine-tuning)
full_epochs = 5
for epoch in range(full_epochs):
    running_loss = 0.0
    
    train_bar = tqdm(full_loader, desc=f"Full Training Epoch {epoch+1}/{full_epochs}")
    for volumes, labels in train_bar:
        volumes = volumes.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = final_model(volumes)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        train_bar.set_postfix(loss=f"{running_loss/len(train_bar):.4f}")
    
    avg_loss = running_loss / len(full_loader)
    print(f"Full Training Epoch {epoch+1}/{full_epochs} | Loss: {avg_loss:.4f}")

# Save the FINAL model trained on all data
torch.save(final_model.state_dict(), "/kaggle/working/final_full_model.pth")
print("\nğŸ�‰ FINAL MODEL TRAINED ON FULL DATASET SAVED!")
print("Use this model for submissions: /kaggle/working/final_full_model.pth")

