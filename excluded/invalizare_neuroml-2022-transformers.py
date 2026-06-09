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


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from scipy import signal
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')




# ============================================================================
# ADVANCED FEATURE ENGINEERING
# ============================================================================

def bandpass_filter(data, lowcut, highcut, fs=160, order=4):
    """Apply bandpass filter"""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=0)

def extract_frequency_features(sequence, fs=160):
    """Extract frequency domain features"""
    # Define frequency bands
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    features = []
    
    for band_name, (low, high) in bands.items():
        filtered = bandpass_filter(sequence.copy(), low, high, fs)
        # Band power
        band_power = np.mean(filtered ** 2, axis=0)
        features.append(band_power)
    
    return np.concatenate(features)

def extract_statistical_features(sequence):
    """Extract statistical features"""
    features = []
    
    # Mean, std, min, max per channel
    features.append(np.mean(sequence, axis=0))
    features.append(np.std(sequence, axis=0))
    features.append(np.min(sequence, axis=0))
    features.append(np.max(sequence, axis=0))
    
    # Skewness and kurtosis
    from scipy.stats import skew, kurtosis
    features.append(skew(sequence, axis=0))
    features.append(kurtosis(sequence, axis=0))
    
    return np.concatenate(features)

class AdvancedEEGDataset(Dataset):
    """Enhanced Dataset with feature engineering"""
    def __init__(self, data, labels=None, scaler=None, fit_scaler=False, 
                 augment=False, use_frequency=True):
        self.labels = labels
        self.augment = augment
        self.use_frequency = use_frequency
        
        # Extract channel data
        channel_cols = [col for col in data.columns if col not in ['time', 'condition', 'epoch']]
        
        # Group by epoch
        self.sequences = []
        self.freq_features = []
        self.stat_features = []
        self.epoch_ids = []
        
        for epoch_id, group in data.groupby('epoch'):
            sequence = group[channel_cols].values
            
            # Extract additional features
            if self.use_frequency:
                freq_feat = extract_frequency_features(sequence)
                self.freq_features.append(freq_feat)
            
            stat_feat = extract_statistical_features(sequence)
            self.stat_features.append(stat_feat)
            
            # Normalize sequence
            if fit_scaler:
                if scaler is None:
                    scaler = RobustScaler()  # More robust to outliers
                sequence = scaler.fit_transform(sequence)
            elif scaler is not None:
                sequence = scaler.transform(sequence)
            
            self.sequences.append(sequence)
            self.epoch_ids.append(epoch_id)
        
        self.sequences = np.array(self.sequences, dtype=np.float32)
        self.freq_features = np.array(self.freq_features, dtype=np.float32) if self.use_frequency else None
        self.stat_features = np.array(self.stat_features, dtype=np.float32)
        self.scaler = scaler
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = torch.FloatTensor(self.sequences[idx])
        
        # Data augmentation
        if self.augment and self.labels is not None:
            # Random noise injection
            if np.random.random() > 0.5:
                noise = torch.randn_like(sequence) * 0.05
                sequence = sequence + noise
            
            # Random scaling
            if np.random.random() > 0.5:
                scale = np.random.uniform(0.95, 1.05)
                sequence = sequence * scale
        
        # Additional features
        features = [torch.FloatTensor(self.stat_features[idx])]
        if self.use_frequency:
            features.append(torch.FloatTensor(self.freq_features[idx]))
        
        extra_features = torch.cat(features)
        
        if self.labels is not None:
            label = torch.FloatTensor([self.labels[idx]])
            return sequence, extra_features, label
        return sequence, extra_features




# ============================================================================
# ENHANCED TRANSFORMER MODEL
# ============================================================================

class MultiHeadSelfAttention(nn.Module):
    """Custom multi-head attention with learnable temperature"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature
        self.temperature = nn.Parameter(torch.ones(1) * np.sqrt(self.head_dim))
        
    def forward(self, x):
        B, T, C = x.shape
        
        qkv = self.qkv(x).reshape(B, T, 3, self.nhead, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) / self.temperature
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        out = self.out(out)
        
        return out

class EnhancedEEGTransformer(nn.Module):
    """Enhanced transformer with multiple improvements"""
    def __init__(self, input_dim=19, extra_feat_dim=114, d_model=256, nhead=8, 
                 num_layers=6, dim_feedforward=1024, dropout=0.2):
        super().__init__()
        
        # Input projection with residual
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )
        
        # Positional encoding (learnable)
        self.pos_embedding = nn.Parameter(torch.randn(1, 500, d_model) * 0.02)
        
        # Transformer encoder layers
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': MultiHeadSelfAttention(d_model, nhead, dropout),
                'norm1': nn.LayerNorm(d_model),
                'ffn': nn.Sequential(
                    nn.Linear(d_model, dim_feedforward),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(dim_feedforward, d_model),
                    nn.Dropout(dropout)
                ),
                'norm2': nn.LayerNorm(d_model)
            })
            for _ in range(num_layers)
        ])
        
        # Process extra features
        self.extra_feat_processor = nn.Sequential(
            nn.Linear(extra_feat_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128)
        )
        
        # Classification head with attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model + 128),
            nn.Linear(d_model + 128, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, x, extra_features):
        B, T, _ = x.shape
        
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = x + self.pos_embedding[:, :T, :]
        
        # Transformer layers with residual connections
        for layer in self.layers:
            # Self-attention
            attn_out = layer['attn'](x)
            x = layer['norm1'](x + attn_out)
            
            # Feedforward
            ffn_out = layer['ffn'](x)
            x = layer['norm2'](x + ffn_out)
        
        # Attention pooling
        attn_weights = self.attention_pool(x)
        x = (x * attn_weights).sum(dim=1)
        
        # Process extra features
        extra = self.extra_feat_processor(extra_features)
        
        # Concatenate and classify
        x = torch.cat([x, extra], dim=1)
        x = self.classifier(x)
        
        return x




# ============================================================================
# ADVANCED TRAINING WITH TECHNIQUES
# ============================================================================

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def train_epoch(model, dataloader, criterion, optimizer, device, use_mixup=True):
    """Train with mixup augmentation"""
    model.train()
    total_loss = 0
    
    for sequences, extra_features, labels in dataloader:
        sequences = sequences.to(device)
        extra_features = extra_features.to(device)
        labels = labels.to(device)
        
        # Mixup augmentation
        if use_mixup and np.random.random() > 0.5:
            lam = np.random.beta(0.2, 0.2)
            indices = torch.randperm(sequences.size(0))
            
            sequences = lam * sequences + (1 - lam) * sequences[indices]
            extra_features = lam * extra_features + (1 - lam) * extra_features[indices]
            labels = lam * labels + (1 - lam) * labels[indices]
        
        optimizer.zero_grad()
        
        outputs = model(sequences, extra_features)
        loss = criterion(outputs, labels)
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for sequences, extra_features, labels in dataloader:
            sequences = sequences.to(device)
            extra_features = extra_features.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences, extra_features)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            
            predictions.extend(torch.sigmoid(outputs).cpu().numpy())
            targets.extend(labels.cpu().numpy())
    
    return total_loss / len(dataloader), np.array(predictions), np.array(targets)

def train_model(train_dataset, train_targets, n_folds=7, epochs=100, 
                batch_size=16, lr=0.0005):
    """Train with advanced techniques"""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    fold_models = []
    fold_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(train_dataset.sequences, train_targets)):
        print(f'\nFold {fold + 1}/{n_folds}')
        print('-' * 50)
        
        # Create augmented training dataset
        train_subset_data = torch.utils.data.Subset(train_dataset, train_idx)
        val_subset = torch.utils.data.Subset(train_dataset, val_idx)
        
        train_loader = DataLoader(train_subset_data, batch_size=batch_size, 
                                 shuffle=True, num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, 
                               shuffle=False, num_workers=0, pin_memory=True)
        
        # Initialize model
        model = EnhancedEEGTransformer(
            input_dim=train_dataset.sequences.shape[2],
            extra_feat_dim=train_dataset.stat_features.shape[1] + 
                          (train_dataset.freq_features.shape[1] if train_dataset.use_frequency else 0),
            d_model=256,
            nhead=8,
            num_layers=6,
            dim_feedforward=1024,
            dropout=0.2
        ).to(device)
        
        # Loss and optimizer
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        
        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=2, eta_min=1e-6
        )
        
        # Training loop
        best_val_auc = 0
        patience_counter = 0
        patience = 15
        
        for epoch in range(epochs):
            train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_preds, val_targets = validate(model, val_loader, criterion, device)
            
            scheduler.step()
            
            # Calculate AUC
            from sklearn.metrics import roc_auc_score
            val_auc = roc_auc_score(val_targets, val_preds)
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f} - '
                      f'Val Loss: {val_loss:.4f} - Val AUC: {val_auc:.4f}')
            
            # Save best model based on AUC
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f'Early stopping at epoch {epoch+1}')
                    break
        
        # Load best model
        model.load_state_dict(best_model_state)
        fold_models.append(model)
        fold_scores.append(best_val_auc)
        
        print(f'Fold {fold + 1} - Best Val AUC: {best_val_auc:.4f}')
    
    print(f'\nMean CV AUC: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})')
    
    return fold_models




# ============================================================================
# PREDICTION WITH TTA
# ============================================================================

def predict_with_tta(models, test_dataset, batch_size=16, n_tta=5):
    """Predict with test-time augmentation"""
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    all_predictions = []
    
    for model in models:
        model.eval()
        
        # Multiple TTA passes
        tta_preds = []
        for _ in range(n_tta):
            predictions = []
            
            with torch.no_grad():
                for sequences, extra_features in test_loader:
                    sequences = sequences.to(device)
                    extra_features = extra_features.to(device)
                    
                    # Add small noise for TTA
                    if _ > 0:
                        sequences = sequences + torch.randn_like(sequences) * 0.02
                    
                    outputs = model(sequences, extra_features)
                    preds = torch.sigmoid(outputs).cpu().numpy()
                    predictions.extend(preds)
            
            tta_preds.append(predictions)
        
        # Average TTA predictions
        model_pred = np.mean(tta_preds, axis=0)
        all_predictions.append(model_pred)
    
    # Average predictions from all folds
    final_predictions = np.mean(all_predictions, axis=0)
    
    return final_predictions



train_df = pd.read_csv('/kaggle/input/neuroml-2022-eeg-bci-prediction/train.csv/train.csv')
test_df = pd.read_csv('/kaggle/input/neuroml-2022-eeg-bci-prediction/test.csv/test.csv')

# Prepare datasets with augmentation
print('Preparing enhanced datasets...')
train_targets = train_df.groupby('epoch')['condition'].first().values
train_targets = (train_targets != 1).astype(int)

train_dataset = AdvancedEEGDataset(train_df, train_targets, fit_scaler=True, 
                                   augment=True, use_frequency=True)
test_dataset = AdvancedEEGDataset(test_df, scaler=train_dataset.scaler, 
                                  fit_scaler=False, augment=False, use_frequency=True)

print(f'Train samples: {len(train_dataset)}')
print(f'Test samples: {len(test_dataset)}')
print(f'Sequence shape: {train_dataset.sequences.shape}')
print(f'Extra features shape: {train_dataset.stat_features.shape}')

# Train models
print('\nTraining enhanced models...')
models = train_model(
    train_dataset, 
    train_targets, 
    n_folds=7,
    epochs=100,
    batch_size=16,
    lr=0.0005
)

# Make predictions with TTA
print('\nMaking predictions with TTA...')
predictions = predict_with_tta(models, test_dataset, n_tta=5)

# Create submission
submission = pd.DataFrame({
    'id': test_dataset.epoch_ids,
    'predicted': predictions.flatten()
})

submission.to_csv('submission.csv', index=False)
print('\nSubmission file created: submission.csv')
print(f'Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]')
print('\nFirst few predictions:')
print(submission.head(10))

