import os
import random
import math
import time
import warnings
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
import timm
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

# ================================
# Configuration & Hyperparameters
# ================================

@dataclass
class Config:
    # Data paths
    kaggle_root: str = "/kaggle/input/cmi-detect-behavior-with-sensor-data"
    train_csv: str = "train.csv"
    test_csv: str = "test.csv"
    test_demo: str = "test_demographics.csv"
    
    # Model configuration
    model_name: str = "resnet1d"
    pretrained: bool = False
    
    # Training hyperparameters
    seed: int = 42
    batch_size: int = 64
    learning_rate: float = 2e-3
    max_epochs: int = 100
    patience: int = 15
    val_split: float = 0.2
    label_smoothing: float = 0.05
    weight_decay: float = 1e-3
    
    # Model specific hyperparameters
    hidden_dim: int = 128  # Reduced for smaller models
    num_layers: int = 2    # Reduced for smaller models
    dropout: float = 0.4
    
    # Data processing
    sequence_percentile: int = 85
    use_augmentation: bool = True
    augmentation_prob: float = 0.2
    
    # Cross-validation
    use_cv: bool = False
    cv_folds: int = 5
    
    # Hardware - Fixed for Kaggle
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 0  # Set to 0 to avoid multiprocessing issues in Kaggle
    pin_memory: bool = False  # Disabled for Kaggle

def set_random_seed(seed: int):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_environment(config: Config):
    """Setup environment and configuration"""
    set_random_seed(config.seed)
    return config

# ================================
# Data Loading & Preprocessing
# ================================

class GestureDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = False, augment_prob: float = 0.3):
        self.X = X
        self.y = y
        self.augment = augment
        self.augment_prob = augment_prob
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx: int):
        x = torch.from_numpy(self.X[idx]).float()
        y = torch.tensor(self.y[idx], dtype=torch.long)
        
        if self.augment and random.random() < self.augment_prob:
            x = self._apply_augmentation(x)
            
        x = x.transpose(0, 1)  # (n_features, seq_len) for Conv1d
        return x, y
    
    def _apply_augmentation(self, x: torch.Tensor) -> torch.Tensor:
        """Apply data augmentation techniques"""
        # Gaussian noise
        if random.random() < 0.4:
            noise = torch.randn_like(x) * 0.005
            x = x + noise
            
        # Time shift
        if random.random() < 0.3:
            shift = random.randint(-3, 3)
            x = torch.roll(x, shift, dims=0)
            
        # Scaling
        if random.random() < 0.3:
            scale = random.uniform(0.95, 1.05)
            x = x * scale
            
        # Mixup on feature level
        if random.random() < 0.2:
            alpha = 0.1
            lam = np.random.beta(alpha, alpha)
            noise_features = torch.randn_like(x) * 0.01
            x = lam * x + (1 - lam) * noise_features
            
        return x

def clip_outliers(data: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Clip outliers using quantile-based method"""
    for col in columns:
        q1, q99 = data[col].quantile([0.01, 0.99])
        data[col] = data[col].clip(q1, q99)
    return data

def handle_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """Handle missing values with robust interpolation"""
    return data.interpolate(method='linear').ffill().bfill().fillna(0)

def normalize_features(data: pd.DataFrame) -> np.ndarray:
    """Normalize features using StandardScaler"""
    scaler = StandardScaler()
    return scaler.fit_transform(data).astype(np.float32)

def preprocess_single_sequence(seq_df: pd.DataFrame, imu_cols: List[str]) -> np.ndarray:
    """Preprocess a single sequence"""
    data = seq_df[imu_cols].copy()
    data = handle_missing_values(data)
    data = clip_outliers(data, imu_cols)
    return normalize_features(data)

def pad_or_truncate_sequence(arr: np.ndarray, maxlen: int, dtype=np.float32) -> np.ndarray:
    """Pad or truncate array to fixed length"""
    t, c = arr.shape
    if t >= maxlen:
        return arr[:maxlen].astype(dtype, copy=False)
    out = np.zeros((maxlen, c), dtype=dtype)
    out[:t] = arr
    return out

def encode_labels(df: pd.DataFrame) -> Tuple[pd.DataFrame, LabelEncoder]:
    """Encode gesture labels"""
    le = LabelEncoder()
    df["gesture"] = le.fit_transform(df["gesture"].astype(str))
    np.save("gesture_classes.npy", le.classes_)
    return df, le

def calculate_padding_length(lengths: List[int], percentile: int) -> int:
    """Calculate optimal padding length based on sequence lengths"""
    pad_len = int(np.percentile(lengths, percentile))
    print(f"Padding/truncating to length: {pad_len}")
    print(f"Length statistics: min={np.min(lengths)}, max={np.max(lengths)}, mean={np.mean(lengths):.1f}")
    np.save("sequence_maxlen.npy", pad_len)
    return pad_len

def process_sequences(seq_groups, imu_cols: List[str]) -> Tuple[List[np.ndarray], List[int], List[int]]:
    """Process all sequences"""
    X_list, y_list, lengths = [], [], []
    
    for seq_id, seq_df in tqdm(seq_groups, desc="Processing sequences"):
        X_seq = preprocess_single_sequence(seq_df, imu_cols)
        X_list.append(X_seq)
        lengths.append(X_seq.shape[0])
        y_list.append(seq_df["gesture"].iloc[0])
    
    return X_list, y_list, lengths

def load_and_process_data(config: Config):
    """Load and preprocess the training data"""
    print("Loading and processing data...")
    
    # Load data
    train_path = os.path.join(config.kaggle_root, config.train_csv)
    df = pd.read_csv(train_path)
    print(f"Loaded {len(df):,} rows.")
    
    # Encode labels
    df, le = encode_labels(df)
    
    # Define sensor columns
    imu_cols = ["acc_x", "acc_y", "acc_z", "rot_w", "rot_x", "rot_y", "rot_z"]
    
    # Process sequences
    print("Building sequences...")
    seq_groups = df.groupby("sequence_id")
    X_list, y_list, lengths = process_sequences(seq_groups, imu_cols)
    
    # Determine padding length
    pad_len = calculate_padding_length(lengths, config.sequence_percentile)
    
    # Pad sequences
    X = np.stack([pad_or_truncate_sequence(arr, pad_len) for arr in X_list])
    y = np.array(y_list, dtype=np.int64)
    
    print(f"Final data shape: X={X.shape}, y={y.shape}")
    print(f"Number of classes: {len(le.classes_)}")
    print(f"Class distribution: {np.bincount(y)}")
    
    return X, y, len(le.classes_), pad_len, le.classes_

# ================================
# Model Architectures - Simplified
# ================================

class ResNet1DBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 stride: int = 1, dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, 
                              padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, 1, 
                              padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection
        self.skip = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
    
    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        return F.relu(out + residual)

class ResNet1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int, channels: List[int] = [16, 32, 64],
                 dropout: float = 0.3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, channels[0], 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm1d(channels[0])
        self.maxpool = nn.MaxPool1d(3, 2, 1)
        self.dropout = nn.Dropout(dropout)
        
        # ResNet blocks
        layers = []
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            stride = 2 if i > 0 else 1
            layers.append(ResNet1DBlock(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        
        self.layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.dropout(x)
        x = self.layers(x)
        x = self.global_pool(x)
        return self.classifier(x)

class EfficientNet1D(nn.Module):
    def __init__(self, n_features: int, n_classes: int, dropout: float = 0.3):
        super().__init__()
        # Simplified stem
        self.stem = nn.Sequential(
            nn.Conv1d(n_features, 16, 3, 2, 1, bias=False),
            nn.BatchNorm1d(16),
            nn.SiLU(),
            nn.Dropout(dropout)
        )
        
        # Simplified blocks
        self.blocks = nn.Sequential(
            self._make_simple_block(16, 32, 3, 1, dropout),
            self._make_simple_block(32, 64, 3, 2, dropout),
        )
        
        # Simplified head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )

    def _make_simple_block(self, in_ch: int, out_ch: int, kernel_size: int, 
                          stride: int, dropout: float):
        return nn.Sequential(
            nn.Conv1d(in_ch, out_ch, kernel_size, stride, kernel_size//2, bias=False),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        return self.head(x)

class LSTMAttention(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 64,
                 num_layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(n_features, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0, 
                           bidirectional=True)
        
        # Simplified attention
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim // 2),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Softmax(dim=1)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        
        # Attention weights
        attention_weights = self.attention(lstm_out)
        weighted_output = torch.sum(lstm_out * attention_weights, dim=1)
        
        return self.classifier(weighted_output)

class CNNLSTM(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dim: int = 32,
                 dropout: float = 0.4):
        super().__init__()
        # Minimal CNN feature extractor
        self.cnn = nn.Sequential(
            nn.Conv1d(n_features, 16, 3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool1d(2),
            
            nn.Conv1d(16, 32, 3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(16)
        )
        
        # Single LSTM layer
        self.lstm = nn.LSTM(32, hidden_dim, 1, batch_first=True, bidirectional=True)
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        # CNN features
        cnn_out = self.cnn(x)
        cnn_out = cnn_out.transpose(1, 2)  # (batch, seq_len, features)
        
        # LSTM
        lstm_out, (h_n, _) = self.lstm(cnn_out)
        # Use last hidden state from both directions
        output = torch.cat([h_n[-2], h_n[-1]], dim=1)
        
        return self.classifier(output)

class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.dropout = nn.Dropout(dropout)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.crop = pad * 2
        
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        res = self.downsample(x)
        out = out[:, :, :-self.crop] if self.crop > 0 else out
        res = res[:, :, -out.shape[2]:] if res.shape[2] > out.shape[2] else res
        return F.relu(out + res)

class TCN(nn.Module):
    def __init__(self, n_features: int, n_classes: int, channels: List[int] = [16, 32],
                 kernel_size: int = 3, dropout: float = 0.4):
        super().__init__()
        layers = []
        in_ch = n_features
        for i, out_ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation=2**i, dropout=dropout))
            in_ch = out_ch
        
        self.tcn = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, n_classes)
        )
    
    def forward(self, x):
        x = self.tcn(x)
        x = self.pool(x)
        return self.classifier(x)

def create_model(model_name: str, n_features: int, n_classes: int, config: Config):
    """Factory function to create models"""
    models = {
        "resnet1d": lambda: ResNet1D(n_features, n_classes, dropout=config.dropout),
        "efficientnet1d": lambda: EfficientNet1D(n_features, n_classes, dropout=config.dropout),
        "lstm_attention": lambda: LSTMAttention(n_features, n_classes, 
                                              hidden_dim=config.hidden_dim, 
                                              num_layers=config.num_layers, 
                                              dropout=config.dropout),
        "cnn_lstm": lambda: CNNLSTM(n_features, n_classes, 
                                   hidden_dim=config.hidden_dim, 
                                   dropout=config.dropout),
        "tcn": lambda: TCN(n_features, n_classes, dropout=config.dropout)
    }
    
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(models.keys())}")
    
    return models[model_name]()

# ================================
# Training Functions
# ================================

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate comprehensive metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted'),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0)
    }

class Trainer:
    def __init__(self, model: nn.Module, config: Config, num_classes: int):
        self.model = model.to(config.device)
        self.config = config
        self.num_classes = num_classes
        self.device = config.device
        
        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate, 
                              weight_decay=config.weight_decay, eps=1e-8)
        
        # Scheduler
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=0.7,
                                          patience=5, verbose=True, min_lr=1e-6)
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        self.best_val_f1 = 0.0
        self.patience_counter = 0
    
    def train_single_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc="Training")
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            # Update progress bar
            if len(all_targets) > 0:
                current_acc = accuracy_score(all_targets, all_preds)
                pbar.set_postfix({
                    'Loss': f'{loss.item():.4f}',
                    'Acc': f'{current_acc:.3f}',
                    'LR': f'{self.optimizer.param_groups[0]["lr"]:.1e}'
                })
        
        avg_loss = total_loss / len(train_loader)
        metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
        return avg_loss, metrics
    
    def validate_single_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch"""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            for data, target in pbar:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                if len(all_targets) > 0:
                    current_acc = accuracy_score(all_targets, all_preds)
                    current_f1 = f1_score(all_targets, all_preds, average='weighted')
                    pbar.set_postfix({
                        'Loss': f'{loss.item():.4f}',
                        'Acc': f'{current_acc:.3f}',
                        'F1': f'{current_f1:.3f}'
                    })
        
        avg_loss = total_loss / len(val_loader)
        metrics = calculate_metrics(np.array(all_targets), np.array(all_preds))
        return avg_loss, metrics
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'epoch': epoch,
            'best_val_f1': self.best_val_f1,
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_metrics': self.train_metrics,
            'val_metrics': self.val_metrics
        }
        
        filename = f"best_{self.config.model_name}_checkpoint.pth"
        torch.save(checkpoint_data, filename)
        return filename
    
    def check_early_stopping(self, val_metrics: Dict[str, float], epoch: int) -> bool:
        """Check if early stopping should be triggered"""
        current_f1 = val_metrics['f1_weighted']
        
        if current_f1 > self.best_val_f1:
            self.best_val_f1 = current_f1
            self.best_val_acc = val_metrics['accuracy']
            self.patience_counter = 0
            
            self.save_checkpoint(epoch, is_best=True)
            print(f"â˜… New best F1: {current_f1:.4f} (Acc: {val_metrics['accuracy']:.4f}) - Model saved!")
            return False
        else:
            self.patience_counter += 1
            print(f"No improvement for {self.patience_counter}/{self.config.patience} epochs")
            
            if self.patience_counter >= self.config.patience:
                print("Early stopping triggered!")
                return True
        
        return False
    
    def print_epoch_results(self, epoch: int, train_loss: float, train_metrics: Dict[str, float],
                           val_loss: float, val_metrics: Dict[str, float]):
        """Print results for current epoch"""
        print(f"\nEpoch {epoch+1}/{self.config.max_epochs}")
        print("-" * 70)
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_metrics['accuracy']:.4f}, "
              f"F1: {train_metrics['f1_weighted']:.4f}")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']:.4f}, "
              f"F1: {val_metrics['f1_weighted']:.4f}")
        print(f"Val Precision: {val_metrics['precision_weighted']:.4f}, "
              f"Recall: {val_metrics['recall_weighted']:.4f}")
        print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
    
    def load_best_model(self):
        """Load the best saved model"""
        checkpoint_path = f"best_{self.config.model_name}_checkpoint.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded best model from {checkpoint_path}")
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader):
        """Main training loop"""
        print(f"Starting training for {self.config.max_epochs} epochs...")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        for epoch in range(self.config.max_epochs):
            # Training
            train_loss, train_metrics = self.train_single_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validation
            val_loss, val_metrics = self.validate_single_epoch(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['f1_weighted'])
            
# Check early stopping
            if self.check_early_stopping(val_metrics, epoch):
                break
        
        print(f"\nTraining completed!")
        print(f"Best validation F1: {self.best_val_f1:.4f}")
        print(f"Best validation Accuracy: {self.best_val_acc:.4f}")
        
        # Load best model
        self.load_best_model()
    
    def plot_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy curves
        train_accs = [m['accuracy'] for m in self.train_metrics]
        val_accs = [m['accuracy'] for m in self.val_metrics]
        axes[0, 1].plot(train_accs, label='Train Accuracy', color='blue')
        axes[0, 1].plot(val_accs, label='Val Accuracy', color='red')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # F1 curves
        train_f1s = [m['f1_weighted'] for m in self.train_metrics]
        val_f1s = [m['f1_weighted'] for m in self.val_metrics]
        axes[1, 0].plot(train_f1s, label='Train F1', color='blue')
        axes[1, 0].plot(val_f1s, label='Val F1', color='red')
        axes[1, 0].set_title('Training and Validation F1 Score')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        if hasattr(self.scheduler, 'get_last_lr'):
            lrs = [group['lr'] for group in self.optimizer.param_groups]
            axes[1, 1].plot(lrs, color='green')
            axes[1, 1].set_title('Learning Rate Schedule')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Learning Rate')
            axes[1, 1].set_yscale('log')
            axes[1, 1].grid(True)
        else:
            axes[1, 1].text(0.5, 0.5, 'Learning Rate\nSchedule\nNot Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        plt.savefig(f'{self.config.model_name}_training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

# ================================
# Main Training Pipeline
# ================================

def main():
    """Main training pipeline"""
    print("ğŸš€ Starting Gesture Recognition Training Pipeline")
    print("=" * 60)
    
    # Setup environment
    config = setup_environment(Config())
    
    # Load and process data
    print("ğŸ“Š Loading and processing data...")
    X, y, num_classes, seq_len, class_names = load_and_process_data(config)
    
    # Train-validation split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.val_split, random_state=config.seed, stratify=y
    )
    
    print(f"ğŸ“ˆ Dataset Statistics:")
    print(f"   Train set: {X_train.shape[0]} samples")
    print(f"   Validation set: {X_val.shape[0]} samples")
    print(f"   Sequence length: {seq_len}")
    print(f"   Features: {X.shape[2]}")
    print(f"   Classes: {num_classes}")
    print(f"   Class names: {class_names}")
    
    # Create datasets and dataloaders
    train_dataset = GestureDataset(X_train, y_train, 
                                  augment=config.use_augmentation, 
                                  augment_prob=config.augmentation_prob)
    val_dataset = GestureDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True,
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers, 
        pin_memory=config.pin_memory
    )
    
    # Create model
    print(f"ğŸ�—ï¸� Creating {config.model_name} model...")
    model = create_model(config.model_name, X.shape[2], num_classes, config)
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer and start training
    print("ğŸ�¯ Starting training...")
    trainer = Trainer(model, config, num_classes)
    trainer.fit(train_loader, val_loader)
    
    # Plot training history
    print("ğŸ“Š Generating training plots...")
    trainer.plot_history()
    
    # Save final artifacts
    print("ğŸ’¾ Saving final artifacts...")
    final_checkpoint = {
        'model_state_dict': trainer.model.state_dict(),
        'config': config,
        'class_names': class_names,
        'num_classes': num_classes,
        'sequence_length': seq_len,
        'best_val_f1': trainer.best_val_f1,
        'best_val_acc': trainer.best_val_acc,
        'final_train_metrics': trainer.train_metrics[-1] if trainer.train_metrics else None,
        'final_val_metrics': trainer.val_metrics[-1] if trainer.val_metrics else None,
        'training_history': {
            'train_losses': trainer.train_losses,
            'val_losses': trainer.val_losses,
            'train_metrics': trainer.train_metrics,
            'val_metrics': trainer.val_metrics
        }
    }
    
    torch.save(final_checkpoint, f"final_{config.model_name}_checkpoint.pth")
    
    # Save additional artifacts
    np.save(f"{config.model_name}_training_losses.npy", trainer.train_losses)
    np.save(f"{config.model_name}_validation_losses.npy", trainer.val_losses)
    
    print("âœ… Training completed and artifacts saved!")
    print(f"   Best validation F1: {trainer.best_val_f1:.4f}")
    print(f"   Best validation Accuracy: {trainer.best_val_acc:.4f}")
    
    return trainer, final_checkpoint

# ================================
# Model Comparison Utility
# ================================

def compare_models(models_to_test: List[str] = ["resnet1d", "efficientnet1d", "lstm_attention", "tcn"]):
    """Compare different model architectures"""
    print("ğŸ”� Starting Model Comparison")
    print("=" * 60)
    
    results = {}
    config = setup_environment(Config())
    
    # Load data once
    print("ğŸ“Š Loading data for comparison...")
    X, y, num_classes, seq_len, class_names = load_and_process_data(config)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.val_split, random_state=config.seed, stratify=y
    )
    
    # Create datasets
    train_dataset = GestureDataset(X_train, y_train, 
                                  augment=config.use_augmentation,
                                  augment_prob=config.augmentation_prob)
    val_dataset = GestureDataset(X_val, y_val, augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=config.num_workers)
    
    # Test each model
    for i, model_name in enumerate(models_to_test):
        print(f"\n{'='*50}")
        print(f"Testing {model_name.upper()} ({i+1}/{len(models_to_test)})")
        print('='*50)
        
        # Update config for current model
        original_model_name = config.model_name
        config.model_name = model_name
        
        try:
            # Create and train model
            model = create_model(model_name, X.shape[2], num_classes, config)
            trainer = Trainer(model, config, num_classes)
            
            print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
            
            # Train with reduced epochs for comparison
            original_epochs = config.max_epochs
            config.max_epochs = min(50, config.max_epochs)  # Limit epochs for comparison
            
            trainer.fit(train_loader, val_loader)
            
            # Store results
            results[model_name] = {
                'best_val_f1': trainer.best_val_f1,
                'best_val_acc': trainer.best_val_acc,
                'final_train_acc': trainer.train_metrics[-1]['accuracy'] if trainer.train_metrics else 0,
                'final_val_acc': trainer.val_metrics[-1]['accuracy'] if trainer.val_metrics else 0,
                'final_train_f1': trainer.train_metrics[-1]['f1_weighted'] if trainer.train_metrics else 0,
                'final_val_f1': trainer.val_metrics[-1]['f1_weighted'] if trainer.val_metrics else 0,
                'num_params': sum(p.numel() for p in model.parameters()),
                'epochs_trained': len(trainer.train_losses)
            }
            
            # Save individual model
            torch.save({
                'model_state_dict': trainer.model.state_dict(),
                'results': results[model_name],
                'config': config
            }, f"comparison_{model_name}_checkpoint.pth")
            
            # Restore original epochs
            config.max_epochs = original_epochs
            
        except Exception as e:
            print(f"â�Œ Error training {model_name}: {str(e)}")
            results[model_name] = {
                'error': str(e),
                'best_val_f1': 0,
                'best_val_acc': 0,
                'num_params': 0
            }
        
        # Restore original model name
        config.model_name = original_model_name
    
    # Print comparison results
    print(f"\n{'='*80}")
    print("ğŸ�† MODEL COMPARISON RESULTS")
    print('='*80)
    print(f"{'Model':<15} {'Best F1':<10} {'Best Acc':<10} {'Params':<12} {'Final F1':<10} {'Epochs':<8}")
    print('-'*80)
    
    # Sort by best F1 score
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('best_val_f1', 0), reverse=True)
    
    for model_name, metrics in sorted_results:
        if 'error' not in metrics:
            print(f"{model_name:<15} {metrics['best_val_f1']:<10.4f} "
                  f"{metrics['best_val_acc']:<10.4f} {metrics['num_params']:<12,} "
                  f"{metrics['final_val_f1']:<10.4f} {metrics['epochs_trained']:<8}")
        else:
            print(f"{model_name:<15} {'ERROR':<10} {'ERROR':<10} {'ERROR':<12} {'ERROR':<10} {'ERROR':<8}")
    
    # Save comparison results
    comparison_results = {
        'results': results,
        'config_used': config,
        'dataset_info': {
            'num_classes': num_classes,
            'sequence_length': seq_len,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'class_names': class_names
        }
    }
    
    torch.save(comparison_results, "model_comparison_results.pth")
    
    # Create comparison plot
    plt.figure(figsize=(12, 8))
    
    # Plot F1 scores
    plt.subplot(2, 2, 1)
    models = [name for name, res in sorted_results if 'error' not in res]
    f1_scores = [res['best_val_f1'] for name, res in sorted_results if 'error' not in res]
    plt.bar(models, f1_scores, color='skyblue')
    plt.title('Best Validation F1 Score by Model')
    plt.xticks(rotation=45)
    plt.ylabel('F1 Score')
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    acc_scores = [res['best_val_acc'] for name, res in sorted_results if 'error' not in res]
    plt.bar(models, acc_scores, color='lightgreen')
    plt.title('Best Validation Accuracy by Model')
    plt.xticks(rotation=45)
    plt.ylabel('Accuracy')
    
    # Plot parameters
    plt.subplot(2, 2, 3)
    param_counts = [res['num_params'] for name, res in sorted_results if 'error' not in res]
    plt.bar(models, param_counts, color='coral')
    plt.title('Model Parameters Count')
    plt.xticks(rotation=45)
    plt.ylabel('Parameters')
    plt.yscale('log')
    
    # Plot efficiency (F1 / params)
    plt.subplot(2, 2, 4)
    efficiency = [f1/params*1000000 for f1, params in zip(f1_scores, param_counts) if params > 0]
    plt.bar(models, efficiency, color='gold')
    plt.title('Model Efficiency (F1 Score per Million Parameters)')
    plt.xticks(rotation=45)
    plt.ylabel('Efficiency')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nâœ… Model comparison completed!")
    print(f"   Best model: {sorted_results[0][0]} (F1: {sorted_results[0][1]['best_val_f1']:.4f})")
    
    return results

# ================================
# Inference Functions
# ================================

def load_trained_model(checkpoint_path: str, device: str = None):
    """Load a trained model from checkpoint"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config_used = checkpoint['config']
    
    # Create model
    model = create_model(
        config_used.model_name, 
        7,  # Number of IMU features
        checkpoint['num_classes'], 
        config_used
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, checkpoint

def predict_gesture(model, sequence_data: np.ndarray, class_names: List[str], device: str = "cpu"):
    """Predict gesture from sequence data"""
    model.eval()
    with torch.no_grad():
        # Preprocess sequence
        if len(sequence_data.shape) == 2:
            sequence_data = sequence_data[np.newaxis, :]  # Add batch dimension
        
        x = torch.from_numpy(sequence_data).float().to(device)
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        
        # Get predictions
        outputs = model(x)
        probabilities = F.softmax(outputs, dim=1)
        predicted_class = outputs.argmax(dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        return {
            'predicted_class': predicted_class,
            'predicted_gesture': class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': probabilities[0].cpu().numpy()
        }

# ================================
# Run Training
# ================================

if __name__ == "__main__":
    # Setup environment
    config = setup_environment(Config())
    
    print("ğŸ�¯ Gesture Recognition Training")
    print(f"Device: {config.device}")
    print(f"Model: {config.model_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Max epochs: {config.max_epochs}")
    
    # Run main training
    trainer, checkpoint = main()
    
    # Optionally run model comparison
    print("\n" + "="*60)
    comparison_results = compare_models()
    
    print("\nğŸ�‰ All done! Check the saved artifacts:")
    print("   - final_*_checkpoint.pth (best model)")
    print("   - *_training_history.png (training plots)")
    print("   - gesture_classes.npy (class names)")
    print("   - sequence_maxlen.npy (sequence length)")
    if 'comparison_results' in locals():
        print("   - model_comparison_results.pth (comparison results)")
        print("   - model_comparison.png (comparison plots)")

