import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import warnings
from scipy.stats import pearsonr
import random
import os
import pickle
from pathlib import Path
warnings.filterwarnings('ignore')

def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"ğŸŒ± Set all random seeds to {seed}")

class Swish(nn.Module):
    """Swish activation function - prevents dead neurons and provides smooth gradients"""
    def forward(self, x):
        return x * torch.sigmoid(x)

class GaussianNoise(nn.Module):
    """Gaussian noise layer for data augmentation and overfitting prevention"""
    def __init__(self, std=0.05):
        super(GaussianNoise, self).__init__()
        self.std = std
        
    def forward(self, x):
        if self.training:
            noise = torch.randn_like(x) * self.std
            return x + noise
        return x

class AutoEncoder(nn.Module):
    """AutoEncoder for feature learning with proper expanding encoder and compressing decoder"""
    def __init__(self, input_size, encoding_size=128, dropout=0.7):
        super(AutoEncoder, self).__init__()
        
        # Calculate intermediate dimensions for smooth expansion/compression
        # For input_size=25, we want: 25 -> 64 -> 128 -> 128 (encoding)
        hidden1_size = max(input_size * 2, 64)  # First expansion
        hidden2_size = max(hidden1_size, encoding_size)  # Second expansion to at least encoding_size
        
        # Encoder pathway - EXPAND dimensions for richer feature representation
        self.encoder = nn.Sequential(
            # First expansion: input_size -> hidden1_size (25 -> 64)
            nn.Linear(input_size, hidden1_size),
            nn.BatchNorm1d(hidden1_size),
            Swish(),
            nn.Dropout(dropout),
            
            # Second expansion: hidden1_size -> hidden2_size (64 -> 128)
            nn.Linear(hidden1_size, hidden2_size),
            nn.BatchNorm1d(hidden2_size),
            Swish(),
            nn.Dropout(dropout),
            
            # Final encoding layer: hidden2_size -> encoding_size (128 -> 128)
            nn.Linear(hidden2_size, encoding_size),
            nn.BatchNorm1d(encoding_size),
            Swish()
        )
        
        # Decoder pathway - COMPRESS back to original dimensions
        self.decoder = nn.Sequential(
            # First decompression: encoding_size -> hidden2_size (128 -> 128)
            nn.Linear(encoding_size, hidden2_size),
            nn.BatchNorm1d(hidden2_size),
            Swish(),
            nn.Dropout(dropout),
            
            # Second decompression: hidden2_size -> hidden1_size (128 -> 64)
            nn.Linear(hidden2_size, hidden1_size),
            nn.BatchNorm1d(hidden1_size),
            Swish(),
            nn.Dropout(dropout),
            
            # Final reconstruction: hidden1_size -> input_size (64 -> 25)
            nn.Linear(hidden1_size, input_size)
        )
        
        # Store dimensions for reference
        self.input_size = input_size
        self.encoding_size = encoding_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        
        print(f"AutoEncoder Architecture: {input_size} -> {hidden1_size} -> {hidden2_size} -> {encoding_size} -> {hidden2_size} -> {hidden1_size} -> {input_size}")
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class CryptoMLPWithAutoEncoder(nn.Module):
    """Enhanced MLP with AutoEncoder feature learning and Gaussian noise augmentation"""
    def __init__(self, input_size, encoding_size=128, dropout=0.7, hidden_size=256, noise_std=0.05):
        super(CryptoMLPWithAutoEncoder, self).__init__()
        
        # Gaussian noise layer for data augmentation
        self.noise_layer = GaussianNoise(std=noise_std)
        
        # AutoEncoder for feature learning (now with proper expanding/compressing architecture)
        self.autoencoder = AutoEncoder(input_size, encoding_size, dropout=0.7)
        
        # Combined input size: original features + encoded features
        combined_input_size = input_size + encoding_size
        self.input_bn = nn.BatchNorm1d(combined_input_size)
        
        # Block 0: 3 dense layers with hidden_size units
        self.block0_layer1 = nn.Linear(combined_input_size, hidden_size)
        self.block0_bn1 = nn.BatchNorm1d(hidden_size)
        self.block0_layer2 = nn.Linear(hidden_size, hidden_size)
        self.block0_bn2 = nn.BatchNorm1d(hidden_size)
        self.block0_layer3 = nn.Linear(hidden_size, hidden_size)
        self.block0_bn3 = nn.BatchNorm1d(hidden_size)
        
        # Blocks 1-8: 2 dense layers each with hidden_size units
        self.blocks = nn.ModuleList()
        self.blocks_bn1 = nn.ModuleList()
        self.blocks_bn2 = nn.ModuleList()
        
        for i in range(8):
            block = nn.ModuleDict({
                'layer1': nn.Linear(hidden_size, hidden_size),
                'layer2': nn.Linear(hidden_size, hidden_size)
            })
            self.blocks.append(block)
            self.blocks_bn1.append(nn.BatchNorm1d(hidden_size))
            self.blocks_bn2.append(nn.BatchNorm1d(hidden_size))
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.swish = Swish()
        
        # Output layer
        self.output = nn.Linear(hidden_size, 1)
        
        # Store sizes for reference
        self.hidden_size = hidden_size
        self.encoding_size = encoding_size
        self.input_size = input_size
        
        print(f"Main Network: {combined_input_size} features -> {hidden_size} hidden -> 1 output")
        
    def forward(self, x, return_ae_loss=False):
        # Apply Gaussian noise for data augmentation (only during training)
        x_noisy = self.noise_layer(x)
        
        # Get autoencoder features
        encoded, decoded = self.autoencoder(x_noisy)
        
        # Concatenate original features with encoded features
        x_combined = torch.cat([x, encoded], dim=1)
        
        # Input block with BatchNorm
        x_combined = self.input_bn(x_combined)
        
        # Block 0: 3 dense layers
        x0 = self.block0_layer1(x_combined)
        x0 = self.block0_bn1(x0)
        x0 = self.swish(x0)
        x0 = self.dropout(x0)
        
        x0 = self.block0_layer2(x0)
        x0 = self.block0_bn2(x0)
        x0 = self.swish(x0)
        x0 = self.dropout(x0)
        
        x0 = self.block0_layer3(x0)
        x0 = self.block0_bn3(x0)
        x0 = self.swish(x0)
        
        # Blocks 1-8 with skip connections from Block 0
        x_current = x0
        for i, (block, bn1, bn2) in enumerate(zip(self.blocks, self.blocks_bn1, self.blocks_bn2)):
            # First layer of block i+1
            x_block = block['layer1'](x_current)
            x_block = bn1(x_block)
            x_block = self.swish(x_block)
            x_block = self.dropout(x_block)
            
            # Second layer of block i+1
            x_block = block['layer2'](x_block)
            x_block = bn2(x_block)
            x_block = self.swish(x_block)
            
            # Skip connection from Block 0
            x_current = x_block + x0
        
        # Output layer
        out = self.output(x_current)
        
        if return_ae_loss:
            return out, decoded, x_noisy
        return out

class CryptoDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels) if labels is not None else None
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        if self.labels is not None:
            return self.features[idx], self.labels[idx]
        return self.features[idx]

def calculate_correlation(predictions, targets):
    """Calculate Pearson correlation coefficient"""
    predictions = predictions.flatten()
    targets = targets.flatten()
    
    # Remove any NaN or inf values
    mask = np.isfinite(predictions) & np.isfinite(targets)
    if mask.sum() < 2:
        return 0.0
    
    predictions = predictions[mask]
    targets = targets[mask]
    
    if np.std(predictions) == 0 or np.std(targets) == 0:
        return 0.0
    
    try:
        corr, _ = pearsonr(predictions, targets)
        return corr if not np.isnan(corr) else 0.0
    except:
        return 0.0

class PearsonCorrelationLoss(nn.Module):
    """Pearson correlation coefficient loss function"""
    def __init__(self, eps=1e-8):
        super(PearsonCorrelationLoss, self).__init__()
        self.eps = eps
        
    def forward(self, y_pred, y_true):
        y_pred = y_pred.view(-1)
        y_true = y_true.view(-1)
        
        y_pred_centered = y_pred - torch.mean(y_pred)
        y_true_centered = y_true - torch.mean(y_true)
        
        numerator = torch.sum(y_pred_centered * y_true_centered)
        
        pred_std = torch.sqrt(torch.sum(y_pred_centered ** 2) + self.eps)
        true_std = torch.sqrt(torch.sum(y_true_centered ** 2) + self.eps)
        denominator = pred_std * true_std
        
        correlation = numerator / denominator
        return -correlation

class CombinedLossWithAE(nn.Module):
    """Combined MSE, Correlation Loss, and AutoEncoder Reconstruction Loss"""
    def __init__(self, mse_weight=0.25, corr_weight=0.6, ae_weight=0.15, eps=1e-8):
        super(CombinedLossWithAE, self).__init__()
        self.mse_weight = mse_weight
        self.corr_weight = corr_weight
        self.ae_weight = ae_weight
        self.eps = eps
        self.mse_loss = nn.MSELoss()
        
    def forward(self, y_pred, y_true, decoded=None, original=None):
        # Primary prediction loss (MSE)
        mse = self.mse_loss(y_pred, y_true)
        
        # Correlation loss
        y_pred_flat = y_pred.view(-1)
        y_true_flat = y_true.view(-1)
        
        y_pred_centered = y_pred_flat - torch.mean(y_pred_flat)
        y_true_centered = y_true_flat - torch.mean(y_true_flat)
        
        numerator = torch.sum(y_pred_centered * y_true_centered)
        pred_std = torch.sqrt(torch.sum(y_pred_centered ** 2) + self.eps)
        true_std = torch.sqrt(torch.sum(y_true_centered ** 2) + self.eps)
        denominator = pred_std * true_std
        
        correlation = numerator / denominator
        
        # AutoEncoder reconstruction loss
        ae_loss = 0.0
        if decoded is not None and original is not None:
            ae_loss = self.mse_loss(decoded, original)
        
        # Combined loss
        total_loss = (self.mse_weight * mse - 
                     self.corr_weight * correlation + 
                     self.ae_weight * ae_loss)
        
        return total_loss, mse.item(), correlation.item(), ae_loss.item() if isinstance(ae_loss, torch.Tensor) else ae_loss

def load_and_preprocess_data():
    """Load and preprocess the crypto data"""
    train_path = "/kaggle/input/drw-crypto-market-prediction/train.parquet"
    test_path = "/kaggle/input/drw-crypto-market-prediction/test.parquet"
    
    print("Loading data...")
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)
    
    print(f"Train shape: {train_df.shape}")
    print(f"Test shape: {test_df.shape}")
    
    # Remove redundant columns and keep only the selected features
    cols_to_keep = [
        "X863", "X856", "X344", "X598", "X862", "X385", "X852", "X603", "X860", "X674",
        "X415", "X345", "X137", "X855", "X174", "X302", "X178", "X532", "X168", "X612",
        "bid_qty", "ask_qty", "buy_qty", "sell_qty", "volume", 'label'
    ]
    
    # Ensure all columns exist in both datasets
    available_cols_train = [col for col in cols_to_keep if col in train_df.columns]
    available_cols_test = [col for col in cols_to_keep if col in test_df.columns and col != 'label']
    
    print(f"Available features: {len(available_cols_train) - 1}")  # -1 for label
    
    train_df = train_df[available_cols_train]
    test_df = test_df[available_cols_test + ['label']]  # test has label column but it's all zeros
    
    # Prepare features and labels
    feature_cols = [col for col in train_df.columns if col not in ['timestamp', 'label']]
    
    print(f"Final feature count: {len(feature_cols)}")
    print(f"Features: {feature_cols}")
    
    # Sort by timestamp for time series split
    if 'timestamp' in train_df.columns:
        train_df = train_df.sort_values('timestamp')
    train_df = train_df.reset_index(drop=True)
    
    X_full = train_df[feature_cols].values
    y_full = train_df['label'].values
    
    # Handle missing values and infinities
    X_full = np.nan_to_num(X_full, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Prepare test data
    X_test = test_df[feature_cols].values
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    print(f"Train features shape: {X_full.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    return X_full, X_test, y_full, train_df, X_full.shape[1]

def train_single_fold(fold_idx, train_idx, val_idx, X_full, y_full, X_test, num_features, 
                     model_dir, num_epochs=80, lr=0.0001, encoding_size=128, seed=42):
    """Train a single enhanced model for one time series fold"""
    
    print(f"\n{'='*80}")
    print(f"ğŸ�‹ï¸�â€�â™‚ï¸� Training Enhanced Fold {fold_idx + 1} with Corrected AutoEncoder")
    print(f"{'='*80}")
    print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
    print(f"Input features: {num_features}, Encoding size: {encoding_size}")
    
    # Set seed for this fold
    set_random_seeds(seed + fold_idx)  # Different seed per fold for diversity
    
    # Get train and validation data for this fold
    X_train = X_full[train_idx]
    X_val = X_full[val_idx]
    y_train = y_full[train_idx]
    y_val = y_full[val_idx]
    
    # Create datasets and dataloaders
    train_dataset = CryptoDataset(X_train, y_train)
    val_dataset = CryptoDataset(X_val, y_val)
    
    batch_size = 4096
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize enhanced model with corrected AutoEncoder
    model = CryptoMLPWithAutoEncoder(
        input_size=num_features, 
        encoding_size=encoding_size,
        dropout=0.7,
        hidden_size=256,
        noise_std=0.05
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Initialize weights
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    
    model.apply(init_weights)
    
    # Use combined loss with AutoEncoder
    criterion = CombinedLossWithAE(mse_weight=0.25, corr_weight=0.6, ae_weight=0.15)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=5, factor=0.5, min_lr=1e-6
    )
    
    best_val_corr = -float('inf')
    best_model_state = None
    best_epoch = 0
    early_stop_counter = 0
    patience = 10
    
    print(f"Training on {device} | Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_mse_losses = []
        train_corr_losses = []
        train_ae_losses = []
        
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass with AutoEncoder loss
            outputs, decoded, noisy_input = model(batch_features, return_ae_loss=True)
            
            # Calculate combined loss
            loss, mse_loss, corr_loss, ae_loss = criterion(
                outputs, batch_labels, decoded, noisy_input
            )
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            train_mse_losses.append(mse_loss)
            train_corr_losses.append(corr_loss)
            train_ae_losses.append(ae_loss)
        
        # Validation phase
        model.eval()
        val_losses = []
        val_predictions = []
        val_targets = []
        val_mse_losses = []
        val_corr_losses = []
        val_ae_losses = []
        
        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                batch_features = batch_features.to(device)
                batch_labels = batch_labels.to(device)
                
                outputs, decoded, noisy_input = model(batch_features, return_ae_loss=True)
                loss, mse_loss, corr_loss, ae_loss = criterion(
                    outputs, batch_labels, decoded, noisy_input
                )
                
                val_losses.append(loss.item())
                val_mse_losses.append(mse_loss)
                val_corr_losses.append(corr_loss)
                val_ae_losses.append(ae_loss)
                val_predictions.extend(outputs.cpu().numpy().flatten())
                val_targets.extend(batch_labels.cpu().numpy().flatten())
        
        # Calculate metrics
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        avg_train_mse = np.mean(train_mse_losses)
        avg_val_mse = np.mean(val_mse_losses)
        avg_train_ae = np.mean(train_ae_losses)
        avg_val_ae = np.mean(val_ae_losses)
        
        val_pred_array = np.array(val_predictions)
        val_target_array = np.array(val_targets)
        val_corr = calculate_correlation(val_pred_array, val_target_array)
        
        scheduler.step(val_corr)
        
        # Save best model
        improvement = ""
        if val_corr > best_val_corr:
            best_val_corr = val_corr
            best_model_state = model.state_dict().copy()
            best_epoch = epoch + 1
            early_stop_counter = 0
            improvement = " â­�"
        else:
            early_stop_counter += 1
        
        # Print progress every 10 epochs with detailed loss breakdown
        if (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Loss: {avg_val_loss:.6f} | "
                  f"MSE: {avg_val_mse:.6f} | "
                  f"AE: {avg_val_ae:.6f} | "
                  f"Corr: {val_corr:.6f} | "
                  f"Best: {best_val_corr:.6f} | "
                  f"LR: {current_lr:.2e} | "
                  f"ES: {early_stop_counter}/{patience}{improvement}")
        
        # Early stopping
        if early_stop_counter >= patience:
            print(f"ğŸ›‘ Early stopping at epoch {epoch+1}")
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    # Save model
    model_path = model_dir / f"enhanced_model_fold_{fold_idx}.pt"
    torch.save({
        'model_state_dict': best_model_state,
        'best_val_corr': best_val_corr,
        'best_epoch': best_epoch,
        'fold_idx': fold_idx,
        'num_features': num_features,
        'encoding_size': encoding_size
    }, model_path)
    
    # Make predictions on test set
    test_dataset = CryptoDataset(X_test)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    model.eval()
    test_predictions = []
    
    with torch.no_grad():
        for batch_features in test_loader:
            batch_features = batch_features.to(device)
            outputs = model(batch_features, return_ae_loss=False)
            test_predictions.append(outputs.cpu().numpy())
    
    test_pred_array = np.vstack(test_predictions)[:, 0]
    
    print(f"âœ… Enhanced Fold {fold_idx + 1} completed | Best Correlation: {best_val_corr:.6f} (Epoch {best_epoch})")
    
    return {
        'fold_idx': fold_idx,
        'model': model,
        'best_val_corr': best_val_corr,
        'best_epoch': best_epoch,
        'test_predictions': test_pred_array,
        'model_path': model_path
    }

def enhanced_timeseries_ensemble_training(n_splits=5, max_train_size=100_000_000, gap=1, 
                                        num_epochs=80, lr=0.0001, encoding_size=128, seed=42):
    """Train enhanced ensemble with corrected AutoEncoder using TimeSeriesSplit"""
    
    print(f"ğŸš€ DRW Crypto Enhanced TimeSeriesSplit Ensemble Training Framework")
    print(f"ğŸ”§ Enhancements: Corrected AutoEncoder + Gaussian Noise + Swish Activation")
    print(f"N Splits: {n_splits}")
    print(f"Max Train Size: {max_train_size:,}")
    print(f"Gap: {gap}")
    print(f"Epochs per Model: {num_epochs}")
    print(f"Encoding Size: {encoding_size}")
    print(f"Learning Rate: {lr}")
    print("=" * 80)
    
    # Create model directory
    model_dir = Path("/kaggle/working/enhanced_ensemble_models")
    model_dir.mkdir(exist_ok=True)
    
    # Load data once
    print("ğŸ”„ Loading data...")
    X_full, X_test, y_full, train_df, num_features = load_and_preprocess_data()
    
    # Create TimeSeriesSplit
    tss = TimeSeriesSplit(n_splits=n_splits, max_train_size=max_train_size, gap=gap)
    all_splits = list(tss.split(train_df.index))
    
    print(f"ğŸ“Š TimeSeriesSplit created {len(all_splits)} folds")
    
    # Store all results
    all_results = []
    all_test_predictions = []
    
    # Train each fold
    for fold_idx, (train_idx, val_idx) in enumerate(all_splits):
        print(f"\nğŸš€ Training Enhanced Fold {fold_idx + 1}/{len(all_splits)}")
        
        result = train_single_fold(
            fold_idx=fold_idx,
            train_idx=train_idx,
            val_idx=val_idx,
            X_full=X_full,
            y_full=y_full,
            X_test=X_test,
            num_features=num_features,
            model_dir=model_dir,
            num_epochs=num_epochs,
            lr=lr,
            encoding_size=encoding_size,
            seed=seed
        )
        
        all_results.append(result)
        all_test_predictions.append(result['test_predictions'])
        
        # Print intermediate summary
        correlations = [r['best_val_corr'] for r in all_results]
        print(f"ğŸ“Š Progress: {fold_idx + 1}/{len(all_splits)} | "
              f"Current: {result['best_val_corr']:.6f} | "
              f"Avg so far: {np.mean(correlations):.6f} | "
              f"Best so far: {np.max(correlations):.6f}")
    
    # Ensemble predictions
    print(f"\nğŸ”® Creating Enhanced Ensemble Predictions...")
    
    # Simple average ensemble
    ensemble_predictions = np.mean(all_test_predictions, axis=0)
    
    # Weighted ensemble based on validation performance
    weights = np.array([r['best_val_corr'] for r in all_results])
    weights = np.maximum(weights, 0)  # Ensure non-negative weights
    weights = weights / np.sum(weights)  # Normalize
    
    weighted_ensemble_predictions = np.average(all_test_predictions, axis=0, weights=weights)
    
    # Create submissions
    sample_sub_path = "/kaggle/input/drw-crypto-market-prediction/sample_submission.csv"
    sample_submission = pd.read_csv(sample_sub_path)
    
    # Simple ensemble submission
    simple_submission = sample_submission.copy()
    simple_submission.iloc[:, 1] = ensemble_predictions
    simple_submission.to_csv('/kaggle/working/enhanced_ensemble_simple_submission.csv', index=False)
    
    # Weighted ensemble submission
    weighted_submission = sample_submission.copy()
    weighted_submission.iloc[:, 1] = weighted_ensemble_predictions
    weighted_submission.to_csv('/kaggle/working/enhanced_ensemble_weighted_submission.csv', index=False)
    
    # Save ensemble results
    ensemble_results = {
        'all_results': all_results,
        'ensemble_predictions': ensemble_predictions,
        'weighted_ensemble_predictions': weighted_ensemble_predictions,
        'weights': weights,
        'all_splits': all_splits,
        'encoding_size': encoding_size
    }
    
    with open(model_dir / 'enhanced_ensemble_results.pkl', 'wb') as f:
        pickle.dump(ensemble_results, f)
    
    # Print final summary
    print(f"\nğŸ�‰ Enhanced TimeSeriesSplit Ensemble Training Completed!")
    print("=" * 80)
    print("ğŸ“Š Individual Fold Performance:")
    
    for i, result in enumerate(all_results):
        print(f"  Fold {i+1:2d}: "
              f"Corr = {result['best_val_corr']:8.6f} | "
              f"Epoch = {result['best_epoch']:3d} | "
              f"Weight = {weights[i]:6.4f}")
    
    correlations = [r['best_val_corr'] for r in all_results]
    print(f"\nğŸ“ˆ Enhanced Ensemble Statistics:")
    print(f"  Average Correlation: {np.mean(correlations):.6f}")
    print(f"  Best Single Fold:   {np.max(correlations):.6f}")
    print(f"  Worst Single Fold:  {np.min(correlations):.6f}")
    print(f"  Standard Deviation:  {np.std(correlations):.6f}")
    
    print(f"\nğŸ”§ Architecture Enhancements Applied:")
    print(f"  âœ… Corrected AutoEncoder: Expanding Encoder -> Compressing Decoder")
    print(f"  âœ… AutoEncoder latent space: {encoding_size}D")
    print(f"  âœ… Gaussian Noise data augmentation (Ïƒ=0.05)")
    print(f"  âœ… Swish activation for smooth gradients")
    print(f"  âœ… Combined loss: MSE(0.25) + Correlation(0.6) + AE(0.15)")
    
    print(f"\nğŸ’¾ Enhanced Files Saved:")
    print(f"  Simple Ensemble:   enhanced_ensemble_simple_submission.csv")
    print(f"  Weighted Ensemble: enhanced_ensemble_weighted_submission.csv")
    print(f"  Model Directory:   {model_dir}")
    print(f"  Results Pickle:    enhanced_ensemble_results.pkl")
    
    return ensemble_results

def main():
    """Main enhanced ensemble training pipeline"""
    
    # Run Enhanced TimeSeriesSplit ensemble training
    results = enhanced_timeseries_ensemble_training(
        n_splits=4,
        max_train_size=100_000_000,
        gap=5000,
        num_epochs=80,  # Adjust based on your time constraints
        lr=0.0001,
        encoding_size=64,  # AutoEncoder latent dimension
        seed=42
    )
    
    return results

if __name__ == "__main__":
    enhanced_ensemble_results = main()

