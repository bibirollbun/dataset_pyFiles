
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from scipy import signal
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
import warnings
import os

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# --- 1. Improved PyTorch Neural Network Definition ---
# --- 1. IMPROVED PyTorch Neural Network Definition (Better Accuracy) ---

class ResidualBlock(nn.Module):
    """An improved residual block with dropout and better activation."""
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.act = nn.SiLU()  # Swish activation (better than ReLU/ELU in many cases)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout2(out)
        out += self.shortcut(x)
        out = self.act(out)
        return out

class RamanResNet(nn.Module):
    """A deeper ResNet-style 1D CNN with improved accuracy."""
    def __init__(self, input_channels=3, num_classes=3):
        super(RamanResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=9, stride=2, padding=4)
        self.bn1 = nn.BatchNorm1d(64)
        self.act = nn.SiLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.SiLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        strides = [stride] + [1] * (num_blocks - 1)
        for s in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride=s))
            self.in_channels = out_channels
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
       

# --- 2. Data Loading and Advanced Preprocessing ---

def load_and_preprocess_data(filepath, is_train=True):
    # (This function remains unchanged)
    if is_train:
        df = pd.read_csv(filepath)
        target_cols = ['Glucose (g/L)', 'Sodium Acetate (g/L)', 'Magnesium Acetate (g/L)']
        y = df[target_cols].dropna().values
        X = df.iloc[:, :-4]
    else:
        df = pd.read_csv(filepath, header=None)
        X = df; y = None
    X.columns = ["sample_id"] + [str(i) for i in range(X.shape[1]-1)]
    X['sample_id'] = X['sample_id'].ffill()
    if is_train: X['sample_id'] = X['sample_id'].str.strip()
    else: X['sample_id'] = X['sample_id'].str.strip().str.replace('sample', '').astype(int)
    spectral_cols = X.columns[1:]
    for col in spectral_cols:
        X[col] = X[col].astype(str).str.replace('[', '', regex=False).str.replace(']', '', regex=False)
        X[col] = pd.to_numeric(X[col], errors='coerce')
    return X, y

def get_advanced_spectra_features(X):
    """Create multi-channel features from spectra: raw, 1st derivative, 2nd derivative."""
    X_processed = np.zeros_like(X)
    # Baseline correction and SNV
    for i in range(X.shape[0]):
        poly = np.polyfit(np.arange(X.shape[1]), X[i], 3)
        baseline = np.polyval(poly, np.arange(X.shape[1]))
        corrected_spec = X[i] - baseline
        X_processed[i] = (corrected_spec - corrected_spec.mean()) / (corrected_spec.std() + 1e-8)

    # Calculate derivatives
    deriv1 = signal.savgol_filter(X_processed, window_length=11, polyorder=3, deriv=1, axis=1)
    deriv2 = signal.savgol_filter(X_processed, window_length=11, polyorder=3, deriv=2, axis=1)

    # Stack as channels
    return np.stack([X_processed, deriv1, deriv2], axis=1)

# --- 3. Model Training and Evaluation with Enhancements ---

def evaluate_model(model, val_loader, criterion):
    # (This function remains unchanged)
    model.eval()
    total_loss = 0; all_targets = []; all_predictions = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            all_targets.append(targets.numpy())
            all_predictions.append(outputs.numpy())
    avg_loss = total_loss / len(val_loader)
    all_targets = np.concatenate(all_targets, axis=0)
    all_predictions = np.concatenate(all_predictions, axis=0)
    r2 = r2_score(all_targets, all_predictions)
    return avg_loss, r2

def train_nn_model(X_train, y_train, X_val, y_val, num_epochs=200, batch_size=7, learning_rate=1e-5):
    """Train the RamanResNet model with scheduler, early stopping, and checkpointing."""
    print(f"Training ResNet model for up to {num_epochs} epochs...")
    
    # DataLoaders
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)), batch_size=batch_size)

    # Model, Loss, Optimizer, and Scheduler
    model = RamanResNet(input_channels=X_train.shape[1], num_classes=y_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=10, verbose=True)

    # Early Stopping and Checkpointing variables
    best_val_r2 = -np.inf
    epochs_no_improve = 0
    patience = 15 # Number of epochs to wait for improvement
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        val_loss, val_r2 = evaluate_model(model, val_loader, criterion)
        
        print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val R²: {val_r2:.4f}')
        
        scheduler.step(val_loss)

        # Checkpointing
        if val_r2 > best_val_r2:
            best_val_r2 = val_r2
            epochs_no_improve = 0
            best_model_state = model.state_dict()
            print(f"  -> New best validation R²: {best_val_r2:.4f}. Saving model.")
        else:
            epochs_no_improve += 1

        # Early stopping
        if epochs_no_improve >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break
            
    print(f"Training complete. Best validation R²: {best_val_r2:.4f}")
    model.load_state_dict(best_model_state) # Load the best model
    return model

def make_nn_predictions(model, X_test):
    # (This function remains unchanged)
    print("Generating predictions with the best trained model...")
    model.eval()
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    with torch.no_grad():
        predictions = model(X_test_tensor).numpy()
    return predictions

def post_process_predictions(preds):
    # (This function remains unchanged)
    print("Post-processing predictions (ensuring non-negativity)...")
    return np.maximum(preds, 0)

# --- 4. Main Execution Pipeline ---

def main():
    """Main execution pipeline using the improved neural network."""
    print("="*80)
    print("IMPROVED RAMAN REGRESSION PIPELINE (ResNet, Advanced Preprocessing)")
    print("="*80)
    
    # File paths
    train_file = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/transfer_plate.csv'
    test_file = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/96_samples.csv'
    submission_file = '/kaggle/input/dig-4-bio-raman-transfer-learning-challenge/sample_submission.csv'

    # 1. Load Data
    print("1. Loading data...")
    X_train_raw, y_train_full = load_and_preprocess_data(train_file, is_train=True)
    X_test_raw, _ = load_and_preprocess_data(test_file, is_train=False)
    
    X_train_full_array = X_train_raw.drop('sample_id', axis=1).values.reshape(-1, 2, 2048).mean(axis=1)
    X_test_array = X_test_raw.drop('sample_id', axis=1).values.reshape(-1, 2, 2048).mean(axis=1)
    
    # 2. Advanced Feature Engineering & Data Splitting
    print("\n2. Applying advanced spectral preprocessing and splitting data...")
    X_processed = get_advanced_spectra_features(X_train_full_array)
    X_test_processed = get_advanced_spectra_features(X_test_array)
    print(f"Input feature shape: {X_processed.shape} (Samples, Channels, Length)")

    X_train, X_val, y_train, y_val = train_test_split(X_processed, y_train_full, test_size=0.20, random_state=42)
    print(f"Training set: {X_train.shape[0]} samples | Validation set: {X_val.shape[0]} samples")

    # 3. Train the Enhanced Neural Network Model
    print("\n3. Training the RamanResNet model...")
    model = train_nn_model(X_train, y_train, X_val, y_val)

    # 4. Generate and Post-Process Predictions
    print("\n4. Generating and post-processing predictions...")
    predictions = make_nn_predictions(model, X_test_processed)
    final_predictions = post_process_predictions(predictions)

    # 5. Create Submission File
    print("\n5. Saving submission file...")
    try:
        sub = pd.read_csv(submission_file)
        sub['Glucose'] = final_predictions[:, 0]
        sub['Sodium Acetate'] = final_predictions[:, 1]
        sub['Magnesium Sulfate'] = final_predictions[:, 2]
        sub.to_csv('submission.csv', index=False)
        print("Submission file 'submission.csv' created successfully.")
    except FileNotFoundError:
        print(f"Warning: '{submission_file}' not found. Creating submission from scratch.")
        sub = pd.DataFrame({'id': X_test_raw['sample_id'].unique(), 'Glucose': final_predictions[:, 0], 'Sodium Acetate': final_predictions[:, 1], 'Magnesium Sulfate': final_predictions[:, 2]})
        sub.to_csv('submission.csv', index=False)
        print("Submission file 'submission.csv' created successfully.")

    print("\n" + "="*80)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()



