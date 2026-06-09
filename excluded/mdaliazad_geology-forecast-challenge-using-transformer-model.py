import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
import warnings
warnings.filterwarnings('ignore')


# Set random seed for reproducibility
def seed_everything(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything()


# Load data
train = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/train.csv").fillna(0)
test = pd.read_csv("/kaggle/input/geology-forecast-challenge-open/data/test.csv").fillna(0)
sub = pd.read_csv('/kaggle/input/geology-forecast-challenge-open/data/sample_submission.csv')



test.head()


# Define features and targets
FEATURES = [c for c in test.columns if c != 'geology_id']
TARGETS = [c for c in sub.columns if c != 'geology_id']


# Enhanced Transformer-based Model
class GeologyTransformer(nn.Module):
    def __init__(self, input_dim=300, output_dim=3000, num_heads=8, num_layers=4, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 256)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, output_dim)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        x = self.decoder(x)
        return x



class EnhancedGeologyTransformer(nn.Module):
    def __init__(self, input_dim=300, output_dim=3000, num_heads=8, num_layers=6, dropout=0.2):
        super().__init__()
        
        # Improved embedding with residual connections
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256)
        )
        
        # Transformer with relative positional encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, 
            nhead=num_heads,
            dim_feedforward=1024,  # Increased FFN dimension
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Multi-scale prediction heads
        self.long_range_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 1500)  # Predicts first half
        )
        self.short_range_head = nn.Sequential(
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Linear(512, 1500)  # Predicts second half
        )
        
        # Attention-based fusion
        self.fusion = nn.MultiheadAttention(256, num_heads=4, dropout=dropout)
        self.output_proj = nn.Linear(256, output_dim)
        
    def forward(self, x):
        # Enhanced embedding
        x = self.embedding(x)
        
        # Transformer processing
        x = self.transformer(x)
        
        # Multi-scale predictions
        long_range = self.long_range_head(x)
        short_range = self.short_range_head(x)
        
        # Attention-based fusion
        fused, _ = self.fusion(x, x, x)
        fused = self.output_proj(fused)
        
        # Combine predictions
        return torch.cat([long_range, short_range], dim=1) + fused


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGeologyPredictor(nn.Module):
    def __init__(self, input_size=300, output_size=3000):
        super().__init__()
        
        # Feature extraction
        self.feature_net = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Main prediction network
        self.predictor = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, output_size)
        )
        
        # Uncertainty estimation head
        self.uncertainty = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_size)
        )
        
    def forward(self, x):
        features = self.feature_net(x)
        
        # Base prediction
        pred = self.predictor(features)
        
        # Uncertainty weights
        weights = torch.sigmoid(self.uncertainty(features))
        
        # Combine base prediction with 10% variation for realizations
        base = pred[:, :300]  # First realization
        variations = torch.randn(9, *base.shape, device=x.device) * weights[:, :300].abs() * 0.1
        realizations = base.unsqueeze(0) + variations
        realizations = torch.cat([base.unsqueeze(0), realizations], dim=0)
        
        return realizations.permute(1, 0, 2).reshape(x.shape[0], -1)  # Flatten to [batch, 3000]



# Fixed Custom loss function
class GeologicalNLLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Precompute competition weights
        self.weights = torch.zeros(300)
        log_slopes = [1.0406028049510443, 0.0, 7.835345062351012]
        log_offsets = [-6.4306028049510443, -2.1617411566043896, -45.24876794412965]
        
        # Region 1 (positions 1-60)
        x = torch.arange(1, 61)
        self.weights[:60] = torch.exp(log_slopes[0] * torch.log(x) + log_offsets[0])
        
        # Region 2 (positions 61-244)
        self.weights[60:244] = torch.exp(torch.tensor(log_offsets[1]))
        
        # Region 3 (positions 245-300)
        x = torch.arange(245, 301)
        self.weights[244:] = torch.exp(log_slopes[2] * torch.log(x) + log_offsets[2])
        
        self.weights = 1. / (self.weights * 6000)
        
    def forward(self, preds, targets):
        batch_size = preds.shape[0]
        
        # Reshape predictions and targets to [batch_size, 10, 300]
        preds = preds.view(batch_size, 10, 300)
        targets = targets.view(batch_size, 10, 300)
        
        # Compute weighted MSE
        diff = preds - targets
        weighted_diff = diff * self.weights.to(preds.device)
        loss_per_realization = torch.sum(weighted_diff ** 2, dim=2)
        
        # Compute NLL
        exp_loss = torch.exp(-0.5 * loss_per_realization)
        mean_exp_loss = torch.mean(exp_loss, dim=1)
        nll = -torch.log(mean_exp_loss + 1e-8)
        
        return torch.mean(nll)

# Data preparation
def prepare_data(train, test, FEATURES, TARGETS):
    # Feature engineering
    X_train = np.log1p(train[FEATURES].values + 30)
    X_test = np.log1p(test[FEATURES].values + 30)
    y_train = train[TARGETS].values
    
    # Scaling
    x_scaler = StandardScaler()
    X_train = x_scaler.fit_transform(X_train)
    X_test = x_scaler.transform(X_test)
    
    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train)
    
    return X_train, X_test, y_train, x_scaler, y_scaler


# Training function
def train_model(model, train_loader, valid_loader, optimizer, loss_fn, device, epochs=50):
    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for X, y in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        valid_loss = 0
        with torch.no_grad():
            for X, y in valid_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                valid_loss += loss_fn(outputs, y).item()
        
        train_loss /= len(train_loader)
        valid_loss /= len(valid_loader)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Valid Loss = {valid_loss:.4f}")
        
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return best_loss


# Main training pipeline
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    X_train, X_test, y_train, x_scaler, y_scaler = prepare_data(train, test, FEATURES, TARGETS)
    
    # KFold training
    folds = 5
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)
    test_preds = np.zeros((folds, len(test), len(TARGETS)))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold + 1}/{folds}")
        
        # Split data
        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]
        
        # Create datasets
        train_dataset = TensorDataset(
            torch.tensor(X_train_fold, dtype=torch.float32),
            torch.tensor(y_train_fold, dtype=torch.float32)
        )
        val_dataset = TensorDataset(
            torch.tensor(X_val_fold, dtype=torch.float32),
            torch.tensor(y_val_fold, dtype=torch.float32)
        )
        test_dataset = TensorDataset(
            torch.tensor(X_test, dtype=torch.float32)
        )
        
        # Create dataloaders
        batch_size = 256 if torch.cuda.is_available() else 64
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=batch_size*2, shuffle=False)
        
        # Initialize model
        model = SimpleGeologyPredictor().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
        loss_fn = GeologicalNLLLoss().to(device)
        
        # Train
        best_loss = train_model(model, train_loader, val_loader, optimizer, loss_fn, device)
        print(f"Fold {fold + 1} best validation loss: {best_loss:.4f}")
        
        # Load best model and predict
        model.load_state_dict(torch.load('best_model.pth'))
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for X in test_loader:
                X = X[0].to(device)
                outputs = model(X)
                fold_preds.append(outputs.cpu().numpy())
        
        test_preds[fold] = np.concatenate(fold_preds)
    
    # Average predictions across folds and inverse transform
    final_preds = test_preds.mean(axis=0)
    final_preds = y_scaler.inverse_transform(final_preds)
    
    # Create submission
    sub[TARGETS] = final_preds
    sub.to_csv("submission-5.csv", index=False)
    print("Submission saved!")


if __name__ == "__main__":
    main()

