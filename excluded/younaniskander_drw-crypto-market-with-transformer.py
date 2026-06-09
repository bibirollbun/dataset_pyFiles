# DRW Crypto Market Prediction - Robust Transformer Implementation
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import gc

# Configuration
class Config:
    # Only use features that exist in the data
    CORE_FEATURES = ['bid_qty', 'ask_qty', 'buy_qty', 'sell_qty', 'volume']  # Basic features
    LAGS = [1, 5, 15, 60]  # 1min to 1hr
    SEQ_LEN = 60  # Lookback window
    BATCH_SIZE = 256
    EPOCHS = 15  # Reduced for faster iteration
    N_FOLDS = 3
    LEARNING_RATE = 3e-4
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CryptoTransformer(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 64)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=64, 
            nhead=8, 
            dim_feedforward=256, 
            dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.regressor = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)  # (seq_len, batch, features)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.regressor(x).squeeze()

class CryptoDataset(Dataset):
    def __init__(self, data, targets, sequence_length):
        self.data = data
        self.targets = targets
        self.seq_len = sequence_length
        
    def __len__(self):
        return len(self.data) - self.seq_len
        
    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_len]
        target = self.targets[idx+self.seq_len]
        return torch.FloatTensor(seq), torch.FloatTensor([target])

def get_available_features(df, requested_features):
    """Return only features that exist in the dataframe"""
    return [f for f in requested_features if f in df.columns]

def prepare_data():
    train = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/train.parquet")
    test = pd.read_parquet("/kaggle/input/drw-crypto-market-prediction/test.parquet")
    
    # Get available features
    available_features = get_available_features(train, Config.CORE_FEATURES)
    print(f"Available features: {available_features}")
    
    # Feature engineering
    for feat in available_features:
        for lag in Config.LAGS:
            train[f'{feat}_lag_{lag}'] = train[feat].shift(lag)
            test[f'{feat}_lag_{lag}'] = train[feat].iloc[-lag:].values[0]
    
    train['imbalance'] = (train['bid_qty'] - train['ask_qty']) / (train['bid_qty'] + train['ask_qty'] + 1e-6)
    test['imbalance'] = (test['bid_qty'] - test['ask_qty']) / (test['bid_qty'] + test['ask_qty'] + 1e-6)
    
    # Find all generated features
    features = available_features + [col for col in train.columns if 'lag_' in col] + ['imbalance']
    train = train.dropna()
    
    # Normalization
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[features])
    X_test = scaler.transform(test[features])
    y_train = train['label'].values
    
    return X_train, y_train, X_test, test.index, features

def main():
    X_train, y_train, X_test, test_ids, features = prepare_data()
    print(f"Final feature set ({len(features)}): {features}")
    
    kf = GroupKFold(n_splits=Config.N_FOLDS)
    groups = np.arange(len(X_train))
    test_preds = np.zeros(len(X_test))
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train, groups)):
        print(f"\nFold {fold+1} Training")
        
        # Create datasets
        train_dataset = CryptoDataset(X_train[train_idx], y_train[train_idx], Config.SEQ_LEN)
        val_dataset = CryptoDataset(X_train[val_idx], y_train[val_idx], Config.SEQ_LEN)
        
        # Initialize model
        model = CryptoTransformer(input_dim=len(features)).to(Config.DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
        criterion = nn.MSELoss()
        
        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE)
        
        # Training loop
        for epoch in range(Config.EPOCHS):
            model.train()
            train_loss = 0
            for seq, target in train_loader:
                seq, target = seq.to(Config.DEVICE), target.to(Config.DEVICE)
                optimizer.zero_grad()
                output = model(seq)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for seq, target in val_loader:
                    seq, target = seq.to(Config.DEVICE), target.to(Config.DEVICE)
                    output = model(seq)
                    val_loss += criterion(output, target).item()
            
            print(f"Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        
        # Prepare test set for prediction
        test_dataset = []
        for i in range(len(X_test) - Config.SEQ_LEN):
            test_dataset.append(X_test[i:i+Config.SEQ_LEN])
        test_dataset = torch.FloatTensor(np.array(test_dataset))
        
        # Predict
        model.eval()
        fold_preds = []
        with torch.no_grad():
            for i in range(0, len(test_dataset), Config.BATCH_SIZE):
                batch = test_dataset[i:i+Config.BATCH_SIZE].to(Config.DEVICE)
                fold_preds.extend(model(batch).cpu().numpy())
        
        # Pad with zeros for the initial SEQ_LEN positions
        padded_preds = np.zeros(len(X_test))
        padded_preds[Config.SEQ_LEN:] = np.array(fold_preds)
        test_preds += padded_preds / Config.N_FOLDS
    
    # Create submission
    submission = pd.DataFrame({
        'ID': test_ids,
        'prediction': test_preds
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission created successfully!")

if __name__ == "__main__":
    main()

