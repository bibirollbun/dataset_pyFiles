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


# Kaggle Playground Series S5E10 - Road Accident Risk Prediction
# Transformer-Based Solution using FT-Transformer
# No API keys required - fully standalone implementation

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ============================================================================
# FT-Transformer Implementation (Feature Tokenizer Transformer)
# ============================================================================

class NumericalFeatureTokenizer(nn.Module):
    """Converts numerical features into embeddings"""
    def __init__(self, num_features, d_token, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(num_features, d_token))
        self.bias = nn.Parameter(torch.Tensor(num_features, d_token)) if bias else None
        nn.init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x):
        # x: [batch_size, num_features]
        # output: [batch_size, num_features, d_token]
        x = x.unsqueeze(-1)  # [batch_size, num_features, 1]
        x = x * self.weight.unsqueeze(0)  # [batch_size, num_features, d_token]
        if self.bias is not None:
            x = x + self.bias.unsqueeze(0)
        return x

class CategoricalFeatureTokenizer(nn.Module):
    """Converts categorical features into embeddings"""
    def __init__(self, categories, d_token):
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_categories, d_token)
            for num_categories in categories
        ])
    
    def forward(self, x):
        # x: [batch_size, num_categorical_features]
        # output: [batch_size, num_categorical_features, d_token]
        return torch.stack([
            embedding(x[:, i])
            for i, embedding in enumerate(self.embeddings)
        ], dim=1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_token, n_heads, dropout=0.1):
        super().__init__()
        assert d_token % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_token // n_heads
        
        self.qkv = nn.Linear(d_token, 3 * d_token)
        self.out = nn.Linear(d_token, d_token)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        batch_size, seq_len, d_token = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.n_heads, self.d_head)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, d_head]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, d_token)
        
        return self.out(attn_output)

class TransformerLayer(nn.Module):
    def __init__(self, d_token, n_heads, d_ffn, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_token, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_token)
        self.norm2 = nn.LayerNorm(d_token)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_token),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Multi-head attention with residual
        x = x + self.attention(self.norm1(x))
        # Feed-forward network with residual
        x = x + self.ffn(self.norm2(x))
        return x

class FTTransformer(nn.Module):
    """
    Feature Tokenizer Transformer for tabular data
    Paper: "Revisiting Deep Learning Models for Tabular Data" (Yandex, 2021)
    """
    def __init__(self, 
                 num_continuous, 
                 categories,
                 d_token=32,
                 n_layers=6,
                 n_heads=8,
                 d_ffn=128,
                 dropout=0.1,
                 output_dim=1):
        super().__init__()
        
        # Feature tokenizers
        self.numerical_tokenizer = NumericalFeatureTokenizer(num_continuous, d_token)
        if categories:
            self.categorical_tokenizer = CategoricalFeatureTokenizer(categories, d_token)
        else:
            self.categorical_tokenizer = None
        
        # CLS token for aggregation
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Transformer layers
        self.transformer_layers = nn.ModuleList([
            TransformerLayer(d_token, n_heads, d_ffn, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_token)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(d_token, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, output_dim)
        )
        
    def forward(self, x_cont, x_cat=None):
        # Tokenize features
        x = self.numerical_tokenizer(x_cont)  # [batch, num_cont, d_token]
        
        if x_cat is not None and self.categorical_tokenizer is not None:
            cat_embeddings = self.categorical_tokenizer(x_cat)  # [batch, num_cat, d_token]
            x = torch.cat([x, cat_embeddings], dim=1)
        
        # Add CLS token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # Apply transformer layers
        for layer in self.transformer_layers:
            x = layer(x)
        
        x = self.norm(x)
        
        # Use CLS token for prediction
        cls_output = x[:, 0]
        output = self.head(cls_output)
        
        return output.squeeze(-1)

# ============================================================================
# Dataset and Training Functions
# ============================================================================

class TabularDataset(Dataset):
    def __init__(self, x_cont, x_cat, y=None):
        self.x_cont = torch.FloatTensor(x_cont)
        self.x_cat = torch.LongTensor(x_cat) if x_cat is not None else None
        self.y = torch.FloatTensor(y) if y is not None else None
        
    def __len__(self):
        return len(self.x_cont)
    
    def __getitem__(self, idx):
        if self.y is not None:
            if self.x_cat is not None:
                return self.x_cont[idx], self.x_cat[idx], self.y[idx]
            else:
                return self.x_cont[idx], None, self.y[idx]
        else:
            if self.x_cat is not None:
                return self.x_cont[idx], self.x_cat[idx]
            else:
                return self.x_cont[idx], None

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in loader:
        x_cont, x_cat, y = batch
        x_cont = x_cont.to(device)
        x_cat = x_cat.to(device) if x_cat is not None else None
        y = y.to(device)
        
        optimizer.zero_grad()
        output = model(x_cont, x_cat)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)

def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
            x_cont, x_cat, y = batch
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device) if x_cat is not None else None
            y = y.to(device)
            
            output = model(x_cont, x_cat)
            loss = criterion(output, y)
            
            total_loss += loss.item()
            predictions.extend(output.cpu().numpy())
    
    return total_loss / len(loader), np.array(predictions)

def predict(model, loader, device):
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                x_cont, x_cat = batch
            else:
                x_cont, x_cat, _ = batch
            
            x_cont = x_cont.to(device)
            x_cat = x_cat.to(device) if x_cat is not None else None
            
            output = model(x_cont, x_cat)
            predictions.extend(output.cpu().numpy())
    
    return np.array(predictions)

# ============================================================================
# Main Training Pipeline
# ============================================================================

def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
    
    # Separate target
    target_col = 'accident_risk'  # Adjust based on actual column name
    X = train_df.drop(['id', target_col], axis=1)
    y = train_df[target_col].values
    X_test = test_df.drop(['id'], axis=1)
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    print(f"Numerical features: {len(numerical_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    
    # Encode categorical features
    label_encoders = {}
    categories = []
    
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        X_test[col] = le.transform(X_test[col].astype(str))
        label_encoders[col] = le
        categories.append(len(le.classes_))
    
    # Scale numerical features
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])
    
    # Prepare arrays
    x_cont = X[numerical_cols].values
    x_cat = X[categorical_cols].values if categorical_cols else None
    x_test_cont = X_test[numerical_cols].values
    x_test_cat = X_test[categorical_cols].values if categorical_cols else None
    
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    config = {
        'd_token': 32,
        'n_layers': 6,
        'n_heads': 8,
        'd_ffn': 128,
        'dropout': 0.15,
        'lr': 1e-3,
        'batch_size': 256,
        'epochs': 25,
        'n_folds': 5,
        'early_stopping_patience': 15
    }
    
    # K-Fold Cross Validation
    kfold = KFold(n_splits=config['n_folds'], shuffle=True, random_state=42)
    oof_predictions = np.zeros(len(X))
    test_predictions = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X)):
        print(f"\n{'='*50}")
        print(f"Fold {fold + 1}/{config['n_folds']}")
        print('='*50)
        
        # Split data
        x_train_cont, x_val_cont = x_cont[train_idx], x_cont[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        if x_cat is not None:
            x_train_cat, x_val_cat = x_cat[train_idx], x_cat[val_idx]
        else:
            x_train_cat, x_val_cat = None, None
        
        # Create datasets
        train_dataset = TabularDataset(x_train_cont, x_train_cat, y_train)
        val_dataset = TabularDataset(x_val_cont, x_val_cat, y_val)
        
        train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
        
        # Initialize model
        model = FTTransformer(
            num_continuous=len(numerical_cols),
            categories=categories if categories else None,
            d_token=config['d_token'],
            n_layers=config['n_layers'],
            n_heads=config['n_heads'],
            d_ffn=config['d_ffn'],
            dropout=config['dropout']
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=1e-5)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, verbose=True
        )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(config['epochs']):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_preds = validate(model, val_loader, criterion, device)
            
            scheduler.step(val_loss)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{config['epochs']} - Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f}")
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save OOF predictions
                oof_predictions[val_idx] = val_preds
                # Save best model state
                best_model_state = model.state_dict()
            else:
                patience_counter += 1
                if patience_counter >= config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        print(f"Fold {fold+1} Best Val Loss (RMSE): {np.sqrt(best_val_loss):.6f}")
        
        # Load best model and predict on test
        model.load_state_dict(best_model_state)
        test_dataset = TabularDataset(x_test_cont, x_test_cat)
        test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)
        fold_test_preds = predict(model, test_loader, device)
        test_predictions.append(fold_test_preds)
    
    # Calculate OOF score
    oof_rmse = np.sqrt(np.mean((oof_predictions - y) ** 2))
    print(f"Overall OOF RMSE: {oof_rmse:.6f}")
    
    # Average test predictions
    final_test_preds = np.mean(test_predictions, axis=0)
    
    # Create submission
    submission = pd.DataFrame({
        'id': test_df['id'],
        target_col: final_test_preds
    })
    submission.to_csv('submission.csv', index=False)
    print("\nSubmission file created: submission.csv")
    
    return oof_rmse




if __name__ == "__main__":
    main()




