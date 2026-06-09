import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# Load data
df_train = pd.read_csv('/kaggle/input/playground-series-s3e19/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s3e19/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s3e19/sample_submission.csv')

# Data preprocessing
def prepare_data(df_train, df_test, sequence_length=30):
    # Convert date
    df_train['date'] = pd.to_datetime(df_train['date'])
    df_test['date'] = pd.to_datetime(df_test['date'])
    
    # Encode categorical variables
    le_country = LabelEncoder()
    le_store = LabelEncoder()
    le_product = LabelEncoder()
    
    # Fit on combined train and test data
    all_countries = pd.concat([df_train['country'], df_test['country']]).unique()
    all_stores = pd.concat([df_train['store'], df_test['store']]).unique()
    all_products = pd.concat([df_train['product'], df_test['product']]).unique()
    
    le_country.fit(all_countries)
    le_store.fit(all_stores)
    le_product.fit(all_products)
    
    df_train['country_encoded'] = le_country.transform(df_train['country'])
    df_train['store_encoded'] = le_store.transform(df_train['store'])
    df_train['product_encoded'] = le_product.transform(df_train['product'])
    
    df_test['country_encoded'] = le_country.transform(df_test['country'])
    df_test['store_encoded'] = le_store.transform(df_test['store'])
    df_test['product_encoded'] = le_product.transform(df_test['product'])
    
    # Create time features
    for df in [df_train, df_test]:
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['weekofyear'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
    
    # Normalize numerical features
    scaler = MinMaxScaler()
    num_features = ['year', 'month', 'day', 'dayofweek', 'dayofyear', 'weekofyear', 'quarter', 'is_weekend',
                   'country_encoded', 'store_encoded', 'product_encoded']
    
    df_train[num_features] = scaler.fit_transform(df_train[num_features])
    df_test[num_features] = scaler.transform(df_test[num_features])
    
    return df_train, df_test, scaler, le_country, le_store, le_product, num_features

# Create sequences for CNN-LSTM
def create_sequences(data, sequence_length=30):
    sequences = []
    targets = []
    
    unique_combinations = data.groupby(['country_encoded', 'store_encoded', 'product_encoded'])
    
    for (country, store, product), group in unique_combinations:
        group = group.sort_values('date')
        feature_values = group[num_features].values
        
        # We need to align features with target
        for i in range(len(group) - sequence_length):
            # Features for the sequence
            seq_features = feature_values[i:(i + sequence_length), :]
            
            # Target is the num_sold at the next time step
            target = group.iloc[i + sequence_length]['num_sold']
            
            sequences.append(seq_features)
            targets.append(target)
    
    return np.array(sequences), np.array(targets)

# Create test sequences
def create_test_sequences(data, sequence_length=30):
    sequences = []
    test_combinations = []
    
    unique_combinations = data.groupby(['country_encoded', 'store_encoded', 'product_encoded'])
    
    for (country, store, product), group in unique_combinations:
        group = group.sort_values('date')
        feature_values = group[num_features].values
        
        # For test data, use the last 'sequence_length' points for each combination
        if len(feature_values) >= sequence_length:
            seq = feature_values[-sequence_length:, :]
        else:
            # Pad with zeros if sequence is shorter
            padding = np.zeros((sequence_length - len(feature_values), feature_values.shape[1]))
            seq = np.vstack([padding, feature_values])
        
        sequences.append(seq)
        test_combinations.append((country, store, product))
    
    return np.array(sequences), test_combinations

# Fixed CNN-LSTM Model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_dim, sequence_length, hidden_dim=64, num_layers=2, output_dim=1, dropout_rate=0.2):
        super(CNNLSTMModel, self).__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # CNN layers for feature extraction from time series
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,  # Number of features as channels
            out_channels=64, 
            kernel_size=3, 
            padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64, 
            out_channels=128, 
            kernel_size=3, 
            padding=1
        )
        
        # Pooling layer
        self.pool = nn.AdaptiveAvgPool1d(1)  # Global average pooling
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=128,  # From CNN output
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_rate if num_layers > 1 else 0
        )
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32, output_dim)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        batch_size = x.size(0)
        
        # Permute for CNN: (batch_size, input_dim, sequence_length)
        x = x.permute(0, 2, 1)
        
        # CNN layers
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        
        # Global average pooling
        x = self.pool(x)  # (batch_size, 128, 1)
        
        # Reshape for LSTM: (batch_size, 1, 128)
        x = x.permute(0, 2, 1)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use the last hidden state
        x = hidden[-1]  # (batch_size, hidden_dim)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Alternative simpler CNN-LSTM model
class SimpleCNNLSTM(nn.Module):
    def __init__(self, input_dim, sequence_length, hidden_dim=64, output_dim=1, dropout_rate=0.2):
        super(SimpleCNNLSTM, self).__init__()
        
        # CNN part
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool1d(1)  # Global average pooling
        )
        
        # LSTM part
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, output_dim)
        )
    
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        
        # CNN processing
        x = x.permute(0, 2, 1)  # (batch_size, input_dim, sequence_length)
        x = self.conv_layers(x)  # (batch_size, 128, 1)
        
        # Prepare for LSTM
        x = x.permute(0, 2, 1)  # (batch_size, 1, 128)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Use last hidden state
        x = hidden[-1]  # (batch_size, hidden_dim)
        
        # Output
        x = self.output_layers(x)
        
        return x

# Custom Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = sequences
        self.targets = targets
        
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor([self.targets[idx]])

# Prepare data
print("Preparing data...")
df_train_processed, df_test_processed, scaler, le_country, le_store, le_product, num_features = prepare_data(df_train, df_test)

print(f"Number of features: {len(num_features)}")
print(f"Features: {num_features}")

# Create sequences
print("Creating sequences...")
sequence_length = 30
X_sequences, y_targets = create_sequences(df_train_processed, sequence_length)

print(f"Training sequences shape: {X_sequences.shape}")
print(f"Training targets shape: {y_targets.shape}")

# Split data
train_size = int(0.8 * len(X_sequences))
X_train, X_val = X_sequences[:train_size], X_sequences[train_size:]
y_train, y_val = y_targets[:train_size], y_targets[train_size:]

print(f"Train sequences: {X_train.shape}, Validation sequences: {X_val.shape}")

# Create datasets and dataloaders
train_dataset = TimeSeriesDataset(X_train, y_train)
val_dataset = TimeSeriesDataset(X_val, y_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# Model parameters
input_dim = X_train.shape[2]  # Number of features
sequence_len = X_train.shape[1]  # Sequence length
hidden_dim = 64
output_dim = 1

print(f"Input dimension: {input_dim}")
print(f"Sequence length: {sequence_len}")

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Use the simpler model to avoid initialization issues
model = SimpleCNNLSTM(
    input_dim=input_dim,
    sequence_length=sequence_len,
    hidden_dim=hidden_dim,
    output_dim=output_dim,
    dropout_rate=0.3
).to(device)

print(f"Model architecture:\n{model}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params:,}")

criterion = nn.L1Loss()  # MAE loss
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

# Training function
def train_model(model, train_loader, val_loader, epochs=30):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_cnn_lstm_model.pth')
        
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}')
    
    return train_losses, val_losses

# Train the model
print("Training CNN-LSTM model...")
train_losses, val_losses = train_model(model, train_loader, val_loader, epochs=30)

# Load best model
model.load_state_dict(torch.load('best_cnn_lstm_model.pth'))
print("Loaded best model for prediction")

# Prepare test data for prediction
print("Preparing test data for prediction...")
X_test_sequences, test_combinations = create_test_sequences(df_test_processed, sequence_length)

print(f"Test sequences shape: {X_test_sequences.shape}")
print(f"Number of test combinations: {len(test_combinations)}")

# Make predictions
def predict(model, test_sequences):
    model.eval()
    predictions = []
    
    test_tensor = torch.FloatTensor(test_sequences)
    test_loader = DataLoader(TensorDataset(test_tensor), batch_size=64, shuffle=False)
    
    with torch.no_grad():
        for batch_X in test_loader:
            batch_X = batch_X[0].to(device)
            outputs = model(batch_X)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions).flatten()

print("Making predictions...")
test_predictions = predict(model, X_test_sequences)

print(f"Number of test predictions: {len(test_predictions)}")

# Create submission
submission = df_test[['id']].copy()

# Map predictions to test samples based on combinations
prediction_dict = {}
for (country, store, product), pred in zip(test_combinations, test_predictions):
    prediction_dict[(country, store, product)] = pred

# Assign predictions
for idx, row in df_test_processed.iterrows():
    combo = (row['country_encoded'], row['store_encoded'], row['product_encoded'])
    if combo in prediction_dict:
        submission.loc[idx, 'num_sold'] = prediction_dict[combo]
    else:
        # Fallback: use mean of predictions or training mean
        submission.loc[idx, 'num_sold'] = np.mean(test_predictions) if len(test_predictions) > 0 else df_train['num_sold'].mean()

# Ensure non-negative predictions and reasonable values
submission['num_sold'] = np.maximum(0, submission['num_sold'])

# Clip extreme values based on training data distribution
train_mean = df_train['num_sold'].mean()
train_std = df_train['num_sold'].std()
upper_bound = train_mean + 3 * train_std
submission['num_sold'] = np.clip(submission['num_sold'], 0, upper_bound)

# Save submission
submission_file = 'submission.csv'
submission.to_csv(submission_file, index=False)
print(f"Submission saved to {submission_file}")

# Display results
print("\nPrediction Statistics:")
print(f"Min: {submission['num_sold'].min():.2f}")
print(f"Max: {submission['num_sold'].max():.2f}")
print(f"Mean: {submission['num_sold'].mean():.2f}")
print(f"Std: {submission['num_sold'].std():.2f}")

print("\nFirst 10 predictions:")
print(submission.head(10))

# Compare with training data statistics
print(f"\nTraining data statistics:")
print(f"Min: {df_train['num_sold'].min():.2f}")
print(f"Max: {df_train['num_sold'].max():.2f}")
print(f"Mean: {df_train['num_sold'].mean():.2f}")
print(f"Std: {df_train['num_sold'].std():.2f}")

# Plot training history
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('CNN-LSTM Training History')
plt.xlabel('Epochs')
plt.ylabel('MAE Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
plt.show()

