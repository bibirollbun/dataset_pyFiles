# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Machine learning libraries
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_log_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
import xgboost as xgb
import lightgbm as lgb

# Date parsing
from dateutil import parser as dt_parser
import scipy.stats

# Set display options
pd.set_option("display.max_rows", 5)
sns.set(rc = {'figure.figsize':(15,8)})

# Random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

print("Libraries imported successfully!")


import shutil

shutil.unpack_archive("../input/nyc-taxi-trip-duration/train.zip", "/kaggle/working/nyc-taxi-trip-duration")
shutil.unpack_archive("../input/nyc-taxi-trip-duration/test.zip", "/kaggle/working/nyc-taxi-trip-duration")
shutil.unpack_archive("../input/nyc-taxi-trip-duration/sample_submission.zip", "/kaggle/working/nyc-taxi-trip-duration")


import pandas as pd
from pathlib import Path

# Базовый каталог соревнования
DATA_PATH = Path("/kaggle/working/nyc-taxi-trip-duration/")

# Загружаем CSV-файлы
train_data = pd.read_csv(DATA_PATH / "train.csv")
test_data = pd.read_csv(DATA_PATH / "test.csv")
sample_submission = pd.read_csv(DATA_PATH / "sample_submission.csv")

# Базовая информация о данных
print(f"Training data shape : {train_data.shape}")
print(f"Test data shape     : {test_data.shape}")
print(f"Sample submission   : {sample_submission.shape}\n")

print("Training data info:")
train_data.info()

print("\nFirst 5 rows:")
display(train_data.head())  # display() удобен внутри Kaggle-ноутбука



# Load and examine the dataset
print("Loading data...")
train_data_full = pd.read_csv(DATA_PATH/"train.csv")
test_data_full = pd.read_csv(DATA_PATH/"test.csv") 
sample_submission = pd.read_csv(DATA_PATH/"sample_submission.csv")

# Limit to first 1000 elements for faster execution
print("Limiting to first 1000 elements for faster execution...")
train_data = train_data_full.sample(n=1000, random_state=92)
test_data = test_data_full.sample(n=1000, random_state=92)

print(f"Full training data shape: {train_data_full.shape}")
print(f"Limited training data shape: {train_data.shape}")
print(f"Full test data shape: {test_data_full.shape}")
print(f"Limited test data shape: {test_data.shape}")
print(f"Sample submission shape: {sample_submission.shape}")

print("\nTraining data info:")
train_data.info()

print("\nFirst 5 rows:")
train_data.head()



# Data visualization to illustrate the dataset
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. Trip duration distribution
axes[0,0].hist(train_data['trip_duration'], bins=50, alpha=0.7, color='skyblue')
axes[0,0].set_title('Розподіл тривалості поїздок', fontsize=14)
axes[0,0].set_xlabel('Тривалість (секунди)')
axes[0,0].set_ylabel('Частота')
axes[0,0].set_xlim(0, 5000)  # Focus on main distribution

# 2. Geographic distribution of pickup points
# Use all available data since we have only 1000 points
sample_data = train_data

axes[0,1].scatter(sample_data['pickup_longitude'], sample_data['pickup_latitude'], 
                 alpha=0.6, s=1, c=sample_data['trip_duration'], cmap='viridis')
axes[0,1].set_title('Географічний розподіл місць посадки', fontsize=14)
axes[0,1].set_xlabel('Довгота')
axes[0,1].set_ylabel('Широта')

# 3. Trip duration by hour
train_data['pickup_hour'] = pd.to_datetime(train_data['pickup_datetime']).dt.hour
hourly_duration = train_data.groupby('pickup_hour')['trip_duration'].mean()
axes[1,0].bar(hourly_duration.index, hourly_duration.values, color='orange', alpha=0.7)
axes[1,0].set_title('Середня тривалість поїздки за годинами', fontsize=14)
axes[1,0].set_xlabel('Година дня')
axes[1,0].set_ylabel('Середня тривалість (секунди)')

# 4. Distance calculation and distribution
train_data['distance'] = np.sqrt(
    (train_data['pickup_longitude'] - train_data['dropoff_longitude'])**2 + 
    (train_data['pickup_latitude'] - train_data['dropoff_latitude'])**2
)

axes[1,1].hist(train_data['distance'], bins=50, alpha=0.7, color='lightcoral')
axes[1,1].set_title('Розподіл відстаней поїздок', fontsize=14)
axes[1,1].set_xlabel('Відстань (градуси)')
axes[1,1].set_ylabel('Частота')
axes[1,1].set_xlim(0, 0.1)  # Focus on main distribution

plt.tight_layout()
plt.show()

print(f"Дані містять {len(train_data):,} записів про поїздки таксі")
print(f"Середня тривалість поїздки: {train_data['trip_duration'].mean():.0f} секунд")
print(f"Медіанна тривалість поїздки: {train_data['trip_duration'].median():.0f} секунд")



def feature_engineering(df):
    """Enhanced feature engineering function"""
    df = df.copy()
    
    # DateTime features
    pickup_datetime = pd.to_datetime(df['pickup_datetime'])
    df['pickup_year'] = pickup_datetime.dt.year
    df['pickup_month'] = pickup_datetime.dt.month
    df['pickup_day'] = pickup_datetime.dt.day
    df['pickup_hour'] = pickup_datetime.dt.hour
    df['pickup_minute'] = pickup_datetime.dt.minute
    df['pickup_weekday'] = pickup_datetime.dt.weekday
    df['pickup_yday'] = pickup_datetime.dt.dayofyear
    df['pickup_weekend'] = (pickup_datetime.dt.weekday >= 5).astype(int)
    
    # Time category features
    df['is_rush_hour'] = ((df['pickup_hour'] >= 7) & (df['pickup_hour'] <= 9) | 
                         (df['pickup_hour'] >= 17) & (df['pickup_hour'] <= 19)).astype(int)
    df['is_night'] = ((df['pickup_hour'] >= 22) | (df['pickup_hour'] <= 5)).astype(int)
    
    # Distance features
    df['distance_euclidean'] = np.sqrt(
        np.square(df['pickup_longitude'] - df['dropoff_longitude']) + 
        np.square(df['pickup_latitude'] - df['dropoff_latitude'])
    )
    
    df['distance_manhattan'] = (
        np.abs(df['pickup_longitude'] - df['dropoff_longitude']) + 
        np.abs(df['pickup_latitude'] - df['dropoff_latitude'])
    )
    
    # Direction
    df['direction'] = np.degrees(np.arctan2(
        (df['dropoff_latitude'] - df['pickup_latitude']),
        (df['dropoff_longitude'] - df['pickup_longitude'])
    ))
    
    # Center coordinates
    df['center_latitude'] = (df['pickup_latitude'] + df['dropoff_latitude']) / 2
    df['center_longitude'] = (df['pickup_longitude'] + df['dropoff_longitude']) / 2
    
    # Distance from Times Square (40.7580, -73.9855)
    times_square_lat, times_square_lon = 40.7580, -73.9855
    df['pickup_distance_from_center'] = np.sqrt(
        np.square(df['pickup_latitude'] - times_square_lat) + 
        np.square(df['pickup_longitude'] - times_square_lon)
    )
    df['dropoff_distance_from_center'] = np.sqrt(
        np.square(df['dropoff_latitude'] - times_square_lat) + 
        np.square(df['dropoff_longitude'] - times_square_lon)
    )
    
    # Binary features
    df['store_and_fwd_flag'] = (df['store_and_fwd_flag'] == 'Y').astype(int)
    
    return df

def clean_data(df):
    """Data cleaning function"""
    df = df.copy()
    
    print(f"Initial data shape: {df.shape}")
    
    # Remove extreme trip durations (keep 30 seconds to 6 hours)
    if 'trip_duration' in df.columns:
        df = df[(df['trip_duration'] > 30) & (df['trip_duration'] < 3600 * 6)]
        print(f"After trip duration filtering: {df.shape}")
    
    # Remove zero distance trips
    df = df[df['distance_euclidean'] > 0]
    print(f"After zero distance filtering: {df.shape}")
    
    # NYC geographic bounds
    nyc_bounds = {
        'min_lat': 40.5, 'max_lat': 41.0,
        'min_lon': -74.3, 'max_lon': -73.7
    }
    
    # Filter by geographic bounds
    df = df[
        (df['pickup_latitude'] >= nyc_bounds['min_lat']) & 
        (df['pickup_latitude'] <= nyc_bounds['max_lat']) &
        (df['pickup_longitude'] >= nyc_bounds['min_lon']) & 
        (df['pickup_longitude'] <= nyc_bounds['max_lon']) &
        (df['dropoff_latitude'] >= nyc_bounds['min_lat']) & 
        (df['dropoff_latitude'] <= nyc_bounds['max_lat']) &
        (df['dropoff_longitude'] >= nyc_bounds['min_lon']) & 
        (df['dropoff_longitude'] <= nyc_bounds['max_lon'])
    ]
    
    print(f"After geographic filtering: {df.shape}")
    
    return df

# Apply feature engineering and cleaning
print("Застосування інженерії ознак...")
train_enhanced = feature_engineering(train_data)
train_clean = clean_data(train_enhanced)

print(f"\nФінальна форма тренувальних даних: {train_clean.shape}")
print(f"Створені ознаки: {[col for col in train_clean.columns if col not in train_data.columns]}")



# Import PyTorch for neural networks
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau

print("PyTorch imported successfully!")
print(f"PyTorch version: {torch.__version__}")

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_STATE)



# Prepare features for modeling
def prepare_features(df):
    """Prepare features for modeling"""
    
    # Select feature columns (excluding non-predictive columns)
    exclude_cols = ['id', 'pickup_datetime', 'dropoff_datetime', 'trip_duration']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].copy()
    
    # Handle any missing values
    X = X.fillna(X.median())
    
    if 'trip_duration' in df.columns:
        y = df['trip_duration'].copy()
        return X, y
    else:
        return X

# Prepare training data
X, y = prepare_features(train_clean)

# Apply log transformation
y_log = np.log1p(y)  # log(1 + y) to handle zeros

# Add log-transformed distance features
X_processed = X.copy()
distance_cols = ['distance_euclidean', 'distance_manhattan']
for col in distance_cols:
    if col in X_processed.columns:
        X_processed[f'log_{col}'] = np.log1p(X_processed[col])

print(f"Feature matrix shape: {X_processed.shape}")
print(f"Target vector shape: {y_log.shape}")
print(f"Feature columns: {list(X_processed.columns)}")

# Train-validation split
X_train, X_val, y_train, y_val = train_test_split(
    X_processed, y_log, test_size=0.2, random_state=RANDOM_STATE
)

# Standardize features for neural networks
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print(f"\nTraining set: {X_train_scaled.shape}")
print(f"Validation set: {X_val_scaled.shape}")
print(f"Target statistics - Mean: {y_log.mean():.4f}, Std: {y_log.std():.4f}")



# Import MLP from sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, X_train, X_val, y_train, y_val, model_name):
    """Evaluate model performance"""
    # Predictions
    y_train_pred = model.predict(X_train)
    y_val_pred = model.predict(X_val)
    
    # Convert back from log space
    y_train_orig = np.expm1(y_train)
    y_val_orig = np.expm1(y_val)
    y_train_pred_orig = np.expm1(y_train_pred)
    y_val_pred_orig = np.expm1(y_val_pred)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    train_rmsle = np.sqrt(mean_squared_log_error(y_train_orig, y_train_pred_orig))
    val_rmsle = np.sqrt(mean_squared_log_error(y_val_orig, y_val_pred_orig))
    
    train_r2 = r2_score(y_train_orig, y_train_pred_orig)
    val_r2 = r2_score(y_val_orig, y_val_pred_orig)
    
    results = {
        'model': model_name,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_rmsle': train_rmsle,
        'val_rmsle': val_rmsle,
        'train_r2': train_r2,
        'val_r2': val_r2
    }
    
    return results

# Sklearn MLP experiments with different architectures
print("Експерименти з Sklearn MLP моделями...")
mlp_results = []

# MLP configurations to test
mlp_configs = [
    {'hidden_layer_sizes': (50,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 50},
    {'hidden_layer_sizes': (100,), 'activation': 'relu', 'solver': 'adam', 'max_iter': 50},
    {'hidden_layer_sizes': (100, 50), 'activation': 'relu', 'solver': 'adam', 'max_iter': 50},
    {'hidden_layer_sizes': (150, 100, 50), 'activation': 'relu', 'solver': 'adam', 'max_iter': 50},
    {'hidden_layer_sizes': (200, 100), 'activation': 'tanh', 'solver': 'adam', 'max_iter': 50},
    {'hidden_layer_sizes': (100, 50, 25), 'activation': 'relu', 'solver': 'lbfgs', 'max_iter': 50}
]

for i, config in enumerate(mlp_configs):
    layer_sizes = str(config['hidden_layer_sizes'])
    print(f"\\nТренування MLP {i+1}: {layer_sizes}")
    
    mlp = MLPRegressor(random_state=RANDOM_STATE, **config)
    mlp.fit(X_train_scaled, y_train)
    
    model_name = f'MLP_{i+1}_{layer_sizes}'
    results = evaluate_model(mlp, X_train_scaled, X_val_scaled, y_train, y_val, model_name)
    mlp_results.append(results)
    
    print(f"Val RMSLE: {results['val_rmsle']:.4f}, Val R²: {results['val_r2']:.4f}")

# Find best MLP model
best_mlp_idx = np.argmin([r['val_rmsle'] for r in mlp_results])
best_mlp_config = mlp_configs[best_mlp_idx]
print(f"\\nНайкраща MLP конфігурація: {best_mlp_config['hidden_layer_sizes']}")
print(f"Найкращий Val RMSLE: {mlp_results[best_mlp_idx]['val_rmsle']:.4f}")



class PyTorchRegressor(nn.Module):
    """PyTorch neural network for regression"""
    def __init__(self, input_dim, layers_config, activation='relu', dropout_rate=0.2):
        super(PyTorchRegressor, self).__init__()
        
        self.layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        self.batch_norm_layers = nn.ModuleList()
        
        # Input layer
        prev_dim = input_dim
        for units in layers_config:
            self.layers.append(nn.Linear(prev_dim, units))
            self.batch_norm_layers.append(nn.BatchNorm1d(units))
            self.dropout_layers.append(nn.Dropout(dropout_rate))
            prev_dim = units
            
        # Output layer
        self.output_layer = nn.Linear(prev_dim, 1)
        
        # Activation function
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'elu':
            self.activation = nn.ELU()
        else:
            self.activation = nn.ReLU()
    
    def forward(self, x):
        for i, (layer, bn, dropout) in enumerate(zip(self.layers, self.batch_norm_layers, self.dropout_layers)):
            x = layer(x)
            x = bn(x)
            x = self.activation(x)
            x = dropout(x)
        
        x = self.output_layer(x)
        return x

def train_pytorch_model(model, X_train, X_val, y_train, y_val, config):
    """Train PyTorch model"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Convert data to tensors
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.FloatTensor(y_train.values).to(device)
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.FloatTensor(y_val.values).to(device)
    
    # Create datasets and loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config['factor'], 
                                patience=config['patience']//2, min_lr=1e-7)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    epoch_losses = []
    
    for epoch in range(config['epochs']):
        # Training
        model.train()
        train_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X).squeeze()
            loss = F.mse_loss(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor).squeeze()
            val_loss = F.mse_loss(val_outputs, y_val_tensor).item()
        
        epoch_losses.append(val_loss)
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= config['patience']:
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    return len(epoch_losses)

def evaluate_pytorch_model(model, X_train, X_val, y_train, y_val, model_name):
    """Evaluate PyTorch model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    with torch.no_grad():
        # Predictions
        X_train_tensor = torch.FloatTensor(X_train).to(device)
        X_val_tensor = torch.FloatTensor(X_val).to(device)
        
        y_train_pred = model(X_train_tensor).squeeze().cpu().numpy()
        y_val_pred = model(X_val_tensor).squeeze().cpu().numpy()
    
    # Convert back from log space
    y_train_orig = np.expm1(y_train)
    y_val_orig = np.expm1(y_val)
    y_train_pred_orig = np.expm1(y_train_pred)
    y_val_pred_orig = np.expm1(y_val_pred)
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    
    train_rmsle = np.sqrt(mean_squared_log_error(y_train_orig, y_train_pred_orig))
    val_rmsle = np.sqrt(mean_squared_log_error(y_val_orig, y_val_pred_orig))
    
    train_r2 = r2_score(y_train_orig, y_train_pred_orig)
    val_r2 = r2_score(y_val_orig, y_val_pred_orig)
    
    results = {
        'model': model_name,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'train_rmsle': train_rmsle,
        'val_rmsle': val_rmsle,
        'train_r2': train_r2,
        'val_r2': val_r2
    }
    
    return results

# PyTorch model configurations for experiments
print("Експерименти з PyTorch нейронними мережами...")
pytorch_results = []

# Different network architectures and parameters to test
pytorch_configs = [
    {
        'layers': [64, 32], 
        'activation': 'relu', 
        'dropout': 0.2, 
        'batch_size': 32, 
        'epochs': 50,
        'patience': 5,
        'factor': 0.5
    },
    {
        'layers': [128, 64, 32], 
        'activation': 'relu', 
        'dropout': 0.3, 
        'batch_size': 64, 
        'epochs': 50,
        'patience': 7,
        'factor': 0.3
    },
    {
        'layers': [256, 128, 64], 
        'activation': 'tanh', 
        'dropout': 0.25, 
        'batch_size': 128, 
        'epochs': 50,
        'patience': 5,
        'factor': 0.2
    },
    {
        'layers': [512, 256, 128, 64], 
        'activation': 'relu', 
        'dropout': 0.4, 
        'batch_size': 64, 
        'epochs': 50,
        'patience': 10,
        'factor': 0.5
    },
    {
        'layers': [100, 50], 
        'activation': 'elu', 
        'dropout': 0.15, 
        'batch_size': 16, 
        'epochs': 50,
        'patience': 5,
        'factor': 0.7
    }
]

input_dim = X_train_scaled.shape[1]

for i, config in enumerate(pytorch_configs):
    print(f"\\nТренування PyTorch моделі {i+1}: {config['layers']}")
    print(f"Параметри: activation={config['activation']}, dropout={config['dropout']}, batch_size={config['batch_size']}")
    
    # Create model
    model = PyTorchRegressor(
        input_dim=input_dim,
        layers_config=config['layers'],
        activation=config['activation'],
        dropout_rate=config['dropout']
    )
    
    # Train model
    epochs_trained = train_pytorch_model(model, X_train_scaled, X_val_scaled, y_train, y_val, config)
    
    # Evaluate model
    model_name = f"PyTorch_{i+1}_{config['layers']}"
    results = evaluate_pytorch_model(model, X_train_scaled, X_val_scaled, y_train, y_val, model_name)
    pytorch_results.append(results)
    
    print(f"Val RMSLE: {results['val_rmsle']:.4f}, Val R²: {results['val_r2']:.4f}")
    print(f"Епох навчання: {epochs_trained}")

# Find best PyTorch model
best_pytorch_idx = np.argmin([r['val_rmsle'] for r in pytorch_results])
best_pytorch_config = pytorch_configs[best_pytorch_idx]
print(f"\\nНайкраща PyTorch конфігурація: {best_pytorch_config['layers']}")
print(f"Найкращий Val RMSLE: {pytorch_results[best_pytorch_idx]['val_rmsle']:.4f}")



# Combine all results for comparison
all_results = mlp_results + pytorch_results
results_df = pd.DataFrame(all_results)
results_df = results_df.sort_values('val_rmsle')

print("Порівняння усіх моделей (відсортовані за Val RMSLE):")
print("=" * 80)
print(f"{'Model':<25} {'Val RMSLE':<12} {'Val R²':<12} {'Train RMSLE':<12} {'Overfitting':<12}")
print("=" * 80)

for _, row in results_df.iterrows():
    overfitting = row['train_rmsle'] - row['val_rmsle']
    print(f"{row['model']:<25} {row['val_rmsle']:<12.4f} {row['val_r2']:<12.4f} {row['train_rmsle']:<12.4f} {overfitting:<12.4f}")

# Determine optimal model
print(f"\\n{'='*50}")
print("РЕЗУЛЬТАТИ ЕКСПЕРИМЕНТІВ:")
print(f"{'='*50}")

best_overall = results_df.iloc[0]
print(f"Найкраща загальна модель: {best_overall['model']}")
print(f"Найкращий Val RMSLE: {best_overall['val_rmsle']:.4f}")
print(f"Найкращий Val R²: {best_overall['val_r2']:.4f}")

# Compare MLP vs PyTorch
best_mlp = min(mlp_results, key=lambda x: x['val_rmsle'])
best_pytorch = min(pytorch_results, key=lambda x: x['val_rmsle'])

print(f"\\nПорівняння кращих моделей:")
print(f"Найкраща MLP (sklearn): {best_mlp['model']} - RMSLE: {best_mlp['val_rmsle']:.4f}")
print(f"Найкраща PyTorch: {best_pytorch['model']} - RMSLE: {best_pytorch['val_rmsle']:.4f}")

if best_pytorch['val_rmsle'] < best_mlp['val_rmsle']:
    improvement = ((best_mlp['val_rmsle'] - best_pytorch['val_rmsle']) / best_mlp['val_rmsle']) * 100
    print(f"\\n✅ PyTorch модель перевищила sklearn MLP на {improvement:.2f}%")
else:
    difference = ((best_pytorch['val_rmsle'] - best_mlp['val_rmsle']) / best_mlp['val_rmsle']) * 100
    print(f"\\n❌ PyTorch модель програла sklearn MLP на {difference:.2f}%")



# Visualization of model comparison
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. RMSLE comparison
model_names = [r['model'] for r in all_results]
val_rmsle = [r['val_rmsle'] for r in all_results]

axes[0,0].bar(range(len(model_names)), val_rmsle)
axes[0,0].set_xticks(range(len(model_names)))
axes[0,0].set_xticklabels(model_names, rotation=45, ha='right')
axes[0,0].set_title('Порівняння моделей за Val RMSLE', fontsize=14)
axes[0,0].set_ylabel('RMSLE')

# 2. R² comparison
val_r2 = [r['val_r2'] for r in all_results]
axes[0,1].bar(range(len(model_names)), val_r2, color='orange')
axes[0,1].set_xticks(range(len(model_names)))
axes[0,1].set_xticklabels(model_names, rotation=45, ha='right')
axes[0,1].set_title('Порівняння моделей за Val R²', fontsize=14)
axes[0,1].set_ylabel('R²')

# 3. MLP vs PyTorch comparison
mlp_rmsle = [r['val_rmsle'] for r in mlp_results]
pytorch_rmsle = [r['val_rmsle'] for r in pytorch_results]

x_pos = [0, 1]
avg_rmsle = [np.mean(mlp_rmsle), np.mean(pytorch_rmsle)]
std_rmsle = [np.std(mlp_rmsle), np.std(pytorch_rmsle)]

axes[1,0].bar(x_pos, avg_rmsle, yerr=std_rmsle, capsize=5, 
              color=['lightblue', 'lightcoral'], alpha=0.7)
axes[1,0].set_xticks(x_pos)
axes[1,0].set_xticklabels(['Sklearn MLP', 'PyTorch'])
axes[1,0].set_title('Середня продуктивність: MLP vs PyTorch', fontsize=14)
axes[1,0].set_ylabel('Середня Val RMSLE')

# 4. Overfitting analysis
overfitting = [r['train_rmsle'] - r['val_rmsle'] for r in all_results]
colors = ['red' if x > 0.01 else 'green' for x in overfitting]

axes[1,1].bar(range(len(model_names)), overfitting, color=colors, alpha=0.7)
axes[1,1].set_xticks(range(len(model_names)))
axes[1,1].set_xticklabels(model_names, rotation=45, ha='right')
axes[1,1].set_title('Аналіз перенавчання (Train RMSLE - Val RMSLE)', fontsize=14)
axes[1,1].set_ylabel('Різниця RMSLE')
axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()

# Summary statistics
print("\\nСтатистика експериментів:")
print(f"Всього протестовано моделей: {len(all_results)}")
print(f"MLP моделей: {len(mlp_results)}")
print(f"PyTorch моделей: {len(pytorch_results)}")
print(f"\\nДіапазон Val RMSLE: {min(val_rmsle):.4f} - {max(val_rmsle):.4f}")
print(f"Середня Val RMSLE: {np.mean(val_rmsle):.4f} ± {np.std(val_rmsle):.4f}")


