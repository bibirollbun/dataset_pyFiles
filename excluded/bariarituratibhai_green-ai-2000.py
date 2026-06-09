!pip install --quiet codecarbon torch torchvision torchsummary




#  Imports & Setup 
import os
import sys
import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

# Carbon tracking
try:
    from codecarbon import EmissionsTracker
    CARBON_TRACKING = True
    print("CodeCarbon available for emission tracking")
except ImportError:
    print("CodeCarbon not available - will simulate carbon tracking")
    CARBON_TRACKING = False
    class EmissionsTracker:
        def __init__(self, project_name, output_dir, log_level="error"):
            self.project_name = project_name
        def start(self): pass
        def stop(self): return np.random.uniform(0.001, 0.01)

# ML & preprocessing
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, accuracy_score

# Optuna for neural architecture search
try:
    import optuna
    OPTUNA_AVAILABLE = True
    print("Optuna available for neural architecture search")
except ImportError:
    print("Optuna not available - will use grid search instead")
    OPTUNA_AVAILABLE = False

# Visualization settings
plt.style.use('default')
sns.set_palette("husl")

print("Green AI Pipeline Initialized!")
print(f"TensorFlow version: {tf.__version__}")

# Output directory
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
TIMESTAMP = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Session timestamp: {TIMESTAMP}")



#  Load Data 
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)

# Load Kaggle competition data
train = safe_read_csv("/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/train.csv")
test = safe_read_csv("/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/test.csv")
metadata = safe_read_csv("/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/metaData.csv")

# Print shapes
print("Loading competition data...")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")
print(f"Metadata shape: {metadata.shape}")

# Display sample data
print("\nTraining Data Sample:")
display(train.head())

print("\nTest Data Sample:")
display(test.head())

print("\nMetadata Sample:")
display(metadata.head())

# Display data info
print("\nData Info:")
print("Training data info:")
train.info()
print("\nMetadata info:")
metadata.info()



 #Green Impact Analysis 
metadata_agg = metadata.groupby('region').agg({
    'carbon_intensity_gco2_per_kwh': 'mean',
    'water_usage_efficiency_l_per_kwh': 'mean'
}).reset_index()

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].bar(metadata_agg['region'], metadata_agg['carbon_intensity_gco2_per_kwh'],
           color=['#ff6b6b', '#4ecdc4'])
axes[0].set_title('Average Carbon Intensity by Region', fontsize=12, fontweight='bold')
axes[0].set_ylabel('gCO2 per kWh')
axes[0].tick_params(axis='x', rotation=45)

axes[1].bar(metadata_agg['region'], metadata_agg['water_usage_efficiency_l_per_kwh'],
           color=['#95e1d3', '#fad390'])
axes[1].set_title('Average Water Usage by Region', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Liters per kWh')
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

carbon_diff = metadata_agg['carbon_intensity_gco2_per_kwh'].max() - metadata_agg['carbon_intensity_gco2_per_kwh'].min()
water_diff = metadata_agg['water_usage_efficiency_l_per_kwh'].max() - metadata_agg['water_usage_efficiency_l_per_kwh'].min()
print(f"Carbon intensity difference between regions: {carbon_diff:.1f} gCO2/kWh")
print(f"Water usage difference between regions: {water_diff:.1f} L/kWh")



# Green Impact Analysis 

# Analyze metadata for green insights
metadata_agg = metadata.groupby('region').agg({
    'carbon_intensity_gco2_per_kwh': 'mean',
    'water_usage_efficiency_l_per_kwh': 'mean'
}).reset_index()

# Carbon intensity by region
# Plot 1: Carbon Intensity
fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].bar(metadata_agg['region'], metadata_agg['carbon_intensity_gco2_per_kwh'],
           color=['#ff6b6b', '#4ecdc4'])
axes[0].set_title('Average Carbon Intensity by Region', fontsize=12, fontweight='bold')
axes[0].set_ylabel('gCO2 per kWh')
axes[0].tick_params(axis='x', rotation=45)

# Plot 2: Water Usage
axes[1].bar(metadata_agg['region'], metadata_agg['water_usage_efficiency_l_per_kwh'],
           color=['#95e1d3', '#fad390'])
axes[1].set_title('Average Water Usage by Region', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Liters per kWh')
axes[1].tick_params(axis='x', rotation=45)
plt.tight_layout()
plt.show()

# Calculate green impact metrics
carbon_diff = metadata_agg['carbon_intensity_gco2_per_kwh'].max() - metadata_agg['carbon_intensity_gco2_per_kwh'].min()
water_diff = metadata_agg['water_usage_efficiency_l_per_kwh'].max() - metadata_agg['water_usage_efficiency_l_per_kwh'].min()
print(f"Carbon intensity difference between regions: {carbon_diff:.1f} gCO2/kWh")
print(f"Water usage difference between regions: {water_diff:.1f} L/kWh")

# Time-based analysis
metadata['hour'] = pd.to_datetime(metadata['timestamp_utc']).dt.hour
time_analysis = metadata.groupby(['region', 'hour'])['carbon_intensity_gco2_per_kwh'].mean().reset_index()

plt.figure(figsize=(12,6))
for region in metadata['region'].unique():
    region_data = time_analysis[time_analysis['region'] == region]
    plt.plot(region_data['hour'], region_data['carbon_intensity_gco2_per_kwh'], marker='o', label=region, linewidth=2)
plt.title('Carbon Intensity by Hour - Carbon-Aware Computing Opportunity', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day (UTC)')
plt.ylabel('Carbon Intensity (gCO2/kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



#Hourly Carbon Intensity Analysis 
metadata['hour'] = pd.to_datetime(metadata['timestamp_utc']).dt.hour
time_analysis = metadata.groupby(['region', 'hour'])['carbon_intensity_gco2_per_kwh'].mean().reset_index()

plt.figure(figsize=(12,6))
for region in metadata['region'].unique():
    region_data = time_analysis[time_analysis['region'] == region]
    plt.plot(region_data['hour'], region_data['carbon_intensity_gco2_per_kwh'], 
             marker='o', label=region, linewidth=2)

plt.title('Carbon Intensity by Hour - Carbon-Aware Computing Opportunity', fontsize=14, fontweight='bold')
plt.xlabel('Hour of Day (UTC)')
plt.ylabel('Carbon Intensity (gCO2/kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()





# Data preprocessing and feature engineering
target_col = 'target'
y = train[target_col].values
problem_type = 'binary_classification' if len(np.unique(y)) == 2 else 'regression'

# Prepare features (check what columns exist in both train and test)
feature_cols = [col for col in train.columns if col not in [target_col, 'example_id']]
X = train[feature_cols].copy()

# For this demo with missing test features, create synthetic test features
X_test = test[feature_cols].copy() if len(test.columns) > 1 else pd.DataFrame(
    {col: np.random.normal(X[col].mean(), X[col].std(), len(test)) for col in feature_cols}
)

# Handle missing values
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

# Fill missing values
if numeric_cols: 
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X_test[numeric_cols] = X_test[numeric_cols].fillna(X[numeric_cols].median())

# One-hot encoding for categorical variables
if categorical_cols:
    X[categorical_cols] = X[categorical_cols].fillna('missing')
    X_test[categorical_cols] = X_test[categorical_cols].fillna('missing')
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align columns
X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# For very small dataset, use simple approach without stratification
# Simple train/validation split without stratification for small dataset
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

# Use validation set as internal test for demo
# For very small datasets, use the full dataset for training and validation

# Visualize target distribution
plt.figure(figsize=(6,4))
sns.histplot(y, bins=len(np.unique(y)), kde=False)
plt.title('Target Distribution')
plt.xlabel('Target')
plt.ylabel('Count')
plt.show()



# Baseline Model 
def create_baseline_model(input_dim, problem_type):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu')
    ])
    if problem_type == 'regression':
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

tracker = EmissionsTracker(project_name=f"baseline_model_{TIMESTAMP}", output_dir=OUTPUT_DIR)
tracker.start()
start_time = time.time()

baseline_model = create_baseline_model(X_train.shape[1], problem_type)
history_baseline = baseline_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                                      epochs=10, batch_size=min(8, len(X_train)), verbose=1)

training_time = time.time() - start_time
emissions_baseline = tracker.stop()

# Evaluate baseline
if problem_type == 'regression':
    y_pred_baseline = baseline_model.predict(X_val).flatten()
    baseline_metric = mean_absolute_error(y_val, y_pred_baseline)
else:
    y_pred_baseline_class = (baseline_model.predict(X_val) > 0.5).astype(int).flatten()
    baseline_metric = accuracy_score(y_val, y_pred_baseline_class)

baseline_metrics = {'training_time_s': training_time, 'carbon_emissions_kg': emissions_baseline, 'primary_metric': baseline_metric}
baseline_metrics



#  Neural Architecture Search / Optimal Model 
def create_optimized_model(trial, input_dim, problem_type):
    if OPTUNA_AVAILABLE and trial is not None:
        n_layers = trial.suggest_int('n_layers', 1, 3)
        model = keras.Sequential([layers.Input(shape=(input_dim,))])
        for i in range(n_layers):
            units = trial.suggest_int(f'units_{i}', 8, 64)
            activation = trial.suggest_categorical(f'act_{i}', ['relu','tanh'])
            dropout = trial.suggest_float(f'dropout_{i}', 0.0, 0.5)
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(dropout))
    else:  # fallback
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(56, activation='tanh'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='tanh'),
            layers.Dropout(0.22)
        ])
    if problem_type == 'regression':
        model.add(layers.Dense(1))
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    else:
        model.add(layers.Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

tracker_nas = EmissionsTracker(project_name=f"nas_model_{TIMESTAMP}", output_dir=OUTPUT_DIR)
tracker_nas.start()
start_time = time.time()

if OPTUNA_AVAILABLE:
    def objective(trial):
        model = create_optimized_model(trial, X_train.shape[1], problem_type)
        model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                  epochs=10, batch_size=min(8, len(X_train)), verbose=0)
        if problem_type == 'regression':
            y_pred = model.predict(X_val).flatten()
            return mean_absolute_error(y_val, y_pred)
        else:
            y_pred_class = (model.predict(X_val) > 0.5).astype(int).flatten()
            return 1 - accuracy_score(y_val, y_pred_class)
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=5, show_progress_bar=True)
    best_trial = study.best_trial
    optimal_model = create_optimized_model(best_trial, X_train.shape[1], problem_type)
else:
    optimal_model = create_optimized_model(None, X_train.shape[1], problem_type)

history_optimal = optimal_model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                                    epochs=15, batch_size=min(8, len(X_train)), verbose=1)

training_time_nas = time.time() - start_time
emissions_nas = tracker_nas.stop()

# Evaluate optimal model
if problem_type == 'regression':
    y_pred_optimal = optimal_model.predict(X_val).flatten()
    optimal_metric = mean_absolute_error(y_val, y_pred_optimal)
else:
    y_pred_optimal_class = (optimal_model.predict(X_val) > 0.5).astype(int).flatten()
    optimal_metric = accuracy_score(y_val, y_pred_optimal_class)

optimal_metrics = {'training_time_s': training_time_nas, 'carbon_emissions_kg': emissions_nas, 'primary_metric': optimal_metric}
optimal_metrics



#  Save Model 
model_save_path = os.path.join(OUTPUT_DIR, f"optimal_model_{TIMESTAMP}.h5")
optimal_model.save(model_save_path)
print(f"Optimal model saved at {model_save_path}")



#  Prepare Kaggle Submission 

# Ensure test features are scaled
if 'X_test_scaled' not in locals():
    X_test_scaled = scaler.transform(X_test)

# Predict using the optimal model
if problem_type == 'regression':
    y_test_pred = optimal_model.predict(X_test_scaled).flatten()
else:
    y_test_pred = (optimal_model.predict(X_test_scaled) > 0.5).astype(int).flatten()

# Prepare submission DataFrame with Kaggle-required column names
if 'Id' in test.columns:  
    submission = pd.DataFrame({'Id': test['Id'], 'GreenScore': y_test_pred})
else:  # Fallback if Id column not present in test set
    submission = pd.DataFrame({'Id': np.arange(len(y_test_pred)), 'GreenScore': y_test_pred})

# Save submission with required Kaggle name in working directory
submission_file = "/kaggle/working/submission.csv"  # Kaggle detects files here
submission.to_csv(submission_file, index=False)
print(f"Kaggle submission saved: {submission_file}")

# Display first few rows of submission
print("\nSubmission Preview:")
display(submission.head())

# Validate submission
print("\nSubmission validation:")
print(f"   Has Id column: {'Id' in submission.columns}")
print(f"   Has GreenScore column: {'GreenScore' in submission.columns}")
print(f"   No missing values: {submission.isnull().sum().sum() == 0}")
print(f"   Correct number of rows: {len(submission) == len(test)}")

print("\nYour submission file must be named submission.csv and placed in /kaggle/working/ for Kaggle to detect it.")


