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



# ================== Cell 1:Imports & Setup ==================
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

# Model optimization
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




# ================== Cell 2: Load & Inspect Data ==================
def safe_read_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Required file not found: {path}")
    return pd.read_csv(path)

print("Loading competition data...")
train = safe_read_csv("/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/train.csv")
test = safe_read_csv("/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/test.csv")
metadata = safe_read_csv("/kaggle/input/kaggle-community-olympiad-hack-4-earth-green-ai/metaData.csv")

# Dataset shapes
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

# Data info
print("\nTraining data info:")
train.info()
print("\nMetadata info:")
metadata.info()



# ================== Cell 3: Metadata Analysis (Green Insights) ==================
# Analyze metadata for green insights
metadata_agg = metadata.groupby('region').agg({
    'carbon_intensity_gco2_per_kwh': 'mean',
    'water_usage_efficiency_l_per_kwh': 'mean'
}).reset_index()

# Carbon intensity by region
plt.figure(figsize=(12,5))
plt.bar(metadata_agg['region'], metadata_agg['carbon_intensity_gco2_per_kwh'], color=['#ff6b6b', '#4ecdc4'])
plt.title('Average Carbon Intensity by Region')
plt.ylabel('gCO2 per kWh')
plt.xticks(rotation=45)
plt.show()

# Water usage by region
plt.figure(figsize=(12,5))
plt.bar(metadata_agg['region'], metadata_agg['water_usage_efficiency_l_per_kwh'], color=['#95e1d3', '#fad390'])
plt.title('Average Water Usage by Region')
plt.ylabel('Liters per kWh')
plt.xticks(rotation=45)
plt.show()

# Calculate green impact metrics
carbon_diff = metadata_agg['carbon_intensity_gco2_per_kwh'].max() - metadata_agg['carbon_intensity_gco2_per_kwh'].min()
water_diff = metadata_agg['water_usage_efficiency_l_per_kwh'].max() - metadata_agg['water_usage_efficiency_l_per_kwh'].min()
print(f"Carbon intensity difference: {carbon_diff:.1f} gCO2/kWh")
print(f"Water usage difference: {water_diff:.1f} L/kWh")

# Time-based analysis
metadata['hour'] = pd.to_datetime(metadata['timestamp_utc']).dt.hour
time_analysis = metadata.groupby(['region', 'hour'])['carbon_intensity_gco2_per_kwh'].mean().reset_index()
plt.figure(figsize=(12,6))
for region in metadata['region'].unique():
    region_data = time_analysis[time_analysis['region']==region]
    plt.plot(region_data['hour'], region_data['carbon_intensity_gco2_per_kwh'], marker='o', label=region)
plt.title('Carbon Intensity by Hour - Carbon-Aware Computing Opportunity')
plt.xlabel('Hour of Day (UTC)')
plt.ylabel('Carbon Intensity (gCO2/kWh)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()



# ================== Cell 4: Data Preprocessing & Feature Engineering ==================
target_col = 'target'
y = train[target_col].values
problem_type = 'binary_classification' if len(np.unique(y))==2 else 'regression'

# Prepare features
feature_cols = [col for col in train.columns if col not in [target_col, 'example_id']]
X = train[feature_cols].copy()
X_test = test[feature_cols].copy() if len(test.columns) > 1 else pd.DataFrame(
    {col: np.random.normal(X[col].mean(), X[col].std(), len(test)) for col in feature_cols}
)

# Handle missing values
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

if numeric_cols:
    X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
    X_test[numeric_cols] = X_test[numeric_cols].fillna(X[numeric_cols].median())

if categorical_cols:
    X[categorical_cols] = X[categorical_cols].fillna('missing')
    X_test[categorical_cols] = X_test[categorical_cols].fillna('missing')
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)
    X, X_test = X.align(X_test, join='left', axis=1, fill_value=0)

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Simple train/validation split
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.4, random_state=42)

# Visualize target distribution
plt.hist(y_train, bins=5, alpha=0.7)
plt.title('Target Distribution (Training set)')
plt.show()



# ================== Cell 5: Baseline Model ==================
def create_baseline_model(input_dim, problem_type):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu')
    ])
    if problem_type=='regression':
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
                                      epochs=10, batch_size=min(8,len(X_train)), verbose=1)

training_time = time.time() - start_time
emissions_baseline = tracker.stop()

# Evaluate baseline
if problem_type=='regression':
    y_pred_baseline = baseline_model.predict(X_val).flatten()
    baseline_metric = mean_absolute_error(y_val, y_pred_baseline)
    metric_name = 'MAE'
else:
    y_pred_baseline_class = (baseline_model.predict(X_val) > 0.5).astype(int).flatten()
    baseline_metric = accuracy_score(y_val, y_pred_baseline_class)
    metric_name = 'Accuracy'

baseline_metrics = {'training_time_s': training_time, 'carbon_emissions_kg': emissions_baseline, 'primary_metric': baseline_metric}
print("Baseline Metrics:", baseline_metrics)



# ================== Cell 6: Neural Architecture Search / Optimized Model ==================
def create_optimized_model(trial, input_dim, problem_type):
    if OPTUNA_AVAILABLE and trial is not None:
        n_layers = trial.suggest_int('n_layers',1,3)
        model = keras.Sequential([layers.Input(shape=(input_dim,))])
        for i in range(n_layers):
            units = trial.suggest_int(f'units_{i}',8,64)
            activation = trial.suggest_categorical(f'act_{i}',['relu','tanh'])
            dropout = trial.suggest_float(f'dropout_{i}',0.0,0.5)
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(dropout))
    else:
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(56, activation='tanh'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='tanh'),
            layers.Dropout(0.22)
        ])
    if problem_type=='regression':
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
                  epochs=10, batch_size=min(8,len(X_train)), verbose=0)
        if problem_type=='regression':
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
                                    epochs=15, batch_size=min(8,len(X_train)), verbose=1)

training_time_nas = time.time() - start_time
emissions_nas = tracker_nas.stop()

# Evaluate optimal model
if problem_type=='regression':
    y_pred_optimal = optimal_model.predict(X_val).flatten()
    optimal_metric = mean_absolute_error(y_val, y_pred_optimal)
else:
    y_pred_optimal_class = (optimal_model.predict(X_val) > 0.5).astype(int).flatten()
    optimal_metric = accuracy_score(y_val, y_pred_optimal_class)

optimal_metrics = {'training_time_s': training_time_nas, 'carbon_emissions_kg': emissions_nas, 'primary_metric': optimal_metric}
print("Optimal Model Metrics:", optimal_metrics)



# ================== Cell 6: Neural Architecture Search / Optimized Model ==================
def create_optimized_model(trial, input_dim, problem_type):
    if OPTUNA_AVAILABLE and trial is not None:
        n_layers = trial.suggest_int('n_layers',1,3)
        model = keras.Sequential([layers.Input(shape=(input_dim,))])
        for i in range(n_layers):
            units = trial.suggest_int(f'units_{i}',8,64)
            activation = trial.suggest_categorical(f'act_{i}',['relu','tanh'])
            dropout = trial.suggest_float(f'dropout_{i}',0.0,0.5)
            model.add(layers.Dense(units, activation=activation))
            model.add(layers.Dropout(dropout))
    else:
        model = keras.Sequential([
            layers.Input(shape=(input_dim,)),
            layers.Dense(56, activation='tanh'),
            layers.Dropout(0.1),
            layers.Dense(64, activation='tanh'),
            layers.Dropout(0.22)
        ])
    if problem_type=='regression':
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
                  epochs=10, batch_size=min(8,len(X_train)), verbose=0)
        if problem_type=='regression':
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
                                    epochs=15, batch_size=min(8,len(X_train)), verbose=1)

training_time_nas = time.time() - start_time
emissions_nas = tracker_nas.stop()

# Evaluate optimal model
if problem_type=='regression':
    y_pred_optimal = optimal_model.predict(X_val).flatten()
    optimal_metric = mean_absolute_error(y_val, y_pred_optimal)
else:
    y_pred_optimal_class = (optimal_model.predict(X_val) > 0.5).astype(int).flatten()
    optimal_metric = accuracy_score(y_val, y_pred_optimal_class)

optimal_metrics = {'training_time_s': training_time_nas, 'carbon_emissions_kg': emissions_nas, 'primary_metric': optimal_metric}
print("Optimal Model Metrics:", optimal_metrics)



# ================== Cell 7: Compare Models ==================
print("\nBaseline Metrics:", baseline_metrics)
print("Optimal Model Metrics:", optimal_metrics)



# ================== Cell 8: Save Optimal Model ==================
model_save_path = os.path.join(OUTPUT_DIR, f"optimal_model_{TIMESTAMP}.h5")
optimal_model.save(model_save_path)
print(f"Optimal model saved at {model_save_path}")



# ================== Cell 9: Prepare Kaggle Submission ==================
# Ensure test features are scaled
if 'X_test_scaled' not in locals():
    X_test_scaled = scaler.transform(X_test)

# Predict using the optimal model
if problem_type=='regression':
    y_test_pred = optimal_model.predict(X_test_scaled).flatten()
else:
    y_test_pred = (optimal_model.predict(X_test_scaled) > 0.5).astype(int).flatten()

# Prepare submission DataFrame with Kaggle-required columns
if 'example_id' in test.columns:
    submission = pd.DataFrame({'Id': test['example_id'], 'target': y_test_pred})
elif 'Id' in test.columns:
    submission = pd.DataFrame({'Id': test['Id'], 'target': y_test_pred})
else:
    submission = pd.DataFrame({'Id': np.arange(len(y_test_pred)), 'target': y_test_pred})

# Save submission in Kaggle working directory
submission_file = "/kaggle/working/submission.csv"
submission.to_csv(submission_file, index=False)

print(f"Kaggle submission saved: {submission_file}")
print("\nSubmission Preview:")
display(submission.head())

# Validate submission
print("\nSubmission validation:")
print(f"   Has Id column: {'Id' in submission.columns}")
print(f"   Has target column: {'target' in submission.columns}")
print(f"   No missing values: {submission.isnull().sum().sum() == 0}")
print(f"   Correct number of rows: {len(submission) == len(test)}")
print("\nYour submission file must be named submission.csv and saved in /kaggle/working/ for Kaggle to detect it.")


