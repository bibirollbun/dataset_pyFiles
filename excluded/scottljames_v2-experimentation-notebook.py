# ============================================================================
# HACK4EARTH Green AI - ADVANCED MODEL OPTIMIZATION (FIXED VERSION)
# From Large Neural Networks to Efficient Edge Models
# ============================================================================

# Install required packages
!pip install codecarbon scikit-learn pandas numpy matplotlib seaborn lightgbm optuna -q

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from codecarbon import EmissionsTracker
import optuna
import time
import os
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 120)
print(" " * 35 + "HACK4EARTH GREEN AI - ADVANCED OPTIMIZATION")
print(" " * 30 + "From Large Models to Efficient Edge Deployment")
print("=" * 120)

# ============================================================================
# PART 1: ENHANCED DATA GENERATION
# ============================================================================

def generate_complex_energy_dataset(n_samples=50000):
    """Generate large-scale realistic energy data"""
    print("\n" + "="*120)
    print("PART 1: GENERATING LARGE-SCALE DATASET")
    print("="*120)
    
    start_date = pd.Timestamp('2023-01-01')
    timestamps = pd.date_range(start=start_date, periods=n_samples, freq='H')
    
    hours = timestamps.hour.values
    day_of_week = timestamps.dayofweek.values
    day_of_year = timestamps.dayofyear.values
    month = timestamps.month.values
    is_weekend = (day_of_week >= 5).astype(int)
    is_holiday = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])
    
    # Weather patterns
    temperature = 20 + 15 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2)
    temperature += np.random.normal(0, 3, n_samples)
    temperature += 5 * np.sin(2 * np.pi * hours / 24)
    
    humidity = 70 - 0.5 * (temperature - 20) + np.random.normal(0, 10, n_samples)
    humidity = np.clip(humidity, 20, 98)
    
    wind_speed = np.abs(np.random.normal(10, 5, n_samples))
    cloud_cover = np.random.uniform(0, 100, n_samples)
    
    solar_base = 800 * np.sin(2 * np.pi * day_of_year / 365 - np.pi/2) + 400
    solar_hour_factor = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    solar_radiation = solar_base * solar_hour_factor * (1 - cloud_cover/150)
    solar_radiation = np.maximum(solar_radiation + np.random.normal(0, 30, n_samples), 0)
    
    # Building features
    building_types = np.random.choice(['office', 'retail', 'residential', 'industrial'], n_samples)
    building_type_code = np.array([['office', 'retail', 'residential', 'industrial'].index(bt) for bt in building_types])
    
    occupancy_base = {'office': 50, 'retail': 80, 'residential': 30, 'industrial': 20}
    occupancy = np.array([occupancy_base[bt] for bt in building_types])
    occupancy = occupancy * ((hours >= 8) & (hours <= 18)) * (1 - is_weekend * 0.7)
    occupancy = occupancy + np.random.poisson(10, n_samples)
    
    building_age = np.random.randint(1, 50, n_samples)
    building_size = np.random.uniform(500, 10000, n_samples)
    insulation_quality = np.clip(1.0 - building_age * 0.01 + np.random.normal(0, 0.1, n_samples), 0.3, 1.0)
    hvac_efficiency = np.random.uniform(0.6, 0.95, n_samples)
    has_solar_panels = (building_age < 15).astype(int) * np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Energy consumption
    base_load = 30 + building_size * 0.015
    temp_comfort = 22
    hvac_load = 4 * np.abs(temperature - temp_comfort) * building_size * 0.0008 / (insulation_quality * hvac_efficiency)
    hvac_load += 0.5 * wind_speed
    
    occupancy_multiplier = {'office': 2.0, 'retail': 2.5, 'residential': 1.5, 'industrial': 1.0}
    occupancy_effect = occupancy * np.array([occupancy_multiplier[bt] for bt in building_types])
    
    time_patterns = {
        'office': 40 * ((hours >= 8) & (hours <= 18)),
        'retail': 50 * ((hours >= 9) & (hours <= 21)),
        'residential': 30 * ((hours <= 8) | (hours >= 18)),
        'industrial': 60 * np.ones(n_samples)
    }
    time_effect = np.array([time_patterns[bt][i] for i, bt in enumerate(building_types)])
    
    weekend_effect = -30 * is_weekend * (building_type_code <= 1)
    holiday_effect = -50 * is_holiday * (building_type_code <= 1)
    seasonal_effect = 20 * np.abs(np.sin(2 * np.pi * day_of_year / 365))
    solar_offset = -solar_radiation * 0.015 * has_solar_panels
    
    energy_consumption = (
        base_load + hvac_load + occupancy_effect + time_effect + 
        weekend_effect + holiday_effect + seasonal_effect + solar_offset +
        0.5 * temperature * occupancy * 0.01 +
        np.random.normal(0, 8, n_samples)
    )
    energy_consumption = np.maximum(energy_consumption, 15)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'hour': hours,
        'day_of_week': day_of_week,
        'day_of_year': day_of_year,
        'month': month,
        'is_weekend': is_weekend,
        'is_holiday': is_holiday,
        'temperature': temperature,
        'humidity': humidity,
        'wind_speed': wind_speed,
        'cloud_cover': cloud_cover,
        'solar_radiation': solar_radiation,
        'building_type': building_types,
        'building_type_code': building_type_code,
        'occupancy': occupancy,
        'building_age': building_age,
        'building_size': building_size,
        'insulation_quality': insulation_quality,
        'hvac_efficiency': hvac_efficiency,
        'has_solar_panels': has_solar_panels,
        'energy_consumption_kwh': energy_consumption
    })
    
    print(f"\nDataset created: {len(df):,} samples, {df.shape[1]} features")
    return df

df = generate_complex_energy_dataset(50000)

# ============================================================================
# PART 2: FEATURE ENGINEERING
# ============================================================================

print("\n" + "="*120)
print("PART 2: FEATURE ENGINEERING")
print("="*120)

# Cyclical encoding
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Polynomial and interaction features
df['temp_squared'] = df['temperature'] ** 2
df['occupancy_squared'] = df['occupancy'] ** 2
df['temp_occupancy'] = df['temperature'] * df['occupancy']
df['temp_humidity'] = df['temperature'] * df['humidity']
df['comfort_distance'] = np.abs(df['temperature'] - 22)

# One-hot encode building type
building_type_dummies = pd.get_dummies(df['building_type'], prefix='building')
df = pd.concat([df, building_type_dummies], axis=1)

print(f"\nFeature engineering complete: {len([c for c in df.columns if c not in ['timestamp', 'building_type', 'energy_consumption_kwh']])} features")

# Select features
feature_cols = [c for c in df.columns if c not in ['timestamp', 'building_type', 'energy_consumption_kwh']]
X = df[feature_cols].values
y = df['energy_consumption_kwh'].values

# Train-validation-test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)

print(f"\nData split: Train={len(X_train):,}, Val={len(X_val):,}, Test={len(X_test):,}")

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# PART 3: BASELINE MODELS
# ============================================================================

print("\n" + "="*120)
print("PART 3: TRAINING BASELINE MODELS")
print("="*120)

model_results = {}

# ============================================================================
# Model 1: Large DNN (Baseline)
# ============================================================================

print("\nMODEL 1: LARGE DEEP NEURAL NETWORK")
print("-" * 120)

tracker = EmissionsTracker(project_name="large_dnn", output_dir=".", log_level="error")
tracker.start()
start_time = time.time()

large_dnn = tf.keras.Sequential([
    tf.keras.layers.Dense(512, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
], name='Large_DNN')

large_dnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history_large = large_dnn.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=256,
    callbacks=[early_stop],
    verbose=0
)

training_time_large = time.time() - start_time
emissions_large = tracker.stop()

y_pred_large = large_dnn.predict(X_test_scaled, verbose=0).flatten()
mae_large = mean_absolute_error(y_test, y_pred_large)
rmse_large = np.sqrt(mean_squared_error(y_test, y_pred_large))
r2_large = r2_score(y_test, y_pred_large)

large_dnn.save('large_dnn_model.h5')
model_size_large = os.path.getsize('large_dnn_model.h5') / (1024 ** 2)

model_results['Large DNN'] = {
    'model': large_dnn,
    'mae': mae_large,
    'rmse': rmse_large,
    'r2': r2_large,
    'params': large_dnn.count_params(),
    'size_mb': model_size_large,
    'training_time': training_time_large,
    'emissions': emissions_large,
    'predictions': y_pred_large,
    'history': history_large
}

print(f"\nLarge DNN Results:")
print(f"  Parameters:     {large_dnn.count_params():,}")
print(f"  Model Size:     {model_size_large:.2f} MB")
print(f"  Training Time:  {training_time_large:.2f}s")
print(f"  Test MAE:       {mae_large:.3f} kWh")
print(f"  Test R²:        {r2_large:.4f}")
print(f"  Carbon:         {emissions_large*1000:.4f} g CO₂")

# ============================================================================
# Model 2: Small Efficient DNN
# ============================================================================

print("\nMODEL 2: SMALL EFFICIENT DNN")
print("-" * 120)

tracker = EmissionsTracker(project_name="small_dnn", output_dir=".", log_level="error")
tracker.start()
start_time = time.time()

small_dnn = tf.keras.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
], name='Small_DNN')

small_dnn.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

history_small = small_dnn.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=50,
    batch_size=256,
    callbacks=[early_stop],
    verbose=0
)

training_time_small = time.time() - start_time
emissions_small = tracker.stop()

y_pred_small = small_dnn.predict(X_test_scaled, verbose=0).flatten()
mae_small = mean_absolute_error(y_test, y_pred_small)
rmse_small = np.sqrt(mean_squared_error(y_test, y_pred_small))
r2_small = r2_score(y_test, y_pred_small)

small_dnn.save('small_dnn_model.h5')
model_size_small = os.path.getsize('small_dnn_model.h5') / (1024 ** 2)

model_results['Small DNN'] = {
    'model': small_dnn,
    'mae': mae_small,
    'rmse': rmse_small,
    'r2': r2_small,
    'params': small_dnn.count_params(),
    'size_mb': model_size_small,
    'training_time': training_time_small,
    'emissions': emissions_small,
    'predictions': y_pred_small,
    'history': history_small
}

print(f"\nSmall DNN Results:")
print(f"  Parameters:     {small_dnn.count_params():,}")
print(f"  Model Size:     {model_size_small:.2f} MB")
print(f"  Training Time:  {training_time_small:.2f}s")
print(f"  Test MAE:       {mae_small:.3f} kWh (Δ {mae_small - mae_large:+.3f})")
print(f"  Test R²:        {r2_small:.4f}")
print(f"  Size Reduction: {(1-model_size_small/model_size_large)*100:.1f}%")
print(f"  Carbon:         {emissions_small*1000:.4f} g CO₂ ({(1-emissions_small/emissions_large)*100:.1f}% reduction)")

# ============================================================================
# Model 3: LightGBM (Tree-based Baseline)
# ============================================================================

print("\nMODEL 3: LIGHTGBM (EFFICIENT TREE-BASED)")
print("-" * 120)

tracker = EmissionsTracker(project_name="lightgbm", output_dir=".", log_level="error")
tracker.start()
start_time = time.time()

lgbm_model = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=15,
    num_leaves=63,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)

lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

training_time_lgbm = time.time() - start_time
emissions_lgbm = tracker.stop()

y_pred_lgbm = lgbm_model.predict(X_test)
mae_lgbm = mean_absolute_error(y_test, y_pred_lgbm)
rmse_lgbm = np.sqrt(mean_squared_error(y_test, y_pred_lgbm))
r2_lgbm = r2_score(y_test, y_pred_lgbm)

import pickle
with open('lightgbm_model.pkl', 'wb') as f:
    pickle.dump(lgbm_model, f)
model_size_lgbm = os.path.getsize('lightgbm_model.pkl') / (1024 ** 2)

model_results['LightGBM'] = {
    'model': lgbm_model,
    'mae': mae_lgbm,
    'rmse': rmse_lgbm,
    'r2': r2_lgbm,
    'params': 0,
    'size_mb': model_size_lgbm,
    'training_time': training_time_lgbm,
    'emissions': emissions_lgbm,
    'predictions': y_pred_lgbm
}

print(f"\nLightGBM Results:")
print(f"  Model Size:     {model_size_lgbm:.2f} MB")
print(f"  Training Time:  {training_time_lgbm:.2f}s")
print(f"  Test MAE:       {mae_lgbm:.3f} kWh")
print(f"  Test R²:        {r2_lgbm:.4f}")
print(f"  Carbon:         {emissions_lgbm*1000:.4f} g CO₂")

# ============================================================================
# Model 4: Optimized with Optuna
# ============================================================================

print("\nMODEL 4: NEURAL ARCHITECTURE SEARCH (OPTUNA)")
print("-" * 120)

def create_optimized_model(trial):
    n_layers = trial.suggest_int('n_layers', 2, 4)
    
    model = tf.keras.Sequential()
    
    for i in range(n_layers):
        n_units = trial.suggest_int(f'n_units_l{i}', 32, 256, step=32)
        model.add(tf.keras.layers.Dense(n_units, activation='relu', 
                                        input_shape=(X_train_scaled.shape[1],) if i == 0 else None))
        
        if trial.suggest_categorical(f'use_bn_l{i}', [True, False]):
            model.add(tf.keras.layers.BatchNormalization())
        
        dropout = trial.suggest_float(f'dropout_l{i}', 0.0, 0.3, step=0.1)
        if dropout > 0:
            model.add(tf.keras.layers.Dropout(dropout))
    
    model.add(tf.keras.layers.Dense(1))
    
    lr = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae']
    )
    
    return model

def objective(trial):
    model = create_optimized_model(trial)
    
    history = model.fit(
        X_train_scaled, y_train,
        validation_data=(X_val_scaled, y_val),
        epochs=20,
        batch_size=256,
        callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)],
        verbose=0
    )
    
    return min(history.history['val_mae'])

print("\nRunning architecture search (10 trials)...")
study = optuna.create_study(direction='minimize', study_name='energy_nas')
study.optimize(objective, n_trials=10, show_progress_bar=False)

print(f"\nBest trial: MAE={study.best_trial.value:.3f} kWh")
print(f"Best params: {study.best_trial.params}")

# Train final model
tracker = EmissionsTracker(project_name="nas_optimized", output_dir=".", log_level="error")
tracker.start()
start_time = time.time()

nas_model = create_optimized_model(study.best_trial)
history_nas = nas_model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=30,
    batch_size=256,
    callbacks=[early_stop],
    verbose=0
)

training_time_nas = time.time() - start_time
emissions_nas = tracker.stop()

y_pred_nas = nas_model.predict(X_test_scaled, verbose=0).flatten()
mae_nas = mean_absolute_error(y_test, y_pred_nas)
rmse_nas = np.sqrt(mean_squared_error(y_test, y_pred_nas))
r2_nas = r2_score(y_test, y_pred_nas)

nas_model.save('nas_optimized_model.h5')
model_size_nas = os.path.getsize('nas_optimized_model.h5') / (1024 ** 2)

model_results['NAS Optimized'] = {
    'model': nas_model,
    'mae': mae_nas,
    'rmse': rmse_nas,
    'r2': r2_nas,
    'params': nas_model.count_params(),
    'size_mb': model_size_nas,
    'training_time': training_time_nas,
    'emissions': emissions_nas,
    'predictions': y_pred_nas
}

print(f"\nNAS Optimized Model:")
print(f"  Parameters:     {nas_model.count_params():,}")
print(f"  Model Size:     {model_size_nas:.2f} MB")
print(f"  Test MAE:       {mae_nas:.3f} kWh")
print(f"  Test R²:        {r2_nas:.4f}")
print(f"  Carbon:         {emissions_nas*1000:.4f} g CO₂")

# ============================================================================
# PART 4: COMPREHENSIVE COMPARISON
# ============================================================================

print("\n" + "="*120)
print("PART 4: COMPREHENSIVE MODEL COMPARISON")
print("="*120)

comparison_data = []
for name, results in model_results.items():
    comparison_data.append({
        'Model': name,
        'Parameters': results.get('params', 0),
        'Size (MB)': results['size_mb'],
        'MAE': results['mae'],
        'RMSE': results['rmse'],
        'R²': results['r2'],
        'Time (s)': results['training_time'],
        'CO₂ (g)': results['emissions'] * 1000
    })

comparison_df = pd.DataFrame(comparison_data)
print("\nMODEL COMPARISON TABLE:")
print(comparison_df.to_string(index=False))

# Calculate metrics
baseline_mae = comparison_df.iloc[0]['MAE']
baseline_size = comparison_df.iloc[0]['Size (MB)']
baseline_carbon = comparison_df.iloc[0]['CO₂ (g)']

comparison_df['Size Reduction (%)'] = (1 - comparison_df['Size (MB)'] / baseline_size) * 100
comparison_df['Carbon Reduction (%)'] = (1 - comparison_df['CO₂ (g)'] / baseline_carbon) * 100
comparison_df['MAE Change (%)'] = ((comparison_df['MAE'] - baseline_mae) / baseline_mae) * 100

print("\nEFFICIENCY METRICS:")
print(comparison_df[['Model', 'Size Reduction (%)', 'Carbon Reduction (%)', 'MAE Change (%)']].to_string(index=False))

# Visualizations
fig, axes = plt.subplots(2, 3, figsize=(20, 12))

# 1. Model Size
ax = axes[0, 0]
bars = ax.barh(comparison_df['Model'], comparison_df['Size (MB)'], alpha=0.7, edgecolor='black')
for i, bar in enumerate(bars):
    bars[i].set_color('red' if i == 0 else 'green')
ax.set_xlabel('Model Size (MB)')
ax.set_title('Model Size Comparison')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# 2. Accuracy
ax = axes[0, 1]
bars = ax.barh(comparison_df['Model'], comparison_df['MAE'], alpha=0.7, edgecolor='black')
for i, bar in enumerate(bars):
    bars[i].set_color('red' if i == 0 else 'green')
ax.set_xlabel('MAE (kWh)')
ax.set_title('Prediction Accuracy (Lower is Better)')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# 3. Carbon Footprint
ax = axes[0, 2]
bars = ax.barh(comparison_df['Model'], comparison_df['CO₂ (g)'], alpha=0.7, edgecolor='black')
for i, bar in enumerate(bars):
    bars[i].set_color('red' if i == 0 else 'green')
ax.set_xlabel('CO₂ Emissions (g)')
ax.set_title('Training Carbon Footprint')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# 4. Size vs Accuracy Trade-off
ax = axes[1, 0]
ax.scatter(comparison_df['Size (MB)'], comparison_df['MAE'], s=200, alpha=0.6, edgecolors='black', linewidth=2)
for i, model in enumerate(comparison_df['Model']):
    ax.annotate(model, (comparison_df['Size (MB)'].iloc[i], comparison_df['MAE'].iloc[i]), fontsize=9)
ax.set_xlabel('Model Size (MB)')
ax.set_ylabel('MAE (kWh)')
ax.set_title('Accuracy vs Size Trade-off')
ax.grid(True, alpha=0.3)

# 5. R² Score
ax = axes[1, 1]
bars = ax.barh(comparison_df['Model'], comparison_df['R²'], alpha=0.7, edgecolor='black')
for i, bar in enumerate(bars):
    bars[i].set_color('red' if i == 0 else 'green')
ax.set_xlabel('R² Score')
ax.set_title('Explained Variance')
ax.axvline(x=0.9, color='green', linestyle='--', linewidth=2, alpha=0.5)
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

# 6. Training Time
ax = axes[1, 2]
bars = ax.barh(comparison_df['Model'], comparison_df['Time (s)'], alpha=0.7, edgecolor='black')
for i, bar in enumerate(bars):
    bars[i].set_color('red' if i == 0 else 'green')
ax.set_xlabel('Training Time (s)')
ax.set_title('Training Speed')
ax.invert_yaxis()
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("\nVisualization saved as 'model_comparison.png'")
plt.show()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "="*120)
print("FINAL SUMMARY")
print("="*120)

best_mae = comparison_df.loc[comparison_df['MAE'].idxmin()]
smallest = comparison_df.loc[comparison_df['Size (MB)'].idxmin()]

print(f"""
MODELS DEVELOPED: {len(model_results)}

BEST ACCURACY:
  Model:              {best_mae['Model']}
  MAE:                {best_mae['MAE']:.3f} kWh
  R²:                 {best_mae['R²']:.4f}
  Size:               {best_mae['Size (MB)']:.2f} MB

SMALLEST MODEL:
  Model:              {smallest['Model']}
  Size:               {smallest['Size (MB)']:.2f} MB
  Reduction:          {smallest['Size Reduction (%)']:.1f}%
  MAE:                {smallest['MAE']:.3f} kWh

COMPRESSION ACHIEVEMENTS:
  Best Size Reduction:    {comparison_df['Size Reduction (%)'].max():.1f}%
  Best Carbon Reduction:  {comparison_df['Carbon Reduction (%)'].max():.1f}%

READY FOR SUBMISSION!
""")

# Generate submission
submission = pd.DataFrame({
    'id': range(len(model_results[best_mae['Model']]['predictions'])),
    'predicted_energy_consumption': model_results[best_mae['Model']]['predictions']
})
submission.to_csv('submission.csv', index=False)

print("submission.csv created")
print("="*120)
print("COMPLETE - Ready for Kaggle + Devpost submission!")
print("="*120)

