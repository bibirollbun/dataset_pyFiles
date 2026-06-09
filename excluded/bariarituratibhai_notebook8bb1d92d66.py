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


# ============================================================================
# HACK4EARTH GREEN AI 2.0 - CLEANPOWER FORECASTING
# Low-Carbon Model Optimization for Smart Energy Systems
# ============================================================================

!pip install codecarbon scikit-learn pandas numpy matplotlib seaborn lightgbm optuna tensorflow -q

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from codecarbon import EmissionsTracker
import optuna, time, os, pickle, warnings, matplotlib.pyplot as plt, seaborn as sns
warnings.filterwarnings("ignore")

np.random.seed(42)
tf.random.set_seed(42)

print("="*120)
print("ğŸŒ� HACK4EARTH GREEN AI 2.0 - CLEANPOWER FORECASTING")
print("âš¡ Sustainable Energy Prediction with Low-Carbon Optimization")
print("="*120)

# ============================================================================
# PART 1: SYNTHETIC SMART ENERGY DATASET
# ============================================================================

def generate_cleanpower_data(n_samples=45000):
    """Simulate renewable-based smart grid data for CleanPower forecasting."""
    timestamps = pd.date_range("2024-01-01", periods=n_samples, freq="H")
    hour = timestamps.hour
    day = timestamps.dayofyear
    weekday = timestamps.dayofweek
    is_weekend = (weekday >= 5).astype(int)
    
    # Renewable inputs
    solar_irradiance = np.maximum(0, np.sin((hour - 6)/12*np.pi)) * 1000
    wind_speed = np.abs(np.random.normal(8, 3, n_samples))
    temperature = 15 + 10*np.sin(2*np.pi*day/365) + np.random.normal(0, 2, n_samples)
    
    # Energy demand patterns
    base_demand = 300 + 80*np.sin(2*np.pi*hour/24) + 40*np.random.randn(n_samples)
    industrial_factor = np.random.uniform(0.8, 1.2, n_samples)
    ev_charging = np.maximum(0, np.sin((hour - 17)/5*np.pi)) * np.random.uniform(50, 200, n_samples)
    
    total_demand = base_demand * industrial_factor + ev_charging
    solar_gen = solar_irradiance * np.random.uniform(0.15, 0.25, n_samples)
    wind_gen = wind_speed * np.random.uniform(10, 20, n_samples)
    
    # Net grid load (demand - renewables)
    grid_load = total_demand - (solar_gen + wind_gen*0.7)
    grid_load = np.maximum(grid_load, 50)
    
    df = pd.DataFrame({
        "timestamp": timestamps,
        "hour": hour,
        "day_of_year": day,
        "is_weekend": is_weekend,
        "temperature": temperature,
        "wind_speed": wind_speed,
        "solar_irradiance": solar_irradiance,
        "ev_charging_kw": ev_charging,
        "total_demand_kw": total_demand,
        "grid_load_kw": grid_load
    })
    return df

df = generate_cleanpower_data(45000)
print(f"âœ… Dataset generated: {len(df):,} samples")

# ============================================================================
# PART 2: FEATURE ENGINEERING
# ============================================================================

print("\nğŸ”§ Feature Engineering...")
df["hour_sin"] = np.sin(2*np.pi*df["hour"]/24)
df["hour_cos"] = np.cos(2*np.pi*df["hour"]/24)
df["solar_temp"] = df["solar_irradiance"] * df["temperature"]
df["wind_temp"] = df["wind_speed"] * df["temperature"]

features = [c for c in df.columns if c not in ["timestamp", "grid_load_kw"]]
X = df[features].values
y = df["grid_load_kw"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.176, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print(f"Data Split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")

# ============================================================================
# PART 3: MODEL TRAINING
# ============================================================================

results = {}

def evaluate(model, X, y, emissions, t_time, size_mb):
    y_pred = model.predict(X, verbose=0).flatten() if hasattr(model, "predict") else model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return dict(MAE=mae, RMSE=rmse, R2=r2, CO2_g=emissions*1000, Size_MB=size_mb, Time_s=t_time, y_pred=y_pred)

# ----------------------- Efficient DNN -----------------------
print("\nğŸ§  Training Efficient DNN...")
tracker = EmissionsTracker(project_name="efficient_dnn", log_level="error")
tracker.start(); t0 = time.time()

dnn = keras.Sequential([
    keras.layers.Dense(128, activation="relu", input_shape=(X_train_s.shape[1],)),
    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])
dnn.compile(optimizer="adam", loss="mse", metrics=["mae"])
dnn.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=40, batch_size=256, verbose=0)

t_time = time.time()-t0; emissions = tracker.stop()
dnn.save("efficient_dnn.h5")
size = os.path.getsize("efficient_dnn.h5")/(1024**2)
results["Efficient DNN"] = evaluate(dnn, X_test_s, y_test, emissions, t_time, size)

# ----------------------- LightGBM -----------------------
print("\nğŸŒ² Training LightGBM...")
tracker = EmissionsTracker(project_name="lightgbm_cleanpower", log_level="error")
tracker.start(); t0 = time.time()

lgbm = lgb.LGBMRegressor(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

t_time = time.time()-t0; emissions = tracker.stop()
with open("lightgbm_cleanpower.pkl", "wb") as f:
    pickle.dump(lgbm, f)
size = os.path.getsize("lightgbm_cleanpower.pkl")/(1024**2)
results["LightGBM"] = evaluate(lgbm, X_test, y_test, emissions, t_time, size)

# ----------------------- Optuna DNN -----------------------
print("\nğŸ”� Running Optuna Optimization...")

def objective(trial):
    n_layers = trial.suggest_int("n_layers", 2, 4)
    model = keras.Sequential()
    for i in range(n_layers):
        units = trial.suggest_int(f"units_{i}", 32, 256, step=32)
        model.add(keras.layers.Dense(units, activation="relu", input_shape=(X_train_s.shape[1],) if i==0 else None))
    model.add(keras.layers.Dense(1))
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss="mse", metrics=["mae"])
    hist = model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=15, batch_size=256, verbose=0)
    return min(hist.history["val_mae"])

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=8, show_progress_bar=False)
print("âœ… Best Optuna Params:", study.best_trial.params)

tracker = EmissionsTracker(project_name="optuna_dnn_cleanpower", log_level="error")
tracker.start(); t0 = time.time()

best = study.best_trial.params
opt_model = keras.Sequential()
for i in range(best["n_layers"]):
    opt_model.add(keras.layers.Dense(best[f"units_{i}"], activation="relu", input_shape=(X_train_s.shape[1],) if i==0 else None))
opt_model.add(keras.layers.Dense(1))
opt_model.compile(optimizer=keras.optimizers.Adam(learning_rate=best["lr"]), loss="mse", metrics=["mae"])
opt_model.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=25, batch_size=256, verbose=0)

t_time = time.time()-t0; emissions = tracker.stop()
opt_model.save("optuna_dnn_cleanpower.h5")
size = os.path.getsize("optuna_dnn_cleanpower.h5")/(1024**2)
results["Optuna DNN"] = evaluate(opt_model, X_test_s, y_test, emissions, t_time, size)

# ============================================================================
# PART 4: COMPARISON AND VISUALIZATION
# ============================================================================

comparison = pd.DataFrame(results).T.reset_index().rename(columns={"index":"Model"})
print("\nğŸ“Š Model Comparison:")
print(comparison[["Model","MAE","R2","Size_MB","CO2_g","Time_s"]])

# Efficiency relative to baseline
baseline = comparison.iloc[0]
comparison["Î”MAE%"] = (comparison["MAE"]/baseline["MAE"] - 1)*100
comparison["â†“Size%"] = (1 - comparison["Size_MB"]/baseline["Size_MB"])*100
comparison["â†“CO2%"] = (1 - comparison["CO2_g"]/baseline["CO2_g"])*100

print("\nâ™»ï¸� Efficiency Summary:")
print(comparison[["Model","Î”MAE%","â†“Size%","â†“CO2%"]])

plt.figure(figsize=(10,5))
sns.barplot(data=comparison, x="Model", y="MAE", palette="crest")
plt.title("CleanPower Forecasting: MAE Comparison (Lower = Better)")
plt.savefig("cleanpower_mae_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# ============================================================================
# PART 5: FINAL SUMMARY AND EXPORT
# ============================================================================

best_model = comparison.loc[comparison["MAE"].idxmin()]
print("\nğŸ�† Best Model:", best_model["Model"])
print(f"MAE: {best_model['MAE']:.2f} | RÂ²: {best_model['R2']:.3f} | Size: {best_model['Size_MB']:.2f} MB")

submission = pd.DataFrame({
    "id": np.arange(len(results[best_model["Model"]]["y_pred"])),
    "predicted_grid_load_kw": results[best_model["Model"]]["y_pred"]
})
submission.to_csv("cleanpower_submission.csv", index=False)
print("âœ… Submission file 'cleanpower_submission.csv' created successfully.")


