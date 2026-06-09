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


# ==================================================================================================
# ğŸŒ¾ GREEN AI 3.0 â€” EcoCrop Yield Forecasting
# Sustainable Agriculture Intelligence with Low-Carbon AI
# ==================================================================================================

!pip install -q codecarbon lightgbm tensorflow==2.16.2 scikit-learn pandas numpy matplotlib seaborn

import os, time, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
from codecarbon import EmissionsTracker

warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)

print("="*115)
print("ğŸŒ¾ GREEN AI 3.0 â€” EcoCrop Yield Forecasting")
print("ğŸ�ƒ Predicting sustainable crop yields using energy-efficient AI")
print("="*115)

# ==================================================================================================
# ğŸŒ¦ï¸� STEP 1: SYNTHETIC ECO-CROP DATA GENERATION
# ==================================================================================================

def generate_crop_data(n_samples=30000):
    np.random.seed(42)
    temperature = np.random.normal(25, 5, n_samples)                   # Â°C
    rainfall = np.random.normal(800, 200, n_samples)                   # mm/year
    soil_ph = np.random.normal(6.5, 0.5, n_samples)
    soil_quality = np.random.uniform(50, 100, n_samples)               # index 0â€“100
    fertilizer_use = np.random.uniform(100, 300, n_samples)            # kg/ha
    irrigation = np.random.uniform(0, 1, n_samples)                    # binary 0/1 scaled
    crop_type = np.random.choice(["Wheat", "Rice", "Maize"], n_samples)

    crop_factor = {"Wheat": 1.0, "Rice": 1.2, "Maize": 0.9}
    base_yield = (
        2.5 * (temperature/25)
        + 0.002 * rainfall
        + 0.05 * soil_quality
        + 0.01 * fertilizer_use
        + 3 * irrigation
    ) * np.vectorize(crop_factor.get)(crop_type)

    noise = np.random.normal(0, 2, n_samples)
    yield_ton_ha = np.maximum(base_yield + noise, 0.5)

    df = pd.DataFrame({
        "temperature": temperature,
        "rainfall": rainfall,
        "soil_ph": soil_ph,
        "soil_quality": soil_quality,
        "fertilizer_use": fertilizer_use,
        "irrigation": irrigation,
        "crop_type": crop_type,
        "yield_ton_ha": yield_ton_ha
    })
    return df

df = generate_crop_data()
print(f"âœ… Generated {len(df):,} synthetic crop records")

# ==================================================================================================
# ğŸ§© STEP 2: FEATURE ENGINEERING
# ==================================================================================================

df = pd.get_dummies(df, columns=["crop_type"], drop_first=True)
features = [c for c in df.columns if c != "yield_ton_ha"]

X, y = df[features].values, df["yield_ton_ha"].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.176, random_state=42)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_val_s = scaler.transform(X_val)
X_test_s = scaler.transform(X_test)

print("ğŸ“Š Data Split:")
print(f"Train={len(X_train)} | Val={len(X_val)} | Test={len(X_test)}")

# ==================================================================================================
# âš™ï¸� STEP 3: MODEL TRAINING
# ==================================================================================================

results = {}

def evaluate(model, X, y, emissions, t_time, size_mb):
    y_pred = model.predict(X).flatten() if hasattr(model, "predict") else model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    return dict(MAE=mae, RMSE=rmse, R2=r2, CO2_g=emissions*1000, Size_MB=size_mb, Time_s=t_time, y_pred=y_pred)

# ---------------- DNN MODEL ----------------
print("\nğŸŒ± Training EcoCrop DNN...")
tracker = EmissionsTracker(project_name="ecocrop_dnn", log_level="error")
tracker.start(); t0 = time.time()

dnn = keras.Sequential([
    keras.layers.Dense(64, activation="relu", input_shape=(X_train_s.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(1)
])
dnn.compile(optimizer="adam", loss="mse", metrics=["mae"])
dnn.fit(X_train_s, y_train, validation_data=(X_val_s, y_val), epochs=30, batch_size=128, verbose=0)

t_time = time.time() - t0; emissions = tracker.stop()
dnn.save("ecocrop_dnn.h5")
results["EcoCrop DNN"] = evaluate(dnn, X_test_s, y_test, emissions, t_time, os.path.getsize("ecocrop_dnn.h5")/(1024**2))

# ---------------- LIGHTGBM MODEL ----------------
print("\nğŸŒ¾ Training LightGBM...")
tracker = EmissionsTracker(project_name="ecocrop_lgbm", log_level="error")
tracker.start(); t0 = time.time()

lgbm = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=64, random_state=42)
lgbm.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
)

t_time = time.time() - t0; emissions = tracker.stop()
with open("ecocrop_lgbm.pkl", "wb") as f: pickle.dump(lgbm, f)
results["LightGBM"] = evaluate(lgbm, X_test, y_test, emissions, t_time, os.path.getsize("ecocrop_lgbm.pkl")/(1024**2))

# ==================================================================================================
# ğŸ“Š STEP 4: RESULTS & SUBMISSION EXPORT
# ==================================================================================================

comparison = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Model"})
print("\nğŸ“ˆ Model Performance:")
print(comparison[["Model", "MAE", "R2", "Size_MB", "CO2_g", "Time_s"]])

plt.figure(figsize=(8,4))
sns.barplot(data=comparison, x="Model", y="MAE", palette="Greens")
plt.title("EcoCrop Yield Forecasting: Model MAE Comparison")
plt.savefig("ecocrop_model_comparison.png", dpi=300, bbox_inches="tight")
plt.show()

# ==================================================================================================
# ğŸ“¦ FIXED SUBMISSION EXPORT SECTION
# ==================================================================================================

submission = pd.DataFrame({
    "Id": np.arange(1, len(best_data["y_pred"]) + 1),   # Capital 'I' + start from 1
    "Predicted": best_data["y_pred"]                    # Generic, competition-friendly name
})

submission_path = os.path.join(os.getcwd(), "submission.csv")
submission.to_csv(submission_path, index=False)

if os.path.exists(submission_path):
    print(f"\nâœ… Submission file successfully created: {submission_path}")
    print(f"ğŸŒ¾ Best Model Used: {best_model_name}")
    print("ğŸ“� Columns:", list(submission.columns))
else:
    print("â�Œ Error: submission.csv not found.")



# ==================================================================================================
# ğŸ“¦ SMART KAGGLE-ADAPTIVE SUBMISSION EXPORT
# ==================================================================================================

import pandas as pd
import os

# Path to save final file
submission_path = os.path.join(os.getcwd(), "submission.csv")

# Try to read sample submission if available
sample_path = "sample_submission.csv"

if os.path.exists(sample_path):
    print(f"ğŸ“„ Found sample submission: {sample_path}")
    sample = pd.read_csv(sample_path)
    print(f"ğŸ§© Sample submission columns: {list(sample.columns)}")

    # If it has 2 columns: ID + Target
    if len(sample.columns) == 2:
        id_col, target_col = sample.columns
        submission = pd.DataFrame({
            id_col: sample[id_col] if len(sample) == len(best_data["y_pred"]) else np.arange(1, len(best_data["y_pred"]) + 1),
            target_col: best_data["y_pred"]
        })
    else:
        # Unexpected structure â†’ fallback
        print("âš ï¸� Unexpected sample format, using fallback schema.")
        submission = pd.DataFrame({
            "Id": np.arange(1, len(best_data["y_pred"]) + 1),
            "Predicted": best_data["y_pred"]
        })
else:
    print("âš ï¸� No sample_submission.csv found. Using default format (Id, Predicted).")
    submission = pd.DataFrame({
        "Id": np.arange(1, len(best_data["y_pred"]) + 1),
        "Predicted": best_data["y_pred"]
    })

# Save file
submission.to_csv(submission_path, index=False)

# Verify creation
if os.path.exists(submission_path):
    print(f"\nâœ… Submission file successfully created: {submission_path}")
    print(f"ğŸŒ¾ Best Model Used: {best_model_name}")
    print(f"ğŸ“� Final Columns: {list(submission.columns)} | Rows: {len(submission)}")
else:
    print("â�Œ Error: submission.csv not found.")


