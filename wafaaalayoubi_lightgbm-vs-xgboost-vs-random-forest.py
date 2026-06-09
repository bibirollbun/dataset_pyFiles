# --- Core Libraries ---
import numpy as np
import pandas as pd
import time
import warnings

# --- Machine Learning Libraries & Metrics ---
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# --- Tweak Settings ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)


# --- Load Data ---
print("Loading data...")
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
except FileNotFoundError:
    print("Please adjust file paths for local execution.")

print("Data loaded successfully!")
print(f"Training data shape: {train_df.shape}")


# Confirm new libraries are imported
print(f"LightGBM version: {lgb.__version__}")
print(f"XGBoost version: {xgb.__version__}")


print("Starting preprocessing...")

# --- Make a copy to avoid modifying the original dataframe ---
processed_df = train_df.copy()

# --- Drop unnecessary columns ---
processed_df = processed_df.drop(columns=['id'])
print("Dropped 'id' and 'num_reported_accidents'.")

# --- Define feature types based on EDA ---
categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day', 'num_lanes']
boolean_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']

# --- Convert boolean features to integers (0 or 1) ---
for col in boolean_features:
    processed_df[col] = processed_df[col].astype(int)
print("Converted boolean features to integers.")

# --- Apply One-Hot Encoding to categorical features ---
processed_df = pd.get_dummies(processed_df, columns=categorical_features, drop_first=False)
print("Applied One-Hot Encoding.")

# --- Separate features (X) and target (y) ---
X = processed_df.drop(columns=['accident_risk'])
y = processed_df['accident_risk']

print("\nPreprocessing complete!")
print(f"Final features shape (X): {X.shape}")
print(f"Final target shape (y): {y.shape}")

print("\n--- Displaying a sample of the final processed features ---")
display(X.head())


# --- 1. Split the data into a training set (80%) and a validation set (20%) ---
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {len(X_train)}")
print(f"Validation set size: {len(X_val)}")
print("-" * 30)


# --- 2. Define the models to be tested ---
# We use simple, comparable parameters for a fair baseline test.
# n_estimators is set to a moderate 200 to keep runtime reasonable for all models.
models = {
    "LightGBM": lgb.LGBMRegressor(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1),
    "XGBoost": xgb.XGBRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1, min_samples_leaf=10)
}


# --- 3. Loop through models to train, time, and evaluate ---
results = []
for name, model in models.items():
    print(f"Training {name}...")
    
    # Start timer
    start_time = time.time()
    
    # Train model
    model.fit(X_train, y_train)
    
    # End timer
    end_time = time.time()
    training_time = end_time - start_time
    
    # Make predictions and calculate RMSE
    predictions = model.predict(X_val)
    rmse = mean_squared_error(y_val, predictions, squared=False)
    
    # Store results
    results.append({
        "Model": name,
        "Validation RMSE": rmse,
        "Training Time (s)": training_time
    })
    
    print(f"{name} trained in {training_time:.2f} seconds. RMSE: {rmse:.5f}")
    print("-" * 30)


# --- 4. Display the results in a DataFrame ---
results_df = pd.DataFrame(results).sort_values(by="Validation RMSE")

print("\n\n===== SHOWDOWN RESULTS =====\n")
display(results_df)


import matplotlib.pyplot as plt
import seaborn as sns

# --- Create Subplots ---
fig, axes = plt.subplots(1, 2, figsize=(18, 7))
fig.suptitle('Model Showdown: Performance & Speed Comparison', fontsize=20, weight='bold')

# --- Plot 1: Validation RMSE (Lower is Better) ---
sns.barplot(x='Model', y='Validation RMSE', data=results_df, ax=axes[0], palette='viridis')
axes[0].set_title('Validation RMSE (Lower is Better)', fontsize=16)
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('RMSE', fontsize=12)
# Set Y-axis to start from a reasonable value to highlight the small differences
axes[0].set_ylim(results_df['Validation RMSE'].min() - 0.0001, results_df['Validation RMSE'].max() + 0.0001)


# --- Plot 2: Training Time (Lower is Better) ---
sns.barplot(x='Model', y='Training Time (s)', data=results_df, ax=axes[1], palette='plasma')
axes[1].set_title('Training Time in Seconds (Lower is Better)', fontsize=16)
axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('Time (s)', fontsize=12)

# --- Display the plots ---
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()




