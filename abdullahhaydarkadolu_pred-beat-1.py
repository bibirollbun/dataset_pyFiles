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


# ----------------------------------------------------------------------------------
# STEP 1: IMPORT LIBRARIES & EXPLORE DATA
# This cell sets up the project foundation. It imports the necessary libraries,
# merges the training and test data, and performs basic exploratory data analysis
# (EDA) to understand the structure of the dataset.
# ----------------------------------------------------------------------------------

# === 1. Import Necessary Libraries ===
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

# Visualization settings
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

# Suppress warnings
warnings.filterwarnings('ignore')

print("Libraries loaded successfully.")





# === 2. Load Datasets ===
# On Kaggle, data is usually located at '/kaggle/input/<competition-folder-name>/'.
# Please update the paths below according to the location of your competition files.
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")
    sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e9/sample_submission.csv")
    print("Datasets loaded successfully.")
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
except FileNotFoundError:
    print("ERROR: Check file paths! 'train.csv' and 'test.csv' not found.")
    # Create empty DataFrames in case of an error to prevent the script from crashing.
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()





# === 3. Basic Data Analysis (EDA) ===
# --- General Data Structure ---
print("\n--- First Look at the Dataset ---")
print("\nTraining Data First 5 Rows:")
display(train_df.head())

print("\nData Types and Memory Usage:")
train_df.info()

# --- Missing Value Check ---
print("\n--- Missing Value Analysis ---")
missing_values = train_df.isnull().sum()
missing_count = missing_values[missing_values > 0]
if len(missing_count) == 0:
    print("No missing values found in the dataset. Great!")
else:
    print("Missing Values:")
    print(missing_count)

# --- Statistical Summary ---
print("\n--- Statistical Summary of Numerical Columns ---")
# Using .T (transpose) for a more readable format.
display(train_df.describe().T)


# === 4. Target Variable (BeatsPerMinute) Analysis ===
print("\n--- Target Variable Analysis: BeatsPerMinute ---")
plt.figure(figsize=(14, 6))
sns.histplot(train_df['BeatsPerMinute'], kde=True, bins=50, color='blue')
plt.title('Distribution of BeatsPerMinute', fontsize=18)
plt.xlabel('Beats Per Minute (BPM)', fontsize=15)
plt.ylabel('Frequency', fontsize=15)
# Add mean and median lines to the plot
plt.axvline(train_df['BeatsPerMinute'].mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {train_df["BeatsPerMinute"].mean():.2f}')
plt.axvline(train_df['BeatsPerMinute'].median(), color='green', linestyle='-', linewidth=2, label=f'Median: {train_df["BeatsPerMinute"].median():.2f}')
plt.legend()
plt.show()




# === 5. Correlation Analysis ===
print("\n--- Correlation of Features with Each Other and the Target Variable ---")
# We exclude the 'id' column from the analysis
correlation_matrix = train_df.drop('id', axis=1).corr()

plt.figure(figsize=(16, 12))
sns.heatmap(correlation_matrix, annot=False, cmap='viridis', fmt='.2f')
plt.title('Feature Correlation Heatmap', fontsize=18)
plt.show()

# Let's sort the features most correlated with the target variable
print("\nTop 5 Features with Highest Correlation to 'BeatsPerMinute':")
print(correlation_matrix['BeatsPerMinute'].abs().sort_values(ascending=False).head(6))



# ----------------------------------------------------------------------------------
# STEP 2: FEATURE ENGINEERING
# In this step, we create new features from the existing data to improve model
# performance. By combining or transforming features, we can uncover new patterns.
# ----------------------------------------------------------------------------------

# === 1. Combine Datasets for Consistent Processing ===
# We combine train and test sets to ensure that any transformation is applied
# to both datasets equally.

# Concatenate the dataframes, keeping track of the original shapes
train_len = len(train_df)
# We drop the target variable from the training set before combining
combined_df = pd.concat([train_df.drop('BeatsPerMinute', axis=1), test_df], ignore_index=True)

print("Train and test data combined for feature engineering.")
print(f"Shape of combined data: {combined_df.shape}")




# === 2. Create New Features ===
# Based on our initial analysis and domain intuition (music features),
# let's create some interaction and polynomial features.

print("\nCreating new features...")

# --- Interaction Features ---
# Multiply features that might have a combined effect.
combined_df['Energy_Loudness_Interaction'] = combined_df['Energy'] * combined_df['AudioLoudness']
combined_df['Rhythm_Mood_Interaction'] = combined_df['RhythmScore'] * combined_df['MoodScore']

# --- Ratio Features ---
# Create ratios to capture relationships between features.
# Add a small constant (1e-6) to avoid division by zero.
combined_df['Vocal_to_Instrumental_Ratio'] = combined_df['VocalContent'] / (combined_df['InstrumentalScore'] + 1e-6)
combined_df['Acoustic_to_Energy_Ratio'] = combined_df['AcousticQuality'] / (combined_df['Energy'] + 1e-6)

# --- Polynomial Features ---
# Squaring features can help the model capture non-linear relationships.
combined_df['RhythmScore_sq'] = combined_df['RhythmScore'] ** 2
combined_df['MoodScore_sq'] = combined_df['MoodScore'] ** 2

# --- Combined "Quality" Score ---
# A simple sum of quality-related scores.
combined_df['Overall_Audio_Quality'] = combined_df['AcousticQuality'] + combined_df['InstrumentalScore']

# --- Time-based Ratios ---
# Normalize scores by the track duration to get a "per second" metric.
combined_df['Energy_per_Second'] = combined_df['Energy'] / (combined_df['TrackDurationMs'] / 1000 + 1e-6)


print("New features created successfully.")



# === 3. Separate Data Back into Train and Test Sets ===
# Now that our feature engineering is complete, we split the combined dataframe back
# into the training and testing sets.

# The first `train_len` rows belong to the original training data
train_processed = combined_df.iloc[:train_len].copy()

# The remaining rows belong to the test data
test_processed = combined_df.iloc[train_len:].copy()

# Re-attach the target variable to our processed training set
train_processed['BeatsPerMinute'] = train_df['BeatsPerMinute']

print("\nData separated back into training and test sets.")
print(f"New training data shape: {train_processed.shape}")
print(f"New test data shape: {test_processed.shape}")


# --- Display the new features ---
print("\nFirst 5 rows of the new training data with engineered features:")
display(train_processed.head())


# ----------------------------------------------------------------------------------
# STEP 3: MODELING, BLENDING, AND SUBMISSION
# In this step, we train three different powerful models: CatBoost, LightGBM, and XGBoost.
# We then evaluate their performance, select the top two, and blend their
# predictions with a 60/40 weighted average to create the final submission.
# This version avoids using the scikit-learn library to prevent import errors.
# ----------------------------------------------------------------------------------

# === 0. Install XGBoost ===
# We install xgboost as it's not always pre-installed.
!pip install -q xgboost

# === 1. Import Modeling Libraries ===
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
# We use numpy for the RMSE calculation to avoid sklearn dependency.
import numpy as np
import pandas as pd

print("Modeling libraries imported successfully.")


# === 2. Prepare Data for Modeling ===
# Define the features (X) and the target variable (y).
X = train_processed.drop(['id', 'BeatsPerMinute'], axis=1)
y = train_processed['BeatsPerMinute']
X_test = test_processed.drop('id', axis=1)

# Ensure all column names are strings for compatibility
X.columns = X.columns.astype(str)
X_test.columns = X_test.columns.astype(str)

print("Data prepared for modeling.")
print(f"Features shape (X): {X.shape}")
print(f"Test features shape (X_test): {X_test.shape}")


# Custom RMSE function to avoid sklearn dependency
def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred)**2))


# === 3. Train and Evaluate Individual Models ===
model_results = []

# --- 3.1 CatBoost Model ---
print("\n--- Training CatBoost Model ---")
cb_model = CatBoostRegressor(
    iterations=2000,
    learning_rate=0.05,
    depth=7,
    loss_function='RMSE',
    eval_metric='RMSE',
    random_seed=42,
    verbose=0,
    early_stopping_rounds=50,
    task_type='GPU'
)
cb_model.fit(X, y)
cb_train_preds = cb_model.predict(X)
cb_rmse = rmse(y, cb_train_preds)
cb_test_preds = cb_model.predict(X_test)
model_results.append({'name': 'CatBoost', 'rmse': cb_rmse, 'preds': cb_test_preds})
print(f"CatBoost Training RMSE: {cb_rmse:.4f}")


# --- 3.2 LightGBM Model ---
print("\n--- Training LightGBM Model ---")
lgbm_model = LGBMRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    device='gpu', # Use GPU
    n_jobs=-1
)
lgbm_model.fit(X, y)
lgbm_train_preds = lgbm_model.predict(X)
lgbm_rmse = rmse(y, lgbm_train_preds)
lgbm_test_preds = lgbm_model.predict(X_test)
model_results.append({'name': 'LightGBM', 'rmse': lgbm_rmse, 'preds': lgbm_test_preds})
print(f"LightGBM Training RMSE: {lgbm_rmse:.4f}")


# --- 3.3 XGBoost Model ---
print("\n--- Training XGBoost Model ---")
xgb_model = XGBRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=7,
    random_state=42,
    tree_method='gpu_hist',  # Use GPU
    n_jobs=-1
)
xgb_model.fit(X, y)
xgb_train_preds = xgb_model.predict(X)
xgb_rmse = rmse(y, xgb_train_preds)
xgb_test_preds = xgb_model.predict(X_test)
model_results.append({'name': 'XGBoost', 'rmse': xgb_rmse, 'preds': xgb_test_preds})
print(f"XGBoost Training RMSE: {xgb_rmse:.4f}")


# === 4. Select Best Models and Blend Predictions ===
print("\n--- Model Performance Summary ---")
for result in model_results:
    print(f"Model: {result['name']}, Training RMSE: {result['rmse']:.4f}")

# Sort models by RMSE (lower is better)
model_results.sort(key=lambda x: x['rmse'])

best_model = model_results[0]
second_best_model = model_results[1]

print(f"\nBest model: {best_model['name']} (RMSE: {best_model['rmse']:.4f})")
print(f"Second best model: {second_best_model['name']} (RMSE: {second_best_model['rmse']:.4f})")

# Blend predictions with a 60/40 weight
print("\nBlending predictions with a 60/40 weight...")
blended_predictions = 0.6 * best_model['preds'] + 0.4 * second_best_model['preds']


# === 5. Create Submission File ===
# Create the submission DataFrame in the required format
submission_df = pd.DataFrame({
    'id': test_df['id'],
    'BeatsPerMinute': blended_predictions
})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("First 5 rows of the submission file:")
display(submission_df.head())













