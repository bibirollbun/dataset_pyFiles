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


# Import necessary libraries
import matplotlib.pyplot as plt
import seaborn as sns

# Set some display options for pandas
pd.set_option('display.max_columns', None)

# Load the datasets
try:
    # Running in Kaggle
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
    sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')
except FileNotFoundError:
    # For running locally if you've downloaded the files
    print("Kaggle file paths not found, trying local paths...")
    # Update these paths to where you've saved the data
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')
    sample_submission_df = pd.read_csv('sample_submission.csv')


# --- Initial Health Check ---

# 1. Look at the first few rows to understand the features
print("--- Training Data Head ---")
print(train_df.head())
print("\n" + "="*50 + "\n")

# 2. Check the data types and look for missing values
print("--- Training Data Info ---")
train_df.info()
print("\n" + "="*50 + "\n")

# 3. Get a statistical summary of the numerical features
print("--- Training Data Description ---")
print(train_df.describe())


# Set the aesthetic style of the plots
sns.set_style('whitegrid')

# --- 1. Visualize the Target Variable Distribution ---
plt.figure(figsize=(12, 6))
sns.histplot(train_df['BeatsPerMinute'], kde=True, bins=50)
plt.title('Distribution of BeatsPerMinute (BPM)', fontsize=15)
plt.xlabel('BPM')
plt.ylabel('Frequency')
plt.show()


# --- 2. Visualize Feature Distributions ---
# We'll drop 'id' and the target 'BeatsPerMinute' for this plot
features = train_df.drop(columns=['id', 'BeatsPerMinute']).columns
plt.figure(figsize=(16, 12))
for i, feature in enumerate(features):
    plt.subplot(3, 3, i + 1) # Creating a 3x3 grid of plots
    sns.histplot(train_df[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
plt.tight_layout()
plt.show()


# --- 3. Visualize Correlations with a Heatmap ---
plt.figure(figsize=(14, 10))
# Calculate the correlation matrix
correlation_matrix = train_df.drop(columns=['id']).corr()
sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False)
plt.title('Correlation Matrix of Features', fontsize=15)
plt.show()

# To see the exact correlation values with the target
print("\n--- Correlation with BeatsPerMinute ---")
print(correlation_matrix['BeatsPerMinute'].sort_values(ascending=False))


# Import the necessary tools for modeling
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np

# --- 1. Prepare Data for Modeling ---

# Define our features (X) and target (y)
# We drop 'id' because it's just an identifier, and 'BeatsPerMinute' because it's our target
features = train_df.drop(columns=['id', 'BeatsPerMinute'])
target = train_df['BeatsPerMinute']

# Create a validation set to test our model's performance
# We'll use 80% of the data for training and 20% for validation
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}")
print(f"Validation data shape: {X_val.shape}")


# --- 2. Train the LightGBM Model ---

# Initialize the LightGBM Regressor
# We use basic parameters for now. 'random_state' ensures our results are reproducible.
lgbm = lgb.LGBMRegressor(random_state=42)

print("\nTraining the LightGBM model...")
# Train the model on our training data
lgbm.fit(X_train, y_train)


# --- 3. Evaluate the Model ---

print("Making predictions on the validation data...")
# Predict the BPM for our unseen validation data
predictions = lgbm.predict(X_val)

# Calculate the Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("\n" + "="*50)
print(f"Validation RMSE: {rmse:.4f}")
print("="*50)


# We'll use the same imports as before
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np
import pandas as pd # Make sure pandas is available

# --- 1. Engineer New Features ---
# Let's work with a fresh copy to be safe
train_featured_df = train_df.copy()

print("Creating new features...")
train_featured_df['MoodEnergy'] = train_featured_df['MoodScore'] * train_featured_df['Energy']
train_featured_df['LoudnessQuality'] = train_featured_df['AudioLoudness'] * train_featured_df['AcousticQuality']
# Add a small epsilon to prevent division by zero
epsilon = 1e-6 
train_featured_df['VocalInstrumentalRatio'] = train_featured_df['VocalContent'] / (train_featured_df['InstrumentalScore'] + epsilon)

print("New features created:")
print(train_featured_df[['MoodEnergy', 'LoudnessQuality', 'VocalInstrumentalRatio']].head())


# --- 2. Prepare Data for Modeling (with new features) ---

# Define our features (X) and target (y)
features = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
target = train_featured_df['BeatsPerMinute']

# Create a validation set
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

print(f"\nNew training data shape: {X_train.shape}")


# --- 3. Train and Evaluate the Model ---

lgbm = lgb.LGBMRegressor(random_state=42)

print("Training the LightGBM model with new features...")
lgbm.fit(X_train, y_train)

print("Making predictions on the validation data...")
predictions = lgbm.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("\n" + "="*50)
print(f"Validation RMSE with new features: {rmse:.4f}")
print("="*50)# We'll use the same imports as before
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np
import pandas as pd # Make sure pandas is available

# --- 1. Engineer New Features ---
# Let's work with a fresh copy to be safe
train_featured_df = train_df.copy()

print("Creating new features...")
train_featured_df['MoodEnergy'] = train_featured_df['MoodScore'] * train_featured_df['Energy']
train_featured_df['LoudnessQuality'] = train_featured_df['AudioLoudness'] * train_featured_df['AcousticQuality']
# Add a small epsilon to prevent division by zero
epsilon = 1e-6 
train_featured_df['VocalInstrumentalRatio'] = train_featured_df['VocalContent'] / (train_featured_df['InstrumentalScore'] + epsilon)

print("New features created:")
print(train_featured_df[['MoodEnergy', 'LoudnessQuality', 'VocalInstrumentalRatio']].head())


# --- 2. Prepare Data for Modeling (with new features) ---

# Define our features (X) and target (y)
features = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
target = train_featured_df['BeatsPerMinute']

# Create a validation set
X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

print(f"\nNew training data shape: {X_train.shape}")


# --- 3. Train and Evaluate the Model ---

lgbm = lgb.LGBMRegressor(random_state=42)

print("Training the LightGBM model with new features...")
lgbm.fit(X_train, y_train)

print("Making predictions on the validation data...")
predictions = lgbm.predict(X_val)

rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("\n" + "="*50)
print(f"Validation RMSE with new features: {rmse:.4f}")
print("="*50)


# Install optuna if you haven't already
!pip install optuna

# Import the necessary libraries
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import numpy as np

# --- 1. Prepare Data (using the features we already created) ---

train_featured_df = train_df.copy()
train_featured_df['MoodEnergy'] = train_featured_df['MoodScore'] * train_featured_df['Energy']
train_featured_df['LoudnessQuality'] = train_featured_df['AudioLoudness'] * train_featured_df['AcousticQuality']
epsilon = 1e-6 
train_featured_df['VocalInstrumentalRatio'] = train_featured_df['VocalContent'] / (train_featured_df['InstrumentalScore'] + epsilon)

features = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
target = train_featured_df['BeatsPerMinute']

X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

# --- 2. Define the Objective Function for Optuna ---

def objective(trial):
    # Define the search space for hyperparameters
    params = {
        'objective': 'regression_l1',  # MAE is often more robust to outliers
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }
    
    # Train the model with the suggested hyperparameters
    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)]) # Stop if no improvement after 100 rounds
    
    # Make predictions and calculate RMSE
    preds = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, preds))
    
    return rmse

# --- 3. Run the Optimization ---

print("Starting hyperparameter optimization with Optuna...")
# We direct it to minimize the RMSE
study = optuna.create_study(direction='minimize')
# We'll run it for a limited number of trials to save time. 25-50 is a good start.
study.optimize(objective, n_trials=30) 

print("Optimization finished.")
print("Best trial's RMSE:", study.best_value)
print("Best hyperparameters:", study.best_params)


# --- 4. Train Final Model with Best Parameters ---

print("\nTraining final model with the best hyperparameters...")
best_params = study.best_params
best_params['n_estimators'] = 2000 # Increase estimators for the final model
best_params['random_state'] = 42
best_params['verbose'] = -1
best_params['objective'] = 'regression_l1'
best_params['metric'] = 'rmse'

final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_train, y_train,
                eval_set=[(X_val, y_val)],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(100, verbose=False)])

predictions = final_model.predict(X_val)
final_rmse = np.sqrt(mean_squared_error(y_val, predictions))

print("\n" + "="*50)
print(f"Final Validation RMSE after tuning: {final_rmse:.4f}")
print("="*50)


import pandas as pd
import numpy as np
import lightgbm as lgb

print("Loading data...")
# Load the original datasets
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')
sample_submission_df = pd.read_csv('/kaggle/input/playground-series-s5e9/sample_submission.csv')

# --- 1. Apply Feature Engineering to BOTH train and test data ---
def create_features(df):
    df_copy = df.copy()
    df_copy['MoodEnergy'] = df_copy['MoodScore'] * df_copy['Energy']
    df_copy['LoudnessQuality'] = df_copy['AudioLoudness'] * df_copy['AcousticQuality']
    epsilon = 1e-6 
    df_copy['VocalInstrumentalRatio'] = df_copy['VocalContent'] / (df_copy['InstrumentalScore'] + epsilon)
    return df_copy

print("Engineering features for train and test sets...")
train_featured_df = create_features(train_df)
test_featured_df = create_features(test_df)

# --- 2. Prepare Full Dataset for Final Training ---

# All training data
X_full = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
y_full = train_featured_df['BeatsPerMinute']

# Test data for prediction
X_test = test_featured_df.drop(columns=['id'])

# Ensure columns are in the same order
X_test = X_test[X_full.columns]

# --- 3. Train the Final Model on 100% of the Data ---

# Use the best hyperparameters we found with Optuna
best_params = {
    'learning_rate': 0.018440012649975284, 
    'num_leaves': 35, 
    'max_depth': 3, 
    'min_child_samples': 9, 
    'feature_fraction': 0.970664260192119, 
    'bagging_fraction': 0.7517661116460606, 
    'bagging_freq': 7, 
    'lambda_l1': 2.691050348593101e-08, 
    'lambda_l2': 0.1023360977484737,
    'objective': 'regression_l1',
    'metric': 'rmse',
    'n_estimators': 2000, # Using a generous number of estimators
    'random_state': 42,
    'verbose': -1,
    'n_jobs': -1
}

print("Training final model on all data...")
final_model = lgb.LGBMRegressor(**best_params)

# No need for validation set here, we use all data to train
final_model.fit(X_full, y_full)

# --- 4. Make Predictions and Create Submission File ---
print("Making predictions on the test set...")
predictions = final_model.predict(X_test)

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': predictions})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("\n'submission.csv' file created successfully!")
print("Head of the submission file:")
print(submission_df.head())



















