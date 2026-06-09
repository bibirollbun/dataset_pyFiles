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


import pandas as pd
import numpy as np
import lightgbm as lgb

# --- 1. Load All Datasets ---
print("Loading competition and external datasets...")
# Competition data
comp_train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# External data from the path you provided
try:
    external_df = pd.read_csv('/kaggle/input/song-popularity-dataset/song_data.csv')
    print("Successfully loaded the external dataset.")
except FileNotFoundError:
    print("External dataset not found at the specified path. Please check the path and data source.")
    external_df = None

# --- 2. Align and Combine Datasets ---
if external_df is not None:
    # Define the mapping from external to competition column names
    column_mapping = {
        'tempo': 'BeatsPerMinute',
        'loudness': 'AudioLoudness',
        'acousticness': 'AcousticQuality',
        'instrumentalness': 'InstrumentalScore',
        'liveness': 'LivePerformanceLikelihood',
        'energy': 'Energy',
        'song_duration_ms': 'TrackDurationMs',
        'speechiness': 'VocalContent',
        'danceability': 'RhythmScore',
        'audio_valence': 'MoodScore'
    }
    
    # Rename columns and select only the ones we can map
    external_renamed_df = external_df.rename(columns=column_mapping)
    mapped_cols = list(column_mapping.values())
    
    # Ensure 'id' is handled correctly - it's not in the external data
    competition_cols = [col for col in comp_train_df.columns if col in mapped_cols or col == 'id']
    
    # Filter both dataframes to only the common, mapped columns
    external_final_df = external_renamed_df[mapped_cols]
    comp_final_df = comp_train_df[competition_cols]

    # Combine the two training dataframes
    full_train_df = pd.concat([comp_final_df, external_final_df], ignore_index=True)
    
    # Remove duplicate rows
    full_train_df = full_train_df.drop_duplicates()
    print(f"Combined training data shape: {full_train_df.shape}")
else:
    full_train_df = comp_train_df # Fallback if external data fails to load

# --- 3. Feature Engineering on All Data ---
def create_features(df):
    df_copy = df.copy()
    # This function uses features that are common to both datasets after mapping
    df_copy['MoodEnergy'] = df_copy['MoodScore'] * df_copy['Energy']
    df_copy['LoudnessQuality'] = df_copy['AudioLoudness'] * df_copy['AcousticQuality']
    epsilon = 1e-6 
    df_copy['VocalInstrumentalRatio'] = df_copy['VocalContent'] / (df_copy['InstrumentalScore'] + epsilon)
    return df_copy

print("Engineering features for the combined train set and the test set...")
train_featured_df = create_features(full_train_df)
test_featured_df = create_features(test_df)

# --- 4. Final Model Training ---
X_full = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
y_full = train_featured_df['BeatsPerMinute']
X_test = test_featured_df.drop(columns=['id'])
X_test = X_test[X_full.columns] # Ensure column order matches

# Use the best hyperparameters we found previously
best_params = {
    'learning_rate': 0.01844, 'num_leaves': 35, 'max_depth': 3, 
    'min_child_samples': 9, 'feature_fraction': 0.970, 'bagging_fraction': 0.751, 
    'bagging_freq': 7, 'lambda_l1': 2.691e-08, 'lambda_l2': 0.1023,
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 3000,
    'random_state': 42, 'verbose': -1, 'n_jobs': -1
}

print("Training final model on the combined and aligned dataset...")
final_model = lgb.LGBMRegressor(**best_params)
final_model.fit(X_full, y_full)

# --- 5. Create New Submission ---
print("Making predictions and creating new submission file...")
predictions = final_model.predict(X_test)

submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': predictions})
submission_df.to_csv('submission_final_boost.csv', index=False)

print("\n'submission_final_boost.csv' created successfully!")
print("This model is trained on integrated data. This should give us a major boost on the leaderboard.")


import pandas as pd
import numpy as np
import lightgbm as lgb

# --- 1. Load Original Data ---
print("Loading original competition data...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# --- 2. Apply Feature Engineering ---
def create_features(df):
    df_copy = df.copy()
    df_copy['MoodEnergy'] = df_copy['MoodScore'] * df_copy['Energy']
    df_copy['LoudnessQuality'] = df_copy['AudioLoudness'] * df_copy['AcousticQuality']
    epsilon = 1e-6
    df_copy['VocalInstrumentalRatio'] = df_copy['VocalContent'] / (df_copy['InstrumentalScore'] + epsilon)
    return df_copy

print("Engineering features...")
train_featured_df = create_features(train_df)
test_featured_df = create_features(test_df)

# Prepare dataframes for modeling
X_train = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
y_train = train_featured_df['BeatsPerMinute']
X_test = test_featured_df.drop(columns=['id'])

# --- 3. Generate Pseudo-Labels ---
print("Training the first model to generate pseudo-labels...")
# Use the best hyperparameters we found before
best_params = {
    'learning_rate': 0.01844, 'num_leaves': 35, 'max_depth': 3,
    'min_child_samples': 9, 'feature_fraction': 0.970, 'bagging_fraction': 0.751,
    'bagging_freq': 7, 'lambda_l1': 2.691e-08, 'lambda_l2': 0.1023,
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
    'random_state': 42, 'verbose': -1, 'n_jobs': -1
}

# Model to generate labels
pseudo_label_model = lgb.LGBMRegressor(**best_params)
pseudo_label_model.fit(X_train, y_train)

# Predict on the test set to create the pseudo-labels
print("Generating pseudo-labels from the test set...")
pseudo_labels = pseudo_label_model.predict(X_test)

# --- 4. Combine Real and Pseudo-Labeled Data ---
print("Combining original training data with pseudo-labeled test data...")
# Create a new dataframe from the test features and pseudo-labels
test_pseudo_df = X_test.copy()
test_pseudo_df['BeatsPerMinute'] = pseudo_labels

# Combine with the original training data
# We don't need the 'id' column for this combined training set
X_train_no_id = train_featured_df.drop(columns=['id'])
combined_df = pd.concat([X_train_no_id, test_pseudo_df], ignore_index=True)

# Prepare the final combined training data
X_combined = combined_df.drop(columns=['BeatsPerMinute'])
y_combined = combined_df['BeatsPerMinute']

# --- 5. Train Final Model and Submit ---
print("Training the final model on the combined dataset...")
# We can use the same parameters, or slightly adjust them (e.g., more estimators)
final_params = best_params.copy()
final_params['n_estimators'] = 3000 # Increase estimators slightly for the larger dataset

final_model = lgb.LGBMRegressor(**final_params)
final_model.fit(X_combined, y_combined)

print("Making final predictions...")
final_predictions = final_model.predict(X_test)

# Create the submission file
submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': final_predictions})
submission_df.to_csv('submission_pseudo_label.csv', index=False)

print("\n'submission_pseudo_label.csv' created successfully!")


# Install necessary libraries if you haven't already
!pip install xgboost catboost

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb

# --- 1. Load and Prepare the Final Dataset (with Pseudo-Labels) ---
print("Loading original competition data...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

# Apply the same feature engineering
def create_features(df):
    df_copy = df.copy()
    df_copy['MoodEnergy'] = df_copy['MoodScore'] * df_copy['Energy']
    df_copy['LoudnessQuality'] = df_copy['AudioLoudness'] * df_copy['AcousticQuality']
    epsilon = 1e-6
    df_copy['VocalInstrumentalRatio'] = df_copy['VocalContent'] / (df_copy['InstrumentalScore'] + epsilon)
    return df_copy

print("Engineering features...")
train_featured_df = create_features(train_df)
test_featured_df = create_features(test_df)

X_train = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
y_train = train_featured_df['BeatsPerMinute']
X_test = test_featured_df.drop(columns=['id'])

# Generate the pseudo-labels from our best single model (LightGBM)
print("Generating pseudo-labels...")
lgbm_params = {
    'learning_rate': 0.01844, 'num_leaves': 35, 'max_depth': 3,
    'min_child_samples': 9, 'feature_fraction': 0.970, 'bagging_fraction': 0.751,
    'bagging_freq': 7, 'lambda_l1': 2.691e-08, 'lambda_l2': 0.1023,
    'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
    'random_state': 42, 'verbose': -1, 'n_jobs': -1
}
pseudo_label_model = lgb.LGBMRegressor(**lgbm_params)
pseudo_label_model.fit(X_train, y_train)
pseudo_labels = pseudo_label_model.predict(X_test)

# Create the final combined training dataset
print("Creating the final combined (pseudo-labeled) dataset...")
test_pseudo_df = X_test.copy()
test_pseudo_df['BeatsPerMinute'] = pseudo_labels
X_train_no_id = train_featured_df.drop(columns=['id'])
combined_df = pd.concat([X_train_no_id, test_pseudo_df], ignore_index=True)

X_combined = combined_df.drop(columns=['BeatsPerMinute'])
y_combined = combined_df['BeatsPerMinute']

# --- 2. Train the Ensemble Models ---

# Model 1: LightGBM
print("Training LightGBM model...")
lgbm_final = lgb.LGBMRegressor(**lgbm_params)
lgbm_final.fit(X_combined, y_combined)
lgbm_preds = lgbm_final.predict(X_test)

# Model 2: XGBoost
print("Training XGBoost model...")
xgb_params = {'objective': 'reg:squarederror', 'n_estimators': 2000, 'learning_rate': 0.02,
              'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8, 'random_state': 42}
xgb_final = xgb.XGBRegressor(**xgb_params)
xgb_final.fit(X_combined, y_combined)
xgb_preds = xgb_final.predict(X_test)

# Model 3: CatBoost
print("Training CatBoost model...")
ctb_params = {'iterations': 2000, 'learning_rate': 0.02, 'depth': 4,
              'l2_leaf_reg': 3, 'loss_function': 'RMSE', 'random_seed': 42, 'verbose': 0}
ctb_final = ctb.CatBoostRegressor(**ctb_params)
ctb_final.fit(X_combined, y_combined)
ctb_preds = ctb_final.predict(X_test)

# --- 3. Blend Predictions and Create Submission ---
print("Blending predictions...")
# Simple average of all three models' predictions
ensemble_preds = (lgbm_preds + xgb_preds + ctb_preds) / 3.0

# Create the final submission file
submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': ensemble_preds})
submission_df.to_csv('submission_ensemble.csv', index=False)

print("\n'submission_ensemble.csv' created successfully!")
print("This is the final push. Submit this file and let's see where we land.")


# Install necessary libraries if you haven't already
!pip install xgboost catboost scikit-learn

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# --- 1. Load and Prepare Data ---
print("Loading and preparing data...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

def create_features(df):
    df_copy = df.copy()
    df_copy['MoodEnergy'] = df_copy['MoodScore'] * df_copy['Energy']
    df_copy['LoudnessQuality'] = df_copy['AudioLoudness'] * df_copy['AcousticQuality']
    epsilon = 1e-6
    df_copy['VocalInstrumentalRatio'] = df_copy['VocalContent'] / (df_copy['InstrumentalScore'] + epsilon)
    return df_copy

train_featured_df = create_features(train_df)
test_featured_df = create_features(test_df)

X = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
y = train_featured_df['BeatsPerMinute']
X_test = test_featured_df.drop(columns=['id'])

# --- 2. Train Base Models with K-Fold CV ---
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Placeholders for OOF predictions and test predictions
oof_preds_lgbm = np.zeros(X.shape[0])
sub_preds_lgbm = np.zeros(X_test.shape[0])
oof_preds_xgb = np.zeros(X.shape[0])
sub_preds_xgb = np.zeros(X_test.shape[0])
oof_preds_ctb = np.zeros(X.shape[0])
sub_preds_ctb = np.zeros(X_test.shape[0])

# Define model parameters
lgbm_params = {'random_state': 42, 'n_jobs': -1, 'verbose': -1, **{'learning_rate': 0.01844, 'num_leaves': 35, 'max_depth': 3, 'min_child_samples': 9, 'feature_fraction': 0.970, 'bagging_fraction': 0.751, 'bagging_freq': 7, 'lambda_l1': 2.691e-08, 'lambda_l2': 0.1023, 'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 3000}}
xgb_params = {'random_state': 42, **{'objective': 'reg:squarederror', 'n_estimators': 3000, 'learning_rate': 0.02, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8}}
ctb_params = {'random_seed': 42, 'verbose': 0, **{'iterations': 3000, 'learning_rate': 0.02, 'depth': 4, 'l2_leaf_reg': 3, 'loss_function': 'RMSE'}}

print("Starting K-Fold training for base models...")
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    
    print(f"--- Fold {n_fold+1} ---")
    
    # LightGBM
    lgbm = lgb.LGBMRegressor(**lgbm_params)
    lgbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgb.early_stopping(150, verbose=False)])
    oof_preds_lgbm[valid_idx] = lgbm.predict(X_valid)
    sub_preds_lgbm += lgbm.predict(X_test) / folds.n_splits
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(**xgb_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=150, verbose=False)
    oof_preds_xgb[valid_idx] = xgb_model.predict(X_valid)
    sub_preds_xgb += xgb_model.predict(X_test) / folds.n_splits
    
    # CatBoost
    ctb_model = ctb.CatBoostRegressor(**ctb_params)
    ctb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=150, verbose=False)
    oof_preds_ctb[valid_idx] = ctb_model.predict(X_valid)
    sub_preds_ctb += ctb_model.predict(X_test) / folds.n_splits

# --- 3. Train Meta-Model on OOF Predictions ---
print("\nTraining meta-model on OOF predictions...")

# Create a new training set for the meta-model
X_meta = pd.DataFrame({
    'lgbm': oof_preds_lgbm,
    'xgb': oof_preds_xgb,
    'ctb': oof_preds_ctb
})

# Meta-model (Ridge is a good, simple choice)
meta_model = Ridge(random_state=42)
meta_model.fit(X_meta, y)

# --- 4. Make Final Predictions ---
print("Making final predictions with the stacking ensemble...")

# Create the test set for the meta-model
X_test_meta = pd.DataFrame({
    'lgbm': sub_preds_lgbm,
    'xgb': sub_preds_xgb,
    'ctb': sub_preds_ctb
})

# Final prediction is from the meta-model
final_predictions = meta_model.predict(X_test_meta)

# --- 5. Create Submission File ---
submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': final_predictions})
submission_df.to_csv('submission_stacking_ensemble.csv', index=False)

print("\n'submission_stacking_ensemble.csv' created successfully!")
print("This is the culmination of our efforts. Good luck.")


import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler

# --- 1. Load, Feature Engineer, and Cluster ---
print("Loading and preparing data...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

def create_features(df):
    df_copy = df.copy()
    df_copy['MoodEnergy'] = df_copy['MoodScore'] * df_copy['Energy']
    df_copy['LoudnessQuality'] = df_copy['AudioLoudness'] * df_copy['AcousticQuality']
    epsilon = 1e-6
    df_copy['VocalInstrumentalRatio'] = df_copy['VocalContent'] / (df_copy['InstrumentalScore'] + epsilon)
    return df_copy

train_featured_df = create_features(train_df)
test_featured_df = create_features(test_df)

X = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
y = train_featured_df['BeatsPerMinute']
X_test = test_featured_df.drop(columns=['id'])

print("Performing KMeans clustering to find hidden song types...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
X['cluster_id'] = kmeans.fit_predict(X_scaled)
X_test['cluster_id'] = kmeans.predict(X_test_scaled)

# --- 2. Train Stacking Ensemble with Cluster Feature ---
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds_lgbm, sub_preds_lgbm = np.zeros(X.shape[0]), np.zeros(X_test.shape[0])
oof_preds_xgb, sub_preds_xgb = np.zeros(X.shape[0]), np.zeros(X_test.shape[0])
oof_preds_ctb, sub_preds_ctb = np.zeros(X.shape[0]), np.zeros(X_test.shape[0])

# Define model parameters (re-using our best ones)
lgbm_params = {'random_state': 42, 'n_jobs': -1, 'verbose': -1, **{'learning_rate': 0.01844, 'num_leaves': 35, 'max_depth': 3, 'min_child_samples': 9, 'feature_fraction': 0.970, 'bagging_fraction': 0.751, 'bagging_freq': 7, 'lambda_l1': 2.691e-08, 'lambda_l2': 0.1023, 'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 3000}}
xgb_params = {'random_state': 42, **{'objective': 'reg:squarederror', 'n_estimators': 3000, 'learning_rate': 0.02, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8}}
ctb_params = {'random_seed': 42, 'verbose': 0, **{'iterations': 3000, 'learning_rate': 0.02, 'depth': 4, 'l2_leaf_reg': 3, 'loss_function': 'RMSE'}}

print("Starting K-Fold training for base models (with cluster feature)...")
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]
    
    print(f"--- Fold {n_fold+1} ---")
    # LGBM, XGB, CatBoost training loop (as before)...
    lgbm = lgb.LGBMRegressor(**lgbm_params); lgbm.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], callbacks=[lgb.early_stopping(150, verbose=False)]); oof_preds_lgbm[valid_idx] = lgbm.predict(X_valid); sub_preds_lgbm += lgbm.predict(X_test) / folds.n_splits
    xgb_model = xgb.XGBRegressor(**xgb_params); xgb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=150, verbose=False); oof_preds_xgb[valid_idx] = xgb_model.predict(X_valid); sub_preds_xgb += xgb_model.predict(X_test) / folds.n_splits
    ctb_model = ctb.CatBoostRegressor(**ctb_params); ctb_model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], early_stopping_rounds=150, verbose=False); oof_preds_ctb[valid_idx] = ctb_model.predict(X_valid); sub_preds_ctb += ctb_model.predict(X_test) / folds.n_splits

# Train Meta-Model
print("\nTraining meta-model...")
X_meta = pd.DataFrame({'lgbm': oof_preds_lgbm, 'xgb': oof_preds_xgb, 'ctb': oof_preds_ctb})
meta_model = Ridge(random_state=42); meta_model.fit(X_meta, y)
X_test_meta = pd.DataFrame({'lgbm': sub_preds_lgbm, 'xgb': sub_preds_xgb, 'ctb': sub_preds_ctb})
stacked_oof_preds = meta_model.predict(X_meta)
stacked_test_preds = meta_model.predict(X_test_meta)

# --- 3. Model the Error (Residuals) ---
print("Performing final error correction...")
errors = y - stacked_oof_preds
# Use the original features (plus cluster_id) to predict the error
error_model = DecisionTreeRegressor(max_depth=4, random_state=42)
error_model.fit(X, errors)
error_correction = error_model.predict(X_test)

# --- 4. Final Prediction and Submission ---
final_predictions = stacked_test_preds + error_correction

submission_df = pd.DataFrame({'id': test_df['id'], 'BeatsPerMinute': final_predictions})
submission_df.to_csv('submission_final_architecture.csv', index=False)

print("\n'submission_final_architecture.csv' created successfully!")
print("This is the most advanced model we can build. The final result awaits.")


# Install SHAP if you haven't already
!pip install shap

import pandas as pd
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import catboost as ctb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import shap

# --- 1. Re-create the Final Model and Data ---
# Note: This block contains the full training pipeline to ensure our model is ready.
print("Loading and preparing data...")
train_df = pd.read_csv('/kaggle/input/playground-series-s5e9/train.csv')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e9/test.csv')

def create_features(df):
    df_copy = df.copy()
    df_copy['MoodEnergy'] = df_copy['MoodScore'] * df_copy['Energy']
    df_copy['LoudnessQuality'] = df_copy['AudioLoudness'] * df_copy['AcousticQuality']
    epsilon = 1e-6
    df_copy['VocalInstrumentalRatio'] = df_copy['VocalContent'] / (df_copy['InstrumentalScore'] + epsilon)
    return df_copy

train_featured_df = create_features(train_df)
test_featured_df = create_features(test_df)

X = train_featured_df.drop(columns=['id', 'BeatsPerMinute'])
y = train_featured_df['BeatsPerMinute']
X_test = test_featured_df.drop(columns=['id'])

print("Creating cluster feature...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)
kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
X['cluster_id'] = kmeans.fit_predict(X_scaled)
X_test['cluster_id'] = kmeans.predict(X_test_scaled)

print("Training the full stacking ensemble to analyze...")
# This is a simplified training for analysis purposes. We use the full dataset.
# Model 1: LightGBM
lgbm_params = {'random_state': 42, 'n_jobs': -1, 'verbose': -1, **{'learning_rate': 0.01844, 'num_leaves': 35, 'max_depth': 3, 'min_child_samples': 9, 'feature_fraction': 0.970, 'bagging_fraction': 0.751, 'bagging_freq': 7, 'lambda_l1': 2.691e-08, 'lambda_l2': 0.1023, 'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000}}
lgbm_final = lgb.LGBMRegressor(**lgbm_params)
lgbm_final.fit(X, y)

# Model 2: XGBoost
xgb_params = {'random_state': 42, **{'objective': 'reg:squarederror', 'n_estimators': 2000, 'learning_rate': 0.02, 'max_depth': 3, 'subsample': 0.8, 'colsample_bytree': 0.8}}
xgb_final = xgb.XGBRegressor(**xgb_params)
xgb_final.fit(X, y)

# Model 3: CatBoost
ctb_params = {'random_seed': 42, 'verbose': 0, **{'iterations': 2000, 'learning_rate': 0.02, 'depth': 4, 'l2_leaf_reg': 3, 'loss_function': 'RMSE'}}
ctb_final = ctb.CatBoostRegressor(**ctb_params)
ctb_final.fit(X, y)

# Meta-Model
X_meta = pd.DataFrame({
    'lgbm': lgbm_final.predict(X),
    'xgb': xgb_final.predict(X),
    'ctb': ctb_final.predict(X)
})
meta_model = Ridge(random_state=42)
meta_model.fit(X_meta, y)

# --- 2. Perform SHAP Analysis ---
print("\nCalculating SHAP values... This may take a moment.")

# To analyze the stacking model, we need a single prediction function
def stacked_predict(data):
    # Base model predictions
    lgbm_p = lgbm_final.predict(data)
    xgb_p = xgb_final.predict(data)
    ctb_p = ctb_final.predict(data)
    # Meta model input
    meta_input = pd.DataFrame({'lgbm': lgbm_p, 'xgb': xgb_p, 'ctb': ctb_p})
    # Meta model prediction
    return meta_model.predict(meta_input)

# We use a KernelExplainer for our complex, stacked model.
# We'll use a sample of the data for speed.
explainer = shap.KernelExplainer(stacked_predict, shap.sample(X, 50))
shap_values = explainer.shap_values(shap.sample(X, 200))

# --- 3. Visualize the Insights ---
print("\n--- SHAP Summary Plot ---")
print("This plot shows the most important features and their impact.")
# The summary plot shows which features are most important and their impact direction.
shap.summary_plot(shap_values, shap.sample(X, 200), plot_type="bar")

print("\n--- SHAP Force Plot for a Single Prediction ---")
print("This plot explains a single prediction, showing what pushed the BPM up or down.")
# The force plot shows how each feature contributed to a single prediction.
shap.initjs() # required for force plots in notebooks
shap.force_plot(explainer.expected_value, shap_values[0,:], X.iloc[0,:])




