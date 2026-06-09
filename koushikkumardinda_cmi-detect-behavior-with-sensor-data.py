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


import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
import pickle

# Load datasets
train_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
demographics_df = pd.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")

# ðŸ“Š Feature Engineering Function
def create_features(df, demographics_df):
    """
    Groups data by sequence_id and calculates features.
    This function is identical to the one used in the submission notebook.
    """
    
    # Merge demographics data
    df = pd.merge(df, demographics_df, on='subject', how='left')

    # Create 'has_full_sensors' feature
    df['has_full_sensors'] = df['thm_1'].notna().astype(int)

    # Define columns for feature engineering
    IMU_COLS = ['acc_x', 'acc_y', 'acc_z', 'rot_w', 'rot_x', 'rot_y', 'rot_z']
    THERMOPILE_COLS = [f'thm_{i}' for i in range(1, 6)]
    TOF_COLS = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
    DEMOGRAPHIC_COLS = list(demographics_df.drop('subject', axis=1).columns)

    # 1. IMU Features
    df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
    df['rot_mag'] = np.sqrt(df['rot_w']**2 + df['rot_x']**2 + df['rot_y']**2 + df['rot_z']**2)

    features = []
    
    # Group by sequence_id to calculate aggregate statistics
    agg_features = df.groupby('sequence_id').agg(
        # Statistical features for all relevant columns
        mean=pd.NamedAgg(column='acc_mag', aggfunc='mean'),
        std=pd.NamedAgg(column='acc_mag', aggfunc='std'),
        max=pd.NamedAgg(column='acc_mag', aggfunc='max'),
        min=pd.NamedAgg(column='acc_mag', aggfunc='min'),
        # Add more features as needed (e.g., median, IQR for all columns)
    ).reset_index()

    features.append(agg_features)
    
    # 2. TOF Features
    # Note: Replace -1 with a suitable value or handle separately. Here, we count them.
    for col in TOF_COLS:
        df[f'{col}_is_negative1'] = (df[col] == -1).astype(int)
    
    tof_agg = df.groupby('sequence_id').agg(
        # Count of -1s
        **{f'tof_neg1_count_{i}': ('tof_1_v0_is_negative1', 'sum') for i in range(1, 6)}
        # Add more features for TOF, e.g., mean, std of the pixel data
    ).reset_index()

    features.append(tof_agg)

    # 3. Demographics and Sensor Presence
    demo_and_presence = df.groupby('sequence_id').first()[DEMOGRAPHIC_COLS + ['has_full_sensors']].reset_index()
    features.append(demo_and_presence)

    # Merge all features into a single dataframe
    final_features = features[0]
    for feat_df in features[1:]:
        final_features = pd.merge(final_features, feat_df, on='sequence_id', how='left')

    # Clean up and prepare for model
    final_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    final_features.fillna(0, inplace=True)

    return final_features


# Create features for training data
print(f"Original train_df size: {train_df.shape}")
X_features = create_features(train_df, demographics_df)
print(f"X_features size after creation: {X_features.shape}")

y_target = train_df.groupby('sequence_id')['gesture'].first()
subjects = train_df.groupby('sequence_id')['subject'].first()

# The critical section to debug
print(f"Number of unique sequences in y_target: {y_target.shape[0]}")
print(f"Number of unique sequences in X_features: {X_features.shape[0]}")

# Set 'sequence_id' as index for proper alignment
# This is a key step that may be missing in your create_features function.
X_features.set_index('sequence_id', inplace=True)
y_target.index = y_target.index.astype(str)
X_features.index = X_features.index.astype(str)

# Now, align X and y by their common sequence_id.
common_sequences = X_features.index.intersection(y_target.index)

print(f"Number of common sequences: {len(common_sequences)}")

# Final alignment
if len(common_sequences) == 0:
    raise ValueError("No common sequences found. Your feature engineering function is likely dropping all data.")

X = X_features.loc[common_sequences]
y = y_target.loc[common_sequences]
subjects = subjects.loc[common_sequences]

print(f"Final X size for training: {X.shape}")
print(f"Final y size for training: {y.shape}")

# ðŸ¤– Model Training
categorical_features = ['adult_child', 'sex', 'handedness', 'has_full_sensors']
lgbm = lgb.LGBMClassifier(objective='multiclass', random_state=42, n_estimators=1000)

# GroupKFold Cross-Validation
gkf = GroupKFold(n_splits=5)
for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=subjects)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

    lgbm.fit(X_train, y_train, 
             eval_set=[(X_val, y_val)], 
             eval_metric='multi_logloss', 
             callbacks=[lgb.early_stopping(100)],
             categorical_feature=categorical_features,
             )

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(lgbm, f)

print("Model trained and saved as 'model.pkl'")


# This script is a complete Kaggle submission file.
# It reads the test data, applies the necessary feature engineering,
# uses a pre-trained model to make predictions, and saves the output
# to a 'submission.csv' file, which Kaggle uses for scoring.

import pandas as pd
import numpy as np
import lightgbm as lgb
import pickle
import os
import sys

# Define file paths for the input data and the pre-trained model.
# These paths are standard for Kaggle competition notebooks.
TEST_DATA_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv'
DEMOGRAPHICS_DATA_PATH = '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv'
# IMPORTANT: You must change the path below to match the directory where
# you have saved your pre-trained 'model.pkl' file.
MODEL_PATH = '/kaggle/working/model.pkl'

# 1. Load Data and Model
print("Loading data and model...")
try:
    # Load the test data and demographics.
    test_df = pd.read_csv(TEST_DATA_PATH)
    demographics_df = pd.read_csv(DEMOGRAPHICS_DATA_PATH)
    
    # Load the pre-trained LightGBM model.
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    # --- FIX START ---
    # Add a check to ensure the loaded object is a valid model with a 'predict' method.
    if not hasattr(model, 'predict'):
        print("Error: The loaded object from 'model.pkl' does not have a '.predict()' method.")
        print("Please ensure you are pickling the trained model object itself, not an array of predictions.")
        sys.exit(1)
    # --- FIX END ---

except FileNotFoundError as e:
    print(f"Error: {e}. Please ensure that the data and model files are correctly included.")
    sys.exit(1)

print("Data and model loaded successfully.")

# 2. Feature Engineering
print("Starting feature engineering...")

# Merge the test data and demographics.
full_test_df = pd.merge(test_df, demographics_df, on='subject', how='left')

# Define columns for feature engineering.
TOF_COLS = [f'tof_{i}_v{j}' for i in range(1, 6) for j in range(64)]
DEMOGRAPHIC_COLS = list(demographics_df.drop('subject', axis=1).columns)

# Calculate magnitude features for accelerometer and gyroscope.
full_test_df['acc_mag'] = np.sqrt(full_test_df['acc_x']**2 + full_test_df['acc_y']**2 + full_test_df['acc_z']**2)
full_test_df['rot_mag'] = np.sqrt(full_test_df['rot_w']**2 + full_test_df['rot_x']**2 + full_test_df['rot_y']**2 + full_test_df['rot_z']**2)

# Create a feature indicating if all sensors are present.
full_test_df['has_full_sensors'] = full_test_df['thm_1'].notna().astype(int)

# Group the data by 'sequence_id' to generate aggregate features.
features = []
agg_features = full_test_df.groupby('sequence_id').agg(
    mean=pd.NamedAgg(column='acc_mag', aggfunc='mean'),
    std=pd.NamedAgg(column='acc_mag', aggfunc='std'),
    max=pd.NamedAgg(column='acc_mag', aggfunc='max'),
    min=pd.NamedAgg(column='acc_mag', aggfunc='min'),
).reset_index()
features.append(agg_features)

# Handle the specific case of -1 values in ToF sensors.
for col in TOF_COLS:
    full_test_df[f'{col}_is_negative1'] = (full_test_df[col] == -1).astype(int)

# Count the number of -1 values per sequence for each ToF sensor.
tof_agg = full_test_df.groupby('sequence_id').agg(
    **{f'tof_neg1_count_{i}': ('tof_1_v0_is_negative1', 'sum') for i in range(1, 6)}
).reset_index()
features.append(tof_agg)

# Aggregate demographic and sensor presence features.
demo_and_presence = full_test_df.groupby('sequence_id').first()[DEMOGRAPHIC_COLS + ['has_full_sensors']].reset_index()
features.append(demo_and_presence)

# Merge all the engineered feature sets into a single DataFrame.
final_features = features[0]
for feat_df in features[1:]:
    final_features = pd.merge(final_features, feat_df, on='sequence_id', how='left')

# Replace any infinite values with NaN and then fill NaNs with 0.
final_features.replace([np.inf, -np.inf], np.nan, inplace=True)
final_features.fillna(0, inplace=True)

print("Feature engineering complete.")

# 3. Make Predictions
print("Making predictions...")

# Get the list of feature columns for prediction.
training_columns = list(final_features.drop('sequence_id', axis=1).columns)

# Use the loaded model to predict the gesture.
predictions = model.predict(final_features[training_columns])

print("Predictions complete.")

# 4. Create and Save Submission File
print("Creating submission file...")

# Create the submission DataFrame with 'sequence_id' and predicted 'gesture'.
submission_df = pd.DataFrame({
    'sequence_id': final_features['sequence_id'],
    'gesture': predictions
})

# Save the DataFrame to a CSV file. The file name must be 'submission.csv' for Kaggle.
submission_df.to_csv('submission.csv', index=False)

print("Submission file 'submission.csv' created successfully.")
print("Script finished.")


