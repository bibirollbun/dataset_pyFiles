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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e9/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e9/test.csv")


print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


# Check if test data has target column (some competitions include it, some don't)
has_target_in_test = 'BeatsPerMinute' in test_df.columns
if has_target_in_test:
    print("⚠️  Target column found in test data - removing it")
    test_df = test_df.drop('BeatsPerMinute', axis=1)


def preprocess_data(df, is_training=True):
    """
    Preprocess the dataframe for CatBoost training
    """
    df_processed = df.copy()
    
    # Remove ID column if exists
    if 'id' in df_processed.columns:
        print(f"  Removing ID column")
        df_processed = df_processed.drop('id', axis=1)
    
    # Convert object columns to categorical
    object_columns = df_processed.select_dtypes(include=['object']).columns.tolist()
    
    # Don't convert target variable
    if is_training and 'BeatsPerMinute' in object_columns:
        object_columns.remove('BeatsPerMinute')
    
    print(f"  Converting to categorical: {object_columns}")
    
    for col in object_columns:
        df_processed[col] = df_processed[col].astype('category')
        print(f"    {col}: {df_processed[col].nunique()} categories")
    
    return df_processed


# Preprocess training and test data
train_processed = preprocess_data(train_df, is_training=True)
test_processed = preprocess_data(test_df, is_training=False)

print(f"\nProcessed training shape: {train_processed.shape}")
print(f"Processed test shape: {test_processed.shape}")


# Prepare feature lists
target_col = 'BeatsPerMinute'
id_cols = ['id'] if 'id' in train_df.columns else []
exclude_cols = [target_col] + id_cols


# Get feature columns
feature_columns = [col for col in train_processed.columns if col not in exclude_cols]
print(f"Selected features ({len(feature_columns)}): {feature_columns}")


# Prepare X and y
X_train = train_processed[feature_columns]
y_train = train_processed[target_col]
X_test = test_processed[feature_columns]

# Check for any missing columns in test set
missing_in_test = set(feature_columns) - set(X_test.columns)
if missing_in_test:
    print(f"⚠️  WARNING: Features missing in test set: {missing_in_test}")

print(f"\nFinal training features shape: {X_train.shape}")
print(f"Final test features shape: {X_test.shape}")


from flaml import AutoML
from flaml import tune


# Minimal AutoML settings
automl_settings = {
    "task": "regression",
    "metric": "rmse",
    "time_budget": 3600,  # 1 hour
    "estimator_list": ["lgbm"],
    "retrain_full": True,
    "log_file_name": "lgbm_automl.log",
    "eval_method": "holdout",
    "split_ratio": 0.1,
    "seed": 42,
    "early_stop": True,
    }


# Initialize AutoML
automl = AutoML()


automl.fit(X_train=X_train, y_train=y_train, **automl_settings)


import matplotlib.pyplot as plt

plt.barh(automl.feature_names_in_, automl.feature_importances_)


from flaml.automl.data import get_output_from_log
import numpy as np

time_history, best_valid_loss_history, valid_loss_history, config_history, metric_history = get_output_from_log(filename=automl_settings['log_file_name'], time_budget=3600)
plt.title('Learning Curve')
plt.xlabel('Wall Clock Time (s)')
plt.ylabel('Validation ROC-AUC')
plt.step(time_history, 1 - np.array(best_valid_loss_history), where='post')
plt.show()


# Generate predictions
print("Generating predictions on test set...")
test_predictions = automl.predict(X_test)

print(f"✅ Predictions generated!")
print(f"Prediction distribution:")
print(f"  - Prediction range: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")



# Create submission dataframe
submission = pd.DataFrame()

# Add ID column (adjust based on your competition format)
if 'id' in test_df.columns:
    submission['id'] = test_df['id']
elif 'Id' in test_df.columns:
    submission['Id'] = test_df['Id']
else:
    # If no ID column, create index-based IDs
    submission['id'] = range(len(test_df))
    print("⚠️  No ID column found, using index as ID")

# Add predictions (adjust column name based on competition requirements)
# Common formats: 'y', 'target', 'prediction', 'Survived', etc.
SUBMISSION_TARGET_COLUMN = 'BeatsPerMinute'  # Change this to match your competition

if SUBMISSION_TARGET_COLUMN == 'BeatsPerMinute':
    # For binary classification, some competitions want probabilities, others want binary
    # Check your competition requirements!
    
    # Option 1: Prediction of regression (more common)
    submission[SUBMISSION_TARGET_COLUMN] = test_predictions


print(f"Submission format:")
print(f"  Columns: {list(submission.columns)}")
print(f"  Shape: {submission.shape}")
print(f"  Sample:")
print(submission.head())

# Save submission file
submission_filename = 'submission.csv'
submission.to_csv(submission_filename, index=False)

print(f"\n✅ SUBMISSION SAVED: {submission_filename}")




