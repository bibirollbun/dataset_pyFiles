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


# Load training and test data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


# Check if test data has target column (some competitions include it, some don't)
has_target_in_test = 'y' in test_df.columns
if has_target_in_test:
    print("⚠️  Target column found in test data - removing it")
    test_df = test_df.drop('y', axis=1)


# Check target distribution
if 'y' in train_df.columns:
    print(f"\nTarget distribution in training data:")
    print(train_df['y'].value_counts())
    print(f"Target distribution (%):")
    print(train_df['y'].value_counts(normalize=True) * 100)



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
    if is_training and 'y' in object_columns:
        object_columns.remove('y')
    
    print(f"  Converting to categorical: {object_columns}")
    
    for col in object_columns:
        df_processed[col] = df_processed[col].astype('category')
        print(f"    {col}: {df_processed[col].nunique()} categories")
    
    # Handle target variable if training data
    if is_training and 'y' in df_processed.columns:
        if df_processed['y'].dtype == 'object':
            print(f"  Converting target variable to binary")
            # Handle different possible encodings
            if set(df_processed['y'].unique()) == {'no', 'yes'}:
                df_processed['y'] = df_processed['y'].map({'no': 0, 'yes': 1})
            elif set(df_processed['y'].unique()) == {'0', '1'}:
                df_processed['y'] = df_processed['y'].astype(int)
            elif set(df_processed['y'].unique()) == {0, 1}:
                pass  # Already binary
            else:
                print(f"    Unique target values: {df_processed['y'].unique()}")
                # Assume first unique value is negative class (0), second is positive (1)
                unique_vals = sorted(df_processed['y'].unique())
                df_processed['y'] = df_processed['y'].map({unique_vals[0]: 0, unique_vals[1]: 1})
            
            print(f"    Target mapping completed. Distribution: {df_processed['y'].value_counts().to_dict()}")
    
    return df_processed


# Preprocess training and test data
train_processed = preprocess_data(train_df, is_training=True)
test_processed = preprocess_data(test_df, is_training=False)

print(f"\nProcessed training shape: {train_processed.shape}")
print(f"Processed test shape: {test_processed.shape}")


# Prepare feature lists
target_col = 'y'
id_cols = ['id'] if 'id' in train_df.columns else []
exclude_cols = [target_col] + id_cols

# Get feature columns
feature_columns = [col for col in train_df.columns if col not in exclude_cols]
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


# Get categorical features for CatBoost
categorical_features = [i for i, col in enumerate(X_train.columns) 
                       if X_train[col].dtype.name == 'category']

print(f"Categorical features: {[X_train.columns[i] for i in categorical_features]}")


from sklearn.utils.class_weight import compute_class_weight

# Option 1: Using sklearn to compute balanced weights
classes = np.unique(y_train)
class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
class_weight_dict = {classes[i]: class_weights[i] for i in range(len(classes))}


class_weight_dict


# Define CatBoost parameters
catboost_params = {
    'class_weights': class_weight_dict,
    'iterations': 5000,
    'learning_rate': 0.01,
    'random_seed': 42,
    'verbose': 500,
    # 'early_stopping_rounds': 100,
    'eval_metric': 'AUC',
    'l2_leaf_reg': 3,
    'bootstrap_type': 'Bayesian',
    'bagging_temperature': 1,
    'task_type': 'GPU',
    'devices': '0,1'
}


from catboost import CatBoostClassifier


# Train final model on full training data
final_model = CatBoostClassifier(**catboost_params)


print("Training final model on full dataset...")
final_model.fit(
    X_train, y_train,
    cat_features=categorical_features,
)


# Show feature importance
feature_importance = final_model.get_feature_importance()
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

print(importance_df.head(10).to_string(index=False))


# Generate predictions
print("Generating predictions on test set...")
test_predictions_proba = final_model.predict_proba(X_test)[:, 1]
test_predictions_binary = final_model.predict(X_test)

print(f"✅ Predictions generated!")
print(f"Prediction distribution:")
print(f"  - Probability range: [{test_predictions_proba.min():.6f}, {test_predictions_proba.max():.6f}]")
print(f"  - Binary predictions: {pd.Series(test_predictions_binary).value_counts().to_dict()}")


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
SUBMISSION_TARGET_COLUMN = 'y'  # Change this to match your competition

if SUBMISSION_TARGET_COLUMN == 'y':
    # For binary classification, some competitions want probabilities, others want binary
    # Check your competition requirements!
    
    # Option 1: Probabilities (more common)
    submission[SUBMISSION_TARGET_COLUMN] = test_predictions_proba
    
    # Option 2: Binary predictions (uncomment if needed)
    # submission[SUBMISSION_TARGET_COLUMN] = test_predictions_binary
    
else:
    submission[SUBMISSION_TARGET_COLUMN] = test_predictions_proba

print(f"Submission format:")
print(f"  Columns: {list(submission.columns)}")
print(f"  Shape: {submission.shape}")
print(f"  Sample:")
print(submission.head())

# Save submission file
submission_filename = 'submission.csv'
submission.to_csv(submission_filename, index=False)

print(f"\n✅ SUBMISSION SAVED: {submission_filename}")




