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


train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")


print("Step 1: Data Preprocessing")
print("=" * 30)
print(f"Dataset shape: {train_df.shape}")
print(f"Target distrbution:")
print(train_df["y"].value_counts())
print(f"Target distribution:")
print(train_df["y"].value_counts(normalize=True) * 100)


print("Converting object columns to categorical...")
object_columns = train_df.select_dtypes(include=['object']).columns.tolist()
print(f"Object columns found: {object_columns}")



for col in object_columns:
    if col != 'y':
        train_df[col] = train_df[col].astype('category')
        print(f"  {col}: {train_df[col].nunique()} unique categories")


print("Step 2: Feature Preparation")
print("=" * 30)

# Remove ID column if exists
features_to_remove = ['id'] if 'id' in train_df.columns else []
print(f"Removing features: {features_to_remove}")

# Prepare features WITHOUT duration
features_without_duration = [col for col in train_df.columns 
                           if col not in ['y', 'duration'] + features_to_remove]

# Prepare features WITH duration
features_with_duration = [col for col in train_df.columns if col not in ['y'] + features_to_remove]

print(f"Features WITHOUT duration ({len(features_without_duration)}): {features_without_duration}")
print(f"Features WITH duration ({len(features_with_duration)}): {features_with_duration}")



from sklearn.model_selection import train_test_split, cross_val_score


print("Step 3: Train-Test Split")
print("-" *  30)

# Split data
X_without_duration = train_df[features_without_duration].copy()
X_with_duration = train_df[features_with_duration].copy()
y = train_df['y']

# Train-test split
X_train_no_dur, X_test_no_dur, y_train, y_test = train_test_split(
    X_without_duration, y, test_size=0.2, random_state=42, stratify=y
)

X_train_with_dur, X_test_with_dur, _, _ = train_test_split(
    X_with_duration, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train_no_dur.shape[0]}")
print(f"Test set size: {X_test_no_dur.shape[0]}")
print(f"Training target distribution: {y_train.value_counts().to_dict()}")
print()


from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from catboost import CatBoostClassifier


print("Step 4: Model Training and Evaluation")
print("=" * 50)

# Get categorical feature indices for CatBoost
def get_categorical_features(X):
    return [i for i, col in enumerate(X.columns) if X[col].dtype.name == 'category']

cat_features_no_dur = get_categorical_features(X_train_no_dur)
cat_features_with_dur = get_categorical_features(X_train_with_dur)

print(f"Categorical features (no duration): {[X_train_no_dur.columns[i] for i in cat_features_no_dur]}")
print(f"Categorical features (with duration): {[X_train_with_dur.columns[i] for i in cat_features_with_dur]}")
print()


# Define CatBoost parameters
catboost_params = {
    'iterations': 5000,
    'learning_rate': 0.01,
    'random_seed': 42,
    'verbose': 500,
    'early_stopping_rounds': 100,
    'eval_metric': 'AUC',
    'task_type': 'GPU',
    'devices': '0,1'
}


# MODEL 1: WITHOUT DURATION
print("ğŸ”� MODEL 1: WITHOUT DURATION")
print("-" * 40)

model_no_dur = CatBoostClassifier(**catboost_params)
model_no_dur.fit(
    X_train_no_dur, y_train,
    cat_features=cat_features_no_dur,
    eval_set=(X_test_no_dur, y_test),
)

# Predictions without duration
y_pred_no_dur = model_no_dur.predict(X_test_no_dur)
y_prob_no_dur = model_no_dur.predict_proba(X_test_no_dur)[:, 1]

# Metrics without duration
acc_no_dur = accuracy_score(y_test, y_pred_no_dur)
auc_no_dur = roc_auc_score(y_test, y_prob_no_dur)

print(f"Accuracy: {acc_no_dur:.4f}")
print(f"AUC-ROC: {auc_no_dur:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_no_dur))


# MODEL 2: WITH DURATION
print("ğŸ”� MODEL 2: WITH DURATION")
print("-" * 40)

model_with_dur = CatBoostClassifier(**catboost_params)
model_with_dur.fit(
    X_train_with_dur, y_train,
    cat_features=cat_features_with_dur,
    eval_set=(X_test_with_dur, y_test),
    plot=False
)

# Predictions with duration
y_pred_with_dur = model_with_dur.predict(X_test_with_dur)
y_prob_with_dur = model_with_dur.predict_proba(X_test_with_dur)[:, 1]

# Metrics with duration
acc_with_dur = accuracy_score(y_test, y_pred_with_dur)
auc_with_dur = roc_auc_score(y_test, y_prob_with_dur)

print(f"Accuracy: {acc_with_dur:.4f}")
print(f"AUC-ROC: {auc_with_dur:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred_with_dur))


comparison_df = pd.DataFrame({
    'Model': ['Without Duration', 'With Duration'],
    'Test Accuracy': [acc_no_dur, acc_with_dur],
    'Test AUC': [auc_no_dur, auc_with_dur],
})

print(comparison_df.round(4))
print()

# Performance improvement analysis
auc_improvement = auc_with_dur - auc_no_dur
acc_improvement = acc_with_dur - acc_no_dur

print("ANALYSIS:")
print(f"â€¢ AUC improvement with duration: {auc_improvement:.4f}")
print(f"â€¢ Accuracy improvement with duration: {acc_improvement:.4f}")

if auc_improvement > 0.1:  # Significant improvement
    print("âš ï¸�  WARNING: Large performance improvement suggests possible data leakage!")
    print("   Duration might be the current contact duration, not historical.")
    print("   Recommend using model WITHOUT duration for production.")
elif auc_improvement > 0.05:
    print("âš ï¸�  CAUTION: Moderate improvement - investigate duration feature meaning.")
else:
    print("âœ… Small improvement - duration might be safe to use.")


# STEP 6: FEATURE IMPORTANCE
print("Step 6: Feature Importance")
print("=" * 50)

# Feature importance without duration
print("ğŸ�† TOP 10 FEATURES (WITHOUT DURATION):")
feature_importance_no_dur = model_no_dur.get_feature_importance()
feature_names_no_dur = X_train_no_dur.columns
importance_df_no_dur = pd.DataFrame({
    'feature': feature_names_no_dur,
    'importance': feature_importance_no_dur
}).sort_values('importance', ascending=False)

print(importance_df_no_dur.head(10).to_string(index=False))
print()

# Feature importance with duration
print("ğŸ�† TOP 10 FEATURES (WITH DURATION):")
feature_importance_with_dur = model_with_dur.get_feature_importance()
feature_names_with_dur = X_train_with_dur.columns
importance_df_with_dur = pd.DataFrame({
    'feature': feature_names_with_dur,
    'importance': feature_importance_with_dur
}).sort_values('importance', ascending=False)

print(importance_df_with_dur.head(10).to_string(index=False))


# Check if duration is the top feature
if 'duration' in importance_df_with_dur.head(3)['feature'].values:
    print("\nâš ï¸�  WARNING: Duration is among top 3 features - likely data leakage!")

print("\n" + "="*60)
print("RECOMMENDATION:")
if auc_improvement > 0.1:
    print("Use the model WITHOUT duration for production deployment.")
else:
    print("Both models perform similarly. Duration might be safe to use.")
print("="*60)


# Load training and test data
train_df = pd.read_csv("/kaggle/input/playground-series-s5e8/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e8/test.csv")


print(f"Training data shape: {train_df.shape}")
print(f"Test data shape: {test_df.shape}")


# Check if test data has target column (some competitions include it, some don't)
has_target_in_test = 'y' in test_df.columns
if has_target_in_test:
    print("âš ï¸�  Target column found in test data - removing it")
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


# Based on our earlier analysis, decide whether to include duration
# You can modify this based on your earlier findings
INCLUDE_DURATION = True  # Set to True if you determined duration is safe to use

if INCLUDE_DURATION:
    print("âœ… Including DURATION feature (make sure it's not data leakage!)")
else:
    print("âš ï¸�  EXCLUDING DURATION feature (safer for production)")

# Prepare feature lists
target_col = 'y'
id_cols = ['id'] if 'id' in train_df.columns else []
exclude_cols = [target_col] + id_cols

if not INCLUDE_DURATION and 'duration' in train_processed.columns:
    exclude_cols.append('duration')
    print("   Removing duration from features")

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
    print(f"âš ï¸�  WARNING: Features missing in test set: {missing_in_test}")

print(f"\nFinal training features shape: {X_train.shape}")
print(f"Final test features shape: {X_test.shape}")


# Get categorical features for CatBoost
categorical_features = [i for i, col in enumerate(X_train.columns) 
                       if X_train[col].dtype.name == 'category']

print(f"Categorical features: {[X_train.columns[i] for i in categorical_features]}")


# Define CatBoost parameters
catboost_params = {
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

print(f"âœ… Predictions generated!")
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
    print("âš ï¸�  No ID column found, using index as ID")

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

print(f"\nâœ… SUBMISSION SAVED: {submission_filename}")




