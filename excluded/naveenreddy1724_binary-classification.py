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


train_df = pd.read_csv('/kaggle/input/playground-series-s5e8/train.csv').set_index('id')
test_df = pd.read_csv('/kaggle/input/playground-series-s5e8/test.csv').set_index('id')
test_df['y'] = -1
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e8/sample_submission.csv')
original_df = pd.read_csv('/kaggle/input/bank-marketing-dataset-full/bank-full.csv',delimiter=";")


print(f'Training data shape: {train_df.shape}')
print(f'Training data shape: {test_df.shape}')
print(f'Training data shape: {original_df.shape}')


# Display sample of each dataset
print("Sample of Training Data:")
display(train_df.head())

print("\nSample of Test Data:")
display(test_df.head())

print("\nSample of Original Data:")
display(original_df.head())


# Combine all datasets to extract more features
combined_data = pd.concat([train_df,test_df,original_df], axis=0)
print(f"Combine shape of dataset {combined_data.shape}")

# Analyse features and separate categorical and numerical features
categorical_features = []
numerical_features = []

print("Feature Analysis:")

for col in combined_data.columns[:-1]:
    unique_count = combined_data[col].nunique()
    missing_count = combined_data[col].isna().sum()
    
    if combined_data[col].dtype == 'object':
        categorical_features.append(col)
        feature_type = "Categorical"
    else:
        numerical_features.append(col)
        feature_type = "Numerical"
    print(f"[{feature_type:11}] {col:15} | Unique: {unique_count:5} | Missing: {missing_count:5}")
print(f"\nFeature Summary:")
print(f"   Categorical features: {categorical_features}")
print(f"   Numerical features: {numerical_features}")
print(f"   Total features: {len(categorical_features) + len(numerical_features)}")


# Initialize containers for factorized features
factorized_categorical = []  # For original categorical features
factorized_numerical = []   # For factorized numerical features
feature_cardinalities = {}  # Track unique values for each feature

print(" Applying factorization encoding...")

# Process all features (both numerical and categorical)
for feature in numerical_features + categorical_features:
    
    if feature in numerical_features:
        # For numerical features, create a factorized version with suffix '2'
        factorized_name = f"{feature}2"
        factorized_numerical.append(factorized_name)
    else:
        # For categorical features, keep the same name
        factorized_name = feature
        factorized_categorical.append(feature)
    
    # Apply factorization (converts to integer codes)
    factorized_values, unique_values = pd.factorize(combined_data[feature])
    combined_data[factorized_name] = factorized_values
    
    # Store cardinality (number of unique values)
    feature_cardinalities[factorized_name] = len(unique_values)
    
    # Convert to memory-efficient int32
    combined_data[feature] = combined_data[feature].astype('int32')
    combined_data[factorized_name] = combined_data[factorized_name].astype('int32')

print(f" Created {len(factorized_numerical)} factorized numerical features")
print(f" Processed {len(factorized_categorical)} categorical features")
print(f"\n New factorized features: {factorized_numerical}")
print(f" Feature cardinalities: {feature_cardinalities}")


from itertools import combinations

# Generate all pairwise combinations of factorized features
all_factorized_features = categorical_features + factorized_numerical
feature_pairs = list(combinations(all_factorized_features, 2))

print(f" Creating pairwise feature combinations...")
print(f" Total possible pairs: {len(feature_pairs)}")

# Container for new combination features
combination_features = {}
combination_feature_names = []

# Create combination features
for feature1, feature2 in feature_pairs:
    # Create standardized name (alphabetically sorted)
    combination_name = "_".join(sorted([feature1, feature2]))
    
    # Create combination using mathematical encoding
    # This ensures unique values for each combination
    cardinality_f2 = feature_cardinalities[feature2]
    combination_values = combined_data[feature1] * cardinality_f2 + combined_data[feature2]
    
    combination_features[combination_name] = combination_values
    combination_feature_names.append(combination_name)

# Add combination features to the dataset
if combination_features:
    combination_df = pd.DataFrame(combination_features, index=combined_data.index)
    combined_data = pd.concat([combined_data, combination_df], axis=1)

print(f" Created {len(combination_feature_names)} pairwise combination features")
print(f" New dataset shape: {combined_data.shape}")


# Create count encoding features
count_encoded_features = []
all_categorical_features = categorical_features + factorized_numerical + combination_feature_names

print(f" Creating count encoding features...")
print(f" Processing {len(all_categorical_features)} features...")

# Process in batches for better performance tracking
batch_size = 10
for i, feature in enumerate(all_categorical_features):
    if i % batch_size == 0:
        print(f"   Progress: {i}/{len(all_categorical_features)} features processed")
    
    # Calculate count encoding (frequency of each value)
    count_encoding = combined_data.groupby(feature)['y'].count()
    count_encoding = count_encoding.astype('int32')
    count_encoding.name = f"COUNT_{feature}"
    
    # Add to main dataset
    combined_data = combined_data.merge(count_encoding, on=feature, how='left')
    count_encoded_features.append(f"COUNT_{feature}")

print(f" Created {len(count_encoded_features)} count encoding features")
print(f" Updated dataset shape: {combined_data.shape}")


# Split combined dataset back into original components
train_processed = combined_data.iloc[:len(train_df)].copy()
test_processed = combined_data.iloc[len(train_df):len(train_df)+len(test_df)].copy()
original_processed = combined_data.iloc[-len(original_df):].copy()

print(" Dataset splitting completed:")
print(f"   Training data: {train_processed.shape}")
print(f"   Test data: {test_processed.shape}")
print(f"   Original data: {original_processed.shape}")

# Clean up memory
del combined_data

# Define feature sets for modeling
model_features = numerical_features + categorical_features + factorized_numerical + combination_feature_names + count_encoded_features
print(f"\n Total features for modeling: {len(model_features)}")
print(f"   Numerical: {len(numerical_features)}")
print(f"   Categorical: {len(categorical_features)}")
print(f"   Factorized numerical: {len(factorized_numerical)}")
print(f"   Combinations: {len(combination_feature_names)}")
print(f"   Count encodings: {len(count_encoded_features)}")


import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import optuna
import warnings
warnings.filterwarnings('ignore')

print(" Starting XGBoost Training Pipeline.....")


# Prepare training data
X_train = train_processed[model_features].copy()
y_train = train_processed['y'].copy()
X_test = test_processed[model_features].copy()

print(f" Data Preparation Complete:")
print(f"   Training features: {X_train.shape}")
print(f"   Training target: {y_train.shape}")
print(f"   Test features: {X_test.shape}")
print(f"   Target distribution: {y_train.value_counts().to_dict()}")

# Check for any remaining missing values
print(f"\n Data Quality Check:")
print(f"   Training missing values: {X_train.isnull().sum().sum()}")
print(f"   Test missing values: {X_test.isnull().sum().sum()}")

# Handle any potential data type issues
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
y_train = y_train.astype('int32')

print(f"   Data types optimized: float32 for features, int32 for target")



# Random seed for reproducibility
RANDOM_SEED = 42

# XGBoost parameters optimized for banking data
xgb_parameters = {
    "objective": "binary:logistic",     # Binary classification
    "eval_metric": "auc",               # ROC AUC evaluation
    "learning_rate": 0.1,               # Conservative learning rate
    "max_depth": 0,                     # Use max_leaves instead
    "subsample": 0.8,                   # Row sampling for regularization
    "colsample_bytree": 0.7,           # Column sampling for regularization
    "seed": RANDOM_SEED,                # Reproducibility
    "device": "cpu",                    # CPU (change to "cuda" if GPU available)
    "grow_policy": "lossguide",         # Leaf-wise tree growth
    "max_leaves": 32,                   # Control tree complexity
    "alpha": 2.0,                       # L1 regularization
    "verbosity": 0                      # Suppress output during training
}

# Additional parameters for training
n_estimators = 1000  # Number of boosting rounds
early_stopping_rounds = 150  # Early stopping patience

print(f" Parameters Set!")
print(f"   Optimized specifically for banking/financial data")
print(f"   Using leaf-wise growth with {xgb_parameters['max_leaves']} max leaves")
print(f"   L1 regularization: {xgb_parameters['alpha']}")
print(f"   Learning rate: {xgb_parameters['learning_rate']}")

# Store parameters for final model
best_params = xgb_parameters.copy()


print(f"\n Training Final Model with 5-Fold Cross-Validation...")

# Update best parameters with fixed settings
final_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cpu',
    'verbosity': 0,
    'random_state': 42
}
final_params.update(best_params)

# 5-fold stratified cross-validation for final model
n_folds = 5
kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

# Containers for results
cv_scores = []
feature_importance_list = []
oof_predictions = np.zeros(len(X_train))
test_predictions = np.zeros(len(X_test))
models = []

print(f" Cross-Validation Progress:")
print("-" * 30)

for fold, (train_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
    print(f"Training Fold {fold + 1}/{n_folds}...")
    
    # Split data
    X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
    y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_fold_train, label=y_fold_train)
    dval = xgb.DMatrix(X_fold_val, label=y_fold_val)
    dtest = xgb.DMatrix(X_test)
    
    # Train model with early stopping
    model = xgb.train(
        final_params,
        dtrain,
        num_boost_round=n_estimators,
        evals=[(dtrain, 'train'), (dval, 'eval')],
        early_stopping_rounds=early_stopping_rounds,
        verbose_eval=False
    )
    
    # Out-of-fold predictions
    oof_pred = model.predict(dval)
    oof_predictions[val_idx] = oof_pred
    
    # Test predictions (will be averaged later)
    test_pred = model.predict(dtest)
    test_predictions += test_pred / n_folds
    
    # Calculate fold score
    fold_auc = roc_auc_score(y_fold_val, oof_pred)
    cv_scores.append(fold_auc)
    
    # Store feature importance
    feature_importance_list.append(model.get_score(importance_type='weight'))
    models.append(model)
    
    print(f"   Fold {fold + 1} AUC: {fold_auc:.6f} | Best Iteration: {model.best_iteration}")

# Calculate overall CV performance
overall_auc = roc_auc_score(y_train, oof_predictions)
mean_cv_auc = np.mean(cv_scores)
std_cv_auc = np.std(cv_scores)

print(f"\n Cross-Validation Results:")
print(f"   Overall OOF AUC: {overall_auc:.6f}")
print(f"   Mean CV AUC: {mean_cv_auc:.6f} ± {std_cv_auc:.6f}")
print(f"   Individual fold scores: {[f'{score:.6f}' for score in cv_scores]}")



print(f"\n Feature Importance Analysis...")

# Aggregate feature importance across folds
all_features = set()
for importance_dict in feature_importance_list:
    all_features.update(importance_dict.keys())

feature_importance_avg = {}
for feature in all_features:
    importance_scores = [d.get(feature, 0) for d in feature_importance_list]
    feature_importance_avg[feature] = np.mean(importance_scores)

# Sort features by importance
sorted_features = sorted(feature_importance_avg.items(), key=lambda x: x[1], reverse=True)

print(f" Top 20 Most Important Features:")
for i, (feature, importance) in enumerate(sorted_features[:20]):
    print(f"   {i+1:2d}. {feature:<25} | Importance: {importance:8.2f}")


print(f"\n Generating Final Predictions...")

# Apply threshold for binary classification (optional analysis)
threshold = 0.5
binary_predictions = (test_predictions > threshold).astype(int)

# Create submission dataframe
submission = submission_df.copy()
submission['y'] = test_predictions  # Use probability scores for submission

# Save submission file
submission_filename = 'submission.csv'
submission.to_csv(submission_filename, index=False)

print(f" Submission file created: {submission_filename}")
print(f"   Shape: {submission.shape}")
print(f"   Prediction statistics:")
print(f"      Min: {test_predictions.min():.6f}")
print(f"      Max: {test_predictions.max():.6f}")
print(f"      Mean: {test_predictions.mean():.6f}")
print(f"      Std: {test_predictions.std():.6f}")

# Display first few predictions
print(f"\n Sample Predictions:")
print(submission.head(10))


submission_data = pd.read_csv('/kaggle/working/submission.csv')


submission_data.head()




