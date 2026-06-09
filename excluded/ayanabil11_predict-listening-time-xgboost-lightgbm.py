# Install required packages
!pip install optuna category_encoders -q


# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import optuna
from category_encoders import TargetEncoder

# Set random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# Load datasets
train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


# Display basic info
print("Train Shape:", train.shape)
print("Test Shape:", test.shape)
print("\nTrain Info:")
print(train.info())
print("\nTrain Description:")
print(train.describe())


# Display column names
print("\nColumns in Train:")
print(train.columns.tolist())

# Check for missing values
print("\nMissing Values in Train:")
print(train.isna().sum())

# Check for missing values in Test
print("\nMissing Values in Test:")
print(test.isna().sum())


# Plot target distribution
plt.figure(figsize=(10, 6))
sns.histplot(train['Listening_Time_minutes'], bins=50)
plt.title('Distribution of Listening Time (Minutes)')
plt.xlabel('Listening Time (Minutes)')
plt.ylabel('Frequency')
plt.show()

# Target statistics
print("\nListening Time Statistics:")
print(train['Listening_Time_minutes'].describe())


# Numerical features distribution
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                  'Guest_Popularity_percentage', 'Number_of_Ads']
for col in numerical_cols:
    plt.figure(figsize=(8, 5))
    sns.histplot(train[col], bins=30)
    plt.title(f'Distribution of {col}')
    plt.show()

# Categorical features unique values
categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Publication_Day', 
                    'Publication_Time', 'Episode_Sentiment']
for col in categorical_cols:
    print(f"\nUnique values in {col}: {train[col].nunique()}")
    print(train[col].value_counts().head())


def preprocess_data(train, test, target_col='Listening_Time_minutes'):
    """
    Preprocess train and test data, including feature engineering.
    
    Args:
        train (pd.DataFrame): Training data
        test (pd.DataFrame): Test data
        target_col (str): Target column name
    
    Returns:
        train_processed (pd.DataFrame): Processed training data
        test_processed (pd.DataFrame): Processed test data
        target_encoder: Target encoder object for categorical encoding
    """
    # Create copies to avoid modifying original data
    train = train.copy()
    test = test.copy()

    # Handle missing values in numerical columns
    numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 
                      'Guest_Popularity_percentage', 'Number_of_Ads']
    for col in numerical_cols:
        # Compute median from train and apply to both
        median_value = train[col].median()
        train[col] = train[col].fillna(median_value)
        test[col] = test[col].fillna(median_value)

    # Cap popularity percentages
    train['Host_Popularity_percentage'] = train['Host_Popularity_percentage'].clip(upper=100)
    train['Guest_Popularity_percentage'] = train['Guest_Popularity_percentage'].clip(upper=100)
    test['Host_Popularity_percentage'] = test['Host_Popularity_percentage'].clip(upper=100)
    test['Guest_Popularity_percentage'] = test['Guest_Popularity_percentage'].clip(upper=100)

    # Replace 0 in Episode_Length_minutes
    median_length = train['Episode_Length_minutes'].median()
    train['Episode_Length_minutes'] = train['Episode_Length_minutes'].replace(0, median_length)
    test['Episode_Length_minutes'] = test['Episode_Length_minutes'].replace(0, median_length)

    # Time-based features
    train['Is_Weekend'] = train['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    test['Is_Weekend'] = test['Publication_Day'].isin(['Saturday', 'Sunday']).astype(int)
    # Publication_Time has values like 'Morning', 'Night', etc., so we map to Is_Evening
    train['Is_Evening'] = train['Publication_Time'].apply(lambda x: 1 if x == 'Night' else 0)
    test['Is_Evening'] = test['Publication_Time'].apply(lambda x: 1 if x == 'Night' else 0)

    # Interaction features
    train['Host_Guest_Popularity_Interaction'] = (train['Host_Popularity_percentage'] * 
                                                 train['Guest_Popularity_percentage'])
    test['Host_Guest_Popularity_Interaction'] = (test['Host_Popularity_percentage'] * 
                                                test['Guest_Popularity_percentage'])
    train['Length_Popularity'] = (train['Episode_Length_minutes'] * 
                                 train['Host_Popularity_percentage'])
    test['Length_Popularity'] = (test['Episode_Length_minutes'] * 
                                test['Host_Popularity_percentage'])

    # Target encoding for high-cardinality categoricals
    categorical_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 
                        'Episode_Sentiment', 'Publication_Day', 'Publication_Time']
    target_encoder = TargetEncoder(cols=categorical_cols)
    
    # Fit the encoder on training data only (where target is not NaN)
    target_encoder.fit(train[categorical_cols], train[target_col])
    
    # Transform both train and test
    train_encoded = target_encoder.transform(train[categorical_cols])
    test_encoded = target_encoder.transform(test[categorical_cols])
    
    # Update the categorical columns in train and test
    train[categorical_cols] = train_encoded
    test[categorical_cols] = test_encoded

    # Drop unnecessary columns
    train_processed = train.drop(columns=['id'])
    test_processed = test.drop(columns=['id'])

    return train_processed, test_processed, target_encoder


# Apply preprocessing
train_processed, test_processed, target_encoder = preprocess_data(train, test)
print("Processed Train Shape:", train_processed.shape)
print("Processed Test Shape:", test_processed.shape)


# Transform target to handle skewness
y = np.log1p(train_processed['Listening_Time_minutes'])
X = train_processed.drop(columns=['Listening_Time_minutes'])

# Test features
X_test = test_processed


# Objective function for XGBoost
def objective_xgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    model = XGBRegressor(**params, random_state=RANDOM_SEED)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rmses = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)  # Reverse log-transform
        y_val_original = np.expm1(y_val)
        rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))
        rmses.append(rmse)
    return np.mean(rmses)

# Objective function for LightGBM
def objective_lgb(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
    }
    model = LGBMRegressor(**params, random_state=RANDOM_SEED, verbose=-1)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rmses = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        y_pred_log = model.predict(X_val)
        y_pred = np.expm1(y_pred_log)
        y_val_original = np.expm1(y_val)
        rmse = np.sqrt(mean_squared_error(y_val_original, y_pred))
        rmses.append(rmse)
    return np.mean(rmses)

# Tune XGBoost
study_xgb = optuna.create_study(direction='minimize')
study_xgb.optimize(objective_xgb, n_trials=20)
print('Best XGBoost params:', study_xgb.best_params)
print('Best XGBoost RMSE:', study_xgb.best_value)

# Tune LightGBM
study_lgb = optuna.create_study(direction='minimize')
study_lgb.optimize(objective_lgb, n_trials=20)
print('Best LightGBM params:', study_lgb.best_params)
print('Best LightGBM RMSE:', study_lgb.best_value)


# Train final models
xgb_model = XGBRegressor(**study_xgb.best_params, random_state=RANDOM_SEED)
lgb_model = LGBMRegressor(**study_lgb.best_params, random_state=RANDOM_SEED, verbose=-1)

xgb_model.fit(X, y)
lgb_model.fit(X, y)


# Predict with both models
xgb_pred_log = xgb_model.predict(X_test)
lgb_pred_log = lgb_model.predict(X_test)

# Ensemble: average the predictions
final_pred_log = (xgb_pred_log + lgb_pred_log) / 2
final_pred = np.expm1(final_pred_log)  # Reverse log-transform

# Clip predictions to avoid negative values
final_pred = np.clip(final_pred, 0, None)


# Create submission DataFrame
submission = pd.DataFrame({
    'id': test['id'],
    'Listening_Time_minutes': final_pred
})

# Save submission
submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")




