# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
# import lightgbm as lgb  # Example model
from sklearn.ensemble import RandomForestRegressor # Chosen model
# import xgboost as xgb   # Example model
# import optuna  # For hyperparameter optimization (commented out in final run)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold

# Improve display for DataFrames in Jupyter environment
from IPython.display import display

import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/kaggle/input/playground-series-s5e4/train.csv', usecols=lambda x: x != 'id')
test = pd.read_csv('/kaggle/input/playground-series-s5e4/test.csv', usecols=lambda x: x != 'id')
sample_submission = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')


print(f"Initial train shape: {train.shape}")
print(f"Initial test shape: {test.shape}")


print("\n--- Exploratory Data Analysis ---")


print("\nData Types (Train):\n", train.dtypes)


print("\nSummary Statistics (Train):\n")
display(train.describe())


print("\nMissing Values (Train):\n", train.isnull().sum())

# Visualize missing values
plt.figure(figsize=(10, 4))
sns.heatmap(train.isnull(), cbar=False, cmap='viridis')
plt.title("Missing Values Heatmap (Train)")
plt.show()


# Select numerical columns for outlier check (excluding target for now, but can be added)
numerical_cols_for_outliers = [
    'Episode_Length_minutes',
    'Host_Popularity_percentage',
    'Guest_Popularity_percentage',
    'Number_of_Ads'
]

# Create box plots - handle potential NaN values before plotting
plt.figure(figsize=(15, 4))
for i, col in enumerate(numerical_cols_for_outliers):
    plt.subplot(1, 4, i + 1) 
    sns.boxplot(y=train[col].dropna())
    plt.title(f'Box Plot of {col}')
    plt.ylabel(col)

plt.tight_layout()
plt.show()


print("\n--- Identifying Outliers using IQR ---")

for col in numerical_cols_for_outliers:
    # Calculate Q1 (25th percentile) and Q3 (75th percentile)
    # Drop NaNs before calculating quantiles to avoid errors/warnings
    Q1 = train[col].dropna().quantile(0.25)
    Q3 = train[col].dropna().quantile(0.75)
    IQR = Q3 - Q1

    # Define outlier bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Identify outliers
    # Ensure we only check non-NaN values
    outliers = train[(train[col].notna()) & ((train[col] < lower_bound) | (train[col] > upper_bound))]

    print(f"\nColumn: {col}")
    print(f"  Q1: {Q1:.2f}, Q3: {Q3:.2f}, IQR: {IQR:.2f}")
    print(f"  Lower Bound (Potential Outliers <): {lower_bound:.2f}")
    print(f"  Upper Bound (Potential Outliers >): {upper_bound:.2f}")
    print(f"  Number of potential outliers found: {len(outliers)}")


print("\n--- Data Cleaning ---")


print("Imputing missing values...")
train_median_length = train['Episode_Length_minutes'].median()
train_mean_guest_pop = train['Guest_Popularity_percentage'].mean()
train_median_ads = train['Number_of_Ads'].median() # Calculate before loop

for df in [train, test]:
    df['Episode_Length_minutes'] = df['Episode_Length_minutes'].fillna(train_median_length)
    df['Guest_Popularity_percentage'] = df['Guest_Popularity_percentage'].fillna(train_mean_guest_pop)
    # Ensure 'Number_of_Ads' exists before imputing (it might only be missing in train)
    if 'Number_of_Ads' in df.columns:
         df['Number_of_Ads'] = df['Number_of_Ads'].fillna(train_median_ads)

# Verify imputation
missing_df = pd.DataFrame({'Train': train.isnull().sum(), 'Test': test.isnull().sum()})
missing_df = missing_df.fillna(0)  # Avoid NaNs in display formatting
missing_df


print("\nApplying outlier capping (95th percentile)...")
# Based on EDA/IQR analysis, decide which columns need capping.
outlier_cols_to_cap = ['Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']
for col in outlier_cols_to_cap:
     # Ensure column exists in both dataframes for capping
    if col in train.columns and col in test.columns:
        # Using 95th percentile capping as in your original code
        threshold = train[col].quantile(0.95)
        print(f"Capping {col} at 95th percentile: {threshold:.2f}")
        train[col] = train[col].clip(upper=threshold)
        test[col] = test[col].clip(upper=threshold) # Use train threshold for test set
    else:
        print(f"Column {col} not found for capping, skipping.")


print("\nConverting categorical columns to 'category' type...")
categorical_cols_initial = ['Podcast_Name', 'Episode_Title', 'Publication_Day', 'Publication_Time', 'Episode_Sentiment', 'Genre']
for df in [train, test]:
    for col in categorical_cols_initial:
        if col in df.columns:
             df[col] = df[col].astype('category')


num_duplicates = train.duplicated().sum()
print(f"\nNumber of duplicate rows in train before removal: {num_duplicates}")
if num_duplicates > 0:
    train = train.drop_duplicates()
    print(f"Train shape after removing duplicates: {train.shape}")
else:
    print("No duplicate rows found in train.")

# Display cleaned data head
print("\nCleaned Train Data Head:")
display(train.head())


print("\n--- Feature Engineering ---")

# Define numerical columns for transformations (original ones before poly features)
numerical_cols = ['Episode_Length_minutes', 'Host_Popularity_percentage', 'Guest_Popularity_percentage', 'Number_of_Ads']


print("Generating Polynomial Features (degree=2)...")
poly = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)

# Fit on training data numerical columns
poly.fit(train[numerical_cols])

# Transform both train and test
poly_features_train = poly.transform(train[numerical_cols])
poly_features_test = poly.transform(test[numerical_cols])


# Create DataFrame for polynomial features
poly_feature_names = poly.get_feature_names_out(numerical_cols)

# Identify only the new (non-original) features
original_feature_set = set(numerical_cols)
new_poly_features = [f for f in poly_feature_names if f not in original_feature_set]

# Create DataFrames for just the new features
poly_df_train = pd.DataFrame(poly_features_train, columns=poly_feature_names, index=train.index)[new_poly_features]
poly_df_test = pd.DataFrame(poly_features_test, columns=poly_feature_names, index=test.index)[new_poly_features]

# Concatenate polynomial-only features
train = pd.concat([train, poly_df_train], axis=1)
test = pd.concat([test, poly_df_test], axis=1)


print(f"Train shape after adding polynomial features: {train.shape}")
print(f"Test shape after adding polynomial features: {test.shape}")


print("\nExtracting Date/Time Features...")
publication_time_mapping = {'Morning': 8, 'Afternoon': 14, 'Evening': 19, 'Night': 23} # Approximate hours

for df in [train, test]:
    # Day of Week (Monday=0, Sunday=6)
    df['Publication_DayOfWeek'] = pd.to_datetime(df['Publication_Day'].astype(str), format='%A').dt.dayofweek
    # Approximate Hour based on Time category
    df['Publication_Hour'] = df['Publication_Time'].map(publication_time_mapping).astype(float).fillna(12)
    # Weekend Flag
    df['Is_Weekend'] = df['Publication_DayOfWeek'].isin([5, 6]).astype(int)


print("\nApplying Target Encoding...")
# Note: 'Podcast_Name' and 'Episode_Title' have very high cardinality. Target encoding might be noisy or lead to overfitting.
# Consider frequency encoding, hashing, or dropping them if target encoding doesn't improve results.
target_encode_cols = ['Podcast_Name', 'Episode_Title', 'Genre', 'Episode_Sentiment', 'Publication_Day', 'Publication_Time']
global_mean = train['Listening_Time_minutes'].mean() # Calculate global mean for handling unseen categories in test

for col in target_encode_cols:
    if col in train.columns and col in test.columns: # Check if column exists
        print(f"Target encoding: {col}")
        target_mean_map = train.groupby(col)['Listening_Time_minutes'].mean()
        train[f'{col}_target_encoded'] = train[col].map(target_mean_map).astype(float)
        test[f'{col}_target_encoded'] = test[col].map(target_mean_map).astype(float)


        # Fill NaNs created during mapping (e.g., categories present only in test) with the global mean
        train[f'{col}_target_encoded'] = train[f'{col}_target_encoded'].fillna(global_mean)
        test[f'{col}_target_encoded'] = test[f'{col}_target_encoded'].fillna(global_mean)
    else:
         print(f"Skipping target encoding for {col} as it's not in both train/test")


print("\nApplying Feature Scaling (StandardScaler)...")

# Identify all numerical columns to scale (original + engineered)

# Rebuild the final numerical list (no duplicates!)
numerical_cols_all = (
    numerical_cols +
    new_poly_features +
    ['Publication_DayOfWeek', 'Publication_Hour'] +
    [f'{col}_target_encoded' for col in target_encode_cols if f'{col}_target_encoded' in train.columns]
)

# Ensure no NaNs exist in these columns before scaling (should have been handled earlier, but double-check)
for col in numerical_cols_all:
     if train[col].isnull().any():
         print(f"Warning: NaN found in {col} before scaling in train. Imputing with median.")
         train[col] = train[col].fillna(train[col].median())
     if test[col].isnull().any():
         print(f"Warning: NaN found in {col} before scaling in test. Imputing with train median.")
         test[col] = test[col].fillna(train[col].median()) # Use train median

scaler = StandardScaler()

# Fit scaler ONLY on the training data
scaler.fit(train[numerical_cols_all])

# Transform both training and test data
train[numerical_cols_all] = scaler.transform(train[numerical_cols_all])
test[numerical_cols_all] = scaler.transform(test[numerical_cols_all])

print("Scaling complete.")
print("\nTrain Data Head after Feature Engineering:")
display(train.head())
print("\nTest Data Head after Feature Engineering:")
display(test.head())


print("\n--- Data Splitting ---")

features = [col for col in numerical_cols_all if col in train.columns]
X = train[features]
y = train['Listening_Time_minutes']

# Stratify by quantiles
try:
    y_quantiles = pd.qcut(y, q=10, labels=False, duplicates='drop')
except ValueError:
    try:
        y_quantiles = pd.qcut(y, q=5, labels=False, duplicates='drop')
    except ValueError:
        print("Warning: Stratification failed. Proceeding without it.")
        y_quantiles = None

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y_quantiles
)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")



# # Define objective functions for Optuna
# def objective_lgbm(trial):
#     params = {
#         'objective': 'regression',
#         'metric': 'rmse',
#         'verbosity': -1,
#         'boosting_type': trial.suggest_categorical('boosting_type', ['gbdt', 'dart', 'goss']),
#         'num_leaves': trial.suggest_int('num_leaves', 2, 256),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#     }

#     model = lgb.LGBMRegressor(**params)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_val)
#     rmse = mean_squared_error(y_val, preds, squared=False)
#     return rmse

# def objective_rf(trial):
#     params = {
#         'n_estimators': trial.suggest_int('n_estimators', 50, 500),
#         'max_depth': trial.suggest_int('max_depth', 3, 30),
#         'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
#     }

#     model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)  # n_jobs=-1 for parallel processing
#     model.fit(X_train, y_train)
#     preds = model.predict(X_val)
#     rmse = mean_squared_error(y_val, preds, squared=False)
#     return rmse

# def objective_xgb(trial):
#     params = {
#         'objective': 'reg:squarederror',
#         'eval_metric': 'rmse',
#         'verbosity': 0,
#         'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
#         'max_depth': trial.suggest_int('max_depth', 3, 10),
#         'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
#     }

#     model = xgb.XGBRegressor(**params)
#     model.fit(X_train, y_train)
#     preds = model.predict(X_val)
#     rmse = mean_squared_error(y_val, preds, squared=False)
#     return rmse

# # Optimize hyperparameters with Optuna
# study_lgbm = optuna.create_study(direction='minimize')
# study_lgbm.optimize(objective_lgbm, n_trials=10)  # Reduced n_trials for demonstration
# best_params_lgbm = study_lgbm.best_params

# study_rf = optuna.create_study(direction='minimize')
# study_rf.optimize(objective_rf, n_trials=10)
# best_params_rf = study_rf.best_params

# study_xgb = optuna.create_study(direction='minimize')
# study_xgb.optimize(objective_xgb, n_trials=10)
# best_params_xgb = study_xgb.best_params

# # Train models with best hyperparameters
# lgbm_model = lgb.LGBMRegressor(**best_params_lgbm)
# lgbm_model.fit(X_train, y_train)

# rf_model = RandomForestRegressor(**best_params_rf, random_state=42, n_jobs=-1)
# rf_model.fit(X_train, y_train)

# xgb_model = xgb.XGBRegressor(**best_params_xgb)
# xgb_model.fit(X_train, y_train)

# # Generate predictions
# lgbm_preds = lgbm_model.predict(X_val)
# rf_preds = rf_model.predict(X_val)
# xgb_preds = xgb_model.predict(X_val)

# # Calculate RMSE
# lgbm_rmse = np.sqrt(mean_squared_error(y_val, lgbm_preds))
# rf_rmse = np.sqrt(mean_squared_error(y_val, rf_preds))
# xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_preds))

# # Print RMSE for each model
# print(f"LightGBM RMSE: {lgbm_rmse}")
# print(f"Random Forest RMSE: {rf_rmse}")
# print(f"XGBoost RMSE: {xgb_rmse}")


%%time

# Same parameters (some are not used by sklearn and will be ignored)
sklearn_rf_params = {
    'n_estimators': 456,
    'max_depth': 24,
    'max_features': 1.0,
    'random_state': 42,
    'n_jobs': -1  # Use all CPU cores
}

# Fit model on CPU
sklearn_rf = RandomForestRegressor(**sklearn_rf_params)
sklearn_rf.fit(X_train, y_train)

print("Scikit-learn Random Forest training complete (CPU).")



# Predictions on test dataset
y_pred_val = sklearn_rf.predict(X_val)

rmse = mean_squared_error(y_val, y_pred_val, squared=False)
print(f"Validation RMSE: {rmse:.4f}")


# Ensure both y_val and y_pred_val are aligned
df_plot = pd.DataFrame({
    'Actual': y_val.reset_index(drop=True),
    'Predicted': pd.Series(y_pred_val)
})

window = 100
df_plot['Actual_smooth'] = df_plot['Actual'].rolling(window).mean()
df_plot['Predicted_smooth'] = df_plot['Predicted'].rolling(window).mean()

plt.figure(figsize=(14, 6))
plt.plot(df_plot['Actual_smooth'], label='Actual (Smoothed)', linewidth=2.5, color='steelblue')
plt.plot(df_plot['Predicted_smooth'], label='Predicted (Smoothed)', linewidth=2.5, linestyle='--', color='tomato')

plt.title('ğŸ“ˆ Smoothed Predictions vs Actual', fontsize=16)
plt.xlabel('Sample Index')
plt.ylabel('Listening Time (minutes)')
plt.legend()
plt.tight_layout()
plt.show()


print("\n--- Preparing Test Data for Prediction ---")

# Select the same features from the processed test set as used for training
# Ensure columns are in the same order as X_train
X_test = test[features] # 'features' list was defined during splitting

# Verify shapes and check for any remaining missing values (should be none after cleaning/engineering)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")
print("\nMissing values check in final X_test:")
print(X_test.isnull().sum().sum()) # Sum of all missing values should be 0


print("\n--- Prediction and Submission ---")

# Generate predictions on the test set
print("Generating predictions on the test set...")
test_predictions = sklearn_rf.predict(X_test)

# Create the submission DataFrame using the original sample_submission 'id' column
submission_df = pd.read_csv('/kaggle/input/playground-series-s5e4/sample_submission.csv')
submission_df['Listening_Time_minutes'] = test_predictions
submission_df.to_csv('submission.csv', index=False)

print(f"Submission file '{submission_df}' created successfully.")
print("\nSubmission File Head:")
display(submission_df.head())




