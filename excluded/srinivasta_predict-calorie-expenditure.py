
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold # Use KFold for cross-validation
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_log_error
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Import StandardScaler
import gc # Import garbage collection

# Define RMSLE function
def rmsle(y_true, y_pred):
    """
    Calculates the Root Mean Squared Logarithmic Error (RMSLE).

    Args:
        y_true: Array-like of true target values.
        y_pred: Array-like of predicted target values.

    Returns:
        The RMSLE score.
    """
    # Ensure predictions are non-negative
    # Add 1 to y_true and y_pred before taking log to handle zero values
    return np.sqrt(mean_squared_log_error(y_true + 1, y_pred + 1))


# Load data
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
    print("Data loaded successfully.")
except FileNotFoundError:
    print("Ensure train.csv and test.csv are in the correct directory.")
    exit() # Exit if data not found

# Store test IDs for submission
test_ids = test_df['id']

# Handle negative 'Calories' (apply shift to both train and test if predicting shifted target)
# We apply shift to training target for modeling, but evaluate and submit on original scale
# It's safer to only apply the shift to the training target here.
shift_value = abs(train_df['Calories'].min()) + 100
train_df['Calories_shifted'] = train_df['Calories'] + shift_value

# --- Feature Engineering ---

# 1. BMI (Body Mass Index) - Using height in meters
train_df['BMI'] = train_df['Weight'] / ((train_df['Height'] / 100) ** 2)
test_df['BMI'] = test_df['Weight'] / ((test_df['Height'] / 100) ** 2)

# 2. BMR (Basal Metabolic Rate) Estimation (Vectorized)
train_df['BMR'] = np.where(train_df['Sex'] == 'male',
                           (10 * train_df['Weight']) + (6.25 * train_df['Height']) - (5 * train_df['Age']) + 5,
                           (10 * train_df['Weight']) + (6.25 * train_df['Height']) - (5 * train_df['Age']) - 161)

test_df['BMR'] = np.where(test_df['Sex'] == 'male',
                          (10 * test_df['Weight']) + (6.25 * test_df['Height']) - (5 * test_df['Age']) + 5,
                          (10 * test_df['Weight']) + (6.25 * test_df['Height']) - (5 * test_df['Age']) - 161)

# 3. Exercise Intensity (Vectorized)
conditions_train = [
    (train_df['Heart_Rate'] > 160) & (train_df['Body_Temp'] > 38),
    (train_df['Heart_Rate'] > 120) & (train_df['Body_Temp'] > 37.5)
]
conditions_test = [
    (test_df['Heart_Rate'] > 160) & (test_df['Body_Temp'] > 38),
    (test_df['Heart_Rate'] > 120) & (test_df['Body_Temp'] > 37.5)
]
choices = ['high', 'moderate']

train_df['Exercise_Intensity'] = np.select(conditions_train, choices, default='low')
test_df['Exercise_Intensity'] = np.select(conditions_test, choices, default='low')

# 4. Interaction Terms
train_df['Duration_Heart_Rate_Interaction'] = train_df['Duration'] * train_df['Heart_Rate']
train_df['Duration_BodyTemp_Interaction'] = train_df['Duration'] * train_df['Body_Temp']
train_df['HeartRate_BodyTemp_Interaction'] = train_df['Heart_Rate'] * train_df['Body_Temp']

test_df['Duration_Heart_Rate_Interaction'] = test_df['Duration'] * test_df['Heart_Rate']
test_df['Duration_BodyTemp_Interaction'] = test_df['Duration'] * test_df['Body_Temp']
test_df['HeartRate_BodyTemp_Interaction'] = test_df['Heart_Rate'] * test_df['Body_Temp']

# 5. Activity Level based on Exercise Intensity and Duration
intensity_mapping = {'low': 1, 'moderate': 2, 'high': 3}
train_df['Activity_Level'] = train_df['Exercise_Intensity'].map(intensity_mapping) * train_df['Duration']
test_df['Activity_Level'] = test_df['Exercise_Intensity'].map(intensity_mapping) * test_df['Duration']

# 6. Heart Rate Zones (Example - simplified zones)
# Vectorize this function as well
def heart_rate_zone_vec(heart_rate):
    conditions = [
        heart_rate > 170,
        heart_rate > 140,
        heart_rate > 110
    ]
    choices = ['very hard', 'hard', 'moderate zone']
    return np.select(conditions, choices, default='light zone')

train_df['Heart_Rate_Zone'] = heart_rate_zone_vec(train_df['Heart_Rate'])
test_df['Heart_Rate_Zone'] = heart_rate_zone_vec(test_df['Heart_Rate'])


# 7. Temperature Deviation from Normal
normal_temp = 37.0
train_df['Temp_Deviation'] = abs(train_df['Body_Temp'] - normal_temp)
test_df['Temp_Deviation'] = abs(test_df['Body_Temp'] - normal_temp)

# 8. BMI Categories (Vectorized)
def bmi_category_vec(bmi):
    conditions = [
        bmi < 18.5,
        bmi < 25,
        bmi < 30
    ]
    choices = ['underweight', 'normal', 'overweight']
    return np.select(conditions, choices, default='obese')

train_df['BMI_Category'] = bmi_category_vec(train_df['BMI'])
test_df['BMI_Category'] = bmi_category_vec(test_df['BMI'])

# 9. Age Group (Vectorized)
def age_group_vec(age):
    conditions = [
        age < 30,
        age < 50
    ]
    choices = ['young adult', 'adult']
    return np.select(conditions, choices, default='senior')

train_df['Age_Group'] = age_group_vec(train_df['Age'])
test_df['Age_Group'] = age_group_vec(test_df['Age'])


# --- Data Preprocessing ---

# Define categorical and numerical features, including the new ones
# Make sure 'id' and original 'Calories' are not included in features
categorical_features = ['Sex', 'Exercise_Intensity', 'Heart_Rate_Zone', 'BMI_Category', 'Age_Group']
numerical_features = ['Age', 'Height', 'Weight', 'Duration', 'Heart_Rate', 'Body_Temp', 'BMI', 'BMR',
                        'Duration_Heart_Rate_Interaction', 'Duration_BodyTemp_Interaction',
                        'HeartRate_BodyTemp_Interaction', 'Activity_Level', 'Temp_Deviation']

# Check if all numerical features exist in both train and test
if not all(f in train_df.columns for f in numerical_features):
    print("Error: Some numerical features missing in training data.")
    print([f for f in numerical_features if f not in train_df.columns])
    exit()
if not all(f in test_df.columns for f in numerical_features):
    print("Error: Some numerical features missing in test data.")
    print([f for f in numerical_features if f not in test_df.columns])
    exit()

# Check if all categorical features exist in both train and test
if not all(f in train_df.columns for f in categorical_features):
    print("Error: Some categorical features missing in training data.")
    print([f for f in categorical_features if f not in train_df.columns])
    exit()
if not all(f in test_df.columns for f in categorical_features):
    print("Error: Some categorical features missing in test data.")
    print([f for f in categorical_features if f not in test_df.columns])
    exit()


# One-hot encoding for categorical features
# Fit on the combined training and test data to handle unseen categories in test if handle_unknown='ignore' isn't sufficient,
# OR fit only on train and use handle_unknown='ignore'. Fitting on train + test is generally safer for consistent columns.
# Let's fit on train data only and use handle_unknown='ignore' as originally planned.
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
encoder.fit(train_df[categorical_features]) # Fit only on training data

encoded_train = encoder.transform(train_df[categorical_features])
encoded_test = encoder.transform(test_df[categorical_features])

encoded_train_df = pd.DataFrame(encoded_train, columns=encoder.get_feature_names_out(categorical_features), index=train_df.index)
encoded_test_df = pd.DataFrame(encoded_test, columns=encoder.get_feature_names_out(categorical_features), index=test_df.index)

# Concatenate the encoded features with the numerical features
X = pd.concat([train_df[numerical_features], encoded_train_df], axis=1)
X_test_processed = pd.concat([test_df[numerical_features], encoded_test_df], axis=1) # Features for final test prediction
y = train_df['Calories_shifted'] # Use the shifted target for training

# Scale numerical features
scaler = StandardScaler()
# Fit and transform the scaler on the numerical features of the training data (X)
X[numerical_features] = scaler.fit_transform(X[numerical_features])
# Transform the numerical features of the test data (X_test_processed) using the fitted scaler
X_test_processed[numerical_features] = scaler.transform(X_test_processed[numerical_features])


# Clean up memory
del train_df, test_df, encoded_train, encoded_test, encoded_train_df, encoded_test_df
gc.collect()

print("Feature Engineering and Preprocessing complete.")
print(f"Shape of training features (X): {X.shape}")
print(f"Shape of test features (X_test_processed): {X_test_processed.shape}")

# --- Model Training with K-Fold Cross-Validation ---

n_splits = 5 # Number of folds for CV
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

rmsle_scores = []
# Array to store predictions for the test set from each fold
test_preds_folds = np.zeros(len(X_test_processed))

# --- CatBoost Hyperparameters ---
# These are example parameters. RIGOROUS TUNING IS REQUIRED TO OPTIMIZE THIS.
# Use libraries like Optuna or Hyperopt for proper tuning.
catboost_params = {
    'iterations': 2000, # Increased iterations
    'learning_rate': 0.02, # Fine-tuned learning rate
    'depth': 9, # Increased depth
    'l2_leaf_reg': 3,
    'loss_function': 'RMSE',
    'eval_metric': 'RMSE',
    'random_seed': 42,
    'verbose': 200, # Print progress every 200 iterations
    'early_stopping_rounds': 150, # Stop early if validation metric doesn't improve
    'allow_writing_files': False # Prevent writing debug files
    # Add more parameters like 'border_count', 'bagging_temperature', 'random_strength' etc.
}


print(f"\nStarting {n_splits}-Fold Cross-Validation...")

for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
    print(f"\n--- Fold {fold+1}/{n_splits} ---")

    X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    # Initialize CatBoost model for the fold
    model = CatBoostRegressor(**catboost_params)

    # Train the model on the training fold
    # Pass validation data to monitor performance and enable early stopping
    model.fit(X_train_fold, y_train_fold,
              eval_set=[(X_val_fold, y_val_fold)],
              verbose=catboost_params['verbose'])


    # Make predictions on the validation fold
    y_pred_val_fold = model.predict(X_val_fold)

    # Reverse the shift for evaluation on the original scale
    y_val_original_scale = y_val_fold - shift_value
    y_pred_val_original_scale = y_pred_val_fold - shift_value
    y_pred_val_original_scale = np.clip(y_pred_val_original_scale, a_min=0, a_max=None)

    # Calculate RMSLE for the fold
    fold_rmsle = rmsle(y_val_original_scale, y_pred_val_original_scale)
    print(f"Fold {fold+1} Validation RMSLE: {fold_rmsle}")
    rmsle_scores.append(fold_rmsle)

    # Make predictions on the test set with the model from this fold
    test_preds_folds += model.predict(X_test_processed) / n_splits # Average predictions over folds

    # Clean up memory
    del model, X_train_fold, X_val_fold, y_train_fold, y_val_fold, y_pred_val_fold, y_val_original_scale, y_pred_val_original_scale
    gc.collect()


print(f"\n--- Cross-Validation Results ---")
print(f"Average Validation RMSLE: {np.mean(rmsle_scores)}")
print(f"Std Deviation of Validation RMSLE: {np.std(rmsle_scores)}")


# --- Final Predictions and Submission ---

# The test_preds_folds array now holds the averaged predictions from the CV models.
# This is a simple form of ensembling (averaging k models).

# Apply the final shift reversal and clipping to the averaged test predictions
final_test_predictions = test_preds_folds - shift_value
final_test_predictions = np.clip(final_test_predictions, a_min=0, a_max=None)


# Create submission file
submission = pd.DataFrame({'id': test_ids, 'Calories': final_test_predictions})
submission.to_csv('submission.csv', index=False)

print("\n--- Submission File Created ---")
print("Predictions saved to submission.csv")
print(submission.head())

