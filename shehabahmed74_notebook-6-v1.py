# Install required libraries
!pip install autograd==1.7.0 autograd-gamma==0.5.0 interface_meta==1.3.0 formulaic==1.0.2 lifelines==0.30.0
!pip install optuna xgboost lightgbm catboost pandas numpy scikit-learn

# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from lifelines.utils import concordance_index
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.feature_selection import SelectKBest, f_regression
import optuna
import os

# Step 1: Load and Preprocess Dataset
def load_and_preprocess_data(train_path, test_path):
    train_data = pd.read_csv(train_path)
    test_data = pd.read_csv(test_path)

    # Separate numeric and categorical columns
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = train_data.select_dtypes(include=["object"]).columns.tolist()

    # Handle missing values
    train_data[numeric_cols] = train_data[numeric_cols].fillna(train_data[numeric_cols].median())
    test_data = test_data.reindex(columns=train_data.columns, fill_value=0)
    test_data[numeric_cols] = test_data[numeric_cols].fillna(train_data[numeric_cols].median())

    train_data[categorical_cols] = train_data[categorical_cols].fillna(train_data[categorical_cols].mode().iloc[0])
    test_data[categorical_cols] = test_data[categorical_cols].fillna(train_data[categorical_cols].mode().iloc[0])

    # Key columns
    key_columns = ["ID", "efs", "efs_time"]
    key_train_data = train_data[key_columns]
    key_test_data = test_data[["ID"]]

    # One-hot encode categorical variables
    train_data = pd.get_dummies(train_data.drop(columns=key_columns), drop_first=True)
    test_data = pd.get_dummies(test_data.drop(columns=["ID"]), drop_first=True)

    train_data, test_data = train_data.align(test_data, join="left", axis=1)
    test_data.fillna(0, inplace=True)

    # Reattach key columns
    train_data = pd.concat([key_train_data, train_data], axis=1)
    test_data = pd.concat([key_test_data, test_data], axis=1)

    train_data = shuffle(train_data, random_state=42)
    return train_data, test_data

# Step 2: Prepare Features and Targets
def prepare_features_and_targets(train_data, test_data):
    X = train_data.drop(columns=["ID", "efs", "efs_time"])
    y = train_data["efs_time"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    selector = SelectKBest(f_regression, k=min(10, X.shape[1]))
    X_selected = selector.fit_transform(X_scaled, y)

    X_test = test_data.drop(columns=["ID"])
    X_test_scaled = scaler.transform(X_test)
    X_test_selected = selector.transform(X_test_scaled)

    return X_selected, y, X_test_selected, test_data["ID"]

# Step 3: Define Objective Function for Optuna
def objective(trial, X, y):
    model_type = trial.suggest_categorical('model_type', ['xgboost', 'lightgbm', 'catboost'])
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.1),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_uniform('subsample', 0.6, 1.0)
    }
    if model_type == 'xgboost':
        params['colsample_bytree'] = trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
        params['reg_alpha'] = trial.suggest_loguniform('reg_alpha', 1e-4, 1e-1)
        params['reg_lambda'] = trial.suggest_loguniform('reg_lambda', 1e-4, 1e-1)
        model = XGBRegressor(**params, random_state=42)
    elif model_type == 'lightgbm':
        params['colsample_bytree'] = trial.suggest_uniform('colsample_bytree', 0.6, 1.0)
        params['reg_alpha'] = trial.suggest_loguniform('reg_alpha', 1e-4, 1e-1)
        params['reg_lambda'] = trial.suggest_loguniform('reg_lambda', 1e-4, 1e-1)
        model = LGBMRegressor(**params, random_state=42)
    elif model_type == 'catboost':
        model = CatBoostRegressor(**params, random_state=42, verbose=0)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    c_indices = []
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        model.fit(X_train, y_train)
        val_predictions = model.predict(X_val)
        c_index = concordance_index(y_val, -val_predictions)
        c_indices.append(c_index)

    return np.mean(c_indices)

# Step 4: Hyperparameter Optimization
def optimize_hyperparameters(X, y, n_trials=100):
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)
    return study.best_params

# Step 5: Train Final Model and Make Predictions
def train_and_predict(X, y, X_test, best_params):
    if best_params['model_type'] == 'xgboost':
        model = XGBRegressor(**best_params, random_state=42)
    elif best_params['model_type'] == 'lightgbm':
        model = LGBMRegressor(**best_params, random_state=42)
    elif best_params['model_type'] == 'catboost':
        model = CatBoostRegressor(**best_params, random_state=42, verbose=0)
    
    model.fit(X, y)
    predictions = model.predict(X_test)
    return predictions

# Main Execution
train_path = "/kaggle/input/equity-post-HCT-survival-predictions/train.csv"
test_path = "/kaggle/input/equity-post-HCT-survival-predictions/test.csv"
train_data, test_data = load_and_preprocess_data(train_path, test_path)
X, y, X_test, test_ids = prepare_features_and_targets(train_data, test_data)

# Hyperparameter Optimization
best_params = optimize_hyperparameters(X, y, n_trials=100)
print("Best parameters:", best_params)

# Train Final Model and Make Predictions
predictions = train_and_predict(X, y, X_test, best_params)

# Step 1: Load the Submission File
submission = pd.read_csv("/kaggle/input/equity-post-HCT-survival-predictions/sample_submission.csv")

# Step 3: Save the Submission File
submission = pd.DataFrame({
    "ID": test_ids.astype(int),
    "prediction": predictions
})
submission.to_csv("/kaggle/working/submission.csv", index=False)

# Step 4: Print the Shape of Submission File
print("Sub shape:", submission.shape)

# Step 5: Display the First Few Rows
print(submission.head())

# Additional Evaluation Metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

def evaluate_model(X, y, best_params):
    if best_params['model_type'] == 'xgboost':
        model = XGBRegressor(**best_params, random_state=42)
    elif best_params['model_type'] == 'lightgbm':
        model = LGBMRegressor(**best_params, random_state=42)
    elif best_params['model_type'] == 'catboost':
        model = CatBoostRegressor(**best_params, random_state=42, verbose=0)
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    mae_scores



import os

# Step 3: Save the Submission File
submission = pd.DataFrame({
    "ID": test_ids.astype(int),
    "prediction": predictions
})

# Save the file and confirm its creation
submission_file_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_file_path, index=False)
print(f"Submission file saved to: {submission_file_path}")

# Check if the file exists
if os.path.exists(submission_file_path):
    print("Submission file exists.")
else:
    print("Error: Submission file does not exist!")

# Display the first few rows of the file
try:
    generated_submission = pd.read_csv(submission_file_path)
    print("Preview of the generated submission file:")
    print(generated_submission.head())
    print(f"Submission file shape: {generated_submission.shape}")
except Exception as e:
    print(f"Error reading the generated submission file: {e}")



print(f"Number of rows in test data: {test_data.shape[0]}")
print(f"Length of predictions: {len(predictions)}")
print(f"Length of test_ids: {len(test_ids)}")
# Check test data size
print(f"Number of rows in test data: {test_data.shape[0]}")

# Check predictions and IDs
print(f"Length of predictions: {len(predictions)}")
print(f"Length of test_ids: {len(test_ids)}")

# Debug alignment issues if lengths mismatch
if len(predictions) != len(test_ids):
    print("Error: Mismatch between predictions and test IDs!")



test_data_raw = pd.read_csv(test_path)
print(test_data_raw.head())
print(f"Test dataset shape: {test_data_raw.shape}")


