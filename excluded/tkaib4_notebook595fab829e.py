import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import cohen_kappa_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import joblib
import optuna
import warnings
warnings.filterwarnings('ignore')


TRAIN_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/train.csv'
TEST_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/test.csv'
TRAIN_TS_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/series_train.parquet'
TEST_TS_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/series_test.parquet'
SUBMISSION_PATH = '/kaggle/input/child-mind-institute-problematic-internet-use/sample_submission.csv'
OUTPUT_PATH = '/kaggle/working/'


# Define QWK function
def quadratic_weighted_kappa(y_true, y_pred):
    """Calculate quadratic weighted kappa."""
    # Ensure inputs are in the right format
    y_true = np.array(y_true, dtype=int)
    y_pred = np.round(np.array(y_pred)).astype(int)
    
    # Clip predictions to be within the range of observed values
    min_val, max_val = min(y_true), max(y_true)
    y_pred = np.clip(y_pred, min_val, max_val)
    
    return cohen_kappa_score(y_true, y_pred, weights='quadratic')


# Custom calibration function to optimize QWK
def optimize_predictions_for_qwk(y_true, y_pred, return_params=False):
    """Apply a linear transformation to predictions to maximize QWK."""
    def objective(params):
        a, b = params
        calibrated = a * y_pred + b
        return -quadratic_weighted_kappa(y_true, calibrated)
    
    # Starting with identity transformation
    initial_params = [1.0, 0.0]
    
    # Optimize the parameters
    result = minimize(objective, initial_params, method='Nelder-Mead')
    a, b = result.x
    
    if return_params:
        return a, b
    
    # Apply the transformation
    calibrated_preds = a * y_pred + b
    
    # Clip to observed range
    min_val, max_val = min(y_true), max(y_true)
    calibrated_preds = np.clip(calibrated_preds, min_val, max_val)
    
    return calibrated_preds


def create_sample_weights(X):
    weights = np.ones(len(X))

    # Feature importances (from your graph), normalized so they sum to 1
    feature_weights = {
        'Basic_Demos-Age': 0.1731,
        'PreInt_EduHx-computerinternet_hoursday': 0.13,
        'SDS-SDS_Total_Raw' : 0.0742,
        'SDS-SDS_Total_T': 0.0607,
        'Physical-Height': 0.0531,
        'Physical-Weight': 0.0461,
        'FGC-FGC_CU': 0.0342,
        'Basic_Demos-Sex': 0.0243,
    }

    # Normalize weights to sum to 1 or scale as needed
    total_importance = sum(feature_weights.values())
    for feature, importance in feature_weights.items():
        if feature in X.columns:
            scaled_importance = importance / total_importance
            feature_series = X[feature]
            # Mean/std deviation safely with skipna=True
            mean = feature_series.mean(skipna=True)
            std = feature_series.std(skipna=True)
            deviation = np.abs(feature_series - mean) / (std + 1e-8)
            weights += scaled_importance * deviation.fillna(0)
    
    return weights


# Load and prepare data
print("Loading data...")
data = pd.read_csv(TRAIN_PATH)
print(f"Original data shape: {data.shape}")
print(f"Number of NaN values in 'sii': {data['sii'].isna().sum()}")


# Remove rows where 'sii' is NaN
data_clean = data.dropna(subset=['sii'])
print(f"Data shape after removing NaN targets: {data_clean.shape}")


# Separate target variable
y = data_clean['sii'].astype(int)  # Ensure target is integer for QWK
print(f"Target values range: {y.min()} to {y.max()}")
print(f"Unique target values: {sorted(y.unique())}")


# Select features, excluding PCIAT columns
X_columns = [col for col in data_clean.columns if not col.startswith('PCIAT') and col != 'sii' and col != 'id']
X = data_clean[X_columns]
print(f"Number of features: {len(X_columns)}")


# Split data - use stratified split if possible to maintain distribution
# For regression, we can bin the target for stratification
try:
    from sklearn.model_selection import StratifiedShuffleSplit
    # Create bins for stratification
    bins = pd.qcut(y, q=min(10, len(y.unique())), labels=False, duplicates='drop')
    stratified_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_idx, test_idx in stratified_split.split(X, bins):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    print("Used stratified split to maintain target distribution")
except:
    # Fallback to regular split if stratification fails
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Used regular train-test split")



# Identify numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
print(f"Numeric features: {len(numeric_features)}")
print(f"Categorical features: {len(categorical_features)}")


transformer_list = [
    ('num', Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]), numeric_features)
]

if categorical_features:
    transformer_list.append(
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_features)
    )


preprocessor = ColumnTransformer(transformers=transformer_list)


# Define the objective function for Optuna optimization
def objective(trial):
    # Hyperparameters for GradientBoostingRegressor specifically tuned for QWK
    params = {
    'n_estimators': trial.suggest_int('n_estimators', 400, 700),
    'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.03, log=True),
    'max_depth': trial.suggest_int('max_depth', 3, 6),
    'min_samples_split': trial.suggest_int('min_samples_split', 5, 15),
    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 15),
    'subsample': trial.suggest_float('subsample', 0.5, 0.9),
    'max_features': trial.suggest_float('max_features', 0.5, 0.9),
    'loss': trial.suggest_categorical('loss', ['squared_error', 'huber', 'quantile']),
    'alpha': trial.suggest_float('alpha', 0.5, 0.9),
    'random_state': 42
}
    
    # Create model
    model = GradientBoostingRegressor(**params)
    
    # Create pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Use cross-validation with QWK optimization
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    qwk_scores = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(k_fold.split(X_train)):
        # Split data
        X_fold_train, X_fold_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        sample_weights = create_sample_weights(X_fold_train)
        
        # Fit model
        pipeline.fit(X_fold_train, y_fold_train, model__sample_weight=sample_weights)
        
        # Get predictions
        y_pred = pipeline.predict(X_fold_val)
        
        # Optimize predictions for QWK and evaluate
        y_pred_calibrated = optimize_predictions_for_qwk(y_fold_val, y_pred)
        qwk = quadratic_weighted_kappa(y_fold_val, y_pred_calibrated)
        qwk_scores.append(qwk)
        
        # Report intermediate value for pruning
        trial.report(-qwk, fold_idx)
        
        # Handle pruning based on the intermediate value
        if trial.should_prune():
            raise optuna.TrialPruned()
        
    mean_qwk = np.mean(qwk_scores)
    return -mean_qwk  # Negative because Optuna minimizes


# Run Optuna study
print("\nStarting Optuna optimization to maximize QWK...")
study = optuna.create_study(direction='minimize')  # Using minimize since we return -QWK
n_trials = 50  # Increase for better results if you have the compute resources
study.optimize(objective, n_trials=n_trials)


# Print optimization results
print("\nOptimization completed!")
print(f"Best trial: {study.best_trial.number}")
print(f"Best QWK: {-study.best_value:.4f}")
print("Best hyperparameters:")
for key, value in study.best_params.items():
    print(f"    {key}: {value}")


# Create and train the final model with best parameters
print("\nTraining final model with best parameters...")
best_params = study.best_params.copy()
best_model = GradientBoostingRegressor(**best_params)

final_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', best_model)
])


# Train on full training data
final_pipeline.fit(X_train, y_train)


test_data = pd.read_csv(TEST_PATH)
# need to feature engineer this as well and then do prediction


# Get predictions on test data
y_pred = final_pipeline.predict(X_test)



# Calibrate predictions to maximize QWK
print("\nCalibrating predictions to maximize QWK...")
y_pred_calibrated = optimize_predictions_for_qwk(y_test, y_pred)



# Evaluate both raw and calibrated predictions
qwk_raw = quadratic_weighted_kappa(y_test, y_pred)
qwk_calibrated = quadratic_weighted_kappa(y_test, y_pred_calibrated)
rmse_raw = np.sqrt(mean_squared_error(y_test, y_pred))
rmse_calibrated = np.sqrt(mean_squared_error(y_test, y_pred_calibrated))
r2_raw = r2_score(y_test, y_pred)
r2_calibrated = r2_score(y_test, y_pred_calibrated)

print("\nTest set evaluation:")
print(f"Raw predictions - QWK: {qwk_raw:.4f}, RMSE: {rmse_raw:.4f}, R²: {r2_raw:.4f}")
print(f"Calibrated predictions - QWK: {qwk_calibrated:.4f}, RMSE: {rmse_calibrated:.4f}, R²: {r2_calibrated:.4f}")


# Load test data
test_data = pd.read_csv(TEST_PATH)
test_ids = test_data['id']  # Save IDs for submission

# Select the same features as used in training
X_test_submission = test_data[X_columns]

# Get raw predictions using the trained pipeline
test_preds_raw = final_pipeline.predict(X_test_submission)

# We need to calibrate these predictions for optimal QWK
# For this, we'll use the training data that the model has already seen
calibration_preds = final_pipeline.predict(X_train)
calibration_params = optimize_predictions_for_qwk(y_train, calibration_preds, return_params=True)

# Apply calibration to test predictions
a, b = calibration_params
test_preds_calibrated = a * test_preds_raw + b

# Clip to the valid range (0-3 based on your training data)
test_preds_calibrated = np.clip(test_preds_calibrated, 0, 3)

# Round to integers since SII appears to be an integer score
test_preds_calibrated = np.round(test_preds_calibrated).astype(int)

# Create submission dataframe
submission = pd.DataFrame({
    'id': test_ids,
    'sii': test_preds_calibrated
})
import os
# Save submission file
submission_path = os.path.join(OUTPUT_PATH, "submission.csv")
submission.to_csv(submission_path, index=False)
print(f"Submission file saved to {submission_path}")











# # Feature importance analysis
# print("\nAnalyzing feature importance...")
# try:
#     # Get feature names after preprocessing
#     feature_names = []
#     for name, transformer, features in preprocessor.transformers_:
#         if name == 'cat':
#             # Get one-hot encoded feature names
#             encoder = transformer.named_steps['onehot']
#             cats = encoder.categories_
#             for i, feature in enumerate(features):
#                 feature_names.extend([f"{feature}_{cat}" for cat in cats[i]])
#         else:
#             feature_names.extend(features)
    
#     # Get feature importances
#     importances = final_pipeline.named_steps['model'].feature_importances_
#     indices = np.argsort(importances)[::-1]
    
#     # Print top features
#     print("\nTop 20 features by importance:")
#     top_n = min(20, len(feature_names))
#     for i in range(top_n):
#         try:
#             print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
#         except:
#             print(f"{i+1}. Feature #{indices[i]}: {importances[indices[i]]:.4f}")
    
#     # Plot feature importances
#     plt.figure(figsize=(12, 10))
#     top_indices = indices[:top_n]
#     plt.barh(range(top_n), importances[top_indices])
#     plt.yticks(range(top_n), [feature_names[i] if i < len(feature_names) else f"Feature #{i}" for i in top_indices])
#     plt.xlabel('Importance')
#     plt.title('Top 20 Feature Importances')
#     plt.tight_layout()
#     # plt.savefig('qwk_feature_importance.png')
#     # print("Feature importance plot saved as 'qwk_feature_importance.png'")
# except Exception as e:
#     print(f"Error analyzing feature importance: {str(e)}")


# # Visualize predictions
# plt.figure(figsize=(10, 6))
# plt.scatter(y_test, y_pred_calibrated, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
# plt.xlabel('Actual SII')
# plt.ylabel('Predicted SII (Calibrated)')
# plt.title(f'Actual vs Predicted SII (QWK: {qwk_calibrated:.4f})')
# plt.tight_layout()
# # plt.savefig('qwk_predictions.png')
# print("Prediction plot saved as 'qwk_predictions.png'")


# # Error analysis - examine instances with largest errors
# errors = abs(y_test - y_pred_calibrated)
# error_df = pd.DataFrame({
#     'Actual': y_test,
#     'Predicted': y_pred_calibrated,
#     'Error': errors
# })
# error_df = error_df.sort_values('Error', ascending=False)

# print("\nLargest prediction errors:")
# print(error_df.head(10))


# # Distribution of errors
# plt.figure(figsize=(10, 6))
# plt.hist(errors, bins=20)
# plt.xlabel('Absolute Error')
# plt.ylabel('Frequency')
# plt.title('Distribution of Prediction Errors')
# plt.tight_layout()
# # plt.savefig('qwk_error_distribution.png')
# print("Error distribution plot saved as 'qwk_error_distribution.png'")


# # Save the model
# joblib.dump({
#     'pipeline': final_pipeline,
#     'calibration_function': optimize_predictions_for_qwk
# }, 'qwk_optimized_model.pkl')
# print("\nQWK-optimized model saved as 'qwk_optimized_model.pkl'")


# Function to make predictions with the saved model
# print("\nExample code to use the saved model:")
# print("""
# # Load model
# import joblib
# model_data = joblib.load('qwk_optimized_model.pkl')
# pipeline = model_data['pipeline']
# calibration_func = model_data['calibration_function']

# # Make predictions (with new data)
# raw_predictions = pipeline.predict(new_data)

# # For optimal QWK, use the saved model on some validation data first
# # and then apply the calibration
# validation_actual = ... # Some known labels
# validation_pred = pipeline.predict(validation_data)
# calibrated_predictions = calibration_func(validation_actual, validation_pred)
# """)

