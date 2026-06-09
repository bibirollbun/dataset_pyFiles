# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV # A common meta-model
import xgboost as xgb
from sklearn.metrics import mean_squared_error

# ## Load Data
print("Files in input directory:")
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


df_train_full = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
X_test_original = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv") # Load test set early

print("\nTrain Data Head:")
df_train_full.head()


print("\nTrain Data Info:")
df_train_full.info()


print("\nTrain Data Null Check:")
df_train_full.isnull().sum()


# ## Prepare Training Data

# Separate Target Variable
y = df_train_full['accident_risk']
X = df_train_full.drop('accident_risk', axis=1)

# Define column types
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']

# --- Preprocessing Function (Optional but good practice) ---
# It's often helpful to put preprocessing steps in a function
# to ensure they are applied identically to train and test data.

def preprocess_features(df, categorical_cols, fit_cols=None):
    """Applies preprocessing steps: dummies, bool->int, drop id."""
    df_processed = pd.get_dummies(df, columns=categorical_cols)

    for col in df_processed.columns:
        if df_processed[col].dtype == 'bool':
            df_processed[col] = df_processed[col].astype(int)

    if 'id' in df_processed.columns:
         df_processed = df_processed.drop('id', axis=1)

    # Align columns if fit_cols are provided (for test set)
    if fit_cols is not None:
        df_processed = df_processed.reindex(columns=fit_cols, fill_value=0)

    return df_processed

# --- Preprocess Training Data ---
X_processed = preprocess_features(X, categorical_cols)

print("\nProcessed Training Data Head:")
X_processed.head()


print(f"Processed Training Data Shape: {X_processed.shape}")


# (Optional) Correlation Heatmap - Can be slow with many features
# plt.figure(figsize=(15,12))
# sns.heatmap(X_processed.corr(), annot=False, cmap="YlGnBu", fmt=".2f", linewidths=0.5) # Turn off annotation if too cluttered
# plt.title("Correlation Heatmap")
# plt.show()

# ## Train/Validation Split
X_train, X_val, y_train, y_val = train_test_split(X_processed, y, test_size=0.2, random_state=42)
print(f"\nTraining features shape: {X_train.shape}")
print(f"Validation features shape: {X_val.shape}")


rf_best_params = {
    'n_estimators': 200,
    'max_depth': 15,
    'min_samples_split': 2,
    'random_state': 42,
    'n_jobs': -1
}


xgb_best_params = {
    'objective': 'reg:squarederror',
    'n_estimators': 200, # Example - use your tuned value or let fit decide with early stopping
    'learning_rate': 0.1,
    'max_depth': 9,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'reg_alpha': 0.01,
    'reg_lambda': 0.1,
    'random_state': 42,
    'n_jobs': -1,
    'tree_method': 'hist'
}


# xgb_model = xgb.XGBRegressor(
#     objective='reg:squarederror', # Objective function for regression
#     n_estimators=200,           # Start with a potentially high number, early stopping will find the best
#     learning_rate=0.1,          # Typical starting point
#     max_depth=7,                 # Typical starting point
#     subsample=0.8,               # Fraction of samples used per tree
#     colsample_bytree=0.8,        # Fraction of features used per tree
#     random_state=42,
#     n_jobs=-1,
#     tree_method='hist',           # Often faster for large datasets
#     early_stopping_rounds=50,
# )


rf_base = RandomForestRegressor(**rf_best_params)
xgb_base = xgb.XGBRegressor(**xgb_best_params)


# param_distributions_xgb = {
#     'max_depth': [5, 9],
#     'learning_rate': [0.01, 0.1],
#     'n_estimators': [100, 200], # Will often be overridden by early stopping if used in fit
#     'reg_lambda': [0.1, 1]   # L2 regularization
# }


# # explicitly require this experimental feature
# from sklearn.experimental import enable_halving_search_cv # noqa
# # now you can import normally from model_selection
# from sklearn.model_selection import HalvingRandomSearchCV

# xgb_search = HalvingRandomSearchCV(
#     estimator=xgb_model,
#     param_distributions=param_distributions_xgb, # Number of parameter settings that are sampled
#     scoring='neg_root_mean_squared_error',
#     n_jobs=-1,                  # Use all available cores
#     cv=3,                       # Use 3-fold cross-validation (faster)
#     verbose=1,
#     random_state=42
# )


# print("\nStarting HalvingRandomSearchCV for XGBoost...")
# # Fit using the training data, enabling early stopping within the search
# # Use X_val, y_val as the evaluation set for early stopping
# xgb_search.fit(X_train, y_train,
#                eval_set=[(X_val, y_val)], # Use validation set for early stopping
#                verbose=False)


# print(f"Best parameters found: {xgb_search.best_params_}")
# best_xgb_model = xgb_search.best_estimator_


base_models = [
    ('RandomForest', rf_base),
    ('XGBoost', xgb_base)
]


meta_model = RidgeCV()


stacking_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5, # Cross-validation strategy to generate predictions for the final estimator
    n_jobs=-1
)


from sklearn.model_selection import KFold, cross_val_score

n_splits_outer = 5 # Number of folds for outer cross-validation
kf_outer = KFold(n_splits=n_splits_outer, shuffle=True, random_state=42)

print(f"\nPerforming {n_splits_outer}-Fold Cross-Validation on Stacking Regressor...")
rmse_scores = cross_val_score(
    stacking_model,
    X_train, # Use the full processed training data
    y_train,           # Use the full target variable
    scoring='neg_root_mean_squared_error',
    cv=kf_outer,
    n_jobs=-1 # Use parallelism for outer folds if possible
)


# Scores are negative RMSE, so take the absolute value and mean
mean_cv_rmse = np.mean(np.abs(rmse_scores))
std_cv_rmse = np.std(np.abs(rmse_scores))
print(f"Mean CV RMSE: {mean_cv_rmse:.6f} (+/- {std_cv_rmse:.6f})")


# --- Fit Final Stacking Model on ALL Training Data ---
print("\nFitting final Stacking Regressor on all training data...")
# If using XGBoost with high n_estimators and potential early stopping:
# You might need a more complex setup to fit with early stopping here,
# or ensure n_estimators from tuning is reasonable.
# For simplicity now, we fit without XGBoost's early stopping.
stacking_model.fit(X_train, y_train)
print("Final model fitting complete.")


# # --- Evaluate Tuned XGBoost Model ---
# print("\nEvaluating tuned XGBoost model on validation set...")
# val_preds_xgb = best_xgb_model.predict(X_val)
# xgb_rmse = np.sqrt(mean_squared_error(y_val, val_preds_xgb))
# print(f"Validation RMSE (Tuned XGBoost): {xgb_rmse:.6f}")


# --- Evaluate Tuned Stacking Model ---
print("\nEvaluating tuned Stacking model on validation set...")
val_preds_stacking = stacking_model.predict(X_val)
stacking_rmse = np.sqrt(mean_squared_error(y_val, val_preds_stacking))
print(f"Validation RMSE (Tuned Stacking): {stacking_rmse:.6f}")


# # ## Model Training
# # Using the manually defined parameters from your cell 48
# print("\nTraining RandomForestRegressor...")
# rf_model_best = RandomForestRegressor(n_estimators=200,
#                                       min_samples_split=2,
#                                       max_depth=15, # Note: max_depth=10 might limit the model significantly
#                                       random_state=42,
#                                       n_jobs=-1)
# rf_model_best.fit(X_train, y_train)
# print("Training complete.")


# # ## Local Validation
# print("\nEvaluating on validation set...")
# val_preds = rf_model_best.predict(X_val)
# selected_rmse = np.sqrt(mean_squared_error(y_val, val_preds))
# print(f"Validation RMSE: {selected_rmse:.6f}") # Use formatted output



# ## Prepare Test Data and Predict

print("\nProcessing test data...")
test_ids = X_test_original['id'] # Store original IDs

# Apply the SAME preprocessing, passing X_train.columns for alignment
X_test_aligned = preprocess_features(X_test_original,
                                     categorical_cols,
                                     fit_cols=X_train.columns) # Pass training columns

print(f"Processed Test Data Shape: {X_test_aligned.shape}")


print("\nPredicting on test data...")
y_pred = stacking_model.predict(X_test_aligned)
print("Prediction complete.")


# ## Create Submission File
print("\nCreating submission file...")
submission = pd.DataFrame({
    'id': test_ids,
    'accident_risk': y_pred
})
submission.to_csv('submission.csv', index=False)
print("Predictions saved to submission.csv")


print("\nSubmission File Head:")
submission.head()




