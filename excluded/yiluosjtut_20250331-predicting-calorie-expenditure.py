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


import pandas as pd
import numpy as np
import os
import time
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_log_error # Using RMSLE directly for clarity
from sklearn.preprocessing import StandardScaler

# import lightgbm # This can remain if other lightgbm.xxx calls are made, or be removed if not.
from lightgbm import LGBMRegressor
from lightgbm.callback import early_stopping as lgb_early_stopping # Specific import for early_stopping
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

# from category_encoders import TargetEncoder # Keep for potential future use

from tqdm.auto import tqdm
from itertools import combinations
import warnings
warnings.simplefilter('ignore')


# --- 1. Load Data ---
print("Loading data...")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e5/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e5/test.csv")
sample_submission = pd.read_csv("/kaggle/input/playground-series-s5e5/sample_submission.csv")
print("Data loaded successfully.")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# --- 2. Basic EDA (Placeholder for more detailed EDA) ---
print("\n--- Basic EDA ---")
print("Train data description:")
print(train_df.describe())
print("\nMissing values in train_df:\n", train_df.isnull().sum())
print("\nMissing values in test_df:\n", test_df.isnull().sum())


# --- 3. Feature Engineering ---
print("\n--- Feature Engineering ---")
def feature_engineer(df):
    df_fe = df.copy()

    # Original numerical features
    numerical_features = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"]

    # a. BMI (Body Mass Index)
    # Height is in cm, convert to meters for BMI calculation
    df_fe['BMI'] = df_fe['Weight'] / ((df_fe['Height'] / 100) ** 2)

    # b. Ratio Features
    df_fe['Duration_x_Heart_Rate'] = df_fe['Duration'] * df_fe['Heart_Rate'] # Interaction
    df_fe['Intensity'] = df_fe['Heart_Rate'] / (df_fe['Duration'] + 1e-6) # Avoid division by zero
    df_fe['Weight_x_Height'] = df_fe['Weight'] * df_fe['Height']
    df_fe['Age_x_BMI'] = df_fe['Age'] * df_fe['BMI']
    df_fe['Body_Temp_x_Duration'] = df_fe['Body_Temp'] * df_fe['Duration']


    # c. Polynomial Features (Squared and Square Root)
    for col in numerical_features + ['BMI']: # Include BMI here
        df_fe[f'{col}_sq'] = df_fe[col] ** 2
        df_fe[f'{col}_sqrt'] = np.sqrt(df_fe[col].clip(0)) # Clip at 0 for sqrt

    # d. Cross-product terms (Interaction Features)
    # Update numerical_features to include BMI for cross terms
    extended_numerical_features = numerical_features + ['BMI']
    for i in range(len(extended_numerical_features)):
        for j in range(i + 1, len(extended_numerical_features)):
            feature1 = extended_numerical_features[i]
            feature2 = extended_numerical_features[j]
            cross_term_name = f"{feature1}_cross_{feature2}"
            df_fe[cross_term_name] = df_fe[feature1] * df_fe[feature2]

    # e. Label Encode 'Sex'
    # Done after creating interactions with original numerical features,
    # but before interactions that might use the encoded 'Sex' if desired.
    # For now, we'll encode it and XGBoost/LGBM/CatBoost can handle it.
    if 'Sex' in df_fe.columns:
        le = LabelEncoder()
        df_fe['Sex_encoded'] = le.fit_transform(df_fe['Sex'])
        # df_fe['Sex_encoded'] = df_fe['Sex_encoded'].astype('category') # Models can handle this

    # f. Interactions with Sex (using the original numerical values)
    # Example: df_fe['Sex_Male_x_Age'] = (df_fe['Sex'] == 'male').astype(int) * df_fe['Age']
    # This can be expanded. For simplicity, we'll rely on the models to pick up 'Sex_encoded' interactions.

    return df_fe

train_fe = feature_engineer(train_df.copy())
test_fe = feature_engineer(test_df.copy())

# Align columns - crucial for consistent feature sets
train_labels = train_fe['Calories']
train_ids = train_fe['id']
test_ids = test_fe['id']

# Drop original Sex, id, and Calories (if it exists in test_fe, though it shouldn't)
train_X = train_fe.drop(columns=['id', 'Calories', 'Sex'])
test_X = test_fe.drop(columns=['id', 'Sex', 'Calories'], errors='ignore')


# Ensure columns are in the same order and only common columns are kept
common_cols = list(set(train_X.columns) & set(test_X.columns))
train_X = train_X[common_cols]
test_X = test_X[common_cols]

print(f"Train_X shape after FE: {train_X.shape}")
print(f"Test_X shape after FE: {test_X.shape}")
print(f"Number of common features: {len(common_cols)}")


# Target transformation
y = np.log1p(train_labels)


# --- 4. Model Training ---
print("\n--- Model Training ---")
FOLDS = 5
RANDOM_STATE = 42

# KFold setup
kf = KFold(n_splits=FOLDS, shuffle=True, random_state=RANDOM_STATE)

# To store OOF (Out-of-Fold) predictions and test predictions for each model
oof_preds = {}
test_preds = {}
models_rmse = {}

features = common_cols # Use the common columns

# --- XGBoost ---
print("\nTraining XGBoost...")
oof_xgb = np.zeros(len(train_X))
pred_xgb = np.zeros(len(test_X))
xgb_feature_importances = pd.DataFrame(index=features)

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_X, y)):
    print(f"\n{'#'*10} XGBoost Fold {fold+1} {'#'*10}")
    X_train_fold, y_train_fold = train_X.iloc[train_idx], y.iloc[train_idx]
    X_valid_fold, y_valid_fold = train_X.iloc[valid_idx], y.iloc[valid_idx]

    # --- Hyperparameter Optimization Placeholder ---
    # For robust results, use Optuna, GridSearchCV, or RandomizedSearchCV here.
    
    # Determine device for XGBoost robustly
    _xgb_device_val_check = XGBRegressor().get_params().get('device')
    _computed_xgb_device = 'cuda' if isinstance(_xgb_device_val_check, str) and 'cuda' in _xgb_device_val_check else 'cpu'
    
    # Example (manual params, tune these!):
    xgb_params = {
        'objective': 'reg:squarederror', # for regression
        'eval_metric': 'rmse',           # RMSLE on log-transformed target is RMSE
        'eta': 0.02,                     # learning_rate
        'max_depth': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'seed': RANDOM_STATE + fold,
        'n_estimators': 3000,           # Increased, but with early stopping
        'early_stopping_rounds': 150,
        # 'tree_method': 'hist',        # Faster, try 'gpu_hist' if GPU available
        'device': _computed_xgb_device   # Use the robustly determined device
    }
    
    X_train_fold_processed_xgb = X_train_fold.copy()
    X_valid_fold_processed_xgb = X_valid_fold.copy()
    test_X_fold_copy_xgb = test_X.copy()

    if 'Sex_encoded' in X_train_fold_processed_xgb.columns: # XGBoost can use this if type is category
        X_train_fold_processed_xgb['Sex_encoded'] = X_train_fold_processed_xgb['Sex_encoded'].astype('category')
        X_valid_fold_processed_xgb['Sex_encoded'] = X_valid_fold_processed_xgb['Sex_encoded'].astype('category')
        if 'Sex_encoded' in test_X_fold_copy_xgb.columns:
            test_X_fold_copy_xgb['Sex_encoded'] = test_X_fold_copy_xgb['Sex_encoded'].astype('category')
        xgb_params['enable_categorical'] = True


    model_xgb = XGBRegressor(**xgb_params)
    model_xgb.fit(X_train_fold_processed_xgb, y_train_fold,
                  eval_set=[(X_valid_fold_processed_xgb, y_valid_fold)],
                  verbose=500)

    oof_xgb[valid_idx] = model_xgb.predict(X_valid_fold_processed_xgb)
    pred_xgb += model_xgb.predict(test_X_fold_copy_xgb) / FOLDS # Use the consistently processed test_X_fold_copy_xgb
    
    # Ensure feature_importances_ are available (e.g. tree is not empty)
    if hasattr(model_xgb, 'feature_importances_') and model_xgb.feature_importances_ is not None:
        xgb_feature_importances[f'fold_{fold+1}'] = model_xgb.feature_importances_
    else: # Handle cases where feature importances might not be available (e.g. no splits)
        xgb_feature_importances[f'fold_{fold+1}'] = np.zeros(len(features))


# Calculate RMSLE for the last fold (or consider averaging if needed, though overall CV is better)
# Note: y.iloc[valid_idx] and oof_xgb[valid_idx] refer to the *last* validation fold here after the loop
fold_rmsle_xgb = mean_squared_log_error(np.expm1(y.iloc[valid_idx]), np.expm1(oof_xgb[valid_idx]), squared=False)
print(f"XGBoost Last Fold ({fold+1}) RMSLE (on original scale): {fold_rmsle_xgb:.4f}")


oof_preds['xgb'] = oof_xgb
test_preds['xgb'] = pred_xgb
overall_rmsle_xgb = mean_squared_log_error(np.expm1(y), np.expm1(oof_xgb), squared=False)
models_rmse['xgb'] = overall_rmsle_xgb
print(f"\nOverall XGBoost CV RMSLE (on original scale): {overall_rmsle_xgb:.4f}")

# --- LightGBM ---
print("\nTraining LightGBM...")
oof_lgb = np.zeros(len(train_X))
pred_lgb = np.zeros(len(test_X))
lgb_feature_importances = pd.DataFrame(index=features)

for fold, (train_idx, valid_idx) in enumerate(kf.split(train_X, y)):
    print(f"\n{'#'*10} LightGBM Fold {fold+1} {'#'*10}")
    X_train_fold, y_train_fold = train_X.iloc[train_idx], y.iloc[train_idx]
    X_valid_fold, y_valid_fold = train_X.iloc[valid_idx], y.iloc[valid_idx]

    X_train_fold_processed_lgb = X_train_fold.copy()
    X_valid_fold_processed_lgb = X_valid_fold.copy()
    test_X_fold_copy_lgb = test_X.copy()

    # Convert 'Sex_encoded' to category for LightGBM if it exists
    if 'Sex_encoded' in X_train_fold_processed_lgb.columns:
        X_train_fold_processed_lgb['Sex_encoded'] = X_train_fold_processed_lgb['Sex_encoded'].astype('category')
        X_valid_fold_processed_lgb['Sex_encoded'] = X_valid_fold_processed_lgb['Sex_encoded'].astype('category')
        if 'Sex_encoded' in test_X_fold_copy_lgb.columns:
            test_X_fold_copy_lgb['Sex_encoded'] = test_X_fold_copy_lgb['Sex_encoded'].astype('category')


    # --- Hyperparameter Optimization Placeholder ---
    lgb_params = {
        'objective': 'regression_l1', # Or 'rmse' or 'regression' which is L2
        'metric': 'rmse',
        'n_estimators': 3000,
        'learning_rate': 0.02,
        'feature_fraction': 0.7, # colsample_bytree
        'bagging_fraction': 0.8, # subsample
        'bagging_freq': 1,
        'lambda_l1': 0.1,        # L1 regularization
        'lambda_l2': 0.1,        # L2 regularization
        'num_leaves': 31,        # Default is 31, adjust based on max_depth idea
        'max_depth': -1,         # No limit, num_leaves is more restrictive
        'seed': RANDOM_STATE + fold,
        'n_jobs': -1,
        'verbose': -1,
        'boosting_type': 'gbdt',
        # 'device_type': 'gpu', # if GPU available and LightGBM compiled with GPU support
    }

    model_lgb = LGBMRegressor(**lgb_params)
    model_lgb.fit(X_train_fold_processed_lgb, y_train_fold,
                  eval_set=[(X_valid_fold_processed_lgb, y_valid_fold)],
                  eval_metric='rmse',
                  callbacks=[lgb_early_stopping(150, verbose=False)]) # Use aliased specific import

    oof_lgb[valid_idx] = model_lgb.predict(X_valid_fold_processed_lgb)
    pred_lgb += model_lgb.predict(test_X_fold_copy_lgb) / FOLDS
    
    if hasattr(model_lgb, 'feature_importances_') and model_lgb.feature_importances_ is not None:
        lgb_feature_importances[f'fold_{fold+1}'] = model_lgb.feature_importances_
    else:
        lgb_feature_importances[f'fold_{fold+1}'] = np.zeros(len(features))


fold_rmsle_lgb = mean_squared_log_error(np.expm1(y.iloc[valid_idx]), np.expm1(oof_lgb[valid_idx]), squared=False)
print(f"LightGBM Last Fold ({fold+1}) RMSLE (on original scale): {fold_rmsle_lgb:.4f}")


oof_preds['lgb'] = oof_lgb
test_preds['lgb'] = pred_lgb
overall_rmsle_lgb = mean_squared_log_error(np.expm1(y), np.expm1(oof_lgb), squared=False)
models_rmse['lgb'] = overall_rmsle_lgb
print(f"\nOverall LightGBM CV RMSLE (on original scale): {overall_rmsle_lgb:.4f}")


# --- CatBoost ---
print("\nTraining CatBoost...")
oof_cat = np.zeros(len(train_X))
pred_cat = np.zeros(len(test_X))
cat_feature_importances = pd.DataFrame(index=features)

# Identify categorical features for CatBoost
cat_features_indices = [train_X.columns.get_loc(col) for col in ['Sex_encoded'] if col in train_X.columns]


for fold, (train_idx, valid_idx) in enumerate(kf.split(train_X, y)):
    print(f"\n{'#'*10} CatBoost Fold {fold+1} {'#'*10}")
    X_train_fold, y_train_fold = train_X.iloc[train_idx], y.iloc[train_idx]
    X_valid_fold, y_valid_fold = train_X.iloc[valid_idx], y.iloc[valid_idx]
    
    # CatBoost handles categorical features internally if specified, no need to change dtype for X_train_fold etc.
    # unless specifically required for other reasons.

    # --- Hyperparameter Optimization Placeholder ---
    cat_params = {
        'iterations': 3000,
        'learning_rate': 0.02,
        'depth': 8, # CatBoost depth tends to be smaller
        'l2_leaf_reg': 3,
        'loss_function': 'RMSE',
        'eval_metric': 'RMSE',
        'random_seed': RANDOM_STATE + fold,
        'verbose': 500,
        'early_stopping_rounds': 150,
        # 'task_type': 'GPU', # if GPU available
        'cat_features': cat_features_indices if cat_features_indices else None
    }

    model_cat = CatBoostRegressor(**cat_params)
    model_cat.fit(X_train_fold, y_train_fold, # CatBoost can take original DFs if cat_features is set
                  eval_set=[(X_valid_fold, y_valid_fold)],
                  cat_features=cat_features_indices if cat_features_indices else None) 

    oof_cat[valid_idx] = model_cat.predict(X_valid_fold)
    pred_cat += model_cat.predict(test_X) / FOLDS # test_X is fine for CatBoost if cat_features was used in training
    
    if hasattr(model_cat, 'get_feature_importance'):
        cat_feature_importances[f'fold_{fold+1}'] = model_cat.get_feature_importance()
    else:
        cat_feature_importances[f'fold_{fold+1}'] = np.zeros(len(features))


fold_rmsle_cat = mean_squared_log_error(np.expm1(y.iloc[valid_idx]), np.expm1(oof_cat[valid_idx]), squared=False)
print(f"CatBoost Last Fold ({fold+1}) RMSLE (on original scale): {fold_rmsle_cat:.4f}")


oof_preds['cat'] = oof_cat
test_preds['cat'] = pred_cat
overall_rmsle_cat = mean_squared_log_error(np.expm1(y), np.expm1(oof_cat), squared=False)
models_rmse['cat'] = overall_rmsle_cat
print(f"\nOverall CatBoost CV RMSLE (on original scale): {overall_rmsle_cat:.4f}")



# --- 5. Ensembling ---
print("\n--- Ensembling ---")
print("CV RMSLE Scores:")
for model_name, score in models_rmse.items():
    print(f"{model_name.upper()}: {score:.4f}")

# Simple Weighted Averaging (Tune these weights based on CV scores or using optimization)
# Example weights (start with equal or based on individual CV performance)
# Ensure scores are positive before division
safe_scores = {k: (v if v > 1e-9 else 1e-9) for k,v in models_rmse.items()} # handle zero or negative scores if any
total_inverse_rmsle = sum(1.0/score for score in safe_scores.values())
if total_inverse_rmsle == 0: # All scores were effectively zero or problematic
    weights = {key: 1/len(safe_scores) if len(safe_scores) > 0 else 0 for key in safe_scores}
else:
    weights = {
        model_name: (1.0/score) / total_inverse_rmsle for model_name, score in safe_scores.items()
    }


# Fallback if weights could not be computed (e.g. no models ran)
if not weights and len(models_rmse) > 0: # if models_rmse has entries but weights are empty
    num_models_available = len(models_rmse)
    weights = {model_name: 1/num_models_available for model_name in models_rmse.keys()}
elif not models_rmse: # No models ran at all
    weights = {'xgb': 1/3, 'lgb': 1/3, 'cat': 1/3} # Default placeholder


print(f"\nEnsemble Weights: {weights}")

# Ensure all models ran and have predictions
final_oof_preds = np.zeros(len(train_X))
final_test_preds = np.zeros(len(test_X))
num_models_in_ensemble = 0

active_models_for_ensemble = [m for m in ['xgb', 'lgb', 'cat'] if m in oof_preds and m in test_preds and m in weights]

if not active_models_for_ensemble:
    print("Error: No models available for ensembling with valid predictions and weights.")
    # Fallback: use predictions from the first available model if any, or zeros
    if 'xgb' in test_preds: final_test_preds = test_preds['xgb']
    elif 'lgb' in test_preds: final_test_preds = test_preds['lgb']
    elif 'cat' in test_preds: final_test_preds = test_preds['cat']
    else: final_test_preds = np.zeros(len(test_X)) # Should not happen if models ran
else:
    # Normalize weights for active models only
    active_weights_sum = sum(weights[m] for m in active_models_for_ensemble)
    if active_weights_sum > 1e-9: # Avoid division by zero
        for model_name in active_models_for_ensemble:
            normalized_weight = weights[model_name] / active_weights_sum
            final_oof_preds += normalized_weight * oof_preds[model_name]
            final_test_preds += normalized_weight * test_preds[model_name]
        num_models_in_ensemble = len(active_models_for_ensemble)
    else: # Fallback if active_weights_sum is zero (e.g. all weights became zero)
        print("Warning: Sum of active model weights is zero. Falling back to equal weighting for active models.")
        equal_weight = 1.0 / len(active_models_for_ensemble) if active_models_for_ensemble else 0
        for model_name in active_models_for_ensemble:
            final_oof_preds += equal_weight * oof_preds[model_name]
            final_test_preds += equal_weight * test_preds[model_name]
        num_models_in_ensemble = len(active_models_for_ensemble)


if num_models_in_ensemble > 0 and len(final_oof_preds) == len(y):
    ensemble_rmsle = mean_squared_log_error(np.expm1(y), np.expm1(final_oof_preds), squared=False)
    print(f"\nEnsemble CV RMSLE (on original scale): {ensemble_rmsle:.4f}")
else:
    print("\nCould not compute ensemble CV RMSLE.")




# --- 6. Submission ---
print("\n--- Creating Submission File ---")
# Inverse transform predictions
predictions_original_scale = np.expm1(final_test_preds)

# Clip predictions (based on observed min/max in train or competition guidelines)
# From your original notebook, min=1, max=314
predictions_clipped = np.clip(predictions_original_scale, 1, 314)

print(f"Predictions mean (original scale): {predictions_original_scale.mean():.2f}")
print(f"Predictions median (original scale): {np.median(predictions_original_scale):.2f}")
print(f"Predictions mean (clipped): {predictions_clipped.mean():.2f}")
print(f"Predictions median (clipped): {np.median(predictions_clipped):.2f}")

submission_df = pd.DataFrame({'id': test_ids, 'Calories': predictions_clipped})
submission_df.to_csv('submission.csv', index=False)
print("\nSubmission file created: submission.csv")
print(submission_df.head())

# --- 7. Feature Importances (Example for XGBoost) ---
# You can create similar plots for LGBM and CatBoost
# Averaging feature importances across folds
if 'xgb' in oof_preds and not xgb_feature_importances.empty: # Check if XGBoost ran and importances were collected
    xgb_feature_importances['mean_importance'] = xgb_feature_importances.mean(axis=1)
    xgb_feature_importances.sort_values(by='mean_importance', ascending=False, inplace=True)

    plt.figure(figsize=(10, max(15, len(features) // 2))) # Adjust height based on num features
    sns.barplot(x='mean_importance', y=xgb_feature_importances.index[:50], data=xgb_feature_importances.head(50)) # Top 50
    plt.title('XGBoost - Top 50 Feature Importances (Averaged over Folds)')
    plt.tight_layout()
    plt.savefig('xgb_feature_importances.png')
    plt.show()
    print("\nSaved XGBoost feature importances plot to xgb_feature_importances.png")

print("\nNotebook execution finished.")

