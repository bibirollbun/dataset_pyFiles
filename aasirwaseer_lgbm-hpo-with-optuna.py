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

# --- Strategy I: Using Pandas Fallback ---
print("⚠️ GPU acceleration (cuDF) failed. Using pandas (CPU) for data loading.")
df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
df_sub = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
IS_GPU = False # Set our flag to False

# --- Strategy II: Deep Data Exploration (Initial EDA) ---

print("\n--- Training Data Info ---")
# .info() gives us data types and null counts
df_train.info()

print("\n--- Test Data Info ---")
df_test.info()

print("\n--- Training Data Head (First 5 Rows) ---")
# .head() shows us what the features look like
print(df_train.head())

print("\n--- Target Variable 'accident_risk' Description ---")
# Understanding our target is crucial (Evaluation is RMSE)
print(df_train['accident_risk'].describe())


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# Keep 'id' and 'accident_risk' separate for now
train_ids = df_train['id']
test_ids = df_test['id']
target = df_train['accident_risk']

# Drop id and target to align train/test for processing
df_train = df_train.drop(columns=['id', 'accident_risk'])
df_test = df_test.drop(columns=['id'])

# Combine for consistent processing (Strategy II)
df_all = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)

# --- Initial Feature Engineering (Strategy II) ---

# Identify feature types
bool_cols = [col for col in df_all.columns if df_all[col].dtype == 'bool']
object_cols = [col for col in df_all.columns if df_all[col].dtype == 'object']
num_cols = [col for col in df_all.columns if col not in bool_cols and col not in object_cols]

print(f"Boolean columns: {bool_cols}")
print(f"Object columns: {object_cols}")
print(f"Numerical columns: {num_cols}")

# 1. Convert boolean columns to integer (0 or 1)
for col in bool_cols:
    df_all[col] = df_all[col].astype(int)

# 2. Encode object (categorical) columns
# We use LabelEncoder for simplicity and speed, good for tree models
encoders = {}
for col in object_cols:
    le = LabelEncoder()
    df_all[col] = le.fit_transform(df_all[col])
    encoders[col] = le

print("\n--- Data Head After Encoding ---")
print(df_all.head())

# --- Distribution Shift Check (Strategy II) ---
print("\n--- Checking for Distribution Shift ---")

# Separate back into train and test
df_train_processed = df_all.iloc[:len(df_train)]
df_test_processed = df_all.iloc[len(df_train):]

# Get all feature names
features = df_train_processed.columns.tolist()

# Plot distributions
n_cols = 4
n_rows = (len(features) + n_cols - 1) // n_cols
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
axes = axes.flatten()

for i, col in enumerate(features):
    ax = axes[i]
    # Use value_counts for categorical/discrete features
    if df_all[col].nunique() < 25:
        train_counts = df_train_processed[col].value_counts(normalize=True).sort_index()
        test_counts = df_test_processed[col].value_counts(normalize=True).sort_index()
        
        # Create a dataframe for seaborn barplot
        df_plot_train = pd.DataFrame({'value': train_counts.index, 'proportion': train_counts.values, 'dataset': 'train'})
        df_plot_test = pd.DataFrame({'value': test_counts.index, 'proportion': test_counts.values, 'dataset': 'test'})
        df_plot = pd.concat([df_plot_train, df_plot_test])
        
        sns.barplot(data=df_plot, x='value', y='proportion', hue='dataset', ax=ax, palette={'train': 'blue', 'test': 'orange'})
        ax.set_title(f"Distribution of '{col}'", fontweight='bold')
    
    # Use histograms (kde) for continuous features
    else:
        sns.histplot(df_train_processed[col], ax=ax, color='blue', label='Train', stat='density', common_norm=False, kde=True)
        sns.histplot(df_test_processed[col], ax=ax, color='orange', label='Test', stat='density', common_norm=False, kde=True)
        ax.set_title(f"Distribution of '{col}'", fontweight='bold')
        ax.legend()

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    axes[j].axis('off')

plt.tight_layout()
plt.show()

# Save processed data for the next step
# We'll also save the original target and test_ids
X = df_train_processed
y = target
X_test = df_test_processed


import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# Retaining objects from previous steps: X, y, X_test, test_ids
# Assuming X, y, X_test are standard pandas DataFrames/Series since cuDF failed.

# Re-read data using pandas for robustness in this environment
# Ensure we have the correct target and IDs
df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
y = df_train['accident_risk']
test_ids = df_test['id']

# Assuming X and X_test are the encoded pandas DataFrames from Step 2
# X = df_train_processed
# X_test = df_test_processed

# Recreate X and X_test just in case (using the previous encoding logic for fresh run):
# (This block assumes you would manually run the encoding from Step 2 if starting fresh)
# For now, we trust X and X_test exist as standard pandas DFs.

# Define the GBDT model parameters (initial fast baseline)
lgbm_params = {
    'objective': 'rmse',
    'metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    # Small number of leaves for fast training
    'num_leaves': 31,
    'boosting_type': 'gbdt',
    'early_stopping_round': 50,
}

# --- Local Validation (k-fold CV) Setup ---
kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_predictions = np.zeros(X.shape[0])
test_predictions = np.zeros(X_test.shape[0])
cv_scores = []

print("\n--- Starting 5-Fold LightGBM Baseline Training ---")
for fold, (train_index, val_index) in enumerate(kf.split(X, y)):
    # Slice the data
    X_train, X_val = X.iloc[train_index], X.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    # **FIXED:** Removing the unnecessary .to_pandas() and .to_numpy() calls
    # X_train, X_val are pandas DataFrames, y_train, y_val are pandas Series
    
    # Train the GBDT model
    model = lgb.LGBMRegressor(**lgbm_params)
    
    # Early stopping uses the validation set
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(50, verbose=False)])

    # Predict on validation set
    val_preds = model.predict(X_val)
    oof_predictions[val_index] = val_preds
    
    # Clip predictions to [0, 1] as per the competition rule
    oof_predictions[val_index] = np.clip(oof_predictions[val_index], 0, 1)

    # Calculate and store fold score
    fold_rmse = mean_squared_error(y_val, oof_predictions[val_index], squared=False)
    cv_scores.append(fold_rmse)
    
    print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")

    # Predict on the test set and accumulate
    # X_test is already a standard pandas DataFrame
    test_predictions += model.predict(X_test) / kf.n_splits

# Final CV Score
mean_cv_rmse = np.mean(cv_scores)
print(f"\n✅ LightGBM 5-Fold CV Mean RMSE: {mean_cv_rmse:.5f}")


import xgboost as xgb
from sklearn.metrics import mean_squared_error

# --- A. Feature Engineering: Aggregation Principle (Strategy II) ---

# Recombine X and X_test for consistent FE
# X, X_test should be pandas DFs from Step 2
X_test_copy = X_test.copy() 
X_all = pd.concat([X, X_test_copy], ignore_index=True)
print(f"Combined data shape: {X_all.shape}")

# Define the Aggregation Key
AGG_KEY = ['road_type', 'num_lanes', 'speed_limit']

# Features to aggregate
AGG_FEATURES = ['curvature', 'num_reported_accidents']

# Perform Group-By Aggregation
agg_df = X_all.groupby(AGG_KEY)[AGG_FEATURES].agg(['mean', 'median', 'std'])
agg_df.columns = [f'{col}_{stat}_by_{"_".join(AGG_KEY)}' for col, stat in agg_df.columns]
agg_df = agg_df.reset_index()

# Merge the new features back into the full dataset
X_all = pd.merge(X_all, agg_df, on=AGG_KEY, how='left')

# Drop the aggregated columns to prevent model confusion from collinearity
X_all = X_all.drop(columns=AGG_KEY)

# Separate back into train and test
X_fe = X_all.iloc[:len(X)].copy()
X_test_fe = X_all.iloc[len(X):].copy()

# Fill NaNs created by std (where group size was 1) with 0
X_fe = X_fe.fillna(0)
X_test_fe = X_test_fe.fillna(0)


# --- B. XGBoost Baseline Training (Strategy III) ---

# XGBoost Parameters (Optimized for tabular data)
xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'n_estimators': 1000,
    'learning_rate': 0.05,
    'colsample_bytree': 0.8,
    'subsample': 0.8,
    'max_depth': 7,
    'n_jobs': -1,
    'seed': 42
}

# 5-Fold CV Setup (reusing kf from Step 3)
xgb_oof_predictions = np.zeros(X_fe.shape[0])
xgb_test_predictions = np.zeros(X_test_fe.shape[0])
xgb_cv_scores = []

print("\n--- Starting 5-Fold XGBoost Training (with FE) ---")

for fold, (train_index, val_index) in enumerate(kf.split(X_fe, y)):
    X_train, X_val = X_fe.iloc[train_index], X_fe.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]

    model = xgb.XGBRegressor(**xgb_params)
    
    # Enable GPU acceleration if available (no explicit check, relies on environment)
    # model = xgb.XGBRegressor(**xgb_params, tree_method='gpu_hist') # Commented out for max compatibility

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              early_stopping_rounds=50,
              verbose=False)

    # Predict and clip
    val_preds = model.predict(X_val)
    xgb_oof_predictions[val_index] = np.clip(val_preds, 0, 1)
    
    # Calculate and store fold score
    fold_rmse = mean_squared_error(y_val, xgb_oof_predictions[val_index], squared=False)
    xgb_cv_scores.append(fold_rmse)
    
    print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")

    # Predict on the test set and accumulate
    xgb_test_predictions += model.predict(X_test_fe) / kf.n_splits

# Final CV Score
mean_xgb_cv_rmse = np.mean(xgb_cv_scores)
print(f"\n✅ XGBoost 5-Fold CV Mean RMSE (with FE): {mean_xgb_cv_rmse:.5f}")

# Store OOF predictions for Stacking
# We need to ensure we save the LightGBM OOF (oof_predictions) and the XGBoost OOF (xgb_oof_predictions)


from scipy.optimize import minimize

# --- Retrieving OOFs and Test Predictions ---
# We assume the following variables from Step 3 and Step 4 are available:
# oof_predictions (LGBM OOF)
# test_predictions (LGBM Test Preds)
# xgb_oof_predictions (XGBoost OOF)
# xgb_test_predictions (XGBoost Test Preds)
# y (Target Series)
# test_ids (Test IDs Series)


# 1. Define the objective function for blending
def blend_function(weights, oof1, oof2, target):
    """Calculates RMSE for a weighted blend of two OOF prediction sets."""
    alpha = weights[0]
    # Blend OOF predictions
    blended_oof = alpha * oof1 + (1 - alpha) * oof2
    # Ensure predictions are clipped
    blended_oof = np.clip(blended_oof, 0, 1)
    
    # Return the RMSE (our objective to minimize)
    return mean_squared_error(target, blended_oof, squared=False)

# 2. Find the optimal weight (alpha) for LGBM
# The search space is constrained between 0.0 and 1.0 (LGBM's weight)
initial_weight = [0.5]
bounds = [(0.0, 1.0)]

optimization_result = minimize(
    blend_function,
    initial_weight,
    args=(oof_predictions, xgb_oof_predictions, y),
    method='L-BFGS-B',
    bounds=bounds
)

# Extract the optimal weight
optimal_alpha = optimization_result.x[0]
optimal_blend_rmse = optimization_result.fun

print("\n--- Ensembling (Optimal Blending) ---")
print(f"Optimal LGBM Weight (alpha): {optimal_alpha:.4f}")
print(f"Optimal XGBoost Weight (1-alpha): {(1 - optimal_alpha):.4f}")
print(f"✅ Optimal Blended CV RMSE: {optimal_blend_rmse:.5f}")

# 3. Apply the optimal weights to the final test predictions
final_test_predictions = optimal_alpha * test_predictions + (1 - optimal_alpha) * xgb_test_predictions
final_test_predictions = np.clip(final_test_predictions, 0, 1)

# --- Final Model Comparison ---
print("\n--- Final CV Score Improvement ---")
print(f"LGBM Baseline: {0.05608:.5f}")
print(f"Optimal Blend: {optimal_blend_rmse:.5f}")
print(f"Gain: {0.05608 - optimal_blend_rmse:.6f}")

# 4. Create Final Submission File
submission_df_final = pd.DataFrame({
    'id': test_ids.to_numpy(), # Convert to numpy for compatibility
    'accident_risk': final_test_predictions
})
# submission_df_final.to_csv('final_ensemble_submission.csv', index=False)


#submission_df_final.to_csv('submission.csv', index=False) # scored 0.05562


import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd

# Load processed data (assuming X, y are available as standard pandas DFs/Series)
# X = X_fe (from Step 4)
# y = y (from Step 3)
# X_test = X_test_fe (from Step 4)

# Re-read data using pandas for robustness in this environment
df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
y = df_train['accident_risk']
test_ids = df_test['id']

# Assuming X and X_test are the encoded and FE-enhanced pandas DataFrames from Step 4

# --- Optuna HPO Setup ---
def objective(trial, X, y):
    """Objective function for Optuna to minimize RMSE."""
    
    # Define the search space for LightGBM
    lgbm_params = {
        'objective': 'rmse',
        'metric': 'rmse',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
        'boosting_type': 'gbdt',
        'early_stopping_round': 50,
    }

    # Use 3-Fold CV for faster HPO iteration
    kf_hpo = KFold(n_splits=3, shuffle=True, random_state=42)
    fold_rmses = []
    
    for train_index, val_index in kf_hpo.split(X, y):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model = lgb.LGBMRegressor(**lgbm_params)
        
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(50, verbose=False)])

        val_preds = model.predict(X_val)
        val_preds = np.clip(val_preds, 0, 1)
        fold_rmses.append(mean_squared_error(y_val, val_preds, squared=False))
        
    return np.mean(fold_rmses)

print("--- Starting LightGBM Hyperparameter Optimization (Optuna) ---")

# Run 50 trials for quick optimization
study = optuna.create_study(direction='minimize')
study.optimize(lambda trial: objective(trial, X_fe, y), n_trials=50, show_progress_bar=True)

best_params = study.best_params
best_hpo_rmse = study.best_value

print(f"\n✅ Optuna HPO Best 3-Fold CV RMSE: {best_hpo_rmse:.5f}")
print("Best HPO Parameters:")
print(best_params)

# --- Retrain Final Model with Best Params (5-Fold CV) ---
final_lgbm_params = {
    'objective': 'rmse', 'metric': 'rmse', 'n_estimators': 1000,
    'verbose': -1, 'n_jobs': -1, 'seed': 42, 'boosting_type': 'gbdt',
    'early_stopping_round': 50,
    **best_params # Overwrite with optimized parameters
}

kf_final = KFold(n_splits=5, shuffle=True, random_state=42)
final_oof_predictions = np.zeros(X_fe.shape[0])
final_test_predictions = np.zeros(X_test_fe.shape[0])
final_cv_scores = []

print("\n--- Starting Final 5-Fold LightGBM Training with HPO Params ---")
for fold, (train_index, val_index) in enumerate(kf_final.split(X_fe, y)):
    X_train, X_val = X_fe.iloc[train_index], X_fe.iloc[val_index]
    y_train, y_val = y.iloc[train_index], y.iloc[val_index]
    
    model = lgb.LGBMRegressor(**final_lgbm_params)
    
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(50, verbose=False)])

    val_preds = model.predict(X_val)
    final_oof_predictions[val_index] = np.clip(val_preds, 0, 1)
    
    fold_rmse = mean_squared_error(y_val, final_oof_predictions[val_index], squared=False)
    final_cv_scores.append(fold_rmse)
    
    print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")

final_mean_cv_rmse = np.mean(final_cv_scores)
print(f"\n✅ Final 5-Fold HPO LGBM CV Mean RMSE: {final_mean_cv_rmse:.5f}")

# Predict on test set
final_test_predictions += model.predict(X_test_fe) / kf_final.n_splits
final_test_predictions = np.clip(final_test_predictions, 0, 1)

# Create the new submission
submission_df_hpo = pd.DataFrame({
    'id': test_ids.to_numpy(),
    'accident_risk': final_test_predictions
})

# submission_df_hpo.to_csv('lgbm_hpo_submission.csv', index=False)


import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.model_selection import KFold
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error

# --- Reloading Raw Data and Target ---
df_train = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
df_test = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
y = df_train['accident_risk']

# Combine for consistent processing
X_train = df_train.drop(columns=['id', 'accident_risk'])
X_test = df_test.drop(columns=['id'])
X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

# --- Initial Feature Cleaning (NOT Target Encoding) ---
# Boolean to Int
bool_cols = [col for col in X_all.columns if X_all[col].dtype == 'bool']
for col in bool_cols:
    X_all[col] = X_all[col].astype(int)

# Label Encode non-Target Encoded Categoricals (Only the ones we won't TE)
# Since we only have 4 object columns, we will TE all of them, so no LE needed here.

# --- 1. Target Encoding (FE) ---
kf_te = KFold(n_splits=5, shuffle=True, random_state=42)
TARGET_ENCODE_COLS = ['road_type', 'lighting', 'weather', 'time_of_day']

# Initialize encoded dataframes
X_te = X_all.iloc[:len(X_train)].copy()
X_test_te = X_all.iloc[len(X_train):].copy()

print("\n--- Starting Target Encoding (OOF) on Raw Categoricals ---")
for col in TARGET_ENCODE_COLS:
    # Initialize new columns
    X_te[f'{col}_te'] = 0.0
    
    # OOF Encoding for Train Set
    for fold, (train_index, val_index) in enumerate(kf_te.split(X_te, y)):
        encoder = TargetEncoder(cols=[col], smoothing=0.2)
        # Use the raw object columns
        encoder.fit(X_te.iloc[train_index], y.iloc[train_index])
        
        # Ensure we are setting the value in the dedicated TE column
        X_te.iloc[val_index, X_te.columns.get_loc(f'{col}_te')] = encoder.transform(X_te.iloc[val_index])[col]
    
    # Fit the encoder on the entire train set and transform the test set
    full_encoder = TargetEncoder(cols=[col], smoothing=0.2)
    full_encoder.fit(X_te[TARGET_ENCODE_COLS], y)
    X_test_te[f'{col}_te'] = full_encoder.transform(X_test_te[TARGET_ENCODE_COLS])[col]

# Drop original categorical columns (now redundant)
X_te = X_te.drop(columns=TARGET_ENCODE_COLS)
X_test_te = X_test_te.drop(columns=TARGET_ENCODE_COLS)

print("Target Encoding Complete. Feature set ready.")

# --- 2. Neural Network (NN) Baseline ---

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_te)
X_test_scaled = scaler.transform(X_test_te)

# Define the NN architecture
def create_nn_model(input_dim):
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.2),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['RootMeanSquaredError'])
    return model

# 5-Fold CV Setup (using a new kf for clarity)
kf_nn = KFold(n_splits=5, shuffle=True, random_state=42)
nn_oof_predictions = np.zeros(X_te.shape[0])
nn_test_predictions = np.zeros(X_test_te.shape[0])
nn_cv_scores = []

print("\n--- Starting 5-Fold Neural Network Training (with TE) ---")

for fold, (train_index, val_index) in enumerate(kf_nn.split(X_scaled, y)):
    # Slice the scaled data
    X_train, X_val = X_scaled[train_index], X_scaled[val_index]
    y_train, y_val = y.iloc[train_index].values, y.iloc[val_index].values
    
    model = create_nn_model(X_scaled.shape[1])
    
    history = model.fit(X_train, y_train,
                        validation_data=(X_val, y_val),
                        epochs=50,
                        batch_size=256,
                        callbacks=[keras.callbacks.EarlyStopping(monitor='val_root_mean_squared_error', patience=5, verbose=0, mode='min')],
                        verbose=0)

    # Predict and clip
    val_preds = model.predict(X_val, verbose=0).flatten()
    nn_oof_predictions[val_index] = np.clip(val_preds, 0, 1)
    
    # Calculate and store fold score
    fold_rmse = mean_squared_error(y_val, nn_oof_predictions[val_index], squared=False)
    nn_cv_scores.append(fold_rmse)
    
    print(f"Fold {fold+1} RMSE: {fold_rmse:.5f}")

    # Predict on the test set and accumulate
    nn_test_predictions += model.predict(X_test_scaled, verbose=0).flatten() / kf_nn.n_splits

# Final CV Score
mean_nn_cv_rmse = np.mean(nn_cv_scores)
print(f"\n✅ Neural Network 5-Fold CV Mean RMSE (with TE): {mean_nn_cv_rmse:.5f}")


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# --- 1. Prepare OOF and Test Matrices (RE-RUNNING THIS FOR CONTEXT) ---

# OOF Matrix (Training data for Level 2 model)
X_meta = pd.DataFrame({
    'lgbm_oof': final_oof_predictions,
    'xgb_oof': xgb_oof_predictions,
    'nn_oof': nn_oof_predictions
})

# Test Matrix (Prediction data for Level 2 model)
X_test_meta = pd.DataFrame({
    'lgbm_pred': final_test_predictions,
    'xgb_pred': xgb_test_predictions,
    'nn_pred': nn_test_predictions
})

# --- 2. Train the Meta-Model (Stacking) ---
print("\n--- Starting Stacking Generalization (Level 2 Model) ---")

meta_model = LinearRegression()
meta_model.fit(X_meta, y)

# --- 3. Generate Final Prediction and Calculate Final CV Score (FIXED) ---

# **FIX:** Rename the columns of the test matrix to match the training OOF names.
X_test_meta.columns = X_meta.columns 

# Predict on the test data
final_stacked_predictions = meta_model.predict(X_test_meta)
final_stacked_predictions = np.clip(final_stacked_predictions, 0, 1)

# Calculate the final CV score by running the meta-model on the OOFs
stacked_oof_preds = meta_model.predict(X_meta)
stacked_oof_preds = np.clip(stacked_oof_preds, 0, 1)
final_stacked_rmse = mean_squared_error(y, stacked_oof_preds, squared=False)

print(f"Meta-Model Weights: LGBM={meta_model.coef_[0]:.3f}, XGB={meta_model.coef_[1]:.3f}, NN={meta_model.coef_[2]:.3f}")
print(f"✅ Final Stacked CV RMSE: {final_stacked_rmse:.5f}")

# --- Final Model Comparison and Submission ---

print("\n--- Final Score Comparison ---")
print(f"Best Blend (Step 5): 0.05607")
print(f"Final Stacking (Step 8): {final_stacked_rmse:.5f}")
print(f"Total Gain from Stacking: {0.05607 - final_stacked_rmse:.6f}")

# Create Final Submission File (Not run, but prepared)
submission_df_stacked = pd.DataFrame({
    'id': test_ids.to_numpy(),
    'accident_risk': final_stacked_predictions
})
submission_df_stacked.to_csv('final_stacked_submission.csv', index=False)

