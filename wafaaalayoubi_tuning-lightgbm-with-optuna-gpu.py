# --- Core Libraries ---
import numpy as np
import pandas as pd
import warnings
import torch # We use torch to easily check for GPU availability

# --- Machine Learning & Optimization ---
import lightgbm as lgb
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


# --- Tweak Settings ---
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)
# This will make Optuna's output cleaner
optuna.logging.set_verbosity(optuna.logging.WARNING)


# --- GPU Check ---
if torch.cuda.is_available():
    print("âœ… GPU is available and ready!")
    # Set the device for LightGBM
    LGBM_DEVICE = 'gpu'
else:
    print("âš ï¸� GPU not available, switching to CPU. This will be slower.")
    LGBM_DEVICE = 'cpu'


# --- Load Data ---
print("\nLoading data...")
try:
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e10/train.csv')
except FileNotFoundError:
    print("Please adjust file paths for local execution.")

print("Data loaded successfully!")
print(f"Training data shape: {train_df.shape}")


def create_features(df):
    """
    Creates the winning BASELINE feature set.
    This robust version works for both train and test dataframes.
    """

    # Use a copy to avoid modifying the original dataframe
    df_copy = df.copy()

    # Drop id and leaky columns. errors='ignore' prevents errors if columns are already gone.
    df_copy = df_copy.drop(columns=['id'], errors='ignore') 
    
    # Convert booleans
    boolean_features = ['road_signs_present', 'public_road', 'holiday', 'school_season']
    for col in boolean_features:
        if col in df_copy.columns:
            df_copy[col] = df_copy[col].astype(int)

    # One-Hot Encode
    categorical_features = ['road_type', 'lighting', 'weather', 'time_of_day', 'num_lanes']
    df_copy = pd.get_dummies(df_copy, columns=categorical_features, drop_first=False)

    # Separate features and target ONLY if target exists
    if 'accident_risk' in df_copy.columns:
        X = df_copy.drop(columns=['accident_risk'])
        y = df_copy['accident_risk']
        return X, y
    else: # This is test data
        X = df_copy
        return X, None


# --- Generate the feature set for tuning ---
print("Preparing data using the winning baseline pipeline...")
X, y = create_features(train_df)
print("Data preparation complete!")
print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")


def objective(trial):
    """
    The objective function for Optuna to optimize.
    It takes a trial object, suggests hyperparameters, runs CV, and returns the mean RMSE.
    """
    
    # 1. Define the hyperparameter search space
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': trial.suggest_int('n_estimators', 400, 2000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 5, 16),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.001, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 1.0),
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'device': LGBM_DEVICE # CRITICAL: This enables GPU training
    }
    
    # 2. Set up cross-validation
    N_SPLITS = 3
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    rmse_scores = []
    
    # 3. Run the cross-validation loop
    for train_index, val_index in kf.split(X):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]
        
        model = lgb.LGBMRegressor(**params)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  eval_metric='rmse',
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        rmse_scores.append(rmse)
        
    # 4. Return the mean RMSE
    return np.mean(rmse_scores)



# --- Just a confirmation that the function is defined ---
print("Objective function for Optuna has been defined successfully.")


# --- 1. Create a study object ---
# We specify the direction as 'minimize' because we want the lowest RMSE.
study = optuna.create_study(direction='minimize', study_name='LGBM Optimization')


# --- 2. Run the optimization ---
# N_TRIALS controls how many different hyperparameter combinations Optuna will test.
# 50 is a good starting point. Increase to 100 or more for a more exhaustive search.
N_TRIALS = 50 
print(f"Starting Optuna study with {N_TRIALS} trials...")

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)


# --- 3. Print the results of the best trial ---
print("\n\n" + "="*50)
print("          OPTIMIZATION STUDY COMPLETE")
print("="*50)
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial's validation RMSE: {study.best_value:.6f}")
print("\n--- Best Hyperparameters ---")
# Prettily print the best parameters
best_params = study.best_params
for key, value in best_params.items():
    print(f"{key}: {value}")


# ===================================================================
#  STEP 4: RUN THE OPTUNA STUDY
# ===================================================================

# --- 1. Create a study object ---
study = optuna.create_study(direction='minimize', study_name='LGBM Optimization')

# --- 2. Run the optimization ---
N_TRIALS = 50 
print(f"Starting Optuna study with {N_TRIALS} trials...")

study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)

# --- 3. Print the results of the best trial ---
print("\n\n" + "="*50)
print("          OPTIMIZATION STUDY COMPLETE")
print("="*50)
print(f"Number of finished trials: {len(study.trials)}")
print(f"Best trial's validation RMSE: {study.best_value:.6f}")
print("\n--- Best Hyperparameters ---")
best_params = study.best_params
for key, value in best_params.items():
    print(f"{key}: {value}")


# ===================================================================
#  STEP 5: CREATE FINAL SUBMISSION WITH TUNED PARAMETERS
# ===================================================================

print("\n\n" + "="*50)
print("          CREATING FINAL SUBMISSION FILE")
print("="*50)

# --- 1. Load and process the official test data ---
print("Loading and processing test data...")
test_df = pd.read_csv('/kaggle/input/playground-series-s5e10/test.csv')
test_ids = test_df['id']
X_test, _ = create_features(test_df)
# Align columns to ensure consistency
X_test = X_test.reindex(columns=X.columns, fill_value=0)
print("Test data processed successfully.")


# --- 2. Initialize the final model with the BEST parameters from Optuna ---
# We also add back some fixed parameters needed for prediction.
final_params = best_params.copy()
final_params.update({
    'objective': 'regression_l1',
    'metric': 'rmse',
    'seed': 42,
    'n_jobs': -1,
    'verbose': -1,
    'device': LGBM_DEVICE
})

final_model = lgb.LGBMRegressor(**final_params)

# --- 3. Train the final model on 100% of the training data ---
print("\nTraining final model on 100% of the training data...")
final_model.fit(X, y)
print("Final model training complete!")


# --- 4. Generate predictions and create the submission file ---
print("Generating final predictions...")
final_predictions = final_model.predict(X_test)

submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': final_predictions})
submission_df['accident_risk'] = submission_df['accident_risk'].clip(0, 1)
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully with tuned parameters!")
print("This is ready to submit!")
display(submission_df.head())

