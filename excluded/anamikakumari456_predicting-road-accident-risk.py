import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# Load the data from the uploaded files
try:
    train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
    test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")
    sample_submission_df = pd.read_csv("/kaggle/input/playground-series-s5e10/sample_submission.csv")
except FileNotFoundError:
    print("Ensure the Kaggle dataset is added to your notebook environment.")
    # Add dummy dataframes to prevent errors in subsequent cells if files aren't found
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()


# Display the first few rows of the training data
print("Training Data:")
print(train_df.head())


train_df.info()


print(train_df.describe().T)


import matplotlib.pyplot as plt
import seaborn as sns

# Set a style for the plots for better aesthetics
sns.set_style("whitegrid")

# --- 1. Analyze the Target Variable (accident_risk) ---
plt.figure(figsize=(12, 6))
sns.histplot(train_df['accident_risk'], kde=True, bins=60, color='skyblue')
plt.title('Distribution of Accident Risk (Target)', fontsize=16)
plt.xlabel('Accident Risk', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.axvline(train_df['accident_risk'].mean(), color='red', linestyle='--', label=f"Mean: {train_df['accident_risk'].mean():.2f}")
plt.axvline(train_df['accident_risk'].median(), color='green', linestyle='-', label=f"Median: {train_df['accident_risk'].median():.2f}")
plt.legend()
plt.show()


# --- 2. Analyze Relationships: Categorical Features vs. Target ---
# We will use box plots to see how the distribution of accident_risk changes for each category.
categorical_cols = ['road_type', 'lighting', 'weather', 'time_of_day']
boolean_cols = ['road_signs_present', 'public_road', 'holiday', 'school_season']

for col in categorical_cols + boolean_cols:
    plt.figure(figsize=(12, 7))
    sns.boxplot(x=col, y='accident_risk', data=train_df)
    plt.title(f'Accident Risk vs. {col}', fontsize=16)
    plt.xlabel(col, fontsize=12)
    plt.ylabel('Accident Risk', fontsize=12)
    plt.xticks(rotation=15) # Rotate labels slightly for better readability
    plt.show()



# --- 3. Analyze Relationships: Numerical Features vs. Target (Correlation) ---
# A heatmap is the best way to visualize correlations.
numerical_cols = ['num_lanes', 'curvature', 'speed_limit', 'num_reported_accidents', 'accident_risk']
corr_matrix = train_df[numerical_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Numerical Features', fontsize=16)
plt.show()


import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import warnings

warnings.filterwarnings('ignore')

# --- Load Data (assuming train_df and test_df are already loaded) ---
# If not, uncomment these lines:
# train_df = pd.read_csv("train.csv")
# test_df = pd.read_csv("test.csv")

# --- 1. PREPROCESSING ---

# Store test IDs for the final submission file and drop the ID column
test_ids = test_df['id']
train_df = train_df.drop('id', axis=1)
test_df = test_df.drop('id', axis=1)

# Combine train and test sets for consistent encoding
# This ensures that if a category exists in test but not train (or vice-versa), the code won't break.
combined_df = pd.concat([train_df.drop('accident_risk', axis=1), test_df], ignore_index=True)

# Loop through all columns that are of type 'object' or 'bool' and apply LabelEncoder
for col in combined_df.select_dtypes(include=['object', 'bool']).columns:
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])

# Separate the combined dataframe back into training and testing sets
train_processed = combined_df.iloc[:len(train_df)]
test_processed = combined_df.iloc[len(train_df):]

print("--- Data Preprocessing Complete ---")
print("Processed training data head:")
print(train_processed.head())


# --- 2. MODEL TRAINING (LightGBM with 5-Fold Cross-Validation) ---

# Define our features (X) and target (y)
features = [col for col in train_processed.columns]
X = train_processed[features]
y = train_df['accident_risk']
X_test = test_processed[features]

# Setup the K-Fold cross-validation
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)

# Create arrays to store out-of-fold and test predictions
oof_preds = np.zeros(train_df.shape[0])
sub_preds = np.zeros(test_df.shape[0])

print("\n--- Starting Model Training... ---")

# Loop through each fold
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y)):
    X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y.iloc[valid_idx]

    # Define the LightGBM model parameters - these are a good starting point
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 2000,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 1,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'num_leaves': 31,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42 + n_fold,
        'boosting_type': 'gbdt',
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    # Store predictions for this fold
    oof_preds[valid_idx] = model.predict(X_valid)
    sub_preds += model.predict(X_test) / folds.n_splits
    
    fold_rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[valid_idx]))
    print(f"Fold {n_fold+1} RMSE: {fold_rmse}")

# Calculate and print the overall cross-validation score
final_rmse = np.sqrt(mean_squared_error(y, oof_preds))
print(f"\n--- Overall Cross-Validation RMSE: {final_rmse} ---")


# --- 3. CREATE SUBMISSION FILE ---

# As a good practice, clip predictions to be within the valid [0, 1] range
sub_preds = np.clip(sub_preds, 0, 1)

# Create the submission DataFrame
submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': sub_preds})

# Save the submission file
submission_df.to_csv('submission.csv', index=False)

print("\n--- Submission file 'submission.csv' created successfully! ---")
print("Submission head:")
print(submission_df.head())


# --- 1. Analyze Feature Importance from the Trained Model ---
# (This code assumes the 'model' variable from the last fold of your previous step is available)
# (It also assumes 'features' list is available)

# Create a dataframe for feature importances
feature_importance_df = pd.DataFrame()
feature_importance_df["feature"] = features
feature_importance_df["importance"] = model.feature_importances_

# Sort by importance and plot
plt.figure(figsize=(12, 10))
sns.barplot(x="importance", y="feature", data=feature_importance_df.sort_values(by="importance", ascending=False))
plt.title("LightGBM Feature Importance", fontsize=16)
plt.xlabel("Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.show()


# --- 1. LOAD DATA ---
print("--- Part 1: Loading Data & Engineering Features ---")
train_df = pd.read_csv("/kaggle/input/playground-series-s5e10/train.csv")
test_df = pd.read_csv("/kaggle/input/playground-series-s5e10/test.csv")

# Store IDs and the target variable for later use
test_ids = test_df['id']
y_target = train_df['accident_risk']


# --- 2. FEATURE ENGINEERING ---
# Combine train and test sets for consistent feature creation
combined_df = pd.concat([train_df.drop('accident_risk', axis=1), test_df], ignore_index=True)
combined_df = combined_df.drop('id', axis=1)

print("Creating new features...")

# a) Interaction Features (capturing combined effects)
combined_df['lighting_weather'] = combined_df['lighting'].astype(str) + '_' + combined_df['weather'].astype(str)
combined_df['road_x_lighting'] = combined_df['road_type'].astype(str) + '_' + combined_df['lighting'].astype(str)
combined_df['time_of_day_x_lighting'] = combined_df['time_of_day'].astype(str) + '_' + combined_df['lighting'].astype(str)

# b) Polynomial Features (capturing non-linear relationships)
combined_df['curvature_sq'] = combined_df['curvature']**2
combined_df['speed_limit_cubed'] = combined_df['speed_limit']**3

# c) Ratio and Density Features (creating more meaningful metrics)
epsilon = 1e-6 # A small constant to prevent division by zero
combined_df['speed_per_lane'] = combined_df['speed_limit'] / (combined_df['num_lanes'] + epsilon)
combined_df['accidents_per_lane'] = combined_df['num_reported_accidents'] / (combined_df['num_lanes'] + epsilon)
combined_df['curvature_x_speed'] = combined_df['curvature'] * combined_df['speed_limit']

print("\n--- Feature Engineering Complete ---")
print("Data head with new features:")
print(combined_df.head())


from sklearn.preprocessing import LabelEncoder

# --- 3. PREPROCESSING (ENCODING) ---
print("\n--- Part 2: Preprocessing and Preparing Data for Model ---")

# Select all columns of type 'object' or 'bool' to be encoded
for col in combined_df.select_dtypes(include=['object', 'bool']).columns:
    le = LabelEncoder()
    # Fit on the entire column and transform it
    combined_df[col] = le.fit_transform(combined_df[col])

# Separate the combined dataframe back into training (X) and testing (X_test) sets
X = combined_df.iloc[:len(train_df)]
X_test = combined_df.iloc[len(train_df):]

print(f"--- Data is now ready for training with {X.shape[1]} features ---")
print("\nProcessed training data head:")
print(X.head())


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb

# --- 4. MODEL TRAINING (LightGBM with 5-Fold Cross-Validation) ---
print("\n--- Part 3: Training Model, Evaluating, and Submitting ---")

NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(X_test.shape[0])

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y_target)):
    X_train, y_train = X.iloc[train_idx], y_target.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y_target.iloc[valid_idx]

    params = {
        'objective': 'regression_l1', 'metric': 'rmse', 'n_estimators': 2000,
        'learning_rate': 0.01, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
        'bagging_freq': 1, 'lambda_l1': 0.1, 'lambda_l2': 0.1,
        'num_leaves': 31, 'verbose': -1, 'n_jobs': -1, 'seed': 42 + n_fold,
        'boosting_type': 'gbdt',
    }

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    oof_preds[valid_idx] = model.predict(X_valid)
    sub_preds += model.predict(X_test) / folds.n_splits
    fold_rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[valid_idx]))
    print(f"Fold {n_fold+1} RMSE: {fold_rmse}")

# --- 5. EVALUATE & SUBMIT ---
final_rmse = np.sqrt(mean_squared_error(y_target, oof_preds))
baseline_rmse = 0.056303 # Your score from the previous run
print(f"\n--- Baseline CV RMSE: {baseline_rmse} ---")
print(f"--- New CV RMSE with Engineered Features: {final_rmse} ---")

if final_rmse < baseline_rmse:
    print("\n✅ SUCCESS! The new features improved the score.")
else:
    print("\n❌ The new features did not improve the score. Time to re-evaluate.")

sub_preds = np.clip(sub_preds, 0, 1)
submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': sub_preds})
submission_df.to_csv('submission_with_features.csv', index=False)
print("\n--- New submission file 'submission_with_features.csv' created successfully! ---")


import optuna
from sklearn.model_selection import train_test_split

# --- Part 1: Defining the Optimization Objective ---
# We will use the dataset WITH your engineered features, as they might
# become powerful once the model is tuned properly.
# (The 'X' and 'y_target' variables should still be available from your last run)

def objective(trial):
    # Define the search space for the hyperparameters
    params = {
        'objective': 'regression_l1',
        'metric': 'rmse',
        'n_estimators': 1000, # We use a fixed large number and rely on early stopping
        'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 300),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'seed': 42,
        'n_jobs': -1,
        'verbose': -1,
        'boosting_type': 'gbdt',
    }

    # For speed, we will use a single validation split during the search
    # A full CV would be too slow for many trials.
    X_train, X_valid, y_train, y_valid = train_test_split(X, y_target, test_size=0.2, random_state=42)

    model = lgb.LGBMRegressor(**params)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(50, verbose=False)])

    preds = model.predict(X_valid)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))
    return rmse

print("--- Part 1: Optimization Objective Defined ---")


# --- Part 2: Running the Optimization Search ---
# We create a 'study' object and ask it to minimize our objective function

# To run a new study:
study = optuna.create_study(direction='minimize')
# To continue a study you already started:
# study = optuna.create_study(direction='minimize', study_name="lgbm_tuning_s5e10", storage="sqlite:///lgbm_tuning.db", load_if_exists=True)


# Start the search. More trials can lead to better results but will take longer.
study.optimize(objective, n_trials=50)

print("\n--- Optimization Complete ---")
print("Number of finished trials: ", len(study.trials))
print("Best trial:")
best_trial = study.best_trial
print("  Value (RMSE): ", best_trial.value)
print("  Params: ")
for key, value in best_trial.params.items():
    print(f"    {key}: {value}")

# Store the best parameters in a variable for the next step
best_params = best_trial.params


# --- Part 3: Re-training with Optimal Parameters ---
print("\n--- Part 3: Re-training Final Model with Best Parameters ---")

# We use our original 5-fold CV setup
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds = np.zeros(X.shape[0])
sub_preds = np.zeros(X_test.shape[0])

# Add some fixed parameters to the best ones found by Optuna
best_params['objective'] = 'regression_l1'
best_params['metric'] = 'rmse'
best_params['n_estimators'] = 2000 # Use a high number with early stopping
best_params['n_jobs'] = -1
best_params['verbose'] = -1

for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y_target)):
    X_train, y_train = X.iloc[train_idx], y_target.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y_target.iloc[valid_idx]
    
    # Use the best_params found by Optuna
    model = lgb.LGBMRegressor(**best_params, seed=42 + n_fold)
    model.fit(X_train, y_train,
              eval_set=[(X_valid, y_valid)],
              eval_metric='rmse',
              callbacks=[lgb.early_stopping(100, verbose=False)])

    oof_preds[valid_idx] = model.predict(X_valid)
    sub_preds += model.predict(X_test) / folds.n_splits
    fold_rmse = np.sqrt(mean_squared_error(y_valid, oof_preds[valid_idx]))
    print(f"Fold {n_fold+1} RMSE: {fold_rmse}")

# --- Final Evaluation & Submission ---
final_rmse_tuned = np.sqrt(mean_squared_error(y_target, oof_preds))
print(f"\n--- Baseline CV RMSE: 0.056303 ---")
print(f"--- Tuned Model CV RMSE: {final_rmse_tuned} ---")

if final_rmse_tuned < 0.056303:
    print("\n✅ SUCCESS! Hyperparameter tuning improved the score significantly.")
else:
    print("\n⚠️ The tuned model did not improve the score. This is unusual but possible.")

# Create the new submission file
sub_preds = np.clip(sub_preds, 0, 1)
submission_df = pd.DataFrame({'id': test_ids, 'accident_risk': sub_preds})
submission_df.to_csv('submission_tuned.csv', index=False)
print("\n--- New submission file 'submission_tuned.csv' created successfully! ---")


import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

# --- Part 1: Training a New XGBoost Model ---
print("--- Training a new XGBoost model ---")

# The 'X', 'X_test', and 'y_target' variables should still be available from your last run.
# If not, you may need to re-run the data loading and feature engineering parts.

# --- Cross-Validation Setup ---
NFOLDS = 5
folds = KFold(n_splits=NFOLDS, shuffle=True, random_state=42)
oof_preds_xgb = np.zeros(X.shape[0])
sub_preds_xgb = np.zeros(X_test.shape[0])

# --- Model Training Loop ---
for n_fold, (train_idx, valid_idx) in enumerate(folds.split(X, y_target)):
    X_train, y_train = X.iloc[train_idx], y_target.iloc[train_idx]
    X_valid, y_valid = X.iloc[valid_idx], y_target.iloc[valid_idx]

    # Define XGBoost parameters (a good starting set)
    # Note: enable 'gpu_hist' for much faster training if you have a GPU enabled in Kaggle
    params_xgb = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'eta': 0.02, # learning_rate
        'max_depth': 7,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'seed': 42 + n_fold,
        'n_jobs': -1,
        # 'tree_method': 'gpu_hist' # Uncomment for GPU
    }

    model_xgb = xgb.XGBRegressor(**params_xgb, n_estimators=2000)
    model_xgb.fit(X_train, y_train,
                  eval_set=[(X_valid, y_valid)],
                  verbose=False,
                  early_stopping_rounds=100)

    oof_preds_xgb[valid_idx] = model_xgb.predict(X_valid)
    sub_preds_xgb += model_xgb.predict(X_test) / folds.n_splits
    fold_rmse = np.sqrt(mean_squared_error(y_valid, oof_preds_xgb[valid_idx]))
    print(f"Fold {n_fold+1} XGBoost RMSE: {fold_rmse}")

# --- Evaluate XGBoost Model ---
final_rmse_xgb = np.sqrt(mean_squared_error(y_target, oof_preds_xgb))
print(f"\n--- Overall XGBoost CV RMSE: {final_rmse_xgb} ---")

# Store the predictions for the next step
# We are creating a new variable so we don't overwrite our previous best predictions
lgbm_test_preds = sub_preds # Assuming 'sub_preds' holds the tuned LGBM predictions from the last step
xgb_test_preds = sub_preds_xgb


# --- Part 2: Blending the Predictions ---
print("\n--- Blending LGBM and XGBoost Predictions ---")

# It's good practice to load your best single-model submission to ensure you're using the right predictions
try:
    # 'sub_preds' should still hold your tuned LGBM predictions. If not, load them from the CSV.
    # submission_lgbm = pd.read_csv('submission_tuned.csv')
    # lgbm_test_preds = submission_lgbm['accident_risk']
    print("Using LightGBM predictions from the previous step.")
except FileNotFoundError:
    print("Could not find 'submission_tuned.csv'. Make sure the 'sub_preds' variable is correct.")
    # Handle error appropriately if the file is missing


# Simple 50/50 Blend
blended_preds = 0.5 * lgbm_test_preds + 0.5 * xgb_test_preds

# It's always a good idea to clip the final predictions
blended_preds = np.clip(blended_preds, 0, 1)

print("Blending complete.")


# --- Part 3: Creating the Final Ensemble Submission ---

# test_ids should still be available from your previous runs
submission_ensemble = pd.DataFrame({'id': test_ids, 'accident_risk': blended_preds})
submission_ensemble.to_csv('submission_ensemble.csv', index=False)

print("\n--- New submission file 'submission_ensemble.csv' created successfully! ---")
print("\nSubmission head:")
print(submission_ensemble.head())

