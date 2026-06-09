!pip install --upgrade lightgbm


import lightgbm
print(lightgbm.__version__)


import lightgbm as lgb
import numpy as np
import pandas as pd
from datetime import datetime
import joblib

# LightGBM
from lightgbm import LGBMRegressor

# Sklearn
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Holidays
from holidays import CountryHoliday

# Optuna
import optuna

# --------------------------------------------------------
# LightGBM Version Check
# --------------------------------------------------------
print("LightGBM Version:", lgb.__version__)

# Ensure you're using LightGBM version 4.5.0
assert lgb.__version__ == "4.5.0", "Please ensure LightGBM version 4.5.0 is installed."

# --------------------------------------------------------
# MAPE Metric Definition
# --------------------------------------------------------
def mape(y_true, y_pred, eps=1e-9):
    """
    Mean Absolute Percentage Error.
    y_true, y_pred are arrays of actual and predicted values (in normal scale).
    """
    y_true = np.clip(y_true, eps, None)  # Avoid division by zero
    return np.mean(np.abs(y_true - y_pred) / y_true)

# --------------------------------------------------------
# Feature Engineering Function
# --------------------------------------------------------
def create_features(df):
    """
    Creates date-based and holiday features.
    Drops the 'date' column at the end.
    """
    df['date'] = pd.to_datetime(df['date'])

    # Basic date features
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek
    df['day_of_year'] = df['date'].dt.dayofyear
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)

    # Weekend flag
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Cyclical encoding for month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Cyclical encoding for day_of_year
    df['dayofyear_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['dayofyear_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)

    # Simple holiday feature
    df['holiday'] = 0
    unique_countries = df['country'].unique().tolist()

    # Define the range of years to consider for holidays
    YEARS_TO_CONSIDER = range(2010, 2025)
    holiday_map = {c: CountryHoliday(c, years=YEARS_TO_CONSIDER) for c in unique_countries}

    # Vectorized approach for holidays to improve performance
    for country, holidays in holiday_map.items():
        df.loc[(df['country'] == country) & (df['date'].isin(holidays)), 'holiday'] = 1

    # Drop the original date column
    df = df.drop(columns=['date'])
    return df

# --------------------------------------------------------
# Time-Based Cross-Validation Split Function
# --------------------------------------------------------
def time_based_cv_splits(df, date_col, n_splits=5):
    """
    Generate (train_idx, val_idx) pairs for an expanding time-based CV.
    """
    # Sort by date
    df_sorted = df.sort_values(date_col)
    unique_dates = df_sorted[date_col].unique()
    n_dates = len(unique_dates)

    # Check if there are enough unique dates
    if n_dates < (n_splits + 1):
        print(f"WARNING: You have {n_dates} unique dates but requested {n_splits} folds.")
        print("Some folds may be empty or invalid; consider reducing n_splits.\n")

    # Calculate segment size
    segment_size = n_dates // (n_splits + 1)
    if segment_size == 0:
        print("ERROR: segment_size = 0 => Not enough dates to form the desired folds.")
        return []

    splits = []
    for i in range(n_splits):
        train_end = (i + 1) * segment_size
        val_end = (i + 2) * segment_size

        # Clamp val_end to the total number of unique dates
        val_end = min(val_end, n_dates)

        train_dates = unique_dates[:train_end]
        val_dates = unique_dates[train_end:val_end]

        # If no validation dates, skip
        if len(val_dates) == 0:
            continue

        train_idx = df_sorted[df_sorted[date_col].isin(train_dates)].index
        val_idx = df_sorted[df_sorted[date_col].isin(val_dates)].index

        # Skip if train_idx or val_idx is empty
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue

        splits.append((train_idx, val_idx))

    return splits

# --------------------------------------------------------
# Load Data
# --------------------------------------------------------
# Replace the file paths with your actual data paths
train = pd.read_csv("/kaggle/input/playground-series-s5e1/train.csv")
test  = pd.read_csv("/kaggle/input/playground-series-s5e1/test.csv")
submission = pd.read_csv("/kaggle/input/playground-series-s5e1/sample_submission.csv")

# Preserve original date in a separate column (for CV splitting)
train['original_date'] = pd.to_datetime(train['date'])

# --------------------------------------------------------
# Create Features
# --------------------------------------------------------
train = create_features(train)
test  = create_features(test)

# --------------------------------------------------------
# Prepare Target
# --------------------------------------------------------
target_col = "num_sold"
train.dropna(subset=[target_col], inplace=True)
y = np.log1p(train[target_col])  # Log-transform the target for better performance
X = train.drop(columns=[target_col])

# --------------------------------------------------------
# Handle Categorical Features
# --------------------------------------------------------
# Updated: Remove 'year' and 'month' from categorical columns
cat_cols = ['country', 'store', 'product']
for c in cat_cols:
    # Fill missing values with 'missing' before converting to string
    X[c] = X[c].fillna('missing').astype(str)
    test[c] = test[c].fillna('missing').astype(str)

# Verify that there are no remaining NaNs in categorical columns
print("Missing values in categorical columns after preprocessing:")
print(X[cat_cols].isnull().sum())
print(test[cat_cols].isnull().sum())

# --------------------------------------------------------
# Ordinal Encoding (fit once on entire training set)
# --------------------------------------------------------
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
oe.fit(X[cat_cols])

# --------------------------------------------------------
# Insert Date Column (for CV Splits)
# --------------------------------------------------------
X['original_date'] = train['original_date']
X = X.sort_values('original_date')
y = y.loc[X.index]

# --------------------------------------------------------
# Create Time-Based CV Splits
# --------------------------------------------------------
n_folds = 5  # Number of cross-validation folds
fold_splits = time_based_cv_splits(X, date_col='original_date', n_splits=n_folds)

# --------------------------------------------------------
# Data Rescaling
# --------------------------------------------------------
# Initialize the scaler
scaler = StandardScaler()

# Identify numerical columns (excluding categorical and date-related features)
numerical_cols = ['year', 'month', 'day_of_week', 'day_of_year', 'week_of_year',
                  'month_sin', 'month_cos', 'dayofyear_sin', 'dayofyear_cos', 'is_weekend', 'holiday']

# Fit the scaler on the training data
scaler.fit(X[numerical_cols])

# Transform both training and test data
X[numerical_cols] = scaler.transform(X[numerical_cols])
test[numerical_cols] = scaler.transform(test[numerical_cols])

# Save the scaler for future use
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
scaler_filename = f"scaler_{timestamp}.joblib"
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to: {scaler_filename}")

# --------------------------------------------------------
# Optuna Objective Function
# --------------------------------------------------------
def objective(trial):
    """
    Optuna objective function:
    1. Suggest hyperparameters.
    2. Perform time-based CV.
    3. Return the average MAPE.
    """
    # 1. Hyperparameters to tune for linear trees
    params = {
        "boosting_type": "gbdt",      # Standard Gradient Boosting Decision Tree
        "linear_tree": True,          # Enable linear trees
        "n_estimators": trial.suggest_int("n_estimators", 1000, 3000, step=100),
        "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),   # L1 regularization
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True), # L2 regularization
        "random_state": 42,
        "device": "cpu",               # Ensure CPU is used; change to 'gpu' if GPU is available
        "tree_learner": "serial",      # Required for linear trees
    }

    # 2. Time-based CV
    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(fold_splits):
        X_train = X.loc[train_idx].copy()
        X_val   = X.loc[val_idx].copy()
        y_train = y.loc[train_idx]
        y_val   = y.loc[val_idx]

        # Encode categorical features
        try:
            X_train[cat_cols] = oe.transform(X_train[cat_cols])
            X_val[cat_cols]   = oe.transform(X_val[cat_cols])
        except Exception as e:
            print(f"Encoding error in fold {fold}: {e}")
            return np.inf  # Assign a large error to prune this trial

        # Drop date column from features
        X_train_fold = X_train.drop(columns=['original_date'])
        X_val_fold   = X_val.drop(columns=['original_date'])

        # 3. Train LightGBM using callbacks for early stopping
        model = LGBMRegressor(**params)
        try:
            model.fit(
                X_train_fold,
                y_train,
                eval_set=[(X_val_fold, y_val)],
                eval_metric="mape",
                callbacks=[lgb.early_stopping(stopping_rounds=50)]
            )
        except Exception as e:
            print(f"Training error in fold {fold}: {e}")
            return np.inf  # Assign a large error to prune this trial

        # Predict on validation set
        try:
            val_preds_log = model.predict(X_val_fold, num_iteration=model.best_iteration_)
            val_preds = np.expm1(val_preds_log)  # Inverse log-transform
            val_preds_rounded = np.round(val_preds)  # Rounding to nearest integer
        except Exception as e:
            print(f"Prediction error in fold {fold}: {e}")
            return np.inf  # Assign a large error to prune this trial

        # Calculate MAPE for the current fold
        try:
            fold_mape_val = mape(np.expm1(y_val), val_preds_rounded)
            fold_scores.append(fold_mape_val)
        except Exception as e:
            print(f"MAPE calculation error in fold {fold}: {e}")
            return np.inf  # Assign a large error to prune this trial

    # 4. Return average MAPE over all folds
    mean_mape_val = np.mean(fold_scores)
    return mean_mape_val

# --------------------------------------------------------
# Optuna Study
# --------------------------------------------------------
# Create an Optuna study object and optimize the objective function
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=500)

# --------------------------------------------------------
# Print Best Trial Information
# --------------------------------------------------------
print("\nBest trial:")
print(f"  Value (MAPE): {study.best_trial.value:.5f}")
print("  Params:")
for k, v in study.best_trial.params.items():
    print(f"    {k}: {v}")

# --------------------------------------------------------
# Retrain on Full Data with Best Hyperparameters
# --------------------------------------------------------
best_params = study.best_trial.params.copy()
best_params["boosting_type"] = "gbdt"      # Standard Gradient Boosting Decision Tree
best_params["linear_tree"] = True          # Enable linear trees
best_params["n_estimators"] = int(best_params["n_estimators"])  # Ensure integer
best_params["reg_alpha"] = float(best_params["reg_alpha"])      # Ensure float
best_params["reg_lambda"] = float(best_params["reg_lambda"])    # Ensure float
best_params["random_state"] = 42                           # Ensure reproducibility
best_params["device"] = "cpu"                              # Ensure CPU is used; change to 'gpu' if available
best_params["tree_learner"] = "serial"                     # Required for linear trees

# Encode full training data
X_enc = X.drop(columns=['original_date']).copy()
X_enc[cat_cols] = oe.transform(X_enc[cat_cols])

# Rescale numerical features in full training data
X_enc[numerical_cols] = scaler.transform(X_enc[numerical_cols])

# Verify no NaNs exist in categorical columns after encoding
if X_enc[cat_cols].isnull().any().any():
    print("Warning: NaNs found in categorical columns after encoding.")
    print(X_enc[cat_cols].isnull().sum())

# Train the final model with the best hyperparameters
final_model = LGBMRegressor(**best_params)
try:
    final_model.fit(
        X_enc, y,
        eval_set=[(X_enc, y)],
        eval_metric='mape',
        callbacks=[lgb.early_stopping(stopping_rounds=50)]
    )
    print("\nFinal model trained on full dataset with best hyperparameters using CPU.")
except Exception as e:
    print(f"Error during final model training: {e}")

# --------------------------------------------------------
# Predict on Test Set
# --------------------------------------------------------
test_enc = test.copy()
test_enc[cat_cols] = oe.transform(test_enc[cat_cols])

# Rescale numerical features in test data
test_enc[numerical_cols] = scaler.transform(test_enc[numerical_cols])

# If the test set is missing any columns (unlikely, but possible), add them
missing_cols = set(X_enc.columns) - set(test_enc.columns)
if missing_cols:
    print("Missing columns in test:", missing_cols)
    for col in missing_cols:
        test_enc[col] = 0  # Assign a default value

# Reorder test columns to match training
test_enc = test_enc[X_enc.columns]

# Predict and round
try:
    test_preds_log = final_model.predict(test_enc)
    test_preds = np.expm1(test_preds_log)  # Inverse log-transform
    test_preds_rounded = np.round(test_preds).astype(int)  # Rounding to nearest integer
except Exception as e:
    print(f"Error during prediction on test set: {e}")
    test_preds_rounded = np.zeros(len(test_enc), dtype=int)  # Assign default predictions

# --------------------------------------------------------
# Prepare Submission
# --------------------------------------------------------
submission[target_col] = test_preds_rounded
sub_filename = f"sub_linear_{timestamp}.csv"  # Updated filename to reflect linear trees
submission.to_csv(sub_filename, index=False)
print(f"Submission saved to: {sub_filename}")

# --------------------------------------------------------
# Save the Trained Model
# --------------------------------------------------------
model_filename = f"model_optuna_linear_{timestamp}.joblib"  # Updated filename to reflect linear trees
joblib.dump(final_model, model_filename)
print(f"Model saved to: {model_filename}")

# --------------------------------------------------------
# Save the Ordinal Encoder
# --------------------------------------------------------
oe_filename = f"ordinal_encoder_{timestamp}.joblib"
joblib.dump(oe, oe_filename)
print(f"Ordinal Encoder saved to: {oe_filename}")

# --------------------------------------------------------
# Save the Scaler
# --------------------------------------------------------
scaler_filename = f"scaler_{timestamp}.joblib"
joblib.dump(scaler, scaler_filename)
print(f"Scaler saved to: {scaler_filename}")


