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
import optuna 
import xgboost as xgb
import catboost as cb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.preprocessing import OrdinalEncoder , LabelEncoder
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import optuna


df_train = pd.read_csv('/kaggle/input/playground-series-s5e6/train.csv')
df_test = pd.read_csv('/kaggle/input/playground-series-s5e6/test.csv')
df_sub = pd.read_csv('/kaggle/input/playground-series-s5e6/sample_submission.csv')


df_train = df_train.drop(columns=['id'])
df_test = df_test.drop(columns=['id'])


train = df_train
test = df_test


train.info()


test.info()


import optuna
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold # For K-Fold CV
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import gc # For garbage collection, helpful in CV loops





# --- 2. Separate X and y, and Preprocessing (Encoding) ---

# Define target variable
y = df_train['Fertilizer Name']
# Define features (all columns except 'Fertilizer Name')
X = df_train.drop(columns=['Fertilizer Name'])

# Identify categorical columns for Ordinal Encoding (excluding the target)
# Ensure this correctly identifies your actual categorical columns
cat_cols = X.select_dtypes(include='object').columns
print(f"\nCategorical columns for Ordinal Encoding: {list(cat_cols)}")

# Initialize OrdinalEncoder
ordinal_encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)

# Apply Ordinal Encoding to training and test features
X[cat_cols] = ordinal_encoder.fit_transform(X[cat_cols].astype(str))
df_test[cat_cols] = ordinal_encoder.transform(df_test[cat_cols].astype(str))

# Label Encode the target variable (y)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded)) # Get number of classes from encoded y

print(f"\nNumber of classes (Fertilizer Names): {num_classes}")
print(f"X after encoding categorical features (first 5 rows):\n{X.head()}")
print(f"y_encoded (first 5):\n{y_encoded[:5]}")

# --- 3. MAPK Function ---
def mapk(actual, predicted, k=3):
    def apk(a, p, k):
        p = p[:k] # Consider only top k predictions
        score = 0.0
        hits = 0
        seen = set() # To ensure unique predictions are counted only once
        for i, pred in enumerate(p):
            if pred in a and pred not in seen:
                hits += 1
                score += hits / (i + 1.0) # Precision at current recall point
                seen.add(pred)
        # Handle case where actual list might be empty or k is 0
        return score / min(len(a), k) if min(len(a), k) > 0 else 0.0
    
    # Ensure 'actual' is treated as a list of lists for consistency with apk
    # (e.g., if actual is [1, 0, 2], apk expects [[1], [0], [2]])
    # The zip creates pairs (actual_single_value, predicted_row)
    return np.mean([apk([a], p, k) for a, p in zip(actual, predicted)])

# --- 4. Optuna Objective Function with K-Fold for XGBoost ---
def objective(trial):
    # Only XGBoost for this specific request
    model_type = "xgboost" # Fixed to xgboost as per request

    # Hyperparameters for XGBoost
    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "eval_metric": "mlogloss",
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
        "max_depth": trial.suggest_int("max_depth", 5, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 0.9),
        "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.6, 0.9),
        "colsample_bynode": trial.suggest_float("colsample_bynode", 0.6, 0.9),
        "n_estimators": trial.suggest_int("n_estimators", 500, 2000), # Number of boosting rounds
        "verbosity": 0, # Suppress verbose output during training
        "random_state": 42,
        # Uncomment for GPU if available and configured:
        # "tree_method": "gpu_hist",
        # "device": "cuda",
    }

    # K-Fold Cross-Validation setup
    n_splits = 5 # As requested, 5 folds
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_scores = []

    # Loop through each fold
    for fold, (train_idx, valid_idx) in enumerate(kf.split(X, y_encoded)):
        # Using .iloc for DataFrame X and direct indexing for numpy array y_encoded
        fold_train_X, fold_valid_X = X.iloc[train_idx], X.iloc[valid_idx]
        fold_train_y, fold_valid_y = y_encoded[train_idx], y_encoded[valid_idx]

        model = XGBClassifier(**params)
        model.fit(fold_train_X, fold_train_y,
                  eval_set=[(fold_valid_X, fold_valid_y)],
                  early_stopping_rounds=100, # Early stopping within each fold
                  verbose=False)

        # Predict probabilities and calculate score for the current fold
        pred_probs = model.predict_proba(fold_valid_X)
        # Get indices of top K (e.g., 3) predictions for each sample
        top_3_preds = np.argsort(pred_probs, axis=1)[:, -3:][:, ::-1]
        
        fold_score = mapk(fold_valid_y.flatten(), top_3_preds)
        fold_scores.append(fold_score)

        # Clear memory after each fold to prevent memory issues
        del model
        gc.collect()

    # Return the average score across all folds for this trial
    return np.mean(fold_scores)

# --- 5. Run Optuna Study ---
print("\nStarting Optuna study for XGBoost with K-Fold CV...")
# Using TPESampler with a seed for reproducibility of the search process
study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
# n_trials=5 as requested
study.optimize(objective, n_trials=5)

print("\nâœ… Optuna study finished.")
print("Best MAP@3:", study.best_value)
print("Best params:", study.best_params)

# --- 6. Training Final Model with Best Params and Prediction ---
best_params = study.best_params.copy()

# Add fixed parameters for the final model training
final_params = {
    "objective": "multi:softprob",
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "n_estimators": best_params.pop("n_estimators"), # Use the best n_estimators found
    "verbosity": 0,
    "random_state": 42,
    **best_params # Add all other best parameters
}

# Uncomment for GPU if used during tuning:
# final_params["tree_method"] = "gpu_hist"
# final_params["device"] = "cuda"

print(f"\nğŸš€ Training final XGBoost model on full dataset with best params...")
final_model = XGBClassifier(**final_params)
# Ensure X is passed as a DataFrame and y_encoded as a numpy array
final_model.fit(X, y_encoded) # Train on the full dataset

# --- 7. Predict on test and create submission ---
print("\nGenerating predictions for test data...")
# Ensure df_test is passed as a DataFrame
test_probs = final_model.predict_proba(df_test)
top_3_preds = np.argsort(test_probs, axis=1)[:, -3:][:, ::-1]

# Decode numerical predictions back to original labels
top_3_labels = le.inverse_transform(top_3_preds.ravel()).reshape(top_3_preds.shape)

# Create submission DataFrame
# Convert each label to string before joining
submission = pd.DataFrame({
    "id": df_sub["id"],
    "Fertilizer Name": [' '.join(str(label) for label in row) for row in top_3_labels]})

submission_filename = "submission_xgboost_kfold.csv"
submission.to_csv(submission_filename, index=False)
print(f"ğŸ“� Submission saved to '{submission_filename}'")




