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
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
import gc



# --- Utility Function for Memory Reduction ---
def reduce_mem_usage(df, verbose=True):
    """
    Iterate through all the columns of a dataframe and modify the data type
    to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object and col_type.name != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # Using float32 instead of float16 for better compatibility with models
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(f'Memory usage after optimization is: {end_mem:.2f} MB')
        print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')

    return df



# --- 1. Load Data using full Kaggle paths ---
try:
    # UPDATED: Using the full paths you provided
    train_df = pd.read_csv('/kaggle/input/playground-series-s5e7/train.csv')
    test_df = pd.read_csv('/kaggle/input/playground-series-s5e7/test.csv')
except FileNotFoundError as e:
    print(f"Error loading data: {e}")
    print("Please ensure the file paths are correct for your Kaggle environment.")
    exit()

print("Data loaded successfully.")
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# --- 2. Preprocessing ---
X = train_df.drop('Personality', axis=1)
y = train_df['Personality']
X_test = test_df.copy()

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns
numerical_cols = X.select_dtypes(include=np.number).columns.drop('id')

# Impute missing values (numerical)
num_imputer = SimpleImputer(strategy='median')
X[numerical_cols] = num_imputer.fit_transform(X[numerical_cols])
X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

# Impute missing values (categorical)
cat_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])

# Convert categorical columns to 'category' dtype for LightGBM
for col in categorical_cols:
    X[col] = X[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Encode target variable
le_personality = LabelEncoder()
y_encoded = le_personality.fit_transform(y)

print("Preprocessing complete.")

# Drop ID columns
X = X.drop('id', axis=1)
X_test_ids = X_test['id']
X_test = X_test.drop('id', axis=1)
del train_df, test_df # Free up memory
gc.collect()



# --- 3. Reduce Memory Usage ---
print("\nOptimizing memory usage...")
X = reduce_mem_usage(X)
X_test = reduce_mem_usage(X_test)
gc.collect()

# --- 4. Model Training (Single Model without CV) ---
print("\nSplitting data for training and validation...")
# Create a validation set for early stopping
X_train, X_val, y_train, y_val = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

lgb_params = {
    'objective': 'binary',
    'metric': 'binary_logloss',
    'n_estimators': 2000,
    'learning_rate': 0.02,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 1,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'num_leaves': 31,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
    'boosting_type': 'gbdt',
}

print("Starting model training...")
model = lgb.LGBMClassifier(**lgb_params)

model.fit(X_train, y_train,
          eval_set=[(X_val, y_val)],
          eval_metric='accuracy',
          callbacks=[lgb.early_stopping(100, verbose=False)], # Set verbose=False to keep output clean
          categorical_feature='auto') # 'auto' is standard and robust

# Evaluate model on the validation set
val_preds = model.predict(X_val)
accuracy = accuracy_score(y_val, val_preds)
print(f"\nValidation Accuracy: {accuracy:.5f}")




# --- 5. Create Submission File ---
print("Generating predictions on the test set...")
test_preds_proba = model.predict_proba(X_test)[:, 1]
final_predictions_encoded = (test_preds_proba > 0.5).astype(int)
final_predictions = le_personality.inverse_transform(final_predictions_encoded)

submission_df = pd.DataFrame({'id': X_test_ids, 'Personality': final_predictions})
submission_df.to_csv('submission.csv', index=False)

print("\nSubmission file 'submission.csv' created successfully!")
print("Submission file head:")
print(submission_df.head())



# ============================================
# Improved Accuracy with Stratified K-Fold CV
# (Fixed version without early_stopping_rounds)
# ============================================

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import lightgbm as lgb
import numpy as np

# Assume X and y are already defined in your notebook above
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
    print(f"Training Fold {fold+1}")
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    model = lgb.LGBMClassifier(
        n_estimators=500,
        learning_rate=0.01,
        num_leaves=31,
        max_depth=7,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    acc = accuracy_score(y_val, preds)
    print(f"Fold {fold+1} Accuracy: {acc:.5f}")
    accuracies.append(acc)

print("\n===============================")
print("Average Cross-Validated Accuracy: {:.5f}".format(np.mean(accuracies)))
print("===============================\n")




# ===============================================
# LightGBM Hyperparameter Tuning with Optuna
# Target: Accuracy > 0.977327
# ===============================================
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import lightgbm as lgb

# Split a small fixed portion for validation
X_train_full, X_val_full, y_train_full, y_val_full = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

def objective(trial):
    param = {
        'objective': 'binary',
        'metric': 'binary_error',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-3, 0.05),
        'num_leaves': trial.suggest_int('num_leaves', 20, 100),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.6, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.6, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'verbosity': -1,
        'random_state': 42
    }

    dtrain = lgb.Dataset(X_train_full, label=y_train_full)
    dval = lgb.Dataset(X_val_full, label=y_val_full)

    model = lgb.train(
        param,
        dtrain,
        valid_sets=[dval],
        num_boost_round=1000,
        early_stopping_rounds=50,
        verbose_eval=False
    )

    preds = model.predict(X_val_full)
    preds_binary = (preds > 0.5).astype(int)
    return 1.0 - accuracy_score(y_val_full, preds_binary)  # minimize error

study = optuna.create_study()
study.optimize(objective, n_trials=30)

print("Best params:", study.best_params)

# Train final model on full training set
best_params = study.best_params
best_params.update({
    'objective': 'binary',
    'metric': 'binary_error',
    'verbosity': -1,
    'boosting_type': 'gbdt',
    'random_state': 42
})

final_model = lgb.train(
    best_params,
    lgb.Dataset(X_train_full, label=y_train_full),
    valid_sets=[lgb.Dataset(X_val_full, label=y_val_full)],
    num_boost_round=1000,
    early_stopping_rounds=50,
    verbose_eval=100
)

final_preds = (final_model.predict(X_val_full) > 0.5).astype(int)
final_acc = accuracy_score(y_val_full, final_preds)
print(f"Final Tuned Accuracy: {final_acc:.5f}")


