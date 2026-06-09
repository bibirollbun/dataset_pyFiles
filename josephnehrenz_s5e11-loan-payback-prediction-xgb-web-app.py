# Install and Import Libraries

# Install Libraries
!pip install xgboost colorama > /dev/null 2>&1

import gc
import time
import sys
import platform
import json # Used for pretty printing complex objects
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve
from colorama import Fore, Style
from IPython.core.display import HTML
from datetime import datetime
import warnings

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
gc.collect()

# Center all plots
HTML("""<style>.output_png {    display: table-cell;    text-align: center;    vertical-align: middle;}</style>""")

# Set a consistent default figure size
plt.rcParams['figure.figsize'] = (8, 6)

# Standardized output formatting
def print_step(message):
    """Standardized step printing"""
    print(f"ğŸ“Š {message}")

def print_success(message):
    """Standardized success printing"""
    print(f"âœ… {message}")

def print_warning(message):
    """Standardized warning printing"""
    print(f"âš ï¸�  {message}")

def print_fold_header(fold):
    """Standardized fold header"""
    print(f"\n{Fore.GREEN}ğŸ�¯ {'='*15} FOLD {fold} {'='*15}{Style.RESET_ALL}")

def print_section_header(title):
    """Standardized section header"""
    print(f"\n{Fore.GREEN}{'='*20} {title} {'='*20}{Style.RESET_ALL}")

def print_header(title):
    """Prints a large, centered header banner."""
    print(f"\n{'=' * 20} {title.upper()} {'=' * 20}")

def print_section(title, symbol='-'):
    """Prints a smaller section divider using your standardized color/style."""
    try:
        # Use existing notebook function if available
        print_section_header(title) 
    except NameError:
        # Fallback if print_section_header is not defined in this scope
        print(f"\n{symbol * 5} {title} {symbol * 5}")

def print_list_nicely(data_list, items_per_row=4, prefix="* ", indent=2, sort=True):
    """Prints a list formatted with a fixed number of items per row for cleaner display."""
    if not data_list:
        return
        
    if sort:
        data_list = sorted(data_list)
        
    spacer = ' ' * indent
    num_items = len(data_list)
    
    # Calculate padding needed for aligning columns
    max_len = max(len(str(item)) for item in data_list) if data_list else 0
    col_width = max_len + len(prefix) + 2 # Prefix length + buffer

    rows = []
    for i in range(0, num_items, items_per_row):
        row_items = data_list[i:i + items_per_row]
        
        # Format each item with the prefix and fixed width
        formatted_row = [f"{prefix}{item:<{col_width - len(prefix)}}" for item in row_items]
        rows.append(spacer + "".join(formatted_row).rstrip())
    
    print('\n'.join(rows))

def print_dict_nicely(data_dict, indent=2):
    """Prints a dictionary with keys and values aligned."""
    if not data_dict:
        return
        
    spacer = ' ' * indent
    max_key_len = max(len(str(k)) for k in data_dict.keys()) if data_dict else 0
    
    for key, value in data_dict.items():
        if isinstance(value, dict):
             # Handle nested dicts (like NUMERICAL_STATS)
            print(f"{spacer}{key:<{max_key_len}}: {{", end="")
            nested_items = []
            for nk, nv in value.items():
                if isinstance(nv, (float, int)):
                    nested_items.append(f"'{nk}': {nv:.4f}")
                else:
                    nested_items.append(f"'{nk}': {json.dumps(nv)}")
            print(", ".join(nested_items), "}")
        else:
            # Handle simple key-value pairs
            print(f"{spacer}{key:<{max_key_len}}: {value}")



# Load Data

print_section_header("LOADING DATA")
print_step("Loading training, test, and external data")

# Update: Load external data with robust path handling
train = pd.read_csv("/kaggle/input/playground-series-s5e11/train.csv")
test = pd.read_csv("/kaggle/input/playground-series-s5e11/test.csv")
orig = pd.read_csv("/kaggle/input/loan-prediction-dataset-2025/loan_dataset_20000.csv")

print_success(f"Training data loaded: {train.shape}")
print_success(f"Test data loaded: {test.shape}")
print_success(f"External data loaded: {orig.shape if not orig.empty else 'N/A'}")

# Combine all data for consistent feature creation
TARGET = 'loan_paid_back'
ID_COL = 'id'
test[TARGET] = -1 # Sentinel value for the target in the test set
combine = pd.concat([train, test, orig], axis=0, ignore_index=True)
print_success(f"Combined data for processing: {combine.shape}")


# Data Dimensions / Exploration

print_section_header("DATA EXPLORATION")

# Data Dimensions
print_step("Checking data structure")
print(f"Train shape: {train.shape}")
print(f"Test shape: {test.shape}")

# Data Head (Requested Addition)
print_step("First 5 rows of Training Data (All Variables):")
# This shows the actual values for all columns, including categorical features and IDs
display(train.head())

# Statistical Summary (Training Data)
print_step("Statistical summary (Training Data):")
display(train.describe().style.format("{:.2f}"))


import warnings
warnings.filterwarnings("ignore")

# Target Distribution
plt.figure()
sns.histplot(train[TARGET], kde=True, bins=50)
plt.title('Distribution of Loan Paid Back (1=Paid, 0=Default)')
plt.xlabel('Loan Paid Back')
plt.show()


import warnings
warnings.filterwarnings("ignore")

# Correlation Heatmap
numeric_cols = train.select_dtypes(include=[np.number]).columns
correlation_matrix = train[numeric_cols].corr()
plt.figure()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, fmt='.2f')
plt.title('Core Numerical Feature Correlation Heatmap')
plt.show()


# Feature Engineering

def create_enhanced_features(df):
    """Creates advanced financial ratios, risk scores, and grade features."""
    
    print_step(f"Creating advanced financial and risk features")
    
    # --- 1. Core Affordability & Debt Metrics ---
    # Using small epsilon (1e-6) to prevent division by zero
    df['income_loan_ratio'] = df['annual_income'] / (df['loan_amount'] + 1e-6)
    df['loan_to_income'] = df['loan_amount'] / (df['annual_income'] + 1e-6)
    df['total_debt'] = df['debt_to_income_ratio'] * df['annual_income']
    df['available_income'] = df['annual_income'] * (1 - df['debt_to_income_ratio'])
    
    # --- 2. Payment Analysis ---
    # Approximate monthly payment (Interest rate is in percent)
    df['monthly_payment_approx'] = df['loan_amount'] * (df['interest_rate'] / 100 / 12)
    df['payment_to_income'] = df['monthly_payment_approx'] / (df['annual_income'] / 12 + 1e-6)

    # --- 3. Custom Default Risk Score (High-impact feature) ---
    df['default_risk_score'] = (
        df['debt_to_income_ratio'] * 0.40 + 
        (850 - df['credit_score']) / 850 * 0.35 + 
        df['interest_rate'] / 100 * 0.25
    )
    
    # --- 4. Grade Parsing ---
    df['grade_letter'] = df['grade_subgrade'].str[0]
    df['grade_number'] = df['grade_subgrade'].str[1].astype(int)
    
    print_success(f"Created 9 new continuous features and 2 parsed categorical features.")
    return df

# Apply feature engineering to the combined dataframe
combine = create_enhanced_features(combine)

# --- Define Features ---
BASE_NUMERICAL = ['annual_income', 'debt_to_income_ratio', 'credit_score', 'loan_amount', 'interest_rate']
BASE_CATEGORICAL = ['gender', 'marital_status', 'education_level', 'employment_status', 'loan_purpose', 'grade_subgrade']
NEW_NUMERICAL = ['income_loan_ratio', 'loan_to_income', 'total_debt', 'available_income', 'monthly_payment_approx', 'payment_to_income', 'default_risk_score', 'grade_number']
NEW_CATEGORICAL = ['grade_letter']

NUMERICAL_FEATURES = BASE_NUMERICAL + NEW_NUMERICAL
CATEGORICAL_FEATURES = BASE_CATEGORICAL + NEW_CATEGORICAL
print_success(f"Total Features Defined - Numerical: {len(NUMERICAL_FEATURES)}, Categorical: {len(CATEGORICAL_FEATURES)}")


# --- Target Encoding Implementation ---
def target_mean_encoding(df_features, target_series, df_val, df_orig, col):
    """
    Applies Target Mean Encoding. df_features and target_series are used
    to calculate the mapping, which is then applied to all splits.
    """
    # 1. Combine features and target temporarily for calculation
    df_train = pd.concat([df_features[col], target_series], axis=1)
    
    # 2. Calculate the mean of the target for each category in the TRAIN set
    mapping = df_train.groupby(col)[TARGET].mean()
    
    # 3. Get the global mean of the target for fallback
    global_mean = target_series.mean()
    
    # 4. Apply the encoding to all splits
    train_encoded = df_features[col].map(mapping).fillna(global_mean)
    val_encoded = df_val[col].map(mapping).fillna(global_mean)
    orig_encoded = df_orig[col].map(mapping).fillna(global_mean)
    
    return train_encoded, val_encoded, orig_encoded


# --- Data Splitting, Encoding, and Scaling ---
print_section_header("DATA PROCESSING AND TARGET ENCODING")

# 1. Split the combined data back
train_len = len(train)
test_len = len(test)

train_clean = combine.iloc[:train_len].copy()
test_clean = combine.iloc[train_len:train_len + test_len].copy()
orig_clean = combine.iloc[train_len + test_len:].copy()

# 2. Store IDs and Target
y = train_clean[TARGET]
X = train_clean.drop(TARGET, axis=1)
X_test = test_clean.drop(TARGET, axis=1)

train_ids = X[ID_COL]
test_ids = X_test[ID_COL]

X = X.drop(ID_COL, axis=1)
X_test = X_test.drop(ID_COL, axis=1)

X_orig, y_orig = pd.DataFrame(), pd.Series()
if not orig_clean.empty:
    X_orig = orig_clean.drop([ID_COL, TARGET], axis=1).copy()
    y_orig = orig_clean[TARGET]
    print_success(f"External data prepared for training augmentation: {X_orig.shape}")


# 3. Target Encoding
print_step("Target encoding categorical features for efficiency")

CATS_TO_ENCODE = CATEGORICAL_FEATURES
ENCODED_COLS = [f"TE_{c}" for c in CATS_TO_ENCODE]

for col in CATS_TO_ENCODE:
    X[f'TE_{col}'], X_test[f'TE_{col}'], X_orig[f'TE_{col}'] = target_mean_encoding(
        X.copy(), y.copy(), X_test.copy(), X_orig.copy(), col
    )
# Drop original categorical features
X = X.drop(CATEGORICAL_FEATURES, axis=1, errors='ignore')
X_test = X_test.drop(CATEGORICAL_FEATURES, axis=1, errors='ignore')
if not X_orig.empty:
    X_orig = X_orig.drop(CATEGORICAL_FEATURES, axis=1, errors='ignore')

# 4. Scaling Numerical Features
print_step("Scaling numerical features")
scaler = StandardScaler()
X[NUMERICAL_FEATURES] = scaler.fit_transform(X[NUMERICAL_FEATURES])
X_test[NUMERICAL_FEATURES] = scaler.transform(X_test[NUMERICAL_FEATURES])
if not X_orig.empty:
    X_orig[NUMERICAL_FEATURES] = scaler.transform(X_orig[NUMERICAL_FEATURES])

# Final feature list
FINAL_FEATURES = NUMERICAL_FEATURES + ENCODED_COLS

print_success(f"Processing completed. Final feature count: {len(FINAL_FEATURES)}")
print_warning("The feature count is reduced and highly efficient, enabling fast training.")
gc.collect()

print("\nFinal Feature List (20 Features):")
print_list_nicely(X.columns.tolist(), items_per_row=3, prefix="* ")


# Model Training

print_section_header("COMPETITIVE XGBOOST TRAINING")

# --- HYPERPARAMETERS (Optimized for competition performance) ---
HYPERPARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'learning_rate': 0.009,         # Low rate for better convergence
    'n_estimators': 15000,          # High number with early stopping
    'max_depth': 0,                 # Use max_leaves for faster, better trees
    'max_leaves': 36,               # Key structural optimization
    'subsample': 0.82,               
    'colsample_bytree': 0.72,       
    'random_state': 42,
    'early_stopping_rounds': 500,
    'tree_method': 'hist',          # Best for speed/scale on large datasets
    'reg_alpha': 2.2,               # L1 regularization
    'reg_lambda': 4.5,              # L2 regularization
    'scale_pos_weight': 0.78,       # Address class imbalance
    'enable_categorical': False,    # Target Encoded features are treated as numerical
}

# Use 8 folds for stable validation and competitive results
N_SPLITS = 8
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(X_test))

start_time = time.time()
print_step(f"Training started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print_step(f"Using {N_SPLITS}-fold CV with {len(FINAL_FEATURES)} features")

fold_scores = []
models = []  

# Use the final feature list
X_final = X[FINAL_FEATURES]
X_test_final = X_test[FINAL_FEATURES]
X_orig_final = X_orig[FINAL_FEATURES] if not X_orig.empty else pd.DataFrame()

for fold, (train_idx, val_idx) in enumerate(skf.split(X_final, y), 1):
    print_fold_header(fold)
    
    gc.collect()
    
    # 1. Prepare Validation Data
    X_val, y_val = X_final.iloc[val_idx], y.iloc[val_idx]

    # 2. Prepare Training Data with AUGMENTATION
    X_train, y_train = X_final.iloc[train_idx], y.iloc[train_idx]
    
    if not X_orig_final.empty:
        X_train = pd.concat([X_train, X_orig_final], ignore_index=True)
        y_train = pd.concat([y_train, y_orig], ignore_index=True)

    print_step(f"Data shapes - Train (Augmented): {X_train.shape}, Val: {X_val.shape}")

    model = XGBClassifier(**HYPERPARAMS)
    
    # Train the model with early stopping
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=1000,
              callbacks=[xgb.callback.EarlyStopping(rounds=HYPERPARAMS['early_stopping_rounds'], save_best=True)]
              )

    # Predict on validation and test sets
    val_preds = model.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = val_preds
    
    # Predict on test set using the best iteration
    best_ntree_limit = model.best_iteration + 1
    test_preds += model.predict_proba(X_test_final, iteration_range=(0, best_ntree_limit))[:, 1] / N_SPLITS
    models.append(model)  

    fold_score = roc_auc_score(y_val, val_preds)
    fold_scores.append(fold_score)
    print_success(f"Fold {fold} AUC: {fold_score:.6f} | Best iteration: {best_ntree_limit}")

oof_score = roc_auc_score(y, oof_preds)
total_time = (time.time() - start_time) / 3600
print_success(f"Training completed in {total_time:.2f} hours")
print_success(f"Final OOF ROC AUC: {oof_score:.6f}")


import warnings
warnings.filterwarnings("ignore")

# Model Evaluation

print_section_header("MODEL EVALUATION")

# Evaluation 1: ROC Curve
print_step("Generating ROC curve")
fpr, tpr, thresholds = roc_curve(y, oof_preds)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'OOF ROC AUC = {oof_score:.4f}')
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Evaluation 2: Feature Importance
print_step("Plotting feature importance")
# Use the last model for feature importance
feature_importances = models[-1].feature_importances_
# Ensure the feature names match the ones used in the model
importance_df = pd.DataFrame({'Feature': X_final.columns, 'Importance': feature_importances})
importance_df = importance_df.sort_values('Importance', ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(15))
plt.title('Top 15 Feature Importance (Target Encoded & Engineered Features Dominate)')
plt.show()


# Submission

print_section_header("SUBMISSION")
print_step("Creating submission file")
submission = pd.DataFrame({'loan_paid_back': test_preds}, index=test_ids)
submission.index.name = 'id'

print_step("Saving submission file")
submission.to_csv('submission.csv', header=True)

print_success(f"Submission file created: {submission.shape}")
print_step("First 5 rows of submission:")
display(submission.head())

print_success(f"ğŸ�‰ FINAL SCORE: {oof_score:.6f}")
print_step(f"Training time: {total_time:.2f} hours")


# Save the final model for deployment
print_step("Saving the best XGBoost model...")

# This line saves the model to /kaggle/working/xgboost_model.json
models[-1].save_model("xgboost_model.json")
print_success("Model saved as xgboost_model.json")

# Verification Step
import os
try:
    file_size_mb = os.path.getsize('xgboost_model.json') / (1024*1024)
    print_success(f"Verification: Model file size: {file_size_mb:.2f} MB")
except FileNotFoundError:
    print_error("ERROR: xgboost_model.json was not found in the current directory!")


# --- Feature Type Discovery ---
NUMERICAL_FEATURES = []
BINARY_FEATURES = []
TE_COLS_IN_X = []
ORIGINAL_CATEGORICAL_COLS = [] # Used to create the TE features

for col in X.columns:
    if col.startswith('TE_'):
        TE_COLS_IN_X.append(col)
        # Extract the original column name (e.g., 'TE_loan_purpose' -> 'loan_purpose')
        ORIGINAL_CATEGORICAL_COLS.append(col.replace('TE_', ''))
    # Use a try/except block for .nunique() as some mock columns might be float
    elif X[col].nunique() <= 2 and X[col].dropna().isin([0, 1]).all():
        BINARY_FEATURES.append(col)
    else:
        NUMERICAL_FEATURES.append(col)

# --- Extract and Print ALL Required Constants ---

print_header("CONSTANTS FOR STREAMLIT APP EXPORT")

# --- Feature Lists ---
print_section("Feature Lists for Streamlit Input Structure")
print("\n# NUMERICAL_FEATURES")
print_list_nicely(NUMERICAL_FEATURES, items_per_row=3)
print("\n# BINARY_FEATURES")
print_list_nicely(BINARY_FEATURES, items_per_row=3)
print("\n# TE_COLS_IN_X (Encoded Features)")
print_list_nicely(TE_COLS_IN_X, items_per_row=3)
print("\n# ORIGINAL_CATEGORICAL_COLS (Original Feature Names)")
print_list_nicely(ORIGINAL_CATEGORICAL_COLS, items_per_row=3)

# --- Global Mean Target & Final Feature List ---
TARGET = 'loan_paid_back'
GLOBAL_MEAN_TARGET = train[TARGET].mean() # Use train dataframe to calculate mean

print_section("Global Mean Target & Final Feature List")
print(f"ğŸ�¯ GLOBAL_MEAN_TARGET = {GLOBAL_MEAN_TARGET:.6f}")
print(f"ğŸ“Š FINAL_FEATURE_COUNT = {len(X.columns)}")

print("\n# FINAL_FEATURES (Complete list)")
print_list_nicely(X.columns.tolist(), items_per_row=3)

# --- Numerical Feature Statistics ---
print_section("Numerical Feature Statistics (for Synthetic Data)")
NUMERICAL_STATS = {}
for col in NUMERICAL_FEATURES:
    NUMERICAL_STATS[col] = {
        'mean': X[col].mean(), 
        'std': X[col].std(), 
        'min': X[col].min(), 
        'max': X[col].max()
    }
    
print("NUMERICAL_STATS = {")
print_dict_nicely(NUMERICAL_STATS)
print("}")

# --- Target Encoding Mappings ---
print_section("D. Target Encoding Mappings (TE_MAPPINGS)")
TE_MAPPINGS = {}
for col in ORIGINAL_CATEGORICAL_COLS:
    try:
        # Use the original training data (train) and target (TARGET) to calculate the means
        mapping = train.groupby(col)[TARGET].mean().apply(lambda x: round(x, 4)).to_dict()
        
        # Ensure keys are strings for JSON/Python compatibility
        TE_MAPPINGS[col] = {str(k): v for k, v in mapping.items()}
    
    except KeyError:
        # In a real run, this would warn if the column was missing from the training data
        print_warning(f"Original categorical column '{col}' not found in the 'train' DataFrame. Skipping mapping.")

print("TE_MAPPINGS = {")
print_dict_nicely(TE_MAPPINGS)
print("}")


# Environment summary
print_header("ENVIRONMENT SUMMARY")

env_summary = {
    "python": sys.version,
    "os": platform.platform(),
    "numpy": np.__version__,
    "pandas": pd.__version__
}

# Clean up the python version string for cleaner output
env_summary['python'] = env_summary['python'].split('\n')[0]

print("\n# ENVIRONMENT VERSIONS")
print_dict_nicely(env_summary, indent=2)

